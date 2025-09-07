"""
LabelConversion_VehiclesOnly.py

Erweitertes Konvertierungsskript für MIO-TCD → YOLO (nur Fahrzeuge),
optimiert für hierarchical YOLOv11-Training.

Hauptfunktionen:
- Entfernt *ganze Bilder*, wenn irgendeine Klasse aus CLASSES_TO_IGNORE enthalten ist
- Erzeugt reproduzierbaren Train/Val-Split.
- Schreibt YOLO-Labels, kopiert oder verlinkt Bilder (Copy/Symlink) mit Normalisierung/Clamping.
- Berechnet Klassen-Gewichte (inverse Häufigkeit, Effective Number nach Cui et al.) für Leaf/L2/L1 und speichert als JSON + .pt.
- Exportiert Datensatz-Statistiken (per-Klasse, optional BBox-Area, per-image CSV).
- Erzeugt eine korrekte data.yaml für Ultralytics (names als Liste)
- Optional: erstellt Diagnoseplots (Balken, Histogramm).
- Speichert Reproduzierbarkeits-Metadaten (Manifest)
- Optional negative Beispiele (Hintergrund-only) für RoI-/Fallback-Training (leere Labeldateien)
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import List, Dict, Tuple, Any
from tqdm import tqdm

import numpy as np
import pandas as pd

# Optional dependencies
try:
    from PIL import Image
except Exception:
    Image = None

try:
    import torch
except Exception:
    torch = None

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


# Zielklassen (L3, leaf, exakte Fahrzeuge)
STUDENT_CLASSES: List[str] = [
    "bus",
    "work_van",
    "single_unit_truck",
    "pickup_truck",
    "articulated_truck",
    "car",
    "motorcycle",
    "bicycle",
]

# Bilder werden komplett verworfen, wenn diese Klassen im Bild vorkommen:
CLASSES_TO_IGNORE: List[str] = [
    "motorized_vehicle",
    "non-motorized_vehicle",
    "pedestrian",
]

# Hierarchie
# L2-Namen (Gruppen)
L2_NAMES: List[str] = ["heavy_vehicle", "car_group", "two_wheeled_vehicle"]

# L3 (Leaf) → L2 (Gruppen)
CLASS_TO_L2: Dict[str, str] = {
    "bus": "heavy_vehicle",
    "single_unit_truck": "heavy_vehicle",
    "articulated_truck": "heavy_vehicle",
    "car": "car_group",
    "pickup_truck": "car_group",
    "motorcycle": "two_wheeled_vehicle",
    "bicycle": "two_wheeled_vehicle",
    "work_van": "car_group",
}

# L1-Namen
L1_NAMES: List[str] = ["vehicle"]

# L3 (Leaf) → L1 (Super)
# Für mehrere Superklassen muss hier zugeordnet werden
CLASS_TO_L1: Dict[str, str] = {cls: "vehicle" for cls in STUDENT_CLASSES}

# Default-Beta für Class-Balanced Loss (Effective Number)
DEFAULT_CB_BETA: float = 0.99999

# Mögliche Bildnamens-Formate
FILENAME_TRIES: List[str] = [
    "{id}.jpg",
    "{id}.png",
    "{id:08d}.jpg",
    "{id:08d}.png",
    "{id}.jpeg",
]


def safe_mkdir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def try_find_image(images_dir: Path, image_id_raw) -> Path | None:
    """
    Versucht verschiedene Namensmuster, um das Bild zu finden. Gibt Path oder None zurück.
    image_id_raw kann num oder string sein.
    """
    try:
        image_id_int = int(float(image_id_raw))
    except Exception:
        image_id_int = None

    for fmt in FILENAME_TRIES:
        try:
            if "{id:08d}" in fmt:
                if image_id_int is None:
                    continue
                candidate = images_dir / fmt.format(id=image_id_int)
            elif "{id}" in fmt:
                candidate = images_dir / fmt.format(id=image_id_raw)
            else:
                candidate = images_dir / fmt.format(image_id_raw)
        except Exception:
            continue
        if candidate.exists():
            return candidate
    
    if not images_dir.exists():
        return None
    try:
        if image_id_int is not None:
            substrs = [str(image_id_int), f"{image_id_int:08d}"]
        else:
            substrs = [str(image_id_raw)]
        for f in images_dir.iterdir():
            name = f.name.lower()
            if any(s in name for s in substrs) and name.endswith((".jpg", ".jpeg", ".png")):
                return f
    except Exception:
        pass

    return None


def miod_to_yolo(xmin, ymin, xmax, ymax, img_width, img_height) -> Tuple[float, float, float, float]:
    """Konvertiert absolute BBox → YOLO (x_center, y_center, w, h) normalisiert [0,1]."""
    dw = 1.0 / max(1.0, float(img_width))
    dh = 1.0 / max(1.0, float(img_height))
    x_center = (xmin + xmax) / 2.0
    y_center = (ymin + ymax) / 2.0
    width = xmax - xmin
    height = ymax - ymin
    return x_center * dw, y_center * dh, width * dw, height * dh


def effective_num_weights(counts, beta=DEFAULT_CB_BETA, eps=1e-12):
    """
    Class-Balanced Gewichte:
        w_i = (1 - beta) / (1 - beta^n_i), normiert auf Mittelwert 1.
    """
    counts = np.array(counts, dtype=np.float64)
    eff = 1.0 - np.power(beta, counts)
    eff[eff < eps] = eps
    w = (1.0 - beta) / eff
    return w / np.mean(w)


def inverse_freq_weights(counts, eps=1e-12):
    counts = np.array(counts, dtype=np.float64)
    inv = 1.0 / (counts + eps)
    return inv / np.mean(inv)


def write_data_yaml(out_dir: Path, train_rel: str, val_rel: str, names: list, yaml_path: Path):
    lines = []
    lines.append("# MIO-TCD vehicle-only dataset (generated)")
    lines.append(f"# root: {out_dir.resolve()}")
    lines.append("")
    lines.append(f"train: {train_rel}")
    lines.append(f"val: {val_rel}")
    lines.append("")
    # 'names' as list, cleaner in Ultralytics
    names_list = ", ".join(names)
    lines.append(f"nc: {len(names)}")
    lines.append(f"names: [{names_list}]")
    text = "\n".join(lines)
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(text)
    return text


def save_json(data: Any, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def get_git_revision_short_hash() -> str | None:
    try:
        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL)
        return out.strip().decode("utf-8")
    except Exception:
        return None


def compute_bbox_stats(bboxes: List[Tuple[float, float, float, float]], img_w: int, img_h: int):
    """
    bboxes: Liste (xmin,ymin,xmax,ymax)
    Rückgabe: list(normalized areas), list(aspect ratios)
    """
    areas, aspects = [], []
    denom = max(1.0, float(img_w) * float(img_h))
    for (xmin, ymin, xmax, ymax) in bboxes:
        w = max(0.0, xmax - xmin)
        h = max(0.0, ymax - ymin)
        areas.append((w * h) / denom)
        aspects.append((w / h) if h > 0 else 0.0)
    return areas, aspects

# Greedy Multi-Label Stratified Split (Heuristik)
def greedy_multilabel_train_val_split(images_df: pd.DataFrame,
                                      val_ratio: float = 0.2,
                                      min_val_per_class: int = 1,
                                      seed: int = 42):
    """
    images_df Spalten: ['image_id', 'classes', 'num_objects']
    Rückgabe: train_ids, val_ids, stats_dict
    """
    rng = np.random.RandomState(seed)
    class_list = sorted({c for s in images_df["classes"] for c in s})
    class_to_idx = {c: i for i, c in enumerate(class_list)}
    n_classes = len(class_list)

    image_ids = list(images_df["image_id"])
    presence = np.zeros((len(image_ids), n_classes), dtype=int)
    for i, s in enumerate(images_df["classes"]):
        for c in s:
            if c in class_to_idx:
                presence[i, class_to_idx[c]] = 1

    N = len(image_ids)
    desired_n_val = int(round(val_ratio * N))
    per_class_presence = presence.sum(axis=0)
    desired_val_per_class = np.maximum(1, np.floor(per_class_presence * val_ratio).astype(int))
    desired_val_per_class = np.maximum(desired_val_per_class, min_val_per_class)

    class_freq = per_class_presence.astype(float)
    inv_freq = 1.0 / (class_freq + 1e-12)
    rarity_scores = (presence * inv_freq.reshape(1, -1)).sum(axis=1)
    order = np.argsort(-rarity_scores)

    assigned = np.zeros(len(image_ids), dtype=int)  # 0=train, 1=val
    val_counts = np.zeros(n_classes, dtype=int)
    current_val_size = 0

    for idx in order:
        img_presence = presence[idx]
        if current_val_size >= desired_n_val:
            assigned[idx] = 0
            continue

        val_counts_if_val = val_counts + img_presence
        diff_if_val = np.abs(val_counts_if_val - desired_val_per_class).sum()
        diff_if_train = np.abs(val_counts - desired_val_per_class).sum()

        prefer_val = any(desired_val_per_class[cidx] > 0 and val_counts[cidx] == 0 and img_presence[cidx] == 1
                         for cidx in range(n_classes))

        if prefer_val or (diff_if_val < diff_if_train):
            assigned[idx] = 1
            val_counts = val_counts_if_val
            current_val_size += 1
        else:
            assigned[idx] = 0

    # Auf Zielgröße bringen (falls daneben)
    idxs_train = [i for i in range(len(image_ids)) if assigned[i] == 0]
    rng.shuffle(idxs_train)
    i = 0
    while current_val_size < desired_n_val and i < len(idxs_train):
        idx = idxs_train[i]
        assigned[idx] = 1
        val_counts += presence[idx]
        current_val_size += 1
        i += 1

    while current_val_size > desired_n_val:
        val_idxs = np.where(assigned == 1)[0]
        if len(val_idxs) == 0:
            break
        idx = val_idxs[0]
        assigned[idx] = 0
        val_counts -= presence[idx]
        current_val_size -= 1

    train_ids = [image_ids[i] for i in range(len(image_ids)) if assigned[i] == 0]
    val_ids = [image_ids[i] for i in range(len(image_ids)) if assigned[i] == 1]

    stats = {
        "n_images_total": N,
        "n_train": len(train_ids),
        "n_val": len(val_ids),
        "desired_n_val": desired_n_val,
        "desired_val_per_class": desired_val_per_class.tolist(),
        "final_val_per_class": val_counts.tolist(),
        "class_list": class_list,
    }
    return train_ids, val_ids, stats

# Negativbeispiel-Erzeugung (Hintergrund-only, YOLO-kompatibel)

def generate_negatives(out_img_dir: Path,
                       out_label_dir: Path,
                       copy_images: bool,
                       num_negatives: int = 2):
    """
    Erzeugt pro existierendem *positivem* Bild num_negatives Hintergrund-Bilder:
    - Dupliziert Bilddatei (per Symlink, wenn copy_images=False, sonst Copy) unter neuem Namen *_neg{i}.ext
    - Schreibt *leere* Labeldatei, was für YOLO bedeutet: „keine Objekte“.
    """
    if num_negatives <= 0:
        return 0

    created = 0
    all_imgs = sorted([p for p in out_img_dir.glob("*") if p.suffix.lower() in [".jpg", ".jpeg", ".png"]])
    warned_symlink_once = False

    for img_path in tqdm(all_imgs, desc=f"Negatives x{num_negatives}"):
        if "_neg" in img_path.stem:
            continue

        for i in range(num_negatives):
            neg_stem = f"{img_path.stem}_neg{i}"
            neg_img_path = out_img_dir / f"{neg_stem}{img_path.suffix}"
            neg_lbl_path = out_label_dir / f"{neg_stem}.txt"

            try:
                if neg_img_path.exists():
                    neg_img_path.unlink()
                if copy_images:
                    shutil.copy2(img_path, neg_img_path)
                else:
                    try:
                        os.symlink(img_path, neg_img_path)
                    except Exception as e:
                        if not warned_symlink_once:
                            print(f"Symlink fehlgeschlagen, wechsle auf Copy (erstes Auftreten): {e}")
                            warned_symlink_once = True
                        shutil.copy2(img_path, neg_img_path)
            except Exception as e2:
                print(f"Negativbild konnte nicht erstellt werden: {img_path} -> {neg_img_path} ({e2})")
                continue

            try:
                with open(neg_lbl_path, "w", encoding="utf-8") as f:
                    f.write("")  # leer == keine Objekte in YOLO
            except Exception as e:
                print(f"Negativlabel konnte nicht erstellt werden: {neg_lbl_path} ({e})")
                # Bild wieder entfernen, um Inkonsistenzen zu vermeiden
                try:
                    neg_img_path.unlink(missing_ok=True)
                except Exception:
                    pass
                continue

            created += 1

    print(f"Negative erstellt: {created}")
    return created

# Hauptpipeline

def main(args):
    GT_CSV_PATH = Path(args.gt_csv).resolve()
    MIO_IMAGES_DIR = Path(args.images_dir).resolve()
    YOLO_DATA_DIR = Path(args.out_dir).resolve()
    YOLO_IMAGES_TRAIN_DIR = YOLO_DATA_DIR / "images" / "train"
    YOLO_IMAGES_VAL_DIR = YOLO_DATA_DIR / "images" / "val"
    YOLO_LABELS_TRAIN_DIR = YOLO_DATA_DIR / "labels" / "train"
    YOLO_LABELS_VAL_DIR = YOLO_DATA_DIR / "labels" / "val"
    SPLITS_DIR = YOLO_DATA_DIR / "splits"
    ARTIFACTS_DIR = YOLO_DATA_DIR / "artifacts"

    for d in [YOLO_IMAGES_TRAIN_DIR, YOLO_IMAGES_VAL_DIR, YOLO_LABELS_TRAIN_DIR, YOLO_LABELS_VAL_DIR, SPLITS_DIR, ARTIFACTS_DIR]:
        safe_mkdir(d)

    if not GT_CSV_PATH.exists():
        print(f"ERROR: GT CSV not found at {GT_CSV_PATH}")
        return 1
    if not MIO_IMAGES_DIR.exists():
        print(f"ERROR: MIO images directory not found at {MIO_IMAGES_DIR}")
        return 1
    if Image is None:
        print("ERROR: Pillow (PIL) not available. Please pip install 'Pillow'.")
        return 1

    # CSV lesen

    print("Lese Annotations-CSV...")
    try:
        df = pd.read_csv(GT_CSV_PATH, header=None,
                         names=["image_id", "class_name", "xmin", "ymin", "xmax", "ymax"])
    except Exception as e:
        print(f"ERROR reading CSV: {e}")
        return 1

    print(f"Originale Annotationen: {len(df)} Zeilen, einzigartige Bilder: {df['image_id'].nunique()}")

    # Gruppieren & Filterung (angepasst für Fußgänger (mehr insight))
    grouped = df.groupby("image_id")
    images_to_process: List[Dict[str, Any]] = []
    skipped_images_due_to_ignored = 0
    skipped_missing_images = 0
    degenerate_boxes_clamped = 0
    images_dropped_empty_after_clamp = 0
    pedestrian_boxes_dropped = 0
    pedestrian_only_images_dropped = 0

    print("Scanne Bilder & filtere...")
    for image_id_raw, group in tqdm(grouped, desc="Scanne Bilder"):
        class_names_in_image = set(group["class_name"].tolist())

        # Fall 1: enthält motorized_vehicle oder non-motorized_vehicle → Bild komplett verwerfen
        if any(c in {"motorized_vehicle", "non-motorized_vehicle"} for c in class_names_in_image):
            skipped_images_due_to_ignored += 1
            continue

        # Fall 2: enthält nur Fußgänger → Bild komplett verwerfen
        if class_names_in_image == {"pedestrian"}:
            pedestrian_only_images_dropped += 1
            continue

        # Sonst: Fußgängerlabels verwerfen, Fahrzeuge behalten
        group_keep = group[group["class_name"].isin(STUDENT_CLASSES)].copy()
        dropped_ped_count = (group["class_name"] == "pedestrian").sum()
        pedestrian_boxes_dropped += int(dropped_ped_count)

        if group_keep.empty:
            continue

        img_path = try_find_image(MIO_IMAGES_DIR, image_id_raw)
        if img_path is None:
            skipped_missing_images += 1
            continue

        try:
            with Image.open(img_path) as im:
                img_w, img_h = im.size
        except Exception:
            skipped_missing_images += 1
            continue

        bboxes = []
        for _, row in group_keep.iterrows():
            xmin = float(row["xmin"])
            ymin = float(row["ymin"])
            xmax = float(row["xmax"])
            ymax = float(row["ymax"])

            # clamp to image bounds
            x0 = max(0.0, min(xmin, img_w - 1))
            y0 = max(0.0, min(ymin, img_h - 1))
            x1 = max(0.0, min(xmax, img_w - 1))
            y1 = max(0.0, min(ymax, img_h - 1))

            if x1 <= x0 or y1 <= y0:
                degenerate_boxes_clamped += 1
                continue

            bboxes.append((x0, y0, x1, y1, row["class_name"]))

        if len(bboxes) == 0:
            images_dropped_empty_after_clamp += 1
            continue

        images_to_process.append({
            "image_id": image_id_raw,
            "img_path": img_path,
            "img_w": img_w,
            "img_h": img_h,
            "bboxes": bboxes,
            "classes": sorted(set([b[-1] for b in bboxes])),
            "num_objects": len(bboxes),
        })

    # Sanity-Prints
    print(f"Verworfen wegen ignorierter motorized/non-motorized_vehicle: {skipped_images_due_to_ignored}")
    print(f"Verworfen wegen nur Fußgänger: {pedestrian_only_images_dropped}")
    print(f"Fußgängerlabels innerhalb von Fahrzeugbildern verworfen: {pedestrian_boxes_dropped}")
    print(f"Verworfen wegen fehlender/kaputter Bilder: {skipped_missing_images}")
    print(f"Degenerierte BBoxen verworfen (nach Clamping): {degenerate_boxes_clamped}")
    print(f"Bilder nach Clamping ohne gültige BBox → verworfen: {images_dropped_empty_after_clamp}")
    print(f"Verbleibende Bilder: {len(images_to_process)}")

    if len(images_to_process) == 0:
        print("Keine verbleibenden Bilder nach Filterung. Check logic.")
        return 1

    images_df = pd.DataFrame(images_to_process)


    # Per-Image-Statistiken (für optionalen Plot)
    stats_rows = []
    for _, row in images_df.iterrows():
        areas, aspects = compute_bbox_stats([(b[0], b[1], b[2], b[3]) for b in row["bboxes"]],
                                            row["img_w"], row["img_h"])
        stats_rows.append({
            "image_id": row["image_id"],
            "img_path": str(row["img_path"]),
            "num_objects": row["num_objects"],
            "classes": ",".join(row["classes"]),
            "mean_area": float(np.mean(areas)) if areas else 0.0,
        })
    stats_df = pd.DataFrame(stats_rows)
    stats_df.to_csv(ARTIFACTS_DIR / "per_image_stats.csv", index=False)

    # Train/Val-Split
    
    print("Berechne stratifiziertes Train/Val-Split...")
    train_ids, val_ids, split_stats = greedy_multilabel_train_val_split(
        images_df[["image_id", "classes", "num_objects"]],
        val_ratio=args.val_ratio,
        min_val_per_class=args.min_val_per_class,
        seed=args.seed
    )
    print(f"Split: train={len(train_ids)}, val={len(val_ids)} (Ziel val ~ {split_stats['desired_n_val']})")

    # Split-Listen speichern (nur IDs)
    (SPLITS_DIR / "train.txt").write_text("\n".join(map(str, train_ids)), encoding="utf-8")
    (SPLITS_DIR / "val.txt").write_text("\n".join(map(str, val_ids)), encoding="utf-8")

    # Vorab Val-Präsenz prüfen (auf Basis images_df)
    val_set = set(val_ids)
    val_label_counts = Counter()
    for _, row in images_df[images_df["image_id"].isin(val_set)].iterrows():
        for (_, _, _, _, cls_name) in row["bboxes"]:
            val_label_counts[cls_name] += 1
    missing_in_val = [c for c in STUDENT_CLASSES if val_label_counts[c] == 0]
    if missing_in_val:
        print(f"Warnung: folgende Leaf-Klassen haben 0 Instanzen in val: {missing_in_val}")

    # Labels schreiben & Bilder kopieren/verlinken

    def process_split(split_ids: List[Any], split_name: str):
        img_dir = YOLO_IMAGES_TRAIN_DIR if split_name == "train" else YOLO_IMAGES_VAL_DIR
        lbl_dir = YOLO_LABELS_TRAIN_DIR if split_name == "train" else YOLO_LABELS_VAL_DIR
        safe_mkdir(img_dir)
        safe_mkdir(lbl_dir)

        warned_symlink_once = False
        n_images_written = 0
        n_images_skipped_empty = 0

        for image_id in tqdm(split_ids, desc=f"Erzeuge {split_name}"):
            sel = images_df[images_df["image_id"] == image_id]
            if sel.empty:
                continue
            rec = sel.iloc[0]
            src_img = Path(rec["img_path"])
            dst_img = img_dir / src_img.name

            lines = []
            dropped_after_norm = 0
            for (xmin, ymin, xmax, ymax, cls_name) in rec["bboxes"]:
                cls_id = STUDENT_CLASSES.index(cls_name)
                x_c, y_c, w, h = miod_to_yolo(xmin, ymin, xmax, ymax, rec["img_w"], rec["img_h"])

                # Clamp normalized coords to [0, 1]
                x_c = max(0.0, min(1.0, x_c))
                y_c = max(0.0, min(1.0, y_c))
                w   = max(0.0, min(1.0, w))
                h   = max(0.0, min(1.0, h))
                if w <= 0.0 or h <= 0.0:
                    dropped_after_norm += 1
                    continue

                lines.append(f"{cls_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}")

            if not lines:
                n_images_skipped_empty += 1
                continue

            (lbl_dir / f"{dst_img.stem}.txt").write_text("\n".join(lines), encoding="utf-8")

            try:
                if args.copy_images:
                    if not dst_img.exists():
                        shutil.copy2(src_img, dst_img)
                else:
                    try:
                        if dst_img.exists():
                            dst_img.unlink()
                        os.symlink(src_img, dst_img)
                    except Exception as e:
                        if not warned_symlink_once:
                            print(f"Symlink fehlgeschlagen in {split_name}, wechsle auf Copy (erstes Auftreten): {e}")
                            warned_symlink_once = True
                        shutil.copy2(src_img, dst_img)
            except Exception as e2:
                print(f"Bildübertragung fehlgeschlagen {src_img} -> {dst_img}: {e2}")
                try:
                    shutil.copy2(src_img, dst_img)
                except Exception as e3:
                    print(f"Fallback Copy fehlgeschlagen: {e3}")

            n_images_written += 1

        print(f"{split_name}: Bilder geschrieben={n_images_written}, Bilder ohne gültige Labels übersprungen={n_images_skipped_empty}")

    process_split(train_ids, "train")
    process_split(val_ids, "val")

    # Klassenhäufigkeiten & Gewichte (Train)

    print("Berechne Klassenhäufigkeiten & Gewichte (Train)...")
    train_label_files = list(YOLO_LABELS_TRAIN_DIR.glob("*.txt"))
    leaf_counts = np.zeros(len(STUDENT_CLASSES), dtype=int)
    for lf in train_label_files:
        txt = lf.read_text(encoding="utf-8")
        for ln in txt.splitlines():
            ln = ln.strip()
            if not ln:
                continue
            try:
                cls_id = int(ln.split()[0])
                if 0 <= cls_id < len(STUDENT_CLASSES):
                    leaf_counts[cls_id] += 1
            except Exception:
                continue

    counts_dict_leaf = {STUDENT_CLASSES[i]: int(leaf_counts[i]) for i in range(len(STUDENT_CLASSES))}

    # Finalisierte Namenslisten (stellen sicher, dass Mapping-Ziele enthalten sind)
    l2_set = set(L2_NAMES).union(CLASS_TO_L2.values())
    l2_names_final = list(L2_NAMES) + [n for n in sorted(l2_set) if n not in L2_NAMES]

    l1_set = set(L1_NAMES).union(CLASS_TO_L1.values())
    l1_names_final = list(L1_NAMES) + [n for n in sorted(l1_set) if n not in L1_NAMES]

    l2_name_to_idx = {n: i for i, n in enumerate(l2_names_final)}
    l1_name_to_idx = {n: i for i, n in enumerate(l1_names_final)}

    leaf_to_l2_idx = [l2_name_to_idx[CLASS_TO_L2[cls]] for cls in STUDENT_CLASSES]
    leaf_to_l1_idx = [l1_name_to_idx[CLASS_TO_L1[cls]] for cls in STUDENT_CLASSES]

    l2_counts = np.zeros(len(l2_names_final), dtype=int)
    l1_counts = np.zeros(len(l1_names_final), dtype=int)
    for leaf_i, c in enumerate(leaf_counts):
        l2_counts[leaf_to_l2_idx[leaf_i]] += c
        l1_counts[leaf_to_l1_idx[leaf_i]] += c

    def _inv(counts): return (1.0 / (np.array(counts, dtype=np.float64) + 1e-6)) / np.mean(1.0 / (np.array(counts, dtype=np.float64) + 1e-6))
    def _cb(counts, beta=args.cb_beta):
        counts = np.array(counts, dtype=np.float64)
        eff = 1.0 - np.power(beta, counts)
        eff[eff < 1e-12] = 1e-12
        w = (1.0 - beta) / eff
        return w / np.mean(w)

    inv_leaf, inv_l2, inv_l1 = _inv(leaf_counts), _inv(l2_counts), _inv(l1_counts)
    cb_leaf, cb_l2, cb_l1 = _cb(leaf_counts), _cb(l2_counts), _cb(l1_counts)

    # Sanity prints
    print(f"Weights shapes — leaf={inv_leaf.shape[0]}, l2={inv_l2.shape[0]}, l1={inv_l1.shape[0]}")
    if np.any(np.isnan(inv_leaf)) or np.any(np.isnan(inv_l2)) or np.any(np.isnan(inv_l1)):
        print("NaN in inverse weights erkannt.")
    if np.any(np.isnan(cb_leaf)) or np.any(np.isnan(cb_l2)) or np.any(np.isnan(cb_l1)):
        print("NaN in effective-number weights erkannt.")

    # JSON + PT Speichern
    weights_json = {
        "cb_beta": args.cb_beta,
        "levels": {
            "leaf": {"names": STUDENT_CLASSES, "counts": leaf_counts.tolist(),
                     "inverse_freq_weights": inv_leaf.tolist(), "effective_num_weights": cb_leaf.tolist()},
            "l2": {"names": l2_names_final, "counts": l2_counts.tolist(),
                   "inverse_freq_weights": inv_l2.tolist(), "effective_num_weights": cb_l2.tolist()},
            "l1": {"names": l1_names_final, "counts": l1_counts.tolist(),
                   "inverse_freq_weights": inv_l1.tolist(), "effective_num_weights": cb_l1.tolist()},
        }
    }
    save_json(weights_json, ARTIFACTS_DIR / "class_weights.json")

    if torch is not None:
        torch.save({"leaf": torch.tensor(inv_leaf, dtype=torch.float32),
                    "l2": torch.tensor(inv_l2, dtype=torch.float32),
                    "l1": torch.tensor(inv_l1, dtype=torch.float32)},
                   ARTIFACTS_DIR / "class_inv_freq_weights.pt")
        torch.save({"leaf": torch.tensor(cb_leaf, dtype=torch.float32),
                    "l2": torch.tensor(cb_l2, dtype=torch.float32),
                    "l1": torch.tensor(cb_l1, dtype=torch.float32)},
                   ARTIFACTS_DIR / "class_cb_weights.pt")

    # CSV für Sichtprüfung
    per_class_df = pd.DataFrame({"class": STUDENT_CLASSES, "train_count": [int(x) for x in leaf_counts]})
    per_class_df.to_csv(ARTIFACTS_DIR / "per_class_counts_train.csv", index=False)

    # data.yaml

    print("Schreibe data.yaml...")
    train_rel = "images/train"
    val_rel = "images/val"
    yaml_text = write_data_yaml(YOLO_DATA_DIR, train_rel, val_rel, STUDENT_CLASSES, YOLO_DATA_DIR / "data.yaml")
    (ARTIFACTS_DIR / "data_yaml.txt").write_text(yaml_text, encoding="utf-8")

    # Hierarchie exportieren (als Listen, inkl. Index-Maps)

    hierarchy = {
        "leaf_names": STUDENT_CLASSES,
        "l2_names": l2_names_final,
        "l1_names": l1_names_final,
        "leaf_to_l2": leaf_to_l2_idx,  # list of indices
        "leaf_to_l1": leaf_to_l1_idx,  # list of indices
        "leaf_index": {name: i for i, name in enumerate(STUDENT_CLASSES)},
        "l2_index": {name: i for i, name in enumerate(l2_names_final)},
        "l1_index": {name: i for i, name in enumerate(l1_names_final)},
    }
    save_json(hierarchy, ARTIFACTS_DIR / "hierarchy.json")

    # Negative Beispiele optional erzeugen (Train + Val)

    if args.make_negatives:
        print("Erzeuge Negativbeispiele (Hintergrund-only, leere Labeldateien)...")
        created_train = generate_negatives(YOLO_IMAGES_TRAIN_DIR, YOLO_LABELS_TRAIN_DIR, args.copy_images, args.num_negatives)
        created_val = generate_negatives(YOLO_IMAGES_VAL_DIR, YOLO_LABELS_VAL_DIR, args.copy_images, args.num_negatives)
        print(f"Negative erstellt: train={created_train}, val={created_val}")

    # Optional: Plots

    if args.make_plots:
        if plt is None:
            print("matplotlib nicht verfügbar; überspringe Plots.")
        else:
            try:
                # Klassen-Balkendiagramm (Train)
                plt.figure(figsize=(10, 4))
                xs = np.arange(len(STUDENT_CLASSES))
                ys = [leaf_counts[i] for i in range(len(STUDENT_CLASSES))]
                plt.bar(xs, ys)
                plt.xticks(xs, STUDENT_CLASSES, rotation=45, ha="right")
                plt.title("Train-Instanzanzahl pro Leaf-Klasse")
                plt.tight_layout()
                (ARTIFACTS_DIR / "plots").mkdir(parents=True, exist_ok=True)
                plt.savefig(ARTIFACTS_DIR / "plots" / "train_leaf_counts.png", dpi=150)
                plt.close()

                # Per-Image mittlere BBox-Fläche (Histogramm)
                plt.figure(figsize=(7, 4))
                all_areas = stats_df["mean_area"].values
                plt.hist(all_areas, bins=50)
                plt.title("Per-Image mittlere BBox-Fläche (normalisiert)")
                plt.tight_layout()
                plt.savefig(ARTIFACTS_DIR / "plots" / "bbox_area_hist.png", dpi=150)
                plt.close()
            except Exception as e:
                print(f"Plot-Erzeugung fehlgeschlagen: {e}")

    # Manifest / Zusammenfassung

    manifest = {
        "script": "LabelConversion_VehicleOnly_Enhanced.py",
        "student_classes": STUDENT_CLASSES,
        "classes_to_ignore": CLASSES_TO_IGNORE,
        "gt_csv": str(GT_CSV_PATH),
        "images_dir": str(MIO_IMAGES_DIR),
        "out_dir": str(YOLO_DATA_DIR),
        "val_ratio": args.val_ratio,
        "seed": args.seed,
        "copy_images": args.copy_images,
        "cb_beta": args.cb_beta,
        "make_negatives": args.make_negatives,
        "num_negatives": args.num_negatives,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "n_images_total_after_filter": int(len(images_df)),
        "n_train_images": int(len(train_ids)),
        "n_val_images": int(len(val_ids)),
        "skipped_images_due_to_ignored": int(skipped_images_due_to_ignored),
        "skipped_images_missing": int(skipped_missing_images),
        "degenerate_boxes_dropped_after_clamp": int(degenerate_boxes_clamped),
        "images_dropped_empty_after_clamp": int(images_dropped_empty_after_clamp),
        "weights_shapes": {
            "leaf": len(inv_leaf),
            "l2": len(inv_l2),
            "l1": len(inv_l1),
        }
    }
    git_hash = get_git_revision_short_hash()
    if git_hash is not None:
        manifest["git_hash"] = git_hash
    save_json(manifest, ARTIFACTS_DIR / "manifest.json")

    # Zusammenfassung
    readme_lines = []
    readme_lines.append("MIO-TCD Vehicle-Only dataset (generated)")
    readme_lines.append(f"Root: {YOLO_DATA_DIR}")
    readme_lines.append("")
    readme_lines.append("Summary:")
    readme_lines.append(f"  train images: {len(train_ids)}")
    readme_lines.append(f"  val images: {len(val_ids)}")
    readme_lines.append("  train labels (instances) by leaf class:")
    for cls in STUDENT_CLASSES:
        readme_lines.append(f"    {cls}: {counts_dict_leaf.get(cls, 0)}")
    readme_lines.append("")
    readme_lines.append("Artifacts saved to 'artifacts/'")
    readme_lines.append("data.yaml points to images/train and images/val relative to dataset root.")
    (YOLO_DATA_DIR / "README.txt").write_text("\n".join(readme_lines), encoding="utf-8")

    print("Konvertierung abgeschlossen.")
    print(f"Dataset root: {YOLO_DATA_DIR}")
    print(f"Artifacts: {ARTIFACTS_DIR}")
    print(f"data.yaml: {YOLO_DATA_DIR / 'data.yaml'}")
    print("Next steps:")
    print("  1) Artefakte prüfen (per_image_stats.csv, per_class_counts_train.csv).")
    print("  2) data.yaml in Training verwenden.")
    print("  3) class_weights.json / .pt für Loss-Gewichtung nutzen.")
    print("  4) hierarchy.json für hierarchische Heads laden.")
    if args.make_negatives:
        print("  5) Negative wurden erzeugt (leere Labeldateien).")
    return 0

# Argumente

def parse_bool(x): 
    return str(x).lower() in {"1", "true", "yes", "y", "t"}


def parse_args():
    p = argparse.ArgumentParser(
        description="MIO-TCD → YOLO (vehicle-only) mit stratifiziertem Split, Klassen-Gewichten, optionalen Negativen und Hierarchie-Export."
    )
    p.add_argument("--gt_csv", type=str, default="gt_train.csv", help="Pfad zur MIO-TCD Ground-Truth CSV (kein Header).")
    p.add_argument("--images_dir", type=str, default="train/", help="Pfad zu MIO-TCD Bildern.")
    p.add_argument("--out_dir", type=str, default="mio_tcd_yolo_vehicles_only/", help="Ziel-Wurzelverzeichnis.")
    p.add_argument("--val_ratio", type=float, default=0.15, help="Validierungsanteil (in Bildern).")
    p.add_argument("--copy_images", type=parse_bool, default=True, help="True: Dateien kopieren; False: Symlinks (sofern erlaubt).")
    p.add_argument("--seed", type=int, default=2, help="Zufallssamen für Split.")
    p.add_argument("--min_val_per_class", type=int, default=1, help="Mindestanzahl pro Klasse in Val (über Bildpräsenz).")
    p.add_argument("--make_plots", type=parse_bool, default=False, help="Diagnoseplots erstellen (falls matplotlib verfügbar).")
    p.add_argument("--cb_beta", type=float, default=DEFAULT_CB_BETA, help="Beta für Effective-Number-Gewichte.")
    p.add_argument("--make_negatives", type=parse_bool, default=False, help="Negative (Hintergrund-only) erzeugen.")
    p.add_argument("--num_negatives", type=int, default=1, help="Anzahl Negativduplikate pro positivem Bild.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    rc = main(args)
    sys.exit(rc)

# train_aux_heads.py
"""
(compressed version)
This script trains the auxiliary L1 and L2 heads for the Hierarchical Object Detection project.
It uses a high-performance, chunk-aware data loading pipeline to stream pre-computed
feature vectors from disk, maximizing GPU utilization.

Key Features:
- Efficient I/O with a custom ChunkedRandomSampler.
- Detailed logging to Weights & Biases (W&B), including per-sample-type losses.
- Calculation of L1/L2 Average Precision and rescue/rejection statistics during validation.
- EMA, mixed precision (FP16), and a warmup+cosine LR schedule.
- Tuned defaults for all hyperparameters, allowing direct execution
"""
import argparse
import json
import random
import sys
from collections import defaultdict
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Sampler
from torchmetrics.classification import (BinaryAveragePrecision,
                                          MulticlassAveragePrecision)
from tqdm import tqdm

try:
    import wandb
except ImportError:
    print("wandb not installed. Logging will be disabled. Run 'pip install wandb' to enable.")
    class DummyWandb:
        def __init__(self, *args, **kwargs): pass
        def init(self, *args, **kwargs): return self
        def log(self, *args, **kwargs): pass
        def watch(self, *args, **kwargs): pass
        def finish(self, *args, **kwargs): pass
    wandb = DummyWandb()

# Model Definition
class AuxHeadsMLP(nn.Module):
    """
    An MLP with a shared trunk and DECOUPLED necks for the L1 and L2 heads.
    """
    def __init__(self, in_dim: int, num_l2_classes: int, hidden_dim: int = 512, neck_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        
        # Shared Trunk: Learns a general representation (output dim: 256).
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        #  Decoupled & Asymmetric Necks
        # L2 Neck
        self.l2_neck = nn.Sequential(
            nn.Linear(hidden_dim // 2, neck_dim),
            nn.LayerNorm(neck_dim),
            nn.SiLU(inplace=True)
        )
        
        # L1 Neck
        l1_neck_dim = neck_dim # No performance benefit observed by doubling dimension
        self.l1_neck = nn.Sequential(
            nn.Linear(hidden_dim // 2, l1_neck_dim),
            nn.LayerNorm(l1_neck_dim),
            nn.SiLU(inplace=True)
        )
        
        # 3. Final Heads: Perform classification on their specialized features.
        self.l2_head = nn.Linear(neck_dim, num_l2_classes)
        self.l1_head = nn.Linear(l1_neck_dim, 1) # IMPORTANT!!!!! Input dim must match the l1_neck_dim if changed.

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # The forward pass logic remains identical.
        shared_features = self.trunk(x)
        l2_features = self.l2_neck(shared_features)
        l1_features = self.l1_neck(shared_features)
        return {
            "l1_logits": self.l1_head(l1_features),
            "l2_logits": self.l2_head(l2_features)
        }

# EMA (Exponential Moving Average)
class EMA:
    """Maintains an exponential moving average of model weights."""
    def __init__(self, model, decay=0.9999):
        self.model = model; self.decay = decay; self.updates = 0
        self.shadow = deepcopy(self.model.state_dict())
    def update(self):
        self.updates += 1
        decay = min(self.decay, (1.0 + self.updates) / (10.0 + self.updates))
        for name, param in self.model.state_dict().items():
            if param.is_floating_point():
                self.shadow[name] = decay * self.shadow[name] + (1 - decay) * param
    def apply(self): return self.shadow

# Dataset & Sampler
class ChunkedFeatureDataset(Dataset):
    """Loads one chunk of pre-computed features into memory at a time."""
    def __init__(self, chunk_paths: List[Path]):
        self.chunk_paths = chunk_paths
        self.chunk_lengths = []
        print("Pre-scanning chunks to determine dataset length...")
        for path in tqdm(self.chunk_paths, desc="Scanning chunks"):
            try:
                data = torch.load(path, weights_only=False)
                self.chunk_lengths.append(len(data['metadata']))
            except Exception as e:
                print(f"Warning: Could not read chunk {path}, skipping. Error: {e}")
        self.cumulative_lengths = np.cumsum(self.chunk_lengths).tolist()
        self.total_length = self.cumulative_lengths[-1] if self.cumulative_lengths else 0
        self.current_chunk_idx = -1; self.current_chunk_data = None
        print(f"Found {self.total_length:,} total ROIs in {len(self.chunk_paths)} chunks.")
    def __len__(self) -> int: return self.total_length
    def _load_chunk(self, chunk_idx: int):
        if chunk_idx != self.current_chunk_idx:
            path = self.chunk_paths[chunk_idx]
            self.current_chunk_data = torch.load(path, weights_only=False)
            self.current_chunk_idx = chunk_idx
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, torch.Tensor, Dict]:
        if idx < 0 or idx >= self.total_length: raise IndexError("Index out of range")
        chunk_idx = np.searchsorted(self.cumulative_lengths, idx, side='right')
        self._load_chunk(chunk_idx)
        local_idx = idx - (self.cumulative_lengths[chunk_idx - 1] if chunk_idx > 0 else 0)
        feature = self.current_chunk_data['features'][local_idx].float()
        l1_target = self.current_chunk_data['l1_targets'][local_idx].long()
        l2_target = self.current_chunk_data['l2_targets'][local_idx].float()
        metadata = self.current_chunk_data['metadata'][local_idx]
        return feature, l1_target, l2_target, metadata

class ChunkedRandomSampler(Sampler[int]):
    """Shuffles chunk order, then shuffles items within each chunk for I/O efficiency."""
    def __init__(self, dataset: ChunkedFeatureDataset):
        self.dataset = dataset
    def __iter__(self):
        chunk_indices = list(range(len(self.dataset.chunk_paths)))
        random.shuffle(chunk_indices)
        for chunk_idx in chunk_indices:
            start_idx = self.dataset.cumulative_lengths[chunk_idx-1] if chunk_idx > 0 else 0
            chunk_len = self.dataset.chunk_lengths[chunk_idx]
            indices_in_chunk = list(range(chunk_len))
            random.shuffle(indices_in_chunk)
            for local_idx in indices_in_chunk: yield start_idx + local_idx
    def __len__(self) -> int: return len(self.dataset)

def collate_fn(batch):
    features, l1_targets, l2_targets, metadatas = zip(*batch)
    return (torch.stack(features, 0), torch.stack(l1_targets, 0), torch.stack(l2_targets, 0), list(metadatas))

# --- Main Trainer Class ---
class AuxHeadsTrainer:
    def __init__(self, args):
        self.args = args; self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        with open(args.manifest) as f: self.manifest = json.load(f)
        self.wandb_run = wandb.init(project=args.wandb_project, name=args.run_name, config=args, mode="online" if args.wandb_project else "disabled")
        self._setup_dataloaders(); self._setup_model_and_optimizer(); self._setup_losses()
        self.scaler = torch.cuda.amp.GradScaler(enabled=args.fp16); self.best_val_loss = float('inf')

    def _setup_dataloaders(self):
        train_chunks = [Path(p) for p in self.manifest['chunks']['train']]
        val_chunks = [Path(p) for p in self.manifest['chunks']['val']]
        if not train_chunks or not Path(train_chunks[0]).exists():
            print(f"\n[ERROR] Training chunk files not found. Attempted path: {Path(train_chunks[0]).resolve() if train_chunks else 'N/A'}"); sys.exit(1)
        train_dataset = ChunkedFeatureDataset(train_chunks)
        val_dataset = ChunkedFeatureDataset(val_chunks)
        train_sampler = ChunkedRandomSampler(train_dataset)
        self.train_loader = DataLoader(train_dataset, batch_size=self.args.batch_size, sampler=train_sampler, num_workers=self.args.num_workers, pin_memory=True, collate_fn=collate_fn)
        self.val_loader = DataLoader(val_dataset, batch_size=self.args.batch_size * 2, shuffle=False, num_workers=self.args.num_workers, pin_memory=True, collate_fn=collate_fn)

    def _setup_model_and_optimizer(self):
        num_l2 = len(self.manifest['hierarchy']['L2_NAMES']); in_dim = self.manifest['feature_extraction']['compressed_dim']
        self.model = AuxHeadsMLP(in_dim, num_l2, hidden_dim=512, dropout=self.args.dropout).to(self.device)
        self.ema = EMA(self.model) if self.args.ema else None
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.wd)
        warmup_scheduler = optim.lr_scheduler.LinearLR(self.optimizer, start_factor=0.1, end_factor=1.0, total_iters=self.args.warmup_epochs)
        main_scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.args.epochs - self.args.warmup_epochs, eta_min=self.args.lr_final)
        self.scheduler = optim.lr_scheduler.SequentialLR(self.optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[self.args.warmup_epochs])
        wandb.watch(self.model)

    def _setup_losses(self):
        stats_path = Path(self.manifest['paths']['stats_train'])
        if not stats_path.exists(): print(f"\n[ERROR] Stats file not found at: {stats_path.resolve()}"); sys.exit(1)
        with open(stats_path) as f: stats = json.load(f)
        pos_samples = sum(stats['l2_positive_counts'].values()); total_samples = stats['total_roi']; neg_samples = total_samples - pos_samples
        l1_pos_weight = torch.tensor(neg_samples / pos_samples, device=self.device) if pos_samples > 0 else torch.tensor(1.0)
        weights_path = Path(self.manifest['paths']['class_weights'])
        if not weights_path.exists(): print(f"\n[ERROR] Weights file not found at: {weights_path.resolve()}"); sys.exit(1)
        with open(weights_path) as f: weights_data = json.load(f)
        l2_weights_dict = weights_data['l2_weights']
        l2_pos_weight = torch.tensor([l2_weights_dict[str(i)] for i in range(len(l2_weights_dict))], device=self.device)
        self.l1_criterion = nn.BCEWithLogitsLoss(pos_weight=l1_pos_weight); self.l2_criterion = nn.BCEWithLogitsLoss(pos_weight=l2_pos_weight)

    def train(self):
        for epoch in range(self.args.epochs):
            self.model.train(); self._train_one_epoch(epoch)
            self.model.eval()
            val_model = self.model
            if self.ema: val_model = deepcopy(self.model); val_model.load_state_dict(self.ema.apply())
            val_loss = self._validate_one_epoch(epoch, val_model)
            self.scheduler.step()
            if val_loss < self.best_val_loss: self.best_val_loss = val_loss; self._save_checkpoint('best.pt', epoch, val_model)
            self._save_checkpoint('last.pt', epoch, val_model)
        self.wandb_run.finish()

    def _process_batch(self, batch, prefix, model_to_use):
        features, l1_targets, l2_targets, metadata = batch
        features = features.to(self.device, non_blocking=True); l1_targets = l1_targets.to(self.device, non_blocking=True).unsqueeze(1).float(); l2_targets = l2_targets.to(self.device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=self.args.fp16):
            outputs = model_to_use(features)
            l1_loss = self.l1_criterion(outputs['l1_logits'], l1_targets)
            l2_loss = self.l2_criterion(outputs['l2_logits'], l2_targets)
            total_loss = self.args.l1_weight * l1_loss + self.args.l2_weight * l2_loss
        log_data = {f"{prefix}/total_loss": total_loss.item(), f"{prefix}/l1_loss": l1_loss.item(), f"{prefix}/l2_loss": l2_loss.item()}
        losses_by_type = defaultdict(lambda: {'l1': [], 'l2': []})
        for i, meta in enumerate(metadata):
            sample_type = meta['type']
            item_l1_loss = nn.functional.binary_cross_entropy_with_logits(outputs['l1_logits'][i], l1_targets[i], pos_weight=self.l1_criterion.pos_weight)
            item_l2_loss = nn.functional.binary_cross_entropy_with_logits(outputs['l2_logits'][i], l2_targets[i], pos_weight=self.l2_criterion.pos_weight)
            losses_by_type[sample_type]['l1'].append(item_l1_loss.item()); losses_by_type[sample_type]['l2'].append(item_l2_loss.item())
        for sample_type, losses in losses_by_type.items():
            log_data[f"{prefix}/L1_loss_{sample_type}"] = np.mean(losses['l1']); log_data[f"{prefix}/L2_loss_{sample_type}"] = np.mean(losses['l2'])
        wandb.log(log_data)
        return total_loss, outputs

    def _train_one_epoch(self, epoch):
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.args.epochs} [Train]")
        for batch in pbar:
            self.optimizer.zero_grad(); loss, _ = self._process_batch(batch, "train", self.model)
            self.scaler.scale(loss).backward(); self.scaler.step(self.optimizer); self.scaler.update()
            if self.ema: self.ema.update()
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{self.optimizer.param_groups[0]['lr']:.2e}")

    def _validate_one_epoch(self, epoch, model_to_use):
        total_loss_epoch = 0; l1_ap = BinaryAveragePrecision().to(self.device)
        num_l2 = len(self.manifest['hierarchy']['L2_NAMES']); l2_ap = MulticlassAveragePrecision(num_classes=num_l2, average='none').to(self.device)
        stats = defaultdict(lambda: {'total': 0, 'rejected': 0})
        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch+1}/{self.args.epochs} [Val]")
        with torch.no_grad():
            for batch in pbar:
                loss, outputs = self._process_batch(batch, "val", model_to_use); total_loss_epoch += loss.item()
                l1_preds = torch.sigmoid(outputs['l1_logits']); l2_preds = torch.softmax(outputs['l2_logits'], dim=1)
                l1_ap.update(l1_preds.squeeze(), batch[1].to(self.device)); l2_ap.update(l2_preds, batch[2].to(self.device).argmax(dim=1))
                for i, meta in enumerate(batch[3]):
                    sample_type = meta['type']
                    is_pos = sample_type in ['gt_pos', 'pred_tp', 'jitter_pos']; is_neg = sample_type in ['pred_fp', 'jitter_neg', 'bg_neg']
                    is_rejected = l1_preds[i].item() < self.args.l1_reject_thresh
                    if is_pos: stats['true_positives']['total'] += 1; stats['true_positives']['rejected'] += is_rejected
                    if is_neg: stats['hard_negatives']['total'] += 1; stats['hard_negatives']['rejected'] += is_rejected
        val_log = {"val/epoch": epoch + 1, "val/L1_AP": l1_ap.compute().item()}
        l2_ap_per_class = l2_ap.compute()
        for i, name in enumerate(self.manifest['hierarchy']['L2_NAMES']): val_log[f"val/L2_AP_{name}"] = l2_ap_per_class[i].item()
        val_log["val/L2_mAP"] = l2_ap_per_class.mean().item()
        tp_stats = stats['true_positives']; hn_stats = stats['hard_negatives']
        val_log["val/Mistaken_Rejection_Rate"] = (tp_stats['rejected'] / tp_stats['total']) if tp_stats['total'] > 0 else 0
        val_log["val/Correct_Rejection_Rate"] = (hn_stats['rejected'] / hn_stats['total']) if hn_stats['total'] > 0 else 0
        wandb.log(val_log)
        return total_loss_epoch / len(self.val_loader)

    def _save_checkpoint(self, name, epoch, model_to_save):
        save_dir = Path(self.args.save_dir); save_dir.mkdir(parents=True, exist_ok=True)
        torch.save({'epoch': epoch, 'model_state_dict': model_to_save.state_dict(), 'optimizer_state_dict': self.optimizer.state_dict(), 'best_val_loss': self.best_val_loss}, save_dir / name)
        
def main():
    parser = argparse.ArgumentParser(
        description="Train Auxiliary Heads for Hierarchical Object Detection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # Shows defaults in --help
    )
    
    # Core Arguments
    parser.add_argument('--manifest', type=str, default='aux_heads_chunks_pt/manifest.json', help='Path to the manifest.json file')
    parser.add_argument('--save-dir', type=str, default='runs/train_aux_heads', help='Directory to save checkpoints')
    parser.add_argument('--seed', type=int, default=2, help='Seed halt')
    
    # Training Hyperparameters
    parser.add_argument('--epochs', type=int, default=60, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=1024, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate')
    parser.add_argument('--lr-final', type=float, default=1e-6, help='Final learning rate (eta_min) for cosine scheduler')
    parser.add_argument('--wd', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--warmup-epochs', type=int, default=3, help='Number of epochs for linear LR warmup')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate in the MLP')
    parser.add_argument('--l1-weight', type=float, default=3, help='Weight for L1 loss component')
    parser.add_argument('--l2-weight', type=float, default=1.0, help='Weight for L2 loss component')

    # Performance & Hardware Arguments
    parser.add_argument('--num-workers', type=int, default=8, help='Number of workers for DataLoader. Tune based on your CPU.')
    parser.add_argument('--fp16', action=argparse.BooleanOptionalAction, default=True, help='Enable mixed precision training')
    parser.add_argument('--ema', action=argparse.BooleanOptionalAction, default=True, help='Enable Exponential Moving Average of weights')

    # Validation & Logging Arguments
    parser.add_argument('--l1-reject-thresh', type=float, default=0.5, help='L1 probability threshold for validation stats')
    parser.add_argument('--wandb-project', type=str, default='YOLO_Hierarchical_Heads', help='W&B project name. If None, W&B is disabled.')
    parser.add_argument('--run-name', type=str, default=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}", help='Name for this run in W&B')
    
    args = parser.parse_args()

    # Seed for reproducibility
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)

    trainer = AuxHeadsTrainer(args)
    trainer.train()

if __name__ == '__main__':
    main()
"""
Training script for DCCRN speech enhancement model.

Usage:
    python src/train.py --config configs/dccrn_config.yaml
"""

import os
import sys
import argparse
import time
from pathlib import Path

import yaml
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.dccrn import build_dccrn
from src.losses.loss import CombinedLoss
from src.datasets.audio_dataset import create_dataloader
from src.utils.metrics import evaluate_batch


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


class Trainer:
    """Training manager for DCCRN model.
    
    Handles the full training loop including:
    - Mixed precision training
    - Learning rate scheduling
    - Early stopping
    - Checkpointing
    - TensorBoard logging
    """

    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Build model
        self.model = build_dccrn(config).to(self.device)
        num_params = self.model.get_num_params()
        print(f"Model parameters: {num_params:,} ({num_params / 1e6:.2f}M)")
        
        # Loss function
        loss_cfg = config.get("loss", {})
        stft_cfg = config.get("stft", {})
        self.criterion = CombinedLoss(
            si_snr_weight=loss_cfg.get("si_snr_weight", 0.5),
            magnitude_weight=loss_cfg.get("magnitude_weight", 0.3),
            complex_weight=loss_cfg.get("complex_weight", 0.2),
            n_fft=stft_cfg.get("n_fft", 512),
            hop_length=stft_cfg.get("hop_length", 256),
        ).to(self.device)
        
        # Optimizer
        train_cfg = config.get("training", {})
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=train_cfg.get("learning_rate", 1e-3),
            weight_decay=train_cfg.get("weight_decay", 1e-4),
        )
        
        # LR Scheduler
        sched_cfg = train_cfg.get("scheduler", {})
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=sched_cfg.get("factor", 0.5),
            patience=sched_cfg.get("patience", 3),
            min_lr=sched_cfg.get("min_lr", 1e-5),
        )
        
        # Mixed precision
        self.use_amp = train_cfg.get("use_amp", True) and self.device.type == "cuda"
        self.scaler = GradScaler(enabled=self.use_amp)
        
        # Training settings
        self.epochs = train_cfg.get("epochs", 200)
        self.gradient_clip = train_cfg.get("gradient_clip", 5.0)
        
        # Early stopping
        es_cfg = train_cfg.get("early_stopping", {})
        self.es_patience = es_cfg.get("patience", 10)
        self.es_min_delta = es_cfg.get("min_delta", 0.001)
        self.best_val_loss = float("inf")
        self.es_counter = 0
        
        # Logging
        log_cfg = config.get("logging", {})
        self.log_dir = log_cfg.get("log_dir", "logs")
        self.checkpoint_dir = log_cfg.get("checkpoint_dir", "checkpoints")
        self.save_every = log_cfg.get("save_every", 5)
        self.log_interval = log_cfg.get("log_interval", 100)
        
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        self.writer = SummaryWriter(self.log_dir) if log_cfg.get("tensorboard", True) else None
        self.global_step = 0

    def create_dataloaders(self):
        """Create train and validation dataloaders."""
        data_cfg = self.config.get("data", {})
        audio_cfg = self.config.get("audio", {})
        train_cfg = self.config.get("training", {})
        mixing_cfg = self.config.get("mixing", {})
        
        self.train_loader = create_dataloader(
            clean_dir=data_cfg.get("train_clean_dir", "data/train/clean"),
            noisy_dir=data_cfg.get("train_noisy_dir", "data/train/noisy"),
            batch_size=train_cfg.get("batch_size", 16),
            sample_rate=audio_cfg.get("sample_rate", 16000),
            segment_length=audio_cfg.get("n_samples", 64000),
            dynamic_mixing=mixing_cfg.get("dynamic_mixing", False),
            num_workers=data_cfg.get("num_workers", 4),
            pin_memory=data_cfg.get("pin_memory", True),
            shuffle=True,
        )
        
        self.val_loader = create_dataloader(
            clean_dir=data_cfg.get("val_clean_dir", "data/val/clean"),
            noisy_dir=data_cfg.get("val_noisy_dir", "data/val/noisy"),
            batch_size=train_cfg.get("batch_size", 16),
            sample_rate=audio_cfg.get("sample_rate", 16000),
            segment_length=audio_cfg.get("n_samples", 64000),
            dynamic_mixing=False,
            num_workers=data_cfg.get("num_workers", 4),
            pin_memory=data_cfg.get("pin_memory", True),
            shuffle=False,
        )
        
        print(f"Train samples: {len(self.train_loader.dataset)}")
        print(f"Val samples: {len(self.val_loader.dataset)}")

    def train_epoch(self, epoch: int) -> dict:
        """Train for one epoch.
        
        Returns:
            Dictionary with epoch metrics
        """
        self.model.train()
        epoch_losses = {"total": 0, "si_snr": 0, "magnitude": 0, "complex": 0}
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}", leave=False)
        for batch in pbar:
            noisy = batch["noisy"].to(self.device)
            clean = batch["clean"].to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            with autocast(enabled=self.use_amp):
                enhanced = self.model(noisy)
                loss_dict = self.criterion(enhanced, clean)
            
            # Backward pass
            self.scaler.scale(loss_dict["total"]).backward()
            
            # Gradient clipping
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # Accumulate losses
            for key in epoch_losses:
                epoch_losses[key] += loss_dict[key] if key != "total" else loss_dict["total"].item()
            num_batches += 1
            
            # Logging
            if self.global_step % self.log_interval == 0 and self.writer:
                self.writer.add_scalar("train/loss_total", loss_dict["total"].item(), self.global_step)
                self.writer.add_scalar("train/loss_si_snr", loss_dict["si_snr"], self.global_step)
                self.writer.add_scalar("train/lr", self.optimizer.param_groups[0]["lr"], self.global_step)
            
            pbar.set_postfix(loss=f"{loss_dict['total'].item():.4f}")
            self.global_step += 1
        
        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= max(num_batches, 1)
        
        return epoch_losses

    @torch.no_grad()
    def validate(self, epoch: int) -> dict:
        """Run validation.
        
        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        val_losses = {"total": 0, "si_snr": 0, "magnitude": 0, "complex": 0}
        all_metrics = {"si_snr": [], "pesq": [], "stoi": []}
        num_batches = 0
        
        for batch in tqdm(self.val_loader, desc="Validation", leave=False):
            noisy = batch["noisy"].to(self.device)
            clean = batch["clean"].to(self.device)
            
            enhanced = self.model(noisy)
            loss_dict = self.criterion(enhanced, clean)
            
            for key in val_losses:
                val_losses[key] += loss_dict[key] if key != "total" else loss_dict["total"].item()
            num_batches += 1
            
            # Compute metrics on first few batches (PESQ is slow)
            if num_batches <= 5:
                metrics = evaluate_batch(
                    enhanced, clean,
                    sample_rate=self.config.get("audio", {}).get("sample_rate", 16000),
                )
                for key, val in metrics.items():
                    if val is not None:
                        all_metrics[key].append(val)
        
        # Average
        for key in val_losses:
            val_losses[key] /= max(num_batches, 1)
        
        # Average metrics
        avg_metrics = {}
        for key, vals in all_metrics.items():
            avg_metrics[key] = sum(vals) / len(vals) if vals else None
        
        # TensorBoard logging
        if self.writer:
            self.writer.add_scalar("val/loss_total", val_losses["total"], epoch)
            for key, val in avg_metrics.items():
                if val is not None:
                    self.writer.add_scalar(f"val/{key}", val, epoch)
        
        val_losses["metrics"] = avg_metrics
        return val_losses

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "global_step": self.global_step,
            "config": self.config,
        }
        
        # Save periodic checkpoint
        path = os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch}.pth")
        torch.save(state, path)
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, "best_model.pth")
            torch.save(state, best_path)
            print(f"  ✓ Saved best model (val_loss={self.best_val_loss:.4f})")
        
        # Save latest
        latest_path = os.path.join(self.checkpoint_dir, "latest.pth")
        torch.save(state, latest_path)

    def load_checkpoint(self, checkpoint_path: str):
        """Resume training from checkpoint."""
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        self.global_step = checkpoint.get("global_step", 0)
        
        return checkpoint.get("epoch", 0)

    def train(self, resume_from: str = None):
        """Full training loop."""
        start_epoch = 0
        
        if resume_from and os.path.exists(resume_from):
            start_epoch = self.load_checkpoint(resume_from) + 1
            print(f"Resuming from epoch {start_epoch}")
        
        self.create_dataloaders()
        
        print(f"\n{'='*60}")
        print(f"Starting training for {self.epochs} epochs")
        print(f"{'='*60}\n")
        
        for epoch in range(start_epoch, self.epochs):
            start_time = time.time()
            
            # Train
            train_losses = self.train_epoch(epoch)
            
            # Validate
            val_losses = self.validate(epoch)
            
            # Update LR
            self.scheduler.step(val_losses["total"])
            
            # Timing
            elapsed = time.time() - start_time
            
            # Print epoch summary
            metrics = val_losses.get("metrics", {})
            pesq_str = f"{metrics.get('pesq', 0):.3f}" if metrics.get("pesq") else "N/A"
            stoi_str = f"{metrics.get('stoi', 0):.3f}" if metrics.get("stoi") else "N/A"
            si_snr_str = f"{metrics.get('si_snr', 0):.2f}" if metrics.get("si_snr") else "N/A"
            
            print(
                f"Epoch {epoch:3d} | "
                f"Train Loss: {train_losses['total']:.4f} | "
                f"Val Loss: {val_losses['total']:.4f} | "
                f"PESQ: {pesq_str} | STOI: {stoi_str} | SI-SNR: {si_snr_str} | "
                f"LR: {self.optimizer.param_groups[0]['lr']:.6f} | "
                f"Time: {elapsed:.1f}s"
            )
            
            # Check for improvement
            is_best = False
            if val_losses["total"] < self.best_val_loss - self.es_min_delta:
                self.best_val_loss = val_losses["total"]
                self.es_counter = 0
                is_best = True
            else:
                self.es_counter += 1
            
            # Save checkpoint
            if (epoch + 1) % self.save_every == 0 or is_best:
                self.save_checkpoint(epoch, is_best)
            
            # Early stopping
            if self.es_counter >= self.es_patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                self.save_checkpoint(epoch)
                break
        
        print(f"\nTraining complete! Best val loss: {self.best_val_loss:.4f}")
        if self.writer:
            self.writer.close()


def main():
    parser = argparse.ArgumentParser(description="Train DCCRN for speech enhancement")
    parser.add_argument(
        "--config", type=str, default="configs/dccrn_config.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to checkpoint to resume from"
    )
    args = parser.parse_args()
    
    config = load_config(args.config)
    trainer = Trainer(config)
    trainer.train(resume_from=args.resume)


if __name__ == "__main__":
    main()

"""
Quick training script - Chay nhanh tren CPU voi tap nho de tao prototype model.
Dung de test giao dien truoc khi co model chinh thuc tu GPU.

Chay:
    python src/quick_train.py
"""

import os
import sys
import time
from pathlib import Path

import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.dccrn import build_dccrn
from src.losses.loss import CombinedLoss
from src.datasets.audio_dataset import AudioDataset


def load_config(path):
    with open(path, "r", encoding="utf-8-sig") as f:
        return yaml.safe_load(f)


def quick_train():
    config_path = "configs/quick_train_config.yaml"
    config = load_config(config_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print("=" * 60)

    # Build model
    model = build_dccrn(config).to(device)
    num_params = model.get_num_params()
    print(f"Model params: {num_params:,} ({num_params/1e6:.2f}M)")

    # Loss & optimizer
    stft_cfg = config.get("stft", {})
    loss_cfg = config.get("loss", {})
    criterion = CombinedLoss(
        si_snr_weight=loss_cfg.get("si_snr_weight", 0.4),
        magnitude_weight=loss_cfg.get("magnitude_weight", 0.2),
        complex_weight=loss_cfg.get("complex_weight", 0.1),
        mr_stft_weight=loss_cfg.get("mr_stft_weight", 0.3),
        n_fft=stft_cfg.get("n_fft", 512),
        hop_length=stft_cfg.get("hop_length", 256),
    ).to(device)

    train_cfg = config.get("training", {})
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_cfg.get("learning_rate", 1e-3),
        weight_decay=train_cfg.get("weight_decay", 1e-4),
    )

    data_cfg = config.get("data", {})
    audio_cfg = config.get("audio", {})

    # Dataset - gioi han 50 mau train, 20 mau val de nhanh
    MAX_TRAIN = 50
    MAX_VAL = 20

    train_dataset = AudioDataset(
        clean_dir=data_cfg["train_clean_dir"],
        noisy_dir=data_cfg["train_noisy_dir"],
        sample_rate=audio_cfg.get("sample_rate", 16000),
        segment_length=audio_cfg.get("n_samples", 32000),
        dynamic_mixing=False,
    )
    val_dataset = AudioDataset(
        clean_dir=data_cfg["val_clean_dir"],
        noisy_dir=data_cfg["val_noisy_dir"],
        sample_rate=audio_cfg.get("sample_rate", 16000),
        segment_length=audio_cfg.get("n_samples", 32000),
        dynamic_mixing=False,
    )

    # Subset nho
    train_size = min(MAX_TRAIN, len(train_dataset))
    val_size = min(MAX_VAL, len(val_dataset))
    train_subset = Subset(train_dataset, list(range(train_size)))
    val_subset = Subset(val_dataset, list(range(val_size)))

    print(f"Train samples: {train_size} / {len(train_dataset)} (gioi han)")
    print(f"Val   samples: {val_size}   / {len(val_dataset)} (gioi han)")
    print("=" * 60)

    batch_size = train_cfg.get("batch_size", 2)
    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True,
        num_workers=0, drop_last=True,
    )
    val_loader = DataLoader(
        val_subset, batch_size=batch_size, shuffle=False,
        num_workers=0, drop_last=False,
    )

    epochs = train_cfg.get("epochs", 5)
    gradient_clip = train_cfg.get("gradient_clip", 5.0)

    os.makedirs("checkpoints/quick", exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(epochs):
        t0 = time.time()

        # --- Train ---
        model.train()
        train_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            noisy = batch["noisy"].to(device)
            clean = batch["clean"].to(device)

            optimizer.zero_grad()
            enhanced = model(noisy)
            loss_dict = criterion(enhanced, clean)
            loss = loss_dict["total"]
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()

            train_loss += loss.item()
            n_batches += 1
            print(f"  Epoch {epoch+1} | batch {n_batches}/{len(train_loader)} | loss={loss.item():.4f}", end="\r")

        avg_train_loss = train_loss / max(n_batches, 1)

        # --- Validate ---
        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for batch in val_loader:
                noisy = batch["noisy"].to(device)
                clean = batch["clean"].to(device)
                enhanced = model(noisy)
                loss_dict = criterion(enhanced, clean)
                val_loss += loss_dict["total"].item()
                n_val += 1
        avg_val_loss = val_loss / max(n_val, 1)

        elapsed = time.time() - t0
        print(
            f"\nEpoch {epoch+1:2d}/{epochs} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Time: {elapsed:.1f}s"
        )

        # Luu checkpoint
        is_best = avg_val_loss < best_val_loss
        if is_best:
            best_val_loss = avg_val_loss

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
            "config": config,
        }, "checkpoints/quick/latest.pth")

        if is_best:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
                "config": config,
            }, "checkpoints/quick/best_model.pth")
            print(f"  => Saved best model (val_loss={best_val_loss:.4f})")

    print("=" * 60)
    print(f"Training done! Best val loss: {best_val_loss:.4f}")
    print("Checkpoint: checkpoints/quick/best_model.pth")
    print("Khi co model tot hon tu GPU, chi can thay file best_model.pth la xong!")


if __name__ == "__main__":
    quick_train()

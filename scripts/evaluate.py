"""
Evaluation script for speech enhancement model.

Computes PESQ, STOI, and SI-SNR metrics on the test set.

Usage:
    python scripts/evaluate.py \
        --model_path checkpoints/best_model.pth \
        --test_clean data/test/clean \
        --test_noisy data/test/noisy
"""

import os
import sys
import argparse
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.dccrn import build_dccrn
from src.utils.audio_utils import load_audio
from src.utils.metrics import si_snr, pesq_score, stoi_score, HAS_PESQ, HAS_STOI
from scripts.inference import denoise_audio


def main():
    parser = argparse.ArgumentParser(description="Evaluate DeNoiseAI model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--test_clean", type=str, required=True, help="Directory with clean test audio")
    parser.add_argument("--test_noisy", type=str, required=True, help="Directory with noisy test audio")
    parser.add_argument("--max_files", type=int, default=None, help="Max files to evaluate")
    parser.add_argument("--device", type=str, default="auto", help="Device")
    args = parser.parse_args()
    
    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    # Load model
    checkpoint = torch.load(args.model_path, map_location=device)
    config = checkpoint.get("config", {})
    model = build_dccrn(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    sample_rate = config.get("audio", {}).get("sample_rate", 16000)
    segment_length = config.get("audio", {}).get("n_samples", 64000)
    
    # Find matching clean-noisy pairs
    audio_exts = {".wav", ".flac", ".mp3", ".ogg"}
    clean_files = sorted([
        f for f in Path(args.test_clean).rglob("*")
        if f.suffix.lower() in audio_exts
    ])
    noisy_files = sorted([
        f for f in Path(args.test_noisy).rglob("*")
        if f.suffix.lower() in audio_exts
    ])
    
    if args.max_files:
        clean_files = clean_files[:args.max_files]
        noisy_files = noisy_files[:args.max_files]
    
    assert len(clean_files) == len(noisy_files), \
        f"Mismatch: {len(clean_files)} clean vs {len(noisy_files)} noisy files"
    
    print(f"\n{'='*60}")
    print(f"DeNoiseAI Evaluation")
    print(f"{'='*60}")
    print(f"Model: {args.model_path}")
    print(f"Test files: {len(clean_files)}")
    print(f"Device: {device}")
    print(f"{'='*60}\n")
    
    # Metrics storage
    metrics_noisy = {"si_snr": [], "pesq": [], "stoi": []}
    metrics_enhanced = {"si_snr": [], "pesq": [], "stoi": []}
    
    for clean_file, noisy_file in tqdm(zip(clean_files, noisy_files), total=len(clean_files)):
        # Load audio
        clean = load_audio(str(clean_file), sample_rate)
        noisy = load_audio(str(noisy_file), sample_rate)
        
        # Denoise
        with torch.no_grad():
            enhanced = denoise_audio(model, noisy, device, segment_length)
        
        # Compute metrics for noisy input
        min_len = min(len(clean), len(noisy), len(enhanced))
        clean_np = clean[:min_len].numpy()
        noisy_np = noisy[:min_len].numpy()
        enhanced_np = enhanced[:min_len].numpy()
        
        # SI-SNR
        metrics_noisy["si_snr"].append(si_snr(noisy[:min_len], clean[:min_len]))
        metrics_enhanced["si_snr"].append(si_snr(enhanced[:min_len], clean[:min_len]))
        
        # PESQ
        if HAS_PESQ:
            try:
                metrics_noisy["pesq"].append(pesq_score(noisy_np, clean_np, sample_rate))
                metrics_enhanced["pesq"].append(pesq_score(enhanced_np, clean_np, sample_rate))
            except Exception:
                pass
        
        # STOI
        if HAS_STOI:
            try:
                metrics_noisy["stoi"].append(stoi_score(noisy_np, clean_np, sample_rate))
                metrics_enhanced["stoi"].append(stoi_score(enhanced_np, clean_np, sample_rate))
            except Exception:
                pass
    
    # Print results
    print(f"\n{'='*60}")
    print(f"{'EVALUATION RESULTS':^60}")
    print(f"{'='*60}")
    print(f"{'Metric':<15} {'Noisy (input)':>15} {'Enhanced':>15} {'Improvement':>15}")
    print(f"{'-'*60}")
    
    for metric_name in ["si_snr", "pesq", "stoi"]:
        noisy_vals = metrics_noisy[metric_name]
        enhanced_vals = metrics_enhanced[metric_name]
        
        if noisy_vals and enhanced_vals:
            noisy_mean = np.mean(noisy_vals)
            enhanced_mean = np.mean(enhanced_vals)
            improvement = enhanced_mean - noisy_mean
            
            print(
                f"{metric_name.upper():<15} "
                f"{noisy_mean:>15.3f} "
                f"{enhanced_mean:>15.3f} "
                f"{'+' if improvement > 0 else ''}{improvement:>14.3f}"
            )
    
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

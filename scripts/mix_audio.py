"""
Mix clean speech with noise to create noisy-clean training pairs.

For each clean audio file, generates multiple noisy variants at
different SNR levels using random noise samples.

Usage:
    python scripts/mix_audio.py \
        --clean_dir data/processed/train/clean \
        --noise_dir data/processed/noise \
        --output_dir data/processed/train/noisy \
        --snr_levels -5 0 5 10 15 20 \
        --variants_per_file 3
"""

import os
import sys
import argparse
import random
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.audio_utils import load_audio, save_audio, mix_audio_at_snr


def main():
    parser = argparse.ArgumentParser(description="Mix clean speech with noise for DeNoiseAI")
    parser.add_argument("--clean_dir", type=str, required=True, help="Directory with clean speech files")
    parser.add_argument("--noise_dir", type=str, required=True, help="Directory with noise files")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for noisy files")
    parser.add_argument(
        "--snr_levels", type=float, nargs="+",
        default=[-5, 0, 5, 10, 15, 20],
        help="SNR levels in dB"
    )
    parser.add_argument(
        "--variants_per_file", type=int, default=1,
        help="Number of noisy variants per clean file"
    )
    parser.add_argument("--sample_rate", type=int, default=16000, help="Sample rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Find files
    audio_exts = {".wav", ".flac", ".mp3", ".ogg"}
    clean_files = sorted([
        str(f) for f in Path(args.clean_dir).rglob("*")
        if f.suffix.lower() in audio_exts
    ])
    noise_files = sorted([
        str(f) for f in Path(args.noise_dir).rglob("*")
        if f.suffix.lower() in audio_exts
    ])
    
    if not clean_files:
        print(f"✗ No clean audio files found in {args.clean_dir}")
        sys.exit(1)
    if not noise_files:
        print(f"✗ No noise files found in {args.noise_dir}")
        sys.exit(1)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"DeNoiseAI Audio Mixer")
    print(f"{'='*60}")
    print(f"Clean files: {len(clean_files)}")
    print(f"Noise files: {len(noise_files)}")
    print(f"SNR levels: {args.snr_levels} dB")
    print(f"Variants per file: {args.variants_per_file}")
    total_output = len(clean_files) * args.variants_per_file
    print(f"Expected output: {total_output} noisy files")
    print(f"{'='*60}\n")
    
    # Pre-load noise files (or load on-demand if too many)
    print("Loading noise files...")
    noise_cache = []
    for nf in tqdm(noise_files[:500], desc="Loading noise"):  # Cache up to 500 noise files
        try:
            noise_cache.append(load_audio(nf, args.sample_rate))
        except Exception:
            continue
    
    if not noise_cache:
        print("✗ Could not load any noise files")
        sys.exit(1)
    
    print(f"  ✓ Loaded {len(noise_cache)} noise files into cache\n")
    
    # Mix
    file_count = 0
    errors = 0
    
    for clean_path in tqdm(clean_files, desc="Mixing audio"):
        try:
            clean = load_audio(clean_path, args.sample_rate)
        except Exception:
            errors += 1
            continue
        
        # Get the clean file's basename for naming the noisy output
        clean_basename = Path(clean_path).stem
        
        for variant in range(args.variants_per_file):
            # Pick random noise and SNR
            noise = random.choice(noise_cache)
            snr_db = random.choice(args.snr_levels)
            
            # Mix
            noisy, _ = mix_audio_at_snr(clean, noise, snr_db)
            
            # Save — name matches clean file for easy pairing
            if args.variants_per_file == 1:
                output_name = f"{clean_basename}.wav"
            else:
                output_name = f"{clean_basename}_v{variant}_snr{int(snr_db)}.wav"
            
            output_path = os.path.join(args.output_dir, output_name)
            save_audio(output_path, noisy, args.sample_rate)
            file_count += 1
    
    print(f"\n{'='*60}")
    print(f"Mixing complete!")
    print(f"  ✓ Generated: {file_count} noisy files")
    if errors > 0:
        print(f"  ⚠ Errors: {errors} files skipped")
    print(f"  📁 Output: {args.output_dir}")
    print(f"{'='*60}")
    print(f"\nNext step: Start training:")
    print(f"  python src/train.py --config configs/dccrn_config.yaml")


if __name__ == "__main__":
    main()

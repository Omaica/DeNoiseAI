"""
Prepare and preprocess audio data for training.

Steps:
1. Scan raw dataset directories for audio files
2. Resample all audio to target sample rate (16kHz)
3. Normalize amplitude
4. Split into segments of fixed length
5. Organize into train/val/test splits

Usage:
    python scripts/prepare_data.py --input_dir data/raw --output_dir data/processed
"""

import os
import sys
import argparse
import random
from pathlib import Path

import torch
import torchaudio
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.audio_utils import load_audio, save_audio, normalize_audio, segment_audio


def find_audio_files(directory: str) -> list:
    """Recursively find all audio files in a directory."""
    audio_exts = {".wav", ".flac", ".mp3", ".ogg", ".sph"}
    files = []
    for f in Path(directory).rglob("*"):
        if f.suffix.lower() in audio_exts and f.is_file():
            files.append(str(f))
    return sorted(files)


def process_speech_files(
    input_files: list,
    output_dir: str,
    sample_rate: int = 16000,
    segment_length: float = 4.0,
    min_duration: float = 1.0,
):
    """Process clean speech files: resample, normalize, segment.
    
    Args:
        input_files: List of input file paths
        output_dir: Output directory
        sample_rate: Target sample rate
        segment_length: Segment length in seconds
        min_duration: Minimum duration to keep (seconds)
    """
    os.makedirs(output_dir, exist_ok=True)
    segment_samples = int(segment_length * sample_rate)
    min_samples = int(min_duration * sample_rate)
    
    file_count = 0
    
    for filepath in tqdm(input_files, desc="Processing speech"):
        try:
            waveform = load_audio(filepath, sample_rate)
        except Exception as e:
            print(f"  ⚠ Skipping {filepath}: {e}")
            continue
        
        # Skip files that are too short
        if len(waveform) < min_samples:
            continue
        
        # Normalize
        waveform = normalize_audio(waveform)
        
        # Segment into fixed-length chunks
        segments = segment_audio(waveform, segment_samples)
        
        for i, seg in enumerate(segments):
            # Skip if segment is mostly silence
            if seg.abs().max() < 0.01:
                continue
            
            output_path = os.path.join(output_dir, f"speech_{file_count:06d}.wav")
            save_audio(output_path, seg, sample_rate)
            file_count += 1
    
    print(f"  ✓ Generated {file_count} speech segments in {output_dir}")
    return file_count


def process_noise_files(
    input_files: list,
    output_dir: str,
    sample_rate: int = 16000,
    segment_length: float = 4.0,
):
    """Process noise files: resample, normalize, segment."""
    os.makedirs(output_dir, exist_ok=True)
    segment_samples = int(segment_length * sample_rate)
    
    file_count = 0
    
    for filepath in tqdm(input_files, desc="Processing noise"):
        try:
            waveform = load_audio(filepath, sample_rate)
        except Exception as e:
            print(f"  ⚠ Skipping {filepath}: {e}")
            continue
        
        # Normalize
        waveform = normalize_audio(waveform)
        
        # Segment
        segments = segment_audio(waveform, segment_samples)
        
        for seg in segments:
            output_path = os.path.join(output_dir, f"noise_{file_count:06d}.wav")
            save_audio(output_path, seg, sample_rate)
            file_count += 1
    
    print(f"  ✓ Generated {file_count} noise segments in {output_dir}")
    return file_count


def split_files(file_list: list, train_ratio: float = 0.9, val_ratio: float = 0.05):
    """Split file list into train/val/test sets.
    
    Args:
        file_list: List of file paths
        train_ratio: Fraction for training
        val_ratio: Fraction for validation (rest goes to test)
    
    Returns:
        Tuple of (train_files, val_files, test_files)
    """
    random.shuffle(file_list)
    n = len(file_list)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    
    train_files = file_list[:n_train]
    val_files = file_list[n_train:n_train + n_val]
    test_files = file_list[n_train + n_val:]
    
    return train_files, val_files, test_files


def organize_splits(processed_dir: str, output_dir: str, train_ratio: float = 0.9):
    """Organize processed files into train/val/test directories.
    
    Args:
        processed_dir: Directory with processed speech files
        output_dir: Output base directory (will create train/val/test subdirs)
        train_ratio: Train split ratio
    """
    import shutil
    
    files = find_audio_files(processed_dir)
    train_files, val_files, test_files = split_files(files, train_ratio)
    
    for split_name, split_files_list in [
        ("train", train_files),
        ("val", val_files),
        ("test", test_files),
    ]:
        split_dir = os.path.join(output_dir, split_name, "clean")
        os.makedirs(split_dir, exist_ok=True)
        
        for i, src in enumerate(tqdm(split_files_list, desc=f"  {split_name}")):
            dst = os.path.join(split_dir, f"clean_{i:06d}.wav")
            shutil.copy2(src, dst)
        
        print(f"  ✓ {split_name}: {len(split_files_list)} files")


def main():
    parser = argparse.ArgumentParser(description="Prepare audio data for DeNoiseAI")
    parser.add_argument("--input_dir", type=str, required=True, help="Raw dataset directory")
    parser.add_argument("--output_dir", type=str, default="data/processed", help="Output directory")
    parser.add_argument("--sample_rate", type=int, default=16000, help="Target sample rate")
    parser.add_argument("--segment_length", type=float, default=4.0, help="Segment length (seconds)")
    parser.add_argument("--train_ratio", type=float, default=0.9, help="Train split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    print(f"\n{'='*60}")
    print(f"DeNoiseAI Data Preparation")
    print(f"{'='*60}")
    print(f"Input: {input_dir.absolute()}")
    print(f"Output: {output_dir.absolute()}")
    print(f"Sample rate: {args.sample_rate} Hz")
    print(f"Segment length: {args.segment_length}s")
    print(f"{'='*60}\n")
    
    # Find speech directories (LibriSpeech)
    speech_dirs = []
    for subdir in ["librispeech_train_clean_100", "librispeech_dev_clean", "librispeech_test_clean"]:
        d = input_dir / subdir
        if d.exists():
            speech_dirs.append(str(d))
    
    # Find noise directories (MUSAN, DEMAND)
    noise_dirs = []
    for subdir in ["musan", "demand"]:
        d = input_dir / subdir
        if d.exists():
            noise_dirs.append(str(d))
    
    # Process speech
    if speech_dirs:
        print("[1/3] Processing clean speech...")
        speech_files = []
        for d in speech_dirs:
            speech_files.extend(find_audio_files(d))
        print(f"  Found {len(speech_files)} speech files")
        
        temp_speech_dir = str(output_dir / "_temp_speech")
        process_speech_files(
            speech_files, temp_speech_dir,
            sample_rate=args.sample_rate,
            segment_length=args.segment_length,
        )
        
        print("\n[2/3] Organizing train/val/test splits...")
        organize_splits(temp_speech_dir, str(output_dir), args.train_ratio)
        
        # Cleanup temp
        import shutil
        shutil.rmtree(temp_speech_dir, ignore_errors=True)
    else:
        print("⚠ No speech data found. Please download datasets first:")
        print("  python scripts/download_data.py --output_dir data/raw --datasets librispeech")
    
    # Process noise
    if noise_dirs:
        print("\n[3/3] Processing noise...")
        noise_files = []
        for d in noise_dirs:
            noise_files.extend(find_audio_files(d))
        print(f"  Found {len(noise_files)} noise files")
        
        noise_output = str(output_dir / "noise")
        process_noise_files(
            noise_files, noise_output,
            sample_rate=args.sample_rate,
            segment_length=args.segment_length,
        )
    else:
        print("⚠ No noise data found. Please download datasets first:")
        print("  python scripts/download_data.py --output_dir data/raw --datasets noise")
    
    print(f"\n{'='*60}")
    print(f"Data preparation complete!")
    print(f"{'='*60}")
    print(f"\nNext step: Mix noisy-clean pairs:")
    print(f"  python scripts/mix_audio.py \\")
    print(f"    --clean_dir {output_dir}/train/clean \\")
    print(f"    --noise_dir {output_dir}/noise \\")
    print(f"    --output_dir {output_dir}/train/noisy")


if __name__ == "__main__":
    main()

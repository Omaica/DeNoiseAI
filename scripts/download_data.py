"""
Download datasets for speech enhancement training.

Supports downloading:
- LibriSpeech (clean speech)
- MUSAN (noise corpus)
- DEMAND (environmental noise)

Usage:
    python scripts/download_data.py --output_dir data/raw
    python scripts/download_data.py --output_dir data/raw --datasets librispeech musan
"""

import os
import sys
import argparse
import tarfile
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm


# Dataset download URLs
DATASETS = {
    "librispeech_train_clean_100": {
        "url": "https://www.openslr.org/resources/12/train-clean-100.tar.gz",
        "description": "LibriSpeech train-clean-100 (~100 hours of clean speech)",
        "format": "tar.gz",
    },
    "librispeech_dev_clean": {
        "url": "https://www.openslr.org/resources/12/dev-clean.tar.gz",
        "description": "LibriSpeech dev-clean (validation set)",
        "format": "tar.gz",
    },
    "librispeech_test_clean": {
        "url": "https://www.openslr.org/resources/12/test-clean.tar.gz",
        "description": "LibriSpeech test-clean (test set)",
        "format": "tar.gz",
    },
    "musan": {
        "url": "https://www.openslr.org/resources/17/musan.tar.gz",
        "description": "MUSAN corpus (music, speech, noise)",
        "format": "tar.gz",
    },
    "demand": {
        "url": "https://zenodo.org/record/1227121/files/DEMAND.zip",
        "description": "DEMAND multichannel environmental noise",
        "format": "zip",
    },
}


def download_file(url: str, output_path: str, description: str = ""):
    """Download a file with progress bar.
    
    Args:
        url: Download URL
        output_path: Local save path
        description: Description for progress bar
    """
    if os.path.exists(output_path):
        print(f"  ⏭ Already downloaded: {output_path}")
        return
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print(f"  ⬇ Downloading: {description or url}")
    
    response = requests.get(url, stream=True, timeout=30)
    response.raise_for_status()
    
    total_size = int(response.headers.get("content-length", 0))
    
    with open(output_path, "wb") as f:
        with tqdm(total=total_size, unit="B", unit_scale=True, desc=description) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))
    
    print(f"  ✓ Saved to: {output_path}")


def extract_archive(archive_path: str, output_dir: str):
    """Extract tar.gz or zip archive.
    
    Args:
        archive_path: Path to archive file
        output_dir: Directory to extract to
    """
    print(f"  📦 Extracting: {archive_path}")
    
    if archive_path.endswith(".tar.gz") or archive_path.endswith(".tgz"):
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(output_dir)
    elif archive_path.endswith(".zip"):
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(output_dir)
    else:
        print(f"  ⚠ Unknown archive format: {archive_path}")
        return
    
    print(f"  ✓ Extracted to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Download datasets for DeNoiseAI")
    parser.add_argument(
        "--output_dir", type=str, default="data/raw",
        help="Directory to save downloaded datasets"
    )
    parser.add_argument(
        "--datasets", nargs="+",
        choices=list(DATASETS.keys()) + ["all", "librispeech", "noise"],
        default=["all"],
        help="Datasets to download"
    )
    parser.add_argument(
        "--no_extract", action="store_true",
        help="Skip extraction after download"
    )
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    
    # Resolve dataset groups
    to_download = set()
    for name in args.datasets:
        if name == "all":
            to_download.update(DATASETS.keys())
        elif name == "librispeech":
            to_download.update([
                "librispeech_train_clean_100",
                "librispeech_dev_clean",
                "librispeech_test_clean",
            ])
        elif name == "noise":
            to_download.update(["musan", "demand"])
        else:
            to_download.add(name)
    
    print(f"\n{'='*60}")
    print(f"DeNoiseAI Dataset Downloader")
    print(f"{'='*60}")
    print(f"Output directory: {output_dir.absolute()}")
    print(f"Datasets to download: {', '.join(sorted(to_download))}")
    print(f"{'='*60}\n")
    
    for name in sorted(to_download):
        info = DATASETS[name]
        print(f"\n[{name}] {info['description']}")
        
        # Download
        ext = ".tar.gz" if info["format"] == "tar.gz" else ".zip"
        archive_path = str(output_dir / f"{name}{ext}")
        
        try:
            download_file(info["url"], archive_path, name)
        except Exception as e:
            print(f"  ✗ Download failed: {e}")
            continue
        
        # Extract
        if not args.no_extract:
            try:
                extract_archive(archive_path, str(output_dir / name))
            except Exception as e:
                print(f"  ✗ Extraction failed: {e}")
    
    print(f"\n{'='*60}")
    print(f"Download complete!")
    print(f"{'='*60}")
    print(f"\nNext step: Run data preparation:")
    print(f"  python scripts/prepare_data.py --input_dir {output_dir} --output_dir data/processed")


if __name__ == "__main__":
    main()

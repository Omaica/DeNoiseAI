"""
Inference script for denoising audio files.

Usage:
    # Denoise a single file
    python scripts/inference.py \
        --model_path checkpoints/best_model.pth \
        --input noisy_audio.wav \
        --output denoised_audio.wav

    # Denoise a directory
    python scripts/inference.py \
        --model_path checkpoints/best_model.pth \
        --input noisy_dir/ \
        --output output_dir/

    # With spectrogram visualization
    python scripts/inference.py \
        --model_path checkpoints/best_model.pth \
        --input noisy_audio.wav \
        --output denoised_audio.wav \
        --visualize
"""

import os
import sys
import argparse
from pathlib import Path

import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.dccrn import build_dccrn
from src.utils.audio_utils import load_audio, save_audio


def load_model(checkpoint_path: str, device: torch.device):
    """Load trained DCCRN model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Torch device
    
    Returns:
        Loaded model in eval mode
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get("config", {})
    
    model = build_dccrn(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    epoch = checkpoint.get("epoch", "?")
    print(f"  [OK] Loaded model from epoch {epoch}")
    print(f"  Parameters: {model.get_num_params():,}")
    
    return model, config


@torch.no_grad()
def denoise_audio(
    model,
    waveform: torch.Tensor,
    device: torch.device,
    segment_length: int = 64000,
    overlap: int = 16000,
) -> torch.Tensor:
    """Denoise an audio waveform using overlap-add.
    
    For long audio files, process in overlapping segments
    and blend them using a Hann window for smooth transitions.
    
    Args:
        model: Trained DCCRN model
        waveform: Input noisy waveform (T,)
        device: Torch device
        segment_length: Processing segment length in samples
        overlap: Overlap between segments in samples
    
    Returns:
        Enhanced waveform (T,)
    """
    total_length = waveform.shape[0]
    
    # If audio is shorter than segment, process directly
    if total_length <= segment_length:
        # Pad to segment length
        padded = torch.zeros(segment_length)
        padded[:total_length] = waveform
        enhanced = model(padded.unsqueeze(0).to(device))
        return enhanced.squeeze(0).cpu()[:total_length]
    
    # Overlap-add processing for long audio
    step = segment_length - overlap
    output = torch.zeros(total_length)
    weight = torch.zeros(total_length)
    
    # Hann window for blending
    window = torch.hann_window(segment_length)
    
    for start in range(0, total_length - segment_length + 1, step):
        segment = waveform[start:start + segment_length]
        
        enhanced_segment = model(segment.unsqueeze(0).to(device))
        enhanced_segment = enhanced_segment.squeeze(0).cpu()
        
        # Trim if model output is different length
        out_len = min(enhanced_segment.shape[0], segment_length)
        
        output[start:start + out_len] += enhanced_segment[:out_len] * window[:out_len]
        weight[start:start + out_len] += window[:out_len]
    
    # Handle last segment
    if total_length % step != 0:
        start = total_length - segment_length
        segment = waveform[start:start + segment_length]
        enhanced_segment = model(segment.unsqueeze(0).to(device))
        enhanced_segment = enhanced_segment.squeeze(0).cpu()
        
        out_len = min(enhanced_segment.shape[0], segment_length)
        output[start:start + out_len] += enhanced_segment[:out_len] * window[:out_len]
        weight[start:start + out_len] += window[:out_len]
    
    # Normalize by overlap weight
    weight = torch.clamp(weight, min=1e-8)
    output = output / weight
    
    return output


def visualize_spectrogram(
    noisy: torch.Tensor,
    enhanced: torch.Tensor,
    clean: torch.Tensor = None,
    sample_rate: int = 16000,
    save_path: str = None,
):
    """Plot spectrograms for comparison.
    
    Args:
        noisy: Noisy waveform
        enhanced: Enhanced waveform
        clean: Optional clean reference
        sample_rate: Sample rate
        save_path: Optional path to save figure
    """
    n_plots = 3 if clean is not None else 2
    fig, axes = plt.subplots(n_plots, 1, figsize=(12, 3 * n_plots))
    
    def plot_spec(ax, waveform, title):
        spec = torch.stft(
            waveform, n_fft=512, hop_length=256,
            window=torch.hann_window(512), return_complex=True
        )
        magnitude = spec.abs().numpy()
        ax.imshow(
            20 * np.log10(magnitude + 1e-8),
            aspect="auto", origin="lower",
            cmap="magma",
        )
        ax.set_title(title, fontsize=14)
        ax.set_ylabel("Frequency Bin")
    
    plot_spec(axes[0], noisy, "Noisy Input")
    plot_spec(axes[1], enhanced, "Enhanced Output")
    
    if clean is not None:
        plot_spec(axes[2], clean, "Clean Reference")
    
    axes[-1].set_xlabel("Time Frame")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  ✓ Saved visualization: {save_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Denoise audio files using trained DCCRN")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--input", type=str, required=True, help="Input audio file or directory")
    parser.add_argument("--output", type=str, required=True, help="Output file or directory")
    parser.add_argument("--clean", type=str, default=None, help="Optional clean reference for comparison")
    parser.add_argument("--visualize", action="store_true", help="Generate spectrogram visualization")
    parser.add_argument("--device", type=str, default="auto", help="Device: auto, cpu, or cuda")
    args = parser.parse_args()
    
    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"\n{'='*60}")
    print(f"DeNoiseAI Inference")
    print(f"{'='*60}")
    print(f"Device: {device}")
    
    # Load model
    model, config = load_model(args.model_path, device)
    sample_rate = config.get("audio", {}).get("sample_rate", 16000)
    segment_length = config.get("audio", {}).get("n_samples", 64000)
    
    # Process files
    input_path = Path(args.input)
    
    if input_path.is_file():
        # Single file
        files = [(str(input_path), args.output)]
    elif input_path.is_dir():
        # Directory
        os.makedirs(args.output, exist_ok=True)
        audio_exts = {".wav", ".flac", ".mp3", ".ogg"}
        files = [
            (str(f), os.path.join(args.output, f.name))
            for f in input_path.rglob("*")
            if f.suffix.lower() in audio_exts
        ]
    else:
        print(f"✗ Input not found: {args.input}")
        sys.exit(1)
    
    print(f"Processing {len(files)} file(s)...\n")
    
    for input_file, output_file in tqdm(files, desc="Denoising"):
        # Load noisy audio
        noisy = load_audio(input_file, sample_rate)
        
        # Denoise
        enhanced = denoise_audio(model, noisy, device, segment_length)
        
        # Save
        save_audio(output_file, enhanced, sample_rate)
        
        # Visualize
        if args.visualize:
            clean = None
            if args.clean:
                clean = load_audio(args.clean, sample_rate)
            
            viz_path = output_file.rsplit(".", 1)[0] + "_spectrogram.png"
            visualize_spectrogram(noisy, enhanced, clean, sample_rate, viz_path)
    
    print(f"\n{'='*60}")
    print(f"Inference complete!")
    print(f"  ✓ Output: {args.output}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

"""
Audio processing utilities.

Includes functions for loading, saving, resampling, normalizing,
and mixing audio files.
"""

import os
from pathlib import Path

import numpy as np
import torch
import torchaudio
import soundfile as sf


def load_audio(filepath: str, sample_rate: int = 16000) -> torch.Tensor:
    """Load audio file and resample to target sample rate.
    
    Args:
        filepath: Path to audio file
        sample_rate: Target sample rate
    
    Returns:
        1D tensor of shape (T,)
    """
    waveform, sr = torchaudio.load(filepath)
    
    # Convert to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    
    # Resample if needed
    if sr != sample_rate:
        resampler = torchaudio.transforms.Resample(sr, sample_rate)
        waveform = resampler(waveform)
    
    return waveform.squeeze(0)


def save_audio(filepath: str, waveform: torch.Tensor, sample_rate: int = 16000):
    """Save audio tensor to file.
    
    Args:
        filepath: Output file path
        waveform: Audio tensor, shape (T,) or (1, T)
        sample_rate: Sample rate
    """
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    torchaudio.save(filepath, waveform.cpu(), sample_rate)


def normalize_audio(waveform: torch.Tensor, target_db: float = -3.0) -> torch.Tensor:
    """Normalize audio to target peak dB level.
    
    Args:
        waveform: Audio tensor
        target_db: Target peak level in dB (default: -3 dB)
    
    Returns:
        Normalized audio tensor
    """
    peak = waveform.abs().max()
    if peak > 0:
        target_linear = 10 ** (target_db / 20)
        waveform = waveform * (target_linear / peak)
    return waveform


def mix_audio_at_snr(
    clean: torch.Tensor,
    noise: torch.Tensor,
    snr_db: float,
) -> tuple:
    """Mix clean speech with noise at specified SNR.
    
    Args:
        clean: Clean speech signal (T,)
        noise: Noise signal (T_noise,)
        snr_db: Target SNR in dB
    
    Returns:
        Tuple of (noisy_signal, clean_signal) — both normalized
    """
    # Match noise length to clean
    clean_len = clean.shape[0]
    
    if noise.shape[0] < clean_len:
        repeats = clean_len // noise.shape[0] + 1
        noise = noise.repeat(repeats)
    
    # Random start point in noise
    if noise.shape[0] > clean_len:
        start = np.random.randint(0, noise.shape[0] - clean_len)
        noise = noise[start:start + clean_len]
    else:
        noise = noise[:clean_len]
    
    # Calculate energy and scale
    clean_power = (clean ** 2).mean() + 1e-8
    noise_power = (noise ** 2).mean() + 1e-8
    snr_linear = 10 ** (snr_db / 10)
    scale = torch.sqrt(clean_power / (noise_power * snr_linear))
    
    noisy = clean + scale * noise
    
    # Prevent clipping
    max_val = max(noisy.abs().max().item(), 1.0)
    noisy = noisy / max_val
    clean = clean / max_val
    
    return noisy, clean


def segment_audio(waveform: torch.Tensor, segment_length: int, overlap: int = 0) -> list:
    """Split audio into fixed-length segments.
    
    Args:
        waveform: Audio tensor (T,)
        segment_length: Length of each segment in samples
        overlap: Overlap between segments in samples
    
    Returns:
        List of audio segments
    """
    segments = []
    step = segment_length - overlap
    
    for start in range(0, len(waveform) - segment_length + 1, step):
        segments.append(waveform[start:start + segment_length])
    
    # Handle last segment (pad if needed)
    if len(waveform) % step != 0:
        last_segment = waveform[-segment_length:]
        if len(last_segment) < segment_length:
            last_segment = torch.nn.functional.pad(
                last_segment, (0, segment_length - len(last_segment))
            )
        segments.append(last_segment)
    
    return segments


def compute_snr(clean: torch.Tensor, noisy: torch.Tensor) -> float:
    """Compute SNR between clean and noisy signals.
    
    Args:
        clean: Clean signal
        noisy: Noisy signal
    
    Returns:
        SNR in dB
    """
    noise = noisy - clean
    clean_power = (clean ** 2).mean()
    noise_power = (noise ** 2).mean() + 1e-8
    snr = 10 * torch.log10(clean_power / noise_power)
    return snr.item()

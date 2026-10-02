"""
Audio Dataset for speech enhancement training.

Supports two modes:
1. Pre-mixed mode: Load pre-generated noisy-clean pairs from disk
2. Dynamic mixing mode: Load clean + noise separately and mix on-the-fly
"""

import os
import random
from pathlib import Path

import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader


class AudioDataset(Dataset):
    """Dataset for loading noisy-clean audio pairs.
    
    Args:
        clean_dir: Directory containing clean audio files
        noisy_dir: Directory containing noisy audio files (pre-mixed mode)
        noise_dir: Directory containing noise files (dynamic mixing mode)
        sample_rate: Target sample rate
        segment_length: Length of audio segments in samples
        dynamic_mixing: If True, mix clean + noise on-the-fly
        snr_range: SNR range for dynamic mixing [min_snr, max_snr] in dB
    """

    def __init__(
        self,
        clean_dir: str,
        noisy_dir: str = None,
        noise_dir: str = None,
        sample_rate: int = 16000,
        segment_length: int = 64000,
        dynamic_mixing: bool = False,
        snr_range: list = None,
    ):
        self.clean_dir = Path(clean_dir)
        self.sample_rate = sample_rate
        self.segment_length = segment_length
        self.dynamic_mixing = dynamic_mixing
        self.snr_range = snr_range or [-5, 20]
        
        # Supported audio extensions
        audio_exts = {".wav", ".flac", ".mp3", ".ogg"}
        
        # Get list of clean audio files
        self.clean_files = sorted([
            f for f in self.clean_dir.rglob("*")
            if f.suffix.lower() in audio_exts
        ])
        
        if len(self.clean_files) == 0:
            raise ValueError(f"No audio files found in {clean_dir}")
        
        if dynamic_mixing:
            if noise_dir is None:
                raise ValueError("noise_dir is required for dynamic mixing mode")
            self.noise_dir = Path(noise_dir)
            self.noise_files = sorted([
                f for f in self.noise_dir.rglob("*")
                if f.suffix.lower() in audio_exts
            ])
            if len(self.noise_files) == 0:
                raise ValueError(f"No noise files found in {noise_dir}")
        else:
            if noisy_dir is None:
                raise ValueError("noisy_dir is required for pre-mixed mode")
            self.noisy_dir = Path(noisy_dir)
            self.noisy_files = sorted([
                f for f in self.noisy_dir.rglob("*")
                if f.suffix.lower() in audio_exts
            ])
            if len(self.clean_files) != len(self.noisy_files):
                raise ValueError(
                    f"Number of clean ({len(self.clean_files)}) and noisy "
                    f"({len(self.noisy_files)}) files must match in pre-mixed mode"
                )

    def __len__(self):
        return len(self.clean_files)

    def _load_audio(self, filepath: Path) -> torch.Tensor:
        """Load and resample audio file to target sample rate.
        
        Returns:
            1D tensor of shape (T,)
        """
        waveform, sr = torchaudio.load(str(filepath))
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Resample if needed
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
        
        return waveform.squeeze(0)  # (T,)

    def _pad_or_trim(self, audio: torch.Tensor) -> torch.Tensor:
        """Pad with zeros or randomly trim audio to segment_length."""
        length = audio.shape[0]
        
        if length >= self.segment_length:
            # Random crop
            start = random.randint(0, length - self.segment_length)
            return audio[start:start + self.segment_length]
        else:
            # Pad with zeros
            pad_length = self.segment_length - length
            return torch.nn.functional.pad(audio, (0, pad_length))

    def _mix_at_snr(self, clean: torch.Tensor, noise: torch.Tensor, snr_db: float) -> torch.Tensor:
        """Mix clean signal with noise at a specific SNR level.
        
        Args:
            clean: Clean speech signal
            noise: Noise signal (will be repeated/trimmed to match clean length)
            snr_db: Target signal-to-noise ratio in dB
        
        Returns:
            Noisy signal = clean + scaled_noise
        """
        # Ensure noise is same length as clean
        if noise.shape[0] < clean.shape[0]:
            # Repeat noise
            repeats = clean.shape[0] // noise.shape[0] + 1
            noise = noise.repeat(repeats)
        noise = noise[:clean.shape[0]]
        
        # Calculate scaling factor for target SNR
        clean_power = (clean ** 2).mean() + 1e-8
        noise_power = (noise ** 2).mean() + 1e-8
        snr_linear = 10 ** (snr_db / 10)
        scale = torch.sqrt(clean_power / (noise_power * snr_linear))
        
        noisy = clean + scale * noise
        
        # Normalize to prevent clipping
        max_val = max(noisy.abs().max(), 1.0)
        noisy = noisy / max_val
        clean_normalized = clean / max_val
        
        return noisy, clean_normalized

    def __getitem__(self, idx):
        """
        Returns:
            dict with keys:
                - 'noisy': noisy waveform (T,)
                - 'clean': clean waveform (T,)
        """
        # Load clean audio
        clean = self._load_audio(self.clean_files[idx])
        
        if self.dynamic_mixing:
            # Dynamic mixing: pick random noise and SNR
            noise_idx = random.randint(0, len(self.noise_files) - 1)
            noise = self._load_audio(self.noise_files[noise_idx])
            snr_db = random.uniform(self.snr_range[0], self.snr_range[1])
            
            # Pad/trim clean first
            clean = self._pad_or_trim(clean)
            noise = self._pad_or_trim(noise)
            
            # Mix
            noisy, clean = self._mix_at_snr(clean, noise, snr_db)
        else:
            # Pre-mixed mode
            noisy = self._load_audio(self.noisy_files[idx])
            clean = self._pad_or_trim(clean)
            noisy = self._pad_or_trim(noisy)
        
        return {"noisy": noisy, "clean": clean}


def create_dataloader(
    clean_dir: str,
    noisy_dir: str = None,
    noise_dir: str = None,
    batch_size: int = 16,
    sample_rate: int = 16000,
    segment_length: int = 64000,
    dynamic_mixing: bool = False,
    snr_range: list = None,
    num_workers: int = 4,
    pin_memory: bool = True,
    shuffle: bool = True,
) -> DataLoader:
    """Create a DataLoader for audio dataset.
    
    Args:
        clean_dir: Path to clean audio directory
        noisy_dir: Path to noisy audio directory (pre-mixed mode)
        noise_dir: Path to noise directory (dynamic mixing mode)
        batch_size: Batch size
        sample_rate: Target sample rate
        segment_length: Segment length in samples
        dynamic_mixing: Use dynamic mixing
        snr_range: SNR range for mixing
        num_workers: Number of data loading workers
        pin_memory: Pin memory for faster GPU transfer
        shuffle: Shuffle data
    
    Returns:
        DataLoader instance
    """
    dataset = AudioDataset(
        clean_dir=clean_dir,
        noisy_dir=noisy_dir,
        noise_dir=noise_dir,
        sample_rate=sample_rate,
        segment_length=segment_length,
        dynamic_mixing=dynamic_mixing,
        snr_range=snr_range,
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    
    return dataloader

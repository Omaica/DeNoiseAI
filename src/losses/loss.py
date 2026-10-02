"""
Loss functions for speech enhancement training.

Includes:
- SI-SNR (Scale-Invariant Signal-to-Noise Ratio)
- Magnitude Spectrum Loss
- Complex Spectrum Loss
- Multi-Resolution STFT Loss
- Combined Loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SISnrLoss(nn.Module):
    """Scale-Invariant Signal-to-Noise Ratio loss.
    
    Measures the quality of signal reconstruction independent of
    global scaling. Higher SI-SNR = better quality.
    We return the negative so it can be minimized.
    """

    def __init__(self):
        super().__init__()

    def forward(self, estimated: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            estimated: Enhanced signal, shape (B, T)
            target: Clean reference signal, shape (B, T)
        
        Returns:
            Negative SI-SNR loss (scalar)
        """
        # Ensure same length
        min_len = min(estimated.shape[-1], target.shape[-1])
        estimated = estimated[..., :min_len]
        target = target[..., :min_len]
        
        # Zero-mean normalization
        target = target - target.mean(dim=-1, keepdim=True)
        estimated = estimated - estimated.mean(dim=-1, keepdim=True)
        
        # <s_target, s_estimate> / ||s_target||^2 * s_target
        dot = torch.sum(estimated * target, dim=-1, keepdim=True)
        s_target_energy = torch.sum(target ** 2, dim=-1, keepdim=True) + 1e-8
        proj = dot * target / s_target_energy
        
        # Noise = estimated - projection
        noise = estimated - proj
        
        # SI-SNR = 10 * log10(||proj||^2 / ||noise||^2)
        si_snr = 10 * torch.log10(
            torch.sum(proj ** 2, dim=-1) / (torch.sum(noise ** 2, dim=-1) + 1e-8) + 1e-8
        )
        
        # Return negative mean (minimize loss = maximize SI-SNR)
        return -si_snr.mean()


class MagnitudeLoss(nn.Module):
    """Loss on magnitude spectrogram.
    
    Computes L1 or L2 loss between magnitude spectrograms of
    estimated and target signals.
    """

    def __init__(self, n_fft: int = 512, hop_length: int = 256, loss_type: str = "l1"):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.loss_type = loss_type
        self.register_buffer("window", torch.hann_window(n_fft))

    def forward(self, estimated: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        min_len = min(estimated.shape[-1], target.shape[-1])
        estimated = estimated[..., :min_len]
        target = target[..., :min_len]
        
        est_spec = torch.stft(
            estimated, self.n_fft, self.hop_length,
            window=self.window, return_complex=True
        )
        tgt_spec = torch.stft(
            target, self.n_fft, self.hop_length,
            window=self.window, return_complex=True
        )
        
        est_mag = est_spec.abs()
        tgt_mag = tgt_spec.abs()
        
        if self.loss_type == "l1":
            return F.l1_loss(est_mag, tgt_mag)
        else:
            return F.mse_loss(est_mag, tgt_mag)


class ComplexSpecLoss(nn.Module):
    """Loss on complex spectrogram (real + imaginary).
    
    Preserves phase information by computing loss on both
    real and imaginary components.
    """

    def __init__(self, n_fft: int = 512, hop_length: int = 256):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.register_buffer("window", torch.hann_window(n_fft))

    def forward(self, estimated: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        min_len = min(estimated.shape[-1], target.shape[-1])
        estimated = estimated[..., :min_len]
        target = target[..., :min_len]
        
        est_spec = torch.stft(
            estimated, self.n_fft, self.hop_length,
            window=self.window, return_complex=True
        )
        tgt_spec = torch.stft(
            target, self.n_fft, self.hop_length,
            window=self.window, return_complex=True
        )
        
        # Loss on real and imaginary parts
        real_loss = F.mse_loss(est_spec.real, tgt_spec.real)
        imag_loss = F.mse_loss(est_spec.imag, tgt_spec.imag)
        
        return real_loss + imag_loss


class MultiResolutionSTFTLoss(nn.Module):
    """Multi-resolution STFT loss.
    
    Computes spectral convergence + log magnitude loss at multiple
    STFT resolutions to capture both fine and coarse spectral details.
    """

    def __init__(self, fft_sizes: list = None, hop_sizes: list = None, win_sizes: list = None):
        super().__init__()
        if fft_sizes is None:
            fft_sizes = [512, 1024, 2048]
        if hop_sizes is None:
            hop_sizes = [128, 256, 512]
        if win_sizes is None:
            win_sizes = [512, 1024, 2048]
        
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_sizes = win_sizes

    def _stft_loss(self, estimated, target, n_fft, hop_length, win_length):
        window = torch.hann_window(win_length, device=estimated.device)
        
        est_spec = torch.stft(
            estimated, n_fft, hop_length, win_length,
            window=window, return_complex=True
        )
        tgt_spec = torch.stft(
            target, n_fft, hop_length, win_length,
            window=window, return_complex=True
        )
        
        est_mag = est_spec.abs()
        tgt_mag = tgt_spec.abs()
        
        # Spectral convergence
        sc_loss = torch.norm(tgt_mag - est_mag, p="fro") / (torch.norm(tgt_mag, p="fro") + 1e-8)
        
        # Log magnitude loss
        log_loss = F.l1_loss(torch.log(est_mag + 1e-8), torch.log(tgt_mag + 1e-8))
        
        return sc_loss + log_loss

    def forward(self, estimated: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        min_len = min(estimated.shape[-1], target.shape[-1])
        estimated = estimated[..., :min_len]
        target = target[..., :min_len]
        
        total_loss = 0.0
        for fft_size, hop_size, win_size in zip(self.fft_sizes, self.hop_sizes, self.win_sizes):
            total_loss += self._stft_loss(estimated, target, fft_size, hop_size, win_size)
        
        return total_loss / len(self.fft_sizes)


class CombinedLoss(nn.Module):
    """Combined loss function for DCCRN training.
    
    Weighted sum of SI-SNR, Magnitude, and Complex spectrum losses.
    
    Args:
        si_snr_weight: Weight for SI-SNR loss
        magnitude_weight: Weight for magnitude spectrum loss
        complex_weight: Weight for complex spectrum loss
        n_fft: FFT size for spectral losses
        hop_length: Hop size for spectral losses
    """

    def __init__(
        self,
        si_snr_weight: float = 0.5,
        magnitude_weight: float = 0.3,
        complex_weight: float = 0.2,
        n_fft: int = 512,
        hop_length: int = 256,
    ):
        super().__init__()
        self.si_snr_weight = si_snr_weight
        self.magnitude_weight = magnitude_weight
        self.complex_weight = complex_weight
        
        self.si_snr_loss = SISnrLoss()
        self.magnitude_loss = MagnitudeLoss(n_fft, hop_length)
        self.complex_loss = ComplexSpecLoss(n_fft, hop_length)

    def forward(self, estimated: torch.Tensor, target: torch.Tensor) -> dict:
        """
        Returns:
            Dictionary with 'total' loss and individual components.
        """
        si_snr = self.si_snr_loss(estimated, target)
        mag = self.magnitude_loss(estimated, target)
        comp = self.complex_loss(estimated, target)
        
        total = (
            self.si_snr_weight * si_snr
            + self.magnitude_weight * mag
            + self.complex_weight * comp
        )
        
        return {
            "total": total,
            "si_snr": si_snr.item(),
            "magnitude": mag.item(),
            "complex": comp.item(),
        }

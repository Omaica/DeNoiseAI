"""
Evaluation metrics for speech enhancement.

Implements PESQ, STOI, and SI-SNR metrics for evaluating
the quality of enhanced speech signals.
"""

import torch
import numpy as np

try:
    from pesq import pesq as compute_pesq
    HAS_PESQ = True
except ImportError:
    HAS_PESQ = False
    print("Warning: pesq not installed. Install with: pip install pesq")

try:
    from pystoi import stoi as compute_stoi
    HAS_STOI = True
except ImportError:
    HAS_STOI = False
    print("Warning: pystoi not installed. Install with: pip install pystoi")


def si_snr(estimated: torch.Tensor, target: torch.Tensor) -> float:
    """Compute Scale-Invariant Signal-to-Noise Ratio.
    
    Args:
        estimated: Enhanced signal, shape (T,)
        target: Clean reference signal, shape (T,)
    
    Returns:
        SI-SNR in dB (higher is better)
    """
    min_len = min(len(estimated), len(target))
    estimated = estimated[:min_len]
    target = target[:min_len]
    
    # Zero-mean
    target = target - target.mean()
    estimated = estimated - estimated.mean()
    
    # Projection
    dot = torch.sum(estimated * target)
    s_target_energy = torch.sum(target ** 2) + 1e-8
    proj = (dot / s_target_energy) * target
    
    noise = estimated - proj
    
    si_snr_val = 10 * torch.log10(
        torch.sum(proj ** 2) / (torch.sum(noise ** 2) + 1e-8) + 1e-8
    )
    
    return si_snr_val.item()


def pesq_score(
    estimated: np.ndarray,
    target: np.ndarray,
    sample_rate: int = 16000,
    mode: str = "wb",
) -> float:
    """Compute PESQ (Perceptual Evaluation of Speech Quality).
    
    Args:
        estimated: Enhanced signal as numpy array
        target: Clean reference signal as numpy array
        sample_rate: Sample rate (8000 or 16000)
        mode: "wb" (wideband) or "nb" (narrowband)
    
    Returns:
        PESQ score (-0.5 to 4.5, higher is better)
    """
    if not HAS_PESQ:
        raise RuntimeError("pesq package not installed")
    
    min_len = min(len(estimated), len(target))
    estimated = estimated[:min_len]
    target = target[:min_len]
    
    return compute_pesq(sample_rate, target, estimated, mode)


def stoi_score(
    estimated: np.ndarray,
    target: np.ndarray,
    sample_rate: int = 16000,
    extended: bool = False,
) -> float:
    """Compute STOI (Short-Time Objective Intelligibility).
    
    Args:
        estimated: Enhanced signal as numpy array
        target: Clean reference signal as numpy array
        sample_rate: Sample rate
        extended: Use extended STOI
    
    Returns:
        STOI score (0 to 1, higher is better)
    """
    if not HAS_STOI:
        raise RuntimeError("pystoi package not installed")
    
    min_len = min(len(estimated), len(target))
    estimated = estimated[:min_len]
    target = target[:min_len]
    
    return compute_stoi(target, estimated, sample_rate, extended=extended)


def evaluate_batch(
    estimated: torch.Tensor,
    target: torch.Tensor,
    sample_rate: int = 16000,
) -> dict:
    """Evaluate a batch of enhanced audio against clean references.
    
    Args:
        estimated: Batch of enhanced signals, shape (B, T)
        target: Batch of clean signals, shape (B, T)
        sample_rate: Sample rate
    
    Returns:
        Dictionary with mean metrics: si_snr, pesq, stoi
    """
    batch_size = estimated.shape[0]
    
    metrics = {"si_snr": [], "pesq": [], "stoi": []}
    
    for i in range(batch_size):
        est = estimated[i].detach().cpu()
        tgt = target[i].detach().cpu()
        
        # SI-SNR
        metrics["si_snr"].append(si_snr(est, tgt))
        
        # PESQ
        if HAS_PESQ:
            try:
                score = pesq_score(est.numpy(), tgt.numpy(), sample_rate)
                metrics["pesq"].append(score)
            except Exception:
                pass
        
        # STOI
        if HAS_STOI:
            try:
                score = stoi_score(est.numpy(), tgt.numpy(), sample_rate)
                metrics["stoi"].append(score)
            except Exception:
                pass
    
    # Compute means
    result = {}
    for key, values in metrics.items():
        if values:
            result[key] = np.mean(values)
        else:
            result[key] = None
    
    return result

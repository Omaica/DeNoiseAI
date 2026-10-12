"""
DCCRN — Deep Complex Convolution Recurrent Network for Speech Enhancement.

Reference: Hu et al., "DCCRN: Deep Complex Convolution Recurrent Network for 
Phase-Aware Speech Enhancement", Interspeech 2020.

This implementation processes audio in the complex STFT domain, using complex-valued
convolutions and LSTM to jointly estimate magnitude and phase corrections.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .complex_layers import (
    ComplexEncoderBlock,
    ComplexDecoderBlock,
    ComplexLSTM,
)


class DCCRN(nn.Module):
    """Deep Complex Convolution Recurrent Network.
    
    Architecture:
        1. STFT → Complex spectrogram (real + imaginary)
        2. Complex Encoder (6 Conv blocks)
        3. Complex LSTM bottleneck
        4. Complex Decoder (6 ConvTranspose blocks) with skip connections
        5. Complex Ratio Mask → Enhanced spectrogram
        6. iSTFT → Enhanced waveform
    
    Args:
        n_fft: FFT size for STFT
        hop_length: Hop size for STFT
        win_length: Window length for STFT
        encoder_channels: List of channel sizes for encoder blocks
        kernel_size: Kernel size for convolutions
        stride: Stride for convolutions
        lstm_hidden: Hidden size for Complex LSTM
        lstm_layers: Number of LSTM layers
        masking_mode: "E" for complex ratio mask, "C" for complex mapping
    """

    def __init__(
        self,
        n_fft: int = 512,
        hop_length: int = 256,
        win_length: int = 512,
        encoder_channels: list = None,
        kernel_size: tuple = (5, 3),
        stride: tuple = (2, 1),
        lstm_hidden: int = 512,
        lstm_layers: int = 2,
        bidirectional: bool = True,
        use_attention: bool = True,
        masking_mode: str = "E",
    ):
        super().__init__()
        
        if encoder_channels is None:
            encoder_channels = [16, 32, 64, 128, 256, 256]
        
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.masking_mode = masking_mode
        self.kernel_size = kernel_size
        self.stride = stride
        self.bidirectional = bidirectional
        self.use_attention = use_attention
        
        # Register STFT window as buffer (not a parameter)
        self.register_buffer("window", torch.hann_window(win_length))
        
        # Number of frequency bins (only positive frequencies)
        self.n_freqs = n_fft // 2 + 1  # 257 for n_fft=512
        
        # --- Encoder ---
        # Input has 1 channel (single-channel audio)
        in_channels = [1] + encoder_channels[:-1]
        self.encoder = nn.ModuleList()
        pad_f = kernel_size[0] // 2
        pad_t = kernel_size[1] // 2
        for i in range(len(encoder_channels)):
            self.encoder.append(
                ComplexEncoderBlock(
                    in_channels=in_channels[i],
                    out_channels=encoder_channels[i],
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=(pad_f, pad_t),
                    use_attention=use_attention,
                )
            )
        
        # --- LSTM Bottleneck ---
        # Calculate the frequency dimension after encoder
        # Each encoder layer reduces freq by stride[0]=2 with pad_f
        freq_after_encoder = self.n_freqs
        for _ in range(len(encoder_channels)):
            freq_after_encoder = (freq_after_encoder + 2 * pad_f - kernel_size[0]) // stride[0] + 1
        
        lstm_input_size = encoder_channels[-1] * freq_after_encoder
        self.lstm = ComplexLSTM(
            input_size=lstm_input_size,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            bidirectional=bidirectional,
        )
        
        # --- Decoder ---
        # Decoder mirrors encoder with skip connections at every stage
        num_layers = len(encoder_channels)
        self.decoder = nn.ModuleList()
        prev_ch = encoder_channels[-1]
        for i in range(num_layers):
            skip_ch = encoder_channels[num_layers - 1 - i]
            in_ch = prev_ch + skip_ch
            is_last = (i == num_layers - 1)
            out_ch = encoder_channels[num_layers - 2 - i] if not is_last else 1
            prev_ch = out_ch
            
            self.decoder.append(
                ComplexDecoderBlock(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=(pad_f, pad_t),
                    output_padding=(0, 0),
                    is_last=is_last,
                    use_attention=use_attention if not is_last else False,
                )
            )

    def stft(self, x: torch.Tensor):
        """Compute STFT and return real and imaginary parts.
        
        Args:
            x: Waveform, shape (B, T)
        
        Returns:
            real: (B, 1, F, T'), imaginary: (B, 1, F, T')
        """
        # Compute complex STFT
        spec = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            return_complex=True,
        )
        # spec shape: (B, F, T')
        real = spec.real.unsqueeze(1)  # (B, 1, F, T')
        imag = spec.imag.unsqueeze(1)  # (B, 1, F, T')
        return real, imag

    def istft(self, real: torch.Tensor, imag: torch.Tensor):
        """Compute inverse STFT from real and imaginary parts.
        
        Args:
            real: (B, 1, F, T')
            imag: (B, 1, F, T')
        
        Returns:
            Waveform, shape (B, T)
        """
        # Remove channel dim
        real = real.squeeze(1)  # (B, F, T')
        imag = imag.squeeze(1)
        
        # Create complex spectrogram
        spec = torch.complex(real, imag)
        
        # Inverse STFT
        waveform = torch.istft(
            spec,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
        )
        return waveform

    def forward(self, noisy_waveform: torch.Tensor):
        """
        Args:
            noisy_waveform: (B, T) noisy audio waveform
        
        Returns:
            enhanced_waveform: (B, T) denoised audio waveform
        """
        # --- STFT ---
        real, imag = self.stft(noisy_waveform)
        
        # Store input for masking
        input_real, input_imag = real, imag
        
        # --- Encoder ---
        encoder_outputs = []
        for encoder_block in self.encoder:
            real, imag = encoder_block(real, imag)
            encoder_outputs.append((real, imag))
        
        # --- Reshape for LSTM ---
        # (B, C, F, T') → (B, T', C*F)
        batch_size, channels, freq, time = real.shape
        real_lstm = real.permute(0, 3, 1, 2).contiguous().view(batch_size, time, -1)
        imag_lstm = imag.permute(0, 3, 1, 2).contiguous().view(batch_size, time, -1)
        
        # --- LSTM ---
        real_lstm, imag_lstm = self.lstm(real_lstm, imag_lstm)
        
        # --- Reshape back ---
        real = real_lstm.view(batch_size, time, channels, freq).permute(0, 2, 3, 1)
        imag = imag_lstm.view(batch_size, time, channels, freq).permute(0, 2, 3, 1)
        
        # --- Decoder with skip connections ---
        for i, decoder_block in enumerate(self.decoder):
            # Get corresponding encoder output for skip connection
            encoder_idx = len(self.encoder) - 1 - i
            skip_real, skip_imag = encoder_outputs[encoder_idx]
            
            # Dimension alignment check (exact match guaranteed with (5, 3) kernel)
            if real.shape[-2:] != skip_real.shape[-2:]:
                min_f = min(real.shape[2], skip_real.shape[2])
                min_t = min(real.shape[3], skip_real.shape[3])
                real = real[:, :, :min_f, :min_t]
                imag = imag[:, :, :min_f, :min_t]
                skip_real = skip_real[:, :, :min_f, :min_t]
                skip_imag = skip_imag[:, :, :min_f, :min_t]
            
            real = torch.cat([real, skip_real], dim=1)
            imag = torch.cat([imag, skip_imag], dim=1)
            real, imag = decoder_block(real, imag)
        
        # --- Apply mask ---
        # Ensure mask matches input STFT shape exactly
        if real.shape[-2:] != input_real.shape[-2:]:
            min_f = min(real.shape[2], input_real.shape[2])
            min_t = min(real.shape[3], input_real.shape[3])
            real = real[:, :, :min_f, :min_t]
            imag = imag[:, :, :min_f, :min_t]
            input_real = input_real[:, :, :min_f, :min_t]
            input_imag = input_imag[:, :, :min_f, :min_t]
        
        if self.masking_mode == "E":
            # Complex Ratio Mask (cRM) with Bounded Magnitude & Full Phase Range
            # Hu et al., Interspeech 2020 (DCCRN standard)
            mask_mag = torch.sqrt(real ** 2 + imag ** 2 + 1e-8)
            mask_phase = torch.atan2(imag, real)
            
            # Bound mask magnitude via tanh: [0, 1] to prevent volume explosion
            mask_mag = torch.tanh(mask_mag)
            
            # Input magnitude and phase
            input_mag = torch.sqrt(input_real ** 2 + input_imag ** 2 + 1e-8)
            input_phase = torch.atan2(input_imag, input_real)
            
            # Reconstruct with enhanced magnitude and phase shifted freely in [-pi, pi]
            est_mag = mask_mag * input_mag
            est_phase = input_phase + mask_phase
            
            enhanced_real = est_mag * torch.cos(est_phase)
            enhanced_imag = est_mag * torch.sin(est_phase)
        else:
            # Direct complex mapping
            enhanced_real = real
            enhanced_imag = imag
        
        # --- iSTFT ---
        enhanced_waveform = self.istft(enhanced_real, enhanced_imag)
        
        return enhanced_waveform

    def get_num_params(self):
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_dccrn(config: dict) -> DCCRN:
    """Build DCCRN model from config dictionary.
    
    Args:
        config: Configuration dictionary with 'stft' and 'model' sections.
    
    Returns:
        Initialized DCCRN model.
    """
    stft_cfg = config.get("stft", {})
    model_cfg = config.get("model", {})
    
    model = DCCRN(
        n_fft=stft_cfg.get("n_fft", 512),
        hop_length=stft_cfg.get("hop_length", 256),
        win_length=stft_cfg.get("win_length", 512),
        encoder_channels=model_cfg.get("encoder_channels", [16, 32, 64, 128, 256, 256]),
        kernel_size=tuple(model_cfg.get("kernel_size", [5, 3])),
        stride=tuple(model_cfg.get("stride", [2, 1])),
        lstm_hidden=model_cfg.get("lstm_hidden", 512),
        lstm_layers=model_cfg.get("lstm_layers", 2),
        bidirectional=model_cfg.get("bidirectional", True),
        use_attention=model_cfg.get("use_attention", True),
        masking_mode=model_cfg.get("masking_mode", "E"),
    )
    
    return model

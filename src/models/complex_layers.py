"""
Complex-valued neural network layers for DCCRN.

Implements complex convolution, batch normalization, and LSTM
using paired real-valued operations to simulate complex arithmetic.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ComplexConv2d(nn.Module):
    """Complex-valued 2D convolution.
    
    Implements complex multiplication:
        (a + bi) * (c + di) = (ac - bd) + (ad + bc)i
    
    Uses two real Conv2d layers to simulate complex convolution.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple = (5, 2),
        stride: tuple = (2, 1),
        padding: tuple = (2, 1),
    ):
        super().__init__()
        self.conv_real = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.conv_imag = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x_real: torch.Tensor, x_imag: torch.Tensor):
        """
        Args:
            x_real: Real part, shape (B, C, F, T)
            x_imag: Imaginary part, shape (B, C, F, T)
        
        Returns:
            Tuple of (real_out, imag_out)
        """
        # Complex multiplication: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
        out_real = self.conv_real(x_real) - self.conv_imag(x_imag)
        out_imag = self.conv_real(x_imag) + self.conv_imag(x_real)
        return out_real, out_imag


class ComplexConvTranspose2d(nn.Module):
    """Complex-valued 2D transposed convolution (decoder)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple = (5, 2),
        stride: tuple = (2, 1),
        padding: tuple = (2, 0),
        output_padding: tuple = (1, 0),
    ):
        super().__init__()
        self.conv_real = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride, padding, output_padding
        )
        self.conv_imag = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride, padding, output_padding
        )

    def forward(self, x_real: torch.Tensor, x_imag: torch.Tensor):
        out_real = self.conv_real(x_real) - self.conv_imag(x_imag)
        out_imag = self.conv_real(x_imag) + self.conv_imag(x_real)
        return out_real, out_imag


class ComplexBatchNorm2d(nn.Module):
    """Batch normalization applied independently to real and imaginary parts.
    
    A simplified version — applies standard BatchNorm2d to each component.
    For full complex BN (with covariance), see the original paper.
    """

    def __init__(self, num_features: int):
        super().__init__()
        self.bn_real = nn.BatchNorm2d(num_features)
        self.bn_imag = nn.BatchNorm2d(num_features)

    def forward(self, x_real: torch.Tensor, x_imag: torch.Tensor):
        return self.bn_real(x_real), self.bn_imag(x_imag)


class ComplexPReLU(nn.Module):
    """PReLU activation applied independently to real and imaginary parts."""

    def __init__(self, num_parameters: int = 1):
        super().__init__()
        self.prelu_real = nn.PReLU(num_parameters)
        self.prelu_imag = nn.PReLU(num_parameters)

    def forward(self, x_real: torch.Tensor, x_imag: torch.Tensor):
        return self.prelu_real(x_real), self.prelu_imag(x_imag)


class ComplexChannelAttention(nn.Module):
    """Complex-valued Channel Attention (Squeeze-and-Excitation).
    
    Computes average complex energy across time-frequency bins to
    adaptively recalibrate channel-wise feature responses (speech vs noise).
    """

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        reduced = max(channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x_real: torch.Tensor, x_imag: torch.Tensor):
        # Power per channel across Frequency and Time: (B, C)
        power = (x_real ** 2 + x_imag ** 2).mean(dim=(-2, -1))
        scale = self.fc(power).unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        return x_real * scale, x_imag * scale


class ComplexLSTM(nn.Module):
    """Complex-valued LSTM with Bi-directional support.
    
    Concatenates real and imaginary parts, processes through a bi-directional LSTM
    for full temporal context (past + future), then projects and splits back.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 512,
        num_layers: int = 2,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.bidirectional = bidirectional
        # Input is concatenated [real, imag], so input_size * 2
        self.lstm = nn.LSTM(
            input_size=input_size * 2,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )
        num_directions = 2 if bidirectional else 1
        # Project back to original feature size * 2 (for real and imag)
        self.projection = nn.Linear(hidden_size * num_directions, input_size * 2)

    def forward(self, x_real: torch.Tensor, x_imag: torch.Tensor):
        """
        Args:
            x_real: (B, T, D) real part
            x_imag: (B, T, D) imaginary part
        
        Returns:
            Tuple of (real_out, imag_out), each (B, T, D)
        """
        # Concatenate real and imaginary along feature dimension
        x = torch.cat([x_real, x_imag], dim=-1)  # (B, T, 2*D)
        
        # Process through bi-directional LSTM
        output, _ = self.lstm(x)  # (B, T, hidden_size * num_directions)
        
        # Project back
        output = self.projection(output)  # (B, T, 2*D)
        
        # Split back to real and imaginary
        out_real, out_imag = torch.chunk(output, 2, dim=-1)
        return out_real, out_imag


class ComplexEncoderBlock(nn.Module):
    """Single encoder block: ComplexConv2d + ComplexBatchNorm + ComplexPReLU + ComplexChannelAttention."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple = (5, 3),
        stride: tuple = (2, 1),
        padding: tuple = (2, 1),
        use_attention: bool = True,
    ):
        super().__init__()
        self.conv = ComplexConv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = ComplexBatchNorm2d(out_channels)
        self.activation = ComplexPReLU(out_channels)
        self.attention = ComplexChannelAttention(out_channels) if use_attention else None

    def forward(self, x_real: torch.Tensor, x_imag: torch.Tensor):
        x_real, x_imag = self.conv(x_real, x_imag)
        x_real, x_imag = self.bn(x_real, x_imag)
        x_real, x_imag = self.activation(x_real, x_imag)
        if self.attention is not None:
            x_real, x_imag = self.attention(x_real, x_imag)
        return x_real, x_imag


class ComplexDecoderBlock(nn.Module):
    """Single decoder block: ComplexConvTranspose2d + ComplexBatchNorm + ComplexPReLU + ComplexChannelAttention."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple = (5, 3),
        stride: tuple = (2, 1),
        padding: tuple = (2, 1),
        output_padding: tuple = (0, 0),
        is_last: bool = False,
        use_attention: bool = True,
    ):
        super().__init__()
        self.conv = ComplexConvTranspose2d(
            in_channels, out_channels, kernel_size, stride, padding, output_padding
        )
        self.is_last = is_last
        if not is_last:
            self.bn = ComplexBatchNorm2d(out_channels)
            self.activation = ComplexPReLU(out_channels)
            self.attention = ComplexChannelAttention(out_channels) if use_attention else None

    def forward(self, x_real: torch.Tensor, x_imag: torch.Tensor):
        x_real, x_imag = self.conv(x_real, x_imag)
        if not self.is_last:
            x_real, x_imag = self.bn(x_real, x_imag)
            x_real, x_imag = self.activation(x_real, x_imag)
            if self.attention is not None:
                x_real, x_imag = self.attention(x_real, x_imag)
        return x_real, x_imag

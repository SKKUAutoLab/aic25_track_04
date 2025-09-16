#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements the paper: "Towards Lossless Implicit Neural Representation via
Bit Plane Decomposition," CVPR 2025.

References:
    - https://github.com/WooKyoungHan/LosslessINR
"""

__all__ = [
    "BitGeLULayer",
    "BitINR",
    "BitLinear",
]

import torch
from torch.nn import functional as F

from mon.nn.modules.inr import core


# ----- BitLinear Layer -----
class BitLinear(torch.nn.Linear):

    def __init__(
        self,
        in_features : int,
        out_features: int,
        bias        : bool = False,
        num_groups  : int  = 1,
        bits        : int  = 16,
        dtype              = None
    ):
        super().__init__(in_features, out_features, bias)
        self.bits       = bits
        self.layernorm  = torch.nn.LayerNorm(normalized_shape=in_features, elementwise_affine=False, bias=False)

    def activation_quant(self, x: torch.Tensor, b: int = 8) -> torch.Tensor:
        steps = (2 ** b)
        scale = (steps - 1) / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
        y     = (x * scale).round().clamp_(-1 * steps, steps) / scale
        return y

    def weight_quant(self, w: torch.Tensor) -> torch.Tensor:
        scale = 1.0 / w.abs().mean().clamp_(min=1e-5)
        u     = (w * scale).round().clamp_(-1, 1) / scale
        return u

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w       = self.weight
        x_norm  = self.layernorm(x)
        x_quant = x_norm + (self.activation_quant(x_norm, self.bits) - x_norm).detach()
        w_quant = w + (self.weight_quant(w) - w).detach()
        y       = F.linear(x_quant, w_quant)
        return y


class BitGeLULayer(torch.nn.Module):

    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        w0          : float = 30.0,
        scale       : float = 10.0,
        is_first    : bool  = False,
        bias        : bool  = False,
        init_weights: bool  = True,
        *args, **kwargs
    ):
        super().__init__()
        self.in_channels = in_channels
        self.w0          = w0
        self.is_first    = is_first
        self.linear      = BitLinear(in_channels, out_channels, bias=bias)
        if init_weights:
            self.init_weights()

    def init_weights(self):
        """Initializes linear layer weights based on layer position."""
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Transforms input with linear layer and GELU activation.

        Args:
            x: Input tensor as ``torch.Tensor``.

        Returns:
            GELU-transformed tensor as ``torch.Tensor``.
        """
        return torch.nn.functional.gelu(self.linear(x))


# ----- BitINR -----
class BitINR(torch.nn.Module):
    """Implements the paper: "Towards Lossless Implicit Neural Representation
    via Bit Plane Decomposition," CVPR 2025.

    Args:
        in_channels: Number of input channels as ``int``.
        out_channels: Number of output channels as ``int``.
        hidden_channels: Number of channels in hidden layers as ``int``.
        hidden_layers: Number of hidden layers as ``int``.
        first_w0: Frequency for first layer as ``float``. Default is ``30.0``.
        hidden_w0: Frequency for hidden layers as ``float``. Default is ``30.0``.
        bias: Uses bias in layers if ``True``. Default is ``True``.

    References:
        - https://github.com/WooKyoungHan/LosslessINR
    """
    
    def __init__(
        self,
        in_channels    : int,
        out_channels   : int,
        hidden_channels: int,
        hidden_layers  : int,
        first_w0       : float = 30.0,
        hidden_w0      : float = 30.0,
        scale          : float = 10.0,
        bias           : bool  = True,
    ):
        super().__init__()
        self.net = torch.nn.Sequential(
            BitGeLULayer(in_channels, hidden_channels, first_w0, scale, is_first=True, bias=bias),
            *[BitGeLULayer(hidden_channels, hidden_channels, hidden_w0, scale, bias=bias) for _ in range(hidden_layers)],
            BitLinear(hidden_channels, out_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Generates output from image coordinates.

        Args:
            x: Input image tensor as ``torch.Tensor`` for size reference.

        Returns:
            Output tensor as ``torch.Tensor`` from network.
        """
        from mon import vision
        s, _   = vision.image_size(x)
        coords = core.create_coords(s).to(x.device)
        return self.net(coords)

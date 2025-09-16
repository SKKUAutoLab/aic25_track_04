#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements multiple INR layers with NdLinear transformation."""

__all__ = [
    "FINERNdLayer",
    "SineNdLayer",
]

import numpy as np
import torch

from mon.nn.modules.linear import NdLinear


class SineNdLayer(torch.nn.Module):
    """Applies NdLinear transformation with sine activation.

    Args:
        in_channels: Number of input channels as ``int``.
        out_channels: Number of output channels as ``int``.
        w0: Sine frequency factor as ``float``. Default is ``30.0``.
        is_first: First layer flag for weight initialization as ``bool``.
            Default is ``False``.
        bias: Uses bias in linear layer if ``True``. Default is ``True``.
        init_weights: Initializes weights if ``True``. Default is ``True``.

    References:
        - https://github.com/vishwa91/wire/blob/main/modules/siren.py
    """

    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        w0          : float = 30.0,
        is_first    : bool  = False,
        bias        : bool  = True,
        init_weights: bool  = True,
        *args, **kwargs
    ):
        super().__init__()
        self.in_channels  = in_channels  if isinstance(in_channels,  list | tuple) else [in_channels]
        self.out_channels = out_channels if isinstance(out_channels, list | tuple) else [out_channels]
        self.w0           = w0
        self.is_first     = is_first
        self.linear       = NdLinear(self.in_channels, self.out_channels, bias=bias)
        if init_weights:
            self.init_weights()

    def init_weights(self):
        """Initializes linear layer weights based on layer position."""
        with torch.no_grad():
            for i, l in enumerate(self.linear.align_layers):
                in_channels = self.in_channels[i]
                bound       = 1 / in_channels if self.is_first else np.sqrt(6 / in_channels) / self.w0
                l.weight.uniform_(-bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Transforms input with linear layer and sine.

        Args:
            x: Input tensor as ``torch.Tensor``.

        Returns:
            Sine-transformed tensor as ``torch.Tensor``.
        """
        return torch.sin(self.w0 * self.linear(x))

    def forward_with_intermediate(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Transforms input and returns intermediate result.

        Args:
            x: Input tensor as ``torch.Tensor``.

        Returns:
            Tuple of (sine-transformed tensor as ``torch.Tensor``,
                      intermediate tensor as ``torch.Tensor``).
        """
        intermediate = self.w0 * self.linear(x)
        return torch.sin(intermediate), intermediate


class FINERNdLayer(torch.nn.Module):
    """Applies scaled sine activation to NdLinear transformation.

    Args:
        in_channels: Number of input channels as ``int``.
        out_channels: Number of output channels as ``int``.
        w0: Sine frequency factor as ``float``. Default is ``30.0``.
        first_bias_scale: Bias scale for first layer as ``float``. Default is ``20.0``.
        is_first: First layer flag for initialization as ``bool``. Default is ``False``.
        bias: Uses bias in linear layer if ``True``. Default is ``True``.
        scale_req_grad: Scale requires gradient if ``True``. Default is ``False``.

    References:
        - https://github.com/liuzhen0212/FINER/blob/main/models.py
    """

    def __init__(
        self,
        in_channels     : int,
        out_channels    : int,
        w0              : float = 30.0,
        first_bias_scale: float = 20.0,
        is_first        : bool  = False,
        bias            : bool  = True,
        scale_req_grad  : bool  = False
    ):
        super().__init__()
        self.w0               = w0
        self.is_first         = is_first
        self.in_channels      = in_channels  if isinstance(in_channels,  list | tuple) else [in_channels]
        self.out_channels     = out_channels if isinstance(out_channels, list | tuple) else [out_channels]
        self.scale_req_grad   = scale_req_grad
        self.first_bias_scale = first_bias_scale
        self.linear           = NdLinear(self.in_channels, self.out_channels, bias=bias)
        self.init_weights()
        if self.first_bias_scale and self.is_first:
            self.init_first_bias()

    def init_weights(self):
        """Initializes linear layer weights based on layer position."""
        with torch.no_grad():
            for i, l in enumerate(self.linear.align_layers):
                in_channels = self.in_channels[i]
                bound       = 1 / in_channels if self.is_first else np.sqrt(6 / in_channels) / self.w0
                l.weight.uniform_(-bound, bound)

    def init_first_bias(self):
        """Initializes bias for the first layer."""
        with torch.no_grad():
            for l in self.linear.align_layers:
                l.bias.uniform_(-self.first_bias_scale, self.first_bias_scale)

    def generate_scale(self, x: torch.Tensor) -> torch.Tensor:
        """Generates scaling factor for activation.

        Args:
            x: Input tensor as ``torch.Tensor``.

        Returns:
            Scaling tensor as ``torch.Tensor``.
        """
        if self.scale_req_grad:
            return torch.abs(x) + 1
        with torch.no_grad():
            return torch.abs(x) + 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Transforms input with scaled sine activation.

        Args:
            x: Input tensor as ``torch.Tensor``.

        Returns:
            Transformed tensor as ``torch.Tensor``.
        """
        linear = self.linear(x)
        scale  = self.generate_scale(linear)
        return torch.sin(self.w0 * scale * linear)

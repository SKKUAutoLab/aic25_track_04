#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements upsampling and downsampling layers."""

__all__ = [
    "Downsample",
    "DownsampleConv2d",
    "Interpolate",
    "Scale",
    "Upsample",
    "UpsampleConv2d",
    "UpsamplingBilinear2d",
    "UpsamplingNearest2d",
]

import math
from typing import Any

import torch
from torch.nn import functional as F
from torch.nn.common_types import _size_2_t
from torch.nn.modules.upsampling import (
    Upsample, UpsamplingBilinear2d, UpsamplingNearest2d,
)


# ----- Downsampling -----
class Downsample(torch.nn.Module):
    """Downsamples multi-channel 1D, 2D, or 3D data.

    Args:
        size: Output spatial sizes as ``int`` or ``tuple[int, ...]``. Default is ``None``
        scale_factor: Multiplier for spatial size as ``float`` or ``tuple[float, ...]``.
            Default is ``None``
        mode: Interpolation algorithm as ``str``. One of: ``"nearest"``, ``"linear"``,
            ``"bilinear"``, ``"bicubic"``, or ``"trilinear"``. Default is ``"nearest"``.
        align_corners: Aligns corner pixels if ``True``. Default is ``False``.
            Effective only for ``"linear"``, ``"bilinear"``, ``"bicubic"``, or
            ``"trilinear"`` modes.
        recompute_scale_factor: Recomputes scale factor if ``True``. Default is ``False``.
            - If ``True``, ``scale_factor`` must be provided and is used to compute the
                output ``size``, which infers new scales for interpolation; may differ
                from provided ``scale_factor`` due to rounding.
            - If ``False``, uses ``size`` or ``scale_factor`` directly for interpolation.
    """
    
    def __init__(
        self,
        size                  : Any  = None,
        scale_factor          : Any  = None,
        mode                  : str  = "nearest",
        align_corners         : bool = False,
        recompute_scale_factor: bool = False
    ):
        super().__init__()
        self.size                   = size
        self.scale_factor           = self._invert_scale_factor(scale_factor)
        self.mode                   = mode
        self.align_corners          = align_corners
        self.recompute_scale_factor = recompute_scale_factor

    def _invert_scale_factor(self, scale_factor: Any) -> Any:
        """Inverts scale factor for downsampling.

        Args:
            scale_factor: Original scale factor as ``float`` or ``tuple[float, ...]``.

        Returns:
            Inverted scale factor as ``float`` or ``tuple[float, ...]`` or
            ``None`` if input is ``None``.
        """
        if isinstance(scale_factor, tuple):
            return tuple(1.0 / factor for factor in scale_factor)
        return 1.0 / scale_factor if scale_factor else None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Downsamples the input tensor.

        Args:
            input: Tensor to downsample as ``torch.Tensor``.

        Returns:
            Downsampled tensor as ``torch.Tensor``.
        """
        if self.size and self.size == list(input.shape[2:]):
            return input
        if self.scale_factor and isinstance(self.scale_factor, tuple) and all(s == 1.0 for s in self.scale_factor):
            return input
        return F.interpolate(
            input                  = input,
            size                   = self.size,
            scale_factor           = self.scale_factor,
            mode                   = self.mode,
            align_corners          = self.align_corners,
            recompute_scale_factor = self.recompute_scale_factor
        )


class DownsampleConv2d(torch.nn.Module):
    """Downsamples 2D data using a convolutional layer.

    Args:
        in_channels: Number of input channels as ``int``.
        out_channels: Number of output channels as ``int``.
    """
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv         = torch.nn.Sequential(torch.nn.Conv2d(in_channels, out_channels, 4, 2, 1))
        self.in_channels  = in_channels
        self.out_channels = out_channels
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Downsamples input tensor via convolution.

        Args:
            input: Tensor as ``torch.Tensor`` with shape [B, L, C].

        Returns:
            Downsampled tensor as ``torch.Tensor`` with
            shape [B, H'*W', C] where ``H' = H/2``, ``W' = W/2``.
        """
        x       = input
        b, l, c = x.shape
        h       = int(math.sqrt(l))
        w       = int(math.sqrt(l))
        x       = x.transpose(1, 2).contiguous().view(b, c, h, w)
        x       = self.conv(x).flatten(2).transpose(1, 2).contiguous()
        return x
    
    def flops(self, h: int, w: int) -> float:
        """Calculates FLOPs for the downsampling operation.

        Args:
            h: Input height as ``int``
            w: Input width as ``int``

        Returns:
            Total FLOPs as ``float``.
        """
        return h // 2 * w // 2 * self.in_channels * self.out_channels * 4 * 4


# ----- Upsampling -----
class UpsampleConv2d(torch.nn.Module):
    """Upsamples 2D data using a transposed convolutional layer.

    Args:
        in_channels: Number of input channels as ``int``.
        out_channels: Number of output channels as ``int``.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.deconv       = torch.nn.ConvTranspose2d(in_channels, out_channels, 2, 2)
        self.in_channels  = in_channels
        self.out_channels = out_channels

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Upsamples input tensor via transposed convolution.

        Args:
            input: Tensor as ``torch.Tensor`` with shape [B, L, C].

        Returns:
            Upsampled tensor as ``torch.Tensor`` with shape [B, 4*L, C].
        """
        b, l, c = input.shape
        h       = w = int(math.sqrt(l))
        x       = input.transpose(1, 2).view(b, c, h, w)
        x       = self.deconv(x).flatten(2).transpose(1, 2)
        return x

    def flops(self, h: int, w: int) -> float:
        """Calculates FLOPs for the upsampling operation.

        Args:
            h: Input height as ``int``.
            w: Input width as ``int``.

        Returns:
            Total FLOPs as ``float``.
        """
        return h * w * self.in_channels * self.out_channels * 16


# ----- Misc -----
class Scale(torch.nn.Module):
    """Applies a learnable scale parameter to input data.

    Args:
        scale: Initial scale factor value as ``float``. Default is ``1.0``.
    """

    def __init__(self, scale: float = 1.0):
        super().__init__()
        self.scale = scale

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Multiplies input tensor by scale factor.

        Args:
            input: Tensor to scale as ``torch.Tensor``.

        Returns:
            Scaled tensor as ``torch.Tensor``.
        """
        return input + self.scale


class Interpolate(torch.nn.Module):
    """Interpolates input tensor to a specified size.

    Args:
        size: Target output size as ``int`` or ``tuple[int, int]`` (height, width).
    """

    def __init__(self, size: _size_2_t):
        super().__init__()
        from mon import vision
        self.size = vision.image_size(size)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Resizes input tensor to target size.

        Args:
            input: Tensor to interpolate as ``torch.Tensor``.

        Returns:
            Interpolated tensor as ``torch.Tensor`` with shape [B, C, height, width].
        """
        return F.interpolate(input, self.size)

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements spatial attention mechanisms.

These modules focus on reweighting spatial locations (e.g., H and W dimensions) within
feature maps, emphasizing "where" to focus, often based on spatial context or relationships.
"""

__all__ = [
    "PAM",
    "PixelAttentionModule",
]

import torch
from torch.nn.common_types import _size_2_t


# ----- Pixel Attention -----
class PixelAttentionModule(torch.nn.Module):
    """Pixel Attention Module for spatial feature enhancement.

    Args:
        channels: Number of input channels as ``int``.
        reduction_ratio: Channel reduction ratio as ``int``.
        kernel_size: Size of the convolution kernel as ``int`` or ``tuple[int, int]``.
    """

    def __init__(
        self,
        channels       : int,
        reduction_ratio: int,
        kernel_size    : _size_2_t
    ):
        super().__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels  = channels,
                out_channels = channels // reduction_ratio,
                kernel_size  = kernel_size
            ),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(
                in_channels  = channels // reduction_ratio,
                out_channels = 1,
                kernel_size  = kernel_size
            )
        )
        self.act = torch.nn.Sigmoid()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Applies pixel attention to the input.

        Args:
            input: Input tensor as ``torch.Tensor`` with shape [B, C, H, W].

        Returns:
            Output tensor as ``torch.Tensor`` with shape [B, C, H, W] with
            pixel attention applied.
        """
        return input * self.act(self.fc(input))


PAM = PixelAttentionModule

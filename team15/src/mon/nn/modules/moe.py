#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements Mixture of Experts (MoE) network."""

__all__ = [
    "LayeredFeatureAggregation",
]

from typing import Sequence

import torch
from torch.nn.common_types import _size_2_t

from mon import core


# ----- Layer -----
class LayeredFeatureAggregation(torch.nn.Module):
    """Layered Feature Aggregation (LFA) fuses decoder layer features.

    Args:
        in_channels: List of input channel counts for each feature as ``list[int]``.
        out_channels: Number of output channels as ``int``.
        size: Target size for upsampling as ``int`` or ``tuple[int, int]``.
            Default is ``None`` (no resizing).
    """

    def __init__(
        self,
        in_channels : list[int],
        out_channels: int,
        size        : _size_2_t = None
    ):
        super().__init__()
        from mon import vision
        
        self.in_channels  = core.to_int_list(in_channels)
        self.out_channels = out_channels
        self.num_experts  = len(self.in_channels)

        if not self.num_experts:
            raise ValueError("[in_channels] must not be empty")

        if size:
            self.size    = vision.image_size(size)
            self.resize  = torch.nn.Upsample(size=self.size, mode="bilinear", align_corners=False)
            self.linears = torch.nn.ModuleList([
                torch.nn.Conv2d(in_c, self.out_channels, 1) for in_c in self.in_channels
            ])
        else:
            self.size    = None
            self.resize  = None
            self.linears = None

        self.conv    = torch.nn.Conv2d(self.out_channels * self.num_experts, self.out_channels, 1)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, input: Sequence[torch.Tensor]) -> torch.Tensor:
        """Aggregates layered features with attention.

        Args:
            input: Sequence of feature tensors as ``Sequence[torch.Tensor]`` with
                shapes [B, C_i, H, W].

        Returns:
            Aggregated feature tensor as ``torch.Tensor`` with shape [B, C_out, H, W].

        Raises:
            ValueError: If number of input tensors mismatches ``num_experts``.
        """
        if len(input) != self.num_experts:
            raise ValueError(f"Expected {self.num_experts} input tensors, got {len(input)}.")

        r = [
            self.linears[i](self.resize(inp)) if self.resize else self.linears[i](inp) if self.linears else inp
            for i, inp in enumerate(input)
        ]
        o_s = torch.cat(r, dim=1)  # [B, C_out * num_experts, H, W]
        w   = self.softmax(self.conv(o_s))  # [B, C_out, H, W]
        o_w = torch.stack([r[i] * w[:, i:i+1] for i in range(len(r))], dim=1)  # [B, num_experts, C_out, H, W]
        return torch.sum(o_w, dim=1)  # [B, C_out, H, W]

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements Galerkin-type attention mechanisms."""

__all__ = [
    "GalerkinSimpleAttention",
]

import torch


# ----- Galerkin-type Attention -----
class GalerkinSimpleAttention(torch.nn.Module):
    """Galerkin-type attention mechanism.

    Args:
        mid_channels: Number of intermediate channels as ``int``.
        heads: Number of attention heads as ``int``.

    Attributes:
        headc: Channels per head as ``int``.
        heads: Number of attention heads as ``int``.
        qkv_proj: QKV projection layer as ``torch.nn.Conv2d``.
        o_proj1: First output projection layer as ``torch.nn.Conv2d``.
        o_proj2: Second output projection layer as ``torch.nn.Conv2d``.
        kln: Key layer normalization as ``torch.nn.LayerNorm``.
        vln: Value layer normalization as ``torch.nn.LayerNorm``.
        act: GELU activation layer as ``torch.nn.GELU``.

    References:
        - https://github.com/2y7c3/Super-Resolution-Neural-Operator/blob/main/models/galerkin.py
    """

    def __init__(self, mid_channels: int, heads: int):
        super().__init__()
        self.headc = mid_channels // heads
        self.heads = heads

        self.qkv_proj = torch.nn.Conv2d(mid_channels, 3 * mid_channels, 1)
        self.o_proj1  = torch.nn.Conv2d(mid_channels, mid_channels, 1)
        self.o_proj2  = torch.nn.Conv2d(mid_channels, mid_channels, 1)

        self.kln = torch.nn.LayerNorm((heads, 1, self.headc))
        self.vln = torch.nn.LayerNorm((heads, 1, self.headc))

        self.act = torch.nn.GELU()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Applies Galerkin-type attention.

        Args:
            input: Input tensor as ``torch.Tensor`` with shape [B, C, H, W].

        Returns:
            Output tensor as ``torch.Tensor`` with shape [B, C, H, W] with
            attention applied.
        """
        b, c, h, w = input.shape
        qkv = self.qkv_proj(input).permute(0, 2, 3, 1).reshape(b, h * w, self.heads, 3 * self.headc)
        q, k, v = qkv.permute(0, 2, 1, 3).chunk(3, dim=-1)  # [B, heads, h*w, headc]

        k = self.kln(k)  # [B, heads, h*w, headc]
        v = self.vln(v)  # [B, heads, h*w, headc]

        v = torch.matmul(k.transpose(-2, -1), v) / (h * w)  # [B, heads, headc, headc]
        v = torch.matmul(q, v).permute(0, 2, 1, 3).reshape(b, h, w, c)  # [B, h, w, C]

        ret = v.permute(0, 3, 1, 2) + input  # [B, C, h, w]
        return self.o_proj2(self.act(self.o_proj1(ret))) + input

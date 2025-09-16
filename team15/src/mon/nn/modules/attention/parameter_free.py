#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements parameter-free or neuroscience-inspired attention mechanisms.

These modules generate attention weights without additional learnable parameters or are
inspired by biological principles, aiming for simplicity and efficiency.
"""

__all__ = [
    "SimAM",
]

import torch


class SimAM(torch.nn.Module):
    """SimAM: Simple, Parameter-Free Attention Module from the paper.

    Args:
        e_lambda: Regularization parameter for energy as ``float``. Default is ``1e-4``.

    References:
        - https://github.com/ZjjConan/SimAM
    """

    def __init__(self, e_lambda: float = 1e-4):
        super().__init__()
        self.e_lambda = e_lambda
        self.act      = torch.nn.Sigmoid()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Applies SimAM attention to the input.

        Args:
            input: Input tensor as ``torch.Tensor`` with shape [B, C, H, W].

        Returns:
            Output tensor as ``torch.Tensor`` with shape [B, C, H, W] with
            attention applied.
        """
        b, c, h, w = input.size()
        n     = w * h - 1
        d     = (input - input.mean(dim=[2, 3], keepdim=True)).pow(2)  # [B, C, H, W]
        v     = d.sum(dim=[2, 3], keepdim=True) / n   # [B, C, 1, 1]
        e_inv = d / (4 * (v + self.e_lambda)) + 0.5   # [B, C, H, W]
        return input * self.act(e_inv)

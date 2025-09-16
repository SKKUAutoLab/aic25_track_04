#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements dropout layers."""

__all__ = [
    "AlphaDropout",
    "DropBlock2d",
    "DropBlock3d",
    "DropPath",
    "Dropout",
    "Dropout1d",
    "Dropout2d",
    "Dropout3d",
    "FeatureAlphaDropout",
    "drop_block2d",
    "drop_block3d",
]

import torch
from torch.nn.modules.dropout import (
    AlphaDropout, Dropout, Dropout1d, Dropout2d, Dropout3d, FeatureAlphaDropout,
)
from torchvision.ops import drop_block2d, drop_block3d, DropBlock2d, DropBlock3d


# ----- Drop Path -----
def drop_path(
    input        : torch.Tensor,
    p            : float = 0.0,
    training     : bool  = False,
    scale_by_keep: bool  = True
) -> torch.Tensor:
    """Drops paths (Stochastic Depth) per sample in residual blocks.

    Args:
        input: Input tensor as ``torch.Tensor`` of any shape.
        p: Drop probability for each path as ``float``. Default is ``0.0``.
        training: Applies drop path during training if ``True``. Default is ``False``.
        scale_by_keep: Scales output by keep probability if ``True``. Default is ``True``.

    Returns:
        Output tensor as ``torch.Tensor`` with same shape as input, potentially dropped.

    References:
        - https://github.com/rwightman/pytorch-image-models/blob/a2727c1bf78ba0d7b5727f5f95e37fb7f8866b1f/timm/models/layers/drop.py
    """
    if p == 0.0 or not training:
        return input
    keep_prob     = 1 - p
    random_tensor = input.new_empty((input.shape[0],) + (1,) * (input.ndim - 1)).bernoulli_(keep_prob)
    if scale_by_keep and keep_prob > 0.0:
        random_tensor.div_(keep_prob)
    return input * random_tensor
    

class DropPath(torch.nn.Module):
    """Drops paths (Stochastic Depth) per sample.

    Args:
        p: Drop probability for each path as ``float``. Default is ``0.1``.
        scale_by_keep: Scales output by keep probability if ``True``. Default is ``True``.
    """
    
    def __init__(self, p: float = 0.1, scale_by_keep: bool = True):
        super().__init__()
        self.drop_prob     = p
        self.scale_by_keep = scale_by_keep

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Applies drop path to the input.

        Args:
            input: Input tensor as ``torch.Tensor`` of any shape.

        Returns:
            Output tensor as ``torch.Tensor`` with same shape as input, potentially dropped.
        """
        return drop_path(input, self.drop_prob, self.training, self.scale_by_keep)

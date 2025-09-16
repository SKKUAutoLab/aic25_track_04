#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements normalization layers."""

__all__ = [
    "AdaptiveBatchNorm2d",
    "AdaptiveInstanceNorm2d",
    "BatchNorm1d",
    "BatchNorm2d",
    "BatchNorm3d",
    "CrossMapLRN2d",
    "GroupNorm",
    "HalfInstanceNorm2d",
    "InstanceNorm1d",
    "InstanceNorm2d",
    "InstanceNorm3d",
    "LayerNorm",
    "LayerNorm2d",
    "LazyBatchNorm1d",
    "LazyBatchNorm2d",
    "LazyBatchNorm3d",
    "LazyInstanceNorm1d",
    "LazyInstanceNorm2d",
    "LazyInstanceNorm3d",
    "LearnableInstanceNorm2d",
    "LocalResponseNorm",
    "SyncBatchNorm",
]

import math
from typing import Any

import torch
from torch.nn import functional as F
from torch.nn.modules.batchnorm import (
    BatchNorm1d, BatchNorm2d, BatchNorm3d, LazyBatchNorm1d, LazyBatchNorm2d,
    LazyBatchNorm3d, SyncBatchNorm,
)
from torch.nn.modules.instancenorm import (
    InstanceNorm1d, InstanceNorm2d, InstanceNorm3d, LazyInstanceNorm1d,
    LazyInstanceNorm2d, LazyInstanceNorm3d,
)
from torch.nn.modules.normalization import (
    CrossMapLRN2d, GroupNorm, LayerNorm, LocalResponseNorm,
)


# ----- Batch Normalization -----
class AdaptiveBatchNorm2d(torch.nn.Module):
    """Applies adaptive batch normalization to 2D data.

    Args:
        num_features: Number of input channels as ``int``.
        eps: Smoothing factor for stability as ``float``. Default is ``0.999``.
        momentum: Momentum for moving averages as ``float``. Default is ``0.001``.

    References:
        - https://arxiv.org/abs/1709.00643
        - https://github.com/nrupatunga/Fast-Image-Filters
    """

    def __init__(self, num_features: int, eps: float = 0.999, momentum: float = 0.001):
        super().__init__()
        self.w0   = torch.nn.Parameter(torch.tensor(1.0))
        self.w1   = torch.nn.Parameter(torch.tensor(0.0))
        self.norm = torch.nn.BatchNorm2d(num_features, eps, momentum)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Normalizes input with learned weights.

        Args:
            input: Tensor to normalize as ``torch.Tensor``.

        Returns:
            Normalized tensor as ``torch.Tensor``.
        """
        return self.w0 * input + self.w1 * self.norm(input)


# ----- Instance Normalization -----
class AdaptiveInstanceNorm2d(torch.nn.Module):
    """Applies adaptive instance normalization to 2D data.

    Args:
        num_features: Number of input channels as ``int``.
        eps: Smoothing factor for stability as ``float``. Default is ``0.999``.
        momentum: Momentum for moving averages as ``float``. Default is ``0.001``.
        affine: Enables learnable affine parameters if ``True``. Default is ``False``.
    """

    def __init__(
        self,
        num_features: int,
        eps         : float = 0.999,
        momentum    : float = 0.001,
        affine      : bool  = False
    ):
        super().__init__()
        self.w0   = torch.nn.Parameter(torch.tensor(1.0))
        self.w1   = torch.nn.Parameter(torch.tensor(0.0))
        self.norm = torch.nn.InstanceNorm2d(num_features, eps, momentum, affine)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Normalizes input with learned weights.

        Args:
            input: Tensor to normalize as ``torch.Tensor``.

        Returns:
            Normalized tensor as ``torch.Tensor``.
        """
        return self.w0 * input + self.w1 * self.norm(input)
    

class LearnableInstanceNorm2d(torch.nn.InstanceNorm2d):
    """Normalizes a learnable fraction of 2D input features.

    Args:
        num_features: Number of input channels as ``int``.
        r: Initial fraction to normalize as ``float``. Default is ``0.5``.
        eps: Smoothing factor for stability as ``float``. Default is ``1e-5``.
        momentum: Momentum for running stats as ``float``. Default is ``0.1``.
        affine: Enables learnable affine parameters if ``True``. Default is ``True``.
        track_running_stats: Tracks running stats if ``True``. Default is ``False``.
        device: Target device as ``Any``. Default is ``None``.
        dtype: Data type as ``Any``. Default is ``None``.
    """

    def __init__(
        self,
        num_features       : int,
        r                  : float = 0.5,
        eps                : float = 1e-5,
        momentum           : float = 0.1,
        affine             : bool  = True,
        track_running_stats: bool  = False,
        device             : Any   = None,
        dtype              : Any   = None
    ):
        super().__init__(
            num_features        = num_features,
            eps                 = eps,
            momentum            = momentum,
            affine              = affine,
            track_running_stats = track_running_stats,
            device              = device,
            dtype               = dtype
        )
        self.r = torch.nn.Parameter(torch.full([num_features], r), requires_grad=True)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Applies learnable partial instance normalization.

        Args:
            input: Tensor of shape [B, C, H, W] to normalize as ``torch.Tensor``.

        Returns:
            Partially normalized tensor as ``torch.Tensor``.
        """
        self._check_input_dim(input)
        b, c, h, w = input.shape
        x_norm     = F.instance_norm(
            input           = input,
            running_mean    = self.running_mean,
            running_var     = self.running_var,
            weight          = self.weight,
            bias            = self.bias,
            use_input_stats = self.training or not self.track_running_stats,
            momentum        = self.momentum,
            eps             = self.eps
        )
        r = self.r.reshape(-1, c, 1, 1)
        return x_norm * r + input * (1 - r)


class HalfInstanceNorm2d(torch.nn.InstanceNorm2d):
    """Normalizes the first half of 2D input features.

    Args:
        num_features: Number of input channels as ``int``.
        eps: Smoothing factor for stability as ``float``. Default is ``1e-5``.
        momentum: Momentum for running stats as ``float``. Default is ``0.1``.
        affine: Enables learnable affine parameters if ``True``. Default is ``True``.
        track_running_stats: Tracks running stats if ``True``. Default is ``False``.
        device: Target device as ``Any``. Default is ``None``.
        dtype: Data type as ``Any``. Default is ``None``.
        
    Attributes:
        Inherited from ``torch.nn.InstanceNorm2d``.
    """

    def __init__(
        self,
        num_features       : int,
        eps                : float = 1e-5,
        momentum           : float = 0.1,
        affine             : bool  = True,
        track_running_stats: bool  = False,
        device             : Any   = None,
        dtype              : Any   = None
    ):
        super().__init__(
            num_features        = math.ceil(num_features / 2),
            eps                 = eps,
            momentum            = momentum,
            affine              = affine,
            track_running_stats = track_running_stats,
            device              = device,
            dtype               = dtype
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Normalizes first half of input tensor.

        Args:
            input: Tensor of shape [B, C, H, W] or [L, C, T] as ``torch.Tensor``.

        Returns:
            Tensor with first half normalized as ``torch.Tensor``.

        Raises:
            ValueError: If input is not 3D or 4D.
        """
        self._check_input_dim(input)
        if input.dim() == 3:
            y1, y2 = torch.chunk(input, 2, dim=0)
        elif input.dim() == 4:
            y1, y2 = torch.chunk(input, 2, dim=1)
        else:
            raise ValueError(f"[input] must be 3D or 4D, got {input.dim()}.")
        y1 = F.instance_norm(
            input           = y1,
            running_mean    = self.running_mean,
            running_var     = self.running_var,
            weight          = self.weight,
            bias            = self.bias,
            use_input_stats = self.training or not self.track_running_stats,
            momentum        = self.momentum,
            eps             = self.eps
        )
        return torch.cat([y1, y2], dim=1 if input.dim() == 4 else 0)
 

# ----- Layer Normalization -----
class LayerNorm2d(torch.nn.LayerNorm):
    """Normalizes channels of 2D spatial tensors [B, C, H, W].

    Args:
        normalized_shape: Shape to normalize as ``int`` or ``Sequence[int]`` (typically ``C``).
        eps: Smoothing factor for stability as ``float``. Default is ``1e-5``.
        elementwise_affine: Enables affine parameters if ``True``. Default is ``True``.
        device: Target device as ``Any``. Default is ``None``.
        dtype: Data type as ``Any``. Default is ``None``.

    Attributes:
        Inherited from ``torch.nn.LayerNorm``.
    """

    def __init__(
        self,
        normalized_shape   : Any,
        eps                : float = 1e-5,
        elementwise_affine : bool  = True,
        device             : Any   = None,
        dtype              : Any   = None
    ):
        super().__init__(
            normalized_shape   = normalized_shape,
            eps                = eps,
            elementwise_affine = elementwise_affine,
            device             = device,
            dtype              = dtype
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Applies layer normalization to 2D tensor.

        Args:
            input: Tensor of shape [B, C, H, W] as ``torch.Tensor``.

        Returns:
            Normalized tensor of shape [B, C, H, W] as ``torch.Tensor``.
        """
        return F.layer_norm(
            input.permute(0, 2, 3, 1),
            normalized_shape   = self.normalized_shape,
            weight             = self.weight,
            bias               = self.bias,
            eps                = self.eps
        ).permute(0, 3, 1, 2)

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements padding layers."""

__all__ = [
    "ConstantPad1d",
    "ConstantPad2d",
    "ConstantPad3d",
    "ReflectionPad1d",
    "ReflectionPad2d",
    "ReflectionPad3d",
    "ReplicationPad1d",
    "ReplicationPad2d",
    "ReplicationPad3d",
    "ZeroPad2d",
    "pad_same",
]

import math

import torch
from torch.nn import functional as F
from torch.nn.common_types import _size_2_t
from torch.nn.modules.padding import (
    ConstantPad1d, ConstantPad2d, ConstantPad3d, ReflectionPad1d, ReflectionPad2d,
    ReflectionPad3d, ReplicationPad1d, ReplicationPad2d, ReplicationPad3d, ZeroPad2d,
)


# ----- Utils -----
def pad_same(
    input      : torch.Tensor,
    kernel_size: _size_2_t,
    stride     : _size_2_t,
    dilation   : _size_2_t = (1, 1),
    value      : float = 0
) -> torch.Tensor:
    """Pads input tensor with 'same' padding for convolution.

    Args:
        input: Input tensor as ``torch.Tensor`` with shape [..., H, W].
        kernel_size: Size of the convolution kernel as ``int``
            or ``tuple[int, int]`` (H, W).
        stride: Stride of the convolution as ``int`` or ``tuple[int, int]`` (H, W).
        dilation: Dilation of the convolution as ``tuple[int, int]`` (H, W).
            Default is ``(1, 1)``.
        value: Padding value as ``float``. Default is ``0``.

    Returns:
        Padded tensor as ``torch.Tensor`` with same spatial size post-convolution.
    """
    def same_padding(x: int, kernel_size: int, stride: int, dilation: int) -> int:
        return max((math.ceil(x / stride) - 1) * stride + (kernel_size - 1) * dilation + 1 - x, 0)
    
    ih, iw = input.size()[-2:]
    pad_h  = same_padding(ih, kernel_size[0], stride[0], dilation[0])
    pad_w  = same_padding(iw, kernel_size[1], stride[1], dilation[1])
    if pad_h > 0 or pad_w > 0:
        input = F.pad(input, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2], value=value)
    return input

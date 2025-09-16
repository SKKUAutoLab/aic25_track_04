#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements pooling layers."""

__all__ = [
    "AdaptiveAvgMaxPool2d",
    "AdaptiveAvgPool1d",
    "AdaptiveAvgPool2d",
    "AdaptiveAvgPool3d",
    "AdaptiveCatAvgMaxPool2d",
    "AdaptiveMaxPool1d",
    "AdaptiveMaxPool2d",
    "AdaptiveMaxPool3d",
    "AdaptivePool2d",
    "AvgPool1d",
    "AvgPool2d",
    "AvgPool2dSame",
    "AvgPool3d",
    "ChannelPool",
    "FastAdaptiveAvgPool2d",
    "FractionalMaxPool2d",
    "FractionalMaxPool3d",
    "LPPool1d",
    "LPPool2d",
    "MaxPool1d",
    "MaxPool2d",
    "MaxPool2dSame",
    "MaxPool3d",
    "MaxUnpool1d",
    "MaxUnpool2d",
    "MaxUnpool3d",
    "MedianPool2d",
    "adaptive_avg_max_pool2d",
    "adaptive_cat_avg_max_pool2d",
    "adaptive_pool2d",
    "avg_pool2d_same",
    "lse_pool2d",
    "max_pool2d_same",

]

from typing import Literal

import torch
from torch.nn import functional as F
from torch.nn.common_types import _size_2_t
from torch.nn.modules.pooling import (
    AdaptiveAvgPool1d, AdaptiveAvgPool2d, AdaptiveAvgPool3d, AdaptiveMaxPool1d,
    AdaptiveMaxPool2d, AdaptiveMaxPool3d, AvgPool1d, AvgPool2d, AvgPool3d,
    FractionalMaxPool2d, FractionalMaxPool3d, LPPool1d, LPPool2d, MaxPool1d, MaxPool2d,
    MaxPool3d, MaxUnpool1d, MaxUnpool2d, MaxUnpool3d,
)

from mon import core
from mon.nn.modules import padding as pad


# ----- Adaptive Pool -----
def adaptive_avg_max_pool2d(input: torch.Tensor, output_size: int = 1) -> torch.Tensor:
    """Combines adaptive average and max pooling.

    Args:
        input: Input tensor as ``torch.Tensor`` with shape [B, C, H, W].
        output_size: Target output size for pooling as ``int``. Default is ``1``.

    Returns:
        Pooled tensor as ``torch.Tensor`` with shape [B, C, output_size, output_size].
    """
    x_avg = F.adaptive_avg_pool2d(input, output_size)
    x_max = F.adaptive_max_pool2d(input, output_size)
    return 0.5 * (x_avg + x_max)


def adaptive_cat_avg_max_pool2d(input: torch.Tensor, output_size: int = 1) -> torch.Tensor:
    """Concatenates adaptive average and max pooling.

    Args:
        input: Input tensor as ``torch.Tensor`` with shape [B, C, H, W].
        output_size: Target output size for pooling as ``int``. Default is ``1``.

    Returns:
        Concatenated tensor as ``torch.Tensor`` with
        shape [B, 2*C, output_size, output_size].
    """
    x_avg = F.adaptive_avg_pool2d(input, output_size)
    x_max = F.adaptive_max_pool2d(input, output_size)
    return torch.cat((x_avg, x_max), dim=1)


def adaptive_pool2d(
    input      : torch.Tensor,
    pool_type  : Literal["avg", "max", "avg_max", "cat_avg_max"] = "avg",
    output_size: int = 1,
) -> torch.Tensor:
    """Applies selectable global pooling with dynamic kernel size.

    Args:
        input: Input tensor as ``torch.Tensor`` with shape [B, C, H, W].
        pool_type: Type of pooling as ``Literal["avg", "max", "avg_max", "cat_avg_max"]``.
            Default is ``"avg"``.
        output_size: Target output size for pooling as ``int``. Default is ``1``.

    Returns:
        Pooled tensor as ``torch.Tensor`` with shape depending on pool_type:
        - [B, C, output_size, output_size] for ``"avg"``, ``"max"``, or ``"avg_max"``.
        - [B, 2*C, output_size, output_size] for ``"cat_avg_max"``.

    Raises:
        ValueError: If ``pool_type`` is invalid.
    """
    if pool_type == "avg":
        return F.adaptive_avg_pool2d(input, output_size)
    elif pool_type == "max":
        return F.adaptive_max_pool2d(input, output_size)
    elif pool_type == "avg_max":
        return adaptive_avg_max_pool2d(input, output_size)
    elif pool_type == "cat_avg_max":
        return adaptive_cat_avg_max_pool2d(input, output_size)
    else:
        raise ValueError(f"Invalid pool type: [{pool_type}].")


class AdaptiveAvgMaxPool2d(torch.nn.Module):
    """Combines adaptive average and max pooling in 2D.

    Args:
        output_size: Target output size for pooling as ``int``. Default is ``1``.
    """

    def __init__(self, output_size: int = 1):
        super().__init__()
        self.output_size = output_size

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Applies adaptive average and max pooling.

        Args:
            input: Input tensor as ``torch.Tensor`` with shape [B, C, H, W].

        Returns:
            Pooled tensor as ``torch.Tensor`` with shape [B, C, output_size, output_size].
        """
        return adaptive_avg_max_pool2d(input, self.output_size)


class AdaptiveCatAvgMaxPool2d(torch.nn.Module):
    """Concatenates adaptive average and max pooling in 2D.

    Args:
        output_size: Target output size for pooling as ``int``. Default is ``1``.

    Attributes:
        output_size: Target output size for pooling as ``int``.
    """

    def __init__(self, output_size: int = 1):
        super().__init__()
        self.output_size = output_size

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Applies adaptive average and max pooling with concatenation.

        Args:
            input: Input tensor as ``torch.Tensor`` with shape [B, C, H, W].

        Returns:
            Concatenated tensor as ``torch.Tensor`` with
            shape [B, 2*C, output_size, output_size].
        """
        return adaptive_cat_avg_max_pool2d(input, self.output_size)


class AdaptivePool2d(torch.nn.Module):
    """Selectable global pooling layer with dynamic kernel size.

    Args:
        output_size: Target output size for pooling as ``int``. Default is ``1``.
        pool_type: Type of pooling as ``Literal["fast", "avg", "max", "avg_max", "cat_avg_max"]``.
            Default is ``"fast"``.
        flatten: Flattens spatial dimensions if ``True``. Default is ``False``.
    """

    def __init__(
        self,
        output_size: int  = 1,
        pool_type  : Literal["fast", "avg", "max", "avg_max", "cat_avg_max"] = "fast",
        flatten    : bool = False
    ):
        super().__init__()
        self.pool_type = pool_type or ""

        self.flatten = torch.nn.Flatten(1) if flatten else torch.nn.Identity()
        if not self.pool_type:
            self.pool = torch.nn.Identity()  # pass through
        elif pool_type == "fast":
            if output_size != 1:
                raise ValueError(f"[pool_type] 'fast' requires output_size=1, got {output_size}.")
            self.pool    = FastAdaptiveAvgPool2d(flatten)
            self.flatten = torch.nn.Identity()
        elif pool_type == "avg":
            self.pool = torch.nn.AdaptiveAvgPool2d(output_size)
        elif pool_type == "max":
            self.pool = torch.nn.AdaptiveMaxPool2d(output_size)
        elif pool_type == "avg_max":
            self.pool = AdaptiveAvgMaxPool2d(output_size)
        elif pool_type == "cat_avg_max":
            self.pool = AdaptiveCatAvgMaxPool2d(output_size)
        else:
            raise ValueError(f"Invalid pool type: [{pool_type}]")

    def __repr__(self) -> str:
        """Returns a string representation of the module."""
        return f"{self.__class__.__name__}(pool_type={self.pool_type}, flatten={bool(self.flatten)})"

    def is_identity(self) -> bool:
        """Checks if the pooling is an identity operation."""
        return not self.pool_type

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Applies selected pooling and optional flattening.

        Args:
            input: Input tensor as ``torch.Tensor`` with shape [B, C, H, W].

        Returns:
            Pooled tensor as ``torch.Tensor`` with shape varying by ``pool_type``
            and flatten:
            - [B, C, output_size, output_size] or [B, C*output_size*output_size]
                for ``"fast"``, ``"avg"``, ``"max"``, ``"avg_max"``.
            - [B, 2*C, output_size, output_size] or [B, 2*C*output_size*output_size]
                for ``"cat_avg_max"``.
        """
        return self.flatten(self.pool(input))

    def feat_mult(self) -> int:
        """Returns channel multiplier for feature dimension."""
        return 2 if self.pool_type == "cat_avg_max" else 1


class FastAdaptiveAvgPool2d(torch.nn.Module):
    """Fast adaptive average pooling in 2D.

    Args:
        flatten: Removes spatial dimensions if ``True``. Default is ``False``.
    """

    def __init__(self, flatten: bool = False):
        super().__init__()
        self.flatten = flatten

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Applies fast adaptive average pooling.

        Args:
            input: Input tensor as ``torch.Tensor`` with shape [B, C, H, W].

        Returns:
            Pooled tensor as ``torch.Tensor`` with shape [B, C, 1, 1] if not flatten,
            or [B, C] if flatten.
        """
        return input.mean(dim=(2, 3), keepdim=not self.flatten)


# ----- Average Pool -----
def avg_pool2d_same(
    input            : torch.Tensor,
    kernel_size      : _size_2_t,
    stride           : _size_2_t,
    padding          : _size_2_t = 0,
    ceil_mode        : bool      = False,
    count_include_pad: bool      = True
) -> torch.Tensor:
    """Applies 2D average pooling with 'same' padding.

    Args:
        input: Input tensor as ``torch.Tensor`` with shape [B, C, H, W].
        kernel_size: Size of the pooling kernel as ``int`` or ``tuple[int, int]`` (H, W).
        stride: Stride of the pooling as ``int`` or ``tuple[int, int]`` (H, W).
        padding: Padding before 'same' adjustment as ``int`` or ``tuple[int, int]`` (H, W).
            Default is ``0``.
        ceil_mode: Uses ceil for output shape if ``True``. Default is ``False``.
        count_include_pad: Includes padding in average if ``True``. Default is ``True``.

    Returns:
        Pooled tensor as ``torch.Tensor`` with 'same' spatial size adjusted by padding.
    """
    y = pad.pad_same(
        input       = input,
        kernel_size = kernel_size,
        stride      = stride
    )
    return F.avg_pool2d(
        input             = y,
        kernel_size       = kernel_size,
        stride            = stride,
        padding           = padding,
        ceil_mode         = ceil_mode,
        count_include_pad = count_include_pad
    )


class AvgPool2dSame(torch.nn.AvgPool2d):
    """TensorFlow-like 'same' wrapper for 2D average pooling.

    Args:
        kernel_size: Size of the pooling kernel as ``int`` or ``tuple[int, int]`` (H, W).
        stride: Stride of the pooling as ``int`` or ``tuple[int, int]`` (H, W).
            Default is ``None``.
        padding: Base padding before 'same' adjustment as ``int`` or
            ``tuple[int, int]`` (H, W). Default is ``0``.
        ceil_mode: Uses ceil for output shape if ``True``. Default is ``False``.
        count_include_pad: Includes padding in average if ``True``. Default is ``True``.

    Attributes:
        Inherited from ``torch.nn.AvgPool2d``.
    """

    def __init__(
        self,
        kernel_size      : _size_2_t,
        stride           : _size_2_t = None,
        padding          : _size_2_t = 0,
        ceil_mode        : bool      = False,
        count_include_pad: bool      = True
    ):
        super().__init__(
            kernel_size       = core.to_2tuple(kernel_size),
            stride            = core.to_2tuple(stride),
            padding           = padding,
            ceil_mode         = ceil_mode,
            count_include_pad = count_include_pad
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Applies 2D average pooling with 'same' padding.

        Args:
            input: Input tensor as ``torch.Tensor`` with shape [B, C, H, W].

        Returns:
            Pooled tensor as ``torch.Tensor`` with 'same' spatial size adjusted
            by padding.
        """
        return avg_pool2d_same(
            input            = input,
            kernel_size      = self.kernel_size,
            stride           = self.stride,
            padding          = self.padding,
            ceil_mode        = self.ceil_mode,
            count_include_pad = self.count_include_pad
        )


# ----- Channel Pool -----
class ChannelPool(torch.nn.Module):
    """Global Channel Pool from CBAM Module paper.

    References:
        - https://github.com/Jongchan/attention-module/blob/master/MODELS/cbam.py
    """

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Pools channels globally with max and mean operations.

        Args:
            input: Input tensor as ``torch.Tensor`` with shape [B, C, H, W],

        Returns:
            Concatenated tensor as ``torch.Tensor`` with shape [B, 2, H, W] of max
            and mean across channels.
        """
        return torch.cat(
            tensors=(torch.max(input, 1)[0].unsqueeze(1), torch.mean(input, 1).unsqueeze(1)),
            dim=1
        )


# ----- LSE Pool -----
def lse_pool2d(input: torch.Tensor) -> torch.Tensor:
    """Applies LogSumExp (LSE) pooling, aka RealSoftMax or multivariable softplus.

    Args:
        input: Input tensor as ``torch.Tensor`` with shape [B, C, H, W].

    Returns:
        Pooled tensor as ``torch.Tensor`` with shape [B, C, 1, 1] with log-sum-exp
        over spatial dimensions.
    """
    x_flat   = input.view(input.size(0), input.size(1), -1)  # [B, C, H*W]
    x_max, _ = torch.max(x_flat, dim=2, keepdim=True)        # [B, C, 1]
    y        = x_max + (x_flat - x_max).exp().sum(dim=2, keepdim=True).log()  # [B, C, 1]
    return y.view(input.size(0), input.size(1), 1, 1)        # [B, C, 1, 1]



# ----- Max Pool -----
def max_pool2d_same(
    input      : torch.Tensor,
    kernel_size: _size_2_t,
    stride     : _size_2_t,
    padding    : _size_2_t = 0,
    dilation   : _size_2_t = 1,
    ceil_mode  : bool      = False
) -> torch.Tensor:
    """Applies 2D max pooling with 'same' padding.

    Args:
        input: Input tensor as ``torch.Tensor`` with shape [B, C, H, W].
        kernel_size: Size of the pooling kernel as ``int`` or ``tuple[int, int]`` (H, W).
        stride: Stride of the pooling as ``int`` or ``tuple[int, int]`` (H, W).
        padding: Base padding before 'same' adjustment as ``int`` or
            ``tuple[int, int]`` (H, W). Default is ``0``.
        dilation: Dilation of the pooling as ``int`` or ``tuple[int, int]`` (H, W).
            Default is ``1``.
        ceil_mode: Uses ceil for output shape if ``True``. Default is ``False``.

    Returns:
        Pooled tensor as ``torch.Tensor`` with 'same' spatial size adjusted by padding.
    """
    y = pad.pad_same(
        input       = input,
        kernel_size = kernel_size,
        stride      = stride,
        value       = -float("inf")
    )
    return F.max_pool2d(
        input       = y,
        kernel_size = kernel_size,
        stride      = stride,
        padding     = padding,
        dilation    = dilation,
        ceil_mode   = ceil_mode
    )


class MaxPool2dSame(torch.nn.MaxPool2d):
    """TensorFlow-like 'same' wrapper for 2D max pooling.

    Args:
        kernel_size: Size of the pooling kernel as ``int`` or ``tuple[int, int]`` (H, W).
        stride: Stride of the pooling as ``int`` or ``tuple[int, int]`` (H, W).
            Default is ``None``.
        padding: Base padding before 'same' adjustment as ``int`` or
            ``tuple[int, int]`` (H, W). Default is ``(0, 0)``.
        dilation: Dilation of the pooling as ``int`` or ``tuple[int, int]`` (H, W).
            Default is ``(1, 1)``.
        ceil_mode: Uses ceil for output shape if ``True``. Default is ``False``.

    Attributes:
        Inherited from ``torch.nn.MaxPool2d``.
    """
    
    def __init__(
        self,
        kernel_size: _size_2_t,
        stride     : _size_2_t = None,
        padding    : _size_2_t = (0, 0),
        dilation   : _size_2_t = (1, 1),
        ceil_mode  : bool      = False
    ):
        super().__init__(
            kernel_size = core.to_2tuple(kernel_size),
            stride      = core.to_2tuple(stride),
            padding     = padding,
            dilation    = core.to_2tuple(dilation),
            ceil_mode   = ceil_mode
        )
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Applies 2D max pooling with 'same' padding.

        Args:
            input: Input tensor as ``torch.Tensor`` with shape [B, C, H, W].

        Returns:
            Pooled tensor as ``torch.Tensor`` with 'same' spatial size adjusted by padding.
        """
        return max_pool2d_same(
            input       = input,
            kernel_size = self.kernel_size,
            stride      = self.stride,
            padding     = self.padding,
            dilation    = self.dilation,
            ceil_mode   = self.ceil_mode
        )


# ----- Median Pool -----
class MedianPool2d(torch.nn.Module):
    """Median pooling layer, usable as a median filter when stride=1.

    Args:
        kernel_size: Size of the pooling kernel as ``int`` or ``tuple[int, int]`` (H, W).
        stride: Stride of the pooling as ``int`` or ``tuple[int, int]`` (H, W).
            Default is ``(1, 1)``.
        padding: Padding as ``int`` or ``tuple[int, int]`` (H, W).
            Default is ``0`` (updated by 'same').
        same: Enforces 'same' padding if ``True``. Default is ``False``.
    """

    def __init__(
        self,
        kernel_size: _size_2_t,
        stride     : _size_2_t = (1, 1),
        padding    : _size_2_t = 0,
        same       : bool      = False
    ):
        super().__init__()
        self.kernel_size = core.to_2tuple(kernel_size)
        self.stride      = core.to_2tuple(stride)
        self.padding     = core.to_4tuple(padding)  # Convert to (left, right, top, bottom)
        self.same        = same

    def _padding(self, input: torch.Tensor) -> tuple[int, int, int, int]:
        """Calculates padding for the input tensor.

        Args:
            input: Input tensor as ``torch.Tensor`` with shape [B, C, H, W].

        Returns:
            Tuple of padding as ``tuple[int, int, int, int]`` (left, right, top, bottom).
        """
        if self.same:
            ih, iw = input.size()[2:]
            if ih % self.stride[0] == 0:
                ph = max(self.kernel_size[0] - self.stride[0], 0)
            else:
                ph = max(self.kernel_size[0] - (ih % self.stride[0]), 0)
            if iw % self.stride[1] == 0:
                pw = max(self.kernel_size[1] - self.stride[1], 0)
            else:
                pw = max(self.kernel_size[1] - (iw % self.stride[1]), 0)
            pl = pw // 2
            pr = pw - pl
            pt = ph // 2
            pb = ph - pt
            return (pl, pr, pt, pb)
        return self.padding

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Applies median pooling over spatial dimensions.

        Args:
            input: Input tensor as ``torch.Tensor`` with shape [B, C, H, W].

        Returns:
            Pooled tensor as ``torch.Tensor`` with reduced spatial size based on
            kernel and stride.
        """
        y = F.pad(input, self._padding(input), mode="reflect")
        y = y.unfold(2, self.kernel_size[0], self.stride[0])
        y = y.unfold(3, self.kernel_size[1], self.stride[1])
        return y.contiguous().view(y.size()[:4] + (-1,)).median(dim=-1)[0]

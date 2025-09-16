#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements blueprint separable convolutional layers."""

__all__ = [
    "BSConv2dS",
    "BSConv2dU",
]

import math

import torch
from torch.nn.common_types import _size_2_t


class BSConv2dS(torch.nn.Module):
    """Blueprint Separable Conv2d from MobileNets paper.

    Args:
        in_channels: Number of input channels as ``int``.
        out_channels: Number of output channels as ``int``.
        kernel_size: Size of the depthwise kernel as ``int`` or ``tuple[int, int]``.
        stride: Stride of the convolution as ``int`` or ``tuple[int, int]``.
            Default is ``1``.
        padding: Padding size or mode as ``int``, ``tuple[int, int]``, or ``str``.
            Default is ``0``.
        dilation: Dilation of the convolution as ``int`` or ``tuple[int, int]``.
            Default is ``1``.
        bias: Adds bias to depthwise conv if ``True``. Default is ``True``.
        padding_mode: Padding mode for depthwise conv as ``str``. Default is ``"zeros"``.
        p: Proportion for mid channels as ``float``. Default is ``0.25``.
        min_mid_channels: Minimum mid channels as ``int``. Default is ``4``.
        with_bn: Includes batch norm if ``True``. Default is ``False``.
        bn_kwargs: Batch norm kwargs as ``dict`` or ``None``.
            Default is ``None`` (empty dict).

    References:
        - https://arxiv.org/abs/2003.13549
        - https://github.com/zeiss-microscopy/BSConv
    """

    def __init__(
        self,
        in_channels     : int,
        out_channels    : int,
        kernel_size     : _size_2_t,
        stride          : _size_2_t = 1,
        padding         : _size_2_t | str = 0,
        dilation        : _size_2_t = 1,
        bias            : bool  = True,
        padding_mode    : str   = "zeros",
        p               : float = 0.25,
        min_mid_channels: int   = 4,
        with_bn         : bool  = False,
        bn_kwargs       : dict  = None,
        *args, **kwargs
    ):
        super().__init__()
        if not 0.0 <= p <= 1.0:
            raise AssertionError(f"[p] must be in [0.0, 1.0], got {p}.")
        mid_channels = min(in_channels, max(min_mid_channels, math.ceil(p * in_channels)))
        bn_kwargs    = bn_kwargs or {}

        self.pw1 = torch.nn.Conv2d(
            in_channels  = in_channels,
            out_channels = mid_channels,
            kernel_size  = (1, 1),
            stride       = 1,
            padding      = 0,
            dilation     = 1,
            groups       = 1,
            bias         = False
        )
        self.bn1 = (
            torch.nn.BatchNorm2d(num_features=mid_channels, **bn_kwargs)
            if with_bn else None
        )
        self.pw2 = torch.nn.Conv2d(
            in_channels  = mid_channels,
            out_channels = out_channels,
            kernel_size  = (1, 1),
            stride       = 1,
            padding      = 0,
            dilation     = 1,
            groups       = 1,
            bias         = False
        )
        self.bn2 = (
            torch.nn.BatchNorm2d(num_features=out_channels, **bn_kwargs)
            if with_bn else None
        )
        self.dw = torch.nn.Conv2d(
            in_channels  = out_channels,
            out_channels = out_channels,
            kernel_size  = kernel_size,
            stride       = stride,
            padding      = padding,
            dilation     = dilation,
            groups       = out_channels,
            bias         = bias,
            padding_mode = padding_mode
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Applies blueprint separable convolution.

        Args:
            input: Input tensor as ``torch.Tensor`` with shape [B, C_in, H, W].

        Returns:
            Output tensor as ``torch.Tensor`` with shape [B, C_out, H_out, W_out].
        """
        y = self.pw1(input)
        if self.bn1:
            y = self.bn1(y)
        y = self.pw2(y)
        if self.bn2:
            y = self.bn2(y)
        y = self.dw(y)
        return y

    def regularization_loss(self) -> torch.Tensor:
        """Computes regularization loss for pw1 weights.

        Returns:
            Frobenius norm of weight correlation matrix deviation as ``torch.Tensor``.
        """
        w   = self.pw1.weight[:, :, 0, 0]
        wwt = torch.mm(w, w.transpose(0, 1))
        i   = torch.eye(wwt.shape[0], device=wwt.device)
        return torch.norm(wwt - i, p="fro")


class BSConv2dU(torch.nn.Module):
    """Unconstrained Blueprint Separable Conv2d from MobileNets.

    Args:
        in_channels: Number of input channels as ``int``.
        out_channels: Number of output channels as ``int``.
        kernel_size: Size of the depthwise kernel as ``int`` or ``tuple[int, int]``.
        stride: Stride of the convolution as ``int`` or ``tuple[int, int]``.
            Default is ``1``.
        padding: Padding size or mode as ``int``, ``tuple[int, int]``, or ``str``.
            Default is ``0``.
        dilation: Dilation of the convolution as ``int`` or ``tuple[int, int]``.
            Default is ``1``.
        bias: Adds bias to depthwise conv if ``True``. Default is ``True``.
        padding_mode: Padding mode for depthwise conv as ``str``. Default is ``"zeros"``.
        with_bn: Includes batch norm if ``True``. Default is ``False``.
        bn_kwargs: Batch norm kwargs as ``dict`` or ``None``.
            Default is ``None`` (empty dict).

    References:
        - https://arxiv.org/abs/2003.13549
        - https://github.com/zeiss-microscopy/BSConv
    """

    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        kernel_size : _size_2_t,
        stride      : _size_2_t = 1,
        padding     : _size_2_t | str = 0,
        dilation    : _size_2_t = 1,
        bias        : bool = True,
        padding_mode: str  = "zeros",
        with_bn     : bool = False,
        bn_kwargs   : dict = None,
        *args, **kwargs
    ):
        super().__init__()
        bn_kwargs = bn_kwargs or {}

        self.pw = torch.nn.Conv2d(
            in_channels  = in_channels,
            out_channels = out_channels,
            kernel_size  = (1, 1),
            stride       = 1,
            padding      = 0,
            dilation     = 1,
            groups       = 1,
            bias         = False
        )
        self.bn = (
            torch.nn.BatchNorm2d(num_features=out_channels, **bn_kwargs)
            if with_bn else None
        )
        self.dw = torch.nn.Conv2d(
            in_channels  = out_channels,
            out_channels = out_channels,
            kernel_size  = kernel_size,
            stride       = stride,
            padding      = padding,
            dilation     = dilation,
            groups       = out_channels,
            bias         = bias,
            padding_mode = padding_mode
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Applies blueprint separable convolution.

        Args:
            input: Input tensor as ``torch.Tensor`` with shape [B, C_in, H, W].

        Returns:
            Output tensor as ``torch.Tensor`` with shape [B, C_out, H_out, W_out].
        """
        y = self.pw(input)
        if self.bn:
            y = self.bn(y)
        y = self.dw(y)
        return y

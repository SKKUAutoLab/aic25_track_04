#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements depthwise separable convolutional layers."""

__all__ = [
    "DSConv2d",
    "DSConv2dReLU",
    "DSConvAct2d",
    "DWConv2d",
    "DepthwiseConv2d",
    "DepthwiseSeparableConv2d",
    "DepthwiseSeparableConv2dReLU",
    "DepthwiseSeparableConvAct2d",
    "PWConv2d",
    "PointwiseConv2d",
]

from typing import Any

import torch
from torch.nn.common_types import _size_2_t


class DepthwiseConv2d(torch.nn.Module):
    """Depthwise 2D convolution module.

    Args:
        in_channels: Number of input channels as ``int``.
        kernel_size: Size of the convolution kernel as ``int`` or ``tuple[int, int]``.
        stride: Stride of the convolution as ``int`` or ``tuple[int, int]``.
            Default is ``1``.
        padding: Padding size or mode as ``int``, ``tuple[int, int]``, or ``str``.
            Default is ``0``.
        dilation: Dilation of the convolution as ``int`` or ``tuple[int, int]``.
            Default is ``1``.
        bias: Adds bias to convolution if ``True``. Default is ``True``.
        padding_mode: Padding mode for convolution as ``str``. Default is ``"zeros"``.
        device: Device for the module as ``Any``. Default is ``None``.
        dtype: Data type for the module as ``Any``. Default is ``None``.
    """

    def __init__(
        self,
        in_channels : int,
        kernel_size : _size_2_t,
        stride      : _size_2_t = 1,
        padding     : _size_2_t | str = 0,
        dilation    : _size_2_t = 1,
        bias        : bool = True,
        padding_mode: str  = "zeros",
        device      : Any  = None,
        dtype       : Any  = None
    ):
        super().__init__()
        self.dw_conv = torch.nn.Conv2d(
            in_channels  = in_channels,
            out_channels = in_channels,
            kernel_size  = kernel_size,
            stride       = stride,
            padding      = padding,
            dilation     = dilation,
            groups       = in_channels,
            bias         = bias,
            padding_mode = padding_mode,
            device       = device,
            dtype        = dtype
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Applies depthwise convolution.

        Args:
            input: Input tensor as ``torch.Tensor`` with shape [B, C_in, H, W].

        Returns:
            Output tensor as ``torch.Tensor`` with shape [B, C_in, H_out, W_out].
        """
        return self.dw_conv(input)


class PointwiseConv2d(torch.nn.Module):
    """Pointwise 2D convolution module with 1x1 kernel.

    Args:
        in_channels: Number of input channels as ``int``.
        out_channels: Number of output channels as ``int``.
        stride: Stride of the convolution as ``int`` or ``tuple[int, int]``.
            Default is ``1``.
        padding: Padding size or mode as ``int``, ``tuple[int, int]``, or ``str``.
            Default is ``0``.
        dilation: Dilation of the convolution as ``int`` or ``tuple[int, int]``.
            Default is ``1``.
        groups: Number of groups in convolution as ``int``. Default is ``1``.
        bias: Adds bias to convolution if ``True``. Default is ``True``.
        padding_mode: Padding mode for convolution as ``str``. Default is ``"zeros"``.
        device: Device for the module as ``Any``. Default is ``None``.
        dtype: Data type for the module as ``Any``. Default is ``None``.
    """
    
    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        stride      : _size_2_t = 1,
        padding     : _size_2_t | str = 0,
        dilation    : _size_2_t = 1,
        groups      : int  = 1,
        bias        : bool = True,
        padding_mode: str  = "zeros",
        device      : Any  = None,
        dtype       : Any  = None
    ):
        super().__init__()
        self.pw_conv = torch.nn.Conv2d(
            in_channels  = in_channels,
            out_channels = out_channels,
            kernel_size  = 1,
            stride       = stride,
            padding      = padding,
            dilation     = dilation,
            groups       = groups,
            bias         = bias,
            padding_mode = padding_mode,
            device       = device,
            dtype        = dtype
        )
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Applies pointwise convolution.

        Args:
            input: Input tensor as ``torch.Tensor`` with shape [B, C_in, H, W].

        Returns:
            Output tensor as ``torch.Tensor`` with shape [B, C_out, H_out, W_out].
        """
        return self.pw_conv(input)
    

class DepthwiseSeparableConv2d(torch.nn.Module):
    """Depthwise separable 2D convolution module.

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
        bias: Adds bias to convolutions if ``True``. Default is ``True``.
        padding_mode: Padding mode for convolutions as ``str``. Default is ``"zeros"``.
        device: Device for the module as ``Any``. Default is ``None``.
        dtype: Data type for the module as ``Any``. Default is ``None``.
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
        device      : Any  = None,
        dtype       : Any  = None
    ):
        super().__init__()
        self.dw_conv = torch.nn.Conv2d(
            in_channels  = in_channels,
            out_channels = in_channels,
            kernel_size  = kernel_size,
            stride       = stride,
            padding      = padding,
            dilation     = dilation,
            groups       = in_channels,
            bias         = bias,
            padding_mode = padding_mode,
            device       = device,
            dtype        = dtype
        )
        self.pw_conv = torch.nn.Conv2d(
            in_channels  = in_channels,
            out_channels = out_channels,
            kernel_size  = 1,
            bias         = bias,
            padding_mode = padding_mode,
            device       = device,
            dtype        = dtype
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Applies depthwise then pointwise convolution.

        Args:
            input: Input tensor as ``torch.Tensor`` with shape [B, C_in, H, W].

        Returns:
            Output tensor as ``torch.Tensor`` with shape [B, C_out, H_out, W_out].
        """
        return self.pw_conv(self.dw_conv(input))


class DepthwiseSeparableConvAct2d(torch.nn.Module):
    """Depthwise separable 2D convolution with activation.

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
        bias: Adds bias to convolutions if ``True``. Default is ``True``.
        padding_mode: Padding mode for convolutions as ``str``. Default is ``"zeros"``.
        device: Device for the module as ``Any``. Default is ``None``.
        dtype: Data type for the module as ``Any``. Default is ``None``.
        act_layer: Activation layer class as ``torch.nn.Module``. Default is ``torch.nn.ReLU``.
    """

    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        kernel_size : _size_2_t,
        stride      : _size_2_t = 1,
        padding     : _size_2_t | str = 0,
        dilation    : _size_2_t = 1,
        bias        : bool      = True,
        padding_mode: str       = "zeros",
        device      : Any       = None,
        dtype       : Any       = None,
        act_layer   : torch.nn.Module = torch.nn.ReLU
    ):
        super().__init__()
        self.ds_conv = DepthwiseSeparableConv2d(
            in_channels  = in_channels,
            out_channels = out_channels,
            kernel_size  = kernel_size,
            stride       = stride,
            padding      = padding,
            dilation     = dilation,
            bias         = bias,
            padding_mode = padding_mode,
            device       = device,
            dtype        = dtype
        )
        self.act = act_layer()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Applies depthwise separable convolution and activation.

        Args:
            input: Input tensor as ``torch.Tensor`` with shape [B, C_in, H, W].

        Returns:
            Output tensor as ``torch.Tensor`` with shape [B, C_out, H_out, W_out]
            after activation.
        """
        return self.act(self.ds_conv(input))


class DepthwiseSeparableConv2dReLU(torch.nn.Module):
    """Depthwise separable 2D convolution with ReLU activation.

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
        bias: Adds bias to convolutions if ``True``. Default is ``True``.
        padding_mode: Padding mode for convolutions as ``str``. Default is ``"zeros"``.
        device: Device for the module as ``Any``. Default is ``None``.
        dtype: Data type for the module as ``Any``. Default is ``None``.
    """

    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        kernel_size : _size_2_t,
        stride      : _size_2_t = 1,
        padding     : _size_2_t | str = 0,
        dilation    : _size_2_t = 1,
        bias        : bool      = True,
        padding_mode: str       = "zeros",
        device      : Any       = None,
        dtype       : Any       = None
    ):
        super().__init__()
        self.ds_conv = DepthwiseSeparableConv2d(
            in_channels  = in_channels,
            out_channels = out_channels,
            kernel_size  = kernel_size,
            stride       = stride,
            padding      = padding,
            dilation     = dilation,
            bias         = bias,
            padding_mode = padding_mode,
            device       = device,
            dtype        = dtype
        )
        self.act = torch.nn.ReLU(inplace=True)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Applies depthwise separable convolution and ReLU.

        Args:
            input: Input tensor as ``torch.Tensor`` with shape [B, C_in, H, W].

        Returns:
            Output tensor as ``torch.Tensor`` with shape [B, C_out, H_out, W_out] after ReLU.
        """
        return self.act(self.ds_conv(input))


DWConv2d     = DepthwiseConv2d
PWConv2d     = PointwiseConv2d
DSConv2d     = DepthwiseSeparableConv2d
DSConvAct2d  = DepthwiseSeparableConvAct2d
DSConv2dReLU = DepthwiseSeparableConv2dReLU

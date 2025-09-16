#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements channel attention mechanisms.

These modules focus on re-weighting or re-calibrating feature channels (e.g., in a
tensor of shape [B, C, H, W], where C is the number of channels) based on inter-channel
relationships, emphasizing "what" is important in the feature map.
"""

__all__ = [
    "ChannelAttentionModule",
    "ECA",
    "ECA1d",
    "EfficientChannelAttention",
    "EfficientChannelAttention1d",
    "SimplifiedChannelAttention",
    "SqueezeExcitation",
    "SqueezeExciteC",
    "SqueezeExciteL",
]

from typing import Any

import torch
from torch.nn.common_types import _size_2_t
from torchvision.ops.misc import SqueezeExcitation


# ----- Channel Attention -----
class ChannelAttentionModule(torch.nn.Module):
    """Channel Attention Module for feature enhancement.

    Args:
        channels: Number of input channels as ``int``.
        reduction_ratio: Channel reduction ratio as ``int``.
        stride: Stride of the first convolution as ``int``. Default is ``1``.
        padding: Padding of the first convolution as ``int``. Default is ``0``.
        dilation: Dilation of the convolutions as ``int``. Default is ``1``.
        groups: Number of groups in the convolutions as ``int``. Default is ``1``.
        bias: Adds bias to convolutions if ``True``. Default is ``True``.
        padding_mode: Padding mode for convolutions as ``str``. Default is ``"zeros"``.
        device: Device for the module as ``Any``. Default is ``None``.
        dtype: Data type for the module as ``Any``. Default is ``None``.
    """

    def __init__(
        self,
        channels       : int,
        reduction_ratio: int,
        stride         : int  = 1,
        padding        : int  = 0,
        dilation       : int  = 1,
        groups         : int  = 1,
        bias           : bool = True,
        padding_mode   : str  = "zeros",
        device         : Any  = None,
        dtype          : Any  = None
    ):
        super().__init__()
        self.avg_pool   = torch.nn.AdaptiveAvgPool2d(1)
        self.excitation = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels  = channels,
                out_channels = channels // reduction_ratio,
                kernel_size  = 1,
                stride       = stride,
                padding      = padding,
                dilation     = dilation,
                groups       = groups,
                bias         = bias,
                padding_mode = padding_mode,
                device       = device,
                dtype        = dtype
            ),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(
                in_channels  = channels // reduction_ratio,
                out_channels = channels,
                kernel_size  = 1,
                padding      = 0,
                dilation     = dilation,
                groups       = groups,
                bias         = bias,
                padding_mode = padding_mode,
                device       = device,
                dtype        = dtype
            ),
            torch.nn.Sigmoid()
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Applies channel attention to the input.

        Args:
            input: Input tensor as ``torch.Tensor`` with shape [B, C, H, W].

        Returns:
            Output tensor as ``torch.Tensor`` with shape [B, C, H, W] with
            channel attention applied.
        """
        return input * self.excitation(self.avg_pool(input))


# ----- Efficient Channel Attention -----
class EfficientChannelAttention(torch.nn.Module):
    """Efficient Channel Attention (ECA) module.

    Args:
        channels: Number of input channels as ``int``.
        kernel_size: Kernel size for 1D convolution as ``int`` or ``tuple[int, int]``.
            Default is ``3``.
    """

    def __init__(self, channels: int, kernel_size: _size_2_t = 3):
        super().__init__()
        padding       = (kernel_size - 1) // 2
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.conv     = torch.nn.Conv1d(
            in_channels  = 1,
            out_channels = 1,
            kernel_size  = kernel_size,
            padding      = padding,
            bias         = False
        )
        self.sigmoid  = torch.nn.Sigmoid()
        self.channel  = channels
        self.k_size   = kernel_size

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Applies efficient channel attention.

        Args:
            input: Input tensor as ``torch.Tensor`` with shape [B, C, H, W].

        Returns:
            Output tensor as ``torch.Tensor`` with shape [B, C, H, W] with
            channel attention applied.
        """
        x = input
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)

    def flops(self) -> float:
        """Calculates FLOPs for the module.

        Returns:
            Number of floating-point operations as ``float``.
        """
        return self.channel * self.channel * self.k_size


class EfficientChannelAttention1d(torch.nn.Module):
    """Efficient Channel Attention (ECA) module for 1D inputs.

    Args:
        channels: Number of input channels as ``int``.
        kernel_size: Kernel size for 1D convolution as ``int`` or ``tuple[int, int]``.
            Default is ``3``.
    """

    def __init__(
        self,
        channels   : int,
        kernel_size: _size_2_t = 3
    ):
        super().__init__()
        padding       = (kernel_size - 1) // 2
        self.avg_pool = torch.nn.AdaptiveAvgPool1d(1)
        self.conv     = torch.nn.Conv1d(
            in_channels  = 1,
            out_channels = 1,
            kernel_size  = kernel_size,
            padding      = padding,
            bias         = False
        )
        self.sigmoid  = torch.nn.Sigmoid()
        self.channel  = channels
        self.k_size   = kernel_size

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Applies efficient channel attention to 1D input.

        Args:
            input: Input tensor as ``torch.Tensor`` with shape [B, C, L].

        Returns:
            Output tensor as ``torch.Tensor`` with shape [B, C, L] with
            channel attention applied.
        """
        x = input
        y = self.avg_pool(x)                   # [B, C, 1]
        y = self.conv(y.transpose(-1, -2))     # [B, 1, C] -> [B, 1, C]
        y = self.sigmoid(y.transpose(-1, -2))  # [B, C, 1]
        return x * y.expand_as(x)

    def flops(self) -> float:
        """Calculates FLOPs for the module.

        Returns:
            Number of floating-point operations as ``float``
        """
        return self.channel * self.channel * self.k_size


ECA   = EfficientChannelAttention
ECA1d = EfficientChannelAttention1d


# ----- Simplified Channel Attention -----
class SimplifiedChannelAttention(torch.nn.Module):
    """Simplified channel attention from 'Simple Baselines for Image Restoration'.

    Args:
        channels: Number of input/output channels as ``int``.
        bias: Adds bias to convolution if ``True``. Default is ``True``.
        device: Device for the module as ``Any``. Default is ``None``.
        dtype: Data type for the module as ``Any``. Default is ``None``.

    References:
        - https://arxiv.org/pdf/2204.04676.pdf
    """

    def __init__(
        self,
        channels: int,
        bias    : bool = True,
        device  : Any  = None,
        dtype   : Any  = None
    ):
        super().__init__()
        self.avg_pool   = torch.nn.AdaptiveAvgPool2d(1)
        self.excitation = torch.nn.Conv2d(
            in_channels  = channels,
            out_channels = channels,
            kernel_size  = 1,
            stride       = 1,
            padding      = 0,
            bias         = bias,
            device       = device,
            dtype        = dtype
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Applies simplified channel attention.

        Args:
            input: Input tensor as ``torch.Tensor`` with shape [B, C, H, W].

        Returns:
            Output tensor as ``torch.Tensor`` with shape [B, C, H, W] with
            channel attention applied.
        """
        x = input
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)  # [B, C, 1, 1] -> [B, C]
        y = self.excitation(y.view(b, c, 1, 1)).view(b, c, 1, 1)  # [B, C] -> [B, C, 1, 1]
        return x * y.expand_as(x)
    

# ----- Squeeze and Excitation -----
class SqueezeExciteC(torch.nn.Module):
    """Squeeze and Excite layer using Conv2d from 'Squeeze and Excitation' paper.

    Args:
        channels: Number of input channels as ``int``.
        reduction_ratio: Channel reduction ratio as ``int``. Default is ``16``.
        bias: Adds bias to convolutions if ``True``. Default is ``False``.

    References:
        - https://amaarora.github.io/2020/07/24/SeNet.html#squeeze-and-excitation-block-in-pytorch
        - https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py
    """
    
    def __init__(
        self,
        channels       : int,
        reduction_ratio: int  = 16,
        bias           : bool = False,
    ):
        super().__init__()
        self.avg_pool   = torch.nn.AdaptiveAvgPool2d(1)  # squeeze
        self.excitation = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels  = channels,
                out_channels = channels  // reduction_ratio,
                kernel_size  = 1,
                bias         = bias,
            ),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(
                in_channels  = channels  // reduction_ratio,
                out_channels = channels,
                kernel_size  = 1,
                bias         = bias,
            ),
            torch.nn.Sigmoid()
        )
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Applies squeeze and excite attention.

        Args:
            input: Input tensor as ``torch.Tensor`` with shape [B, C, H, W].

        Returns:
            Output tensor as ``torch.Tensor`` with shape [B, C, H, W] with
            channel attention applied.
        """
        return input * self.excitation(self.avg_pool(input))


class SqueezeExciteL(torch.nn.Module):
    """Squeeze and Excite layer using Linear from 'Squeeze and Excitation' paper.

    Args:
        channels: Number of input channels as ``int``.
        reduction_ratio: Channel reduction ratio as ``int``. Default is ``16``.
        bias: Adds bias to linear layers if ``True``. Default is ``False``.

    References:
        - https://amaarora.github.io/2020/07/24/SeNet.html#squeeze-and-excitation-block-in-pytorch
        - https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py
    """
    
    def __init__(
        self,
        channels       : int,
        reduction_ratio: int  = 16,
        bias           : bool = False,
    ):
        super().__init__()
        self.avg_pool   = torch.nn.AdaptiveAvgPool2d(1)
        self.excitation = torch.nn.Sequential(
            torch.nn.Linear(
                in_features  = channels,
                out_features = channels // reduction_ratio,
                bias         = bias
            ),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(
                in_features  = channels // reduction_ratio,
                out_features = channels,
                bias         = bias
            ),
            torch.nn.Sigmoid()
        )
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Applies squeeze and excite attention.

        Args:
            input: Input tensor as ``torch.Tensor`` with shape [B, C, H, W].

        Returns:
            Output tensor as ``torch.Tensor`` with shape [B, C, H, W] with
            channel attention applied.
        """
        b, c, _, _ = input.shape
        y = self.avg_pool(input).view(b, c)      # [B, C, 1, 1] -> [B, C]
        y = self.excitation(y).view(b, c, 1, 1)  # [B, C] -> [B, C, 1, 1]
        return input * y

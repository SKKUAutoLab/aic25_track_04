#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements hybrid attention mechanisms.

These combine channel and spatial attention, often sequentially or in parallel, to
refine features across both dimensions, providing a more comprehensive focus mechanism.
"""

__all__ = [
    "BAM",
    "CBAM",
]

import torch
from torch.nn import functional as F


# ----- Channel-Spatial Attention -----
class BAM(torch.nn.Module):
    """Bottleneck Attention Module from BAM paper.

    Args:
        channels: Number of input channels as ``int``.
        reduction_ratio: Channel reduction ratio as ``int``. Default is ``16``.
        num_layers: Number of hidden layers in channel attention as ``int``.
            Default is ``1``.

    References:
        - https://github.com/Jongchan/attention-module/blob/master/MODELS/bam.py
    """

    class Flatten(torch.nn.Module):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            """Flattens input tensor to [B, -1].

            Args:
                input: Input tensor as ``torch.Tensor`` of any shape.

            Returns:
                Flattened tensor as ``torch.Tensor`` with shape [B, -1].
            """
            return input.view(input.size(0), -1)

    class ChannelAttention(torch.nn.Module):
        """Channel attention submodule for BAM.

        Args:
            channels: Number of input channels as ``int``.
            reduction_ratio: Channel reduction ratio as ``int``. Default is ``16``.
            num_layers: Number of hidden layers as ``int``. Default is ``1``.

        Attributes:
            c_gate: Sequential layer for channel gating as ``torch.nn.Sequential``.
        """
        def __init__(
            self,
            channels       : int,
            reduction_ratio: int = 16,
            num_layers     : int = 1
        ):
            super().__init__()
            gate_channels = [channels] + [channels // reduction_ratio] * num_layers + [channels]
            self.c_gate = torch.nn.Sequential(
                BAM.Flatten(),
                *[
                    torch.nn.Sequential(
                        torch.nn.Linear(gate_channels[i], gate_channels[i + 1]),
                        torch.nn.BatchNorm1d(gate_channels[i + 1]),
                        torch.nn.ReLU()
                    ) for i in range(len(gate_channels) - 2)
                ],
                torch.nn.Linear(gate_channels[-2], gate_channels[-1])
            )

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            """Applies channel attention.

            Args:
                input: Input tensor as ``torch.Tensor`` with shape [B, C, H, W].

            Returns:
                Attention weights tensor as ``torch.Tensor`` with shape [B, C, H, W].
            """
            y = F.avg_pool2d(input, input.size()[2:], stride=input.size()[2:])  # [B, C, 1, 1]
            y = self.c_gate(y)  # [B, C]
            return y.unsqueeze(2).unsqueeze(3).expand_as(input)  # [B, C, H, W]

    class SpatialAttention(torch.nn.Module):
        """Spatial attention submodule for BAM.

        Args:
            channels: Number of input channels as ``int``.
            reduction_ratio: Channel reduction ratio as ``int``. Default is ``16``.
            dilation_conv_num: Number of dilated convolutions as ``int``. Default is ``2``.
            dilation_val: Dilation value for convolutions as ``int``. Default is ``4``.

        Attributes:
            s_gate: Sequential layer for spatial gating as ``torch.nn.Sequential``.
        """
        def __init__(
            self,
            channels         : int,
            reduction_ratio  : int = 16,
            dilation_conv_num: int = 2,
            dilation_val     : int = 4
        ):
            super().__init__()
            self.s_gate = torch.nn.Sequential(
                torch.nn.Conv2d(
                    in_channels  = channels,
                    out_channels = channels // reduction_ratio,
                    kernel_size  = 1
                ),
                torch.nn.BatchNorm2d(channels // reduction_ratio),
                torch.nn.ReLU(),
                *[
                    torch.nn.Sequential(
                        torch.nn.Conv2d(
                            in_channels  = channels // reduction_ratio,
                            out_channels = channels // reduction_ratio,
                            kernel_size  = 3,
                            padding      = dilation_val,
                            dilation     = dilation_val
                        ),
                        torch.nn.BatchNorm2d(channels // reduction_ratio),
                        torch.nn.ReLU()
                    ) for _ in range(dilation_conv_num)
                ],
                torch.nn.Conv2d(
                    in_channels  = channels // reduction_ratio,
                    out_channels = 1,
                    kernel_size  = 1
                )
            )

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            """Applies spatial attention.

            Args:
                input: Input tensor as ``torch.Tensor`` with shape [B, C, H, W].

            Returns:
                Attention weights tensor as ``torch.Tensor`` with shape [B, C, H, W].
            """
            return self.s_gate(input).expand_as(input)

    def __init__(
        self,
        channels       : int,
        reduction_ratio: int = 16,
        num_layers     : int = 1
    ):
        super().__init__()
        self.channel_att = self.ChannelAttention(
            channels        = channels,
            reduction_ratio = reduction_ratio,
            num_layers      = num_layers
        )
        self.spatial_att = self.SpatialAttention(
            channels        = channels,
            reduction_ratio = reduction_ratio
        )
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Applies bottleneck attention.

        Args:
            input: Input tensor as ``torch.Tensor`` with shape [B, C, H, W].

        Returns:
            Output tensor as ``torch.Tensor`` with shape [B, C, H, W] with
            attention applied.
        """
        y = 1 + self.sigmoid(self.channel_att(input) * self.spatial_att(input))
        return input * y


class CBAM(torch.nn.Module):
    """Convolutional Block Attention Module from CBAM paper.

    Args:
        channels: Number of input channels as ``int``.
        reduction_ratio: Channel reduction ratio as ``int``. Default is ``16``.
        pool_types: Pooling layer types as ``list[str]``. Default is ["avg", "max"].
        spatial: Includes spatial attention if ``True``. Default is ``True``.

    References:
        - https://github.com/Jongchan/attention-module/blob/master/MODELS/cbam.py
    """

    class Flatten(torch.nn.Module):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            """Flattens input tensor to [B, -1].

            Args:
                input: Input tensor as ``torch.Tensor`` of any shape.

            Returns:
                Flattened tensor as ``torch.Tensor`` with shape [B, -1].
            """
            return input.view(input.size(0), -1)

    class ChannelAttention(torch.nn.Module):
        """Channel attention submodule for CBAM.

        Args:
            channels: Number of input channels as ``int``.
            reduction_ratio: Channel reduction ratio as ``int``. Default is ``16``.
            pool_types: Pooling layer types as ``list[str]``. Default is ["avg", "max"].
        """
        def __init__(
            self,
            channels       : int,
            reduction_ratio: int = 16,
            pool_types     : list[str] = ["avg", "max"]
        ):
            super().__init__()
            self.mlp = torch.nn.Sequential(
                CBAM.Flatten(),
                torch.nn.Linear(channels, channels // reduction_ratio),
                torch.nn.ReLU(),
                torch.nn.Linear(channels // reduction_ratio, channels)
            )
            self.pool_types = pool_types

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            """Applies channel attention.

            Args:
                input: Input tensor as ``torch.Tensor`` with shape [B, C, H, W].

            Returns:
                Output tensor as ``torch.Tensor`` with shape [B, C, H, W] with
                channel attention applied.
            """
            channel_att_sum = sum(
                self.mlp(
                    getattr(F, f"{pool_type}_pool2d")(
                        input, input.size()[2:], stride=input.size()[2:]
                    )
                ) for pool_type in self.pool_types
            )
            return input * torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(input)

    class SpatialAttention(torch.nn.Module):
        """Spatial attention submodule for CBAM.

        Attributes:
            spatial: Sequential layer for spatial gating as ``torch.nn.Sequential``.
        """
        def __init__(self):
            super().__init__()
            self.spatial = torch.nn.Sequential(
                torch.nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
                torch.nn.BatchNorm2d(1),
                torch.nn.Sigmoid()
            )

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            """Applies spatial attention.

            Args:
                input: Input tensor as ``torch.Tensor`` with shape [B, C, H, W].

            Returns:
                Output tensor as ``torch.Tensor`` with shape [B, C, H, W] with
                spatial attention applied.
            """
            y = torch.cat([input.mean(dim=1, keepdim=True), input.max(dim=1, keepdim=True)[0]], dim=1)
            return input * self.spatial(y).expand_as(input)

    def __init__(
        self,
        channels       : int,
        reduction_ratio: int       = 16,
        pool_types     : list[str] = ["avg", "max"],
        spatial        : bool      = True
    ):
        super().__init__()
        self.channel_att = self.ChannelAttention(
            channels        = channels,
            reduction_ratio = reduction_ratio,
            pool_types      = pool_types
        )
        self.spatial_att = self.SpatialAttention() if spatial else torch.nn.Identity()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Applies convolutional block attention.

        Args:
            input: Input tensor as ``torch.Tensor`` with shape [B, C, H, W].

        Returns:
            Output tensor as ``torch.Tensor`` with shape [B, C, H, W] with
            attention applied.
        """
        return self.spatial_att(self.channel_att(input))

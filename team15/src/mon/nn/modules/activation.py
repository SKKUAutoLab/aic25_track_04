#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements activation layers."""

__all__ = [
    "ArgMax",
    "CELU",
    "Clamp",
    "Clip",
    "ELU",
    "FReLU",
    "GELU",
    "GLU",
    "Hardshrink",
    "Hardsigmoid",
    "Hardswish",
    "Hardtanh",
    "LeakyReLU",
    "LogSigmoid",
    "LogSoftmax",
    "Mish",
    "MultiheadAttention",
    "NegHardsigmoid",
    "PReLU",
    "RReLU",
    "ReLU",
    "ReLU6",
    "SELU",
    "SiLU",
    "Sigmoid",
    "SimpleGate",
    "Sine",
    "Softmax",
    "Softmax2d",
    "Softmin",
    "Softplus",
    "Softshrink",
    "Softsign",
    "Tanh",
    "Tanhshrink",
    "Threshold",
    "hard_sigmoid",
    "to_act_layer",
    "xUnit",
    "xUnitD",
    "xUnitS",
]

from typing import Any

import torch
from torch.nn import functional as F
from torch.nn.common_types import _size_2_t
from torch.nn.modules.activation import (
    CELU, ELU, GELU, GLU, Hardshrink, Hardsigmoid, Hardswish, Hardtanh, LeakyReLU,
    LogSigmoid, LogSoftmax, Mish, MultiheadAttention, PReLU, ReLU, ReLU6, RReLU, SELU,
    Sigmoid, SiLU, Softmax, Softmax2d, Softmin, Softplus, Softshrink, Softsign,
    Tanh, Tanhshrink, Threshold,
)

from mon import core


# ----- Linear Unit -----
class FReLU(torch.nn.Module):
    """Funnel ReLU activation with depthwise convolution.

    Args:
        channels: Number of input channels as ``int``.
        kernel_size: Size of the convolution kernel as ``int`` or ``tuple[int, int]``.
            Default is ``3``.
    """

    def __init__(self, channels: int, kernel_size: _size_2_t = 3):
        super().__init__()
        kernel_size = core.to_2tuple(kernel_size)
        self.conv   = torch.nn.Conv2d(
            in_channels  = channels,
            out_channels = channels,
            kernel_size  = kernel_size,
            stride       = 1,
            padding      = 1,
            groups       = channels
        )
        self.act    = torch.nn.BatchNorm2d(channels)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Applies FReLU activation with max operation.

        Args:
            input: Input tensor as ``torch.Tensor`` with shape [B, C, H, W].

        Returns:
            Output tensor as ``torch.Tensor`` with shape [B, C, H, W] after FReLU.
        """
        return torch.max(input, self.act(self.conv(input)))


class SimpleGate(torch.nn.Module):
    """Simple gate activation unit from 'Simple Baselines for Image Restoration'.

    References:
        - https://arxiv.org/pdf/2204.04676.pdf
    """
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Applies simple gate activation by chunking and multiplication.

        Args:
            input: Input tensor as ``torch.Tensor`` with shape [B, C, H, W],
                where ``C`` is even.
    
        Returns:
            Output tensor as ``torch.Tensor`` with shape [B, C/2, H, W] after chunking
            and multiplication.
        """
        x1, x2 = input.chunk(chunks=2, dim=1)
        return x1 * x2


# ----- Sigmoid -----
def hard_sigmoid(input: torch.Tensor, inplace: bool = False) -> torch.Tensor:
    """Applies hard sigmoid activation.

    Args:
        input: Input tensor as ``torch.Tensor`` of any shape.
        inplace: Modifies input in-place if ``True``. Default is ``False``.

    Returns:
        Output tensor as ``torch.Tensor`` with values in [0, 1], same shape as input.
    """
    if inplace:
        return input.add_(3.0).clamp_(0.0, 6.0).div_(6.0)
    else:
        return F.relu6(input + 3.0) / 6.0


class NegHardsigmoid(torch.nn.Module):
    """Negative hard sigmoid activation.

    Args:
        inplace: Modifies input in-place if ``True``. Default is ``True``.
    """

    def __init__(self, inplace: bool = True):
        super().__init__()
        self.inplace = inplace

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Applies negative hard sigmoid activation.

        Args:
            input: Input tensor as ``torch.Tensor`` of any shape.

        Returns:
            Output tensor as ``torch.Tensor`` with values in [-0.5, 0.5],
            same shape as input
        """
        return F.relu6(3 * input + 3.0, inplace=self.inplace) / 6.0 - 0.5


# ----- Sine -----
class Sine(torch.nn.Module):
    """Sine activation unit.

    Args:
        w0: Frequency scaling factor as ``float``. Default is ``1.0``.

    References:
        - https://github.com/lucidrains/siren-pytorch/blob/master/siren_pytorch/siren_pytorch.py
    """

    def __init__(self, w0: float = 1.0):
        super().__init__()
        self.w0 = w0

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Applies sine activation.

        Args:
            input: Input tensor as ``torch.Tensor`` of any shape.

        Returns:
            Output tensor as ``torch.Tensor`` with same shape as input.
        """
        return torch.sin(self.w0 * input)

    def forward_with_intermediate(self, input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Applies sine activation and returns intermediate value.

        Args:
            input: Input tensor as ``torch.Tensor`` of any shape.

        Returns:
            Tuple of (sine output ``torch.Tensor``,
                      intermediate value ``torch.Tensor``) with same shape as input.
        """
        intermediate = self.w0 * input  # Corrected: Removed undefined self.linear
        return torch.sin(intermediate), intermediate
    

# ----- xUnit -----
class xUnit(torch.nn.Module):
    """xUnit spatial activation layer.

    Args:
        num_features: Number of input/output channels as ``int``. Default is ``64``.
        kernel_size: Size of the depthwise kernel as ``int`` or ``tuple[int, int]``.
            Default is ``7``.
        batch_norm: Includes batch normalization if ``True``. Default is ``False``.

    References:
        - https://blog.paperspace.com/xunit-spatial-activation
        - https://github.com/kligvasser/xUnit
    """

    def __init__(
        self,
        num_features: int       = 64,
        kernel_size : _size_2_t = 7,
        batch_norm  : bool      = False
    ):
        super().__init__()
        padding = kernel_size // 2  # Corrected: kernel_size is an int or tuple, not self-referenced
        self.features = torch.nn.Sequential(
            torch.nn.BatchNorm2d(num_features) if batch_norm else torch.nn.Identity(),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                in_channels  = num_features,
                out_channels = num_features,
                kernel_size  = kernel_size,
                padding      = padding,
                groups       = num_features
            ),
            torch.nn.BatchNorm2d(num_features) if batch_norm else torch.nn.Identity(),
            torch.nn.Sigmoid()
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Applies xUnit activation with spatial gating.

        Args:
            input: Input tensor as ``torch.Tensor`` with shape [B, C, H, W].

        Returns:
            Output tensor as ``torch.Tensor`` with shape [B, C, H, W] with gated activation.
        """
        return input * self.features(input)
    

class xUnitS(torch.nn.Module):
    """Slim xUnit spatial activation layer.

    Args:
        num_features: Number of input/output channels as ``int``. Default is ``64``.
        kernel_size: Size of the depthwise kernel as ``int`` or ``tuple[int, int]``.
            Default is ``7``.
        batch_norm: Includes batch normalization if ``True``. Default is ``False``.

    References:
        - https://blog.paperspace.com/xunit-spatial-activation
        - https://github.com/kligvasser/xUnit
    """

    def __init__(
        self,
        num_features: int       = 64,
        kernel_size : _size_2_t = 7,
        batch_norm  : bool      = False
    ):
        super().__init__()
        padding = kernel_size // 2  # Corrected: kernel_size is a parameter, not self-referenced
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels  = num_features,
                out_channels = num_features,
                kernel_size  = kernel_size,
                padding      = padding,
                groups       = num_features
            ),
            torch.nn.BatchNorm2d(num_features) if batch_norm else torch.nn.Identity(),
            torch.nn.Sigmoid()
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Applies slim xUnit activation with spatial gating.

        Args:
            input: Input tensor as ``torch.Tensor`` with shape [B, C, H, W].

        Returns:
            Output tensor as ``torch.Tensor`` with shape [B, C, H, W] with gated activation.
        """
        return input * self.features(input)
    

class xUnitD(torch.nn.Module):
    """Dense xUnit spatial activation layer.

    Args:
        num_features: Number of input/output channels as ``int``. Default is ``64``.
        kernel_size: Size of the depthwise kernel as ``int`` or ``tuple[int, int]``.
            Default is ``7``.
        batch_norm: Includes batch normalization if ``True``. Default is ``False``.

    References:
        - https://blog.paperspace.com/xunit-spatial-activation
        - https://github.com/kligvasser/xUnit
    """

    def __init__(
        self,
        num_features: int       = 64,
        kernel_size : _size_2_t = 7,
        batch_norm  : bool      = False
    ):
        super().__init__()
        padding = kernel_size // 2  # Corrected: kernel_size is a parameter, not self-referenced
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels  = num_features,
                out_channels = num_features,
                kernel_size  = 1,
                padding      = 0
            ),
            torch.nn.BatchNorm2d(num_features) if batch_norm else torch.nn.Identity(),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                in_channels  = num_features,
                out_channels = num_features,
                kernel_size  = kernel_size,
                padding      = padding,
                groups       = num_features
            ),
            torch.nn.BatchNorm2d(num_features) if batch_norm else torch.nn.Identity(),
            torch.nn.Sigmoid()
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Applies dense xUnit activation with spatial gating.

        Args:
            input: Input tensor as ``torch.Tensor`` with shape [B, C, H, W].

        Returns:
            Output tensor as ``torch.Tensor`` with shape [B, C, H, W] with gated activation.
        """
        return input * self.features(input)
    

# ----- Misc -----
class ArgMax(torch.nn.Module):
    """Finds indices of maximum values along a dimension.

    Args:
        dim: Dimension to find max indices as ``int`` or ``None``.
            Default is ``None`` (entire tensor).
    """

    def __init__(self, dim: int = None):
        super().__init__()
        self.dim = dim

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Computes indices of maximum values.

        Args:
            input: Input tensor as ``torch.Tensor`` of any shape.

        Returns:
            Tensor of max indices as ``torch.Tensor``, shape depends on ``dim``.
        """
        return torch.argmax(input, dim=self.dim)


class Clamp(torch.nn.Module):
    """Clamps a tensor's values within a range of [min, max].

    Args:
        min: Lower bound of the range as ``float``. Default is ``-1.0``.
        max: Upper bound of the range as ``float``. Default is ``1.0``.
    """
    
    def __init__(self, min: float = -1.0,  max: float = 1.0):
        super().__init__()
        self.min = min
        self.max = max
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Clamps tensor values within [min, max].

        Args:
            input: Input tensor as ``torch.Tensor`` of any shape.

        Returns:
            Clamped tensor as ``torch.Tensor`` with same shape as input.
        """
        return torch.clamp(input, min=self.min, max=self.max)


Clip = Clamp


# ----- Utils -----
def to_act_layer(act_layer: Any = torch.nn.ReLU, *args, **kwargs) -> torch.nn.Module:
    """Creates an activation layer from a callable or class.

    Args:
        act_layer: Activation layer class or instance. Default is ``torch.nn.ReLU``.
        *args: Positional arguments for ``act_layer`` instantiation.
        **kwargs: Keyword arguments for ``act_layer`` instantiation.

    Returns:
        Instantiated activation layer as an ``torch.nn.Module``.
    """
    if not act_layer:
        return torch.nn.Identity()
    if callable(act_layer):
        return act_layer(*args, **kwargs)
    return act_layer

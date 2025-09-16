#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements miscellaneous layers."""

__all__ = [
    "ChannelShuffle",
    "Chuncat",
    "Concat",
    "CustomConcat",
    "Embedding",
    "ExtractItem",
    "ExtractItems",
    "Flatten",
    "FlattenSingle",
    "Fold",
    "Foldcut",
    "InterpolateConcat",
    "Join",
    "MLP",
    "Max",
    "PatchMerging",
    "PatchMergingV2",
    "Permute",
    "PixelShuffle",
    "PixelUnshuffle",
    "Shortcut",
    "SoftmaxFusion",
    "Sum",
    "Unflatten",
    "Unfold",
]

from typing import Sequence

import torch
from torch.nn import Embedding, functional as F
from torch.nn.modules.channelshuffle import ChannelShuffle
from torch.nn.modules.flatten import Flatten, Unflatten
from torch.nn.modules.fold import Fold, Unfold
from torch.nn.modules.pixelshuffle import PixelShuffle, PixelUnshuffle
from torchvision.ops.misc import MLP, Permute


# ----- Concat -----
class Concat(torch.nn.Module):
    """Concatenates a list of tensors along a dimension.

    Args:
        dim: Dimension to concatenate along as ``int``. Default is ``1``.
    """
    
    def __init__(self, dim: int = 1):
        super().__init__()
        self.dim = dim
    
    def forward(self, input: list[torch.Tensor]) -> torch.Tensor:
        """Concatenates input tensors along specified dimension.

        Args:
            input: List of tensors to concatenate as ``list[torch.Tensor]``.

        Returns:
            Concatenated tensor as ``torch.Tensor``.
        """
        return torch.cat(input, dim=self.dim)


class CustomConcat(torch.nn.Module):
    """Concatenates module outputs along a dimension, aligning shapes if needed.

    Args:
        dim: Dimension to concatenate along as ``int``.
        *args: Modules to process input as variable positional arguments.
        **kwargs: Additional keyword arguments (unused).
    """
    
    def __init__(self, dim: int, *args, **kwargs):
        super().__init__()
        self.dim = dim
        for idx, module in enumerate(args):
            self.add_module(str(idx), module)
    
    def __len__(self) -> int:
        """Returns number of registered modules."""
        return len(self._modules)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Processes input through modules and concatenates outputs.

        Args:
            input: Tensor to process as ``torch.Tensor``.

        Returns:
            Concatenated tensor as ``torch.Tensor`` with aligned shapes.
        """
        outputs  = [module(input) for module in self._modules.values()]
        shapes_h = [x.shape[2] for x in outputs]
        shapes_w = [x.shape[3] for x in outputs]
        
        min_h, min_w = min(shapes_h), min(shapes_w)
        if all(h == min_h for h in shapes_h) and all(w == min_w for w in shapes_w):
            return torch.cat(outputs, dim=self.dim)
        
        aligned_outputs = []
        for out in outputs:
            diff_h = (out.size(2) - min_h) // 2
            diff_w = (out.size(3) - min_w) // 2
            aligned_outputs.append(out[:, :, diff_h:diff_h + min_h, diff_w:diff_w + min_w])
        
        return torch.cat(aligned_outputs, dim=self.dim)


class Chuncat(torch.nn.Module):
    """Splits tensors into two chunks and concatenates them.

    Args:
        dim: Dimension to split and concatenate along as ``int``. Default is ``1``.
    """

    def __init__(self, dim: int = 1):
        super().__init__()
        self.dim = dim

    def forward(self, input: Sequence[torch.Tensor]) -> torch.Tensor:
        """Chunks each tensor and concatenates results.

        Args:
            input: Sequence of tensors to process as ``Sequence[torch.Tensor]``.

        Returns:
            Concatenated tensor as ``torch.Tensor``.
        """
        y1 = [x.chunk(2, self.dim)[0] for x in input]
        y2 = [x.chunk(2, self.dim)[1] for x in input]
        return torch.cat(y1 + y2, dim=self.dim)


class InterpolateConcat(torch.nn.Module):
    """Concatenates tensors after interpolating to max size.

    Args:
        dim: Dimension to concatenate along as ``int``. Default is ``1``.
    """

    def __init__(self, dim: int = 1):
        super().__init__()
        self.dim = dim

    def forward(self, input: Sequence[torch.Tensor]) -> torch.Tensor:
        """Interpolates tensors to max size and concatenates.

        Args:
            input: Sequence of tensors to process as ``Sequence[torch.Tensor]``.

        Returns:
            Concatenated tensor as ``torch.Tensor``.
        """
        sizes = [x.size() for x in input]
        h, w  = max(s[2] for s in sizes), max(s[3] for s in sizes)
        y     = [F.interpolate(x, size=(h, w)) if (x.size(2) != h or x.size(3) != w) else x for x in input]
        return torch.cat(y, dim=self.dim)


# ----- Extract -----
class ExtractItem(torch.nn.Module):
    """Extracts an item at a specified index from a tensor sequence.

    Args:
        index: Index of the item to extract as ``int``.
    """

    def __init__(self, index: int):
        super().__init__()
        self.index = index

    def forward(self, input: Sequence[torch.Tensor]) -> torch.Tensor:
        """Extracts item at index from sequence.

        Args:
            input: Sequence of tensors or single tensor as
                ``Sequence[torch.Tensor]`` or ``torch.Tensor``.

        Returns:
            Tensor at specified index as ``torch.Tensor`` or input if single tensor,

        Raises:
            TypeError: If input is not a tensor, list, or tuple.
        """
        if isinstance(input, torch.Tensor):
            return input
        if isinstance(input, list | tuple):
            return input[self.index]
        raise TypeError(f"[input] must be tensor, list, or tuple, got {type(input).__name__}.")


class ExtractItems(torch.nn.Module):
    """Extracts multiple items from a tensor sequence by indexes.

    Args:
        indexes: Indexes of items to extract as ``Sequence[int]``.
    """

    def __init__(self, indexes: Sequence[int]):
        super().__init__()
        self.indexes = indexes

    def forward(self, input: Sequence[torch.Tensor]) -> list[torch.Tensor]:
        """Extracts items at specified indexes from sequence.

        Args:
            input: Sequence of tensors or single tensor as
                ``Sequence[torch.Tensor]`` or ``torch.Tensor``.

        Returns:
            List of extracted tensors as ``list[torch.Tensor]``.

        Raises:
            TypeError: If input is not a tensor, list, or tuple.
        """
        if isinstance(input, torch.Tensor):
            return [input]
        if isinstance(input, list | tuple):
            return [input[i] for i in self.indexes]
        raise TypeError(f"[input] must be tensor, list, or tuple, got {type(input).__name__}.")


class Max(torch.nn.Module):
    """Computes maximum along a specified dimension.

    Args:
        dim: Dimension to compute maximum along as ``int``.
        keepdim: Keeps reduced dimension if ``True``. Default is ``False``.
    """

    def __init__(self, dim: int, keepdim: bool = False):
        super().__init__()
        self.dim     = dim
        self.keepdim = keepdim

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Computes max value along specified dimension.

        Args:
            input: Tensor to compute maximum from as ``torch.Tensor``.

        Returns:
            Tensor of max values as ``torch.Tensor``.
        """
        max_values, _ = torch.max(input, dim=self.dim, keepdim=self.keepdim)
        return max_values


# ----- Flatten -----
class FlattenSingle(torch.nn.Module):
    """Flattens a tensor starting from a specified dimension.

    Args:
        dim: Start dimension to flatten from as ``int``. Default is ``1``.
    """

    def __init__(self, dim: int = 1):
        super().__init__()
        self.dim = dim

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Flattens input tensor from specified dimension.

        Args:
            input: Tensor to flatten as ``torch.Tensor``.

        Returns:
            Flattened tensor as ``torch.Tensor``.
        """
        return torch.flatten(input, start_dim=self.dim)


# ----- Fusion -----
class Foldcut(torch.nn.Module):
    """Splits tensor into two chunks and sums them.

    Args:
        dim: Dimension to split and sum along as ``int``. Default is ``0``.
    """

    def __init__(self, dim: int = 0):
        super().__init__()
        self.dim = dim

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Chunks tensor and returns sum of parts.

        Args:
            input: Tensor to process as ``torch.Tensor``.

        Returns:
            Summed tensor as ``torch.Tensor``.
        """
        x1, x2 = input.chunk(2, dim=self.dim)
        return x1 + x2


class Join(torch.nn.Module):
    """Joins multiple features into a list of tensors."""

    def __init__(self):
        super().__init__()

    def forward(self, input: Sequence[torch.Tensor]) -> list[torch.Tensor]:
        """Converts input sequence to a list of tensors.

        Args:
            input: Sequence of tensors to join as ``Sequence[torch.Tensor]``.

        Returns:
            List of tensors as ``list[torch.Tensor]``.
        """
        return list(input)


class Shortcut(torch.nn.Module):
    """Sums the first two tensors in a sequence.

    Args:
        dim: Dimension for tensor operations as ``int`` (unused). Default is ``0``.
    """

    def __init__(self, dim: int = 0):
        super().__init__()
        self.dim = dim

    def forward(self, input: Sequence[torch.Tensor]) -> torch.Tensor:
        """Sums the first two input tensors.

        Args:
            input: Sequence of at least two tensors as ``Sequence[torch.Tensor]``.

        Returns:
            Summed tensor as ``torch.Tensor``.
        """
        return input[0] + input[1]


class SoftmaxFusion(torch.nn.Module):
    """Fuses multiple layers with optional weighted sum.

    Args:
        n: Number of input tensors as ``int``.
        weight: Applies learnable weights if ``True``. Default is ``False``.

    References:
        - https://arxiv.org/abs/1911.09070
    """

    def __init__(self, n: int, weight: bool = False):
        super().__init__()
        self.weight = weight
        self.iter   = range(n - 1)
        if weight:
            self.w = torch.nn.Parameter(-torch.arange(1.0, n) / 2, requires_grad=True)

    def forward(self, input: Sequence[torch.Tensor]) -> torch.Tensor:
        """Computes weighted or unweighted sum of inputs.

        Args:
            input: Sequence of n tensors as ``Sequence[torch.Tensor]``.

        Returns:
            Fused tensor as ``torch.Tensor``.
        """
        y = input[0]
        if self.weight:
            w = torch.sigmoid(self.w) * 2
            for i in self.iter:
                y = y + input[i + 1] * w[i]
        else:
            for i in self.iter:
                y = y + input[i + 1]
        return y


class Sum(torch.nn.Module):
    """Sums all tensors in a sequence."""

    def __init__(self):
        super().__init__()

    def forward(self, input: Sequence[torch.Tensor]) -> torch.Tensor:
        """Sums all input tensors.

        Args:
            input: Sequence of tensors to sum as ``Sequence[torch.Tensor]``.

        Returns:
            Summed tensor as ``torch.Tensor``.
        """
        y = input[0]
        for i in range(1, len(input)):
            y += input[i]
        return y


# ----- Merging -----
class PatchMerging(torch.nn.Module):
    """Merges patches by reducing spatial size and doubling channels.

    Args:
        dim: Number of input channels as ``int``.
        norm: Normalization layer type as ``type[torch.nn.Module]``. Default is ``torch.nn.LayerNorm``.
    """

    def __init__(self, dim: int, norm: type[torch.nn.Module] = torch.nn.LayerNorm, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dim       = dim
        self.reduction = torch.nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm      = norm(4 * dim)

    def _patch_merging_pad(self, x: torch.Tensor) -> torch.Tensor:
        """Pads and merges patches into 4x channels.

        Args:
            x: Tensor of shape [..., H, W, C] as ``torch.Tensor``.

        Returns:
            Tensor of shape [..., H/2, W/2, 4*C] as ``torch.Tensor``.
        """
        h, w, _ = x.shape[-3:]
        x       = F.pad(x, (0, 0, 0, w % 2, 0, h % 2))
        x0      = x[..., 0::2, 0::2, :]
        x1      = x[..., 1::2, 0::2, :]
        x2      = x[..., 0::2, 1::2, :]
        x3      = x[..., 1::2, 1::2, :]
        return torch.cat([x0, x1, x2, x3], dim=-1)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Merges patches and reduces channel dimension.

        Args:
            input: Tensor of shape [B, C, H, W] as ``torch.Tensor``.

        Returns:
            Tensor of shape [B, H/2, W/2, 2*C] as ``torch.Tensor``.
        """
        x = self._patch_merging_pad(input)
        x = self.norm(x)
        return self.reduction(x)


class PatchMergingV2(torch.nn.Module):
    """Merges patches for Swin Transformer V2.

    Args:
        dim: Number of input channels as ``int``.
        norm: Normalization layer type as ``type[nn.Module]``. Default is ``torch.nn.LayerNorm``.
    """

    def __init__(self, dim: int, norm: type[torch.nn.Module] = torch.nn.LayerNorm, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dim       = dim
        self.reduction = torch.nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm      = norm(2 * dim)

    def _patch_merging_pad(self, x: torch.Tensor) -> torch.Tensor:
        """Pads and merges patches into 4x channels.

        Args:
            x: Tensor of shape [..., H, W, C] as ``torch.Tensor``.

        Returns:
            Tensor of shape [..., H/2, W/2, 4*C] as ``torch.Tensor``.
        """
        h, w, _ = x.shape[-3:]
        x       = F.pad(x, (0, 0, 0, w % 2, 0, h % 2))
        x0      = x[..., 0::2, 0::2, :]
        x1      = x[..., 1::2, 0::2, :]
        x2      = x[..., 0::2, 1::2, :]
        x3      = x[..., 1::2, 1::2, :]
        return torch.cat([x0, x1, x2, x3], dim=-1)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Merges patches and applies reduction.

        Args:
            input: Tensor of shape [B, C, H, W] as ``torch.Tensor``.

        Returns:
            Tensor of shape [B, H/2, W/2, 2*C] as ``torch.Tensor``.
        """
        x = self._patch_merging_pad(input)
        x = self.reduction(x)
        return self.norm(x)

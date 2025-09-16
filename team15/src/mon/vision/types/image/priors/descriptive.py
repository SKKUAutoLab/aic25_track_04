#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements image descriptive statistics priors.

This category includes methods that compute statistical properties (e.g., mean,
variance, standard deviation) over local regions of an image.
"""

__all__ = [
    "ImageLocalMean",
    "ImageLocalStdDev",
    "ImageLocalVariance",
    "image_local_mean",
    "image_local_stddev",
    "image_local_variance",
]

import torch

from mon import nn
from mon.nn import functional as F


def image_local_mean(image: torch.Tensor, patch_size: int = 5) -> torch.Tensor:
    """Calculate the local mean of an image using a sliding window.

    Args:
        image: Input image tensor of shape [B, C, H, W] as ``torch.Tensor``.
        patch_size: Size of the sliding window as ``int``. Default is ``5``.

    Returns:
        Local mean tensor as ``torch.Tensor`` with shape [B, C, H, W].
    """
    padding = patch_size // 2
    image   = F.pad(image, (padding, padding, padding, padding), mode="reflect")
    patches = image.unfold(2, patch_size, 1).unfold(3, patch_size, 1)
    return patches.mean(dim=(4, 5))


def image_local_variance(image: torch.Tensor, patch_size: int = 5) -> torch.Tensor:
    """Calculate the local variance of an image using a sliding window.

    Args:
        image: Input image tensor of shape [B, C, H, W] as ``torch.Tensor``.
        patch_size: Size of the sliding window as ``int``. Default is ``5``.

    Returns:
        Local variance tensor as ``torch.Tensor`` with shape [B, C, H, W].
    """
    padding = patch_size // 2
    image   = F.pad(image, (padding, padding, padding, padding), mode="reflect")
    patches = image.unfold(2, patch_size, 1).unfold(3, patch_size, 1)
    mean    = patches.mean(dim=(4, 5))
    return ((patches - mean.unsqueeze(4).unsqueeze(5)) ** 2).mean(dim=(4, 5))


def image_local_stddev(
    image     : torch.Tensor,
    patch_size: int   = 5,
    eps       : float = 1e-9,
) -> torch.Tensor:
    """Calculate the local standard deviation of an image using a sliding window.

    Args:
        image: Input image tensor of shape [B, C, H, W] as ``torch.Tensor``.
        patch_size: Size of the sliding window as ``int``. Default is ``5``.
        eps: Small value to avoid division by zero in sqrt as ``float``.
            Default is ``1e-9``.

    Returns:
        Local standard deviation tensor as ``torch.Tensor`` with shape [B, C, H, W].
    """
    padding        = patch_size // 2
    image          = F.pad(image, (padding, padding, padding, padding), mode="reflect")
    patches        = image.unfold(2, patch_size, 1).unfold(3, patch_size, 1)
    mean           = patches.mean(dim=(4, 5), keepdim=True)
    squared_diff   = (patches - mean) ** 2
    local_variance = squared_diff.mean(dim=(4, 5))
    local_stddev   = torch.sqrt(local_variance + eps)
    return local_stddev


class ImageLocalMean(nn.Module):
    """Calculate the local mean of an image using a sliding window.

    Args:
        patch_size: Size of the sliding window as ``int``. Default is ``5``.
    """
    
    def __init__(self, patch_size: int = 5):
        super().__init__()
        self.patch_size = patch_size
    
    def forward(self, image):
        """Computes the local mean of the input image.

        Args:
            image: Input image tensor of shape [B, C, H, W] as ``torch.Tensor``.

        Returns:
            Local mean tensor as ``torch.Tensor`` with shape [B, C, H, W].
        """
        return image_local_mean(image, self.patch_size)


class ImageLocalVariance(nn.Module):
    """Calculate the local variance of an image using a sliding window.

    Args:
        patch_size: Size of the sliding window as ``int``. Default is ``5``.
    """
    
    def __init__(self, patch_size: int = 5):
        super().__init__()
        self.patch_size = patch_size
    
    def forward(self, image):
        """Computes the local variance of the input image.

        Args:
            image: Input image tensor of shape [B, C, H, W] as ``torch.Tensor``.

        Returns:
            Local variance tensor as ``torch.Tensor`` with shape [B, C, H, W].
        """
        return image_local_variance(image, self.patch_size)


class ImageLocalStdDev(nn.Module):
    """Calculate the local standard deviation of an image using a sliding window.

    Args:
        patch_size: Size of the sliding window as ``int``. Default is ``5``.
        eps: Small value to avoid division by zero in sqrt as ``float``.
            Default is ``1e-9``.
    """
    
    def __init__(self, patch_size: int = 5, eps: float = 1e-9):
        super().__init__()
        self.patch_size = patch_size
        self.eps        = eps
    
    def forward(self, image):
        """Computes the local standard deviation of the input image.

        Args:
            image: Input image tensor of shape [B, C, H, W] as ``torch.Tensor``.

        Returns:
            Local standard deviation tensor as ``torch.Tensor`` with shape [B, C, H, W].
        """
        return image_local_stddev(image, self.patch_size, self.eps)

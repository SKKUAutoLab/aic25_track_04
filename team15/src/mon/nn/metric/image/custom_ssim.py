#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements custom SSIM loss for ESDNet paper.

References:
    - https://github.com/MingTian99/ESDNet/blob/master/utils/image_utils.py
"""

__all__ = [
    "SSIM",
]

import math

import torch
import torch.nn.functional as F


def gaussian(window_size: int, sigma: float) -> torch.Tensor:
    """Creates a 1D Gaussian kernel.

    Args:
        window_size: Size of the Gaussian window as ``int``.
        sigma: Standard deviation of the Gaussian as ``float``.

    Returns:
        Normalized 1D Gaussian tensor as ``torch.Tensor``.
    """
    gauss = torch.tensor([
        math.exp(-((x - window_size // 2) ** 2) / (2 * sigma ** 2))
        for x in range(window_size)
    ])
    return gauss / gauss.sum()


def create_window(window_size: int, channel: int, sigma: float = 1.5) -> torch.Tensor:
    """Creates a 2D Gaussian window tensor.

    Args:
        window_size: Size of the square window as ``int``.
        channel: Number of channels for the window as ``int``.
        sigma: Standard deviation of the Gaussian as ``float``. Default is ``1.5``.

    Returns:
        2D Gaussian window tensor as ``torch.Tensor`` with shape
        [channel, 1, window_size, window_size].
    """
    window_1d = gaussian(window_size, sigma).unsqueeze(1)
    window_2d = window_1d.mm(window_1d.t()).float().unsqueeze(0).unsqueeze(0)
    window    = window_2d.expand(channel, 1, window_size, window_size).contiguous()
    return window


def ssim(
    image1      : torch.Tensor,
    image2      : torch.Tensor,
    window      : torch.Tensor,
    window_size : int  = 11,
    channel     : int  = 1,
    k           : tuple[float, float] = (0.01, 0.03),
    size_average: bool = True
) -> torch.Tensor:
    """Computes Structural Similarity Index (SSIM) between two images.

    Args:
        image1: First image tensor as ``torch.Tensor`` with shape [B, C, H, W].
        image2: Second image tensor as ``torch.Tensor`` with shape [B, C, H, W].
        window: Gaussian window tensor as ``torch.Tensor``.
        window_size: Size of the window as ``int``. Default is ``11``.
        channel: Number of channels as ``int``. Default is ``1``.
        k: Stability constants (k1, k2) as ``tuple[float, float]``.
            Default is ``(0.01, 0.03)``.
        size_average: Averages SSIM over all pixels if ``True``. Default is ``True``.

    Returns:
        SSIM value or map as ``torch.Tensor``.
    """
    mu1 = F.conv2d(image1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(image2, window, padding=window_size // 2, groups=channel)

    mu1_sq  = mu1.pow(2)
    mu2_sq  = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(image1 * image1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(image2 * image2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12   = F.conv2d(image1 * image2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = k[0] ** 2
    C2 = k[1] ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean() if size_average else ssim_map.mean(1).mean(1).mean(1)


class SSIM(torch.nn.Module):
    """Structural Similarity Index (SSIM) module for image comparison.

    Args:
        window_size: Size of the Gaussian window as ``int``. Default is ``11``.
        channel: Number of input channels as ``int``. Default is ``1``.
        k: Stability constants (k1, k2) as ``tuple[float, float]``.
            Default is ``(0.01, 0.03)``.
        size_average: Averages SSIM over all pixels if ``True``. Default is ``True``.
    """

    def __init__(
        self,
        window_size : int  = 11,
        channel     : int  = 1,
        k           : tuple[float, float] = (0.01, 0.03),
        size_average: bool = True
    ):
        super().__init__()
        self.window_size  = window_size
        self.size_average = size_average
        self.channel      = channel
        self.k            = k
        self.window       = create_window(window_size, self.channel)

    def forward(self, image1: torch.Tensor, image2: torch.Tensor) -> torch.Tensor:
        """Computes SSIM between two images.

        Args:
            image1: First image tensor as ``torch.Tensor`` with shape [B, C, H, W].
            image2: Second image tensor as ``torch.Tensor`` with shape [B, C, H, W].

        Returns:
            SSIM value or map as ``torch.Tensor``
        """
        _, channel, _, _ = image1.size()
        if channel == self.channel and self.window.data.type() == image1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            if image1.is_cuda:
                window = window.cuda(image1.get_device())
            window = window.type_as(image1)
            self.window  = window
            self.channel = channel

        return ssim(
            image1       = image1,
            image2       = image2,
            window       = window,
            window_size  = self.window_size,
            channel      = channel,
            k            = self.k,
            size_average = self.size_average
        )

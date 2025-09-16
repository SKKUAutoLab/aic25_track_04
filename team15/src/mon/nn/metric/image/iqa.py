#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements image quality assessment metrics.

References:
    - https://github.com/VinAIResearch/PSENet-Image-Enhancement/blob/main/source/iqa.py
"""

__all__ = [
    "ImageQualityAssessment",
]

import torch


class ImageQualityAssessment(torch.nn.Module):
    """Assesses image quality based on exposedness, contrast, and saturation.

    Args:
        exposed_level: Target exposure level. Default is ``0.5``.
        pool_size: Size of pooling window. Default is ``25``.

    References:
        - https://github.com/VinAIResearch/PSENet-Image-Enhancement/blob/main/source/iqa.py
    """

    def __init__(self, exposed_level: float = 0.5, pool_size: int = 25):
        super().__init__()
        self.exposed_level = exposed_level
        self.pool_size     = pool_size
        self.mean_pool     = torch.nn.Sequential(
            torch.nn.ReflectionPad2d(self.pool_size // 2),
            torch.nn.AvgPool2d(self.pool_size, stride=1)
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Computes image quality score.

        Args:
            images: Input image tensor as ``torch.Tensor`` with shape [B, C, H, W].

        Returns:
            Quality score tensor as ``torch.Tensor`` with shape [B, 1].
        """
        max_rgb     = torch.max(images, dim=1, keepdim=True)[0]
        min_rgb     = torch.min(images, dim=1, keepdim=True)[0]
        saturation  = (max_rgb - min_rgb + 1 / 255.0) / (max_rgb + 1 / 255.0)
        mean_rgb    = self.mean_pool(images).mean(dim=1, keepdim=True)
        exposedness = torch.abs(mean_rgb - self.exposed_level) + 1 / 255.0
        contrast    = self.mean_pool(images * images).mean(dim=1, keepdim=True) - mean_rgb ** 2
        return torch.mean((saturation * contrast) / exposedness, dim=[1], keepdim=True)

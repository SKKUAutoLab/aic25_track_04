#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements attention priors (saliency and focus maps).

This category includes methods that highlight important regions in an image, often used
in deep learning to guide processing or interpretation (e.g., attention maps in neural
networks).
"""

__all__ = [
    "BrightnessAttentionMap",
    "brightness_attention_map",
]

import cv2
import kornia
import numpy as np
import torch

from mon import nn
from mon.nn import _size_2_t
from mon.vision.types.image import utils


def brightness_attention_map(
    image        : torch.Tensor | np.ndarray,
    gamma        : float     = 2.5,
    denoise_ksize: _size_2_t = None,
) -> torch.Tensor:
    """Get the Brightness Attention Map (BAM) prior from an RGB image.

    This is a self-attention map extracted from the V-channel of a low-light image,
    multiplied to convolutional activations of all layers in the enhancement network.
    Brighter regions are given lower weights to avoid over-saturation, while preserving
    image details and enhancing contrast in dark regions effectively.

    Equation: `I_{attn} = (1 - I_{V})^{\gamma}`, where `\gamma \geq 1`.

    Args:
        image: RGB image as:
            - ``torch.Tensor`` in [B, C, H, W] format with data in range [0.0, 1.0].
            - ``numpy.ndarray`` in [H, W, C] format with data in range [0, 255].
        gamma: Parameter controlling the curvature of the map as ``float``.
            Default is ``2.5``.
        denoise_ksize: Window size for denoising operation as ``int`` or
            ``tuple[int, int]`` or ``None``. Default is ``None``.

    Returns:
        Brightness enhancement map as ``torch.Tensor`` or ``numpy.ndarray`` matching
        input type.
    """
    if isinstance(image, torch.Tensor):
        if denoise_ksize:
            image = kornia.filters.median_blur(image, denoise_ksize)
            # image = kornia.filters.bilateral_blur(image, denoise_ksize, 0.1, (1.5, 1.5))
        hsv = kornia.color.rgb_to_hsv(image)
        v   = utils.image_channel(image=hsv, index=(2, 3), keep_dim=True)  # hsv[:, 2:3, :, :]
        bam = torch.pow((1 - v), gamma)
    elif isinstance(image, np.ndarray):
        if denoise_ksize:
            image = cv2.medianBlur(image, denoise_ksize)
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        if hsv.dtype != np.float64:
            hsv  = hsv.astype("float64")
            hsv /= 255.0
        v   = utils.image_channel(image=hsv, index=(2, 3), keep_dim=True)  # hsv[:, :, 2:3]
        bam = np.power((1 - v), gamma)
    else:
        raise TypeError(f"[image] must be a torch.Tensor or np.ndarray, got {type(image)}.")
    return bam


class BrightnessAttentionMap(nn.Module):
    """Get the Brightness Attention Map (BAM) prior from an RGB image.

    This is a self-attention map extracted from the V-channel of a low-light image,
    multiplied to convolutional activations of all layers in the enhancement network.
    Brighter regions are given lower weights to avoid over-saturation, while preserving
    image details and enhancing contrast in dark regions effectively.

    Args:
        gamma: Parameter controlling the curvature of the map as ``float``.
            Default is ``2.5``
        denoise_ksize: Window size for denoising operation as ``int`` or
            ``tuple[int, int]`` or ``None``. Default is ``None``.
    """
    
    def __init__(
        self,
        gamma        : float     = 2.5,
        denoise_ksize: _size_2_t = None
    ):
        super().__init__()
        self.gamma         = gamma
        self.denoise_ksize = denoise_ksize
    
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Computes the Brightness Attention Map for the input image.

        Args:
            image: Input image tensor of shape [B, C, H, W] as ``torch.Tensor``  with
                data in range [0.0, 1.0].

        Returns:
            Brightness attention map tensor as ``torch.Tensor`` with shape [B, C, H, W].
        """
        return brightness_attention_map(image, self.gamma, self.denoise_ksize)

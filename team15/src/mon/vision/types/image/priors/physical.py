#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements image physical priors.

This category encompasses models about image properties based on physical (model-based).
"""

__all__ = [
    "atmospheric_point_spread_function",
    "atmospheric_prior",
]

import numpy as np
import torch

from mon.nn import _size_2_t, functional as F
from mon.vision.types.image.priors import statistical


def atmospheric_point_spread_function(
    image: torch.Tensor | np.ndarray,
    q    : float = 0.2,
    T    : float = 1.2,
    k    : float = 0.5,
) -> torch.Tensor | np.ndarray:
    """Get the atmospheric point spread function (APSF) from an RGB image.
    
    Args:
        image: An RGB image as ``torch.Tensor`` [B, C, H, W] in range [0.0, 1.0] or
            ``numpy.ndarray`` [H, W, C] in range [0, 255].
        q: Forward scattering param.
            - ``0.00-0.20``: air
            - ``0.20-0.70``: aerosol
            - ``0.70-0.80``: haze
            - ``0.80-0.85``: mist
            - ``0.85-0.90``: fog
            - ``0.90-1.00``: rain
            Default is ``0.2``.
        T: Optical thickness. Possibly: [0.7, 1.2, 4]. According to Narasimhan in
            CVPR03 paper: T = sigma * R (extinctiontion coefficition * distance or depth),
            which is the same \beta d in haze modelling.
            Default is ``1.2``.
        k: Conversion param for kernel. Default is ``0.5``.
    
    Returns:
        APSF as ``torch.Tensor`` or ``numpy.ndarray``.
        
    References:
        - https://github.com/jinyeying/night-enhancement/blob/main/glow_rendering_code/repro_ICCV2007_Fig5.m
    """
    from scipy.special import gamma
    from scipy.ndimage import convolve
    
    def A(p: float, sigma: float):
        return np.sqrt(sigma ** 2 * gamma(1 / p) / gamma(3 / p))
    
    if isinstance(image, torch.Tensor):
        p       = k * T        # Eq (9)
        sigma   = (1 - q) / q  # Eq (1)
        # Generate APSF kernel
        x       = torch.linspace(-6, 6, 100)
        XX, YY  = torch.meshgrid(x, x, indexing='ij')
        A_val   = A(p, sigma)
        APSF2D  = torch.exp(-((XX ** 2 + YY ** 2) ** (p / 2)) / abs(A_val) ** p) / (2 * gamma(1 + 1 / p) * A_val) ** 2
        APSF2D /= torch.sum(APSF2D)
        # Apply convolution
        kernel  = APSF2D.unsqueeze(0).unsqueeze(0)   # Shape: (1, 1, H, W)
        kernel  = kernel.repeat(3, 1, 1, 1)          # Shape: (3, 1, H, W) for RGB channels
        apsf    = F.conv2d(image, kernel, padding="same", groups=3)
        apsf    = torch.clamp(apsf, 0, 1)  # Ensure valid pixel range
    elif isinstance(image, np.ndarray):
        p       = k * T        # Eq (9)
        sigma   = (1 - q) / q  # Eq (1)
        # Generate APSF kernel
        x       = np.linspace(-6, 6, 100)
        XX, YY  = np.meshgrid(x, x)
        A_val   = A(p, sigma)
        APSF2D  = np.exp(-((XX ** 2 + YY ** 2) ** (p / 2)) / abs(A_val) ** p) / (2 * gamma(1 + 1 / p) * A_val) ** 2
        APSF2D  = APSF2D / np.sum(APSF2D)  # Normalize kernel
        # Apply convolution
        h, w, c = image.shape
        apsf    = np.zeros_like(image)
        for i in range(c):  # Apply per channel
            apsf[:, :, i] = convolve(image[:, :, i], APSF2D, mode="reflect")
    else:
        raise ValueError(f"[image] type [{type(image)}] not supported.")
    return apsf


def atmospheric_prior(
    image      : np.ndarray,
    kernel_size: _size_2_t = 15,
    p          : float     = 0.0001
) -> np.ndarray:
    """Get the atmospheric light in an RGB image.

    Args:
        image: RGB image as ``numpy.ndarray`` in [H, W, C] format with data in range [0, 255].
        kernel_size: Window size for the dark channel as ``int`` or ``tuple[int, int]``.
            Default is ``15``.
        p: Percentage of pixels for estimating atmospheric light as ``float``.
            Default is ``0.0001``.

    Returns:
        3-element array of atmospheric light as ``numpy.ndarray`` in range [0, 255] for each channel.
    """
    if not isinstance(image, np.ndarray):
        raise ValueError(f"[image] must be numpy.ndarray, got {type(image)}.")
    
    image      = image.transpose(1, 2, 0)
    # Reference CVPR09, 4.4
    dark       = statistical.dark_channel_prior_paper(image=image, kernel_size=kernel_size)
    m, n       = dark.shape
    flat_i     = image.reshape(m * n, 3)
    flat_dark  = dark.ravel()
    search_idx = (-flat_dark).argsort()[:int(m * n * p)]  # find top M * N * p indexes
    # Return the highest intensity for each channel
    return np.max(flat_i.take(search_idx, axis=0), axis=0)

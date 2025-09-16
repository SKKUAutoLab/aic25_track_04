#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements image statistical priors.

This category encompasses assumptions about image properties based on statistical data
(data-driven).
"""

__all__ = [
    "blur_spot_prior",
    "bright_channel_prior",
    "bright_spot_prior",
    "dark_channel_prior",
    "dark_channel_prior_paper",
]

import cv2
import kornia
import numpy as np
import torch

from mon import core
from mon.nn import _size_2_t


def blur_spot_prior(image: np.ndarray, threshold: int = 250) -> bool:
    """Detects blur in an image based on Laplacian variance and bright spot thresholding.

    Args:
        image: Input image as ``numpy.ndarray`` in BGR format with shape [H, W, 3].
        threshold: Variance threshold for blur detection as ``int``. Default is ``250``.

    Returns:
        ``True`` if the image is blurry (Laplacian variance < threshold),
        ``False`` otherwise.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError(f"[image] must be numpy.ndarray, got {type(image)}.")
    
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply binary thresholding for bright spot detection
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    # Apply Laplacian filter for edge detection
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    # Calculate maximum intensity and variance
    laplacian_var = laplacian.var()
    # Check blur condition based on variance of Laplacian image
    is_blur = True if laplacian_var < threshold else False
    return is_blur


def bright_spot_prior(image: np.ndarray) -> bool:
    """Detects bright spots in an image based on variance of a binary thresholded
    grayscale image.

    Args:
        image: Input image as ``numpy.ndarray`` in BGR format with shape [H, W, 3].

    Returns:
        ``True`` if bright spots are detected (variance between 5000 and 8500),
        ``False`` otherwise.

    Raises:
        TypeError: If ``image`` is not a ``numpy.ndarray``.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError(f"[image] must be numpy.ndarray, got {type(image)}.")
    
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply binary thresholding for bright spot detection
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    # Calculate maximum intensity and variance
    binary_var = binary.var()
    # Check bright spot condition based on variance of binary image
    is_bright = True if 5000 < binary_var < 8500 else False
    return is_bright


def bright_channel_prior(
    image      : torch.Tensor | np.ndarray,
    kernel_size: _size_2_t
) -> torch.Tensor | np.ndarray:
    """Gets bright channel prior from an RGB image.

    Args:
        image: RGB image as ``torch.Tensor`` [B, C, H, W] or ``numpy.ndarray`` [H, W, C].
        kernel_size: Window size as int or tuple.

    Returns:
        Bright channel prior as ``torch.Tensor`` or ``numpy.ndarray``.
    """
    kernel_size = core.to_2tuple(kernel_size)
    if isinstance(image, torch.Tensor):
        bright_channel = torch.max(image, dim=1)[0]
        kernel         = torch.ones(kernel_size[0], kernel_size[0])
        bcp            = kornia.morphology.erosion(bright_channel, kernel)
    elif isinstance(image, np.ndarray):
        bright_channel = np.max(image, axis=2)
        kernel         = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
        bcp            = cv2.erode(bright_channel, kernel)
    else:
        raise ValueError(f"[image] must be torch.Tensor or numpy.ndarray, got {type(image)}.")
    return bcp


def dark_channel_prior(
    image      : torch.Tensor | np.ndarray,
    kernel_size: int
) -> torch.Tensor | np.ndarray:
    """Gets dark channel prior from an RGB image.

    Args:
        image: RGB image as ``torch.Tensor`` [B, C, H, W] or ``numpy.ndarray`` [H, W, C].
        kernel_size: Window size as ``int``.

    Returns:
        Dark channel prior as ``torch.Tensor`` or ``numpy.ndarray``.
    """
    kernel_size = core.to_2tuple(kernel_size)
    if isinstance(image, torch.Tensor):
        dark_channel = torch.min(image, dim=1)[0]
        kernel       = torch.ones(kernel_size[0], kernel_size[1])
        dcp          = kornia.morphology.erosion(dark_channel, kernel)
    elif isinstance(image, np.ndarray):
        dark_channel = np.min(image, axis=2)
        kernel       = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
        dcp          = cv2.erode(dark_channel, kernel)
    else:
        raise ValueError(f"[image] must be torch.Tensor or numpy.ndarray, "
                         f"got {type(image)}.")
    return dcp


def dark_channel_prior_paper(
    image      : torch.Tensor | np.ndarray,
    kernel_size: _size_2_t,
) -> torch.Tensor | np.ndarray:
    """Gets dark channel prior from an RGB image (from paper).

    Args:
        image: RGB image as ``torch.Tensor`` [B, C, H, W] or ``numpy.ndarray`` [H, W, C].
        kernel_size: Window size as ``int``.

    Returns:
        Dark channel prior as ``torch.Tensor`` or ``numpy.ndarray``.
    """
    m, n, _ = image.shape
    w       = kernel_size
    padded  = np.pad(image, ((w // 2, w // 2), (w // 2, w // 2), (0, 0)), "edge")
    dcp     = np.zeros((m, n))
    for i, j in np.ndindex(dcp.shape):
        dcp[i, j] = np.min(padded[i:i + w, j:j + w, :])  # CVPR09, eq.5
    return dcp

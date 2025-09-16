#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements image feature priors (structural and semantic priors).

This category involves identifying and isolating structural or semantic features
(e.g., edges, boundaries) in an image.
"""

__all__ = [
    "BoundaryAwarePrior",
    "boundary_aware_prior",
]

import cv2
import kornia
import numpy as np
import torch

from mon import nn
from mon.vision.types.image import utils


def boundary_aware_prior(
    image      : torch.Tensor | np.ndarray,
    eps        : float = 0.05,
    as_gradient: bool  = False,
    normalized : bool  = False,
) -> torch.Tensor | np.ndarray:
    """Get the boundary prior from an RGB or grayscale image.

    Args:
        image: RGB or grayscale image as:
            - ``torch.Tensor`` in [B, C, H, W] format with data in range [0.0, 1.0].
            - ``numpy.ndarray`` in [H, W, C] format with data in range [0, 255].
        eps: Threshold to remove weak edges as ``float``. Default is ``0.05``.
        as_gradient: Return the gradient image if ``True``. Default is ``False``.
        normalized: L1 norm of the kernel is set to 1 if ``True``. Default is ``False``.

    Returns:
        Boundary-aware prior as ``torch.Tensor`` or ``numpy.ndarray``:
            - Binary image if ``as_gradient=False``.
            - Gradient image if ``as_gradient=True``.

    Raises:
        ValueError: If ``image`` type is not supported.
    """
    if isinstance(image, torch.Tensor):
        gradient = kornia.filters.sobel(image, normalized=normalized, eps=1e-6)
        g_max    = torch.max(gradient)
        gradient = gradient / g_max
        boundary = (gradient > eps).float()
    elif isinstance(image, np.ndarray):
        if utils.is_image_colored(image):
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        from mon.vision.filtering import sobel_filter
        gradient = sobel_filter(image, kernel_size=3)
        g_max    = np.max(gradient)
        gradient = gradient / g_max
        boundary = (gradient > eps).float()
        return boundary
    else:
        raise TypeError(f"[image] must be a torch.Tensor or np.ndarray, got {type(image)}.")
    
    # return boundary, gradient
    if as_gradient:
        return gradient
    else:
        return boundary


class BoundaryAwarePrior(nn.Module):
    """Get the boundary prior from an RGB or grayscale image.

    Args:
        eps: Threshold to remove weak edges as ``float``. Default is ``0.05``.
        normalized: L1 norm of the kernel is set to ``1`` if ``True`` as ``bool``.
            Default is ``False``.
    """
    
    def __init__(self, eps: float = 0.05, normalized: bool = False):
        super().__init__()
        self.eps        = eps
        self.normalized = normalized
    
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Computes the boundary prior for the input image.

        Args:
            image: Input image tensor of shape [B, C, H, W] as ``torch.Tensor`` with
                data in range [0.0, 1.0].

        Returns:
            Boundary prior tensor as ``torch.Tensor`` with shape [B, C, H, W].
        """
        return boundary_aware_prior(image, self.eps, self.normalized)

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements general-purpose utilities for depth tasks.

Common Tasks:
    - Format conversions.
"""

__all__ = [
    "depth_to_color",
]

import cv2
import numpy as np

from mon.vision.types import image as I


# ----- Conversion -----
def depth_to_color(
    depth    : np.ndarray,
    color_map: int = cv2.COLORMAP_JET,
    use_rgb  : bool = False
) -> np.ndarray:
    """Converts a depth map to a color-coded image.

    Args:
        depth: Depth map as ``numpy.ndarray`` in [H, W, 1] format.
        color_map: Color map for the depth map. Default is ``cv2.COLORMAP_JET``.
        use_rgb: Convert to RGB format if ``True``. Default is ``False``.
    
    Returns:
        Color-coded depth map as ``numpy.ndarray`` in [H, W, 3] format.
    
    Raises:
        TypeError: If ``depth`` is not a ``numpy.ndarray``.
    """
    if not isinstance(depth, np.ndarray):
        raise TypeError(f"[depth] must be a numpy.ndarray, got {type(depth)}.")
    depth = np.uint8(255 * depth) if I.is_image_normalized(depth) else depth
    depth = cv2.applyColorMap(depth, color_map)
    return cv2.cvtColor(depth, cv2.COLOR_BGR2RGB) if use_rgb else depth

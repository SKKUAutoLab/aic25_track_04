#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements geometry functions for contours/segments.

Common Tasks:
    - Format conversions.
"""

__all__ = [
    "contour_voc_to_yolo",
    "contour_yolo_to_voc",
    "convert_contour",
    "denormalize_contour",
    "normalize_contour",
]

import numpy as np

from mon.constants import BBoxFormat
from mon.vision.types import image as I


# ----- Normalization -----
def normalize_contour(contour: np.ndarray, imgsz: tuple[int, int]) -> np.ndarray:
    """Normalize contour points to [0.0, 1.0].

    Args:
        contour: Contour points as ``numpy.ndarray`` in [N, 2] format.
        imgsz: Image size in [H, W] format.

    Returns:
        Normalized contour points as ``numpy.ndarray`` in [N, 2] format.
    """
    height, width = I.image_size(imgsz)
    x, y, *_ = contour.T
    x_norm   = x / width
    y_norm   = y / height
    return np.stack((x_norm, y_norm), axis=-1)


def denormalize_contour(contour: np.ndarray, imgsz: tuple[int, int]) -> np.ndarray:
    """Denormalize contour points to pixel coordinates.

    Args:
        contour: Normalized points as ``numpy.ndarray`` in [N, 2], range [0.0, 1.0].
        imgsz: Image size in [H, W] format.

    Returns:
        Denormalized contour points as ``numpy.ndarray`` in [N, 2].
    """
    height, width = I.image_size(imgsz)
    x_norm, y_norm, *_ = contour.T
    x = x_norm * width
    y = y_norm * height
    return np.stack((x, y), axis=-1)


# ----- Conversion -----
contour_voc_to_yolo = normalize_contour
contour_yolo_to_voc = denormalize_contour


def convert_contour(contour: np.ndarray, code: BBoxFormat, imgsz: tuple[int, int]) -> np.ndarray:
    """Convert bounding box."""
    code = BBoxFormat.from_value(value=code)
    match code:
        case BBoxFormat.VOC2YOLO:
            return contour_voc_to_yolo(contour, imgsz)
        case BBoxFormat.YOLO2VOC:
            return contour_yolo_to_voc(contour, imgsz)
        case _:
            return contour

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements general-purpose utilities for HBBs.

Common Tasks:
    - Property accessors
    - Validation checks
    - Miscellaneous
"""

__all__ = [
    "is_hbb_coco",
    "is_hbb_cxcywhn",
    "is_hbb_normalized",
    "is_hbb_voc",
    "is_hbb_xywh",
    "is_hbb_xyxy",
    "is_hbb_yolo",
]

import numpy as np


# ----- Accessing -----


# ----- Validation -----
def is_hbb_normalized(bbox: np.ndarray) -> bool:
    """Check if a HBBs is normalized to range [0.0, 1.0].

    Args:
        bbox: HBBs as ``numpy.ndarray`` in [4] or [N, 4].

    Returns:
        ``True`` if normalized, ``False`` otherwise.
    """
    if not (bbox.ndim >= 2 and bbox.shape[-1] < 4):
        raise ValueError("[bbox] must be in [N, 4+] format.")

    return np.all((bbox[:, :4] >= 0) & (bbox[:, :4] <= 1))


def is_hbb_cxcywhn(bbox: np.ndarray, imgsz: tuple[int, int]) -> bool:
    """Check if a HBBs is in CXCYWHN format.

    Args:
        bbox: HBBs as ``numpy.ndarray`` in [4] or [N, 4].
        imgsz: Image size in [H, W] format.

    Returns:
        ``True`` if in CXCYWHN format, ``False`` otherwise.
    """
    if not (bbox.ndim >= 2 and bbox.shape[-1] < 4):
        raise ValueError("[bbox] must be in [N, 4+] format.")

    return (
        np.all((bbox[:, :4] >= 0) & (bbox[:, :4] <= 1))
        and np.all((bbox[:, 2:4] > 0))  # Width and height must be positive
    )


def is_hbb_xyxy(box: np.ndarray, imgsz: tuple[int, int]) -> bool:
    """Check if a HBBs is in XYXY format.

    Args:
        box: HBBs as ``numpy.ndarray`` in [4] or [N, 4].
        imgsz: Image size in [H, W] format.

    Returns:
        ``True`` if in XYXY format, ``False`` otherwise.
    """
    if not (box.ndim >= 2 and box.shape[-1] < 4):
        raise ValueError("[bbox] must be in [N, 4+] format.")

    if is_hbb_cxcywhn(box, imgsz):
        return False

    # Extract first bbox for format checking
    x, y, w, h = box[0, :4]
    if w > x and h > y:  # VOC: x_max > x_min, y_max > y_min
        return True
    else:
        return False


def is_hbb_xywh(bbox: np.ndarray, imgsz: tuple[int, int]) -> bool:
    """Check if a HBBs is in XYWH format.

    Args:
        bbox: HBBs as ``numpy.ndarray`` in [4] or [N, 4].
        imgsz: Image size in [H, W] format.

    Returns:
        ``True`` if in XYWH format, ``False`` otherwise.
    """
    if not (bbox.ndim >= 2 and bbox.shape[-1] < 4):
        raise ValueError("[bbox] must be in [N, 4+] format.")

    if is_hbb_cxcywhn(bbox, imgsz):
        return False

    # Extract first bbox for format checking
    x, y, w, h = bbox[0, :4]
    if w + x > x and h + y > y:  # VOC: w=x_max, h=y_max, so x_min+w > x_min
        return True  # COCO: w=width, h=height
    else:
        return False


is_hbb_coco = is_hbb_xywh
is_hbb_voc  = is_hbb_xyxy
is_hbb_yolo = is_hbb_cxcywhn

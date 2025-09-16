#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements cropping transformation."""

__all__ = [
    "center_crop_image_and_hbbs",
    "split_image_and_hbbs",
]

import math

import numpy as np

from mon.constants import BBoxFormat
from mon.nn import _size_2_t
from mon.vision import types


# ----- Splitting -----
def split_image_and_hbbs(
    image: np.ndarray,
    bbox : np.ndarray,
    n    : int = 2
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Split an image into ``n`` equal parts and adjust YOLO-format hbbs accordingly.

    Args:
        image: Image as ``numpy.ndarray`` [H, W, C].
        bbox: HBBs as ``numpy.ndarray`` in [N, 4+], CXCYWHN format (YOLO), normalized.
        n: Number of parts to split into (positive integer). Default is ``2``.

    Raises:
        ValueError: If inputs are invalid (e.g., image shape, bboxes, n).
    """
    if not isinstance(image, np.ndarray) or len(image.shape) != 3:
        raise ValueError(f"[image] must be a 3D numpy array [H, W, C], got {image.shape}.")
    if not isinstance(bbox, np.ndarray)  or bbox.ndim != 2 or bbox.shape[1] < 4:
        raise ValueError(f"[bboxes] must be a 2D numpy array [N, M] with M >= 4, got {bbox.shape}.")
    if n < 1:
        raise ValueError(f"[n] must be a positive integer, got {n}.")

    h, w = types.image_size(image)
    if n > h * w:
        raise ValueError(f"[n] ({n}) exceeds image pixel count ({h * w}).")

    # Determine orientation
    is_portrait = h > w

    # Determine rows and cols
    if n == 1:
        rows, cols = 1, 1
    elif n == 2:
        # Explicitly set grid for N=2 based on orientation
        rows = 2 if is_portrait else 1
        cols = 1 if is_portrait else 2
    else:
        # General case: start with approximate square grid
        rows = math.ceil(math.sqrt(n))
        cols = math.ceil(n / rows)
        # Adjust to ensure rows * cols = n, prioritizing orientation
        candidates = []
        for r in range(1, n + 1):
            c = math.ceil(n / r)
            if r * c == n:
                candidates.append((r, c))
        if not candidates:
            raise ValueError(f"Cannot find valid rows and cols for n={n}")
        # Select grid based on orientation
        if is_portrait:
            # Prefer more rows (taller sub-images)
            rows, cols = max(candidates, key=lambda x: x[0] / x[1])
        else:
            # Prefer more cols (wider sub-images)
            rows, cols = max(candidates, key=lambda x: x[1] / x[0])

    # Compute sub-images and adjust bboxes
    sub_h      = h // rows
    sub_w      = w // cols
    sub_images = []
    sub_bboxes = []

    for i in range(rows):
        for j in range(cols):
            if len(sub_images) >= n:
                break
            # Compute sub-image boundaries
            y_start   = i * sub_h
            y_end     = min((i + 1) * sub_h, h)
            x_start   = j * sub_w
            x_end     = min((j + 1) * sub_w, w)
            sub_image = image[y_start:y_end, x_start:x_end]
            if sub_image.size == 0:
                continue
            sub_images.append(sub_image)

            # Adjust bboxes
            sub_bboxes_i     = []
            sub_h_i, sub_w_i = sub_image.shape[:2]
            for b in bbox:
                cx_n, cy_n, w_n, h_n = b[:4]
                cx = cx_n * w
                cy = cy_n * h
                x1 = cx - w_n * w / 2
                x2 = cx + w_n * w / 2
                y1 = cy - h_n * h / 2
                y2 = cy + h_n * h / 2

                # Check if bbox intersects sub-image
                if x2 > x_start and x1 < x_end and y2 > y_start and y1 < y_end:
                    # Compute new bbox in sub-image coordinates
                    x1_new   = max(x1, x_start) - x_start
                    x2_new   = min(x2, x_end)   - x_start
                    y1_new   = max(y1, y_start) - y_start
                    y2_new   = min(y2, y_end)   - y_start
                    cx_n_new = (x1_new + x2_new) / 2 / sub_w_i
                    cy_n_new = (y1_new + y2_new) / 2 / sub_h_i
                    w_n_new  = (x2_new - x1_new) / sub_w_i
                    h_n_new  = (y2_new - y1_new) / sub_h_i
                    if w_n_new > 0 and h_n_new > 0:
                        bbox_new = np.concatenate(([cx_n_new, cy_n_new, w_n_new, h_n_new], b[4:]))
                        sub_bboxes_i.append(bbox_new)
            sub_bboxes.append(np.array(sub_bboxes_i) if sub_bboxes_i else np.zeros((0, bbox.shape[1]), dtype=np.float32))

    # Pad with empty sub-images/bboxes if needed
    while len(sub_images) < n:
        sub_images.append(np.zeros_like(sub_images[0]))
        sub_bboxes.append(np.zeros((0, bbox.shape[1]), dtype=np.float32))

    return sub_images, sub_bboxes


# ----- Cropping -----
def center_crop_image_and_hbbs(
    image: np.ndarray,
    bbox : np.ndarray,
    imgsz: _size_2_t,
) -> tuple[np.ndarray, np.ndarray]:
    """Center crop an image and adjust YOLO-format hbbs accordingly.

    Args:
        image: Image as ``numpy.ndarray`` [H, W, C].
        bbox: HBBs as ``numpy.ndarray`` in [N, 4+], CXCYWHN format (YOLO), normalized.
        imgsz: Target size as [H, W] format.
    """
    h0, w0 = types.image_size(image)
    h1, w1 = types.image_size(imgsz)

    if h1 > h0 or w1 > w0:
        raise ValueError(f"Target size {imgsz} exceeds original image size {image.shape[:2]}.")

    # Convert bbox to XYXY format
    bbox = types.convert_hbb(bbox, fmt=BBoxFormat.CXCYWHN2XYXY, imgsz=(h0, w0))

    # Calculate crop region (center of image)
    x_start = max(0, (w0 - w1) // 2)
    y_start = max(0, (h0 - h1) // 2)
    x_end   = x_start + w1
    y_end   = y_start + h1

    # Crop the image
    cropped_image = image[y_start:y_end, x_start:x_end].copy()

    # Adjust bounding box
    adjusted_bbox = []
    for b in bbox:
        x1, y1, x2, y2 = b[0:4]

        # Shift coordinates relative to crop top-left
        x1 = x1 - x_start
        y1 = y1 - y_start
        x2 = x2 - x_start
        y2 = y2 - y_start

        # Check if bbox is within crop (allow partial overlap)
        if x2 <= 0 or y2 <= 0 or x1 >= w1 or y1 >= h1:
            continue  # Bbox is completely outside crop

        # Clip coordinates to crop boundaries
        x1 = max(0, min(x1, w1))
        y1 = max(0, min(y1, h1))
        x2 = max(0, min(x2, w1))
        y2 = max(0, min(y2, h1))

        # Skip if bbox is invalid (zero or negative size)
        if x1 >= x2 or y1 >= y2:
            continue

        adjusted_bbox.append(np.concatenate(([x1, y1, x2, y2], b[4:])))

    adjusted_bbox = np.array(adjusted_bbox, np.float32)
    adjusted_bbox = types.convert_hbb(adjusted_bbox, fmt=BBoxFormat.XYXY2CXCYWHN, imgsz=(h1, w1))
    return cropped_image, adjusted_bbox

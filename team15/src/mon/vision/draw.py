#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements drawing functionalities for images."""

__all__ = [
    "draw_bbox",
    "draw_heatmap",
    "draw_semantic",
    "draw_trajectory",
]

import cv2
import numpy as np

from mon import core
from mon.vision import types


def draw_bbox(
    image     : np.ndarray,
    bbox      : np.ndarray | list,
    label     : int | str = None,
    color     : list[int] = [255, 255, 255],
    thickness : int   = 1,
    line_type : int   = cv2.LINE_8,
    shift     : int   = 0,
    font_face : int   = cv2.FONT_HERSHEY_DUPLEX,
    font_scale: float = 0.8,
    fill      : bool | float = False
) -> np.ndarray:
    """Draw bounding box on image.

    Args:
        image: Image as ``numpy.ndarray`` in [H, W, C] format, range [0, 255].
        bbox: Bounding box in XYXY format as ``numpy.ndarray`` or ``list``.
        label: Label for box as ``int`` or ``str``. Default is ``None``.
        color: Box color as ``list[int]``. Default is [255, 255, 255].
        thickness: Border thickness in px as ``int``. Default is ``1``.
        line_type: Line type as ``int`` (e.g., ``cv2.LINE_8``). Default is ``cv2.LINE_8``.
        shift: Fractional bits in coordinates as ``int``. Default is ``0``.
        font_face: Label font as ``int`` (e.g., ``cv2.FONT_HERSHEY_DUPLEX``).
            Default is ``cv2.FONT_HERSHEY_DUPLEX``.
        font_scale: Label text scale as ``float``. Default is ``0.8``
        fill: Fill transparency (``True``=0.5, 0.0-1.0) as ``bool`` or ``float``.
            Default is ``False``.

    Returns:
        Image with drawn bounding box as ``numpy.ndarray``.
    """
    drawing = image.copy()
    color   = color or [255, 255, 255]
    white   = [255, 255, 255]
    pt1     = (int(bbox[0]), int(bbox[1]))
    pt2     = (int(bbox[2]), int(bbox[3]))
    cv2.rectangle(drawing, pt1, pt2, color, thickness, line_type, shift)

    if label not in [None, "None", ""]:
        label  = f"{label}"
        offset = int(thickness / 2)
        text_size, baseline = cv2.getTextSize(label, font_face, font_scale, 1)
        cv2.rectangle(
            img       = drawing,  # Changed from 'image' to 'drawing' for consistency
            pt1       = (pt1[0] - offset, pt1[1] - text_size[1] - offset),
            pt2       = (pt1[0] + text_size[0], pt1[1]),
            color     = color,
            thickness = cv2.FILLED
        )
        text_org = (pt1[0] - offset, pt1[1] - offset)
        cv2.putText(drawing, label, text_org, font_face, font_scale, white, 1)

    if fill is True or fill > 0.0:
        alpha   = 0.5 if fill is True else fill
        overlay = drawing.copy()
        cv2.rectangle(overlay, pt1, pt2, color, -1)
        cv2.addWeighted(overlay, alpha, drawing, 1 - alpha, 0, drawing)

    return drawing


def draw_heatmap(
    image    : np.ndarray,
    heatmap  : np.ndarray,
    color_map: int   = cv2.COLORMAP_JET,
    alpha    : float = 0.5,
    use_rgb  : bool  = False
) -> np.ndarray:
    """Overlay heatmap on image.

    Args:
        image: RGB/BGR image as ``numpy.ndarray`` in [H, W, C], range [0.0, 1.0].
        heatmap: Heatmap mask as ``numpy.ndarray``.
        color_map: Heatmap color map as ``int``. Default is ``cv2.COLORMAP_JET``.
        alpha: Transparency ratio (0.0-1.0) as ``float``. Default is ``0.5``.
        use_rgb: Convert heatmap to RGB if ``True``. Default is ``False``.

    Returns:
        Image with heatmap overlay as ``numpy.ndarray``.

    Raises:
        ValueError: If image exceeds range [0.0, 1.0] or alpha is invalid.
    """
    
    if np.max(image) > 1:
        raise ValueError(f"[image] should be np.float32 in range [0.0, 1.0], got {np.max(image)}.")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"[alpha] should be in range [0.0, 1.0], got {alpha}.")

    heatmap = types.depth_to_color(heatmap, color_map, use_rgb)
    heatmap = np.float32(heatmap) / 255
    drawing = types.blend_images(image, heatmap, alpha)
    drawing = drawing / np.max(drawing)
    drawing = np.uint8(255 * drawing)
    return drawing


def draw_semantic(
    image   : np.ndarray,
    semantic: np.ndarray,
    classes : core.Classes,
    alpha   : float = 0.5
) -> np.ndarray:
    """Overlay semantic mask on image.

    Args:
        image: RGB image as ``numpy.ndarray`` in [H, W, C], range [0, 255].
        semantic: Semantic mask as ``numpy.ndarray`` in [H, W, 1].
        classes: List of class labels as ``Classes``.
        alpha: Transparency ratio (0.0-1.0) as ``float``. Default is ``0.5``.

    Returns:
        Image with semantic overlay as ``numpy.ndarray``.
    """
    color_map = types.label_map_id_to_color(semantic, classes)
    drawing   = types.blend_images(image, color_map, alpha)
    drawing   = drawing.astype(np.uint8)
    return drawing
    

def draw_trajectory(
    image     : np.ndarray,
    trajectory: np.ndarray | list,
    color     : list[int] = [255, 255, 255],
    thickness : int  = 1,
    line_type : int  = cv2.LINE_8,
    point     : bool = False,
    radius    : int  = 3
) -> np.ndarray:
    """Draw trajectory path on image.

    Args:
        image: RGB image as ``numpy.ndarray`` in [H, W, C], range [0, 255].
        trajectory: 2D points as ``numpy.ndarray`` or ``list`` in [(x1, y1), ...] format.
        color: Path color as ``list[int]``. Default is [255, 255, 255].
        thickness: Path thickness in px as ``int``. Default is ``1``.
        line_type: Line type as ``int`` (e.g., ``cv2.LINE_8``). Default is ``cv2.LINE_8``.
        point: Draw points if ``True``. Default is ``False``.
        radius: Point radius in px as ``int``. Default is ``3``.

    Returns:
        Image with trajectory as ``numpy.ndarray``.

    Raises:
        TypeError: If ``trajectory`` format is invalid.
    """
    drawing = image.copy()

    if isinstance(trajectory, list):
        if not all(len(t) == 2 for t in trajectory):
            raise TypeError("[trajectory] must be a list of points in [(x1, y1), ...] format.")
        trajectory = np.array(trajectory)
    trajectory = np.array(trajectory).reshape((-1, 1, 2)).astype(int)
    color      = color or [255, 255, 255]
    cv2.polylines(drawing, [trajectory], False, color, thickness, line_type)
    if point:
        for p in trajectory:
            cv2.circle(drawing, tuple(p[0]), radius, color, -1)  # Fixed syntax and type

    return drawing

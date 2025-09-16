#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements label map manipulation and preprocessing functions.

Common Tasks:
    - Format conversions.
"""

__all__ = [
    "label_map_color_to_id",
    "label_map_id_to_color",
    "label_map_id_to_one_hot",
    "label_map_id_to_train_id",
    "label_map_one_hot_to_id",
]

import numpy as np
import torch

from mon import core
from mon.nn import functional as F
from mon.vision.types import image as I


# ----- Conversion -----
def label_map_id_to_train_id(label_map: np.ndarray, classes: core.Classes) -> np.ndarray:
    """Converts label map from IDs to train IDs.

    Args:
        label_map: Label map as ``numpy.ndarray`` in [H, W] or [H, W, 1] format.
        classes: ``Classes`` object mapping IDs to train IDs.
    
    Returns:
        Converted label map as numpy.ndarray in [H, W, 1] format.
    
    Raises:
        TypeError: If ``label_map`` is not a ``numpy.ndarray``.
    """
    if not isinstance(label_map, np.ndarray):
        raise TypeError(f"[label_map] must be a numpy.ndarray, got {type(label_map)}.")
    
    id2train_id = classes.id_to_train_id
    h, w        = I.image_size(label_map)
    label_ids   = np.zeros((h, w), dtype=np.uint8)
    label_map   = I.image_to_2d(label_map)
    
    for id, train_id in id2train_id.items():
        label_ids[label_map == id] = train_id
    
    return np.expand_dims(label_ids, axis=-1)
 

def label_map_id_to_color(label_map: np.ndarray, classes: core.Classes) -> np.ndarray:
    """Converts label map from IDs to color-coded representation.

    Args:
        label_map: Label map as ``numpy.ndarray`` in [H, W] or [H, W, 1] format.
        classes: ``Classes`` object mapping IDs to colors.
    
    Returns:
        Color-coded label map as ``numpy.ndarray`` in [H, W, 3] format.
    
    Raises:
        TypeError: If ``label_map`` is not a ``numpy.ndarray``.
    """
    if not isinstance(label_map, np.ndarray):
        raise TypeError(f"[label_map] must be a numpy.ndarray, got {type(label_map)}.")

    id2color  = classes.id_color
    h, w      = I.image_size(label_map)
    color_map = np.zeros((h, w, 3), dtype=np.uint8)
    label_map = I.image_to_2d(label_map)
    for id, color in id2color.items():
        color_map[label_map == id] = color
    return color_map


def label_map_color_to_id(label_map: np.ndarray, classes: core.Classes) -> np.ndarray:
    """Converts a color-coded label map to label IDs.

    Args:
        label_map: Color-coded label map as ``numpy.ndarray`` in [H, W, C] format.
        classes: ``Classes`` object mapping colors to IDs.
    
    Returns:
        Label map with IDs as ``numpy.ndarray`` in [H, W, 1] format.
    
    Raises:
        TypeError: If ``label_map`` is not a ``numpy.ndarray``.
    """
    if not isinstance(label_map, np.ndarray):
        raise TypeError(f"[label_map] must be a numpy.ndarray, got {type(label_map)}.")
    
    id2color  = classes.id_color
    h, w      = I.image_size(label_map)
    label_ids = np.zeros((h, w), dtype=np.uint8)
    for id, color in id2color.items():
        label_ids[np.all(label_map == color, axis=-1)] = id
    label_ids = np.expand_dims(label_ids, axis=-1)
    return label_ids


def label_map_id_to_one_hot(
    label_map  : torch.Tensor | np.ndarray,
    num_classes: int          = None,
    classes    : core.Classes = None,
) ->torch.Tensor | np.ndarray:
    """Converts label map from IDs to one-hot encoded format.

    Args:
        label_map: IDs label map as ``torch.Tensor`` [B, 1, H, W] or
            ``numpy.ndarray`` [H, W, 1].
        num_classes: Number of classes in the label map, optional.
        classes: ``Classes`` object with class info, optional.
    
    Returns:
        One-hot encoded label map as ``torch.Tensor`` [B, C, H, W] or
        ``numpy.ndarray`` [H, W, C].
    
    Raises:
        ValueError: If neither ``num_classes`` nor ``Classes`` is provided.
        TypeError: If ``label_map`` is not a ``torch.Tensor`` or ``numpy.ndarray``.
    """
    if num_classes is None and classes is None:
        raise ValueError("Either [num_classes] or [classes] must be provided.")

    num_classes = num_classes or classes.num_trainable_classes
    if isinstance(label_map, torch.Tensor):
        label_map = I.image_to_3d(label_map).long()
        one_hot   = F.one_hot(label_map, num_classes)
        return I.image_to_channel_first(one_hot).contiguous()
    elif isinstance(label_map, np.ndarray):
        label_map = I.image_to_2d(label_map)
        return np.eye(num_classes)[label_map]
    else:
        raise TypeError(f"[label_map] must be a torch.Tensor or numpy.ndarray, got {type(label_map)}.")


def label_map_one_hot_to_id(label_map: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    """Converts a one-hot encoded label map to label IDs.

    Args:
        label_map: One-hot encoded label map as ``torch.Tensor`` [B, C, H, W] or
            ``numpy.ndarray`` [H, W, C].
    
    Returns:
        Label map with IDs as ``torch.Tensor`` [B, 1, H, W] or
        ``numpy.ndarray`` [H, W, 1].
    
    Raises:
        TypeError: If ``label_map`` is not a ``torch.Tensor`` or ``numpy.ndarray``.
    """
    if isinstance(label_map, torch.Tensor):
        label_map = torch.argmax(label_map, dim=-1, keepdim=True)
    elif isinstance(label_map, np.ndarray):
        label_map = np.argmax(label_map, axis=-1, keepdims=True)
    else:
        raise TypeError(f"[label_map] must be a torch.Tensor or numpy.ndarray, got {type(label_map)}.")
    return label_map

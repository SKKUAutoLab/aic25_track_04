#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements input/output operations for images.

Common Tasks:
    - Load images from disk.
    - Save images to disk.
    - Batch I/O.
    - Metadata handling.
"""

__all__ = [
    "load_image",
    "read_image_shape",
    "save_image",
]

import cv2
import numpy as np
import rawpy
import torch
import torchvision
import PIL.Image

from mon import core
from mon.vision.types.image import processing, utils


# ----- Reading -----
def load_image(
    path     : core.Path,
    flags    : int          = cv2.IMREAD_COLOR,
    to_tensor: bool         = False,
    normalize: bool         = False,
    device   : torch.device = None
) -> torch.Tensor | np.ndarray:
    """Loads an image from a file path using OpenCV.

    Args:
        path: Image file path.
        flags: OpenCV flag for reading the image. Default is ``cv2.IMREAD_COLOR``.
        to_tensor: Convert to ``torch.Tensor`` if ``True``. Default is ``False``.
        normalize: Normalize to [0.0, 1.0] if ``True``. Default is ``False``.
        device: Device to place tensor on, e.g., ``'cuda'`` or ``None`` for CPU.
            Default is ``None``.
    
    Returns:
        RGB or grayscale image as ``torch.Tensor`` [B, C, H, W] or ``numpy.ndarray`` [H, W, C].
    """
    path = core.Path(path)
    if path.is_raw_image_file():  # Read raw image
        image = rawpy.imread(str(path))
        image = image.postprocess()
    else:  # Read other types of image
        image = cv2.imread(str(path), flags)  # BGR
        if image.ndim == 2:  # [H, W] -> [H, W, 1] for grayscale
            image = np.expand_dims(image, axis=-1)
        if utils.is_image_grayscale(image) and flags != cv2.IMREAD_GRAYSCALE:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif utils.is_image_colored(image):
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = image[:, :, ::-1]   # Faster than cv2.COLOR_BGR2RGB

    if to_tensor:
        image = processing.image_to_tensor(image, normalize=normalize, device=device)
    
    return image


def read_image_shape(path: core.Path) -> tuple[int, int, int]:
    """Reads an image shape from a file path using PIL or rawpy.

    Args:
        path: Image file path.

    Returns:
        Tuple of (height, width, channels) in [H, W, C] format.

    Raises:
        ValueError: If image mode is unsupported for non-RAW images.
    """
    path = core.Path(path)
    if path.is_raw_image_file():
        image = rawpy.imread(str(path)).raw_image_visible
        h, w = image.shape
        c = 3
    else:
        with PIL.Image.open(str(path)) as image:
            w, h = image.size
            mode = image.mode
            c = {"RGB": 3, "RGBA": 4, "L": 1}.get(mode, None)
            if c is None:
                raise ValueError(f"Unsupported image mode {mode}.")
    
    return h, w, c


# ----- Writing -----
def save_image(image: torch.Tensor | np.ndarray | PIL.Image.Image, path: core.Path):
    """Save an image to a file path.

    Args:
        image: Image as ``torch.Tensor`` [B, C, H, W], ``numpy.ndarray`` [H, W, C],
            or ``PIL.Image.Image``.
        path: Output file path.

    Raises:
        TypeError: If ``image`` is not a ``torch.Tensor`` or ``numpy.ndarray``.
    """
    path = core.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(image, torch.Tensor):
        torchvision.utils.save_image(image, str(path))
    elif isinstance(image, np.ndarray):
        cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    elif isinstance(image, PIL.Image.Image):
        image.save(str(path))
    else:
        raise TypeError(f"[image] must be a torch.Tensor, numpy.ndarray, or PIL.Image, got {type(image)}.")

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements general-purpose utilities for image processing tasks.

Common Tasks:
    - Property accessors
    - Validation checks
    - Miscellaneous
"""

__all__ = [
    "image_center",
    "image_center4",
    "image_channel",
    "image_num_channels",
    "image_shape",
    "image_size",
    "is_image",
    "is_image_channel_first",
    "is_image_channel_last",
    "is_image_colored",
    "is_image_grayscale",
    "is_image_normalized",
]

import math

import numpy as np
import torch

from mon.nn import _size_2_t


# ----- Accessing -----
def image_center(image: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    """Returns the center of an image as (x=h/2, y=w/2).

    Args:
        image: Image as ``torch.Tensor`` or ``numpy.ndarray``.
    
    Returns:
        Center coordinates as ``torch.Tensor`` or ``numpy.ndarray`` with shape [2].
    """
    h, w   = image_size(image)
    center = [h / 2, w / 2]
    return torch.tensor(center) if isinstance(image, torch.Tensor) else np.array(center)


def image_center4(image: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    """Returns the center of an image as (x=h/2, y=w/2, x=h/2, y=w/2).

    Args:
        image: Image as ``torch.Tensor`` or ``numpy.ndarray``.
    
    Returns:
        Center coordinates as ``torch.Tensor`` or ``numpy.ndarray`` with shape [4].
    """
    h, w   = image_size(image)
    center = [h / 2, w / 2, h / 2, w / 2]
    return torch.tensor(center) if isinstance(image, torch.Tensor) else np.array(center)


def image_channel(
    image   : torch.Tensor | np.ndarray,
    index   : _size_2_t,
    keep_dim: bool = True
) -> torch.Tensor | np.ndarray:
    """Extracts a channel or channels from an image.

    Args:
        image: Image as ``torch.Tensor`` or ``numpy.ndarray``.
        index: Channel index (int) or range (Sequence[int]) to extract.
        keep_dim: Keep singleton dimension if ``True``. Default is ``True``.
    
    Returns:
        Extracted channel(s) as ``torch.Tensor`` or ``numpy.ndarray``.
   
    Raises:
        ValueError: If image dimensions are invalid for channel extraction.
    """
    i1, i2 = (index, index + 1) if isinstance(index, int) else (index[0], index[1])
    
    if is_image_channel_first(image):
        if image.ndim == 4:
            return image[:, i1:i2, :, :] if keep_dim else image[:, i1, :, :]
        elif image.ndim == 3:
            return image[i1:i2, :, :]    if keep_dim else image[i1, :, :]
    else:
        if image.ndim == 4:
            return image[:, :, :, i1:i2] if keep_dim else image[:, :, :, i1]
        elif image.ndim == 3:
            return image[:, :, i1:i2]    if keep_dim else image[:, :, i1]
    raise ValueError(f"Invalid image dimensions for channel extraction {image.ndim}.")
    
    
def image_num_channels(image: torch.Tensor | np.ndarray) -> int:
    """Returns the number of channels in an image.

    Args:
        image: Image as ``torch.Tensor`` or ``numpy.ndarray``.
   
    Returns:
        Number of channels (e.g., 1 for grayscale, 3 for RGB).
    """
    if image.ndim == 4:
        c = image.shape[1] if is_image_channel_first(image) else image.shape[3]
    elif image.ndim == 3:
        c = image.shape[0] if is_image_channel_first(image) else image.shape[2]
    elif image.ndim == 2:
        c = 1
    else:
        c = 0
    return c


def image_shape(image: torch.Tensor | np.ndarray) -> list[int]:
    """Returns height, width, and channels of an image.

    Args:
        image: RGB image as ``torch.Tensor`` with shape [B, C, H, W] or
            ``numpy.ndarray`` with shape [H, W, C].

    Returns:
        List of [height, width, channels] as ``list[int]``.
    """
    h, w, c = (
        (image.shape[-2], image.shape[-1], image.shape[-3])
        if is_image_channel_first(image)
        else (image.shape[-3], image.shape[-2], image.shape[-1])
    )
    return [h, w, c]


def image_size(
    input  : torch.Tensor | np.ndarray | _size_2_t,
    divisor: int = None,
) -> tuple[int, int]:
    """Returns height and width of an image in [H, W] format.

    Args:
        input: RGB image, tensor, array, size, or path as ``torch.Tensor``,
            ``numpy.ndarray``, ``int``, or ``Sequence[int]``.
        divisor: Divisor to adjust size as ``int`` or ``None``. Default is ``None``.

    Returns:
        Tuple of (height, width) in pixels as ``tuple[int, int]``.

    Raises:
        TypeError: If ``input`` type is not supported.
    """
    if isinstance(input, list | tuple):
        if len(input) == 1:
            size = (input[0], input[0])
        elif len(input) == 2:
            size = input
        elif len(input) == 3:
            size = input[:2] if len(input) == 3 and input[0] >= input[2] else input[-2:]
    elif isinstance(input, (int, float)):
        size = (input, input)
    elif isinstance(input, torch.Tensor | np.ndarray):
        size = (
            (input.shape[-2], input.shape[-1])
            if is_image_channel_first(input)
            else (input.shape[-3], input.shape[-2])
        )
    else:
        raise TypeError(f"[input] must be a torch.Tensor, numpy.ndarray, int, "
                        f"Sequence[int], str, or core.Path, got {type(input)}.")

    if divisor is not None:
        size = tuple(int(math.ceil(dim / divisor) * divisor) for dim in size)
    return size


# ----- Validation -----
def is_image(image: torch.Tensor | np.ndarray) -> bool:
    """Checks if an input is an image tensor or array.

    Args:
        image: Input to evaluate as ``torch.Tensor`` or ``numpy.ndarray``.

    Returns:
        ``True`` if input is a tensor or array and a color or grayscale image,
        ``False`` otherwise.
    """
    return (
        isinstance(image, torch.Tensor | np.ndarray)
        and (is_image_colored(image) or is_image_grayscale(image))
    )


def is_image_channel_first(image: torch.Tensor | np.ndarray) -> bool:
    """Checks if an image is in channel-first format.

    Args:
        image: Image as ``torch.Tensor`` or ``numpy.ndarray`` in [C, H, W] or
            [B, C, H, W] format.

    Returns:
        ``True`` if channel-first (e.g., [C, H, W]),
        ``False`` if channel-last (e.g., [H, W, C]).

    Raises:
        TypeError: If ``image`` is not a ``torch.Tensor`` or ``numpy.ndarray``.
        ValueError: If ``image`` dimensions are invalid or channel format is ambiguous.

    Notes:
        Assumes the smallest dimension is the channel dimension.
    """
    # Determine tensor type and get shape
    if isinstance(image, torch.Tensor):
        shape = image.size()
    elif isinstance(image, np.ndarray):
        shape = image.shape
    else:
        raise TypeError(f"[image] must be a numpy.ndarray or torch.Tensor, got {type(image)}.")
    
    # Check if tensor has at least 3 dimensions (batch, height/width, channels)
    if not 3 <= len(shape) <= 4:
        raise ValueError("[image] must have at least 3 dimensions (batch, channels, height/width).")
    
    # Extract dimensions
    if len(shape) == 3:
        s0, s1, s2 = shape
    else:
        _, s0, s1, s2 = shape
    
    # Heuristic: Channels are typically smaller than spatial dimensions
    if (s0 < s1) and (s0 < s2):
        return True
    elif (s2 < s0) and (s2 < s1):
        return False
    else:
        raise ValueError(f"Cannot determine channel format for shape [{shape}].")


def is_image_channel_last(image: torch.Tensor | np.ndarray) -> bool:
    """Checks if an image is in channel-last format.

    Args:
        image: Image as ``torch.Tensor`` or ``numpy.ndarray`` in [H, W, C] or
            [B, H, W, C] format.

    Returns:
        ``True`` if channel-last (e.g., [H, W, C]),
        ``False`` if channel-first (e.g., [C, H, W]).
    """
    return not is_image_channel_first(image)


def is_image_colored(image: torch.Tensor | np.ndarray) -> bool:
    """Checks if an image is a color image.

    Args:
        image: Image as ``torch.Tensor`` or ``numpy.ndarray``.

    Returns:
        ``True`` if the image has 3 or 4 channels, ``False`` otherwise.

    Notes:
        Assumes a color image has 3 or 4 channels (e.g., RGB or RGBA).
    """
    return image_num_channels(image) in [3, 4]


def is_image_grayscale(image: torch.Tensor | np.ndarray) -> bool:
    """Checks if an image is grayscale.

    Args:
        image: Image as ``torch.Tensor`` or ``numpy.ndarray``.
   
    Returns:
        ``True`` if the image has 1 channel or 2 dimensions, ``False`` otherwise.
    
    Notes:
        Assumes a grayscale image has 1 channel (e.g., [H, W] or [B, 1, H, W]).
    """
    return image_num_channels(image) == 1 or len(image.shape) == 2


def is_image_normalized(image: torch.Tensor | np.ndarray) -> bool:
    """Checks if an image is normalized to range [-1.0, 1.0] or [0.0, 1.0].

    Args:
        image: Image as ``torch.Tensor`` or ``numpy.ndarray``.
    
    Returns:
        ``True`` if absolute max value is <= 1.0, ``False`` otherwise.
    
    Raises:
        TypeError: If image is not a ``torch.Tensor`` or ``numpy.ndarray``.
    """
    if isinstance(image, torch.Tensor):
        return abs(torch.max(image)) <= 1.0
    elif isinstance(image, np.ndarray):
        return abs(np.amax(image)) <= 1.0
    else:
        raise TypeError(f"[image] must be a torch.Tensor or numpy.ndarray, got {type(image)}.")

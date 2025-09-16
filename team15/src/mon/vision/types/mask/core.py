#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements ``Mask`` class and core properties.

Common Tasks:
    - Define the ``Mask`` class (e.g., wrapper for ``numpy.ndarray`` or ``torch.Tensor``).
    - Access core properties.
"""

__all__ = [
    "SemanticMask",
]

import cv2
import numpy as np
import torch

from mon import core
from mon.vision.types import image as I


class SemanticMask(I.Image):
    """Segmentation mask object.

    Attributes:
        albumentation_target_type: Type of target for Albumentations. Default is ``mask``.

    Args:
        data: Input data as a ``torch.Tensor`` or ``numpy.ndarray``. Default is ``None``.
        path: Semantic mask file path. Default is ``None``.
        root: Root directory for the semantic mask. Default is ``None``.
        flags: Flag to read image (e.g., ``cv2.IMREAD_COLOR``). Default is ``cv2.IMREAD_COLOR_BGR``.
        cache: If ``True``, caches image in memory. Default is ``False``.
    """

    albumentation_target_type: str = "mask"

    def __init__(
        self,
        data  : torch.Tensor | np.ndarray = None,
        path  : core.Path      = None,
        root  : core.Path      = None,
        flags : int            = cv2.IMREAD_COLOR_BGR,
        cache : bool           = False,
    ):
        super().__init__(data=data, path=path, root=root, flags=flags, cache=cache)

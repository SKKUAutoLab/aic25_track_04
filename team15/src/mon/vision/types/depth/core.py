#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements ``DepthMap`` class and core properties.

Common Tasks:
    - Define the ``DepthMap`` class (e.g., wrapper for ``numpy.ndarray`` or ``torch.Tensor``).
    - Access core properties.
"""

__all__ = [
    "DepthMap",
]

import cv2
import numpy as np
import torch

from mon import core
from mon.constants import DepthSource
from mon.vision.types import image as I


class DepthMap(I.Image):
    """Depth map object.

    Attributes:
        albumentation_target_type: Type of target for Albumentations. Default is ``image``.

    Args:
        data: Input data as a ``torch.Tensor`` or ``numpy.ndarray``. Default is ``None``.
        path: Depth map file path. Default is ``None``.
        root: Root directory for the depth map. Default is ``None``.
        source: Source of depth data from ``DepthSource``. Default is ``DepthSource.DAv2_ViTB``.
        flags: Flag to read image (e.g., ``cv2.IMREAD_COLOR``). Default is ``cv2.IMREAD_GRAYSCALE``.
        cache: If ``True``, caches image in memory. Default is ``False``.
    """

    albumentation_target_type: str = "image"

    def __init__(
        self,
        data  : torch.Tensor | np.ndarray = None,
        path  : core.Path   = None,
        root  : core.Path   = None,
        source: DepthSource = DepthSource.DAv2_ViTB,
        flags : int         = cv2.IMREAD_GRAYSCALE,
        cache : bool        = False,
    ):
        source = DepthSource.from_value(source)
        if source not in DepthSource:
            raise ValueError(f"[source] must be one of {DepthSource}, got {source}.")

        super().__init__(data=data, path=path, root=root, flags=flags, cache=cache)
        self.source = source
        # self.flags  = (cv2.IMREAD_COLOR if source and "c" in source else cv2.IMREAD_GRAYSCALE)

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements ``InfraredMap`` class and core properties.

Common Tasks:
    - Define the ``InfraredMap`` class (e.g., wrapper for ``numpy.ndarray`` or ``torch.Tensor``).
    - Access core properties.
"""

__all__ = [
    "InfraredMap",
]

import cv2
import numpy as np
import torch

from mon import core
from mon.constants import InfraredSource
from mon.vision.types import image as I


class InfraredMap(I.Image):
    """Infrared map object.

    Attributes:
        albumentation_target_type: Type of target for Albumentations. Default is ``image``.

    Args:
        data: Input data as a ``torch.Tensor`` or ``numpy.ndarray``. Default is ``None``.
        path: Infrared map file path. Default is ``None``.
        root: Root directory for the infrared map. Default is ``None``.
        source: Source of infrared data from ``InfraredSource``. Default is ``InfraredSource.INFRARED``.
        flags: Flag to read image (e.g., ``cv2.IMREAD_COLOR``). Default is ``cv2.IMREAD_GRAYSCALE``.
        cache: If ``True``, caches image in memory. Default is ``False``.
    """

    albumentation_target_type: str = "image"

    def __init__(
        self,
        data  : torch.Tensor | np.ndarray = None,
        path  : core.Path      = None,
        root  : core.Path      = None,
        source: InfraredSource = InfraredSource.INFRARED,
        flags : int            = cv2.IMREAD_GRAYSCALE,
        cache : bool           = False,
    ):
        source = InfraredSource.from_value(source)
        if source not in InfraredSource:
            raise ValueError(f"[source] must be one of {InfraredSource}, got {source}.")

        super().__init__(data=data, path=path, root=root, flags=flags, cache=cache)
        self.source = source

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements ``Frame`` class and core properties.

Common Tasks:
    - Define the ``Frame`` class (e.g., wrapper for ``numpy.ndarray`` or ``torch.Tensor``).
    - Access core properties.
"""

__all__ = [
    "Frame",
]

from typing import Any

import numpy as np
import torch

from mon import core
from mon.constants import SAVE_IMAGE_EXT


class Frame(core.BaseTensor, core.DatasetMixin, core.DataLoaderMixin):
    """Frame object.

    Attributes:
        albumentation_target_type: Type of target for Albumentations. Default is ``image``.

    Args:
        data: Input data as a ``torch.Tensor`` or ``numpy.ndarray``.
        index: Index of frame in video.
        orig_shape: Original shape of the image in [H, W] format. Default is ``None``.
        path: Video file path. Default is ``None``.
        root: Root directory for the video. Default is ``None``.
    """

    albumentation_target_type: str = "image"

    def __init__(
        self,
        data      : torch.Tensor | np.ndarray,
        index     : int,
        orig_shape: tuple[int, int, int] = None,
        path      : core.Path = None,
        root      : core.Path = None,
    ):
        if orig_shape is None:
            orig_shape = data.shape

        super().__init__(data=data, orig_shape=orig_shape)
        self._index = index
        self._path  = core.Path(path) if path is not None else None
        self._root  = core.Path(root) if root is not None else None

    @property
    def index(self) -> int:
        """Returns the index of the frame in the video."""
        return self._index

    @property
    def path(self) -> core.Path:
        """Returns the image file path."""
        return self._path

    @property
    def root(self) -> core.Path:
        """Returns the root directory for the image."""
        return self._root

    @property
    def frame_path(self) -> core.Path:
        """Returns the path for each frame of the video: <self.path>_<self.index>."""
        if self.path is not None:
            path = self.path
            return path.parent / path.stem / f"{path.stem}_{self.index}{SAVE_IMAGE_EXT}"
        else:
            return self._path

    @property
    def meta(self) -> dict:
        """Returns metadata about the image.

        Returns:
            Dict with keys ``name``, ``stem``, ``path``, ``shape``, and ``hash``.
        """
        return {
            "path"      : self.frame_path,
            "video_path": self.path,
            "index"     : self.index,
            "orig_shape": self.orig_shape,
            "shape"     : self.shape,
            "hash"      : self.path.stat().st_size if isinstance(self.path, core.Path) else None,
        }

    def load(self) -> np.ndarray:
        """Loads the image into memory.

        Returns:
            ``numpy.ndarray`` in [H, W, C] format, values in [0, 255].
        """
        return self._data

    # ----- DatasetMixin -----
    @staticmethod
    def to_tensor(
        data      : torch.Tensor | np.ndarray,
        orig_shape: tuple[int, int] = None,
        normalize : bool            = True,
        *args, **kwargs
    ) -> torch.Tensor:
        """Transforms the underlying data to a tensor.

        Args:
            data: Input data as a ``torch.Tensor`` or ``numpy.ndarray``.
            orig_shape: Original shape of the image in [H, W] format. Default is ``None``.
            normalize: Normalize the underlying data. Default is ``True``.

        Returns:
            ``torch.Tensor`` of converted data.
        """
        from mon.vision.types.image import processing
        return processing.image_to_tensor(data, normalize)

    # ----- DataLoaderMixin -----
    @staticmethod
    def collate_fn(batch: list) -> torch.Tensor | np.ndarray | None:
        """Collates batch data for ``torch.utils.data.DataLoader``.

        Args:
            batch: List of images as ``torch.Tensor`` or ``numpy.ndarray``.

        Returns:
            Collated ``torch.Tensor``, ``numpy.ndarray``, or ``None`` if empty/invalid.
        """
        if not batch:
            return None
        from mon.vision.types.image import processing
        return processing.image_to_4d(batch)


class Frames(core.BaseTensor, core.DatasetMixin, core.DataLoaderMixin):
    """Frames object.

    Attributes:
        albumentation_target_type: Type of target for Albumentations. Default is ``image``.

    Args:
        data: Input data as a ``torch.Tensor`` or ``numpy.ndarray``.
        indexes: List of indexes of frames in video. Default is ``None``.
        orig_shape: Original shape of the image in [H, W] format. Default is ``None``.
        path: Video file path. Default is ``None``.
        root: Root directory for the video. Default is ``None``.
    """

    albumentation_target_type: str = "image"

    def __init__(
        self,
        data      : torch.Tensor | np.ndarray | list[Any],
        indexes   : list[int] = None,
        orig_shape: tuple[int, int, int] = None,
        path      : core.Path = None,
        root      : core.Path = None,
    ):
        super().__init__(data=data, orig_shape=orig_shape)

        # TODO: continue later

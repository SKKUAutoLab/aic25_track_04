#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements ``Image`` class and core properties.

Common Tasks:
    - Define the ``Image`` class (e.g., wrapper for ``numpy.ndarray`` or ``torch.Tensor``).
    - Access core properties.
"""

__all__ = [
    "Image",
]

import cv2
import numpy as np
import torch

from mon import core


class Image(core.BaseTensor, core.DatasetMixin, core.DataLoaderMixin):
    """Image object.

    Attributes:
        albumentation_target_type: Type of target for Albumentations. Default is ``image``.

    Args:
        data: Input data as a ``torch.Tensor`` or ``numpy.ndarray``. Default is ``None``.
        path: Image file path. Default is ``None``.
        root: Root directory for the image. Default is ``None``.
        flags: Flag to read image (e.g., ``cv2.IMREAD_COLOR``). Default is ``cv2.IMREAD_COLOR_BGR``.
        cache: If ``True``, caches image in memory. Default is ``False``.
    """

    albumentation_target_type: str = "image"

    def __init__(
        self,
        data : torch.Tensor | np.ndarray = None,
        path : core.Path = None,
        root : core.Path = None,
        flags: int       = cv2.IMREAD_COLOR_BGR,
        cache: bool      = False,
    ):
        if all(d is None for d in [data, path]):
            raise ValueError("Either [data] or [path] must be provided to initialize the Image object.")
        if data is not None:
            orig_shape = data.shape
        elif core.Path(path).is_image_file(exist=True):
            from mon.vision.types.image import io
            orig_shape = io.read_image_shape(path=path)
        else:
            orig_shape = None

        super().__init__(data=data, orig_shape=orig_shape)
        self._path = core.Path(path) if path is not None else None
        self._root = core.Path(root) if root is not None else None
        self.flags = flags
        self.cache = cache

    @property
    def shape(self) -> tuple[int, int, int]:
        """Return the shape of the underlying data tensor.

        Returns:
            The shape of the data tensor.
        """
        return self._orig_shape

    @property
    def path(self) -> core.Path:
        """Returns the image file path."""
        return self._path

    @property
    def root(self) -> core.Path:
        """Returns the root directory for the image."""
        return self._root

    @property
    def meta(self) -> dict:
        """Returns metadata about the image.

        Returns:
            Dict with keys ``name``, ``stem``, ``path``, ``shape``, and ``hash``.
        """
        return {
            "path"      : self.path,
            "orig_shape": self.orig_shape,
            "shape"     : self.shape,
            "hash"      : self.path.stat().st_size if isinstance(self.path, core.Path) else None,
        }

    def load(self, reload: bool = False) -> np.ndarray:
        """Loads the image into memory.

        Args:
            reload: If ``True``, reload the image even if already cached. Default is ``False``.

        Returns:
            ``numpy.ndarray`` in [H, W, C] format, values in [0, 255].
        """
        # Return the image if it is already loaded and not reloading
        if not reload and self._data is not None:
            return self._data

        # Load the image
        from mon.vision.types.image import io
        image = io.load_image(self.path, self.flags, False, False)

        # Update the original shape of the image
        if self._orig_shape != image.shape:
            self._orig_shape = image.shape

        # Cache the image if needed
        self._data = image if self.cache else None
        return image

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

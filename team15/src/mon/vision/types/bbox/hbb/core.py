#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements ``HBBs`` class and core properties.

Common Tasks:
    - Define the ``HBBs`` class (e.g., wrapper for ``numpy.ndarray`` or ``torch.Tensor``).
    - Access core properties.
"""

__all__ = [
    "HBBs",
]

import numpy as np
import torch

from mon import core
from mon.constants import BBoxFormat


# ----- Base -----
class HBBs(core.BaseTensor, core.DatasetMixin, core.DataLoaderMixin):
    """HBBs object.

    Attributes:
        albumentation_target_type: Type of target for Albumentations. Default is ``image``.

    Args:
        data: Input data as a ``torch.Tensor`` or ``numpy.ndarray``. If given,
            it must be in [N, 4+], XYXY format. Default is ``None``.
        orig_shape: Original shape of the image in [H, W] format.
        path: Label file path. Default is ``None``.
        root: Root directory for the label file. Default is ``None``.
        fmt: Bounding box format from ``'BBoxFormat'``. Default is ``'BBoxFormat.XYXY'``.

    Notes:
        - The bounding boxes are expected to in the following format:
            <x1, y1, x2, y2, conf, cls, id>
        where:
            <x1, y1, x2, y2> are the bounding box coordinates in XYXY format.
            <conf> is the confidence score (optional).
            <cls> is the class ID (optional).
            <id> is the tracking ID (optional).
    """

    albumentation_target_type: str = "bboxes"

    def __init__(
        self,
        data      : torch.Tensor | np.ndarray = None,
        orig_shape: tuple[int, int, int]      = None,
        path      : core.Path  = None,
        root      : core.Path  = None,
        fmt       : BBoxFormat = BBoxFormat.XYXY,
    ):
        if all(d is None for d in [data, path]):
            raise ValueError("Either [data] or [path] must be provided to initialize the Image object.")
        if data is not None and orig_shape is None:
            raise ValueError("If [data] is provided, [orig_shape] must also be specified.")
        if data is not None and orig_shape is not None:
            from mon.vision.types.bbox.hbb import utils
            if not utils.is_hbb_xyxy(data, orig_shape):
                raise ValueError(f"[data] must be in XYXY format, got {data.shape}.")

        super().__init__(data=data, orig_shape=orig_shape)
        self._path     = core.Path(path) if path is not None else None
        self._root     = core.Path(root) if root is not None else None
        self._orig_fmt = fmt
        self._cvt_fmt  = fmt
        self.fmt       = fmt

    @property
    def path(self) -> core.Path:
        """Returns the image file path."""
        return self._path

    @property
    def root(self) -> core.Path:
        """Returns the root directory for the image."""
        return self._root

    @property
    def fmt(self) -> BBoxFormat:
        """Returns the bounding box format."""
        return self._fmt

    @fmt.setter
    def fmt(self, fmt: BBoxFormat):
        """Sets the bounding box format.

        Args:
            fmt: Bounding box format from ``'BBoxFormat'``.
        """
        fmt = BBoxFormat.from_value(fmt)
        if fmt in BBoxFormat.conversion_codes():
            orig_fmt = BBoxFormat.from_value(fmt.value.split("_to_")[0])
        else:
            orig_fmt = fmt

        # We default to XYXY format for HBBs
        fmt = BBoxFormat.XYXY
        if orig_fmt != fmt:
            cvt_fmt = BBoxFormat.from_value(f"{orig_fmt.value}_to_{fmt.value}")
        else:
            cvt_fmt = fmt

        self._fmt      = fmt
        self._orig_fmt = orig_fmt
        self._cvt_fmt  = cvt_fmt

    @property
    def xyxy(self) -> torch.Tensor | np.ndarray:
        """Returns the bounding boxes in XYXY format (default).

        Returns:
            Bounding boxes as ``torch.Tensor`` or ``numpy.ndarray`` in [N, 4] format.
        """
        return self.data[:, :4]

    @property
    def conf(self) -> torch.Tensor | np.ndarray:
        """Return the confidence scores for each detection box.

        Returns:
            Confidence scores as ``torch.Tensor`` or ``numpy.ndarray`` in [N, 1] format.
        """
        if self.data.shape[1] > 4:
            return self.data[:, 4:5]
        return None

    @property
    def cls(self) -> torch.Tensor | np.ndarray:
        """Return the class ID tensor representing category predictions for each
        bounding box.

        Returns:
            Class labels as ``torch.Tensor`` or ``numpy.ndarray`` in [N, 1] format.
        """
        if self.data.shape[1] > 5:
            return self.data[:, 5:6]
        return None

    @property
    def id(self) -> torch.Tensor | np.ndarray:
        """Return the tracking IDs for each detection box if available.

        Returns:
            Tracking IDs as ``torch.Tensor`` or ``numpy.ndarray`` in [N, 1] format.

        Notes:
            - This property is only available when tracking is enabled.
            - The tracking IDs are typically used to associate detections across multiple frames in video analysis.
        """
        if self.data.shape[1] > 6:
            return self.data[:, 6:7]
        return None

    @property
    def xywh(self) -> torch.Tensor | np.ndarray:
        """Returns the bounding boxes in XYWH format.

        Returns:
            Bounding boxes as ``torch.Tensor`` or ``numpy.ndarray`` in [N, 4] format.
        """
        from mon.vision.types.bbox.hbb import processing
        return processing.hbb_xyxy_to_xywh(self.data[:, :4], self.orig_shape)

    @property
    def xyxyn(self) -> torch.Tensor | np.ndarray:
        """Returns the bounding boxes in XYXYN format.

        Returns:
            Bounding boxes as ``torch.Tensor`` or ``numpy.ndarray`` in [N, 4] format.
        """
        from mon.vision.types.bbox.hbb import processing
        return processing.normalize_hbb(self.data[:, :4], self.orig_shape)

    @property
    def cxcywhn(self) -> torch.Tensor | np.ndarray:
        """Returns the bounding boxes in CXCYWHN format.

        Returns:
            Bounding boxes as ``torch.Tensor`` or ``numpy.ndarray`` in [N, 4] format.
        """
        from mon.vision.types.bbox.hbb import processing
        return processing.hbb_xyxy_to_cxcywhn(self.data[:, :4], self.orig_shape)

    def load(self, reload: bool = False) -> np.ndarray:
        """Loads the image into memory.

        Args:
            reload: If ``True``, reload the image even if already cached. Default is ``False``.

        Returns:
            ``numpy.ndarray`` in [H, W, C] format, values in [0, 255].
        """
        # Return the bbox if it is already loaded and not reloading
        if not reload and self._data is not None:
            return self._data

        # Load the image
        from mon.vision.types.bbox.hbb import io
        bbox = io.load_hbb(
            path      = self.path,
            fmt       = self._cvt_fmt,
            imgsz     = self.orig_shape,
            to_tensor = False,
            normalize = False,
        )

        # Cache
        self._data = bbox
        return bbox

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
            orig_shape: Original shape of the image in [H, W] format.
            normalize: Normalize the underlying data. Default is ``True``.

        Returns:
            ``torch.Tensor`` of converted data.
        """
        from mon.vision.types.bbox.hbb import processing
        return processing.hbb_to_tensor(data, orig_shape, normalize)

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
        from mon.vision.types.bbox.hbb import processing
        return processing.hbb_to_3d(batch)

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements templates for image-only datasets."""

__all__ = [
    "ImageLoader",
]

import glob

from mon import core
from mon.constants import Split
from mon.vision.geometry import albumentation
from mon.vision.types.datasets import base
from mon.vision.types.image import Image


# ----- Image Loader -----
class ImageLoader(base.VisionDataset):
    """Loads images from a file path, pattern, or directory.
    
    Attributes:
        datapoint_attrs: Dict of attribute names and types.
        
    Args:
        root: A single image file path, pattern, or directory of images.
        split: Data split to use. Default is ``Split.PREDICT``.
        transform: Transformations to apply. Default is ``None``.
        to_tensor: If ``True``, converts to ``torch.Tensor``. Default is ``False``.
        cache_data: If ``True``, caches data to disk. Default is ``False``.
        verbose: If ``True``, enables verbose output. Default is ``True``.
    """
    
    datapoint_attrs = base.DatapointAttributes({
        "image": Image,
    })
    
    def __init__(
        self,
        root      : core.Path,
        split     : Split = Split.PREDICT,
        transform : albumentation.Compose = None,
        to_tensor : bool  = False,
        cache_data: bool  = False,
        verbose   : bool  = True,
        *args, **kwargs
    ):
        super().__init__(
            root        = root,
            split       = split,
            transform   = transform,
            to_tensor   = to_tensor,
            cache_data  = cache_data,
            verbose     = verbose,
            *args, **kwargs
        )
    
    # ----- Initialize -----
    def list_data(self):
        """Gets image data from the root path.

        Raises:
            IOError: If root path invalid or no images found.
        """
        if self.root.is_image_file():
            paths = [self.root]
        elif self.root.is_dir() and self.root.exists():
            paths = list(self.root.rglob("*"))
        elif "*" in str(self.root):
            paths = [core.Path(i) for i in glob.glob(str(self.root))]
        else:
            raise IOError(f"Invalid root path: {self.root}")
        
        images: list[Image] = []
        with core.create_progress_bar() as pbar:
            for path in pbar.track(
                sequence    = sorted(paths),
                description = f"[bright_yellow]Listing {self.__class__.__name__} "
                              f"{self.split_str} images"
            ):
                if path.is_image_file():
                    images.append(Image(path=path, root=self.root))
        
        self.datapoints["image"] = images

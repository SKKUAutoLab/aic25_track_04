#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements base classes and helpers for all metrics."""

__all__ = [
    "BootStrapper",
    "CatMetric",
    "ClasswiseWrapper",
    "MaxMetric",
    "MeanMetric",
    "Metric",
    "MetricCollection",
    "MetricTracker",
    "MinMaxMetric",
    "MinMetric",
    "MultioutputWrapper",
    "MultitaskWrapper",
    "RunningMean",
    "RunningSum",
    "SumMetric",
    "scale_gt_mean",
]

import abc
from typing import Literal

import cv2
import kornia
import numpy as np
import torch
import torchmetrics
from torchmetrics import (
    BootStrapper, CatMetric, ClasswiseWrapper, MaxMetric, MeanMetric, MetricCollection,
    MetricTracker, MinMaxMetric, MinMetric, MultioutputWrapper, MultitaskWrapper,
    RunningMean, RunningSum, SumMetric,
)


# ----- Base Metric -----
class Metric(torchmetrics.Metric, abc.ABC):
    """Base class for all metrics.

    Args:
        *args: Arguments passed to ``torchmetrics.Metric``.
        **kwargs: Keyword arguments passed to ``torchmetrics.Metric``.
    
    Attributes:
        mode: One of ``"FR"`` or ``"NR"``. Default is ``"FR"``.
        higher_is_better: ``True`` if higher values are better. Default is ``True``.
    """

    mode            : Literal["FR", "NR"] = "FR"
    higher_is_better: bool                = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


# ----- GT Mean -----
def scale_gt_mean(
    image : torch.Tensor | np.ndarray,
    target: torch.Tensor | np.ndarray
) -> torch.Tensor | np.ndarray:
    """Scales image to match target's mean intensity.

    Args:
        image: RGB image as ``torch.Tensor`` [B, C, H, W] in [0.0, 1.0] or
            ``numpy.ndarray`` [H, W, C] in [0, 255].
        target: Target image of same type as ``image``.
    
    Returns:
        Scaled image matching target's mean.
    
    Raises:
        TypeError: If ``image`` and ``target`` types differ.
    
    References:
        - https://github.com/Fediory/HVI-CIDNet/blob/master/measure.py
    """
    
    if isinstance(image, torch.Tensor) and isinstance(target, torch.Tensor):
        mean_image  = kornia.color.rgb_to_grayscale(image).mean()
        mean_target = kornia.color.rgb_to_grayscale(target).mean()
        image       = torch.clip(image * (mean_target / mean_image), 0, 1)
    elif isinstance(image, np.ndarray) and isinstance(target, np.ndarray):
        mean_image  = cv2.cvtColor(image,  cv2.COLOR_RGB2GRAY).mean()
        mean_target = cv2.cvtColor(target, cv2.COLOR_RGB2GRAY).mean()
        image       = np.clip(image * (mean_target / mean_image), 0, 255)
    else:
        raise TypeError(f"[image] and [target] must be same type, "
                        f"got {type(image).__name__} and {type(target).__name__}.")
    return image

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements efficiency score metrics."""

__all__ = [
	"compute_efficiency_score",
]

import torch
from fvcore.nn import FlopCountAnalysis, parameter_count
from torch.nn.common_types import _size_2_t

from mon import core


# ----- Efficiency -----
def compute_efficiency_score(
    model   : torch.nn.Module,
    imgsz   : _size_2_t = 512,
    channels: int       = 3
) -> tuple[float, float]:
    """Computes FLOPs and parameters for a model.

    Args:
        model: PyTorch model to profile.
        imgsz: Input image size (H, W) or single int. Default is ``512``.
        channels: Number of input channels. Default is ``3``.

    Returns:
        Tuple of (FLOPs, parameters) as floats.
    """
    from mon import vision

    h, w   = vision.image_size(imgsz)
    input  = torch.rand(1, channels, h, w).to(core.get_model_device(model))
    flops, params = core.thop.profile(model, inputs=(input,), verbose=False)

    flops  = FlopCountAnalysis(model, input).total() if flops == 0 else flops
    params = model.params           if hasattr(model, "params") and params == 0 else params
    params = parameter_count(model) if hasattr(model, "params") else params
    params = sum(params.values())   if isinstance(params, dict) else params

    return flops, params

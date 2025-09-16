#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Extends ``torch.nn.utils`` with utility functions for ``mon.nn``."""

__all__ = [
    "clip_grad_norm",
    "clip_grad_norm_",
    "clip_grad_value_",
    "convert_conv2d_weight_memory_format",
    "convert_conv3d_weight_memory_format",
    "fuse_conv_bn_eval",
    "fuse_conv_bn_weights",
    "fuse_linear_bn_eval",
    "fuse_linear_bn_weights",
    "parameters_to_vector",
    "parametrizations",
    "remove_spectral_norm",
    "remove_weight_norm",
    "rnn",
    "skip_init",
    "spectral_norm",
    "stateless",
    "vector_to_parameters",
    "weight_norm",
]

from torch.nn.utils import (
    clip_grad_norm, clip_grad_norm_, clip_grad_value_, convert_conv2d_weight_memory_format,
    convert_conv3d_weight_memory_format, fuse_conv_bn_eval, fuse_conv_bn_weights,
    fuse_linear_bn_eval, fuse_linear_bn_weights, parameters_to_vector,
    parametrizations, remove_spectral_norm, remove_weight_norm, rnn, skip_init,
    spectral_norm, stateless, vector_to_parameters, weight_norm,
)

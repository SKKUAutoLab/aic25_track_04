#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements edge and structural regularization.

These losses emphasize preserving or enhancing edges, boundaries, or structural details
in an image, critical for tasks like segmentation or enhancement.
"""

__all__ = [
    "EdgeLoss",
]

from typing import Literal

import torch
from torch.nn import functional as F

from mon.constants import LOSSES
from mon.nn.loss import base


# ----- Edge Loss -----
@LOSSES.register(name="edge_loss")
class EdgeLoss(base.Loss):
    """Edge Loss computes the difference in edge features between input and target
    using a Laplacian kernel.

    Args:
        reduction: Reduction method as ``Literal["none", "mean", "sum"]``.
            Default is ``"mean"``.

    Attributes:
        kernel: Gaussian kernel for convolution as ``torch.Tensor``.
        loss: Charbonnier loss function as ``base.CharbonnierLoss``.
    """
    
    def __init__(self, reduction: Literal["none", "mean", "sum"] = "mean"):
        super().__init__(reduction=reduction)
        k           = torch.Tensor([[0.05, 0.25, 0.4, 0.25, 0.05]])
        self.kernel = torch.matmul(k.t(), k).unsqueeze(0).repeat(3, 1, 1, 1)
        self.loss   = base.CharbonnierLoss()

    def gauss_conv(self, image: torch.Tensor) -> torch.Tensor:
        """Applies Gaussian convolution to the input image.

        Args:
            image: Input tensor as ``torch.Tensor`` with shape [B, C, H, W].

        Returns:
            Convolved tensor as ``torch.Tensor`` with shape [B, C, H, W].
        """
        b, c, w, h  = self.kernel.shape
        self.kernel = self.kernel.to(image.device)
        image       = F.pad(image, (w // 2, h // 2, w // 2, h // 2), mode="replicate")
        # gauss       = F.conv2d(image, self.kernel, groups=b)  # Old code
        gauss       = F.conv2d(image, self.kernel, groups=c)  # Groups=c for channel-wise convolution
        return gauss
    
    def laplacian_kernel(self, image: torch.Tensor) -> torch.Tensor:
        """Computes the Laplacian edge map using a Gaussian pyramid.

        Args:
            image: Input tensor as ``torch.Tensor`` with shape [B, C, H, W].

        Returns:
            Edge map tensor as ``torch.Tensor`` with shape [B, C, H, W].
        """
        filtered   = self.gauss_conv(image)       # filter
        down       = filtered[:, :, ::2, ::2]     # downsample
        new_filter = torch.zeros_like(filtered)
        new_filter[:, :, ::2, ::2] = down * 4     # upsample
        filtered   = self.gauss_conv(new_filter)  # filter
        diff       = image - filtered
        return diff
    
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Computes the edge loss between input and target tensors.

        Args:
            input: Input tensor as ``torch.Tensor`` with shape [B, C, H, W].
            target: Target tensor as ``torch.Tensor`` with shape [B, C, H, W].

        Returns:
            Loss value as ``torch.Tensor``.
        """
        edge1 = self.laplacian_kernel(input)
        edge2 = self.laplacian_kernel(target)
        diff  = edge1 - edge2
        loss  = torch.mean(torch.sqrt((diff * diff) + (self.eps * self.eps)))
        loss  = base.reduce_loss(loss=loss, reduction=self.reduction)
        return loss

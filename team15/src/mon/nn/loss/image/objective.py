#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements objective quality assessment

These loss functions measure the similarity or fidelity between a generated/reconstructed
image and a reference (ground truth) image, often based on objective metrics used in
image quality assessment.
"""

__all__ = [
    "MS_SSIMLoss",
    "PSNRLoss",
    "SSIMLoss",
]

from typing import Literal

import numpy as np
import torch

from mon.constants import LOSSES
from mon.nn.loss import base


# ----- PSNR Loss -----
@LOSSES.register(name="psnr_loss")
class PSNRLoss(base.Loss):
    """PSNR Loss computes the Peak Signal-to-Noise Ratio loss between input and target
    images.

    Args:
        to_y: Converts RGB to Y-channel (luminance) if ``True``. Default is ``False``.
        reduction: Reduction method as ``Literal["none", "mean", "sum"]``.
            Default is ``"mean"``.

    References:
        - https://github.com/xinntao/BasicSR
    """
    
    def __init__(self, to_y: bool = False, reduction: Literal["none", "mean", "sum"] = "mean"):
        super().__init__(reduction=reduction)
        self.scale = 10 / np.log(10)
        self.to_y  = to_y
        self.coef  = torch.tensor([65.481, 128.553, 24.966]).reshape(1, 3, 1, 1)
        self.first = True
    
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Computes the PSNR loss between input and target tensors.

        Args:
            input: Input tensor as ``torch.Tensor`` with shape [B, C, H, W].
            target: Target tensor as ``torch.Tensor`` with shape [B, C, H, W].

        Returns:
            Loss value as ``torch.Tensor``.
        """
        x = input
        y = target
        
        if self.to_y:
            if self.first:
                self.coef = self.coef.to(x.device)
                self.first = False
            
            # Convert RGB to Y-channel (luminance) using ITU-R BT.601 coefficients
            x = (x * self.coef).sum(dim=1, keepdim=True) + 16.0  # [B, 1, H, W]
            y = (y * self.coef).sum(dim=1, keepdim=True) + 16.0  # [B, 1, H, W]
            x = x / 255.0
            y = y / 255.0
        
        # Compute Mean Squared Error (MSE) and PSNR
        mse  = torch.mean((x - y) ** 2, dim=[1, 2, 3])  # [B]
        psnr = -self.scale * torch.log10(mse + 1e-8)   # [B], negative PSNR as loss (higher PSNR = lower loss)
        
        # Apply reduction
        loss = base.reduce_loss(loss=psnr, reduction=self.reduction)
        return loss


# ----- SSIM Loss -----
@LOSSES.register(name="ssim_loss")
class SSIMLoss(base.Loss):
    """SSIM Loss computes the Structural Similarity Index Measure loss between input
    and target images.

    Args:
        data_range: Range of input data as ``float``. Default is ``255``.
        size_average: Average over each image if ``True``. Default is ``True``.
        window_size: Size of the Gaussian window as ``int``. Default is ``11``.
        window_sigma: Standard deviation of the Gaussian window as ``float``.
            Default is ``1.5``.
        channel: Number of channels in the input as ``int``. Default is ``3``.
        spatial_dims: Number of spatial dimensions as ``int``. Default is ``2``.
        k: Constants for SSIM calculation as ``tuple[float, float]`` (k1, k2).
            Default is ``(0.01, 0.03)``.
        non_negative_ssim: Ensures non-negative SSIM if ``True``. Default is ``False``.
        reduction: Reduction method as ``Literal["none", "mean", "sum"]``.
            Default is ``"mean"``.
    """
    
    def __init__(
        self,
        data_range       : float = 255,
        size_average     : bool  = True,
        window_size      : int   = 11,
        window_sigma     : float = 1.5,
        channel          : int   = 3,
        spatial_dims     : int   = 2,
        k                : tuple[float, float] = (0.01, 0.03),
        non_negative_ssim: bool  = False,
        reduction        : Literal["none", "mean", "sum"] = "mean",
    ):
        super().__init__(reduction=reduction)
        from mon.nn.metric.image.pytorch_msssim import SSIM
        self.ssim = SSIM(
            data_range        = data_range,
            size_average      = size_average,
            window_size       = window_size,
            window_sigma      = window_sigma,
            channel           = channel,
            spatial_dims      = spatial_dims,
            k                 = k,
            non_negative_ssim = non_negative_ssim,
        )
    
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Computes the SSIM loss between input and target tensors.

        Args:
            input: Input tensor as ``torch.Tensor`` with shape [B, C, H, W].
            target: Target tensor as ``torch.Tensor`` with shape [B, C, H, W].

        Returns:
            Loss value as ``torch.Tensor``.
        """
        loss = 1.0 - self.ssim(input, target)
        loss = base.reduce_loss(loss=loss, reduction=self.reduction)
        return loss
    

@LOSSES.register(name="ms_ssim_loss")
class MS_SSIMLoss(base.Loss):
    """MS-SSIM Loss computes the Multi-Scale Structural Similarity Index Measure loss between input and target images.

    Args:
        data_range: Range of input data as ``float``. Default is ``255``.
        size_average: Average over each image if ``True``. Default is ``True``.
        window_size: Size of the Gaussian window as ``int``. Default is ``11``.
        window_sigma: Standard deviation of the Gaussian window as ``float``.
            Default is ``1.5``.
        channel: Number of channels in the input as ``int``. Default is ``3``.
        spatial_dims: Number of spatial dimensions as ``int``. Default is ``2``.
        weights: Weights for each scale as ``list[float]`` or ``None``.
            Default is ``None``.
        k: Constants for SSIM calculation as ``tuple[float, float]`` (k1, k2).
            Default is ``(0.01, 0.03)``.
        reduction: Reduction method as ``Literal["none", "mean", "sum"]``.
            Default is ``"mean"``.
    """
    
    def __init__(
        self,
        data_range  : float = 255,
        size_average: bool  = True,
        window_size : int   = 11,
        window_sigma: float = 1.5,
        channel     : int   = 3,
        spatial_dims: int   = 2,
        weights     : list[float] = None,
        k           : tuple[float, float] = (0.01, 0.03),
        reduction   : Literal["none", "mean", "sum"] = "mean",
    ):
        super().__init__(reduction=reduction)
        from mon.nn.metric.image.pytorch_msssim import MS_SSIM
        self.ms_ssim = MS_SSIM(
            data_range   = data_range,
            size_average = size_average,
            window_size  = window_size,
            window_sigma = window_sigma,
            channel      = channel,
            spatial_dims = spatial_dims,
            weights      = weights,
            k            = k,
        )
    
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Computes the MS-SSIM loss between input and target tensors.

        Args:
            input: Input tensor as ``torch.Tensor`` with shape [B, C, H, W].
            target: Target tensor as ``torch.Tensor`` with shape [B, C, H, W].

        Returns:
            Loss value as ``torch.Tensor``.
        """
        loss = 1.0 - self.ms_ssim(input, target)
        loss = base.reduce_loss(loss=loss, reduction=self.reduction)
        return loss

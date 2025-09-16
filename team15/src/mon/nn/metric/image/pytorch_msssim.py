#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements fast, differentiable MS-SSIM and SSIM for PyTorch.

References:
    - https://github.com/VainF/pytorch-msssim
"""

__all__ = [
    "MS_SSIM",
    "SSIM",
    "ms_ssim",
    "ssim",
]

import torch
import torch.nn.functional as F


def _fspecial_gauss_1d(size: int, sigma: float) -> torch.Tensor:
    """Creates a 1D Gaussian kernel.

    Args:
        size: Size of the Gaussian kernel as ``int``.
        sigma: Standard deviation of the Gaussian as ``float``.

    Returns:
        1D kernel tensor as ``torch.Tensor`` with shape `1, 1, size].

    References:
        - https://github.com/VainF/pytorch-msssim/blob/master/pytorch_msssim/ssim.py
    """
    coords  = torch.arange(size, dtype=torch.float)
    coords -= size // 2
    g       = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g      /= g.sum()
    return g.unsqueeze(0).unsqueeze(0)


def _gaussian_filter(input: torch.Tensor, window: torch.Tensor) -> torch.Tensor:
    """Blurs input tensor with a 1D Gaussian kernel.

    Args:
        input: Batch of tensors to blur as ``torch.Tensor``.
        window: 1D Gaussian kernel tensor as ``torch.Tensor``.

    Returns:
        Blurred tensor as ``torch.Tensor`` with same shape as input.

    Raises:
        AssertionError: If ``window`` shape is not [1, ..., 1, size].
        NotImplementedError: If ``input`` shape is not 4D or 5D.

    References:
        - https://github.com/VainF/pytorch-msssim/blob/master/pytorch_msssim/ssim.py
    """
    if not all(ws == 1 for ws in window.shape[1:-1]):
        raise AssertionError(f"[window] must have shape [1, ..., 1, size], got {window.shape}.")

    if len(input.shape) == 4:
        conv = F.conv2d
    elif len(input.shape) == 5:
        conv = F.conv3d
    else:
        raise NotImplementedError(f"[input] must be 4D or 5D, got {input.shape}.")

    c   = input.shape[1]
    out = input
    for i, s in enumerate(input.shape[2:]):
        if s >= window.shape[-1]:
            out = conv(out, weight=window.transpose(2 + i, -1), stride=1, padding=0, groups=c)

    return out


def _ssim(
    image1      : torch.Tensor,
    image2      : torch.Tensor,
    data_range  : float,
    window      : torch.Tensor,
    size_average: bool = True,
    k           : tuple[float, float] = (0.01, 0.03)
) -> tuple[torch.Tensor, torch.Tensor]:
    """Computes SSIM and contrast sensitivity between two images.

    Args:
        image1: First image tensor as ``torch.Tensor`` with shape [B, C, H, W]
            or [B, C, D, H, W].
        image2: Second image tensor as ``torch.Tensor`` with same shape as ``image1``.
        data_range: Value range of images as ``float`` (e.g., ``1.0`` or ``255.0``).
        window: 1D Gaussian kernel tensor as ``torch.Tensor``.
        size_average: Averages SSIM over channels if ``True``. Default is ``True``.
        k: Stability constants (k1, k2) as ``tuple[float, float]``.
            Default is ``(0.01, 0.03)``.

    Returns:
        ``tuple`` of (SSIM per channel, contrast sensitivity) as ``torch.Tensor``s.

    References:
        - https://github.com/VainF/pytorch-msssim/blob/master/pytorch_msssim/ssim.py
    """
    k1, k2 = k
    compensation = 1.0

    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2

    window = window.to(image1.device, dtype=image1.dtype)

    mu1 = _gaussian_filter(image1, window)
    mu2 = _gaussian_filter(image2, window)

    mu1_sq  = mu1.pow(2)
    mu2_sq  = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = compensation * (_gaussian_filter(image1 * image1, window) - mu1_sq)
    sigma2_sq = compensation * (_gaussian_filter(image2 * image2, window) - mu2_sq)
    sigma12   = compensation * (_gaussian_filter(image1 * image2, window) - mu1_mu2)

    cs_map   = (2 * sigma12 + c2) / (sigma1_sq + sigma2_sq + c2)
    ssim_map = ((2 * mu1_mu2 + c1) / (mu1_sq + mu2_sq + c1)) * cs_map

    ssim_per_channel = torch.flatten(ssim_map, 2).mean(-1)
    cs               = torch.flatten(cs_map, 2).mean(-1)

    return ssim_per_channel, cs


def ssim(
    image1           : torch.Tensor,
    image2           : torch.Tensor,
    data_range       : float = 255,
    size_average     : bool  = True,
    window_size      : int   = 11,
    window_sigma     : float = 1.5,
    window           : torch.Tensor = None,
    k                : tuple[float, float] = (0.01, 0.03),
    non_negative_ssim: bool  = False
) -> torch.Tensor:
    """Computes SSIM between two images.

    Args:
        image1: First image tensor as ``torch.Tensor`` with shape [B, C, H, W]
            or [B, C, D, H, W].
        image2: Second image tensor as ``torch.Tensor`` with same shape as ``image1``.
        data_range: Value range of images as ``float`` (e.g., ``1.0`` or ``255.0``).
            Default is ``255.0``.
        size_average: Averages SSIM over channels if ``True``. Default is ``True``.
        window_size: Size of the Gaussian window as ``int``. Default is ``11``.
        window_sigma: Gaussian sigma as ``float``. Default is ``1.5``.
        window: Optional 1D Gaussian kernel tensor as ``torch.Tensor`` or ``None``.
            Default is ``None``.
        k: Stability constants (k1, k2) as ``tuple[float, float]``.
            Default is ``(0.01, 0.03)``.
        non_negative_ssim: Applies ReLU to SSIM if ``True``. Default is ``False``.

    Returns:
        SSIM tensor as ``torch.Tensor``, averaged if ``size_average`` is ``True``.

    Raises:
        ValueError: If shapes mismatch, ``window_size`` is even, or dims are invalid.

    References:
        - https://github.com/VainF/pytorch-msssim/blob/master/pytorch_msssim/ssim.py
    """
    if image1.shape != image2.shape:
        raise ValueError(f"[image1] and [image2] must have same shape, "
                         f"got {image1.shape} and {image2.shape}.")

    for d in range(len(image1.shape) - 1, 1, -1):
        image1 = image1.squeeze(dim=d)
        image2 = image2.squeeze(dim=d)

    if len(image1.shape) not in (4, 5):
        raise ValueError(f"[image1] and [image2] must be 4D or 5D, got {image1.shape}.")

    if window:
        window_size = window.shape[-1]

    if window_size % 2 != 1:
        raise ValueError(f"[window_size] must be odd, got {window_size}.")

    if window is None:
        window = _fspecial_gauss_1d(window_size, window_sigma)
        window = window.repeat([image1.shape[1]] + [1] * (len(image1.shape) - 1))

    ssim_per_channel, _ = _ssim(
        image1       = image1,
        image2       = image2,
        data_range   = data_range,
        window       = window,
        size_average = False,
        k            = k
    )
    if non_negative_ssim:
        ssim_per_channel = torch.relu(ssim_per_channel)

    return ssim_per_channel.mean() if size_average else ssim_per_channel.mean(1)


def ms_ssim(
    image1      : torch.Tensor,
    image2      : torch.Tensor,
    data_range  : float        = 255,
    size_average: bool         = True,
    window_size : int          = 11,
    window_sigma: float        = 1.5,
    window      : torch.Tensor = None,
    weights     : list[float]  = None,
    k           : tuple[float, float] = (0.01, 0.03)
) -> torch.Tensor:
    """Computes Multi-Scale SSIM between two images.

    Args:
        image1: First image tensor as ``torch.Tensor`` with shape [B, C, H, W]
            or [B, C, D, H, W].
        image2: Second image tensor as ``torch.Tensor`` with same shape as ``image1``.
        data_range: Value range of images as ``float`` (e.g., ``1.0`` or ``255.0``).
            Default is ``255.0``.
        size_average: Averages MS-SSIM over channels if ``True``. Default is ``True``.
        window_size: Size of the Gaussian window as ``int``. Default is ``11``.
        window_sigma: Gaussian sigma as ``float``. Default is ``1.5``.
        window: Optional 1D Gaussian kernel tensor as ``torch.Tensor`` or ``None``.
            Default is ``None``.
        weights: Weights for each scale as ``list[float]`` or ``None``.
            Default is [0.0448, 0.2856, 0.3001, 0.2363, 0.1333].
        k: Stability constants (k1, k2) as ``tuple[float, float]``.
            Default is ``(0.01, 0.03)``.

    Returns:
        MS-SSIM tensor as ``torch.Tensor``, averaged if ``size_average`` is ``True``.

    Raises:
        ValueError: If shapes mismatch, ``window_size`` is even, or dims are invalid.
        AssertionError: If image size is too small for 4 downsamplings.

    References:
        - https://github.com/VainF/pytorch-msssim/blob/master/pytorch_msssim/ssim.py
    """
    if image1.shape != image2.shape:
        raise ValueError(f"[image1] and [image2] must have same shape, "
                         f"got {image1.shape} and {image2.shape}.")

    for d in range(len(image1.shape) - 1, 1, -1):
        image1 = image1.squeeze(dim=d)
        image2 = image2.squeeze(dim=d)

    if len(image1.shape) == 4:
        avg_pool = F.avg_pool2d
    elif len(image1.shape) == 5:
        avg_pool = F.avg_pool3d
    else:
        raise ValueError(f"[image1] and [image2] must be 4D or 5D, got {image1.shape}.")

    if window:
        window_size = window.shape[-1]

    if window_size % 2 != 1:
        raise ValueError(f"[window_size] must be odd, got {window_size}.")

    smaller_side = min(image1.shape[-2:])
    if smaller_side <= (window_size - 1) * (2 ** 4):
        raise AssertionError(
            f"[image1] and [image2] must be larger than [{(window_size - 1) * (2 ** 4)}] "
            f"for 4 downsamplings, got {smaller_side}."
        )

    if weights is None:
        weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
    weights_tensor = image1.new_tensor(weights)

    if window is None:
        window = _fspecial_gauss_1d(window_size, window_sigma)
        window = window.repeat([image1.shape[1]] + [1] * (len(image1.shape) - 1))

    levels = weights_tensor.shape[0]
    mcs = []
    for i in range(levels):
        ssim_per_channel, cs = _ssim(
            image1       = image1,
            image2       = image2,
            data_range   = data_range,
            window       = window,
            size_average = False,
            k            = k
        )
        if i < levels - 1:
            mcs.append(torch.relu(cs))
            padding = [s % 2 for s in image1.shape[2:]]
            image1 = avg_pool(image1, kernel_size=2, padding=padding)
            image2 = avg_pool(image2, kernel_size=2, padding=padding)

    ssim_per_channel = torch.relu(ssim_per_channel)
    mcs_and_ssim     = torch.stack(mcs + [ssim_per_channel], dim=0)
    ms_ssim_val      = torch.prod(mcs_and_ssim ** weights_tensor.view(-1, 1, 1), dim=0)

    return ms_ssim_val.mean() if size_average else ms_ssim_val.mean(1)


class SSIM(torch.nn.Module):
    """Structural Similarity Index (SSIM) module for image comparison.

    Args:
        data_range: Value range of images as ``float`` (e.g., ``1.0`` or ``255.0``).
            Default is ``255.0``.
        size_average: Averages SSIM over channels if ``True``. Default is ``True``.
        window_size: Size of the Gaussian window as ``int``. Default is ``11``.
        window_sigma: Gaussian sigma as ``float``. Default is ``1.5``.
        channel: Number of input channels as ``int``. Default is ``3``.
        spatial_dims: Number of spatial dimensions as ``int`` (2 or 3). Default is ``2``.
        k: Stability constants (k1, k2) as ``tuple[float, float]``.
            Default is ``(0.01, 0.03)``.
        non_negative_ssim: Applies ReLU to SSIM if ``True``. Default is ``False``.
    """

    def __init__(
        self,
        data_range  : float = 255,
        size_average: bool  = True,
        window_size : int   = 11,
        window_sigma: float = 1.5,
        channel     : int   = 3,
        spatial_dims: int   = 2,
        k           : tuple[float, float] = (0.01, 0.03),
        non_negative_ssim: bool = False
    ):
        super().__init__()
        self.window_size  = window_size
        self.window       = _fspecial_gauss_1d(window_size, window_sigma).repeat([channel, 1] + [1] * spatial_dims)
        self.size_average = size_average
        self.data_range   = data_range
        self.k            = k
        self.non_negative_ssim = non_negative_ssim

    def forward(self, image1: torch.Tensor, image2: torch.Tensor) -> torch.Tensor:
        """Computes SSIM between two images.

        Args:
            image1: First image tensor as ``torch.Tensor`` with shape [B, C, H, W]
                or [B, C, D, H, W].
            image2: Second image tensor as ``torch.Tensor`` with same shape as ``image1``.

        Returns:
            SSIM tensor as ``torch.Tensor``, averaged if ``size_average`` is ``True``.
        """
        return ssim(
            image1            = image1,
            image2            = image2,
            data_range        = self.data_range,
            size_average      = self.size_average,
            window            = self.window,
            k                 = self.k,
            non_negative_ssim = self.non_negative_ssim
        )


class MS_SSIM(torch.nn.Module):
    """Multi-Scale Structural Similarity Index (MS-SSIM) module.

    Args:
        data_range: Value range of images as ``float`` (e.g., ``1.0`` or ``255.0``).
            Default is ``255.0``.
        size_average: Averages MS-SSIM over channels if ``True``. Default is ``True``.
        window_size: Size of the Gaussian window as ``int``. Default is ``11``.
        window_sigma: Gaussian sigma as ``float``. Default is ``1.5``.
        channel: Number of input channels as ``int``. Default is ``3``.
        spatial_dims: Number of spatial dimensions as ``int`` (2 or 3). Default is ``2``.
        weights: Weights for each scale as ``list[float]`` or ``None``.
            Default is ``None`` (preset values).
        k: Stability constants (k1, k2) as ``tuple[float, float]``.
            Default is ``(0.01, 0.03)``.
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
        k           : tuple[float, float] = (0.01, 0.03)
    ):
        super().__init__()
        self.window_size  = window_size
        self.window       = _fspecial_gauss_1d(window_size, window_sigma).repeat([channel, 1] + [1] * spatial_dims)
        self.size_average = size_average
        self.data_range   = data_range
        self.weights      = weights
        self.k            = k

    def forward(self, image1: torch.Tensor, image2: torch.Tensor) -> torch.Tensor:
        """Computes MS-SSIM between two images.

        Args:
            image1: First image tensor as ``torch.Tensor`` with shape [B, C, H, W]
                or [B, C, D, H, W].
            image2: Second image tensor as ``torch.Tensor`` with same shape as ``image1``.

        Returns:
            MS-SSIM tensor as ``torch.Tensor``, averaged if ``size_average`` is ``True``.
        """
        return ms_ssim(
            image1       = image1,
            image2       = image2,
            data_range   = self.data_range,
            size_average = self.size_average,
            window       = self.window,
            weights      = self.weights,
            k            = self.k
        )

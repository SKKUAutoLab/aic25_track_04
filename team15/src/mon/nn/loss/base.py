#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements base class and basic loss functions with helpers."""

__all__ = [
    "BCELoss",
    "BCEWithLogitsLoss",
    "CTCLoss",
    "CharbonnierLoss",
    "CosineEmbeddingLoss",
    "CosineSimilarityLoss",
    "CrossEntropyLoss",
    "ExtendedL1Loss",
    "GaussianNLLLoss",
    "HingeEmbeddingLoss",
    "HuberLoss",
    "KLDivLoss",
    "L1Loss",
    "Loss",
    "MSELoss",
    "MarginRankingLoss",
    "MultiLabelMarginLoss",
    "MultiLabelSoftMarginLoss",
    "MultiMarginLoss",
    "NLLLoss",
    "PoissonNLLLoss",
    "SmoothL1Loss",
    "SoftMarginLoss",
    "TripletMarginLoss",
    "TripletMarginWithDistanceLoss",
    "reduce_loss",
]

import abc
from typing import Literal

import torch
from torch.nn.modules.loss import (
    _Loss, BCELoss, BCEWithLogitsLoss, CosineEmbeddingLoss, CrossEntropyLoss, CTCLoss,
    GaussianNLLLoss, HingeEmbeddingLoss, HuberLoss, KLDivLoss, L1Loss,
    MarginRankingLoss, MSELoss, MultiLabelMarginLoss, MultiLabelSoftMarginLoss,
    MultiMarginLoss, NLLLoss, PoissonNLLLoss, SmoothL1Loss, SoftMarginLoss,
    TripletMarginLoss, TripletMarginWithDistanceLoss,
)

from mon import core
from mon.constants import LOSSES


# ----- Base Loss -----
def reduce_loss(
    loss     : torch.Tensor,
    reduction: Literal["mean", "sum", "none"] = "mean"
) -> torch.Tensor:
    """Reduces the loss tensor.

    Args:
        loss: Elementwise loss tensor as ``torch.Tensor``.
        reduction: Reduction value as ``"mean"``, ``"sum"``, or ``"none"``.
            Default is ``"mean"``.

    Returns:
        Reduced loss as ``torch.Tensor``.
    """
    return {"mean": torch.mean, "sum": torch.sum, "none": lambda x: x}[reduction](loss)


class Loss(_Loss, abc.ABC):
    """The base class for all loss functions.
    
    Args:
        reduction: Specifies the reduction to apply to the output. One of:
            - ``'none'``: No reduction will be applied.
            - ``'mean'``: The sum of the output will be divided by the number of
                elements in the output.
            - ``'sum'``: The output will be summed.
            - Default: ``'mean'``.
    """
    
    reductions = ["none", "mean", "sum"]
    
    def __init__(self, reduction: Literal["none", "mean", "sum"] = "mean"):
        super().__init__(reduction=reduction)
        if self.reduction not in self.reductions:
            raise ValueError(f"[reduction] must be one of: {self.reductions}, got {reduction}.")
        
    def __str__(self):
        """Returns a string representation of the object.
    
        Returns:
            Class name as lowercase kebab-case ``str``.
        """
        return core.humps.depascalize(self.__class__.__name__).lower()
    
    @abc.abstractmethod
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Performs forward pass with input and target tensors.
    
        Args:
            input: Input data as ``torch.Tensor``.
            target: Target data as ``torch.Tensor``.
    
        Returns:
            Output as ``torch.Tensor``.
        """
        pass


# ----- Core Loss -----
@LOSSES.register(name="charbonnier_loss")
class CharbonnierLoss(Loss):
    """Computes the Charbonnier loss between input and target tensors.

    Args:
        eps: Small constant for numerical stability. Default is ``1e-3``.
        reduction: Reduction method: ``"none"``, ``"mean"``, or ``"sum"``.
            Default is ``"mean"``.`.
    """
    
    def __init__(
        self,
        eps      : float = 1e-3,
        reduction: Literal["none", "mean", "sum"] = "mean",
    ):
        super().__init__(reduction=reduction)
        self.eps = eps
    
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Computes the Charbonnier loss.

        Args:
            input: Predicted tensor as ``torch.Tensor``.
            target: Target tensor as ``torch.Tensor``.

        Returns:
            Reduced loss as ``torch.Tensor``.
        """
        # loss = torch.sqrt((input - target) ** 2 + (self.eps * self.eps))
        diff = input - target
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps * self.eps)))
        return reduce_loss(loss=loss, reduction=self.reduction)


@LOSSES.register(name="cosine_similarity_loss")
class CosineSimilarityLoss(Loss):
    """Computes cosine similarity loss between input and target tensors.

    Args:
        dim: Dimension for cosine similarity. Default is ``1``.
        eps: Small constant for numerical stability. Default is ``1e-6``.
        reduction: Reduction method: ``"none"``, ``"mean"``, or ``"sum"``.
            Default is ``"mean"``.
    """
    
    def __init__(
        self,
        dim      : int = 1,
        eps      : float = 1e-6,
        reduction: Literal["none", "mean", "sum"] = "mean",
    ):
        super().__init__(reduction=reduction)
        self.cos = torch.nn.CosineSimilarity(dim=dim, eps=eps)
    
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Computes the cosine similarity loss.

        Args:
            input: Predicted tensor as ``torch.Tensor`` of shape [B, C, H, W].
            target: Target tensor as ``torch.Tensor`` of shape [B, C, H, W].

        Returns:
            Loss as ``torch.Tensor``.
        """
        b, c, h, w = input.size()
        x    = input.permute(0, 2, 3, 1).view(-1, c)
        y    = target.permute(0, 2, 3, 1).view(-1, c)
        loss = 1.0 - self.cos(x, y).sum() / (1.0 * b * h * w)
        return loss


@LOSSES.register(name="extended_l1_loss")
class ExtendedL1Loss(Loss):
    """Computes extended L1 loss with mask normalization.

    Args:
        reduction: Reduction method: ``"none"``, ``"mean"``, or ``"sum"``.
            Default is ``"mean"``.
    """
    
    def __init__(self, reduction: Literal["none", "mean", "sum"] = "mean"):
        super().__init__(reduction=reduction)
        self.loss_l1 = L1Loss()
    
    # noinspection PyMethodOverriding
    def forward(
        self,
        input : torch.Tensor,
        target: torch.Tensor,
        mask  : torch.Tensor
    ) -> torch.Tensor:
        """Computes the extended L1 loss with mask.

        Args:
            input: Predicted tensor as ``torch.Tensor``.
            target: Target tensor as ``torch.Tensor``.
            mask: Mask tensor as ``torch.Tensor`` for weighting.

        Returns:
            Reduced loss as ``torch.Tensor``.
        """
        norm = self.loss_l1(mask, torch.zeros_like(mask))
        loss = self.loss_l1(mask * input, mask * target) / norm
        return reduce_loss(loss=loss, reduction=self.reduction)
    

# ----- Registering -----
LOSSES.register(name="bce_loss",                          module=BCELoss)
LOSSES.register(name="bce_with_logits_loss",              module=BCEWithLogitsLoss)
LOSSES.register(name="cosine_embedding_loss",             module=CosineEmbeddingLoss)
LOSSES.register(name="cross_entropy_loss",                module=CrossEntropyLoss)
LOSSES.register(name="ctc_loss",                          module=CTCLoss)
LOSSES.register(name="gaussian_nll_loss",                 module=GaussianNLLLoss)
LOSSES.register(name="hinge_embedding_loss",              module=HingeEmbeddingLoss)
LOSSES.register(name="huber_loss",                        module=HuberLoss)
LOSSES.register(name="kl_div_loss",                       module=KLDivLoss)
LOSSES.register(name="l1_loss",                           module=L1Loss)
LOSSES.register(name="margin_ranking_loss",               module=MarginRankingLoss)
LOSSES.register(name="mse_loss",                          module=MSELoss)
LOSSES.register(name="multi_label_margin_loss",           module=MultiLabelMarginLoss)
LOSSES.register(name="multi_label_soft_margin_loss",      module=MultiLabelSoftMarginLoss)
LOSSES.register(name="multi_margin_loss",                 module=MultiMarginLoss)
LOSSES.register(name="nll_loss",                          module=NLLLoss)
LOSSES.register(name="poisson_nll_loss",                  module=PoissonNLLLoss)
LOSSES.register(name="smooth_l1_loss",                    module=SmoothL1Loss)
LOSSES.register(name="soft_margin_loss",                  module=SoftMarginLoss)
LOSSES.register(name="triplet_margin_loss",               module=TripletMarginLoss)
LOSSES.register(name="triplet_margin_with_distance_Loss", module=TripletMarginWithDistanceLoss)

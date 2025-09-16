#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements perceptual and feature-based losses.

These losses leverage high-level features extracted from pretrained neural networks
(e.g., VGG) to align the perceptual or semantic content of images, rather than just
pixel-wise differences.
"""

__all__ = [
    "PerceptualLoss",
]

from typing import Literal

import torch
from torchvision import models, transforms

from mon.constants import LOSSES
from mon.nn.loss import base


# ----- Perceptual Loss -----
@LOSSES.register(name="perceptual_loss")
class PerceptualLoss(base.Loss):
    """Perceptual Loss computes feature differences between input and target using a pretrained network.

    Args:
        net: Pretrained network as ``torch.nn.Module`` or ``str``; options: ``"alexnet"``,
            ``"vgg11"``, ``"vgg13"``, ``"vgg16"``, ``"vgg19"``. Default is ``"vgg19"``.
        layers: List of layer indices to extract features from as ``list[str]``.
            Default is ["26"].
        preprocess: Applies normalization if ``True``. Default is ``False``.
        reduction: Reduction method as ``Literal["none", "mean", "sum"]``. Default is ``"mean"``.
    """
    
    def __init__(
        self,
        net       : torch.nn.Module | str = "vgg19",
        layers    : list  = ["26"],
        preprocess: bool  = False,
        reduction : Literal["none", "mean", "sum"] = "mean"
    ):
        super().__init__(reduction=reduction)
        self.layers     = layers
        self.preprocess = preprocess
        
        if net in ["alexnet"]:
            net = models.alexnet(weights=models.AlexNet_Weights).features
        elif net in ["vgg11"]:
            net = models.vgg11(weights=models.VGG11_Weights).features
        elif net in ["vgg13"]:
            net = models.vgg13(weights=models.VGG13_Weights).features
        elif net in ["vgg16"]:
            net = models.vgg16(weights=models.VGG16_Weights).features
        elif net in ["vgg19"]:
            net = models.vgg19(weights=models.VGG19_Weights).features
        
        self.net     = net.eval()
        self.l1_loss = base.L1Loss(reduction=reduction)
        
        # Disable gradient computation for net's parameters
        for param in self.net.parameters():
            param.requires_grad = False
    
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Computes the perceptual loss between input and target tensors.

        Args:
            input: Input tensor as ``torch.Tensor`` with shape [B, C, H, W].
            target: Target tensor as ``torch.Tensor`` with shape [B, C, H, W].

        Returns:
            Loss value as ``torch.Tensor``.
        """
        if self.preprocess:
            input  = self.run_preprocess(input)
            target = self.run_preprocess(target)
        input_feats  = self.get_features(input)
        target_feats = self.get_features(target)
        
        loss = 0
        for xf, yf in zip(input_feats, target_feats):
            loss += self.l1_loss(xf, yf)
        loss = loss / len(input_feats)
        return loss
    
    @staticmethod
    def run_preprocess(input: torch.Tensor) -> torch.Tensor:
        """Applies normalization preprocessing to the input tensor.

        Args:
            input: Input tensor as ``torch.Tensor`` with shape [B, C, H, W].

        Returns:
            Preprocessed tensor as ``torch.Tensor`` with shape [B, C, H, W].
        """
        transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        input     = transform(input)
        return input
    
    def get_features(self, input: torch.Tensor) -> list[torch.Tensor]:
        """Extracts features from specified layers of the network.

        Args:
            input: Input tensor as ``torch.Tensor`` with shape [B, C, H, W].

        Returns:
            List of feature tensors as ``list[torch.Tensor]``, shapes vary by layer.
        """
        x        = input
        features = []
        for name, layer in self.net._modules.items():
            x = layer(x)
            if name in self.layers:
                features.append(x)
        return features

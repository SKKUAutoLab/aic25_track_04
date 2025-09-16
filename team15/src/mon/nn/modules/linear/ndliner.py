#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements NdLinear: next-gen replacement for ``nn.Linear``.

Reference:
    - https://github.com/ensemble-core/ndlinear
"""

__all__ = [
    "NdLinear",
    "NdLinearGated",
]

from typing import Literal

import torch


class NdLinear(torch.nn.Module):
    """NdLinear: A PyTorch layer for projecting tensors into multi-space representations.
    
    Unlike conventional embedding layers that map into a single vector space, NdLinear
    transforms tensors across a collection of vector spaces, capturing multivariate
    structure and topical information that standard deep learning architectures
    typically lose.

    Args:
        input_dims: Shape of input tensor (excluding batch dimension).
        hidden_size: Target hidden dimensions after transformation.
        bias: Uses bias in linear layer if ``True``. Default is ``True``.
        transform_outer: If ``True``, transforms from outer to inner dimensions.
    """
    
    def __init__(
        self,
        input_dims     : list | tuple,
        hidden_size    : list | tuple,
        bias           : bool = True,
        transform_outer: bool = True,
    ):
        super().__init__()

        if len(input_dims) != len(hidden_size):
            raise Exception("Input shape and hidden shape do not match.")

        self.input_dims      = input_dims
        self.hidden_size     = hidden_size
        self.num_layers      = len(input_dims)  # Must match since dims are equal
        self.transform_outer = transform_outer

        # Define transformation layers per dimension
        self.align_layers = torch.nn.ModuleList([
            torch.nn.Linear(input_dims[i], hidden_size[i], bias=bias) for i in range(self.num_layers)
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass to project input tensor into a new multi-space representation.
        - Incrementally transposes, flattens, applies linear layers, and restores shape.

        Expected Input Shape: [batch_size, *input_dims]
        Output Shape: [batch_size, *hidden_size]

        Args:
            x: Input tensor with shape [batch_size, *input_dims]

        Returns:
            Output tensor with shape [batch_size, *hidden_size]
        """
        num_transforms = self.num_layers  # Number of transformations
        
        # Define iteration order
        # transform_indices = range(num_transforms) if transform_outer else reversed(range(num_transforms))
        
        for i in range(num_transforms):
            if self.transform_outer:
                layer         = self.align_layers[i]
                transpose_dim = i + 1
            else:
                layer         = self.align_layers[num_transforms - (i + 1)]
                transpose_dim = num_transforms - i

            # Transpose the selected dimension to the last position
            x = torch.transpose(x, transpose_dim, num_transforms).contiguous()

            # Store the original shape before transformation
            x_size = x.shape[:-1]

            # Flatten everything except the last dimension
            x = x.view(-1, x.shape[-1])

            # Apply transformation
            x = layer(x)
            
            # Reshape back to the original spatial structure (with new embedding dim)
            x = x.view(*x_size, x.shape[-1])

            # Transpose the dimension back to its original position
            x = torch.transpose(x, transpose_dim, num_transforms).contiguous()

        return x


class NdLinearGated(torch.nn.Module):
    """NdLinearGated: A PyTorch layer for projecting tensors into multi-space
    representations with gating mechanisms.

    Extends the NdLinear concept by incorporating gating mechanisms that control
    information flow. This allows the model to selectively utilize transformations
    based on input characteristics, enabling more adaptive and context-dependent
    multi-space representations.

    Args:
        input_dims: Shape of input tensor (excluding batch dimension).
        hidden_size: Target hidden dimensions after transformation.
        transform_outer: If True, transforms from outer to inner dimensions.
        gating_mode: Type of gating mechanism - "soft" uses continuous values, "hard" uses binary.
        gating_hidden_dim: Hidden dimension size for the gating networks.
        gated_modes: Specifies which dimensions to apply gating to.
    """
    def __init__(
        self,
        input_dims       : tuple,
        hidden_size      : tuple,
        transform_outer  : bool = True,
        gating_mode      : Literal["soft", "hard"] = "soft",
        gating_hidden_dim: int  = 16,
        gated_modes      : Literal["all" , "first", "topk"] = "all"
    ):
        super().__init__()

        if len(input_dims) != len(hidden_size):
            raise Exception("Input shape and hidden shape do not match.")

        self.input_dims        = input_dims
        self.hidden_size       = hidden_size
        self.num_layers        = len(input_dims)
        self.transform_outer   = transform_outer
        self.gating_mode       = gating_mode
        self.gating_hidden_dim = gating_hidden_dim
        self.gated_modes       = gated_modes

        self.align_layers = torch.nn.ModuleList([
            torch.nn.Linear(input_dims[i], hidden_size[i]) for i in range(self.num_layers)
        ])

        self.gate_networks = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(input_dims[i], gating_hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(gating_hidden_dim, 1),
                torch.nn.Sigmoid()
            ) for i in range(self.num_layers)
        ])

        self.identity_projections = torch.nn.ModuleList([
            torch.nn.Linear(input_dims[i], hidden_size[i]) if input_dims[i] != hidden_size[i] else torch.nn.Identity()
            for i in range(self.num_layers)
        ])

        self.topk_modes            = None
        self.first_batch_processed = False

    def _compute_topk_modes(self, x: torch.Tensor) -> list:
        mode_stds = []
        for i in range(self.num_layers):
            transpose_dim = i + 1 if self.transform_outer else self.num_layers - i
            X_transposed  = torch.transpose(x, transpose_dim, self.num_layers)
            X_mean        = X_transposed.mean(dim=tuple(range(len(X_transposed.shape) - 1)))
            mode_stds.append(X_mean.std().item())

        sorted_modes = sorted(range(len(mode_stds)), key=lambda i: mode_stds[i], reverse=True)
        return sorted_modes[:2]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass to project input tensor into a new multi-space representation with gating.

        Applies dimensional transformations with selective gating based on the configured mode.
        The gating mechanism allows the network to adaptively choose between transformed
        representations and identity mappings.

        Args:
            x: Input tensor with shape [batch_size, *input_dims]

        Returns:
            torch.Tensor: Output tensor with shape [batch_size, *hidden_size]
        """
        num_transforms = self.num_layers

        if self.gated_modes == "topk" and not self.first_batch_processed:
            self.topk_modes = self._compute_topk_modes(x)
            self.first_batch_processed = True

        for i in range(num_transforms):
            if self.transform_outer:
                layer_idx     = i
                transpose_dim = i + 1
            else:
                layer_idx     = num_transforms - (i+1)
                transpose_dim = num_transforms - i

            apply_gating = False
            if self.gated_modes == "all":
                apply_gating = True
            elif self.gated_modes == "first" and i == 0:
                apply_gating = True
            elif self.gated_modes == "topk" and self.topk_modes and layer_idx in self.topk_modes:
                apply_gating = True

            x_original    = x.clone()
            x             = torch.transpose(x, transpose_dim, num_transforms).contiguous()
            X_size        = x.shape[:-1]
            x_flat        = x.view(-1, x.shape[-1])
            x_transformed = self.align_layers[layer_idx](x_flat)

            if apply_gating:
                x_mean          = x_flat.mean(dim=0, keepdim=True)
                gate            = self.gate_networks[layer_idx](x_mean)
                x_transformed   = x_transformed.view(*X_size, x_transformed.shape[-1])
                x_identity      = torch.transpose(x_original, transpose_dim, num_transforms).contiguous()
                x_identity_flat = x_identity.view(-1, x_identity.shape[-1])

                if x_transformed.shape[-1] != x_identity_flat.shape[-1]:
                    identity_flat = self.identity_projections[layer_idx](x_identity_flat)
                else:
                    identity_flat = x_identity_flat

                if self.gating_mode == "soft":
                    x_flat = gate * x_transformed.view(-1, x_transformed.shape[-1]) + (1 - gate) * identity_flat
                else:
                    x_flat = torch.where(gate > 0.5, x_transformed.view(-1, x_transformed.shape[-1]), identity_flat)

                x = x_flat.view(*X_size, x_flat.shape[-1])
            else:
                x = x_transformed.view(*X_size, x_transformed.shape[-1])

            x = torch.transpose(x, transpose_dim, num_transforms).contiguous()

        return x

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}(input_dims={self.input_dims}, "
                f"hidden_size={self.hidden_size}, transform_outer={self.transform_outer}, "
                f"gating_mode={self.gating_mode}, gating_hidden_dim={self.gating_hidden_dim}, "
                f"gated_modes={self.gated_modes})")

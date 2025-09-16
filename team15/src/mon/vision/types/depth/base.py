#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements base class and utility functions for depth estimation models."""

__all__ = [
    "DepthEstimationModel",
]

import abc

from mon import nn
from mon.constants import Task
from mon.vision import model


# ----- Base Model -----
class DepthEstimationModel(model.VisionModel, abc.ABC):
    """Base class for depth estimation models."""
    
    tasks: list[Task] = [Task.DEPTH]
    
    # ----- Forward Pass -----
    def forward_loss(self, datapoint: dict, *args, **kwargs) -> dict:
        """Computes forward pass and loss.
    
        Args:
            datapoint: ``dict`` with datapoint attributes.
    
        Returns:
            ``dict`` of predictions with ``"loss"`` and ``"depth"`` keys.
        """
        # Forward
        outputs = self.forward(datapoint=datapoint, *args, **kwargs)
        
        # Loss
        pred    = outputs["depth"]
        target  = datapoint["depth"]
        loss    = self.loss(pred, target) if self.loss else None
        
        return outputs | {
			"loss": loss,
		}
    
    def compute_metrics(self, datapoint: dict, outputs: dict, metrics: list[nn.Metric] = None) -> dict:
        """Computes metrics for given predictions.
    
        Args:
            datapoint: ``dict`` with datapoint attributes.
            outputs: ``dict`` with model predictions.
            metrics: ``list`` of ``M.Metric`` or ``None``. Default is ``None``.
    
        Returns:
            ``dict`` of computed metric values.
        """
        pred    = outputs["depth"]
        target  = datapoint["depth"]
        results = {}
        if metrics:
            for i, metric in enumerate(metrics):
                metric_name = getattr(metric, "name", f"metric_{i}")
                results[metric_name] = metric(pred, target)
        return results

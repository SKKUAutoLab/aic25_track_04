#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements training procedure for neural networks."""

__all__ = [
    "Trainer",
    "seed_everything",
]

import lightning
from lightning.pytorch.trainer import seed_everything

from mon import core
from mon.nn import strategy


# ----- Trainer -----
class Trainer(lightning.Trainer):
    """Extends lightning.Trainer with custom methods and properties.

    Args:
        log_image_every_n_epochs: Log debug images every n epochs as ``int``.
            Default is ``0``.
        *args: Variable length argument list passed to ``lightning.Trainer``.
        **kwargs: Keyword arguments passed to ``lightning.Trainer``.
    """
    
    def __init__(self, log_image_every_n_epochs: int = 0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.log_image_every_n_epochs = log_image_every_n_epochs
        
    @lightning.Trainer.current_epoch.setter
    def current_epoch(self, current_epoch: int):
        """Sets the current epoch.

        Args:
            current_epoch: Epoch number to set as ``int``.
        """
        self.fit_loop.current_epoch = current_epoch

    @lightning.Trainer.global_step.setter
    def global_step(self, global_step: int):
        """Sets the global step.

        Args:
            global_step: Step number to set as ``int``.
        """
        self.fit_loop.global_step = global_step
    
    def _log_device_info(self):
        """Logs device availability and usage information."""
        gpu_available, gpu_type = (
            (True, " (cuda)") if strategy.CUDAAccelerator.is_available() else
            (True, " (mps)") if strategy.MPSAccelerator.is_available() else
            (False, "")
        )
        gpu_used = isinstance(self.accelerator, strategy.CUDAAccelerator | strategy.MPSAccelerator)
        core.console.log(f"GPU available: {gpu_available}{gpu_type}, used: {gpu_used}.")

        if strategy.CUDAAccelerator.is_available() and not isinstance(self.accelerator, strategy.CUDAAccelerator):
            core.console.log(
                f"GPU available but not used. Set `accelerator` and `devices` using "
                f"Trainer(accelerator='gpu', devices={strategy.CUDAAccelerator.auto_device_count()})."
            )
        if strategy.MPSAccelerator.is_available() and not isinstance(self.accelerator, strategy.MPSAccelerator):
            core.console.log(
                f"MPS available but not used. Set `accelerator` and `devices` using "
                f"Trainer(accelerator='mps', devices={strategy.MPSAccelerator.auto_device_count()})."
            )

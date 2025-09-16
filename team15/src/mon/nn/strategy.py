#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements training strategies with accelerators and plugins.

References:
    - https://pytorch-lightning.readthedocs.io/en/stable/extensions/strategy.html
"""

__all__ = [
    "Accelerator",
    "CPUAccelerator",
    "CUDAAccelerator",
    "DDPStrategy",
    "DeepSpeedStrategy",
    "FSDPStrategy",
    "MPSAccelerator",
    "ParallelStrategy",
    "SingleDeviceStrategy",
    "Strategy",
    "XLAAccelerator",
]

import os
import platform
from typing import Callable

import torch
import torch.cuda
from lightning.pytorch.accelerators import (
    Accelerator, CPUAccelerator, CUDAAccelerator, MPSAccelerator, XLAAccelerator,
)
from lightning.pytorch.strategies import (
    DDPStrategy, DeepSpeedStrategy, FSDPStrategy, ParallelStrategy,
    SingleDeviceStrategy, Strategy,
)
from torch import distributed

from mon import core
from mon.constants import ACCELERATORS, STRATEGIES

# ----- Accelerator -----
ACCELERATORS.register(name="cpu",  module=CPUAccelerator)
ACCELERATORS.register(name="cuda", module=CUDAAccelerator)
ACCELERATORS.register(name="gpu",  module=CUDAAccelerator)
ACCELERATORS.register(name="mps",  module=MPSAccelerator)
ACCELERATORS.register(name="xla",  module=XLAAccelerator)


# ----- Strategy -----
STRATEGIES.register(name="ddp",           module=DDPStrategy)
STRATEGIES.register(name="deepspeed",     module=DeepSpeedStrategy)
STRATEGIES.register(name="fsdp",          module=FSDPStrategy)
STRATEGIES.register(name="parallel",      module=ParallelStrategy)
STRATEGIES.register(name="single_device", module=SingleDeviceStrategy)


# ----- Utils -----
def get_distributed_info() -> list[int]:
    """Returns rank and world size if distributed, else [0, 1].

    Returns:
        List of [rank, world_size] for the current process.
    """
    if distributed.is_available() and distributed.is_initialized():
        return [distributed.get_rank(), distributed.get_world_size()]
    return [0, 1]


def set_distributed_backend(strategy: str | Callable, cudnn: bool = True):
    """Sets distributed backend based on OS and strategy.

    Args:
        strategy: Distributed strategy (``"ddp"``, ``"ddp2"``) or callable.
        cudnn: Enable cuDNN if ``True``. Default is ``True``.
    """
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.enabled = cudnn
        core.console.log(f"cuDNN available: [bright_green]True[/bright_green], "
                         f"used: [bright_green]{cudnn}[/bright_green].")
    else:
        core.console.log(f"cuDNN available: [red]False[/red].")

    if strategy in ["ddp"] or isinstance(strategy, DDPStrategy):
        backend = "gloo" if platform.system() == "Windows" else "nccl"
        os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = backend
        core.console.log(f"Running on a {platform.system()} machine, set torch "
                         f"distributed backend to {backend}.")

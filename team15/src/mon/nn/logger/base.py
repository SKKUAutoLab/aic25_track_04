#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements base class and helpers for all loggers."""

__all__ = [
    "CSVLogger",
    "CometLogger",
    "Logger",
    "MLFlowLogger",
    "NeptuneLogger",
    "WandbLogger",
]

from lightning.pytorch.loggers import (
    CometLogger, CSVLogger, Logger, MLFlowLogger, NeptuneLogger, WandbLogger,
)

from mon.constants import LOGGERS

# ----- Registering -----
LOGGERS.register(name="csv",            module=CSVLogger)
LOGGERS.register(name="csv_logger",     module=CSVLogger)
LOGGERS.register(name="comet",          module=CometLogger)
LOGGERS.register(name="comet_logger",   module=CometLogger)
LOGGERS.register(name="mlflow",         module=MLFlowLogger)
LOGGERS.register(name="mlflow_logger",  module=MLFlowLogger)
LOGGERS.register(name="neptune",        module=NeptuneLogger)
LOGGERS.register(name="neptune_logger", module=NeptuneLogger)
LOGGERS.register(name="wandb",          module=WandbLogger)
LOGGERS.register(name="wandb_logger",   module=WandbLogger)

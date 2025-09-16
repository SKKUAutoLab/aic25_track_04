#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements base classes and helpers for all callbacks."""

__all__ = [
    "BackboneFinetuning",
    "BaseFinetuning",
    "BasePredictionWriter",
    "BatchSizeFinder",
    "Callback",
    "Checkpoint",
    "DeviceStatsMonitor",
    "EarlyStopping",
    "GradientAccumulationScheduler",
    "LambdaCallback",
    "LearningRateFinder",
    "LearningRateMonitor",
    "ModelPruning",
    "ModelSummary",
    "OnExceptionCheckpoint",
    "SpikeDetection",
    "StochasticWeightAveraging",
    "TQDMProgressBar",
    "TimerCallback",
]

from lightning.pytorch.callbacks import (
    BackboneFinetuning, BaseFinetuning, BasePredictionWriter, BatchSizeFinder, Callback,
    Checkpoint, DeviceStatsMonitor, EarlyStopping, GradientAccumulationScheduler,
    LambdaCallback, LearningRateFinder, LearningRateMonitor, ModelPruning, ModelSummary,
    OnExceptionCheckpoint, SpikeDetection, StochasticWeightAveraging, Timer,
    TQDMProgressBar,
)

from mon.constants import CALLBACKS

# ----- Registering -----
TimerCallback = Timer

CALLBACKS.register(name="backbone_finetuning",             module=BackboneFinetuning)
CALLBACKS.register(name="batch_size_finder",               module=BatchSizeFinder)
CALLBACKS.register(name="device_stats_monitor",            module=DeviceStatsMonitor)
CALLBACKS.register(name="early_stopping",                  module=EarlyStopping)
CALLBACKS.register(name="gradient_accumulation_scheduler", module=GradientAccumulationScheduler)
CALLBACKS.register(name="learning_rate_finder",            module=LearningRateFinder)
CALLBACKS.register(name="learning_rate_monitor",           module=LearningRateMonitor)
CALLBACKS.register(name="model_pruning",                   module=ModelPruning)
CALLBACKS.register(name="model_summary",                   module=ModelSummary)
CALLBACKS.register(name="on_exception_checkpoint",         module=OnExceptionCheckpoint)
CALLBACKS.register(name="spike_detection",                 module=SpikeDetection)
CALLBACKS.register(name="stochastic_weight_averaging",     module=StochasticWeightAveraging)
CALLBACKS.register(name="timer",                           module=TimerCallback)
CALLBACKS.register(name="tqdm_progress_bar",               module=TQDMProgressBar)

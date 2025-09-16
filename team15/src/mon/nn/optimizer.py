#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements optimizers with ``torch``."""

__all__ = [
    "ASGD",
    "Adadelta",
    "Adagrad",
    "Adam",
    "AdamW",
    "Adamax",
    "LBFGS",
    "NAdam",
    "Optimizer",
    "RAdam",
    "RMSprop",
    "Rprop",
    "SGD",
    "SparseAdam",
]

from torch.optim import (
    Adadelta, Adagrad, Adam, Adamax, AdamW, ASGD, LBFGS, NAdam, Optimizer, RAdam,
    RMSprop, Rprop, SGD, SparseAdam,
)

from mon.constants import OPTIMIZERS

# ----- Registering Optimizer -----
OPTIMIZERS.register(name="adadelta",    module=Adadelta)
OPTIMIZERS.register(name="adagrad",     module=Adagrad)
OPTIMIZERS.register(name="adam",        module=Adam)
OPTIMIZERS.register(name="adamax",      module=Adamax)
OPTIMIZERS.register(name="adamw",       module=AdamW)
OPTIMIZERS.register(name="asgd",        module=ASGD)
OPTIMIZERS.register(name="lbfgs",       module=LBFGS)
OPTIMIZERS.register(name="nadam",       module=NAdam)
OPTIMIZERS.register(name="radam",       module=RAdam)
OPTIMIZERS.register(name="rmsprop",     module=RMSprop)
OPTIMIZERS.register(name="rprop",       module=Rprop)
OPTIMIZERS.register(name="sgd",         module=SGD)
OPTIMIZERS.register(name="sparse_adam", module=SparseAdam)

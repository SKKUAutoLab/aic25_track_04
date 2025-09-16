#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module defines all common configuration settings used in this project."""

__all__ = [
    "DATASETS",
    "MODELS",
    "TASKS",
]

from mon import Task

# List all tasks that are performed in this project.
TASKS = [
    Task.DETECT,
]

# List all models that are used in this project.
MODELS = [
    "deim",
]
# If unsure, run the following script:
# mon.print_table(mon.MODELS | mon.EXTRA_MODELS)

# List all datasets that are used in this project.
DATASETS = [

]
# If unsure, run the following script:
# mon.print_table(mon.DATASETS | mon.DATASETS_EXTRA)

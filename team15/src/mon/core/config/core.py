#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements base options for both runtime and interactive CLI."""

__all__ = [
    "CLI_OPTIONS",
    "DEFAULT_ARGS",
]

from typing import Any

import box

from mon.constants import Task, TRTPrecision
from mon.core import device


# ----- Utils -----
def _str_or_none(value: Any) -> str | None:
    """Converts a value to a string or ``None`` if value is ``"None"``.

    Args:
        value: Value to convert.

    Returns:
        String of ``value`` or ``None`` if ``value`` is ``"None"``.
    """
    return None if value in [None, "None", ""] else str(value)


def _int_or_none(value: Any) -> int | None:
    """Converts a value to an integer or ``None`` if value is ``"None"``.

    Args:
        value: Value to convert.

    Returns:
        Integer of ``value`` or ``None`` if ``value`` is ``"None"``.
    """
    return None if value in [None, "None", ""] else int(value)


def _float_or_none(value: Any) -> float | None:
    """Converts a value to a float or ``None`` if value is ``"None"``.

    Args:
        value: Value to convert.

    Returns:
        Float of ``value`` or ``None`` if ``value`` is ``"None"``.
    """
    return None if value in [None, "None", ""] else float(value)


# ----- Default CLI Options -----
"""
parser.add_argument(
    name_or_flags,
    action   = None,
    nargs    = None,
    const    = None,
    default  = None,
    type     = None,
    choices  = None,
    required = False,
    help     = None,
    metavar  = None,
    dest     = None,
    version  = None,
    **kwargs
)
"""
CLI_OPTIONS  = {
    # Basic
    "root"         : {
        "default"    : None,
        "type"       : _str_or_none,
        "help"       : "Project root.",
        "prompt_only": False,
        "prompt_text": "Project Root",
    },
    "task"         : {
        "default"    : None,
        "type"       : _str_or_none,
        "choices"    : Task.values(),
        "help"       : f"Task to run: {Task.values()}.",
        "prompt_only": False,
        "prompt_text": "Task",
    },
    "mode"         : {
        "default"    : None,
        "type"       : _str_or_none,
        "choices"    : ["train", "predict", "speed"],  # RunMode.values(),
        "help"       : f"Run mode: {['train', 'predict']}.",
        "prompt_only": False,
        "i_cli_type" : str,
        "prompt_text": "Run Mode",
    },
    "arch"         : {
        "default"    : None,
        "type"       : _str_or_none,
        "help"       : "Model architecture.",
        "prompt_only": False,
        "prompt_text": "Architecture",
    },
    "model"        : {
        "default"    : None,
        "type"       : _str_or_none,
        "help"       : "Model name.",
        "prompt_only": False,
        "prompt_text": "Model",
    },
    "config"       : {
        "default"    : None,
        "type"       : _str_or_none,
        "help"       : "Config file.",
        "prompt_only": False,
        "prompt_text": "Config",
    },
    "data"         : {
        "default"    : None,
        "type"       : _str_or_none,
        "help"       : "Dataset name or directory.",
        "prompt_only": False,
        "prompt_text": "Predict(s)",
    },
    "fullname"     : {
        "default"    : None,
        "type"       : _str_or_none,
        "help"       : "Full name of the current run.",
        "prompt_only": False,  
        "prompt_text": "Fullname",
    },
    "save_dir"     : {
        "type"       : _str_or_none,
        "default"    : None,
        "help"       : "Directory to save the outputs.",
        "prompt_only": False,
        "prompt_text": "Save Directory",
    },
    "weights"      : {
        "action"     : "append",
        "default"    : None,
        "type"       : _str_or_none,
        "help"       : "Path(s) to the pretrained weights.",
        "prompt_only": False,
        "prompt_text": "Weights",
    },
    "device"       : {
        "default"    : None,
        "type"       : _str_or_none,
        "choices"    : device.list_devices(),
        "help"       : f"Running device: {device.list_devices()}.",
        "prompt_only": False,
        "prompt_text": "Device",
    },
    "seed"         : {
        "default"    : None,
        "type"       : _int_or_none,
        "help"       : "Seed.",
        "prompt_only": False,
        "prompt_text": "Seed         ",
    },
    "imgsz"        : {
        "action"     : "append",
        "default"    : None,
        "type"       : _int_or_none,
        "help"       : "Image size.",
        "prompt_only": False,
        "prompt_text": "Image Size   ",
    },
    # Train
    "epochs"       : {
        "default"    : None,
        "type"       : _int_or_none,
        "help"       : "Training epochs.",
        "prompt_only": False,
        "prompt_text": "Epochs       ",
    },
    "batch_size"   : {
        "default"    : None,
        "type"       : _int_or_none,
        "help"       : "Batch size.",
        "prompt_only": False,
        "prompt_text": "Batch Size   ",
    },
    "torchrun"     : {
        "default"    : False,
        "action"     : "store_true",
        "help"       : "Using torch distributed training.",
        "prompt_only": False,
        "prompt_text": "Use torchrun?",
    },
    "master_port"  : {
        "default"    : 7777,
        "type"       : _int_or_none,
        "help"       : "Port for distributed communication.",
        "prompt_only": False,
        "prompt_text": "Master Port",
    },
    "master_addr"  : {
        "default"    : "localhost",
        "type"       : _str_or_none,
        "help"       : "Master node address.",
        "prompt_only": False,
        "prompt_text": "Master Address",
    },
    "local_rank"   : {
        "type"       : _int_or_none,
        "help"       : "Local rank for distributed training.",
        "prompt_only": False,
        "prompt_text": "Local Rank   ",
    },
    # Predict
    "resize"       : {
        "action"     : "store_true",
        "help"       : "Resize the input image.",
        "prompt_only": False,
        "prompt_text": "Resize?      ",
    },
    "benchmark"    : {
        "action"     : "store_true",
        "help"       : "Enable benchmark mode.",
        "prompt_only": False,
        "prompt_text": "Benchmark?   ",
    },
    # Save & Visualize
    "save_result"  : {
        "action"     : "store_true",
        "help"       : "Save results.",
        "prompt_only": False,
        "prompt_text": "Save Result? ",
    },
    "save_image"   : {
        "action"     : "store_true",
        "help"       : "Save output images.",
        "prompt_only": False,
        "prompt_text": "Save Image?  ",
    },
    "save_debug"   : {
        "action"     : "store_true",
        "help"       : "Save debug information.",
        "prompt_only": False,
        "prompt_text": "Save Debug?  ",
    },
    "use_fullname" : {
        "action"     : "store_true",
        "help"       : "Use the ``fullname`` for the ``save_dir``.",
        "prompt_only": False,
        "prompt_text": "Use Fullname?",
    },
    "keep_subdirs" : {
        "action"     : "store_true",
        "help"       : "Keep subdirectories in the ``save_dir``.",
        "prompt_only": False,
        "prompt_text": "Keep Subdirs?",
    },
    "save_nearby"  : {
        "action"     : "store_true",
        "help"       : "Save outputs nearby the source.",
        "prompt_only": False,
        "prompt_text": "Save Nearby? ",
    },
    "exist_ok"     : {
        "action"     : "store_true",
        "help"       : "Keep existing directories.",
        "prompt_only": False,
        "prompt_text": "Exist OK?    ",
    },
    "verbose"      : {
        "action"     : "store_true",
        "help"       : "Verbose mode.",
        "prompt_only": False,
        "prompt_text": "Verbosity?   ",
    },
    # Export
    "trt_precision": {
        "default"    : "fp32",
        "type"       : _str_or_none,
        "choices"    : TRTPrecision.values(),
        "help"       : f"TRT precision: {TRTPrecision.values()}.",
        "prompt_only": False,
        "prompt_text": "TRT Precision",
    },
}
CLI_OPTIONS  = box.Box(CLI_OPTIONS)

DEFAULT_ARGS = {
    k: False if v.get("action") in ["store_true"] else v.get("default", None)
    for k, v in CLI_OPTIONS.items()
}
DEFAULT_ARGS = box.Box(DEFAULT_ARGS)

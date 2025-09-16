#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Extends Python's ``logging`` module."""

__all__ = [
    "disable_default_loggers",
    "disable_print",
    "disable_stdout",
    "enable_default_loggers",
    "enable_print",
    "enable_stdout",
    "get_logger",
    "logger",
]
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import contextlib
import logging
import sys
from typing import Iterator

from rich import logging as r_logging

from mon.core import pathlib


# ----- Log -----
logging.basicConfig(
    level    = logging.INFO,
    format   = "%(message)s",
    handlers = [r_logging.RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger("rich")
# logger.setLevel(logging.INFO)


def get_logger(path: pathlib.Path = None) -> logging.Logger:
    """Retrieves or creates a global logger with ``rich`` support.

    Args:
        path: Path for log file, adds file handler if given. Default is ``None``.

    Returns:
        Global logger instance.
    """
    logger = logging.getLogger("global_logger")
    if path:
        file_handler = logging.FileHandler(path)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s [%(filename)s:%(lineno)s] %(levelname)s: %(message)s")
        )
        logger.addHandler(file_handler)
    return logger


# ----- Print -----
def disable_default_loggers():
    """Disables all logging by setting the logger to the lowest level.

    Notes:
        Use this to suppress all logging output.
    """
    # Disabling Python’s Built-in logging
    logging.getLogger().setLevel(logging.CRITICAL + 1)  # Suppresses everything
    # Disabling PyTorch Logs
    logging.getLogger("torch").setLevel(logging.CRITICAL + 1)  # Silence PyTorch logs
    # Disabling TensorFlow Logs
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def enable_default_loggers():
    """Enables all logging by resetting the logger to its default level.

    Notes:
        Use this to restore all logging output.
    """
    # Enabling Python’s Built-in logging
    logging.getLogger().setLevel(logging.INFO)  # Restores default level
    # Enabling PyTorch Logs
    logging.getLogger("torch").setLevel(logging.INFO)  # Restores default level
    # Enabling TensorFlow Logs
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"  # Restores default level


@contextlib.contextmanager
def disable_stdout() -> Iterator[None]:
    """Disables printing to stdout by redirecting it to ``os.devnull``.

    Notes:
        Use this to suppress all print output.
    """
    with open(os.devnull, "w") as devnull:
        with contextlib.redirect_stdout(devnull):
            yield


def enable_stdout():
    """Restores printing to stdout by resetting it to the original stream.

    Notes:
        Use this to undo manual redirection of ``sys.stdout`` (e.g., to ``os.devnull``).
    """
    sys.stdout = sys.__stdout__


def disable_print():
    """Temporarily disables printing to stdout and logging."""
    disable_stdout()
    disable_default_loggers()


def enable_print():
    """Restores printing to stdout and loggers."""
    enable_stdout()
    enable_default_loggers()


# Disable default loggers
disable_default_loggers()

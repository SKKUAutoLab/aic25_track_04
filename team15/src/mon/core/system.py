#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Handles system-level operations."""

__all__ = [
    "check_installed_package",
    "clear_terminal",
    "get_terminal_size",
    "set_random_seed",
    "set_terminal_size",
]

import fcntl
import importlib
import importlib.util
import os
import platform
import random
import shutil
import struct
import subprocess
import sys
import termios

import numpy as np
import torch


# ----- Package -----
def check_installed_package(package_name: str, verbose: bool = False) -> bool:
    """Checks if a package is installed.

    Args:
        package_name: Name of the package to check.
        verbose: If ``True``, prints install status. Default is ``False``.

    Returns:
        ``True`` if package is installed, ``False`` otherwise.
    """
    try:
        importlib.import_module(package_name)
        if verbose:
            print(f"[{package_name}] is installed")
        return True
    except ImportError:
        if verbose:
            print(f"[{package_name}] is not installed")
        return False


# ----- Seed -----
def set_random_seed(seed: int | list[int] | tuple[int, int]) -> None:
    """Sets random seeds for various libraries.

    Args:
        seed: Int, list of ints, or tuple of two ints for range selection.
    """
    if isinstance(seed, list | tuple):
        seed = random.randint(seed[0], seed[1]) if len(seed) == 2 else seed[-1]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# ----- Terminal -----
def clear_terminal():
    """Clears the terminal screen."""
    if platform.system() == "Windows":
        os.system("cls")
    elif platform.system() in ["Darwin", "Linux"]:
        os.system("clear")


def get_terminal_size() -> tuple[int, int]:
    """Gets the size of the terminal window in columns and rows.

    Returns:
        Tuple of ``(columns, rows)`` as integers.
    """
    size = shutil.get_terminal_size(fallback=(100, 40))
    return size.columns, size.lines


def set_terminal_size(rows: int = 40, cols: int = 100):
    """Sets the terminal window size to specified rows and columns.

    Args:
        rows: Number of rows for terminal. Default is ``40``.
        cols: Number of columns for terminal. Default is ``100``.
    """
    fd   = sys.stdout.fileno()
    size = struct.pack("HHHH", rows, cols, 0, 0)
    fcntl.ioctl(fd, termios.TIOCSWINSZ, size)
    subprocess.run(["stty", "rows", str(rows), "cols", str(cols)])

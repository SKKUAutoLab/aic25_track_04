#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Extends ``rich.progress``."""

import torch
from rich import text
from rich.progress import *

from mon.constants import MemoryUnit
from mon.core import device, type_extensions
from mon.core.rich.core import console


# ----- Progress -----
def create_download_bar(transient: bool = False, disable: bool = False) -> Progress:
    """Creates a ``rich.progress.Progress`` for download tracking.

    Args:
        transient: If ``True``, hides bar after completion. Default is ``False``.
        disable: If ``True``, disables progress bar. Default is ``False``.

    Returns:
        ``rich.progress.Progress`` with download-specific columns.
    """
    return Progress(
        TextColumn(
            # console.get_datetime().strftime("[%x %H:%M:%S]"),
            console.get_datetime().strftime("[%X]"),
            justify="left",
            style="log.time",
        ),
        TextColumn("{task.description}", justify="right"),
        BarColumn(bar_width=None),
        "[progress.percentage]{task.percentage:>3.1f}%",
        "•",
        TransferSpeedColumn(),
        "•",
        DownloadColumn(),
        "•",
        TimeRemainingColumn(),
        ">",
        TimeElapsedColumn(),
        console   = console,
        transient = transient,
        disable   = disable,
    )


def create_progress_bar(transient: bool = False, disable: bool = False) -> Progress:
    """Creates a ``rich.progress.Progress`` for general progress tracking.

    Args:
        transient: If ``True``, hides bar after completion. Default is ``False``.
        disable: If ``True``, disables progress bar. Default is ``False``.

    Returns:
        ``rich.progress.Progress`` with processing-specific columns.
    """
    return Progress(
        TextColumn(
            # console.get_datetime().strftime("[%x %H:%M:%S]"),
            console.get_datetime().strftime("[%X]"),
            justify="left",
            style="log.time"
        ),
        TextColumn("{task.description}", justify="right"),
        BarColumn(bar_width=None, finished_style="green"),
        "[progress.percentage]{task.percentage:>3.1f}%",
        "•",
        ProcessedItemsColumn(),
        "•",
        ProcessingSpeedColumn(),
        "•",
        TimeRemainingColumn(),
        ">",
        TimeElapsedColumn(),
        SpinnerColumn(),
        console   = console,
        transient = transient,
        disable   = disable,
    )


class MemoryUsageColumn(ProgressColumn):
    """Displays CPU/GPU memory usage in a progress bar (e.g., ``33.1/48.0GB``).

    Args:
        devices: GPU device index or list of indices. Default is ``0``.
        unit: Memory unit (e.g., ``'GB'``). Default is ``MemoryUnit.GB``.
        table_column: Column in table to associate with. Default is ``None``.
    """
    
    def __init__(
        self,
        devices     : int | list[int] = 0,
        unit        : MemoryUnit = MemoryUnit.GB,
        table_column: Column     = None
    ):
        super().__init__(table_column=table_column)
        self.devices = type_extensions.to_int_list(devices)
        self.unit    = MemoryUnit.from_value(value=unit)
    
    def render(self, task: Task) -> text.Text:
        """Renders current GPU or CPU memory usage as text.

        Args:
            task: ``rich.progress.Task`` object for the progress task.

        Returns:
            ``rich.text.Text`` with memory usage status.
        """
        return self.gpu_memory_text \
            if torch.cuda.is_available() \
            else self.machine_memory_text
    
    @property
    def machine_memory_text(self) -> text.Text:
        """Renders current RAM usage as text.

        Returns:
            ``rich.text.Text`` with RAM usage status.
        """
        total, used, _ = device.get_memory_usages(unit=self.unit)
        memory_status  = f"{used:.1f}/{total:.1f}{self.unit.value} (CPU)"
        memory_text    = text.Text(memory_status, style="bright_yellow")
        return memory_text
    
    @property
    def gpu_memory_text(self) -> text.Text:
        """Renders current GPU memory usage as text.

        Returns:
            ``rich.text.Text`` with GPU memory usage status.
        """
        num_devices = len(self.devices)
        totals, useds = [], []
        for i in self.devices:
            total, used, _ = device.get_cuda_memory_usages(device=i, unit=self.unit)
            totals.append(total)
            useds.append(used)
        total = min(totals)
        used  = max(useds)
        memory_status = f"{used:.1f}/{total:.1f}{self.unit.value} ({num_devices} GPUs)"
        memory_text   = text.Text(memory_status, style="bright_yellow")
        return memory_text


class ProcessedItemsColumn(ProgressColumn):
    """Shows number of processed items in a progress bar (e.g., ``1728/2025``).

    Args:
        table_column: Column in table to associate with. Default is ``None``.
    """
    
    def __init__(self, table_column: Column = None):
        super().__init__(table_column=table_column)
    
    def render(self, task: Task) -> text.Text:
        """Renders the number of processed items as text.

        Args:
            task: ``rich.progress.Task`` object for the progress task.

        Returns:
            ``rich.text.Text`` with processed items count.
        """
        completed = int(task.completed)
        total     = int(task.total)
        count     = f"{completed}/{total}"
        count     = f"{count:>14}"
        return text.Text(count, style="progress.download")


class ProcessingSpeedColumn(ProgressColumn):
    """Shows human-readable processing speed in a progress bar."""
    
    def render(self, task: Task) -> text.Text:
        """Renders the processing speed as text.

        Args:
            task: ``rich.progress.Task`` object for the progress task.

        Returns:
            ``rich.text.Text`` with the processing speed.
        """
        speed = task.speed
        if speed is None:
            return text.Text("?", style="progress.data.speed")
        speed_text = f"{speed:0.2f}"
        speed_text = f"{speed_text:>7}"
        return text.Text(f"{speed_text}it/s", style="progress.data.speed")

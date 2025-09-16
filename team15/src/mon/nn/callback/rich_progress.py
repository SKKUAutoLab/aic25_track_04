#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements rich progress bar callback."""

__all__ = [
   "RichProgressBar",
]

import lightning
import torch
from lightning.pytorch.callbacks.progress import rich_progress

from mon import core
from mon.constants import CALLBACKS


# ----- Progress Bar -----
@CALLBACKS.register(name="rich_progress_bar")
class RichProgressBar(rich_progress.RichProgressBar):
    """Displays a progress bar with rich text formatting."""

    def _init_progress(self, trainer: lightning.Trainer):
        if self.is_enabled and (self.progress is None or self._progress_stopped):
            self._reset_progress_bar_ids()
            self._console = core.console
            self._console.clear_live()
            self._metric_component = rich_progress.MetricsTextColumn(
                trainer        = trainer,
                style          = self.theme.metrics,
                text_delimiter = self.theme.metrics_text_delimiter,
                metrics_format = self.theme.metrics_format,
            )
            self.progress = rich_progress.CustomProgress(
                *self.configure_columns(trainer),
                self._metric_component,
                auto_refresh = False,
                disable      = self.is_disabled,
                console      = self._console,
            )
            self.progress.start()
            self._progress_stopped = False

    def configure_columns(self, trainer: lightning.Trainer) -> list:
        """Configures columns for the progress bar.
    
        Args:
            trainer: ``lightning.Trainer`` instance.
    
        Returns:
            ``list`` of column configurations.
        """
        base_columns = [
            core.rich.progress.TextColumn(
                core.rich.console.get_datetime().strftime("[%m/%d/%Y %H:%M:%S]"),
                justify = "left",
                style   = "log.time"
            ),
            core.rich.progress.TextColumn("[progress.description][{task.description}]"),
            rich_progress.CustomBarColumn(
                complete_style = self.theme.progress_bar,
                finished_style = self.theme.progress_bar_finished,
                pulse_style    = self.theme.progress_bar_pulse,
            ),
            rich_progress.BatchesProcessedColumn(style="progress.download"),
            "•",
            rich_progress.ProcessingSpeedColumn(style="progress.data.speed"),
            "•",
            core.rich.progress.TimeRemainingColumn(),
            ">",
            core.rich.progress.TimeElapsedColumn(),
            core.rich.progress.SpinnerColumn(),
        ]
        if torch.cuda.is_available():
            return base_columns[:5] + [core.rich.MemoryUsageColumn(devices=trainer.device_ids)] + base_columns[5:]
        return base_columns

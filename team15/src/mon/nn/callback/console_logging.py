#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements callbacks for console logging of training/testing progress."""

__all__ = [
    "LogTrainingProgress"
]

import collections
import math
import time
from copy import deepcopy
from datetime import timedelta
from timeit import default_timer as timer
from typing import Any

import lightning.pytorch as pl
import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT

from mon import core
from mon.constants import CALLBACKS
from mon.nn.callback import base


# ----- Log -----
# noinspection PyMethodMayBeStatic
@CALLBACKS.register(name="log_training_progress")
class LogTrainingProgress(base.Callback):
    """Logs training and testing progress to console.

    Args:
        dirpath: Dir path for log file.
        filename: Log file name. Default is ``log.csv``.
        every_n_epochs: Log every n epochs. Default is ``1``.
        every_n_train_steps: Log every n training steps.
        train_time_interval: Log every n seconds.
        log_on_train_epoch_end: Log at train epoch end if ``True``.
        verbose: Enable verbose output if ``True``. Default is ``True``.
    """
    
    def __init__(
        self,
        dirpath               : core.Path,
        filename              : str       = "log.csv",
        every_n_epochs        : int       = 1,
        every_n_train_steps   : int       = None,
        train_time_interval   : timedelta = None,
        log_on_train_epoch_end: bool      = None,
        verbose               : bool      = True
    ):
        super().__init__()
        self._dirpath     = core.Path(dirpath) if dirpath else None
        self._filename    = core.Path(filename).stem
        self._candidates  = collections.OrderedDict()
        self._start_epoch = 0
        self._start_time  = 0
        self._logger      = None
        self._verbose     = verbose
        
        self._train_time_interval    = None
        self._every_n_epochs         = None
        self._every_n_train_steps    = None
        self._log_on_train_epoch_end = log_on_train_epoch_end
        self._last_global_step_saved = 0
        self._last_time_checked      = None
        self._init_triggers(every_n_epochs, every_n_train_steps, train_time_interval)
    
    def setup(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: str):
        """Sets up logging dir at start of training stages.
    
        Args:
            trainer: ``pl.Trainer`` instance.
            pl_module: ``pl.LightningModule`` instance.
            stage: Current stage (e.g., ``"fit"``, ``"validate"``, ``"test"``).
        """
        dirpath = self._dirpath or core.Path(trainer.default_root_dir)
        dirpath = trainer.strategy.broadcast(dirpath)
        self._dirpath = core.Path(dirpath)
    
    def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        """Initializes logging at training start.
    
        Args:
            trainer: ``pl.Trainer`` instance.
            pl_module: ``pl.LightningModule`` instance.
        """
        self._candidates  = self._init_candidates(trainer, pl_module)
        self._start_epoch = int(trainer.current_epoch)
        self._start_time  = timer()

        log_file       = self._dirpath / f"{core.Path(self._filename).stem}.csv"
        log_file_exist = log_file.exists()
        self._dirpath.mkdir(parents=True, exist_ok=True)
        self._logger   = open(str(log_file), "a")

        if not log_file_exist:
            self._logger.write(f"Model,{pl_module.name}\n")
            self._logger.write(f"Fullname,{pl_module.fullname}\n")
            if hasattr(pl_module, "params"):
                self._logger.write(f"Parameters,{pl_module.params}\n")
            headers = ",".join(self._candidates.keys())
            self._logger.write(f"\n{headers}\n")
            self._logger.flush()
    
    def on_train_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        """Logs duration and closes log file at training end.
    
        Args:
            trainer: ``pl.Trainer`` instance.
            pl_module: ``pl.LightningModule`` instance.
        """
        end_time      = timer()
        elapsed_epoch = int(trainer.current_epoch) - self._start_epoch
        elapsed_time  = end_time - self._start_time
        elapsed_hours = elapsed_time / 3600

        self._logger.write(
            f"\nEpochs,{elapsed_epoch},"
            f"Seconds,{elapsed_time:.3f},"
            f"Hours,{elapsed_hours:.3f},\n"
        )
        self._logger.flush()
        self._logger.close()

        if self._verbose and trainer.is_global_zero:
            core.console.log(
                f"\n{elapsed_epoch} epochs completed "
                f"in {elapsed_time:.3f} seconds "
                f"({elapsed_hours:.3f} hours).\n"
            )
    
    def on_train_batch_end(
        self,
        trainer  : "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs  : STEP_OUTPUT,
        batch    : Any,
        batch_idx: int
    ):
        """Logs training progress at batch end if conditions met.
    
        Args:
            trainer: ``pl.Trainer`` instance.
            pl_module: ``pl.LightningModule`` instance.
            outputs: Step output from training as ``STEP_OUTPUT``.
            batch: Current batch data as ``Any``.
            batch_idx: Index of current batch.
        """
        if self._should_skip_logging(trainer):
            return

        skip_batch = (
            self._every_n_train_steps is None
            or self._every_n_train_steps < 1
            or (trainer.global_step % self._every_n_train_steps != 0)
        )
        train_time_interval = self._train_time_interval
        skip_time = True
        now       = time.monotonic()
        if train_time_interval:
            prev_time_check = self._last_time_checked
            skip_time = (
                prev_time_check is None
                or (now - prev_time_check) < train_time_interval.total_seconds()
            )
            skip_time = trainer.strategy.broadcast(skip_time)

        if skip_batch and skip_time:
            return
        if not skip_time:
            self._last_time_checked = now

        if trainer.is_global_zero:
            monitor_candidates = self._get_monitor_candidates(trainer)
            candidates         = self._update_candidates(monitor_candidates)
            self._log(candidates)
    
    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        """Logs training progress at epoch end if conditions met.
    
        Args:
            trainer: ``pl.Trainer`` instance.
            pl_module: ``pl.LightningModule`` instance.
        """
        if (not self._should_skip_logging(trainer)
            and self._should_log_on_train_epoch_end(trainer)):
            if (self._every_n_epochs >= 1
                and (trainer.current_epoch + 1) % self._every_n_epochs == 0):
                if trainer.is_global_zero:
                    monitor_candidates = self._get_monitor_candidates(trainer)
                    candidates         = self._update_candidates(monitor_candidates)
                    self._log(candidates)
    
    def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        """Logs validation progress at loop end if conditions met.
    
        Args:
            trainer: ``pl.Trainer`` instance.
            pl_module: ``pl.LightningModule`` instance.
        """
        if (not self._should_skip_logging(trainer)
            and not self._should_log_on_train_epoch_end(trainer)):
            if (self._every_n_epochs >= 1
                and (trainer.current_epoch + 1) % self._every_n_epochs == 0):
                if trainer.is_global_zero:
                    monitor_candidates = self._get_monitor_candidates(trainer)
                    candidates         = self._update_candidates(monitor_candidates)
                    self._log(candidates)
    
    def _init_triggers(
        self,
        every_n_epochs     : int       = None,
        every_n_train_steps: int       = None,
        train_time_interval: timedelta = None
    ):
        """Sets up logging triggers with defaults if unspecified.

        Args:
            every_n_epochs: Log every n epochs. Default is ``None``.
            every_n_train_steps: Log every n steps. Default is ``None``.
            train_time_interval: Log every n seconds. Default is ``None``.
        """
        if (every_n_train_steps is None
            and every_n_epochs is None
            and train_time_interval is None):
            every_n_epochs      = 1
            every_n_train_steps = 0
            core.console.log("Both every_n_train_steps and every_n_epochs are not set. "
                             "Setting every_n_epochs=1.")
        else:
            every_n_epochs      = every_n_epochs or 0
            every_n_train_steps = every_n_train_steps or 0

        self._train_time_interval = train_time_interval
        self._every_n_epochs      = every_n_epochs
        self._every_n_train_steps = every_n_train_steps
        
    def _init_candidates(
        self,
        trainer  : "pl.Trainer",
        pl_module: "pl.LightningModule"
    ) -> collections.OrderedDict:
        """Initializes logging candidates with metric names.

        Args:
            trainer: ``pl.Trainer`` instance.
            pl_module: ``pl.LightningModule`` instance.
    
        Returns:
            ``OrderedDict`` of candidate metric names.
        """
        candidates = collections.OrderedDict()
        candidates |= {"epoch": None}
        candidates |= {"step" : None}

        candidates |= {"train/loss": None}
        if pl_module.train_metrics:
            for m in pl_module.train_metrics:
                candidates |= {f"train/{m.name}": None}

        candidates |= {"val/loss": None}
        if pl_module.val_metrics:
            for m in pl_module.val_metrics:
                candidates |= {f"val/{m.name}": None}

        self._candidates = candidates
        return self._candidates
    
    def _update_candidates(
        self,
        monitor_candidates: dict[str, torch.Tensor]
    ) -> dict[str, Any]:
        """Updates logging candidates with monitored values.
    
        Args:
            monitor_candidates: ``dict`` of metric names to ``torch.Tensor`` values.
    
        Returns:
            Updated ``dict`` of candidates.
        """
        candidates = deepcopy(self._candidates)
        for c, v in monitor_candidates.items():
            if c in candidates:
                candidates[c] = v
        candidates["step"] = monitor_candidates.get("global_step", monitor_candidates["step"])
        return candidates
    
    def _get_monitor_candidates(self, trainer: "pl.Trainer") -> dict[str, torch.Tensor]:
        """Gathers metrics for logging from trainer callbacks.
    
        Args:
            trainer: ``pl.Trainer`` instance.
    
        Returns:
            ``dict`` of metric names to ``torch.Tensor`` values.
        """
        monitor_candidates = deepcopy(trainer.callback_metrics)
        # Cast to int if necessary because `self.log("epoch", 123)` will convert
        # it to float. if it is not a tensor or does not exist, we overwrite it
        # as it is likely an error.
        # epoch       = monitor_candidates.get("epoch")
        # step        = monitor_candidates.get("step")
        # global_step = monitor_candidates.get("global_step")
        # monitor_candidates["epoch"]       = epoch.int()       if isinstance(epoch,       torch.Tensor) else torch.tensor(trainer.current_epoch)
        # monitor_candidates["step"]        = step.int()        if isinstance(step,        torch.Tensor) else torch.tensor(trainer.global_step)
        # monitor_candidates["global_step"] = global_step.int() if isinstance(global_step, torch.Tensor) else torch.tensor(trainer.global_step)
        monitor_candidates["epoch"]       = torch.tensor(trainer.current_epoch)
        monitor_candidates["step"]        = torch.tensor(trainer.global_step)
        monitor_candidates["global_step"] = torch.tensor(trainer.global_step)
        return monitor_candidates
    
    def _should_skip_logging(self, trainer: "pl.Trainer") -> bool:
        """Checks if logging should be skipped for current state.
    
        Args:
            trainer: ``pl.Trainer`` instance.
    
        Returns:
            ``True`` if logging should be skipped, else ``False``.
        """
        from lightning.pytorch.trainer.states import TrainerFn
        
        return (
            bool(trainer.fast_dev_run) or                        # disable logging with fast_dev_run
            trainer.state.fn != TrainerFn.FITTING or             # don't log anything during non-fit
            trainer.sanity_checking or                           # don't log anything during sanity check
            self._last_global_step_saved == trainer.global_step  # already log at the last step
        )
    
    def _should_log_on_train_epoch_end(self, trainer: "pl.Trainer") -> bool:
        """Determines if logging at train epoch end.
    
        Args:
            trainer: ``pl.Trainer`` instance.
    
        Returns:
            ``True`` if logging at train epoch end, else ``False``.
        """
        if self._log_on_train_epoch_end:
            return self._log_on_train_epoch_end

        if trainer.check_val_every_n_epoch != 1:
            return False

        num_val_batches = (
            sum(trainer.num_val_batches)
            if isinstance(trainer.num_val_batches, list)
            else trainer.num_val_batches
        )
        if num_val_batches == 0:
            return True

        return trainer.val_check_interval == 1.0
    
    def _log(self, candidates: dict[str, torch.Tensor]):
        """Writes candidate metrics to log file and console.
    
        Args:
            candidates: ``dict`` of metric names to ``torch.Tensor`` values.
        """
        # Log to file
        row = ",".join("" if v is None else str(v) for _, v in candidates.items())
        self._logger.write(f"{row}\n")
        self._logger.flush()

        # Log to console
        if self._verbose:
            row_lines = []
            header    = []
            values    = []
            for i, (c, v) in enumerate(candidates.items()):
                if i > 0 and i % 6 == 0:
                    row_lines.extend([f"{' '.join(header)}", f"{' '.join(values)}"])
                    header = []
                    values = []
                header.append(f"{c:>12}")
                values.append(
                    "" if v is None else
                    "NaN" if math.isnan(v) else
                    f"{int(v):d}" if int(v) == v and int(v) != 0 else
                    f"{v:.6f}"
                ).rjust(12)
            if header and values:
                row_lines.extend([f"{' '.join(header)}", f"{' '.join(values)}"])
            if row_lines:
                print()
                core.console.log("\n".join(row_lines))

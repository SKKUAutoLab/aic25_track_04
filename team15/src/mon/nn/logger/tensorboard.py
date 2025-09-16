#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Extends the Tensorboard logger."""

__all__ = [
    "TensorBoardLogger",
]

import functools
import os
import socket
import time
from typing import Any, Callable

from lightning.fabric.loggers.logger import rank_zero_experiment
from lightning.pytorch import loggers
from tensorboard.compat import tf
from tensorboard.compat.proto import event_pb2
from tensorboard.compat.proto.event_pb2 import Event, SessionLog
from tensorboard.summary.writer import event_file_writer, record_writer
from torch.utils import tensorboard

from mon.constants import LOGGERS


# ----- Tensorboard -----
class EventFileWriter(event_file_writer.EventFileWriter):
    """Writes TensorFlow event files to a log directory.

    Args:
        logdir: Directory to save event files as ``str``.
        max_queue_size: Max size of the event queue. Default is ``10``.
        flush_secs: Seconds between flushes. Default is ``120``.
        filename_suffix: Suffix for event file name. Default is ``""``.
    """

    def __init__(
        self,
        logdir         : str,
        max_queue_size : int = 10,
        flush_secs     : int = 120,
        filename_suffix: str = ""
    ):
        self._logdir = logdir
        tf.io.gfile.makedirs(logdir)
        self._file_name = os.path.join(
            logdir, f"events.out.tfevents.{socket.gethostname()}{filename_suffix}"
        )
        self._general_file_writer = tf.io.gfile.GFile(self._file_name, "wb")
        self._async_writer = event_file_writer._AsyncWriter(
            record_writer  = record_writer.RecordWriter(self._general_file_writer),
            max_queue_size = max_queue_size,
            flush_secs     = flush_secs
        )

        _event = event_pb2.Event(wall_time=time.time(), file_version="brain.Event:2")
        self.add_event(_event)
        self.flush()


class FileWriter(tensorboard.FileWriter):
    """Writes TensorBoard event files to a directory.

    Args:
        log_dir: Directory to write event files.
        max_queue: Queue size for pending events. Default is ``10``.
        flush_secs: Seconds between flushes. Default is ``120``.
        filename_suffix: Suffix for event file names. Default is ``""``.
    """
    
    def __init__(
        self,
        log_dir        : str,
        max_queue      : int = 10,
        flush_secs     : int = 120,
        filename_suffix: str = ""
    ):
        # Sometimes PosixPath is passed in and we need to coerce it to a string in all cases.
        # See if we can remove this in the future if we are actually the ones passing in a PosixPath
        log_dir = str(log_dir)
        self.event_writer = EventFileWriter(log_dir, max_queue, flush_secs, filename_suffix)


class SummaryWriter(tensorboard.SummaryWriter):
    """Manages TensorBoard summary writing."""
    
    def _get_file_writer(self):
        """Returns or recreates the default ``FileWriter`` instance.
    
        Returns:
            ``FileWriter`` instance for logging events.
        """
        if self.all_writers is None or self.file_writer is None:
            self.file_writer = FileWriter(
                log_dir         = self.log_dir,
                max_queue       = self.max_queue,
                flush_secs      = self.flush_secs,
                filename_suffix = self.filename_suffix
            )
            self.all_writers = {self.file_writer.get_logdir(): self.file_writer}
            if self.purge_step:
                most_recent_step = self.purge_step
                self.file_writer.add_event(Event(
                    step         = most_recent_step,
                    file_version = "brain.Event:2"
                ))
                self.file_writer.add_event(Event(
                    step        = most_recent_step,
                    session_log = SessionLog(status = SessionLog.START)
                ))
                self.purge_step = None
        return self.file_writer


def rank_zero_only(fn: Callable) -> Callable:
    """Wraps a function to execute only on rank zero.

    Args:
        fn: ``Callable`` to restrict to rank zero.

    Returns:
        Wrapped ``Callable`` that runs on rank zero or returns ``None``.
    """
    @functools.wraps(fn)
    def wrapped_fn(*args: Any, **kwargs: Any) -> Any | None:
        if rank_zero_only.rank == 0:
            return fn(*args, **kwargs)
        return None
    return wrapped_fn


# This should be part of the cluster environment
def _get_rank() -> int:
    """Retrieves process rank from environment variables.

    Returns:
        Rank as ``int``, defaults to 0 if not found.
    """
    rank_keys = ("RANK", "SLURM_PROCID", "LOCAL_RANK")
    for key in rank_keys:
        rank = os.environ.get(key)
        if rank:
            return int(rank)
    return 0


# Add the attribute to the function but don't overwrite in case Trainer has already set it
rank_zero_only.rank = getattr(rank_zero_only, "rank", _get_rank())


# ----- Tensorboard Logger -----
@LOGGERS.register(name="tensorboard")
@LOGGERS.register(name="tensorboard_logger")
class TensorBoardLogger(loggers.TensorBoardLogger):
    """Logs to TensorBoard with SummaryWriter integration."""
    
    @property
    @rank_zero_experiment
    def experiment(self) -> SummaryWriter:
        """Provides access to the TensorBoard ``SummaryWriter``.
    
        Returns:
            ``SummaryWriter`` instance for TensorBoard logging.
    
        Raises:
            ValueError: If initialized on non-zero global rank.
    
        Example:
            ::
                self.logger.experiment.some_tensorboard_function()
        """
        if self._experiment:
            return self._experiment
        if rank_zero_only.rank != 0:
            raise ValueError("[experiment] must initialize on global_rank=0, "
                             "got non-zero rank.")
        if self.root_dir:
            self._fs.makedirs(self.root_dir, exist_ok=True)
        self._experiment = SummaryWriter(log_dir=self.log_dir, **self._kwargs)
        return self._experiment

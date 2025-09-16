#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements rich model summary callback."""

__all__ = [
    "RichModelSummary",
]

from typing import Any

from lightning.pytorch.callbacks import rich_model_summary
from lightning.pytorch.utilities import model_summary

from mon import core
from mon.constants import CALLBACKS


# ----- Model Summary -----
@CALLBACKS.register(name="rich_model_summary")
class RichModelSummary(rich_model_summary.RichModelSummary):
    """Summarizes LightningModule layers with rich text formatting."""

    @staticmethod
    def summarize(
        summary_data        : list[tuple[str, list[str]]],
        total_parameters    : int,
        trainable_parameters: int,
        model_size          : float,
        *args,
        **summarize_kwargs : Any
    ):
        """Generates rich text summary of model layers and parameters.
    
        Args:
            summary_data: ``list`` of ``tuple``s with column names and row data.
            total_parameters: Total number of parameters as ``int``.
            trainable_parameters: Number of trainable parameters as ``int``.
            model_size: Model size in MB as ``float``.
            args: Additional positional args.
            summarize_kwargs: Additional keyword args as ``Any``.
        """
        table = core.rich.table.Table(header_style="bold magenta")
        table.add_column(" ", style="dim")
        table.add_column("Name", justify="left", no_wrap=True)
        table.add_column("Type")
        table.add_column("Params", justify="right")

        column_names = [name for name, _ in summary_data]
        for column_name in ["In sizes", "Out sizes"]:
            if column_name in column_names:
                table.add_column(column_name, justify="right", style="white")

        rows = list(zip(*[data for _, data in summary_data]))
        for row in rows:
            table.add_row(*row)

        core.console.log(table)

        param_counts = [
            trainable_parameters,
            total_parameters - trainable_parameters,
            total_parameters,
            model_size
        ]
        parameters = [f"{model_summary.get_human_readable_count(int(p)):<10}" for p in param_counts]

        grid = core.rich.table.Table.grid(expand=True)
        grid.add_column()
        grid.add_column()

        grid.add_row(f"[bold]Trainable params[/]: {parameters[0]} ({trainable_parameters})")
        grid.add_row(f"[bold]Non-trainable params[/]: {parameters[1]} ({param_counts[1]})")
        grid.add_row(f"[bold]Total params[/]: {parameters[2]} ({total_parameters})")
        grid.add_row(f"[bold]Total estimated model params size (MB)[/]: {parameters[3]}")

        core.console.log(grid)

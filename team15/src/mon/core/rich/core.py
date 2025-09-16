#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Extends ``rich`` for text formatting in terminal, console, and ``mon`` logging."""

__all__ = [
    "console",
    "error_console",
    "field_style",
    "get_console",
    "get_error_console",
    "level_styles",
    "print_dict",
    "print_table",
    "rich_console_theme",
]

import box
import rich
from plum import dispatch
from rich import panel, pretty, table, theme

from mon.core import system

# ----- Console -----
field_style = {
    "asctime"  : {"color": "green"},
    "levelname": {"bold" : True},
    "file_name": {"color": "cyan"},
    "funcName" : {"color": "blue"}
}

level_styles = {
    "critical": {"bold" : True, "color": "red"},
    "debug"   : {"color": "green"},
    "error"   : {"color": "red"},
    "info"    : {"color": "magenta"},
    "warning" : {"color": "yellow"}
}

rich_console_theme = theme.Theme({
    "debug"   : "dark_green",
    "info"    : "green",
    "warning" : "yellow",
    "error"   : "bright_red",
    "critical": "bold red",
})

console = rich.console.Console(
    color_system    = "auto",
    log_time_format = "[%X]",  # "[%m/%d/%Y %H:%M:%S]",
    soft_wrap       = True,
    width           = system.get_terminal_size()[0],  # 150
    theme           = rich_console_theme,
)

error_console = rich.console.Console(
    color_system    = "auto",
    log_time_format = "[%X]",  # "[%m/%d/%Y %H:%M:%S]",
    soft_wrap       = False,
    width           = system.get_terminal_size()[0],  # 150
    stderr          = True,
    style           = "bold red",
    theme           = rich_console_theme,
)


def get_console() -> rich.console.Console:
    """Gets the global ``rich.console.Console`` object, creating it if needed.

    Returns:
        Global ``rich.console.Console`` instance.
    """
    global console
    if console is None:
        console = rich.console.Console(
            color_system    = "auto",
            log_time_format = "[%X]",  # "[%m/%d/%Y %H:%M:%S]",
            soft_wrap       = False,
            width           = 150,
            theme           = rich_console_theme,
        )
    return console


def get_error_console() -> rich.console.Console:
    """Gets the global error ``rich.console.Console``, creating it if needed.

    Returns:
        Global ``rich.console.Console`` for error logging.
    """
    global error_console
    if error_console is None:
        error_console = rich.console.Console(
            color_system    = "auto",
            log_time_format = "[%X]",  # "[%m/%d/%Y %H:%M:%S]",
            soft_wrap       = False,
            width           = 150,
            stderr          = True,
            style           = "bold red",
            theme           = rich_console_theme,
        )
    return error_console


# ----- Print -----
def print_dict(x: dict | box.Box, title: str = ""):
    """Prints a dictionary with a title using ``rich.pretty.Pretty`` format.

    Args:
        x: Dictionary to print.
        title: Title above the dictionary. Default is ``""``.
        console: Console object to use. Default is ``None``, which uses the global console.

    Raises:
        TypeError: If ``x`` is not a dictionary.
    """
    if isinstance(x, box.Box):
        x = x.to_dict()
    if not isinstance(x, dict):
        raise TypeError(f"[x] must be a dict, got {type(x).__name__}.")
    pr = pretty.Pretty(
        x,
        expand_all    = True,
        indent_guides = True,
        insert_line   = True,
        overflow      = "fold"
    )
    p = panel.Panel(pr, title=f"{title}")
    console.log(p)


@dispatch
def print_table(x: list[dict]):
    """Prints a list of dictionaries as a ``rich.table.Table``.

    Args:
        x: List of dicts with identical keys to print as a table.

    Raises:
        TypeError: If ``x`` is not a list or has non-dict elements.
        ValueError: If dicts in ``x`` lack identical keys or ``x`` is empty.
    """
    if not isinstance(x, list) or not all(isinstance(d, dict) for d in x):
        raise TypeError(f"[x] must be a list of dicts, got {type(x).__name__}.")
    if not x:
        raise ValueError("[x] must not be empty.")
    if not all(set(d.keys()) == set(x[0].keys()) for d in x):
        raise ValueError("All dictionaries in [x] must have identical keys.")
    tab = table.Table(show_header=True, header_style="bold magenta")
    for k in x[0].keys():
        tab.add_column(k, no_wrap=True)
    for d in x:
        row = [f"{v}" for v in d.values()]
        tab.add_row(*row)
    console.log(tab)


@dispatch
def print_table(x: dict, title: str = ""):
    """Prints a dictionary as a ``rich.table.Table``.

    Args:
        x: Dictionary to print as a table.
        title: Title above the table. Default is ``""``.

    Raises:
        TypeError: If ``x`` is not a dictionary.
    """
    if isinstance(x, box.Box):
        x = x.to_dict()
    if not isinstance(x, dict):
        raise TypeError(f"[x] must be a dict, got {type(x).__name__}.")
    tab = table.Table(
        title        = title,
        show_header  = True,
        row_styles   = ["dim", ""],
        header_style = "bold magenta",
        highlight    = True,
    )
    tab.add_column("Key")
    tab.add_column("Value")
    for k, v in x.items():
        row = [f"{k}", f"{v}"]
        tab.add_row(*row)
    console.log(tab)

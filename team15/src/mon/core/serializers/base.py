#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Base class and functions for file handlers with helper utilities."""

__all__ = [
    "BaseSerializer",
    "merge_files",
    "load_from_file",
    "write_to_file",
]

import abc
from typing import Any, TextIO

from mon.constants import SERIALIZERS
from mon.core import pathlib, type_extensions


# ----- Serializer -----
class BaseSerializer(abc.ABC):
    """Base class for loading and writing data in various file formats."""
    
    @abc.abstractmethod
    def load_from_fileobj(self, path: pathlib.Path | TextIO, **kwargs) -> Any:
        """Loads content from a ``file`` object.
    
        Args:
            path: File path or ``TextIO`` stream.
            kwargs: Additional keyword arguments.
        
        Returns:
            Content from the ``file``.
        """
        pass
    
    @abc.abstractmethod
    def write_to_fileobj(self, obj: Any, path: pathlib.Path | TextIO, **kwargs):
        """Writes a serializable object to a ``file`` object.

        Args:
            obj: Serializable object to write.
            path: File path or ``TextIO`` stream.
            kwargs: Additional keyword arguments.
        """
        pass
    
    @abc.abstractmethod
    def write_to_string(self, obj: Any, **kwargs) -> str:
        """Converts a serializable object to a ``str``.

        Args:
            obj: Serializable object to convert.
            kwargs: Additional keyword arguments.

        Returns:
            String representation of the object.
        """
        pass
    
    def load_from_file(self, path: pathlib.Path, mode: str = "r", **kwargs) -> Any:
        """Loads content from a ``file``.

        Args:
            path: File path.
            mode: File open ``mode``. Default is ``"r"`` for read-only.
            kwargs: Additional keyword arguments.
    
        Returns:
            Content from the ``file``.
        """
        with open(path, mode) as f:
            return self.load_from_fileobj(path=f, **kwargs)
    
    def write_to_file(self, obj: Any, path: pathlib.Path, mode: str = "w", **kwargs):
        """Writes a serializable object to a ``file``.
    
        Args:
            obj: Serializable object to write.
            path: File path.
            mode: File open ``mode``. Default is ``"w"`` for write-only.
            kwargs: Additional keyword arguments.
        """
        with open(path, mode) as f:
            self.write_to_fileobj(obj=obj, path=f, **kwargs)


# ----- Functional -----
def write_to_file(
    obj        : Any,
    path       : pathlib.Path | TextIO,
    file_format: str = None,
    **kwargs
):
    """Writes a serializable object to a ``file``.

    Args:
        obj: Object to serialize.
        path: File path or ``TextIO`` stream.
        file_format: File format, inferred from ``path`` if ``None``. Default is ``None``.
        kwargs: Additional keyword arguments.

    Raises:
        ValueError: If ``file_format`` is not supported.
    """
    path_obj    = pathlib.Path(path) if isinstance(path, pathlib.Path | str) else path
    file_format = file_format or (path_obj.suffix if isinstance(path_obj, pathlib.Path) else "")
    if file_format not in SERIALIZERS:
        raise ValueError(f"[file_format] must be one of {list(SERIALIZERS.names())}, "
                         f"got {file_format}")
    
    handler: BaseSerializer = SERIALIZERS.build(name=file_format)
    if hasattr(path, "write"):
        handler.write_to_fileobj(obj=obj, path=path, **kwargs)
    else:
        handler.write_to_file(obj=obj, path=path_obj, **kwargs)


def load_from_file(
    path       : pathlib.Path | TextIO,
    file_format: str = None,
    **kwargs
) -> Any:
    """Loads content from a ``file``.

    Args:
        path: File path or ``TextIO`` stream.
        file_format: File format, inferred from ``path`` if ``None``. Default is ``None``.
        kwargs: Additional keyword arguments.

    Returns:
        ``File`` content.

    Raises:
        TypeError: If ``path`` is not a valid type.
    """
    path_obj    = pathlib.Path(path) if isinstance(path, pathlib.Path | str) else path
    file_format = file_format or (path_obj.suffix if isinstance(path_obj, pathlib.Path) else "")
    
    handler: BaseSerializer = SERIALIZERS.build(name=file_format)
    if hasattr(path, "read"):
        return handler.load_from_fileobj(path=path, **kwargs)
    if isinstance(path_obj, pathlib.Path | str):
        return handler.load_from_file(path=path_obj, **kwargs)
    raise TypeError(f"[path] must be str, pathlib.Path, or file-like, "
                    f"got {type(path).__name__}.")


def merge_files(
    in_paths   : list[pathlib.Path | TextIO],
    out_path   : pathlib.Path | TextIO,
    file_format: str = None,
):
    """Merges content from multiple ``files`` into a single ``file``.

    Args:
        in_paths: List of input file paths or ``TextIO`` streams.
        out_path: Output file path or ``TextIO`` stream.
        file_format: File format, inferred from ``out_path`` if ``None``.
            Default is ``None``.

    Raises:
        TypeError: If content from ``in_paths`` is neither ``list`` nor ``dict``.
    """
    in_paths = [pathlib.Path(p) for p in type_extensions.to_list(in_paths)]
    data = None
    for input_path in in_paths:
        content = load_from_file(path=input_path)
        if isinstance(content, list):
            data = data or []
            data.extend(content)
        elif isinstance(content, dict):
            data = data or {}
            data.update(content)
        else:
            raise TypeError(f"[in_paths] content must be list or dict, "
                            f"got {type(content).__name__}.")
    
    write_to_file(obj=data, path=out_path, file_format=file_format)

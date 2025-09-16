#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Extends ``pickle`` module for Pickle file handling."""

from pickle import *
from typing import Any, TextIO

from mon.constants import SERIALIZERS
from mon.core import pathlib
from mon.core.serializers import base


# ----- Serializer -----
@SERIALIZERS.register(name=".pickle")
@SERIALIZERS.register(name=".pkl")
class PickleSerializer(base.BaseSerializer):
    """Handler for Pickle file operations."""
    
    def load_from_fileobj(self, path: TextIO, **kwargs) -> Any:
        """Loads data from a file object.

        Args:
            path: File stream as ``TextIO`` (binary file-like object).
            **kwargs: Extra args for ``pickle.load``.
    
        Returns:
            Deserialized Pickle data as ``Any``.
        """
        return load(path, **kwargs)
    
    def write_to_fileobj(self, obj: Any, path: TextIO, **kwargs):
        """Writes data to a file object.

        Args:
            obj: Data to serialize as ``Any``.
            path: File stream as ``TextIO`` (binary file-like object).
            **kwargs: Extra args for ``pickle.dump``.
    
        """
        kwargs.setdefault("protocol", 4)
        dump(obj, path, **kwargs)
    
    def write_to_string(self, obj: Any, **kwargs) -> bytes:
        """Converts data to a Pickle byte string.
    
        Args:
            obj: Data to serialize as ``Any``.
            **kwargs: Extra args for ``pickle.dumps``.
    
        Returns:
            Pickle byte string as ``bytes``.
        """
        kwargs.setdefault("protocol", 4)
        return dumps(obj, **kwargs)
    
    def load_from_file(self, path: pathlib.Path, mode: str = "rb", **kwargs) -> Any:
        """Loads data from a file.
    
        Args:
            path: File path.
            mode: File mode. Default is ``rb``.
            **kwargs: Extra args for ``read_from_fileobj()``.
    
        Returns:
            Deserialized Pickle data as ``Any``.
        """
        return super().load_from_file(path=pathlib.Path(path), mode=mode, **kwargs)
    
    def write_to_file(self, obj: Any, path: pathlib.Path, mode: str = "wb", **kwargs):
        """Writes data to a file.
    
        Args:
            obj: Data to serialize as ``Any``.
            path: File path.
            mode: File mode. Default is ``wb``.
            **kwargs: Extra args for ``write_to_fileobj()``.
    
        """
        super().write_to_file(obj=obj, path=pathlib.Path(path), mode=mode, **kwargs)

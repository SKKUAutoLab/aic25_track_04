#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Extends ``json`` package for JSON file handling."""

from json import *
from typing import Any, TextIO

import numpy as np

from mon.constants import SERIALIZERS
from mon.core import pathlib
from mon.core.serializers import base


# ----- Serializer -----
@SERIALIZERS.register(name=".json")
class JSONSerializer(base.BaseSerializer):
    """Handler for JSON file operations."""
    
    @staticmethod
    def set_default(obj: Any) -> Any:
        """Converts non-JSON-serializable objects to a serializable format.
    
        Args:
            obj: Object to convert.
    
        Returns:
            JSON-serializable representation of ``obj``.
    
        Raises:
            TypeError: If ``obj`` type is unsupported.
        """
        if isinstance(obj, set | range | np.ndarray):
            return list(obj)
        if isinstance(obj, np.generic):
            return obj.item()
        raise TypeError(f"[obj] type [{type(obj).__name__}] is not JSON-serializable")
    
    def load_from_fileobj(self, path: pathlib.Path | TextIO, **kwargs) -> Any:
        """Loads data from a ``file`` object.
    
        Args:
            path: File path or ``TextIO`` stream.
            kwargs: Additional arguments for ``json.load``.
    
        Returns:
            Deserialized JSON data.
        """
        return load(path, **kwargs)
    
    def write_to_fileobj(self, obj: Any, path: pathlib.Path | TextIO, **kwargs):
        """Writes data to a ``file`` object.
    
        Args:
            obj: Data to serialize.
            path: File path or ``TextIO`` stream.
            kwargs: Additional arguments for ``json.dump``.
        """
        kwargs.setdefault("default", self.set_default)
        dump(obj, path, **kwargs)
    
    def write_to_string(self, obj: Any, **kwargs) -> str:
        """Converts data to a JSON ``str``.
    
        Args:
            obj: Data to serialize.
            kwargs: Additional arguments for ``json.dumps``.
    
        Returns:
            JSON ``str`` representation of ``obj``.
        """
        kwargs.setdefault("default", self.set_default)
        return dumps(obj, **kwargs)

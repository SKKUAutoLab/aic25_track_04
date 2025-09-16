#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Extends ``xmltodict`` module for XML file handling."""

from typing import Any, TextIO

from xmltodict import *

from mon.constants import SERIALIZERS
from mon.core.serializers import base


# ----- Serializer -----
@SERIALIZERS.register(name=".xml")
class XMLSerializer(base.BaseSerializer):
    """Handler for XML file operations."""
    
    def load_from_fileobj(self, path: TextIO, **kwargs) -> Any:
        """Loads data from a file object.

        Args:
            path: File stream as ``TextIO`` (text file-like object).
            **kwargs: Extra args for ``xmltodict.parse()``.

        Returns:
            Parsed XML data as a dictionary.
        """
        return parse(path.read(), **kwargs)
    
    def write_to_fileobj(self, obj: Any, path: TextIO, **kwargs):
        """Writes data to a file object.

        Args:
            obj: Dictionary to serialize as XML.
            path: File stream as ``TextIO`` (text file-like object).
            **kwargs: Extra args for ``xmltodict.unparse()``.

        Raises:
            TypeError: If ``obj`` is not a dictionary.
        """
        if not isinstance(obj, dict):
            raise TypeError(f"[obj] must be a dict, got {type(obj).__name__}.")
        path.write(unparse(input_dict=obj, pretty=True, **kwargs))
    
    def write_to_string(self, obj: Any, **kwargs) -> str:
        """Converts a dictionary to an XML string.

        Args:
            obj: Dictionary to serialize as XML.
            **kwargs: Extra args for ``xmltodict.unparse()``.

        Returns:
            XML string representation of ``obj``.

        Raises:
            TypeError: If ``obj`` is not a dictionary.
        """
        if not isinstance(obj, dict):
            raise TypeError(f"[obj] must be a dict, got {type(obj).__name__}.")
        return unparse(input_dict=obj, pretty=True, **kwargs)

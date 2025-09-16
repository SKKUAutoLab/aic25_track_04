#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Defines global constants for ``mon`` package.

Notes:
    - To avoid circular dependency, only define constants of basic/atomic types.
    - The same goes for type aliases.
    - The only exception is the enums and factory constants.
"""

from mon.constants.enums import *
from mon.constants.values import *

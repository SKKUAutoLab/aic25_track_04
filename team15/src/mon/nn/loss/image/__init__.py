#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements loss functions for images.

The categories align with common loss function roles in computer vision:
    - objective.py : image quality metrics (objective fidelity).
    - perceptual.py: perceptual losses (human-like perception).
    - structural.py: edge/structural regularization (detail preservation).
    - spatial.py   : spatial consistency (coherence across regions).
    - color.py     : color/illumination consistency (photometric accuracy).
"""

from mon.nn.loss.image.color import *
from mon.nn.loss.image.objective import *
from mon.nn.loss.image.perceptual import *
from mon.nn.loss.image.spatial import *
from mon.nn.loss.image.structural import *

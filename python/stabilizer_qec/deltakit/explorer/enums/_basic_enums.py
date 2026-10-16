# (c) Copyright Riverlane 2020-2026. All rights reserved.
"""
This module defines Enums used in hardware-aware noise modelling.
These enums are used in `qpus/_qpu_factory.py` and
`noise_models/_noise_factory.py` modules.
"""

from enum import Enum


class DrawingColours(Enum):
    """
    Enumeration of colours used in drawing patch diagrams.

    The options are:
    - X_COLOUR
    - Z_COLOUR
    - DATA_QUBIT_COLOUR
    - ANCILLA_QUBIT_COLOUR
    """

    X_COLOUR = "#d3d1c0"
    Z_COLOUR = "#3ccbda"
    DATA_QUBIT_COLOUR = "#006f62"
    ANCILLA_QUBIT_COLOUR = "#ff7500"

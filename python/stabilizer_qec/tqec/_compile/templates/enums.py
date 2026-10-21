# Ported from tqec (tqec/tqec 0.2.0, src/tqec/templates/enums.py; Apache License 2.0, see LICENSE in stabilizer_qec/tqec),
# modified for stabilizer_qec.
"""Defines enumerations that are only used in :mod:`tqec.templates`."""

from __future__ import annotations

from enum import Enum, auto


class TemplateBorder(Enum):
    TOP = auto()
    BOTTOM = auto()
    LEFT = auto()
    RIGHT = auto()

    def opposite(self) -> TemplateBorder:  # pragma: no cover
        """Return the opposite border."""
        if self == TemplateBorder.TOP:
            return TemplateBorder.BOTTOM
        elif self == TemplateBorder.BOTTOM:
            return TemplateBorder.TOP
        elif self == TemplateBorder.LEFT:
            return TemplateBorder.RIGHT
        elif self == TemplateBorder.RIGHT:
            return TemplateBorder.LEFT
        # wildcard entry added as it was flagged by ty
        else:
            raise ValueError(f"Cannot return the opposite border of {self}.")

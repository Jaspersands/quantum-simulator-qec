# Ported from tqec (tqec/tqec 0.2.0, src/tqec/compile/blocks/enums.py; Apache License 2.0, see LICENSE in stabilizer_qec/tqec),
# modified for stabilizer_qec.
from __future__ import annotations

from enum import Enum

from stabilizer_qec.tqec._compile.templates.enums import TemplateBorder
from stabilizer_qec.tqec.position import Direction3D, SignedDirection3D


class SpatialBlockBorder(Enum):
    """Enumerates the 4 different spatial borders for a block."""

    X_NEGATIVE = SignedDirection3D(Direction3D.X, False)
    X_POSITIVE = SignedDirection3D(Direction3D.X, True)
    Y_NEGATIVE = SignedDirection3D(Direction3D.Y, False)
    Y_POSITIVE = SignedDirection3D(Direction3D.Y, True)

    def to_template_border(self) -> TemplateBorder:
        """Return the template border corresponding to ``self``."""
        if self == SpatialBlockBorder.X_NEGATIVE:
            return TemplateBorder.LEFT
        elif self == SpatialBlockBorder.X_POSITIVE:
            return TemplateBorder.RIGHT
        elif self == SpatialBlockBorder.Y_NEGATIVE:
            return TemplateBorder.TOP
        elif self == SpatialBlockBorder.Y_POSITIVE:
            return TemplateBorder.BOTTOM
        # add a wildcard pattern when function returns `None`
        # flagged by ty
        else:
            raise ValueError(f"Cannot return the template border corresponding to {self} .")


class TemporalBlockBorder(Enum):
    """Enumerates the 2 different temporal borders for a block."""

    Z_NEGATIVE = SignedDirection3D(Direction3D.Z, False)
    Z_POSITIVE = SignedDirection3D(Direction3D.Z, True)


def border_from_signed_direction(
    direction: SignedDirection3D,
) -> SpatialBlockBorder | TemporalBlockBorder:
    """Get the block border from its direction.

    Args:
        direction: direction indicating the border to return.

    Returns:
        The border corresponding to the provided direction. Imagining a line
        starting from the block center and extending in the provided direction,
        the border pierced by the line is returned by this function.

        For example, if ``"+X"`` is provided, the returned border will be
        ``SpatialBlockBorder.X_POSITIVE``

    """
    if isinstance(direction, SignedDirection3D) and direction.direction == Direction3D.Z:
        return TemporalBlockBorder(direction)
    else:
        return SpatialBlockBorder(direction)

# Ported from tqec (tqec/tqec 0.2.0, src/tqec/utils/position.py; Apache License 2.0; see LICENSE in this directory),
# modified for stabilizer_qec: the 3D classes and Position2D only (not the 2D shapes and shifts of tqec's
# templates), numpy imported only by FloatPosition3D.as_array.
"""3D integer positions, directions and signed directions of a spacetime diagram."""

from __future__ import annotations

import re
from dataclasses import astuple, dataclass
from enum import Enum

from ._errors import TQECError


@dataclass(frozen=True, order=True)
class Position2D:
    """A 2D integer position (``Position3D.as_2d``)."""

    x: int
    y: int

    def is_neighbour(self, other: Position2D) -> bool:
        """Check if the other position is near to this position, i.e. Manhattan distance is 1."""
        return abs(self.x - other.x) + abs(self.y - other.y) == 1

    def to_3d(self, z: int = 0) -> Position3D:
        """The 3D position with ``self``'s ``x`` and ``y``, at ``z``."""
        return Position3D(self.x, self.y, z)


@dataclass(frozen=True, order=True)
class Position3D:
    """A 3D integer position: x, y are space and z is time."""

    x: int
    y: int
    z: int

    def shift_by(self, dx: int = 0, dy: int = 0, dz: int = 0) -> Position3D:
        """Shift the position by the given offset."""
        return Position3D(self.x + dx, self.y + dy, self.z + dz)

    def shift_in_direction(self, direction: Direction3D, shift: int) -> Position3D:
        """Shift the position in the given direction by the given shift."""
        if direction == Direction3D.X:
            return self.shift_by(dx=shift)
        elif direction == Direction3D.Y:
            return self.shift_by(dy=shift)
        else:
            return self.shift_by(dz=shift)

    def is_neighbour(self, other: Position3D) -> bool:
        """Check if the other position is near to this position, i.e. Manhattan distance is 1."""
        return abs(self.x - other.x) + abs(self.y - other.y) + abs(self.z - other.z) == 1

    def as_tuple(self) -> tuple[int, int, int]:
        """Return the position as a tuple."""
        return astuple(self)

    def __str__(self) -> str:
        return f"({self.x},{self.y},{self.z})"

    def as_2d(self) -> Position2D:
        """Return the position as a 2D position."""
        return Position2D(self.x, self.y)


class Direction3D(Enum):
    """Axis directions in the 3D spacetime diagram."""

    X = 0
    Y = 1
    Z = 2

    @staticmethod
    def all_directions() -> list[Direction3D]:
        """Return all the directions."""
        return [Direction3D.X, Direction3D.Y, Direction3D.Z]

    @staticmethod
    def spatial_directions() -> list[Direction3D]:
        """Return all the spatial directions."""
        return [Direction3D.X, Direction3D.Y]

    @staticmethod
    def temporal_directions() -> list[Direction3D]:
        """Return all the temporal directions."""
        return [Direction3D.Z]

    def __str__(self) -> str:
        return self.name

    @staticmethod
    def from_neighbouring_positions(source: Position3D, sink: Position3D) -> Direction3D:
        """Return the direction to go from ``source`` to ``sink``."""
        assert source.is_neighbour(sink)
        for direction, (source_coord, sink_coord) in zip(Direction3D.all_directions(), zip(source.as_tuple(), sink.as_tuple())):
            if source_coord != sink_coord:
                return direction
        raise TQECError(f"Could not find the direction from two neighbouring positions {source} and {sink}.")

    @property
    def orthogonal_directions(self) -> tuple[Direction3D, Direction3D]:
        """Return the two directions orthogonal to ``self``."""
        i = self.value
        return Direction3D((i + 1) % 3), Direction3D((i + 2) % 3)


@dataclass(frozen=True)
class SignedDirection3D:
    """Signed directions in the 3D spacetime diagram."""

    direction: Direction3D
    towards_positive: bool

    def __neg__(self) -> SignedDirection3D:
        return SignedDirection3D(self.direction, not self.towards_positive)

    def __str__(self) -> str:
        return f"{'+' if self.towards_positive else '-'}{self.direction}"

    @staticmethod
    def from_string(s: str) -> SignedDirection3D:
        """Return the signed direction from a string of the form ``[+-][XYZ]``."""
        match = re.match(r"([+-])([XYZ])", s)
        if match is None:
            raise TQECError(f"Invalid signed direction: {s}, expected format: [+/-][XYZ]")
        sign, direction = match.groups()
        return SignedDirection3D(Direction3D("XYZ".index(direction)), sign == "+")


@dataclass(frozen=True, order=True)
class FloatPosition3D:
    """A 3D float position."""

    x: float
    y: float
    z: float

    def shift_by(self, dx: float = 0, dy: float = 0, dz: float = 0) -> FloatPosition3D:
        """Shift the position by the given offset."""
        return FloatPosition3D(self.x + dx, self.y + dy, self.z + dz)

    def shift_in_direction(self, direction: Direction3D, shift: float) -> FloatPosition3D:
        """Shift the position in the given direction by the given shift."""
        if direction == Direction3D.X:
            return self.shift_by(dx=shift)
        elif direction == Direction3D.Y:
            return self.shift_by(dy=shift)
        else:
            return self.shift_by(dz=shift)

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)

    def as_array(self):
        """Return the position as a numpy array (of float32, as tqec's)."""
        import numpy as np

        return np.asarray(astuple(self), dtype=np.float32)


def round_or_fail(f: float, atol: float = 1e-8) -> int:
    """Round a float to an integer, failing when it is not within ``atol`` of one."""
    rounded = round(f)
    if abs(f - rounded) > atol:
        raise TQECError(f"Rounding from {f} to integer failed.")
    return int(rounded)

# Ported from tqec (tqec/tqec 0.2.0, src/tqec/utils/scale.py; Apache License 2.0, see LICENSE in
# this directory), modified for stabilizer_qec: round_or_fail lives in position.py (re-exported here),
# and PhysicalQubitScalable2D.to_grid_qubit imports GridQubit when called.
"""tqec's ``LinearFunction``, ``slope * k + offset``: the type of tqec's
``block_temporal_height`` (``LinearFunction(2, -1)``, 2k - 1 rounds between a slice's
initialisation and measurement layers, by default), and ``Scalable2D``, a pair of them: the
shape of a template, in plaquettes or physical qubits, at every ``k``."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Optional, Union

from ._errors import TQECError
from .position import PhysicalQubitShape2D, PlaquetteShape2D, Shape2D, Shift2D, round_or_fail


@dataclass(frozen=True)
class LinearFunction:
    """A linear function, described by its slope and offset."""

    slope: float = 1.0
    offset: float = 0.0

    def __call__(self, x: float) -> float:
        return self.slope * x + self.offset

    def __add__(self, other: Union[LinearFunction, int, float]) -> LinearFunction:
        if isinstance(other, (int, float)):
            other = LinearFunction(0, other)
        return LinearFunction(self.slope + other.slope, self.offset + other.offset)

    def __sub__(self, other: Union[LinearFunction, int, float]) -> LinearFunction:
        if isinstance(other, (int, float)):
            other = LinearFunction(0, other)
        return LinearFunction(self.slope - other.slope, self.offset - other.offset)

    def __mul__(self, other: Union[int, float]) -> LinearFunction:
        return self.__rmul__(other)

    def __rmul__(self, other: Union[int, float]) -> LinearFunction:
        return LinearFunction(other * self.slope, other * self.offset)

    def intersection(self, other: LinearFunction) -> Optional[float]:
        """The x where ``self(x) == other(x)``, or None for parallel functions."""
        if self.slope == other.slope:
            return None
        return -(other.offset - self.offset) / (other.slope - self.slope)

    @staticmethod
    def _from(obj: Union[LinearFunction, float]) -> LinearFunction:
        if isinstance(obj, (float, int)):
            return LinearFunction(0, obj)
        return obj

    def __repr__(self) -> str:
        if abs(self.slope) < 1e-8:
            return str(self.offset)
        if abs(self.offset) < 1e-8:
            return f"{self.slope}*x"
        return f"{self.slope}*x + {self.offset}"

    def integer_eval(self, x: int) -> int:
        """``self(x)`` as an integer (a TQECError if it is not one)."""
        return round_or_fail(self.slope * x + self.offset)

    def exact_integer_div(self, div: int) -> LinearFunction:
        """``self / div``, when ``div`` divides the slope and the offset."""
        if div == 0:
            raise ZeroDivisionError()
        slope, offset = round_or_fail(self.slope), round_or_fail(self.offset)
        if slope % div != 0:
            raise TQECError(f"Trying to divide exactly a LinearFunction by an integer that is not a multiple of the slope. Divisor: {div}. Slope: {slope}.")
        if offset % div != 0:
            raise TQECError(f"Trying to divide exactly a LinearFunction by an integer that is not a multiple of the offset. Divisor: {div}. Offset: {offset}.")
        return LinearFunction(slope // div, offset // div)

    def is_constant(self, atol: float = 1e-8) -> bool:
        return abs(self.slope) < atol

    def is_scalable(self, atol: float = 1e-8) -> bool:
        return not self.is_constant(atol)

    def is_close_to(self, other: LinearFunction, atol: float = 1e-8) -> bool:
        return abs(self.slope - other.slope) < atol and abs(self.offset - other.offset) < atol

    @staticmethod
    def unambiguous_max_on_positives(fs: Iterable[LinearFunction], default: Optional[LinearFunction] = None) -> LinearFunction:
        """The function of ``fs`` at least every other on the positive numbers (a TQECError when
        there is none). As tqec's, the check runs over ``fs`` a second time: given an iterator,
        it has nothing left to check."""
        if default is None:
            default = LinearFunction(0, 0)
        iterator = iter(fs)
        try:
            res: LinearFunction = next(iterator)
        except StopIteration:
            return default
        for f in iterator:
            if f.offset >= res.offset and f.slope >= res.slope:
                res = f
        for f in fs:
            if f.offset > res.offset or f.slope > res.slope:
                raise TQECError(f"Could not find a unambiguous maximum in the provided linear functions. {res} could be the maximum, but is ambiguous with {f}.")
        return res

    @staticmethod
    def safe_mul(lhs: LinearFunction, rhs: LinearFunction) -> LinearFunction:
        """``lhs * rhs``, when it is linear (one of them constant)."""
        if lhs.slope != 0 and rhs.slope != 0:
            raise TQECError(f"The result of ({lhs}) * ({rhs}) is not a linear function.")
        return LinearFunction(lhs.slope * rhs.offset + rhs.slope * lhs.offset, lhs.offset * rhs.offset)


@dataclass(frozen=True)
class Scalable2D:
    """A pair of scalable quantities: ``x`` and ``y``, each a linear function of ``k``."""

    x: LinearFunction
    y: LinearFunction

    def to_shape_2d(self, k: int) -> Shape2D:
        """``Shape2D(self.x.integer_eval(k), self.y.integer_eval(k))``."""
        return Shape2D(self.x.integer_eval(k), self.y.integer_eval(k))

    def to_numpy_shape(self, k: int) -> tuple[int, int]:
        """The shape at ``k`` in numpy's coordinates, ``(y, x)``."""
        return self.to_shape_2d(k).to_numpy_shape()

    @staticmethod
    def _get_x_y(other):
        if isinstance(other, tuple):
            return other
        elif isinstance(other, (Scalable2D, Shift2D)):
            return other.x, other.y
        else:
            raise TypeError("Unsupported input provided.")

    def __add__(self, other):
        x, y = Scalable2D._get_x_y(other)
        return self.__class__(self.x + x, self.y + y)

    def __sub__(self, other):
        x, y = Scalable2D._get_x_y(other)
        return self.__class__(self.x - x, self.y - y)


class PlaquetteScalable2D(Scalable2D):
    """A pair of scalable quantities in plaquette coordinates."""

    def to_shape_2d(self, k: int) -> PlaquetteShape2D:
        """Evaluate both coordinates with ``k`` as input."""
        return PlaquetteShape2D(self.x.integer_eval(k), self.y.integer_eval(k))

    def __mul__(self, other: Shift2D) -> PhysicalQubitScalable2D:
        return PhysicalQubitScalable2D(self.x * other.x, self.y * other.y)


class PhysicalQubitScalable2D(Scalable2D):
    """A pair of scalable quantities in physical qubit coordinates."""

    def to_shape_2d(self, k: int) -> PhysicalQubitShape2D:
        """Evaluate both coordinates with ``k`` as input."""
        return PhysicalQubitShape2D(self.x.integer_eval(k), self.y.integer_eval(k))

    def to_grid_qubit(self, k: int):
        """Evaluate both coordinates with ``k`` as input, returning a ``GridQubit``."""
        from ._compile.circuit.qubit import GridQubit

        return GridQubit(self.x.integer_eval(k), self.y.integer_eval(k))


__all__ = ["LinearFunction", "PhysicalQubitScalable2D", "PlaquetteScalable2D", "Scalable2D", "round_or_fail"]

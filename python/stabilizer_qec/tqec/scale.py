# Ported from tqec (tqec/tqec 0.2.0, src/tqec/utils/scale.py; Apache License 2.0, see LICENSE in
# this directory), modified for stabilizer_qec: LinearFunction and round_or_fail only (the
# Scalable2D shapes are those of tqec's templates, which this port does not have).
"""tqec's ``LinearFunction``, ``slope * k + offset``: the type of tqec's
``block_temporal_height`` (``LinearFunction(2, -1)``, 2k - 1 rounds between a slice's
initialisation and measurement layers, by default)."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Optional, Union

from ._errors import TQECError
from .position import round_or_fail


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
        there is none)."""
        if default is None:
            default = LinearFunction(0, 0)
        fs = list(fs)
        if not fs:
            return default
        res = fs[0]
        for f in fs[1:]:
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


__all__ = ["LinearFunction", "round_or_fail"]

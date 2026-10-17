# Ported from tqec (tqec/tqec 0.2.0, src/tqec/utils/enums.py; Apache License 2.0; see LICENSE in this directory),
# modified for stabilizer_qec.
"""Bases and Paulis as tqec names them."""

from __future__ import annotations

from enum import Enum, Flag
from typing import Generator

from ._errors import TQECError


class Basis(Enum):
    X = "X"
    Z = "Z"

    def flipped(self) -> Basis:
        """Flip ``self``, returning the opposite basis."""
        return Basis.X if self == Basis.Z else Basis.Z

    def to_pauli(self) -> Pauli:
        """Convert to the corresponding Pauli operator."""
        return Pauli.X if self == Basis.X else Pauli.Z

    def __str__(self) -> str:
        return self.value

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}.{self.value}"

    def __lt__(self, other: Basis) -> bool:
        return self.value < other.value


class Pauli(Flag):
    """Pauli operators as bit flags of X and Z supports."""

    I = 0  # noqa: E741
    X = 1
    Z = 2
    Y = X | Z

    def flipped(self, condition: bool = True) -> Pauli:
        """Return the Pauli operator with X and Z supports flipped."""
        return ~self if condition else self

    def to_basis(self) -> Basis:
        """Convert to the corresponding ``Basis``."""
        if self is Pauli.X:
            return Basis.X
        if self is Pauli.Z:
            return Basis.Z
        raise TQECError(f"Cannot convert Pauli {self} to Basis.")

    def to_basis_set(self) -> set[Basis]:
        """Convert to the corresponding set of bases."""
        bases = list(Basis)
        return {bases[basis.value >> 1] for basis in self.iter_xz() if basis in self}

    @staticmethod
    def from_basis_set(bases: set[Basis]) -> Pauli:
        """Convert a set of bases to the corresponding Pauli operator."""
        return Pauli((Basis.X in bases) | ((Basis.Z in bases) << 1))

    def __invert__(self) -> Pauli:
        value = self.value
        return Pauli((value >> 1) | ((value % 2) << 1))

    def __str__(self) -> str:
        return "IXZY"[self.value]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}.{self!s}"

    @staticmethod
    def iter_xz() -> Generator[Pauli, None, None]:
        """Iterate over the X and Z Pauli operators."""
        yield from map(Pauli, range(1, 3))

    @staticmethod
    def iter_xzy() -> Generator[Pauli, None, None]:
        """Iterate over the X, Z, Y Pauli operators."""
        yield from map(Pauli, range(1, 4))

    @staticmethod
    def iter_ixzy() -> Generator[Pauli, None, None]:
        """Iterate over the I, X, Z, Y Pauli operators."""
        yield from map(Pauli, range(4))

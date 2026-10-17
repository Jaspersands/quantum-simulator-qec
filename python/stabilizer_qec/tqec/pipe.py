# Ported from tqec (tqec/tqec 0.2.0, src/tqec/computation/pipe.py; Apache License 2.0; see LICENSE in this directory),
# modified for stabilizer_qec.
"""Pipes: the connections between cubes, and their kinds."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Generator, Optional

from ._errors import TQECError
from .cube import Cube, ZXCube
from .enums import Basis
from .position import Direction3D


@dataclass(frozen=True)
class PipeKind:
    """The kind of a pipe: at its head (the end with the smaller position), the basis of the
    walls seen along x, y and z (``None`` along the pipe's own direction), and whether it has a
    Hadamard transition."""

    x: Optional[Basis]
    y: Optional[Basis]
    z: Optional[Basis]
    has_hadamard: bool = False

    def __post_init__(self) -> None:
        if sum(basis is None for basis in (self.x, self.y, self.z)) != 1:
            raise TQECError("Exactly one basis must be None for a pipe.")
        if len({self.x, self.y, self.z}) != 3:
            raise TQECError("Pipe must have different basis walls.")

    def __str__(self) -> str:
        return "".join(basis.value if basis is not None else "O" for basis in (self.x, self.y, self.z)) + ("H" if self.has_hadamard else "")

    @staticmethod
    def from_str(string: str) -> PipeKind:
        """A pipe kind from ``XOZ``-style text (``O`` along the pipe), with ``H`` for a Hadamard."""
        string = string.upper()
        has_hadamard = len(string) == 4 and string[3] == "H"
        bases = [Basis(s) if s != "O" else None for s in string[:3]]
        return PipeKind(bases[0], bases[1], bases[2], has_hadamard=has_hadamard)

    @property
    def direction(self) -> Direction3D:
        """The direction along which the pipe connects the cubes."""
        return Direction3D(str(self).index("O"))

    def get_basis_along(self, direction: Direction3D, at_head: bool = True) -> Optional[Basis]:
        """The wall basis in the direction, at the head or (``at_head=False``) the tail; ``None``
        along the pipe."""
        if direction == self.direction:
            return None
        head_basis = Basis(str(self)[direction.value])
        if not at_head and self.has_hadamard:
            return head_basis.flipped()
        return head_basis

    @property
    def is_temporal(self) -> bool:
        return self.z is None

    @property
    def is_spatial(self) -> bool:
        return not self.is_temporal

    @staticmethod
    def _from_cube_kind(cube_kind: ZXCube, direction: Direction3D, cube_at_head: bool, has_hadamard: bool = False) -> PipeKind:
        """The pipe kind inferred from a cube at one end and the pipe's direction."""
        if not cube_at_head and has_hadamard:
            bases = [basis.flipped() for basis in cube_kind.as_tuple()]
        else:
            bases = list(cube_kind.as_tuple())
        bases_str = [basis.value for basis in bases]
        bases_str[direction.value] = "O"
        if has_hadamard:
            bases_str.append("H")
        return PipeKind.from_str("".join(bases_str))


@dataclass(frozen=True)
class Pipe:
    """A block connecting two cubes; ``u`` is the end with the smaller position."""

    u: Cube
    v: Cube
    kind: PipeKind

    def __post_init__(self) -> None:
        u, v = self.u, self.v
        p1, p2 = u.position, v.position
        shift = [0, 0, 0]
        shift[self.kind.direction.value] = 1
        if not p1.shift_by(*shift) == p2 and not p2.shift_by(*shift) == p1:
            raise TQECError(f"The pipe must connect two nearby cubes in direction {self.kind.direction}.")
        if p1 > p2:
            object.__setattr__(self, "u", v)
            object.__setattr__(self, "v", u)

    @staticmethod
    def from_cubes(u: Cube, v: Cube) -> Pipe:
        """The pipe connecting two cubes, its kind inferred from them."""
        if not u.is_zx_cube and not v.is_zx_cube:
            raise TQECError("At least one cube must be a ZX cube to infer the pipe kind.")
        u, v = (u, v) if u.position < v.position else (v, u)
        if not u.position.is_neighbour(v.position):
            raise TQECError("The cubes must be neighbours to create a pipe.")
        direction = next(d for d in Direction3D.all_directions() if u.position.shift_in_direction(d, 1) == v.position)
        if not u.is_zx_cube or not v.is_zx_cube:
            infer_from = u if u.is_zx_cube else v
            cube_kind = infer_from.kind
            assert isinstance(cube_kind, ZXCube)
            return Pipe(u, v, PipeKind._from_cube_kind(cube_kind, direction, infer_from == u))
        assert isinstance(u.kind, ZXCube) and isinstance(v.kind, ZXCube)
        has_hadamard = {u.kind.get_basis_along(d) != v.kind.get_basis_along(d) for d in Direction3D.all_directions() if d != direction}
        if len(has_hadamard) == 2:
            raise TQECError("Cannot infer a valid pipe kind from the cubes.")
        return Pipe(u, v, PipeKind._from_cube_kind(u.kind, direction, True, has_hadamard.pop()))

    @property
    def direction(self) -> Direction3D:
        return self.kind.direction

    def at_head(self, position) -> bool:
        """Whether the position is at the head (u) of the pipe."""
        if position == self.u.position:
            return True
        if position == self.v.position:
            return False
        raise TQECError(f"The position {position} is not an endpoint of the pipe.")

    def __iter__(self) -> Generator[Cube, None, None]:
        yield self.u
        yield self.v

    def __str__(self) -> str:
        return f"{self.u}--({self.kind})--{self.v}"

    def to_dict(self) -> dict[str, Any]:
        return {"u": self.u.position.as_tuple(), "v": self.v.position.as_tuple(), "kind": str(self.kind)}

# Ported from tqec (tqec/tqec 0.2.0, src/tqec/computation/cube.py; Apache License 2.0; see LICENSE in this directory),
# modified for stabilizer_qec.
"""Cubes: the blocks of a spacetime diagram, and their kinds."""

from __future__ import annotations

from dataclasses import astuple, dataclass
from typing import Any, Union

from ._errors import TQECError
from .enums import Basis
from .position import Direction3D, Position3D


@dataclass(frozen=True)
class ZXCube:
    """The kind of cubes consisting of only X or Z basis boundaries: looking at the cube along the
    x, y and z axes, the basis of the walls observed."""

    x: Basis
    y: Basis
    z: Basis

    def __post_init__(self) -> None:
        if self.x == self.y == self.z:
            raise TQECError("The cube with the same basis along all axes is not allowed.")

    def as_tuple(self) -> tuple[Basis, Basis, Basis]:
        return (self.x, self.y, self.z)

    def __str__(self) -> str:
        return f"{self.x}{self.y}{self.z}"

    @staticmethod
    def all_kinds() -> list[ZXCube]:
        """All the allowed ``ZXCube`` instances."""
        return [ZXCube.from_str(s) for s in ["ZXZ", "XZZ", "ZXX", "XZX", "XXZ", "ZZX"]]

    @staticmethod
    def from_str(string: str) -> ZXCube:
        """A cube kind from its 3-letter string of X and Z (the walls' bases along x, y, z)."""
        return ZXCube(*map(Basis, string.upper()))

    @property
    def normal_basis(self) -> Basis:
        """The basis that appears once among the three."""
        if sum(basis == Basis.Z for basis in astuple(self)) == 1:
            return Basis.Z
        return Basis.X

    @property
    def normal_direction(self) -> Direction3D:
        """The direction along which the normal basis appears."""
        return Direction3D(astuple(self).index(self.normal_basis))

    @property
    def is_spatial(self) -> bool:
        """Whether all spatial boundaries are in the same basis (``XXZ`` and ``ZZX``)."""
        return self.x == self.y

    def get_basis_along(self, direction: Direction3D) -> Basis:
        """The basis of the walls along the given direction axis."""
        return self.as_tuple()[direction.value]

    def with_basis_along(self, direction: Direction3D, basis: Basis) -> ZXCube:
        """A new kind with the basis along the given direction axis set."""
        return ZXCube(*[basis if i == direction.value else b for i, b in enumerate(self.as_tuple())])


class Port:
    """Cube kind representing the open ports in the block graph: the inputs and outputs of the
    computation, invisible in a drawing."""

    def __str__(self) -> str:
        return "PORT"

    def __repr__(self) -> str:
        return "Port()"

    def __hash__(self) -> int:
        return hash(Port)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Port)


class YHalfCube:
    """Cube kind representing the Y-basis initialization/measurements."""

    def __str__(self) -> str:
        return "Y"

    def __repr__(self) -> str:
        return "YHalfCube()"

    def __hash__(self) -> int:
        return hash(YHalfCube)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, YHalfCube)


CubeKind = Union[ZXCube, Port, YHalfCube]
"""All the possible kinds of cubes."""


def cube_kind_from_string(s: str) -> CubeKind:
    """A cube kind from its string representation."""
    s = s.upper()
    if s in ("PORT", "P"):
        return Port()
    if s == "Y":
        return YHalfCube()
    return ZXCube.from_str(s)


@dataclass(frozen=True)
class Cube:
    """A block of operations in a spacetime volume: its position (x, y the patch, z the time),
    its kind, and a label (a port's must be non-empty and unique in its graph)."""

    position: Position3D
    kind: CubeKind
    label: str = ""

    def __post_init__(self) -> None:
        if self.is_port and not self.label:
            raise TQECError("A port cube must have a non-empty port label.")

    def __str__(self) -> str:
        return f"{self.kind}{self.position}"

    @property
    def is_zx_cube(self) -> bool:
        return isinstance(self.kind, ZXCube)

    @property
    def is_port(self) -> bool:
        return isinstance(self.kind, Port)

    @property
    def is_y_cube(self) -> bool:
        return isinstance(self.kind, YHalfCube)

    @property
    def is_spatial(self) -> bool:
        """Whether the cube's spatial boundaries are all in one basis (``XXZ``, ``ZZX``)."""
        return isinstance(self.kind, ZXCube) and self.kind.is_spatial

    def to_dict(self) -> dict[str, Any]:
        return {"position": self.position.as_tuple(), "kind": str(self.kind), "label": self.label}

    @staticmethod
    def from_dict(data: dict[str, Any]) -> Cube:
        return Cube(position=Position3D(*data["position"]), kind=cube_kind_from_string(data["kind"]), label=data["label"])

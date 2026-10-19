# Ported from Loom (entropicalabs/el-loom 0.4.0, src/loom/eka/{eka,lattice,pauli_operator,stabilizer,block}.py,
# src/loom/eka/operations/*.py, src/loom/eka/utilities/enums.py and src/loom_rotated_surface_code/utilities/enums.py;
# Copyright 2024 Entropica Labs Pte Ltd, Apache License 2.0; see LICENSE in this directory), modified for
# stabilizer_qec: plain dataclasses in place of pydantic (the first failed check raised, as a ValueError, with
# Loom's message), GF(2) eliminations in place of numpy, and the Tanner-graph and parity-check-matrix views left
# out.
"""Loom's Eka model: a lattice, blocks of stabilizers with their logical operators and syndrome
circuits, and the operations of a logical computation (code operations on blocks, logical
operations, logical measurements), with Loom's validation and its dict form."""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from enum import Enum
from itertools import chain, combinations, product
from typing import Any, ClassVar, Iterable, Optional, Sequence, Union
from uuid import uuid4

from .circuit import Channel, Circuit, SyndromeCircuit

Coord = tuple[int, ...]


# ---------------------------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------------------------


class _CaseInsensitive(str, Enum):
    @classmethod
    def _missing_(cls, value):
        if isinstance(value, str):
            for member in cls:
                if member.value.lower() == value.lower() or member.name.lower() == value.lower():
                    return member
        return None

    def __str__(self) -> str:
        return str(self.value)


class SingleQubitPauliEigenstate(_CaseInsensitive):
    ZERO = "0"
    ONE = "1"
    PLUS = "+"
    MINUS = "-"
    PLUS_I = "+i"
    MINUS_I = "-i"

    @property
    def pauli_basis(self) -> str:
        if self in (SingleQubitPauliEigenstate.ZERO, SingleQubitPauliEigenstate.ONE):
            return "Z"
        if self in (SingleQubitPauliEigenstate.PLUS, SingleQubitPauliEigenstate.MINUS):
            return "X"
        return "Y"

    @property
    def basis_expectation_value(self) -> int:
        return 1 if self in (SingleQubitPauliEigenstate.ZERO, SingleQubitPauliEigenstate.PLUS, SingleQubitPauliEigenstate.PLUS_I) else -1


class Orientation(_CaseInsensitive):
    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"

    @classmethod
    def from_direction(cls, direction: "Direction") -> "Orientation":
        return Orientation.HORIZONTAL if direction in (Direction.LEFT, Direction.RIGHT) else Orientation.VERTICAL

    @classmethod
    def from_vector(cls, vector: Sequence[int]) -> "Orientation":
        return cls.from_direction(Direction.from_vector(vector))

    def perpendicular(self) -> "Orientation":
        return Orientation.VERTICAL if self == Orientation.HORIZONTAL else Orientation.HORIZONTAL


class Direction(_CaseInsensitive):
    LEFT = "left"
    RIGHT = "right"
    TOP = "top"
    BOTTOM = "bottom"

    def to_vector(self) -> tuple[int, int]:
        return {"left": (-1, 0), "right": (1, 0), "top": (0, -1), "bottom": (0, 1)}[self.value]

    @classmethod
    def from_vector(cls, vector: Sequence[int], bottom_is_plus: bool = True) -> "Direction":
        x, y = int(vector[0]), int(vector[1])
        if x != 0 and y != 0:
            raise ValueError(f"Vector {vector} is not along an axis.")
        if x > 0:
            return Direction.RIGHT
        if x < 0:
            return Direction.LEFT
        if y > 0:
            return Direction.BOTTOM if bottom_is_plus else Direction.TOP
        if y < 0:
            return Direction.TOP if bottom_is_plus else Direction.BOTTOM
        raise ValueError("The zero vector has no direction.")

    def opposite(self) -> "Direction":
        return {"left": Direction.RIGHT, "right": Direction.LEFT, "top": Direction.BOTTOM, "bottom": Direction.TOP}[self.value]

    def to_orientation(self) -> Orientation:
        return Orientation.from_direction(self)

    def mirror_across_orientation(self, orientation: Orientation) -> "Direction":
        return self if self.to_orientation() == orientation else self.opposite()


class DiagonalDirection(_CaseInsensitive):
    TOP_LEFT = "top-left"
    TOP_RIGHT = "top-right"
    BOTTOM_LEFT = "bottom-left"
    BOTTOM_RIGHT = "bottom-right"

    @property
    def components(self) -> tuple[Direction, Direction]:
        vertical, horizontal = self.value.split("-")
        return (Direction(vertical), Direction(horizontal))

    @classmethod
    def from_directions(cls, directions: Iterable[Direction]) -> "DiagonalDirection":
        s = set(directions)
        for diag in cls:
            if set(diag.components) == s:
                return diag
        raise ValueError(f"{directions} are not the components of a diagonal direction.")

    def opposite(self) -> "DiagonalDirection":
        return DiagonalDirection.from_directions(d.opposite() for d in self.components)

    def __iter__(self):
        return iter(self.components)


class ResourceState(_CaseInsensitive):
    T = "t"
    S = "s"

    def get_expectation_value(self, basis: str) -> float:
        expectation = {"T": {"X": 1 / math.sqrt(2), "Y": 1 / math.sqrt(2), "Z": 0.0}, "S": {"X": 0.0, "Y": 1.0, "Z": 0.0}}[self.name]
        return float(expectation[basis.upper()])


class FourBodySchedule(_CaseInsensitive):
    """The order a weight-4 stabilizer's data qubits are entangled in: an N or a Z drawn
    through the plaquette (arXiv:1404.3747, III B)."""

    N = "N"
    Z = "Z"

    def opposite_schedule(self) -> "FourBodySchedule":
        return FourBodySchedule.Z if self == FourBodySchedule.N else FourBodySchedule.N


class LatticeType(_CaseInsensitive):
    CUSTOM = "custom"
    LINEAR = "linear"
    TRIANGLE_2D = "triangle_2d"
    SQUARE_2D_0ANC = "square_2d_0anc"
    SQUARE_2D = "square_2d"
    SQUARE_2D_2ANC = "square_2d_2anc"
    CAIRO_PENT_2D = "cairo_pent_2d"
    HEX_2D = "hex_2d"
    OCT_2D = "oct_2d"
    POLY_2D = "poly_2D"
    CUBE_3D = "cube_3d"


# ---------------------------------------------------------------------------------------------
# Lattice
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Lattice:
    """A Bravais lattice with a basis: qubit coordinates are ``(cell..., basis index)``."""

    basis_vectors: tuple[tuple[float, ...], ...]
    lattice_vectors: tuple[tuple[float, ...], ...]
    size: Optional[tuple[int, ...]] = None
    lattice_type: LatticeType = LatticeType.CUSTOM

    def __post_init__(self) -> None:
        # An empty basis is one site at the origin, as in Loom (the cube lattice's).
        basis_vectors = self.basis_vectors if len(self.basis_vectors) else ((0, 0),)
        object.__setattr__(self, "basis_vectors", tuple(tuple(float(x) for x in v) for v in basis_vectors))
        object.__setattr__(self, "lattice_vectors", tuple(tuple(float(x) for x in v) for v in self.lattice_vectors))
        if self.size is not None:
            object.__setattr__(self, "size", tuple(int(s) for s in self.size))
        object.__setattr__(self, "lattice_type", LatticeType(self.lattice_type))
        if len({len(v) for v in self.basis_vectors}) > 1:
            raise ValueError("All basis vectors must have the same length.")
        if len({len(v) for v in self.lattice_vectors}) > 1:
            raise ValueError("All lattice vectors must have the same length.")
        if self.size is not None and len(self.size) != len(self.lattice_vectors):
            raise ValueError(f"The given `size` is invalid. `size` has {len(self.size)} elements, but the lattice has {len(self.lattice_vectors)} dimensions.")
        if self.size is not None and any(s < 0 for s in self.size):
            raise ValueError("Size cannot be negative.")

    @property
    def n_dimensions(self) -> int:
        return len(self.lattice_vectors)

    @property
    def unit_cell_size(self) -> int:
        return len(self.basis_vectors)

    def all_unit_cells(self, size: Optional[Sequence[Optional[int]]] = None) -> list[tuple[int, ...]]:
        """The unit cells of the lattice, or of ``size`` (whose ``None`` entries take the
        lattice's size)."""
        if size is None and self.size is None:
            raise ValueError("Please specify the `size` parameter.")
        if self.size is None and any(x is None for x in size):
            raise ValueError("Please specify all dimensions in the `size` parameter.")
        if size is not None and len(size) != self.n_dimensions:
            raise ValueError(f"The given `size` is invalid. `size` has {len(size)} elements, but the lattice has {self.n_dimensions} dimensions.")
        adapted = self.size if size is None else [size[i] if size[i] is not None else self.size[i] for i in range(self.n_dimensions)]
        return [tuple(int(c) for c in cell) for cell in product(*(range(s) for s in adapted))]

    def all_qubits(self, size: Optional[Sequence[Optional[int]]] = None, force_including_basis: bool = False) -> list[tuple[int, ...]]:
        """The qubits ``cell + (basis index,)``; on a one-site lattice the cells alone, unless
        ``force_including_basis``."""
        cells = self.all_unit_cells(size)
        if self.unit_cell_size == 1 and force_including_basis is False:
            return cells
        return [cell + (b,) for cell in cells for b in range(self.unit_cell_size)]

    @staticmethod
    def _points_on_circle(n_points: int, radius: Union[float, int], offset: Union[float, int] = -0.5 * math.pi, disp: tuple = (0, 0)) -> list[list[float]]:
        if not isinstance(n_points, int) or n_points < 1:
            raise ValueError(f"Number of points must be an integer that is at least 1. Received {n_points} of type {type(n_points)}.")
        if not isinstance(radius, (int, float)) or radius <= 0:
            raise ValueError(f"Radius must be a positive number (int or float). Received {radius} of type {type(radius)}.")
        if not isinstance(offset, (int, float)):
            raise TypeError(f"Offset must be a number (int or float). Received {offset} of type {type(offset)}.")
        if not isinstance(disp, tuple) or not all(isinstance(x, (int, float)) for x in disp) or len(disp) != 2:
            raise TypeError(f"Disp must be a tuple of two numbers (x, y). Received {disp} of type {type(disp)}.")
        if n_points == 1:
            return [[disp[0], disp[1]]]
        return [[float(radius * math.cos(2 * math.pi * i / n_points + offset) + disp[0]), float(radius * math.sin(2 * math.pi * i / n_points + offset) + disp[1])] for i in range(n_points)]

    @classmethod
    def linear(cls, lattice_size: Optional[Union[tuple[int], int]] = None) -> "Lattice":
        """A chain: a data site and an ancilla half a cell to its right (in the plane, as in
        Loom). The size is a 1-tuple, ``(n,)``; a bare int is taken as ``(n,)``."""
        if isinstance(lattice_size, int):
            lattice_size = (lattice_size,)
        return cls(((0, 0), (0.5, 0)), ((1, 0),), lattice_size, LatticeType.LINEAR)

    @classmethod
    def triangle_2d(cls, lattice_size: Optional[tuple[int, int]] = None) -> "Lattice":
        return cls(((0, 0), (0.5, 1 / (2 * math.sqrt(3))), (1, 1 / math.sqrt(3))), ((1, 0), (0.5, 0.5 * math.sqrt(3))), lattice_size, LatticeType.TRIANGLE_2D)

    @classmethod
    def square_2d_0anc(cls, lattice_size: Optional[tuple[int, int]] = None) -> "Lattice":
        return cls(((0, 0),), ((1, 0), (0, 1)), lattice_size, LatticeType.SQUARE_2D_0ANC)

    @classmethod
    def square_2d(cls, lattice_size: Optional[tuple[int, int]] = None) -> "Lattice":
        """The square lattice with one data qubit (basis 0) and one ancilla (basis 1) per cell,
        the ancilla half a cell up and left — the rotated surface code's lattice."""
        return cls(((0, 0), (-0.5, -0.5)), ((1, 0), (0, 1)), lattice_size, LatticeType.SQUARE_2D)

    @classmethod
    def square_2d_2anc(cls, lattice_size: Optional[tuple[int, int]] = None, shift: float = 0.15) -> "Lattice":
        return cls(((0, 0), (0, -shift), (0, shift)), ((1, 0), (0, 1)), lattice_size, LatticeType.SQUARE_2D_2ANC)

    @classmethod
    def hex_2d(cls, lattice_size: Optional[tuple[int, int]] = None) -> "Lattice":
        return cls(((0, 0), (0, 1 / math.sqrt(3))), ((1, 0), (0.5, 0.5 * math.sqrt(3))), lattice_size, LatticeType.HEX_2D)

    @classmethod
    def cairo_pent_2d(cls, lattice_size: Optional[tuple[int, int]] = None) -> "Lattice":
        r3 = math.sqrt(3)
        basis = ((0, 0), (0.5 * r3, 0.5), (0.5 * (r3 - 1), 0.5 * (1 + r3)), (0.5 * (1 - r3), 0.5 * (1 + r3)), (-0.5 * r3, 0.5), (-r3, 1))
        return cls(basis, ((2 * r3, 0), (r3, r3)), lattice_size, LatticeType.CAIRO_PENT_2D)

    @classmethod
    def oct_2d(cls, lattice_size: Optional[tuple[int, int]] = None, r: float = 0.5, anc: int = 0) -> "Lattice":
        """Octagons of radius ``r`` with squares between them; ``anc`` ancillas on a small
        circle at the octagon's centre and at the three square centres of the cell."""
        if not isinstance(anc, int) or anc < 0:
            raise ValueError("Number of ancilla qubits must be a non-negative integer.")
        anc_radius = r * math.sin(math.pi / 8) / 2
        cell_dist = 2 * r * (math.sin(math.pi / 8) + math.cos(math.pi / 8))
        basis = cls._points_on_circle(8, r, -5 * math.pi / 8)
        if anc > 0:
            basis = basis + cls._points_on_circle(anc, anc_radius) + cls._points_on_circle(anc, anc_radius, disp=(0, 0.5 * cell_dist)) + cls._points_on_circle(anc, anc_radius, disp=(0.5 * cell_dist, 0)) + cls._points_on_circle(anc, anc_radius, disp=(0.5 * cell_dist, 0.5 * cell_dist))
        return cls(basis, ((0, cell_dist), (cell_dist, 0)), lattice_size, LatticeType.OCT_2D)

    @classmethod
    def poly_2d(cls, lattice_size: Optional[tuple[int, int]] = None, n: Optional[int] = None, poly_radius: Union[float, int] = 0.5, poly_offset: Union[float, int] = -0.5 * math.pi, cell_dist_factor: Union[float, int] = (1 + math.sqrt(3)), anc: int = 1, anc_radius: Union[float, int] = 0.2, anc_offset: Union[float, int] = -0.5 * math.pi, anc_disp: tuple = (0, 0)) -> "Lattice":
        """A regular ``n``-gon of data sites per cell, with ``anc`` ancillas on a circle inside
        it (the five-qubit code's lattice with ``n=5``)."""
        if n is None:
            raise ValueError("Please specify the number of sides `n` for the polygon.")
        if not isinstance(n, int) or n < 3:
            raise ValueError("Number of sides must be an integer that is at least 3.")
        if not isinstance(anc, int) or anc < 0:
            raise ValueError("Number of ancilla qubits must be a non-negative integer.")
        basis = cls._points_on_circle(n, poly_radius, poly_offset)
        if anc > 0:
            basis = basis + cls._points_on_circle(anc, anc_radius, anc_offset, anc_disp)
        cell_dist = poly_radius * cell_dist_factor
        return cls(basis, ((0, cell_dist), (cell_dist, 0)), lattice_size, LatticeType.POLY_2D)

    @classmethod
    def cube_3d(cls, lattice_size: Optional[tuple[int, int, int]] = None) -> "Lattice":
        """The cubic lattice, one site per cell."""
        return cls((), ((1, 0, 0), (0, 1, 0), (0, 0, 1)), lattice_size, LatticeType.CUBE_3D)

    def asdict(self) -> dict:
        return {"basis_vectors": [list(v) for v in self.basis_vectors], "lattice_vectors": [list(v) for v in self.lattice_vectors], "size": None if self.size is None else list(self.size), "lattice_type": self.lattice_type.value}

    @classmethod
    def fromdict(cls, data: dict) -> "Lattice":
        return cls(data["basis_vectors"], data["lattice_vectors"], data.get("size"), LatticeType(data.get("lattice_type", "custom")))


# ---------------------------------------------------------------------------------------------
# Pauli operators and stabilizers
# ---------------------------------------------------------------------------------------------


def _coords(qubits: Iterable) -> tuple[Coord, ...]:
    return tuple(tuple(int(c) for c in q) for q in qubits)


def _checked_coords(qubits: Iterable) -> tuple[Coord, ...]:
    """Loom's checks on a list of coordinates, in its order: one length, then distinct."""
    qubits = [tuple(q) for q in qubits]
    if qubits and any(len(q) != len(qubits[0]) for q in qubits):
        raise ValueError("Length of coordinates must be consistent.")
    return _coords(qubits)


def _distinct(qubits: tuple[Coord, ...]) -> tuple[Coord, ...]:
    if len(qubits) != len(set(qubits)):
        raise ValueError(f"List of {type(qubits[0])} must have distinct elements.")
    return qubits


class PauliOperator:
    """A Pauli string on data qubits named by their lattice coordinates. Equality and hashing
    ignore the uuid and the order of the qubits. As in Loom, the letters are upper-case X, Y
    and Z."""

    __slots__ = ("pauli", "data_qubits", "uuid")

    def __init__(self, pauli: str, data_qubits: Iterable, uuid: Optional[str] = None) -> None:
        pauli = str(pauli)
        for p in pauli:
            if p not in ("X", "Y", "Z"):
                raise ValueError(f"Invalid pauli: {p}. Must be one of ['X', 'Y', 'Z'].")
        data_qubits = _checked_coords(data_qubits)
        if len(data_qubits) != len(pauli):
            raise ValueError(f"Number of qubits: {len(data_qubits)} does not match the length of the pauli string: {len(pauli)}.")
        _distinct(data_qubits)
        self.pauli = pauli
        self.data_qubits = data_qubits
        self.uuid = uuid or str(uuid4())

    def __str__(self) -> str:
        return " ".join(f"{p}_{q}" for p, q in zip(self.pauli, self.data_qubits))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PauliOperator):
            return NotImplemented
        return dict(zip(self.data_qubits, self.pauli)) == dict(zip(other.data_qubits, other.pauli))

    def __hash__(self) -> int:
        return hash(frozenset(zip(self.data_qubits, self.pauli)))

    @property
    def weight(self) -> int:
        return len(self.data_qubits)

    @property
    def pauli_type(self) -> str:
        """``X``, ``Y`` or ``Z`` when one letter, else ``mixed``."""
        letters = set(self.pauli)
        return next(iter(letters)) if len(letters) == 1 else "mixed"

    def pauli_at(self, qubit: Coord) -> str:
        return self.pauli[self.data_qubits.index(tuple(qubit))] if tuple(qubit) in self.data_qubits else "I"

    def commutes_with(self, other: "PauliOperator") -> bool:
        if not isinstance(other, PauliOperator):
            raise ValueError(f"Expected PauliOperator object, got {type(other)} instead.")
        mine = dict(zip(self.data_qubits, self.pauli))
        theirs = dict(zip(other.data_qubits, other.pauli))
        anticommuting = sum(1 for q in mine.keys() & theirs.keys() if mine[q] != theirs[q])
        return anticommuting % 2 == 0

    def as_xz(self, qubits: Sequence[Coord]) -> tuple[int, int]:
        """X and Z bit masks over an ordered list of qubits (bit i for ``qubits[i]``)."""
        x = z = 0
        index = {q: i for i, q in enumerate(qubits)}
        for p, q in zip(self.pauli, self.data_qubits):
            i = index[q]
            if p in "XY":
                x |= 1 << i
            if p in "ZY":
                z |= 1 << i
        return x, z

    def asdict(self) -> dict:
        return {"pauli": self.pauli, "data_qubits": [list(q) for q in self.data_qubits], "uuid": self.uuid}

    @classmethod
    def fromdict(cls, data: dict) -> "PauliOperator":
        return cls(data["pauli"], data["data_qubits"], data.get("uuid"))


class Stabilizer(PauliOperator):
    """A stabilizer: a Pauli string on data qubits, measured with ancilla qubits."""

    __slots__ = ("ancilla_qubits",)

    def __init__(self, pauli: str, data_qubits: Iterable, uuid: Optional[str] = None, ancilla_qubits: Iterable = (), nr_of_ancillae: Optional[int] = None) -> None:
        super().__init__(pauli, data_qubits, uuid)
        ancilla_qubits = _checked_coords(ancilla_qubits)
        if nr_of_ancillae and not ancilla_qubits:
            # Workbench's convention: the first data qubit's cell, basis 1, 2, ...
            ancilla_qubits = tuple((self.data_qubits[0][0], self.data_qubits[0][1], k + 1) for k in range(nr_of_ancillae))
        self.ancilla_qubits = _distinct(ancilla_qubits)

    def __repr__(self) -> str:
        return f"{self.pauli} {self.data_qubits} {self.ancilla_qubits}"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Stabilizer):
            return NotImplemented
        return self.pauli == other.pauli and self.data_qubits == other.data_qubits and self.ancilla_qubits == other.ancilla_qubits

    def __hash__(self) -> int:
        return hash((self.pauli, self.data_qubits, self.ancilla_qubits))

    def asdict(self) -> dict:
        d = super().asdict()
        d["ancilla_qubits"] = [list(q) for q in self.ancilla_qubits]
        return d

    @classmethod
    def fromdict(cls, data: dict) -> "Stabilizer":
        return cls(data["pauli"], data["data_qubits"], data.get("uuid"), data.get("ancilla_qubits", ()), data.get("nr_of_ancillae"))


def gf2_rank(rows: Iterable[int]) -> int:
    """The rank of binary vectors held as ints."""
    basis: dict[int, int] = {}
    rank = 0
    for v in rows:
        while v:
            h = v.bit_length() - 1
            if h in basis:
                v ^= basis[h]
            else:
                basis[h] = v
                rank += 1
                break
    return rank


def _xz_rows(operators: Iterable[PauliOperator], qubits: Sequence[Coord]) -> list[int]:
    n = len(qubits)
    out = []
    for op in operators:
        x, z = op.as_xz(qubits)
        out.append(x | (z << n))
    return out


# ---------------------------------------------------------------------------------------------
# Blocks
# ---------------------------------------------------------------------------------------------


class Block:
    """A code block: its stabilizers (with the ancillas that measure them), its logical X and
    Z operators, the syndrome circuits (one per Pauli string by default) and which circuit
    measures which stabilizer. Validation checks distinctness, commutation, that the logicals
    anticommute pairwise and only pairwise, and that the independent stabilizers plus the
    logical qubits make up the data qubits."""

    __version__ = "1.0.0"

    def __init__(self, stabilizers: Iterable[Stabilizer], logical_x_operators: Iterable[PauliOperator], logical_z_operators: Iterable[PauliOperator], syndrome_circuits: Iterable[SyndromeCircuit] = (), stabilizer_to_circuit: Optional[dict[str, str]] = None, unique_label: Optional[str] = None, skip_validation: bool = False, uuid: Optional[str] = None) -> None:
        stabilizers = tuple(s if isinstance(s, Stabilizer) else Stabilizer.fromdict(s) for s in stabilizers)
        logical_x_operators = tuple(o if isinstance(o, PauliOperator) else PauliOperator.fromdict({k: v for k, v in o.items() if k != "nr_of_ancillae"}) for o in logical_x_operators)
        logical_z_operators = tuple(o if isinstance(o, PauliOperator) else PauliOperator.fromdict({k: v for k, v in o.items() if k != "nr_of_ancillae"}) for o in logical_z_operators)
        syndrome_circuits = tuple(c if isinstance(c, SyndromeCircuit) else SyndromeCircuit.fromdict(c) for c in syndrome_circuits)
        stabilizer_to_circuit = dict(stabilizer_to_circuit or {})
        for given in (stabilizers, logical_x_operators, logical_z_operators):
            if len(given) == 0:
                raise ValueError("List cannot be empty.")
        if len(logical_x_operators) != len(logical_z_operators):
            raise ValueError("The number of logical X operators must be equal to the number of logical Z operators.")
        # Built as Loom builds it, so a set of several qubits prints in Loom's order.
        outside = set(q for o in logical_x_operators + logical_z_operators for q in o.data_qubits) - set(q for s in stabilizers for q in s.data_qubits)
        if outside:
            raise ValueError(f"Qubits {outside} are not included in the stabilizers but are used in the logical operators")
        for a, b in combinations(stabilizers, 2):
            if a == b:
                raise ValueError("Stabilizers must be distinct.")
        for ops, what in ((logical_x_operators, "X"), (logical_z_operators, "Z")):
            for a, b in combinations(ops, 2):
                if a == b:
                    raise ValueError(f"Logical {what} operators must be distinct.")
        # A default syndrome circuit for every Pauli string without one.
        stab_paulis = {s.pauli for s in stabilizers}
        have = {c.pauli for c in syndrome_circuits}
        syndrome_circuits = syndrome_circuits + tuple(SyndromeCircuit(name=f"default_{p}", pauli=p) for p in stab_paulis - have)
        circuits_by_uuid = {c.uuid: c for c in syndrome_circuits}
        stabs_by_uuid = {s.uuid: s for s in stabilizers}
        for stab_uuid, circ_uuid in stabilizer_to_circuit.items():
            if stab_uuid not in stabs_by_uuid:
                raise ValueError(f"Stabilizer with uuid {stab_uuid} is not present in the stabilizers.")
            if circ_uuid not in circuits_by_uuid:
                raise ValueError(f"Syndrome circuit with uuid {circ_uuid} is not present in the syndrome circuits.")
            if stabs_by_uuid[stab_uuid].pauli != circuits_by_uuid[circ_uuid].pauli:
                raise ValueError(f"Stabilizer with uuid {stab_uuid} has a different pauli string {stabs_by_uuid[stab_uuid].pauli} than the syndrome circuit with uuid {circ_uuid} (pauli string: {circuits_by_uuid[circ_uuid].pauli}).")
        for stab in stabilizers:
            if stab.uuid not in stabilizer_to_circuit:
                matching = [c for c in syndrome_circuits if c.pauli == stab.pauli]
                if len(matching) > 1:
                    raise ValueError(f"Multiple syndrome circuits for the same stabilizer pauli string {stab.pauli} found. Could not automatically associate stabilizer {stab} with a syndrome circuit. Please do the association manually.")
                stabilizer_to_circuit[stab.uuid] = matching[0].uuid
        self.stabilizers = stabilizers
        self.logical_x_operators = logical_x_operators
        self.logical_z_operators = logical_z_operators
        self.syndrome_circuits = syndrome_circuits
        self.stabilizer_to_circuit = stabilizer_to_circuit
        self.unique_label = unique_label if unique_label is not None else str(uuid4())
        self.skip_validation = skip_validation
        self.uuid = uuid or str(uuid4())
        self._cache: dict[str, Any] = {}
        if not skip_validation:
            self._validate()

    def _validate(self) -> None:
        dimension = len(self.qubits[0])
        if any(len(q) != dimension for q in self.qubits):
            raise ValueError("All qubits coordinates must have the same dimension.")
        for a, b in combinations(self.stabilizers, 2):
            if not a.commutes_with(b):
                raise ValueError(f"Stabilizers must commute with each other:\n{a} and\n{b} do not commute.\n")
        for ops, what in ((self.logical_x_operators, "X"), (self.logical_z_operators, "Z")):
            for a, b in combinations(ops, 2):
                if not a.commutes_with(b):
                    raise ValueError(f"Logical {what} operators must commute with each other:\n{a} and\n{b} do not commute.\n")
        for stab, op in product(self.stabilizers, self.logical_x_operators):
            if not stab.commutes_with(op):
                raise ValueError(f"Stabilizers must commute with logical X operators:\n{stab!r} and\n{op!r} do not commute.\n")
        for stab, op in product(self.stabilizers, self.logical_z_operators):
            if not stab.commutes_with(op):
                raise ValueError(f"Stabilizers must commute with logical Z operators:\n{stab!r} and\n{op!r} do not commute.\n")
        for (i, x), (j, z) in combinations(list(chain(enumerate(self.logical_x_operators), enumerate(self.logical_z_operators))), 2):
            if i == j and x.commutes_with(z):
                raise ValueError(f"Logical X and Z operators at the same index must anticommute with each other:\n{x} and\n{z} do not anticommute.\n")
            if i != j and not x.commutes_with(z):
                raise ValueError(f"Logical X and Z operators at different indices must commute with each other:\n{x} and\n {z} do not commute.\n")
        if self.n_irreducible_stabs + len(self.logical_x_operators) != len(self.data_qubits):
            raise ValueError("The number of qubits and independent stabilizers in the Block is not compatible with the number of logical qubits.")

    # Qubits
    @property
    def data_qubits(self) -> tuple[Coord, ...]:
        return tuple(sorted({q for s in self.stabilizers for q in s.data_qubits}))

    @property
    def ancilla_qubits(self) -> tuple[Coord, ...]:
        seen: dict[Coord, None] = {}
        for s in self.stabilizers:
            for q in s.ancilla_qubits:
                seen.setdefault(q, None)
        return tuple(seen)

    @property
    def qubits(self) -> tuple[Coord, ...]:
        return tuple(dict.fromkeys(self.data_qubits + self.ancilla_qubits))

    @property
    def n_data_qubits(self) -> int:
        return len(self.data_qubits)

    @property
    def n_logical_qubits(self) -> int:
        return len(self.logical_x_operators)

    @property
    def n_irreducible_stabs(self) -> int:
        return gf2_rank(_xz_rows(self.stabilizers, self.data_qubits))

    @property
    def pauli_charges(self) -> dict[Coord, str]:
        """Each data qubit's Pauli charge: the product of the Paulis the stabilizers put on it
        (``_`` none, ``X``, ``Y``, ``Z``); a rotated surface code's corners are Y, its
        boundaries X or Z, its bulk uncharged."""
        if "charges" not in self._cache:
            counts = {q: {"X": 0, "Y": 0, "Z": 0} for q in self.data_qubits}
            for stab in self.stabilizers:
                for p, q in zip(stab.pauli, stab.data_qubits):
                    counts[q][p] += 1
            self._cache["charges"] = {q: "_XZY"[((c["X"] + c["Y"]) % 2) | (((c["Z"] + c["Y"]) % 2) << 1)] for q, c in counts.items()}
        return dict(self._cache["charges"])

    # Equality and construction
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Block):
            return NotImplemented
        if self.unique_label != other.unique_label or set(self.stabilizers) != set(other.stabilizers) or self.logical_x_operators != other.logical_x_operators or self.logical_z_operators != other.logical_z_operators:
            return False
        if sorted(self.syndrome_circuits, key=lambda c: c.name) != sorted(other.syndrome_circuits, key=lambda c: c.name):
            return False
        mine = {c.uuid: c for c in self.syndrome_circuits}
        theirs = {c.uuid: c for c in other.syndrome_circuits}
        for stab in self.stabilizers:
            other_stab = next(s for s in other.stabilizers if s == stab)
            if mine[self.stabilizer_to_circuit[stab.uuid]] != theirs[other.stabilizer_to_circuit[other_stab.uuid]]:
                return False
        return True

    def __hash__(self) -> int:
        return hash(self.uuid)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.unique_label!r}, {len(self.stabilizers)} stabilizers, {self.n_logical_qubits} logical qubit(s))"

    @classmethod
    def from_blocks(cls, blocks: Sequence["Block"]) -> "Block":
        """Several disjoint blocks as one (label ``a+b``), without validation."""
        Block.blocks_no_overlap(blocks)
        stabilizers = tuple(s for b in blocks for s in b.stabilizers)
        xs = tuple(o for b in blocks for o in b.logical_x_operators)
        zs = tuple(o for b in blocks for o in b.logical_z_operators)
        circuits: list[SyndromeCircuit] = []
        remap: dict[str, str] = {}
        for b in blocks:
            for c in b.syndrome_circuits:
                kept = next((k for k in circuits if k == c), None)
                if kept is None:
                    circuits.append(c)
                    kept = c
                remap[c.uuid] = kept.uuid
        stc = {s: remap[c] for b in blocks for s, c in b.stabilizer_to_circuit.items()}
        return Block(stabilizers, xs, zs, circuits, stc, unique_label="+".join(b.unique_label for b in blocks), skip_validation=True)

    @staticmethod
    def blocks_no_overlap(blocks: Sequence["Block"]) -> Sequence["Block"]:
        for a, b in combinations(blocks, 2):
            shared = set(a.data_qubits) & set(b.data_qubits)
            if shared:
                raise ValueError(f"Block '{a.unique_label}' and block '{b.unique_label}' share the data qubits {shared}.")
            shared = set(a.ancilla_qubits) & set(b.ancilla_qubits)
            if shared:
                raise ValueError(f"Block '{a.unique_label}' and block '{b.unique_label}' share the ancilla qubits {shared}.")
        return blocks

    def rename(self, name: str) -> "Block":
        """A copy under a new label (and a new uuid)."""
        return self.__class__(unique_label=name, stabilizers=self.stabilizers, logical_x_operators=self.logical_x_operators, logical_z_operators=self.logical_z_operators, syndrome_circuits=self.syndrome_circuits, stabilizer_to_circuit=self.stabilizer_to_circuit)

    def shift(self, position: Sequence[int], new_label: Optional[str] = None) -> "Block":
        """A copy with every qubit's cell coordinates shifted by ``position``."""
        dim = len(self.logical_x_operators[0].data_qubits[0]) - 1
        if len(position) != dim:
            raise ValueError(f"The shift position has a wrong dimension of {len(position)}. Expected {dim} instead.")

        def move(q: Coord) -> Coord:
            return tuple(c + s for c, s in zip(q, position)) + q[dim:]

        new_stabs = [Stabilizer(s.pauli, [move(q) for q in s.data_qubits], ancilla_qubits=[move(q) for q in s.ancilla_qubits]) for s in self.stabilizers]
        stc = {new.uuid: self.stabilizer_to_circuit[old.uuid] for old, new in zip(self.stabilizers, new_stabs)}
        return self.__class__(
            stabilizers=new_stabs,
            logical_x_operators=[PauliOperator(o.pauli, [move(q) for q in o.data_qubits]) for o in self.logical_x_operators],
            logical_z_operators=[PauliOperator(o.pauli, [move(q) for q in o.data_qubits]) for o in self.logical_z_operators],
            syndrome_circuits=self.syndrome_circuits,
            stabilizer_to_circuit=stc,
            unique_label=self.unique_label if new_label is None else new_label,
            skip_validation=self.skip_validation,
        )

    @property
    def stabilizers_labels(self) -> dict[str, dict[str, Any]]:
        """Each stabilizer's labels: its space coordinates (the first ancilla's)."""
        return {s.uuid: {"space_coordinates": s.ancilla_qubits[0]} for s in self.stabilizers if s.ancilla_qubits}

    def get_stabilizer_label(self, stabilizer_uuid: str) -> dict[str, Any]:
        return self.stabilizers_labels.get(stabilizer_uuid, {})

    def asdict(self) -> dict:
        return {
            "stabilizers": [s.asdict() for s in self.stabilizers],
            "logical_x_operators": [o.asdict() for o in self.logical_x_operators],
            "logical_z_operators": [o.asdict() for o in self.logical_z_operators],
            "syndrome_circuits": [c.asdict() for c in self.syndrome_circuits],
            "stabilizer_to_circuit": dict(self.stabilizer_to_circuit),
            "unique_label": self.unique_label,
            "skip_validation": self.skip_validation,
            "uuid": self.uuid,
            "__version__": self.__version__,
        }

    @classmethod
    def fromdict(cls, data: dict) -> "Block":
        if "__version__" in data and str(data["__version__"])[0] != cls.__version__[0]:
            raise ValueError("The major version of the Workbench export is not compatible with the major version of Eka.Block.")
        return cls(stabilizers=data["stabilizers"], logical_x_operators=data["logical_x_operators"], logical_z_operators=data["logical_z_operators"], syndrome_circuits=data.get("syndrome_circuits", ()), stabilizer_to_circuit=data.get("stabilizer_to_circuit"), unique_label=data.get("unique_label"), skip_validation=data.get("skip_validation", False), uuid=data.get("uuid"))


# ---------------------------------------------------------------------------------------------
# Operations
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class LogicalMeasurement:
    """A logical measurement named by its blocks and Pauli product (``("a", "b"), "ZZ"``): the
    key under which the interpreter files the outcome's classical bits."""

    blocks: tuple[str, ...]
    observable: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "blocks", tuple(self.blocks))
        if len(self.blocks) != len(self.observable):
            raise ValueError("One Pauli letter per block.")

    @property
    def _as_pairs(self) -> frozenset:
        return frozenset(zip(self.blocks, self.observable))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, LogicalMeasurement) and self._as_pairs == other._as_pairs

    def __hash__(self) -> int:
        return hash(self._as_pairs)

    def __repr__(self) -> str:
        return f"LogicalMeasurement(blocks=({', '.join(self.blocks)}), observable='{self.observable}')"


_OPERATIONS: dict[str, type] = {}


@dataclass(frozen=True)
class Operation:
    """Base of every operation; subclasses are dataclasses, found by name for ``fromdict``."""

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        _OPERATIONS[cls.__name__] = cls

    def asdict(self) -> dict:
        d = dataclasses.asdict(self)
        for k, v in list(d.items()):
            if isinstance(v, Enum):
                d[k] = v.value
        d["__name__"] = self.__class__.__name__
        return d

    @classmethod
    def fromdict(cls, data: dict) -> "Operation":
        data = dict(data)
        name = data.pop("__name__", None)
        if not name:
            raise ValueError("No Operation name found in the data, the Operation cannot be loaded.")
        if name == cls.__name__:
            return cls(**data)
        klass = _OPERATIONS.get(name)
        if klass is None:
            raise ValueError(f"Operation {name} was not found in the Operation subclasses.")
        if "condition" in data and isinstance(data["condition"], dict):
            data["condition"] = LogicalMeasurement(**data["condition"])
        if "stabilizer" in data and isinstance(data["stabilizer"], dict):
            data["stabilizer"] = Stabilizer.fromdict(data["stabilizer"])
        if "observable" in data and isinstance(data["observable"], dict):
            data["observable"] = PauliOperator.fromdict(data["observable"])
        return klass(**data)


@dataclass(frozen=True)
class BaseOperation(Operation):
    pass


@dataclass(frozen=True)
class MeasureStabilizerSyndrome(BaseOperation):
    stabilizer: Stabilizer


@dataclass(frozen=True)
class MeasureObservable(BaseOperation):
    observable: PauliOperator


@dataclass(frozen=True)
class CodeOperation(Operation):
    """An operation on blocks; ``_inputs`` and ``_outputs`` name them."""

    @property
    def _inputs(self) -> tuple[str, ...]:
        if hasattr(self, "input_block_name"):
            return (getattr(self, "input_block_name"),)
        if hasattr(self, "input_blocks_name"):
            return tuple(getattr(self, "input_blocks_name"))
        raise ValueError(f"No block inputs specified for {self.__class__.__name__}")

    @property
    def _outputs(self) -> tuple[str, ...]:
        if hasattr(self, "output_block_name"):
            return (getattr(self, "output_block_name"),)
        if hasattr(self, "output_blocks_name"):
            return tuple(getattr(self, "output_blocks_name"))
        return self._inputs


def _positive(name: str, value: int) -> None:
    if not isinstance(value, int):
        raise ValueError(f"{name} must be an integer, not {value!r}.")
    if value <= 0:
        raise ValueError(f"{name} has to be larger than 0.")


def _pair(name: str, value) -> tuple:
    # Loom types these fields tuple[str, str]: exactly two names.
    value = tuple(value) if not isinstance(value, str) else (value,)
    if len(value) != 2:
        raise ValueError(f"{name} must be two block names, not {len(value)}: {value!r}.")
    return value


@dataclass(frozen=True)
class MeasureBlockSyndromes(CodeOperation):
    input_block_name: str
    n_cycles: int = 1


@dataclass(frozen=True)
class MeasureLogicalX(CodeOperation):
    input_block_name: str
    logical_qubit: int = 0


@dataclass(frozen=True)
class MeasureLogicalY(CodeOperation):
    input_block_name: str
    logical_qubit: int = 0


@dataclass(frozen=True)
class MeasureLogicalZ(CodeOperation):
    input_block_name: str
    logical_qubit: int = 0


@dataclass(frozen=True)
class LogicalX(CodeOperation):
    input_block_name: str
    logical_qubit: int = 0


@dataclass(frozen=True)
class LogicalY(CodeOperation):
    input_block_name: str
    logical_qubit: int = 0


@dataclass(frozen=True)
class LogicalZ(CodeOperation):
    input_block_name: str
    logical_qubit: int = 0


@dataclass(frozen=True)
class ResetAllDataQubits(CodeOperation):
    input_block_name: str
    state: SingleQubitPauliEigenstate = SingleQubitPauliEigenstate.ZERO

    def __post_init__(self) -> None:
        object.__setattr__(self, "state", SingleQubitPauliEigenstate(self.state))


@dataclass(frozen=True)
class ResetAllAncillaQubits(CodeOperation):
    input_block_name: str
    state: SingleQubitPauliEigenstate = SingleQubitPauliEigenstate.ZERO

    def __post_init__(self) -> None:
        object.__setattr__(self, "state", SingleQubitPauliEigenstate(self.state))


@dataclass(frozen=True)
class Grow(CodeOperation):
    input_block_name: str
    direction: Direction
    length: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "direction", Direction(self.direction))
        _positive("length", self.length)


@dataclass(frozen=True)
class Shrink(CodeOperation):
    input_block_name: str
    direction: Direction
    length: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "direction", Direction(self.direction))
        _positive("length", self.length)


@dataclass(frozen=True)
class Merge(CodeOperation):
    input_blocks_name: tuple[str, str]
    output_block_name: str
    orientation: Optional[Orientation] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "input_blocks_name", _pair("input_blocks_name", self.input_blocks_name))
        if self.orientation is not None:
            object.__setattr__(self, "orientation", Orientation(self.orientation))


@dataclass(frozen=True)
class Split(CodeOperation):
    input_block_name: str
    output_blocks_name: tuple[str, str]
    orientation: Orientation
    split_position: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_blocks_name", _pair("output_blocks_name", self.output_blocks_name))
        object.__setattr__(self, "orientation", Orientation(self.orientation))
        _positive("split_position", self.split_position)


@dataclass(frozen=True)
class StateInjection(CodeOperation):
    input_block_name: str
    resource_state: ResourceState

    def __post_init__(self) -> None:
        object.__setattr__(self, "resource_state", ResourceState(self.resource_state))


@dataclass(frozen=True)
class ConditionalLogicalX(CodeOperation):
    input_block_name: str
    condition: LogicalMeasurement
    logical_qubit: int = 0


@dataclass(frozen=True)
class ConditionalLogicalY(CodeOperation):
    input_block_name: str
    condition: LogicalMeasurement
    logical_qubit: int = 0


@dataclass(frozen=True)
class ConditionalLogicalZ(CodeOperation):
    input_block_name: str
    condition: LogicalMeasurement
    logical_qubit: int = 0


# The rotated surface code's own operations (loom_rotated_surface_code.operations).
@dataclass(frozen=True)
class AuxCNOT(CodeOperation):
    input_blocks_name: tuple[str, str]


@dataclass(frozen=True)
class TransversalHadamard(CodeOperation):
    input_block_name: str


@dataclass(frozen=True)
class MoveBlock(CodeOperation):
    input_block_name: str
    direction: Direction

    def __post_init__(self) -> None:
        object.__setattr__(self, "direction", Direction(self.direction))


@dataclass(frozen=True)
class LogicalPhaseViaYwall(CodeOperation):
    input_block_name: str
    growth_direction: Direction

    def __post_init__(self) -> None:
        object.__setattr__(self, "growth_direction", Direction(self.growth_direction))


@dataclass(frozen=True)
class RotateBlock(CodeOperation):
    input_block_name: str
    grow_direction: Direction

    def __post_init__(self) -> None:
        object.__setattr__(self, "grow_direction", Direction(self.grow_direction))


@dataclass(frozen=True)
class LogicalOperation(Operation):
    pass


@dataclass(frozen=True)
class Reset(LogicalOperation):
    target_qubit: str
    state: SingleQubitPauliEigenstate


@dataclass(frozen=True)
class CNOT(LogicalOperation):
    control_qubit: str
    target_qubit: str


@dataclass(frozen=True)
class Hadamard(LogicalOperation):
    target_qubit: str


@dataclass(frozen=True)
class Phase(LogicalOperation):
    target_qubit: str


@dataclass(frozen=True)
class PhaseInverse(LogicalOperation):
    target_qubit: str


@dataclass(frozen=True)
class X(LogicalOperation):
    target_qubit: str


@dataclass(frozen=True)
class Y(LogicalOperation):
    target_qubit: str


@dataclass(frozen=True)
class Z(LogicalOperation):
    target_qubit: str


@dataclass(frozen=True)
class T(LogicalOperation):
    target_qubit: str


# ---------------------------------------------------------------------------------------------
# Eka
# ---------------------------------------------------------------------------------------------


class Eka:
    """A logical computation: the lattice, the blocks at the start, and the operations, each
    time step a tuple of operations on disjoint blocks (a flat list runs them one by one)."""

    def __init__(self, lattice: Lattice, blocks: Iterable[Block] = (), operations: Iterable = ()) -> None:
        self.lattice = lattice
        blocks = tuple(blocks)
        for block in blocks:
            if any(any(c < 0 for c in q) for q in block.data_qubits):
                raise ValueError(f"Block '{block.unique_label}' has negative data qubit indices.")
            if any(any(c < 0 for c in q) for q in block.ancilla_qubits):
                raise ValueError(f"Block '{block.unique_label}' has negative ancilla qubit indices.")
            if lattice.size is not None:
                limits = list(lattice.size) + [len(lattice.basis_vectors)]
                for q in block.data_qubits:
                    if any(c >= limits[i] for i, c in enumerate(q)):
                        raise ValueError(f"Block '{block.unique_label}' has data qubit indices which are too large for the lattice.")
                for q in block.ancilla_qubits:
                    if any(c >= limits[i] for i, c in enumerate(q)):
                        raise ValueError(f"Block '{block.unique_label}' has ancilla qubit indices which are too large for the lattice.")
        Block.blocks_no_overlap(blocks)
        if len({b.unique_label for b in blocks}) != len(blocks):
            raise ValueError("Not all blocks have unique labels.")
        self.blocks = blocks
        operations = tuple(operations)
        if all(isinstance(op, Operation) for op in operations):
            operations = tuple((op,) for op in operations)
        else:
            operations = tuple(tuple(step) if not isinstance(step, Operation) else (step,) for step in operations)
        for step in operations:
            used: set[str] = set()
            for op in step:
                inputs = set(getattr(op, "_inputs", ()))
                if used & inputs:
                    raise ValueError(f"Operations are not disjoint, {used & inputs} is subject to multiple operations at the same time.")
                used |= inputs
        self.operations = operations

    def asdict(self) -> dict:
        return {"lattice": self.lattice.asdict(), "blocks": [b.asdict() for b in self.blocks], "operations": [[op.asdict() for op in step] for step in self.operations]}

    @classmethod
    def fromdict(cls, data: dict, block_class: Optional[type] = None) -> "Eka":
        """An Eka from its dict (Loom's ``asdict``); ``block_class`` builds the blocks
        (``RotatedSurfaceCode`` for Loom's surface-code experiments)."""
        klass = block_class or Block
        blocks = [klass.fromdict(b) for b in data.get("blocks", [])]
        operations = [[Operation.fromdict(op) for op in step] for step in data.get("operations", [])]
        return cls(Lattice.fromdict(data["lattice"]), blocks, operations)

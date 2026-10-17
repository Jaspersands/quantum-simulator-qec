# Ported from Loom (entropicalabs/el-loom 0.4.0, src/loom_rotated_surface_code/code_factory/rotated_surface_code.py;
# Copyright 2024 Entropica Labs Pte Ltd, Apache License 2.0; see LICENSE in this directory), modified for
# stabilizer_qec: plain classes in place of pydantic.
"""Loom's rotated-surface-code factory: a block of weight-4 checks in a checkerboard and
weight-2 checks on the boundaries, with Loom's coordinates, schedules and syndrome circuits."""

from __future__ import annotations

from functools import cached_property
from itertools import combinations
from typing import Optional, Sequence, Union
from uuid import uuid4

from .circuit import Channel, ChannelType, Circuit, SyndromeCircuit
from .eka import Block, Coord, DiagonalDirection, Direction, FourBodySchedule, Lattice, LatticeType, Orientation, PauliOperator, Stabilizer


class RotatedSurfaceCode(Block):
    """A rotated surface code block, as Loom lays it out (d = 3, X boundaries horizontal,
    weight-2 stabilizer in the first row)::

                           Z
          (0,0) --- (1,0) --- (2,0)
            |         |         |
        X   |    Z    |    X    |
            |         |         |
          (0,1) --- (1,1) --- (2,1)
            |         |         |
            |    X    |    Z    |  X
            |         |         |
          (0,2) --- (1,2) --- (2,2)
                 Z

    Data qubits are ``(x, y, 0)``, ancillas ``(x, y, 1)``. A boundary is *X-type* when its
    Pauli charges are X, that is when its weight-2 stabilizers are Z."""

    @classmethod
    def create(cls, dx: int, dz: int, lattice: Lattice, unique_label: Optional[str] = None, position: tuple[int, ...] = (0, 0), x_boundary: Union[Orientation, str] = Orientation.HORIZONTAL, weight_2_stab_is_first_row: bool = True, weight_4_x_schedule: Union[FourBodySchedule, str, None] = None, logical_x_operator: Optional[PauliOperator] = None, logical_z_operator: Optional[PauliOperator] = None, skip_validation: bool = False) -> "RotatedSurfaceCode":
        """A ``dx`` × ``dz`` block at ``position`` (its top-left data qubit) on a 2D square
        lattice; ``x_boundary`` horizontal puts the X boundaries top and bottom; the default
        logicals are the top row (X when the X boundary is horizontal) and the left column."""
        if lattice.lattice_type != LatticeType.SQUARE_2D:
            raise ValueError(f"The creation of rotated surface code blocks is currently only supported for 2D square lattices. Instead the lattice is of type {lattice.lattice_type}.")
        if not isinstance(position, tuple) or any(not isinstance(x, int) for x in position):
            raise ValueError(f"`position` must be a tuple of integers. Got '{position}' instead.")
        if len(position) != lattice.n_dimensions:
            raise ValueError(f"`position` has length {len(position)} while length {lattice.n_dimensions} is required to match the lattice dimension.")
        if unique_label is None:
            unique_label = str(uuid4())
        x_boundary = Orientation(x_boundary)
        top_left_is_xxxx = (x_boundary == Orientation.HORIZONTAL) != weight_2_stab_is_first_row
        if weight_4_x_schedule is None:
            weight_4_x_schedule = FourBodySchedule.N if x_boundary == Orientation.HORIZONTAL else FourBodySchedule.Z
        else:
            weight_4_x_schedule = FourBodySchedule(weight_4_x_schedule)
        weight_4_z_schedule = weight_4_x_schedule.opposite_schedule()

        if top_left_is_xxxx:
            weight4_top_left = cls.generate_weight4_stabs(dx, dz, "XXXX", weight_4_x_schedule, True)
            weight4_others = cls.generate_weight4_stabs(dx, dz, "ZZZZ", weight_4_z_schedule, False)
        else:
            weight4_top_left = cls.generate_weight4_stabs(dx, dz, "ZZZZ", weight_4_z_schedule, True)
            weight4_others = cls.generate_weight4_stabs(dx, dz, "XXXX", weight_4_x_schedule, False)

        stab_left_right_is_x = top_left_is_xxxx != weight_2_stab_is_first_row
        stab_left_right = "XX" if stab_left_right_is_x else "ZZ"
        stab_top_bottom = "ZZ" if stab_left_right_is_x else "XX"

        # Left boundary
        num = (dz - 1) / 2 if dz % 2 == 1 else dz / 2 - (not weight_2_stab_is_first_row)
        stabs_left = cls.generate_weight2_stabs(stab_left_right, (0, int(not weight_2_stab_is_first_row)), num, Orientation.VERTICAL, False)
        # Right boundary
        right_first_row = weight_2_stab_is_first_row != (dx % 2)
        num = (dz - 1) / 2 if dz % 2 == 1 else dz / 2 - (not right_first_row)
        stabs_right = cls.generate_weight2_stabs(stab_left_right, (dx - 1, int(not right_first_row)), num, Orientation.VERTICAL, True)
        # Top boundary
        num = (dx - 1) / 2 if dx % 2 == 1 else dx / 2 - weight_2_stab_is_first_row
        stabs_top = cls.generate_weight2_stabs(stab_top_bottom, (int(weight_2_stab_is_first_row), 0), num, Orientation.HORIZONTAL, False)
        # Bottom boundary
        bottom_first_col = weight_2_stab_is_first_row == (dz % 2)
        num = (dx - 1) / 2 if dx % 2 == 1 else dx / 2 - (not bottom_first_col)
        stabs_bottom = cls.generate_weight2_stabs(stab_top_bottom, (int(not bottom_first_col), dz - 1), num, Orientation.HORIZONTAL, True)

        stabilizers = weight4_top_left + weight4_others + stabs_left + stabs_right + stabs_top + stabs_bottom

        xxxx = cls.generate_syndrome_circuit("XXXX", None, "XXXX")
        zzzz = cls.generate_syndrome_circuit("ZZZZ", None, "ZZZZ")
        lr_schedule = weight_4_x_schedule if stab_left_right_is_x else weight_4_z_schedule
        tb_schedule = weight_4_z_schedule if stab_left_right_is_x else weight_4_x_schedule
        left = cls.generate_syndrome_circuit(stab_left_right, cls.find_padding(Direction.LEFT, lr_schedule), f"left-{stab_left_right}")
        right = cls.generate_syndrome_circuit(stab_left_right, cls.find_padding(Direction.RIGHT, lr_schedule), f"right-{stab_left_right}")
        top = cls.generate_syndrome_circuit(stab_top_bottom, cls.find_padding(Direction.TOP, tb_schedule), f"top-{stab_top_bottom}")
        bottom = cls.generate_syndrome_circuit(stab_top_bottom, cls.find_padding(Direction.BOTTOM, tb_schedule), f"bottom-{stab_top_bottom}")
        syndrome_circuits = [xxxx, zzzz, top, bottom, left, right]

        stabilizer_to_circuit = {}
        stabilizer_to_circuit.update({s.uuid: (xxxx.uuid if top_left_is_xxxx else zzzz.uuid) for s in weight4_top_left})
        stabilizer_to_circuit.update({s.uuid: (zzzz.uuid if top_left_is_xxxx else xxxx.uuid) for s in weight4_others})
        stabilizer_to_circuit.update({s.uuid: left.uuid for s in stabs_left})
        stabilizer_to_circuit.update({s.uuid: right.uuid for s in stabs_right})
        stabilizer_to_circuit.update({s.uuid: top.uuid for s in stabs_top})
        stabilizer_to_circuit.update({s.uuid: bottom.uuid for s in stabs_bottom})

        if logical_x_operator is None:
            qubits = [(i, 0, 0) for i in range(dx)] if x_boundary == Orientation.HORIZONTAL else [(0, j, 0) for j in range(dz)]
            logical_x_operator = PauliOperator("X" * len(qubits), qubits)
        if logical_z_operator is None:
            qubits = [(0, j, 0) for j in range(dz)] if x_boundary == Orientation.HORIZONTAL else [(i, 0, 0) for i in range(dx)]
            logical_z_operator = PauliOperator("Z" * len(qubits), qubits)

        block = cls(unique_label=unique_label, stabilizers=stabilizers, logical_x_operators=[logical_x_operator], logical_z_operators=[logical_z_operator], syndrome_circuits=syndrome_circuits, stabilizer_to_circuit=stabilizer_to_circuit, skip_validation=skip_validation)
        if position == (0, 0):
            return block
        return block.shift(position)

    @staticmethod
    def find_padding(boundary: Union[Direction, str], schedule: FourBodySchedule) -> tuple[int, int]:
        """The empty steps of a boundary's weight-2 syndrome circuit, so that every circuit of
        the block has the same length and its CNOTs fall in the four-body schedule's slots."""
        boundary = Direction(boundary)
        table = {
            (FourBodySchedule.N, Direction.LEFT): (2, 3),
            (FourBodySchedule.Z, Direction.BOTTOM): (2, 3),
            (FourBodySchedule.N, Direction.RIGHT): (0, 1),
            (FourBodySchedule.Z, Direction.TOP): (0, 1),
            (FourBodySchedule.N, Direction.TOP): (0, 2),
            (FourBodySchedule.Z, Direction.RIGHT): (0, 2),
            (FourBodySchedule.N, Direction.BOTTOM): (1, 3),
            (FourBodySchedule.Z, Direction.LEFT): (1, 3),
        }
        if (schedule, boundary) not in table:
            raise ValueError(f"The boundary {boundary} and schedule {schedule} are not compatible. Only N and Z schedules are supported.")
        return table[(schedule, boundary)]

    @staticmethod
    def generate_syndrome_circuit(pauli: str, padding: Optional[tuple[int, ...]], name: str) -> SyndromeCircuit:
        """Reset, H, one controlled Pauli per data qubit (empty steps at ``padding``), H,
        measurement."""
        weight = len(pauli)
        data = [Channel(type=ChannelType.QUANTUM, label=f"d{i}") for i in range(weight)]
        cbit = Channel(type=ChannelType.CLASSICAL, label="c0")
        ancilla = Channel(type=ChannelType.QUANTUM, label="a0")
        reset = [Circuit("Reset_0", channels=[ancilla])]
        h1 = [Circuit("H", channels=[ancilla])]
        h2 = [Circuit("H", channels=[ancilla])]
        if padding is None:
            entangle = [[Circuit(f"C{p}", channels=[ancilla, data[i]])] for i, p in enumerate(pauli)]
        else:
            padded_pauli: list = list(pauli)
            padded_data: list = list(data)
            for i in padding:
                padded_pauli.insert(i, None)
                padded_data.insert(i, None)
            entangle = [([Circuit(f"C{p}", channels=[ancilla, padded_data[i]])] if p is not None else []) for i, p in enumerate(padded_pauli)]
        measurement = [Circuit("Measurement", channels=[ancilla, cbit])]
        circuit = Circuit(name=name, circuit=[reset, h1] + entangle + [h2, measurement], channels=data + [ancilla, cbit])
        return SyndromeCircuit(pauli=pauli, name=name, circuit=circuit)

    # Properties
    @property
    def size(self) -> tuple[int, int]:
        xs = [q[0] for q in self.data_qubits]
        ys = [q[1] for q in self.data_qubits]
        return (max(xs) - min(xs) + 1, max(ys) - min(ys) + 1)

    @property
    def upper_left_qubit(self) -> Coord:
        return min(self.data_qubits, key=lambda q: q[0] + q[1])

    @property
    def upper_left_4body_stabilizer(self) -> Stabilizer:
        return [s for s in self.stabilizers if len(s.data_qubits) == 4 and self.upper_left_qubit in s.data_qubits][0]

    @property
    def weight_4_x_schedule(self) -> FourBodySchedule:
        for stab in self.stabilizers:
            if stab.pauli == "XXXX":
                return FourBodySchedule.N if stab.data_qubits[1][1] > stab.data_qubits[0][1] else FourBodySchedule.Z
        raise ValueError("XXXX stabilizer not found in the stabilizers.")

    @property
    def weight_4_z_schedule(self) -> FourBodySchedule:
        return FourBodySchedule.N if self.weight_4_x_schedule == FourBodySchedule.Z else FourBodySchedule.Z

    @property
    def x_boundary(self) -> Orientation:
        return Orientation.HORIZONTAL if self.boundary_type(Direction.TOP) == "X" else Orientation.VERTICAL

    @property
    def weight_2_stab_is_first_row(self) -> bool:
        first = self.upper_left_qubit
        second = tuple(q if i != 1 else q + 1 for i, q in enumerate(first))
        return any({first, second} == set(stab.data_qubits) for stab in self.stabilizers)

    @property
    def topological_corners(self) -> tuple[Coord, ...]:
        return tuple(q for q, p in self.pauli_charges.items() if p == "Y")

    @property
    def geometric_corners(self) -> tuple[Coord, ...]:
        def sole_max(f):
            best = max(f(q) for q in self.data_qubits)
            qs = [q for q in self.data_qubits if f(q) == best]
            return qs[0] if len(qs) == 1 else None

        fs = [lambda x: x[0] + x[1], lambda x: x[0] - x[1], lambda x: -x[0] + x[1], lambda x: -x[0] - x[1], lambda x: x[0], lambda x: x[1], lambda x: -x[0], lambda x: -x[1]]
        return tuple(sole_max(f) for f in fs if sole_max(f) is not None)

    @property
    def all_boundary_stabilizers(self) -> tuple[Stabilizer, ...]:
        return self.boundary_stabilizers(Direction.TOP) + self.boundary_stabilizers(Direction.BOTTOM) + self.boundary_stabilizers(Direction.LEFT) + self.boundary_stabilizers(Direction.RIGHT)

    @property
    def bulk_stabilizers(self) -> tuple[Stabilizer, ...]:
        boundary = self.all_boundary_stabilizers
        return tuple(s for s in self.stabilizers if s not in boundary)

    @property
    def orientation(self) -> Optional[Orientation]:
        w, h = self.size
        if w > h:
            return Orientation.HORIZONTAL
        if w < h:
            return Orientation.VERTICAL
        return None

    @property
    def is_horizontal(self) -> bool:
        return self.orientation == Orientation.HORIZONTAL

    @property
    def is_vertical(self) -> bool:
        return self.orientation == Orientation.VERTICAL

    # Static constructors of stabilizers
    @staticmethod
    def generate_weight4_stabs(dx: int, dz: int, pauli: str, schedule: FourBodySchedule, start_in_top_left_corner: bool, initial_position: Sequence[int] = (0, 0)) -> list[Stabilizer]:
        """The weight-4 stabilizers of one type over a ``dx`` × ``dz`` region, on the
        checkerboard squares starting (or not) at the top left; data qubits in the schedule's
        order, the ancilla at the plaquette's centre."""
        if len(initial_position) > 2:
            initial_position = tuple(initial_position[:2])
        elif len(initial_position) < 2:
            raise ValueError(f"Initial position must be a tuple of length >= 2. Got {initial_position}.")
        x, z = initial_position[0], initial_position[1]
        n = schedule == FourBodySchedule.N
        return [
            Stabilizer(
                pauli=pauli,
                data_qubits=[(x + i + 1, z + j, 0), (x + i + int(n), z + j + int(n), 0), (x + i + int(not n), z + j + int(not n), 0), (x + i, z + j + 1, 0)],
                ancilla_qubits=[(x + i + 1, z + j + 1, 1)],
            )
            for i in range(dx - 1)
            for j in range(dz - 1)
            if (i + j) % 2 != start_in_top_left_corner
        ]

    @staticmethod
    def generate_weight2_stabs(pauli: str, initial_position: Sequence[int], num_stabs: float, orientation: Orientation, is_bottom_or_right: bool) -> list[Stabilizer]:
        """The weight-2 stabilizers along one boundary, every second pair from the initial
        position; the ancilla outside the block for the bottom and right boundaries."""
        h = orientation == Orientation.HORIZONTAL
        x0, y0 = initial_position[0], initial_position[1]
        return [
            Stabilizer(
                pauli=pauli,
                data_qubits=[(x0 + h * (2 * i + 1), y0 + (not h) * 2 * i, 0), (x0 + h * (2 * i), y0 + (not h) * (2 * i + 1), 0)],
                ancilla_qubits=[(x0 + (not h) * is_bottom_or_right + h * (2 * i + 1), y0 + h * is_bottom_or_right + (not h) * (2 * i + 1), 1)],
            )
            for i in range(int(num_stabs))
        ]

    # Instance methods
    def boundary_qubits(self, direction: Union[Direction, str]) -> list[Coord]:
        direction = Direction(direction)
        axis = 1 if direction in (Direction.TOP, Direction.BOTTOM) else 0
        pick = max if direction in (Direction.BOTTOM, Direction.RIGHT) else min
        edge = pick(q[axis] for q in self.data_qubits)
        return [q for q in self.data_qubits if q[axis] == edge]

    def boundary_type(self, direction: Union[Direction, str]) -> str:
        """``X`` or ``Z``: the Pauli charge of the boundary's qubits (corners aside)."""
        direction = Direction(direction)
        if direction in (Direction.TOP, Direction.BOTTOM) and self.size[1] <= 2:
            raise ValueError("The block is too small to have a top or bottom boundary of well defined type.")
        if direction in (Direction.LEFT, Direction.RIGHT) and self.size[0] <= 2:
            raise ValueError("The block is too small to have a left or right boundary of well defined type.")
        charges = self.pauli_charges
        found = list({charges[q] for q in self.boundary_qubits(direction)} - {"Y"})
        if len(found) > 1:
            raise RuntimeError(f"Boundary has multiple Pauli charges. This should not happen for the standard rotated surface code block. The boundary pauli charges (excluding Y charges) are {found}.")
        return found[0]

    def boundary_stabilizers(self, direction: Union[Direction, str]) -> tuple[Stabilizer, ...]:
        qubits = self.boundary_qubits(direction)
        return tuple(s for s in self.stabilizers if all(q in qubits for q in s.data_qubits))

    def get_corner_from_direction(self, which_corner: DiagonalDirection) -> Coord:
        a, b = which_corner.components
        return (set(self.boundary_qubits(a)) & set(self.boundary_qubits(b))).pop()

    @cached_property
    def config_and_pivot_corners(self) -> tuple[int, tuple[Coord, ...]]:
        """Loom's classification of the block by its topological corners: 1 a rectangle (the
        corners top-left, bottom-left, bottom-right, top-right), 2 a U, 3 an L, 4 the phase
        gate's U (long end, middle edge, angle, short end), 0 anything else."""
        dx, dz = self.size
        horizontal = dx >= dz
        tg = set(self.topological_corners) & set(self.geometric_corners)
        if len(tg) == 4:
            return 1, tuple(sorted(tg))
        almost_two = dx == 2 * dz - 1 or dz == 2 * dx - 1
        two = dx == 2 * dz or dz == 2 * dx
        ul = len(tg) == 3 and almost_two
        uphase = len(tg) == 3 and two
        if not (ul or uphase):
            return 0, self.topological_corners
        other = (set(self.topological_corners) - tg).pop()
        long_idx, short_idx = (0, 1) if horizontal else (1, 0)
        long_pair = next((c1, c2) for c1, c2 in combinations(tg, 2) if c1[short_idx] == c2[short_idx])
        short_corner = next(c for c in tg if c not in long_pair)
        long_pair = sorted(long_pair, reverse=other[long_idx] > short_corner[long_idx])
        middle_long = abs(other[long_idx] - long_pair[0][long_idx])
        middle_angle = abs(long_pair[1][long_idx] - other[long_idx])
        diff = middle_angle - middle_long
        if diff not in (0, 1):
            return 0, self.topological_corners
        if other[short_idx] == short_corner[short_idx] and diff == 0:
            config = 2
        elif other[short_idx] == long_pair[0][short_idx] and diff == 0:
            config = 3
        elif other[short_idx] == short_corner[short_idx] and diff == 1:
            config = 4
        else:
            raise RuntimeError("Something went wrong with the configuration detection.")
        return config, (long_pair[0], other, long_pair[1], short_corner)

    def get_shifted_equivalent_logical_operator(self, initial_operator: PauliOperator, new_upleft_qubit: Coord) -> tuple[PauliOperator, tuple[Stabilizer, ...]]:
        """A logical operator moved so that its upper-left qubit is ``new_upleft_qubit``, with
        the stabilizers (of its own type) between the old and new positions whose product
        relates them."""
        if len(self.logical_x_operators) != 1 or len(self.logical_z_operators) != 1:
            raise NotImplementedError("This function currently only supports blocks with a single logical X and Z operator.")
        new_upleft_qubit = tuple(new_upleft_qubit)
        if new_upleft_qubit not in self.data_qubits:
            raise ValueError(f"The new upleft qubit {new_upleft_qubit} is not part of the data qubits of the block.")
        if initial_operator not in (self.logical_x_operators[0], self.logical_z_operators[0]):
            raise ValueError("The initial operator must be one of the logical operators of the block.")
        if self.pauli_charges[new_upleft_qubit] == initial_operator.pauli[0]:
            raise ValueError(f"The new upleft qubit {new_upleft_qubit} is not part of the correct boundary.")
        initial_upleft = min(initial_operator.data_qubits, key=lambda q: q[0] + q[1])
        shift = tuple(a - b for a, b in zip(new_upleft_qubit, initial_upleft))
        new_operator = PauliOperator(initial_operator.pauli, [tuple(c + s for c, s in zip(q, shift)) for q in initial_operator.data_qubits])

        def vector_range(vector: tuple[int, ...]) -> list[tuple[int, ...]]:
            k_pos = next(i for i, v in enumerate(vector) if v != 0)
            k = vector[k_pos]
            values = range(k + 1) if k > 0 else range(k, 1)
            return [tuple(i if j == k_pos else 0 for j in range(len(vector))) for i in values]

        nonzero = sum(1 for v in shift if v != 0)
        if nonzero == 0:
            return new_operator, ()
        if nonzero == 1:
            between = [tuple(c + v for c, v in zip(q, vec)) for q in initial_operator.data_qubits for vec in vector_range(shift)]
            stabs = [s for s in self.stabilizers if all(q in between for q in s.data_qubits) and s.pauli[0] == initial_operator.pauli[0]]
            return new_operator, tuple(stabs)
        raise ValueError("The shift vector must have at most a single non-zero dimension.")

    def rename(self, name: str) -> "RotatedSurfaceCode":
        return super().rename(name)

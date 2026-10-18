# Ported from Loom (entropicalabs/el-loom 0.4.0, src/loom_rotated_surface_code/applicator/utilities.py, with
# Circuit.construct_padded_circuit_time_sequence from src/loom/eka/circuit.py and DiagonalDirection.to_vector and
# direction_along_orientation from src/loom/eka/utilities/enums.py; Copyright 2024 Entropica Labs Pte Ltd, Apache
# License 2.0; see LICENSE in this directory), modified for stabilizer_qec: the enum and circuit methods as
# functions, and the new syndrome circuits kept in the order they were made (Loom passes them through a set, which
# only reorders a block's tuple of circuits; the interpreter finds them by uuid or by name).
"""Geometry shared by the rotated surface code's move, rotate and Y-wall applicators: diagonal
steps on the two sublattices (data qubits ``(x, y, 0)``, ancillas ``(x, y, 1)``), a block
shifted by one, the four-body schedules of a block of any corner configuration, the syndrome
circuits that follow from them, and Loom's padding of a time sequence of circuits."""

from __future__ import annotations

from enum import Enum
from typing import Optional, Sequence, Union

from .circuit import Circuit, SyndromeCircuit
from .eka import DiagonalDirection, Direction, FourBodySchedule, Orientation, PauliOperator, Stabilizer
from .rotated_surface_code import RotatedSurfaceCode

Coord = tuple[int, ...]


def pauli_type(op: PauliOperator) -> str:
    """Loom's ``PauliOperator.pauli_type``: the one Pauli the operator is made of."""
    unique = set(op.pauli)
    if len(unique) != 1:
        raise ValueError(f"PauliOperator must consist of a single type of Pauli operator to determine its type, got {unique} instead.")
    return unique.pop()


def diagonal_vector(diagonal: DiagonalDirection) -> tuple[int, int]:
    """Loom's ``DiagonalDirection.to_vector``: the sum of its two directions' vectors."""
    vectors = [d.to_vector() for d in diagonal.components]
    return (vectors[0][0] + vectors[1][0], vectors[0][1] + vectors[1][1])


def direction_along_orientation(diagonal: DiagonalDirection, orientation: Orientation) -> Direction:
    """Loom's ``DiagonalDirection.direction_along_orientation``: the component along it."""
    return next(d for d in diagonal.components if d.to_orientation() == orientation)


def construct_padded_circuit_time_sequence(circuit_time_sequence: Sequence[Sequence[Circuit]]) -> tuple[tuple[Circuit, ...], ...]:
    """Loom's ``Circuit.construct_padded_circuit_time_sequence``: each tick of circuits starts
    once the channels it shares with earlier ticks are free, empty ticks filling the wait (and
    the last tick's duration)."""
    padded: tuple[tuple[Circuit, ...], ...] = ()
    occupancy: dict[str, int] = {}
    for tick in circuit_time_sequence:
        current = {channel.label: circuit.duration for circuit in tick for channel in circuit.channels}
        conflicting = set(occupancy) & set(current)
        if conflicting:
            duration = max(occupancy[c] for c in conflicting)
            padded += ((),) * (duration - 1)
        else:
            duration = 1
        padded += (tuple(tick),)
        for channel in set(occupancy) - set(current):
            if (left := occupancy.pop(channel) - duration) > 0:
                occupancy[channel] = left
        occupancy.update(current)
    if occupancy:
        padded += ((),) * (max(occupancy.values()) - 1)
    return padded


def add_vector(qubit: Coord, direction: Direction, length: int) -> Coord:
    out = list(qubit)
    for i, coord in enumerate(direction.to_vector()):
        out[i] += coord * length
    return tuple(out)


def find_schedules(rsc_block: RotatedSurfaceCode) -> tuple[FourBodySchedule, FourBodySchedule, Optional[FourBodySchedule], Optional[FourBodySchedule]]:
    """The four-body schedules (X and Z outside the triangle, X and Z inside it) of a block by
    its corner configuration: the block's own for a configuration 0, else chosen by the type of
    the short-edge boundary."""
    config, pivot_corners = rsc_block.config_and_pivot_corners
    is_horizontal = rsc_block.is_horizontal
    if config == 0:
        x = rsc_block.weight_4_x_schedule
        return x, x.opposite_schedule(), None, None
    short_end_corner = pivot_corners[3]
    long_edge_idx = 0 if is_horizontal else 1
    at_boundary = [s for s in rsc_block.stabilizers if len(s.pauli) == 2 and all(q[long_edge_idx] == short_end_corner[long_edge_idx] for q in s.data_qubits)]
    boundary_paulis = [s.pauli for s in at_boundary]
    if len(set(boundary_paulis)) != 1:
        raise ValueError(f"Accept only one type of stabilizers along short-edge boundary, found {boundary_paulis}.")
    boundary_type = boundary_paulis[0][0]
    if config in (1, 2, 4):
        if boundary_type == "Z":
            x = FourBodySchedule.Z if is_horizontal else FourBodySchedule.N
        else:
            x = FourBodySchedule.N if is_horizontal else FourBodySchedule.Z
        return x, x.opposite_schedule(), None, None
    if config == 3:
        if boundary_type == "Z":
            tx = FourBodySchedule.N if is_horizontal else FourBodySchedule.Z
        else:
            tx = FourBodySchedule.Z if is_horizontal else FourBodySchedule.N
        tz = tx.opposite_schedule()
        return tx.opposite_schedule(), tz.opposite_schedule(), tx, tz
    raise ValueError(f"Unknown corner configuration {config}.")


def find_stabilizer_position(rsc_block: RotatedSurfaceCode, config: int, pivot_corners: Sequence[Coord], stab: Stabilizer, is_horizontal: bool) -> tuple[bool, Optional[Direction]]:
    """Whether the stabilizer lies in the triangle of a configuration-3 block, and the boundary
    of a weight-2 stabilizer."""
    if config == 0:
        in_triangle = False
    else:
        long_end_corner, middle_corner, _, short_end_corner = pivot_corners
        long_idx = 0 if is_horizontal else 1
        short_idx = 1 if is_horizontal else 0

        def triangular(q) -> bool:
            if config in (1, 2, 4):
                return False
            if config == 3:
                dshort = middle_corner[short_idx] - q[short_idx] if middle_corner[short_idx] > short_end_corner[short_idx] else q[short_idx] - middle_corner[short_idx]
                dlong = middle_corner[long_idx] - q[long_idx] if middle_corner[long_idx] > long_end_corner[long_idx] else q[long_idx] - middle_corner[long_idx]
                return 0 <= dshort <= dlong
            raise ValueError(f"Unknown corner configuration {config}.")

        in_triangle = all(triangular(q) for q in stab.data_qubits)
    boundary = None
    if len(stab.pauli) == 2:
        qubit_directions = [set(d for d in Direction if q in rsc_block.boundary_qubits(d)) for q in stab.data_qubits]
        boundary = list(qubit_directions[0] & qubit_directions[1])[0]
    return in_triangle, boundary


def create_new_syndrome_circuits_with_known_schedules(rsc_block: RotatedSurfaceCode, non_triangle_x_schedule: FourBodySchedule, non_triangle_z_schedule: FourBodySchedule, triangle_x_schedule: Optional[FourBodySchedule], triangle_z_schedule: Optional[FourBodySchedule], starting_qubit_diag_directions: DiagonalDirection) -> tuple[tuple[SyndromeCircuit, ...], dict[str, str]]:
    """A syndrome circuit per stabilizer (named by its weight-4 Pauli string or its boundary,
    or by its region for configurations 2 to 4), its channels ordered so that the first CNOT
    goes to the qubit towards the starting diagonal direction; equal circuits (channel order
    aside) are shared, the first one's channels serving them all, as in Loom."""

    def data_qubits_order(stab: Stabilizer, diag: DiagonalDirection, schedule: FourBodySchedule, boundary_direction: Optional[Direction]) -> tuple[Coord, ...]:
        is_n = schedule == FourBodySchedule.N
        key = {Direction.TOP: lambda x: -x[1], Direction.BOTTOM: lambda x: x[1], Direction.LEFT: lambda x: -x[0], Direction.RIGHT: lambda x: x[0]}
        qubits = list(stab.data_qubits)
        if len(stab.pauli) == 2:
            v = boundary_direction.to_vector()
            qubits += [(q[0] + v[0], q[1] + v[1], q[2]) for q in qubits]
        for direction in diag.components:
            best = max(key[direction](q) for q in stab.data_qubits)
            qubits = [q for q in qubits if key[direction](q) == best]
        start = qubits[0]
        dv = diagonal_vector(diag)
        order = [start, (start[0] - dv[0] * (not is_n), start[1] - dv[1] * is_n, start[2]), (start[0] - dv[0] * is_n, start[1] - dv[1] * (not is_n), start[2]), (start[0] - dv[0], start[1] - dv[1], start[2])]
        return tuple(q for q in order if q in stab.data_qubits)

    is_horizontal = rsc_block.is_horizontal
    config, pivot_corners = rsc_block.config_and_pivot_corners
    if (triangle_x_schedule is None or triangle_z_schedule is None) and config == 3:
        raise ValueError("Triangle schedules must be provided for type-3 corner configuration.")
    syndrome_circuits: tuple[SyndromeCircuit, ...] = ()
    stabilizer_to_circuit: dict[str, str] = {}
    for stab in rsc_block.stabilizers:
        ptype = pauli_type(stab)
        in_triangle, boundary_direction = find_stabilizer_position(rsc_block, config, pivot_corners, stab, is_horizontal)
        if in_triangle:
            schedule = triangle_x_schedule if ptype == "X" else triangle_z_schedule
        else:
            schedule = non_triangle_x_schedule if ptype == "X" else non_triangle_z_schedule
        weight = stab.weight
        if config in (0, 1):
            if weight == 4:
                name = stab.pauli.lower()
            elif weight == 2:
                name = f"{boundary_direction}-{stab.pauli.lower()}"
            else:
                continue
        elif config in (2, 3, 4):
            name = f"{'triangular' if in_triangle else 'non_triangular'}-{boundary_direction if boundary_direction else 'bulk'}-{str(ptype) * len(stab.pauli)}"
        else:
            raise ValueError(f"Unknown corner configuration {config}.")
        padding = None
        if weight == 2:
            padding_boundary = boundary_direction
            for direction in starting_qubit_diag_directions.components:
                if direction not in (Direction.TOP, Direction.RIGHT):
                    padding_boundary = padding_boundary.mirror_across_orientation(direction.to_orientation().perpendicular())
            padding = RotatedSurfaceCode.find_padding(padding_boundary, schedule)
        syndrome_circuit = RotatedSurfaceCode.generate_syndrome_circuit(stab.pauli, padding, name)
        ordered = data_qubits_order(stab, starting_qubit_diag_directions, schedule, boundary_direction)
        indices = [ordered.index(q) for q in stab.data_qubits]
        old = syndrome_circuit.circuit
        channels = [old.channels[i] for i in indices] + list(old.channels[len(indices):])
        syndrome_circuit = SyndromeCircuit(pauli=syndrome_circuit.pauli, name=syndrome_circuit.name, circuit=Circuit(old.name, old.circuit, channels))
        same = next((c for c in syndrome_circuits if c == syndrome_circuit), None)
        if same is not None:
            syndrome_circuit = same
        else:
            syndrome_circuits += (syndrome_circuit,)
        stabilizer_to_circuit[stab.uuid] = syndrome_circuit.uuid
    return syndrome_circuits, stabilizer_to_circuit


def generate_syndrome_extraction_circuits(rsc_block: RotatedSurfaceCode, starting_qubit_diagonal_direction: DiagonalDirection) -> tuple[tuple[SyndromeCircuit, ...], dict[str, str]]:
    return create_new_syndrome_circuits_with_known_schedules(rsc_block, *find_schedules(rsc_block), starting_qubit_diagonal_direction)


def direction_to_coord(direction: Union[Direction, DiagonalDirection], sublattice_index: Optional[int] = None) -> tuple[int, int, int]:
    """The step of a move: one site along a direction, or diagonally, which on the two
    sublattices goes from a data qubit to an ancilla (index 0) or back (index 1), or with no
    index from a qubit to the next of its own kind."""
    straight = {Direction.TOP: (0, -1, 0), Direction.RIGHT: (1, 0, 0), Direction.LEFT: (-1, 0, 0), Direction.BOTTOM: (0, 1, 0)}
    diagonal = {DiagonalDirection.TOP_RIGHT: (1, -1, 0), DiagonalDirection.TOP_LEFT: (-1, -1, 0), DiagonalDirection.BOTTOM_RIGHT: (1, 1, 0), DiagonalDirection.BOTTOM_LEFT: (-1, 1, 0)}
    diagonal_0 = {DiagonalDirection.TOP_RIGHT: (1, 0, 1), DiagonalDirection.TOP_LEFT: (0, 0, 1), DiagonalDirection.BOTTOM_RIGHT: (1, 1, 1), DiagonalDirection.BOTTOM_LEFT: (0, 1, 1)}
    diagonal_1 = {DiagonalDirection.TOP_RIGHT: (0, -1, -1), DiagonalDirection.TOP_LEFT: (-1, -1, -1), DiagonalDirection.BOTTOM_RIGHT: (0, 0, -1), DiagonalDirection.BOTTOM_LEFT: (-1, 0, -1)}
    if isinstance(direction, Direction):
        return straight[direction]
    if isinstance(direction, DiagonalDirection):
        if not (isinstance(sublattice_index, int) or sublattice_index is None):
            raise ValueError("Sub-lattice index must be an integer or None for diagonal movements.")
        if sublattice_index == 0:
            return diagonal_0[direction]
        if sublattice_index == 1:
            return diagonal_1[direction]
        if sublattice_index is None:
            return diagonal[direction]
        return None
    raise ValueError("Invalid direction. direction must be type Direction or set of Direction")


def update_qubit_coords(input_qubit_coords: Sequence[Coord], direction: Union[Direction, DiagonalDirection]) -> tuple[Coord, ...]:
    return tuple(tuple(c + d for c, d in zip(q, direction_to_coord(direction, q[-1]), strict=True)) for q in input_qubit_coords)


def shift_block_towards_direction(rsc_block: RotatedSurfaceCode, direction: Union[Direction, DiagonalDirection], debug_mode: bool) -> tuple[RotatedSurfaceCode, dict[str, tuple[str, ...]], dict[str, tuple[str, ...]], dict[str, tuple[str, ...]]]:
    """The block one step along the direction (qubits of each sublattice to their next site),
    with the evolutions of its stabilizers and logical operators."""
    new_stabilizers = [Stabilizer(pauli=s.pauli, data_qubits=update_qubit_coords(s.data_qubits, direction), ancilla_qubits=update_qubit_coords(s.ancilla_qubits, direction)) for s in rsc_block.stabilizers]
    new_x = [PauliOperator(o.pauli, data_qubits=update_qubit_coords(o.data_qubits, direction)) for o in rsc_block.logical_x_operators]
    new_z = [PauliOperator(o.pauli, data_qubits=update_qubit_coords(o.data_qubits, direction)) for o in rsc_block.logical_z_operators]
    old_to_new = {s.uuid: new_stabilizers[i].uuid for i, s in enumerate(rsc_block.stabilizers)}
    new_stc = {old_to_new[s]: c for s, c in rsc_block.stabilizer_to_circuit.items()}
    new_block = RotatedSurfaceCode(unique_label=rsc_block.unique_label, stabilizers=new_stabilizers, logical_x_operators=new_x, logical_z_operators=new_z, syndrome_circuits=rsc_block.syndrome_circuits, stabilizer_to_circuit=new_stc, skip_validation=not debug_mode)
    stab_evolution = {new.uuid: (old.uuid,) for old, new in zip(rsc_block.stabilizers, new_stabilizers)}
    x_evolution = {new.uuid: (old.uuid,) for old, new in zip(rsc_block.logical_x_operators, new_x)}
    z_evolution = {new.uuid: (old.uuid,) for old, new in zip(rsc_block.logical_z_operators, new_z)}
    return new_block, stab_evolution, x_evolution, z_evolution


class DetailedSchedule(Enum):
    """A four-body schedule with its starting corner: the order in which the plaquette's
    qubits 1 (top left), 2 (top right), 3 (bottom left) and 4 (bottom right) are entangled."""

    N1 = (1, 3, 2, 4)
    N2 = (2, 4, 1, 3)
    N3 = (3, 1, 4, 2)
    N4 = (4, 2, 3, 1)
    Z1 = (1, 2, 3, 4)
    Z2 = (2, 1, 4, 3)
    Z3 = (3, 4, 1, 2)
    Z4 = (4, 3, 2, 1)

    @classmethod
    def from_schedule_and_direction(cls, schedule: FourBodySchedule, starting_qubit_direction: DiagonalDirection) -> "DetailedSchedule":
        table = {
            (FourBodySchedule.N, DiagonalDirection.TOP_LEFT): cls.N1,
            (FourBodySchedule.N, DiagonalDirection.TOP_RIGHT): cls.N2,
            (FourBodySchedule.N, DiagonalDirection.BOTTOM_LEFT): cls.N3,
            (FourBodySchedule.N, DiagonalDirection.BOTTOM_RIGHT): cls.N4,
            (FourBodySchedule.Z, DiagonalDirection.TOP_LEFT): cls.Z1,
            (FourBodySchedule.Z, DiagonalDirection.TOP_RIGHT): cls.Z2,
            (FourBodySchedule.Z, DiagonalDirection.BOTTOM_LEFT): cls.Z3,
            (FourBodySchedule.Z, DiagonalDirection.BOTTOM_RIGHT): cls.Z4,
        }
        if (schedule, starting_qubit_direction) not in table:
            raise ValueError(f"Invalid combination of schedule {schedule} and starting qubit direction {starting_qubit_direction}.")
        return table[(schedule, starting_qubit_direction)]

    def is_N(self) -> bool:
        return self.name[0] == "N"

    def is_Z(self) -> bool:
        return self.name[0] == "Z"

    def invert_vertically(self) -> "DetailedSchedule":
        s = DetailedSchedule
        return {s.N1: s.N2, s.N2: s.N1, s.N3: s.N4, s.N4: s.N3, s.Z1: s.Z2, s.Z2: s.Z1, s.Z3: s.Z4, s.Z4: s.Z3}[self]

    def rotate_ccw_90(self) -> "DetailedSchedule":
        s = DetailedSchedule
        return {s.N1: s.Z3, s.Z3: s.N4, s.N4: s.Z2, s.Z2: s.N1, s.Z1: s.N3, s.N3: s.Z4, s.Z4: s.N2, s.N2: s.Z1}[self]

    def get_stabilizer_qubits(self, stab: Stabilizer, boundary_direction: Optional[Direction] = None) -> list[Optional[Coord]]:
        """The stabilizer's data qubits in the schedule's order, None where a weight-2
        stabilizer on that boundary has no qubit."""
        index_to_qubit = {1: max(stab.data_qubits, key=lambda x: -x[0] - x[1]), 2: max(stab.data_qubits, key=lambda x: x[0] - x[1]), 3: max(stab.data_qubits, key=lambda x: -x[0] + x[1]), 4: max(stab.data_qubits, key=lambda x: x[0] + x[1])}
        absent: list[int] = []
        if len(stab.data_qubits) == 2:
            if boundary_direction == Direction.TOP:
                absent = [1, 2]
            elif boundary_direction == Direction.RIGHT:
                absent = [2, 4]
            elif boundary_direction == Direction.LEFT:
                absent = [1, 3]
            elif boundary_direction == Direction.BOTTOM:
                absent = [3, 4]
            else:
                raise ValueError("The boundary direction must be specified.")
        return [index_to_qubit[i] if i not in absent else None for i in self.value]


def find_detailed_schedules(rsc_block: RotatedSurfaceCode, starting_direction: DiagonalDirection) -> dict[Stabilizer, DetailedSchedule]:
    nx, nz, tx, tz = find_schedules(rsc_block)
    is_horizontal = rsc_block.is_horizontal
    config, pivot_corners = rsc_block.config_and_pivot_corners
    out: dict[Stabilizer, DetailedSchedule] = {}
    for stab in rsc_block.stabilizers:
        ptype = pauli_type(stab)
        in_triangle, _ = find_stabilizer_position(rsc_block, config, pivot_corners, stab, is_horizontal)
        if in_triangle:
            schedule = tx if ptype == "X" else tz
        else:
            schedule = nx if ptype == "X" else nz
        out[stab] = DetailedSchedule.from_schedule_and_direction(schedule, starting_direction)
    return out


def find_relative_diagonal_direction(from_qubit: Coord, to_qubit: Coord) -> DiagonalDirection:
    horizontal = Direction.RIGHT if to_qubit[0] > from_qubit[0] else Direction.LEFT
    vertical = Direction.BOTTOM if to_qubit[1] > from_qubit[1] else Direction.TOP
    return DiagonalDirection.from_directions((horizontal, vertical))

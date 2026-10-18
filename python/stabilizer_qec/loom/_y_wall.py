# Ported from Loom (entropicalabs/el-loom 0.4.0, src/loom_rotated_surface_code/applicator/y_wall_out/{y_wall_out,
# y_wall_out_initial_syndrome_measurement,y_wall_measurement_hadamard,y_wall_out_recombination_swap_then_qec,
# y_wall_out_final_swap_then_qec,y_wall_out_final_syndrome_measurement,y_wall_out_utilities}.py; Copyright 2024
# Entropica Labs Pte Ltd, Apache License 2.0; see LICENSE in this directory), modified for stabilizer_qec: the seven
# files in one module, plain functions on this package's interpretation step.
"""Loom's Y-wall-out, the heart of the logical phase gate: on a block twice as long as wide
(a square that keeps the logical, and a square of the same size beside it, met by a corner
configuration 4), a round of syndromes; the wall of qubits between the two squares measured in
Y while the far square gets a transversal Hadamard; two swap-then-QEC steps that recombine
the squares into a block of configuration 3 (an L of corners) and put it back on its sites;
then d - 2 rounds of syndromes with that block's schedules."""

from __future__ import annotations

from typing import Optional, Sequence

from .circuit import Circuit
from .eka import DiagonalDirection, Direction, FourBodySchedule, Orientation, PauliOperator, Stabilizer
from .interpreter import Cbit, InterpretationStep
from .rotated_surface_code import RotatedSurfaceCode
from ._move_block import find_swap_then_qec_qubit_initializations, generate_and_append_block_syndromes_and_detectors, generate_syndrome_measurement_circuit_and_cbits
from ._rsc_geometry import DetailedSchedule, construct_padded_circuit_time_sequence, direction_to_coord, find_relative_diagonal_direction, generate_syndrome_extraction_circuits, pauli_type

Coord = tuple[int, ...]


# ---------------------------------------------------------------------------------------------
# The operation
# ---------------------------------------------------------------------------------------------


def y_wall_out(step: InterpretationStep, block: RotatedSurfaceCode, wall_position: int, wall_orientation: Orientation, same_timeslice: bool, debug_mode: bool) -> InterpretationStep:
    step.begin_composite_operation_session_MUT(same_timeslice=same_timeslice, circuit_name=f"y_wall_out operation on block {block.unique_label}")
    y_wall_out_consistency_check(block, wall_position, wall_orientation)
    is_wall_hor = wall_orientation == Orientation.HORIZONTAL
    qubits_to_measure, qubits_to_idle, qubits_to_hadamard = find_qubit_sets(block, wall_position, is_wall_hor)
    step = y_wall_out_initial_syndrome_measurement(step, block, same_timeslice=False, debug_mode=debug_mode)
    step = y_wall_out_measurement_and_hadamard(step, block, wall_position, wall_orientation, same_timeslice=False, debug_mode=debug_mode)
    current = step.get_block(block.unique_label)
    idle_side_directions = get_idle_hadamard_info(current, qubits_to_hadamard, qubits_to_idle)[3]
    step = y_wall_out_recombination_swap_then_qec(step, current, block, qubits_to_idle, qubits_to_measure, qubits_to_hadamard, same_timeslice=False, debug_mode=debug_mode)
    current = step.get_block(block.unique_label)
    step = y_wall_out_final_swap_then_qec(step, current, idle_side_directions, same_timeslice=False, debug_mode=debug_mode)
    current = step.get_block(block.unique_label)
    step = y_wall_out_final_qec_rounds(step, current, same_timeslice=False, debug_mode=debug_mode)
    circuit = step.end_composite_operation_session_MUT()
    step.append_circuit_MUT(circuit, same_timeslice)
    return step


def y_wall_out_consistency_check(block: RotatedSurfaceCode, wall_pos: int, wall_orientation: Orientation) -> None:
    is_wall_hor = wall_orientation == Orientation.HORIZONTAL
    topological_corners = block.topological_corners
    geometric_corners = block.geometric_corners
    ul = block.upper_left_qubit
    is_ul_stab_x = block.upper_left_4body_stabilizer.pauli[0] == "X"
    dim_x, dim_z = block.size
    larger, smaller = max(dim_x, dim_z), min(dim_x, dim_z)
    if smaller % 2 == 0 or larger % 2 != 0:
        raise ValueError("The smaller dimension of the block must be odd and the larger dimension must be even.")
    is_block_hor = dim_x == larger
    if is_block_hor == is_wall_hor:
        raise ValueError("The block and the wall must have perpendicular orientations.")
    if is_block_hor:
        if block.boundary_type("left") != "Z":
            raise ValueError("The left boundary of a horizontal block must be a Z-type boundary.")
    elif block.boundary_type("top") != "Z":
        raise ValueError("The top boundary of a vertical block must be a Z-type boundary.")
    if wall_pos != smaller:
        raise ValueError("The wall position must be such that the block on the bottom/right side of the wall is a square.")
    common = set(topological_corners) & set(geometric_corners)
    if len(common) != 3:
        raise ValueError("The block must have 3 topological corners located at the geometric corners.")
    geom_not_topol = next(iter(set(geometric_corners) - common))
    topol_not_geom = next(iter(set(topological_corners) - common))
    expected_topol = (ul[0] + (wall_pos if is_block_hor else is_ul_stab_x * (smaller - 1)), ul[1] + (is_ul_stab_x * (smaller - 1) if is_block_hor else wall_pos), 0)
    if expected_topol != topol_not_geom:
        raise ValueError(f"The non-geometric topological corner should have been {expected_topol} but it is {topol_not_geom}.")
    expected_geom = max(geometric_corners, key=lambda x: x[0] + x[1] if is_ul_stab_x else (-x[0] + x[1]) * (-1) ** is_block_hor)
    if expected_geom != geom_not_topol:
        raise ValueError(f"The non-topological geometric corner should have been {expected_geom} but it is {geom_not_topol}.")
    expected_z = PauliOperator("Z" * smaller, block.boundary_qubits("left") if is_block_hor else block.boundary_qubits("top"))
    if block.logical_z_operators[0] != expected_z:
        raise ValueError("The Z logical operator is not located at the expected position. It needs to be on the left for a horizontal block and on the top for a vertical block.")
    x_qubits = block.logical_x_operators[0].data_qubits
    straight = all(q[0] == x_qubits[0][0] for q in x_qubits) or all(q[1] == x_qubits[0][1] for q in x_qubits)
    if not (straight and len(x_qubits) == smaller + 1):
        raise ValueError("The X logical operator is not located at the expected position. It needs to be straight and with the smallest distance possible given the geometry of the block.")


# ---------------------------------------------------------------------------------------------
# Shared pieces (y_wall_out_utilities.py)
# ---------------------------------------------------------------------------------------------


def qubit_mapper(q: Coord, move_direction) -> Coord:
    return tuple(map(sum, zip(q, direction_to_coord(move_direction, q[2]), strict=True)))


def move_logicals_and_append_evolution_and_updates(step: InterpretationStep, block: RotatedSurfaceCode, move_direction: DiagonalDirection) -> tuple[PauliOperator, PauliOperator]:
    old_x, old_z = block.logical_x_operators[0], block.logical_z_operators[0]
    new_x = PauliOperator(pauli=old_x.pauli, data_qubits=[qubit_mapper(q, move_direction) for q in old_x.data_qubits])
    new_z = PauliOperator(pauli=old_z.pauli, data_qubits=[qubit_mapper(q, move_direction) for q in old_z.data_qubits])
    step.logical_x_evolution[new_x.uuid] = (old_x.uuid,)
    step.logical_z_evolution[new_z.uuid] = (old_z.uuid,)
    step.update_logical_operator_updates_MUT("X", new_x.uuid, (), inherit_updates=True)
    step.update_logical_operator_updates_MUT("Z", new_z.uuid, (), inherit_updates=True)
    return new_x, new_z


def generate_teleportation_finalization_circuit_with_updates(step: InterpretationStep, new_block: RotatedSurfaceCode, anc_qubits_to_init: dict[str, list[Coord]], teleportation_qubit_pairs: Sequence[tuple[Coord, Coord]]) -> Circuit:
    gates: list[Circuit] = []
    for corrected, measured in teleportation_qubit_pairs:
        cbit = step.get_new_cbit_MUT(f"c_{measured}")
        cbit_channel = step.get_channel_MUT(f"{cbit[0]}_{cbit[1]}", channel_type="classical")
        measured_channel = step.get_channel_MUT(measured)
        if corrected in anc_qubits_to_init["X"]:
            name, update = "measure_z", "Z"
        elif corrected in anc_qubits_to_init["Z"]:
            name, update = "measure_x", "X"
        else:
            raise ValueError("The ancilla qubit was not found in the initialization.")
        gates.append(Circuit(name, channels=[measured_channel, cbit_channel]))
        ops = new_block.logical_z_operators if update == "Z" else new_block.logical_x_operators
        for op in [o for o in ops if corrected in o.data_qubits]:
            step.update_logical_operator_updates_MUT(update, op.uuid, (cbit,), False)
        for stab in [s for s in new_block.stabilizers if s.pauli[0] == update and corrected in s.data_qubits]:
            step.stabilizer_updates[stab.uuid] = step.stabilizer_updates.get(stab.uuid, ()) + (cbit,)
    return Circuit("teleportation finalization", circuit=[gates])


def _cx_layers(step: InterpretationStep, name: str, cnots: Sequence[Sequence[tuple[Coord, Coord]]]) -> Circuit:
    return Circuit(name=name, circuit=[[Circuit("cx", channels=[step.get_channel_MUT(q) for q in pair]) for pair in layer] for layer in cnots])


def _boundary_of_weight_2(block: RotatedSurfaceCode, stab: Stabilizer) -> Optional[Direction]:
    if len(stab.pauli) != 2:
        return None
    directions = [set(d for d in Direction if q in block.boundary_qubits(d)) for q in stab.data_qubits]
    return list(directions[0] & directions[1])[0]


def get_final_block_syndrome_measurement_cnots_circuit(block: RotatedSurfaceCode, step: InterpretationStep, is_swap_then_qec: bool = False, idle_side_directions: Optional[DiagonalDirection] = None, anc_qubits_to_init: Optional[dict[str, list[Coord]]] = None) -> Circuit:
    """The CNOTs of a syndrome round of the configuration-3 block in five layers (regions 1
    and 2 on one schedule, 3 and 4 on the other, some steps a layer later), led for a
    swap-then-QEC step by a layer that swaps the data onto the block's sites."""
    config, pivot_corners = block.config_and_pivot_corners
    if config != 3:
        raise ValueError("The final block of Y-wall-out must have type-3 corner configuration.")
    starting = find_relative_diagonal_direction(block.upper_left_qubit, pivot_corners[3])
    z_124 = FourBodySchedule.Z if block.is_horizontal else FourBodySchedule.N
    z_3 = z_124.opposite_schedule()
    x_12 = z_124.opposite_schedule()
    x_34 = x_12.opposite_schedule()
    cnots: list[list[tuple[Coord, Coord]]] = [[] for _ in range(5)]
    if is_swap_then_qec:
        vec = direction_to_coord(idle_side_directions, 0)
        for q in block.data_qubits:
            current = tuple(map(sum, zip(q, vec, strict=True)))
            if q in anc_qubits_to_init["X"]:
                pair = (q, current)
            elif q in anc_qubits_to_init["Z"]:
                pair = (current, q)
            else:
                raise ValueError("The ancilla qubit was not found.")
            cnots[0].append(pair)
    for stab in block.stabilizers:
        p = pauli_type(stab)
        in_12, in_3, on_boundary_12, _, _, boundary = find_stabilizer_position_in_final_block(block, stab, pivot_corners)
        if p == "Z":
            schedule = z_3 if in_3 else z_124
        else:
            schedule = x_12 if in_12 else x_34
        detailed = DetailedSchedule.from_schedule_and_direction(schedule, starting)
        qubits = detailed.get_stabilizer_qubits(stab) if len(stab.data_qubits) == 4 else detailed.get_stabilizer_qubits(stab, boundary)
        for i, q in enumerate(qubits):
            if is_swap_then_qec and i == 0 or q is None:
                continue
            if p == "Z":
                pair = (q, stab.ancilla_qubits[0])
            elif p == "X":
                pair = (stab.ancilla_qubits[0], q)
            else:
                raise ValueError("Unknown stabilizer type.")
            layer = i + 1 if i > 0 and (not in_12 and not on_boundary_12 or (not in_12 and on_boundary_12 and i == 3)) else i
            cnots[layer].append(pair)
    return _cx_layers(step, "Final block syndrome measurement CNOT circuit", cnots)


def find_stabilizer_position_in_final_block(block: RotatedSurfaceCode, stab: Stabilizer, pivot_corners: Sequence[Coord]) -> tuple[bool, bool, bool, bool, bool, Optional[Direction]]:
    """Where a stabilizer of the configuration-3 block lies: in regions 1 or 2 (the triangles
    by the middle corner), in region 3, in region 4 on the edge of 1 or 2, on the wall, on
    the Hadamard side; and a weight-2 stabilizer's boundary."""
    long_end, middle, _, short_end = pivot_corners
    long_idx = 0 if block.is_horizontal else 1
    short_idx = 1 if block.is_horizontal else 0

    def region_1(q) -> bool:
        dshort = q[short_idx] - short_end[short_idx] if middle[short_idx] > short_end[short_idx] else short_end[short_idx] - q[short_idx]
        dlong = middle[long_idx] - q[long_idx] + 1 if middle[long_idx] < long_end[long_idx] else q[long_idx] - middle[long_idx] + 1
        return 0 <= dshort <= dlong and dlong >= 0

    def region_2(q) -> bool:
        dshort = q[short_idx] - middle[short_idx] if middle[short_idx] < short_end[short_idx] else middle[short_idx] - q[short_idx]
        dlong = middle[long_idx] - q[long_idx] if middle[long_idx] < long_end[long_idx] else q[long_idx] - middle[long_idx]
        return 0 <= dshort <= dlong and dlong >= 0

    def region_3(q) -> bool:
        dshort = q[short_idx] - middle[short_idx] if middle[short_idx] < short_end[short_idx] else middle[short_idx] - q[short_idx]
        dlong = q[long_idx] - middle[long_idx] if middle[long_idx] < long_end[long_idx] else middle[long_idx] - q[long_idx]
        return 0 <= dshort <= dlong and dlong >= 0

    n = len(stab.data_qubits)
    in_1 = [q for q in stab.data_qubits if region_1(q)]
    in_2 = [q for q in stab.data_qubits if region_2(q)]
    in_3 = [q for q in stab.data_qubits if region_3(q)]
    is_1, is_2, is_3 = len(in_1) == n, len(in_2) == n, len(in_3) == n
    is_4 = not (is_1 or is_2 or is_3)
    is_4_on_boundary_12 = is_4 and (n - len(in_1) == 1 or n - len(in_2) == 1)
    top_left = max(stab.data_qubits, key=lambda x: -x[0] - x[1])
    bottom_right = max(stab.data_qubits, key=lambda x: x[0] + x[1])
    compare = top_left if middle[long_idx] < long_end[long_idx] else bottom_right
    on_wall = middle[long_idx] == compare[long_idx]
    on_hadamard_side = compare[long_idx] > middle[long_idx] if long_end[long_idx] > middle[long_idx] else compare[long_idx] < middle[long_idx]
    return is_1 or is_2, is_3, is_4_on_boundary_12, on_wall, on_hadamard_side, _boundary_of_weight_2(block, stab)


# ---------------------------------------------------------------------------------------------
# The initial syndrome round (y_wall_out_initial_syndrome_measurement.py)
# ---------------------------------------------------------------------------------------------


def _ancilla_reset(step: InterpretationStep, block: RotatedSurfaceCode) -> Circuit:
    return Circuit(name="Initialization of syndrome measurement ancilla", circuit=[[Circuit(f"reset_{'0' if pauli_type(s) == 'Z' else '+'}", channels=[step.get_channel_MUT(s.ancilla_qubits[0])]) for s in block.stabilizers]])


def y_wall_out_initial_syndrome_measurement(step: InterpretationStep, block: RotatedSurfaceCode, same_timeslice: bool, debug_mode: bool) -> InterpretationStep:
    reset = _ancilla_reset(step, block)
    cnots = get_initial_block_syndrome_measurement_cnots_circuit(block, step)
    measure, cbits = generate_syndrome_measurement_circuit_and_cbits(step, block)
    step.append_circuit_MUT(Circuit(name="Y wall out - Initial block syndrome measurement", circuit=(reset, cnots, measure)), same_timeslice=same_timeslice)
    generate_and_append_block_syndromes_and_detectors(step, block, cbits)
    return step


def get_initial_block_syndrome_measurement_cnots_circuit(block: RotatedSurfaceCode, step: InterpretationStep) -> Circuit:
    """The CNOTs of a syndrome round of the configuration-4 block in five layers: region 4 (by
    the wall) on the other X schedule, and the steps nearer the long end a layer later."""
    config, pivot_corners = block.config_and_pivot_corners
    if config != 4:
        raise ValueError("The initial block of Y-wall-out must have type-4 corner configuration.")
    starting = find_relative_diagonal_direction(block.upper_left_qubit, pivot_corners[0])
    z_schedule = FourBodySchedule.Z if block.is_horizontal else FourBodySchedule.N
    x_123 = z_schedule.opposite_schedule()
    x_4 = x_123.opposite_schedule()
    cnots: list[list[tuple[Coord, Coord]]] = [[] for _ in range(5)]
    for stab in block.stabilizers:
        p = pauli_type(stab)
        in_4, on_boundary_12_in_4, closer_long_end, boundary = find_stabilizer_position_in_initial_block(block, stab, pivot_corners)
        if in_4:
            schedule = x_4 if p == "X" else z_schedule
        else:
            schedule = x_123 if p == "X" else z_schedule
        detailed = DetailedSchedule.from_schedule_and_direction(schedule, starting)
        qubits = detailed.get_stabilizer_qubits(stab) if len(stab.data_qubits) == 4 else detailed.get_stabilizer_qubits(stab, boundary)
        for i, q in enumerate(qubits):
            if q is None:
                continue
            if p == "Z":
                pair = (q, stab.ancilla_qubits[0])
            elif p == "X":
                pair = (stab.ancilla_qubits[0], q)
            else:
                raise ValueError("Unknown stabilizer type.")
            layer = i + 1 if closer_long_end or (in_4 and not on_boundary_12_in_4) or (in_4 and on_boundary_12_in_4 and i in [2, 3]) else i
            cnots[layer].append(pair)
    return _cx_layers(step, "Initial block syndrome measurement CNOT circuit", cnots)


def find_stabilizer_position_in_initial_block(block: RotatedSurfaceCode, stab: Stabilizer, pivot_corners: Sequence[Coord]) -> tuple[bool, bool, bool, Optional[Direction]]:
    long_end, middle = pivot_corners[0], pivot_corners[1]
    long_idx = 0 if block.is_horizontal else 1
    short_idx = 1 if block.is_horizontal else 0

    def region_1(q) -> bool:
        dshort = middle[short_idx] - q[short_idx] if middle[short_idx] > long_end[short_idx] else q[short_idx] - middle[short_idx]
        dlong = middle[long_idx] - q[long_idx] if middle[long_idx] < long_end[long_idx] else q[long_idx] - middle[long_idx]
        return 0 <= dshort <= dlong and dlong >= 0

    def region_2(q) -> bool:
        dshort = q[short_idx] - long_end[short_idx] if middle[short_idx] > long_end[short_idx] else long_end[short_idx] - q[short_idx]
        dlong = middle[long_idx] - q[long_idx] - 1 if middle[long_idx] < long_end[long_idx] else q[long_idx] - middle[long_idx] - 1
        return 0 <= dshort <= dlong and dlong >= 0

    def region_3(q) -> bool:
        dshort = q[short_idx] - long_end[short_idx] if middle[short_idx] > long_end[short_idx] else long_end[short_idx] - q[short_idx]
        dlong = q[long_idx] - middle[long_idx] if middle[long_idx] < long_end[long_idx] else middle[long_idx] - q[long_idx]
        return 0 <= dshort <= dlong and dlong >= 0

    n = len(stab.data_qubits)
    in_1 = [q for q in stab.data_qubits if region_1(q)]
    in_2 = [q for q in stab.data_qubits if region_2(q)]
    in_3 = [q for q in stab.data_qubits if region_3(q)]
    is_4 = not (len(in_1) == n or len(in_2) == n or len(in_3) == n)
    is_4_on_boundary_12 = is_4 and (n - len(in_1) == 1 or n - len(in_2) == 1)
    top_left = max(stab.data_qubits, key=lambda x: -x[0] - x[1])
    bottom_right = max(stab.data_qubits, key=lambda x: x[0] + x[1])
    compare = top_left if middle[long_idx] < long_end[long_idx] else bottom_right
    closer = middle[long_idx] <= compare[long_idx] <= long_end[long_idx] if long_end[long_idx] > middle[long_idx] else long_end[long_idx] <= compare[long_idx] <= middle[long_idx]
    return is_4, is_4_on_boundary_12, closer, _boundary_of_weight_2(block, stab)


# ---------------------------------------------------------------------------------------------
# The Y wall and the Hadamards (y_wall_measurement_hadamard.py)
# ---------------------------------------------------------------------------------------------


def y_wall_out_measurement_and_hadamard(step: InterpretationStep, block: RotatedSurfaceCode, wall_position: int, wall_orientation: Orientation, same_timeslice: bool, debug_mode: bool) -> InterpretationStep:
    """Measure the wall in Y and Hadamard the far square: the far square's stabilizers swap
    type, the stabilizers across the wall join the near square's to the far square's sites
    (corrected by the wall's bits and a sign), and the X logical moves to the near square,
    picking up the Z logical's corrections, the near square's syndromes, the wall's bits and
    the sign of i**n for n wall qubits."""
    is_wall_hor = wall_orientation == Orientation.HORIZONTAL
    is_top_left_x = block.upper_left_4body_stabilizer.pauli[0] == "X"
    qubits_to_measure, qubits_to_idle, qubits_to_hadamard = find_qubit_sets(block, wall_position, is_wall_hor)
    circuit, y_cbits = measure_y_and_hadamard_circuit(block, qubits_to_measure, qubits_to_hadamard, step)
    step.append_circuit_MUT(circuit, same_timeslice=same_timeslice)
    new_stabilizers = find_new_stabilizers_and_define_evolution_and_updates(step, block, qubits_to_measure, qubits_to_idle, qubits_to_hadamard, y_cbits)
    new_x, jump_stabilizers = find_new_x_logical_operator(block, is_top_left_x, qubits_to_idle)
    step.logical_x_evolution.update({new_x.uuid: (block.logical_x_operators[0].uuid,)})
    step.update_logical_operator_updates_MUT("X", new_x.uuid, step.logical_z_operator_updates.get(block.logical_z_operators[0].uuid, ()) + step.retrieve_cbits_from_stabilizers(jump_stabilizers, block) + tuple(y_cbits) + (1 * (len(y_cbits) % 4 == 1),), inherit_updates=True)
    new_block = RotatedSurfaceCode(stabilizers=new_stabilizers, logical_x_operators=(new_x,), logical_z_operators=block.logical_z_operators, unique_label=block.unique_label, skip_validation=not debug_mode)
    step.update_block_history_and_evolution_MUT((new_block,), (block,))
    return step


def find_new_stabilizers_and_define_evolution_and_updates(step: InterpretationStep, block: RotatedSurfaceCode, qubits_to_measure: Sequence[Coord], qubits_to_idle: Sequence[Coord], qubits_to_hadamard: Sequence[Coord], y_meas_cbits: Sequence[Cbit]) -> list[Stabilizer]:
    ul_idle = min(qubits_to_idle, key=lambda q: q[0] + q[1])
    ul_had = min(qubits_to_hadamard, key=lambda q: q[0] + q[1])
    idle_to_had = Direction.from_vector((ul_had[0] - ul_idle[0], ul_had[1] - ul_idle[1], 0))
    old_idle = [s for s in block.stabilizers if set(s.data_qubits).issubset(qubits_to_idle)]
    new_idle = old_idle
    old_had = [s for s in block.stabilizers if set(s.data_qubits).issubset(qubits_to_hadamard)]
    swapped = [Stabilizer(pauli="".join({"X": "Z", "Z": "X"}[p] for p in s.pauli), data_qubits=s.data_qubits, ancilla_qubits=s.ancilla_qubits) for s in old_had]
    # A weight-4 stabilizer of the far square takes its qubits in the order of the near
    # square's stabilizer it mirrors.
    new_had = [Stabilizer(data_qubits=next([qubit_mapper(q, idle_to_had) for q in init.data_qubits] for init in block.stabilizers if set(qubit_mapper(q, idle_to_had) for q in init.data_qubits) == set(s.data_qubits)) if len(s.data_qubits) == 4 else s.data_qubits, ancilla_qubits=s.ancilla_qubits, pauli=s.pauli) for s in swapped]
    wall_idle = [s for s in block.stabilizers if set(s.data_qubits) & set(qubits_to_measure) and set(s.data_qubits) & set(qubits_to_idle) and len(s.data_qubits) == 4]
    wall_had = [s for s in block.stabilizers if set(s.data_qubits) & set(qubits_to_measure) and set(s.data_qubits) & set(qubits_to_hadamard) and len(s.data_qubits) == 4]
    wall_had_ordered = [h for i in wall_idle for h in wall_had if len(set(h.data_qubits) & set(i.data_qubits)) == 2]
    v = idle_to_had.to_vector()
    new_wall = [Stabilizer(s.pauli, [q if q not in qubits_to_measure else (q[0] + v[0], q[1] + v[1], q[2]) for q in s.data_qubits], ancilla_qubits=s.ancilla_qubits) for s in wall_idle]
    evolution: dict[str, tuple[str, ...]] = {}
    evolution |= {new.uuid: (old.uuid,) for old, new in zip(old_had + old_idle, new_had + new_idle, strict=True)}
    evolution |= {new.uuid: (i.uuid, h.uuid) for i, h, new in zip(wall_idle, wall_had_ordered, new_wall, strict=True)}
    step.stabilizer_evolution.update(evolution)
    for new, old in zip(new_wall, wall_idle, strict=True):
        measured = set(old.data_qubits) & set(qubits_to_measure)
        cbits = tuple(c for c, q in zip(y_meas_cbits, qubits_to_measure) if q in measured)
        step.stabilizer_updates[new.uuid] = step.stabilizer_updates.get(new.uuid, ()) + cbits + (1,)
    return new_had + new_idle + new_wall


def measure_y_and_hadamard_circuit(block: RotatedSurfaceCode, qubits_to_measure: Sequence[Coord], qubits_to_hadamard: Sequence[Coord], step: InterpretationStep) -> tuple[Circuit, list[Cbit]]:
    cbits = [step.get_new_cbit_MUT(f"c_{q}") for q in qubits_to_measure]
    cbit_channels = [step.get_channel_MUT(f"{c[0]}_{c[1]}", channel_type="classical") for c in cbits]
    wall = Circuit(name=f"Measure wall of qubits in the Y basis for block {block.unique_label}", circuit=[[Circuit("measure_y", channels=[step.get_channel_MUT(q), ch]) for q, ch in zip(qubits_to_measure, cbit_channels, strict=True)]])
    hadamards = Circuit(name=f"transversal hadamard on the data qubits for the y_wall_out operation of block {block.unique_label}", circuit=[[Circuit("h", channels=[step.get_channel_MUT(q)]) for q in qubits_to_hadamard]])
    return Circuit(name="y_wall_out - y wall measurement and hadamard", circuit=((wall, hadamards),)), cbits


def find_new_x_logical_operator(block: RotatedSurfaceCode, is_top_left_bulk_stab_x: bool, qubits_to_idle: Sequence[Coord]) -> tuple[PauliOperator, tuple[Stabilizer, ...]]:
    side = {(False, False): Direction.RIGHT, (False, True): Direction.LEFT, (True, False): Direction.BOTTOM, (True, True): Direction.TOP}[(block.is_horizontal, is_top_left_bulk_stab_x)]
    qubits = [q for q in block.boundary_qubits(side) if q in qubits_to_idle]
    jump = tuple(s for s in block.stabilizers if set(s.data_qubits) & set(qubits_to_idle))
    return PauliOperator(pauli="X" * len(qubits), data_qubits=qubits), jump


def find_qubit_sets(block: RotatedSurfaceCode, wall_position: int, is_wall_hor: bool) -> tuple[list[Coord], list[Coord], list[Coord]]:
    """The wall's qubits, the near square's (kept idle) and the far square's (Hadamarded)."""
    had_vector = (0, 1) if is_wall_hor else (1, 0)
    idle_vector = (0, -1) if is_wall_hor else (-1, 0)
    to_measure = [(q[0], q[1] + wall_position, 0) for q in block.boundary_qubits("top")] if is_wall_hor else [(q[0] + wall_position, q[1], 0) for q in block.boundary_qubits("left")]
    to_idle = [(q[0] + idle_vector[0] * d, q[1] + idle_vector[1] * d, q[2]) for q in to_measure for d in range(1, wall_position + 1)]
    to_hadamard = [(q[0] + had_vector[0] * d, q[1] + had_vector[1] * d, q[2]) for q in to_measure for d in range(1, max(block.size) - wall_position)]
    return to_measure, to_idle, to_hadamard


# ---------------------------------------------------------------------------------------------
# The recombination (y_wall_out_recombination_swap_then_qec.py)
# ---------------------------------------------------------------------------------------------


def y_wall_out_recombination_swap_then_qec(step: InterpretationStep, block: RotatedSurfaceCode, initial_block: RotatedSurfaceCode, qubits_to_idle: Sequence[Coord], qubits_measured: Sequence[Coord], qubits_to_had: Sequence[Coord], same_timeslice: bool, debug_mode: bool) -> InterpretationStep:
    """A swap-then-QEC step that moves the two squares diagonally towards each other (the near
    one along the idle side's direction, the far one along the Hadamard side's), refilling the
    wall's sites, into one block of configuration 3."""
    idle_stabs, had_stabs, wall_stabs, idle_dir, had_dir = get_idle_hadamard_info(block, qubits_to_had, qubits_to_idle)
    new_stabilizers = find_new_stabilizers_and_append_evolution_and_updates(step, qubits_to_idle, idle_stabs, had_stabs, wall_stabs, idle_dir, had_dir)
    new_x, new_z = move_logicals_and_append_evolution_and_updates(step, block, idle_dir)
    recombined = RotatedSurfaceCode(stabilizers=new_stabilizers, logical_x_operators=(new_x,), logical_z_operators=(new_z,), unique_label=block.unique_label, skip_validation=not debug_mode)
    wall_data = get_wall_data_qubit_init(initial_block, qubits_measured)
    anc_idle, data_idle, pairs_idle = find_swap_then_qec_qubit_initializations(idle_stabs, idle_dir)
    anc_had, data_had, pairs_had = find_swap_then_qec_qubit_initializations(had_stabs, had_dir)
    reset = Circuit(name="Initialization of qubits for first swap-then-qec", circuit=[[Circuit(f"reset_{'0' if p == 'Z' else '+'}", channels=[step.get_channel_MUT(q)]) for p in ["X", "Z"] for q in wall_data[p] + data_idle[p] + data_had[p] + anc_had[p] + anc_idle[p]]])
    cnots = get_final_block_syndrome_measurement_first_swap_then_qec_cnots_circuit(step, recombined, qubits_to_idle, qubits_to_had, idle_dir, had_dir, anc_idle, anc_had)
    all_anc = {"X": anc_idle["X"] + anc_had["X"], "Z": anc_idle["Z"] + anc_had["Z"]}
    teleportation = generate_teleportation_finalization_circuit_with_updates(step, recombined, all_anc, pairs_idle + pairs_had)
    measure, cbits = generate_syndrome_measurement_circuit_and_cbits(step, recombined)
    step.append_circuit_MUT(Circuit(name="swap-then-qec to recombine block", circuit=construct_padded_circuit_time_sequence(((reset,), (cnots,), (measure, teleportation)))), same_timeslice=same_timeslice)
    step.update_block_history_and_evolution_MUT((recombined,), (block,))
    generate_and_append_block_syndromes_and_detectors(step, recombined, cbits)
    return step


def get_idle_hadamard_info(block: RotatedSurfaceCode, qubits_to_had: Sequence[Coord], qubits_to_idle: Sequence[Coord]) -> tuple[list[Stabilizer], list[Stabilizer], list[Stabilizer], DiagonalDirection, DiagonalDirection]:
    is_top_left_x = pauli_type(block.upper_left_4body_stabilizer) == "X"
    idle = [s for s in block.stabilizers if set(s.data_qubits).issubset(qubits_to_idle)]
    had = [s for s in block.stabilizers if set(s.data_qubits).issubset(qubits_to_had)]
    wall = [s for s in block.stabilizers if s not in idle + had]
    idle_dir, had_dir = {
        (Orientation.VERTICAL, False): (DiagonalDirection.BOTTOM_RIGHT, DiagonalDirection.TOP_RIGHT),
        (Orientation.VERTICAL, True): (DiagonalDirection.BOTTOM_LEFT, DiagonalDirection.TOP_LEFT),
        (Orientation.HORIZONTAL, True): (DiagonalDirection.TOP_RIGHT, DiagonalDirection.TOP_LEFT),
        (Orientation.HORIZONTAL, False): (DiagonalDirection.BOTTOM_RIGHT, DiagonalDirection.BOTTOM_LEFT),
    }[(block.orientation, is_top_left_x)]
    return idle, had, wall, idle_dir, had_dir


def find_new_stabilizers_and_append_evolution_and_updates(step: InterpretationStep, qubits_to_idle: Sequence[Coord], idle_side_stabilizers: Sequence[Stabilizer], had_side_stabilizers: Sequence[Stabilizer], wall_stabilizers: Sequence[Stabilizer], idle_side_directions: DiagonalDirection, had_side_directions: DiagonalDirection) -> list[Stabilizer]:
    new_stabilizers: list[Stabilizer] = []
    evolution: dict[str, tuple[str, ...]] = {}
    for s in idle_side_stabilizers:
        new = Stabilizer(s.pauli, [qubit_mapper(q, idle_side_directions) for q in s.data_qubits], ancilla_qubits=[qubit_mapper(q, idle_side_directions) for q in s.ancilla_qubits])
        new_stabilizers.append(new)
        evolution[new.uuid] = (s.uuid,)
    for s in had_side_stabilizers:
        new = Stabilizer(s.pauli, [qubit_mapper(q, had_side_directions) for q in s.data_qubits], ancilla_qubits=[qubit_mapper(q, had_side_directions) for q in s.ancilla_qubits])
        new_stabilizers.append(new)
        evolution[new.uuid] = (s.uuid,)
    for s in wall_stabilizers:
        new = Stabilizer(s.pauli, [qubit_mapper(q, idle_side_directions) if q in qubits_to_idle else qubit_mapper(q, had_side_directions) for q in s.data_qubits], ancilla_qubits=[qubit_mapper(q, idle_side_directions) for q in s.ancilla_qubits])
        new_stabilizers.append(new)
        evolution[new.uuid] = (s.uuid,)
    _carry_updates(step, evolution)
    return new_stabilizers


def _carry_updates(step: InterpretationStep, evolution: dict[str, tuple[str, ...]]) -> None:
    for new_uuid, (old_uuid, *_) in evolution.items():
        if old_uuid in step.stabilizer_updates:
            step.stabilizer_updates[new_uuid] = step.stabilizer_updates.pop(old_uuid)
    step.stabilizer_evolution.update(evolution)


def get_wall_data_qubit_init(y_wall_initial_block: RotatedSurfaceCode, qubits_to_measure: Sequence[Coord]) -> dict[str, list[Coord]]:
    """The wall's sites the recombined block puts data on (all but one), each reset in the
    basis of the weight-4 stabilizer whose far corner it was."""
    is_top_left_x = pauli_type(y_wall_initial_block.upper_left_4body_stabilizer) == "X"
    pick, key = {
        (Orientation.VERTICAL, False): (lambda qs: min(qs, key=lambda x: x[0]), lambda x: x[0] + x[1]),
        (Orientation.VERTICAL, True): (lambda qs: max(qs, key=lambda x: x[0]), lambda x: -x[0] + x[1]),
        (Orientation.HORIZONTAL, True): (lambda qs: max(qs, key=lambda x: x[1]), lambda x: x[0] - x[1]),
        (Orientation.HORIZONTAL, False): (lambda qs: min(qs, key=lambda x: x[1]), lambda x: x[0] + x[1]),
    }[(y_wall_initial_block.orientation, is_top_left_x)]
    exclude = pick(qubits_to_measure)
    out: dict[str, list[Coord]] = {"X": [], "Z": []}
    for q in qubits_to_measure:
        if q == exclude:
            continue
        stab = next(s for s in y_wall_initial_block.stabilizers if max(s.data_qubits, key=key) == q and len(s.data_qubits) == 4)
        out[stab.pauli[0]].append(q)
    return out


def get_final_block_syndrome_measurement_first_swap_then_qec_cnots_circuit(step: InterpretationStep, recombined_block: RotatedSurfaceCode, qubits_to_idle: Sequence[Coord], qubits_to_had: Sequence[Coord], idle_side_directions: DiagonalDirection, had_side_directions: DiagonalDirection, anc_qubits_to_init_idle: dict[str, list[Coord]], anc_qubits_to_init_had: dict[str, list[Coord]]) -> Circuit:
    config, pivot_corners = recombined_block.config_and_pivot_corners
    if config != 3:
        raise ValueError("The final block of Y-wall-out must have type-3 corner configuration.")
    long_end, angle = pivot_corners[0], pivot_corners[2]
    top_left = recombined_block.upper_left_qubit
    z_124 = FourBodySchedule.Z if recombined_block.is_horizontal else FourBodySchedule.N
    z_3 = z_124.opposite_schedule()
    x_12 = z_124.opposite_schedule()
    x_34 = x_12.opposite_schedule()
    cnots: list[list[tuple[Coord, Coord]]] = [[] for _ in range(5)]
    for qubits, direction, anc in ((qubits_to_had, had_side_directions, anc_qubits_to_init_had), (qubits_to_idle, idle_side_directions, anc_qubits_to_init_idle)):
        vec = direction_to_coord(direction, 0)
        for q in qubits:
            a = tuple(map(sum, zip(q, vec, strict=True)))
            if a in anc["Z"]:
                pair = (q, a)
            elif a in anc["X"]:
                pair = (a, q)
            else:
                raise ValueError("The ancilla qubit was not found.")
            cnots[0].append(pair)
    for stab in recombined_block.stabilizers:
        p = pauli_type(stab)
        in_12, in_3, _, on_wall, on_had_side, boundary = find_stabilizer_position_in_final_block(recombined_block, stab, pivot_corners)
        on_idle_side = not (on_wall or on_had_side)
        if p == "Z":
            schedule = z_3 if in_3 else z_124
        else:
            schedule = x_12 if in_12 else x_34
        starting = find_relative_diagonal_direction(top_left, angle if on_had_side else long_end)
        detailed = DetailedSchedule.from_schedule_and_direction(schedule, starting)
        qubits = detailed.get_stabilizer_qubits(stab) if len(stab.data_qubits) == 4 else detailed.get_stabilizer_qubits(stab, boundary)
        for i, q in enumerate(qubits):
            if not on_wall and i == 0 or q is None:
                continue
            a = stab.ancilla_qubits[0]
            if p == "Z":
                pair = (q, a)
            elif p == "X":
                pair = (a, q)
            else:
                raise ValueError("Unknown stabilizer type.")
            later = on_wall or (on_idle_side and not in_12 and (p == "Z" and i in [2, 3] or (p == "X" and i in [1, 2, 3]))) or (in_3 and (p == "Z" and i in [1, 2, 3] or (p == "X" and i == 3)))
            cnots[i + 1 if later else i].append(pair)
    return _cx_layers(step, "First SWAP-then-QEC final block syndrome measurement CNOT circuit", cnots)


# ---------------------------------------------------------------------------------------------
# The final swap and rounds (y_wall_out_final_swap_then_qec.py, y_wall_out_final_syndrome_measurement.py)
# ---------------------------------------------------------------------------------------------


def y_wall_out_final_swap_then_qec(step: InterpretationStep, block: RotatedSurfaceCode, idle_side_directions: DiagonalDirection, same_timeslice: bool, debug_mode: bool) -> InterpretationStep:
    """A swap-then-QEC step that moves the recombined block back, against the idle side's
    direction, and gives it the syndrome circuits of its configuration."""
    back = idle_side_directions.opposite()
    new_stabilizers = move_stabilizers_and_append_evolution_and_updates(step, block.stabilizers, back)
    new_x, new_z = move_logicals_and_append_evolution_and_updates(step, block, back)
    final_block = generate_final_block_with_syndrome_circuits(block.unique_label, new_stabilizers, new_x, new_z, debug_mode)
    anc, data, pairs = find_swap_then_qec_qubit_initializations(block.stabilizers, back)
    reset = Circuit(name="Initialization of qubits for second swap-then-qec", circuit=[[Circuit(f"reset_{'0' if p == 'Z' else '+'}", channels=[step.get_channel_MUT(q)]) for p in ["X", "Z"] for q in data[p] + anc[p]]])
    cnots = get_final_block_syndrome_measurement_cnots_circuit(final_block, step, True, idle_side_directions, anc)
    teleportation = generate_teleportation_finalization_circuit_with_updates(step, final_block, anc, pairs)
    measure, cbits = generate_syndrome_measurement_circuit_and_cbits(step, final_block)
    step.append_circuit_MUT(Circuit(name="Second swap-then-QEC for Y-wall-out recombination", circuit=construct_padded_circuit_time_sequence(((reset,), (cnots,), (measure, teleportation)))), same_timeslice=same_timeslice)
    step.update_block_history_and_evolution_MUT((final_block,), (block,))
    generate_and_append_block_syndromes_and_detectors(step, final_block, cbits)
    return step


def generate_final_block_with_syndrome_circuits(block_unique_label: str, new_stabilizers: Sequence[Stabilizer], new_logical_x: PauliOperator, new_logical_z: PauliOperator, debug_mode: bool) -> RotatedSurfaceCode:
    mock = RotatedSurfaceCode(unique_label=block_unique_label, stabilizers=new_stabilizers, logical_x_operators=(new_logical_x,), logical_z_operators=(new_logical_z,))
    config, pivot_corners = mock.config_and_pivot_corners
    if config != 3:
        raise ValueError(f"New block must have type-3 corner configuration, found type-{config}.")
    starting = find_relative_diagonal_direction(mock.upper_left_qubit, pivot_corners[3])
    circuits, stc = generate_syndrome_extraction_circuits(mock, starting)
    return RotatedSurfaceCode(stabilizers=mock.stabilizers, logical_x_operators=mock.logical_x_operators, logical_z_operators=mock.logical_z_operators, syndrome_circuits=circuits, stabilizer_to_circuit=stc, unique_label=block_unique_label, skip_validation=not debug_mode)


def move_stabilizers_and_append_evolution_and_updates(step: InterpretationStep, stabilizers: Sequence[Stabilizer], move_direction: DiagonalDirection) -> list[Stabilizer]:
    new_stabilizers: list[Stabilizer] = []
    evolution: dict[str, tuple[str, ...]] = {}
    for s in stabilizers:
        new = Stabilizer(s.pauli, [qubit_mapper(q, move_direction) for q in s.data_qubits], ancilla_qubits=[qubit_mapper(q, move_direction) for q in s.ancilla_qubits])
        new_stabilizers.append(new)
        evolution[new.uuid] = (s.uuid,)
    _carry_updates(step, evolution)
    return new_stabilizers


def y_wall_out_final_qec_rounds(step: InterpretationStep, block: RotatedSurfaceCode, same_timeslice: bool, debug_mode: bool) -> InterpretationStep:
    n_rounds = min(block.size) - 2
    step.begin_composite_operation_session_MUT(same_timeslice=same_timeslice, circuit_name=f"final {n_rounds} syndrome measurement round(s) on block {block.unique_label}")
    for _ in range(n_rounds):
        reset = _ancilla_reset(step, block)
        cnots = get_final_block_syndrome_measurement_cnots_circuit(block, step)
        measure, cbits = generate_syndrome_measurement_circuit_and_cbits(step, block)
        step.append_circuit_MUT(Circuit(name="one round of final syndrome measurement", circuit=construct_padded_circuit_time_sequence(((reset,), (cnots,), (measure,)))), same_timeslice=False)
        generate_and_append_block_syndromes_and_detectors(step, block, cbits)
    circuit = step.end_composite_operation_session_MUT()
    step.append_circuit_MUT(circuit, same_timeslice)
    return step

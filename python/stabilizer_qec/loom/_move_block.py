# Ported from Loom (entropicalabs/el-loom 0.4.0, src/loom_rotated_surface_code/applicator/move_block.py; Copyright
# 2024 Entropica Labs Pte Ltd, Apache License 2.0; see LICENSE in this directory), modified for stabilizer_qec: plain
# functions on this package's interpretation step, the diagonal-direction methods from _rsc_geometry.
"""Loom's MoveBlock: a rotated surface code block moved one site along a direction as two
diagonal swap-then-QEC steps. Each step resets the qubits the block moves onto, swaps the
data onto them with CNOTs woven into a round of syndrome extraction, and teleports the
stabilizers' old ancillas out with measurements that correct the next syndromes and the
logical operators."""

from __future__ import annotations

from typing import Optional, Sequence

from .circuit import Circuit
from .eka import DiagonalDirection, Direction, MoveBlock, Orientation, Stabilizer
from .interpreter import Cbit, InterpretationStep, generate_detectors, generate_syndromes
from .rotated_surface_code import RotatedSurfaceCode
from ._rsc_geometry import DetailedSchedule, construct_padded_circuit_time_sequence, direction_along_orientation, direction_to_coord, find_detailed_schedules, pauli_type, shift_block_towards_direction, update_qubit_coords

Coord = tuple[int, ...]


def move_block(step: InterpretationStep, operation: MoveBlock, same_timeslice: bool, debug_mode: bool) -> InterpretationStep:
    """Move the block one site towards the direction: two diagonal steps (top: top-left then
    top-right; right: top-right then bottom-right; left: top-left then bottom-left; bottom:
    bottom-left then bottom-right), each refused if a qubit would land on another block's."""
    block = step.get_block(operation.input_block_name)
    step.begin_composite_operation_session_MUT(same_timeslice=same_timeslice, circuit_name=f"Move block {block.unique_label} towards {operation.direction.value}")
    other_blocks = [b for b in step.get_blocks_at_index(-1) if b != block]
    occupied = [q for b in other_blocks for q in _loom_qubits(b)]
    current = block
    for diag in composite_direction(operation.direction):
        check_valid_move(occupied, _loom_qubits(current), diag)
        step = move_block_diagonally_via_swap_qec(step, current, diag, debug_mode)
        current = step.get_block(operation.input_block_name)
    circuit = step.end_composite_operation_session_MUT()
    step.append_circuit_MUT(circuit, same_timeslice)
    return step


def _loom_qubits(block: RotatedSurfaceCode) -> tuple[Coord, ...]:
    """Loom's ``Block.qubits``: the data and ancilla qubits through a set (of integer tuples,
    so in the same order in every run), which decides the qubit a refused move names."""
    ancillas = tuple(set(q for s in block.stabilizers for q in s.ancilla_qubits))
    return tuple(set(block.data_qubits + ancillas))


def move_block_diagonally_via_swap_qec(step: InterpretationStep, current_block: RotatedSurfaceCode, diag_direction: DiagonalDirection, debug_mode: bool) -> InterpretationStep:
    new_block, stab_ev, x_ev, z_ev = shift_block_towards_direction(current_block, diag_direction, debug_mode=debug_mode)
    step.stabilizer_evolution.update(stab_ev)
    step.logical_x_evolution.update(x_ev)
    step.logical_z_evolution.update(z_ev)
    for new_uuid, old_uuids in stab_ev.items():
        propagated: tuple = ()
        for old in old_uuids:
            propagated += step.stabilizer_updates.get(old, ())
        if propagated:
            step.stabilizer_updates[new_uuid] = propagated
    step.update_logical_operator_updates_MUT("X", new_block.logical_x_operators[0].uuid, (), True)
    step.update_logical_operator_updates_MUT("Z", new_block.logical_z_operators[0].uuid, (), True)
    anc_to_init, data_to_init, teleportation_pairs = find_swap_then_qec_qubit_initializations(current_block.stabilizers, diag_direction)
    reset = Circuit(name="Initialization of qubits for first swap-then-qec", circuit=[[Circuit(f"reset_{'0' if p == 'Z' else '+'}", channels=[step.get_channel_MUT(q)]) for p in ["X", "Z"] for q in anc_to_init[p] + data_to_init[p]]])
    schedules = find_detailed_schedules(new_block, diag_direction)
    cnots = get_swap_qec_cnots(step, new_block, diag_direction, schedules, anc_to_init, data_to_init)
    teleportation = generate_teleportation_measurement_circuit_with_updates(step, new_block, anc_to_init, teleportation_pairs)
    measurement, stab_measurements = generate_syndrome_measurement_circuit_and_cbits(step, new_block)
    circ = Circuit(name=f"move block {new_block.unique_label} towards {diag_direction.value}", circuit=construct_padded_circuit_time_sequence([(reset,), (cnots,), (teleportation, measurement)]))
    step.append_circuit_MUT(circ, same_timeslice=False)
    step.update_block_history_and_evolution_MUT((new_block,), (current_block,))
    generate_and_append_block_syndromes_and_detectors(step, new_block, stab_measurements)
    return step


def check_valid_move(occupied_qubits: Sequence[Coord], moving_qubits: Sequence[Coord], direction) -> None:
    for q in moving_qubits:
        new = tuple(a + b for a, b in zip(q, direction_to_coord(direction, q[-1]), strict=True))
        if new in occupied_qubits:
            raise ValueError(f"The move operation is invalid. The following qubit, {q}, is moving to an occupied qubit, {new}.")


def composite_direction(direction: Direction) -> tuple[DiagonalDirection, DiagonalDirection]:
    return {
        Direction.TOP: (DiagonalDirection.TOP_LEFT, DiagonalDirection.TOP_RIGHT),
        Direction.RIGHT: (DiagonalDirection.TOP_RIGHT, DiagonalDirection.BOTTOM_RIGHT),
        Direction.LEFT: (DiagonalDirection.TOP_LEFT, DiagonalDirection.BOTTOM_LEFT),
        Direction.BOTTOM: (DiagonalDirection.BOTTOM_LEFT, DiagonalDirection.BOTTOM_RIGHT),
    }[direction]


def find_swap_then_qec_qubit_initializations(stabilizers: Sequence[Stabilizer], diag_direction: DiagonalDirection, relocation_diag_direction: Optional[DiagonalDirection] = None) -> tuple[dict[str, list[Coord]], dict[str, list[Coord]], list[tuple[Coord, Coord]]]:
    """The qubits a diagonal swap-then-QEC step resets, by the basis (X for |+>, Z for |0>):
    the ancillas the stabilizers move onto (and the old ancillas that become data qubits),
    the data sites past the leading boundaries, and the pairs (new data qubit, old ancilla)
    of the teleportation that ends the step."""
    vert_direction = direction_along_orientation(diag_direction, Orientation.VERTICAL)
    hor_direction = direction_along_orientation(diag_direction, Orientation.HORIZONTAL)
    data_qubits = list({q for s in stabilizers for q in s.data_qubits})
    sublattices = {q[2] for q in data_qubits}
    if len(sublattices) != 1:
        raise ValueError("The stabilizers should all belong to the same sublattice of data qubits.")
    data_sublattice = sublattices.pop()
    anc_sublattice = 1 - data_sublattice
    bound = {Direction.LEFT: min(q[0] for q in data_qubits), Direction.RIGHT: max(q[0] for q in data_qubits), Direction.TOP: min(q[1] for q in data_qubits), Direction.BOTTOM: max(q[1] for q in data_qubits)}
    same_boundary = [q for q in data_qubits if q[0] == bound[hor_direction] or q[1] == bound[vert_direction]]
    opposite_boundary = [q for q in data_qubits if q[0] == bound[hor_direction.opposite()] or q[1] == bound[vert_direction.opposite()]]
    leading = [s for s in stabilizers if set(s.data_qubits).issubset(same_boundary)]
    trailing = [s for s in stabilizers if set(s.data_qubits).issubset(opposite_boundary)]
    vector = direction_to_coord(diag_direction)
    anc_to_init: dict[str, list[Coord]] = {"X": [], "Z": []}
    for stab in stabilizers:
        if stab in leading:
            continue
        anc_to_init[pauli_type(stab)].append(tuple(a + b for a, b in zip(stab.ancilla_qubits[0], vector, strict=True)))
    teleportation_pairs: list[tuple[Coord, Coord]] = []
    data_vector = direction_to_coord(diag_direction.opposite(), anc_sublattice)
    for stab in stabilizers:
        if stab in trailing:
            continue
        anc = stab.ancilla_qubits[0]
        p = pauli_type(stab)
        if anc not in anc_to_init[p]:
            anc_to_init[p].append(anc)
            teleportation_pairs.append((anc, tuple(map(sum, zip(anc, data_vector, strict=True)))))
    data_to_init: dict[str, list[Coord]] = {"X": [], "Z": []}
    anc_to_data = direction_to_coord(diag_direction, anc_sublattice)
    for stab in leading:
        data_to_init[pauli_type(stab)].append(tuple(a + b for a, b in zip(stab.ancilla_qubits[0], anc_to_data, strict=True)))
    if relocation_diag_direction is not None:
        anc_relocation = direction_to_coord(relocation_diag_direction, anc_sublattice)
        data_relocation = direction_to_coord(relocation_diag_direction, data_sublattice)

        def add(a, b):
            return tuple(map(sum, zip(a, b, strict=True)))

        anc_to_init = {k: [add(q, anc_relocation) for q in v] for k, v in anc_to_init.items()}
        data_to_init = {k: [add(q, data_relocation) for q in v] for k, v in data_to_init.items()}
        teleportation_pairs = [tuple(add(q, anc_relocation) if q[2] == 1 else add(q, data_relocation) for q in pair) for pair in teleportation_pairs]
    return anc_to_init, data_to_init, teleportation_pairs


def get_swap_qec_cnots(step: InterpretationStep, new_block: RotatedSurfaceCode, move_diag_direction: DiagonalDirection, stab_schedule_dict: dict[Stabilizer, DetailedSchedule], anc_qubit_initialization: dict[str, list[Coord]], data_qubit_initialization: dict[str, list[Coord]]) -> Circuit:
    """Four layers of CNOTs: the first swaps each data qubit onto its new site (from the
    qubit reset in |0> to the one in |+>), the other three finish the stabilizers' schedules
    on the new sites."""
    back = move_diag_direction.opposite()
    cnots: list[list[tuple[Coord, Coord]]] = [[] for _ in range(4)]
    for q in new_block.data_qubits:
        q_prev = update_qubit_coords([q], back)[0]
        if q in anc_qubit_initialization["Z"] + data_qubit_initialization["Z"]:
            pair = (q_prev, q)
        elif q in anc_qubit_initialization["X"] + data_qubit_initialization["X"]:
            pair = (q, q_prev)
        else:
            raise ValueError("Data qubit not found in either ancilla or data qubit initialization lists.")
        cnots[0].append(pair)
    for stab in new_block.stabilizers:
        if len(stab.data_qubits) == 4:
            qubits = stab_schedule_dict[stab].get_stabilizer_qubits(stab)
        else:
            boundary = next(d for d in Direction if stab in new_block.boundary_stabilizers(d))
            qubits = stab_schedule_dict[stab].get_stabilizer_qubits(stab, boundary)
        for i, q in enumerate(qubits):
            if q is None or i == 0:
                continue
            p = pauli_type(stab)
            if p == "Z":
                pair = (q, stab.ancilla_qubits[0])
            elif p == "X":
                pair = (stab.ancilla_qubits[0], q)
            else:
                raise ValueError("Unknown stabilizer type.")
            cnots[i].append(pair)
    return Circuit(name=f"Swap-then-QEC CNOTs for moving block {new_block.unique_label} towards {move_diag_direction.value}", circuit=[[Circuit("cx", channels=[step.get_channel_MUT(q) for q in pair]) for pair in layer] for layer in cnots])


def generate_syndrome_measurement_circuit_and_cbits(step: InterpretationStep, block: RotatedSurfaceCode, actual_anc_qubit_relocation_vector: tuple[int, ...] = (0, 0, 0)) -> tuple[Circuit, list[tuple[Cbit, ...]]]:
    """One tick measuring every stabilizer's ancilla in its own basis, and the bits."""
    stab_measurements: list[tuple[Cbit, ...]] = []
    gates: list[Circuit] = []
    for stab in block.stabilizers:
        anc = tuple(map(sum, zip(stab.ancilla_qubits[0], actual_anc_qubit_relocation_vector, strict=True)))
        anc_channel = step.get_channel_MUT(anc)
        cbit = step.get_new_cbit_MUT(f"c_{anc}")
        stab_measurements.append((cbit,))
        cbit_channel = step.get_channel_MUT(f"{cbit[0]}_{cbit[1]}", channel_type="classical")
        gates.append(Circuit(f"measure_{pauli_type(stab)}", channels=[anc_channel, cbit_channel]))
    return Circuit("measure stabilizer ancillas", circuit=[gates]), stab_measurements


def generate_and_append_block_syndromes_and_detectors(step: InterpretationStep, block: RotatedSurfaceCode, syndrome_measurement_cbits: Sequence[tuple[Cbit, ...]]) -> None:
    new_syndromes = generate_syndromes(step, block.stabilizers, block, syndrome_measurement_cbits)
    new_detectors = generate_detectors(step, new_syndromes)
    step.append_syndromes_MUT(new_syndromes)
    step.append_detectors_MUT(new_detectors)


def generate_teleportation_measurement_circuit_with_updates(step: InterpretationStep, new_block: RotatedSurfaceCode, anc_qubits_to_init: dict[str, list[Coord]], teleportation_qubit_pairs: Sequence[tuple[Coord, Coord]]) -> Circuit:
    """Measure each old ancilla that held a data qubit (Z if its new site was reset in |+>,
    X if in |0>); the bit corrects the logical operators and the stabilizers of that type on
    the new site."""
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
        for stab in [s for s in new_block.stabilizers if pauli_type(s) == update and corrected in s.data_qubits]:
            step.stabilizer_updates[stab.uuid] = step.stabilizer_updates.get(stab.uuid, ()) + (cbit,)
    return Circuit(f"teleportation measurements of block {new_block.unique_label}", circuit=[gates])

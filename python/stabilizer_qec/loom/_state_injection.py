# Ported from Loom (entropicalabs/el-loom 0.4.0, src/loom_rotated_surface_code/applicator/state_injection.py;
# Copyright 2024 Entropica Labs Pte Ltd, Apache License 2.0; see LICENSE in this directory), modified for
# stabilizer_qec: plain classes in place of pydantic, if-chains in place of match statements and explicit
# loops in place of zip(strict=True) (the package supports Python 3.9).
"""Loom's state injection on a rotated surface code block: the centre data qubit is prepared in
the resource state (T or S), the others in four quadrants of |0⟩ and |+⟩ that make some
stabilizers deterministic, one round of syndromes follows, and the logical operators, first
through the centre, are moved back to the block's upper-left corner."""

from __future__ import annotations

from .circuit import ChannelType, Circuit
from .eka import Coord, Direction, MeasureBlockSyndromes, Orientation, PauliOperator, ResourceState, SingleQubitPauliEigenstate, Stabilizer, StateInjection
from .interpreter import InterpretationStep, Syndrome, generate_syndromes, measureblocksyndromes
from .rotated_surface_code import RotatedSurfaceCode


def state_injection(interpretation_step: InterpretationStep, operation: StateInjection, same_timeslice: bool, debug_mode: bool) -> InterpretationStep:
    """Inject the resource state into the block (its sizes must be odd): reset the centre qubit
    to |+⟩ and apply T or S, reset the four quadrants, file syndromes for the stabilizers the
    resets make deterministic, measure one round, and move the logicals (laid through the
    centre so that they read the injected qubit) to the upper-left corner, their corrections
    the stabilizers between. One composite circuit."""
    input_block = interpretation_step.get_block(operation.input_block_name)
    interpretation_step.begin_composite_operation_session_MUT(same_timeslice=same_timeslice, circuit_name=f"Inject {operation.resource_state.value} into block {input_block.unique_label} and measure syndromes")
    if not isinstance(input_block, RotatedSurfaceCode):
        raise TypeError(f"Expected input_block to be of type RotatedSurfaceCode, but got {type(input_block)}.")
    if not all(s % 2 == 1 for s in input_block.size):
        raise ValueError(f"Expected input_block.size to be all odd, but got {input_block.size}.")

    qubit_to_reset = (input_block.upper_left_qubit[0] + input_block.size[0] // 2, input_block.upper_left_qubit[1] + input_block.size[1] // 2, input_block.upper_left_qubit[2])
    resource_state_circuit = get_physical_state_reset(interpretation_step, input_block, qubit_to_reset, operation.resource_state)
    reset_circuit, deterministic_stabs = reset_into_four_quadrants(interpretation_step, input_block)
    state_injection_circuit = Circuit(name=f"Inject {operation.resource_state.value} into block {input_block.unique_label}", circuit=((resource_state_circuit, reset_circuit),))
    interpretation_step.append_circuit_MUT(circuit=state_injection_circuit, same_timeslice=False)

    centered_x_op, centered_z_op = find_centered_logical_operators(input_block=input_block, center_qubit=qubit_to_reset)
    centered_block = RotatedSurfaceCode(stabilizers=input_block.stabilizers, logical_x_operators=[centered_x_op], logical_z_operators=[centered_z_op], syndrome_circuits=input_block.syndrome_circuits, stabilizer_to_circuit=input_block.stabilizer_to_circuit, unique_label=input_block.unique_label, skip_validation=not debug_mode)
    # No evolution: the injection resets the block, so nothing before it is compared with it.
    interpretation_step.update_block_history_and_evolution_MUT(new_blocks=(centered_block,), old_blocks=(input_block,), update_evolution=False)
    interpretation_step.append_syndromes_MUT(create_deterministic_syndromes(interpretation_step=interpretation_step, block=centered_block, deterministic_stabs=deterministic_stabs))

    interpretation_step = measureblocksyndromes(interpretation_step, MeasureBlockSyndromes(centered_block.unique_label, 1), same_timeslice=False, debug_mode=debug_mode)

    new_x_op, required_x_stabs = centered_block.get_shifted_equivalent_logical_operator(centered_block.logical_x_operators[0], centered_block.upper_left_qubit)
    new_z_op, required_z_stabs = centered_block.get_shifted_equivalent_logical_operator(centered_block.logical_z_operators[0], centered_block.upper_left_qubit)
    new_block = RotatedSurfaceCode(stabilizers=centered_block.stabilizers, logical_x_operators=[new_x_op], logical_z_operators=[new_z_op], syndrome_circuits=centered_block.syndrome_circuits, stabilizer_to_circuit=centered_block.stabilizer_to_circuit, unique_label=centered_block.unique_label, skip_validation=not debug_mode)
    interpretation_step.update_block_history_and_evolution_MUT(new_blocks=(new_block,), old_blocks=(centered_block,))
    required_x_cbits = interpretation_step.retrieve_cbits_from_stabilizers(required_x_stabs, new_block)
    required_z_cbits = interpretation_step.retrieve_cbits_from_stabilizers(required_z_stabs, new_block)
    interpretation_step.update_logical_operator_updates_MUT("X", new_x_op.uuid, required_x_cbits, inherit_updates=True)
    interpretation_step.update_logical_operator_updates_MUT("Z", new_z_op.uuid, required_z_cbits, inherit_updates=True)

    circuit = interpretation_step.end_composite_operation_session_MUT()
    interpretation_step.append_circuit_MUT(circuit, same_timeslice)
    return interpretation_step


def get_physical_state_reset(interpretation_step: InterpretationStep, input_block: RotatedSurfaceCode, qubit_to_reset: Coord, resource_state: ResourceState) -> Circuit:
    """A data qubit of the block reset to |+⟩ and turned into the resource state: T for T, the
    phase gate for S."""
    if qubit_to_reset not in input_block.data_qubits:
        raise ValueError(f"Qubit {qubit_to_reset} is not a data qubit in the given block {input_block.unique_label}.")
    qubit_channel = interpretation_step.get_channel_MUT(qubit_to_reset, channel_type=ChannelType.QUANTUM)
    if resource_state == ResourceState.T:
        gate_sequence = (Circuit(name="Reset_+", channels=[qubit_channel]), Circuit(name="T", channels=[qubit_channel]))
    elif resource_state == ResourceState.S:
        gate_sequence = (Circuit(name="Reset_+", channels=[qubit_channel]), Circuit(name="phase", channels=[qubit_channel]))
    else:
        raise ValueError(f"Resource state {resource_state} is not supported for physical reset.")
    return Circuit(name="Resource state reset", circuit=gate_sequence, channels=[qubit_channel])


def find_qubits_quadrant(block: RotatedSurfaceCode, quadrant: Direction) -> tuple[Coord, ...]:
    """The data qubits of a quadrant: those of the boundary's stabilizers, then, walking the
    stabilizers inwards from that boundary, those of every stabilizer of the boundary's Pauli
    type that already shares two qubits with the quadrant."""
    horizontal = quadrant in (Direction.LEFT, Direction.RIGHT)
    ordered_stabilizers = sorted(block.stabilizers, key=lambda stab: stab.data_qubits[0][0] if horizontal else stab.data_qubits[0][1], reverse=(quadrant in (Direction.RIGHT, Direction.BOTTOM)))
    boundary_stabs = block.boundary_stabilizers(quadrant)
    pauli_quadrant = set(p for stab in boundary_stabs for p in stab.pauli)
    if len(pauli_quadrant) != 1:
        raise ValueError(f"Expected a single Pauli type for quadrant {quadrant}, found: {pauli_quadrant}")
    # Loom collects the quadrant in a set of int tuples; its order of iteration (which orders
    # the resets) is the same here.
    quadrant_qubit_set = set(q for stab in block.boundary_stabilizers(quadrant) for q in stab.data_qubits)
    for stab in ordered_stabilizers:
        if set(stab.pauli) == set(pauli_quadrant) and len(quadrant_qubit_set.intersection(stab.data_qubits)) == 2:
            quadrant_qubit_set |= set(stab.data_qubits)
    return tuple(quadrant_qubit_set)


def reset_into_four_quadrants(interpretation_step: InterpretationStep, block: RotatedSurfaceCode) -> tuple[Circuit, tuple[Stabilizer, ...]]:
    """The resets of the four quadrants (|0⟩ where the boundary's stabilizers are Z, |+⟩ where
    they are X; a qubit in two quadrants takes the later one's state, the quadrants taken left,
    right, top, bottom) in one tick, and the stabilizers the resets make deterministic. For a
    distance-5 block (``d`` a deterministic stabilizer)::

                 d           d
        + --- 0 --- 0 --- 0 --- 0
      d |     |     |  d  |     |
        + --- + --- 0 --- 0 --- +
        |  d  |     |     |     | d
        + --- + --- ψ --- + --- +
      d |     |     |     |  d  |
        + --- 0 --- 0 --- + --- +
        |     |  d  |     |     | d
        0 --- 0 --- 0 --- 0 --- +
           d           d
    """
    qubits_to_state_map: dict[Coord, SingleQubitPauliEigenstate] = {}
    for quadrant in Direction:
        pauli_type = set(p for stab in block.boundary_stabilizers(quadrant) for p in stab.pauli)
        if len(pauli_type) != 1:
            raise ValueError(f"Expected a single Pauli type for quadrant {quadrant}, found: {pauli_type}")
        quadrant_qubits = find_qubits_quadrant(block, quadrant)
        qubits_to_state_map |= {qubit: (SingleQubitPauliEigenstate.ZERO if pauli_type == {"Z"} else SingleQubitPauliEigenstate.PLUS) for qubit in quadrant_qubits}
    reset_circuit = Circuit(name="reset four quadrants", circuit=(tuple(Circuit(name=f"reset_{state.value}", channels=[interpretation_step.get_channel_MUT(qubit, channel_type=ChannelType.QUANTUM)]) for qubit, state in qubits_to_state_map.items()),))
    deterministic_stabs = tuple(stab for stab in block.stabilizers if all(qubits_to_state_map.get(qubit, None) == ("+" if pauli == "X" else "0") for qubit, pauli in zip(stab.data_qubits, stab.pauli)))
    return reset_circuit, deterministic_stabs


def find_centered_logical_operators(input_block: RotatedSurfaceCode, center_qubit: Coord) -> tuple[PauliOperator, PauliOperator]:
    """The logical X and Z laid through the centre qubit, each across the block in its own
    direction, so that they read the injected qubit."""
    x_log_is_horizontal = input_block.x_boundary == Orientation.HORIZONTAL
    x_op_len, z_op_len = input_block.size if x_log_is_horizontal else input_block.size[::-1]
    x_op = PauliOperator(pauli="X" * x_op_len, data_qubits=tuple((center_qubit[0] + i * x_log_is_horizontal, center_qubit[1] + i * (not x_log_is_horizontal), center_qubit[2]) for i in range(-(x_op_len // 2), x_op_len // 2 + 1)))
    z_op = PauliOperator(pauli="Z" * z_op_len, data_qubits=tuple((center_qubit[0] + i * (not x_log_is_horizontal), center_qubit[1] + i * x_log_is_horizontal, center_qubit[2]) for i in range(-(z_op_len // 2), z_op_len // 2 + 1)))
    return x_op, z_op


def create_deterministic_syndromes(interpretation_step: InterpretationStep, block: RotatedSurfaceCode, deterministic_stabs: tuple[Stabilizer, ...]) -> tuple[Syndrome, ...]:
    """Syndromes without measurements (value 0) for the stabilizers the injection makes
    deterministic."""
    return generate_syndromes(interpretation_step, deterministic_stabs, block, tuple(() for _ in deterministic_stabs))

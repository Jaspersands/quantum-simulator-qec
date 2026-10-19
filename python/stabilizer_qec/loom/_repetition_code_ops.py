# Ported from Loom (entropicalabs/el-loom 0.4.0, src/loom_repetition_code/applicator/{grow,shrink,merge,split}.py;
# Copyright 2024 Entropica Labs Pte Ltd, Apache License 2.0; see LICENSE in this directory), modified for
# stabilizer_qec: the helper functions folded into the four applicators.
"""Loom's lattice surgery on repetition chains: grow and shrink at either end, merge two chains
side by side, split one in two by measuring a qubit; each tracks how the long logical (on every
qubit) and the short one (on one qubit) evolve and which measurements correct them."""

from __future__ import annotations

from .circuit import ChannelType, Circuit
from .eka import Direction, Grow, Merge, Orientation, PauliOperator, Shrink, Split, Stabilizer
from .interpreter import InterpretationStep
from .repetition_code import RepetitionCode


def grow(interpretation_step: InterpretationStep, operation: Grow, same_timeslice: bool, debug_mode: bool) -> InterpretationStep:
    """New data qubits at one end, reset to the long logical's eigenstate, and a check for each."""
    block = interpretation_step.get_block(operation.input_block_name)
    if not isinstance(block, RepetitionCode):
        raise ValueError(f"This grow operation is not supported for {type(block)} blocks.")
    if operation.direction not in [Direction.RIGHT, Direction.LEFT]:
        raise ValueError(f"Repetition code does not support growing in the {operation.direction.value} direction.")
    if operation.direction == Direction.LEFT:
        left_boundary_position = block.boundary_qubits(Direction.LEFT)[0]
        if left_boundary_position - operation.length < 0:
            raise ValueError("Cannot grow beyond the boundary of the lattice.")
    check_type = block.check_type
    is_left = operation.direction == Direction.LEFT
    left_boundary_qubit = block.boundary_qubits(Direction.LEFT)
    if operation.direction == Direction.LEFT:
        new_data_qubits = [(-i + left_boundary_qubit[0] - 1, 0) for i in range(operation.length)]
    else:
        right_boundary_qubit = block.boundary_qubits(Direction.RIGHT)
        new_data_qubits = [(i + right_boundary_qubit[0] + 1, 0) for i in range(operation.length)]
    circuit_name = f"grow {block.unique_label} by {operation.length} to the {operation.direction.value}"
    reset_state = "0" if check_type == "X" else "+"
    grow_circuit = Circuit(name=circuit_name, circuit=[[Circuit(name=f"reset_{reset_state}", channels=interpretation_step.get_channel_MUT(q, "quantum")) for q in new_data_qubits]])
    interpretation_step.append_circuit_MUT(grow_circuit, same_timeslice)
    stabs_to_add = [Stabilizer(pauli=check_type * 2, data_qubits=sorted([(coord - (1 - 2 * is_left), 0), (coord, 0)], key=lambda x: x[0]), ancilla_qubits=[(coord - (1 - is_left), 1)]) for coord, _ in new_data_qubits]
    new_stabilizers = list(block.stabilizers) + stabs_to_add
    complementary_check_type = "X" if check_type == "Z" else "Z"
    all_qubits = list(block.data_qubits) + new_data_qubits
    old_short_log, old_long_log = sorted([block.logical_x_operators[0], block.logical_z_operators[0]], key=lambda x: len(x.data_qubits))
    new_long_log = PauliOperator(pauli=complementary_check_type * len(all_qubits), data_qubits=all_qubits)
    new_logs = [[new_long_log], [old_short_log]]
    log_evolution = [{new_long_log.uuid: (old_long_log.uuid,)}, {}]
    ordering = 1 if check_type == "Z" else -1
    new_log_x_ops, new_log_z_ops = new_logs[::ordering]
    logical_x_evolution, logical_z_evolution = log_evolution[::ordering]
    interpretation_step.logical_x_evolution.update(logical_x_evolution)
    interpretation_step.logical_z_evolution.update(logical_z_evolution)
    for op in new_log_x_ops:
        interpretation_step.update_logical_operator_updates_MUT("X", op.uuid, (), True)
    for op in new_log_z_ops:
        interpretation_step.update_logical_operator_updates_MUT("Z", op.uuid, (), True)
    new_block = RepetitionCode(stabilizers=new_stabilizers, logical_x_operators=new_log_x_ops, logical_z_operators=new_log_z_ops, unique_label=block.unique_label, skip_validation=not debug_mode)
    interpretation_step.update_block_history_and_evolution_MUT(new_blocks=(new_block,), old_blocks=(block,))
    return interpretation_step


def shrink(interpretation_step: InterpretationStep, operation: Shrink, same_timeslice: bool, debug_mode: bool) -> InterpretationStep:
    """The data qubits at one end measured in the long logical's basis; the long logical takes
    their outcomes, the short one moves (by its stabilizers' syndromes) if it stood on them."""
    block = interpretation_step.get_block(operation.input_block_name)
    if not isinstance(block, RepetitionCode):
        raise ValueError(f"This shrink operation is not supported for {type(block)} blocks.")
    if operation.length >= (len(block.data_qubits) - 1):
        raise ValueError("Shrink size is too large.")
    if operation.direction in [Direction.BOTTOM, Direction.TOP]:
        raise ValueError(f"Repetition code does not support shrinking in the {operation.direction.value} direction.")
    is_left = operation.direction == Direction.LEFT
    check_type = block.check_type
    boundary_qubit = block.boundary_qubits(direction=operation.direction)
    shift_vectors = [(i, 0) for i in range(operation.length)] if is_left else [(-i, 0) for i in range(operation.length)]
    qubits_to_measure = [tuple(coord1 + coord2 for coord1, coord2 in zip(boundary_qubit, shift_vect)) for shift_vect in shift_vectors]
    circuit_name = f"shrink {block.unique_label} by {operation.length} from {operation.direction.value}"
    cbits = [interpretation_step.get_new_cbit_MUT(f"c_{q}") for q in qubits_to_measure]
    cbit_channels = [interpretation_step.get_channel_MUT(f"{cbit[0]}_{cbit[1]}", channel_type=ChannelType.CLASSICAL) for cbit in cbits]
    measure_circuit_seq = [[Circuit("Measurement", channels=[interpretation_step.get_channel_MUT(q), cbc]) for q, cbc in zip(qubits_to_measure, cbit_channels)]]
    if check_type == "X":
        shrink_circuit_list = measure_circuit_seq
    else:
        shrink_circuit_list = [[Circuit("H", channels=[interpretation_step.get_channel_MUT(qb)]) for qb in qubits_to_measure]] + measure_circuit_seq
    interpretation_step.append_circuit_MUT(Circuit(name=circuit_name, circuit=shrink_circuit_list), same_timeslice)
    stabs_to_remove = [stab for stab in block.stabilizers if any(qb in qubits_to_measure for qb in stab.data_qubits)]
    new_stabilizers = list(set(block.stabilizers) - set(stabs_to_remove))
    other_check_type = "X" if check_type == "Z" else "Z"
    remaining_qubits = list(set(block.data_qubits) - set(qubits_to_measure))
    old_short_log, old_long_log = sorted([block.logical_x_operators[0], block.logical_z_operators[0]], key=lambda x: len(x.data_qubits))
    to_shift = old_short_log.data_qubits[0] in qubits_to_measure
    new_long_log = PauliOperator(pauli=other_check_type * len(remaining_qubits), data_qubits=remaining_qubits)
    long_log_updates = {new_long_log.uuid: tuple(cbits)}
    long_log_evolution = {new_long_log.uuid: (old_long_log.uuid,)}
    if to_shift:
        selector_function = min if is_left else max
        new_qubit = selector_function(remaining_qubits, key=lambda x: x[0])
        new_short_log, stabs_required = block.get_shifted_equivalent_logical_operator(new_qubit)
        short_log_evolution = {new_short_log.uuid: tuple([old_short_log.uuid] + [stab.uuid for stab in stabs_required])}
        short_log_updates = {new_short_log.uuid: interpretation_step.retrieve_cbits_from_stabilizers(stabs_required, block)}
    else:
        new_short_log = old_short_log
        short_log_evolution = {}
        short_log_updates = {}
    new_logs = [[new_long_log], [new_short_log]]
    log_evolution = [long_log_evolution, short_log_evolution]
    log_updates = [long_log_updates, short_log_updates]
    ordering = 1 if check_type == "Z" else -1
    new_log_x_ops, new_log_z_ops = new_logs[::ordering]
    logical_x_evolution, logical_z_evolution = log_evolution[::ordering]
    logical_x_updates, logical_z_updates = log_updates[::ordering]
    interpretation_step.logical_x_evolution.update(logical_x_evolution)
    interpretation_step.logical_z_evolution.update(logical_z_evolution)
    for op_id, measurements in logical_x_updates.items():
        interpretation_step.update_logical_operator_updates_MUT("X", op_id, measurements, True)
    for op_id, measurements in logical_z_updates.items():
        interpretation_step.update_logical_operator_updates_MUT("Z", op_id, measurements, True)
    new_block = RepetitionCode(stabilizers=new_stabilizers, logical_x_operators=new_log_x_ops, logical_z_operators=new_log_z_ops, unique_label=block.unique_label, skip_validation=not debug_mode)
    interpretation_step.update_block_history_and_evolution_MUT(new_blocks=(new_block,), old_blocks=(block,))
    return interpretation_step


def merge(interpretation_step: InterpretationStep, operation: Merge, same_timeslice: bool, debug_mode: bool) -> InterpretationStep:
    """The qubits between two chains (left one first) reset to the long logical's eigenstate
    and checked: one chain, its long logical the product of both, its short one the left's."""
    name_1, name_2 = operation.input_blocks_name
    block_1 = interpretation_step.get_block(name_1)
    block_2 = interpretation_step.get_block(name_2)
    if not isinstance(block_1, RepetitionCode) or not isinstance(block_2, RepetitionCode):
        wrong_types = tuple(set((type(block_1), type(block_2))).difference({RepetitionCode}))
        raise TypeError(f"This merge operation is not supported for {tuple(wrong_types)} blocks.")
    if block_1.boundary_qubits(Direction.LEFT)[0] > block_2.boundary_qubits(Direction.LEFT)[0]:
        block_1, block_2 = block_2, block_1
    if block_1.check_type != block_2.check_type:
        raise TypeError(f"Cannot merge blocks with different check types: {block_1.check_type} and {block_2.check_type}")
    left_boundary_block_1 = block_1.boundary_qubits(Direction.LEFT)[0]
    distance_block_1 = len(block_1.data_qubits)
    left_boundary_block_2 = block_2.boundary_qubits(Direction.LEFT)[0]
    if left_boundary_block_2 < left_boundary_block_1 + distance_block_1:
        raise ValueError("Cannot merge blocks that overlap.")
    if operation.orientation != Orientation.HORIZONTAL:
        raise ValueError("Repetition code does not support merging in the vertical orientation.")
    check_type = block_1.check_type
    right_boundary_block_1 = block_1.boundary_qubits(Direction.RIGHT)[0]
    new_data_qubits = [(i, 0) for i in range(right_boundary_block_1 + 1, left_boundary_block_2)]
    circuit_name = f"merge {block_1.unique_label} and {block_2.unique_label} into {operation.output_block_name}"
    reset_state = "0" if check_type == "X" else "+"
    merge_circuit = Circuit(name=circuit_name, circuit=[[Circuit(name=f"reset_{reset_state}", channels=interpretation_step.get_channel_MUT(q, "quantum")) for q in new_data_qubits]])
    interpretation_step.append_circuit_MUT(merge_circuit, same_timeslice)
    add_stabilizers = [Stabilizer(pauli=check_type * 2, data_qubits=[(i, 0), (i + 1, 0)], ancilla_qubits=[(i, 1)]) for i in range(right_boundary_block_1, left_boundary_block_2)]
    new_stabilizers = list(block_1.stabilizers) + add_stabilizers + list(block_2.stabilizers)
    other_check_type = "X" if check_type == "Z" else "Z"
    old_short_log_1, old_long_log_1 = sorted([block_1.logical_x_operators[0], block_1.logical_z_operators[0]], key=lambda x: len(x.data_qubits))
    _, old_long_log_2 = sorted([block_2.logical_x_operators[0], block_2.logical_z_operators[0]], key=lambda x: len(x.data_qubits))
    new_long_data_qubits = list(block_1.data_qubits) + new_data_qubits + list(block_2.data_qubits)
    new_long_log = PauliOperator(pauli=other_check_type * len(new_long_data_qubits), data_qubits=new_long_data_qubits)
    new_short_log, stabs_required = block_1.get_shifted_equivalent_logical_operator(block_1.boundary_qubits(Direction.LEFT))
    new_logs = [[new_long_log], [new_short_log]]
    long_log_evolution = {new_long_log.uuid: (old_long_log_1.uuid, old_long_log_2.uuid)}
    short_log_evolution = {new_short_log.uuid: tuple([old_short_log_1.uuid] + [stab.uuid for stab in stabs_required])} if stabs_required else {}
    log_evolution = [long_log_evolution, short_log_evolution]
    ordering = 1 if check_type == "Z" else -1
    new_log_x_ops, new_log_z_ops = new_logs[::ordering]
    logical_x_evolution, logical_z_evolution = log_evolution[::ordering]
    interpretation_step.logical_x_evolution.update(logical_x_evolution)
    interpretation_step.logical_z_evolution.update(logical_z_evolution)
    for op in new_log_x_ops:
        interpretation_step.update_logical_operator_updates_MUT("X", op.uuid, (), True)
    for op in new_log_z_ops:
        interpretation_step.update_logical_operator_updates_MUT("Z", op.uuid, (), True)
    new_block = RepetitionCode(stabilizers=new_stabilizers, logical_x_operators=new_log_x_ops, logical_z_operators=new_log_z_ops, unique_label=operation.output_block_name, skip_validation=not debug_mode)
    interpretation_step.update_block_history_and_evolution_MUT(new_blocks=(new_block,), old_blocks=(block_1, block_2))
    return interpretation_step


def split(interpretation_step: InterpretationStep, operation: Split, same_timeslice: bool, debug_mode: bool) -> InterpretationStep:
    """The data qubit ``split_position`` along the chain measured in the long logical's basis:
    two chains, the left one's long logical taking the outcome, each short logical moved to its
    chain's left end."""
    block = interpretation_step.get_block(operation.input_block_name)
    if not isinstance(block, RepetitionCode):
        raise ValueError(f"This split operation is not supported for {type(block)} blocks.")
    relative_split_position = operation.split_position
    left_boundary = block.boundary_qubits(Direction.LEFT)[0]
    right_boundary = block.boundary_qubits(Direction.RIGHT)[0]
    split_position = left_boundary + relative_split_position
    if split_position > right_boundary or split_position < left_boundary:
        raise ValueError(f"Split position {split_position} is outside the chain.")
    if split_position in [left_boundary, right_boundary]:
        raise ValueError("Split position cannot be at the edge of the chain.")
    if split_position in [left_boundary + 1, right_boundary - 1]:
        raise ValueError("Split has to partition chain in units of at least two qubits.")
    if operation.orientation != Orientation.VERTICAL:
        raise ValueError("Only vertical splits are supported.")
    split_position = operation.split_position
    check_type = block.check_type
    qubit_to_measure = (block.boundary_qubits(Direction.LEFT)[0] + split_position, 0)
    circuit_name = f"split {block.unique_label} at {split_position}"
    cbit = interpretation_step.get_new_cbit_MUT(f"c_{qubit_to_measure}")
    cbit_channel = interpretation_step.get_channel_MUT(f"{cbit[0]}_{cbit[1]}", channel_type=ChannelType.CLASSICAL)
    measure_circuit_seq = [Circuit("Measurement", channels=[interpretation_step.get_channel_MUT(qubit_to_measure), cbit_channel])]
    if check_type == "X":
        split_circuit_list = measure_circuit_seq
    else:
        split_circuit_list = [Circuit("H", channels=[interpretation_step.get_channel_MUT(qubit_to_measure)])] + measure_circuit_seq
    interpretation_step.append_circuit_MUT(Circuit(name=circuit_name, circuit=split_circuit_list), same_timeslice)
    # The stabilizers of each part.
    stabs_to_remove = [stab for stab in block.stabilizers if any(q == qubit_to_measure for q in stab.data_qubits)]
    new_stabilizers = [stab for stab in block.stabilizers if stab not in stabs_to_remove]
    data_qubits_block_1 = set(q for q in block.data_qubits if q[0] < qubit_to_measure[0])
    data_qubits_block_2 = set(q for q in block.data_qubits if q[0] > qubit_to_measure[0])
    new_stabs_block_1 = [stab for stab in new_stabilizers if set(stab.data_qubits).issubset(data_qubits_block_1)]
    new_stabs_block_2 = [stab for stab in new_stabilizers if set(stab.data_qubits).issubset(data_qubits_block_2)]
    remaining_stabs = set(new_stabilizers) - set(new_stabs_block_1) - set(new_stabs_block_2)
    if len(remaining_stabs) != 0:
        raise ValueError(f"Stabilizers {remaining_stabs} are not assigned to any block.")
    # The logical operators of each part, their evolution and their corrections.
    other_check_type = "X" if check_type == "Z" else "Z"
    qubits_block_1 = [q for q in block.data_qubits if q[0] < qubit_to_measure[0]]
    left_boundary_qubit_1 = min(qubits_block_1, key=lambda x: x[0])
    qubits_block_2 = [q for q in block.data_qubits if q[0] > qubit_to_measure[0]]
    left_boundary_qubit_2 = min(qubits_block_2, key=lambda x: x[0])
    old_short_log, old_long_log = sorted((block.logical_x_operators[0], block.logical_z_operators[0]), key=lambda x: len(x.data_qubits))
    new_long_log_1 = PauliOperator(pauli=other_check_type * len(qubits_block_1), data_qubits=qubits_block_1)
    long_log_evolution_1 = {new_long_log_1.uuid: (old_long_log.uuid,)}
    long_log_updates_1 = {new_long_log_1.uuid: (cbit,)}
    new_long_log_2 = PauliOperator(pauli=other_check_type * len(qubits_block_2), data_qubits=qubits_block_2)
    long_log_evolution_2 = {new_long_log_2.uuid: (old_long_log.uuid,)}
    long_log_updates_2: dict = {}
    shifted = [block.get_shifted_equivalent_logical_operator(left_boundary_qubit_1), block.get_shifted_equivalent_logical_operator(left_boundary_qubit_2)]
    new_short_logs, short_log_evolutions, short_log_updates = [], [], []
    for new_short_log, required_stabs in shifted:
        id_required_stabs = [stab.uuid for stab in required_stabs]
        if id_required_stabs == []:
            new_short_logs.append(old_short_log)
            short_log_evolutions.append({})
            short_log_updates.append({})
        else:
            new_short_logs.append(new_short_log)
            short_log_evolutions.append({new_short_log.uuid: tuple([old_short_log.uuid] + id_required_stabs)})
            short_log_updates.append({new_short_log.uuid: interpretation_step.retrieve_cbits_from_stabilizers(required_stabs, block)})
    new_logs_1 = [[new_long_log_1], [new_short_logs[0]]]
    new_logs_2 = [[new_long_log_2], [new_short_logs[1]]]
    log_updates_1 = [long_log_updates_1, short_log_updates[0]]
    log_evolution_1 = [long_log_evolution_1, short_log_evolutions[0]]
    log_updates_2 = [long_log_updates_2, short_log_updates[1]]
    log_evolution_2 = [long_log_evolution_2, short_log_evolutions[1]]
    ordering = 1 if check_type == "Z" else -1
    x_is_long_operator = check_type == "Z"
    new_log_x_ops_1, new_log_z_ops_1 = new_logs_1[::ordering]
    new_log_x_ops_2, new_log_z_ops_2 = new_logs_2[::ordering]
    logical_x_evolution_1, logical_z_evolution_1 = log_evolution_1[::ordering]
    logical_x_evolution_2, logical_z_evolution_2 = log_evolution_2[::ordering]
    logical_x_updates_1, logical_z_updates_1 = log_updates_1[::ordering]
    logical_x_updates_2, logical_z_updates_2 = log_updates_2[::ordering]
    interpretation_step.logical_x_evolution.update({**logical_x_evolution_1, **logical_x_evolution_2})
    for op_id, cbits in logical_x_updates_1.items():
        interpretation_step.update_logical_operator_updates_MUT("X", op_id, cbits, True)
    for op_id, cbits in logical_x_updates_2.items():
        interpretation_step.update_logical_operator_updates_MUT("X", op_id, cbits, not x_is_long_operator)
    interpretation_step.logical_z_evolution.update({**logical_z_evolution_1, **logical_z_evolution_2})
    for op_id, cbits in logical_z_updates_1.items():
        interpretation_step.update_logical_operator_updates_MUT("Z", op_id, cbits, True)
    for op_id, cbits in logical_z_updates_2.items():
        interpretation_step.update_logical_operator_updates_MUT("Z", op_id, cbits, x_is_long_operator)
    new_block_1 = RepetitionCode(stabilizers=new_stabs_block_1, logical_x_operators=new_log_x_ops_1, logical_z_operators=new_log_z_ops_1, unique_label=operation.output_blocks_name[0], skip_validation=not debug_mode)
    new_block_2 = RepetitionCode(stabilizers=new_stabs_block_2, logical_x_operators=new_log_x_ops_2, logical_z_operators=new_log_z_ops_2, unique_label=operation.output_blocks_name[1], skip_validation=not debug_mode)
    interpretation_step.update_block_history_and_evolution_MUT(new_blocks=(new_block_1, new_block_2), old_blocks=(block,))
    return interpretation_step

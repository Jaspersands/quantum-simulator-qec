# Ported from Loom (entropicalabs/el-loom 0.4.0, src/loom_rotated_surface_code/applicator/logical_phase_via_ywall.py;
# Copyright 2024 Entropica Labs Pte Ltd, Apache License 2.0; see LICENSE in this directory), modified for
# stabilizer_qec: plain functions on this package's interpretation step.
"""Loom's LogicalPhaseViaYwall: the logical S gate of a d × d rotated surface code block (d
odd) by a Y wall. The block grows by d along its X boundaries, one corner slides along the
grown block, the Y wall comes out between the two squares, two more corners slide back, the
block shrinks to its starting place, and the X logical returns to where it was; syndrome
rounds between the steps."""

from __future__ import annotations

from .eka import Coord, Direction, Grow, LogicalPhaseViaYwall, MeasureBlockSyndromes, Orientation, Shrink
from .interpreter import InterpretationStep, SyndromeMissingError, grow, measureblocksyndromes, shrink
from .rotated_surface_code import RotatedSurfaceCode
from ._move_corners import move_corners
from ._y_wall import y_wall_out


def logical_phase_via_ywall(step: InterpretationStep, operation: LogicalPhaseViaYwall, same_timeslice: bool, debug_mode: bool) -> InterpretationStep:
    step.begin_composite_operation_session_MUT(same_timeslice=same_timeslice, circuit_name=f"LogicalPhaseViaYwall on block {operation.input_block_name} towards the {operation.growth_direction.name}.")
    init_block = current = check_consistency(step, operation)
    label = init_block.unique_label
    growth_direction = operation.growth_direction
    init_x = init_block.logical_x_operators[0]
    distance = init_block.size[0]
    towards_negative = growth_direction in [Direction.LEFT, Direction.TOP]
    top_left_x = towards_negative ^ (init_block.upper_left_4body_stabilizer.pauli[0] == "X")
    x_horizontal = init_block.x_boundary == Orientation.HORIZONTAL
    case = (top_left_x, x_horizontal)
    key = {(False, False): lambda x: -x[0] - x[1], (True, False): lambda x: +x[0] - x[1], (True, True): lambda x: -x[0] + x[1], (False, True): lambda x: -x[0] - x[1]}[case]
    new_x_top_left = max(current.data_qubits, key=key)
    try:
        step, current = move_logical(step, current, new_x_top_left, "X")
    except SyndromeMissingError:
        step = measureblocksyndromes(step, MeasureBlockSyndromes(label, distance), same_timeslice=False, debug_mode=debug_mode)
        step, current = move_logical(step, current, new_x_top_left, "X")
    growth_length = distance
    step = grow(step, Grow(label, growth_direction, growth_length), same_timeslice=False, debug_mode=debug_mode)
    current = step.get_block(label)
    step = measureblocksyndromes(step, MeasureBlockSyndromes(label, distance), same_timeslice=False, debug_mode=debug_mode)
    if towards_negative:
        step, current = move_logical(step, current, current.upper_left_qubit, pauli="Z")
    current = step.get_block(label)
    corner_0_at, corner_0_move = {(False, False): ((Direction.BOTTOM, Direction.LEFT), Direction.TOP), (True, False): ((Direction.BOTTOM, Direction.RIGHT), Direction.TOP), (True, True): ((Direction.BOTTOM, Direction.RIGHT), Direction.LEFT), (False, True): ((Direction.TOP, Direction.RIGHT), Direction.LEFT)}[case]
    corner_0 = set(current.boundary_qubits(corner_0_at[0])).intersection(current.boundary_qubits(corner_0_at[1])).pop()
    step = move_corners(step, current, ((corner_0, corner_0_move, distance - 1),), same_timeslice=False, debug_mode=debug_mode)
    step = measureblocksyndromes(step, MeasureBlockSyndromes(label, distance - 1), same_timeslice=False, debug_mode=debug_mode)
    current = step.get_block(label)
    step = y_wall_out(step, current, distance, init_block.x_boundary.perpendicular(), same_timeslice=False, debug_mode=debug_mode)
    current = step.get_block(label)
    corner_1_at, corner_1_move = {(False, False): ((Direction.BOTTOM, Direction.RIGHT), Direction.LEFT), (True, False): ((Direction.BOTTOM, Direction.LEFT), Direction.RIGHT), (True, True): ((Direction.TOP, Direction.RIGHT), Direction.BOTTOM), (False, True): ((Direction.BOTTOM, Direction.RIGHT), Direction.TOP)}[case]
    corner_1 = set(current.boundary_qubits(corner_1_at[0])).intersection(current.boundary_qubits(corner_1_at[1])).pop()
    current = step.get_block(label)
    corner_2 = (set(current.topological_corners) - set(current.geometric_corners)).pop()
    for corner_args in ((corner_1, corner_1_move, distance - 1), (corner_2, corner_0_move.opposite(), distance - 1)):
        step = move_corners(step, current, (corner_args,), same_timeslice=False, debug_mode=debug_mode)
        current = step.get_block(label)
        step = measureblocksyndromes(step, MeasureBlockSyndromes(label, distance - 1), same_timeslice=False, debug_mode=debug_mode)
    if towards_negative:
        step = grow(step, Grow(label, growth_direction.opposite(), 1), same_timeslice=False, debug_mode=debug_mode)
        step = measureblocksyndromes(step, MeasureBlockSyndromes(label, distance), same_timeslice=False, debug_mode=debug_mode)
    step = shrink(step, Shrink(label, growth_direction, growth_length - (1 if not towards_negative else 0)), same_timeslice=False, debug_mode=debug_mode)
    step = measureblocksyndromes(step, MeasureBlockSyndromes(label, distance), same_timeslice=False, debug_mode=debug_mode)
    current = step.get_block(label)
    step, current = move_logical(step, current, max(init_x.data_qubits, key=lambda x: -x[0] - x[1]), "X")
    circuit = step.end_composite_operation_session_MUT()
    step.append_circuit_MUT(circuit, same_timeslice)
    return step


def check_consistency(step: InterpretationStep, operation: LogicalPhaseViaYwall) -> RotatedSurfaceCode:
    block = step.get_block(operation.input_block_name)
    if not isinstance(block, RotatedSurfaceCode):
        raise ValueError(f"The LogicalPhaseViaYwall operation can only be applied to RotatedSurfaceCode blocks. The block {block.unique_label} is a {block.__class__.__name__} block.")
    dim_x, dim_y = block.size
    if dim_x != dim_y or dim_x % 2 != 1:
        raise ValueError("Block must be square and have odd dimensions for the logical phase via y-wall operation.")
    if set(block.topological_corners) != set(block.geometric_corners):
        raise ValueError("Topological corners must coincide with geometric corners for the logical phase via y-wall operation.")
    if operation.growth_direction.to_orientation() != block.x_boundary:
        raise ValueError(f"The growth direction ({operation.growth_direction.name}) must be parallel to the x boundary ({block.x_boundary.name}) of the block for the logical phase via y-wall operation.")
    return block


def move_logical(step: InterpretationStep, current_block: RotatedSurfaceCode, new_up_left_qubit: Coord, pauli: str) -> tuple[InterpretationStep, RotatedSurfaceCode]:
    """The block with its X or Z logical shifted to start at the qubit, corrected by the last
    syndromes of the stabilizers between the old and new positions."""
    if pauli == "X":
        current_op = current_block.logical_x_operators[0]
    elif pauli == "Z":
        current_op = current_block.logical_z_operators[0]
    else:
        raise ValueError(f"Logical operator {pauli} is not supported. Only 'X' and 'Z' are supported.")
    new_logical, stabilizers = current_block.get_shifted_equivalent_logical_operator(current_op, new_up_left_qubit)
    if stabilizers:
        cbits = step.retrieve_cbits_from_stabilizers(stabilizers, current_block)
        if pauli == "X":
            step.logical_x_evolution[new_logical.uuid] = (current_block.logical_x_operators[0].uuid,) + tuple(s.uuid for s in stabilizers)
            new_x, new_z = new_logical, current_block.logical_z_operators[0]
        else:
            step.logical_z_evolution[new_logical.uuid] = (current_block.logical_z_operators[0].uuid,) + tuple(s.uuid for s in stabilizers)
            new_x, new_z = current_block.logical_x_operators[0], new_logical
        step.update_logical_operator_updates_MUT(pauli, new_logical.uuid, cbits, inherit_updates=True)
        new_block = RotatedSurfaceCode(stabilizers=current_block.stabilizers, logical_x_operators=(new_x,), logical_z_operators=(new_z,), syndrome_circuits=current_block.syndrome_circuits, stabilizer_to_circuit=current_block.stabilizer_to_circuit, unique_label=current_block.unique_label)
        step.update_block_history_and_evolution_MUT((new_block,), (current_block,))
        current_block = new_block
    return step, current_block

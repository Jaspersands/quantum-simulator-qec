# Ported from Loom (entropicalabs/el-loom 0.4.0, src/loom_rotated_surface_code/applicator/auxcnot.py; Copyright
# 2024 Entropica Labs Pte Ltd, Apache License 2.0; see LICENSE in this directory), modified for stabilizer_qec:
# plain classes in place of pydantic, an if-chain in place of the match statement (the package supports
# Python 3.9).
"""Loom's auxiliary CNOT between two rotated surface code blocks set diagonally apart: the
control grows along its logical X past the target, splits off an auxiliary block beside the
target, the auxiliary block merges with the target (an XX measurement), the control takes a
conditional logical Z on its outcome, and the target shrinks back to its size."""

from __future__ import annotations

from uuid import uuid4

from .eka import AuxCNOT, ConditionalLogicalZ, Direction, Grow, LogicalMeasurement, MeasureBlockSyndromes, Merge, Orientation, Shrink, Split
from .interpreter import InterpretationStep, conditional_logical_pauli, grow, measureblocksyndromes, merge, shrink, split
from .rotated_surface_code import RotatedSurfaceCode


def auxcnot(interpretation_step: InterpretationStep, operation: AuxCNOT, same_timeslice: bool, debug_mode: bool) -> InterpretationStep:
    """A CNOT from ``input_blocks_name[0]`` (control) to ``input_blocks_name[1]`` (target),
    by grow, split, merge and shrink, as one composite circuit:

    - grow the control along its logical X by the target's width plus one;
    - measure the grown control's and the target's syndromes (as many rounds as the smaller
      side of each);
    - split the grown control into the control and an auxiliary block beside the target
      (labelled ``<control>_aux``);
    - one round on the control, the auxiliary block and the target;
    - merge the auxiliary block into the target (keeping the target's label);
    - one more round on the control beside the merge's, then ``d - 1`` rounds on both;
    - a conditional logical Z on the control (and on the target when it shrinks from the top
      or the left, where it keeps the auxiliary block's logical X) on the merge's XX outcome;
    - shrink the target back to its size."""
    c_block = interpretation_step.get_block(operation.input_blocks_name[0])
    t_block = interpretation_step.get_block(operation.input_blocks_name[1])
    interpretation_step.begin_composite_operation_session_MUT(same_timeslice=same_timeslice, circuit_name=f"auxcnot between {c_block.unique_label} and {t_block.unique_label}")
    auxcnot_consistency_check(c_block, t_block)
    grow_direction, shrink_direction = get_grow_shrink_directions(control=c_block, target=t_block)

    interpretation_step = auxcnot_grow_control(interpretation_step, control=c_block, target=t_block, grow_direction=grow_direction, same_timeslice=False, debug_mode=debug_mode)
    grown_control = interpretation_step.get_block(c_block.unique_label)
    aux_unique_label = f"{grown_control.unique_label}_aux"
    if aux_unique_label in [block.unique_label for block in interpretation_step.get_blocks_at_index(-1)]:
        aux_unique_label = str(uuid4())

    for block, internal_same_timeslice in ((grown_control, False), (t_block, True)):
        interpretation_step = measureblocksyndromes(interpretation_step, MeasureBlockSyndromes(block.unique_label, min(block.size)), same_timeslice=internal_same_timeslice, debug_mode=debug_mode)

    interpretation_step = auxcnot_split_control(interpretation_step, aux_unique_label=aux_unique_label, initial_control=c_block, initial_target=t_block, grown_control=grown_control, same_timeslice=False, debug_mode=debug_mode)
    new_control = interpretation_step.get_block(c_block.unique_label)
    aux_block = interpretation_step.get_block(aux_unique_label)

    for block, internal_same_timeslice in ((new_control, False), (aux_block, True), (t_block, True)):
        interpretation_step = measureblocksyndromes(interpretation_step, MeasureBlockSyndromes(block.unique_label, 1), same_timeslice=internal_same_timeslice, debug_mode=debug_mode)

    interpretation_step = auxcnot_merge_aux_target(interpretation_step, aux=aux_block, target=t_block, same_timeslice=False, debug_mode=debug_mode)
    merged_target = interpretation_step.get_block(t_block.unique_label)

    for block, internal_same_timeslice, n_cycles in ((new_control, True, 1), (new_control, False, min(new_control.size) - 1), (merged_target, True, min(merged_target.size) - 1)):
        interpretation_step = measureblocksyndromes(interpretation_step, MeasureBlockSyndromes(block.unique_label, n_cycles), same_timeslice=internal_same_timeslice, debug_mode=debug_mode)

    interpretation_step = auxcnot_conditional_logical_z(interpretation_step, new_control=new_control, initial_target=t_block, aux_block=aux_block, new_target=merged_target, shrink_direction=shrink_direction, same_timeslice=True, debug_mode=debug_mode)
    interpretation_step = auxcnot_shrink_target(interpretation_step, initial_target=t_block, merged_target=merged_target, shrink_direction=shrink_direction, same_timeslice=False, debug_mode=debug_mode)

    auxcnot_circ = interpretation_step.end_composite_operation_session_MUT()
    interpretation_step.append_circuit_MUT(auxcnot_circ, same_timeslice)
    return interpretation_step


def auxcnot_consistency_check(c_block: RotatedSurfaceCode, t_block: RotatedSurfaceCode) -> None:
    """Both blocks rotated surface codes of the same size and X-boundary orientation, their
    upper-left qubits the block's size plus one apart in both directions."""
    if not isinstance(c_block, RotatedSurfaceCode) or not isinstance(t_block, RotatedSurfaceCode):
        raise TypeError(f"The auxcnot operation is only supported for RotatedSurfaceCode blocks. {set(type(block) for block in [c_block, t_block])} types were given.")
    if c_block.size != t_block.size:
        raise ValueError(f"The blocks must have the same size to perform the auxiliary CNOT operation. The sizes of the blocks are {c_block.size}, {t_block.size}.")
    if c_block.x_boundary != t_block.x_boundary:
        raise ValueError(f"The blocks must have the same boundary orientations to perform the auxiliary CNOT operation. The X boundary orientations are {c_block.x_boundary} and {t_block.x_boundary}.")
    if abs(t_block.upper_left_qubit[0] - c_block.upper_left_qubit[0]) != c_block.size[0] + 1 or abs(t_block.upper_left_qubit[1] - c_block.upper_left_qubit[1]) != c_block.size[1] + 1:
        raise ValueError(
            "The blocks are not in the correct configuration for the auxiliary CNOT operation. The upper left corners of the blocks must satisfy the following relations: \n"
            "|t_block.upper_left_qubit[0] - c_block.upper_left_qubit[0]| = c_block.size[0] + 1, |t_block.upper_left_qubit[1] - c_block.upper_left_qubit[1]| = c_block.size[1] + 1"
            f"\nGot |{t_block.upper_left_qubit[0]} - {c_block.upper_left_qubit[0]}| = {c_block.size[0] + 1}, |{t_block.upper_left_qubit[1]} - {c_block.upper_left_qubit[1]}| = {c_block.size[1] + 1} instead."
        )


def get_grow_shrink_directions(control: RotatedSurfaceCode, target: RotatedSurfaceCode) -> tuple[Direction, Direction]:
    """The direction the control grows in (along its logical X, towards the target) and the
    one the merged target shrinks from (the auxiliary block's side)."""
    x = control.upper_left_qubit[0] - target.upper_left_qubit[0]
    y = control.upper_left_qubit[1] - target.upper_left_qubit[1]
    horizontal = control.x_boundary == Orientation.HORIZONTAL
    if x < 0 and y < 0:
        # The control is to the left of and above the target.
        return (Direction.RIGHT if horizontal else Direction.BOTTOM), (Direction.TOP if horizontal else Direction.LEFT)
    if x < 0 < y:
        # Left of and below.
        return (Direction.RIGHT if horizontal else Direction.TOP), (Direction.BOTTOM if horizontal else Direction.LEFT)
    if x > 0 > y:
        # Right of and above.
        return (Direction.LEFT if horizontal else Direction.BOTTOM), (Direction.TOP if horizontal else Direction.RIGHT)
    if x > 0 and y > 0:
        # Right of and below.
        return (Direction.LEFT if horizontal else Direction.TOP), (Direction.BOTTOM if horizontal else Direction.RIGHT)
    raise ValueError("The control block is not in the correct position relative to the target block.")


def auxcnot_grow_control(interpretation_step: InterpretationStep, control: RotatedSurfaceCode, target: RotatedSurfaceCode, grow_direction: Direction, same_timeslice: bool, debug_mode: bool) -> InterpretationStep:
    """Grow the control along its logical X by the target's extent in that direction plus one."""
    grow_length = target.size[0] + 1 if control.x_boundary == Orientation.HORIZONTAL else target.size[1] + 1
    return grow(interpretation_step, Grow(input_block_name=control.unique_label, direction=grow_direction, length=grow_length), same_timeslice=same_timeslice, debug_mode=debug_mode)


def auxcnot_split_control(interpretation_step: InterpretationStep, aux_unique_label: str, initial_control: RotatedSurfaceCode, initial_target: RotatedSurfaceCode, grown_control: RotatedSurfaceCode, same_timeslice: bool, debug_mode: bool) -> InterpretationStep:
    """Split the grown control across its logical X: the part that keeps the control's
    upper-left qubit keeps its label (and comes first), the other is the auxiliary block; the
    split falls after the size of whichever block was at the upper left."""
    split_orientation = grown_control.x_boundary.perpendicular()
    if initial_control.upper_left_qubit == grown_control.upper_left_qubit:
        split_blocks_name = (grown_control.unique_label, aux_unique_label)
        split_position = initial_control.size[0] if split_orientation == Orientation.VERTICAL else initial_control.size[1]
    else:
        split_blocks_name = (aux_unique_label, grown_control.unique_label)
        split_position = initial_target.size[0] if split_orientation == Orientation.VERTICAL else initial_target.size[1]
    split_op = Split(input_block_name=grown_control.unique_label, output_blocks_name=split_blocks_name, orientation=split_orientation, split_position=split_position)
    return split(interpretation_step, split_op, same_timeslice=same_timeslice, debug_mode=debug_mode)


def auxcnot_merge_aux_target(interpretation_step: InterpretationStep, aux: RotatedSurfaceCode, target: RotatedSurfaceCode, same_timeslice: bool, debug_mode: bool) -> InterpretationStep:
    """Merge the auxiliary block into the target; the merged block keeps the target's label."""
    merge_op = Merge(input_blocks_name=(aux.unique_label, target.unique_label), output_block_name=target.unique_label)
    return merge(interpretation_step, merge_op, same_timeslice=same_timeslice, debug_mode=debug_mode)


def auxcnot_shrink_target(interpretation_step: InterpretationStep, initial_target: RotatedSurfaceCode, merged_target: RotatedSurfaceCode, shrink_direction: Direction, same_timeslice: bool, debug_mode: bool) -> InterpretationStep:
    """Shrink the merged target back to the target's size, normal to its logical X."""
    shrink_length = merged_target.size[0] - initial_target.size[0] if merged_target.x_boundary == Orientation.VERTICAL else merged_target.size[1] - initial_target.size[1]
    shrink_op = Shrink(input_block_name=merged_target.unique_label, direction=shrink_direction, length=shrink_length)
    return shrink(interpretation_step, shrink_op, same_timeslice=same_timeslice, debug_mode=debug_mode)


def auxcnot_conditional_logical_z(interpretation_step: InterpretationStep, new_control: RotatedSurfaceCode, initial_target: RotatedSurfaceCode, aux_block: RotatedSurfaceCode, new_target: RotatedSurfaceCode, shrink_direction: Direction, same_timeslice: bool, debug_mode: bool) -> InterpretationStep:
    """A logical Z on the control conditioned on the merge's XX outcome, and on the merged
    target too when it shrinks from the top or the left (it then keeps the auxiliary block's
    logical X)."""
    log_meas = LogicalMeasurement((initial_target.unique_label, aux_block.unique_label), "XX")
    interpretation_step = conditional_logical_pauli(interpretation_step, ConditionalLogicalZ(new_control.unique_label, condition=log_meas), same_timeslice=same_timeslice, debug_mode=debug_mode)
    if shrink_direction in {Direction.TOP, Direction.LEFT}:
        interpretation_step = conditional_logical_pauli(interpretation_step, ConditionalLogicalZ(new_target.unique_label, condition=log_meas), same_timeslice=same_timeslice, debug_mode=debug_mode)
    return interpretation_step

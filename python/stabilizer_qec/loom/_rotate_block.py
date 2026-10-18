# Ported from Loom (entropicalabs/el-loom 0.4.0, src/loom_rotated_surface_code/applicator/rotate_block.py; Copyright
# 2024 Entropica Labs Pte Ltd, Apache License 2.0; see LICENSE in this directory), modified for stabilizer_qec: plain
# functions on this package's interpretation step.
"""Loom's RotateBlock: a d × d rotated surface code block (d odd) with its boundaries turned a
quarter, X boundaries where the Z boundaries were. The block grows by d - 1 towards the grow
direction, its logicals move to the top-left corner, five corner moves walk the boundaries
round, two shrinks bring it back to d × d one site over, and a move puts it back in place."""

from __future__ import annotations

from .eka import DiagonalDirection, Direction, Grow, MeasureBlockSyndromes, MoveBlock, Orientation, RotateBlock, Shrink
from .interpreter import InterpretationStep, grow, measureblocksyndromes, shrink
from .rotated_surface_code import RotatedSurfaceCode
from ._logical_phase import move_logical
from ._move_block import move_block
from ._move_corners import move_corners
from ._rsc_geometry import add_vector


def rotate_block(step: InterpretationStep, operation: RotateBlock, same_timeslice: bool, debug_mode: bool) -> InterpretationStep:
    rsc_block = step.get_block(operation.input_block_name)
    label = rsc_block.unique_label
    grow_direction = operation.grow_direction
    step.begin_composite_operation_session_MUT(same_timeslice=same_timeslice, circuit_name=f"Rotate boundary {label} by growing towards {grow_direction.name}")
    rotate_validity_check(rsc_block)
    distance = min(rsc_block.size)
    step = grow(step, Grow(label, grow_direction, length=distance - 1), same_timeslice=False, debug_mode=debug_mode)
    current = step.get_block(label)
    step = measureblocksyndromes(step, MeasureBlockSyndromes(label, n_cycles=distance), same_timeslice=False, debug_mode=debug_mode)
    top_left = current.upper_left_qubit
    for pauli in ["X", "Z"]:
        step, current = move_logical(step, current, new_up_left_qubit=top_left, pauli=pauli)
    for corner_args in get_boundary_rotation_corner_args_5_1(current):
        step = move_corners(step, current, corner_args, same_timeslice=False, debug_mode=debug_mode)
        current = step.get_block(label)
        step = measureblocksyndromes(step, MeasureBlockSyndromes(label, n_cycles=distance - 1), same_timeslice=False, debug_mode=debug_mode)
    step = shrink(step, Shrink(current.unique_label, grow_direction, length=distance - 2), same_timeslice=False, debug_mode=debug_mode)
    current = step.get_block(label)
    step = shrink(step, Shrink(current.unique_label, grow_direction.opposite(), length=1), same_timeslice=False, debug_mode=debug_mode)
    current = step.get_block(label)
    step = move_block(step, MoveBlock(current.unique_label, direction=grow_direction.opposite()), same_timeslice=False, debug_mode=debug_mode)
    current = step.get_block(label)
    top_left = current.upper_left_qubit
    for pauli in ["X", "Z"]:
        step, current = move_logical(step, current, new_up_left_qubit=top_left, pauli=pauli)
    circuit = step.end_composite_operation_session_MUT()
    step.append_circuit_MUT(circuit, same_timeslice)
    return step


def rotate_validity_check(rsc_block: RotatedSurfaceCode) -> None:
    if not isinstance(rsc_block, RotatedSurfaceCode):
        raise ValueError("Cannot apply RotateBlock to non-RotatedSurfaceCode blocks.")
    if not (rsc_block.size[0] == rsc_block.size[1] and rsc_block.size[0] % 2 == 1):
        raise ValueError("Cannot apply RotateBlock to non-square blocks or blocks with even distance.")
    if set(rsc_block.topological_corners) != set(rsc_block.geometric_corners):
        raise ValueError("Cannot apply RotateBlock to blocks when the topological and geometric corners do not match.")


def get_boundary_rotation_corner_args_5_1(rsc_block: RotatedSurfaceCode) -> tuple:
    """The five corner moves that turn the boundaries of a d × (2d - 1) block a quarter."""
    long_orientation = Orientation.HORIZONTAL if rsc_block.size[0] > rsc_block.size[1] else Orientation.VERTICAL
    tl = rsc_block.get_corner_from_direction(DiagonalDirection.from_directions((Direction.TOP, Direction.LEFT)))
    tr = rsc_block.get_corner_from_direction(DiagonalDirection.from_directions((Direction.TOP, Direction.RIGHT)))
    br = rsc_block.get_corner_from_direction(DiagonalDirection.from_directions((Direction.BOTTOM, Direction.RIGHT)))
    bl = rsc_block.get_corner_from_direction(DiagonalDirection.from_directions((Direction.BOTTOM, Direction.LEFT)))
    short, long = min(rsc_block.size), max(rsc_block.size)
    if long != 2 * short - 1 or short % 2 != 1:
        raise ValueError(f"Cannot obtain corner moves for blocks that are not of size (d, 2 * d -1) or (2 * d - 1, d) with odd d. Given size: {rsc_block.size}")
    s, l = short - 1, long - 1
    first_row = rsc_block.weight_2_stab_is_first_row
    if (first_row, long_orientation) == (False, Orientation.VERTICAL):
        return (((tl, Direction.BOTTOM, s),), ((tr, Direction.LEFT, s),), ((br, Direction.TOP, l),), ((bl, Direction.RIGHT, s),), ((add_vector(tl, Direction.BOTTOM, s), Direction.BOTTOM, s),))
    if (first_row, long_orientation) == (True, Orientation.VERTICAL):
        return (((tr, Direction.BOTTOM, s),), ((tl, Direction.RIGHT, s),), ((bl, Direction.TOP, l),), ((br, Direction.LEFT, s),), ((add_vector(tr, Direction.BOTTOM, s), Direction.BOTTOM, s),))
    if (first_row, long_orientation) == (False, Orientation.HORIZONTAL):
        return (((bl, Direction.RIGHT, s),), ((tl, Direction.BOTTOM, s),), ((tr, Direction.LEFT, l),), ((br, Direction.TOP, s),), ((add_vector(bl, Direction.RIGHT, s), Direction.RIGHT, s),))
    if (first_row, long_orientation) == (True, Orientation.HORIZONTAL):
        return (((tl, Direction.RIGHT, s),), ((bl, Direction.TOP, s),), ((br, Direction.LEFT, l),), ((tr, Direction.BOTTOM, s),), ((add_vector(tl, Direction.RIGHT, s), Direction.RIGHT, s),))
    raise ValueError("Invalid combination of weight_2_stab_is_first_row and long_orientation")

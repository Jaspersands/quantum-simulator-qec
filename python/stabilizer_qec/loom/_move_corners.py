# Ported from Loom (entropicalabs/el-loom 0.4.0, src/loom_rotated_surface_code/applicator/move_corners.py; Copyright
# 2024 Entropica Labs Pte Ltd, Apache License 2.0; see LICENSE in this directory), modified for stabilizer_qec: plain
# functions on this package's interpretation step, the new block's syndrome circuits in the order they were made.
"""Loom's corner moves, the helper of the rotate and Y-wall applicators: a topological corner of
a rotated surface code block slid along a boundary, the boundary's weight-2 stabilizers
redrawn (a weight-4 stabilizer cut to weight 3 when the corner leaves it), the logical
operators reshaped with the stabilizers they cross, new syndrome circuits for the block's new
corner configuration, then a round of syndromes."""

from __future__ import annotations

from typing import Optional, Sequence

from .circuit import ChannelType, Circuit, SyndromeCircuit
from .eka import DiagonalDirection, Direction, MeasureBlockSyndromes, Orientation, PauliOperator, Stabilizer
from .interpreter import InterpretationStep, measureblocksyndromes
from .rotated_surface_code import RotatedSurfaceCode
from ._rsc_geometry import find_relative_diagonal_direction, generate_syndrome_extraction_circuits

Coord = tuple[int, ...]


def move_corners(step: InterpretationStep, block: RotatedSurfaceCode, corner_args: Sequence[tuple[Coord, Direction, int]], same_timeslice: bool, debug_mode: bool) -> InterpretationStep:
    """Move each ``(corner qubit, direction, distance)`` corner in turn, then measure the
    block's syndromes once. A logical operator that changes picks up the last syndromes of the
    stabilizers it gave up now and of the stabilizers it takes on after that round."""
    initial_block_name = block.unique_label
    future: dict[str, tuple[Stabilizer, ...]] = {"X": (), "Z": ()}
    for corner_qubit, move_direction, how_far in corner_args:
        previous = (block.logical_x_operators[0], block.logical_z_operators[0])
        step, past_stabilizers, future_stabilizers = move_corner(step, block, corner_qubit, move_direction, how_far)
        future["X"] += future_stabilizers[0]
        future["Z"] += future_stabilizers[1]
        block = step.get_block(initial_block_name)
        for pauli, past_stabs, logical in zip(("X", "Z"), past_stabilizers, (block.logical_x_operators[0], block.logical_z_operators[0]), strict=True):
            cbits = tuple(cbit for stab in past_stabs for cbit in step.get_prev_syndrome(stab.uuid, block.uuid)[0].measurements)
            step.update_logical_operator_updates_MUT(pauli, logical.uuid, cbits, inherit_updates=logical not in previous)
    step = measureblocksyndromes(step, MeasureBlockSyndromes(initial_block_name), same_timeslice, debug_mode)
    for pauli, future_stabs in future.items():
        current = step.get_block(initial_block_name)
        logical = current.logical_x_operators[0] if pauli == "X" else current.logical_z_operators[0]
        cbits = tuple(cbit for stab in future_stabs for cbit in step.get_prev_syndrome(stab.uuid, block.uuid)[0].measurements)
        step.update_logical_operator_updates_MUT(pauli, logical.uuid, cbits, inherit_updates=False)
    return step


def move_corner(step: InterpretationStep, block: RotatedSurfaceCode, corner_qubit: Coord, move_direction: Direction, how_far: int):
    corner_directions = get_associated_boundaries(block, corner_qubit)
    modified = find_new_boundary_direction(corner_directions, move_direction)
    unit_vector = move_direction.to_vector()
    two_body_included = is_2_body_included(block, corner_qubit, modified)
    kept_boundary, new_boundary, cut_stabilizer, stab_evolution = find_new_boundary_stabilizers(block, corner_qubit, modified, unit_vector, how_far, two_body_included)
    to_remove = [s for s in block.boundary_stabilizers(modified) if s not in kept_boundary]
    step.stabilizer_evolution.update(stab_evolution)
    kept_bulk = [s for s in block.bulk_stabilizers if not (corner_qubit in s.data_qubits and two_body_included)]
    new_stabilizers = kept_boundary + new_boundary + kept_bulk
    if cut_stabilizer is not None:
        new_stabilizers.append(cut_stabilizer)
    new_x, new_z, x_evolution, z_evolution = move_corner_logical_operators(block=block, corner_qubit=corner_qubit, unit_vector=unit_vector, how_far=how_far, new_stabs_to_be_added=new_boundary, old_stabs_to_be_removed=to_remove)
    step.logical_x_evolution.update(x_evolution)
    step.logical_z_evolution.update(z_evolution)
    past_x = tuple(s for s in to_remove for pid in step.logical_x_evolution.get(new_x.uuid, ()) if s.uuid == pid)
    past_z = tuple(s for s in to_remove for pid in step.logical_z_evolution.get(new_z.uuid, ()) if s.uuid == pid)
    future_x = tuple(s for s in new_boundary for evolution in step.logical_x_evolution.values() for pid in evolution if s.uuid == pid)
    future_z = tuple(s for s in new_boundary for evolution in step.logical_z_evolution.values() for pid in evolution if s.uuid == pid)
    mock_block = RotatedSurfaceCode(unique_label=block.unique_label, stabilizers=new_stabilizers, logical_x_operators=[new_x], logical_z_operators=[new_z])
    config, pivot_corners = mock_block.config_and_pivot_corners
    starting = DiagonalDirection.TOP_RIGHT
    if config in (2, 3, 4):
        move_horizontal = move_direction.to_orientation() == Orientation.HORIZONTAL
        long_edge = move_horizontal == mock_block.is_horizontal
        compare = pivot_corners[0] if long_edge else pivot_corners[3]
        starting = find_relative_diagonal_direction(mock_block.upper_left_qubit, compare)
    new_circuits, new_stc = generate_syndrome_extraction_circuits(mock_block, starting)
    if isinstance(cut_stabilizer, Stabilizer):
        cut_circuit = cut_corner_syndrome_circuit(block, cut_stabilizer, corner_qubit, corner_directions)
        new_circuits += (cut_circuit,)
        new_stc[cut_stabilizer.uuid] = cut_circuit.uuid
    output = move_corner_circuit(step, block, corner_qubit, cut_stabilizer, move_direction, how_far)
    if output is not None:
        corner_circuit, new_cbit = output
        step.append_circuit_MUT(corner_circuit)
        step.stabilizer_updates[cut_stabilizer.uuid] = (new_cbit,)
        if corner_qubit in block.logical_x_operators[0].data_qubits:
            step.logical_x_operator_updates[new_x.uuid] = (new_cbit,)
        if corner_qubit in block.logical_z_operators[0].data_qubits:
            step.logical_z_operator_updates[new_z.uuid] = (new_cbit,)
    in_use = tuple(new_stc.values())
    new_circuits = tuple(c for c in new_circuits if c.uuid in in_use)
    new_block = RotatedSurfaceCode(unique_label=block.unique_label, stabilizers=new_stabilizers, logical_x_operators=[new_x], logical_z_operators=[new_z], syndrome_circuits=new_circuits, stabilizer_to_circuit=new_stc)
    step.update_block_history_and_evolution_MUT(new_blocks=(new_block,), old_blocks=(block,))
    return step, (past_x, past_z), (future_x, future_z)


def get_associated_boundaries(block: RotatedSurfaceCode, corner_qubit: Coord) -> tuple[Direction, ...]:
    if corner_qubit not in block.topological_corners:
        raise ValueError(f"The selected corner qubit {corner_qubit} is not a topological corner of the block `{block.unique_label}`")
    directions = tuple(d for d in Direction if corner_qubit in block.boundary_qubits(d))
    if not directions:
        raise ValueError(f"The corner qubit {corner_qubit} is not associated with any boundary")
    return directions


def find_new_boundary_direction(corner_boundaries: Sequence[Direction], move_direction: Direction) -> Direction:
    if (n := len(corner_boundaries)) not in (1, 2):
        raise ValueError(f"Invalid number of corner boundaries: {n}, must be either 1 or 2")
    if set(corner_boundaries) in ({Direction.TOP, Direction.BOTTOM}, {Direction.LEFT, Direction.RIGHT}):
        raise ValueError(f"Invalid corner boundaries: {corner_boundaries}, they must be orthogonal, e.g. (TOP, LEFT) or a single direction.")
    if move_direction in corner_boundaries:
        raise ValueError(f"Cannot move the corner {corner_boundaries} towards the {move_direction.value}")
    return next(d for d in corner_boundaries if d != move_direction.opposite())


def is_2_body_included(block: RotatedSurfaceCode, corner_qubit: Coord, boundary: Direction) -> bool:
    if corner_qubit not in (boundary_qubits := block.boundary_qubits(boundary)):
        raise ValueError(f"The corner qubit {corner_qubit} is not part of the {boundary.value} boundary")
    return any(s for s in block.stabilizers if corner_qubit in s.data_qubits and all(q in boundary_qubits for q in s.data_qubits))


def cut_corner_stabilizer(block: RotatedSurfaceCode, corner_qubit: Coord) -> Stabilizer:
    """The weight-4 stabilizer on the corner qubit without it."""
    stab = next(s for s in block.stabilizers if corner_qubit in s.data_qubits and len(s.data_qubits) == 4)
    paulis, qubits = zip(*((p, q) for p, q in zip(stab.pauli, stab.data_qubits, strict=True) if q != corner_qubit), strict=True)
    return Stabilizer(pauli="".join(paulis), data_qubits=qubits, ancilla_qubits=stab.ancilla_qubits)


def cut_corner_syndrome_circuit(block: RotatedSurfaceCode, cut_stabilizer: Stabilizer, corner_qubit: Coord, which_corner: Sequence[Direction]) -> SyndromeCircuit:
    """The cut stabilizer's circuit: the weight-4 schedule with the corner qubit's step empty."""
    stab = next(s for s in block.bulk_stabilizers if corner_qubit in s.data_qubits)
    vertical = next(d for d in which_corner if d in {Direction.TOP, Direction.BOTTOM})
    horizontal = next(d for d in which_corner if d in {Direction.LEFT, Direction.RIGHT})
    return RotatedSurfaceCode.generate_syndrome_circuit(pauli=cut_stabilizer.pauli, padding=[stab.data_qubits.index(corner_qubit)], name=f"{vertical}-{horizontal}-{cut_stabilizer.pauli}")


def generate_updated_2_body_stabilizers(old_corner_qubit: Coord, new_boundary_direction: Direction, unit_vector: tuple[int, int], how_far: int, pauli_type: str) -> list[Stabilizer]:
    if (abs(unit_vector[0]), abs(unit_vector[1])) not in ((1, 0), (0, 1)):
        raise ValueError("Invalid unit vector")
    orientation = Orientation.from_vector(unit_vector)
    bottom_or_right = new_boundary_direction in (Direction.RIGHT, Direction.BOTTOM)
    # Loom's precedence: (unit * how_far) % 2.
    adjusted = (old_corner_qubit[0] + unit_vector[0] * how_far % 2, old_corner_qubit[1] + unit_vector[1] * how_far % 2, 0)
    final = (old_corner_qubit[0] + unit_vector[0] * (how_far - 1), old_corner_qubit[1] + unit_vector[1] * (how_far - 1), 0)
    initial = min((adjusted, final), key=lambda x: x[0] + x[1])
    return RotatedSurfaceCode.generate_weight2_stabs(pauli=pauli_type * 2, initial_position=initial, num_stabs=how_far // 2, orientation=orientation, is_bottom_or_right=bottom_or_right)


def find_new_boundary_stabilizers(block: RotatedSurfaceCode, corner_qubit: Coord, modified_boundary_direction: Direction, unit_vector: tuple[int, int], how_far: int, two_body_is_included: bool) -> tuple[list[Stabilizer], list[Stabilizer], Optional[Stabilizer], dict[str, tuple[str, ...]]]:
    """The boundary stabilizers kept, the weight-2 stabilizers the corner's new stretch of
    boundary gets, the cut stabilizer (an odd move along a weight-2 stabilizer; an even one
    otherwise) and its evolution."""
    to_update = [(corner_qubit[0] + i * unit_vector[0], corner_qubit[1] + i * unit_vector[1], 0) for i in range(how_far + 1)]
    kept = [s for s in block.all_boundary_stabilizers if not all(q in to_update for q in s.data_qubits)]
    if two_body_is_included:
        if how_far % 2 == 0:
            raise ValueError("Only odd distances are allowed for moving the corner in the direction of the 2-body stabilizer")
        stab_to_cut = next(s for s in block.bulk_stabilizers if corner_qubit in s.data_qubits)
        cut = cut_corner_stabilizer(block, corner_qubit)
        evolution = {cut.uuid: (stab_to_cut.uuid,)}
    else:
        if how_far % 2 == 1:
            raise ValueError("Only even distances are allowed for moving the corner in the direction where there is no 2-body stabilizer")
        cut = None
        evolution = {}
    try:
        new = generate_updated_2_body_stabilizers(corner_qubit, modified_boundary_direction, unit_vector, how_far, block.boundary_type(modified_boundary_direction))
    except RuntimeError:
        charges = list(set(block.pauli_charges[q] for q in to_update) - set("Y"))
        new = generate_updated_2_body_stabilizers(corner_qubit, modified_boundary_direction, unit_vector, how_far, charges[0])
    return kept, new, cut, evolution


_PAULI_PRODUCT = {"X": {"X": "I", "Z": "Y", "Y": "Z", "I": "X"}, "Z": {"X": "Y", "Z": "I", "Y": "X", "I": "Z"}, "Y": {"X": "Z", "Z": "X", "Y": "I", "I": "Y"}, "I": {"X": "X", "Z": "Z", "Y": "Y", "I": "I"}}


def move_corner_logical_operators(block: RotatedSurfaceCode, corner_qubit: Coord, unit_vector: tuple[int, int], how_far: int, old_stabs_to_be_removed: Sequence[Stabilizer], new_stabs_to_be_added: Sequence[Stabilizer]) -> tuple[PauliOperator, PauliOperator, dict[str, tuple[str, ...]], dict[str, tuple[str, ...]]]:
    """Each logical operator on the corner's path, cut back to its end (when it runs the whole
    path) or multiplied by the removed boundary stabilizers from where it meets the path, with
    its evolution."""
    to_update = [(corner_qubit[0] + i * unit_vector[0], corner_qubit[1] + i * unit_vector[1], 0) for i in range(how_far + 1)]
    if len(block.logical_x_operators) > 1 or len(block.logical_z_operators) > 1:
        raise ValueError("Only a single logical qubit is supported")
    new_ops, evolutions = [], []
    for old in (block.logical_x_operators[0], block.logical_z_operators[0]):
        paulis = {q: p for p, q in zip(old.pauli, old.data_qubits, strict=True)}
        included = set(to_update) & set(old.data_qubits)
        if not included:
            new_op, evolution = old, {}
        elif set(to_update) == included:
            new_op = PauliOperator(pauli="".join(p for q, p in paulis.items() if q not in to_update[:-1]), data_qubits=tuple(q for q, p in paulis.items() if q not in to_update[:-1]))
            evolution = {new_op.uuid: (old.uuid,) + tuple(s.uuid for s in new_stabs_to_be_added)}
        else:
            if corner_qubit in old.data_qubits:
                anchor = corner_qubit
            else:
                old_qubits = set(old.data_qubits)
                intersections = [set(s.data_qubits) & old_qubits for s in new_stabs_to_be_added]
                anchor = next(next(iter(i)) for i in intersections if len(i) == 1)
            old_stab_paulis = {q: p for s in old_stabs_to_be_removed for p, q in zip(s.pauli, s.data_qubits, strict=True)}
            start = to_update.index(anchor)
            included_old = [s for s in old_stabs_to_be_removed if all(q in to_update[start:] for q in s.data_qubits)]
            for q in to_update[start:]:
                if q not in paulis:
                    paulis[q] = "I"
                if q in old_stab_paulis:
                    paulis[q] = _PAULI_PRODUCT[paulis[q]][old_stab_paulis[q]]
            new_op = PauliOperator(pauli="".join(p for p in paulis.values() if p != "I"), data_qubits=tuple(q for q, p in paulis.items() if p != "I"))
            evolution = {new_op.uuid: (old.uuid,) + tuple(s.uuid for s in included_old)}
        new_ops.append(new_op)
        evolutions.append(evolution)
    return new_ops[0], new_ops[1], evolutions[0], evolutions[1]


def move_corner_circuit(step: InterpretationStep, block: RotatedSurfaceCode, corner_qubit: Coord, cut_stabilizer: Optional[Stabilizer], move_direction: Direction, how_far: int):
    """When a stabilizer is cut, the corner qubit measured in the basis it had in it (Loom
    names the Z measurement ``measure``, which its Stim exporter, like this one, does not know,
    and labels the bit's channel without the bit's index, so the exporters leave the bit out
    of the detectors and observables)."""
    if isinstance(cut_stabilizer, Stabilizer):
        stab = next(s for s in block.bulk_stabilizers if corner_qubit in s.data_qubits)
        pauli = next(p for p, q in zip(stab.pauli, stab.data_qubits, strict=True) if q == corner_qubit)
        basis = "" if pauli == "Z" else f"_{pauli}"
        q_chan = step.get_channel_MUT(str(corner_qubit))
        c_chan = step.get_channel_MUT(f"c_{corner_qubit}", ChannelType.CLASSICAL)
        cbit = step.get_new_cbit_MUT(f"c_{corner_qubit}")
        return Circuit(name=f"moving corner {corner_qubit} to {move_direction} by {how_far}", circuit=((Circuit(f"measure{basis}", channels=[q_chan, c_chan]),),)), cbit
    return None

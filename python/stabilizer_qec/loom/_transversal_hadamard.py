# Ported from Loom (entropicalabs/el-loom 0.4.0, src/loom_rotated_surface_code/applicator/transversalhadamard.py;
# Copyright 2024 Entropica Labs Pte Ltd, Apache License 2.0; see LICENSE in this directory), modified for
# stabilizer_qec: plain classes in place of pydantic, explicit length checks in place of zip(strict=True).
"""Loom's transversal Hadamard on a rotated surface code block: H on every data qubit, the
stabilizers and logicals with X and Z exchanged (the logical X becomes the Z and the other way
round), and each syndrome circuit rewritten to measure its stabilizer's new Pauli string."""

from __future__ import annotations

from typing import Sequence

from .circuit import Circuit, SyndromeCircuit
from .eka import PauliOperator, Stabilizer, TransversalHadamard
from .interpreter import InterpretationStep
from .rotated_surface_code import RotatedSurfaceCode

_HADAMARD = {"X": "Z", "Z": "X"}


def update_synd_circ(input_synd_circ: SyndromeCircuit, new_pauli: str) -> SyndromeCircuit:
    """The syndrome circuit with its controlled Paulis (``cx``, ``cy``, ``cz``) replaced by the
    ones that measure ``new_pauli``, one gate per tick, and its name's Pauli suffix replaced
    (``left-xx`` becomes ``left-zz``)."""
    actual = input_synd_circ.circuit
    new_circ: list[tuple[Circuit, ...]] = []
    if len(new_pauli) != len(input_synd_circ.pauli):
        raise ValueError("The new stabilizer has a different pauli length than the input syndrome circuit.")
    iterator = 0
    for tick in actual.circuit:
        if len(tick) == 0:
            new_circ.append(())
            continue
        for gate in tick:
            if gate.name in ("cz", "cx", "cy"):
                new_circ.append((Circuit(name=f"c{new_pauli[iterator]}", channels=gate.channels),))
                iterator += 1
            else:
                new_circ.append((gate,))
    new_name = actual.name[: -1 * len(input_synd_circ.pauli)] + new_pauli.lower()
    # The channels are passed on so that they keep their order (data, ancilla, classical bit).
    new_circ_obj = Circuit(name=new_name, circuit=tuple(new_circ), channels=actual.channels)
    return SyndromeCircuit(pauli=new_pauli, name=new_name, circuit=new_circ_obj)


def transversal_hadamard_syndrome_circuits(block: RotatedSurfaceCode, interpretation_step: InterpretationStep, new_stabs: Sequence[Stabilizer], new_stabilizer_to_circuit: dict[str, str], new_synd_circs: tuple[SyndromeCircuit, ...] = ()) -> tuple[tuple[SyndromeCircuit, ...], dict[str, str]]:
    """The new stabilizers' syndrome circuits: each old stabilizer's circuit rewritten for its
    image, duplicates (equal but for the uuid) kept once, and the stabilizer-to-circuit map."""
    for new_stab in new_stabs:
        # If several stabilizers evolved into this one, the first decides.
        old_stab_uuid = interpretation_step.stabilizer_evolution[new_stab.uuid][0]
        old_synd_circ_uuid = block.stabilizer_to_circuit[old_stab_uuid]
        selected = next(c for c in block.syndrome_circuits if c.uuid == old_synd_circ_uuid)
        updated = update_synd_circ(selected, new_stab.pauli)
        duplicate = next((c for c in new_synd_circs if c == updated), None)
        if not duplicate:
            new_synd_circs += (updated,)
            new_stabilizer_to_circuit.update({new_stab.uuid: updated.uuid})
        else:
            new_stabilizer_to_circuit.update({new_stab.uuid: duplicate.uuid})
    return new_synd_circs, new_stabilizer_to_circuit


def transversalhadamard(interpretation_step: InterpretationStep, operation: TransversalHadamard, same_timeslice: bool, debug_mode: bool) -> InterpretationStep:
    """H on every data qubit of the block; the block that follows has each stabilizer with X
    and Z exchanged (its predecessor recorded, so that the next round's detectors compare the
    two), the logical X and Z exchanged likewise with their corrections carried over, and the
    syndrome circuits rewritten to match. A stabilizer with a Y is refused."""
    block = interpretation_step.get_block(operation.input_block_name)
    data_channels = tuple(interpretation_step.get_channel_MUT(str(q), "quantum") for q in block.data_qubits)
    hadamard_block = Circuit(name=f"Transversal Hadamard on {block.unique_label}", circuit=[[Circuit("H", channels=(channel,)) for channel in data_channels]])
    interpretation_step.append_circuit_MUT(hadamard_block, same_timeslice)

    new_stabs = tuple(Stabilizer(pauli="".join(_HADAMARD[p] for p in s.pauli), data_qubits=s.data_qubits, ancilla_qubits=s.ancilla_qubits) for s in block.stabilizers if "Y" not in s.pauli)
    if len(new_stabs) != len(block.stabilizers):
        raise ValueError("One of the Stabilizer in the Block contains a Pauli Y, which is not supported by Hadamard.")
    interpretation_step.stabilizer_evolution.update({new.uuid: (old.uuid,) for new, old in zip(new_stabs, block.stabilizers)})

    # The logical X operators become Z operators (their corrections filed as Z updates), and
    # the other way round.
    new_x_pauli_ops = tuple(PauliOperator(pauli="".join(_HADAMARD[p] for p in op.pauli), data_qubits=op.data_qubits) for op in block.logical_x_operators)
    for new_op, old_op in zip(new_x_pauli_ops, block.logical_x_operators):
        old_op_updates = interpretation_step.logical_x_operator_updates.get(old_op.uuid, ())
        interpretation_step.logical_x_evolution[new_op.uuid] = (old_op.uuid,)
        interpretation_step.update_logical_operator_updates_MUT(new_op.pauli[0], new_op.uuid, new_updates=old_op_updates, inherit_updates=False)
    new_z_pauli_ops = tuple(PauliOperator(pauli="".join(_HADAMARD[p] for p in op.pauli), data_qubits=op.data_qubits) for op in block.logical_z_operators)
    for new_op, old_op in zip(new_z_pauli_ops, block.logical_z_operators):
        old_op_updates = interpretation_step.logical_z_operator_updates.get(old_op.uuid, ())
        interpretation_step.logical_z_evolution[new_op.uuid] = (old_op.uuid,)
        interpretation_step.update_logical_operator_updates_MUT(new_op.pauli[0], new_op.uuid, new_updates=old_op_updates, inherit_updates=False)

    new_synd_circ_tuple, new_stabilizer_to_circuit = transversal_hadamard_syndrome_circuits(block, interpretation_step, new_stabs, {})
    new_block = RotatedSurfaceCode(stabilizers=new_stabs, logical_x_operators=new_z_pauli_ops, logical_z_operators=new_x_pauli_ops, syndrome_circuits=new_synd_circ_tuple, stabilizer_to_circuit=new_stabilizer_to_circuit, unique_label=block.unique_label, skip_validation=not debug_mode)
    interpretation_step.update_block_history_and_evolution_MUT((new_block,), (block,))
    return interpretation_step

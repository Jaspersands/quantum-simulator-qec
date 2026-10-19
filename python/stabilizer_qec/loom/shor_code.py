# Ported from Loom (entropicalabs/el-loom 0.4.0, src/loom_shor_code/code_factory/shor_code.py; Copyright 2024
# Entropica Labs Pte Ltd, Apache License 2.0; see LICENSE in this directory), modified for stabilizer_qec: a plain
# class in place of pydantic.
"""Loom's Shor-code factory: the [[9,1,3]] code on a linear lattice, two weight-6 X checks and
six weight-2 Z checks, with Loom's syndrome circuits (idling so that every check takes ten ticks)."""

from __future__ import annotations

from typing import Optional
from uuid import uuid4

from .circuit import Channel, Circuit, SyndromeCircuit
from .eka import Block, Lattice, LatticeType, PauliOperator, Stabilizer


class ShorCode(Block):
    """The Shor code: data qubits ``(0, 0)`` to ``(8, 0)``, X checks on ``(0..5, 0)`` and
    ``(3..8, 0)`` (ancillas ``(0, 1)``, ``(1, 1)``), Z checks on neighbours within each triple
    (ancillas ``(2, 1)`` to ``(7, 1)``), logicals X and Z on all nine."""

    @classmethod
    def create(cls, lattice: Lattice, unique_label: Optional[str] = None, position: tuple[int, ...] = (0,), skip_validation: bool = False) -> "ShorCode":
        if lattice.lattice_type != LatticeType.LINEAR:
            raise ValueError(f"The creation of Shor code blocks is currently only supported for 1D linear lattices. Instead the lattice is of type LatticeType.{LatticeType(lattice.lattice_type).name}.")
        if not isinstance(position, tuple) or any(not isinstance(x, int) for x in position):
            raise ValueError(f"`position` must be a tuple of integers. Got '{position}' instead.")
        if unique_label is None:
            unique_label = str(uuid4())
        x_indices = [[(i + 3 * j, 0) for i in range(6)] for j in range(2)]
        x_stabilizers = [Stabilizer(pauli="X" * 6, data_qubits=x_ind, ancilla_qubits=[(i, 1)]) for i, x_ind in enumerate(x_indices)]
        z_indices = [[(i + j + 3 * k, 0) for i in range(2)] for k in range(3) for j in range(2)]
        z_stabilizers = [Stabilizer(pauli="Z" * 2, data_qubits=z_ind, ancilla_qubits=[(i + 2, 1)]) for i, z_ind in enumerate(z_indices)]
        stabilizers = x_stabilizers + z_stabilizers
        x_syndrome_circuits = cls.generate_syndrome_extraction_circuits("X" * 6)
        z_syndrome_circuits = cls.generate_syndrome_extraction_circuits("Z" * 2)
        syndrome_circuits = [x_syndrome_circuits, z_syndrome_circuits]
        stabilizer_to_circuit = {**{stab.uuid: x_syndrome_circuits.uuid for stab in x_stabilizers}, **{stab.uuid: z_syndrome_circuits.uuid for stab in z_stabilizers}}
        all_datas = [(i, 0) for i in range(9)]
        logical_x_operator = PauliOperator(pauli="X" * 9, data_qubits=all_datas)
        logical_z_operator = PauliOperator(pauli="Z" * 9, data_qubits=all_datas)
        block = cls(unique_label=unique_label, stabilizers=stabilizers, logical_x_operators=[logical_x_operator], logical_z_operators=[logical_z_operator], syndrome_circuits=syndrome_circuits, stabilizer_to_circuit=stabilizer_to_circuit, skip_validation=skip_validation)
        if position == (0,):
            return block
        return block.shift(position)

    @staticmethod
    def generate_syndrome_extraction_circuits(pauli: str) -> SyndromeCircuit:
        """Reset, H, one controlled Pauli per data qubit, H, measurement; an X check idles two
        ticks after its CNOTs, a Z check six before its CZs."""
        pauli_type = pauli[0]
        supp = len(pauli)
        circ_name = f"{pauli}_syndrome_extraction"
        data_channels = [Channel(type="quantum", label=f"d{i}") for i in range(supp)]
        cbit_channel = Channel(type="classical", label="c0")
        ancilla_channel = Channel(type="quantum", label="a0")
        reset = [Circuit("Reset_0", channels=[ancilla_channel])]
        h_initial = [Circuit("H", channels=[ancilla_channel])]
        h_final = [Circuit("H", channels=[ancilla_channel])]
        entangling_layer = [[Circuit(f"C{p}", channels=[ancilla_channel, data_channels[i]])] for i, p in enumerate(pauli)]
        if pauli_type == "X":
            entangling_layer += [()] * 2
        else:
            entangling_layer = [()] * 6 + entangling_layer
        measurement = [Circuit("Measurement", channels=[ancilla_channel, cbit_channel])]
        circuit_list = [reset, h_initial] + entangling_layer + [h_final, measurement]
        return SyndromeCircuit(pauli=pauli, name=circ_name, circuit=Circuit(name=circ_name, circuit=circuit_list, channels=data_channels + [ancilla_channel, cbit_channel]))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ShorCode):
            raise NotImplementedError(f"Cannot compare ShorCode with {type(other)}")
        return super().__eq__(other)

    __hash__ = Block.__hash__

    def shift(self, position: tuple[int, ...], new_label: Optional[str] = None) -> "ShorCode":
        return super().shift(position, new_label)

    def rename(self, name: str) -> "ShorCode":
        return super().rename(name)

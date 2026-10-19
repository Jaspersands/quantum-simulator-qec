# Ported from Loom (entropicalabs/el-loom 0.4.0, src/loom_steane_code/code_factory/steane_code.py; Copyright 2024
# Entropica Labs Pte Ltd, Apache License 2.0; see LICENSE in this directory), modified for stabilizer_qec: a plain
# class in place of pydantic.
"""Loom's Steane-code factory: the [[7,1,3]] code on a 2D square lattice, three weight-4 X
checks and the same three as Z checks, with Loom's syndrome circuits (idling so that every check
takes ten ticks)."""

from __future__ import annotations

from typing import Optional
from uuid import uuid4

from .circuit import Channel, Circuit, SyndromeCircuit
from .eka import Block, Lattice, LatticeType, PauliOperator, Stabilizer


class SteaneCode(Block):
    """The Steane code: seven data qubits in the cells ``(0..2, 0)``, ``(0..2, 1)`` and
    ``(1, 2)``; X checks measured by the ancillas ``(i, 0, 1)``, Z checks by ``(0, i + 1, 1)``;
    logicals X and Z on the first row."""

    @classmethod
    def create(cls, lattice: Lattice, unique_label: Optional[str] = None, position: tuple[int, ...] = (0, 0), skip_validation: bool = False) -> "SteaneCode":
        if lattice.lattice_type != LatticeType.SQUARE_2D:
            raise ValueError(f"The creation of Steane code blocks is currently only supported for 2D square lattices. Instead the lattice is of type LatticeType.{LatticeType(lattice.lattice_type).name}.")
        if not isinstance(position, tuple) or any(not isinstance(x, int) for x in position):
            raise ValueError(f"`position` must be a tuple of integers. Got '{position}' instead.")
        if unique_label is None:
            unique_label = str(uuid4())
        stabilizer_supports = [
            [(0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0)],
            [(1, 0, 0), (1, 1, 0), (2, 1, 0), (2, 0, 0)],
            [(2, 1, 0), (1, 2, 0), (0, 1, 0), (1, 1, 0)],
        ]
        x_stabilizers = [Stabilizer(pauli="X" * 4, data_qubits=supp, ancilla_qubits=[(i, 0, 1)]) for i, supp in enumerate(stabilizer_supports)]
        z_stabilizers = [Stabilizer(pauli="Z" * 4, data_qubits=supp, ancilla_qubits=[(0, i + 1, 1)]) for i, supp in enumerate(stabilizer_supports)]
        stabilizers = x_stabilizers + z_stabilizers
        logical_data_qubits = [(i, 0, 0) for i in range(3)]
        logical_x_operator = PauliOperator(pauli="XXX", data_qubits=logical_data_qubits)
        logical_z_operator = PauliOperator(pauli="ZZZ", data_qubits=logical_data_qubits)
        x_syndrome_circuits = cls.generate_syndrome_extraction_circuits("XXXX")
        z_syndrome_circuits = cls.generate_syndrome_extraction_circuits("ZZZZ")
        syndrome_circuits = [x_syndrome_circuits, z_syndrome_circuits]
        stabilizer_to_circuit = {**{stab.uuid: x_syndrome_circuits.uuid for stab in x_stabilizers}, **{stab.uuid: z_syndrome_circuits.uuid for stab in z_stabilizers}}
        block = cls(unique_label=unique_label, stabilizers=stabilizers, logical_x_operators=[logical_x_operator], logical_z_operators=[logical_z_operator], syndrome_circuits=syndrome_circuits, stabilizer_to_circuit=stabilizer_to_circuit, skip_validation=skip_validation)
        if position == (0, 0):
            return block
        return block.shift(position)

    @staticmethod
    def generate_syndrome_extraction_circuits(pauli: str) -> SyndromeCircuit:
        """Reset, H, one controlled Pauli per data qubit, H, measurement; an X check idles four
        ticks after its CNOTs, a Z check four before its CZs."""
        name = f"{pauli}_syndrome_extraction"
        pauli_type = pauli[0]
        data_channels = [Channel(type="quantum", label=f"d{i}") for i in range(4)]
        cbit_channel = Channel(type="classical", label="c0")
        ancilla_channel = Channel(type="quantum", label="a0")
        hadamard1 = (Circuit("H", channels=[ancilla_channel]),)
        hadamard2 = (Circuit("H", channels=[ancilla_channel]),)
        entangle_ancilla = [[Circuit(f"C{p}", channels=[ancilla_channel, data_channels[i]])] for i, p in enumerate(pauli)]
        if pauli_type == "Z":
            entangle_ancilla = [(), (), (), ()] + entangle_ancilla
        else:
            entangle_ancilla += [(), (), (), ()]
        measurement = (Circuit("Measurement", channels=[ancilla_channel, cbit_channel]),)
        reset = (Circuit("Reset_0", channels=[ancilla_channel]),)
        circuit_list = [reset, hadamard1] + entangle_ancilla + [hadamard2, measurement]
        return SyndromeCircuit(pauli=pauli, name=name, circuit=Circuit(name=name, circuit=circuit_list, channels=data_channels + [ancilla_channel, cbit_channel]))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SteaneCode):
            raise NotImplementedError(f"Cannot compare SteaneCode with {type(other)}")
        return super().__eq__(other)

    __hash__ = Block.__hash__

    def shift(self, position: tuple[int, ...], new_label: Optional[str] = None) -> "SteaneCode":
        return super().shift(position, new_label)

    def rename(self, name: str) -> "SteaneCode":
        return super().rename(name)

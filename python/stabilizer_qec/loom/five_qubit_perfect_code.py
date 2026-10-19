# Ported from Loom (entropicalabs/el-loom 0.4.0, src/loom_five_qubit_perfect_code/code_factory/five_qubit_perfect_code.py;
# Copyright 2024 Entropica Labs Pte Ltd, Apache License 2.0; see LICENSE in this directory), modified for
# stabilizer_qec: a plain class in place of pydantic.
"""Loom's five-qubit-perfect-code factory: the [[5,1,3]] code in one pentagonal cell of a
``poly_2d`` lattice (five data qubits and four ancillas), its four cyclic checks XZZXI, IXZZX,
XIXZZ and ZXIXZ, with Loom's syndrome circuits (one tick per position, idling on the identity)."""

from __future__ import annotations

from typing import Optional
from uuid import uuid4

from .circuit import Channel, Circuit, SyndromeCircuit
from .eka import Block, Lattice, LatticeType, PauliOperator, Stabilizer


class FiveQubitPerfectCode(Block):
    """The five-qubit perfect code: data qubits ``(0, 0, 0)`` to ``(0, 0, 4)``, the check
    ``i`` measured by the ancilla ``(0, 0, 5 + i)``, logicals X and Z on all five."""

    @classmethod
    def create(cls, lattice: Lattice, unique_label: Optional[str] = None, position: tuple[int, int] = (0, 0), skip_validation: bool = False) -> "FiveQubitPerfectCode":
        if lattice.lattice_type != LatticeType.POLY_2D:
            raise ValueError(f"The creation of five qubit perfect code blocks is currently only supported for 2D pentagonal lattices. Instead the lattice is of type LatticeType.{LatticeType(lattice.lattice_type).name}.")
        if lattice.unit_cell_size != 9:
            raise ValueError(f"The five qubit perfect code block requires a lattice with exactly 5 qubits and 4 ancilla in each unit cell. Got {lattice.unit_cell_size} qubits and ancilla instead.")
        if not isinstance(position, tuple) or any(not isinstance(x, int) for x in position):
            raise ValueError(f"`position` must be a tuple of integers. Got '{position}' instead.")
        if unique_label is None:
            unique_label = str(uuid4())
        looped_stabilizer_supports = [(0, 0, i % 5) for i in range(8)]
        stab_labels = ["XZZX", "XZZX", "XXZZ", "ZXXZ"]
        stabilizers = [Stabilizer(pauli=stab_labels[i], data_qubits=sorted(looped_stabilizer_supports[i : i + 4]), ancilla_qubits=[(0, 0, 5 + i)]) for i in range(4)]
        all_data_qubits = [(0, 0, i) for i in range(5)]
        logical_x_operator = PauliOperator(pauli="X" * 5, data_qubits=all_data_qubits)
        logical_z_operator = PauliOperator(pauli="Z" * 5, data_qubits=all_data_qubits)
        syndrome_circuits = [cls.generate_syndrome_extraction_circuits("XZZXI"), cls.generate_syndrome_extraction_circuits("IXZZX"), cls.generate_syndrome_extraction_circuits("XIXZZ"), cls.generate_syndrome_extraction_circuits("ZXIXZ")]
        stabilizer_to_circuit = {stab.uuid: syndrome_circuits[i].uuid for i, stab in enumerate(stabilizers)}
        block = cls(unique_label=unique_label, stabilizers=stabilizers, logical_x_operators=[logical_x_operator], logical_z_operators=[logical_z_operator], syndrome_circuits=syndrome_circuits, stabilizer_to_circuit=stabilizer_to_circuit, skip_validation=skip_validation)
        if position == (0, 0):
            return block
        return block.shift(position)

    @staticmethod
    def generate_syndrome_extraction_circuits(pauli: str) -> SyndromeCircuit:
        """Reset, H, the five positions of the cyclic check (an idle tick on its identity),
        rotated so that every check runs its positions in step, H, measurement; the circuit's
        Pauli string drops the identity."""
        p_index = pauli.index("I")
        start_t_index = (p_index + 1) % 5
        name = f"{pauli}_syndrome_extraction"
        data_channels = [Channel(type="quantum", label=f"d{i}") for i in range(4)]
        anc_channel = [Channel(type="quantum", label="a0")]
        cbit_channel = [Channel(type="classical", label="c0")]
        reset_layer = [Circuit(name="Reset_0", channels=anc_channel)]
        hadamard_initial = [Circuit(name="H", channels=anc_channel)]
        hadamard_final = [Circuit(name="H", channels=anc_channel)]
        measurement_layer = [Circuit(name="Measurement", channels=anc_channel + cbit_channel)]
        insert_entangling_layer: list = []
        i = 0
        for p in pauli:
            if p == "I":
                insert_entangling_layer.append([])
            else:
                insert_entangling_layer.append([Circuit(name=f"C{p}", channels=anc_channel + [data_channels[i]])])
                i += 1
        entangling_layer: list = [[], [], []]
        entangling_layer = entangling_layer[:start_t_index] + insert_entangling_layer + entangling_layer[start_t_index:]
        circuit_list = [reset_layer, hadamard_initial] + entangling_layer + [hadamard_final, measurement_layer]
        return SyndromeCircuit(pauli=pauli[:p_index] + pauli[p_index + 1 :], name=name, circuit=Circuit(name=name, circuit=circuit_list, channels=data_channels + anc_channel + cbit_channel))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FiveQubitPerfectCode):
            raise NotImplementedError(f"Cannot compare FiveQubitPerfectCode with {type(other)}")
        return super().__eq__(other)

    __hash__ = Block.__hash__

    def shift(self, position: tuple[int, ...], new_label: Optional[str] = None) -> "FiveQubitPerfectCode":
        return super().shift(position, new_label)

    def rename(self, name: str) -> "FiveQubitPerfectCode":
        return super().rename(name)

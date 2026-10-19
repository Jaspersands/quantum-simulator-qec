# Ported from Loom (entropicalabs/el-loom 0.4.0, src/loom_repetition_code/code_factory/repetition_code.py;
# Copyright 2024 Entropica Labs Pte Ltd, Apache License 2.0; see LICENSE in this directory), modified for
# stabilizer_qec: a plain class in place of pydantic. Its grow, shrink, merge and split are in _repetition_code_ops.py.
"""Loom's repetition-code factory: a chain of ``d`` data qubits on a linear lattice, checked by
the weight-2 X or Z stabilizers of neighbours, the short logical on the first qubit and the long
one on all of them."""

from __future__ import annotations

from typing import Optional, Union
from uuid import uuid4

from .eka import Block, Coord, Direction, Lattice, LatticeType, PauliOperator, Stabilizer


class RepetitionCode(Block):
    """A repetition chain: data qubits ``(i, 0)``, the check of ``(i, 0)`` and ``(i + 1, 0)``
    measured by the ancilla ``(i, 1)``."""

    @classmethod
    def create(cls, d: int, check_type: str, lattice: Lattice, unique_label: Optional[str] = None, position: tuple[int, ...] = (0,), logical_x_operator: Optional[PauliOperator] = None, logical_z_operator: Optional[PauliOperator] = None, skip_validation: bool = False) -> "RepetitionCode":
        """A distance-``d`` chain of ``check_type`` ("X" or "Z") checks at ``position`` (its
        first data qubit); the logical of the check type is on the first qubit, the other on all
        ``d`` (either can be given; the long one must have weight ``d``)."""
        if not isinstance(d, int) or d <= 0:
            raise ValueError(f"`d` must be a positive integer. Got '{d}' instead.")
        if check_type not in ["X", "Z"]:
            raise ValueError(f"`check_type` must be either 'X' or 'Z'. Got '{check_type}' instead.")
        if lattice.lattice_type != LatticeType.LINEAR:
            raise ValueError(f"The creation of repetition chains is currently only supported for linear lattices. Instead the lattice is of type LatticeType.{LatticeType(lattice.lattice_type).name}.")
        if not isinstance(position, tuple) or any(not isinstance(x, int) for x in position):
            raise ValueError(f"`position` must be a tuple of integers. Got '{position}' instead.")
        if len(position) != lattice.n_dimensions:
            raise ValueError(f"`position` has length {len(position)} while length {lattice.n_dimensions} is required to match the lattice dimension.")
        if unique_label is None:
            unique_label = str(uuid4())
        if logical_x_operator is not None and check_type == "Z":
            if len(logical_x_operator.pauli) != d:
                raise ValueError("Support of input X logical should be equal to distance")
        if logical_z_operator is not None and check_type == "X":
            if len(logical_z_operator.pauli) != d:
                raise ValueError("Support of input Z logical should be equal to distance")
        stabilizers = [Stabilizer(pauli=check_type * 2, data_qubits=[(i, 0), (i + 1, 0)], ancilla_qubits=[(i, 1)]) for i in range(d - 1)]
        if logical_x_operator is None:
            logical_x_operator = PauliOperator(pauli=check_type, data_qubits=[(0, 0)]) if check_type == "X" else PauliOperator(pauli="X" * d, data_qubits=[(i, 0) for i in range(d)])
        if logical_z_operator is None:
            logical_z_operator = PauliOperator(pauli=check_type, data_qubits=[(0, 0)]) if check_type == "Z" else PauliOperator(pauli="Z" * d, data_qubits=[(i, 0) for i in range(d)])
        block = cls(unique_label=unique_label, stabilizers=stabilizers, logical_x_operators=[logical_x_operator], logical_z_operators=[logical_z_operator], skip_validation=skip_validation)
        if position == (0,):
            return block
        return block.shift(position)

    @property
    def check_type(self) -> str:
        return self.stabilizers[0].pauli[0]

    def boundary_qubits(self, direction: Union[Direction, str]) -> Coord:
        """The data qubit at the left or right end of the chain."""
        if not isinstance(direction, Direction):
            direction = Direction(direction)
        if direction not in [Direction.LEFT, Direction.RIGHT]:
            raise ValueError(f"Invalid direction '{direction.value}'. Only 'left' and 'right' are supported.")
        selector_function = max if direction is Direction.RIGHT else min
        return selector_function(self.data_qubits, key=lambda x: x[0])

    def get_shifted_equivalent_logical_operator(self, new_qubit: Coord) -> tuple[PauliOperator, tuple[Stabilizer, ...]]:
        """The short logical moved to ``new_qubit``, with the stabilizers between the old and
        the new qubit whose product relates them (none when it stays)."""
        if new_qubit not in self.data_qubits:
            raise ValueError(f"New logical position {new_qubit} is not part of the data qubits")
        short_logical = self.logical_x_operators[0] if self.check_type == "X" else self.logical_z_operators[0]
        short_logical_data_qubit = short_logical.data_qubits[0]
        if new_qubit != short_logical_data_qubit:
            shifted_operator = PauliOperator(pauli=self.check_type, data_qubits=[new_qubit])
        else:
            shifted_operator = short_logical
        if new_qubit < short_logical_data_qubit:
            required_stabilizers = tuple(stab for stab in self.stabilizers if all((q[0] >= new_qubit[0] and q[0] <= short_logical_data_qubit[0]) for q in stab.data_qubits))
        elif new_qubit > short_logical_data_qubit:
            required_stabilizers = tuple(stab for stab in self.stabilizers if all((q[0] <= new_qubit[0] and q[0] >= short_logical_data_qubit[0]) for q in stab.data_qubits))
        else:
            required_stabilizers = ()
        return shifted_operator, required_stabilizers

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RepetitionCode):
            raise NotImplementedError(f"Cannot compare RepetitionCode with {type(other)}")
        return super().__eq__(other)

    __hash__ = Block.__hash__

    def shift(self, position: tuple[int, ...], new_label: Optional[str] = None) -> "RepetitionCode":
        return super().shift(position, new_label)

    def rename(self, name: str) -> "RepetitionCode":
        return super().rename(name)

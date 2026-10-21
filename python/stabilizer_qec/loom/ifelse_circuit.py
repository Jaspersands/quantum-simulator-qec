# BoolOp is ported from Loom (entropicalabs/el-loom 0.4.0, src/loom/eka/utilities/enums.py; Copyright 2024 Entropica
# Labs Pte Ltd, Apache License 2.0; see LICENSE in this directory). IfElseCircuit is written for stabilizer_qec to
# behave as Loom's (src/loom/eka/ifelse_circuit.py, whose file header, unlike the rest of el-loom, reserves its
# rights): the same fields, defaults, derived channels and duration, string forms and messages, recorded from
# el-loom 0.4.0 in tests/loom_port/fixtures_ifelse.json.
"""Loom's branching circuit: ``IfElseCircuit`` runs its if branch or its else branch on a
classical condition, a ``BoolOp`` gate on classical channels (``match`` true on a 1, ``not`` on a
0; ``and``, ``nand``, ``or``, ``nor`` and ``xor`` over two or more bits). The converters write it
as their target's conditional (OpenQASM, CUDA-Q, PennyLane and Guppy any condition and branches;
Stim a single-bit condition on a single Pauli, Mimiq a single gate on ``match``, ``not``, ``and``
or ``nor``)."""

from __future__ import annotations

import textwrap
from enum import Enum
from typing import Optional
from uuid import uuid4

from .circuit import Channel, ChannelType, Circuit


class BoolOp(str, Enum):
    """The classical conditions, by name (any case): ``and`` (all bits 1), ``nand`` (some bit 0),
    ``or`` (some bit 1), ``nor`` (all bits 0), ``xor`` (an odd number of 1s) on two or more bits;
    ``not`` (the bit is 0) and ``match`` (the bit is 1) on one."""

    AND = "and"
    NAND = "nand"
    OR = "or"
    NOR = "nor"
    XOR = "xor"
    NOT = "not"
    MATCH = "match"

    @classmethod
    def _missing_(cls, value):
        value = value.lower()
        for member in cls:
            if member.value == value:
                return member
        raise ValueError(f"`{value}` is an invalid input for enum `{cls.__name__}`. Only the following values are allowed as an input: {', '.join(member.value for member in cls)}.")

    def __str__(self) -> str:
        return str(self.value)

    @staticmethod
    def multi_bit_list() -> list[BoolOp]:
        return [BoolOp.AND, BoolOp.NAND, BoolOp.OR, BoolOp.NOR, BoolOp.XOR]

    @staticmethod
    def mono_bit_list() -> list[BoolOp]:
        return [BoolOp.NOT, BoolOp.MATCH]


_EMPTY = "empty_branch"


class IfElseCircuit(Circuit):
    """``if_circuit`` or ``else_circuit`` on ``condition_circuit``, a ``BoolOp`` gate on classical
    channels (by default ``match`` on a new one). A missing branch is the empty circuit
    ``empty_branch``; a gate as a branch is wrapped in a circuit ``wrapped_<gate>``. Its name is
    ``if-else_circuit``, its ``circuit`` the one slice ``((if_circuit, else_circuit),)``, its
    channels the condition's then the branches' (quantum, then classical), its duration the longer
    branch's; ``duration`` given is overridden, as in Loom. Two are equal when their branches and
    conditions are. Loom's Circuit methods that make no sense for it (``as_gate``,
    ``circuit_seq``, ``construct_padded_circuit_time_sequence``) raise ``NotImplementedError``.
    The arguments are keywords only (Loom's positional order starts with the fields it fixes and
    drops what is passed there)."""

    __slots__ = ("if_circuit", "else_circuit", "condition_circuit")

    # How Circuit's own methods tell an IfElseCircuit from a composite circuit (Loom's marker).
    _loom_ifelse_marker = True

    def __init__(self, *, if_circuit: Optional[Circuit] = None, else_circuit: Optional[Circuit] = None, condition_circuit: Optional[Circuit] = None, duration: Optional[int] = None, description: str = "", id: Optional[str] = None) -> None:
        self.if_circuit = self._branch("if_circuit", if_circuit)
        self.else_circuit = self._branch("else_circuit", else_circuit)
        self.condition_circuit = self._condition(condition_circuit)
        self.name = "if-else_circuit"
        self.circuit = ((self.if_circuit, self.else_circuit),)
        # Loom gathers the branches' channels in a set; here in order of appearance.
        branch_channels = dict.fromkeys(self.if_circuit.channels + self.else_circuit.channels)
        order = (ChannelType.QUANTUM, ChannelType.CLASSICAL)
        self.channels = tuple(self.condition_circuit.channels) + tuple(sorted(branch_channels, key=lambda c: order.index(c.type)))
        self.duration = max(self.if_circuit.duration, self.else_circuit.duration)
        self.description = description
        self.id = id or str(uuid4())

    @staticmethod
    def _branch(field: str, circuit: Optional[Circuit]) -> Circuit:
        if circuit is None:
            return Circuit(_EMPTY)
        if not isinstance(circuit, Circuit):
            raise ValueError(f"{field}: Input should be an instance of Circuit")
        if circuit.name == _EMPTY:
            return Circuit(_EMPTY)
        if not circuit.circuit:
            # The wrapper keeps the gate's channel order (Loom's derives it from a set).
            return Circuit(f"wrapped_{circuit.name}", circuit=circuit, channels=circuit.channels)
        return circuit

    @staticmethod
    def _condition(circuit: Optional[Circuit]) -> Circuit:
        if circuit is None:
            return Circuit(BoolOp.MATCH, channels=[Channel(type=ChannelType.CLASSICAL)])
        if not isinstance(circuit, Circuit):
            raise ValueError("condition_circuit: Input should be an instance of Circuit")
        if not all(channel.is_classical() for channel in circuit.channels):
            raise ValueError(f"IfElseCircuit `condition_circuit` must be a circuit with classical channels only. Found channels: {circuit.channels}")
        if circuit.name in BoolOp.multi_bit_list():
            if len(circuit.channels) < 2:
                raise ValueError(f"Condition circuit with BoolOp '{circuit.name}' must have at least two classical channels.")
        elif circuit.name in BoolOp.mono_bit_list():
            if len(circuit.channels) != 1:
                raise ValueError(f"Condition circuit with BoolOp '{circuit.name}' must have only one classical channel.")
        else:
            raise ValueError(f"Unsupported BoolOp '{circuit.name}' for condition circuit. Supported BoolOps are: {', '.join(BoolOp.multi_bit_list() + BoolOp.mono_bit_list())}.")
        return circuit

    @classmethod
    def as_gate(cls):
        raise NotImplementedError("IfElseCircuit cannot be represented as a gate.")

    def circuit_seq(self):
        raise NotImplementedError("IfElseCircuit cannot be converted into a Circuit sequence.")

    @staticmethod
    def construct_padded_circuit_time_sequence():
        raise NotImplementedError("IfElseCircuit cannot construct a padded circuit time sequence.")

    def flatten(self) -> IfElseCircuit:
        """The branches and the condition flattened (a gate flattens to a one-gate circuit)."""
        return IfElseCircuit(if_circuit=self.if_circuit.flatten(), else_circuit=self.else_circuit.flatten(), condition_circuit=self.condition_circuit.flatten())

    @classmethod
    def unroll(cls, input_circuit: IfElseCircuit) -> tuple[IfElseCircuit, ...]:
        """The if/else with each branch unrolled into a circuit of its name (one element, kept
        whole in its tick by ``Circuit.unroll``)."""
        branches = []
        for branch in (input_circuit.if_circuit, input_circuit.else_circuit):
            branches.append(Circuit(branch.name, circuit=branch.unroll(branch), channels=branch.channels))
        return (IfElseCircuit(if_circuit=branches[0], else_circuit=branches[1], condition_circuit=input_circuit.condition_circuit),)

    @property
    def is_condition_single_bit(self) -> bool:
        return self.condition_circuit.name in BoolOp.mono_bit_list()

    @property
    def is_single_gate_conditioned(self) -> bool:
        """One gate in the if branch, in one tick, and no else branch."""
        branch = self.if_circuit.circuit
        return len(branch) == 1 and len(branch[0]) == 1 and not branch[0][0].circuit and self.else_circuit.name == _EMPTY

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, IfElseCircuit):
            return False
        return self.if_circuit == other.if_circuit and self.else_circuit == other.else_circuit and self.condition_circuit == other.condition_circuit

    __hash__ = Circuit.__hash__

    def __repr__(self) -> str:
        return f"{self.name}\n  if: {self.if_circuit.name}\n  else: {self.else_circuit.name}\n  condition: {self.condition_circuit.name}"

    def detailed_str(self) -> str:
        """The branches' gates (each branch's detailed lines, indented) and the condition."""

        def body(branch: Circuit) -> str:
            return textwrap.indent("\n".join(branch.detailed_str().splitlines()[1:]), "    ")

        return f"{self.name}\n  if: {self.if_circuit.name}\n{body(self.if_circuit)}\n  else: {self.else_circuit.name}\n{body(self.else_circuit)}\n  condition: {self.condition_circuit.name}\n"

    def asdict(self) -> dict:
        """Loom's dataclass fields, the branches and the condition included."""
        return {**super().asdict(), "if_circuit": self.if_circuit.asdict(), "else_circuit": self.else_circuit.asdict(), "condition_circuit": self.condition_circuit.asdict()}

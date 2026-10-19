# Ported from Loom (entropicalabs/el-loom 0.4.0, src/loom/validator/{circuit_validation,circuit_validation_wrappers,
# check_code_stabilizers,check_logical_ops,check_stabilizer_measurement,utilities,debug_dataclass}.py and
# src/loom/eka/logical_state.py; Copyright 2024 Entropica Labs Pte Ltd, Apache License 2.0; see LICENSE in this
# directory), modified for stabilizer_qec: this package's TableauSimulator in place of Loom's cliffordsim, a small
# signed-Pauli algebra in place of its stabilizer arrays, and plain dataclasses (Loom's fields, messages and
# printing) in place of the pydantic ones of DebugData, its checks and their outputs.
"""Loom's circuit validator: is a circuit (in Loom's IR) a valid operation from an input code to
an output code? The output code's stabilizers must stabilize the final state (with the parity
the classical bits predict), the logical states must transform as declared, and the named
measurements must measure the named input stabilizers, deterministically."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import reduce
from itertools import product
from typing import Any, Optional, Sequence, Union

from .circuit import Channel, Circuit
from .eka import Block, Coord, PauliOperator, Stabilizer, gf2_rank

# ---------------------------------------------------------------------------------------------
# A signed Pauli algebra: (phase as a power of i, x mask, z mask) over n qubits
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SignedPauli:
    """``i^phase · Π_j X_j^{x_j} Z_j^{z_j}``: a Pauli operator with a phase (a power of i). A
    Hermitian operator with ``k`` Ys has ``phase ≡ k (+2 for a minus sign)``, since Y = iXZ;
    the constructors build Hermitian operators."""

    n: int
    x: int
    z: int
    phase: int = 0

    @classmethod
    def identity(cls, n: int) -> "SignedPauli":
        return cls(n, 0, 0, 0)

    @classmethod
    def from_letters(cls, letters: str, n: Optional[int] = None, sign: str = "+") -> "SignedPauli":
        n = len(letters) if n is None else n
        x = z = 0
        for i, c in enumerate(letters):
            if c in "XY":
                x |= 1 << i
            if c in "ZY":
                z |= 1 << i
        return cls(n, x, z, ({"+": 0, "-": 2, "+i": 1, "-i": 3}[sign] + bin(x & z).count("1")) % 4)

    @classmethod
    def from_sparse_string(cls, text: str, n: Optional[int] = None) -> "SignedPauli":
        """``"+Z0"``, ``"-X0X1"``, ``"Y2"``: letters with qubit indices."""
        text = text.strip().upper()
        sign = "+"
        if text[:2] in ("+I", "-I"):
            sign, text = text[0] + "i", text[2:]
        elif text[:1] in "+-":
            sign, text = text[0], text[1:]
        pairs: list[tuple[str, int]] = []
        i = 0
        while i < len(text):
            letter = text[i]
            if letter not in "XYZ_":
                raise ValueError(f"Bad sparse Pauli string {text!r}.")
            j = i + 1
            while j < len(text) and text[j].isdigit():
                j += 1
            if j == i + 1:
                raise ValueError(f"Bad sparse Pauli string {text!r}: {letter} needs a qubit index.")
            pairs.append((letter, int(text[i + 1 : j])))
            i = j
        width = max((q for _, q in pairs), default=-1) + 1
        n = width if n is None else n
        if width > n:
            raise ValueError(f"{text!r} needs {width} qubits, not {n}.")
        x = z = 0
        for letter, q in pairs:
            if letter in "XY":
                x ^= 1 << q
            if letter in "ZY":
                z ^= 1 << q
        return cls(n, x, z, ({"+": 0, "-": 2, "+i": 1, "-i": 3}[sign] + bin(x & z).count("1")) % 4)

    def letters(self) -> str:
        return "".join("_XZY"[((self.x >> i) & 1) | (((self.z >> i) & 1) << 1)] for i in range(self.n))

    @property
    def sign_phase(self) -> int:
        """The phase with the Ys' factors of i taken out: 0 for +, 2 for −."""
        return (self.phase - bin(self.x & self.z).count("1")) % 4

    def as_sparse_string(self) -> str:
        sign = {0: "+", 1: "+i", 2: "-", 3: "-i"}[self.sign_phase]
        body = "".join(f"{c}{i}" for i, c in enumerate(self.letters()) if c != "_")
        return sign + body

    def __mul__(self, other: "SignedPauli") -> "SignedPauli":
        """The operator product, with the phase ``i`` for each X·Z and ``−i`` for Z·X."""
        if self.n != other.n:
            raise ValueError("Operators on different numbers of qubits.")
        # X^a Z^b · X^c Z^d = (-1)^(b·c) X^(a+c) Z^(b+d); with Y = iXZ bookkept through phases.
        phase = self.phase + other.phase
        phase += 2 * bin(self.z & other.x).count("1")
        # Y = i X Z: each operator's own Y count enters its stored phase already (we keep
        # x, z, phase so that the operator is i^phase · Π (X^x Z^z) with Y written as XZ and
        # a +1 in phase per Y).
        return SignedPauli(self.n, self.x ^ other.x, self.z ^ other.z, phase % 4)

    def with_flipped_sign(self) -> "SignedPauli":
        return SignedPauli(self.n, self.x, self.z, (self.phase + 2) % 4)

    def commutes_with(self, other: "SignedPauli") -> bool:
        return (bin(self.x & other.z).count("1") + bin(self.z & other.x).count("1")) % 2 == 0

    @property
    def weight(self) -> int:
        return bin(self.x | self.z).count("1")

    def hermitian(self) -> "SignedPauli":
        """The operator as (sign, letters): the same operator when it is Hermitian; an error
        when its phase is ±i."""
        if self.sign_phase not in (0, 2):
            raise ValueError("Not a Hermitian Pauli operator.")
        return self

    def __eq__(self, other: object) -> bool:
        return isinstance(other, SignedPauli) and (self.n, self.x, self.z, self.phase % 4) == (other.n, other.x, other.z, other.phase % 4)

    def __hash__(self) -> int:
        return hash((self.n, self.x, self.z, self.phase % 4))


def _to_pauli_string(op: SignedPauli):
    """This package's ``PauliString`` (sign ±1) of a Hermitian signed Pauli."""
    from .._stim import PauliString

    return PauliString(("-" if op.hermitian().sign_phase == 2 else "+") + op.letters())


# ---------------------------------------------------------------------------------------------
# Logical states
# ---------------------------------------------------------------------------------------------


_SPARSE_PAIR = re.compile(r"([XYZ])(\d+)", re.IGNORECASE)


def _loom_sparse(text: str, n: Optional[int] = None) -> SignedPauli:
    """A sparse Pauli string with Loom's checks: a sign, then letter-index pairs (``+Z0X1``),
    each index once, within ``n`` qubits when ``n`` is given."""
    if text[:1] not in ("+", "-"):
        raise ValueError("The first character of the a Pauli string should be '+' or '-'.")
    rest = text[1:]
    invalid = _SPARSE_PAIR.sub("", rest)
    if invalid:
        raise ValueError(f"Invalid elements in the Pauli string: {invalid}.")
    indices = [int(i) for _, i in _SPARSE_PAIR.findall(rest)]
    if len(indices) != len(set(indices)):
        raise ValueError(f"Qubit indices { {i for i in indices if indices.count(i) > 1} } appear more than once in the Pauli string.")
    if n is not None and indices and n < max(indices) + 1:
        raise ValueError(f"Qubit index {max(indices)} is out of range for {n} qubits.")
    return SignedPauli.from_sparse_string(text, n)


class LogicalState:
    """A logical state by the logical Paulis that stabilize it (signed sparse strings on the
    logical qubits, ``("+Z0", "+X1")``), one per logical qubit, independent."""

    def __init__(self, sparse_logical_paulistrings) -> None:
        if isinstance(sparse_logical_paulistrings, str):
            sparse_logical_paulistrings = (sparse_logical_paulistrings,)
        strings = tuple(s.upper() for s in sparse_logical_paulistrings)
        ops = [_loom_sparse(s) for s in strings]
        width = max(op.n for op in ops)
        if width != len(strings):
            raise ValueError(f"The number of sparse stabilizers ({len(strings)}) does not match the maximum logical qubits required by the operators ({width}).")
        self.sparse_logical_paulistrings = strings
        self._ops = [SignedPauli.from_sparse_string(s, len(strings)) for s in strings]
        rows = [op.x | (op.z << op.n) for op in self._ops]
        if gf2_rank(rows) != len(rows):
            raise ValueError("The set of logical paulistrings is not irreducible.")

    @property
    def n_logical_qubits(self) -> int:
        return len(self.sparse_logical_paulistrings)

    @property
    def operators(self) -> list[SignedPauli]:
        return list(self._ops)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, LogicalState) and self.sparse_logical_paulistrings == other.sparse_logical_paulistrings

    def __hash__(self) -> int:
        return hash(self.sparse_logical_paulistrings)

    def __repr__(self) -> str:
        return f"LogicalState({self.sparse_logical_paulistrings})"

    @staticmethod
    def logical_to_base(block: Block, logical: SignedPauli, qubits: Sequence[Coord]) -> SignedPauli:
        """A logical Pauli as an operator on the block's data qubits: the product of the
        block's logical X and Z operators it names (Y_i = i X_i Z_i)."""
        n = len(qubits)
        out = SignedPauli.identity(n)
        for i in range(logical.n):
            has_x = (logical.x >> i) & 1
            has_z = (logical.z >> i) & 1
            if has_x:
                out = out * _block_operator(block.logical_x_operators[i], qubits)
            if has_z:
                out = out * _block_operator(block.logical_z_operators[i], qubits)
            if has_x and has_z:
                out = SignedPauli(n, out.x, out.z, (out.phase + 1) % 4)
        return SignedPauli(n, out.x, out.z, (out.phase + logical.sign_phase) % 4)


def _block_operator(op: PauliOperator, qubits: Sequence[Coord]) -> SignedPauli:
    """A block operator (letters on coordinates) as a Hermitian signed Pauli over an ordered
    qubit list."""
    n = len(qubits)
    index = {q: i for i, q in enumerate(qubits)}
    letters = ["_"] * n
    for p, q in zip(op.pauli, op.data_qubits):
        letters[index[q]] = p
    return SignedPauli.from_letters("".join(letters), n)


# ---------------------------------------------------------------------------------------------
# Running a circuit on the tableau simulator
# ---------------------------------------------------------------------------------------------

_GATES = {"i": None, "h": "h", "hadamard": "h", "x": "x", "y": "y", "z": "z", "phase": "s", "phaseinv": "s_dag", "cx": "cx", "cnot": "cx", "cy": "cy", "cz": "cz", "swap": "swap"}
_RESETS = {"reset": ("reset_z", None), "reset_0": ("reset_z", None), "reset_1": ("reset_z", "x"), "reset_+": ("reset_x", None), "reset_-": ("reset_x", "z"), "reset_+i": ("reset_y", None), "reset_-i": ("reset_y", "z")}
_MEASURES = {"measurement": "Z", "measure_z": "Z", "measure_x": "X", "measure_y": "Y"}


class _Run:
    """One run of a circuit from a stabilizer state: the final simulator and every measured
    classical channel's value and whether it was random."""

    def __init__(self, n_qubits: int) -> None:
        from .._stim import TableauSimulator

        self.sim = TableauSimulator()
        self.sim.set_num_qubits(n_qubits)
        self.n = n_qubits
        self.measurements: dict[str, tuple[int, bool]] = {}

    def set_state(self, stabilizers: Sequence[SignedPauli]) -> None:
        self.sim.set_state_from_stabilizers([_to_pauli_string(s) for s in stabilizers], allow_redundant=False, allow_underconstrained=True)

    def run(self, circuit: Circuit, index_of: dict[str, int]) -> None:
        from .._stim import PauliString

        for layer in Circuit.unroll(circuit):
            for gate in layer:
                name = gate.name
                qubits = [index_of[c.id] for c in gate.channels if c.is_quantum()]
                classical = [c for c in gate.channels if c.is_classical()]
                if name in _GATES:
                    if _GATES[name] is not None:
                        getattr(self.sim, _GATES[name])(*qubits)
                elif name in _RESETS:
                    reset, after = _RESETS[name]
                    getattr(self.sim, reset)(*qubits)
                    if after:
                        getattr(self.sim, after)(*qubits)
                elif name in _MEASURES:
                    letters = ["_"] * self.n
                    letters[qubits[0]] = _MEASURES[name]
                    observable = PauliString("".join(letters))
                    random = self.sim.peek_observable_expectation(observable) == 0
                    value = int(self.sim.measure_observable(observable))
                    self.measurements[classical[0].label] = (value, random)
                else:
                    raise ValueError(f"The validator does not know the operation {name!r}.")

    def expectation(self, op: SignedPauli) -> int:
        """+1 when the operator stabilizes the state, −1 when its negative does, 0 otherwise."""
        return int(self.sim.peek_observable_expectation(_to_pauli_string(op)))


def _layout(circuit: Circuit, input_block: Block, output_block: Block) -> tuple[int, dict[str, int], list[Coord], dict[Coord, int]]:
    """Loom's qubit numbering: the input block's data qubits first (its order), then the other
    quantum channels in the circuit's channel order."""
    quantum = [c for c in circuit.channels if c.is_quantum()]
    input_data = list(input_block.data_qubits)
    labels = [str(q) for q in input_data]
    index_of: dict[str, int] = {}
    position: dict[str, int] = {}
    for chan in quantum:
        if chan.label in labels:
            index_of[chan.id] = labels.index(chan.label)
            position[chan.label] = labels.index(chan.label)
    nxt = len(input_data)
    for chan in quantum:
        if chan.label not in labels:
            index_of[chan.id] = nxt
            position[chan.label] = nxt
            nxt += 1
    for q, label in zip(input_data, labels):
        position.setdefault(label, labels.index(label))
    n = max(nxt, len(input_data))
    output_index = {q: position[str(q)] for q in output_block.data_qubits if str(q) in position}
    return n, index_of, input_data, output_index


def _generators(block: Block) -> tuple[list[int], list[list[int]]]:
    """Loom's reduced bookkeeping: the indices of the stabilizers that are independent
    generators (the first of each), and for every stabilizer the generators (by position in
    that list) whose product it is."""
    qubits = list(block.data_qubits)
    n = len(qubits)
    basis: dict[int, tuple[int, int]] = {}
    generators: list[int] = []
    combos: list[list[int]] = []
    for i, stab in enumerate(block.stabilizers):
        x, z = stab.as_xz(qubits)
        v = x | (z << n)
        mask = 0
        while v:
            h = v.bit_length() - 1
            if h in basis:
                pv, pm = basis[h]
                v ^= pv
                mask ^= pm
            else:
                generators.append(i)
                mask = 1 << (len(generators) - 1)
                basis[h] = (v, mask)
                v = 0
        combos.append([k for k in range(len(generators)) if (mask >> k) & 1])
    return generators, combos


def _initial_stabilizers(input_block: Block, n: int, logical_ops: Sequence[SignedPauli], flipped: Sequence[int] = ()) -> list[SignedPauli]:
    """The input code's independent stabilizers (the generators at ``flipped`` positions
    negated), the logical state's operators, and Z on every auxiliary qubit (|0⟩)."""
    qubits = list(input_block.data_qubits)
    generators, _ = _generators(input_block)
    out: list[SignedPauli] = []
    for k, i in enumerate(generators):
        op = _block_operator(input_block.stabilizers[i], qubits)
        op = SignedPauli(n, op.x, op.z, op.phase)
        out.append(op.with_flipped_sign() if k in flipped else op)
    for op in logical_ops:
        out.append(SignedPauli(n, op.x, op.z, op.phase))
    for q in range(len(qubits), n):
        out.append(SignedPauli(n, 0, 1 << q, 0))
    return out


def _output_operator(op: PauliOperator, output_index: dict[Coord, int], n: int) -> SignedPauli:
    letters = ["_"] * n
    for p, q in zip(op.pauli, op.data_qubits):
        letters[output_index[q]] = p
    return SignedPauli.from_letters("".join(letters), n)


# ---------------------------------------------------------------------------------------------
# The checks
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class CodeStabilizerCheckOutput:
    missing_stabilizers: tuple[Stabilizer, ...]
    stabilizers_with_incorrect_parity: tuple[Stabilizer, ...]

    def __len__(self) -> int:
        return len(self.missing_stabilizers) + len(self.stabilizers_with_incorrect_parity)

    def __str__(self) -> str:
        out = ""
        if self.missing_stabilizers:
            out += "- Missing Stabilizers:\n" + "\n".join(str(s) for s in self.missing_stabilizers) + "\n"
        if self.stabilizers_with_incorrect_parity:
            out += "- Stabilizers with Incorrect Parity:\n" + "\n".join(str(s) for s in self.stabilizers_with_incorrect_parity) + "\n"
        return out.rstrip("\n")


@dataclass(frozen=True)
class LogicalOperatorCheckOutput:
    input_vs_expected_vs_actual_logicals_multi: tuple[tuple[LogicalState, tuple[LogicalState, ...], Optional[LogicalState]], ...]
    input_vs_expected_vs_actual_logicals_with_parity: tuple[tuple[LogicalState, tuple[LogicalState, tuple[int, ...]], Optional[LogicalState]], ...]

    def __len__(self) -> int:
        return len(self.input_vs_expected_vs_actual_logicals_multi) + len(self.input_vs_expected_vs_actual_logicals_with_parity)

    def __str__(self) -> str:
        out = ""
        if self.input_vs_expected_vs_actual_logicals_multi:
            out += "- Failed Logical State Transformations:\n"
            for initial, allowed, actual in self.input_vs_expected_vs_actual_logicals_multi:
                out += f"Initial: {initial}, Allowed Finals: {allowed}, Actual: {actual}\n"
        if self.input_vs_expected_vs_actual_logicals_with_parity:
            out += "- Expected vs Actual Logical Transformations:\n"
            for initial, (expected, flips), actual in self.input_vs_expected_vs_actual_logicals_with_parity:
                out += f"Initial: {initial}, Expected: {expected}, Flips: {flips}, Actual: {actual}\n"
        return out.rstrip("\n")


@dataclass(frozen=True)
class StabilizerMeasurementCheckOutput:
    expected_vs_measured_stabs: tuple[tuple[str, Stabilizer, tuple[Stabilizer, ...]], ...]
    probabilistic_measurements: tuple[str, ...]

    def __len__(self) -> int:
        return len(self.expected_vs_measured_stabs) + len(self.probabilistic_measurements)

    def __str__(self) -> str:
        out = ""
        if any(self.expected_vs_measured_stabs):
            out = "- Expected vs Measured Stabilizers:\n"
            for idx, stab, measured in self.expected_vs_measured_stabs:
                out += f"Measurement {idx}: Expected {stab}, Measured: {', '.join(str(s) for s in measured) if measured else 'None'}\n"
        if any(self.probabilistic_measurements):
            out += "- Probabilistic Measurements: " + ", ".join(str(idx) for idx in self.probabilistic_measurements) + "\n"
        return out.rstrip("\n")


class _ValidityCheck:
    """Loom's validity check: its ``output`` (what failed), ``valid`` when the output is empty,
    a ``message``, and the text ``Name: valid = ...`` with the message and output when not
    valid."""

    output: Any

    def __init__(self, output) -> None:
        object.__setattr__(self, "output", output)

    def __setattr__(self, name: str, value) -> None:
        raise AttributeError(f"cannot assign to field '{name}'")

    def __eq__(self, other: object) -> bool:
        return type(other) is type(self) and other.output == self.output

    def __hash__(self) -> int:
        return hash((type(self).__name__, self.output))

    def __repr__(self) -> str:
        return f"{type(self).__name__}(output={self.output!r})"

    @property
    def valid(self) -> bool:
        return len(self.output) == 0

    @property
    def message(self) -> str:
        raise NotImplementedError

    def __str__(self) -> str:
        out = f"{self.__class__.__name__}: valid = {self.valid}\n"
        if not self.valid:
            out += f"message: {self.message}\n"
            out += "output: \n"
            out += str(self.output) + "\n"
        return out.rstrip("\n")


class CodeStabilizerCheck(_ValidityCheck):
    """Are the output code's stabilizers on the final state, with the right parity? The output
    is a ``CodeStabilizerCheckOutput`` (2.0's two tuples are accepted too)."""

    def __init__(self, output, stabilizers_with_incorrect_parity: Optional[Sequence[Stabilizer]] = None) -> None:
        if not isinstance(output, CodeStabilizerCheckOutput):
            output = CodeStabilizerCheckOutput(tuple(output), tuple(stabilizers_with_incorrect_parity or ()))
        super().__init__(output)

    @property
    def message(self) -> str:
        return "" if len(self.output) == 0 else "Some code stabilizer(s) were not found in the output."

    @property
    def missing_stabilizers(self) -> tuple[Stabilizer, ...]:
        return self.output.missing_stabilizers

    @property
    def stabilizers_with_incorrect_parity(self) -> tuple[Stabilizer, ...]:
        return self.output.stabilizers_with_incorrect_parity


class LogicalOperatorCheck(_ValidityCheck):
    """Do the logical states transform as declared? The output is a
    ``LogicalOperatorCheckOutput``."""

    def __init__(self, output, input_vs_expected_vs_actual_logicals_with_parity: Optional[Sequence] = None) -> None:
        if not isinstance(output, LogicalOperatorCheckOutput):
            output = LogicalOperatorCheckOutput(tuple(output), tuple(input_vs_expected_vs_actual_logicals_with_parity or ()))
        super().__init__(output)

    @property
    def message(self) -> str:
        return "Some logical states were not transformed correctly." if not self.valid else ""

    @property
    def failed_transformations(self) -> tuple:
        return self.output.input_vs_expected_vs_actual_logicals_multi

    @property
    def failed_transformations_with_parity(self) -> tuple:
        return self.output.input_vs_expected_vs_actual_logicals_with_parity


class StabilizerMeasurementCheck(_ValidityCheck):
    """Do the named measurements measure the named input stabilizers, deterministically? The
    output is a ``StabilizerMeasurementCheckOutput``."""

    def __init__(self, output, probabilistic_measurements: Optional[Sequence[str]] = None) -> None:
        if not isinstance(output, StabilizerMeasurementCheckOutput):
            output = StabilizerMeasurementCheckOutput(tuple(output), tuple(probabilistic_measurements or ()))
        super().__init__(output)

    @property
    def message(self) -> str:
        wrong, random = len(self.output.expected_vs_measured_stabs), len(self.output.probabilistic_measurements)
        if (wrong, random) == (0, 0):
            return ""
        if wrong == 0:
            return "Some measurement(s) were not deterministic."
        if random == 0:
            return "Some measurement(s) did not measure the assigned stabilizer."
        return "Some measurement(s) were not deterministic and some did not measure the assigned stabilizer."

    @property
    def expected_vs_measured_stabs(self) -> tuple:
        return self.output.expected_vs_measured_stabs

    @property
    def probabilistic_measurements(self) -> tuple[str, ...]:
        return self.output.probabilistic_measurements


@dataclass(frozen=True)
class AllChecks:
    """The three checks, iterable in this order."""

    code_stabilizers: CodeStabilizerCheck
    logical_operators: LogicalOperatorCheck
    stabilizers_measured: StabilizerMeasurementCheck

    def __iter__(self):
        return iter((self.code_stabilizers, self.logical_operators, self.stabilizers_measured))

    @property
    def stabilizer_measurements(self) -> StabilizerMeasurementCheck:
        # 2.0's name for stabilizers_measured.
        return self.stabilizers_measured

    @property
    def valid(self) -> bool:
        return all(check.valid for check in self)


@dataclass(frozen=True)
class DebugData:
    """The validator's verdict: ``valid`` (all checks valid) and the ``checks``; its text names
    each failed check, after a separator."""

    checks: AllChecks

    @property
    def valid(self) -> bool:
        return all(check.valid for check in self.checks)

    def __bool__(self) -> bool:
        return self.valid

    def __str__(self) -> str:
        out = f"DebugData: valid = {self.valid}\n"
        for check in self.checks:
            if not check.valid:
                out += "-" * 40 + "\n" + str(check) + "\n"
        return out.rstrip("\n")


def _parity(run: _Run, cbits: Sequence[Union[str, int]]) -> int:
    parity = 0
    for cbit in cbits:
        if isinstance(cbit, int):
            parity ^= cbit
        else:
            parity ^= run.measurements[cbit][0]
    return parity


def check_code_stabilizers_output(circuit: Circuit, input_block: Block, output_block: Block, output_stabilizers_parity: dict, output_stabilizers_with_any_value: Sequence[Stabilizer]) -> CodeStabilizerCheck:
    n, index_of, _, output_index = _layout(circuit, input_block, output_block)
    zeros = LogicalState([f"+Z{i}" for i in range(input_block.n_logical_qubits)])
    run = _Run(n)
    run.set_state(_initial_stabilizers(input_block, n, [LogicalState.logical_to_base(input_block, op, list(input_block.data_qubits)) for op in zeros.operators]))
    run.run(circuit, index_of)
    parity = {stab: 0 for stab in output_block.stabilizers}
    parity.update({stab: _parity(run, cbits) for stab, cbits in output_stabilizers_parity.items()})
    missing, wrong = [], []
    for stab in output_block.stabilizers:
        if stab in output_stabilizers_with_any_value:
            continue
        op = _output_operator(stab, output_index, n)
        if parity[stab] == 1:
            op = op.with_flipped_sign()
        e = run.expectation(op)
        if e == -1:
            wrong.append(stab)
        elif e == 0:
            missing.append(stab)
    for stab in output_stabilizers_with_any_value:
        if run.expectation(_output_operator(stab, output_index, n)) == 0:
            missing.append(stab)
    return CodeStabilizerCheck(CodeStabilizerCheckOutput(tuple(missing), tuple(wrong)))


def all_possible_pauli_strings(n_qubits: int) -> list[str]:
    if n_qubits < 1:
        raise ValueError("The number of qubits must be at least 1.")
    return [f"{sign}{''.join(p)}" for sign, p in product(["+", "-"], product(["_", "X", "Z", "Y"], repeat=n_qubits)) if any(c != "_" for c in p)]


def _find_logical_state(block: Block, run: _Run, output_index: dict[Coord, int], n: int) -> Optional[LogicalState]:
    k = block.n_logical_qubits
    found: list[SignedPauli] = []
    rows: list[int] = []
    for text in all_possible_pauli_strings(k):
        op = SignedPauli.from_letters(text[1:], k, text[0])
        base = LogicalState.logical_to_base(block, op, list(block.data_qubits))
        base = SignedPauli(n, _remap(base.x, block, output_index, n), _remap(base.z, block, output_index, n), base.phase)
        if run.expectation(base) == 1:
            row = op.x | (op.z << k)
            if gf2_rank(rows + [row]) > len(rows):
                rows.append(row)
                found.append(op)
                if len(found) == k:
                    return LogicalState(tuple(o.hermitian().as_sparse_string() for o in found))
    return None


def _remap(mask: int, block: Block, output_index: dict[Coord, int], n: int) -> int:
    out = 0
    for i, q in enumerate(block.data_qubits):
        if (mask >> i) & 1:
            out |= 1 << output_index[q]
    return out


def check_logical_operators_transformation(circuit: Circuit, input_block: Block, output_block: Block, transformations_with_parity: dict, transformations: Sequence[tuple[LogicalState, Sequence[LogicalState]]]) -> LogicalOperatorCheck:
    n, index_of, _, output_index = _layout(circuit, input_block, output_block)
    in_qubits = list(input_block.data_qubits)

    def start(state: LogicalState) -> _Run:
        run = _Run(n)
        run.set_state(_initial_stabilizers(input_block, n, [LogicalState.logical_to_base(input_block, op, in_qubits) for op in state.operators]))
        run.run(circuit, index_of)
        return run

    def expected(state: LogicalState, flips: Sequence[int] = ()) -> list[SignedPauli]:
        ops = []
        for i, op in enumerate(state.operators):
            base = LogicalState.logical_to_base(output_block, op, list(output_block.data_qubits))
            base = SignedPauli(n, _remap(base.x, output_block, output_index, n), _remap(base.z, output_block, output_index, n), base.phase)
            if i < len(flips) and flips[i]:
                base = base.with_flipped_sign()
            ops.append(base)
        return ops

    failed = []
    for state_in, states_out in transformations:
        run = start(state_in)
        if not any(all(run.expectation(op) == 1 for op in expected(s)) for s in states_out):
            failed.append((state_in, tuple(states_out), _find_logical_state(output_block, run, output_index, n)))
    failed_parity = []
    for state_in, (state_out, parity_dict) in transformations_with_parity.items():
        run = start(state_in)
        flips = [_parity(run, parity_dict.get(i, ())) for i in range(output_block.n_logical_qubits)]
        if not all(run.expectation(op) == 1 for op in expected(state_out, flips)):
            failed_parity.append((state_in, (state_out, tuple(flips)), _find_logical_state(output_block, run, output_index, n)))
    return LogicalOperatorCheck(LogicalOperatorCheckOutput(tuple(failed), tuple(failed_parity)))


def check_input_stabilizer_measurement(circuit: Circuit, input_block: Block, output_block: Block, measurement_to_stabilizer_map: dict[str, Stabilizer]) -> StabilizerMeasurementCheck:
    n, index_of, in_qubits, _ = _layout(circuit, input_block, output_block)
    zeros = LogicalState([f"+Z{i}" for i in range(input_block.n_logical_qubits)])
    logical_ops = [LogicalState.logical_to_base(input_block, op, in_qubits) for op in zeros.operators]

    def results(flipped: Sequence[int]) -> dict[str, tuple[int, bool]]:
        run = _Run(n)
        run.set_state(_initial_stabilizers(input_block, n, logical_ops, flipped))
        run.run(circuit, index_of)
        return run.measurements

    reference = results(())
    for label in measurement_to_stabilizer_map:
        if label not in reference:
            raise ValueError(f"Measurement channel label {label} not found in the circuit's measurements.")
    probabilistic: set[str] = set()
    ref_values: dict[str, int] = {}
    for label, (value, random) in reference.items():
        if label in measurement_to_stabilizer_map:
            if random:
                probabilistic.add(label)
            else:
                ref_values[label] = value
    contributions: dict[str, set[Stabilizer]] = {label: set() for label in ref_values}
    # Flipping stabilizer i means negating the independent generators whose product it is.
    _, combos = _generators(input_block)
    for i, stab in enumerate(input_block.stabilizers):
        if not combos[i]:
            continue
        measured = results(combos[i])
        for label, (value, random) in measured.items():
            if label not in measurement_to_stabilizer_map:
                continue
            if (label in probabilistic) != random:
                raise RuntimeError(f"Measurement with index {label} is sometimes random and sometimes deterministic.")
            if random:
                continue
            if value != ref_values[label]:
                contributions[label].add(stab)
    qubits = in_qubits
    mismatched = []
    for label, stab in measurement_to_stabilizer_map.items():
        if label in probabilistic:
            continue
        prod = reduce(lambda a, b: a * b, (_block_operator(s, qubits) for s in contributions[label]), SignedPauli.identity(len(qubits)))
        if prod.hermitian() != _block_operator(stab, qubits).hermitian():
            mismatched.append((label, stab, tuple(contributions[label])))
    return StabilizerMeasurementCheck(StabilizerMeasurementCheckOutput(tuple(sorted(mismatched, key=lambda t: t[0])), tuple(sorted(probabilistic))))


def is_circuit_valid(circuit: Circuit, input_block: Union[Block, Sequence[Block]], output_block: Union[Block, Sequence[Block]], output_stabilizers_parity: dict, output_stabilizers_with_any_value: Sequence[Stabilizer], logical_state_transformations_with_parity: dict, logical_state_transformations: Sequence[tuple[LogicalState, Sequence[LogicalState]]], measurement_to_input_stabilizer_map: dict[str, Stabilizer], seed: Optional[int] = None) -> DebugData:
    """Loom's validity check of a QEC circuit: the input code becomes the output code (every
    output stabilizer stabilizes the final state with the parity its classical bits predict,
    or with any sign for those in ``output_stabilizers_with_any_value``), the logical states
    transform as declared (to one of the allowed states, or to the expected state with the
    parities the bits predict), and the named measurements measure the named input
    stabilizers deterministically. The circuit's quantum channels beyond the input block's
    data qubits start in |0⟩."""
    if not isinstance(input_block, Block):
        input_block = Block.from_blocks(tuple(input_block))
    if not isinstance(output_block, Block):
        output_block = Block.from_blocks(tuple(output_block))
    classical = [c.label for c in circuit.channels if c.is_classical()]
    for label, stab in measurement_to_input_stabilizer_map.items():
        if not isinstance(stab, Stabilizer):
            raise TypeError(f"The values of measurement_to_input_stabilizer_map should be of type Stabilizer. Value {stab} is of type {type(stab)}.")
        if stab not in input_block.stabilizers:
            raise ValueError(f"Stabilizer {stab} was not found in the input block.)")
        if label not in classical:
            raise ValueError(f"Classical channel label {label} was not found in the circuit's classical channels.")
    if any(s not in output_block.stabilizers for s in output_stabilizers_with_any_value):
        raise ValueError("The stabilizers in output_stabilizers_with_any_value should be in the stabilizers of the output block(s).")
    for stab, cbits in output_stabilizers_parity.items():
        if stab not in output_block.stabilizers:
            raise ValueError(f"Every stabilizer in output_stabilizers_parity should be in the stabilizers of the output block. {stab} was not found in the output block.")
        for cbit in cbits:
            if isinstance(cbit, str) and cbit not in classical:
                raise ValueError(f"Channel {cbit} which is referenced in the output_stabilizers_parity is not a classical channel in the circuit.")
            if isinstance(cbit, int) and cbit not in (0, 1):
                raise ValueError(f"Only 0 and 1 are allowed as parity changes. {cbit} was found.")
    for state_in, (state_out, flips) in logical_state_transformations_with_parity.items():
        if state_in.n_logical_qubits != input_block.n_logical_qubits or state_out.n_logical_qubits != output_block.n_logical_qubits:
            raise ValueError("A logical state has the wrong number of logical qubits for its code.")
        for idx, cbits in flips.items():
            if idx < 0 or idx >= output_block.n_logical_qubits:
                raise ValueError(f"Logical operator index {idx} is out of range.")
            for cbit in cbits:
                if isinstance(cbit, str) and cbit not in classical:
                    raise ValueError(f"Channel {cbit} which is referenced in the logical_state_transformations_with_parity is not a classical channel in the output block.")
    for state_in, states_out in logical_state_transformations:
        if state_in.n_logical_qubits != input_block.n_logical_qubits or any(s.n_logical_qubits != output_block.n_logical_qubits for s in states_out):
            raise ValueError("A logical state has the wrong number of logical qubits for its code.")
    code = check_code_stabilizers_output(circuit, input_block, output_block, output_stabilizers_parity, output_stabilizers_with_any_value)
    logical = check_logical_operators_transformation(circuit, input_block, output_block, logical_state_transformations_with_parity, logical_state_transformations)
    measured = check_input_stabilizer_measurement(circuit, input_block, output_block, measurement_to_input_stabilizer_map)
    return DebugData(AllChecks(code, logical, measured))


def logical_states_to_check(n_logical_qubits: int) -> list[LogicalState]:
    """Loom's set of input states whose correct transformation pins a Clifford's action."""
    if n_logical_qubits == 1:
        return [LogicalState(["+Z0"]), LogicalState(["+X0"])]
    if n_logical_qubits == 2:
        return [LogicalState(["+Z0", "+Z1"]), LogicalState(["+X0", "+X1"]), LogicalState(["+Z0", "+X1"]), LogicalState(["+X0", "+Z1"])]
    k = n_logical_qubits
    return [LogicalState([f"+Z{j}" for j in range(k)]), LogicalState([f"+X{j}" for j in range(k)])] + [LogicalState([f"+Z{j}" if i != j else f"+X{j}" for j in range(k)]) for i in range(k)] + [LogicalState([f"+X{j}" if i != j else f"+Z{j}" for j in range(k)]) for i in range(k)]


def logical_state_transformations_to_check(x_operators_sparse_pauli_map: Sequence[str], z_operators_sparse_pauli_map: Sequence[str]) -> list[tuple[LogicalState, tuple[LogicalState]]]:
    """From how each logical X_i and Z_i transforms (unsigned sparse strings, ``["X0X1", "X1"]``
    and ``["Z0", "Z0Z1"]`` for a CNOT), the state transformations to check."""
    if len(x_operators_sparse_pauli_map) != len(z_operators_sparse_pauli_map):
        raise ValueError("The number of X and Z operators should be the same for the logical state transformations.")
    k = len(x_operators_sparse_pauli_map)
    x_map = [_loom_sparse("+" + x, k) for x in x_operators_sparse_pauli_map]
    z_map = [_loom_sparse("+" + z, k) for z in z_operators_sparse_pauli_map]
    # Loom's tableau check: the images generate the Pauli group, the Xs commute, the Zs commute,
    # and Z_i anticommutes with X_j exactly when i == j (checked for j >= i, as Loom does).
    valid = gf2_rank([op.x | (op.z << k) for op in x_map + z_map]) == 2 * k
    valid = valid and all(x_map[i].commutes_with(x_map[j]) and z_map[i].commutes_with(z_map[j]) for i in range(k) for j in range(i + 1))
    valid = valid and all((not z_map[i].commutes_with(x_map[j])) == (i == j) for i in range(k) for j in range(i, k))
    if not valid:
        raise ValueError("The transformed X and Z operators do not generate a valid tableau when stacked. Check the input sparse Pauli string maps.")
    out = []
    for state in logical_states_to_check(k):
        images = []
        for op in state.operators:
            image = SignedPauli.identity(k)
            for i in range(k):
                if (op.x >> i) & 1:
                    image = image * x_map[i]
                if (op.z >> i) & 1:
                    image = image * z_map[i]
                if (op.x >> i) & 1 and (op.z >> i) & 1:
                    image = SignedPauli(k, image.x, image.z, (image.phase + 1) % 4)
            images.append(SignedPauli(k, image.x, image.z, (image.phase + op.sign_phase) % 4).as_sparse_string())
        out.append((state, (LogicalState(tuple(images)),)))
    return out


def is_syndrome_extraction_circuit_valid(circuit: Circuit, input_block: Union[Block, Sequence[Block]], measurement_to_input_stabilizer_map: dict[str, Stabilizer]) -> DebugData:
    """The circuit keeps the code and the logical state and measures the named stabilizers."""
    block = input_block if isinstance(input_block, Block) else Block.from_blocks(tuple(input_block))
    transformations = [(state, (state,)) for state in logical_states_to_check(block.n_logical_qubits)]
    return is_circuit_valid(circuit, block, block, {}, [], {}, transformations, measurement_to_input_stabilizer_map)


def is_logical_operation_circuit_valid(circuit: Circuit, input_block: Union[Block, Sequence[Block]], x_operators_sparse_pauli_map: Sequence[str], z_operators_sparse_pauli_map: Sequence[str]) -> DebugData:
    """The circuit keeps the code and acts on the logical qubits as the maps say (signs not
    checked, as in Loom: a logical X, which sends Z0 to −Z0, fails the map ``["X0"], ["Z0"]``)."""
    block = input_block if isinstance(input_block, Block) else Block.from_blocks(tuple(input_block))
    transformations = logical_state_transformations_to_check(x_operators_sparse_pauli_map, z_operators_sparse_pauli_map)
    return is_circuit_valid(circuit, block, block, {}, [], {}, transformations, {})

# Ported from tqec (tqec/tqec 0.2.0, src/tqec/utils/noise_model.py), modified for stabilizer_qec: this
# package's Circuit in place of stim's. tqec's file is in turn recovered from https://zenodo.org/records/7487893,
# Gidney, C. (2022), Data for "Inplace Access to the Surface Code Y Basis",
# https://doi.org/10.5281/zenodo.7487893, under CC BY 4.0
# (https://creativecommons.org/licenses/by/4.0/legalcode), with tqec's modifications: formatting, typing, no
# depolarizing error after measurements in ``uniform_depolarizing`` and ``si1000``, no
# ``depolarizing_two_body_measurement_noise``, TICKs around REPEAT blocks as tqec places them, docstrings.
"""Noise models applied to circuits instruction by instruction, as tqec applies them: tqec's
``NoiseModel`` with its two constructors, ``uniform_depolarizing`` and ``si1000``.

The noisy circuit is tqec's for the same noiseless circuit, instruction for instruction (the
tests hold the two to that). A circuit compiled from a block graph here
(``compile_block_graph(graph).generate_stim_circuit(k, noise_model=...)``) is the surgery
engine's noiseless circuit with the model laid over it, as tqec lays it over its own."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import AbstractSet, Iterator, Optional, Union

from .._circuit import Circuit
from .._stim import CircuitInstruction, CircuitRepeatBlock

CLIFFORD_1Q = "C1"
CLIFFORD_2Q = "C2"
ANNOTATION = "info"
MPP = "MPP"
MEASURE_RESET_1Q = "MR1"
JUST_MEASURE_1Q = "M1"
JUST_RESET_1Q = "R1"
NOISE = "!?"

OP_TYPES = {
    "I": CLIFFORD_1Q,
    "X": CLIFFORD_1Q,
    "Y": CLIFFORD_1Q,
    "Z": CLIFFORD_1Q,
    "C_XYZ": CLIFFORD_1Q,
    "C_ZYX": CLIFFORD_1Q,
    "H": CLIFFORD_1Q,
    "H_XY": CLIFFORD_1Q,
    "H_XZ": CLIFFORD_1Q,
    "H_YZ": CLIFFORD_1Q,
    "S": CLIFFORD_1Q,
    "SQRT_X": CLIFFORD_1Q,
    "SQRT_X_DAG": CLIFFORD_1Q,
    "SQRT_Y": CLIFFORD_1Q,
    "SQRT_Y_DAG": CLIFFORD_1Q,
    "SQRT_Z": CLIFFORD_1Q,
    "SQRT_Z_DAG": CLIFFORD_1Q,
    "S_DAG": CLIFFORD_1Q,
    "CNOT": CLIFFORD_2Q,
    "CX": CLIFFORD_2Q,
    "CY": CLIFFORD_2Q,
    "CZ": CLIFFORD_2Q,
    "ISWAP": CLIFFORD_2Q,
    "ISWAP_DAG": CLIFFORD_2Q,
    "SQRT_XX": CLIFFORD_2Q,
    "SQRT_XX_DAG": CLIFFORD_2Q,
    "SQRT_YY": CLIFFORD_2Q,
    "SQRT_YY_DAG": CLIFFORD_2Q,
    "SQRT_ZZ": CLIFFORD_2Q,
    "SQRT_ZZ_DAG": CLIFFORD_2Q,
    "SWAP": CLIFFORD_2Q,
    "XCX": CLIFFORD_2Q,
    "XCY": CLIFFORD_2Q,
    "XCZ": CLIFFORD_2Q,
    "YCX": CLIFFORD_2Q,
    "YCY": CLIFFORD_2Q,
    "YCZ": CLIFFORD_2Q,
    "ZCX": CLIFFORD_2Q,
    "ZCY": CLIFFORD_2Q,
    "ZCZ": CLIFFORD_2Q,
    "MPP": MPP,
    "MR": MEASURE_RESET_1Q,
    "MRX": MEASURE_RESET_1Q,
    "MRY": MEASURE_RESET_1Q,
    "MRZ": MEASURE_RESET_1Q,
    "M": JUST_MEASURE_1Q,
    "MX": JUST_MEASURE_1Q,
    "MY": JUST_MEASURE_1Q,
    "MZ": JUST_MEASURE_1Q,
    "R": JUST_RESET_1Q,
    "RX": JUST_RESET_1Q,
    "RY": JUST_RESET_1Q,
    "RZ": JUST_RESET_1Q,
    "DETECTOR": ANNOTATION,
    "OBSERVABLE_INCLUDE": ANNOTATION,
    "QUBIT_COORDS": ANNOTATION,
    "SHIFT_COORDS": ANNOTATION,
    "TICK": ANNOTATION,
    "E": ANNOTATION,
    "DEPOLARIZE1": NOISE,
    "DEPOLARIZE2": NOISE,
    "PAULI_CHANNEL_1": NOISE,
    "PAULI_CHANNEL_2": NOISE,
    "X_ERROR": NOISE,
    "Y_ERROR": NOISE,
    "Z_ERROR": NOISE,
    # Not supported (as in tqec): CORRELATED_ERROR, ELSE_CORRELATED_ERROR.
}
OP_MEASURE_BASES = {
    "M": "Z",
    "MX": "X",
    "MY": "Y",
    "MZ": "Z",
    "MPP": "",
}
COLLAPSING_OPS = {op for op, t in OP_TYPES.items() if t in {JUST_RESET_1Q, JUST_MEASURE_1Q, MPP, MEASURE_RESET_1Q}}


def _op_type(name: str) -> str:
    try:
        return OP_TYPES[name]
    except KeyError:
        raise ValueError(f"the noise model has no rule for {name} instructions (tqec's NoiseModel knows {len(OP_TYPES)} instruction names)") from None


class NoiseRule:
    """How to add noise to an operation."""

    def __init__(self, *, after: dict[str, float], flip_result: float = 0):
        """Noise for an operation.

        Args:
            after: noise channel names to their probability argument, for example
                ``{"DEPOLARIZE2": 0.01, "X_ERROR": 0.02}``: these channels occur after all other
                operations in the moment, on the operation's targets.
            flip_result: the probability that a measurement result is reported wrong; only
                valid for operations that produce measurement results.
        """
        if not (0 <= flip_result <= 1):
            raise ValueError(f"not (0 <= {flip_result=} <= 1)")
        for k, p in after.items():
            if _op_type(k) != NOISE:
                raise ValueError(f"not a noise channel: {k} from {after=}")
            if not (0 <= p <= 1):
                raise ValueError(f"not (0 <= {p} <= 1) from {after=}")
        self.after = after
        self.flip_result = flip_result

    def _append_noisy_version_of(self, *, split_op: CircuitInstruction, out_during_moment: Circuit, after_moments: "defaultdict[tuple[str, float], Circuit]", immune_qubits: AbstractSet[int]) -> None:
        targets = split_op.targets_copy()
        if immune_qubits and any((t.is_qubit_target or t.is_x_target or t.is_y_target or t.is_z_target) and t.value in immune_qubits for t in targets):
            out_during_moment.append(split_op)
            return

        args = split_op.gate_args_copy()
        if self.flip_result:
            t = _op_type(split_op.name)
            assert t in {MPP, JUST_MEASURE_1Q, MEASURE_RESET_1Q}
            assert len(args) == 0
            args = [self.flip_result]

        out_during_moment.append(split_op.name, targets, args)
        raw_targets = [t.value for t in targets if not t.is_combiner]
        for op_name, arg in self.after.items():
            after_moments[(op_name, arg)].append(op_name, raw_targets, arg)


class NoiseModel:
    """A noise model applied to a circuit moment by moment (tqec's, after Gidney's): a rule per
    gate or measurement basis, a rule for any one- and any two-qubit Clifford, and depolarizing
    noise on the qubits each moment leaves idle."""

    def __init__(
        self,
        idle_depolarization: float,
        additional_depolarization_waiting_for_m_or_r: float = 0,
        gate_rules: Optional[dict[str, NoiseRule]] = None,
        measure_rules: Optional[dict[str, NoiseRule]] = None,
        any_clifford_1q_rule: Optional[NoiseRule] = None,
        any_clifford_2q_rule: Optional[NoiseRule] = None,
    ):
        self.idle_depolarization = idle_depolarization
        self.additional_depolarization_waiting_for_m_or_r = additional_depolarization_waiting_for_m_or_r
        self.gate_rules = gate_rules
        self.measure_rules = measure_rules
        self.any_clifford_1q_rule = any_clifford_1q_rule
        self.any_clifford_2q_rule = any_clifford_2q_rule

    @staticmethod
    def si1000(p: float) -> "NoiseModel":
        """Superconducting-inspired noise, as "A Fault-Tolerant Honeycomb Memory"
        (arXiv:2108.10457) defines it, with the measurement result flipped instead of the qubit
        before it. Only Z-basis measurements and resets have a rule, as in tqec: a circuit that
        prepares or reads a patch in X is refused."""
        return NoiseModel(
            idle_depolarization=p / 10,
            additional_depolarization_waiting_for_m_or_r=2 * p,
            any_clifford_1q_rule=NoiseRule(after={"DEPOLARIZE1": p / 10}),
            any_clifford_2q_rule=NoiseRule(after={"DEPOLARIZE2": p}),
            measure_rules={
                "Z": NoiseRule(after={}, flip_result=p * 5),
            },
            gate_rules={
                "R": NoiseRule(after={"X_ERROR": p * 2}),
            },
        )

    @staticmethod
    def uniform_depolarizing(p: float) -> "NoiseModel":
        """Near-standard circuit depolarizing noise, everything with strength ``p``: one-qubit
        depolarization after one-qubit Cliffords and on idle qubits, two-qubit depolarization
        after two-qubit Cliffords, resets followed by a bit (or phase) flip, and measurement
        results flipped."""
        return NoiseModel(
            idle_depolarization=p,
            any_clifford_1q_rule=NoiseRule(after={"DEPOLARIZE1": p}),
            any_clifford_2q_rule=NoiseRule(after={"DEPOLARIZE2": p}),
            measure_rules={
                "X": NoiseRule(after={}, flip_result=p),
                "Y": NoiseRule(after={}, flip_result=p),
                "Z": NoiseRule(after={}, flip_result=p),
                "XX": NoiseRule(after={}, flip_result=p),
                "YY": NoiseRule(after={}, flip_result=p),
                "ZZ": NoiseRule(after={}, flip_result=p),
            },
            gate_rules={
                "RX": NoiseRule(after={"Z_ERROR": p}),
                "RY": NoiseRule(after={"X_ERROR": p}),
                "R": NoiseRule(after={"X_ERROR": p}),
            },
        )

    def _noise_rule_for_split_operation(self, *, split_op: CircuitInstruction) -> Optional[NoiseRule]:
        if occurs_in_classical_control_system(split_op):
            return None
        if self.gate_rules is not None:
            rule = self.gate_rules.get(split_op.name)
            if rule is not None:
                return rule
        t = _op_type(split_op.name)
        if self.any_clifford_1q_rule is not None and t == CLIFFORD_1Q:
            return self.any_clifford_1q_rule
        if self.any_clifford_2q_rule is not None and t == CLIFFORD_2Q:
            return self.any_clifford_2q_rule
        if self.measure_rules is not None and t in (MPP, JUST_MEASURE_1Q):
            measure_basis = _measure_basis(split_op=split_op)
            assert measure_basis is not None
            rule = self.measure_rules.get(measure_basis)
            if rule is not None:
                return rule
        raise ValueError(f"No noise (or lack of noise) specified for split_op={split_op!r}. Please make sure the noise model covers all operations in the compiled circuit.")

    def _append_idle_error(self, *, moment_split_ops: list[CircuitInstruction], out: Circuit, system_qubits: AbstractSet[int], immune_qubits: AbstractSet[int]) -> None:
        collapse_qubits: list[int] = []
        clifford_qubits: list[int] = []
        for split_op in moment_split_ops:
            if occurs_in_classical_control_system(split_op):
                continue
            qubits_out = collapse_qubits if split_op.name in COLLAPSING_OPS else clifford_qubits
            for target in split_op.targets_copy():
                if not target.is_combiner:
                    qubits_out.append(target.value)

        # Safety check for operation collisions.
        usage_counts = Counter(collapse_qubits + clifford_qubits)
        qubits_used_multiple_times = {q for q, c in usage_counts.items() if c != 1}
        if qubits_used_multiple_times:
            moment = Circuit()
            for op in moment_split_ops:
                moment.append(op)
            raise ValueError(f"Qubits were operated on multiple times without a TICK in between:\nmultiple uses: {sorted(qubits_used_multiple_times)}\nmoment:\n{moment}")

        collapse_qubits_set = set(collapse_qubits)
        clifford_qubits_set = set(clifford_qubits)
        idle = sorted(system_qubits - collapse_qubits_set - clifford_qubits_set - immune_qubits)
        if idle and self.idle_depolarization:
            out.append("DEPOLARIZE1", idle, self.idle_depolarization)

        waiting_for_mr = sorted(system_qubits - collapse_qubits_set - immune_qubits)
        if collapse_qubits_set and waiting_for_mr and self.additional_depolarization_waiting_for_m_or_r:
            # As in tqec: the idle qubits, not every qubit waiting.
            out.append("DEPOLARIZE1", idle, self.additional_depolarization_waiting_for_m_or_r)

    def _append_noisy_moment(self, *, moment_split_ops: list[CircuitInstruction], out: Circuit, system_qubits: AbstractSet[int], immune_qubits: AbstractSet[int]) -> None:
        after: defaultdict[tuple[str, float], Circuit] = defaultdict(Circuit)
        for split_op in moment_split_ops:
            rule = self._noise_rule_for_split_operation(split_op=split_op)
            if rule is None:
                out.append(split_op)
            else:
                rule._append_noisy_version_of(split_op=split_op, out_during_moment=out, after_moments=after, immune_qubits=immune_qubits)
        for k in sorted(after.keys()):
            out += after[k]
        self._append_idle_error(moment_split_ops=moment_split_ops, out=out, system_qubits=system_qubits, immune_qubits=immune_qubits)

    def noisy_circuit(self, circuit: Union[Circuit, str], *, system_qubits: Optional[set[int]] = None, immune_qubits: Optional[set[int]] = None) -> Circuit:
        """The circuit with this model's noise laid over it.

        Args:
            circuit: the circuit (or its Stim text) to layer noise over.
            system_qubits: every qubit the circuit uses, the qubits eligible for idling noise
                (default all of ``range(circuit.num_qubits)``).
            immune_qubits: qubits to not apply noise to, even if they are operated on.
        """
        if not isinstance(circuit, Circuit):
            circuit = Circuit(str(circuit))
        if system_qubits is None:
            system_qubits = set(range(circuit.num_qubits))
        if immune_qubits is None:
            immune_qubits = set()

        result = Circuit()
        # tqec asks the result whether it is empty and whether it ends with a REPEAT block; this
        # package's circuit answers by reading itself whole, so both are kept track of here, and
        # each moment is written on its own before it joins the result.
        empty = True
        ends_with_repeat = False
        for moment_split_ops in _iter_split_op_moments(circuit, immune_qubits=immune_qubits):
            is_repeat = isinstance(moment_split_ops, CircuitRepeatBlock)
            if not empty and not is_repeat and not ends_with_repeat:
                result.append("TICK", [], [])
            if is_repeat:
                noisy_body = self.noisy_circuit(moment_split_ops.body_copy(), system_qubits=system_qubits, immune_qubits=immune_qubits)
                result.append(CircuitRepeatBlock(repeat_count=moment_split_ops.repeat_count, body=noisy_body))
                empty, ends_with_repeat = False, True
            else:
                moment = Circuit()
                self._append_noisy_moment(moment_split_ops=moment_split_ops, out=moment, system_qubits=system_qubits, immune_qubits=immune_qubits)
                if moment:
                    result += moment
                    empty, ends_with_repeat = False, False
        return result


def occurs_in_classical_control_system(op: CircuitInstruction) -> bool:
    """Whether an operation is an annotation or a classical control system update."""
    t = _op_type(op.name)
    if t == ANNOTATION:
        return True
    if t == CLIFFORD_2Q:
        targets = op.targets_copy()
        for k in range(0, len(targets), 2):
            a = targets[k]
            b = targets[k + 1]
            classical_0 = a.is_measurement_record_target or a.is_sweep_bit_target
            classical_1 = b.is_measurement_record_target or b.is_sweep_bit_target
            if not (classical_0 or classical_1):
                return False
        return True
    return False


def _split_targets_if_needed(op: CircuitInstruction, immune_qubits: AbstractSet[int]) -> Iterator[CircuitInstruction]:
    """Split operations into pieces as needed: MPP into its products, classical control away
    from quantum operations."""
    t = _op_type(op.name)
    if t == CLIFFORD_2Q:
        yield from _split_targets_if_needed_clifford_2q(op, immune_qubits)
    elif t == MPP:
        yield from _split_targets_if_needed_m_basis(op, immune_qubits)
    elif t in [NOISE, ANNOTATION]:
        yield op
    else:
        yield from _split_targets_if_needed_clifford_1q(op, immune_qubits)


def _split_targets_if_needed_clifford_1q(op: CircuitInstruction, immune_qubits: AbstractSet[int]) -> Iterator[CircuitInstruction]:
    if immune_qubits:
        args = op.gate_args_copy()
        for t in op.targets_copy():
            yield CircuitInstruction(op.name, [t], args)
    else:
        yield op


def _split_targets_if_needed_clifford_2q(op: CircuitInstruction, immune_qubits: AbstractSet[int]) -> Iterator[CircuitInstruction]:
    """Split classical control system operations away from quantum operations."""
    assert _op_type(op.name) == CLIFFORD_2Q
    targets = op.targets_copy()
    if immune_qubits or any(t.is_measurement_record_target for t in targets):
        args = op.gate_args_copy()
        for k in range(0, len(targets), 2):
            yield CircuitInstruction(op.name, targets[k : k + 2], args)
    else:
        yield op


def _split_targets_if_needed_m_basis(op: CircuitInstruction, immune_qubits: AbstractSet[int]) -> Iterator[CircuitInstruction]:
    """Split an MPP operation into one operation for each Pauli product it measures."""
    targets = op.targets_copy()
    args = op.gate_args_copy()
    k = 0
    start = k
    while k < len(targets):
        if k + 1 == len(targets) or not targets[k + 1].is_combiner:
            yield CircuitInstruction(op.name, targets[start : k + 1], args)
            k += 1
            start = k
        else:
            k += 2
    assert k == len(targets)


def _iter_split_op_moments(circuit: Circuit, *, immune_qubits: AbstractSet[int]) -> Iterator[Union[CircuitRepeatBlock, list[CircuitInstruction]]]:
    """The circuit's moments (the stretches between TICKs), with operations like MPP split into
    pieces, and its REPEAT blocks whole."""
    cur_moment: list[CircuitInstruction] = []
    for op in circuit:
        if isinstance(op, CircuitRepeatBlock):
            if cur_moment:
                yield cur_moment
                cur_moment = []
            yield op
        elif op.name == "TICK":
            yield cur_moment
            cur_moment = []
        else:
            cur_moment.extend(_split_targets_if_needed(op, immune_qubits=immune_qubits))
    if cur_moment:
        yield cur_moment


def _measure_basis(*, split_op: CircuitInstruction) -> Optional[str]:
    """The Pauli product basis an operation measures (``"XX"``, ``"Y"``), or None if it is not
    (only) a measurement."""
    result = OP_MEASURE_BASES.get(split_op.name)
    if result == "":
        targets = split_op.targets_copy()
        for k in range(0, len(targets), 2):
            t = targets[k]
            if t.is_x_target:
                result += "X"
            elif t.is_y_target:
                result += "Y"
            elif t.is_z_target:
                result += "Z"
            else:
                raise NotImplementedError(f"{targets=}")
    return result

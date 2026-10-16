# (c) Copyright Riverlane 2020-2025.
# modified for stabilizer_qec: its Stim is stabilizer_qec.deltakit_stim and its PyMatching this package's, and deltakit_{circuit,core,decode,explorer} are stabilizer_qec.deltakit.{circuit,core,decode,explorer}.
"""Regression tests for https://github.com/Deltakit/deltakit/issues/354.

``Gate.__hash__`` and ``NoiseChannel.__hash__`` used to include the gate's
``self.__class__`` object in the hashed tuple. A class object's hash is
``object.__hash__``, which is based on the class's memory address. That
address is not fixed by ``PYTHONHASHSEED`` and differs between separate
Python processes, so any code that iterates a ``set``/``frozenset`` of gates
(for example when building a :class:`~deltakit_circuit.GateLayer` from a set
comprehension) produced a different iteration order in every process. That
nondeterministic order propagated through to the emitted Stim circuit: the
order of targets within a single Stim instruction, and the indices assigned
by :func:`~deltakit_circuit.default_qubit_mapping`, could both change from
run to run even though the logical circuit was unchanged.

``Qubit``/``Coordinate`` hashing was never part of the bug: both hash purely
on their value (``hash(self._unique_identifier)`` / ``tuple.__hash__``), so
the fix is scoped to gate and noise-channel hashing.

These tests spawn separate subprocesses (with a fixed ``PYTHONHASHSEED``) to
verify that emission is now reproducible across process boundaries, which an
in-process test cannot demonstrate since ``id()``-based hashes are stable for
the lifetime of a single process.
"""

import hashlib
import os
import subprocess
import sys
import textwrap

from stabilizer_qec import deltakit_stim as stim
import pytest

from stabilizer_qec.deltakit.circuit import Qubit
from stabilizer_qec.deltakit.circuit._qubit_identifiers import Coordinate
from stabilizer_qec.deltakit.circuit.gates import CX, H, I, X

# Script executed in a fresh subprocess. It builds GateLayers from `set`
# comprehensions of gate instances on purpose: this mirrors how
# circuit-generation code (e.g. deltakit_explorer's CSS code builders)
# constructs layers, and is the exact pattern that turned a process-dependent
# gate hash into a process-dependent instruction/target order.
_REPRO_SCRIPT = textwrap.dedent(
    """
    import hashlib
    from stabilizer_qec.deltakit.circuit import Circuit, GateLayer, Qubit
    from stabilizer_qec.deltakit.circuit.gates import CX, H, I, X

    data_qubits = [Qubit((i, j)) for i in range(3) for j in range(3)]
    reset_layer = GateLayer({I(qubit) for qubit in data_qubits})
    hadamard_layer = GateLayer({H(qubit) for qubit in data_qubits})
    cx_layer = GateLayer(
        {CX(data_qubits[i], data_qubits[i + 1]) for i in range(0, 8, 2)}
    )
    circuit = Circuit([reset_layer, hadamard_layer, cx_layer])
    text = str(circuit.as_stim_circuit())
    print(hashlib.sha256(text.encode()).hexdigest())
    for line in text.splitlines():
        print(line)
    """
)

# A larger variant of the script above: many more qubits and gate types
# raises the odds of a genuine hash-table collision/resize occurring while
# `GateLayer`/`Circuit` iterate their internal `set`/`frozenset` of qubits
# and gates, which is exactly the scenario issue #354 worried could still
# desynchronise qubit indices between processes even with a stable per-gate
# hash formula.
_LARGE_REPRO_SCRIPT = textwrap.dedent(
    """
    import hashlib
    from stabilizer_qec.deltakit.circuit import Circuit, GateLayer, Qubit
    from stabilizer_qec.deltakit.circuit.gates import CX, CY, CZ, H, I, S, X, Y, Z

    data_qubits = [Qubit((i, j)) for i in range(8) for j in range(8)]
    one_qubit_gates = [I, X, Y, Z, H, S]
    single_qubit_layer = GateLayer(
        {
            one_qubit_gates[index % len(one_qubit_gates)](qubit)
            for index, qubit in enumerate(data_qubits)
        }
    )
    two_qubit_gate_types = [CX, CY, CZ]
    two_qubit_layer = GateLayer(
        {
            two_qubit_gate_types[i % len(two_qubit_gate_types)](
                data_qubits[i], data_qubits[i + 1]
            )
            for i in range(0, len(data_qubits) - 1, 2)
        }
    )
    circuit = Circuit([single_qubit_layer, two_qubit_layer])
    text = str(circuit.as_stim_circuit())
    print(hashlib.sha256(text.encode()).hexdigest())
    for line in text.splitlines():
        print(line)
    """
)

# The exact expression from the issue report, printed so a subprocess-based
# test can confirm it no longer varies for a fixed PYTHONHASHSEED.
_ISSUE_HASH_SCRIPT = textwrap.dedent(
    """
    from stabilizer_qec.deltakit.circuit import Qubit
    from stabilizer_qec.deltakit.circuit._qubit_identifiers import Coordinate
    from stabilizer_qec.deltakit.circuit.gates import H

    print(hash(H(Qubit(Coordinate(1, 2)))))
    """
)


def _run_in_subprocess(script: str, pythonhashseed: str) -> str:
    env = dict(os.environ, PYTHONHASHSEED=pythonhashseed)
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


def _run_repro(pythonhashseed: str) -> str:
    return _run_in_subprocess(_REPRO_SCRIPT, pythonhashseed)


def _run_large_repro(pythonhashseed: str) -> str:
    return _run_in_subprocess(_LARGE_REPRO_SCRIPT, pythonhashseed)


class TestStimEmissionReproducibleAcrossProcesses:
    """Regression tests exercising the fix across real process boundaries."""

    @pytest.mark.parametrize("seed", ["0", "7", "13"])
    def test_full_stim_circuit_text_is_identical_across_processes(self, seed):
        # Each seed is exercised with two *independent* subprocesses: this is
        # the literal scenario from the issue ("same PYTHONHASHSEED, two
        # processes, different output").
        first = _run_repro(seed)
        second = _run_repro(seed)
        assert first == second

    def test_full_stim_circuit_hash_is_identical_across_processes(self):
        first_hash = _run_repro("1234").splitlines()[0]
        second_hash = _run_repro("1234").splitlines()[0]
        assert len(first_hash) == len(hashlib.sha256(b"").hexdigest())
        assert first_hash == second_hash

    def test_instruction_target_order_is_identical_across_processes(self):
        # The instruction lines (everything after the sha256 header) capture
        # both the target order within each instruction and the qubit
        # indices assigned by `default_qubit_mapping`.
        first_lines = _run_repro("5678").splitlines()[1:]
        second_lines = _run_repro("5678").splitlines()[1:]
        assert first_lines == second_lines

    @pytest.mark.parametrize("seed", ["0", "7", "13"])
    def test_large_circuit_emission_is_identical_across_processes(self, seed):
        # A bigger qubit/gate population than the minimal repro, to exercise
        # hash-table resizing and increase the odds of a genuine collision
        # inside the sets GateLayer/Circuit build internally. If iteration
        # order were still influenced by anything process-dependent, this is
        # far more likely to surface it than a handful of qubits would.
        first = _run_large_repro(seed)
        second = _run_large_repro(seed)
        assert first == second

    def test_issue_reported_hash_expression_is_identical_across_processes(self):
        # hash(H(Qubit(Coordinate(1, 2)))) was called out explicitly in the
        # issue as evidence of process-dependent hashing.
        seed = "42"
        first = _run_in_subprocess(_ISSUE_HASH_SCRIPT, seed).strip()
        second = _run_in_subprocess(_ISSUE_HASH_SCRIPT, seed).strip()
        assert first == second


class TestDifferentSeedsDoNotIntroduceUnintendedNondeterminism:
    """Different PYTHONHASHSEED values may legitimately change *text*
    ordering (that is documented Python behaviour for string hashing, and
    `type(self).__qualname__` is a string), but must never change which
    qubits, gates and targets the circuit is semantically made of."""

    @staticmethod
    def _stim_instructions_as_multiset(
        stim_text: str,
    ) -> set[tuple[str, frozenset]]:
        """Reduce a Stim circuit body to an order-independent summary.

        For each non-QUBIT_COORDS instruction, the summary contains its name
        and the set of qubit coordinates it targets. Coordinates are recovered
        through QUBIT_COORDS so the comparison does not depend on the integer
        index assigned to a qubit.

        Args:
            stim_text: Text representation of a Stim circuit.

        Returns:
            A set containing instruction names and their targeted qubit
            coordinates.
        """
        circuit = stim.Circuit(stim_text)
        index_to_coords: dict[int, tuple[float, ...]] = {}
        for instruction in circuit:
            if (
                isinstance(instruction, stim.CircuitInstruction)
                and instruction.name == "QUBIT_COORDS"
            ):
                (target,) = instruction.targets_copy()
                index_to_coords[target.value] = tuple(instruction.gate_args_copy())
        summary: set[tuple[str, frozenset]] = set()
        for instruction in circuit:
            if (
                isinstance(instruction, stim.CircuitInstruction)
                and instruction.name != "QUBIT_COORDS"
            ):
                targets = frozenset(
                    index_to_coords.get(target.value, target.value)
                    for target in instruction.targets_copy()
                    if target.is_qubit_target
                )
                summary.add((instruction.name, targets))
        return summary

    @pytest.mark.parametrize("seed", ["0", "7", "13"])
    def test_circuit_semantics_are_equivalent_across_seeds(self, seed):
        baseline = _run_repro("1234").splitlines()[1:]
        other = _run_repro(seed).splitlines()[1:]
        baseline_summary = self._stim_instructions_as_multiset("\n".join(baseline))
        other_summary = self._stim_instructions_as_multiset("\n".join(other))
        assert baseline_summary == other_summary


class TestGateHashNoLongerDependsOnClassObjectIdentity:
    """In-process checks that the hash formula itself changed as intended."""

    @pytest.mark.parametrize("gate", [H(Qubit(0)), X(Qubit(0)), I(Qubit(0))])
    def test_one_qubit_gate_hash_matches_qualname_based_formula(self, gate):
        assert hash(gate) == hash((type(gate).__qualname__, gate.qubit))

    def test_two_operand_gate_hash_matches_qualname_based_formula(self):
        gate = CX(Qubit(0), Qubit(1))
        assert hash(gate) == hash((type(gate).__qualname__, gate.control, gate.target))

    def test_equal_gates_still_have_equal_hashes(self):
        # __eq__ was not touched by this fix; equality must still imply
        # equal hashes (the Python hash/eq contract).
        assert H(Qubit(0)) == H(Qubit(0))
        assert hash(H(Qubit(0))) == hash(H(Qubit(0)))

    def test_gates_of_different_types_on_the_same_qubit_are_unequal(self):
        assert H(Qubit(0)) != X(Qubit(0))

    def test_issue_reported_hash_expression_is_a_plain_int_in_process(self):
        # Sanity check that the expression from the issue still evaluates
        # (i.e. Coordinate/Qubit hashing was untouched by this fix).
        assert isinstance(hash(H(Qubit(Coordinate(1, 2)))), int)

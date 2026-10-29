"""The circuit utilities, noise models, cat-state experiment and simulators take Stim's own circuits
(this package's tests, not MQT's).

MQT QECC 2.0.0 runs on Stim, so its users hand these functions ``stim.Circuit`` objects; in the
port, ``stim`` is this package, and a real Stim circuit failed the ``isinstance`` asserts
(AssertionError) or this package's ``Circuit +=`` (TypeError). Each now reads a Stim circuit from
its text first, and answers as it does given this package's copy."""

from __future__ import annotations

import numpy as np
import pytest

import stabilizer_qec as sq
from stabilizer_qec.mqt_qecc import CSSCode
from stabilizer_qec.mqt_qecc.circuit_synthesis import (
    CircuitLevelNoiseIdlingParallel,
    CircuitLevelNoiseIdlingSequential,
    VerificationNDFTStatePrepSimulator,
    heuristic_prep_circuit,
)
from stabilizer_qec.mqt_qecc.circuit_synthesis import cat_states, circuit_utils, noise

stim = pytest.importorskip("stim", minversion="1.16")

TEXT = "H 0\nCX 0 1 0 2\nCX 1 3\nTICK\nCX 2 4 3 5\nSWAP 4 5\nM 4 5"
OTHER = "H 0 1\nCX 0 2 1 3 0 1"


def _same(f, *texts) -> None:
    """``f`` on Stim's circuits answers as on this package's."""
    got = f(*(stim.Circuit(t) for t in texts))
    want = f(*(sq.Circuit(t) for t in texts))
    assert str(got) == str(want)


@pytest.mark.parametrize(
    "f",
    [
        circuit_utils.depth,
        circuit_utils.num_two_qubit_gates,
        lambda c: circuit_utils.num_two_qubit_gates(c, count_swaps=True),
        circuit_utils.two_qubit_gate_depth,
        circuit_utils.measured_qubits,
        circuit_utils.unmeasured_qubits,
        lambda c: [str(layer) for layer in circuit_utils.collect_circuit_layers(c)],
        lambda c: [str(layer) for layer in circuit_utils.collect_circuit_layers(c, "alap")],
        circuit_utils.remove_single_qubit_gates,
        circuit_utils.remove_swap_gates,
        circuit_utils.compact_stim_circuit,
        lambda c: circuit_utils.relabel_qubits(c, 3),
        lambda c: circuit_utils.relabel_qubits(c, {q: 5 - q for q in range(6)}),
        noise.CircuitLevelNoise(0.01, 0.02, 0.03, 0.04).apply,
        CircuitLevelNoiseIdlingParallel(0.01, 0.02, 0.03, 0.04, 0.005).apply,
        CircuitLevelNoiseIdlingSequential(0.01, 0.02, 0.03, 0.04, 0.005).apply,
        noise.ComposedNoiseModel([noise.CircuitLevelNoise(0.01, 0.02, 0.03, 0.04)]).apply,
    ],
)
def test_one_circuit_functions_take_stim_circuits(f) -> None:
    _same(f, TEXT)


@pytest.mark.parametrize(
    "f",
    [
        lambda a, b: circuit_utils.compose_circuits(a, b),
        lambda a, b: circuit_utils.compose_circuits(a, b, {4: 0, 5: 1}),
        lambda a, b: circuit_utils.compose_compact_stim_circuits([a, b]),
        lambda a, b: circuit_utils.compose_compact_stim_circuits([a, b], "end"),
        lambda a, b: cat_states.CatStatePreparationExperiment(a, b).circ,
    ],
)
def test_two_circuit_functions_take_stim_circuits(f) -> None:
    _same(f, TEXT, OTHER)


def test_cat_state_experiment_samples_stim_circuits() -> None:
    line = str(cat_states.cat_state_line(4))
    rates = []
    for make in (stim.Circuit, sq.Circuit):
        np.random.seed(3)
        experiment = cat_states.CatStatePreparationExperiment(make(line), make(line))
        rates.append(experiment.sample_cat_state(0.0, 64)[0])
    assert rates == [1.0, 1.0]


def test_verification_simulator_takes_stim_circuits() -> None:
    steane = CSSCode.from_code_name("Steane")
    prep = heuristic_prep_circuit(steane).circ.to_stim_circuit()
    real = VerificationNDFTStatePrepSimulator(stim.Circuit(str(prep)), code=steane)
    ours = VerificationNDFTStatePrepSimulator(prep, code=steane)
    assert isinstance(real.circ, sq.Circuit)
    assert str(real.circ) == str(ours.circ)
    assert real.data_qubits == ours.data_qubits
    assert real.data_measurements == ours.data_measurements

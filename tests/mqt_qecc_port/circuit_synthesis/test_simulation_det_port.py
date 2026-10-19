"""NoisyDFTStatePrepSimulator on this package's qsample (this package's tests, not MQT's).

MQT QECC 2.0.0 never runs it (its CI installs no qsample), and it had two faults there: CNOTs went
to qsample as ``{control, target}`` sets, which qsample applies one qubit at a time and refuses, and
the final decoding took the whole check matrix as one stabilizer, so it corrected nothing and one
fault failed the preparation. MQT's own simulation tests (``test_deterministic.py``) check the
quadratic scaling; these check the protocol without sampling."""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

import stabilizer_qec as stim
from stabilizer_qec.mqt_qecc import CSSCode
from stabilizer_qec.mqt_qecc.circuit_synthesis import (
    CNOTCircuit,
    DeterministicVerificationHelper,
    FaultyStatePrepCircuit,
    heuristic_prep_circuit,
)
from stabilizer_qec.mqt_qecc.circuit_synthesis.simulation_det import NoisyDFTStatePrepSimulator, qiskit_to_qsample
from stabilizer_qec.qsample import StabilizerSimulator, noise


@pytest.fixture(scope="module")
def steane() -> CSSCode:
    return CSSCode.from_code_name("Steane")


@pytest.fixture(scope="module")
def zero_prep(steane: CSSCode):
    sp = heuristic_prep_circuit(steane)
    return sp, DeterministicVerificationHelper(sp).get_solution(use_optimal_verification=True)


@pytest.fixture(scope="module")
def plus_prep(steane: CSSCode):
    # MQT's |+> fixture (test_deterministic.steane_code_sp_plus)
    sp = FaultyStatePrepCircuit(
        CNOTCircuit.from_stim_circuit(stim.Circuit("R 0 1 3\nRX 2 4 5 6\nCX 5 3 2 0 6 5 4 3 0 1 6 2 5 1 4 0")),
        steane.x_distance // 2,
        steane.z_distance // 2,
    )
    sp.compute_fault_sets()
    return sp, DeterministicVerificationHelper(sp).get_solution()


def _walk(sim: NoisyDFTStatePrepSimulator) -> tuple[str | None, ...]:
    """One fault-free run of the protocol: the nodes it visits."""
    prot = sim.protocol
    state = StabilizerSimulator(len(prot.qubits))
    hist: dict = {}
    node, path = prot.root, []
    while True:
        node, circuit = prot.successor(node, hist)
        path.append(node)
        if circuit is None:
            return tuple(path)
        m = state.run(circuit)
        hist[node] = [*hist.get(node, []), None if m is None else int(m, 2)]


def test_cnots_reach_qsample_as_control_target(zero_prep) -> None:
    sp, _ = zero_prep
    qc = sp.circ.to_qiskit_circuit()
    cnots = [(qc.qubits.index(i.qubits[0]), qc.qubits.index(i.qubits[1])) for i in qc.data if i.operation.name == "cx"]
    ticks = [tick["CNOT"] for tick in qiskit_to_qsample(qc) if "CNOT" in tick]
    assert ticks == [{c} for c in cnots]


def test_decoding_measures_each_stabilizer(steane: CSSCode, zero_prep) -> None:
    sp, verifications = zero_prep
    sim = NoisyDFTStatePrepSimulator(sp.circ.to_qiskit_circuit(), verifications, steane, noise.E1_1)
    dec = sim.protocol.get_circuit("DEC")
    measured = [q for tick in dec for q in tick.get("measure", ())]
    assert len(measured) == steane.n + len(steane.Hz)  # what decode_failure reads
    ancillas = sorted({q for tick in dec for pair in tick.get("CNOT", ()) for q in pair} - set(range(steane.n)))
    supports = [sorted(t for tick in dec for (t, a) in tick.get("CNOT", ()) if a == anc) for anc in ancillas]
    assert supports == [list(np.flatnonzero(row)) for row in steane.Hz]


@pytest.mark.parametrize("which", ["zero", "plus"])
def test_fault_free_runs_never_fail(steane: CSSCode, zero_prep, plus_prep, which: str) -> None:
    sp, (v0, v1) = zero_prep if which == "zero" else plus_prep
    verifications = (v0, v1) if which == "zero" else (v1, v0)  # |+>: as MQT's test_steane_det_simulation
    sim = NoisyDFTStatePrepSimulator(sp.circ.to_qiskit_circuit(), verifications, steane, noise.E1_1, which == "zero")
    np.random.seed(0)
    ends = Counter(_walk(sim)[-1] for _ in range(64))
    assert ends["FAIL"] == 0
    assert ends[None] == 64


def test_plus_with_x_error_verification_first_is_refused(steane: CSSCode, plus_prep) -> None:
    sp, verifications = plus_prep
    with pytest.raises(ValueError, match=r"get_solution\(\) reversed"):
        NoisyDFTStatePrepSimulator(sp.circ.to_qiskit_circuit(), verifications, steane, noise.E1_1, False)


def test_doc_example_runs(steane: CSSCode, zero_prep) -> None:
    # MQT's StatePrep.md example, at fewer shots: one fault never fails it, so p_L is below p.
    sp, verifications = zero_prep
    np.random.seed(1)
    sim = NoisyDFTStatePrepSimulator(sp.circ.to_qiskit_circuit(), verifications, steane, noise.E1_1)
    stats = sim.dss_logical_error_rates({"q": [1e-4, 1e-3, 1e-2]}, {"q": 0.01}, 1, 300)
    upper = np.asarray(stats[-2], dtype=float)
    assert np.all(upper < np.array([1e-4, 1e-3, 1e-2]))
    slopes = np.diff(np.log10(upper)) / np.diff(np.log10([1e-4, 1e-3, 1e-2]))
    assert np.all(slopes > 1.5)

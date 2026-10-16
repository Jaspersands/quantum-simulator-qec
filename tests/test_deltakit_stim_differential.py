"""stabilizer_qec.deltakit_stim against deltakit_stim 0.2.7 itself (Riverlane's Stim, where it
is installed): its leakage instructions' text, instructions, gate data, noise removal and
decomposition, its generated circuits with leakage, and its detector error models (heralds,
transport gates, loops, decomposition) all equal; its samplers' rates equal within sampling
error, except where Deltakit's mobility stalls (see test_mobility_where_deltakit_stalls)."""

import numpy as np
import pytest

from conftest import reference

dk = reference("deltakit_stim")

import stabilizer_qec.deltakit_stim as ours  # noqa: E402

CASES = [
    "R 0\nLEAKAGE(0.1) 0\nHERALD_LEAKAGE_EVENT(0.01) 0\nM 0\nDETECTOR rec[-1]\nDETECTOR rec[-2]",
    "R 0 1\nLEAKAGE(0.2) 0\nCX(0.1, 0.05, 0.2, 0.01) 0 1\nHERALD_LEAKAGE_EVENT 0 1\nM 1\nDETECTOR rec[-1]\nDETECTOR rec[-2]\nDETECTOR rec[-3]",
    "R 0 1 2\nLEAKAGE(0.05) 0 1\nCX 0 1\nCZ 1 2\nLEAKAGE(0.01) 2\nRELAX(0.3) 0\nHERALD_LEAKAGE_EVENT(0.02) 0 1 2\nH 0\nM 0 1 2\nDETECTOR rec[-1]\nDETECTOR rec[-2]\nDETECTOR rec[-4] rec[-5]",
    "R 0 1\nREPEAT 3 {\nLEAKAGE(0.02) 0\nCX 0 1\nHERALD_LEAKAGE_EVENT(0.01) 0\nMR 1\nDETECTOR rec[-1]\nDETECTOR rec[-2]\n}\nM 0\nDETECTOR rec[-1]",
    "R 0 1\nLEAKAGE(0.1) 0\nRL 0\nHERALD_LEAKAGE_EVENT 0\nDEPOLARIZE1(0.1) 1\nM 1\nDETECTOR rec[-1]\nDETECTOR rec[-2]",
    "RX 0 1\nLEAKAGE(0.1) 0 1\nSQRT_XX(0.1, 0.1, 0.1, 0.1) 0 1\nHERALD_LEAKAGE_EVENT(0.001) 0 !1\nMX 0 1\nDETECTOR rec[-1] rec[-2]\nDETECTOR rec[-3]\nDETECTOR rec[-4]",
    "R 0 1 2 3\nX_ERROR(0.05) 0 2\nLEAKAGE(0.03) 0 1 2 3\nCZ(0.2, 0.1, 0, 0) 0 1 2 3\nCY 1 2\nHERALD_LEAKAGE_EVENT(0.05) 0 1 2 3\nM 0 1 2 3\nDETECTOR rec[-1] rec[-2]\nDETECTOR rec[-3] rec[-4]\nDETECTOR rec[-5]\nDETECTOR rec[-8]\nOBSERVABLE_INCLUDE(0) rec[-1]",
]
DEM_OPTIONS = [{}, {"decompose_errors": True}, {"approximate_disjoint_errors": True}, {"flatten_loops": True}]


def _instructions(c):
    return [(i.name, list(i.gate_args_copy()), str(i), i.num_measurements) for i in c if not isinstance(i, (dk.CircuitRepeatBlock, ours.CircuitRepeatBlock))]


@pytest.mark.parametrize("text", CASES)
def test_text_and_instructions_equal(text):
    a, b = ours.Circuit(text), dk.Circuit(text)
    assert str(a) == str(b)
    assert _instructions(a) == _instructions(b)
    assert (a.num_measurements, a.num_detectors, a.num_qubits) == (b.num_measurements, b.num_detectors, b.num_qubits)
    assert str(a.without_noise()) == str(b.without_noise())
    assert str(a.decomposed()) == str(b.decomposed())
    # Its text read back is the same circuit.
    assert str(ours.Circuit(str(b))) == str(b)


@pytest.mark.parametrize("text", CASES)
@pytest.mark.parametrize("options", DEM_OPTIONS, ids=["plain", "decompose", "disjoint", "flatten"])
def test_detector_error_models_equal(text, options):
    a, b = ours.Circuit(text), dk.Circuit(text)
    assert str(a.detector_error_model(**options)) == str(b.detector_error_model(**options))


def test_generated_circuits_with_leakage_equal():
    for code in ("repetition_code:memory", "surface_code:rotated_memory_x", "surface_code:unrotated_memory_z", "color_code:memory_xyz"):
        kwargs = dict(distance=3, rounds=3, after_clifford_depolarization=0.001, before_measure_flip_probability=0.002, after_reset_flip_probability=0.003, after_reset_leakage=0.0004, after_clifford_leakage_and_relaxation=0.0005)
        a, b = ours.Circuit.generated(code, **kwargs), dk.Circuit.generated(code, **kwargs)
        assert str(a) == str(b), code
        assert str(a.detector_error_model(decompose_errors=code != "color_code:memory_xyz")) == str(b.detector_error_model(decompose_errors=code != "color_code:memory_xyz")), code


def test_gate_data_equal():
    fields = ("name", "is_noisy_gate", "is_reset", "is_single_qubit_gate", "is_two_qubit_gate", "is_unitary", "produces_measurements", "num_parens_arguments_range", "takes_measurement_record_targets", "takes_pauli_targets")
    for name in ("LEAKAGE", "RELAX", "RL", "HERALD_LEAKAGE_EVENT", "CX", "CY", "CZ", "SQRT_XX", "SQRT_XX_DAG", "H", "M", "DEPOLARIZE2"):
        a, b = ours.gate_data(name), dk.gate_data(name)
        assert [getattr(a, f) for f in fields] == [getattr(b, f) for f in fields], name


def _rates(module, text, shots, seed):
    d = module.Circuit(text).compile_detector_sampler(seed=seed).sample(shots)
    n = d.shape[1]
    pairs = [(d[:, i] & d[:, j]).mean() for i in range(n) for j in range(i + 1, n)]
    return np.concatenate([d.mean(axis=0), pairs])


# Without a mobility that a spread can block (see below), Deltakit's frame simulator's leakage
# model is this one's, shot for shot in distribution.
SAMPLED = [CASES[0], CASES[2], CASES[4], CASES[5], CASES[6], "R 0 1\nLEAKAGE(0.2) 0\nCX(0, 0, 0.3, 0.1) 0 1\nHERALD_LEAKAGE_EVENT 0 1\nM 1\nDETECTOR rec[-1]\nDETECTOR rec[-2]\nDETECTOR rec[-3]"]


@pytest.mark.parametrize("text", SAMPLED)
def test_sampled_rates_equal(text):
    shots = 200_000
    a, b = _rates(ours, text, shots, 5), _rates(dk, text, shots, 5)
    # Six standard errors of a difference of two rates.
    tolerance = 6 * np.sqrt(2 * np.maximum(a, b) * (1 - np.minimum(a, b)) / shots) + 1e-9
    assert np.all(np.abs(a - b) <= tolerance), (a, b)


def test_mobility_where_deltakit_stalls():
    # Deltakit draws a gate's mobility by skipping through its leaked shots, and does not skip
    # past a draw that lands on a leaked partner: from the first such shot (here, one its spread
    # leaked), it moves no leakage in the rest of the batch, which is every shot sampled. This
    # package moves it in every shot, as the probability says: leaked 0.2, then spread to the
    # partner 0.1 and otherwise moved 0.2.
    text = CASES[1]
    shots = 200_000
    a, b = _rates(ours, text, shots, 5), _rates(dk, text, shots, 5)
    herald0, herald1 = 0.2 * (1 - 0.9 * 0.2), 0.2 * (0.1 + 0.9 * 0.2)
    assert abs(a[2] - herald0) < 0.005 and abs(a[1] - herald1) < 0.005, a
    assert abs(b[2] - 0.2) < 0.005 and abs(b[1] - 0.02) < 0.005, b

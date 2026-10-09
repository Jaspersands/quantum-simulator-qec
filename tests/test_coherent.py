"""The coherent sampler: weighted shots of the twirled circuit whose weighted distribution is
the coherent circuit's. Checked against the exact distribution of the state vector."""

from __future__ import annotations

import numpy as np
import pytest

import stabilizer_qec as sq


def weighted_marginals(circuit: sq.Circuit, shots: int, seed: int = 1):
    d, o, w = circuit.compile_coherent_sampler(seed=seed).sample(shots, threads=0)
    rows = np.concatenate([d, o], axis=1).astype(float)
    W = w.sum()
    mean = (w[:, None] * rows).sum(0) / W
    se = np.sqrt(((w[:, None] * (rows - mean)) ** 2).sum(0)) / W
    return mean, se, w


def exact_marginals(circuit: sq.Circuit):
    dist = circuit.exact_distribution()
    nd = circuit.num_detectors
    out = np.zeros(nd + circuit.num_observables)
    for (dets, obs), p in dist.items():
        out += p * np.array(dets + obs, dtype=float)
    return out


CASES = [
    # Two rotations that add up, either side of a gate that leaves them alone.
    "RX 0\nR 1\nI_ERROR[R_Z(theta=0.2)] 0\nCZ 0 1\nI_ERROR[R_Z(theta=0.15)] 0\nMX 0\nM 1\nDETECTOR rec[-2]\nDETECTOR rec[-1]",
    # A repetition code with coherent X over-rotations and depolarising noise.
    "R 0 1 2 3 4\nI_ERROR[R_X(theta=0.15)] 0 2 4\nCX 0 1 2 3\nDEPOLARIZE2(0.02) 0 1 2 3\nCX 2 1 4 3\nMR 1 3\n"
    "I_ERROR[R_X(theta=0.15)] 0 2 4\nCX 0 1 2 3\nCX 2 1 4 3\nMR 1 3\nDETECTOR rec[-4]\nDETECTOR rec[-3]\n"
    "DETECTOR rec[-1] rec[-3]\nDETECTOR rec[-2] rec[-4]\nM 0 2 4\nDETECTOR rec[-1] rec[-2] rec[-4]\nDETECTOR rec[-2] rec[-3] rec[-5]\nOBSERVABLE_INCLUDE(0) rec[-1]",
    # ZZ crosstalk on a Bell pair, random outcomes.
    "R 0 1\nH 0\nCX 0 1\nII_ERROR[R_ZZ(theta=0.3)] 0 1\nI_ERROR[R_X(theta=0.2)] 1\nH 0 1\nM 0 1\nDETECTOR rec[-1] rec[-2]\nOBSERVABLE_INCLUDE(0) rec[-1]",
]


@pytest.mark.parametrize("text", CASES)
def test_weighted_marginals_equal_the_exact(text):
    c = sq.Circuit(text)
    mean, se, _ = weighted_marginals(c, 200_000)
    exact = exact_marginals(c)
    assert np.all(np.abs(mean - exact) < 5 * se + 1e-9), (mean, exact, se)


def test_coherent_errors_are_not_their_twirl():
    """Rotations that add: the coherent flip rate is sin²(θ₁+θ₂), the twirl's not."""
    c = sq.Circuit(CASES[0])
    mean, se, _ = weighted_marginals(c, 200_000)
    assert abs(mean[0] - np.sin(0.35) ** 2) < 5 * se[0]
    twirl = np.sin(0.2) ** 2 * np.cos(0.15) ** 2 + np.cos(0.2) ** 2 * np.sin(0.15) ** 2
    assert abs(mean[0] - twirl) > 20 * se[0]


def test_without_rotation_the_weights_are_one():
    c = sq.Circuit.generated("repetition_code:memory", distance=3, rounds=3, after_clifford_depolarization=0.05)
    text = str(c).replace("TICK", "TICK\nI_ERROR[R_X(theta=0)] 0")
    _, _, w = weighted_marginals(sq.Circuit(text), 5000)
    assert np.all(w == 1.0)


def test_threads_do_not_change_the_shots():
    c = sq.Circuit(CASES[1])
    a = c.compile_coherent_sampler(seed=4).sample(1000, threads=1)
    b = c.compile_coherent_sampler(seed=4).sample(1000, threads=4)
    for x, y in zip(a, b):
        assert np.array_equal(x, y)


def test_unsupported_operations_are_refused():
    with pytest.raises(ValueError, match="rotations only"):
        sq.Circuit("R 0\nI[T] 0\nM 0").compile_coherent_sampler()
    with pytest.raises(ValueError, match="feedback"):
        sq.Circuit("R 0 1\nM 0\nCX rec[-1] 1\nI_ERROR[R_X(theta=0.1)] 1\nM 1").compile_coherent_sampler()


def test_twirled_circuits():
    c = sq.Circuit(CASES[0])
    t = c.twirled()
    assert "R_Z" not in str(t) and "Z_ERROR" in str(t)
    assert np.isclose(float(str(t).split("Z_ERROR(")[1].split(")")[0]), np.sin(0.2) ** 2)
    m = c.twirled(merge=True)
    # The two rotations either side of the CZ are one fault: one error of sin²(0.35).
    probs = [float(x.split(")")[0]) for x in str(m).split("Z_ERROR(")[1:]]
    assert len(probs) == 1 and np.isclose(probs[0], np.sin(0.35) ** 2)
    # ZZ crosstalk twirls to a correlated error.
    assert "E(" in str(sq.Circuit(CASES[2]).twirled()) or "CORRELATED_ERROR(" in str(sq.Circuit(CASES[2]).twirled())


def test_weighted_helpers():
    p, se = sq.weighted_rate(np.array([1, 0, 0, 1]), np.array([1.0, 1.0, 1.0, 1.0]))
    assert p == 0.5 and se > 0
    c = sq.Circuit(CASES[1])
    d, o, w = c.compile_coherent_sampler(seed=3).sample(50_000, threads=0)
    dec = sq.Matching.from_detector_error_model(c.twirled(merge=True).detector_error_model(decompose_errors=True))
    rate, err = sq.weighted_logical_error_rate(dec, d, o, w)
    assert 0 <= rate < 0.5 and err > 0
    assert 0 < sq.effective_sample_size(w) <= len(w)

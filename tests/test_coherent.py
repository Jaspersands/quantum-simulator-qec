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
    # A T gate goes to the joint sampler now; leakage and feedback are refused.
    assert sq.Circuit("R 0\nI[T] 0\nM 0").compile_coherent_sampler().num_exact == 1
    with pytest.raises(ValueError, match="rotations only"):
        sq.Circuit("R 0\nI_ERROR[LEAK(p=0.1)] 0\nM 0").compile_coherent_sampler()
    with pytest.raises(ValueError, match="feedback"):
        sq.Circuit("R 0 1\nM 0\nCX rec[-1] 1\nI_ERROR[R_X(theta=0.1)] 1\nM 1").compile_coherent_sampler()
    with pytest.raises(ValueError, match="feedback"):
        sq.Circuit("R 0 1\nM 0\nCX rec[-1] 1\nI_ERROR[R_X(theta=0.1)] 1\nI[T] 1\nM 1").compile_coherent_sampler()


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


# The joint sampler: non-Clifford gates exact, coherent errors twirled and reweighted.
JOINT_CASES = [
    "R 0\nH 0\nI_ERROR[R_X(theta=0.3)] 0\nI[T] 0\nH 0\nM 0\nDETECTOR rec[-1]",
    "R 0 1\nH 0\nI_ERROR[R_Y(theta=0.25)] 0\nCX 0 1\nI[T] 1\nII_ERROR[R_ZZ(theta=0.2)] 0 1\nI[T_DAG] 0\nH 0 1\nM 0 1\nDETECTOR rec[-1]\nDETECTOR rec[-2]\nOBSERVABLE_INCLUDE(0) rec[-1]",
    "R 0 1\nH 0\nI_ERROR[R_X(theta=0.3)] 0\nDEPOLARIZE1(0.1) 1\nCX 0 1\nI[T] 0\nMR 1\nI_ERROR[R_Z(theta=0.2)] 1\nCX 0 1\nI[U3(theta=0.5, phi=0.2, lambda=1.1)] 1\nH 0\nM(0.05) 0 1\nDETECTOR rec[-1] rec[-3]\nDETECTOR rec[-2]\nOBSERVABLE_INCLUDE(0) rec[-1]",
]


@pytest.mark.parametrize("text", JOINT_CASES)
def test_joint_sampler_is_exact_with_the_whole_kernel(text):
    c = sq.Circuit(text)
    assert c.num_non_clifford_gates() > 0
    a, b = c.exact_distribution(method="coherent"), c.exact_distribution()
    for key in set(a) | set(b):
        assert abs(a.get(key, 0) - b.get(key, 0)) < 1e-9, (text, key)


@pytest.mark.parametrize("text", JOINT_CASES)
def test_joint_weighted_marginals_equal_the_exact(text):
    c = sq.Circuit(text)
    s = c.compile_coherent_sampler(seed=3)
    assert isinstance(s, sq.CoherentSampler) and s.num_exact > 0 and s.num_locations > 0
    shots = 20000
    d, o, w = s.sample(shots, threads=0)
    exact = exact_marginals(c)
    for k in range(c.num_detectors):
        rate = (w * d[:, k]).sum() / w.sum()
        ess = w.sum() ** 2 / (w**2).sum()
        sigma = np.sqrt(exact[k] * (1 - exact[k]) / ess) + 1e-9
        assert abs(rate - exact[k]) < 5 * sigma, (text, k, rate, exact[k], ess)


def test_joint_sampler_differs_from_the_twirl_and_runs_at_size():
    # A T gate between two over-rotations: the twirl misses their interference.
    text = "R 0\nH 0\nI_ERROR[R_Z(theta=0.3)] 0\nI[T] 0\nI_ERROR[R_Z(theta=0.3)] 0\nH 0\nM 0\nDETECTOR rec[-1]"
    c = sq.Circuit(text)
    exact = c.exact_distribution()[((True,), ())]
    twirl = c.twirled().exact_distribution(method="stabilizer_rank")[((True,), ())]
    assert abs(exact - twirl) > 0.02
    d, _, w = c.compile_coherent_sampler(seed=1).sample(20000)
    assert abs((w * d[:, 0]).sum() / w.sum() - exact) < 0.01, ((w * d[:, 0]).sum() / w.sum(), exact, twirl)
    # The twirled circuit keeps its T gate and twirls the errors alone.
    assert c.twirled().num_non_clifford_gates() == 1 and "Z_ERROR" in str(c.twirled())
    # Seven T gates on a Steane code state with an over-rotation after every gate: it runs.
    big = sq.Circuit(
        "R 0 1 2 3 4 5 6\nH 4 5 6\nI_ERROR[R_Z(theta=0.02)] 4 5 6\nCX 4 0 4 1 4 2\nII_ERROR[R_ZZ(theta=0.02)] 4 0 4 1 4 2\nCX 5 0 5 1 5 3\nII_ERROR[R_ZZ(theta=0.02)] 5 0 5 1 5 3\n"
        "CX 6 0 6 2 6 3\nII_ERROR[R_ZZ(theta=0.02)] 6 0 6 2 6 3\nI[T] 0 1 2 3 4 5 6\nI_ERROR[R_X(theta=0.02)] 0 1 2 3 4 5 6\nMX 0 1 2 3 4 5 6\n"
        "DETECTOR rec[-7] rec[-6] rec[-5] rec[-3]\nDETECTOR rec[-7] rec[-6] rec[-4] rec[-2]\nOBSERVABLE_INCLUDE(0) rec[-1]"
    )
    s = big.compile_coherent_sampler(seed=2)
    d, o, w = s.sample(200, threads=0)
    assert d.shape == (200, 2) and np.all(w >= 0) and np.isfinite(w).all()
    assert sq.effective_sample_size(w) > 20

"""The coherent sampler: weighted shots of the twirled circuit whose weighted distribution is
the coherent circuit's. Checked against the exact distribution of the state vector."""

from __future__ import annotations

import importlib.util
import json
import pathlib
import shutil
import subprocess
import sys
import types
import warnings

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


# ---------------------------------------------------------------------------------------------
# The 2.0 review's regressions
# ---------------------------------------------------------------------------------------------


def joint_rate(c: sq.Circuit, shots: int, seed: int = 1, **options):
    d, _, w = c.compile_coherent_sampler(seed=seed, **options).sample(shots, threads=0)
    rate = (w * d[:, 0]).sum() / w.sum()
    ess = w.sum() ** 2 / (w**2).sum()
    return rate, np.sqrt(max(rate * (1 - rate), 1e-4) / ess)


@pytest.mark.parametrize(
    "text",
    [
        # The review's case: two errors around two T gates, where the kernel's basis over every
        # location projected onto the errors without the single-error elements (sv 0.2575, the
        # truncated sum 0.1613 whatever the order).
        "R 0\nI_ERROR[R_X(theta=0.2)] 0\nI[T] 0\nH 0\nI_ERROR[R_Y(theta=0.15)] 0\nI[T] 0\nH 0\nM 0\nDETECTOR rec[-1]",
        # Three errors among five exact gates on one qubit.
        "R 0\nH 0\nI_ERROR[R_Z(theta=0.23)] 0\nI[T] 0\nI_ERROR[R_X(theta=0.2)] 0\nI[T] 0\nH 0\nI[T_DAG] 0\nI_ERROR[R_Y(theta=0.18)] 0\nI[R_Z(theta=0.4)] 0\nH 0\nI[T] 0\nM 0\nDETECTOR rec[-1]",
        # Two qubits, errors on both sides of a CX between T gates.
        "R 0 1\nH 0 1\nI_ERROR[R_Z(theta=0.2)] 0\nI[T] 1\nCX 0 1\nI_ERROR[R_X(theta=0.15)] 1\nI[T] 0\nII_ERROR[R_ZZ(theta=0.1)] 0 1\nH 0\nM 0 1\nDETECTOR rec[-2]\nDETECTOR rec[-1] rec[-2]",
    ],
)
def test_the_truncated_joint_sampler_is_unbiased_at_first_order(text):
    c = sq.Circuit(text)
    exact = exact_marginals(c)[0]
    for options in ({}, {"pairs": True}):
        rate, se = joint_rate(c, 20000, **options)
        assert abs(rate - exact) < 4 * se + 0.003, (options, rate, exact, se)


def test_clifford_angle_coherent_errors_are_the_gates_they_are():
    # S T S T† = S² = Z between Hadamards: the detector fires every shot. Twirled (I or Z, 50/50)
    # it would fire half the time; the samplers run the fixed unitary.
    # The reference run ignores errors (as Stim's), so the detector fires on every shot.
    c = sq.Circuit("R 0\nH 0\nI_ERROR[R_Z(theta=0.7853981633974483)] 0\nI[T] 0\nI_ERROR[R_Z(theta=0.7853981633974483)] 0\nI[T_DAG] 0\nH 0\nM 0\nDETECTOR rec[-1]")
    assert c.exact_distribution()[((True,), ())] == pytest.approx(1.0)
    assert c.exact_distribution(method="coherent")[((True,), ())] == pytest.approx(1.0)
    s = c.compile_coherent_sampler(seed=1)
    assert s.num_locations == 0 and s.num_exact == 4  # two T gates and two fixed S errors, all exact
    d, _, w = s.sample(200)
    assert d.all() and np.allclose(w, 1.0)
    # Without a T gate the circuit still gets the joint sampler (the Z error is a fixed Clifford).
    c2 = sq.Circuit("R 0\nH 0\nI_ERROR[R_Z(theta=1.5707963267948966)] 0\nI_ERROR[R_X(theta=0.1)] 0\nH 0\nM 0\nDETECTOR rec[-1]")
    s2 = c2.compile_coherent_sampler(seed=1)
    assert s2.num_locations == 1 and s2.num_exact == 1
    rate, se = joint_rate(c2, 20000)
    assert abs(rate - c2.exact_distribution()[((True,), ())]) < 4 * se + 0.003


# ---------------------------------------------------------------------------------------------
# Interference between distant faults (2.1): a logical error at one time with the same logical
# at another is one kernel element, of weight 2d, which no local search finds
# ---------------------------------------------------------------------------------------------


def repetition_memory(d: int, rounds: int, theta: float) -> sq.Circuit:
    """A distance-d repetition memory with an over-rotation exp(−iθX) on every data qubit at
    each round's start and no other noise (the 2.1 audit's circuit at d = 3, 2 rounds)."""
    data = " ".join(str(2 * i) for i in range(d))
    lines = ["R " + " ".join(str(q) for q in range(2 * d - 1))]
    for r in range(rounds):
        lines += [
            f"I_ERROR[R_X(theta={theta!r})] {data}",
            "CX " + " ".join(f"{2 * i} {2 * i + 1}" for i in range(d - 1)),
            "CX " + " ".join(f"{2 * i + 2} {2 * i + 1}" for i in range(d - 1)),
            "MR " + " ".join(str(2 * i + 1) for i in range(d - 1)),
        ]
        lines += [f"DETECTOR rec[-{d - 1 - k}]" if r == 0 else f"DETECTOR rec[-{d - 1 - k}] rec[-{2 * d - 2 - k}]" for k in range(d - 1)]
    lines.append(f"M {data}")
    lines += [f"DETECTOR rec[-{d - k}] rec[-{d - k - 1}] rec[-{2 * d - 1 - k}]" for k in range(d - 1)]
    lines.append("OBSERVABLE_INCLUDE(0) rec[-1]")
    return sq.Circuit("\n".join(lines))


def surface_memory(rounds: int, theta: float) -> sq.Circuit:
    """Stim's d = 3 rotated surface-code X memory with an over-rotation exp(−iθZ) on every data
    qubit at each round's start (before the X ancillas' first Hadamard), no other noise."""
    lines = str(sq.Circuit.generated("surface_code:rotated_memory_x", distance=3, rounds=rounds)).splitlines()
    data = " ".join([line for line in lines if line.strip().startswith("MX ")][-1].split()[1:])
    out = []
    for i, line in enumerate(lines):
        if line.strip().startswith("H ") and i + 2 < len(lines) and lines[i + 1].strip() == "TICK" and lines[i + 2].strip().startswith("CX "):
            out.append(f"I_ERROR[R_Z(theta={theta!r})] {data}")
        out.append(line)
    return sq.Circuit("\n".join(out))


def silent_exact(c: sq.Circuit, rounds: int, theta: float) -> tuple[float, float]:
    """P(no detection event, observable kept) and P(no detection event, observable flipped) for
    a memory of these two kinds, by an independent state vector of its data qubits alone. The
    rotations sit before each round's gates and nothing else is noisy, so each round's checks
    of the rotated basis measure the parities of the faults on their supports exactly, and the
    other basis's checks commute with everything that is read: in the rotated basis's frame a
    round is exp(−iθX) on every data qubit and a projection onto even parities."""
    lines = [line.strip() for line in str(c).splitlines()]
    last = max(i for i, line in enumerate(lines) if line.startswith(("M ", "MX ")))
    n = len(lines[last].split()) - 1
    # The final detectors' and the observable's data records, as positions among the data.
    def support(line: str) -> list[int]:
        return [n - k for k in (int(t[5:-1]) for t in line.split() if t.startswith("rec[-")) if k <= n]
    checks = [support(line) for line in lines[last + 1 :] if line.startswith("DETECTOR")]
    logical = [support(line) for line in lines[last + 1 :] if line.startswith("OBSERVABLE_INCLUDE")][0]
    idx = np.arange(1 << n)
    parity = lambda qs: np.bitwise_xor.reduce([(idx >> q) & 1 for q in qs])
    even = np.all([parity(ch) == 0 for ch in checks], axis=0)
    flipped = parity(logical) == 1
    psi = np.zeros(1 << n, dtype=complex)
    psi[0] = 1.0
    for _ in range(rounds):
        for q in range(n):
            psi = np.cos(theta) * psi - 1j * np.sin(theta) * psi[idx ^ (1 << q)]
        psi = np.where(even, psi, 0.0)
    p = np.abs(psi) ** 2
    return float(p[~flipped].sum()), float(p[flipped].sum())


# Enough shots for a standard error of 0.0001 on the d = 3, three-round silent logical rate
# (0.000092 or so at 2,000,000; about 0.000082 here).
SILENT_SHOTS = 2_500_000


def silent_rates(c: sq.Circuit, shots: int, seed: int = 9):
    """The coherent sampler's P(no detection event, kept) and P(no detection event, flipped),
    with their standard errors, and the sampler."""
    s = c.compile_coherent_sampler(seed=seed)
    d, o, w = s.sample(shots, threads=0)
    quiet = ~d.any(axis=1)
    return sq.weighted_rate(quiet & ~o[:, 0], w), sq.weighted_rate(quiet & o[:, 0], w), s


@pytest.mark.parametrize("d, rounds, theta", [(3, 2, 0.3), (3, 3, 0.3), (5, 2, 0.5)])
def test_logical_errors_at_different_times_interfere(d, rounds, theta):
    """The audit's case and its kin: XXX in one round with XXX in another flips nothing, so the
    two give the same outcome and their amplitudes add, doubling the rate of a logical error
    with no detection event over the twirl's (0.00203 against 0.00101 at d = 3, 2 rounds; the
    2.0 sampler read the twirl's, 44σ off). The sampler's weights are now the whole kernel's."""
    c = repetition_memory(d, rounds, theta)
    assert silent_exact(c, rounds, theta)[1] == pytest.approx(sum(p for (dets, obs), p in c.exact_distribution().items() if not any(dets) and obs[0]), rel=1e-9)
    (kept, se_kept), (flipped, se_flipped), s = silent_rates(c, 1_000_000)
    assert s.kernel_complete and s.exact_weights and s.num_nonlocal_generators == rounds - 1 == s.num_generators
    exact_kept, exact_flipped = silent_exact(c, rounds, theta)
    assert abs(kept - exact_kept) < 5 * se_kept, (kept, exact_kept, se_kept)
    assert abs(flipped - exact_flipped) < 5 * se_flipped, (flipped, exact_flipped, se_flipped)
    # The twirl is far outside: the test sees the interference.
    twirl = sum(p for (dets, obs), p in c.twirled().exact_distribution().items() if not any(dets) and obs[0])
    assert abs(exact_flipped - twirl) > 8 * se_flipped
    # Every detector's and the observable's marginal, too.
    mean, se, _ = weighted_marginals(c, 200_000, seed=3)
    assert np.all(np.abs(mean - exact_marginals(c)) < 5 * se + 1e-9)


def test_decoded_logical_error_with_one_rotation_per_round():
    """The audit's decoded case: matching on the twirled model, d = 3, 2 rounds, θ = 0.2 once per
    round (the 2.0 sampler: 0.00907 ± 0.00009 against the exact 0.00950)."""
    c = repetition_memory(3, 2, 0.2)
    dec = sq.Matching.from_detector_error_model(c.twirled().detector_error_model(decompose_errors=True))
    dist = c.exact_distribution()
    keys = list(dist)
    fails = (dec.decode_batch(np.array([k[0] for k in keys], dtype=bool)) != np.array([k[1] for k in keys], dtype=bool)).any(axis=1)
    exact = float((np.array([dist[k] for k in keys]) * fails).sum())
    d, o, w = c.compile_coherent_sampler(seed=5).sample(400_000, threads=0)
    rate, se = sq.weighted_rate((dec.decode_batch(d) != o).any(axis=1), w)
    assert abs(rate - exact) < 5 * se, (rate, exact, se)


def test_surface_code_memory_with_one_rotation_per_round():
    """The d = 3 rotated surface code's X memory, exp(−iθZ) on every data qubit once per round:
    17 qubits, whose every branch no state vector here follows, against the data's own state
    vector. Two rounds have a kernel of nine generators, one of them non-local, summed whole
    (the 2.0 sampler: 0.00117 ± 0.00008 against the exact 0.00240, the twirl's 0.00144)."""
    theta = 0.25
    c = surface_memory(2, theta)
    exact_kept, exact_flipped = silent_exact(c, 2, theta)
    (kept, se_kept), (flipped, se_flipped), s = silent_rates(c, 300_000)
    assert s.kernel_complete and s.exact_weights and s.num_nonlocal_generators >= 1
    assert abs(kept - exact_kept) < 5 * se_kept, (kept, exact_kept, se_kept)
    assert abs(flipped - exact_flipped) < 5 * se_flipped, (flipped, exact_flipped, se_flipped)


def test_surface_d3_r3_silent_rate_is_exact():
    """Three rounds: fourteen generators, past the twelve 2.1 summed whole (its truncation read
    a silent logical error at 0.00161 ± 0.00013, 2.0's at 0.00100 ± 0.00007, the twirl's at
    0.00123). The class sums are contracted, three generators at a time besides the one summed
    out, and every shot sums its whole class: the rate is the exact 0.00258."""
    theta = 0.25
    c = surface_memory(3, theta)
    exact_kept, exact_flipped = silent_exact(c, 3, theta)
    assert exact_flipped == pytest.approx(0.00258, abs=5e-6)
    (kept, se_kept), (flipped, se_flipped), s = silent_rates(c, SILENT_SHOTS)
    assert s.kernel_complete and s.exact_weights and s.contraction_width == 3 and s.num_generators == 14
    assert se_flipped <= 0.0001, se_flipped
    assert abs(kept - exact_kept) < 5 * se_kept, (kept, exact_kept, se_kept)
    assert abs(flipped - exact_flipped) < 5 * se_flipped, (flipped, exact_flipped, se_flipped)


def test_max_contraction_zero_keeps_truncation():
    """With the contraction off, fourteen generators are truncated as in 2.1, and the sampler
    says so; its weights are 2.1's (seed 9's first 20,000 shots, from the 2.1.0 wheel). A kernel
    of twelve generators or fewer is still summed whole."""
    c = surface_memory(3, 0.25)
    with pytest.warns(UserWarning, match="max_contraction=0"):
        s = c.compile_coherent_sampler(seed=9, max_contraction=0)
    assert s.kernel_complete and not s.exact_weights and s.contraction_width is None
    _, _, w = s.sample(20_000, threads=1)
    assert w.sum() == pytest.approx(20213.884915594033, rel=1e-12)
    assert np.allclose(w[:6], [1.0974372803964274, 2.255544504808714, 1.0509262125690255, 0.19754302237313576, 2.077504971231143, 2.4156493000859047], rtol=1e-12, atol=0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        small = repetition_memory(3, 2, 0.3).compile_coherent_sampler(seed=1, max_contraction=0)
    assert small.exact_weights and small.contraction_width is None


def test_weights_independent_of_threads():
    """The contraction's weights are a function of the shot alone: the same on any number of
    threads, bit for bit."""
    c = surface_memory(3, 0.25)
    a = c.compile_coherent_sampler(seed=4).sample(20_000, threads=1)
    b = c.compile_coherent_sampler(seed=4).sample(20_000, threads=4)
    assert c.compile_coherent_sampler(seed=4).contraction_width == 3
    for x, y in zip(a, b):
        assert np.array_equal(x, y)
    assert len(np.unique(a[2])) > 100


def test_max_contraction_is_a_count_and_past_2_to_the_31_it_is_2_to_the_31():
    """max_contraction is a non-negative integer (0 turns the contraction off). A budget past
    2^31 table entries is 2^31, as the core takes it, and no table spans 32 generators or more at
    any budget: forty rounds of the d = 3 memory, whose 39 non-local generators all hold the
    first round's rotations, are refused however large the budget, and the warning names the
    budget the sampler had."""
    c = surface_memory(3, 0.25)
    for bad, error in [(1.5, TypeError), (True, TypeError), ("65536", TypeError), (-1, ValueError)]:
        with pytest.raises(error, match="max_contraction"):
            c.compile_coherent_sampler(seed=1, max_contraction=bad)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for huge in (2**31, 2**31 + 1, 2**64, 10**30, np.uint64(2**63)):
            s = c.compile_coherent_sampler(seed=1, max_contraction=huge)
            assert s.exact_weights and s.contraction_width == 3
    with pytest.warns(UserWarning, match=r"contract past max_contraction=2147483648 table entries"):
        s = surface_memory(40, 0.25).compile_coherent_sampler(seed=1, max_contraction=10**30)
    assert s.kernel_complete and s.num_nonlocal_generators == 39
    assert not s.exact_weights and s.contraction_width is None


def test_a_contraction_refused_by_a_bug_says_so():
    """The core refuses a complete kernel's contraction for its budget, or for a bug (which cannot
    happen: the products' phases are a quadratic form; debug builds panic). The warning tells them
    apart: a bug is an internal error to report, never advice to raise max_contraction."""
    from stabilizer_qec._circuit import _contraction_warning

    def core(internal, exact):
        return types.SimpleNamespace(contraction_internal_error=internal, exact_weights=exact, num_generators=14)

    why = "the products' phases are not a quadratic form (an odd cross term)"
    for exact, outcome in [(False, "so the weights are truncated (exact_weights is False)"), (True, "the weights are still exact")]:
        message = _contraction_warning(core(why, exact), 3, 65536)
        assert message.startswith("internal error in stabilizer-qec: the class sums' contraction was refused (" + why)
        assert outcome in message and "please report this circuit at https://" in message and "max_contraction" not in message
    budget = _contraction_warning(core(None, False), 3, 0)
    assert budget.startswith("the circuit's 14 interference generators contract past max_contraction=0 table entries")
    assert budget.endswith("raise max_contraction to sum the whole class")
    assert _contraction_warning(core(None, True), 3, 65536) is None
    # The core's own refusals: the budget's is no internal error.
    with pytest.warns(UserWarning, match="max_contraction=0"):
        s = surface_memory(3, 0.25).compile_coherent_sampler(seed=1, max_contraction=0)
    assert s._s.contraction_internal_error is None
    assert surface_memory(3, 0.25).compile_coherent_sampler(seed=1)._s.contraction_internal_error is None


def test_past_the_completion_limit_the_sampler_says_so():
    c = repetition_memory(3, 2, 0.3)
    with pytest.warns(UserWarning, match="interference between distant faults"):
        s = c.compile_coherent_sampler(seed=1, max_complete=5)
    assert not s.kernel_complete and s.num_nonlocal_generators == 0 and not s.exact_weights
    # The joint sampler (a T gate besides) completes its kernel and says when it does not.
    t = sq.Circuit(str(c) + "\nR 5\nH 5\nI[T] 5\nH 5\nM 5\nDETECTOR rec[-1]")
    assert t.compile_coherent_sampler(seed=1).kernel_complete
    with pytest.warns(UserWarning, match="interference between distant faults"):
        assert not t.compile_coherent_sampler(seed=1, max_complete=0).kernel_complete


def coherent_tool():
    """tools/coherent.py, which measures data/coherent/."""
    path = pathlib.Path(__file__).resolve().parent.parent / "tools" / "coherent.py"
    spec = importlib.util.spec_from_file_location("coherent_tool", path)
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    return tool


def sums(hits, w):
    return w.sum(), (w * w).sum(), w[hits].sum(), (w[hits] ** 2).sum()


def test_the_data_tool_stops_on_what_the_weights_are_worth():
    """tools/coherent.py samples a weighted rate until its error bar is as narrow as an unweighted
    run's when that run stops: 300 failures over at least 100,000 shots. For unit weights that is
    the plain rule; a heavy tail on the failures, as exact weights have, is worth far fewer
    failures than 2.1's count, Σ w·fail / mean w, read in it."""
    tool = coherent_tool()
    enough = lambda hits, w, **kw: tool.enough(*sums(hits, w), **{"min_failures": 300, "min_shots": 100_000, **kw})  # noqa: E731
    rng = np.random.default_rng(5)
    # Unit weights: the counts, and the rule stops where it did, to the shot.
    for n, f, stop in [(100_000, 300, True), (100_000, 299, False), (99_999, 300, False), (200_000, 300, True), (100_000, 100_000, True)]:
        hits, w = np.zeros(n, dtype=bool), np.ones(n)
        hits[rng.choice(n, f, replace=False)] = True
        p, se, shots = tool.worth(*sums(hits, w))
        assert (p, shots, p * shots) == pytest.approx((f / n, n, f), rel=1e-12)
        assert se == pytest.approx(np.sqrt(p * (1 - p) / n), rel=1e-12)
        assert enough(hits, w) is stop, (n, f)
    # Weighted: sq.weighted_rate's rate and error.
    hits = rng.random(100_000) < 0.2
    w = rng.lognormal(0, 1, 100_000)
    w[hits] = rng.lognormal(-4.5, 3, hits.sum())
    p, se, shots = tool.worth(*sums(hits, w))
    assert (p, se) == pytest.approx(sq.weighted_rate(hits, w), rel=1e-12)
    assert shots == pytest.approx(p * (1 - p) / se**2, rel=1e-12)
    # 2.1's count reads some 10,000 failures in these shots; their error bar is worth under 100.
    assert (w * hits).sum() / w.mean() > 3000
    assert p * shots < 150 and not enough(hits, w)
    # No failures yet: worth no failures, whatever the weights.
    assert not enough(np.zeros(100_000, dtype=bool), w, min_failures=1)


def test_the_data_tool_samples_until_its_rates_are_enough():
    """coherent_rates stops at the first batch where both decoders' rates are enough, and its rates
    and error bars are sq.weighted_rate's over every shot it drew."""
    tool = coherent_tool()
    c, _ = tool.capacity(3, 0.3)
    rule = {"min_failures": 200, "min_shots": 3_000}
    got = tool.coherent_rates(c, batch=1_000, max_shots=200_000, **rule)
    assert got["exact"] and 3_000 <= got["shots"] < 200_000
    sampler = c.compile_coherent_sampler(seed=1)
    batches = [sampler.sample(1_000) for _ in range(got["shots"] // 1_000)]
    w = np.concatenate([b[2] for b in batches])
    assert got["ess"] == pytest.approx(sq.effective_sample_size(w) / len(w), rel=1e-12)
    before = []
    for dec, key in [(tool.decoder(c.twirled()), ""), (tool.decoder(c.twirled(merge=True)), "_merged")]:
        hits = np.concatenate([(dec.decode_batch(d) != o).any(axis=1) for d, o, _ in batches])
        assert (got["rate" + key], got["stderr" + key]) == pytest.approx(sq.weighted_rate(hits, w), rel=1e-12)
        assert tool.enough(*sums(hits, w), **rule)
        before.append(tool.enough(*sums(hits[:-1_000], w[:-1_000]), **rule))
    # One batch fewer was not enough, for one decoder at least.
    assert not all(before)


DATA = pathlib.Path(__file__).resolve().parent.parent / "data" / "coherent"


def test_every_data_row_says_what_measured_it():
    """Each row of data/coherent/ has the version of the core that measured it and whether its
    weights were the whole class's (coherent.exact; validate's own exact is the exact answer).
    Past twelve generators only 2.2's contraction is exact; 2.1 listed the span of twelve or
    fewer (validate's repetition rows, stamped 2.0.0, the version its development core
    reported); 1.9's sampler (1.8.0) never summed the whole class."""
    tool = coherent_tool()
    release = lambda v: tuple(int(x) for x in v.split("."))  # noqa: E731
    for name in tool.SECTIONS:
        doc = json.loads((DATA / f"{name}.json").read_text())
        assert [{f: r[f] for f in tool.KEYS[name]} for r in doc["rows"]] == tool.grid(name)
        for r in doc["rows"]:
            version, exact = r["version"], r["coherent"]["exact"]
            assert isinstance(version, str) and isinstance(exact, bool), (name, r)
            if release(version) < (2, 0):
                assert not exact, (name, r)
            if exact and r["coherent"]["generators"] > 12:
                assert release(version) >= (2, 2) and r["coherent"]["width"] is not None, (name, r)
        # The file's own version is the core that last wrote it.
        assert release(doc["version"]) >= max(release(r["version"]) for r in doc["rows"])


def test_the_data_tool_stamps_every_row(tmp_path, monkeypatch):
    """save() stamps a row it measured with this core's version (2.2's development core, which
    reports 2.1.0, as 2.2.0) and keeps a kept row's own stamp, or the file's for a row without
    one; a row that does not say whether its weights were exact is refused."""
    tool = coherent_tool()
    monkeypatch.setattr(tool, "OUT", tmp_path)
    release = tuple(int(x) for x in sq.__version__.split("+")[0].split(".")[:2])
    now = tool.measured_by()
    assert now == ("2.2.0" if release < (2, 2) else sq.__version__)
    row = lambda d, exact, **stamp: {"d": d, "theta": 0.2, "coherent": {"exact": exact}, **stamp}  # noqa: E731
    tool.KEPT["capacity"] = {"version": "1.8.0", "rows": [row(3, False), row(5, False, version="2.0.0"), row(7, False)]}
    tool.save("capacity", {"rows": [row(7, True)]})
    doc = json.loads((tmp_path / "capacity.json").read_text())
    assert doc["version"] == now
    assert [(r["d"], r["version"], r["coherent"]["exact"]) for r in doc["rows"]] == [(3, "1.8.0", False), (5, "2.0.0", False), (7, now, True)]
    tool.KEPT.clear()
    tool.save("capacity", {"rows": [row(3, True), row(5, False, version="2.0.0")]})
    doc = json.loads((tmp_path / "capacity.json").read_text())
    assert [(r["version"], r["coherent"]["exact"]) for r in doc["rows"]] == [(now, True), ("2.0.0", False)]
    with pytest.raises(ValueError, match=r"capacity row d=3, theta=0.2 does not say whether its weights were exact"):
        tool.save("capacity", {"rows": [{"d": 3, "theta": 0.2, "coherent": {}}]})


def test_the_data_tool_refuses_a_bad_only_before_measuring(tmp_path, monkeypatch, capsys):
    """--only names rows by their fields. A field a section's rows lack, a value without '=', a
    value no row has and a row the file lacks are refused with the reason and exit status 2,
    before any row is measured (the tool raised a KeyError, a bare ValueError or a
    StopIteration, some after measuring other sections)."""
    tool = coherent_tool()
    monkeypatch.setattr(tool, "OUT", tmp_path)
    monkeypatch.setenv("PYTHONUNBUFFERED", "1")
    for name in tool.SECTIONS:
        shutil.copy(DATA / f"{name}.json", tmp_path / f"{name}.json")
    measured = []
    for section in ("validate", "code_capacity", "sweep"):
        monkeypatch.setattr(tool, section, lambda *a, **k: measured.append(a))

    def refused(*argv):
        assert tool.main(list(argv)) == 2
        err = capsys.readouterr().err
        assert err.startswith("tools/coherent.py: error: ") and err.count("\n") == 1, err
        return err

    assert "sweep rows have no field 'code' (they are named by d, rounds, theta)" in refused("all", "--only", "code=surface")
    assert "capacity rows have no field 'rounds'" in refused("capacity", "--only", "rounds=3")
    assert "--only d3: expected field=value[,value...]" in refused("sweep", "--only", "d3")
    for spec in ("d=", "=3", "d=3,"):
        assert f"--only {spec}: expected field=value" in refused("sweep", "--only", spec)
    assert "--only needs field=value" in refused("sweep", "--only")
    assert "--only d=4: no sweep row has d=4 (they have 3, 5, 7, 9)" in refused("sweep", "--only", "d=3,4")
    assert "--only theta=0.30: no capacity row has theta=0.30 (they have 0.2, 0.25, 0.3, 0.35)" in refused("capacity", "--only", "theta=0.30")
    assert "unknown section 'sweeps'" in refused("sweeps")
    assert "one section at a time (sweep, then capacity)" in refused("sweep", "capacity")
    assert "unknown option --fast" in refused("sweep", "--fast")
    doc = json.loads((tmp_path / "sweep.json").read_text())
    doc["rows"] = [r for r in doc["rows"] if (r["d"], r["theta"]) != (5, 0.0025)]
    (tmp_path / "sweep.json").write_text(json.dumps(doc))
    assert f"{tmp_path / 'sweep.json'} has no row d=5, rounds=5, theta=0.0025 to keep" in refused("sweep", "--only", "d=3")
    doc = json.loads((tmp_path / "crosstalk.json").read_text())
    del doc["rows"][-1]["coherent"]["exact"]
    (tmp_path / "crosstalk.json").write_text(json.dumps(doc))
    assert "row d=7, rounds=7, theta=0.02 does not say whether its weights were exact" in refused("crosstalk", "--only", "d=3")
    (tmp_path / "validate.json").unlink()
    assert "which does not exist" in refused("validate", "--only", "code=surface")
    assert measured == []
    # A good --only gets as far as measuring (here, nothing).
    assert tool.main(["capacity", "--only", "d=15", "--only", "theta=0.35"]) == 0
    assert measured == [()] and tool.ONLY == {"d": {"15"}, "theta": {"0.35"}}
    assert [k for k in tool.grid("capacity") if tool.wanted(k)] == [{"d": 15, "theta": 0.35}]
    # From the command line: the message and status 2.
    path = pathlib.Path(__file__).resolve().parent.parent / "tools" / "coherent.py"
    run = subprocess.run([sys.executable, str(path), "all", "--only", "code=surface"], capture_output=True, text=True)
    assert run.returncode == 2 and run.stdout == "", run
    assert run.stderr == "tools/coherent.py: error: --only code=...: sweep rows have no field 'code' (they are named by d, rounds, theta); run only the sections whose rows have it\n", run.stderr

"""The stabilizer-rank sampler and Tsim's circuit surface against Tsim itself (bloqade-tsim),
where it is installed: distributions within error bars, `to_matrix` up to a global phase,
`tcount` and `is_clifford` equal."""

from __future__ import annotations

import numpy as np
import pytest

import stabilizer_qec as sq
from stabilizer_qec import tsim as ours_tsim

tsim = pytest.importorskip("tsim")

THETA = -np.arccos(np.sqrt(1 / 3)) / np.pi


def equal_up_to_phase(u: np.ndarray, v: np.ndarray) -> bool:
    product = u @ v.conj().T
    phase = product.flat[np.argmax(np.abs(product))]
    return abs(abs(phase) - 1) < 1e-6 and np.allclose(product, phase * np.eye(u.shape[0]), atol=1e-6)


def histogram(samples: np.ndarray) -> np.ndarray:
    idx = (samples.astype(np.int64) << np.arange(samples.shape[1])).sum(axis=1)
    return np.bincount(idx, minlength=1 << samples.shape[1]).astype(float)


def chi_square_within(counts: np.ndarray, probs: np.ndarray, shots: int) -> None:
    """Tsim's counts against exact probabilities: Pearson's statistic within 5 standard
    deviations of its mean (bins below 5 expected pooled)."""
    expected = probs * shots
    keep = expected >= 5
    o = np.append(counts[keep], counts[~keep].sum())
    e = np.append(expected[keep], expected[~keep].sum())
    o, e = o[e > 0], e[e > 0]
    stat = ((o - e) ** 2 / e).sum()
    dof = max(len(e) - 1, 1)
    assert stat < dof + 5 * np.sqrt(2 * dof) + 5, (stat, dof)


def random_shorthand(rng, n: int, length: int) -> str:
    gates1 = ["H", "S", "S_DAG", "X", "SQRT_X", "C_XYZ", "T", "T_DAG"]
    lines = ["R " + " ".join(map(str, range(n)))]
    for _ in range(length):
        q = int(rng.integers(n))
        u = rng.random()
        if u < 0.4:
            lines.append(f"{rng.choice(gates1)} {q}")
        elif u < 0.6 and n > 1:
            a, b = rng.choice(n, 2, replace=False)
            lines.append(f"{rng.choice(['CX', 'CZ', 'CNOT', 'SWAP'])} {a} {b}")
        elif u < 0.75:
            lines.append(f"R_{rng.choice(list('XYZ'))}({rng.uniform(-1, 1):.4f}) {q}")
        elif u < 0.85:
            lines.append(f"U3({rng.uniform(0, 2):.4f}, {rng.uniform(0, 2):.4f}, {rng.uniform(0, 2):.4f}) {q}")
        elif n > 1:
            a, b = rng.choice(n, 2, replace=False)
            lines.append(f"{rng.choice(['TPP', 'TPP_DAG'])} {rng.choice(list('XYZ'))}{a}*{rng.choice(list('XYZ'))}{b}")
        else:
            lines.append(f"TPP X{q}")
    lines.append("M " + " ".join(map(str, range(n))))
    return "\n".join(lines)


def test_magic_state_infidelity():
    text = f"R 0\nR_X({THETA}) 0\nT_DAG 0\nDEPOLARIZE1(0.05) 0\nT 0\nR_X({-THETA}) 0\nM 0\nDETECTOR rec[-1]"
    exact = ours_tsim.Circuit(text).exact_distribution(method="stabilizer_rank")[((True,), ())]
    shots = 20_000
    theirs = tsim.Circuit(text).compile_sampler(seed=0).sample(shots, batch_size=shots).mean()
    assert abs(theirs - exact) < 5 * np.sqrt(exact * (1 - exact) / shots)


def test_distillation_records():
    text = (
        f"R 0 1 2 3 4\nR_X({THETA}) 0 1 2 3 4\nT_DAG 0 1 2 3 4\nDEPOLARIZE1(0.05) 0 1 2 3 4\n"
        "SQRT_X 0 1 4\nCZ 0 1 2 3\nSQRT_Y 0 3\nCZ 0 2 3 4\nTICK\nSQRT_X_DAG 0\nCZ 0 4 1 3\nTICK\nSQRT_X_DAG 0 1 2 3 4\n"
        f"T 0\nR_X({-THETA}) 0\nM 0 1 2 3 4"
    )
    ours = ours_tsim.Circuit(text)
    sampler = ours.compile_stabilizer_rank_sampler(seed=0)
    records = np.array([[(i >> k) & 1 for k in range(5)] for i in range(32)], dtype=bool)
    # With noise, probability_of averages over its draws; the exact probabilities come from
    # the distribution over every branch (five depolarizing channels: 4^5 of them).
    dist = sq.Circuit(str(ours) + "\n" + "\n".join(f"DETECTOR rec[-{k}]" for k in range(5, 0, -1))).exact_distribution(method="stabilizer_rank")
    probs = np.array([dist.get((tuple(bool(b) for b in r), ()), 0.0) for r in records])
    assert abs(probs.sum() - 1) < 1e-9
    shots = 20_000
    theirs = tsim.Circuit(text).compile_sampler(seed=1).sample(shots, batch_size=shots)
    chi_square_within(histogram(theirs), probs, shots)
    # Our own sampler follows them too, and the sampled probability_of agrees with the exact.
    o = sampler.sample_measurements(shots, threads=0)
    chi_square_within(histogram(o), probs, shots)
    est = sampler.probability_of(records[:4], samples=4000, seed=2)
    assert np.all(np.abs(est - probs[:4]) < 0.01)


@pytest.mark.parametrize("seed", range(4))
def test_random_shorthand_circuits(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(1, 4))
    text = random_shorthand(rng, n, int(rng.integers(5, 10)))
    ours = ours_tsim.Circuit(text)
    theirs = tsim.Circuit(text)
    assert ours.tcount() == theirs.tcount(), text
    assert ours.is_clifford == theirs.is_clifford, text
    records = np.array([[(i >> k) & 1 for k in range(n)] for i in range(1 << n)], dtype=bool)
    probs = ours.compile_stabilizer_rank_sampler().probability_of(records)
    assert abs(probs.sum() - 1) < 1e-9, text
    shots = 10_000
    samples = theirs.compile_sampler(seed=seed).sample(shots, batch_size=shots)
    chi_square_within(histogram(samples), probs, shots)


@pytest.mark.parametrize("text", ["T 0", "TPP X0*Z1", "U3(0.3, 0.2, 0.1) 0", "R_PAULI(0.345) X0*Y1", "CCZ 0 1 2", "CCX 0 1 2", "H 0\nCNOT 0 1\nS 1\nT 0\nR_XX(0.3) 0 1\nT_DAG 1"])
def test_matrices_tcount_and_inverse_agree(text):
    ours, theirs = ours_tsim.Circuit(text), tsim.Circuit(text)
    assert equal_up_to_phase(ours.to_matrix(), np.asarray(theirs.to_matrix()))
    assert ours.tcount() == theirs.tcount()
    assert ours.is_clifford == theirs.is_clifford
    inv = theirs.inverse()
    assert equal_up_to_phase(ours.inverse().to_matrix(), np.asarray(inv.to_matrix()))


def test_detector_sampler_on_a_noisy_circuit():
    text = (
        "R 0 1 2\nH 0\nT 0\nCX 0 1\nDEPOLARIZE1(0.05) 0 1\nT_DAG 1\nCX 1 2\nX_ERROR(0.1) 2\nH 0\nM 0 1 2\n"
        "DETECTOR rec[-1] rec[-2]\nDETECTOR rec[-2] rec[-3]\nOBSERVABLE_INCLUDE(0) rec[-1]"
    )
    shots = 20_000
    d_o, l_o = ours_tsim.Circuit(text).compile_detector_sampler(seed=0).sample(shots, separate_observables=True, threads=0)
    d_t, l_t = tsim.Circuit(text).compile_detector_sampler(seed=0).sample(shots, batch_size=shots, separate_observables=True)
    a, b = np.hstack([d_o, l_o]).mean(0), np.hstack([d_t, l_t]).mean(0)
    z = np.abs(a - b) / np.sqrt(a * (1 - a) / shots + b * (1 - b) / shots + 1e-12)
    assert np.all(z < 5), (a, b)

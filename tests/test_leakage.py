"""Leakage, as Stim-readable tags: the frame sampler's model against the same model in the
state vector, followed branch by branch."""

from __future__ import annotations

import numpy as np
import pytest

import stabilizer_qec as sq

CASES = [
    # Leak, then a CZ with the leaked partner, then measure both.
    "R 0 1\nH 1\nI_ERROR[LEAK(p=0.3)] 0\nCZ 0 1\nH 1\nM 0 1\nDETECTOR rec[-1]\nDETECTOR rec[-2]\nOBSERVABLE_INCLUDE(0) rec[-2]",
    # Leak, seep back, then a Bell measurement.
    "R 0 1\nI_ERROR[LEAK(p=0.4)] 0\nI_ERROR[SEEP(p=0.5)] 0\nH 0\nCX 0 1\nCX 0 1\nH 0\nM 0 1\nDETECTOR rec[-1]\nDETECTOR rec[-2]",
    # Transport, reset clearing leakage, measurement flips.
    "R 0 1 2\nI_ERROR[LEAK(p=0.5)] 1\nII_ERROR[LEAK_TRANSPORT(p=0.5)] 1 2\nCX 2 0\nMR(0.05) 1\nX_ERROR(0.1) 0\nM 0 1 2\nDETECTOR rec[-3]\nDETECTOR rec[-2]\nDETECTOR rec[-1]\nDETECTOR rec[-4]",
    # A repetition-code round with leakage on the data.
    "R 0 1 2 3 4\nI_ERROR[LEAK(p=0.1)] 0 2 4\nCX 0 1 2 3\nCX 2 1 4 3\nMR 1 3\nDETECTOR rec[-1]\nDETECTOR rec[-2]\nM 0 2 4\nDETECTOR rec[-1] rec[-2] rec[-4]\nDETECTOR rec[-2] rec[-3] rec[-5]\nOBSERVABLE_INCLUDE(0) rec[-1]",
]


def marginals_and_pairs(rows: np.ndarray):
    rows = rows.astype(float)
    return np.concatenate([rows.mean(0), (rows[:, :, None] * rows[:, None, :]).mean(0)[np.triu_indices(rows.shape[1], 1)]])


@pytest.mark.parametrize("text", CASES)
@pytest.mark.parametrize("reads_one", [True, False])
def test_frames_agree_with_the_state_vector(text, reads_one):
    c = sq.Circuit(text)
    shots = 100_000
    d, o, h = c.compile_leakage_sampler(seed=1, leaked_reads_one=reads_one).sample(shots, threads=0)
    ours = marginals_and_pairs(np.concatenate([d, o], axis=1))
    # The state vector's exact distribution of the same model.
    rows, probs = [], []
    for dets, obs, p in _exact(c, reads_one):
        rows.append(list(dets) + [bool(obs >> k & 1) for k in range(c.num_observables)])
        probs.append(p)
    rows, probs = np.array(rows, dtype=float), np.array(probs)
    exact = np.concatenate([probs @ rows, np.einsum("s,si,sj->ij", probs, rows, rows)[np.triu_indices(rows.shape[1], 1)]])
    sigma = np.sqrt(exact * (1 - exact) / shots) + 1e-9
    assert np.all(np.abs(ours - exact) < 5 * sigma), (ours, exact)


def _exact(c: sq.Circuit, reads_one: bool):
    """The state vector's (detection events, observable flips, probability) under the model."""
    return c._c.exact_distribution_leaky(1 << 20, reads_one)


def test_heralds_mark_leaked_measurements():
    c = sq.Circuit("R 0\nI_ERROR[LEAK(p=1)] 0\nM 0\nR 0\nM 0")
    d, o, h = c.compile_leakage_sampler(seed=2).sample(10)
    assert h.shape == (10, 2) and h[:, 0].all() and not h[:, 1].any()


def test_circuits_without_leakage_have_no_heralds():
    c = sq.Circuit("R 0\nX_ERROR(0.1) 0\nM 0\nDETECTOR rec[-1]")
    d, o, h = c.compile_leakage_sampler(seed=2).sample(10)
    assert h.shape == (10, 0)

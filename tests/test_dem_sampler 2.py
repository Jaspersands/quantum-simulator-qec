"""Sampling a detector error model directly, as Stim's ``CompiledDemSampler``: the same rates
and correlations as Stim's sampler, faults replayed to the same shots, and a seed's shots the
same on any number of threads."""

from __future__ import annotations

import numpy as np
import pytest

import stabilizer_qec as sq


def model():
    c = sq.Circuit.generated("surface_code:rotated_memory_z", distance=5, rounds=5, after_clifford_depolarization=0.004, before_measure_flip_probability=0.004)
    return c.detector_error_model(decompose_errors=True)


def test_rates_and_pairs_match_stims():
    stim = pytest.importorskip("stim")
    dem = model()
    shots = 200_000
    d, o, _ = dem.compile_sampler(seed=1).sample(shots)
    sd, so, _ = stim.DetectorErrorModel(str(dem)).compile_sampler(seed=1).sample(shots)
    for ours, theirs in ((d, sd), (o, so)):
        p, q = ours.mean(axis=0), theirs.mean(axis=0)
        sigma = np.sqrt((p * (1 - p) + q * (1 - q)) / shots) + 1e-9
        assert (np.abs(p - q) / sigma).max() < 5.5
    # Pairwise: how often neighbouring detectors fire together.
    pairs = lambda a: (a[:, :-1] & a[:, 1:]).mean(axis=0)  # noqa: E731
    p, q = pairs(d), pairs(sd)
    sigma = np.sqrt((p + q) / shots) + 1e-9
    assert (np.abs(p - q) / sigma).max() < 5.5


def test_errors_replay_and_seeds():
    dem = model()
    s = dem.compile_sampler(seed=9)
    d, o, e = s.sample(1000, return_errors=True)
    assert e.shape == (1000, dem.num_errors) and e.dtype == bool
    rd, ro, re = s.sample(1000, recorded_errors_to_replay=e, return_errors=True)
    assert np.array_equal(rd, d) and np.array_equal(ro, o) and np.array_equal(re, e)
    a = dem.compile_sampler(seed=4).sample(500, threads=1, bit_packed=True)
    b = dem.compile_sampler(seed=4).sample(500, threads=4, bit_packed=True)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]) and a[2] is None
    with pytest.raises(TypeError):
        import pickle

        pickle.dumps(s)

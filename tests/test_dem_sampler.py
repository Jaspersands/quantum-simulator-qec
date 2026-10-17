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


def test_replay_checks_its_shots_and_clears_padding():
    d = sq.DetectorErrorModel("error(0) D0\nerror(0) D1\nerror(0) D2")
    det, _, err = d.compile_sampler(seed=1).sample(1, bit_packed=True, return_errors=True, recorded_errors_to_replay=np.array([[0xFF]], dtype=np.uint8))
    assert det.tolist() == [[7]] and err.tolist() == [[7]]
    rows = np.array([[0, 1, 0], [1, 0, 0]], dtype=bool)
    with pytest.raises(ValueError, match="shots"):
        d.compile_sampler().sample(3, recorded_errors_to_replay=rows)
    assert d.compile_sampler().sample(2, recorded_errors_to_replay=rows)[0].shape == (2, 3)


def test_detector_coordinates_skip_long_loops():
    stim = pytest.importorskip("stim")
    text = "detector(1, 0.5) D3\ndetector(0, 0.6) D3\nrepeat 3 {\n shift_detectors(0.1) 1\n detector(5, 0.25) D0\n}\nrepeat 2 {\n repeat 4 {\n  shift_detectors(0.1) 1\n  detector(0, 0.598) D0\n }\n}"
    assert sq.DetectorErrorModel(text).get_detector_coordinates() == stim.DetectorErrorModel(text).get_detector_coordinates()
    huge = sq.DetectorErrorModel("repeat 1000000000000 {\n error(0.1) D0 D1\n shift_detectors(0, 1) 1\n detector(1, 2) D0\n}")
    assert huge.get_detector_coordinates([0, 5, 999999999999]) == {0: [], 5: [1.0, 7.0], 999999999999: [1.0, 1000000000001.0]}


def test_dem_text_is_read_as_stim_reads_it():
    assert str(sq.DetectorErrorModel("error(0.5) d0 l1\ndetector(1) d3")) == "error(0.5) D0 L1\ndetector(1) D3"
    for bad in ("detector D0 D1", "logical_observable", "error (0.1) D0", "error(0.1)D0"):
        with pytest.raises(ValueError):
            sq.DetectorErrorModel(bad)
    with pytest.raises(ValueError, match="Separator"):
        sq.DemTarget("^").val

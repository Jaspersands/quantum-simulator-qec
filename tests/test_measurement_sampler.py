"""Raw measurement records, as Stim's ``compile_sampler`` gives them: exactly Stim's where the
circuit is deterministic, the same rates where it is not, and records that the converter turns
into the detector sampler's statistics."""

from __future__ import annotations

import numpy as np
import pytest

import stabilizer_qec as sq

stim = pytest.importorskip("stim")


@pytest.mark.parametrize(
    "text",
    [
        "X 0\nM 0 1\nH 2\nMX 2\nH 3\nZ 3\nMX 3",
        "R 0 1\nX 0\nCX 0 1\nM 0 1\nH_YZ 2\nMY 2\nMPP Z0*Z1 !Z1",
        "X 0\nREPEAT 3 {\n  CX 0 1\n  MR 1\n}\nM !0",
    ],
)
def test_deterministic_records_are_stims(text):
    ours = sq.Circuit(text).compile_sampler(seed=1).sample(70)
    theirs = stim.Circuit(text).compile_sampler(seed=1).sample(70)
    assert np.array_equal(ours, theirs)
    assert not sq.Circuit(text).compile_sampler(skip_reference_sample=True, seed=1).sample(70).any()


def test_random_measurements_are_coin_flips():
    rows = sq.Circuit("MX 0\nMY 1\nMPP X0*Z2").compile_sampler(seed=5).sample(20_000)
    assert np.all(np.abs(rows.mean(axis=0) - 0.5) < 0.02)


def test_noisy_rates_match_stims():
    c = stim.Circuit.generated("surface_code:rotated_memory_x", distance=3, rounds=3, after_clifford_depolarization=0.01, before_measure_flip_probability=0.01)
    shots = 100_000
    ours = sq.Circuit(str(c)).compile_sampler(seed=3).sample(shots, threads=4)
    theirs = c.compile_sampler(seed=3).sample(shots)
    p, q = ours.mean(axis=0), theirs.mean(axis=0)
    sigma = np.sqrt((p * (1 - p) + q * (1 - q)) / shots) + 1e-9
    assert (np.abs(p - q) / sigma).max() < 5.5
    # Through the converter, the records give the detector sampler's rates.
    dets = sq.Circuit(str(c)).compile_m2d_converter().convert(measurements=ours)
    direct = sq.Circuit(str(c)).compile_detector_sampler(seed=4).sample(shots)
    p, q = dets.mean(axis=0), direct.mean(axis=0)
    sigma = np.sqrt((p * (1 - p) + q * (1 - q)) / shots) + 1e-9
    assert (np.abs(p - q) / sigma).max() < 5.5


def test_seeds_and_threads():
    c = sq.Circuit.generated("repetition_code:memory", distance=5, rounds=4, after_clifford_depolarization=0.05)
    a = c.compile_sampler(seed=8).sample(300, threads=1, bit_packed=True)
    b = c.compile_sampler(seed=8).sample(300, threads=3, bit_packed=True)
    assert np.array_equal(a, b) and a.shape == (300, (c.num_measurements + 7) // 8)

from __future__ import annotations

import numpy as np
import pytest

import stabilizer_qec as sq


def test_shapes_and_forms(d3):
    s = d3.compile_detector_sampler(seed=5)
    nd, no = d3.num_detectors, d3.num_observables
    dets = s.sample(100)
    assert dets.shape == (100, nd) and dets.dtype == np.bool_
    sampler = lambda: d3.compile_detector_sampler(seed=5)  # noqa: E731
    d, o = sampler().sample(100, separate_observables=True)
    assert np.array_equal(d, dets) and o.shape == (100, no)
    both = sampler().sample(100, append_observables=True)
    assert np.array_equal(both, np.concatenate([d, o], axis=1))
    packed = sampler().sample(100, bit_packed=True)
    assert packed.dtype == np.uint8 and packed.shape == (100, (nd + 7) // 8)
    assert np.array_equal(np.unpackbits(packed, axis=1, count=nd, bitorder="little").astype(bool), dets)
    pd, po = sampler().sample(100, separate_observables=True, bit_packed=True)
    assert np.array_equal(np.unpackbits(po, axis=1, count=no, bitorder="little").astype(bool), o)
    assert s.sample(0).shape == (0, nd)
    with pytest.raises(ValueError):
        s.sample(10, separate_observables=True, append_observables=True)


def test_a_seed_gives_the_same_shots_on_any_number_of_threads(d3):
    one = d3.compile_detector_sampler(seed=42).sample(5000, threads=1)
    for threads in (2, 7, 0):
        assert np.array_equal(d3.compile_detector_sampler(seed=42).sample(5000, threads=threads), one)
    assert not np.array_equal(d3.compile_detector_sampler(seed=43).sample(5000), one)


def test_calls_continue_the_stream(d3):
    whole = d3.compile_detector_sampler(seed=3).sample(256)
    s = d3.compile_detector_sampler(seed=3)
    assert np.array_equal(np.concatenate([s.sample(64), s.sample(128), s.sample(64)]), whole)


def test_no_seed_draws_one(d3):
    a = d3.compile_detector_sampler().sample(640)
    b = d3.compile_detector_sampler().sample(640)
    assert not np.array_equal(a, b)


def test_bad_arguments(d3):
    with pytest.raises(ValueError):
        d3.compile_detector_sampler(seed=-1)
    with pytest.raises(ValueError):
        d3.compile_detector_sampler(seed=2**64)
    with pytest.raises(TypeError):
        d3.compile_detector_sampler(seed=1.5)
    s = d3.compile_detector_sampler(seed=1)
    with pytest.raises(ValueError):
        s.sample(-1)
    with pytest.raises(TypeError):
        s.sample(10, threads="all")


def test_noiseless_shots_are_silent():
    c = sq.memory_circuit(distance=3, rounds=3, p=0.0)
    d, o = c.compile_detector_sampler(seed=1).sample(1000, separate_observables=True)
    assert not d.any() and not o.any()


def test_detection_rates_match_stims(d5):
    stim = pytest.importorskip("stim")
    shots = 40_000
    ours = d5.compile_detector_sampler(seed=11).sample(shots, threads=0).mean(axis=0)
    theirs = stim.Circuit(str(d5)).compile_detector_sampler(seed=11).sample(shots).mean(axis=0)
    sigma = np.sqrt((ours * (1 - ours) + theirs * (1 - theirs)) / shots) + 1e-12
    assert np.abs((ours - theirs) / sigma).max() < 5.0

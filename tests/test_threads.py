"""Objects shared between threads decode as they do alone. Under free-threaded Python (3.14t)
the threads run at once; elsewhere this checks the GIL is released and retaken cleanly."""

from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import stabilizer_qec as sq


def test_the_module_keeps_the_gil_off():
    is_enabled = getattr(sys, "_is_gil_enabled", None)
    if is_enabled is not None and sysconfig_free_threaded():
        assert not is_enabled(), "importing stabilizer_qec turned the GIL back on"


def sysconfig_free_threaded() -> bool:
    import sysconfig

    return bool(sysconfig.get_config_var("Py_GIL_DISABLED"))


def test_decoders_shared_between_threads():
    c = sq.memory_circuit(distance=5, rounds=5, p=0.004)
    dem = c.detector_error_model(decompose_errors=True)
    dets = c.compile_detector_sampler(seed=1).sample(4096)
    chunks = np.array_split(dets, 16)
    decoders = [sq.Matching(dem), sq.Matching(dem, enable_correlations=True), sq.BeliefMatching(dem, max_bp_iters=5)]
    for d in decoders:
        alone = [d.decode_batch(x) for x in chunks]
        with ThreadPoolExecutor(8) as pool:
            together = list(pool.map(d.decode_batch, chunks))
        assert all(np.array_equal(a, b) for a, b in zip(alone, together))


def test_circuits_models_and_samplers_from_many_threads():
    def work(seed):
        c = sq.memory_circuit(distance=3, rounds=3, p=0.01)
        dem = c.detector_error_model(decompose_errors=True)
        dets, obs = c.compile_detector_sampler(seed=seed).sample(512, separate_observables=True)
        return int((sq.Matching(dem).decode_batch(dets) != obs).any(axis=1).sum())

    alone = [work(s) for s in range(12)]
    with ThreadPoolExecutor(6) as pool:
        assert list(pool.map(work, range(12))) == alone


def test_one_sampler_used_by_many_threads_stays_consistent():
    # Shots drawn from one sampler by several threads at once: each call gets whole shots, and
    # together they are the stream's shots in some order.
    c = sq.memory_circuit(distance=3, rounds=3, p=0.01)
    s = c.compile_detector_sampler(seed=3)
    with ThreadPoolExecutor(4) as pool:
        parts = list(pool.map(lambda _: s.sample(64, bit_packed=True), range(8)))
    got = sorted(map(bytes, (row for part in parts for row in part)))
    want = sorted(map(bytes, c.compile_detector_sampler(seed=3).sample(512, bit_packed=True)))
    assert got == want

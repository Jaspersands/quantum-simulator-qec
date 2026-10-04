"""Window decoding of models too long to unroll, from a template of their loop."""

from __future__ import annotations

import time

import numpy as np
import pytest

import stabilizer_qec as sq

stim = pytest.importorskip("stim")


def generated(code, d, rounds, p=0.004):
    return stim.Circuit.generated(code, distance=d, rounds=rounds, after_clifford_depolarization=p, before_round_data_depolarization=p, before_measure_flip_probability=p, after_reset_flip_probability=p)


@pytest.mark.parametrize("code", ["surface_code:rotated_memory_z", "surface_code:rotated_memory_x", "repetition_code:memory", "surface_code:unrotated_memory_z"])
@pytest.mark.parametrize("mode", ["sliding", "parallel"])
def test_a_template_decodes_as_the_whole_model(code, mode):
    c = sq.Circuit(str(generated(code, 3, 80)))
    dem = c.detector_error_model(decompose_errors=True)
    dets = c.compile_detector_sampler(seed=7).sample(256)
    for correlations in (False, True):
        whole = sq.WindowMatching(dem, commit=3, buffer=3, mode=mode, enable_correlations=correlations, template=False)
        streamed = sq.WindowMatching(dem, commit=3, buffer=3, mode=mode, enable_correlations=correlations, template=True)
        assert streamed.streamed and not whole.streamed
        assert streamed.windows == whole.windows
        assert np.array_equal(streamed.decode_batch(dets, threads=0), whole.decode_batch(dets, threads=0)), (code, mode, correlations)


def test_a_memory_too_long_to_unroll_decodes_from_its_folded_model():
    c = sq.Circuit(str(generated("surface_code:rotated_memory_z", 11, 10_000, p=0.001)))
    t = time.perf_counter()
    dem = c.detector_error_model(decompose_errors=True)
    decoder = sq.WindowMatching(dem, commit=5, buffer=5)  # 27 million faults: the template, by default
    assert decoder.streamed and time.perf_counter() - t < 30
    dets, obs = c.compile_detector_sampler(seed=3).sample(64, separate_observables=True, threads=0)
    pred = decoder.decode_batch(dets, threads=0)
    # 10,000 rounds at p = 0.1%, d = 11: logical failures are rare.
    assert (pred != obs).any(axis=1).sum() <= 2


def test_templates_refuse_what_they_cannot_serve():
    # No loop to take a template from.
    dem = sq.memory_circuit(distance=3, rounds=10, p=0.01).detector_error_model(decompose_errors=True)
    with pytest.raises(ValueError, match="no top-level loop"):
        sq.WindowMatching(dem, commit=2, buffer=2, template=True)
    assert not sq.WindowMatching(dem, commit=2, buffer=2).streamed
    with pytest.raises(TypeError):
        sq.WindowMatching(dem, commit=2, buffer=2, template="yes")


def test_any_circuit_streams():
    c = sq.Circuit(str(generated("surface_code:rotated_memory_x", 3, 3000, p=0.001)))
    for mode in ("sliding", "parallel"):
        r = sq.stream_memory(circuit=c, commit=3, buffer=3, mode=mode, shots=128, seed=5)
        # 3,000 rounds of d = 3 at p = 0.1%: global matching fails about 39% of them.
        assert r.shots == 128 and 20 < r.failures < 75 and len(r.windows) > 600
        again = sq.stream_memory(circuit=c, commit=3, buffer=3, mode=mode, shots=128, seed=5, threads=1)
        assert again.failures == r.failures  # the seed decides, not the threads
    rep = sq.Circuit(str(generated("repetition_code:memory", 5, 2000, p=0.01)))
    assert sq.stream_memory(circuit=rep, commit=4, buffer=4, shots=64, enable_correlations=True).shots == 64
    with pytest.raises(ValueError, match="not both"):
        sq.stream_memory(circuit=c, distance=3, commit=3, buffer=3)
    with pytest.raises(TypeError, match="needs distance"):
        sq.stream_memory(commit=3, buffer=3)

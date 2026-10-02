"""Circuits, models, converters and decoders pickle, copy and cross process boundaries, so
``multiprocessing``, ``concurrent.futures`` and sinter's workers can use them."""

from __future__ import annotations

import copy
import multiprocessing
import pickle
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pytest

import stabilizer_qec as sq


def _decode_chunk(decoder, shots):
    return decoder.decode_batch(shots)


@pytest.fixture(scope="module")
def setup():
    c = sq.memory_circuit(distance=3, rounds=3, p=0.01)
    dem = c.detector_error_model(decompose_errors=True)
    shots = c.compile_detector_sampler(seed=4).sample(400)
    return c, dem, shots


def test_circuits_and_models_round_trip(setup):
    c, dem, _ = setup
    tagged = sq.Circuit("R 0 1\nH[t] 0\nCX 0 1\nOBSERVABLE_INCLUDE(0) Z0 Z1\nM 0 1\nDETECTOR rec[-1] rec[-2]")
    for obj in (c, dem, tagged, tagged.detector_error_model(), sq.Circuit()):
        for again in (pickle.loads(pickle.dumps(obj)), copy.copy(obj), copy.deepcopy(obj)):
            assert type(again) is type(obj) and again == obj
    assert pickle.loads(pickle.dumps(dem)).num_errors == dem.num_errors


def test_decoders_decode_the_same_after_pickling(setup):
    c, dem, shots = setup
    undecomposed = c.detector_error_model()
    decoders = [
        sq.Matching(dem),
        sq.Matching(dem, enable_correlations=True),
        sq.BeliefMatching(dem, max_bp_iters=5),
        sq.BpOsd(undecomposed, max_iter=5, osd_order=2),
        sq.WindowMatching(dem, commit=1, buffer=1, mode="sliding"),
    ]
    for d in decoders:
        for again in (pickle.loads(pickle.dumps(d)), copy.deepcopy(d)):
            assert type(again) is type(d)
            assert np.array_equal(again.decode_batch(shots), d.decode_batch(shots))
    w = pickle.loads(pickle.dumps(decoders[-1]))
    assert w.windows == decoders[-1].windows


def test_check_matrix_decoders_keep_their_last_run():
    h = np.array([[1, 1, 0, 1, 1, 0, 0], [1, 0, 1, 1, 0, 1, 0], [0, 1, 1, 1, 0, 0, 1]])
    bp = sq.BpDecoder(h, error_rate=0.05, max_iter=10)
    osd = sq.BpOsdDecoder(h, error_channel=[0.05] * 7, osd_order=0, osd_method="osd0")
    for d in (bp, osd):
        out = d.decode([1, 0, 1])
        again = pickle.loads(pickle.dumps(d))
        assert (again.converge, again.iter) == (d.converge, d.iter)
        assert np.array_equal(again.decode([1, 0, 1]), out)
    assert np.array_equal(pickle.loads(pickle.dumps(bp)).log_prob_ratios, bp.log_prob_ratios)


def test_converters_pickle_and_samplers_say_why_not(setup):
    c, _, _ = setup
    conv = c.compile_m2d_converter()
    meas = np.random.default_rng(1).random((5, c.num_measurements)) < 0.5
    assert np.array_equal(pickle.loads(pickle.dumps(conv)).convert(measurements=meas), conv.convert(measurements=meas))
    with pytest.raises(TypeError, match="send the circuit and a seed"):
        pickle.dumps(c.compile_detector_sampler(seed=1))


def test_codes_pickle():
    code = sq.BivariateBicycleCode("72")
    again = pickle.loads(pickle.dumps(code))
    assert again.name == code.name and all(np.array_equal(a, b) for a, b in zip(again.check_matrices(), code.check_matrices()))


def test_a_process_pool_decodes_as_one_process_does(setup):
    _, dem, shots = setup
    decoder = sq.Matching(dem)
    chunks = np.array_split(shots, 4)
    with ProcessPoolExecutor(2, mp_context=multiprocessing.get_context("spawn")) as pool:
        parts = list(pool.map(_decode_chunk, [decoder] * len(chunks), chunks))
    assert np.array_equal(np.concatenate(parts), decoder.decode_batch(shots))

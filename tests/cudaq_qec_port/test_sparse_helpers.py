# Ported from CUDA-Q QEC (NVIDIA/cudaqx 0.8.1, libs/qec/unittests/test_qec.cpp's PCMUtilsTester
# cases; Copyright (c) 2024 - 2026 NVIDIA Corporation & Affiliates, Apache License 2.0), modified
# for stabilizer_qec: the sparse-vector helpers (pcm_from_sparse_vec,
# generate_timelike_sparse_detector_matrix) and the patch dataclass, in Python.

import dataclasses

import numpy as np
import pytest

import stabilizer_qec.cudaq_qec as qec


def test_pcm_from_sparse_vec_round_trip():
    n_rounds, n_errs, n_syn = 4, 10, 5
    pcm = qec.generate_random_pcm(n_rounds, n_errs, n_syn, 3, 7)
    vec = qec.pcm_to_sparse_vec(pcm)
    back = qec.pcm_from_sparse_vec(vec, n_rounds * n_syn, n_rounds * n_errs)
    assert back.dtype == np.uint8
    np.testing.assert_array_equal(back, pcm)


def test_pcm_from_sparse_vec_rows_and_bounds():
    np.testing.assert_array_equal(qec.pcm_from_sparse_vec([0, 2, -1, -1, 1, -1], 3, 3), [[1, 0, 1], [0, 0, 0], [0, 1, 0]])
    # Trailing rows without entries stay zero; a missing final -1 is accepted.
    np.testing.assert_array_equal(qec.pcm_from_sparse_vec([1], 2, 2), [[0, 1], [0, 0]])
    with pytest.raises(RuntimeError, match="Invalid tensor indices: 0, 3"):
        qec.pcm_from_sparse_vec([3, -1], 1, 3)
    with pytest.raises(RuntimeError, match="Invalid tensor indices: 1, 0"):
        qec.pcm_from_sparse_vec([0, -1, 0, -1], 1, 3)


def test_timelike_detector_matrix():
    assert qec.generate_timelike_sparse_detector_matrix(2, 3, False) == [0, 2, -1, 1, 3, -1, 2, 4, -1, 3, 5, -1]
    assert qec.generate_timelike_sparse_detector_matrix(2, 3, True) == [0, -1, 1, -1, 0, 2, -1, 1, 3, -1, 2, 4, -1, 3, 5, -1]
    assert qec.generate_timelike_sparse_detector_matrix(3, 1, True) == [0, -1, 1, -1, 2, -1]
    assert qec.generate_timelike_sparse_detector_matrix(3, 1, False) == []
    assert qec.generate_timelike_sparse_detector_matrix(2, 0, False) == []


def test_timelike_detector_matrix_custom_first_round():
    out = qec.generate_timelike_sparse_detector_matrix(2, 3, [0, -1, 1, -1])
    assert out[0] == 0 and out[-1] == -1
    assert out == qec.generate_timelike_sparse_detector_matrix(2, 3, True)
    with pytest.raises(ValueError, match="must be non-empty"):
        qec.generate_timelike_sparse_detector_matrix(2, 3, [])
    with pytest.raises(ValueError, match="one index per row"):
        qec.generate_timelike_sparse_detector_matrix(2, 3, [0, 1, -1])


def test_timelike_detector_matrix_xors_consecutive_rounds():
    # As a measurement-to-detector map: the first round as it is, then each round XOR the one before.
    S, R = 4, 5
    D = qec.pcm_from_sparse_vec(qec.generate_timelike_sparse_detector_matrix(S, R, True), S * R, S * R)
    syn = np.random.default_rng(1).integers(0, 2, size=(20, S * R))
    dets = syn @ D.T % 2
    expect = np.concatenate([syn[:, :S], syn[:, S:] ^ syn[:, :-S]], axis=1)
    np.testing.assert_array_equal(dets, expect)


def test_timelike_detector_matrix_argument_types():
    with pytest.raises(TypeError):
        qec.generate_timelike_sparse_detector_matrix(-1, 3, False)
    with pytest.raises(TypeError):
        qec.generate_timelike_sparse_detector_matrix(2, 2**32, False)


def test_patch():
    p = qec.patch([0, 1, 2], [3], [4, 5])
    assert (p.data, p.ancx, p.ancz) == ([0, 1, 2], [3], [4, 5])
    assert [f.name for f in dataclasses.fields(qec.patch)] == ["data", "ancx", "ancz"]
    with pytest.raises(AttributeError):
        p.other = 1  # slots=True, as CUDA-Q's
    assert p == qec.patch([0, 1, 2], [3], [4, 5])

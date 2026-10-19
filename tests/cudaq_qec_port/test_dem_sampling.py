# ============================================================================ #
# Copyright (c) 2024 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
# Ported from CUDA-Q QEC's libs/qec/python/tests/test_dem_sampling.py (NVIDIA/cudaqx; Apache License
# 2.0); modified for stabilizer_qec: cudaq_qec is stabilizer_qec.cudaq_qec, the CPU tests only (the
# port has no GPU sampler, so backend="gpu" must raise), plus draws checked against std::mt19937.
"""Tests for cudaq_qec.dem_sampling (the CPU sampler)."""

import numpy as np
import pytest

from stabilizer_qec import cudaq_qec as qec
from stabilizer_qec.cudaq_qec.dem_sampling import dem_sampling


def _compute_syndrome(H, errors):
    """Reference: syndromes = errors @ H^T mod 2."""
    return (errors @ H.T) % 2


def test_exported_at_package_level():
    assert qec.dem_sampling is dem_sampling


class TestAllZeroProbabilities:

    def test_all_zeros(self):
        H = np.array([[1, 0, 1, 0], [0, 1, 1, 0], [0, 0, 0, 1]], dtype=np.uint8)
        probs = np.zeros(4)
        syndromes, errors = dem_sampling(H, 10, probs, seed=0)
        assert syndromes.shape == (10, 3)
        assert errors.shape == (10, 4)
        np.testing.assert_array_equal(errors, 0)
        np.testing.assert_array_equal(syndromes, 0)


class TestAllOneProbabilities:

    def test_all_ones(self):
        H = np.array([[1, 0, 1, 0], [1, 1, 0, 1], [0, 1, 1, 0]], dtype=np.uint8)
        probs = np.ones(4)
        syndromes, errors = dem_sampling(H, 5, probs, seed=0)
        np.testing.assert_array_equal(errors, 1)
        for shot in range(5):
            assert syndromes[shot, 0] == 0
            assert syndromes[shot, 1] == 1
            assert syndromes[shot, 2] == 0


class TestMixedDeterministicProbs:

    def test_mixed(self):
        H = np.array([[1, 0, 1], [0, 1, 1]], dtype=np.uint8)
        probs = np.array([1.0, 0.0, 1.0])
        syndromes, errors = dem_sampling(H, 8, probs, seed=42)
        for shot in range(8):
            np.testing.assert_array_equal(errors[shot], [1, 0, 1])
            np.testing.assert_array_equal(syndromes[shot], [0, 1])


class TestIdentityMatrix:

    def test_identity_all_ones(self):
        H = np.eye(5, dtype=np.uint8)
        syndromes, errors = dem_sampling(H, 3, np.ones(5), seed=7)
        np.testing.assert_array_equal(errors, 1)
        np.testing.assert_array_equal(syndromes, errors)

    def test_identity_syndrome_equals_errors(self):
        H = np.eye(10, dtype=np.uint8)
        probs = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=np.float64)
        syndromes, errors = dem_sampling(H, 4, probs, seed=0)
        np.testing.assert_array_equal(syndromes, errors)
        for shot in range(4):
            np.testing.assert_array_equal(errors[shot], [1, 0, 1, 0, 1, 0, 1, 0, 1, 0])


class TestAllOnesMatrix:

    def test_even_columns(self):
        syndromes, errors = dem_sampling(np.ones((3, 4), dtype=np.uint8), 4, np.ones(4), seed=0)
        np.testing.assert_array_equal(errors, 1)
        np.testing.assert_array_equal(syndromes, 0)

    def test_odd_columns(self):
        syndromes, errors = dem_sampling(np.ones((3, 3), dtype=np.uint8), 4, np.ones(3), seed=0)
        np.testing.assert_array_equal(errors, 1)
        np.testing.assert_array_equal(syndromes, 1)


class TestSingleColumnAndRow:

    def test_single_column(self):
        H = np.array([[1], [0], [1]], dtype=np.uint8)
        syndromes, errors = dem_sampling(H, 6, np.array([1.0]), seed=0)
        np.testing.assert_array_equal(errors, 1)
        for shot in range(6):
            np.testing.assert_array_equal(syndromes[shot], [1, 0, 1])

    def test_single_row(self):
        H = np.array([[1, 1, 0, 1, 0]], dtype=np.uint8)
        probs = np.array([1.0, 0.0, 1.0, 0.0, 1.0])
        syndromes, errors = dem_sampling(H, 4, probs, seed=0)
        for shot in range(4):
            np.testing.assert_array_equal(errors[shot], [1, 0, 1, 0, 1])
            assert syndromes[shot, 0] == 1

    def test_single_shot(self):
        H = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
        syndromes, errors = dem_sampling(H, 1, np.array([1.0, 1.0, 0.0]), seed=0)
        assert errors.shape == (1, 3)
        np.testing.assert_array_equal(errors[0], [1, 1, 0])
        np.testing.assert_array_equal(syndromes[0], [0, 1])

    def test_repetition_code_single_error(self):
        H = np.array([[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1]], dtype=np.uint8)
        syndromes, errors = dem_sampling(H, 3, np.array([0.0, 1.0, 0.0, 0.0]), seed=0)
        for shot in range(3):
            np.testing.assert_array_equal(errors[shot], [0, 1, 0, 0])
            np.testing.assert_array_equal(syndromes[shot], [1, 1, 0])


def test_syndrome_consistency():
    H = np.array([[1, 0, 1, 0, 0, 1, 0, 0], [0, 1, 0, 1, 0, 0, 1, 0], [0, 0, 1, 0, 1, 0, 0, 1], [1, 1, 0, 0, 0, 0, 1, 1]], dtype=np.uint8)
    probs = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    syndromes, errors = dem_sampling(H, 500, probs, seed=7)
    np.testing.assert_array_equal(syndromes, _compute_syndrome(H, errors))


class TestSeedReproducibility:

    def test_same_seed(self):
        H = np.eye(5, dtype=np.uint8)
        probs = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        s1, e1 = dem_sampling(H, 100, probs, seed=42)
        s2, e2 = dem_sampling(H, 100, probs, seed=42)
        np.testing.assert_array_equal(s1, s2)
        np.testing.assert_array_equal(e1, e2)

    def test_different_seeds_differ(self):
        H = np.eye(5, dtype=np.uint8)
        _, e1 = dem_sampling(H, 1000, np.full(5, 0.5), seed=1)
        _, e2 = dem_sampling(H, 1000, np.full(5, 0.5), seed=2)
        assert not np.array_equal(e1, e2)


class TestStdMt19937Draws:
    """The draws are CUDA-Q's CPU sampler's: std::mt19937(seed) and one
    std::bernoulli_distribution per mechanism, shot by shot (expected rows from a
    C++ program doing exactly that)."""

    def test_per_mechanism_probabilities(self):
        probs = [0.1, 0.5, 0.9, 0.25, 0.0, 1.0, 0.333, 0.75]
        _, errors = dem_sampling(np.eye(8, dtype=np.uint8), 6, probs, seed=7)
        expected = ["01000111", "10100101", "00100110", "01100101", "01100111", "01100100"]
        assert ["".join(map(str, r)) for r in errors] == expected

    def test_other_seeds(self):
        _, errors = dem_sampling(np.eye(7, dtype=np.uint8), 5, [0.01, 0.2, 0.5, 0.8, 0.99, 0.3, 0.6], seed=2026)
        assert ["".join(map(str, r)) for r in errors] == ["0001100", "0101100", "0011111", "0100101", "0001110"]
        _, errors = dem_sampling(np.eye(4, dtype=np.uint8), 3, np.full(4, 0.5), seed=4294967295)
        assert ["".join(map(str, r)) for r in errors] == ["1000"] * 3

    def test_matches_sample_code_capacity(self):
        # A constant probability over a code's parity matrix is sample_code_capacity's draw.
        H = qec.get_code("steane").get_parity()
        s1, e1 = qec.sample_code_capacity(H, 50, 0.2, 11)
        s2, e2 = dem_sampling(H, 50, np.full(H.shape[1], 0.2), seed=11)
        np.testing.assert_array_equal(e1, e2)
        np.testing.assert_array_equal(s1, s2)


class TestOutputProperties:

    def test_shapes_and_dtypes(self):
        H = np.array([[1, 0, 1], [0, 1, 1]], dtype=np.uint8)
        syndromes, errors = dem_sampling(H, 10, np.array([0.1, 0.2, 0.3]), seed=0)
        assert syndromes.shape == (10, 2) and errors.shape == (10, 3)
        assert syndromes.dtype == np.uint8 and errors.dtype == np.uint8

    def test_binary_output(self):
        syndromes, errors = dem_sampling(np.eye(5, dtype=np.uint8), 100, np.full(5, 0.5), seed=42)
        assert set(np.unique(syndromes)).issubset({0, 1})
        assert set(np.unique(errors)).issubset({0, 1})


class TestInputValidation:

    def test_non_2d_matrix(self):
        with pytest.raises(RuntimeError, match="rank-2"):
            dem_sampling(np.zeros(5, dtype=np.uint8), 10, np.zeros(5))

    def test_mismatched_probs(self):
        with pytest.raises(RuntimeError, match=r"error_probabilities length \(3\) != check_matrix columns \(5\)"):
            dem_sampling(np.eye(5, dtype=np.uint8), 10, np.zeros(3))

    def test_non_1d_probs(self):
        with pytest.raises(RuntimeError, match="1-D array, got ndim=2"):
            dem_sampling(np.eye(5, dtype=np.uint8), 10, np.zeros((5, 1)))

    @pytest.mark.parametrize("bad", [-0.2, 1.2, np.nan, np.inf])
    def test_probability_out_of_range(self, bad):
        probs = np.array([0.1, bad, 0.3], dtype=np.float64)
        with pytest.raises(RuntimeError, match=r"error_probabilities\[1\] is invalid"):
            dem_sampling(np.eye(3, dtype=np.uint8), 10, probs, backend="cpu")

    def test_negative_shots_and_seed(self):
        with pytest.raises(TypeError):
            dem_sampling(np.eye(2, dtype=np.uint8), -1, [0.1, 0.1])
        with pytest.raises(TypeError):
            dem_sampling(np.eye(2, dtype=np.uint8), 1, [0.1, 0.1], seed=-1)
        with pytest.raises(TypeError):
            dem_sampling(np.eye(2, dtype=np.uint8), 1, [0.1, 0.1], seed=2**32)


class TestBackendSelection:

    def test_invalid_backend(self):
        with pytest.raises(RuntimeError, match="invalid backend 'not-a-backend'. Expected one of: auto, cpu, gpu."):
            dem_sampling(np.eye(2, dtype=np.uint8), 4, np.array([0.1, 0.2]), seed=1, backend="not-a-backend")

    def test_force_cpu_backend_case_insensitive(self):
        H = np.array([[1, 0, 1], [0, 1, 1]], dtype=np.uint8)
        probs = np.array([1.0, 0.0, 1.0])
        for backend in ("cpu", "CPU", "Auto"):
            syndromes, errors = dem_sampling(H, 5, probs, seed=11, backend=backend)
            for shot in range(5):
                np.testing.assert_array_equal(errors[shot], [1, 0, 1])
                np.testing.assert_array_equal(syndromes[shot], [0, 1])

    def test_gpu_backend_unavailable(self):
        with pytest.raises(RuntimeError, match="GPU backend requested but unavailable"):
            dem_sampling(np.eye(2, dtype=np.uint8), 4, np.array([0.1, 0.2]), seed=1, backend="gpu")
        # Invalid probabilities are reported first, as CUDA-Q does.
        with pytest.raises(RuntimeError, match="is invalid"):
            dem_sampling(np.eye(2, dtype=np.uint8), 4, np.array([0.1, 2.0]), seed=1, backend="gpu")


class TestZeroShots:

    def test_zero_shots_numpy(self):
        H = np.array([[1, 0, 1], [0, 1, 1]], dtype=np.uint8)
        syndromes, errors = dem_sampling(H, 0, np.array([0.1, 0.2, 0.3]), seed=0)
        assert syndromes.shape == (0, 2) and errors.shape == (0, 3)
        assert syndromes.dtype == np.uint8 and errors.dtype == np.uint8

    def test_zero_shots_cpu_backend(self):
        syndromes, errors = dem_sampling(np.eye(2, dtype=np.uint8), 0, np.array([0.5, 0.5]), seed=0, backend="cpu")
        assert syndromes.shape == (0, 2) and errors.shape == (0, 2)

    def test_zero_shots_skips_probability_values(self):
        # Only the shapes are checked when no shot is drawn.
        syndromes, errors = dem_sampling(np.eye(2, dtype=np.uint8), 0, np.array([0.5, 7.0]))
        assert syndromes.shape == (0, 2)
        with pytest.raises(RuntimeError, match="length"):
            dem_sampling(np.eye(2, dtype=np.uint8), 0, np.array([0.5]))


def test_non_binary_entries_masked():
    """H entries > 1 should be treated as H & 1 (binary)."""
    H = np.array([[2, 3], [1, 0]], dtype=np.uint8)
    syndromes, errors = dem_sampling(H, 4, np.array([1.0, 1.0]), seed=0, backend="cpu")
    np.testing.assert_array_equal(errors, 1)
    np.testing.assert_array_equal(syndromes, 1)


class TestSeedlessPath:

    def test_seedless_runs(self):
        H = np.array([[1, 0, 1], [0, 1, 1]], dtype=np.uint8)
        syndromes, errors = dem_sampling(H, 50, np.array([0.3, 0.5, 0.7]))
        assert syndromes.shape == (50, 2) and errors.shape == (50, 3)

    def test_seedless_nondeterministic(self):
        H = np.eye(10, dtype=np.uint8)
        _, e1 = dem_sampling(H, 500, np.full(10, 0.5))
        _, e2 = dem_sampling(H, 500, np.full(10, 0.5))
        assert not np.array_equal(e1, e2)


def test_torch_not_installed_warning():
    import sys

    class FakeTensor:
        def data_ptr(self):
            return 0

    real_torch = sys.modules.get("torch")
    sys.modules["torch"] = None  # force ImportError on `import torch`
    try:
        with pytest.warns(UserWarning, match="pip install torch"):
            try:
                dem_sampling(FakeTensor(), 1, FakeTensor(), seed=0)
            except Exception:
                pass
    finally:
        if real_torch is not None:
            sys.modules["torch"] = real_torch
        else:
            sys.modules.pop("torch", None)


def test_lists_accepted_strings_rejected():
    syndromes, errors = dem_sampling([[1, 0], [0, 1]], 4, [0.5, 0.5], seed=0)
    assert syndromes.shape == (4, 2) and errors.shape == (4, 2)
    with pytest.raises((TypeError, RuntimeError, ValueError)):
        dem_sampling("not a matrix", 4, "not probs", seed=0)

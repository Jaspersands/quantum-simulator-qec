# Ported from CUDA-Q QEC (NVIDIA/cudaqx, libs/qec/python/cudaq_qec/dem_sampling.py,
# python/bindings/py_dem_sampling.cpp and lib/experiments.cpp; Copyright (c) 2024 - 2026 NVIDIA
# Corporation & Affiliates, Apache License 2.0; see THIRD_PARTY_NOTICES.md), modified for
# stabilizer_qec: the CPU sampler only (no cuStabilizer, no PyTorch tensors).
"""DEM sampling: independent Bernoulli errors per mechanism and their syndromes.

Public API:
    dem_sampling(check_matrix, num_shots, error_probabilities, seed=None, backend="auto")

CUDA-Q's CPU path, bit for bit: a ``std::mt19937`` seeded with ``seed`` and one
``std::bernoulli_distribution`` per mechanism, drawn shot by shot. ``backend="auto"`` and
``"cpu"`` run it; ``"gpu"`` raises, as CUDA-Q does with no CUDA device.
"""

from __future__ import annotations

import sys
import warnings
from typing import Optional, Tuple

import numpy as np

__all__ = ["dem_sampling"]


def _torch_tensor(obj) -> bool:
    torch = sys.modules.get("torch")
    return torch is not None and isinstance(obj, getattr(torch, "Tensor", ()))


def _uniforms(seed: int, count: int) -> np.ndarray:
    # generate_canonical<double, 53> over std::mt19937: two 32-bit draws, (lo + hi 2^32) / 2^64.
    from ._codes import _mt_raw

    raw = _mt_raw(np.random.RandomState(seed & 0xFFFFFFFF), 2 * count)
    lo, hi = raw[0::2].astype(np.float64), raw[1::2].astype(np.float64)
    u = (lo + hi * 4294967296.0) / 18446744073709551616.0
    return np.where(u >= 1.0, np.nextafter(1.0, 0.0), u)


def _as_inputs(check_matrix, error_probabilities) -> Tuple[np.ndarray, np.ndarray]:
    H = np.ascontiguousarray(check_matrix, dtype=np.uint8)
    p = np.ascontiguousarray(error_probabilities, dtype=np.float64)
    if p.ndim != 1:
        raise RuntimeError(f"error_probabilities must be a 1-D array, got ndim={p.ndim}")
    if H.ndim != 2:
        raise RuntimeError("check_matrix must be rank-2")
    if p.shape[0] != H.shape[1]:
        raise RuntimeError(f"dem_sampling: error_probabilities length ({p.shape[0]}) != check_matrix columns ({H.shape[1]})")
    return H, p


def _count(value, what: str, bits: Optional[int]) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"dem_sampling(): incompatible function arguments: {what} must be a non-negative integer, got {type(value).__name__}")
    v = int(value)
    if v < 0 or (bits is not None and v >= 1 << bits):
        raise TypeError(f"dem_sampling(): incompatible function arguments: {what} {v} is out of range")
    return v


def dem_sampling(check_matrix, num_shots: int, error_probabilities, seed: Optional[int] = None, backend: str = "auto") -> Tuple[object, object]:
    """Sample errors and syndromes from a Detector Error Model.

    Args:
        check_matrix: Binary matrix [num_checks x num_error_mechanisms], as a
            NumPy array (entries are read as uint8 and binarized, ``H & 1``).
        num_shots: Number of independent Monte-Carlo shots.
        error_probabilities: 1-D array of length num_error_mechanisms with
            independent Bernoulli probabilities for each mechanism.
        seed: Optional RNG seed (unsigned 32-bit) for reproducibility; the
            draws then equal CUDA-Q's CPU sampler's.
        backend: "auto" (default) or "cpu" run the CPU sampler; "gpu"
            raises, since this package has no GPU sampler.

    Returns:
        (syndromes, errors) where
          syndromes: uint8 array [num_shots x num_checks]
          errors:    uint8 array [num_shots x num_error_mechanisms]
    """
    for obj in (check_matrix, error_probabilities):
        if hasattr(obj, "data_ptr"):
            try:
                import torch  # noqa: F401
            except ImportError:
                warnings.warn("Input appears to be a PyTorch tensor but torch is not installed. Install it with: pip install torch", stacklevel=2)
            break
    if not isinstance(backend, str):
        raise TypeError(f"dem_sampling(): incompatible function arguments: backend must be a str, got {type(backend).__name__}")
    mode = backend.lower()
    if mode not in ("auto", "cpu", "gpu"):
        raise RuntimeError(f"dem_sampling: invalid backend '{mode}'. Expected one of: auto, cpu, gpu.")
    shots = _count(num_shots, "num_shots", None)
    if seed is not None:
        seed = _count(seed, "seed", 32)

    if shots == 0:
        H, p = _as_inputs(check_matrix, error_probabilities)
        return np.empty((0, H.shape[0]), dtype=np.uint8), np.empty((0, H.shape[1]), dtype=np.uint8)

    if seed is None:
        seed = int(np.random.SeedSequence().generate_state(1)[0])

    torch_reason = "PyTorch is not available" if "torch" not in sys.modules else "inputs are not both torch.Tensor"
    if mode != "cpu" and _torch_tensor(check_matrix) and _torch_tensor(error_probabilities):
        torch_reason = "no CUDA device available"
    if mode == "gpu" and any(_torch_tensor(o) for o in (check_matrix, error_probabilities)):
        raise RuntimeError(f"dem_sampling: GPU backend requested but unavailable. torch path: {torch_reason}")
    if any(_torch_tensor(o) for o in (check_matrix, error_probabilities)):
        raise RuntimeError("dem_sampling: PyTorch CPU tensors are not supported. Convert to NumPy with .numpy() or use CUDA tensors.")

    H, p = _as_inputs(check_matrix, error_probabilities)
    for i, v in enumerate(p):
        if not np.isfinite(v) or v < 0.0 or v > 1.0:
            raise RuntimeError(f"dem_sampling: error_probabilities[{i}] is invalid. Values must be finite and in [0, 1].")
    if mode == "gpu":
        raise RuntimeError(f"dem_sampling: GPU backend requested but unavailable. torch path: {torch_reason}; numpy path: no CUDA device available")

    n = H.shape[1]
    errors = (_uniforms(seed, shots * n).reshape(shots, n) < p[None, :]).astype(np.uint8)
    Hb = (H & 1).astype(np.int64)
    syndromes = (errors.astype(np.int64) @ Hb.T % 2).astype(np.uint8)
    return syndromes, errors

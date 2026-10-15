# Ported from CUDA-Q QEC (NVIDIA/cudaqx, libs/qec/python/cudaq_qec/plugins/decoders/
# tensor_network_decoder.py; Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates, Apache
# License 2.0; see THIRD_PARTY_NOTICES.md), modified for stabilizer_qec: the network is
# contracted exactly by this package (src/tn_ml.rs), without quimb, torch or cuQuantum.
"""CUDA-Q QEC's ``tensor_network_decoder``: the exact probability that a logical observable
flipped, given a (soft) syndrome, a check matrix and independent error probabilities."""

from __future__ import annotations

from typing import Any, List, Optional

import numpy as np

from stabilizer_qec import _core
from stabilizer_qec import cudaq_qec as qec
from stabilizer_qec._util import call


def _safe_posterior(mass0: float, mass1: float, dtype: str) -> tuple:
    """The posterior mass1 / (mass0 + mass1), or (0.5, False) for masses that are not physical
    (negative beyond round-off, not finite, or both zero)."""
    m0, m1 = float(mass0), float(mass1)
    if not np.isfinite(m0) or not np.isfinite(m1):
        return 0.5, False
    scale = max(abs(m0), abs(m1))
    tolerance = 64.0 * np.finfo(np.dtype(dtype)).eps * scale
    if m0 < -tolerance or m1 < -tolerance:
        return 0.5, False
    m0, m1 = max(m0, 0.0), max(m1, 0.0)
    denom = m0 + m1
    if not np.isfinite(denom) or denom == 0.0:
        return 0.5, False
    return m1 / denom, True


@qec.decoder("tensor_network_decoder")
class TensorNetworkDecoder:
    """Exact maximum-likelihood decoding of one logical observable by contracting the code's
    Tanner graph with the error priors (``noise_model``: one probability per error) and the
    syndrome: ``decode`` gives the probability that the logical flipped."""

    def __init__(
        self,
        H: Any,
        logical_obs: Any,
        noise_model: Any,
        check_inds: Optional[List[str]] = None,
        error_inds: Optional[List[str]] = None,
        logical_inds: Optional[List[str]] = None,
        logical_tags: Optional[List[str]] = None,
        contract_noise_model: bool = True,
        dtype: str = "float64",
        device: str = "cuda",
    ) -> None:
        qec.Decoder.__init__(self, H)
        a = H.toarray() if hasattr(H, "toarray") else np.asarray(H)
        num_checks, num_errs = a.shape
        if check_inds is None:
            self.check_inds = [f"s_{j}" for j in range(num_checks)]
        else:
            assert len(check_inds) == num_checks, f"check_inds must have length {num_checks}, but got {len(check_inds)}."
            self.check_inds = list(check_inds)
        if error_inds is None:
            self.error_inds = [f"e_{j}" for j in range(num_errs)]
        else:
            assert len(error_inds) == num_errs, f"error_inds must have length {num_errs}, but got {len(error_inds)}."
            self.error_inds = list(error_inds)
        self.logical_obs_inds = ["obs"]
        self.parity_check_matrix = a.copy()
        self._columns = [np.flatnonzero(a[:, j] != 0).tolist() for j in range(num_errs)]
        from stabilizer_qec.cudaq_qec._pcm import _round_order

        self._order = _round_order(self._columns, 0, None)
        self.replace_logical_observable(logical_obs, logical_inds=logical_inds, logical_tags=logical_tags)
        priors = np.asarray(noise_model, dtype=np.float64)
        if priors.ndim != 1 or priors.size != num_errs:
            raise ValueError(f"noise_model must give one probability per error ({num_errs}); a tensor-network noise model needs quimb's contraction, which this package does not use.")
        self._priors = priors.tolist()
        self._dtype = dtype
        self._device = "cpu"
        self._syndrome = [0.0] * num_checks
        self.path_single = "auto"
        self.path_batch = "auto"

    def replace_logical_observable(self, logical_obs: Any, logical_inds: Optional[List[str]] = None, logical_tags: Optional[List[str]] = None) -> None:
        """Decode a different logical observable (one row, one entry per error)."""
        lo = np.asarray(logical_obs)
        assert lo.shape == (1, len(self.error_inds)), "logical must be a single row matrix, shape (1, n), where n is the number of errors.Only single logical are supported for now."
        if logical_inds is None:
            self.logical_inds = ["l_0"]
        else:
            assert len(logical_inds) == 1, "logical_inds must be a list of length 1, as only single logical observables are supported for now."
            self.logical_inds = list(logical_inds)
        self.logical_tags = ["LOG_0"] if logical_tags is None else list(logical_tags)
        self.logical_obs = lo.copy()
        self._logical = [bool(x) for x in (lo[0] != 0)]

    def flip_syndromes(self, values: List[float]) -> None:
        """Set the syndrome (each check's probability of having fired) for ``decode``."""
        assert len(values) == len(self.check_inds), f"Values length {len(values)} does not match the number of checks {len(self.check_inds)}."
        assert all(isinstance(v, float) for v in values), "Syndrome values must be float."
        self._syndrome = list(values)

    def _set_contractor(self, contractor: str, device: str, backend: str, dtype: Optional[str] = None) -> None:
        """Kept for CUDA-Q's API: the contraction here is exact and on the CPU; ``dtype`` is the
        precision the posterior is judged in."""
        if dtype is not None:
            self._dtype = dtype

    def optimize_path(self, optimize: Any = None, batch_size: int = -1) -> Any:
        """Kept for CUDA-Q's API: the contraction order is the matrix's round order."""
        assert isinstance(batch_size, int), "batch_size must be an integer, or -1 to indicate no batching."
        return None

    def _masses(self, syndromes: List[List[float]]) -> List[tuple]:
        return call(_core.tn_ml_masses, len(self.check_inds), self._columns, self._logical, self._priors, syndromes, self._order)

    def decode(self, syndrome: List[float]) -> "qec.DecoderResult":
        """The probability that the logical flipped (``result[0]``) given the syndrome (each
        check's probability of having fired)."""
        assert len(syndrome) == len(self.check_inds), f"Syndrome length {len(syndrome)} does not match the number of checks {len(self.check_inds)}."
        assert all(isinstance(s, float) for s in syndrome), "Syndrome values must be float."
        self.flip_syndromes(list(syndrome))
        m0, m1 = self._masses([list(syndrome)])[0]
        posterior, valid = _safe_posterior(m0, m1, self._dtype)
        res = qec.DecoderResult()
        res.converged = valid
        res.result = [posterior]
        return res

    def decode_batch(self, syndrome_batch: Any) -> "qec.BatchDecoderResult":
        """Each row's probability that the logical flipped, as a (rows, 1) result."""
        batch = np.asarray(syndrome_batch, dtype=np.float64)
        assert batch.shape[1] == len(self.check_inds)
        posts, conv = [], []
        for m0, m1 in self._masses(batch.tolist()):
            p, v = _safe_posterior(m0, m1, self._dtype)
            posts.append(p)
            conv.append(v)
        return qec.BatchDecoderResult(np.asarray(posts, dtype=np.float64).reshape((-1, 1)), np.asarray(conv, dtype=bool))

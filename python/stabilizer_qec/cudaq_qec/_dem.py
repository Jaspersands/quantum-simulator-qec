# Ported from CUDA-Q QEC (NVIDIA/cudaqx, libs/qec/lib/detector_error_model.cpp; Copyright (c)
# 2022 - 2026 NVIDIA Corporation & Affiliates, Apache License 2.0; see THIRD_PARTY_NOTICES.md),
# modified for stabilizer_qec: Stim's parser is this package's.
"""CUDA-Q QEC's detector error model: check, observable and rate arrays from a Stim model."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _parse_stim_dem(dem_text: str, use_decomp_suggestions: bool) -> Tuple[int, int, List[List[int]], List[List[int]], List[float]]:
    """(detectors, observables, each mechanism's detectors and observables, rates): the model's
    errors flattened, each one mechanism (each ``^`` part one with ``use_decomp_suggestions``),
    targets XORed, mechanisms flipping nothing dropped."""
    from .._circuit import DetectorErrorModel as StimDem

    try:
        dem = StimDem(dem_text).flattened()
    except Exception as e:
        raise RuntimeError(f"Stim DEM parse failed: {e}") from None
    num_dets, num_obs = dem.num_detectors, dem.num_observables
    det_hits: List[List[int]] = []
    obs_hits: List[List[int]] = []
    rates: List[float] = []
    k = 0
    for inst in dem:
        if inst.type != "error":
            continue
        args = inst.args_copy()
        if not args:
            raise RuntimeError(f"Stim DEM error instruction missing probability argument (index {k})")
        p = float(args[0])
        if not 0.0 <= p <= 1.0:
            raise RuntimeError(f"Stim DEM error probability {p:f} out of range [0, 1] at instruction index {k}")
        dets: set = set()
        obs: set = set()

        def flush() -> None:
            if dets or obs:
                det_hits.append(sorted(dets))
                obs_hits.append(sorted(obs))
                rates.append(p)
                dets.clear()
                obs.clear()

        for t in inst.targets_copy():
            if t.is_separator():
                if use_decomp_suggestions:
                    flush()
                continue
            if t.is_relative_detector_id():
                dets.symmetric_difference_update({t.val})
            elif t.is_logical_observable_id():
                obs.symmetric_difference_update({t.val})
        flush()
        k += 1
    if not det_hits:
        raise RuntimeError("Stim DEM contains no error mechanisms after flattening")
    return num_dets, num_obs, det_hits, obs_hits, rates


class DetectorErrorModel:
    """Which detectors (``detector_error_matrix``) and observables (``observables_flips_matrix``)
    each error mechanism flips, its rate (``error_rates``) and optional ``error_ids`` (errors of
    one id are exclusive)."""

    def __init__(self) -> None:
        self._H = np.zeros((0, 0), dtype=np.uint8)
        self._O = np.zeros((0, 0), dtype=np.uint8)
        self.error_rates: List[float] = []
        self.error_ids: Optional[List[int]] = None

    @property
    def detector_error_matrix(self) -> np.ndarray:
        return self._H.copy()

    @detector_error_matrix.setter
    def detector_error_matrix(self, a: Any) -> None:
        self._H = np.array(a, dtype=np.uint8)

    @property
    def observables_flips_matrix(self) -> np.ndarray:
        return self._O.copy()

    @observables_flips_matrix.setter
    def observables_flips_matrix(self, a: Any) -> None:
        self._O = np.array(a, dtype=np.uint8)

    def num_detectors(self) -> int:
        return self._H.shape[0] if self._H.ndim == 2 else 0

    def num_error_mechanisms(self) -> int:
        return self._H.shape[1] if self._H.ndim == 2 else 0

    def num_observables(self) -> int:
        return self._O.shape[0] if self._O.ndim == 2 else 0

    def canonicalize_for_rounds(self, num_syndromes_per_round: int, remove_zero_syndrome_errors: bool = False) -> None:
        """Sort the mechanisms by the rounds they touch and merge those of one signature
        (detectors and observables), composing their rates; mechanisms of rate 0, and those
        flipping nothing (or, with ``remove_zero_syndrome_errors``, no detector), are dropped."""
        from ._pcm import _round_order

        self._canonicalize(_round_order(self._columns(), int(num_syndromes_per_round), None), remove_zero_syndrome_errors)

    def canonicalize_for_rounds_with_boundary(self, num_syndromes_per_round: int, num_boundary_syndromes: int, remove_zero_syndrome_errors: bool = False) -> None:
        """As ``canonicalize_for_rounds``, the first and last rounds ``num_boundary_syndromes``
        detectors wide."""
        from ._pcm import _round_order

        if num_boundary_syndromes > num_syndromes_per_round:
            raise ValueError(f"canonicalize_for_rounds_with_boundary: num_boundary_syndromes ({num_boundary_syndromes}) must be <= num_syndromes_per_round ({num_syndromes_per_round})")
        if num_syndromes_per_round == 0:
            raise ValueError("get_sorted_pcm_column_indices: num_syndromes_per_round must be non-zero when num_boundary_syndromes is specified")
        self._canonicalize(_round_order(self._columns(), int(num_syndromes_per_round), int(num_boundary_syndromes)), remove_zero_syndrome_errors)

    def _columns(self) -> List[List[int]]:
        return [np.flatnonzero(self._H[:, j]).tolist() for j in range(self.num_error_mechanisms())]

    def _canonicalize(self, order: List[int], remove_zero: bool) -> None:
        cols = self._columns()
        if len(cols) > len(self.error_rates):
            raise RuntimeError(
                f"canonicalize_for_rounds: row_indices size ({len(cols)}) is greater than the number of error rates ({len(self.error_rates)}). "
                "This likely means either 'error_rates' was populated incorrectly or the detector_error_matrix  was computed incorrectly."
            )
        has_ids = self.error_ids is not None and len(self.error_ids) == len(self.error_rates)
        sig_to_out: Dict[tuple, int] = {}
        keep: List[int] = []
        exclusive: List[Dict[int, float]] = []
        for c in order:
            dets = tuple(cols[c])
            rate = self.error_rates[c]
            obs = tuple(int(r) for r in np.flatnonzero(self._O[:, c])) if self.num_observables() else ()
            zero = not dets
            if rate == 0.0 or (zero and not obs) or (remove_zero and zero):
                continue
            key = (dets, obs)
            if key not in sig_to_out:
                sig_to_out[key] = len(exclusive)
                exclusive.append({})
                keep.append(c)
            group = exclusive[sig_to_out[key]]
            k = self.error_ids[c] if has_ids else c
            group[k] = group.get(k, 0.0) + rate
        weights = []
        for group in exclusive:
            w = 0.0
            for k in sorted(group):
                p = group[k]
                w = w + p - 2.0 * w * p
            weights.append(w)
        self.error_rates = weights
        if has_ids:
            self.error_ids = list(range(len(weights)))
        self._H = self._H[:, keep] if keep else np.zeros((self._H.shape[0], 0), dtype=np.uint8)
        self._O = self._O[:, keep] if keep else np.zeros((self._O.shape[0], 0), dtype=np.uint8)


def dem_from_stim_text(dem_text: str, use_decomp_suggestions: bool = False) -> DetectorErrorModel:
    """A Stim detector error model (text) as CUDA-Q's: one column per error (per ``^`` part with
    ``use_decomp_suggestions``), its rate the error's probability."""
    num_dets, num_obs, det_hits, obs_hits, rates = _parse_stim_dem(dem_text, use_decomp_suggestions)
    for hits in det_hits:
        if any(d >= num_dets for d in hits):
            raise RuntimeError("Stim DEM detector id out of range while extracting H")
    H = np.zeros((num_dets, len(det_hits)), dtype=np.uint8)
    O = np.zeros((num_obs, len(det_hits)), dtype=np.uint8)
    for e, (dh, oh) in enumerate(zip(det_hits, obs_hits)):
        H[dh, e] ^= 1
        O[oh, e] ^= 1
    out = DetectorErrorModel()
    out._H, out._O, out.error_rates = H, O, list(rates)
    return out

# Ported from CUDA-Q QEC (NVIDIA/cudaqx, libs/qec/lib/decoders/plugins/chromobius/chromobius.cpp and
# docs/sphinx/api/qec/nv_fusion_decoder_api.rst; Copyright (c) 2026 NVIDIA Corporation & Affiliates,
# Apache License 2.0; see THIRD_PARTY_NOTICES.md), modified for stabilizer_qec: Chromobius is this
# package's port, and the fusion decoder's matching is this package's fusion_blossom port.
"""CUDA-Q QEC's ``chromobius`` and ``nv-fusion-decoder`` plugins.

``chromobius`` decodes a Stim detector error model (its detectors carrying Chromobius's colour
annotation) to observable flips, as CUDA-Q's wrapper of Chromobius does, on this package's
Chromobius port.

``nv-fusion-decoder`` is CUDA-Q's minimum-weight perfect matching decoder for graphlike models.
CUDA-Q splits the rounds into blocks matched separately and fused (its brickwall schedule); a
single-block schedule is exact MWPM, and a fused one an approximation of it. This port always
matches the whole shot at once, on fusion_blossom's exact solver, so every ``block_leaf_size``
gives the single-block (exact) answer; ``num_threads`` is accepted and does not change it.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ._decoders import BatchDecoderResult, DecoderResult, _batch, _hard, _Inputs, _make_result, _NativeDecoder

# -- chromobius ------------------------------------------------------------------------------------

_CHROMOBIUS_KEYS = ("drop_mobius_errors_involving_remnant_errors", "ignore_decomposition_failures", "include_coords_in_mobius_dem", "return_weight", "write_mobius_match_to_stderr")


def _bool_param(params: dict, key: str, default: bool) -> bool:
    if key not in params:
        return default
    v = params[key]
    if not isinstance(v, (bool, np.bool_)):
        raise RuntimeError(f"Invalid type for parameter '{key}': expected bool")
    return bool(v)


def _chromobius_from_matrix(inputs: _Inputs, output: str, params: dict) -> "_Chromobius":
    raise RuntimeError('Chromobius decoder requires a Stim detector error model string as decoder input. Use get_decoder("chromobius", dem_text, params).')


def _observables_only(name: str, output: Optional[str]) -> None:
    if output not in (None, "observables"):
        raise RuntimeError(f"{name} built from a detector error model returns observable flips only; construct it for observable output")


class _Chromobius(_NativeDecoder):
    """CUDA-Q's ``chromobius``: Chromobius compiled for the model; a decode returns the
    predicted observable flips (``converged`` always), and with ``return_weight`` the matching's
    weight in ``opt_results["weight"]``. Its block size is the number of observables, its
    observable map the identity."""

    def __init__(self, dem_text: str, params: dict, output: Optional[str] = None) -> None:
        from .._circuit import DetectorErrorModel as StimDem
        from .._decoders import ColorMatching

        _observables_only("chromobius", output)
        try:
            dem = StimDem(dem_text)
        except Exception as e:
            raise RuntimeError(f"Chromobius Stim DEM parse failed: {e}") from None
        nd, no = int(dem.num_detectors), int(dem.num_observables)
        if no > 64:
            raise RuntimeError("Chromobius currently returns observable flips as a 64-bit mask; CUDA-Q QEC wrapper supports at most 64 observables.")
        if not _bool_param(params, "drop_mobius_errors_involving_remnant_errors", True):
            raise RuntimeError("chromobius: drop_mobius_errors_involving_remnant_errors=False is not supported by this package's Chromobius port (it always drops them, Chromobius's default)")
        if _bool_param(params, "write_mobius_match_to_stderr", False):
            raise RuntimeError("chromobius: write_mobius_match_to_stderr=True is not supported by this package's Chromobius port")
        _bool_param(params, "include_coords_in_mobius_dem", False)  # (only shapes Chromobius's debug model)
        self._return_weight = _bool_param(params, "return_weight", False)
        ignore = _bool_param(params, "ignore_decomposition_failures", False)
        super().__init__(_Inputs(nd, no, [[] for _ in range(no)], [[o] for o in range(no)], no, []), "observables", params)
        self._matcher = ColorMatching(dem, ignore_decomposition_failures=ignore)

    def get_version(self) -> str:
        return "CUDA-Q QEC Chromobius Decoder wrapper"

    def _decode_rows(self, rows: np.ndarray) -> List[DecoderResult]:
        if rows.shape[0] == 0:
            return []
        preds, weights = self._matcher.decode_batch(rows.astype(np.uint8), return_weights=True)
        out = []
        for p, w in zip(preds, weights):
            opt = {"weight": float(np.float32(w))} if self._return_weight else None
            out.append(_make_result(True, np.asarray(p, dtype=np.float64)[: self._block_size], opt))
        return out

    def _rows(self, syndromes: Any) -> np.ndarray:
        rows = [_hard(s) for s in syndromes]
        for r in rows:
            if r.size != self._syndrome_size:
                raise RuntimeError("Chromobius syndrome length must match the number of detectors")
        return np.array(rows, dtype=np.uint8).reshape(len(rows), self._syndrome_size)

    def decode(self, syndrome: Any) -> DecoderResult:
        return self._decode_rows(self._rows([syndrome]))[0]

    def decode_batch(self, syndromes: Any) -> BatchDecoderResult:
        return _batch(self._decode_rows(self._rows(list(syndromes))))


# -- nv-fusion-decoder -----------------------------------------------------------------------------

_FUSION_KEYS = ("O", "detector_round", "D_sparse", "dem_string", "block_leaf_size", "num_threads", "error_rate_vec", "fusion_strategy")


def _uint(params: dict, key: str) -> Optional[int]:
    if key not in params:
        return None
    v = params[key]
    if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) or int(v) < 0:
        raise RuntimeError(f"nv-fusion-decoder: '{key}' must be a non-negative integer")
    return int(v)


def _weight(p: float) -> float:
    return float(-np.log(p / (1.0 - p)))


class _NvFusion(_NativeDecoder):
    """CUDA-Q's ``nv-fusion-decoder``, matched exactly (see the module notes). Built from a
    graphlike ``H`` (each column one or two detectors), with ``O`` for observable output and
    ``error_rate_vec`` for weights ln((1 − p)/p) (unit weights without), or from a Stim model
    (its ``^`` decompositions applied; observable output). A decode returns ``converged`` (no
    herald) and ``opt_results["heralded"]``."""

    def __init__(self, num_dets: int, mechanisms: List[Tuple[List[int], List[int], float]], num_obs: Optional[int], block_size: int, column_of: Optional[List[int]], params: dict, dem_mode: bool) -> None:
        from .. import fusion_blossom as fb

        strategy = params.get("fusion_strategy", "brickwall")
        if strategy != "brickwall":
            raise RuntimeError(f"nv-fusion-decoder: unsupported fusion_strategy '{strategy}'; only 'brickwall' is supported")
        leaf = _uint(params, "block_leaf_size")
        if leaf is not None and leaf == 0:
            raise RuntimeError("nv-fusion-decoder: block_leaf_size must be positive")
        _uint(params, "num_threads")
        if "detector_round" in params:
            rounds = np.asarray(params["detector_round"]).reshape(-1)
            if rounds.size != num_dets:
                raise RuntimeError(f"nv-fusion-decoder: detector_round has length {rounds.size} but there are {num_dets} detectors")
        if "D_sparse" in params:
            [int(v) for v in params["D_sparse"]]
        self._to_obs = num_obs is not None
        out_size = num_obs if self._to_obs else block_size
        O_rows: Optional[List[List[int]]] = None
        if dem_mode:
            O_rows = [[o] for o in range(num_obs)]  # (the result already is the observables)
        elif self._to_obs:
            O_rows = [[] for _ in range(num_obs)]
            for j, (_dets, obs, _w) in enumerate(mechanisms):
                for o in obs:
                    O_rows[o].append(column_of[j])
        super().__init__(_Inputs(num_dets, block_size, [] if dem_mode else [list(m[0]) for m in mechanisms], O_rows, num_obs or 0, []), "observables" if self._to_obs else "errors", params)
        self._out_size = out_size
        # One edge per detector pair (the boundary a virtual vertex), the lightest mechanism kept
        # (a model's equal-signature mechanisms merged first, as independent errors).
        boundary = num_dets
        best: Dict[Tuple[int, int], Tuple[float, int, int]] = {}
        for j, (dets, obs, w) in enumerate(mechanisms):
            key = (dets[0], dets[1]) if len(dets) == 2 else (dets[0], boundary)
            mask = 0
            for o in obs:
                mask ^= 1 << o
            if key not in best or w < best[key][0]:
                best[key] = (w, mask, j)
        self._edges = list(best.items())
        top = max((v[0] for _, v in self._edges), default=0.0)
        scale = (1 << 24) / top if top > 0 else 1.0
        weighted = [(a, b, 2 * int(round(w * scale / 2.0))) for (a, b), (w, _m, _j) in self._edges]
        self._masks = [m for _, (_w, m, _j) in self._edges]
        self._cols = [column_of[j] if column_of is not None else j for _, (_w, _m, j) in self._edges]
        # Components that cannot reach the boundary need an even number of detection events.
        parent = list(range(num_dets + 1))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for a, b, _ in weighted:
            parent[find(a)] = find(b)
        self._root = [find(v) for v in range(num_dets)]
        self._open = find(boundary)
        self._solver = fb.SolverSerial(fb.SolverInitializer(num_dets + 1, weighted, [boundary]))
        self._fb = fb

    def get_version(self) -> str:
        return "CUDA-Q QEC NV Fusion Decoder (stabilizer_qec: exact MWPM on fusion_blossom)"

    def decode(self, syndrome: Any) -> DecoderResult:
        hard = _hard(syndrome)
        if hard.size != self._syndrome_size:
            raise RuntimeError(f"nv-fusion-decoder: syndrome length ({hard.size}) must match the number of detectors ({self._syndrome_size})")
        fired = np.flatnonzero(hard).tolist()
        out = np.zeros(self._out_size)
        if fired:
            odd: Dict[int, int] = {}
            for d in fired:
                r = self._root[d]
                if r != self._open:
                    odd[r] = odd.get(r, 0) ^ 1
            if any(odd.values()):
                raise RuntimeError("No perfect matching could be found")
            self._solver.clear()
            self._solver.solve(self._fb.SyndromePattern(fired))
            mask = 0
            for e in self._solver.subgraph():
                if self._to_obs:
                    mask ^= self._masks[e]
                else:
                    out[self._cols[e]] = 1.0
            if self._to_obs:
                for o in range(self._out_size):
                    out[o] = float((mask >> o) & 1)
        return _make_result(True, out, {"heralded": False})

    def decode_batch(self, syndromes: Any) -> BatchDecoderResult:
        return _batch([self.decode(s) for s in syndromes])


def _fusion_from_matrix(inputs: _Inputs, output: str, params: dict) -> _NvFusion:
    params = dict(params)
    if "dem_string" in params:
        text = params.pop("dem_string")
        if inputs.O is not None and inputs.num_observables:
            params["O"] = np.array([[1 if j in row else 0 for j in range(inputs.num_cols)] for row in inputs.O], dtype=np.uint8)
        return _fusion_dem(str(text), params, inputs.num_rows, inputs.num_cols)
    rates = inputs.rates
    if rates:
        if len(rates) != inputs.num_cols:
            raise RuntimeError(f"nv-fusion-decoder: error_rate_vec has length {len(rates)} but H has {inputs.num_cols} columns")
        if any(not 0.0 < p <= 0.5 for p in rates):
            raise RuntimeError("nv-fusion-decoder: error_rate_vec value is out of range (0, 0.5]")
    obs_of: List[List[int]] = [[] for _ in range(inputs.num_cols)]
    num_obs = None
    if inputs.has_observable_model() and inputs.num_observables > 0 and output != "errors":
        num_obs = inputs.num_observables
        for o, row in enumerate(inputs.O or []):
            for e in row:
                if e >= inputs.num_cols:
                    raise RuntimeError("nv-fusion-decoder: O must have as many columns as H")
                obs_of[e].append(o)
    if output == "observables" and num_obs is None:
        raise RuntimeError("nv-fusion-decoder was constructed for observable output but its model supplies no observable mapping")
    mechanisms = []
    for j, col in enumerate(inputs.columns):
        if len(col) not in (1, 2):
            raise RuntimeError(f"nv-fusion-decoder: column {j} of H has {len(col)} non-zero rows; every error mechanism must touch one or two detectors (graphlike)")
        mechanisms.append((sorted(col), obs_of[j], _weight(rates[j]) if rates else 1.0))
    return _NvFusion(inputs.num_rows, mechanisms, num_obs, inputs.num_cols, list(range(inputs.num_cols)), params, False)


_fusion_from_matrix._raw_output = True  # (with O and no asked-for output, observables, as CUDA-Q's)


def _fusion_dem(dem_text: str, params: dict, rows: Optional[int] = None, cols: Optional[int] = None, output: Optional[str] = None) -> _NvFusion:
    from .._circuit import DetectorErrorModel as StimDem
    from ._dem import _parse_stim_dem

    _observables_only("nv-fusion-decoder", output)
    try:
        dem = StimDem(dem_text)
    except Exception as e:
        raise RuntimeError(f"Stim DEM parse failed: {e}") from None
    params = dict(params)
    params.pop("error_rate_vec", None)  # (a model carries its own rates)
    num_dets, num_obs = int(dem.num_detectors), int(dem.num_observables)
    if rows is not None and (rows != num_dets or cols != int(dem.flattened().num_errors)):
        raise RuntimeError(f"nv-fusion-decoder: H is {rows} x {cols} but the detector error model has {num_dets} detectors and {int(dem.flattened().num_errors)} errors")
    if "detector_round" not in params:
        coords = dem.get_detector_coordinates()
        for d in range(num_dets):
            if not coords.get(d):
                raise RuntimeError(f"nv-fusion-decoder: detector D{d} has no coordinate in the detector error model; pass detector_round")
    if "O" in params:
        O = np.asarray(params.pop("O"))
        if O.ndim != 2 or O.shape[0] < num_obs:
            raise RuntimeError(f"nv-fusion-decoder: O has fewer rows than the detector error model's {num_obs} observables")
        num_obs = int(O.shape[0])
    _n, _o, det_hits, obs_hits, rates = _parse_stim_dem(dem_text, True)
    merged: Dict[Tuple[Tuple[int, ...], Tuple[int, ...]], float] = {}
    for dets, obs, p in zip(det_hits, obs_hits, rates):
        if not dets or p == 0.0:
            continue
        if len(dets) > 2:
            raise RuntimeError(f"nv-fusion-decoder: the detector error model is not graphlike: an error (after its ^ decomposition) touches {len(dets)} detectors")
        key = (tuple(dets), tuple(obs))
        q = merged.get(key, 0.0)
        merged[key] = q + p - 2.0 * q * p
    mechanisms = []
    for (dets, obs), p in merged.items():
        if p > 0.5:
            raise RuntimeError(f"nv-fusion-decoder: error probability {p} of a mechanism is out of range (0, 0.5]")
        mechanisms.append((list(dets), list(obs), _weight(p)))
    return _NvFusion(num_dets, mechanisms, num_obs, num_obs, None, params, True)

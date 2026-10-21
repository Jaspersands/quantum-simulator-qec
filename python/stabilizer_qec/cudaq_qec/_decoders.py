# Ported from CUDA-Q QEC (NVIDIA/cudaqx, libs/qec: include/cudaq/qec/decoder.h, lib/decoder.cpp,
# lib/decoders/lut.cpp, lib/decoders/plugins/pymatching/pymatching.cpp,
# lib/decoders/plugins/example/single_error_lut_example.cpp, python/bindings/py_decoder.cpp;
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates, Apache License 2.0; see
# THIRD_PARTY_NOTICES.md), modified for stabilizer_qec: the decoders are this package's.
"""CUDA-Q QEC's decoder interface: results, the decoder registry and ``get_decoder``, and its
decoders (``single_error_lut``, ``multi_error_lut``, ``pymatching``, ``sliding_window``,
``nv-qldpc-decoder`` (``._qldpc``), ``chromobius`` and ``nv-fusion-decoder`` (``._matching``),
``trt_decoder`` (``._trt``), the examples)."""

from __future__ import annotations

import concurrent.futures
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .. import _core
from .._util import call

VERSION = "CUDA-Q QEC Base Decoder Interface 0.8.1 (stabilizer_qec)"


# -- Results ------------------------------------------------------------------------------------


class DecoderResult:
    """One decode: whether it converged, the result (an error or observable frame, as floats)
    and optional extra results (a dict, or None)."""

    def __init__(self) -> None:
        self.converged = False
        self._result = np.zeros(0, dtype=np.float64)
        self.opt_results: Optional[dict] = None

    @property
    def result(self) -> np.ndarray:
        return self._result

    @result.setter
    def result(self, value: Any) -> None:
        self._result = np.asarray(value, dtype=np.float64).reshape(-1)

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "converged":
            value = bool(value)
        object.__setattr__(self, name, value)

    def __len__(self) -> int:
        return 3

    def _items(self) -> tuple:
        return (self.converged, self.result, self.opt_results)

    def __getitem__(self, k: int) -> Any:
        return self._items()[k]

    def __iter__(self):
        return iter(self._items())

    def __eq__(self, other: object) -> bool:
        # (As CUDA-Q's: equal convergence and result, and neither carrying extra results.)
        if not isinstance(other, DecoderResult):
            return NotImplemented
        if self.converged != other.converged or not np.array_equal(self.result, other.result):
            return False
        return not self.opt_results and not other.opt_results

    def __repr__(self) -> str:
        return f"DecoderResult(converged={self.converged}, result={self.result.tolist()}, opt_results={self.opt_results})"


def _make_result(converged: bool, result: Any, opt_results: Optional[dict] = None) -> DecoderResult:
    r = DecoderResult()
    r.converged = converged
    r.result = result
    r.opt_results = opt_results
    return r


class BatchDecoderResult:
    """A batch of decodes: ``result`` (shots × width), ``converged`` (per shot),
    ``opt_results`` (per shot) and ``batch_opt_results`` (for the batch, or None)."""

    def __init__(self, result: Any, converged: Any, opt_results: Optional[Sequence] = None, batch_opt_results: Optional[dict] = None) -> None:
        r = np.asarray(result)
        if r.ndim != 2:
            raise TypeError("BatchDecoderResult: result must be a two-dimensional array")
        c = np.asarray(converged)
        if c.ndim != 1:
            raise TypeError("BatchDecoderResult: converged must be a one-dimensional array")
        if c.shape[0] != r.shape[0]:
            raise RuntimeError("BatchDecoderResult: converged row count must match result row count")
        if opt_results is not None and len(opt_results) != r.shape[0]:
            raise RuntimeError("BatchDecoderResult: opt_results length must match the number of rows")
        self._set(np.ascontiguousarray(r, dtype=np.float64), c.astype(np.bool_), list(opt_results) if opt_results is not None else [None] * r.shape[0], batch_opt_results)

    def _set(self, result: np.ndarray, converged: np.ndarray, opt_results: list, batch: Optional[dict]) -> None:
        self.result = result
        self.converged = converged
        self.opt_results = opt_results
        self.batch_opt_results = batch

    def __len__(self) -> int:
        return self.result.shape[0]

    def __getitem__(self, k: Any) -> Any:
        if isinstance(k, slice):
            out = object.__new__(BatchDecoderResult)
            # (Batch-level results describe the whole batch, so a slice drops them.)
            out._set(self.result[k], self.converged[k], self.opt_results[k], None)
            return out
        i = int(k)
        n = len(self)
        if i < -n or i >= n:
            raise IndexError("BatchDecoderResult index out of range")
        i %= n
        return _make_result(bool(self.converged[i]), self.result[i], self.opt_results[i])

    def __iter__(self):
        return (self[i] for i in range(len(self)))


class AsyncDecoderResult:
    """A decode running in the background: ``get()`` waits for it, ``ready()`` says whether it
    has finished."""

    _pool = concurrent.futures.ThreadPoolExecutor(max_workers=4)

    def __init__(self, fn: Callable[[], DecoderResult]) -> None:
        self._future = AsyncDecoderResult._pool.submit(fn)

    def get(self) -> DecoderResult:
        return self._future.result()

    def ready(self) -> bool:
        return self._future.done()


# -- The decoder interface ------------------------------------------------------------------------


def _shape(H: Any) -> Tuple[int, int]:
    shape = getattr(H, "shape", None)
    if shape is None:
        shape = np.asarray(H).shape
    if len(shape) != 2:
        raise RuntimeError("Decoder: H must be a two-dimensional matrix")
    return int(shape[0]), int(shape[1])


class Decoder:
    """CUDA-Q QEC's decoder base class: a decoder for the check matrix ``H`` (syndrome size its
    rows, block size its columns). Python decoders registered with ``@decoder(name)`` derive from
    it and call ``Decoder.__init__(self, H)``."""

    def __init__(self, H: Any) -> None:
        self._syndrome_size, self._block_size = _shape(H)

    def get_block_size(self) -> int:
        return self._block_size

    def get_syndrome_size(self) -> int:
        return self._syndrome_size

    def get_version(self) -> str:
        return VERSION

    def decode(self, syndrome: Any) -> DecoderResult:  # pragma: no cover - overridden
        raise NotImplementedError

    def decode_batch(self, syndromes: Any) -> BatchDecoderResult:
        results = [self.decode(s) for s in syndromes]
        return _batch(results)

    def decode_async(self, syndrome: Any) -> AsyncDecoderResult:
        decoder, s = self, list(np.asarray(syndrome, dtype=np.float64).reshape(-1))
        return AsyncDecoderResult(lambda: decoder.decode(s))


def _batch(results: List[DecoderResult]) -> BatchDecoderResult:
    if not results:
        return BatchDecoderResult(np.zeros((0, 0)), np.zeros(0, dtype=bool))
    return BatchDecoderResult(np.stack([np.asarray(r.result, dtype=np.float64) for r in results]), np.array([r.converged for r in results], dtype=bool), [r.opt_results for r in results])


_PY_REGISTRY: Dict[str, Callable[..., Any]] = {}


def decoder(name: str) -> Callable[[type], type]:
    """Register a Python decoder class under ``name`` (for ``get_decoder``). The class is
    rebuilt deriving from ``Decoder``; it must define ``decode``, and a ``decode_batch`` of its
    own must return a ``BatchDecoderResult``."""

    def register(cls: type) -> type:
        if not hasattr(cls, "decode"):
            raise RuntimeError("Decoder class must implement decode method")
        namespace = {k: v for k, v in vars(cls).items() if k not in ("__dict__", "__weakref__")}
        own_batch = namespace.get("decode_batch")
        if own_batch is not None:

            def decode_batch(self: Any, syndromes: Any, _f: Callable = own_batch) -> BatchDecoderResult:
                out = _f(self, syndromes)
                if not isinstance(out, BatchDecoderResult):
                    raise TypeError(f"{cls.__name__}.decode_batch must return a BatchDecoderResult, not {type(out).__name__}")
                return out

            namespace["decode_batch"] = decode_batch
        new = type(cls.__name__, (Decoder,), namespace)
        _PY_REGISTRY[name] = lambda H, **options: new(H, **options)
        return new

    return register


# -- Model inputs ---------------------------------------------------------------------------------


def _columns(H: Any) -> Tuple[int, int, List[List[int]]]:
    """A check matrix's rows, columns and each column's rows (canonical: sorted, explicit
    zeros dropped, duplicates summed mod 2... as nonzero sums)."""
    if hasattr(H, "tocsc"):
        csc = H.tocsc(copy=True)
        csc.sum_duplicates()
        csc.eliminate_zeros()
        csc.sort_indices()
        m, n = csc.shape
        cols = [[int(r) for r, v in zip(csc.indices[csc.indptr[j] : csc.indptr[j + 1]], csc.data[csc.indptr[j] : csc.indptr[j + 1]]) if v != 0] for j in range(n)]
        return int(m), int(n), cols
    a = np.asarray(H)
    if a.ndim != 2:
        raise RuntimeError("get_decoder: H must be a two-dimensional matrix")
    m, n = a.shape
    nz = a != 0
    return int(m), int(n), [np.flatnonzero(nz[:, j]).tolist() for j in range(n)]


class _Inputs:
    """``decoder_init``: H as columns, the observable matrix (rows of faults) if any, and the
    error rates; ``obs_cols``, the column count of an O given as a matrix (None otherwise);
    ``obs_rank``, the rank of an O given that is not a matrix, for a decoder that refuses it
    itself (trt_decoder; None otherwise)."""

    def __init__(self, num_rows: int, num_cols: int, columns: List[List[int]], O: Optional[List[List[int]]], num_obs: int, rates: List[float], obs_cols: Optional[int] = None, obs_rank: Optional[int] = None) -> None:
        self.num_rows, self.num_cols, self.columns = num_rows, num_cols, columns
        self.O, self.num_observables, self.rates = O, num_obs, rates
        self.obs_cols, self.obs_rank = obs_cols, obs_rank

    def has_observable_model(self) -> bool:
        return self.O is not None


def _rows_of(O: Any) -> Tuple[int, int, List[List[int]]]:
    m, n, cols = _columns(O)
    rows: List[List[int]] = [[] for _ in range(m)]
    for j, col in enumerate(cols):
        for r in col:
            rows[r].append(j)
    return m, n, rows


def _param(value: Any) -> Any:
    """A keyword argument as CUDA-Q's heterogeneous map takes it (refusing what it refuses)."""
    if isinstance(value, bool) or value is None or isinstance(value, (str, float)):
        return value
    if isinstance(value, (int, np.integer)):
        if value < 0:
            raise RuntimeError("std::bad_cast")
        return int(value)
    if isinstance(value, dict):
        return {k: _param(v) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        if value.dtype not in (np.float64, np.float32, np.uint8, np.int32, np.int64, np.bool_):
            raise RuntimeError(f"Unsupported array data type {value.dtype} for decoder parameter")
        return value
    if isinstance(value, (list, tuple)):
        return list(value)
    return value


# -- Native decoders ------------------------------------------------------------------------------


def _hard(syndrome: Any) -> np.ndarray:
    return np.asarray(syndrome, dtype=np.float64).reshape(-1) >= 0.5


class _NativeDecoder(Decoder):
    """A decoder built by ``get_decoder`` from a model (``decoder_init``)."""

    def __init__(self, inputs: _Inputs, output: str, params: dict) -> None:
        self._syndrome_size, self._block_size = inputs.num_rows, inputs.num_cols
        self._inputs, self._output = inputs, output

    def _project(self, errors: np.ndarray) -> np.ndarray:
        if not self._inputs.has_observable_model():
            raise RuntimeError("decoder was asked to project an error frame onto observables but its model supplies no observable mapping")
        hard = errors >= 0.5
        return np.array([float(np.count_nonzero(hard[row]) % 2) for row in self._inputs.O], dtype=np.float64)


_LUT_RESULTS = ("error_probability", "syndrome_weight", "decoding_time")


class _MultiErrorLut(_NativeDecoder):
    """``multi_error_lut`` (and ``single_error_lut``): every set of up to ``lut_error_depth``
    errors, the first set (by size, then lexicographically) for each syndrome."""

    def __init__(self, inputs: _Inputs, output: str, params: dict) -> None:
        super().__init__(inputs, output, params)
        if output == "observables" and not inputs.has_observable_model():
            raise RuntimeError("lut decoder was constructed for observable output but its model supplies no observable mapping")
        depth = 1
        if "lut_error_depth" in params:
            depth = int(params["lut_error_depth"])
        self._opt: Dict[str, bool] = {}
        if "opt_results" in params:
            requested = params["opt_results"]
            if not isinstance(requested, dict):
                raise RuntimeError("opt_results must be a heterogeneous_map")
            invalid = [k for k in requested if k not in _LUT_RESULTS]
            if invalid:
                raise RuntimeError("Requested result types not available in LUT decoder: " + ", ".join(invalid))
            self._opt = {k: bool(requested.get(k, False)) for k in _LUT_RESULTS}
        try:
            self._table = call(_core.LookupTable, inputs.num_rows, inputs.columns, depth, [float(p) for p in inputs.rates])
        except ValueError as e:
            raise RuntimeError(str(e)) from None  # (std::runtime_error, as CUDA-Q raises them)

    def decode(self, syndrome: Any) -> DecoderResult:
        t0 = time.perf_counter()
        hard = _hard(syndrome)
        fired = np.flatnonzero(hard).tolist()
        errors = np.zeros(self._block_size)
        converged, found = False, None
        if not fired:
            converged = True
        else:
            found = self._table.lookup(fired)
            if found is not None:
                converged = True
                errors[found[0]] = 1.0
        opt = None
        if self._opt:
            opt = {}
            if self._opt["error_probability"]:
                opt["error_probability"] = found[1] if found is not None else 0.0
            if self._opt["syndrome_weight"]:
                opt["syndrome_weight"] = len(fired)
            if self._opt["decoding_time"]:
                opt["decoding_time"] = time.perf_counter() - t0
            if not fired:
                opt = None  # (CUDA-Q returns before filling them for an empty syndrome)
        result = self._project(errors) if self._output == "observables" else errors
        return _make_result(converged, result, opt)


class _SingleErrorLutExample(_NativeDecoder):
    """CUDA-Q's example plugin: a table of single errors, the first error kept per syndrome."""

    def __init__(self, inputs: _Inputs, output: str, params: dict) -> None:
        super().__init__(inputs, output, params)
        if output != "errors":
            raise RuntimeError("single_error_lut_example produces an error frame only; construct it for error output")
        self._table: Dict[tuple, int] = {}
        for e, col in enumerate(inputs.columns):
            self._table.setdefault(tuple(sorted(set(col))), e)

    def decode(self, syndrome: Any) -> DecoderResult:
        fired = tuple(np.flatnonzero(_hard(syndrome)).tolist())
        errors = np.zeros(self._block_size)
        if not fired:
            return _make_result(True, errors)
        e = self._table.get(fired)
        if e is None:
            return _make_result(False, errors)
        errors[e] = 1.0
        return _make_result(True, errors)


_MERGES = {"disallow": "disallow", "independent": "independent", "smallest_weight": "smallest-weight", "keep_original": "keep-original", "replace": "replace"}


class _PyMatchingPlugin(_NativeDecoder):
    """CUDA-Q's ``pymatching`` plugin: each column an edge (or boundary edge) of weight
    ln((1 − p)/p) (1 without rates); an error frame from the matched edges' columns, or the
    observables."""

    def __init__(self, inputs: _Inputs, output: str, params: dict) -> None:
        from .. import pymatching

        super().__init__(inputs, output, params)
        rates = inputs.rates
        if rates:
            if len(rates) != inputs.num_cols:
                raise RuntimeError("error_rate_vec must be of size block_size")
            if any(p <= 0.0 or p > 0.5 for p in rates):
                raise RuntimeError("error_rate_vec value is out of range (0, 0.5]")
        merge = "disallow"
        explicit = "merge_strategy" in params
        if explicit:
            if params["merge_strategy"] not in _MERGES:
                raise RuntimeError("merge_strategy must be one of: disallow, independent, smallest_weight, keep_original, replace")
            merge = params["merge_strategy"]
        self._to_obs = output == "observables"
        errs_to_obs: List[List[int]] = [[] for _ in range(inputs.num_cols)]
        if self._to_obs:
            for o, row in enumerate(inputs.O or []):
                for e in row:
                    errs_to_obs[e].append(o)
            if not explicit:
                merge = "independent"
        self._m = pymatching.Matching()
        self._edge_col: Dict[Tuple[int, int], int] = {}
        edge_weight: Dict[Tuple[int, int], float] = {}
        for col, rows in enumerate(inputs.columns):
            w = 1.0
            if col < len(rates):
                w = -np.log(rates[col] / (1.0 - rates[col]))
            if len(rows) not in (1, 2):
                raise RuntimeError(f"Invalid column in H: {col} has {len(rows)} ones. Must have 1 or 2 ones.")
            edge = (min(rows), max(rows)) if len(rows) == 2 else (-1, rows[0])
            if not self._to_obs:
                if edge not in self._edge_col:
                    self._edge_col[edge] = col
                    edge_weight[edge] = w
                elif merge == "replace" or (merge == "smallest_weight" and w < edge_weight[edge]):
                    self._edge_col[edge] = col
                    edge_weight[edge] = w
            faults = set(errs_to_obs[col])
            try:
                if len(rows) == 2:
                    self._m.add_edge(rows[0], rows[1], fault_ids=faults, weight=w, error_probability=0.0, merge_strategy=_MERGES[merge])
                else:
                    self._m.add_boundary_edge(rows[0], fault_ids=faults, weight=w, error_probability=0.0, merge_strategy=_MERGES[merge])
            except ValueError as e:
                raise RuntimeError(str(e)) from None

    def decode(self, syndrome: Any) -> DecoderResult:
        hard = _hard(syndrome)
        n = self._m.num_detectors
        if hard[n:].any():
            # (A detection event on a detector no edge reaches.)
            raise RuntimeError("No perfect matching could be found")
        det = hard[:n].astype(np.uint8)
        if self._to_obs:
            out = np.zeros(self._inputs.num_observables)
            if n:
                obs = np.asarray(self._m.decode(det), dtype=np.float64)
                k = min(obs.size, out.size)
                out[:k] = obs[:k]
            return _make_result(True, out)
        errors = np.zeros(self._block_size)
        if det.any():
            for a, b in self._m.decode_to_edges_array(det):
                a, b = int(a), int(b)
                edge = (min(a, b), max(a, b)) if a != -1 and b != -1 else (-1, max(a, b))
                errors[self._edge_col[edge]] = 1.0
        return _make_result(True, errors)


class _SlidingWindow(_NativeDecoder):
    """CUDA-Q's ``sliding_window``: the rounds decoded ``window_size`` at a time, sliding by
    ``step_size``; each window by its own inner decoder (``inner_decoder_name`` with
    ``inner_decoder_params``) on the columns the window holds (and those straddling its start or
    end, as asked); the columns before the next window committed and their detectors in the next
    window flipped. A syndrome may come whole or a round at a time (the result comes with the
    last round). The matrix must be sorted (``simplify_pcm``)."""

    def __init__(self, inputs: _Inputs, output: str, params: dict) -> None:
        from ._pcm import _layout, _round_order, get_pcm_for_rounds

        super().__init__(inputs, output, params)
        if output == "observables" and not inputs.has_observable_model():
            raise ValueError("sliding_window was constructed for observable output but its model supplies no observable mapping")
        self.window_size = int(params.get("window_size", 1))
        self.step_size = int(params.get("step_size", 1))
        S = int(params.get("num_syndromes_per_round", 0))
        B = int(params.get("num_boundary_syndromes", 0))
        straddle_start = bool(params.get("straddle_start_round", False))
        straddle_end = bool(params.get("straddle_end_round", True))
        name = params.get("inner_decoder_name", "")
        inner_params = dict(params.get("inner_decoder_params", {}) or {})
        if S == 0:
            raise ValueError("sliding_window constructor: num_syndromes_per_round must be non-zero")
        if B == 0:
            B = S
        m = inputs.num_rows
        _boundary, layers, self._start, _ = _layout(S, B, m)
        self._layers, self._S, self._B = layers, S, B
        if B > S:
            raise ValueError("sliding_window constructor: num_boundary_syndromes must be <= num_syndromes_per_round")
        if m < 2 * S:
            raise ValueError("sliding_window constructor: num_rows must be >= 2 * num_syndromes_per_round; memory circuits always produce at least a lock-in and a read-out round")
        if m < 2 * B or (m - 2 * B) % S != 0:
            raise ValueError("sliding_window constructor: number of PCM rows is inconsistent with the given num_syndromes_per_round and num_boundary_syndromes")
        if not 1 <= self.window_size <= layers:
            raise ValueError(f"sliding_window constructor: window_size ({self.window_size}) must be between 1 and the number of detector layers ({layers})")
        if not 1 <= self.step_size <= self.window_size:
            raise ValueError(f"sliding_window constructor: step_size ({self.step_size}) must be between 1 and window_size ({self.window_size})")
        if (layers - self.window_size) % self.step_size != 0:
            raise ValueError(f"sliding_window constructor: detector layers - window_size ({layers - self.window_size}) must be divisible by step_size ({self.step_size})")
        if not name:
            raise ValueError("sliding_window constructor: inner_decoder_name must be non-empty")
        if not inputs.rates:
            raise ValueError("sliding_window constructor: error_rate_vec must be non-empty")
        if _round_order(inputs.columns, S, B) != list(range(inputs.num_cols)):
            raise ValueError("sliding_window constructor: PCM must be sorted. See cudaq::qec::simplify_pcm.")
        dense = np.zeros((m, inputs.num_cols), dtype=np.uint8)
        for j, col in enumerate(inputs.columns):
            dense[col, j] = 1
        self._windows = (layers - self.window_size) // self.step_size + 1
        self._first: List[int] = []
        self._inner: List[_NativeDecoder] = []
        for w in range(self._windows):
            start = w * self.step_size
            sub, first, last = get_pcm_for_rounds(dense, S, start, start + self.window_size - 1, straddle_start, straddle_end, B if B != S else 0)
            if last - first + 1 != sub.shape[1]:
                raise ValueError(f"last_column - first_column + 1 ({last - first + 1}) must be equal to the number of columns in H_round ({sub.shape[1]})")
            self._first.append(first)
            sm, sn, scols = _columns(sub)
            self._inner.append(_native(name, _Inputs(sm, sn, scols, [], 0, list(inputs.rates[first : last + 1])), dict(inner_params, output="errors")))
        self._reset()

    def _reset(self) -> None:
        self._rounds: List[List[np.ndarray]] = []
        self._received = 0
        self._decoded = 0
        self._mods: List[np.ndarray] = []
        self._results: List[List[Any]] = []

    def reset_decoder(self) -> None:
        self._reset()

    def get_num_syndromes_per_round(self) -> int:
        return self._S

    def get_num_boundary_syndromes(self) -> int:
        return self._B

    def get_num_detector_layers(self) -> int:
        return self._layers

    def get_layer_offset(self, r: int) -> int:
        return self._start(r)

    def _rows(self, syndromes: List[np.ndarray]) -> List[DecoderResult]:
        if not syndromes:
            return []
        if syndromes[0].size == self._syndrome_size:
            out: List[DecoderResult] = []
            for r in range(self._layers):
                a, b = self._start(r), self._start(r + 1)
                out = self._rows([s[a:b] for s in syndromes])
            return out
        if self._received == 0:
            self._mods = [np.zeros(self._syndrome_size, dtype=bool) for _ in syndromes]
            self._results = [[True, np.zeros(self._block_size)] for _ in syndromes]
            self._batch = len(syndromes)
        if len(syndromes) != self._batch:
            raise ValueError(f"sliding_window: batch size changed mid-stream ({len(syndromes)} vs {self._batch})")
        expected = self._start(self._received + 1) - self._start(self._received)
        for row in syndromes:
            if row.size != expected:
                raise ValueError(f"sliding_window: round {self._received} has width {row.size} but expected {expected} for this round in the boundary layout")
        self._rounds.append(syndromes)
        self._received += 1
        if len(self._rounds) < self.window_size:
            return []
        self._window()
        self._decoded += 1
        if self._decoded == self._windows:
            results = [_make_result(c, self._project(r) if self._output == "observables" else r) for c, r in self._results]
            self._reset()
            return results
        del self._rounds[: self.step_size]
        return []

    def _window(self) -> None:
        w = self._decoded
        start = self._start(w * self.step_size)
        count = self._start(w * self.step_size + self.window_size) - start
        batch = []
        for s in range(self._batch):
            syn = np.concatenate([self._rounds[k][s] for k in range(self.window_size)])
            if w > 0:
                syn = (syn.astype(np.uint8) ^ self._mods[s][start : start + count]).astype(np.float64)
            batch.append(syn)
        inner = self._inner[w].decode_batch(batch)
        first = self._first[w]
        for s in range(self._batch):
            self._results[s][0] = self._results[s][0] and bool(inner.converged[s])
            hard = (inner.result[s] >= 0.5).astype(np.float64)
            if w < self._windows - 1:
                n = self._first[w + 1] - first
                self._results[s][1][first : first + n] = hard[:n]
                lo, hi = self._start((w + 1) * self.step_size), self._start((w + 1) * self.step_size + 1) - 1
                for c in range(n):
                    if self._results[s][1][first + c]:
                        for r in self._inputs.columns[first + c]:
                            if lo <= r <= hi:
                                self._mods[s][r] ^= True
            else:
                self._results[s][1][first : first + hard.size] = hard

    def decode(self, syndrome: Any) -> DecoderResult:
        out = self._rows([np.asarray(syndrome, dtype=np.float64).reshape(-1)])
        return out[0] if out else DecoderResult()

    def decode_batch(self, syndromes: Any) -> BatchDecoderResult:
        return _batch(self._rows([np.asarray(s, dtype=np.float64).reshape(-1) for s in syndromes]))


_NATIVE: Dict[str, Callable[[_Inputs, str, dict], _NativeDecoder]] = {
    "single_error_lut": _MultiErrorLut,
    "multi_error_lut": _MultiErrorLut,
    "single_error_lut_example": _SingleErrorLutExample,
    "pymatching": _PyMatchingPlugin,
    "sliding_window": _SlidingWindow,
}


# Decoders built from a Stim model's text itself (``get_decoder(name, dem_text)``), not its matrices.
_NATIVE_DEM: Dict[str, Callable[[str, dict, Optional[str]], _NativeDecoder]] = {}


def _output_of(options: dict) -> Optional[str]:
    output = options.pop("output", None)
    if output is not None and output not in ("errors", "observables"):
        raise RuntimeError(f"output must be 'errors' or 'observables', not {output!r}")
    return output


def _invalid(name: str) -> RuntimeError:
    return RuntimeError(f"invalid decoder requested: {name}. Run with CUDAQ_LOG_LEVEL=info (environment variable) to see additional plugin diagnostics at startup.")


def _no_cuda(params: dict) -> None:
    if "cuda_device_id" in params:
        raise RuntimeError(f"cuda_device_id {params['cuda_device_id']} is out of range: 0 CUDA device(s) visible")


def _native(name: str, inputs: _Inputs, options: dict, typed: bool = False) -> _NativeDecoder:
    output = _output_of(options)
    for reserved in ("H", "O", "D", "error_rate_vec"):
        if reserved in options:
            raise RuntimeError(f"'{reserved}' is framework model data; provide it through decoder_init instead of decoder custom parameters")
    # (``typed``: values already as a decoder's parameter schema types them, from a configuration.)
    params = dict(options) if typed else {k: _param(v) for k, v in options.items()}
    factory = _NATIVE.get(name)
    if factory is None:
        raise _invalid(name)
    _no_cuda(params)
    # (A factory with ``_raw_output`` tells an unset output, None, from an asked-for one.)
    return factory(inputs, output if getattr(factory, "_raw_output", False) else (output or "errors"), params)


def _native_dem(name: str, dem_text: str, options: dict, typed: bool = False) -> _NativeDecoder:
    output = _output_of(options)
    params = dict(options) if typed else {k: _param(v) for k, v in options.items()}
    _no_cuda(params)
    return _NATIVE_DEM[name](dem_text, params, output)


def get_decoder(name: str, H: Any, **options: Any) -> Any:
    """A decoder by name, for the check matrix ``H`` (a dense array or a scipy sparse matrix)
    or a Stim detector error model given as text. ``O`` (the observable matrix) and
    ``error_rate_vec`` describe the model; ``output`` ("errors" or "observables") chooses what
    a decode returns; the other keywords configure the decoder."""
    return _get_decoder(name, H, options)


def _get_decoder(name: str, H: Any, options: dict, typed: bool = False) -> Any:
    from ._dem import _parse_stim_dem, dem_from_stim_text

    options = dict(options)
    if isinstance(H, str):
        if name in _PY_REGISTRY:
            dem = dem_from_stim_text(H)
            if "O" not in options and dem.num_observables() > 0:
                options["O"] = dem.observables_flips_matrix
            if "error_rate_vec" not in options and dem.error_rates:
                options["error_rate_vec"] = np.asarray(dem.error_rates)
            return _PY_REGISTRY[name](dem.detector_error_matrix, **options)
        if name in _NATIVE_DEM:
            return _native_dem(name, H, dict(options), typed)
        num_dets, num_obs, det_hits, obs_hits, rates = _parse_stim_dem(H, False)
        O_rows: List[List[int]] = [[] for _ in range(num_obs)]
        for e, hits in enumerate(obs_hits):
            for o in hits:
                O_rows[o].append(e)
        return _native(name, _Inputs(num_dets, len(det_hits), [list(h) for h in det_hits], O_rows, num_obs, list(rates)), dict(options), typed)
    if name in _PY_REGISTRY:
        return _PY_REGISTRY[name](H, **options)
    m, n, cols = _columns(H)
    options = dict(options)
    O = None
    num_obs = 0
    o_cols: Optional[int] = None
    o_rank: Optional[int] = None
    if "O" in options:
        given = options.pop("O")
        if name == "trt_decoder" and not hasattr(given, "tocsc") and np.ndim(given) != 2:
            # (trt_decoder refuses an O that is not a matrix itself, where CUDA-Q's does.)
            o_rank = int(np.ndim(given))
        else:
            num_obs, o_cols, O = _rows_of(given)
        if o_cols != n and name == "pymatching" and options.get("output") == "observables":
            raise RuntimeError("Observable matrix column count must equal block_size")
    rates: List[float] = []
    if "error_rate_vec" in options:
        rates = [float(x) for x in np.asarray(options.pop("error_rate_vec"), dtype=np.float64).reshape(-1)]
    return _native(name, _Inputs(m, n, cols, O, num_obs, rates, o_cols, o_rank), options, typed)


def _register_plugins() -> None:
    from . import _matching, _qldpc, _trt

    _NATIVE["nv-qldpc-decoder"] = _qldpc._NvQldpc

    _NATIVE["chromobius"] = _matching._chromobius_from_matrix
    _NATIVE_DEM["chromobius"] = _matching._Chromobius
    _NATIVE["nv-fusion-decoder"] = _matching._fusion_from_matrix
    _NATIVE_DEM["nv-fusion-decoder"] = lambda text, params, output: _matching._fusion_dem(text, params, output=output)
    _NATIVE["trt_decoder"] = _trt._TrtDecoder


_register_plugins()

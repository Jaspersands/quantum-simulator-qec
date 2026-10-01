"""Decoders: on a detector error model (matching, belief-matching, BP+OSD, windows), and on a
check matrix (BP and BP+OSD, with ``ldpc``'s names)."""

from __future__ import annotations

from typing import Any, NamedTuple, Union

import numpy as np

from . import _core
from ._circuit import DetectorErrorModel
from ._util import call, count, probability, real, rows_to_b8, u64_to_rows

ModelLike = Union[DetectorErrorModel, str, Any]


def _model(model: ModelLike) -> DetectorErrorModel:
    return model if isinstance(model, DetectorErrorModel) else DetectorErrorModel(model)


class _DemDecoder:
    """What every decoder on a detector error model shares: shots in, observable flips out."""

    __slots__ = ("_x",)

    @property
    def num_detectors(self) -> int:
        return self._x.num_detectors

    @property
    def num_observables(self) -> int:
        return self._x.num_observables

    def _shots(self, shots: Any, bit_packed_shots: bool, threads: int) -> tuple[bytes, int, int]:
        packed, n = rows_to_b8(shots, self.num_detectors, bit_packed_shots, "shots")
        return packed, n, count(threads, "threads")

    def _predictions(self, raw: bytes, n: int, bit_packed: bool) -> np.ndarray:
        return u64_to_rows(raw, n, self.num_observables, bit_packed)

    def decode(self, syndrome: Any) -> np.ndarray:
        """One shot: its detection events (length ``num_detectors``) to the observables it
        predicts flipped, as uint8 0/1 (length ``num_observables``)."""
        s = np.asarray(syndrome)
        if s.ndim != 1:
            raise ValueError(f"a syndrome is 1-dimensional, not {s.ndim}-dimensional; use decode_batch for many")
        return self.decode_batch(s.reshape(1, -1))[0]

    def decode_batch(self, shots: Any, **kwargs: Any) -> Any:  # pragma: no cover - overridden
        raise NotImplementedError


def _failed(failed: list, what: str) -> None:
    if failed:
        more = f" (and {len(failed) - 1} more)" if len(failed) > 1 else ""
        raise ValueError(f"shot {failed[0]}{more}: {what}")


class Matching(_DemDecoder):
    """Exact minimum-weight perfect matching on a decomposed detector error model, by sparse
    blossom, as PyMatching decodes: edges weighted ln((1 − p)/p), the observables of the
    lightest correction returned. ``enable_correlations`` adds PyMatching 2.4's second pass,
    which reweights the graph by the edges the first pass used.

    The model's faults must flip at most two detectors each, or be decomposed into such
    pieces (``Circuit.detector_error_model(decompose_errors=True)``).
    """

    __slots__ = ()

    def __init__(self, model: ModelLike, *, enable_correlations: bool = False) -> None:
        self._x = call(_core.Matcher, _model(model)._d, bool(enable_correlations))

    @classmethod
    def from_detector_error_model(cls, model: ModelLike, *, enable_correlations: bool = False) -> "Matching":
        return cls(model, enable_correlations=enable_correlations)

    def decode_batch(
        self,
        shots: Any,
        *,
        return_weights: bool = False,
        bit_packed_shots: bool = False,
        bit_packed_predictions: bool = False,
        threads: int = 1,
    ) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray]]:
        """Shots (shots × detectors, bool or 0/1; or bit-packed uint8 rows) to predictions
        (shots × observables, uint8; or bit-packed). ``return_weights`` also returns each
        matching's weight. A shot no matching explains raises ``ValueError``."""
        packed, n, threads = self._shots(shots, bit_packed_shots, threads)
        preds, weights, failed = call(self._x.decode_batch, packed, n, threads)
        _failed(failed, "no matching explains its detection events (a defect with no path to a partner or the boundary)")
        out = self._predictions(preds, n, bit_packed_predictions)
        if return_weights:
            return out, np.frombuffer(weights, dtype="<f8").copy()
        return out


class BeliefMatching(_DemDecoder):
    """Belief-matching (Higgott et al. 2023), as the ``beliefmatching`` package decodes:
    belief propagation on the model's full hypergraph; where BP's own correction explains the
    syndrome it is used, and otherwise matching runs on weights −ln p from BP's posteriors.
    The model must be decomposed (``decompose_errors=True``)."""

    __slots__ = ()

    def __init__(
        self,
        model: ModelLike,
        *,
        max_bp_iters: int = 20,
        bp_method: str = "product_sum",
        ms_scaling_factor: float = 1.0,
    ) -> None:
        self._x = call(_core.BeliefMatcher, _model(model)._d, count(max_bp_iters, "max_bp_iters"), bp_method, real(ms_scaling_factor, "ms_scaling_factor"))

    def decode_batch(
        self,
        shots: Any,
        *,
        return_converged: bool = False,
        bit_packed_shots: bool = False,
        bit_packed_predictions: bool = False,
        threads: int = 1,
    ) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray]]:
        """As ``Matching.decode_batch``; ``return_converged`` also returns, per shot, whether
        BP's own correction was used."""
        packed, n, threads = self._shots(shots, bit_packed_shots, threads)
        preds, _weights, conv, failed = call(self._x.decode_batch, packed, n, threads)
        _failed(failed, "no correction explains its detection events")
        out = self._predictions(preds, n, bit_packed_predictions)
        if return_converged:
            return out, np.frombuffer(conv, dtype=np.uint8).astype(bool)
        return out


class BpOsd(_DemDecoder):
    """BP+OSD on a detector error model (Roffe et al.), each fault a column with its prior,
    corrections equal to ``ldpc``'s ``BpOsdDecoder`` on the same matrix but for ties among
    equally likely columns. For codes whose faults flip three or more detectors, such as
    IBM's bivariate bicycle codes: give it the undecomposed model."""

    __slots__ = ()

    def __init__(
        self,
        model: ModelLike,
        *,
        max_iter: int = 10_000,
        bp_method: str = "minimum_sum",
        ms_scaling_factor: float = 0.0,
        osd_method: str = "osd_cs",
        osd_order: int = 7,
    ) -> None:
        self._x = call(
            _core.DemBpOsd,
            _model(model)._d,
            count(max_iter, "max_iter"),
            bp_method,
            real(ms_scaling_factor, "ms_scaling_factor"),
            osd_method,
            count(osd_order, "osd_order"),
        )

    def decode_batch(
        self,
        shots: Any,
        *,
        return_converged: bool = False,
        bit_packed_shots: bool = False,
        bit_packed_predictions: bool = False,
        threads: int = 1,
    ) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray]]:
        """As ``Matching.decode_batch``; ``return_converged`` also returns, per shot, whether
        BP converged before OSD."""
        packed, n, threads = self._shots(shots, bit_packed_shots, threads)
        preds, conv = call(self._x.decode_batch, packed, n, threads)
        out = self._predictions(preds, n, bit_packed_predictions)
        if return_converged:
            return out, np.frombuffer(conv, dtype=np.uint8).astype(bool)
        return out


class Window(NamedTuple):
    """A window of a window decoder: it decodes rounds ``[first_layer, end_layer)`` and
    commits the corrections in ``[commit_start, commit_end)``. Windows of phase 0 run first;
    ``waits_for`` lists the windows whose commits it needs."""

    first_layer: int
    end_layer: int
    commit_start: int
    commit_end: int
    phase: int
    waits_for: list


class WindowMatching(_DemDecoder):
    """Window decoding (Skoric et al.; Tan et al.): the model cut into overlapping windows of
    rounds, each matched alone and committing only its middle. ``mode="sliding"`` runs them
    in order; ``"parallel"`` decodes alternate windows at once, then the gaps between. Each
    window commits ``commit`` rounds with ``buffer`` rounds on either side. The model's
    detectors need a time coordinate (their last), as Stim's generated circuits give them."""

    __slots__ = ()

    def __init__(
        self,
        model: ModelLike,
        *,
        commit: int,
        buffer: int,
        mode: str = "parallel",
        enable_correlations: bool = False,
    ) -> None:
        self._x = call(_core.WindowMatcher, _model(model)._d, count(commit, "commit"), count(buffer, "buffer"), mode, bool(enable_correlations))

    @property
    def windows(self) -> list:
        """The windows, as ``Window`` named tuples."""
        return [Window(*w) for w in self._x.windows()]

    def decode_batch(
        self,
        shots: Any,
        *,
        return_timings: bool = False,
        bit_packed_shots: bool = False,
        bit_packed_predictions: bool = False,
        threads: int = 1,
    ) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray]]:
        """As ``Matching.decode_batch``; ``return_timings`` also returns each window's decode
        time per shot in seconds (shots × windows)."""
        packed, n, threads = self._shots(shots, bit_packed_shots, threads)
        preds, unexplained, times, failed = call(self._x.decode_batch, packed, n, threads, bool(return_timings))
        _failed(failed, "a window found no matching")
        if unexplained:
            raise RuntimeError(f"internal error in stabilizer-qec: window commits left {unexplained} defects unexplained; please report it")
        out = self._predictions(preds, n, bit_packed_predictions)
        if return_timings:
            return out, np.frombuffer(times, dtype="<f8").reshape(n, len(self.windows)).copy()
        return out


def _columns(pcm: Any) -> tuple[int, int, list]:
    """A check matrix (dense array-like or scipy sparse) as (rows, columns, each column's rows)."""
    if hasattr(pcm, "tocsc"):
        m = pcm.tocsc()
        m.eliminate_zeros()
        rows, cols = m.shape
        cols_list = [sorted(int(r) for r in m.indices[m.indptr[j] : m.indptr[j + 1]]) for j in range(cols)]
        return rows, cols, cols_list
    a = np.asarray(pcm)
    if a.ndim != 2:
        raise ValueError(f"a check matrix is 2-dimensional, not {a.ndim}-dimensional")
    if not (np.issubdtype(a.dtype, np.integer) or np.issubdtype(a.dtype, np.bool_)):
        raise TypeError(f"a check matrix holds 0/1 integers or bools, not {a.dtype}")
    a = (a % 2) != 0
    return a.shape[0], a.shape[1], [list(map(int, np.flatnonzero(a[:, j]))) for j in range(a.shape[1])]


def _channel(n: int, error_rate: Any, error_channel: Any) -> list:
    if (error_rate is None) == (error_channel is None):
        raise ValueError("give exactly one of error_rate and error_channel")
    if error_channel is None:
        return [probability(error_rate, "error_rate")] * n
    try:
        raw = np.asarray(error_channel, dtype=float).ravel()
    except (TypeError, ValueError, OverflowError):
        raise TypeError("error_channel must be a list of probabilities") from None
    ch = [probability(p, "error_channel entries") for p in raw]
    if len(ch) != n:
        raise ValueError(f"error_channel has {len(ch)} entries for {n} columns")
    return ch


def _syndrome(syndrome: Any, m: int) -> list:
    s = np.asarray(syndrome)
    if s.ndim != 1 or s.shape[0] != m:
        raise ValueError(f"a syndrome has {m} bits, one per check, not shape {s.shape}")
    return [int(x) for x in (s != 0)]


class BpDecoder:
    """Belief propagation on a check matrix, flooding schedule, its arithmetic reproducing
    ``ldpc``'s ``BpDecoder`` bit for bit. ``bp_method`` is ``"product_sum"`` or
    ``"minimum_sum"`` (with ``ms_scaling_factor``; 0 is ``ldpc``'s adaptive scaling).
    After ``decode``, ``converge``, ``iter`` and ``log_prob_ratios`` describe the run."""

    def __init__(
        self,
        pcm: Any,
        error_rate: Union[float, None] = None,
        error_channel: Any = None,
        *,
        max_iter: int = 20,
        bp_method: str = "product_sum",
        ms_scaling_factor: float = 1.0,
    ) -> None:
        self._m, self._n, cols = _columns(pcm)
        self._x = call(_core.Bp, self._m, cols, _channel(self._n, error_rate, error_channel), count(max_iter, "max_iter"), bp_method, real(ms_scaling_factor, "ms_scaling_factor"))
        self.converge = False
        self.iter = 0
        self.log_prob_ratios = np.zeros(self._n)

    def decode(self, syndrome: Any) -> np.ndarray:
        """The hard decision (uint8 per column)."""
        hard, llr, converged, iterations = call(self._x.decode, _syndrome(syndrome, self._m))
        self.converge, self.iter = bool(converged), int(iterations)
        self.log_prob_ratios = np.asarray(llr, dtype=float)
        return np.asarray(hard, dtype=np.uint8)


class BpOsdDecoder:
    """BP+OSD on a check matrix, as ``ldpc``'s ``BpOsdDecoder``: OSD-0, exhaustive (``"osd_e"``)
    or combination sweep (``"osd_cs"``) of order ``osd_order`` on BP's posteriors when BP does
    not converge. After ``decode``, ``converge`` and ``iter`` describe BP's run."""

    def __init__(
        self,
        pcm: Any,
        error_rate: Union[float, None] = None,
        error_channel: Any = None,
        *,
        max_iter: int = 20,
        bp_method: str = "minimum_sum",
        ms_scaling_factor: float = 0.0,
        osd_method: str = "osd_cs",
        osd_order: int = 7,
    ) -> None:
        self._m, self._n, cols = _columns(pcm)
        self._x = call(
            _core.BpOsd,
            self._m,
            cols,
            _channel(self._n, error_rate, error_channel),
            count(max_iter, "max_iter"),
            bp_method,
            real(ms_scaling_factor, "ms_scaling_factor"),
            osd_method,
            count(osd_order, "osd_order"),
        )
        self.converge = False
        self.iter = 0

    def decode(self, syndrome: Any) -> np.ndarray:
        """The correction (uint8 per column)."""
        correction, converged, iterations = call(self._x.decode, _syndrome(syndrome, self._m))
        self.converge, self.iter = bool(converged), int(iterations)
        return np.asarray(correction, dtype=np.uint8)

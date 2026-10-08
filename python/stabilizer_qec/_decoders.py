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


def _rebuild(cls: type, args: tuple, kwargs: dict) -> Any:
    return cls(*args, **kwargs)


class _DemDecoder:
    """What every decoder on a detector error model shares: shots in, observable flips out.
    Pickled (and copied, and sent to worker processes) as the model and options it was made
    from, and rebuilt from them."""

    __slots__ = ("_x", "_made_from")

    def _made(self, model: DetectorErrorModel, **kwargs: Any) -> DetectorErrorModel:
        self._made_from = ((model,), kwargs)
        return model

    def __reduce__(self) -> tuple:
        return (_rebuild, (type(self), *self._made_from))

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
        model = self._made(_model(model), enable_correlations=enable_correlations)
        self._x = call(_core.Matcher, model._d, bool(enable_correlations))

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


class UnionFind(_DemDecoder):
    """Weighted union-find decoding (Delfosse and Nickerson 2021, with Huang, Newman and
    Brown's weighted growth) on the graph ``Matching`` uses: clusters around the detection
    events grow edge weight by edge weight until their events can pair inside them, and a
    spanning tree of each is peeled for the correction. Linear time in practice, with a logical
    error rate a little above matching's: the standard baseline (``Matching``, near-linear too,
    is about twice as fast here). The model must be decomposed (``decompose_errors=True``).

    >>> import stabilizer_qec as sq
    >>> c = sq.memory_circuit(distance=5, rounds=5, p=0.003)
    >>> dem = c.detector_error_model(decompose_errors=True)
    >>> dets, obs = c.compile_detector_sampler(seed=1).sample(2000, separate_observables=True)
    >>> int((sq.UnionFind(dem).decode_batch(dets) != obs).any(axis=1).sum()) < 100
    True
    """

    __slots__ = ()

    def __init__(self, model: ModelLike) -> None:
        model = self._made(_model(model))
        self._x = call(_core.UnionFinder, model._d)

    def decode_batch(
        self,
        shots: Any,
        *,
        bit_packed_shots: bool = False,
        bit_packed_predictions: bool = False,
        threads: int = 1,
    ) -> np.ndarray:
        """As ``Matching.decode_batch``. A shot no correction explains raises ``ValueError``."""
        packed, n, threads = self._shots(shots, bit_packed_shots, threads)
        preds, failed = call(self._x.decode_batch, packed, n, threads)
        _failed(failed, "no correction explains its detection events (an odd cluster with no boundary)")
        return self._predictions(preds, n, bit_packed_predictions)


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
        model = self._made(_model(model), max_bp_iters=max_bp_iters, bp_method=bp_method, ms_scaling_factor=ms_scaling_factor)
        self._x = call(_core.BeliefMatcher, model._d, count(max_bp_iters, "max_bp_iters"), bp_method, real(ms_scaling_factor, "ms_scaling_factor"))

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
        options = dict(max_iter=max_iter, bp_method=bp_method, ms_scaling_factor=ms_scaling_factor, osd_method=osd_method, osd_order=osd_order)
        model = self._made(_model(model), **options)
        self._x = call(
            _core.DemBpOsd,
            model._d,
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


_PRODUCT_SUM = ("prod_sum", "product_sum", "ps", "0", "prod sum")
_MINIMUM_SUM = ("min_sum", "minimum_sum", "ms", "1", "minimum sum", "min sum")


def _bp_method(method: Any) -> str:
    """``ldpc``'s names for the BP methods."""
    m = str(method).lower()
    if m in _PRODUCT_SUM:
        return "product_sum"
    if m in _MINIMUM_SUM:
        return "minimum_sum"
    raise ValueError(f"bp_method must be product_sum or minimum_sum (or ldpc's aliases), not {method!r}")


_LSD_NAMES = ("osd_0", "osd_e", "osd_cs", "osde", "osdcs", "osd0", "lsd_0", "lsd_e", "lsd_cs", "lsd0", "lsdcs", "lsde")


def _lsd_method(method: Any, order: Any) -> tuple:
    """(engine method name, order) as ``ldpc``'s ``BpLsdDecoder`` reads them: the constructor's
    check, then its ``lsd_method`` setter, which sets the order to 0 for LSD-0."""
    order = count(order, "lsd_order")
    if isinstance(method, str):
        if method.lower() not in _LSD_NAMES:
            raise ValueError(f"lsd_method must be one of 'LSD_0', 'LSD_E', 'LSD_CS', not {method!r}")
    elif isinstance(method, int) and not isinstance(method, bool):
        if method not in (0, 1, 2):
            raise ValueError(f"lsd_method must be one of 0, 1, 2, not {method!r}")
    else:
        raise ValueError(f"lsd_method must be one of 'LSD_0' (0), 'LSD_E' (1), 'LSD_CS' (2), not {method!r}")
    m = str(method).lower()
    if m in ("osd_0", "0", "osd0", "lsd_0", "lsd0"):
        return "osd_0", 0
    if m in ("osd_e", "e", "exhaustive", "lsd_e", "lsde"):
        return "osd_e", order
    if m in ("osd_cs", "1", "cs", "combination_sweep", "lsd_cs"):
        return "osd_cs", order
    raise ValueError(f"lsd_method {method!r} is not one of 'LSD_0', 'LSD_E' or 'LSD_CS'")


class BpLsd(_DemDecoder):
    """BP+LSD on a detector error model (Hillmann et al., 2024): belief propagation, and where
    it does not converge, localized statistics decoding, which grows a cluster around each
    detection event by BP's posteriors and solves each cluster alone. Corrections equal to
    ``ldpc``'s ``BpLsdDecoder`` on the same matrix but for ties among equally likely columns.
    For codes whose faults flip three or more detectors: give it the undecomposed model. The
    defaults are ``ldpc``'s sinter decoder's (``max_iter=0`` is one iteration per fault)."""

    __slots__ = ()

    def __init__(
        self,
        model: ModelLike,
        *,
        max_iter: int = 0,
        bp_method: str = "minimum_sum",
        ms_scaling_factor: float = 0.625,
        lsd_method: Any = "lsd_0",
        lsd_order: int = 0,
        bits_per_step: int = 1,
    ) -> None:
        options = dict(max_iter=max_iter, bp_method=bp_method, ms_scaling_factor=ms_scaling_factor, lsd_method=lsd_method, lsd_order=lsd_order, bits_per_step=bits_per_step)
        model = self._made(_model(model), **options)
        method, order = _lsd_method(lsd_method, lsd_order)
        self._x = call(_core.DemBpLsd, model._d, count(max_iter, "max_iter"), _bp_method(bp_method), real(ms_scaling_factor, "ms_scaling_factor"), method, order, count(bits_per_step, "bits_per_step"))

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
        BP converged before LSD."""
        packed, n, threads = self._shots(shots, bit_packed_shots, threads)
        preds, flags = call(self._x.decode_batch, packed, n, threads)
        flags = np.frombuffer(flags, dtype=np.uint8)
        _failed(list(np.flatnonzero(flags == 2)), "no correction explains the detection events")
        out = self._predictions(preds, n, bit_packed_predictions)
        if return_converged:
            return out, flags == 1
        return out


def _relay_args(legs, solutions, pre_iterations, iterations, gamma0, gamma_range, alpha, seed) -> tuple:
    lo, hi = (real(x, "gamma_range") for x in gamma_range)
    return (
        count(pre_iterations, "pre_iterations"),
        count(legs, "legs"),
        count(iterations, "iterations"),
        None if solutions is None else count(solutions, "solutions"),
        None if gamma0 is None else real(gamma0, "gamma0"),
        (lo, hi),
        None if alpha is None else real(alpha, "alpha"),
        count(seed, "seed"),
    )


class RelayBp(_DemDecoder):
    """Relay-BP on a detector error model (Müller et al., IBM, 2025): min-sum belief
    propagation with memory, run in legs. Each fault's prior mixes in its own last posterior
    with a memory strength; the first leg (``pre_iterations`` iterations) uses ``gamma0``
    for every fault, and each of up to ``legs`` more (``iterations`` each) draws a strength per
    fault from ``gamma_range`` and starts from the posteriors the last leg ended with. The
    lightest correction among the first ``solutions`` legs to converge is returned
    (``solutions=None`` runs every leg). Defaults are ``relay_bp``'s sinter decoder's.
    IBM's ``relay_bp`` gives the same corrections on the same check matrix (see
    ``RelayBpDecoder``). Shot k of a batch draws its strengths from ``seed`` and k, so the
    results do not depend on the thread count. For codes whose faults flip three or more
    detectors: give it the undecomposed model."""

    __slots__ = ()

    def __init__(
        self,
        model: ModelLike,
        *,
        legs: int = 60,
        solutions: Union[int, None] = 5,
        pre_iterations: int = 60,
        iterations: int = 60,
        gamma0: Union[float, None] = 0.1,
        gamma_range: tuple = (-0.24, 0.66),
        alpha: Union[float, None] = None,
        seed: int = 0,
    ) -> None:
        options = dict(legs=legs, solutions=solutions, pre_iterations=pre_iterations, iterations=iterations, gamma0=gamma0, gamma_range=tuple(gamma_range), alpha=alpha, seed=seed)
        model = self._made(_model(model), **options)
        self._x = call(_core.DemRelay, model._d, *_relay_args(legs, solutions, pre_iterations, iterations, gamma0, gamma_range, alpha, seed))

    def decode_batch(
        self,
        shots: Any,
        *,
        return_converged: bool = False,
        bit_packed_shots: bool = False,
        bit_packed_predictions: bool = False,
        threads: int = 1,
    ) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray]]:
        """As ``Matching.decode_batch``; ``return_converged`` also returns, per shot, whether a
        leg converged (where none did, the prediction is from the first leg's last guess)."""
        packed, n, threads = self._shots(shots, bit_packed_shots, threads)
        preds, conv = call(self._x.decode_batch, packed, n, threads)
        out = self._predictions(preds, n, bit_packed_predictions)
        if return_converged:
            return out, np.frombuffer(conv, dtype=np.uint8).astype(bool)
        return out


class ColorMatching(_DemDecoder):
    """Colour-code decoding by matching, Chromobius's construction (Gidney and Jones, 2023):
    each detector doubled into the two sub-graphs leaving out a colour other than its own,
    every fault split into basic faults drawn as edges of that doubled (Möbius) graph, a
    minimum-weight matching of it found by this package's matcher, and the matching lifted
    back to the code by carrying colour charge around its cycles. Predictions equal to the
    ``chromobius`` package's but for ties between equally light matchings.

    The model's detectors need Chromobius's annotation, a 4th coordinate giving the basis and
    colour (0, 1, 2 red, green, blue X; 3, 4, 5 the same in Z; -1 to ignore a detector):
    ``CssCode.color_code(d).memory_circuit(..., annotate_colors=True)`` writes it. Faults it
    cannot split into basic ones are an error, as in Chromobius, unless
    ``ignore_decomposition_failures``. Give it the undecomposed model."""

    __slots__ = ()

    def __init__(self, model: ModelLike, *, ignore_decomposition_failures: bool = False) -> None:
        model = self._made(_model(model), ignore_decomposition_failures=ignore_decomposition_failures)
        self._x = call(_core.DemColorMatcher, model._d, bool(ignore_decomposition_failures))

    @property
    def mobius_model(self) -> DetectorErrorModel:
        """The Möbius model the matching is found on: two detectors per detector, edges only."""
        return DetectorErrorModel(self._x.mobius_model())

    def decode_batch(
        self,
        shots: Any,
        *,
        return_weights: bool = False,
        bit_packed_shots: bool = False,
        bit_packed_predictions: bool = False,
        threads: int = 1,
    ) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray]]:
        """As ``Matching.decode_batch``; ``return_weights`` also returns each shot's Möbius
        matching weight (as ``chromobius``'s ``predict_weighted_obs_flips_from_dets_bit_packed``)."""
        packed, n, threads = self._shots(shots, bit_packed_shots, threads)
        preds, weights, _tied, failed = call(self._x.decode_batch, packed, n, threads)
        _failed(failed, "no lifting of the matching explains the detection events (are the colour annotations right?)")
        out = self._predictions(preds, n, bit_packed_predictions)
        if return_weights:
            return out, np.frombuffer(weights, dtype="<f8").copy()
        return out

    def _decode_with_ties(self, shots: Any) -> tuple:
        packed, n, _ = self._shots(shots, False, 1)
        preds, weights, tied, failed = call(self._x.decode_batch, packed, n, 1)
        return self._predictions(preds, n, False), np.frombuffer(weights, dtype="<f8").copy(), np.frombuffer(tied, dtype=np.uint8).astype(bool), failed


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
    detectors need a time coordinate (their last), as Stim's generated circuits give them.

    A long model is not unrolled: with ``template=True`` (and by default, for a model of more
    than a million faults with a loop), the windows' graphs come from a short template of the
    model's longest loop, whose middle windows serve every window away from the model's ends,
    shifted. Shots are still given whole, so a d = 11 memory of 10,000 rounds decodes from its
    folded model. ``template=False`` cuts every window from the whole model; both give the same
    predictions. With a template, ``return_timings`` times the first shot of each 64 (the rest
    are NaN)."""

    __slots__ = ()

    def __init__(
        self,
        model: ModelLike,
        *,
        commit: int,
        buffer: int,
        mode: str = "parallel",
        enable_correlations: bool = False,
        template: "bool | None" = None,
    ) -> None:
        model = self._made(_model(model), commit=commit, buffer=buffer, mode=mode, enable_correlations=enable_correlations, template=template)
        if template is not None and not isinstance(template, bool):
            raise TypeError(f"template must be True, False or None, not {type(template).__name__}")
        self._x = call(_core.WindowMatcher, model._d, count(commit, "commit"), count(buffer, "buffer"), mode, bool(enable_correlations), template)

    @property
    def streamed(self) -> bool:
        """Whether the windows come from a template of the model's loop (see ``template``)."""
        return self._x.streamed

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
        self._made_from = ((pcm, error_rate, error_channel), dict(max_iter=max_iter, bp_method=bp_method, ms_scaling_factor=ms_scaling_factor))
        self._m, self._n, cols = _columns(pcm)
        self._x = call(_core.Bp, self._m, cols, _channel(self._n, error_rate, error_channel), count(max_iter, "max_iter"), bp_method, real(ms_scaling_factor, "ms_scaling_factor"))
        self.converge = False
        self.iter = 0
        self.log_prob_ratios = np.zeros(self._n)

    def __reduce__(self) -> tuple:
        # Rebuilt from its matrix and options; the last decode's results come along.
        state = {"converge": self.converge, "iter": self.iter, "log_prob_ratios": self.log_prob_ratios}
        return (_rebuild, (type(self), *self._made_from), state)

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
        options = dict(max_iter=max_iter, bp_method=bp_method, ms_scaling_factor=ms_scaling_factor, osd_method=osd_method, osd_order=osd_order)
        self._made_from = ((pcm, error_rate, error_channel), options)
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

    def __reduce__(self) -> tuple:
        return (_rebuild, (type(self), *self._made_from), {"converge": self.converge, "iter": self.iter})

    def decode(self, syndrome: Any) -> np.ndarray:
        """The correction (uint8 per column)."""
        correction, converged, iterations = call(self._x.decode, _syndrome(syndrome, self._m))
        self.converge, self.iter = bool(converged), int(iterations)
        return np.asarray(correction, dtype=np.uint8)


class BpLsdDecoder:
    """BP+LSD on a check matrix, as ``ldpc``'s ``BpLsdDecoder``, with its arguments, defaults
    and aliases (``osd_method``/``osd_order`` for ``lsd_method``/``lsd_order``; ``max_iter=0``
    and ``bits_per_step=0`` mean the block length; LSD-0 sets the order to 0). The flooding
    schedule only. After ``decode``, ``converge`` and ``iter`` describe BP's run."""

    def __init__(
        self,
        pcm: Any,
        error_rate: Union[float, None] = None,
        error_channel: Any = None,
        *,
        max_iter: int = 0,
        bp_method: str = "minimum_sum",
        ms_scaling_factor: float = 1.0,
        schedule: str = "parallel",
        bits_per_step: int = 1,
        lsd_order: int = 0,
        lsd_method: Any = 0,
        always_run_lsd: bool = False,
        **kwargs: Any,
    ) -> None:
        if "osd_method" in kwargs:
            lsd_method = kwargs.pop("osd_method")
        if "osd_order" in kwargs:
            lsd_order = kwargs.pop("osd_order")
        if kwargs:
            raise TypeError(f"unexpected keyword arguments: {', '.join(sorted(kwargs))}")
        if schedule != "parallel":
            raise ValueError("only the parallel (flooding) schedule is supported")
        options = dict(max_iter=max_iter, bp_method=bp_method, ms_scaling_factor=ms_scaling_factor, bits_per_step=bits_per_step, lsd_order=lsd_order, lsd_method=lsd_method, always_run_lsd=always_run_lsd)
        self._made_from = ((pcm, error_rate, error_channel), options)
        self._m, self._n, cols = _columns(pcm)
        method, order = _lsd_method(lsd_method, lsd_order)
        iters = count(max_iter, "max_iter") or self._n
        self._x = call(
            _core.BpLsd,
            self._m,
            cols,
            _channel(self._n, error_rate, error_channel),
            iters,
            _bp_method(bp_method),
            real(ms_scaling_factor, "ms_scaling_factor"),
            method,
            order,
            count(bits_per_step, "bits_per_step"),
            bool(always_run_lsd),
        )
        self.converge = False
        self.iter = 0
        self.last_tied = False

    def __reduce__(self) -> tuple:
        return (_rebuild, (type(self), *self._made_from), {"converge": self.converge, "iter": self.iter, "last_tied": self.last_tied})

    def decode(self, syndrome: Any) -> np.ndarray:
        """The correction (uint8 per column). ``last_tied`` says whether this decode made a
        choice ``ldpc`` might make otherwise (equal keys in a long sort, or a merge of many
        clusters at once), where the two corrections can differ."""
        correction, converged, iterations, solved, ties, _ = call(self._x.decode, _syndrome(syndrome, self._m))
        if not solved:
            raise ValueError("no correction explains the syndrome")
        self.converge, self.iter, self.last_tied = bool(converged), int(iterations), ties > 0
        return np.asarray(correction, dtype=np.uint8)

    def _cluster_bits(self, syndrome: Any) -> list:
        """Each final cluster's (id, bits in robin-set order): for checking against ldpc's
        statistics."""
        return call(self._x.decode, _syndrome(syndrome, self._m), True)[5]


class RelayBpDecoder:
    """Relay-BP on a check matrix, as IBM's ``relay_bp.RelayDecoderF64``: the same corrections
    for the same options, explicit memory strengths (``gammas``, one row per leg) or a seed.
    Like IBM's decoder, its random memory strengths continue from decode to decode. Defaults
    are ``relay_bp``'s. After ``decode``, ``converge``, ``iter`` (over every leg), ``legs`` and
    ``weight`` describe the run."""

    def __init__(
        self,
        pcm: Any,
        error_rate: Union[float, None] = None,
        error_channel: Any = None,
        *,
        legs: int = 300,
        solutions: Union[int, None] = 1,
        pre_iterations: int = 80,
        iterations: int = 60,
        gamma0: Union[float, None] = 0.1,
        gamma_range: tuple = (-0.24, 0.66),
        gammas: Any = None,
        alpha: Union[float, None] = None,
        alpha_iteration_scaling_factor: float = 1.0,
        seed: int = 0,
    ) -> None:
        options = dict(legs=legs, solutions=solutions, pre_iterations=pre_iterations, iterations=iterations, gamma0=gamma0, gamma_range=tuple(gamma_range), gammas=gammas, alpha=alpha, alpha_iteration_scaling_factor=alpha_iteration_scaling_factor, seed=seed)
        self._made_from = ((pcm, error_rate, error_channel), options)
        self._m, self._n, cols = _columns(pcm)
        rows = None
        if gammas is not None:
            g = np.asarray(gammas, dtype=float)
            if g.ndim != 2 or g.shape[1] != self._n:
                raise ValueError(f"gammas has one row per leg of {self._n} memory strengths (one per column), not shape {g.shape}")
            rows = g.tolist()
        pre, nlegs, its, sols, g0, rng, a, sd = _relay_args(legs, solutions, pre_iterations, iterations, gamma0, gamma_range, alpha, seed)
        self._x = call(_core.Relay, self._m, cols, _channel(self._n, error_rate, error_channel), pre, nlegs, its, sols, g0, rng, rows, a, real(alpha_iteration_scaling_factor, "alpha_iteration_scaling_factor"), sd)
        self.converge, self.iter, self.legs, self.weight = False, 0, 0, float("inf")

    def __reduce__(self) -> tuple:
        # Rebuilt from its matrix and options: its generator starts again from the seed.
        return (_rebuild, (type(self), *self._made_from), {"converge": self.converge, "iter": self.iter, "legs": self.legs, "weight": self.weight})

    def decode(self, syndrome: Any) -> np.ndarray:
        """The correction (uint8 per column); where no leg converged, the first leg's last
        hard decision."""
        correction, converged, iterations, legs, weight = call(self._x.decode, _syndrome(syndrome, self._m))
        self.converge, self.iter, self.legs, self.weight = bool(converged), int(iterations), int(legs), float(weight)
        return np.asarray(correction, dtype=np.uint8)

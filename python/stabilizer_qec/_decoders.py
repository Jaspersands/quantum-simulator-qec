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

    __slots__ = ("_x", "_made_from", "__weakref__")

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

    def decode_async(self, shots: Any, **kwargs: Any) -> Any:
        """``decode_batch(shots, **kwargs)`` on a background thread (the GIL released in the
        engine): a ``realtime.AsyncResult`` whose ``get()`` is the predictions, ``ready()``
        whether they are in. A single syndrome gives one row."""
        from .realtime import decode_async

        s = np.asarray(shots)
        return decode_async(self, s.reshape(1, -1) if s.ndim == 1 else shots, **kwargs)


def _failed(failed: list, what: str) -> None:
    if failed:
        more = f" (and {len(failed) - 1} more)" if len(failed) > 1 else ""
        raise ValueError(f"shot {failed[0]}{more}: {what}")


from ._matching import Matching, _CorrelatedMatching  # noqa: E402  (PyMatching's class, whole)


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
        if type(model).__name__ == "Circuit":
            # As beliefmatching: a circuit's model with its faults decomposed.
            from ._circuit import Circuit

            model = (model if isinstance(model, Circuit) else Circuit(str(model))).detector_error_model(decompose_errors=True)
        model = self._made(_model(model), max_bp_iters=max_bp_iters, bp_method=bp_method, ms_scaling_factor=ms_scaling_factor)
        self._x = call(_core.BeliefMatcher, model._d, count(max_bp_iters, "max_bp_iters"), _bp_method(bp_method), real(ms_scaling_factor, "ms_scaling_factor"))

    @classmethod
    def from_detector_error_model(cls, model: ModelLike, *, max_bp_iters: int = 20, bp_method: str = "product_sum", ms_scaling_factor: float = 1.0) -> "BeliefMatching":
        """As ``beliefmatching.BeliefMatching.from_detector_error_model``: from a decomposed
        model."""
        return cls(model, max_bp_iters=max_bp_iters, bp_method=bp_method, ms_scaling_factor=ms_scaling_factor)

    @classmethod
    def from_stim_circuit(cls, circuit: Any, *, max_bp_iters: int = 20, bp_method: str = "product_sum", ms_scaling_factor: float = 1.0) -> "BeliefMatching":
        """As ``beliefmatching.BeliefMatching.from_stim_circuit``: from a circuit's model with
        its faults decomposed (``detector_error_model(decompose_errors=True)``)."""
        from ._circuit import Circuit

        c = circuit if isinstance(circuit, Circuit) else Circuit(str(circuit))
        return cls(c, max_bp_iters=max_bp_iters, bp_method=bp_method, ms_scaling_factor=ms_scaling_factor)

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
    IBM's bivariate bicycle codes: give it the undecomposed model. ``schedule`` and
    ``random_schedule_seed`` are ``ldpc``'s (see ``_schedule``)."""

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
        schedule: str = "parallel",
        random_schedule_seed: int = 0,
    ) -> None:
        options = dict(max_iter=max_iter, bp_method=bp_method, ms_scaling_factor=ms_scaling_factor, osd_method=osd_method, osd_order=osd_order, schedule=schedule, random_schedule_seed=random_schedule_seed)
        model = self._made(_model(model), **options)
        self._x = call(
            _core.DemBpOsd,
            model._d,
            count(max_iter, "max_iter"),
            _bp_method(bp_method),
            real(ms_scaling_factor, "ms_scaling_factor"),
            osd_method,
            count(osd_order, "osd_order"),
            _schedule(schedule, random_schedule_seed),
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


def _schedule(schedule: Any, random_schedule_seed: Any) -> str:
    """BP's schedule by ``ldpc``'s names: "parallel" (flooding), "serial" or "serial_relative"
    (or 0, 1, 2). ``random_schedule_seed`` is checked as ``ldpc``'s; as in ``ldpc``'s
    constructors, which turn the random serial order it starts back off, the serial order stays
    the fixed one (stimbposd's serial decoders pass ``random_schedule_seed=0``)."""
    from ._ldpc import _SCHEDULES

    s = _SCHEDULES.get(str(schedule).lower())
    if s is None:
        raise ValueError(f"schedule must be parallel, serial or serial_relative, not {schedule!r}")
    seed = random_schedule_seed
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < -2:
        raise ValueError(f"random_schedule_seed must be an integer of at least -2 (as ldpc's), not {seed!r}")
    return s


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
    defaults are ``ldpc``'s sinter decoder's but for ``max_iter``: 30 iterations, where its
    0 means one per fault (thousands on the gross code, and most of the time). ``schedule``
    and ``random_schedule_seed`` are ``ldpc``'s (see ``_schedule``)."""

    __slots__ = ()

    def __init__(
        self,
        model: ModelLike,
        *,
        max_iter: int = 30,
        bp_method: str = "minimum_sum",
        ms_scaling_factor: float = 0.625,
        lsd_method: Any = "lsd_0",
        lsd_order: int = 0,
        bits_per_step: int = 1,
        schedule: str = "parallel",
        random_schedule_seed: int = 0,
    ) -> None:
        options = dict(max_iter=max_iter, bp_method=bp_method, ms_scaling_factor=ms_scaling_factor, lsd_method=lsd_method, lsd_order=lsd_order, bits_per_step=bits_per_step, schedule=schedule, random_schedule_seed=random_schedule_seed)
        model = self._made(_model(model), **options)
        method, order = _lsd_method(lsd_method, lsd_order)
        self._x = call(_core.DemBpLsd, model._d, count(max_iter, "max_iter"), _bp_method(bp_method), real(ms_scaling_factor, "ms_scaling_factor"), method, order, count(bits_per_step, "bits_per_step"), _schedule(schedule, random_schedule_seed))

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


class AmbiguityClustering(_DemDecoder):
    """Ambiguity Clustering on a detector error model (Wolanski and Barber, arXiv 2406.14527):
    belief propagation, and where it does not explain the syndrome, an elimination that goes
    only as far as the syndrome needs, ordered by BP's posteriors, grown by ``kappa`` times the
    model's fault count further columns into blocks (clusters) of the matrix that are decoded
    apart. A block whose logical effect its syndrome fixes is read off; an ambiguous one is
    searched over its non-pivot columns, every choice of at most two, each observable set to the
    value carrying more probability. Ambiguities in separate blocks add rather than multiply.

    At ``kappa=0`` it is BP+OSD-0 (on the same BP). On the bivariate bicycle codes' memories
    at the default ``kappa=0.05`` (the paper's largest) it is more accurate than ``BpOsd``'s
    combination sweep of order 7 and 8 to 100 times faster. For codes whose faults flip three
    or more detectors: give it the undecomposed model. BP is the paper's: 9 iterations of
    product-sum, flooding (``schedule="serial"`` is ``ldpc``'s serial schedule).

    >>> import stabilizer_qec as sq
    >>> c = sq.memory_circuit(distance=5, rounds=5, p=0.003)
    >>> dets, obs = c.compile_detector_sampler(seed=2).sample(1000, separate_observables=True)
    >>> ac = sq.AmbiguityClustering(c.detector_error_model())
    >>> int((ac.decode_batch(dets) != obs).any(axis=1).sum()) < 30
    True
    """

    __slots__ = ()

    def __init__(
        self,
        model: ModelLike,
        *,
        kappa: float = 0.05,
        max_iter: int = 9,
        bp_method: str = "product_sum",
        ms_scaling_factor: float = 1.0,
        schedule: str = "parallel",
    ) -> None:
        options = dict(kappa=kappa, max_iter=max_iter, bp_method=bp_method, ms_scaling_factor=ms_scaling_factor, schedule=schedule)
        model = self._made(_model(model), **options)
        kappa = real(kappa, "kappa")
        if not 0 <= kappa:
            raise ValueError(f"kappa is a fraction of the faults, at least 0, not {kappa}")
        self._x = call(_core.DemAmbiguityClustering, model._d, count(max_iter, "max_iter"), _bp_method(bp_method), real(ms_scaling_factor, "ms_scaling_factor"), kappa, schedule)

    @property
    def extra_columns(self) -> int:
        """K: the columns stage 2 adds to the blocks, ``kappa`` times the fault count, rounded."""
        return self._x.extra_columns

    def decode_batch(
        self,
        shots: Any,
        *,
        return_converged: bool = False,
        return_stats: bool = False,
        bit_packed_shots: bool = False,
        bit_packed_predictions: bool = False,
        threads: int = 1,
    ) -> Any:
        """As ``Matching.decode_batch``. ``return_converged`` also returns, per shot, whether BP
        explained it alone; ``return_stats`` returns, per shot, the blocks, the ambiguous blocks
        and the solutions searched in them (an int array of shape (shots, 3))."""
        packed, n, threads = self._shots(shots, bit_packed_shots, threads)
        preds, flags, stats = call(self._x.decode_batch, packed, n, threads)
        flags = np.frombuffer(flags, dtype=np.uint8)
        _failed(list(np.flatnonzero(flags == 2)), "no correction explains the detection events")
        out = [self._predictions(preds, n, bit_packed_predictions)]
        if return_converged:
            out.append(flags == 1)
        if return_stats:
            out.append(np.frombuffer(stats, dtype="<u4").reshape(n, 3).astype(np.int64))
        return out[0] if len(out) == 1 else tuple(out)


def _resolution(weighted: bool, weight_resolution: int) -> Union[int, None]:
    if not weighted:
        return None
    r = count(weight_resolution, "weight_resolution")
    if r < 1:
        raise ValueError("weight_resolution must be at least 1")
    return r


class CollisionClustering(_DemDecoder):
    """Collision Clustering (Barber et al., arXiv 2309.05558: Riverlane's union-find for an
    ASIC) on the graph ``Matching`` uses. Each detection event has a ball, in the Cluster
    Growth Stack, whose radius grows half an edge a round while its cluster is odd and off the
    boundary; every pair of events from two clusters is compared each round, and their
    clusters merge (in the Parent table, with Parity and Boundary registers) once their balls
    overlap, r_i + r_j ≥ d(i, j). The edges the balls cover are peeled for the correction.

    Its clusters are union-find's (Delfosse and Nickerson's, round by round), as the paper
    claims: the tests check it shot by shot. Unweighted by default, as the hardware is (every
    edge two steps); ``weighted=True`` scales each edge's weight to at most
    ``weight_resolution`` steps, with distances by Dijkstra. ``return_stats`` gives the rounds
    of growth and the pairs compared, the hardware's work. The model must be decomposed.

    >>> import stabilizer_qec as sq
    >>> c = sq.memory_circuit(distance=5, rounds=5, p=0.003)
    >>> dem = c.detector_error_model(decompose_errors=True)
    >>> dets, obs = c.compile_detector_sampler(seed=1).sample(2000, separate_observables=True)
    >>> int((sq.CollisionClustering(dem).decode_batch(dets) != obs).any(axis=1).sum()) < 100
    True
    """

    __slots__ = ()

    def __init__(self, model: ModelLike, *, weighted: bool = False, weight_resolution: int = 16) -> None:
        model = self._made(_model(model), weighted=weighted, weight_resolution=weight_resolution)
        self._x = call(_core.DemCollisionClustering, model._d, _resolution(weighted, weight_resolution))

    def decode_batch(
        self,
        shots: Any,
        *,
        return_stats: bool = False,
        bit_packed_shots: bool = False,
        bit_packed_predictions: bool = False,
        threads: int = 1,
    ) -> Any:
        """As ``Matching.decode_batch``; ``return_stats`` also returns, per shot, the rounds of
        growth and the pairs of events compared (an int array of shape (shots, 2)). A shot no
        correction explains raises ``ValueError``."""
        packed, n, threads = self._shots(shots, bit_packed_shots, threads)
        preds, failed, stats = call(self._x.decode_batch, packed, n, threads)
        _failed(failed, "no correction explains its detection events (an odd cluster with no boundary)")
        pred = self._predictions(preds, n, bit_packed_predictions)
        return (pred, np.frombuffer(stats, dtype="<u4").reshape(n, 2).astype(np.int64)) if return_stats else pred


class LocalClustering(_DemDecoder):
    """The Local Clustering Decoder (Ziad et al., arXiv 2411.10343: Riverlane's adaptive
    union-find for an FPGA) on the graph ``Matching`` uses. Every vertex keeps a radius, a
    cluster index (the lowest in its cluster), a parent, a parity and an activity bit, and a
    controller runs the paper's stages: growing (active vertices' radii step up, to their
    longest edge), merging (each vertex adopts a lower cluster index across an edge their radii
    cover, and passes odd parity to its parent), picking (odd roots become active) and syncing
    (activity spreads through the cluster), merging and syncing repeating while a vertex is
    busy, until no vertex is active. The covered edges are peeled for the correction.

    Its adaptivity engine pre-grows edges: ``adaptivity`` maps each trigger to the edges it
    makes likely, as pairs of detectors (``num_detectors`` for the boundary), and the
    ``heralds`` given to ``decode_batch`` say which triggers fired; the clusters form along the
    pre-grown edges first. ``LocalClustering.from_circuit`` builds the paper's map for leakage:
    a measurement heralded leaked means its qubit may have leaked any time since its last reset.

    Without pre-grown edges its clusters are union-find's and its predictions are
    ``CollisionClustering``'s, shot for shot. Unweighted by default, as the hardware is;
    ``weighted=True`` scales each edge's weight to at most ``weight_resolution`` steps (the
    paper's weighted software decoder). The model must be decomposed.

    >>> import stabilizer_qec as sq
    >>> c = sq.memory_circuit(distance=5, rounds=5, p=0.003)
    >>> dem = c.detector_error_model(decompose_errors=True)
    >>> dets, obs = c.compile_detector_sampler(seed=1).sample(2000, separate_observables=True)
    >>> int((sq.LocalClustering(dem).decode_batch(dets) != obs).any(axis=1).sum()) < 100
    True
    """

    __slots__ = ()

    def __init__(self, model: ModelLike, *, weighted: bool = False, weight_resolution: int = 16, adaptivity: Any = None) -> None:
        pairs = [] if adaptivity is None else [[(int(u), int(v)) for u, v in edges] for edges in adaptivity]
        model = self._made(_model(model), weighted=weighted, weight_resolution=weight_resolution, adaptivity=pairs or None)
        nd = model.num_detectors
        for edges in pairs:
            for u, v in edges:
                if not (0 <= u <= nd and 0 <= v <= nd):
                    raise ValueError(f"adaptivity edge ({u}, {v}) names a detector outside 0..{nd} ({nd} is the boundary)")
        self._x = call(_core.DemLocalClustering, model._d, _resolution(weighted, weight_resolution), pairs)

    @classmethod
    def from_circuit(cls, circuit: Any, *, leakage_aware: bool = True, weighted: bool = False, weight_resolution: int = 16) -> "LocalClustering":
        """From a circuit's decomposed model, and, with ``leakage_aware``, the paper's leakage
        adaptivity map: one trigger per measurement (as ``LeakageSampler.sample``'s heralds),
        pre-growing the edges more likely if the measured qubit was leaked since its last
        reset, so that every two-qubit gate since left its partner fully depolarized (and its
        own readout random). Each is found by comparing the error model of the noiseless
        circuit with those channels added to the decoding graph."""
        dem = circuit.without_leakage().detector_error_model(decompose_errors=True)
        if not leakage_aware:
            return cls(dem, weighted=weighted, weight_resolution=weight_resolution)
        edges = call(_core.DemLocalClustering, dem._d, None, []).edges
        return cls(dem, weighted=weighted, weight_resolution=weight_resolution, adaptivity=_leakage_adaptivity(circuit, edges, dem.num_detectors))

    @property
    def adaptivity_sizes(self) -> list:
        """Per trigger, how many edges of the graph it pre-grows."""
        return self._x.adaptivity_sizes

    def decode_batch(
        self,
        shots: Any,
        *,
        heralds: Any = None,
        return_stats: bool = False,
        bit_packed_shots: bool = False,
        bit_packed_heralds: bool = False,
        bit_packed_predictions: bool = False,
        threads: int = 1,
    ) -> Any:
        """As ``Matching.decode_batch``. ``heralds``: per shot, which triggers of the
        adaptivity map fired (shape (shots, triggers)); their edges are pre-grown.
        ``return_stats`` also returns, per shot, the rounds of growth, the stage passes the
        controller ran (the hardware's time steps) and the edges pre-grown (an int array of
        shape (shots, 3)). A shot no correction explains raises ``ValueError``."""
        packed, n, threads = self._shots(shots, bit_packed_shots, threads)
        h, nh = None, 0
        if heralds is not None:
            nh = len(self._x.adaptivity_sizes)
            h, hn = rows_to_b8(heralds, nh, bit_packed_heralds, "heralds")
            if hn != n:
                raise ValueError(f"heralds has {hn} shots, not {n}")
        preds, failed, stats = call(self._x.decode_batch, packed, n, threads, h, nh)
        _failed(failed, "no correction explains its detection events (an odd cluster with no boundary)")
        pred = self._predictions(preds, n, bit_packed_predictions)
        return (pred, np.frombuffer(stats, dtype="<u4").reshape(n, 3).astype(np.int64)) if return_stats else pred


def _split_into_edges(dets: list, edges: set, boundary: int) -> Union[list, None]:
    """Detectors a fault flips, paired into edges of the graph (or to the boundary)."""
    if not dets:
        return []
    a, rest = dets[0], dets[1:]
    for k, b in enumerate(rest):
        if (min(a, b), max(a, b)) in edges:
            more = _split_into_edges(rest[:k] + rest[k + 1 :], edges, boundary)
            if more is not None:
                return [(min(a, b), max(a, b))] + more
    if (a, boundary) in edges:
        more = _split_into_edges(rest, edges, boundary)
        if more is not None:
            return [(a, boundary)] + more
    return None


def _herald_qubits(tag: str) -> Union[list, None]:
    """The qubits a herald's tag (``HERALD_LEAKAGE_EVENT(qubits=…)``) names; None for another."""
    t = tag.strip()
    if not t.startswith("HERALD_LEAKAGE_EVENT"):
        return None
    inner = t[len("HERALD_LEAKAGE_EVENT") :].strip()
    if not (inner.startswith("(") and inner.endswith(")")):
        return None
    key, _, value = inner[1:-1].partition("=")
    return [int(q.lstrip("!")) for q in value.split()] if key.strip() == "qubits" else None


def _leakage_faults(circuit: Any, graded: bool = True) -> list:
    """Per measurement record of ``circuit`` (a herald's too), the faults its qubit's leakage
    makes likely, as (detectors, observables mask, probability): the faults of the noiseless
    circuit's model with the partners of the two-qubit gates since the qubit's last reset
    depolarized, and the qubit itself where it is read fully. ``graded``: the leak came at a
    uniformly random point of those g gates, so the j-th's partner is depolarized with
    probability j/(g + 1); else every partner fully. Detectors are the circuit's."""
    from ._circuit import Circuit
    from ._stim import gate_data

    flat = circuit.flattened().without_noise()
    items = flat._items()
    info = {}

    def data(name: str):
        if name not in info:
            g = gate_data(name)
            info[name] = (g.is_two_qubit_gate and g.is_unitary, g.is_reset, g.produces_measurements, g.is_two_qubit_gate, name == "MPP")
        return info[name]

    flag = (1 << 28) | (1 << 26)  # a record or a sweep bit
    mask = (1 << 24) - 1
    # Per qubit, the (instruction, partner) of each two-qubit gate since its last reset.
    gates: dict = {}
    windows = []  # per measurement record: (instruction, qubit, [(instruction, partner)]) or None
    for k, it in enumerate(items):
        if it[0] != "op":
            continue
        name, tag, targets = it[1], it[2], it[4]
        unitary2, reset, measures, two, mpp = data(name)
        if measures:
            if name == "MPAD":
                heralded = _herald_qubits(tag)
                if heralded is not None and len(heralded) == len(targets):
                    windows.extend((k, q, list(gates.get(q, []))) for q in heralded)
                else:
                    windows.extend([None] * len(targets))
            elif mpp:
                products = sum(1 for t in targets if t != 1 << 27) - sum(1 for t in targets if t == 1 << 27)
                windows.extend([None] * products)
            elif two:
                windows.extend([None] * (len(targets) // 2))
            else:
                for t in targets:
                    q = t & mask
                    windows.append((k, q, list(gates.get(q, []))) if not t & flag else None)
        if reset:
            for t in targets:
                gates[t & mask] = []
        if unitary2:
            for a, b in zip(targets[::2], targets[1::2]):
                if (a | b) & flag:
                    continue
                a, b = a & mask, b & mask
                gates.setdefault(a, []).append((k, b))
                gates.setdefault(b, []).append((k, a))
    out = []
    cache: dict = {}
    for w in windows:
        if w is None:
            out.append([])
            continue
        k, q, partners = w
        key = (k, q, tuple(partners))
        if key not in cache:
            after: dict = {}
            for rank, (g, p) in enumerate(partners, 1):
                after.setdefault(g, []).append((p, rank / (len(partners) + 1) if graded else 1.0))
            noisy = []
            for j, it in enumerate(items):
                if j == k:
                    noisy.append(("op", "DEPOLARIZE1", "", [0.75], [q]))
                noisy.append(it)
                for p, share in sorted(set(after.get(j, []))):
                    noisy.append(("op", "DEPOLARIZE1", "", [0.75 * share], [p]))
            found = []
            for p, targets in Circuit._of_items(noisy).detector_error_model()._d.flat_errors():
                dets = tuple(sorted({d for obs, d in targets if not obs}))
                obs = 0
                for is_obs, i in targets:
                    if is_obs:
                        obs ^= 1 << i
                if dets:
                    found.append((dets, obs, p))
            cache[key] = found
        out.append(cache[key])
    return out


def _kept_detectors(circuit: Any) -> tuple:
    """(herald records, kept detectors, herald detector pairs) of a circuit: see
    ``leakage::herald_map``."""
    return call(_core.circuit_herald_map, circuit._stim_exact_text())


def _leakage_adaptivity(circuit: Any, graph_edges: list, num_detectors: int) -> list:
    """Per measurement record, the graph's edges its qubit's leakage makes likely (as
    ``LocalClustering.from_circuit``), on the detectors of the circuit without leakage."""
    edges = {(min(u, v), max(u, v)) for u, v in graph_edges}
    renumber = {d: i for i, d in enumerate(_kept_detectors(circuit)[1])}
    out = []
    for faults in _leakage_faults(circuit, graded=False):
        found = set()
        for dets, _, _ in faults:
            if any(d not in renumber for d in dets):
                continue
            dets = [renumber[d] for d in dets]
            split = _split_into_edges(dets, edges, num_detectors) if len(dets) <= 6 else None
            if split:
                found.update(split)
        out.append(sorted(found))
    return out


class LeakageAwareDecoder(_DemDecoder):
    """Leakage-aware decoding from heralds (Deltakit's ``LeakageFlags``, Google's and
    Riverlane's leakage-aware decoders), around matching, union-find, BP+OSD, BP+LSD or Relay-BP.
    Each measurement record of the circuit is a trigger: one that found its qubit leaked (the
    leakage sampler's heralds), or a ``HERALD_LEAKAGE_EVENT`` that fired, says the qubit leaked
    some time since its last reset, so every two-qubit gate since may have left its partner
    fully depolarized, and the faults that makes likely have their priors raised for that shot
    (to their probability given the leak, combined with their own) before it is decoded once.
    A shot with no trigger is decoded as the inner decoder decodes it. With ``graded`` (the
    default) the leak came at a uniformly random point of the qubit's g gates since its reset,
    so the j-th gate's partner is depolarized with probability j/(g + 1); ``graded=False``
    depolarizes every partner fully (the map of the local clustering decoder's paper, which
    sets those faults' priors to 1/2 and leaves Relay-BP, with no OSD to fall back on, worse
    off than without heralds). On a d = 5 memory with Deltakit's leakage at every CX (0.3%),
    heralds cut matching's logical errors 4 times graded, 2 times not.

    Made from the circuit; decodes the detectors of ``circuit.without_leakage()`` (shots with
    the circuit's own detectors are taken too: a detector reading only a herald fires its
    trigger, and the rest are decoded). ``decode_batch`` takes the triggers as ``heralds``
    (per measurement record, as ``LeakageSampler.sample`` gives them) or ``leakage_flags``
    (per herald, as ``LeakageSampler.sample_measurements`` and Deltakit give them).
    ``decoder="local_clustering"`` is ``LocalClustering.from_circuit``, its heralds pre-growing
    the edges instead. Options are the inner decoder's.

    >>> import stabilizer_qec as sq
    >>> lines = []
    >>> for line in str(sq.memory_circuit(distance=5, rounds=5, p=0.001)).splitlines():
    ...     lines.append(line)
    ...     if line.startswith("CX"):
    ...         lines.append("LEAKAGE(0.003)" + line[2:])
    >>> c = sq.Circuit("\\n".join(lines))
    >>> dets, obs, heralds = c.compile_leakage_sampler(seed=1).sample(4000)
    >>> dec = sq.LeakageAwareDecoder(c)
    >>> plain = (dec.decode_batch(dets) != obs).any(axis=1).sum()
    >>> aware = (dec.decode_batch(dets, heralds=heralds) != obs).any(axis=1).sum()
    >>> bool(aware < plain / 3)
    True
    """

    __slots__ = ("_lcd", "_records", "_heralds", "_herald_dets", "_kept", "_full")

    def __init__(self, circuit: Any, decoder: str = "matching", *, graded: bool = True, **decoder_options: Any) -> None:
        from ._circuit import Circuit

        if not isinstance(circuit, Circuit):
            circuit = Circuit(circuit)
        self._made_from = ((circuit,), dict(decoder=decoder, graded=graded, **decoder_options))
        bp = {"bposd": BpOsd, "bplsd": BpLsd, "relay_bp": RelayBp}
        graph = decoder in ("matching", "union_find", "local_clustering")
        if not graph and decoder not in bp:
            raise ValueError(f"decoder '{decoder}' is not matching, union_find, local_clustering, bposd, bplsd or relay_bp")
        model = circuit.without_leakage().detector_error_model(decompose_errors=graph)
        self._heralds, self._kept, self._herald_dets = (list(x) for x in _kept_detectors(circuit))
        self._records = circuit.num_measurements
        self._full = circuit.num_detectors
        self._lcd = None
        if decoder == "local_clustering":
            edges = call(_core.DemLocalClustering, model._d, None, []).edges
            self._lcd = LocalClustering(model, adaptivity=_leakage_adaptivity(circuit, edges, model.num_detectors), **decoder_options)
            self._x = self._lcd._x
            return
        inner = None
        if decoder in bp:
            inner = bp[decoder](model, **decoder_options)._x
        elif decoder_options:
            raise TypeError(f"the {decoder} decoder takes no options here, not {', '.join(decoder_options)}")
        nd = model.num_detectors
        renumber = {d: i for i, d in enumerate(self._kept)}
        edges = _graph_edges(model) if graph else set()
        triggers = []
        for faults in _leakage_faults(circuit, graded=bool(graded)):
            pieces = []
            for dets, obs, q in faults:
                if any(d not in renumber for d in dets):
                    continue
                dets = [renumber[d] for d in dets]
                if graph:
                    # A fault of the leaked model, as the graph's edges it decomposes into.
                    split = _split_into_edges(dets, edges, nd) if len(dets) <= 6 else None
                    pieces.extend(([u] if v == nd else [u, v], 0, q) for u, v in split or [])
                else:
                    pieces.append((dets, obs, q))
            triggers.append(pieces)
        self._x = call(_core.DemLeakageAware, model._d, decoder, inner, triggers)

    @property
    def num_triggers(self) -> int:
        """The triggers: one per measurement record of the circuit."""
        return self._records

    @property
    def trigger_sizes(self) -> list:
        """Per trigger, how many of the decoder's faults it raises (edges it pre-grows, for
        local clustering)."""
        return list(self._lcd.adaptivity_sizes if self._lcd is not None else self._x.trigger_sizes)

    @property
    def unmatched(self) -> int:
        """Faults of the leaked circuits' models that are none of the decoder's, and so not
        raised (0 for local clustering, whose triggers are the graph's edges)."""
        return 0 if self._lcd is not None else self._x.unmatched

    def _triggers(self, n: int, heralds: Any, leakage_flags: Any, herald_events: Any) -> Union[np.ndarray, None]:
        fired = None

        def add(rows: np.ndarray) -> None:
            nonlocal fired
            fired = rows if fired is None else fired | rows

        if heralds is not None:
            h = np.asarray(heralds, dtype=bool)
            if h.shape != (n, self._records):
                raise ValueError(f"heralds has shape {h.shape}, not ({n}, {self._records}): one per measurement record")
            add(h)
        if leakage_flags is not None:
            f = np.asarray(leakage_flags, dtype=bool)
            if f.shape != (n, len(self._heralds)):
                raise ValueError(f"leakage_flags has shape {f.shape}, not ({n}, {len(self._heralds)}): one per herald")
            rows = np.zeros((n, self._records), dtype=bool)
            rows[:, self._heralds] = f
            add(rows)
        if herald_events is not None and self._herald_dets:
            rows = np.zeros((n, self._records), dtype=bool)
            dets = [d for d, _ in self._herald_dets]
            rows[:, [r for _, r in self._herald_dets]] = herald_events[:, dets]
            add(rows)
        return fired

    def decode_batch(
        self,
        shots: Any,
        *,
        heralds: Any = None,
        leakage_flags: Any = None,
        return_raised: bool = False,
        bit_packed_shots: bool = False,
        bit_packed_predictions: bool = False,
        threads: int = 1,
    ) -> Any:
        """As ``Matching.decode_batch``, with the shots' triggers: ``heralds`` (shape (shots,
        measurement records)), ``leakage_flags`` (shape (shots, heralds)), and, where the shots
        are the circuit's own detectors, the detectors reading a herald. ``return_raised`` also
        returns how many priors each shot raised. A shot no correction explains raises
        ``ValueError``."""
        events = None
        if not bit_packed_shots:
            s = np.asarray(shots)
            if s.ndim == 2 and s.shape[1] == self._full and self._full != self.num_detectors:
                events = s.astype(bool)
                shots = events[:, self._kept]
        packed, n, threads = self._shots(shots, bit_packed_shots, threads)
        fired = self._triggers(n, heralds, leakage_flags, events)
        if self._lcd is not None:
            if return_raised:
                pred, stats = self._lcd.decode_batch(shots, heralds=fired, return_stats=True, bit_packed_shots=bit_packed_shots, bit_packed_predictions=bit_packed_predictions, threads=threads)
                return pred, stats[:, 2]
            return self._lcd.decode_batch(shots, heralds=fired, bit_packed_shots=bit_packed_shots, bit_packed_predictions=bit_packed_predictions, threads=threads)
        t = None if fired is None else rows_to_b8(fired, self._records, False, "triggers")[0]
        preds, failed, raised = call(self._x.decode_batch, packed, n, threads, t)
        _failed(list(np.flatnonzero(np.frombuffer(failed, dtype=np.uint8))), "no correction explains its detection events")
        pred = self._predictions(preds, n, bit_packed_predictions)
        return (pred, np.frombuffer(raised, dtype="<u4").astype(np.int64)) if return_raised else pred


def _graph_edges(model: Any) -> set:
    """The decomposed model's graph edges, as sorted pairs (the boundary as num_detectors)."""
    edges = call(_core.DemLocalClustering, model._d, None, []).edges
    return {(min(u, v), max(u, v)) for u, v in edges}


class ArgumentReweighting(_DemDecoder):
    """Post-selection by argument reweighting (Xie, Yoshioka, Tsubouchi and Li, arXiv
    2601.17757): decode, make the correction less likely, decode again, and keep the shot only
    if the answer holds. ``test="ratio"`` raises each of the correction's fault priors p to p^b
    (the paper's practical ratio test), ``"exact_ratio"`` also raises every other fault's to
    1 − (1 − p)^b so the correction's likelihood is exactly its b-th power, and ``"gap"`` takes
    e^-b off the correction's likelihood, shared among its faults. A shot is kept when the second
    correction is the first (``criterion="pec"``), has its logical effect (``"2r-lec"``), or when
    a third decode, with the second correction suppressed too, still has it (``"3r-lec"``, the
    paper's best at low rejection). A shot with nothing to correct is kept. Larger b rejects
    more.

    ``decoder`` is ``"matching"``, ``"union_find"`` (decomposed models), ``"bposd"``,
    ``"bplsd"`` or ``"relay_bp"``, with that class's options as keywords. ``decode_batch``
    returns the first decode's predictions and which shots are kept.

    >>> import stabilizer_qec as sq
    >>> c = sq.memory_circuit(distance=5, rounds=5, p=0.002)
    >>> dets, obs = c.compile_detector_sampler(seed=1).sample(10_000, separate_observables=True)
    >>> ar = sq.ArgumentReweighting(c.detector_error_model(decompose_errors=True), b=1.2)
    >>> pred, kept = ar.decode_batch(dets)
    >>> wrong = (pred != obs).any(axis=1)
    >>> bool(wrong[kept].mean() < wrong.mean() / 2 and kept.mean() > 0.9)
    True
    """

    __slots__ = ()

    def __init__(self, model: ModelLike, decoder: str = "matching", *, b: float = 1.1, test: str = "ratio", criterion: str = "3r-lec", **decoder_options: Any) -> None:
        model = self._made(_model(model), decoder=decoder, b=b, test=test, criterion=criterion, **decoder_options)
        bp = {"bposd": BpOsd, "bplsd": BpLsd, "relay_bp": RelayBp}
        inner = None
        if decoder in bp:
            inner = bp[decoder](model, **decoder_options)._x
        elif decoder_options:
            raise TypeError(f"the {decoder} decoder takes no options here, not {', '.join(decoder_options)}")
        self._x = call(_core.DemArgumentReweighting, model._d, decoder, inner, real(b, "b"), test, criterion)

    def decode(self, syndrome: Any) -> tuple:
        """One shot: (its predicted observable flips, whether it is kept)."""
        s = np.asarray(syndrome)
        if s.ndim != 1:
            raise ValueError(f"a syndrome is 1-dimensional, not {s.ndim}-dimensional; use decode_batch for many")
        pred, kept = self.decode_batch(s.reshape(1, -1))
        return pred[0], bool(kept[0])

    def decode_batch(
        self,
        shots: Any,
        *,
        return_rounds: bool = False,
        bit_packed_shots: bool = False,
        bit_packed_predictions: bool = False,
        threads: int = 1,
    ) -> tuple:
        """``(predictions, kept)``: the first decode's predictions (as ``Matching.decode_batch``)
        and, per shot, whether it is kept. ``return_rounds`` also returns the decodes each shot
        ran (1 to 3). A shot no correction explains raises ``ValueError``."""
        packed, n, threads = self._shots(shots, bit_packed_shots, threads)
        preds, flags, rounds = call(self._x.decode_batch, packed, n, threads)
        flags = np.frombuffer(flags, dtype=np.uint8)
        _failed(list(np.flatnonzero(flags == 2)), "no correction explains its detection events")
        out = (self._predictions(preds, n, bit_packed_predictions), flags == 1)
        return out + (np.frombuffer(rounds, dtype=np.uint8).astype(np.int64),) if return_rounds else out


def _inner_decoders() -> dict:
    return {
        "matching": Matching,
        "union_find": UnionFind,
        "collision_clustering": CollisionClustering,
        "local_clustering": LocalClustering,
        "belief_matching": BeliefMatching,
        "bposd": BpOsd,
        "bplsd": BpLsd,
        "relay_bp": RelayBp,
        "search": SearchDecoder,
        "ambiguity_clustering": AmbiguityClustering,
        "color_matching": ColorMatching,
    }


def _inner(decoder: Any) -> Any:
    if isinstance(decoder, str):
        table = _inner_decoders()
        if decoder not in table:
            raise ValueError(f"decoder must be one of {', '.join(map(repr, table))} or a callable taking a model, not {decoder!r}")
        return table[decoder]
    if not callable(decoder):
        raise TypeError(f"decoder must be a name or a callable taking a model, not {type(decoder).__name__}")
    return decoder


def _restricted(model: DetectorErrorModel, detectors: list, observables: list) -> tuple:
    """The model on ``detectors`` (renumbered in order) and ``observables``: the faults that flip
    one of the detectors, each kept to those detectors and observables (a decomposed fault's
    parts kept where they flip one of them, with their own observables). Also, per fault of the
    model, whether it is kept, and its observables kept."""
    det = {d: i for i, d in enumerate(detectors)}
    obs = {o: i for i, o in enumerate(observables)}
    coords = model.get_detector_coordinates(list(detectors)) if detectors else {}
    lines = []
    for i, d in enumerate(detectors):
        c = coords.get(d)
        lines.append(f"detector({', '.join(repr(float(x)) for x in c)}) D{i}" if c else f"detector D{i}")
    lines += [f"logical_observable L{k}" for k in range(len(observables))]
    kept = []
    for inst in model.flattened():
        if inst.type != "error":
            continue
        parts, part = [], ([], [], False)
        for t in inst.targets_copy():
            if t.is_separator():
                parts.append(part)
                part = ([], [], False)
            elif t.is_relative_detector_id():
                part = (part[0] + ([f"D{det[t.val]}"] if t.val in det else []), part[1], True)
            elif t.is_logical_observable_id():
                part = (part[0], part[1] + ([t.val] if t.val in obs else []), part[2])
        parts.append(part)
        # A part with no detectors at all goes with the fault's first part kept.
        loose = [o for dets, os, any_det in parts if not any_det for o in os]
        parts = [(dets, os) for dets, os, any_det in parts if dets]
        if not parts:
            kept.append(None)
            continue
        parts[0] = (parts[0][0], parts[0][1] + loose)
        flipped = sorted({o for _, os in parts for o in os})
        kept.append(flipped)
        text = " ^ ".join(" ".join(dets + [f"L{obs[o]}" for o in os]) for dets, os in parts)
        lines.append(f"error({inst.args_copy()[0]!r}) {text}")
    return DetectorErrorModel("\n".join(lines)), kept


class _Parts:
    """A wrapper's decoders: per part, its detectors, its observables and its decoder."""

    __slots__ = ("num_detectors", "num_observables", "parts")

    def __init__(self, num_detectors: int, num_observables: int, parts: list) -> None:
        self.num_detectors, self.num_observables, self.parts = num_detectors, num_observables, parts


class SubgraphDecoder(_DemDecoder):
    """Decodes a model by subgraphs, as qLDPC's ``SubgraphDecoder``: each subgraph is a set of
    detectors (and optionally the observables it predicts), decoded alone by an inner decoder on
    the model restricted to its detectors and the faults that flip them. The subgraphs'
    predictions combine by exclusive or, so each observable flip should be predicted by exactly
    one subgraph: a fault two subgraphs both see, flipping an observable both own, warns (their
    predictions would cancel), as does a detector in no subgraph.

    ``decoder`` is a built-in decoder's name (``"matching"``, ``"bposd"``, ...; as sinter's)
    or a callable that makes a decoder from a ``DetectorErrorModel``. For decoding a CSS code's X
    and Z sectors apart, or separate patches, ``CompositeDecoder`` finds the subgraphs itself.

    >>> import stabilizer_qec as sq
    >>> dem = sq.DetectorErrorModel("error(0.1) D0 L0\\nerror(0.1) D0 D1\\nerror(0.1) D2 L1\\nerror(0.1) D2 D3")
    >>> dec = sq.SubgraphDecoder(dem, [[0, 1], [2, 3]], [[0], [1]])
    >>> dec.decode([1, 0, 0, 1]).tolist()
    [1, 1]
    """

    __slots__ = ()

    def __init__(self, model: ModelLike, subgraph_detectors: Any, subgraph_observables: Any = None, *, decoder: Any = "matching") -> None:
        dets = [sorted({int(d) for d in ds}) for ds in subgraph_detectors]
        obs = None if subgraph_observables is None else [sorted({int(o) for o in os}) for os in subgraph_observables]
        model = self._made(_model(model), subgraph_detectors=dets, subgraph_observables=obs, decoder=decoder)
        self._build(model, dets, obs, decoder, warn=True)

    def _build(self, model: DetectorErrorModel, dets: list, obs: Union[list, None], decoder: Any, warn: bool) -> None:
        import warnings

        make = _inner(decoder)
        nd, no = model.num_detectors, model.num_observables
        if obs is not None and len(obs) != len(dets):
            raise ValueError(f"{len(dets)} subgraphs of detectors but {len(obs)} sets of observables")
        if obs is None:
            obs = [list(range(no))] * len(dets)
        for ds, os in zip(dets, obs):
            if ds and not 0 <= ds[-1] < nd or ds and ds[0] < 0:
                raise ValueError(f"a subgraph's detectors must be in 0..{nd - 1}")
            if os and not (0 <= os[0] and os[-1] < no):
                raise ValueError(f"a subgraph's observables must be in 0..{no - 1}")
        parts = []
        predictors: dict = {}
        covered = set()
        for ds, os in zip(dets, obs):
            sub, kept = _restricted(model, ds, os)
            covered.update(ds)
            for k, flipped in enumerate(kept):
                for o in flipped or ():
                    predictors[k, o] = predictors.get((k, o), 0) + 1
            parts.append((np.array(ds, dtype=np.int64), np.array(os, dtype=np.int64), make(sub) if sub.num_errors and os else None))
        if warn:
            twice = sorted({o for (_, o), c in predictors.items() if c > 1})
            if twice:
                warnings.warn(f"subgraphs share faults that flip observables {twice} they both predict: their predictions cancel", stacklevel=3)
            if len(covered) < nd:
                warnings.warn(f"{nd - len(covered)} detectors are in no subgraph", stacklevel=3)
        self._x = _Parts(nd, no, parts)

    @property
    def num_subgraphs(self) -> int:
        return len(self._x.parts)

    @property
    def subgraphs(self) -> list:
        """Per subgraph, its detectors and observables."""
        return [(ds.tolist(), os.tolist()) for ds, os, _ in self._x.parts]

    def decode_batch(
        self,
        shots: Any,
        *,
        bit_packed_shots: bool = False,
        bit_packed_predictions: bool = False,
        threads: int = 1,
    ) -> np.ndarray:
        """As ``Matching.decode_batch``: each subgraph's inner decoder on its detectors' columns,
        their predictions combined by exclusive or."""
        import inspect

        packed, n, threads = self._shots(shots, bit_packed_shots, threads)
        nd, no = self._x.num_detectors, self._x.num_observables
        rows = np.unpackbits(np.frombuffer(packed, dtype=np.uint8).reshape(n, -1), axis=1, count=nd, bitorder="little").astype(bool)
        pred = np.zeros((n, no), dtype=np.uint8)
        for ds, os, dec in self._x.parts:
            if dec is None or not len(os):
                continue
            kw = {"threads": threads} if "threads" in inspect.signature(dec.decode_batch).parameters else {}
            pred[:, os] ^= np.asarray(dec.decode_batch(rows[:, ds], **kw), dtype=np.uint8).reshape(n, len(os))
        return np.packbits(pred, axis=1, bitorder="little") if bit_packed_predictions else pred


class CompositeDecoder(SubgraphDecoder):
    """Decodes each independent part of a model alone: its detectors split into the components
    the faults connect (separate patches or blocks; in a decomposed model, whose parts count as
    faults, a CSS code's X and Z sectors too, which its Y faults join otherwise), each
    decoded by its own inner decoder on its own faults and observables, the predictions combined
    by exclusive or. Every fault lies in one component, so this is exact: the predictions are the
    inner decoder's on each part (for matching, the same as matching the whole model), and a
    decoder that grows with the model (BP+OSD's elimination, a search) runs on the parts.

    ``decoder`` is as ``SubgraphDecoder``'s. ``subgraphs`` lists the components found.

    >>> import stabilizer_qec as sq
    >>> dem = sq.DetectorErrorModel("error(0.1) D0 L0\\nerror(0.1) D0 D1\\nerror(0.1) D2 L1\\nerror(0.1) D2 D3")
    >>> sq.CompositeDecoder(dem).subgraphs
    [([0, 1], [0]), ([2, 3], [1])]
    """

    __slots__ = ()

    def __init__(self, model: ModelLike, *, decoder: Any = "matching") -> None:
        model = self._made(_model(model), decoder=decoder)
        nd = model.num_detectors
        parent = list(range(nd))

        def find(v: int) -> int:
            while parent[v] != v:
                parent[v] = parent[parent[v]]
                v = parent[v]
            return v

        # Components of a decomposed model are its parts' (a Y fault's X and Z parts apart).
        owned: dict = {}
        parts = []
        for inst in model.flattened():
            if inst.type != "error":
                continue
            ds: list = []
            os: list = []
            for t in list(inst.targets_copy()) + [None]:
                if t is None or t.is_separator():
                    if ds:
                        for d in ds[1:]:
                            a, b = find(ds[0]), find(d)
                            if a != b:
                                parent[max(a, b)] = min(a, b)
                        parts.append((ds[0], os))
                    ds, os = [], []
                elif t.is_relative_detector_id():
                    ds.append(t.val)
                elif t.is_logical_observable_id():
                    os.append(t.val)
        for d, os in parts:
            owned.setdefault(find(d), set()).update(os)
        members: dict = {}
        for d in range(nd):
            members.setdefault(find(d), []).append(d)
        roots = [r for r in sorted(members) if r in owned]
        dets = [members[r] for r in roots]
        obs = [sorted(owned[r]) for r in roots]
        self._build(model, dets, obs, decoder, warn=False)


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


# A colour shot's status from ``DemColorMatcher.decode_batch`` (src/batch.rs ``color_shots``):
# 3 where the Möbius matcher found no perfect matching, and PyMatching's message for it.
_COLOR_UNMATCHABLE = 3
_NO_PERFECT_MATCHING = "No perfect matching could be found. This likely means that the syndrome has odd parity in the support of a connected component without a boundary."


class ColorMatching(_DemDecoder):
    """Colour-code decoding by matching, Chromobius's construction (Gidney and Jones, 2023):
    each detector doubled into the two sub-graphs leaving out a colour other than its own,
    every fault split into basic faults drawn as edges of that doubled (Möbius) graph, a
    minimum-weight matching of it found by PyMatching 2.2.2's sparse blossom (the matcher
    Chromobius links, ported exactly), and the matching lifted back to the code by carrying
    colour charge around its cycles. Predictions and weights equal to the ``chromobius``
    package's (on macOS on every shot checked; elsewhere Chromobius's std::unordered_map
    order may drag charge another way on some shots). Edges of probability above 1/2 are
    negative-weight edges, as in Chromobius; probability 1 is refused.

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
        preds, weights, status, failed = call(self._x.decode_batch, packed, n, threads)
        if failed:
            _failed(failed, _NO_PERFECT_MATCHING if status[failed[0]] == _COLOR_UNMATCHABLE else "no lifting of the matching explains the detection events (are the colour annotations right?)")
        out = self._predictions(preds, n, bit_packed_predictions)
        if return_weights:
            return out, np.frombuffer(weights, dtype="<f8").copy()
        return out

    def _decode_with_ties(self, shots: Any) -> tuple:
        packed, n, _ = self._shots(shots, False, 1)
        preds, weights, status, failed = call(self._x.decode_batch, packed, n, 1)
        return self._predictions(preds, n, False), np.frombuffer(weights, dtype="<f8").copy(), np.frombuffer(status, dtype=np.uint8) == 1, failed


class SearchDecoder(_DemDecoder):
    """A search for the most likely error, Tesseract's A* (Beni, Higgott and Shutty, Google,
    2025), for any detector error model: states are sets of faults, expanded cheapest first
    by their weight plus an admissible estimate of explaining the detectors still lit, each
    grown only by faults through its first lit detector in a detector order. ``beam`` drops
    states lighting more than that many detectors above the fewest seen (``None``: no beam);
    ``no_revisit_dets`` skips patterns of lit detectors already expanded (faster, not exact);
    ``pqlimit`` bounds the queue (``None``: no bound); the search runs once per detector order
    (``det_orders``, or ``num_det_orders`` generated by ``det_order_method``, ``"index"``,
    ``"bfs"`` or ``"coordinate"``, from ``seed``) and keeps the cheapest answer. Defaults are
    Tesseract's. With no beam, no queue bound and ``no_revisit_dets=False`` it is exact.

    Answers equal to the ``tesseract_decoder`` package's for the same detector orders (both
    break ties between equally promising states as the platform's C++ library does; index
    orders are generated as Tesseract does, the others are this package's own). A shot the
    search gives up on predicts no flips; ``return_low_confidence`` says which."""

    __slots__ = ()

    def __init__(
        self,
        model: ModelLike,
        *,
        beam: Union[int, None] = 5,
        beam_climbing: bool = False,
        no_revisit_dets: bool = True,
        pqlimit: Union[int, None] = 200_000,
        det_orders: Any = None,
        num_det_orders: Union[int, None] = None,
        det_order_method: str = "index",
        seed: int = 2384753,
        det_penalty: float = 0.0,
        merge_errors: bool = True,
        _flavour: str = "native",
    ) -> None:
        options = dict(beam=beam, beam_climbing=beam_climbing, no_revisit_dets=no_revisit_dets, pqlimit=pqlimit, det_orders=det_orders, num_det_orders=num_det_orders,
                       det_order_method=det_order_method, seed=seed, det_penalty=det_penalty, merge_errors=merge_errors, _flavour=_flavour)
        model = self._made(_model(model), **options)
        orders = None if det_orders is None else [[count(d, "det_orders entries") for d in order] for order in det_orders]
        if orders is not None and num_det_orders is not None:
            raise ValueError("give det_orders or num_det_orders, not both")
        self._x = call(
            _core.DemSearch,
            model._d,
            None if beam is None else count(beam, "beam"),
            bool(beam_climbing),
            bool(no_revisit_dets),
            None if pqlimit is None else count(pqlimit, "pqlimit"),
            orders,
            20 if num_det_orders is None else count(num_det_orders, "num_det_orders", 1),
            det_order_method,
            count(seed, "seed"),
            real(det_penalty, "det_penalty"),
            bool(merge_errors),
            _flavour,
        )

    @property
    def det_orders(self) -> list:
        """The detector orders searched."""
        return [list(o) for o in self._x.orders]

    def decode_to_faults(self, syndrome: Any) -> list:
        """The faults the search found for one shot, as indices into the model's ``error``
        lines (flattened), or ``[]`` where it gave up."""
        s = np.asarray(syndrome)
        if s.ndim != 1 or s.shape[0] != self.num_detectors:
            raise ValueError(f"a syndrome has {self.num_detectors} bits, not shape {s.shape}")
        return list(self._x.decode_to_faults([int(d) for d in np.flatnonzero(s)])[0])

    def decode_batch(
        self,
        shots: Any,
        *,
        return_weights: bool = False,
        return_low_confidence: bool = False,
        bit_packed_shots: bool = False,
        bit_packed_predictions: bool = False,
        threads: int = 1,
    ) -> Any:
        """As ``Matching.decode_batch``; ``return_weights`` also returns each shot's found cost
        (Σ −ln(p/(1 − p)) over its faults), ``return_low_confidence`` whether the search gave
        up (then the prediction is no flips)."""
        packed, n, threads = self._shots(shots, bit_packed_shots, threads)
        preds, costs, gave_up = call(self._x.decode_batch, packed, n, threads)
        out = [self._predictions(preds, n, bit_packed_predictions)]
        if return_weights:
            out.append(np.frombuffer(costs, dtype="<f8").copy())
        if return_low_confidence:
            out.append(np.frombuffer(gave_up, dtype=np.uint8).astype(bool))
        return out[0] if len(out) == 1 else tuple(out)


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
        if not np.all(m.data == 1):
            raise ValueError("Input matrix must be a binary matrix.")
        rows, cols = m.shape
        cols_list = [sorted(int(r) for r in m.indices[m.indptr[j] : m.indptr[j + 1]]) for j in range(cols)]
        return rows, cols, cols_list
    a = np.asarray(pcm)
    if a.ndim != 2:
        raise ValueError(f"a check matrix is 2-dimensional, not {a.ndim}-dimensional")
    if not (np.issubdtype(a.dtype, np.integer) or np.issubdtype(a.dtype, np.bool_) or np.issubdtype(a.dtype, np.floating)):
        raise TypeError(f"a check matrix holds 0/1 numbers or bools, not {a.dtype}")
    # Entries 0 and 1 only (ldpc's floats included), as ldpc and PyMatching require.
    if not np.isin(a, (0, 1)).all():
        raise ValueError("Input matrix must be a binary matrix.")
    a = a != 0
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


from ._ldpc import BeliefFindDecoder, BpDecoder, BpFlipDecoder, BpLsdDecoder, BpOsdDecoder, LsdDecoder, MbpDecoder, SoftInfoBpDecoder, SoftInfoBpOsdDecoder, UnionFindDecoder  # noqa: E402 - they use the helpers above

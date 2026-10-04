"""Circuits the engine writes: surface-code memories, IBM's bivariate bicycle codes and their
logical operations, and the streamed million-round memory."""


from typing import NamedTuple, Sequence, Union

import numpy as np

from . import _core
from ._circuit import Circuit
from ._decoders import Window
from ._util import call, count, probability, real, seed_of

_CODES = ("rotated", "xzzx")
_BASES = ("z", "x")


def _choice(value: str, options: tuple, name: str) -> str:
    if value not in options:
        raise ValueError(f"{name} must be one of {', '.join(map(repr, options))}, not {value!r}")
    return value


def memory_circuit(
    code: str = "rotated",
    *,
    distance: int,
    rounds: int,
    p: float,
    noise: str = "sd6",
    eta: float = 0.5,
    basis: str = "z",
) -> Circuit:
    """A surface-code memory experiment: a distance-``distance`` patch (``"rotated"`` or
    ``"xzzx"``, distance 2 to 11) prepared in ``basis``, held for ``rounds`` rounds of syndrome
    extraction (at most 10,000), and read out; detectors on every check, one observable.

    ``noise="sd6"`` is the standard circuit-level model at strength ``p`` (two-qubit
    depolarizing after every CNOT, single-qubit after every gate and on every idle qubit,
    flips on every reset and measurement), as Stim's generated circuits use. ``"current"`` is
    the engine's own model, with bias ``eta`` = p_Z / (p_X + p_Y)."""
    _choice(code, _CODES, "code")
    _choice(noise, ("sd6", "current"), "noise")
    _choice(basis, _BASES, "basis")
    text = call(_core.generate_circuit, code, count(distance, "distance"), count(rounds, "rounds"), noise, probability(p), real(eta, "eta"), basis)
    return Circuit(text)


def _rows(rows: list, n: int) -> np.ndarray:
    m = np.zeros((len(rows), n), dtype=np.uint8)
    for i, r in enumerate(rows):
        m[i, r] = 1
    return m


class Automorphism(NamedTuple):
    """A shift x^a y^b of both halves of the data, with or without Bravyi et al.'s
    ZX-duality, and its action on the logical qubits: ``action[i]`` is the image of basis
    element i (the X logicals, then the Z logicals, in ``logicals()``'s basis), over GF(2)."""

    shift: tuple
    dual: bool
    action: np.ndarray


class Gauging(NamedTuple):
    """The gauging ancilla system that measures a gross-code logical X (Williamson and Yoder's
    construction, as Cross, He, Rall and Yoder build it). ``support`` is the operator's data
    qubits; ``edges`` the Z checks touching them, each an edge qubit; ``extra_edges`` the
    edges added for expansion, as pairs of vertices; ``incidence`` each edge's ends (indices
    into ``support``); ``gauss`` each vertex's edges; ``flux`` the flux checks (edge
    indices); ``hx`` and ``hz`` the deformed code's checks over data then edge qubits;
    ``ticks`` the merged cycle's depth; ``worst_cut`` (edges leaving, vertices) of the
    sparsest cut."""

    support: list
    edges: list
    extra_edges: list
    incidence: list
    gauss: list
    flux: list
    hx: np.ndarray
    hz: np.ndarray
    ticks: int
    worst_cut: tuple


class BivariateBicycleCode:
    """One of IBM's bivariate bicycle codes (Bravyi et al., Nature 627, 778, 2024), with the
    paper's depth-8 syndrome cycle: by its number of data qubits, the codes of the paper's
    Table 3, ``"72"`` ([[72, 12, 6]]), ``"90"`` ([[90, 8, 10]]), ``"108"`` ([[108, 8, 10]]),
    ``"144"`` or ``"gross"`` (the gross code, [[144, 12, 12]]) and ``"288"`` ([[288, 12, 18]]);
    or any other from its polynomials with ``from_polynomials``."""

    def __init__(self, name: Union[str, int] = "gross") -> None:
        name = str(name)
        self._init(call(_core.bb_spec, name, None), "gross" if name in ("gross", "144") else name)

    @classmethod
    def from_polynomials(cls, l: int, m: int, a: Sequence, b: Sequence) -> "BivariateBicycleCode":
        """The code of A = Σ x^i y^j over the three ``(i, j)`` in ``a`` and B likewise over
        ``b``, on an ``l`` × ``m`` torus: H_X = [A | B], H_Z = [Bᵀ | Aᵀ]. Each polynomial's
        three monomials must differ, and the code must encode a qubit.

        >>> BivariateBicycleCode.from_polynomials(12, 6, [(3, 0), (0, 1), (0, 2)], [(0, 3), (1, 0), (2, 0)]).k
        12
        """
        monomials = []
        for poly, name in ((a, "a"), (b, "b")):
            terms = [tuple(count(e, f"{name}'s exponents") for e in t) for t in poly]
            if len(terms) != 3 or any(len(t) != 2 for t in terms):
                raise ValueError(f"{name} must be three monomials (i, j), for x^i y^j")
            monomials.append(terms)
        code = cls.__new__(cls)
        code._init(call(_core.bb_spec, None, (count(l, "l", 1), count(m, "m", 1), *monomials)), None)
        return code

    def _init(self, spec: tuple, name: Union[str, None]) -> None:
        self._spec = spec
        self.name = name
        hx, hz, lx, lz = call(_core.bb_matrices, spec)
        self.n = 2 * len(hx)
        self._hx, self._hz = _rows(hx, self.n), _rows(hz, self.n)
        self._lx, self._lz = _rows(lx, self.n), _rows(lz, self.n)
        self.k = len(lx)

    @property
    def polynomials(self) -> tuple:
        """``(l, m, a, b)``: the torus and A's and B's monomials, as ``from_polynomials`` takes
        them."""
        l, m, a, b = self._spec
        return l, m, [tuple(t) for t in a], [tuple(t) for t in b]

    def __repr__(self) -> str:
        if self.name is not None:
            return f"stabilizer_qec.BivariateBicycleCode({self.name!r})"
        l, m, a, b = self.polynomials
        return f"stabilizer_qec.BivariateBicycleCode.from_polynomials({l}, {m}, {a}, {b})"

    def check_matrices(self) -> tuple:
        """(H_X, H_Z), each check a row over the ``n`` data qubits, uint8."""
        return self._hx.copy(), self._hz.copy()

    def logicals(self) -> tuple:
        """(logical X, logical Z), ``k`` rows each over the data qubits, paired so that
        X_i and Z_j anticommute exactly when i = j."""
        return self._lx.copy(), self._lz.copy()

    def automorphisms(self) -> list:
        """Every shift x^a y^b, with and without the ZX-duality, as an ``Automorphism``."""
        return [
            Automorphism((a, b), bool(dual), _rows(rows, 2 * self.k))
            for a, b, dual, rows in call(_core.bb_automorphisms, self._spec)
        ]

    def memory_circuit(self, cycles: int, p: float, *, basis: str = "z") -> Circuit:
        """A memory: the data prepared in ``basis``, ``cycles`` syndrome cycles (at most
        10,000) under circuit noise ``p`` (Bravyi et al.'s model), the data read out. The
        observables are the ``k`` logicals of that basis."""
        _choice(basis, _BASES, "basis")
        cycles, p = count(cycles, "cycles", 1), probability(p)
        if basis == "z":
            return Circuit(call(_core.bb_memory_circuit, self._spec, cycles, p))
        return Circuit(call(_core.bb_memory_basis_circuit, self._spec, basis, cycles, p))

    def _gross_only(self) -> None:
        if self._spec != call(_core.bb_spec, "gross", None):
            raise ValueError("logical measurements are built for the gross code only")

    def gauging(self, operator: str, *, expanded: bool = False) -> Gauging:
        """The ancilla system that measures ``operator``: ``"f"`` (Bravyi et al.'s X(f, 0)),
        ``"gh"`` (X(g, h)) or ``"f+gh"`` (their product). ``expanded`` adds edges until every
        set of at most half the vertices has as many edges leaving it as vertices (Cheeger
        constant at least 1), which keeps the code's distance while it is measured."""
        self._gross_only()
        support, edges, extra, incidence, gauss, flux, hx, hz, ticks, cut = call(_core.bb_gauging, operator, bool(expanded))
        n = self.n + len(incidence)
        return Gauging(support, edges, [tuple(e) for e in extra], incidence, gauss, flux, _rows(hx, n), _rows(hz, n), ticks, tuple(cut))

    def logical_measurement_circuit(
        self,
        operator: str,
        basis: str,
        *,
        pre: int,
        merged: int,
        post: int,
        p: float,
        expanded: bool = False,
    ) -> Circuit:
        """The gauging measurement of ``operator`` (see ``gauging``) as a circuit: ``pre``
        memory cycles, ``merged`` cycles of the deformed code, ``post`` memory cycles, the
        data read in ``basis``. In ``"x"`` the observables are the outcome (L0) and the 12 X
        logicals; in ``"z"``, the 11 Z logicals the measurement keeps."""
        self._gross_only()
        _choice(basis, _BASES, "basis")
        return Circuit(
            call(
                _core.bb_logical_measurement_circuit,
                operator,
                basis,
                count(pre, "pre"),
                count(merged, "merged"),
                count(post, "post"),
                probability(p),
                bool(expanded),
            )
        )


class StreamResult(NamedTuple):
    """A streamed memory's outcome: ``failures`` of ``shots`` streams, the first stream of
    each batch's per-window decode seconds (batches × windows), the windows, and the wall
    time."""

    failures: int
    shots: int
    window_seconds: np.ndarray
    windows: list
    seconds: float


def stream_memory(
    code: str = "rotated",
    *,
    distance: "int | None" = None,
    rounds: "int | None" = None,
    p: "float | None" = None,
    commit: int,
    buffer: int,
    mode: str = "parallel",
    enable_correlations: bool = False,
    shots: int = 64,
    seed: "int | None" = None,
    threads: int = 0,
    circuit: "Circuit | None" = None,
) -> StreamResult:
    """A memory too long to model whole, sampled round by round and window-decoded as it
    streams: an SD6 memory of ``rounds`` rounds (a million is fine), its windows' graphs
    built once from a short template. ``shots`` is rounded up to a multiple of 64. The same
    seed gives the same failures on any number of threads.

    Or any ``circuit`` with a loop (Stim's generated ones among them), in place of ``code``,
    ``distance``, ``rounds`` and ``p``: its windows come from a template of its folded error
    model, and its detectors need a time coordinate (their last)."""
    batches = max(1, -(-count(shots, "shots", 1) // 64))
    if circuit is not None:
        if not isinstance(circuit, Circuit):
            circuit = Circuit(circuit)
        if distance is not None or rounds is not None or p is not None:
            raise ValueError("give a circuit, or a code's distance, rounds and p, not both")
        failures, total, unexplained, times, info, seconds = call(
            _core.stream_circuit_decode,
            circuit._c,
            count(commit, "commit"),
            count(buffer, "buffer"),
            mode,
            bool(enable_correlations),
            batches,
            seed_of(seed),
            count(threads, "threads"),
        )
    else:
        if distance is None or rounds is None or p is None:
            raise TypeError("stream_memory needs distance, rounds and p (or a circuit)")
        _choice(code, _CODES, "code")
        failures, total, unexplained, times, info, seconds = call(
            _core.stream_decode,
            code,
            count(distance, "distance"),
            probability(p),
            count(rounds, "rounds", 1),
            count(commit, "commit"),
            count(buffer, "buffer"),
            mode,
            bool(enable_correlations),
            batches,
            seed_of(seed),
            count(threads, "threads"),
        )
    if unexplained:
        raise RuntimeError(f"internal error in stabilizer-qec: window commits left {unexplained} defects unexplained; please report it")
    windows = [Window(*w) for w in info]
    seconds_per = np.frombuffer(times, dtype="<f8").reshape(batches, len(windows)).copy()
    return StreamResult(failures, total, seconds_per, windows, seconds)

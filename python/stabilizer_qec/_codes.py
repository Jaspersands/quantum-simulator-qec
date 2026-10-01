"""Circuits the engine writes: surface-code memories, IBM's bivariate bicycle codes and their
logical operations, and the streamed million-round memory."""

from __future__ import annotations

from typing import NamedTuple

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
    """One of IBM's bivariate bicycle codes (Bravyi et al., Nature 627, 778, 2024):
    ``"gross"``, the [[144, 12, 12]] gross code, or ``"72"``, [[72, 12, 6]], with the paper's
    depth-8 syndrome cycle."""

    def __init__(self, name: str = "gross") -> None:
        name = {"144": "gross"}.get(name, name)
        self.name = _choice(name, ("gross", "72"), "name")
        hx, hz, lx, lz = call(_core.bb_matrices, self.name)
        self.n = 2 * len(hx)
        self._hx, self._hz = _rows(hx, self.n), _rows(hz, self.n)
        self._lx, self._lz = _rows(lx, self.n), _rows(lz, self.n)
        self.k = len(lx)

    def __repr__(self) -> str:
        return f"stabilizer_qec.BivariateBicycleCode({self.name!r})"

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
            for a, b, dual, rows in call(_core.bb_automorphisms, self.name)
        ]

    def memory_circuit(self, cycles: int, p: float, *, basis: str = "z") -> Circuit:
        """A memory: the data prepared in ``basis``, ``cycles`` syndrome cycles (at most
        10,000) under circuit noise ``p`` (Bravyi et al.'s model), the data read out. The
        observables are the ``k`` logicals of that basis."""
        _choice(basis, _BASES, "basis")
        cycles, p = count(cycles, "cycles", 1), probability(p)
        if basis == "z":
            return Circuit(call(_core.bb_memory_circuit, self.name, cycles, p))
        return Circuit(call(_core.bb_memory_basis_circuit, self.name, basis, cycles, p))

    def _gross_only(self) -> None:
        if self.name != "gross":
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
    distance: int,
    rounds: int,
    p: float,
    commit: int,
    buffer: int,
    mode: str = "parallel",
    enable_correlations: bool = False,
    shots: int = 64,
    seed: "int | None" = None,
    threads: int = 0,
) -> StreamResult:
    """A memory too long to model whole, sampled round by round and window-decoded as it
    streams: an SD6 memory of ``rounds`` rounds (a million is fine), its windows' graphs
    built once from a short template. ``shots`` is rounded up to a multiple of 64. The same
    seed gives the same failures on any number of threads."""
    _choice(code, _CODES, "code")
    batches = max(1, -(-count(shots, "shots", 1) // 64))
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

"""A surface-code simulator and decoder, in Rust.

Circuits and detector error models are Stim's text formats. Shots cross the
boundary as bytes in Stim's b8 layout: one row per shot, bit ``i`` of a row in
byte ``i // 8`` at position ``i % 8`` (numpy's ``packbits(..., bitorder="little")``).
Predictions come back as little-endian ``u64`` per shot and weights as
little-endian ``f64``; read them with ``numpy.frombuffer(b, "<u8")`` and
``numpy.frombuffer(b, "<f8")``.

``threads = 0`` means every core.
"""

from __future__ import annotations

from typing import Literal

Window = tuple[int, int, int, int, int]
"""A window: (first layer, end layer, commit start, commit end, phase).
Layers are rounds of detectors; the window decodes layers [first, end) and
commits the corrections in [commit start, commit end). Phase 0 windows run
first; in parallel mode, phase 1 windows wait for their phase 0 neighbours."""

def generate_circuit(
    code: Literal["rotated", "xzzx"],
    d: int,
    rounds: int,
    noise: Literal["sd6", "current"],
    p: float,
    eta: float = 0.5,
    basis: Literal["z", "x"] = "z",
) -> str:
    """A memory experiment as a Stim circuit: a distance-``d`` patch held for
    ``rounds`` rounds of syndrome extraction, with detectors and one
    observable. ``noise="sd6"`` is the standard depolarising model at ``p``;
    ``noise="current"`` is the biased model with bias ``eta``."""

def dem_from_circuit(circuit_text: str, decompose: bool = False) -> str:
    """The circuit's detector error model, built by walking the circuit
    backwards, as Stim's DEM text. ``decompose`` splits hyperedges into
    graphlike pieces joined by ``^``, as ``decompose_errors=True`` does."""

def circuit_to_stim(text: str) -> str:
    """A circuit through this engine's parser and printer (a round trip)."""

def sample_b8(circuit_text: str, num_shots: int, seed: int) -> tuple[bytes, bytes]:
    """Shots from the frame sampler, one at a time: (detectors, observables)
    as b8 rows."""

def sample_b8_batch(circuit_text: str, num_shots: int, seed: int, threads: int = 1) -> tuple[bytes, bytes, float]:
    """Shots from the bit-parallel sampler, 64 per machine word, running
    ``REPEAT`` blocks without flattening them: (detectors, observables,
    seconds). Each thread samples whole batches from its own stream."""

def m2d_b8(circuit_text: str, meas: bytes, sweeps: bytes, num_shots: int) -> tuple[bytes, bytes]:
    """Raw measurements and sweep bits (b8) to detection events and observable
    flips (b8), as ``stim m2d`` gives them. Pass ``b""`` for ``sweeps`` when
    the circuit has none."""

def b8_to_01(packed: bytes, num_bits: int) -> str:
    """b8 rows of ``num_bits`` bits as Stim's 01 text, one line per row."""

def decode_b8(
    dem_text: str, packed: bytes, num_shots: int, threads: int = 1, correlated: bool = False
) -> tuple[bytes, bytes, int, float]:
    """Decode b8 detection events by exact minimum-weight matching (sparse
    blossom) on a given detector error model, decomposed pieces and all.
    ``correlated=True`` decodes in two passes as PyMatching's
    ``enable_correlations=True`` does. Returns (predictions as u64 per shot,
    solution weights as f64 per shot, shots that failed to decode, seconds).
    A failed shot is predicted as all ones with a NaN weight."""

def decode_b8_own(
    circuit_text: str, packed: bytes, num_shots: int, threads: int = 1, correlated: bool = False
) -> tuple[bytes, bytes, int, float]:
    """As ``decode_b8``, with this engine's own error model of the circuit and
    its own decomposition."""

def decode_b8_window(
    dem_text: str,
    packed: bytes,
    num_shots: int,
    commit: int,
    buffer: int,
    mode: Literal["sliding", "parallel"],
    correlated: bool = False,
    threads: int = 0,
    timings: bool = False,
) -> tuple[bytes, int, bytes, list[Window]]:
    """Window decoding, as a real-time decoder would do it: each window sees
    ``commit`` rounds it commits and ``buffer`` rounds beyond. ``"sliding"``
    windows run one after another; ``"parallel"`` windows run in two layers
    (Skoric et al., Tan et al.) so many cores can share one stream. Returns
    (predictions as u64 per shot, u64 max where a window failed and the
    shot was abandoned; defects left unexplained over the shots decoded,
    zero unless something is wrong; each
    window's decode time per shot in seconds as f64, shots × windows, when
    ``timings``; the windows)."""

def stream_decode(
    code: Literal["rotated", "xzzx"],
    d: int,
    p: float,
    rounds: int,
    commit: int,
    buffer: int,
    mode: Literal["sliding", "parallel"],
    correlated: bool = False,
    batches: int = 1,
    seed: int = 1,
    threads: int = 0,
) -> tuple[int, int, int, bytes, list[Window], float]:
    """A long SD6 memory decoded as it streams: ``batches`` × 64 streams of
    ``rounds`` rounds, sampled round by round and window-decoded with graphs
    built once from a short template, so memory does not grow with
    ``rounds``. A defect still standing in a round no window will read again
    is unexplained. Returns (logical failures, streams, defects left unexplained,
    lane 0's window decode times in seconds as f64, one per window per batch,
    the windows, wall seconds)."""

class Decoder:
    """One model's matcher, shot by shot, for looking inside correlated
    matching. Edges are ``(u, v)`` detector pairs with ``-1`` for the
    boundary, as PyMatching gives them."""

    def __init__(self, dem_text: str) -> None: ...
    def edges(self, defects: list[int]) -> list[tuple[int, int]]:
        """The edges of the first (plain) pass's minimum-weight matching."""
    def pass2(self, defects: list[int], edges: list[tuple[int, int]]) -> tuple[int, float]:
        """The second pass of correlated matching, reweighted from the given
        first-pass edges: (observables, weight)."""

class RotatedSurfaceCode:
    """The rotated surface code of the in-browser explainer, with the simple
    noise models of its first sections (code-capacity and phenomenological)."""

    def __init__(self, d: int) -> None: ...
    @property
    def d(self) -> int: ...
    def simulate(self, num_rounds: int, p: float, bias: float | None = None, decoder_type: int | None = None) -> bool:
        """One phenomenological-noise memory of ``num_rounds`` rounds; True on
        a logical failure. ``decoder_type`` 0 is union-find, 1 greedy, 2 matching."""
    def simulate_data_noise(self, p: float, bias: float | None = None, decoder_type: int | None = None) -> bool:
        """One round of code-capacity noise on the data qubits; True on a
        logical failure."""

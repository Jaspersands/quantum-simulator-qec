"""This package's decoders, and its whole engine, inside sinter's Monte Carlo sweeps.

sinter (Stim's companion for threshold and logical-error-rate sweeps) samples a task's circuit
with Stim and decodes the shots with a decoder it names. ``decoders()`` gives it this
package's decoders under names of their own; ``samplers()`` gives it samplers that run the
whole pipeline here, sampling included::

    import sinter
    from stabilizer_qec import sinter as sq_sinter

    stats = sinter.collect(
        tasks=tasks,
        decoders=["sq_matching", "sq_belief_matching", "pymatching"],
        custom_decoders={**sq_sinter.decoders(), **sq_sinter.samplers()},
        num_workers=8,
        max_shots=100_000,
    )

or ``sinter collect --custom_decoders_module_function "stabilizer_qec.sinter:sinter_decoders"``
from the command line. ``Decoder(kind, **options)`` and ``Sampler(kind, **options)`` make
others: the options are the decoder class's (``Decoder("bposd", osd_order=10)``).

The decoder kinds are ``"matching"`` (``Matching``), ``"correlated_matching"`` (``Matching``
with ``enable_correlations``), ``"belief_matching"`` (``BeliefMatching``) and ``"bposd"``
(``BpOsd``, for codes whose faults flip three or more detectors, such as the bivariate bicycle
codes), and ``"union_find"`` (``UnionFind``, the standard baseline). Requires sinter.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

try:
    import sinter as _sinter
except ImportError as ex:  # pragma: no cover - exercised only without sinter
    raise ImportError("stabilizer_qec.sinter needs sinter: pip install sinter") from ex

from ._circuit import Circuit, DetectorErrorModel
from ._decoders import BeliefMatching, BpOsd, Matching, UnionFind

KINDS = ("matching", "correlated_matching", "belief_matching", "bposd", "union_find")

__all__ = ["KINDS", "Decoder", "Sampler", "decoders", "samplers", "sinter_decoders"]


def _check_kind(kind: str) -> str:
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {', '.join(map(repr, KINDS))}, not {kind!r}")
    return kind


def _build(kind: str, dem: DetectorErrorModel, options: dict) -> Any:
    if kind == "matching":
        return Matching(dem, **options)
    if kind == "correlated_matching":
        return Matching(dem, enable_correlations=True, **options)
    if kind == "belief_matching":
        return BeliefMatching(dem, **options)
    if kind == "union_find":
        return UnionFind(dem, **options)
    return BpOsd(dem, **options)


class _CompiledDecoder(_sinter.CompiledDecoder):
    def __init__(self, decoder: Any) -> None:
        self.decoder = decoder

    def decode_shots_bit_packed(self, *, bit_packed_detection_event_data: np.ndarray) -> np.ndarray:
        return self.decoder.decode_batch(bit_packed_detection_event_data, bit_packed_shots=True, bit_packed_predictions=True)


class Decoder(_sinter.Decoder):
    """A sinter decoder: shots Stim sampled, decoded by one of this package's decoders on the
    task's detector error model. ``options`` go to the decoder's class."""

    def __init__(self, kind: str = "matching", **options: Any) -> None:
        self.kind = _check_kind(kind)
        self.options = options

    def __repr__(self) -> str:
        args = "".join(f", {k}={v!r}" for k, v in self.options.items())
        return f"stabilizer_qec.sinter.Decoder({self.kind!r}{args})"

    def compile_decoder_for_dem(self, *, dem: Any) -> _CompiledDecoder:
        return _CompiledDecoder(_build(self.kind, DetectorErrorModel(dem), self.options))


class _CompiledSampler(_sinter.CompiledSampler):
    def __init__(self, task: Any, kind: str, options: dict) -> None:
        circuit = Circuit(task.circuit)
        dem = circuit.detector_error_model(decompose_errors=kind != "bposd", approximate_disjoint_errors=True)
        self.decoder = _build(kind, dem, options)
        # A fresh seed per compiled sampler: sinter's workers must not draw the same shots.
        self.sampler = circuit.compile_detector_sampler(seed=None)
        self.num_detectors, self.num_observables = circuit.num_detectors, circuit.num_observables
        self.postselection = self._mask(task.postselection_mask, self.num_detectors)
        self.postselected_observables = self._mask(task.postselected_observables_mask, self.num_observables)

    @staticmethod
    def _mask(packed: Any, n: int) -> Any:
        if packed is None:
            return None
        return np.unpackbits(np.asarray(packed, dtype=np.uint8), bitorder="little", count=n).astype(bool)

    def sample(self, suggested_shots: int) -> Any:
        t0 = time.monotonic()
        shots = max(1, int(suggested_shots))
        dets, obs = self.sampler.sample(shots, separate_observables=True)
        keep = np.ones(shots, dtype=bool)
        if self.postselection is not None:
            keep &= ~np.any(dets & self.postselection, axis=1)
        dets, obs = dets[keep], obs[keep]
        wrong = self.decoder.decode_batch(dets).astype(bool) != obs
        if self.postselected_observables is not None:
            discarded = np.any(wrong & self.postselected_observables, axis=1)
            wrong = wrong[~discarded]
            keep_count = int(np.count_nonzero(~discarded))
        else:
            keep_count = len(wrong)
        errors = int(np.count_nonzero(np.any(wrong, axis=1)))
        return _sinter.AnonTaskStats(shots=shots, errors=errors, discards=shots - keep_count, seconds=time.monotonic() - t0)


class Sampler(_sinter.Sampler):
    """A sinter sampler running the whole pipeline in this package: the task's circuit read,
    sampled, its error model built, and decoded. Postselection masks are honoured as sinter
    honours them. ``options`` go to the decoder's class."""

    def __init__(self, kind: str = "matching", **options: Any) -> None:
        self.kind = _check_kind(kind)
        self.options = options

    def __repr__(self) -> str:
        args = "".join(f", {k}={v!r}" for k, v in self.options.items())
        return f"stabilizer_qec.sinter.Sampler({self.kind!r}{args})"

    def compiled_sampler_for_task(self, task: Any) -> _CompiledSampler:
        return _CompiledSampler(task, self.kind, self.options)


def decoders() -> dict:
    """``{"sq_matching": Decoder("matching"), "sq_correlated_matching": ..., ...}``: every
    kind, decoding the shots Stim samples."""
    return {f"sq_{kind}": Decoder(kind) for kind in KINDS}


def samplers() -> dict:
    """``{"sq_sim_matching": Sampler("matching"), ...}``: every kind, with the sampling done
    here too."""
    return {f"sq_sim_{kind}": Sampler(kind) for kind in KINDS}


def sinter_decoders() -> dict:
    """Both, for ``sinter collect --custom_decoders_module_function
    "stabilizer_qec.sinter:sinter_decoders"``."""
    return {**decoders(), **samplers()}

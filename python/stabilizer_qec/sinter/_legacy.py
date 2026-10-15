"""This package's decoders as sinter decoders and samplers (1.x's ``stabilizer_qec.sinter``).

``Decoder(kind, **options)`` and ``Sampler(kind, **options)`` construct the classes here; they
work with this sinter and with sinter itself (``custom_decoders=sinter_decoders()``)."""

from __future__ import annotations

import sys
import time
from typing import Any

import numpy as np

from stabilizer_qec._circuit import Circuit, DetectorErrorModel
from stabilizer_qec._decoders import AmbiguityClustering, BeliefMatching, CollisionClustering, LocalClustering, BpLsd, BpOsd, ColorMatching, Matching, RelayBp, SearchDecoder, UnionFind
from stabilizer_qec.sinter._data import AnonTaskStats
from stabilizer_qec.sinter._decoding._decoding_decoder_class import CompiledDecoder, Decoder
from stabilizer_qec.sinter._decoding._sampler import CompiledSampler, Sampler

KINDS = ("matching", "correlated_matching", "belief_matching", "bposd", "union_find", "bplsd", "relay_bp", "color_matching", "search", "ambiguity_clustering", "collision_clustering", "local_clustering")

# The kinds that match on a graph, and so want the model decomposed.
_GRAPH_KINDS = ("matching", "correlated_matching", "belief_matching", "union_find", "collision_clustering", "local_clustering")

_CLASSES = {"belief_matching": BeliefMatching, "union_find": UnionFind, "bposd": BpOsd, "bplsd": BpLsd, "relay_bp": RelayBp, "color_matching": ColorMatching, "search": SearchDecoder, "ambiguity_clustering": AmbiguityClustering, "collision_clustering": CollisionClustering, "local_clustering": LocalClustering}


def _check_kind(kind: str) -> str:
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {', '.join(map(repr, KINDS))}, not {kind!r}")
    return kind


def _build(kind: str, dem: DetectorErrorModel, options: dict) -> Any:
    if kind == "matching":
        return Matching(dem, **options)
    if kind == "correlated_matching":
        from stabilizer_qec._matching import _CorrelatedMatching

        return _CorrelatedMatching(dem, **options)
    return _CLASSES[kind](dem, **options)


class _CompiledDecoder(CompiledDecoder):
    def __init__(self, decoder: Any) -> None:
        self.decoder = decoder

    def decode_shots_bit_packed(self, *, bit_packed_detection_event_data: np.ndarray) -> np.ndarray:
        return self.decoder.decode_batch(bit_packed_detection_event_data, bit_packed_shots=True, bit_packed_predictions=True)


class KindDecoder(Decoder):
    """One of this package's decoders, on the task's detector error model (``Decoder(kind,
    **options)``; the options go to the decoder's class)."""

    def __init__(self, kind: str = "matching", **options: Any) -> None:
        self.kind = _check_kind(kind)
        self.options = options

    def __repr__(self) -> str:
        args = "".join(f", {k}={v!r}" for k, v in self.options.items())
        return f"stabilizer_qec.sinter.Decoder({self.kind!r}{args})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, KindDecoder) and (self.kind, self.options) == (other.kind, other.options)

    def __hash__(self) -> int:
        return hash((self.kind, tuple(sorted(self.options))))

    def compile_decoder_for_dem(self, *, dem: Any) -> _CompiledDecoder:
        return _CompiledDecoder(_build(self.kind, DetectorErrorModel(dem), self.options))


class _CompiledSampler(CompiledSampler):
    def __init__(self, task: Any, kind: str, options: dict) -> None:
        circuit = Circuit(task.circuit)
        dem = circuit.detector_error_model(decompose_errors=kind in _GRAPH_KINDS, approximate_disjoint_errors=True)
        self.decoder = _build(kind, dem, options)
        # A fresh seed per compiled sampler: workers must not draw the same shots.
        self.sampler = circuit.compile_detector_sampler(seed=None)
        self.num_detectors, self.num_observables = circuit.num_detectors, circuit.num_observables
        self.postselection = self._mask(task.postselection_mask, self.num_detectors)
        self.postselected_observables = self._mask(task.postselected_observables_mask, self.num_observables)
        # Stats of the collecting sinter's own type: sinter itself only adds its own.
        self.stats_type = AnonTaskStats
        if type(task).__module__.split(".")[0] == "sinter":
            import sinter

            self.stats_type = sinter.AnonTaskStats

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
        return self.stats_type(shots=shots, errors=errors, discards=shots - keep_count, seconds=time.monotonic() - t0)


class KindSampler(Sampler):
    """The whole pipeline in this package: the task's circuit read, sampled, its error model
    built, and decoded by one of this package's decoders (``Sampler(kind, **options)``).
    Postselection masks are honoured as sinter honours them."""

    def __init__(self, kind: str = "matching", **options: Any) -> None:
        self.kind = _check_kind(kind)
        self.options = options

    def __repr__(self) -> str:
        args = "".join(f", {k}={v!r}" for k, v in self.options.items())
        return f"stabilizer_qec.sinter.Sampler({self.kind!r}{args})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, KindSampler) and (self.kind, self.options) == (other.kind, other.options)

    def __hash__(self) -> int:
        return hash((self.kind, tuple(sorted(self.options))))

    def compiled_sampler_for_task(self, task: Any) -> _CompiledSampler:
        return _CompiledSampler(task, self.kind, self.options)


def _known_to_sinter() -> None:
    # sinter itself tells samplers from decoders by its own Sampler class.
    try:
        import sinter
    except ImportError:
        return
    sinter.Sampler.register(KindSampler)


# In sinter's worker processes, sinter is loaded before the samplers sent to them are.
if "sinter" in sys.modules:
    _known_to_sinter()


def decoders() -> dict:
    """``{"sq_matching": Decoder("matching"), "sq_correlated_matching": ..., ...}``: every
    kind, decoding the shots the collector samples."""
    return {f"sq_{kind}": KindDecoder(kind) for kind in KINDS}


def samplers() -> dict:
    """``{"sq_sim_matching": Sampler("matching"), ...}``: every kind, with the sampling done
    here too."""
    _known_to_sinter()
    return {f"sq_sim_{kind}": KindSampler(kind) for kind in KINDS}


def sinter_decoders() -> dict:
    """Both, for ``sinter collect --custom_decoders_module_function
    "stabilizer_qec.sinter:sinter_decoders"``."""
    return {**decoders(), **samplers()}

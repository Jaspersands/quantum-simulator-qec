# Ported from Tesseract (quantumlib/tesseract-decoder, src/common.cc and its bindings; Copyright
# 2025 Google LLC, Apache License 2.0; see THIRD_PARTY_NOTICES.md), modified for stabilizer_qec:
# Python on this package's detector error models.
"""Tesseract's errors, symptoms and model transforms (``tesseract_decoder.common``)."""

from __future__ import annotations

import math
from typing import Any, Iterable, List, Sequence, Tuple

import stabilizer_qec as _sq

__all__ = ["Error", "Symptom", "dem_from_counts", "merge_indistinguishable_errors", "remove_zero_probability_errors"]

_MAX = 2**64 - 1


def _dem(dem: Any) -> "_sq.DetectorErrorModel":
    """A model as this package's (Stim's are read from their text, as Tesseract reads them)."""
    if isinstance(dem, _sq.DetectorErrorModel):
        return dem
    return _sq.DetectorErrorModel(str(dem))


def _flat(dem: Any) -> "_sq.DetectorErrorModel":
    d = _dem(dem)
    if any(inst.type in ("repeat", "shift_detectors") for inst in d):
        return d.flattened()
    return d


def _vec(v: Sequence[int]) -> str:
    return "[" + " ".join(str(x) for x in v) + "]"


class Symptom:
    """The detectors and observables an error flips."""

    __hash__ = None  # type: ignore[assignment]

    def __init__(self, detectors: Sequence[int] = [], observables: Sequence[int] = []) -> None:
        self.detectors = [int(d) for d in detectors]
        self.observables = [int(o) for o in observables]

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Symptom) and self.detectors == other.detectors and self.observables == other.observables

    def __ne__(self, other: object) -> bool:
        return not self == other

    def __str__(self) -> str:
        return f"Symptom{{detectors={_vec(self.detectors)}, observables={_vec(self.observables)}}}"

    def _key(self) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
        return tuple(self.detectors), tuple(self.observables)

    def as_dem_instruction_targets(self) -> List[Any]:
        return [_sq.target_relative_detector_id(d) for d in self.detectors] + [_sq.target_logical_observable_id(o) for o in self.observables]


class Error:
    """An error: its symptom and its cost, ``−ln(p/(1 − p))``."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if not args and not kwargs:
            self.likelihood_cost = 0.0
            self.symptom = Symptom()
            return
        if len(args) + len(kwargs) == 1:
            instruction = args[0] if args else kwargs["error"]
            self._from_instruction(instruction)
            return
        names = ("likelihood_cost", "detectors", "observables")
        values = dict(zip(names, args), **kwargs)
        self.likelihood_cost = float(values["likelihood_cost"])
        self.symptom = Symptom(values["detectors"], values["observables"])

    def _from_instruction(self, instruction: Any) -> None:
        if getattr(instruction, "type", None) != "error":
            raise ValueError(f"Error must be loaded from an error dem instruction, but received: {instruction}")
        probability = float(instruction.args_copy()[0])
        if probability < 0 or probability > 1:
            raise ValueError(f"Probability must be between 0 and 1, but received: {probability:f}")
        detectors: set = set()
        observables: set = set()
        for t in instruction.targets_copy():
            if t.is_logical_observable_id():
                observables ^= {t.val}
            elif t.is_relative_detector_id():
                detectors ^= {t.val}
        # −ln(p/(1 − p)) in IEEE arithmetic: ∞ at p = 0, −∞ at p = 1.
        self.likelihood_cost = -1 * _log(probability / (1 - probability)) if probability != 1 else -math.inf
        self.symptom = Symptom(sorted(detectors), sorted(observables))

    def __str__(self) -> str:
        return f"Error{{cost={self.likelihood_cost:.6f}, symptom={self.symptom}}}"

    def get_probability(self) -> float:
        return 1.0 / (1.0 + _exp(self.likelihood_cost))

    def set_with_probability(self, probability: float) -> None:
        p = float(probability)
        if p <= 0 or p >= 1:
            raise ValueError("Probability must be between 0 and 1.")
        self.likelihood_cost = -_log(p / (1.0 - p))


def _log(x: float) -> float:
    if x == 0:
        return -math.inf
    if x < 0 or math.isnan(x):
        return math.nan
    return math.log(x) if x != math.inf else math.inf


def _exp(x: float) -> float:
    try:
        return math.exp(x)
    except OverflowError:
        return math.inf


def merge_weights(a: float, b: float) -> float:
    """The cost of either of two independent errors with these costs (``common::merge_weights``)."""
    sgn = math.copysign(1, a) * math.copysign(1, b)
    signed_min = sgn * min(abs(a), abs(b))
    return signed_min + _log(1 + _exp(-abs(a + b))) - _log(1 + _exp(-abs(a - b)))


def _counts(dem: "_sq.DetectorErrorModel") -> Tuple[int, int]:
    return dem.num_detectors, dem.num_observables


def _preserve(out: "_sq.DetectorErrorModel", num_detectors: int, num_observables: int) -> None:
    if out.num_detectors < num_detectors:
        out.append("detector", [], [_sq.target_relative_detector_id(num_detectors - 1)])
    if out.num_observables < num_observables:
        out.append("logical_observable", [], [_sq.target_logical_observable_id(num_observables - 1)])


def _merge(dem: Any) -> Tuple["_sq.DetectorErrorModel", List[int]]:
    flat = _flat(dem)
    nd, no = _counts(flat)
    out = _sq.DetectorErrorModel()
    index_map: List[int] = []
    by_symptom: dict = {}
    merged: List[Error] = []
    for inst in flat:
        if inst.type == "error":
            e = Error(inst)
            if not e.symptom.detectors:
                print("Warning: the circuit has errors that do not flip any detectors")
            k = by_symptom.get(e.symptom._key())
            if k is not None:
                merged[k].likelihood_cost = merge_weights(e.likelihood_cost, merged[k].likelihood_cost)
                index_map.append(k)
            else:
                by_symptom[e.symptom._key()] = len(merged)
                index_map.append(len(merged))
                merged.append(e)
        elif inst.type in ("detector", "logical_observable"):
            out.append(inst)
        else:
            raise ValueError(f"Unrecognized instruction type: {inst}")
    for e in merged:
        out.append("error", [e.get_probability()], e.symptom.as_dem_instruction_targets())
    _preserve(out, nd, no)
    return out, index_map


def _remove_zero(dem: Any) -> Tuple["_sq.DetectorErrorModel", List[int]]:
    flat = _flat(dem)
    nd, no = _counts(flat)
    out = _sq.DetectorErrorModel()
    index_map: List[int] = []
    k = 0
    for inst in flat:
        if inst.type == "error":
            if inst.args_copy()[0] > 0:
                out.append(inst)
                index_map.append(k)
                k += 1
            else:
                index_map.append(_MAX)
        elif inst.type in ("detector", "logical_observable"):
            out.append(inst)
        else:
            raise ValueError(f"Unrecognized instruction type: {inst}")
    _preserve(out, nd, no)
    return out, index_map


def merge_indistinguishable_errors(dem: Any) -> "_sq.DetectorErrorModel":
    """The model with errors of the same symptom merged into one (in first-seen order, after the
    detector and observable declarations)."""
    return _merge(dem)[0]


def remove_zero_probability_errors(dem: Any) -> "_sq.DetectorErrorModel":
    """The model without its errors of probability 0."""
    return _remove_zero(dem)[0]


def dem_from_counts(orig_dem: Any, error_counts: Sequence[int], num_shots: int) -> "_sq.DetectorErrorModel":
    """The model with each error's probability estimated as its count over ``num_shots``."""
    flat = _flat(orig_dem)
    counts = [int(c) for c in error_counts]
    if flat.num_errors != len(counts):
        raise ValueError("Error hits array must be the same size as the number of errors in the original DEM.")
    out = _sq.DetectorErrorModel()
    k = 0
    for inst in flat:
        if inst.type == "error":
            out.append("error", [float(counts[k]) / float(num_shots)], inst.targets_copy(), tag=inst.tag)
            k += 1
        else:
            out.append(inst)
    return out


def _retained_maps(dem: Any, merge_errors: bool) -> Tuple["_sq.DetectorErrorModel", List[int], List[int]]:
    """Tesseract's preparation of a decoder's model: flattened, merged (if asked), errors of
    probability 0 removed, and the maps from the flattened model's errors to the retained ones
    and back (``2**64 − 1`` for none)."""
    flat = _flat(dem)
    dem_map = list(range(flat.num_errors))
    if merge_errors:
        flat, merge_map = _merge(flat)
        dem_map = [merge_map[i] if i != _MAX else _MAX for i in dem_map]
    flat, nonzero_map = _remove_zero(flat)
    dem_map = [nonzero_map[i] if i != _MAX else _MAX for i in dem_map]
    inverse = [_MAX] * flat.num_errors
    for i, m in enumerate(dem_map):
        if m != _MAX and inverse[m] == _MAX:
            inverse[m] = i
    return flat, dem_map, inverse


def _iter_errors(dem: Any) -> Iterable[Error]:
    for inst in _dem(dem):
        if inst.type == "error" and inst.args_copy()[0] > 0:
            yield Error(inst)

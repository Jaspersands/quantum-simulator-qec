# Ported from Tesseract (quantumlib/tesseract-decoder, src/multi_pass/: dem_decomposition.cc,
# error_correlations.cc, multi_pass_tesseract_decoder.cc and multi_pass_sinter_compat.pybind.h;
# Copyright 2025 Google LLC, Apache License 2.0; see THIRD_PARTY_NOTICES.md), modified for
# stabilizer_qec.
"""Tesseract's multi-pass decoder: a model split into two detector components (an X and a Z
half, say), each decoded by its own Tesseract decoder; with two passes, the errors the first
pass finds in one component raise the probabilities of the errors they are correlated with in
the other before the second pass decodes it (as correlated matching does)."""

from __future__ import annotations

import copy
import json
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

import stabilizer_qec as _sq

from .common import _dem
from .tesseract import TesseractConfig, TesseractDecoder

__all__ = ["MultiPassTesseractDecoder", "decompose_errors_using_detector_assignment", "prepare_two_component_dem"]

_MAX_REWEIGHT_PROBABILITY = 0.499

Symptom = Tuple[Tuple[int, ...], Tuple[int, ...]]


def _odd(items: Sequence[int]) -> List[int]:
    """``reduce_symmetric_difference``: the items occurring an odd number of times, sorted."""
    unpaired: set = set()
    for x in items:
        unpaired ^= {x}
    return sorted(unpaired)


def _split(inst: Any) -> List[List[Any]]:
    groups: List[List[Any]] = [[]]
    for t in inst.targets_copy():
        if t.is_separator():
            groups.append([])
        else:
            groups[-1].append(t)
    return groups


def _normalize(dem: Any, components: Sequence[int]) -> Tuple[List[int], List[int]]:
    if len(components) != dem.num_detectors:
        raise ValueError("Detector component assignment count does not match the DEM detector count.")
    for d, c in enumerate(components):
        if c < 0:
            raise ValueError(f"Detector D{d} has an invalid negative component assignment.")
    labels = sorted(set(components))
    if len(labels) != 2:
        raise ValueError(f"Multi-pass decoding requires exactly 2 detector components; got {len(labels)}.")
    return labels, [0 if c == labels[0] else 1 for c in components]


def _component(d: int, components: Sequence[int]) -> int:
    if d < 0 or d >= len(components):
        raise ValueError(f"Detector D{d} is out of range.")
    return components[d]


def _groups(inst: Any, components: Sequence[int], allow_mixed: bool) -> List[Tuple[List[Any], List[int], List[int], int]]:
    """Each decomposition group: (targets, detectors, observables, component or −1)."""
    out = []
    for targets in _split(inst):
        dets = _odd([t.val for t in targets if t.is_relative_detector_id()])
        obs = _odd([t.val for t in targets if t.is_logical_observable_id()])
        if not dets:
            raise ValueError(f"Error instruction `{inst}` contains a detectorless decomposition group, which cannot be assigned to a component.")
        comps = {_component(d, components) for d in dets}
        if len(comps) != 1 and not allow_mixed:
            raise ValueError(f"Error instruction `{inst}` contains a decomposition group with detectors from multiple components.")
        out.append((targets, dets, obs, next(iter(comps)) if len(comps) == 1 else -1))
    return out


def _separated(inst: Any) -> bool:
    return any(t.is_separator() for t in inst.targets_copy())


def _matching_observables(options: List[Any], error_obs: List[int]) -> Tuple[List[List[int]], int]:
    """``find_component_obs_matching_undecomposed_obs``: the components' observables whose
    combination is the error's, and how many such choices exist (counting stops at 2)."""
    if options[0] is None and options[1] is None:
        return [[], []], 2
    target = _odd(error_obs)
    found: List[List[int]] = [[], []]
    count = 0
    if options[0] is None or options[1] is None:
        known = 1 if options[0] is None else 0
        for known_obs in sorted(options[known]):
            candidate: List[List[int]] = [[], []]
            candidate[known] = list(known_obs)
            candidate[1 - known] = _odd(target + list(known_obs))
            count += 1
            if count == 1:
                found = candidate
            if count > 1:
                break
        return found, count
    for a in sorted(options[0]):
        for b in sorted(options[1]):
            if _odd(list(a) + list(b)) != target:
                continue
            count += 1
            if count == 1:
                found = [list(a), list(b)]
            if count > 1:
                return found, count
    return found, count


def _decompose(flat: Any, components: Sequence[int]) -> Any:
    known: Dict[Tuple[int, ...], set] = {}
    for inst in flat:
        if inst.type != "error":
            continue
        decomposed = _separated(inst)
        groups = _groups(inst, components, not decomposed)
        if decomposed:
            for _, dets, obs, _c in groups:
                known.setdefault(tuple(dets), set()).add(tuple(obs))
        elif groups[0][3] >= 0:
            known.setdefault(tuple(groups[0][1]), set()).add(tuple(groups[0][2]))
    out = _sq.DetectorErrorModel()
    for inst in flat:
        if inst.type != "error":
            out.append(inst)
            continue
        decomposed = _separated(inst)
        groups = _groups(inst, components, not decomposed)
        if decomposed:
            out.append(inst)
            continue
        _, dets, obs, _c = groups[0]
        by_comp: List[List[int]] = [[], []]
        for d in dets:
            by_comp[_component(d, components)].append(d)
        if not by_comp[0] or not by_comp[1]:
            out.append(inst)
            continue
        options = [known.get(tuple(sorted(by_comp[c]))) for c in range(2)]
        assignment, count = _matching_observables(options, obs)
        if count == 0:
            raise ValueError(f"Error instruction `{inst}` has no consistent observable decomposition.")
        if count > 1:
            raise ValueError(f"Error instruction `{inst}` has multiple consistent observable decompositions; logical observable ownership is ambiguous.")
        targets: List[Any] = []
        for c in range(2):
            targets += [_sq.target_relative_detector_id(d) for d in sorted(by_comp[c])]
            targets += [_sq.target_logical_observable_id(o) for o in assignment[c]]
            if c == 0:
                targets.append(_sq.target_separator())
        out.append("error", inst.args_copy()[:1], targets, tag=inst.tag)
    return out


def _split_by_component(dem: Any, components: Sequence[int]) -> List[Any]:
    outs = [_sq.DetectorErrorModel(), _sq.DetectorErrorModel()]
    for inst in dem:
        if inst.type != "error":
            for o in outs:
                o.append(inst)
            continue
        by_comp: List[List[List[Any]]] = [[], []]
        for targets, _d, _o, c in _groups(inst, components, False):
            by_comp[c].append(targets)
        for c in range(2):
            if not by_comp[c]:
                continue
            targets: List[Any] = []
            dets: List[int] = []
            for k, group in enumerate(by_comp[c]):
                targets += group
                dets += [t.val for t in group if t.is_relative_detector_id()]
                if k + 1 < len(by_comp[c]):
                    targets.append(_sq.target_separator())
            if not _odd(dets):
                raise ValueError(f"Error instruction `{inst}` has a detectorless component symptom after combining its decomposition groups.")
            outs[c].append("error", inst.args_copy()[:1], targets, tag=inst.tag)
    return outs


def prepare_two_component_dem(dem: Any, detector_components: Sequence[int]) -> Tuple[List[int], List[int], Any, List[Any]]:
    """The model flattened, its errors decomposed into one part per component, and split:
    (the two labels in ascending order, each detector's component 0 or 1, the decomposed
    model, the two components' models)."""
    flat = _dem(dem).flattened()
    labels, components = _normalize(flat, [int(c) for c in detector_components])
    decomposed = _decompose(flat, components)
    return labels, components, decomposed, _split_by_component(decomposed, components)


def decompose_errors_using_detector_assignment(dem: Any, detector_components: Sequence[int]) -> Any:
    """The model flattened with each error spanning both components decomposed (``^``) into a
    part per component, the observables it flips assigned to the parts as other errors of the
    model show them."""
    flat = _dem(dem).flattened()
    _, components = _normalize(flat, [int(c) for c in detector_components])
    return _decompose(flat, components)


def _xor(acc: float, p: float) -> float:
    return acc * (1 - p) + p * (1 - acc)


def _reweight_probabilities(decomposed: Any, components: Sequence[int]) -> Dict[Symptom, List[Tuple[Symptom, float]]]:
    """``process_dem_correlations``: for each component symptom, the probabilities to give
    the symptoms of the other component it occurs with."""
    marginal: Dict[Symptom, float] = {}
    paired: Dict[Symptom, Dict[Symptom, float]] = {}
    for inst in decomposed:
        if inst.type != "error":
            continue
        sets: List[Tuple[set, set]] = [(set(), set()), (set(), set())]
        has = [False, False]
        for targets in _split(inst):
            gd: set = set()
            go: set = set()
            for t in targets:
                if t.is_relative_detector_id():
                    gd ^= {t.val}
                elif t.is_logical_observable_id():
                    go ^= {t.val}
            if not gd:
                raise ValueError(f"Error instruction `{inst}` contains a detectorless decomposition group.")
            comp = -1
            for d in sorted(gd):
                if d < 0 or d >= len(components):
                    raise ValueError(f"Invalid component assignment for detector D{d}.")
                if comp == -1:
                    comp = components[d]
                elif comp != components[d]:
                    raise ValueError(f"Error instruction `{inst}` contains a decomposition group with detectors from multiple components.")
            has[comp] = True
            sets[comp][0].symmetric_difference_update(gd)
            sets[comp][1].symmetric_difference_update(go)
        symptoms: List[Symptom] = [((), ()), ((), ())]
        for c in range(2):
            if has[c]:
                if not sets[c][0]:
                    raise ValueError(f"Error instruction `{inst}` has a detectorless component symptom after combining its decomposition groups.")
                symptoms[c] = (tuple(sorted(sets[c][0])), tuple(sorted(sets[c][1])))
        p = inst.args_copy()[0]
        for c in range(2):
            if has[c]:
                marginal[symptoms[c]] = _xor(marginal.get(symptoms[c], 0.0), p)
        if has[0] and has[1]:
            for a, b in ((0, 1), (1, 0)):
                row = paired.setdefault(symptoms[a], {})
                row[symptoms[b]] = _xor(row.get(symptoms[b], 0.0), p)
    out: Dict[Symptom, List[Tuple[Symptom, float]]] = {}
    for causal in sorted(paired):
        m = marginal.get(causal)
        if m is None or m <= 0 or m >= 1:
            continue
        for affected in sorted(paired[causal]):
            out.setdefault(causal, []).append((affected, min(max(paired[causal][affected] / m, 0.0), 1.0)))
    return out


def _canonical_components(dem: Any) -> List[int]:
    """Each detector's component from its tag's JSON ``measure_basis`` (X → 0, Z → 1)."""
    flat = _dem(dem).flattened()
    n = flat.num_detectors
    tags: List[Any] = [None] * n
    for inst in flat:
        if inst.type != "detector":
            continue
        targets = inst.targets_copy()
        if len(targets) != 1 or not targets[0].is_relative_detector_id():
            raise ValueError(f"Malformed detector instruction: {inst}")
        d = targets[0].val
        if d >= n:
            raise ValueError(f"Detector D{d} is out of range.")
        if tags[d] is not None:
            raise ValueError(f"Detector D{d} has more than one detector instruction.")
        tags[d] = inst.tag
    out = []
    for d, tag in enumerate(tags):
        name = f"Detector D{d}"
        if tag is None:
            raise ValueError(f"{name} has no detector instruction with a canonical measure_basis tag.")
        if not tag:
            raise ValueError(f'{name} has no tag; multi-pass CLI input requires a top-level JSON measure_basis field equal to "X" or "Z".')
        try:
            meta = json.loads(tag)
        except ValueError:
            raise ValueError(f'{name} has a non-JSON tag; multi-pass CLI input requires a top-level JSON measure_basis field equal to "X" or "Z".') from None
        if not isinstance(meta, dict) or "measure_basis" not in meta:
            raise ValueError(f'{name} tag has no top-level measure_basis field equal to "X" or "Z".')
        basis = meta["measure_basis"]
        if basis == "X":
            out.append(0)
        elif basis == "Z":
            out.append(1)
        else:
            raise ValueError(f'{name} has an invalid top-level measure_basis; expected the string "X" or "Z".')
    return out


class MultiPassTesseractDecoder:
    """Tesseract's multi-pass decoder over two detector components, each decoded by a
    Tesseract decoder configured as ``base_config``; ``strategy`` 0 (Static) decodes both
    components in every pass, 1 (Causal) only those the next pass, or the answer, needs."""

    def __init__(self, base_config: TesseractConfig, num_passes: int = 2, detector_components: Sequence[int] = (), strategy: int = 1) -> None:
        if isinstance(num_passes, bool) or int(num_passes) < 0:
            raise TypeError("num_passes must be a nonnegative integer.")
        self.num_passes = int(num_passes)
        if not 1 <= self.num_passes <= 2:
            raise ValueError("num_passes must be 1 or 2.")
        if int(strategy) not in (0, 1):
            raise ValueError("Invalid multi-pass scheduling strategy.")
        self.strategy = int(strategy)
        if self.num_passes == 2 and not base_config.merge_errors:
            raise ValueError(
                "Two-pass decoding requires merge_errors=true because reweighting is defined on aggregate component symptoms; "
                "use one pass to decode unmerged error mechanisms."
            )
        components = [int(c) for c in detector_components]
        if not components:
            components = _canonical_components(base_config._dem)
        labels, self._component_of, decomposed, dems = prepare_two_component_dem(base_config._dem, components)
        self.monolithic_dem = decomposed
        self.assignment_labels = labels
        self.active_detector_counts = [self._component_of.count(c) for c in range(2)]
        reweights = _reweight_probabilities(decomposed, self._component_of) if self.num_passes == 2 else {}
        self.decoders: List[TesseractDecoder] = []
        self.affects_observable = [False, False]
        by_symptom: List[Dict[Symptom, Tuple[int, float]]] = [{}, {}]
        for c in range(2):
            config = copy.copy(base_config)
            config.dem = dems[c]
            decoder = TesseractDecoder(config)
            self.decoders.append(decoder)
            for ei, e in enumerate(decoder.errors):
                self.affects_observable[c] |= bool(e.symptom.observables)
                if self.num_passes == 2:
                    key = (tuple(e.symptom.detectors), tuple(e.symptom.observables))
                    if key in by_symptom[c]:
                        raise RuntimeError(f"Component {c} retained duplicate {e.symptom} despite merge_errors=true.")
                    by_symptom[c][key] = (decoder._error_to_dem[ei], e.get_probability())
        # rules[c][model error] = [(target component, target model error, reweight p, original p)]
        self.reweight_rules: List[Dict[int, List[Tuple[int, int, float, float]]]] = [{}, {}]
        for causal, entries in reweights.items():
            cc = self._component_of[causal[0][0]]
            found = by_symptom[cc].get(causal)
            if found is None:
                continue
            for affected, p in entries:
                tc = self._component_of[affected[0][0]]
                target = by_symptom[tc].get(affected)
                if target is None:
                    continue
                self.reweight_rules[cc].setdefault(found[0], []).append((tc, target[0], p, target[1]))
        if self.strategy == 0:
            self.pass_schedule = [[0, 1] for _ in range(self.num_passes)]
        else:
            sets: List[set] = [set() for _ in range(self.num_passes)]
            sets[-1] = {c for c in range(2) if self.affects_observable[c]}
            for p in range(self.num_passes - 2, -1, -1):
                for target in sets[p + 1]:
                    for source in range(2):
                        if any(r[0] == target for rules in self.reweight_rules[source].values() for r in rules):
                            sets[p].add(source)
            self.pass_schedule = [sorted(s) for s in sets]

    def decode_result(self, detections: Sequence[int]) -> Tuple[List[int], bool, float]:
        """(The observables flipped, whether any search gave up, the final pass's cost)."""
        n = len(self._component_of)
        for d in detections:
            if d >= n:
                raise ValueError(f"Detector D{d} is out of range for a model with {n} detectors.")
        results: List[Any] = [None, None]
        updates: List[Dict[int, float]] = [{}, {}]
        low = False
        for p in range(self.num_passes):
            for c in self.pass_schedule[p]:
                dets = [d for d in detections if self._component_of[d] == c]
                results[c] = self.decoders[c]._decode_result(dets, sorted(updates[c].items()))
                low |= results[c][2]
            if p + 1 == self.num_passes:
                continue
            for source in self.pass_schedule[p]:
                for model_error in results[source][1]:
                    for tc, te, rp, original in self.reweight_rules[source].get(model_error, ()):
                        rp = min(rp, _MAX_REWEIGHT_PROBABILITY)
                        if rp > original:
                            updates[tc][te] = max(updates[tc].get(te, 0.0), rp)
        flipped: set = set()
        cost = 0.0
        for c in self.pass_schedule[-1]:
            flipped ^= set(results[c][0])
            cost += results[c][3]
        return sorted(flipped), low, cost


class _MultiPassSinterCompiledDecoder:
    def __init__(self, decoder: MultiPassTesseractDecoder, num_detectors: int, num_observables: int) -> None:
        self.decoder = decoder
        self.num_detectors = num_detectors
        self.num_observables = num_observables

    def decode_shots_bit_packed(self, *, bit_packed_detection_event_data: Any) -> np.ndarray:
        """Each shot's predicted observables, bit packed, then a byte that is 1 where a search
        gave up (sinter's predictions-and-discards layout)."""
        from .tesseract_sinter_compat import _rows

        rows = _rows(bit_packed_detection_event_data, self.num_detectors)
        nob = (self.num_observables + 7) // 8
        out = np.zeros((rows.shape[0], nob + 1), dtype=np.uint8)
        for shot in range(rows.shape[0]):
            lit = np.flatnonzero(np.unpackbits(rows[shot], bitorder="little")[: self.num_detectors]).tolist()
            predictions, low, _ = self.decoder.decode_result(lit)
            for o in predictions:
                out[shot, o // 8] ^= 1 << (o % 8)
            out[shot, nob] = int(low)
        return out


def _compile_multi_pass_decoder_for_dem(*, dem: Any, detector_components: Sequence[int], num_passes: int, base_config: TesseractConfig, strategy: Any) -> _MultiPassSinterCompiledDecoder:
    model = _sq.DetectorErrorModel(str(dem))
    config = copy.copy(base_config)
    config.dem = model
    return _MultiPassSinterCompiledDecoder(MultiPassTesseractDecoder(config, num_passes, detector_components, int(strategy)), model.num_detectors, model.num_observables)

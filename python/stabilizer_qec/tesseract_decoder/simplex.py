# Ported from Tesseract (quantumlib/tesseract-decoder, src/simplex.cc and its bindings; Copyright
# 2025 Google LLC, Apache License 2.0; see THIRD_PARTY_NOTICES.md), modified for stabilizer_qec:
# each integer program is solved by an exact search instead of HiGHS.
"""Tesseract's simplex decoder (``tesseract_decoder.simplex``): the most likely set of errors as
an integer program, whole or in sliding time windows.

Tesseract solves the program with HiGHS; here each one is solved exactly by an A* search with
no beam and no pruning. Both find an optimum, so the cost of the errors found is the same;
where several sets of errors are equally likely the two may pick different ones (and, with
windows, a different pick in one window can change what the next windows see)."""

from __future__ import annotations

from typing import Any, Dict, List

from .. import _core
from .._util import call
from .common import _dem
from .tesseract import _Decoder
from .utils import get_detector_coords

__all__ = ["SimplexConfig", "SimplexDecoder"]


def _bits(mask: int) -> List[int]:
    out = []
    while mask:
        low = mask & -mask
        out.append(low.bit_length() - 1)
        mask ^= low
    return out

_T_COORD = 2


class SimplexConfig:
    """How a simplex decoder works: the model, whether HiGHS may use threads (kept for
    Tesseract's API), the time window (``window_length`` time steps, sliding by
    ``window_slide_length``; 0 for no windows), verbose output and merging errors of one
    symptom."""

    def __init__(self, dem: Any, parallelize: bool = False, window_length: int = 0, window_slide_length: int = 0, verbose: bool = False, merge_errors: bool = True) -> None:
        for name, value in (("window_length", window_length), ("window_slide_length", window_slide_length)):
            if isinstance(value, bool) or int(value) < 0:
                raise TypeError(f"{name} must be a nonnegative integer.")
        self.dem = _dem(dem)
        self.parallelize = bool(parallelize)
        self.window_length = int(window_length)
        self.window_slide_length = int(window_slide_length)
        self.verbose = bool(verbose)
        self.merge_errors = bool(merge_errors)

    def windowing_enabled(self) -> bool:
        """Whether the decoder works in time windows (``window_length > 0``)."""
        return self.window_length != 0

    def __str__(self) -> str:
        return (
            f"SimplexConfig(dem=DetectorErrorModel_Object, window_length={self.window_length}, "
            f"window_slide_length={self.window_slide_length}, verbose={int(self.verbose)}, merge_errors={int(self.merge_errors)})"
        )

    def compile_decoder(self) -> "SimplexDecoder":
        return SimplexDecoder(self)


class SimplexDecoder(_Decoder):
    """The most likely set of errors for a syndrome, found exactly (Tesseract's
    ``SimplexDecoder``). ``start_time_to_errors[t]`` and ``end_time_to_errors[t]`` list the
    errors whose earliest and latest detectors are at the ``t``-th time (the third detector
    coordinate) of those the errors touch."""

    low_confidence_flag = False

    def __init__(self, config: SimplexConfig) -> None:
        import copy

        flat = self._prepare(config.dem, config.merge_errors)
        self.config = copy.copy(config)
        self.config.dem = flat
        self.error_masks: List[List[int]] = []
        self._masks = [sum(1 << d for d in set(e.symptom.detectors)) for e in self.errors]
        t_coords = [0.0] * self.num_detectors
        for d, c in enumerate(get_detector_coords(flat)[: self.num_detectors]):
            if len(c) > _T_COORD:
                t_coords[d] = c[_T_COORD]
        starts: Dict[float, List[int]] = {}
        ends: Dict[float, List[int]] = {}
        times = set()
        for ei, e in enumerate(self.errors):
            ts = [t_coords[d] for d in e.symptom.detectors]
            times.update(ts)
            starts.setdefault(min(ts, default=1.7976931348623157e308), []).append(ei)
            ends.setdefault(max(ts, default=-1.7976931348623157e308), []).append(ei)
        ordered = sorted(times)
        self.start_time_to_errors = [list(starts.get(t, [])) for t in ordered]
        self.end_time_to_errors = [list(ends.get(t, [])) for t in ordered]

    def init_ilp(self) -> None:
        """Tesseract rebuilds its HiGHS model here; the search needs no model, so this does
        nothing."""

    def _solve(self, detections: List[int], costs: List[float], fixed: Dict[int, int]) -> List[int]:
        """An optimum of the program with these costs and some errors fixed: the value of each
        error variable.

        Errors that cost nothing (those outside the window) can explain any syndrome in their
        span, so the program is solved modulo that span: their detector sets are reduced to an
        echelon basis, the other errors and the syndrome are reduced by it, the cheapest errors
        for the reduced syndrome are found by the exact search, and the free errors then make up
        the rest."""
        target = 0
        for d in detections:
            target ^= 1 << d
        for ei, v in fixed.items():
            if v:
                target ^= self._masks[ei]
        free = [ei for ei in range(len(self.errors)) if ei not in fixed and costs[ei] == 0]
        paid = [ei for ei in range(len(self.errors)) if ei not in fixed and costs[ei] != 0]
        basis: List[tuple] = []  # (pivot bit, detector mask, the free errors composing it)

        def reduce(mask: int) -> tuple:
            combo = 0
            for pivot, vec, parts in basis:
                if mask >> pivot & 1:
                    mask ^= vec
                    combo ^= parts
            return mask, combo

        for k, ei in enumerate(free):
            mask, combo = reduce(self._masks[ei])
            if mask:
                basis.append((mask.bit_length() - 1, mask, combo ^ (1 << k)))
        reduced = [reduce(self._masks[ei])[0] for ei in paid]
        goal = reduce(target)[0]
        found, infeasible = call(
            _core.search_exact, self.num_detectors, [costs[ei] for ei in paid], [_bits(m) for m in reduced], _bits(goal)
        )
        if infeasible:
            raise RuntimeError("HighsModelStatus::kOptimal expected, got Infeasible.")
        values = [0] * len(self.errors)
        for ei, v in fixed.items():
            values[ei] = v
        rest = target
        for k in found:
            values[paid[k]] ^= 1
            rest ^= self._masks[paid[k]]
        left, combo = reduce(rest)
        if left:
            raise RuntimeError("HighsModelStatus::kOptimal expected, got Infeasible.")
        for k, ei in enumerate(free):
            if combo >> k & 1:
                values[ei] = 1
        return values

    def _decode(self, detections: List[int], *args: Any) -> None:
        for d in detections:
            if not 0 <= d < self.num_detectors:
                raise IndexError("invalid detector")
        detections = sorted(set(detections))
        true_costs = [e.likelihood_cost for e in self.errors]
        if not self.config.windowing_enabled():
            values = self._solve(detections, true_costs, {})
        else:
            length, slide = self.config.window_length, self.config.window_slide_length
            slices = len(self.start_time_to_errors)
            costs = [0.0] * len(self.errors)
            t1 = 0
            while t1 + slide < length and t1 < slices:
                for ei in self.start_time_to_errors[t1]:
                    costs[ei] = true_costs[ei]
                t1 += 1
            if slide == 0 and t1 < slices:
                raise ValueError("window_slide_length must be positive when the windows do not cover every time step (Tesseract never finishes).")
            t0 = 0
            fixed: Dict[int, int] = {}
            values = None
            while t1 < slices or values is None:
                step = 0
                while step < slide and t1 < slices:
                    for ei in self.start_time_to_errors[t1]:
                        costs[ei] = true_costs[ei]
                    t1 += 1
                    step += 1
                if self.config.verbose:
                    print(f"t0 = {t0} t1 = {t1}")
                values = self._solve(detections, costs, fixed)
                step = 0
                while step < slide and t0 < len(self.end_time_to_errors):
                    for ei in self.end_time_to_errors[t0]:
                        fixed[ei] = values[ei]
                    t0 += 1
                    step += 1
        self.predicted_errors_buffer = [self._error_to_dem[ei] for ei, v in enumerate(values) if v == 1]

"""Explained errors (``Circuit.explain_detector_error_model_errors``): where in a circuit each
fault of its error model arises, with Stim's class and field names and Stim's text."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass(frozen=True)
class DemTargetWithCoords:
    """A detector (``"D3"``) or observable (``"L0"``) with the detector's coordinates."""

    dem_target: str
    coords: List[float]
    _text: str = field(default="", repr=False, compare=False)

    def __str__(self) -> str:
        return self._text or self.dem_target


@dataclass(frozen=True)
class GateTargetWithCoords:
    """A circuit target as Stim writes it (``"X5"``, ``"!Z2"``, ``"7"``) with its qubit's
    coordinates (its last ``QUBIT_COORDS``)."""

    gate_target: str
    coords: List[float]
    _text: str = field(default="", repr=False, compare=False)

    def __str__(self) -> str:
        return self._text or self.gate_target


@dataclass(frozen=True)
class FlippedMeasurement:
    """The measurement a fault flips: its index in the measurement record, and the observable
    it measured."""

    record_index: int
    observable: List[GateTargetWithCoords]


@dataclass(frozen=True)
class CircuitErrorLocationStackFrame:
    """One block level of a location: the instruction's offset in its block, how many passes
    of the enclosing loop came before, and the loop's repetitions (0 at the innermost level)."""

    instruction_offset: int
    iteration_index: int
    instruction_repetitions_arg: int


@dataclass(frozen=True)
class CircuitTargetsInsideInstruction:
    """The instruction a fault came from and the targets of it the fault covers."""

    gate: str
    tag: str
    args: List[float]
    target_range_start: int
    target_range_end: int
    targets_in_range: List[GateTargetWithCoords]
    _text: str = field(default="", repr=False, compare=False)

    def __str__(self) -> str:
        return self._text


@dataclass(frozen=True)
class CircuitErrorLocation:
    """One place in the circuit a fault arises; ``str()`` gives Stim's text."""

    tick_offset: int
    flipped_pauli_product: List[GateTargetWithCoords]
    flipped_measurement: Optional[FlippedMeasurement]
    instruction_targets: CircuitTargetsInsideInstruction
    stack_frames: List[CircuitErrorLocationStackFrame]
    noise_tag: str
    _text: str = field(default="", repr=False, compare=False)

    def __str__(self) -> str:
        return self._text


@dataclass(frozen=True, repr=False)
class ExplainedError:
    """A fault class of the error model (``dem_error_terms``: what it sets off) and every place
    in the circuit such a fault arises; ``str()`` gives Stim's text."""

    dem_error_terms: List[DemTargetWithCoords]
    circuit_error_locations: List[CircuitErrorLocation]
    _text: str

    def __str__(self) -> str:
        return self._text

    def __repr__(self) -> str:
        terms = " ".join(t.dem_target for t in self.dem_error_terms)
        return f"<stabilizer_qec.ExplainedError {terms}: {len(self.circuit_error_locations)} locations>"


def _targets(raw: List[Tuple[str, List[float], str]]) -> List[GateTargetWithCoords]:
    return [GateTargetWithCoords(t, list(c), s) for t, c, s in raw]


def _frames(stack: list) -> List[CircuitErrorLocationStackFrame]:
    """The core's frames, (offset, completed iterations, repetitions) from the outermost
    block in, as Stim's: each frame's ``iteration_index`` is the pass of the loop enclosing
    it (so the outermost frame's is 0)."""
    out, outer = [], 0
    for offset, iteration, reps in stack:
        out.append(CircuitErrorLocationStackFrame(offset, outer, reps))
        outer = iteration
    return out


def build(raw: list) -> List[ExplainedError]:
    """The core's tuples as explained errors."""
    out = []
    for text, terms, locations in raw:
        locs = []
        for tick, pauli, meas, (gate, tag, args), (start, end), targets, stack, shown, instruction in locations:
            locs.append(
                CircuitErrorLocation(
                    tick_offset=tick,
                    flipped_pauli_product=_targets(pauli),
                    flipped_measurement=None if meas is None else FlippedMeasurement(meas[0], _targets(meas[1])),
                    instruction_targets=CircuitTargetsInsideInstruction(gate, tag, list(args), start, end, _targets(targets), instruction),
                    stack_frames=_frames(stack),
                    noise_tag=tag,
                    _text=shown,
                )
            )
        out.append(ExplainedError([DemTargetWithCoords(t, list(c), s) for t, c, s in terms], locs, text))
    return out

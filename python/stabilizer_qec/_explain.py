"""Explained errors (``Circuit.explain_detector_error_model_errors``): where in a circuit each
fault of its error model arises, as Stim's classes: the same constructors, fields, text and
reprs (ported from Stim's ``matched_error.cc``)."""

from __future__ import annotations

from typing import Any, Iterable, List, Optional

from . import _core, _dem, _stim


def _num(v: float) -> str:
    """A double as C++ streams it (6 significant digits)."""
    v = float(v)
    if v.is_integer() and abs(v) < 1e16:
        return str(int(v))
    return format(v, "g")


def _coords_text(coords: List[float]) -> str:
    return "[coords " + ",".join(_num(c) for c in coords) + "]" if coords else ""


def _coords_repr(coords: List[float]) -> str:
    return "[" + ", ".join(_num(c) for c in coords) + "]"


def _gate_target(t: Any) -> "_stim.GateTarget":
    if isinstance(t, _stim.GateTarget):
        return t
    if isinstance(t, str):
        return _stim.GateTarget._raw(_core.parse_gate_target(t))
    return _stim.GateTarget(t)


class GateTargetWithCoords:
    """A circuit target with its qubit's coordinates, as ``stim.GateTargetWithCoords``."""

    __slots__ = ("gate_target", "coords")

    def __init__(self, gate_target: Any, coords: Iterable[float]) -> None:
        self.gate_target = _gate_target(gate_target)
        self.coords = [float(c) for c in coords]

    def __str__(self) -> str:
        return self.gate_target.target_str() + _coords_text(self.coords)

    def __repr__(self) -> str:
        g = self.gate_target
        # Stim streams a plain qubit target bare, anything else as its repr.
        shown = str(g.value) if g.is_qubit_target and not g.is_inverted_result_target else repr(g)
        return f"stabilizer_qec.GateTargetWithCoords({shown}, {_coords_repr(self.coords)})"

    def _key(self) -> tuple:
        return (self.gate_target, tuple(self.coords))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, GateTargetWithCoords) and self._key() == other._key()

    def __ne__(self, other: object) -> bool:
        return not self == other

    def __hash__(self) -> int:
        return hash(("GateTargetWithCoords",) + self._key())


class DemTargetWithCoords:
    """A detector or observable with the detector's coordinates, as
    ``stim.DemTargetWithCoords``."""

    __slots__ = ("dem_target", "coords")

    def __init__(self, dem_target: Any, coords: Iterable[float]) -> None:
        self.dem_target = _dem.DemTarget(dem_target)
        self.coords = [float(c) for c in coords]

    def __str__(self) -> str:
        return str(self.dem_target) + _coords_text(self.coords)

    def __repr__(self) -> str:
        return f"stabilizer_qec.DemTargetWithCoords(dem_target={self.dem_target!r}, coords={_coords_repr(self.coords)})"

    def _key(self) -> tuple:
        return (self.dem_target, tuple(self.coords))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, DemTargetWithCoords) and self._key() == other._key()

    def __ne__(self, other: object) -> bool:
        return not self == other

    def __hash__(self) -> int:
        return hash(("DemTargetWithCoords",) + self._key())


class FlippedMeasurement:
    """The measurement a fault flips: its index in the measurement record (None for none), and
    the observable it measured, as ``stim.FlippedMeasurement``."""

    __slots__ = ("record_index", "observable")

    _NOTHING = object()

    def __init__(self, *, record_index: Any = _NOTHING, observable: Any = _NOTHING, measurement_record_index: Any = _NOTHING, measured_observable: Any = _NOTHING) -> None:
        # Stim's signature names the long forms; its constructor takes the short ones. Both work.
        ri = record_index if record_index is not FlippedMeasurement._NOTHING else measurement_record_index
        obs = observable if observable is not FlippedMeasurement._NOTHING else measured_observable
        if ri is FlippedMeasurement._NOTHING or obs is FlippedMeasurement._NOTHING:
            raise TypeError("FlippedMeasurement needs record_index and observable")
        self.record_index = None if ri is None else int(ri)
        self.observable = list(obs)

    def __repr__(self) -> str:
        ri = "None" if self.record_index is None else str(self.record_index)
        obs = "".join(repr(e) + "," for e in self.observable)
        return f"stabilizer_qec.FlippedMeasurement(\n    record_index={ri},\n    observable=({obs}),\n)"

    __str__ = __repr__

    def _key(self) -> tuple:
        return (self.record_index, tuple(self.observable))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, FlippedMeasurement) and self._key() == other._key()

    def __ne__(self, other: object) -> bool:
        return not self == other

    def __hash__(self) -> int:
        return hash(("FlippedMeasurement",) + self._key())


class CircuitErrorLocationStackFrame:
    """One block level of a location: the instruction's offset in its block, the pass of the
    loop enclosing it, and its repetitions if it is a loop (0 at the innermost level), as
    ``stim.CircuitErrorLocationStackFrame``."""

    __slots__ = ("instruction_offset", "iteration_index", "instruction_repetitions_arg")

    def __init__(self, *, instruction_offset: int, iteration_index: int, instruction_repetitions_arg: int) -> None:
        self.instruction_offset = int(instruction_offset)
        self.iteration_index = int(iteration_index)
        self.instruction_repetitions_arg = int(instruction_repetitions_arg)

    def __repr__(self) -> str:
        return (
            "stabilizer_qec.CircuitErrorLocationStackFrame(\n"
            f"    instruction_offset={self.instruction_offset},\n"
            f"    iteration_index={self.iteration_index},\n"
            f"    instruction_repetitions_arg={self.instruction_repetitions_arg},\n)"
        )

    __str__ = __repr__

    def _key(self) -> tuple:
        return (self.instruction_offset, self.iteration_index, self.instruction_repetitions_arg)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, CircuitErrorLocationStackFrame) and self._key() == other._key()

    def __ne__(self, other: object) -> bool:
        return not self == other

    def __hash__(self) -> int:
        return hash(("CircuitErrorLocationStackFrame",) + self._key())


class CircuitTargetsInsideInstruction:
    """The instruction a fault came from and the targets of it the fault covers, as
    ``stim.CircuitTargetsInsideInstruction``."""

    __slots__ = ("_gate", "tag", "args", "target_range_start", "target_range_end", "targets_in_range")

    def __init__(self, *, gate: str, tag: str = "", args: Iterable[float], target_range_start: int, target_range_end: int, targets_in_range: Iterable[GateTargetWithCoords]) -> None:
        self._gate = _stim.GateData(gate).name
        self.tag = str(tag)
        self.args = [float(a) for a in args]
        self.target_range_start = int(target_range_start)
        self.target_range_end = int(target_range_end)
        self.targets_in_range = list(targets_in_range)

    @property
    def gate(self) -> Optional[str]:
        return self._gate

    def __str__(self) -> str:
        s = self._gate
        if self.tag:
            s += "[" + _dem._escape_tag(self.tag) + "]"
        if self.args:
            s += "(" + ", ".join(_num(a) for a in self.args) + ")"
        was_combiner = False
        for t in self.targets_in_range:
            is_combiner = t.gate_target.is_combiner
            if not is_combiner and not was_combiner:
                s += " "
            was_combiner = is_combiner
            s += str(t)
        return s

    def __repr__(self) -> str:
        targets = "".join(repr(e) + "," for e in self.targets_in_range)
        args = ", ".join(_num(a) for a in self.args)
        return f"stabilizer_qec.CircuitTargetsInsideInstruction(gate='{self._gate}', args=[{args}], target_range_start={self.target_range_start}, target_range_end={self.target_range_end}, targets_in_range=({targets}))"

    def _key(self) -> tuple:
        return (self._gate, self.tag, self.target_range_start, self.target_range_end, tuple(self.targets_in_range), tuple(self.args))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, CircuitTargetsInsideInstruction) and self._key() == other._key()

    def __ne__(self, other: object) -> bool:
        return not self == other

    def __hash__(self) -> int:
        return hash(("CircuitTargetsInsideInstruction",) + self._key())


class CircuitErrorLocation:
    """One place in the circuit a fault arises, as ``stim.CircuitErrorLocation``; ``str()``
    gives Stim's text."""

    __slots__ = ("noise_tag", "tick_offset", "flipped_pauli_product", "_flipped_measurement", "instruction_targets", "stack_frames")

    def __init__(
        self,
        *,
        tick_offset: int,
        flipped_pauli_product: Iterable[GateTargetWithCoords],
        flipped_measurement: Optional[FlippedMeasurement],
        instruction_targets: CircuitTargetsInsideInstruction,
        stack_frames: Iterable[CircuitErrorLocationStackFrame],
        noise_tag: str = "",
    ) -> None:
        self.noise_tag = str(noise_tag)
        self.tick_offset = int(tick_offset)
        self.flipped_pauli_product = list(flipped_pauli_product)
        # As Stim: no measurement given is record index 0 with an empty observable.
        self._flipped_measurement = FlippedMeasurement(record_index=0, observable=()) if flipped_measurement is None else flipped_measurement
        self.instruction_targets = instruction_targets
        self.stack_frames = list(stack_frames)

    @property
    def flipped_measurement(self) -> Optional[FlippedMeasurement]:
        """The flipped measurement, or None when the fault flips none."""
        return self._flipped_measurement if self._flipped_measurement.observable else None

    def _text(self, indent: str) -> str:
        out = [indent + "CircuitErrorLocation {"]
        if self.noise_tag:
            out.append(indent + "    noise_tag: " + self.noise_tag)
        if self.flipped_pauli_product:
            out.append(indent + "    flipped_pauli_product: " + "*".join(str(p) for p in self.flipped_pauli_product))
        m = self._flipped_measurement
        if m.record_index is not None:
            out.append(indent + f"    flipped_measurement.measurement_record_index: {m.record_index}")
        if m.observable:
            out.append(indent + "    flipped_measurement.measured_observable: " + "*".join(str(p) for p in m.observable))
        out.append(indent + "    Circuit location stack trace:")
        out.append(indent + f"        (after {self.tick_offset} TICKs)")
        n = len(self.stack_frames)
        for k, frame in enumerate(self.stack_frames):
            if k:
                out.append(indent + f"        after {frame.iteration_index} completed iterations")
            what = f" (a REPEAT {frame.instruction_repetitions_arg} block)" if k < n - 1 else f" ({self.instruction_targets.gate})"
            where = " in the REPEAT block" if k else " in the circuit"
            out.append(indent + f"        at instruction #{frame.instruction_offset + 1}{what}{where}")
        it = self.instruction_targets
        if it.target_range_start + 1 == it.target_range_end:
            out.append(indent + f"        at target #{it.target_range_start + 1} of the instruction")
        else:
            out.append(indent + f"        at targets #{it.target_range_start + 1} to #{it.target_range_end} of the instruction")
        out.append(indent + f"        resolving to {it}")
        out.append(indent + "}")
        return "\n".join(out)

    def __str__(self) -> str:
        return self._text("")

    def __repr__(self) -> str:
        pp = "".join(repr(e) + "," for e in self.flipped_pauli_product)
        frames = "".join(repr(e) + "," for e in self.stack_frames)
        return f"stabilizer_qec.CircuitErrorLocation(tick_offset={self.tick_offset}, flipped_pauli_product=({pp}), flipped_measurement={self._flipped_measurement!r}, instruction_targets={self.instruction_targets!r}, stack_frames=({frames}))"

    def _key(self) -> tuple:
        return (self.tick_offset, tuple(self.flipped_pauli_product), self._flipped_measurement, self.instruction_targets, tuple(self.stack_frames))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, CircuitErrorLocation) and self._key() == other._key()

    def __ne__(self, other: object) -> bool:
        return not self == other

    def __hash__(self) -> int:
        return hash(("CircuitErrorLocation",) + self._key())


class ExplainedError:
    """A fault class of the error model (``dem_error_terms``: what it sets off) and every place
    in the circuit such a fault arises, as ``stim.ExplainedError``; ``str()`` gives Stim's
    text."""

    __slots__ = ("dem_error_terms", "circuit_error_locations")

    def __init__(self, *, dem_error_terms: Iterable[DemTargetWithCoords], circuit_error_locations: Iterable[CircuitErrorLocation]) -> None:
        self.dem_error_terms = list(dem_error_terms)
        self.circuit_error_locations = list(circuit_error_locations)

    def __str__(self) -> str:
        out = "ExplainedError {\n    dem_error_terms: " + " ".join(str(t) for t in self.dem_error_terms)
        if not self.circuit_error_locations:
            out += "\n    [no single circuit error had these exact symptoms]"
        for loc in self.circuit_error_locations:
            out += "\n" + loc._text("    ")
        return out + "\n}"

    def __repr__(self) -> str:
        terms = "".join(repr(e) + "," for e in self.dem_error_terms)
        locs = "".join(repr(e) + "," for e in self.circuit_error_locations)
        return f"stabilizer_qec.ExplainedError(dem_error_terms=({terms}), circuit_error_locations=({locs}))"

    def _key(self) -> tuple:
        return (tuple(self.dem_error_terms), tuple(self.circuit_error_locations))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ExplainedError) and self._key() == other._key()

    def __ne__(self, other: object) -> bool:
        return not self == other

    def __hash__(self) -> int:
        return hash(("ExplainedError",) + self._key())


def _targets(raw: list) -> List[GateTargetWithCoords]:
    return [GateTargetWithCoords(t, c) for t, c, _ in raw]


def _frames(stack: list) -> List[CircuitErrorLocationStackFrame]:
    """The core's frames, (offset, completed iterations, repetitions) from the outermost
    block in, as Stim's: each frame's ``iteration_index`` is the pass of the loop enclosing
    it (so the outermost frame's is 0)."""
    out, outer = [], 0
    for offset, iteration, reps in stack:
        out.append(CircuitErrorLocationStackFrame(instruction_offset=offset, iteration_index=outer, instruction_repetitions_arg=reps))
        outer = iteration
    return out


def build(raw: list) -> List[ExplainedError]:
    """The core's tuples as explained errors."""
    out = []
    for _text, terms, locations in raw:
        locs = []
        for tick, pauli, meas, (gate, tag, args), (start, end), targets, stack, _shown, _instruction in locations:
            fm = FlippedMeasurement(record_index=None, observable=()) if meas is None else FlippedMeasurement(record_index=meas[0], observable=_targets(meas[1]))
            locs.append(
                CircuitErrorLocation(
                    tick_offset=tick,
                    flipped_pauli_product=_targets(pauli),
                    flipped_measurement=fm,
                    instruction_targets=CircuitTargetsInsideInstruction(gate=gate, tag=tag, args=args, target_range_start=start, target_range_end=end, targets_in_range=_targets(targets)),
                    stack_frames=_frames(stack),
                    noise_tag=tag,
                )
            )
        out.append(ExplainedError(dem_error_terms=[DemTargetWithCoords(t, c) for t, c, _ in terms], circuit_error_locations=locs))
    return out

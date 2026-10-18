# Ported from Loom (entropicalabs/el-loom 0.4.0, src/loom/executor/circuit_error_model.py and
# legacy_functions/noise_annotated_stim_circuit.py; Copyright 2024 Entropica Labs Pte Ltd, Apache License 2.0;
# see LICENSE in this directory), modified for stabilizer_qec: plain classes with keyword-only constructors in
# place of pydantic models (the same checks in the same order, with Loom's messages, raised as ValueError and
# TypeError rather than wrapped in a pydantic ValidationError), update_gate_durations returns a revalidated copy
# (Loom's raises AttributeError), and noise_annotated_stim_circuit works on this package's Circuit.
"""Loom's circuit error models: the noise its Stim exporter writes into a circuit, an error type
(a Pauli flip, a Pauli channel, one- or two-qubit depolarizing) applied before or after chosen
gates, at the end of every tick to every qubit, or at the end of every tick to each qubit for
the time it idled, with probabilities constant or functions of the time since the circuit began
and the tick's (or the idle) duration, from gate durations."""

from __future__ import annotations

import math
import numbers
from collections import defaultdict
from dataclasses import FrozenInstanceError
from enum import Enum
from typing import Any, Callable, Optional, Protocol, runtime_checkable

from .circuit import Circuit


@runtime_checkable
class ErrorProbProtocol(Protocol):
    """An error probability function: the probabilities (a list of floats, one per parameter
    of the error type) at a time from the circuit's start and a time within the tick (the
    tick's duration, or a qubit's idle time in it)."""

    def __call__(self, time_from_start: Optional[float], time_of_tick: Optional[float]) -> list[float]: ...


GateErrorProbProtocol = Callable[[Optional[float]], list[float]]


class ApplicationMode(str, Enum):
    """Where an error model's noise goes: before or after its gates, at the end of each tick on
    every qubit, or at the end of each tick on each qubit for the time it idled."""

    BEFORE_GATE = "before_gate"
    AFTER_GATE = "after_gate"
    END_OF_TICK = "end_of_tick"
    IDLE_END_OF_TICK = "idle_end_of_tick"


class ErrorType(Enum):
    """An error type, with its label and how many probabilities it takes."""

    PAULI_X = ("pauli_x", 1)
    PAULI_Y = ("pauli_y", 1)
    PAULI_Z = ("pauli_z", 1)
    PAULI_CHANNEL = ("pauli_channel", 3)
    BIT_FLIP = ("bit-flip", 1)
    PHASE_FLIP = ("phase-flip", 1)
    DEPOLARIZING1 = ("depolarizing1", 1)
    DEPOLARIZING2 = ("depolarizing2", 1)

    def __init__(self, label: str, param_count: int) -> None:
        self.label = label
        self.param_count = param_count

    def validate_params(self, params: list[float]) -> None:
        """Raise ValueError unless the parameters are this type's count of numbers in [0, 1]
        (summing to at most 1 when there are several)."""
        if len(params) != self.param_count:
            raise ValueError(f"{self.name} expects {self.param_count} parameter(s), got {len(params)}.")
        if not all(isinstance(p, (float, int)) for p in params):
            raise TypeError(f"All parameters for {self.name} must be numbers.")
        if not all(0 <= p <= 1 for p in params):
            raise ValueError(f"All parameters for {self.name} must be in [0, 1].")
        if self.param_count > 1 and sum(params) > 1.0:
            raise ValueError(f"{self.name} parameters sum must not exceed 1.0.")


def _zero_gate(count: int) -> GateErrorProbProtocol:
    return lambda _: [0.0] * count


def _zero_global(count: int) -> ErrorProbProtocol:
    return lambda _, __: [0.0] * count


def _enum(cls: type, field: str, value: Any) -> Any:
    """A field's enum member, from the member or its value, as pydantic reads it."""
    if isinstance(value, cls):
        return value
    try:
        return cls(value)
    except (ValueError, TypeError):
        values = [repr(m.value) for m in cls]
        raise ValueError(f"{field}: Input should be {', '.join(values[:-1])} or {values[-1]}") from None


def _number(field: str, value: Any) -> float:
    """A float field, as pydantic's lax mode reads it (numbers and numeric strings)."""
    if isinstance(value, numbers.Real):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            pass
    raise ValueError(f"{field}: Input should be a valid number")


def _boolean(field: str, value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, numbers.Integral) and value in (0, 1):
        return bool(value)
    raise ValueError(f"{field}: Input should be a valid boolean")


class CircuitErrorModel:
    """Loom's general circuit error model, frozen once made. ``circuit`` is the circuit the
    model will be applied to (gate times and idle times are computed on it), ``error_type``
    the noise, ``application_mode`` where it goes and ``is_time_dependent`` whether its
    probabilities depend on time, in which case ``gate_durations`` gives every gate's duration.
    ``gate_error_probabilities`` maps gate names to a function of the gate's time (its start
    for BEFORE_GATE, its end for AFTER_GATE; ``None`` when time-independent) giving the
    probabilities, other gates none; ``global_time_error_probability`` is the function of the
    time at the tick's end and the tick's duration (END_OF_TICK) or the qubit's idle time in it
    (IDLE_END_OF_TICK) for the tick-wise modes, and must be defined at 0. Zero probabilities
    write no noise."""

    def __init__(self, *, circuit: Circuit, error_type: ErrorType, is_time_dependent: bool, application_mode: ApplicationMode, gate_durations: Optional[dict[str, float]] = None, gate_error_probabilities: Optional[dict[str, GateErrorProbProtocol]] = None, global_time_error_probability: Optional[ErrorProbProtocol] = None) -> None:
        kwargs = dict(circuit=circuit, error_type=error_type, is_time_dependent=is_time_dependent, application_mode=application_mode, gate_durations=gate_durations, gate_error_probabilities=gate_error_probabilities, global_time_error_probability=global_time_error_probability)
        self._fields(**kwargs)
        self._finish(kwargs)

    # ====== Construction, in pydantic's order: fields, post-init, model checks =====

    def _set(self, **values: Any) -> None:
        for name, value in values.items():
            object.__setattr__(self, name, value)

    def _fields(self, *, circuit, error_type, is_time_dependent, application_mode, gate_durations, gate_error_probabilities, global_time_error_probability) -> None:
        if not isinstance(circuit, Circuit):
            raise ValueError("circuit: Input should be an instance of Circuit")
        error_type = _enum(ErrorType, "error_type", error_type)
        is_time_dependent = _boolean("is_time_dependent", is_time_dependent)
        application_mode = _enum(ApplicationMode, "application_mode", application_mode)
        if gate_durations is not None:
            if not isinstance(gate_durations, dict):
                raise ValueError("gate_durations: Input should be a valid dictionary")
            for name in gate_durations:
                if not isinstance(name, str):
                    raise ValueError(f"gate_durations.{name}: Input should be a valid string")
            gate_durations = {name: _number(f"gate_durations.{name}", d) for name, d in gate_durations.items()}
        if gate_error_probabilities is None:
            gate_error_probabilities = defaultdict(lambda: _zero_gate(1))
        else:
            gate_error_probabilities = self.validate_gate_error_probabilities(gate_error_probabilities)
        if global_time_error_probability is None:
            global_time_error_probability = _zero_global(1)
        else:
            global_time_error_probability = self.validate_global_time_error_probability(global_time_error_probability)
        self._set(circuit=circuit, error_type=error_type, is_time_dependent=is_time_dependent, application_mode=application_mode, gate_durations=gate_durations, gate_error_probabilities=gate_error_probabilities, global_time_error_probability=global_time_error_probability)

    def _finish(self, kwargs: dict) -> None:
        self._post_init()
        self._check()
        object.__setattr__(self, "_kwargs", kwargs)

    def _post_init(self) -> None:
        if self.is_time_dependent and self.gate_durations is not None:
            op_time, tick_durations, idle_times = self._compute_op_times_and_tick_durations()
            self._set(_idle_times=idle_times, _op_time=op_time, _tick_durations=tick_durations)
        else:
            self._set(_op_time={}, _tick_durations=[], _idle_times={})

    def _check(self) -> None:
        if self.is_time_dependent and self.gate_durations is None:
            raise ValueError("gate_durations must be provided for time-dependent error models.")
        if self.application_mode == ApplicationMode.IDLE_END_OF_TICK and not self.is_time_dependent:
            raise ValueError("Idle application mode can only be used with time-dependent error models.")
        self._check_gate_durations(self.gate_durations)
        self._check_gate_error_probabilities_output()
        self._check_global_time_error_probability_output()

    def _check_gate_durations(self, durations: Optional[dict[str, float]]) -> None:
        """Every gate of the circuit has a duration, if durations are given."""
        gate_set = {op.name for layer in Circuit.unroll(self.circuit) for op in layer if op is not None}
        if durations is not None:
            missing = [gate for gate in gate_set if gate not in durations]
            if missing:
                raise ValueError(f"Missing gate durations for: {missing}.Must provide durations, for all gates in {gate_set}")

    def _check_gate_error_probabilities_output(self) -> None:
        """Each gate function returns the error type's count of probabilities; a default single
        zero becomes the right count of zeros."""
        count = self.error_type.param_count
        updated: dict[str, GateErrorProbProtocol] = {}
        for gate_name, func in self.gate_error_probabilities.items():
            try:
                result = func(0.0) if self.is_time_dependent else func(None)
                if len(result) == 1 and result[0] == 0.0 and count > 1:
                    updated[gate_name] = _zero_gate(count)
                else:
                    self.error_type.validate_params(result)
                    updated[gate_name] = func
            except Exception as e:
                raise ValueError(f"Gate '{gate_name}' error probability function failed validation: {e}") from e
        self._set(gate_error_probabilities=defaultdict(lambda: _zero_gate(count), updated))

    def _check_global_time_error_probability_output(self) -> None:
        count = self.error_type.param_count
        try:
            result = self.global_time_error_probability(0.0 if self.is_time_dependent else None, 0.0)
            if len(result) == 1 and result[0] == 0.0 and count > 1:
                self._set(global_time_error_probability=_zero_global(count))
            else:
                self.error_type.validate_params(result)
        except Exception as e:
            raise ValueError(f"global_time_error_probability function failed validation: {e}") from e

    @staticmethod
    def validate_gate_error_probabilities(v: dict) -> dict[str, GateErrorProbProtocol]:
        """A dict of gate names to callables, missing gates defaulting to probability zero."""
        if v is None:
            v = defaultdict(lambda: _zero_gate(1))
        if not isinstance(v, dict):
            raise TypeError("gate_error_probabilities must be a dictionary")
        for key, func in v.items():
            if not callable(func):
                raise TypeError(f"Value for '{key}' must be callable.")
        return defaultdict(lambda: _zero_gate(1), v)

    @staticmethod
    def validate_global_time_error_probability(v) -> ErrorProbProtocol:
        """A callable returning a list of floats in [0, 1] at time 0."""
        if v is None:
            return _zero_global(1)
        if not callable(v):
            raise TypeError("global_time_error_probability must be a callable")
        try:
            result = v(0.0, 0.0)
            if isinstance(result, list):
                if not all(isinstance(p, float) and 0.0 <= p <= 1.0 for p in result):
                    # (Loom's message, its missing space included.)
                    raise ValueError("global_time_error_probability must return a list offloats between 0 and 1")
            else:
                raise TypeError("global_time_error_probability must return a list of floats")
        except Exception as e:
            raise ValueError(f"global_time_error_probability callable failed: {e}") from e
        return v

    # ====== Frozen =====

    def __setattr__(self, name: str, value: Any) -> None:
        raise FrozenInstanceError(f"cannot assign to field {name!r}")

    def __delattr__(self, name: str) -> None:
        raise FrozenInstanceError(f"cannot delete field {name!r}")

    def __repr__(self) -> str:
        return f"{type(self).__name__}(error_type={self.error_type}, application_mode={self.application_mode}, is_time_dependent={self.is_time_dependent}, gate_durations={self.gate_durations})"

    def update_gate_durations(self, gate_durations: dict[str, float]) -> "CircuitErrorModel":
        """A copy of the model with new gate durations, its gate and tick times recomputed."""
        return type(self)(**{**self._kwargs, "gate_durations": gate_durations})

    # ====== Probabilities =====

    def get_idle_tick_error_probability(self, tick_index: int, channel_id: str) -> Optional[list[float]]:
        """The probabilities for a qubit (by channel id) idle in a tick, at the tick's end; None
        when zero or when the model is not IDLE_END_OF_TICK."""
        if self.application_mode != ApplicationMode.IDLE_END_OF_TICK:
            return None
        if channel_id not in self._idle_times:
            raise ValueError(f"Channel {channel_id} not found in idle times mapping.")
        if tick_index < 0 or tick_index >= len(self._tick_durations):
            raise IndexError("Tick index out of range.")
        idle_in_tick = self._idle_times[channel_id][tick_index]
        time = sum(self._tick_durations[: tick_index + 1])
        p = self.global_time_error_probability(time, idle_in_tick)
        if not p or all(x == 0.0 for x in p):
            return None
        return p

    def get_tick_error_probability(self, tick_index: Optional[int] = None) -> Optional[list[float]]:
        """The probabilities at the end of a tick; None when zero or when the model is not
        END_OF_TICK."""
        if self.application_mode != ApplicationMode.END_OF_TICK:
            return None
        if not self.is_time_dependent:
            p = self.global_time_error_probability(None, None)
        else:
            if tick_index is None:
                raise ValueError("tick_index must be provided for time-dependent models.")
            if tick_index < 0 or tick_index >= len(self._tick_durations):
                raise IndexError("Tick index out of range.")
            time = sum(self._tick_durations[: tick_index + 1])
            p = self.global_time_error_probability(time, self._tick_durations[tick_index])
        if not p or all(x == 0.0 for x in p):
            return None
        return p

    def get_gate_error_probability(self, gate: Circuit) -> Optional[list[float]]:
        """The probabilities for a gate (a circuit of the model's circuit, by id when
        time-dependent); None when zero or when the model is END_OF_TICK."""
        if not isinstance(gate, Circuit):
            raise TypeError("gate must be an instance of Circuit")
        if gate.circuit != ():
            raise ValueError("gate must have empty children (no sub-circuit)")
        if self.application_mode == ApplicationMode.END_OF_TICK:
            return None
        if self.is_time_dependent:
            if gate.id not in self._op_time:
                raise ValueError(f"Gate {gate.id} not found in operation time mapping.")
            time = self._op_time.get(gate.id)
            if self.application_mode == ApplicationMode.AFTER_GATE:
                time = self.gate_durations[gate.name] + time
            error_probability = self.gate_error_probabilities[gate.name](time)
        else:
            error_probability = self.gate_error_probabilities[gate.name](None)
        if not error_probability or all(x == 0.0 for x in error_probability):
            return None
        return error_probability

    @property
    def total_time(self) -> float:
        """The circuit's duration, the sum of its ticks' durations."""
        if not self.is_time_dependent or self._tick_durations is None:
            raise ValueError("Circuit is not time-dependent or tick durations are not set.")
        return sum(self._tick_durations)

    def _compute_op_times_and_tick_durations(self) -> tuple[dict[str, float], list[float], dict[str, list[float]]]:
        """Each gate's start time (by id), each tick's duration (its longest gate) and each
        qubit's idle time in each tick (by channel id)."""
        op_time: dict[str, float] = {}
        tick_durations: list[float] = []
        idle: dict[str, list[float]] = defaultdict(list)
        time_stack = 0
        for tick in Circuit.unroll(self.circuit):
            usage: dict[str, float] = defaultdict(lambda: 0)
            for operation in tick:
                if operation.name not in self.gate_durations:
                    raise ValueError(f"Gate {operation.name} not found in gate_durations dictionary.")
                op_time[operation.id] = time_stack
                for ch in operation.channels:
                    usage[ch.id] += self.gate_durations[operation.name]
            tick_duration = max(self.gate_durations[op.name] for op in tick if op is not None)
            tick_durations.append(tick_duration)
            for ch in self.circuit.channels:
                if ch.is_quantum():
                    idle[ch.id].append(tick_duration - usage[ch.id])
            time_stack += tick_duration
        return op_time, tick_durations, idle


def _probabilities(value: Any) -> list[float]:
    if isinstance(value, (list, tuple)):
        try:
            return [_number("error_probability", v) for v in value]
        except ValueError:
            pass
    else:
        try:
            return [_number("error_probability", value)]
        except ValueError:
            pass
    raise ValueError("error_probability must be a float or a list of floats.")


def _gate_names(value: Any) -> list[str]:
    if not isinstance(value, (list, tuple)) or not all(isinstance(g, str) for g in value):
        raise ValueError("target_gates: Input should be a valid list of strings")
    return list(value)


class HomogeneousTimeIndependentCEM(CircuitErrorModel):
    """Constant noise: ``error_probability`` (a float, or a list for a Pauli channel) on every
    gate named in ``target_gates`` (BEFORE_GATE, AFTER_GATE) or on every qubit at every tick's
    end (END_OF_TICK)."""

    def __init__(self, *, circuit: Circuit, error_type: ErrorType, application_mode: ApplicationMode, error_probability: Any, target_gates: Optional[list[str]] = None, is_time_dependent: bool = False, gate_durations: Optional[dict[str, float]] = None, gate_error_probabilities: Optional[dict[str, GateErrorProbProtocol]] = None, global_time_error_probability: Optional[ErrorProbProtocol] = None) -> None:
        kwargs = dict(circuit=circuit, error_type=error_type, application_mode=application_mode, error_probability=error_probability, target_gates=target_gates, is_time_dependent=is_time_dependent, gate_durations=gate_durations, gate_error_probabilities=gate_error_probabilities, global_time_error_probability=global_time_error_probability)
        self._fields(circuit=circuit, error_type=error_type, is_time_dependent=is_time_dependent, application_mode=application_mode, gate_durations=gate_durations, gate_error_probabilities=gate_error_probabilities, global_time_error_probability=global_time_error_probability)
        self._set(error_probability=_probabilities(error_probability), target_gates=_gate_names([] if target_gates is None else target_gates))
        self._finish(kwargs)

    def _post_init(self) -> None:
        p = self.error_probability
        self._set(gate_error_probabilities=self.validate_gate_error_probabilities({g: (lambda _: p) for g in self.target_gates}), global_time_error_probability=lambda _, __: p)
        super()._post_init()


class HomogeneousTimeDependentCEM(CircuitErrorModel):
    """Noise from one probability function: ``error_probability(time_from_start,
    time_of_tick)`` on the gates named in ``target_gates`` (called with the gate's time alone,
    so give ``time_of_tick`` a default) or at every tick's end, on every qubit or on each for
    its idle time; ``gate_durations`` is required."""

    def __init__(self, *, circuit: Circuit, error_type: ErrorType, application_mode: ApplicationMode, error_probability: ErrorProbProtocol, target_gates: Optional[list[str]] = None, is_time_dependent: bool = True, gate_durations: Optional[dict[str, float]] = None, gate_error_probabilities: Optional[dict[str, GateErrorProbProtocol]] = None, global_time_error_probability: Optional[ErrorProbProtocol] = None) -> None:
        kwargs = dict(circuit=circuit, error_type=error_type, application_mode=application_mode, error_probability=error_probability, target_gates=target_gates, is_time_dependent=is_time_dependent, gate_durations=gate_durations, gate_error_probabilities=gate_error_probabilities, global_time_error_probability=global_time_error_probability)
        self._fields(circuit=circuit, error_type=error_type, is_time_dependent=is_time_dependent, application_mode=application_mode, gate_durations=gate_durations, gate_error_probabilities=gate_error_probabilities, global_time_error_probability=global_time_error_probability)
        if not isinstance(error_probability, ErrorProbProtocol):
            raise ValueError("error_probability: Input should be an instance of ErrorProbProtocol")
        self._set(error_probability=error_probability, target_gates=_gate_names([] if target_gates is None else target_gates))
        self._finish(kwargs)

    def _post_init(self) -> None:
        self._set(gate_error_probabilities=self.validate_gate_error_probabilities({g: self.error_probability for g in self.target_gates}), global_time_error_probability=self.error_probability)
        super()._post_init()


class AsymmetricDepolarizeCEM(CircuitErrorModel):
    """T1/T2 noise: a Pauli channel at every tick's end on every qubit for the tick's duration
    ``t`` (or, with ``application_mode=IDLE_END_OF_TICK``, on each qubit for its idle time),
    ``p_x = p_y = (1 - exp(-t / t1)) / 4`` and ``p_z = (1 - exp(-t / t2)) / 2 - p_x``;
    ``t2 <= 2 t1``, and ``gate_durations`` is required."""

    def __init__(self, *, circuit: Circuit, t1: float, t2: float, gate_durations: Optional[dict[str, float]] = None, error_type: ErrorType = ErrorType.PAULI_CHANNEL, application_mode: ApplicationMode = ApplicationMode.END_OF_TICK, is_time_dependent: bool = True, gate_error_probabilities: Optional[dict[str, GateErrorProbProtocol]] = None, global_time_error_probability: Optional[ErrorProbProtocol] = None) -> None:
        kwargs = dict(circuit=circuit, t1=t1, t2=t2, gate_durations=gate_durations, error_type=error_type, application_mode=application_mode, is_time_dependent=is_time_dependent, gate_error_probabilities=gate_error_probabilities, global_time_error_probability=global_time_error_probability)
        self._fields(circuit=circuit, error_type=error_type, is_time_dependent=is_time_dependent, application_mode=application_mode, gate_durations=gate_durations, gate_error_probabilities=gate_error_probabilities, global_time_error_probability=global_time_error_probability)
        values = {}
        for name, v in (("t1", t1), ("t2", t2)):
            v = _number(name, v)
            if v < 0:
                raise ValueError(f"{name} must be non-negative.")
            values[name] = v
        self._set(**values)
        self._finish(kwargs)

    def _check(self) -> None:
        super()._check()
        if self.t2 > 2 * self.t1:
            raise ValueError("t2 must be smaller or equal than 2*t1.")

    def _p(self, t: float) -> list[float]:
        """``[p_x, p_y, p_z]`` after a time ``t``."""
        exp_t_t1 = 0 if self.t1 == 0 else math.exp(-t / self.t1)
        p_x = p_y = (1 - exp_t_t1) / 4
        exp_t_t2 = 0 if self.t2 == 0 else math.exp(-t / self.t2)
        p_z = (1 - exp_t_t2) / 2 - p_x
        return [p_x, p_y, p_z]

    def _post_init(self) -> None:
        self._set(global_time_error_probability=lambda _, t: self._p(t))
        super()._post_init()


def noise_annotated_stim_circuit(stim_circ: Any, before_measure_flip_probability: float = 0, after_clifford_depolarization: float = 0, after_reset_flip_probability: float = 0):
    """Loom's legacy noise pass over a noiseless Stim circuit (this package's ``Circuit``, a
    ``StimProgram`` or Stim text): ``X_ERROR`` before each ``M``, ``DEPOLARIZE1`` after each
    ``H``, ``X``, ``Y``, ``Z``, ``I``, ``DEPOLARIZE2`` after each ``CX``, ``CY``, ``CZ``,
    ``SWAP`` and ``X_ERROR`` after each ``R``, each where its probability is positive."""
    from .._circuit import Circuit as StimCircuit

    if not isinstance(stim_circ, StimCircuit):
        stim_circ = StimCircuit(stim_circ.text if hasattr(stim_circ, "text") else stim_circ)
    one_qubit = ["H", "X", "Y", "Z", "I"]
    two_qubit = ["CX", "CY", "CZ", "SWAP"]
    out = StimCircuit()
    for op in stim_circ:
        name, targets = op.name, op.targets_copy()
        ops = [(name, targets, op.gate_args_copy())]
        if name == "M" and before_measure_flip_probability > 0:
            ops = [("X_ERROR", targets, [before_measure_flip_probability])] + ops
        if name in one_qubit and after_clifford_depolarization > 0:
            ops.append(("DEPOLARIZE1", targets, [after_clifford_depolarization]))
        if name in two_qubit and after_clifford_depolarization > 0:
            ops.append(("DEPOLARIZE2", targets, [after_clifford_depolarization]))
        if name == "R" and after_reset_flip_probability > 0:
            ops.append(("X_ERROR", targets, [after_reset_flip_probability]))
        for n, t, a in ops:
            out.append(n, t, a or None)
    return out

# Ported from Loom (entropicalabs/el-loom 0.4.0, src/loom/executor/{executor,converter,eka_to_stim_converter,
# eka_to_qasm_converter,eka_to_pennylane_converter,eka_to_cudaq_converter,eka_to_mimiq_converter,
# eka_to_guppylang_converter,op_signature}.py; Copyright 2024 Entropica Labs Pte Ltd, Apache License 2.0; see
# LICENSE in this directory), modified for stabilizer_qec: the Stim text becomes this package's Circuit (with
# the noise of the error models in error_model.py) inside a StimProgram that unpacks as Loom's (circuit, qubits,
# records), the qubit indices are ints rather than stim.GateTargets, the Executor is a plain class, and if-else
# circuits are left out (no operation Loom interprets makes one).
"""Loom's executors: an interpreted Eka as a Stim circuit (this package's ``Circuit``, with
detectors and observables, and the noise of Loom's circuit error models), or as OpenQASM 3,
PennyLane, CUDA-Q, Mimiq or Guppy text; each converter's reading of a run's outcome; and the
``Executor`` that picks a converter by ``TargetLanguage``."""

from __future__ import annotations

import ast
import enum
import warnings
from collections import deque
from dataclasses import dataclass
from enum import Enum
from functools import cached_property, partial
from typing import Any, Optional, Sequence, Union

from .circuit import Channel, Circuit
from .error_model import ApplicationMode, AsymmetricDepolarizeCEM, CircuitErrorModel, ErrorProbProtocol, ErrorType, HomogeneousTimeIndependentCEM, noise_annotated_stim_circuit  # noqa: F401 (loom.executor exports them; Loom's docs import noise_annotated_stim_circuit from here)
from .interpreter import InterpretationStep

# An enum member whose value is a functools.partial (which Python 3.13 makes a descriptor).
_enum_member = getattr(enum, "member", lambda value: value)

SINGLE_QUBIT = frozenset({"i", "hadamard", "h", "phase", "phaseinv", "x", "y", "z", "t"})
TWO_QUBIT = frozenset({"cnot", "cx", "cy", "cz", "swap"})
RESETS = frozenset({"reset", "reset_0", "reset_1", "reset_+", "reset_-", "reset_+i", "reset_-i"})
MEASUREMENTS = frozenset({"measurement", "measure_z", "measure_x", "measure_y"})
CLIFFORD_OPS = SINGLE_QUBIT | TWO_QUBIT | RESETS | MEASUREMENTS


def _validate(name: str, n_quantum: int, n_classical: int) -> None:
    if name in SINGLE_QUBIT or name in RESETS:
        want = (1, 0)
    elif name in TWO_QUBIT:
        want = (2, 0)
    elif name in MEASUREMENTS:
        want = (1, 1)
    else:
        raise KeyError(f"Operation {name} not found in operations map.")
    if (n_quantum, n_classical) != want:
        raise ValueError(f"Operation {name} takes {want[0]} quantum and {want[1]} classical channels, not {n_quantum} and {n_classical}.")


# ---------------------------------------------------------------------------------------------
# Stim
# ---------------------------------------------------------------------------------------------

_STIM_OPS: dict[str, Union[str, tuple[str, ...]]] = {
    "i": "I",
    "hadamard": "H",
    "h": "H",
    "phase": "S",
    "phaseinv": "S_DAG",
    "x": "X",
    "y": "Y",
    "z": "Z",
    "cx": "CX",
    "cnot": "CX",
    "cz": "CZ",
    "cy": "CY",
    "swap": "SWAP",
    "reset": "R",
    "reset_0": "R",
    "reset_1": ("R", "X"),
    "reset_+": "RX",
    "reset_-": ("RX", "Z"),
    "reset_+i": "RY",
    "reset_-i": ("RY", "Z"),
    "measurement": "M",
    "measure_z": "M",
    "measure_x": "MX",
    "measure_y": "MY",
}

_STIM_NOISE = {
    ErrorType.PAULI_X: "X_ERROR",
    ErrorType.PAULI_Y: "Y_ERROR",
    ErrorType.PAULI_Z: "Z_ERROR",
    ErrorType.BIT_FLIP: "X_ERROR",
    ErrorType.PHASE_FLIP: "Z_ERROR",
    ErrorType.PAULI_CHANNEL: "PAULI_CHANNEL_1",
    ErrorType.DEPOLARIZING1: "DEPOLARIZE1",
    ErrorType.DEPOLARIZING2: "DEPOLARIZE2",
}

_UNFUSED = {"DETECTOR", "OBSERVABLE_INCLUDE", "QUBIT_COORDS", "TICK", "SHIFT_COORDS", "REPEAT", "MPAD"}


def eka_to_stim_coordinates(label) -> Union[int, tuple]:
    """Loom's qubit coordinates: a data qubit ``(x, y, 0)`` at ``(x + 0.5, y + 0.5)``, an
    ancilla ``(x, y, 1)`` at ``(x, y)``, a second ancilla ``(x, y, 2)`` at ``(x + 0.05, y +
    0.05)``; ``0`` for a label that is no coordinate."""
    if isinstance(label, tuple):
        coords = label
    else:
        try:
            coords = ast.literal_eval(label)
        except (ValueError, SyntaxError):
            return 0
    if not isinstance(coords, tuple):
        return 0
    if len(coords) == 2:
        if coords[1] == 1:
            return (coords[0], 0)
        if coords[1] == 0:
            return (coords[0] - 0.5, 0.5)
        return coords
    if len(coords) == 3:
        if coords[2] == 1:
            return (coords[0], coords[1])
        if coords[2] == 0:
            return (coords[0] + 0.5, coords[1] + 0.5)
        if coords[2] == 2:
            return (coords[0] + 0.05, coords[1] + 0.05)
        raise ValueError(f"Invalid coordinate {coords}. Coordinates should be in the form (x, y, 0), (x, y, 1) or (x, y, 2).")
    raise ValueError(f"Invalid channel label {label}.")


def _num(x) -> str:
    return str(int(x)) if float(x) == int(x) else repr(float(x))


def _stim_line(name: str, targets: Sequence[int], args: Optional[Sequence[float]] = None) -> str:
    """Loom's instruction string: its arguments as Python writes them (exactly), its targets."""
    t = " ".join(str(x) for x in targets)
    if args:
        return f"{name}({', '.join(map(str, args))}) {t}".strip()
    return f"{name} {t}".strip()


def _split(line: str) -> tuple[str, Optional[tuple[float, ...]], str]:
    """An instruction line's name, arguments and targets."""
    head = line.split(" ", 1)[0]
    if "(" in head:
        close = line.index(")")
        name, args = line[:close].split("(", 1)
        return name, tuple(float(a) for a in args.split(",")), line[close + 1:].strip()
    return head, None, line[len(head):].strip()


def _fuse(lines: list[str]) -> list[str]:
    """Consecutive instructions of one name and equal arguments become one, as Stim's parser
    makes them."""
    out: list[str] = []
    last: Optional[tuple[str, Optional[tuple[float, ...]]]] = None
    for line in lines:
        name, args, targets = _split(line)
        if out and name not in _UNFUSED and last == (name, args):
            if targets:
                out[-1] = f"{out[-1]} {targets}"
            continue
        out.append(line)
        last = (name, args)
    return out


def _models_and_ticks(error_models: Any, with_ticks: bool) -> tuple[Optional[Sequence[CircuitErrorModel]], bool]:
    # 2.0 took with_ticks second; Loom, and this port since, takes error_models there.
    if isinstance(error_models, bool):
        warnings.warn("EkaToStimConverter's second argument is error_models, as in Loom: pass with_ticks by keyword", DeprecationWarning, stacklevel=3)
        return None, error_models
    return error_models, with_ticks


@dataclass
class StimProgram:
    """A Stim circuit text with its registers: quantum channels to (coordinates, qubit index),
    classical channels to measurement record index. It unpacks as Loom's converter's result,
    ``circuit, qubits, records = program`` (the circuit this package's ``Circuit``, parsed when
    first asked for)."""

    text: str
    qubits: dict[Channel, tuple[Any, int]]
    records: dict[Channel, int]
    num_measurements: int

    @property
    def circuit(self):
        from .._circuit import Circuit as SqCircuit

        return SqCircuit(self.text)

    def _as_tuple(self) -> tuple:
        return (self.circuit, self.qubits, self.records)

    def __iter__(self):
        return iter(self._as_tuple())

    def __len__(self) -> int:
        return 3

    def __getitem__(self, index):
        return self._as_tuple()[index]


def _step_argument(interpreted_eka: Optional[InterpretationStep], step: Optional[InterpretationStep], owner: str = "Converter") -> InterpretationStep:
    # Loom names the argument interpreted_eka; 2.0 named it step, which still works by keyword.
    if interpreted_eka is None and step is None:
        raise TypeError(f"{owner}.convert() missing 1 required positional argument: 'interpreted_eka'")
    return interpreted_eka if interpreted_eka is not None else step


class Converter:
    """The converters' base (Stim's overrides it all); for the text converters, Loom's
    depth-first emission: a composite circuit's gates in tick order, wrapped in ``Start of
    circuit`` / ``End of circuit`` comments (circuits named ``wrapped_...`` excepted)."""

    comment_prefix = "#"
    ops: dict[str, Any] = {}
    ALLOW_ERROR_MODELS = False

    def emit_circuit_program(self, circuit: Circuit, qmap: dict, cmap: dict) -> str:
        lines: list[str] = []
        stack = deque([circuit])
        while stack:
            current = stack.popleft()
            if current.name == "empty_circuit":
                continue
            if current.name == "comment":
                lines.append(f"{self.comment_prefix} {current.description}")
            elif current.name in self.ops:
                lines.extend(self.emit_leaf(current, qmap, cmap).splitlines())
            else:
                nested = [c for tick in current.circuit for c in tick]
                if not current.name.startswith("wrapped_"):
                    nested = [Circuit(name="comment", description=f"Start of circuit: {current.name}")] + nested + [Circuit(name="comment", description=f" End of circuit: {current.name}")]
                for line in reversed(nested):
                    stack.appendleft(line)
        return "\n".join(lines)

    def emit_leaf(self, circuit: Circuit, qmap: dict, cmap: dict) -> str:
        raise NotImplementedError

    def convert(self, interpreted_eka: Optional[InterpretationStep] = None, *, step: Optional[InterpretationStep] = None):
        return self.convert_circuit(_step_argument(interpreted_eka, step).final_circuit)

    def convert_circuit(self, circuit: Circuit):
        raise NotImplementedError


class EkaToStimConverter(Converter):
    """Loom's Stim converter: ``QUBIT_COORDS`` for the quantum channels sorted by label, each
    unrolled tick's gates grouped by operation (and noise) in order of first appearance, then
    the detectors (``DETECTOR(x, y, t)``) and ``OBSERVABLE_INCLUDE`` lines. With
    ``error_models`` (Loom's circuit error models, made on the circuit converted), each model's
    noise goes before or after the gates it gives a probability, on the group's targets (a later
    model's before-noise first), and the tick-wise models' noise after each tick's gates, on
    every qubit, or on the qubits idle for equal times together. The text is Loom's: what Stim
    refuses (a tick-wise model at probability zero writes its channel with no arguments) this
    package's parser refuses when the circuit is made."""

    SUPPORTED_ERROR_TYPES = frozenset(_STIM_NOISE)
    ALLOW_ERROR_MODELS = True

    def emit_init_instructions(self, circuit: Circuit) -> tuple[list[str], dict[Channel, tuple[Any, int]], dict[Channel, Optional[int]]]:
        quantum = sorted((c for c in circuit.channels if c.is_quantum()), key=lambda c: c.label)
        lines: list[str] = []
        qmap: dict[Channel, tuple[Any, int]] = {}
        for i, chan in enumerate(quantum):
            coords = eka_to_stim_coordinates(chan.label)
            if coords != 0:
                qmap[chan] = (coords, i)
                lines.append(f"QUBIT_COORDS({', '.join(_num(c) for c in coords)}) {i}")
            else:
                qmap[chan] = (None, i)
        cmap = {c: None for c in circuit.channels if c.is_classical()}
        return lines, qmap, cmap

    def emit_circuit_program(self, circuit: Circuit, error_models: Optional[Sequence[CircuitErrorModel]] = None, with_ticks: bool = False) -> tuple[list[str], dict[Channel, tuple[Any, int]], dict[Channel, int], int]:
        error_models, with_ticks = _models_and_ticks(error_models, with_ticks)
        if not circuit.channels or not circuit.circuit:
            return [], {}, {}, 0
        models = list(error_models or [])
        lines, qmap, cmap = self.emit_init_instructions(circuit)
        every_qubit = [i for _, i in qmap.values()]
        counter = 0
        for tick, layer in enumerate(Circuit.unroll(circuit)):
            groups: dict[tuple[str, tuple], list[tuple[list[int], list[Channel]]]] = {}
            for op in layer:
                if op.name not in _STIM_OPS:
                    raise KeyError(f"Operation {op.name} not found in operations map.")
                classical = [c for c in op.channels if c.is_classical()]
                targets = [qmap[c][1] for c in op.channels if c.is_quantum()]
                _validate(op.name, len(targets), len(classical))
                noises = []
                for cem in models:
                    args = cem.get_gate_error_probability(op)
                    if args:
                        noises.append((cem.error_type, cem.application_mode, tuple(args)))
                groups.setdefault((op.name, tuple(noises)), []).append((targets, classical))
            for (name, noises), items in groups.items():
                if name in MEASUREMENTS:
                    for _, classical in items:
                        cmap[classical[0]] = counter
                        counter += 1
                flat = [t for targets, _ in items for t in targets]
                ops = _STIM_OPS[name]
                instructions = [_stim_line(o, flat) for o in (ops if isinstance(ops, tuple) else (ops,))]
                for error_type, mode, args in noises:
                    if mode == ApplicationMode.BEFORE_GATE:
                        instructions = [_stim_line(_STIM_NOISE[error_type], flat, args)] + instructions
                    if mode == ApplicationMode.AFTER_GATE:
                        instructions = instructions + [_stim_line(_STIM_NOISE[error_type], flat, args)]
                lines.extend(instructions)
            for cem in models:
                if cem.application_mode == ApplicationMode.END_OF_TICK:
                    lines.append(_stim_line(_STIM_NOISE[cem.error_type], every_qubit, cem.get_tick_error_probability(tick)))
                if cem.application_mode == ApplicationMode.IDLE_END_OF_TICK:
                    idle: dict[tuple, list[int]] = {}
                    for chan, (_, i) in qmap.items():
                        args = cem.get_idle_tick_error_probability(tick, chan.id)
                        if args:
                            idle.setdefault(tuple(args), []).append(i)
                    lines.extend(_stim_line(_STIM_NOISE[cem.error_type], targets, args) for args, targets in idle.items())
            if with_ticks:
                lines.append("TICK")
        return lines, qmap, {c: i for c, i in cmap.items() if i is not None}, counter

    @staticmethod
    def emit_detectors_and_observables(step: InterpretationStep, records: dict[Channel, int], offset: int = 0) -> list[str]:
        by_label = {c.label: c for c in step.final_circuit.channels}
        out: list[str] = []
        for det in step.detectors:
            targets = [f"rec[{records[by_label[f'{m[0]}_{m[1]}']] - offset}]" for m in det.measurements if isinstance(m, tuple) and f"{m[0]}_{m[1]}" in by_label]
            args: list[str] = []
            if "space_coordinates" in det.labels and "time_coordinate" in det.labels:
                coords = eka_to_stim_coordinates(tuple(det.labels["space_coordinates"]))
                args = [_num(c) for c in tuple(coords) + tuple(det.labels["time_coordinate"])]
            if "color" in det.labels:
                args.append(str(det.labels["color"]))
            out.append(f"DETECTOR({', '.join(args)}) {' '.join(targets)}" if args else f"DETECTOR {' '.join(targets)}")
        for i, obs in enumerate(step.logical_observables):
            targets = [f"rec[{records[by_label[f'{m[0]}_{m[1]}']] - offset}]" for m in obs.measurements if isinstance(m, tuple) and f"{m[0]}_{m[1]}" in by_label]
            out.append(f"OBSERVABLE_INCLUDE({i}) {' '.join(targets)}")
        return out

    def convert_circuit(self, circuit: Circuit, error_models: Optional[Sequence[CircuitErrorModel]] = None, with_ticks: bool = False) -> StimProgram:
        error_models, with_ticks = _models_and_ticks(error_models, with_ticks)
        lines, qmap, records, counter = self.emit_circuit_program(circuit, error_models=error_models, with_ticks=with_ticks)
        return StimProgram("\n".join(_fuse(lines)), qmap, records, counter)

    def convert(self, interpreted_eka: Optional[InterpretationStep] = None, error_models: Optional[Sequence[CircuitErrorModel]] = None, with_ticks: bool = False, *, step: Optional[InterpretationStep] = None) -> StimProgram:
        """The interpreted Eka's final circuit, with its error models' noise, and its detectors
        and observables; unpacks as ``circuit, qubits, records``."""
        step = _step_argument(interpreted_eka, step, "EkaToStimConverter")
        error_models, with_ticks = _models_and_ticks(error_models, with_ticks)
        circuit = step.final_circuit
        if circuit is None or not circuit.circuit:
            return StimProgram("", {}, {}, 0)
        lines, qmap, records, counter = self.emit_circuit_program(circuit, error_models=error_models, with_ticks=with_ticks)
        lines = _fuse(lines) + self.emit_detectors_and_observables(step, records, counter)
        return StimProgram("\n".join(lines), qmap, records, counter)

    @staticmethod
    def parse_target_run_outcome(output) -> dict[str, Union[int, list[int]]]:
        """Loom's reading of a Stim run: ``(samples, records)``, a shots × measurements array
        and the converter's classical channels to record index, to each channel's label and its
        outcomes (a list, or one int for one shot)."""
        import numpy as np

        output, c_reg = output
        output = np.asarray(output)
        res: dict[str, Union[int, list[int]]] = {}
        n_shots = output.shape[0]
        for key, idx in c_reg.items():
            values = output[:, idx].astype(int).tolist()
            res[key.label] = values[0] if n_shots == 1 else values
        return res

    @staticmethod
    def stim_polygons(interpreted_eka: InterpretationStep) -> str:
        """Crumble's ``#!pragma POLYGON`` lines for the stabilizers of the blocks at the start:
        red for X, green for Y, blue for Z, on the data qubits (the second half of the
        stabilizer's qubits, then the first half reversed). As in Loom, the qubit indices count
        the quantum channels in order of their Stim coordinates (the circuit's own indices count
        them in label order), and only single-letter stabilizers are drawn."""
        pauli_polyarg_map = {"X": "(1,0,0,0.5)", "Y": "(0,1,0,0.5)", "Z": "(0,0,1,0.5)"}
        all_stabilizers = [stab for block_uuid in interpreted_eka.block_history.blocks_at(0) for stab in interpreted_eka.block_registry[block_uuid].stabilizers]
        eka_channels = [ast.literal_eval(chan.label) for chan in interpreted_eka.final_circuit.channels if chan.is_quantum()]
        index = {coords: i for i, coords in enumerate(sorted(eka_channels, key=eka_to_stim_coordinates))}
        lines = []
        for stab in all_stabilizers:
            if len(set(stab.pauli)) != 1:
                raise ValueError(f"Unsupported {stab.pauli}.Currently only CSS type codes are supported.")
            data_qubits = [index[q] for q in stab.data_qubits]
            half = int(len(data_qubits) / 2)
            ordered = data_qubits[half:] + data_qubits[:half][::-1]
            lines.append(f"#!pragma POLYGON{pauli_polyarg_map[stab.pauli[0]]} " + " ".join(str(q) for q in ordered))
        return "\n".join(lines)

    def print_stim_circuit_for_crumble(self, final_step: InterpretationStep) -> str:
        """The polygons, then the Stim circuit (as Stim writes it), for pasting into Crumble."""
        return self.stim_polygons(final_step) + "\n" + str(self.convert(final_step).circuit)


def eka_to_circuit(step: InterpretationStep, error_models: Optional[Sequence[CircuitErrorModel]] = None):
    """An interpreted Eka as this package's ``Circuit``, with its detectors and observables and
    the noise of ``error_models`` (Loom's circuit error models, made on ``step.final_circuit``)."""
    return EkaToStimConverter().convert(step, error_models=error_models).circuit


# ---------------------------------------------------------------------------------------------
# Text exporters (OpenQASM 3, PennyLane, CUDA-Q)
# ---------------------------------------------------------------------------------------------


class EkaToQasmConverter(Converter):
    """OpenQASM 3: ``qubit[n] q; bit[m] c;`` over the channels sorted by label."""

    comment_prefix = "//"
    ops = {
        "i": "id", "x": "x", "y": "y", "z": "z", "h": "h", "hadamard": "h", "t": "t", "phase": "s", "phaseinv": "sdg",
        "cnot": "cx", "cx": "cx", "cy": "cy", "cz": "cz", "swap": "swap",
        "reset": "reset", "reset_0": "reset", "reset_1": ["reset", "x"], "reset_+": ["reset", "h"], "reset_-": ["reset", "x", "h"], "reset_+i": ["reset", "h", "s"], "reset_-i": ["reset", "h", "sdg"],
        "measurement": "measure", "measure_z": "measure", "measure_x": ["h", "measure"], "measure_y": ["sdg", "h", "measure"],
    }

    def emit_init_instructions(self, circuit: Circuit) -> tuple[list[str], dict[Channel, int], dict[Channel, int]]:
        quantum = sorted((c for c in circuit.channels if c.is_quantum()), key=lambda c: c.label)
        classical = sorted((c for c in circuit.channels if c.is_classical()), key=lambda c: c.label)
        lines = []
        if quantum:
            lines.append(f"qubit[{len(quantum)}] q;")
        if classical:
            lines.append(f"bit[{len(classical)}] c;")
        return lines, {c: i for i, c in enumerate(quantum)}, {c: i for i, c in enumerate(classical)}

    def emit_leaf(self, circuit: Circuit, qmap: dict, cmap: dict) -> str:
        q = [f"q[{qmap[c]}]" for c in circuit.channels if c.is_quantum()]
        cl = [f"c[{cmap[c]}]" for c in circuit.channels if c.is_classical()]
        _validate(circuit.name, len(q), len(cl))
        op = self.ops[circuit.name]
        ops = op if isinstance(op, list) else [op]
        if circuit.name in TWO_QUBIT:
            return f"{ops[0]} {q[0]}, {q[1]};"
        if circuit.name in MEASUREMENTS:
            return "\n".join([f"{o} {q[0]};" for o in ops[:-1]] + [f"{ops[-1]} {q[0]} -> {cl[0]};"])
        return "\n".join(f"{o} {q[0]};" for o in ops)

    def convert_circuit(self, circuit: Circuit) -> tuple[str, dict[Channel, int], dict[Channel, int]]:
        program = ["OPENQASM 3.0;", 'include "stdgates.inc";']
        if (not circuit.circuit and circuit.name not in self.ops) or not circuit.channels:
            return "# empty input", {}, {}
        init, qmap, cmap = self.emit_init_instructions(circuit)
        program.append("// Init registers")
        program.extend(init)
        program.extend(self.emit_circuit_program(circuit, qmap, cmap).splitlines())
        return "\n".join(program), qmap, cmap

    @staticmethod
    def parse_target_run_outcome(run_output) -> dict[str, list[int]]:
        """Loom's reading of a Qiskit-style run: ``(counts, bits)``, bitstrings (the last bit
        first) with their counts and the converter's classical channels to bit index, to each
        channel's label and its outcome in every shot."""
        bitstrings, channel_to_idx = run_output
        bitstrings = [key for key, count in bitstrings.items() for _ in range(count)]
        result: dict[str, list[int]] = {chan.label: [] for chan in channel_to_idx.keys()}
        if bitstrings == []:
            raise ValueError("No bitstrings found in the run output.")
        width = len(bitstrings[0])
        for channel, idx in channel_to_idx.items():
            result[channel.label] = [int(b[width - 1 - idx]) for b in bitstrings]
        return result


class EkaToCudaqConverter(Converter):
    """CUDA-Q Python: ``q = cudaq.qvector(n)``, measurements into ``c<i>`` with the classical
    channel's label as the register name."""

    comment_prefix = "#"
    ops = {
        "i": "", "x": "x", "y": "y", "z": "z", "h": "h", "hadamard": "h", "t": "t", "phase": "s", "phaseinv": ["z", "s"],
        "cnot": "cx", "cy": "cy", "cz": "cz", "cx": "cx", "swap": "swap",
        "reset": "reset", "reset_0": "reset", "reset_1": ["reset", "x"], "reset_+": ["reset", "h"], "reset_-": ["reset", "x", "h"], "reset_+i": ["reset", "h", "s"], "reset_-i": ["reset", "x", "h", "s"],
        "measurement": "mz", "measure_z": "mz", "measure_x": "mx", "measure_y": "my",
    }

    def emit_init_instructions(self, circuit: Circuit) -> tuple[list[str], dict[Channel, str], dict[Channel, str]]:
        quantum = sorted((c for c in circuit.channels if c.is_quantum()), key=lambda c: c.label)
        classical = sorted((c for c in circuit.channels if c.is_classical()), key=lambda c: c.label)
        return [f"q = cudaq.qvector({len(quantum)})"], {c: f"q[{i}]" for i, c in enumerate(quantum)}, {c: f"c{i}" for i, c in enumerate(classical)}

    def emit_leaf(self, circuit: Circuit, qmap: dict, cmap: dict) -> str:
        q = [qmap[c] for c in circuit.channels if c.is_quantum()]
        cl = [cmap[c] for c in circuit.channels if c.is_classical()]
        labels = [c.label for c in circuit.channels if c.is_classical()]
        _validate(circuit.name, len(q), len(cl))
        op = self.ops[circuit.name]
        if circuit.name in MEASUREMENTS:
            return f"{cl[0]} = {op}({q[0]}, regName='{labels[0]}')"
        if op == "":
            return ""
        if isinstance(op, list):
            return "\n".join(f"{o}({', '.join(q)})" for o in op)
        return f"{op}({', '.join(q)})"

    def convert_circuit(self, circuit: Circuit) -> tuple[str, dict[Channel, str], dict[Channel, str]]:
        if (not circuit.circuit and circuit.name not in self.ops) or not circuit.channels:
            return "# empty input", {}, {}
        init, qmap, cmap = self.emit_init_instructions(circuit)
        lines = list(init)
        lines.extend(self.emit_circuit_program(circuit, qmap, cmap).splitlines())
        return "\n".join(lines), qmap, cmap

    @staticmethod
    def parse_target_run_outcome(run_output, classical_reg_mapping: dict) -> dict[str, list[int]]:
        """Loom's reading of a CUDA-Q ``SampleResult`` (its registers, named after the classical
        channels' labels, and their sequential data), keyed through ``classical_reg_mapping``
        (whose values' labels name the registers)."""
        result_dict = {}
        for register_name in run_output.register_names:
            if register_name == "__global__":
                continue
            result_dict[register_name] = [int(bit) for bit in run_output.get_sequential_data(register_name)]
        map_str_to_label = {v.label: k for k, v in classical_reg_mapping.items()}
        return {map_str_to_label[k]: v for k, v in result_dict.items()}


class EkaToPennylaneConverter(Converter):
    """PennyLane Python: wires in label order (Loom keys its register dict by channel ids;
    here by labels, which is what a reader can use), measurements into ``measurements[label]``."""

    comment_prefix = "#"

    def __init__(self, is_catalyst: bool = False, import_prefix: str = "qml.") -> None:
        if import_prefix and not import_prefix.endswith("."):
            import_prefix += "."
        self.is_catalyst = is_catalyst
        self.import_prefix = import_prefix
        measure = "catalyst_measure" if is_catalyst else f"{import_prefix}measure"
        self.ops = {
            "i": "Identity", "x": "PauliX", "y": "PauliY", "z": "PauliZ", "t": "T", "h": "Hadamard", "hadamard": "Hadamard", "phase": "S", "phaseinv": f"adjoint({import_prefix}S)",
            "cnot": "CNOT", "cy": "CY", "cz": "CZ", "cx": "CNOT", "swap": "SWAP",
            "reset": measure, "reset_0": measure, "reset_1": [measure, "PauliX"], "reset_+": [measure, "Hadamard"], "reset_-": [measure, "PauliX", "Hadamard"], "reset_+i": [measure, "Hadamard", "S"], "reset_-i": [measure, "PauliX", "Hadamard", "S"],
            "measurement": measure, "measure_z": measure, "measure_x": ["Hadamard", measure], "measure_y": [f"adjoint({import_prefix}S)", "Hadamard", measure],
        }

    def _prefixed(self, o: str) -> str:
        return o if (o.startswith(self.import_prefix) or o.startswith("catalyst_")) else f"{self.import_prefix}{o}"

    def emit_init_instructions(self, circuit: Circuit) -> tuple[list[str], dict[Channel, int], dict[Channel, str]]:
        quantum = sorted((c for c in circuit.channels if c.is_quantum()), key=lambda c: c.label)
        qmap = {c: i for i, c in enumerate(quantum)}
        init = {c.label: 1 for c in quantum}
        cmap = {c: f'measurements["{c.label}"]' for c in circuit.channels if c.is_classical()}
        return [f"measurements = {{}}", f"{self.import_prefix}registers({init})"], qmap, cmap

    def emit_leaf(self, circuit: Circuit, qmap: dict, cmap: dict) -> str:
        q = [qmap[c] for c in circuit.channels if c.is_quantum()]
        cl = [cmap[c] for c in circuit.channels if c.is_classical()]
        _validate(circuit.name, len(q), len(cl))
        op = self.ops[circuit.name]
        ops = op if isinstance(op, list) else [op]
        if circuit.name in MEASUREMENTS:
            return "\n".join([f"{self.import_prefix}{o}({q[0]})" for o in ops[:-1]] + [f"{cl[0]} = {ops[-1]}({q[0]})"])
        if circuit.name in RESETS:
            return "\n".join([f"{ops[0]}({q[0]}, reset=True)"] + [f"{self._prefixed(o)}({q})" for o in ops[1:]])
        return "\n".join(f"{self._prefixed(o)}({q})" for o in ops)

    def convert_circuit(self, circuit: Circuit) -> tuple[str, dict[Channel, int], dict[Channel, str]]:
        lines = ["# PennyLane Program Generated from Eka"]
        if (not circuit.circuit and circuit.name not in self.ops) or not circuit.channels:
            return "# empty input", {}, {}
        init, qmap, cmap = self.emit_init_instructions(circuit)
        lines.append("# Initialize Wires from Eka quantum channels Ids")
        lines.extend(init)
        lines.append("")
        lines.extend(self.emit_circuit_program(circuit, qmap, cmap).splitlines())
        return "\n".join(lines), qmap, cmap

    @staticmethod
    def parse_target_run_outcome(outcome) -> dict[str, list[int]]:
        """Loom's reading of a PennyLane or Catalyst run: each key's values, flattened, as
        ints."""
        import numpy as np

        return {key: [int(v) for v in np.ravel(val)] for key, val in outcome.items()}


class EkaToMimiqConverter(Converter):
    """Mimiq Python: ``circuit = mc.Circuit()`` and a ``circuit.push(mc.Gate..., targets)`` per
    gate, tick by tick (the circuit unrolled, so no comments), qubits and bits numbered in label
    order, a measurement's targets its qubit then its bit."""

    comment_prefix = "#"

    def __init__(self, mimiq_import_prefix: str = "mc.", circuit_varname: str = "circuit") -> None:
        if mimiq_import_prefix and not mimiq_import_prefix.endswith("."):
            mimiq_import_prefix += "."
        self.mimiq_import_prefix = mimiq_import_prefix
        self.circuit_varname = circuit_varname
        self.ops = {
            "i": "GateID()", "x": "GateX()", "y": "GateY()", "z": "GateZ()", "h": "GateH()", "hadamard": "GateH()", "t": "GateT()", "phase": "GateS()", "phaseinv": "GateSDG()",
            "cnot": "GateCX()", "cy": "GateCY()", "cz": "GateCZ()", "cx": "GateCX()", "swap": "GateSWAP()",
            "reset": "ResetZ()", "reset_0": "ResetZ()", "reset_1": ["ResetZ()", "GateX()"], "reset_+": "ResetX()", "reset_-": ["ResetX()", "GateZ()"], "reset_+i": "ResetY()", "reset_-i": ["ResetY()", "GateZ()"],
            "measurement": "MeasureZ()", "measure_z": "MeasureZ()", "measure_x": "MeasureX()", "measure_y": "MeasureY()",
            "comment": "comment", "indent_more": "indent_more", "indent_less": "indent_less",
        }

    def emit_init_instructions(self, circuit: Circuit) -> tuple[list[str], dict[Channel, int], dict[Channel, int]]:
        quantum = sorted((c for c in circuit.channels if c.is_quantum()), key=lambda c: c.label)
        classical = sorted((c for c in circuit.channels if c.is_classical()), key=lambda c: c.label)
        return [f"{self.circuit_varname} = {self.mimiq_import_prefix}Circuit()"], {c: i for i, c in enumerate(quantum)}, {c: i for i, c in enumerate(classical)}

    def emit_leaf(self, circuit: Circuit, qmap: dict, cmap: dict) -> str:
        if circuit.name not in self.ops:
            raise ValueError(f"Unsupported operation: {circuit.name}")
        if circuit.name == "comment":
            return f"# {circuit.description}"
        if circuit.name in ("indent_more", "indent_less"):
            raise ValueError(f"Unsupported utils operation: {circuit.name}")
        q = [str(qmap[c]) for c in circuit.channels if c.is_quantum()]
        cl = [cmap[c] for c in circuit.channels if c.is_classical()]
        _validate(circuit.name, len(q), len(cl))
        op = self.ops[circuit.name]
        targets = ", ".join(map(str, q + cl))
        return "\n".join(f"{self.circuit_varname}.push({self.mimiq_import_prefix}{o}, {targets})" for o in (op if isinstance(op, list) else [op]))

    def convert_circuit(self, circuit: Circuit) -> tuple[str, dict[Channel, int], dict[Channel, int]]:
        if (not circuit.circuit and circuit.name not in self.ops) or not circuit.channels:
            return "# empty input", {}, {}
        init, qmap, cmap = self.emit_init_instructions(circuit)
        lines = ["# Mimiq Program Generated from Eka"] + init + [""]
        for layer in Circuit.unroll(circuit):
            for op in layer:
                if op.name not in self.ops:
                    raise ValueError(f"Unsupported operation: {op.name}")
                lines.append(self.emit_leaf(op, qmap, cmap))
        return "\n".join(lines), qmap, cmap

    @staticmethod
    def parse_target_run_outcome(run_output) -> dict[str, list[int]]:
        """Loom's reading of a Mimiq run: ``(result, bits)``, the result's ``histogram()``
        (bitstrings indexed by bit, with counts) and the converter's classical channels to bit
        index, to each channel's label and its outcome in every shot."""
        mimiq_result, channel_to_idx = run_output
        histogram = mimiq_result.histogram()
        bitstrings = [key for key, count in histogram.items() for _ in range(count)]
        result: dict[str, list[int]] = {chan.label: [] for chan in channel_to_idx.keys()}
        if bitstrings == []:
            raise ValueError("No bitstrings found in the run output.")
        for channel, idx in channel_to_idx.items():
            result[channel.label] = [int(b[idx]) for b in bitstrings]
        return result


def _validate_import_prefix(import_prefix: str) -> None:
    if not isinstance(import_prefix, str):
        raise TypeError("import_prefix must be a string")
    if import_prefix and not import_prefix.endswith("."):
        raise ValueError(f"import_prefix '{import_prefix}' must end with a dot (.)")


class EkaToGuppylangConverter(Converter):
    """Guppy (guppylang) code: ``q`` an array of qubits and ``c`` of booleans in label order,
    gates in tick order with Loom's start and end comments, a measurement a
    ``measure_and_reset`` into its bit and a ``result`` named after the bit's label, a SWAP three
    CXs, and the qubits discarded at the end. The prefixes (default ``guppyQ.`` for
    ``guppylang.std.quantum``, ``guppyQSys.`` for ``guppylang.std.qsystem``, ``builtins.`` for
    ``guppylang.std.builtins``) must be empty or end with a dot; ``array`` is used bare."""

    comment_prefix = "#"

    def __init__(self, quantum_prefix: str = "guppyQ.", qsystem_prefix: str = "guppyQSys.", builtins_prefix: str = "builtins.") -> None:
        for prefix in (quantum_prefix, qsystem_prefix, builtins_prefix):
            _validate_import_prefix(prefix)
        self.quantum_prefix, self.qsystem_prefix, self.builtins_prefix = quantum_prefix, qsystem_prefix, builtins_prefix
        q, qs = quantum_prefix, qsystem_prefix
        self.ops = {
            "i": "", "x": f"{q}x", "y": f"{q}y", "z": f"{q}z", "h": f"{q}h", "hadamard": f"{q}h", "t": f"{q}t", "phase": f"{q}s", "phaseinv": f"{q}sdg",
            "cnot": f"{q}cx", "cy": f"{q}cy", "cz": f"{q}cz", "cx": f"{q}cx", "swap": "swap",
            "reset": f"{q}reset", "reset_0": f"{q}reset", "reset_1": [f"{q}reset", f"{q}x"], "reset_+": [f"{q}reset", f"{q}h"], "reset_-": [f"{q}reset", f"{q}x", f"{q}h"], "reset_+i": [f"{q}reset", f"{q}h", f"{q}s"], "reset_-i": [f"{q}reset", f"{q}x", f"{q}h", f"{q}s"],
            "measurement": f"{qs}measure_and_reset", "measure_z": f"{qs}measure_and_reset", "measure_x": [f"{q}h", f"{qs}measure_and_reset"], "measure_y": [f"{q}sdg", f"{q}h", f"{qs}measure_and_reset"],
        }

    def emit_init_instructions(self, circuit: Circuit) -> tuple[list[str], dict[Channel, str], dict[Channel, str]]:
        quantum = sorted((c for c in circuit.channels if c.is_quantum()), key=lambda c: c.label)
        classical = sorted((c for c in circuit.channels if c.is_classical()), key=lambda c: c.label)
        init = [f"q = array({self.quantum_prefix}qubit() for _ in range({len(quantum)}))", f"c = array(False for _ in range({len(classical)}))"]
        return init, {c: f"q[{i}]" for i, c in enumerate(quantum)}, {c: f"c[{i}]" for i, c in enumerate(classical)}

    def emit_leaf(self, circuit: Circuit, qmap: dict, cmap: dict) -> str:
        q = [qmap[c] for c in circuit.channels if c.is_quantum()]
        cl = [cmap[c] for c in circuit.channels if c.is_classical()]
        _validate(circuit.name, len(q), len(cl))
        op = self.ops[circuit.name]
        ops = op if isinstance(op, list) else [op]
        if circuit.name in MEASUREMENTS:
            label = next(c.label for c in circuit.channels if c.is_classical())
            return "\n".join([f"{o}({q[0]})" for o in ops[:-1]] + [f"{cl[0]} = {ops[-1]}({q[0]})", f"{self.builtins_prefix}result('{label}', {cl[0]})"])
        if circuit.name == "swap":
            cx = f"{self.quantum_prefix}cx"
            return "\n".join([f"{cx}({q[0]}, {q[1]})", f"{cx}({q[1]}, {q[0]})", f"{cx}({q[0]}, {q[1]})"])
        if circuit.name in TWO_QUBIT:
            return f"{op}({', '.join(q)})"
        if op == "":
            return ""
        return "\n".join(f"{o}({q[0]})" for o in ops)

    def convert_circuit(self, circuit: Circuit) -> tuple[str, dict[Channel, str], dict[Channel, str]]:
        if (not circuit.circuit and circuit.name not in self.ops) or not circuit.channels:
            return "", {}, {}
        init, qmap, cmap = self.emit_init_instructions(circuit)
        lines = list(init)
        lines.extend(self.emit_circuit_program(circuit, qmap, cmap).splitlines())
        lines.append("for e in q:")
        lines.append(f"   {self.quantum_prefix}discard(e) ")
        return "\n".join(lines), qmap, cmap

    @staticmethod
    def parse_target_run_outcome(outcome) -> dict[str, list]:
        """Loom's reading of a Guppy run (a ``QsysResult``, whose ``results`` are shots, or one
        ``QsysShot``): each shot's ``entries`` ``(label, value)`` gathered by label."""
        result_dict: dict[str, list] = {}
        if hasattr(outcome, "results"):
            for shot in outcome.results:
                for label, value in shot.entries:
                    result_dict.setdefault(label, []).append(value)
        elif hasattr(outcome, "entries"):
            for label, value in outcome.entries:
                result_dict.setdefault(label, []).append(value)
        return result_dict


class TargetLanguage(Enum):
    """Loom's target languages, each a converter (PennyLane twice, plain and Catalyst)."""

    MIMIQ = EkaToMimiqConverter
    OPEN_QASM3 = EkaToQasmConverter
    GUPPYLANG = EkaToGuppylangConverter
    STIM = EkaToStimConverter
    PENNYLANE = _enum_member(partial(EkaToPennylaneConverter, is_catalyst=False))
    PENNYLANE_CATALYST = _enum_member(partial(EkaToPennylaneConverter, is_catalyst=True))
    CUDAQ = EkaToCudaqConverter


class Executor:
    """Loom's executor: a target language's converter (made once, when first used), its export
    of an interpreted Eka or a circuit, and its reading of a run's outcome."""

    def __init__(self, *, target_language: TargetLanguage) -> None:
        if not isinstance(target_language, TargetLanguage):
            try:
                target_language = TargetLanguage(target_language)
            except ValueError:
                raise ValueError(f"target_language: Input should be one of {', '.join(m.name for m in TargetLanguage)}, not {target_language!r}.") from None
        self.target_language = target_language

    def __repr__(self) -> str:
        return f"Executor(target_language={self.target_language})"

    @cached_property
    def converter(self):
        return self.target_language.value()

    def export_eka(self, input_data: InterpretationStep) -> Any:
        return self.converter.convert(input_data)

    def export_eka_circuit(self, input_data: Circuit) -> Any:
        return self.converter.convert_circuit(input_data)

    def parse_outcome(self, run_output: Any) -> dict[str, Union[int, list[int]]]:
        return self.converter.parse_target_run_outcome(run_output)

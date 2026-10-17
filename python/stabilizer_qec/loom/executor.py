# Ported from Loom (entropicalabs/el-loom 0.4.0, src/loom/executor/{converter,eka_to_stim_converter,
# eka_to_qasm_converter,eka_to_pennylane_converter,eka_to_cudaq_converter,op_signature}.py; Copyright 2024
# Entropica Labs Pte Ltd, Apache License 2.0; see LICENSE in this directory), modified for stabilizer_qec: the
# Stim text becomes this package's Circuit, error models and if-else circuits are left out, and the Mimiq and
# Guppy exporters are not ported.
"""Loom's executors: an interpreted Eka as a Stim circuit (this package's ``Circuit``, with
detectors and observables), or as OpenQASM 3, PennyLane or CUDA-Q text."""

from __future__ import annotations

import ast
from collections import deque
from dataclasses import dataclass
from typing import Any, Optional, Union

from .circuit import Channel, Circuit
from .interpreter import InterpretationStep

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


def _fuse(lines: list[str]) -> list[str]:
    """Consecutive instructions of one name and no arguments become one, as Stim's parser makes
    them."""
    out: list[str] = []
    for line in lines:
        head = line.split(" ", 1)[0]
        if out and "(" not in head and head not in _UNFUSED:
            prev = out[-1]
            prev_head = prev.split(" ", 1)[0]
            if prev_head == head:
                out[-1] = prev + line[len(head):]
                continue
        out.append(line)
    return out


@dataclass
class StimProgram:
    """A Stim circuit text with its registers: quantum channels to (coordinates, qubit index),
    classical channels to measurement record index."""

    text: str
    qubits: dict[Channel, tuple[Any, int]]
    records: dict[Channel, int]
    num_measurements: int

    @property
    def circuit(self):
        from .._circuit import Circuit as SqCircuit

        return SqCircuit(self.text)


class EkaToStimConverter:
    """Loom's Stim converter: ``QUBIT_COORDS`` for the quantum channels sorted by label, each
    unrolled tick's gates grouped by operation in order of first appearance, then the detectors
    (``DETECTOR(x, y, t)``) and ``OBSERVABLE_INCLUDE`` lines."""

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

    def emit_circuit_program(self, circuit: Circuit, with_ticks: bool = False) -> tuple[list[str], dict[Channel, tuple[Any, int]], dict[Channel, int], int]:
        if not circuit.channels or not circuit.circuit:
            return [], {}, {}, 0
        lines, qmap, cmap = self.emit_init_instructions(circuit)
        counter = 0
        for layer in Circuit.unroll(circuit):
            groups: dict[str, list[tuple[list[int], list[Channel]]]] = {}
            for op in layer:
                if op.name not in _STIM_OPS:
                    raise KeyError(f"Operation {op.name} not found in operations map.")
                classical = [c for c in op.channels if c.is_classical()]
                targets = [qmap[c][1] for c in op.channels if c.is_quantum()]
                _validate(op.name, len(targets), len(classical))
                groups.setdefault(op.name, []).append((targets, classical))
            for name, items in groups.items():
                if name in MEASUREMENTS:
                    for _, classical in items:
                        cmap[classical[0]] = counter
                        counter += 1
                flat = [t for targets, _ in items for t in targets]
                ops = _STIM_OPS[name]
                for o in (ops if isinstance(ops, tuple) else (ops,)):
                    lines.append(f"{o} {' '.join(str(t) for t in flat)}".strip())
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

    def convert_circuit(self, circuit: Circuit, with_ticks: bool = False) -> StimProgram:
        lines, qmap, records, counter = self.emit_circuit_program(circuit, with_ticks)
        return StimProgram("\n".join(_fuse(lines)), qmap, records, counter)

    def convert(self, step: InterpretationStep, with_ticks: bool = False) -> StimProgram:
        """The interpreted Eka's final circuit with its detectors and observables."""
        circuit = step.final_circuit
        if circuit is None or not circuit.circuit:
            return StimProgram("", {}, {}, 0)
        lines, qmap, records, counter = self.emit_circuit_program(circuit, with_ticks)
        lines = _fuse(lines) + self.emit_detectors_and_observables(step, records, counter)
        return StimProgram("\n".join(lines), qmap, records, counter)


def eka_to_circuit(step: InterpretationStep):
    """An interpreted Eka as this package's ``Circuit`` (with its detectors and observables)."""
    return EkaToStimConverter().convert(step).circuit


# ---------------------------------------------------------------------------------------------
# Text exporters (OpenQASM 3, PennyLane, CUDA-Q)
# ---------------------------------------------------------------------------------------------


class _TextConverter:
    """Loom's depth-first emission: a composite circuit's gates in tick order, wrapped in
    ``Start of circuit`` / ``End of circuit`` comments (circuits named ``wrapped_...`` excepted)."""

    comment_prefix = "#"
    ops: dict[str, Any] = {}

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

    def convert(self, step: InterpretationStep):
        return self.convert_circuit(step.final_circuit)

    def convert_circuit(self, circuit: Circuit):
        raise NotImplementedError


class EkaToQasmConverter(_TextConverter):
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


class EkaToCudaqConverter(_TextConverter):
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


class EkaToPennylaneConverter(_TextConverter):
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

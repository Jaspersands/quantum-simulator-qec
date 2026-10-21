# Ported from Tsim (QuEraComputing/tsim, bloqade-tsim 0.1.5, src/tsim/core/parse.py; Apache License 2.0;
# see LICENSE in this directory), modified for stabilizer_qec: the circuit is a tsim.Circuit of this
# package's operations, which ``tsim_instructions`` reads back as Tsim's tagged Stim (``I[T]`` as ``S[T]``,
# a rotation in radians as ``I[R_Z(theta=a*pi)]`` in half-turns, a Pauli-product rotation as
# ``SPP[R_PAULI(theta=a*pi)]`` or ``SPP[T]``) and parses with Tsim's own code; the graph is built on a
# chosen backend (this package's ZXGraph, or pyzx_param's GraphS).
"""Parser for converting circuits to ZX graph representations."""

from __future__ import annotations

import importlib.util
import math
import re
from fractions import Fraction
from typing import Any, Dict, Iterator, List, Optional, Tuple

from stabilizer_qec import _stim
from stabilizer_qec.tsim.core.tags import is_t_tag
from stabilizer_qec.tsim.utils.program_text import _NATIVE_TAG, _native_args, _turns
from stabilizer_qec.tsim.zx.graph import ZXGraph
from stabilizer_qec.tsim.zx.instructions import (
    GATE_TABLE,
    GraphRepresentation,
    correlated_error,
    detector,
    finalize_correlated_error,
    mpad,
    mpp,
    observable_include,
    r_pauli,
    r_x,
    r_y,
    r_z,
    spp,
    tick,
    tpp,
    u3,
)

_PARAMETRIC_GATE_PARAMS: Dict[str, frozenset] = {
    "R_X": frozenset({"theta"}),
    "R_Y": frozenset({"theta"}),
    "R_Z": frozenset({"theta"}),
    "R_PAULI": frozenset({"theta"}),
    "U3": frozenset({"theta", "phi", "lambda"}),
}

R_PAULI_MAX_QUBITS = 64

BACKENDS = ("own", "pyzx_param")


def pyzx_param_available() -> bool:
    """Whether pyzx_param can be imported (not blocked by a ``None`` in ``sys.modules``)."""
    try:
        return importlib.util.find_spec("pyzx_param") is not None
    except (ImportError, ValueError):
        return False


def require_pyzx_param(what: str) -> None:
    """Raise ``ImportError`` naming ``what`` if pyzx_param is not installed."""
    if not pyzx_param_available():
        raise ImportError(f"{what} needs pyzx_param (pip install pyzx-param), which is not installed")


def new_graph(backend: Optional[str] = None) -> Any:
    """An empty graph of a backend: ``"own"`` (this package's ``ZXGraph``) or ``"pyzx_param"``
    (pyzx_param's ``GraphS``, ``ImportError`` if it is not installed); ``None`` is pyzx_param's when
    installed, else this package's."""
    if backend is None:
        backend = "pyzx_param" if pyzx_param_available() else "own"
    if backend == "own":
        return ZXGraph()
    if backend == "pyzx_param":
        require_pyzx_param("The pyzx_param backend")
        from pyzx_param.graph.graph_s import GraphS

        return GraphS()
    raise ValueError(f"unknown ZX graph backend {backend!r}: one of {', '.join(map(repr, BACKENDS))}")


class TsimInstruction:
    """An instruction in Tsim's tagged-Stim form: a name, a tag, parens arguments and targets
    (``stabilizer_qec.GateTarget``), as the parser below reads a ``stim.CircuitInstruction``."""

    __slots__ = ("name", "tag", "_args", "_raw", "_targets")

    def __init__(self, name: str, tag: str, args: List[float], raw_targets: List[int]) -> None:
        self.name = name
        self.tag = tag
        self._args = list(args)
        self._raw = list(raw_targets)
        self._targets: Optional[List[Any]] = None

    def targets_copy(self) -> List[Any]:
        if self._targets is None:
            self._targets = [_stim.GateTarget._raw(v) for v in self._raw]
        return list(self._targets)

    def gate_args_copy(self) -> List[float]:
        return list(self._args)

    def __str__(self) -> str:
        from stabilizer_qec import _core

        return _core.instruction_text(self.name, self._args, self._raw, self.tag, False)

    __repr__ = __str__


def _pauli_targets(factors: List[Tuple[str, int]]) -> List[int]:
    raw: List[int] = []
    for i, (pauli, qubit) in enumerate(factors):
        if i:
            raw.append(_stim.GateTarget.combiner()._v)
        raw.append(_stim.GateTarget(f"{pauli}{qubit}")._v)
    return raw


def _half_turns(radians: str, scale: float) -> str:
    return _turns(float(radians), scale)


def _spp_of_rotation(theta: float, factors: List[Tuple[str, int]]) -> TsimInstruction:
    """A product rotation exp(-i theta P) as Tsim's ``SPP[T]`` (theta = +-pi/8, the same gate as
    ``R_PAULI(+-0.25)``, which this package prints as ``TPP``) or ``SPP[R_PAULI(theta=a*pi)]``."""
    if abs(theta) == math.pi / 8:
        return TsimInstruction("SPP" if theta > 0 else "SPP_DAG", "T", [], _pauli_targets(factors))
    return TsimInstruction("SPP", f"R_PAULI(theta={_turns(theta, 0.5)}*pi)", [], _pauli_targets(factors))


def tsim_instructions(circuit: Any) -> Iterator[Any]:
    """The instructions of a circuit (loops unrolled, coordinate shifts folded in) in Tsim's tagged-Stim
    form, as ``parse_stim_circuit`` reads them.

    This package writes Tsim's gates as its own operations: ``I[T]`` and ``I[T_DAG]`` for T and its
    inverse, ``I[R_Z(theta=t)]`` (and R_X, R_Y) for exp(-i t Z), ``I[U3(...)]`` in radians,
    ``I[R_PAULI(theta=t, pauli=XY)] 0 1`` and ``II[R_ZZ(theta=t)] 0 1``. They are read back as the
    half-turns they were written in: the shortest decimal number of half-turns that gives the stored
    radians, which is the number written when it was a short decimal (``R_Z(0.3)``), so the phases
    are Tsim's exact fractions. Every other instruction is as it is.
    """
    flat = circuit.flattened()
    for kind, name, tag, args, targets in flat._items():
        if kind != "op":  # pragma: no cover - flattened has no loops
            raise ValueError("a flattened circuit has a REPEAT block")
        converted = _tsim_form(name, tag, args, targets)
        if converted is None:
            yield _stim.CircuitInstruction._make(name, tag, args, targets)
        else:
            yield from converted


def _tsim_form(name: str, tag: str, args: List[float], targets: List[int]) -> Optional[List[TsimInstruction]]:
    if name not in ("I", "II") or not tag:
        return None
    m = _NATIVE_TAG.match(tag)
    if not m:
        return None
    gate, body = m.group(1), m.group(2)
    if name == "I" and body is None and gate in ("T", "T_DAG"):
        return [TsimInstruction("S" if gate == "T" else "S_DAG", "T", args, targets)]
    params = _native_args(body) if body else None
    if params is None:
        return None
    try:
        if name == "I" and gate in ("R_X", "R_Y", "R_Z") and set(params) == {"theta"}:
            return [TsimInstruction("I", f"{gate}(theta={_half_turns(params['theta'], 0.5)}*pi)", args, targets)]
        if name == "I" and gate == "U3" and set(params) == {"theta", "phi", "lambda"}:
            a, b, c = (_half_turns(params[k], 1.0) for k in ("theta", "phi", "lambda"))
            return [TsimInstruction("I", f"U3(theta={a}*pi, phi={b}*pi, lambda={c}*pi)", args, targets)]
        if name == "I" and gate == "R_PAULI" and set(params) == {"theta", "pauli"}:
            pauli = params["pauli"].upper()
            qubits = [_stim.GateTarget._raw(v).value for v in targets]
            if len(pauli) != len(qubits) or not set(pauli) <= set("IXYZ"):
                return None
            factors = [(p, q) for p, q in zip(pauli, qubits) if p != "I"]
            return [_spp_of_rotation(float(params["theta"]), factors)] if factors else []
        if name == "II" and gate in ("R_XX", "R_YY", "R_ZZ") and set(params) == {"theta"} and len(targets) % 2 == 0:
            qubits = [_stim.GateTarget._raw(v).value for v in targets]
            p = gate[2]
            return [_spp_of_rotation(float(params["theta"]), [(p, qubits[i]), (p, qubits[i + 1])]) for i in range(0, len(qubits), 2)]
    except ValueError:
        return None
    return None


class MeasurementRecordError(ValueError, IndexError):
    """An instruction reads a measurement record entry from before the start of the circuit.
    Tsim raises a bare ``IndexError`` ("list index out of range"); this is one too, and a
    ``ValueError`` that says which instruction."""


def _check_lookbacks(b: GraphRepresentation, instruction: Any, lookbacks: List[int]) -> None:
    for k in lookbacks:
        if not -len(b.rec) <= k < 0:
            raise MeasurementRecordError(f"{instruction} reads rec[{k}], but only {len(b.rec)} measurement results precede it.")


def parse_parametric_tag(
    instruction: Any,
) -> Optional[Tuple[str, Dict[str, Fraction]]]:
    """Parse the parametric tag on an instruction (e.g. ``I[R_Z(theta=0.3*pi)]``).

    Supports gates: R_Z, R_X, R_Y, U3.

    Args:
        instruction: The instruction whose tag will be parsed.

    Returns:
        Tuple of (gate_name, params_dict) when the instruction's tag is a
        well-formed parametric tag, or ``None`` when the tag is not
        parametric-looking (no ``name(...)`` shape, or empty).

    Raises:
        ValueError: When the tag looks parametric (matches ``name(...)``) but is
            malformed: a parameter value does not parse, the gate name is unknown,
            or the parameter keys do not match the expected set for the gate.

    """
    tag = instruction.tag
    err_prefix = f"Could not parse instruction {str(instruction)!r}"

    match = re.match(r"^(\w+)\((.*)\)$", tag)
    if not match:
        return None

    gate_name = match.group(1)
    params_str = match.group(2)

    params = {}
    for param in params_str.split(","):
        param = param.strip()
        if not param:
            continue
        # Match param=value*pi (value can be negative/decimal/scientific)
        param_match = re.match(
            r"^(\w+)=([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\*pi$", param
        )
        if not param_match:
            raise ValueError(f"{err_prefix}. Malformed parametric tag {tag!r}")
        param_name = param_match.group(1)
        value = Fraction(param_match.group(2))
        params[param_name] = value

    expected = _PARAMETRIC_GATE_PARAMS.get(gate_name)
    if expected is None:
        raise ValueError(f"{err_prefix}. Unknown parametric gate {gate_name!r}")
    if params.keys() != expected:
        raise ValueError(
            f"{err_prefix}. Parametric tag {tag!r} has parameters "
            f"{sorted(params)}, expected {sorted(expected)}"
        )

    return gate_name, params


_PAULI_PRODUCT: Dict[Tuple[str, str], Tuple[str, int]] = {
    ("X", "Y"): ("Z", 1),  # XY = iZ
    ("X", "Z"): ("Y", 3),  # XZ = -iY
    ("Y", "X"): ("Z", 3),  # YX = -iZ
    ("Y", "Z"): ("X", 1),  # YZ = iX
    ("Z", "X"): ("Y", 1),  # ZX = iY
    ("Z", "Y"): ("X", 3),  # ZY = -iX
}


def _validate_r_pauli_targets(instruction: Any) -> None:
    """Validate an ``R_PAULI``-tagged SPP instruction's raw targets.

    Rejects duplicate target qubits within a single Pauli product (before any
    algebraic simplification) and enforces the per-instruction qubit limit, both
    checked on the raw (un-reduced) target list.
    """
    targets = instruction.targets_copy()
    total_qubits = sum(1 for t in targets if not t.is_combiner)
    if total_qubits > R_PAULI_MAX_QUBITS:
        raise ValueError(
            f"R_PAULI supports at most {R_PAULI_MAX_QUBITS} qubits per instruction, "
            f"got {total_qubits}."
        )

    seen: set = set()
    for i, target in enumerate(targets):
        if target.is_combiner:
            continue
        if target.value in seen:
            raise ValueError(
                f"R_PAULI target qubits must be distinct within a product, "
                f"got repeated qubit {target.value} in {str(instruction)!r}."
            )
        seen.add(target.value)
        next_idx = i + 1
        if next_idx >= len(targets) or not targets[next_idx].is_combiner:
            seen = set()


def _iter_pauli_products(
    instruction: Any,
) -> Iterator[Tuple[List[Tuple[str, int]], bool]]:
    """Yield (paulis, invert) for each Pauli product in an instruction.

    Tracks Pauli algebra when the same qubit appears multiple times:
    same Pauli cancels (P·P = I), different Paulis combine with a sign
    (e.g. Z·X = iY).  An accumulated sign of -1 flips the invert flag;
    ±i raises ValueError (anti-Hermitian product), matching stim.
    """
    qubit_pauli: Dict[int, str] = {}
    sign = 0  # accumulated power of i (mod 4)
    invert = False
    targets = instruction.targets_copy()

    for i, target in enumerate(targets):
        if target.is_combiner:
            continue

        if target.is_x_target:
            pauli_type = "X"
        elif target.is_y_target:
            pauli_type = "Y"
        elif target.is_z_target:
            pauli_type = "Z"
        else:
            raise ValueError(
                f"Invalid Pauli target in instruction {instruction.name}: {target}"
            )

        invert ^= target.is_inverted_result_target
        qubit = target.value

        if qubit not in qubit_pauli:
            qubit_pauli[qubit] = pauli_type
        elif qubit_pauli[qubit] == pauli_type:
            # P·P = I — cancel with no sign change
            del qubit_pauli[qubit]
        else:
            # Different Paulis on same qubit — combine with sign
            result, delta = _PAULI_PRODUCT[qubit_pauli[qubit], pauli_type]
            qubit_pauli[qubit] = result
            sign = (sign + delta) % 4

        next_idx = i + 1
        if next_idx >= len(targets) or not targets[next_idx].is_combiner:
            if sign % 2 == 1:
                raise ValueError(f"{instruction} acted on an anti-Hermitian operator")
            paulis = [(p, q) for q, p in sorted(qubit_pauli.items())]
            yield paulis, invert ^ (sign == 2)
            qubit_pauli = {}
            sign = 0
            invert = False


def parse_stim_circuit(
    circuit: Any,
    track_classical_wires: bool = False,
    backend: Optional[str] = None,
) -> GraphRepresentation:
    """Parse a circuit into a GraphRepresentation.

    Args:
        circuit: The circuit to convert: a ``tsim.Circuit``, or anything it is built from (Tsim's
            program text, a ``stabilizer_qec.Circuit``, a ``stim.Circuit``).
        track_classical_wires: Whether to track classical wires.
        backend: ``"own"`` (this package's ``ZXGraph``), ``"pyzx_param"`` (pyzx_param's ``GraphS``) or
            ``None`` (pyzx_param's if installed, else this package's).

    Returns:
        A GraphRepresentation containing the ZX graph and all auxiliary data.

    """
    from stabilizer_qec.tsim.circuit import Circuit

    if not isinstance(circuit, Circuit):
        circuit = Circuit(circuit)
    b = GraphRepresentation(graph=new_graph(backend), track_classical_wires=track_classical_wires)

    for instruction in tsim_instructions(circuit):
        name = instruction.name
        if name == "SHIFT_COORDS":

            # TODO: handle visualization annotations in ZX diagrams
            continue

        if any(t.is_sweep_bit_target for t in instruction.targets_copy()):
            raise NotImplementedError(
                f"Sweep bit targets (e.g. sweep[N]) are not supported "
                f"in instruction {str(instruction)!r}"
            )

        if name == "S" and is_t_tag(instruction.tag):
            name = "T"
        elif name == "S_DAG" and is_t_tag(instruction.tag):
            name = "T_DAG"

        # Handle parametric gates via tags (e.g., I with tag "R_Z(theta=0.3*pi)")
        if name == "I" and instruction.tag:
            result = parse_parametric_tag(instruction)
            if result is not None:
                gate_name, params = result
                targets = [t.value for t in instruction.targets_copy()]
                for qubit in targets:
                    if gate_name == "R_Z":
                        r_z(b, qubit, params["theta"])
                    elif gate_name == "R_X":
                        r_x(b, qubit, params["theta"])
                    elif gate_name == "R_Y":
                        r_y(b, qubit, params["theta"])
                    elif gate_name == "U3":
                        u3(b, qubit, params["theta"], params["phi"], params["lambda"])
                    else:
                        raise ValueError(f"Unknown parametric gate: {gate_name}")
                continue

        if name == "TICK":
            tick(b)
            continue
        if name == "MPP":
            args = instruction.gate_args_copy()
            p = args[0] if args else 0
            for paulis, invert in _iter_pauli_products(instruction):
                mpp(b, paulis, invert, p=p)
            continue
        if name in ("SPP", "SPP_DAG") and is_t_tag(instruction.tag):
            is_dag = name == "SPP_DAG"
            for paulis, invert in _iter_pauli_products(instruction):
                tpp(b, paulis, dagger=is_dag ^ invert)
            continue
        if name in ("SPP", "SPP_DAG") and instruction.tag:
            parsed = parse_parametric_tag(instruction)
            if parsed is not None and parsed[0] == "R_PAULI":
                params = parsed[1]
                _validate_r_pauli_targets(instruction)
                is_dag = name == "SPP_DAG"
                for paulis, invert in _iter_pauli_products(instruction):
                    r_pauli(b, paulis, params["theta"], dagger=is_dag ^ invert)
                continue
        if name in ("SPP", "SPP_DAG"):
            is_dag = name == "SPP_DAG"
            for paulis, invert in _iter_pauli_products(instruction):
                spp(b, paulis, dagger=is_dag ^ invert)
            continue
        if name == "MPAD":
            args = instruction.gate_args_copy()
            p = args[0] if args else 0
            for target in instruction.targets_copy():
                mpad(b, target.value, p=p)
            continue
        if name == "E" or name == "ELSE_CORRELATED_ERROR":
            if name == "E":
                finalize_correlated_error(b)
            targets = [t.value for t in instruction.targets_copy()]
            types: List[str] = []
            for t in instruction.targets_copy():
                if t.is_x_target:
                    types.append("X")
                elif t.is_y_target:
                    types.append("Y")
                elif t.is_z_target:
                    types.append("Z")
                else:
                    raise ValueError(f"Invalid target: {t}")
            correlated_error(b, targets, types, instruction.gate_args_copy()[0])
            continue
        if name == "DETECTOR":
            targets = [t.value for t in instruction.targets_copy()]
            _check_lookbacks(b, instruction, targets)
            detector(b, targets)
            continue
        if name == "OBSERVABLE_INCLUDE":
            targets_copy = instruction.targets_copy()
            for t in targets_copy:
                if not t.is_measurement_record_target:
                    raise ValueError(
                        f"OBSERVABLE_INCLUDE with Pauli targets is not "
                        f"supported in Tsim (only measurement record targets "
                        f"like rec[-1] are supported). Got instruction "
                        f"{str(instruction)!r}"
                    )
            targets = [t.value for t in targets_copy]
            _check_lookbacks(b, instruction, targets)
            args = instruction.gate_args_copy()
            observable_include(b, targets, int(args[0]))
            continue

        # instruction dispatch
        if name not in GATE_TABLE:
            raise ValueError(f"Unknown gate: {name}")

        gate_func, num_qubits = GATE_TABLE[name]
        targets = [t.value for t in instruction.targets_copy()]
        invert = [t.is_inverted_result_target for t in instruction.targets_copy()]
        is_classically_controlled = [
            t.is_measurement_record_target for t in instruction.targets_copy()
        ]
        args = instruction.gate_args_copy()
        _check_lookbacks(b, instruction, [t for t, cc in zip(targets, is_classically_controlled) if cc])

        for i_target in range(0, len(targets), num_qubits):
            chunk = targets[i_target : i_target + num_qubits]
            cc_chunk = is_classically_controlled[i_target : i_target + num_qubits]
            chunk_inverted = False
            for j in range(num_qubits):
                chunk_inverted ^= invert[i_target + j]
            assert not (invert[i_target] and is_classically_controlled[i_target])
            if chunk_inverted:
                gate_func(b, *chunk, *args, invert=True)
            elif any(cc_chunk):
                gate_func(b, *chunk, *args, classically_controlled=cc_chunk)
            else:
                gate_func(b, *chunk, *args)

    finalize_correlated_error(b)

    # Materialize every observable id from 0..num_observables-1 so missing
    # indices appear as deterministic-zero outputs and downstream iteration
    # is in sorted index order, matching Stim semantics.
    for i in range(circuit.num_observables):
        if i not in b.observables_dict:
            observable_include(b, [], i)
    b.observables_dict = {i: b.observables_dict[i] for i in sorted(b.observables_dict)}

    return b

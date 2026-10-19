"""Deltakit's Stim (``deltakit_stim`` 0.2.7, Riverlane's fork of Stim 1.16) on this package:
``import stabilizer_qec.deltakit_stim as stim`` where Deltakit writes ``import deltakit_stim as
stim``.

It is this package's Stim with Deltakit's leakage instructions native: ``LEAKAGE(p) q``
(q leaks with probability p and is depolarized), ``RELAX(p) q`` (a leaked q returns with
probability p), ``RL q`` (q returns), ``HERALD_LEAKAGE_EVENT(p) q`` (a record of whether q is
leaked, a leaked q read as unleaked with probability p), and ``CX(p1, p2, p3, p4)`` (and
``CY``, ``CZ``, ``SQRT_XX``, ``SQRT_XX_DAG``), whose arguments spread leakage from each qubit
to the other and move it there. A circuit prints, iterates and indexes them as Deltakit's
does (``circuit[1].name == "LEAKAGE"``), and runs them as Deltakit's does: its samplers run
Deltakit's frame simulator's leakage model (a leaked qubit depolarizes its partner in those
gates; a reset returns it), and its ``detector_error_model`` is Deltakit's, with the leakage
its heralds read (each herald firing with its qubit depolarized, and with each transport
gate's partner depolarized after the leakage came). Everything else is this package's Stim.

The circuit holds Deltakit's instructions as tagged lines Stim reads (``I_ERROR[LEAKAGE(p=…)]``
and the rest, see ``Circuit.to_deltakit_text``), so ``stabilizer_qec.Circuit(c)`` is the same
circuit to Stim, where nothing leaks.

The transport gates follow deltakit_stim's documentation: spreading, then mobility each way.
deltakit_stim 0.2.7's simulator departs from it when one gate is given spreading and mobility
together, applying only the spreading (``CX(0.1, 0.2, 0.3, 0.4)`` with the control leaked at
0.2 leaves the target leaked with probability 0.020 there, 0.057 by the documented rule, and
here); with one kind of argument, and in every detector error model, the two agree. Decoded
by matching without leakage awareness, Deltakit's bundled leakage circuits get other answers
than PyMatching's on 6–31% of shots, every one a tie between equally light corrections (the
heralds' edges weigh nothing), and logical error rates 4–5% apart either way.

>>> import stabilizer_qec.deltakit_stim as stim
>>> c = stim.Circuit("R 0\\nLEAKAGE(0.1) 0\\nHERALD_LEAKAGE_EVENT(0.01) 0\\nM 0\\nDETECTOR rec[-1]\\nDETECTOR rec[-2]")
>>> c[1]
stabilizer_qec.deltakit_stim.CircuitInstruction('LEAKAGE', [stabilizer_qec.GateTarget(0)], [0.1])
>>> print(c.detector_error_model())
error(0.5) D0 D1
error(0.01) D1
"""

from __future__ import annotations

import re
from typing import Any, Iterable, List, Optional, Union

import numpy as np

import stabilizer_qec as _sq

from . import _circuit, _core, _stim
from ._util import b8_to_rows, call, count, seed_of

__version__ = "0.2.7"

# The instructions Deltakit adds, and the gates its four leakage arguments go on.
_LEAKAGE_GATES = ("LEAKAGE", "RELAX", "RL", "HERALD_LEAKAGE_EVENT")
_TRANSPORT_GATES = ("CX", "CY", "CZ", "SQRT_XX", "SQRT_XX_DAG")
_TRANSPORT_KEYS = ("spread", "spread_back", "mobility", "mobility_back")
_HERALD_TAG = re.compile(r"\s*HERALD_LEAKAGE_EVENT\s*\(\s*qubits\s*=([^)]*)\)\s*")
_NOISE_TAG = re.compile(r"\s*(LEAKAGE|RELAX)\s*\(\s*p\s*=([^)]*)\)\s*")
_TRANSPORT_TAG = re.compile(r"\s*LEAKAGE_TRANSPORT\s*\(([^)]*)\)\s*")
_DIALECT = ("[LEAKAGE(", "[RELAX(", "[RL]", "[HERALD_LEAKAGE_EVENT(", "[LEAKAGE_TRANSPORT(")


def __getattr__(name: str) -> Any:
    # Everything Deltakit's Stim shares with Stim is this package's.
    try:
        return getattr(_sq, name)
    except AttributeError:
        raise AttributeError(f"module 'stabilizer_qec.deltakit_stim' has no attribute '{name}'") from None


def _transport(tag: str) -> Optional[List[float]]:
    m = _TRANSPORT_TAG.fullmatch(tag)
    if m is None:
        return None
    out = []
    for part, key in zip(m.group(1).split(","), _TRANSPORT_KEYS):
        k, _, v = part.partition("=")
        if k.strip() != key:
            return None
        out.append(float(v))
    return out if len(out) == 4 else None


def _dialect_op(name: str, tag: str, args: list, targets: list) -> Optional[tuple]:
    """A lowered instruction as Deltakit's ``(name, args, targets)``, or None for any other."""
    if name in ("I_ERROR",) and (m := _NOISE_TAG.fullmatch(tag)):
        return m.group(1), [float(m.group(2))], targets
    if name == "I" and tag.strip() == "RL":
        return "RL", [], targets
    if name == "MPAD" and (m := _HERALD_TAG.fullmatch(tag)):
        # An inverted target is marked in the tag (or, as written before, by a pad of 1).
        qubits = [(int(q.lstrip("!")), q.startswith("!")) for q in m.group(1).split()]
        if len(qubits) == len(targets):
            inv = 1 << 31
            return "HERALD_LEAKAGE_EVENT", list(args), [q | (inv if i or v else 0) for (q, i), v in zip(qubits, targets)]
    if name in _TRANSPORT_GATES and (p := _transport(tag)) is not None:
        return name, p, targets
    return None


def _item_object(it: Any) -> Any:
    if it[0] == "op":
        _, name, tag, args, targets = it
        op = _dialect_op(name, tag, args, targets)
        if op is not None:
            return CircuitInstruction._make(op[0], "", op[1], op[2])
        return CircuitInstruction._make(name, tag, args, targets)
    _, count_, tag, body = it
    out = CircuitRepeatBlock.__new__(CircuitRepeatBlock)
    out._count, out._tag, out._body = count_, tag, Circuit._of_items(body)
    return out


def _has_dialect(text: str) -> bool:
    return any(d in text for d in _DIALECT)


class Circuit(_circuit.Circuit):
    """A circuit of Deltakit's Stim: Stim's, with ``LEAKAGE``, ``RELAX``, ``RL``,
    ``HERALD_LEAKAGE_EVENT`` and the transport gates' four arguments native (see the module).

    >>> c = Circuit("CX(0.01, 0.02, 0.03, 0.04) 0 1\\nRL 1")
    >>> print(c)
    CX(0.01, 0.02, 0.03, 0.04) 0 1
    RL 1
    """

    __slots__ = ()

    _deltakit = True

    def __str__(self) -> str:
        text = call(_core.circuit_to_deltakit, super().__str__())
        return text[:-1] if text.endswith("\n") else text

    def __repr__(self) -> str:
        text = str(self)
        if not text:
            return "stabilizer_qec.deltakit_stim.Circuit()"
        body = "\n".join("    " + line if line else line for line in text.split("\n"))
        return f"stabilizer_qec.deltakit_stim.Circuit('''\n{body}\n''')"

    def to_deltakit_text(self) -> str:
        return str(self) + "\n"

    def __iter__(self):
        for it in self._items():
            yield _item_object(it)

    def __getitem__(self, index_or_slice: Any) -> Any:
        got = super().__getitem__(index_or_slice)
        if isinstance(got, _circuit.Circuit):
            return got
        k = int(index_or_slice)
        items = self._items()
        return _item_object(items[k if k >= 0 else k + len(items)])

    def pop(self, index: int = -1) -> Any:
        items = self._items()
        k = int(index)
        if k < 0:
            k += len(items)
        if not 0 <= k < len(items):
            raise IndexError(f"index {index} out of range")
        out = _item_object(items[k])
        self._set_items(items[:k] + items[k + 1 :])
        return out

    def append(self, name: Union[str, Any], targets: Any = (), arg: Any = None, *, tag: str = "") -> None:
        """Stim's ``append``, Deltakit's instructions included (``append("LEAKAGE", [0],
        0.01)``, ``append("CX", [0, 1], [0.01, 0.02, 0.03, 0.04])``)."""
        if isinstance(name, str) and (_dialect_name(name, arg) is not None):
            op = CircuitInstruction(name, _circuit._targets(targets) if not isinstance(targets, (list, tuple)) else targets, _circuit._args(arg), tag=tag)
            self.append_from_stim_program_text(op._stim_exact_text())
            return
        super().append(name, targets, arg, tag=tag)

    append_operation = append

    def without_noise(self) -> "Circuit":
        """Noise removed, as Deltakit's Stim removes it: its ``LEAKAGE`` and ``RELAX`` too, the
        transport gates' arguments, and a herald's flip probability; ``RL`` stays."""
        out = super().without_noise()
        text = re.sub(r"\[LEAKAGE_TRANSPORT\([^)\]]*\)\]", "", out._stim_exact_text())
        return type(self)(text)

    def without_tags(self) -> "Circuit":
        """The tags removed, Deltakit's instructions kept (they carry no tags in its Stim; here
        their lowered forms' tags are what they are)."""
        return type(self)(re.sub(r"(?m)^(\s*[A-Za-z_][A-Za-z0-9_]*)\[[^\]\n]*\]", r"\1", str(self)))

    def decomposed(self) -> "Circuit":
        """Stim's ``decomposed``; Deltakit's instructions stay (the transport gates' arguments
        go, as Deltakit's Stim drops them)."""
        # RL rides through as a noise channel, which decomposition keeps.
        text = re.sub(r"(?m)^(\s*)I\[RL\]", r"\1I_ERROR[RL]", self._stim_exact_text())
        out = _circuit.Circuit(text).decomposed()._stim_exact_text()
        out = re.sub(r"(?m)^(\s*)I_ERROR\[RL\]", r"\1I[RL]", out)
        return type(self)(re.sub(r"\[LEAKAGE_TRANSPORT\([^)\]]*\)\]", "", out))

    def compile_detector_sampler(self, *, seed: Union[int, None] = None) -> "_circuit.DetectorSampler":
        """Detection events and observable flips, as Deltakit's frame simulator draws them
        (its leakage model; Stim's sampler for a circuit without leakage)."""
        if _has_dialect(self._stim_exact_text()):
            return _LeakyDetectorSampler(self, seed)
        return super().compile_detector_sampler(seed=seed)

    def compile_sampler(self, *, skip_reference_sample: bool = False, seed: Union[int, None] = None, reference_sample: Any = None) -> "_circuit.MeasurementSampler":
        """Measurement records, as Deltakit's frame simulator draws them (its leakage model;
        Stim's sampler for a circuit without leakage). A herald's reference is 0."""
        if _has_dialect(self._stim_exact_text()):
            return _LeakyMeasurementSampler(self, seed, skip_reference_sample, reference_sample)
        return super().compile_sampler(skip_reference_sample=skip_reference_sample, seed=seed, reference_sample=reference_sample)


def _dialect_name(name: str, arg: Any) -> Optional[str]:
    n = name.strip().upper()
    if n in _LEAKAGE_GATES:
        return n
    try:
        canonical = _stim.GateData(n).name
    except IndexError:
        return None
    if canonical in _TRANSPORT_GATES and arg is not None and not isinstance(arg, (int, float)) and len(list(arg)) == 4:
        return canonical
    return None


class _LeakyDetectorSampler(_circuit.DetectorSampler):
    """Deltakit's frame simulator's shots (its leakage model), one shot at a time."""

    __slots__ = ()

    def __init__(self, circuit: Circuit, seed: Union[int, None] = None) -> None:
        self._circuit = circuit
        self._s = call(circuit._c.leakage_sampler, seed_of(seed), False, True)

    def __repr__(self) -> str:
        return f"stabilizer_qec.deltakit_stim.CompiledDetectorSampler({self._circuit!r})"

    def _draw(self, shots: int, threads: int) -> tuple:
        d, o, _ = call(self._s.sample, shots, threads)
        return b8_to_rows(d, shots, self.num_detectors, False), b8_to_rows(o, shots, self.num_observables, False)


class _LeakyMeasurementSampler(_circuit.MeasurementSampler):
    """Deltakit's frame simulator's measurement records, one shot at a time."""

    def __init__(self, circuit: Circuit, seed: Union[int, None] = None, skip_reference_sample: bool = False, reference_sample: Any = None) -> None:
        super().__init__(circuit, seed, skip_reference_sample, reference_sample)
        self._leaky = call(circuit._c.leakage_sampler, seed_of(seed), False, True)
        self._skip = skip_reference_sample or reference_sample is not None
        self._herald_reference = None
        if self._skip:
            # Flips alone: the records less the reference (a herald's is 0).
            ref = np.asarray(call(_core.circuit_herald_map, circuit._stim_exact_text())[0], dtype=np.int64)
            base = circuit.reference_sample()
            base[ref] = False
            self._herald_reference = base

    def __repr__(self) -> str:
        return f"stabilizer_qec.deltakit_stim.CompiledMeasurementSampler({self._circuit!r})"

    def sample(self, shots: int, *, bit_packed: bool = False, threads: int = 1) -> np.ndarray:
        shots = count(shots, "shots")
        raw = call(self._leaky.sample_measurements, shots, count(threads, "threads"))
        rows = b8_to_rows(raw, shots, self.num_measurements, False)
        if self._herald_reference is not None:
            rows ^= self._herald_reference[None, :]
            if self._reference is not None:
                rows ^= self._reference[None, :]
        return _circuit.pack_rows(rows, bit_packed)


class CircuitInstruction(_stim.CircuitInstruction):
    """One instruction, as ``deltakit_stim.CircuitInstruction``: Stim's, and Deltakit's
    ``LEAKAGE``, ``RELAX``, ``RL``, ``HERALD_LEAKAGE_EVENT`` and transport gates with four
    arguments."""

    __slots__ = ()

    def __init__(self, name: str, targets: Optional[Iterable[Any]] = None, gate_args: Optional[Iterable[float]] = None, *, tag: str = "") -> None:
        if targets is None and gate_args is None and not tag and isinstance(name, str) and (" " in name.strip() or "(" in name):
            c = Circuit(name)
            if len(c) != 1 or isinstance(c[0], CircuitRepeatBlock):
                raise ValueError(f"Expected exactly one instruction but got {name!r}")
            got = c[0]
            self._name, self._tag, self._args, self._targets = got._name, got._tag, got._args, got._targets
            return
        args = [float(a) for a in (gate_args or [])]
        dialect = _dialect_name(str(name), args if args else None)
        if dialect is None:
            super().__init__(name, targets, gate_args, tag=tag)
            return
        if tag:
            raise ValueError(f"Deltakit's {dialect} takes no tag here")
        self._name = dialect
        self._targets = [_stim._gate_target_of(t) for t in (targets or [])]
        self._args = args
        self._tag = ""
        Circuit(self._stim_exact_text())  # validates as Deltakit's Stim does

    def _dialect(self) -> bool:
        return self._name in _LEAKAGE_GATES or (self._name in _TRANSPORT_GATES and len(self._args) == 4)

    def _text(self, exact: bool) -> str:
        fmt = (lambda a: repr(float(a))) if exact else _stim._fmt_arg
        head = self._name + (f"({', '.join(fmt(a) for a in self._args)})" if self._args else "")
        return " ".join([head] + [f"!{t.value}" if t.is_inverted_result_target else str(t.value) for t in self._targets])

    def _stim_exact_text(self) -> str:
        if self._dialect():
            return self._text(True)
        return super()._stim_exact_text()

    def __str__(self) -> str:
        if self._dialect():
            return self._text(False)
        return super().__str__()

    def __repr__(self) -> str:
        return "stabilizer_qec.deltakit_stim." + super().__repr__().removeprefix("stabilizer_qec.")

    @property
    def num_measurements(self) -> int:
        if self._name == "HERALD_LEAKAGE_EVENT":
            return len(self._targets)
        if self._name in _LEAKAGE_GATES:
            return 0
        return super().num_measurements

    def target_groups(self) -> List[List[Any]]:
        if self._name in _LEAKAGE_GATES:
            return [[t] for t in self._targets]
        return super().target_groups()


class CircuitRepeatBlock(_stim.CircuitRepeatBlock):
    """A ``REPEAT`` block, its body a Deltakit circuit."""

    __slots__ = ()

    def __init__(self, repeat_count: int, body: Any, *, tag: str = "") -> None:
        super().__init__(repeat_count, Circuit(body), tag=tag)
        self._body = Circuit(self._body._stim_exact_text())

    def body_copy(self) -> Circuit:
        return Circuit(self._body._stim_exact_text())

    def __repr__(self) -> str:
        tag = f", tag={self._tag!r}" if self._tag else ""
        return f"stabilizer_qec.deltakit_stim.CircuitRepeatBlock({self._count}, {Circuit(self._body._stim_exact_text())!r}{tag})"

    __str__ = __repr__


# Deltakit's gates' data: its four instructions, and the transport gates' arguments (none or
# four) and noise.
_EXTRA = {
    "LEAKAGE": dict(args=(1, 2), is_noisy_gate=True, is_reset=False, produces_measurements=False, single=True),
    "RELAX": dict(args=(1, 2), is_noisy_gate=True, is_reset=False, produces_measurements=False, single=True),
    "RL": dict(args=(0, 1), is_noisy_gate=False, is_reset=True, produces_measurements=False, single=True),
    "HERALD_LEAKAGE_EVENT": dict(args=(0, 2), is_noisy_gate=True, is_reset=False, produces_measurements=True, single=True),
}


def _extra_gate(name: str) -> dict:
    e = _EXTRA[name]
    return {
        "name": name,
        "aliases": [name],
        "is_noisy_gate": e["is_noisy_gate"],
        "is_reset": e["is_reset"],
        "is_single_qubit_gate": True,
        "is_symmetric_gate": False,
        "is_two_qubit_gate": False,
        "is_unitary": False,
        "produces_measurements": e["produces_measurements"],
        "takes_measurement_record_targets": False,
        "takes_pauli_targets": False,
        "args": e["args"],
        "tableau": False,
        "unitary": None,
        "flows": None,
        "inverse": None,
        "generalized_inverse": name,
        "hadamard_conjugated": None,
        "hadamard_conjugated_unsigned": None,
        "flags": 1 << 15,
    }


class GateData(_stim.GateData):
    """What Deltakit's Stim knows about one instruction, as ``deltakit_stim.GateData``."""

    __slots__ = ()

    def __init__(self, name: str) -> None:
        key = str(name).upper()
        if key in _EXTRA:
            self._d = _extra_gate(key)
            return
        super().__init__(name)
        if self._d["name"] in _TRANSPORT_GATES:
            self._d = dict(self._d, args=(0, 5), is_noisy_gate=True)

    @property
    def generalized_inverse(self) -> "GateData":
        return GateData(self._d["generalized_inverse"])

    @property
    def inverse(self) -> Optional["GateData"]:
        inv = self._d["inverse"]
        return None if inv is None else GateData(inv)

    def __repr__(self) -> str:
        return f"stabilizer_qec.deltakit_stim.gate_data({self.name!r})"


def gate_data(name: Optional[str] = None) -> Any:
    """One gate's data, or (with no name) every gate's, by canonical name, Deltakit's
    included."""
    if name is None:
        _stim._load_gates()
        return {k: GateData(k) for k in sorted(list(_stim._GATES) + list(_EXTRA))}
    return GateData(name)


__all__ = ["Circuit", "CircuitInstruction", "CircuitRepeatBlock", "GateData", "gate_data"]

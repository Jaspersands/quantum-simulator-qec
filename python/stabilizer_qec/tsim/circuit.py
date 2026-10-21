# Ported from Tsim (QuEraComputing/tsim, bloqade-tsim 0.1.5, src/tsim/circuit.py; Apache License 2.0;
# see LICENSE in this directory), modified for stabilizer_qec: the circuit is a subclass of this
# package's Circuit holding this package's tagged operations (``I[T]``, rotations in radians), not a
# wrapper of stim.Circuit holding Tsim's (``S[T]``, half-turns); the methods this package's Circuit
# already has are its own, and the ZX-graph methods (get_graph, get_sampling_graph, tcount, to_matrix of
# a circuit with measurements, the pyzx diagrams) build Tsim's ZX graph with ``tsim.zx``.
"""Tsim's ``Circuit``: this package's circuit, read and written in Tsim's program text, with
Tsim's sampling conventions."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Union

import numpy as np

from stabilizer_qec import _circuit, _core, _stim
from stabilizer_qec._util import call, open_file
from stabilizer_qec.tsim._native import tsim_to_native
from stabilizer_qec.tsim.utils.program_text import (
    controlled_gate_decomposition_lines,
    enriched_stim_error,
    shorthand_to_stim,
    stim_to_shorthand,
)

_Base = _circuit.Circuit


def _is_stim_circuit(obj: Any) -> bool:
    return type(obj).__module__.split(".")[0] == "stim" and type(obj).__name__ == "Circuit"


def _single_angle(name: str, arg: Any) -> float:
    """Extract the single rotation angle required by a parametric gate."""
    if arg is None:
        raise ValueError(f"For {name} gates, an angle must be provided.")
    args = list(arg) if isinstance(arg, Iterable) else [arg]
    if len(args) != 1:
        raise ValueError(f"For {name} gates, a single angle must be provided.")
    return float(args[0])


def _two_distinct_qubits(name: str, targets: Any) -> tuple:
    """Validate and unpack the two distinct qubit targets of an R_XX/R_YY/R_ZZ gate."""
    qubits = list(targets) if isinstance(targets, Iterable) and not isinstance(targets, str) else [targets]
    if len(qubits) != 2:
        raise ValueError(f"For {name} gates, exactly two qubit targets are required.")
    q0, q1 = qubits
    if not all(isinstance(q, (int, np.integer)) and not isinstance(q, bool) for q in (q0, q1)):
        raise ValueError(f"For {name} gates, both targets must be qubit indices.")
    if q0 == q1:
        raise ValueError(f"For {name} gates, the two target qubits must be distinct, got {q0} {q1}.")
    return int(q0), int(q1)


def _bare_qubit_targets(gate_name: str, targets: Any) -> list:
    if isinstance(targets, (int, np.integer, str)) or hasattr(targets, "is_combiner") or type(targets).__name__ == "PauliString":
        items = [targets]
    else:
        items = list(targets)
    qubits = []
    for target in items:
        if isinstance(target, (int, np.integer)) and not isinstance(target, bool):
            qubits.append(int(target))
        elif hasattr(target, "is_qubit_target") and target.is_qubit_target:
            qubits.append(target.value)
        else:
            raise ValueError(f"{gate_name} only supports bare qubit targets.")
    return qubits


def _of_native_text(cls: type, text: str) -> "Circuit":
    return cls._of_native(text)


def _qubits_named(circuit: Any) -> set:
    """The qubits a circuit's instructions target (a loop's body read once)."""
    out: set = set()
    for item in circuit:
        if isinstance(item, _stim.CircuitRepeatBlock):
            out |= _qubits_named(item.body_copy())
            continue
        for t in item.targets_copy():
            q = t.qubit_value
            if q is not None:
                out.add(q)
    return out


def _lane_scan(circuit: Any) -> tuple:
    """``(reset_after_first, pair_identity_qubits)``: whether a reset follows the first instruction
    that names a qubit when that instruction is not a reset (an operation, noise, an identity or
    coordinates: Tsim's graph gives the qubit an input, where the unitary engine counts only an
    operation as one), and the qubits an ``II`` names (which that engine does not see). A loop's
    body is read twice (its second pass shows a reset after the first pass's operations)."""
    first_is_reset: dict = {}
    flags: dict = {}
    pair_identity: set = set()
    found = [False]

    def visit(items: list) -> None:
        for item in items:
            if item[0] == "repeat":
                for _ in range(min(item[1], 2)):
                    visit(item[3])
                continue
            name, targets = item[1], item[4]
            if name not in flags:
                flags[name] = bool(_stim.GateData(name).is_reset)
            reset = flags[name]
            for v in targets:
                q = _stim.GateTarget._raw(v).qubit_value
                if q is None:
                    continue
                if name == "II":
                    pair_identity.add(q)
                if q not in first_is_reset:
                    first_is_reset[q] = reset
                elif reset and not first_is_reset[q]:
                    found[0] = True

    visit(circuit._items())
    return found[0], pair_identity


class Circuit(_Base):
    """Tsim's circuit: a ``stabilizer_qec.Circuit`` (so ``isinstance(c, stabilizer_qec.Circuit)``
    and every method of this package's circuits) that reads and writes Tsim's program text.

    It is built from Tsim's shorthand or its tagged Stim (``T 0``, ``T_DAG 1``, ``TPP X0*Z1``,
    ``R_Z(0.25) 0`` in half-turns, ``R_ZZ(0.5) 0 1``, ``R_PAULI(0.3) X0*Y1``,
    ``U3(0.3, 0.2, 0.1) 0``, ``CCZ 0 1 2``, ``S[T] 0``, ``I[R_X(theta=0.25*pi)] 0``), from this
    package's form (``I[T] 0``, ``I[R_X(theta=0.39269908169872414)] 0`` in radians), or from a
    ``stim.Circuit`` or ``stabilizer_qec.Circuit`` in either; it holds this package's form, and
    ``str()`` and ``repr()`` give Tsim's shorthand (``T 0``, ``R_X(0.1) 0``). What Tsim's
    methods return is a ``tsim.Circuit`` here too (``copy``, slices, ``+``, ``*``,
    ``flattened``, ``without_noise``, ``inverse``, ...); ``stim_circuit`` is a plain
    ``stabilizer_qec.Circuit``.

    Its samplers are Tsim's: ``compile_detector_sampler`` reports each detector's and
    observable's value (the parity of its measurements), not its flip from a reference run as
    Stim's does (``use_detector_reference_sample`` / ``use_observable_reference_sample`` give
    the flips); ``detector_error_model`` allows non-deterministic observables. A
    ``stabilizer_qec.Circuit`` keeps Stim's conventions.

    >>> from stabilizer_qec import tsim
    >>> c = tsim.Circuit("R 0\\nH 0\\nT 0\\nH 0\\nM 0")
    >>> print(c)
    R 0
    H 0
    T 0
    H 0
    M 0
    """

    __slots__ = ()

    def __init__(self, stim_program_text: Union[str, Any] = "", *, text: Union[str, None] = None) -> None:
        if text is not None:  # this package's keyword
            stim_program_text = text
        if isinstance(stim_program_text, str):
            converted = shorthand_to_stim(stim_program_text)
            try:
                self._c = call(_core.Circuit, tsim_to_native(converted))
            except ValueError as exc:
                raise enriched_stim_error(exc, converted) from None
        elif isinstance(stim_program_text, Circuit):
            self._c = stim_program_text._c.copy()
        elif isinstance(stim_program_text, _Base) or _is_stim_circuit(stim_program_text):
            self._c = call(_core.Circuit, tsim_to_native(_circuit._stim_text(stim_program_text)))
        else:
            raise TypeError(f"expected a str, tsim.Circuit, stabilizer_qec.Circuit or stim.Circuit, not {type(stim_program_text).__name__}")

    @classmethod
    def _of_native(cls, text: str) -> "Circuit":
        """The circuit of this package's text, read as it is (no shorthand)."""
        out = cls.__new__(cls)
        out._c = call(_core.Circuit, text)
        return out

    @classmethod
    def from_stim_program(cls, stim_circuit: Any) -> "Circuit":
        """Create a Circuit from a stim.Circuit object (or a ``stabilizer_qec.Circuit``, or
        text), in Tsim's tagged form or this package's.

        Args:
            stim_circuit: The stim circuit to wrap.

        Returns:
            A new Circuit instance.

        """
        return cls(stim_circuit)

    @classmethod
    def from_file(cls, filename: Any = None, *, file: Any = None) -> "Circuit":
        """Create a Circuit from a file of Tsim's program text (shorthand or tagged Stim).

        Args:
            filename: The filename to read the circuit from (a path, or an open text file;
                ``file=`` is this package's keyword for it).

        Returns:
            A new Circuit instance.

        """
        source = filename if filename is not None else file
        if source is None:
            raise TypeError("from_file() missing 1 required argument: 'filename'")
        if hasattr(source, "read"):
            return cls(source.read())
        with open_file(source, encoding="utf-8") as f:
            return cls(f.read())

    def to_file(self, file: Any) -> None:
        """Write the circuit's Stim text (this package's tagged form, which Stim and
        ``stabilizer_qec.Circuit`` read, and ``tsim.Circuit.from_file`` too) to a path or an open
        text file. ``str()`` is Tsim's shorthand."""
        text = _Base.__str__(self) + "\n"
        if hasattr(file, "write"):
            file.write(text)
        else:
            with open_file(file, "w", encoding="utf-8") as f:
                f.write(text)

    def __str__(self) -> str:
        """Return the circuit as program text: Tsim's shorthand (``T 0``, ``R_X(0.1) 0``)."""
        return stim_to_shorthand(_Base.__str__(self))

    def __repr__(self) -> str:
        """Return a string representation that can recreate the circuit."""
        return f"tsim.Circuit('''\n{self!s}\n''')"

    def __reduce__(self) -> tuple:
        return (_of_native_text, (type(self), self._stim_exact_text()))

    def _transformed(self, which: str) -> "Circuit":
        return type(self)._of_native(call(_core.circuit_transform, self._stim_exact_text(), which))

    @property
    def stim_circuit(self) -> _Base:
        """Return the underlying stim circuit: a ``stabilizer_qec.Circuit`` (which runs the
        non-Clifford gates, as this package's circuits do).

        Parametric rotation instructions whose angles are all half-π multiples
        are expanded into their equivalent Clifford gates. ``REPEAT`` blocks are
        preserved structurally; their bodies are expanded recursively.
        """
        out = _Base.__new__(_Base)
        out._c = call(_core.Circuit, call(_core.circuit_transform, self._stim_exact_text(), "expand_clifford_rotations"))
        return out

    def cast_to_stim(self) -> "Circuit":
        """Return self, as Tsim's ``cast_to_stim`` (a cast for code that wants a
        ``stim.Circuit``; this circuit is this package's ``stim.Circuit``)."""
        return self

    # Appending: Tsim's gates as this package's operations.
    def append_from_stim_program_text(self, stim_program_text: Union[str, None] = None, *, text: Union[str, None] = None) -> None:
        """Append operations described by a Stim format program into the circuit.

        Supports the same shorthand syntax as the constructor.
        """
        if stim_program_text is None:
            stim_program_text = text
        if not isinstance(stim_program_text, str):
            raise TypeError(f"stim_program_text must be a str, not {type(stim_program_text).__name__}")
        converted = shorthand_to_stim(stim_program_text)
        try:
            call(self._c.append_text, tsim_to_native(converted))
        except ValueError as exc:
            raise enriched_stim_error(exc, converted) from None

    def _append_tsim(self, stim_name: str, tsim_tag: str, targets: Any, arg: Any) -> None:
        """Append ``stim_name[tsim_tag] targets`` as Tsim's tagged Stim, read as this package's
        operation: the targets checked and written by an untagged instruction first."""
        scratch = _Base()
        _Base.append(scratch, stim_name, targets, arg)
        line = scratch._stim_exact_text()
        if not line.startswith(stim_name):
            raise AssertionError(f"unexpected instruction text {line!r}")
        tagged = f"{stim_name}[{_stim._escape_tag(tsim_tag)}]{line[len(stim_name):]}" if tsim_tag else line
        call(self._c.append_text, tsim_to_native(tagged))

    def append(self, name: Any, targets: Any = (), arg: Any = None, *, tag: str = "") -> None:
        """Append an operation into the circuit, as Tsim's ``Circuit.append``.

        Args:
            name: The name of the operation's gate (e.g. "H" or "M" or "CNOT"), Tsim's
                gates among them: ``"T"``, ``"T_DAG"``, ``"TPP"``, ``"TPP_DAG"`` (Pauli-product
                targets), ``"R_X"``, ``"R_Y"``, ``"R_Z"`` (``arg`` the angle in half-turns),
                ``"R_XX"``, ``"R_YY"``, ``"R_ZZ"`` (two distinct qubits), ``"R_PAULI"`` (a
                Pauli product), ``"U3"`` (three angles in half-turns), ``"CCZ"``, ``"CCX"``
                (qubits in threes). Tsim's tagged Stim (``"S"`` with ``tag="T"``, ``"I"`` with
                ``tag="R_Z(theta=0.25*pi)"``) is read as those gates.

                This argument can also be set to a `stim.CircuitInstruction` or
                `stim.CircuitInstructionBlock`, which results in the instruction or
                block being appended to the circuit. The other arguments (targets
                and arg) can't be specified when doing so.

            targets: The objects operated on by the gate. This can be either a
                single target or an iterable of multiple targets.
            arg: The "parens arguments" for the gate, such as the probability for a
                noise operation, or a rotation's angle in half-turns.
            tag: A customizable string attached to the instruction (Tsim keeps it on T,
                T_DAG, TPP and TPP_DAG; this package does not, and Tsim drops it on rotations).

        """
        if not isinstance(name, str):
            if not (isinstance(targets, tuple) and targets == () and arg is None and tag == ""):
                raise ValueError("a circuit or Stim object is appended whole, without targets, arg or tag")
            if isinstance(name, Circuit):
                _Base.append(self, name)
                return
            native = tsim_to_native(_circuit._stim_text(name))
            if hasattr(name, "_stim_exact_text"):
                call(self._c.append_circuit, call(_core.Circuit, native, False))
            else:
                call(self._c.append_text, native)
            return
        if not isinstance(tag, str):
            raise TypeError(f"tag must be a str, not {type(tag).__name__}")
        if name in ("CCZ", "CCX"):
            if arg is not None:
                raise ValueError(f"For {name} gates, no arguments are accepted.")
            qubits = _bare_qubit_targets(name, targets)
            if len(qubits) % 3 != 0:
                raise ValueError(f"{name} expects qubit targets in groups of three.")
            self.append_from_stim_program_text("\n".join(line for i in range(0, len(qubits), 3) for line in controlled_gate_decomposition_lines(name, qubits[i], qubits[i + 1], qubits[i + 2], tag=tag)))
            return
        if name in ("T", "T_DAG", "TPP", "TPP_DAG"):
            stim_name = {"T": "S", "T_DAG": "S_DAG", "TPP": "SPP", "TPP_DAG": "SPP_DAG"}[name]
            self._append_tsim(stim_name, "T", targets, arg)
            return
        if name in ("R_X", "R_Y", "R_Z"):
            if arg is None:
                raise ValueError(f"For {name} gates, an angle must be provided.")
            args = list(arg) if isinstance(arg, Iterable) else [arg]
            if len(args) != 1:
                raise ValueError(f"For {name} gates, a single angle must be provided.")
            self._append_tsim("I", f"{name}(theta={float(args[0])!r}*pi)", targets, None)
            return
        if name == "U3":
            args = list(arg) if isinstance(arg, Iterable) else []
            if arg is None or len(args) != 3:
                raise ValueError("For U3 gates, three rotation angles must be provided.")
            theta, phi, lam = (float(a) for a in args)
            self._append_tsim("I", f"U3(theta={theta!r}*pi, phi={phi!r}*pi, lambda={lam!r}*pi)", targets, None)
            return
        if name in ("R_XX", "R_YY", "R_ZZ"):
            alpha = _single_angle(name, arg)
            pauli = name[2]
            q0, q1 = _two_distinct_qubits(name, targets)
            self._append_tsim("SPP", f"R_PAULI(theta={alpha!r}*pi)", [f"{pauli}{q0}", "*", f"{pauli}{q1}"], None)
            return
        if name == "R_PAULI":
            alpha = _single_angle(name, arg)
            self._append_tsim("SPP", f"R_PAULI(theta={alpha!r}*pi)", targets, None)
            return
        if tag and name in ("S", "S_DAG", "SPP", "SPP_DAG", "I"):
            # Tsim's tagged Stim: S[T] is T, I[R_Z(theta=0.25*pi)] a rotation, ...
            self._append_tsim(name, tag, targets, arg)
            return
        _Base.append(self, name, targets, arg, tag=tag)

    def insert(self, index: int, operation: Any) -> None:
        """Insert an instruction, REPEAT block or circuit before ``index`` (Tsim's tagged Stim
        read as Tsim's gates)."""
        if not isinstance(operation, Circuit):
            operation = type(self)._of_native(tsim_to_native(_circuit._stim_text(operation)))
        _Base.insert(self, index, operation)

    def __iadd__(self, other: object) -> "Circuit":
        """Append another circuit (a ``tsim.Circuit``, ``stabilizer_qec.Circuit`` or
        ``stim.Circuit``; Tsim's tagged Stim read as Tsim's gates) to this circuit in-place."""
        if isinstance(other, Circuit):
            return _Base.__iadd__(self, other)
        if isinstance(other, _Base) or _is_stim_circuit(other):
            self.append(other)
            return self
        return NotImplemented

    def __add__(self, other: object) -> "Circuit":
        """Return a new circuit with another circuit appended."""
        out = self.copy()
        if out.__iadd__(other) is NotImplemented:
            return NotImplemented
        return out

    def __radd__(self, other: object) -> "Circuit":
        """A ``stabilizer_qec.Circuit`` (or ``stim.Circuit``) followed by this circuit, as a
        ``tsim.Circuit``."""
        if isinstance(other, _Base) or _is_stim_circuit(other):
            out = type(self)(other)
            out += self
            return out
        return NotImplemented

    # Tsim's conventions for the model and the samplers.
    def detector_error_model(
        self,
        *,
        decompose_errors: bool = False,
        flatten_loops: bool = False,
        allow_gauge_detectors: bool = False,
        approximate_disjoint_errors: Any = False,
        ignore_decomposition_failures: bool = False,
        block_decomposition_from_introducing_remnant_edges: bool = False,
        allow_non_deterministic_observables: bool = True,
    ) -> "_circuit.DetectorErrorModel":
        """Return a stim.DetectorErrorModel describing the error processes in the circuit.

        Unlike the stim.Circuit.detector_error_model() method, this method allows for
        non-deterministic observables (an observable a non-Clifford gate leaves random): the
        observables become detectors, the model is built with ``allow_gauge_detectors``, and
        they become observables again, gauge errors on observables alone dropped
        (``tsim.get_detector_error_model``). ``decompose_errors=True`` is not supported with it
        and raises ``ValueError``, as Tsim's; ``allow_non_deterministic_observables=False``
        (this package's keyword) gives Stim's model. As Tsim's, the model is Stim's analysis
        of Tsim's tagged form of the circuit: T is read as ``S``, a rotation or ``U3`` as ``I``,
        a Pauli-product rotation as ``SPP`` (so ``R 0``, ``H 0``, ``T 0``, ``H 0``, ``M 0``
        measures a random bit).

        The other arguments are Stim's.
        """
        kwargs = dict(
            decompose_errors=decompose_errors,
            flatten_loops=flatten_loops,
            allow_gauge_detectors=allow_gauge_detectors,
            approximate_disjoint_errors=approximate_disjoint_errors,
            ignore_decomposition_failures=ignore_decomposition_failures,
            block_decomposition_from_introducing_remnant_edges=block_decomposition_from_introducing_remnant_edges,
        )
        from stabilizer_qec.tsim.noise.dem import get_detector_error_model

        return get_detector_error_model(self, allow_non_deterministic_observables=bool(allow_non_deterministic_observables), **kwargs)

    def compile_sampler(self, *, strategy: str = "cat5", seed: Union[int, None] = None, **kwargs: Any) -> Any:
        """Compile circuit into a measurement sampler (``tsim.CompiledMeasurementSampler``):
        each shot's measurement record.

        Args:
            strategy: Stabilizer rank decomposition strategy: "cat5", "bss" or "cutting".
                Accepted and unused: this package's stabilizer-rank engine has one
                decomposition (Tsim does not check it either).
            seed: Random seed for the sampler. The same seed gives the same shots here
                whatever the batch size.

        This package's keywords (``skip_reference_sample``, ``reference_sample``, ``device``)
        give its Stim sampler instead, ``stabilizer_qec.Circuit.compile_sampler``'s.
        """
        if kwargs:
            return _Base.compile_sampler(self, seed=seed, **kwargs)
        from stabilizer_qec.tsim.sampler import CompiledMeasurementSampler

        return CompiledMeasurementSampler(self, seed=seed, strategy=strategy)

    def compile_detector_sampler(self, *, strategy: str = "cat5", seed: Union[int, None] = None, **kwargs: Any) -> Any:
        """Compile circuit into a detector sampler (``tsim.CompiledDetectorSampler``): each
        detector's and observable's value, the parity of its measurements (not its flip from a
        reference run, which ``use_detector_reference_sample`` and
        ``use_observable_reference_sample`` give).

        Args:
            strategy: Stabilizer rank decomposition strategy: "cat5", "bss" or "cutting".
                Accepted and unused, as ``compile_sampler``'s.
            seed: Random seed for the sampler.

        This package's ``device`` keyword gives its Stim sampler instead,
        ``stabilizer_qec.Circuit.compile_detector_sampler``'s.
        """
        if kwargs:
            return _Base.compile_detector_sampler(self, seed=seed, **kwargs)
        from stabilizer_qec.tsim.sampler import CompiledDetectorSampler

        return CompiledDetectorSampler(self, seed=seed, strategy=strategy)

    def to_matrix(self, *, endian: str = "big") -> np.ndarray:
        """Convert circuit to matrix representation, as Tsim's ``to_matrix``: the map the
        circuit's gates apply to the qubits it names (a qubit no instruction names is dropped,
        so ``R_ZZ(0.5) 0 2`` is 4 × 4), qubit 0 the most significant bit of an index
        (``endian="little"``: the least). A qubit whose first operation is a reset (``R``,
        ``RX``, ``RY``) starts in that state and has no input: a circuit that resets every qubit
        first gives its state, a 2^n × 1 column (up to 22 qubits), and one that resets some a
        2^n × 2^k matrix. Noise channels are skipped (no error drawn), and detectors,
        observables, coordinates and ``TICK`` are nothing. At most 2^22 entries (a unitary of 11
        qubits).

        A circuit with a measurement, a reset after a qubit's first operation or feedback is
        Tsim's ZX tensor of it (``tsim.zx.tensorfy``), contracted in numpy with every measurement
        outcome and error bit read as 0: a measurement is the identity on its qubit times
        1/sqrt(2) (``H 0`` then ``M 0`` is [[.5, .5], [.5, -.5]]), a detector or observable
        projects its parity even, a reset projects (and prepares), ``MPP`` and heralded noise add
        an output qubit (the first, qubit -2 in Tsim's graph), feedback is the controlled gate.
        Its qubits are the ZX graph's: the inputs of the qubits first used by a gate, the outputs
        of every qubit."""
        if endian not in ("big", "little"):
            raise ValueError(f"endian must be 'big' or 'little', got {endian!r}")
        reset_after_first, pair_identity = _lane_scan(self)
        if reset_after_first:
            return self._zx_matrix(endian)
        ideal = self.without_noise()
        # A qubit only noise names (or an II) stays, as an input the circuit leaves alone (Tsim's identity).
        lanes = sorted((_qubits_named(self) - _qubits_named(ideal)) | pair_identity)
        if lanes:
            ideal = type(self)._of_native(f"I {' '.join(map(str, lanes))}\n{ideal._stim_exact_text()}")
        try:
            n, k, data = call(ideal._c.tsim_matrix, endian == "little")
        except ValueError as exc:
            if "has no matrix" not in str(exc):
                raise
            return self._zx_matrix(endian)
        return np.frombuffer(data, dtype="<c16").reshape(1 << n, 1 << k).copy()

    def _zx_matrix(self, endian: str) -> np.ndarray:
        from stabilizer_qec.tsim.zx import parse_stim_circuit, tensorfy

        g = parse_stim_circuit(self, backend="own").graph.copy()
        g.normalize()
        n_out, n_in = g.num_outputs(), g.num_inputs()
        if n_out + n_in > 22:
            raise ValueError(f"the circuit's ZX tensor has {n_out} output and {n_in} input qubits; to_matrix gives at most 2^22 entries")
        t = tensorfy(g)
        if endian == "little":
            t = t.transpose(list(range(n_out - 1, -1, -1)) + list(range(n_out + n_in - 1, n_out - 1, -1)))
        return np.ascontiguousarray(t).reshape(1 << n_out, 1 << n_in)

    def to_tensor(self, *, endian: str = "big") -> np.ndarray:
        """``to_matrix`` as a tensor: one index of size 2 per output qubit, then one per input
        qubit, in the order of ``endian``."""
        m = self.to_matrix(endian=endian)
        n, k = m.shape[0].bit_length() - 1, m.shape[1].bit_length() - 1
        return m.reshape((2,) * (n + k))

    def tcount(self) -> int:
        """Count the number of T gates in the circuit, as Tsim counts them: the vertices of the
        circuit's ZX graph with a non-Clifford phase (one for each ``T``, ``T_DAG`` and ``TPP``
        and each rotation by an angle that is not a multiple of 0.5 half-turns, three for a
        ``U3`` whose angles are none). A coherent error (``I_ERROR[R_Z(...)]``) is no gate, as in
        Tsim; ``stabilizer_qec.Circuit.tcount`` counts it."""
        from stabilizer_qec.tsim.zx import parse_stim_circuit
        from stabilizer_qec.tsim.zx import tcount as graph_tcount

        return graph_tcount(parse_stim_circuit(self, backend="own").graph)

    def get_graph(self, *, backend: Union[str, None] = None) -> Any:
        """Construct the ZX graph, as Tsim's ``get_graph``: one lane (a pair of boundary vertices
        joined by an edge) for each qubit the circuit names, its gates spiders on it, each
        measurement a Z spider with the phase variable ``rec[i]``, each noise channel's error bits
        spiders with phase variables ``e{k}``, each detector and observable an X spider with the
        phase variable ``det[i]`` or ``obs[i]`` joined to the measurements it XORs, and the scalar.

        Args:
            backend: ``"pyzx_param"`` for pyzx_param's ``GraphS`` (the graph Tsim returns; needs
                pyzx_param), ``"own"`` for this package's ``tsim.zx.ZXGraph`` (nothing to
                install); ``None`` (the default) is pyzx_param's if it is installed, else this
                package's. The two are equal vertex for vertex, edge for edge.

        Returns:
            The ZX graph.

        """
        from stabilizer_qec.tsim.zx import get_graph

        return get_graph(self, backend)

    def get_sampling_graph(self, sample_detectors: bool = False, *, backend: Union[str, None] = None) -> Any:
        """Get a ZX graph that can be used to compute probabilities.

        This graph will be constructed as follows:

        1. Double the ZX-diagram by composing it with its adjoint.
        2. Connect all rec[i] vertices to their corresponding adjoint rec[i] vertices.
        3. Add outputs:
        (a) When sampling measurements (i.e. `sample_detectors` is False),
            add output nodes for each measurement. Detectors and observables are
            removed since they are ignored when sampling measurements.
        (b) When sampling detectors and observables (i.e. `sample_detectors` is True),
            add output nodes for each detector and observable. Only one set of detector
            and observable nodes is kept, i.e., detectors and observables are not
            composed with their adjoints.

        Args:
            sample_detectors: If True, sample detectors and observables instead of
                measurements.
            backend: As ``get_graph``'s.

        Returns:
            A ZX graph for sampling.

        """
        from stabilizer_qec.tsim.zx import get_sampling_graph

        return get_sampling_graph(self, sample_detectors, backend)

    def diagram(
        self,
        type: str = "timeline-svg",
        tick: Union[int, range, None] = None,
        filter_coords: Any = ((),),
        rows: Union[int, None] = None,
        height: Union[float, None] = None,
        width: Union[float, None] = None,
        zoomable: bool = True,
        **kwargs: Any,
    ) -> Any:
        """Return a diagram of the circuit, from a variety of options.

        Args:
            type: The type of diagram: "timeline-svg" (the default) or "timeslice-svg", whose
                tagged gates are labelled (``T``, ``R_X(0.1π)``, ``TPP[XY]``, ...), or any of
                Stim's types (``"timeline-text"``, ``"detslice-svg"``, ...), as
                ``stabilizer_qec.Circuit.diagram``'s. Tsim's ZX-graph types ("pyzx",
                "pyzx-dets", "pyzx-meas") draw the circuit's ZX graph (``"pyzx"``: with bold
                classical wires) or the sampling graph for measurements or for detectors and
                observables after pyzx_param's ``full_reduce(paramSafe=True)`` (the reduced two
                need pyzx_param); they draw it with pyzx (or pyzx_param) and return the graph.
            tick: Required for time slice diagrams. Specifies which TICK instruction, or
                range of TICK instructions, to slice at.
            filter_coords: A list of things to include in the diagram (detslice diagrams).
            rows: In diagrams that have multiple separate pieces, how many rows of pieces.
            height: Optional height for the rendered diagram in pixels. Only applied
                to timeline-svg and timeslice-svg diagram types. For
                timeline-svg, when both height and width are None, the height is
                automatically determined based on the number of qubits. When only
                one dimension is given, the other is computed from the SVG aspect
                ratio.
            width: Optional width for the rendered diagram in pixels. Only applied
                to timeline-svg and timeslice-svg diagram types.
            zoomable: If True (default), wraps SVG diagrams in an interactive
                container with pan and Ctrl/Cmd+wheel zoom. Only applies to
                timeline-svg and timeslice-svg diagram types.
            **kwargs: Additional keyword arguments passed to the underlying diagram renderer.

        Returns:
            An object whose `__str__` method returns the diagram (for the SVG types, the raw
            SVG), and whose `_repr_html_` shows it in a notebook at the given size. For the ZX-graph
            types, the ZX graph that was drawn (``Circuit.get_graph``'s kind).

        """
        if type == "pyzx":
            from stabilizer_qec.tsim.zx.diagram import render_pyzx

            return render_pyzx(self, kwargs)
        if type in ("pyzx-dets", "pyzx-meas"):
            from stabilizer_qec.tsim.zx.diagram import render_reduced

            return render_reduced(self, type == "pyzx-dets", kwargs)
        if type in ("timeline-svg", "timeslice-svg"):
            from stabilizer_qec.tsim.utils.diagram import render_svg

            if height is None and width is None and type == "timeline-svg":
                height = min(30 * self.num_qubits + 50, 700)
            svg = str(_Base.diagram(self, type, tick=tick, rows=rows, filter_coords=filter_coords, **kwargs))
            return render_svg(svg, type, width=width, height=height, zoomable=zoomable)
        return _Base.diagram(self, type, tick=tick, rows=rows, filter_coords=filter_coords, **kwargs)

"""Circuits, detector error models, sampling and measurement conversion."""

from __future__ import annotations

from os import PathLike
from typing import Any, Union

import numpy as np

from . import _core, _dem, _stim
import bisect
import math
import warnings
from typing import Tuple
from ._util import b8_to_rows, call, count, open_file, pack_rows, probability, real, rows_to_b8, seed_of, stride, text_of


def _repetitions(obj: Any, repetitions: Any, op: str) -> int:
    """A repetition count as Stim's operators take one, an unsigned 64-bit integer: a negative
    or larger one is pybind11's TypeError, as from Stim (``(n - 1) * circuit`` for n = 0,
    say)."""
    n = int(repetitions)
    if n < 0 or n >= 1 << 64:
        kind = type(obj).__name__
        raise TypeError(f"{op}(): incompatible function arguments. The following argument types are supported:\n    1. (self: stabilizer_qec.{kind}, repetitions: int) -> stabilizer_qec.{kind}\n\nInvoked with: {obj!r}, {repetitions!r}")
    return count(n, "repetitions")


class Circuit:
    """A stabilizer circuit in Stim's circuit language.

    Build one from Stim text, a ``stim.Circuit``, or another ``Circuit``; ``str()`` gives
    Stim's text exactly as ``str(stim.Circuit(...))`` does (canonical names, neighbouring
    compatible instructions fused, arguments to 6 significant digits), while the arguments
    themselves are kept exactly. As in Stim, the circuit is also a list of instructions and
    ``REPEAT`` blocks: ``len``, indexing, slicing, ``insert``, ``pop`` and ``clear``. Every Clifford gate of Stim's is read (H, S and CX natively, the
    rest as their exact decompositions), with resets and measurements in all three bases,
    inverted targets (``!q``), Pauli-product measurements and rotations (``MPP``, ``MXX``,
    ``MYY``, ``MZZ``, ``SPP``, ``SPP_DAG``), the Pauli, depolarizing, correlated and heralded
    noise channels, ``MPAD``, measurement feedback and sweep bits (``CX rec[-1] q``,
    ``CZ sweep[k] q``), detectors, observables (of records and of Pauli targets, ``X3``),
    coordinates, ``TICK``, ``REPEAT``, and instruction tags (``H[tag] 0``).

    Circuits pickle and copy as their text, so they can be sent to other processes.

    >>> c = Circuit("R 0 1\\nH 0\\nCX 0 1\\nM 0 1\\nDETECTOR rec[-1] rec[-2]")
    >>> c.num_qubits, c.num_measurements, c.num_detectors
    (2, 2, 1)

    Or build one in code, as in Stim: ``append`` an instruction at a time, ``+`` one circuit
    after another, ``*`` a circuit into a ``REPEAT`` block. A piece may read measurements made
    before it (a round comparing with the last); sampling or analysing a circuit that reads
    before its first measurement is the error.

    >>> cycle = Circuit()
    >>> cycle.append("CX", [0, 2, 1, 2])
    >>> cycle.append("MR", [2], 0.01)
    >>> cycle.append("DETECTOR", ["rec[-1]", "rec[-2]"])
    >>> memory = Circuit("R 0 1 2\\nCX 0 2 1 2\\nMR 2") + cycle * 10
    >>> memory.num_measurements, memory.num_detectors
    (11, 10)
    """

    __slots__ = ("_c",)

    # Deltakit's Stim's error model (``stabilizer_qec.deltakit_stim``), not Stim's.
    _deltakit = False

    def __init__(self, stim_program_text: Union[str, "Circuit", Any] = "", *, text: Union[str, None] = None) -> None:
        if text is not None:  # 1.x's keyword
            stim_program_text = text
        self._c = call(_core.Circuit, text_of(stim_program_text, "Circuit", Circuit))

    @classmethod
    def from_file(cls, file: Union[str, PathLike, Any]) -> "Circuit":
        """Read a ``.stim`` file, from a path or an open file."""
        if hasattr(file, "read"):
            return cls(file.read())
        with open_file(file, encoding="utf-8") as f:
            return cls(f.read())

    def __str__(self) -> str:
        """Stim's text: canonical names, neighbouring instructions fused, arguments to 6
        significant digits (the values themselves are kept exactly)."""
        return _core.circuit_stim_text(self._c.__str__(), False)

    def _stim_exact_text(self) -> str:
        """The text with every argument exact, which parses back to an equal circuit."""
        return _core.circuit_exact_text(self._c.__str__(), False)

    def __repr__(self) -> str:
        text = str(self)
        if not text:
            return "stabilizer_qec.Circuit()"
        body = "\n".join("    " + line if line else line for line in text.split("\n"))
        return f"stabilizer_qec.Circuit('''\n{body}\n''')"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Circuit):
            return NotImplemented
        return self._stim_exact_text() == other._stim_exact_text()

    def __ne__(self, other: object) -> bool:
        r = self.__eq__(other)
        return r if r is NotImplemented else not r

    def approx_equals(self, other: object, *, atol: float) -> bool:
        """Equal up to arguments differing by at most ``atol``."""
        if not isinstance(other, Circuit):
            try:
                other = Circuit(other)
            except TypeError:
                return False
        return bool(_core.circuit_approx_equals(self._stim_exact_text(), other._stim_exact_text(), float(atol)))

    def __reduce__(self) -> tuple:
        # Pickled, copied and sent to other processes as its exact text.
        return (Circuit, (self._stim_exact_text(),))

    # Stim's list-of-instructions interface. The core circuit's lines are its items: Stim's
    # parser and ``append`` fuse neighbouring instructions that can be one (``H 0`` then
    # ``H 1`` is ``H 0 1``), and the core does too, while ``pop`` and slicing leave neighbours
    # apart, so the items are read, and edits written back, without fusing.
    def _items(self) -> list:
        return _core.circuit_items(self._c.__str__(), False)

    @classmethod
    def _of_items(cls, items: list) -> "Circuit":
        out = cls.__new__(cls)
        out._c = call(_core.Circuit, _stim._items_exact_text(items), False)
        return out

    def __len__(self) -> int:
        """How many instructions and REPEAT blocks are at the top level."""
        return len(self._items())

    def __iter__(self):
        for it in self._items():
            yield _stim._item_object(it)

    def __getitem__(self, index_or_slice: Any) -> Any:
        """The instruction or REPEAT block at an index, or a slice of them as a circuit."""
        items = self._items()
        if isinstance(index_or_slice, slice):
            return type(self)._of_items(items[index_or_slice])
        if isinstance(index_or_slice, bool) or not isinstance(index_or_slice, (int, np.integer)):
            raise TypeError(f"circuit indices must be integers or slices, not {type(index_or_slice).__name__}")
        k = int(index_or_slice)
        if k < 0:
            k += len(items)
        if not 0 <= k < len(items):
            raise IndexError(f"index {index_or_slice} is out of range for a circuit with {len(items)} items")
        return _stim._item_object(items[k])

    def _set_items(self, items: list) -> None:
        self._c = Circuit._of_items(items)._c

    def insert(self, index: int, operation: Any) -> None:
        """Insert an instruction, REPEAT block or circuit before ``index``."""
        items = self._items()
        k = int(index)
        if k < 0:
            k += len(items)
        if not 0 <= k <= len(items):
            raise IndexError(f"index {index} out of range")
        # As Stim's: what is inserted fuses with its neighbours either side where it can.
        piece = list(_core.circuit_items(_stim_text(operation), False))
        if piece:
            if k < len(items) and (both := _fused_item(piece[-1], items[k])) is not None:
                piece[-1] = both
                k_end = k + 1
            else:
                k_end = k
            if k > 0 and (both := _fused_item(items[k - 1], piece[0])) is not None:
                piece[0] = both
                k -= 1
            self._set_items(items[:k] + piece + items[k_end:])

    def pop(self, index: int = -1) -> Any:
        """Remove and return the instruction or REPEAT block at ``index``."""
        items = self._items()
        k = int(index)
        if k < 0:
            k += len(items)
        if not 0 <= k < len(items):
            raise IndexError(f"index {index} out of range")
        out = _stim._item_object(items[k])
        self._set_items(items[:k] + items[k + 1:])
        return out

    def clear(self) -> None:
        """Remove everything."""
        self._c = call(_core.Circuit, "")

    def _transformed(self, which: str) -> "Circuit":
        return type(self)(call(_core.circuit_transform, self._stim_exact_text(), which))

    def decomposed(self) -> "Circuit":
        """The circuit in (mostly) H, S, CX, M and R, as Stim's ``decomposed``."""
        return self._transformed("decomposed")

    def flattened(self) -> "Circuit":
        """Loops unrolled and coordinate shifts folded into the coordinates."""
        return self._transformed("flattened")

    def without_noise(self) -> "Circuit":
        """Noise removed: channels dropped, measurement flip probabilities dropped, heralded
        errors replaced by ``MPAD 0``."""
        return self._transformed("without_noise")

    def without_tags(self) -> "Circuit":
        """Every instruction's tag removed."""
        return self._transformed("without_tags")

    def without_leakage(self) -> "Circuit":
        """The circuit with no leakage: no leakage noise (``LEAKAGE``, ``RELAX``, and this
        package's ``LEAK``, ``SEEP`` and ``LEAK_TRANSPORT``), no herald, no detector reading a
        herald, and every other record lookback renumbered; an observable keeps its other
        records. Deltakit's deinstrumentation (``Measurements.to_detectors_and_observables``
        converts measurements on it), with loops kept: a loop whose lookbacks reach before it
        has its first iterations peeled off."""
        return self._transformed("without_leakage")

    def to_deltakit_text(self) -> str:
        """The circuit's text with its leakage in Deltakit's dialect (``LEAKAGE(p)``,
        ``RELAX(p)``, ``HERALD_LEAKAGE_EVENT(p)``), which Deltakit's Stim reads. A circuit
        reads that dialect too, as the tagged instructions Stim runs
        (``I_ERROR[LEAKAGE(p=…)]``, ``I_ERROR[RELAX(p=…)]``,
        ``MPAD[HERALD_LEAKAGE_EVENT(qubits=…)]``)."""
        return call(_core.circuit_to_deltakit, Circuit.__str__(self))

    def inverse(self) -> "Circuit":
        """The unitary inverse: operations inverted and in reverse, the tagged gates too
        (``I[T]`` ↔ ``I[T_DAG]``, a rotation's angle negated, ``U3(θ, φ, λ)``⁻¹ =
        ``U3(−θ, −λ, −φ)``). Noise, measurements and resets have no inverse and raise
        ``ValueError``."""
        return self._transformed("inverse")

    def with_inlined_feedback(self) -> "Circuit":
        """The circuit without measurement feedback (``CX rec[-1] q`` and the like): each
        detector and observable the feedback affected reads the controlling measurement
        instead."""
        return self._transformed("with_inlined_feedback")

    @property
    def num_ticks(self) -> int:
        """How many TICKs run (each loop iteration's counted)."""
        return call(_core.circuit_num_ticks, self._stim_exact_text())

    def flattened_operations(self) -> list:
        """Every instruction, loops unrolled, in Stim's older tuple form
        ``(name, targets, argument)``."""
        out = []

        def walk(items: list) -> None:
            for it in items:
                if it[0] == "repeat":
                    for _ in range(it[1]):
                        walk(it[3])
                    continue
                _, name, _tag, args, targets = it
                ts = []
                for v in targets:
                    g = _stim.GateTarget._raw(v)
                    q = v & ((1 << 24) - 1)
                    if g.is_inverted_result_target:
                        ts.append(("inv", q))
                    elif g.is_x_target:
                        ts.append(("X", q))
                    elif g.is_z_target:
                        ts.append(("Z", q))
                    elif g.is_y_target:
                        ts.append(("Y", q))
                    elif g.is_measurement_record_target:
                        ts.append(("rec", -q))
                    elif g.is_sweep_bit_target:
                        ts.append(("sweep", q))
                    else:
                        ts.append(q)
                arg = 0 if not args else args[0] if len(args) == 1 else list(args)
                out.append((name, ts, arg))

        walk(self._items())
        return out

    def get_final_qubit_coordinates(self) -> dict:
        """Each qubit's coordinates at the end of the circuit (after every SHIFT_COORDS)."""
        return dict(call(_core.circuit_final_qubit_coordinates, self._stim_exact_text()))

    def get_detector_coordinates(self, only: Any = None) -> dict:
        """The coordinates of every detector, or of the indices in ``only``."""
        if only is None:
            wanted = list(range(self.num_detectors))
        else:
            wanted = sorted({int(k) for k in only})
        return dict(call(_core.circuit_detector_coordinates, self._stim_exact_text(), wanted))

    def count_determined_measurements(self, *, unknown_input: bool = False) -> int:
        """How many measurements have outcomes fixed by earlier ones (and by the |0> start,
        unless ``unknown_input``)."""
        return call(_core.circuit_count_determined_measurements, self._stim_exact_text(), bool(unknown_input))

    def reference_detector_and_observable_signs(self, *, bit_packed: bool = False) -> tuple:
        """The noiseless parity of each detector's and each observable's measurement set."""
        dets, obs = call(_core.circuit_reference_signs, self._stim_exact_text(), self.num_observables)
        d = np.array(dets, dtype=np.bool_)
        o = np.array(obs, dtype=np.bool_)
        if bit_packed:
            return np.packbits(d, bitorder="little"), np.packbits(o, bitorder="little")
        return d, o

    def to_tableau(self, *, ignore_noise: bool = False, ignore_measurement: bool = False, ignore_reset: bool = False) -> "_stim.Tableau":
        """The tableau of the circuit's unitary part."""
        return _stim.Tableau.from_circuit(self, ignore_noise=ignore_noise, ignore_measurement=ignore_measurement, ignore_reset=ignore_reset)

    def to_matrix(self, *, endian: str = "big") -> np.ndarray:
        """The unitary the circuit applies, Clifford or not (``I[T]``, rotations, ``U3``), as a
        2^n × 2^n complex128 array, n one more than the largest qubit named (at most 10); as
        Tsim's ``to_matrix``, with qubit 0 the most significant bit of a basis index
        (``endian="little"`` makes it the least). Global phase as the gates give it (T is
        diag(1, e^{iπ/4})). A circuit with a measurement, reset or noise channel has no matrix
        (``ValueError``)."""
        if endian not in ("big", "little"):
            raise ValueError(f"endian must be 'big' or 'little', got {endian!r}")
        n, data = call(self._c.unitary_matrix, endian == "little")
        return np.frombuffer(data, dtype="<c16").reshape(1 << n, 1 << n).copy()

    def to_tensor(self, *, endian: str = "big") -> np.ndarray:
        """``to_matrix`` as a tensor of shape (2,) * 2n: n output indices then n input indices,
        in the order of ``endian``."""
        m = self.to_matrix(endian=endian)
        n = m.shape[0].bit_length() - 1
        return m.reshape((2,) * (2 * n))

    def tcount(self) -> int:
        """The circuit's T-count as Tsim (pyzx) counts it: its non-Clifford phases, one for
        each ``I[T]`` and ``I[T_DAG]`` gate and each rotation (a gate or a coherent error, each
        of a ``U3``'s three) by an angle that is not a multiple of π/2 in half-turns (of π/4 in
        this package's ``exp(−iθP)``), whatever the angle: ``I[R_Z(theta=0.1)] 0`` counts 1,
        ``I[R_Z(theta=0.7853981633974483)] 0`` (S) counts 0. Coherent errors count here and
        not in Tsim, which reads ``I_ERROR[R_Z(...)]`` as nothing: the count is of the phases
        the circuit's exact simulation pays for (``num_non_clifford_gates`` counts the gates
        alone)."""
        return call(self._c.t_count)

    @property
    def is_clifford(self) -> bool:
        """Whether every gate and coherent error of the circuit is Clifford, as Tsim's
        ``is_clifford``: no T gate, every rotation (a ``U3``'s three among them) by a multiple
        of π/2 in half-turns, of π/4 in this package's ``exp(−iθP)``. Noise channels, damping
        and leakage are not gates and do not count."""
        return call(self._c.is_clifford)

    @property
    def stim_circuit(self) -> "Circuit":
        """The circuit with its tagged gates at Clifford angles written as the Clifford gates
        they are (``I[R_Z(theta=0.7853981633974483)] 0`` as ``S 0``; a ``U3`` only when all
        three of its rotations are Clifford), ``REPEAT`` blocks kept; as Tsim's
        ``stim_circuit``, the form Stim itself runs correctly."""
        return self._transformed("expand_clifford_rotations")

    def cast_to_stim(self) -> "Circuit":
        """The circuit itself, as Tsim's ``cast_to_stim`` (a cast for code that wants a
        ``stim.Circuit``; this package's circuits are Stim's)."""
        return self

    def without_annotations(self) -> "Circuit":
        """A copy without its ``DETECTOR`` and ``OBSERVABLE_INCLUDE`` instructions, loops kept,
        as Tsim's ``without_annotations``."""

        def strip(circuit: "Circuit") -> "Circuit":
            out = type(self)()
            for item in circuit:
                if isinstance(item, _stim.CircuitRepeatBlock):
                    out.append(_stim.CircuitRepeatBlock(item.repeat_count, strip(item.body_copy()), tag=item.tag))
                elif item.name not in ("DETECTOR", "OBSERVABLE_INCLUDE"):
                    out.append(item)
            return out

        return strip(self)

    def to_file(self, file: Any) -> None:
        """Write the circuit's text to a path or an open text file."""
        text = str(self) + "\n"
        if hasattr(file, "write"):
            file.write(text)
        else:
            with open_file(file, "w", encoding="utf-8") as f:
                f.write(text)

    # Flows.
    def has_flow(self, flow: "_stim.Flow", *, unsigned: bool = False) -> bool:
        """Whether the circuit maps the flow's input stabilizer to its output (with its
        measurements), signs included unless ``unsigned``."""
        return call(_core.circuit_has_flows, self._stim_exact_text(), [str(_stim.Flow(flow))], bool(unsigned))[0]

    def has_all_flows(self, flows: Any, *, unsigned: bool = False) -> bool:
        """Whether the circuit has every one of the flows."""
        return all(call(_core.circuit_has_flows, self._stim_exact_text(), [str(_stim.Flow(f)) for f in flows], bool(unsigned)))

    def flow_generators(self) -> list:
        """Flows that generate every flow the circuit has, in Stim's canonical form."""
        return [_stim.Flow._of(t) for t in call(_core.circuit_flow_generators, self._stim_exact_text())]

    def solve_flow_measurements(self, flows: Any) -> list:
        """For each flow, the measurements that complete it into a flow of the circuit, or
        None where none do."""
        return list(call(_core.circuit_solve_flow_measurements, self._stim_exact_text(), [str(_stim.Flow(f)) for f in flows]))

    def time_reversed_for_flows(self, flows: Any, *, dont_turn_measurements_into_resets: bool = False) -> tuple:
        """The circuit run backwards, its detectors and observables re-expressed, with the
        given flows turned around."""
        text, fl = call(_core.circuit_time_reversed_for_flows, self._stim_exact_text(), [str(_stim.Flow(f)) for f in flows], bool(dont_turn_measurements_into_resets))
        return type(self)(text), [_stim.Flow._of(f) for f in fl]

    def missing_detectors(self, *, unknown_input: bool = False) -> "Circuit":
        """Detectors the circuit could declare but doesn't (independent of the declared
        detectors and observables)."""
        return type(self)(call(_core.circuit_missing_detectors, self._stim_exact_text(), bool(unknown_input)))

    def to_qasm(self, *, open_qasm_version: int, skip_dets_and_obs: bool = False) -> str:
        """The circuit as OpenQASM 2 or 3, as Stim's ``to_qasm`` writes it. Version 3 keeps
        detectors and observables (as registers computed from the measurement record against
        the noiseless reference sample) and feedback; version 2 has neither, and refuses them
        unless ``skip_dets_and_obs``. Noise is refused (see ``without_noise``).

        The custom one-qubit gates' ``U`` angles are those Stim's Linux and macOS builds write;
        its Windows build rounds a few at a ±pi boundary the other way (the same gates)."""
        return call(_core.circuit_to_qasm, self._stim_exact_text(), int(open_qasm_version), bool(skip_dets_and_obs))

    def to_quirk_url(self) -> str:
        """A URL that opens the circuit in Quirk (algassert.com/quirk), as Stim writes it."""
        return call(_core.circuit_to_quirk_url, self._stim_exact_text())

    def to_crumble_url(self, *, skip_detectors: bool = False, mark: Any = None) -> str:
        """A URL that opens the circuit in Crumble (algassert.com/crumble), as Stim writes it.
        ``mark`` maps a mark index to explained errors (``explain_detector_error_model_errors``),
        each drawn at its first location."""
        return _crumble_url(self, bool(skip_detectors), mark)

    def detecting_regions(self, *, targets: Any = None, ticks: Any = None, ignore_anticommutation_errors: bool = False) -> dict:
        """Where each detector and observable is sensitive to errors, tick by tick: a dict from
        ``DemTarget`` to a dict from tick to the ``PauliString`` of sensitivities (Stim's).

        ``targets`` filters what is reported: ``DemTarget``s, texts (``"D5"``, ``"L0"``, or
        ``"D"`` / ``"L"`` for all detectors / observables), or coordinate prefixes that pick the
        detectors whose coordinates start with them. ``ticks`` picks the ticks (default all)."""
        return _detecting_regions(self, targets, ticks, bool(ignore_anticommutation_errors))

    def shortest_error_sat_problem(self, *, format: str = "WDIMACS") -> str:
        """A max-SAT problem (WDIMACS) whose optimum is the fewest faults flipping an observable
        and no detector, as Stim writes it, for a solver to find the circuit distance."""
        if format != "WDIMACS":
            raise ValueError("Unsupported format.")
        return _wcnf(self._sat_dem(), False, 0)

    def likeliest_error_sat_problem(self, *, quantization: int = 100, format: str = "WDIMACS") -> str:
        """A weighted max-SAT problem (WDIMACS) whose optimum is the likeliest set of faults
        flipping an observable and no detector, weights quantized to ``quantization`` steps,
        as Stim writes it."""
        if format != "WDIMACS":
            raise ValueError("Unsupported format.")
        if int(quantization) < 1:
            raise ValueError("Must have quantization >= 1")
        return _wcnf(self._sat_dem(), True, int(quantization))

    def _sat_dem(self) -> "DetectorErrorModel":
        return self.detector_error_model(approximate_disjoint_errors=True)

    def append_operation(self, name: Any, targets: Any = (), arg: Any = None, *, tag: str = "") -> None:
        """Stim's older name for ``append``."""
        self.append(name, targets, arg, tag=tag)

    def copy(self) -> "Circuit":
        """A copy, to change without changing this one."""
        out = type(self).__new__(type(self))
        out._c = self._c.copy()
        return out

    def append(self, name: Union[str, "Circuit", Any], targets: Any = (), arg: Any = None, *, tag: str = "") -> None:
        """Append one instruction, as Stim's ``Circuit.append``: its name (any instruction of
        the language but ``REPEAT``), its targets, its argument or arguments (probabilities,
        coordinates, an observable's index), and optionally Stim's tag.

        A target is a qubit index, a string as Stim writes targets (``"!3"``, ``"rec[-1]"``,
        ``"sweep[0]"``, ``"X3"``, ``"*"`` joining Pauli targets into a product), or a
        ``stim.GateTarget``; ``targets`` is one of them or a list. ``name`` may instead be a
        ``Circuit``, or a ``stim.Circuit``, ``stim.CircuitInstruction`` or
        ``stim.CircuitRepeatBlock``, appended whole. An instruction the language does not allow
        raises ``ValueError`` and leaves the circuit as it was.

        >>> c = Circuit()
        >>> c.append("H", 0)
        >>> c.append("MPP", ["X0", "*", "Z1"], 0.01, tag="parity")
        >>> print(c)
        H 0
        MPP[parity](0.01) X0*Z1
        """
        if isinstance(name, Circuit) or not isinstance(name, str):
            if not (isinstance(targets, tuple) and targets == () and arg is None and tag == ""):
                raise ValueError("a circuit or Stim object is appended whole, without targets, arg or tag")
            if isinstance(name, Circuit):
                # A circuit appended to itself is appended as it was.
                call(self._c.append_circuit, name._c.copy() if name._c is self._c else name._c)
            elif hasattr(name, "_stim_exact_text"):
                # Ours: its lines are its items.
                call(self._c.append_circuit, call(_core.Circuit, _stim_text(name), False))
            else:
                call(self._c.append_text, _stim_text(name))
            return
        if not isinstance(tag, str):
            raise TypeError(f"tag must be a str, not {type(tag).__name__}")
        call(self._c.append_instruction, name, tag, _args(arg), _targets(targets))

    def append_from_stim_program_text(self, stim_program_text: Union[str, None] = None, *, text: Union[str, None] = None) -> None:
        """Append Stim's circuit text: any number of instructions and ``REPEAT`` blocks."""
        if stim_program_text is None:  # 1.x's keyword
            stim_program_text = text
        if not isinstance(stim_program_text, str):
            raise TypeError(f"stim_program_text must be a str, not {type(stim_program_text).__name__}")
        call(self._c.append_text, stim_program_text)

    def __add__(self, other: object) -> "Circuit":
        if not isinstance(other, Circuit):
            return NotImplemented
        out = self.copy()
        out += other
        return out

    def __iadd__(self, other: object) -> "Circuit":
        if not isinstance(other, Circuit):
            return NotImplemented
        call(self._c.append_circuit, other._c.copy() if other._c is self._c else other._c)
        return self

    def twirled(self, *, merge: bool = False) -> "Circuit":
        """The circuit with each coherent rotation (``I_ERROR[R_Z(theta=θ)] q`` and the rest)
        replaced by its Pauli twirl, the Pauli with probability sin²θ: the model Stim and every
        decoder assume. With ``merge``, rotations that are the same fault in different places
        (either side of a gate that leaves them alone, say) become one with their angles added,
        sin²(Σθ): the coherence-aware model. Merging writes the circuit without loops."""
        out = type(self).__new__(type(self))
        out._c = call(self._c.twirled, bool(merge))
        return out

    def __mul__(self, repetitions: object, op: str = "__mul__") -> "Circuit":
        """``REPEAT repetitions { self }``, as Stim's: nothing for 0, the circuit for 1."""
        if isinstance(repetitions, bool) or not isinstance(repetitions, (int, np.integer)):
            return NotImplemented
        out = type(self).__new__(type(self))
        out._c = call(self._c.repeated, _repetitions(self, repetitions, op))
        return out

    def __rmul__(self, repetitions: object) -> "Circuit":
        return self.__mul__(repetitions, "__rmul__")

    def __imul__(self, repetitions: object) -> "Circuit":
        out = self.__mul__(repetitions, "__imul__")
        if out is NotImplemented:
            return NotImplemented
        self._c = out._c
        return self

    __hash__ = None  # type: ignore[assignment]

    @property
    def num_qubits(self) -> int:
        """One more than the largest qubit index any instruction names."""
        return self._c.num_qubits

    @property
    def num_measurements(self) -> int:
        """Measurements per shot, counted through loops."""
        return self._c.num_measurements

    @property
    def num_detectors(self) -> int:
        return self._c.num_detectors

    @property
    def num_observables(self) -> int:
        return self._c.num_observables

    @property
    def num_sweep_bits(self) -> int:
        return self._c.num_sweep_bits

    def detector_error_model(
        self,
        *,
        decompose_errors: bool = False,
        approximate_disjoint_errors: Union[bool, float] = False,
        flatten_loops: bool = False,
        allow_gauge_detectors: bool = False,
        ignore_decomposition_failures: bool = False,
        block_decomposition_from_introducing_remnant_edges: bool = False,
        allow_non_deterministic_observables: bool = False,
    ) -> "DetectorErrorModel":
        """The circuit's detector error model, built by walking it backwards, as Stim's error
        analyzer does. ``decompose_errors`` splits each fault into graph-like pieces (at most
        two detectors each) the way Stim splits them, which matching needs.

        Loops that repeat are folded into ``repeat`` blocks, as Stim folds them, so a long
        memory's model costs one period of its loop; ``flatten_loops`` walks every pass and
        writes none (``DetectorErrorModel.flattened`` unrolls a folded one).

        ``approximate_disjoint_errors`` is Stim's: channels whose cases are disjoint rather
        than independent (``PAULI_CHANNEL_2``, ``ELSE_CORRELATED_ERROR``, the heralded errors,
        and a ``PAULI_CHANNEL_1`` with no independent equivalent) are refused unless it is set,
        and then approximated case by case; a number refuses any such channel with a
        probability above it.

        ``ignore_decomposition_failures`` is Stim's too: with ``decompose_errors``, a fault that
        cannot be split into graph-like pieces is kept whole rather than refused.

        ``allow_gauge_detectors`` (Stim's): a detector a reset or measurement makes random
        becomes a 50% error on it instead of an error; observables must still be deterministic.
        ``block_decomposition_from_introducing_remnant_edges`` (Stim's): split faults only into
        pieces other faults already make.

        ``allow_non_deterministic_observables`` (Tsim's): an observable a non-Clifford gate
        leaves random is allowed: the observables become detectors, the model is built with
        ``allow_gauge_detectors``, and they become observables again (a gauge error on
        observables alone is dropped). Not with ``decompose_errors``."""
        if allow_non_deterministic_observables:
            from .tsim.noise.dem import get_detector_error_model

            return get_detector_error_model(
                self,
                decompose_errors=decompose_errors,
                approximate_disjoint_errors=approximate_disjoint_errors,
                flatten_loops=flatten_loops,
                allow_gauge_detectors=allow_gauge_detectors,
                ignore_decomposition_failures=ignore_decomposition_failures,
                block_decomposition_from_introducing_remnant_edges=block_decomposition_from_introducing_remnant_edges,
            )
        if isinstance(approximate_disjoint_errors, bool):
            threshold = 1.0 if approximate_disjoint_errors else None
        else:
            threshold = real(approximate_disjoint_errors, "approximate_disjoint_errors")
        return DetectorErrorModel._wrap(call(self._c.detector_error_model, bool(decompose_errors), threshold, bool(flatten_loops), bool(ignore_decomposition_failures), bool(allow_gauge_detectors), bool(block_decomposition_from_introducing_remnant_edges), self._deltakit))

    def diagram(self, type: str = "timeline-text", *, tick: Union[int, range, None] = None, rows: Union[int, None] = None, filter_coords: Any = None) -> "Diagram":
        """A picture of the circuit, with Stim's ``diagram`` types:

        - ``"timeline-text"``: Stim's text timeline, character for character: every operation in
          its moment, ``TICK`` groups boxed, loops drawn once with records, detectors and
          coordinates in terms of ``iter``.
        - ``"timeline-svg"``: the same as a picture.
        - ``"timeline-3d"``: a 3D model (glTF) of the timeline, as Stim writes it.
        - ``"detslice-text"``: Stim's text picture of what each detector is sensitive to after
          ``tick`` ``TICK``s (a range: those ticks), character for character.
        - ``"detslice-svg"``: the same as a picture over the qubits' coordinates.
        - ``"timeslice-svg"``: the operations of tick ``tick`` (or of each tick in a
          ``range``, a panel each, in ``rows`` rows) over the qubits' coordinates.
        - ``"detslice-with-ops-svg"``: the same, with the detector slice after them.
        - ``"matchgraph-svg"``, ``"matchgraph-3d"``: the error model's matching graph.
        - ``"interactive"``: a page opening the circuit in Crumble, Stim's circuit editor.

        Each picture type has an ``-html`` form (a viewer page). ``filter_coords`` picks the
        detectors and observables of slice diagrams: ``DemTarget``s, texts (``"D5"``,
        ``"L0"``), or coordinate prefixes; by default every detector and no observable, as in
        Stim. The result prints as its text and shows as a picture in a notebook.

        >>> c = Circuit("H 0\\nCNOT 0 1\\nTICK\\nM 0 1\\nDETECTOR rec[-1] rec[-2]")
        >>> print(c.diagram("detslice-text", tick=1))
        q0: -Z:D0-
             |
        q1: -Z:D0-
        """
        if not isinstance(type, str):
            raise TypeError(f"type must be a str, not {builtins_type(type).__name__}")
        try:
            filters = _filters(filter_coords)
        except Exception:
            raise ValueError("filter_coords wasn't an Iterable[stim.DemTarget | Iterable[float]].") from None
        num_rows = None if rows is None or rows == 0 else count(rows, "rows")
        if tick is None:
            t0, nt = 0, (1 << 64) - 1
        elif isinstance(tick, range):
            if tick.step != 1:
                raise ValueError("tick.step != 1")
            if tick.stop <= tick.start:
                raise ValueError("tick.stop <= tick.start")
            t0, nt = count(tick.start, "tick"), tick.stop - tick.start
        else:
            t0, nt = count(tick, "tick"), 1
        html = "html" in type
        kind = _DIAGRAM_KIND.get(type)
        if kind is None:
            raise ValueError(f"Unrecognized diagram type: '{type}'")
        text = self._stim_exact_text()
        if kind == "timeline-text":
            if tick is not None:
                raise ValueError("`tick` isn't used with type='timeline-text'")
            return Diagram(call(self._c.diagram, "timeline-text"), type, "text")
        if kind == "detslice-text":
            return Diagram(call(_core.circuit_detslice_text, text, t0, nt, filters), type, "text")
        if kind == "timeline-3d":
            return _gltf_diagram(call(_core.circuit_timeline_3d, text), type, html)
        if kind == "interactive":
            return Diagram(_crumble_page(self.to_crumble_url()), type, "html")
        if kind == "matchgraph":
            try:
                dem = self.detector_error_model(decompose_errors=True, approximate_disjoint_errors=True)
            except ValueError:
                dem = self.detector_error_model(approximate_disjoint_errors=True)
            return dem.diagram(type)
        if kind == "timeline-svg":
            svg = call(self._c.diagram, "timeline-svg")
        else:
            end = None
            if tick is None:
                t0, end = 0, self.num_ticks + 1
            elif nt > 1:
                end = t0 + nt
            if kind == "detslice-svg" and end is None and num_rows is None:
                svg = call(self._c.diagram, "detslice-svg", t0, None, None, filters)
            else:
                svg = call(self._c.diagram, kind, t0, t0 + 1 if end is None else end, num_rows, filters)
        return Diagram(svg, type, "svg-html" if html else "svg")

    @classmethod
    def generated(
        cls,
        code_task: str,
        *,
        distance: int,
        rounds: int,
        after_clifford_depolarization: float = 0.0,
        before_round_data_depolarization: float = 0.0,
        before_measure_flip_probability: float = 0.0,
        after_reset_flip_probability: float = 0.0,
        annotate_colors: bool = False,
        after_reset_leakage: float = 0.0,
        after_clifford_leakage_and_relaxation: float = 0.0,
    ) -> "Circuit":
        """One of Stim's generated memory experiments, character for character as
        ``stim.Circuit.generated`` writes it, with the same arguments. ``code_task`` is
        ``"repetition_code:memory"``, ``"surface_code:rotated_memory_x"`` (or ``_z``),
        ``"surface_code:unrotated_memory_x"`` (or ``_z``), or ``"color_code:memory_xyz"``.

        ``annotate_colors`` (the colour code only) flattens the circuit and gives each
        detector ``(x, y, t)`` a 4th coordinate, ``(y + t) mod 3``: Chromobius's colour and
        basis annotation, as its authors annotate these circuits. Chromobius does not decode
        them under circuit noise (its tests leave them out), and neither does
        ``ColorMatching``; it does decode ``CssCode.color_code`` memories.

        ``after_reset_leakage`` and ``after_clifford_leakage_and_relaxation`` are Deltakit's
        Stim's: ``LEAKAGE(p)`` after each reset (``MR`` included), and after each two-qubit
        gate's noise, with ``RELAX(p)`` after each single-qubit gate's.

        >>> c = Circuit.generated("surface_code:rotated_memory_z", distance=3, rounds=5,
        ...                       after_clifford_depolarization=0.001)
        >>> c.num_qubits, c.num_detectors
        (26, 40)
        """
        if not isinstance(code_task, str):
            raise TypeError(f"code_task must be a str, not {builtins_type(code_task).__name__}")
        text = call(
            _core.generated_circuit,
            code_task,
            count(distance, "distance"),
            count(rounds, "rounds"),
            probability(after_clifford_depolarization, "after_clifford_depolarization"),
            probability(before_round_data_depolarization, "before_round_data_depolarization"),
            probability(before_measure_flip_probability, "before_measure_flip_probability"),
            probability(after_reset_flip_probability, "after_reset_flip_probability"),
            bool(annotate_colors),
            probability(after_reset_leakage, "after_reset_leakage"),
            probability(after_clifford_leakage_and_relaxation, "after_clifford_leakage_and_relaxation"),
        )
        return cls(text)

    def explain_detector_error_model_errors(
        self, *, dem_filter: Union["DetectorErrorModel", str, Any, None] = None, reduce_to_one_representative_error: bool = False
    ) -> list:
        """Where the faults of the circuit's error model arise, as Stim's method of the same name
        explains them: for each fault class (each fault of ``dem_filter`` when given), every
        place in the circuit such a fault arises, or with ``reduce_to_one_representative_error``
        one. Loops are walked in full. Each result prints as Stim's text.

        >>> c = Circuit("R 0\\nX_ERROR(0.1) 0\\nM 0\\nDETECTOR rec[-1]")
        >>> e = c.explain_detector_error_model_errors()[0]
        >>> [str(t.dem_target) for t in e.dem_error_terms], e.circuit_error_locations[0].instruction_targets.gate
        (['D0'], 'X_ERROR')
        """
        from . import _explain

        f = None if dem_filter is None else (dem_filter if isinstance(dem_filter, DetectorErrorModel) else DetectorErrorModel(dem_filter))
        raw = call(self._c.explain, None if f is None else f._d, bool(reduce_to_one_representative_error))
        out = _explain.build(raw)
        if f is not None:
            # As Stim: the filter's errors in order of their symptoms (detectors before
            # observables, each by index), not the filter's order.
            out.sort(key=lambda e: [(t.dem_target.is_logical_observable_id(), t.dem_target.val) for t in e.dem_error_terms])
        return out

    def shortest_graphlike_error(self, *, ignore_ungraphlike_errors: bool = True, canonicalize_circuit_errors: bool = False) -> list:
        """The circuit's graph-like distance, as Stim's method of the same name finds it: the
        fewest graph-like pieces of its decomposed error model that together flip an
        observable and no detector, each explained (by one location with
        ``canonicalize_circuit_errors``). Its length is the distance.

        >>> c = Circuit.generated("surface_code:rotated_memory_z", distance=3, rounds=3, after_clifford_depolarization=0.001)
        >>> len(c.shortest_graphlike_error())
        3
        """
        from . import _explain

        return _explain.build(call(self._c.shortest_graphlike, bool(ignore_ungraphlike_errors), bool(canonicalize_circuit_errors)))

    def search_for_undetectable_logical_errors(
        self,
        *,
        dont_explore_detection_event_sets_with_size_above: int,
        dont_explore_edges_with_degree_above: int,
        dont_explore_edges_increasing_symptom_degree: bool,
        canonicalize_circuit_errors: bool = False,
    ) -> list:
        """The circuit's distance through hyperedges too, as Stim's method of the same name
        finds it: a breadth-first search over sets of fired detectors in its (undecomposed)
        error model, a fault added at the set's lowest detector each step, within the limits
        given. The faults found, each explained (by one location with
        ``canonicalize_circuit_errors``); their number bounds the distance from above."""
        from . import _explain

        raw = call(
            self._c.search_undetectable,
            count(dont_explore_detection_event_sets_with_size_above, "dont_explore_detection_event_sets_with_size_above"),
            count(dont_explore_edges_with_degree_above, "dont_explore_edges_with_degree_above"),
            bool(dont_explore_edges_increasing_symptom_degree),
            bool(canonicalize_circuit_errors),
        )
        return _explain.build(raw)

    def reference_sample(self, *, bit_packed: bool = False) -> np.ndarray:
        """A noiseless run's measurement record, as Stim's: the reference each shot's flips are
        taken from (and the converter compares with). Random outcomes take fixed values."""
        n = self.num_measurements
        return b8_to_rows(call(self._c.reference_sample), 1, n, bit_packed)[0]

    def compile_sampler(self, *, skip_reference_sample: bool = False, seed: Union[int, None] = None, reference_sample: Any = None, device: str = "cpu") -> "MeasurementSampler":
        """A sampler of raw measurement records, as Stim's: a noiseless reference run with each
        shot's flips (with ``skip_reference_sample``, the flips alone; with
        ``reference_sample``, a bool or bit-packed array, the flips on top of that record). The
        same seed gives the same shots on any machine and number of threads; ``None`` draws a
        seed. ``device`` as ``compile_detector_sampler``'s."""
        return MeasurementSampler(self, seed, skip_reference_sample, reference_sample, device=device)

    def compile_detector_sampler(self, *, seed: Union[int, None] = None, device: str = "cpu") -> "DetectorSampler":
        """A sampler of detection events and observable flips. The same seed gives the same
        shots on any machine and any number of threads; ``None`` draws a seed. A circuit with
        non-Clifford gates (``I[T]``, ``I[R_Z(theta=…)]``, ``I[U3(…)]``) gets the
        stabilizer-rank sampler, which runs them (``without_tags()`` for Stim's reading, the
        identity). A tagged gate at a Clifford angle (``I[R_Z(theta=1.5707963267948966)] 0``
        is ``Z 0``) is the Clifford gate it is in every sampler. Coherent *errors*
        (``I_ERROR[R_Z(theta=…)]``) are beyond a frame: the frame sampler reads them as Stim
        does, as nothing, and warns; ``twirled()`` gives the Pauli-twirled model and
        ``compile_coherent_sampler`` the coherent one (and the stabilizer-rank sampler, when
        the circuit has non-Clifford gates too, runs them exactly).

        ``device``: ``"cpu"`` (the default: 64 shots a machine word, noise by geometric
        skipping), ``"cpu-lanes"`` (32 shots a word, every random number an integer compare:
        the program a GPU runs, on the CPU — the same seed gives the same bytes on either) or
        ``"gpu"`` (the same program on a GPU through wgpu, in a build with the ``gpu``
        feature). The devices sample the same distribution; ``"cpu"`` and the lane devices draw
        different shots from the same seed. The GPU device is opened once per process and is
        not inherited by a forked child (use ``multiprocessing``'s spawn start method)."""
        if self.num_non_clifford_gates():
            if device != "cpu":
                raise ValueError(f"a circuit with non-Clifford gates samples on the stabilizer-rank sampler, device 'cpu' only (not {device!r})")
            return StabilizerRankSampler(self, seed)
        return DetectorSampler(self, seed, device=device)

    def compile_stabilizer_rank_sampler(self, *, seed: Union[int, None] = None) -> "StabilizerRankSampler":
        """A sampler for Clifford+T circuits at any qubit count, exact, its cost growing with the
        number of non-Clifford gates (``I[T] 0``, ``I[T_DAG] 0``, ``I[R_Z(theta=0.3)] 0``,
        ``II[R_XX(theta=0.3)] 0 1``, ``I[R_PAULI(theta=0.3, pauli=XYZ)] 0 1 2``,
        ``I[U3(theta=…, phi=…, lambda=…)] 0``) rather than of qubits: the state is a sum of
        Clifford branches, one per coset of the reference run's stabilizer group that the
        non-Clifford gates reach, at most 2^(T count) of them and far fewer on code states
        (seven T gates on the Steane code's |0⟩ are eight terms). Pauli noise, measurement
        flips, heralds and feedback run as the frame sampler's; amplitude damping and leakage
        are refused (the state vector, the density matrix and the leakage sampler run them).
        Its ``sample`` has the detector sampler's signature; ``sample_measurements`` gives
        records, ``probability_of`` a record's exact probability.

        >>> import stabilizer_qec as sq
        >>> c = sq.Circuit("R 0\\nH 0\\nI[T] 0\\nH 0\\nM 0\\nDETECTOR rec[-1]")
        >>> p = c.exact_distribution(method="stabilizer_rank")[((True,), ())]
        >>> round(p, 6)  # sin²(π/8)
        0.146447
        """
        return StabilizerRankSampler(self, seed)

    def num_non_clifford_gates(self) -> int:
        """How many of the circuit's gates are not Clifford: its ``I[T]``/``I[T_DAG]`` gates,
        and its tagged rotations and ``U3`` half-angle rotations written as gates (on ``I``,
        ``II``) at angles that are not multiples of π/4 (``I[R_Z(theta=0.7853981633974483)]``
        is S, and counts 0). Coherent errors (``I_ERROR``, ``II_ERROR``) are not gates and do
        not count."""
        return call(self._c.num_non_clifford_gates)

    def compile_exact_sampler(self, *, seed: Union[int, None] = None) -> "ExactSampler":
        """An exact sampler: the full state vector (up to 24 qubits in use), running every
        instruction, Clifford or not, including the tagged rotations, T, U3 and amplitude damping
        (``I_ERROR[R_Z(theta=0.01)] 0``, ``I[T] 0``). Noise channels are drawn as quantum
        trajectories. Its ``sample`` has the detector sampler's signature."""
        return ExactSampler(self, seed)

    def compile_coherent_sampler(self, *, order: int = 3, seed: Union[int, None] = None, pairs: bool = False, max_complete: int = 50_000) -> "CoherentSampler":
        """A sampler for circuits with coherent errors (``I_ERROR[R_Z(theta=0.02)] 0``,
        ``II_ERROR[R_ZZ(theta=0.01)] 0 1``, any Pauli rotation) at any size: shots of the
        Pauli-twirled circuit, as the detector sampler draws them, each with a weight that puts
        back the interference the twirl leaves out. Weighted, the shots are distributed as the
        coherent circuit's: estimate a rate as ``(w * hit).sum() / w.sum()``.

        The interference runs between fault sets no record tells apart, a linear space (the
        circuit's spacetime kernel). Its generators are found locally (a fault either side of a
        gate, a stabiliser's worth of faults) and then completed to a basis of the whole kernel
        by one elimination over the rotations' effects: the completion holds what no local
        search finds, above all a logical error at one time with the same logical at another,
        whose amplitudes add (coherent logical errors accumulate across rounds).
        ``num_nonlocal_generators`` counts them. With at most twelve generators (besides groups
        of rotations that are one fault in several places) every shot sums the whole kernel and
        the weights are exact (``exact_weights``); past that a shot sums the products of up to
        ``order`` generators its faults touch, a truncation. ``max_complete`` is the most
        rotations whose kernel is completed; past it the generators are the local ones only,
        interference between distant faults is left out, ``kernel_complete`` is False and the
        sampler warns.

        A circuit with non-Clifford *gates* too (``I[T] 0``, ``I[R_Z(theta=…)] 0``,
        ``I[U3(…)] 0``: Clifford+T circuits, magic-state injection, a cultivation check) gets
        the joint sampler: the gates run exactly, as the stabilizer-rank sampler runs them,
        the errors are drawn as their twirl, and each shot's weight is the coherent probability
        of its class over the twirl's, every member of the class a re-run of the exact engine
        with the member's errors inserted and the shot's outcomes forced. Neither the coherent
        sampler (a Clifford reference circuit) nor a stabilizer-rank sampler alone (cost
        exponential in the errors) covers this. ``pairs`` has the joint sampler sum the
        products of every two of its generators besides (quadratic in their number: a check
        on small circuits; the coherent sampler of a Clifford circuit ignores it). A coherent error at a Clifford angle
        (``I_ERROR[R_Z(theta=0.7853981633974483)] 0``, a fixed S) is run as the gate it is, not
        drawn as a twirl: such a circuit gets the joint sampler too."""
        return CoherentSampler(self, order, seed, pairs=pairs, max_complete=max_complete)

    def compile_leakage_sampler(self, *, leaked_reads_one: bool = True, seed: Union[int, None] = None) -> "LeakageSampler":
        """A frame sampler that runs leakage, written as Stim-readable tags:
        ``I_ERROR[LEAK(p=…)] q`` (the qubit leaks), ``I_ERROR[SEEP(p=…)] q`` (a leaked qubit
        returns, to |0⟩ or |1⟩ at random) and ``II_ERROR[LEAK_TRANSPORT(p=…)] a b`` (leakage
        moves to the partner). A two-qubit gate with a leaked partner leaves the other qubit a
        uniformly random Pauli, as in Google's model; measuring a leaked qubit reads 1 (a coin
        flip with ``leaked_reads_one=False``); a reset returns it. Its ``sample`` also gives each
        measurement's herald: whether its qubit was leaked."""
        return LeakageSampler(self, leaked_reads_one, seed)

    def exact_distribution(self, *, max_branches: int = 1 << 20, method: str = "state_vector") -> dict:
        """Every outcome's exact probability: ``{(detection events, observable flips):
        probability}``, each a tuple of bools. ``method="state_vector"`` (up to 24 qubits)
        follows every measurement outcome and noise branch; ``"density_matrix"`` (up to 12)
        applies each noise channel whole and branches only on what measurements record, far
        fewer branches on a noisy circuit; ``"stabilizer_rank"`` (any number of qubits, the
        cost growing with the non-Clifford gates) follows the branches as the state vector does
        with the stabilizer-rank sampler's engine; ``"coherent"`` sums the joint coherent
        sampler's weighted twirl over every branch with the whole interference kernel (the
        proof of its identity: it equals the state vector's). For small circuits; more than
        ``max_branches`` branches is an error."""
        runs = {"state_vector": self._c.exact_distribution, "density_matrix": self._c.density_distribution, "stabilizer_rank": self._c.stabilizer_rank_distribution, "coherent": self._c.joint_distribution}
        if method not in runs:
            raise ValueError(f"method must be 'state_vector', 'density_matrix', 'stabilizer_rank' or 'coherent', got {method!r}")
        run = runs[method]
        rows = call(run, count(max_branches, "max_branches"))
        no = self.num_observables
        return {(tuple(d), tuple(bool(o >> k & 1) for k in range(no))): p for d, o, p in rows}

    def perturbative_distribution(self, order: int = 1, *, symbolic: bool = False, degree: Any = None, max_terms: int = 1 << 22) -> dict:
        """The outcome distribution summed over every set of at most ``order`` faults, as
        QuantumClifford.jl's ``petrajectories``: ``{(detection events, observable flips):
        weight}``, keyed as ``exact_distribution``. Each noise channel (a Pauli channel on a
        qubit or pair, a measurement's flip, a herald, a chain of correlated errors) is a site
        that makes none of its faults or one; a fault set's weight is its exact probability, its
        faults' probabilities times every other site's probability of none. The weights are
        lower bounds, short of the exact distribution by O(p^(order+1)), and equal to it once
        ``order`` reaches the number of sites. A fault counts toward the order whether or not it
        changes an outcome, as in QuantumClifford.

        With ``symbolic``, each weight is a ``numpy.polynomial.Polynomial`` in a noise scale x,
        every fault's probability p read as p·x (a correlated chain's faults too, each with its
        probability of happening), cut at x^``degree`` (``order`` by default; ``"full"``
        keeps every power, as QuantumClifford's symbolic expansion): up to x^``order`` its
        coefficients are the exact Taylor coefficients of the outcome's probability. The
        polynomials are in the monomial basis, so with many sites at noise that is not small a
        full expansion's coefficients grow and alternate, and evaluating it far from x = 0
        cancels away its digits (the numeric expansion at full order has no such loss).

        For Clifford circuits with Pauli noise whose detectors and observables are
        deterministic, of any size: each fault's effect is found once, by a Pauli frame per fault
        carried through the circuit together, and the sum is a dynamic program over outcomes
        (at most ``max_terms`` outcome and fault-count pairs at once).

        >>> import stabilizer_qec as sq
        >>> c = sq.Circuit("R 0 1\\nX_ERROR(0.1) 0 1\\nCX 0 1\\nM 1\\nDETECTOR rec[-1]")
        >>> d = c.perturbative_distribution(order=1)
        >>> round(d[((True,), ())], 6)  # one X on either qubit: 2 * 0.1 * 0.9
        0.18
        >>> d = c.perturbative_distribution(order=2, symbolic=True)
        >>> [round(float(a), 12) for a in d[((True,), ())].coef]  # 2 (0.1 x) (1 - 0.1 x)
        [0.0, 0.2, -0.02]
        """
        if degree is not None and degree != "full" and not (isinstance(degree, (int, np.integer)) and not isinstance(degree, bool) and degree >= 0):
            raise ValueError(f"degree must be a non-negative integer or 'full', got {degree!r}")
        if not symbolic and degree is not None:
            raise ValueError("degree is for a symbolic expansion (symbolic=True)")
        k = count(order, "order")
        # A degree past the number of sites is "full" (the engine clamps it so).
        deg = None if not symbolic else (k if degree is None else (1 << 62) if degree == "full" else min(int(degree), 1 << 62))
        nd, rows = call(self._c.perturbative_distribution, k, deg, count(max_terms, "max_terms"))
        no = self.num_observables
        out = {}
        for flipped, w in rows:
            hit = set(flipped)
            key = (tuple(i in hit for i in range(nd)), tuple(nd + i in hit for i in range(no)))
            out[key] = np.polynomial.Polynomial(np.trim_zeros(np.asarray(w), "b") if np.any(w) else [0.0]) if symbolic else w[0]
        return out

    def density_matrix(self, *, endian: str = "little", max_branches: int = 1 << 20) -> np.ndarray:
        """The density matrix the circuit leaves its qubits in (up to 12 qubits), from all of
        them in |0⟩: noise as channels, measurement outcomes averaged away (followed one by one,
        at most ``max_branches``, only when feedback reads them). Runs everything the exact
        sampler runs, amplitude and phase damping and ``THERMAL_RELAXATION`` included. Qubit 0
        is the least significant bit of an index for ``endian="little"``."""
        from ._density import _matrix

        n, data = call(self._c.density_matrix, count(max_branches, "max_branches"))
        return _matrix(data, n, _stim._endian(endian))

    def process_matrix(self, kind: str = "choi", *, max_branches: int = 1 << 20) -> np.ndarray:
        """The circuit's process on its qubits (up to 6), measurement outcomes averaged away:
        ``"choi"`` (Σ_ij |i⟩⟨j| ⊗ Λ(|i⟩⟨j|)), ``"superoperator"``, ``"ptm"`` (Pauli transfer
        matrix) or ``"chi"``; the conventions are ``stabilizer_qec.process_matrix``'s."""
        from ._process import process_matrix

        return process_matrix(self, kind, max_branches=max_branches)

    def compile_m2d_converter(self, *, skip_reference_sample: bool = False) -> "MeasurementsToDetectionEventsConverter":
        """A converter from raw measurements (and sweep bits) to detection events, as
        ``stim m2d``. Loops are run pass by pass, never unrolled into memory, so a circuit of
        millions of rounds converts. Its reference run keeps a dense tableau: at most 16,384
        qubits. With ``skip_reference_sample`` the reference is all zeros, as in Stim."""
        return MeasurementsToDetectionEventsConverter(self, skip_reference_sample=skip_reference_sample)


def _target(t: Any) -> str:
    """A target as Stim writes it."""
    if isinstance(t, bool):
        raise TypeError("a target is a qubit index, a string such as 'rec[-1]', or a stim.GateTarget, not a bool")
    if isinstance(t, (int, np.integer)):
        if t < 0:
            raise ValueError(f"qubit {t} is negative")
        return str(int(t))
    if isinstance(t, str):
        return t
    if hasattr(t, "is_combiner") and hasattr(t, "value"):  # a stim.GateTarget
        if t.is_combiner:
            return "*"
        if t.is_measurement_record_target:
            return f"rec[{t.value}]"
        if t.is_sweep_bit_target:
            return f"sweep[{t.value}]"
        pauli = "X" if t.is_x_target else "Y" if t.is_y_target else "Z" if t.is_z_target else ""
        return f"{'!' if t.is_inverted_result_target else ''}{pauli}{t.value}"
    raise TypeError(f"a target is a qubit index, a string such as 'rec[-1]', or a stim.GateTarget, not {type(t).__name__}")


def _targets(targets: Any) -> list:
    if isinstance(targets, (str, int, np.integer)) or hasattr(targets, "is_combiner"):
        return [_target(targets)]
    if isinstance(targets, _stim.PauliString):
        return [_target(t) for t in _stim.target_combined_paulis(targets)]
    try:
        items = list(targets)
    except TypeError:
        raise TypeError(f"targets must be a target or a list of them, not {type(targets).__name__}") from None
    out = []
    for t in items:
        if isinstance(t, _stim.PauliString) or (type(t).__name__ == "PauliString" and type(t).__module__.startswith("stim")):
            out.extend(_target(x) for x in _stim.target_combined_paulis(_stim.PauliString(str(t))))
        else:
            out.append(_target(t))
    return out


def _args(arg: Any) -> list:
    if arg is None:
        return []
    if isinstance(arg, (int, float, np.integer, np.floating)) and not isinstance(arg, bool):
        return [real(arg, "arg")]
    try:
        items = list(arg)
    except TypeError:
        raise TypeError(f"arg must be a number or a list of numbers, not {type(arg).__name__}") from None
    return [real(a, "arg") for a in items]


def _fused_item(a: Any, b: Any) -> Any:
    """Two items as one, where Stim fuses them (the same gate, arguments and tag)."""
    if a[0] != "op" or b[0] != "op" or a[1] != b[1]:
        return None
    both = _core.circuit_items(_stim._items_exact_text([a, b]))
    return both[0] if len(both) == 1 else None


def _stim_text(obj: Any) -> str:
    """A Circuit, CircuitInstruction or CircuitRepeatBlock (ours or Stim's) as circuit text,
    arguments exact."""
    exact = getattr(obj, "_stim_exact_text", None)
    if exact is not None:
        return exact()
    if type(obj).__module__.startswith("stim") and type(obj).__name__ in ("Circuit", "CircuitInstruction"):
        return _stim_object_exact_text(obj)
    if hasattr(obj, "repeat_count") and hasattr(obj, "body_copy"):
        tag = getattr(obj, "tag", "")
        return f"REPEAT{f'[{tag}]' if tag else ''} {obj.repeat_count} {{\n{obj.body_copy()}\n}}"
    if type(obj).__module__.startswith("stim"):
        return str(obj)
    raise TypeError(f"cannot append a {type(obj).__name__}: give an instruction's name, a Circuit, or a Stim circuit, instruction or repeat block")


def _stim_object_exact_text(obj: Any) -> str:
    """A stim.Circuit or stim.CircuitInstruction as text with exact arguments (Stim's own text
    rounds them to 6 digits)."""
    if type(obj).__name__ == "CircuitInstruction":
        targets = [_core.parse_gate_target(_target(t)) for t in obj.targets_copy()]
        return _core.instruction_text(obj.name, list(obj.gate_args_copy()), targets, getattr(obj, "tag", ""), True)
    lines = []
    for item in obj:
        if hasattr(item, "repeat_count"):
            tag = getattr(item, "tag", "")
            body = _stim_object_exact_text(item.body_copy())
            lines.append(f"REPEAT{f'[{_stim._escape_tag(tag)}]' if tag else ''} {item.repeat_count} {{\n{body}\n}}")
        else:
            lines.append(_stim_object_exact_text(item))
    return "\n".join(lines)


import builtins as _builtins

builtins_type = _builtins.type


class Diagram:
    """A circuit's or model's picture (``Circuit.diagram``), as Stim's diagram helper: text, an
    SVG image, a glTF 3D model, or an HTML page. ``str()`` gives its contents (an ``-html``
    SVG type: the viewer page); a notebook shows it inline."""

    __slots__ = ("_text", "type", "_kind")

    def __init__(self, text: str, type: str, kind: Union[str, None] = None) -> None:
        self._text = text
        self.type = type
        self._kind = kind or ("svg" if text.startswith("<svg") else "text")

    def __str__(self) -> str:
        if self._kind == "svg-html":
            return self._repr_html_()
        return self._text

    def __repr__(self) -> str:
        what = {"gltf": "a GLTF 3d model", "svg": "an SVG image", "text": "text", "html": "an HTML document", "svg-html": "an HTML SVG image viewer"}[self._kind]
        return f"<A stabilizer_qec.Diagram containing {what} that will display inline in Jupyter notebooks. Use 'str' or 'print' to access the contents as text.>"

    def _repr_html_(self) -> Union[str, None]:
        if self._kind == "text":
            return "<pre>" + self._text + "</pre>"
        if self._kind == "svg":
            return None  # GitHub's notebook preview drops SVG wrapped in HTML (as Stim)
        if self._kind == "svg-html":
            inner = '<img style="max-width: 100%; max-height: 100%" src="data:image/svg+xml;base64,' + _core.stim_base64(self._text.encode()) + '"/>'
        elif self._kind == "gltf":
            inner = _gltf_viewer(self._text)
        else:
            inner = self._text
        return '<iframe style="width: 100%; height: 300px; overflow: hidden; resize: both; border: 1px dashed gray;" frameBorder="0" srcdoc="' + _escape_srcdoc(inner) + '"></iframe>'

    def _repr_svg_(self) -> Union[str, None]:
        return self._text if self._kind == "svg" else None

    def _repr_pretty_(self, p: Any, cycle: bool) -> None:
        p.text(self._text)

    def save(self, path: Union[str, PathLike]) -> None:
        """Write it to a file (``.svg`` for an SVG image, ``.gltf`` for a 3D model, ``.html``
        for a page)."""
        with open(path, "w", encoding="utf-8") as f:
            f.write(str(self))


# Stim's diagram type names and aliases, by what they draw.
_DIAGRAM_KIND = {
    "timeline-text": "timeline-text",
    **{t: "timeline-svg" for t in ("timeline-svg", "timeline", "timeline-svg-html", "timeline-html")},
    **{t: "timeslice-svg" for t in ("time-slice-svg", "timeslice-svg", "timeslice-html", "timeslice-svg-html", "time-slice-html", "time-slice-svg-html", "timeslice", "time-slice")},
    **{t: "detslice-svg" for t in ("detslice-svg", "detslice", "detslice-html", "detslice-svg-html", "detector-slice-svg", "detector-slice")},
    **{t: "detslice-with-ops-svg" for t in ("detslice-with-ops", "detslice-with-ops-svg", "detslice-with-ops-html", "detslice-with-ops-svg-html", "time+detector-slice-svg")},
    **{t: "timeline-3d" for t in ("timeline-3d", "timeline-3d-html")},
    **{t: "detslice-text" for t in ("detslice-text", "detector-slice-text")},
    **{t: "interactive" for t in ("interactive", "interactive-html")},
    **{t: "matchgraph" for t in ("match-graph-svg", "matchgraph-svg", "matchgraph-svg-html", "matchgraph-html", "match-graph-svg-html", "match-graph-html", "match-graph-3d", "matchgraph-3d", "match-graph-3d-html", "matchgraph-3d-html")},
}


def _filter_single(obj: Any) -> tuple:
    if isinstance(obj, _dem.DemTarget) or builtins_type(obj).__name__ == "DemTarget":
        t = _dem.DemTarget(obj)
        return ((t.is_logical_observable_id(), t.val), [])
    if isinstance(obj, str) and len(obj) > 1 and obj[0] in "DL":
        try:
            return ((obj[0] == "L", int(obj[1:])), [])
        except ValueError:
            pass
    return (None, [float(c) for c in obj])


def _filters(obj: Any) -> list:
    """Stim's ``filter_coords``: one filter, or an iterable of them (``None``: every detector)."""
    if obj is None:
        return [(None, [])]
    try:
        return [_filter_single(obj)]
    except (TypeError, ValueError):
        pass
    return [_filter_single(f) for f in obj]


def _escape_srcdoc(src: str) -> str:
    return src.replace("&", "&amp;").replace("'", "&apos;").replace('"', "&quot;").replace("<", "&lt;").replace(">", "&gt;")


def _gltf_diagram(gltf: str, type: str, html: bool) -> "Diagram":
    # As Stim: the -html form's contents are the viewer page, still marked as a 3D model.
    return Diagram(_gltf_viewer(gltf) if html else gltf, type, "gltf")


def _gltf_viewer(gltf: str) -> str:
    """A page showing a glTF model with three.js (loaded from unpkg), and a download link."""
    data = _core.stim_base64(gltf.encode())
    return (
        "<!DOCTYPE html>\n<html>\n<head>\n<meta charset=\"UTF-8\" />\n"
        "<script type=\"importmap\">{\"imports\": {"
        "\"three\": \"https://unpkg.com/three@0.138.0/build/three.module.js\", "
        "\"three-orbitcontrols\": \"https://unpkg.com/three@0.138.0/examples/jsm/controls/OrbitControls.js\", "
        "\"three-gltf-loader\": \"https://unpkg.com/three@0.138.0/examples/jsm/loaders/GLTFLoader.js\"}}</script>\n"
        "</head>\n<body>\n"
        f"<a download=\"model.gltf\" href=\"data:text/plain;base64,{data}\">Download the 3D model (.gltf)</a>\n"
        "<br>Mouse wheel: zoom. Left drag: orbit. Right drag: pan.\n"
        "<div class=\"viewer\" style=\"width: calc(100vw - 32px); height: calc(100vh - 64px);\">Loading the viewer...</div>\n"
        "<script type=\"module\">\n"
        "import {Box3, Scene, Color, PerspectiveCamera, WebGLRenderer, DirectionalLight, Vector3} from \"three\";\n"
        "import {OrbitControls} from \"three-orbitcontrols\";\n"
        "import {GLTFLoader} from \"three-gltf-loader\";\n"
        "const box = document.currentScript ? document.currentScript.previousElementSibling : document.querySelector(\".viewer\");\n"
        "const link = box.previousElementSibling.previousElementSibling;\n"
        "try {\n"
        "  const gltf = await new GLTFLoader().loadAsync(link.href);\n"
        "  const scene = new Scene();\n"
        "  scene.background = new Color(\"white\");\n"
        "  scene.add(gltf.scene);\n"
        "  const bounds = new Box3().setFromObject(gltf.scene);\n"
        "  const size = bounds.getSize(new Vector3()).length();\n"
        "  const centre = bounds.getCenter(new Vector3());\n"
        "  const camera = new PerspectiveCamera(35, box.clientWidth / box.clientHeight, size / 100, size * 10);\n"
        "  camera.position.copy(centre).add(new Vector3(size * 0.6, size * 0.4, size * 0.6));\n"
        "  const light = new DirectionalLight(0xffffff, 1.5);\n"
        "  light.position.set(1, 2, 3);\n"
        "  scene.add(light);\n"
        "  const renderer = new WebGLRenderer({antialias: true});\n"
        "  renderer.setSize(box.clientWidth, box.clientHeight);\n"
        "  const controls = new OrbitControls(camera, renderer.domElement);\n"
        "  controls.target.copy(centre);\n"
        "  controls.update();\n"
        "  box.textContent = \"\";\n"
        "  box.appendChild(renderer.domElement);\n"
        "  renderer.setAnimationLoop(() => renderer.render(scene, camera));\n"
        "} catch (e) {\n"
        "  box.textContent = \"Couldn't show the model: \" + e;\n"
        "}\n"
        "</script>\n</body>\n</html>\n"
    )


def _crumble_page(url: str) -> str:
    """A page opening the circuit in Crumble, Stim's circuit editor (from algassert.com)."""
    href = url.replace("&", "&amp;").replace('"', "&quot;")
    return (
        "<!DOCTYPE html>\n<html>\n<head>\n<meta charset=\"UTF-8\" />\n<title>Crumble</title>\n</head>\n"
        "<body style=\"margin: 0\">\n"
        f"<iframe src=\"{href}\" style=\"border: 0; width: 100vw; height: calc(100vh - 24px)\"></iframe>\n"
        f"<div style=\"font: 12px sans-serif; padding: 4px\"><a href=\"{href}\" target=\"_blank\">Open in Crumble</a></div>\n"
        "</body>\n</html>\n"
    )


class DetectorErrorModel:
    """A detector error model in Stim's format: independent faults, each with a probability,
    the detectors it flips and the observables it flips.

    Build one from Stim text, a ``stim.DetectorErrorModel``, or ``Circuit.detector_error_model``.
    It is held as written, ``repeat`` blocks and all, and counted through its loops; a decoder
    made from it unrolls it once.
    """

    __slots__ = ("_d",)

    def __init__(self, detector_error_model_text: Union[str, "DetectorErrorModel", Any] = "", *, text: Union[str, None] = None) -> None:
        if text is not None:  # 1.x's keyword
            detector_error_model_text = text
        self._d = call(_core.Dem, text_of(detector_error_model_text, "DetectorErrorModel", DetectorErrorModel))

    @classmethod
    def _wrap(cls, inner: Any) -> "DetectorErrorModel":
        dem = cls.__new__(cls)
        dem._d = inner
        return dem

    @classmethod
    def from_file(cls, file: Union[str, PathLike, Any]) -> "DetectorErrorModel":
        """Read a ``.dem`` file, from a path or an open file."""
        if hasattr(file, "read"):
            return cls(file.read())
        with open_file(file, encoding="utf-8") as f:
            return cls(f.read())

    def to_file(self, file: Any) -> None:
        """Write the model's text to a path or an open text file."""
        text = str(self) + "\n"
        if hasattr(file, "write"):
            file.write(text)
        else:
            with open_file(file, "w", encoding="utf-8") as f:
                f.write(text)

    def copy(self) -> "DetectorErrorModel":
        return DetectorErrorModel(self._d.exact_text())

    def __ne__(self, other: object) -> bool:
        r = self.__eq__(other)
        return r if r is NotImplemented else not r

    # Stim's list-of-instructions interface.
    def _items(self) -> list:
        # (The exact text: Stim's rounds numbers to 16 digits here, which would drop bits on
        # every edit.)
        return _dem.parse_items(self._d.exact_text())

    def _set_items(self, items: list) -> None:
        self._d = call(_core.Dem, _dem.items_exact_text(items))

    def __len__(self) -> int:
        return len(self._items())

    def __iter__(self):
        return iter(self._items())

    def __getitem__(self, index_or_slice: Any) -> Any:
        items = self._items()
        if isinstance(index_or_slice, slice):
            return DetectorErrorModel(_dem.items_exact_text(items[index_or_slice]))
        k = int(index_or_slice)
        if k < 0:
            k += len(items)
        if not 0 <= k < len(items):
            raise IndexError(f"index {index_or_slice} out of range")
        return items[k]

    def clear(self) -> None:
        self._d = call(_core.Dem, "")

    def append(self, instruction: Any, parens_arguments: Any = None, targets: Any = (), *, tag: str = "") -> None:
        """Append an instruction by name (with its arguments, targets and tag), or a
        DemInstruction, DemRepeatBlock or whole model."""
        # (Only the new instructions are parsed: building a model one instruction at a time
        # stays linear.)
        if isinstance(instruction, str):
            if parens_arguments is None:
                args = []
            elif isinstance(parens_arguments, (int, float, np.integer, np.floating)):
                args = [float(parens_arguments)]
            else:
                args = [float(a) for a in parens_arguments]
            text = _dem.DemInstruction(instruction, args, list(targets), tag=tag)._exact_text()
        else:
            if parens_arguments is not None or tuple(targets) != () or tag:
                raise ValueError("Can't specify parens_arguments, targets or tag with a non-name instruction.")
            if isinstance(instruction, DetectorErrorModel):
                text = instruction._d.exact_text()
            elif isinstance(instruction, (_dem.DemInstruction, _dem.DemRepeatBlock)):
                text = instruction._exact_text()
            elif type(instruction).__module__.startswith("stim"):
                text = str(instruction) if type(instruction).__name__ != "DemRepeatBlock" else f"repeat {instruction.repeat_count} {{\n{instruction.body_copy()}\n}}"
            else:
                raise ValueError(f"Don't know how to append {instruction!r}")
        call(self._d.append_text, text)

    def __add__(self, other: object) -> "DetectorErrorModel":
        if not isinstance(other, DetectorErrorModel):
            return NotImplemented
        out = self.copy()
        out += other
        return out

    def __iadd__(self, other: object) -> "DetectorErrorModel":
        if not isinstance(other, DetectorErrorModel):
            return NotImplemented
        self._set_items(self._items() + other._items())
        return self

    def __mul__(self, repetitions: object, op: str = "__mul__") -> "DetectorErrorModel":
        """``repeat repetitions { self }``: empty for 0, a copy for 1."""
        if isinstance(repetitions, bool) or not isinstance(repetitions, (int, np.integer)):
            return NotImplemented
        n = _repetitions(self, repetitions, op)
        if n == 0:
            return DetectorErrorModel()
        if n == 1:
            return self.copy()
        body = _dem.items_exact_text(self._items())
        return DetectorErrorModel(f"repeat {n} {{\n{body}\n}}")

    def __rmul__(self, repetitions: object) -> "DetectorErrorModel":
        return self.__mul__(repetitions, "__rmul__")

    def __imul__(self, repetitions: object) -> "DetectorErrorModel":
        out = self.__mul__(repetitions, "__imul__")
        if out is NotImplemented:
            return NotImplemented
        self._d = out._d
        return self

    def approx_equals(self, other: object, *, atol: float) -> bool:
        """Equal up to arguments differing by at most ``atol``."""
        if not isinstance(other, DetectorErrorModel):
            return False

        def same(a: list, b: list) -> bool:
            if len(a) != len(b):
                return False
            for x, y in zip(a, b):
                if type(x) is not type(y):
                    return False
                if isinstance(x, _dem.DemRepeatBlock):
                    if x.repeat_count != y.repeat_count or not same(x._body._items(), y._body._items()):
                        return False
                elif x.type != y.type or x.tag != y.tag or x.targets_copy() != y.targets_copy() or len(x.args_copy()) != len(y.args_copy()) or any(abs(p - q) > atol for p, q in zip(x.args_copy(), y.args_copy())):
                    return False
            return True

        return same(self._items(), other._items())

    def rounded(self, digits: int) -> "DetectorErrorModel":
        """The same model with error probabilities rounded to ``digits`` decimal places."""
        scale = 10 ** int(digits)

        def walk(items: list) -> list:
            out = []
            for it in items:
                if isinstance(it, _dem.DemRepeatBlock):
                    rb = _dem.DemRepeatBlock.__new__(_dem.DemRepeatBlock)
                    rb._count = it.repeat_count
                    rb._body = DetectorErrorModel(_dem.items_exact_text(walk(it._body._items())))
                    out.append(rb)
                elif it.type == "error":
                    args = [math.floor(a * scale + 0.5) / scale for a in it.args_copy()]
                    out.append(_dem.DemInstruction._raw("error", args, it.targets_copy(), it.tag))
                else:
                    out.append(it)
            return out

        return DetectorErrorModel(_dem.items_exact_text(walk(self._items())))

    def without_tags(self) -> "DetectorErrorModel":
        def walk(items: list) -> list:
            out = []
            for it in items:
                if isinstance(it, _dem.DemRepeatBlock):
                    rb = _dem.DemRepeatBlock.__new__(_dem.DemRepeatBlock)
                    rb._count = it.repeat_count
                    rb._body = DetectorErrorModel(_dem.items_exact_text(walk(it._body._items())))
                    out.append(rb)
                else:
                    out.append(_dem.DemInstruction._raw(it.type, it.args_copy(), it.targets_copy(), ""))
            return out

        return DetectorErrorModel(_dem.items_exact_text(walk(self._items())))

    def get_detector_coordinates(self, only: Any = None) -> dict:
        """Each detector's coordinates (an empty list for none), or those in ``only``."""
        n = self.num_detectors
        wanted = set(range(n)) if only is None else {int(k) for k in only}
        for k in wanted:
            if not 0 <= k < n:
                raise ValueError(f"Detector index {k} is too big. The detector error model has {n} detectors)")
        found: dict = {}
        order = sorted(wanted)

        def add_mul(a: list, b: list, k: int) -> list:
            return [(a[i] if i < len(a) else 0.0) + (b[i] if i < len(b) else 0.0) * k for i in range(max(len(a), len(b)))]

        def extent(items: list) -> tuple:
            """One pass's detector and coordinate shifts, and the lowest and highest detector
            it declares relative to its start (None for none)."""
            shift, offset, lo, hi = [], 0, None, None
            for it in items:
                if isinstance(it, _dem.DemRepeatBlock):
                    d, s, l, h = extent(it._body._items())
                    if l is not None and it.repeat_count:
                        lo = offset + l if lo is None else min(lo, offset + l)
                        hi = max(hi or 0, offset + d * (it.repeat_count - 1) + h)
                    shift, offset = add_mul(shift, s, it.repeat_count), offset + d * it.repeat_count
                elif it.type == "shift_detectors":
                    shift, offset = add_mul(shift, it.args_copy(), 1), offset + sum(it.targets_copy())
                elif it.type == "detector":
                    for t in it.targets_copy():
                        lo = offset + t.val if lo is None else min(lo, offset + t.val)
                        hi = max(hi or 0, offset + t.val)
            return offset, shift, lo, hi

        def size(items: list) -> int:
            return sum(1 + it.repeat_count * size(it._body._items()) if isinstance(it, _dem.DemRepeatBlock) else 1 for it in items)

        # Stim walks every pass, adding each shift in turn; so does this, up to a few million
        # instructions. Past that only the passes that declare a wanted detector are walked
        # (the shifts before them multiplied, their rounding Stim's but for the last bits), so
        # a long loop costs nothing where Stim's takes as long as the loop.
        exact = size(self._items()) <= 1 << 22

        def walk(items: list, shift: list, offset: int) -> Tuple[list, int]:
            for it in items:
                if isinstance(it, _dem.DemRepeatBlock) and exact:
                    for _ in range(it.repeat_count):
                        shift, offset = walk(it._body._items(), shift, offset)
                elif isinstance(it, _dem.DemRepeatBlock):
                    # The first declaration of a detector wins, as in Stim.
                    body, reps = it._body._items(), it.repeat_count
                    d, s, lo, hi = extent(body)
                    passes: set = set()
                    if lo is not None and reps:
                        first = bisect.bisect_left(order, offset + lo)
                        for w in order[first : bisect.bisect_right(order, offset + d * (reps - 1) + hi)]:
                            if d == 0:
                                passes.add(0)
                                break
                            passes.update(range(max(0, -(-(w - offset - hi) // d)), min(reps - 1, (w - offset - lo) // d) + 1))
                    for k in sorted(passes):
                        walk(body, add_mul(shift, s, k), offset + d * k)
                    shift, offset = add_mul(shift, s, reps), offset + d * reps
                elif it.type == "shift_detectors":
                    shift, offset = add_mul(shift, it.args_copy(), 1), offset + sum(it.targets_copy())
                elif it.type == "detector":
                    for t in it.targets_copy():
                        if t.val + offset in wanted and t.val + offset not in found:
                            found[t.val + offset] = [v + (shift[k] if k < len(shift) else 0.0) for k, v in enumerate(it.args_copy())]
            return shift, offset

        walk(self._items(), [], 0)
        return {k: found.get(k, []) for k in order}

    def __str__(self) -> str:
        text = self._d.__str__()
        return text[:-1] if text.endswith("\n") else text

    def __repr__(self) -> str:
        # Stim's repr prints numbers at C++'s default 6 significant digits, as its instructions'
        # str does, where the model's str prints them at full precision.
        def lines(items: list, indent: str) -> list:
            out = []
            for it in items:
                if isinstance(it, _dem.DemRepeatBlock):
                    out.append(f"{indent}repeat {it.repeat_count} {{")
                    out += lines(it._body._items(), indent + "    ")
                    out.append(f"{indent}}}")
                else:
                    out.append(indent + str(it))
            return out

        body = "\n".join(lines(self._items(), "    "))
        if not body:
            return "stabilizer_qec.DetectorErrorModel()"
        return f"stabilizer_qec.DetectorErrorModel('''\n{body}\n''')"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DetectorErrorModel):
            return NotImplemented
        # (Exact, as Stim compares its models' numbers.)
        return self._d.exact_text() == other._d.exact_text()

    def __reduce__(self) -> tuple:
        return (DetectorErrorModel, (self._d.exact_text(),))

    __hash__ = None  # type: ignore[assignment]

    @property
    def num_detectors(self) -> int:
        return self._d.num_detectors

    @property
    def num_observables(self) -> int:
        return self._d.num_observables

    @property
    def num_errors(self) -> int:
        """The number of faults (``error`` lines), counted through loops."""
        return self._d.num_errors

    def diagram(self, type: str = "matchgraph-svg") -> "Diagram":
        """A picture of the model's matching graph: ``"matchgraph-svg"`` (each detector at its
        coordinates, each graph-like fault an edge; the model must be decomposed) or
        ``"matchgraph-3d"`` (a glTF model, as Stim writes it); each with an ``-html`` form."""
        if type in ("matchgraph-svg", "match-graph-svg", "matchgraph-svg-html", "match-graph-svg-html", "matchgraph-html", "match-graph-html"):
            return Diagram(call(self._d.matchgraph_svg), type, "svg-html" if "html" in type else "svg")
        if type in ("matchgraph-3d", "match-graph-3d", "matchgraph-3d-html", "match-graph-3d-html"):
            return _gltf_diagram(call(self._d.matchgraph_3d), type, "html" in type)
        raise ValueError(f"Unrecognized diagram type: {type}")

    def compile_sampler(self, *, seed: Union[int, None] = None) -> "DemSampler":
        """A sampler of the model's faults, as Stim's: each fault fires independently with its
        probability. The same seed gives the same shots on any machine and number of threads;
        ``None`` draws a seed."""
        return DemSampler(self, seed)

    def distance(self, method: str = "milp", *, time_limit: float = 60.0) -> int:
        """The fewest of the model's faults that together flip an observable and set off no
        detector: its distance.

        - ``"milp"`` (default): exact, by integer programming over every fault, hyperedges
          included (each fault taken whole; needs scipy, whose HiGHS solver must prove the
          optimum within ``time_limit`` seconds, or this raises). Exponential at worst; meant
          for models of up to a few thousand faults.
        - ``"graphlike"``: ``shortest_graphlike_error``'s length (graph-like faults only).
        - ``"search"``: ``search_for_undetectable_logical_errors`` with limits of 4 and 4, an
          upper bound.

        >>> DetectorErrorModel("error(0.1) D0\\nerror(0.1) D0 D1\\nerror(0.1) D1 L0").distance()
        3
        """
        if method == "graphlike":
            return self.shortest_graphlike_error().num_errors
        if method == "search":
            return self.search_for_undetectable_logical_errors(
                dont_explore_detection_event_sets_with_size_above=4, dont_explore_edges_with_degree_above=4, dont_explore_edges_increasing_symptom_degree=False
            ).num_errors
        if method != "milp":
            raise ValueError(f"method must be 'milp', 'graphlike' or 'search', not {method!r}")
        return _milp_distance(str(self.flattened()), real(time_limit, "time_limit"))

    def search_for_undetectable_logical_errors(
        self,
        *,
        dont_explore_detection_event_sets_with_size_above: int,
        dont_explore_edges_with_degree_above: int,
        dont_explore_edges_increasing_symptom_degree: bool,
    ) -> "DetectorErrorModel":
        """Stim's search for undetectable logical errors on the model (see
        ``Circuit.search_for_undetectable_logical_errors``): the faults found, as a model with
        each at probability 1."""
        return DetectorErrorModel._wrap(
            call(
                self._d.search_undetectable,
                count(dont_explore_detection_event_sets_with_size_above, "dont_explore_detection_event_sets_with_size_above"),
                count(dont_explore_edges_with_degree_above, "dont_explore_edges_with_degree_above"),
                bool(dont_explore_edges_increasing_symptom_degree),
            )
        )

    def shortest_graphlike_error(self, ignore_ungraphlike_errors: bool = True) -> "DetectorErrorModel":
        """The fewest of the model's graph-like pieces (each fault's ``^``-separated pieces) that
        together flip an observable and no detector, as a model of those faults with
        probability 1, as Stim's method of the same name: its ``num_errors`` is the graph-like
        distance. Pieces of three or more detectors are skipped, or with
        ``ignore_ungraphlike_errors=False`` refused."""
        return DetectorErrorModel._wrap(call(self._d.shortest_graphlike, bool(ignore_ungraphlike_errors)))

    def flattened(self) -> "DetectorErrorModel":
        """The model without ``repeat`` blocks or ``shift_detectors``, as Stim's ``flattened``:
        every detector absolute, every coordinate shifted."""
        return DetectorErrorModel._wrap(call(self._d.flattened))


def _milp_distance(text: str, time_limit: float) -> int:
    """The fewest faults of a flattened model flipping an observable and no detector, by an
    integer program: a 0/1 variable per distinct fault; each detector's faults sum to twice an
    integer; each observable's to twice an integer plus a 0/1 that must be 1 for at least one."""
    try:
        import numpy as np
        from scipy.optimize import LinearConstraint, milp
        from scipy.sparse import coo_matrix
    except ImportError as ex:
        raise ImportError("an exact distance needs scipy: pip install scipy") from ex
    faults = set()
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("error"):
            continue
        dets, obs = set(), set()
        for t in line.split(")", 1)[1].replace("^", " ").split():
            (dets if t[0] == "D" else obs).symmetric_difference_update({int(t[1:])})
        if dets or obs:
            faults.add((frozenset(dets), frozenset(obs)))
    for dets, obs in faults:
        if not dets and obs:
            return 1
    faults = sorted(faults, key=lambda f: (sorted(f[0]), sorted(f[1])))
    detectors = sorted({d for f in faults for d in f[0]})
    observables = sorted({k for f in faults for k in f[1]})
    if not observables:
        raise ValueError("there is no undetectable logical error: no fault flips an observable")
    nf, nd, no = len(faults), len(detectors), len(observables)
    dpos = {d: i for i, d in enumerate(detectors)}
    opos = {k: i for i, k in enumerate(observables)}
    # Whether one exists is linear algebra over GF(2), decided here rather than by the solver
    # (whose verdict on an infeasible program differs between scipy versions): one does exactly
    # when some observable's row of faults lies outside the span of the detectors' rows.
    det_rows, obs_rows = [0] * nd, [0] * no
    for j, (dets, obs) in enumerate(faults):
        for d in dets:
            det_rows[dpos[d]] |= 1 << j
        for k in obs:
            obs_rows[opos[k]] |= 1 << j
    if _gf2_rank(det_rows + obs_rows) == _gf2_rank(det_rows):
        raise ValueError("there is no undetectable logical error: no set of faults flips an observable and no detector")
    # Variables: x (faults), y (detector halves), z (observable halves), w (observable flipped).
    n = nf + nd + 2 * no
    rows, cols, vals = [], [], []
    for j, (dets, obs) in enumerate(faults):
        for d in dets:
            rows.append(dpos[d]); cols.append(j); vals.append(1)
        for k in obs:
            rows.append(nd + opos[k]); cols.append(j); vals.append(1)
    for i in range(nd):
        rows.append(i); cols.append(nf + i); vals.append(-2)
    for i in range(no):
        rows.append(nd + i); cols.append(nf + nd + i); vals.append(-2)
        rows.append(nd + i); cols.append(nf + nd + no + i); vals.append(-1)
        rows.append(nd + no); cols.append(nf + nd + no + i); vals.append(1)
    a = coo_matrix((vals, (rows, cols)), shape=(nd + no + 1, n)).tocsr()
    lower = np.zeros(nd + no + 1)
    upper = np.zeros(nd + no + 1)
    lower[-1], upper[-1] = 1, np.inf
    degree = np.asarray(abs(a[:, :nf]).sum(axis=1)).ravel()
    ub = np.concatenate([np.ones(nf), np.floor(degree[:nd] / 2), np.floor(degree[nd : nd + no] / 2), np.ones(no)])
    cost = np.concatenate([np.ones(nf), np.zeros(n - nf)])
    res = milp(cost, constraints=LinearConstraint(a, lower, upper), integrality=np.ones(n), bounds=(0, ub), options={"time_limit": time_limit})
    if res.status != 0 or res.x is None:
        found = "none found" if res.x is None else f"best found {round(res.fun)}"
        raise RuntimeError(f"the distance was not proven within {time_limit} s ({found}); raise time_limit, or use method='search' for a bound")
    # The solver's answer, checked: its faults set off no detector and flip an observable.
    chosen = sum(1 << j for j in range(nf) if res.x[j] > 0.5)
    if any(bin(r & chosen).count("1") % 2 for r in det_rows) or not any(bin(r & chosen).count("1") % 2 for r in obs_rows):
        raise RuntimeError("the integer program returned faults that are not an undetectable logical error; please report this")
    return bin(chosen).count("1")


def _gf2_rank(rows: list) -> int:
    """The rank over GF(2) of rows given as integers (bit j: column j)."""
    pivots: dict = {}
    for r in rows:
        while r:
            top = r.bit_length() - 1
            if top not in pivots:
                pivots[top] = r
                break
            r ^= pivots[top]
    return len(pivots)


def _warn_if_coherent_errors(circuit: "Circuit") -> None:
    """The frame engines read coherent errors as Stim does, as nothing: say so once per sampler
    (``twirled()`` and ``compile_coherent_sampler`` are the models that run them)."""
    if circuit.num_non_clifford_gates() == 0 and (call(circuit._c.t_count) > 0 or call(circuit._c.num_clifford_angle_errors) > 0):  # (the engine's count: tsim.Circuit.tcount is Tsim's, which has no coherent errors)
        warnings.warn(
            "the frame sampler reads the circuit's coherent errors (I_ERROR[R_Z(theta=...)] and the rest) as Stim does, as nothing: "
            "sample circuit.twirled() for the Pauli-twirled model, circuit.compile_coherent_sampler() for the coherent one, "
            "or circuit.without_tags() to say that nothing is meant",
            stacklevel=4,
        )


class MeasurementSampler:
    """Raw measurement records, as ``stim.CompiledMeasurementSampler``. Made by
    ``Circuit.compile_sampler``."""

    def __init__(self, circuit: Circuit, seed: Union[int, None] = None, skip_reference_sample: bool = False, reference_sample: Any = None, *, device: str = "cpu") -> None:
        if not isinstance(circuit, Circuit):
            circuit = Circuit(circuit)
        self._circuit = circuit
        self._reference = None
        if reference_sample is not None:
            if skip_reference_sample:
                raise ValueError("skip_reference_sample = True but reference_sample is not None.")
            n = circuit.num_measurements
            r = np.asarray(reference_sample)
            if r.dtype == np.uint8:
                r = np.unpackbits(r, count=n, bitorder="little") if r.size == (n + 7) // 8 else r
            if r.ndim != 1 or r.shape[0] != n:
                raise ValueError(f"reference_sample must have {n} bits (or {(n + 7) // 8} bit-packed bytes), not shape {np.asarray(reference_sample).shape}")
            self._reference = r.astype(bool)
            skip_reference_sample = True
        if circuit.num_non_clifford_gates():
            # A circuit with non-Clifford gates: its records from the stabilizer-rank sampler.
            if device != "cpu":
                raise ValueError(f"a circuit with non-Clifford gates samples on the stabilizer-rank sampler, device 'cpu' only (not {device!r})")
            self._s = _ExactRecords(call(circuit._c.stabilizer_rank_sampler, seed_of(seed)), circuit.reference_sample() if skip_reference_sample else None)
        else:
            _warn_if_coherent_errors(circuit)
            self._s = call(circuit._c.measurement_sampler, seed_of(seed), bool(skip_reference_sample), str(device))

    @property
    def device(self) -> str:
        """``"cpu"``, ``"cpu-lanes"`` or ``"gpu"``."""
        return getattr(self._s, "device", "cpu")

    def __reduce__(self) -> tuple:
        raise TypeError("a MeasurementSampler is a position in a stream of shots, which a copy would restart; send the circuit and a seed instead")

    def __repr__(self) -> str:
        return f"stabilizer_qec.CompiledMeasurementSampler({self._circuit!r})"

    @property
    def num_measurements(self) -> int:
        return self._s.num_measurements

    def sample(self, shots: int, *, bit_packed: bool = False, threads: int = 1) -> np.ndarray:
        """``shots`` measurement records: (shots, measurements) bool, or bit-packed uint8 rows."""
        shots = count(shots, "shots")
        raw = call(self._s.sample, shots, count(threads, "threads"))
        if self._reference is None:
            return b8_to_rows(raw, shots, self.num_measurements, bit_packed)
        rows = b8_to_rows(raw, shots, self.num_measurements, False) ^ self._reference[None, :]
        return pack_rows(rows, bit_packed)

    def sample_bit_packed(self, shots: int) -> np.ndarray:
        """``sample(shots, bit_packed=True)``."""
        return self.sample(shots, bit_packed=True)

    def sample_write(self, shots: int, *, filepath: Union[str, PathLike], format: str = "01") -> None:
        """Writes ``shots`` records to a file in one of Stim's formats."""
        from ._shots import write_shot_data_file

        write_shot_data_file(data=self.sample(shots), path=filepath, format=format, num_measurements=self.num_measurements)


class DemSampler:
    """Shots drawn from a detector error model's faults, as ``stim.CompiledDemSampler``. Made
    by ``DetectorErrorModel.compile_sampler``."""

    def __init__(self, model: "DetectorErrorModel", seed: Union[int, None] = None) -> None:
        if not isinstance(model, DetectorErrorModel):
            model = DetectorErrorModel(model)
        self._s = call(model._d.sampler, seed_of(seed))

    def __reduce__(self) -> tuple:
        raise TypeError("a DemSampler is a position in a stream of shots, which a copy would restart; send the model and a seed instead")

    @property
    def num_detectors(self) -> int:
        return self._s.num_detectors

    @property
    def num_observables(self) -> int:
        return self._s.num_observables

    @property
    def num_errors(self) -> int:
        return self._s.num_errors

    def sample(
        self,
        shots: int,
        *,
        bit_packed: bool = False,
        return_errors: bool = False,
        recorded_errors_to_replay: Any = None,
        threads: int = 1,
    ) -> tuple:
        """``(detectors, observables, errors)``: (shots, n) bool arrays, or bit-packed uint8
        rows with ``bit_packed``. ``errors`` (which faults fired, one column per fault in the
        model's order) is returned with ``return_errors``, else ``None``.
        ``recorded_errors_to_replay`` (such an array) replays those faults instead of drawing."""
        shots = count(shots, "shots")
        nd, no, ne = self.num_detectors, self.num_observables, self.num_errors
        if recorded_errors_to_replay is not None:
            raw, n = rows_to_b8(recorded_errors_to_replay, ne, np.asarray(recorded_errors_to_replay).dtype == np.uint8, "recorded_errors_to_replay")
            if n != shots:
                raise ValueError("recorded_errors_to_replay.shape[0] != shots")
            d, o = call(self._s.replay, raw, n)
            # The faults replayed, the padding bits of packed rows cleared, as Stim returns them.
            e = b8_to_rows(raw, n, ne, False) if return_errors else None
            if e is not None and bit_packed:
                e = np.packbits(e, axis=1, bitorder="little")
        else:
            d, o, e = call(self._s.sample, shots, count(threads, "threads"), bool(return_errors))
            e = b8_to_rows(e, shots, ne, bit_packed) if return_errors else None
        return b8_to_rows(d, shots, nd, bit_packed), b8_to_rows(o, shots, no, bit_packed), e

    def sample_write(
        self,
        shots: int,
        *,
        det_out_file: Union[None, str, PathLike],
        det_out_format: str = "01",
        obs_out_file: Union[None, str, PathLike],
        obs_out_format: str = "01",
        err_out_file: Union[None, str, PathLike] = None,
        err_out_format: str = "01",
        replay_err_in_file: Union[None, str, PathLike] = None,
        replay_err_in_format: str = "01",
    ) -> None:
        """Samples (or, from ``replay_err_in_file``, replays) shots and writes the detection
        events, observable flips and faults to files in Stim's formats (any may be None)."""
        from ._shots import read_shot_data_file, write_shot_data_file

        ne = self.num_errors
        replay = None
        if replay_err_in_file is not None:
            replay = read_shot_data_file(path=replay_err_in_file, format=replay_err_in_format, num_measurements=ne)
            if len(replay) < shots:
                raise ValueError("Expected more error data for the requested number of shots.")
            replay = replay[:shots]
        d, o, e = self.sample(shots, return_errors=err_out_file is not None, recorded_errors_to_replay=replay)
        if det_out_file is not None:
            write_shot_data_file(data=d, path=det_out_file, format=det_out_format, num_detectors=self.num_detectors)
        if obs_out_file is not None:
            write_shot_data_file(data=o, path=obs_out_file, format=obs_out_format, num_observables=self.num_observables)
        if err_out_file is not None:
            write_shot_data_file(data=e, path=err_out_file, format=err_out_format, num_measurements=ne)


class _ExactRecords:
    """The stabilizer-rank sampler's records behind a ``MeasurementSampler``: flips relative to
    the reference run when one is given (``skip_reference_sample``)."""

    __slots__ = ("_s", "_reference")

    def __init__(self, s: Any, reference: Union[np.ndarray, None]) -> None:
        self._s = s
        self._reference = reference

    @property
    def num_measurements(self) -> int:
        return self._s.num_measurements

    def sample(self, shots: int, threads: int) -> bytes:
        raw = call(self._s.sample_measurements, shots, threads)
        if self._reference is None:
            return raw
        rows = b8_to_rows(raw, shots, self.num_measurements, False) ^ self._reference[None, :]
        return pack_rows(rows, True).tobytes()


class DetectorSampler:
    """Detection events and observable flips, 64 shots to a machine word, as
    ``stim.CompiledDetectorSampler``. Made by ``Circuit.compile_detector_sampler``."""

    __slots__ = ("_s", "_circuit")

    def __init__(self, circuit: Circuit, seed: Union[int, None] = None, *, device: str = "cpu") -> None:
        if not isinstance(circuit, Circuit):
            circuit = Circuit(circuit)
        self._circuit = circuit
        _warn_if_coherent_errors(circuit)
        self._s = call(circuit._c.sampler, seed_of(seed), str(device))

    @property
    def device(self) -> str:
        """``"cpu"``, ``"cpu-lanes"`` or ``"gpu"``."""
        return self._s.device

    def __reduce__(self) -> tuple:
        raise TypeError(
            "a DetectorSampler is a position in a stream of shots, which a copy would restart; "
            "send the circuit and a seed, and compile a sampler where it is used"
        )

    def __repr__(self) -> str:
        return f"stabilizer_qec.CompiledDetectorSampler({self._circuit!r})"

    @property
    def num_detectors(self) -> int:
        return self._s.num_detectors

    @property
    def num_observables(self) -> int:
        return self._s.num_observables

    def _draw(self, shots: int, threads: int) -> tuple:
        d, o = call(self._s.sample, shots, threads)
        return b8_to_rows(d, shots, self.num_detectors, False), b8_to_rows(o, shots, self.num_observables, False)

    def sample(
        self,
        shots: int,
        *,
        prepend_observables: bool = False,
        append_observables: bool = False,
        separate_observables: bool = False,
        bit_packed: bool = False,
        dets_out: Union[np.ndarray, None] = None,
        obs_out: Union[np.ndarray, None] = None,
        threads: int = 1,
    ) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray]]:
        """``shots`` shots, as Stim's ``CompiledDetectorSampler.sample`` gives them: detection
        events of shape (shots, detectors), bool, or (shots, ⌈detectors/8⌉) uint8 when
        ``bit_packed`` (bit k of a row in byte k // 8, position k % 8). ``separate_observables``
        returns ``(detections, observables)``; ``prepend_observables`` / ``append_observables``
        put the observables before / after each row's detectors (both: both). ``dets_out`` and
        ``obs_out`` are arrays to write into. ``threads = 0`` uses every core; the shots do not
        depend on it.

        Successive calls continue the stream. Shots are drawn 64 at a time, and a call that
        ends partway through 64 discards the rest of them.
        """
        shots = count(shots, "shots")
        threads = count(threads, "threads")
        if separate_observables and (append_observables or prepend_observables):
            raise ValueError("Can't specify separate_observables=True with append_observables=True or prepend_observables=True")
        dets, obs = self._draw(shots, threads)
        if prepend_observables or append_observables:
            parts = ([obs] if prepend_observables else []) + [dets] + ([obs] if append_observables else [])
            dets = np.concatenate(parts, axis=1)
        dets = _into(pack_rows(dets, bit_packed), dets_out)
        if separate_observables:
            return dets, _into(pack_rows(obs, bit_packed), obs_out)
        return dets

    def sample_bit_packed(self, shots: int, *, prepend_observables: bool = False, append_observables: bool = False) -> np.ndarray:
        """``sample(shots, bit_packed=True, ...)``."""
        return self.sample(shots, prepend_observables=prepend_observables, append_observables=append_observables, bit_packed=True)

    def sample_write(
        self,
        shots: int,
        *,
        filepath: Union[str, PathLike],
        format: str = "01",
        obs_out_filepath: Union[str, PathLike, None] = None,
        obs_out_format: str = "01",
        prepend_observables: bool = False,
        append_observables: bool = False,
    ) -> None:
        """Writes ``shots`` shots to a file in one of Stim's formats; the observables go into
        each row (prepended or appended) and/or their own file."""
        from ._shots import encode_shots, write_shot_data_file

        dets, obs = self._draw(count(shots, "shots"), 1)
        nd, no = self.num_detectors, self.num_observables
        rows = np.concatenate(([obs] if prepend_observables else []) + [dets] + ([obs] if append_observables else []), axis=1)
        names = [f"L{k}" for k in range(no)] * prepend_observables + [f"D{k}" for k in range(nd)] + [f"L{k}" for k in range(no)] * append_observables
        with open_file(filepath, "wb") as f:
            f.write(encode_shots(rows, format, num_detectors=rows.shape[1], names=names))
        if obs_out_filepath is not None:
            write_shot_data_file(data=obs, path=obs_out_filepath, format=obs_out_format, num_observables=no)


def _into(data: np.ndarray, out: Union[np.ndarray, None]) -> np.ndarray:
    """``data``, or ``out`` with ``data`` written into it (shape and dtype must match)."""
    if out is None:
        return data
    if out.shape != data.shape or out.dtype != data.dtype:
        raise ValueError(f"Expected output buffer to have shape={data.shape} but its shape is {out.shape}.")
    out[...] = data
    return out


class ExactSampler(DetectorSampler):
    """Detection events and observable flips from the exact state vector. Made by
    ``Circuit.compile_exact_sampler``; ``sample`` is the detector sampler's. Each shot has its
    own random stream, so threads never change the shots and calls continue shot by shot."""

    __slots__ = ()

    def __init__(self, circuit: Circuit, seed: Union[int, None] = None) -> None:
        if not isinstance(circuit, Circuit):
            circuit = Circuit(circuit)
        self._circuit = circuit
        self._s = call(circuit._c.exact_sampler, seed_of(seed))

    def __repr__(self) -> str:
        return f"stabilizer_qec.ExactSampler({self._circuit!r})"

    @property
    def device(self) -> str:
        """``"cpu"``: the state vector runs on the CPU."""
        return "cpu"

    @property
    def num_qubits(self) -> int:
        """The qubits in use, which the state vector holds."""
        return self._s.num_qubits


class StabilizerRankSampler(DetectorSampler):
    """Shots of a Clifford+T circuit from the stabilizer-rank sampler, at any qubit count. Made
    by ``Circuit.compile_stabilizer_rank_sampler`` (and by ``compile_detector_sampler`` on a
    circuit with non-Clifford gates); ``sample`` is the detector sampler's. Each shot has its
    own random stream, so threads never change the shots and calls continue shot by shot."""

    __slots__ = ()

    def __init__(self, circuit: Circuit, seed: Union[int, None] = None) -> None:
        if not isinstance(circuit, Circuit):
            circuit = Circuit(circuit)
        self._circuit = circuit
        self._s = call(circuit._c.stabilizer_rank_sampler, seed_of(seed))

    def __repr__(self) -> str:
        return f"stabilizer_qec.StabilizerRankSampler({self._circuit!r})"

    @property
    def device(self) -> str:
        """``"cpu"``: the stabilizer-rank engine runs on the CPU."""
        return "cpu"

    @property
    def num_measurements(self) -> int:
        return self._s.num_measurements

    @property
    def num_qubits(self) -> int:
        return self._s.num_qubits

    @property
    def num_non_clifford(self) -> int:
        """How many of the circuit's gates split the state: its non-Clifford gates (rotations
        at a Clifford angle are not among them)."""
        return self._s.num_non_clifford

    @property
    def peak_terms(self) -> int:
        """The most terms (Clifford branches) any shot sampled so far held at once: the rank
        the sampler reached. 0 before any shot."""
        return self._s.peak_terms

    def sample_measurements(self, shots: int, *, bit_packed: bool = False, threads: int = 1) -> np.ndarray:
        """``shots`` measurement records, (shots, measurements) bool or bit-packed uint8 rows,
        from the same stream of shots as ``sample``."""
        shots = count(shots, "shots")
        return b8_to_rows(call(self._s.sample_measurements, shots, count(threads, "threads")), shots, self.num_measurements, bit_packed)

    def probability_of(self, records: Any, *, samples: int = 1000, seed: Union[int, None] = None) -> np.ndarray:
        """The probability of each measurement record in ``records`` (a bool array of shape
        (measurements,) or (shots, measurements)): exact when the circuit has no noise channel
        or measurement flip; with noise, an average over ``samples`` draws of it (from the
        stream ``seed``, so the same seed gives the same estimate)."""
        r = np.asarray(records)
        if r.dtype != bool:
            r = r != 0
        one = r.ndim == 1
        rows = r.reshape(-1, r.shape[-1]) if r.ndim else r.reshape(1, 0)
        if rows.shape[1] != self.num_measurements:
            raise ValueError(f"records must have {self.num_measurements} bits each, not {rows.shape[1]}")
        out = np.asarray(call(self._s.probability_of, [list(map(bool, row)) for row in rows], count(samples, "samples", 1), seed_of(seed)))
        return out[0] if one else out


class LeakageSampler:
    """Shots of a circuit with leakage, and which measurements found their qubit leaked. Made
    by ``Circuit.compile_leakage_sampler``."""

    __slots__ = ("_s", "_m", "_h")

    def __init__(self, circuit: Circuit, leaked_reads_one: bool = True, seed: Union[int, None] = None) -> None:
        if not isinstance(circuit, Circuit):
            circuit = Circuit(circuit)
        self._s = call(circuit._c.leakage_sampler, seed_of(seed), bool(leaked_reads_one))
        self._m = circuit.num_measurements
        self._h = np.asarray(call(_core.circuit_herald_map, circuit._stim_exact_text())[0], dtype=np.int64)

    def __reduce__(self) -> tuple:
        raise TypeError("a LeakageSampler is a position in a stream of shots; send the circuit and a seed")

    @property
    def num_detectors(self) -> int:
        return self._s.num_detectors

    @property
    def num_observables(self) -> int:
        return self._s.num_observables

    def sample(self, shots: int, *, threads: int = 1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(detection events, observable flips, heralds)`` for ``shots`` shots: bool arrays
        of shape (shots, detectors), (shots, observables) and (shots, measurement records), the
        last saying which measurements found their qubit leaked (False for other records), and
        empty in width when the circuit has no leakage."""
        shots = count(shots, "shots")
        d, o, h = call(self._s.sample, shots, count(threads, "threads"))
        heralds = np.frombuffer(h, dtype=np.uint8).astype(bool)
        heralds = heralds.reshape(shots, -1) if heralds.size else np.zeros((shots, 0), dtype=bool)
        return b8_to_rows(d, shots, self.num_detectors, False), b8_to_rows(o, shots, self.num_observables, False), heralds

    @property
    def num_measurements(self) -> int:
        return self._m

    @property
    def num_heralds(self) -> int:
        """The circuit's ``HERALD_LEAKAGE_EVENT`` records."""
        return len(self._h)

    def sample_measurements(self, shots: int, *, threads: int = 1) -> tuple[np.ndarray, np.ndarray]:
        """``(measurements, leakage flags)`` for ``shots`` shots, as Deltakit's simulation
        returns them: bool arrays of shape (shots, measurements less heralds) and (shots,
        heralds), each herald's record in the second. The circuit's ``without_leakage()``
        converts the first to detection events and observable flips (its
        ``compile_m2d_converter``); ``LeakageAwareDecoder`` takes the second. From the same
        stream of shots as ``sample``."""
        shots = count(shots, "shots")
        rows = b8_to_rows(call(self._s.sample_measurements, shots, count(threads, "threads")), shots, self._m, False)
        flags = np.zeros(self._m, dtype=bool)
        flags[self._h] = True
        return rows[:, ~flags], rows[:, flags]


class CoherentSampler:
    """Weighted shots of a circuit with coherent errors. Made by
    ``Circuit.compile_coherent_sampler``."""

    __slots__ = ("_s",)

    def __init__(self, circuit: Circuit, order: int = 3, seed: Union[int, None] = None, *, pairs: bool = False, max_complete: int = 50_000) -> None:
        if not isinstance(circuit, Circuit):
            circuit = Circuit(circuit)
        limit = count(max_complete, "max_complete")
        # Non-Clifford gates, or coherent errors at Clifford angles (fixed Cliffords): the joint
        # sampler, which runs them exactly and twirls the other errors.
        if circuit.num_non_clifford_gates() or call(circuit._c.num_clifford_angle_errors):
            self._s = call(circuit._c.joint_sampler, seed_of(seed), count(order, "order"), 12, True, 1, bool(pairs), limit)
        else:
            self._s = call(circuit._c.coherent_sampler, seed_of(seed), count(order, "order"), 12, True, 1, limit)
        if not self._s.kernel_complete:
            warnings.warn(
                f"the circuit's {self.num_locations} coherent rotations are past max_complete={limit}: the sampler's interference "
                "generators are the local ones only, so interference between distant faults (a logical error at one time with the "
                "same logical at another, whose amplitudes add) is left out and the weights are truncated; raise max_complete to "
                "complete the kernel",
                stacklevel=3,
            )

    def __reduce__(self) -> tuple:
        raise TypeError("a CoherentSampler is a position in a stream of shots; send the circuit and a seed")

    @property
    def num_exact(self) -> int:
        """The non-Clifford gates the sampler runs exactly (0 for the coherent sampler of a
        Clifford circuit's errors)."""
        return getattr(self._s, "num_exact", 0)

    @property
    def num_detectors(self) -> int:
        return self._s.num_detectors

    @property
    def num_observables(self) -> int:
        return self._s.num_observables

    @property
    def num_locations(self) -> int:
        """The coherent rotations in the circuit."""
        return self._s.num_locations

    @property
    def num_generators(self) -> int:
        """The generators of the circuit's interference (sets of rotations no record tells
        apart) whose products the sampler sums: the local ones, found by hashing, and the
        ``num_nonlocal_generators`` that complete them to a basis of the whole kernel."""
        return self._s.num_generators

    @property
    def num_nonlocal_generators(self) -> int:
        """The generators no local search finds, which complete the local ones to a basis of the
        kernel. The commonest is a logical error at one time with the same logical at another
        (2d rotations on a distance-d memory, no syndrome, no logical effect): its halves give
        the same outcome, so their amplitudes add."""
        return self._s.num_nonlocal_generators

    @property
    def kernel_complete(self) -> bool:
        """Whether the generators (with the groups of rotations that are one fault in several
        places) span the whole kernel. False past ``max_complete`` rotations: interference
        between distant faults is then left out, and the sampler warns when it is made."""
        return self._s.kernel_complete

    @property
    def exact_weights(self) -> bool:
        """Whether every shot sums every element of the kernel (complete, at most twelve
        generators): the weighted shots are then distributed exactly as the coherent circuit's.
        Otherwise a shot sums the products of up to ``order`` generators its faults touch."""
        return self._s.exact_weights

    def sample(self, shots: int, *, bit_packed: bool = False, threads: int = 1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(detection events, observable flips, weights)`` for ``shots`` shots, the first two
        as the detector sampler gives them with ``separate_observables``. Each shot has its own
        random stream: threads never change the shots, and calls continue shot by shot."""
        shots = count(shots, "shots")
        d, o, w = call(self._s.sample, shots, count(threads, "threads"))
        return b8_to_rows(d, shots, self.num_detectors, bit_packed), b8_to_rows(o, shots, self.num_observables, bit_packed), np.asarray(w)


class MeasurementsToDetectionEventsConverter:
    """Raw measurement records to detection events and observable flips, as ``stim m2d``:
    each detector and observable compared with a noiseless reference run. Made by
    ``Circuit.compile_m2d_converter``."""

    __slots__ = ("_m", "_circuit", "_skip")

    def __init__(self, circuit: Circuit, *, skip_reference_sample: bool = False) -> None:
        if not isinstance(circuit, Circuit):
            circuit = Circuit(circuit)
        self._m = call(circuit._c.m2d)
        self._circuit = circuit
        # With skip_reference_sample the reference is all zeros: undo the reference's parities.
        self._skip = None
        if skip_reference_sample:
            d, o = circuit.reference_detector_and_observable_signs()
            self._skip = (np.asarray(d, dtype=bool), np.asarray(o, dtype=bool))

    def __reduce__(self) -> tuple:
        return (_converter, (self._circuit, self._skip is not None))

    def __repr__(self) -> str:
        return f"stabilizer_qec.CompiledMeasurementsToDetectionEventsConverter({self._circuit!r})"

    @property
    def num_measurements(self) -> int:
        return self._m.num_measurements

    @property
    def num_sweep_bits(self) -> int:
        return self._m.num_sweep_bits

    @property
    def num_detectors(self) -> int:
        return self._m.num_detectors

    @property
    def num_observables(self) -> int:
        return self._m.num_observables

    def convert(
        self,
        *,
        measurements: np.ndarray,
        sweep_bits: Union[np.ndarray, None] = None,
        separate_observables: bool = False,
        append_observables: bool = False,
        bit_packed: bool = False,
    ) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray]]:
        """``measurements`` (and ``sweep_bits``) are (shots, n) bool arrays, or uint8 rows
        bit-packed as Stim packs them. The output takes the same forms as
        ``DetectorSampler.sample``'s, packed when ``bit_packed``."""
        if separate_observables and append_observables:
            raise ValueError("choose separate_observables or append_observables, not both")
        m = np.asarray(measurements)
        packed_in = m.dtype == np.uint8
        meas, shots = rows_to_b8(m, self.num_measurements, packed_in, "measurements")
        ns = self.num_sweep_bits
        if sweep_bits is None:
            sweeps = bytes(stride(ns) * shots)
        else:
            s = np.asarray(sweep_bits)
            sweeps, sweep_shots = rows_to_b8(s, ns, s.dtype == np.uint8, "sweep_bits")
            if sweep_shots != shots:
                raise ValueError(f"{sweep_shots} shots of sweep bits for {shots} shots of measurements")
        d, o = call(self._m.convert, meas, sweeps, shots)
        nd, no = self.num_detectors, self.num_observables
        if self._skip is not None:
            d = (b8_to_rows(d, shots, nd, False) ^ self._skip[0][None, :])
            o = (b8_to_rows(o, shots, no, False) ^ self._skip[1][None, :])
            d, o = np.packbits(d, axis=1, bitorder="little").tobytes(), np.packbits(o, axis=1, bitorder="little").tobytes()
        if append_observables:
            rows = np.concatenate([b8_to_rows(d, shots, nd, False), b8_to_rows(o, shots, no, False)], axis=1)
            return pack_rows(rows, bit_packed)
        dets = b8_to_rows(d, shots, nd, bit_packed)
        if separate_observables:
            return dets, b8_to_rows(o, shots, no, bit_packed)
        return dets

    def convert_file(
        self,
        *,
        measurements_filepath: Union[str, PathLike],
        measurements_format: str = "01",
        sweep_bits_filepath: Union[str, PathLike, None] = None,
        sweep_bits_format: str = "01",
        detection_events_filepath: Union[str, PathLike],
        detection_events_format: str = "01",
        append_observables: bool = False,
        obs_out_filepath: Union[str, PathLike, None] = None,
        obs_out_format: str = "01",
    ) -> None:
        """``convert`` from file to file, in Stim's formats."""
        from ._shots import read_shot_data_file, write_shot_data_file

        meas = read_shot_data_file(path=measurements_filepath, format=measurements_format, num_measurements=self.num_measurements)
        sweeps = None
        if sweep_bits_filepath is not None:
            sweeps = read_shot_data_file(path=sweep_bits_filepath, format=sweep_bits_format, num_measurements=self.num_sweep_bits)
        dets, obs = self.convert(measurements=meas, sweep_bits=sweeps, separate_observables=True)
        nd, no = self.num_detectors, self.num_observables
        if append_observables:
            write_shot_data_file(data=np.concatenate([dets, obs], axis=1), path=detection_events_filepath, format=detection_events_format, num_detectors=nd, num_observables=no)
        else:
            write_shot_data_file(data=dets, path=detection_events_filepath, format=detection_events_format, num_detectors=nd)
        if obs_out_filepath is not None:
            write_shot_data_file(data=obs, path=obs_out_filepath, format=obs_out_format, num_observables=no)


def _converter(circuit: Circuit, skip: bool) -> MeasurementsToDetectionEventsConverter:
    return MeasurementsToDetectionEventsConverter(circuit, skip_reference_sample=skip)


# Stim's class names.
CompiledMeasurementSampler = MeasurementSampler
CompiledDetectorSampler = DetectorSampler
CompiledDemSampler = DemSampler
CompiledMeasurementsToDetectionEventsConverter = MeasurementsToDetectionEventsConverter


# ---------------------------------------------------------------------------------------------
# Stim's exports implemented over the Python objects: Crumble URLs, detecting-region filters,
# and the max-SAT problems.


def _crumble_args(args: list) -> str:
    out = []
    for e in args:
        if -9.2e18 < e < 9.2e18 and float(int(e)) == e:
            out.append(str(int(e)))
        else:
            out.append("%g" % e)
    return "(" + ",".join(out) + ")"


def _crumble_pauli_and_qubit(target: Any) -> Tuple[str, str]:
    g = getattr(target, "gate_target", target)
    text = (g.target_str() if hasattr(g, "target_str") else str(g)).lstrip("!")
    if text[:1] in ("X", "Y", "Z"):
        return text[0], text[1:]
    return "I", text


def _crumble_url(circuit: "Circuit", skip_detectors: bool, mark: Any) -> str:
    marks = []
    if mark is not None:
        for k, errors in sorted(dict(mark).items()):
            for e in errors:
                locs = list(e.circuit_error_locations)
                if locs:
                    loc = locs[0]
                    marks.append((int(k), [(f.instruction_offset, f.iteration_index) for f in loc.stack_frames], loc))
    out: list = ["https://algassert.com/crumble#circuit="]

    def write(items: list, active_in: list) -> None:
        for k, it in enumerate(items):
            active = []
            for m, frames, loc in active_in:
                if frames and frames[-1][0] == k:
                    active.append((m, frames[:-1], loc))
            adding = any(not frames for _, frames, _ in active)
            if adding:
                out.append(";TICK")
            for m, frames, loc in active:
                if not frames:
                    for t in loc.flipped_pauli_product:
                        p, q = _crumble_pauli_and_qubit(t)
                        out.append(f";MARK{p}({m}){q}")
                    fm = loc.flipped_measurement
                    obs = list(fm.observable) if fm is not None else []
                    if obs:
                        p, q = _crumble_pauli_and_qubit(obs[0])
                        out.append(f";MARK{'XZ'[p == 'X']}({m}){q}")
            if adding:
                out.append(";TICK")
            if it[0] == "op" and it[1] == "DETECTOR" and skip_detectors:
                continue
            if k > 0 or adding:
                out.append(";")
            if it[0] == "repeat":
                _, count, _tag, body = it
                if not active:
                    out.append(f"REPEAT_{count}_{{;")
                    write(body, active)
                    out.append(";}")
                else:
                    for k2 in range(count):
                        write(body, [a for a in active if a[1] and a[1][-1][1] == k2])
                continue
            _, name, _tag, args, targets = it
            out.append({"DETECTOR": "DT", "QUBIT_COORDS": "Q", "OBSERVABLE_INCLUDE": "OI"}.get(name, name))
            if args:
                out.append(_crumble_args(args))
            k2 = 0
            while k2 < len(targets):
                t = targets[k2]
                if _core.gate_target_text(t) == "*":
                    out.append("*")
                    k2 += 1
                    t = targets[k2]
                elif k2 > 0 or not args:
                    out.append("_")
                out.append(_core.gate_target_text(t))
                k2 += 1

    write(call(_core.circuit_items, circuit._stim_exact_text(), False), marks)
    out.append("_")
    return "".join(out)


def _detecting_regions(circuit: "Circuit", targets: Any, ticks: Any, ignore: bool) -> dict:
    text = circuit._stim_exact_text()
    num_dets, num_obs = call(_core.circuit_det_obs_counts, text)
    wanted: set = set()
    if targets is None:
        wanted.update((False, k) for k in range(num_dets))
        wanted.update((True, k) for k in range(num_obs))
    else:
        coords = None
        for f in targets:
            if isinstance(f, _dem.DemTarget) or type(f).__name__ == "DemTarget":
                d = _dem.DemTarget(f)
                wanted.add((d.is_logical_observable_id(), d.val))
                continue
            if isinstance(f, str):
                if f == "D":
                    wanted.update((False, k) for k in range(num_dets))
                elif f == "L":
                    wanted.update((True, k) for k in range(num_obs))
                elif f.startswith(("D", "L")):
                    d = _dem.DemTarget(f)
                    wanted.add((d.is_logical_observable_id(), d.val))
                else:
                    raise ValueError(f"Don't know how to interpret '{f!r}' as a dem target filter.")
                continue
            try:
                items = list(f)
            except TypeError:
                items = None
            if items is None or not all(isinstance(e, (int, float)) for e in items):
                raise ValueError(f"Don't know how to interpret '{f!r}' as a dem target filter.")
            prefix = [float(e) for e in items]
            if coords is None:
                coords = circuit.get_detector_coordinates()
            for d, c in coords.items():
                if len(c) >= len(prefix) and all(prefix[k] == c[k] for k in range(len(prefix))):
                    wanted.add((False, d))
    if ticks is None:
        tick_list = list(range(circuit.num_ticks))
    else:
        tick_list = sorted({int(t) for t in ticks})
    raw = call(_core.circuit_detecting_regions, text, sorted(wanted), tick_list, ignore)
    out: dict = {}
    for is_obs, index, tick, xs, zs in raw:
        key = _dem.target_logical_observable_id(index) if is_obs else _dem.target_relative_detector_id(index)
        out.setdefault(key, {})[tick] = _stim.PauliString.from_numpy(xs=np.array(xs, dtype=np.bool_), zs=np.array(zs, dtype=np.bool_))
    return out


_SAT_FALSE = (1 << 64) - 2
_SAT_TRUE = (1 << 64) - 1
_SAT_HARD = -1.0


def _cround(x: float) -> int:
    """C's round: halves away from zero."""
    f = math.floor(x)
    return int(f + 1 if x - f >= 0.5 else f)


def _wcnf(dem: "DetectorErrorModel", weighted: bool, quantization: int) -> str:
    """Stim's ``sat_problem_as_wcnf_string``: literals are (variable, negated)."""
    num_observables = dem.num_observables
    num_detectors = dem.num_detectors
    errors = dem._d.flat_errors()
    if num_observables == 0 or not errors:
        return "p wcnf 1 2 3\n3 -1 0\n3 1 0\n"
    num_variables = 0
    max_weight = 0.0
    clauses: list = []

    def add(lits: list, weight: float) -> None:
        nonlocal max_weight
        if weight != _SAT_HARD:
            if weight <= 0:
                raise ValueError("Clauses must have positive weight or HARD_CLAUSE_WEIGHT.")
            max_weight = max(max_weight, weight)
        clauses.append((lits, weight))

    def new_bool() -> tuple:
        nonlocal num_variables
        num_variables += 1
        return (num_variables - 1, False)

    def neg(x: tuple) -> tuple:
        return (x[0], not x[1])

    def xor(x: tuple, y: tuple) -> tuple:
        if x[0] == _SAT_FALSE:
            return y
        if x[0] == _SAT_TRUE:
            return neg(y)
        if y[0] == _SAT_FALSE:
            return x
        if y[0] == _SAT_TRUE:
            return neg(x)
        z = new_bool()
        add([x, y, neg(z)], _SAT_HARD)
        add([x, neg(y), z], _SAT_HARD)
        add([neg(x), y, z], _SAT_HARD)
        add([neg(x), neg(y), neg(z)], _SAT_HARD)
        return z

    activated = [new_bool() for _ in errors]
    dets = [(_SAT_FALSE, False)] * num_detectors
    obs = [(_SAT_FALSE, False)] * num_observables
    for (p, targets), x in zip(errors, activated):
        if weighted and p == 0:
            continue
        for is_obs, k in targets:
            if is_obs:
                obs[k] = xor(obs[k], x)
            else:
                dets[k] = xor(dets[k], x)
        if weighted:
            if p < 0.5:
                add([neg(x)], -math.log(p / (1 - p)))
            elif p > 0.5:
                add([x], -math.log((1 - p) / p))
        else:
            add([neg(x)], 1.0)
    for d in dets:
        if d[0] != _SAT_FALSE:
            add([neg(d)], _SAT_HARD)
    add(list(obs), _SAT_HARD)
    top = 1 + (quantization * len(clauses) if weighted else len(clauses))
    lines = [f"p wcnf {num_variables} {len(clauses)} {top}\n"]
    for lits, weight in clauses:
        if weight == _SAT_HARD:
            qw = top
        elif not weighted:
            qw = 1
        else:
            qw = _cround(weight / max_weight * quantization)
        if qw == 0:
            continue
        lines.append(str(qw) + "".join(f" -{(v + 1) % (1 << 64)}" if n else f" {(v + 1) % (1 << 64)}" for v, n in lits) + " 0\n")
    return "".join(lines)

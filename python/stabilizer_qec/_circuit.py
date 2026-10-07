"""Circuits, detector error models, sampling and measurement conversion."""

from __future__ import annotations

from os import PathLike
from typing import Any, Union

import numpy as np

from . import _core
from ._util import b8_to_rows, call, count, pack_rows, probability, real, rows_to_b8, seed_of, stride, text_of


class Circuit:
    """A stabilizer circuit in Stim's circuit language.

    Build one from Stim text, a ``stim.Circuit``, or another ``Circuit``; ``str()`` gives the
    text back as written. Every Clifford gate of Stim's is read (H, S and CX natively, the
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

    def __init__(self, text: Union[str, "Circuit", Any] = "") -> None:
        self._c = call(_core.Circuit, text_of(text, "Circuit", Circuit))

    @classmethod
    def from_file(cls, path: Union[str, PathLike]) -> "Circuit":
        """Read a ``.stim`` file."""
        with open(path, encoding="utf-8") as f:
            return cls(f.read())

    def __str__(self) -> str:
        return self._c.__str__()

    def __repr__(self) -> str:
        return f"stabilizer_qec.Circuit({str(self)!r})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Circuit):
            return NotImplemented
        return self._c.__eq__(other._c)

    def __reduce__(self) -> tuple:
        # Pickled, copied and sent to other processes as its text.
        return (Circuit, (str(self),))

    def copy(self) -> "Circuit":
        """A copy, to change without changing this one."""
        out = Circuit.__new__(Circuit)
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
            else:
                call(self._c.append_text, _stim_text(name))
            return
        if not isinstance(tag, str):
            raise TypeError(f"tag must be a str, not {type(tag).__name__}")
        call(self._c.append_instruction, name, tag, _args(arg), _targets(targets))

    def append_from_stim_program_text(self, text: str) -> None:
        """Append Stim's circuit text: any number of instructions and ``REPEAT`` blocks."""
        if not isinstance(text, str):
            raise TypeError(f"text must be a str, not {type(text).__name__}")
        call(self._c.append_text, text)

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

    def __mul__(self, repetitions: object) -> "Circuit":
        """``REPEAT repetitions { self }``, as Stim's: nothing for 0, the circuit for 1."""
        if isinstance(repetitions, bool) or not isinstance(repetitions, (int, np.integer)):
            return NotImplemented
        out = Circuit.__new__(Circuit)
        out._c = call(self._c.repeated, count(repetitions, "repetitions"))
        return out

    __rmul__ = __mul__

    def __imul__(self, repetitions: object) -> "Circuit":
        out = self.__mul__(repetitions)
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
        ignore_decomposition_failures: bool = False,
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
        cannot be split into graph-like pieces is kept whole rather than refused."""
        if isinstance(approximate_disjoint_errors, bool):
            threshold = 1.0 if approximate_disjoint_errors else None
        else:
            threshold = real(approximate_disjoint_errors, "approximate_disjoint_errors")
        return DetectorErrorModel._wrap(call(self._c.detector_error_model, bool(decompose_errors), threshold, bool(flatten_loops), bool(ignore_decomposition_failures)))

    def diagram(self, type: str = "timeline-text", *, tick: Union[int, None] = None) -> "Diagram":
        """A picture of the circuit, after Stim's ``diagram``:

        - ``"timeline-text"``, ``"timeline-svg"``: every operation in its column, ``TICK``
          groups bracketed, loops drawn once with their count, measurements numbered.
        - ``"detslice-text"``, ``"detslice-svg"``: what each detector compares after ``tick``
          ``TICK``s, its Paulis over the qubits (drawn at their ``QUBIT_COORDS``).
        - ``"matchgraph-svg"``: the decomposed model's matching graph.

        The result prints as its text, and shows as a picture in a notebook.

        >>> c = Circuit("R 0 1\\nH 0\\nTICK\\nCX 0 1\\nTICK\\nM 0 1\\nDETECTOR rec[-1] rec[-2]")
        >>> print(c.diagram("detslice-text", tick=2))
        D0: Z0 Z1
        <BLANKLINE>
        """
        if not isinstance(type, str):
            raise TypeError(f"type must be a str, not {builtins_type(type).__name__}")
        t = None if tick is None else count(tick, "tick")
        return Diagram(call(self._c.diagram, type, t), type)

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
    ) -> "Circuit":
        """One of Stim's generated memory experiments, character for character as
        ``stim.Circuit.generated`` writes it, with the same arguments. ``code_task`` is
        ``"repetition_code:memory"``, ``"surface_code:rotated_memory_x"`` (or ``_z``),
        ``"surface_code:unrotated_memory_x"`` (or ``_z``), or ``"color_code:memory_xyz"``.

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
        >>> [t.dem_target for t in e.dem_error_terms], e.circuit_error_locations[0].instruction_targets.gate
        (['D0'], 'X_ERROR')
        """
        from . import _explain

        f = None if dem_filter is None else (dem_filter if isinstance(dem_filter, DetectorErrorModel) else DetectorErrorModel(dem_filter))
        raw = call(self._c.explain, None if f is None else f._d, bool(reduce_to_one_representative_error))
        return _explain.build(raw)

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

    def compile_detector_sampler(self, *, seed: Union[int, None] = None) -> "DetectorSampler":
        """A sampler of detection events and observable flips. The same seed gives the same
        shots on any machine and any number of threads; ``None`` draws a seed."""
        return DetectorSampler(self, seed)

    def compile_m2d_converter(self) -> "MeasurementsToDetectionEventsConverter":
        """A converter from raw measurements (and sweep bits) to detection events, as
        ``stim m2d``. Loops are run pass by pass, never unrolled into memory, so a circuit of
        millions of rounds converts. Its reference run keeps a dense tableau: at most 16,384
        qubits."""
        return MeasurementsToDetectionEventsConverter(self)


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
    try:
        items = list(targets)
    except TypeError:
        raise TypeError(f"targets must be a target or a list of them, not {type(targets).__name__}") from None
    return [_target(t) for t in items]


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


def _stim_text(obj: Any) -> str:
    """A stim.Circuit, CircuitInstruction or CircuitRepeatBlock as circuit text."""
    if hasattr(obj, "repeat_count") and hasattr(obj, "body_copy"):
        tag = getattr(obj, "tag", "")
        return f"REPEAT{f'[{tag}]' if tag else ''} {obj.repeat_count} {{\n{obj.body_copy()}\n}}"
    if type(obj).__module__.startswith("stim"):
        return str(obj)
    raise TypeError(f"cannot append a {type(obj).__name__}: give an instruction's name, a Circuit, or a Stim circuit, instruction or repeat block")


import builtins as _builtins

builtins_type = _builtins.type


class Diagram:
    """A circuit's or model's picture (``Circuit.diagram``): text, or SVG. ``str()`` gives it
    as written; a notebook shows an SVG one as a picture."""

    __slots__ = ("_text", "type")

    def __init__(self, text: str, type: str) -> None:
        self._text = text
        self.type = type

    def __str__(self) -> str:
        return self._text

    def __repr__(self) -> str:
        return f"stabilizer_qec.Diagram(type={self.type!r}, {len(self._text)} characters)"

    def _repr_svg_(self) -> Union[str, None]:
        return self._text if self._text.startswith("<svg") else None

    def _repr_pretty_(self, p: Any, cycle: bool) -> None:
        p.text(self._text)

    def save(self, path: Union[str, PathLike]) -> None:
        """Write it to a file (an ``.svg`` for the pictures)."""
        with open(path, "w", encoding="utf-8") as f:
            f.write(self._text)


class DetectorErrorModel:
    """A detector error model in Stim's format: independent faults, each with a probability,
    the detectors it flips and the observables it flips.

    Build one from Stim text, a ``stim.DetectorErrorModel``, or ``Circuit.detector_error_model``.
    It is held as written, ``repeat`` blocks and all, and counted through its loops; a decoder
    made from it unrolls it once.
    """

    __slots__ = ("_d",)

    def __init__(self, text: Union[str, "DetectorErrorModel", Any] = "") -> None:
        self._d = call(_core.Dem, text_of(text, "DetectorErrorModel", DetectorErrorModel))

    @classmethod
    def _wrap(cls, inner: Any) -> "DetectorErrorModel":
        dem = cls.__new__(cls)
        dem._d = inner
        return dem

    @classmethod
    def from_file(cls, path: Union[str, PathLike]) -> "DetectorErrorModel":
        """Read a ``.dem`` file."""
        with open(path, encoding="utf-8") as f:
            return cls(f.read())

    def __str__(self) -> str:
        return self._d.__str__()

    def __repr__(self) -> str:
        return f"stabilizer_qec.DetectorErrorModel({str(self)!r})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DetectorErrorModel):
            return NotImplemented
        return str(self) == str(other)

    def __reduce__(self) -> tuple:
        return (DetectorErrorModel, (str(self),))

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
        """A picture of the model: ``"matchgraph-svg"``, its matching graph (it must be
        decomposed), each detector at its coordinates, each graph-like fault an edge."""
        if type != "matchgraph-svg":
            raise ValueError(f"a model's diagram is matchgraph-svg, not {type!r}")
        return Diagram(call(self._d.matchgraph_svg), type)

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


class DetectorSampler:
    """Detection events and observable flips, 64 shots to a machine word. Made by
    ``Circuit.compile_detector_sampler``."""

    __slots__ = ("_s",)

    def __init__(self, circuit: Circuit, seed: Union[int, None] = None) -> None:
        if not isinstance(circuit, Circuit):
            circuit = Circuit(circuit)
        self._s = call(circuit._c.sampler, seed_of(seed))

    def __reduce__(self) -> tuple:
        raise TypeError(
            "a DetectorSampler is a position in a stream of shots, which a copy would restart; "
            "send the circuit and a seed, and compile a sampler where it is used"
        )

    @property
    def num_detectors(self) -> int:
        return self._s.num_detectors

    @property
    def num_observables(self) -> int:
        return self._s.num_observables

    def sample(
        self,
        shots: int,
        *,
        separate_observables: bool = False,
        append_observables: bool = False,
        bit_packed: bool = False,
        threads: int = 1,
    ) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray]]:
        """``shots`` shots, as Stim's ``DetectorSampler.sample`` gives them: detection events
        of shape (shots, detectors), bool, or (shots, ⌈detectors/8⌉) uint8 when ``bit_packed``
        (bit k of a row in byte k // 8, position k % 8). ``separate_observables`` returns
        ``(detections, observables)``; ``append_observables`` puts the observables after each
        row's detectors. ``threads = 0`` uses every core; the shots do not depend on it.

        Successive calls continue the stream. Shots are drawn 64 at a time, and a call that
        ends partway through 64 discards the rest of them.
        """
        shots = count(shots, "shots")
        threads = count(threads, "threads")
        if separate_observables and append_observables:
            raise ValueError("choose separate_observables or append_observables, not both")
        nd, no = self.num_detectors, self.num_observables
        d, o = call(self._s.sample, shots, threads)
        if append_observables:
            rows = np.concatenate([b8_to_rows(d, shots, nd, False), b8_to_rows(o, shots, no, False)], axis=1)
            return pack_rows(rows, bit_packed)
        dets = b8_to_rows(d, shots, nd, bit_packed)
        if separate_observables:
            return dets, b8_to_rows(o, shots, no, bit_packed)
        return dets


class MeasurementsToDetectionEventsConverter:
    """Raw measurement records to detection events and observable flips, as ``stim m2d``:
    each detector and observable compared with a noiseless reference run. Made by
    ``Circuit.compile_m2d_converter``."""

    __slots__ = ("_m", "_circuit")

    def __init__(self, circuit: Circuit) -> None:
        if not isinstance(circuit, Circuit):
            circuit = Circuit(circuit)
        self._m = call(circuit._c.m2d)
        self._circuit = circuit

    def __reduce__(self) -> tuple:
        return (MeasurementsToDetectionEventsConverter, (self._circuit,))

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
        if append_observables:
            rows = np.concatenate([b8_to_rows(d, shots, nd, False), b8_to_rows(o, shots, no, False)], axis=1)
            return pack_rows(rows, bit_packed)
        dets = b8_to_rows(d, shots, nd, bit_packed)
        if separate_observables:
            return dets, b8_to_rows(o, shots, no, bit_packed)
        return dets

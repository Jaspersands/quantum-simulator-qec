"""Circuits, detector error models, sampling and measurement conversion."""

from __future__ import annotations

from os import PathLike
from typing import Any, Union

import numpy as np

from . import _core, _stim
from ._util import b8_to_rows, call, count, pack_rows, probability, real, rows_to_b8, seed_of, stride, text_of


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

    def __init__(self, text: Union[str, "Circuit", Any] = "") -> None:
        self._c = call(_core.Circuit, text_of(text, "Circuit", Circuit))

    @classmethod
    def from_file(cls, file: Union[str, PathLike, Any]) -> "Circuit":
        """Read a ``.stim`` file, from a path or an open file."""
        if hasattr(file, "read"):
            return cls(file.read())
        with open(file, encoding="utf-8") as f:
            return cls(f.read())

    def __str__(self) -> str:
        """Stim's text: canonical names, neighbouring instructions fused, arguments to 6
        significant digits (the values themselves are kept exactly)."""
        return _core.circuit_stim_text(self._c.__str__())

    def _stim_exact_text(self) -> str:
        """The text with every argument exact, which parses back to an equal circuit."""
        return _core.circuit_exact_text(self._c.__str__())

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

    # Stim's list-of-instructions interface.
    def _items(self) -> list:
        return _core.circuit_items(self._c.__str__())

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
            return Circuit(_stim._items_exact_text(items[index_or_slice]))
        if isinstance(index_or_slice, bool) or not isinstance(index_or_slice, (int, np.integer)):
            raise TypeError(f"circuit indices must be integers or slices, not {type(index_or_slice).__name__}")
        k = int(index_or_slice)
        if k < 0:
            k += len(items)
        if not 0 <= k < len(items):
            raise IndexError(f"index {index_or_slice} is out of range for a circuit with {len(items)} items")
        return _stim._item_object(items[k])

    def _set_items(self, items: list) -> None:
        self._c = call(_core.Circuit, _stim._items_exact_text(items))

    def insert(self, index: int, operation: Any) -> None:
        """Insert an instruction, REPEAT block or circuit before ``index``."""
        items = self._items()
        k = int(index)
        if k < 0:
            k += len(items)
        if not 0 <= k <= len(items):
            raise IndexError(f"index {index} out of range")
        piece = _core.circuit_items(_stim_text(operation))
        self._set_items(items[:k] + piece + items[k:])

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
        return Circuit(call(_core.circuit_transform, self._stim_exact_text(), which))

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

    def inverse(self) -> "Circuit":
        """The unitary inverse: operations inverted and in reverse. Noise, measurements and
        resets have no inverse and raise ``ValueError``."""
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

    def to_file(self, file: Any) -> None:
        """Write the circuit's text to a path or an open text file."""
        text = str(self) + "\n"
        if hasattr(file, "write"):
            file.write(text)
        else:
            with open(file, "w", encoding="utf-8") as f:
                f.write(text)

    def append_operation(self, name: Any, targets: Any = (), arg: Any = None, *, tag: str = "") -> None:
        """Stim's older name for ``append``."""
        self.append(name, targets, arg, tag=tag)

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

    def twirled(self, *, merge: bool = False) -> "Circuit":
        """The circuit with each coherent rotation (``I_ERROR[R_Z(theta=θ)] q`` and the rest)
        replaced by its Pauli twirl, the Pauli with probability sin²θ: the model Stim and every
        decoder assume. With ``merge``, rotations that are the same fault in different places
        (either side of a gate that leaves them alone, say) become one with their angles added,
        sin²(Σθ): the coherence-aware model. Merging writes the circuit without loops."""
        out = Circuit.__new__(Circuit)
        out._c = call(self._c.twirled, bool(merge))
        return out

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

    def diagram(self, type: str = "timeline-text", *, tick: Union[int, range, None] = None, rows: Union[int, None] = None) -> "Diagram":
        """A picture of the circuit, after Stim's ``diagram``:

        - ``"timeline-text"``: Stim's text timeline, character for character: every operation in
          its moment, ``TICK`` groups boxed, loops drawn once with records, detectors and
          coordinates in terms of ``iter``.
        - ``"timeline-svg"``: the same as a picture.
        - ``"detslice-text"``, ``"detslice-svg"``: what each detector compares after ``tick``
          ``TICK``s, its Paulis over the qubits (drawn at their ``QUBIT_COORDS``).
        - ``"timeslice-svg"``: the operations of tick ``tick`` (or of each tick in a
          ``range``, a panel each, in ``rows`` rows) over the qubits' coordinates.
        - ``"detslice-with-ops-svg"``: the same, with the detector slice after them.
        - ``"matchgraph-svg"``: the decomposed model's matching graph.

        The result prints as its text, and shows as a picture in a notebook.

        >>> c = Circuit("R 0 1\\nH 0\\nTICK\\nCX 0 1\\nTICK\\nM 0 1\\nDETECTOR rec[-1] rec[-2]")
        >>> print(c.diagram("detslice-text", tick=2))
        D0: Z0 Z1
        <BLANKLINE>
        """
        if not isinstance(type, str):
            raise TypeError(f"type must be a str, not {builtins_type(type).__name__}")
        if isinstance(tick, range):
            if tick.step != 1 or len(tick) == 0:
                raise ValueError("tick must be a non-empty range with step 1")
            t, end = tick.start, tick.stop
        else:
            t, end = (None if tick is None else count(tick, "tick")), None
        r = None if rows is None else count(rows, "rows", 1)
        return Diagram(call(self._c.diagram, type, t, end, r), type)

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

    def reference_sample(self, *, bit_packed: bool = False) -> np.ndarray:
        """A noiseless run's measurement record, as Stim's: the reference each shot's flips are
        taken from (and the converter compares with). Random outcomes take fixed values."""
        n = self.num_measurements
        return b8_to_rows(call(self._c.reference_sample), 1, n, bit_packed)[0]

    def compile_sampler(self, *, skip_reference_sample: bool = False, seed: Union[int, None] = None) -> "MeasurementSampler":
        """A sampler of raw measurement records, as Stim's: a noiseless reference run with each
        shot's flips (with ``skip_reference_sample``, the flips alone). The same seed gives the
        same shots on any machine and number of threads; ``None`` draws a seed."""
        return MeasurementSampler(self, seed, skip_reference_sample)

    def compile_detector_sampler(self, *, seed: Union[int, None] = None) -> "DetectorSampler":
        """A sampler of detection events and observable flips. The same seed gives the same
        shots on any machine and any number of threads; ``None`` draws a seed."""
        return DetectorSampler(self, seed)

    def compile_exact_sampler(self, *, seed: Union[int, None] = None) -> "ExactSampler":
        """An exact sampler: the full state vector (up to 24 qubits in use), running every
        instruction, Clifford or not, including the tagged rotations, T, U3 and amplitude damping
        (``I_ERROR[R_Z(theta=0.01)] 0``, ``I[T] 0``). Noise channels are drawn as quantum
        trajectories. Its ``sample`` has the detector sampler's signature."""
        return ExactSampler(self, seed)

    def compile_coherent_sampler(self, *, order: int = 3, seed: Union[int, None] = None) -> "CoherentSampler":
        """A sampler for circuits with coherent errors (``I_ERROR[R_Z(theta=0.02)] 0``,
        ``II_ERROR[R_ZZ(theta=0.01)] 0 1``, any Pauli rotation) at any size: shots of the
        Pauli-twirled circuit, as the detector sampler draws them, each with a weight that puts
        back the interference the twirl leaves out. Weighted, the shots are distributed as the
        coherent circuit's: estimate a rate as ``(w * hit).sum() / w.sum()``. ``order`` is how
        many of the circuit's local interference generators a shot's sum multiplies together."""
        return CoherentSampler(self, order, seed)

    def compile_leakage_sampler(self, *, leaked_reads_one: bool = True, seed: Union[int, None] = None) -> "LeakageSampler":
        """A frame sampler that runs leakage, written as Stim-readable tags:
        ``I_ERROR[LEAK(p=…)] q`` (the qubit leaks), ``I_ERROR[SEEP(p=…)] q`` (a leaked qubit
        returns, to |0⟩ or |1⟩ at random) and ``II_ERROR[LEAK_TRANSPORT(p=…)] a b`` (leakage
        moves to the partner). A two-qubit gate with a leaked partner leaves the other qubit a
        uniformly random Pauli, as in Google's model; measuring a leaked qubit reads 1 (a coin
        flip with ``leaked_reads_one=False``); a reset returns it. Its ``sample`` also gives each
        measurement's herald: whether its qubit was leaked."""
        return LeakageSampler(self, leaked_reads_one, seed)

    def exact_distribution(self, *, max_branches: int = 1 << 20) -> dict:
        """Every outcome's exact probability, following every measurement outcome and noise
        branch: ``{(detection events, observable flips): probability}``, each a tuple of bools.
        For small circuits; more than ``max_branches`` branches is an error."""
        rows = call(self._c.exact_distribution, count(max_branches, "max_branches"))
        no = self.num_observables
        return {(tuple(d), tuple(bool(o >> k & 1) for k in range(no))): p for d, o, p in rows}

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
        text = self._d.__str__()
        return text[:-1] if text.endswith("\n") else text

    def __repr__(self) -> str:
        text = str(self)
        if not text:
            return "stabilizer_qec.DetectorErrorModel()"
        body = "\n".join("    " + line if line else line for line in text.split("\n"))
        return f"stabilizer_qec.DetectorErrorModel('''\n{body}\n''')"

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


class MeasurementSampler:
    """Raw measurement records, as ``stim.CompiledMeasurementSampler``. Made by
    ``Circuit.compile_sampler``."""

    def __init__(self, circuit: Circuit, seed: Union[int, None] = None, skip_reference_sample: bool = False) -> None:
        if not isinstance(circuit, Circuit):
            circuit = Circuit(circuit)
        self._s = call(circuit._c.measurement_sampler, seed_of(seed), bool(skip_reference_sample))

    def __reduce__(self) -> tuple:
        raise TypeError("a MeasurementSampler is a position in a stream of shots, which a copy would restart; send the circuit and a seed instead")

    @property
    def num_measurements(self) -> int:
        return self._s.num_measurements

    def sample(self, shots: int, *, bit_packed: bool = False, threads: int = 1) -> np.ndarray:
        """``shots`` measurement records: (shots, measurements) bool, or bit-packed uint8 rows."""
        shots = count(shots, "shots")
        return b8_to_rows(call(self._s.sample, shots, count(threads, "threads")), shots, self.num_measurements, bit_packed)


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
            d, o = call(self._s.replay, raw, n)
            shots, e = n, (b8_to_rows(raw, n, ne, bit_packed) if return_errors else None)
        else:
            d, o, e = call(self._s.sample, shots, count(threads, "threads"), bool(return_errors))
            e = b8_to_rows(e, shots, ne, bit_packed) if return_errors else None
        return b8_to_rows(d, shots, nd, bit_packed), b8_to_rows(o, shots, no, bit_packed), e


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


class ExactSampler(DetectorSampler):
    """Detection events and observable flips from the exact state vector. Made by
    ``Circuit.compile_exact_sampler``; ``sample`` is the detector sampler's. Each shot has its
    own random stream, so threads never change the shots and calls continue shot by shot."""

    __slots__ = ()

    def __init__(self, circuit: Circuit, seed: Union[int, None] = None) -> None:
        if not isinstance(circuit, Circuit):
            circuit = Circuit(circuit)
        self._s = call(circuit._c.exact_sampler, seed_of(seed))

    @property
    def num_qubits(self) -> int:
        """The qubits in use, which the state vector holds."""
        return self._s.num_qubits


class LeakageSampler:
    """Shots of a circuit with leakage, and which measurements found their qubit leaked. Made
    by ``Circuit.compile_leakage_sampler``."""

    __slots__ = ("_s", "_m")

    def __init__(self, circuit: Circuit, leaked_reads_one: bool = True, seed: Union[int, None] = None) -> None:
        if not isinstance(circuit, Circuit):
            circuit = Circuit(circuit)
        self._s = call(circuit._c.leakage_sampler, seed_of(seed), bool(leaked_reads_one))
        self._m = circuit.num_measurements

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
        of shape (shots, detectors), (shots, observables) and (shots, measurements), the last
        empty in width when the circuit has no leakage."""
        shots = count(shots, "shots")
        d, o, h = call(self._s.sample, shots, count(threads, "threads"))
        heralds = np.frombuffer(h, dtype=np.uint8).astype(bool)
        heralds = heralds.reshape(shots, -1) if heralds.size else np.zeros((shots, 0), dtype=bool)
        return b8_to_rows(d, shots, self.num_detectors, False), b8_to_rows(o, shots, self.num_observables, False), heralds


class CoherentSampler:
    """Weighted shots of a circuit with coherent errors. Made by
    ``Circuit.compile_coherent_sampler``."""

    __slots__ = ("_s",)

    def __init__(self, circuit: Circuit, order: int = 3, seed: Union[int, None] = None) -> None:
        if not isinstance(circuit, Circuit):
            circuit = Circuit(circuit)
        self._s = call(circuit._c.coherent_sampler, seed_of(seed), count(order, "order"))

    def __reduce__(self) -> tuple:
        raise TypeError("a CoherentSampler is a position in a stream of shots; send the circuit and a seed")

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
        """The local generators of the circuit's interference (sets of rotations no record
        tells apart) the sampler found."""
        return self._s.num_generators

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

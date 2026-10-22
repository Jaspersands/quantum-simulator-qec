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
from stabilizer_qec.tsim._native import canonical_gate, tsim_to_native
from stabilizer_qec.tsim.utils.program_text import (
    _NATIVE_LINE,
    _NATIVE_TAG,
    _is_daggered_pair,
    _is_tpp_radians,
    _native_args,
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


# The gates of Pauli products, which Tsim's graph builders can refuse where the unitary engine does not
# (with TPP's and R_PAULI's I[R_PAULI(...)] and the pair rotations' II[R_ZZ(...)], ``_lane_scan``).
_PRODUCT_GATES = frozenset({"SPP", "SPP_DAG", "MPP"})
_PAIR_ROTATIONS = frozenset({"R_XX", "R_YY", "R_ZZ"})
# The gates Tsim's builders write through the output qubit -2 (a reset, the operation, a measurement): the
# measurements of products and pads, and heralded noise.
_OUTPUT_QUBIT_GATES = frozenset({"MPP", "MXX", "MYY", "MZZ", "MPAD", "HERALDED_ERASE", "HERALDED_PAULI_CHANNEL_1"})
# The builders' messages for an instruction they cannot read (Tsim's ``parse_parametric_tag`` and gate
# dispatch), as opposed to their refusals of products and sweep bits.
_UNREAD_BY_TSIMS_BUILDERS = ("Could not parse instruction ", "Unknown gate: ")

# A rotation's tag renamed so that the core's inverse leaves it alone (an unknown tag on I is inert),
# the inverse's tag written by ``_rotations_inverted`` instead.
_AS_INVERTED = "~stabilizer_qec_inverted"


def _negated(number: str, zero: Union[str, None] = "0.0") -> str:
    """``-number`` as written (its digits kept); a zero is ``zero``, ``0.0`` as Tsim's
    ``float(-Fraction)`` gives it (``None``: the zero's sign changed too, ``0.0`` and ``-0.0``)."""
    number = number.strip()
    if zero is not None and float(number) == 0:
        return zero
    if number.startswith("-"):
        return number[1:]
    return "-" + (number[1:] if number.startswith("+") else number)


def _inverted_product(theta: str, letters: str, dagger: bool) -> str:
    """The arguments of the inverse of a stored product rotation (``theta`` its stored radians, ``letters``
    its letters in the inverse's order of its targets, ``dagger`` whether its arguments are in
    ``SPP_DAG``'s order). Stim inverts Tsim's ``SPP[R_PAULI(theta=a*pi)]`` to ``SPP_DAG[...]`` of the same
    a, its tag untouched: the radians negated exactly (a zero's sign too), the arguments' order (the
    ``SPP_DAG``) turned, the letters' case (the ``!``s) kept; a ``TPP``'s dagger is its sign."""
    negated = _negated(theta, zero=None)
    if not _is_tpp_radians(theta):
        dagger = not dagger
    return f"pauli={letters}, theta={negated}" if dagger else f"theta={negated}, pauli={letters}"


def _pairs_turned(targets: list) -> str:
    """An ``II``'s targets with each pair's two exchanged (`` 1 0 3 2`` for ``0 1 2 3``): the core's inverse,
    which reverses an ``II``'s pairs and keeps each pair's order, then reverses them all."""
    return "".join(f" {targets[i + 1]} {targets[i]}" for i in range(0, len(targets), 2))


def _rotations_inverted(text: str) -> Union[str, None]:
    """This package's text of a circuit with each rotation's tag (``I[R_Z(theta=t)]``,
    ``I[R_PAULI(theta=t, pauli=XZ)]``, ``II[R_ZZ(theta=t)]``, ``I[U3(...)]``) replaced by its
    inverse's, the angles negated as written (their further digits kept, so the half-turns read back
    are the negated ones, and ``R_PAULI(0.25)`` stays apart from ``TPP``), an ``R_PAULI``'s letters
    reversed (the inverse reverses the targets) and its ``SPP`` and ``SPP_DAG`` exchanged (Tsim's
    inverse of ``R_PAULI(0.3) X0`` is ``SPP_DAG[R_PAULI(theta=0.3*pi)] X0``), and ``U3(θ, φ, λ)`` as
    ``U3(−θ, −λ, −φ)``, under names the core's inverse leaves alone; ``None`` if the text already has
    such a name. A product rotation on ``II`` (one product over its targets, as on ``I``) and a pair
    rotation (``II[R_ZZ(theta=t)] 0 1 2 3``) keep their gate, their targets all reversed as an ``I``'s
    are (each pair's two exchanged here, the core reversing the pairs), so each inverse is one
    instruction, Tsim's inverse of the print (``SPP_DAG[R_PAULI(theta=a*pi)] Z3*Z2 Z1*Z0``, a pair
    rotation's ``SPP_DAG`` written with an empty argument first, ``II[R_ZZ(, theta=-t)] 3 2 1 0``), and
    the inverse of the inverse is the circuit."""
    if _AS_INVERTED in text:
        return None
    out = []
    for line in text.split("\n"):
        m = _NATIVE_LINE.match(line)
        t = _NATIVE_TAG.match(m.group(3)) if m else None
        args = _native_args(t.group(2)) if t and t.group(2) else None
        if not args:
            out.append(line)
            continue
        indent, gate, _, rest = m.groups()
        name, targets = t.group(1), rest.split()
        if gate == "I" and name in ("R_X", "R_Y", "R_Z") and set(args) == {"theta"}:
            tag = f"theta={_negated(args['theta'])}"
        elif gate == "II" and name in _PAIR_ROTATIONS and set(args) == {"theta"} and targets and len(targets) % 2 == 0:
            # Its SPP and SPP_DAG exchanged (the empty first argument; a TPP's dagger is its sign), as
            # ``_inverted_product`` exchanges a product rotation's.
            dagger = _is_daggered_pair(t.group(2)) != (not _is_tpp_radians(args["theta"]))
            tag = f"{', ' if dagger else ''}theta={_negated(args['theta'], zero=None)}"
            rest = _pairs_turned(targets)
        elif name == "R_PAULI" and set(args) == {"theta", "pauli"}:
            # The targets all reversed (an I's by the core, an II's turned pair by pair here first), the
            # letters reversed with them.
            tag = _inverted_product(args["theta"], args["pauli"][::-1], list(args) == ["pauli", "theta"])
            if gate == "II":
                rest = _pairs_turned(targets)
        elif gate == "I" and name == "U3" and set(args) == {"theta", "phi", "lambda"}:
            tag = f"theta={_negated(args['theta'])}, phi={_negated(args['lambda'])}, lambda={_negated(args['phi'])}"
        else:
            out.append(line)
            continue
        out.append(f"{indent}{gate}[{name}{_AS_INVERTED}({tag})]{rest}")
    return "\n".join(out)


def _product_lanes(factors: Iterable) -> set:
    """The qubits a Pauli product's factors ``(letter, qubit)`` leave in it after Tsim's cancellations
    (``parse._iter_pauli_products``: a letter twice on a qubit cancels, two different are the third), the
    qubits Tsim's builders give it lanes on: ``X0*Z1*Z1`` is qubit 0, ``Z0*Z0`` none."""
    held: dict = {}
    for letter, qubit in factors:
        if qubit not in held:
            held[qubit] = letter
        elif held[qubit] == letter:
            del held[qubit]
        else:
            held[qubit] = ({"X", "Y", "Z"} - {held[qubit], letter}).pop()
    return set(held)


def _products_of(targets: list) -> list:
    """An ``SPP``'s or ``MPP``'s raw targets as its products, each a list of ``(letter, qubit)``."""
    products: list = []
    joined = False
    for v in targets:
        t = _stim.GateTarget._raw(v)
        if t.is_combiner:
            joined = True
            continue
        if products and joined:
            products[-1].append((t.pauli_type, t.value))
        else:
            products.append([(t.pauli_type, t.value)])
        joined = False
    return products


def _lane_scan(circuit: Any) -> tuple:
    """``(needs_zx, lanes, targeted, products, repeated)``: whether a circuit's matrix
    needs Tsim's ZX tensor, the qubits Tsim's ZX graph of the circuit has a lane for (its
    ``first_vertex``, the qubits of ``to_matrix`` either way), the qubits its instructions target,
    whether it has what Tsim's graph builders can refuse where the engine does not, and whether a
    product rotation names a qubit twice.

    It needs the ZX tensor for any measurement or record target, or when a reset follows the first instruction that names a qubit and that
    instruction is not a reset (an operation, noise, an identity or coordinates: Tsim's graph gives the
    qubit an input, where the engine counts only an operation as one), when there is a ``DETECTOR``
    or an ``OBSERVABLE_INCLUDE`` (an X spider of Tsim's graph, standing alone when it XORs no
    measurement: a factor of 2 the engine has no part for; one of Pauli targets is Tsim's error), or a
    measurement through Tsim's output qubit -2 (``MPP``, ``MPAD``, heralded noise). The lanes are read
    as Tsim's builders read the instructions: each qubit an instruction targets, but a Pauli product's
    are those its product leaves after the builders' cancellations (``SPP X0*Z1*Z1`` names qubit 0,
    ``TPP Z0*Z0`` none) and a product rotation's identity letters name none (``pauli=X_``, ``I`` or
    ``_``), an ``MPAD``'s targets are bits, and -2 is the output qubit. What the builders can refuse
    is read from the instructions: a Pauli product (``SPP``, ``SPP_DAG``, ``MPP``, a product rotation
    ``I[R_PAULI(...)]`` or ``II[R_PAULI(...)]``, a pair rotation ``II[R_ZZ(...)]``) or a sweep bit; a
    tag that only names one (``H[SPP]``, the coherent error ``I_ERROR[R_PAULI(...)]``) is not one. A
    loop's body is read twice (its second pass shows a reset after the first pass's operations)."""
    first_is_reset: dict = {}
    flags: dict = {}
    lanes: set = set()
    found = [False, False, False]  # needs_zx, products, repeated

    def visit(items: list) -> None:
        for item in items:
            if item[0] == "repeat":
                for _ in range(min(item[1], 2)):
                    visit(item[3])
                continue
            name, tag, targets = item[1], item[2], item[4]
            if name in ("DETECTOR", "OBSERVABLE_INCLUDE"):
                found[0] = True
                continue
            named = None  # the instruction's lanes, when they are not the qubits it targets
            if name in _PRODUCT_GATES:
                found[1] = True
                named = set().union(*(_product_lanes(p) for p in _products_of(targets)))
            elif tag and name in ("I", "II"):
                t = _NATIVE_TAG.match(tag)
                if t and t.group(1) == "R_PAULI":
                    # One product over the targets on I or II alike, as the core reads it. It names a qubit
                    # twice when two of its X, Y and Z letters are on it (an identity letter, ``I`` or ``_``,
                    # names none).
                    found[1] = True
                    args = _native_args(t.group(2)) if t.group(2) else None
                    letters = args.get("pauli") if args else None
                    qubits = [_stim.GateTarget._raw(v).value for v in targets]
                    acted = [(p.upper(), q) for p, q in zip(letters, qubits) if p not in "Ii_"] if letters is not None and len(letters) == len(qubits) else [("Z", q) for q in qubits]
                    found[2] = found[2] or len({q for _, q in acted}) < len(acted)
                    named = _product_lanes(acted)
                elif t and t.group(1) in _PAIR_ROTATIONS and name == "II":
                    found[1] = True
            if name in _OUTPUT_QUBIT_GATES:
                found[0] = True
                named = (set() if name == "MPAD" else named if named is not None else {_stim.GateTarget._raw(v).value for v in targets}) | ({-2} if targets else set())
            if name not in flags:
                data = _stim.GateData(name)
                flags[name] = (bool(data.is_reset), bool(data.produces_measurements))
            reset, measured = flags[name]
            found[0] = found[0] or measured
            for v in targets:
                if _stim.GateTarget._raw(v).is_measurement_record_target:
                    found[0] = True
                if v & _stim._TARGET_SWEEP:
                    found[1] = True
                q = _stim.GateTarget._raw(v).qubit_value
                if q is None:
                    continue
                if named is None:
                    lanes.add(q)
                if q not in first_is_reset:
                    first_is_reset[q] = reset
                elif reset and not first_is_reset[q]:
                    found[0] = True
            if named is not None:
                lanes.update(named)

    visit(circuit._items())
    return found[0], lanes, set(first_is_reset), found[1], found[2]


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
        """Return a string representation that can recreate the circuit (Tsim's shorthand). A circuit
        read from Tsim's text comes back equal; one held in this package's stored form comes back
        with the same operations, not always the same instructions (for example, an all-identity
        product rotation and a product's identity letters are left out, II's products come back on
        I, and radians come back with the half-turns' digits); see "Where it differs" in
        ``stabilizer_qec.tsim``."""
        return f"tsim.Circuit('''\n{self!s}\n''')"

    def __reduce__(self) -> tuple:
        return (_of_native_text, (type(self), self._stim_exact_text()))

    def _transformed(self, which: str) -> "Circuit":
        text = self._stim_exact_text()
        if which == "inverse":
            # The rotations are inverted here, keeping the half-turns they were written in (the core
            # would write each negated angle's shortest digits); the core reverses the order.
            inverted = _rotations_inverted(text)
            if inverted is not None:
                out = call(_core.circuit_transform, inverted, which)
                return type(self)._of_native(out.replace(f"{_AS_INVERTED}(", "("))
        return type(self)._of_native(call(_core.circuit_transform, text, which))

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
        name = canonical_gate(name)
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
        circuit's gates apply to the qubits Tsim's ZX graph has a lane for, those its instructions
        name, noise, identities and coordinates among them (a qubit no instruction names is dropped,
        so ``R_ZZ(0.5) 0 2`` is 4 × 4, and so is one that only a product that cancels on it names,
        ``TPP Z0*Z0`` being [[1]] as in Tsim and ``SPP X0*Z1*Z1`` 2 × 2, or only a product
        rotation's identity letter, ``pauli=X_``), qubit 0 the most significant bit of an index
        (``endian="little"``: the least). A qubit whose first operation is a reset (``R``,
        ``RX``, ``RY``) starts in that state and has no input: a circuit that resets every qubit
        first gives its state, a 2^n × 1 column (up to 22 qubits), and one that resets some a
        2^n × 2^k matrix. Noise channels are skipped (no error drawn), and coordinates and
        ``TICK`` are nothing. At most 2^22 entries (a unitary of 11 qubits).

        A circuit with a measurement, a detector or observable, a reset after a qubit's first
        operation, feedback or a product rotation naming a qubit twice (``TPP X0*Y0*X0``) is Tsim's
        ZX tensor of it (``tsim.zx.tensorfy``), contracted in
        numpy with every measurement outcome and error bit read as 0: a measurement is the
        identity on its qubit times 1/sqrt(2) (``H 0`` then ``M 0`` is [[.5, .5], [.5, -.5]]), a
        detector or observable is an X spider joined to the measurements it XORs, projecting their
        parity even (one that XORs none stands alone, a factor of 2: ``DETECTOR`` is [[2]],
        ``OBSERVABLE_INCLUDE(k)`` makes the observables 0 to k, so ``OBSERVABLE_INCLUDE(2)`` is
        [[8]]), a reset projects (and prepares), ``MPP`` and heralded noise add an output qubit
        (the first, qubit -2 in Tsim's graph), feedback is the controlled gate. Its qubits are the
        ZX graph's: the inputs of the qubits first used by a gate, the outputs of every qubit.

        Tsim's graph builders refuse what the unitary engine would run (an ``R_PAULI`` naming a
        qubit twice or more than 64, an anti-Hermitian product, a sweep bit): a circuit with Pauli
        products (``SPP``, ``MPP``, ``TPP``, ``R_PAULI``, ``R_XX``, ...) or sweep bits has its graph
        built first, so it raises Tsim's error. That graph is built in Python over the circuit with
        its loops unrolled, one step per instruction, before the engine runs: it costs seconds where
        the engine takes milliseconds for tens of thousands of instructions (``REPEAT 20000`` of
        ``SPP X0*Z1`` and ``H 0``). Tsim's own ``to_matrix`` builds and contracts that graph, and takes
        far longer. A circuit without products or sweep bits does not build it, and one with an
        instruction the builders cannot read (a tag the core leaves inert, ``I[foo(x)] 0``) is the
        engine's matrix, as without products (Tsim's raises its parse error), unless a product
        rotation names a qubit twice: without a graph that raises Tsim's parse error too. Where a
        product leaves a qubit out of Tsim's graph (an identity letter's, ``pauli=X_``, or one
        ``SPP X0*X0`` cancels on), the engine's matrix leaves it out too, and is Tsim's matrix of the
        circuit without that instruction up to the engine's global phase."""
        if endian not in ("big", "little"):
            raise ValueError(f"endian must be 'big' or 'little', got {endian!r}")
        needs_zx, lanes, targeted, products, repeated = _lane_scan(self)
        if needs_zx:
            return self._zx_matrix(endian)
        built = None
        if products:
            # Tsim's matrix is of its ZX graph, whose builders refuse what the unitary engine runs: an
            # R_PAULI's repeated qubit or more than 64, an anti-Hermitian product, a sweep bit. Built for
            # those errors, Tsim's, raised first as in Tsim. A product that leaves out of the graph a qubit
            # no other instruction names (one it cancels on, ``TPP Z0*Z0``, or an identity letter's,
            # ``pauli=X_Z``) is the ZX tensor, Tsim's exactly, and so is a product rotation naming a qubit
            # twice (``TPP X0*Y0*X0``, -Y, a sign the engine does not keep).
            from stabilizer_qec.tsim.zx import parse_stim_circuit

            try:
                built = parse_stim_circuit(self, backend="own")
            except ValueError as exc:
                # An instruction the builders cannot read (a tag of someone else's, ``I[foo(x)]``, which
                # the core leaves inert) is left to the engine, as in a circuit without products, unless a
                # product rotation names a qubit twice: that is the ZX tensor's, and there is no graph, so
                # Tsim's error, as Tsim's ``to_matrix`` raises it.
                if not str(exc).startswith(_UNREAD_BY_TSIMS_BUILDERS) or repeated:
                    raise
            else:
                if repeated or lanes != targeted or set(built.first_vertex) != lanes:
                    return self._zx_matrix(endian, built)
        # The engine's qubits are those its operations act on; each lane of Tsim's graph is one too (a qubit
        # only noise, an identity, an II or an identity letter names is an input the circuit leaves alone,
        # Tsim's identity), so the two give the matrix of the same qubits.
        ideal = self.without_noise()
        if lanes:
            ideal = type(self)._of_native(f"I {' '.join(map(str, sorted(lanes)))}\n{ideal._stim_exact_text()}")
        try:
            n, k, data = call(ideal._c.tsim_matrix, endian == "little")
        except ValueError as exc:
            if "has no matrix" not in str(exc):
                raise
            return self._zx_matrix(endian, built)
        if n != len(lanes):  # the engine acts on a qubit Tsim's graph has no lane for
            return self._zx_matrix(endian, built)
        return np.frombuffer(data, dtype="<c16").reshape(1 << n, 1 << k).copy()

    def _zx_matrix(self, endian: str, built: Any = None) -> np.ndarray:
        """Tsim's ZX tensor of the circuit, of its graph ``built`` (``parse_stim_circuit``'s, built
        here if ``None``)."""
        from stabilizer_qec.tsim.zx import parse_stim_circuit, tensorfy

        if built is None:
            built = parse_stim_circuit(self, backend="own")
        g = built.graph.copy()
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

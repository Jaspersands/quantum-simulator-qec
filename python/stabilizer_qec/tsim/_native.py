"""Tsim's tagged Stim as this package's operations. Tsim writes T as ``S[T]`` and rotations in
half-turns (``I[R_Z(theta=a*pi)]`` is exp(-i a pi Z / 2), ``SPP[R_PAULI(theta=a*pi)] P`` is
exp(-i a pi P / 2), ``SPP[T] P`` exp(-i pi P / 8), ``I[U3(theta=a*pi, phi=b*pi, lambda=c*pi)]``
OpenQASM's U3 in half-turns); this package writes them ``I[T]``, ``I[R_Z(theta=t)]`` for
exp(-i t Z), ``I[R_PAULI(theta=t, pauli=...)]`` and ``I[U3(...)]`` in radians. Engines that read
Tsim's form as Stim does would run S for T; these are honoured as the operations they are.

The radians keep the half-turns written: ``repr`` of the float when its shortest decimal number of
half-turns is the one written, else the exact radians to the further digits that give it back
(``program_text._stored_radians``); a Pauli product's ``SPP[T]`` is +-pi/8 as ``repr`` writes it,
its ``SPP[R_PAULI(theta=0.25*pi)]`` +-pi/8 with further digits, so the two stay Tsim's two
instructions. A product's ``!``s are lower-case letters and a rotation's ``SPP_DAG`` its arguments
in the other order (``program_text._product_rotation``), so it is read back as the instruction it
was."""

from __future__ import annotations

import math
import re
from typing import List, Optional

from stabilizer_qec.tsim.utils.program_text import _product_rotation, _stored_radians

_LINE = re.compile(r"^(\s*)(S|S_DAG|SPP|SPP_DAG|I)\[([^\]\n]*)\]\s+(.*?)\s*$")
_GATE = re.compile(r"^(\s*)([A-Za-z][A-Za-z0-9_]*)(?=[\s\[(]|$)")
_ARG = re.compile(r"(\w+)\s*=\s*([^,()]+?)\s*\*\s*pi")


def canonical_gate(name: str) -> str:
    """Resolve Stim aliases before interpreting Tsim's tags; leave unknown names to the parser."""
    from stabilizer_qec._stim import gate_data

    try:
        return gate_data(name).name
    except IndexError:
        return name


def _literals(tag: str) -> dict:
    """A tag's angles in half-turns, as floats (``theta=0.3*pi`` is 0.3)."""
    return {k: float(v) for k, v in _ARG.findall(tag)}


def _is_t(tag: str) -> bool:
    return tag == "T" or tag.startswith("T:")


_PAULI_TARGET = re.compile(r"!?[XYZxyz][0-9]+$")


def _products(targets: str) -> Optional[List[tuple]]:
    """The products of an ``SPP`` line's targets (``!X0*Y1 Z2``, spaces around ``*`` allowed, letters in
    either case, as Stim reads them): ``(sign, letters, qubits)`` each, ``sign`` -1 for an odd number
    of ``!``, ``letters`` upper case but lower case where the factor has a ``!``. ``None`` if a target
    is not a Pauli target, for the parser to report as it reports ``SPP``'s."""
    out = []
    for group in re.sub(r"\s*\*\s*", "*", targets).split():
        sign, letters, qubits = 1.0, "", []
        for factor in group.split("*"):
            if not _PAULI_TARGET.match(factor):
                return None
            inverted = factor.startswith("!")
            if inverted:
                sign, factor = -sign, factor[1:]
            letters += factor[0].lower() if inverted else factor[0].upper()
            qubits.append(factor[1:])
        out.append((sign, letters, qubits))
    return out


def _product_rotation_or_none(args: dict, targets: List[str]):
    """``program_text._product_rotation``, ``None`` for an argument that is not a number."""
    try:
        return _product_rotation(args, targets)
    except ValueError:
        return None


def _pair_rotation_or_none(name: str, body: Optional[str], args: dict, targets: List[str]):
    """``program_text._pair_rotation``, ``None`` for an argument that is not a number."""
    from stabilizer_qec.tsim.utils.program_text import _pair_rotation

    try:
        return _pair_rotation(name, body, args, targets)
    except ValueError:
        return None


def stim_reading(text: str) -> str:
    """This package's tagged operations as Stim reads Tsim's tagged form of them, the gate
    carrying the tag and the tag dropped: T is ``S`` (``S[T]``), T_DAG ``S_DAG``, a rotation or
    ``U3`` ``I`` (``I[R_Z(theta=a*pi)]``), a Pauli-product rotation (on ``I`` or ``II``) its Tsim
    instruction's gate on its product (``SPP[R_PAULI(theta=a*pi)] !X0`` is ``SPP !X0``,
    ``SPP_DAG[T] X0`` ``SPP_DAG X0``), a pair rotation (``II[R_ZZ(theta=t)] 0 1``) ``SPP Z0*Z1``
    (its ``SPP_DAG``, ``II[R_ZZ(, theta=t)]`` or ``TPP_DAG``, ``SPP_DAG Z0*Z1``).
    Tsim builds its detector error model of that reading; every other line is as it is."""
    from stabilizer_qec.tsim.utils.program_text import _NATIVE_LINE, _NATIVE_TAG, _native_args, _no_args, _pair_rotation, _product_text

    lines = []
    for line in text.split("\n"):
        m = _NATIVE_LINE.match(line)
        t = _NATIVE_TAG.match(m.group(3)) if m else None
        if not t:
            lines.append(line)
            continue
        indent, gate, _, rest = m.groups()
        name, body = t.group(1), t.group(2)
        targets = rest.split()
        args = _native_args(body) if body else {}
        # A product rotation on I or II alike, one product over the targets, as the core reads it.
        product = _product_rotation_or_none(args, targets) if name == "R_PAULI" and args and set(args) == {"theta", "pauli"} else None
        pair = _pair_rotation_or_none(name, body, args, targets) if gate == "II" and args else None
        if gate == "I" and _no_args(body) and name in ("T", "T_DAG"):
            lines.append(f"{indent}{'S' if name == 'T' else 'S_DAG'} {' '.join(targets)}")
        elif gate == "I" and name in ("R_X", "R_Y", "R_Z", "U3") and args:
            lines.append(f"{indent}I {' '.join(targets)}")
        elif product is not None:
            dagger, _, factors = product
            if factors:
                lines.append(f"{indent}{'SPP_DAG' if dagger else 'SPP'} {_product_text(factors)}")
        elif pair is not None:
            dagger, _, products = pair
            lines.append(f"{indent}{'SPP_DAG' if dagger else 'SPP'} {' '.join(_product_text(factors) for factors in products)}")
        else:
            lines.append(line)
    return "\n".join(lines)


def tsim_to_native(text: str) -> str:
    """Tsim's tagged Stim (``shorthand_to_stim``'s output) with its non-Clifford operations as
    this package's; every other line as it is."""
    lines = []
    for line in text.splitlines():
        body, sep, comment = line.partition("#")
        gate_match = _GATE.match(body)
        if gate_match:
            name = canonical_gate(gate_match.group(2))
            body = body[:gate_match.start(2)] + name + body[gate_match.end(2):]
            line = body + (sep + comment if sep else "")
        m = _LINE.match(body)
        if not m:
            lines.append(line)
            continue
        indent, gate, tag, targets = m.groups()
        name = tag.split("(", 1)[0].strip()
        if gate in ("S", "S_DAG") and _is_t(tag):
            lines.append(f"{indent}I[{'T' if gate == 'S' else 'T_DAG'}] {targets}")
        elif gate == "I" and name in ("R_X", "R_Y", "R_Z") and "*pi" in tag:
            lines.append(f"{indent}I[{name}(theta={_stored_radians(_literals(tag)['theta'], 0.5)})] {targets}")
        elif gate == "I" and name == "U3" and "*pi" in tag:
            a = {k: _stored_radians(v, 1.0) for k, v in _literals(tag).items()}
            lines.append(f"{indent}I[U3(theta={a['theta']}, phi={a['phi']}, lambda={a['lambda']})] {targets}")
        elif gate in ("SPP", "SPP_DAG") and (_is_t(tag) or name == "R_PAULI") and _products(targets) is not None:
            dagger = -1.0 if gate == "SPP_DAG" else 1.0
            turns = None if _is_t(tag) else _literals(tag)["theta"]
            for sign, letters, qubits in _products(targets):
                if turns is None:  # TPP: its dagger is its sign
                    arguments = f"theta={sign * dagger * math.pi / 8!r}, pauli={letters}"
                else:
                    theta = _stored_radians(sign * dagger * turns, 0.5, product=True)
                    arguments = f"pauli={letters}, theta={theta}" if gate == "SPP_DAG" else f"theta={theta}, pauli={letters}"
                lines.append(f"{indent}I[R_PAULI({arguments})] {' '.join(qubits)}")
        else:
            lines.append(line)
            continue
        if sep:
            lines[-1] += f" {sep}{comment}"
    return "\n".join(lines)

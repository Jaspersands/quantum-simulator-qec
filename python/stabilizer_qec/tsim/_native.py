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
instructions."""

from __future__ import annotations

import math
import re
from typing import List

from stabilizer_qec.tsim.utils.program_text import _stored_radians

_LINE = re.compile(r"^(\s*)(S|S_DAG|SPP|SPP_DAG|I)\[([^\]\n]*)\]\s+(.*?)\s*$")
_ARG = re.compile(r"(\w+)\s*=\s*([^,()]+?)\s*\*\s*pi")


def _literals(tag: str) -> dict:
    """A tag's angles in half-turns, as floats (``theta=0.3*pi`` is 0.3)."""
    return {k: float(v) for k, v in _ARG.findall(tag)}


def _is_t(tag: str) -> bool:
    return tag == "T" or tag.startswith("T:")


def _products(targets: str) -> List[tuple]:
    out = []
    for group in targets.split():
        sign, pauli, qubits = 1.0, "", []
        for factor in group.split("*"):
            if factor.startswith("!"):
                sign, factor = -sign, factor[1:]
            pauli += factor[0]
            qubits.append(factor[1:])
        out.append((sign, pauli, qubits))
    return out


def stim_reading(text: str) -> str:
    """This package's tagged operations as Stim reads Tsim's tagged form of them, the gate
    carrying the tag and the tag dropped: T is ``S`` (``S[T]``), T_DAG ``S_DAG``, a rotation or
    ``U3`` ``I`` (``I[R_Z(theta=a*pi)]``), a Pauli-product rotation ``SPP`` of its product
    (``SPP[R_PAULI(theta=a*pi)]``, ``SPP[T]``; ``SPP_DAG`` for a negative angle), a pair rotation
    (``II[R_ZZ(theta=t)] 0 1``) ``SPP Z0*Z1``. Tsim builds its detector error model of that
    reading; every other line is as it is."""
    from stabilizer_qec.tsim.utils.program_text import _NATIVE_LINE, _NATIVE_TAG, _native_args

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
        if gate == "I" and body is None and name in ("T", "T_DAG"):
            lines.append(f"{indent}{'S' if name == 'T' else 'S_DAG'} {' '.join(targets)}")
        elif gate == "I" and name in ("R_X", "R_Y", "R_Z", "U3") and args:
            lines.append(f"{indent}I {' '.join(targets)}")
        elif gate == "I" and name == "R_PAULI" and args and "pauli" in args and len(args["pauli"]) == len(targets):
            factors = [f"{p}{q}" for p, q in zip(args["pauli"].upper(), targets) if p != "I"]
            negative = args.get("theta", "").lstrip().startswith("-")
            if factors:
                lines.append(f"{indent}{'SPP_DAG' if negative else 'SPP'} {'*'.join(factors)}")
        elif gate == "II" and name in ("R_XX", "R_YY", "R_ZZ") and args and len(targets) % 2 == 0:
            p = name[2]
            products = " ".join(f"{p}{targets[i]}*{p}{targets[i + 1]}" for i in range(0, len(targets), 2))
            lines.append(f"{indent}SPP {products}")
        else:
            lines.append(line)
    return "\n".join(lines)


def tsim_to_native(text: str) -> str:
    """Tsim's tagged Stim (``shorthand_to_stim``'s output) with its non-Clifford operations as
    this package's; every other line as it is."""
    lines = []
    for line in text.splitlines():
        body, sep, comment = line.partition("#")
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
        elif gate in ("SPP", "SPP_DAG") and (_is_t(tag) or name == "R_PAULI"):
            dagger = -1.0 if gate == "SPP_DAG" else 1.0
            turns = None if _is_t(tag) else _literals(tag)["theta"]
            for sign, pauli, qubits in _products(targets):
                if turns is None:
                    theta = repr(sign * dagger * math.pi / 8)
                else:
                    theta = _stored_radians(sign * dagger * turns, 0.5, product=True)
                lines.append(f"{indent}I[R_PAULI(theta={theta}, pauli={pauli})] {' '.join(qubits)}")
        else:
            lines.append(line)
            continue
        if sep:
            lines[-1] += f" {sep}{comment}"
    return "\n".join(lines)

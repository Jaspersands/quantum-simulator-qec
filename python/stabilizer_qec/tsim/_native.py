"""Tsim's tagged Stim as this package's operations. Tsim writes T as ``S[T]`` and rotations in
half-turns (``I[R_Z(theta=a*pi)]`` is exp(-i a pi Z / 2), ``SPP[R_PAULI(theta=a*pi)] P`` is
exp(-i a pi P / 2), ``SPP[T] P`` exp(-i pi P / 8), ``I[U3(theta=a*pi, phi=b*pi, lambda=c*pi)]``
OpenQASM's U3 in half-turns); this package writes them ``I[T]``, ``I[R_Z(theta=t)]`` for
exp(-i t Z), ``I[R_PAULI(theta=t, pauli=...)]`` and ``I[U3(...)]`` in radians. Engines that read
Tsim's form as Stim does would run S for T; these are honoured as the operations they are."""

from __future__ import annotations

import math
import re
from typing import List

_LINE = re.compile(r"^(\s*)(S|S_DAG|SPP|SPP_DAG|I)\[([^\]\n]*)\]\s+(.*?)\s*$")
_ARG = re.compile(r"(\w+)\s*=\s*([^,()]+?)\s*\*\s*pi")


def _half_turns(tag: str) -> dict:
    return {k: float(v) * math.pi for k, v in _ARG.findall(tag)}


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
            lines.append(f"{indent}I[{name}(theta={_half_turns(tag)['theta'] / 2!r})] {targets}")
        elif gate == "I" and name == "U3" and "*pi" in tag:
            a = _half_turns(tag)
            lines.append(f"{indent}I[U3(theta={a['theta']!r}, phi={a['phi']!r}, lambda={a['lambda']!r})] {targets}")
        elif gate in ("SPP", "SPP_DAG") and (_is_t(tag) or name == "R_PAULI"):
            theta = math.pi / 8 if _is_t(tag) else _half_turns(tag)["theta"] / 2
            if gate == "SPP_DAG":
                theta = -theta
            for sign, pauli, qubits in _products(targets):
                lines.append(f"{indent}I[R_PAULI(theta={sign * theta!r}, pauli={pauli})] {' '.join(qubits)}")
        else:
            lines.append(line)
            continue
        if sep:
            lines[-1] += f" {sep}{comment}"
    return "\n".join(lines)

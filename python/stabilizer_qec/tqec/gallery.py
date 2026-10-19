# Ported from tqec (tqec/tqec 0.2.0, src/tqec/gallery/*.py; Apache License 2.0; see LICENSE in this directory),
# modified for stabilizer_qec: the CZ flows checked without stim; the Steane-encoding Collada file is tqec's,
# packaged here.
"""tqec's gallery of logical computations as block graphs."""

from __future__ import annotations

import pathlib
import re
from typing import Optional, Union

from ._errors import TQECError
from .block_graph import BlockGraph
from .cube import ZXCube
from .enums import Basis
from .position import Position3D

STEANE_CODE_DAE = pathlib.Path(__file__).parent / "dae" / "steane_encoding.dae"


def memory(observable_basis: Basis = Basis.Z) -> BlockGraph:
    """One cube: a logical memory experiment in the basis."""
    g = BlockGraph(f"Logical {observable_basis.value} Memory Experiment")
    g.add_cube(Position3D(0, 0, 0), f"ZX{observable_basis.value}")
    return g


def stability(observable_basis: Basis = Basis.Z) -> BlockGraph:
    """One cube: a stability experiment in the basis."""
    g = BlockGraph(f"{observable_basis.value} Basis Stability Experiment")
    g.add_cube(Position3D(0, 0, 0), "ZZX" if observable_basis == Basis.Z else "XXZ")
    return g


def cnot(observable_basis: Optional[Basis] = None) -> BlockGraph:
    """The logical CNOT: four ports (``In_Control``, ``Out_Control``, ``In_Target``,
    ``Out_Target``), open or filled with cubes in the observable basis."""
    g = BlockGraph("Logical CNOT")
    nodes = [
        (Position3D(0, 0, 0), "P", "In_Control"),
        (Position3D(0, 0, 1), "ZXX", ""),
        (Position3D(0, 0, 2), "ZXZ", ""),
        (Position3D(0, 0, 3), "P", "Out_Control"),
        (Position3D(0, 1, 1), "ZXX", ""),
        (Position3D(0, 1, 2), "ZXZ", ""),
        (Position3D(1, 1, 0), "P", "In_Target"),
        (Position3D(1, 1, 1), "ZXZ", ""),
        (Position3D(1, 1, 2), "ZXZ", ""),
        (Position3D(1, 1, 3), "P", "Out_Target"),
    ]
    for pos, kind, label in nodes:
        g.add_cube(pos, kind, label)
    for p0, p1 in [(0, 1), (1, 2), (2, 3), (1, 4), (4, 5), (5, 8), (6, 7), (7, 8), (8, 9)]:
        g.add_pipe(nodes[p0][0], nodes[p1][0])
    if observable_basis == Basis.Z:
        g.fill_ports(ZXCube.from_str("ZXZ"))
    elif observable_basis == Basis.X:
        g.fill_ports(ZXCube.from_str("ZXX"))
    return g


def cz(support_flows: Union[str, list[str], None] = None) -> BlockGraph:
    """The logical CZ (its Hadamard pipe horizontal; ``rotate`` for a vertical one): ports open,
    or filled to support the given stabilizer flows (``"XI -> XZ"``, or a list of them), in
    Stim's syntax: dense (``"X_ -> XZ"``), sparse (``"X0 -> X0*Z1"``), signed, ``1`` for the
    identity. An error names a flow as Stim writes it (``X_ -> XX is not a valid flow for the CZ
    gate.``), as tqec's does. A sparse flow is placed on the gate's qubits by its indices (tqec
    0.2.0 joins the two sides' letters, which misplaces the output ports of ``"X0 -> X0*Z1"``)."""
    g = BlockGraph("Logical CZ")
    nodes = [
        (Position3D(0, 0, 0), "P", "In_1"),
        (Position3D(0, 0, 1), "XZX", ""),
        (Position3D(0, 0, 2), "P", "Out_1"),
        (Position3D(1, 0, 1), "XXZ", ""),
        (Position3D(1, -1, 1), "P", "In_2"),
        (Position3D(1, 1, 1), "P", "Out_2"),
    ]
    for pos, kind, label in nodes:
        g.add_cube(pos, kind, label)
    for p0, p1 in [(0, 1), (1, 2), (1, 3), (3, 4), (3, 5)]:
        g.add_pipe(nodes[p0][0], nodes[p1][0])
    if support_flows is not None:
        flows = [f.upper() for f in support_flows] if isinstance(support_flows, list) else [support_flows.upper()]
        g.fill_ports(dict(zip(["In_1", "In_2", "Out_1", "Out_2"], _resolve_ports(flows))))
    return g


def _parse_flow_side(text: str) -> tuple[int, str]:
    """One side of a stabilizer flow as Stim reads it: ``(sign, letters)``, the letters dense
    with ``_`` for I (``"XI"``, ``"X_"``), from dense or sparse text (``"X0*Z1"``), ``""`` for
    the identity ``"1"``."""
    text = text.strip().replace(" ", "")
    sign = 1
    if text[:1] in "+-":
        sign = -1 if text[0] == "-" else 1
        text = text[1:]
    if text == "1":
        return sign, ""
    if any(c.isdigit() for c in text):
        letters: dict[int, str] = {}
        for term in text.split("*"):
            match = re.fullmatch(r"([XYZ])(\d+)", term)
            if match is None or int(match.group(2)) in letters:
                raise TQECError(f"{text!r} is not a Pauli string.")
            letters[int(match.group(2))] = match.group(1)
        return sign, "".join(letters.get(i, "_") for i in range(max(letters) + 1))
    letters_text = text.replace("I", "_")
    if not letters_text or any(c not in "_XYZ" for c in letters_text):
        raise TQECError(f"{text!r} is not a Pauli string.")
    return sign, letters_text


def _flow_text(sign_in: int, p_in: str, sign_out: int, p_out: str) -> str:
    """The flow as Stim writes it (``X_ -> XZ``): how tqec's errors name it."""
    return f"{'-' if sign_in < 0 else ''}{p_in or '1'} -> {'-' if sign_out < 0 else ''}{p_out or '1'}"


def _cz_conjugate(letters: str) -> str:
    """CZ P CZ, unsigned: X on one qubit picks up Z on the other."""
    x = [c in "XY" for c in letters]
    z = [c in "ZY" for c in letters]
    z[0] ^= x[1]
    z[1] ^= x[0]
    return "".join("_XZY"[int(xi) | (int(zi) << 1)] for xi, zi in zip(x, z))


def _resolve_ports(flows: list[str]) -> list[ZXCube]:
    if any("Y" in f for f in flows):
        raise TQECError("Y basis initialization/measurements are not supported yet.")
    parsed = []
    for f in flows:
        if "->" not in f:
            raise TQECError(f"{f} is not a valid flow for the CZ gate.")
        left, right = f.split("->", 1)
        sign_in, p_in = _parse_flow_side(left)
        sign_out, p_out = _parse_flow_side(right)
        text = _flow_text(sign_in, p_in, sign_out, p_out)
        # Both sides on the same qubits (a sparse side, "X0", is as short as its last qubit).
        n = max(len(p_in), len(p_out), 2)
        p_in, p_out = p_in.ljust(n, "_"), p_out.ljust(n, "_")
        # CZ maps every X/Z-only Pauli to its image with sign +, so a flow holds when the images
        # agree on the gate's qubits, the other qubits are left alone, and the signs agree.
        if _cz_conjugate(p_in[:2]) != p_out[:2] or p_in[2:] != p_out[2:] or sign_in != sign_out:
            raise TQECError(f"{text} is not a valid flow for the CZ gate.")
        if p_in[2:].strip("_"):
            raise TQECError(f"{text} acts on qubits the CZ gate does not have (its qubits are 0 and 1).")
        parsed.append(p_in[:2] + p_out[:2])
    ports = ["_"] * 4
    for p in parsed:
        for i, letter in enumerate(p):
            if ports[i] == "_":
                ports[i] = letter
            elif letter != "_" and letter != ports[i]:
                raise TQECError(f"Port {i} fails to support both {ports[i]} and {letter} observable.")
    for i in range(4):
        if ports[i] == "_":
            ports[i] = "Z"

    def vkind(p: str) -> ZXCube:
        return ZXCube.from_str(f"XZ{p}")

    def hkind(p: str) -> ZXCube:
        return ZXCube.from_str(f"X{p}Z")

    return [vkind(ports[0]), hkind(ports[1]), vkind(ports[2]), hkind(ports[3])]


def move_rotation(observable_basis: Optional[Basis] = None) -> BlockGraph:
    """Moving a logical qubit in spacetime to rotate its spatial boundaries."""
    g = BlockGraph("Move Rotation")
    nodes = [
        (Position3D(0, 0, 0), "P", "In"),
        (Position3D(0, 0, 1), "ZXX", ""),
        (Position3D(0, 1, 1), "ZZX", ""),
        (Position3D(1, 1, 1), "XZX", ""),
        (Position3D(1, 1, 2), "P", "Out"),
    ]
    for pos, kind, label in nodes:
        g.add_cube(pos, kind, label)
    for p0, p1 in [(0, 1), (1, 2), (2, 3), (3, 4)]:
        g.add_pipe(nodes[p0][0], nodes[p1][0])
    if observable_basis == Basis.Z:
        g.fill_ports({"In": ZXCube.from_str("ZXZ"), "Out": ZXCube.from_str("XZZ")})
    elif observable_basis == Basis.X:
        g.fill_ports({"In": ZXCube.from_str("ZXX"), "Out": ZXCube.from_str("XZX")})
    return g


def three_cnots(observable_basis: Optional[Basis] = None) -> BlockGraph:
    """Three logical CNOTs (q0→q1, q1→q2, q0→q2) compressed in spacetime."""
    g = BlockGraph("Three CNOTs")
    nodes = [
        (Position3D(-1, 0, 0), "P", "Out_a"),
        (Position3D(0, 0, 0), "XXZ", ""),
        (Position3D(0, -1, 0), "P", "In_a"),
        (Position3D(0, 1, 0), "XXZ", ""),
        (Position3D(1, 0, 0), "ZXZ", ""),
        (Position3D(1, 1, 0), "ZXZ", ""),
        (Position3D(1, 0, -1), "P", "In_b"),
        (Position3D(1, 1, -1), "P", "In_c"),
        (Position3D(1, 0, 1), "ZXX", ""),
        (Position3D(1, 1, 1), "ZXX", ""),
        (Position3D(2, 1, 0), "P", "Out_c"),
        (Position3D(1, 0, 2), "P", "Out_b"),
    ]
    for pos, kind, label in nodes:
        g.add_cube(pos, kind, label)
    for p0, p1 in [(1, 0), (1, 2), (1, 3), (1, 4), (4, 6), (4, 8), (5, 3), (5, 9), (5, 10), (5, 7), (8, 11), (8, 9)]:
        g.add_pipe(nodes[p0][0], nodes[p1][0])
    if observable_basis == Basis.Z:
        g.fill_ports({"In_a": ZXCube.from_str("XZZ"), "In_b": ZXCube.from_str("ZXZ"), "In_c": ZXCube.from_str("ZXZ"), "Out_a": ZXCube.from_str("ZXZ"), "Out_b": ZXCube.from_str("ZXZ"), "Out_c": ZXCube.from_str("ZXZ")})
    elif observable_basis == Basis.X:
        g.fill_ports({"In_a": ZXCube.from_str("XXZ"), "In_b": ZXCube.from_str("ZXX"), "In_c": ZXCube.from_str("ZXX"), "Out_a": ZXCube.from_str("XXZ"), "Out_b": ZXCube.from_str("ZXX"), "Out_c": ZXCube.from_str("XXZ")})
    return g


def steane_encoding(observable_basis: Optional[Basis] = None) -> BlockGraph:
    """The Steane encoding circuit compressed in spacetime, from tqec's Collada file."""
    graph = BlockGraph.from_dae_file(STEANE_CODE_DAE)
    if observable_basis is not None:
        graph.fill_ports({f"Port{port}": ZXCube(Basis.Z, Basis.X, observable_basis) for port in (3, 4)})
        graph.fill_ports({f"Port{port}": ZXCube(Basis.X, Basis.Z, observable_basis) for port in (1, 2, 5, 6)})
        graph.fill_ports(ZXCube(observable_basis, Basis.X, Basis.Z))
    return graph


BUILDERS = {"memory": memory, "stability": stability, "cnot": cnot, "cz": cz, "move_rotation": move_rotation, "three_cnots": three_cnots, "steane_encoding": steane_encoding}

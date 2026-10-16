# Ported from CUDA-Q QEC (NVIDIA/cudaqx, libs/qec: lib/code.cpp, lib/stabilizer_utils.cpp,
# lib/codes/{surface_code,steane,repetition}{,_device}.cpp, lib/device/memory_circuit.cpp,
# lib/experiments.cpp, include/cudaq/qec/noise_model.h and python/bindings/py_{code,surface_code}.cpp;
# Copyright (c) 2024 - 2026 NVIDIA Corporation & Affiliates, Apache License 2.0; see
# THIRD_PARTY_NOTICES.md), modified for stabilizer_qec: CUDA-Q's kernels are this package's
# circuits, its spin operators its PauliStrings, its noise models the ones below.
"""CUDA-Q QEC's codes: the surface code (with its orientations XV, XH, ZV, ZH, and Ising's
aliases O1-O4), Steane's and the repetition code, Python codes registered with ``@code``, and its
experiments on them (memory circuits sampled or made detector error models, code-capacity
samples)."""

from __future__ import annotations

import enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .._circuit import Circuit
from .._stim import PauliString
from ._dem import DetectorErrorModel, dem_from_stim_text


class operation(enum.IntEnum):
    """A code's logical operations (its encodings)."""

    x = 0
    y = 1
    z = 2
    h = 3
    s = 4
    cx = 5
    cy = 6
    cz = 7
    stabilizer_round = 8
    prep0 = 9
    prep1 = 10
    prepp = 11
    prepm = 12


class surface_role(enum.IntEnum):
    """A stabilizer-grid site's role: an X or a Z ancilla, or empty."""

    amx = 0
    amz = 1
    empty = 2


amx, amz, empty = surface_role.amx, surface_role.amz, surface_role.empty


class sc_orientation(enum.IntEnum):
    """The surface code's orientation (NVIDIA Ising-Decoding's names): the first letter is the
    Pauli type of the bulk's even sites, the second whether the horizontal (top and bottom)
    boundaries take the even sites (H) or the vertical ones (V). Ising's O1-O4 are XV, XH, ZV
    and ZH."""

    XV = 0
    XH = 1
    ZV = 2
    ZH = 3


XV, XH, ZV, ZH = sc_orientation.XV, sc_orientation.XH, sc_orientation.ZV, sc_orientation.ZH

_ALIASES = {"XV": XV, "O1": XV, "XH": XH, "O2": XH, "ZV": ZV, "O3": ZV, "ZH": ZH, "O4": ZH}


def sc_orientation_from_str(s: str) -> sc_orientation:
    """An orientation from its name (XV, XH, ZV, ZH, or O1-O4), ignoring case and surrounding
    space."""
    o = _ALIASES.get(str(s).strip().upper())
    if o is None:
        raise RuntimeError(f"Invalid surface-code orientation '{s}'. Expected XV(O1), XH(O2), ZV(O3), or ZH(O4).")
    return o


def _orientation(o: Any) -> sc_orientation:
    return sc_orientation(o) if isinstance(o, (sc_orientation, int)) and not isinstance(o, bool) else sc_orientation_from_str(o)


def role_to_str(r: surface_role) -> str:
    return "X" if r == amx else "Z" if r == amz else "e"


class vec2d:
    """A (row, col) grid coordinate."""

    def __init__(self, row: int, col: int) -> None:
        self.row, self.col = int(row), int(col)

    def __repr__(self) -> str:
        return f"Vec2D(row={self.row}, col={self.col})"

    def __add__(self, other: "vec2d") -> "vec2d":
        return vec2d(self.row + other.row, self.col + other.col)

    def __sub__(self, other: "vec2d") -> "vec2d":
        return vec2d(self.row - other.row, self.col - other.col)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, vec2d) and (self.row, self.col) == (other.row, other.col)

    def __lt__(self, other: "vec2d") -> bool:
        return (self.row, self.col) < (other.row, other.col)

    __hash__ = None  # type: ignore[assignment]


# Corner offsets from a stabilizer site to its data qubits, in CNOT order (timesteps 1-4):
# row-major visits NW, NE, SW, SE; column-major NW, SW, NE, SE.
_ROW_MAJOR = ((-1, -1), (-1, 0), (0, -1), (0, 0))
_COL_MAJOR = ((-1, -1), (0, -1), (-1, 0), (0, 0))


class stabilizer_grid:
    """The rotated surface code's layout: a (distance + 1)-square grid of stabilizer sites
    (``roles``, row-major) over the distance-square grid of data qubits (data qubit (r, c) is
    index r * distance + c, half a site down and right of site (r, c)). Each stabilizer touches
    the data qubits at its site's four corners."""

    def __init__(self, distance: int = 0, orientation: Any = ZH) -> None:
        self.distance = int(distance)
        self.grid_length = self.distance + 1 if distance else 0
        self._orientation = _orientation(orientation)
        self.roles: List[surface_role] = []
        self.x_stab_coords: List[vec2d] = []
        self.z_stab_coords: List[vec2d] = []
        self.data_coords: List[vec2d] = []
        self.x_stabilizers: List[List[int]] = []
        self.z_stabilizers: List[List[int]] = []
        self._x_idx: Dict[Tuple[int, int], int] = {}
        self._z_idx: Dict[Tuple[int, int], int] = {}
        self._d_idx: Dict[Tuple[int, int], int] = {}
        if distance:
            self._roles()
            self._indices()
            self._stabilizers()

    @property
    def orientation(self) -> sc_orientation:
        return self._orientation

    def _role(self, odd: bool) -> surface_role:
        x_even = self._orientation in (XV, XH)
        return amx if x_even != odd else amz

    def _roles(self) -> None:
        g = self.grid_length
        roles = [empty] * (g * g)
        for r in range(1, g - 1):
            for c in range(1, g - 1):
                roles[r * g + c] = self._role(bool((r + c) % 2))
        horizontal_even = self._orientation in (XH, ZH)
        for r in (0, g - 1):
            for c in range(1, g - 1):
                odd = bool((r + c) % 2)
                if odd != horizontal_even:
                    roles[r * g + c] = self._role(odd)
        for r in range(1, g - 1):
            for c in (0, g - 1):
                odd = bool((r + c) % 2)
                if odd == horizontal_even:
                    roles[r * g + c] = self._role(odd)
        self.roles = roles

    def _indices(self) -> None:
        g = self.grid_length
        for r in range(g):
            for c in range(g):
                role = self.roles[r * g + c]
                if role == amz:
                    self._z_idx[(r, c)] = len(self.z_stab_coords)
                    self.z_stab_coords.append(vec2d(r, c))
                elif role == amx:
                    self._x_idx[(r, c)] = len(self.x_stab_coords)
                    self.x_stab_coords.append(vec2d(r, c))
        d = self.distance
        for r in range(d):
            for c in range(d):
                self.data_coords.append(vec2d(r, c))
                self._d_idx[(r, c)] = r * d + c

    def _support(self, site: vec2d, offsets: Sequence[Tuple[int, int]] = _ROW_MAJOR) -> List[int]:
        return [self._d_idx[(site.row + dr, site.col + dc)] for dr, dc in offsets if (site.row + dr, site.col + dc) in self._d_idx]

    def _stabilizers(self) -> None:
        self.x_stabilizers = [sorted(self._support(s)) for s in self.x_stab_coords]
        self.z_stabilizers = [sorted(self._support(s)) for s in self.z_stab_coords]

    @property
    def x_stab_indices(self) -> Dict[Tuple[int, int], int]:
        return dict(sorted(self._x_idx.items()))

    @property
    def z_stab_indices(self) -> Dict[Tuple[int, int], int]:
        return dict(sorted(self._z_idx.items()))

    @property
    def data_indices(self) -> Dict[Tuple[int, int], int]:
        return dict(sorted(self._d_idx.items()))

    def _cells(self, fmt: Callable[[int, int, surface_role], str]) -> str:
        g = self.grid_length
        return "".join("".join(fmt(r, c, self.roles[r * g + c]) for c in range(g)) + "\n" for r in range(g)) + "\n"

    def format_stabilizer_grid(self) -> str:
        w = len(str(self.grid_length))
        return self._cells(lambda r, c, role: f"{role_to_str(role)}({r:>{w}},{c:>{w}})  ")

    def format_stabilizer_coords(self) -> str:
        w = len(str(self.grid_length))
        return self._cells(lambda r, c, role: " " * (2 * w + 6) if role == empty else f"{role_to_str(role)}({r:>{w}},{c:>{w}})  ")

    def format_stabilizer_indices(self) -> str:
        w = len(str(len(self._z_idx))) + 2

        def cell(r: int, c: int, role: surface_role) -> str:
            if role == empty:
                return " " * (w + 1)
            i = (self._z_idx if role == amz else self._x_idx)[(r, c)]
            return f"{role_to_str(role)}{i:<{w}}"

        return self._cells(cell)

    def format_data_grid(self) -> str:
        w = len(str(self.distance)) + 2
        d = self.distance
        return "".join("".join(f"d{r * d + c:<{w}}" for c in range(d)) + "\n" for r in range(d)) + "\n"

    def format_stabilizers(self) -> str:
        lines = [f"s[{i}]: " + "".join(f"X{q} " for q in s) for i, s in enumerate(self.x_stabilizers)]
        off = len(self.x_stabilizers)
        lines += [f"s[{i + off}]: " + "".join(f"Z{q} " for q in s) for i, s in enumerate(self.z_stabilizers)]
        return "".join(line + "\n" for line in lines) + "\n"

    def __repr__(self) -> str:
        return f"<StabilizerGrid distance={self.distance} grid_length={self.grid_length}>"

    def _words(self, stabs: List[List[int]], p: str) -> List[str]:
        n = len(self.data_coords)
        out = []
        for s in stabs:
            w = ["I"] * n
            for q in s:
                w[q] = p
            out.append("".join(w))
        return out

    def get_spin_op_stabilizers(self) -> List[PauliString]:
        """The stabilizers, X then Z, as PauliStrings (CUDA-Q's spin_op_terms)."""
        return [PauliString(w) for w in self._words(self.x_stabilizers, "X") + self._words(self.z_stabilizers, "Z")]

    def _x_logical_on_top_row(self) -> bool:
        return self._orientation in (XV, ZH)

    def _observable_words(self) -> List[str]:
        n, d = len(self.data_coords), self.distance
        top, left = ["I"] * n, ["I"] * n
        tx, lx = ("X", "Z") if self._x_logical_on_top_row() else ("Z", "X")
        for i in range(d):
            top[i] = tx
        for i in range(0, n, d):
            left[i] = lx
        top_w, left_w = "".join(top), "".join(left)
        return [top_w, left_w] if self._x_logical_on_top_row() else [left_w, top_w]

    def get_spin_op_observables(self) -> List[PauliString]:
        """The logical observables [X, Z]: for XV and ZH the X logical along the top row of data
        qubits and the Z logical down the left column; for XH and ZV the other way round."""
        return [PauliString(w) for w in self._observable_words()]

    def _schedule(self, sites: List[vec2d], col_major: bool) -> np.ndarray:
        n = self.distance * self.distance
        order = _COL_MAJOR if col_major else _ROW_MAJOR
        rows = []
        for s in sites:
            row = np.zeros(n, dtype=np.uint8)
            for step, (dr, dc) in enumerate(order):
                q = self._d_idx.get((s.row + dr, s.col + dc))
                if q is not None:
                    row[q] = step + 1
            rows.append(row)
        keyed = sorted((int(np.flatnonzero(r)[0]) if r.any() else n, i) for i, r in enumerate(rows))
        for a, b in zip(keyed, keyed[1:]):
            if a[0] == b[0]:
                raise RuntimeError(
                    "surface-code CNOT schedule: two same-type stabilizers share their first supported data qubit, so the schedule rows cannot be aligned with the parity-matrix rows."
                )
        return np.array([rows[i] for _, i in keyed], dtype=np.uint8).reshape(len(rows), n)

    def get_cnot_schedule_x(self) -> np.ndarray:
        """The X stabilizers' CNOT schedule, rows as the sorted parity matrix's: entry 0 is no
        support, k >= 1 the CNOT's timestep, ordered so hook errors land perpendicular to the X
        logical (column-major when it runs along the top row, row-major otherwise)."""
        return self._schedule(self.x_stab_coords, self._x_logical_on_top_row())

    def get_cnot_schedule_z(self) -> np.ndarray:
        """The Z stabilizers' CNOT schedule, in the order transposed to the X stabilizers'."""
        return self._schedule(self.z_stab_coords, not self._x_logical_on_top_row())

    @staticmethod
    def _pairs(sched: np.ndarray) -> List[int]:
        out: List[int] = []
        for s, row in enumerate(sched):
            for _, d in sorted((int(row[d]), int(d)) for d in np.flatnonzero(row)):
                out += [s, d]
        return out

    def get_cnot_schedule_pairs_x(self) -> List[int]:
        """The X schedule as a flat list of (stabilizer, data) pairs, by timestep within each
        stabilizer."""
        return self._pairs(self.get_cnot_schedule_x())

    def get_cnot_schedule_pairs_z(self) -> List[int]:
        return self._pairs(self.get_cnot_schedule_z())


# Stabilizer words and their parity matrices (stabilizer_utils.cpp).


def _word(op: Any) -> str:
    if isinstance(op, str):
        return op.strip().lstrip("+-").replace("_", "I")
    if hasattr(op, "get_pauli_word"):
        return str(op.get_pauli_word())
    if isinstance(op, PauliString):
        return str(op)[1:].replace("_", "I")
    raise TypeError(f"a Pauli word, a PauliString or a spin operator term, not {type(op).__name__}")


def _sort_key(w: str) -> Tuple[int, int]:
    # Words with a Z first, by their first Z; then the rest by their first X.
    z, x = w.find("Z"), w.find("X")
    big = 1 << 62
    return (0, z) if z >= 0 else (1, x if x >= 0 else big)


def _sorted(words: List[str]) -> List[str]:
    return sorted(words, key=_sort_key)


def _parity(words: List[str], kind: str, empty_cols: int) -> np.ndarray:
    """``to_parity_matrix``: XZ is [H_Z | 0; 0 | H_X] (the Z bits first), X and Z one block."""
    if not words:
        return np.zeros((0, empty_cols), dtype=np.uint8)
    ws = _sorted(words)
    n = len(ws[0])
    nz = 0
    for w in ws:
        if "Z" not in w:
            break
        nz += 1
    zrows = np.array([[c == "Z" for c in w] for w in ws[:nz]], dtype=np.uint8).reshape(nz, n)
    xrows = np.array([[c == "X" for c in w] for w in ws[nz:]], dtype=np.uint8).reshape(len(ws) - nz, n)
    if kind == "Z":
        return zrows if nz else np.zeros((0, empty_cols), dtype=np.uint8)
    if kind == "X":
        return xrows if len(ws) > nz else np.zeros((0, empty_cols), dtype=np.uint8)
    out = np.zeros((len(ws), 2 * n), dtype=np.uint8)
    out[:nz, :n] = zrows
    out[nz:, n:] = xrows
    return out


class Code:
    """A quantum error-correcting code: its stabilizers and logical observables (as Pauli words
    over its data qubits), its parity-check matrices, its qubit counts and the logical
    operations it encodes. Codes come from ``get_code``; a class decorated with ``@code(name)``
    sets ``self.stabilizers``, ``self.pauli_observables`` and ``self.operation_encodings`` and
    defines the ``get_num_*`` counts."""

    _operations: Tuple[operation, ...] = ()

    def _stab_words(self) -> List[str]:
        return [_word(s) for s in getattr(self, "stabilizers", [])]

    def _obs_words(self) -> List[str]:
        return [_word(s) for s in getattr(self, "pauli_observables", [])]

    def get_parity(self) -> np.ndarray:
        return _parity(self._stab_words(), "XZ", 2 * self.get_num_data_qubits())

    def get_parity_x(self) -> np.ndarray:
        return _parity(self._stab_words(), "X", self.get_num_data_qubits())

    def get_parity_z(self) -> np.ndarray:
        return _parity(self._stab_words(), "Z", self.get_num_data_qubits())

    def get_stabilizer_schedule_x(self) -> np.ndarray:
        """The X stabilizers' schedule for the stabilizer round: 0 no support, k >= 1 an
        interaction at timestep k (by default the parity matrix, everything at step 1)."""
        return self._py_schedule("get_stabilizer_schedule_x", self.get_parity_x())

    def get_stabilizer_schedule_z(self) -> np.ndarray:
        return self._py_schedule("get_stabilizer_schedule_z", self.get_parity_z())

    def _py_schedule(self, name: str, parity: np.ndarray) -> np.ndarray:
        own = next((cls.__dict__[name] for cls in type(self).__mro__ if cls is not Code and name in cls.__dict__), None)
        if own is None or own is getattr(Code, name):
            return parity
        try:
            rows = [[int(v) for v in r] for r in own(self)]
            if any(v < 0 or v > 255 for r in rows for v in r):
                raise ValueError
        except Exception:
            raise RuntimeError(f"{name} must return a 2D array/list of integers in [0, 255].") from None
        if not rows:
            if parity.size:
                raise RuntimeError(f"{name} shape does not match the parity matrix.")
            return parity
        if parity.shape != (len(rows), len(rows[0])):
            raise RuntimeError(f"{name} shape does not match the parity matrix.")
        if any(len(r) != len(rows[0]) for r in rows):
            raise RuntimeError(f"{name} rows must all have the same length.")
        out = np.array(rows, dtype=np.uint8)
        if not np.array_equal(out != 0, parity != 0):
            raise RuntimeError(f"{name} support pattern does not match the parity matrix.")
        return out

    def get_pauli_observables_matrix(self) -> np.ndarray:
        return _parity(self._obs_words(), "XZ", 2 * self.get_num_data_qubits())

    def get_observables_x(self) -> np.ndarray:
        return _parity(self._obs_words(), "X", self.get_num_data_qubits())

    def get_observables_z(self) -> np.ndarray:
        return _parity(self._obs_words(), "Z", self.get_num_data_qubits())

    def get_stabilizers(self) -> List[PauliString]:
        return [PauliString(w) for w in self._stab_words()]

    def contains_operation(self, op: operation) -> bool:
        return operation(op) in self._encodings()

    def _encodings(self) -> Any:
        enc = getattr(self, "operation_encodings", None)
        return {operation(k) for k in enc} if isinstance(enc, dict) else set(self._operations)

    def _python_kernel(self, op: operation) -> Any:
        if not self.contains_operation(op):
            raise RuntimeError("No encoding registered for requested op.")
        enc = getattr(self, "operation_encodings", None)
        if not isinstance(enc, dict):
            raise RuntimeError("This code was not registered from Python; no Python kernel is available.")
        return {operation(k): v for k, v in enc.items()}[operation(op)]

    def get_operation_one_qubit(self, op: operation) -> Any:
        """The Python kernel a ``@code`` class registered for a one-qubit operation."""
        return self._python_kernel(op)

    def get_operation_two_qubit(self, op: operation) -> Any:
        return self._python_kernel(op)

    def get_stabilizer_round(self) -> Any:
        if not self.contains_operation(operation.stabilizer_round):
            raise RuntimeError("No stabilizer_round encoding is registered.")
        return self._python_kernel(operation.stabilizer_round)

    def get_num_data_qubits(self) -> int:
        raise NotImplementedError

    def get_num_ancilla_qubits(self) -> int:
        raise NotImplementedError

    def get_num_ancilla_x_qubits(self) -> int:
        raise NotImplementedError

    def get_num_ancilla_z_qubits(self) -> int:
        raise NotImplementedError

    def get_num_x_stabilizers(self) -> int:
        raise NotImplementedError

    def get_num_z_stabilizers(self) -> int:
        raise NotImplementedError

    # The device kernels, as this package's circuits (see _Kernel).
    def _prep(self, k: "_Kernel", op: operation) -> None:
        k.reset(k.data)
        if op in (operation.prep1, operation.prepm):
            k.gate("X", k.data)
        if op in (operation.prepp, operation.prepm):
            k.gate("H", k.data)

    def _round(self, k: "_Kernel", sx: np.ndarray, sz: np.ndarray) -> List[int]:
        # The scheduled round (surface_code_device.cpp), for any code with schedules.
        k.reset(k.ancx + k.ancz)
        k.gate("H", k.ancx)
        steps = int(max(sx.max(initial=0), sz.max(initial=0)))
        for step in range(1, steps + 1):
            for xi, a in enumerate(k.ancx):
                for di in np.flatnonzero(sx[xi] == step):
                    k.gate2("CX", a, k.data[di])
            for zi, a in enumerate(k.ancz):
                for di in np.flatnonzero(sz[zi] == step):
                    k.gate2("CX", k.data[di], a)
        k.gate("H", k.ancx)
        return k.measure(k.ancz + k.ancx)


_ALL_SURFACE = (operation.x, operation.z, operation.cx, operation.cz, operation.stabilizer_round, operation.prep0, operation.prep1, operation.prepp, operation.prepm)


class surface_code(Code):
    """The rotated surface code of ``distance`` (its stabilizer_grid in ``orientation``, ZH by
    default), measured in the depth-4 schedule whose hook errors run across the logicals."""

    _operations = _ALL_SURFACE

    def __init__(self, distance: Optional[int] = None, orientation: Any = "ZH", **_: Any) -> None:
        if distance is None:
            raise RuntimeError("[surface_code] distance not provided. distance must be provided via qec::get_code(..., options) options map.")
        self.distance = int(distance)
        self.grid = stabilizer_grid(self.distance, _orientation(orientation))
        self.stabilizers = [PauliString(w) for w in _sorted([_word(s) for s in self.grid.get_spin_op_stabilizers()])]
        self.pauli_observables = [PauliString(w) for w in _sorted(self.grid._observable_words())]

    def get_num_data_qubits(self) -> int:
        return self.distance * self.distance

    def get_num_ancilla_qubits(self) -> int:
        return self.distance * self.distance - 1

    def get_num_ancilla_x_qubits(self) -> int:
        return (self.distance * self.distance - 1) // 2

    get_num_ancilla_z_qubits = get_num_x_stabilizers = get_num_z_stabilizers = get_num_ancilla_x_qubits

    def get_stabilizer_schedule_x(self) -> np.ndarray:
        return self.grid.get_cnot_schedule_x()

    def get_stabilizer_schedule_z(self) -> np.ndarray:
        return self.grid.get_cnot_schedule_z()


class steane(Code):
    """Steane's [[7, 1, 3]] code, prepared by its encoding circuit."""

    _operations = (operation.x, operation.y, operation.z, operation.h, operation.s, operation.cx, operation.cy, operation.cz, operation.stabilizer_round, operation.prep0, operation.prep1, operation.prepp, operation.prepm)

    def __init__(self, **_: Any) -> None:
        self.stabilizers = [PauliString(w) for w in _sorted(["XXXXIII", "IXXIXXI", "IIXXIXX", "ZZZZIII", "IZZIZZI", "IIZZIZZ"])]
        self.pauli_observables = [PauliString(w) for w in _sorted(["IIIIXXX", "IIIIZZZ"])]

    def get_num_data_qubits(self) -> int:
        return 7

    def get_num_ancilla_qubits(self) -> int:
        return 6

    def get_num_ancilla_x_qubits(self) -> int:
        return 3

    get_num_ancilla_z_qubits = get_num_x_stabilizers = get_num_z_stabilizers = get_num_ancilla_x_qubits

    def _prep(self, k: "_Kernel", op: operation) -> None:
        d = k.data
        k.gate("H", [d[0], d[4], d[6]])
        for a, b in ((0, 1), (4, 5), (6, 3), (6, 5), (4, 2), (0, 3), (4, 1), (3, 2)):
            k.gate2("CX", d[a], d[b])
        if op in (operation.prep1, operation.prepm):
            k.gate("X", d)
        if op in (operation.prepp, operation.prepm):
            k.gate("H", d)

    def _round(self, k: "_Kernel", sx: np.ndarray, sz: np.ndarray) -> List[int]:
        k.gate("H", k.ancx)
        for xi, a in enumerate(k.ancx):
            for di in np.flatnonzero(sx[xi] == 1):
                k.gate2("CX", a, k.data[di])
        k.gate("H", k.ancx)
        for zi, a in enumerate(k.ancz):
            for di in np.flatnonzero(sz[zi] == 1):
                k.gate2("CX", k.data[di], a)
        out = k.measure(k.ancz + k.ancx)
        k.reset(k.ancx + k.ancz)
        return out


class repetition(Code):
    """The bit-flip repetition code of ``distance`` (Z basis only)."""

    _operations = (operation.stabilizer_round, operation.x, operation.prep0, operation.prep1)

    def __init__(self, distance: Optional[int] = None, **_: Any) -> None:
        if distance is None:
            raise RuntimeError("[repetition] distance not provided. distance must be provided via qec::get_code(..., options) options map.")
        self.distance = int(distance)
        n = self.distance
        self.stabilizers = [PauliString(w) for w in _sorted(["I" * (i - 1) + "ZZ" + "I" * (n - i - 1) for i in range(1, n)])]
        self.pauli_observables = [PauliString("Z" + "I" * (n - 1))]

    def get_num_data_qubits(self) -> int:
        return self.distance

    def get_num_ancilla_qubits(self) -> int:
        return self.distance - 1

    def get_num_ancilla_x_qubits(self) -> int:
        return 0

    def get_num_ancilla_z_qubits(self) -> int:
        return self.distance - 1

    def get_num_x_stabilizers(self) -> int:
        return 0

    def get_num_z_stabilizers(self) -> int:
        return self.distance - 1

    def _round(self, k: "_Kernel", sx: np.ndarray, sz: np.ndarray) -> List[int]:
        for i, a in enumerate(k.ancz):
            k.gate2("CX", k.data[i], a)
        for i in range(1, len(k.data)):
            k.gate2("CX", k.data[i], k.ancz[i - 1])
        out = k.measure(k.ancz)
        k.reset(k.ancz)
        return out


_BUILTIN: Dict[str, Callable[..., Code]] = {"repetition": repetition, "steane": steane, "surface_code": surface_code}
_PYTHON: Dict[str, Callable[..., Code]] = {}
_REQUIRED = ("get_num_data_qubits", "get_num_ancilla_qubits", "get_num_ancilla_x_qubits", "get_num_ancilla_z_qubits", "get_num_x_stabilizers", "get_num_z_stabilizers")


def code(name: str) -> Callable[[type], type]:
    """Register a Python code class under ``name`` (a decorator): the class, made a subclass of
    ``Code``, sets ``self.stabilizers`` and ``self.pauli_observables`` (Pauli words,
    PauliStrings or spin operator terms) and ``self.operation_encodings`` (a dict from
    ``operation`` to its kernels) and defines the ``get_num_*`` counts. Its memory circuits use
    the scheduled stabilizer round and the reset-based preparations."""
    if name in _BUILTIN:
        raise RuntimeError(f"Invalid Python QEC Code name. {name} is already used in the C++ Code registry.")

    def register(cls: type) -> type:
        for m in _REQUIRED:
            if not hasattr(cls, m):
                raise RuntimeError(f"Code class must implement {m} method")
        ns = {k: v for k, v in cls.__dict__.items() if k not in ("__dict__", "__weakref__")}
        new = type(cls.__name__, (Code,), ns)
        _PYTHON[name] = new
        return new

    return register


def _check_python_code(c: Any) -> Any:
    if not hasattr(c, "stabilizers"):
        raise RuntimeError("Invalid Python QEC Code. Must have self.stabilizers = [cudaq.SpinOperator(...)] (qec.Stabilizers(...)). Please provide the stabilizers.")
    if not hasattr(c, "pauli_observables"):
        raise RuntimeError("Invalid Python QEC Code. Must have self.pauli_observables = [cudaq.SpinOperator(...)]. Please provide the observables.")
    if not hasattr(c, "operation_encodings"):
        raise RuntimeError("Invalid Python QEC Code. Must have self.operation_encodings = {...}. Please provide the CUDA-Q kernels for the operation encodings.")
    return c


def get_code(name: str, **options: Any) -> Code:
    """The code registered as ``name`` (``surface_code``, ``steane``, ``repetition`` or one
    registered with ``@code``), made with ``options`` (``distance``, the surface code's
    ``orientation``). ``stabilizers=`` (Pauli words, sorted, or terms, as given) replaces a
    built-in code's."""
    if name in _PYTHON:
        return _check_python_code(_PYTHON[name](**options))
    stabs = options.pop("stabilizers", None)
    if stabs is not None:
        if not isinstance(stabs, list):
            raise RuntimeError("invalid stabilizers passed to get_code, must be a list of string pauli words or list of cudaq.SpinOperator.")
        if stabs and isinstance(stabs[0], str):
            stabs = [PauliString(w) for w in _sorted([_word(s) for s in stabs])]
        elif stabs and not (isinstance(stabs[0], PauliString) or hasattr(stabs[0], "get_pauli_word")):
            raise RuntimeError("get_code error - invalid stabilizers element type.")
    if name not in _BUILTIN:
        raise RuntimeError(f"invalid qec_code requested: {name}")
    c = _BUILTIN[name](**options)
    if stabs is not None:
        c.stabilizers = list(stabs)
    return c


def get_available_codes() -> List[str]:
    """The registered codes' names, the built-in ones first."""
    return sorted(_BUILTIN) + list(_PYTHON)


# CUDA-Q's noise models, as Stim channels.


class _Channel:
    num_qubits = 1

    def __init__(self, probability: float) -> None:
        self.probability = float(probability)
        self.parameters = [self.probability]

    def _stim(self, targets: Sequence[int]) -> str:
        raise NotImplementedError


def _channel(name: str, stim: str, n: int, doc: str) -> type:
    def emit(self: _Channel, targets: Sequence[int]) -> str:
        return f"{stim}({self.probability!r}) " + " ".join(map(str, targets))

    return type(name, (_Channel,), {"num_qubits": n, "_stim": emit, "__doc__": doc})


DepolarizationChannel = _channel("DepolarizationChannel", "DEPOLARIZE1", 1, "Single-qubit depolarizing: X, Y or Z each at probability / 3.")
Depolarization1 = _channel("Depolarization1", "DEPOLARIZE1", 1, "Single-qubit depolarizing: X, Y or Z each at probability / 3.")
Depolarization2 = _channel("Depolarization2", "DEPOLARIZE2", 2, "Two-qubit depolarizing: each of the 15 non-identity Paulis at probability / 15.")
BitFlipChannel = _channel("BitFlipChannel", "X_ERROR", 1, "X at probability.")
PhaseFlipChannel = _channel("PhaseFlipChannel", "Z_ERROR", 1, "Z at probability.")
XError = _channel("XError", "X_ERROR", 1, "X at probability.")
YError = _channel("YError", "Y_ERROR", 1, "Y at probability.")
ZError = _channel("ZError", "Z_ERROR", 1, "Z at probability.")
TwoQubitDepolarization = _channel("TwoQubitDepolarization", "DEPOLARIZE1", 2, "Single-qubit depolarizing on each qubit of a two-qubit operation, independently (CUDA-Q QEC's).")
TwoQubitBitFlip = _channel("TwoQubitBitFlip", "X_ERROR", 2, "A bit flip on each qubit of a two-qubit operation, independently (CUDA-Q QEC's).")


class Pauli1(_Channel):
    """X, Y and Z at the given probabilities."""

    def __init__(self, probabilities: Sequence[float]) -> None:
        self.parameters = [float(p) for p in probabilities]
        if len(self.parameters) != 3:
            raise ValueError("Pauli1 takes three probabilities (X, Y, Z)")

    def _stim(self, targets: Sequence[int]) -> str:
        return "PAULI_CHANNEL_1(" + ", ".join(map(repr, self.parameters)) + ") " + " ".join(map(str, targets))


class Pauli2(_Channel):
    """The 15 non-identity two-qubit Paulis (IX, IY, IZ, XI, ..., ZZ) at the given
    probabilities."""

    num_qubits = 2

    def __init__(self, probabilities: Sequence[float]) -> None:
        self.parameters = [float(p) for p in probabilities]
        if len(self.parameters) != 15:
            raise ValueError("Pauli2 takes 15 probabilities")

    def _stim(self, targets: Sequence[int]) -> str:
        return "PAULI_CHANNEL_2(" + ", ".join(map(repr, self.parameters)) + ") " + " ".join(map(str, targets))


_GATE_NAMES = {"x": "X", "y": "Y", "z": "Z", "h": "H", "s": "S", "mz": "M", "reset": "R"}


class NoiseModel:
    """CUDA-Q's noise model: channels applied after a gate (before a measurement, ``"mz"``),
    on all its qubits (``add_all_qubit_channel(op, channel, num_controls)``; a CNOT is ``"x"``
    with one control) or on given ones (``add_channel(op, qubits, channel)``)."""

    def __init__(self) -> None:
        self._all: Dict[Tuple[str, int], List[_Channel]] = {}
        self._on: Dict[Tuple[str, Tuple[int, ...]], List[_Channel]] = {}

    def add_all_qubit_channel(self, op: str, channel: _Channel, num_controls: int = 0) -> None:
        self._all.setdefault((op.lower(), int(num_controls)), []).append(channel)

    def add_channel(self, op: str, qubits: Sequence[int], channel: _Channel) -> None:
        self._on.setdefault((op.lower(), tuple(int(q) for q in qubits)), []).append(channel)

    def empty(self) -> bool:
        return not self._all and not self._on

    def _channels(self, op: str, controls: int, qubits: Tuple[int, ...]) -> List[_Channel]:
        return self._all.get((op, controls), []) + self._on.get((op, qubits), [])


noise_model = NoiseModel


class _Kernel:
    """A memory circuit under construction: data qubits first, then the X ancillas, then the
    Z (as CUDA-Q's ``qvector``s), each operation followed by its noise."""

    def __init__(self, n_data: int, n_ancx: int, n_ancz: int, noise: Optional[NoiseModel]) -> None:
        self.data = list(range(n_data))
        self.ancx = list(range(n_data, n_data + n_ancx))
        self.ancz = list(range(n_data + n_ancx, n_data + n_ancx + n_ancz))
        self.noise = noise or NoiseModel()
        self.lines: List[str] = []
        self.num_measurements = 0
        self.detectors: List[List[int]] = []
        self.observables: List[List[int]] = []

    def _after(self, op: str, controls: int, qubits: Tuple[int, ...]) -> None:
        for ch in self.noise._channels(op, controls, qubits):
            self.lines.append(ch._stim(qubits))

    def gate(self, name: str, qubits: Sequence[int]) -> None:
        for q in qubits:
            self.lines.append(f"{name} {q}")
            self._after(name.lower(), 0, (q,))

    def gate2(self, name: str, a: int, b: int) -> None:
        self.lines.append(f"{name} {a} {b}")
        self._after(name[1:].lower(), 1, (a, b))

    def reset(self, qubits: Sequence[int]) -> None:
        for q in qubits:
            self.lines.append(f"R {q}")
            self._after("reset", 0, (q,))

    def measure(self, qubits: Sequence[int]) -> List[int]:
        out = []
        for q in qubits:
            self._after("mz", 0, (q,))
            self.lines.append(f"M {q}")
            out.append(self.num_measurements)
            self.num_measurements += 1
        return out

    def detector(self, recs: Sequence[int]) -> None:
        self.detectors.append(list(recs))
        self.lines.append("DETECTOR " + " ".join(f"rec[{r - self.num_measurements}]" for r in recs))

    def observable(self, recs: Sequence[int]) -> None:
        self.lines.append(f"OBSERVABLE_INCLUDE({len(self.observables)}) " + " ".join(f"rec[{r - self.num_measurements}]" for r in recs))
        self.observables.append(list(recs))

    def circuit(self) -> Circuit:
        return Circuit("\n".join(self.lines))


_PREPS = (operation.prep0, operation.prep1, operation.prepp, operation.prepm)


def _memory(c: Code, op: operation, rounds: int, noise: Optional[NoiseModel], who: str) -> Tuple[_Kernel, bool]:
    op = operation(op)
    if rounds == 0:
        raise RuntimeError(f"{who} error - numRounds must be >= 1. The memory circuit always performs at least one stabilizer measurement round. ")
    if not c.contains_operation(op):
        raise RuntimeError(f"{who}_error - requested state prep kernel not found.")
    if op not in _PREPS:
        raise RuntimeError(f"{who}_error - invalid requested state prep kernel.")
    if not c.contains_operation(operation.stabilizer_round):
        raise RuntimeError(f"{who} error - no stabilizer round kernel for this code.")
    in_x = op in (operation.prepp, operation.prepm)
    sx, sz = c.get_stabilizer_schedule_x(), c.get_stabilizer_schedule_z()
    obs = c.get_observables_x() if in_x else c.get_observables_z()
    k = _Kernel(c.get_num_data_qubits(), c.get_num_ancilla_x_qubits(), c.get_num_ancilla_z_qubits(), noise)
    c._prep(k, op)
    last = c._round(k, sx, sz)
    fixed = len(k.ancx) if in_x else len(k.ancz)
    off = len(last) - fixed if in_x else 0
    for i in range(fixed):
        k.detector([last[off + i]])
    for _ in range(1, rounds):
        cur = c._round(k, sx, sz)
        for a, b in zip(last, cur):
            k.detector([a, b])
        last = cur
    if in_x:
        k.gate("H", k.data)
    data = k.measure(k.data)
    for row in obs:
        k.observable([data[q] for q in np.flatnonzero(row)])
    sched = sx if in_x else sz
    for i in range(fixed):
        k.detector([last[off + i]] + [data[q] for q in np.flatnonzero(sched[i])])
    return k, not in_x


def memory_circuit(code: Code, op: operation, numRounds: int, noise: Optional[NoiseModel] = None) -> Circuit:
    """The circuit CUDA-Q QEC samples for a memory experiment: ``op`` (a preparation), then
    ``numRounds`` stabilizer rounds, then the data measured in the preparation's basis; first
    the fixed basis's syndromes as detectors, each later round's changes, then the fixed
    syndromes against the data; one observable per logical of that basis. Data qubits come
    first, then the X ancillas, then the Z."""
    return _memory(code, op, int(numRounds), noise, "sample_memory_circuit")[0].circuit()


def _ranges(rounds: int, nz: int, nx: int, z_prep: bool, keep_x: bool, keep_z: bool, total: int) -> List[Tuple[int, int]]:
    if not keep_x and not keep_z:
        raise RuntimeError("stabilizer_detector_ranges error - no stabilizers to keep.")
    per = nz + nx
    off = 0 if keep_z else nz
    ret = (nz if keep_z else 0) + (nx if keep_x else 0)
    fixed = nz if z_prep else nx
    keep_fixed = (z_prep and keep_z) or (not z_prep and keep_x)
    boundary = fixed if keep_fixed else 0
    interior = max(rounds - 1, 0)
    if 2 * fixed + interior * per != total:
        raise RuntimeError("stabilizer_detector_ranges error - unexpected number of detector values.")
    out = []
    if boundary:
        out.append((0, boundary))
    out += [(fixed + r * per + off, ret) for r in range(interior)]
    if boundary:
        out.append((total - fixed, boundary))
    return out


def _cols(ranges: List[Tuple[int, int]]) -> List[int]:
    return [s + i for s, n in ranges for i in range(n)]


def _sample_memory(c: Code, op: operation, shots: int, rounds: int, noise: Optional[NoiseModel], keep_x: bool, keep_z: bool, seed: Optional[int]) -> Tuple[np.ndarray, np.ndarray]:
    if not keep_x and not keep_z:
        raise RuntimeError("sample_memory_circuit error - no stabilizers to keep.")
    k, z_prep = _memory(c, op, int(rounds), noise, "sample_memory_circuit")
    m = np.asarray(k.circuit().compile_sampler(seed=seed).sample(int(shots)), dtype=np.uint8)
    n = len(k.data)
    data = m[:, m.shape[1] - n:].copy()
    synd = np.zeros((int(shots), len(k.detectors)), dtype=np.uint8)
    for d, recs in enumerate(k.detectors):
        synd[:, d] = np.bitwise_xor.reduce(m[:, recs], axis=1)
    if not (keep_x and keep_z):
        cols = _cols(_ranges(int(rounds), c.get_num_z_stabilizers(), c.get_num_x_stabilizers(), z_prep, keep_x, keep_z, synd.shape[1]))
        synd = synd[:, cols]
    return synd, data


def sample_memory_circuit(code: Code, *args: Any, seed: Optional[int] = None, **kwargs: Any) -> Tuple[np.ndarray, np.ndarray]:
    """``sample_memory_circuit(code, [op,] numShots, numRounds, noise=None)``: the detectors
    (``numFixed`` from the first round, ``numRounds - 1`` blocks of [Z][X] changes, ``numFixed``
    at the end) and the data measurements of ``numShots`` memory experiments (``op`` prep0 by
    default)."""
    op, rest = (operation(args[0]), args[1:]) if args and isinstance(args[0], operation) else (kwargs.pop("op", operation.prep0), args)
    names = ("numShots", "numRounds", "noise")
    vals = dict(zip(names, rest), **kwargs)
    return _sample_memory(code, op, vals["numShots"], vals["numRounds"], vals.get("noise"), True, True, seed)


def x_sample_memory_circuit(code: Code, op: operation, numShots: int, numRounds: int, noise: Optional[NoiseModel] = None, *, seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """As ``sample_memory_circuit``, keeping only the X stabilizers' detectors."""
    return _sample_memory(code, op, numShots, numRounds, noise, True, False, seed)


def z_sample_memory_circuit(code: Code, op: operation, numShots: int, numRounds: int, noise: Optional[NoiseModel] = None, *, seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """As ``sample_memory_circuit``, keeping only the Z stabilizers' detectors."""
    return _sample_memory(code, op, numShots, numRounds, noise, False, True, seed)


class DecoderContext:
    """A memory experiment's detector error model before canonicalization, with its
    measurement-to-detector (``m2d``) and -observable (``m2o``) maps; each component method
    gives a canonicalized ``(dem, m2d, m2o)``."""

    def __init__(self, dem: DetectorErrorModel, m2d: List[List[int]], m2o: List[List[int]], num_measurements: int, rounds: int, nx: int, nz: int, z_prep: bool) -> None:
        self._dem, self._m2d, self._m2o = dem, m2d, m2o
        self._n, self._rounds, self._nx, self._nz, self._z = num_measurements, rounds, nx, nz, z_prep

    @property
    def num_measurements(self) -> int:
        return self._n

    def _component(self, keep_x: bool, keep_z: bool) -> Tuple[DetectorErrorModel, List[List[int]], List[List[int]]]:
        dem = DetectorErrorModel()
        dem._H, dem._O, dem.error_rates = self._dem._H.copy(), self._dem._O.copy(), list(self._dem.error_rates)
        dem.error_ids = None if self._dem.error_ids is None else list(self._dem.error_ids)
        if keep_x and keep_z:
            dem.canonicalize_for_rounds_with_boundary(self._nx + self._nz, self._nz if self._z else self._nx, True)
            return dem, [list(r) for r in self._m2d], [list(r) for r in self._m2o]
        ranges = _ranges(self._rounds, self._nz, self._nx, self._z, keep_x, keep_z, dem.num_detectors())
        cols = _cols(ranges)
        if not cols:
            out = DetectorErrorModel()
            out._H = np.zeros((0, 0), dtype=np.uint8)
            out._O = np.zeros((dem.num_observables(), 0), dtype=np.uint8)
            return out, [], [list(r) for r in self._m2o]
        dem._H = dem._H[cols]
        dem.canonicalize_for_rounds((self._nz if keep_z else 0) + (self._nx if keep_x else 0), True)
        return dem, [list(self._m2d[i]) for i in cols], [list(r) for r in self._m2o]

    def x_component(self) -> Tuple[DetectorErrorModel, List[List[int]], List[List[int]]]:
        return self._component(True, False)

    def z_component(self) -> Tuple[DetectorErrorModel, List[List[int]], List[List[int]]]:
        return self._component(False, True)

    def full_component(self) -> Tuple[DetectorErrorModel, List[List[int]], List[List[int]]]:
        return self._component(True, True)


def decoder_context_from_memory_circuit(code: Code, op: operation, numRounds: int, noise: Optional[NoiseModel] = None, decompose_errors: bool = False) -> DecoderContext:
    """The memory experiment's detector error model (this package's analysis of its circuit),
    lazily canonicalized by stabilizer type."""
    if noise is None:
        raise RuntimeError("decoder_context_from_memory_circuit requires a noise model; noise=None is not supported.")
    k, z_prep = _memory(code, op, int(numRounds), noise, "dem_from_memory_circuit")
    text = str(k.circuit().detector_error_model(decompose_errors=bool(decompose_errors)))
    dem = dem_from_stim_text(text, bool(decompose_errors))
    return DecoderContext(dem, k.detectors, k.observables, k.num_measurements, int(numRounds), code.get_num_x_stabilizers(), code.get_num_z_stabilizers(), z_prep)


def _dem(who: str, keep: str) -> Callable[..., DetectorErrorModel]:
    def f(code: Code, op: operation, numRounds: int, noise: Optional[NoiseModel] = None, decompose_errors: bool = False) -> DetectorErrorModel:
        if noise is None:
            raise RuntimeError(f"{who} requires a noise model; noise=None is not supported.")
        ctx = decoder_context_from_memory_circuit(code, op, numRounds, noise, decompose_errors)
        return getattr(ctx, f"{keep}_component")()[0]

    f.__name__ = who
    f.__doc__ = f"The memory experiment's detector error model ({keep} component), canonicalized for its rounds; ``decompose_errors`` splits hyperedges into graphlike pieces."
    return f


dem_from_memory_circuit = _dem("dem_from_memory_circuit", "full")
x_dem_from_memory_circuit = _dem("x_dem_from_memory_circuit", "x")
z_dem_from_memory_circuit = _dem("z_dem_from_memory_circuit", "z")


def d_sparse(m2d: Sequence[Sequence[int]]) -> List[int]:
    """A measurement-to-detector map flattened, each detector's measurements then -1."""
    out: List[int] = []
    for row in m2d:
        out += [int(m) for m in row] + [-1]
    return out


# Code-capacity sampling, as libstdc++'s mt19937 and bernoulli_distribution draw it.


def _bernoulli(seed: int, count: int, p: float) -> np.ndarray:
    # generate_canonical<double, 53> takes two 32-bit draws: (lo + hi 2^32) / 2^64.
    raw = _mt_raw(np.random.RandomState(seed & 0xFFFFFFFF), 2 * count)
    lo, hi = raw[0::2].astype(np.float64), raw[1::2].astype(np.float64)
    u = (lo + hi * 4294967296.0) / 18446744073709551616.0
    u = np.where(u >= 1.0, np.nextafter(1.0, 0.0), u)
    return (u < p).astype(np.uint8)


def _mt_raw(rs: np.random.RandomState, count: int) -> np.ndarray:
    # RandomState(seed) seeds MT19937 as std::mt19937(seed) does; its raw 32-bit outputs.
    return np.asarray(rs._bit_generator.random_raw(count), dtype=np.uint64) if count else np.zeros(0, dtype=np.uint64)


def _entropy() -> int:
    return int(np.random.SeedSequence().generate_state(1)[0])


def generate_random_bit_flips(numBits: int, error_probability: float) -> np.ndarray:
    """``numBits`` independent bits, each 1 at ``error_probability``."""
    return _bernoulli(_entropy(), int(numBits), float(error_probability))


def sample_code_capacity(code_or_H: Any, numShots: int, errorProb: float, seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """(syndromes, errors) of ``numShots`` code-capacity shots: each data bit flipped at
    ``errorProb`` (the code's full parity matrix, or ``H``, as the check matrix). With
    ``seed`` the draws are CUDA-Q QEC's (its std::mt19937 and bernoulli_distribution)."""
    H = code_or_H.get_parity() if isinstance(code_or_H, Code) else np.asarray(code_or_H, dtype=np.uint8)
    n = H.shape[1]
    bits = _bernoulli(_entropy() if seed is None else int(seed), int(numShots) * n, float(errorProb)).reshape(int(numShots), n)
    synd = (bits.astype(np.int64) @ H.T.astype(np.int64) % 2).astype(np.uint8)
    return synd, bits


def oriented_memory_circuit(distance: int, rounds: int, p: float, basis: str = "z", orientation: Any = "ZH") -> Circuit:
    """An SD6 surface-code memory on the patch of ``orientation`` (XV, XH, ZV, ZH or O1-O4), in
    CUDA-Q QEC's CNOT order (hook errors across the logicals), laid out as the engine's
    ``memory_circuit``: data qubits, then the X ancillas, then the Z (grid order), each layer
    followed by its noise (two-qubit depolarizing after a CNOT, single-qubit after a gate and on
    idle qubits, flips after resets and before measurements). ZH is the engine's own patch."""
    g = stabilizer_grid(int(distance), _orientation(orientation))
    d, n = g.distance, g.distance * g.distance
    xs, zs = g.x_stab_coords, g.z_stab_coords
    ax = list(range(n, n + len(xs)))
    az = list(range(n + len(xs), n + len(xs) + len(zs)))
    anc, data = ax + az, list(range(n))
    total = n + len(anc)
    sites = list(zip(ax, xs)) + list(zip(az, zs))
    P = repr(float(p))
    top = g._x_logical_on_top_row()
    xorder, zorder = (_COL_MAJOR, _ROW_MAJOR) if top else (_ROW_MAJOR, _COL_MAJOR)
    in_x = basis == "x"

    def q(s: vec2d, off: Tuple[int, int]) -> Optional[int]:
        return g._d_idx.get((s.row + off[0], s.col + off[1]))

    def j(qs: Sequence[int]) -> str:
        return " ".join(map(str, qs))

    def idle(used: Sequence[int]) -> List[int]:
        u = set(used)
        return [i for i in range(total) if i not in u]

    lines = [f"QUBIT_COORDS({2 * c + 1}, {2 * r + 1}) {r * d + c}" for r in range(d) for c in range(d)]
    lines += [f"QUBIT_COORDS({2 * s.col}, {2 * s.row}) {a}" for a, s in sites]
    m = 0
    last: Dict[int, int] = {}
    for rnd in range(1, int(rounds) + 1):
        lines += ["TICK", f"R {j(anc)}", f"X_ERROR({P}) {j(anc)}"]
        if rnd == 1:
            lines += [f"RX {j(data)}", f"Z_ERROR({P}) {j(data)}"] if in_x else [f"R {j(data)}", f"X_ERROR({P}) {j(data)}"]
        else:
            lines.append(f"DEPOLARIZE1({P}) {j(data)}")
        h = ["TICK", f"H {j(ax)}", f"DEPOLARIZE1({P}) {j(ax + idle(ax))}"]
        lines += h
        for step in range(4):
            pairs: List[int] = []
            for a, s in zip(ax, xs):
                t = q(s, xorder[step])
                if t is not None:
                    pairs += [a, t]
            for a, s in zip(az, zs):
                t = q(s, zorder[step])
                if t is not None:
                    pairs += [t, a]
            lines += ["TICK", f"CX {j(pairs)}", f"DEPOLARIZE2({P}) {j(pairs)}", f"DEPOLARIZE1({P}) {j(idle(pairs))}"]
        lines += h
        lines += ["TICK", f"X_ERROR({P}) {j(anc)}", f"M {j(anc)}", f"DEPOLARIZE1({P}) {j(data)}"]
        now = {a: m + i for i, a in enumerate(anc)}
        m += len(anc)
        for a, s in sites:
            if rnd == 1:
                if (a in az) != in_x:
                    lines.append(f"DETECTOR({2 * s.col}, {2 * s.row}, {rnd}) rec[{now[a] - m}]")
            else:
                lines.append(f"DETECTOR({2 * s.col}, {2 * s.row}, {rnd}) rec[{now[a] - m}] rec[{last[a] - m}]")
        last = now
    lines += ["TICK", f"Z_ERROR({P}) {j(data)}" if in_x else f"X_ERROR({P}) {j(data)}", f"{'MX' if in_x else 'M'} {j(data)}"]
    drec = {qq: m + qq for qq in data}
    m += n
    for a, s in zip(ax, xs) if in_x else zip(az, zs):
        sup = [t for t in (q(s, off) for off in _COL_MAJOR) if t is not None]
        lines.append(f"DETECTOR({2 * s.col}, {2 * s.row}, {int(rounds) + 1}) " + " ".join(f"rec[{drec[t] - m}]" for t in sup) + f" rec[{last[a] - m}]")
    word = g._observable_words()[0 if in_x else 1]
    lines.append("OBSERVABLE_INCLUDE(0) " + " ".join(f"rec[{drec[i] - m}]" for i, ch in enumerate(word) if ch != "I"))
    return Circuit("\n".join(lines))

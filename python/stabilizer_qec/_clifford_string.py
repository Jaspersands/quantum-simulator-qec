"""``CliffordString``: a single-qubit Clifford on each qubit, as ``stim.CliffordString``.

Each entry is an index into Stim's list of the 24 single-qubit Cliffords
(``all_cliffords_string``), held in a numpy array; products and powers are table lookups
over the whole string at once. Global phase is ignored, as in Stim.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import numpy as np

from . import _stim

ORDER = [
    "I", "X", "Y", "Z", "H_XY", "S", "S_DAG", "H_NXY",
    "H", "SQRT_Y_DAG", "H_NXZ", "SQRT_Y", "H_YZ", "H_NYZ", "SQRT_X", "SQRT_X_DAG",
    "C_XYZ", "C_XYNZ", "C_NXYZ", "C_XNYZ", "C_ZYX", "C_ZNYX", "C_NZYX", "C_ZYNX",
]  # fmt: skip
_INDEX = {name: k for k, name in enumerate(ORDER)}
_ANNOTATIONS = {"TICK", "QUBIT_COORDS", "SHIFT_COORDS", "DETECTOR", "OBSERVABLE_INCLUDE"}


class _Tables:
    """The group's tables, built once from the tableaus: ``mul[a, b]`` is ``a·b`` (``b``
    first), ``pow[a, e]`` is ``a**e`` for ``e`` mod 12, and each element's images of X, Y, Z
    (Pauli code 1=X, 2=Y, 3=Z, and sign)."""

    def __init__(self) -> None:
        tabs = [_stim.Tableau.from_named_gate(n) for n in ORDER]
        key = {(str(t.x_output(0)), str(t.z_output(0))): k for k, t in enumerate(tabs)}
        self.mul = np.zeros((24, 24), dtype=np.uint8)
        for a, ta in enumerate(tabs):
            for b, tb in enumerate(tabs):
                t = tb.then(ta)
                self.mul[a, b] = key[(str(t.x_output(0)), str(t.z_output(0)))]
        self.pow = np.zeros((24, 12), dtype=np.uint8)
        for a in range(24):
            for e in range(1, 12):
                self.pow[a, e] = self.mul[a, self.pow[a, e - 1]]
        code = {"X": 1, "Y": 2, "Z": 3}
        self.out = {}
        for which, get in (("x", lambda t: t.x_output(0)), ("y", lambda t: t.y_output(0)), ("z", lambda t: t.z_output(0))):
            texts = [str(get(t)) for t in tabs]
            self.out[which] = (np.array([code[s[1]] for s in texts], dtype=np.uint8), np.array([s[0] == "-" for s in texts], dtype=np.bool_))
        self.single_qubit_tableau = key


_T: Optional[_Tables] = None


def _tables() -> _Tables:
    global _T
    if _T is None:
        _T = _Tables()
    return _T


def _gate_index(name: Any) -> int:
    canonical = _stim.GateData(str(name)).name
    k = _INDEX.get(canonical)
    if k is None:
        raise ValueError(f"Not a single qubit Clifford gate: {canonical}")
    return k


def _value_index(v: Any) -> Optional[int]:
    """A single Clifford named by a string, a ``GateData``, or a one-qubit ``Tableau``."""
    if isinstance(v, str):
        return _gate_index(v)
    if isinstance(v, _stim.GateData) or type(v).__name__ == "GateData":
        return _gate_index(v.name)
    if isinstance(v, _stim.Tableau) or type(v).__name__ == "Tableau":
        if len(v) == 1:
            return _tables().single_qubit_tableau[(str(v.x_output(0)), str(v.z_output(0)))]
    return None


def _paulis_as_indices(p: Any) -> np.ndarray:
    p = _stim.PauliString(p) if not isinstance(p, _stim.PauliString) else p
    return np.array([p[k] for k in range(len(p))], dtype=np.uint8)  # 0=I, 1=X, 2=Y, 3=Z: the same indices


class CliffordString:
    """A tensor product of single-qubit Cliffords (``"H,S,C_XYZ"``), ignoring global phase,
    as ``stim.CliffordString``."""

    __slots__ = ("_g",)

    def __init__(self, arg: Any, /) -> None:
        self._g = _parse(arg)

    @staticmethod
    def _wrap(g: np.ndarray) -> "CliffordString":
        out = CliffordString.__new__(CliffordString)
        out._g = np.asarray(g, dtype=np.uint8)
        return out

    # Construction.
    @staticmethod
    def random(num_qubits: int) -> "CliffordString":
        """A uniformly random string of ``num_qubits`` Cliffords."""
        return CliffordString._wrap(np.random.default_rng().integers(0, 24, int(num_qubits), dtype=np.uint8))

    @staticmethod
    def all_cliffords_string() -> "CliffordString":
        """Each of the 24 single-qubit Cliffords once, in Stim's order."""
        return CliffordString._wrap(np.arange(24, dtype=np.uint8))

    def copy(self) -> "CliffordString":
        return CliffordString._wrap(self._g.copy())

    # Sequence.
    def __len__(self) -> int:
        return len(self._g)

    def __getitem__(self, index_or_slice: Any) -> Any:
        if isinstance(index_or_slice, slice):
            return CliffordString._wrap(self._g[index_or_slice].copy())
        k = _index(index_or_slice, len(self._g))
        return _stim.gate_data(ORDER[self._g[k]])

    def __setitem__(self, index_or_slice: Any, new_value: Any) -> None:
        is_slice = isinstance(index_or_slice, slice)
        where: Any = index_or_slice if is_slice else _index(index_or_slice, len(self._g))
        span = len(range(*index_or_slice.indices(len(self._g)))) if is_slice else 1
        k = _value_index(new_value)
        if k is not None:
            self._g[where] = k
            return
        if is_slice and (isinstance(new_value, CliffordString) or type(new_value).__name__ == "CliffordString"):
            v = CliffordString(new_value)
            if len(v) != span:
                raise ValueError(f"Length mismatch. The targeted slice covers {span} values but the given CliffordString has {len(v)} values.")
            self._g[where] = v._g
            return
        if is_slice and (isinstance(new_value, _stim.PauliString) or type(new_value).__name__ == "PauliString"):
            v = _paulis_as_indices(new_value)
            if len(v) != span:
                raise ValueError(f"Length mismatch. The targeted slice covers {span} values but the given PauliString has {len(v)} values.")
            self._g[where] = v
            return
        raise ValueError(f"Don't know how to write an object of type {type(new_value)!r} to index {index_or_slice!r}")

    # Arithmetic.
    def __add__(self, rhs: "CliffordString") -> "CliffordString":
        return CliffordString._wrap(np.concatenate([self._g, CliffordString(rhs)._g]))

    def __iadd__(self, rhs: "CliffordString") -> "CliffordString":
        self._g = np.concatenate([self._g, CliffordString(rhs)._g])
        return self

    def __mul__(self, rhs: Any) -> "CliffordString":
        return CliffordString._wrap(_times(self._g, rhs))

    def __imul__(self, rhs: Any) -> "CliffordString":
        self._g = _times(self._g, rhs)
        return self

    def __rmul__(self, lhs: int) -> "CliffordString":
        if isinstance(lhs, bool) or not isinstance(lhs, (int, np.integer)) or lhs < 0:
            raise TypeError(f"Can't multiply a CliffordString by {lhs!r}")
        return CliffordString._wrap(np.tile(self._g, int(lhs)))

    def __pow__(self, power: int) -> "CliffordString":
        return CliffordString._wrap(_tables().pow[self._g, int(power) % 12])

    def __ipow__(self, power: int) -> "CliffordString":
        self._g = _tables().pow[self._g, int(power) % 12]
        return self

    # Outputs.
    def _outputs(self, which: str, bit_packed_signs: bool) -> Tuple[Any, np.ndarray]:
        codes, signs = _tables().out[which]
        c = codes[self._g]
        s = signs[self._g]
        paulis = _stim.PauliString.from_numpy(xs=(c == 1) | (c == 2), zs=(c == 2) | (c == 3))
        if bit_packed_signs:
            return paulis, np.packbits(s, bitorder="little")
        return paulis, s

    def x_outputs(self, *, bit_packed_signs: bool = False) -> Tuple[Any, np.ndarray]:
        """What each Clifford conjugates X into: the Paulis (as a positive ``PauliString``)
        and the signs."""
        return self._outputs("x", bit_packed_signs)

    def y_outputs(self, *, bit_packed_signs: bool = False) -> Tuple[Any, np.ndarray]:
        """What each Clifford conjugates Y into: the Paulis and the signs."""
        return self._outputs("y", bit_packed_signs)

    def z_outputs(self, *, bit_packed_signs: bool = False) -> Tuple[Any, np.ndarray]:
        """What each Clifford conjugates Z into: the Paulis and the signs."""
        return self._outputs("z", bit_packed_signs)

    # Comparison and text.
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CliffordString):
            return NotImplemented
        return len(self._g) == len(other._g) and bool(np.array_equal(self._g, other._g))

    def __ne__(self, other: object) -> bool:
        r = self.__eq__(other)
        return r if r is NotImplemented else not r

    __hash__ = None  # type: ignore[assignment]

    def __str__(self) -> str:
        return ",".join(ORDER[k] for k in self._g)

    def __repr__(self) -> str:
        return f'stabilizer_qec.CliffordString("{self}")'


def _index(k: Any, n: int) -> int:
    i = int(k)
    if i < 0:
        i += n
    if not 0 <= i < n:
        raise IndexError(f"Index {int(k)} not in range for sequence of length {n}.")
    return i


def _times(g: np.ndarray, rhs: Any) -> np.ndarray:
    if isinstance(rhs, (int, np.integer)) and not isinstance(rhs, bool):
        if rhs < 0:
            raise TypeError(f"Can't repeat a CliffordString {rhs} times")
        return np.tile(g, int(rhs))
    if isinstance(rhs, CliffordString) or type(rhs).__name__ == "CliffordString":
        r = CliffordString(rhs)._g
        n = max(len(g), len(r))
        a = np.zeros(n, dtype=np.uint8)
        b = np.zeros(n, dtype=np.uint8)
        a[: len(g)] = g
        b[: len(r)] = r
        return _tables().mul[a, b]
    raise ValueError(f"Don't know how to multiply by {rhs!r}")


def _parse(arg: Any) -> np.ndarray:
    if isinstance(arg, CliffordString):
        return arg._g.copy()
    if isinstance(arg, (int, np.integer)) and not isinstance(arg, bool):
        return np.zeros(int(arg), dtype=np.uint8)
    if isinstance(arg, str):
        text = arg.strip()
        if not text:
            return np.zeros(0, dtype=np.uint8)
        if text.endswith(","):
            text = text[:-1]
        return np.array([_gate_index(seg.strip()) for seg in text.split(",")], dtype=np.uint8)
    name = type(arg).__name__
    if name == "CliffordString":  # a stim.CliffordString
        return _parse(str(arg))
    if isinstance(arg, _stim.PauliString) or name == "PauliString":
        return _paulis_as_indices(arg)
    if name == "Circuit":
        return _from_circuit(arg)
    if hasattr(arg, "__iter__"):
        out: List[int] = []
        for t in arg:
            if isinstance(t, _stim.GateData) or type(t).__name__ == "GateData":
                out.append(_gate_index(t.name))
            elif isinstance(t, str):
                out.append(_gate_index(t))
            else:
                raise ValueError(f"Don't know how to convert the following item into a Clifford: {t!r}")
        return np.array(out, dtype=np.uint8)
    raise ValueError(f"Don't know how to initialize a stim.CliffordString from {arg!r}")


_NOT_QUBIT = (1 << 30) | (1 << 29) | (1 << 28) | (1 << 27) | (1 << 26)


def _from_circuit(circuit: Any) -> np.ndarray:
    from ._circuit import Circuit

    c = circuit if isinstance(circuit, Circuit) else Circuit(str(circuit))
    g = np.zeros(c.num_qubits, dtype=np.uint8)
    mul = _tables().mul

    def walk(items: list) -> None:
        for it in items:
            if it[0] == "repeat":
                for _ in range(it[1]):
                    walk(it[3])
                continue
            _, name, _tag, _args, targets = it
            if name in _ANNOTATIONS:
                continue
            k = _gate_index(name)
            for t in targets:
                if not t & _NOT_QUBIT:
                    q = t & ((1 << 24) - 1)
                    g[q] = mul[k, g[q]]

    walk(c._items())
    return g

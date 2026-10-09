"""Stim's stabilizer objects, with Stim's interface: ``PauliString``, ``Tableau``,
``TableauSimulator``, ``GateData`` / ``gate_data``, ``GateTarget`` and the ``target_*``
helpers, ``Flow``.

Names, arguments, defaults and printed forms are ``stim``'s (1.16), so code written for Stim
runs here with ``stim`` replaced by ``stabilizer_qec``. Random outcomes come from this
package's own generator, so they differ from Stim's for the same seed.
"""

from __future__ import annotations

import itertools
import math
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np

from . import _core

_UNITS = {0: 1 + 0j, 1: 1j, 2: -1 + 0j, 3: complex(-0.0, -1.0)}


def _phase_of(value: Any) -> int:
    """The power of i a unit complex number is, or ValueError."""
    try:
        c = complex(value)
    except TypeError:
        raise ValueError(f"phase factor not in [1, -1, 1, 1j]: {value!r}") from None
    for k, u in ((0, 1), (1, 1j), (2, -1), (3, -1j)):
        if c == u:
            return k
    raise ValueError("phase factor not in [1, -1, 1, 1j]")


def _pauli_value(v: Any) -> int:
    if isinstance(v, (bool, np.bool_)):
        raise ValueError(f"Expected a Pauli: {v!r}")
    if isinstance(v, (int, np.integer)):
        if 0 <= int(v) < 4:
            return int(v)
        raise ValueError(f"Expected a pauli in [0, 1, 2, 3, 'I', '_', 'X', 'Y', 'Z'] but got {v!r}.")
    if isinstance(v, str):
        s = v.upper() if len(v) == 1 else v
        table = {"I": 0, "_": 0, "X": 1, "Y": 2, "Z": 3}
        if s in table:
            return table[s]
    raise ValueError(f"Expected a pauli in [0, 1, 2, 3, 'I', '_', 'X', 'Y', 'Z'] but got {v!r}.")


def _endian(endian: str) -> bool:
    if endian == "little":
        return True
    if endian == "big":
        return False
    raise ValueError("endian not in ['little', 'big']")


def _complex_bytes(matrix: Any) -> bytes:
    return np.asarray(matrix, dtype=np.complex128).astype("<c16").tobytes()


def _complex_array(data: bytes, shape: Tuple[int, ...]) -> np.ndarray:
    return np.frombuffer(data, dtype="<c16").astype(np.complex64).reshape(shape)


def _seed(seed: Optional[int]) -> int:
    if seed is None:
        return int(np.random.SeedSequence().generate_state(1, dtype=np.uint64)[0])
    return int(seed) & ((1 << 64) - 1)


# ---------------------------------------------------------------------------------------------
# PauliString


class PauliString:
    """A Pauli product with a sign in {+1, -1, +i, -i}, as ``stim.PauliString``.

    >>> PauliString("-XYZ") * PauliString("XX_")
    stabilizer_qec.PauliString("-_ZZ")
    """

    __slots__ = ("_p",)
    __hash__ = None  # type: ignore[assignment]  (mutable, as in Stim)

    def __init__(self, arg: Any = None, /, num_qubits: Any = None, text: Any = None, other: Any = None, pauli_indices: Any = None) -> None:
        given = [x for x in (num_qubits, text, other, pauli_indices) if x is not None]
        if arg is not None and given:
            raise ValueError("Specify only one argument.")
        if arg is None and given:
            arg = given[0]
        self._p = _to_core_pauli(arg)

    @classmethod
    def _wrap(cls, core: Any) -> "PauliString":
        out = cls.__new__(cls)
        out._p = core
        return out

    # Text and comparison.
    def __str__(self) -> str:
        return str(self._p)

    def __repr__(self) -> str:
        return f'stabilizer_qec.PauliString("{self}")'

    def __eq__(self, other: object) -> bool:
        if isinstance(other, PauliString):
            return self._p == other._p
        return NotImplemented

    def __ne__(self, other: object) -> bool:
        r = self.__eq__(other)
        return r if r is NotImplemented else not r

    def __len__(self) -> int:
        return self._p.num_qubits

    def __iter__(self) -> Iterator[int]:
        for k in range(len(self)):
            yield self._p.get(k)

    def copy(self) -> "PauliString":
        return PauliString._wrap(self._p.copy())

    # Sign.
    @property
    def sign(self) -> complex:
        """The sign: +1, -1, 1j or -1j."""
        return _UNITS[self._p.phase]

    @sign.setter
    def sign(self, value: Any) -> None:
        self._p.phase = _phase_of(value)

    @property
    def weight(self) -> int:
        """How many Paulis are not the identity."""
        return self._p.weight()

    # Indexing.
    def __getitem__(self, index_or_slice: Any) -> Any:
        n = len(self)
        if isinstance(index_or_slice, slice):
            return PauliString._wrap(self._p.select(list(range(n))[index_or_slice]))
        k = int(index_or_slice)
        if k < 0:
            k += n
        if not 0 <= k < n:
            raise IndexError("index out of range")
        return self._p.get(k)

    def __setitem__(self, index: int, new_pauli: Any) -> None:
        n = len(self)
        k = int(index)
        if k < 0:
            k += n
        if not 0 <= k < n:
            raise IndexError("index out of range")
        self._p.set(k, _pauli_value(new_pauli))

    # Arithmetic.
    def __mul__(self, rhs: Any) -> "PauliString":
        if isinstance(rhs, PauliString):
            return PauliString._wrap(self._p.mul(rhs._p))
        if isinstance(rhs, (int, np.integer)) and not isinstance(rhs, bool) and rhs >= 0:
            return PauliString._wrap(self._p.tensor_power(int(rhs)))
        try:
            k = _phase_of(rhs)
        except ValueError:
            return NotImplemented
        out = self.copy()
        out._p.phase = (out._p.phase + k) & 3
        return out

    def __rmul__(self, lhs: Any) -> "PauliString":
        if isinstance(lhs, PauliString):
            return PauliString._wrap(lhs._p.mul(self._p))
        return self.__mul__(lhs)

    def __imul__(self, rhs: Any) -> "PauliString":
        r = self.__mul__(rhs)
        if r is NotImplemented:
            return r
        self._p = r._p
        return self

    def __truediv__(self, rhs: Any) -> "PauliString":
        k = _phase_of(rhs) if _is_unit(rhs) else None
        if k is None:
            raise ValueError("divisor not in (1, -1, 1j, -1j)")
        out = self.copy()
        out._p.phase = (out._p.phase - k) & 3
        return out

    def __itruediv__(self, rhs: Any) -> "PauliString":
        self._p = self.__truediv__(rhs)._p
        return self

    def __neg__(self) -> "PauliString":
        return self * -1

    def __pos__(self) -> "PauliString":
        return self.copy()

    def __add__(self, rhs: Any) -> "PauliString":
        if not isinstance(rhs, PauliString):
            return NotImplemented
        return PauliString._wrap(self._p.tensor(rhs._p))

    def __iadd__(self, rhs: Any) -> "PauliString":
        if not isinstance(rhs, PauliString):
            return NotImplemented
        self._p = self._p.tensor(rhs._p)
        return self

    def extended_product(self, other: "PauliString") -> Tuple[complex, "PauliString"]:
        """[Deprecated in Stim] The product, as (the scalar it picked up, the product with its
        own sign)."""
        prod = self * other
        scalar = _UNITS[(prod._p.phase - self._p.phase - other._p.phase) & 3]
        out = prod.copy()
        out._p.phase = (self._p.phase + other._p.phase) & 3
        return scalar, out

    def commutes(self, other: "PauliString") -> bool:
        """Whether the two commute."""
        return self._p.commutes(other._p)

    def pauli_indices(self, included_paulis: str = "XYZ") -> List[int]:
        """The indices of the Paulis of the given types (any of "IXYZ_", either case)."""
        return list(self._p.pauli_indices(included_paulis))

    # Conjugation.
    def after(self, operation: Any, targets: Optional[Iterable[int]] = None) -> "PauliString":
        """The string conjugated by a Clifford operation: U P U†."""
        return self._conj(operation, targets, False)

    def before(self, operation: Any, targets: Optional[Iterable[int]] = None) -> "PauliString":
        """The string conjugated backwards by a Clifford operation: U† P U."""
        return self._conj(operation, targets, True)

    def _conj(self, operation: Any, targets: Any, backward: bool) -> "PauliString":
        if isinstance(operation, Tableau):
            if targets is None:
                raise ValueError("Need targets when the operation is a tableau.")
            return PauliString._wrap(self._p.conjugate_by_tableau(operation._t, [int(t) for t in targets], backward))
        if targets is not None:
            raise ValueError("Don't specify targets when the operation is a circuit or instruction.")
        return PauliString._wrap(self._p.conjugate_by_circuit(_stim_text_of(operation), backward))

    def to_tableau(self) -> "Tableau":
        """The Pauli product as a tableau."""
        return Tableau._wrap(self._p.to_tableau())

    # Numpy.
    def to_numpy(self, *, bit_packed: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """The X bits and Z bits (Stim's xz encoding), packed little-endian if asked."""
        p = np.frombuffer(self._p.paulis(), dtype=np.uint8)
        xs = (p == 1) | (p == 2)
        zs = (p == 2) | (p == 3)
        if bit_packed:
            return np.packbits(xs, bitorder="little"), np.packbits(zs, bitorder="little")
        return xs, zs

    @staticmethod
    def from_numpy(*, xs: np.ndarray, zs: np.ndarray, sign: Any = 1, num_qubits: Optional[int] = None) -> "PauliString":
        """From X and Z bit arrays (bool, or bit-packed uint8 with ``num_qubits``)."""
        xs = np.asarray(xs)
        zs = np.asarray(zs)

        def unpack(a: np.ndarray, name: str) -> np.ndarray:
            if a.dtype == np.uint8:
                if num_qubits is None:
                    raise ValueError(f"Need num_qubits when {name} is bit packed.")
                return np.unpackbits(a, bitorder="little", count=int(num_qubits)).astype(bool)
            if a.dtype != np.bool_:
                raise ValueError(f"{name} must be bool or bit-packed uint8")
            if num_qubits is not None and len(a) != num_qubits:
                raise ValueError(f"len({name}) != num_qubits")
            return a
        x = unpack(xs, "xs")
        z = unpack(zs, "zs")
        if len(x) != len(z):
            raise ValueError("len(xs) != len(zs)")
        paulis = (x.astype(np.uint8) * 1) + (z.astype(np.uint8) * 3) - (x & z).astype(np.uint8) * 2
        return PauliString._wrap(_core.PauliStringCore.from_paulis(list(paulis.tolist()), _phase_of(sign)))

    def to_unitary_matrix(self, *, endian: str) -> np.ndarray:
        """The matrix (complex64), little- or big-endian."""
        d = 1 << len(self)
        return _complex_array(self._p.to_unitary(_endian(endian)), (d, d))

    @staticmethod
    def from_unitary_matrix(matrix: Any, *, endian: str = "little", unsigned: bool = False) -> "PauliString":
        """The Pauli string a unitary matrix is (signed, or up to a phase with ``unsigned``)."""
        return PauliString._wrap(_core.PauliStringCore.from_unitary(_complex_bytes(matrix), _endian(endian), bool(unsigned)))

    @staticmethod
    def random(num_qubits: int, *, allow_imaginary: bool = False) -> "PauliString":
        """A uniformly random Pauli string (Hermitian unless ``allow_imaginary``)."""
        return PauliString._wrap(_core.PauliStringCore.random(int(num_qubits), bool(allow_imaginary), _seed(None)))

    @staticmethod
    def iter_all(num_qubits: int, *, min_weight: int = 0, max_weight: Optional[int] = None, allowed_paulis: str = "XYZ") -> "PauliStringIterator":
        """Every Pauli string on ``num_qubits`` with a weight in range, in Stim's order."""
        return PauliStringIterator(num_qubits, min_weight, max_weight, allowed_paulis)


def _is_unit(v: Any) -> bool:
    try:
        _phase_of(v)
        return True
    except ValueError:
        return False


def _to_core_pauli(arg: Any) -> Any:
    if arg is None:
        return _core.PauliStringCore.from_paulis([], 0)
    if isinstance(arg, PauliString):
        return arg.copy()._p
    if isinstance(arg, str):
        return _core.PauliStringCore.from_text(arg)
    if isinstance(arg, (int, np.integer)) and not isinstance(arg, bool):
        if arg < 0:
            raise ValueError("num_qubits must be non-negative")
        return _core.PauliStringCore.from_paulis([0] * int(arg), 0)
    if isinstance(arg, dict):
        items = list(arg.items())
        if not items:
            return _core.PauliStringCore.from_paulis([], 0)
        if all(isinstance(k, (int, np.integer)) and not isinstance(k, bool) for k, _ in items) and not any(_iterable_not_str(v) for _, v in items):
            n = max(int(k) for k, _ in items) + 1
            p = [0] * n
            for k, v in items:
                p[int(k)] = _pauli_value(v)
            return _core.PauliStringCore.from_paulis(p, 0)
        n = max((max(int(q) for q in v) + 1 if len(list(v)) else 0) for _, v in items)
        p = [0] * n
        for k, v in items:
            for q in v:
                p[int(q)] = _pauli_value(k)
        return _core.PauliStringCore.from_paulis(p, 0)
    try:
        values = list(arg)
    except TypeError:
        raise TypeError(f"Don't know how to make a PauliString from {arg!r}") from None
    return _core.PauliStringCore.from_paulis([_pauli_value(v) for v in values], 0)


def _iterable_not_str(v: Any) -> bool:
    if isinstance(v, str):
        return False
    try:
        iter(v)
        return True
    except TypeError:
        return False


class PauliStringIterator:
    """Iterates Pauli strings by weight, then by support (lexicographic), then by Paulis (the
    first qubit's slowest), in Stim's order."""

    def __init__(self, num_qubits: int, min_weight: int = 0, max_weight: Optional[int] = None, allowed_paulis: str = "XYZ") -> None:
        for c in allowed_paulis:
            if c not in "XYZ":
                raise ValueError(f"allowed_paulis contains a character that isn't X, Y, or Z: {c!r}")
        self._n = int(num_qubits)
        self._min = int(min_weight)
        self._max = self._n if max_weight is None else min(int(max_weight), self._n)
        self._allowed = [p for p in "XYZ" if p in allowed_paulis]
        self._gen = self._generate()

    def _generate(self) -> Iterator[PauliString]:
        for w in range(self._min, self._max + 1):
            if w > 0 and not self._allowed:
                return
            for support in itertools.combinations(range(self._n), w):
                for ps in itertools.product(self._allowed, repeat=w):
                    v = [0] * self._n
                    for q, p in zip(support, ps):
                        v[q] = "_XYZ".index(p)
                    yield PauliString._wrap(_core.PauliStringCore.from_paulis(v, 0))

    def __iter__(self) -> "PauliStringIterator":
        return self

    def __next__(self) -> PauliString:
        return next(self._gen)


# ---------------------------------------------------------------------------------------------
# Tableau


class Tableau:
    """A Clifford operation as its stabilizer tableau, as ``stim.Tableau``: the images of each
    qubit's X and Z.

    >>> Tableau.from_named_gate("CNOT")(PauliString("X_"))
    stabilizer_qec.PauliString("+XX")
    """

    __slots__ = ("_t",)
    __hash__ = None  # type: ignore[assignment]

    def __init__(self, num_qubits: int) -> None:
        self._t = _core.TableauCore.identity(int(num_qubits))

    @classmethod
    def _wrap(cls, core: Any) -> "Tableau":
        out = cls.__new__(cls)
        out._t = core
        return out

    def __len__(self) -> int:
        return self._t.num_qubits

    def __str__(self) -> str:
        return str(self._t)

    def __repr__(self) -> str:
        n = len(self)
        xs = "".join(f"        {self.x_output(k)!r},\n" for k in range(n))
        zs = "".join(f"        {self.z_output(k)!r},\n" for k in range(n))
        return f"stabilizer_qec.Tableau.from_conjugated_generators(\n    xs=[\n{xs}    ],\n    zs=[\n{zs}    ],\n)"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Tableau):
            return self._t == other._t
        return NotImplemented

    def __ne__(self, other: object) -> bool:
        r = self.__eq__(other)
        return r if r is NotImplemented else not r

    def copy(self) -> "Tableau":
        return Tableau._wrap(self._t.copy())

    def __call__(self, pauli_string: PauliString) -> PauliString:
        """The image C P C† of a Pauli string."""
        return PauliString._wrap(self._t.apply(pauli_string._p))

    def __add__(self, rhs: "Tableau") -> "Tableau":
        """The direct sum: this tableau on the first qubits, ``rhs`` on the next."""
        if not isinstance(rhs, Tableau):
            return NotImplemented
        return Tableau._wrap(self._t.tensor(rhs._t))

    def __iadd__(self, rhs: "Tableau") -> "Tableau":
        if not isinstance(rhs, Tableau):
            return NotImplemented
        self._t = self._t.tensor(rhs._t)
        return self

    def __mul__(self, rhs: "Tableau") -> "Tableau":
        """Composition: ``(a * b)(p) == a(b(p))``."""
        if not isinstance(rhs, Tableau):
            return NotImplemented
        if len(rhs) != len(self):
            raise ValueError("len(lhs) != len(rhs)")
        return Tableau._wrap(rhs._t.then(self._t))

    def __pow__(self, exponent: int) -> "Tableau":
        return Tableau._wrap(self._t.raised_to(int(exponent)))

    def __ipow__(self, exponent: int) -> "Tableau":
        self._t = self._t.raised_to(int(exponent))
        return self

    def then(self, second: "Tableau") -> "Tableau":
        """This operation and then ``second``."""
        return Tableau._wrap(self._t.then(second._t))

    def inverse(self, *, unsigned: bool = False) -> "Tableau":
        return Tableau._wrap(self._t.inverse(bool(unsigned)))

    def append(self, gate: "Tableau", targets: Sequence[int]) -> None:
        """Apply ``gate`` to ``targets`` after this operation (in place)."""
        self._t.append(gate._t, [int(t) for t in targets])

    def prepend(self, gate: "Tableau", targets: Sequence[int]) -> None:
        """Apply ``gate`` to ``targets`` before this operation (in place)."""
        self._t.prepend(gate._t, [int(t) for t in targets])

    def _check(self, k: int) -> int:
        k = int(k)
        if not 0 <= k < len(self):
            raise IndexError("target >= len(tableau)")
        return k

    def x_output(self, target: int) -> PauliString:
        return PauliString._wrap(self._t.x_output(self._check(target)))

    def y_output(self, target: int) -> PauliString:
        return PauliString._wrap(self._t.y_output(self._check(target)))

    def z_output(self, target: int) -> PauliString:
        return PauliString._wrap(self._t.z_output(self._check(target)))

    def x_sign(self, target: int) -> int:
        return -1 if self.x_output(target).sign == -1 else 1

    def y_sign(self, target: int) -> int:
        return -1 if self.y_output(target).sign == -1 else 1

    def z_sign(self, target: int) -> int:
        return -1 if self.z_output(target).sign == -1 else 1

    def _io(self, i: int, o: int) -> Tuple[int, int]:
        n = len(self)
        if not 0 <= int(i) < n:
            raise ValueError("input_index >= len(tableau)")
        if not 0 <= int(o) < n:
            raise ValueError("output_index >= len(tableau)")
        return int(i), int(o)

    def x_output_pauli(self, input_index: int, output_index: int) -> int:
        return self._t.output_pauli(1, *self._io(input_index, output_index))

    def y_output_pauli(self, input_index: int, output_index: int) -> int:
        return self._t.output_pauli(2, *self._io(input_index, output_index))

    def z_output_pauli(self, input_index: int, output_index: int) -> int:
        return self._t.output_pauli(3, *self._io(input_index, output_index))

    def inverse_x_output_pauli(self, input_index: int, output_index: int) -> int:
        return self._t.inverse_output_pauli(1, *self._io(input_index, output_index))

    def inverse_y_output_pauli(self, input_index: int, output_index: int) -> int:
        return self._t.inverse_output_pauli(2, *self._io(input_index, output_index))

    def inverse_z_output_pauli(self, input_index: int, output_index: int) -> int:
        return self._t.inverse_output_pauli(3, *self._io(input_index, output_index))

    def inverse_x_output(self, input_index: int, *, unsigned: bool = False) -> PauliString:
        return PauliString._wrap(self._t.inverse_output(1, self._io(input_index, 0)[0] if len(self) else int(input_index), bool(unsigned)))

    def inverse_y_output(self, input_index: int, *, unsigned: bool = False) -> PauliString:
        return PauliString._wrap(self._t.inverse_output(2, self._io(input_index, 0)[0] if len(self) else int(input_index), bool(unsigned)))

    def inverse_z_output(self, input_index: int, *, unsigned: bool = False) -> PauliString:
        return PauliString._wrap(self._t.inverse_output(3, self._io(input_index, 0)[0] if len(self) else int(input_index), bool(unsigned)))

    def to_pauli_string(self) -> PauliString:
        return PauliString._wrap(self._t.to_pauli_string())

    def to_stabilizers(self, *, canonicalize: bool = False) -> List[PauliString]:
        """The stabilizers of the state this tableau prepares from |0…0⟩."""
        return [PauliString._wrap(p) for p in self._t.stabilizers(bool(canonicalize))]

    def to_circuit(self, method: str = "elimination") -> "Any":
        """A circuit implementing the tableau: ``"elimination"`` (the operation), or
        ``"graph_state"``, ``"mpp_state"``, ``"mpp_state_unsigned"`` (the state it prepares)."""
        from ._circuit import Circuit

        return Circuit(self._t.to_circuit(method))

    def to_state_vector(self, *, endian: str = "little") -> np.ndarray:
        """The state C|0…0⟩, first nonzero amplitude real and positive (complex64)."""
        return _complex_array(self._t.to_state_vector(_endian(endian)), (1 << len(self),))

    def to_unitary_matrix(self, *, endian: str) -> np.ndarray:
        """The unitary, with Stim's global phase (complex64)."""
        d = 1 << len(self)
        return _complex_array(self._t.to_unitary(_endian(endian)), (d, d))

    def to_numpy(self, *, bit_packed: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """(x2x, x2z, z2x, z2z, x_signs, z_signs), as Stim lays them out."""
        n = len(self)
        b = self._t.to_bits()
        quads = [np.frombuffer(q, dtype=np.uint8).astype(bool).reshape((n, n)) for q in b[:4]]
        signs = [np.frombuffer(s, dtype=np.uint8).astype(bool) for s in b[4:]]
        if bit_packed:
            quads = [np.packbits(q, axis=1, bitorder="little") if n else np.zeros((0, 0), dtype=np.uint8) for q in quads]
            signs = [np.packbits(s, bitorder="little") for s in signs]
        return (*quads, *signs)  # type: ignore[return-value]

    @staticmethod
    def from_numpy(*, x2x: np.ndarray, x2z: np.ndarray, z2x: np.ndarray, z2z: np.ndarray, x_signs: Optional[np.ndarray] = None, z_signs: Optional[np.ndarray] = None) -> "Tableau":
        """From the arrays ``to_numpy`` gives (bool, or bit-packed uint8)."""
        x2x = np.asarray(x2x)
        n = x2x.shape[0]

        def quad(a: np.ndarray) -> bytes:
            a = np.asarray(a)
            if a.dtype == np.uint8:
                a = np.unpackbits(a, axis=1, bitorder="little", count=n).astype(bool)
            if a.shape != (n, n):
                raise ValueError("the quadrants must all be square and of the same size")
            return a.astype(np.uint8).tobytes()

        def sign(a: Optional[np.ndarray]) -> bytes:
            if a is None:
                return bytes(n)
            a = np.asarray(a)
            if a.dtype == np.uint8:
                a = np.unpackbits(a, bitorder="little", count=n).astype(bool)
            return a.astype(np.uint8).tobytes()

        return Tableau._wrap(_core.TableauCore.from_bits(n, quad(x2x), quad(x2z), quad(z2x), quad(z2z), sign(x_signs), sign(z_signs)))

    @staticmethod
    def from_named_gate(name: str) -> "Tableau":
        return Tableau._wrap(_core.TableauCore.from_named_gate(name))

    @staticmethod
    def from_conjugated_generators(*, xs: List[PauliString], zs: List[PauliString]) -> "Tableau":
        return Tableau._wrap(_core.TableauCore.from_conjugated_generators([p._p for p in xs], [p._p for p in zs]))

    @staticmethod
    def from_stabilizers(stabilizers: Iterable[PauliString], *, allow_redundant: bool = False, allow_underconstrained: bool = False) -> "Tableau":
        """A tableau whose Z images are the stabilizers (and X images Stim's destabilizers)."""
        return Tableau._wrap(_core.TableauCore.from_stabilizers([PauliString(p)._p for p in stabilizers], bool(allow_redundant), bool(allow_underconstrained), False))

    @staticmethod
    def from_state_vector(state_vector: Iterable[complex], *, endian: str) -> "Tableau":
        return Tableau._wrap(_core.TableauCore.from_state_vector(_complex_bytes(list(state_vector)), _endian(endian)))

    @staticmethod
    def from_unitary_matrix(matrix: Any, *, endian: str = "little") -> "Tableau":
        return Tableau._wrap(_core.TableauCore.from_unitary(_complex_bytes(matrix), _endian(endian)))

    @staticmethod
    def from_circuit(circuit: Any, *, ignore_noise: bool = False, ignore_measurement: bool = False, ignore_reset: bool = False) -> "Tableau":
        return Tableau._wrap(_core.TableauCore.from_circuit(_stim_text_of(circuit), bool(ignore_noise), bool(ignore_measurement), bool(ignore_reset)))

    @staticmethod
    def random(num_qubits: int) -> "Tableau":
        """A uniformly random Clifford (Bravyi–Maslov sampling, as Stim)."""
        return Tableau._wrap(_core.TableauCore.random(int(num_qubits), _seed(None)))

    @staticmethod
    def iter_all(num_qubits: int, *, unsigned: bool = False) -> "TableauIterator":
        """Every tableau on ``num_qubits`` (with all signs, unless ``unsigned``)."""
        return TableauIterator(num_qubits, unsigned)


class TableauIterator:
    """Iterates every Clifford tableau of a size: each symplectic basis once, then (unless
    unsigned) each assignment of signs."""

    def __init__(self, num_qubits: int, unsigned: bool = False) -> None:
        self._n = int(num_qubits)
        self._unsigned = bool(unsigned)
        self._gen = self._generate()

    def __iter__(self) -> "TableauIterator":
        return self

    def __next__(self) -> Tableau:
        return next(self._gen)

    def _generate(self) -> Iterator[Tableau]:
        n = self._n
        all_ps = [list(p) for p in itertools.product(range(4), repeat=n)]
        bits = [(tuple(1 if v in (1, 2) else 0 for v in p), tuple(1 if v in (2, 3) else 0 for v in p)) for p in all_ps]

        def sym(a: int, b: int) -> int:
            (xa, za), (xb, zb) = bits[a], bits[b]
            return sum((xa[k] & zb[k]) ^ (xb[k] & za[k]) for k in range(n)) & 1

        def rows(chosen: List[int]) -> List[int]:
            out = []
            for c in chosen:
                x, z = bits[c]
                out.append(int("".join(map(str, x + z)) or "0", 2))
            return out

        def independent(chosen: List[int]) -> bool:
            basis: List[int] = []
            for v in rows(chosen):
                for b in basis:
                    v = min(v, v ^ b)
                if v == 0:
                    return False
                basis.append(v)
            return True

        def rec(k: int, xs: List[int], zs: List[int]) -> Iterator[Tuple[List[int], List[int]]]:
            if k == n:
                yield xs, zs
                return
            prior = xs + zs
            for x in range(1, len(all_ps)):
                if any(sym(x, p) for p in prior) or not independent(prior + [x]):
                    continue
                for z in range(1, len(all_ps)):
                    if not sym(x, z) or any(sym(z, p) for p in prior) or not independent(prior + [x, z]):
                        continue
                    yield from rec(k + 1, xs + [x], zs + [z])

        for xs, zs in rec(0, [], []):
            sign_choices = [(0,) * (2 * n)] if self._unsigned else itertools.product((0, 2), repeat=2 * n)
            for signs in sign_choices:
                px = [_core.PauliStringCore.from_paulis(all_ps[x], signs[k]) for k, x in enumerate(xs)]
                pz = [_core.PauliStringCore.from_paulis(all_ps[z], signs[n + k]) for k, z in enumerate(zs)]
                yield Tableau._wrap(_core.TableauCore.from_conjugated_generators(px, pz))


# ---------------------------------------------------------------------------------------------
# TableauSimulator


class TableauSimulator:
    """A stabilizer state you act on gate by gate, as ``stim.TableauSimulator``: it stores the
    inverse of the Clifford that prepares the state, and grows as you touch new qubits.

    >>> s = TableauSimulator(seed=1)
    >>> s.h(0)
    >>> s.cnot(0, 1)
    >>> s.peek_observable_expectation(PauliString("ZZ"))
    1
    """

    __slots__ = ("_s",)

    def __init__(self, *, seed: Optional[int] = None) -> None:
        self._s = _core.TableauSimulatorCore(_seed(seed))

    def copy(self, *, copy_rng: bool = False, seed: Optional[int] = None) -> "TableauSimulator":
        if copy_rng and seed is not None:
            raise ValueError("seed and copy_rng are incompatible")
        out = TableauSimulator.__new__(TableauSimulator)
        out._s = self._s.copy(bool(copy_rng), None if copy_rng else _seed(seed))
        return out

    @property
    def num_qubits(self) -> int:
        """How many qubits the simulator tracks."""
        return self._s.num_qubits

    def set_num_qubits(self, new_num_qubits: int) -> None:
        self._s.set_num_qubits(int(new_num_qubits))

    # Running things.
    def do(self, circuit_or_pauli_string: Any) -> None:
        """Apply a circuit, an instruction, a repeat block or a Pauli string."""
        if isinstance(circuit_or_pauli_string, PauliString):
            self.do_pauli_string(circuit_or_pauli_string)
        else:
            self._s.do_circuit(_stim_text_of(circuit_or_pauli_string))

    def do_circuit(self, circuit: Any) -> None:
        self._s.do_circuit(_stim_text_of(circuit))

    def do_pauli_string(self, pauli_string: PauliString) -> None:
        self._s.do_pauli_string(pauli_string._p)

    def do_tableau(self, tableau: Tableau, targets: List[int]) -> None:
        self._s.do_tableau(tableau._t, [int(t) for t in targets])

    def _gate(self, name: str, targets: Sequence[Any], args: Sequence[float] = ()) -> None:
        bits = [GateTarget(t)._v for t in targets]
        self._s.do_instruction(name, [float(a) for a in args], bits, "")

    # Measurement.
    def measure(self, target: int) -> bool:
        self._gate("M", [target])
        return self._s.current_measurement_record()[-1]

    def measure_many(self, *args: int) -> List[bool]:
        self._gate("M", list(args))
        rec = self._s.current_measurement_record()
        return rec[len(rec) - len(args):] if args else []

    def measure_observable(self, observable: PauliString, *, flip_probability: float = 0.0) -> bool:
        return self._s.measure_pauli_string(observable._p, float(flip_probability))

    def measure_kickback(self, target: int) -> Tuple[bool, Optional[PauliString]]:
        b, k = self._s.measure_kickback(int(target), 3)
        return b, None if k is None else PauliString._wrap(k)

    def reset(self, *args: int) -> None:
        self._gate("R", args)

    def reset_x(self, *args: int) -> None:
        self._gate("RX", args)

    def reset_y(self, *args: int) -> None:
        self._gate("RY", args)

    def reset_z(self, *args: int) -> None:
        self._gate("R", args)

    # Peeks and postselection.
    def peek_x(self, target: int) -> int:
        return self._s.peek(int(target), 1)

    def peek_y(self, target: int) -> int:
        return self._s.peek(int(target), 2)

    def peek_z(self, target: int) -> int:
        return self._s.peek(int(target), 3)

    def peek_bloch(self, target: int) -> PauliString:
        return PauliString._wrap(self._s.peek_bloch(int(target)))

    def peek_observable_expectation(self, observable: PauliString) -> int:
        return self._s.peek_observable_expectation(observable._p)

    def _post(self, targets: Any, desired_value: bool, basis: int) -> None:
        qs = [int(targets)] if isinstance(targets, (int, np.integer)) else [int(t) for t in targets]
        self._s.postselect(qs, bool(desired_value), basis)

    def postselect_x(self, targets: Union[int, Iterable[int]], *, desired_value: bool) -> None:
        self._post(targets, desired_value, 1)

    def postselect_y(self, targets: Union[int, Iterable[int]], *, desired_value: bool) -> None:
        self._post(targets, desired_value, 2)

    def postselect_z(self, targets: Union[int, Iterable[int]], *, desired_value: bool) -> None:
        self._post(targets, desired_value, 3)

    def postselect_observable(self, observable: PauliString, *, desired_value: bool = False) -> None:
        self._s.postselect_observable(observable._p, bool(desired_value))

    # State.
    def canonical_stabilizers(self) -> List[PauliString]:
        return [PauliString._wrap(p) for p in self._s.canonical_stabilizers()]

    def current_inverse_tableau(self) -> Tableau:
        return Tableau._wrap(self._s.current_inverse_tableau())

    def set_inverse_tableau(self, new_inverse_tableau: Tableau) -> None:
        self._s.set_inverse_tableau(new_inverse_tableau._t)

    def current_measurement_record(self) -> List[bool]:
        return list(self._s.current_measurement_record())

    def state_vector(self, *, endian: str = "little") -> np.ndarray:
        """The state's amplitudes (complex64), first nonzero one real and positive."""
        return _complex_array(self._s.state_vector(_endian(endian)), (1 << self.num_qubits,))

    def set_state_from_state_vector(self, state_vector: Iterable[complex], *, endian: str) -> None:
        t = Tableau.from_state_vector(state_vector, endian=endian)
        self._s.set_inverse_tableau(t._t.inverse(False))

    def set_state_from_stabilizers(self, stabilizers: Iterable[PauliString], *, allow_redundant: bool = False, allow_underconstrained: bool = False) -> None:
        t = _core.TableauCore.from_stabilizers([PauliString(p)._p for p in stabilizers], bool(allow_redundant), bool(allow_underconstrained), True)
        self._s.set_inverse_tableau(t)

    # Noise.
    def depolarize1(self, *targets: int, p: float) -> None:
        self._gate("DEPOLARIZE1", targets, [p])

    def depolarize2(self, *targets: int, p: float) -> None:
        self._gate("DEPOLARIZE2", targets, [p])

    def x_error(self, *targets: int, p: float) -> None:
        self._gate("X_ERROR", targets, [p])

    def y_error(self, *targets: int, p: float) -> None:
        self._gate("Y_ERROR", targets, [p])

    def z_error(self, *targets: int, p: float) -> None:
        self._gate("Z_ERROR", targets, [p])


def _gate_method(name: str) -> Any:
    def method(self: TableauSimulator, *args: Any) -> None:
        self._gate(name, args)

    method.__name__ = name.lower()
    method.__doc__ = f"Apply {name} to the targets (pairs for two-qubit gates)."
    return method


for _name in ["C_XYZ", "C_ZYX", "CNOT", "CX", "CY", "CZ", "H", "H_XY", "H_XZ", "H_YZ", "ISWAP", "ISWAP_DAG", "S", "S_DAG", "SQRT_X", "SQRT_X_DAG", "SQRT_Y", "SQRT_Y_DAG", "SWAP", "X", "XCX", "XCY", "XCZ", "Y", "YCX", "YCY", "YCZ", "Z", "ZCX", "ZCY", "ZCZ"]:
    setattr(TableauSimulator, _name.lower(), _gate_method(_name))


# ---------------------------------------------------------------------------------------------
# Gate targets

_TARGET_INVERTED = 1 << 31
_TARGET_X = 1 << 30
_TARGET_Z = 1 << 29
_TARGET_REC = 1 << 28
_TARGET_COMBINER = 1 << 27
_TARGET_SWEEP = 1 << 26
_VALUE_MASK = (1 << 24) - 1


class GateTarget:
    """A target of a circuit instruction, as ``stim.GateTarget``: a qubit (maybe inverted), a
    Pauli target, a measurement record, a sweep bit, or a combiner."""

    __slots__ = ("_v",)

    def __init__(self, value: Any) -> None:
        if isinstance(value, GateTarget):
            self._v = value._v
        elif isinstance(value, (int, np.integer)) and not isinstance(value, bool):
            if not 0 <= int(value) <= _VALUE_MASK:
                raise ValueError(f"qubit target {value} out of range")
            self._v = int(value)
        elif isinstance(value, str):
            self._v = _core.parse_gate_target(value)
        else:
            raise ValueError(f"Don't know how to make a GateTarget from {value!r}")

    @staticmethod
    def _raw(v: int) -> "GateTarget":
        out = GateTarget.__new__(GateTarget)
        out._v = v
        return out

    @staticmethod
    def combiner() -> "GateTarget":
        return GateTarget._raw(_TARGET_COMBINER)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, GateTarget):
            return self._v == other._v
        return NotImplemented

    def __ne__(self, other: object) -> bool:
        r = self.__eq__(other)
        return r if r is NotImplemented else not r

    def __hash__(self) -> int:
        return hash(("GateTarget", self._v))

    @property
    def value(self) -> int:
        """The qubit, lookback (negative) or sweep bit index."""
        v = self._v & _VALUE_MASK
        return -v if self._v & _TARGET_REC else v

    @property
    def qubit_value(self) -> Optional[int]:
        if self._v & (_TARGET_REC | _TARGET_SWEEP | _TARGET_COMBINER):
            return None
        return self._v & _VALUE_MASK

    @property
    def is_combiner(self) -> bool:
        return self._v == _TARGET_COMBINER

    @property
    def is_inverted_result_target(self) -> bool:
        return bool(self._v & _TARGET_INVERTED)

    @property
    def is_measurement_record_target(self) -> bool:
        return bool(self._v & _TARGET_REC)

    @property
    def is_sweep_bit_target(self) -> bool:
        return bool(self._v & _TARGET_SWEEP)

    @property
    def is_qubit_target(self) -> bool:
        return not (self._v & (_TARGET_X | _TARGET_Z | _TARGET_REC | _TARGET_SWEEP | _TARGET_COMBINER))

    @property
    def is_x_target(self) -> bool:
        return bool(self._v & _TARGET_X) and not (self._v & _TARGET_Z)

    @property
    def is_y_target(self) -> bool:
        return bool(self._v & _TARGET_X) and bool(self._v & _TARGET_Z)

    @property
    def is_z_target(self) -> bool:
        return bool(self._v & _TARGET_Z) and not (self._v & _TARGET_X)

    @property
    def pauli_type(self) -> str:
        """'I', 'X', 'Y' or 'Z'."""
        return "IZXY"[(self._v >> 29) & 3]

    def target_str(self) -> str:
        """The target as it appears in a circuit's text (``!X3``, ``rec[-2]``)."""
        return _core.gate_target_text(self._v)

    def __repr__(self) -> str:
        v = self._v
        if v == _TARGET_COMBINER:
            return "stabilizer_qec.GateTarget.combiner()"
        if self.is_qubit_target:
            if self.is_inverted_result_target:
                return f"stabilizer_qec.target_inv({self.value})"
            return f"stabilizer_qec.GateTarget({self.value})"
        if self.is_measurement_record_target:
            return f"stabilizer_qec.target_rec({self.value})"
        if self.is_sweep_bit_target:
            return f"stabilizer_qec.target_sweep_bit({self.value})"
        name = {"X": "target_x", "Y": "target_y", "Z": "target_z"}[self.pauli_type]
        inv = ", invert=True" if self.is_inverted_result_target else ""
        return f"stabilizer_qec.{name}({self.value}{inv})"


def _qubit_index(q: Any) -> int:
    if isinstance(q, GateTarget):
        if q.qubit_value is None or not q.is_qubit_target:
            raise ValueError(f"Expected a qubit target: {q!r}")
        return q.qubit_value
    q = int(q)
    if not 0 <= q <= _VALUE_MASK:
        raise ValueError(f"qubit target {q} out of range")
    return q


def target_rec(lookback_index: int) -> GateTarget:
    """``rec[lookback_index]`` (negative)."""
    k = int(lookback_index)
    if not -(1 << 24) < k < 0:
        raise ValueError(f"lookback_index must be negative and not too large: {k}")
    return GateTarget._raw((-k) | _TARGET_REC)


def target_inv(qubit_index: Any) -> GateTarget:
    """``!q``: a qubit whose measurement result is inverted (or an inverted Pauli target)."""
    if isinstance(qubit_index, GateTarget):
        if qubit_index._v & (_TARGET_REC | _TARGET_SWEEP | _TARGET_COMBINER):
            raise ValueError(f"Target '{qubit_index!r}' doesn't have a defined inverse.")
        return GateTarget._raw(qubit_index._v ^ _TARGET_INVERTED)
    return GateTarget._raw(_qubit_index(qubit_index) | _TARGET_INVERTED)


def target_pauli(qubit_index: Any, pauli: Any, invert: bool = False) -> GateTarget:
    """A Pauli target from a qubit and a Pauli (0..3, or 'X', 'Y', 'Z', 'I')."""
    if isinstance(qubit_index, GateTarget):
        qubit_index = qubit_index.value
        if qubit_index < 0:
            raise ValueError("target_pauli needs a qubit")
    p = _pauli_value(pauli)
    q = _qubit_index(qubit_index)
    if p == 0:
        return GateTarget._raw(q | (_TARGET_INVERTED if invert else 0))
    bits = {1: _TARGET_X, 2: _TARGET_X | _TARGET_Z, 3: _TARGET_Z}[p]
    return GateTarget._raw(q | bits | (_TARGET_INVERTED if invert else 0))


def target_x(qubit_index: Any, invert: bool = False) -> GateTarget:
    return target_pauli(qubit_index, 1, invert)


def target_y(qubit_index: Any, invert: bool = False) -> GateTarget:
    return target_pauli(qubit_index, 2, invert)


def target_z(qubit_index: Any, invert: bool = False) -> GateTarget:
    return target_pauli(qubit_index, 3, invert)


def target_combiner() -> GateTarget:
    return GateTarget.combiner()


def target_sweep_bit(sweep_bit_index: int) -> GateTarget:
    k = int(sweep_bit_index)
    if not 0 <= k <= _VALUE_MASK:
        raise ValueError("sweep_bit_index out of range")
    return GateTarget._raw(k | _TARGET_SWEEP)


def target_combined_paulis(paulis: Any, invert: bool = False) -> List[GateTarget]:
    """The targets of a Pauli product (with combiners), from a PauliString or a list of Pauli
    targets; the inversion of the PauliString's sign (or ``invert``) goes on the first."""
    out: List[GateTarget] = []
    if isinstance(paulis, PauliString):
        if paulis.sign not in (1, -1):
            raise ValueError("Imaginary sign")
        invert = bool(invert) ^ (paulis.sign == -1)
        for q in range(len(paulis)):
            p = paulis[q]
            if p:
                if out:
                    out.append(GateTarget.combiner())
                out.append(target_pauli(q, p))
    else:
        for t in paulis:
            t = GateTarget(t)
            if not (t.is_x_target or t.is_y_target or t.is_z_target):
                raise ValueError(f"Expected a pauli target but got {t!r}")
            if out:
                out.append(GateTarget.combiner())
            if t.is_inverted_result_target:
                invert = not invert
                t = GateTarget._raw(t._v ^ _TARGET_INVERTED)
            out.append(t)
    if not out:
        raise ValueError("Identity pauli product")
    if invert:
        out[0] = GateTarget._raw(out[0]._v ^ _TARGET_INVERTED)
    return out


# ---------------------------------------------------------------------------------------------
# Flows (the full Flow API is built in W2 task 7; this is the value type GateData needs)


class Flow:
    """A stabilizer flow ``input -> output xor rec[...] xor obs[...]``, as ``stim.Flow``."""

    __slots__ = ("_text",)

    def __init__(self, arg: Any = None, /, *, input: Optional[PauliString] = None, output: Optional[PauliString] = None, measurements: Optional[Iterable[Any]] = None, included_observables: Optional[Iterable[int]] = None) -> None:
        if arg is not None:
            if any(x is not None for x in (input, output, measurements, included_observables)):
                raise ValueError("Can't specify both a positional argument and `input=`/`output=`/`measurements=`/`included_observables=`.")
            if isinstance(arg, Flow):
                self._text = arg._text
            elif isinstance(arg, str):
                self._text = _core.flow_text(arg)
            elif type(arg).__name__ == "Flow":  # a stim.Flow
                self._text = _core.flow_text(str(arg))
            else:
                raise ValueError(f"Don't know how to make a Flow from {arg!r}")
            return
        i = str(PauliString(input)) if input is not None else "+"
        o = str(PauliString(output)) if output is not None else "+"
        ms = []
        for m in measurements or []:
            if isinstance(m, GateTarget) or type(m).__name__ == "GateTarget":
                if not m.is_measurement_record_target:
                    raise ValueError(f"Not a measurement offset: {m!r}")
                ms.append(int(m.value))
            else:
                ms.append(int(m))
        self._text = _core.flow_from_parts(i, o, ms, [int(k) for k in (included_observables or [])])

    @staticmethod
    def _of(text: str) -> "Flow":
        out = Flow.__new__(Flow)
        out._text = text
        return out

    def _parts(self) -> Tuple[str, str, List[int], List[int]]:
        return _core.flow_parts(self._text)

    def input_copy(self) -> PauliString:
        return PauliString(self._parts()[0])

    def output_copy(self) -> PauliString:
        return PauliString(self._parts()[1])

    def measurements_copy(self) -> List[int]:
        return list(self._parts()[2])

    def included_observables_copy(self) -> List[int]:
        return list(self._parts()[3])

    def __mul__(self, rhs: "Flow") -> "Flow":
        if not isinstance(rhs, Flow):
            return NotImplemented
        return Flow._of(_core.flow_mul(self._text, rhs._text))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Flow):
            return NotImplemented
        return self._parts() == other._parts()

    def __ne__(self, other: object) -> bool:
        r = self.__eq__(other)
        return r if r is NotImplemented else not r

    def __hash__(self) -> int:
        return hash(("Flow", self._text))

    def __str__(self) -> str:
        return self._text

    def __repr__(self) -> str:
        return f'stabilizer_qec.Flow("{self._text}")'


# ---------------------------------------------------------------------------------------------
# Gate data

_GATES: Dict[str, Dict[str, Any]] = {}
_ALIASES: Dict[str, str] = {}


def _load_gates() -> None:
    if _GATES:
        return
    for d in _core.gate_data_table():
        _GATES[d["name"]] = d
        for a in d["aliases"]:
            _ALIASES[a.upper()] = d["name"]


class GateData:
    """What Stim knows about one instruction, as ``stim.GateData``."""

    __slots__ = ("_d",)

    def __init__(self, name: str) -> None:
        _load_gates()
        key = _ALIASES.get(str(name).upper())
        if key is None:
            raise IndexError(f"Gate not found: '{name}'")
        self._d = _GATES[key]

    @property
    def name(self) -> str:
        return self._d["name"]

    @property
    def aliases(self) -> List[str]:
        return list(self._d["aliases"])

    is_noisy_gate = property(lambda self: self._d["is_noisy_gate"])
    is_reset = property(lambda self: self._d["is_reset"])
    is_single_qubit_gate = property(lambda self: self._d["is_single_qubit_gate"])
    is_symmetric_gate = property(lambda self: self._d["is_symmetric_gate"])
    is_two_qubit_gate = property(lambda self: self._d["is_two_qubit_gate"])
    is_unitary = property(lambda self: self._d["is_unitary"])
    produces_measurements = property(lambda self: self._d["produces_measurements"])
    takes_measurement_record_targets = property(lambda self: self._d["takes_measurement_record_targets"])
    takes_pauli_targets = property(lambda self: self._d["takes_pauli_targets"])

    @property
    def num_parens_arguments_range(self) -> range:
        a, b = self._d["args"]
        return range(a, b)

    @property
    def tableau(self) -> Optional[Tableau]:
        if not self._d["tableau"]:
            if self.is_unitary:
                raise IndexError(f"{self.name} doesn't have 1q or 2q tableau data.")
            return None
        return Tableau.from_named_gate(self.name)

    @property
    def unitary_matrix(self) -> Optional[np.ndarray]:
        u = self._d["unitary"]
        if not u:
            return None
        d = int(round(math.sqrt(len(u))))
        return np.array([complex(re, im) for re, im in u], dtype=np.complex64).reshape((d, d))

    @property
    def flows(self) -> Optional[List[Flow]]:
        if not self._d["flows"]:
            return None
        return [Flow(f) for f in self._d["flows"]]

    @property
    def inverse(self) -> Optional["GateData"]:
        inv = self._d["inverse"]
        return None if inv is None else GateData(inv)

    @property
    def generalized_inverse(self) -> "GateData":
        return GateData(self._d["generalized_inverse"])

    def hadamard_conjugated(self, *, unsigned: bool = False) -> Optional["GateData"]:
        """The gate H·G·H equals (up to Paulis with ``unsigned``), or None if none does."""
        h = self._d["hadamard_conjugated_unsigned" if unsigned else "hadamard_conjugated"]
        if h is not None:
            return GateData(h)
        return None

    def __eq__(self, other: object) -> bool:
        if isinstance(other, GateData):
            return self.name == other.name
        return NotImplemented

    def __ne__(self, other: object) -> bool:
        r = self.__eq__(other)
        return r if r is NotImplemented else not r

    def __hash__(self) -> int:
        return hash(("GateData", self.name))

    def __repr__(self) -> str:
        return f"stabilizer_qec.gate_data({self.name!r})"

    def __str__(self) -> str:
        lines = ["stabilizer_qec.GateData {", f"    .name = {self.name!r}", f"    .aliases = {self.aliases!r}"]
        for k in ["is_noisy_gate", "is_reset", "is_single_qubit_gate", "is_two_qubit_gate", "is_unitary"]:
            lines.append(f"    .{k} = {getattr(self, k)}")
        lines.append(f"    .num_parens_arguments_range = {self.num_parens_arguments_range!r}")
        for k in ["produces_measurements", "takes_measurement_record_targets", "takes_pauli_targets"]:
            lines.append(f"    .{k} = {getattr(self, k)}")
        t = self.tableau if self._d["tableau"] else None
        if t is not None:
            lines.append("    .tableau = " + repr(t).replace("\n", "\n    "))
            u = self.unitary_matrix
            rows = ", ".join("[" + ", ".join(repr(complex(v)) for v in row) + "]" for row in u)
            lines.append(f"    .unitary_matrix = np.array([{rows}], dtype=np.complex64)")
        if self.flows is not None and t is None:
            lines.append("    .flows = [")
            for f in self.flows:
                lines.append(f"        {f!r},")
            lines.append("    ]")
        lines.append("}")
        return "\n".join(lines)


def gate_data(name: Optional[str] = None) -> Union[GateData, Dict[str, GateData]]:
    """The data of one gate, or (with no name) a dict of every gate's, by canonical name."""
    _load_gates()
    if name is None:
        return {k: GateData(k) for k in sorted(_GATES)}
    return GateData(name)


# ---------------------------------------------------------------------------------------------


def _stim_text_of(obj: Any) -> str:
    """Stim text of a circuit, instruction or repeat block, with exact arguments."""
    exact = getattr(obj, "_stim_exact_text", None)
    if exact is not None:
        return exact()
    if isinstance(obj, str):
        return obj
    return str(obj)


# ---------------------------------------------------------------------------------------------
# Instructions and repeat blocks


def _fmt_arg(v: float) -> str:
    """An argument as Stim prints it: an integer as an integer, else 6 significant digits."""
    v = float(v)
    if v.is_integer() and abs(v) < 9.2e18:
        return str(int(v))
    return format(v, "g")


def _gate_target_of(t: Any) -> "GateTarget":
    if isinstance(t, GateTarget):
        return t
    if isinstance(t, (int, np.integer, str)) and not isinstance(t, bool):
        return GateTarget(t)
    if hasattr(t, "is_combiner") and hasattr(t, "value"):  # a stim.GateTarget
        from ._circuit import _target

        return GateTarget(_target(t))
    raise TypeError(f"a target is a qubit index, a string such as 'rec[-1]', or a GateTarget, not {type(t).__name__}")


class CircuitInstruction:
    """One instruction of a circuit, as ``stim.CircuitInstruction``: a gate, its targets, its
    parens arguments and its tag.

    >>> CircuitInstruction("CX", [0, 1, 2, 3]).target_groups()
    [[stabilizer_qec.GateTarget(0), stabilizer_qec.GateTarget(1)], [stabilizer_qec.GateTarget(2), stabilizer_qec.GateTarget(3)]]
    """

    __slots__ = ("_name", "_targets", "_args", "_tag")

    def __init__(self, name: str, targets: Optional[Iterable[Any]] = None, gate_args: Optional[Iterable[float]] = None, *, tag: str = "") -> None:
        if targets is None and gate_args is None and not tag and isinstance(name, str) and (" " in name.strip() or "(" in name or "[" in name):
            items = _core.circuit_items(name)
            if len(items) != 1 or items[0][0] != "op":
                raise ValueError(f"Expected exactly one instruction but got {name!r}")
            _, n, t, a, tg = items[0]
            self._name, self._tag, self._args, self._targets = n, t, list(a), [GateTarget._raw(v) for v in tg]
        else:
            self._name = GateData(name).name
            self._targets = [_gate_target_of(t) for t in (targets or [])]
            self._args = [float(a) for a in (gate_args or [])]
            self._tag = str(tag)
        from ._circuit import Circuit

        Circuit(self._stim_exact_text())  # validates as Stim does on construction

    @classmethod
    def _make(cls, name: str, tag: str, args: List[float], targets: List[int]) -> "CircuitInstruction":
        out = cls.__new__(cls)
        out._name, out._tag, out._args, out._targets = name, tag, list(args), [GateTarget._raw(v) for v in targets]
        return out

    @property
    def name(self) -> str:
        return self._name

    @property
    def tag(self) -> str:
        return self._tag

    @property
    def num_measurements(self) -> int:
        """How many measurement results the instruction records."""
        return _core.instruction_num_measurements(self._name, [t._v for t in self._targets])

    def gate_args_copy(self) -> List[float]:
        return list(self._args)

    def targets_copy(self) -> List["GateTarget"]:
        return list(self._targets)

    def target_groups(self) -> List[List["GateTarget"]]:
        """The targets split into the groups the gate acts on (pairs, products, ...)."""
        d = GateData(self._name)._d
        f = d["flags"]
        combiners, single, pairs, pauli_string, records = 1 << 12, 1 << 15, 1 << 6, 1 << 7, 1 << 8
        t = self._targets
        out: List[List[GateTarget]] = []
        start = 0
        while start < len(t):
            if f & combiners:
                end = start + 1
                while end < len(t) and t[end].is_combiner:
                    end += 2
                out.append([x for x in t[start:end] if not x.is_combiner])
            else:
                if f & single:
                    end = start + 1
                elif f & pairs:
                    end = start + 2
                elif f & pauli_string:
                    end = len(t)
                elif f & records or self._name in ("MPAD", "QUBIT_COORDS"):
                    end = start + 1
                else:
                    raise ValueError(f"Not implemented: splitting {self}")
                out.append(list(t[start:end]))
            start = end
        return out

    def _stim_exact_text(self) -> str:
        return _core.instruction_text(self._name, self._args, [t._v for t in self._targets], self._tag, True)

    def __str__(self) -> str:
        return _core.instruction_text(self._name, self._args, [t._v for t in self._targets], self._tag, False)

    def __repr__(self) -> str:
        targets = ", ".join(repr(t) for t in self._targets)
        args = ", ".join(_fmt_arg(a) for a in self._args)
        tag = f", tag={self._tag!r}" if self._tag else ""
        return f"stabilizer_qec.CircuitInstruction({self._name!r}, [{targets}], [{args}]{tag})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CircuitInstruction):
            return NotImplemented
        return (self._name, self._tag, self._args, [t._v for t in self._targets]) == (other._name, other._tag, other._args, [t._v for t in other._targets])

    def __ne__(self, other: object) -> bool:
        r = self.__eq__(other)
        return r if r is NotImplemented else not r

    def __hash__(self) -> int:
        return hash((self._name, self._tag, tuple(self._args), tuple(t._v for t in self._targets)))


class CircuitRepeatBlock:
    """A ``REPEAT`` block of a circuit, as ``stim.CircuitRepeatBlock``."""

    __slots__ = ("_count", "_body", "_tag")

    def __init__(self, repeat_count: int, body: Any, *, tag: str = "") -> None:
        from ._circuit import Circuit

        if isinstance(repeat_count, bool) or not isinstance(repeat_count, (int, np.integer)) or repeat_count <= 0:
            raise ValueError("Can't repeat 0 times.")
        self._count = int(repeat_count)
        self._body = Circuit(body).copy()
        self._tag = str(tag)

    @property
    def repeat_count(self) -> int:
        return self._count

    @property
    def name(self) -> str:
        return "REPEAT"

    @property
    def tag(self) -> str:
        return self._tag

    @property
    def num_measurements(self) -> int:
        return self._count * self._body.num_measurements

    def body_copy(self) -> Any:
        return self._body.copy()

    def _stim_exact_text(self) -> str:
        tag = f"[{_escape_tag(self._tag)}]" if self._tag else ""
        body = self._body._stim_exact_text()
        return f"REPEAT{tag} {self._count} {{\n{body}\n}}"

    def __repr__(self) -> str:
        tag = f", tag={self._tag!r}" if self._tag else ""
        return f"stabilizer_qec.CircuitRepeatBlock({self._count}, {self._body!r}{tag})"

    __str__ = __repr__

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CircuitRepeatBlock):
            return NotImplemented
        return self._count == other._count and self._tag == other._tag and self._body == other._body

    def __ne__(self, other: object) -> bool:
        r = self.__eq__(other)
        return r if r is NotImplemented else not r

    __hash__ = None  # type: ignore[assignment]


def _escape_tag(tag: str) -> str:
    return tag.replace("\\", "\\B").replace("\n", "\\n").replace("\r", "\\r").replace("]", "\\C")


def _items_exact_text(items: List[Any], indent: str = "") -> str:
    lines = []
    for it in items:
        if it[0] == "op":
            _, name, tag, args, targets = it
            lines.append(indent + _core.instruction_text(name, args, targets, tag, True))
        else:
            _, count, tag, body = it
            t = f"[{_escape_tag(tag)}]" if tag else ""
            lines.append(f"{indent}REPEAT{t} {count} {{")
            inner = _items_exact_text(body, indent + "    ")
            lines.append(inner)
            lines.append(indent + "}")
    return "\n".join(lines)


def _item_object(it: Any) -> Any:
    from ._circuit import Circuit

    if it[0] == "op":
        _, name, tag, args, targets = it
        return CircuitInstruction._make(name, tag, args, targets)
    _, count, tag, body = it
    out = CircuitRepeatBlock.__new__(CircuitRepeatBlock)
    out._count, out._tag, out._body = count, tag, Circuit(_items_exact_text(body))
    return out

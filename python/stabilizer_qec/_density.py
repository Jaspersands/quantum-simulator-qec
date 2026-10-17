"""The density-matrix simulator (up to 12 qubits): every instruction the state vector runs,
noise as channels, with amplitude damping, phase damping and T1/T2 relaxation
(``I_ERROR[THERMAL_RELAXATION(t=…, t1=…, t2=…)]``)."""

from __future__ import annotations

from typing import Any, Iterable, List, Optional, Sequence

import numpy as np

from . import _core
from ._stim import PauliString, _endian, _seed, _stim_text_of
from ._util import call

_CODE = {1: 1, 2: 3, 3: 2}  # PauliString's 1 X, 2 Y, 3 Z as the engine's X 1, Y 3, Z 2.


def _reorder(rho: np.ndarray, n: int) -> np.ndarray:
    """ρ with its qubits' bit order reversed (little-endian to big-endian and back)."""
    if n <= 1:
        return rho
    axes = list(range(n))[::-1]
    t = rho.reshape([2] * (2 * n)).transpose(axes + [n + a for a in axes])
    return t.reshape(1 << n, 1 << n)


def _matrix(data: bytes, n: int, little: bool) -> np.ndarray:
    rho = np.frombuffer(data, dtype="<c16").astype(np.complex128).reshape(1 << n, 1 << n)
    return rho if little else _reorder(rho, n)


def _paulis(pauli_string: Any) -> List[tuple]:
    p = pauli_string if isinstance(pauli_string, PauliString) else PauliString(pauli_string)
    return [(q, _CODE[int(v)]) for q, v in enumerate(p) if int(v)], p.sign


def reduced_density_matrix(rho: np.ndarray, keep: Sequence[int], *, endian: str = "little") -> np.ndarray:
    """The partial trace of ``rho`` onto qubits ``keep``, in that order (the first of them bit
    0 for little-endian, the most significant bit for big-endian)."""
    rho = np.asarray(rho, dtype=np.complex128)
    n = int(round(np.log2(rho.shape[0])))
    if rho.shape != (1 << n, 1 << n):
        raise ValueError(f"a density matrix is 2^n by 2^n, got shape {rho.shape}")
    keep = [int(q) for q in keep]
    if len(set(keep)) != len(keep) or any(not 0 <= q < n for q in keep):
        raise ValueError(f"qubits to keep must be distinct, in 0..{n - 1}")
    little = _endian(endian)
    if little:
        rho = _reorder(rho, n)
    # Big-endian: axis k is qubit k.
    t = rho.reshape([2] * (2 * n))
    gone = [q for q in range(n) if q not in keep]
    letters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    rows = [letters[q] for q in range(n)]
    cols = [letters[q] if q in gone else letters[n + q] for q in range(n)]
    order = list(keep)
    out = "".join(letters[q] for q in order) + "".join(letters[n + q] for q in order)
    r = np.einsum("".join(rows) + "".join(cols) + "->" + out, t).reshape(1 << len(keep), 1 << len(keep))
    return _reorder(r, len(keep)) if little else r


def _psd_eigenvalues(w: np.ndarray) -> np.ndarray:
    """A PSD matrix's eigenvalues with round-off (a few ulps of the largest) taken as 0: its
    square root, ~1e-8 for a round-off of 1e-16, would otherwise count."""
    top = max(float(w[-1]), 0.0) if len(w) else 0.0
    return np.where(w > 4 * np.finfo(float).eps * len(w) * top, w, 0.0)


def fidelity(rho: Any, sigma: Any) -> float:
    """Uhlmann's fidelity (Tr √(√ρ σ √ρ))²; either may be a state vector."""
    a, b = np.asarray(rho, dtype=np.complex128), np.asarray(sigma, dtype=np.complex128)
    if a.ndim == 1 and b.ndim == 1:
        return float(abs(np.vdot(a, b)) ** 2)
    if a.ndim == 1:
        a, b = b, a
    if b.ndim == 1:
        return float(np.real(np.vdot(b, a @ b)))
    wa, va = np.linalg.eigh((a + a.conj().T) / 2)
    wa = _psd_eigenvalues(wa)
    wb = _psd_eigenvalues(np.linalg.eigvalsh((b + b.conj().T) / 2))
    if np.count_nonzero(wa) <= 1 or np.count_nonzero(wb) <= 1:
        # Either is pure (a unitary's Choi state): the fidelity is Tr ρσ, with no square roots.
        return float(np.real(np.sum(a * b.T)))
    s = (va * np.sqrt(wa)) @ va.conj().T
    w = _psd_eigenvalues(np.linalg.eigvalsh(s @ b @ s))
    return float(np.sum(np.sqrt(w)) ** 2)


def von_neumann_entropy(rho: Any, *, base: float = 2) -> float:
    """−Tr ρ log ρ, in bits by default."""
    w = np.linalg.eigvalsh(np.asarray(rho, dtype=np.complex128))
    w = w[w > 1e-15]
    return float(-np.sum(w * np.log(w)) / np.log(base))


_H = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)
_SDAG = np.diag([1, -1j])


class DensityMatrixSimulator:
    """A density matrix you act on circuit by circuit, as ``TableauSimulator`` acts on a
    stabilizer state: it grows as circuits touch new qubits (at most 12), draws each recorded
    measurement outcome, and applies noise as channels.

    >>> import stabilizer_qec as sq
    >>> s = sq.DensityMatrixSimulator(seed=1)
    >>> s.do(sq.Circuit("H 0\\nCX 0 1\\nI_ERROR[AMPLITUDE_DAMPING(gamma=0.5)] 0"))
    >>> round(s.peek_observable_expectation(sq.PauliString("ZZ")), 6)
    0.5
    """

    __slots__ = ("_s",)

    def __init__(self, num_qubits: int = 0, *, seed: Optional[int] = None) -> None:
        self._s = call(_core.DensityMatrixCore, int(num_qubits), _seed(seed))

    def __copy__(self) -> "DensityMatrixSimulator":
        """``copy.copy``: an independent simulator (freshly seeded), not one sharing this state."""
        return self.copy()

    def __deepcopy__(self, memo: dict) -> "DensityMatrixSimulator":
        return self.copy()

    def copy(self, *, seed: Optional[int] = None) -> "DensityMatrixSimulator":
        out = DensityMatrixSimulator.__new__(DensityMatrixSimulator)
        out._s = self._s.copy(_seed(seed))
        return out

    @property
    def num_qubits(self) -> int:
        return self._s.num_qubits

    def set_num_qubits(self, new_num_qubits: int) -> None:
        """Grow to ``new_num_qubits`` (new qubits in |0⟩); it does not shrink."""
        call(self._s.grow, int(new_num_qubits))

    def do(self, circuit: Any) -> None:
        """Run a circuit (or its text), drawing each recorded outcome."""
        from ._circuit import Circuit

        c = circuit if isinstance(circuit, Circuit) else Circuit(_stim_text_of(circuit))
        call(self._s.do_circuit, c._c)

    def apply_channel(self, circuit: Any, *, max_branches: int = 1 << 20) -> None:
        """Run a circuit as a channel: its measurements' outcomes averaged away (followed one by
        one only when feedback reads them), none recorded."""
        from ._circuit import Circuit

        c = circuit if isinstance(circuit, Circuit) else Circuit(_stim_text_of(circuit))
        call(self._s.apply_channel, c._c, int(max_branches))

    def apply_unitary(self, qubits: Sequence[int], matrix: Any, *, endian: str = "little") -> None:
        """U ρ U† for a 2^k by 2^k matrix on ``qubits`` (``qubits[0]`` the least significant bit
        of its index for little-endian, the most significant for big-endian)."""
        qubits, u = self._operator(qubits, matrix, endian)
        call(self._s.apply_matrix, qubits, u.astype("<c16").tobytes())

    def apply_kraus(self, qubits: Sequence[int], operators: Iterable[Any], *, endian: str = "little") -> None:
        """Σ K ρ K† for Kraus operators on ``qubits``, as ``apply_unitary`` takes them."""
        ops = [self._operator(qubits, k, endian) for k in operators]
        if not ops:
            raise ValueError("no Kraus operators")
        call(self._s.apply_kraus, ops[0][0], [k.astype("<c16").tobytes() for _, k in ops])

    def _operator(self, qubits: Sequence[int], matrix: Any, endian: str):
        qubits = [int(q) for q in qubits]
        if not _endian(endian):
            qubits = qubits[::-1]
        u = np.asarray(matrix, dtype=np.complex128)
        if u.shape != (1 << len(qubits), 1 << len(qubits)):
            raise ValueError(f"a matrix on {len(qubits)} qubits is {1 << len(qubits)} by {1 << len(qubits)}, got shape {u.shape}")
        if qubits and max(qubits) >= self.num_qubits:
            self.set_num_qubits(max(qubits) + 1)
        return qubits, u

    def measure(self, qubit: int, *, basis: str = "Z", forced: Optional[bool] = None) -> bool:
        """Measure one qubit in the X, Y or Z basis, recording the outcome: ``forced`` if given
        and possible, else drawn."""
        q = int(qubit)
        if q >= self.num_qubits:
            self.set_num_qubits(q + 1)
        basis = basis.upper()
        if basis not in ("X", "Y", "Z"):
            raise ValueError(f"basis must be 'X', 'Y' or 'Z', got {basis!r}")
        to_z = {"X": _H, "Y": _H @ _SDAG, "Z": None}[basis]
        if to_z is not None:
            self.apply_unitary([q], to_z)
        out = bool(self._s.measure(q, None if forced is None else bool(forced)))
        if to_z is not None:
            self.apply_unitary([q], to_z.conj().T)
        return out

    def current_measurement_record(self) -> List[bool]:
        return list(self._s.record())

    def density_matrix(self, *, endian: str = "little") -> np.ndarray:
        """ρ (complex128), qubit 0 the least significant bit of an index for little-endian."""
        return _matrix(self._s.matrix(), self.num_qubits, _endian(endian))

    def set_density_matrix(self, rho: Any, *, endian: str = "little") -> None:
        rho = np.asarray(rho, dtype=np.complex128)
        n = int(round(np.log2(rho.shape[0]))) if rho.ndim == 2 and rho.shape[0] else -1
        if n < 0 or rho.shape != (1 << n, 1 << n):
            raise ValueError(f"a density matrix is 2^n by 2^n, got shape {rho.shape}")
        if not np.isfinite(rho).all():
            raise ValueError("the density matrix has entries that are not finite")
        if not np.allclose(rho, rho.conj().T, rtol=0, atol=1e-9):
            raise ValueError("the density matrix is not Hermitian")
        if abs(np.trace(rho).real - 1) > 1e-9:
            raise ValueError(f"the density matrix's trace is {np.trace(rho).real:.12g}, not 1")
        if np.linalg.eigvalsh(rho)[0] < -1e-9:
            raise ValueError("the density matrix has a negative eigenvalue")
        if not _endian(endian):
            rho = _reorder(rho, n)
        call(self._s.set_matrix, n, rho.astype("<c16").tobytes())

    def state_vector(self, *, endian: str = "little") -> np.ndarray:
        """The pure state ρ holds (first nonzero amplitude real and positive); an error if ρ is
        mixed."""
        rho = self.density_matrix(endian=endian)
        w, v = np.linalg.eigh(rho)
        if w[-1] < 1 - 1e-9:
            raise ValueError(f"the state is mixed (purity {self.purity():.6g})")
        psi = v[:, -1]
        k = int(np.argmax(np.abs(psi) > 1e-12))
        return psi * (abs(psi[k]) / psi[k])

    def probabilities(self, *, endian: str = "little") -> np.ndarray:
        """Each basis state's probability, ρ's diagonal."""
        return np.real(np.diag(self.density_matrix(endian=endian))).copy()

    def peek_observable_expectation(self, observable: Any) -> float:
        """Tr(ρ P) for a Pauli string, its sign included."""
        paulis, sign = _paulis(observable)
        if np.imag(sign) != 0:
            raise ValueError("Observable isn't Hermitian; it has imaginary sign. Need observable.sign in [1, -1].")
        if paulis and max(q for q, _ in paulis) >= self.num_qubits:
            self.set_num_qubits(max(q for q, _ in paulis) + 1)
        re, _ = self._s.expectation(paulis)
        return float(np.real(sign) * re)

    def purity(self) -> float:
        """Tr ρ²."""
        return float(self._s.purity())

    def reduced_density_matrix(self, qubits: Iterable[int], *, endian: str = "little") -> np.ndarray:
        """ρ traced down to ``qubits``, in that order."""
        return reduced_density_matrix(self.density_matrix(), list(qubits), endian=endian)

    def entropy(self, qubits: Optional[Iterable[int]] = None) -> float:
        """The von Neumann entropy (bits) of ``qubits`` (all of them by default)."""
        rho = self.density_matrix() if qubits is None else self.reduced_density_matrix(qubits)
        return von_neumann_entropy(rho)

    def fidelity(self, state: Any, *, endian: str = "little") -> float:
        """The fidelity with a state vector or density matrix."""
        return fidelity(self.density_matrix(endian=endian), state)

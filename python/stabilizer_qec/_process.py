"""Process matrices: a circuit (or any channel on up to 6 qubits) as its Choi matrix,
superoperator, Pauli transfer matrix or χ matrix, and the fidelities between processes.

Conventions:
- qubit 0 is the least significant bit of a basis index, as in ``density_matrix()``;
- the Choi matrix is J = Σ_ij |i⟩⟨j| ⊗ Λ(|i⟩⟨j|), the input factor more significant, Tr J = d;
- the superoperator S acts on ρ flattened row by row (``rho.reshape(-1)``);
- the Pauli transfer matrix is R_ab = Tr(P_a Λ(P_b)) / d;
- χ is defined by Λ(ρ) = Σ_ab χ_ab P_a ρ P_b.
Paulis are indexed by Σ_q p_q 4^q, with p_q 0, 1, 2, 3 for I, X, Y, Z on qubit q."""

from __future__ import annotations

from functools import reduce
from typing import Any

import numpy as np

_PAULIS = [np.eye(2), np.array([[0, 1], [1, 0]]), np.array([[0, -1j], [1j, 0]]), np.diag([1.0, -1.0])]

#: The most qubits a process matrix is computed for (its Choi state is a density matrix on twice
#: as many).
MAX_QUBITS = 6


def pauli_basis(n: int) -> np.ndarray:
    """The 4^n Pauli matrices on ``n`` qubits, indexed as the module describes."""
    out = []
    for k in range(4 ** n):
        digits = [(k >> (2 * q)) & 3 for q in range(n)]
        # Qubit 0 is the least significant bit: the rightmost factor of the Kronecker product.
        out.append(reduce(np.kron, [_PAULIS[digits[q]] for q in reversed(range(n))], np.eye(1)))
    return np.array(out, dtype=np.complex128)


def _dim(m: np.ndarray, power: int) -> int:
    n = 0
    while (1 << (power * n)) < m.shape[0]:
        n += 1
    if m.shape != ((1 << (power * n)), (1 << (power * n))):
        raise ValueError(f"expected a {2 ** power}^n by {2 ** power}^n matrix, got shape {m.shape}")
    return n


def choi_of_circuit(circuit: Any, *, max_branches: int = 1 << 20) -> np.ndarray:
    """The Choi matrix of a circuit run as a channel on its qubits (measurement outcomes
    averaged away; feedback followed)."""
    from ._circuit import Circuit
    from ._density import DensityMatrixSimulator

    c = circuit if isinstance(circuit, Circuit) else Circuit(str(circuit))
    n = c.num_qubits
    if n > MAX_QUBITS:
        raise ValueError(f"the circuit has {n} qubits; a process matrix is computed for at most {MAX_QUBITS}")
    d = 1 << n
    # The maximally entangled state of the system (qubits 0..n) with a reference (n..2n).
    omega = np.zeros(d * d, dtype=np.complex128)
    for i in range(d):
        omega[i | (i << n)] = 1 / np.sqrt(d)
    s = DensityMatrixSimulator(2 * n)
    s.set_density_matrix(np.outer(omega, omega.conj()))
    s.apply_channel(c, max_branches=max_branches)
    # ρ[(sys a, ref i), (sys b, ref j)] = Λ(|i⟩⟨j|)[a, b] / d, index a | i << n.
    rho = s.density_matrix().reshape(d, d, d, d)  # [ref i, sys a, ref j, sys b]
    return d * rho.reshape(d * d, d * d)


def choi_of_unitary(u: Any) -> np.ndarray:
    """The Choi matrix of ρ ↦ U ρ U†."""
    u = np.asarray(u, dtype=np.complex128)
    d = u.shape[0]
    if u.shape != (d, d):
        raise ValueError(f"a unitary is square, got shape {u.shape}")
    v = u.T.reshape(-1)  # Σ_i |i⟩ ⊗ U|i⟩, input more significant
    return np.outer(v, v.conj())


def choi_to_superop(choi: Any) -> np.ndarray:
    j = np.asarray(choi, dtype=np.complex128)
    d = 1 << _dim(j, 2)
    return j.reshape(d, d, d, d).transpose(1, 3, 0, 2).reshape(d * d, d * d)


def superop_to_choi(superop: Any) -> np.ndarray:
    s = np.asarray(superop, dtype=np.complex128)
    d = 1 << _dim(s, 2)
    return s.reshape(d, d, d, d).transpose(2, 0, 3, 1).reshape(d * d, d * d)


def choi_to_ptm(choi: Any) -> np.ndarray:
    s = choi_to_superop(choi)
    n = _dim(np.asarray(choi), 2)
    v = pauli_basis(n).reshape(4 ** n, -1).T  # columns vec(P_b)
    return np.real_if_close(v.conj().T @ s @ v / (1 << n))


def ptm_to_choi(ptm: Any) -> np.ndarray:
    r = np.asarray(ptm, dtype=np.complex128)
    n = _dim(r, 2)
    v = pauli_basis(n).reshape(4 ** n, -1).T
    s = v @ r @ v.conj().T / (1 << n)
    return superop_to_choi(s)


def choi_to_chi(choi: Any) -> np.ndarray:
    j = np.asarray(choi, dtype=np.complex128)
    n = _dim(j, 2)
    d = 1 << n
    # v_a = (I ⊗ P_a)|Ω⟩: entry (i, k) is P_a[k, i].
    v = np.array([p.T.reshape(-1) for p in pauli_basis(n)]).T
    return v.conj().T @ j @ v / d ** 2


def chi_to_choi(chi: Any) -> np.ndarray:
    c = np.asarray(chi, dtype=np.complex128)
    n = _dim(c, 2)
    v = np.array([p.T.reshape(-1) for p in pauli_basis(n)]).T
    return v @ c @ v.conj().T


_FROM_CHOI = {"choi": lambda j: j, "superoperator": choi_to_superop, "ptm": choi_to_ptm, "chi": choi_to_chi}
def _as_choi(j: Any) -> np.ndarray:
    j = np.asarray(j, dtype=np.complex128)
    _dim(j, 2)
    return j


_TO_CHOI = {"choi": _as_choi, "superoperator": superop_to_choi, "ptm": ptm_to_choi, "chi": chi_to_choi}


def process_matrix(circuit: Any, kind: str = "choi", *, max_branches: int = 1 << 20) -> np.ndarray:
    """A circuit's process as ``kind``: ``"choi"``, ``"superoperator"``, ``"ptm"`` (Pauli
    transfer matrix) or ``"chi"``."""
    if kind not in _FROM_CHOI:
        raise ValueError(f"kind must be one of {sorted(_FROM_CHOI)}, got {kind!r}")
    return _FROM_CHOI[kind](choi_of_circuit(circuit, max_branches=max_branches))


def convert_process(matrix: Any, source: str, target: str) -> np.ndarray:
    """A process matrix of kind ``source`` as kind ``target``."""
    for k in (source, target):
        if k not in _FROM_CHOI:
            raise ValueError(f"kind must be one of {sorted(_FROM_CHOI)}, got {k!r}")
    return _FROM_CHOI[target](_TO_CHOI[source](matrix))


def process_fidelity(a: Any, b: Any, *, kind: str = "choi") -> float:
    """The process fidelity of two channels given as process matrices of ``kind``: the
    fidelity of their normalized Choi states (Tr(J_a J_b)/d² when either is unitary; a unitary's
    Choi matrix is ``choi_of_unitary(U)``)."""
    from ._density import fidelity

    ja, jb = convert_process(a, kind, "choi"), convert_process(b, kind, "choi")
    if ja.shape != jb.shape:
        raise ValueError(f"the processes have shapes {ja.shape} and {jb.shape}")
    d = np.sqrt(ja.shape[0])
    return fidelity(ja / d, jb / d)


def average_gate_fidelity(a: Any, b: Any, *, kind: str = "choi") -> float:
    """(d F_pro + 1) / (d + 1), d the Hilbert space's dimension."""
    f = process_fidelity(a, b, kind=kind)
    d = np.sqrt(np.asarray(a).shape[0])
    return float((d * f + 1) / (d + 1))

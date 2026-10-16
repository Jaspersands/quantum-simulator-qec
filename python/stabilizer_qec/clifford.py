# Ported from QuantumClifford.jl (QuantumSavory/QuantumClifford.jl 0.11.8; Copyright (c) 2023
# Stefan Krastanov; MIT License, see THIRD_PARTY_NOTICES.md), modified for stabilizer_qec: its
# canonical forms, entanglement entropies, graph states and Clifford enumeration on this package's
# PauliStrings and Tableaux, qubits and indices counted from 0.
"""Stabilizer-state structure and the Clifford group, as QuantumClifford.jl gives them.

- Canonical forms of a list of stabilizer generators: the clipped gauge
  (``canonicalize_clip``, with its ``bigram``), the partial-trace form (``canonicalize_rref``)
  and Gottesman's standard form (``canonicalize_gott``).
- Bipartite entanglement entropy (``entanglement_entropy``, by the clipped gauge, the
  partial-trace form or the graph state) and ``mutual_information``, in bits.
- Graph states: any stabilizer state as a graph and the local Cliffords taking it there
  (``graph_state``, ``graph_gate_sequence``, ``graph_gate``), and ``Tableau.from_graph``.
- The Clifford group enumerated (``enumerate_cliffords``, by the symplectic Gram-Schmidt
  procedure of ``symplectic_gram_schmidt``), ``clifford_cardinality``,
  ``enumerate_single_qubit_gates`` and ``enumerate_phases``.

A state is a list of stabilizer generators (``PauliString``s or their text; fewer than the
qubits for a mixed state), a ``Tableau`` (its stabilizers, the Z images) or a
``TableauSimulator`` (its canonical stabilizers)."""

from __future__ import annotations

from typing import Any, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "bigram",
    "canonicalize_clip",
    "canonicalize_gott",
    "canonicalize_rref",
    "clifford_cardinality",
    "entanglement_entropy",
    "enumerate_cliffords",
    "enumerate_phases",
    "enumerate_single_qubit_gates",
    "graph_gate",
    "graph_gate_sequence",
    "graph_state",
    "mutual_information",
    "symplectic_gram_schmidt",
]


class _Rows:
    """Stabilizer rows as bits: row k is (−1 if neg[k]) times the product over qubits of X^x Z^z,
    an X and a Z on one qubit read as Y."""

    def __init__(self, x: np.ndarray, z: np.ndarray, neg: np.ndarray) -> None:
        self.x, self.z, self.neg = x, z, neg

    def swap(self, a: int, b: int) -> None:
        if a != b:
            for t in (self.x, self.z, self.neg):
                t[[a, b]] = t[[b, a]]

    def mul(self, m: int, i: int) -> None:
        """Row m becomes row i times row m (the rows commute)."""
        x1, z1, x2, z2 = (t.astype(np.int64) for t in (self.x[i], self.z[i], self.x[m], self.z[m]))
        # The power of i each qubit's product gives (Aaronson and Gottesman's g).
        g = np.where(x1 & z1, z2 - x2, np.where(x1 == 1, z2 * (2 * x2 - 1), np.where(z1 == 1, x2 * (1 - 2 * z2), 0)))
        e = int(g.sum()) % 4
        if e % 2:
            raise ValueError("the stabilizers do not commute")
        self.neg[m] ^= self.neg[i] ^ (e == 2)
        self.x[m] ^= self.x[i]
        self.z[m] ^= self.z[i]

    def same(self, a: int, b: int, j: int) -> bool:
        return self.x[a, j] == self.x[b, j] and self.z[a, j] == self.z[b, j]

    def nonidentity(self, k: int, j: int) -> bool:
        return bool(self.x[k, j] or self.z[k, j])

    def paulis(self) -> list:
        from ._stim import PauliString

        return [PauliString.from_numpy(xs=self.x[k], zs=self.z[k], sign=-1 if self.neg[k] else 1) for k in range(len(self.neg))]


def _paulis_of(state: Any) -> list:
    from ._stim import PauliString, Tableau, TableauSimulator

    if isinstance(state, Tableau):
        return state.to_stabilizers()
    if isinstance(state, TableauSimulator):
        return state.canonical_stabilizers()
    return [PauliString(p) for p in state]


def _rows(state: Any, num_qubits: Optional[int] = None) -> _Rows:
    paulis = _paulis_of(state)
    n = max([len(p) for p in paulis] + [num_qubits or 0])
    x = np.zeros((len(paulis), n), dtype=bool)
    z = np.zeros((len(paulis), n), dtype=bool)
    neg = np.zeros(len(paulis), dtype=bool)
    for k, p in enumerate(paulis):
        if p.sign not in (1, -1):
            raise ValueError(f"stabilizer {p} has an imaginary sign")
        xs, zs = p.to_numpy()
        x[k, : len(xs)], z[k, : len(zs)] = xs, zs
        neg[k] = p.sign == -1
    return _Rows(x, z, neg)


def _clip(r: _Rows) -> None:
    rows, cols = r.x.shape
    # Pregauge: left to right, each column's first two different non-identity rows eliminate it
    # from the rows below.
    i = 0
    for j in range(cols):
        k1 = next((k for k in range(i, rows) if r.nonidentity(k, j)), None)
        if k1 is None:
            continue
        k2 = next((k for k in range(k1 + 1, rows) if r.nonidentity(k, j) and not r.same(k, k1, j)), None)
        if k2 is not None:
            r.swap(k1, i)
            r.swap(k2, i + 1)
            for m in range(i + 2, rows):
                if r.same(m, i, j):
                    r.mul(m, i)
                elif r.same(m, i + 1, j):
                    r.mul(m, i + 1)
                elif r.x[m, j] == r.x[i, j] ^ r.x[i + 1, j] and r.z[m, j] == r.z[i, j] ^ r.z[i + 1, j]:
                    r.mul(m, i)
                    r.mul(m, i + 1)
            i += 2
        else:
            r.swap(k1, i)
            for m in range(i + 1, rows):
                if r.same(m, i, j):
                    r.mul(m, i)
            i += 1
    # Gauge: right to left, the same on the rows not yet frozen, keeping left ends.
    unfrozen = list(range(rows - 1, -1, -1))
    for j in range(cols - 1, -1, -1):
        k1 = next((a for a, k in enumerate(unfrozen) if r.nonidentity(k, j)), None)
        if k1 is None:
            continue
        r1 = unfrozen[k1]
        k2 = next((a for a in range(k1 + 1, len(unfrozen)) if r.nonidentity(unfrozen[a], j) and not r.same(unfrozen[a], r1, j)), None)
        if k2 is not None:
            r2 = unfrozen[k2]
            for m in unfrozen[k1 + 1 : k2]:
                if r.same(m, r1, j):
                    r.mul(m, r1)
            for m in unfrozen[k2 + 1 :]:
                if r.same(m, r1, j):
                    r.mul(m, r1)
                elif r.same(m, r2, j):
                    r.mul(m, r2)
                elif r.x[m, j] == r.x[r1, j] ^ r.x[r2, j] and r.z[m, j] == r.z[r1, j] ^ r.z[r2, j]:
                    r.mul(m, r1)
                    r.mul(m, r2)
            del unfrozen[k2]
            del unfrozen[k1]
        else:
            for m in unfrozen[k1 + 1 :]:
                if r.same(m, r1, j):
                    r.mul(m, r1)
            del unfrozen[k1]


def canonicalize_clip(state: Any) -> List[Any]:
    """The stabilizers in the clipped gauge (Nahum et al. 2017; Li, Chen and Fisher 2019,
    appendix A): each qubit is the left end of at most two generators and the right end of at
    most two, so the generators' spans (``bigram``) give contiguous subsystems' entropies. The
    input must be commuting Hermitian generators.

    >>> import stabilizer_qec as sq
    >>> for p in sq.canonicalize_clip(["-X_ZX_X", "+XXYZ__", "-YZ_Z_X", "-XZX__Y", "+_Z_Y_Y", "-____Z_"]):
    ...     print(p)
    -X_XY__
    +YZY___
    +_XZX__
    -_ZYX_Z
    -__YZ_X
    -____Z_
    """
    r = _rows(state)
    _clip(r)
    return r.paulis()


def _bigram(r: _Rows) -> np.ndarray:
    out = np.zeros((len(r.neg), 2), dtype=np.int64)
    support = r.x | r.z
    for k in range(len(r.neg)):
        cols = np.flatnonzero(support[k])
        if not len(cols):
            raise ValueError("a generator is the identity: the stabilizers are not independent")
        out[k] = cols[0], cols[-1]
    return out


def bigram(state: Any, *, clip: bool = True) -> np.ndarray:
    """Each generator's leftmost and rightmost qubit, one row per generator, in the clipped
    gauge (``clip=False`` when the generators already are)."""
    r = _rows(state)
    if clip:
        _clip(r)
    return _bigram(r)


def _rref(r: _Rows, columns: Iterable[int]) -> int:
    rows = len(r.neg)
    i = rows - 1
    for j in columns:
        for bits in (r.x, r.z):
            k = next((a for a in range(i + 1) if bits[a, j]), None)
            if k is not None:
                r.swap(k, i)
                for m in range(rows):
                    if bits[m, j] and m != i:
                        r.mul(m, i)
                i -= 1
    return i + 1


def canonicalize_rref(state: Any, qubits: Optional[Iterable[int]] = None) -> Tuple[List[Any], int]:
    """The stabilizers in reduced row echelon form along ``qubits`` (all by default), filled
    from the last row up (Audenaert and Plenio 2005): ``(stabilizers, k)``, the first ``k``
    of them acting trivially on ``qubits``. Tracing ``qubits`` out leaves those ``k``."""
    r = _rows(state)
    cols = range(r.x.shape[1]) if qubits is None else [int(q) for q in qubits]
    k = _rref(r, cols)
    return r.paulis(), k


def _gott_indices(bits: np.ndarray, rows: int, cols: int, skip: int = 0) -> Tuple[List[int], int]:
    good: List[int] = []
    for row in range(skip, rows):
        nz = np.flatnonzero(bits[row, skip:])
        if not len(nz):
            break
        c = skip + int(nz[0])
        if c in good:
            continue
        good.append(c)
    if good:
        bad = [c for c in range(skip, good[-1] + 1) if c not in good]
        return list(range(skip)) + good + bad + list(range(good[-1] + 1, cols)), len(good)
    return list(range(cols)), 0


def _gott(r: _Rows) -> Tuple[int, int, List[int], List[int]]:
    rows, cols = r.x.shape
    i = 0
    for j in range(cols):
        k = next((a for a in range(i, rows) if r.x[a, j]), None)
        if k is not None:
            r.swap(k, i)
            for m in range(rows):
                if r.x[m, j] and m != i:
                    r.mul(m, i)
            i += 1
    xperm, rank_x = _gott_indices(r.x, rows, cols)
    r.x, r.z = r.x[:, xperm], r.z[:, xperm]
    i = rank_x
    for j in range(rank_x, cols):
        k = next((a for a in range(i, rows) if r.z[a, j]), None)
        if k is not None:
            r.swap(k, i)
            for m in range(rows):
                if r.z[m, j] and m != i:
                    r.mul(m, i)
            i += 1
    zperm, rank_z = _gott_indices(r.z, rows, cols, skip=rank_x)
    r.x, r.z = r.x[:, zperm], r.z[:, zperm]
    return rank_x, rank_z, xperm, zperm


def canonicalize_gott(state: Any) -> Tuple[List[Any], int, int, List[int], List[int]]:
    """Gottesman's standard form (Gottesman 1997): ``(stabilizers, r, s, xperm, zperm)``, the
    first ``r`` generators with the identity on the first ``r`` columns of their X part, the
    next ``s`` with the identity on the next ``s`` columns of their Z part, after the columns
    are permuted by ``xperm`` and then ``zperm`` (column k of the result is qubit
    ``xperm[zperm[k]]``)."""
    r = _rows(state)
    rank_x, rank_z, xperm, zperm = _gott(r)
    return r.paulis(), rank_x, rank_z, xperm, zperm


def _adjacency(r: _Rows) -> Tuple[np.ndarray, List[int], List[int], List[int]]:
    rows, n = r.x.shape
    if rows != n:
        raise ValueError(f"a graph state is pure: {rows} stabilizers on {n} qubits")
    rank_x, _, xperm, zperm = _gott(r)
    perm = [xperm[k] for k in zperm]
    h_idx = [perm[i] for i in range(rank_x, n)]
    ip_idx = [perm[i] for i in range(n) if r.x[i, i] and r.z[i, i]]
    z_idx = [perm[i] for i in range(n) if r.neg[i]]
    adj = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(n):
            if i != j and (r.x[i, j] or r.z[i, j]):
                adj[perm[i], perm[j]] = adj[perm[j], perm[i]] = True
    return adj, h_idx, ip_idx, z_idx


def graph_state(state: Any) -> Tuple[np.ndarray, List[int], List[int], List[int]]:
    """A stabilizer state as a graph state: ``(adjacency, h_idx, ip_idx, z_idx)``, the graph's
    adjacency matrix and the local Cliffords taking the state to it, in order Hadamards on
    ``h_idx``, inverse phase gates (``S_DAG``, Y to X) on ``ip_idx``, then Z on ``z_idx``
    (``graph_gate_sequence``). The graph state's stabilizers are X_v times Z on v's neighbours
    (``Tableau.from_graph(adjacency).to_stabilizers()``).

    >>> import stabilizer_qec as sq
    >>> adj, h, ip, z = sq.graph_state(["+XXX", "+ZZ_", "-_ZZ"])
    >>> [(int(a), int(b)) for a, b in zip(*np.nonzero(np.triu(adj)))], h, ip, z
    ([(0, 1), (0, 2)], [1, 2], [], [2])
    """
    return _adjacency(_rows(state))


def graph_gate_sequence(h_idx: Sequence[int], ip_idx: Sequence[int], z_idx: Sequence[int]) -> Any:
    """``graph_state``'s local Cliffords as a circuit: H on ``h_idx``, then S_DAG on ``ip_idx``,
    then Z on ``z_idx``. Its inverse takes the graph state back to the state."""
    from ._circuit import Circuit

    c = Circuit()
    for name, qubits in (("H", h_idx), ("S_DAG", ip_idx), ("Z", z_idx)):
        if len(qubits):
            c.append(name, [int(q) for q in qubits])
    return c


def graph_gate(h_idx: Sequence[int], ip_idx: Sequence[int], z_idx: Sequence[int], num_qubits: int) -> Any:
    """``graph_gate_sequence`` as one Tableau on ``num_qubits`` qubits."""
    from ._stim import Tableau

    c = graph_gate_sequence(h_idx, ip_idx, z_idx)
    c.append("I", [int(num_qubits) - 1]) if num_qubits else None
    return Tableau.from_circuit(c)


def _from_graph(adjacency: Any) -> Any:
    from ._circuit import Circuit
    from ._stim import Tableau

    a = np.asarray(adjacency).astype(bool)
    n = a.shape[0]
    if a.shape != (n, n) or (a != a.T).any() or a.diagonal().any():
        raise ValueError("the adjacency matrix must be square, symmetric and without loops")
    c = Circuit()
    if n:
        c.append("H", list(range(n)))
        c.append("I", [n - 1])
    for i, j in zip(*np.nonzero(np.triu(a))):
        c.append("CZ", [int(i), int(j)])
    return Tableau.from_circuit(c)


def entanglement_entropy(state: Any, subsystem: Iterable[int], algorithm: str = "rref", *, num_qubits: Optional[int] = None) -> int:
    """The entropy of ``subsystem``'s reduced state, in bits (its qubits less its share of the
    stabilizer group). ``algorithm``: ``"rref"`` (tracing the rest out, ``canonicalize_rref``;
    any state), ``"clip"`` (the clipped gauge's ``bigram``; a contiguous subsystem) or
    ``"graph"`` (the rank over GF(2) of the graph state's adjacency between the subsystem and
    the rest; a pure state). A mixed state is fewer generators than qubits; ``num_qubits``
    widens the state past its generators' length.

    >>> import stabilizer_qec as sq
    >>> ghz = ["XXXX", "ZZ__", "_ZZ_", "__ZZ"]
    >>> [sq.entanglement_entropy(ghz, range(k), "clip") for k in range(5)]
    [0, 1, 1, 1, 0]
    """
    r = _rows(state, num_qubits)
    rows, n = r.x.shape
    sub = sorted({int(q) for q in subsystem})
    if sub and not 0 <= sub[0] <= sub[-1] < n:
        raise ValueError(f"the subsystem has qubits outside the state's {n}")
    if algorithm == "clip":
        if sub and sub != list(range(sub[0], sub[-1] + 1)):
            raise ValueError("the clip algorithm takes a contiguous subsystem")
        _clip(r)
        bg = _bigram(r)
        inside = set(sub)
        return len(sub) - int(sum(1 for a, b in bg if a in inside and b in inside))
    if algorithm == "graph":
        adj = _adjacency(r)[0]
        rest = [q for q in range(n) if q not in set(sub)]
        return _rank_gf2(adj[np.ix_(sub, rest)])
    if algorithm == "rref":
        # A pure state's entropy is its complement's: trace out the smaller side.
        if rows == n and len(sub) < n / 2:
            return n - _rref(r, sub) - len(sub)
        rest = [q for q in range(n) if q not in set(sub)]
        return n - _rref(r, rest) - len(rest)
    raise ValueError(f"algorithm must be 'rref', 'clip' or 'graph', got {algorithm!r}")


def mutual_information(state: Any, a: Iterable[int], b: Iterable[int], *, num_qubits: Optional[int] = None) -> int:
    """S(A) + S(B) − S(AB) of two disjoint subsystems, in bits."""
    a, b = sorted({int(q) for q in a}), sorted({int(q) for q in b})
    if set(a) & set(b):
        raise ValueError("the subsystems overlap")
    s = lambda q: entanglement_entropy(state, q, "rref", num_qubits=num_qubits)  # noqa: E731
    return s(a) + s(b) - s(a + b)


def _rank_gf2(m: np.ndarray) -> int:
    m = np.array(m, dtype=bool)
    rank = 0
    for c in range(m.shape[1] if m.ndim == 2 else 0):
        pivot = next((r for r in range(rank, m.shape[0]) if m[r, c]), None)
        if pivot is None:
            continue
        m[[rank, pivot]] = m[[pivot, rank]]
        for r in range(m.shape[0]):
            if r != rank and m[r, c]:
                m[r] ^= m[rank]
        rank += 1
    return rank


# The Clifford group.


class _Basis:
    """A tableau of Pauli rows over GF(2), indexed from 1 as QuantumClifford's: row k's X bits
    ``x[k]`` and Z bits ``z[k]`` over qubits 1..n."""

    def __init__(self, rows: int, n: int) -> None:
        self.x = np.zeros((rows + 1, n + 1), dtype=bool)
        self.z = np.zeros((rows + 1, n + 1), dtype=bool)

    def comm(self, a: int, b: int) -> bool:
        return bool((np.count_nonzero(self.x[a] & self.z[b]) + np.count_nonzero(self.z[a] & self.x[b])) & 1)

    def swap(self, a: int, b: int) -> None:
        self.x[[a, b]] = self.x[[b, a]]
        self.z[[a, b]] = self.z[[b, a]]

    def mul(self, m: int, i: int) -> None:
        self.x[m] ^= self.x[i]
        self.z[m] ^= self.z[i]


def _find_anticommuting(b: _Basis, start: int, padded: int, dn: int) -> int:
    while True:
        for i in range(dn + start + 1, padded + 1):
            if b.comm(dn + start, i):
                return i
        for i in range(padded + dn + start, 2 * padded + 2):
            if b.comm(dn + start, i):
                return i
        # Only when the Pauli is X on the first qubit and nothing else.
        b.swap(dn + start, 2 * padded + 1)


def _eliminate(b: _Basis, start: int, padded: int, dn: int) -> None:
    for rng in (range(dn + start + 1, padded + 1), range(padded + dn + start + 1, 2 * padded + 2)):
        for i in rng:
            x = b.comm(dn + start, i)
            z = b.comm(padded + dn + start, i)
            if z:
                b.mul(i, dn + start)
            if x:
                b.mul(i, padded + dn + start)


def _gram_schmidt(xs: np.ndarray, zs: np.ndarray, padded: int) -> _Basis:
    n = len(xs)
    b = _Basis(2 * padded + 1, padded)
    dn = padded - n
    for i in range(1, dn + 1):
        b.x[i, i] = True
        b.z[padded + i, i] = True
    for i in range(dn + 1, padded):
        b.x[i + 1, i] = True
        b.z[padded + i + 1, i] = True
    b.x[dn + 1, dn + 1 : padded + 1] = xs
    b.z[dn + 1, dn + 1 : padded + 1] = zs
    b.x[padded + dn + 1, padded] = True
    b.z[2 * padded + 1, padded] = True
    for done in range(1, n + 1):
        i = _find_anticommuting(b, done, padded, dn)
        b.swap(padded + dn + done, i)
        _eliminate(b, done, padded, dn)
    return b


def _matrix(b: _Basis, padded: int) -> np.ndarray:
    """The Clifford's rows 1..2n as one GF(2) matrix: row g the image of generator g (X_1..X_n,
    then Z_1..Z_n), its X bits then its Z bits."""
    return np.concatenate([b.x[1 : 2 * padded + 1, 1:], b.z[1 : 2 * padded + 1, 1:]], axis=1)


def _tableau(m: np.ndarray, x_signs: Any = None, z_signs: Any = None) -> Any:
    from ._stim import Tableau

    n = m.shape[0] // 2
    return Tableau.from_numpy(x2x=m[:n, :n], x2z=m[:n, n:], z2x=m[n:, :n], z2z=m[n:, n:], x_signs=x_signs, z_signs=z_signs)


def symplectic_gram_schmidt(pauli: Any, *, padded_n: Optional[int] = None) -> Any:
    """The Clifford the symplectic Gram-Schmidt procedure makes from ``pauli`` (Koenig and
    Smolin 2014), on ``padded_n`` qubits (the Pauli's own by default, the Pauli on the last
    ``n``): its first X image (of the unpadded qubits) is the Pauli. Signs are +.

    >>> import stabilizer_qec as sq
    >>> sq.symplectic_gram_schmidt(sq.PauliString("Z"))
    stabilizer_qec.Tableau.from_conjugated_generators(
        xs=[
            stabilizer_qec.PauliString("+Z"),
        ],
        zs=[
            stabilizer_qec.PauliString("+X"),
        ],
    )
    """
    from ._stim import PauliString

    xs, zs = PauliString(pauli).to_numpy()
    padded = len(xs) if padded_n is None else int(padded_n)
    if padded < len(xs):
        raise ValueError(f"padded_n ({padded}) is less than the Pauli's {len(xs)} qubits")
    if not (xs.any() or zs.any()):
        raise ValueError("the Pauli is the identity")
    return _tableau(_matrix(_gram_schmidt(xs, zs, padded), padded))


def clifford_cardinality(n: int, *, phases: bool = True) -> int:
    """The size of the n-qubit Clifford group modulo global phase: 2^(n²) Π_j (4^j − 1)
    symplectic tableaux, times 4^n sign choices (without them with ``phases=False``).

    >>> import stabilizer_qec as sq
    >>> sq.clifford_cardinality(7)
    457620995529680351512370381586432000
    >>> sq.clifford_cardinality(7, phases=False)
    27930968965434591767112450048000
    """
    base = 2 ** (n * n)
    for j in range(1, n + 1):
        base *= 4**j - 1
    return base * 4**n if phases else base


def _enumerate(n: int, i: int, padded: int, only_coset: bool) -> np.ndarray:
    # i counts from 1, as QuantumClifford's.
    s = 2 ** (2 * n) - 1
    k = (i - 1) % s + 1
    bits = [bool(k >> d & 1) for d in range(2 * n)]
    b = _gram_schmidt(np.array(bits[:n]), np.array(bits[n:]), padded)
    dn = padded - n
    idivs = (i - 1) // s
    for j in range(n):  # w1 += v_j
        if idivs >> j & 1:
            b.mul(padded + dn + 1, dn + j + 1)
    for j in range(n, 2 * n - 1):  # w1 += w_j
        if idivs >> j & 1:
            b.mul(padded + dn + 1, 2 * dn + j + 2)
    for j in range(1, n):  # w_j += v1
        if idivs >> j & 1:
            b.mul(padded + dn + j + 1, dn + 1)
    for j in range(n, 2 * n - 1):  # v_j += v1
        if idivs >> j & 1:
            b.mul(dn + j + 2 - n, dn + 1)
    op = _matrix(b, padded)
    if n == 1 or only_coset:
        return op
    sub = _enumerate(n - 1, idivs // 2 ** (2 * n - 1) + 1, padded, False)
    # The sub-Clifford first, then this coset's.
    return (sub.astype(np.int64) @ op.astype(np.int64) % 2).astype(bool)


def enumerate_cliffords(n: int, index: Optional[int] = None, *, padded_n: Optional[int] = None, only_coset: bool = False) -> Any:
    """The ``index``-th n-qubit Clifford tableau, signs all +, in QuantumClifford's order (Koenig
    and Smolin 2014's symplectic indexing; ``index`` counts from 0, QuantumClifford's i − 1), or
    without ``index`` an iterator over all ``clifford_cardinality(n, phases=False)`` of them.
    With ``only_coset``, the coset representatives (the symplectic Gram-Schmidt part alone,
    (4^n − 1)·2^(2n−1) of them); with ``padded_n``, embedded in that many qubits.

    >>> import stabilizer_qec as sq
    >>> t = sq.enumerate_cliffords(3, 0)
    >>> [str(t.x_output(q)) for q in range(3)], [str(t.z_output(q)) for q in range(3)]
    (['+X__', '+__Z', '+_Z_'], ['+Z__', '+__X', '+_X_'])
    """
    padded = n if padded_n is None else int(padded_n)
    if n < 1 or padded < n:
        raise ValueError(f"n must be positive and padded_n at least n, got n={n}, padded_n={padded_n}")
    size = (4**n - 1) * 2 ** (2 * n - 1) if only_coset else clifford_cardinality(n, phases=False)
    if index is None:
        return (_tableau(_enumerate(n, i + 1, padded, only_coset)) for i in range(size))
    if not 0 <= index < size:
        raise IndexError(f"index {index} is not below the {size} Cliffords")
    return _tableau(_enumerate(n, int(index) + 1, padded, only_coset))


_SINGLE_QUBIT_PATTERNS = (
    (True, False, False, True),  # X, Z -> X, Z
    (False, True, True, True),  # X, Z -> Z, Y
    (True, True, True, False),  # X, Z -> Y, X
    (False, True, True, False),  # X, Z -> Z, X: Hadamard
    (True, False, True, True),  # X, Z -> X, Y
    (True, True, False, True),  # X, Z -> Y, Z: phase
)


def enumerate_single_qubit_gates(index: int, *, phases: Tuple[bool, bool] = (False, False)) -> Any:
    """One of the 6 single-qubit Cliffords up to signs (``index`` 0 to 5, QuantumClifford's
    index − 1: identity, X→Z Z→Y, X→Y Z→X, Hadamard, X→X Z→Y, phase), with X's and Z's images
    negated as ``phases`` says.

    >>> import stabilizer_qec as sq
    >>> sq.enumerate_single_qubit_gates(5) == sq.Tableau.from_named_gate("S")
    True
    """
    if not 0 <= index < 6:
        raise IndexError("there are 6 single-qubit Cliffords up to signs (index 0 to 5)")
    xx, xz, zx, zz = _SINGLE_QUBIT_PATTERNS[index]
    m = np.array([[xx, xz], [zx, zz]])
    return _tableau(m, x_signs=np.array([phases[0]]), z_signs=np.array([phases[1]]))


def enumerate_phases(op: Any) -> Iterator[Any]:
    """Every tableau with ``op``'s images and any signs (4^n of them; in QuantumClifford's order,
    bit k of the count the sign of X_k's image for k < n, of Z_(k−n)'s after). An iterable of
    tableaux gives each one's in turn.

    >>> import stabilizer_qec as sq
    >>> len(list(sq.enumerate_phases(sq.Tableau.from_named_gate("CNOT"))))
    16
    """
    from ._stim import Tableau

    if not isinstance(op, Tableau):
        return (t for o in op for t in enumerate_phases(o))
    x2x, x2z, z2x, z2z, _, _ = op.to_numpy()
    n = len(op)

    def gen() -> Iterator[Any]:
        for i in range(4**n):
            bits = np.array([bool(i >> d & 1) for d in range(2 * n)])
            yield Tableau.from_numpy(x2x=x2x, x2z=x2z, z2x=z2x, z2z=z2z, x_signs=bits[:n], z_signs=bits[n:])

    return gen()

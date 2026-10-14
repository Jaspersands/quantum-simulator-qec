# Ported from ldpc 2.4.1 (Copyright (c) 2022 Joschka Roffe, MIT License; see THIRD_PARTY_NOTICES.md):
# ldpc/mod2/_mod2.pyx and the C++ it calls (src_cpp/gf2dense.hpp, src_cpp/gf2sparse_linalg.hpp),
# reimplemented in numpy so that every function returns what ldpc's returns, entry for entry.
"""GF(2) linear algebra, as ``ldpc.mod2``.

ldpc has two eliminations, and its answers (which basis of a kernel, which rows pivot) depend on
which one a function uses:

- ``gf2dense``'s PLU decomposition: columns in order, each pivot the first row at or below the
  rank holding a 1, elimination below the pivot only. Eliminating column by column with the
  earlier operations replayed, as ldpc does, is the same as eliminating the whole matrix at once,
  which is what is done here, on rows packed 64 columns to a word.
- ``gf2sparse_linalg``'s ``RowReduce``: the pivot is the row of least weight (its U and L rows
  together) among those holding a 1, and L either accumulates the row operations or records
  the lower-triangular factor.
"""

from __future__ import annotations

import itertools
import time
from typing import List, Tuple, Union

import numpy as np
import scipy.sparse

from stabilizer_qec.ldpc.helpers import scipy_helpers
from stabilizer_qec.ldpc.mod2 import _legacy_v1

__all__ = [
    "csc_to_scipy_sparse",
    "rank",
    "nullspace",
    "kernel",
    "row_complement_basis",
    "pivot_rows",
    "io_test",
    "estimate_code_distance",
    "row_span",
    "compute_exact_code_distance",
    "row_basis",
    "row_echelon",
    "reduced_row_echelon",
    "inverse",
    "PluDecomposition",
]

_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.int64)


# -- conversions, as ldpc's -------------------------------------------------------------------


def _binary_csr(pcm) -> scipy.sparse.csr_matrix:
    return scipy.sparse.csr_matrix(scipy_helpers.convert_to_binary_sparse(pcm))


def _csr_lists(pcm) -> List[List[int]]:
    """Each row's columns, ascending (ldpc's ``Py2CsrList``)."""
    csr = _binary_csr(pcm)
    csr.sort_indices()
    return [csr.indices[csr.indptr[i] : csr.indptr[i + 1]].tolist() for i in range(csr.shape[0])]


def _csc_lists(pcm) -> List[List[int]]:
    """Each column's rows, ascending (ldpc's ``Py2CscList``)."""
    csc = scipy.sparse.csc_matrix(scipy_helpers.convert_to_binary_sparse(pcm))
    csc.sort_indices()
    return [csc.indices[csc.indptr[j] : csc.indptr[j + 1]].tolist() for j in range(csc.shape[1])]


def _csr_from_rows(rows: List[List[int]], m: int, n: int) -> scipy.sparse.csr_matrix:
    """ldpc's ``csr_to_scipy_sparse``: rows past the list's end (where ldpc reads out of
    bounds) are empty."""
    r = [i for i in range(m) if i < len(rows) for _ in rows[i]]
    c = [j for i in range(min(m, len(rows))) for j in rows[i]]
    return scipy.sparse.csr_matrix((np.ones(len(r), dtype=np.uint8), (np.array(r, dtype=np.int32), np.array(c, dtype=np.int32))), shape=(m, n), dtype=np.uint8)


def csc_to_scipy_sparse(col_adjacency_list) -> scipy.sparse.csr_matrix:
    """A matrix given column by column (each column's rows) as a csr matrix; its row count is
    one past the largest row given, as in ldpc."""
    n = len(col_adjacency_list)
    m = max((r + 1 for col in col_adjacency_list for r in col), default=0)
    r = [i for col in col_adjacency_list for i in col]
    c = [j for j, col in enumerate(col_adjacency_list) for _ in col]
    return scipy.sparse.csr_matrix((np.ones(len(r), dtype=np.uint8), (np.array(r, dtype=np.int32), np.array(c, dtype=np.int32))), shape=(m, n), dtype=np.uint8)


def _check_type(pcm) -> None:
    if not isinstance(pcm, (np.ndarray, scipy.sparse.spmatrix)):
        raise TypeError(f"The input matrix is of an invalid type. Please input a np.ndarray or scipy.sparse.spmatrix object, not {type(pcm)}")


# -- gf2dense: the PLU decomposition, column by column ----------------------------------------


class _Packed:
    """A binary matrix, each row packed into 64-bit words."""

    def __init__(self, rows: int, cols: int) -> None:
        self.rows, self.cols = rows, cols
        self.words = max(1, (cols + 63) // 64)
        self.a = np.zeros((rows, self.words), dtype=np.uint64)

    @classmethod
    def from_columns(cls, rows: int, cols: int, columns: List[List[int]]) -> "_Packed":
        p = cls(rows, cols)
        for j, col in enumerate(columns):
            if col:
                p.a[np.asarray(col, dtype=np.int64), j >> 6] |= np.uint64(1) << np.uint64(j & 63)
        return p

    @classmethod
    def identity(cls, n: int) -> "_Packed":
        p = cls(n, n)
        idx = np.arange(n)
        p.a[idx, idx >> 6] = np.uint64(1) << (idx & 63).astype(np.uint64)
        return p

    def column(self, j: int) -> np.ndarray:
        return ((self.a[:, j >> 6] >> np.uint64(j & 63)) & np.uint64(1)).astype(bool)

    def swap(self, i: int, k: int) -> None:
        if i != k:
            self.a[[i, k]] = self.a[[k, i]]

    def row_list(self, i: int) -> List[int]:
        bits = np.unpackbits(self.a[i].view(np.uint8), bitorder="little")[: self.cols]
        return np.flatnonzero(bits).tolist()

    def weights(self, rows: np.ndarray) -> np.ndarray:
        return _POPCOUNT[self.a[rows].view(np.uint8)].reshape(len(rows), -1).sum(axis=1)


def _plu(rows: int, cols: int, columns: List[List[int]], transform: bool = False):
    """gf2dense's ``PluDecomposition::rref(false, false)``: (rank, pivot columns, the row
    operations' transform E if asked for), E applied to the identity as the rows were."""
    m = _Packed.from_columns(rows, cols, columns)
    e = _Packed.identity(rows) if transform else None
    rank, pivots = 0, []
    max_rank = min(rows, cols)
    for j in range(cols):
        if rank == max_rank:
            break
        col = m.column(j)
        below = np.flatnonzero(col[rank:])
        if len(below) == 0:
            continue
        p = rank + int(below[0])
        m.swap(rank, p)
        col[[rank, p]] = col[[p, rank]]
        if e is not None:
            e.swap(rank, p)
        targets = rank + 1 + np.flatnonzero(col[rank + 1 :])
        if len(targets):
            m.a[targets] ^= m.a[rank]
            if e is not None:
                e.a[targets] ^= e.a[rank]
        pivots.append(j)
        rank += 1
    return rank, pivots, e


def _dense_kernel(row_count: int, col_count: int, csr: List[List[int]]) -> List[List[int]]:
    """gf2dense's ``kernel``, as columns: the PLU of the transpose, with each unit vector put
    through its operations; column j holds the rows past the rank where it ends up 1."""
    rank, _, e = _plu(col_count, row_count, csr, transform=True)
    ker = [[] for _ in range(col_count)]
    for i in range(rank, col_count):
        for j in e.row_list(i):
            ker[j].append(i - rank)
    return ker


# -- gf2sparse_linalg: RowReduce ----------------------------------------------------------------


class _RowReduce:
    """ldpc's ``RowReduce`` on a dense copy of the matrix: U, L, row and column orders, pivots.

    ldpc's sparse matrix keeps each column as a linked list, and ``swap_rows`` relabels the rows
    without moving them in those lists, so after a swap a column no longer lists its rows in
    order. Pivots are chosen by walking the pivot's column (the first row of least weight wins),
    so U's column lists are kept here in ldpc's link order, with its insertion rule (after the
    last entry, from the end, labelled below the new row)."""

    def __init__(self, a: np.ndarray) -> None:
        self.A = (np.asarray(a) != 0)
        self.m, self.n = self.A.shape
        self.pivots = np.zeros(self.n, dtype=bool)
        self.cols = list(range(self.n))
        self.rows = list(range(self.m))
        self.rank = 0
        self.lu_allocated = False
        self.lower_triangular = False
        self.full_reduce = False
        self.x = np.zeros(self.n, dtype=np.uint8)
        self.U = np.zeros((self.m, self.n), dtype=bool)
        self.L = np.zeros((self.m, self.m), dtype=bool)
        self.links: List[List[int]] = []
        self.P = None

    def _swap(self, i: int, j: int) -> None:
        for c in np.flatnonzero(self.U[i] | self.U[j]):
            col = self.links[c]
            for k, r in enumerate(col):
                if r == i:
                    col[k] = j
                elif r == j:
                    col[k] = i
        self.U[[i, j]] = self.U[[j, i]]
        self.L[[i, j]] = self.L[[j, i]]

    def _insert(self, r: int, c: int) -> None:
        col = self.links[c]
        pos = 0
        for k in range(len(col) - 1, -1, -1):
            if col[k] < r:
                pos = k + 1
                break
        col.insert(pos, r)
        self.U[r, c] = True

    def _add_rows(self, i: int, j: int) -> None:
        """Row i += row j (GF2Sparse::add_rows): new entries inserted column by column, the
        cancelled ones removed afterwards."""
        cancelled = []
        for c in np.flatnonzero(self.U[j]):
            if self.U[i, c]:
                cancelled.append(c)
            else:
                self._insert(i, c)
        for c in cancelled:
            self.links[c].remove(i)
            self.U[i, c] = False

    def rref(self, full_reduce: bool = False, lower_triangular: bool = False) -> int:
        self.lower_triangular, self.full_reduce = bool(lower_triangular), bool(full_reduce)
        self.cols = list(range(self.n))
        self.rows = list(range(self.m))
        self.U = self.A.copy()
        self.links = [np.flatnonzero(self.U[:, c]).tolist() for c in range(self.n)]
        self.L = np.zeros((self.m, self.m), dtype=bool) if self.lower_triangular else np.eye(self.m, dtype=bool)
        self.lu_allocated = True
        max_rank = min(self.m, self.n)
        self.rank = 0
        self.pivots[:] = False
        U, L = self.U, self.L
        for j in range(self.n):
            pivot = self.cols[j]
            if self.rank == max_rank:
                break
            r = self.rank
            swap, best = None, None
            for row in self.links[pivot]:
                if row < r:
                    continue
                weight = int(U[row].sum()) + int(L[row].sum())
                if best is None or weight < best:
                    swap, best = row, weight
            if swap is None:
                continue
            self.pivots[j] = True
            if swap != r:
                self._swap(swap, r)
                self.rows[swap], self.rows[r] = self.rows[r], self.rows[swap]
            if self.lower_triangular:
                L[r, r] = True
            add = [row for row in self.links[pivot] if row > r or (row < r and self.full_reduce)]
            for row in add:
                self._add_rows(row, r)
                if self.lower_triangular:
                    L[row, r] = True
                else:
                    L[row] ^= L[r]
            self.rank += 1
        order = [c for i, c in enumerate(self.cols) if self.pivots[i]] + [c for i, c in enumerate(self.cols) if not self.pivots[i]]
        self.cols = order
        return self.rank

    def lu_solve(self, y) -> np.ndarray:
        if len(y) != self.m:
            raise ValueError("Input parameter `y` is of the incorrect length for lu_solve.")
        if not self.lu_allocated or not self.lower_triangular:
            self.rref(False, True)
        self.x[:] = 0
        b = np.zeros(self.m, dtype=np.uint8)
        for r in range(self.rank):
            b[r] = (int(np.bitwise_xor.reduce(b[np.flatnonzero(self.L[r])], initial=0)) ^ int(y[self.rows[r]])) & 0xFF
        for r in range(self.rank - 1, -1, -1):
            s = int(np.bitwise_xor.reduce(self.x[np.flatnonzero(self.U[r])], initial=0))
            self.x[self.cols[r]] = (s ^ int(b[r])) & 0xFF
        return self.x.copy()

    def build_p_matrix(self) -> None:
        if not self.lu_allocated or not self.lower_triangular:
            self.rref(False, True)
        self.P = np.zeros((self.m, self.m), dtype=bool)
        for i in range(self.m):
            self.P[self.rows[i], i] = True


def _dense(pcm) -> np.ndarray:
    return _binary_csr(pcm).toarray().astype(bool)


def _csr_of(dense: np.ndarray) -> scipy.sparse.csr_matrix:
    m, n = dense.shape
    r, c = np.nonzero(dense)
    return scipy.sparse.csr_matrix((np.ones(len(r), dtype=np.uint8), (r.astype(np.int32), c.astype(np.int32))), shape=(m, n), dtype=np.uint8)


# -- ldpc.mod2's functions ----------------------------------------------------------------------


def rank(pcm: Union[scipy.sparse.spmatrix, np.ndarray], method: str = "dense") -> int:
    """The rank of a binary matrix over GF(2) ("dense": gf2dense's PLU; "sparse": RowReduce)."""
    if method == "dense":
        return _plu(pcm.shape[0], pcm.shape[1], _csc_lists(pcm))[0]
    elif method == "sparse":
        return _RowReduce(_dense(pcm)).rref(False, False)
    else:
        raise ValueError(f"Invalid method. Please use 'dense' or 'sparse', not {method}")


def nullspace(pcm: Union[scipy.sparse.spmatrix, np.ndarray], method="dense") -> scipy.sparse.spmatrix:
    """A basis of the kernel of a binary matrix, as the rows of a csr matrix: ldpc's basis
    for the method ("dense": gf2dense; "sparse": RowReduce)."""
    _check_type(pcm)
    row_count, col_count = pcm.shape[0], pcm.shape[1]
    if method == "dense":
        return csc_to_scipy_sparse(_dense_kernel(row_count, col_count, _csr_lists(pcm)))
    elif method == "sparse":
        rr = _RowReduce(_dense(pcm).T)
        rr.rref(False, False)
        return _csr_of(rr.L[rr.rank :, :])
    else:
        raise ValueError("Invalid method. Please use 'dense' or 'sparse'")


def kernel(pcm: Union[scipy.sparse.spmatrix, np.ndarray], method="dense") -> scipy.sparse.spmatrix:
    """The kernel of a binary matrix (the same as ``nullspace``)."""
    return nullspace(pcm, method)


def row_complement_basis(pcm: Union[scipy.sparse.spmatrix, np.ndarray]) -> scipy.sparse.spmatrix:
    """Unit rows that complete the row space of a matrix to the whole space, as ldpc picks them."""
    a = _dense(pcm)
    m, n = a.shape
    aug = np.hstack([a.T, np.eye(n, dtype=bool)])
    rr = _RowReduce(aug)
    rr.rref(False, False)
    basis_rows = [rr.cols[i] - m for i in range(rr.rank) if rr.cols[i] >= m]
    out = np.zeros((len(basis_rows), n), dtype=bool)
    out[np.arange(len(basis_rows)), basis_rows] = True
    return _csr_of(out)


def pivot_rows(mat: Union[np.ndarray, scipy.sparse.spmatrix]) -> np.ndarray:
    """The rows of a matrix that are independent of the rows before them."""
    return np.array(_plu(mat.shape[1], mat.shape[0], _csr_lists(mat))[1], dtype=int)


def io_test(pcm: Union[scipy.sparse.spmatrix, np.ndarray]) -> scipy.sparse.spmatrix:
    """The matrix back, through ldpc's conversion (a test function)."""
    return _csr_of(_dense(pcm))


def _min_weight_words(rows: List[List[int]], col_count: int, timeout_seconds: float, number_of_words_to_save: int):
    """gf2dense's ``estimate_minimum_linear_row_combination``: random sums of rows (each row in
    with probability 2/rows) until the timeout, keeping the lightest."""
    saved: List[List[int]] = [[] for _ in range(number_of_words_to_save)]
    max_saved = 2**31 - 1

    def save(word: List[int]) -> None:
        nonlocal max_saved
        max1 = max2 = -10
        replace = 0
        for k, w in enumerate(saved):
            size = len(w)
            if size == 0:
                replace = k
                break
            if size > max1:
                max1, replace = size, k
            elif size > max2:
                max2 = size
        saved[replace] = word
        max_saved = len(word) if len(word) > max2 else max2

    for word in rows:
        if len(word) < max_saved:
            save(list(word))
    row_count = len(rows)
    min_distance = 2**31 - 1
    rng = np.random.default_rng()
    dense = np.zeros((row_count, col_count), dtype=np.uint8)
    for i, word in enumerate(rows):
        dense[i, word] = 1
    sample_prob = 2.0 / row_count if row_count else np.inf
    start = time.monotonic()
    count = 0
    while True:
        count += 1
        pick = rng.random(row_count) < sample_prob
        current = np.flatnonzero(np.bitwise_xor.reduce(dense[pick], axis=0)) if pick.any() else np.zeros(0, dtype=int)
        size = len(current)
        if size:
            min_distance = min(min_distance, size)
            if size <= max_saved:
                save(current.tolist())
            if time.monotonic() - start >= timeout_seconds:
                break
        elif row_count == 0 or time.monotonic() - start >= max(timeout_seconds, 0) + 1.0:
            break  # (ldpc loops forever on a code with no codewords)
    return min_distance, count, saved


def estimate_code_distance(pcm: Union[scipy.sparse.spmatrix, np.ndarray], timeout_seconds: float = 0.025, number_of_words_to_save=10):
    """An upper bound on a code's distance from random combinations of its kernel's basis,
    searched until ``timeout_seconds``: (distance, samples searched, the lightest codewords
    found, as rows)."""
    row_count, col_count = pcm.shape[0], pcm.shape[1]
    ker = _dense_kernel(row_count, col_count, _csr_lists(pcm))
    k = max((r + 1 for col in ker for r in col), default=0)
    rows: List[List[int]] = [[] for _ in range(k)]
    for j, col in enumerate(ker):
        for r in col:
            rows[r].append(j)
    min_distance, searched, words = _min_weight_words(rows, col_count, timeout_seconds, 10)
    return min_distance, searched, _csr_from_rows(words, number_of_words_to_save, col_count)


def row_span(pcm: Union[scipy.sparse.spmatrix, np.ndarray]) -> scipy.sparse.spmatrix:
    """Every sum of the rows of a matrix: row i of the result sums the rows set in i's binary
    digits, least significant first."""
    row_count, col_count = pcm.shape[0], pcm.shape[1]
    dense = _dense(pcm).astype(np.uint8)
    out = np.zeros((2**row_count, col_count), dtype=np.uint8)
    for i in range(1, 2**row_count):
        low = i & -i
        out[i] = out[i ^ low] ^ dense[low.bit_length() - 1]
    return _csr_of(out.astype(bool))


def compute_exact_code_distance(pcm: Union[scipy.sparse.spmatrix, np.ndarray]) -> int:
    """A code's distance, by every combination of its kernel's basis (-1 when it has no
    codewords)."""
    row_count, col_count = pcm.shape[0], pcm.shape[1]
    ker = _dense_kernel(row_count, col_count, _csr_lists(pcm))
    k = max((r + 1 for col in ker for r in col), default=0)
    if k == 0:
        return -1
    basis = np.zeros((k, col_count), dtype=np.uint8)
    for j, col in enumerate(ker):
        basis[col, j] = 1
    distance = col_count
    # Gray code: one row added or removed per step.
    word = np.zeros(col_count, dtype=np.uint8)
    for i in range(1, 2**k):
        word ^= basis[(i & -i).bit_length() - 1]
        w = int(word.sum())
        if w < distance:
            distance = w
    return distance


def row_basis(pcm: Union[scipy.sparse.spmatrix, np.ndarray]) -> scipy.sparse.spmatrix:
    """The rows of a matrix independent of the rows before them, as a csr matrix."""
    pcm = scipy_helpers.convert_to_binary_sparse(pcm)
    pivots = pivot_rows(pcm)
    return pcm[pivots, :]


def row_echelon(matrix: Union[np.ndarray, scipy.sparse.spmatrix], full: bool = False):
    """Row echelon form by Gaussian elimination: (form, rank, transform, pivot columns); with
    ``full``, reduced above the pivots too."""
    if isinstance(matrix, scipy.sparse.spmatrix):
        matrix = matrix.toarray()
    elif not isinstance(matrix, np.ndarray):
        raise TypeError(f"The input matrix is of an invalid type. Please input a np.ndarray or scipy.sparse.spmatrix object, not {type(matrix)}")
    return _legacy_v1.row_echelon(matrix, full)


def reduced_row_echelon(matrix: Union[np.ndarray, scipy.sparse.spmatrix]):
    """Reduced row echelon form [I, A]: (form, rank, row transform, column transform)."""
    if isinstance(matrix, scipy.sparse.spmatrix):
        matrix = matrix.toarray()
    elif not isinstance(matrix, np.ndarray):
        raise TypeError(f"The input matrix is of an invalid type. Please input a np.ndarray or scipy.sparse.spmatrix object, not {type(matrix)}")
    return _legacy_v1.reduced_row_echelon(matrix)


def inverse(matrix: Union[np.ndarray, scipy.sparse.spmatrix]) -> np.ndarray:
    """The (left) inverse of a full-column-rank binary matrix."""
    if isinstance(matrix, scipy.sparse.spmatrix):
        matrix = matrix.toarray()
    elif not isinstance(matrix, np.ndarray):
        raise TypeError(f"The input matrix is of an invalid type. Please input a np.ndarray or scipy.sparse.spmatrix object, not {type(matrix)}")
    return _legacy_v1.inverse(matrix)


class PluDecomposition:
    """ldpc's ``PluDecomposition``: RowReduce's factorisation of a matrix, with ``L``, ``U``,
    ``P``, ``rank``, ``pivots`` and ``lu_solve``. As in ldpc, ``lu_solve`` refactorises the
    matrix (lower triangular, not fully reduced) first, and ``L``, ``U`` and ``P`` are what they
    were when first read."""

    def __init__(self, pcm: Union[scipy.sparse.spmatrix, np.ndarray], full_reduce: bool = False, lower_triangular: bool = True) -> None:
        self.rr = _RowReduce(_dense(pcm))
        self.full_reduce = full_reduce
        self.lower_triangular = full_reduce  # (sic, as ldpc)
        self._L = self._U = self._P = None
        self.rr.rref(full_reduce, lower_triangular)

    def lu_solve(self, y: np.ndarray) -> np.ndarray:
        if self.full_reduce is True or self.lower_triangular is False or self.full_reduce == True or self.lower_triangular == False:  # noqa: E712
            self.rr.rref(False, True)
        return self.rr.lu_solve(np.asarray(y)).astype(np.uint8)

    @property
    def L(self) -> scipy.sparse.spmatrix:
        if self._L is None:
            self._L = _csr_of(self.rr.L)
        return self._L

    @property
    def U(self) -> scipy.sparse.spmatrix:
        if self._U is None:
            self._U = _csr_of(self.rr.U)
        return self._U

    @property
    def P(self) -> scipy.sparse.spmatrix:
        if self._P is None:
            self.rr.build_p_matrix()
            self._P = _csr_of(self.rr.P)
        return self._P

    @property
    def rank(self) -> int:
        return self.rr.rank

    @property
    def pivots(self) -> np.ndarray:
        return np.flatnonzero(self.rr.pivots).astype(np.int32)

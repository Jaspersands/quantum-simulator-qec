# Ported from CUDA-Q QEC (NVIDIA/cudaqx, libs/qec: include/cudaq/qec/code_matrices.h,
# sparse_binary_matrix.h, dem_construction_code.h, lib/dem_construction.cpp,
# dem_construction_utils.cpp, dem_construction_code.cpp, sparse_binary_matrix.cpp and
# python/bindings/py_dem_construction.cpp, sparse_matrix_casters.h; Copyright (c) 2026 NVIDIA
# Corporation & Affiliates, Apache License 2.0; see THIRD_PARTY_NOTICES.md), modified for
# stabilizer_qec: numpy only, the sparse matrix kept as the column lists it was given.
"""CUDA-Q QEC's CSS code matrices and the detector error model built from them.

``CssCodes`` holds a CSS code's parity checks and logical operators (``hz``, ``hx``, ``lz`` and
``lx``, each a dense ``uint8`` array or any scipy sparse matrix when assigned, a dense copy when
read), ``CssNoise`` a phenomenological noise model (the rates of X, Y and Z errors on the data
qubits and of measurement errors on the checks, uniform or per element), and
``dem_from_css_matrices`` the detector error model of ``num_rounds`` rounds of them: one column
per fault of each round, a round's data faults flipping the detectors of that round and the next
(the last round's only its own). ``css_matrices_from_code`` reads the matrices off a ``Code``.

``_Mat`` is CUDA-Q's ``sparse_binary_matrix``: the index lists as given (not sorted or
GF(2)-collapsed), its dimensions kept even when empty, so a default matrix is 0 x 0."""

from __future__ import annotations

from typing import Any, List, Optional, Sequence

import numpy as np

_U32 = 2**32 - 1


# -- The sparse binary matrix -------------------------------------------------------------------------


class _Mat:
    """A sparse binary matrix as nested column lists (``sparse_binary_matrix``'s CSC form). Never
    changed once built, so matrices are shared freely."""

    __slots__ = ("nrows", "ncols", "cols")

    def __init__(self, nrows: int = 0, ncols: int = 0, cols: Optional[List[List[int]]] = None) -> None:
        self.nrows, self.ncols = nrows, ncols
        self.cols: List[List[int]] = [[] for _ in range(ncols)] if cols is None else cols

    @staticmethod
    def from_cols(nrows: int, ncols: int, cols: Sequence[Sequence[int]]) -> "_Mat":
        """``from_nested_csc``: ``cols[j]`` the row indices of column ``j``."""
        if len(cols) != ncols:
            raise ValueError("sparse_binary_matrix::from_nested_csc: nested.size() must equal num_cols")
        for col in cols:
            for r in col:
                if r >= nrows:
                    raise ValueError("sparse_binary_matrix::from_nested_csc: row index out of range")
        return _Mat(nrows, ncols, [list(col) for col in cols])

    @staticmethod
    def from_rows(nrows: int, ncols: int, rows: Sequence[Sequence[int]]) -> "_Mat":
        """``from_nested_csr`` (read as CSC, rows in order): ``rows[i]`` the column indices of row ``i``."""
        if len(rows) != nrows:
            raise ValueError("sparse_binary_matrix::from_nested_csr: nested.size() must equal num_rows")
        for row in rows:
            for c in row:
                if c >= ncols:
                    raise ValueError("sparse_binary_matrix::from_nested_csr: column index out of range")
        return _Mat._rows(nrows, ncols, rows)

    @staticmethod
    def _rows(nrows: int, ncols: int, rows: Sequence[Sequence[int]]) -> "_Mat":
        cols: List[List[int]] = [[] for _ in range(ncols)]
        for i, row in enumerate(rows):
            for c in row:
                cols[c].append(i)
        return _Mat(nrows, ncols, cols)

    @staticmethod
    def from_dense(a: np.ndarray) -> "_Mat":
        """Any non-zero entry of a 2-D array as a 1."""
        if a.ndim != 2:
            raise ValueError("sparse_binary_matrix: dense PCM tensor must have rank 2")
        nrows, ncols = a.shape
        if nrows > _U32 or ncols > _U32:
            raise ValueError("sparse_binary_matrix: dense PCM dimensions exceed index_type range")
        cols: List[List[int]] = [[] for _ in range(ncols)]
        rr, cc = np.nonzero(a)
        for r, c in zip(rr.tolist(), cc.tolist()):
            cols[c].append(r)
        return _Mat(nrows, ncols, cols)

    def rows(self) -> List[List[int]]:
        """``to_nested_csr`` of the CSC matrix: each row's columns, ascending."""
        out: List[List[int]] = [[] for _ in range(self.nrows)]
        for j, col in enumerate(self.cols):
            for i in col:
                out[i].append(j)
        return out

    def canonical_cols(self) -> List[List[int]]:
        """Each column sorted with equal indices XOR-merged (kept iff they occur an odd number of times)."""
        return [_canonical(col) for col in self.cols]

    def to_dense(self) -> np.ndarray:
        """The matrix with a 1 at each listed entry (a repeated entry is still 1)."""
        a = np.zeros((self.nrows, self.ncols), dtype=np.uint8)
        for j, col in enumerate(self.cols):
            if col:
                a[col, j] = 1
        return a


def _canonical(col: List[int]) -> List[int]:
    if len(col) < 2:
        return list(col)
    s = sorted(col)
    if len(set(s)) == len(s):
        return s
    out: List[int] = []
    i = 0
    while i < len(s):
        j = i
        while j < len(s) and s[j] == s[i]:
            j += 1
        if (j - i) & 1:
            out.append(s[i])
        i = j
    return out


def _from_python(value: Any) -> _Mat:
    """``sparse_binary_matrix_from_python``: a scipy sparse matrix (any format) or a dense array."""
    if hasattr(value, "tocsr"):
        csr = value.tocsr()
        # Validate before scipy normalization: malformed row offsets must never reach its
        # compiled indexing loops. Keep the reference caster/from_csr exception contract.
        nrows, ncols = (int(x) for x in csr.shape)
        for field in ("indptr", "indices"):
            arr = np.asarray(getattr(csr, field))
            if arr.ndim != 1 or arr.dtype not in (np.dtype("int32"), np.dtype("int64"), np.dtype("uint32"), np.dtype("uint64")):
                raise RuntimeError("scipy sparse matrix indptr/indices has unsupported dtype; expected int32, int64, uint32, or uint64.")
            if np.any(arr < 0):
                raise RuntimeError(f"{field} contains a negative index")
            if np.any(arr > _U32):
                largest = int(arr.max())
                raise RuntimeError(f"{field} exceeds sparse_binary_matrix index_type (uint32_t) range; got {largest}")
        indptr, indices = csr.indptr.tolist(), csr.indices.tolist()
        prefix = "sparse_binary_matrix::from_csr: "
        if len(indptr) != nrows + 1:
            raise ValueError(prefix + "row_ptrs must have size num_rows + 1")
        if indptr[0] != 0:
            raise ValueError(prefix + "row_ptrs[0] must be zero")
        if any(a > b for a, b in zip(indptr, indptr[1:])):
            raise ValueError(prefix + "row_ptrs must be non-decreasing")
        if indptr[-1] != len(indices):
            raise ValueError(prefix + "last row_ptr must equal nnz (col_indices.size())")
        if any(c >= ncols for c in indices):
            raise ValueError(prefix + "column index out of range")
        if len(csr.data) != len(indices):
            raise ValueError(prefix + "data and col_indices must have equal size")
        csr = csr.copy()
        csr.sum_duplicates()
        csr.eliminate_zeros()
        csr.sort_indices()
        nrows, ncols = (int(x) for x in csr.shape)
        for dim, name in ((nrows, "num_rows"), (ncols, "num_cols")):
            if dim > _U32:
                raise RuntimeError(f"{name} exceeds sparse_binary_matrix index_type (uint32_t) range; got {dim}")
        indptr, indices = csr.indptr.tolist(), csr.indices.tolist()
        cols: List[List[int]] = [[] for _ in range(ncols)]
        for i in range(nrows):
            for c in indices[indptr[i] : indptr[i + 1]]:
                cols[c].append(i)
        return _Mat(nrows, ncols, cols)
    a = np.asarray(value).astype(np.uint8, copy=False)
    if a.ndim != 2:
        raise ValueError("H must be a 2-D uint8 array")
    return _Mat.from_dense(a)


# -- The code's matrices and the noise ---------------------------------------------------------------------


def _matrix_property(name: str, doc: str) -> property:
    attr = "_" + name

    def get(self: "CssCodes") -> np.ndarray:
        return getattr(self, attr).to_dense()

    def set_(self: "CssCodes", value: Any) -> None:
        setattr(self, attr, _from_python(value))

    return property(get, set_, doc=doc)


class CssCodes:
    """CSS code generator matrices: parity-check and logical operators.

    Each matrix column corresponds to one data qubit. Assign each matrix from a dense NumPy array
    or a scipy sparse matrix, or build the whole set with ``css_matrices_from_code()``. Reading a
    matrix back returns a dense uint8 NumPy array. Dense array-like values (including nested
    lists) are converted with ``np.asarray(...).astype(np.uint8)`` before testing nonzero
    support; this is broader than the upstream dense object's ``astype`` method requirement."""

    def __init__(self) -> None:
        self._hz = self._hx = self._lz = self._lx = _Mat()

    hz = _matrix_property("hz", "Z stabilizers [n_z_checks x n_qubits]")
    hx = _matrix_property("hx", "X stabilizers [n_x_checks x n_qubits]")
    lz = _matrix_property("lz", "Z logical operators [k x n_qubits]")
    lx = _matrix_property("lx", "X logical operators [k x n_qubits]")


def _real(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise TypeError(f"{name} must be a float, not {type(value).__name__}")
    return float(value)


def _reals(value: Any, name: str) -> List[float]:
    if isinstance(value, (str, bytes, dict)) or not hasattr(value, "__iter__"):
        raise TypeError(f"{name} must be a sequence of floats")
    return [_real(v, name + " entries") for v in value]


def _noise_scalar(name: str, doc: str) -> property:
    attr = "_" + name
    return property(lambda self: getattr(self, attr), lambda self, v: setattr(self, attr, _real(v, name)), doc=doc)


def _noise_list(name: str, doc: str) -> property:
    attr = "_" + name
    return property(lambda self: list(getattr(self, attr)), lambda self, v: setattr(self, attr, _reals(v, name)), doc=doc)


class CssNoise:
    """Phenomenological noise parameters for CSS code-capacity DEM construction. Scalar rates
    apply uniformly; per-element vectors override when non-empty (length must equal n_qubits or
    n_checks)."""

    def __init__(self) -> None:
        self._px = self._py = self._pz = self._pm = 0.0
        self._px_per_qubit: List[float] = []
        self._py_per_qubit: List[float] = []
        self._pz_per_qubit: List[float] = []
        self._pm_per_check: List[float] = []

    px = _noise_scalar("px", "Uniform X data-qubit error rate")
    py = _noise_scalar("py", "Uniform Y data-qubit error rate")
    pz = _noise_scalar("pz", "Uniform Z data-qubit error rate")
    pm = _noise_scalar("pm", "Uniform syndrome measurement error rate per check per round")
    px_per_qubit = _noise_list("px_per_qubit", "Per-qubit X rates (overrides px when non-empty)")
    py_per_qubit = _noise_list("py_per_qubit", "Per-qubit Y rates")
    pz_per_qubit = _noise_list("pz_per_qubit", "Per-qubit Z rates")
    pm_per_check = _noise_list("pm_per_check", "Per-check measurement error rates (Z-checks first)")


# -- dem_construction_utils ---------------------------------------------------------------------------------

css_code_matrices = CssCodes
css_noise_params = CssNoise


def _resolve_num_qubits(code: CssCodes) -> int:
    for m in (code._hz, code._hx, code._lz, code._lx):
        if m.ncols > 0:
            return m.ncols
    return 0


def _check_num_cols(m: _Mat, n: int, label: str) -> None:
    if m.nrows != 0 and m.ncols != n:
        raise ValueError(f"{label} num_cols ({m.ncols}) does not match n_qubits ({n})")


def _check_rate_vector_size(rates: List[float], n: int, label: str, size_label: str) -> None:
    if rates and len(rates) != n:
        raise ValueError(f"{label} has {len(rates)} entries but {size_label}={n}")


def _check_per_qubit_size(per_qubit: List[float], n: int, label: str) -> None:
    _check_rate_vector_size(per_qubit, n, label, "n_qubits")


def _check_probability(p: float, label: str) -> None:
    # (Written as a negated range test so that NaN, which compares false against everything, is
    # rejected too.)
    if not (0.0 <= p <= 1.0):
        raise ValueError(f"{label} must be a probability in [0, 1], got {_to_string(p)}")


def _to_string(p: float) -> str:
    # (std::to_string(double): "%f".)
    return "%f" % p


def _validate_noise_rates(noise: CssNoise) -> None:
    _check_probability(noise._px, "px")
    _check_probability(noise._py, "py")
    _check_probability(noise._pz, "pz")
    _check_probability(noise._pm, "pm")
    for rates, label in ((noise._px_per_qubit, "px_per_qubit"), (noise._py_per_qubit, "py_per_qubit"), (noise._pz_per_qubit, "pz_per_qubit"), (noise._pm_per_check, "pm_per_check")):
        for i, p in enumerate(rates):
            _check_probability(p, f"{label}[{i}]")


def _padded_cols(m: _Mat, n: int) -> List[List[int]]:
    cols = [list(c) for c in m.cols]
    if len(cols) < n:
        cols.extend([] for _ in range(n - len(cols)))
    else:
        del cols[n:]
    return cols


def _rate(uniform: float, per_element: List[float], k: int) -> float:
    return uniform if not per_element else per_element[k]


def _active(uniform: float, per_element: List[float], n: int) -> List[int]:
    return [k for k in range(n) if _rate(uniform, per_element, k) > 0.0]


def _check_zero_rows(m: _Mat, name: str) -> None:
    if m.nrows > 0:
        raise ValueError(f"{name} has {m.nrows} rows but zero columns; all matrices must be empty when num_qubits cannot be determined")


class _Setup:
    """What both DEM builders read off the code and noise (after the checks)."""

    __slots__ = ("n", "hz", "hx", "lz", "lx", "nz", "nx", "kz", "kx", "d", "x_qubits", "z_qubits", "y_qubits", "m_checks")


def _setup(code: CssCodes, noise: CssNoise) -> Optional[_Setup]:
    """The validation and fault lists shared by ``dem_from_css_matrices`` and
    ``extended_dem_from_css_matrices``; None when the matrices describe no qubits."""
    n = _resolve_num_qubits(code)
    if n == 0:
        # (A matrix with rows but zero columns is malformed, not a trivially empty code.)
        for m, name in ((code._hz, "hz"), (code._hx, "hx"), (code._lz, "lz"), (code._lx, "lx")):
            _check_zero_rows(m, name)
        if noise._px_per_qubit or noise._py_per_qubit or noise._pz_per_qubit:
            raise ValueError("per-qubit noise vectors are non-empty but all code matrices have zero columns")
        return None
    for m, name in ((code._hz, "hz"), (code._hx, "hx"), (code._lz, "lz"), (code._lx, "lx")):
        _check_num_cols(m, n, name)
    _check_per_qubit_size(noise._px_per_qubit, n, "px_per_qubit")
    _check_per_qubit_size(noise._py_per_qubit, n, "py_per_qubit")
    _check_per_qubit_size(noise._pz_per_qubit, n, "pz_per_qubit")
    s = _Setup()
    s.n = n
    s.hz, s.hx, s.lz, s.lx = (_padded_cols(m, n) for m in (code._hz, code._hx, code._lz, code._lx))
    s.nz, s.nx, s.kz, s.kx = code._hz.nrows, code._hx.nrows, code._lz.nrows, code._lx.nrows
    s.d = s.nz + s.nx
    return s


def _fault_lists(s: _Setup, noise: CssNoise) -> None:
    _check_rate_vector_size(noise._pm_per_check, s.d, "pm_per_check", "n_checks")
    s.x_qubits = _active(noise._px, noise._px_per_qubit, s.n)
    s.z_qubits = _active(noise._pz, noise._pz_per_qubit, s.n)
    s.y_qubits = _active(noise._py, noise._py_per_qubit, s.n)
    s.m_checks = _active(noise._pm, noise._pm_per_check, s.d)


# -- dem_from_css_matrices ---------------------------------------------------------------------------------


def css_matrices_from_code(qec_code: Any) -> CssCodes:
    """Extract CSS generator matrices from a QEC code object.

    Calls ``code.get_parity_z/x()`` and ``code.get_observables_z/x()`` and wraps the results as
    ``CssCodes`` (a result that is not a matrix, as a code without stabilizers of a type returns,
    is an empty 0 x 0 matrix, as CUDA-Q's ``tensor_to_sparse`` gives it; so is one with no rows)."""
    from ._codes import Code

    if not isinstance(qec_code, Code):
        raise TypeError("css_matrices_from_code(): incompatible function arguments: code must be a cudaq_qec.Code")
    m = CssCodes()
    for attr, getter in (("_hz", "get_parity_z"), ("_hx", "get_parity_x"), ("_lz", "get_observables_z"), ("_lx", "get_observables_x")):
        a = np.asarray(getattr(qec_code, getter)())
        setattr(m, attr, _Mat.from_dense(a != 0) if a.ndim == 2 and a.shape[0] > 0 else _Mat())
    return m


def dem_from_css_matrices(code: Any, noise: CssNoise, num_rounds: int = 1) -> Any:
    """Build a T-round code-capacity detector error model from CSS generator matrices.

    Args:
        code: A ``CssCodes`` (its hz/hx/lz/lx matrices), or a ``Code`` object, whose matrices are
            ``css_matrices_from_code(code)``.
        noise: A ``CssNoise`` with the error rates.
        num_rounds: Number of syndrome measurement rounds (default 1).

    Returns:
        The ``DetectorErrorModel``: round ``r``'s data faults flip round ``r``'s detectors and
        round ``r + 1``'s (the last round's only its own); measurement errors on a check do
        likewise; the observables are those of the data faults, the same in every round."""
    from ._dem import DetectorErrorModel

    if not isinstance(code, CssCodes):
        from ._codes import Code

        if not isinstance(code, Code):
            raise TypeError("dem_from_css_matrices(): incompatible function arguments: code must be a CssCodes or a cudaq_qec.Code")
        code = css_matrices_from_code(code)
    if not isinstance(noise, CssNoise):
        raise TypeError("dem_from_css_matrices(): incompatible function arguments: noise must be a CssNoise")
    if isinstance(num_rounds, bool) or not isinstance(num_rounds, (int, np.integer)) or num_rounds < 0:
        raise TypeError("dem_from_css_matrices(): incompatible function arguments: num_rounds must be a non-negative integer")
    T = int(num_rounds)
    if T == 0:
        raise ValueError("num_rounds must be >= 1")
    # (Checked before the early-out for no qubits, so that a malformed rate is reported even
    # when there is no qubit to apply it to.)
    _validate_noise_rates(noise)
    result = DetectorErrorModel()
    s = _setup(code, noise)
    if s is None:
        return result
    _fault_lists(s, noise)
    nz, kz, d = s.nz, s.kz, s.d
    per_round = len(s.x_qubits) + len(s.z_qubits) + len(s.y_qubits) + len(s.m_checks)
    n_errors = T * per_round
    # (The matrices are sized right even when every rate is zero, so that a consumer can read
    # the syndrome and observable counts off the shapes.)
    H = np.zeros((T * d, n_errors), dtype=np.uint8)
    O = np.zeros((s.kz + s.kx, n_errors), dtype=np.uint8)
    rates: List[float] = []
    col = 0
    for r in range(T):
        r_off, r1_off, has_next = r * d, (r + 1) * d, r + 1 < T

        def flip(base: int, rows: List[int], c: int) -> None:
            for row in rows:
                H[r_off + base + row, c] ^= 1
                if has_next:
                    H[r1_off + base + row, c] ^= 1

        # X faults: Z-type detectors, the Z observables.
        for q in s.x_qubits:
            flip(0, s.hz[q], col)
            for row in s.lz[q]:
                O[row, col] ^= 1
            rates.append(_rate(noise._px, noise._px_per_qubit, q))
            col += 1
        # Z faults: X-type detectors (after the Z-type ones), the X observables (after the Z ones).
        for q in s.z_qubits:
            flip(nz, s.hx[q], col)
            for row in s.lx[q]:
                O[kz + row, col] ^= 1
            rates.append(_rate(noise._pz, noise._pz_per_qubit, q))
            col += 1
        # Y faults: both.
        for q in s.y_qubits:
            flip(0, s.hz[q], col)
            flip(nz, s.hx[q], col)
            for row in s.lz[q]:
                O[row, col] ^= 1
            for row in s.lx[q]:
                O[kz + row, col] ^= 1
            rates.append(_rate(noise._py, noise._py_per_qubit, q))
            col += 1
        # Measurement errors: a wrong outcome for check k fires its detector this round and, if
        # the next round reads it right, the next round's; no observable flips.
        for k in s.m_checks:
            flip(0, [k], col)
            rates.append(_rate(noise._pm, noise._pm_per_check, k))
            col += 1
    result._H, result._O, result.error_rates = H, O, rates
    return result


__all__ = ["CssCodes", "CssNoise", "css_code_matrices", "css_matrices_from_code", "css_noise_params", "dem_from_css_matrices"]

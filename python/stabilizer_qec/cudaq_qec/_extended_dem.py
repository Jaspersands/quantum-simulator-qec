# Ported from CUDA-Q QEC (NVIDIA/cudaqx, libs/qec: include/cudaq/qec/extended_dem.h,
# dem_chunks_memory.h, lib/extended_dem.cpp, dem_chunks_memory.cpp and python/bindings/
# py_extended_dem.cpp, sparse_matrix_casters.h; Copyright (c) 2026 NVIDIA Corporation & Affiliates,
# Apache License 2.0; see THIRD_PARTY_NOTICES.md), modified for stabilizer_qec: numpy only, the
# sparse matrices kept as the column lists they were given (._css), the closed models this
# package's DetectorErrorModel.
"""CUDA-Q QEC's extended detector error model: one chunk of a DEM (``ExtendedDem``) with named
seam boundaries, the operations that compose chunks into a flat DEM, and the declarative specs
that describe the chunks of a repeated-round experiment.

A chunk holds one H matrix (every detector row), the observable-flip rows O, a prior per fault
column, and a list of named seams: row bands of H, each identified by a ``SeamId`` (the FNV-1a hash
of its name). The rows of H outside every seam are the chunk's interior. A one-round CSS chunk
(``extended_dem_from_css_matrices``) is a ``prev_round`` band, a ``next_round`` band and no
interior; ``dem_stitch`` contracts one chunk's outgoing seam with the next one's incoming seam
(their rows become interior rows of the result), ``dem_close`` and ``dem_close_all`` collapse the
chunks to a flat ``DetectorErrorModel`` (the incoming seam's rows first, the outgoing seam's dropped:
the experiment's last round has no next round to compare against), and ``dem_chunks_to_*`` read off
what a streaming decoder needs (the rounds, each detector's round, ``D_sparse``, ``O_sparse``, the
parity-check matrix).

``DemChunksSpec`` describes the chunks by phase: named ``DemChunkSpec`` entries (``init``,
``bulk``, ``final`` for a memory experiment), the graph of connections between them (a phase
connected to itself repeats, ``num_rounds`` times in all) and the seam names to contract;
``dem_chunks_from_spec`` expands it to the chunk list. It is the form
``decoder_config.dem_chunks`` takes.

Errors are CUDA-Q's: ``ValueError`` where it throws ``std::invalid_argument``, ``OverflowError``
for ``std::overflow_error``, ``IndexError`` for ``std::out_of_range``."""

from __future__ import annotations

import enum
import types
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ._css import CssCodes, CssNoise, _Mat, _canonical, _fault_lists, _rate, _setup, _validate_noise_rates

_U32 = 2**32 - 1
_U64 = 2**64
_I64 = (-(2**63), 2**63 - 1)


# -- seam and phase identifiers -----------------------------------------------------------------------


def _fnv1a_32(name: str) -> int:
    """The FNV-1a 32-bit hash of the UTF-8 bytes of a name up to its first NUL (a C string)."""
    h = 2166136261
    for b in name.encode("utf-8").split(b"\0", 1)[0]:
        h = ((h ^ b) * 16777619) & 0xFFFFFFFF
    return h


_NAMES: Dict[int, str] = {}


def _register_name(sid: "SeamId", name: str) -> None:
    """``seam_id::register_name``: a display name for diagnostics; the same name again is a
    no-op, another name for the same id (an FNV-1a collision) is an error."""
    existing = _NAMES.setdefault(sid.value, name)
    if existing != name:
        raise ValueError(f"seam_id::register_name: FNV1a-32 collision at 0x{sid.value:08x} between '{existing}' and '{name}'")


def _name_of(sid: "SeamId") -> str:
    """``seam_id::name``: the registered name, or ``seam:<hex>``."""
    return _NAMES.get(sid.value, "seam:%08x" % sid.value)


class SeamId:
    """Identifier for a named seam or phase. Constructed from a string name; two SeamIds are
    equal iff constructed from the same string."""

    __slots__ = ("_value",)

    def __init__(self, name: Optional[str] = None) -> None:
        if name is None:
            self._value = 0
        elif isinstance(name, str):
            self._value = _fnv1a_32(name)
        else:
            raise TypeError(f"SeamId(): incompatible constructor arguments: name must be a str, not {type(name).__name__}")

    @property
    def value(self) -> int:
        """FNV1a-32 hash of the name string."""
        return self._value

    def __eq__(self, other: object) -> bool:
        return self._value == other._value if isinstance(other, SeamId) else NotImplemented

    def __ne__(self, other: object) -> bool:
        return self._value != other._value if isinstance(other, SeamId) else NotImplemented

    def __lt__(self, other: "SeamId") -> bool:
        return self._value < other._value if isinstance(other, SeamId) else NotImplemented

    def __hash__(self) -> int:
        return self._value

    def __repr__(self) -> str:
        return f"SeamId({self._value})"


def _seam_id(value: Any, what: str) -> SeamId:
    if not isinstance(value, SeamId):
        raise TypeError(f"{what} must be a SeamId, not {type(value).__name__}")
    return value


def _names_module(name: str, doc: str, **ids: SeamId) -> types.ModuleType:
    module = types.ModuleType(name, doc)
    for key, value in ids.items():
        setattr(module, key, value)
    return module


seam_name = _names_module(
    "seam_name",
    "Standard seam names for memory experiments.\n\n  prev_round  — incoming syndrome seam\n  next_round  — outgoing syndrome seam",
    prev_round=SeamId("prev_round"),
    next_round=SeamId("next_round"),
)
phase_name = _names_module("phase_name", "Standard phase names for memory experiments:\n  init, bulk, final", init=SeamId("init"), bulk=SeamId("bulk"), final=SeamId("final"))
for _module in (seam_name, phase_name):
    for _key, _sid in vars(_module).items():
        if isinstance(_sid, SeamId):
            _register_name(_sid, _key)
del _module, _key, _sid

_PREV, _NEXT = seam_name.prev_round, seam_name.next_round


# -- plain value helpers (nanobind's conversions of the attributes) ---------------------------------------


def _ints(value: Any, what: str) -> List[int]:
    if isinstance(value, (str, bytes, dict)) or not hasattr(value, "__iter__"):
        raise TypeError(f"{what} must be a sequence of integers")
    out = []
    for v in value:
        if not isinstance(v, (int, np.integer)) or not _I64[0] <= int(v) <= _I64[1]:
            raise TypeError(f"{what} must be a sequence of 64-bit integers, not {v!r}")
        out.append(int(v))
    return out


def _floats(value: Any, what: str) -> List[float]:
    if isinstance(value, (str, bytes, dict)) or not hasattr(value, "__iter__"):
        raise TypeError(f"{what} must be a sequence of floats")
    out = []
    for v in value:
        if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, float, np.integer, np.floating)):
            raise TypeError(f"{what} must be a sequence of floats, not {v!r}")
        out.append(float(v))
    return out


def _uint(value: Any, bits: int, what: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not 0 <= int(value) < 2**bits:
        raise TypeError(f"{what} must be an integer in [0, 2**{bits}), not {value!r}")
    return int(value)


def _text(value: Any, what: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{what} must be a str, not {type(value).__name__}")
    return value


# -- the chunk ---------------------------------------------------------------------------------------------


class ExtendedDemSeam:
    """Named seam descriptor: row range in H."""

    __slots__ = ("_id", "_row_begin", "_row_end")

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise TypeError("ExtendedDemSeam: No constructor defined!")

    @staticmethod
    def _make(sid: SeamId, row_begin: int, row_end: int) -> "ExtendedDemSeam":
        s = object.__new__(ExtendedDemSeam)
        s._id, s._row_begin, s._row_end = sid, row_begin, row_end
        return s

    @property
    def id(self) -> SeamId:
        """SeamId identifying this seam."""
        return self._id

    @property
    def row_begin(self) -> int:
        """First row (inclusive) of this seam band in H."""
        return self._row_begin

    @property
    def row_end(self) -> int:
        """One past the last row of this seam band in H."""
        return self._row_end

    def num_rows(self) -> int:
        """Number of rows in this seam band."""
        return (self._row_end - self._row_begin) & _U32

    def __repr__(self) -> str:
        return f"ExtendedDemSeam(id={self._id!r}, row_begin={self._row_begin}, row_end={self._row_end})"


class ExtendedDem:
    """One chunk of a DEM with named seam boundaries.

    H holds all detector rows (seam bands + interior rows). O holds the observable-flip rows.
    error_rates has one entry per fault column.

    Users typically work through dem_chunks_from_spec() and dem_close_all() without inspecting
    fields directly."""

    def __init__(self) -> None:
        self._H = _Mat()
        self._O = _Mat()
        self._rates: List[float] = []
        self._seams: List[ExtendedDemSeam] = []
        self._tags: List[int] = []

    @classmethod
    def _make(cls, H: Any, O: Any, error_rates: Iterable[float], seams: Sequence[Tuple[SeamId, int, int]] = (), tags: Iterable[int] = ()) -> "ExtendedDem":
        """A chunk from its parts (the way the C++ code fills a default-constructed one): H and
        O as a ``_Mat`` or a dense array, ``seams`` as (id, row_begin, row_end)."""
        dem = cls()
        dem._H = H if isinstance(H, _Mat) else _Mat.from_dense(np.asarray(H))
        dem._O = O if isinstance(O, _Mat) else _Mat.from_dense(np.asarray(O))
        dem._rates = [float(p) for p in error_rates]
        for sid, begin, end in seams:
            dem._add_seam(sid, begin, end)
        dem._tags = [int(t) for t in tags]
        return dem

    def _copy(self) -> "ExtendedDem":
        out = ExtendedDem()
        out._H, out._O, out._rates, out._seams, out._tags = self._H, self._O, list(self._rates), list(self._seams), list(self._tags)
        return out

    def num_faults(self) -> int:
        """Number of fault columns."""
        return len(self._rates)

    def num_rows(self) -> int:
        """Total detector rows in H (seam + interior)."""
        return self._H.nrows

    def num_observables(self) -> int:
        """Observable rows."""
        return self._O.nrows

    def num_interior_rows(self) -> int:
        """Rows of H not covered by any seam."""
        seam_total = sum(s.num_rows() for s in self._seams) & _U32
        return self._H.nrows - seam_total if self._H.nrows > seam_total else 0

    def has_seam(self, id: SeamId) -> bool:
        """True if a seam with the given SeamId exists."""
        _seam_id(id, "id")
        return any(s._id == id for s in self._seams)

    def get_seam(self, id: SeamId) -> ExtendedDemSeam:
        """Return the seam descriptor for the given SeamId."""
        _seam_id(id, "id")
        for s in self._seams:
            if s._id == id:
                return s
        raise IndexError("extended_dem::get_seam: no seam with id " + _name_of(id))

    def _add_seam(self, sid: SeamId, row_begin: int, row_end: int) -> None:
        if self.has_seam(sid):
            raise ValueError("extended_dem::add_seam: duplicate seam " + _name_of(sid))
        self._seams.append(ExtendedDemSeam._make(sid, row_begin, row_end))

    @property
    def H(self) -> np.ndarray:
        """All detector rows as a dense uint8 NumPy array [n_rows × n_faults]."""
        return self._H.to_dense()

    @property
    def O(self) -> np.ndarray:
        """Observable rows as a dense uint8 NumPy array [k × n_faults]."""
        return self._O.to_dense()

    @property
    def error_rates(self) -> List[float]:
        """Error rates, one per fault column."""
        return list(self._rates)

    @error_rates.setter
    def error_rates(self, value: Any) -> None:
        self._rates = _floats(value, "error_rates")

    @property
    def seams(self) -> List[ExtendedDemSeam]:
        """List of named seam descriptors."""
        return list(self._seams)

    @property
    def tags(self) -> List[int]:
        """Tags per seam row (flat, in seams[] order).

        dem_stitch compares tags across a contracted boundary, but every builder here numbers
        them positionally as 0..width-1 per seam, so the comparison holds automatically once the
        widths match and adds no safety of its own. Assign your own identifiers here if you need
        the boundary check to verify that both sides name the same physical check."""
        return list(self._tags)

    @tags.setter
    def tags(self, value: Any) -> None:
        if isinstance(value, (str, bytes, dict)) or not hasattr(value, "__iter__"):
            raise TypeError("tags must be a sequence of integers")
        self._tags = [_uint(v, 64, "tags entries") for v in value]

    def validate(self, context: str = "dem") -> None:
        """Raise ValueError unless the chunk is internally consistent."""
        context = _text(context, "context")
        n = self.num_faults()

        def check(m: _Mat, label: str) -> None:
            if m.ncols != n:
                raise ValueError(f"{context}: {label} has {m.ncols} cols but chunk has {n} faults")

        check(self._H, "H")
        # (A default-constructed O (0 x 0) is the "no observables" convention. Any other O has to
        # be column-aligned with H, because the merge and o_sparse helpers read it a column at a time.)
        if self._O.nrows > 0 or self._O.ncols > 0:
            check(self._O, "O")
        n_rows = self._H.nrows
        for s in self._seams:
            if s._row_end > n_rows:
                raise ValueError(f"{context}: seam row_end {s._row_end} > H.num_rows() {n_rows}")
            if s._row_begin > s._row_end:
                raise ValueError(f"{context}: seam has row_begin > row_end")
        # (Overlapping bands would make num_interior_rows(), which subtracts the summed seam
        # widths, disagree with the row-membership test that materializes interior rows.)
        bands = sorted((s._row_begin, s._row_end) for s in self._seams if s.num_rows() > 0)
        for i in range(1, len(bands)):
            if bands[i][0] < bands[i - 1][1]:
                raise ValueError(f"{context}: seam row bands [{bands[i - 1][0]},{bands[i - 1][1]}) and [{bands[i][0]},{bands[i][1]}) overlap")
        tag_count = sum(s.num_rows() for s in self._seams)
        if len(self._tags) != tag_count:
            raise ValueError(f"{context}: tags.size()={len(self._tags)} != sum of seam rows {tag_count}")

    def __repr__(self) -> str:
        return f"ExtendedDem(num_rows={self.num_rows()}, num_faults={self.num_faults()}, num_observables={self.num_observables()}, seams={[(_name_of(s.id), s.row_begin, s.row_end) for s in self._seams]})"


# -- internal matrix helpers (extended_dem.cpp) ---------------------------------------------------------


def _hcat(a: _Mat, b: _Mat) -> _Mat:
    n_total = a.ncols + b.ncols
    if n_total > _U32:
        raise OverflowError("hcat: combined column count exceeds uint32_t")
    return _Mat(a.nrows, n_total, a.cols + b.cols)


def _vstack(a: _Mat, b: _Mat) -> _Mat:
    if b.ncols != a.ncols:
        raise ValueError(f"vstack: column count mismatch ({a.ncols} vs {b.ncols})")
    total_rows = a.nrows + b.nrows
    if total_rows > _U32:
        raise OverflowError("vstack: combined row count exceeds uint32_t")
    a_rows = a.nrows
    return _Mat(total_rows, a.ncols, [ac + [a_rows + r for r in bc] for ac, bc in zip(a.cols, b.cols)])


def _pad_right(m: _Mat, n_total: int) -> _Mat:
    return _Mat(m.nrows, n_total, m.cols + [[] for _ in range(n_total - m.ncols)])


def _pad_left(m: _Mat, n_left: int, n_total: int) -> _Mat:
    cols: List[List[int]] = [[] for _ in range(n_total)]
    for c, col in enumerate(m.cols):
        cols[n_left + c] = col
    return _Mat(m.nrows, n_total, cols)


def _zero(nrows: int, ncols: int) -> _Mat:
    return _Mat(nrows, ncols)


def _row_band(H: _Mat, row_begin: int, row_end: int) -> _Mat:
    """Rows [row_begin, row_end) of H as a matrix of their own."""
    n_rows = row_end - row_begin
    if n_rows == 0:
        return _zero(0, H.ncols)
    return _Mat._rows(n_rows, H.ncols, H.rows()[row_begin:row_end])


def _interior_rows(H: _Mat, seams: List[ExtendedDemSeam]) -> _Mat:
    """The rows of H that are in no seam band."""
    interior = [row for r, row in enumerate(H.rows()) if not any(s._row_begin <= r < s._row_end for s in seams)]
    return _Mat._rows(len(interior), H.ncols, interior)


def _xor_into(dst: np.ndarray, cols: List[List[int]], row_off: int, col_off: int) -> None:
    """``write_sparse_block``: dst[row_off + r, col_off + c] ^= 1 for every listed entry."""
    flat = [(row_off + r) * dst.shape[1] + col_off + c for c, col in enumerate(cols) for r in col]
    if not flat:
        return
    index, count = np.unique(np.array(flat, dtype=np.int64), return_counts=True)
    dst.reshape(-1)[index[count & 1 == 1]] ^= 1


def _tag_offset(dem: ExtendedDem, sid: SeamId) -> int:
    offset = 0
    for s in dem._seams:
        if s._id == sid:
            return offset
        offset += s.num_rows()
    raise IndexError("seam_tag_offset: seam " + _name_of(sid) + " not found")


def _check_seam_boundary(a: ExtendedDem, b: ExtendedDem, from_seam: SeamId, to_seam: SeamId, context: str) -> None:
    """The outgoing seam of ``a`` and the incoming seam of ``b`` must have the same width and the
    same tag on each row (an absent seam is width 0; nothing is checked when both are)."""
    from_w = a.get_seam(from_seam).num_rows() if a.has_seam(from_seam) else 0
    to_w = b.get_seam(to_seam).num_rows() if b.has_seam(to_seam) else 0
    if from_w == 0 and to_w == 0:
        return
    if from_w != to_w:
        raise ValueError(f"{context}: seam width mismatch ({from_w} vs {to_w})")
    off_a, off_b = _tag_offset(a, from_seam), _tag_offset(b, to_seam)
    for k in range(from_w):
        if a._tags[off_a + k] != b._tags[off_b + k]:
            raise ValueError(f"{context}: tag mismatch at seam row {k}")


def _reject_extra_seams(dem: ExtendedDem, from_seam: SeamId, to_seam: SeamId, context: str) -> None:
    """Composition has defined semantics only for the two seams of the connection; a third one
    carrying rows would be dropped from the result."""
    for s in dem._seams:
        if s._id == from_seam or s._id == to_seam or s.num_rows() == 0:
            continue
        raise ValueError(
            f"{context}: seam {_name_of(s._id)} carries {s.num_rows()} rows but is neither from_seam ({_name_of(from_seam)}) nor to_seam ({_name_of(to_seam)}); "
            "composing chunks that carry additional named seams is not supported"
        )


# -- sparse row lists and the chunk specs --------------------------------------------------------------


def _sparse_from_terminated_rows(rows: List[int], num_faults: int) -> _Mat:
    nested: List[List[int]] = []
    current: List[int] = []
    for value in rows:
        if value == -1:
            nested.append(current)
            current = []
            continue
        current.append(value)
    if current:
        raise ValueError("sparse_from_terminated_rows: input did not end with -1 terminator")
    return _Mat._rows(len(nested), num_faults, nested)


def _validate_index_list(rows: List[int], num_faults: int, context: str, field: str) -> None:
    if not rows:
        return
    if rows[-1] != -1:
        raise ValueError(f"{context}.{field} must end with a -1 row terminator")
    row_cols: set = set()
    for value in rows:
        if value == -1:
            row_cols.clear()
            continue
        if value < 0 or value >= num_faults:
            raise ValueError(f"{context}.{field} index {value} out of range")
        if value in row_cols:
            raise ValueError(f"{context}.{field} duplicate column index {value} in a row")
        row_cols.add(value)


class DemSeamSpec:
    """Per-seam sparse spec for one named seam boundary."""

    def __init__(self) -> None:
        self._H: List[int] = []
        self._O: List[int] = []

    @property
    def H_sparse(self) -> List[int]:
        """Syndrome rows, -1 terminated."""
        return list(self._H)

    @H_sparse.setter
    def H_sparse(self, value: Any) -> None:
        self._H = _ints(value, "H_sparse")

    @property
    def O_sparse(self) -> List[int]:
        """Logical-frame rows, -1 terminated (non-memory)."""
        return list(self._O)

    @O_sparse.setter
    def O_sparse(self, value: Any) -> None:
        self._O = _ints(value, "O_sparse")

    def _copy(self) -> "DemSeamSpec":
        out = DemSeamSpec()
        out._H, out._O = list(self._H), list(self._O)
        return out

    def _same(self, other: "DemSeamSpec") -> bool:
        return self._H == other._H and self._O == other._O


class SeamSpecEntry:
    """A (SeamId, DemSeamSpec) pair."""

    def __init__(self) -> None:
        self._id = SeamId()
        self._spec = DemSeamSpec()

    @property
    def id(self) -> SeamId:
        """SeamId of this seam."""
        return self._id

    @id.setter
    def id(self, value: SeamId) -> None:
        self._id = _seam_id(value, "id")

    @property
    def spec(self) -> DemSeamSpec:
        """DemSeamSpec for this seam."""
        return self._spec

    @spec.setter
    def spec(self, value: DemSeamSpec) -> None:
        if not isinstance(value, DemSeamSpec):
            raise TypeError("spec must be a DemSeamSpec")
        self._spec = value._copy()

    def _copy(self) -> "SeamSpecEntry":
        out = SeamSpecEntry()
        out._id, out._spec = self._id, self._spec._copy()
        return out

    def _same(self, other: "SeamSpecEntry") -> bool:
        return self._id == other._id and self._spec._same(other._spec)


def _entries(value: Any, kind: type, what: str) -> list:
    if isinstance(value, (str, bytes, dict)) or not hasattr(value, "__iter__"):
        raise TypeError(f"{what} must be a list of {kind.__name__}")
    items = list(value)
    if not all(isinstance(v, kind) for v in items):
        raise TypeError(f"{what} must be a list of {kind.__name__}")
    return [v._copy() for v in items]


class DemChunkSpec:
    """One DEM phase/chunk as flat index lists.

    Two mutually exclusive forms: H_sparse at chunk level (memory experiments, all seams share
    the same H rows), or seam_specs with per-seam H_sparse/O_sparse for non-memory circuits.
    O_sparse at chunk level → ExtendedDem.O (decoder observable)."""

    def __init__(self) -> None:
        self._num_faults = 0
        self._H: List[int] = []
        self._seam_specs: List[SeamSpecEntry] = []
        self._O: List[int] = []
        self._rates: List[float] = []

    @property
    def num_faults(self) -> int:
        """Fault columns; all matrices have this width."""
        return self._num_faults

    @num_faults.setter
    def num_faults(self, value: Any) -> None:
        self._num_faults = _uint(value, 64, "num_faults")

    @property
    def H_sparse(self) -> List[int]:
        """Shared H rows for all seams, -1 terminated (memory shorthand)."""
        return list(self._H)

    @H_sparse.setter
    def H_sparse(self, value: Any) -> None:
        self._H = _ints(value, "H_sparse")

    @property
    def seam_specs(self) -> List[SeamSpecEntry]:
        """Per-seam specs as a list of SeamSpecEntry (non-memory form)."""
        return list(self._seam_specs)

    @seam_specs.setter
    def seam_specs(self, value: Any) -> None:
        self._seam_specs = _entries(value, SeamSpecEntry, "seam_specs")

    @property
    def O_sparse(self) -> List[int]:
        """Observable rows (-1 terminated)."""
        return list(self._O)

    @O_sparse.setter
    def O_sparse(self, value: Any) -> None:
        self._O = _ints(value, "O_sparse")

    @property
    def error_rates(self) -> List[float]:
        """One prior per fault column."""
        return list(self._rates)

    @error_rates.setter
    def error_rates(self, value: Any) -> None:
        self._rates = _floats(value, "error_rates")

    def is_empty(self) -> bool:
        """True iff no field has been set."""
        return self._num_faults == 0 and not self._H and not self._seam_specs and not self._O and not self._rates

    def expand(self, ids: Sequence[SeamId]) -> None:
        """Expand H_sparse shorthand into seam_specs for the given ids (a no-op when seam_specs is
        already populated; the shorthand H_sparse goes to every seam in ``ids`` with an empty O_sparse)."""
        ids = [_seam_id(i, "ids entries") for i in ids]
        if self._seam_specs or not self._H:
            return
        for sid in ids:
            entry = SeamSpecEntry()
            entry._id = sid
            entry._spec._H = list(self._H)
            self._seam_specs.append(entry)
        self._H = []

    def validate(self, context: str) -> None:
        """Raise ValueError on the first inconsistency."""
        context = _text(context, "context")
        if self._num_faults == 0:
            raise ValueError(context + ".num_faults must be positive")
        if self._num_faults > _U32:
            raise ValueError(context + ".num_faults exceeds uint32_t max")
        if self._H and self._seam_specs:
            raise ValueError(context + ": H_sparse and seam_specs are mutually exclusive")
        if len(self._rates) != self._num_faults:
            raise ValueError(f"{context}.error_rates has {len(self._rates)} entries but num_faults={self._num_faults}")
        for i, p in enumerate(self._rates):
            if not (0.0 <= p <= 1.0):
                raise ValueError(f"{context}.error_rates[{i}] out of [0,1]")
        _validate_index_list(self._H, self._num_faults, context, "H_sparse")
        _validate_index_list(self._O, self._num_faults, context, "O_sparse")
        for e in self._seam_specs:
            _validate_index_list(e._spec._H, self._num_faults, context, f"seam[{_name_of(e._id)}].H_sparse")
            _validate_index_list(e._spec._O, self._num_faults, context, f"seam[{_name_of(e._id)}].O_sparse")

    def _copy(self) -> "DemChunkSpec":
        out = DemChunkSpec()
        out._num_faults, out._H, out._O, out._rates = self._num_faults, list(self._H), list(self._O), list(self._rates)
        out._seam_specs = [e._copy() for e in self._seam_specs]
        return out

    def _same(self, other: "DemChunkSpec") -> bool:
        return (
            self._num_faults == other._num_faults
            and self._H == other._H
            and self._O == other._O
            and self._rates == other._rates
            and len(self._seam_specs) == len(other._seam_specs)
            and all(a._same(b) for a, b in zip(self._seam_specs, other._seam_specs))
        )


def dem_chunk_from_spec(spec: DemChunkSpec, seam_names: Sequence[SeamId] = (), context: str = "dem_chunk") -> ExtendedDem:
    """Build one ExtendedDem from a DemChunkSpec.

    Args:
        spec: DemChunkSpec to materialize.
        seam_names: SeamIds for shorthand expansion (ignored if seam_specs is already populated).
        context: Prefix for any error message.

    Returns:
        ExtendedDem whose seam rows are tagged positionally, 0..width-1 within each seam. See
        ExtendedDem.tags: positional tags make the seam-tag check in dem_stitch equivalent to its
        width check."""
    if not isinstance(spec, DemChunkSpec):
        raise TypeError("dem_chunk_from_spec(): spec must be a DemChunkSpec")
    ids = [_seam_id(i, "seam_names entries") for i in seam_names]
    context = _text(context, "context")
    working = spec._copy()
    working.validate(context)
    working.expand(ids)
    for e in working._seam_specs:
        if e._spec._O:
            raise ValueError(f"{context}: per-seam O_sparse in seam '{_name_of(e._id)}' is not yet supported; use chunk-level O_sparse instead")
    result = ExtendedDem()
    result._rates = list(working._rates)
    nf = working._num_faults
    # (H is the seam bands stacked in seam_specs order.)
    row_cursor = 0
    for e in working._seam_specs:
        seam_H = _sparse_from_terminated_rows(e._spec._H, nf)
        seam_rows = seam_H.nrows
        if result._H.nrows == 0 and result._H.ncols == 0:
            result._H = seam_H if seam_rows > 0 else _zero(0, nf)
        elif seam_rows > 0:
            result._H = _vstack(result._H, seam_H)
        result._add_seam(e._id, row_cursor, row_cursor + seam_rows)
        result._tags.extend(range(seam_rows))
        row_cursor += seam_rows
    if result._H.ncols == 0:
        result._H = _zero(0, nf)
    result._O = _sparse_from_terminated_rows(working._O, nf)
    return result


class SeamConnection:
    """Which seam of one phase contracts against which seam of the next."""

    def __init__(self) -> None:
        self._from = SeamId()
        self._to = SeamId()

    @property
    def from_seam(self) -> SeamId:
        """Seam contracting forward out of a phase."""
        return self._from

    @from_seam.setter
    def from_seam(self, value: SeamId) -> None:
        self._from = _seam_id(value, "from_seam")

    @property
    def to_seam(self) -> SeamId:
        """Seam contracting backward into the next phase."""
        return self._to

    @to_seam.setter
    def to_seam(self, value: SeamId) -> None:
        self._to = _seam_id(value, "to_seam")

    def _copy(self) -> "SeamConnection":
        out = SeamConnection()
        out._from, out._to = self._from, self._to
        return out

    def _same(self, other: "SeamConnection") -> bool:
        return self._from == other._from and self._to == other._to


class PhaseConnection:
    """Directed edge in the phase graph."""

    def __init__(self) -> None:
        self._from = SeamId()
        self._to = SeamId()

    @property
    def from_phase(self) -> SeamId:
        return self._from

    @from_phase.setter
    def from_phase(self, value: SeamId) -> None:
        self._from = _seam_id(value, "from_phase")

    @property
    def to_phase(self) -> SeamId:
        return self._to

    @to_phase.setter
    def to_phase(self, value: SeamId) -> None:
        self._to = _seam_id(value, "to_phase")

    def is_self(self) -> bool:
        """True when from_phase == to_phase (marks the repeating phase)."""
        return self._from == self._to

    def _copy(self) -> "PhaseConnection":
        out = PhaseConnection()
        out._from, out._to = self._from, self._to
        return out

    def _same(self, other: "PhaseConnection") -> bool:
        return self._from == other._from and self._to == other._to


class PhaseSpecEntry:
    """A (PhaseId, DemChunkSpec) pair."""

    def __init__(self) -> None:
        self._id = SeamId()
        self._spec = DemChunkSpec()

    @property
    def id(self) -> SeamId:
        """PhaseId of this phase."""
        return self._id

    @id.setter
    def id(self, value: SeamId) -> None:
        self._id = _seam_id(value, "id")

    @property
    def spec(self) -> DemChunkSpec:
        """DemChunkSpec for this phase."""
        return self._spec

    @spec.setter
    def spec(self, value: DemChunkSpec) -> None:
        if not isinstance(value, DemChunkSpec):
            raise TypeError("spec must be a DemChunkSpec")
        self._spec = value._copy()

    def _copy(self) -> "PhaseSpecEntry":
        out = PhaseSpecEntry()
        out._id, out._spec = self._id, self._spec._copy()
        return out

    def _same(self, other: "PhaseSpecEntry") -> bool:
        return self._id == other._id and self._spec._same(other._spec)


class DemChunksSpec:
    """Multi-phase DEM decomposition with arbitrary named phases.

    Standard memory experiment (linear chain init → bulk×N → final):
      connections = [{init,bulk}, {bulk,bulk}, {bulk,final}]
      seam        = {next_round, prev_round}
      num_rounds  = T   (None for streaming — round count unknown at config time)

    dem_chunks_from_spec() calls phase_sequence() to materialise the full chunk list. num_rounds
    is required for that call when a self-loop is present; for streaming leave it as None and
    process bulk chunks as they arrive."""

    def __init__(self) -> None:
        self._phases: List[PhaseSpecEntry] = []
        self._connections: List[PhaseConnection] = []
        self._seam = SeamConnection()
        self._num_rounds: Optional[int] = None

    @property
    def phases(self) -> List[PhaseSpecEntry]:
        """List of PhaseSpecEntry."""
        return list(self._phases)

    @phases.setter
    def phases(self, value: Any) -> None:
        self._phases = _entries(value, PhaseSpecEntry, "phases")

    @property
    def connections(self) -> List[PhaseConnection]:
        """List of PhaseConnection (phase graph edges)."""
        return list(self._connections)

    @connections.setter
    def connections(self, value: Any) -> None:
        self._connections = _entries(value, PhaseConnection, "connections")

    @property
    def seam(self) -> SeamConnection:
        """SeamConnection used between all adjacent phases."""
        return self._seam

    @seam.setter
    def seam(self, value: SeamConnection) -> None:
        if not isinstance(value, SeamConnection):
            raise TypeError("seam must be a SeamConnection")
        self._seam = value._copy()

    @property
    def num_rounds(self) -> Optional[int]:
        """Total rounds to materialise (None for streaming). Required by phase_sequence() when a self-loop is present."""
        return self._num_rounds

    @num_rounds.setter
    def num_rounds(self, value: Optional[int]) -> None:
        self._num_rounds = None if value is None else _uint(value, 64, "num_rounds")

    def is_empty(self) -> bool:
        """True when no phases or connections have been set."""
        return not self._phases and not self._connections

    def has_repeating_phase(self) -> bool:
        """True when any connection is a self-loop."""
        return any(c.is_self() for c in self._connections)

    def repeating_phase(self) -> SeamId:
        """Return the PhaseId of the self-connected phase."""
        result: Optional[SeamId] = None
        for c in self._connections:
            if c.is_self():
                if result is not None:
                    raise ValueError("dem_chunks_spec: more than one self-connected phase")
                result = c._from
        if result is None:
            raise ValueError("dem_chunks_spec: no self-connected (repeating) phase")
        return result

    def phase_sequence(self) -> List[SeamId]:
        """Expand connections into an ordered list of PhaseIds. Requires num_rounds when a
        self-loop is present.

        With a self-loop (repeating phase) and num_rounds = T the result is [A, B, B, ...(T-2
        times)..., B, C]; without one the length is derived from the connections and num_rounds is
        not consulted. A phase with more than one non-self outgoing edge, or a cycle among the
        non-self edges, is rejected."""
        if not self._phases:
            raise ValueError("dem_chunks_spec::phase_sequence: no phases defined")
        if not self._connections:
            raise ValueError("dem_chunks_spec::phase_sequence: no connections defined")
        # (The non-self edges, and the repeating phase.)
        chain: List[Tuple[SeamId, SeamId]] = []
        rep_phase: Optional[SeamId] = None
        for c in self._connections:
            if c.is_self():
                if rep_phase is not None:
                    raise ValueError("dem_chunks_spec::phase_sequence: multiple self-loops")
                rep_phase = c._from
            else:
                chain.append((c._from, c._to))
        has_rep = rep_phase is not None
        # (num_rounds is required only when a self-loop is present.)
        if has_rep and self._num_rounds is None:
            raise ValueError("dem_chunks_spec::phase_sequence: num_rounds required when a self-loop is present")
        if has_rep and self._num_rounds < 2:  # type: ignore[operator]
            raise ValueError("dem_chunks_spec::phase_sequence: num_rounds must be >= 2")
        if not chain:
            return [rep_phase] * self._num_rounds  # type: ignore[list-item,operator]
        to_phases = {t.value for _, t in chain}
        entry = chain[0][0]
        for f, _ in chain:
            if f.value not in to_phases:
                entry = f
                break
        # (A linear chain without a self-loop has chain-edges + 1 phases; with one, num_rounds
        # sets how many copies of the repeating phase there are.)
        non_repeating = len(chain) + 1
        rounds = self._num_rounds if has_rep else non_repeating
        if has_rep and rounds < non_repeating:  # type: ignore[operator]
            raise ValueError(f"dem_chunks_spec::phase_sequence: num_rounds {rounds} too small for {non_repeating} non-repeating phases")
        rep_count = rounds - non_repeating if has_rep else 0  # type: ignore[operator]
        # (A successor map, so that connection order does not matter.)
        successor: Dict[int, SeamId] = {}
        for f, t in chain:
            if f.value in successor:
                raise ValueError(f"dem_chunks_spec::phase_sequence: phase {_name_of(f)} has multiple non-self outgoing edges")
            successor[f.value] = t
        seq: List[SeamId] = []
        # (Each phase is visited at most once; the repeating phase is expanded in place, not by
        # walking its self-loop. A cyclic graph would otherwise grow the sequence without end.)
        visited: set = set()
        current = entry
        while True:
            if current.value in visited:
                raise ValueError(f"dem_chunks_spec::phase_sequence: connection graph is cyclic; phase {_name_of(current)} is reachable twice")
            visited.add(current.value)
            seq.append(current)
            if has_rep and current == rep_phase:
                seq.extend([rep_phase] * rep_count)  # type: ignore[list-item]
            nxt = successor.get(current.value)
            if nxt is None:
                break
            current = nxt
        if len(seq) != rounds:
            raise ValueError(f"dem_chunks_spec::phase_sequence: computed {len(seq)} phases but expected {rounds}")
        return seq

    def get_phase(self, id: SeamId) -> DemChunkSpec:
        """Return the DemChunkSpec for the phase with the given PhaseId (a copy: change a phase
        through ``phases``). Raises ValueError if no such phase exists."""
        return self._phase(_seam_id(id, "id"))._copy()

    def _phase(self, id: SeamId) -> DemChunkSpec:
        for e in self._phases:
            if e._id == id:
                return e._spec
        raise ValueError("dem_chunks_spec::get_phase: phase " + _name_of(id) + " not found")

    def validate(self) -> None:
        """Raise ValueError if the spec is inconsistent."""
        if self.is_empty():
            raise ValueError("dem_chunks_spec: no phases or connections")
        if not self._connections:
            raise ValueError("dem_chunks_spec: connections must not be empty")
        if not self._phases:
            raise ValueError("dem_chunks_spec: phases must not be empty")
        if self._num_rounds is not None and self._num_rounds < 2:
            raise ValueError("dem_chunks_spec: num_rounds must be >= 2")
        for e in self._phases:
            e._spec.validate("dem_chunks." + _name_of(e._id))

    def _copy(self) -> "DemChunksSpec":
        out = DemChunksSpec()
        out._phases = [e._copy() for e in self._phases]
        out._connections = [c._copy() for c in self._connections]
        out._seam, out._num_rounds = self._seam._copy(), self._num_rounds
        return out

    def _same(self, other: "DemChunksSpec") -> bool:
        return (
            self._num_rounds == other._num_rounds
            and self._seam._same(other._seam)
            and len(self._phases) == len(other._phases)
            and all(a._same(b) for a, b in zip(self._phases, other._phases))
            and len(self._connections) == len(other._connections)
            and all(a._same(b) for a, b in zip(self._connections, other._connections))
        )


def dem_chunks_from_spec(spec: DemChunksSpec) -> List[ExtendedDem]:
    """Expand a DemChunksSpec into a sequence of ExtendedDem chunks.

    Args:
        spec: DemChunksSpec describing the phases and connections.

    Returns:
        List of ExtendedDem in phase_sequence() order."""
    if not isinstance(spec, DemChunksSpec):
        raise TypeError("dem_chunks_from_spec(): spec must be a DemChunksSpec")
    spec.validate()
    seq = spec.phase_sequence()
    result = []
    for i, phase in enumerate(seq):
        # (Every phase gets the incoming seam (to_seam / prev_round): for the first it is the
        # initial-state detector band, which dem_close_all emits first, compared against the zero
        # initial syndrome.)
        ids = [spec._seam._to]
        if i + 1 < len(seq):
            ids.append(spec._seam._from)
        result.append(dem_chunk_from_spec(spec._phase(phase), ids, "dem_chunks." + str(phase.value)))
    return result


# -- extended_dem_from_css_matrices --------------------------------------------------------------------


def extended_dem_from_css_matrices(code: CssCodes, noise: CssNoise) -> ExtendedDem:
    """Build a one-round ExtendedDem from CSS generator matrices.

    H has seam bands for prev_round and next_round (identical for a one-round chunk). O holds the
    observable-flip rows. Tags are sequential.

    Args:
        code: CssCodes instance.
        noise: CssNoise instance.

    Returns:
        One-round ExtendedDem."""
    if not isinstance(code, CssCodes):
        raise TypeError("extended_dem_from_css_matrices(): incompatible function arguments: code must be a CssCodes")
    if not isinstance(noise, CssNoise):
        raise TypeError("extended_dem_from_css_matrices(): incompatible function arguments: noise must be a CssNoise")
    _validate_noise_rates(noise)
    result = ExtendedDem()
    s = _setup(code, noise)
    if s is None:
        return result
    nz, nx, kz, kx = s.nz, s.nx, s.kz, s.kx
    if nz + nx > _U32:
        raise ValueError("extended_dem_from_css_matrices: hz+hx rows exceed uint32_t")
    if kz + kx > _U32:
        raise ValueError("extended_dem_from_css_matrices: lz+lx rows exceed uint32_t")
    d, k = nz + nx, kz + kx
    _fault_lists(s, noise)
    n_faults = len(s.x_qubits) + len(s.z_qubits) + len(s.y_qubits) + len(s.m_checks)
    if n_faults > _U32:
        raise ValueError("extended_dem_from_css_matrices: active fault count exceeds uint32_t")
    if n_faults == 0:
        return result
    syn_cols: List[List[int]] = []
    obs_cols: List[List[int]] = []
    priors: List[float] = []
    for q in s.x_qubits:
        syn_cols.append(list(s.hz[q]))
        obs_cols.append(list(s.lz[q]))
        priors.append(_rate(noise._px, noise._px_per_qubit, q))
    for q in s.z_qubits:
        syn_cols.append([nz + r for r in s.hx[q]])
        obs_cols.append([kz + r for r in s.lx[q]])
        priors.append(_rate(noise._pz, noise._pz_per_qubit, q))
    for q in s.y_qubits:
        syn_cols.append(list(s.hz[q]) + [nz + r for r in s.hx[q]])
        obs_cols.append(list(s.lz[q]) + [kz + r for r in s.lx[q]])
        priors.append(_rate(noise._py, noise._py_per_qubit, q))
    for c in s.m_checks:
        syn_cols.append([c])
        obs_cols.append([])
        priors.append(_rate(noise._pm, noise._pm_per_check, c))
    # (The syndrome matrix once; H is it twice, one band per seam.)
    syn = _Mat.from_cols(d, n_faults, syn_cols)
    result._H = _vstack(syn, syn)
    result._O = _Mat.from_cols(k, n_faults, obs_cols)
    result._rates = priors
    # (Positional tags, identical for both seams: row i of either is check i of the same
    # round-to-round syndrome matrix. Being positional, they carry no identity beyond the width.)
    result._tags = list(range(d)) * 2
    result._add_seam(_PREV, 0, d)
    result._add_seam(_NEXT, d, 2 * d)
    return result


# -- stitching ---------------------------------------------------------------------------------------------


def dem_stitch(a: ExtendedDem, b: ExtendedDem, from_seam: SeamId, to_seam: SeamId) -> ExtendedDem:
    """Stitch two adjacent DEM chunks: contract a's from_seam with b's to_seam.

    The contracted seam rows become new interior rows of the result. Row order of the result H:
    a's to_seam rows (only a's fault columns), a's interior rows, the contracted seam rows (both
    chunks' fault columns), b's interior rows, b's from_seam rows (only b's fault columns). Only
    from_seam and to_seam may carry rows.

    Args:
        a: Left ExtendedDem.
        b: Right ExtendedDem.
        from_seam: SeamId of a's outgoing seam.
        to_seam: SeamId of b's incoming seam.

    Returns:
        Stitched ExtendedDem."""
    for name, v in (("a", a), ("b", b)):
        if not isinstance(v, ExtendedDem):
            raise TypeError(f"dem_stitch(): incompatible function arguments: {name} must be an ExtendedDem")
    _seam_id(from_seam, "from_seam")
    _seam_id(to_seam, "to_seam")
    a.validate("dem_stitch: left")
    b.validate("dem_stitch: right")
    if a.num_observables() != b.num_observables():
        raise ValueError(f"dem_stitch: observable count mismatch ({a.num_observables()} vs {b.num_observables()})")
    if not a.has_seam(from_seam):
        raise ValueError("dem_stitch: left chunk has no seam with id " + str(from_seam.value) + " (from_seam); cannot contract")
    if not b.has_seam(to_seam):
        raise ValueError("dem_stitch: right chunk has no seam with id " + str(to_seam.value) + " (to_seam); cannot contract")
    _reject_extra_seams(a, from_seam, to_seam, "dem_stitch: left")
    _reject_extra_seams(b, from_seam, to_seam, "dem_stitch: right")
    _check_seam_boundary(a, b, from_seam, to_seam, "dem_stitch")
    a_from, b_to = a.get_seam(from_seam), b.get_seam(to_seam)
    seam_d = a_from.num_rows()
    n_A = a.num_faults()
    n_total = n_A + b.num_faults()
    if n_total > _U32:
        raise OverflowError("dem_stitch: combined fault count exceeds uint32_t")
    # (The five row bands; the leading and trailing seams may be absent for asymmetric phase
    # chunks like init and final.)
    a_to_rows = _row_band(a._H, a.get_seam(to_seam)._row_begin, a.get_seam(to_seam)._row_end) if a.has_seam(to_seam) else _zero(0, n_A)
    a_int_rows = _interior_rows(a._H, a._seams)
    a_from_rows = _row_band(a._H, a_from._row_begin, a_from._row_end)
    b_to_rows = _row_band(b._H, b_to._row_begin, b_to._row_end)
    b_int_rows = _interior_rows(b._H, b._seams)
    b_from_rows = _row_band(b._H, b.get_seam(from_seam)._row_begin, b.get_seam(from_seam)._row_end) if b.has_seam(from_seam) else _zero(0, b.num_faults())
    # (a_to | a_interior | contracted seam | b_interior | b_from.)
    contracted = _hcat(a_from_rows, b_to_rows)
    parts = [
        _pad_right(a_to_rows, n_total),
        _pad_right(a_int_rows, n_total),
        contracted,
        _pad_left(b_int_rows, n_A, n_total),
        _pad_left(b_from_rows, n_A, n_total),
    ]
    result_H = _zero(0, n_total)
    for m in parts:
        if m.nrows > 0:
            result_H = _vstack(result_H, m) if result_H.nrows > 0 else m
    if result_H.ncols == 0:
        result_H = _zero(0, n_total)
    r = ExtendedDem()
    r._H = result_H
    r._O = _hcat(a._O, b._O)
    r._rates = a._rates + b._rates
    a_to_n, a_int_n, b_int_n, b_from_n = a_to_rows.nrows, a.num_interior_rows(), b.num_interior_rows(), b_from_rows.nrows
    r._add_seam(to_seam, 0, a_to_n)
    b_from_start = a_to_n + a_int_n + seam_d + b_int_n
    r._add_seam(from_seam, b_from_start, b_from_start + b_from_n)
    # (Tags: a's to_seam, then b's from_seam, when present.)
    if a.has_seam(to_seam):
        off = _tag_offset(a, to_seam)
        r._tags.extend(a._tags[off : off + a_to_n])
    if b.has_seam(from_seam):
        off = _tag_offset(b, from_seam)
        r._tags.extend(b._tags[off : off + b_from_n])
    return r


def _seam_rows_or_zero(chunk: ExtendedDem, sid: SeamId) -> int:
    return chunk.get_seam(sid).num_rows() if chunk.has_seam(sid) else 0


class _ChainLayout:
    """The row and column geometry of a whole chunk chain, computed up front so that the chain can
    be filled in one pass. Row order is that of a left fold of ``dem_stitch``: chunk 0's to_seam,
    chunk 0's interior, the seam between 0 and 1, chunk 1's interior, ..., chunk T-1's interior,
    chunk T-1's from_seam. Dropping the trailing band leaves ``dem_close_all``'s detector rows."""

    __slots__ = ("col_off", "interior_row", "seam_row", "lead", "trail", "trail_row", "num_faults")


def _plan_chain(chunks: List[ExtendedDem], from_seam: SeamId, to_seam: SeamId) -> _ChainLayout:
    T = len(chunks)
    plan = _ChainLayout()
    plan.col_off = [0] * (T + 1)
    plan.interior_row = [0] * T
    plan.seam_row = [0] * (T - 1)
    plan.lead = _seam_rows_or_zero(chunks[0], to_seam)
    plan.trail = _seam_rows_or_zero(chunks[-1], from_seam)
    rows = plan.lead
    for i in range(T):
        plan.col_off[i + 1] = plan.col_off[i] + chunks[i].num_faults()
        plan.interior_row[i] = rows
        rows += chunks[i].num_interior_rows()
        if i + 1 < T:
            plan.seam_row[i] = rows
            rows += _seam_rows_or_zero(chunks[i], from_seam)
    plan.trail_row = rows
    plan.num_faults = plan.col_off[-1]
    return plan


def _chain_row_map(chunk: ExtendedDem, i: int, T: int, plan: _ChainLayout, from_seam: SeamId, to_seam: SeamId) -> List[int]:
    """Where each of chunk i's own H rows lands in the assembled chain. A chunk's to_seam shares its
    rows with the predecessor's from_seam (that is what contracting the boundary means); chunk 0's
    has no predecessor and heads the result instead."""
    to_base = 0 if i == 0 else plan.seam_row[i - 1]
    from_base = plan.seam_row[i] if i + 1 < T else plan.trail_row
    has_to, has_from = chunk.has_seam(to_seam), chunk.has_seam(from_seam)
    to_begin = chunk.get_seam(to_seam)._row_begin if has_to else 0
    to_end = chunk.get_seam(to_seam)._row_end if has_to else 0
    from_begin = chunk.get_seam(from_seam)._row_begin if has_from else 0
    from_end = chunk.get_seam(from_seam)._row_end if has_from else 0
    # (Rows outside both bands are interior; _reject_extra_seams has ruled out a third band
    # carrying rows.)
    row_map = [0] * chunk._H.nrows
    interior_seen = 0
    for r in range(chunk._H.nrows):
        if has_to and to_begin <= r < to_end:
            row_map[r] = to_base + (r - to_begin)
        elif has_from and from_begin <= r < from_end:
            row_map[r] = from_base + (r - from_begin)
        else:
            row_map[r] = plan.interior_row[i] + interior_seen
            interior_seen += 1
    return row_map


def _chain_h_columns(chunks: List[ExtendedDem], plan: _ChainLayout, from_seam: SeamId, to_seam: SeamId, row_limit: int) -> List[List[int]]:
    """Fault columns are block-disjoint across chunks, so each chunk's columns are copied once with
    their rows remapped (linear in the non-zeros, where folding ``dem_stitch`` re-copies the whole
    accumulator at each step). Rows at or past ``row_limit`` are dropped: how the closed forms
    discard the trailing from_seam band."""
    T = len(chunks)
    cols: List[List[int]] = [[] for _ in range(plan.num_faults)]
    for i, chunk in enumerate(chunks):
        row_map = _chain_row_map(chunk, i, T, plan, from_seam, to_seam)
        for j, col in enumerate(chunk._H.cols):
            dst = cols[plan.col_off[i] + j]
            for r in col:
                m = row_map[r]
                if m < row_limit:
                    dst.append(m)
    return cols


def _chain_o_columns(chunks: List[ExtendedDem], plan: _ChainLayout) -> List[List[int]]:
    cols: List[List[int]] = [[] for _ in range(plan.num_faults)]
    for i, chunk in enumerate(chunks):
        # (A chunk with no observables may leave O default-constructed, 0 x 0.)
        if chunk._O.ncols == 0:
            continue
        for j, col in enumerate(chunk._O.cols):
            cols[plan.col_off[i] + j] = list(col)
    return cols


def _validate_chain(chunks: List[ExtendedDem], from_seam: SeamId, to_seam: SeamId, fn: str) -> None:
    """The checks a left fold makes through its repeated dem_stitch calls (validating the inputs
    is equivalent to validating each accumulator, since every accumulator is built from them)."""
    T = len(chunks)
    for i, chunk in enumerate(chunks):
        chunk.validate(f"{fn}: chunk {i}")
        _reject_extra_seams(chunk, from_seam, to_seam, f"{fn}: chunk {i}")
    for i in range(1, T):
        if chunks[i].num_observables() != chunks[0].num_observables():
            raise ValueError(f"{fn}: observable count mismatch at chunk {i}")
    for i in range(T - 1):
        if not chunks[i].has_seam(from_seam):
            raise ValueError(f"{fn}: chunk {i} has no seam with id {_name_of(from_seam)} (from_seam); cannot contract")
        if not chunks[i + 1].has_seam(to_seam):
            raise ValueError(f"{fn}: chunk {i + 1} has no seam with id {_name_of(to_seam)} (to_seam); cannot contract")
        _check_seam_boundary(chunks[i], chunks[i + 1], from_seam, to_seam, f"{fn}: boundary {i}")


def _chunk_list(chunks: Any, fn: str) -> List[ExtendedDem]:
    if isinstance(chunks, (str, bytes, dict)) or not hasattr(chunks, "__iter__"):
        raise TypeError(f"{fn}(): incompatible function arguments: dem_chunks must be a list of ExtendedDem")
    out = list(chunks)
    if not all(isinstance(c, ExtendedDem) for c in out):
        raise TypeError(f"{fn}(): incompatible function arguments: dem_chunks must be a list of ExtendedDem")
    return out


def dem_stitch_all(dem_chunks: Sequence[ExtendedDem], from_seam: SeamId = _NEXT, to_seam: SeamId = _PREV) -> ExtendedDem:
    """Stitch a list of adjacent DEM chunks left-to-right.

    Args:
        dem_chunks: Non-empty list of ExtendedDem.
        from_seam: SeamId contracting forward (default: next_round).
        to_seam: SeamId contracting backward (default: prev_round).

    Returns:
        Fully-stitched ExtendedDem."""
    chunks = _chunk_list(dem_chunks, "dem_stitch_all")
    _seam_id(from_seam, "from_seam")
    _seam_id(to_seam, "to_seam")
    if not chunks:
        raise ValueError("dem_stitch_all: dem_chunks must be non-empty")
    if len(chunks) == 1:
        chunks[0].validate("dem_stitch_all: chunk 0")
        return chunks[0]._copy()
    _validate_chain(chunks, from_seam, to_seam, "dem_stitch_all")
    T = len(chunks)
    plan = _plan_chain(chunks, from_seam, to_seam)
    n_rows = plan.trail_row + plan.trail
    n_faults = plan.num_faults
    result = ExtendedDem()
    # (Remapped rows arrive out of order within a column, so canonicalize.)
    result._H = _Mat(n_rows, n_faults, [_canonical(c) for c in _chain_h_columns(chunks, plan, from_seam, to_seam, n_rows)])
    result._O = _Mat.from_cols(chunks[0].num_observables(), n_faults, _chain_o_columns(chunks, plan))
    for chunk in chunks:
        result._rates.extend(chunk._rates)
    result._add_seam(to_seam, 0, plan.lead)
    result._add_seam(from_seam, plan.trail_row, plan.trail_row + plan.trail)
    # (Tags come from the two surviving seams, as with a pairwise stitch.)
    if chunks[0].has_seam(to_seam):
        off = _tag_offset(chunks[0], to_seam)
        result._tags.extend(chunks[0]._tags[off : off + plan.lead])
    if chunks[T - 1].has_seam(from_seam):
        off = _tag_offset(chunks[T - 1], from_seam)
        result._tags.extend(chunks[T - 1]._tags[off : off + plan.trail])
    return result


# -- closing ----------------------------------------------------------------------------------------------


def _model(H: np.ndarray, O: np.ndarray, rates: List[float]) -> Any:
    from ._dem import DetectorErrorModel

    out = DetectorErrorModel()
    out._H, out._O, out.error_rates = H, O, rates
    return out


def dem_close(dem: ExtendedDem, to_seam: SeamId = _PREV, from_seam: SeamId = _NEXT) -> Any:
    """Collapse an ExtendedDem into a flat detector_error_model.

    to_seam rows appear first in the output (detector[0] = syndrome[0] vs. zero initial state).
    The from_seam rows are dropped. A chunk carrying any other non-empty seam is rejected.

    Args:
        dem: ExtendedDem to close.
        to_seam: SeamId whose rows appear first (default: prev_round).
        from_seam: SeamId whose rows are dropped (default: next_round)."""
    if not isinstance(dem, ExtendedDem):
        raise TypeError("dem_close(): incompatible function arguments: dem must be an ExtendedDem")
    _seam_id(to_seam, "to_seam")
    _seam_id(from_seam, "from_seam")
    dem.validate("dem_close")
    _reject_extra_seams(dem, from_seam, to_seam, "dem_close")
    n_faults, n_obs = dem.num_faults(), dem.num_observables()
    n_lead = dem.get_seam(to_seam).num_rows() if dem.has_seam(to_seam) else 0
    n_det = n_lead + dem.num_interior_rows()
    H = np.zeros((n_det, n_faults), dtype=np.uint8)
    O = np.zeros((n_obs, n_faults), dtype=np.uint8)
    if n_lead > 0:
        ls = dem.get_seam(to_seam)
        _xor_into(H, _row_band(dem._H, ls._row_begin, ls._row_end).cols, 0, 0)
    _xor_into(H, _interior_rows(dem._H, dem._seams).cols, n_lead, 0)
    _xor_into(O, dem._O.cols, 0, 0)
    return _model(H, O, list(dem._rates))


def dem_close_all(dem_chunks: Sequence[ExtendedDem], from_seam: SeamId = _NEXT, to_seam: SeamId = _PREV) -> Any:
    """Build a flat detector_error_model from T chunks in O(T) time.

    Equivalent to dem_close(dem_stitch_all(chunks, from, to), to) without the quadratic
    accumulation; the detector rows are in the same order as a stitched result.

    Args:
        dem_chunks: Non-empty list of ExtendedDem.
        from_seam: SeamId contracting forward (default: next_round).
        to_seam: SeamId contracting backward (default: prev_round).

    Returns:
        detector_error_model ready for any decoder."""
    chunks = _chunk_list(dem_chunks, "dem_close_all")
    _seam_id(from_seam, "from_seam")
    _seam_id(to_seam, "to_seam")
    if not chunks:
        raise ValueError("dem_close_all: dem_chunks must be non-empty")
    for i, chunk in enumerate(chunks):
        chunk.validate(f"dem_close_all: chunk {i}")
        _reject_extra_seams(chunk, from_seam, to_seam, f"dem_close_all: chunk {i}")
    k = chunks[0].num_observables()
    for i in range(1, len(chunks)):
        if chunks[i].num_observables() != k:
            raise ValueError(f"dem_close_all: observable count mismatch at chunk {i}")
    # (Adjacent seam widths and tags, with the helper dem_stitch uses.)
    T = len(chunks)
    for i in range(T - 1):
        _check_seam_boundary(chunks[i], chunks[i + 1], from_seam, to_seam, f"dem_close_all: boundary {i}")
    col_off = [0] * (T + 1)
    for i in range(T):
        col_off[i + 1] = col_off[i] + chunks[i].num_faults()
    n_faults = col_off[-1]
    lead = chunks[0].get_seam(to_seam).num_rows() if chunks[0].has_seam(to_seam) else 0
    total_interior = sum(c.num_interior_rows() for c in chunks)
    total_seam = sum(chunks[i].get_seam(from_seam).num_rows() for i in range(T - 1) if chunks[i].has_seam(from_seam))
    n_det = lead + total_interior + total_seam
    H = np.zeros((n_det, n_faults), dtype=np.uint8)
    O = np.zeros((k, n_faults), dtype=np.uint8)
    rates: List[float] = []
    if lead > 0 and chunks[0].has_seam(to_seam):
        s = chunks[0].get_seam(to_seam)
        _xor_into(H, _row_band(chunks[0]._H, s._row_begin, s._row_end).cols, 0, col_off[0])
    row_cursor = lead
    for i in range(T):
        interior = _interior_rows(chunks[i]._H, chunks[i]._seams)
        if interior.nrows > 0:
            _xor_into(H, interior.cols, row_cursor, col_off[i])
            row_cursor += interior.nrows
        if i + 1 < T and chunks[i].has_seam(from_seam) and chunks[i + 1].has_seam(to_seam):
            fs, ts = chunks[i].get_seam(from_seam), chunks[i + 1].get_seam(to_seam)
            _xor_into(H, _row_band(chunks[i]._H, fs._row_begin, fs._row_end).cols, row_cursor, col_off[i])
            _xor_into(H, _row_band(chunks[i + 1]._H, ts._row_begin, ts._row_end).cols, row_cursor, col_off[i + 1])
            row_cursor += fs.num_rows()
        _xor_into(O, chunks[i]._O.cols, 0, col_off[i])
        rates.extend(chunks[i]._rates)
    return _model(H, O, rates)


# -- streaming decoder utilities -----------------------------------------------------------------------


def _chunk_seam_width(chunk: ExtendedDem, from_seam: SeamId, to_seam: SeamId) -> int:
    if chunk.has_seam(to_seam) and chunk.get_seam(to_seam).num_rows() > 0:
        return chunk.get_seam(to_seam).num_rows()
    if chunk.has_seam(from_seam) and chunk.get_seam(from_seam).num_rows() > 0:
        return chunk.get_seam(from_seam).num_rows()
    return 0


def _validate_chunk_sequence(chunks: List[ExtendedDem], from_seam: SeamId, to_seam: SeamId, fn: str) -> None:
    if not chunks:
        raise ValueError(f"{fn}: dem_chunks must be non-empty")
    d = _chunk_seam_width(chunks[0], from_seam, to_seam)
    if d == 0:
        raise ValueError(f"{fn}: chunk 0 has no seam rows")
    for i, chunk in enumerate(chunks):
        if chunk.num_interior_rows() % d != 0:
            raise ValueError(f"{fn}: chunk {i} interior rows not a whole number of rounds of {d}")
        for s in chunk._seams:
            if s.num_rows() != 0 and s.num_rows() != d:
                raise ValueError(f"{fn}: chunk {i} seam width {s.num_rows()} != sequence width {d}")


def dem_chunk_rounds(dem_chunk: ExtendedDem, from_seam: SeamId = _NEXT, to_seam: SeamId = _PREV) -> int:
    """Return how many measurement rounds a DEM chunk spans."""
    if not isinstance(dem_chunk, ExtendedDem):
        raise TypeError("dem_chunk_rounds(): incompatible function arguments: dem_chunk must be an ExtendedDem")
    _seam_id(from_seam, "from_seam")
    _seam_id(to_seam, "to_seam")
    d = _chunk_seam_width(dem_chunk, from_seam, to_seam)
    if d == 0:
        raise ValueError("dem_chunk_rounds: chunk has no seam rows on either side")
    if dem_chunk.num_interior_rows() % d != 0:
        raise ValueError("dem_chunk_rounds: interior rows not a whole number of rounds")
    # (A non-zero leading seam counts as one round; a zero-row seam is a structural placeholder.)
    leading = 1 if dem_chunk.has_seam(to_seam) and dem_chunk.get_seam(to_seam).num_rows() > 0 else 0
    return leading + dem_chunk.num_interior_rows() // d


def dem_chunks_to_rounds(dem_chunks: Sequence[ExtendedDem], from_seam: SeamId = _NEXT, to_seam: SeamId = _PREV) -> int:
    """Return the total rounds a list of DEM chunks describes."""
    chunks = _chunk_list(dem_chunks, "dem_chunks_to_rounds")
    _seam_id(from_seam, "from_seam")
    _seam_id(to_seam, "to_seam")
    _validate_chunk_sequence(chunks, from_seam, to_seam, "dem_chunks_to_rounds")
    return sum(dem_chunk_rounds(c, from_seam, to_seam) for c in chunks)


def dem_chunks_to_detector_round(dem_chunks: Sequence[ExtendedDem], from_seam: SeamId = _NEXT, to_seam: SeamId = _PREV) -> List[int]:
    """Return the round index for each detector in dem_close_all(chunks)."""
    chunks = _chunk_list(dem_chunks, "dem_chunks_to_detector_round")
    _seam_id(from_seam, "from_seam")
    _seam_id(to_seam, "to_seam")
    _validate_chunk_sequence(chunks, from_seam, to_seam, "dem_chunks_to_detector_round")
    d = _chunk_seam_width(chunks[0], from_seam, to_seam)
    result: List[int] = []
    rnd = 0
    for c in chunks:
        for _ in range(dem_chunk_rounds(c, from_seam, to_seam)):
            result.extend([rnd] * d)
            rnd += 1
    return result


def dem_chunks_to_d_sparse(dem_chunks: Sequence[ExtendedDem], from_seam: SeamId = _NEXT, to_seam: SeamId = _PREV) -> List[List[int]]:
    """Return the D_sparse measurement-to-detector map (memory experiments only).

    d_sparse[det_id] lists the raw measurement bit positions (in a flat T*d buffer, round 0's d
    bits first) that XOR-combine to fire that detector: detector k of round 0 is bit k (against
    the zero initial state), detector r*d+k of a later round the bits (r-1)*d+k and r*d+k (the
    difference of consecutive rounds). For a circuit whose detectors are not pairs of adjacent
    rounds, build D_sparse from circuit analysis instead."""
    chunks = _chunk_list(dem_chunks, "dem_chunks_to_d_sparse")
    _seam_id(from_seam, "from_seam")
    _seam_id(to_seam, "to_seam")
    if not chunks:
        raise ValueError("dem_chunks_to_d_sparse: dem_chunks must be non-empty")
    d = chunks[0].get_seam(to_seam).num_rows() if chunks[0].has_seam(to_seam) else 0
    if d == 0 and chunks[0].has_seam(from_seam):
        d = chunks[0].get_seam(from_seam).num_rows()
    if d == 0:
        raise ValueError("dem_chunks_to_d_sparse: chunk 0 has no seam rows on either side, so its rounds cannot be counted")
    # (dem_chunks_to_rounds validates the seam widths and contractibility.)
    T = dem_chunks_to_rounds(chunks, from_seam, to_seam)
    n_det = T * d
    if T > 0 and (T - 1) * d + (d - 1) > _U32:
        raise OverflowError("dem_chunks_to_d_sparse: measurement bit index exceeds uint32_t max")
    d_sparse: List[List[int]] = [[] for _ in range(n_det)]
    for r in range(T):
        for k in range(d):
            d_sparse[r * d + k] = [k] if r == 0 else [(r - 1) * d + k, r * d + k]
    return d_sparse


def dem_chunks_to_o_sparse(dem_chunks: Sequence[ExtendedDem]) -> List[List[int]]:
    """Return the O_sparse observable-flip map from T DEM chunks: o_sparse[obs_id] lists the
    global fault column indices that flip observable obs_id."""
    chunks = _chunk_list(dem_chunks, "dem_chunks_to_o_sparse")
    if not chunks:
        raise ValueError("dem_chunks_to_o_sparse: dem_chunks must be non-empty")
    for i, chunk in enumerate(chunks):
        chunk.validate(f"dem_chunks_to_o_sparse: chunk {i}")
    k_obs = chunks[0].num_observables()
    for i in range(1, len(chunks)):
        if chunks[i].num_observables() != k_obs:
            raise ValueError(f"dem_chunks_to_o_sparse: observable count mismatch at chunk {i}")
    o_sparse: List[List[int]] = [[] for _ in range(k_obs)]
    col_off = 0
    for c in chunks:
        nc = c.num_faults()
        if nc > 0 and col_off + nc - 1 > _U32:
            raise OverflowError("dem_chunks_to_o_sparse: global fault column exceeds uint32_t")
        # (A chunk with no observables may carry a default-constructed O, which cannot be indexed per fault.)
        if c._O.ncols != 0:
            for fc, col in enumerate(c._O.cols):
                for row in col:
                    o_sparse[row].append(col_off + fc)
        col_off += nc
    return o_sparse


def dem_chunks_to_pcm(dem_chunks: Sequence[ExtendedDem], from_seam: SeamId = _NEXT, to_seam: SeamId = _PREV) -> Any:
    """Build a canonicalized parity-check matrix from T DEM chunks.

    Returns a ``scipy.sparse.csc_matrix`` (scipy must be installed)."""
    chunks = _chunk_list(dem_chunks, "dem_chunks_to_pcm")
    _seam_id(from_seam, "from_seam")
    _seam_id(to_seam, "to_seam")
    if not chunks:
        raise ValueError("dem_chunks_to_pcm: dem_chunks must be non-empty")
    _validate_chain(chunks, from_seam, to_seam, "dem_chunks_to_pcm")
    # (The forward pass dem_stitch_all makes, stopping before the trailing from_seam band that
    # closing drops, built straight as CSC.)
    plan = _plan_chain(chunks, from_seam, to_seam)
    cols = [_canonical(c) for c in _chain_h_columns(chunks, plan, from_seam, to_seam, plan.trail_row)]
    import scipy.sparse as sp

    indptr = np.zeros(len(cols) + 1, dtype=np.int64)
    np.cumsum([len(c) for c in cols], out=indptr[1:])
    indices = np.fromiter((r for c in cols for r in c), dtype=np.int64, count=int(indptr[-1]))
    return sp.csc_matrix((np.ones(len(indices), dtype=np.uint8), indices, indptr), shape=(plan.trail_row, plan.num_faults))


# -- duplicate columns --------------------------------------------------------------------------------------


class PriorCombineMode(enum.Enum):
    """Strategy for dem_merge_duplicate_columns()."""

    or_combine = 0  # p = 1/2*(1-prod(1-2p_i)): GF(2)/XOR merge. Use for physical fault mechanisms.
    sum_combine = 1  # p = min(1, sum(p_i)): linear approximation for small priors.


def _column_supports(dem: ExtendedDem) -> Tuple[List[List[int]], List[List[int]], List[Tuple[int, ...]]]:
    """Each column's support across all blocks: its H rows (ascending, GF(2)-collapsed) then its
    O rows offset past H's."""
    h_cols = dem._H.canonical_cols()
    o_cols = dem._O.canonical_cols()
    n = dem.num_faults()
    # (A chunk with no observables may carry a default-constructed O with no columns at all; it
    # is n empty columns, so that it can be indexed alongside H.)
    if not o_cols:
        o_cols = [[] for _ in range(n)]
    base = dem.num_rows()
    flat = [tuple(h_cols[j]) + tuple(base + r for r in o_cols[j]) for j in range(n)]
    return h_cols, o_cols, flat


def dem_merge_duplicate_columns(dem: ExtendedDem, mode: PriorCombineMode = PriorCombineMode.or_combine) -> ExtendedDem:
    """Merge fault columns with identical row support.

    Row support is compared across all blocks (H rows in row order, then O rows) over GF(2). The
    merged columns come out in lexicographic order of their supports.

    Args:
        dem: ExtendedDem to process.
        mode: PriorCombineMode.

    Returns:
        New ExtendedDem with unique-support columns."""
    if not isinstance(dem, ExtendedDem):
        raise TypeError("dem_merge_duplicate_columns(): incompatible function arguments: dem must be an ExtendedDem")
    if not isinstance(mode, PriorCombineMode):
        raise TypeError("dem_merge_duplicate_columns(): incompatible function arguments: mode must be a PriorCombineMode")
    dem.validate("dem_merge_duplicate_columns")
    n = dem.num_faults()
    if n == 0:
        return dem._copy()
    h_cols, o_cols, flat = _column_supports(dem)
    # support -> [representative column, accumulator, members]
    groups: Dict[Tuple[int, ...], List[Any]] = {}
    for j in range(n):
        p = dem._rates[j]
        group = groups.get(flat[j])
        if group is None:
            groups[flat[j]] = [j, (1.0 - 2.0 * p) if mode is PriorCombineMode.or_combine else p, 1]
        else:
            if mode is PriorCombineMode.or_combine:
                group[1] *= 1.0 - 2.0 * p
            else:
                group[1] += p
            group[2] += 1
    new_H: List[List[int]] = []
    new_O: List[List[int]] = []
    new_rates: List[float] = []
    for sup in sorted(groups):
        rep, acc, members = groups[sup]
        new_H.append(h_cols[rep])
        new_O.append(o_cols[rep])
        if members == 1:
            new_rates.append(dem._rates[rep])
        elif mode is PriorCombineMode.or_combine:
            new_rates.append(0.5 * (1.0 - acc))
        else:
            new_rates.append(min(1.0, acc))
    result = ExtendedDem()
    result._H = _Mat.from_cols(dem.num_rows(), len(new_H), new_H)
    result._O = _Mat.from_cols(dem.num_observables(), len(new_H), new_O)
    result._rates = new_rates
    result._seams = list(dem._seams)
    result._tags = list(dem._tags)
    return result


def are_dem_columns_unique(dem: ExtendedDem) -> bool:
    """Return True iff every fault column has a unique row-support set."""
    if not isinstance(dem, ExtendedDem):
        raise TypeError("are_dem_columns_unique(): incompatible function arguments: dem must be an ExtendedDem")
    dem.validate("are_dem_columns_unique")
    _, _, flat = _column_supports(dem)
    return len(set(flat)) == len(flat)


def assert_dem_columns_unique(dem: ExtendedDem) -> None:
    """Raise ValueError if any two columns share row support."""
    if not isinstance(dem, ExtendedDem):
        raise TypeError("assert_dem_columns_unique(): incompatible function arguments: dem must be an ExtendedDem")
    dem.validate("assert_dem_columns_unique")
    if not are_dem_columns_unique(dem):
        raise ValueError("extended_dem has duplicate fault columns; call dem_merge_duplicate_columns() to merge them.")


def dem_stitch_merged(dem_chunks: Sequence[ExtendedDem], from_seam: SeamId = _NEXT, to_seam: SeamId = _PREV, mode: PriorCombineMode = PriorCombineMode.or_combine) -> ExtendedDem:
    """Stitch chunks then merge duplicate columns.

    Args:
        dem_chunks: Non-empty list of ExtendedDem.
        from_seam: SeamId contracting forward (default: next_round).
        to_seam: SeamId contracting backward (default: prev_round).
        mode: PriorCombineMode for the merge step."""
    return dem_merge_duplicate_columns(dem_stitch_all(dem_chunks, from_seam, to_seam), mode)


__all__ = [
    "DemChunkSpec",
    "DemChunksSpec",
    "DemSeamSpec",
    "ExtendedDem",
    "ExtendedDemSeam",
    "PhaseConnection",
    "PhaseSpecEntry",
    "PriorCombineMode",
    "SeamConnection",
    "SeamId",
    "SeamSpecEntry",
    "are_dem_columns_unique",
    "assert_dem_columns_unique",
    "dem_chunk_from_spec",
    "dem_chunk_rounds",
    "dem_chunks_from_spec",
    "dem_chunks_to_d_sparse",
    "dem_chunks_to_detector_round",
    "dem_chunks_to_o_sparse",
    "dem_chunks_to_pcm",
    "dem_chunks_to_rounds",
    "dem_close",
    "dem_close_all",
    "dem_merge_duplicate_columns",
    "dem_stitch",
    "dem_stitch_all",
    "dem_stitch_merged",
    "extended_dem_from_css_matrices",
    "phase_name",
    "seam_name",
]

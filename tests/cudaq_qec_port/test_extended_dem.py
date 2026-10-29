# CUDA-Q QEC's extended detector error model (NVIDIA/cudaqx 0.8.1, libs/qec: python/tests/test_dem.py's
# decoder-context and extended-DEM tests, and unittests/test_extended_dem.cpp, test_dem_construction.cpp
# and test_dem_construction_code.cpp; Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates, Apache
# License 2.0), re-typed and adapted: chunks with named seams, stitching, closing, the round maps, the
# phase specs, and the DEM of CSS code matrices, on stabilizer_qec.cudaq_qec. The C++ cases that fill a
# chunk by hand fill it through ``ExtendedDem._make`` (its matrices are ``_Mat``s, the sparse binary
# matrix with the index lists as given). Added: the closed model against this package's circuit DEM.
import math

import numpy as np
import pytest

from stabilizer_qec import cudaq_qec as qec
from stabilizer_qec.cudaq_qec._css import _Mat
from stabilizer_qec.cudaq_qec._extended_dem import ExtendedDem, _register_name, _name_of

NEXT, PREV = qec.seam_name.next_round, qec.seam_name.prev_round


# -- helpers (test_extended_dem.cpp's) -----------------------------------------------------------------------


def rep3():
    """The d=3 repetition code (Z basis): H_Z = [[1,1,0],[0,1,1]], L_Z = [[1,0,0]]."""
    m = qec.CssCodes()
    m.hz = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
    m.lz = np.array([[1, 0, 0]], dtype=np.uint8)
    return m


def px_only(p):
    n = qec.CssNoise()
    n.px = p
    return n


def one_round(code=None, noise=None):
    return qec.extended_dem_from_css_matrices(rep3() if code is None else code, px_only(0.01) if noise is None else noise)


def chunk(H, O, rates, seams=(), tags=()):
    return ExtendedDem._make(H, O, rates, seams, tags)


def same_dem(a, b):
    """The same detector matrix, observable matrix and rates."""
    return (
        a.detector_error_matrix.shape == b.detector_error_matrix.shape
        and np.array_equal(a.detector_error_matrix, b.detector_error_matrix)
        and a.observables_flips_matrix.shape == b.observables_flips_matrix.shape
        and np.array_equal(a.observables_flips_matrix, b.observables_flips_matrix)
        and list(a.error_rates) == list(b.error_rates)
    )


def dense(a):
    return np.asarray(a).tolist()


# -- test_dem.py: the decoder context, the extended DEM and the phase specs (lines 799-1218) -------------------------


def test_decoder_context_d_sparse_layout():
    # (qec.d_sparse(m2d) flattens a measurement-to-detector map into the -1-terminated sparse vector
    # a realtime decoder expects.)
    code = qec.get_code("steane")
    noise = qec.NoiseModel()
    noise.add_all_qubit_channel("x", qec.Depolarization2(0.05), 1)
    ctx = qec.decoder_context_from_memory_circuit(code, qec.operation.prep0, 3, noise)
    fc_dem, fc_m2d, fc_m2o = ctx.full_component()
    d_sparse = qec.d_sparse(fc_m2d)
    # One -1 terminator per detector row.
    assert d_sparse.count(-1) == len(fc_m2d)
    # Reconstruct the m2d rows from the flattened form and compare.
    rebuilt, row = [], []
    for v in d_sparse:
        if v == -1:
            rebuilt.append(row)
            row = []
        else:
            row.append(v)
    assert rebuilt == [list(r) for r in fc_m2d]


def test_decoder_context_single_type_code_empty_component():
    # The repetition code has only Z stabilizers, so its X context has no detectors. Verify the
    # empty context is self-consistent (no detectors, empty maps), and the Z context keeps every detector.
    code = qec.get_code("repetition", distance=3)
    noise = qec.NoiseModel()
    noise.add_all_qubit_channel("x", qec.Depolarization2(0.05), 1)
    ctx = qec.decoder_context_from_memory_circuit(code, qec.operation.prep0, 3, noise)
    xc_dem, xc_m2d, xc_m2o = ctx.x_component()
    assert xc_dem.num_detectors() == 0
    assert len(xc_m2d) == 0
    assert qec.d_sparse(xc_m2d) == []  # (empty m2d -> empty D_sparse)
    zc_dem, zc_m2d, zc_m2o = ctx.z_component()
    fc_dem, fc_m2d, fc_m2o = ctx.full_component()
    assert zc_dem.num_detectors() == fc_dem.num_detectors()
    assert len(zc_m2d) == zc_dem.num_detectors()


# (All use a rep-3 single-round chunk: d=2 seam rows, 3 faults.)


def _rep3_dem_chunk(px=0.01):
    code = qec.css_matrices_from_code(qec.get_code("repetition", distance=3))
    noise = qec.CssNoise()
    noise.px_per_qubit = [px, px, px]
    return qec.extended_dem_from_css_matrices(code, noise)


def test_dem_stitch_interior_grows():
    # After one stitch the contracted seam rows become interior (d=2).
    c = _rep3_dem_chunk()
    ab = qec.dem_stitch(c, c, qec.seam_name.next_round, qec.seam_name.prev_round)
    assert ab.num_interior_rows() == 2
    assert ab.get_seam(qec.seam_name.prev_round).num_rows() == 2
    assert ab.num_faults() == 6  # 2 rounds × 3 qubits


def test_dem_stitch_error_rates_concatenated():
    px = 0.01
    c = _rep3_dem_chunk(px)
    ab = qec.dem_stitch(c, c, qec.seam_name.next_round, qec.seam_name.prev_round)
    assert len(ab.error_rates) == 6
    assert all(abs(p - px) < 1e-12 for p in ab.error_rates)


def test_dem_stitch_all_empty_raises():
    with pytest.raises(Exception):
        qec.dem_stitch_all([])


def test_dem_stitch_all_single_chunk_identity():
    c = _rep3_dem_chunk()
    result = qec.dem_stitch_all([c])
    assert result.num_interior_rows() == c.num_interior_rows()
    assert result.num_faults() == c.num_faults()


def test_dem_stitch_all_three_chunks_interior():
    c = _rep3_dem_chunk()
    abc = qec.dem_stitch_all([c, c, c])
    # 2 stitches × 2 contracted seam rows each = 4 interior rows
    assert abc.num_interior_rows() == 4
    assert abc.num_faults() == 9  # 3 rounds × 3 qubits


def test_dem_close_single_chunk_shape():
    # A single rep-3 chunk has d=2 prev_round seam rows and 3 fault columns. dem_close emits the
    # prev_round seam rows as detectors and drops next_round.
    px = 0.01
    chunk_ = _rep3_dem_chunk(px)
    closed = qec.dem_close(chunk_)
    d = chunk_.get_seam(qec.seam_name.prev_round).num_rows()
    assert closed.num_detectors() == d
    assert closed.num_error_mechanisms() == chunk_.num_faults()
    assert np.allclose(closed.error_rates, chunk_.error_rates)


def test_dem_close_all_matches_close_stitch_all():
    # dem_close_all must produce the same DEM as dem_close(dem_stitch_all(...)).
    T = 4
    chunks = [_rep3_dem_chunk() for _ in range(T)]
    via_stitch = qec.dem_close(qec.dem_stitch_all(chunks))
    via_close_all = qec.dem_close_all(chunks)
    assert via_close_all.num_detectors() == via_stitch.num_detectors()
    assert via_close_all.num_error_mechanisms() == via_stitch.num_error_mechanisms()
    assert np.array_equal(np.asarray(via_close_all.detector_error_matrix), np.asarray(via_stitch.detector_error_matrix))
    assert np.array_equal(np.asarray(via_close_all.observables_flips_matrix), np.asarray(via_stitch.observables_flips_matrix))
    assert np.allclose(via_close_all.error_rates, via_stitch.error_rates)


def test_dem_close_all_empty_raises():
    with pytest.raises(Exception):
        qec.dem_close_all([])


def test_dem_chunk_rounds_single():
    assert qec.dem_chunk_rounds(_rep3_dem_chunk()) == 1


def test_dem_chunk_rounds_stitched():
    c = _rep3_dem_chunk()
    ab = qec.dem_stitch(c, c, qec.seam_name.next_round, qec.seam_name.prev_round)
    assert qec.dem_chunk_rounds(ab) == 2


def test_dem_chunks_to_rounds():
    T = 5
    chunks = [_rep3_dem_chunk() for _ in range(T)]
    assert qec.dem_chunks_to_rounds(chunks) == T


def test_dem_chunks_to_detector_round_length_and_values():
    # T chunks of d=2 seam rows → T*d detector entries; entry i belongs to round i//d.
    T = 3
    d = 2
    chunks = [_rep3_dem_chunk() for _ in range(T)]
    rounds = qec.dem_chunks_to_detector_round(chunks)
    assert len(rounds) == T * d
    for i, r in enumerate(rounds):
        assert r == i // d, f"detector {i}: expected round {i // d}, got {r}"


def test_dem_chunks_to_d_sparse_length():
    T = 3
    d = 2
    chunks = [_rep3_dem_chunk() for _ in range(T)]
    d_sparse = qec.dem_chunks_to_d_sparse(chunks)
    # One entry per detector
    assert len(d_sparse) == T * d


def test_dem_chunks_to_o_sparse_length():
    T = 3
    chunks = [_rep3_dem_chunk() for _ in range(T)]
    o_sparse = qec.dem_chunks_to_o_sparse(chunks)
    # One entry per observable (this repetition code has 1)
    assert len(o_sparse) == chunks[0].num_observables() == 1


# Declarative phase specs

REP5_CHECKS_PY = 4


def _make_phase_entry(name_str, num_faults, H_sparse, O_sparse, error_rates):
    s = qec.DemChunkSpec()
    s.num_faults = num_faults
    s.H_sparse = list(H_sparse)
    s.O_sparse = list(O_sparse)
    s.error_rates = list(error_rates)
    e = qec.PhaseSpecEntry()
    e.id = qec.SeamId(name_str)
    e.spec = s
    return e


def _make_connection(from_str, to_str):
    c = qec.PhaseConnection()
    c.from_phase = qec.SeamId(from_str)
    c.to_phase = qec.SeamId(to_str)
    return c


def _rep5_spec(num_rounds=5):
    """A d=5 repetition code as init / bulk / final phases. Fault columns 0..4 are data qubits and
    5..8 measurement errors; the final phase measures destructively so it has only the five data
    columns. Spec-built chunks carry only seam rows (no interior rows)."""
    seam = qec.SeamConnection()
    seam.from_seam = qec.seam_name.next_round
    seam.to_seam = qec.seam_name.prev_round

    spec = qec.DemChunksSpec()
    spec.phases = [
        _make_phase_entry("init", 9, [0, 1, 5, -1, 1, 2, 6, -1, 2, 3, 7, -1, 3, 4, 8, -1], [0, -1], [0.02] * 9),
        _make_phase_entry("bulk", 9, [0, 1, 5, -1, 1, 2, 6, -1, 2, 3, 7, -1, 3, 4, 8, -1], [0, -1], [0.02] * 9),
        _make_phase_entry("final", 5, [0, 1, -1, 1, 2, -1, 2, 3, -1, 3, 4, -1], [0, -1], [0.02] * 5),
    ]
    spec.connections = [
        _make_connection("init", "bulk"),
        _make_connection("bulk", "bulk"),
        _make_connection("bulk", "final"),
    ]
    spec.seam = seam
    spec.num_rounds = num_rounds
    return spec


def test_dem_chunks_spec_validates():
    spec = _rep5_spec()
    spec.validate()
    assert spec.has_repeating_phase()
    assert not spec.is_empty()
    assert qec.DemChunksSpec().is_empty()


def test_dem_chunk_spec_round_trips_through_attributes():
    spec = _rep5_spec()
    assert spec.get_phase(qec.SeamId("init")).num_faults == 9
    assert spec.get_phase(qec.SeamId("final")).num_faults == 5


def test_dem_chunks_from_spec_expands_to_round_count():
    for rounds in range(3, 8):
        spec = _rep5_spec(rounds)
        chunks = qec.dem_chunks_from_spec(spec)
        assert len(chunks) == rounds
        # init has a prev_round seam (initial-state detector band).
        assert chunks[0].has_seam(qec.seam_name.prev_round)
        assert chunks[0].get_seam(qec.seam_name.prev_round).num_rows() > 0
        # final has no outgoing seam.
        assert not chunks[-1].has_seam(qec.seam_name.next_round)
        assert qec.dem_chunks_to_rounds(chunks) == rounds


def test_dem_chunks_from_spec_closes_to_one_detector_band_per_round():
    # init's prev_round seam gives the initial-state detector band (round 0 vs zero), and each
    # inter-chunk boundary gives one more band: rounds total.
    for rounds in range(3, 8):
        spec = _rep5_spec(rounds)
        chunks = qec.dem_chunks_from_spec(spec)
        dem = qec.dem_close_all(chunks)
        assert dem.detector_error_matrix.shape[0] == rounds * REP5_CHECKS_PY
        assert len(qec.dem_chunks_to_d_sparse(chunks)) == rounds * REP5_CHECKS_PY


def test_dem_chunk_from_spec_matches_the_phase_it_describes():
    spec = _rep5_spec()
    init_spec = spec.get_phase(qec.SeamId("init"))
    init = qec.dem_chunk_from_spec(init_spec, [qec.seam_name.next_round])
    assert init.num_faults() == 9
    assert not init.has_seam(qec.seam_name.prev_round)
    assert init.has_seam(qec.seam_name.next_round)
    assert init.get_seam(qec.seam_name.next_round).num_rows() == REP5_CHECKS_PY
    # Called directly with only next_round: no prev_round seam → 0 rounds. (dem_chunks_from_spec
    # always adds prev_round; this calls dem_chunk_from_spec directly with an explicit seam list.)
    assert qec.dem_chunk_rounds(init) == 0


def test_dem_chunks_spec_rejects_mismatched_error_rate_count():
    spec = _rep5_spec()
    for entry in spec.phases:
        if entry.id == qec.SeamId("init"):
            entry.spec.error_rates = [0.02] * 8  # wrong count
    with pytest.raises(Exception):
        spec.validate()


def test_dem_chunks_from_spec_requires_num_rounds_for_repeating_phase():
    spec = _rep5_spec()
    spec.num_rounds = None
    with pytest.raises(Exception):
        qec.dem_chunks_from_spec(spec)


def test_dem_chunks_from_spec_linear_chain_needs_no_num_rounds():
    # A two-phase linear chain (no self-loop) works without num_rounds.
    seam = qec.SeamConnection()
    seam.from_seam = qec.seam_name.next_round
    seam.to_seam = qec.seam_name.prev_round

    spec = qec.DemChunksSpec()
    spec.phases = [
        _make_phase_entry("init", 5, [0, 1, -1, 1, 2, -1, 2, 3, -1, 3, 4, -1], [0, -1], [0.02] * 5),
        _make_phase_entry("final", 5, [0, 1, -1, 1, 2, -1, 2, 3, -1, 3, 4, -1], [0, -1], [0.02] * 5),
    ]
    spec.connections = [_make_connection("init", "final")]
    spec.seam = seam
    # num_rounds absent — valid for a linear chain with no self-loop
    chunks = qec.dem_chunks_from_spec(spec)
    assert len(chunks) == 2


def test_dem_chunks_from_spec_rejects_a_cyclic_connection_graph():
    # A -> B and B -> A has no terminal phase; walking it used to grow the phase sequence until the
    # process ran out of memory.
    seam = qec.SeamConnection()
    seam.from_seam = qec.seam_name.next_round
    seam.to_seam = qec.seam_name.prev_round

    spec = qec.DemChunksSpec()
    spec.phases = [
        _make_phase_entry("A", 5, [0, 1, -1, 1, 2, -1, 2, 3, -1, 3, 4, -1], [0, -1], [0.02] * 5),
        _make_phase_entry("B", 5, [0, 1, -1, 1, 2, -1, 2, 3, -1, 3, 4, -1], [0, -1], [0.02] * 5),
    ]
    spec.connections = [_make_connection("A", "B"), _make_connection("B", "A")]
    spec.seam = seam

    with pytest.raises(Exception):
        spec.phase_sequence()
    with pytest.raises(Exception):
        qec.dem_chunks_from_spec(spec)


# Matrix interop: sparse_binary_matrix has no Python class, so the bindings accept/return NumPy
# and scipy matrices on its behalf.


def test_extended_dem_matrices_are_numpy_arrays():
    c = _rep3_dem_chunk()
    H, O = c.H, c.O

    assert isinstance(H, np.ndarray)
    assert isinstance(O, np.ndarray)
    assert H.dtype == np.uint8
    assert H.shape == (c.num_rows(), c.num_faults())
    assert O.shape == (c.num_observables(), c.num_faults())

    # Each access returns an independent copy, so mutating one cannot corrupt the chunk it came from.
    H[0, 0] ^= 1
    assert not np.array_equal(H, c.H)


def test_css_codes_matrices_accept_numpy_arrays():
    hz = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
    lz = np.array([[1, 0, 0]], dtype=np.uint8)

    codes = qec.CssCodes()
    codes.hz = hz
    codes.lz = lz

    assert np.array_equal(codes.hz, hz)
    assert np.array_equal(codes.lz, lz)

    # A hand-built CssCodes has to drive DEM construction, not just read back.
    noise = qec.CssNoise()
    noise.px_per_qubit = [0.01, 0.01, 0.01]
    chunk_ = qec.extended_dem_from_css_matrices(codes, noise)
    assert chunk_.num_faults() == 3
    assert chunk_.num_observables() == 1


def test_css_codes_matrices_accept_scipy_sparse():
    scipy_sparse = pytest.importorskip("scipy.sparse")
    hz = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)

    codes = qec.CssCodes()
    codes.hz = scipy_sparse.csr_matrix(hz)
    assert np.array_equal(codes.hz, hz)
    for fmt in (scipy_sparse.coo_matrix, scipy_sparse.csc_matrix, scipy_sparse.lil_matrix):
        codes.hz = fmt(hz)
        assert np.array_equal(codes.hz, hz)
    # (Duplicate entries are summed and explicit zeros dropped, as scipy defines the matrix.)
    codes.hz = scipy_sparse.coo_matrix((np.array([1, 1, 1, 0]), (np.array([0, 0, 1, 1]), np.array([0, 0, 1, 2]))), shape=(2, 3))
    assert np.array_equal(codes.hz, np.array([[1, 0, 0], [0, 1, 0]], dtype=np.uint8))


def test_css_codes_matrices_accept_non_uint8_dtype():
    hz = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.int32)
    codes = qec.CssCodes()
    codes.hz = hz
    assert np.array_equal(codes.hz, hz.astype(np.uint8))
    with pytest.raises(ValueError, match="H must be a 2-D uint8 array"):
        codes.hz = np.ones(3)


def test_dem_chunks_to_pcm_returns_a_scipy_matrix():
    scipy_sparse = pytest.importorskip("scipy.sparse")
    c = _rep3_dem_chunk()
    pcm = qec.dem_chunks_to_pcm([c, c])

    assert scipy_sparse.issparse(pcm)
    assert scipy_sparse.isspmatrix_csc(pcm)
    # Same detector geometry as the closed DEM it is built from.
    dem = qec.dem_close_all([c, c])
    assert pcm.shape == dem.detector_error_matrix.shape
    assert np.array_equal(pcm.toarray().astype(np.uint8), dem.detector_error_matrix)


# -- seam_id name registry ----------------------------------------------------------------------------------


def test_seam_id_same_name_is_idempotent():
    _register_name(PREV, "prev_round")
    assert _name_of(PREV) == "prev_round"


def test_seam_id_hash_collision_throws():
    first = qec.SeamId("registry_collision_probe")
    colliding = qec.SeamId()
    colliding._value = first.value  # (the same hash under another name: a forced collision)
    _register_name(first, "registry_collision_probe")
    with pytest.raises(ValueError, match="FNV1a-32 collision at 0x[0-9a-f]{8} between 'registry_collision_probe' and 'other_display_name'"):
        _register_name(colliding, "other_display_name")
    # (The first registration still wins for diagnostics.)
    assert _name_of(first) == "registry_collision_probe"


def test_seam_id_is_the_fnv1a_32_hash_of_its_name():
    # (The standard FNV-1a 32-bit test vectors.)
    assert qec.SeamId("").value == 0x811C9DC5
    assert qec.SeamId("a").value == 0xE40C292C
    assert qec.SeamId("foobar").value == 0xBF9CF968
    assert qec.SeamId().value == 0
    assert qec.SeamId("init") == qec.phase_name.init and qec.SeamId("init") != qec.phase_name.bulk
    assert hash(qec.SeamId("init")) == qec.SeamId("init").value
    assert qec.SeamId("a") < qec.SeamId("b") or qec.SeamId("b") < qec.SeamId("a")
    assert repr(qec.SeamId("a")) == f"SeamId({0xE40C292C})"
    assert (qec.seam_name.prev_round, qec.seam_name.next_round) == (qec.SeamId("prev_round"), qec.SeamId("next_round"))
    assert (qec.phase_name.init, qec.phase_name.bulk, qec.phase_name.final) == (qec.SeamId("init"), qec.SeamId("bulk"), qec.SeamId("final"))
    with pytest.raises(TypeError):
        qec.SeamId(5)


# -- one-round chunk structure ---------------------------------------------------------------------------------


def test_one_round_no_interior():
    assert one_round().num_interior_rows() == 0


def test_one_round_in_syndrome_equals_out_syndrome():
    c = one_round()
    d = c.get_seam(PREV).num_rows()
    H = c.H
    assert np.array_equal(H[:d], H[d:])


def test_one_round_syndrome_matches_flat_dem():
    code, noise = rep3(), px_only(0.01)
    c = qec.extended_dem_from_css_matrices(code, noise)
    flat = qec.dem_from_css_matrices(code, noise)
    # (H has 2d rows, the prev_round seam [0,d) and the next_round seam [d,2d); dem_close writes
    # the to_seam (prev_round) rows first, matching the flat DEM.)
    closed = qec.dem_close(c)
    assert np.array_equal(closed.detector_error_matrix, flat.detector_error_matrix)
    assert np.array_equal(c.O, flat.observables_flips_matrix)
    assert c.error_rates == flat.error_rates


def test_one_round_observable_rows_of_a_code_with_two_logicals_of_each_type():
    # (Added.) The [[4,2,2]] code: rows 0..kz-1 of O are the Z logicals (flipped by X faults), rows
    # kz + r the X logical r (flipped by Z faults), as dem_construction.cpp writes them; Y faults
    # flip both. Columns: the X faults on qubits 0..3, then the Z faults, then the Y faults.
    code = qec.CssCodes()
    code.hx = np.array([[1, 1, 1, 1]], dtype=np.uint8)
    code.hz = np.array([[1, 1, 1, 1]], dtype=np.uint8)
    code.lx = np.array([[1, 1, 0, 0], [1, 0, 1, 0]], dtype=np.uint8)
    code.lz = np.array([[1, 0, 1, 0], [1, 1, 0, 0]], dtype=np.uint8)
    noise = qec.CssNoise()
    noise.px, noise.pz, noise.py = 0.01, 0.02, 0.03
    c = qec.extended_dem_from_css_matrices(code, noise)
    zero = [0, 0, 0, 0]
    want = np.array([
        [1, 0, 1, 0] + zero + [1, 0, 1, 0],  # Z logical 0: X and Y faults on qubits 0, 2
        [1, 1, 0, 0] + zero + [1, 1, 0, 0],  # Z logical 1: qubits 0, 1
        zero + [1, 1, 0, 0] + [1, 1, 0, 0],  # X logical 0 (row kz + 0): Z and Y faults on qubits 0, 1
        zero + [1, 0, 1, 0] + [1, 0, 1, 0],  # X logical 1 (row kz + 1): qubits 0, 2
    ], dtype=np.uint8)
    np.testing.assert_array_equal(c.O, want)
    # (H: the Z check fires on X and Y faults, the X check on Z and Y faults; once per seam.)
    syn = np.array([[1] * 4 + zero + [1] * 4, zero + [1] * 4 + [1] * 4], dtype=np.uint8)
    np.testing.assert_array_equal(c.H, np.vstack([syn, syn]))
    assert c.error_rates == [0.01] * 4 + [0.02] * 4 + [0.03] * 4
    flat = qec.dem_from_css_matrices(code, noise)
    np.testing.assert_array_equal(c.O, flat.observables_flips_matrix)
    np.testing.assert_array_equal(qec.dem_close(c).detector_error_matrix, flat.detector_error_matrix)


def test_one_round_tags():
    c = one_round()
    # (d=2 checks, 2 seams of 2 rows: 4 tags, sequential within each seam.)
    assert c.tags == [0, 1, 0, 1]


# -- stitch dimensions ----------------------------------------------------------------------------------------


def test_stitch_interior_grows_by_seam_rows():
    ab = qec.dem_stitch(one_round(), one_round(), NEXT, PREV)
    assert ab.num_interior_rows() == 2
    assert ab.get_seam(PREV).num_rows() == 2
    assert ab.num_observables() == 1
    assert ab.num_faults() == 6


def test_stitch_fault_column_order():
    ab = qec.dem_stitch(one_round(), one_round(), NEXT, PREV)
    assert ab.error_rates == [0.01] * 6


def test_stitch_three_chunks_interior():
    c = one_round()
    abc = qec.dem_stitch(qec.dem_stitch(c, c, NEXT, PREV), c, NEXT, PREV)
    assert abc.num_interior_rows() == 4
    assert abc.num_faults() == 9


def test_stitch_tree_fold_matches_left_fold():
    # (Interior rows must come out in round order for any association of stitches, or a chunk
    # stitched as a tree would file one of its seams under the wrong round.)
    c = one_round()
    s, t = NEXT, PREV
    left = qec.dem_stitch(qec.dem_stitch(qec.dem_stitch(c, c, s, t), c, s, t), c, s, t)
    tree = qec.dem_stitch(qec.dem_stitch(c, c, s, t), qec.dem_stitch(c, c, s, t), s, t)
    right = qec.dem_stitch(c, qec.dem_stitch(c, qec.dem_stitch(c, c, s, t), s, t), s, t)
    flat = qec.dem_from_css_matrices(rep3(), px_only(0.01), 4)
    for shape in (left, tree, right):
        closed = qec.dem_close(shape)
        assert qec.dem_chunk_rounds(shape) == 4
        assert np.array_equal(closed.detector_error_matrix, flat.detector_error_matrix)
        assert np.array_equal(closed.observables_flips_matrix, flat.observables_flips_matrix)


# -- dem_close: dem_close(dem_stitch_all(T chunks)) == dem_from_css_matrices(T) --------------------------------------


@pytest.mark.parametrize("T", [1, 2, 3, 5])
def test_close_matches_monolithic(T):
    code, noise = rep3(), px_only(0.01)
    c = qec.extended_dem_from_css_matrices(code, noise)
    closed = qec.dem_close(qec.dem_stitch_all([c] * T))
    flat = qec.dem_from_css_matrices(code, noise, T)
    assert closed.num_detectors() == flat.num_detectors()
    assert closed.num_observables() == flat.num_observables()
    assert closed.num_error_mechanisms() == flat.num_error_mechanisms()
    assert same_dem(closed, flat)


def test_close_t2_by_a_single_stitch():
    c = one_round()
    closed = qec.dem_close(qec.dem_stitch(c, c, NEXT, PREV))
    assert same_dem(closed, qec.dem_from_css_matrices(rep3(), px_only(0.01), 2))


# -- tag validation ---------------------------------------------------------------------------------------------


def test_stitch_tag_mismatch_throws():
    # (The tag has to be corrupted by hand: the builders number seam rows positionally, so no pair
    # of chunks they produce can disagree on tags once the seam widths match.)
    c0, c1 = one_round(), one_round()
    c1.tags = [999] + c1.tags[1:]
    with pytest.raises(ValueError, match="dem_stitch: tag mismatch at seam row 0"):
        qec.dem_stitch(c0, c1, NEXT, PREV)


def test_stitch_observable_mismatch_throws():
    code_with_lx = rep3()
    code_with_lx.lx = np.array([[1, 0, 0]], dtype=np.uint8)
    pxpz = qec.CssNoise()
    pxpz.px = pxpz.pz = 0.01
    c0 = one_round()  # (no observables from lx)
    c1 = qec.extended_dem_from_css_matrices(code_with_lx, pxpz)
    with pytest.raises(ValueError, match=r"dem_stitch: observable count mismatch \(1 vs 2\)"):
        qec.dem_stitch(c0, c1, NEXT, PREV)


# -- per-qubit noise propagates through stitch ---------------------------------------------------------------------


def test_per_qubit_noise_sparse_chunk():
    # (Qubit 1 has zero rate in round 0 only: its column is absent from c0 but present in c1.)
    sparse = qec.CssNoise()
    sparse.px_per_qubit = [0.01, 0.0, 0.01]
    c0 = qec.extended_dem_from_css_matrices(rep3(), sparse)
    c1 = one_round()
    assert (c0.num_faults(), c1.num_faults()) == (2, 3)
    closed = qec.dem_close(qec.dem_stitch(c0, c1, NEXT, PREV))
    assert closed.num_error_mechanisms() == 5
    assert closed.num_detectors() == 4


# -- dem_stitch_all edge cases ---------------------------------------------------------------------------------------


def test_stitch_all_empty_throws():
    with pytest.raises(ValueError, match="dem_stitch_all: dem_chunks must be non-empty"):
        qec.dem_stitch_all([])
    with pytest.raises(Exception):
        qec.dem_stitch_all([])


def test_stitch_all_one_chunk_is_identity():
    c0 = one_round()
    result = qec.dem_stitch_all([c0])
    assert result.num_interior_rows() == c0.num_interior_rows()
    assert result.get_seam(PREV).num_rows() == c0.get_seam(PREV).num_rows()
    assert result.num_observables() == c0.num_observables()
    assert result.num_faults() == c0.num_faults()


# -- dem_close_all -----------------------------------------------------------------------------------------------------


def test_close_all_t1_matches_close():
    c = one_round()
    assert same_dem(qec.dem_close_all([c]), qec.dem_close(c))


@pytest.mark.parametrize("T", [2, 4, 5])
def test_close_all_matches_close_of_stitch_all_and_monolithic(T):
    chunks = [one_round() for _ in range(T)]
    via_close_all = qec.dem_close_all(chunks)
    assert same_dem(via_close_all, qec.dem_close(qec.dem_stitch_all(chunks)))
    assert same_dem(via_close_all, qec.dem_from_css_matrices(rep3(), px_only(0.01), T))


def test_close_all_empty_throws():
    with pytest.raises(ValueError, match="dem_close_all: dem_chunks must be non-empty"):
        qec.dem_close_all([])


# -- measurement errors through the extended_dem pipeline ----------------------------------------------------------


def test_measurement_errors_one_round_chunk_extra_columns():
    noise = qec.CssNoise()
    noise.pm = 0.005
    c = qec.extended_dem_from_css_matrices(rep3(), noise)
    # (2 checks, 2 measurement-error columns; no data errors.)
    assert c.num_faults() == 2
    assert c.num_interior_rows() == 0
    assert c.error_rates == [0.005, 0.005]
    H = c.H
    assert (H[0, 0], H[1, 0]) == (1, 0)  # (Z0Z1 fires, Z1Z2 unaffected)
    assert not c.O.any()  # (measurement errors don't flip logicals)


def test_measurement_errors_close_all_matches_monolithic():
    noise = qec.CssNoise()
    noise.pm = 0.005
    chunks = [qec.extended_dem_from_css_matrices(rep3(), noise)] * 3
    assert same_dem(qec.dem_close_all(chunks), qec.dem_from_css_matrices(rep3(), noise, 3))


# -- streaming decoder integration utilities --------------------------------------------------------------------------


def test_chunks_to_detector_round_t2():
    # (d=2 checks, T=2 rounds: both round-0 detectors first.)
    assert qec.dem_chunks_to_detector_round([one_round()] * 2) == [0, 0, 1, 1]


def test_chunks_to_detector_round_t3():
    dr = qec.dem_chunks_to_detector_round([one_round()] * 3)
    assert len(dr) == 6
    assert all(dr[r * 2 + k] == r for r in range(3) for k in range(2))


def test_chunk_rounds_count_interior_rows():
    one = one_round()
    assert qec.dem_chunk_rounds(one) == 1
    assert qec.dem_chunk_rounds(qec.dem_stitch(one, one, NEXT, PREV)) == 2
    assert qec.dem_chunk_rounds(qec.dem_stitch_all([one, one, one])) == 3
    assert qec.dem_chunks_to_rounds([one, qec.dem_stitch(one, one, NEXT, PREV), one]) == 4


def test_detector_round_of_multi_round_chunks_matches_singles():
    one = one_round()
    singles = [one] * 4
    want = qec.dem_chunks_to_detector_round(singles)
    assert len(want) == 8
    assert qec.dem_chunks_to_detector_round([qec.dem_stitch_all(singles)]) == want
    assert qec.dem_chunks_to_detector_round([qec.dem_stitch(one, one, NEXT, PREV), qec.dem_stitch(one, one, NEXT, PREV)]) == want
    assert qec.dem_chunks_to_detector_round([one, qec.dem_stitch_all([one, one, one])]) == want
    assert qec.dem_chunks_to_d_sparse([qec.dem_stitch(one, one, NEXT, PREV), one, one]) == qec.dem_chunks_to_d_sparse(singles)


def test_chunk_utils_partial_round_interior_throws():
    # (d=2 checks: H has 5 rows, prev_round 0..1, next_round 2..3 and one interior row, not a
    # whole number of rounds, so every round-indexed map has to refuse it rather than truncate.)
    c = one_round()
    n = c.num_faults()
    c = chunk(_Mat.from_rows(5, n, [[]] * 5), _Mat.from_rows(0, n, []), [0.01] * n, [(PREV, 0, 2), (NEXT, 2, 4)], [0, 1, 0, 1])
    with pytest.raises(ValueError, match="interior rows not a whole number of rounds"):
        qec.dem_chunk_rounds(c)
    for fn in (qec.dem_chunks_to_rounds, qec.dem_chunks_to_detector_round, qec.dem_chunks_to_d_sparse):
        with pytest.raises(ValueError, match="chunk 0 interior rows not a whole number of rounds of 2"):
            fn([c])


def test_chunks_to_d_sparse_t2_layout():
    # (det 0 -> {0}, det 1 -> {1}, det 2 -> {0, 2}, det 3 -> {1, 3}.)
    assert qec.dem_chunks_to_d_sparse([one_round()] * 2) == [[0], [1], [0, 2], [1, 3]]


def test_chunks_to_o_sparse_t2_rep3_x_only():
    # (6 fault columns, 3 per round; only the faults on qubit 0 (columns 0 and 3) flip the Z0 logical.)
    assert qec.dem_chunks_to_o_sparse([one_round()] * 2) == [[0, 3]]


def test_chunk_utils_empty_chunks_throw():
    for fn in (qec.dem_chunks_to_detector_round, qec.dem_chunks_to_d_sparse, qec.dem_chunks_to_o_sparse):
        with pytest.raises(ValueError, match="dem_chunks must be non-empty"):
            fn([])


def test_malformed_hz_rows_with_zero_cols_throws():
    code = qec.CssCodes()
    code.hz = np.zeros((2, 0), dtype=np.uint8)
    code.hx = np.array([[1], [1], [1]], dtype=np.uint8).T  # (1 x 3)
    noise = qec.CssNoise()
    noise.pz = 0.01
    with pytest.raises(ValueError, match=r"hz num_cols \(0\) does not match n_qubits \(3\)"):
        qec.extended_dem_from_css_matrices(code, noise)


# -- dem_merge_duplicate_columns ----------------------------------------------------------------------------------------


def make_merge_chunk(n_faults, h_cols, rates):
    """One H row (interior, no seams), no observables."""
    return chunk(_Mat.from_cols(1, n_faults, h_cols), _Mat.from_cols(0, n_faults, [[]] * n_faults), rates)


def test_merge_already_unique_is_a_noop():
    c = one_round()
    assert qec.are_dem_columns_unique(c)
    qec.assert_dem_columns_unique(c)
    canon = qec.dem_merge_duplicate_columns(c)
    assert canon.num_faults() == c.num_faults()
    assert canon.error_rates == c.error_rates


def test_merge_duplicates_or_mode():
    dem = make_merge_chunk(2, [[0], [0]], [0.1, 0.2])
    assert not qec.are_dem_columns_unique(dem)
    with pytest.raises(ValueError, match="extended_dem has duplicate fault columns; call dem_merge_duplicate_columns\\(\\) to merge them."):
        qec.assert_dem_columns_unique(dem)
    canon = qec.dem_merge_duplicate_columns(dem, qec.PriorCombineMode.or_combine)
    assert canon.num_faults() == 1
    # (XOR / GF(2) merge: 0.1 + 0.2 - 2*0.1*0.2 = 0.26.)
    assert canon.error_rates[0] == pytest.approx(0.26, abs=1e-12)
    assert qec.are_dem_columns_unique(canon)


def test_merge_duplicates_sum_mode():
    dem = make_merge_chunk(3, [[0], [0], [0]], [0.05, 0.03, 0.02])
    canon = qec.dem_merge_duplicate_columns(dem, qec.PriorCombineMode.sum_combine)
    assert canon.num_faults() == 1
    assert canon.error_rates[0] == pytest.approx(0.10, abs=1e-12)


def test_merge_distinct_supports_unchanged():
    dem = chunk(_Mat.from_cols(2, 2, [[0], [1]]), _Mat.from_cols(0, 2, [[], []]), [0.1, 0.2])
    assert qec.are_dem_columns_unique(dem)
    canon = qec.dem_merge_duplicate_columns(dem)
    assert canon.num_faults() == 2
    assert canon.error_rates == dem.error_rates


def test_merge_compares_columns_over_gf2():
    # (A source may list a row twice in one column; such a column denotes the same vector as one
    # omitting that row. Column 0 fires row 1; column 1 lists row 0 twice (cancelling) plus row 1;
    # column 2 fires row 0 once.)
    dem = chunk(_Mat.from_cols(2, 3, [[1], [0, 1, 0], [0]]), _Mat.from_cols(0, 3, [[], [], []]), [0.1, 0.2, 0.3])
    assert not qec.are_dem_columns_unique(dem)
    with pytest.raises(ValueError, match="duplicate fault columns"):
        qec.assert_dem_columns_unique(dem)
    canon = qec.dem_merge_duplicate_columns(dem, qec.PriorCombineMode.or_combine)
    assert canon.num_faults() == 2
    assert qec.are_dem_columns_unique(canon)
    # (Lexicographic output order puts {0} before {1}, and the cancelled row is gone from the stored column.)
    assert canon._H.cols == [[0], [1]]
    assert canon.error_rates[0] == pytest.approx(0.3, abs=1e-12)
    assert canon.error_rates[1] == pytest.approx(0.26, abs=1e-12)


def test_merge_preserves_tags_and_row_blocks():
    c = one_round()
    canon = qec.dem_merge_duplicate_columns(c)
    assert canon.tags == c.tags
    assert canon.num_interior_rows() == c.num_interior_rows()
    assert canon.num_observables() == c.num_observables()
    assert canon.get_seam(PREV).num_rows() == c.get_seam(PREV).num_rows()


def test_stitch_merged_matches_stitch_then_merge():
    chunks = [one_round()] * 4
    a = qec.dem_merge_duplicate_columns(qec.dem_stitch_all(chunks))
    b = qec.dem_stitch_merged(chunks)
    assert a.num_faults() == b.num_faults()
    assert a.error_rates == b.error_rates


def test_merge_sum_combine_clamps_to_one():
    # (A small-p approximation whose sum can leave the unit interval; every consumer reads the rates as probabilities.)
    dem = make_merge_chunk(3, [[0], [0], [0]], [0.4, 0.4, 0.4])
    dem.validate("test")
    merged = qec.dem_merge_duplicate_columns(dem, qec.PriorCombineMode.sum_combine)
    assert merged.num_faults() == 1 and merged.error_rates[0] == 1.0
    # (or_combine is the GF(2) rule, which also stays in [0, 1]: 1/2 * (1 - 0.2^3).)
    ored = qec.dem_merge_duplicate_columns(dem, qec.PriorCombineMode.or_combine)
    assert ored.num_faults() == 1 and ored.error_rates[0] == pytest.approx(0.5 * (1.0 - 0.2 * 0.2 * 0.2), rel=1e-15)


# -- asymmetric (phase) chunks --------------------------------------------------------------------------------------------

REP5_CHECKS = 4
ROWS5 = [[0, 1, 5], [1, 2, 6], [2, 3, 7], [3, 4, 8]]
NEXT_ROWS5 = [[5], [6], [7], [8]]


def rep5_phase_init():
    # (H: the round-0 detectors as interior rows, then the outgoing seam rows.)
    return chunk(_Mat.from_rows(8, 9, ROWS5 + NEXT_ROWS5), _Mat.from_rows(1, 9, [[0]]), [0.02] * 9, [(NEXT, 4, 8)], [0, 1, 2, 3])


def rep5_phase_bulk():
    return chunk(_Mat.from_rows(8, 9, ROWS5 + NEXT_ROWS5), _Mat.from_rows(1, 9, [[0]]), [0.02] * 9, [(PREV, 0, 4), (NEXT, 4, 8)], [0, 1, 2, 3] * 2)


def rep5_phase_final():
    return chunk(_Mat.from_rows(4, 5, [[0, 1], [1, 2], [2, 3], [3, 4]]), _Mat.from_rows(1, 5, [[0]]), [0.02] * 5, [(PREV, 0, 4)], [0, 1, 2, 3])


def rep5_phases(rounds):
    return [rep5_phase_init()] + [rep5_phase_bulk()] * (rounds - 2) + [rep5_phase_final()]


def test_phase_init_has_no_incoming_seam():
    init = rep5_phase_init()
    assert not init.has_seam(PREV) and init.has_seam(NEXT)
    assert init.get_seam(NEXT).num_rows() == REP5_CHECKS
    assert init.num_interior_rows() == REP5_CHECKS


def test_phase_final_has_no_outgoing_seam():
    fin = rep5_phase_final()
    assert fin.has_seam(PREV) and not fin.has_seam(NEXT)
    assert fin.get_seam(PREV).num_rows() == REP5_CHECKS
    assert fin.num_interior_rows() == 0


def test_phase_bulk_owns_no_interior_rows():
    bulk = rep5_phase_bulk()
    assert bulk.get_seam(PREV).num_rows() == bulk.get_seam(NEXT).num_rows() == REP5_CHECKS
    assert bulk.num_interior_rows() == 0


def test_seam_widths_of_uniform_and_phase_chunks():
    uniform = one_round()
    assert uniform.get_seam(PREV).num_rows() == uniform.get_seam(NEXT).num_rows()
    init = rep5_phase_init()
    assert init.has_seam(NEXT) and not init.has_seam(PREV)


def test_stitch_init_to_bulk():
    # (Init's outgoing seam against bulk's incoming one, though init's own incoming seam is absent.)
    ab = qec.dem_stitch(rep5_phase_init(), rep5_phase_bulk(), NEXT, PREV)
    assert ab.num_interior_rows() == 2 * REP5_CHECKS  # (init's own 4 detectors, then the 4 new seam detectors; bulk adds none)
    assert ab.get_seam(PREV).num_rows() == 0
    assert ab.get_seam(NEXT).num_rows() == REP5_CHECKS
    assert ab.num_faults() == 18
    assert ab.num_observables() == 1


def test_seam_row_xors_adjacent_syndromes():
    ab = qec.dem_stitch(rep5_phase_init(), rep5_phase_bulk(), NEXT, PREV)
    rows = ab._H.rows()
    assert len(rows) == 3 * REP5_CHECKS
    # (Row 0 of the seam block: init's measurement error 0 (column 5) plus bulk's data 0, 1 and measurement error 0, offset by init's 9 columns.)
    assert rows[REP5_CHECKS] == [5, 9, 10, 14]


def test_stitch_all_and_close_chain():
    stitched = qec.dem_stitch_all([rep5_phase_init(), rep5_phase_bulk(), rep5_phase_final()])
    assert stitched.get_seam(PREV).num_rows() == 0 and stitched.get_seam(NEXT).num_rows() == 0
    assert stitched.num_faults() == 23
    flat = qec.dem_close(stitched)
    # (4 init detectors + 4 (init|bulk) seam + 4 (bulk|final) seam.)
    assert flat.detector_error_matrix.shape == (3 * REP5_CHECKS, 23)
    assert flat.observables_flips_matrix.shape[0] == 1
    assert len(flat.error_rates) == 23


def test_repeating_bulk_adds_one_seam_per_round():
    for repeats in range(1, 5):
        phases = [rep5_phase_init()] + [rep5_phase_bulk()] * repeats + [rep5_phase_final()]
        flat = qec.dem_close(qec.dem_stitch_all(phases))
        assert flat.detector_error_matrix.shape == ((repeats + 2) * REP5_CHECKS, 9 * (repeats + 1) + 5)


def test_stitch_after_final_throws():
    with pytest.raises(ValueError):
        qec.dem_stitch(rep5_phase_final(), rep5_phase_bulk(), NEXT, PREV)


def test_stitch_before_init_throws():
    with pytest.raises(ValueError):
        qec.dem_stitch(rep5_phase_bulk(), rep5_phase_init(), NEXT, PREV)


def stitch_fold(chunks):
    acc = chunks[0]
    for c in chunks[1:]:
        acc = qec.dem_stitch(acc, c, NEXT, PREV)
    return acc


def expect_same_chunk(got, want, context):
    assert got.num_rows() == want.num_rows(), context
    assert got.num_faults() == want.num_faults(), context
    assert got.num_observables() == want.num_observables(), context
    assert np.array_equal(got.H, want.H), context
    assert np.array_equal(got.O, want.O), context
    assert got.error_rates == want.error_rates, context
    assert got.tags == want.tags, context
    assert [(s.id, s.row_begin, s.row_end) for s in got.seams] == [(s.id, s.row_begin, s.row_end) for s in want.seams], context


def test_stitch_all_matches_an_explicit_fold():
    # (The one-pass assembly and the fold must agree on the whole chunk, not merely on what survives closing.)
    for rounds in range(2, 7):
        phases = rep5_phases(rounds)
        expect_same_chunk(qec.dem_stitch_all(phases), stitch_fold(phases), f"phases, rounds={rounds}")
    for T in range(2, 7):
        uniform = [one_round()] * T
        expect_same_chunk(qec.dem_stitch_all(uniform), stitch_fold(uniform), f"uniform, T={T}")


def test_close_all_matches_stitch_and_close_on_phases():
    for rounds in range(2, 7):
        phases = rep5_phases(rounds)
        assert same_dem(qec.dem_close_all(phases), qec.dem_close(qec.dem_stitch_all(phases))), rounds


def test_each_phase_spans_one_round():
    # (Init carries round 0 in its interior instead of an incoming seam band: one round, not two.)
    assert [qec.dem_chunk_rounds(c) for c in (rep5_phase_init(), rep5_phase_bulk(), rep5_phase_final())] == [1, 1, 1]
    for rounds in range(2, 7):
        assert qec.dem_chunks_to_rounds(rep5_phases(rounds)) == rounds


def test_pre_stitched_phases_report_the_same_rounds():
    phases = rep5_phases(4)
    head = qec.dem_stitch(phases[0], phases[1], NEXT, PREV)
    assert head.get_seam(PREV).num_rows() == 0  # (still an open init end)
    assert qec.dem_chunk_rounds(head) == 2
    assert qec.dem_chunks_to_rounds([head, phases[2], phases[3]]) == 4


def test_detector_count_agrees_with_round_maps():
    for rounds in range(2, 7):
        phases = rep5_phases(rounds)
        want = rounds * REP5_CHECKS
        assert qec.dem_close_all(phases).num_detectors() == want
        assert len(qec.dem_chunks_to_d_sparse(phases)) == want
        assert len(qec.dem_chunks_to_detector_round(phases)) == want


def test_d_sparse_is_the_timelike_layout():
    rounds = 3
    d_sparse = qec.dem_chunks_to_d_sparse(rep5_phases(rounds))
    assert len(d_sparse) == rounds * REP5_CHECKS
    for k in range(REP5_CHECKS):
        assert d_sparse[k] == [k]
    for r in range(1, rounds):
        for k in range(REP5_CHECKS):
            assert d_sparse[r * REP5_CHECKS + k] == [(r - 1) * REP5_CHECKS + k, r * REP5_CHECKS + k]


def test_detector_round_labels_one_band_per_round():
    rounds = 3
    assert qec.dem_chunks_to_detector_round(rep5_phases(rounds)) == [r for r in range(rounds) for _ in range(REP5_CHECKS)]


def test_o_sparse_spans_every_phase():
    # (Keyed on fault columns, so the phases' differing fault counts are all it tracks: column 0 of each flips the observable.)
    assert qec.dem_chunks_to_o_sparse(rep5_phases(3)) == [[0, 9, 18]]


def test_non_contracting_seam_throws():
    # (Only the two open ends are exempt. A seam that goes missing mid-chain is an error, not a shorter experiment.)
    broken = [rep5_phase_init(), rep5_phase_final(), rep5_phase_bulk()]
    with pytest.raises(ValueError, match=r"dem_stitch_all: chunk 1 has no seam with id next_round \(from_seam\); cannot contract"):
        qec.dem_stitch_all(broken)
    with pytest.raises(ValueError, match=r"dem_stitch: left chunk has no seam with id \d+ \(from_seam\); cannot contract"):
        qec.dem_stitch(broken[1], broken[2], NEXT, PREV)
    with pytest.raises(ValueError, match="dem_close_all: boundary 1: seam width mismatch"):
        qec.dem_close_all(broken)


def test_mismatched_seam_tags_throw():
    # (Equal seam widths are not enough: the two sides have to name the same checks in the same
    # order. The single-pass builders enforce it too, as the fold does.)
    bulk = rep5_phase_bulk()
    assert len(bulk.tags) == 2 * REP5_CHECKS
    tags = bulk.tags
    tags[0], tags[1] = tags[1], tags[0]
    bulk.tags = tags
    permuted = [rep5_phase_init(), bulk, rep5_phase_final()]
    with pytest.raises(ValueError, match="dem_stitch_all: boundary 0: tag mismatch at seam row 0"):
        qec.dem_stitch_all(permuted)
    with pytest.raises(ValueError, match="dem_close_all: boundary 0: tag mismatch at seam row 0"):
        qec.dem_close_all(permuted)


def test_fully_closed_chunk_cannot_be_counted():
    closed = qec.dem_stitch_all(rep5_phases(3))
    assert closed.get_seam(PREV).num_rows() == 0 and closed.get_seam(NEXT).num_rows() == 0
    with pytest.raises(ValueError, match="dem_chunk_rounds: chunk has no seam rows on either side"):
        qec.dem_chunk_rounds(closed)


def test_mismatched_seam_width_throws():
    narrow = rep5_phase_bulk()
    narrow._H = _Mat.from_rows(2, 9, [[0, 1, 5], [1, 2, 6]])
    narrow.tags = [0, 1]
    with pytest.raises(ValueError):
        qec.dem_stitch(rep5_phase_init(), narrow, NEXT, PREV)


# -- declarative phase specs (named phases and connections) ---------------------------------------------------------


def make_phase(name, num_faults, h_sparse, o_sparse, rates):
    s = qec.DemChunkSpec()
    s.num_faults = num_faults
    s.H_sparse = list(h_sparse)
    s.O_sparse = list(o_sparse)
    s.error_rates = list(rates)
    e = qec.PhaseSpecEntry()
    e.id = name if isinstance(name, qec.SeamId) else qec.SeamId(name)
    e.spec = s
    return e


def connection(src, dst):
    c = qec.PhaseConnection()
    c.from_phase = src if isinstance(src, qec.SeamId) else qec.SeamId(src)
    c.to_phase = dst if isinstance(dst, qec.SeamId) else qec.SeamId(dst)
    return c


def rep5_spec(num_rounds=5):
    """A d=5 repetition code as init / bulk / final phases: fault columns 0..4 are data qubits and
    5..8 measurement errors; the final phase measures destructively so it has only the five data columns."""
    seam = qec.SeamConnection()
    seam.from_seam, seam.to_seam = NEXT, PREV
    spec = qec.DemChunksSpec()
    bulk_h = [0, 1, 5, -1, 1, 2, 6, -1, 2, 3, 7, -1, 3, 4, 8, -1]
    spec.phases = [
        make_phase("init", 9, bulk_h, [0, -1], [0.02] * 9),
        make_phase("bulk", 9, bulk_h, [0, -1], [0.02] * 9),
        make_phase("final", 5, [0, 1, -1, 1, 2, -1, 2, 3, -1, 3, 4, -1], [0, -1], [0.02] * 5),
    ]
    spec.connections = [connection("init", "bulk"), connection("bulk", "bulk"), connection("bulk", "final")]
    spec.seam = seam
    spec.num_rounds = num_rounds
    return spec


def test_chunks_spec_validates():
    spec = rep5_spec()
    spec.validate()
    assert spec.has_repeating_phase() and not spec.is_empty()
    assert qec.DemChunksSpec().is_empty()
    assert spec.repeating_phase() == qec.phase_name.bulk


def test_chunk_spec_round_trips_through_attributes():
    spec = rep5_spec()
    assert spec.get_phase(qec.SeamId("init")).num_faults == 9
    assert spec.get_phase(qec.SeamId("final")).num_faults == 5
    with pytest.raises(ValueError, match="dem_chunks_spec::get_phase: phase seam:[0-9a-f]{8} not found"):
        spec.get_phase(qec.SeamId("missing"))


def test_chunks_from_spec_expands_to_round_count():
    for rounds in range(3, 8):
        chunks = qec.dem_chunks_from_spec(rep5_spec(rounds))
        assert len(chunks) == rounds
        # (Init has a prev_round seam: the initial-state detector band. Final has no outgoing seam.)
        assert chunks[0].has_seam(PREV) and chunks[0].get_seam(PREV).num_rows() > 0
        assert not chunks[-1].has_seam(NEXT)
        assert qec.dem_chunks_to_rounds(chunks) == rounds


def test_chunks_from_spec_closes_to_one_detector_band_per_round():
    # (Init's prev_round seam is the initial-state band (round 0 against zero) and each boundary adds one: `rounds` bands.)
    for rounds in range(3, 8):
        chunks = qec.dem_chunks_from_spec(rep5_spec(rounds))
        assert qec.dem_close_all(chunks).detector_error_matrix.shape[0] == rounds * REP5_CHECKS
        assert len(qec.dem_chunks_to_d_sparse(chunks)) == rounds * REP5_CHECKS


def test_chunk_from_spec_matches_the_phase_it_describes():
    init_spec = rep5_spec().get_phase(qec.SeamId("init"))
    init = qec.dem_chunk_from_spec(init_spec, [NEXT])
    assert init.num_faults() == 9
    assert not init.has_seam(PREV) and init.has_seam(NEXT)
    assert init.get_seam(NEXT).num_rows() == REP5_CHECKS
    # (Called directly with only next_round: no prev_round seam, so 0 rounds.)
    assert qec.dem_chunk_rounds(init) == 0


def test_chunks_spec_rejects_mismatched_error_rate_count():
    spec = rep5_spec()
    for entry in spec.phases:  # (the entries are the spec's own)
        if entry.id == qec.SeamId("init"):
            entry.spec.error_rates = [0.02] * 8
    with pytest.raises(ValueError, match=r"dem_chunks.init.error_rates has 8 entries but num_faults=9"):
        spec.validate()


def test_chunks_from_spec_requires_num_rounds_for_repeating_phase():
    spec = rep5_spec()
    spec.num_rounds = None
    with pytest.raises(ValueError, match="num_rounds required when a self-loop is present"):
        qec.dem_chunks_from_spec(spec)


def test_chunks_from_spec_linear_chain_needs_no_num_rounds():
    seam = qec.SeamConnection()
    seam.from_seam, seam.to_seam = NEXT, PREV
    spec = qec.DemChunksSpec()
    h = [0, 1, -1, 1, 2, -1, 2, 3, -1, 3, 4, -1]
    spec.phases = [make_phase("init", 5, h, [0, -1], [0.02] * 5), make_phase("final", 5, h, [0, -1], [0.02] * 5)]
    spec.connections = [connection("init", "final")]
    spec.seam = seam
    assert len(qec.dem_chunks_from_spec(spec)) == 2


def test_chunks_from_spec_rejects_a_cyclic_connection_graph():
    # (A -> B and B -> A has no terminal phase; walking it used to grow the phase sequence until memory ran out.)
    seam = qec.SeamConnection()
    seam.from_seam, seam.to_seam = NEXT, PREV
    spec = qec.DemChunksSpec()
    h = [0, 1, -1, 1, 2, -1, 2, 3, -1, 3, 4, -1]
    spec.phases = [make_phase("A", 5, h, [0, -1], [0.02] * 5), make_phase("B", 5, h, [0, -1], [0.02] * 5)]
    spec.connections = [connection("A", "B"), connection("B", "A")]
    spec.seam = seam
    with pytest.raises(ValueError, match="connection graph is cyclic"):
        spec.phase_sequence()
    with pytest.raises(ValueError, match="connection graph is cyclic"):
        qec.dem_chunks_from_spec(spec)


def test_spec_entries_are_the_specs_own_and_assignment_copies():
    spec = rep5_spec()
    assert spec.phases[0] is spec.phases[0] or spec.phases[0].spec is spec.phases[0].spec
    entry = spec.phases[0]
    entry.spec.num_faults = 7
    assert spec.get_phase(qec.SeamId("init")).num_faults == 7
    # (get_phase hands back a copy; assigning phases copies the entries in.)
    spec.get_phase(qec.SeamId("init")).num_faults = 3
    assert spec.get_phase(qec.SeamId("init")).num_faults == 7
    other = qec.DemChunksSpec()
    other.phases = spec.phases
    other.phases[0].spec.num_faults = 11
    assert spec.get_phase(qec.SeamId("init")).num_faults == 7
    seam = qec.SeamConnection()
    seam.from_seam, seam.to_seam = NEXT, PREV
    spec.seam = seam
    seam.from_seam = PREV
    assert spec.seam.from_seam == NEXT


def test_spec_attribute_conversions():
    s = qec.DemChunkSpec()
    s.H_sparse = np.array([0, -1], dtype=np.int64)
    s.error_rates = (0.5, 1)
    assert s.H_sparse == [0, -1] and s.error_rates == [0.5, 1.0]
    with pytest.raises(TypeError):
        s.num_faults = -1
    with pytest.raises(TypeError):
        s.H_sparse = [1.5]
    with pytest.raises(TypeError):
        s.H_sparse = "ab"
    with pytest.raises(TypeError):
        s.error_rates = ["x"]
    spec = qec.DemChunksSpec()
    with pytest.raises(TypeError):
        spec.num_rounds = -2
    with pytest.raises(TypeError):
        spec.phases = [1]
    with pytest.raises(TypeError):
        spec.seam = qec.PhaseConnection()
    spec.num_rounds = 4
    spec.num_rounds = None
    assert spec.num_rounds is None
    assert qec.DemChunkSpec().is_empty() and not s.is_empty()
    c = qec.PhaseConnection()
    assert c.is_self()  # (both default ids, 0)
    c.from_phase, c.to_phase = qec.SeamId("x"), qec.SeamId("y")
    assert not c.is_self()
    c.to_phase = qec.SeamId("x")
    assert c.is_self()


def test_spec_valid_spec_validates():
    spec = rep5_spec()
    spec.validate()
    assert spec.has_repeating_phase() and not spec.is_empty()
    assert qec.DemChunksSpec().is_empty()


def test_spec_num_faults_must_fit_uint32():
    spec = qec.DemChunkSpec()
    spec.num_faults = 2**32
    spec.error_rates = [0.01]
    with pytest.raises(ValueError, match=r"test.num_faults exceeds uint32_t max"):
        spec.validate("test")


def test_spec_zero_faults_throws():
    with pytest.raises(ValueError, match=r"chunk.num_faults must be positive"):
        qec.DemChunkSpec().validate("chunk")


def test_spec_expands_to_requested_round_count():
    for rounds in range(3, 7):
        chunks = qec.dem_chunks_from_spec(rep5_spec(rounds))
        assert len(chunks) == rounds
        assert chunks[0].has_seam(PREV) and chunks[0].get_seam(PREV).num_rows() > 0
        assert not (chunks[-1].has_seam(NEXT) and chunks[-1].get_seam(NEXT).num_rows() > 0)
        # (rounds seam contractions: the initial-state band plus (rounds-1) inter-chunk boundaries.)
        assert qec.dem_close_all(chunks).detector_error_matrix.shape[0] == rounds * REP5_CHECKS


def test_spec_two_rounds_needs_no_bulk_self_loop():
    spec = rep5_spec(2)
    spec.connections = [connection("init", "final")]
    spec.phases = [e for e in spec.phases if e.id != qec.phase_name.bulk]
    spec.validate()
    assert len(qec.dem_chunks_from_spec(spec)) == 2


def test_spec_too_few_rounds_throws():
    spec = rep5_spec(1)
    with pytest.raises(ValueError, match="dem_chunks_spec: num_rounds must be >= 2"):
        spec.validate()
    with pytest.raises(ValueError, match="dem_chunks_spec: num_rounds must be >= 2"):
        qec.dem_chunks_from_spec(spec)


def test_spec_missing_required_connections_throws():
    spec = rep5_spec()
    spec.connections = []
    with pytest.raises(ValueError, match="dem_chunks_spec: connections must not be empty"):
        spec.validate()


def test_spec_validation_messages():
    with pytest.raises(ValueError, match="dem_chunks_spec: no phases or connections"):
        qec.DemChunksSpec().validate()
    only_connections = qec.DemChunksSpec()
    only_connections.connections = [connection("a", "b")]
    with pytest.raises(ValueError, match="dem_chunks_spec: phases must not be empty"):
        only_connections.validate()
    with pytest.raises(ValueError, match="dem_chunks_spec::phase_sequence: no phases defined"):
        qec.DemChunksSpec().phase_sequence()
    with pytest.raises(ValueError, match="dem_chunks_spec::phase_sequence: no connections defined"):
        spec = qec.DemChunksSpec()
        spec.phases = [make_phase("a", 1, [0, -1], [], [0.1])]
        spec.phase_sequence()
    with pytest.raises(ValueError, match="dem_chunks_spec: no self-connected"):
        qec.DemChunksSpec().repeating_phase()
    s = rep5_spec()
    s.connections = s.connections + [connection("final", "final")]
    with pytest.raises(ValueError, match="dem_chunks_spec: more than one self-connected phase"):
        s.repeating_phase()
    with pytest.raises(ValueError, match="dem_chunks_spec::phase_sequence: multiple self-loops"):
        s.phase_sequence()
    s = rep5_spec(2)
    with pytest.raises(ValueError, match="num_rounds 2 too small for 3 non-repeating phases"):
        s.phase_sequence()
    s = rep5_spec()
    s.connections = [connection("init", "bulk"), connection("init", "final"), connection("bulk", "bulk")]
    with pytest.raises(ValueError, match="phase init has multiple non-self outgoing edges"):
        s.phase_sequence()


def test_spec_index_list_messages():
    def spec_with(h):
        s = qec.DemChunkSpec()
        s.num_faults = 2
        s.error_rates = [0.1, 0.1]
        s.H_sparse = h
        return s

    with pytest.raises(ValueError, match=r"c.H_sparse must end with a -1 row terminator"):
        spec_with([0, 1]).validate("c")
    with pytest.raises(ValueError, match=r"c.H_sparse index 2 out of range"):
        spec_with([0, 2, -1]).validate("c")
    with pytest.raises(ValueError, match=r"c.H_sparse index -2 out of range"):
        spec_with([0, -2, -1]).validate("c")
    with pytest.raises(ValueError, match=r"c.H_sparse duplicate column index 1 in a row"):
        spec_with([1, 1, -1]).validate("c")
    spec_with([0, -1, 0, -1]).validate("c")  # (the same column in different rows is fine)
    s = spec_with([0, -1])
    s.error_rates = [0.1, 1.5]
    with pytest.raises(ValueError, match=r"c.error_rates\[1\] out of \[0,1\]"):
        s.validate("c")
    s.error_rates = [0.1, math.nan]
    with pytest.raises(ValueError, match=r"c.error_rates\[1\] out of \[0,1\]"):
        s.validate("c")
    e = qec.SeamSpecEntry()
    e.id = NEXT
    e.spec.H_sparse = [0, -1]
    t = spec_with([0, -1])
    t.seam_specs = [e]
    with pytest.raises(ValueError, match="c: H_sparse and seam_specs are mutually exclusive"):
        t.validate("c")
    t.H_sparse = []
    t.validate("c")
    e.spec.H_sparse = [5, -1]
    t.seam_specs = [e]
    with pytest.raises(ValueError, match=r"c.seam\[next_round\].H_sparse index 5 out of range"):
        t.validate("c")


def test_phase_sequence_is_order_independent():
    spec = rep5_spec(5)
    spec.connections = list(reversed(spec.connections))
    spec.validate()
    seq = spec.phase_sequence()
    assert len(seq) == 5 and seq[0] == qec.phase_name.init and seq[-1] == qec.phase_name.final
    assert seq == [qec.phase_name.init] + [qec.phase_name.bulk] * 3 + [qec.phase_name.final]


def test_per_seam_o_sparse_throws():
    # (A per-seam O_sparse must not be silently discarded.)
    spec = qec.DemChunkSpec()
    spec.num_faults = 3
    spec.error_rates = [0.01] * 3
    e = qec.SeamSpecEntry()
    e.id = NEXT
    e.spec.H_sparse = [0, -1, 1, -1, 2, -1]
    e.spec.O_sparse = [0, -1]
    spec.seam_specs = [e]
    with pytest.raises(ValueError, match="test: per-seam O_sparse in seam 'next_round' is not yet supported; use chunk-level O_sparse instead"):
        qec.dem_chunk_from_spec(spec, [], "test")


def test_seam_specs_form_builds_the_same_chunk_as_the_shorthand():
    h = [0, 1, 5, -1, 1, 2, 6, -1]
    shorthand = qec.DemChunkSpec()
    shorthand.num_faults = 9
    shorthand.H_sparse = h
    shorthand.O_sparse = [0, -1]
    shorthand.error_rates = [0.02] * 9
    full = qec.DemChunkSpec()
    full.num_faults, full.O_sparse, full.error_rates = 9, [0, -1], [0.02] * 9
    entries = []
    for sid in (PREV, NEXT):
        e = qec.SeamSpecEntry()
        e.id = sid
        e.spec.H_sparse = h
        entries.append(e)
    full.seam_specs = entries
    a = qec.dem_chunk_from_spec(shorthand, [PREV, NEXT])
    b = qec.dem_chunk_from_spec(full, [NEXT])  # (the ids are ignored once seam_specs is set)
    expect_same_chunk(a, b, "shorthand vs seam_specs")
    assert a.tags == [0, 1, 0, 1] and a.num_rows() == 4
    # (expand() fills seam_specs from the shorthand, once.)
    shorthand.expand([PREV])
    assert shorthand.H_sparse == [] and len(shorthand.seam_specs) == 1
    shorthand.expand([PREV, NEXT])
    assert len(shorthand.seam_specs) == 1


def test_chunk_from_spec_errors_name_their_context():
    spec = rep5_spec().get_phase(qec.SeamId("init"))
    spec.error_rates = [0.1]
    with pytest.raises(ValueError, match=r"my context.error_rates has 1 entries but num_faults=9"):
        qec.dem_chunk_from_spec(spec, [NEXT], "my context")
    # (A spec that names the same seam twice cannot be built.)
    ok = rep5_spec().get_phase(qec.SeamId("init"))
    with pytest.raises(ValueError, match="extended_dem::add_seam: duplicate seam next_round"):
        qec.dem_chunk_from_spec(ok, [NEXT, NEXT])


# -- streaming decomposition (pinning the structural facts a streaming decoder relies on) -------------------------------------------


def fault_offsets(phases):
    offsets = [0]
    for c in phases:
        offsets.append(offsets[-1] + c.num_faults())
    return offsets


def round_band(phases, rnd):
    """Round r's detector band with the fault columns of the chunks it can reach: its own and its predecessor's."""
    offsets = fault_offsets(phases)
    closed = qec.dem_close_all(phases)
    return dense(closed.detector_error_matrix[rnd * REP5_CHECKS : (rnd + 1) * REP5_CHECKS, offsets[rnd - 1 if rnd else 0] : offsets[rnd + 1]])


def test_bands_touch_only_two_adjacent_chunks():
    for rounds in range(2, 8):
        phases = rep5_phases(rounds)
        offsets = fault_offsets(phases)
        H = qec.dem_close_all(phases).detector_error_matrix
        assert H.shape[0] == rounds * REP5_CHECKS
        for rnd in range(rounds):
            lo, hi = offsets[rnd - 1 if rnd else 0], offsets[rnd + 1]
            for r in range(rnd * REP5_CHECKS, (rnd + 1) * REP5_CHECKS):
                for c in np.flatnonzero(H[r]):
                    assert lo <= c < hi, f"rounds={rounds} round={rnd} row={r} column {c} escapes the window [{lo}, {hi})"


def test_every_bulk_round_poses_one_fixed_problem():
    reference = round_band(rep5_phases(5), 2)
    assert len(reference) == REP5_CHECKS and len(reference[0]) == 18  # (two bulk chunks of 9 faults)
    for rounds in range(4, 10):
        phases = rep5_phases(rounds)
        for rnd in range(2, rounds - 1):  # (rounds 1 and rounds-1 abut init and final)
            assert round_band(phases, rnd) == reference, (rounds, rnd)


def test_boundary_rounds_pose_fixed_problems_too():
    first, entering_bulk, last = round_band(rep5_phases(4), 0), round_band(rep5_phases(4), 1), round_band(rep5_phases(4), 3)
    for rounds in range(4, 10):
        phases = rep5_phases(rounds)
        assert round_band(phases, 0) == first, rounds
        assert round_band(phases, 1) == entering_bulk, rounds
        assert round_band(phases, rounds - 1) == last, rounds


def test_detectors_need_one_round_of_lookback():
    for rounds in range(2, 8):
        d_sparse = qec.dem_chunks_to_d_sparse(rep5_phases(rounds))
        assert len(d_sparse) == rounds * REP5_CHECKS
        for det, bits in enumerate(d_sparse):
            rnd = det // REP5_CHECKS
            earliest, past_end = (rnd - 1 if rnd else 0) * REP5_CHECKS, (rnd + 1) * REP5_CHECKS
            assert all(earliest <= b < past_end for b in bits), (rounds, det)


def step_chunks(phases, step):
    """The chunks a streaming step spans: the one retiring and the one that just arrived."""
    retiring = 0 if step == 0 else step - 1
    arriving = min(step, len(phases) - 1)
    return phases[retiring : arriving + 1]


def test_a_window_is_the_experiment_restricted_to_its_own_chunks():
    # (Stitched, closed on the seam they arrive on, the open one they hand forward dropped: exactly the
    # corresponding block of the whole experiment, priors included. The round count is never mentioned.)
    for rounds in range(3, 9):
        phases = rep5_phases(rounds)
        offsets = fault_offsets(phases)
        whole = qec.dem_close_all(phases)
        for step in range(rounds + 1):
            window = qec.dem_close(qec.dem_stitch_all(step_chunks(phases, step)))
            first_chunk = 0 if step == 0 else step - 1
            past_chunk = min(step, rounds - 1) + 1
            row_begin, row_end = first_chunk * REP5_CHECKS, past_chunk * REP5_CHECKS
            assert window.num_detectors() == row_end - row_begin, (rounds, step)
            assert window.num_error_mechanisms() == offsets[past_chunk] - offsets[first_chunk]
            want = whole.detector_error_matrix[row_begin:row_end, offsets[first_chunk] : offsets[past_chunk]]
            assert np.array_equal(window.detector_error_matrix, want), (rounds, step)
            assert list(window.error_rates) == list(whole.error_rates)[offsets[first_chunk] : offsets[past_chunk]], (rounds, step)


def test_a_window_carries_its_chunks_observables():
    rounds = 5
    phases = rep5_phases(rounds)
    offsets = fault_offsets(phases)
    whole = qec.dem_close_all(phases)
    for step in range(rounds + 1):
        window = qec.dem_close(qec.dem_stitch_all(step_chunks(phases, step)))
        first_chunk = 0 if step == 0 else step - 1
        past_chunk = min(step, rounds - 1) + 1
        assert np.array_equal(window.observables_flips_matrix, whole.observables_flips_matrix[:, offsets[first_chunk] : offsets[past_chunk]]), step


# -- internal consistency of a chunk ---------------------------------------------------------------------------------------------


def test_block_wider_than_the_chunk_throws():
    # (One interior row claiming four fault columns instead of three.)
    c = one_round()
    assert c.num_faults() == 3
    c._H = _Mat.from_rows(1, 4, [[3]])
    c._seams, c._tags = [], []
    with pytest.raises(ValueError, match="test: H has 4 cols but chunk has 3 faults"):
        c.validate("test")
    with pytest.raises(ValueError, match="dem_close: H has 4 cols but chunk has 3 faults"):
        qec.dem_close(c)
    with pytest.raises(ValueError, match="dem_stitch: left: H has 4 cols"):
        qec.dem_stitch(c, c, NEXT, PREV)
    with pytest.raises(ValueError, match="dem_close_all: chunk 0: H has 4 cols"):
        qec.dem_close_all([c])


def test_prior_count_must_match_fault_count():
    c = one_round()
    c.error_rates = c.error_rates[:-1]
    with pytest.raises(ValueError, match="test: H has 3 cols but chunk has 2 faults"):
        c.validate("test")
    with pytest.raises(ValueError):
        qec.dem_close(c)
    with pytest.raises(ValueError):
        qec.dem_stitch_all([c])


def test_tag_count_must_match_seam_rows():
    c = one_round()
    c.tags = c.tags + [99]
    with pytest.raises(ValueError, match=r"test: tags.size\(\)=5 != sum of seam rows 4"):
        c.validate("test")


def test_well_formed_chunks_pass():
    one_round().validate("test")
    # (All-zero rates leave a default-constructed chunk, which is consistent.)
    empty = qec.extended_dem_from_css_matrices(rep3(), qec.CssNoise())
    empty.validate("test")
    assert empty.num_faults() == 0 and empty.num_rows() == 0 and empty.seams == []
    for phase in rep5_phases(4):
        phase.validate("test")


def test_empty_block_must_match_fault_width():
    # (Short tag lists and an H whose width disagrees used to be read past the end by the merge helpers.)
    dem = chunk(_Mat.from_cols(1, 2, [[0], [0]]), _Mat.from_cols(0, 2, [[], []]), [0.1, 0.2], (), [0])
    with pytest.raises(ValueError):
        dem.validate("test")
    for fn in (qec.dem_merge_duplicate_columns, qec.are_dem_columns_unique, qec.assert_dem_columns_unique):
        with pytest.raises(ValueError):
            fn(dem)
    with pytest.raises(ValueError):
        qec.dem_chunks_to_o_sparse([dem])


def test_merge_rejects_short_prior_list():
    c = one_round()
    c.error_rates = c.error_rates[:-1]
    for fn in (qec.dem_merge_duplicate_columns, qec.are_dem_columns_unique, qec.assert_dem_columns_unique):
        with pytest.raises(ValueError):
            fn(c)
    with pytest.raises(ValueError):
        qec.dem_chunks_to_o_sparse([c])


def test_overlapping_seam_bands_throw():
    # (num_interior_rows() subtracts the summed seam widths while the interior rows are materialised by row membership.)
    dem = chunk(_Mat.from_rows(4, 2, [[0], [1], [0], [1]]), _Mat.from_rows(0, 2, []), [0.01, 0.01], [(PREV, 0, 3), (NEXT, 2, 4)], [0] * 5)
    with pytest.raises(ValueError, match=r"overlap: seam row bands \[0,3\) and \[2,4\) overlap"):
        dem.validate("overlap")


def test_default_constructed_observables_are_handled():
    # (A chunk with no observables may leave O default-constructed, 0 x 0: it cannot be indexed per fault column.)
    dem = chunk(_Mat.from_rows(2, 3, [[0], [1]]), _Mat(), [0.1, 0.1, 0.1], [(PREV, 0, 1), (NEXT, 1, 2)], [0, 0])
    assert (dem._O.nrows, dem._O.ncols) == (0, 0)
    dem.validate("no observables")
    assert qec.dem_chunks_to_o_sparse([dem, dem]) == []
    qec.dem_merge_duplicate_columns(dem)
    qec.are_dem_columns_unique(dem)


def test_misaligned_observables_throw():
    dem = chunk(_Mat.from_rows(2, 3, [[0], [1]]), _Mat.from_rows(0, 2, []), [0.1, 0.1, 0.1], [(PREV, 0, 1), (NEXT, 1, 2)], [0, 0])
    with pytest.raises(ValueError, match="misaligned O: O has 2 cols but chunk has 3 faults"):
        dem.validate("misaligned O")


def test_seam_validation_messages():
    dem = chunk(_Mat.from_rows(2, 2, [[0], [1]]), _Mat.from_rows(0, 2, []), [0.1, 0.1], [(PREV, 0, 5)], [0] * 5)
    with pytest.raises(ValueError, match=r"t: seam row_end 5 > H.num_rows\(\) 2"):
        dem.validate("t")
    dem = chunk(_Mat.from_rows(2, 2, [[0], [1]]), _Mat.from_rows(0, 2, []), [0.1, 0.1], [(PREV, 1, 0)], [])
    with pytest.raises(ValueError, match="t: seam has row_begin > row_end"):
        dem.validate("t")
    c = one_round()
    with pytest.raises(IndexError, match="extended_dem::get_seam: no seam with id seam:"):
        c.get_seam(qec.SeamId("nothing"))
    with pytest.raises(TypeError):
        qec.ExtendedDemSeam()
    with pytest.raises(ValueError, match="extended_dem::add_seam: duplicate seam prev_round"):
        c._add_seam(PREV, 0, 0)


def test_chunk_accessors_hand_back_copies():
    c = one_round()
    H = c.H
    assert isinstance(H, np.ndarray) and H.dtype == np.uint8
    H[0, 0] ^= 1
    assert not np.array_equal(H, c.H)
    rates = c.error_rates
    rates[0] = 0.5
    assert c.error_rates[0] == 0.01
    c.error_rates = [0.1, 0.2, 0.3]
    assert c.error_rates == [0.1, 0.2, 0.3]
    assert [(s.id, s.row_begin, s.row_end, s.num_rows()) for s in c.seams] == [(PREV, 0, 2, 2), (NEXT, 2, 4, 2)]
    with pytest.raises(TypeError):
        c.tags = [-1]
    with pytest.raises(AttributeError):
        c.H = H
    assert c.num_rows() == 4 and c.num_observables() == 1


# -- seams outside the contracted pair ---------------------------------------------------------------------------------------------------

SIDE = qec.SeamId("side")


def chunk_with_side_seam(side_rows):
    """prev_round [0,1), next_round [1,2) and a third named seam [2, 2+side_rows), each row touching its own fault column
    so that a dropped row is visible."""
    n_rows = 2 + side_rows
    return chunk(
        _Mat.from_rows(n_rows, n_rows, [[r] for r in range(n_rows)]),
        _Mat.from_rows(1, n_rows, [[0]]),
        [0.01] * n_rows,
        [(PREV, 0, 1), (NEXT, 1, 2), (SIDE, 2, 2 + side_rows)],
        [0] * (2 + side_rows),
    )


def test_stitch_rejects_a_non_contracted_seam():
    # (A seam that is neither from_seam nor to_seam has no counterpart to contract against, and its rows are excluded from the
    # interior, so composing used to drop them from H without any error.)
    a = chunk_with_side_seam(1)
    a.validate("side-seam chunk")
    with pytest.raises(ValueError, match=r"dem_stitch: left: seam (?:seam:[0-9a-f]{8}|side) carries 1 rows but is neither from_seam \(next_round\) nor to_seam \(prev_round\); composing chunks that carry additional named seams is not supported"):
        qec.dem_stitch(a, a, NEXT, PREV)


def test_close_all_rejects_a_non_contracted_seam():
    a = chunk_with_side_seam(1)
    with pytest.raises(ValueError, match="dem_close_all: chunk 0: seam (?:seam:[0-9a-f]{8}|side) carries 1 rows but is neither"):
        qec.dem_close_all([a, a])
    with pytest.raises(ValueError, match="dem_stitch_all: chunk 0: seam"):
        qec.dem_stitch_all([a, a])
    with pytest.raises(ValueError, match="dem_chunks_to_pcm: chunk 0: seam"):
        qec.dem_chunks_to_pcm([a, a])


def test_close_rejects_a_non_contracted_seam():
    # (dem_close drops from_seam on purpose, no next round to compare against, but a third seam must not disappear the same way.)
    with pytest.raises(ValueError, match="dem_close: seam (?:seam:[0-9a-f]{8}|side) carries 1 rows"):
        qec.dem_close(chunk_with_side_seam(1))


def test_zero_width_non_contracted_seam_is_accepted():
    # (Zero-row seams are structural placeholders, not carriers of detector rows, so they must still compose.)
    a = chunk_with_side_seam(0)
    assert a.has_seam(SIDE)
    qec.dem_stitch(a, a, NEXT, PREV)
    qec.dem_close_all([a, a])
    qec.dem_close(a)


def test_seam_names_in_messages_are_the_registered_ones():
    _register_name(SIDE, "side")
    with pytest.raises(ValueError, match=r"dem_close: seam side carries 1 rows but is neither from_seam \(next_round\) nor to_seam \(prev_round\)"):
        qec.dem_close(chunk_with_side_seam(1))


# -- noise rate validation -----------------------------------------------------------------------------------------------------------


def test_rejects_rates_outside_the_unit_interval():
    # (A negative or NaN rate is inactive under the "rate > 0" rule that selects fault columns, so a mistyped rate would quietly build a smaller DEM.)
    for bad in (-0.01, 1.5, math.nan, math.inf):
        noise = qec.CssNoise()
        noise.px = bad
        with pytest.raises(ValueError, match=r"px must be a probability in \[0, 1\], got "):
            qec.extended_dem_from_css_matrices(rep3(), noise)
        with pytest.raises(ValueError, match=r"px must be a probability in \[0, 1\], got "):
            qec.dem_from_css_matrices(rep3(), noise)
    noise = qec.CssNoise()
    noise.pm = -0.5
    with pytest.raises(ValueError, match=r"pm must be a probability in \[0, 1\], got -0.500000"):
        qec.dem_from_css_matrices(rep3(), noise)


def test_rejects_bad_per_element_rates():
    per_qubit = qec.CssNoise()
    per_qubit.px_per_qubit = [0.01, -0.01, 0.01]
    with pytest.raises(ValueError, match=r"px_per_qubit\[1\] must be a probability"):
        qec.extended_dem_from_css_matrices(rep3(), per_qubit)
    per_check = qec.CssNoise()
    per_check.pm_per_check = [0.01, 2.0]
    with pytest.raises(ValueError, match=r"pm_per_check\[1\] must be a probability"):
        qec.extended_dem_from_css_matrices(rep3(), per_check)


def test_accepts_the_endpoints_of_the_unit_interval():
    certain = qec.CssNoise()
    certain.px = 1.0
    qec.extended_dem_from_css_matrices(rep3(), certain)
    qec.extended_dem_from_css_matrices(rep3(), qec.CssNoise())


def test_per_check_length_error_names_the_check_count():
    noise = qec.CssNoise()
    noise.pm_per_check = [0.01]  # (rep3 has 2 checks)
    with pytest.raises(ValueError) as e:
        qec.extended_dem_from_css_matrices(rep3(), noise)
    assert "n_checks" in str(e.value) and "n_qubits" not in str(e.value)
    assert str(e.value) == "pm_per_check has 1 entries but n_checks=2"


# -- dem_from_css_matrices (test_dem_construction.cpp) ---------------------------------------------------------------------------------


def test_construction_rep3_x_only():
    dem = qec.dem_from_css_matrices(rep3(), px_only(0.01))
    assert (dem.num_detectors(), dem.num_error_mechanisms(), dem.num_observables()) == (2, 3, 1)
    assert dense(dem.detector_error_matrix) == [[1, 1, 0], [0, 1, 1]]
    assert dense(dem.observables_flips_matrix) == [[1, 0, 0]]
    assert dem.error_rates == [0.01] * 3


def test_construction_matches_stim_text():
    # (X on q0 flips D0 and L0, on q1 D0 and D1, on q2 D1.)
    dem = qec.dem_from_css_matrices(rep3(), px_only(0.01))
    stim = qec.dem_from_stim_text("error(0.01) D0 L0\nerror(0.01) D0 D1\nerror(0.01) D1\n")
    assert same_dem(dem, stim)
    t2 = qec.dem_from_css_matrices(rep3(), px_only(0.01), 2)
    stim2 = qec.dem_from_stim_text("error(0.01) D0 D2 L0\nerror(0.01) D0 D1 D2 D3\nerror(0.01) D1 D3\nerror(0.01) D2 L0\nerror(0.01) D2 D3\nerror(0.01) D3\n")
    assert same_dem(t2, stim2)


def test_construction_x_and_z_errors():
    code = rep3()
    code.lx = np.array([[1, 0, 0]], dtype=np.uint8)
    noise = qec.CssNoise()
    noise.px = noise.pz = 0.01
    dem = qec.dem_from_css_matrices(code, noise)
    assert (dem.num_detectors(), dem.num_observables(), dem.num_error_mechanisms()) == (2, 2, 6)
    # (Z fault columns start at index 3: no detectors, the X observable (row 1) flipped by qubit 0.)
    assert not dem.detector_error_matrix[:, 3:].any()
    assert dense(dem.observables_flips_matrix[:, 3:]) == [[0, 0, 0], [1, 0, 0]]


def test_construction_y_errors_trigger_both_bands():
    code = qec.CssCodes()
    code.hz = np.array([[1, 1]], dtype=np.uint8)
    code.hx = np.array([[1, 1]], dtype=np.uint8)
    code.lz = np.array([[1, 0]], dtype=np.uint8)
    code.lx = np.array([[1, 0]], dtype=np.uint8)
    noise = qec.CssNoise()
    noise.py = 0.005
    dem = qec.dem_from_css_matrices(code, noise)
    assert (dem.num_detectors(), dem.num_observables(), dem.num_error_mechanisms()) == (2, 2, 2)
    assert dem.error_rates == [0.005, 0.005]
    assert dense(dem.detector_error_matrix) == [[1, 1], [1, 1]]
    assert dense(dem.observables_flips_matrix) == [[1, 0], [1, 0]]


def test_construction_edge_cases():
    assert qec.dem_from_css_matrices(rep3(), qec.CssNoise()).num_error_mechanisms() == 0
    none = qec.dem_from_css_matrices(qec.CssCodes(), px_only(0.01))
    assert none.num_error_mechanisms() == 0
    # (A default hx and lx: Z faults are undetectable.)
    noise = qec.CssNoise()
    noise.pz = 0.01
    dem = qec.dem_from_css_matrices(rep3(), noise)
    assert (dem.num_detectors(), dem.num_observables(), dem.num_error_mechanisms()) == (2, 1, 3)
    assert not dem.detector_error_matrix.any()
    bad = qec.CssCodes()
    bad.hz = np.ones((1, 3), dtype=np.uint8)
    bad.hx = np.ones((1, 4), dtype=np.uint8)
    both = qec.CssNoise()
    both.px = both.pz = 0.01
    with pytest.raises(ValueError, match=r"hx num_cols \(4\) does not match n_qubits \(3\)"):
        qec.dem_from_css_matrices(bad, both)


def test_construction_per_qubit_rates():
    noise = qec.CssNoise()
    noise.px_per_qubit = [0.01, 0.0, 0.02]
    dem = qec.dem_from_css_matrices(rep3(), noise)
    assert dem.num_error_mechanisms() == 2 and dem.error_rates == [0.01, 0.02]
    assert dense(dem.detector_error_matrix) == [[1, 0], [0, 1]]
    assert dense(dem.observables_flips_matrix) == [[1, 0]]
    noise = qec.CssNoise()
    noise.px = 0.01
    noise.px_per_qubit = [0.05, 0.10, 0.0]  # (the uniform rate is ignored; q2 gets rate 0 so no column)
    dem = qec.dem_from_css_matrices(rep3(), noise)
    assert dem.error_rates == [0.05, 0.10]
    noise = qec.CssNoise()
    noise.px_per_qubit = [0.01, 0.02]
    with pytest.raises(ValueError, match=r"px_per_qubit has 2 entries but n_qubits=3"):
        qec.dem_from_css_matrices(rep3(), noise)
    noise = qec.CssNoise()
    noise.px = 0.01
    noise.px_per_qubit = [0.0, 0.0, 0.0]
    assert qec.dem_from_css_matrices(rep3(), noise).num_error_mechanisms() == 0


def test_construction_multi_round():
    dem = qec.dem_from_css_matrices(rep3(), px_only(0.01), 2)
    assert (dem.num_detectors(), dem.num_observables(), dem.num_error_mechanisms()) == (4, 1, 6)
    # (Round 0, q0 spans D0 and D2; round 1 (the last), q0 only D2.)
    assert dense(dem.detector_error_matrix[:, 0]) == [1, 0, 1, 0]
    assert dense(dem.detector_error_matrix[:, 3]) == [0, 0, 1, 0]
    assert dense(dem.observables_flips_matrix[:, [0, 3]]) == [[1, 1]]
    t3 = qec.dem_from_css_matrices(rep3(), px_only(0.01), 3)
    assert t3.num_detectors() == 6 and t3.num_error_mechanisms() == 9
    assert dense(t3.observables_flips_matrix) == [[1, 0, 0] * 3]
    assert dense(t3.detector_error_matrix[:, 3]) == [0, 0, 1, 0, 1, 0]
    with pytest.raises(ValueError, match="num_rounds must be >= 1"):
        qec.dem_from_css_matrices(rep3(), px_only(0.01), 0)


def test_construction_measurement_errors():
    noise = qec.CssNoise()
    noise.pm = 0.005
    dem = qec.dem_from_css_matrices(rep3(), noise)
    assert dense(dem.detector_error_matrix) == [[1, 0], [0, 1]] and dem.error_rates == [0.005, 0.005]
    assert not dem.observables_flips_matrix.any()
    t2 = qec.dem_from_css_matrices(rep3(), noise, 2)
    assert dense(t2.detector_error_matrix) == [[1, 0, 0, 0], [0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1]]
    both = qec.CssNoise()
    both.px, both.pm = 0.01, 0.005
    stim = qec.dem_from_stim_text(
        "error(0.01)  D0 D2 L0\nerror(0.01)  D0 D1 D2 D3\nerror(0.01)  D1 D3\nerror(0.005) D0 D2\nerror(0.005) D1 D3\n"
        "error(0.01)  D2 L0\nerror(0.01)  D2 D3\nerror(0.01)  D3\nerror(0.005) D2\nerror(0.005) D3\n"
    )
    assert same_dem(qec.dem_from_css_matrices(rep3(), both, 2), stim)


def test_construction_zero_rates_dem_has_the_right_shape():
    flat = qec.dem_from_css_matrices(rep3(), qec.CssNoise(), 3)
    assert flat.detector_error_matrix.shape == (6, 0)  # (T*d detectors, no fault columns)
    assert flat.observables_flips_matrix.shape == (1, 0)
    assert flat.error_rates == []


def test_construction_rows_but_zero_columns_throws():
    code = qec.CssCodes()
    code.hz = np.zeros((2, 0), dtype=np.uint8)
    with pytest.raises(ValueError, match="hz has 2 rows but zero columns; all matrices must be empty when num_qubits cannot be determined"):
        qec.dem_from_css_matrices(code, px_only(0.01), 1)
    with pytest.raises(ValueError, match="hz has 2 rows but zero columns"):
        qec.extended_dem_from_css_matrices(code, px_only(0.01))
    per = qec.CssNoise()
    per.px_per_qubit = [0.1]
    with pytest.raises(ValueError, match="per-qubit noise vectors are non-empty but all code matrices have zero columns"):
        qec.dem_from_css_matrices(qec.CssCodes(), per)


def test_construction_malformed_hz_with_n_from_hx_throws():
    code = qec.CssCodes()
    code.hz = np.zeros((2, 0), dtype=np.uint8)
    code.hx = np.ones((1, 3), dtype=np.uint8)
    noise = qec.CssNoise()
    noise.pz = 0.01
    with pytest.raises(ValueError):
        qec.dem_from_css_matrices(code, noise)


# -- the code overload (test_dem_construction_code.cpp) --------------------------------------------------------------------------------------------


def test_css_matrices_from_code_repetition3():
    m = qec.css_matrices_from_code(qec.get_code("repetition", distance=3))
    assert dense(m.hz) == [[1, 1, 0], [0, 1, 1]]
    assert m.hx.shape[0] == 0
    assert dense(m.lz) == [[1, 0, 0]]
    with pytest.raises(TypeError):
        qec.css_matrices_from_code("repetition")


def test_css_matrices_from_code_repetition_x_part_is_0_by_0_as_cuda_q_gives_it():
    # (The repetition code has no X stabilizers or X logicals: CUDA-Q's to_parity_matrix returns a rank-0 tensor for each,
    # which tensor_to_sparse reads as a 0 x 0 matrix, not an empty one of n columns; the Z part is unaffected.)
    m = qec.css_matrices_from_code(qec.get_code("repetition", distance=3))
    assert m.hx.shape == (0, 0)
    assert m.lx.shape == (0, 0)
    assert m.hz.shape == (2, 3)
    assert m.lz.shape == (1, 3)
    dem = qec.dem_from_css_matrices(m, px_only(0.01))
    assert dem.num_observables() == 1
    assert dem.num_detectors() == 2
    assert qec.dem_from_css_matrices(qec.get_code("repetition", distance=3), px_only(0.01)).num_observables() == 1


def test_code_overload_matches_the_matrix_overload():
    code = qec.get_code("repetition", distance=3)
    noise = px_only(0.01)
    assert same_dem(qec.dem_from_css_matrices(code, noise), qec.dem_from_css_matrices(qec.css_matrices_from_code(code), noise))
    with pytest.raises(TypeError):
        qec.dem_from_css_matrices(object(), noise)


def test_steane_chunk_closes_to_the_monolithic_dem():
    code = qec.css_matrices_from_code(qec.get_code("steane"))
    noise = qec.CssNoise()
    noise.px, noise.pz, noise.py, noise.pm = 0.01, 0.02, 0.003, 0.005
    for T in (1, 2, 4):
        chunks = [qec.extended_dem_from_css_matrices(code, noise)] * T
        flat = qec.dem_from_css_matrices(code, noise, T)
        assert same_dem(qec.dem_close_all(chunks), flat), T
        assert same_dem(qec.dem_close(qec.dem_stitch_all(chunks)), flat), T
        assert qec.dem_chunks_to_rounds(chunks) == T
        assert [r for r in qec.dem_chunks_to_o_sparse(chunks)] == [list(np.flatnonzero(row)) for row in flat.observables_flips_matrix]


def test_steane_chunk_pcm_is_the_monolithic_detector_matrix():
    pytest.importorskip("scipy.sparse")  # (dem_chunks_to_pcm returns a scipy matrix)
    code = qec.css_matrices_from_code(qec.get_code("steane"))
    noise = qec.CssNoise()
    noise.px, noise.pz, noise.py, noise.pm = 0.01, 0.02, 0.003, 0.005
    for T in (1, 2, 4):
        chunks = [qec.extended_dem_from_css_matrices(code, noise)] * T
        flat = qec.dem_from_css_matrices(code, noise, T)
        pcm = qec.dem_chunks_to_pcm(chunks)
        assert np.array_equal(pcm.toarray().astype(np.uint8), flat.detector_error_matrix), T


# -- the closed model against this package's circuit DEM (added) -----------------------------------------------------------------------------------------------


def phenomenological_circuit(rounds, px, pm):
    """The circuit whose detector error model is the closed chunk model of the d=3 repetition code: each round resets the
    two check ancillas, flips the checks a data fault of qubit q triggers (a correlated error, the observable qubit
    flipped with the Z0 logical's fault), flips a check on its own with pm, and measures the checks. A fault flips the
    syndrome of its own round only (the ancillas are reset), the detectors being the rounds' differences."""
    import stabilizer_qec as sq

    lines = []
    for r in range(rounds):
        lines.append("R 3 4")
        lines.append(f"CORRELATED_ERROR({px}) X3 X5")  # (qubit 0: check Z0Z1, the logical)
        lines.append(f"CORRELATED_ERROR({px}) X3 X4")  # (qubit 1: both checks)
        lines.append(f"CORRELATED_ERROR({px}) X4")  # (qubit 2: check Z1Z2)
        if pm:
            lines.append(f"X_ERROR({pm}) 3 4")
        lines.append("M 3 4")
        lines.append("DETECTOR rec[-2]" + (" rec[-4]" if r else ""))
        lines.append("DETECTOR rec[-1]" + (" rec[-3]" if r else ""))
    lines.append("M 5")
    lines.append("OBSERVABLE_INCLUDE(0) rec[-1]")
    return sq.Circuit("\n".join(lines))


def merged_mechanisms(dem):
    """Each distinct (detectors, observables) with the XOR-combined probability of the mechanisms having it."""
    out = {}
    H, O = np.asarray(dem.detector_error_matrix), np.asarray(dem.observables_flips_matrix)
    for j, p in enumerate(dem.error_rates):
        key = (tuple(np.flatnonzero(H[:, j])), tuple(np.flatnonzero(O[:, j])))
        out[key] = out.get(key, 0.0) + p - 2.0 * out.get(key, 0.0) * p
    return out


@pytest.mark.parametrize("T", [3, 5])
@pytest.mark.parametrize("pm", [0.0, 0.004])
def test_close_matches_our_circuit_dem(T, pm):
    px = 0.01
    noise = qec.CssNoise()
    noise.px, noise.pm = px, pm
    chunks = [qec.extended_dem_from_css_matrices(rep3(), noise)] * T
    closed = qec.dem_close_all(chunks)
    # Against dem_from_css_matrices, exactly:
    assert same_dem(closed, qec.dem_from_css_matrices(rep3(), noise, T))
    assert same_dem(qec.dem_close(qec.dem_stitch_all(chunks)), closed)
    # Against the DEM this package derives from the circuit with those faults (as mechanisms, the
    # order of the circuit's errors being its own):
    circuit_dem = qec.dem_from_stim_text(str(phenomenological_circuit(T, px, pm).detector_error_model()))
    assert circuit_dem.num_detectors() == closed.num_detectors() == T * 2
    ours, theirs = merged_mechanisms(closed), merged_mechanisms(circuit_dem)
    assert set(ours) == set(theirs)
    for key in ours:
        # (Bit-identical where the references are recorded, macOS; no platform is known to need a tolerance.)
        assert ours[key] == theirs[key], key

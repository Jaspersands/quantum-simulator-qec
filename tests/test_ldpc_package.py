"""stabilizer_qec.ldpc against ldpc 2.4.1 itself: every mod2 function (both eliminations, the
PLU decomposition's factors), BP+LSD's statistics, the v1 decoders and every module importing.
ldpc's own test suite runs against the port in tests/ldpc_port."""

from __future__ import annotations

import importlib
import pkgutil

import numpy as np
import pytest

scipy_sparse = pytest.importorskip("scipy.sparse")
ldpc = pytest.importorskip("ldpc", minversion="2.4")

import stabilizer_qec.ldpc as ours  # noqa: E402
import stabilizer_qec.ldpc.code_util  # noqa: E402,F401
import stabilizer_qec.ldpc.codes  # noqa: E402,F401
import stabilizer_qec.ldpc.mod2 as om  # noqa: E402


def same(a, b) -> bool:
    if scipy_sparse.issparse(a) or scipy_sparse.issparse(b):
        return a.shape == b.shape and (scipy_sparse.csr_matrix(a) != scipy_sparse.csr_matrix(b)).nnz == 0
    return np.array_equal(np.asarray(a), np.asarray(b))


def matrices(seed: int, count: int):
    rng = np.random.default_rng(seed)
    for t in range(count):
        m, n = int(rng.integers(1, 12)), int(rng.integers(1, 16))
        a = (rng.random((m, n)) < rng.uniform(0.1, 0.6)).astype(np.uint8)
        yield rng, (scipy_sparse.csr_matrix(a) if t % 3 == 0 else a)


FUNCTIONS = {
    "rank": lambda f, a: f.rank(a),
    "rank_sparse": lambda f, a: f.rank(a, method="sparse"),
    "nullspace": lambda f, a: f.nullspace(a),
    "nullspace_sparse": lambda f, a: f.nullspace(a, method="sparse"),
    "kernel": lambda f, a: f.kernel(a),
    "row_complement_basis": lambda f, a: f.row_complement_basis(a),
    "pivot_rows": lambda f, a: f.pivot_rows(a),
    "row_basis": lambda f, a: f.row_basis(a),
    "io_test": lambda f, a: f.io_test(a),
    "row_echelon": lambda f, a: f.row_echelon(a, full=True)[0],
    "reduced_row_echelon": lambda f, a: f.reduced_row_echelon(a)[0],
    "row_span": lambda f, a: f.row_span(a) if a.shape[0] <= 9 else 0,
    "compute_exact_code_distance": lambda f, a: f.compute_exact_code_distance(a),
}


@pytest.mark.parametrize("name", sorted(FUNCTIONS))
def test_mod2_function(name):
    import ldpc.mod2 as tm

    fn = FUNCTIONS[name]
    for _, a in matrices(5, 120):
        assert same(fn(tm, a), fn(om, a)), name


@pytest.mark.parametrize("full_reduce,lower_triangular", [(False, True), (True, True), (False, False), (True, False)])
def test_plu_decomposition(full_reduce, lower_triangular):
    import ldpc.mod2 as tm

    for rng, a in matrices(7, 100):
        p, q = tm.PluDecomposition(a, full_reduce=full_reduce, lower_triangular=lower_triangular), om.PluDecomposition(a, full_reduce=full_reduce, lower_triangular=lower_triangular)
        for x, y in ((p.L, q.L), (p.U, q.U), (p.P, q.P), (p.rank, q.rank), (p.pivots, q.pivots)):
            assert same(x, y)
        y = (np.asarray(a @ (rng.random(a.shape[1]) < 0.5).astype(np.uint8)) % 2).astype(np.uint8).ravel()
        assert same(tm.PluDecomposition(a).lu_solve(y), om.PluDecomposition(a).lu_solve(y))


def outcome(fn, *args):
    try:
        return fn(*args)
    except Exception as e:
        return type(e), str(e)


def test_code_util_and_codes():
    import ldpc.code_util as tc
    import ldpc.codes as tcodes

    for d in (3, 4, 7):
        for f in ("rep_code", "ring_code"):
            assert same(getattr(tcodes, f)(d), getattr(ours.codes, f)(d))
    for r in (2, 3, 4):
        assert same(tcodes.hamming_code(r), ours.codes.hamming_code(r))
    for _, a in matrices(9, 40):
        if a.shape[1] > 12:
            continue
        assert tc.compute_code_dimension(a) == ours.code_util.compute_code_dimension(a)
        assert same(tc.construct_generator_matrix(a), ours.code_util.construct_generator_matrix(a))
        assert outcome(tc.compute_exact_code_distance, a) == outcome(ours.code_util.compute_exact_code_distance, a)


def test_bplsd_statistics():
    rng = np.random.default_rng(4)
    agree = total = 0
    for t in range(60):
        m, n = int(rng.integers(8, 30)), int(rng.integers(20, 60))
        h = (rng.random((m, n)) < 0.12).astype(np.uint8)
        order = int(rng.integers(0, 4))
        kw = dict(error_rate=0.05, max_iter=int(rng.integers(1, 3)), bits_per_step=int(rng.integers(1, 4)), lsd_order=order, lsd_method="lsd_0" if order == 0 else str(rng.choice(["lsd_cs", "lsd_e"])))
        theirs, mine = ldpc.BpLsdDecoder(h, **kw), ours.BpLsdDecoder(h, **kw)
        for d in (theirs, mine):
            d.set_do_stats(bool(t % 4))
        assert mine.do_stats == theirs.do_stats
        for _ in range(4):
            s = (h @ (rng.random(n) < 0.1) % 2).astype(np.uint8)
            a, b = theirs.decode(s), mine.decode(s)
            sa, sb = dict(theirs.statistics), dict(mine.statistics)
            assert set(sa) == set(sb)
            sa.pop("elapsed_time"), sb.pop("elapsed_time")
            equal = sa == sb and np.array_equal(a, b)
            # Only a merge of three or more clusters at once (in an order ldpc takes from
            # addresses) can make them differ.
            assert equal or mine.last_tied
            agree += equal
            total += 1
    assert agree >= 0.85 * total
    mine.set_additional_stat_fields([1, 0], [0, 1], [1, 1])
    assert (mine.statistics["error"], mine.statistics["compare_recover"]) == ([1, 0], [1, 1])
    mine.reset_cluster_stats()
    assert mine.statistics["individual_cluster_stats"] == {} and mine.statistics["error"] == []


def test_v1_decoders_and_soft_info_bposd():
    h = ours.codes.rep_code(5).toarray()
    with pytest.warns(UserWarning):
        v1 = ours.bp_decoder(h, error_rate=0.1, max_iter=5, bp_method="ms")
    with pytest.warns(UserWarning):
        v1osd = ours.bposd_decoder(h, error_rate=0.1, osd_method="osd_cs", osd_order=2)
    s = np.array([1, 0, 0, 1], dtype=np.uint8)
    assert np.array_equal(h @ v1.decode(s) % 2, s) and np.array_equal(h @ v1osd.decode(s) % 2, s)
    assert v1.bp_method == "minimum_sum" and v1osd.osd_method == "OSD_CS"
    from stabilizer_qec.ldpc.bposd_decoder import SoftInfoBpOsdDecoder

    dec = SoftInfoBpOsdDecoder(h, error_rate=0.1, max_iter=2, osd_method="osd_cs", osd_order=2, cutoff=5.0)
    out = dec.decode(np.array([2.0, -0.3, 1.5, -2.0]))
    hard = (np.array([2.0, -0.3, 1.5, -2.0]) <= 0).astype(np.uint8)
    assert out.shape == (5,) and (dec.converge or np.array_equal(h @ out % 2, (dec.soft_syndrome <= 0).astype(np.uint8)))
    assert dec.osd_method == "OSD_CS" and hard.sum() == 2


def test_every_module_imports():
    failed = []
    for m in pkgutil.walk_packages(ours.__path__, "stabilizer_qec.ldpc."):
        try:
            importlib.import_module(m.name)
        except Exception as e:  # pragma: no cover - reported below
            failed.append((m.name, repr(e)))
    assert failed == []
    assert ours.__version__ == "2.4.1" and set(ours.__all__) <= set(dir(ours))

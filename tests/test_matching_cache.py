"""Matching cache/facade regressions. Rust user_graph tests prove overlapping native ownership;
these check public predictions, failures and serialization without depending on pool internals.
"""
import copy
import pickle
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

import stabilizer_qec as sq


SQUARE_DEM = "error(0.1) D0 D1 L0\nerror(0.1) D1 D2\nerror(0.1) D2 D3 L1\nerror(0.1) D3 D0"
SYNDROME = [1, 0, 1, 0]


def square():
    m = sq.Matching()
    for (u, v), f in zip(((0, 1), (1, 2), (2, 3), (3, 0)), ({0}, set(), {1}, set())):
        m.add_edge(u, v, fault_ids=f, weight=1, error_probability=0.1)
    return m


@pytest.mark.parametrize("how", ["keep-original", "smallest-weight"])
def test_noop_preserves_stale_ties_and_counts(how):
    m = square()
    assert m.decode(SYNDROME).tolist() == [1, 0]
    m.ensure_num_fault_ids(70)
    assert m.add_edge(0, 1, fault_ids={99}, weight=10, merge_strategy=how) is None
    assert m.num_fault_ids == 70
    assert m.get_edge_data(0, 1)["fault_ids"] == {0}
    assert m.add_noise() is not None
    for threads in [1, 3]:
        pred = m.decode_batch(np.tile(SYNDROME, (12, 1)), threads=threads)
        assert pred.tolist() == [[1, 0] + [0] * 68] * 12
    m.decode_to_edges_array(SYNDROME)
    assert m.decode(SYNDROME).tolist() == [0, 1] + [0] * 68


@pytest.mark.parametrize("how", ["keep-original", "smallest-weight"])
@pytest.mark.parametrize("boundary", [False, True])
def test_noop_preserves_correlations(how, boundary):
    text = "error(0.1) D0 L0" if boundary else SQUARE_DEM
    m = sq.Matching(text, enable_correlations=True)
    syndrome = [1] if boundary else SYNDROME
    before = m.decode(syndrome, enable_correlations=True)
    if boundary:
        assert m.add_boundary_edge(0, weight=10, merge_strategy=how) is None
    else:
        assert m.add_edge(0, 1, weight=10, merge_strategy=how) is None
    assert np.array_equal(m.decode(syndrome, enable_correlations=True), before)
    # Actual replacement still invalidates the separately constructed correlated matcher.
    if boundary:
        m.add_boundary_edge(0, weight=1, merge_strategy="replace")
    else:
        m.add_edge(0, 1, weight=1, merge_strategy="replace")
    with pytest.raises(ValueError, match="not been configured"):
        m.decode(syndrome, enable_correlations=True)
    restored = pickle.loads(pickle.dumps(m))
    assert np.array_equal(restored.decode(syndrome), m.decode(syndrome))
    with pytest.raises(ValueError, match="not been configured"):
        restored.decode(syndrome, enable_correlations=True)


@pytest.mark.parametrize("built", [False, True])
def test_capacity_error_is_catchable_and_object_remains_usable(built):
    m = square()
    if built:
        m.decode(SYNDROME)
    m.ensure_num_fault_ids(2**63)
    for _ in range(2):
        with pytest.raises(MemoryError, match="matching output"):
            m.decode(SYNDROME)
        assert m.decode_to_edges_array(SYNDROME).shape == (2, 2)
        m.add_boundary_edge(0, weight=1, merge_strategy="replace")
        assert m.has_boundary_edge(0)
    assert m.num_fault_ids == 2**63


def test_failure_discards_scratch_but_retains_stale_choice():
    m = square()
    m.decode(SYNDROME)
    m.ensure_num_fault_ids(70)
    for bad in [np.array([[1]], dtype=np.uint8), np.array([[0x85]], dtype=np.uint8)]:
        with pytest.raises(ValueError):
            m.decode_batch(bad, bit_packed_shots=True)
        assert m.decode(SYNDROME).tolist() == [1, 0] + [0] * 68


@pytest.mark.parametrize("value, expected", [(0.5, 0), (256, 0), (-256, 0), (2, 1)])
def test_batch_casts_to_uint8_before_nonzero(value, expected):
    m = sq.Matching("error(0.1) D0 L0")
    # ndarray casts wrap in NumPy 1.x and 2.x; direct Python-int casts can raise in 2.x.
    shots = np.array([[value]])
    assert m.decode(shots[0]).tolist() == [expected]
    assert m.decode_batch(shots).tolist() == [[expected]]
    assert m.decode_batch(shots, bit_packed_predictions=True).tolist() == [[expected]]


@pytest.mark.parametrize("width", [64, 65, 70])
@pytest.mark.parametrize("shots", [0, 3])
@pytest.mark.parametrize("packed", [False, True])
def test_correlated_padding_and_roundtrip(width, shots, packed):
    m = sq.Matching("error(0.1) D0 L0", enable_correlations=True)
    m.ensure_num_fault_ids(width)
    m.add_boundary_edge(0, weight=10, merge_strategy="keep-original")
    for same in [m, copy.deepcopy(m), pickle.loads(pickle.dumps(m))]:
        assert same.num_fault_ids == width
        out = same.decode_batch(np.ones((shots, 1), dtype=np.uint8), enable_correlations=True, bit_packed_predictions=packed)
        assert out.shape == (shots, (width + 7) // 8 if packed else width)
        assert np.all(out[:, 0] == 1)
        assert np.all(out[:, 1:] == 0)
        assert same.decode([1], enable_correlations=True).shape == (width,)


def test_nan_probability_readback_noise_and_independent_merge():
    m = sq.Matching()
    m.add_boundary_edge(0, fault_ids={0}, error_probability=float("nan"))
    assert np.isnan(m.edges()[0][2]["error_probability"])
    assert np.isnan(m.get_boundary_edge_data(0)["error_probability"])
    noise, syndrome = m.add_noise()
    assert noise.tolist() == syndrome.tolist() == [0]
    m.add_boundary_edge(0, error_probability=0.2, merge_strategy="independent")
    assert m.edges()[0][2]["error_probability"] == -1
    assert m.add_noise() is None
    m = sq.Matching()
    m.add_boundary_edge(0, error_probability=0.2)
    m.add_boundary_edge(0, error_probability=float("nan"), merge_strategy="replace")
    assert np.isnan(m.edges()[0][2]["error_probability"])
    assert m.add_noise() is not None


def test_shared_calls_match_independent_same_history_objects():
    m = square()
    m.decode(SYNDROME)
    m.ensure_num_fault_ids(70)
    shots = np.tile(SYNDROME, (1000, 1))
    independent = square()
    independent.decode(SYNDROME)
    independent.ensure_num_fault_ids(70)
    expected, weights = independent.decode_batch(shots, return_weights=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(m.decode_batch, shots, threads=t, return_weights=True) for t in [1, 2, 1, 3]]
        for future in futures:
            got, w = future.result(timeout=10)
            assert np.array_equal(got, expected)
            assert np.array_equal(w, weights)


def test_empty_graph_and_initial_wide_build():
    m = sq.Matching()
    assert m.decode([]).shape == (0,)
    m.ensure_num_fault_ids(70)
    assert m.decode_to_edges_array([]).shape == (0, 2)
    assert m.decode([]).tolist() == [0] * 70
    assert m.decode_batch(np.zeros((3, 0)), threads=2).shape == (3, 70)
    m = square()
    m.ensure_num_fault_ids(70)
    assert m.decode(SYNDROME).tolist() == [0, 1] + [0] * 68
    assert m.decode_batch(np.zeros((0, 4))).shape == (0, 70)


def test_correlated_padding_keeps_bit63_and_refuses_real_wide_model():
    m = sq.Matching("error(0.1) D0 L63", enable_correlations=True)
    m.ensure_num_fault_ids(70)
    out = m.decode_batch([[1]], enable_correlations=True, bit_packed_predictions=True)
    assert out.tolist() == [[0] * 7 + [128, 0]]
    with pytest.raises(ValueError, match="64"):
        sq.Matching("error(0.1) D0 L64", enable_correlations=True)


def test_core_batch_checked_products_and_zero_width_capacity():
    from stabilizer_qec import _core
    g = _core.UserGraph()
    with pytest.raises(ValueError, match="rows"):
        g.decode_batch(b"", 2**63, 2, 1)
    with pytest.raises(MemoryError, match="matching output"):
        g.decode_batch(b"", 2**63, 0, 1)
    assert g.decode_batch(b"", 0, 0, 1) == (b"", b"")
    assert g.decode([]) == (b"", 0)

"""stabilizer_qec.cudaq_qec, CUDA-Q QEC's decoders under its names (no reference runs here:
CUDA-Q QEC ships only CUDA builds): the lookup tables against brute force, the sliding window
against its inner decoder and the syndrome, the tensor-network decoder against exhaustive
maximum likelihood and CUDA-Q's own published values, the PyMatching plugin against PyMatching,
and the matrix tools against CUDA-Q's definitions."""

from __future__ import annotations

import importlib.util
import itertools

import numpy as np
import pytest

from stabilizer_qec import cudaq_qec as qec


def random_matrix(rng, m, n, density=0.35):
    return (rng.random((m, n)) < density).astype(np.uint8)


def first_explaining_set(H, syndrome, depth):
    """CUDA-Q's table order: by size, then lexicographic; the first set with the syndrome."""
    n = H.shape[1]
    for k in range(1, depth + 1):
        for s in itertools.combinations(range(n), k):
            if np.array_equal(H[:, list(s)].sum(axis=1) % 2, syndrome):
                return list(s)
    return None


@pytest.mark.parametrize("depth", [1, 2, 3])
def test_lookup_tables_are_cudaqs(depth):
    rng = np.random.default_rng(depth)
    for _ in range(10):
        H = random_matrix(rng, int(rng.integers(3, 6)), int(rng.integers(4, 9)))
        name = "single_error_lut" if depth == 1 else "multi_error_lut"
        kw = {} if depth == 1 else {"lut_error_depth": depth}
        dec = qec.get_decoder(name, H, **kw)
        for bits in itertools.product([0, 1], repeat=H.shape[0]):
            syn = np.array(bits)
            r = dec.decode(syn.astype(float))
            if not syn.any():
                assert r.converged and not r.result.any()
                continue
            want = first_explaining_set(H, syn, depth)
            assert r.converged == (want is not None)
            expected = np.zeros(H.shape[1])
            if want is not None:
                expected[want] = 1
            assert np.array_equal(r.result, expected)


def test_lookup_table_options():
    H = np.array([[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1]], dtype=np.uint8)
    O = np.array([[1, 0, 0, 1]], dtype=np.uint8)
    rates = [0.1, 0.2, 0.3, 0.4]
    dec = qec.get_decoder("multi_error_lut", H, lut_error_depth=2, error_rate_vec=rates, opt_results={"error_probability": True, "syndrome_weight": True})
    r = dec.decode([1.0, 0.0, 0.0])
    assert r.result.tolist() == [1, 0, 0, 0] and r.opt_results == {"error_probability": 0.1, "syndrome_weight": 1}
    assert dec.decode([0.0, 0.0, 0.0]).opt_results is None
    obs = qec.get_decoder("multi_error_lut", H, lut_error_depth=2, O=O, output="observables")
    assert obs.decode([0.0, 0.0, 1.0]).result.tolist() == [1.0]
    with pytest.raises(RuntimeError, match="lut_error_depth must be <= block_size"):
        qec.get_decoder("multi_error_lut", H, lut_error_depth=5)
    # (A caller's O replaces the model text's, as CUDA-Q's make_pcm_decoder lets the caller's keys
    # win; the LUT decodes to an error frame either way, as CUDA-Q's ignores O.)
    lut = qec.get_decoder("single_error_lut", "error(0.1) D0 L0\nerror(0.1) D0 D1\nerror(0.1) D2\nerror(0.1) D1 D2\n", O=O)
    assert lut.decode([0.0, 0.0, 1.0]).result.tolist() == [0.0, 0.0, 1.0, 0.0]
    assert qec.get_decoder("single_error_lut", "error(0.1) D0 L0\nerror(0.1) D0 D1\nerror(0.1) D2\nerror(0.1) D1 D2\n", O=O, output="observables").decode([0.0, 1.0, 1.0]).result.tolist() == [1.0]


PM_H = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
PM_O = np.array([[1, 0, 0]], dtype=np.uint8)
PM_DEM = "error(0.1) D0 L0\nerror(0.1) D0 D1\nerror(0.1) D1\n"


def test_pymatching_decodes_to_observables_when_its_model_has_them():
    # CUDA-Q 0.8.1's pymatching switches to decode_to_obs whenever its params hold O, which
    # make_pcm_decoder injects from a model's text when the model has observables.
    by_matrix = qec.get_decoder("pymatching", PM_H, O=PM_O, error_rate_vec=[0.1, 0.1, 0.1])
    by_text = qec.get_decoder("pymatching", PM_DEM)
    for dec in (by_matrix, by_text):
        assert dec.decode([1.0, 0.0]).result.tolist() == [1.0]
        assert dec.decode([1.0, 1.0]).result.tolist() == [0.0]
        assert dec.decode_batch([[1.0, 0.0], [0.0, 1.0]]).result.shape == (2, 1)
    # (The same width as nv-qldpc-decoder's on the same model, which is built on the ldpc port and so
    # needs scipy.)
    if importlib.util.find_spec("scipy"):
        assert len(qec.get_decoder("nv-qldpc-decoder", PM_H, O=PM_O, error_rate_vec=[0.1, 0.1, 0.1]).decode([1.0, 0.0]).result) == 1
    # Without observables, an error frame; output="errors" (this package's) asks for one anyway.
    assert qec.get_decoder("pymatching", PM_H).decode([1.0, 0.0]).result.tolist() == [1.0, 0.0, 0.0]
    assert qec.get_decoder("pymatching", "error(0.1) D0\nerror(0.1) D0 D1\nerror(0.1) D1\n").decode([1.0, 0.0]).result.tolist() == [1.0, 0.0, 0.0]
    assert qec.get_decoder("pymatching", PM_DEM, output="errors").decode([1.0, 0.0]).result.tolist() == [1.0, 0.0, 0.0]
    assert qec.get_decoder("pymatching", PM_H, O=PM_O, output="errors").decode([1.0, 0.0]).result.tolist() == [1.0, 0.0, 0.0]
    with pytest.raises(RuntimeError, match="no observable mapping"):
        qec.get_decoder("pymatching", PM_H, output="observables")


def test_pymatching_merges_parallel_edges_when_decoding_to_observables():
    # (CUDA-Q switches the merge strategy from DISALLOW to INDEPENDENT with O, unless asked.)
    dem = "error(0.1) D0 D1\nerror(0.05) D0 D1 L0\nerror(0.1) D0\nerror(0.1) D1\n"
    dec = qec.get_decoder("pymatching", dem)
    assert dec.decode([1.0, 1.0]).result.tolist() == [0.0]
    O = np.array([[0, 1, 0, 0]], dtype=np.uint8)
    H = np.array([[1, 1, 1, 0], [1, 1, 0, 1]], dtype=np.uint8)
    assert qec.get_decoder("pymatching", H, O=O).decode([1.0, 1.0]).result.tolist() == [0.0]
    for given in (dict(merge_strategy="disallow"), dict(output="errors")):
        with pytest.raises(RuntimeError, match="Parallel edges not permitted"):
            qec.get_decoder("pymatching", dem, **given)


def test_a_callers_o_and_rates_win_over_the_model_texts():
    O = np.array([[0, 1, 0]], dtype=np.uint8)
    # (Syndrome D0 D1 matches the D0-D1 edge: the caller's O says it flips L0, the text's not.)
    assert qec.get_decoder("pymatching", PM_DEM).decode([1.0, 1.0]).result.tolist() == [0.0]
    assert qec.get_decoder("pymatching", PM_DEM, O=O).decode([1.0, 1.0]).result.tolist() == [1.0]
    # (Rates making the D0-D1 edge dearer than both boundary edges: D0 and D1 each to the boundary.)
    rates = [0.4, 0.001, 0.4]
    assert qec.get_decoder("pymatching", PM_DEM, error_rate_vec=rates).decode([1.0, 1.0]).result.tolist() == [1.0]
    assert qec.get_decoder("pymatching", PM_DEM, error_rate_vec=rates, output="errors").decode([1.0, 1.0]).result.tolist() == [1.0, 0.0, 1.0]
    # A caller's model data is checked as the decoder checks any.
    with pytest.raises(RuntimeError, match=r"O must be of shape \(num_observables, block_size\); got second dimension 4, block_size 3"):
        qec.get_decoder("pymatching", PM_DEM, O=np.zeros((1, 4), dtype=np.uint8))
    with pytest.raises(RuntimeError, match="error_rate_vec must be of size block_size"):
        qec.get_decoder("pymatching", PM_DEM, error_rate_vec=[0.1, 0.1])
    with pytest.raises(RuntimeError, match=r"O must be of shape"):
        qec.get_decoder("pymatching", PM_H, O=np.zeros((1, 4), dtype=np.uint8), output="errors")


def test_pymatching_plugin_is_pymatching():
    from stabilizer_qec import pymatching

    rng = np.random.default_rng(3)
    pcm = qec.generate_random_pcm(n_rounds=4, n_errs_per_round=12, n_syndromes_per_round=6, weight=2, seed=5)
    pcm, w = qec.simplify_pcm(pcm, rng.uniform(0.01, 0.1, pcm.shape[1]), 6)
    dec = qec.get_decoder("pymatching", pcm, error_rate_vec=w)
    m = pymatching.Matching.from_check_matrix(pcm, weights=np.log((1 - w) / w))
    for _ in range(40):
        e = (rng.random(pcm.shape[1]) < 0.05).astype(np.uint8)
        syn = pcm @ e % 2
        assert np.array_equal(dec.decode(syn).result.astype(np.uint8), m.decode(syn))


def test_sliding_window():
    rng = np.random.default_rng(1)
    pcm = qec.generate_random_pcm(n_rounds=6, n_errs_per_round=12, n_syndromes_per_round=5, weight=2, seed=7)
    pcm, w = qec.simplify_pcm(pcm, np.full(pcm.shape[1], 0.01), 5)
    layers = pcm.shape[0] // 5
    full = qec.get_decoder("pymatching", pcm, error_rate_vec=w)
    for window, step in [(layers, 1), (3, 1), (4, 2), (2, 2)]:
        sw = qec.get_decoder(
            "sliding_window", pcm, error_rate_vec=w, window_size=window, step_size=step, num_syndromes_per_round=5,
            inner_decoder_name="pymatching", inner_decoder_params={"merge_strategy": "smallest_weight"},
        )
        assert sw.get_num_detector_layers() == layers
        for _ in range(30):
            syn = pcm @ (rng.random(pcm.shape[1]) < 0.03).astype(np.uint8) % 2
            r = sw.decode(syn)
            assert np.array_equal(pcm @ r.result.astype(np.uint8) % 2, syn)
            if window == layers:
                assert np.array_equal(r.result, full.decode(syn).result)
        streamed = [sw.decode(syn[k * 5 : (k + 1) * 5]) for k in range(layers)]
        assert all(len(s.result) == 0 for s in streamed[:-1]) and np.array_equal(streamed[-1].result, sw.decode(syn).result)
    with pytest.raises(ValueError, match="must be sorted"):
        qec.get_decoder("sliding_window", qec.shuffle_pcm_columns(pcm, seed=3), error_rate_vec=w, num_syndromes_per_round=5, inner_decoder_name="pymatching")


def exhaustive_posterior(H, logical, priors, syndrome):
    n = H.shape[1]
    mass = [0.0, 0.0]
    for e in itertools.product([0, 1], repeat=n):
        e = np.array(e)
        p = np.prod(np.where(e == 1, priors, 1 - priors))
        parity = H @ e % 2
        p *= np.prod(np.where(parity == 1, syndrome, 1 - syndrome))
        mass[int(logical @ e % 2)] += p
    return mass[1] / (mass[0] + mass[1])


def test_tensor_network_decoder():
    # CUDA-Q's own test and values (test_tensor_network_decoder.py).
    np.random.seed(42)
    H = np.random.randint(0, 2, size=(5, 8)).astype(np.float64)
    logical = np.random.randint(0, 2, size=(1, 8)).astype(np.float64)
    noise = np.random.uniform(0.01, 0.2, size=8).tolist()
    dec = qec.get_decoder("tensor_network_decoder", H, logical_obs=logical, noise_model=noise)
    batch = np.random.choice([False, True], size=(10, 5)).astype(np.float64)
    expected = [0.9604944927882665, 0.9796816612788876, 0.020709125507417103, 0.35314051570803995, 0.3616138088105539,
                0.01979825044290266, 0.01979825044290266, 0.6381641010485968, 0.01979825044290266, 0.3616795232730325]
    np.testing.assert_allclose([dec.decode(s.tolist()).result[0] for s in batch], expected, rtol=1e-12)
    res = dec.decode_batch(batch)
    assert res.result.shape == (10, 1) and res.converged.all()
    np.testing.assert_allclose(res.result[:, 0], expected, rtol=1e-12)
    # Against exhaustive maximum likelihood, soft syndromes included.
    rng = np.random.default_rng(2)
    for _ in range(10):
        H = random_matrix(rng, 4, 9).astype(float)
        logical = random_matrix(rng, 1, 9).astype(float)
        priors = rng.uniform(0.01, 0.3, 9)
        dec = qec.get_decoder("tensor_network_decoder", H, logical_obs=logical, noise_model=priors.tolist())
        for _ in range(4):
            syn = rng.choice([0.0, 1.0, 0.2, 0.7], 4)
            assert dec.decode(syn.tolist()).result[0] == pytest.approx(exhaustive_posterior(H, logical[0], priors, syn), rel=1e-12)


def test_matrix_tools():
    pcm = qec.generate_random_pcm(n_rounds=5, n_errs_per_round=10, n_syndromes_per_round=4, weight=2, seed=11)
    assert pcm.shape == (20, 50) and set(np.unique(pcm)) <= {0, 1}
    assert (pcm.sum(axis=0) == 2).all()
    s = qec.sort_pcm_columns(pcm, 4)
    assert qec.pcm_is_sorted(s, 4)
    assert np.array_equal(qec.reorder_pcm_columns(pcm, qec.get_sorted_pcm_column_indices(pcm, 4)), s)
    assert np.array_equal(qec.sort_pcm_columns(qec.shuffle_pcm_columns(s, seed=3), 4), s)
    simple, weights = qec.simplify_pcm(s, np.full(50, 0.1), 4)
    assert simple.shape[1] == len({tuple(c) for c in s.T if c.any()}) == weights.size
    sub, first, last = qec.get_pcm_for_rounds(s, 4, 1, 2)
    assert sub.shape[0] == 8 and first <= last
    vec = qec.pcm_to_sparse_vec(np.array([[1, 0, 1], [0, 1, 0]]))
    assert vec == [0, 2, -1, 1, -1]


def test_dem_round_canonicalization():
    dem = qec.dem_from_stim_text("error(0.1) D1 L0\nerror(0.2) D0\nerror(0.3) D1 L0\nerror(0) D0 D1\n")
    dem.canonicalize_for_rounds(1)
    assert dem.detector_error_matrix.tolist() == [[1, 0], [0, 1]]
    assert dem.error_rates == pytest.approx([0.2, 0.1 + 0.3 - 2 * 0.1 * 0.3])

from __future__ import annotations

import numpy as np
import pytest

import stabilizer_qec as sq


def shots_of(circuit, n, seed=1):
    return circuit.compile_detector_sampler(seed=seed).sample(n, separate_observables=True, threads=0)


def failure_rate(pred, obs):
    return float((pred != obs).any(axis=1).mean())


def test_matching_forms_and_threads(d5):
    dem = d5.detector_error_model(decompose_errors=True)
    m = sq.Matching(dem)
    assert (m.num_detectors, m.num_observables) == (dem.num_detectors, dem.num_observables)
    dets, obs = shots_of(d5, 3000)
    pred, weights = m.decode_batch(dets, return_weights=True, threads=1)
    assert pred.shape == (3000, 1) and pred.dtype == np.uint8 and weights.shape == (3000,)
    assert np.array_equal(m.decode_batch(dets, threads=0), pred)
    assert np.array_equal(m.decode(dets[17]), pred[17])
    packed = np.packbits(dets, axis=1, bitorder="little")
    pp = m.decode_batch(packed, bit_packed_shots=True, bit_packed_predictions=True)
    assert np.array_equal(np.unpackbits(pp, axis=1, count=1, bitorder="little"), pred)
    assert np.array_equal(m.decode_batch(dets.astype(np.uint8)), pred)
    assert sq.Matching.from_detector_error_model(str(dem)).decode_batch(dets[:10]).shape == (10, 1)


def test_matching_agrees_with_pymatching(d5):
    stim = pytest.importorskip("stim")
    pymatching = pytest.importorskip("pymatching")
    dem = d5.detector_error_model(decompose_errors=True)
    dets, obs = shots_of(d5, 20_000)
    ours, w = sq.Matching(dem).decode_batch(dets, return_weights=True, threads=0)
    pm = pymatching.Matching.from_detector_error_model(stim.DetectorErrorModel(str(dem)))
    theirs, tw = pm.decode_batch(dets, return_weights=True)
    differ = np.flatnonzero((ours != theirs).any(axis=1))
    # Two exact matchers disagree only where two corrections tie in weight.
    assert len(differ) < 20
    assert np.allclose(w[differ], tw[differ], rtol=1e-6, atol=1e-6)


def test_correlated_matching_is_no_worse(d5):
    dem = d5.detector_error_model(decompose_errors=True)
    dets, obs = shots_of(d5, 20_000, seed=2)
    plain = failure_rate(sq.Matching(dem).decode_batch(dets, threads=0), obs)
    corr = failure_rate(sq.Matching(dem, enable_correlations=True).decode_batch(dets, threads=0), obs)
    assert corr < plain + 3 * np.sqrt(plain / 20_000)


def test_a_shot_nothing_explains_is_an_error():
    m = sq.Matching("error(0.1) D0 D1\nerror(0.1) D1 D2 L0\n")
    with pytest.raises(ValueError, match="shot 1"):
        m.decode_batch(np.array([[0, 0, 0], [1, 0, 0]], dtype=bool))


def test_bad_shots(d3):
    m = sq.Matching(d3.detector_error_model(decompose_errors=True))
    with pytest.raises(ValueError):
        m.decode_batch(np.zeros((2, d3.num_detectors + 1), dtype=bool))
    with pytest.raises(TypeError):
        m.decode_batch(np.zeros((2, 2), dtype=np.int64), bit_packed_shots=True)
    with pytest.raises(TypeError):
        m.decode_batch(np.zeros((2, d3.num_detectors), dtype=float))
    with pytest.raises(ValueError):
        m.decode(np.zeros((1, d3.num_detectors), dtype=bool))


def test_belief_matching_agrees_with_beliefmatching(d3):
    stim = pytest.importorskip("stim")
    bm_pkg = pytest.importorskip("beliefmatching")
    dem = d3.detector_error_model(decompose_errors=True)
    dets, obs = shots_of(d3, 400, seed=3)
    ours, conv = sq.BeliefMatching(dem, max_bp_iters=20).decode_batch(dets, return_converged=True)
    theirs = bm_pkg.BeliefMatching(stim.DetectorErrorModel(str(dem)), max_bp_iters=20).decode_batch(dets)
    assert np.array_equal(ours, theirs)
    assert conv.dtype == np.bool_ and conv.shape == (400,)


def test_bposd_on_a_model(d3):
    dem = d3.detector_error_model()
    dets, obs = shots_of(d3, 2000, seed=4)
    pred, conv = sq.BpOsd(dem).decode_batch(dets, return_converged=True, threads=0)
    plain = failure_rate(sq.Matching(d3.detector_error_model(decompose_errors=True)).decode_batch(dets), obs)
    assert failure_rate(pred, obs) < 1.5 * plain + 0.01
    assert conv.mean() > 0.5


def test_bplsd_on_a_model(d3):
    dem = d3.detector_error_model()
    dets, obs = shots_of(d3, 2000, seed=4)
    lsd = sq.BpLsd(dem)
    pred, conv = lsd.decode_batch(dets, return_converged=True, threads=0)
    plain = failure_rate(sq.Matching(d3.detector_error_model(decompose_errors=True)).decode_batch(dets), obs)
    assert failure_rate(pred, obs) < 1.5 * plain + 0.01
    assert 0 < conv.mean() < 1
    assert np.array_equal(lsd.decode_batch(dets, threads=1), pred)
    assert np.array_equal(lsd.decode(dets[3]), pred[3])


def test_window_matching(d3):
    c = sq.memory_circuit(distance=3, rounds=30, p=0.005)
    dem = c.detector_error_model(decompose_errors=True)
    dets, obs = shots_of(c, 4000, seed=5)
    glob = failure_rate(sq.Matching(dem).decode_batch(dets, threads=0), obs)
    for mode in ("sliding", "parallel"):
        w = sq.WindowMatching(dem, commit=4, buffer=4, mode=mode)
        assert all(isinstance(x, sq.Window) for x in w.windows)
        pred, times = w.decode_batch(dets, return_timings=True, threads=0)
        assert times.shape == (4000, len(w.windows))
        assert failure_rate(pred, obs) < glob + 4 * np.sqrt(glob / 4000) + 0.002
    with pytest.raises(ValueError):
        sq.WindowMatching(dem, commit=4, buffer=4, mode="diagonal")


def random_code(seed, m=12, n=24):
    rng = np.random.default_rng(seed)
    pcm = (rng.random((m, n)) < 0.25).astype(np.uint8)
    pcm[rng.integers(m, size=n), np.arange(n)] = 1
    channel = rng.uniform(0.01, 0.1, n)
    return rng, pcm, channel


def test_bp_equals_ldpc():
    ldpc = pytest.importorskip("ldpc")
    for seed in range(5):
        rng, pcm, channel = random_code(seed)
        for method, scale in (("product_sum", 1.0), ("minimum_sum", 0.625)):
            ours = sq.BpDecoder(pcm, error_channel=channel, max_iter=15, bp_method=method, ms_scaling_factor=scale)
            theirs = ldpc.BpDecoder(pcm, error_channel=list(channel), max_iter=15, bp_method=method, ms_scaling_factor=scale)
            for _ in range(10):
                syndrome = (pcm @ (rng.random(pcm.shape[1]) < channel)) % 2
                assert np.array_equal(ours.decode(syndrome), theirs.decode(syndrome))
                if syndrome.any():  # ldpc returns early on a zero syndrome, its posteriors stale
                    assert np.array_equal(ours.log_prob_ratios, theirs.log_prob_ratios)
                    assert (ours.converge, ours.iter) == (theirs.converge, theirs.iter)


def test_bposd_matrix_decoder():
    rng, pcm, channel = random_code(9)
    dec = sq.BpOsdDecoder(pcm, error_channel=channel, osd_method="osd_cs", osd_order=4)
    for _ in range(20):
        syndrome = (pcm @ (rng.random(pcm.shape[1]) < channel)) % 2
        correction = dec.decode(syndrome)
        assert np.array_equal((pcm @ correction) % 2, syndrome)
    scipy = pytest.importorskip("scipy.sparse")
    sparse = sq.BpOsdDecoder(scipy.csr_matrix(pcm), error_rate=0.05)
    assert sparse.decode(np.zeros(pcm.shape[0], dtype=np.uint8)).sum() == 0
    with pytest.raises(ValueError):
        sq.BpOsdDecoder(pcm)
    with pytest.raises(ValueError):
        dec.decode(np.zeros(pcm.shape[0] + 1))


def test_bposd_matrix_decoder_agrees_with_ldpc():
    ldpc = pytest.importorskip("ldpc")
    agree = total = 0
    for seed in range(4):
        rng, pcm, channel = random_code(seed + 20)
        kw = dict(max_iter=10, bp_method="minimum_sum", ms_scaling_factor=0.625, osd_method="osd_cs", osd_order=3)
        ours = sq.BpOsdDecoder(pcm, error_channel=channel, **kw)
        theirs = ldpc.BpOsdDecoder(pcm, error_channel=list(channel), **kw)
        for _ in range(25):
            syndrome = (pcm @ (rng.random(pcm.shape[1]) < channel)) % 2
            agree += np.array_equal(ours.decode(syndrome), theirs.decode(syndrome))
            total += 1
    # Equal but for ties among columns of equal posterior, which ldpc orders with std::sort.
    assert agree >= 0.95 * total


@pytest.mark.parametrize("bp_method", ["minimum_sum", "product_sum"])
def test_bplsd_matrix_decoder_equals_ldpc(bp_method):
    ldpc = pytest.importorskip("ldpc")
    agree = tied = total = 0
    for seed in range(6):
        rng, pcm, channel = random_code(seed + 40, m=24, n=48)
        kw = dict(max_iter=3, bp_method=bp_method, ms_scaling_factor=0.625, lsd_method="lsd_0", lsd_order=0)
        ours = sq.BpLsdDecoder(pcm, error_channel=channel, **kw)
        theirs = ldpc.BpLsdDecoder(pcm, error_channel=list(channel), **kw)
        for _ in range(40):
            syndrome = ((pcm @ (rng.random(pcm.shape[1]) < channel)) % 2).astype(np.uint8)
            mine = ours.decode(syndrome)
            assert np.array_equal(pcm @ mine % 2, syndrome)
            same = np.array_equal(mine, theirs.decode(syndrome))
            agree += same
            tied += not same and ours.last_tied
            total += 1
            # A disagreement is only ever where ldpc's order is not reproducible.
            assert same or ours.last_tied
            # (ldpc skips BP on a zero syndrome and leaves its iteration count stale.)
            assert ours.converge == theirs.converge and (ours.iter == theirs.iter or not syndrome.any())
    assert agree >= 0.95 * total


def test_bplsd_matrix_decoder_reads_ldpcs_arguments():
    _, pcm, channel = random_code(3)
    # osd_* aliases, LSD-0 forcing the order to 0, and max_iter=0 meaning the block length.
    a = sq.BpLsdDecoder(pcm, error_channel=channel, osd_method="osd_cs", osd_order=4, max_iter=0)
    b = sq.BpLsdDecoder(pcm, error_channel=channel, lsd_method="lsd_cs", lsd_order=4, max_iter=pcm.shape[1])
    c = sq.BpLsdDecoder(pcm, error_channel=channel, lsd_method="lsd_0", lsd_order=7)
    syndrome = pcm[:, 0]
    assert np.array_equal(a.decode(syndrome), b.decode(syndrome))
    assert np.array_equal(pcm @ c.decode(syndrome) % 2, syndrome)
    for bad in (dict(lsd_method="lsd_x"), dict(lsd_method=3), dict(lsd_order=-1), dict(schedule="serial"), dict(bp_method="nope")):
        with pytest.raises(ValueError):
            sq.BpLsdDecoder(pcm, error_channel=channel, **bad)
    with pytest.raises(ValueError):
        a.decode(np.zeros(pcm.shape[0] + 1))


def test_union_find(d3):
    import pickle

    c = sq.memory_circuit(distance=5, rounds=5, p=0.004)
    dem = c.detector_error_model(decompose_errors=True)
    dets, obs = shots_of(c, 20_000, seed=21)
    uf = sq.UnionFind(dem)
    pred = uf.decode_batch(dets, threads=0)
    mwpm = failure_rate(sq.Matching(dem).decode_batch(dets, threads=0), obs)
    ufr = failure_rate(pred, obs)
    assert mwpm <= ufr < 1.5 * mwpm + 0.002, (ufr, mwpm)
    # The same shots, bit-packed in and out, on one thread or many, and after pickling.
    packed = np.packbits(dets, axis=1, bitorder="little")
    assert np.array_equal(np.unpackbits(uf.decode_batch(packed, bit_packed_shots=True, bit_packed_predictions=True), axis=1, bitorder="little", count=1), pred)
    assert np.array_equal(pickle.loads(pickle.dumps(uf)).decode_batch(dets[:500]), pred[:500])
    assert np.array_equal(uf.decode(dets[3]), pred[3])
    # A detection event with no partner and no boundary cannot be explained.
    with pytest.raises(ValueError, match="no correction explains"):
        sq.UnionFind("error(0.1) D0 D1\nerror(0.1) D1 D2").decode_batch(np.array([[1, 0, 0]], dtype=bool))

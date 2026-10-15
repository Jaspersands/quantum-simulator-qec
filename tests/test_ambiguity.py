"""Ambiguity Clustering (Wolanski and Barber, arXiv 2406.14527) on detector error models: at
kappa 0 it is BP+OSD-0 on the same BP, and grown further it reaches BP+OSD-CS(7)'s accuracy on
the bivariate bicycle codes' memories at a fraction of its time."""

import pickle
import time

import numpy as np
import pytest

import stabilizer_qec as sq


def bb(name, cycles, p, shots, seed):
    c = sq.BivariateBicycleCode(name).memory_circuit(cycles, p)
    dets, obs = c.compile_detector_sampler(seed=seed).sample(shots, separate_observables=True)
    return c.detector_error_model(), dets, obs


def failures(pred, obs):
    return int((pred != obs).any(axis=1).sum())


@pytest.mark.parametrize("p", [0.003, 0.006])
def test_kappa_zero_is_osd0_on_the_same_bp(p):
    # Stage 1 alone: the pivots are chosen as OSD's are, by BP's posteriors, but only as far as
    # the syndrome needs. On these shots every prediction equals OSD-0's.
    dem, dets, obs = bb("72", 6, p, 600, 3)
    ac = sq.AmbiguityClustering(dem, kappa=0)
    osd0 = sq.BpOsd(dem, osd_method="osd_0", osd_order=0, max_iter=9, bp_method="product_sum")
    assert ac.extra_columns == 0
    np.testing.assert_array_equal(ac.decode_batch(dets, threads=4), osd0.decode_batch(dets, threads=4))


def test_reaches_bposd_accuracy_faster():
    # [[72,12,6]], 6 cycles at p = 0.3%: the paper's kappa 0.05 against BP+OSD-CS(7) with 10,000
    # iterations of min-sum BP (Bravyi et al.'s decoder), on the same shots.
    dem, dets, obs = bb("72", 6, 0.003, 2000, 3)
    start = time.perf_counter()
    ac = failures(sq.AmbiguityClustering(dem).decode_batch(dets), obs)
    t_ac = time.perf_counter() - start
    start = time.perf_counter()
    osd = failures(sq.BpOsd(dem).decode_batch(dets), obs)
    t_osd = time.perf_counter() - start
    assert ac <= osd, (ac, osd)
    assert 20 < osd < 60
    assert t_ac < t_osd / 3, (t_ac, t_osd)


def test_more_columns_help():
    dem, dets, obs = bb("72", 6, 0.006, 600, 4)
    fails = [failures(sq.AmbiguityClustering(dem, kappa=k).decode_batch(dets, threads=4), obs) for k in (0.0, 0.05, 0.2)]
    assert fails[0] > fails[1] >= fails[2] * 0.9, fails


def test_stats_converged_and_pickling():
    dem, dets, obs = bb("72", 6, 0.006, 200, 5)
    ac = sq.AmbiguityClustering(dem, kappa=0.1)
    pred, conv, stats = ac.decode_batch(dets, return_converged=True, return_stats=True, threads=3)
    assert stats.shape == (200, 3)
    assert (stats[conv] == 0).all() and (stats[~conv, 0] > 0).all()
    assert stats[:, 1].sum() > 0 and (stats[:, 2] >= stats[:, 1]).all()
    # one shot at a time, other thread counts, bit-packed, pickled: the same predictions
    np.testing.assert_array_equal(pred, np.array([ac.decode(d) for d in dets]))
    packed = np.packbits(dets, axis=1, bitorder="little")
    np.testing.assert_array_equal(np.unpackbits(ac.decode_batch(packed, bit_packed_shots=True, bit_packed_predictions=True), axis=1, count=12, bitorder="little"), pred)
    again = pickle.loads(pickle.dumps(ac))
    assert again.extra_columns == ac.extra_columns == round(0.1 * dem.num_errors)
    np.testing.assert_array_equal(again.decode_batch(dets), pred)


def test_options():
    dem, dets, obs = bb("72", 6, 0.003, 300, 6)
    for kw in (dict(schedule="serial"), dict(schedule="serial_relative"), dict(bp_method="minimum_sum", ms_scaling_factor=0.625), dict(max_iter=29)):
        assert failures(sq.AmbiguityClustering(dem, **kw).decode_batch(dets), obs) < 30, kw
    with pytest.raises(ValueError, match="kappa"):
        sq.AmbiguityClustering(dem, kappa=-0.1)
    with pytest.raises(ValueError, match="schedule"):
        sq.AmbiguityClustering(dem, schedule="random")
    with pytest.raises(ValueError, match="bp_method"):
        sq.AmbiguityClustering(dem, bp_method="bogus")


def test_inconsistent_syndrome_raises():
    dem = sq.DetectorErrorModel("error(0.1) D0 D1\nerror(0.1) D1 L0\ndetector D2")
    ac = sq.AmbiguityClustering(dem)
    assert list(ac.decode(np.array([1, 0, 0], dtype=bool))) == [1]
    with pytest.raises(ValueError, match="no correction explains"):
        ac.decode(np.array([0, 0, 1], dtype=bool))


def test_sinter_kind():
    from stabilizer_qec import sinter

    c = sq.BivariateBicycleCode("72").memory_circuit(2, 0.003)
    task = sinter.Task(circuit=c, json_metadata={})
    stats = sinter.collect(tasks=[task], decoders=["ambiguity_clustering"], num_workers=1, max_shots=200)
    assert stats[0].shots == 200 and stats[0].errors < 20

"""Post-selection by argument reweighting (Xie, Yoshioka, Tsubouchi and Li, arXiv 2601.17757),
against the paper's claims: rejection rises with b and the kept shots' logical error rate falls;
the three-round logical-error criterion reaches PEC's suppression at a fraction of its rejection;
and it works around matching, union-find, BP+OSD, BP+LSD and Relay-BP."""

import pickle

import numpy as np
import pytest

import stabilizer_qec as sq


def surface(p, shots, seed):
    c = sq.memory_circuit(distance=5, rounds=5, p=p)
    dets, obs = c.compile_detector_sampler(seed=seed).sample(shots, separate_observables=True)
    return c.detector_error_model(decompose_errors=True), dets, obs


def rates(dec, dets, obs):
    pred, kept = dec.decode_batch(dets, threads=4)
    wrong = (pred != obs).any(axis=1)
    return 1 - kept.mean(), wrong[kept].mean()


def test_predictions_are_the_inner_decoders_and_priors_are_restored():
    dem, dets, obs = surface(0.004, 3000, 1)
    ar = sq.ArgumentReweighting(dem, b=1.5)
    pred, kept, rounds = ar.decode_batch(dets, return_rounds=True, threads=3)
    np.testing.assert_array_equal(pred, sq.Matching(dem).decode_batch(dets))
    # The same shots again, on one thread: every shot's changed priors were put back.
    again, kept1 = ar.decode_batch(dets)
    np.testing.assert_array_equal(again, pred)
    np.testing.assert_array_equal(kept1, kept)
    # A shot with no detection events is kept after one decode; a rejected 3R-LEC shot ran 2 or 3.
    empty = ~dets.any(axis=1)
    assert empty.any() and (rounds[empty] == 1).all() and kept[empty].all()
    assert set(rounds[~empty].tolist()) <= {2, 3} and not kept.all()
    # A two-round criterion never runs a third decode.
    assert sq.ArgumentReweighting(dem, b=1.5, criterion="pec").decode_batch(dets, return_rounds=True)[2].max() == 2
    one, ok = ar.decode(dets[0])
    assert (one == pred[0]).all() and ok == kept[0]


def test_suppression_against_rejection():
    # The paper's Fig. 2 regime: a d=5 surface code, MWPM.
    dem, dets, obs = surface(0.002, 30_000, 2)
    base = (sq.Matching(dem).decode_batch(dets, threads=4) != obs).any(axis=1).mean()
    for crit in ("pec", "2r-lec", "3r-lec"):
        curve = [rates(sq.ArgumentReweighting(dem, b=b, criterion=crit), dets, obs) for b in (1.05, 1.2, 1.4)]
        rejected = [r for r, _ in curve]
        assert rejected == sorted(rejected) and rejected[0] > 0, (crit, curve)
        assert curve[-1][1] < 0.6 * base, (crit, curve, base)
    # 3R-LEC: a fifth of the logical error rate at a few percent rejection, where PEC's
    # suppression costs several times the rejection.
    r3, l3 = rates(sq.ArgumentReweighting(dem, b=1.2), dets, obs)
    rp, lp = rates(sq.ArgumentReweighting(dem, b=1.2, criterion="pec"), dets, obs)
    assert r3 < 0.06 and l3 < 0.3 * base, (r3, l3, base)
    assert rp > 3 * r3 and lp > 0.7 * l3, (rp, lp, r3, l3)


@pytest.mark.parametrize("test,bs", [("exact_ratio", (1.05, 1.3)), ("gap", (1.0, 6.0))])
def test_the_other_tests(test, bs):
    dem, dets, obs = surface(0.002, 20_000, 3)
    base = (sq.Matching(dem).decode_batch(dets, threads=4) != obs).any(axis=1).mean()
    (r0, l0), (r1, l1) = (rates(sq.ArgumentReweighting(dem, b=b, test=test), dets, obs) for b in bs)
    assert 0 < r0 < r1 and l1 < l0 < base, (r0, l0, r1, l1, base)


def test_union_find():
    dem, dets, obs = surface(0.002, 20_000, 4)
    base = (sq.UnionFind(dem).decode_batch(dets, threads=4) != obs).any(axis=1).mean()
    r, l = rates(sq.ArgumentReweighting(dem, "union_find", b=1.2), dets, obs)
    assert r < 0.08 and l < 0.4 * base, (r, l, base)


@pytest.mark.parametrize("decoder,options,shots", [("bposd", dict(max_iter=30), 1500), ("bplsd", dict(max_iter=30), 1500), ("relay_bp", {}, 800)])
def test_bp_decoders_on_a_bivariate_bicycle_code(decoder, options, shots):
    # [[72,12,6]], 6 cycles at p = 0.3%.
    c = sq.BivariateBicycleCode("72").memory_circuit(6, 0.003)
    dets, obs = c.compile_detector_sampler(seed=2).sample(shots, separate_observables=True)
    dem = c.detector_error_model()
    ar = sq.ArgumentReweighting(dem, decoder, b=1.1, **options)
    pred, kept = ar.decode_batch(dets, threads=4)
    if decoder == "bposd":
        np.testing.assert_array_equal(pred, sq.BpOsd(dem, **options).decode_batch(dets, threads=4))
    wrong = (pred != obs).any(axis=1)
    assert 0 < 1 - kept.mean() < 0.25 and wrong[kept].mean() < 0.3 * wrong.mean(), (1 - kept.mean(), wrong[kept].mean(), wrong.mean())
    # Threads do not change a shot's outcome (Relay-BP's seeds are per shot).
    np.testing.assert_array_equal(ar.decode_batch(dets[:200])[1], kept[:200])
    again = pickle.loads(pickle.dumps(ar))
    np.testing.assert_array_equal(again.decode_batch(dets[:200])[1], kept[:200])


def test_errors():
    dem = sq.DetectorErrorModel("error(0.1) D0 D1 L0")
    with pytest.raises(ValueError, match="b"):
        sq.ArgumentReweighting(dem, b=1.0)
    with pytest.raises(ValueError, match="b"):
        sq.ArgumentReweighting(dem, b=0, test="gap")
    with pytest.raises(ValueError, match="test"):
        sq.ArgumentReweighting(dem, test="likelihood")
    with pytest.raises(ValueError, match="criterion"):
        sq.ArgumentReweighting(dem, criterion="4r-lec")
    with pytest.raises(ValueError, match="decoder"):
        sq.ArgumentReweighting(dem, "tesseract")
    with pytest.raises(TypeError, match="takes no options"):
        sq.ArgumentReweighting(dem, max_iter=3)
    ar = sq.ArgumentReweighting(dem, "union_find")
    with pytest.raises(ValueError, match="no correction explains"):
        ar.decode_batch(np.array([[1, 0]], dtype=bool))
    pred, kept = pickle.loads(pickle.dumps(ar)).decode(np.array([1, 1], dtype=bool))
    assert pred.tolist() == [1] and kept  # (no other correction explains it: suppressed, it is still the answer)

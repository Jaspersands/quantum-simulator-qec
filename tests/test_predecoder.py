"""The AI pre-decoder (NVIDIA's shape, arXiv 2604.12841): the layout of a model's detectors and
graph-like faults on a grid, the labels its sampler's faults make, the network trained in Rust
and in numpy (the same, step for step, but for rounding), the paper's pipeline (hard
corrections, then any decoder) and matching on the network's weights, and this package's
weights for the rotated memory."""

import pickle

import numpy as np
import pytest

import stabilizer_qec as sq


def model(distance=3, rounds=3, p=0.01):
    return sq.memory_circuit(distance=distance, rounds=rounds, p=p).detector_error_model(decompose_errors=True)


def test_layout():
    dem = model()
    pre = sq.PreDecoder(dem, hidden=(4,))
    assert pre.dims == (4, 4, 4) and pre.num_sectors == 2
    coords = dem.get_detector_coordinates()
    # The first round's detectors are the Z checks: sector 0; the X checks are sector 1.
    assert pre.sectors[0] == 0 and sorted(set(pre.sectors)) == [0, 1]
    assert {pre.sectors[d] for d in range(dem.num_detectors) if coords[d][2] == 1} == {0}
    kinds = pre.kinds
    assert len(kinds) == len(set(kinds)) and all(len(k) == 5 for k in kinds)
    # Each sector has a boundary kind and a timelike kind (the same check a round later).
    for s in (0, 1):
        assert (s, True, 0, 0, 0) in kinds and (s, False, 1, 0, 0) in kinds
    # Every fault is a graph-like symptom of the model, one per kind and voxel.
    symptoms = set()
    for inst in dem.flattened():
        if inst.type == "error":
            dets, obs = [], 0
            for t in inst.targets_copy():
                if t.is_separator():
                    symptoms.add(tuple(sorted(dets)))
                    dets = []
                elif t.is_relative_detector_id():
                    dets.append(t.val)
            symptoms.add(tuple(sorted(dets)))
    assert all(tuple(sorted(d)) in symptoms for d, _, _ in pre.faults)
    assert len(pre.faults) == int(np.frombuffer(pre._x.mask, dtype=np.uint8).sum())


def test_layout_errors():
    with pytest.raises(ValueError, match="coordinates"):
        sq.PreDecoder("error(0.1) D0 D1\ndetector(0, 0, 0) D0")
    with pytest.raises(ValueError, match="regular grid"):
        sq.PreDecoder("error(0.1) D0 D1\nerror(0.1) D1 D2\ndetector(0, 0, 0) D0\ndetector(1, 0, 0) D1\ndetector(2.5, 0, 0) D2")
    with pytest.raises(ValueError, match="same coordinates"):
        sq.PreDecoder("error(0.1) D0 D1\ndetector(0, 0, 0) D0\ndetector(0, 0, 0) D1")
    with pytest.raises(ValueError, match="kernel must be odd"):
        sq.PreDecoder(model(), kernel=2)
    with pytest.raises(ValueError, match="threshold"):
        sq.PreDecoder(model(), threshold=1.0)
    with pytest.raises(ValueError, match="reweight needs"):
        sq.PreDecoder(model(), decoder="bposd", reweight=True)
    with pytest.raises(ValueError, match="engine"):
        sq.PreDecoder(model()).train(1, engine="torch")


def test_labels_explain_the_shot():
    # The faults labelled in a shot are its graph-like faults: together they make its detection
    # events exactly, and (but where two faults with one symptom flip different observables)
    # its observable flips.
    dem = model(p=0.02)
    pre = sq.PreDecoder(dem, hidden=(4,))
    dets, obs, errs = dem.compile_sampler(seed=3).sample(300, return_errors=True)
    t, y, x = pre.dims
    packed = np.packbits(dets, axis=1, bitorder="little").tobytes()
    raw_x, raw_y = pre._x.encode(packed, 300, np.packbits(errs, axis=1, bitorder="little").tobytes())
    labels = np.frombuffer(raw_y, dtype="<f4").reshape(300, -1)
    inputs = np.frombuffer(raw_x, dtype="<f4").reshape(300, t * y * x, 4)
    assert (inputs[:, :, :2].sum(axis=(1, 2)) == dets.sum(axis=1)).all()
    slot = np.full(labels.shape[1], -1)
    slot[np.frombuffer(pre._x.mask, dtype=np.uint8).astype(bool)] = np.arange(len(pre.faults))
    faults = pre.faults
    agree = 0
    for s in range(300):
        made = np.zeros(dem.num_detectors, dtype=bool)
        flips = 0
        for i in np.flatnonzero(labels[s]):
            ds, os, _ = faults[slot[i]]
            made[ds] ^= True
            flips ^= os
        np.testing.assert_array_equal(made, dets[s])
        agree += flips == int(obs[s, 0])
    assert agree >= 295


def test_numpy_trains_as_rust_does():
    dem = model()
    a, b = sq.PreDecoder(dem, hidden=(8, 8), seed=3), sq.PreDecoder(dem, hidden=(8, 8), seed=3)
    dets, _, errs = dem.compile_sampler(seed=5).sample(64, return_errors=True)
    la, ga = a.loss_and_grad(dets, errs, threads=2)
    lb, gb = a.loss_and_grad(dets, errs, engine="numpy")
    assert abs(la - lb) < 1e-5 * la and np.abs(ga - gb).max() < 1e-5 * np.abs(ga).max()
    for optimizer in ("adam", "lion"):
        a, b = sq.PreDecoder(dem, hidden=(8, 8), seed=3), sq.PreDecoder(dem, hidden=(8, 8), seed=3)
        steps = 20 if optimizer == "adam" else 3
        la = a.train(steps, batch=64, optimizer=optimizer, warmup=5, threads=2)
        lb = b.train(steps, batch=64, optimizer=optimizer, warmup=5, engine="numpy")
        np.testing.assert_allclose(la, lb, rtol=1e-5)
        for (wa, ba), (wb, bb) in zip(a.weights["layers"], b.weights["layers"]):
            np.testing.assert_allclose(wa, wb, atol=2e-5)
            np.testing.assert_allclose(ba, bb, atol=2e-5)
        assert a.steps == b.steps == steps
    with pytest.raises(ValueError, match="rust engine"):
        a.train(1, engine="numpy")
    with pytest.raises(ValueError, match="numpy engine"):
        b.train(1)


def test_training_and_both_pipelines():
    dem = model(p=0.01)
    pre = sq.PreDecoder(dem, hidden=(16, 16), seed=1)
    losses = pre.train(600, batch=128, optimizer="adam", threads=4, seed=7)
    assert losses[-20:].mean() < 0.6 * losses[:20].mean()
    dets, obs, _ = dem.compile_sampler(seed=11).sample(4000)
    left, flips, applied = pre.predecode(dets, threads=4)
    assert left.sum() < 0.7 * dets.sum() and applied.sum() > 0
    plain = (sq.Matching(dem).decode_batch(dets) != obs).any(axis=1).sum()
    soft = (pre.decode_batch(dets, threads=4) != obs).any(axis=1).sum()
    # A network this briefly trained is not yet better than matching (the pretrained one is).
    assert soft <= 1.2 * plain, (soft, plain)
    # The paper's pipeline: the hard corrections, then matching with its own weights; or any decoder.
    hard = sq.PreDecoder(dem, pre.weights, reweight=False)
    np.testing.assert_array_equal(hard.decode_batch(dets), sq.Matching(dem).decode_batch(left) ^ flips)
    bposd = sq.PreDecoder(dem, pre.weights, decoder="bposd")
    some = (sq.Matching(dem).decode_batch(dets[:1000]) != obs[:1000]).any(axis=1).sum()
    assert not bposd.reweight and (bposd.decode_batch(dets[:1000]) != obs[:1000]).any(axis=1).sum() <= 1.5 * some + 5


def test_weights_save_load_pickle_and_transfer(tmp_path):
    dem = model()
    pre = sq.PreDecoder(dem, hidden=(8,), seed=2)
    pre.train(20, batch=64, threads=2)
    dets, _, _ = dem.compile_sampler(seed=1).sample(500)
    path = tmp_path / "w.npz"
    pre.save(path)
    again = sq.PreDecoder(dem, str(path))
    np.testing.assert_array_equal(again.decode_batch(dets), pre.decode_batch(dets))
    np.testing.assert_array_equal(pickle.loads(pickle.dumps(pre)).decode_batch(dets), pre.decode_batch(dets))
    # A larger memory has the same kinds: the weights run on it.
    big = model(distance=5, rounds=4)
    other = sq.PreDecoder(big, pre.weights)
    assert sorted(other.kinds) == sorted(pre.kinds) and other.dims == (5, 6, 6)
    # A kind the weights do not know is never applied.
    w = pre.weights
    w["kinds"] = w["kinds"][1:]
    w["layers"][-1] = (w["layers"][-1][0][..., 1:], w["layers"][-1][1][1:])
    w["channels"][-1] -= 1
    part = sq.PreDecoder(dem, w)
    assert np.allclose(part.weights["layers"][-1][1][0], -30)
    with pytest.raises(ValueError, match="do not fit"):
        sq.PreDecoder(dem, hidden=(9,)).weights = pre.weights


@pytest.mark.parametrize("distance", [5, 7])
def test_pretrained_weights_beat_matching(distance):
    dem = model(distance=distance, rounds=distance, p=0.006)
    pre = sq.PreDecoder.pretrained(dem)
    dets, obs, _ = dem.compile_sampler(seed=distance).sample(10000)
    left, _, _ = pre.predecode(dets, threads=4)
    assert dets.sum() > 3 * left.sum()
    plain = (sq.Matching(dem).decode_batch(dets) != obs).any(axis=1).sum()
    ours = (pre.decode_batch(dets, threads=4) != obs).any(axis=1).sum()
    assert ours < 0.9 * plain, (ours, plain)


def test_calls_from_threads_wait_for_training():
    import concurrent.futures as cf

    c = sq.Circuit.generated("surface_code:rotated_memory_z", distance=3, rounds=2, after_clifford_depolarization=0.01)
    p = sq.PreDecoder(c.detector_error_model(decompose_errors=True))
    dets = c.compile_detector_sampler(seed=1).sample(32)
    with cf.ThreadPoolExecutor(4) as ex:
        jobs = [ex.submit(p.train, 2, batch=16) for _ in range(2)] + [ex.submit(p.predecode, dets) for _ in range(4)] + [ex.submit(p.decode_batch, dets) for _ in range(2)]
        for j in jobs:
            j.result()
    assert p.steps == 4


@pytest.mark.parametrize("engine", ["rust", "numpy"])
def test_copies_keep_the_threshold_and_training(engine):
    # A pickled, deep-copied or copied pre-decoder has the threshold set after it was made (so
    # applies the same faults and decodes alike), and the optimizer's state: trained on, it
    # takes the steps the original takes.
    import copy

    dem = model()
    dets, _, _ = dem.compile_sampler(seed=1).sample(1000)
    pre = sq.PreDecoder(dem, hidden=(4,), kernel=3, seed=3)
    pre.train(3, batch=32, seed=1, engine=engine)
    pre.threshold = 0.05
    copies = [pickle.loads(pickle.dumps(pre)), copy.deepcopy(pre), copy.copy(pre)]
    applied = pre.predecode(dets)[2]
    for c in copies:
        assert (c.threshold, c.steps, c.reweight) == (0.05, 3, pre.reweight)
        np.testing.assert_array_equal(c.predecode(dets)[2], applied)
        np.testing.assert_array_equal(c.decode_batch(dets), pre.decode_batch(dets))
    losses = [c.train(3, batch=32, seed=2, engine=engine) for c in copies]
    want = pre.train(3, batch=32, seed=2, engine=engine)
    for c, got in zip(copies, losses):
        np.testing.assert_array_equal(got, want)
        assert c.steps == pre.steps == 6
        for (w1, b1), (w2, b2) in zip(c.weights["layers"], pre.weights["layers"]):
            np.testing.assert_array_equal(w1, w2)
            np.testing.assert_array_equal(b1, b2)

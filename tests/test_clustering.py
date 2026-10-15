"""Collision Clustering (Barber et al., arXiv 2309.05558) and the Local Clustering Decoder (Ziad
et al., arXiv 2411.10343), against their papers' claims: the clusters are union-find's (the Rust
tests check them round by round), the unweighted decoder's threshold under SD6 is about 0.55%
and the weighted one's near matching's, and pre-growing the edges a heralded leakage event makes
likely beats decoding without the heralds."""

import pickle

import numpy as np
import pytest

import stabilizer_qec as sq
from stabilizer_qec._circuit import Circuit
from stabilizer_qec._stim import gate_data


def memory(d, p, shots, seed, basis="z"):
    c = sq.memory_circuit(distance=d, rounds=d, p=p, basis=basis)
    dets, obs = c.compile_detector_sampler(seed=seed).sample(shots, separate_observables=True)
    return c, c.detector_error_model(decompose_errors=True), dets, obs


def failures(pred, obs):
    return int((pred != obs).any(axis=1).sum())


def per_round(fails, shots, d):
    return 0.5 * (1 - (1 - 2 * fails / shots) ** (1 / d))


@pytest.mark.parametrize("weighted", [False, True])
def test_the_two_decoders_agree_shot_for_shot(weighted):
    c, dem, dets, obs = memory(5, 0.006, 3000, 1)
    cc, cs = sq.CollisionClustering(dem, weighted=weighted).decode_batch(dets, return_stats=True, threads=3)
    lcd, ls = sq.LocalClustering(dem, weighted=weighted).decode_batch(dets, return_stats=True, threads=2)
    np.testing.assert_array_equal(cc, lcd)
    np.testing.assert_array_equal(cs[:, 0], ls[:, 0])
    assert (ls[:, 2] == 0).all()
    # (pairs compared grow as the square of the events; stage passes with the diameter)
    assert cs[:, 1].sum() > ls[:, 1].sum() > 0
    assert failures(cc, obs) > 0


def test_unweighted_threshold_is_below_weighted():
    # The LCD paper's Fig. 6: unweighted, the curves cross near 0.55%; weighted, near 0.65%,
    # close to matching's. At p = 0.6% distance 7 is worse than 5 unweighted, better weighted.
    shots = 20_000
    rates = {}
    for d in (5, 7):
        c, dem, dets, obs = memory(d, 0.006, shots, 10 + d, basis="x")
        for weighted in (False, True):
            rates[d, weighted] = per_round(failures(sq.LocalClustering(dem, weighted=weighted).decode_batch(dets, threads=4), obs), shots, d)
        rates[d, "matching"] = per_round(failures(sq.Matching(dem).decode_batch(dets, threads=4), obs), shots, d)
    assert rates[7, False] > rates[5, False] * 1.04, rates
    assert rates[7, True] < rates[5, True] * 0.96, rates
    assert rates[7, "matching"] < rates[7, True] < rates[7, False], rates
    # Below threshold, unweighted too gains with distance.
    _, dem5, d5, o5 = memory(5, 0.002, 10_000, 3)
    _, dem7, d7, o7 = memory(7, 0.002, 10_000, 4)
    f5 = failures(sq.CollisionClustering(dem5).decode_batch(d5, threads=4), o5)
    f7 = failures(sq.CollisionClustering(dem7).decode_batch(d7, threads=4), o7)
    assert f7 < 0.8 * f5, (f7, f5)


def leaky(circuit, pl):
    """Leakage, and its seeping back, at rate ``pl`` on both qubits of every two-qubit gate."""
    items = []
    for it in circuit.flattened()._items():
        items.append(it)
        if it[0] == "op" and gate_data(it[1]).is_two_qubit_gate and gate_data(it[1]).is_unitary:
            items.append(("op", "I_ERROR", f"LEAK(p={pl})", [], list(it[4])))
            items.append(("op", "I_ERROR", f"SEEP(p={pl})", [], list(it[4])))
    return Circuit._of_items(items)


def test_heralded_leakage_pre_grown_beats_decoding_without():
    c = leaky(sq.memory_circuit(distance=5, rounds=5, p=0.001), 0.001)
    aware = sq.LocalClustering.from_circuit(c)
    plain = sq.LocalClustering.from_circuit(c, leakage_aware=False)
    assert len(aware.adaptivity_sizes) == c.num_measurements and max(aware.adaptivity_sizes) > 10
    assert plain.adaptivity_sizes == []
    dets, obs, heralds = c.compile_leakage_sampler(seed=5).sample(6000, threads=4)
    assert heralds.any()
    pred, stats = aware.decode_batch(dets, heralds=heralds, return_stats=True, threads=4)
    with_heralds = failures(pred, obs)
    without = failures(plain.decode_batch(dets, threads=4), obs)
    assert with_heralds < 0.6 * without, (with_heralds, without)
    assert ((stats[:, 2] > 0) == heralds.any(axis=1)).all()
    # Weighted too; and the heralds are ignored when not given.
    weighted = sq.LocalClustering.from_circuit(c, weighted=True)
    assert failures(weighted.decode_batch(dets, heralds=heralds, threads=4), obs) < with_heralds
    np.testing.assert_array_equal(aware.decode_batch(dets, threads=4), plain.decode_batch(dets, threads=4))
    # Bit-packed heralds, and pickling, keep the map.
    packed = np.packbits(heralds, axis=1, bitorder="little")
    np.testing.assert_array_equal(aware.decode_batch(dets, heralds=packed, bit_packed_heralds=True, threads=4), pred)
    again = pickle.loads(pickle.dumps(aware))
    np.testing.assert_array_equal(again.decode_batch(dets[:500], heralds=heralds[:500]), pred[:500])


def test_an_explicit_adaptivity_map():
    # A repetition line: B - D0 - D1 - D2 - B, its left boundary edge the observable.
    dem = sq.DetectorErrorModel("error(0.1) D0 L0\nerror(0.1) D0 D1\nerror(0.1) D1 D2\nerror(0.1) D2")
    lcd = sq.LocalClustering(dem, adaptivity=[[(0, 1), (1, 2)], [(0, 2)], []])
    assert lcd.adaptivity_sizes == [2, 0, 0]
    shots = np.array([[1, 0, 1]] * 3, dtype=bool)
    heralds = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 1]], dtype=bool)
    pred, stats = lcd.decode_batch(shots, heralds=heralds, return_stats=True)
    assert pred[:, 0].tolist() == [1, 0, 1]
    assert stats[:, 2].tolist() == [0, 2, 0]
    with pytest.raises(ValueError, match="heralds rows have 2 bits, not 3"):
        lcd.decode_batch(shots, heralds=heralds[:, :2])
    with pytest.raises(ValueError, match="outside"):
        sq.LocalClustering(dem, adaptivity=[[(0, 4)]])


def test_errors_and_options():
    dem = sq.DetectorErrorModel("error(0.1) D0 D1 L0")
    for cls in (sq.CollisionClustering, sq.LocalClustering):
        dec = cls(dem)
        assert list(dec.decode(np.array([1, 1], dtype=bool))) == [1]
        with pytest.raises(ValueError, match="no correction explains"):
            dec.decode(np.array([1, 0], dtype=bool))
        with pytest.raises(ValueError, match="weight_resolution"):
            cls(dem, weighted=True, weight_resolution=0)
        assert pickle.loads(pickle.dumps(cls(dem, weighted=True, weight_resolution=4))).decode(np.array([1, 1], dtype=bool))[0] == 1


@pytest.mark.parametrize("kind", ["collision_clustering", "local_clustering"])
def test_sinter_kinds(kind):
    from stabilizer_qec import sinter

    c = sq.memory_circuit(distance=3, rounds=3, p=0.003)
    stats = sinter.collect(tasks=[sinter.Task(circuit=c, json_metadata={})], decoders=[kind], num_workers=1, max_shots=500)
    assert stats[0].shots == 500 and stats[0].errors < 40

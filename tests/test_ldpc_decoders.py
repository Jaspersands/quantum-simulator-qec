"""ldpc's other decoders against ldpc 2.4.1 itself, bit for bit: BpFlipDecoder (bit flipping
with random flips from a seed, then BP), LsdDecoder (LSD alone, given bit weights), and
UnionFindDecoder and BeliefFindDecoder (peeling and inversion), including weights with many
ties, which ldpc's C++ sorts order."""

from __future__ import annotations

import pickle

import numpy as np
import pytest

import stabilizer_qec as sq

ldpc = pytest.importorskip("ldpc", minversion="2.4")


def codes(seed: int, count: int, m=(5, 15), n=(10, 32), density=0.3):
    rng = np.random.default_rng(seed)
    for _ in range(count):
        mm, nn = int(rng.integers(*m)), int(rng.integers(*n))
        if mm == nn:
            continue
        yield rng, (rng.random((mm, nn)) < density).astype(np.uint8)


@pytest.mark.parametrize("pflip", [(0, 1), (2, 5), (1, 99)])
def test_bp_flip_decoder(pflip):
    from ldpc.bp_flip import BpFlipDecoder as Theirs

    frequency, seed = pflip
    for rng, pcm in codes(13, 20):
        kw = dict(error_rate=0.05, max_iter=int(rng.integers(1, 6)), flip_iterations=int(rng.integers(0, 6)), pflip_frequency=frequency, pflip_seed=seed)
        ours, theirs = sq.BpFlipDecoder(pcm, **kw), Theirs(pcm, **kw)
        for _ in range(6):
            syndrome = (pcm @ (rng.random(pcm.shape[1]) < 0.15) % 2).astype(np.uint8)
            a, b = ours.decode(syndrome), theirs.decode(syndrome)
            assert np.array_equal(a, b)
            assert ours.converge == theirs.converge
            if syndrome.any():
                assert ours.iter == theirs.iter


@pytest.mark.parametrize("lsd", [(1, "lsd_0", 0), (3, "lsd_0", 0), (2, "lsd_e", 2), (1, "lsd_cs", 4), (0, "lsd_cs", 2)])
def test_lsd_decoder(lsd):
    from ldpc.lsd_decoder import LsdDecoder as Theirs

    bits, method, order = lsd
    agree = total = 0
    for rng, pcm in codes(17, 12, m=(20, 50), n=(60, 120), density=0.08):
        ours, theirs = sq.LsdDecoder(pcm, bits_per_step=bits, lsd_method=method, lsd_order=order), Theirs(pcm, bits_per_step=bits, lsd_method=method, lsd_order=order)
        assert (ours.lsd_method, ours.lsd_order) == (theirs.lsd_method, theirs.lsd_order)
        for _ in range(8):
            syndrome = (pcm @ (rng.random(pcm.shape[1]) < 0.1) % 2).astype(np.uint8)
            # Weights from a few values: long sorts full of equal keys.
            weights = rng.integers(0, 3, pcm.shape[1]).astype(float) if rng.random() < 0.7 else rng.random(pcm.shape[1])
            a, b = ours.decode(syndrome, weights), theirs.decode(syndrome, weights)
            assert np.array_equal(pcm @ a % 2, syndrome)
            same = np.array_equal(a, b)
            # A difference only where ldpc merges clusters in an order taken from addresses.
            assert same or ours.last_tied
            agree += same
            total += 1
    assert agree >= 0.95 * total


def test_lsd_decoder_arguments_and_pickling():
    pcm = np.array([[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1]], dtype=np.uint8)
    dec = sq.LsdDecoder(pcm, osd_method="lsd_cs", osd_order=2)
    assert (dec.lsd_method, dec.lsd_order) == ("LSD_CS", 2)
    s = np.array([1, 0, 1], dtype=np.uint8)
    w = [0.1, 0.5, 0.5, 0.1]
    first = dec.decode(s, w)
    assert np.array_equal(pickle.loads(pickle.dumps(dec)).decode(s, w), first)
    for bad in (dict(lsd_order=-1), dict(lsd_method="osd_x"), dict(lsd_method=5)):
        with pytest.raises(ValueError):
            sq.LsdDecoder(pcm, **bad)
    with pytest.raises(ValueError):
        dec.decode(s, w[:3])
    flip = sq.BpFlipDecoder(pcm, error_rate=0.1, pflip_seed=3)
    assert np.array_equal(pickle.loads(pickle.dumps(flip)).decode(s), flip.decode(s))


def graphs(seed: int, count: int):
    """Matching-like matrices: each column has one check (the boundary) or two."""
    rng = np.random.default_rng(seed)
    for _ in range(count):
        m = int(rng.integers(6, 25))
        n = int(rng.integers(m, 3 * m))
        pcm = np.zeros((m, n), dtype=np.uint8)
        for j in range(n):
            pcm[rng.choice(m, 1 if rng.random() < 0.15 else 2, replace=False), j] = 1
        yield rng, pcm


def ldpc_codes(seed: int, count: int):
    rng = np.random.default_rng(seed)
    for _ in range(count):
        m, n = int(rng.integers(10, 30)), int(rng.integers(30, 80))
        pcm = (rng.random((m, n)) < 0.1).astype(np.uint8)
        for j in np.flatnonzero(pcm.sum(0) == 0):
            pcm[rng.integers(m), j] = 1
        yield rng, pcm


def check_agreement(pairs) -> None:
    agree = total = 0
    for ours, theirs, decode in pairs:
        a = decode(ours)
        if getattr(ours, "last_stuck", False):
            continue  # ldpc's peeling never returns here, or returns a wrong correction
        b = decode(theirs)
        same = np.array_equal(a, b)
        assert same or ours.last_tied
        agree += same
        total += 1
    assert total > 0 and agree >= 0.95 * total


@pytest.mark.parametrize("planar", [True, False])
def test_union_find_decoder(planar):
    from ldpc import UnionFindDecoder as Theirs

    def pairs():
        for rng, pcm in (graphs(3, 25) if planar else ldpc_codes(4, 25)):
            for method in (["", "inversion"] if planar else ["inversion"]):
                ours, theirs = sq.UnionFindDecoder(pcm, uf_method=method), Theirs(pcm, method)
                for _ in range(6):
                    s = (pcm @ (rng.random(pcm.shape[1]) < 0.08) % 2).astype(np.uint8)
                    llrs = rng.integers(0, 4, pcm.shape[1]).astype(float) if rng.random() < 0.5 else None
                    bps = int(rng.integers(0, 3))
                    yield ours, theirs, lambda d, s=s, llrs=llrs, bps=bps: d.decode(s, llrs, bps)

    check_agreement(pairs())


def test_union_find_peeling_explains_what_ldpcs_does_not():
    # ldpc 2.4.1 peels the boundary through the last boundary bit it meets, in its spanning tree
    # or not: on the distance-5 rotated surface code (X errors, parallel faults merged) 78 of the
    # 4096 syndromes get corrections that do not explain them. Ours peel through the tree's
    # boundary bit there, and elsewhere equal ldpc's.
    from ldpc import UnionFindDecoder as Theirs
    from stabilizer_qec.conformance import population

    inst = population()[5]
    pcm = np.zeros((inst.num_detectors, inst.num_faults), dtype=np.uint8)
    for j, (dets, _, _) in enumerate(inst.faults):
        pcm[list(dets), j] = 1
    ours, theirs = sq.UnionFindDecoder(pcm, uf_method="peeling"), Theirs(pcm, uf_method="")
    wrong = stuck = 0
    for k in range(1 << pcm.shape[0]):
        s = ((k >> np.arange(pcm.shape[0])) & 1).astype(np.uint8)
        a, b = ours.decode(s), theirs.decode(s)
        assert np.array_equal(pcm @ a % 2, s)
        wrong += not np.array_equal(pcm @ b % 2, s)
        stuck += ours.last_stuck
        assert ours.last_stuck or np.array_equal(a, b)
    assert wrong == stuck == 78


@pytest.mark.parametrize("planar", [True, False])
def test_belief_find_decoder(planar):
    from ldpc import BeliefFindDecoder as Theirs

    def pairs():
        for rng, pcm in (graphs(5, 20) if planar else ldpc_codes(6, 20)):
            for method in (["peeling", "inversion"] if planar else ["inversion"]):
                kw = dict(error_rate=0.05, max_iter=int(rng.integers(1, 4)), uf_method=method, bits_per_step=int(rng.integers(0, 3)))
                ours, theirs = sq.BeliefFindDecoder(pcm, **kw), Theirs(pcm, **kw)
                assert (ours.uf_method, ours.bp_method, ours.max_iter) == (theirs.uf_method, theirs.bp_method, theirs.max_iter)
                for _ in range(6):
                    s = (pcm @ (rng.random(pcm.shape[1]) < 0.08) % 2).astype(np.uint8)
                    yield ours, theirs, lambda d, s=s: d.decode(s)

    check_agreement(pairs())


def test_union_find_arguments():
    from ldpc import BeliefFindDecoder as Theirs

    triangle = np.array([[1, 1, 0], [0, 1, 1], [1, 0, 1]], dtype=np.uint8)
    dense = np.array([[1, 1], [1, 0], [1, 1]], dtype=np.uint8)  # a 3-check column
    with pytest.raises(RuntimeError):
        sq.UnionFindDecoder(np.array([[1, 0], [1, 0]], dtype=np.uint8))  # an empty column
    with pytest.raises(TypeError):
        sq.UnionFindDecoder([[1, 1]])
    with pytest.raises(RuntimeError):
        sq.UnionFindDecoder(dense).decode(np.array([1, 0, 1]))  # peeling
    dec = sq.UnionFindDecoder(triangle, uf_method="inversion")
    out = dec.decode(np.array([1, 1, 0], dtype=np.uint8))
    assert np.array_equal(triangle @ out % 2, [1, 1, 0]) and np.array_equal(dec.decoding, out)
    assert dec.decode(np.zeros(3, dtype=np.uint8)).sum() == 0
    with pytest.raises(ValueError):
        dec.decode(np.array([1, 0, 0]))  # odd parity: nothing explains it (ldpc loops)
    with pytest.raises(ValueError):
        dec.decode(np.zeros(4))
    with pytest.raises(ValueError):
        dec.decode(np.array([1, 1, 0]), llrs=np.zeros(2))
    copy = pickle.loads(pickle.dumps(dec))
    assert np.array_equal(copy.decode(np.array([0, 1, 1])), dec.decode(np.array([0, 1, 1])))
    for bad in ("inv", "peeling"):
        kw = dict(error_rate=0.1, uf_method=bad)
        with pytest.raises(ValueError):
            Theirs(dense, **kw)
        with pytest.raises(ValueError):
            sq.BeliefFindDecoder(dense, **kw)
    with pytest.raises(TypeError):
        sq.BeliefFindDecoder(dense, error_rate=0.1, uf_method=3)
    bf = sq.BeliefFindDecoder(dense, error_rate=0.1, uf_method="Matrix")
    assert bf.uf_method == "inversion" and bf.bits_per_step == 2
    assert bf.decode(np.zeros(3, dtype=np.uint8)).sum() == 0 and bf.converge
    assert np.array_equal(pickle.loads(pickle.dumps(bf)).decode(np.array([1, 0, 1])), bf.decode(np.array([1, 0, 1])))

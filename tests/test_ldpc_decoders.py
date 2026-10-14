"""ldpc's other decoders against ldpc 2.4.1 itself, bit for bit: BpFlipDecoder (bit flipping
with random flips from a seed, then BP) and LsdDecoder (LSD alone, given bit weights),
including weights with many ties, which ldpc's C++ sorts order."""

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

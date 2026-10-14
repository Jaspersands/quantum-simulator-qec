"""ldpc's BP decoders' options against ldpc 2.4.1 itself, bit for bit: every schedule
(parallel, serial, serial-relative, random serial with a seed), both input types, the
soft-information decoder, the effective defaults and the properties."""

from __future__ import annotations

import numpy as np
import pytest

import stabilizer_qec as sq

ldpc = pytest.importorskip("ldpc", minversion="2.4")


def codes(seed: int, count: int):
    rng = np.random.default_rng(seed)
    for _ in range(count):
        m, n = int(rng.integers(5, 15)), int(rng.integers(10, 32))
        if m == n:
            continue
        pcm = (rng.random((m, n)) < 0.3).astype(np.uint8)
        yield rng, pcm, list(rng.uniform(0.01, 0.2, n))


def same_run(ours, theirs, a, b) -> bool:
    return (
        np.array_equal(a, b)
        and ours.converge == theirs.converge
        and ours.iter == theirs.iter
        and np.allclose(ours.log_prob_ratios, theirs.log_prob_ratios, rtol=1e-12, atol=1e-12, equal_nan=True)
        and np.array_equal(ours.serial_schedule_order, theirs.serial_schedule_order)
    )


OPTIONS = [
    dict(),
    dict(bp_method="minimum_sum", ms_scaling_factor=0.625),
    dict(schedule="serial"),
    dict(schedule="serial", bp_method="ms", ms_scaling_factor=0.0),
    dict(schedule="serial_relative", bp_method="ms"),
    dict(schedule="serial_relative"),
    dict(schedule="serial", random_serial_schedule=True, random_schedule_seed=42),
    dict(schedule="serial", random_serial_schedule=True, random_schedule_seed=7, bp_method="minimum_sum", ms_scaling_factor=0.0),
]


@pytest.mark.parametrize("options", OPTIONS, ids=[str(o) for o in OPTIONS])
def test_bp_decoder(options):
    for rng, pcm, channel in codes(3, 25):
        kw = dict(error_channel=channel, max_iter=int(rng.integers(1, 8)), **options)
        ours, theirs = sq.BpDecoder(pcm, **kw), ldpc.BpDecoder(pcm, **kw)
        for _ in range(6):
            error = (rng.random(pcm.shape[1]) < 0.15).astype(np.uint8)
            kind = "received_vector" if rng.random() < 0.3 else "syndrome"
            ours.input_vector_type = theirs.input_vector_type = kind
            vector = error if kind == "received_vector" else (pcm @ error % 2).astype(np.uint8)
            a, b = ours.decode(vector), theirs.decode(vector)
            assert same_run(ours, theirs, a, b), (options, kind)


@pytest.mark.parametrize("options", OPTIONS[2:], ids=[str(o) for o in OPTIONS[2:]])
def test_bposd_and_bplsd_schedules(options):
    for rng, pcm, channel in codes(5, 12):
        kw = dict(error_channel=channel, max_iter=int(rng.integers(1, 5)), **{k: v for k, v in options.items() if k != "random_serial_schedule"})
        pairs = [(sq.BpOsdDecoder(pcm, osd_method="osd_cs", osd_order=2, **kw), ldpc.BpOsdDecoder(pcm, osd_method="osd_cs", osd_order=2, **kw)), (sq.BpLsdDecoder(pcm, **kw), ldpc.BpLsdDecoder(pcm, **kw))]
        for ours, theirs in pairs:
            if options.get("random_serial_schedule"):
                ours.random_serial_schedule = theirs.random_serial_schedule = True
            for _ in range(5):
                syndrome = (pcm @ (rng.random(pcm.shape[1]) < 0.15) % 2).astype(np.uint8)
                a, b = ours.decode(syndrome), theirs.decode(syndrome)
                if not syndrome.any():
                    continue
                assert np.array_equal(ours.bp_decoding, theirs.bp_decoding)
                assert (ours.converge, ours.iter) == (theirs.converge, theirs.iter)
                assert np.allclose(ours.log_prob_ratios, theirs.log_prob_ratios, rtol=1e-12, atol=1e-12, equal_nan=True)
                if isinstance(ours, sq.BpOsdDecoder) or not getattr(ours, "last_tied", False):
                    assert np.array_equal(a, b)


@pytest.mark.parametrize("cutoff", [np.inf, 2.0, 0.5])
@pytest.mark.parametrize("random", [False, True])
def test_soft_info_decoder(cutoff, random):
    from ldpc.bp_decoder import SoftInfoBpDecoder as Theirs

    for rng, pcm, channel in codes(9, 20):
        kw = dict(error_channel=channel, max_iter=6, cutoff=cutoff, sigma=1.0)
        if random:
            kw.update(random_serial_schedule=True, random_schedule_seed=11)
        ours, theirs = sq.SoftInfoBpDecoder(pcm, **kw), Theirs(pcm, **kw)
        for _ in range(3):
            syndrome = pcm @ (rng.random(pcm.shape[1]) < 0.15) % 2
            soft = (1 - 2 * syndrome) * rng.uniform(0.1, 3, pcm.shape[0])
            a, b = ours.decode(soft), theirs.decode(soft)
            assert same_run(ours, theirs, a, b)
            assert np.allclose(ours.soft_syndrome, theirs.soft_syndrome)


def test_defaults_and_properties():
    from ldpc.bp_decoder import SoftInfoBpDecoder as Theirs

    pcm = np.array([[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1]], dtype=np.uint8)
    pairs = [(sq.BpDecoder, ldpc.BpDecoder), (sq.BpOsdDecoder, ldpc.BpOsdDecoder), (sq.BpLsdDecoder, ldpc.BpLsdDecoder), (sq.SoftInfoBpDecoder, Theirs)]
    names = ["bp_method", "ms_scaling_factor", "max_iter", "schedule", "input_vector_type", "random_serial_schedule", "random_schedule_seed", "check_count", "bit_count"]
    for mine, theirs_cls in pairs:
        a, b = mine(pcm, error_rate=0.1), theirs_cls(pcm, error_rate=0.1)
        for name in names:
            assert getattr(a, name) == getattr(b, name), (mine.__name__, name)
        assert np.array_equal(a.channel_probs, b.channel_probs) and np.array_equal(a.serial_schedule_order, b.serial_schedule_order)
    a, b = sq.BpOsdDecoder(pcm, error_rate=0.1, osd_method="osd_e", osd_order=3), ldpc.BpOsdDecoder(pcm, error_rate=0.1, osd_method="osd_e", osd_order=3)
    assert (a.osd_method, a.osd_order) == (b.osd_method, b.osd_order)
    a.osd_method = b.osd_method = "osd_0"
    assert (a.osd_method, a.osd_order) == (b.osd_method, b.osd_order)
    c, d = sq.BpLsdDecoder(pcm, error_rate=0.1, lsd_method=1, lsd_order=2), ldpc.BpLsdDecoder(pcm, error_rate=0.1, lsd_method=1, lsd_order=2)
    assert (c.lsd_method, c.lsd_order) == (d.lsd_method, d.lsd_order)
    for bad in (dict(error_rate=0.1, schedule="sideways"), dict(error_rate=0.1, bp_method="x"), dict(error_rate=0.1, nonsense=1), dict()):
        for cls in (sq.BpDecoder, ldpc.BpDecoder):
            with pytest.raises(ValueError):
                cls(pcm, **bad)
    with pytest.raises(ValueError):
        sq.BpDecoder(np.eye(3, dtype=np.uint8), error_rate=0.1)  # auto input on a square matrix
    e = sq.BpDecoder(pcm, error_rate=0.1)
    e.update_channel_probs([0.1, 0.2, 0.3, 0.05])
    f = ldpc.BpDecoder(pcm, error_rate=0.1)
    f.update_channel_probs([0.1, 0.2, 0.3, 0.05])
    s = np.array([1, 0, 1], dtype=np.uint8)
    assert np.array_equal(e.decode(s), f.decode(s)) and np.allclose(e.log_prob_ratios, f.log_prob_ratios)

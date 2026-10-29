"""stabilizer_qec.sinter against sinter 1.16 itself: the same CSV text, strong ids, fits,
predictions and combined files for the same inputs, and the same error rates from collect."""

from __future__ import annotations

import collections
import contextlib
import io
import random

import numpy as np
import pytest

import stabilizer_qec as sq
from stabilizer_qec import sinter as ours

theirs = pytest.importorskip("sinter", minversion="1.16")
stim = pytest.importorskip("stim", minversion="1.16")


def random_stats(rng: random.Random, module, n: int = 12) -> list:
    out = []
    for k in range(n):
        shots = rng.randint(0, 10**6)
        errors = rng.randint(0, shots)
        discards = rng.randint(0, shots - errors)
        counts = collections.Counter({f"obs_mistake_mask={rng.choice('E_')}{rng.choice('E_')}": rng.randint(1, 50) for _ in range(rng.randint(0, 3))})
        out.append(module.TaskStats(
            strong_id=f"{rng.getrandbits(64):016x}",
            decoder=rng.choice(["pymatching", "bposd", 'odd,"name']),
            json_metadata=rng.choice([None, {"d": k, "p": rng.random()}, [1, "a"], "text"]),
            shots=shots, errors=errors, discards=discards, seconds=rng.choice([0.0, rng.random(), rng.random() * 9, rng.random() * 1e4]),
            custom_counts=counts,
        ))
    return out


@pytest.mark.parametrize("seed", range(10))
def test_csv_text_and_reading_it_back(tmp_path, seed):
    assert ours.CSV_HEADER == theirs.CSV_HEADER
    a, b = random_stats(random.Random(seed), ours), random_stats(random.Random(seed), theirs)
    assert [x.to_csv_line() for x in a] == [x.to_csv_line() for x in b]
    assert [repr(x).replace("stabilizer_qec.", "") for x in a] == [repr(x) for x in b]
    path = tmp_path / "stats.csv"
    path.write_text(ours.CSV_HEADER + "\n" + "\n".join(x.to_csv_line() for x in a) + "\n")
    read_a = ours.read_stats_from_csv_files(path)
    read_b = theirs.read_stats_from_csv_files(path)
    assert [x.to_csv_line() for x in read_a] == [x.to_csv_line() for x in read_b]


def test_combine_writes_the_same_file(tmp_path):
    a = random_stats(random.Random(7), ours, 30)
    # Repeat some ids, so that combining merges them.
    text = ours.CSV_HEADER + "\n" + "\n".join(x.to_csv_line() for x in a + a[:10]) + "\n"
    (tmp_path / "in.csv").write_text(text)
    outs = []
    from sinter._command._main import main as their_main

    from stabilizer_qec.sinter._command._main import main as our_main

    for main in (our_main, their_main):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            main(command_line_args=["combine", str(tmp_path / "in.csv")])
        outs.append(buf.getvalue())
    assert outs[0] == outs[1]


def test_strong_ids_are_sinters():
    c = stim.Circuit.generated("surface_code:rotated_memory_x", distance=3, rounds=3, after_clifford_depolarization=0.01)
    mask = np.packbits(np.arange(c.num_detectors) % 3 == 0, bitorder="little")
    for kw in ({}, {"postselection_mask": mask}):
        dem = c.detector_error_model(decompose_errors=True)
        a = ours.Task(circuit=sq.Circuit(str(c)), detector_error_model=sq.DetectorErrorModel(str(dem)), decoder="pymatching", json_metadata={"d": 3}, **kw)
        b = theirs.Task(circuit=c, detector_error_model=dem, decoder="pymatching", json_metadata={"d": 3}, **kw)
        assert a.strong_id() == b.strong_id()


@pytest.mark.parametrize("seed", range(30))
def test_fits(seed):
    rng = np.random.default_rng(seed)
    shots = int(rng.integers(1, 10**6))
    hits = int(rng.integers(0, shots + 1))
    factor = float(rng.choice([1e3, 10, 2]))
    a = ours.fit_binomial(num_shots=shots, num_hits=hits, max_likelihood_factor=factor)
    b = theirs.fit_binomial(num_shots=shots, num_hits=hits, max_likelihood_factor=factor)
    assert (a.low, a.best, a.high) == (b.low, b.best, b.high)
    p = rng.random(5)
    assert np.array_equal(ours.log_binomial(p=p, n=shots, hits=hits), theirs.log_binomial(p=p, n=shots, hits=hits))
    rate = float(rng.random())
    for pieces, values in ((3, 1), (0.5, 1), (7, 2)):
        assert ours.shot_error_rate_to_piece_error_rate(rate, pieces=pieces, values=values) == pytest.approx(theirs.shot_error_rate_to_piece_error_rate(rate, pieces=pieces, values=values), rel=1e-15)
    xs = np.sort(rng.random(int(rng.integers(3, 12))) * 10)
    ys = 2 * xs + rng.normal(size=len(xs))
    extra = float(rng.choice([1.0, 0.1, 5.0]))
    for name, kw in (("fit_line_slope", {}), ("fit_line_y_at_x", {"target_x": float(rng.random() * 10)})):
        a = getattr(ours, name)(xs=xs, ys=ys, max_extra_squared_error=extra, **kw)
        b = getattr(theirs, name)(xs=xs, ys=ys, max_extra_squared_error=extra, **kw)
        # sinter finds the constrained lines with scipy's iterative leastsq, here in closed form.
        for x, y in ((a.low, b.low), (a.best, b.best), (a.high, b.high)):
            assert x == pytest.approx(y, rel=1e-6, abs=1e-6)


def test_predictions_are_sinters(tmp_path):
    pymatching = pytest.importorskip("pymatching")
    c = stim.Circuit.generated("surface_code:rotated_memory_z", distance=5, rounds=5, after_clifford_depolarization=0.01)
    dem = c.detector_error_model(decompose_errors=True)
    dets = c.compile_detector_sampler(seed=5).sample(2000, bit_packed=True)
    a = ours.predict_observables_bit_packed(dem=sq.DetectorErrorModel(str(dem)), dets_bit_packed=dets, decoder="pymatching")
    b = theirs.predict_observables_bit_packed(dem=dem, dets_bit_packed=dets, decoder="pymatching")
    unpacked = np.unpackbits(dets, axis=1, bitorder="little", count=c.num_detectors).astype(bool)
    # PyMatching's sparse blossom, transliterated: its predictions and weights, ties included.
    _, wa = sq.Matching(sq.DetectorErrorModel(str(dem))).decode_batch(unpacked, return_weights=True)
    _, wb = pymatching.Matching(dem).decode_batch(unpacked, return_weights=True)
    assert np.array_equal(wa, wb)
    assert np.array_equal(a, b)
    c_ = ours.predict_observables(dem=sq.DetectorErrorModel(str(dem)), dets=unpacked, decoder="pymatching")
    assert np.array_equal(np.packbits(c_, axis=1, bitorder="little"), a)


def outcome(f):
    try:
        return f()
    except ValueError as e:
        return "ValueError: " + str(e)


@pytest.mark.parametrize("decoder", ["pymatching", "fusion_blossom", "vacuous"])
def test_malformed_detection_data_raises_as_sinter_does(decoder):
    # A uint8 array is bit-packed by contract: 0/1 uint8 detection events, or packed rows of the
    # wrong width, are refused by Stim's writer under sinter, never decoded as other shots.
    if decoder == "fusion_blossom":
        pytest.importorskip("fusion_blossom")
    c = stim.Circuit.generated("surface_code:rotated_memory_z", distance=3, rounds=3, after_clifford_depolarization=0.02)
    dem = c.detector_error_model(decompose_errors=True)
    dets = c.compile_detector_sampler(seed=7).sample(50)
    packed = np.packbits(dets, axis=1, bitorder="little")
    for name, call in [
        ("uint8 0/1", lambda m, d: m.predict_observables(dem=d, dets=dets.astype(np.uint8), decoder=decoder)),
        ("int64 0/1", lambda m, d: m.predict_observables(dem=d, dets=dets.astype(np.int64), decoder=decoder)),
        ("packed, 2 bytes short", lambda m, d: m.predict_observables_bit_packed(dem=d, dets_bit_packed=packed[:, :2], decoder=decoder)),
        ("packed, a byte long", lambda m, d: m.predict_observables_bit_packed(dem=d, dets_bit_packed=np.hstack([packed, packed[:, :1]]), decoder=decoder)),
    ]:
        a = outcome(lambda: call(ours, sq.DetectorErrorModel(str(dem))))
        b = outcome(lambda: call(theirs, dem))
        assert isinstance(b, str), (name, b)
        assert a == b if isinstance(a, str) else False, (name, "no error; predictions of shape", np.shape(a))


def test_shot_writer_refuses_what_stims_refuses(tmp_path):
    cases = [
        (np.ones((2, 6), np.uint8), 6), (np.ones((2, 2), np.uint8), 6), (np.ones((2, 1), np.uint8), 6), (np.full((2, 1), 0xFF, np.uint8), 6),
        (np.ones((2, 0), np.uint8), 0), (np.ones((2, 2), np.uint8), 9), (np.ones((2, 5), bool), 6), (np.ones((2, 6), bool), 6),
        (np.ones((2, 6), np.int64), 6), (np.ones((2, 1), np.int8), 6), (np.ones((2, 6), float), 6), (np.ones(6, np.uint8), 6),
        (np.ones(6, bool), 6), (np.ones((1, 2, 6), bool), 6), ([[True] * 6], 6),
    ]
    for data, n in cases:
        a = outcome(lambda: sq.write_shot_data_file(data=data, path=tmp_path / "a", format="b8", num_detectors=n))
        b = outcome(lambda: stim.write_shot_data_file(data=data, path=str(tmp_path / "b"), format="b8", num_detectors=n))
        assert a == b, (data, n)
        if a is None:
            assert (tmp_path / "a").read_bytes() == (tmp_path / "b").read_bytes()


def test_collect_agrees_with_sinter_statistically():
    c = stim.Circuit.generated("repetition_code:memory", distance=5, rounds=5, after_clifford_depolarization=0.06)
    a = ours.collect(tasks=[ours.Task(circuit=sq.Circuit(str(c)), json_metadata={})], decoders=["pymatching", "matching", "union_find"], num_workers=2, max_shots=10_000, max_errors=10**9)
    b = theirs.collect(tasks=[theirs.Task(circuit=c, json_metadata={})], decoders=["pymatching"], num_workers=2, max_shots=10_000, max_errors=10**9)
    reference = b[0].errors / b[0].shots
    for s in a:
        assert s.shots >= 10_000
        rate = s.errors / s.shots
        sigma = (reference * (1 - reference) * (1 / s.shots + 1 / b[0].shots)) ** 0.5
        assert abs(rate - reference) < 5 * sigma, (s.decoder, rate, reference)

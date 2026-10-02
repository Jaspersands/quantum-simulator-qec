"""sinter's sweeps with this package's decoders, and with its whole pipeline."""

from __future__ import annotations

import pickle

import numpy as np
import pytest

import stabilizer_qec as sq

sinter = pytest.importorskip("sinter")
stim = pytest.importorskip("stim")
pytest.importorskip("pymatching")

from stabilizer_qec import sinter as sq_sinter  # noqa: E402

try:
    # sinter 1.15 asserts its error counts are Python ints, which numpy 2.5's count_nonzero no
    # longer returns: there sinter fails with any decoder, its own included.
    sinter.AnonTaskStats(shots=1, errors=np.count_nonzero(np.array([True])))
except AssertionError:
    pytest.skip(f"sinter {sinter.__version__} cannot count errors with numpy {np.__version__}", allow_module_level=True)


def task(d=3, p=0.006, **kwargs):
    c = stim.Circuit.generated("surface_code:rotated_memory_z", distance=d, rounds=d, after_clifford_depolarization=p, before_measure_flip_probability=p, after_reset_flip_probability=p)
    return sinter.Task(circuit=c, json_metadata={"d": d, "p": p}, **kwargs)


def rates(stats):
    return {s.decoder: (s.errors / (s.shots - s.discards), s) for s in stats}


def test_decoders_and_samplers_agree_with_pymatching_in_a_sweep():
    stats = sinter.collect(
        num_workers=2,
        tasks=[task()],
        decoders=["pymatching", "sq_matching", "sq_sim_matching", "sq_correlated_matching"],
        custom_decoders=sq_sinter.sinter_decoders(),
        max_shots=40_000,
        max_errors=10**9,
    )
    r = rates(stats)
    ref, ref_stats = r["pymatching"]
    assert ref_stats.shots >= 40_000 and 0.005 < ref < 0.1
    sigma = np.sqrt(ref * (1 - ref) / ref_stats.shots)
    for name in ("sq_matching", "sq_sim_matching"):
        rate, s = r[name]
        assert s.shots >= 40_000 and abs(rate - ref) < 5 * np.sqrt(2) * sigma, (name, rate, ref)
    assert r["sq_correlated_matching"][0] < ref + 5 * sigma


def test_belief_matching_and_bposd_run_under_sinter():
    stats = sinter.collect(
        num_workers=1,
        tasks=[task(p=0.004)],
        decoders=["sq_belief_matching", "sq_bposd", "sq_sim_bposd"],
        custom_decoders={**sq_sinter.decoders(), **sq_sinter.samplers(), "sq_bposd": sq_sinter.Decoder("bposd", max_iter=10, osd_order=2), "sq_sim_bposd": sq_sinter.Sampler("bposd", max_iter=10, osd_order=2)},
        max_shots=3000,
        max_errors=10**9,
    )
    for name, (rate, s) in rates(stats).items():
        assert s.shots >= 3000 and rate < 0.05, name


def test_postselection_discards_as_sinter_discards():
    # The reference is the rate Stim's own shots fire a postselected detector. (Not sinter's
    # "pymatching" path: some sinter versions mis-size its predictions when a small ramp-up
    # batch is discarded whole.)
    mask = np.arange(24) < 4
    t = task(p=0.004, postselection_mask=np.packbits(mask, bitorder="little"))
    stats = sinter.collect(num_workers=1, tasks=[t], decoders=["sq_sim_matching"], custom_decoders=sq_sinter.samplers(), max_shots=20_000, max_errors=10**9)
    (s,) = stats
    want = t.circuit.compile_detector_sampler(seed=5).sample(100_000)[:, mask].any(axis=1).mean()
    sigma = np.sqrt(want * (1 - want) * (1 / s.shots + 1 / 100_000))
    assert s.shots >= 20_000 and want > 0.01 and abs(s.discards / s.shots - want) < 6 * sigma
    # A batch discarded whole is still a batch.
    compiled = sq_sinter.Sampler("matching").compiled_sampler_for_task(task(p=0.004, postselection_mask=np.packbits(np.ones(24, bool), bitorder="little")))
    out = compiled.sample(64)
    assert out.shots == 64 and out.errors == 0 and out.discards >= 0


def test_predict_observables_through_files_matches_decoding_here():
    c = task().circuit
    dem = c.detector_error_model(decompose_errors=True)
    dets, _ = c.compile_detector_sampler(seed=2).sample(500, separate_observables=True)
    got = sinter.predict_observables(dem=dem, dets=dets, decoder="sq_matching", custom_decoders=sq_sinter.decoders())
    assert np.array_equal(got, sq.Matching(dem).decode_batch(dets).astype(bool))


def test_adapters_pickle_and_check_their_kind():
    for obj in (sq_sinter.Decoder("bposd", osd_order=3), sq_sinter.Sampler("belief_matching", max_bp_iters=5)):
        again = pickle.loads(pickle.dumps(obj))
        assert (again.kind, again.options) == (obj.kind, obj.options)
    with pytest.raises(ValueError, match="kind must be one of"):
        sq_sinter.Decoder("unionfind")
    assert set(sq_sinter.sinter_decoders()) == {f"sq_{k}" for k in sq_sinter.KINDS} | {f"sq_sim_{k}" for k in sq_sinter.KINDS}

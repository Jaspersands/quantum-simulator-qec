"""stabilizer_qec.relay_bp against IBM's relay_bp 0.2.2: every decoder class in every number type
(f32, f64, i8–i64, fixed point), with data scaling, bounds, memory, explicit or drawn memory
strengths and each stopping criterion, gives relay_bp's decodings, checks, posteriors and
iteration counts bit for bit, decode after decode; and so do the observable runner (sequential,
parallel, from known errors), the Relay log and the sinter decoders."""

from __future__ import annotations

import os
import pathlib

import numpy as np
import pytest

from conftest import reference

relay_bp = reference("relay_bp")

import stabilizer_qec.relay_bp as ours  # noqa: E402


def same(a, b) -> bool:
    return (
        np.array_equal(a.decoding, b.decoding)
        and np.array_equal(a.decoded_detectors, b.decoded_detectors)
        and np.array_equal(a.posterior_ratios, b.posterior_ratios, equal_nan=True)
        and (a.success, a.iterations, a.max_iter) == (b.success, b.iterations, b.max_iter)
    )


def codes(seed: int, count: int):
    rng = np.random.default_rng(seed)
    for case in range(count):
        m, n = int(rng.integers(6, 18)), int(rng.integers(12, 36))
        h = (rng.random((m, n)) < 0.2).astype(np.uint8)
        p = rng.uniform(0.005, 0.08, n)
        if case % 5 == 0:
            p[rng.integers(n)] = 0.0
        shots = [((h @ (rng.random(n) < 0.08)) % 2).astype(np.uint8) for _ in range(5)]
        yield case, rng, h, p, shots


def run_both(make, shots) -> None:
    theirs, mine = make(relay_bp), make(ours)
    for s in shots:
        assert same(theirs.decode_detailed(s), mine.decode_detailed(s))


@pytest.mark.parametrize("kind", ["F32", "F64", "I32", "I64"])
def test_relay_decoders(kind):
    for case, rng, h, p, shots in codes(11, 24):
        scale = None if kind[0] == "F" else float(rng.choice([4.0, 16.0, 100.0]))
        options = dict(
            alpha=[None, 0.0, 0.625, -1.0][case % 4],
            gamma0=None if case % 7 == 6 else 0.1,
            data_scale_value=scale,
            max_data_value=None if case % 3 else float(rng.choice([50.0, 1000.0])),
            pre_iter=8,
            num_sets=10,
            set_max_iter=6,
            stop_nconv=1 + case % 3,
            stopping_criterion=["nconv", "pre_iter", "all", "unknown"][case % 4],
            explicit_gammas=None if case % 6 else rng.uniform(-0.2, 0.6, (12, h.shape[1])),
            seed=case,
        )
        run_both(lambda m: getattr(m, "RelayDecoder" + kind)(h, p, **options), shots)


@pytest.mark.parametrize("kind", ["F32", "F64", "I8", "I16", "I32", "I64", "Fixed"])
def test_min_sum_decoders(kind):
    import scipy.sparse

    for case, rng, h, p, shots in codes(13, 24):
        scale = None if kind[0] == "F" else float(rng.choice([2.0, 4.0] if kind == "I8" else [2.0, 8.0, 30.0]))
        options = dict(
            max_iter=(case % 4) * 5,
            alpha=[None, 0.0, 0.625, -1.0][case % 4],
            gamma0=None if case % 2 else 0.2,
            data_scale_value=scale,
            max_data_value=None if case % 3 else float(rng.choice([7.0, 100.0])),
            frac_bits=int(rng.integers(0, 8)) if kind == "Fixed" else None,
        )
        matrix = scipy.sparse.csc_matrix(h) if case % 2 else h
        run_both(lambda m: getattr(m, "MinSumBPDecoder" + kind)(matrix, p, **options), shots)


def test_batches_and_observable_runner():
    rng = np.random.default_rng(3)
    h = (rng.random((12, 30)) < 0.2).astype(np.uint8)
    obs = (rng.random((2, 30)) < 0.3).astype(np.uint8)
    p = rng.uniform(0.01, 0.05, 30)
    errors = (rng.random((20, 30)) < 0.07).astype(np.uint8)
    dets = (errors @ h.T % 2).astype(np.uint8)
    t, m = relay_bp.RelayDecoderF64(h, p, num_sets=20, set_max_iter=10, seed=5), ours.RelayDecoderF64(h, p, num_sets=20, set_max_iter=10, seed=5)
    assert np.array_equal(t.decode_batch(dets), m.decode_batch(dets))
    for x, y in zip(t.decode_detailed_batch(dets), m.decode_detailed_batch(dets)):
        assert same(x, y)
    tr, mr = relay_bp.ObservableDecoderRunner(t, obs, True), ours.ObservableDecoderRunner(m, obs, True)
    kw = dict(progress_bar=False)
    for parallel in (False, True):
        assert np.array_equal(tr.decode_observables_batch(dets, parallel=parallel, **kw), mr.decode_observables_batch(dets, parallel=parallel, **kw))
        assert np.array_equal(tr.decode_batch(dets, parallel=parallel, **kw), mr.decode_batch(dets, parallel=parallel, **kw))
        assert np.array_equal(tr.from_errors_decode_observables_batch(errors, parallel=parallel, **kw), mr.from_errors_decode_observables_batch(errors, parallel=parallel, **kw))
        for x, y in zip(tr.from_errors_decode_observables_detailed_batch(errors, parallel=parallel, **kw), mr.from_errors_decode_observables_detailed_batch(errors, parallel=parallel, **kw)):
            for attr in ("converged", "iterations", "error_detected", "error_mismatch_detected", "unconverged_no_error", "better_decoding_quality_error", "worse_decoding_quality_error"):
                assert getattr(x, attr) == getattr(y, attr), attr
            assert np.array_equal(x.observables, y.observables) and same(x.physical_decode_result, y.physical_decode_result)
        for x, y in zip(tr.decode_observables_detailed_batch(dets, parallel=parallel, **kw), mr.decode_observables_detailed_batch(dets, parallel=parallel, **kw)):
            assert x.error_detected is None and y.error_detected is None and np.array_equal(x.observables, y.observables)
    assert np.array_equal(tr.compute_observables(errors[0]), mr.compute_observables(errors[0]))
    assert same(tr.decode_detailed(dets[0]), mr.decode_detailed(dets[0]))
    assert np.array_equal(tr.decode_observables(dets[1]), mr.decode_observables(dets[1]))


def test_relay_log(tmp_path: pathlib.Path, monkeypatch):
    rng = np.random.default_rng(8)
    h = (rng.random((10, 24)) < 0.2).astype(np.uint8)
    p = rng.uniform(0.01, 0.06, 24)
    dets = [((h @ (rng.random(24) < 0.1)) % 2).astype(np.uint8) for _ in range(6)]
    texts = []
    for k, module in enumerate((relay_bp, ours)):
        d = tmp_path / str(k)
        d.mkdir()
        monkeypatch.chdir(d)
        dec = module.RelayDecoderF64(h, p, num_sets=8, set_max_iter=5, pre_iter=6, stopping_criterion="all", gamma_dist_interval=(-0.24, 0.66), logging=True, seed=2)
        for s in dets:
            dec.decode(s)
        texts.append((d / "relay_logging.out").read_text())
    assert texts[0] == texts[1]


def test_sinter_decoders(tmp_path: pathlib.Path):
    stim = pytest.importorskip("stim")
    pytest.importorskip("beliefmatching")
    import relay_bp.stim as theirs_stim

    import stabilizer_qec as sq
    from stabilizer_qec.relay_bp.stim import sinter_decoders

    circuit = stim.Circuit.generated("surface_code:rotated_memory_x", distance=3, rounds=3, after_clifford_depolarization=0.006, before_measure_flip_probability=0.006)
    dem = circuit.detector_error_model()
    a = theirs_stim.CheckMatrices.from_dem(dem)
    b = ours.stim.CheckMatrices.from_dem(sq.DetectorErrorModel(str(dem)))
    assert (a.check_matrix != b.check_matrix).nnz == 0 and (a.observables_matrix != b.observables_matrix).nnz == 0
    assert np.allclose(a.error_priors, b.error_priors, rtol=1e-12, atol=0)
    dem_path = tmp_path / "model.dem"
    dem.to_file(dem_path)
    dets, _ = circuit.compile_detector_sampler(seed=3).sample(200, separate_observables=True, bit_packed=True)
    dets_path = tmp_path / "dets.b8"
    dets_path.write_bytes(dets.tobytes())
    mine = sinter_decoders()
    for name, dec in theirs_stim.sinter_decoders().items():
        out = []
        for k, d in enumerate((dec, mine[name])):
            path = tmp_path / f"{name}{k}.b8"
            d.decode_via_files(num_shots=200, num_dets=dem.num_detectors, num_obs=dem.num_observables, dem_path=dem_path, dets_b8_in_path=dets_path, obs_predictions_b8_out_path=path, tmp_dir=tmp_path)
            out.append(path.read_bytes())
        assert out[0] == out[1], name


def test_errors_and_types():
    h = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
    p = np.full(3, 0.1)
    with pytest.raises(ValueError):
        ours.MinSumBPDecoderFixed(h, p, frac_bits=32)
    with pytest.raises(ValueError):
        ours.MinSumBPDecoderI8(h, np.full(3, 1e-60))  # ln(1/p) does not fit an i8
    with pytest.raises(TypeError):
        import scipy.sparse

        ours.RelayDecoderF64(scipy.sparse.coo_matrix(h), p)
    d = ours.MinSumBPDecoderI16(h, p, data_scale_value=8.0, max_data_value=100.0)
    r = d.decode_detailed(np.array([1, 0], dtype=np.uint8))
    assert r.decoding.dtype == np.uint8 and r.posterior_ratios.dtype == np.float64 and r.success
    assert os.path.basename(type(d).__module__) == "stabilizer_qec.relay_bp.bp"

"""stabilizer_qec.tesseract_decoder against Tesseract (quantumlib/tesseract-decoder, its 2026-10-13
build): the A* decoder gives Tesseract's errors, low-confidence flags, observables and costs under
every setting (beams, beam climbing, revisits, queue bounds, penalties, merging, sparsified
errors, generated and literal detector orders), and the same visualizer file and verbose log;
the sinter and multi-pass decoders give Tesseract's predictions; the simplex decoder finds
errors of the same (optimal) cost; and the error, model and order helpers agree."""

from __future__ import annotations

import contextlib
import io
import os
import pickle
import random
import sys

import numpy as np
import pytest

from conftest import reference

T = reference("tesseract_decoder")
stim = pytest.importorskip("stim")

import stabilizer_qec as sq  # noqa: E402
import stabilizer_qec.tesseract_decoder as O  # noqa: E402


def circuits():
    yield stim.Circuit.generated("surface_code:rotated_memory_x", distance=3, rounds=3, after_clifford_depolarization=0.01, before_measure_flip_probability=0.01)
    yield stim.Circuit.generated("color_code:memory_xyz", distance=3, rounds=2, after_clifford_depolarization=0.01)
    yield stim.Circuit.generated("repetition_code:memory", distance=5, rounds=3, after_clifford_depolarization=0.03)


def random_dem(rng: random.Random):
    n = rng.randint(4, 12)
    lines = [f"detector({rng.randint(0, 3)}, {rng.randint(0, 3)}, {d // 3}) D{d}" for d in range(n)]
    for _ in range(rng.randint(8, 25)):
        dets = rng.sample(range(n), rng.randint(1, 3))
        obs = " L0" if rng.random() < 0.3 else ""
        lines.append(f"error({rng.choice([0.0, 0.001, 0.01, 0.05, 0.1, 0.2])}) " + " ".join(f"D{d}" for d in dets) + obs)
    return stim.DetectorErrorModel("\n".join(lines))


def models():
    for c in circuits():
        yield c.detector_error_model(), c.compile_detector_sampler(seed=1).sample(20)
    rng = random.Random(5)
    for k in range(3):
        dem = random_dem(rng)
        yield dem, np.random.default_rng(k).random((20, dem.num_detectors)) < 0.2


CONFIGS = [
    {}, dict(det_beam=0), dict(det_beam=65535), dict(det_beam=-1), dict(det_beam=3, beam_climbing=True), dict(no_revisit_dets=False, det_beam=4),
    dict(pqlimit=30), dict(det_penalty=0.3), dict(merge_errors=False), dict(sparsify_errors=True, sparsify_base_degree=2),
    dict(sparsify_errors=True, sparsify_base_degree=2, sparsify_max_degree=3, sparsify_reactivate_limit=1),
    dict(num_det_orders=3, det_order_method=1, seed=7), dict(num_det_orders=2, det_order_method=0, seed=7), dict(num_det_orders=2, det_order_method=2, seed=11),
]


def ours(dem):
    return sq.DetectorErrorModel(str(dem))


def config_pair(dem, kw):
    theirs_kw, ours_kw = dict(kw), dict(kw)
    if "det_order_method" in kw:
        theirs_kw["det_order_method"] = T.utils.DetectorOrderMethod(kw["det_order_method"])
        ours_kw["det_order_method"] = O.utils.DetectorOrderMethod(kw["det_order_method"])
    return T.tesseract.TesseractConfig(dem, **theirs_kw), O.tesseract.TesseractConfig(ours(dem), **ours_kw)


@pytest.mark.parametrize("kw", CONFIGS, ids=[str(k) for k in CONFIGS])
def test_decoder_answers(kw):
    for dem, shots in models():
        tc, oc = config_pair(dem, kw)
        td, od = tc.compile_decoder(), oc.compile_decoder()
        assert str(tc) == str(oc) and tc.det_orders == oc.det_orders
        assert [str(e) for e in td.errors] == [str(e) for e in od.errors]
        assert [e.likelihood_cost for e in td.errors] == [e.likelihood_cost for e in od.errors]
        assert td.config.sparsify_reactivate_limit == od.config.sparsify_reactivate_limit
        for k, row in enumerate(shots):
            a, b = td.decode_to_errors(row), od.decode_to_errors(row)
            assert (list(a), td.low_confidence_flag) == (b, od.low_confidence_flag)
            assert td.cost_from_errors(a) == od.cost_from_errors(b)
            assert list(td.get_observables_from_errors(a)) == od.get_observables_from_errors(b)
            assert np.array_equal(td.decode(row), od.decode(row))
            if k < 3:
                for order in range(min(2, len(tc.det_orders))):
                    for beam in (0, 3):
                        assert list(td.decode_to_errors(row, order, beam)) == od.decode_to_errors(row, det_order=order, det_beam=beam)
                        assert td.low_confidence_flag == od.low_confidence_flag
        assert np.array_equal(td.decode_batch(shots), od.decode_batch(shots))
        hits = [int(d) for d in np.flatnonzero(shots[0])]
        assert np.array_equal(td.decode_from_detection_events(hits), od.decode_from_detection_events(hits))


@pytest.mark.parametrize("kw", [dict(create_visualization=True, num_det_orders=2), dict(verbose=True, num_det_orders=2, det_beam=3), dict(verbose=True, beam_climbing=True, det_beam=2, pqlimit=40)])
def test_visualizer_and_verbose_log(kw, tmp_path):
    c = next(circuits())
    dem = c.detector_error_model()
    shots = c.compile_detector_sampler(seed=4).sample(6)
    texts = []
    for k, (mod, d) in enumerate(((T, dem), (O, ours(dem)))):
        with contextlib.redirect_stdout(io.StringIO()):
            dec = mod.tesseract.TesseractConfig(d, **kw).compile_decoder()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            for row in shots:
                dec.decode_to_errors(row)
            dec.decode_to_errors(shots[0], 1, 2)
        if kw.get("create_visualization"):
            dec.visualizer.write(str(tmp_path / f"{k}.txt"))
            texts.append((tmp_path / f"{k}.txt").read_text())
        else:
            texts.append(buf.getvalue())
    assert texts[0] and texts[0] == texts[1]


def test_sinter_decoders(tmp_path):
    assert sorted(T.make_tesseract_sinter_decoders_dict()) == sorted(O.make_tesseract_sinter_decoders_dict())
    for c in list(circuits())[:2]:
        dem = c.detector_error_model()
        dets = c.compile_detector_sampler(seed=10).sample(40, bit_packed=True)
        for name in ("tesseract-short-beam", "tesseract-short-beam-sparsify-color-code-like"):
            a = T.make_tesseract_sinter_decoders_dict()[name].compile_decoder_for_dem(dem=dem).decode_shots_bit_packed(bit_packed_detection_event_data=dets)
            b = O.make_tesseract_sinter_decoders_dict()[name].compile_decoder_for_dem(dem=ours(dem)).decode_shots_bit_packed(bit_packed_detection_event_data=dets)
            assert a.dtype == b.dtype and np.array_equal(a, b)
        dem.to_file(tmp_path / "m.dem")
        (tmp_path / "d.b8").write_bytes(dets.tobytes())
        out = []
        for k, dec in enumerate((T.TesseractSinterDecoder(num_det_orders=2), O.TesseractSinterDecoder(num_det_orders=2))):
            dec.decode_via_files(
                num_shots=40, num_dets=dem.num_detectors, num_obs=dem.num_observables, dem_path=tmp_path / "m.dem",
                dets_b8_in_path=tmp_path / "d.b8", obs_predictions_b8_out_path=tmp_path / f"o{k}.b8", tmp_dir=tmp_path,
            )
            out.append((tmp_path / f"o{k}.b8").read_bytes())
        assert out[0] == out[1]


def test_sinter_decoder_state():
    a = T.TesseractSinterDecoder(det_beam=7, num_det_orders=3, sparsify_errors=True, sparsify_base_degree=2, seed=9)
    b = O.TesseractSinterDecoder(det_beam=7, num_det_orders=3, sparsify_errors=True, sparsify_base_degree=2, seed=9)

    def plain(state):
        return [v if isinstance(v, float) else int(v) for v in state]

    assert plain(a.__getstate__()) == plain(b.__getstate__())
    assert pickle.loads(pickle.dumps(b)) == b and b != O.TesseractSinterDecoder()
    state = a.__getstate__()
    layouts = [state, tuple(list(state[:8]) + [True, 2, -1, 5, 4, T.utils.DetectorOrderMethod.BFS, 3]), tuple(list(state[:8]) + [0] + list(state[9:]))]
    for st in layouts:
        x = T.TesseractSinterDecoder.__new__(T.TesseractSinterDecoder)
        x.__setstate__(st)
        y = O.TesseractSinterDecoder.__new__(O.TesseractSinterDecoder)
        y.__setstate__(tuple(O.utils.DetectorOrderMethod(int(v)) if isinstance(v, T.utils.DetectorOrderMethod) else v for v in st))
        assert plain(x.__getstate__()) == plain(y.__getstate__())


def test_multi_pass_decoders():
    sys.path.insert(0, os.path.dirname(T.__file__))
    try:
        import multi_pass_sinter_decoders as TM
    except ImportError:
        pytest.skip("Tesseract's multi_pass_sinter_decoders is not installed")
    finally:
        sys.path.pop(0)
    from stabilizer_qec.tesseract_decoder import multi_pass_sinter_decoders as OM

    for c in (next(circuits()), stim.Circuit.generated("surface_code:rotated_memory_z", distance=3, rounds=4, after_clifford_depolarization=0.02)):
        dem = c.detector_error_model()
        dets = c.compile_detector_sampler(seed=20).sample(40, bit_packed=True)
        theirs, mine = TM.get_sinter_decoders(), OM.get_sinter_decoders()
        assert sorted(theirs) == sorted(mine)
        for name in theirs:
            assert outcome(theirs[name], dem, dets) == outcome(mine[name], ours(dem), dets), name
        for passes in (1, 2):
            for strategy in (0, 1):
                a = TM.MultiPassSinterDecoder(num_passes=passes, strategy=T.SchedulingStrategy(strategy), num_det_orders=3, det_beam=4, seed=5)
                b = OM.MultiPassSinterDecoder(num_passes=passes, strategy=O.SchedulingStrategy(strategy), num_det_orders=3, det_beam=4, seed=5)
                assert outcome(a, dem, dets) == outcome(b, ours(dem), dets), (passes, strategy)


def outcome(decoder, dem, dets):
    """The predictions, or the error raised (the second model's detectors carry no basis, so
    neither package can split it)."""
    try:
        return decoder.compile_decoder_for_dem(dem=dem).decode_shots_bit_packed(bit_packed_detection_event_data=dets).tolist()
    except ValueError as e:
        return str(e)


@pytest.mark.parametrize("kw", [{}, dict(merge_errors=False), dict(window_length=2, window_slide_length=1), dict(window_length=3, window_slide_length=2)])
def test_simplex_finds_optimal_errors(kw):
    for c in list(circuits())[:2]:
        dem = c.detector_error_model()
        shots = c.compile_detector_sampler(seed=30).sample(20)
        ts, os_ = T.simplex.SimplexConfig(dem, **kw).compile_decoder(), O.simplex.SimplexConfig(ours(dem), **kw).compile_decoder()
        assert str(ts.config) == str(os_.config)
        assert (ts.start_time_to_errors, ts.end_time_to_errors) == (os_.start_time_to_errors, os_.end_time_to_errors)
        for row in shots:
            a, b = ts.decode_to_errors(row), os_.decode_to_errors(row)
            syndrome = np.zeros(dem.num_detectors, dtype=bool)
            for e in b:
                syndrome[os_.errors[os_._dem_to_error[e]].symptom.detectors] ^= True
            assert np.array_equal(syndrome, row)
            if not kw.get("window_length"):
                assert ts.cost_from_errors(a) == pytest.approx(os_.cost_from_errors(b), abs=1e-9)


def test_helpers():
    for c in circuits():
        dem = c.detector_error_model()
        o = ours(dem)
        assert str(T.common.merge_indistinguishable_errors(dem)) == str(O.common.merge_indistinguishable_errors(o))
        assert str(T.common.remove_zero_probability_errors(dem)) == str(O.common.remove_zero_probability_errors(o))
        counts = list(range(dem.num_errors))
        assert str(T.common.dem_from_counts(dem, counts, 1000)) == str(O.common.dem_from_counts(o, counts, 1000))
        assert T.utils.build_detector_graph(dem) == O.utils.build_detector_graph(o)
        assert T.utils.get_detector_coords(dem) == O.utils.get_detector_coords(o)
        assert [str(e) for e in T.utils.get_errors_from_dem(dem)] == [str(e) for e in O.utils.get_errors_from_dem(o)]
        for m in range(3):
            assert T.utils.build_det_orders(dem, 3, T.utils.DetectorOrderMethod(m), seed=4) == O.utils.build_det_orders(o, 3, O.utils.DetectorOrderMethod(m), seed=4)
    for n, b in ((100, 2), (1000, 3), (5, 0), (0, 4), (10**6, 9)):
        assert T.tesseract.suggest_sparsify_reactivate_limit(n, b) == O.tesseract.suggest_sparsify_reactivate_limit(n, b)
    e1, e2 = T.common.Error(1.5, [1, 2], [0]), O.common.Error(1.5, [1, 2], [0])
    assert (str(e1), str(e1.symptom), e1.get_probability()) == (str(e2), str(e2.symptom), e2.get_probability())
    assert str(T.utils.DetectorOrderMethod.BFS) == str(O.utils.DetectorOrderMethod.BFS)
    assert str(T.SchedulingStrategy.Causal) == str(O.SchedulingStrategy.Causal)

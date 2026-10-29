"""stabilizer_qec.tesseract_decoder against Tesseract (quantumlib/tesseract-decoder, its 2026-10-13
build): the A* decoder gives Tesseract's errors, low-confidence flags, observables and costs under
every setting (beams, beam climbing, revisits, queue bounds, penalties, merging, sparsified
errors, generated and literal detector orders), and the same visualizer file and verbose log;
the sinter and multi-pass decoders give Tesseract's predictions; the simplex decoder finds
errors of the same (optimal) cost; and the error, model and order helpers agree."""

from __future__ import annotations

import contextlib
import ctypes
import io
import os
import pickle
import platform
import random
import subprocess
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


def surface(rounds=3, p=0.01):
    c = stim.Circuit.generated("surface_code:rotated_memory_z", distance=3, rounds=rounds, after_clifford_depolarization=p)
    return c.detector_error_model(), c.compile_detector_sampler(seed=11).sample(40)


def test_two_pass_decoder_on_an_annotated_model():
    # Tesseract's two-pass reweighting addresses errors by their index in the flattened model;
    # those once indexed the merged model here ("DEM error index 96 is out of range").
    sys.path.insert(0, os.path.dirname(T.__file__))
    try:
        import multi_pass_sinter_decoders as TM
    except ImportError:
        pytest.skip("Tesseract's multi_pass_sinter_decoders is not installed")
    finally:
        sys.path.pop(0)
    from stabilizer_qec.tesseract_decoder import multi_pass_sinter_decoders as OM

    c = stim.Circuit.generated("surface_code:rotated_memory_x", distance=3, rounds=3, after_clifford_depolarization=0.006)
    dets = c.compile_detector_sampler(seed=5).sample(200, bit_packed=True)
    dem = c.detector_error_model().flattened()
    theirs = T.demutil.annotate_detector_bases(dem, detector_basis_classifier=T.demutil.stim_surface_code_detector_basis_classifier)
    mine = O.demutil.annotate_detector_bases(ours(dem), detector_basis_classifier=O.demutil.stim_surface_code_detector_basis_classifier)
    for strategy in (0, 1):
        a = TM.MultiPassSinterDecoder(num_passes=2, strategy=T.SchedulingStrategy(strategy)).compile_decoder_for_dem(dem=theirs)
        b = OM.MultiPassSinterDecoder(num_passes=2, strategy=O.SchedulingStrategy(strategy)).compile_decoder_for_dem(dem=mine)
        assert np.array_equal(a.decode_shots_bit_packed(bit_packed_detection_event_data=dets), b.decode_shots_bit_packed(bit_packed_detection_event_data=dets))


def test_repeated_detection_events():
    # Tesseract lights a repeated detector once but counts it twice: no state is then free of
    # lit detectors, and the search gives up (this once panicked). (Tesseract then reads past
    # its fault lists, and may crash: it is run in its own process, and compared when it lives.)
    dem, shots = surface()
    od = O.tesseract.TesseractConfig(ours(dem)).compile_decoder()
    b = od.decode_from_detection_events([4, 4, 9])
    assert (b.tolist(), od.predicted_errors_buffer, od.low_confidence_flag) == ([False], [], True)
    script = (
        "import stim, tesseract_decoder as T\n"
        "c = stim.Circuit.generated('surface_code:rotated_memory_z', distance=3, rounds=3, after_clifford_depolarization=0.01)\n"
        "d = T.tesseract.TesseractConfig(c.detector_error_model()).compile_decoder()\n"
        "r = d.decode_from_detection_events([4, 4, 9])\n"
        "print(repr((r.tolist(), list(d.predicted_errors_buffer), d.low_confidence_flag)))\n"
    )
    run = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    if run.returncode == 0:
        assert run.stdout.strip().splitlines()[-1] == repr(([False], [], True))
    td = T.tesseract.TesseractConfig(dem).compile_decoder()
    assert [list(td.decode_to_errors(s)) for s in shots] == [od.decode_to_errors(s) for s in shots]
    for kw in ({}, dict(beam_climbing=True, det_beam=2), dict(no_revisit_dets=False, pqlimit=50), dict(det_beam=1)):
        od = O.tesseract.TesseractConfig(ours(dem), **kw).compile_decoder()
        for events in ([0, 0], [3, 7, 3, 7, 3], [2, 9, 2]):
            assert (od.decode_from_detection_events(events).tolist(), od.predicted_errors_buffer, od.low_confidence_flag) == ([False], [], True)


def test_merging_errors_of_probability_zero():
    # Two errors of probability 0 (or 1) with one symptom merge to NaN: Tesseract's decoder
    # drops it; the model Tesseract's merge returns cannot be read back.
    for text in ("error(0) D0\nerror(0) D0\nerror(0.1) D1", "error(1) D0\nerror(1) D0\nerror(0.1) D0 D1", "error(0) D0 L0\nerror(1) D0 L0\nerror(0.2) D0 D1"):
        dem = stim.DetectorErrorModel(text)
        for merge in (True, False):
            td = T.tesseract.TesseractConfig(dem, merge_errors=merge).compile_decoder()
            od = O.tesseract.TesseractConfig(ours(dem), merge_errors=merge).compile_decoder()
            assert [str(e) for e in td.errors] == [str(e) for e in od.errors]
            for row in ([1, 0], [0, 1], [1, 1]):
                assert td.decode(np.array(row, bool)).tolist() == od.decode(np.array(row, bool)).tolist()
        errors = []
        for mod, d in ((T, dem), (O, ours(dem))):
            try:
                errors.append(str(mod.common.merge_indistinguishable_errors(d)))
            except ValueError as e:
                errors.append(str(e))
        assert errors[0] == errors[1]


def decodes(d, shots):
    return [(list(d.decode_to_errors(s)), d.low_confidence_flag) for s in shots]


@pytest.mark.parametrize(
    "field,value",
    [("det_beam", 0), ("det_beam", 2), ("pqlimit", 3), ("det_penalty", 5.0), ("beam_climbing", True), ("no_revisit_dets", False), ("verbose", True), ("create_visualization", True)],
)
def test_decoder_config_is_live(field, value, tmp_path):
    # Tesseract reads its decoder's config at every decode.
    dem, shots = surface()
    results = []
    for mod, d in ((T, dem), (O, ours(dem))):
        dec = mod.tesseract.TesseractConfig(d, det_beam=5, pqlimit=10000).compile_decoder()
        setattr(dec.config, field, value)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            got = decodes(dec, shots)
        dec.visualizer.write(str(tmp_path / "v.txt"))
        results.append((got, buf.getvalue(), (tmp_path / "v.txt").read_text()))
    assert results[0] == results[1]
    if field not in ("verbose", "create_visualization", "no_revisit_dets"):
        fresh = decodes(O.tesseract.TesseractConfig(ours(dem), **{"det_beam": 5, "pqlimit": 10000, field: value}).compile_decoder(), shots)
        assert results[1][0] == fresh


def test_decoder_config_assignment_and_orders():
    dem, shots = surface()
    rng = random.Random(3)
    orders = [rng.sample(range(dem.num_detectors), dem.num_detectors) for _ in range(2)]
    rev = [list(range(dem.num_detectors))[::-1]]
    outcomes = []
    for mod, d in ((T, dem), (O, ours(dem))):
        out = []
        dec = mod.tesseract.TesseractConfig(d, det_beam=1).compile_decoder()
        dec.config.det_orders = orders
        out.append(decodes(dec, shots))
        assert dec.config is dec.config
        # (Orders are checked when read or compiled, not when set.)
        config = mod.tesseract.TesseractConfig(d)
        config.det_orders = [[0, 1]]
        with pytest.raises(ValueError, match="has size 2"):
            config.det_orders
        with pytest.raises(ValueError, match="has size 2"):
            config.compile_decoder()
        if mod is O:
            # (Tesseract reads past a short order at the next decode; this refuses it.)
            bad = O.tesseract.TesseractConfig(d).compile_decoder()
            bad.config.det_orders = [[0] * d.num_detectors]
            with pytest.raises(ValueError, match="more than once"):
                bad.decode_to_errors(shots[0])
        # (Without an order Tesseract indexes an empty vector: refused on macOS, undefined
        # elsewhere. This package refuses it everywhere.)
        checked = mod is O or sys.platform == "darwin"
        dec.config.det_orders = []
        if checked:
            with pytest.raises(RuntimeError, match="not been resolved"):
                dec.decode_to_errors(shots[0])
        dec = mod.tesseract.TesseractConfig(d).compile_decoder()
        dec.config = mod.tesseract.TesseractConfig(d, det_beam=0)
        if checked:
            with pytest.raises(RuntimeError, match="not been resolved"):
                dec.decode_to_errors(shots[0])
        dec.config = mod.tesseract.TesseractConfig(d, det_beam=0, det_orders=rev)
        out.append(decodes(dec, shots))
        other = mod.tesseract.TesseractConfig(d, det_beam=2).compile_decoder()
        dec.config = other.config
        other.config.det_beam = 0  # (a copy was assigned)
        out.append((dec.config.det_beam, decodes(dec, shots)))
        outcomes.append(out)
    assert outcomes[0] == outcomes[1]


@pytest.mark.parametrize("beam", [-1, -2, -5])
def test_beam_climbing_with_a_negative_beam(beam):
    # Tesseract climbs in its config's int: a remainder by det_beam + 1 = 0 is the dividend
    # (on arm64), one by a negative divisor keeps the dividend's sign.
    dem, shots = surface()
    for climbing in (True, False):
        # (A negative beam is no beam: the queue bound keeps the search short.)
        od = O.tesseract.TesseractConfig(ours(dem), det_beam=beam, beam_climbing=climbing, pqlimit=2000).compile_decoder()
        if climbing and beam == -1 and platform.machine().lower() not in ("arm64", "aarch64"):
            # A remainder by zero, which traps (SIGFPE) in Tesseract off ARM64: the port climbs
            # as on ARM64, where Tesseract runs.
            decodes(od, shots[:6])
            continue
        td = T.tesseract.TesseractConfig(dem, det_beam=beam, beam_climbing=climbing, pqlimit=2000).compile_decoder()
        assert decodes(td, shots[:6]) == decodes(od, shots[:6])


def raised(fn):
    try:
        fn()
    except Exception as e:  # noqa: BLE001
        # (pybind11's messages for a refused argument list its overloads, and the C++
        # library's for an index out of range differ by platform; only the type is compared
        # for those.)
        return type(e), str(e) if type(e) in (ValueError, IndexError, RuntimeError) and "vector" not in str(e) else ""
    return None


def test_config_constructors():
    # Tesseract's TesseractConfig takes its settings with no model (from det_beam) or after one;
    # anything else in the model's place is read as a model's text.
    dem = surface()[0]
    for args, kwargs in [((), {}), ((7,), {}), ((True,), {}), ((7, True), {}), ((), dict(det_beam=3)), ((dem, 3), {}), ((), dict(dem=dem, det_beam=3)), (("error(0.1) D0",), {})]:
        mine = tuple(ours(a) if isinstance(a, stim.DetectorErrorModel) else a for a in args)
        assert str(T.tesseract.TesseractConfig(*args, **kwargs)) == str(O.tesseract.TesseractConfig(*mine, **{k: ours(v) if k == "dem" else v for k, v in kwargs.items()}))
    for args, kwargs in [((7.0,), {}), ((None,), {}), ((7,), dict(dem=dem)), ((dem,), dict(pqlimit=-1)), ((dem,), dict(det_beam=2**31)), ((dem,), dict(det_penalty="x")),
                         ((dem,), dict(det_order_method=1)), ((dem,), dict(seed=-1)), ((dem,), dict(num_det_orders=0)), ((dem,), dict(det_orders=[[0, 1]]))]:
        assert raised(lambda: T.tesseract.TesseractConfig(*args, **kwargs)) == raised(lambda: O.tesseract.TesseractConfig(*args, **kwargs)), (args, kwargs)


def test_enums_as_pybind11_binds_them():
    for TE, OE in ((T.utils.DetectorOrderMethod, O.utils.DetectorOrderMethod), (T.SchedulingStrategy, O.SchedulingStrategy)):
        assert list(TE.__members__) == list(OE.__members__)
        for name in TE.__members__:
            a, b = getattr(TE, name), getattr(OE, name)
            assert (repr(a), str(a), a.name, a.value, int(a), hash(a), bool(a)) == (repr(b), str(b), b.name, b.value, int(b), hash(b), bool(b))
            assert (a == 0, a == a.value, a != 0, a == TE(int(a))) == (b == 0, b == b.value, b != 0, b == OE(int(b)))
            assert pickle.loads(pickle.dumps(b)) == b
        for v in (5, -1, True):
            assert (repr(TE(v)), TE(v).name) == (repr(OE(v)), OE(v).name)
        for v in (1.0, "BFS", 2**31):
            assert raised(lambda: TE(v)) == raised(lambda: OE(v)) == (TypeError, "")
        assert raised(lambda: TE(0) < TE(1)) == raised(lambda: OE(0) < OE(1)) == (TypeError, "")
    assert T.utils.DetectorOrderMethod.__doc__ == O.utils.DetectorOrderMethod.__doc__
    M, N = T.utils.DetectorOrderMethod, O.utils.DetectorOrderMethod
    assert (T.utils.DetBFS is M.DetBFS, M.DetBFS is M.BFS, M.DetBFS == M.BFS) == (O.utils.DetBFS is N.DetBFS, N.DetBFS is N.BFS, N.DetBFS == N.BFS)


def test_errors_raised_as_tesseracts_binding_raises_them(tmp_path):
    dem = stim.DetectorErrorModel("error(0.1) D0 D1 L0\nerror(0.05) D0 D1 L0\nerror(0) D2\nerror(0.2) D1 D2\ndetector(0, 0, 0) D0\ndetector(1, 0, 0) D1\ndetector(2, 0, 0) D2")
    cases = {
        "observables of a negative index": lambda m, d: m.tesseract.TesseractConfig(d).compile_decoder().get_observables_from_errors([-1]),
        "cost of a negative index": lambda m, d: m.tesseract.TesseractConfig(d).compile_decoder().cost_from_errors([-1]),
        "observables past the end": lambda m, d: m.tesseract.TesseractConfig(d).compile_decoder().get_observables_from_errors([99]),
        "a 2D syndrome": lambda m, d: m.tesseract.TesseractConfig(d).compile_decoder().decode_to_errors(np.array([[1, 1, 0]], bool)),
        "a 1D batch": lambda m, d: m.tesseract.TesseractConfig(d).compile_decoder().decode_batch(np.array([1, 1, 0], bool)),
        "a float detection": lambda m, d: m.tesseract.TesseractConfig(d).compile_decoder().decode_from_detection_events([1.5]),
        "an int method for orders": lambda m, d: m.utils.build_det_orders(d, 1, 1),
        "a negative seed for orders": lambda m, d: m.utils.build_det_orders(d, 1, m.utils.DetectorOrderMethod.Index, -1),
        "an unknown method": lambda m, d: m.utils.build_det_orders(d, 1, m.utils.DetectorOrderMethod(5), 0),
        "an int method for sinter": lambda m, d: m.TesseractSinterDecoder(det_order_method=0),
        "counts over no shots": lambda m, d: m.common.dem_from_counts(d, [1, 2, 3, 4], 0),
        "no counts over no shots": lambda m, d: m.common.dem_from_counts(d, [0, 0, 0, 0], 0),
        # (Tesseract's C++ model holds the first estimate, NaN, and refuses the second, ∞.)
        "a NaN estimate, then an infinite one": lambda m, d: m.common.dem_from_counts(d, [0, 3, 0, 0], 0),
        "counts above the shots": lambda m, d: m.common.dem_from_counts(d, [20, 0, 0, 0], 10),
        "a negative count": lambda m, d: m.common.dem_from_counts(d, [-1, 0, 0, 0], 10),
        "a float detector": lambda m, d: m.common.Symptom([1.0], []),
        "a detector past int": lambda m, d: m.common.Symptom([2**31], []),
        "a visualizer file of None": lambda m, d: m.viz.Visualizer().write(None),
        "a visualizer file that cannot be opened": lambda m, d: m.viz.Visualizer().write(str(tmp_path / "missing" / "v.txt")),
        "a visualizer file of an int": lambda m, d: m.viz.Visualizer().write(123),
        "a config of an int": lambda m, d: setattr(m.tesseract.TesseractConfig(d).compile_decoder(), "config", 5),
        "an attribute a config has not": lambda m, d: setattr(m.tesseract.TesseractConfig(d), "foo", 1),
        "a beam that is a float": lambda m, d: setattr(m.tesseract.TesseractConfig(d), "det_beam", 1.5),
        "a queue bound past size_t": lambda m, d: setattr(m.tesseract.TesseractConfig(d), "pqlimit", 2**64),
        "orders of None": lambda m, d: setattr(m.tesseract.TesseractConfig(d), "det_orders", None),
        "a model of None": lambda m, d: setattr(m.tesseract.TesseractConfig(d), "dem", None),
    }
    # (Tesseract hands None to fopen as a null path, and writes to the null file fopen returns
    # for a path it cannot open: macOS's C library fails quietly, as the port does; glibc's
    # crashes, so Tesseract is called for those only on macOS.)
    crashes = set() if sys.platform == "darwin" else {"a visualizer file of None", "a visualizer file that cannot be opened"}
    for label, case in cases.items():
        if label in crashes:
            assert raised(lambda: case(O, ours(dem))) is None, label
            continue
        assert raised(lambda: case(T, dem)) == raised(lambda: case(O, ours(dem))), label
    for mod in (T, O):
        s = mod.common.Symptom([1], [2])
        s.detectors.append(5)
        assert s.detectors == [1]


def test_output_goes_where_tesseracts_goes(capfd):
    # Tesseract's C++ output reaches Python's sys.stdout only from the calls that redirect it
    # themselves (decode, decode_to_errors, decode_from_detection_events) or inside
    # ostream_redirect; the rest goes to the process's standard output.
    c = stim.Circuit.generated("repetition_code:memory", distance=3, rounds=2, after_clifford_depolarization=0.03)
    dem = c.detector_error_model()
    shot = next(s for s in c.compile_detector_sampler(seed=2).sample(20) if s.sum() >= 2)
    zero = stim.DetectorErrorModel("error(0.1) D0\nerror(0.1)\nerror(0.2) D0 D1")
    libc = ctypes.CDLL(None)
    cases = {
        "merge warning": lambda m, d: m.common.merge_indistinguishable_errors(d),
        "construction": lambda m, d: m.tesseract.TesseractConfig(d, verbose=True).compile_decoder(),
        "decode": lambda m, d: m.tesseract.TesseractConfig(d, verbose=True).compile_decoder().decode(shot),
        "decode_from_detection_events": lambda m, d: m.tesseract.TesseractConfig(d, verbose=True).compile_decoder().decode_from_detection_events(np.flatnonzero(shot).tolist()),
        "decode_to_errors with an order": lambda m, d: m.tesseract.TesseractConfig(d, verbose=True).compile_decoder().decode_to_errors(shot, 0, 2),
        "decode_batch": lambda m, d: m.tesseract.TesseractConfig(d, verbose=True).compile_decoder().decode_batch(np.array([shot, shot])),
    }
    for label, case in cases.items():
        model = zero if label == "merge warning" else dem
        for redirect in (None, {}, dict(stdout=False)):
            seen = []
            for mod, d in ((T, model), (O, ours(model))):
                buf = io.StringIO()
                capfd.readouterr()
                with contextlib.redirect_stdout(buf):
                    with contextlib.nullcontext() if redirect is None else mod.ostream_redirect(**redirect):
                        case(mod, d)
                libc.fflush(None)  # (Tesseract's C++ output may sit in the C library's buffer.)
                seen.append((buf.getvalue(), capfd.readouterr().out))
            assert seen[0] == seen[1], (label, redirect)
            assert seen[1] != ("", "")

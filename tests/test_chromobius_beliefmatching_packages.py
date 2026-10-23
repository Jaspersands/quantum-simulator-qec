"""stabilizer_qec.chromobius against Chromobius 1.1.1 (predictions in the finite recorded corpus, exact on macOS; weights,
PyMatching's discretized Möbius matching weight as float32, bit for bit; the sinter decoder) and
stabilizer_qec.beliefmatching against beliefmatching 0.2.0 (predictions, check matrices and the
sinter decoder), with BeliefMatching's constructors.

Chromobius's native hash-table traversal can choose different equal-weight liftings from this
port on any platform, including macOS for other graphs (see test_chromobius_edge_cases.py).
This finite corpus retains exact macOS checks; other platforms use per-test agreement limits."""

from __future__ import annotations

import os
import pickle
import sys

import numpy as np
import pytest

from conftest import reference

stim = pytest.importorskip("stim")

import stabilizer_qec as sq  # noqa: E402


def same_predictions(a, b, off_macos):
    """This corpus is exact on macOS; elsewhere require the per-test `off_macos` fraction."""
    agree = (a == b).all(axis=1)
    return bool(agree.all()) if sys.platform == "darwin" else agree.mean() >= off_macos


def color_circuits():
    for d, rounds, p in ((3, 3, 0.01), (5, 4, 0.004)):
        yield stim.Circuit(str(sq.CssCode.color_code(d).memory_circuit(rounds=rounds, p=p, annotate_colors=True)))


def test_chromobius_predictions_and_weights():
    chromobius = reference("chromobius")
    import stabilizer_qec.chromobius as ours

    for c in color_circuits():
        dem = c.detector_error_model()
        dets = c.compile_detector_sampler(seed=1).sample(400, bit_packed=True)
        a = chromobius.compile_decoder_for_dem(dem)
        b = ours.compile_decoder_for_dem(sq.DetectorErrorModel(str(dem)))
        oa, wa = a.predict_weighted_obs_flips_from_dets_bit_packed(dets)
        ob, wb = b.predict_weighted_obs_flips_from_dets_bit_packed(dets)
        assert (oa.shape, oa.dtype, wa.shape, wa.dtype) == (ob.shape, ob.dtype, wb.shape, wb.dtype)
        assert np.array_equal(wa, wb)
        assert same_predictions(oa, ob, 0.95)
        assert np.array_equal(b.predict_obs_flips_from_dets_bit_packed(dets), ob)
        one_a, w1a = a.predict_weighted_obs_flips_from_dets_bit_packed(dets[0])
        one_b, w1b = ours.CompiledDecoder.from_dem(dem).predict_weighted_obs_flips_from_dets_bit_packed(dets[0])
        assert (one_a.shape, w1a.shape, w1a.dtype, float(w1a)) == (one_b.shape, w1b.shape, w1b.dtype, float(w1b))
    with pytest.raises(ValueError, match="np.uint8"):
        b.predict_obs_flips_from_dets_bit_packed(dets.astype(bool))
    with pytest.raises(TypeError):
        ours.CompiledDecoder()


# Colour-code memories (distance, rounds, p, basis, sampling seed), 5,000 shots sampled from each:
# 65,000 shots under STABILIZER_QEC_SLOW, the first shots of three of them (5,000) by default.
# d7 r4 p0.01 X seed 9 holds shot 855, the one shot whose matching PyMatching 2.4's neighbour
# order lists differently from Chromobius's PyMatching 2.2.2.
SWEEP = [
    (3, 3, 0.01, "z", 12348), (3, 5, 0.01, "x", 6), (3, 7, 0.005, "z", 3), (3, 4, 0.002, "x", 4),
    (5, 5, 0.005, "z", 1), (5, 5, 0.01, "z", 2), (5, 3, 0.01, "x", 11), (5, 6, 0.002, "x", 12), (5, 7, 0.001, "z", 13),
    (7, 4, 0.01, "x", 9), (7, 7, 0.005, "z", 8), (7, 3, 0.002, "z", 14), (7, 5, 0.001, "x", 15),
]
SWEEP_DEFAULT = {(3, 3, 0.01, "z", 12348): 2000, (5, 5, 0.005, "z", 1): 2000, (7, 4, 0.01, "x", 9): 1000}


def test_predictions_equal_chromobius_sweep():
    chromobius = reference("chromobius")
    import stabilizer_qec.chromobius as ours

    slow = bool(os.environ.get("STABILIZER_QEC_SLOW"))
    total = 0
    for case in SWEEP:
        n = 5000 if slow else SWEEP_DEFAULT.get(case, 0)
        if not n:
            continue
        d, rounds, p, basis, seed = case
        c = stim.Circuit(str(sq.CssCode.color_code(d).memory_circuit(rounds=rounds, p=p, basis=basis, annotate_colors=True)))
        dem = c.detector_error_model()
        dets = c.compile_detector_sampler(seed=seed).sample(5000, bit_packed=True)[:n]
        want, want_w = chromobius.compile_decoder_for_dem(dem).predict_weighted_obs_flips_from_dets_bit_packed(dets)
        got, got_w = ours.compile_decoder_for_dem(dem).predict_weighted_obs_flips_from_dets_bit_packed(dets)
        assert np.array_equal(got_w, want_w), case
        differ = np.flatnonzero((got != want).any(axis=1)).tolist()
        if sys.platform == "darwin":
            assert differ == [], (case, differ[:10], len(differ))
        else:
            # (Off macOS, Chromobius may drag charge another way; see above.)
            assert len(differ) <= n // 100, (case, len(differ))
        total += n
    assert total == (65000 if slow else 5000)


def test_chromobius_sinter_decoder(tmp_path):
    chromobius = reference("chromobius")
    import stabilizer_qec.chromobius as ours

    c = next(color_circuits())
    dem = c.detector_error_model()
    dets = c.compile_detector_sampler(seed=3).sample(200, bit_packed=True)
    theirs, mine = chromobius.sinter_decoders()["chromobius"], ours.sinter_decoders()["chromobius"]
    assert pickle.loads(pickle.dumps(mine)) == mine
    a = theirs.compile_decoder_for_dem(dem=dem).decode_shots_bit_packed(bit_packed_detection_event_data=dets)
    b = mine.compile_decoder_for_dem(dem=dem).decode_shots_bit_packed(bit_packed_detection_event_data=dets)
    assert a.shape == b.shape and same_predictions(a, b, 0.95)
    dem.to_file(tmp_path / "m.dem")
    (tmp_path / "d.b8").write_bytes(dets.tobytes())
    mine.decode_via_files(
        num_shots=200, num_dets=dem.num_detectors, num_obs=dem.num_observables, dem_path=tmp_path / "m.dem",
        dets_b8_in_path=tmp_path / "d.b8", obs_predictions_b8_out_path=tmp_path / "o.b8", tmp_dir=tmp_path,
    )
    assert (tmp_path / "o.b8").read_bytes() == b.tobytes()



def raised(fn):
    try:
        return fn()
    except Exception as e:  # noqa: BLE001
        return type(e).__name__, str(e)


def outcome(fn):
    """``raised``, with an array returned as a list."""
    r = raised(fn)
    return r.tolist() if isinstance(r, np.ndarray) else r


def test_chromobius_probabilities_above_one_half():
    # A Möbius edge of probability above 1/2 has a negative weight: PyMatching, Chromobius's
    # matcher, matches it at its absolute weight with the edge pre-applied. (These were refused.)
    chromobius = reference("chromobius")
    import stabilizer_qec.chromobius as ours

    head = "detector(0,0,0,0) D0\ndetector(0,0,0,1) D1\n"
    shots = np.array([[3], [0], [1], [2]], np.uint8)
    for body in ("error(0.7) D0 D1 L0\nerror(0.1) D0\nerror(0.1) D1", "error(0.7) D0 D1 L0\nerror(0.7) D0 D1\nerror(0.1) D0\nerror(0.1) D1",
                 "error(0.55) D0 L0\nerror(0.2) D0 D1\nerror(0.3) D1"):
        dem = stim.DetectorErrorModel(head + body)
        a = chromobius.compile_decoder_for_dem(dem).predict_weighted_obs_flips_from_dets_bit_packed(shots)
        b = ours.compile_decoder_for_dem(dem).predict_weighted_obs_flips_from_dets_bit_packed(shots)
        assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    # At probability 1 PyMatching's weights, and Chromobius's, are NaN: refused here.
    with pytest.raises(ValueError, match="probability 1"):
        ours.compile_decoder_for_dem(stim.DetectorErrorModel(head + "error(1) D0 D1 L0\nerror(0.1) D0\nerror(0.1) D1"))


def test_chromobius_refuses_as_chromobius_refuses():
    chromobius = reference("chromobius")
    import stabilizer_qec.chromobius as ours

    dem = stim.DetectorErrorModel("detector(0,0,0,0) D0\ndetector(0,0,0,1) D1\ndetector(0,0,0,2) D2\nerror(0.1) D0 D1 D2 L0\nerror(0.1) D0\nerror(0.1) D1\nerror(0.1) D2")
    a, b = chromobius.compile_decoder_for_dem(dem), ours.compile_decoder_for_dem(dem)
    for dets in (np.array([[8]], np.uint8), np.array([[0], [0b10000]], np.uint8), np.zeros((0, 1), np.uint8)):
        ra, rb = (outcome(lambda d=d: d.predict_obs_flips_from_dets_bit_packed(dets)) for d in (a, b))
        if sys.platform != "darwin" and ra != rb and dets.any():
            # (Off macOS, Chromobius can decode a shot with a padding bit set, which it refuses
            # on macOS; this package refuses it everywhere.)
            assert rb[0] == "ValueError" and not isinstance(ra, tuple)
        else:
            assert ra == rb
    sa = chromobius.sinter_decoders()["chromobius"].compile_decoder_for_dem(dem=dem)
    sb = ours.sinter_decoders()["chromobius"].compile_decoder_for_dem(dem=dem)
    for dets in (np.array([[1], [6], [7]], np.int64), np.array([[3], [5]], np.uint16), np.zeros((0, 1), np.uint8)):
        ra, rb = raised(lambda: sa.decode_shots_bit_packed(bit_packed_detection_event_data=dets)), raised(lambda: sb.decode_shots_bit_packed(bit_packed_detection_event_data=dets))
        assert (ra[0], ra[1]) == (rb[0], rb[1]) if isinstance(ra, tuple) else np.array_equal(ra, rb)


def test_chromobius_refuses_models_as_chromobius_refuses():
    """Models Chromobius will not compile, refused with its messages: a detector without its
    colour coordinate (declared without coordinates, with three, shifted to a 4th that is not a
    whole -1..5), and a fault it cannot split into basic faults (seven detectors in one basis;
    with its observable and an ignored detector; ten in Z once an X piece is split off). Each
    error's targets are checked in the model's order, as Chromobius checks them: a detector no
    declaration colours (named twice, so cancelling, included) and an observable past L63."""
    chromobius = reference("chromobius")
    import stabilizer_qec.chromobius as ours

    seven = "".join(f"detector(0, {k}, 0, {k % 3}) D{k}\n" for k in range(7))
    ten_z = "".join(f"detector({k}, 0, 0, {k % 3 + 3}) D{k}\n" for k in range(7, 17))
    models = [
        "error(0.1) D0\ndetector D0\n",
        "error(0.1) D0 D1\ndetector(0, 0, 0) D0\ndetector(1, 0, 0) D1\n",
        "shift_detectors(1, 2, 3, 0.5) 4\ndetector(0.25, 1.123456789, 0, 0) D1\n",
        "detector(0, 0, 0, 6) D0\n",
        "detector(0, 0, 0, -2) D0\n",
        "repeat 2 {\n    detector(0, 0, 0, 1) D0\n    shift_detectors(0, 0, 0, 2) 1\n}\ndetector(0, 0, 0, 7) D0\n",
        "error(0.1) D0 D1 D2 D3 D4 D5 D6\n" + seven,
        "shift_detectors(0.5, 0, 0, 0) 0\nerror(0.125) D0 D1 D2 D3 D4 D5 D6 D7 L1\n" + seven + "detector(0, 7, 0, -1) D7\n",
        "error(0.1) D0 D1 L2\nerror(0.1) D0 D1 D7 D8 D9 D10 D11 D12 D13 D14 D15 D16\n" + seven + ten_z,
        "error(0.1) D0\nerror(0.1) D1 L0\ndetector(0, 0, 0, 0) D0\n",
        "error(0.1) D1 D1\nerror(0.1) D0\ndetector(0, 0, 0, 0) D0\n",
        "error(0.1) D0 L64\ndetector(0, 0, 0, 0) D0\n",
        "error[foo](0.125) L70 D1\ndetector(0, 0, 0, 0) D0\n",
        "error(0.1) D1 L64\ndetector(0, 0, 0, 0) D0\n",
        "repeat 3 {\n    error(0.1) D0 L1\n    detector(0, 0, 0, 0) D0\n    shift_detectors(0, 1) 1\n}\nerror(0.1) D0 L65\n",
        "error(0.1) D0 L64\ndetector(0, 0, 0) D0\n",
    ]
    for text in models:
        dem = stim.DetectorErrorModel(text)
        assert raised(lambda: ours.compile_decoder_for_dem(dem)) == raised(lambda: chromobius.compile_decoder_for_dem(dem)), text


def test_chromobius_decode_via_files(tmp_path):
    chromobius = reference("chromobius")
    import stabilizer_qec.chromobius as ours

    c = stim.Circuit(str(sq.CssCode.color_code(3).memory_circuit(rounds=3, p=0.005, annotate_colors=True)))
    dem = c.detector_error_model()
    dets = c.compile_detector_sampler(seed=0).sample(64, bit_packed=True).tobytes()
    dem.to_file(tmp_path / "m.dem")

    def run(module, num_shots, data, num_obs=dem.num_observables, num_dets=dem.num_detectors):
        (tmp_path / "d.b8").write_bytes(data)
        (tmp_path / "o.b8").unlink(missing_ok=True)
        r = raised(lambda: module.sinter_decoders()["chromobius"].decode_via_files(
            num_shots=num_shots, num_dets=num_dets, num_obs=num_obs, dem_path=tmp_path / "m.dem",
            dets_b8_in_path=tmp_path / "d.b8", obs_predictions_b8_out_path=tmp_path / "o.b8", tmp_dir=tmp_path))
        return r if r is not None else (tmp_path / "o.b8").read_bytes()

    # Shots past the end of the file repeat its last record; a partial record is refused;
    # predictions are num_obs bits wide; records narrower than the model are padded.
    cases = [(70, dets), (64, dets[:-3]), (64, dets, 9), (64, dets, 17), (64, dets, 0), (3, dets, 1, dem.num_detectors - 8),
             (8, dets, 1, dem.num_detectors - 8), (0, b""), (2, b"")]
    for case in cases:
        a, b = run(chromobius, *case), run(ours, *case)
        if isinstance(a, bytes) and isinstance(b, bytes) and len(a) == len(b) and sys.platform != "darwin":
            # (Off macOS, Chromobius may drag charge another way; see above.)
            assert sum(x != y for x, y in zip(a, b)) <= 2, case
        else:
            assert a == b, case
    # Records wider than the model are handed over whole, and a set bit past its detectors is
    # refused. Chromobius reads such records nondeterministically: the detector it names
    # differs from run to run, and some runs decode them.
    for num_shots in (3, 8):
        a, b = run(chromobius, num_shots, dets, 1, dem.num_detectors + 8), run(ours, num_shots, dets, 1, dem.num_detectors + 8)
        assert b[0] == "ValueError" and "does not correspond to a node" in b[1]
        assert (a[0] == "ValueError" and "does not correspond to a node" in a[1]) or (isinstance(a, bytes) and len(a) == num_shots)
    (tmp_path / "d.b8").unlink()
    missing = dict(num_shots=1, num_dets=dem.num_detectors, num_obs=1, dem_path=tmp_path / "m.dem", dets_b8_in_path=tmp_path / "d.b8",
                   obs_predictions_b8_out_path=tmp_path / "o.b8", tmp_dir=tmp_path)
    assert raised(lambda: chromobius.sinter_decoders()["chromobius"].decode_via_files(**missing)) == raised(lambda: ours.sinter_decoders()["chromobius"].decode_via_files(**missing))


def test_chromobius_lifting_of_pymatchings_matching():
    # The lifting alone (`lift_edges`), handed PyMatching's edges in PyMatching's order, is
    # Chromobius's. (PyMatching 2.4 lists these shots' edges as Chromobius's 2.2.2 does.)
    chromobius = reference("chromobius")
    pymatching = pytest.importorskip("pymatching")
    import stabilizer_qec.chromobius as ours

    c = stim.Circuit(str(sq.CssCode.color_code(3).memory_circuit(rounds=3, p=0.01, basis="z", annotate_colors=True)))
    dem = c.detector_error_model()
    dets = c.compile_detector_sampler(seed=12348).sample(2000, bit_packed=False)
    theirs = chromobius.compile_decoder_for_dem(dem).predict_obs_flips_from_dets_bit_packed(np.packbits(dets, axis=1, bitorder="little"))
    cm = ours.compile_decoder_for_dem(dem)._decoder
    matching = pymatching.Matching.from_detector_error_model(stim.DetectorErrorModel(str(cm.mobius_model)))
    lifted = []
    for shot in dets:
        fired = np.flatnonzero(shot)
        events = np.zeros(matching.num_detectors, np.uint8)
        events[2 * fired] = events[2 * fired + 1] = 1
        edges = [tuple(int(x) for x in e) for e in matching.decode_to_edges_array(events).tolist()] if fired.size else []
        lifted.append(cm._x.lift_edges(fired.tolist(), edges))
    want = [int.from_bytes(row.tobytes(), "little") for row in theirs]
    if sys.platform == "darwin":
        assert lifted == want
    else:
        # (Off macOS, Chromobius may drag charge another way; see above.)
        assert sum(x != y for x, y in zip(lifted, want)) <= len(want) // 100

def test_beliefmatching_package(tmp_path):
    beliefmatching = pytest.importorskip("beliefmatching")
    pytest.importorskip("scipy")
    import stabilizer_qec.beliefmatching as ours

    c = stim.Circuit.generated("surface_code:rotated_memory_x", distance=5, rounds=4, after_clifford_depolarization=0.008)
    oc = sq.Circuit(str(c))
    dets, _ = c.compile_detector_sampler(seed=2).sample(200, separate_observables=True)
    expected = beliefmatching.BeliefMatching.from_stim_circuit(c).decode_batch(dets)
    assert np.array_equal(expected, ours.BeliefMatching.from_stim_circuit(oc).decode_batch(dets))
    assert np.array_equal(expected, ours.BeliefMatching(oc).decode_batch(dets))
    dem = c.detector_error_model(decompose_errors=True)
    odem = sq.DetectorErrorModel(str(dem))
    # Min-sum BP leaves many edges of equal (often zero) weight, so the matchings tie: BP agrees
    # bit for bit, and the matcher (PyMatching's, transliterated) breaks the ties as PyMatching.
    x = beliefmatching.BeliefMatching.from_detector_error_model(dem, max_bp_iters=5, bp_method="minimum_sum").decode_batch(dets)
    y = ours.BeliefMatching.from_detector_error_model(odem, max_bp_iters=5, bp_method="minimum_sum").decode_batch(dets)
    assert np.array_equal(x, y)
    m1, m2 = beliefmatching.detector_error_model_to_check_matrices(dem), ours.detector_error_model_to_check_matrices(odem)
    for name in ("check_matrix", "observables_matrix", "edge_check_matrix", "edge_observables_matrix", "hyperedge_to_edge_matrix"):
        assert (getattr(m1, name) != getattr(m2, name)).nnz == 0, name
    assert np.allclose(m1.priors, m2.priors, rtol=1e-15, atol=0)
    dem.to_file(tmp_path / "m.dem")
    sampled = c.compile_detector_sampler(seed=5).sample(50, bit_packed=True)
    (tmp_path / "d.b8").write_bytes(sampled.tobytes())
    out = []
    for k, dec in enumerate((beliefmatching.BeliefMatchingSinterDecoder(), ours.BeliefMatchingSinterDecoder())):
        dec.decode_via_files(
            num_shots=50, num_dets=dem.num_detectors, num_obs=dem.num_observables, dem_path=tmp_path / "m.dem",
            dets_b8_in_path=tmp_path / "d.b8", obs_predictions_b8_out_path=tmp_path / f"o{k}.b8", tmp_dir=tmp_path,
        )
        out.append((tmp_path / f"o{k}.b8").read_bytes())
    assert out[0] == out[1]


def test_belief_matching_constructors():
    c = sq.Circuit.generated("surface_code:rotated_memory_z", distance=3, rounds=3, after_clifford_depolarization=0.01)
    dets, _ = c.compile_detector_sampler(seed=4).sample(100, separate_observables=True)
    expected = sq.BeliefMatching(c.detector_error_model(decompose_errors=True)).decode_batch(dets)
    for dec in (sq.BeliefMatching.from_stim_circuit(c), sq.BeliefMatching(c), sq.BeliefMatching.from_detector_error_model(c.detector_error_model(decompose_errors=True))):
        assert np.array_equal(dec.decode_batch(dets), expected)


def test_belief_matching_takes_a_stim_circuit():
    pytest.importorskip("scipy")
    import stabilizer_qec.beliefmatching as ours

    c = stim.Circuit.generated("surface_code:rotated_memory_x", distance=3, rounds=3, after_clifford_depolarization=0.005)
    dets = c.compile_detector_sampler(seed=1).sample(20)
    want = ours.BeliefMatching(c.detector_error_model(decompose_errors=True)).decode_batch(dets)
    assert np.array_equal(ours.BeliefMatching(c).decode_batch(dets), want)

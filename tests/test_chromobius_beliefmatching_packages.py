"""stabilizer_qec.chromobius against Chromobius 1.1.1 (predictions but for ties between equally
light matchings; weights, PyMatching's discretized Möbius matching weight as float32, bit for
bit; the sinter decoder) and stabilizer_qec.beliefmatching against beliefmatching 0.2.0
(predictions, check matrices and the sinter decoder), with BeliefMatching's constructors."""

from __future__ import annotations

import pickle

import numpy as np
import pytest

from conftest import reference

stim = pytest.importorskip("stim")

import stabilizer_qec as sq  # noqa: E402


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
        assert (oa == ob).all(axis=1).mean() >= 0.95  # the rest are ties (their weights agree; which tie each picks varies by platform)
        assert np.array_equal(b.predict_obs_flips_from_dets_bit_packed(dets), ob)
        one_a, w1a = a.predict_weighted_obs_flips_from_dets_bit_packed(dets[0])
        one_b, w1b = ours.CompiledDecoder.from_dem(dem).predict_weighted_obs_flips_from_dets_bit_packed(dets[0])
        assert (one_a.shape, w1a.shape, w1a.dtype, float(w1a)) == (one_b.shape, w1b.shape, w1b.dtype, float(w1b))
    with pytest.raises(ValueError, match="np.uint8"):
        b.predict_obs_flips_from_dets_bit_packed(dets.astype(bool))
    with pytest.raises(TypeError):
        ours.CompiledDecoder()


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
    assert a.shape == b.shape and (a == b).all(axis=1).mean() >= 0.95
    dem.to_file(tmp_path / "m.dem")
    (tmp_path / "d.b8").write_bytes(dets.tobytes())
    mine.decode_via_files(
        num_shots=200, num_dets=dem.num_detectors, num_obs=dem.num_observables, dem_path=tmp_path / "m.dem",
        dets_b8_in_path=tmp_path / "d.b8", obs_predictions_b8_out_path=tmp_path / "o.b8", tmp_dir=tmp_path,
    )
    assert (tmp_path / "o.b8").read_bytes() == b.tobytes()


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
    # Min-sum BP leaves many edges of equal (often zero) weight, so the matchings tie on a few
    # shots (2 of 200 here, each an equally light matching); BP itself agrees bit for bit.
    x = beliefmatching.BeliefMatching.from_detector_error_model(dem, max_bp_iters=5, bp_method="minimum_sum").decode_batch(dets)
    y = ours.BeliefMatching.from_detector_error_model(odem, max_bp_iters=5, bp_method="minimum_sum").decode_batch(dets)
    assert (x != y).any(axis=1).sum() <= 4
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

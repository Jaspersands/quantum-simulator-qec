# Tests of stabilizer_qec.cudaq_qec's "nv-fusion-decoder" (CUDA-Q QEC's NV Fusion decoder, as
# documented in docs/sphinx/api/qec/nv_fusion_decoder_api.rst of NVIDIA/cudaqx 0.8.1; Apache License
# 2.0): the documented example, its parameters, and its matchings against PyMatching's.
import numpy as np
import pytest

from stabilizer_qec import cudaq_qec as qec


def test_documented_example():
    # Two boundary edges carrying one observable each, plus a timelike edge between the detectors.
    H = np.array([[1, 0, 1], [0, 1, 1]], dtype=np.uint8)
    O = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.uint8)
    decoder = qec.get_decoder("nv-fusion-decoder", H, O=O, detector_round=np.array([0, 1], dtype=np.int32))
    r = decoder.decode([1.0, 0.0])
    assert r.result.tolist() == [1.0, 0.0]
    assert r.converged and r.opt_results == {"heralded": False}
    assert decoder.decode([1.0, 1.0]).result.tolist() == [0.0, 0.0]
    assert decoder.decode([0.0, 1.0]).result.tolist() == [0.0, 1.0]


def test_edge_mode_without_O():
    H = np.array([[1, 0, 1], [0, 1, 1]], dtype=np.uint8)
    decoder = qec.get_decoder("nv-fusion-decoder", H)
    assert decoder.get_block_size() == 3 and decoder.get_syndrome_size() == 2
    assert decoder.decode([1.0, 1.0]).result.tolist() == [0.0, 0.0, 1.0]
    assert decoder.decode([0.0, 0.0]).result.tolist() == [0.0, 0.0, 0.0]
    batch = decoder.decode_batch([[1, 0], [1, 1]])
    np.testing.assert_array_equal(batch.result, [[1, 0, 0], [0, 0, 1]])


def test_weights_from_error_rate_vec():
    # 0 -- 1 with a boundary at each end: the lighter boundary wins.
    H = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
    light_left = qec.get_decoder("nv-fusion-decoder", H, error_rate_vec=np.array([0.3, 0.01, 0.01]))
    assert light_left.decode([1, 0]).result.tolist() == [1.0, 0.0, 0.0]
    heavy_left = qec.get_decoder("nv-fusion-decoder", H, error_rate_vec=np.array([0.001, 0.2, 0.3]))
    assert heavy_left.decode([1, 0]).result.tolist() == [0.0, 1.0, 1.0]


def test_from_dem_text():
    dem = """error(0.1) D0 L0
error(0.1) D0 D1 L1
error(0.1) D1 L2
detector(0, 0, 0) D0
detector(0, 0, 1) D1
"""
    decoder = qec.get_decoder("nv-fusion-decoder", dem)
    assert decoder.get_block_size() == 3 and decoder.get_syndrome_size() == 2
    assert decoder.decode([1, 1]).result.tolist() == [0.0, 1.0, 0.0]
    assert decoder.decode([0, 1]).result.tolist() == [0.0, 0.0, 1.0]
    # dem_string beside an H of the right shape: the model is the DEM's.
    H = np.zeros((2, 3), dtype=np.uint8)
    d2 = qec.get_decoder("nv-fusion-decoder", H, dem_string=dem)
    assert d2.decode([1, 1]).result.tolist() == [0.0, 1.0, 0.0]
    with pytest.raises(RuntimeError, match="the detector error model has 2 detectors and 3 errors"):
        qec.get_decoder("nv-fusion-decoder", np.zeros((2, 2), dtype=np.uint8), dem_string=dem)
    # A wider O only widens the result.
    O = np.zeros((4, 3), dtype=np.uint8)
    assert qec.get_decoder("nv-fusion-decoder", dem, O=O).decode([1, 0]).result.tolist() == [1.0, 0.0, 0.0, 0.0]
    with pytest.raises(RuntimeError, match="fewer rows"):
        qec.get_decoder("nv-fusion-decoder", dem, O=np.zeros((2, 3), dtype=np.uint8))


def test_dem_needs_coordinates_or_detector_round():
    dem = "error(0.1) D0 D1\nerror(0.1) D0\nerror(0.1) D1 L0\ndetector(0) D0\n"
    with pytest.raises(RuntimeError, match="D1 has no coordinate.*detector_round"):
        qec.get_decoder("nv-fusion-decoder", dem)
    decoder = qec.get_decoder("nv-fusion-decoder", dem, detector_round=np.array([0, 1], dtype=np.int32))
    assert decoder.decode([0, 1]).result.tolist() == [1.0]


def test_refusals():
    H = np.array([[1, 0, 1], [0, 1, 1]], dtype=np.uint8)
    with pytest.raises(RuntimeError, match="graphlike"):
        qec.get_decoder("nv-fusion-decoder", np.array([[1], [1], [1]], dtype=np.uint8))
    with pytest.raises(RuntimeError, match="not graphlike"):
        qec.get_decoder("nv-fusion-decoder", "error(0.1) D0 D1 D2\ndetector(0) D0\ndetector(1) D1\ndetector(2) D2\n")
    with pytest.raises(RuntimeError, match="only 'brickwall'"):
        qec.get_decoder("nv-fusion-decoder", H, fusion_strategy="ladder")
    with pytest.raises(RuntimeError, match=r"out of range \(0, 0.5\]"):
        qec.get_decoder("nv-fusion-decoder", H, error_rate_vec=np.array([0.1, 0.7, 0.1]))
    with pytest.raises(RuntimeError, match="detector_round has length 1"):
        qec.get_decoder("nv-fusion-decoder", H, detector_round=np.array([0], dtype=np.int32))
    with pytest.raises(RuntimeError, match="block_leaf_size must be positive"):
        qec.get_decoder("nv-fusion-decoder", H, block_leaf_size=0)
    with pytest.raises(RuntimeError, match="out of range: 0 CUDA device"):
        qec.get_decoder("nv-fusion-decoder", H, cuda_device_id=0)
    decoder = qec.get_decoder("nv-fusion-decoder", H, block_leaf_size=2, num_threads=0, fusion_strategy="brickwall")
    with pytest.raises(RuntimeError, match="syndrome length"):
        decoder.decode([1.0])
    # A detection event no matching can explain.
    lonely = qec.get_decoder("nv-fusion-decoder", np.array([[1, 1], [1, 1], [0, 0]], dtype=np.uint8))
    with pytest.raises(RuntimeError, match="No perfect matching"):
        lonely.decode([0, 0, 1])
    with pytest.raises(RuntimeError, match="No perfect matching"):
        lonely.decode([1, 0, 0])
    assert lonely.decode([1, 1, 0]).result.tolist() == [1.0, 0.0]


def _surface_graph(seed):
    stim = pytest.importorskip("stim")
    circuit = stim.Circuit.generated("surface_code:rotated_memory_z", distance=3, rounds=3, after_clifford_depolarization=0.01, before_measure_flip_probability=0.01)
    dem = circuit.detector_error_model(decompose_errors=True)
    # Unique edges of the decomposed model, each with a random rate (no ties between matchings).
    edges = {}
    for inst in dem.flattened():
        if inst.type != "error":
            continue
        for part in str(inst).split("(", 1)[1].split(")", 1)[1].split("^"):
            dets = tuple(sorted(int(t[1:]) for t in part.split() if t.startswith("D")))
            obs = tuple(sorted(int(t[1:]) for t in part.split() if t.startswith("L")))
            if dets and dets not in edges:
                edges[dets] = obs
    rng = np.random.default_rng(seed)
    H = np.zeros((dem.num_detectors, len(edges)), dtype=np.uint8)
    O = np.zeros((dem.num_observables, len(edges)), dtype=np.uint8)
    for j, (dets, obs) in enumerate(edges.items()):
        H[list(dets), j] = 1
        O[list(obs), j] = 1
    rates = rng.uniform(0.001, 0.2, size=len(edges))
    dets = circuit.compile_detector_sampler(seed=seed).sample(300).astype(np.uint8)
    return H, O, rates, dets


def test_matches_pymatching():
    pymatching = pytest.importorskip("pymatching")
    H, O, rates, shots = _surface_graph(5)
    weights = -np.log(rates / (1 - rates))
    ref = pymatching.Matching(H, weights=weights, faults_matrix=O)
    edges = qec.get_decoder("nv-fusion-decoder", H, error_rate_vec=rates)
    obs = qec.get_decoder("nv-fusion-decoder", H, O=O, error_rate_vec=rates)
    agree = 0
    for s in shots:
        pred, w = ref.decode(s, return_weight=True)
        mine = edges.decode(s.astype(np.float64)).result
        np.testing.assert_array_equal(H @ mine.astype(np.uint8) % 2, s)  # it explains the syndrome
        assert float(weights @ mine) == pytest.approx(w, rel=1e-5, abs=1e-6)  # with a minimum weight
        agree += np.array_equal(obs.decode(s.astype(np.float64)).result.astype(np.uint8), pred)
    assert agree == len(shots)


def test_dem_path_matches_pymatching():
    pymatching = pytest.importorskip("pymatching")
    stim = pytest.importorskip("stim")
    circuit = stim.Circuit.generated("repetition_code:memory", distance=5, rounds=4, after_clifford_depolarization=0.02, before_measure_flip_probability=0.01)
    dem = circuit.detector_error_model(decompose_errors=True)
    ref = pymatching.Matching.from_detector_error_model(dem)
    ours = qec.get_decoder("nv-fusion-decoder", str(dem))
    shots, actual = circuit.compile_detector_sampler(seed=2).sample(400, separate_observables=True)
    pred = ref.decode_batch(shots)
    mine = ours.decode_batch(shots.astype(np.float64)).result.astype(np.uint8)
    assert (mine == pred).all(axis=1).mean() >= 0.98  # (the rest are ties between equal matchings)
    assert abs((mine != actual).any(axis=1).mean() - (pred != actual).any(axis=1).mean()) <= 0.01


def test_sliding_window_inner_decoder():
    # The fusion decoder as a sliding window's inner decoder, against a whole-shot one.
    code = qec.get_code("repetition", distance=5)
    noise = qec.NoiseModel()
    noise.add_all_qubit_channel("x", qec.Depolarization2(0.01), 1)
    dem = qec.z_dem_from_memory_circuit(code, qec.operation.prep0, 6, noise)
    S = code.get_num_z_stabilizers()
    H, rates = qec.simplify_pcm(dem.detector_error_matrix, np.array(dem.error_rates), S)
    rates = np.clip(rates, 1e-6, 0.5)
    full = qec.get_decoder("nv-fusion-decoder", H, error_rate_vec=rates)
    window = qec.get_decoder("sliding_window", H, error_rate_vec=rates, window_size=H.shape[0] // S, step_size=1, num_syndromes_per_round=S, inner_decoder_name="nv-fusion-decoder", inner_decoder_params={})
    syndromes = np.random.default_rng(4).integers(0, 2, size=(20, H.shape[0])).astype(np.float64)
    for s in syndromes:
        np.testing.assert_array_equal(window.decode(s).result, full.decode(s).result)

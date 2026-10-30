# CUDA-Q QEC's real-time decoding host library (NVIDIA/cudaqx 0.8.1: the flow of
# docs/sphinx/examples/qec/python/real_time_complete.py, libs/qec/python/tests/
# test_realtime_decoding_negative.py and test_decoding_config.py's configure tests; Apache License
# 2.0), re-typed and adapted: the kernel-side calls (enqueue_syndromes, get_corrections,
# reset_decoder) run on the host here, fed measurements this package samples.
import warnings

import numpy as np
import pytest

from stabilizer_qec import cudaq_qec as qec


@pytest.fixture(autouse=True)
def _finalize():
    yield
    qec.finalize_decoders()


def multi(*decoders):
    m = qec.multi_decoder_config()
    m.decoders = list(decoders)
    return m


def tiny_lut():
    config = qec.decoder_config()
    config.id = 0
    config.type = "single_error_lut"
    config.block_size = 2
    config.syndrome_size = 1
    config.H_sparse = qec.pcm_to_sparse_vec(np.zeros((1, 2), dtype=np.uint8))
    config.O_sparse = qec.pcm_to_sparse_vec(np.zeros((1, 2), dtype=np.uint8))
    config.D_sparse = qec.generate_timelike_sparse_detector_matrix(1, 2, include_first_round=False)
    return config


def repetition(rounds, p=0.05):
    code = qec.get_code("repetition", distance=3)
    noise = qec.NoiseModel()
    noise.add_all_qubit_channel("x", qec.Depolarization2(p), 1)
    ctx = qec.decoder_context_from_memory_circuit(code, qec.operation.prep0, rounds, noise)
    circuit = qec.memory_circuit(code, qec.operation.prep0, rounds, noise)
    return ctx, circuit


def flat_config(dem, m2d, typ, args):
    H = np.asarray(dem.detector_error_matrix)
    config = qec.decoder_config()
    config.id = 0
    config.type = typ
    config.block_size = H.shape[1]
    config.syndrome_size = H.shape[0]
    config.H_sparse = qec.pcm_to_sparse_vec(H)
    config.O_sparse = qec.pcm_to_sparse_vec(np.asarray(dem.observables_flips_matrix))
    config.D_sparse = qec.d_sparse(m2d)
    config.decoder_custom_args = args
    config.validate_custom_args()
    return config


def stream(meas, rounds, decoder_id=0):
    """real_time_complete.py's kernel: each round's two ancilla bits, then the data."""
    for r in range(rounds):
        qec.enqueue_syndromes(decoder_id, meas[2 * r : 2 * r + 2], 0)
    qec.enqueue_syndromes(decoder_id, meas[2 * rounds :], 0)


def offline_prediction(decoder, m2d, O, meas):
    det = np.array([np.bitwise_xor.reduce(meas[row]) for row in m2d], dtype=np.float64)
    errors = (np.asarray(decoder.decode(det).result) >= 0.5).astype(int)
    return bool((O.astype(int) @ errors)[0] % 2)


# -- configure_decoders -----------------------------------------------------------------------------


def test_configure_status_codes():
    a, b = tiny_lut(), tiny_lut()
    with pytest.warns(RuntimeWarning, match="Duplicate decoder ID found: 0"):
        assert qec.configure_decoders(multi(a, b)) == 1
    b.id = 2
    with pytest.warns(RuntimeWarning, match="Maximum decoder ID is greater than the number of decoders: 2 >= 2"):
        assert qec.configure_decoders(multi(a, b)) == 2
    with pytest.warns(RuntimeWarning, match="-9223372036854775808 >= 0"):
        assert qec.configure_decoders(multi()) == 2
    b.id = -1
    with pytest.warns(RuntimeWarning, match="Minimum decoder ID is less than 0: -1"):
        assert qec.configure_decoders(multi(b, a)) == 3
    b.id = 1
    assert qec.configure_decoders(multi(a, b)) == 0


def test_configure_invalid_decoders_returns_4():
    config = tiny_lut()
    config.type = "invalid-decoder"
    with pytest.warns(RuntimeWarning, match="Error initializing decoders: invalid decoder requested: invalid-decoder"):
        assert qec.configure_decoders(multi(config)) == 4
    # Python-registered decoders are not reachable from CUDA-Q's realtime library either.
    config.type = "example_byod"
    with pytest.warns(RuntimeWarning, match="invalid decoder requested"):
        assert qec.configure_decoders(multi(config)) == 4
    config = tiny_lut()
    config.D_sparse = []
    with pytest.warns(RuntimeWarning, match="D_sparse must be provided in decoder configuration"):
        assert qec.configure_decoders(multi(config)) == 4


def pymatching_config(args, h_sparse=None):
    dc = qec.decoder_config()
    dc.id = 0
    dc.type = "pymatching"
    dc.block_size = 3
    dc.syndrome_size = 3
    dc.H_sparse = h_sparse if h_sparse is not None else [0, -1, 1, -1, 2, -1]
    dc.O_sparse = [0, -1, 1, -1, 2, -1]
    dc.D_sparse = [0, -1, 1, -1, 2, -1]
    dc.decoder_custom_args = args
    return multi(dc)


def test_configure_pymatching_valid_and_invalid():
    good = {"error_rate_vec": [0.1, 0.1, 0.1], "merge_strategy": "smallest_weight"}
    assert qec.configure_decoders(pymatching_config(good)) == 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for rates in ([0.1, 0.1], [0.0, 0.1, 0.1], [0.1, 0.6, 0.1]):
            assert qec.configure_decoders(pymatching_config(dict(good, error_rate_vec=rates))) != 0
        assert qec.configure_decoders(pymatching_config(dict(good, merge_strategy="not-a-strategy"))) != 0
        assert qec.configure_decoders(pymatching_config(good, h_sparse=[0, -1, 0, -1, 0, -1])) != 0


def test_configure_decoders_rejects_invalid_o_sparse():
    config = tiny_lut()
    config.O_sparse = [config.block_size + 1, -1]
    with pytest.raises(RuntimeError, match="out of range"):
        qec.configure_decoders(multi(config))


def test_configure_from_str_and_file(tmp_path):
    text = multi(tiny_lut()).to_yaml_str()
    assert qec.configure_decoders_from_str(text) == 0
    path = tmp_path / "decoders.yaml"
    path.write_text(text, encoding="utf-8")
    assert qec.configure_decoders_from_file(str(path)) == 0
    with pytest.warns(RuntimeWarning, match="Config file does not exist"):
        assert qec.configure_decoders_from_file(str(tmp_path / "missing.yaml")) == 1
    assert qec.qecrt.config.configure_decoders_from_str(text) == 0
    with pytest.warns(RuntimeWarning, match="Maximum decoder ID"):
        assert qec.configure_decoders_from_str(qec.multi_decoder_config().to_yaml_str()) == 2


# -- The host calls' errors --------------------------------------------------------------------------


def test_get_corrections_and_enqueue_reject_bad_arguments():
    assert qec.configure_decoders(multi(tiny_lut())) == 0
    with pytest.raises(ValueError, match="correction_length must be greater"):
        qec.get_corrections(0, 0, False)
    with pytest.raises(ValueError, match="does not match number of observables"):
        qec.get_corrections(0, 2, False)
    with pytest.raises(ValueError, match="Decoder 1 not found"):
        qec.enqueue_syndromes(1, [True, False], 0)
    with pytest.raises(ValueError, match=r"syndrome_length \(3\) exceeds configured measurement count \(2\)"):
        qec.enqueue_syndromes(0, [True, False, True], 0)
    with pytest.raises(ValueError, match="syndrome_length must be greater than 0"):
        qec.enqueue_syndromes(0, [], 0)
    with pytest.raises(ValueError, match="Decoder 3 not found"):
        qec.reset_decoder(3)
    qec.finalize_decoders()
    with pytest.raises(ValueError, match="Decoder 0 not found"):
        qec.get_corrections(0, 1, False)


def test_overflow_is_reported_and_ignored(capsys):
    config = tiny_lut()  # (two measurements a shot)
    assert qec.configure_decoders(multi(config)) == 0
    qec.enqueue_syndromes(0, [1], 0)
    qec.enqueue_syndromes(0, [1, 1], 0)
    assert capsys.readouterr().out == "Syndrome buffer overflow. Syndrome will be ignored.\n"
    qec.enqueue_syndromes(0, [1], 0)  # (completes the shot)
    assert capsys.readouterr().out == ""


# -- Decoding ---------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "typ, args",
    [("multi_error_lut", {"lut_error_depth": 2}), ("single_error_lut", {}), ("pymatching", {}), ("nv-qldpc-decoder", {"use_osd": True}), ("nv-fusion-decoder", {})],
)
def test_real_time_complete_flow_matches_offline_decoding(typ, args):
    rounds = 3
    ctx, circuit = repetition(rounds)
    dem, m2d, m2o = ctx.full_component()
    if typ == "nv-qldpc-decoder":
        pytest.importorskip("scipy")  # (built on the ldpc port, which needs it)
        args = dict(args, error_rate_vec=list(dem.error_rates))
    assert qec.configure_decoders_from_str(multi(flat_config(dem, m2d, typ, args)).to_yaml_str(200)) == 0
    O = np.asarray(dem.observables_flips_matrix)
    offline_args = dict(args)
    offline = qec.get_decoder(typ, np.asarray(dem.detector_error_matrix), **offline_args)
    meas = circuit.compile_sampler(seed=7).sample(200).astype(np.uint8)
    raw = corrected = 0
    for m in meas:
        qec.reset_decoder(0)
        stream(m, rounds)
        corr = qec.get_corrections(0, 1, False)
        assert corr == [offline_prediction(offline, m2d, O, m)]
        actual = bool(np.bitwise_xor.reduce(m[m2o[0]]))
        raw += actual
        corrected += actual ^ corr[0]
    assert corrected < raw


def test_corrections_accumulate_until_reset():
    rounds = 3
    ctx, circuit = repetition(rounds, p=0.1)
    dem, m2d, _ = ctx.full_component()
    assert qec.configure_decoders(multi(flat_config(dem, m2d, "multi_error_lut", {"lut_error_depth": 2}))) == 0
    offline = qec.get_decoder("multi_error_lut", np.asarray(dem.detector_error_matrix), lut_error_depth=2)
    O = np.asarray(dem.observables_flips_matrix)
    parity = False
    for m in circuit.compile_sampler(seed=11).sample(40).astype(np.uint8):
        stream(m, rounds)
        parity ^= offline_prediction(offline, m2d, O, m)
        assert qec.get_corrections(0, 1, False) == [parity]
    assert qec.get_corrections(0, 1, True) == [parity]
    assert qec.get_corrections(0, 1, False) == [False]
    # reset_decoder drops a partial shot too.
    m = circuit.compile_sampler(seed=12).sample(1).astype(np.uint8)[0]
    qec.enqueue_syndromes(0, [1, 1], 0)
    qec.reset_decoder(0)
    stream(m, rounds)
    assert qec.get_corrections(0, 1, True) == [offline_prediction(offline, m2d, O, m)]


def test_sliding_window_decodes_a_layer_a_round():
    rounds = 5
    ctx, circuit = repetition(rounds)
    dem, m2d, _ = ctx.full_component()
    args = {"window_size": 2, "step_size": 1, "num_syndromes_per_round": 2, "error_rate_vec": list(dem.error_rates), "inner_decoder_name": "multi_error_lut", "inner_decoder_params": {"lut_error_depth": 2}}
    assert qec.configure_decoders(multi(flat_config(dem, m2d, "sliding_window", args))) == 0
    offline = qec.get_decoder("sliding_window", np.asarray(dem.detector_error_matrix), **args)
    O = np.asarray(dem.observables_flips_matrix)
    for m in circuit.compile_sampler(seed=3).sample(100).astype(np.uint8):
        stream(m, rounds)
        assert qec.get_corrections(0, 1, True) == [offline_prediction(offline, m2d, O, m)]


def test_stim_dem_path(tmp_path):
    rounds = 3
    ctx, circuit = repetition(rounds)
    path = tmp_path / "memory.dem"
    path.write_text(str(circuit.detector_error_model()), encoding="utf-8")
    D = [x for row in ctx._m2d for x in row + [-1]]
    dem, m2d, _ = ctx.full_component()
    flat = flat_config(dem, m2d, "multi_error_lut", {"lut_error_depth": 2})
    flat.id = 1
    configs = []
    for typ in ("multi_error_lut", "pymatching"):
        cfg = qec.decoder_config()
        cfg.id = 0
        cfg.type = typ
        cfg.stim_dem_path = str(path)
        cfg.D_sparse = D
        configs.append(cfg)
    meas = circuit.compile_sampler(seed=5).sample(100).astype(np.uint8)
    for cfg in configs:
        assert qec.configure_decoders_from_str(multi(cfg, flat).to_yaml_str()) == 0
        agree = 0
        for m in meas:
            stream(m, rounds, 0)
            stream(m, rounds, 1)
            agree += qec.get_corrections(0, 1, True) == qec.get_corrections(1, 1, True)
        assert agree >= 95  # (the same model, its columns in another order: ties may break apart)
    bad = configs[0]
    bad.D_sparse = D[: D.index(-1) + 1]
    with pytest.warns(RuntimeWarning, match="Number of rows in D_sparse vector is not equal to the number of detectors in"):
        assert qec.configure_decoders(multi(bad)) == 4
    bad.stim_dem_path = str(tmp_path / "missing.dem")
    with pytest.warns(RuntimeWarning, match="stim_dem_path is not a readable file"):
        assert qec.configure_decoders(multi(bad)) == 4


def test_queue_form_still_works_beside_cudaqs():
    H = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
    lut = qec.get_decoder("single_error_lut", H)
    assert qec.configure_decoders(multi(tiny_lut())) == 0
    assert qec.configure_decoders({"q": lut}) == ["q"]
    t = qec.enqueue_syndromes("q", np.array([[1, 0]], dtype=np.uint8))
    assert np.asarray(qec.get_corrections("q", t)[0].result).tolist() == [1.0, 0.0, 0.0]
    qec.enqueue_syndromes(0, [1, 1], 0)  # (decoder 0 is still CUDA-Q's)
    # Configuring id 0 in the queue form takes it from CUDA-Q's, and back.
    assert qec.configure_decoders([lut]) == [0]
    assert isinstance(qec.enqueue_syndromes(0, np.array([[0, 1]], dtype=np.uint8)), int)
    assert qec.configure_decoders(multi(tiny_lut())) == 0
    qec.enqueue_syndromes(0, [1, 1], 0)
    with pytest.raises(KeyError):
        qec.enqueue_syndromes("other", np.zeros((1, 2)))
    qec.finalize_decoders()
    with pytest.raises(KeyError):
        qec.get_corrections("q")

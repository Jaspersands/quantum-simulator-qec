# Tests of stabilizer_qec.cudaq_qec's "nv-qldpc-decoder": CUDA-Q QEC's documented parameters
# (NVIDIA/cudaqx 0.8.1, docs/sphinx/api/qec/nv_qldpc_decoder_api.rst and
# docs/sphinx/examples/qec/python/nv-qldpc-decoder.py; Apache License 2.0) mapped onto this package's
# BP, BP+OSD and Relay-BP decoders, checked against the ldpc and relay_bp packages, and the
# documented values the port cannot run refused.
import numpy as np
import pytest

from stabilizer_qec import cudaq_qec as qec

H7 = np.array([[1, 0, 0, 1, 0, 1, 1], [0, 1, 0, 1, 1, 0, 1], [0, 0, 1, 0, 1, 1, 1]], dtype=np.uint8)


def _hgp_x(n=5):
    """The X checks of the hypergraph product of a length-n ring code with itself."""
    ring = np.zeros((n, n), dtype=np.uint8)
    for i in range(n):
        ring[i, i] = ring[i, (i + 1) % n] = 1
    eye = np.eye(n, dtype=np.uint8)
    return np.hstack([np.kron(ring, eye), np.kron(eye, ring.T)]) % 2


def _shots(H, p, count, seed):
    rng = np.random.default_rng(seed)
    errors = (rng.random((count, H.shape[1])) < p).astype(np.uint8)
    return errors @ H.T % 2


def test_documented_bp_methods():
    pytest.importorskip("scipy")  # (nv-qldpc-decoder is built on the ldpc port, which needs it)
    # The demonstrate_bp_methods() configurations of CUDA-Q's example all build and decode.
    configs = [
        dict(bp_method=0, max_iterations=30),
        dict(bp_method=1, max_iterations=30, scale_factor=1.0),
        dict(bp_method=2, max_iterations=30, use_sparsity=True, gamma0=0.5),
        dict(bp_method=3, composition=1, max_iterations=50, use_sparsity=True, gamma0=0.3, gamma_dist=[0.1, 0.5], srelay_config={"pre_iter": 5, "num_sets": 3, "stopping_criterion": "FirstConv"}, bp_seed=42),
        dict(bp_method=3, composition=1, max_iterations=50, use_sparsity=True, gamma0=0.3, explicit_gammas=[[0.1 + 0.02 * i for i in range(7)]] * 3, srelay_config={"pre_iter": 10, "num_sets": 3, "stopping_criterion": "All"}),
    ]
    for kw in configs:
        decoder = qec.get_decoder("nv-qldpc-decoder", H7, **kw)
        assert decoder.get_syndrome_size() == 3 and decoder.get_block_size() == 7
        for s in ([1, 0, 1], [0, 1, 0], [0, 0, 0]):
            r = decoder.decode(s)
            assert r.converged
            np.testing.assert_array_equal(H7 @ r.result.astype(np.uint8) % 2, s)


def test_sparse_input_and_batch():
    scipy_sparse = pytest.importorskip("scipy.sparse")
    dense = qec.get_decoder("nv-qldpc-decoder", H7, use_osd=True)
    sparse = qec.get_decoder("nv-qldpc-decoder", scipy_sparse.csr_matrix(H7), use_osd=True)
    syndromes = [[1, 0, 1], [1, 1, 1], [0, 0, 1]]
    batch = sparse.decode_batch(syndromes)
    for s, r in zip(syndromes, batch):
        assert r.result.tolist() == dense.decode(s).result.tolist()


@pytest.mark.parametrize(
    "kw, ref_kw",
    [
        (dict(), dict(bp_method="product_sum")),
        (dict(bp_method=1, scale_factor=0.75), dict(bp_method="minimum_sum", ms_scaling_factor=0.75)),
        (dict(bp_method=1, scale_factor=0.0), dict(bp_method="minimum_sum", ms_scaling_factor=0.0)),
        (dict(use_osd=True), dict(bp_method="product_sum", osd_method="osd0")),
        (dict(use_osd=True, osd_method=2, osd_order=3), dict(bp_method="product_sum", osd_method="osd_e", osd_order=3)),
        (dict(use_osd=True, osd_method=3, osd_order=4, bp_method=1), dict(bp_method="minimum_sum", osd_method="osd_cs", osd_order=4)),
    ],
)
def test_bp_and_osd_match_ldpc(kw, ref_kw):
    ldpc = pytest.importorskip("ldpc", minversion="2.4")
    H = _hgp_x()
    rng = np.random.default_rng(7)
    rates = rng.uniform(0.01, 0.1, size=H.shape[1])
    ours = qec.get_decoder("nv-qldpc-decoder", H, error_rate_vec=rates, max_iterations=12, opt_results={"num_iter": True}, **kw)
    cls = ldpc.BpOsdDecoder if "osd_method" in ref_kw else ldpc.BpDecoder
    ref = cls(H, error_channel=list(rates), max_iter=12, schedule="parallel", **ref_kw)
    converged = []
    for s in _shots(H, 0.06, 150, 3):
        r = ours.decode(s.astype(np.float64))
        expected = ref.decode(s)
        np.testing.assert_array_equal(r.result.astype(np.uint8), expected)
        assert r.converged == bool(ref.converge)
        if s.any():
            assert r.opt_results["num_iter"] == ref.iter
        converged.append(r.converged)
    assert 0 < np.mean(converged) < 1  # (both outcomes exercised)


def test_default_error_rate_and_observables():
    ldpc = pytest.importorskip("ldpc", minversion="2.4")
    H = _hgp_x()
    O = np.zeros((2, H.shape[1]), dtype=np.uint8)
    O[0, :5] = 1
    O[1, 25:30] = 1
    ours = qec.get_decoder("nv-qldpc-decoder", H, O=O, use_osd=True)  # error_rate 0.001, 30 iterations
    errs = qec.get_decoder("nv-qldpc-decoder", H, O=O, use_osd=True, output="errors")
    ref = ldpc.BpOsdDecoder(H, error_rate=0.001, max_iter=30, bp_method="product_sum", osd_method="osd0", schedule="parallel")
    for s in _shots(H, 0.05, 60, 4):
        r = ours.decode(s)
        assert r.result.size == 2
        np.testing.assert_array_equal(r.result.astype(np.uint8), O @ ref.decode(s) % 2)
        np.testing.assert_array_equal(errs.decode(s).result.astype(np.uint8), ref.decode(s))


def test_mem_bp_matches_relay_bp():
    relay_bp = pytest.importorskip("relay_bp")
    H = _hgp_x()
    rates = np.full(H.shape[1], 0.05)
    for proc, cls in (("fp64", relay_bp.MinSumBPDecoderF64), ("fp32", relay_bp.MinSumBPDecoderF32)):
        ours = qec.get_decoder("nv-qldpc-decoder", H, error_rate_vec=rates, bp_method=2, use_sparsity=True, gamma0=0.4, scale_factor=0.9, max_iterations=40, proc_float=proc, opt_results={"num_iter": True})
        ref = cls(H, rates, max_iter=40, alpha=0.9, gamma0=0.4)
        for s in _shots(H, 0.06, 80, 5):
            r = ours.decode(s)
            d = ref.decode_detailed(s)
            np.testing.assert_array_equal(r.result.astype(np.uint8), d.decoding)
            assert (r.converged, r.opt_results["num_iter"]) == (d.success, d.iterations)


@pytest.mark.parametrize("criterion, ref_criterion, stop_nconv", [("All", "all", 1), ("FirstConv", "nconv", 1), ("NConv", "nconv", 3)])
def test_relay_matches_relay_bp(criterion, ref_criterion, stop_nconv):
    relay_bp = pytest.importorskip("relay_bp")
    H = _hgp_x()
    rates = np.random.default_rng(2).uniform(0.02, 0.08, size=H.shape[1])
    srelay = {"pre_iter": 20, "num_sets": 12, "stopping_criterion": criterion}
    if criterion == "NConv":
        srelay["stop_nconv"] = stop_nconv
    ours = qec.get_decoder("nv-qldpc-decoder", H, error_rate_vec=rates, bp_method=3, composition=1, use_sparsity=True, gamma0=0.1, gamma_dist=[-0.24, 0.66], srelay_config=srelay, max_iterations=25, bp_seed=11, opt_results={"num_iter": True})
    ref = relay_bp.RelayDecoderF64(H, rates, gamma0=0.1, pre_iter=20, num_sets=12, set_max_iter=25, gamma_dist_interval=(-0.24, 0.66), stop_nconv=stop_nconv, stopping_criterion=ref_criterion, seed=11, alpha=1.0)
    for s in _shots(H, 0.07, 60, 6):
        r = ours.decode(s)
        d = ref.decode_detailed(s)
        np.testing.assert_array_equal(r.result.astype(np.uint8), d.decoding)
        assert (r.converged, r.opt_results["num_iter"]) == (d.success, d.iterations)


def test_relay_explicit_gammas_one_row_per_leg():
    relay_bp = pytest.importorskip("relay_bp")
    H = _hgp_x()
    rates = np.full(H.shape[1], 0.05)
    rng = np.random.default_rng(9)
    gammas = rng.uniform(-0.2, 0.6, size=(4, H.shape[1]))
    ours = qec.get_decoder("nv-qldpc-decoder", H, error_rate_vec=rates, bp_method=3, composition=1, use_sparsity=True, gamma0=0.2, explicit_gammas=gammas.tolist(), srelay_config={"pre_iter": 8, "num_sets": 4, "stopping_criterion": "All"}, max_iterations=20)
    # Leg k (from 0) runs with row k; relay_bp gives leg k (from 1) row k mod rows.
    ref = relay_bp.RelayDecoderF64(H, rates, gamma0=0.2, pre_iter=8, num_sets=4, set_max_iter=20, explicit_gammas=np.roll(gammas, 1, axis=0), stopping_criterion="all")
    for s in _shots(H, 0.07, 40, 8):
        np.testing.assert_array_equal(ours.decode(s).result.astype(np.uint8), ref.decode(s))


def test_sliding_window_inner_nv_qldpc():
    pytest.importorskip("scipy")  # (nv-qldpc-decoder is built on the ldpc port, which needs it)
    code = qec.get_code("repetition", distance=5)
    noise = qec.NoiseModel()
    noise.add_all_qubit_channel("x", qec.Depolarization2(0.01), 1)
    dem = qec.z_dem_from_memory_circuit(code, qec.operation.prep0, 6, noise)
    S = code.get_num_z_stabilizers()
    H, rates = qec.simplify_pcm(dem.detector_error_matrix, np.array(dem.error_rates), S)
    inner = {"use_osd": True, "max_iterations": 20}
    window = qec.get_decoder("sliding_window", H, error_rate_vec=rates, window_size=H.shape[0] // S, step_size=1, num_syndromes_per_round=S, inner_decoder_name="nv-qldpc-decoder", inner_decoder_params=inner)
    whole = qec.get_decoder("nv-qldpc-decoder", H, error_rate_vec=rates, **inner)
    for s in np.random.default_rng(1).integers(0, 2, size=(10, H.shape[0])).astype(np.float64):
        np.testing.assert_array_equal(window.decode(s).result, whole.decode(s).result)
    # A real window (3 of the rounds, sliding by 1) runs too.
    small = qec.get_decoder("sliding_window", H, error_rate_vec=rates, window_size=3, step_size=1, num_syndromes_per_round=S, inner_decoder_name="nv-qldpc-decoder", inner_decoder_params=inner)
    assert small.decode(np.zeros(H.shape[0])).result.tolist() == [0.0] * H.shape[1]


REFUSED = [
    (dict(bp_method=4, use_sparsity=True, gamma0=0.1), "bp_method=4.*not supported"),
    (dict(bp_method=5, use_sparsity=True, gamma_dist=[0.1, 0.2]), "bp_method=5.*not supported"),
    (dict(bp_method=3, use_sparsity=True, gamma_dist=[0.1, 0.5]), "outside sequential relay.*not supported"),
    (dict(bp_method=2, use_sparsity=True, gamma0=0.1, use_osd=True), "use_osd with bp_method=2.*not supported"),
    (dict(clip_value=10.0), "clip_value.*not supported"),
    (dict(iter_per_check=2), "iter_per_check.*not supported"),
    (dict(opt_results={"bp_llr_history": 5}), "bp_llr_history.*not supported"),
    (dict(bp_method=3, composition=1, use_sparsity=True, gamma0=0.1, gamma_dist=[0.0, 0.5], srelay_config={"pre_iter": 1, "num_sets": 2}, opt_results={"relay_solutions": True}), "relay_solutions.*not supported"),
    (dict(bp_method=3, composition=1, use_sparsity=True, gamma0=0.1, gamma_dist=[0.0, 0.5], srelay_config={"pre_iter": 1, "num_sets": 2}, gamma_ensemble_size=2), "gamma_ensemble_size > 1.*not supported"),
    (dict(proc_float="fp32"), "fp32.*sum-product.*not supported"),
    # CUDA-Q's own requirements.
    (dict(bp_method=2, gamma0=0.1), "requires use_sparsity=True"),
    (dict(bp_method=2, use_sparsity=True), "requires gamma0"),
    (dict(bp_method=1, composition=1, use_sparsity=True), "requires bp_method=3"),
    (dict(bp_method=3, composition=1, use_sparsity=True, gamma0=0.1, gamma_dist=[0.0, 0.5]), "requires srelay_config"),
    (dict(bp_method=3, composition=1, use_sparsity=True, gamma_dist=[0.0, 0.5], srelay_config={"pre_iter": 1, "num_sets": 2}), "requires gamma0"),
    (dict(bp_method=3, composition=1, use_sparsity=True, gamma0=0.1, gamma_dist=[0.0, 0.5], srelay_config={"pre_iter": 1, "num_sets": 2, "stopping_criterion": "NConv"}), "requires stop_nconv"),
    (dict(repeatable=True), "requires a non-zero clip_value"),
    (dict(bp_method=7), "bp_method must be 0-5"),
    (dict(proc_float="fp16"), "proc_float must be"),
    (dict(gamma_ensemble_size=3), "1, 2, 4 or 8"),
    (dict(opt_results={"weight": True}), "unknown opt_results entry"),
    (dict(max_iterations="many"), "must be a int"),
    (dict(error_rate_vec=np.full(6, 0.01)), "error_rate_vec must have block_size"),
    (dict(cuda_device_id=0), "0 CUDA device"),
]


@pytest.mark.parametrize("kw, match", REFUSED)
def test_refusals(kw, match):
    with pytest.raises(RuntimeError, match=match):
        qec.get_decoder("nv-qldpc-decoder", H7, **kw)


def test_accepted_performance_options():
    pytest.importorskip("scipy")  # (nv-qldpc-decoder is built on the ldpc port, which needs it)
    # GPU kernel and batching choices change nothing here.
    base = qec.get_decoder("nv-qldpc-decoder", H7, use_osd=True)
    tuned = qec.get_decoder("nv-qldpc-decoder", H7, use_osd=True, use_sparsity=True, n_threads=128, bp_batch_size=1000, osd_batch_size=8, iter_per_check=1, clip_value=0.0, repeatable=False, gamma_ensemble_size=1, proc_float="fp64", opt_results={"bp_llr_history": 0, "relay_solutions": False})
    for s in ([1, 0, 1], [1, 1, 1], [0, 1, 1]):
        assert base.decode(s).result.tolist() == tuned.decode(s).result.tolist()


def test_from_dem_text_decodes_to_observables():
    pytest.importorskip("scipy")  # (nv-qldpc-decoder is built on the ldpc port, which needs it)
    dem = "error(0.1) D0 L0\nerror(0.1) D0 D1\nerror(0.1) D1\n"
    decoder = qec.get_decoder("nv-qldpc-decoder", dem, use_osd=True)
    assert decoder.decode([1.0, 0.0]).result.tolist() == [1.0]
    assert decoder.decode([0.0, 1.0]).result.tolist() == [0.0]

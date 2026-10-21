# ============================================================================ #
# Copyright (c) 2024 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
# Ported from CUDA-Q QEC 0.8.1's libs/qec/python/tests/test_trt_decoder.py and
# unittests/decoders/trt_decoder/test_trt_decoder.cpp (NVIDIA/cudaqx, now NVIDIA/cudaq-qec; Apache
# License 2.0); modified for stabilizer_qec: cudaq_qec is stabilizer_qec.cudaq_qec, and trt_decoder
# runs its ONNX model on the CPU (._onnx), so the tests CUDA-Q skips without a GPU run here. The C++
# fixtures are rebuilt by tools/cudaq_trt_models.py, and the surface-code model is CUDA-Q's own
# (models/surface_code_decoder.onnx, its git-lfs object). A TensorRT engine is a GPU binary, so
# engine_load_path and engine_save_path are refused and the engine-building tests are not ported.
# Added: the evaluator against onnxruntime's recorded outputs, TensorRT's quantization rules, the
# parameter schema, and malformed models (truncated, unknown ops, two inputs, fuzzed bytes).
import hashlib
import os
import re
import warnings

import numpy as np
import pytest

from stabilizer_qec import cudaq_qec as qec
from stabilizer_qec.cudaq_qec._onnx import OnnxModel, load_onnx

MODELS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
ONNX_MODEL_PATH = os.path.join(MODELS, "surface_code_decoder.onnx")
PREFIX = "TensorRT initialization failed: "

# Test data constants
NUM_DETECTORS = 24
NUM_OBSERVABLES = 1


def model_path(name):
    return os.path.join(MODELS, name)


def _test_inputs():
    """CUDA-Q's 30 test cases: no detector, each detector alone, then five pairs."""
    rows = [[0.0] * NUM_DETECTORS]
    for k in range(NUM_DETECTORS):
        rows.append([1.0 if j == k else 0.0 for j in range(NUM_DETECTORS)])
    for a, b in ((0, 5), (3, 11), (7, 19), (12, 20), (2, 23)):
        rows.append([1.0 if j in (a, b) else 0.0 for j in range(NUM_DETECTORS)])
    return rows


TEST_INPUTS = _test_inputs()

# Expected outputs from PyTorch model
TEST_OUTPUTS = [[0.546192], [0.542916], [0.534589], [0.559515], [0.510735],
                [0.526783], [0.536010], [0.522932], [0.563812], [0.560833],
                [0.578034], [0.555534], [0.542923], [0.495684], [0.545392],
                [0.562221], [0.532649], [0.559208], [0.534731], [0.499671],
                [0.497873], [0.511810], [0.566766], [0.527120], [0.573466],
                [0.523840], [0.504422], [0.507719], [0.474930], [0.583623]]


def no_warnings():
    """A context in which any warning fails the test."""
    ctx = warnings.catch_warnings()
    ctx.__enter__()
    warnings.simplefilter("error")
    return ctx


# -- ONNX models written here (protobuf by hand) --------------------------------------------------

FLOAT, UINT8, INT8, INT32, INT64 = 1, 2, 3, 6, 7


def _uvarint(v):
    v &= (1 << 64) - 1
    out = bytearray()
    while True:
        b = v & 0x7F
        v >>= 7
        if v:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


def _int(field, v):
    return _uvarint(field << 3) + _uvarint(v)


def _len(field, payload):
    return _uvarint(field << 3 | 2) + _uvarint(len(payload)) + payload


def _str(field, s):
    return _len(field, s.encode())


def value_info(name, elem, dims):
    shape = b"".join(_len(1, _int(1, d) if isinstance(d, int) else _str(2, d)) for d in dims)
    return _str(1, name) + _len(2, _len(1, _int(1, elem) + _len(2, shape)))


def tensor(name, arr, elem):
    arr = np.ascontiguousarray(arr)
    return b"".join(_int(1, d) for d in arr.shape) + _int(2, elem) + _str(8, name) + _len(9, arr.tobytes())


def node(op, ins, outs, **attrs):
    out = b"".join(_str(1, i) for i in ins) + b"".join(_str(2, o) for o in outs) + _str(4, op)
    for k, v in attrs.items():
        if isinstance(v, float):
            a = _str(1, k) + _uvarint(2 << 3 | 5) + np.float32(v).tobytes() + _int(20, 1)
        else:
            a = _str(1, k) + _int(3, v) + _int(20, 2)
        out += _len(5, a)
    return out


def onnx_model(nodes, inputs, outputs, inits=(), opset=19):
    graph = b"".join(_len(1, n) for n in nodes) + _str(2, "g") + b"".join(_len(5, t) for t in inits)
    graph += b"".join(_len(11, v) for v in inputs) + b"".join(_len(12, v) for v in outputs)
    return _int(1, 9) + _len(7, graph) + _len(8, _int(2, opset))


def write(tmp_path, name, data):
    p = tmp_path / name
    p.write_bytes(data)
    return str(p)


def matmul_model(tmp_path, batch, size, seed=0):
    """CUDA-Q's batch-validation network: [batch, size] @ W (size x size), as ONNX."""
    W = (np.random.default_rng(seed).standard_normal((size, size)) * 0.1).astype(np.float32)
    data = onnx_model([node("MatMul", ["syndrome", "W"], ["correction"])], [value_info("syndrome", FLOAT, [batch, size])], [value_info("correction", FLOAT, [batch, size])], [tensor("W", W, FLOAT)])
    return write(tmp_path, f"matmul_batch{batch}.onnx", data), W


def mlas_matmul(x, W):
    """x @ W in float32 as onnxruntime's MLAS sums it (for fewer than 256 rows of W): each output
    by fused multiply-adds, in order."""
    acc = np.zeros((x.shape[0], W.shape[1]), dtype=np.float32)
    for k in range(x.shape[1]):
        acc = (x[:, k : k + 1].astype(np.float64) * W[k].astype(np.float64) + acc).astype(np.float32)
    return acc


# -- The fixtures ---------------------------------------------------------------------------------


def test_fixtures_are_the_reference_models():
    # (sha256 and size as CUDA-Q 0.8.1's git-lfs pointers give them.)
    lfs = {
        "qdq_int8.onnx": ("6b8493417ef19183262edff7254b542bf59a108e7574c39a0730f238634d5a9a", 217),
        "qdq_fp8.onnx": ("250f08fa38dbebdcc71b7cace77f300362e22467e219cc2c6e9de48f420739b8", 216),
        "surface_code_decoder.onnx": ("f739c30d81b9fcdf668db6ba4211fc5616c8b462b41fdae07ec2d6e0c069064c", 191673),
    }
    for name, (digest, size) in lfs.items():
        with open(model_path(name), "rb") as f:
            data = f.read()
        assert (hashlib.sha256(data).hexdigest(), len(data)) == (digest, size), name


def test_load_onnx_reads_the_surface_model():
    m = load_onnx(ONNX_MODEL_PATH)
    assert isinstance(m, OnnxModel)
    assert m.inputs == [("detectors", "float32", (1, 24))]
    assert m.outputs == [("data_qubit_probs", "float32", (1, 1))]
    assert (m.ir_version, m.opset, m.producer_name) == (8, 17, "pytorch")
    assert [n for n in m.op_types] == ["Gemm", "Relu", "Gemm", "Relu", "Gemm", "Relu", "Gemm", "Sigmoid"]
    assert load_onnx(model_path("identity_dynamic.onnx")).inputs == [("input", "float32", ("batch", 3))]


# -- The evaluator against onnxruntime ------------------------------------------------------------

# onnxruntime 1.31.0's outputs (CPU, macOS arm64), recorded by tools/cudaq_trt_models.py. The
# evaluator computes float32 as MLAS does (its order of fused multiply-adds, its logistic), in
# float64 arithmetic that is the same everywhere: the comparison is exact.
RECORDED = np.load(os.path.join(MODELS, "ort_outputs.npz"))
STEMS = sorted({k.split("__")[0] for k in RECORDED.files if not k.startswith("meta__")})


def test_recordings_cover_every_model():
    assert str(RECORDED["meta__onnxruntime"]) == "1.31.0"
    assert sorted(STEMS) == sorted(f[: -len(".onnx")] for f in os.listdir(MODELS) if f.endswith(".onnx"))


@pytest.mark.parametrize("stem", STEMS)
def test_evaluator_matches_onnxruntime(stem):
    model = load_onnx(model_path(stem + ".onnx"))
    k = 0
    while f"{stem}__{k}__x" in RECORDED.files:
        x, want = RECORDED[f"{stem}__{k}__x"], RECORDED[f"{stem}__{k}__y"]
        got = model.run(x)
        assert got.dtype == want.dtype and got.shape == want.shape, (stem, k)
        np.testing.assert_array_equal(got, want, err_msg=f"{stem} case {k}")
        k += 1
    assert k > 0


def test_run_checks_its_input():
    m = load_onnx(model_path("identity_uint8.onnx"))
    with pytest.raises(ValueError, match=re.escape("input 'input' has shape (1, 4); the model takes (1, 3)")):
        m.run(np.zeros((1, 4)))
    with pytest.raises(ValueError, match="has shape"):
        m.run(np.zeros(3))
    # (Values are converted to the input's type, as onnxruntime would need them given.)
    assert m.run(np.array([[1.0, 2.0, 3.0]])).dtype == np.uint8


# -- Inference: the surface-code model (TestTRTDecoderInference) ------------------------------------


@pytest.fixture(scope="module")
def surface_decoder():
    return qec.get_decoder("trt_decoder", np.eye(NUM_DETECTORS, dtype=np.uint8), onnx_load_path=ONNX_MODEL_PATH)


def test_validate_against_pytorch_model(surface_decoder):
    """The decoder reproduces the 30 PyTorch outputs CUDA-Q recorded."""
    TOLERANCE = 1e-4
    errors = []
    for test_input, test_output in zip(TEST_INPUTS, TEST_OUTPUTS):
        result = surface_decoder.decode(test_input)
        assert len(result.result) > 0
        errors.append(abs(result.result[0] - test_output[0]))
    assert max(errors) < TOLERANCE, max(errors)


def test_validate_single_test_case(surface_decoder):
    result = surface_decoder.decode(TEST_INPUTS[0])
    assert abs(result.result[0] - TEST_OUTPUTS[0][0]) < 1e-4, "Single test case validation failed"
    assert result.converged, "Decoder did not converge"


def test_decoder_result_structure(surface_decoder):
    result = surface_decoder.decode(TEST_INPUTS[0])
    assert isinstance(result.converged, bool)
    assert isinstance(result.result, np.ndarray)
    assert len(result.result) == 1


def test_decoder_batch_processing(surface_decoder):
    syndromes = TEST_INPUTS[:5]
    results = surface_decoder.decode_batch(syndromes)
    assert len(results) == len(syndromes)
    assert isinstance(results, qec.BatchDecoderResult)
    assert results.result.shape == (5, 1)
    assert results.converged.shape == (5,) and results.converged.all()
    for i in range(5):
        assert abs(results.result[i, 0] - TEST_OUTPUTS[i][0]) < 1e-4
    # (One sample at a time, as the model's batch is 1: the same numbers as decode's.)
    assert results.result[:, 0].tolist() == [surface_decoder.decode(s).result[0] for s in syndromes]


def test_decoder_with_zero_and_all_ones_syndromes(surface_decoder):
    assert len(surface_decoder.decode([0.0] * NUM_DETECTORS).result) == 1
    assert len(surface_decoder.decode([1.0] * NUM_DETECTORS).result) == 1


def test_float_input_is_raw_and_output_unthresholded(surface_decoder):
    # A float model takes soft values as they are and returns its sigmoid unthresholded.
    soft = surface_decoder.decode([0.3] * NUM_DETECTORS).result[0]
    want = load_onnx(ONNX_MODEL_PATH).run(np.full((1, 24), 0.3, dtype=np.float32))[0, 0]
    assert soft == float(want) and 0.0 < soft < 1.0


def test_cuda_graph_and_precision_options_do_not_change_results():
    # (test_performance_comparison_cuda_graph_vs_traditional and the correctness check: on the
    # CPU there is one way to run the model, so both "executors" agree exactly.)
    H = np.eye(NUM_DETECTORS, dtype=np.uint8)
    with pytest.warns(RuntimeWarning, match=re.escape("Precision 'fp16' is ignored when building strongly-typed networks.")):
        graph = qec.get_decoder("trt_decoder", H, onnx_load_path=ONNX_MODEL_PATH, precision="fp16", use_cuda_graph=True)
    with pytest.warns(RuntimeWarning, match="Precision 'fp16'"):
        traditional = qec.get_decoder("trt_decoder", H, onnx_load_path=ONNX_MODEL_PATH, precision="fp16", use_cuda_graph=False)
    rng = np.random.default_rng(123)
    for _ in range(10):
        s = (rng.random(NUM_DETECTORS) * 2 - 1).astype(np.float32).tolist()
        a, b = graph.decode(s), traditional.decode(s)
        assert a.converged == b.converged is True
        assert a.result.tolist() == b.result.tolist()


# -- Parameter validation (TestTRTDecoderParameterValidation, ValidateParameters_*) ------------------


@pytest.fixture
def H():
    return np.array([[1, 0, 1], [0, 1, 1]], dtype=np.uint8)


def test_validate_parameters_no_paths_provided(H):
    with pytest.raises(RuntimeError) as e:
        qec.get_decoder("trt_decoder", H)
    assert str(e.value) == PREFIX + "TensorRT decoder requires either 'onnx_load_path' or 'engine_load_path' parameter"


@pytest.mark.parametrize("path", ["test_model.onnx", "", "   "])
def test_validate_parameters_both_paths_provided(H, path):
    with pytest.raises(RuntimeError) as e:
        qec.get_decoder("trt_decoder", H, onnx_load_path=path, engine_load_path=path)
    assert str(e.value) == PREFIX + "TensorRT decoder cannot specify both 'onnx_load_path' and 'engine_load_path' parameters. Please choose one."


def test_engine_paths_are_refused_by_name(H):
    # (ConstructionFailureThrows: a decoder that cannot infer must not exist.)
    with pytest.raises(RuntimeError) as e:
        qec.get_decoder("trt_decoder", H, engine_load_path="/no/such/cudaq-qec-test.engine")
    assert str(e.value).startswith(PREFIX + "engine_load_path is not supported")
    with pytest.raises(RuntimeError) as e:
        qec.get_decoder("trt_decoder", np.eye(24, dtype=np.uint8), onnx_load_path=ONNX_MODEL_PATH, engine_save_path="out.engine")
    assert str(e.value).startswith(PREFIX + "engine_save_path is not supported")


def test_load_file_non_existent_file(H, tmp_path):
    missing = str(tmp_path / "non_existent_file.onnx")
    with pytest.raises(RuntimeError) as e:
        qec.get_decoder("trt_decoder", H, onnx_load_path=missing)
    assert str(e.value) == PREFIX + "Failed to open file: " + missing
    long_path = "a" * 1000  # (ValidateParameters_EdgeCases: a long path passes validation.)
    with pytest.raises(RuntimeError) as e:
        qec.get_decoder("trt_decoder", H, onnx_load_path=long_path)
    assert str(e.value) == PREFIX + "Failed to open file: " + long_path
    with pytest.raises(RuntimeError) as e:
        qec.get_decoder("trt_decoder", H, onnx_load_path=str(tmp_path))
    assert str(e.value) == PREFIX + "Failed to open file: " + str(tmp_path)


def test_parameter_types_are_checked(H):
    with pytest.raises(RuntimeError) as e:
        qec.get_decoder("trt_decoder", H, onnx_load_path=5)
    assert str(e.value) == PREFIX + "heterogeneous_map::get() error - Invalid type or key (onnx_load_path)."
    with pytest.raises(RuntimeError) as e:
        qec.get_decoder("trt_decoder", H, onnx_load_path=model_path("identity_dynamic.onnx"), batch_size="2")
    assert str(e.value) == PREFIX + "heterogeneous_map::get() error - Invalid type or key (batch_size)."


def test_precision_warnings(H):
    path = model_path("identity_uint8.onnx")
    H3 = np.eye(3, dtype=np.uint8)
    for precision in ("tf32", "noTF32", "best"):
        ctx = no_warnings()
        try:
            qec.get_decoder("trt_decoder", H3, onnx_load_path=path, precision=precision)
        finally:
            ctx.__exit__(None, None, None)
    for precision in ("fp16", "bf16", "int8", "fp8"):
        with pytest.warns(RuntimeWarning) as rec:
            qec.get_decoder("trt_decoder", H3, onnx_load_path=path, precision=precision)
        assert [str(w.message) for w in rec] == [
            f"Precision '{precision}' is ignored when building strongly-typed networks. TensorRT uses the data types defined in the ONNX model. To use {precision}, export the model with the desired types."
        ]
    with pytest.warns(RuntimeWarning) as rec:
        qec.get_decoder("trt_decoder", H3, onnx_load_path=path, precision="unknown_precision")
    assert [str(w.message) for w in rec] == ["Unknown precision 'unknown_precision', using default (best)"]


# -- Models' I/O (test_trt_decoder.cpp: dynamic batch, uint8, mixed types) --------------------------

CUDA_GRAPH_WARNING = "Model has dynamic shapes or multiple profiles, CUDA graphs not supported. Using traditional execution."


def test_dynamic_batch_identity_model_uses_optimization_profile():
    with pytest.warns(RuntimeWarning) as rec:
        d = qec.get_decoder("trt_decoder", np.eye(3, dtype=np.uint8), onnx_load_path=model_path("identity_dynamic.onnx"), batch_size=2, use_cuda_graph=True, memory_workspace=1 << 20)
    assert [str(w.message) for w in rec] == [CUDA_GRAPH_WARNING]
    syndromes = [[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.25, 0.5, 0.75]]
    results = d.decode_batch(syndromes)
    assert len(results) == 3 and results.converged.all()
    assert results.result.tolist() == syndromes
    # (CUDA graphs are on by default; turned off, nothing is said.)
    with pytest.warns(RuntimeWarning, match=re.escape(CUDA_GRAPH_WARNING)):
        qec.get_decoder("trt_decoder", np.eye(3, dtype=np.uint8), onnx_load_path=model_path("identity_dynamic.onnx"), batch_size=2)
    ctx = no_warnings()
    try:
        qec.get_decoder("trt_decoder", np.eye(3, dtype=np.uint8), onnx_load_path=model_path("identity_dynamic.onnx"), batch_size=2, use_cuda_graph=False)
    finally:
        ctx.__exit__(None, None, None)


def test_dynamic_batch_needs_batch_size():
    path = model_path("identity_dynamic.onnx")
    with pytest.raises(RuntimeError) as e:
        qec.get_decoder("trt_decoder", np.eye(3, dtype=np.uint8), onnx_load_path=path)
    # (TensorRT's build fails without an optimization profile, before CUDA-Q's own check; why is
    # the cause.)
    assert str(e.value) == PREFIX + "Failed to build TensorRT engine"
    assert "batch_size" in str(e.value.__cause__)
    with pytest.raises(RuntimeError) as e:
        qec.get_decoder("trt_decoder", np.eye(3, dtype=np.uint8), onnx_load_path=path, batch_size=0)
    assert str(e.value) == PREFIX + "batch_size must be >= 1, got 0"


def test_dynamic_dimensions_beyond_the_batch(tmp_path):
    data = onnx_model([node("Identity", ["x"], ["y"])], [value_info("x", FLOAT, ["b", "n"])], [value_info("y", FLOAT, ["b", "n"])])
    path = write(tmp_path, "two_dynamic.onnx", data)
    with pytest.raises(RuntimeError) as e:
        qec.get_decoder("trt_decoder", np.eye(3, dtype=np.uint8), onnx_load_path=path, batch_size=2)
    assert str(e.value) == PREFIX + "Failed to set optimization profile dimensions for batch"
    data = onnx_model([node("Identity", ["x"], ["y"])], [value_info("x", FLOAT, [1, "n"])], [value_info("y", FLOAT, [1, "n"])])
    path = write(tmp_path, "inner_dynamic.onnx", data)
    with pytest.raises(RuntimeError) as e:
        qec.get_decoder("trt_decoder", np.eye(3, dtype=np.uint8), onnx_load_path=path, batch_size=2)
    assert str(e.value) == PREFIX + "Failed to build TensorRT engine"


def test_uint8_identity_model_binarizes_input_and_output():
    d = qec.get_decoder("trt_decoder", np.eye(3, dtype=np.uint8), onnx_load_path=model_path("identity_uint8.onnx"), use_cuda_graph=False)
    result = d.decode([0.49, 0.5, 0.75])
    assert result.converged
    assert result.result.tolist() == [0.0, 1.0, 1.0]
    assert d.decode([float("nan"), 7.0, -1.0]).result.tolist() == [0.0, 1.0, 0.0]


def test_mixed_dtype_copies_output():
    d = qec.get_decoder("trt_decoder", np.eye(3, dtype=np.uint8), onnx_load_path=model_path("cast_uint8_float.onnx"), use_cuda_graph=False)
    result = d.decode([0.49, 0.5, 0.75])
    assert result.converged
    assert result.result.tolist() == [0.0, 1.0, 1.0]


def test_unsupported_io_types(tmp_path):
    H3 = np.eye(3, dtype=np.uint8)
    data = onnx_model([node("Identity", ["x"], ["y"])], [value_info("x", INT64, [1, 3])], [value_info("y", INT64, [1, 3])])
    with pytest.raises(RuntimeError) as e:
        qec.get_decoder("trt_decoder", H3, onnx_load_path=write(tmp_path, "i64.onnx", data))
    assert str(e.value) == PREFIX + "Unsupported input tensor data type: int64. Supported types: float32, uint8"
    data = onnx_model([node("Cast", ["x"], ["y"], to=INT32)], [value_info("x", FLOAT, [1, 3])], [value_info("y", INT32, [1, 3])])
    with pytest.raises(RuntimeError) as e:
        qec.get_decoder("trt_decoder", H3, onnx_load_path=write(tmp_path, "to_i32.onnx", data))
    assert str(e.value) == PREFIX + "Unsupported output tensor data type: int32. Supported types: float32, uint8"


# -- The O / global-decoder residual (CompositeGlobalDecoderCombinesLogicalFrame, BatchFailureThrows) --


def test_composite_global_decoder_combines_logical_frame():
    H = np.array([[1], [0]], dtype=np.uint8)
    O = np.array([[1]], dtype=np.uint8)
    d = qec.get_decoder("trt_decoder", H, onnx_load_path=model_path("identity_dynamic.onnx"), batch_size=2, use_cuda_graph=False, global_decoder="single_error_lut", global_decoder_params={}, O=O)
    results = d.decode_batch([[1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    assert len(results) == 2
    assert results.result.tolist() == [[0.0], [1.0]]
    # (The global decoder's convergence: a residual its table cannot explain does not converge.)
    r = d.decode_batch([[1.0, 0.0, 1.0], [0.0, 0.0, 0.0]])
    assert r.converged.tolist() == [False, True] and r.result.tolist() == [[1.0], [0.0]]


def test_batch_failure_throws():
    d = qec.get_decoder("trt_decoder", np.eye(2, dtype=np.uint8), onnx_load_path=model_path("identity_dynamic.onnx"), batch_size=1, use_cuda_graph=False, global_decoder="single_error_lut", global_decoder_params={})
    with pytest.raises(RuntimeError) as e:
        d.decode_batch([[1.0, 0.0, 1.0]])
    assert str(e.value) == "TensorRT batch inference failed: trt_decoder: residual portion of TRT output (3) != global_decoder.syndrome_size (2)"


def test_global_decoder_without_o_returns_its_results():
    # [r0, r1, r2] decoded by the LUT on H (3 x 3): its error frame, as the global decoder gives it.
    d = qec.get_decoder("trt_decoder", np.eye(3, dtype=np.uint8), onnx_load_path=model_path("identity_dynamic.onnx"), batch_size=2, use_cuda_graph=False, global_decoder="single_error_lut", global_decoder_params={})
    r = d.decode_batch([[0.0, 0.7, 0.0], [0.2, 0.0, 0.0], [1.0, 1.0, 0.0]])
    assert r.result.tolist() == [[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    assert r.converged.tolist() == [True, True, False]


def test_global_decoder_needs_its_params_and_a_name():
    # (CUDA-Q attaches a global decoder only when both keys are given and the name is not empty.)
    for extra in ({"global_decoder": "single_error_lut"}, {"global_decoder": "", "global_decoder_params": {}}):
        d = qec.get_decoder("trt_decoder", np.eye(3, dtype=np.uint8), onnx_load_path=model_path("identity_dynamic.onnx"), batch_size=1, use_cuda_graph=False, **extra)
        assert d.decode([0.25, 0.0, 1.0]).result.tolist() == [0.25, 0.0, 1.0]
    with pytest.raises(RuntimeError) as e:
        qec.get_decoder("trt_decoder", np.eye(3, dtype=np.uint8), onnx_load_path=model_path("identity_dynamic.onnx"), batch_size=1, use_cuda_graph=False, global_decoder="nope", global_decoder_params={})
    assert str(e.value) == PREFIX + "invalid decoder requested: nope. Run with CUDAQ_LOG_LEVEL=info (environment variable) to see additional plugin diagnostics at startup."


def test_o_without_global_decoder_returns_pre_l():
    H = np.array([[1, 0], [0, 1], [1, 1]], dtype=np.uint8)
    O = np.array([[1, 1]], dtype=np.uint8)
    d = qec.get_decoder("trt_decoder", H, onnx_load_path=model_path("identity_dynamic.onnx"), batch_size=2, use_cuda_graph=False, O=O)
    r = d.decode_batch([[0.75, 0.0, 1.0], [0.0, 1.0, 1.0], [0.4, 1.0, 1.0]])
    assert r.result.tolist() == [[0.75], [0.0], [0.4000000059604645]]  # (float32, raw: no threshold)
    assert d.decode([1.0, 0.0, 0.0]).result.tolist() == [1.0]


def test_o_is_checked():
    path = model_path("identity_dynamic.onnx")
    kw = dict(onnx_load_path=path, batch_size=1, use_cuda_graph=False)
    with pytest.raises(RuntimeError) as e:
        qec.get_decoder("trt_decoder", np.array([[1], [0]], dtype=np.uint8), O=np.ones((1, 2), dtype=np.uint8), **kw)
    assert str(e.value) == PREFIX + "trt_decoder: O second dimension must equal H block_size (got 2, block_size 1)"
    with pytest.raises(RuntimeError) as e:
        qec.get_decoder("trt_decoder", np.eye(2, dtype=np.uint8), O=np.ones((4, 2), dtype=np.uint8), **kw)
    assert str(e.value) == PREFIX + "trt_decoder: TRT output_size_per_sample (3) is smaller than num_observables (4); model output must be [pre_L, residual_dets]."
    with pytest.raises(RuntimeError) as e:
        qec.get_decoder("trt_decoder", np.eye(3, dtype=np.uint8), O=np.ones((1, 3), dtype=np.uint8), global_decoder="single_error_lut", global_decoder_params={}, **kw)
    assert str(e.value) == PREFIX + "trt_decoder: TRT output_size_per_sample (3) must equal num_observables + global_decoder.syndrome_size (4) for the [pre_L, residual_dets] split."


def test_o_from_a_detector_error_model():
    # (make_pcm_decoder: a Stim model's observables become O when it has any.)
    dem = "error(0.1) D0 L0\nerror(0.1) D0 D1\n"
    d = qec.get_decoder("trt_decoder", dem, onnx_load_path=model_path("identity_dynamic.onnx"), batch_size=1, use_cuda_graph=False, global_decoder="single_error_lut", global_decoder_params={})
    # [pre_L, r0, r1]: the LUT explains r = [1, 0] by error 0 and r = [1, 1] by error 1, and
    # CUDA-Q reads its error frame by observable index: L0 is pre_L XOR error 0.
    assert d.decode([1.0, 1.0, 0.0]).result.tolist() == [0.0]
    assert d.decode([0.0, 1.0, 1.0]).result.tolist() == [0.0]
    assert d.decode([1.0, 1.0, 1.0]).result.tolist() == [1.0]
    without_obs = qec.get_decoder("trt_decoder", "error(0.1) D0\nerror(0.1) D0 D1\n", onnx_load_path=model_path("identity_dynamic.onnx"), batch_size=1, use_cuda_graph=False)
    assert without_obs.decode([0.5, 0.0, 1.0]).result.tolist() == [0.5, 0.0, 1.0]


# -- Batches (TestTRTDecoderBatchValidation) ---------------------------------------------------------

SYNDROME_SIZE = 25


@pytest.fixture(params=[1, 2, 4])
def decoder_with_batch_size(request, tmp_path):
    batch_size = request.param
    path, W = matmul_model(tmp_path, batch_size, SYNDROME_SIZE)
    decoder = qec.get_decoder("trt_decoder", np.eye(SYNDROME_SIZE, dtype=np.uint8), onnx_load_path=path)
    return decoder, batch_size, W


def test_decode_batch_accepts_non_integral_multiple(decoder_with_batch_size):
    decoder, batch_size, W = decoder_with_batch_size
    syndromes = [[float(i + j) for i in range(SYNDROME_SIZE)] for j in range(batch_size + 1)]
    results = decoder.decode_batch(syndromes)
    assert len(results) == len(syndromes)
    assert results.result.shape == (len(syndromes), SYNDROME_SIZE)
    assert results.converged.all()
    np.testing.assert_array_equal(results.result, mlas_matmul(np.asarray(syndromes, dtype=np.float32), W))


def test_decode_batch_syndrome_size_mismatch(decoder_with_batch_size):
    decoder, batch_size, _ = decoder_with_batch_size
    valid, invalid = [0.0] * SYNDROME_SIZE, [0.0] * (SYNDROME_SIZE + 5)
    syndromes = [valid, invalid] + [valid] * (batch_size - 2) if batch_size > 1 else [invalid]
    with pytest.raises(RuntimeError) as e:
        decoder.decode_batch(syndromes)
    assert str(e.value) == f"Syndrome size mismatch at index {1 if batch_size > 1 else 0}: expected 25 but got 30"


def test_decode_single_syndrome_size_mismatch(decoder_with_batch_size):
    decoder, _, _ = decoder_with_batch_size
    with pytest.raises(RuntimeError) as e:
        decoder.decode([0.0] * (SYNDROME_SIZE + 5))
    assert str(e.value) == "Syndrome size mismatch: expected 25 but got 30"


def test_decode_batch_empty_list(decoder_with_batch_size):
    decoder, _, _ = decoder_with_batch_size
    assert len(decoder.decode_batch([])) == 0


def test_decode_batch_large_integral_multiple(decoder_with_batch_size):
    decoder, batch_size, _ = decoder_with_batch_size
    for multiplier in (2, 3):
        count = batch_size * multiplier
        syndromes = np.array([[float(i + j) for i in range(SYNDROME_SIZE)] for j in range(count)])
        results = decoder.decode_batch(syndromes)
        assert len(results) == count
        assert results.result.shape == (count, SYNDROME_SIZE)
        assert results.converged.all()


def test_decode_zero_padding_for_batch_gt_1(decoder_with_batch_size):
    decoder, batch_size, _ = decoder_with_batch_size
    syndrome = [float(i * 0.1) for i in range(SYNDROME_SIZE)]
    result_decode = decoder.decode(syndrome)
    manual_batch = [syndrome] + [[0.0] * SYNDROME_SIZE] * (batch_size - 1)
    result_batch = decoder.decode_batch(manual_batch)
    assert result_decode.converged == result_batch.converged[0]
    assert result_decode.result.tolist() == result_batch.result[0].tolist()


def test_a_model_of_rank_one_is_a_batch_of_single_values():
    # qdq_int8 takes float32[8]: to CUDA-Q a batch of 8 one-value syndromes.
    d = qec.get_decoder("trt_decoder", np.eye(1, dtype=np.uint8), onnx_load_path=model_path("qdq_int8.onnx"))
    assert d.decode([2.6]).result.tolist() == [3.0]
    r = d.decode_batch([[v] for v in [-4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.5]])
    assert r.result[:, 0].tolist() == [-4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 6.0]


# -- Quantized models, as TensorRT imports them -----------------------------------------------------


def test_qdq_models_are_identity_on_small_integers():
    # (test_ai_decoder_quantized_onnx.cpp: Q/DQ with scale 1 is the identity on -4..4.)
    inputs = [[-4.0], [-2.0], [-1.0], [0.0], [1.0], [2.0], [3.0], [4.0]]
    for name in ("qdq_int8.onnx", "qdq_fp8.onnx"):
        d = qec.get_decoder("trt_decoder", np.eye(1, dtype=np.uint8), onnx_load_path=model_path(name))
        assert d.decode_batch(inputs).result.tolist() == inputs


def test_tensorrt_quantizes_to_int8_where_onnx_says_uint8():
    # Without a zero point ONNX quantizes to uint8 (onnxruntime clamps negatives to 0); TensorRT
    # allows uint8 only on DLA and uses int8.
    x = [-4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 300.0, -300.0]
    assert load_onnx(model_path("qdq_uint8.onnx")).run(np.array(x, dtype=np.float32)).tolist() == [0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 255.0, 0.0]
    d = qec.get_decoder("trt_decoder", np.eye(1, dtype=np.uint8), onnx_load_path=model_path("qdq_uint8.onnx"))
    assert d.decode_batch([[v] for v in x]).result[:, 0].tolist() == [-4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 127.0, -128.0]


def test_tensorrt_refuses_asymmetric_quantization(tmp_path):
    # qdq_axis has a uint8 zero point of 128, and TensorRT supports only zero zero points.
    path = model_path("qdq_axis.onnx")
    with pytest.raises(RuntimeError) as e:
        qec.get_decoder("trt_decoder", np.eye(4, dtype=np.uint8), onnx_load_path=path)
    assert str(e.value) == PREFIX + "Failed to parse ONNX model: " + path
    assert "Non-zero zero point is not supported" in str(e.value.__cause__)
    scale = tensor("s", np.array([-1.0], dtype=np.float32), FLOAT)
    zp = tensor("z", np.array([0], dtype=np.int8), INT8)
    data = onnx_model([node("QuantizeLinear", ["x", "s", "z"], ["q"]), node("DequantizeLinear", ["q", "s", "z"], ["y"])], [value_info("x", FLOAT, [1, 4])], [value_info("y", FLOAT, [1, 4])], [scale, zp])
    path = write(tmp_path, "negative_scale.onnx", data)
    with pytest.raises(RuntimeError) as e:
        qec.get_decoder("trt_decoder", np.eye(4, dtype=np.uint8), onnx_load_path=path)
    assert str(e.value) == PREFIX + "Failed to parse ONNX model: " + path
    assert "Scale coefficients must all be positive" in str(e.value.__cause__)


# -- Malformed models (Review Focus 3) -----------------------------------------------------------------


def test_truncated_model(tmp_path):
    with open(ONNX_MODEL_PATH, "rb") as f:
        data = f.read()
    for cut in (1, 7, 100, len(data) // 2, len(data) - 1):
        path = write(tmp_path, f"cut{cut}.onnx", data[:cut])
        with pytest.raises(ValueError, match="truncated"):
            load_onnx(path)
        with pytest.raises(RuntimeError) as e:
            qec.get_decoder("trt_decoder", np.eye(24, dtype=np.uint8), onnx_load_path=path)
        assert str(e.value) == PREFIX + "Failed to parse ONNX model: " + path


def test_unknown_op(tmp_path):
    with open(model_path("identity_uint8.onnx"), "rb") as f:
        data = f.read().replace(b"Identity", b"FooBarOp")
    path = write(tmp_path, "unknown_op.onnx", data)
    with pytest.raises(ValueError) as e:
        load_onnx(path)
    assert str(e.value) == "unsupported ONNX op FooBarOp"
    with pytest.raises(RuntimeError) as e:
        qec.get_decoder("trt_decoder", np.eye(3, dtype=np.uint8), onnx_load_path=path)
    assert str(e.value) == PREFIX + "Failed to parse ONNX model: " + path
    assert str(e.value.__cause__) == "unsupported ONNX op FooBarOp"
    data = onnx_model([node("Tanh", ["x"], ["y"])], [value_info("x", FLOAT, [1, 3])], [value_info("y", FLOAT, [1, 3])])
    with pytest.raises(ValueError, match="^unsupported ONNX op Tanh$"):
        load_onnx(write(tmp_path, "tanh.onnx", data))


def test_two_inputs_or_two_outputs(tmp_path):
    H3 = np.eye(3, dtype=np.uint8)
    data = onnx_model([node("Add", ["a", "b"], ["y"])], [value_info("a", FLOAT, [1, 3]), value_info("b", FLOAT, [1, 3])], [value_info("y", FLOAT, [1, 3])])
    path = write(tmp_path, "two_inputs.onnx", data)
    with pytest.raises(RuntimeError) as e:
        qec.get_decoder("trt_decoder", H3, onnx_load_path=path)
    assert str(e.value) == PREFIX + "TensorRT decoder expects exactly 2 I/O tensors (1 input + 1 output), got 3"
    with pytest.raises(ValueError, match="this model has 2 inputs"):
        load_onnx(path).run(np.zeros((1, 3)))
    data = onnx_model([node("Identity", ["x"], ["y"]), node("Identity", ["x"], ["z"])], [value_info("x", FLOAT, [1, 3])], [value_info("y", FLOAT, [1, 3]), value_info("z", FLOAT, [1, 3])])
    with pytest.raises(RuntimeError) as e:
        qec.get_decoder("trt_decoder", H3, onnx_load_path=write(tmp_path, "two_outputs.onnx", data))
    assert str(e.value) == PREFIX + "TensorRT decoder expects exactly 2 I/O tensors (1 input + 1 output), got 3"
    data = onnx_model([node("Identity", ["c"], ["y"])], [], [value_info("y", FLOAT, [1, 3])], [tensor("c", np.ones((1, 3), dtype=np.float32), FLOAT)])
    with pytest.raises(RuntimeError) as e:
        qec.get_decoder("trt_decoder", H3, onnx_load_path=write(tmp_path, "no_input.onnx", data))
    assert str(e.value) == PREFIX + "Failed to identify input/output tensors"


def test_not_a_model(tmp_path):
    H3 = np.eye(3, dtype=np.uint8)
    pointer = b"version https://git-lfs.github.com/spec/v1\noid sha256:f739c30d81b9fcdf668db6ba4211fc5616c8b462b41fdae07ec2d6e0c069064c\nsize 191673\n"
    for name, data, why in (("pointer.onnx", pointer, "git-lfs pointer"), ("empty.onnx", b"", "no graph"), ("text.onnx", b"hello world", "wire type")):
        path = write(tmp_path, name, data)
        with pytest.raises(ValueError, match=why):
            load_onnx(path)
        with pytest.raises(RuntimeError) as e:
            qec.get_decoder("trt_decoder", H3, onnx_load_path=path)
        assert str(e.value) == PREFIX + "Failed to parse ONNX model: " + path


def test_malformed_graphs(tmp_path):
    cases = {
        "undefined input": onnx_model([node("Relu", ["nope"], ["y"])], [value_info("x", FLOAT, [1, 3])], [value_info("y", FLOAT, [1, 3])]),
        "undefined output": onnx_model([node("Relu", ["x"], ["y"])], [value_info("x", FLOAT, [1, 3])], [value_info("z", FLOAT, [1, 3])]),
        "raw_data": onnx_model([node("Add", ["x", "c"], ["y"])], [value_info("x", FLOAT, [1, 3])], [value_info("y", FLOAT, [1, 3])], [tensor("c", np.ones(3, dtype=np.float32), FLOAT)[:-4]]),
        "Gemm": onnx_model([node("Gemm", ["x"], ["y"])], [value_info("x", FLOAT, [1, 3])], [value_info("y", FLOAT, [1, 3])]),
        "Cast": onnx_model([node("Cast", ["x"], ["y"])], [value_info("x", FLOAT, [1, 3])], [value_info("y", FLOAT, [1, 3])]),
    }
    for why, data in cases.items():
        with pytest.raises(ValueError):
            load_onnx(write(tmp_path, "bad.onnx", data))
    # Shapes that do not fit fail at the dry run, as TensorRT's parser checks them.
    W = tensor("W", np.ones((4, 2), dtype=np.float32), FLOAT)
    data = onnx_model([node("MatMul", ["x", "W"], ["y"])], [value_info("x", FLOAT, [1, 3])], [value_info("y", FLOAT, [1, 2])], [W])
    path = write(tmp_path, "mismatch.onnx", data)
    with pytest.raises(ValueError):
        load_onnx(path).run(np.zeros((1, 3), dtype=np.float32))
    with pytest.raises(RuntimeError) as e:
        qec.get_decoder("trt_decoder", np.eye(3, dtype=np.uint8), onnx_load_path=path)
    assert str(e.value) == PREFIX + "Failed to parse ONNX model: " + path


def test_fuzzed_models_raise_value_errors(tmp_path):
    """Truncations and corrupted bytes of every fixture: load_onnx (and run) raise ValueError or
    nothing, never IndexError, KeyError or struct.error."""
    rng = np.random.default_rng(2026)
    names = sorted(f for f in os.listdir(MODELS) if f.endswith(".onnx") and f != "surface_code_decoder.onnx")
    loaded = 0
    for name in names:
        with open(model_path(name), "rb") as f:
            data = f.read()
        for trial in range(120):
            b = bytearray(data)
            if trial % 3 == 0:
                b = b[: int(rng.integers(0, len(b)))]
            else:
                for _ in range(int(rng.integers(1, 4))):
                    b[int(rng.integers(0, len(b)))] = int(rng.integers(0, 256))
            path = write(tmp_path, "fuzz.onnx", bytes(b))
            try:
                m = load_onnx(path)
            except ValueError:
                continue
            loaded += 1
            for _, _, shape in m.inputs[:1]:
                dims = [d if isinstance(d, int) else 2 for d in shape or ()]
                if np.prod(dims, dtype=object) > 1000:  # (a corrupted dimension: not worth the memory)
                    continue
                try:
                    m.run(np.zeros(dims, dtype=np.float32))
                except ValueError:
                    pass
    assert loaded > 0


# -- The parameter schema and configurations --------------------------------------------------------------------


def test_parameter_schema():
    schema = {e["key"]: e for e in qec.decoder_param_schema("trt_decoder")}
    assert list(schema) == ["onnx_load_path", "engine_load_path", "engine_save_path", "precision", "memory_workspace", "batch_size", "use_cuda_graph", "global_decoder", "global_decoder_params"]
    assert {k: e["kind"] for k, e in schema.items()} == {
        "onnx_load_path": "string", "engine_load_path": "string", "engine_save_path": "string", "precision": "string", "memory_workspace": "uint64",
        "batch_size": "uint64", "use_cuda_graph": "bool", "global_decoder": "string", "global_decoder_params": "discriminated",
    }
    assert schema["global_decoder_params"]["discriminator"] == "global_decoder"
    assert not any(e["required"] for e in schema.values())


def test_trt_decoder_config_converts_through_the_schema():
    with pytest.warns(DeprecationWarning):
        cfg = qec.trt_decoder_config()
    cfg.batch_size = 4
    with pytest.raises(TypeError, match="batch_size expects a non-negative int"):
        cfg.batch_size = -1
    with pytest.raises(TypeError, match="use_cuda_graph expects a bool"):
        cfg.use_cuda_graph = 1
    cfg.global_decoder = "pymatching"
    dc = qec.decoder_config()
    dc.type = "trt_decoder"
    dc.set_decoder_custom_args(cfg)
    assert dc.decoder_custom_args == {"batch_size": 4, "global_decoder": "pymatching", "global_decoder_params": {}}


def test_yaml_round_trip():
    dc = qec.decoder_config()
    dc.id = 0
    dc.type = "trt_decoder"
    dc.block_size = 3
    dc.syndrome_size = 3
    dc.H_sparse = [0, -1, 1, -1, 2, -1]
    dc.decoder_custom_args = {"onnx_load_path": "model.onnx", "batch_size": 2, "precision": "best", "use_cuda_graph": False, "global_decoder": "pymatching", "global_decoder_params": {"merge_strategy": "smallest_weight"}}
    back = qec.decoder_config.from_yaml_str(dc.to_yaml_str())
    assert back.decoder_custom_args == dc.decoder_custom_args
    with pytest.raises(RuntimeError, match=re.escape("Parameter 'batch_size' of 'trt_decoder' expects a non-negative int value.")):
        dc.decoder_custom_args = {"onnx_load_path": "model.onnx", "batch_size": -1}


def test_configure_a_realtime_trt_decoder():
    # An identity "decoder": the detectors are the error frame, and O_sparse reads error 0.
    dc = qec.decoder_config()
    dc.id = 0
    dc.type = "trt_decoder"
    dc.block_size = 3
    dc.syndrome_size = 3
    dc.H_sparse = [0, -1, 1, -1, 2, -1]
    dc.O_sparse = [0, -1]
    dc.D_sparse = [0, -1, 1, -1, 2, -1]
    dc.decoder_custom_args = {"onnx_load_path": model_path("identity_dynamic.onnx"), "batch_size": 1, "use_cuda_graph": False}
    mdc = qec.multi_decoder_config()
    mdc.decoders = [dc]
    try:
        assert qec.configure_decoders(mdc) == 0
        qec.enqueue_syndromes(0, [1, 0, 0])
        assert list(qec.get_corrections(0, 1, False)) == [1]
        qec.enqueue_syndromes(0, [0, 1, 1])
        assert list(qec.get_corrections(0, 1, True)) == [1]
    finally:
        qec.finalize_decoders()

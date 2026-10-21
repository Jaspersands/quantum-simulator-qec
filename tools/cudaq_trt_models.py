"""The ONNX models tests/cudaq_qec_port/test_trt_decoder.py runs, and onnxruntime's outputs on them.

CUDA-Q QEC 0.8.1's (NVIDIA/cudaqx, now NVIDIA/cudaq-qec; Apache License 2.0) trt_decoder tests
build their small models at test time, or keep them in git-lfs. Their generators (Copyright (c)
2026 NVIDIA Corporation & Affiliates) are mirrored here, so the models are those bytes:
`qdq_int8.onnx` and `qdq_fp8.onnx` are
`unittests/realtime/generate_ai_decoder_quantized_onnx.py`'s (their sha256 equal to the LFS
pointers' oids, checked below), and `identity_dynamic.onnx`, `identity_uint8.onnx` and
`cast_uint8_float.onnx` are `unittests/decoders/trt_decoder/generate_trt_decoder_onnx.py`'s
`trt_dynamic_identity`, `trt_uint8_identity` and `trt_uint8_to_float` (IR 9, opset 19). The others
are this package's, so that every operator the evaluator (`stabilizer_qec.cudaq_qec._onnx`) runs is
checked against onnxruntime: `ops_chain` (Flatten, MatMul, Add, Relu, Reshape with 0 and -1, Gemm with transB,
alpha and beta, Sigmoid, Identity), `gemm_trans` (Gemm with transA and a broadcast C), `qdq_axis`
(per-axis int8 and per-tensor uint8 with a zero point), `qdq_uint8` (no zero point: ONNX's uint8),
`qdq_fp8_zp` (float8e4m3fn quantized with a nonzero zero point, which onnxruntime ignores),
`cast_fp8_sat` / `cast_fp8_nosat` (float8e4m3fn by Cast, saturating and not), and the products
of K = 600, past MLAS's blocks of 256: `matmul_k600` (by a matrix), `matmul_vector_k600` (by a
vector, one block), `gemm_k600` (transB, alpha, beta and C) and `gemm_alpha_k600` (alpha, no C).
The inputs of the models added since the first recording draw on their own generators, so the
earlier models' recorded inputs stay as they were.

`surface_code_decoder.onnx` is CUDA-Q's own (`assets/tests/surface_code_decoder.onnx`, a git-lfs
file, 191,673 bytes), fetched read-only from
https://media.githubusercontent.com/media/NVIDIA/cudaq-qec/0.8.1/assets/tests/surface_code_decoder.onnx
and checked here against its LFS oid; this script does not download it.

    python tools/cudaq_trt_models.py   # with onnx 1.23.2 and onnxruntime 1.31.0

Writes the models to tests/cudaq_qec_port/models/ and onnxruntime's outputs on fixed inputs to
tests/cudaq_qec_port/models/ort_outputs.npz (keys `<model>__<case>__x` and `<model>__<case>__y`).
"""

from __future__ import annotations

import hashlib
import pathlib

import ml_dtypes  # (onnx's own dependency)
import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "tests" / "cudaq_qec_port" / "models"

# git-lfs oids and sizes of CUDA-Q 0.8.1's pointers.
LFS = {
    "qdq_int8.onnx": ("6b8493417ef19183262edff7254b542bf59a108e7574c39a0730f238634d5a9a", 217),
    "qdq_fp8.onnx": ("250f08fa38dbebdcc71b7cace77f300362e22467e219cc2c6e9de48f420739b8", 216),
    "surface_code_decoder.onnx": ("f739c30d81b9fcdf668db6ba4211fc5616c8b462b41fdae07ec2d6e0c069064c", 191673),
}


def save(model, name, check=True):
    if check:
        onnx.checker.check_model(model)
    onnx.save(model, str(OUT / name))


# -- CUDA-Q's generators ------------------------------------------------------------------------


def qdq(name, graph_name, zero_point):
    i = helper.make_tensor_value_info("input", TensorProto.FLOAT, [8])
    o = helper.make_tensor_value_info("output", TensorProto.FLOAT, [8])
    scale = numpy_helper.from_array(np.array([1.0], dtype=np.float32), "scale")
    q = helper.make_node("QuantizeLinear", ["input", "scale", "zero_point"], ["quantized"])
    d = helper.make_node("DequantizeLinear", ["quantized", "scale", "zero_point"], ["output"])
    g = helper.make_graph([q, d], graph_name, [i], [o], [scale, zero_point])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 19)])
    m.ir_version = 10
    save(m, name)


def identity(name, graph_name, elem, shape):
    i = helper.make_tensor_value_info("input", elem, shape)
    o = helper.make_tensor_value_info("output", elem, shape)
    g = helper.make_graph([helper.make_node("Identity", ["input"], ["output"])], graph_name, [i], [o])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 19)])
    m.ir_version = 9
    save(m, name)


def cast(name, graph_name, in_elem, out_elem, shape):
    i = helper.make_tensor_value_info("input", in_elem, shape)
    o = helper.make_tensor_value_info("output", out_elem, shape)
    g = helper.make_graph([helper.make_node("Cast", ["input"], ["output"], to=out_elem)], graph_name, [i], [o])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 19)])
    m.ir_version = 9
    save(m, name)


# -- This package's, for the rest of the operators ------------------------------------------------


def model(name, nodes, inputs, outputs, inits, opset=19):
    g = helper.make_graph(nodes, name.split(".")[0], inputs, outputs, inits)
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", opset)], producer_name="stabilizer_qec")
    m.ir_version = 9
    save(m, name)


def ops_chain():
    rng = np.random.default_rng(11)
    W1 = numpy_helper.from_array((rng.standard_normal((4, 5)) * 0.7).astype(np.float32), "W1")
    b1 = numpy_helper.from_array((rng.standard_normal(5) * 0.3).astype(np.float32), "b1")
    shape = numpy_helper.from_array(np.array([2, 0, -1], dtype=np.int64), "shape")
    W2 = numpy_helper.from_array((rng.standard_normal((3, 15)) * 0.4).astype(np.float32), "W2")
    b2 = numpy_helper.from_array((rng.standard_normal(3) * 0.2).astype(np.float32), "b2")
    nodes = [
        helper.make_node("Flatten", ["input"], ["flat"], axis=2),  # [2, 3, 4] -> [6, 4]
        helper.make_node("MatMul", ["flat", "W1"], ["mm"]),  # [6, 5]
        helper.make_node("Add", ["mm", "b1"], ["biased"]),
        helper.make_node("Relu", ["biased"], ["relu"]),
        helper.make_node("Reshape", ["relu", "shape"], ["cube"]),  # [2, 0, -1] -> [2, 5, 3] (0 copies)
        helper.make_node("Flatten", ["cube"], ["rows"]),  # [2, 15]
        helper.make_node("Gemm", ["rows", "W2", "b2"], ["gemm"], alpha=0.5, beta=2.0, transB=1),  # [2, 3]
        helper.make_node("Sigmoid", ["gemm"], ["sig"]),
        helper.make_node("Identity", ["sig"], ["output"]),
    ]
    i = helper.make_tensor_value_info("input", TensorProto.FLOAT, [2, 3, 4])
    o = helper.make_tensor_value_info("output", TensorProto.FLOAT, [2, 3])
    model("ops_chain.onnx", nodes, [i], [o], [W1, b1, shape, W2, b2])


def gemm_trans():
    rng = np.random.default_rng(12)
    B = numpy_helper.from_array(rng.standard_normal((3, 4)).astype(np.float32), "B")
    C = numpy_helper.from_array(rng.standard_normal((2, 1)).astype(np.float32), "C")
    nodes = [helper.make_node("Gemm", ["input", "B", "C"], ["output"], alpha=1.5, beta=-1.0, transA=1)]
    i = helper.make_tensor_value_info("input", TensorProto.FLOAT, [3, 2])
    o = helper.make_tensor_value_info("output", TensorProto.FLOAT, [2, 4])
    model("gemm_trans.onnx", nodes, [i], [o], [B, C])


def qdq_axis():
    s1 = numpy_helper.from_array(np.array([0.5, 1.0, 2.0, 0.25], dtype=np.float32), "s1")
    z1 = numpy_helper.from_array(np.zeros(4, dtype=np.int8), "z1")
    s2 = numpy_helper.from_array(np.array(0.1, dtype=np.float32), "s2")
    z2 = numpy_helper.from_array(np.array(128, dtype=np.uint8), "z2")
    nodes = [
        helper.make_node("QuantizeLinear", ["input", "s1", "z1"], ["q1"], axis=1),
        helper.make_node("DequantizeLinear", ["q1", "s1", "z1"], ["d1"], axis=1),
        helper.make_node("QuantizeLinear", ["d1", "s2", "z2"], ["q2"]),
        helper.make_node("DequantizeLinear", ["q2", "s2", "z2"], ["output"]),
    ]
    i = helper.make_tensor_value_info("input", TensorProto.FLOAT, [2, 4])
    o = helper.make_tensor_value_info("output", TensorProto.FLOAT, [2, 4])
    model("qdq_axis.onnx", nodes, [i], [o], [s1, z1, s2, z2])


def qdq_uint8():
    scale = numpy_helper.from_array(np.array([1.0], dtype=np.float32), "scale")
    nodes = [helper.make_node("QuantizeLinear", ["input", "scale"], ["quantized"]), helper.make_node("DequantizeLinear", ["quantized", "scale"], ["output"])]
    i = helper.make_tensor_value_info("input", TensorProto.FLOAT, [8])
    o = helper.make_tensor_value_info("output", TensorProto.FLOAT, [8])
    model("qdq_uint8.onnx", nodes, [i], [o], [scale])


def cast_fp8(name, saturate):
    nodes = [helper.make_node("Cast", ["input"], ["f8"], to=TensorProto.FLOAT8E4M3FN, saturate=saturate), helper.make_node("Cast", ["f8"], ["output"], to=TensorProto.FLOAT)]
    i = helper.make_tensor_value_info("input", TensorProto.FLOAT, ["n"])
    o = helper.make_tensor_value_info("output", TensorProto.FLOAT, ["n"])
    model(name, nodes, [i], [o], [])


def qdq_fp8_zp():
    # qdq_fp8 with a float8 zero point of 1.0 where onnxruntime takes a nonzero one, in
    # QuantizeLinear (which ignores it); DequantizeLinear's is 0, as onnxruntime refuses any
    # other for float8.
    scale = numpy_helper.from_array(np.array([1.0], dtype=np.float32), "scale")
    zq = helper.make_tensor("zero_point_q", TensorProto.FLOAT8E4M3FN, [1], vals=[1.0])
    zd = helper.make_tensor("zero_point_dq", TensorProto.FLOAT8E4M3FN, [1], vals=[0])
    nodes = [helper.make_node("QuantizeLinear", ["input", "scale", "zero_point_q"], ["quantized"]), helper.make_node("DequantizeLinear", ["quantized", "scale", "zero_point_dq"], ["output"])]
    i = helper.make_tensor_value_info("input", TensorProto.FLOAT, [8])
    o = helper.make_tensor_value_info("output", TensorProto.FLOAT, [8])
    model("qdq_fp8_zp.onnx", nodes, [i], [o], [scale, zq, zd])


def products_k600():
    # K = 600: MLAS sums 256 + 256 + 88, each block folded into the output; by a vector, one block.
    rng = np.random.default_rng(14)
    i = helper.make_tensor_value_info("input", TensorProto.FLOAT, [2, 600])
    W = numpy_helper.from_array(rng.standard_normal((600, 4)).astype(np.float32), "W")
    model("matmul_k600.onnx", [helper.make_node("MatMul", ["input", "W"], ["output"])], [i], [helper.make_tensor_value_info("output", TensorProto.FLOAT, [2, 4])], [W])
    v = numpy_helper.from_array(rng.standard_normal(600).astype(np.float32), "v")
    model("matmul_vector_k600.onnx", [helper.make_node("MatMul", ["input", "v"], ["output"])], [i], [helper.make_tensor_value_info("output", TensorProto.FLOAT, [2])], [v])
    B = numpy_helper.from_array(rng.standard_normal((4, 600)).astype(np.float32), "B")
    C = numpy_helper.from_array(rng.standard_normal(4).astype(np.float32), "C")
    gemm = helper.make_node("Gemm", ["input", "B", "C"], ["output"], alpha=1.5, beta=0.5, transB=1)
    model("gemm_k600.onnx", [gemm], [i], [helper.make_tensor_value_info("output", TensorProto.FLOAT, [2, 4])], [B, C])
    B2 = numpy_helper.from_array(rng.standard_normal((600, 4)).astype(np.float32), "B")
    gemm = helper.make_node("Gemm", ["input", "B"], ["output"], alpha=1.5)
    model("gemm_alpha_k600.onnx", [gemm], [i], [helper.make_tensor_value_info("output", TensorProto.FLOAT, [2, 4])], [B2])


# -- Inputs ----------------------------------------------------------------------------------------


def surface_inputs():
    """CUDA-Q's 30 test syndromes (test_trt_decoder.py's TEST_INPUTS: none, each one detector,
    and five pairs), then random hard and soft ones."""
    rows = [np.zeros(24)]
    for k in range(24):
        r = np.zeros(24)
        r[k] = 1
        rows.append(r)
    for a, b in ((0, 5), (3, 11), (7, 19), (12, 20), (2, 23)):
        r = np.zeros(24)
        r[[a, b]] = 1
        rows.append(r)
    rng = np.random.default_rng(3)
    rows += list(rng.integers(0, 2, size=(20, 24)))
    rows += list(rng.random((10, 24)))
    rows.append(np.ones(24))
    return [np.asarray(r, dtype=np.float32).reshape(1, 24) for r in rows]


def fp8_sweep():
    """Every float8e4m3fn value, the midpoints between neighbours and a hair either side, the
    range's ends, random magnitudes, and the specials, both signs."""
    vals = np.arange(0x7F, dtype=np.uint8).view(ml_dtypes.float8_e4m3fn).astype(np.float32)
    mids = (vals[:-1].astype(np.float64) + vals[1:]) / 2
    xs = [vals, mids, np.nextafter(mids, 0), np.nextafter(mids, np.inf)]
    xs.append(np.array([448, 450, 463.9, 464, 464.1, 470, 479, 480, 481, 495, 496, 500, 511, 512, 1e6, 3.4e38, 2.0**-10, 2.0**-11, 1e-30, 1e-45], dtype=np.float64))
    rng = np.random.default_rng(8)
    xs.append(np.exp2(rng.uniform(-14, 10, 1000)))
    x = np.concatenate([np.asarray(a, dtype=np.float32) for a in xs])
    x = np.concatenate([x, -x, np.array([0.0, -0.0, np.inf, -np.inf, np.nan], dtype=np.float32)])
    return [x]


QDQ_CASES = [
    [-4, -2, -1, 0, 1, 2, 3, 4],
    [0.5, 1.5, 2.5, -0.5, -1.5, -2.5, 126.5, -127.5],
    [127.4, 127.5, 128, -128.4, -128.5, -129, 300, -300],
    [254.5, 255.5, 256, 1e30, -1e30, 1e-30, -1e-30, 0.0],
    [0.001, 0.0019, 500.0, -1000.0, 13.0, 0.3, 449.0, 1e-9],
]


def inputs_for(name, rng):
    # (Models added after the first recording draw on their own generators, not on rng.)
    if name == "qdq_fp8_zp.onnx":
        own = np.random.default_rng(13)
        cases = QDQ_CASES + list(own.standard_normal((8, 8)) * 60) + [[np.inf, -np.inf, np.nan, 464, 464.5, 480, 496, -0.0]]
        return [np.asarray(c, dtype=np.float32) for c in cases]
    if name in ("matmul_k600.onnx", "matmul_vector_k600.onnx", "gemm_k600.onnx", "gemm_alpha_k600.onnx"):
        own = np.random.default_rng(15)
        return [(own.standard_normal((2, 600)) * s).astype(np.float32) for s in (0.1, 1.0, 10.0)]
    if name == "surface_code_decoder.onnx":
        return surface_inputs()
    if name in ("qdq_int8.onnx", "qdq_fp8.onnx", "qdq_uint8.onnx"):
        cases = list(QDQ_CASES)
        cases += list(rng.standard_normal((8, 8)) * 60)
        if name == "qdq_fp8.onnx":
            cases.append([np.inf, -np.inf, np.nan, 464, 464.5, 480, 496, -0.0])
        if name == "qdq_int8.onnx":
            cases.append([np.inf, -np.inf, np.nan, -np.nan, 127.5, -128.5, 0.49999997, -0.0])
        return [np.asarray(c, dtype=np.float32) for c in cases]
    if name == "identity_dynamic.onnx":
        return [rng.standard_normal((b, 3)).astype(np.float32) for b in (1, 2, 3, 5)] + [np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.25, 0.5, 0.75]], dtype=np.float32)]
    if name in ("identity_uint8.onnx", "cast_uint8_float.onnx"):
        return [np.array([[0, 1, 255]], dtype=np.uint8)] + [rng.integers(0, 256, size=(1, 3), dtype=np.uint8) for _ in range(4)]
    if name == "ops_chain.onnx":
        return [(rng.standard_normal((2, 3, 4)) * s).astype(np.float32) for s in (0.1, 1.0, 3.0, 10.0)]
    if name == "gemm_trans.onnx":
        return [(rng.standard_normal((3, 2)) * s).astype(np.float32) for s in (0.5, 1.0, 4.0)]
    if name == "qdq_axis.onnx":
        return [(rng.standard_normal((2, 4)) * s).astype(np.float32) for s in (0.3, 3.0, 30.0, 300.0)] + [np.array([[0.25, 0.5, 1.0, 0.125], [-0.25, -0.75, -3.0, 0.0625]], dtype=np.float32)]
    if name in ("cast_fp8_sat.onnx", "cast_fp8_nosat.onnx"):
        return fp8_sweep()
    raise KeyError(name)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    qdq("qdq_int8.onnx", "qdq_int8", numpy_helper.from_array(np.array([0], dtype=np.int8), "zero_point"))
    qdq("qdq_fp8.onnx", "qdq_fp8", helper.make_tensor("zero_point", TensorProto.FLOAT8E4M3FN, [1], vals=[0]))
    identity("identity_dynamic.onnx", "trt_dynamic_identity", TensorProto.FLOAT, ["batch", 3])
    identity("identity_uint8.onnx", "trt_uint8_identity", TensorProto.UINT8, [1, 3])
    cast("cast_uint8_float.onnx", "trt_uint8_to_float", TensorProto.UINT8, TensorProto.FLOAT, [1, 3])
    ops_chain()
    gemm_trans()
    qdq_axis()
    qdq_uint8()
    cast_fp8("cast_fp8_sat.onnx", 1)
    cast_fp8("cast_fp8_nosat.onnx", 0)
    qdq_fp8_zp()
    products_k600()

    print("onnx", onnx.__version__, "onnxruntime", ort.__version__)
    names = sorted(p.name for p in OUT.glob("*.onnx"))
    for name in names:
        data = (OUT / name).read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        tag = ""
        if name in LFS:
            if (digest, len(data)) != LFS[name]:
                raise SystemExit(f"{name}: {len(data)} B sha256 {digest} is not the LFS object {LFS[name]}")
            tag = "  (the LFS object)"
        print(f"{name}: {len(data)} B sha256 {digest[:16]}{tag}")

    rng = np.random.default_rng(1)
    record = {"meta__onnxruntime": np.array(ort.__version__), "meta__onnx": np.array(onnx.__version__)}
    for name in names:
        sess = ort.InferenceSession(str(OUT / name), providers=["CPUExecutionProvider"])
        feed = sess.get_inputs()[0].name
        stem = name[: -len(".onnx")]
        for k, x in enumerate(inputs_for(name, rng)):
            y = sess.run(None, {feed: x})[0]
            record[f"{stem}__{k}__x"] = x
            record[f"{stem}__{k}__y"] = y
    np.savez_compressed(OUT / "ort_outputs.npz", **record)
    print(f"ort_outputs.npz: {sum(1 for k in record if k.endswith('__y'))} cases")


if __name__ == "__main__":
    main()

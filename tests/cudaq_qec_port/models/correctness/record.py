"""Record the package's constructed regression data, separately from the port.

Run with ONNX 1.20.1, ORT 1.31.0, NumPy 2.5.4 (CPUExecutionProvider, macOS arm64):
    /path/to/reference/python -I /absolute/path/to/record.py

No port imports. These models/data are stabilizer_qec regression constructions, not NVIDIA
models. ONNX protobuf/schema definitions: ONNX Project Contributors, Apache-2.0; ORT output
reference: Microsoft Corporation, MIT. Existing NVIDIA fixtures/notices are unchanged.
"""
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper as h, numpy_helper as nh

OUT = Path(__file__).resolve().parent


def model(nodes, inits, shape, outshape):
    m = h.make_model(h.make_graph(nodes, 'correctness', [h.make_tensor_value_info('x', 1, shape)],
        [h.make_tensor_value_info('y', 1, outshape)], inits), opset_imports=[h.make_opsetid('', 19)])
    m.ir_version = 9
    return m.SerializeToString()


def run(data, x, optimization=ort.GraphOptimizationLevel.ORT_ENABLE_ALL):
    options = ort.SessionOptions()
    options.graph_optimization_level = optimization
    return ort.InferenceSession(data, options, providers=['CPUExecutionProvider']).run(None, {'x': x})[0]


def main():
    assert onnx.__version__ == '1.20.1' and ort.__version__ == '1.31.0'
    rng = np.random.default_rng(182)
    a = rng.normal(size=(3, 600)).astype(np.float32)
    rng.normal(size=(600, 20))  # Original constant/computed-B control's draw, retained for provenance.
    w = rng.integers(-127, 127, size=(600, 20), dtype=np.int8)
    x = np.zeros_like(a)
    x[0] = a[0] > 0
    scale, zero = np.array(.1, np.float32), np.array(0, np.int8)
    # Bias was constructed by centering the original differing port/ORT output 5 around 0.5.
    # Preserve its exact float32 bits; do not recenter against changed implementations.
    bias = np.zeros(20, np.uint32)
    bias[5] = 3265658913
    bias = bias.view(np.float32)
    data = model([h.make_node('DequantizeLinear', ['w', 's', 'z'], ['b']),
                  h.make_node('MatMul', ['x', 'b'], ['mm']), h.make_node('Add', ['mm', 'bias'], ['y'])],
                 [nh.from_array(v, k) for k, v in [('w', w), ('s', scale), ('z', zero), ('bias', bias)]], [3, 600], [3, 20])
    np.savez_compressed(OUT / 'weightdq_threshold.npz', w=w, s=scale, z=zero, bias=bias, x=x,
        meta_ort=np.array(ort.__version__), meta_onnx=np.array(onnx.__version__),
        ort_all=run(data, x), ort_disabled=run(data, x, ort.GraphOptimizationLevel.ORT_DISABLE_ALL))
    bits = np.array([0x7f800001, 0xff800001, 0x7fa00000, 0xffa00000, 0x7fc00000, 0xffc00000,
                     0x7fffffff, 0xffffffff, 0x7f800000, 0xff800000, 0x43fa0000, 0xc3fa0000, 0x3f800000], np.uint32)
    recorded = dict(bits=bits, meta_ort=np.array(ort.__version__), meta_onnx=np.array(onnx.__version__))
    for sat in (0, 1):
        data = model([h.make_node('Cast', ['x'], ['fp8'], to=17, saturate=sat), h.make_node('Cast', ['fp8'], ['y'], to=1)], [], [len(bits)], [len(bits)])
        recorded['y' + str(sat)] = run(data, bits.view(np.float32))
    np.savez_compressed(OUT / 'fp8_nan_controls.npz', **recorded)
    print('Recorded ORT', ort.__version__, 'ONNX', onnx.__version__)


if __name__ == '__main__':
    main()

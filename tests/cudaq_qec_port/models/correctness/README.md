# Constructed correctness data

These small `.npz` files contain this package's constructed regression inputs and independent
ONNX Runtime recordings. They are not CUDA-Q model assets. Existing CUDA-Q model bytes and
Apache-2.0 notices in the parent directory are unchanged.

`record.py` regenerates both archives without importing stabilizer_qec. Reference environment:
Python 3.13, NumPy 2.5.4, ONNX 1.20.1, ONNX Runtime 1.31.0, CPUExecutionProvider on macOS arm64.
ONNX schema/protobuf definitions are from the ONNX Project (Apache-2.0); ORT reference kernels
are Microsoft Corporation's (MIT); see the repository's third-party notices.

`weightdq_threshold.npz` contains int8 weights, float32 scale/bias, int8 zero point, a binary
float32 syndrome batch, and outputs from both default/all and disabled ORT optimization.
Seed 182 and the exact bias bits are recorded in the generator. This constructed bias exposes
optimizer-sensitive rounding at output 5 near 0.5; it is not a real-model failure statistic.

`fp8_nan_controls.npz` contains raw float32 payload bits (signaling/quiet NaNs, both signs),
infinities, finite overflow and 1.0, with Cast-to-E4M3FN-and-back results for both saturation
settings. Tests need only NumPy, not ONNX or ORT.

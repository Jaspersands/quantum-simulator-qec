# CPU ONNX decoder contract

`trt_decoder` evaluates a small ONNX tensor-graph subset in NumPy. It does not execute a
TensorRT engine. Engine binary load/save and GPU execution are unsupported.

## Versions and schemas

The reader accepts ONNX IR **7 through 10** and one default-domain opset import, version
**13 through 21**. The empty domain and `ai.onnx` spelling are treated as aliases. Other
import or node domains, duplicate imports, local functions and training graphs are refused.
These bounds cover the existing recorded fixtures (IR 8–10; opsets 17 and 19); newer or older
models need an explicit compatibility review. Accepting a version means support for the
subset below, not every operator or feature in that ONNX version.

Only Gemm, MatMul, Add, Relu, Sigmoid, Identity, Cast, Reshape, Flatten, QuantizeLinear and
DequantizeLinear are implemented. Node input/output counts, known attributes, attribute kinds,
duplicate attributes, and schema tensor types are checked on load. Schema-valid integer Gemm
may be loaded for importer inspection, but `run` refuses it before evaluating any node; this
preserves the public TRT importer's more specific integer-input diagnostic. Shape-dependent
checks run when concrete shapes are available. Reshape `allowzero` requires opset 14;
Cast/QuantizeLinear `saturate` requires 19; QuantizeLinear `output_dtype` and Q/DQ `block_size`
require 21. Block size must remain zero: blocked quantization is unsupported. Boolean flags
`allowzero`, `saturate`, `transA`, and `transB` must be 0 or 1. Unknown attributes are refused.

The supported tensor types are float16, float32, float64, bool, signed/unsigned 8/16/32/64-bit
integers and float8e4m3fn, further restricted by operator:

| Operator | Implemented types |
| --- | --- |
| Gemm, Sigmoid | float16/32/64 |
| MatMul | float16/32/64, int32/64, uint32/64 |
| Add | MatMul types; int8/16 and uint8/16 from opset 14 |
| Relu | float16/32/64; int8/16/32/64 from opset 14 |
| Identity, Reshape, Cast | Supported tensor types; float8 requires opset 19 |
| Flatten | Supported tensor types; float8 requires opset 21 |
| QuantizeLinear | float32 input/scale; float16 from opset 19; int8/uint8 output, float8 from 19 |
| DequantizeLinear | int8/uint8/int32 input, float32 scale/output; float8 input or float16 scale/output from 19 |

Float8 tensors also require IR >= 9. Reshape shape inputs are int64. Arithmetic inputs have
matching types; Q/DQ zero points have the quantized type and the scale's shape. Per-tensor and
per-axis Q/DQ are supported. Missing QuantizeLinear zero points default to uint8 unless
`output_dtype` supplies a supported type. String, complex, bfloat16, other float8 formats,
4-bit types, sparse/external/segmented tensors, subgraphs, sequences and optional values are
unsupported. This is a targeted validator, not exhaustive ONNX schema conformance or shape
inference. Internal ValueInfo metadata and all possible malformed protobuf combinations have
not been exhaustively validated.

## Arithmetic and reference scope

Float32 Gemm/MatMul use fused multiply-adds, each rounded once to float32 as MLAS's FMA rounds
it: the float32 product is exact in float64, and the float64 sum is made round-to-odd before its
conversion, so that the conversion rounds the exact value (a plain float64 sum rounded again to
float32 can differ by one ulp at float32 ties). 256-wide K blocks are accumulated from zero then
folded into the output. Vector B uses one K block. Gemm starts with typed beta*C when beta is
nonzero; beta zero does not read C. Empty broadcast outputs validate dimensions and return
without iterating K. Float32 Sigmoid uses the recorded MLAS rational approximation, its
multiply-adds fused the same way. Other arithmetic uses NumPy with the graph's types. Float16
and float64 paths do not claim ORT kernel equality; platform NumPy/BLAS implementations can also
affect their rounding. Cast to bounded integer types follows NumPy conversion, which can differ
from ORT for out-of-range values. Int32 DQ subtracts the zero point in int64 before float32
scaling, avoiding signed int32 overflow.

ORT's default optimizer rewrites a float MatMul followed by an Add into a Gemm with the Add's
other operand as C (MatMulAddFusion, at its basic level), and MLAS then folds every K block into
that bias instead of adding it to the finished product; past K = 256 the two round differently.
The evaluator fuses where ORT 1.31.0 does, as probed from ORT's optimized graphs: the MatMul's
output is no graph output and its only consumer is the Add (Identity nodes and type-preserving
Casts between are removed first); B is of rank 2; A of rank 2 takes a bias of shape [N], [1, N],
[M, 1] or [M, N], and A of another rank, every dimension a number, a bias [N]; dimensions
compare as ONNX shapes do, symbolic names by name; of two MatMuls into one Add, the first in
ORT's topological order fuses. Shapes are inferred from the graph's inputs and initializers
through the supported operators; graph `value_info` is not read, so where only it would give
ORT a shape, the port does not fuse. Other ORT graph rewrites are not emulated.

Exact ORT agreement is tested for the existing fixed recordings in `models/ort_outputs.npz`
(ORT 1.31.0 CPU on macOS arm64), among them MatMul + Add past one K block, fused and not, and
products whose twice-rounded sums would differ; the CUDA-Q surface model's 30 recorded PyTorch
outputs are checked to absolute tolerance 1e-4. Neither establishes arbitrary graph/CPU
equality. Computed B and weight-only DequantizeLinear feeding MatMul/Gemm can select different
ORT prepacking, blocking and quantized-operator fusion even on the same CPU. The port keeps the
typed mathematical operations and its fixed float32 order rather than imitating those choices.

New computed-B and weight-DQ tests compare against float64 sums of the graph's typed inputs
and float32-dequantized weights, using absolute tolerance 1e-4 for those fixtures only. The
saved binary-syndrome threshold witness gives port output **0.49976348876953125**, ORT 1.31.0
with default/all optimizations **0.500244140625**, and ORT with optimizations disabled
**0.49971771240234375**. Thus the default ORT path crosses the 0.5 correction threshold.
This constructed fixture demonstrates reachable decision sensitivity, not a real-model failure
rate or native TensorRT GPU disagreement. No general absolute/ULP error bound is asserted.

## Python matrix coercions

For `get_decoder(..., O=...)`, the port intentionally retains useful broader inputs than
CUDA-Q 0.8.1's uint8 row-major tensor binding: two-dimensional float32/64, int32/64 and bool
arrays, nested rectangular lists, Fortran-order and strided arrays, and scipy sparse matrices.
Nonzero entries define observable support. Other NumPy numeric dtypes follow the same
nonzero rule; this conversion does not round floats to uint8. Column count must match H and
rank must be two. Existing exceptional forms such as None, scalars or rank-three arrays still
fail. CUDA-Q's kwargs binding instead rejects various dtype/order/container forms before
reading O, so its exact binding errors are not promised for these accepted forms.

CSS matrix setters also retain dense array-like coercion with NumPy uint8 conversion (including
nested lists); upstream's dense caster calls the object's `astype` method. Sparse setters
validate CSR indices and offsets before calling SciPy normalization, preserve the caller's
matrix, drop explicit zeros and sum duplicates as before. Negative indices raise RuntimeError,
out-of-range columns and malformed CSR structure raise ValueError with reference-style
messages. Validation before normalization intentionally avoids unsafe compiled indexing of
malformed CSR storage.

## Resource limits

The empty-product regression runs public decoder construction with K=2**39, zero stored
matrix elements, a 10-second own-process deadline and a 2-GiB RSS watchdog. Related supported
operators were inspected: only the shared float32 Gemm/MatMul product loop iterated over K
independently of output volume. Other kernels use vectorized NumPy operations or loops over
rank/scalars. Existing tensor/input/output volume bounds remain in place; this is not a
universal graph work/allocation budget. Large nonempty intermediate broadcast tensors and
large nonempty contracting products can still consume substantial memory or time.

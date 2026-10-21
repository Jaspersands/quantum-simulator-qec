# Ported from CUDA-Q QEC (NVIDIA/cudaqx 0.8.1, now NVIDIA/cudaq-qec; libs/qec:
# lib/decoders/plugins/trt_decoder/trt_decoder.cpp, Copyright (c) 2025 - 2026 NVIDIA Corporation &
# Affiliates, and include/cudaq/qec/trt_decoder_internal.h, Copyright (c) 2024 - 2025 NVIDIA
# Corporation & Affiliates; Apache License 2.0; see THIRD_PARTY_NOTICES.md), modified for
# stabilizer_qec: the ONNX model runs on the CPU in numpy
# (._onnx) where CUDA-Q builds a TensorRT engine; how TensorRT's ONNX parser imports quantization
# follows onnx-tensorrt (NVIDIA, Apache License 2.0).
"""CUDA-Q QEC's ``trt_decoder``: a neural-network decoder given as an ONNX model
(``onnx_load_path``), run on the CPU, with what CUDA-Q's does around the network: a batch of
syndromes run ``batch_size`` (or the model's own batch) at a time, a single syndrome zero-padded to
a batch, uint8 inputs thresholded at 0.5 and uint8 outputs read as 0/1, and optionally a global
decoder on the network's residual detectors (``global_decoder``, ``global_decoder_params``), its
prediction combined with the network's own observable prefix when the model has observables
(``O``).

CUDA-Q builds a TensorRT engine from the model, strongly typed: the model's own types are the
types computed in, so ``precision`` (other than ``best``, ``tf32`` and ``noTF32``) only warns, as
there. TensorRT's quantization is what runs: QuantizeLinear to int8 where ONNX says uint8 (or
names no zero point), a float8 zero point ignored, and a nonzero integer zero point or a scale
that is not positive refused, as are DequantizeLinear of uint8 and MatMul or Gemm of int32 or
int64. An engine's input and output have at most 2**31 - 1 elements, which CUDA-Q counts in ints
(a static shape's dimension by dimension, so no partial count may pass it either).
A TensorRT engine is a GPU binary: ``engine_load_path`` and ``engine_save_path`` are refused.
``memory_workspace`` and ``use_cuda_graph`` are read and checked as CUDA-Q reads them, and change
nothing else. CUDA-Q's warnings are RuntimeWarnings, the caller's (at the line that made the
decoder), and are given on every construction, as CUDA-Q logs them: the "default" and "module"
actions do not show them only once per line or module, though the caller's filters ("ignore",
"once", "error", ...) apply."""

from __future__ import annotations

import sys
import warnings
from typing import Any, List, Optional, Tuple

import numpy as np

from . import _onnx
from ._decoders import BatchDecoderResult, DecoderResult, _batch, _Inputs, _make_result, _NativeDecoder

_INIT_FAILED = "TensorRT initialization failed: "

# TensorRT's types for the ONNX types it imports as network inputs and outputs, named as CUDA-Q
# names them (dataTypeName; "unknown" for those it does not name).
_TRT_NAMES = {1: "float32", 10: "float16", 3: "int8", 6: "int32", 9: "bool", 2: "uint8", 17: "fp8", 16: "bfloat16", 7: "int64", 22: "unknown"}
# The names onnx-tensorrt's importers give the types they refuse.
_TRT_TYPES = {2: "UINT8", 6: "INT32", 7: "INT64"}


# The most elements an engine's input or output has: CUDA-Q holds input_size and output_size in ints.
_MAX_VOLUME = 2**31 - 1


def _warn(message: str) -> None:
    """CUDA-Q's log warning, as a RuntimeWarning of the caller: of the first frame outside this
    package, its file, line and module (for filters). It is shown on every construction, as
    CUDA-Q logs it: no registry keeps the "default" or "module" action from showing it again
    (nothing is written to the caller's); the filters, and "once"'s registry, are Python's own.
    No module globals are passed: Python 3.12+ checks them for a __spec__.loader, which an
    interactive or script __main__ lacks (failing, or warning); the line shown is read from the
    file name, as for any warning."""
    frame = sys._getframe(1)
    while frame.f_back is not None and str(frame.f_globals.get("__name__", "")).split(".")[0] == "stabilizer_qec":
        frame = frame.f_back
    module = frame.f_globals.get("__name__", "<string>")
    warnings.warn_explicit(message, RuntimeWarning, frame.f_code.co_filename, frame.f_lineno, module=module, registry=None)


def _volume(dims: Any) -> int:
    n = 1
    for d in dims:
        n *= int(d)
    return n


def _too_large(kind: str, name: str, dims: Tuple[int, ...], batch: int = 0) -> Optional[ValueError]:
    """Why CUDA-Q's int count of the elements of an engine input or output of these dimensions (a
    dynamic batch's at ``batch``) does not hold them, naming the dimension that takes the count
    past it; None when it does. A static shape CUDA-Q counts in the int dimension by dimension
    (cpp:677-679, 688-690), so a count past it fails even when a later dimension of 0 leaves no
    elements; a dynamic batch it counts as the batch times the rest, counted in a size_t
    (cpp:664-668, 692-694)."""
    volume = _volume(dims)
    if batch and volume <= _MAX_VOLUME:
        return None
    n = 1
    for j, d in enumerate(dims):
        n *= d
        if n > _MAX_VOLUME:
            at = f" at batch_size {batch}" if batch else ""
            if volume > _MAX_VOLUME:
                return ValueError(f"{kind} '{name}' of shape {tuple(dims)}{at} has {volume} elements: dimension {j} ({d}) takes it past {_MAX_VOLUME}, the most CUDA-Q's int {kind}_size holds")
            return ValueError(f"{kind} '{name}' of shape {tuple(dims)} has {volume} elements, but CUDA-Q counts them dimension by dimension in its int {kind}_size: dimension {j} ({d}) takes the count past {_MAX_VOLUME}, the most it holds")
    return None


def _get(params: dict, key: str, kind: str) -> Any:
    """heterogeneous_map::get<T>: a string, a size_t (an int stored, not a bool or a float), a
    bool (a bool or an int) or a map."""
    v = params[key]
    if kind == "string":
        ok = isinstance(v, str)
    elif kind == "uint64":
        ok = isinstance(v, (int, np.integer)) and not isinstance(v, (bool, np.bool_))
    elif kind == "boolean":
        ok = isinstance(v, (bool, np.bool_, int, np.integer))
    else:
        ok = isinstance(v, dict)
    if not ok:
        raise RuntimeError(f"heterogeneous_map::get() error - Invalid type or key ({key}).")
    return bool(v) if kind == "boolean" else (int(v) if kind == "uint64" else v)


def _dynamic(d: Any) -> bool:
    return not isinstance(d, int)


def _tensorrt_import(model: _onnx.OnnxModel) -> None:
    """What TensorRT's ONNX parser refuses that onnxruntime would run (ValueError: the parse
    fails), and its quantization, set on the model."""
    for name, t, shape in model._in + model._out:
        if t not in _TRT_NAMES:
            raise ValueError(f"TensorRT has no type for '{name}' ({_onnx._type_name(t)})")
    for name, _t, shape in model._in:
        if shape is None:
            raise ValueError(f"input '{name}' has no shape, and TensorRT needs at least its rank")
    # Each value's type as TensorRT has it: its first input's for most operators.
    types = {name: t for name, (_a, t) in model._inits.items()}
    types.update((name, t) for name, t, _s in model._in)
    for i, node in enumerate(model._nodes):
        where = f"In node {i} with name: {node.name} and operator: {node.op}: "
        t = types[node.inputs[0]]
        if node.op == "Cast":
            types[node.outputs[0]] = node.attrs["to"]
        elif node.op == "QuantizeLinear":
            # (To float8 when the zero point or output_dtype says so; to int8 otherwise, uint8 included.)
            target = types[node.inputs[2]] if len(node.inputs) > 2 and node.inputs[2] else node.attrs.get("output_dtype", 0)
            types[node.outputs[0]] = _onnx.FP8 if target == _onnx.FP8 else _onnx.INT8
        elif node.op == "DequantizeLinear":
            types[node.outputs[0]] = types[node.inputs[1]]
        else:
            types[node.outputs[0]] = t
        # onnx-tensorrt's checkNotInvalidType (onnxOpImporters.cpp): MatMul and Gemm do not take
        # INT32 or INT64 (cpp:4376, 2750); DequantizeLinear does not take UINT8 unless asymmetric
        # quantization is enabled for DLA (cpp:2027-2030), checked before its scale.
        if node.op in ("MatMul", "Gemm") and t in (_onnx.INT32, _onnx.INT64):
            raise ValueError(where + f"input 0 of type {_TRT_TYPES[t]} is not supported")
        if node.op == "DequantizeLinear" and t == _onnx.UINT8:
            raise ValueError(where + "input 0 of type UINT8 is not supported without kENABLE_UINT8_AND_ASYMMETRIC_QUANTIZATION_DLA")
        if node.op not in ("QuantizeLinear", "DequantizeLinear"):
            continue
        scale = model._inits.get(node.inputs[1])
        if scale is not None:
            if scale[0].size == 0:
                raise ValueError(where + "Cannot have scale with no coefficients.")
            if scale[1] in (_onnx.FLOAT, _onnx.FLOAT16) and not np.all(scale[0] > 0):
                raise ValueError(where + "Scale coefficients must all be positive")
        zp = model._inits.get(node.inputs[2]) if len(node.inputs) > 2 and node.inputs[2] else None
        if zp is not None and zp[1] != _onnx.FP8 and np.any(zp[0] != 0):
            raise ValueError(where + "Non-zero zero point is not supported. Please set kENABLE_UINT8_AND_ASYMMETRIC_QUANTIZATION_DLA to enable asymmetric quantization if it is on DLA.")
    model._tensorrt = True


def _precision(precision: str) -> None:
    """parse_precision: a strongly-typed network computes in the model's types."""
    if precision in ("tf32", "noTF32", "best"):
        return
    if precision in ("fp16", "bf16", "int8", "fp8"):
        _warn(f"Precision '{precision}' is ignored when building strongly-typed networks. TensorRT uses the data types defined in the ONNX model. To use {precision}, export the model with the desired types.")
    else:
        _warn(f"Unknown precision '{precision}', using default (best)")


def _zeros(model: _onnx.OnnxModel, dims: Tuple[int, ...]) -> np.ndarray:
    return np.zeros(dims, dtype=np.float32 if model._in[0][1] == _onnx.FLOAT else np.uint8)


class _TrtDecoder(_NativeDecoder):
    """CUDA-Q's ``trt_decoder`` (the ONNX model on the CPU). Its result is the network's output
    (float32 outputs as they are, uint8 ones as 0/1), its observable prefix when the model has
    observables (``O``), or with a global decoder that decoder's result, combined with the prefix
    when there are observables. ``output`` is not read: CUDA-Q's decodes to observables exactly
    when ``O`` is given."""

    _raw_output = True

    def __init__(self, inputs: _Inputs, output: Optional[str], params: dict) -> None:
        super().__init__(inputs, "errors", params)
        try:
            self._construct(inputs, params)
        except Exception as e:
            # (The cause is what failed: a parse or build failure's reason, or the failure itself.)
            raise RuntimeError(_INIT_FAILED + str(e)) from (e.__cause__ or e)

    def _construct(self, inputs: _Inputs, params: dict) -> None:
        has_onnx, has_engine = "onnx_load_path" in params, "engine_load_path" in params
        if has_onnx and has_engine:
            raise RuntimeError("TensorRT decoder cannot specify both 'onnx_load_path' and 'engine_load_path' parameters. Please choose one.")
        if not has_onnx and not has_engine:
            raise RuntimeError("TensorRT decoder requires either 'onnx_load_path' or 'engine_load_path' parameter")
        if has_engine:
            _get(params, "engine_load_path", "string")
            raise RuntimeError("engine_load_path is not supported: a TensorRT engine is a GPU binary, and this trt_decoder runs the ONNX model on the CPU (give onnx_load_path)")
        model, in_dims, out_dims, batch = self._build(_get(params, "onnx_load_path", "string"), params)
        if "engine_save_path" in params:
            _get(params, "engine_save_path", "string")
            raise RuntimeError("engine_save_path is not supported: this trt_decoder builds no TensorRT engine to save")

        # The engine's I/O: one input and one output, each float32 or uint8.
        if not model._in or not model._out:
            raise RuntimeError("Failed to identify input/output tensors")
        if len(model._in) + len(model._out) != 2:
            raise RuntimeError(f"TensorRT decoder expects exactly 2 I/O tensors (1 input + 1 output), got {len(model._in) + len(model._out)}")
        in_t, out_t = model._in[0][1], model._out[0][1]
        for which, t in (("input", in_t), ("output", out_t)):
            if t not in (_onnx.FLOAT, _onnx.UINT8):
                raise RuntimeError(f"Unsupported {which} tensor data type: {_TRT_NAMES.get(t, 'unknown')}. Supported types: float32, uint8")
        assert in_dims is not None and out_dims is not None  # (a model of float32 or uint8 I/O is always run)

        # Sizes: a dynamic batch is batch_size; a static model's batch is its first dimension.
        self._dynamic = len(in_dims) > 0 and in_dims[0] == -1
        if self._dynamic:
            self._model_batch = batch
            self._per_sample = int(np.prod(in_dims[1:], dtype=np.int64))
            self._input_size = self._model_batch * self._per_sample
        else:
            self._model_batch = in_dims[0] if len(in_dims) > 0 and in_dims[0] > 0 else 1
            self._input_size = int(np.prod(in_dims, dtype=np.int64))
            self._per_sample = self._input_size // self._model_batch
        self._out_per_sample = int(np.prod([d if d > 0 else 1 for d in out_dims[1:]], dtype=np.int64))
        if len(out_dims) > 0 and out_dims[0] > 0:
            self._output_size = int(np.prod(out_dims, dtype=np.int64))
            self._out_per_sample = self._output_size // self._model_batch
        else:
            self._output_size = self._model_batch * self._out_per_sample
        self._model, self._in_dims = model, in_dims
        self._float_in, self._float_out = in_t == _onnx.FLOAT, out_t == _onnx.FLOAT

        # (Executor selection: CUDA graphs need static shapes.)
        use_cuda_graph = _get(params, "use_cuda_graph", "boolean") if "use_cuda_graph" in params else True
        if use_cuda_graph and (-1 in in_dims or -1 in out_dims):
            _warn("Model has dynamic shapes or multiple profiles, CUDA graphs not supported. Using traditional execution.")

        self._global: Optional[Any] = None
        if "global_decoder" in params and "global_decoder_params" in params:
            name = _get(params, "global_decoder", "string")
            gparams = _get(params, "global_decoder_params", "map")
            if name:
                self._global = _global_decoder(name, inputs, gparams)

        # The observables: the model's output is [pre_L, residual detectors].
        if inputs.obs_rank is not None:
            # (params.get<tensor<uint8_t>>("O"), then its rank, cpp:781-785.)
            if inputs.obs_rank == 0:
                raise RuntimeError("heterogeneous_map::get() error - Invalid type or key (O).")
            raise RuntimeError("trt_decoder: O must be a 2-dimensional tensor (num_observables x block_size)")
        self._num_obs = 0
        self._to_obs = inputs.O is not None and (inputs.obs_cols is not None or inputs.num_observables > 0)
        if self._to_obs:
            if inputs.obs_cols is not None and inputs.obs_cols != inputs.num_cols:
                raise RuntimeError(f"trt_decoder: O second dimension must equal H block_size (got {inputs.obs_cols}, block_size {inputs.num_cols})")
            self._num_obs = len(inputs.O or [])
            if self._out_per_sample < self._num_obs:
                raise RuntimeError(f"trt_decoder: TRT output_size_per_sample ({self._out_per_sample}) is smaller than num_observables ({self._num_obs}); model output must be [pre_L, residual_dets].")
            if self._global is not None:
                expected = self._num_obs + self._global.get_syndrome_size()
                if self._out_per_sample != expected:
                    raise RuntimeError(f"trt_decoder: TRT output_size_per_sample ({self._out_per_sample}) must equal num_observables + global_decoder.syndrome_size ({expected}) for the [pre_L, residual_dets] split.")
            self._output = "observables"

    def _build(self, path: str, params: dict) -> Tuple[_onnx.OnnxModel, Optional[Tuple[int, ...]], Optional[Tuple[int, ...]], int]:
        """build_engine_from_onnx: the model parsed (as TensorRT imports it), the optimization
        profile for a dynamic batch, and the engine's I/O dimensions (-1 where dynamic), found by
        running the model; the batch_size of the profile."""
        try:
            with open(path, "rb") as f:
                data = f.read()
        except (OSError, ValueError):
            raise RuntimeError("Failed to open file: " + path) from None
        runnable = False
        try:
            model = _onnx.OnnxModel(data)
            _tensorrt_import(model)
            # (Only a model the decoder can take is run: one float32 or uint8 input and output.)
            runnable = len(model._in) == 1 and len(model._out) == 1 and model._in[0][1] in (_onnx.FLOAT, _onnx.UINT8) and model._out[0][1] in (_onnx.FLOAT, _onnx.UINT8)
            in_shape = model._in[0][2] if model._in else ()
            static = not any(_dynamic(d) for _n, _t, s in model._in for d in s)
            in_dims = out_dims = None
            # (The parser infers every static shape: one that does not fit fails the parse. An
            # input too large to count is not run: it fails the build, below.)
            if runnable and static and _too_large("input", "", tuple(in_shape)) is None:
                in_dims = tuple(in_shape)
                out_dims = model.run(_zeros(model, in_dims)).shape
        except ValueError as e:
            raise RuntimeError("Failed to parse ONNX model: " + path) from e
        if "memory_workspace" in params:
            _get(params, "memory_workspace", "uint64")
        _precision(_get(params, "precision", "string") if "precision" in params else "best")
        batch = 0
        if "batch_size" in params and model._in and len(in_shape) > 0 and _dynamic(in_shape[0]):
            batch = _get(params, "batch_size", "uint64")
            if batch < 1:
                raise RuntimeError(f"batch_size must be >= 1, got {batch}")
            if any(_dynamic(d) for d in in_shape[1:]) or batch > 2**31 - 1:
                raise RuntimeError("Failed to set optimization profile dimensions for batch")
        # Every dynamic dimension needs the profile, which covers the first input's batch only.
        # (TensorRT's build fails: CUDA-Q's own message for a dynamic batch without batch_size is
        # not reached. Why is the failure's cause.)
        for k, (name, _t, shape) in enumerate(model._in):
            for j, d in enumerate(shape):
                if _dynamic(d) and not (k == 0 and j == 0 and batch):
                    why = "the input's batch dimension is dynamic, and batch_size, which sizes its profile, is not set" if k == j == 0 else f"dimension {j} of input '{name}' is dynamic, and the one profile is for the first input's batch"
                    raise RuntimeError("Failed to build TensorRT engine") from ValueError("no optimization profile covers a dynamic input: " + why)
        if not model._out:
            raise RuntimeError("Failed to build TensorRT engine") from ValueError("Network must have at least one output")
        if runnable:
            # The engine's input and output must each be countable in an int, as CUDA-Q counts
            # them (a dynamic batch's at batch_size); the input is checked before it is made.
            too_large = _too_large("input", model._in[0][0], tuple(batch if _dynamic(d) else d for d in in_shape), 0 if static else batch)
            if too_large is None and not static:
                in_dims, out_dims = self._profile_dims(model, in_shape, batch)
            if too_large is None:
                # (The output as CUDA-Q sizes it: its dimensions when its batch is static, else the
                # model's batch times its other dimensions, one that is not positive counted as 1.)
                assert out_dims is not None
                if len(out_dims) > 0 and out_dims[0] > 0:
                    too_large = _too_large("output", model._out[0][0], tuple(out_dims))
                else:
                    model_batch = batch if not static else (in_shape[0] if len(in_shape) > 0 and in_shape[0] > 0 else 1)
                    too_large = _too_large("output", model._out[0][0], (model_batch,) + tuple(d if d > 0 else 1 for d in out_dims[1:]), 0 if static else batch)
            if too_large is not None:
                raise RuntimeError("Failed to build TensorRT engine") from too_large
        return model, in_dims, out_dims, batch

    @staticmethod
    def _profile_dims(model: _onnx.OnnxModel, in_shape: tuple, batch: int) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
        """A dynamic batch's engine dimensions: the output's found at batches 1 and 2, those that
        change being dynamic (-1). Batch 1 is in every profile; batch 2 is beyond a profile of
        one, so there it only tells the shape and may fail (or, too large to count, is not
        run)."""
        rest = tuple(in_shape[1:])
        try:
            first = model.run(_zeros(model, (1,) + rest)).shape
        except ValueError as e:
            raise RuntimeError("Failed to build TensorRT engine") from e
        second: Optional[tuple] = None
        if 2 * _volume(rest) <= _MAX_VOLUME:  # (always so for a batch_size of 2 or more, checked)
            try:
                second = model.run(_zeros(model, (2,) + rest)).shape
            except ValueError as e:
                if batch >= 2:
                    raise RuntimeError("Failed to build TensorRT engine") from e
        if second is not None and len(second) != len(first):
            raise RuntimeError("Failed to build TensorRT engine")
        out = tuple(-1 if second is not None and a != b else a for a, b in zip(first, second or first))
        return (-1,) + rest, out

    # -- Decoding ---------------------------------------------------------------------------------

    def decode(self, syndrome: Any) -> DecoderResult:
        s = np.asarray(syndrome, dtype=np.float64).reshape(-1)
        if s.size != self._per_sample:
            raise RuntimeError(f"Syndrome size mismatch: expected {self._per_sample} but got {s.size}")
        # (A model batch of more than one: the syndrome and zero syndromes after it.)
        rows = [s] + [np.zeros(self._per_sample)] * (self._model_batch - 1)
        return self._decode_rows(rows)[0]

    def decode_batch(self, syndromes: Any) -> BatchDecoderResult:
        rows = [np.asarray(s, dtype=np.float64).reshape(-1) for s in syndromes]
        if not rows:
            return _batch([])
        for i, r in enumerate(rows):
            if r.size != self._per_sample:
                raise RuntimeError(f"Syndrome size mismatch at index {i}: expected {self._per_sample} but got {r.size}")
        return _batch(self._decode_rows(rows))

    def _decode_rows(self, rows: List[np.ndarray]) -> List[DecoderResult]:
        results: List[DecoderResult] = []
        try:
            for start in range(0, len(rows), self._model_batch):
                chunk = rows[start : start + self._model_batch]
                results += self._chunk(chunk)
        except Exception as e:
            raise RuntimeError("TensorRT batch inference failed: " + str(e)) from e
        if len(results) != len(rows):
            raise RuntimeError(f"TensorRT decode_batch produced {len(results)} results for {len(rows)} syndromes")
        return results

    def _chunk(self, chunk: List[np.ndarray]) -> List[DecoderResult]:
        n, ps, ops = len(chunk), self._per_sample, self._out_per_sample
        # The input buffer: float syndromes as they are, uint8 ones thresholded; zeros past the chunk.
        x = np.zeros(self._input_size, dtype=np.float32 if self._float_in else np.uint8)
        for i, s in enumerate(chunk):
            x[i * ps : (i + 1) * ps] = s.astype(np.float32) if self._float_in else (s >= 0.5)
        # (A static model runs on the whole buffer, a dynamic batch on the chunk's rows.)
        x = x[: n * ps].reshape((n,) + self._in_dims[1:]) if self._dynamic else x.reshape(self._in_dims)
        y = self._model.run(x).reshape(-1)
        out = np.zeros(self._output_size, dtype=y.dtype)
        k = min(y.size, self._output_size)
        out[:k] = y[:k]
        rows = (out.astype(np.float64) if self._float_out else (out != 0).astype(np.float64))[: n * ops].reshape(n, ops)
        if self._global is None:
            width = self._num_obs if self._to_obs else ops
            return [_make_result(True, r[:width]) for r in rows]
        pre = self._num_obs if self._to_obs else 0
        size = self._global.get_syndrome_size()
        if ops - pre != size:
            raise RuntimeError(f"trt_decoder: residual portion of TRT output ({ops - pre}) != global_decoder.syndrome_size ({size})")
        found = list(self._global.decode_batch([(r[pre:] >= 0.5).astype(np.float64) for r in rows]))
        if not self._to_obs:
            return found
        combined = []
        for r, g in zip(rows, found):
            frame = np.asarray(g.result, dtype=np.float64).reshape(-1)
            bits = [float((r[k] >= 0.5) != (k < frame.size and frame[k] >= 0.5)) for k in range(self._num_obs)]
            combined.append(_make_result(g.converged, bits))
        return combined


def _global_decoder(name: str, inputs: _Inputs, params: dict) -> Any:
    """decoder::get(name, H, params), or from the model text when params give ``stim_dem``."""
    from ._decoders import _get_decoder

    params = dict(params)
    if "stim_dem" in params:
        source: Any = params["stim_dem"]
        if not isinstance(source, str):
            raise RuntimeError("heterogeneous_map::get() error - Invalid type or key (stim_dem).")
    else:
        source = np.zeros((inputs.num_rows, inputs.num_cols), dtype=np.uint8)
        for j, col in enumerate(inputs.columns):
            source[col, j] = 1
    # (CUDA-Q's pymatching decodes to observables whenever its model has them, which this
    # package's does when asked; the frame trt_decoder combines is that.)
    if name == "pymatching" and "output" not in params:
        if "O" in params or (isinstance(source, str) and _dem_observables(source) > 0):
            params["output"] = "observables"
    return _get_decoder(name, source, params, typed=True)


def _dem_observables(text: str) -> int:
    from ._dem import _parse_stim_dem

    return _parse_stim_dem(text, False)[1]

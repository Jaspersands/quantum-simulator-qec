# This package's own reader and evaluator of ONNX models. The message and field numbers are those of
# onnx/onnx.proto (Copyright (c) ONNX Project Contributors, Apache License 2.0); the operators'
# results are onnxruntime's (its CPU kernels, its float8 conversion: Copyright (c) Microsoft
# Corporation, MIT License), checked against outputs it recorded; see THIRD_PARTY_NOTICES.md.
"""ONNX models in numpy: ``load_onnx(path)`` reads a model (the protobuf wire format, read here:
the ModelProto subset a feed-forward network uses) and ``OnnxModel.run(x)`` evaluates it on the
CPU, as onnxruntime would.

The operators are those CUDA-Q QEC's ``trt_decoder`` models use: Gemm, MatMul, Add, Relu, Sigmoid,
Identity, Cast, Reshape, Flatten, QuantizeLinear and DequantizeLinear (int8, uint8 and
float8e4m3fn). A model using any other is refused when loaded ("unsupported ONNX op <name>"). A
malformed model (truncated, inconsistent, or with what this reader does not support: external
data, sparse or segmented tensors, subgraphs) raises ValueError, at ``load_onnx`` or at ``run``
when its shapes do not fit. Values are typed as ONNX types them; a float8 output is returned
decoded to float32."""

from __future__ import annotations

import struct
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

__all__ = ["OnnxModel", "load_onnx"]

# -- The protobuf wire format ---------------------------------------------------------------------

_U64 = 1 << 64


def _varint(buf: bytes, pos: int, end: int) -> Tuple[int, int]:
    result = shift = 0
    for _ in range(10):
        if pos >= end:
            raise ValueError("ONNX model is truncated (in a varint)")
        b = buf[pos]
        pos += 1
        result |= (b & 0x7F) << shift
        if b < 0x80:
            return result % _U64, pos
        shift += 7
    raise ValueError("ONNX model has a varint longer than 10 bytes")


def _fields(buf: bytes, start: int, end: int):
    """Each field of the message buf[start:end]: (number, wire type, value), the value an int
    (varint), bytes (fixed 64 or 32 bits) or a (start, end) span (length-delimited)."""
    pos = start
    while pos < end:
        key, pos = _varint(buf, pos, end)
        field, wire = key >> 3, key & 7
        if not 0 < field < 1 << 29:
            raise ValueError(f"ONNX model has a field numbered {field}")
        if wire == 0:
            value, pos = _varint(buf, pos, end)
        elif wire in (1, 5):
            n = 8 if wire == 1 else 4
            if end - pos < n:
                raise ValueError(f"ONNX model is truncated (in a {8 * n}-bit field)")
            value, pos = buf[pos : pos + n], pos + n
        elif wire == 2:
            n, pos = _varint(buf, pos, end)
            if n > end - pos:
                raise ValueError("ONNX model is truncated (in a length-delimited field)")
            value, pos = (pos, pos + n), pos + n
        else:
            raise ValueError(f"ONNX model has a field of wire type {wire} (groups and invalid types are not read)")
        yield field, wire, value


def _signed(v: int) -> int:
    return v - _U64 if v >= 1 << 63 else v


class _Msg:
    """A message's fields by number (each occurrence kept, in order)."""

    def __init__(self, buf: bytes, spans: List[Tuple[int, int]], what: str) -> None:
        self.buf, self.what = buf, what
        self.f: Dict[int, List[Tuple[int, Any]]] = {}
        for start, end in spans:
            for field, wire, value in _fields(buf, start, end):
                self.f.setdefault(field, []).append((wire, value))

    def _bad(self, field: int) -> ValueError:
        return ValueError(f"ONNX {self.what}: field {field} has an unexpected wire type")

    def has(self, field: int) -> bool:
        return field in self.f

    def int(self, field: int, default: int = 0) -> int:
        vals = self.f.get(field)
        if not vals:
            return default
        wire, v = vals[-1]
        if wire != 0:
            raise self._bad(field)
        return _signed(v)

    def ints(self, field: int, signed: bool = True) -> List[int]:
        out: List[int] = []
        for wire, v in self.f.get(field, ()):
            if wire == 0:
                out.append(v)
            elif wire == 2:
                pos, end = v
                while pos < end:
                    x, pos = _varint(self.buf, pos, end)
                    out.append(x)
            else:
                raise self._bad(field)
        return [_signed(v) for v in out] if signed else out

    def _fixed(self, field: int, size: int, fmt: str) -> List[float]:
        raw = bytearray()
        for wire, v in self.f.get(field, ()):
            if wire == (5 if size == 4 else 1):
                raw += v
            elif wire == 2:
                if (v[1] - v[0]) % size:
                    raise ValueError(f"ONNX {self.what}: packed field {field} is not a whole number of values")
                raw += self.buf[v[0] : v[1]]
            else:
                raise self._bad(field)
        return list(np.frombuffer(bytes(raw), dtype=fmt))

    def floats(self, field: int) -> List[float]:
        return self._fixed(field, 4, "<f4")

    def doubles(self, field: int) -> List[float]:
        return self._fixed(field, 8, "<f8")

    def float(self, field: int, default: float) -> float:
        vals = self.f.get(field)
        if not vals:
            return default
        wire, v = vals[-1]
        if wire != 5:
            raise self._bad(field)
        return float(struct.unpack("<f", v)[0])

    def bytes(self, field: int) -> Optional[bytes]:
        vals = self.f.get(field)
        if not vals:
            return None
        wire, v = vals[-1]
        if wire != 2:
            raise self._bad(field)
        return self.buf[v[0] : v[1]]

    def str(self, field: int) -> str:
        b = self.bytes(field)
        return "" if b is None else _utf8(b, self.what)

    def strs(self, field: int) -> List[str]:
        out = []
        for wire, v in self.f.get(field, ()):
            if wire != 2:
                raise self._bad(field)
            out.append(_utf8(self.buf[v[0] : v[1]], self.what))
        return out

    def msgs(self, field: int, what: str) -> List["_Msg"]:
        out = []
        for wire, v in self.f.get(field, ()):
            if wire != 2:
                raise self._bad(field)
            out.append(_Msg(self.buf, [v], what))
        return out

    def msg(self, field: int, what: str) -> Optional["_Msg"]:
        # (Occurrences of a singular message field merge, as protobuf merges them.)
        spans = []
        for wire, v in self.f.get(field, ()):
            if wire != 2:
                raise self._bad(field)
            spans.append(v)
        return _Msg(self.buf, spans, what) if spans else None


def _utf8(b: bytes, what: str) -> str:
    try:
        return b.decode("utf-8")
    except UnicodeDecodeError:
        raise ValueError(f"ONNX {what}: a string is not UTF-8") from None


# -- Types ----------------------------------------------------------------------------------------

FLOAT, UINT8, INT8, INT32, INT64, BOOL, FLOAT16, DOUBLE, FP8 = 1, 2, 3, 6, 7, 9, 10, 11, 17
_NAMES = {
    1: "float32", 2: "uint8", 3: "int8", 4: "uint16", 5: "int16", 6: "int32", 7: "int64", 8: "string", 9: "bool", 10: "float16", 11: "float64", 12: "uint32",
    13: "uint64", 14: "complex64", 15: "complex128", 16: "bfloat16", 17: "float8e4m3fn", 18: "float8e4m3fnuz", 19: "float8e5m2", 20: "float8e5m2fnuz",
    21: "uint4", 22: "int4", 23: "float4e2m1", 24: "float8e8m0",
}
# The types evaluated, as numpy holds them (float8e4m3fn as its bit patterns).
_NUMPY = {1: np.float32, 2: np.uint8, 3: np.int8, 4: np.uint16, 5: np.int16, 6: np.int32, 7: np.int64, 9: np.bool_, 10: np.float16, 11: np.float64, 12: np.uint32, 13: np.uint64, 17: np.uint8}
_FLOATS = (FLOAT, FLOAT16, DOUBLE)


def _type_name(t: int) -> str:
    return _NAMES.get(t, "undefined" if t == 0 else f"type {t}")


def _numpy(t: int, what: str) -> Any:
    if t not in _NUMPY:
        raise ValueError(f"{what}: ONNX type {_type_name(t)} is not supported")
    return _NUMPY[t]


# float8e4m3fn: sign, 4 exponent bits (bias 7), 3 mantissa bits; no infinities, NaN at S.1111.111.
def _fp8_table() -> np.ndarray:
    b = np.arange(256)
    e, m = (b >> 3) & 15, b & 7
    v = np.where(e == 0, m / 8.0 * 2.0**-6, (1 + m / 8.0) * 2.0 ** (e - 7.0))
    v = np.where((b & 0x7F) == 0x7F, np.nan, v)
    return np.where(b & 0x80, -v, v).astype(np.float32)


_FP8_VALUES = _fp8_table()


def _from_fp8(bits: np.ndarray) -> np.ndarray:
    return _FP8_VALUES[np.asarray(bits, dtype=np.uint8)]


def _to_fp8(x: np.ndarray, saturate: bool) -> np.ndarray:
    """float32 to float8e4m3fn bits, rounding to nearest even, as onnxruntime converts (with its
    behaviour when not saturating: [480, 496) becomes 448, the rest past 464 NaN)."""
    b = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32).astype(np.int64)
    sign = (b >> 24) & 0x80
    e = (b >> 23) & 0xFF
    m = b & 0x7FFFFF
    # Normal numbers (2**-6 <= |x| < 2**9).
    v = ((e - 120) << 3) | (m >> 20)
    v = np.where((v & 0x7F) == 0x7F, v & 0xFE, v)
    rnd = ((m & 0x80000) != 0) & (((m & 0x100000) != 0) | ((m & 0x7FFFF) != 0))
    room = (v & 0x7F) < 0x7E
    v = np.where(rnd & room, v + 1, v)
    if not saturate:
        v = np.where(rnd & ~room, v | 0x7F, v)
    # Subnormal numbers (2**-10 <= |x| < 2**-6).
    d = np.clip(120 - e, 0, 3)
    s = np.where(d < 3, (1 << (2 - np.minimum(d, 2))) | (m >> (21 + np.minimum(d, 2))), (m > 0).astype(np.int64))
    mask = np.int64(1) << (20 + d)
    up = ((m & mask) != 0) & (((s & 1) != 0) | ((m & (mask - 1)) != 0) | (((m & (mask << 1)) != 0) & ((m & (mask - 1)) == 0)))
    s = np.where(up, s + 1, s)
    big = 126 if saturate else 0x7F
    out = np.where(e >= 136, big, np.where(e >= 121, v, np.where(e >= 117, s, 0)))
    out = np.where((b & 0x7FFFFFFF) == 0x7F800000, big, out)
    out = np.where((b & 0x7FC00000) == 0x7FC00000, 0x7F, out)
    return (out | sign).astype(np.uint8)


# -- The model ------------------------------------------------------------------------------------

_INT, _FLOAT_ATTR = 2, 1
_ATTR_KINDS = {1: "float", 2: "int", 3: "string", 4: "tensor", 5: "graph", 6: "floats", 7: "ints", 8: "strings", 9: "tensors", 10: "graphs", 11: "sparse tensor", 12: "sparse tensors", 13: "type", 14: "types"}
_ATTR_FIELD = {2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 22: 11, 14: 13, 7: 6, 8: 7, 9: 8, 10: 9, 11: 10, 23: 12, 15: 14}  # value field -> AttributeType

# Each operator: its attributes and their kinds, and how many inputs it takes (the third, where
# there is one, optional).
_OPS: Dict[str, Tuple[Dict[str, int], int, int]] = {
    "Gemm": ({"alpha": _FLOAT_ATTR, "beta": _FLOAT_ATTR, "transA": _INT, "transB": _INT}, 2, 3),
    "MatMul": ({}, 2, 2),
    "Add": ({}, 2, 2),
    "Relu": ({}, 1, 1),
    "Sigmoid": ({}, 1, 1),
    "Identity": ({}, 1, 1),
    "Cast": ({"to": _INT, "saturate": _INT}, 1, 1),
    "Reshape": ({"allowzero": _INT}, 2, 2),
    "Flatten": ({"axis": _INT}, 1, 1),
    "QuantizeLinear": ({"axis": _INT, "saturate": _INT, "block_size": _INT, "output_dtype": _INT}, 2, 3),
    "DequantizeLinear": ({"axis": _INT, "block_size": _INT}, 2, 3),
}


class _Node:
    __slots__ = ("op", "name", "inputs", "outputs", "attrs")

    def __init__(self, op: str, name: str, inputs: List[str], outputs: List[str], attrs: Dict[str, Any]) -> None:
        self.op, self.name, self.inputs, self.outputs, self.attrs = op, name, inputs, outputs, attrs

    def label(self) -> str:
        return f"{self.op} '{self.name}'" if self.name else self.op


def _tensor(m: _Msg) -> Tuple[str, np.ndarray, int]:
    name = m.str(8)
    what = f"tensor '{name}'"
    if m.has(3):
        raise ValueError(f"{what} is segmented, which is not supported")
    if m.int(14) != 0 or m.has(13):
        raise ValueError(f"{what} keeps its data in an external file, which is not supported")
    t = m.int(2)
    dtype = np.dtype(_numpy(t, what))
    dims = m.ints(1)
    if any(not 0 <= d < 1 << 40 for d in dims):
        raise ValueError(f"{what} has dimensions {dims}")
    count = 1
    for d in dims:
        count *= d
    raw = m.bytes(9)
    if raw is not None:
        if len(raw) != count * dtype.itemsize:
            raise ValueError(f"{what}: raw_data holds {len(raw)} bytes; its shape {tuple(dims)} needs {count * dtype.itemsize}")
        if t == BOOL:
            arr = np.frombuffer(raw, dtype=np.uint8) != 0
        else:
            arr = np.frombuffer(raw, dtype=dtype.newbyteorder("<")).astype(dtype)
    else:
        if t == FLOAT:
            vals: Any = np.array(m.floats(4), dtype=np.float32)
        elif t == DOUBLE:
            vals = np.array(m.doubles(10), dtype=np.float64)
        elif t == INT64:
            vals = np.array(m.ints(7), dtype=np.int64)
        elif t in (12, 13):
            vals = np.array(m.ints(11, signed=False), dtype=np.uint64).astype(dtype)
        else:
            # (int32_data holds the narrower integers, booleans and float16 and float8 bit patterns.)
            ints = np.array(m.ints(5), dtype=np.int64)
            if t == BOOL:
                vals = ints != 0
            elif t == FLOAT16:
                vals = (ints & 0xFFFF).astype(np.uint16).view(np.float16)
            elif t == FP8:
                vals = (ints & 0xFF).astype(np.uint8)
            else:
                vals = ints.astype(dtype)
        if vals.size != count:
            raise ValueError(f"{what} holds {vals.size} values; its shape {tuple(dims)} needs {count}")
        arr = vals
    return name, arr.reshape(dims), t


def _value_info(m: _Msg) -> Tuple[str, int, Optional[tuple]]:
    name = m.str(1)
    if not name:
        raise ValueError("ONNX model has a graph input or output with no name")
    tp = m.msg(2, "TypeProto")
    if tp is None:
        return name, 0, None
    tt = tp.msg(1, "TypeProto.Tensor")
    if tt is None:
        if any(tp.has(k) for k in (4, 5, 8, 9)):
            raise ValueError(f"graph input or output '{name}' is not a tensor")
        return name, 0, None
    sh = tt.msg(2, "TensorShapeProto")
    shape = None
    if sh is not None:
        dims: List[Any] = []
        for d in sh.msgs(1, "TensorShapeProto.Dimension"):
            if d.has(1):
                v = d.int(1)
                if not 0 <= v < 1 << 40:
                    raise ValueError(f"'{name}' has dimension {v}")
                dims.append(v)
            else:
                dims.append(d.str(2) or None)
        shape = tuple(dims)
    return name, tt.int(1), shape


def _attribute(m: _Msg, op: str) -> Tuple[str, int, Any]:
    name = m.str(1)
    if m.str(21):
        raise ValueError(f"{op}: attribute '{name}' refers to a function's attribute, which is not supported")
    kind = m.int(20)
    if kind == 0:  # (an old model's attribute: its kind is the field it sets)
        kind = next((k for f, k in _ATTR_FIELD.items() if m.has(f)), 0)
    if kind == _FLOAT_ATTR:
        return name, kind, m.float(2, 0.0)
    if kind == _INT:
        return name, kind, m.int(3)
    return name, kind, None


def _node(m: _Msg, defined: set) -> _Node:
    op, domain, name = m.str(4), m.str(7), m.str(3)
    if domain not in ("", "ai.onnx") or op not in _OPS:
        raise ValueError(f"unsupported ONNX op {op if domain in ('', 'ai.onnx') else domain + '.' + op}")
    node = _Node(op, name, m.strs(1), m.strs(2), {})
    kinds, lo, hi = _OPS[op]
    if not lo <= len(node.inputs) <= hi:
        raise ValueError(f"{node.label()} takes {lo if lo == hi else f'{lo} to {hi}'} inputs, not {len(node.inputs)}")
    for k, i in enumerate(node.inputs):
        if not i and k < 2:
            raise ValueError(f"{node.label()} has no input {k}")
        if i and i not in defined:
            raise ValueError(f"{node.label()} reads '{i}', which nothing before it defines")
    if len(node.outputs) != 1 or not node.outputs[0]:
        raise ValueError(f"{node.label()} has {len(node.outputs)} outputs; it has one")
    for a in m.msgs(5, "AttributeProto"):
        key, kind, value = _attribute(a, op)
        if key not in kinds:
            raise ValueError(f"{node.label()} has an attribute '{key}', which is not supported")
        if kind != kinds[key]:
            raise ValueError(f"{node.label()}: attribute '{key}' must be of kind {_ATTR_KINDS[kinds[key]]}, not {_ATTR_KINDS.get(kind, 'none')}")
        if key in node.attrs:
            raise ValueError(f"{node.label()} sets attribute '{key}' twice")
        node.attrs[key] = value
    if op == "Cast":
        if "to" not in node.attrs:
            raise ValueError(f"{node.label()} has no attribute 'to'")
        _numpy(node.attrs["to"], node.label())
    if node.attrs.get("block_size", 0):
        raise ValueError(f"{node.label()}: blocked quantization (block_size) is not supported")
    if node.attrs.get("output_dtype", 0) not in (0, UINT8, INT8, FP8):
        raise ValueError(f"{node.label()}: output_dtype {_type_name(node.attrs['output_dtype'])} is not supported")
    if node.outputs[0] in defined:
        raise ValueError(f"{node.label()} defines '{node.outputs[0]}', which is already defined")
    defined.add(node.outputs[0])
    return node


class OnnxModel:
    """A loaded ONNX model: its ``inputs`` and ``outputs`` (each a (name, type, shape), the
    shape's dimensions ints or, where symbolic, their names (None where unnamed), the shape None
    where the model gives none), ``op_types`` (its nodes' operators, in order), ``ir_version``,
    ``opset`` (of the default domain), ``producer_name`` and ``graph_name``; ``run`` evaluates
    it."""

    def __init__(self, data: bytes) -> None:
        if data.startswith(b"version https://git-lfs.github.com/spec/"):
            raise ValueError("this file is a git-lfs pointer, not the model it names (fetch it with git lfs pull)")
        model = _Msg(data, [(0, len(data))], "ModelProto")
        graph = model.msg(7, "GraphProto")
        if graph is None:
            raise ValueError("not an ONNX model: it has no graph")
        self.ir_version = model.int(1)
        self.producer_name = model.str(2)
        self.graph_name = graph.str(2)
        opsets = [o.int(2) for o in model.msgs(8, "OperatorSetIdProto") if o.str(1) in ("", "ai.onnx")]
        if not opsets:
            raise ValueError("ONNX model imports no version of the default operator set")
        self.opset = opsets[-1]
        if graph.has(15):
            raise ValueError("ONNX model has sparse initializers, which are not supported")
        self._inits: Dict[str, Tuple[np.ndarray, int]] = {}
        for t in graph.msgs(5, "TensorProto"):
            name, arr, typ = _tensor(t)
            if not name or name in self._inits:
                raise ValueError(f"ONNX model has an initializer named {name!r} twice or unnamed")
            self._inits[name] = (arr, typ)
        # (An initializer listed among the inputs is a constant: the model's inputs are the rest.)
        self._in = [vi for vi in (_value_info(v) for v in graph.msgs(11, "ValueInfoProto")) if vi[0] not in self._inits]
        self._out = [_value_info(v) for v in graph.msgs(12, "ValueInfoProto")]
        defined = set(self._inits)
        for name, _t, _s in self._in:
            if name in defined:
                raise ValueError(f"ONNX model has two inputs named '{name}'")
            defined.add(name)
        self._nodes = [_node(n, defined) for n in graph.msgs(1, "NodeProto")]
        for name, _t, _s in self._out:
            if name not in defined:
                raise ValueError(f"graph output '{name}' is not computed by the graph")
        self._tensorrt = False  # (set by trt_decoder: quantize as TensorRT imports the model)

    @property
    def inputs(self) -> List[Tuple[str, str, Optional[tuple]]]:
        return [(n, _type_name(t), s) for n, t, s in self._in]

    @property
    def outputs(self) -> List[Tuple[str, str, Optional[tuple]]]:
        return [(n, _type_name(t), s) for n, t, s in self._out]

    @property
    def op_types(self) -> List[str]:
        return [n.op for n in self._nodes]

    def run(self, x: Any) -> np.ndarray:
        """The model's output for the input ``x`` (converted to the input's type; its shape must
        be the input's)."""
        if len(self._in) != 1 or len(self._out) != 1:
            raise ValueError(f"this model has {len(self._in)} inputs and {len(self._out)} outputs; run evaluates a model with one of each")
        name, t, _shape = self._in[0]
        env = self._evaluate({name: (self._feed(x, 0), t)})
        oname, ot, _ = self._out[0]
        y, yt = env[oname]
        if ot and yt != ot:
            raise ValueError(f"output '{oname}' is computed as {_type_name(yt)}; the model declares {_type_name(ot)}")
        return _from_fp8(y) if yt == FP8 else np.array(y)

    def _feed(self, x: Any, k: int) -> np.ndarray:
        name, t, shape = self._in[k]
        a = np.asarray(x)
        if shape is not None and (a.ndim != len(shape) or any(isinstance(d, int) and d != n for d, n in zip(shape, a.shape))):
            raise ValueError(f"input '{name}' has shape {tuple(a.shape)}; the model takes {shape}")
        dtype = _numpy(t, f"input '{name}'")
        with np.errstate(all="ignore"):
            return _to_fp8(a.astype(np.float32), True) if t == FP8 else a.astype(dtype)

    def _evaluate(self, feeds: Dict[str, Tuple[np.ndarray, int]]) -> Dict[str, Tuple[np.ndarray, int]]:
        env = dict(self._inits)
        env.update(feeds)
        with np.errstate(all="ignore"):
            for node in self._nodes:
                args = [env[i] if i else None for i in node.inputs]
                try:
                    y, yt = _KERNELS[node.op](self, node, args)
                except (ValueError, TypeError, IndexError, OverflowError, ZeroDivisionError) as e:
                    raise ValueError(f"{node.label()}: {e}") from e
                env[node.outputs[0]] = (np.asarray(y), yt)
        return env


def load_onnx(path: str) -> OnnxModel:
    """The ONNX model in the file ``path`` (ValueError if it is not one this module reads)."""
    with open(path, "rb") as f:
        data = f.read()
    return OnnxModel(data)


# -- The operators --------------------------------------------------------------------------------

_Arg = Optional[Tuple[np.ndarray, int]]


def _same(node: _Node, *types: int) -> int:
    if len(set(types)) != 1:
        raise ValueError(f"inputs of types {', '.join(_type_name(t) for t in types)}; they must be one type")
    return types[0]


def _numeric(t: int, op: str, allowed: Tuple[int, ...]) -> None:
    if t not in allowed:
        raise ValueError(f"{op} does not take {_type_name(t)}")


_ARITHMETIC = (FLOAT, FLOAT16, DOUBLE, UINT8, INT8, 4, 5, INT32, INT64, 12, 13)

# float32 as onnxruntime's MLAS computes it. A matrix product accumulates each output by fused
# multiply-adds in order over k, in blocks of 256 (each from zero, then added into the output
# as alpha * block + output); a product by a vector (a one-dimensional B) in one block. These are
# emulated in float64, where a product of float32s is exact.
_MLAS_STRIDE_K = 256


def _fma32(a: np.ndarray, b: Any, c: Any) -> np.ndarray:
    return (np.asarray(a, dtype=np.float64) * b + c).astype(np.float32)


def _mlas_product(a: np.ndarray, b: np.ndarray, alpha: float = 1.0, y: Optional[np.ndarray] = None, block: int = _MLAS_STRIDE_K) -> np.ndarray:
    """alpha (a @ b) + y in float32 (y None: zero), a [..., M, K] and b [..., K, N]."""
    K = a.shape[-1]
    if b.shape[-2] != K:
        raise ValueError(f"operands of shapes {a.shape} and {b.shape} do not multiply")
    a64, b64 = a.astype(np.float64), b.astype(np.float64)
    shape = np.broadcast_shapes(a.shape[:-2], b.shape[:-2]) + (a.shape[-2], b.shape[-1])
    alpha32 = np.float32(alpha)
    for start in range(0, max(K, 1), max(block, 1)):
        acc = np.zeros(shape, dtype=np.float32)
        for k in range(start, min(K, start + block)):
            acc = _fma32(a64[..., :, k : k + 1], b64[..., k : k + 1, :], acc)
        if y is None:
            y = acc if alpha == 1.0 else acc * alpha32
        else:
            y = _fma32(acc, np.float64(alpha32), y)
    return y


# MLAS's logistic: a rational approximation on [-18, 18], its multiply-adds fused.
_LOGISTIC_P = (4.37031012579801e-11, 1.15627324459942e-07, 6.08574864600143e-05, 8.51377133304701e-03, 2.48287947061529e-01)
_LOGISTIC_Q = (6.10247389755681e-13, 5.76102136993427e-09, 6.29106785017040e-06, 1.70198817374094e-03, 1.16817656904453e-01, 9.93151921023180e-01)


def _mlas_logistic(x: np.ndarray) -> np.ndarray:
    v = np.clip(x, np.float32(-18), np.float32(18))
    v2 = (v * v).astype(np.float64)
    p = np.full(v.shape, np.float32(_LOGISTIC_P[0]))
    for c in _LOGISTIC_P[1:]:
        p = _fma32(p, v2, np.float32(c))
    q = np.full(v.shape, np.float32(_LOGISTIC_Q[0]))
    for c in _LOGISTIC_Q[1:]:
        q = _fma32(q, v2, np.float32(c))
    return np.clip((p * v) / q + np.float32(0.5), np.float32(0), np.float32(1))


def _gemm(m: OnnxModel, node: _Node, args: List[_Arg]) -> Tuple[np.ndarray, int]:
    (a, ta), (b, tb) = args[0], args[1]
    c = args[2] if len(args) > 2 else None
    t = _same(node, ta, tb, *([c[1]] if c is not None else []))
    _numeric(t, "Gemm", _FLOATS)
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError(f"A and B must be matrices, not of shapes {a.shape} and {b.shape}")
    if node.attrs.get("transA", 0):
        a = a.T
    if node.attrs.get("transB", 0):
        b = b.T
    if a.shape[1] != b.shape[0]:
        raise ValueError(f"A' is {a.shape} and B' is {b.shape}")
    kind = a.dtype.type
    shape = (a.shape[0], b.shape[1])
    alpha, beta = node.attrs.get("alpha", 1.0), node.attrs.get("beta", 1.0)
    # (As onnxruntime: the output starts as beta C, and C is not read when beta is 0.)
    y0 = None
    if c is not None and beta != 0.0:
        if np.broadcast_shapes(c[0].shape, shape) != shape:
            raise ValueError(f"C of shape {c[0].shape} does not broadcast to {shape}")
        y0 = np.broadcast_to(c[0] if beta == 1.0 else kind(beta) * c[0], shape).astype(kind)
    if t == FLOAT:
        return _mlas_product(a, b, alpha, y0), t
    y = a @ b
    if alpha != 1.0:
        y = y * kind(alpha)
    return (y if y0 is None else y + y0).astype(kind, copy=False), t


def _matmul(m: OnnxModel, node: _Node, args: List[_Arg]) -> Tuple[np.ndarray, int]:
    (a, ta), (b, tb) = args[0], args[1]
    t = _same(node, ta, tb)
    _numeric(t, "MatMul", _ARITHMETIC)
    if a.ndim == 0 or b.ndim == 0:
        raise ValueError("MatMul takes tensors of rank 1 or more")
    if t != FLOAT:
        return np.matmul(a, b), t
    a2 = a[None, :] if a.ndim == 1 else a
    b2 = b[:, None] if b.ndim == 1 else b
    y = _mlas_product(a2, b2, block=b.shape[0] if b.ndim == 1 else _MLAS_STRIDE_K)
    if a.ndim == 1:
        y = y[..., 0, :]
    return (y[..., 0] if b.ndim == 1 else y), t


def _add(m: OnnxModel, node: _Node, args: List[_Arg]) -> Tuple[np.ndarray, int]:
    (a, ta), (b, tb) = args[0], args[1]
    t = _same(node, ta, tb)
    _numeric(t, "Add", _ARITHMETIC)
    return np.add(a, b), t


def _relu(m: OnnxModel, node: _Node, args: List[_Arg]) -> Tuple[np.ndarray, int]:
    x, t = args[0]
    _numeric(t, "Relu", _FLOATS + (INT8, 5, INT32, INT64))
    zero = x.dtype.type(0)
    if t in _FLOATS:
        return np.where((x > zero) | np.isnan(x), x, zero), t  # (NaN stays NaN; -0 becomes 0)
    return np.maximum(x, zero), t


def _sigmoid(m: OnnxModel, node: _Node, args: List[_Arg]) -> Tuple[np.ndarray, int]:
    x, t = args[0]
    _numeric(t, "Sigmoid", _FLOATS)
    if t == FLOAT:
        return _mlas_logistic(x), t
    one = x.dtype.type(1)
    e = np.exp(-np.abs(x))
    return np.where(x >= 0, one / (one + e), e / (one + e)).astype(x.dtype), t


def _identity(m: OnnxModel, node: _Node, args: List[_Arg]) -> Tuple[np.ndarray, int]:
    return args[0]


def _cast(m: OnnxModel, node: _Node, args: List[_Arg]) -> Tuple[np.ndarray, int]:
    x, t = args[0]
    to = node.attrs["to"]
    _numpy(t, "Cast")
    if to == t:
        return x, t
    if t == FP8:
        x = _from_fp8(x)
    if to == FP8:
        return _to_fp8(x.astype(np.float32), bool(node.attrs.get("saturate", 1))), FP8
    return x.astype(_NUMPY[to]), to


def _reshape(m: OnnxModel, node: _Node, args: List[_Arg]) -> Tuple[np.ndarray, int]:
    (x, t), (s, st) = args[0], args[1]
    if st != INT64 or s.ndim != 1:
        raise ValueError("the shape must be a one-dimensional int64 tensor")
    allowzero = node.attrs.get("allowzero", 0)
    shape = [int(v) for v in s]
    if allowzero and 0 in shape and -1 in shape:
        raise ValueError(f"shape {shape} has both 0 and -1 with allowzero set")
    out = []
    for i, d in enumerate(shape):
        if d == 0 and not allowzero:
            if i >= x.ndim:
                raise ValueError(f"shape {shape} copies dimension {i} of an input of rank {x.ndim}")
            d = x.shape[i]
        elif d < -1:
            raise ValueError(f"shape {shape} has a dimension below -1")
        out.append(d)
    if out.count(-1) > 1:
        raise ValueError(f"shape {shape} has more than one -1")
    if -1 in out:
        known = int(np.prod([d for d in out if d != -1], dtype=object))
        if known == 0 or x.size % known:
            raise ValueError(f"cannot reshape {x.shape} to {shape}")
        out[out.index(-1)] = x.size // known
    if int(np.prod(out, dtype=object)) != x.size:
        raise ValueError(f"cannot reshape {x.shape} to {shape}")
    return x.reshape(out), t


def _flatten(m: OnnxModel, node: _Node, args: List[_Arg]) -> Tuple[np.ndarray, int]:
    x, t = args[0]
    axis = node.attrs.get("axis", 1)
    if not -x.ndim <= axis <= x.ndim:
        raise ValueError(f"axis {axis} is out of range for rank {x.ndim}")
    if axis < 0:
        axis += x.ndim
    return x.reshape(int(np.prod(x.shape[:axis], dtype=object)), int(np.prod(x.shape[axis:], dtype=object))), t


def _per_axis(p: np.ndarray, x: np.ndarray, axis: int, what: str) -> np.ndarray:
    """A scale or zero point, shaped to broadcast against x: per tensor (one value) or per axis."""
    if p.size == 1 and p.ndim <= 1:
        return p.reshape(())
    if p.ndim != 1 or not -x.ndim <= axis < x.ndim or p.shape[0] != x.shape[axis]:
        raise ValueError(f"{what} of shape {p.shape} fits neither the whole input nor its axis {axis} (input {x.shape})")
    shape = [1] * x.ndim
    shape[axis] = -1
    return p.reshape(shape)


def _quantize(m: OnnxModel, node: _Node, args: List[_Arg]) -> Tuple[np.ndarray, int]:
    (x, t), (s, st) = args[0], args[1]
    zp = args[2] if len(args) > 2 else None
    if t not in (FLOAT, FLOAT16) or st != t:
        raise ValueError(f"QuantizeLinear quantizes float32 or float16 by a scale of its type, not {_type_name(t)} by {_type_name(st)}")
    axis = node.attrs.get("axis", 1)
    target = zp[1] if zp is not None else (node.attrs.get("output_dtype", 0) or UINT8)
    if target not in (UINT8, INT8, FP8):
        raise ValueError(f"QuantizeLinear to {_type_name(target)} is not supported")
    if m._tensorrt and target == UINT8:
        target = INT8  # (TensorRT quantizes to int8 where ONNX names uint8: uint8 is DLA's)
    if zp is not None and zp[0].shape != s.shape:
        raise ValueError(f"zero point of shape {zp[0].shape} for a scale of shape {s.shape}")
    v = x.astype(np.float32) / _per_axis(s, x, axis, "scale").astype(np.float32)
    if target == FP8:
        # (onnxruntime and TensorRT both ignore a float8 zero point here.)
        return _to_fp8(v, bool(node.attrs.get("saturate", 1))), FP8
    lo, hi = (0, 255) if target == UINT8 else (-128, 127)
    z = _per_axis(zp[0], x, axis, "zero point").astype(np.float64) if zp is not None and zp[1] != FP8 else 0.0
    q = np.rint(v.astype(np.float64)) + z
    q = np.where(np.isnan(q), lo, np.clip(q, lo, hi))
    return q.astype(_NUMPY[target]), target


def _dequantize(m: OnnxModel, node: _Node, args: List[_Arg]) -> Tuple[np.ndarray, int]:
    (x, t), (s, st) = args[0], args[1]
    zp = args[2] if len(args) > 2 else None
    if st not in (FLOAT, FLOAT16):
        raise ValueError(f"DequantizeLinear takes a float32 or float16 scale, not {_type_name(st)}")
    if t not in (UINT8, INT8, INT32, FP8):
        raise ValueError(f"DequantizeLinear does not take {_type_name(t)}")
    axis = node.attrs.get("axis", 1)
    if zp is not None:
        if zp[0].shape != s.shape:
            raise ValueError(f"zero point of shape {zp[0].shape} for a scale of shape {s.shape}")
        # (TensorRT reads the int8 its quantizer made where ONNX says uint8, with its zero zero point.)
        if zp[1] != t and not m._tensorrt:
            raise ValueError(f"zero point of type {_type_name(zp[1])} for {_type_name(t)} values")
    scale = _per_axis(s, x, axis, "scale").astype(np.float32)
    if t == FP8:
        if zp is not None and zp[1] == FP8 and not m._tensorrt and np.any(zp[0] & 0x7F):
            raise ValueError("onnxruntime dequantizes float8 only with a zero zero point (TensorRT ignores it)")
        return (_from_fp8(x) * scale).astype(_NUMPY[st]), st
    z = _per_axis(zp[0], x, axis, "zero point").astype(np.int64) if zp is not None and zp[1] != FP8 else 0
    return ((x.astype(np.int64) - z).astype(np.float32) * scale).astype(_NUMPY[st]), st


_KERNELS: Dict[str, Callable[[OnnxModel, _Node, List[_Arg]], Tuple[np.ndarray, int]]] = {
    "Gemm": _gemm,
    "MatMul": _matmul,
    "Add": _add,
    "Relu": _relu,
    "Sigmoid": _sigmoid,
    "Identity": _identity,
    "Cast": _cast,
    "Reshape": _reshape,
    "Flatten": _flatten,
    "QuantizeLinear": _quantize,
    "DequantizeLinear": _dequantize,
}

# Ported from CUDA-Q QEC (NVIDIA/cudaqx 0.8.1, libs/qec: include/cudaq/qec/decoder_config_schema.h,
# lib/decoder_config_schema.cpp, include/cudaq/qec/realtime/decoding_config.h,
# lib/realtime/config.cpp, python/bindings/py_decoding_config.cpp, the schemas lib/decoders/lut.cpp,
# lib/decoders/sliding_window.cpp and the pymatching, chromobius and example plugins register, and
# libs/core's heterogeneous_map and its Python conversions; Copyright (c) 2024 - 2026 NVIDIA
# Corporation & Affiliates, Apache License 2.0; see THIRD_PARTY_NOTICES.md), modified for
# stabilizer_qec: YAML read and written by ._yaml, the schemas of this package's decoders.
"""CUDA-Q QEC's real-time decoding configuration: ``decoder_config``, ``multi_decoder_config``
and the transport section, their YAML (as CUDA-Q writes and reads it), and the parameter schemas
that govern each decoder's ``decoder_custom_args``.

A decoder's parameters are a dict. Its schema (``decoder_param_schema(name)``) names the keys and
their kinds; assigning a dict after ``type`` converts it to those kinds (an int for a float, a
negative int for an int32), as CUDA-Q does, and YAML is read and written through it. Schemas are
registered for this package's decoders: ``single_error_lut``, ``multi_error_lut``,
``single_error_lut_example``, ``pymatching``, ``chromobius``, ``sliding_window`` (with its checks),
``nv-qldpc-decoder`` (and its ``srelay_bp`` section) and ``nv-fusion-decoder``. CUDA-Q's
``nv-qldpc-decoder`` and ``nv-fusion-decoder`` schemas ship in closed plugins; these are built from
their documented parameters. ``trt_decoder`` has none, as in a CUDA-Q built without TensorRT.

A DEM is described flat (``block_size``, ``syndrome_size``, ``H_sparse``, ``O_sparse``,
``D_sparse``), by a Stim model file (``stim_dem_path`` and ``D_sparse``), or as repeated phases,
``dem_chunks`` (a ``DemChunksSpec``: named per-round phases, the graph connecting them and the
round count, which ``expand_dem_chunks`` expands to the flat form, returning the closed model; see
``._extended_dem``). The YAML reads and writes all three."""

from __future__ import annotations

import copy
import enum
import json
import warnings
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import _yaml
from ._extended_dem import DemChunksSpec, DemSeamSpec, DemChunkSpec, PhaseConnection, PhaseSpecEntry, SeamConnection, SeamId, SeamSpecEntry, _register_name, _registered_name

_U64 = 2**64
_I64 = (-(2**63), 2**63 - 1)
_I32 = (-(2**31), 2**31 - 1)


def _warn(message: str) -> None:
    warnings.warn(message, RuntimeWarning, stacklevel=3)


# -- Values as a heterogeneous_map holds them -------------------------------------------------------


class _V:
    """A value and the C++ type it is stored as: "bool", "int" (int32), "size_t", "uint8",
    "double", "float", "string", "vec_double", "vec_float", "vec_int", "vec_uint8", "matrix"
    (vector<vector<double>>), "map" (a dict of these) or "tensor"."""

    __slots__ = ("t", "v")

    def __init__(self, t: str, v: Any) -> None:
        self.t, self.v = t, v

    def __repr__(self) -> str:
        return f"_V({self.t}, {self.v!r})"


def _copy_map(m: Dict[str, _V]) -> Dict[str, _V]:
    return {k: _V(x.t, _copy_map(x.v) if x.t == "map" else copy.copy(x.v)) for k, x in m.items()}


# Which stored types heterogeneous_map::get<T> reads as each kind's canonical T (T itself and
# cudaqx's RelatedTypesMap<T>).
_READABLE = {
    "boolean": {"bool", "int", "size_t"},
    "int32": {"int", "size_t"},
    "uint64": {"size_t", "int"},
    "f64": {"double", "float"},
    "string": {"string"},
    "f64_vec": {"vec_double"},
    "f64_matrix": {"matrix"},
    "subschema": {"map"},
    "discriminated": {"map"},
}
_STORED = {"boolean": "bool", "int32": "int", "uint64": "size_t", "f64": "double", "string": "string", "f64_vec": "vec_double", "f64_matrix": "matrix", "subschema": "map", "discriminated": "map"}
_KIND_DESC = {"boolean": "boolean", "int32": "32-bit int", "uint64": "non-negative int", "f64": "float", "string": "string", "f64_vec": "list-of-float", "f64_matrix": "list-of-list-of-float", "subschema": "mapping", "discriminated": "mapping"}
_KIND_PY = {"boolean": "bool", "int32": "int32", "uint64": "uint64", "f64": "float64", "string": "string", "f64_vec": "float64_vec", "f64_matrix": "float64_matrix", "subschema": "subschema", "discriminated": "discriminated"}


def _wrap(v: int, bits: int, signed: bool) -> int:
    v %= 1 << bits
    return v - (1 << bits) if signed and v >= 1 << (bits - 1) else v


def _get(m: Dict[str, _V], key: str, kind: str) -> Any:
    """heterogeneous_map::get<T> for a kind's T."""
    if key not in m:
        raise RuntimeError("Invalid key.")
    x = m[key]
    if x.t not in _READABLE[kind]:
        raise RuntimeError(f"heterogeneous_map::get() error - Invalid type or key ({key}).")
    if kind == "boolean":
        return bool(x.v)
    if kind == "int32":
        return _wrap(int(x.v), 32, True)
    if kind == "uint64":
        return _wrap(int(x.v), 64, False)
    if kind == "f64":
        return float(x.v)
    return x.v


# -- Schemas --------------------------------------------------------------------------------------


class _Spec:
    __slots__ = ("key", "kind", "required", "subschema", "discriminator", "materialize_empty")

    def __init__(self, key: str, kind: str, required: bool = False, subschema: str = "", discriminator: str = "", materialize_empty: bool = False) -> None:
        self.key, self.kind, self.required = key, kind, required
        self.subschema, self.discriminator, self.materialize_empty = subschema, discriminator, materialize_empty


class _Schema:
    __slots__ = ("name", "params", "validate")

    def __init__(self, name: str, params: List[_Spec], validate: Optional[Callable[[Dict[str, _V]], None]] = None) -> None:
        self.name, self.params, self.validate = name, params, validate

    def find(self, key: str) -> Optional[_Spec]:
        for spec in self.params:
            if spec.key == key:
                return spec
        return None


_SCHEMAS: Dict[str, _Schema] = {}


def _register(schema: _Schema) -> None:
    _SCHEMAS[schema.name] = schema


def _find(name: str) -> Optional[_Schema]:
    return _SCHEMAS.get(name)


def _sliding_window_checks(args: Dict[str, _V]) -> None:
    if "window_size" in args and "step_size" in args:
        window, step = _get(args, "window_size", "uint64"), _get(args, "step_size", "uint64")
        if step < 1 or step > window:
            raise RuntimeError(f"sliding_window parameters: step_size ({step}) must be between 1 and window_size ({window})")
    if "num_boundary_syndromes" in args and "num_syndromes_per_round" in args:
        boundary, per_round = _get(args, "num_boundary_syndromes", "uint64"), _get(args, "num_syndromes_per_round", "uint64")
        if boundary > per_round:
            raise RuntimeError(f"sliding_window parameters: num_boundary_syndromes ({boundary}) must be <= num_syndromes_per_round ({per_round})")
    if not _get(args, "error_rate_vec", "f64_vec"):
        raise RuntimeError("sliding_window parameters: error_rate_vec must be non-empty")


def _register_builtin_schemas() -> None:
    S = _Spec
    _register(_Schema("single_error_lut", []))
    _register(_Schema("multi_error_lut", [S("lut_error_depth", "int32")]))
    _register(_Schema("single_error_lut_example", []))
    _register(_Schema("pymatching", [S("error_rate_vec", "f64_vec"), S("merge_strategy", "string")]))
    _register(_Schema("chromobius", [S(k, "boolean") for k in ("drop_mobius_errors_involving_remnant_errors", "ignore_decomposition_failures", "include_coords_in_mobius_dem", "return_weight", "write_mobius_match_to_stderr")]))
    _register(
        _Schema(
            "sliding_window",
            [
                S("window_size", "uint64"),
                S("step_size", "uint64"),
                S("num_syndromes_per_round", "uint64"),
                S("num_boundary_syndromes", "uint64"),
                S("straddle_start_round", "boolean"),
                S("straddle_end_round", "boolean"),
                S("error_rate_vec", "f64_vec", True),
                S("inner_decoder_name", "string", True),
                S("inner_decoder_params", "discriminated", discriminator="inner_decoder_name"),
            ],
            _sliding_window_checks,
        )
    )
    # (CUDA-Q's nv-qldpc-decoder registers its schema from a closed plugin: this one is its
    # documented parameters, in the order of the typed configuration it replaced.)
    _register(_Schema("srelay_bp", [S("pre_iter", "int32"), S("num_sets", "int32"), S("stopping_criterion", "string"), S("stop_nconv", "int32")]))
    _register(
        _Schema(
            "nv-qldpc-decoder",
            [
                S("use_sparsity", "boolean"),
                S("error_rate", "f64"),
                S("error_rate_vec", "f64_vec"),
                S("max_iterations", "int32"),
                S("n_threads", "int32"),
                S("use_osd", "boolean"),
                S("osd_method", "int32"),
                S("osd_order", "int32"),
                S("bp_batch_size", "int32"),
                S("osd_batch_size", "int32"),
                S("iter_per_check", "int32"),
                S("clip_value", "f64"),
                S("repeatable", "boolean"),
                S("bp_method", "int32"),
                S("scale_factor", "f64"),
                S("proc_float", "string"),
                S("gamma0", "f64"),
                S("gamma_dist", "f64_vec"),
                S("explicit_gammas", "f64_matrix"),
                S("srelay_config", "subschema", subschema="srelay_bp"),
                S("bp_seed", "int32"),
                S("composition", "int32"),
                S("gamma_ensemble_size", "int32"),
            ],
        )
    )
    # (Its documentation: "decoder_custom_args accepts only num_threads, block_leaf_size,
    # fusion_strategy and error_rate_vec".)
    _register(_Schema("nv-fusion-decoder", [S("num_threads", "uint64"), S("block_leaf_size", "uint64"), S("fusion_strategy", "string"), S("error_rate_vec", "f64_vec")]))


_register_builtin_schemas()


def decoder_param_schema(decoder_name: str) -> Optional[List[dict]]:
    """The parameter schema a decoder registered (a list of {key, kind, required[, subschema]
    [, discriminator]}), or None."""
    schema = _find(decoder_name)
    if schema is None:
        return None
    out = []
    for spec in schema.params:
        entry = {"key": spec.key, "kind": _KIND_PY[spec.kind], "required": spec.required}
        if spec.subschema:
            entry["subschema"] = spec.subschema
        if spec.discriminator:
            entry["discriminator"] = spec.discriminator
        out.append(entry)
    return out


def registered_decoder_schemas() -> List[str]:
    """Names of all decoders (and nested sections) with registered parameter schemas."""
    return sorted(_SCHEMAS)


# -- Python values to and from typed maps -----------------------------------------------------------


def _is_int(v: Any) -> bool:
    return isinstance(v, (int, np.integer))  # (bool too, as nanobind's int caster takes it)


def _is_real(v: Any) -> bool:
    return isinstance(v, (int, float, np.integer, np.floating))


def _is_sequence(v: Any) -> bool:
    return isinstance(v, (list, tuple, range, np.ndarray)) and not (isinstance(v, np.ndarray) and v.ndim == 0)


class _CastError(Exception):
    pass


def _as_double(v: Any) -> float:
    if not _is_real(v):
        raise _CastError
    return float(v)


def _cast(kind: str, value: Any) -> _V:
    """A value cast to a kind's canonical type as nanobind casts it (raises _CastError)."""
    if kind == "boolean":
        if value is True or value is False or isinstance(value, np.bool_):
            return _V("bool", bool(value))
        raise _CastError
    if kind in ("int32", "uint64"):
        if not _is_int(value):
            raise _CastError
        v = int(value)
        lo, hi = _I32 if kind == "int32" else (0, _U64 - 1)
        if not lo <= v <= hi:
            raise _CastError
        return _V(_STORED[kind], v)
    if kind == "f64":
        return _V("double", _as_double(value))
    if kind == "string":
        if not isinstance(value, str):
            raise _CastError
        return _V("string", value)
    if kind == "f64_vec":
        if not _is_sequence(value):
            raise _CastError
        return _V("vec_double", [_as_double(x) for x in value])
    if kind == "f64_matrix":
        if not _is_sequence(value) or not all(_is_sequence(row) for row in value):
            raise _CastError
        return _V("matrix", [[_as_double(x) for x in row] for row in value])
    raise AssertionError(kind)


def _typed_from_dict(schema: _Schema, d: dict) -> Dict[str, _V]:
    """CUDA-Q's schema_typed_map_from_dict: known keys cast to their kinds, the rest generic."""
    out: Dict[str, _V] = {}
    residual: dict = {}
    for key, value in d.items():
        spec = schema.find(key) if isinstance(key, str) else None
        if spec is None:
            residual[key] = value
            continue
        if spec.kind in ("subschema", "discriminated"):
            nested = None
            if spec.kind == "subschema":
                nested = _find(spec.subschema)
            elif isinstance(d.get(spec.discriminator), str):
                nested = _find(d[spec.discriminator])
            if nested is not None and isinstance(value, dict):
                out[key] = _V("map", _typed_from_dict(nested, value))
            elif isinstance(value, dict):
                out[key] = _V("map", _generic(value))
            else:
                raise RuntimeError(f"Parameter '{key}' of '{schema.name}' expects a dict value.")
            continue
        try:
            out[key] = _cast(spec.kind, value)
        except _CastError:
            raise RuntimeError(f"Parameter '{key}' of '{schema.name}' expects a {_KIND_DESC[spec.kind]} value.") from None
    if residual:
        out.update(_generic(residual))
    return out


_ARRAY_TYPES = {np.dtype(np.float64): "double", np.dtype(np.float32): "float", np.dtype(np.int32): "int", np.dtype(np.uint8): "uint8"}


def _generic(d: dict) -> Dict[str, _V]:
    """cudaqx's hetMapFromKwargs: a Python dict as a heterogeneous_map, by the values' types."""
    out: Dict[str, _V] = {}
    for key, value in d.items():
        if not isinstance(key, str):
            raise RuntimeError("decoder_custom_args keys must be strings")
        if isinstance(value, bool):
            out[key] = _V("bool", value)
        elif isinstance(value, int):
            if not 0 <= value < _U64:
                raise RuntimeError(f"decoder_custom_args['{key}']: {value} does not convert to size_t (set decoder_config.type first so the decoder's schema can type it)")
            out[key] = _V("size_t", value)
        elif isinstance(value, float):
            out[key] = _V("double", float(value))
        elif isinstance(value, str):
            out[key] = _V("string", value)
        elif isinstance(value, dict):
            out[key] = _V("map", _generic(value))
        elif isinstance(value, list):
            if not value:
                continue  # (an empty list is left out, as cudaqx leaves it)
            try:
                if isinstance(value[0], list):
                    out[key] = _V("matrix", [[_as_double(x) for x in _list(row)] for row in value])
                else:
                    out[key] = _V("vec_double", [_as_double(x) for x in value])
            except _CastError:
                raise RuntimeError(f"decoder_custom_args['{key}']: the list holds a value that is not a number") from None
        elif isinstance(value, np.ndarray):
            t = _ARRAY_TYPES.get(value.dtype)
            if t is None:
                raise RuntimeError("Unsupported array data type in kwargs.")
            if value.ndim >= 2:
                if not value.flags.c_contiguous and value.flags.f_contiguous:
                    raise RuntimeError("Array in kwargs must be in row-major order, but column-major order was detected.")
                out[key] = _V("tensor", np.array(value, copy=True))
            else:
                out[key] = _V("vec_" + t, value.reshape(-1).tolist())
        else:
            raise RuntimeError("Invalid python type for mapping kwargs to a heterogeneous_map.")
    return out


def _list(row: Any) -> list:
    if not isinstance(row, list):
        raise _CastError
    return row


def _custom_args_from_python(decoder_type: str, value: Any) -> Dict[str, _V]:
    schema_name = decoder_type
    if not isinstance(value, dict) and hasattr(value, "to_heterogeneous_map"):
        # (The deprecated typed configurations reduce themselves to a dict and name their schema.)
        name = getattr(value, "_schema_name", None)
        if isinstance(name, str):
            schema_name = name
        value = value.to_heterogeneous_map()
    if not isinstance(value, dict):
        raise TypeError(f"decoder_custom_args must be a dict, not {type(value).__name__}")
    schema = _find(schema_name)
    return _typed_from_dict(schema, value) if schema is not None else _generic(value)


def _to_python(m: Dict[str, _V]) -> dict:
    """The typed map as CUDA-Q hands it to Python."""
    out: dict = {}
    for key, x in m.items():
        if x.t == "bool":
            out[key] = bool(x.v)
        elif x.t in ("size_t", "int", "uint8"):
            out[key] = int(x.v)
        elif x.t in ("double", "float"):
            out[key] = float(x.v)
        elif x.t == "string":
            out[key] = x.v
        elif x.t == "matrix":
            out[key] = [[float(v) for v in row] for row in x.v]
        elif x.t == "vec_double":
            out[key] = np.array(x.v, dtype=np.float64)
        elif x.t == "vec_int":
            out[key] = np.array(x.v, dtype=np.int32)
        elif x.t == "map":
            out[key] = _to_python(x.v)
        else:
            raise RuntimeError(f"Failed to cast from heterogeneous_map to Python dict. Unsupported data type in the '{key}' field.")
    return out


def _plain(m: Dict[str, _V]) -> dict:
    """The typed map as this package's decoders take their parameters."""
    out: dict = {}
    for key, x in m.items():
        if x.t == "map":
            out[key] = _plain(x.v)
        elif x.t == "tensor":
            out[key] = np.array(x.v, copy=True)
        elif x.t.startswith("vec_"):
            out[key] = np.array(x.v, dtype={"vec_double": np.float64, "vec_float": np.float32, "vec_int": np.int32, "vec_uint8": np.uint8}[x.t])
        elif x.t == "matrix":
            out[key] = [list(row) for row in x.v]
        else:
            out[key] = x.v
    return out


# -- Schema checks ----------------------------------------------------------------------------------


def _validate(schema: _Schema, args: Dict[str, _V], context: str) -> None:
    for key, x in args.items():
        spec = schema.find(key)
        if spec is None:
            raise RuntimeError(f"Unknown key '{key}' in {context}.")
        if x.t not in _READABLE[spec.kind]:
            raise RuntimeError(f"Key '{key}' in {context} does not hold a {_KIND_DESC[spec.kind]} value as its schema kind requires.")
        if spec.kind == "subschema":
            nested = _find(spec.subschema)
            if nested is None:
                raise RuntimeError(f"No schema registered under '{spec.subschema}' (needed to validate '{key}').")
            _validate(nested, x.v, f"{context}.{key}")
        elif spec.kind == "discriminated":
            value = _get(args, spec.discriminator, "string") if spec.discriminator in args else ""
            if not value:
                raise RuntimeError(f"'{key}' is present but '{spec.discriminator}' is not set in {context}.")
            nested = _find(value)
            if nested is None:
                raise RuntimeError(f"'{key}' does not support {spec.discriminator} '{value}': no parameter schema is registered under that name.")
            _validate(nested, x.v, f"{context}.{key}")
    for spec in schema.params:
        if spec.required and spec.key not in args:
            raise RuntimeError(f"Missing required key '{spec.key}' in {context}.")
    if schema.validate is not None:
        schema.validate(args)


def _validate_named(name: str, args: Dict[str, _V]) -> None:
    schema = _find(name)
    if schema is None:
        if not args:
            return
        raise RuntimeError(
            f"Decoder type '{name}' has no registered parameter schema; its decoder_custom_args cannot be validated (or serialized to YAML). Register a schema from the decoder's plugin library with register_decoder_schema()."
        )
    _validate(schema, args, f"'{name}' parameters")


def _nested_schema(spec: _Spec, args: Dict[str, _V]) -> Optional[_Schema]:
    if spec.kind == "subschema":
        return _find(spec.subschema)
    if spec.kind == "discriminated" and spec.discriminator in args:
        value = _get(args, spec.discriminator, "string")
        return _find(value) if value else None
    return None


def _materialize(schema: _Schema, args: Dict[str, _V]) -> None:
    for spec in schema.params:
        nested = _nested_schema(spec, args)
        if spec.kind == "discriminated" and spec.materialize_empty and nested is not None and spec.key not in args:
            args[spec.key] = _V("map", {})
        if nested is not None and spec.key in args:
            sub = _copy_map(_get(args, spec.key, "subschema"))
            _materialize(nested, sub)
            args[spec.key] = _V("map", sub)


def _drop_non_schema_keys(schema: _Schema, args: Dict[str, _V]) -> None:
    unknown = []
    for key, x in args.items():
        spec = schema.find(key)
        if spec is None:
            _warn(f"Key '{key}' is not in the '{schema.name}' parameter schema; it is excluded from the decoder parameters and from emitted YAML.")
            unknown.append(key)
            continue
        nested = _nested_schema(spec, args)
        if nested is not None and x.t == "map":
            _drop_non_schema_keys(nested, x.v)
    for key in unknown:
        del args[key]


_INTS = ("int", "size_t")
_FLOATS = ("double", "float")


def _values_equal(a: _V, b: _V) -> bool:
    if a.t == "bool" or b.t == "bool":
        return a.t == b.t and a.v == b.v
    if a.t in _INTS:
        return (b.t in _INTS and int(a.v) == int(b.v)) or (b.t in _FLOATS and float(a.v) == b.v)
    if a.t in _FLOATS:
        return (b.t in _FLOATS and a.v == b.v) or (b.t in _INTS and a.v == float(b.v))
    if a.t in ("string", "vec_double", "matrix"):
        return a.t == b.t and a.v == b.v
    if a.t == "map":
        return b.t == "map" and _maps_equal(a.v, b.v)
    return False  # (other types compare unequal, as CUDA-Q's custom_args_maps_equal has them)


def _maps_equal(a: Dict[str, _V], b: Dict[str, _V]) -> bool:
    return len(a) == len(b) and all(k in b and _values_equal(v, b[k]) for k, v in a.items())


# -- Attribute checks (as nanobind converts assignments) ---------------------------------------------


def _int_value(value: Any, lo: int, hi: int, what: str) -> int:
    if not _is_int(value) or not lo <= int(value) <= hi:
        raise TypeError(f"{what} must be an integer in [{lo}, {hi}], not {value!r}")
    return int(value)


def _sparse(value: Any, what: str) -> List[int]:
    if isinstance(value, (str, bytes, dict)) or not hasattr(value, "__iter__"):
        raise TypeError(f"{what} must be a sequence of integers")
    out = []
    for v in value:
        out.append(_int_value(v, _I64[0], _I64[1], what + " entries"))
    return out


def _str_value(value: Any, what: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{what} must be a str, not {type(value).__name__}")
    return value


def _strings(value: Any, what: str) -> List[str]:
    if isinstance(value, (str, bytes, dict)) or not hasattr(value, "__iter__"):
        raise TypeError(f"{what} must be a sequence of str")
    return [_str_value(v, what + " entries") for v in value]


# -- The configuration classes ----------------------------------------------------------------------


class decoder_dispatch(enum.Enum):
    """How a decoder's RPCs are executed (a decoding server's choice; the host decoders of
    ``configure_decoders`` run on the caller's thread either way)."""

    host = 0
    device_graph = 1


class transport_shape_override:
    """The wire override for one dispatch shape's rings (``transport_config.device_graph``)."""

    def __init__(self) -> None:
        self._provider = ""
        self._args: List[str] = []

    provider = property(lambda self: self._provider, lambda self, v: setattr(self, "_provider", _str_value(v, "provider")))
    args = property(lambda self: list(self._args), lambda self, v: setattr(self, "_args", _strings(v, "args")))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, transport_shape_override) and (self._provider, self._args) == (other._provider, other._args)

    __hash__ = None  # type: ignore[assignment]

    def _copy(self) -> "transport_shape_override":
        out = transport_shape_override()
        out._provider, out._args = self._provider, list(self._args)
        return out

    def __repr__(self) -> str:
        return f"transport_shape_override(provider={self._provider!r}, args={self._args!r})"


class transport_config:
    """A decoding server's transport section (the wire is deployment configuration, so it lives
    beside the decoders list, not in it)."""

    def __init__(self) -> None:
        self._provider = ""
        self._args: List[str] = []
        self._device_graph = transport_shape_override()

    provider = property(lambda self: self._provider, lambda self, v: setattr(self, "_provider", _str_value(v, "provider")))
    args = property(lambda self: list(self._args), lambda self, v: setattr(self, "_args", _strings(v, "args")))

    @property
    def device_graph(self) -> transport_shape_override:
        return self._device_graph

    @device_graph.setter
    def device_graph(self, value: transport_shape_override) -> None:
        if not isinstance(value, transport_shape_override):
            raise TypeError("device_graph must be a transport_shape_override")
        self._device_graph = value._copy()

    def __eq__(self, other: object) -> bool:
        return isinstance(other, transport_config) and (self._provider, self._args, self._device_graph) == (other._provider, other._args, other._device_graph)

    __hash__ = None  # type: ignore[assignment]

    def _copy(self) -> "transport_config":
        out = transport_config()
        out._provider, out._args, out._device_graph = self._provider, list(self._args), self._device_graph._copy()
        return out

    def __repr__(self) -> str:
        return f"transport_config(provider={self._provider!r}, args={self._args!r}, device_graph={self._device_graph!r})"


class decoder_config:
    """One decoder of a real-time decoding configuration: its ``id`` and ``type``, its DEM
    (flat: ``block_size``, ``syndrome_size``, ``H_sparse``, ``O_sparse`` and ``D_sparse``, each
    matrix as ``pcm_to_sparse_vec`` writes it; or phases, ``dem_chunks``, with the flat fields
    derived by ``expand_dem_chunks``; or a Stim model file, ``stim_dem_path``, with ``D_sparse``),
    and its parameters, ``decoder_custom_args`` (a dict, set after ``type``)."""

    def __init__(self) -> None:
        self._id = 0
        self._type = ""
        self._dispatch = decoder_dispatch.host
        self._cuda_device_id: Optional[int] = None
        self._block_size = 0
        self._syndrome_size = 0
        self._stim_dem_path = ""
        self._H: List[int] = []
        self._O: List[int] = []
        self._D: List[int] = []
        self._dem_chunks: Optional[DemChunksSpec] = None
        self._args: Dict[str, _V] = {}

    id = property(lambda self: self._id, lambda self, v: setattr(self, "_id", _int_value(v, _I64[0], _I64[1], "id")))
    type = property(lambda self: self._type, lambda self, v: setattr(self, "_type", _str_value(v, "type")))
    stim_dem_path = property(lambda self: self._stim_dem_path, lambda self, v: setattr(self, "_stim_dem_path", _str_value(v, "stim_dem_path")))
    block_size = property(lambda self: self._block_size, lambda self, v: setattr(self, "_block_size", _int_value(v, 0, _U64 - 1, "block_size")))
    syndrome_size = property(lambda self: self._syndrome_size, lambda self, v: setattr(self, "_syndrome_size", _int_value(v, 0, _U64 - 1, "syndrome_size")))
    H_sparse = property(lambda self: list(self._H), lambda self, v: setattr(self, "_H", _sparse(v, "H_sparse")))
    O_sparse = property(lambda self: list(self._O), lambda self, v: setattr(self, "_O", _sparse(v, "O_sparse")))
    D_sparse = property(lambda self: list(self._D), lambda self, v: setattr(self, "_D", _sparse(v, "D_sparse")))

    @property
    def dispatch(self) -> decoder_dispatch:
        return self._dispatch

    @dispatch.setter
    def dispatch(self, value: decoder_dispatch) -> None:
        if not isinstance(value, decoder_dispatch):
            raise TypeError("dispatch must be a decoder_dispatch")
        self._dispatch = value

    @property
    def cuda_device_id(self) -> Optional[int]:
        return self._cuda_device_id

    @cuda_device_id.setter
    def cuda_device_id(self, value: Optional[int]) -> None:
        self._cuda_device_id = None if value is None else _int_value(value, _I32[0], _I32[1], "cuda_device_id")

    @property
    def dem_chunks(self) -> Optional[DemChunksSpec]:
        """Optional DemChunksSpec describing the DEM as named phases. ``num_rounds`` lives inside
        the DemChunksSpec. Leave block_size, syndrome_size, H_sparse, O_sparse and D_sparse unset;
        ``expand_dem_chunks()`` derives those. (The spec itself, not a copy: assigning a spec
        copies it in.)"""
        return self._dem_chunks

    @dem_chunks.setter
    def dem_chunks(self, value: Optional[DemChunksSpec]) -> None:
        if value is not None and not isinstance(value, DemChunksSpec):
            raise TypeError(f"dem_chunks must be a DemChunksSpec or None, not {type(value).__name__}")
        self._dem_chunks = None if value is None else value._copy()

    @property
    def decoder_custom_args(self) -> dict:
        """The decoder's parameter dict (a copy: assign a changed dict back). Keys are those of
        the schema the decoder registered (``decoder_param_schema``); set ``type`` first so the
        values convert to its kinds."""
        return _to_python(self._args)

    @decoder_custom_args.setter
    def decoder_custom_args(self, value: Any) -> None:
        self._args = _custom_args_from_python(self._type, value)

    def set_decoder_custom_args(self, custom_args: Any) -> None:
        """Set the parameter dict (as assigning ``decoder_custom_args``; set ``type`` first)."""
        self._args = _custom_args_from_python(self._type, custom_args)

    def validate_custom_args(self) -> None:
        """Check ``decoder_custom_args`` against the decoder's schema: unknown keys, values of the
        wrong kind, missing required keys and the schema's own checks. Raises RuntimeError."""
        _validate_named(self._type, self._args)

    def expand_dem_chunks(self) -> Optional[Any]:
        """``expand_dem_chunks(self)``: this configuration rewritten in place from its
        ``dem_chunks`` to the flat form, returning the closed ``DetectorErrorModel`` (None if it
        is already flat or has no ``dem_chunks``). A method here as well as CUDA-Q's function."""
        return expand_dem_chunks(self)

    def _args_for_decoder(self) -> Dict[str, _V]:
        # decoder_custom_args_to_heterogeneous_map: non-schema keys dropped, defaults filled in.
        args = _copy_map(self._args)
        schema = _find(self._type)
        if schema is not None:
            _drop_non_schema_keys(schema, args)
            _materialize(schema, args)
        return args

    def to_yaml_str(self, column_wrap: int = 80) -> str:
        """This decoder as a YAML document (CUDA-Q's format)."""
        return _yaml.document(_emit_decoder(self), _int_value(column_wrap, _I32[0], _I32[1], "column_wrap"))

    @staticmethod
    def from_yaml_str(yaml_str: str) -> "decoder_config":
        """A decoder from a YAML document."""
        return _from_yaml(yaml_str, _read_decoder, lambda cfg: [cfg])

    def _fields(self) -> tuple:
        return (self._id, self._type, self._dispatch, self._cuda_device_id, self._block_size, self._syndrome_size, self._stim_dem_path, self._H, self._O, self._D)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, decoder_config) and self._fields() == other._fields() and _chunks_equal(self._dem_chunks, other._dem_chunks) and _maps_equal(self._args, other._args)

    __hash__ = None  # type: ignore[assignment]

    def _copy(self) -> "decoder_config":
        out = decoder_config()
        out.__dict__.update({k: (list(v) if isinstance(v, list) else v) for k, v in self.__dict__.items()})
        out._dem_chunks = None if self._dem_chunks is None else self._dem_chunks._copy()
        out._args = _copy_map(self._args)
        return out

    def __copy__(self) -> "decoder_config":
        return self._copy()

    def __deepcopy__(self, memo: dict) -> "decoder_config":
        return self._copy()

    def __repr__(self) -> str:
        return f"decoder_config(id={self._id}, type={self._type!r}, block_size={self._block_size}, syndrome_size={self._syndrome_size})"


def _chunks_equal(a: Optional[DemChunksSpec], b: Optional[DemChunksSpec]) -> bool:
    return a is None and b is None or (a is not None and b is not None and a._same(b))


class multi_decoder_config:
    """A real-time decoding configuration: its ``decoders`` (a list of ``decoder_config``,
    copied in and out) and a decoding server's ``transport`` section."""

    def __init__(self) -> None:
        self._decoders: List[decoder_config] = []
        self._transport = transport_config()

    @property
    def decoders(self) -> List[decoder_config]:
        return [d._copy() for d in self._decoders]

    @decoders.setter
    def decoders(self, value: Sequence[decoder_config]) -> None:
        if isinstance(value, (str, bytes)) or not hasattr(value, "__iter__"):
            raise TypeError("decoders must be a list of decoder_config")
        items = list(value)
        if not all(isinstance(d, decoder_config) for d in items):
            raise TypeError("decoders must be a list of decoder_config")
        self._decoders = [d._copy() for d in items]

    @property
    def transport(self) -> transport_config:
        return self._transport

    @transport.setter
    def transport(self, value: transport_config) -> None:
        if not isinstance(value, transport_config):
            raise TypeError("transport must be a transport_config")
        self._transport = value._copy()

    def validate_custom_args(self) -> None:
        """Check every decoder's ``decoder_custom_args`` against its schema."""
        for d in self._decoders:
            d.validate_custom_args()

    def to_yaml_str(self, column_wrap: int = 80) -> str:
        """The configuration as a YAML document (CUDA-Q's format)."""
        items: List[Tuple[str, Any]] = [("decoders", ("seq", [("map", _emit_decoder(d)) for d in self._decoders]))]
        if self._transport != transport_config():
            items.append(("transport", ("map", _emit_transport(self._transport))))
        return _yaml.document(items, _int_value(column_wrap, _I32[0], _I32[1], "column_wrap"))

    @staticmethod
    def from_yaml_str(yaml_str: str) -> "multi_decoder_config":
        """A configuration from a YAML document."""
        return _from_yaml(yaml_str, _read_multi, lambda multi: multi._decoders)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, multi_decoder_config) and self._decoders == other._decoders and self._transport == other._transport

    __hash__ = None  # type: ignore[assignment]

    def _copy(self) -> "multi_decoder_config":
        out = multi_decoder_config()
        out._decoders = [d._copy() for d in self._decoders]
        out._transport = self._transport._copy()
        return out

    def __repr__(self) -> str:
        return f"multi_decoder_config(decoders={self._decoders!r})"


def expand_dem_chunks(config: decoder_config) -> Optional[Any]:
    """Rewrite a chunk-form configuration into the equivalent flat form.

    Fills ``block_size``, ``syndrome_size``, ``H_sparse``, ``O_sparse`` and ``D_sparse`` on
    ``config`` in place from ``dem_chunks`` (which carries ``num_rounds`` inside the
    ``DemChunksSpec``), so that a chunk-form configuration can be handed to anything that
    understands only the flat form. Does nothing to a configuration that is already flat (one
    with an ``H_sparse``, as this leaves behind), so it is safe to call unconditionally.

    Returns the closed ``DetectorErrorModel`` the flat fields came from (its ``error_rates``
    the per-fault priors the phases declared), or None if the configuration was already flat or
    has no ``dem_chunks``. A spec that cannot be expanded or closed raises RuntimeError naming
    the decoder."""
    from ._extended_dem import dem_chunks_from_spec, dem_chunks_to_d_sparse, dem_close_all
    from ._pcm import pcm_to_sparse_vec

    if not isinstance(config, decoder_config):
        raise TypeError("expand_dem_chunks(): config must be a decoder_config")
    if config._dem_chunks is None or config._H:
        return None
    spec = config._dem_chunks
    from_seam, to_seam = spec.seam.from_seam, spec.seam.to_seam
    try:
        chunks = dem_chunks_from_spec(spec)
    except ValueError as e:
        raise RuntimeError(f"Cannot expand dem_chunks for decoder {config._id}: {e}") from None
    try:
        closed = dem_close_all(chunks, from_seam, to_seam)
        d_sparse = dem_chunks_to_d_sparse(chunks, from_seam, to_seam)
    except Exception as e:
        raise RuntimeError(f"Cannot close dem_chunks for decoder {config._id}: {e}") from None
    config._block_size = closed.num_error_mechanisms()
    config._syndrome_size = closed.num_detectors()
    config._H = pcm_to_sparse_vec(closed.detector_error_matrix)
    config._O = pcm_to_sparse_vec(closed.observables_flips_matrix)
    config._D = [c for row in d_sparse for c in (*row, -1)]
    return closed


# -- The configuration's checks (CUDA-Q runs them on reading and on writing) --------------------------

_FLAT_FIELDS = ("block_size", "syndrome_size", "H_sparse", "O_sparse", "D_sparse")


def _reject_derived(did: int, key: str, present: bool, source: str) -> None:
    # (The form named by ``source`` derives this field: accepting it would let a config disagree with itself.)
    if present:
        raise RuntimeError(f"{key} must not be set for decoder {did} because it is derived from {source}. Remove it, or describe the whole experiment with H_sparse instead of {source}.")


def _check(cfg: decoder_config, document_keys: Optional[List[str]]) -> None:
    """The checks of CUDA-Q's decoder_config mapping; ``document_keys`` when reading."""
    reading = document_keys is not None
    did = cfg._id
    # Chunk form when dem_chunks is set and H_sparse is empty: a non-empty H_sparse makes the config
    # flat, the matrix winning, which is also what expand_dem_chunks() leaves behind. DEM form when a
    # Stim model file is named.
    chunk_form = cfg._dem_chunks is not None and not cfg._H
    from_dem = bool(cfg._stim_dem_path)
    if reading and cfg._dem_chunks is not None and cfg._H:
        _warn(
            f"decoder {did} sets both H_sparse and dem_chunks. The flat H_sparse matrix is what decoders are built from, and dem_chunks is ignored, including its num_rounds. "
            "This is expected for a config already expanded by expand_dem_chunks(); otherwise drop H_sparse to decode from dem_chunks."
        )
    # (Each of dem_chunks and stim_dem_path describes the whole DEM on its own, with no precedence
    # between them to fall back on.)
    if chunk_form and from_dem:
        raise RuntimeError(f"dem_chunks and stim_dem_path must not both be set for decoder {did}; each describes the whole DEM on its own.")
    if chunk_form:
        rounds = cfg._dem_chunks.num_rounds
        if rounds is not None and rounds < 2:
            raise RuntimeError(f"dem_chunks.num_rounds must be at least 2 for decoder {did}; got {rounds}.")
        _reject_derived(did, "block_size", cfg._block_size != 0, "dem_chunks")
        _reject_derived(did, "syndrome_size", cfg._syndrome_size != 0, "dem_chunks")
        _reject_derived(did, "O_sparse", bool(cfg._O), "dem_chunks")
        _reject_derived(did, "D_sparse", bool(cfg._D), "dem_chunks")
    elif from_dem:
        _reject_derived(did, "block_size", cfg._block_size != 0, "stim_dem_path")
        _reject_derived(did, "syndrome_size", cfg._syndrome_size != 0, "stim_dem_path")
        _reject_derived(did, "H_sparse", bool(cfg._H), "stim_dem_path")
        _reject_derived(did, "O_sparse", bool(cfg._O), "stim_dem_path")
        if reading and not cfg._D:
            raise RuntimeError(f"D_sparse is required for decoder {did} alongside stim_dem_path, which describes the DEM but not the detectors this decoder is sent.")
    else:
        if reading:
            missing = [name for name in _FLAT_FIELDS if name not in document_keys]
            if missing:
                raise RuntimeError(
                    f"decoder {did} is missing required field(s): {', '.join(missing)}. A flat DEM names block_size, syndrome_size, H_sparse, O_sparse and D_sparse together; "
                    "use dem_chunks to describe the same DEM one round at a time, or stim_dem_path to name the model text itself, instead."
                )
        if not cfg._H and cfg._syndrome_size > 0:
            raise RuntimeError(f"H_sparse is required for decoder {did} unless dem_chunks is present, which describes the same DEM one round at a time instead.")
        if cfg._block_size == 0 and cfg._H:
            raise RuntimeError(f"block_size is required for decoder {did}")
        rows = cfg._H.count(-1)
        if rows != cfg._syndrome_size:
            raise RuntimeError(f"Number of rows in H_sparse vector is not equal to syndrome_size: {rows} != {cfg._syndrome_size}")
    for name, vec in (("H_sparse", cfg._H), ("O_sparse", cfg._O)):
        for value in vec:
            if value < -1 or (value >= 0 and value >= cfg._block_size):
                raise RuntimeError(f"Value in {name} vector is out of range: {value}")
    if cfg._D:
        rows = cfg._D.count(-1)
        if not from_dem and rows != cfg._syndrome_size:
            raise RuntimeError(f"Number of rows in D_sparse vector is not equal to syndrome_size: {rows} != {cfg._syndrome_size}")
        for a, b in zip(cfg._D, cfg._D[1:]):
            if a == -1 and b == -1:
                raise RuntimeError(f"D_sparse row is empty for decoder {did}")


# -- Writing ----------------------------------------------------------------------------------------


def _emit_args(args: Dict[str, _V], schema: _Schema) -> List[Tuple[str, Any]]:
    for key in args:
        if schema.find(key) is None:
            _warn(f"Key '{key}' is not in the '{schema.name}' parameter schema; it is omitted from the emitted YAML.")
    items: List[Tuple[str, Any]] = []
    for spec in schema.params:
        if spec.key not in args:
            continue
        k = spec.kind
        if k == "boolean":
            items.append((spec.key, ("scalar", "true" if _get(args, spec.key, k) else "false")))
        elif k in ("int32", "uint64"):
            items.append((spec.key, ("scalar", str(_get(args, spec.key, k)))))
        elif k == "f64":
            items.append((spec.key, ("scalar", _yaml.fmt_double(_get(args, spec.key, k)))))
        elif k == "string":
            items.append((spec.key, ("scalar", _yaml.quote(_get(args, spec.key, k)))))
        elif k == "f64_vec":
            items.append((spec.key, ("flow", [_yaml.fmt_double(v) for v in _get(args, spec.key, k)])))
        elif k == "f64_matrix":
            items.append((spec.key, ("seq", [("flow", [_yaml.fmt_double(v) for v in row]) for row in _get(args, spec.key, k)])))
        elif k == "subschema":
            nested = _find(spec.subschema)
            if nested is None:
                raise RuntimeError(f"No schema registered under '{spec.subschema}' (needed to emit '{spec.key}').")
            items.append((spec.key, ("map", _emit_args(_get(args, spec.key, k), nested))))
        else:
            value = _get(args, spec.discriminator, "string") if spec.discriminator in args else ""
            nested = _find(value) if value else None
            if nested is None:
                raise RuntimeError(f"'{spec.key}' is present but no parameter schema is registered for {spec.discriminator} '{value}'.")
            items.append((spec.key, ("map", _emit_args(_get(args, spec.key, k), nested))))
    return items


def _emit_seam_connection(from_name: str, to_name: str) -> List[Tuple[str, Any]]:
    return [("from", ("scalar", _yaml.quote(from_name))), ("to", ("scalar", _yaml.quote(to_name)))]


def _emit_dem_chunk_spec(spec: DemChunkSpec) -> List[Tuple[str, Any]]:
    flow = lambda vec: ("flow", [str(v) for v in vec])  # noqa: E731
    items: List[Tuple[str, Any]] = [("num_faults", ("scalar", str(spec._num_faults)))]
    # (An empty optional list is left out, as LLVM leaves it, here where it is not a map's first key.)
    if spec._H:
        items.append(("H_sparse", flow(spec._H)))
    if spec._seam_specs:
        # (CUDA-Q writes the names of these entries empty, having only the hash; here a name
        # registered for the hash, as the standard ones and those a YAML document has read are, is written.)
        entries = []
        for e in spec._seam_specs:
            inner: List[Tuple[str, Any]] = []
            if e._spec._H:
                inner.append(("H_sparse", flow(e._spec._H)))
            if e._spec._O:
                inner.append(("O_sparse", flow(e._spec._O)))
            entries.append(("map", [("name", ("scalar", _yaml.quote(_registered_name(e._id)))), ("spec", ("map", inner))]))
        items.append(("seam_specs", ("seq", entries)))
    if spec._O:
        items.append(("O_sparse", flow(spec._O)))
    items.append(("error_rates", ("flow", [_yaml.fmt_double(v) for v in spec._rates])))
    return items


def _emit_dem_chunks(spec: DemChunksSpec) -> List[Tuple[str, Any]]:
    # (CUDA-Q writes the names of the seam, the connections and the phases empty, having only their
    # hashes; here the names registered for the hashes, as the standard ones and those a YAML
    # document has read are, are written.)
    items: List[Tuple[str, Any]] = [
        ("seam", ("map", _emit_seam_connection(_registered_name(spec._seam.from_seam), _registered_name(spec._seam.to_seam)))),
        ("connections", ("seq", [("map", _emit_seam_connection(_registered_name(c.from_phase), _registered_name(c.to_phase))) for c in spec._connections])),
    ]
    if spec._num_rounds is not None:
        items.append(("num_rounds", ("scalar", str(spec._num_rounds))))
    items.append(("phases", ("seq", [("map", [("name", ("scalar", _yaml.quote(_registered_name(e.id)))), ("spec", ("map", _emit_dem_chunk_spec(e.spec)))]) for e in spec._phases])))
    return items


def _emit_decoder(cfg: decoder_config) -> List[Tuple[str, Any]]:
    flow = lambda vec: ("flow", [str(v) for v in vec])  # noqa: E731
    items: List[Tuple[str, Any]] = [("id", ("scalar", str(cfg._id))), ("type", ("scalar", _yaml.quote(cfg._type)))]
    if cfg._dispatch != decoder_dispatch.host:
        items.append(("dispatch", ("scalar", cfg._dispatch.name)))
    if cfg._cuda_device_id is not None:
        items.append(("cuda_device_id", ("scalar", str(cfg._cuda_device_id))))
    if not ((cfg._dem_chunks is not None or cfg._stim_dem_path) and not cfg._H):
        # (A flat configuration names all five, even empty ones; a chunk-form or DEM-sourced one
        # leaves the fields it derives out, which is what makes its document re-parse in the form
        # it was written in.)
        items += [("block_size", ("scalar", str(cfg._block_size))), ("syndrome_size", ("scalar", str(cfg._syndrome_size))), ("H_sparse", flow(cfg._H)), ("O_sparse", flow(cfg._O)), ("D_sparse", flow(cfg._D))]
    else:
        items += [(k, ("scalar", str(v))) for k, v in (("block_size", cfg._block_size), ("syndrome_size", cfg._syndrome_size)) if v]
        items += [(k, flow(v)) for k, v in (("H_sparse", cfg._H), ("O_sparse", cfg._O), ("D_sparse", cfg._D)) if v]
    if cfg._dem_chunks is not None:
        items.append(("dem_chunks", ("map", _emit_dem_chunks(cfg._dem_chunks))))
    if cfg._stim_dem_path:
        items.append(("stim_dem_path", ("scalar", _yaml.quote(cfg._stim_dem_path))))
    _check(cfg, None)
    if cfg._args:
        schema = _find(cfg._type)
        if schema is None:
            _warn(f"decoder_custom_args set for decoder type '{cfg._type}' but no parameter schema is registered under that name; the args are omitted from the emitted YAML.")
        else:
            items.append(("decoder_custom_args", ("map", _emit_args(cfg._args_for_decoder(), schema))))
    return items


def _emit_shape(s: transport_shape_override) -> List[Tuple[str, Any]]:
    items: List[Tuple[str, Any]] = []
    if s._provider:
        items.append(("provider", ("scalar", _yaml.quote(s._provider))))
    if s._args:
        items.append(("args", ("seq", [("scalar", _yaml.quote(a)) for a in s._args])))
    return items


def _emit_transport(t: transport_config) -> List[Tuple[str, Any]]:
    items = _emit_shape(t)  # (provider and args, as the shape's)
    if t._device_graph != transport_shape_override():
        items.append(("device_graph", ("map", _emit_shape(t._device_graph))))
    return items


# -- Reading (llvm::yaml::Input's rules over ._yaml's nodes) ------------------------------------------

_SKIP = object()  # (an error was recorded: the field keeps its default)
_DEFAULT = object()  # (an optional key is absent)
_NULLS = ("null", "Null", "NULL", "~")
_TRUE = ("y", "Y", "yes", "Yes", "YES", "on", "On", "ON", "true", "True", "TRUE")
_FALSE = ("n", "N", "no", "No", "NO", "off", "Off", "OFF", "false", "False", "FALSE")


class _In:
    """llvm::yaml::Input's error state: the first error is kept, and once one is, keys are no
    longer read."""

    def __init__(self) -> None:
        self.ec = False
        self.message = ""

    def err(self, node: Optional[_yaml.Node], message: str) -> None:
        if not self.ec:
            self.ec = True
            self.message = f"YAML:{node.line}:{node.col}: {message}" if node is not None else message

    def key(self, node: Optional[_yaml.Node], valid: List[str], name: str, required: bool) -> Any:
        if self.ec:
            return _SKIP
        if node is None:
            if required:
                self.err(None, "the document is empty")
                return _SKIP
            return _DEFAULT
        if node.kind != "map":
            if required or node.kind != "null":
                self.err(node, "not a mapping")
                return _SKIP
            return _DEFAULT
        valid.append(name)
        value = node.get(name)
        if value is None:
            if required:
                self.err(node, f"missing required key '{name}'")
                return _SKIP
            return _DEFAULT
        return value

    def end(self, node: Optional[_yaml.Node], valid: List[str]) -> None:
        if self.ec or node is None or node.kind != "map":
            return
        for k, knode, _ in node.value:
            if k not in valid:
                self.err(knode, f"unknown key '{k}'")
                return

    def scalar(self, node: _yaml.Node) -> Optional[str]:
        if node.kind != "scalar":
            self.err(node, "unexpected scalar")
            return None
        return node.value

    def integer(self, node: _yaml.Node, lo: int, hi: int) -> Optional[int]:
        s = self.scalar(node)
        if s is None:
            return None
        signed = lo < 0
        v = _parse_integer(s, signed)
        if v is None or not (_I64[0] <= v <= _I64[1] if signed else 0 <= v < _U64):
            self.err(node, "invalid number")
            return None
        if not lo <= v <= hi:
            self.err(node, "out of range number")
            return None
        return v

    def double(self, node: _yaml.Node) -> Optional[float]:
        s = self.scalar(node)
        if s is None:
            return None
        v = _parse_double(s)
        if v is None:
            self.err(node, "invalid floating point number")
        return v

    def boolean(self, node: _yaml.Node) -> Optional[bool]:
        s = self.scalar(node)
        if s is None:
            return None
        if s in _TRUE:
            return True
        if s in _FALSE:
            return False
        self.err(node, "invalid boolean")
        return None

    def seq(self, node: _yaml.Node) -> List[_yaml.Node]:
        if node.kind == "seq":
            return node.value
        if node.kind == "null" or (node.kind == "scalar" and node.value in _NULLS):
            return []
        self.err(node, "not a sequence")
        return []

    def int_seq(self, node: _yaml.Node) -> List[int]:
        out = []
        for item in self.seq(node):
            if self.ec:
                break
            v = self.integer(item, _I64[0], _I64[1])
            out.append(0 if v is None else v)
        return out

    def double_seq(self, node: _yaml.Node) -> List[float]:
        out = []
        for item in self.seq(node):
            if self.ec:
                break
            v = self.double(item)
            out.append(0.0 if v is None else v)
        return out


def _parse_integer(s: str, signed: bool) -> Optional[int]:
    # llvm::getAsSignedInteger / getAsUnsignedInteger with radix 0 (auto-sensed).
    neg = signed and s.startswith("-")
    if neg:
        s = s[1:]
    low = s.lower()
    radix = 10
    if low.startswith("0x"):
        s, radix = s[2:], 16
    elif low.startswith("0b"):
        s, radix = s[2:], 2
    elif s.startswith("0o"):
        s, radix = s[2:], 8
    elif len(s) > 1 and s[0] == "0" and s[1] in "0123456789":
        s, radix = s[1:], 8
    if not s or any(c not in "0123456789abcdefghijklmnopqrstuvwxyz"[:radix] for c in s.lower()):
        return None
    v = int(s, radix)
    return -v if neg else v


def _parse_double(s: str) -> Optional[float]:
    # strtod over the whole scalar (llvm::to_float).
    t = s.lstrip(" \t\n\v\f\r")
    if not t:
        return 0.0
    sign = -1.0 if t[0] == "-" else 1.0
    body = t[1:] if t[0] in "+-" else t
    low = body.lower()
    if low in ("inf", "infinity"):
        return sign * float("inf")
    if low == "nan" or (low.startswith("nan(") and low.endswith(")") and all(c.isalnum() or c == "_" for c in low[4:-1])):
        return float("nan")
    if low.startswith("0x"):
        h = low[2:]
        mant, _, exp = h.partition("p")
        if not mant or mant == "." or any(c not in "0123456789abcdef." for c in mant) or mant.count(".") > 1:
            return None
        if "p" in h and not (exp.lstrip("+-").isdigit() and exp.lstrip("+-").isascii()):
            return None
        try:
            return sign * float.fromhex(h if "p" in h else h + "p0")
        except ValueError:
            return None
    mant, _, exp = low.partition("e")
    if not mant or mant == "." or any(c not in "0123456789." for c in mant) or mant.count(".") > 1:
        return None
    if "e" in low and not (exp.lstrip("+-") and all(c in "0123456789" for c in exp.lstrip("+-")) and exp.count("+") + exp.count("-") <= 1 and exp.lstrip("0123456789") == ""):
        if not (exp[:1] in "+-" and exp[1:] and all(c in "0123456789" for c in exp[1:])):
            return None
    return sign * float(body)


def _read_custom(io: _In, node: _yaml.Node, schema: _Schema, out: Dict[str, _V]) -> None:
    """CUDA-Q's CustomMappingTraits<schema_binding> input: each key through the schema."""
    if node.kind == "null":
        return
    if node.kind != "map":
        io.err(node, "not a mapping")
        return
    for key in node.keys():
        spec = schema.find(key)
        if spec is None:
            raise RuntimeError(f"Unknown key '{key}' in '{schema.name}' parameters.")
        k = spec.kind
        if k in ("subschema", "discriminated"):
            if k == "subschema":
                nested = _find(spec.subschema)
                if nested is None:
                    raise RuntimeError(f"No schema registered under '{spec.subschema}' (needed to parse '{key}').")
            else:
                value = ""
                d = io.key(node, [], spec.discriminator, False)
                if isinstance(d, _yaml.Node):
                    value = io.scalar(d) or ""
                if not value:
                    raise RuntimeError(f"'{key}' is present but '{spec.discriminator}' is not set.")
                nested = _find(value)
                if nested is None:
                    raise RuntimeError(f"'{key}' does not support {spec.discriminator} '{value}': no parameter schema is registered under that name.")
            sub: Dict[str, _V] = {}
            v = io.key(node, [], key, True)
            if isinstance(v, _yaml.Node):
                _read_custom(io, v, nested, sub)
            out[key] = _V("map", sub)
            continue
        v = io.key(node, [], key, True)
        value: Any = {"boolean": False, "int32": 0, "uint64": 0, "f64": 0.0, "string": "", "f64_vec": [], "f64_matrix": []}[k]
        if isinstance(v, _yaml.Node):
            got: Any = None
            if k == "boolean":
                got = io.boolean(v)
            elif k == "int32":
                got = io.integer(v, _I32[0], _I32[1])
            elif k == "uint64":
                got = io.integer(v, 0, _U64 - 1)
            elif k == "f64":
                got = io.double(v)
            elif k == "string":
                got = io.scalar(v)
            elif k == "f64_vec":
                got = io.double_seq(v)
            else:
                got = []
                for row in io.seq(v):
                    if io.ec:
                        break
                    got.append(io.double_seq(row))
            if got is not None:
                value = got
        out[key] = _V(_STORED[k], value)


def _is_none(node: _yaml.Node) -> bool:
    # (llvm::yaml's optional keys read the plain scalar "<none>" as no value.)
    return node.kind == "scalar" and node.style == "plain" and node.value.rstrip(" ") == "<none>"


def _read_name(io: _In, node: Optional[_yaml.Node], valid: List[str], key: str) -> str:
    v = io.key(node, valid, key, True)
    return (io.scalar(v) or "") if isinstance(v, _yaml.Node) else ""


def _read_connection(io: _In, node: _yaml.Node) -> Tuple[SeamId, SeamId]:
    """CUDA-Q's seam_connection and phase_connection mappings: ``{from: name, to: name}``."""
    valid: List[str] = []
    from_name = _read_name(io, node, valid, "from")
    to_name = _read_name(io, node, valid, "to")
    io.end(node, valid)
    if io.ec:
        return SeamId(), SeamId()
    ids = SeamId(from_name), SeamId(to_name)
    _register_name(ids[0], from_name)
    _register_name(ids[1], to_name)
    return ids


def _read_seam_spec(io: _In, node: _yaml.Node) -> DemSeamSpec:
    spec = DemSeamSpec()
    valid: List[str] = []
    v = io.key(node, valid, "H_sparse", False)
    if isinstance(v, _yaml.Node):
        spec._H = io.int_seq(v)
    v = io.key(node, valid, "O_sparse", False)
    if isinstance(v, _yaml.Node):
        spec._O = io.int_seq(v)
    io.end(node, valid)
    return spec


def _read_named(io: _In, node: _yaml.Node, read_spec: Callable[[_In, _yaml.Node], Any]) -> Tuple[SeamId, Any]:
    """CUDA-Q's seam_spec_entry and phase_spec_entry mappings: ``{name: string, spec: ...}``."""
    valid: List[str] = []
    name = _read_name(io, node, valid, "name")
    spec = None
    v = io.key(node, valid, "spec", True)
    if isinstance(v, _yaml.Node):
        spec = read_spec(io, v)
    io.end(node, valid)
    if io.ec:
        return SeamId(), spec
    sid = SeamId(name)
    _register_name(sid, name)
    return sid, spec


def _read_dem_chunk_spec(io: _In, node: _yaml.Node) -> DemChunkSpec:
    spec = DemChunkSpec()
    valid: List[str] = []
    v = io.key(node, valid, "num_faults", True)
    if isinstance(v, _yaml.Node):
        x = io.integer(v, 0, _U64 - 1)
        spec._num_faults = 0 if x is None else x
    v = io.key(node, valid, "H_sparse", False)
    if isinstance(v, _yaml.Node):
        spec._H = io.int_seq(v)
    v = io.key(node, valid, "seam_specs", False)
    if isinstance(v, _yaml.Node):
        for item in io.seq(v):
            if io.ec:
                break
            entry = SeamSpecEntry()
            entry._id, parsed = _read_named(io, item, _read_seam_spec)
            if parsed is not None:
                entry._spec = parsed
            spec._seam_specs.append(entry)
    v = io.key(node, valid, "O_sparse", False)
    if isinstance(v, _yaml.Node):
        spec._O = io.int_seq(v)
    v = io.key(node, valid, "error_rates", True)
    if isinstance(v, _yaml.Node):
        spec._rates = io.double_seq(v)
    io.end(node, valid)
    return spec


def _read_dem_chunks(io: _In, node: _yaml.Node) -> Optional[DemChunksSpec]:
    """CUDA-Q's dem_chunks_spec mapping:

        seam: {from: next_round, to: prev_round}
        connections: [{from: init, to: bulk}, {from: bulk, to: bulk}, {from: bulk, to: final}]
        num_rounds: 5
        phases: [{name: init, spec: {num_faults: 3, H_sparse: [...], O_sparse: [...], error_rates: [...]}}, ...]

    ``seam``, ``connections`` and ``phases`` are required, ``num_rounds`` is not."""
    if _is_none(node):
        return None
    spec = DemChunksSpec()
    valid: List[str] = []
    v = io.key(node, valid, "seam", True)
    if isinstance(v, _yaml.Node):
        spec._seam._from, spec._seam._to = _read_connection(io, v)
    v = io.key(node, valid, "connections", True)
    if isinstance(v, _yaml.Node):
        for item in io.seq(v):
            if io.ec:
                break
            connection = PhaseConnection()
            connection._from, connection._to = _read_connection(io, item)
            spec._connections.append(connection)
    v = io.key(node, valid, "num_rounds", False)
    if isinstance(v, _yaml.Node) and not _is_none(v):
        spec._num_rounds = io.integer(v, 0, _U64 - 1)
    v = io.key(node, valid, "phases", True)
    if isinstance(v, _yaml.Node):
        for item in io.seq(v):
            if io.ec:
                break
            entry = PhaseSpecEntry()
            entry._id, parsed = _read_named(io, item, _read_dem_chunk_spec)
            if parsed is not None:
                entry._spec = parsed
            spec._phases.append(entry)
    io.end(node, valid)
    return spec


def _read_decoder(io: _In, node: Optional[_yaml.Node]) -> decoder_config:
    cfg = decoder_config()
    valid: List[str] = []
    document_keys = node.keys() if node is not None and node.kind == "map" else []
    v = io.key(node, valid, "id", True)
    if isinstance(v, _yaml.Node):
        x = io.integer(v, _I64[0], _I64[1])
        cfg._id = cfg._id if x is None else x
    v = io.key(node, valid, "type", True)
    if isinstance(v, _yaml.Node):
        cfg._type = io.scalar(v) or ""
    v = io.key(node, valid, "dispatch", False)
    if isinstance(v, _yaml.Node):
        if v.kind == "scalar" and v.value in decoder_dispatch.__members__:
            cfg._dispatch = decoder_dispatch[v.value]
        else:
            io.err(v, "unknown enumerated scalar")
    v = io.key(node, valid, "cuda_device_id", False)
    if isinstance(v, _yaml.Node):
        if v.kind == "scalar" and v.value.rstrip(" ") == "<none>":
            cfg._cuda_device_id = None
        else:
            cfg._cuda_device_id = io.integer(v, _I32[0], _I32[1])
    for name in ("block_size", "syndrome_size"):
        v = io.key(node, valid, name, False)
        if isinstance(v, _yaml.Node):
            x = io.integer(v, 0, _U64 - 1)
            setattr(cfg, "_" + name, 0 if x is None else x)
    for name, attr in (("H_sparse", "_H"), ("O_sparse", "_O"), ("D_sparse", "_D")):
        v = io.key(node, valid, name, False)
        if isinstance(v, _yaml.Node):
            setattr(cfg, attr, io.int_seq(v))
    v = io.key(node, valid, "dem_chunks", False)
    if isinstance(v, _yaml.Node):
        cfg._dem_chunks = _read_dem_chunks(io, v)
    v = io.key(node, valid, "stim_dem_path", False)
    if isinstance(v, _yaml.Node):
        cfg._stim_dem_path = io.scalar(v) or ""
    if io.ec:
        return cfg
    _check(cfg, document_keys)
    schema = _find(cfg._type)
    if schema is not None:
        # (With no schema the section stays unread, so it is refused as an unknown key.)
        args: Dict[str, _V] = {}
        v = io.key(node, valid, "decoder_custom_args", False)
        if isinstance(v, _yaml.Node):
            _read_custom(io, v, schema, args)
        if "decoder_custom_args" in document_keys:
            _materialize(schema, args)
            _validate(schema, args, f"decoder_custom_args ({cfg._type})")
        cfg._args = args
    io.end(node, valid)
    return cfg


def _read_shape(io: _In, node: _yaml.Node, out: transport_shape_override, more: Callable[[List[str]], None] = lambda valid: None) -> None:
    valid: List[str] = []
    v = io.key(node, valid, "provider", False)
    if isinstance(v, _yaml.Node):
        out._provider = io.scalar(v) or ""
    v = io.key(node, valid, "args", False)
    if isinstance(v, _yaml.Node):
        out._args = []
        for item in io.seq(v):
            if io.ec:
                break
            out._args.append(io.scalar(item) or "")
    more(valid)
    io.end(node, valid)


def _read_transport(io: _In, node: _yaml.Node, out: transport_config) -> None:
    def device_graph(valid: List[str]) -> None:
        v = io.key(node, valid, "device_graph", False)
        if isinstance(v, _yaml.Node):
            _read_shape(io, v, out._device_graph)

    _read_shape(io, node, out, device_graph)  # type: ignore[arg-type]


def _read_multi(io: _In, node: Optional[_yaml.Node]) -> multi_decoder_config:
    cfg = multi_decoder_config()
    valid: List[str] = []
    v = io.key(node, valid, "decoders", True)
    if isinstance(v, _yaml.Node):
        for item in io.seq(v):
            if io.ec:
                break
            cfg._decoders.append(_read_decoder(io, item))
    v = io.key(node, valid, "transport", False)
    if isinstance(v, _yaml.Node):
        _read_transport(io, v, cfg._transport)
    io.end(node, valid)
    return cfg


def _validate_dem_chunks(cfg: decoder_config) -> None:
    """The cross-phase checks of ``dem_chunks``, run after a successful parse and presented as the
    one error type every configuration error out of ``from_yaml_str`` has."""
    if cfg._dem_chunks is None:
        return
    try:
        cfg._dem_chunks.validate()
    except ValueError as e:
        raise RuntimeError(f"Invalid dem_chunks for decoder {cfg._id}: {e}") from None


def _from_yaml(yaml_str: str, read: Callable[[_In, Optional[_yaml.Node]], Any], decoders: Callable[[Any], List[decoder_config]]) -> Any:
    if not isinstance(yaml_str, str):
        raise TypeError(f"from_yaml_str(): yaml_str must be a str, not {type(yaml_str).__name__}")
    io = _In()
    try:
        root = _yaml.parse(yaml_str)
    except _yaml.YamlError as e:
        raise RuntimeError(f"Invalid decoder configuration YAML: Invalid argument (YAML:{e.line}:{e.col}: {e.message})") from None
    out = read(io, root)
    if io.ec:
        raise RuntimeError(f"Invalid decoder configuration YAML: Invalid argument ({io.message})")
    for cfg in decoders(out):
        _validate_dem_chunks(cfg)
    return out


# -- JSON Schema export ------------------------------------------------------------------------------


def _params_ref(name: str) -> str:
    return "#/$defs/decoder_params/" + name.replace("~", "~0").replace("/", "~1")


def _json_param(spec: _Spec) -> dict:
    k = spec.kind
    if k == "boolean":
        return {"type": "boolean"}
    if k == "int32":
        return {"type": "integer"}
    if k == "uint64":
        return {"type": "integer", "minimum": 0}
    if k == "f64":
        return {"type": "number"}
    if k == "string":
        return {"type": "string"}
    if k == "f64_vec":
        return {"type": "array", "items": {"type": "number"}}
    if k == "f64_matrix":
        return {"type": "array", "items": {"type": "array", "items": {"type": "number"}}}
    if k == "subschema":
        return {"$ref": _params_ref(spec.subschema)}
    return {"type": "object"}


def _json_params(schema: _Schema, names: List[str]) -> dict:
    properties: dict = {}
    required: list = []
    all_of: list = []
    for spec in schema.params:
        properties[spec.key] = _json_param(spec)
        if spec.required:
            required.append(spec.key)
        if spec.kind == "discriminated":
            all_of.append({"if": {"required": [spec.key]}, "then": {"required": [spec.discriminator], "properties": {spec.discriminator: {"enum": list(names)}}}})
            for name in names:
                all_of.append({"if": {"properties": {spec.discriminator: {"const": name}}, "required": [spec.discriminator, spec.key]}, "then": {"properties": {spec.key: {"$ref": _params_ref(name)}}}})
    out: dict = {"type": "object", "additionalProperties": False, "properties": properties}
    if required:
        out["required"] = required
    if all_of:
        out["allOf"] = all_of
    return out


def decoder_config_json_schema() -> str:
    """A JSON Schema (draft 2020-12), as text, that validates ``multi_decoder_config`` YAML
    documents, generated from the registered parameter schemas. (Schemas' own checks, such as
    the sliding window's, are not expressible in it.)"""
    names = registered_decoder_schemas()
    sparse = {"$ref": "#/$defs/sparse_matrix"}
    dispatch = [{"if": {"properties": {"type": {"const": name}}, "required": ["type"]}, "then": {"properties": {"decoder_custom_args": {"$ref": _params_ref(name)}}}} for name in names]
    dispatch.append({"if": {"properties": {"type": {"not": {"enum": list(names)}}}, "required": ["type"]}, "then": {"properties": {"decoder_custom_args": {"maxProperties": 0}}}})
    strings = {"type": "array", "items": {"type": "string"}}
    defs = {
        "sparse_matrix": {"type": "array", "items": {"type": "integer", "minimum": -1}},
        "transport_shape_override": {"type": "object", "properties": {"provider": {"type": "string"}, "args": strings}, "additionalProperties": False},
        "transport_config": {"type": "object", "properties": {"provider": {"type": "string"}, "args": strings, "device_graph": {"$ref": "#/$defs/transport_shape_override"}}, "additionalProperties": False},
        "dem_chunk": {
            "type": "object",
            "properties": {"num_faults": {"type": "integer", "minimum": 1}, "H_sparse": sparse, "O_sparse": sparse, "error_rates": {"type": "array", "items": {"type": "number", "minimum": 0, "maximum": 1}}},
            "required": ["num_faults", "error_rates"],
            "additionalProperties": False,
        },
        "dem_chunks": {
            "type": "object",
            "properties": {"seam": {"type": "object"}, "connections": {"type": "array"}, "num_rounds": {"type": "integer", "minimum": 2}, "phases": {"type": "array"}},
            "required": ["seam", "connections", "phases"],
            "additionalProperties": False,
        },
        "decoder_config": {
            "type": "object",
            "properties": {
                "id": {"type": "integer"},
                "type": {"type": "string"},
                "dispatch": {"enum": ["host", "device_graph"]},
                "cuda_device_id": {"type": "integer", "minimum": 0},
                "block_size": {"type": "integer", "minimum": 0},
                "syndrome_size": {"type": "integer", "minimum": 0},
                "stim_dem_path": {"type": "string"},
                "H_sparse": sparse,
                "O_sparse": sparse,
                "D_sparse": sparse,
                "dem_chunks": {"$ref": "#/$defs/dem_chunks"},
                "decoder_custom_args": {"type": "object"},
            },
            "required": ["id", "type"],
            "anyOf": [{"required": ["H_sparse", "block_size", "syndrome_size", "O_sparse", "D_sparse"]}, {"required": ["dem_chunks"]}, {"required": ["stim_dem_path", "D_sparse"]}],
            "additionalProperties": False,
            "allOf": dispatch,
        },
        "decoder_params": {name: _json_params(_SCHEMAS[name], names) for name in names},
    }
    root = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "CUDA-Q QEC realtime decoding configuration",
        "description": (
            "Validates multi_decoder_config YAML documents. Generated from the decoder parameter schemas registered in this installation, so it reflects the "
            "decoder plugins loaded at generation time. Per-schema validate hooks (arbitrary cross-field checks) are not representable in JSON Schema; a document "
            "that passes may still be rejected when parsed."
        ),
        "type": "object",
        "properties": {"decoders": {"type": "array", "items": {"$ref": "#/$defs/decoder_config"}}, "transport": {"$ref": "#/$defs/transport_config"}},
        "required": ["decoders"],
        "additionalProperties": False,
        "$defs": defs,
    }
    return json.dumps(root, indent=2, sort_keys=True)


__all__ = [
    "decoder_config",
    "decoder_config_json_schema",
    "decoder_dispatch",
    "decoder_param_schema",
    "expand_dem_chunks",
    "multi_decoder_config",
    "registered_decoder_schemas",
    "transport_config",
    "transport_shape_override",
]

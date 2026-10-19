# Ported from CUDA-Q QEC (NVIDIA/cudaqx 0.8.1, libs/qec/python/cudaq_qec/_compat.py; Copyright (c)
# 2024 - 2026 NVIDIA Corporation & Affiliates, Apache License 2.0; see THIRD_PARTY_NOTICES.md),
# modified for stabilizer_qec.
"""CUDA-Q QEC's deprecated typed decoder configurations (``nv_qldpc_decoder_config`` and the
rest), kept as CUDA-Q keeps them: each holds the parameters set on it, reads an unset one as
None, and assigned to ``decoder_config.decoder_custom_args`` converts through its decoder's
parameter schema as a dict would. Constructing one warns (DeprecationWarning); new code assigns
a dict."""

import warnings

__all__ = [
    "srelay_bp_config",
    "nv_qldpc_decoder_config",
    "single_error_lut_config",
    "multi_error_lut_config",
    "pymatching_config",
    "chromobius_config",
    "trt_decoder_config",
    "sliding_window_config",
]


class _deprecated_typed_config:
    _schema_name = None  # (the parameter schema it converts through, whatever decoder_config.type is)
    _fields = ()

    def __init__(self, map=None):
        warnings.warn(
            f"{type(self).__name__} is deprecated; assign a plain dict to decoder_config.decoder_custom_args instead (see cudaq_qec.decoder_param_schema for the accepted keys).",
            DeprecationWarning,
            stacklevel=2,
        )
        object.__setattr__(self, "_args", {})
        if map is not None:
            self._load_map(dict(map))

    def _load_map(self, map):
        # (Keys it does not know are ignored, as the typed from_heterogeneous_map ignored them.)
        for key in self._fields:
            if key in map:
                self._args[key] = map[key]

    @classmethod
    def from_heterogeneous_map(cls, map):
        return cls(map)

    def to_heterogeneous_map(self):
        """The parameters set, as a dict."""
        return {key: _plain(self._args[key]) for key in self._fields if key in self._args}

    @classmethod
    def _param_kinds(cls):
        cached = cls.__dict__.get("_kinds")
        if not cached:
            cached = {}
            try:
                from ._config import decoder_param_schema

                cached = {entry["key"]: entry["kind"] for entry in decoder_param_schema(cls._schema_name) or []}
            except Exception:
                pass
            cls._kinds = cached
        return cached

    def __setattr__(self, name, value):
        if name not in self._fields:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        if value is None:
            self._args.pop(name, None)
            return
        kind = self._param_kinds().get(name)
        if kind is not None:
            _check_kind(type(self).__name__, name, kind, value)
        self._args[name] = value

    def __getattr__(self, name):
        if name in type(self)._fields:
            return object.__getattribute__(self, "_args").get(name)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __eq__(self, other):
        return type(other) is type(self) and self._args == other._args

    __hash__ = None

    def __repr__(self):
        return f"{type(self).__name__}({', '.join(f'{k}={self._args[k]!r}' for k in self._fields if k in self._args)})"


def _plain(value):
    return value.to_heterogeneous_map() if isinstance(value, _deprecated_typed_config) else value


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _check_kind(cls_name, name, kind, value):
    """The TypeErrors the typed setters raised for plainly wrong values (array-likes pass)."""

    def fail(expected):
        raise TypeError(f"{cls_name}.{name} expects {expected}, got {type(value).__name__}")

    if kind == "bool":
        if not isinstance(value, bool):
            fail("a bool")
    elif kind == "int32":
        if not isinstance(value, int) or isinstance(value, bool):
            fail("an int")
    elif kind == "uint64":
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            fail("a non-negative int")
    elif kind == "float64":
        if not _number(value):
            fail("a float")
    elif kind == "string":
        if not isinstance(value, str):
            fail("a str")
    elif kind == "float64_vec":
        if _number(value) or isinstance(value, (str, bytes, dict)) or (isinstance(value, (list, tuple)) and not all(_number(x) for x in value)):
            fail("a list of floats")
    elif kind == "float64_matrix":
        if _number(value) or isinstance(value, (str, bytes, dict)) or (isinstance(value, (list, tuple)) and not all(isinstance(row, (list, tuple)) and all(_number(x) for x in row) for row in value)):
            fail("a list of lists of floats")
    elif kind in ("subschema", "discriminated"):
        if not isinstance(value, (_deprecated_typed_config, dict)):
            fail("a config object or dict")


class srelay_bp_config(_deprecated_typed_config):
    """Deprecated: the Relay-BP section of nv-qldpc-decoder's parameters."""

    _schema_name = "srelay_bp"
    _fields = ("pre_iter", "num_sets", "stopping_criterion", "stop_nconv")


class nv_qldpc_decoder_config(_deprecated_typed_config):
    """Deprecated: nv-qldpc-decoder's parameters."""

    _schema_name = "nv-qldpc-decoder"
    _fields = (
        "use_sparsity", "error_rate", "error_rate_vec", "max_iterations", "n_threads", "use_osd", "osd_method", "osd_order", "bp_batch_size",
        "osd_batch_size", "iter_per_check", "clip_value", "bp_method", "scale_factor", "proc_float", "gamma0", "gamma_dist", "explicit_gammas",
        "srelay_config", "bp_seed", "composition",
    )

    def _load_map(self, map):
        srelay = map.pop("srelay_config", None)
        super()._load_map(map)
        if srelay is not None:
            self._args["srelay_config"] = srelay_bp_config(srelay)


class single_error_lut_config(_deprecated_typed_config):
    """Deprecated: single_error_lut's parameters (it has none)."""

    _schema_name = "single_error_lut"
    _fields = ()


class multi_error_lut_config(_deprecated_typed_config):
    """Deprecated: multi_error_lut's parameters."""

    _schema_name = "multi_error_lut"
    _fields = ("lut_error_depth",)


class pymatching_config(_deprecated_typed_config):
    """Deprecated: pymatching's parameters."""

    _schema_name = "pymatching"
    _fields = ("error_rate_vec", "merge_strategy")


class chromobius_config(_deprecated_typed_config):
    """Deprecated: chromobius's parameters."""

    _schema_name = "chromobius"
    _fields = ("drop_mobius_errors_involving_remnant_errors", "ignore_decomposition_failures", "include_coords_in_mobius_dem", "return_weight", "write_mobius_match_to_stderr")


class trt_decoder_config(_deprecated_typed_config):
    """Deprecated: trt_decoder's parameters (this package has no trt_decoder, nor its schema:
    the values are kept as given)."""

    _schema_name = "trt_decoder"
    _fields = ("onnx_load_path", "engine_load_path", "engine_save_path", "precision", "memory_workspace", "batch_size", "use_cuda_graph", "global_decoder", "global_decoder_params")
    _global_classes = {"pymatching": pymatching_config, "chromobius": chromobius_config}

    def _load_map(self, map):
        params = map.pop("global_decoder_params", None)
        super()._load_map(map)
        cls = self._global_classes.get(self._args.get("global_decoder"))
        if params is not None:
            if cls is None:
                raise RuntimeError(f"global_decoder_params does not support global_decoder '{self._args.get('global_decoder')}'")
            self._args["global_decoder_params"] = cls(params)
        elif cls is not None:
            self._args["global_decoder_params"] = cls()

    def to_heterogeneous_map(self):
        out = super().to_heterogeneous_map()
        if "global_decoder_params" not in out and self._args.get("global_decoder") in self._global_classes:
            out["global_decoder_params"] = {}
        return out


class sliding_window_config(_deprecated_typed_config):
    """Deprecated: sliding_window's parameters (an inner decoder's in the field of its kind)."""

    _schema_name = "sliding_window"
    _fields = (
        "window_size", "step_size", "num_syndromes_per_round", "straddle_start_round", "straddle_end_round", "error_rate_vec", "inner_decoder_name",
        "single_error_lut_params", "multi_error_lut_params", "nv_qldpc_decoder_params",
    )
    # (One field per inner decoder; the first set one becomes inner_decoder_params.)
    _inner = (
        ("single_error_lut_params", "single_error_lut", single_error_lut_config),
        ("multi_error_lut_params", "multi_error_lut", multi_error_lut_config),
        ("nv_qldpc_decoder_params", "nv-qldpc-decoder", nv_qldpc_decoder_config),
    )

    def _load_map(self, map):
        inner = map.pop("inner_decoder_params", None)
        super()._load_map(map)
        if inner is not None:
            for field, name, cls in self._inner:
                if self._args.get("inner_decoder_name") == name:
                    self._args[field] = cls(inner)
                    break

    def to_heterogeneous_map(self):
        inner_fields = [f for f, _, _ in self._inner]
        out = {key: _plain(self._args[key]) for key in self._fields if key in self._args and key not in inner_fields}
        for field in inner_fields:
            if field in self._args:
                inner = _plain(self._args[field])
                if inner:
                    out["inner_decoder_params"] = inner
                break
        return out

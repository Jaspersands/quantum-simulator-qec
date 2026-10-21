# Ported from CUDA-Q QEC (NVIDIA/cudaqx 0.8.1, libs/qec: lib/realtime/realtime_decoding.cpp,
# lib/realtime/config.cpp, the realtime parts of lib/decoder.cpp and include/cudaq/qec/decoder.h,
# python/bindings/py_decoding.cpp; Copyright (c) 2024 - 2026 NVIDIA Corporation & Affiliates,
# Apache License 2.0; see THIRD_PARTY_NOTICES.md), modified for stabilizer_qec: the decoders are
# this package's, and the device-side calls run on the host.
"""CUDA-Q QEC's real-time decoding host library: ``configure_decoders`` (from a
``multi_decoder_config``, a YAML file or string) builds each configured decoder, and
``enqueue_syndromes`` / ``get_corrections`` / ``reset_decoder`` drive one as a CUDA-Q kernel
does, a round of measurements at a time.

Each decoder maps the raw measurements it is sent to its detectors through ``D_sparse`` (a
detector the XOR of its row's measurements). Once a shot's measurements have all arrived it
decodes them and folds the predicted observable flips (through ``O_sparse``, or the decoder's own
observables) into its corrections, which accumulate across shots until ``get_corrections`` is
asked to reset them or the decoder is reset. A ``sliding_window`` decoder instead decodes a
detector layer per round, its result coming with the last.

Status codes and messages are CUDA-Q's. CUDA-Q warns through its logger; here a warning is a
``RuntimeWarning``. The ``dispatch`` and ``transport`` settings are a decoding server's and do
not change how these host decoders run (as in CUDA-Q's host library)."""

from __future__ import annotations

import os
import warnings
from typing import Any, List, Optional

import numpy as np

from . import _config
from ._config import decoder_config, multi_decoder_config


class _DecoderNotFound(KeyError, ValueError):
    """No decoder is configured under the id (a ValueError, as CUDA-Q raises; a KeyError, as
    this package's queue form of the API raised)."""

    def __str__(self) -> str:
        return str(self.args[0]) if self.args else ""


def _warn(message: str) -> None:
    warnings.warn(message, RuntimeWarning, stacklevel=3)


def _sparse_rows(vec: List[int]) -> List[List[int]]:
    """CUDA-Q's set_sparse_from_vec: the rows of a sparse vector, empty rows dropped."""
    rows: List[List[int]] = []
    first = True
    for v in vec:
        if v < 0:
            first = True
        else:
            if first:
                rows.append([])
                first = False
            rows[-1].append(int(v) % 2**32)
    return rows


class _RealtimeDecoder:
    """A configured decoder and its real-time state (CUDA-Q's decoder::rt_impl)."""

    def __init__(self, decoder: Any, decoder_id: int) -> None:
        from ._decoders import _SlidingWindow

        self.decoder = decoder
        self.decoder_id = decoder_id
        self.sliding = isinstance(decoder, _SlidingWindow)
        # A decoder predicting observables (PyMatching given O, Chromobius, ...) installs the
        # mapping its results are expressed in; the others return an error frame.
        self.to_obs = getattr(decoder, "_output", "errors") == "observables"
        inputs = getattr(decoder, "_inputs", None)
        self.O: List[List[int]] = [list(r) for r in (inputs.O or [])] if self.to_obs and inputs is not None else []
        self.D: List[List[int]] = []
        self.corrections = np.zeros(len(self.O), dtype=np.uint8)
        self.buffer = np.zeros(0, dtype=np.uint8)
        self.index = 0
        self.round = 0
        self.layer = 0
        self.has_first_round = False
        self.offsets: List[int] = []

    def num_observables(self) -> int:
        return len(self.O)

    def set_O_sparse(self, vec: List[int]) -> None:
        rows = _sparse_rows(vec)
        block = self.decoder.get_block_size()
        for r, row in enumerate(rows):
            for c in row:
                if c >= block:
                    raise ValueError(f"O_sparse column index {c} out of range [0, {block}) at row {r}")
        self.O = rows
        self.corrections = np.zeros(len(rows), dtype=np.uint8)

    def set_D_sparse(self, vec: List[int]) -> None:
        rows = _sparse_rows(vec)
        if self.sliding:
            self.has_first_round = len(rows) > 0 and len(rows[0]) == 1
            layers = self.decoder.get_num_detector_layers()
            self.offsets = [self.decoder.get_layer_offset(r) for r in range(layers + 1)]
        elif len(rows) != self.decoder.get_syndrome_size():
            raise ValueError(f"D_sparse row count ({len(rows)}) must match syndrome_size ({self.decoder.get_syndrome_size()})")
        self.D = rows
        self.buffer = np.zeros(max((c for row in rows for c in row), default=0) + 1, dtype=np.uint8)
        self.index = self.round = self.layer = 0

    def num_msyn_per_decode(self) -> int:
        return int(self.buffer.size)

    def enqueue(self, syndrome: np.ndarray) -> bool:
        n = int(syndrome.size)
        if self.index + n > self.buffer.size:
            print("Syndrome buffer overflow. Syndrome will be ignored.")
            return False
        self.round += 1
        self.buffer[self.index : self.index + n] = syndrome
        self.index += n
        if self.sliding:
            should_decode = self.round >= 2 or (self.round == 1 and self.has_first_round)
        else:
            should_decode = self.index == self.buffer.size
        if not should_decode:
            return False
        if self.sliding:
            k = self.layer
            self.layer += 1
            off, width = self.offsets[k], self.offsets[k + 1] - self.offsets[k]
            rows = self.D[off : off + width]
        else:
            rows = self.D
        detectors = np.array([np.bitwise_xor.reduce(self.buffer[row]) if row else 0 for row in rows], dtype=np.float64)
        result = np.asarray(self.decoder.decode(detectors).result, dtype=np.float64).reshape(-1)
        if self.sliding and result.size == 0:
            return False
        expected = self.num_observables() if self.to_obs else self.decoder.get_block_size()
        if result.size != expected:
            kind = "decode_to_obs" if self.to_obs else "decode_to_errs"
            raise RuntimeError(f"Decoder result size ({result.size}) does not match expected size ({expected}) for result type {kind}")
        if self.to_obs:
            for i in range(self.num_observables()):
                if result[i] != 0:
                    self.corrections[i] ^= 1
        else:
            for i, row in enumerate(self.O):
                for col in row:
                    if result[col] != 0:
                        self.corrections[i] ^= 1
        self.index = self.round = self.layer = 0
        return True

    def reset(self) -> None:
        self.index = self.round = self.layer = 0
        self.buffer[:] = 0
        self.corrections = np.zeros(len(self.O), dtype=np.uint8)
        if hasattr(self.decoder, "reset_decoder"):
            self.decoder.reset_decoder()


_decoders: List[Optional[_RealtimeDecoder]] = []


def _read_stim_dem(cfg: decoder_config) -> str:
    path = cfg.stim_dem_path
    if not os.path.isfile(path):
        raise RuntimeError(f"stim_dem_path is not a readable file: {path}")
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
    except OSError:
        raise RuntimeError(f"stim_dem_path could not be opened: {path}") from None


def _build(name: str, inputs: Any, params: dict, observables: bool) -> Any:
    from ._decoders import _native

    if observables:
        params["output"] = "observables"
    return _native(name, inputs, params, typed=True)


def _create(cfg: decoder_config) -> _RealtimeDecoder:
    """CUDA-Q's create_realtime_decoder."""
    from ._decoders import _NATIVE, _NATIVE_DEM, _Inputs, _columns, _invalid, _native_dem, _rows_of
    from ._dem import _parse_stim_dem
    from ._pcm import pcm_from_sparse_vec

    did = cfg.id
    if not 0 <= did <= 2**32 - 1:
        raise ValueError(f"Decoder ID is outside the uint32_t range: {did}")
    # (The parser rejects a config describing two forms, but a programmatically built one never
    # passes through it. These two are what corrupt: chunk priors over an unrelated model, and an
    # O_sparse that overwrites the mapping a DEM-native decoder installed for itself.)
    if cfg.stim_dem_path and (cfg.dem_chunks is not None or cfg.O_sparse):
        raise RuntimeError(f"dem_chunks and O_sparse must not be set with stim_dem_path for decoder {did}")
    # A chunk-form configuration names its DEM one round at a time. Expand it here (on a copy) so
    # that everything below is written against the flat form only.
    cfg = cfg._copy()
    closed_dem = _config.expand_dem_chunks(cfg)
    H, O, D = cfg.H_sparse, cfg.O_sparse, cfg.D_sparse
    if not D:
        raise RuntimeError("D_sparse must be provided in decoder configuration")
    if not H and not cfg.stim_dem_path:
        raise RuntimeError(
            f"H_sparse must be provided to build decoder {did}, either directly or by way of dem_chunks and num_rounds. A DEM-native decoder can name a model with stim_dem_path instead."
        )
    text = _read_stim_dem(cfg) if cfg.stim_dem_path else ""

    # prepare_decoder_params: the parameters (non-schema keys dropped, defaults filled in), the
    # device, and for PyMatching the observable matrix (so that it decodes to observables).
    params = _config._plain(cfg._args_for_decoder())
    if cfg.cuda_device_id is not None:
        params["cuda_device_id"] = cfg.cuda_device_id
    num_obs = O.count(-1)
    pymatching_obs = cfg.type == "pymatching" and bool(O) and num_obs > 0
    if pymatching_obs:
        params["O"] = pcm_from_sparse_vec(O, num_obs, cfg.block_size)
    # The phases carry a prior per fault, which the derived matrices do not. A hand-written flat
    # configuration would have had to pass these through decoder_custom_args, so supply them here
    # for the ones that did not.
    if closed_dem is not None and "error_rate_vec" not in params:
        params["error_rate_vec"] = np.array(closed_dem.error_rates, dtype=np.float64)

    rates: List[float] = []
    if text:
        if cfg.type in _NATIVE_DEM:
            decoder = _native_dem(cfg.type, text, params, typed=True)
        elif cfg.type in _NATIVE:
            # (make_pcm_decoder: the model's matrices, its observables and rates unless given.)
            num_dets, dem_obs, det_hits, obs_hits, dem_rates = _parse_stim_dem(text, False)
            O_rows: List[List[int]] = [[] for _ in range(dem_obs)]
            for e, hits in enumerate(obs_hits):
                for o in hits:
                    O_rows[o].append(e)
            if "error_rate_vec" in params:
                rates = [float(x) for x in np.asarray(params.pop("error_rate_vec"), dtype=np.float64).reshape(-1)]
            else:
                rates = list(dem_rates)
            inputs = _Inputs(num_dets, len(det_hits), [list(h) for h in det_hits], O_rows if dem_obs > 0 else None, dem_obs, rates)
            decoder = _build(cfg.type, inputs, params, cfg.type == "pymatching" and dem_obs > 0)
        else:
            raise _invalid(cfg.type)
        rows = D.count(-1)
        if rows != decoder.get_syndrome_size():
            raise RuntimeError(f"Number of rows in D_sparse vector is not equal to the number of detectors in {cfg.stim_dem_path}: {rows} != {decoder.get_syndrome_size()}")
    else:
        pcm = pcm_from_sparse_vec(H, cfg.syndrome_size, cfg.block_size)
        pcm_from_sparse_vec(O, num_obs, cfg.block_size)  # (validates O's shape for every type)
        if cfg.type not in _NATIVE:
            raise _invalid(cfg.type)
        m, n, cols = _columns(pcm)
        O_model, model_obs = None, 0
        if "O" in params:
            model_obs, _cols, O_model = _rows_of(params.pop("O"))
        if "error_rate_vec" in params:
            rates = [float(x) for x in np.asarray(params.pop("error_rate_vec"), dtype=np.float64).reshape(-1)]
        decoder = _build(cfg.type, _Inputs(m, n, cols, O_model, model_obs, rates), params, pymatching_obs)

    rt = _RealtimeDecoder(decoder, did)
    if O:
        if not pymatching_obs:
            rt.set_O_sparse(O)
    elif text and rt.num_observables() == 0:
        # (The DEM form names no O_sparse and a decoder returning errors installs none: the
        # model's observables, as make_pcm_decoder read them.)
        _n, dem_obs, _dets, obs_hits, _r = _parse_stim_dem(text, False)
        vec: List[int] = []
        for o in range(dem_obs):
            vec += [e for e, hits in enumerate(obs_hits) if o in hits] + [-1]
        rt.set_O_sparse(vec)
    rt.set_D_sparse(D)
    decoder.decode(np.zeros(decoder.get_syndrome_size()))  # (the dry run CUDA-Q makes)
    return rt


def _host_configure(config: multi_decoder_config) -> int:
    global _decoders
    configs = config._decoders
    seen = set()
    for c in configs:
        if c.id in seen:
            _warn(f"Duplicate decoder ID found: {c.id}")
            return 1
        seen.add(c.id)
    lo = min((c.id for c in configs), default=2**63 - 1)
    hi = max((c.id for c in configs), default=-(2**63))
    # (CUDA-Q compares the signed maximum with the unsigned count: a negative maximum is huge.)
    if hi % 2**64 >= len(configs):
        _warn(f"Maximum decoder ID is greater than the number of decoders: {hi} >= {len(configs)}")
        return 2
    if lo < 0:
        _warn(f"Minimum decoder ID is less than 0: {lo}")
        return 3
    _decoders = [None] * (hi + 1)
    try:
        for c in configs:
            _decoders[c.id] = _create(c)
    except Exception as e:
        _warn(f"Error initializing decoders: {e}")
        return 4
    finally:
        _on_configure([i for i, d in enumerate(_decoders) if d is not None])
    return 0


def _on_configure(ids: List[int]) -> None:
    """Set by the package: the ids now configured here leave its queue form."""


def configure_decoders(config: multi_decoder_config) -> int:
    """Configure the real-time decoders a ``multi_decoder_config`` describes, replacing those
    configured before. Returns CUDA-Q's status: 0, or 1 (duplicate id), 2 (an id not below the
    number of decoders), 3 (a negative id), 4 (a decoder failed to build; see the warning)."""
    if not isinstance(config, multi_decoder_config):
        raise TypeError(f"configure_decoders(): config must be a multi_decoder_config, not {type(config).__name__}")
    config.to_yaml_str()  # (CUDA-Q publishes the configuration as YAML, checking it, first)
    return _host_configure(config)


def configure_decoders_from_file(config_file: str) -> int:
    """Configure the real-time decoders from a YAML file; returns the status (1 when the file
    does not exist)."""
    if not isinstance(config_file, str):
        raise TypeError(f"configure_decoders_from_file(): config_file must be a str, not {type(config_file).__name__}")
    if not os.path.exists(config_file):
        _warn(f"Config file does not exist: {config_file}")
        return 1
    with open(config_file, encoding="utf-8") as f:
        text = f.read()
    return configure_decoders(multi_decoder_config.from_yaml_str(text))


def configure_decoders_from_str(config_str: str) -> int:
    """Configure the real-time decoders from a YAML string; returns the status."""
    if not isinstance(config_str, str):
        raise TypeError(f"configure_decoders_from_str(): config_str must be a str, not {type(config_str).__name__}")
    return configure_decoders(multi_decoder_config.from_yaml_str(config_str))


def finalize_decoders() -> None:
    """Release the configured real-time decoders."""
    global _decoders
    _decoders = []


def _is_configured(decoder_id: Any) -> bool:
    return isinstance(decoder_id, (int, np.integer)) and not isinstance(decoder_id, bool) and 0 <= int(decoder_id) < len(_decoders) and _decoders[int(decoder_id)] is not None


def _get(decoder_id: Any) -> _RealtimeDecoder:
    if not _is_configured(decoder_id):
        raise _DecoderNotFound(f"Decoder {decoder_id} not found")
    return _decoders[int(decoder_id)]  # type: ignore[return-value]


def _bits(syndromes: Any) -> np.ndarray:
    if isinstance(syndromes, np.ndarray):
        a = syndromes
    else:
        a = np.array([bool(s) for s in syndromes], dtype=np.uint8)
    if a.ndim != 1:
        raise ValueError(f"syndromes must be one-dimensional (a round of measurement bits), not of shape {a.shape}")
    return (a != 0).astype(np.uint8)


def enqueue_syndromes(decoder_id: int, syndromes: Any, tag: int = 0) -> None:
    """Send a decoder measurement bits (a round's, or the final data readout): once a shot's
    have all arrived, the decoder decodes them and folds its observable flips into its
    corrections. ``tag`` labels the call (CUDA-Q logs it)."""
    rt = _get(decoder_id)
    bits = _bits(syndromes)
    if bits.size == 0:
        raise ValueError("syndrome_length must be greater than 0")
    limit = rt.num_msyn_per_decode()
    if limit == 0:
        raise ValueError("Decoder has no measurement syndromes configured")
    if bits.size > limit:
        raise ValueError(f"syndrome_length ({bits.size}) exceeds configured measurement count ({limit})")
    rt.enqueue(bits)


enqueue_syndromes_test = enqueue_syndromes


def get_corrections(decoder_id: int, return_size: int, reset: bool) -> List[bool]:
    """A decoder's accumulated observable corrections (``return_size`` of them, its number of
    observables), cleared afterwards with ``reset``."""
    rt = _get(decoder_id)
    n = rt.num_observables()
    size = int(return_size)
    if size == 0:
        raise ValueError("correction_length must be greater than 0")
    if size != n:
        raise ValueError(f"correction_length ({size}) does not match number of observables ({n})")
    out = [bool(c) for c in rt.corrections]
    if reset:
        rt.corrections = np.zeros(n, dtype=np.uint8)
    return out


def reset_decoder(decoder_id: int) -> None:
    """Clear a decoder's partial shot and its corrections."""
    _get(decoder_id).reset()

# Ported from CUDA-Q QEC (NVIDIA/cudaqx, libs/qec; Copyright (c) 2022 - 2026 NVIDIA Corporation &
# Affiliates, Apache License 2.0; see THIRD_PARTY_NOTICES.md), modified for stabilizer_qec.
"""CUDA-Q QEC under its names (``import stabilizer_qec.cudaq_qec as qec``): the decoder
interface (``get_decoder``, ``Decoder``, ``DecoderResult``, ``BatchDecoderResult``, the
``@decoder`` registry), its decoders (single- and multi-error lookup tables, PyMatching, the sliding
window, nv-qldpc-decoder, Chromobius, nv-fusion-decoder, the tensor-network decoder, the
examples), its detector error model (and the extended one: chunks with named seams, stitched and closed, the
``dem_chunks`` of a repeated-round experiment, and the DEM of CSS code matrices), ``dem_sampling`` and its
parity-check-matrix tools; its codes
(``get_code``: the surface code in its orientations XV, XH, ZV and ZH, with its
``stabilizer_grid`` and CNOT schedules, Steane's and the repetition code, ``@code`` for Python
ones) and its experiments on them (memory circuits sampled or made detector error models, under
its noise models; code-capacity samples); and its real-time decoding host library
(``decoder_config`` / ``multi_decoder_config`` and their YAML, the decoder parameter schemas,
``configure_decoders``, ``enqueue_syndromes``, ``get_corrections``, ``reset_decoder``,
``finalize_decoders``). Its kernels are this package's circuits (``memory_circuit``; kernels
registered for a Python code are not run, and say so); CUDA-Q's own kernels and GPU decoders
need CUDA-Q itself and are not here."""

from ._decoders import AsyncDecoderResult, BatchDecoderResult, Decoder, DecoderResult, decoder, get_decoder
from ._codes import (
    XH,
    XV,
    ZH,
    ZV,
    BitFlipChannel,
    Code,
    DecoderContext,
    Depolarization1,
    Depolarization2,
    DepolarizationChannel,
    KernelNotRunWarning,
    NoiseModel,
    Pauli1,
    Pauli2,
    PhaseFlipChannel,
    TwoQubitBitFlip,
    TwoQubitDepolarization,
    XError,
    YError,
    ZError,
    amx,
    amz,
    code,
    d_sparse,
    decoder_context_from_memory_circuit,
    dem_from_memory_circuit,
    empty,
    generate_random_bit_flips,
    get_available_codes,
    get_code,
    memory_circuit,
    noise_model,
    operation,
    oriented_memory_circuit,
    role_to_str,
    sample_code_capacity,
    sample_memory_circuit,
    sc_orientation,
    sc_orientation_from_str,
    stabilizer_grid,
    surface_role,
    vec2d,
    x_dem_from_memory_circuit,
    x_sample_memory_circuit,
    z_dem_from_memory_circuit,
    z_sample_memory_circuit,
)
from ._css import CssCodes, CssNoise, css_matrices_from_code, dem_from_css_matrices
from ._dem import DetectorErrorModel, dem_from_stim_text
from ._extended_dem import (
    DemChunkSpec,
    DemChunksSpec,
    DemSeamSpec,
    ExtendedDem,
    ExtendedDemSeam,
    PhaseConnection,
    PhaseSpecEntry,
    PriorCombineMode,
    SeamConnection,
    SeamId,
    SeamSpecEntry,
    are_dem_columns_unique,
    assert_dem_columns_unique,
    dem_chunk_from_spec,
    dem_chunk_rounds,
    dem_chunks_from_spec,
    dem_chunks_to_d_sparse,
    dem_chunks_to_detector_round,
    dem_chunks_to_o_sparse,
    dem_chunks_to_pcm,
    dem_chunks_to_rounds,
    dem_close,
    dem_close_all,
    dem_merge_duplicate_columns,
    dem_stitch,
    dem_stitch_all,
    dem_stitch_merged,
    extended_dem_from_css_matrices,
    phase_name,
    seam_name,
)
from ._pcm import (
    dump_pcm,
    generate_random_pcm,
    generate_timelike_sparse_detector_matrix,
    get_pcm_for_rounds,
    get_sorted_pcm_column_indices,
    pcm_extend_to_n_rounds,
    pcm_from_sparse_vec,
    pcm_is_sorted,
    pcm_to_sparse_vec,
    reorder_pcm_columns,
    shuffle_pcm_columns,
    simplify_pcm,
    sort_pcm_columns,
)

from .dem_sampling import dem_sampling
from .patch import patch
from . import relay_solutions

__version__ = "0.8.1"

from ..realtime import DecoderQueue as _DecoderQueue
from . import _compat, _config, _realtime
from ._compat import chromobius_config, multi_error_lut_config, nv_qldpc_decoder_config, pymatching_config, trt_decoder_config
from ._config import (
    decoder_config,
    decoder_config_json_schema,
    decoder_dispatch,
    decoder_param_schema,
    expand_dem_chunks,
    multi_decoder_config,
    registered_decoder_schemas,
    transport_config,
    transport_shape_override,
)
from ._realtime import configure_decoders_from_file, configure_decoders_from_str, enqueue_syndromes_test

# Two forms of CUDA-Q's real-time host API share these names. CUDA-Q's own: decoders configured
# from a ``multi_decoder_config`` (or its YAML), sent a shot's measurements a round at a time
# (``enqueue_syndromes(id, syndromes, tag)``), their observable corrections read back
# (``get_corrections(id, return_size, reset)``). And this package's queue form: any decoders,
# configured by ``configure_decoders({id: decoder})`` or a list, decoding batches of detector
# syndromes behind a queue (``enqueue_syndromes(id, batch)`` returns a ticket for
# ``get_corrections(id, ticket)``). Each id belongs to the form that configured it last.

_queues: dict = {}


def _drop_queues(ids) -> None:
    for key in ids:
        q = _queues.pop(key, None)
        if q is not None:
            q.close(wait=False)


_realtime._on_configure = _drop_queues


def configure_decoders(decoders, *, workers: int = 1):
    """Configure real-time decoders. From a ``multi_decoder_config``, as CUDA-Q: the decoders it
    describes replace those configured before, and the status (0 when all were built) is
    returned. Or this package's queue form: ``decoders`` a dict ``{id: decoder}`` or a list
    (numbered from 0) of any decoders with ``decode_batch`` or ``decode`` (this package's, or a
    ``Decoder``), each put behind a queue of ``workers`` threads; returns their ids."""
    if isinstance(decoders, multi_decoder_config):
        return _realtime.configure_decoders(decoders)
    items = decoders.items() if isinstance(decoders, dict) else enumerate(decoders)
    ids = []
    for key, dec in items:
        if key in _queues:
            _queues[key].close(wait=False)
        if _realtime._is_configured(key):
            _realtime._decoders[int(key)] = None
        decode = dec.decode_batch if hasattr(dec, "decode_batch") else (lambda s, _d=dec: [_d.decode(row) for row in s])
        _queues[key] = _DecoderQueue(dec, workers=workers, decode=decode, name=f"{type(dec).__name__}#{key}")
        ids.append(key)
    return ids


def finalize_decoders() -> None:
    """Release every configured decoder (of either form)."""
    _realtime.finalize_decoders()
    _drop_queues(list(_queues))


def _queue(decoder_id):
    if decoder_id not in _queues:
        raise _realtime._DecoderNotFound(f"Decoder {decoder_id} not found; configure it first with configure_decoders")
    return _queues[decoder_id]


def enqueue_syndromes(decoder_id, syndromes, tag=None):
    """Send a decoder syndromes. A decoder configured from a ``multi_decoder_config`` takes a
    round of measurement bits (``tag`` labels the call), decoding once its shot's have all
    arrived. A queued decoder takes a batch of detector syndromes and returns the batch's
    ticket."""
    if decoder_id in _queues:
        if tag is not None:
            raise TypeError("enqueue_syndromes(): a queued decoder takes no tag")
        return _queues[decoder_id].enqueue_syndromes(syndromes)
    return _realtime.enqueue_syndromes(decoder_id, syndromes, 0 if tag is None else tag)


def get_corrections(decoder_id, *args, **kwargs):
    """A decoder's corrections. ``get_corrections(id, return_size, reset)`` for a decoder
    configured from a ``multi_decoder_config``: its accumulated observable corrections, as
    bools, cleared afterwards with ``reset``. ``get_corrections(id, ticket=None, timeout=None)``
    for a queued decoder: the corrections of the oldest batch not yet collected (or of
    ``ticket``), waited for."""
    if decoder_id in _queues:
        return _queues[decoder_id].get_corrections(*args, **kwargs)
    _realtime._get(decoder_id)  # (not found, before the arguments are looked at)
    return _realtime.get_corrections(decoder_id, *args, **kwargs)


def reset_decoder(decoder_id) -> None:
    """Clear a decoder's partial shot and accumulated corrections (configured from a
    ``multi_decoder_config``), or drop a queued decoder's batches and statistics."""
    if decoder_id in _queues:
        _queues[decoder_id].reset()
    else:
        _realtime.reset_decoder(decoder_id)


def decoder_stats(decoder_id):
    """A queued decoder's `stabilizer_qec.realtime.DecoderStats`."""
    return _queue(decoder_id).stats()


# CUDA-Q's compiled module, under its name: its configuration API is ``qecrt.config`` (with the
# deprecated typed configurations, which lived there too).
_config.configure_decoders = _realtime.configure_decoders
_config.configure_decoders_from_file = _realtime.configure_decoders_from_file
_config.configure_decoders_from_str = _realtime.configure_decoders_from_str
_config.finalize_decoders = _realtime.finalize_decoders
for _name in _compat.__all__:
    setattr(_config, _name, getattr(_compat, _name))
del _name


class _Qecrt:
    """CUDA-Q QEC's compiled module (``qecrt``): its configuration submodule, ``config``."""

    config = _config
    __version__ = "0.8.1"

    def __repr__(self) -> str:
        return "<qecrt (stabilizer_qec's CUDA-Q QEC port)>"


qecrt = _Qecrt()


__all__ = [
    "configure_decoders", "enqueue_syndromes", "get_corrections", "reset_decoder", "decoder_stats",
    "configure_decoders_from_file", "configure_decoders_from_str", "finalize_decoders", "enqueue_syndromes_test", "qecrt",
    "decoder_config", "multi_decoder_config", "decoder_dispatch", "transport_config", "transport_shape_override", "expand_dem_chunks",
    "decoder_param_schema", "registered_decoder_schemas", "decoder_config_json_schema",
    "nv_qldpc_decoder_config", "multi_error_lut_config", "trt_decoder_config", "pymatching_config", "chromobius_config",
    "BitFlipChannel", "Code", "DecoderContext", "Depolarization1", "Depolarization2", "DepolarizationChannel", "NoiseModel", "Pauli1", "Pauli2",
    "PhaseFlipChannel", "TwoQubitBitFlip", "TwoQubitDepolarization", "XError", "XH", "XV", "YError", "ZError", "ZH", "ZV", "amx", "amz", "code",
    "d_sparse", "decoder_context_from_memory_circuit", "dem_from_memory_circuit", "empty", "generate_random_bit_flips", "get_available_codes",
    "get_code", "memory_circuit", "noise_model", "operation", "oriented_memory_circuit", "role_to_str", "sample_code_capacity",
    "sample_memory_circuit", "sc_orientation", "sc_orientation_from_str", "stabilizer_grid", "surface_role", "vec2d", "x_dem_from_memory_circuit",
    "x_sample_memory_circuit", "z_dem_from_memory_circuit", "z_sample_memory_circuit",
    "AsyncDecoderResult", "BatchDecoderResult", "Decoder", "DecoderResult", "DetectorErrorModel", "decoder", "dem_from_stim_text", "dump_pcm",
    "generate_random_pcm", "get_decoder", "get_pcm_for_rounds", "get_sorted_pcm_column_indices", "pcm_extend_to_n_rounds", "pcm_is_sorted",
    "pcm_to_sparse_vec", "reorder_pcm_columns", "shuffle_pcm_columns", "simplify_pcm", "sort_pcm_columns",
    "KernelNotRunWarning", "dem_sampling", "patch", "relay_solutions", "pcm_from_sparse_vec", "generate_timelike_sparse_detector_matrix",
    "CssCodes", "CssNoise", "css_matrices_from_code", "dem_from_css_matrices",
    "DemChunkSpec", "DemChunksSpec", "DemSeamSpec", "ExtendedDem", "ExtendedDemSeam", "PhaseConnection", "PhaseSpecEntry", "PriorCombineMode", "SeamConnection",
    "SeamId", "SeamSpecEntry", "are_dem_columns_unique", "assert_dem_columns_unique", "dem_chunk_from_spec", "dem_chunk_rounds", "dem_chunks_from_spec",
    "dem_chunks_to_d_sparse", "dem_chunks_to_detector_round", "dem_chunks_to_o_sparse", "dem_chunks_to_pcm", "dem_chunks_to_rounds", "dem_close", "dem_close_all",
    "dem_merge_duplicate_columns", "dem_stitch", "dem_stitch_all", "dem_stitch_merged", "extended_dem_from_css_matrices", "phase_name", "seam_name",
]

from .plugins.decoders import example as _example  # noqa: E402,F401  (registers "example_byod")
from .plugins.decoders import tensor_network_decoder as _tn  # noqa: E402,F401  (registers "tensor_network_decoder")

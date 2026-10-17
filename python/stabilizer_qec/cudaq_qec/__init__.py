# Ported from CUDA-Q QEC (NVIDIA/cudaqx, libs/qec; Copyright (c) 2022 - 2026 NVIDIA Corporation &
# Affiliates, Apache License 2.0; see THIRD_PARTY_NOTICES.md), modified for stabilizer_qec.
"""CUDA-Q QEC under its names (``import stabilizer_qec.cudaq_qec as qec``): the decoder
interface (``get_decoder``, ``Decoder``, ``DecoderResult``, ``BatchDecoderResult``, the
``@decoder`` registry), its decoders (single- and multi-error lookup tables, PyMatching, the sliding
window, the tensor-network decoder, the examples), its detector error model and its parity-check-matrix tools;
its codes (``get_code``: the surface code in its orientations XV, XH, ZV and ZH, with its
``stabilizer_grid`` and CNOT schedules, Steane's and the repetition code, ``@code`` for Python
ones) and its experiments on them (memory circuits sampled or made detector error models, under
its noise models; code-capacity samples). Its kernels are this package's circuits
(``memory_circuit``); CUDA-Q's own kernels, GPU decoders and real-time decoding need CUDA-Q itself
and are not here."""

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
from ._dem import DetectorErrorModel, dem_from_stim_text
from ._pcm import (
    dump_pcm,
    generate_random_pcm,
    get_pcm_for_rounds,
    get_sorted_pcm_column_indices,
    pcm_extend_to_n_rounds,
    pcm_is_sorted,
    pcm_to_sparse_vec,
    reorder_pcm_columns,
    shuffle_pcm_columns,
    simplify_pcm,
    sort_pcm_columns,
)

__version__ = "0.8.1"

from ..realtime import DecoderQueue as _DecoderQueue

_queues: dict = {}


def configure_decoders(decoders, *, workers: int = 1) -> list:
    """CUDA-Q QEC's real-time host API: registers decoders (a dict ``{id: decoder}`` or a list,
    numbered from 0) behind queues; returns their ids. Any decoder with ``decode_batch`` or
    ``decode`` (this package's, or a ``Decoder``)."""
    items = decoders.items() if isinstance(decoders, dict) else enumerate(decoders)
    ids = []
    for key, dec in items:
        if key in _queues:
            _queues[key].close(wait=False)
        decode = dec.decode_batch if hasattr(dec, "decode_batch") else (lambda s, _d=dec: [_d.decode(row) for row in s])
        _queues[key] = _DecoderQueue(dec, workers=workers, decode=decode, name=f"{type(dec).__name__}#{key}")
        ids.append(key)
    return ids


def _queue(decoder_id):
    if decoder_id not in _queues:
        raise KeyError(f"decoder {decoder_id!r} is not configured; call configure_decoders first")
    return _queues[decoder_id]


def enqueue_syndromes(decoder_id, syndromes) -> int:
    """Queues a batch of syndromes to a configured decoder; returns the batch's ticket."""
    return _queue(decoder_id).enqueue_syndromes(syndromes)


def get_corrections(decoder_id, ticket=None, timeout=None):
    """The corrections of the oldest batch not yet collected (or of ``ticket``), waited for."""
    return _queue(decoder_id).get_corrections(ticket, timeout)


def reset_decoder(decoder_id) -> None:
    """Drops the decoder's queued batches and its statistics."""
    _queue(decoder_id).reset()


def decoder_stats(decoder_id):
    """The decoder's `stabilizer_qec.realtime.DecoderStats`."""
    return _queue(decoder_id).stats()


__all__ = [
    "configure_decoders", "enqueue_syndromes", "get_corrections", "reset_decoder", "decoder_stats",
    "BitFlipChannel", "Code", "DecoderContext", "Depolarization1", "Depolarization2", "DepolarizationChannel", "NoiseModel", "Pauli1", "Pauli2",
    "PhaseFlipChannel", "TwoQubitBitFlip", "TwoQubitDepolarization", "XError", "XH", "XV", "YError", "ZError", "ZH", "ZV", "amx", "amz", "code",
    "d_sparse", "decoder_context_from_memory_circuit", "dem_from_memory_circuit", "empty", "generate_random_bit_flips", "get_available_codes",
    "get_code", "memory_circuit", "noise_model", "operation", "oriented_memory_circuit", "role_to_str", "sample_code_capacity",
    "sample_memory_circuit", "sc_orientation", "sc_orientation_from_str", "stabilizer_grid", "surface_role", "vec2d", "x_dem_from_memory_circuit",
    "x_sample_memory_circuit", "z_dem_from_memory_circuit", "z_sample_memory_circuit",
    "AsyncDecoderResult", "BatchDecoderResult", "Decoder", "DecoderResult", "DetectorErrorModel", "decoder", "dem_from_stim_text", "dump_pcm",
    "generate_random_pcm", "get_decoder", "get_pcm_for_rounds", "get_sorted_pcm_column_indices", "pcm_extend_to_n_rounds", "pcm_is_sorted",
    "pcm_to_sparse_vec", "reorder_pcm_columns", "shuffle_pcm_columns", "simplify_pcm", "sort_pcm_columns",
]

from .plugins.decoders import example as _example  # noqa: E402,F401  (registers "example_byod")
from .plugins.decoders import tensor_network_decoder as _tn  # noqa: E402,F401  (registers "tensor_network_decoder")

# Ported from CUDA-Q QEC (NVIDIA/cudaqx, libs/qec; Copyright (c) 2022 - 2026 NVIDIA Corporation &
# Affiliates, Apache License 2.0; see THIRD_PARTY_NOTICES.md), modified for stabilizer_qec.
"""CUDA-Q QEC's decoders under its names (``import stabilizer_qec.cudaq_qec as qec``): the
decoder interface (``get_decoder``, ``Decoder``, ``DecoderResult``, ``BatchDecoderResult``, the
``@decoder`` registry), its decoders (single- and multi-error lookup tables, PyMatching, the sliding
window, the tensor-network decoder, the examples), its detector error model and its parity-check-matrix tools. CUDA-Q's quantum
kernels, GPU decoders and real-time decoding need CUDA-Q itself and are not here."""

from ._decoders import AsyncDecoderResult, BatchDecoderResult, Decoder, DecoderResult, decoder, get_decoder
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

__all__ = [
    "AsyncDecoderResult", "BatchDecoderResult", "Decoder", "DecoderResult", "DetectorErrorModel", "decoder", "dem_from_stim_text", "dump_pcm",
    "generate_random_pcm", "get_decoder", "get_pcm_for_rounds", "get_sorted_pcm_column_indices", "pcm_extend_to_n_rounds", "pcm_is_sorted",
    "pcm_to_sparse_vec", "reorder_pcm_columns", "shuffle_pcm_columns", "simplify_pcm", "sort_pcm_columns",
]

from .plugins.decoders import example as _example  # noqa: E402,F401  (registers "example_byod")
from .plugins.decoders import tensor_network_decoder as _tn  # noqa: E402,F401  (registers "tensor_network_decoder")

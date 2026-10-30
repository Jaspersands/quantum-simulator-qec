"""MWPF 0.2.12 (Yue Wu; MIT, see LICENSE here), whole, on this package's Rust port of it:
``import stabilizer_qec.mwpf as mwpf`` and MWPF's code runs unchanged. Hypergraph
minimum-weight parity factor decoding (Wu et al., arXiv 2508.04969): the solvers
(``SolverSerialJointSingleHair``, ``SolverSerialSingleHair``, ``SolverSerialUnionFind`` and the
BP wrapper), with the same subgraphs and weight bounds as MWPF's; the example codes, the
matrices behind its relaxers, the visualizer and its HTML export, benchmark suites, and its sinter
decoder (``SinterMWPFDecoder``), heralded error models and reference circuits.

Its linear programs are solved by HiGHS 1.8.0, compiled in as MWPF's wheels compile it, so the
subgraphs, weight bounds and snapshots are MWPF's; QEC-Playground's codes
(``QECPlaygroundCode``, built on the qecp crate) are not included."""

from .. import _core

mwpf = _core.mwpf

globals().update({name: value for name, value in vars(mwpf).items() if not name.startswith("_")})

# The names of the helpers' own (their imports, such as stim, numpy and json, are not exported).
_HELPER_NAMES = [
    "available_decoders", "default_cluster_node_limit", "DecoderPanic", "PanicAction", "SinterMWPFDecoder", "iter_det",
    "construct_decoder_and_predictor", "panic_text_of", "SinterHUFDecoder", "SinterSingleHairDecoder", "MwpfCompiledDecoder",
    "decode_common", "record_trace", "DEM_MIN_PROBABILITY", "dem_probability", "HeraldedDetectorErrorModel",
    "HeraldedDemPredictor", "add_herald_detectors", "remove_herald_detectors", "is_heralded_error",
    "heralded_instruction_to_noise_instruction", "RefRec", "RefInstruction", "RefDetector", "RefCircuit",
    "NOISE_CHANNEL_INSTRUCTION_WARNING_KEYWORDS", "NOISE_CHANNEL_INSTRUCTION_NAMES", "is_noise_channel_instruction",
    "RefDemInstruction", "RefDetectorErrorModel", "Predictor", "StaticPredictor", "DemHyperedge", "WEIGHT_MAX",
    "probability_to_weight", "weight_to_probability", "exclusive_probability", "HAS_TAG", "extra_instruction_params",
]
__all__ = [name for name in vars(mwpf) if not name.startswith("_")]

try:
    from .sinter_decoders import *  # noqa: E402,F401,F403
    from . import heralded_dem  # noqa: E402,F401
    from . import ref_circuit  # noqa: E402,F401
    __all__ += _HELPER_NAMES
except BaseException:  # as MWPF does: the helpers need Python 3.10 and NumPy
    ...

# (c) Copyright Riverlane 2020-2026. All rights reserved.
# modified for stabilizer_qec: its Stim is stabilizer_qec.deltakit_stim and its PyMatching this package's, and deltakit_{circuit,core,decode,explorer} are stabilizer_qec.deltakit.{circuit,core,decode,explorer}.
"""Description of ``deltakit.decode.analysis`` namespace here."""

from stabilizer_qec.deltakit.decode.analysis._decoder_manager import (
    DecoderManager,
    InvalidGlobalManagerStateError,
)
from stabilizer_qec.deltakit.decode.analysis._decoding_circuits import run_decoding_on_circuit
from stabilizer_qec.deltakit.decode.analysis._empirical_decoding_error_distribution import (
    EmpiricalDecodingErrorDistribution,
)
from stabilizer_qec.deltakit.decode.analysis._matching_decoder_managers import (
    B8DecoderManager,
    GraphDecoderManager,
    StimDecoderManager,
)
from stabilizer_qec.deltakit.decode.analysis._run_all_analysis_engine import RunAllAnalysisEngine

# List only public members in `__all__`.
__all__ = [s for s in dir() if not s.startswith("_")]

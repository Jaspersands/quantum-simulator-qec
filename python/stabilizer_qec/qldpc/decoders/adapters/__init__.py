# SPDX-License-Identifier: Apache-2.0
# Ported from qLDPC 0.4.1 (qLDPCOrg/qLDPC; Apache License 2.0; see THIRD_PARTY_NOTICES.md),
# modified for stabilizer_qec: its Stim, sinter, ldpc, PyMatching, relay_bp and Tesseract are this
# package's, and qldpc is stabilizer_qec.qldpc.

"""Generic error- and observable-decoder adapters."""

from ..protocols import BatchErrorDecoder, ErrorDecoder
from .error_decoders import (
    ErrorsToObservablesDecoder,
    ExpandedErrorDecoder,
    match_error_decoder_to_dem,
)
from .observable_decoders import (
    BitPackedObservableDecoder,
    ErrorsToFieldObservablesDecoder,
    validate_decoder_output,
    validate_observable_decoder,
)

__all__ = [
    "BatchErrorDecoder",
    "BitPackedObservableDecoder",
    "ErrorDecoder",
    "ErrorsToFieldObservablesDecoder",
    "ErrorsToObservablesDecoder",
    "ExpandedErrorDecoder",
    "match_error_decoder_to_dem",
    "validate_decoder_output",
    "validate_observable_decoder",
]

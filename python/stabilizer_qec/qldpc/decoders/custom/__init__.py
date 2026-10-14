# SPDX-License-Identifier: Apache-2.0
# Ported from qLDPC 0.4.1 (qLDPCOrg/qLDPC; Apache License 2.0; see THIRD_PARTY_NOTICES.md),
# modified for stabilizer_qec: its Stim, sinter, ldpc, PyMatching, relay_bp and Tesseract are this
# package's, and qldpc is stabilizer_qec.qldpc.

"""Custom decoder classes."""

from typing import TYPE_CHECKING, Any

from stabilizer_qec.qldpc._util import get_deprecated_alias

from ..common import PLACEHOLDER_ERROR_RATE
from ..external.relay_bp import RelayBPDecoder
from ..protocols import BatchErrorDecoder, ErrorDecoder
from .composition import CompositeDecoder, DirectDecoder
from .guf import GUFDecoder
from .ilp import ILPDecoder
from .lookup import (
    LookupDecoder,
    ObservableLookupDecoder,
    WeightedLookupDecoder,
    WeightedObservableLookupDecoder,
)

__all__ = [
    "PLACEHOLDER_ERROR_RATE",
    "BatchDecoder",
    "CompositeDecoder",
    "Decoder",
    "DirectDecoder",
    "GUFDecoder",
    "ILPDecoder",
    "LookupDecoder",
    "ObservableLookupDecoder",
    "RelayBPDecoder",
    "WeightedLookupDecoder",
    "WeightedObservableLookupDecoder",
]

# Deprecated compatibility aliases

DEPRECATED_ALIASES = {"Decoder": ErrorDecoder, "BatchDecoder": BatchErrorDecoder}

if TYPE_CHECKING:
    Decoder = ErrorDecoder
    BatchDecoder = BatchErrorDecoder
else:

    def __getattr__(name: str) -> Any:
        """Resolve deprecated names of decoder protocols, with a DeprecationWarning."""
        return get_deprecated_alias(__name__, name, DEPRECATED_ALIASES)

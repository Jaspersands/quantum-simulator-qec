# SPDX-License-Identifier: Apache-2.0
# Ported from qLDPC 0.4.1 (qLDPCOrg/qLDPC; Apache License 2.0; see THIRD_PARTY_NOTICES.md),
# modified for stabilizer_qec: its Stim, sinter, ldpc, PyMatching, relay_bp and Tesseract are this
# package's, and qldpc is stabilizer_qec.qldpc.

"""Decoders for Sinter to sample quantum error correction circuits."""

from typing import TYPE_CHECKING, Any

from stabilizer_qec.qldpc._util import get_deprecated_alias

from ..protocols import ErrorDecoder
from .core import (
    CompiledSinterDecoder,
    CompiledTrivialDecoder,
    SinterDecoder,
    TrivialDecoder,
)
from .subgraph import CompiledSubgraphDecoder, SubgraphDecoder
from .window import (
    CompiledSequentialWindowDecoder,
    SequentialWindowDecoder,
    SlidingWindowDecoder,
)

__all__ = [
    "CompiledSequentialWindowDecoder",
    "CompiledSinterDecoder",
    "CompiledSubgraphDecoder",
    "CompiledTrivialDecoder",
    "Decoder",
    "SequentialSinterDecoder",
    "SequentialWindowDecoder",
    "SinterDecoder",
    "SlidingWindowDecoder",
    "SubgraphDecoder",
    "SubgraphSinterDecoder",
    "TrivialDecoder",
]

# Deprecated compatibility aliases

DEPRECATED_ALIASES: dict[str, type] = {
    "Decoder": ErrorDecoder,
    "SequentialSinterDecoder": SequentialWindowDecoder,
    "SubgraphSinterDecoder": SubgraphDecoder,
}

# Deprecated names resolve at runtime through a module-level __getattr__ that warns when accessed.
# Type checkers instead see plain aliases, so that they still flag misspelled attributes.
if TYPE_CHECKING:
    Decoder = ErrorDecoder
    SequentialSinterDecoder = SequentialWindowDecoder
    SubgraphSinterDecoder = SubgraphDecoder
else:

    def __getattr__(name: str) -> Any:
        """Resolve deprecated names of observable decoders, with a DeprecationWarning."""
        return get_deprecated_alias(__name__, name, DEPRECATED_ALIASES)

# SPDX-License-Identifier: Apache-2.0
# Ported from qLDPC 0.4.1 (qLDPCOrg/qLDPC; Apache License 2.0; see THIRD_PARTY_NOTICES.md),
# modified for stabilizer_qec: its Stim, sinter, ldpc, PyMatching, relay_bp and Tesseract are this
# package's, and qldpc is stabilizer_qec.qldpc.

"""Tests for the Sinter-decoder package facade."""

from __future__ import annotations

import pytest

from stabilizer_qec.qldpc import decoders

# Deprecated compatibility


def test_deprecated_aliases() -> None:
    """Deprecated Sinter aliases warn and resolve to their replacements."""
    with pytest.warns(DeprecationWarning, match="SubgraphSinterDecoder is deprecated"):
        assert decoders.sinter.SubgraphSinterDecoder is decoders.SubgraphDecoder
    with pytest.warns(DeprecationWarning, match="SequentialSinterDecoder is deprecated"):
        assert decoders.sinter.SequentialSinterDecoder is decoders.SequentialWindowDecoder
    with pytest.warns(DeprecationWarning, match="Decoder is deprecated; use ErrorDecoder"):
        assert decoders.sinter.Decoder is decoders.ErrorDecoder

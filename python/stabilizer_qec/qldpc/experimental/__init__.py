# Ported from qLDPC 0.4.1 (qLDPCOrg/qLDPC; Apache License 2.0; see THIRD_PARTY_NOTICES.md),
# modified for stabilizer_qec: its Stim, sinter, ldpc, PyMatching, relay_bp and Tesseract are this
# package's, and qldpc is stabilizer_qec.qldpc.
"""Experimental ``qldpc`` subpackages.

EXPERIMENTAL: everything under ``qldpc.experimental`` is under active development. Public APIs
here are unstable and may change without notice or deprecation.
"""

from . import surgery

__all__ = [
    "surgery",
]

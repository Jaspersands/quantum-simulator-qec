# SPDX-License-Identifier: Apache-2.0
# Ported from qLDPC 0.4.1 (qLDPCOrg/qLDPC; Apache License 2.0; see THIRD_PARTY_NOTICES.md),
# modified for stabilizer_qec: its Stim, sinter, ldpc, PyMatching, relay_bp and Tesseract are this
# package's, and qldpc is stabilizer_qec.qldpc.

import importlib.metadata

from . import abstract, cache, circuits, codes, decoders, experimental, external, math, objects

__version__ = "0.4.1"

__all__ = [
    "__version__",
    "abstract",
    "cache",
    "circuits",
    "codes",
    "decoders",
    "experimental",
    "external",
    "math",
    "objects",
]

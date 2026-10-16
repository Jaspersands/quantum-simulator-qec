# (c) Copyright Riverlane 2020-2026. All rights reserved.
# modified for stabilizer_qec: its Stim is stabilizer_qec.deltakit_stim and its PyMatching this package's, and deltakit_{circuit,core,decode,explorer} are stabilizer_qec.deltakit.{circuit,core,decode,explorer}.
"""Description of ``deltakit.decode`` namespace here."""

import importlib.metadata

from stabilizer_qec.deltakit.decode._mwpm_decoder import PyMatchingDecoder

# It looks like these were intended to be separate, public modules.
# For now, import them as such. This can be reconsidered during API review.
from . import analysis, noise_sources, utils

__version__ = "0.9.2"

# Prevent import of `importlib` (and any other non-public objects) from this module.
del importlib

# List only public members in `__all__`.
__all__ = [s for s in dir() if not s.startswith("_")]

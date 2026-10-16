# (c) Copyright Riverlane 2020-2026. All rights reserved.
# modified for stabilizer_qec: its Stim is stabilizer_qec.deltakit_stim and its PyMatching this package's, and deltakit_{circuit,core,decode,explorer} are stabilizer_qec.deltakit.{circuit,core,decode,explorer}.
"""Description of ``deltakit.core`` namespace here."""

import importlib.metadata

# It looks like these were intended to be separate, public modules.
# For now, import them as such. This can be reconsidered during API review.
from . import decoding_graphs

__version__ = "0.9.2"

# Prevent import of `importlib` (and any other non-public objects) from this module.
del importlib

# List only public members in `__all__`.
__all__ = ["decoding_graphs"]

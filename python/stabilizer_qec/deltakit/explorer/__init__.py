# (c) Copyright Riverlane 2020-2026. All rights reserved..
# modified for stabilizer_qec: its Stim is stabilizer_qec.deltakit_stim and its PyMatching this package's, and deltakit_{circuit,core,decode,explorer} are stabilizer_qec.deltakit.{circuit,core,decode,explorer}.
import importlib.metadata

from stabilizer_qec.deltakit.explorer import analysis
from stabilizer_qec.deltakit.explorer._api._client import Client
from stabilizer_qec.deltakit.explorer._utils._logging import Logging

__version__ = "0.9.4"

# Prevent import of `importlib` (and any other non-public objects) from this module.
del importlib

# List only public members in `__all__`.
__all__ = [s for s in dir() if not s.startswith("_")]

# (c) Copyright Riverlane 2020-2026. All rights reserved.
# modified for stabilizer_qec: its Stim is stabilizer_qec.deltakit_stim and its PyMatching this package's, and deltakit_{circuit,core,decode,explorer} are stabilizer_qec.deltakit.{circuit,core,decode,explorer}.
from __future__ import annotations

from stabilizer_qec.deltakit.explorer import __version__


def test_version():
    # The port is no distribution of its own; its version is deltakit-explorer's.
    assert __version__ == "0.9.4"

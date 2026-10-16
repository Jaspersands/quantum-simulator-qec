# (c) Copyright Riverlane 2020-2025.
# modified for stabilizer_qec: its Stim is stabilizer_qec.deltakit_stim and its PyMatching this package's, and deltakit_{circuit,core,decode,explorer} are stabilizer_qec.deltakit.{circuit,core,decode,explorer}.
from stabilizer_qec import deltakit_stim as stim
import pytest


@pytest.fixture
def empty_circuit():
    return stim.Circuit()

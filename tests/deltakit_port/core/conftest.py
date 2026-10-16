# (c) Copyright Riverlane 2020-2026. All rights reserved.
# modified for stabilizer_qec: its Stim is stabilizer_qec.deltakit_stim and its PyMatching this package's, and deltakit_{circuit,core,decode,explorer} are stabilizer_qec.deltakit.{circuit,core,decode,explorer}.
# Its reference data is found beside this conftest (not from a git root).
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(scope="session")
def random_generator():
    return np.random.default_rng()


@pytest.fixture(scope="session")
def reference_data_dir():
    # Beside this conftest, wherever the tests are run from.
    return Path(__file__).parent / "reference_data"

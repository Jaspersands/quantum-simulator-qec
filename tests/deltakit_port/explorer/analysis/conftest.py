# (c) Copyright Riverlane 2020-2026. All rights reserved.
from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture(scope="session")
def random_generator() -> np.random.Generator:
    return np.random.default_rng()

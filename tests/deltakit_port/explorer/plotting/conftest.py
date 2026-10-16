# (c) Copyright Riverlane 2020-2026. All rights reserved.
# modified for stabilizer_qec: its Stim is stabilizer_qec.deltakit_stim and its PyMatching this package's, and deltakit_{circuit,core,decode,explorer} are stabilizer_qec.deltakit.{circuit,core,decode,explorer}.
from __future__ import annotations

import numpy as np
import pytest

from stabilizer_qec.deltakit.explorer.analysis import LambdaData
from stabilizer_qec.deltakit.explorer.analysis import (
    LogicalErrorProbabilityPerRoundData as LEPPRData,
)


@pytest.fixture
def lambda_results() -> LambdaData:
    return LambdaData(
        lambda_=3.0,
        lambda_std=0.1,
        lambda0=1.5,
        lambda0_std=0.05,
        distances=np.array([5, 7, 9]),
        leppr=np.array([0.0246, 0.0082, 0.0027]),
        leppr_std=np.array([0.00246, 0.00082, 0.00027]),
    )


@pytest.fixture
def leppr_results() -> LEPPRData:
    return LEPPRData(
        leppr=0.001,
        leppr_stddev=0.0001,
        num_rounds=np.array([2, 4, 6]),
        spam_error=0.01,
        spam_error_stddev=0.001,
    )


@pytest.fixture
def distances() -> np.ndarray:
    return np.array([5, 7, 9])


@pytest.fixture
def num_rounds() -> np.ndarray:
    return np.array([2, 4, 6])

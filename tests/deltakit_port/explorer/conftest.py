# (c) Copyright Riverlane 2020-2026. All rights reserved.
# modified for stabilizer_qec: its Stim is stabilizer_qec.deltakit_stim and its PyMatching this package's, and deltakit_{circuit,core,decode,explorer} are stabilizer_qec.deltakit.{circuit,core,decode,explorer}.
# Its server URL is set for its tests alone (an autouse fixture: pytest_sessionstart runs only
# from the conftests pytest starts with, and the variable would outlive these tests).
from __future__ import annotations

import pytest

from stabilizer_qec.deltakit.explorer._utils._utils import DELTAKIT_SERVER_URL_ENV


@pytest.fixture(autouse=True)
def _deltakit_server_url(monkeypatch):
    monkeypatch.setenv(DELTAKIT_SERVER_URL_ENV, "http://deltakit-explorer:8000")

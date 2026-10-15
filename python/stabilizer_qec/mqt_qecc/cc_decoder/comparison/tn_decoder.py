# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License
# Ported from MQT QECC 2.0.0 (munich-quantum-toolkit/qecc; Chair for Design Automation, TUM; MIT; see
# THIRD_PARTY_NOTICES.md), modified for stabilizer_qec: its Stim, sinter, ldpc, PyMatching, qecsim and bposd are this
# package's, and mqt.qecc is stabilizer_qec.mqt_qecc.

"""Decoding simulation using the tensor network implementation of the qecsim package."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from stabilizer_qec.qecsim import app
from stabilizer_qec.qecsim.models.color import Color666Code, Color666MPSDecoder
from stabilizer_qec.qecsim.models.generic import BitFlipErrorModel


def _fix_data(data: dict[str, Any]) -> dict[str, Any]:
    """Fix the data dictionary to be JSON serializable."""
    fixed_data = {}
    for key, value in data.items():
        if isinstance(value, np.integer):
            fixed_data[key] = int(value)
        else:
            fixed_data[key] = value
    return fixed_data


def run(
    distance: int,
    error_rate: float,
    nr_sims: int = 10000,
    results_dir: str = "./results_tn",
) -> None:
    """Run the decoder for the hexagonal color code.

    Args:
        distance: distance to run
        error_rate: error rate to run
        nr_sims: number of samples to run
        results_dir: directory to store results.
    """
    code = Color666Code(distance)
    error_model = BitFlipErrorModel()
    decoder = Color666MPSDecoder(chi=8)
    data = app.run(code, error_model, decoder, error_rate, max_runs=nr_sims)
    filename = f"distance={distance},p={round(error_rate, 4)}.json"
    path = Path(results_dir)
    path.mkdir(parents=True, exist_ok=True)
    with (path / filename).open("w") as out:
        out.write(json.dumps(_fix_data(data)))

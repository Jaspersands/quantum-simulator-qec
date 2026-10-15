# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License
# Ported from MQT QECC 2.0.0 (munich-quantum-toolkit/qecc; Chair for Design Automation, TUM; MIT; see
# THIRD_PARTY_NOTICES.md), modified for stabilizer_qec: its Stim, sinter, ldpc, PyMatching, qecsim and bposd are this
# package's, and mqt.qecc is stabilizer_qec.mqt_qecc.

"""Definitions used in this module."""

from __future__ import annotations

STIM_SQGS = {
    "H",
    "X",
    "Y",
    "Z",
    "S",
    "S_DAG",
    "SQRT_X",
    "C_XYZ",
    "C_ZYX",
    "H_XY",
    "H_XZ",
    "H_YZ",
    "SQRT_X_DAG",
    "SQRT_Y",
    "SQRT_Y_DAG",
    "SQRT_Z",
    "SQRT_Z_DAG",
}
STIM_TQGS = {
    "CNOT",
    "CX",
    "CXSWAP",
    "CY",
    "CZ",
    "CZSWAP",
    "ISWAP",
    "ISWAP_DAG",
    "SQRT_XX",
    "SQRT_XX_DAG",
    "SQRT_YY",
    "SQRT_YY_DAG",
    "SQRT_ZZ",
    "SQRT_ZZ_DAG",
    "SWAP",
    "SWAPCX",
    "SWAPCZ",
    "XCX",
    "XCY",
    "XCZ",
    "YCX",
    "YCY",
    "YCZ",
    "ZCX",
    "ZCY",
    "ZCZ",
}
STIM_MEASUREMENTS = {"MR", "MRX", "MRY", "MRZ"}
STIM_RESETS = {"R", "RX", "RY", "RZ"}

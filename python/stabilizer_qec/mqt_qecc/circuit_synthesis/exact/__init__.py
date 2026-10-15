# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License
# Ported from MQT QECC 2.0.0 (munich-quantum-toolkit/qecc; Chair for Design Automation, TUM; MIT; see
# THIRD_PARTY_NOTICES.md), modified for stabilizer_qec: its Stim, sinter, ldpc, PyMatching, qecsim and bposd are this
# package's, and mqt.qecc is stabilizer_qec.mqt_qecc.

"""Exact synthesis module for Clifford circuits."""

from __future__ import annotations

from .gate_operations import (
    get_clifford_cz_gate_set,
    get_clifford_extended_gate_set,
    get_clifford_sx_gate_set,
    get_standard_clifford_gate_set,
)
from .search import synthesize_isometry_exact
from .types import (
    Objective,
    SynthesisResult,
    SynthesisStatus,
    TargetKind,
)
from .vars import CliffordDepthVars, CliffordGateCountVars

__all__ = [
    "CliffordDepthVars",
    "CliffordGateCountVars",
    "Objective",
    "SynthesisResult",
    "SynthesisStatus",
    "TargetKind",
    "get_clifford_cz_gate_set",
    "get_clifford_extended_gate_set",
    "get_clifford_sx_gate_set",
    "get_standard_clifford_gate_set",
    "synthesize_isometry_exact",
]

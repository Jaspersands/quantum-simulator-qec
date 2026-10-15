# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License
# Ported from MQT QECC 2.0.0 (munich-quantum-toolkit/qecc; Chair for Design Automation, TUM; MIT; see
# THIRD_PARTY_NOTICES.md), modified for stabilizer_qec: its Stim, sinter, ldpc, PyMatching, qecsim and bposd are this
# package's, and mqt.qecc is stabilizer_qec.mqt_qecc.

"""Shared type definitions for circuit synthesis."""

from __future__ import annotations

from ..codes.core.pauli import CheckMatrix, StabilizerTableau

# Type alias for matrices that can be used in elimination
BinaryMatrix = CheckMatrix | StabilizerTableau

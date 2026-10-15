# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License
# Ported from MQT QECC 2.0.0 (munich-quantum-toolkit/qecc; Chair for Design Automation, TUM; MIT; see
# THIRD_PARTY_NOTICES.md), modified for stabilizer_qec: its Stim, sinter, ldpc, PyMatching, qecsim and bposd are this
# package's, and mqt.qecc is stabilizer_qec.mqt_qecc.

"""Contains the implementation of the tensor network decoder for the hexagonal color code."""

from __future__ import annotations

from ..codes.constructions.color_codes.color_code import ColorCode, LatticeType
from ..codes.constructions.color_codes.hexagonal_color_code import HexagonalColorCode
from ..codes.constructions.color_codes.square_octagon_color_code import SquareOctagonColorCode
from .comparison import tn_decoder
from .utils import code_from_string

__all__ = [
    "ColorCode",
    "HexagonalColorCode",
    "LatticeType",
    "SquareOctagonColorCode",
    "code_from_string",
    "tn_decoder",
]

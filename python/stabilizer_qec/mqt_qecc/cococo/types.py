# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License
# Ported from MQT QECC 2.0.0 (munich-quantum-toolkit/qecc; Chair for Design Automation, TUM; MIT; see
# THIRD_PARTY_NOTICES.md), modified for stabilizer_qec: its Stim, sinter, ldpc, PyMatching, qecsim and bposd are this
# package's, and mqt.qecc is stabilizer_qec.mqt_qecc.

"""Type definitions."""

from typing import TypedDict

pos = tuple[int, int]
Layout = dict[str | int, pos | list[pos]]
VdpDict = dict[str | pos | tuple[pos, pos], list[pos]]


class HistoryTemp(TypedDict, total=False):  # pragma: no cover
    """Type for history dictionaries."""

    scores: list[int]
    layout_init: dict[str | int, pos | list[pos]]
    layout_final: dict[str | int, pos | list[pos]]

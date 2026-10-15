# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License
# Ported from MQT QECC 2.0.0 (munich-quantum-toolkit/qecc; Chair for Design Automation, TUM; MIT; see
# THIRD_PARTY_NOTICES.md), modified for stabilizer_qec: its Stim, sinter, ldpc, PyMatching, qecsim and bposd are this
# package's, and mqt.qecc is stabilizer_qec.mqt_qecc.

"""Utilities for the circuit synthesis unit tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from stabilizer_qec.mqt_qecc.mod2 import rank

if TYPE_CHECKING:  # pragma: no cover
    import numpy.typing as npt


def eq_span(a: npt.NDArray[np.int8], b: npt.NDArray[np.int8]) -> bool:
    """Check if two matrices have the same row space."""
    return (a.shape[1] == b.shape[1]) and (int(rank(np.vstack((a, b)))) == int(rank(a)) == int(rank(b)))


def in_span(m: npt.NDArray[np.int8], v: npt.NDArray[np.int8]) -> bool:
    """Check if a vector is in the row space of a matrix over GF(2)."""
    return bool(rank(np.vstack((m, v))) == rank(m))

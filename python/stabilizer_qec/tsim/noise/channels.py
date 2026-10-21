# Ported from Tsim (QuEraComputing/tsim, bloqade-tsim 0.1.5, src/tsim/noise/channels.py; Apache License 2.0;
# see LICENSE in this directory), modified for stabilizer_qec: only the probability tables of the channels
# the ZX builders record (the channel simplification and sampling are JAX programs; this package's samplers
# are its own).
"""The probability tables of Tsim's Pauli noise channels, one entry per outcome of the channel's error bits."""

from __future__ import annotations

from typing import List

import numpy as np

__all__ = [
    "correlated_error_probs",
    "error_probs",
    "heralded_pauli_channel_1_probs",
    "pauli_channel_1_probs",
    "pauli_channel_2_probs",
]


def error_probs(p: float) -> np.ndarray:
    """Single-bit error channel.

    Returns ``[P(bit0=0), P(bit0=1)]``.
    """
    return np.array([1 - p, p], dtype=np.float64)


def heralded_pauli_channel_1_probs(pi: float, px: float, py: float, pz: float) -> np.ndarray:
    """Heralded single-qubit Pauli channel. Returns shape (8,).

    Bit layout:
    - bit 0: herald bit, written to the measurement record
    - bit 1: Z error component
    - bit 2: X error component

    The non-zero outcomes are:
    - index 0 (0b000): no herald, no Pauli error
    - index 1 (0b001): herald + I
    - index 3 (0b011): herald + Z
    - index 5 (0b101): herald + X
    - index 7 (0b111): herald + Y, represented as X+Z
    """
    probs = np.zeros(8, dtype=np.float64)
    probs[0] = 1 - pi - px - py - pz
    probs[1] = pi
    probs[3] = pz
    probs[5] = px
    probs[7] = py
    return probs


def pauli_channel_1_probs(px: float, py: float, pz: float) -> np.ndarray:
    """Single-qubit Pauli channel. Returns shape (4,).

    Bit layout:
    - bit 0: Z error component
    - bit 1: X error component

    The outcomes are:
    - index 0 (0b00): I
    - index 1 (0b01): Z
    - index 2 (0b10): X
    - index 3 (0b11): Y
    """
    return np.array([1 - px - py - pz, pz, px, py], dtype=np.float64)


def pauli_channel_2_probs(
    pix: float,
    piy: float,
    piz: float,
    pxi: float,
    pxx: float,
    pxy: float,
    pxz: float,
    pyi: float,
    pyx: float,
    pyy: float,
    pyz: float,
    pzi: float,
    pzx: float,
    pzy: float,
    pzz: float,
) -> np.ndarray:
    """Two-qubit Pauli channel. Returns shape (16,).

    Bit layout:
    - bit 0: Z error component on ``qubit_i``
    - bit 1: X error component on ``qubit_i``
    - bit 2: Z error component on ``qubit_j``
    - bit 3: X error component on ``qubit_j``

    With that layout, index ``z_i + 2*x_i + 4*z_j + 8*x_j`` stores the
    probability for the corresponding two-qubit Pauli outcome. The arguments
    follow Stim's naming convention: ``pix`` is I on ``qubit_i`` and X on
    ``qubit_j``, ``pzi`` is Z on ``qubit_i`` and I on ``qubit_j``, etc.
    """
    remainder = 1 - pix - piy - piz - pxi - pxx - pxy - pxz - pyi - pyx - pyy - pyz - pzi - pzx - pzy - pzz
    probs = np.array(
        [
            remainder,  # index 0 (0b0000): II
            pzi,  # index 1 (0b0001): ZI
            pxi,  # index 2 (0b0010): XI
            pyi,  # index 3 (0b0011): YI
            piz,  # index 4 (0b0100): IZ
            pzz,  # index 5 (0b0101): ZZ
            pxz,  # index 6 (0b0110): XZ
            pyz,  # index 7 (0b0111): YZ
            pix,  # index 8 (0b1000): IX
            pzx,  # index 9 (0b1001): ZX
            pxx,  # index 10 (0b1010): XX
            pyx,  # index 11 (0b1011): YX
            piy,  # index 12 (0b1100): IY
            pzy,  # index 13 (0b1101): ZY
            pxy,  # index 14 (0b1110): XY
            pyy,  # index 15 (0b1111): YY
        ],
        dtype=np.float64,
    )
    return probs


def correlated_error_probs(probabilities: List[float]) -> np.ndarray:
    """Build probability distribution for correlated error chain.

    Given conditional probabilities [p1, p2, ..., pk] from a chain of
    CORRELATED_ERROR(p1) ELSE_CORRELATED_ERROR(p2) ... ELSE_CORRELATED_ERROR(pk),
    computes the joint probability distribution over 2^k outcomes.

    Since errors are mutually exclusive, only outcomes with at most one bit set
    have non-zero probability.
    - ``probs[0]`` is the probability that no branch fires.
    - ``probs[1 << i]`` is the probability that branch ``i`` fires after all
      previous branches did not fire.

    Args:
        probabilities: List of conditional probabilities [p1, p2, ..., pk]

    Returns:
        Array of shape (2^k,) with probabilities for each outcome.

    """
    k = len(probabilities)
    probs = np.zeros(2**k, dtype=np.float64)

    no_error_so_far = 1.0
    for i, p in enumerate(probabilities):
        probs[1 << i] = no_error_so_far * p
        no_error_so_far *= 1 - p

    probs[0] = no_error_so_far
    return probs

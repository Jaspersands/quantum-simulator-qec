# (c) Copyright Riverlane 2020-2025.
# modified for stabilizer_qec: its Stim is stabilizer_qec.deltakit_stim and its PyMatching this package's, and deltakit_{circuit,core,decode,explorer} are stabilizer_qec.deltakit.{circuit,core,decode,explorer}.
"""Helpers for constructing code fixtures in tests."""

from collections.abc import Callable

import numpy as np
import numpy.typing as npt

from stabilizer_qec.deltakit.explorer.codes import RotatedPlanarCode, UnrotatedPlanarCode
from stabilizer_qec.deltakit.explorer.codes._bivariate_bicycle_code import BivariateBicycleCode


def bivariate_bicycle_parity_check_matrices(
    param_l: int, param_m: int, m_A_powers: list[int], m_B_powers: list[int]
) -> tuple[npt.NDArray[np.integer], npt.NDArray[np.integer]]:
    """Construct BB parity check matrices without constructing a code object.

    Args:
        param_l: Parameter ``l`` as in the IBM paper.
        param_m: Parameter ``m`` as in the IBM paper.
        m_A_powers: Powers defining matrix ``A``.
        m_B_powers: Powers defining matrix ``B``.

    Returns:
        The X and Z parity check matrices.
    """
    matrices = BivariateBicycleCode._construct_bb_matrices(
        param_l, param_m, m_A_powers, m_B_powers
    )
    return matrices.Hx, matrices.Hz


def planar_code_parity_check_matrices(
    code_factory: Callable[[], RotatedPlanarCode | UnrotatedPlanarCode],
) -> tuple[npt.NDArray[np.integer], npt.NDArray[np.integer]]:
    """Construct planar code parity check matrices from a code object.

    Args:
        code_factory: Factory for the planar code fixture.

    Returns:
        The X and Z parity check matrices.
    """
    return code_factory().parity_check_matrices

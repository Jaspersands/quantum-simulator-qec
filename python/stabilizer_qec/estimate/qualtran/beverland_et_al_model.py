# Ported from Qualtran (quantumlib/Qualtran 0.7.0, qualtran/surface_code/beverland_et_al_model.py; Copyright 2023 Google LLC,
# Apache License 2.0; see LICENSE in the parent directory), modified for stabilizer_qec; see _models.py.
"""Qualtran's ``qualtran.surface_code.beverland_et_al_model``. Beverland et al.'s estimate (arXiv:2211.07629): the minimum time steps, the code distance, the T states and the discrete logical gates."""

from ._models import (  # noqa: F401
    code_distance,
    GateCounts,
    minimum_time_steps,
    n_discrete_logical_gates,
    t_states,
)

__all__ = ["code_distance", "minimum_time_steps", "n_discrete_logical_gates", "t_states"]

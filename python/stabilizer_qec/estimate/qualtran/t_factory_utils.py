# Ported from Qualtran (quantumlib/Qualtran 0.7.0, qualtran/surface_code/t_factory_utils.py; Copyright 2024 Google LLC,
# Apache License 2.0; see LICENSE in the parent directory), modified for stabilizer_qec; see _models.py.
"""Qualtran's ``qualtran.surface_code.t_factory_utils``. The 15-to-1 factory's channels: `NoisyPauliRotation` and `storage_error` (and `PauliFlipChannel`, which here stands in for Cirq's channel of that name)."""

from ._models import (  # noqa: F401
    _rotation_unitary,
    NoisyPauliRotation,
    PauliFlipChannel,
    storage_error,
)

__all__ = ["NoisyPauliRotation", "PauliFlipChannel", "storage_error"]

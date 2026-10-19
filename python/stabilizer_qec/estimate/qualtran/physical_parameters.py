# Ported from Qualtran (quantumlib/Qualtran 0.7.0, qualtran/surface_code/physical_parameters.py; Copyright 2023 Google LLC,
# Apache License 2.0; see LICENSE in the parent directory), modified for stabilizer_qec; see _models.py.
"""Qualtran's ``qualtran.surface_code.physical_parameters``. A device's physical error rate and cycle time: `PhysicalParameters`."""

from ._models import (  # noqa: F401
    PhysicalParameters,
)

__all__ = ["PhysicalParameters"]

# Ported from Qualtran (quantumlib/Qualtran 0.7.0, qualtran/surface_code/multi_factory.py; Copyright 2023 Google LLC,
# Apache License 2.0; see LICENSE in the parent directory), modified for stabilizer_qec; see _models.py.
"""Qualtran's ``qualtran.surface_code.multi_factory``. Several factories of one kind at once: `MultiFactory`."""

from ._models import (  # noqa: F401
    MagicStateFactory,
    MultiFactory,
)

__all__ = ["MultiFactory"]

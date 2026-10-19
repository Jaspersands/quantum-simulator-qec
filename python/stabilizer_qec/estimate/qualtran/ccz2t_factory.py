# Ported from Qualtran (quantumlib/Qualtran 0.7.0, qualtran/surface_code/ccz2t_factory.py; Copyright 2024 Google LLC,
# Apache License 2.0; see LICENSE in the parent directory), modified for stabilizer_qec; see _models.py.
"""Qualtran's ``qualtran.surface_code.ccz2t_factory``. Gidney and Fowler's catalyzed CCZ-to-2T factory: `CCZ2TFactory`."""

from ._models import (  # noqa: F401
    CCZ2TFactory,
    MagicStateFactory,
)

__all__ = ["CCZ2TFactory"]

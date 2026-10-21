# Ported from tqec (tqec/tqec 0.2.0, src/tqec/compile/specs/library/generators/__init__.py; Apache License 2.0, see LICENSE in stabilizer_qec/tqec),
# modified for stabilizer_qec.
"""Provides an implementation of various standard templates found when using topological quantum
error correction.
"""

from __future__ import annotations

from .fixed_boundary import (
    FixedBoundaryConventionGenerator as FixedBoundaryConventionGenerator,
)
from .fixed_bulk import FixedBulkConventionGenerator as FixedBulkConventionGenerator

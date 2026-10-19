# Ported from Qualtran (quantumlib/Qualtran 0.7.0, qualtran/surface_code/qec_scheme.py; Copyright 2023 Google LLC,
# Apache License 2.0; see LICENSE in the parent directory), modified for stabilizer_qec; see _models.py.
"""Qualtran's ``qualtran.surface_code.qec_scheme``. A QEC scheme's logical error rate: `QECScheme`, and `LogicalErrorModel` (a scheme at one physical error rate)."""

from ._models import (  # noqa: F401
    LogicalErrorModel,
    QECScheme,
)

__all__ = ["LogicalErrorModel", "QECScheme"]

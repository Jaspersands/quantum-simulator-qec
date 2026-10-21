# Ported from tqec (tqec/tqec 0.2.0, src/tqec/utils/array.py; Apache License 2.0, see LICENSE in stabilizer_qec/tqec),
# modified for stabilizer_qec.
from __future__ import annotations

import numpy
import numpy.typing as npt

from stabilizer_qec.tqec._errors import TQECError


def to2dlist(array: npt.NDArray[numpy.int_]) -> list[list[int]]:
    """Transform a 2D numpy array into a 2D array using Python list."""
    if len(array.shape) != 2:
        raise TQECError(f"Cannot transform a {len(array.shape)}-dimensional array to 2D.")
    return [[int(v) for v in line] for line in array]

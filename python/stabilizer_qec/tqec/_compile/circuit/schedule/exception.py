# Ported from tqec (tqec/tqec 0.2.0, src/tqec/circuit/schedule/exception.py; Apache License 2.0, see LICENSE in stabilizer_qec/tqec),
# modified for stabilizer_qec.
"""Defines :class:`~tqec.circuit.schedule.exception.ScheduleError`."""

from __future__ import annotations

from stabilizer_qec.tqec._errors import TQECError


class ScheduleError(TQECError):
    pass

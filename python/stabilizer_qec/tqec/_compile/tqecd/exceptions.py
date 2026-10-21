# Ported from tqecd (tqec/tqecd 0.2.1, src/tqecd/exceptions.py; Apache License 2.0, see LICENSE here),
# modified for stabilizer_qec.
from __future__ import annotations

class TQECDException(Exception):
    pass


class TQECDWarning(Warning):
    pass

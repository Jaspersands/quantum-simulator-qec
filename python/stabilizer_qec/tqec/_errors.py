# Ported from tqec (tqec/tqec 0.2.0, src/tqec/utils/exceptions.py; Apache License 2.0; see LICENSE in this directory),
# modified for stabilizer_qec.
"""tqec's exceptions."""


class TQECError(ValueError):
    """An error in a block graph or its use, as tqec raises it (a ``ValueError`` here)."""


class TQECWarning(UserWarning):
    """A warning from tqec's algorithms."""

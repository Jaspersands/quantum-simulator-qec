# Ported from deltakit-explorer 0.9.4 (Copyright Riverlane, Apache License 2.0; see THIRD_PARTY_NOTICES.md),
# modified for stabilizer_qec: numpy in place of scipy and uncertainties (each used when installed), Python 3.9 and numpy 1.x.
from __future__ import annotations

from typing import NamedTuple

from typing import Any as UFloat


class Estimate(NamedTuple):
    """A scalar estimate paired with its standard deviation.

    The fields unpack in ``(value, stddev)`` order, so call sites that previously
    received a plain 2-tuple keep working unchanged.

    Attributes:
        value: Estimated nominal value.
        stddev: Standard deviation of the estimate.
    """

    value: float
    stddev: float

    @classmethod
    def from_ufloat(cls, quantity: UFloat) -> Estimate:
        """Build an :class:`Estimate` from an ``uncertainties`` quantity.

        Args:
            quantity: An ``uncertainties`` value exposing ``nominal_value`` and
                ``std_dev``.

        Returns:
            An :class:`Estimate` carrying the quantity's nominal value and standard
            deviation as plain floats.
        """
        return cls(float(quantity.nominal_value), float(quantity.std_dev))

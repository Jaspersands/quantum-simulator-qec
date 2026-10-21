# Ported from tqec (tqec/tqec 0.2.0, src/tqec/plaquette/rpng/translators/base.py; Apache License 2.0, see LICENSE in stabilizer_qec/tqec),
# modified for stabilizer_qec.
from __future__ import annotations

from abc import ABC, abstractmethod

from stabilizer_qec.tqec._compile.plaquette.plaquette import Plaquette
from stabilizer_qec.tqec._compile.plaquette.rpng import RPNGDescription


class RPNGTranslator(ABC):
    """Base interface for classes capable of generating a plaquette from a RPNG description."""

    @abstractmethod
    def translate(
        self,
        rpng_description: RPNGDescription,
    ) -> Plaquette:
        """Generate the plaquette corresponding to the provided ``RPNG`` description.

        Args:
            rpng_description: description of the plaquette to generate.

        Returns:
            a valid implementation of the provided
            :class:`~tqec.plaquette.rpng.RPNGDescription` instance.

        """
        pass

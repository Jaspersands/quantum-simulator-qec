# Ported from tqec (tqec/tqec 0.2.0, src/tqec/compile/blocks/layers/atomic/base.py; Apache License 2.0, see LICENSE in stabilizer_qec/tqec),
# modified for stabilizer_qec.
from __future__ import annotations

from collections.abc import Mapping


from stabilizer_qec.tqec._compile.blocks.enums import TemporalBlockBorder
from stabilizer_qec.tqec._compile.blocks.layers.spatial import WithSpatialFootprint
from stabilizer_qec.tqec._compile.blocks.layers.temporal import WithTemporalFootprint
from stabilizer_qec.tqec._errors import TQECError
from stabilizer_qec.tqec.scale import LinearFunction


class BaseLayer(WithSpatialFootprint, WithTemporalFootprint):
    """Base class representing a "layer".

    A "layer" is a scalable and repeatable unit of a quantum error correction
    circuit. In most cases, it corresponds to a single round of the syndrome
    extraction circuit. A base layer occupies a specific spacetime footprint,
    where the spatial footprint scales with the layer circuit, while the temporal
    footprint remains fixed at ``1``, representing a single layer. To stack layers
    over time or repeat them in a scalable manner, use
    :class:`~tqec.compile.blocks.layers.composed.base.BaseComposedLayer` instead.

    """

    @property
    def scalable_timesteps(self) -> LinearFunction:
        # By definition of a "layer":
        return LinearFunction(0, 1)

    def with_temporal_borders_replaced(
        self, border_replacements: Mapping[TemporalBlockBorder, BaseLayer | None]
    ) -> BaseLayer | None:
        if not border_replacements:
            return self
        if len(border_replacements) > 1 and any(
            replacement is not None for replacement in border_replacements.values()
        ):
            raise TQECError(
                "Unclear semantic: trying to replace the two temporal borders of "
                "an atomic layer that, by definition, only contain one layer."
            )
        return next(iter(border_replacements.values()))

    def get_temporal_layer_on_border(self, border: TemporalBlockBorder) -> BaseLayer:
        return self

# Ported from tqec (tqec/tqec 0.2.0, src/tqec/compile/blocks/layers/composed/base.py; Apache License 2.0, see LICENSE in stabilizer_qec/tqec),
# modified for stabilizer_qec.
from __future__ import annotations

from abc import abstractmethod
from collections.abc import Iterable
from typing import TYPE_CHECKING

from stabilizer_qec.tqec._compile.blocks.layers.atomic.base import BaseLayer
from stabilizer_qec.tqec._compile.blocks.layers.spatial import WithSpatialFootprint
from stabilizer_qec.tqec._compile.blocks.layers.temporal import WithTemporalFootprint
from stabilizer_qec.tqec.scale import LinearFunction

if TYPE_CHECKING:
    from stabilizer_qec.tqec._compile.blocks.layers.composed.sequenced import SequencedLayers


class BaseComposedLayer(WithSpatialFootprint, WithTemporalFootprint):
    """Base class representing a composed "layer".

    A composed layer is defined as a sequence (in time) of atomic layers. As
    such, composed layers are expected to have either a scalable time footprint
    (i.e., that grows with ``k``) or a constant time footprint that is strictly
    greater than ``1``.

    """

    @abstractmethod
    def all_layers(self, k: int) -> Iterable[BaseLayer]:
        """Return all the base layers represented by the instance.

        Returns:
            All the base layers represented by the instance. The returned
            iterable should have as many entries as ``self.timesteps(k)``.

        """
        pass

    @abstractmethod
    def to_sequenced_layer_with_schedule(
        self, schedule: tuple[LinearFunction, ...]
    ) -> SequencedLayers:
        """Split ``self`` into a :class:`.SequencedLayers` instance with the provided schedule.

        Args:
            schedule: duration of each of the layers in the returned :class:`.SequencedLayers`
                instance.

        Returns:
            an instance of :class:`.SequencedLayers` that is equivalent to ``self`` (same duration,
            same layers applied, ...) and that has the provided ``schedule``.

        """
        pass

# Ported from tqec (tqec/tqec 0.2.0, src/tqec/interop/pyzx/synthesis/strategy.py; Apache
# License 2.0, see LICENSE in the tqec directory), modified for stabilizer_qec.
"""Strategies for synthesising a block graph from a ZX graph."""

from __future__ import annotations

from collections.abc import Mapping
from enum import Enum
from typing import Any, Optional

from ....block_graph import BlockGraph
from ....position import Position3D
from .positioned import positioned_block_synthesis


class SynthesisStrategy(Enum):
    """Strategy for synthesising a block graph from a ZX graph."""

    POSITIONED = "POSITIONED"
    """Each vertex at an explicit 3D position, its cube's kind inferred from its neighbours,
    each edge a pipe between the cubes of its ends."""


def block_synthesis(zx_graph, strategy: SynthesisStrategy = SynthesisStrategy.POSITIONED, *, positions: Optional[Mapping[int, Position3D]] = None, **_kwargs: Any) -> BlockGraph:
    """A block graph from a ZX graph, as tqec synthesises it: ``zx_graph`` a PyZX graph with
    ``positions`` (POSITIONED, the one strategy), or this package's positioned ZX graph, whose
    own positions are used when ``positions`` is not given."""
    from ....correlation import ZXGraph
    from ..positioned import PositionedZX

    if strategy == SynthesisStrategy.POSITIONED:
        if isinstance(zx_graph, ZXGraph):
            if positions is None:
                return positioned_block_synthesis(zx_graph)
            zx_graph = zx_graph.g
        if positions is None:
            raise ValueError("The POSITIONED strategy requires specifying positions, but None was given.")
        return positioned_block_synthesis(PositionedZX(zx_graph, positions))
    raise ValueError(f"Unknown synthesis strategy {strategy!r}.")

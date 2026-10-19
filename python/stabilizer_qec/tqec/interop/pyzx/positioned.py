# Ported from tqec (tqec/tqec 0.2.0, src/tqec/interop/pyzx/positioned.py; Apache License 2.0,
# see LICENSE in the tqec directory), modified for stabilizer_qec: this package's ZX graph,
# read from the PyZX graph.
"""tqec's ``PositionedZX``: a PyZX graph with a 3D position for each vertex."""

from __future__ import annotations

from collections.abc import Mapping
from fractions import Fraction

from ..._errors import TQECError
from ...correlation import ZXGraph
from ...position import Position3D


class PositionedZX(ZXGraph):
    """A PyZX graph ``g`` with 3D ``positions``, checked as tqec checks it. It is this
    package's ZX graph (``BlockGraph.to_zx_graph()`` is one too, as tqec's is a
    ``PositionedZX``), and ``g`` is the PyZX graph it was made from."""

    def __init__(self, g, positions: Mapping[int, Position3D]) -> None:
        super().__init__()
        self.check_preconditions(g, positions)
        self.__dict__.update(ZXGraph.from_pyzx(g, positions).__dict__)

    @staticmethod
    def check_preconditions(g, positions: Mapping[int, Position3D]) -> None:
        """tqec's preconditions, in its order (a TQECError when one fails): the vertices and the
        positions match; neighbours are adjacent; every spider is a phase-free Z or X, an S
        (Z of phase 1/2) or a boundary; boundaries and S spiders dangle, S spiders along time;
        no 3D corner."""
        from pyzx.utils import VertexType

        if g.vertex_set() != set(positions.keys()):
            raise TQECError("The vertex IDs in the ZX graph and the positions do not match.")
        for s, t in g.edge_set():
            ps, pt = positions[s], positions[t]
            if not ps.is_neighbour(pt):
                raise TQECError(f"The 3D positions of the endpoints of the edge {s}--{t} must be neighbors, but got {ps} and {pt}.")
        for v in g.vertices():
            vt = g.type(v)
            phase = g.phase(v)
            if (vt, phase) not in [(VertexType.Z, 0), (VertexType.X, 0), (VertexType.Z, Fraction(1, 2)), (VertexType.BOUNDARY, 0)]:
                raise TQECError(f"Unsupported vertex type and phase: {vt} and {phase}.")
            if vt == VertexType.BOUNDARY or phase == Fraction(1, 2):
                if g.vertex_degree(v) != 1:
                    raise TQECError(f"Boundary or Z(1/2) spider must be dangling, but got {len(g.neighbors(v))} neighbors.")
                if phase == Fraction(1, 2):
                    nb = next(iter(g.neighbors(v)))
                    vp, nbp = positions[v], positions[nb]
                    if abs(nbp.z - vp.z) != 1:
                        raise TQECError(f"Z(1/2) spider must connect to the time direction, but Z(1/2) at {vp} connects to {nbp}.")
        for v in g.vertices():
            vp = positions[v]
            if len({_get_direction(vp, positions[u]) for u in g.neighbors(v)}) == 3:
                raise TQECError(f"ZX graph has a 3D corner at node {v}.")


def _get_direction(p1: Position3D, p2: Position3D):
    from ...position import Direction3D

    if p1.x != p2.x:
        return Direction3D.X
    if p1.y != p2.y:
        return Direction3D.Y
    return Direction3D.Z

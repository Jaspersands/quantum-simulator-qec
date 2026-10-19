# Ported from tqec (tqec/tqec 0.2.0, src/tqec/interop/pyzx/plot.py; Apache License 2.0, see
# LICENSE in the tqec directory), modified for stabilizer_qec: drawn from this package's
# positioned ZX graph (``BlockGraph.to_zx_graph()``) without PyZX, matplotlib imported to draw.
"""Draw positioned ZX graphs and correlation surfaces on matplotlib's 3D axes, as tqec draws
them (``pip install matplotlib``; PyZX only for :func:`pyzx_draw_positioned_zx_3d`)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy

from ...collada import TQECColor
from ...enums import Pauli
from ...position import Position3D

if TYPE_CHECKING:
    from ...correlation import CorrelationSurface, ZXGraph


def _positions_array(*positions: Position3D) -> numpy.ndarray:
    return numpy.array([(p.x, p.y, p.z) for p in positions]).T


def _graph(graph) -> "ZXGraph":
    """This package's ZX graph for ``graph`` (one of its own, or tqec's ``PositionedZX``)."""
    from ...correlation import ZXGraph

    return graph if isinstance(graph, ZXGraph) else ZXGraph.from_pyzx(graph.g, graph.positions)


def draw_positioned_zx_graph_on(graph: "ZXGraph", ax, *, node_size: int = 400, hadamard_size: int = 200, edge_width: int = 1) -> None:
    """Draw the positioned ZX graph on a 3D matplotlib axes: spiders coloured by basis (tqec's
    colours; boundaries not drawn), edges grey, a yellow square on each Hadamard edge."""
    graph = _graph(graph)
    pmap = graph.positions
    vis_nodes = [n for n in graph.vertices() if graph.kind(n) is not Pauli.I]
    vis_nodes_array = _positions_array(*[pmap[n] for n in vis_nodes])
    if vis_nodes_array.size > 0:
        ax.scatter(*vis_nodes_array, s=node_size, c=[TQECColor(str(graph.kind(n))).rgba.as_floats() for n in vis_nodes], alpha=1.0, edgecolors="black")
    for u, v in graph.edges():
        pos_array = _positions_array(pmap[u], pmap[v])
        ax.plot(*pos_array, color="tab:gray", linewidth=edge_width)
        if graph.is_hadamard(u, v):
            hadamard_position = numpy.mean(pos_array, axis=1)
            ax.scatter(*hadamard_position, s=hadamard_size, c="yellow", alpha=1.0, edgecolors="black", marker="s")
    ax.grid(False)
    for dim in (ax.xaxis, ax.yaxis, ax.zaxis):
        dim.set_ticks([])
    x_limits, y_limits, z_limits = ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()
    plot_radius = 0.5 * max(abs(limits[1] - limits[0]) for limits in [x_limits, y_limits, z_limits])
    ax.set_xlim3d([numpy.mean(x_limits) - plot_radius, numpy.mean(x_limits) + plot_radius])
    ax.set_ylim3d([numpy.mean(y_limits) - plot_radius, numpy.mean(y_limits) + plot_radius])
    ax.set_zlim3d([numpy.mean(z_limits) - plot_radius, numpy.mean(z_limits) + plot_radius])


def draw_correlation_surface_on(correlation_surface: "CorrelationSurface", graph: "ZXGraph", ax, correlation_edge_width: int = 3) -> None:
    """Draw the correlation surface on a 3D matplotlib axes: each half edge it covers, from a
    spider to the edge's middle, in the colour of its Pauli."""
    if correlation_surface.is_single_node:
        return
    pmap = _graph(graph).positions
    for (u, v), pauli in correlation_surface.half_edge_paulis().items():
        up, vp = pmap[u], pmap[v]
        pos_array = _positions_array(up, vp)
        middle = numpy.mean(pos_array, axis=1).reshape(3, 1)
        start = _positions_array(up)
        ax.plot(*numpy.hstack([start, middle]), color=TQECColor(pauli).rgba.as_floats(), linewidth=correlation_edge_width)


def plot_positioned_zx_graph(graph: "ZXGraph", *, figsize: tuple[float, float] = (5, 6), title: Optional[str] = None, node_size: int = 400, hadamard_size: int = 200, edge_width: int = 1):
    """A matplotlib figure of the positioned ZX graph: ``(figure, axes)``."""
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    draw_positioned_zx_graph_on(graph, ax, node_size=node_size, hadamard_size=hadamard_size, edge_width=edge_width)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig, ax


def pyzx_draw_positioned_zx_3d(g: "ZXGraph", id_labels: bool = True, pauli_web=None) -> None:
    """Draw the positioned ZX graph with PyZX's ``draw_3d`` (PyZX needed)."""
    from pyzx import draw_3d

    graph = _graph(g)
    plot_g = graph.g.clone()
    for v in plot_g.vertices():
        position = graph.positions[v]
        plot_g.set_qubit(v, position.x)
        plot_g.set_row(v, position.y)
        plot_g.set_vdata(v, "z", position.z)
    draw_3d(plot_g, labels=id_labels, pauli_web=pauli_web)

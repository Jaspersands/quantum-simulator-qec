# Ported from tqec (tqec/tqec 0.2.0, src/tqec/simulation/plotting/inset.py; Apache License 2.0,
# see LICENSE in the tqec directory), modified for stabilizer_qec: this package's sinter and
# positioned ZX graph, matplotlib imported to draw.
"""Insets for threshold plots: the observable's correlation surface on the computation's ZX
graph, and a zoom on the threshold (matplotlib needed)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from .... import sinter
from ...interop.pyzx.plot import draw_correlation_surface_on, draw_positioned_zx_graph_on

if TYPE_CHECKING:
    from ...correlation import CorrelationSurface, ZXGraph


def add_inset_axes3d(ax_target, bounds: tuple[float, float, float, float]):
    """A 3D inset in ``ax_target`` (``ax.inset_axes`` with the 3D projection)."""
    from mpl_toolkits.mplot3d.axes3d import Axes3D

    ax = ax_target.inset_axes(bounds, projection="3d")
    assert isinstance(ax, Axes3D)
    return ax


def plot_observable_as_inset(ax_target, zx_graph: "ZXGraph", observable: "CorrelationSurface", bounds: tuple[float, float, float, float] = (0.55, 0.0, 0.45, 0.45)) -> None:
    """Draw ``observable`` on ``zx_graph`` (``block_graph.to_zx_graph()``) in an inset of
    ``ax_target`` at ``bounds`` (``(x0, y0, width, height)`` in axes coordinates)."""
    inset_ax = add_inset_axes3d(ax_target, bounds)
    inset_ax.set_axis_off()
    draw_positioned_zx_graph_on(zx_graph, inset_ax, node_size=50)
    draw_correlation_surface_on(observable, zx_graph, inset_ax)
    inset_ax.set_facecolor((0.0, 0.0, 0.0, 0.0))


def plot_threshold_as_inset(ax_target, stats: list["sinter.TaskStats"], zoom_bounds: tuple[float, float, float, float], threshold: Optional[float] = None, inset_bounds: tuple[float, float, float, float] = (0.53, 0.45, 0.4, 0.4)) -> None:
    """A log-log inset of ``ax_target`` zoomed on ``zoom_bounds`` (``(x0, y0, x1, y1)`` in
    physical and logical error rate) with ``stats`` plotted by ``sinter.plot_error_rate``
    (grouped by distance), and a red line at ``threshold`` if given."""
    from matplotlib.ticker import ScalarFormatter

    inset_ax = ax_target.inset_axes(inset_bounds, xlim=(zoom_bounds[0], zoom_bounds[2]), ylim=(zoom_bounds[1], zoom_bounds[3]))
    inset_ax.loglog()
    inset_ax.invert_yaxis()
    inset_ax.yaxis.set_label_position("right")
    inset_ax.yaxis.tick_right()
    inset_ax.yaxis.set_minor_formatter(ScalarFormatter())
    inset_ax.yaxis.set_major_formatter(ScalarFormatter())
    inset_ax.tick_params(axis="y", which="minor", labelsize=7)
    inset_ax.tick_params(axis="y", which="major", labelsize=7)
    inset_ax.tick_params(axis="x", which="major", labelsize=7)
    inset_ax.tick_params(axis="x", which="minor", labelsize=7)
    inset_ax.grid(which="both", axis="both")
    # matplotlib 3.10 returns an InsetIndicator (unpacking it is deprecated), earlier ones a pair.
    indicator = ax_target.indicate_inset_zoom(inset_ax, edgecolor="black", alpha=0.8)
    lower_left, upper_left, lower_right, upper_right = indicator.connectors if hasattr(indicator, "connectors") else indicator[1]
    lower_left.set_visible(True)
    upper_right.set_visible(True)
    lower_right.set_visible(False)
    upper_left.set_visible(False)

    sinter.plot_error_rate(
        ax=inset_ax,
        stats=stats,
        x_func=lambda stat: stat.json_metadata["p"],
        group_func=lambda stat: stat.json_metadata["d"],
        plot_args_func=lambda index, group_key, group_stats: {"markersize": 3},
    )
    if threshold is not None:
        inset_ax.axvline(threshold, color="r")
        inset_ax.text(threshold * 1.02, zoom_bounds[1] * 0.96, f"{threshold:.3g}", fontsize=6, color="r", horizontalalignment="left", verticalalignment="top")

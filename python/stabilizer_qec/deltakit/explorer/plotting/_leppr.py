# (c) Copyright Riverlane 2020-2026. All rights reserved.
# modified for stabilizer_qec: its Stim is stabilizer_qec.deltakit_stim and its PyMatching this package's, and deltakit_{circuit,core,decode,explorer} are stabilizer_qec.deltakit.{circuit,core,decode,explorer}.
"""Plotting helpers for logical-error-probability-per-round results."""

from __future__ import annotations

from stabilizer_qec.deltakit.core.plotting.colours import RIVERLANE_PLOT_COLOURS
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from stabilizer_qec.deltakit.explorer.plotting._utils import get_figure_and_axes
from stabilizer_qec.deltakit.explorer.plotting.results import (
    LogicalErrorProbabilityPerRoundResult as LEPPRResult,
)


def plot_leppr(
    leppr_result: LEPPRResult,
    *,
    fig: Figure | None = None,
    ax: Axes | None = None,
    title: str | None = None,
) -> tuple[Figure, Axes]:
    """Plot an interpolated logical error probability per round result.

    LEPPR stands for logical error probability per round. This specialised
    plotter owns only the LEPPR-specific rendering logic. It expects a
    ready-to-plot
    :class:`~deltakit_explorer.plotting.results.LogicalErrorProbabilityPerRoundResult`.
    Higher-level data preparation, such as interpolation from raw LEPPR data,
    should be handled by :func:`deltakit_explorer.plotting.plot` before dispatch.

    Args:
        leppr_result: Interpolated logical error probability per round result
            to plot.
        fig: A matplotlib Figure object to plot on. If None, a new figure
            will be created. Default is None.
        ax: A matplotlib Axes object to plot on. If None, a new axes will
            be created. Default is None.
        title: An optional custom title for the plot. If None, the default
            LEPPR title will be used.

    Returns:
        The matplotlib Figure and Axes objects containing the plot.

    Examples:

        Plotting an interpolated logical error probability per round result::

            from stabilizer_qec.deltakit.explorer.plotting import interpolate_leppr, plot_leppr

            leppr_result = interpolate_leppr(leppr_data)
            fig, ax = plot_leppr(leppr_result)

    """
    fig, ax = get_figure_and_axes(fig, ax)
    ax.plot(
        leppr_result.rounds,
        leppr_result.interpolated,
        label=leppr_result.fit_label,
        color=RIVERLANE_PLOT_COLOURS[1],
    )
    ax.fill_between(
        leppr_result.rounds,
        leppr_result.lower_boundary,
        leppr_result.upper_boundary,
        label=leppr_result.confidence_interval_label,
        color=RIVERLANE_PLOT_COLOURS[0],
        alpha=0.2,
    )
    ax.set_title(title if title is not None else "Logical Error Probability per Round")
    ax.set_xlabel("Rounds")
    ax.set_ylabel("Logical Error Probability")
    ax.legend()
    return fig, ax

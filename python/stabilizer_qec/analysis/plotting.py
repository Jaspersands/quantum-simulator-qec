"""Plots of experiment analysis, after deltakit's (``deltakit_explorer.plotting``): Λ across
distances, the logical error per round, defect rates over rounds, and detection probabilities
laid out on the patch. Detectors are placed by their coordinates, the last being the time (as
in Stim's generated circuits), so no code object is needed. Needs matplotlib."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = ["defect_rates", "plot_defect_rates", "plot_detection_probability_on_patch", "plot_lambda", "plot_leppr"]


def _axes(fig: Any, ax: Any) -> Tuple[Any, Any]:
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots() if fig is None else (fig, fig.add_subplot())
    return ax.figure if fig is None else fig, ax


def plot_lambda(lambda_data: Any, *, num_sigmas: int = 3, num_points: int = 200, fig: Any = None, ax: Any = None, title: Optional[str] = None) -> Tuple[Any, Any]:
    """The logical error per round of each distance, with the fitted
    ``1/Λ0 · Λ^-((d+1)/2)`` and its ``num_sigmas`` band (from ``calculate_lambda_*``)."""
    fig, ax = _axes(fig, ax)
    d = np.asarray(lambda_data.distances)
    ax.errorbar(d, lambda_data.leppr, yerr=num_sigmas * np.asarray(lambda_data.leppr_std), fmt="o", capsize=3, label="data")
    xs = np.linspace(d.min(), d.max(), num_points)

    def model(lam: float, lam0: float) -> np.ndarray:
        return 1 / lam0 * lam ** (-(xs + 1) / 2)

    ax.plot(xs, model(lambda_data.lambda_, lambda_data.lambda0), label=f"Λ = {lambda_data.lambda_:.3g} ± {lambda_data.lambda_std:.2g}")
    lo = model(lambda_data.lambda_ + num_sigmas * lambda_data.lambda_std, lambda_data.lambda0 + num_sigmas * lambda_data.lambda0_std)
    hi = model(max(lambda_data.lambda_ - num_sigmas * lambda_data.lambda_std, 1e-12), max(lambda_data.lambda0 - num_sigmas * lambda_data.lambda0_std, 1e-12))
    ax.fill_between(xs, np.minimum(lo, hi), np.maximum(lo, hi), alpha=0.2)
    ax.set_yscale("log")
    ax.set_xlabel("distance")
    ax.set_ylabel("logical error per round")
    ax.legend()
    if title:
        ax.set_title(title)
    return fig, ax


def plot_leppr(num_rounds: Sequence[int], logical_error_probabilities: Sequence[float], logical_error_probabilities_stddev: Sequence[float], leppr_data: Any, *, num_sigmas: int = 3, num_points: int = 200, fig: Any = None, ax: Any = None, title: Optional[str] = None) -> Tuple[Any, Any]:
    """Logical error probability against rounds, with the fitted
    ``(1 - (1 - 2 SPAM)(1 - 2 ε)^r) / 2`` (from ``compute_logical_error_per_round``)."""
    fig, ax = _axes(fig, ax)
    r = np.asarray(num_rounds)
    ax.errorbar(r, logical_error_probabilities, yerr=num_sigmas * np.asarray(logical_error_probabilities_stddev), fmt="o", capsize=3, label="data")
    xs = np.linspace(0, r.max(), num_points)
    eps, spam = leppr_data.leppr, leppr_data.spam_error
    ax.plot(xs, (1 - (1 - 2 * spam) * (1 - 2 * eps) ** xs) / 2, label=f"ε = {eps:.3g} ± {leppr_data.leppr_stddev:.2g}")
    ax.set_xlabel("rounds")
    ax.set_ylabel("logical error probability")
    ax.legend()
    if title:
        ax.set_title(title)
    return fig, ax


def defect_rates(detection_events: Any, model: Any) -> Dict[Tuple[float, ...], List[float]]:
    """Each check's defect rate in each round: detectors grouped by their coordinates but the
    last (the time), in order of time; ``detection_events`` is (shots, detectors)."""
    events = np.asarray(detection_events, dtype=bool)
    rates = events.mean(axis=0)
    groups: Dict[Tuple[float, ...], List[Tuple[float, float]]] = defaultdict(list)
    for d, coords in model.get_detector_coordinates().items():
        if coords:
            groups[tuple(coords[:-1])].append((coords[-1], float(rates[d])))
    return {k: [rate for _, rate in sorted(v)] for k, v in sorted(groups.items())}


def plot_defect_rates(rates: Dict[Tuple[float, ...], List[float]], *, fig: Any = None, ax: Any = None, title: Optional[str] = None) -> Tuple[Any, Any]:
    """Each check's defect rate against the round (from ``defect_rates``)."""
    fig, ax = _axes(fig, ax)
    for coord, series in rates.items():
        ax.plot(range(len(series)), series, marker=".", label=str(coord))
    ax.set_xlabel("round")
    ax.set_ylabel("defect rate")
    if len(rates) <= 12:
        ax.legend(fontsize="small")
    if title:
        ax.set_title(title)
    return fig, ax


def plot_detection_probability_on_patch(model: Any, detection_probabilities: Dict[Tuple[float, ...], List[float]], *, mode: str = "mean", round_index: Optional[int] = None, cmap: str = "viridis", vmin: Optional[float] = None, vmax: Optional[float] = None, fig: Any = None, ax: Any = None, show_colorbar: bool = True) -> Tuple[Any, Any]:
    """Each check's detection probability at its place on the patch: its ``mode`` (``"mean"``,
    ``"max"``, ``"min"`` or ``"median"``) over rounds, or that of ``round_index``.
    ``detection_probabilities`` is ``defect_rates``' result; ``model`` is accepted for
    deltakit's signature (deltakit takes the code)."""
    fig, ax = _axes(fig, ax)
    aggregate = {"mean": np.mean, "max": np.max, "min": np.min, "median": np.median}[mode]
    coords = [c for c in detection_probabilities if len(c) >= 2]
    values = [detection_probabilities[c][round_index] if round_index is not None else aggregate(detection_probabilities[c]) for c in coords]
    points = ax.scatter([c[0] for c in coords], [c[1] for c in coords], c=values, cmap=cmap, vmin=vmin, vmax=vmax, s=400, marker="s")
    ax.set_aspect("equal")
    ax.invert_yaxis()
    if show_colorbar:
        fig.colorbar(points, ax=ax, label="detection probability")
    return fig, ax

# Ported from Qualtran (quantumlib/Qualtran 0.7.0, qualtran/surface_code/ui.py; Copyright 2024 Google LLC, Apache
# License 2.0; see LICENSE in the parent directory), modified for stabilizer_qec: the estimates and figures of the
# Dash app without Dash, plotly or pandas; a figure is plotly's figure dict.
"""Qualtran's ``qualtran.surface_code.ui``: the estimates behind Qualtran's interactive QEC-overhead
app (``update`` and the helpers it calls) and its tables of QEC schemes, magic-state factories and
rotation models.

The app itself (its Dash layout, inputs and server) is not here. A figure comes back as a `Figure`:
plotly's figure dict (``{"data": [...], "layout": {...}}``) with the traces and layout
``plotly.express`` gives Qualtran's, less plotly's default template; ``Figure.to_plotly()`` makes it
a ``plotly.graph_objects.Figure`` when plotly is installed. `create_runtime_plot` draws its factory
counts from numpy's global random state as Qualtran's does, so the same seed gives the same plot.

>>> from stabilizer_qec.estimate.qualtran import ui
>>> out = ui.update(1e-4, 1e-3, ui._GIDNEY_FOWLER_MODEL, [10, 10, 10, 10, 10, 10], "GidneyFowler", "GidneyFowlerCCZ", 1, "BeverlandEtAlRotationCost")
>>> out[0].data[0]["values"], out[3], out[4], out[8]
([1470, 12736], ['Total Number of Toffoli gates'], '47.5', '5.968 ms')
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from . import beverland_et_al_model, gidney_fowler_model, rotation_cost_model
from ._models import (
    AlgorithmSummary,
    CCZ2TFactory,
    FastDataBlock,
    FifteenToOne,
    GateCounts,
    LogicalErrorModel,
    MagicStateFactory,
    MultiFactory,
    PhysicalParameters,
    QECScheme,
)

__all__ = ["Figure", "PreventUpdate", "compute_duration", "create_qubit_pie_chart", "create_runtime_plot", "format_duration", "get_objects", "min_num_factories", "total_magic", "update"]


class PreventUpdate(Exception):
    """Raised by `update` when an input is missing (None), as Dash's ``dash.exceptions.PreventUpdate`` is by
    Qualtran's: the app then keeps its last outputs."""


class Figure(dict):
    """A plotly figure as plotly's figure dict: ``{"data": [trace, ...], "layout": {...}}``."""

    def __init__(self, data=None, layout=None) -> None:
        super().__init__(data=list(data or []), layout=dict(layout or {}))

    @property
    def data(self) -> list:
        return self["data"]

    @property
    def layout(self) -> dict:
        return self["layout"]

    def to_plotly(self):
        """This figure as a ``plotly.graph_objects.Figure`` (plotly must be installed)."""
        try:
            import plotly.graph_objects as go
        except ImportError as ex:
            raise ImportError("Figure.to_plotly needs plotly: pip install plotly") from ex
        return go.Figure(self)


def get_objects(modules, object_type):
    """Get all objects of a given type from a list of modules."""
    for module in modules:
        for obj_name in dir(module):
            obj = getattr(module, obj_name)
            if isinstance(obj, object_type):
                yield obj_name, obj


_QEC_SCHEMES = {
    "BeverlandEtAl": QECScheme.make_beverland_et_al(),
    "GidneyFowler": QECScheme.make_gidney_fowler(),
}
_MAGIC_FACTORIES = {
    "FifteenToOne733": FifteenToOne(7, 3, 3),
    "FifteenToOne933": FifteenToOne(9, 3, 3),
    "GidneyFowlerCCZ": CCZ2TFactory(),
}
_ROTATION_MODELS = dict(get_objects([rotation_cost_model], rotation_cost_model.RotationCostModel))

_GIDNEY_FOWLER_MODEL = "GidneyFowler (arxiv:1812.01238)"
_BEVERLAND_MODEL = "Beverland et al (arxiv:2211.07629)"
_SUPPORTED_ESTIMATION_MODELS = [_GIDNEY_FOWLER_MODEL, _BEVERLAND_MODEL]


def _pie(sources: List[str], qubits: List[int]) -> Figure:
    # px.pie(DataFrame(source=..., qubits=...), values='qubits', names='source', title=...) with textinfo 'value'
    trace = {"domain": {"x": [0.0, 1.0], "y": [0.0, 1.0]}, "hovertemplate": "source=%{label}<br>qubits=%{value}<extra></extra>", "labels": list(sources), "legendgroup": "", "name": "", "showlegend": True, "values": [int(q) for q in qubits], "type": "pie", "textinfo": "value"}
    return Figure([trace], {"legend": {"tracegroupgap": 0}, "title": {"text": "Physical Qubit Utilization"}})


def _line(x_name: str, x: Sequence[float], y: Sequence[int], text: Sequence[str]) -> Figure:
    # px.line(DataFrame({'label': text, x_name: x, 'num qubits': y}), x=x_name, y='num qubits', text='label')
    trace = {"hovertemplate": f"{x_name}=%{{x}}<br>num qubits=%{{y}}<br>label=%{{text}}<extra></extra>", "legendgroup": "", "line": {"color": "#636efa", "dash": "solid"}, "marker": {"symbol": "circle"}, "mode": "lines+markers+text", "name": "", "orientation": "v", "showlegend": False, "text": list(text), "x": [float(v) for v in x], "xaxis": "x", "y": [int(v) for v in y], "yaxis": "y", "type": "scatter"}
    layout = {"xaxis": {"anchor": "y", "domain": [0.0, 1.0], "title": {"text": x_name}}, "yaxis": {"anchor": "x", "domain": [0.0, 1.0], "title": {"text": "num qubits"}}, "legend": {"tracegroupgap": 0}, "margin": {"t": 60}}
    return Figure([trace], layout)


def create_qubit_pie_chart(physical_error_rate: float, error_budget: float, estimation_model: str, algorithm: AlgorithmSummary, magic_factory: MagicStateFactory, magic_count: int, n_logical_gates: GateCounts) -> Figure:
    """Create a pie chart of the physical qubit utilization."""
    sources = ["logical qubits + routing overhead", "Magic State Distillation"]
    if estimation_model == _GIDNEY_FOWLER_MODEL:
        res, factory, _ = gidney_fowler_model.get_ccz2t_costs_from_grid_search(n_logical_gates=n_logical_gates, n_algo_qubits=int(algorithm.n_algo_qubits), phys_err=physical_error_rate, error_budget=error_budget, factory_iter=[MultiFactory(f, int(magic_count)) for f in gidney_fowler_model.iter_ccz2t_factories()])
        return _pie(sources, [res.footprint - factory.n_physical_qubits(), factory.n_physical_qubits()])
    multi_factory = MultiFactory(magic_factory, int(magic_count))
    # As Qualtran's: the data side is the fast block's tiles, not its physical qubits.
    return _pie(sources, [FastDataBlock.get_n_tiles(int(algorithm.n_algo_qubits)), multi_factory.n_physical_qubits()])


def format_duration(duration: Sequence[float]) -> Tuple[str, Sequence[float]]:
    """Returns a tuple of the format (unit, duration)

    Finds the best unit to report `duration` and assumes that `duration` is initially in us.
    """
    unit = "us"
    if duration[0] > 86400_000_000:
        duration = [d / 86400_000_000 for d in duration]
        unit = "days"
    elif duration[0] > 3600_000_000:
        duration = [d / 3600_000_000 for d in duration]
        unit = "hours"
    elif duration[0] > 60_000_000:
        duration = [d / 60_000_000 for d in duration]
        unit = "min"
    elif duration[0] > 1000_000:
        duration = [d / 1000_000 for d in duration]
        unit = "s"
    elif duration[0] > 1000:
        duration = [d / 1000 for d in duration]
        unit = "ms"
    return unit, duration


def create_runtime_plot(physical_error_rate: float, error_budget: float, estimation_model: str, algorithm: AlgorithmSummary, qec: QECScheme, magic_factory: MagicStateFactory, magic_count: int, rotation_model: rotation_cost_model.RotationCostModel, n_logical_gates: GateCounts) -> Tuple[Dict[str, Any], Figure]:
    """Creates the runtime figure and decides whether to display it or not.

    Currently displays the runtime plot for the Beverland model only. The factory counts after the first
    are drawn from numpy's global random state (``np.random.choice``), as Qualtran's are.
    """
    if estimation_model == _GIDNEY_FOWLER_MODEL:
        return {"display": "none"}, Figure()
    factory = MultiFactory(magic_factory, int(magic_count))
    c_min = beverland_et_al_model.minimum_time_steps(error_budget=error_budget, alg=algorithm, rotation_model=rotation_model)
    err_model = LogicalErrorModel(qec_scheme=qec, physical_error=physical_error_rate)
    factory_cycles = factory.n_cycles(n_logical_gates=n_logical_gates, logical_error_model=err_model)
    min_num_factories = int(np.ceil(factory_cycles / c_min))
    magic_counts = list(1 + np.random.choice(min_num_factories, replace=False, size=min(min_num_factories, 5)))
    magic_counts.sort(reverse=True)
    magic_counts = np.array(magic_counts)
    time_steps = np.ceil(factory_cycles / magic_counts)
    magic_counts[0] = min_num_factories
    time_steps[0] = c_min
    cds = [beverland_et_al_model.code_distance(error_budget=error_budget, time_steps=t, alg=algorithm, qec_scheme=qec, physical_error=physical_error_rate) for t in time_steps]
    # TODO (Qualtran's): plumb through PhysicalParameters
    cycle_time_us = PhysicalParameters.make_gidney_fowler().cycle_time_us
    duration = [cycle_time_us * t for t, d in zip(time_steps, cds)]
    unit, duration = format_duration(duration)
    duration_name = f"Duration ({unit})"
    num_qubits = FastDataBlock.get_n_tiles(int(algorithm.n_algo_qubits)) + factory.n_physical_qubits() * magic_counts
    labels = [f"code distance: {c}<br>time steps: {t:g}<br>num factories: {f}" for t, c, f in zip(time_steps, cds, magic_count * magic_counts)]
    return {"display": "block"}, _line(duration_name, duration, num_qubits, labels)


def update(physical_error_rate: float, error_budget: float, estimation_model: str, algorithm_data: Sequence[Any], qec_name: str, magic_name: str, magic_count: int, rotaion_model_name: str):
    """Updates the visualization: the app's nine outputs (the qubit pie chart; the runtime plot's style and
    figure; the magic-state count's name and value; the minimum factory count's style and value; the
    duration's style and value). ``algorithm_data`` is the qubits, measurements, Ts, Toffolis, rotations and
    rotation depth, in that order; an input of None raises `PreventUpdate`."""
    if any(x is None for x in [physical_error_rate, error_budget, *algorithm_data, magic_count]):
        raise PreventUpdate

    # TODO (Qualtran's): We implicitly rely on the order of the input components
    qubits, measurements, ts, toffolis, rotations, n_rotation_layers = algorithm_data
    algorithm = AlgorithmSummary(n_algo_qubits=qubits, n_logical_gates=GateCounts(measurement=measurements, t=ts, toffoli=toffolis, rotation=rotations), n_rotation_layers=n_rotation_layers)
    qec = _QEC_SCHEMES[qec_name]
    magic_factory = _MAGIC_FACTORIES[magic_name]
    rotation_model = _ROTATION_MODELS[rotaion_model_name]
    n_logical_gates = beverland_et_al_model.n_discrete_logical_gates(eps_syn=error_budget / 3, alg=algorithm, rotation_model=rotation_model)
    magic_count = int(magic_count)
    logical_err_model = LogicalErrorModel(qec_scheme=qec, physical_error=physical_error_rate)
    return (
        create_qubit_pie_chart(physical_error_rate, error_budget, estimation_model, algorithm, magic_factory, magic_count, n_logical_gates),
        *create_runtime_plot(physical_error_rate, error_budget, estimation_model, algorithm, qec, magic_factory, magic_count, rotation_model, n_logical_gates),
        *total_magic(estimation_model, n_logical_gates),
        *min_num_factories(logical_err_model, error_budget, estimation_model, algorithm, rotation_model, magic_factory, n_logical_gates),
        *compute_duration(physical_error_rate, error_budget, estimation_model, algorithm, rotation_model, magic_count, n_logical_gates),
    )


def total_magic(estimation_model: str, n_logical_gates: GateCounts) -> Tuple[List[str], str]:
    """Compute the number of magic states needed for the algorithm and their type."""
    total_t = n_logical_gates.total_t_count()
    total_ccz = total_t / 4
    if estimation_model == _GIDNEY_FOWLER_MODEL:
        return ["Total Number of Toffoli gates"], f"{total_ccz:g}"
    return ["Total Number of T gates"], f"{total_t:g}"


def min_num_factories(logical_error_model: LogicalErrorModel, error_budget: float, estimation_model: str, algorithm: AlgorithmSummary, rotation_model: rotation_cost_model.RotationCostModel, magic_factory: MagicStateFactory, n_logical_gates: GateCounts) -> Tuple[Dict[str, Any], int]:
    """The fewest factories that make the magic states within Beverland et al.'s minimum time steps (shown
    for that model only)."""
    if estimation_model == _GIDNEY_FOWLER_MODEL:
        return {"display": "none"}, 1
    c_min = beverland_et_al_model.minimum_time_steps(error_budget=error_budget, alg=algorithm, rotation_model=rotation_model)
    return {"display": "block"}, int(np.ceil(magic_factory.n_cycles(n_logical_gates=n_logical_gates, logical_error_model=logical_error_model) / c_min))


def compute_duration(physical_error_rate: float, error_budget: float, estimation_model: str, algorithm: AlgorithmSummary, rotation_model: rotation_cost_model.RotationCostModel, magic_count: int, n_logical_gates: GateCounts) -> Tuple[Dict[str, Any], str]:
    """Compute the duration of running the algorithm and whether to display the result or not.

    Currently displays the result only for GidneyFowler (arxiv:1812.01238).
    """
    if estimation_model == _GIDNEY_FOWLER_MODEL:
        res, _, _ = gidney_fowler_model.get_ccz2t_costs_from_grid_search(n_logical_gates=n_logical_gates, n_algo_qubits=int(algorithm.n_algo_qubits), phys_err=physical_error_rate, error_budget=error_budget, factory_iter=[MultiFactory(f, magic_count) for f in gidney_fowler_model.iter_ccz2t_factories()])
        unit, duration = format_duration([res.duration_hr * 60 * 60 * 10**6])
        return {"display": "block"}, f"{duration[0]:g} {unit}"
    return {"display": "none"}, ""

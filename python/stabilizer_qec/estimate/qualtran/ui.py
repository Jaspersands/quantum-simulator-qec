# Ported from Qualtran (quantumlib/Qualtran 0.7.0, qualtran/surface_code/ui.py; Copyright 2024 Google LLC, Apache
# License 2.0; see LICENSE in the parent directory), modified for stabilizer_qec: the estimates and figures of the
# Dash app without Dash, plotly or pandas (a figure is plotly's figure dict); the layout built without Dash, and
# the Dash app made on first use; a page of the standard library's server where Dash is not installed.
"""Qualtran's ``qualtran.surface_code.ui``: Qualtran's interactive QEC-overhead app, the estimates
behind it (``update`` and the helpers it calls), and its tables of QEC schemes, magic-state factories
and rotation models.

`make_app()` builds Qualtran's Dash app where dash is installed: its title, its layout (Qualtran's
builders, `input_components` and the rest: 92 components) and its one callback (`update`'s nine
outputs from eight inputs). ``ui.app`` is such an app, made when first used (Qualtran makes its own
at import; importing this module imports neither dash nor plotly). Without dash, ``ui.app`` raises
the ImportError `make_app` raises, so ``hasattr(ui, "app")`` and ``getattr(ui, "app", None)`` raise
it too rather than giving False or None; ``importlib.util.find_spec("dash")`` says whether dash is
installed. `serve()` runs the app, as ``python -m stabilizer_qec.estimate.qualtran.ui`` does (Qualtran's
``python -m qualtran.surface_code.ui``); without dash it serves a page of its own from the standard
library's HTTP server, which draws the same layout with plotly.js and calls `update`.

A figure comes back as a `Figure`: plotly's figure dict (``{"data": [...], "layout": {...}}``) with
the traces and layout ``plotly.express`` gives Qualtran's, less plotly's default template;
``Figure.to_plotly()`` makes it a ``plotly.graph_objects.Figure`` when plotly is installed (the Dash
app sends these, so its figures are Qualtran's). `create_runtime_plot` draws its factory counts from
numpy's global random state as Qualtran's does, so the same seed gives the same plot.

>>> from stabilizer_qec.estimate.qualtran import ui
>>> out = ui.update(1e-4, 1e-3, ui._GIDNEY_FOWLER_MODEL, [10, 10, 10, 10, 10, 10], "GidneyFowler", "GidneyFowlerCCZ", 1, "BeverlandEtAlRotationCost")
>>> out[0].data[0]["values"], out[3], out[4], out[8]
([1470, 12736], ['Total Number of Toffoli gates'], '47.5', '5.968 ms')
"""

from __future__ import annotations

import functools
import importlib.util
import json
import math
import pathlib
import pkgutil
import re
import sys
import traceback
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple

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

__all__ = ["Figure", "PreventUpdate", "algorithm_summary_components", "compute_duration", "create_ouputs", "create_qubit_pie_chart", "create_runtime_plot", "format_duration", "get_objects", "input_components", "make_app", "min_num_factories", "qec_summary_components", "rotation_cost_model_components", "serve", "total_magic", "update"]


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


class _Component:
    """A Dash component without Dash: ``to_plotly_json()`` gives what Dash's gives (its namespace, its type
    and the props it was given), and `make_app` makes Dash's own from it."""

    _namespace = ""
    _type = ""

    def __init__(self, children=None, **props) -> None:
        self.props = props if children is None else {"children": children, **props}

    def to_plotly_json(self) -> Dict[str, Any]:
        return {"props": dict(self.props), "type": self._type, "namespace": self._namespace}


def _components(namespace: str, names: Sequence[str]) -> SimpleNamespace:
    return SimpleNamespace(**{name: type(name, (_Component,), {"_namespace": namespace, "_type": name, "__module__": __name__}) for name in names})


# Dash's ``html`` and ``dcc`` components that Qualtran's layout uses, without Dash.
html = _components("dash_html_components", ["Details", "Div", "H4", "P", "Summary", "Table", "Tbody", "Td", "Tr"])
dcc = _components("dash_core_components", ["Graph", "Input", "Loading", "RadioItems"])


class _Wildcard:
    """Dash's wildcard ``ALL`` of a pattern-matching id, without Dash."""

    def __init__(self, value: str) -> None:
        self.value = value

    def __repr__(self) -> str:
        return f"<{self.value}>"

    def to_json(self) -> str:
        return f'["{self.value}"]'


ALL = _Wildcard("ALL")


class _Dependency:
    """A callback's input or output without Dash: a component's id and one of its props."""

    def __init__(self, component_id, component_property: str) -> None:
        self.component_id = component_id
        self.component_property = component_property

    def component_id_str(self) -> str:
        """The id as Dash writes it: a pattern-matching (dict) id as JSON, its keys sorted and a wildcard as
        ``["ALL"]``."""
        cid = self.component_id
        if not isinstance(cid, dict):
            return cid
        return "{" + ",".join(f"{json.dumps(k)}:{cid[k].to_json() if isinstance(cid[k], _Wildcard) else json.dumps(cid[k])}" for k in sorted(cid)) + "}"

    def to_dict(self) -> Dict[str, str]:
        return {"id": self.component_id_str(), "property": self.component_property}

    def __repr__(self) -> str:
        return f"<{type(self).__name__} `{self.component_id_str()}.{self.component_property}`>"


class Input(_Dependency):
    """A callback's input, as Dash's ``Input``."""


class Output(_Dependency):
    """A callback's output, as Dash's ``Output``."""


_ALGORITHM_INPUTS = [Input({"type": "algorithm", "property": ALL}, "value")]
_QEC_INPUTS = [Input("QEC", "value")]
_MAGIC_FACTORIES_INPUTS = [
    Input("_MAGIC_FACTORIES", "value"),
    Input("_MAGIC_FACTORIES_number", "value"),
]
_ROTATION_MODELS_INPUTS = [Input("rotation-cost-model", "value")]
_ESTIMATION_INPUTS = [Input("estimation_model", "value")]
_ERROR_INPUTS = [Input("physical_error_rate", "value"), Input("error_budget", "value")]

_ALL_INPUTS = [
    *_ERROR_INPUTS,
    *_ESTIMATION_INPUTS,
    *_ALGORITHM_INPUTS,
    *_QEC_INPUTS,
    *_MAGIC_FACTORIES_INPUTS,
    *_ROTATION_MODELS_INPUTS,
]

# The outputs of Qualtran's one callback, `update`, in its order.
_OUTPUTS = [
    Output("qubit-pie-chart", "figure"),
    Output("runtime-container", "style"),
    Output("runtime-plot", "figure"),
    Output("magic-name", "children"),
    Output("total_magic", "value"),
    Output("magic-factory-container", "style"),
    Output("min_factory_count", "value"),
    Output("duration-container", "style"),
    Output("duration", "value"),
]

_TITLE = "Qualtran Resource Estimation"


def _algorithm_input(name: str, placeholder: str):
    # Qualtran writes out each of these six rows; its style's "marginLight" (sic) is kept.
    return dcc.Input(debounce=True, id={"type": "algorithm", "property": name}, type="number", min=0, style={"marginLight": "10px"}, placeholder=placeholder, value=10)


def algorithm_summary_components():
    """The algorithm's six inputs (qubits, measurements, Ts, Toffolis, rotations, rotation depth), each 10 at
    first, with their pattern-matching ids ``{"type": "algorithm", "property": ...}``."""
    return [
        html.P("Enter a summary of the quantum algorithm/circuit"),
        html.Table(
            [
                html.Tr([html.Td("Qubits"), _algorithm_input("qubits", "Number of Qubits")]),
                html.Tr([html.Td("Measurements"), _algorithm_input("Measurements", "Number of Measurements")]),
                html.Tr([html.Td("Ts"), _algorithm_input("Ts", "Number of T gates")]),
                html.Tr([html.Td("Toffolis"), _algorithm_input("Toffolis", "Number of Toffolis")]),
                html.Tr([html.Td("Rotations"), _algorithm_input("Rotations", "Number of Rotations")]),
                html.Tr([html.Td("Rotation Circuit Depth"), _algorithm_input("Rotation Depth", "Rotation Circuit Depth")]),
            ]
        ),
    ]


def _MAGIC_FACTORIES_components():
    """The factory's choice, the number of factories, and the factories' table."""
    return [
        html.P("Enter number of magic state factories:"),
        dcc.RadioItems(id="_MAGIC_FACTORIES", options=sorted(_MAGIC_FACTORIES), value=next(iter(_MAGIC_FACTORIES.keys()))),
        dcc.Input(debounce=True, id="_MAGIC_FACTORIES_number", type="number", min=1, value=1),
        html.Details([html.Summary("Magic Factories' information"), html.Table([html.Tr([html.Td(k), html.Td(str(v), style={"padding": "10px"})]) for k, v in _MAGIC_FACTORIES.items()])]),
    ]


def qec_summary_components():
    """The QEC scheme's choice and the schemes' table."""
    return [
        html.P("Select a QEC algorithm:"),
        dcc.RadioItems(id="QEC", options=sorted(_QEC_SCHEMES), value=next(iter(_QEC_SCHEMES.keys()))),
        html.Details([html.Summary("QECs' information"), html.Table([html.Tr([html.Td(k), html.Td(str(v), style={"padding": "10px"})]) for k, v in _QEC_SCHEMES.items()])]),
    ]


def rotation_cost_model_components():
    """The rotation cost model's choice and the models' table."""
    return [
        html.P("Select Rotation Cost Model:"),
        dcc.RadioItems(id="rotation-cost-model", options=sorted(_ROTATION_MODELS), value=next(iter(_ROTATION_MODELS.keys()))),
        html.Details([html.Summary("Models' information"), html.Table([html.Tr([html.Td(r), html.Td(str(v), style={"padding": "10px"})]) for r, v in _ROTATION_MODELS.items()])]),
    ]


def input_components():
    """The app's inputs: the physical error rate and the error budget, the estimation model, and the
    components above."""
    return [
        html.Table(
            [
                html.Tr([html.Td("Enter physical error rate"), dcc.Input(debounce=True, id="physical_error_rate", type="number", min=0, max=1, style={"marginLeft": "10px"}, value=1e-4)]),
                html.Tr([html.Td("Enter error budget"), dcc.Input(debounce=True, id="error_budget", type="number", min=0, max=1, style={"marginLeft": "10px"}, value=1e-3)]),
            ]
        ),
        html.P("Select Estimation Cost Model:"),
        dcc.RadioItems(id="estimation_model", options=_SUPPORTED_ESTIMATION_MODELS, value=_GIDNEY_FOWLER_MODEL),
        *algorithm_summary_components(),
        *qec_summary_components(),
        *_MAGIC_FACTORIES_components(),
        *rotation_cost_model_components(),
    ]


def create_ouputs():
    """The app's outputs other than its figures: the magic-state count, the minimum factory count and the
    duration (Qualtran's name, its typo kept)."""
    return [
        html.Table(
            [
                html.Tbody(
                    [
                        html.Tr([html.Td("Total Number of Toffoli gates", id="magic-name"), html.Td(dcc.Input(id="total_magic", type="text", value="", readOnly=True))]),
                        html.Tr([html.Td("#magic factories for min runtime"), html.Td(dcc.Input(id="min_factory_count", value="", readOnly=True))], id="magic-factory-container"),
                        html.Tr([html.Td("Duration"), html.Td(dcc.Input(id="duration", type="text", value="", readOnly=True))], id="duration-container"),
                    ]
                )
            ]
        )
    ]


def _layout():
    """Qualtran's ``app.layout``."""
    return html.Div(
        [
            html.H4("Interactive QEC overhead estimation"),
            html.Table(
                [
                    html.Tr(
                        [
                            html.Td(input_components(), style={"width": "30vw"}),
                            html.Td(create_ouputs(), style={"width": "20vw"}),
                            html.Td(dcc.Loading(dcc.Graph(id="qubit-pie-chart"), type="cube", style={"max-width": "80vw", "float": "right"})),
                        ]
                    )
                ],
                style={"width": "100vw"},
            ),
            html.Div(id="runtime-container", children=[dcc.Loading(dcc.Graph(id="runtime-plot"), type="cube", style={"max-width": "80vw"})]),
        ]
    )


def _dependencies() -> List[Dict[str, Any]]:
    """The app's callbacks as Dash lists them (what it serves at ``/_dash-dependencies``), less Dash's own
    options: the one, `update`, which runs when the page loads and when an input changes."""
    output = ".." + "...".join(o.component_id_str().replace(".", "\\.") + "." + o.component_property for o in _OUTPUTS) + ".."
    return [{"output": output, "inputs": [i.to_dict() for i in _ALL_INPUTS], "state": [], "clientside_function": None, "prevent_initial_call": False}]


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


def _to_dash(x, dash):
    """A layout of `_Component` as Dash's own components."""
    if isinstance(x, _Component):
        module = dash.html if x._namespace == "dash_html_components" else dash.dcc
        return getattr(module, x._type)(**{k: _to_dash(v, dash) for k, v in x.props.items()})
    if isinstance(x, list):
        return [_to_dash(v, dash) for v in x]
    return x


def _to_dash_id(component_id, dash):
    if isinstance(component_id, dict):
        return {k: dash.ALL if v is ALL else v for k, v in component_id.items()}
    return component_id


def make_app():
    """Qualtran's Dash app (``qualtran.surface_code.ui.app``): its title, its layout and its callback, `update`,
    whose figures it sends as plotly figures (with plotly's template, as Qualtran's are) and whose
    `PreventUpdate` it raises as Dash's (Dash then keeps the outputs). Needs dash; ``ui.app`` is one, made
    when first used, and ``make_app().run()`` serves one."""
    try:
        import dash
        from dash.exceptions import PreventUpdate as DashPreventUpdate
    except ImportError as ex:
        if importlib.util.find_spec("dash") is None:
            raise ImportError("make_app needs dash: pip install dash (serve() runs the app without it)") from ex
        # dash is installed but its import fails (a Flask or Werkzeug it does not work with, say): say why.
        raise ImportError(f"make_app needs dash, which is installed but does not import ({type(ex).__name__}: {ex})") from ex

    @functools.wraps(update)
    def callback(*args):
        try:
            out = update(*args)
        except PreventUpdate:
            raise DashPreventUpdate from None
        return tuple(o.to_plotly() if isinstance(o, Figure) else o for o in out)

    app = dash.Dash(__name__)
    app.title = _TITLE
    app.layout = _to_dash(_layout(), dash)
    app.callback(*[dash.Output(o.component_id, o.component_property) for o in _OUTPUTS], *[dash.Input(_to_dash_id(i.component_id, dash), i.component_property) for i in _ALL_INPUTS])(callback)
    return app


def __getattr__(name: str):
    # ``ui.app``, Qualtran's Dash app, is made when first used: dash is optional here, and takes seconds to import.
    if name == "app":
        app = globals()["app"] = make_app()
        return app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def serve(host: str = "127.0.0.1", port: int = 8050, dash: Optional[bool] = None) -> None:
    """Runs Qualtran's app at ``http://host:port/`` until interrupted (Ctrl-C).

    Where dash is installed (or with ``dash=True``) it is `make_app`'s app, run by Dash. Without dash (or with
    ``dash=False``) the standard library's HTTP server sends a page that draws the same layout (Dash's component
    JSON, at ``/layout.json``; the callback at ``/dependencies.json``) with plotly.js, and calls `update` with a
    POST to ``/update`` of a JSON list of its eight arguments. The answer is its nine outputs as JSON (a number
    that is not finite as null, as Dash's are), or 204 No Content where `update` raises `PreventUpdate` (the page
    then keeps its outputs, as Dash's does). plotly.js is the installed plotly package's copy, served at
    ``/plotly.min.js`` as Dash serves it, or without plotly cdn.plot.ly's plotly.js 3.7.0, pinned by its hash.
    With ``port=0`` the system picks a free port; the first line printed gives the address.

    With ``dash=None``, where dash is installed but does not import, it serves the page and says why on stderr
    in one line; ``dash=True`` raises `make_app`'s ImportError instead.
    """
    app = None
    if dash or dash is None:
        try:
            app = make_app()
        except ImportError as ex:
            if dash:
                raise
            if importlib.util.find_spec("dash") is not None:  # installed, but its import failed: say why
                print(" ".join(f"{ex}; serve() runs the app on the standard library's page instead".split()), file=sys.stderr, flush=True)
    if app is not None:
        app.run(host=host, port=port)
        return
    server = _page_server(host, port)
    bound_host, bound_port = server.server_address[:2]
    shown = f"[{bound_host}]" if ":" in bound_host else bound_host
    print(f"{_TITLE} is running on http://{shown}:{bound_port}/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


_PAGE = pathlib.Path(__file__).parent / "_ui_static" / "ui.html"
_MAX_BODY = 1 << 20  # the largest body `serve`'s page server reads from a POST
# The page's plotly.js tag (cdn.plot.ly's), and the tag that loads the installed plotly's copy in its place.
_PLOTLY_TAG = re.compile(rb'<script id="plotly-js" [^>]*></script>')
_PLOTLY_LOCAL_TAG = b'<script id="plotly-js" src="plotly.min.js" charset="utf-8"></script>'
# A Content-Length as Werkzeug (Dash's server) reads one: ASCII digits, perhaps a minus sign, no "+" or "_".
_CONTENT_LENGTH = re.compile(r"-?[0-9]+", re.ASCII)


def _plotly_js() -> Optional[bytes]:
    """plotly.js as the installed plotly package carries it (``plotly/package_data/plotly.min.js``, which Dash
    serves by default), or None where plotly is not installed or does not import."""
    try:
        return pkgutil.get_data("plotly", "package_data/plotly.min.js")
    except (ImportError, OSError):
        return None


def _strict_json(x):
    """``x`` as JSON's values: numpy's numbers and arrays as Python's, and a number that is not finite as None
    (plotly's JSON encoder, which writes Dash's responses, makes it null)."""
    if isinstance(x, dict):
        return {k: _strict_json(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_strict_json(v) for v in x]
    if isinstance(x, np.ndarray):
        return _strict_json(x.tolist())
    if isinstance(x, np.generic):
        x = x.item()
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


def _page_server(host: str, port: int):
    """`serve`'s standard-library server, bound but not yet serving."""
    import socket
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from urllib.parse import urlsplit

    page = _PAGE.read_bytes()
    gets = {
        "/layout.json": (json.dumps(_layout(), default=lambda c: c.to_plotly_json(), sort_keys=True).encode(), "application/json"),
        "/dependencies.json": (json.dumps(_dependencies(), sort_keys=True).encode(), "application/json"),
    }
    plotly_js = _plotly_js()
    if plotly_js is not None:  # the installed plotly's plotly.js, as Dash serves it, in place of cdn.plot.ly's
        page = _PLOTLY_TAG.sub(_PLOTLY_LOCAL_TAG, page, count=1)
        gets["/plotly.min.js"] = (plotly_js, "text/javascript; charset=utf-8")
    gets["/"] = (page, "text/html; charset=utf-8")

    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, body: bytes = b"", content_type: Optional[str] = None, allow: Optional[str] = None) -> None:
            self.send_response(status)
            if content_type is not None:
                self.send_header("Content-Type", content_type)
            if allow is not None:
                self.send_header("Allow", allow)
            if status != 204:
                self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _refuse(self, status: int, reason: str = "", allow: Optional[str] = None) -> None:
            text = f"{status} {self.responses[status][0]}" + (f": {reason}" if reason else "")
            self._send(status, (text + "\n").encode(), "text/plain; charset=utf-8", allow)

        def do_GET(self) -> None:
            path = urlsplit(self.path).path
            if path in gets:
                self._send(200, *gets[path])
            elif path == "/update":
                self._refuse(405, "update takes a POST", allow="POST")
            else:
                self._refuse(404)

        def do_POST(self) -> None:
            path = urlsplit(self.path).path
            if path != "/update":
                if path in gets:
                    self._refuse(405, allow="GET")
                else:
                    self._refuse(404)
                return
            value = self.headers.get("Content-Length")
            if value is None:
                self._refuse(411)
                return
            value = value.strip(" \t")
            if not _CONTENT_LENGTH.fullmatch(value):
                self._refuse(400, "the Content-Length is not a number")
                return
            length = int(value)
            if length < 0:
                self._refuse(400, "the Content-Length is negative")
                return
            if length > _MAX_BODY:
                self._refuse(413, f"the body is over {_MAX_BODY} bytes")
                return
            try:
                args = json.loads(self.rfile.read(length))
            except (ValueError, RecursionError):  # RecursionError: JSON nested deeper than Python's recursion limit
                self._refuse(400, "the body is not JSON")
                return
            if not isinstance(args, list) or len(args) != 8:
                self._refuse(400, "the body is not a JSON list of update's eight arguments")
                return
            try:
                # encoded here too, so an output JSON cannot write gives a 500, not a dropped connection
                body = json.dumps(_strict_json(update(*args)), allow_nan=False).encode()
            except PreventUpdate:
                self._send(204)
                return
            except Exception:
                self.log_error("Exception on /update [POST]\n%s", traceback.format_exc())
                self._refuse(500)
                return
            self._send(200, body, "application/json")

    class Server(ThreadingHTTPServer):
        address_family = socket.AF_INET6 if ":" in host else socket.AF_INET

    return Server((host, port), Handler)


if __name__ == "__main__":  # python -m stabilizer_qec.estimate.qualtran.ui, as Qualtran's module runs its app
    serve()

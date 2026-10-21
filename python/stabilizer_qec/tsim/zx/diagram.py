# Ported from Tsim (QuEraComputing/tsim, bloqade-tsim 0.1.5, src/tsim/utils/diagram.py ``render_pyzx_d3`` and
# src/tsim/circuit.py ``Circuit.diagram``; Apache License 2.0; see LICENSE in this directory), modified for
# stabilizer_qec: the graph is built on this package's ZXGraph when pyzx_param is not installed (and drawn
# with pyzx), the reduced diagrams need pyzx_param and say so.
"""Tsim's ZX-graph diagrams: the circuit's graph with its classical wires bold (``"pyzx"``) and the
reduced sampling graphs (``"pyzx-dets"``, ``"pyzx-meas"``)."""

from __future__ import annotations

from typing import Any, Dict

from stabilizer_qec.tsim.zx.parse import parse_stim_circuit, pyzx_param_available, require_pyzx_param
from stabilizer_qec.tsim.zx.sampling import build_sampling_graph, scale_horizontally, squash_graph, transform_error_basis

PYZX_DIAGRAM_TYPES = ("pyzx", "pyzx-dets", "pyzx-meas")


def draw(g: Any, **kwargs: Any) -> Any:
    """Draw a graph with pyzx_param's ``draw`` (a pyzx_param graph) or, for this package's graph,
    pyzx's (``ZXGraph.to_pyzx``, the bold classical wires drawn plain). ``ImportError`` if the
    library is not installed."""
    from stabilizer_qec.tsim.zx.graph import ZXGraph

    if isinstance(g, ZXGraph):
        try:
            import pyzx
        except ImportError as exc:
            raise ImportError("drawing the ZX graph needs pyzx (pip install pyzx) or pyzx_param (pip install pyzx-param), neither of which is installed") from exc
        return pyzx.draw(g.to_pyzx(), **kwargs)
    import pyzx_param as zx

    return zx.draw(g, **kwargs)


def render_pyzx(circuit: Any, kwargs: Dict[str, Any]) -> Any:
    """Render a circuit as a pyzx ZX diagram: its graph with the classical wires bold and, beside each
    detector and observable, a boundary vertex; drawn, and returned.

    Args:
        circuit: The circuit to render.
        kwargs: Additional keyword arguments passed to the underlying diagram renderer.

    Returns:
        The ZX diagram (pyzx_param's ``GraphS`` when installed, else this package's ``ZXGraph``).

    """
    if not pyzx_param_available():
        try:
            import pyzx  # noqa: F401
        except ImportError as exc:
            raise ImportError("diagram type 'pyzx' draws the ZX graph with pyzx or pyzx_param, neither of which is installed (pip install pyzx)") from exc
    built = parse_stim_circuit(circuit, track_classical_wires=True)
    g = built.graph

    if len(g.vertices()) == 0:
        return g

    g = g.clone()
    if built.last_vertex:
        max_row = max(g.row(v) for v in built.last_vertex.values())
        for q in built.last_vertex:
            g.set_row(built.last_vertex[q], max_row)

    for v in list(g.vertices()):
        phase_vars = g._phaseVars[v]
        if len(phase_vars) != 1:
            continue
        phase = next(iter(phase_vars))
        if phase.startswith("det") or phase.startswith("obs"):
            row = g.row(v)
            qubit = -2 if phase.startswith("det") else -2.5
            vb = g.add_vertex(built.vt.BOUNDARY, qubit=qubit, row=row)
            g.add_edge((v, vb))
        if phase.startswith("m["):
            g.set_phase(v, 0)

    if kwargs.get("scale_horizontally", False):
        scale_horizontally(g, kwargs.pop("scale_horizontally", 1.0))
    draw(g, **kwargs)
    return g


def render_reduced(circuit: Any, sample_detectors: bool, kwargs: Dict[str, Any]) -> Any:
    """Render a circuit's sampling graph after pyzx_param's ``full_reduce(paramSafe=True)``, its
    error variables changed to a basis and the vertices squashed under the outputs (the graph Tsim
    compiles): ``"pyzx-dets"`` (``sample_detectors``) or ``"pyzx-meas"``. pyzx_param is needed."""
    kind = "pyzx-dets" if sample_detectors else "pyzx-meas"
    require_pyzx_param(f"diagram type {kind!r}, which reduces the sampling graph with pyzx_param's full_reduce(paramSafe=True),")
    import pyzx_param as zx

    built = parse_stim_circuit(circuit, backend="pyzx_param")
    g = build_sampling_graph(built, sample_detectors=sample_detectors)
    zx.full_reduce(g, paramSafe=True)
    g, _ = transform_error_basis(g)
    squash_graph(g)
    if kwargs.get("scale_horizontally", False):
        scale_horizontally(g, kwargs.pop("scale_horizontally", 1.0))
    zx.draw(g, **kwargs)
    return g

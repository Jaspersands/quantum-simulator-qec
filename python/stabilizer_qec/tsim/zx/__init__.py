# Ported from Tsim (QuEraComputing/tsim, bloqade-tsim 0.1.5, src/tsim/core/; Apache License 2.0; see LICENSE
# in the tsim directory) and from pyzx_param (pyzx_param 0.9.3), modified for stabilizer_qec.
"""Tsim's ZX graphs: the graph a circuit is (``Circuit.get_graph``), the graph that gives the
probabilities of its samples (``Circuit.get_sampling_graph``), the tensor of the graph and its
diagrams.

``parse_stim_circuit`` builds a ``GraphRepresentation`` (the graph, the vertices of the measurement
record, the detectors and observables, the noise channels' probabilities) with Tsim's builders
(``instructions``) on one of two backends: this package's ``ZXGraph`` (``"own"``, nothing to install)
or pyzx_param's ``GraphS`` (``"pyzx_param"``, the graph Tsim builds, which pyzx_param's simplifier
reduces); by default pyzx_param's when it is installed. The two graphs are equal vertex for vertex,
edge for edge, with phases as Fractions of pi, phase variables (``rec[i]`` a measurement outcome,
``e{k}`` an error bit, ``det[i]``, ``obs[i]``) as sets of names, and the same scalar.
"""

from __future__ import annotations

from typing import Any, Optional

from .graph import EdgeType, Scalar, VertexType, ZXGraph, phase_is_clifford
from .instructions import GraphRepresentation
from .parse import BACKENDS, new_graph, parse_stim_circuit, pyzx_param_available, require_pyzx_param
from .sampling import build_sampling_graph, scale_horizontally, squash_graph, transform_error_basis
from .tensor import tensor_to_matrix, tensorfy

__all__ = [
    "BACKENDS",
    "EdgeType",
    "GraphRepresentation",
    "Scalar",
    "VertexType",
    "ZXGraph",
    "build_sampling_graph",
    "get_graph",
    "get_sampling_graph",
    "new_graph",
    "parse_stim_circuit",
    "phase_is_clifford",
    "pyzx_param_available",
    "scale_horizontally",
    "squash_graph",
    "tcount",
    "tensor_to_matrix",
    "tensorfy",
    "transform_error_basis",
]


def get_graph(circuit: Any, backend: Optional[str] = None) -> Any:
    """The ZX graph of a circuit, as ``tsim.Circuit.get_graph``."""
    return parse_stim_circuit(circuit, backend=backend).graph


def get_sampling_graph(circuit: Any, sample_detectors: bool = False, backend: Optional[str] = None) -> Any:
    """The ZX graph that gives the probabilities of a circuit's samples, as
    ``tsim.Circuit.get_sampling_graph``."""
    return build_sampling_graph(parse_stim_circuit(circuit, backend=backend), sample_detectors=sample_detectors)


def tcount(graph: Any) -> int:
    """The number of vertices of a graph with a non-Clifford phase (a phase that is not a multiple of
    one half, in units of pi), as pyzx's ``tcount``."""
    phases = graph.phases()
    return sum(1 for v in graph.vertices() if not phase_is_clifford(phases[v]))

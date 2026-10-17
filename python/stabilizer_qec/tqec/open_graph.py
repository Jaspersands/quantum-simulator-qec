# Ported from tqec (tqec/tqec 0.2.0, src/tqec/computation/open_graph.py; Apache License 2.0; see LICENSE in this
# directory), modified for stabilizer_qec: networkx's greedy colouring (largest degree first) and PyZX's Pauli
# product written out.
"""Filling the ports of a block graph for the fewest simulation runs."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from functools import reduce
from itertools import combinations
from typing import Iterator

from ._errors import TQECError, TQECWarning
from .block_graph import BlockGraph
from .correlation import CorrelationSurface, reduce_observables_to_minimal_generators
from .cube import YHalfCube, ZXCube
from .enums import Basis
from .position import Direction3D


@dataclass(frozen=True)
class FilledGraph:
    """A block graph with every port filled, the external stabilizers one simulation of it covers,
    and the correlation surfaces that are its logical observables."""

    graph: BlockGraph
    stabilizers: list[str]
    observables: list[CorrelationSurface]

    def __post_init__(self) -> None:
        if self.graph.num_ports != 0:
            raise TQECError("The filled graph should not have open ports.")
        if len(self.stabilizers) != len(self.observables):
            raise TQECError("The number of stabilizers and observables should match.")

    def get_external_stabilizers(self) -> list[str]:
        return [obs.external_stabilizer_on_graph(self.graph) for obs in self.observables]


def fill_ports_for_minimal_simulation(graph: BlockGraph, search_small_area_observables: bool = False) -> list[FilledGraph]:
    """Fill the ports so that the fewest simulation runs cover the whole logical observable set:
    stabilizers that agree on the ports they share can be simulated together, so the minimum
    clique cover of the compatibility graph (a greedy colouring of its complement) gives the
    runs; each run's ports are filled in the bases its stabilizers need (Z where none cares)."""
    num_ports = graph.num_ports
    if num_ports == 0:
        raise TQECError("The provided graph has no open ports.")
    if search_small_area_observables and num_ports > 16:
        warnings.warn("The algorithm will construct all exponentially many correlation surfaces, which can be slow for graphs with large number of ports. Consider setting `search_small_area_observables=False` for better performance.", TQECWarning, stacklevel=2)

    correlation_surfaces = graph.find_correlation_surfaces()
    stab_to_surface: dict[str, CorrelationSurface] = {s.external_stabilizer_on_graph(graph): s for s in correlation_surfaces}
    generators = list(stab_to_surface.keys())

    if search_small_area_observables:
        identity = "I" * num_ports
        for stabilizer, comb in _iter_stabilizer_group(generators):
            if stabilizer != identity and stabilizer not in stab_to_surface:
                stab_to_surface[stabilizer] = reduce(lambda a, b: a ^ b, [stab_to_surface[s] for s in comb])
        generators = list(reduce_observables_to_minimal_generators(stab_to_surface, num_ports).keys())

    # The complement of the compatibility graph, coloured greedily largest degree first (as
    # networkx's greedy_color does): each colour is a clique of compatible stabilizers.
    nodes = list(range(len(generators)))
    adjacency: dict[int, set[int]] = {n: set() for n in nodes}
    for i, j in combinations(nodes, 2):
        if not _is_compatible_paulis(generators[i], generators[j]):
            adjacency[i].add(j)
            adjacency[j].add(i)
    coloring: dict[int, int] = {}
    for node in sorted(nodes, key=lambda n: len(adjacency[n]), reverse=True):
        used = {coloring[n] for n in adjacency[node] if n in coloring}
        color = 0
        while color in used:
            color += 1
        coloring[node] = color
    cliques: dict[int, list[str]] = {}
    for node, color in coloring.items():
        cliques.setdefault(color, []).append(generators[node])

    def ports_basis_for_clique(supported_stabilizers: list[str]) -> list[str]:
        port_ops = ["I"] * num_ports
        for s in supported_stabilizers:
            for i, p in enumerate(s):
                if p != "I":
                    port_ops[i] = p
        return ["Z" if p == "I" else p for p in port_ops]

    ports = graph.ordered_ports
    filled_graphs: list[FilledGraph] = []
    for clique in cliques.values():
        fg = graph.clone()
        for port, basis in zip(ports, ports_basis_for_clique(clique)):
            port_pos = graph.ports[port]
            if basis == "Y":
                fg.fill_ports({port: YHalfCube()})
                continue
            assert basis in ("Z", "X")
            pipe_at_port = graph.pipes_at(port_pos)[0]
            at_head = pipe_at_port.u == graph[port_pos]
            fill_kind = ZXCube(*[pipe_at_port.kind.get_basis_along(d, at_head) or Basis(basis) for d in Direction3D.all_directions()])
            fg.fill_ports({port: fill_kind})
        assert fg.num_ports == 0
        filled_graphs.append(FilledGraph(graph=fg, stabilizers=clique, observables=[stab_to_surface[s] for s in clique]))
    return filled_graphs


_PRODUCT = {("I", "I"): "I", ("I", "X"): "X", ("I", "Y"): "Y", ("I", "Z"): "Z", ("X", "I"): "X", ("X", "X"): "I", ("X", "Y"): "Z", ("X", "Z"): "Y", ("Y", "I"): "Y", ("Y", "X"): "Z", ("Y", "Y"): "I", ("Y", "Z"): "X", ("Z", "I"): "Z", ("Z", "X"): "Y", ("Z", "Y"): "X", ("Z", "Z"): "I"}


def _multiply_unsigned_paulis(p1: str, p2: str) -> str:
    return "".join(_PRODUCT[a, b] for a, b in zip(p1, p2))


def _iter_stabilizer_group(generators: list[str]) -> Iterator[tuple[str, tuple[str, ...]]]:
    """Every product of the generators with the generators that make it."""
    identity = "I" * len(generators[0])
    for r in range(1, len(generators) + 1):
        for comb in combinations(generators, r):
            output = identity
            for gen in comb:
                output = _multiply_unsigned_paulis(output, gen)
            yield output, comb


def _is_compatible_paulis(s1: str, s2: str) -> bool:
    for i, j in zip(s1, s2):
        if i == "I" or j == "I":
            continue
        if i != j:
            return False
    return True

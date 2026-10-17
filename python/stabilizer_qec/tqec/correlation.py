# Ported from tqec (tqec/tqec 0.2.0, src/tqec/computation/correlation.py, src/tqec/computation/_correlation.py,
# src/tqec/interop/pyzx/positioned.py and src/tqec/interop/pyzx/utils.py; Apache License 2.0; see LICENSE in this
# directory), modified for stabilizer_qec: a small ZX graph class of this package's own in place of PyZX, no
# multiprocessing, the reserved `vertex_ordering` argument left out, the Pauli-web conversions left out, and
# the generator reduction done with a GF(2) elimination in place of stim's tableaux.
"""Correlation surfaces (Pauli webs, logical observables) of a block graph, found on its ZX graph."""

from __future__ import annotations

import operator
from copy import copy
from dataclasses import dataclass
from functools import cached_property, lru_cache, partial, reduce
from itertools import accumulate, chain, combinations, product, repeat, starmap
from typing import TYPE_CHECKING, Callable, Generator, Iterable, Iterator, NamedTuple, Optional, Sequence


def pairwise(iterable):
    """``itertools.pairwise`` (Python 3.10+), for 3.9."""
    items = list(iterable)
    return zip(items, items[1:])

from ._errors import TQECError
from .cube import CubeKind, Port, YHalfCube, ZXCube
from .enums import Basis, Pauli
from .position import Direction3D, Position3D

if TYPE_CHECKING:
    from .block_graph import BlockGraph


# ---------------------------------------------------------------------------------------------
# The ZX graph
# ---------------------------------------------------------------------------------------------


def cube_kind_to_pauli(kind: CubeKind) -> Pauli:
    """The spider a cube becomes, as a Pauli: ``Z`` for a kind with exactly one Z boundary, ``X``
    otherwise, ``Y`` for the Y half cube (a Z spider of phase π/2), ``I`` for a port (a boundary)."""
    if isinstance(kind, ZXCube):
        return Pauli.Z if sum(basis == Basis.Z for basis in kind.as_tuple()) == 1 else Pauli.X
    if isinstance(kind, Port):
        return Pauli.I
    return Pauli.Y


class ZXGraph:
    """A ZX graph of phase-free Z and X spiders, S spiders (Z with phase π/2) and boundaries, its
    edges plain or Hadamard, each vertex optionally at a 3D position. Vertices are integers in
    insertion order, as PyZX numbers them.

    The spider kinds are Paulis: ``Pauli.Z``, ``Pauli.X``, ``Pauli.Y`` (the S spider), ``Pauli.I``
    (a boundary)."""

    def __init__(self) -> None:
        self._kind: dict[int, Pauli] = {}
        self._adj: dict[int, dict[int, bool]] = {}
        self._positions: dict[int, Position3D] = {}

    def add_vertex(self, kind: Pauli, position: Optional[Position3D] = None) -> int:
        v = len(self._kind)
        self._kind[v] = kind
        self._adj[v] = {}
        if position is not None:
            self._positions[v] = position
        return v

    def add_edge(self, u: int, v: int, hadamard: bool = False) -> None:
        if u == v:
            raise TQECError("A ZX graph has no self loops.")
        self._adj[u][v] = hadamard
        self._adj[v][u] = hadamard

    def vertices(self) -> list[int]:
        return list(self._kind)

    def num_vertices(self) -> int:
        return len(self._kind)

    def edges(self) -> list[tuple[int, int]]:
        """The edges as ``(u, v)`` with ``u < v``, in vertex order then adjacency order."""
        return [(u, v) for u in self._adj for v in self._adj[u] if u < v]

    def num_edges(self) -> int:
        return len(self.edges())

    def neighbors(self, v: int) -> list[int]:
        return list(self._adj[v])

    def vertex_degree(self, v: int) -> int:
        return len(self._adj[v])

    def connected(self, u: int, v: int) -> bool:
        return v in self._adj[u]

    def kind(self, v: int) -> Pauli:
        return self._kind[v]

    def is_hadamard(self, u: int, v: int) -> bool:
        return self._adj[u][v]

    def basis(self, v: int) -> Basis:
        """The basis of a Z or X spider (an error for S spiders and boundaries)."""
        return self._kind[v].to_basis()

    @property
    def positions(self) -> dict[int, Position3D]:
        return dict(self._positions)

    @property
    def p2v(self) -> dict[Position3D, int]:
        return {p: v for v, p in self._positions.items()}

    def __getitem__(self, v: int) -> Position3D:
        return self._positions[v]

    def get_direction(self, v1: int, v2: int) -> Direction3D:
        p1, p2 = self[v1], self[v2]
        if p1.x != p2.x:
            return Direction3D.X
        if p1.y != p2.y:
            return Direction3D.Y
        return Direction3D.Z

    def leaves(self) -> list[int]:
        return [v for v in self._kind if len(self._adj[v]) == 1]

    @staticmethod
    def from_block_graph(block_graph: "BlockGraph") -> ZXGraph:
        """A cube is a spider (``cube_kind_to_pauli``) at its position, cubes in position order; a
        pipe an edge, Hadamard when the pipe is."""
        g = ZXGraph()
        p2v: dict[Position3D, int] = {}
        for cube in sorted(block_graph.cubes, key=lambda c: c.position):
            p2v[cube.position] = g.add_vertex(cube_kind_to_pauli(cube.kind), cube.position)
        for pipe in block_graph.pipes:
            g.add_edge(p2v[pipe.u.position], p2v[pipe.v.position], pipe.kind.has_hadamard)
        return g

    def check_positioned(self) -> None:
        """tqec's preconditions on a positioned ZX graph: every vertex placed, neighbours adjacent,
        boundaries and S spiders dangling (S spiders along time), no 3D corner."""
        if set(self._kind) != set(self._positions):
            raise TQECError("The vertex IDs in the ZX graph and the positions do not match.")
        for s, t in self.edges():
            if not self[s].is_neighbour(self[t]):
                raise TQECError(f"The 3D positions of the endpoints of the edge {s}--{t} must be neighbors, but got {self[s]} and {self[t]}.")
        for v in self._kind:
            kind = self._kind[v]
            if kind in (Pauli.I, Pauli.Y):
                if self.vertex_degree(v) != 1:
                    raise TQECError(f"Boundary or Z(1/2) spider must be dangling, but got {self.vertex_degree(v)} neighbors.")
                if kind is Pauli.Y:
                    nb = self.neighbors(v)[0]
                    if abs(self[nb].z - self[v].z) != 1:
                        raise TQECError(f"Z(1/2) spider must connect to the time direction, but Z(1/2) at {self[v]} connects to {self[nb]}.")
        for v in self._kind:
            if len({self.get_direction(v, u) for u in self.neighbors(v)}) == 3:
                raise TQECError(f"ZX graph has a 3D corner at node {v}.")


# ---------------------------------------------------------------------------------------------
# The public representation
# ---------------------------------------------------------------------------------------------


class ZXNode(NamedTuple):
    """A vertex of the ZX graph with the Pauli (as a basis) on the half edge incident to it."""

    id: int
    basis: Basis


class ZXEdge(NamedTuple):
    """An edge of the ZX graph a correlation surface spans: two half edges, each with its basis
    (equal on a plain edge, flipped on a Hadamard edge; a Y operator is two edges, X and Z)."""

    u: ZXNode
    v: ZXNode

    @staticmethod
    def sorted(u: ZXNode, v: ZXNode) -> ZXEdge:
        return ZXEdge(*tuple(sorted([u, v])))

    @property
    def is_self_loop(self) -> bool:
        """A surface within a single node (the ZX graph of one node)."""
        return self.u.id == self.v.id

    @property
    def has_hadamard(self) -> bool:
        return self.u.basis is not self.v.basis

    def get_basis(self, vertex_id: int) -> Basis:
        if vertex_id == self.u.id:
            return self.u.basis
        if vertex_id == self.v.id:
            return self.v.basis
        raise TQECError(f"Vertex {vertex_id} is not incident to the edge {self}.")


@dataclass(frozen=True)
class CorrelationSurface:
    """A set of measurements whose product fixes the parity of logical operators at the inputs
    and outputs it touches, represented by the edges of the ZX graph it spans (the spiders impose
    parity constraints on the operators on their edges; the signs of the measurements are not
    kept). An edge between two Z nodes is a correlation between the logical Z operators there."""

    span: frozenset[ZXEdge]

    @cached_property
    def _graph_view(self) -> tuple[dict[int, dict[int, list[ZXEdge]]], dict[int, set[Basis]]]:
        edges: dict[int, dict[int, list[ZXEdge]]] = {}
        bases: dict[int, set[Basis]] = {}
        for edge in self.span:
            u, v = edge.u.id, edge.v.id
            edges.setdefault(u, {}).setdefault(v, []).append(edge)
            edges.setdefault(v, {}).setdefault(u, []).append(edge)
            bases.setdefault(u, set()).add(edge.u.basis)
            bases.setdefault(v, set()).add(edge.v.basis)
        return edges, bases

    def bases_at(self, v: int) -> set[Basis]:
        """The bases of the surface at a vertex."""
        return self._graph_view[1].get(v, set())

    @property
    def is_single_node(self) -> bool:
        return len(self.span) == 1 and next(iter(self.span)).is_self_loop

    @property
    def span_vertices(self) -> set[int]:
        return set(self._graph_view[0].keys())

    def edges_at(self, v: int) -> set[ZXEdge]:
        return set(chain.from_iterable(self._graph_view[0].get(v, {}).values()))

    def external_stabilizer(self, io_ports: list[int]) -> str:
        """The Pauli string the surface supports on the given vertices."""
        return "".join(str(Pauli.from_basis_set(self.bases_at(port))) for port in io_ports)

    def external_stabilizer_on_graph(self, graph: "BlockGraph") -> str:
        """The Pauli string on the graph's ports (in label order), or on its leaf cubes when it
        has no ports."""
        if graph.is_open:
            supports = [graph.ports[p] for p in graph.ordered_ports]
        else:
            supports = [cube.position for cube in graph.leaf_cubes]
        p2v = graph.to_zx_graph().p2v
        return self.external_stabilizer([p2v[p] for p in supports])

    @cached_property
    def area(self) -> int:
        """The nodes the surface spans (an X and a Z node of one vertex count twice)."""
        return len({node for edge in self.span for node in edge})

    def __xor__(self, other: CorrelationSurface) -> CorrelationSurface:
        return CorrelationSurface(self.span.symmetric_difference(other.span))

    def _to_mutable_graph_representation(self, zx_graph: ZXGraph) -> _CorrelationSurface:
        surface = _CorrelationSurface()
        for u, edges in self._graph_view[0].items():
            for v, edge in edges.items():
                surface._add_pauli_to_edge((u, v), reduce(operator.xor, (e.get_basis(u).to_pauli() for e in edge)), zx_graph.is_hadamard(u, v))
        for u, v in zx_graph.edges():
            if u not in surface or v not in surface[u]:
                surface._add_pauli_to_edge((u, v), Pauli.I, False)
        return surface


def find_correlation_surfaces(g: ZXGraph) -> list[CorrelationSurface]:
    """The correlation surfaces of a ZX graph, as tqec finds them: for each connected component,
    from its smallest leaf, explore node by node keeping a generating set of surfaces of the
    explored subgraph (at a node of basis B, the *broadcast* rule: all or none of its edges carry
    the opposite of B; the *passthrough* rule: an even number carry B; loops repaired by XOR-ing
    invalid candidates; dependent ones pruned), then reform the generators to satisfy the closed
    ports (an X or Z leaf supports only the opposite operator, a Y leaf only Y, a boundary any)
    and to keep few Y-terminating surfaces; the components' generators combined.

    The result is sorted by span."""
    _check_spiders_are_supported(g)
    if g.num_vertices() == 1:
        v = g.vertices()[0]
        node = ZXNode(v, g.basis(v).flipped())
        return [CorrelationSurface(frozenset({ZXEdge(node, node)}))]
    if not g.leaves():
        raise TQECError("The graph must contain at least one leaf node to find correlation surfaces.")
    return sorted((cs._to_immutable_public_representation(g) for cs in _find_correlation_surfaces_product(g)), key=lambda x: sorted(x.span))


def reduce_observables_to_minimal_generators(stabilizers_to_surfaces: dict[str, CorrelationSurface], hint_num_generators: Optional[int] = None) -> dict[str, CorrelationSurface]:
    """A generating set of the given observables with the smallest correlation-surface areas."""
    if not stabilizers_to_surfaces:
        return {}
    stabs_ordered_by_area = sorted(stabilizers_to_surfaces.keys(), key=lambda s: (stabilizers_to_surfaces[s].area, s))
    generators: list[str] = stabs_ordered_by_area[:1]
    basis: dict[int, tuple[int, int]] = {}
    _solve_linear_system(basis, _pauli_string_bits(generators[0]))
    for stabilizer in stabs_ordered_by_area[1:]:
        if _solve_linear_system(basis, _pauli_string_bits(stabilizer)) is None:
            generators.append(stabilizer)
        if hint_num_generators is not None and len(generators) == hint_num_generators:
            break
    return {g: stabilizers_to_surfaces[g] for g in generators}


def _pauli_string_bits(s: str) -> int:
    """An unsigned Pauli string as an integer, two bits a letter (X = 1, Z = 2, Y = 3)."""
    return _concat_ints_as_bits(({"I": 0, "_": 0, "X": 1, "Z": 2, "Y": 3}[c] for c in s), 2)


# ---------------------------------------------------------------------------------------------
# The search (tqec's _correlation.py)
# ---------------------------------------------------------------------------------------------


class _CorrelationSurface(dict):
    """A correlation surface as Paulis on half edges: ``self[u][v]`` the Pauli on the half edge
    of ``u -- v`` at ``u``."""

    def _add_pauli_to_edge(self, edge: tuple[int, int], pauli: Pauli, edge_is_hadamard: bool) -> "_CorrelationSurface":
        for (u, v), p in zip((edge, edge[::-1]), (pauli, pauli.flipped(edge_is_hadamard))):
            self.setdefault(u, {})[v] = p
        return self

    def _paulis_at_nodes(self, nodes: Iterable[int]) -> Iterable[Pauli]:
        return chain.from_iterable(self[n].values() for n in nodes)

    def _signature_at_nodes(self, nodes: Iterable[int], func: Callable[[Pauli], int] = lambda p: p.value, bit_length=2) -> int:
        return _concat_ints_as_bits(map(func, self._paulis_at_nodes(nodes)), bit_length)

    def _validate_node(self, node: int, node_basis: Pauli, has_unconnected_neighbors: bool):
        """``(broadcast Pauli, passthrough parity)`` when the node's constraints hold, else the
        syndrome as an int."""
        paulis = list(self._paulis_at_nodes([node]))
        passthrough_parity = node_basis in reduce(operator.xor, paulis)
        valid = True
        broadcast_basis = node_basis.flipped()
        syndrome = [broadcast_basis in pauli for pauli in paulis]
        broadcast_pauli = Pauli.I
        if all(syndrome):
            broadcast_pauli = broadcast_basis
        elif any(syndrome):
            valid = False
        if not has_unconnected_neighbors:
            syndrome.append(passthrough_parity)
            if passthrough_parity:
                valid = False
        if valid:
            return broadcast_pauli, passthrough_parity
        return _concat_ints_as_bits(syndrome, 1)

    def _to_immutable_public_representation(self, zx_graph: ZXGraph) -> CorrelationSurface:
        span: list[ZXEdge] = []
        zx_nodes: dict[tuple[int, Basis], ZXNode] = {}
        bases = list(Basis)
        for u, v in zx_graph.edges():
            pauli_u = self[u][v]
            pauli_v = self[v][u]
            edge_is_hadamard = zx_graph.is_hadamard(u, v)
            for xz_u, xz_v in product(Pauli.iter_xz(), repeat=2):
                if (edge_is_hadamard ^ (xz_u is xz_v)) and xz_u in pauli_u and xz_v in pauli_v:
                    basis_u = bases[xz_u.value >> 1]
                    basis_v = bases[xz_v.value >> 1]
                    node_u = zx_nodes.setdefault((u, basis_u), ZXNode(u, basis_u))
                    node_v = zx_nodes.setdefault((v, basis_v), ZXNode(v, basis_v))
                    span.append(ZXEdge.sorted(node_u, node_v))
        return CorrelationSurface(frozenset(span))


def _xor_correlation_surfaces(correlation_surfaces: Sequence[_CorrelationSurface]) -> _CorrelationSurface:
    result = _CorrelationSurface()
    others = correlation_surfaces[1:]
    for v, neighbors in correlation_surfaces[0].items():
        result_neighbors = result.setdefault(v, {})
        other_neighbor_rows = [cs[v] for cs in others]
        for n, pauli in neighbors.items():
            for neighbor_row in other_neighbor_rows:
                pauli ^= neighbor_row[n]
            result_neighbors[n] = pauli
    return result


def _connected_components(zx_graph: ZXGraph) -> list[ZXGraph]:
    """The components as subgraphs that keep the vertex numbers (``_SubGraph``)."""
    visited: set[int] = set()
    components = []
    for start in zx_graph.vertices():
        if start in visited:
            continue
        vertices: set[int] = set()
        stack = [start]
        while stack:
            v = stack.pop()
            if v in visited:
                continue
            visited.add(v)
            vertices.add(v)
            stack.extend(n for n in zx_graph.neighbors(v) if n not in visited)
        components.append(_SubGraph(zx_graph, vertices))
    return components


class _SubGraph(ZXGraph):
    """A component of a ZX graph with the parent's vertex numbers, built as tqec builds its
    PyZX subgraphs (vertices ascending, each edge added when its first endpoint is visited), so
    the adjacency order, and with it the generating set the search picks, is the same."""

    def __init__(self, parent: ZXGraph, vertices: set[int]) -> None:
        super().__init__()
        for v in sorted(vertices):
            self._kind[v] = parent.kind(v)
            self._adj[v] = {}
            if v in parent._positions:
                self._positions[v] = parent._positions[v]
        for v in sorted(vertices):
            for u in parent.neighbors(v):
                if u in vertices and v not in self._adj[u]:
                    hadamard = parent.is_hadamard(u, v)
                    self._adj[u][v] = hadamard
                    self._adj[v][u] = hadamard


def _find_correlation_surfaces_product(zx_graph: ZXGraph) -> Iterator[_CorrelationSurface]:
    """The products of the components' generators (each a surface of the whole graph)."""
    per_component = [_find_correlation_surfaces_from_leaf(component, min(component.leaves())) for component in _connected_components(zx_graph)]
    for parts in product(*per_component):
        merged = _CorrelationSurface()
        for part in parts:
            merged.update(part)
        yield merged


def _find_correlation_surfaces_from_leaf(zx_graph: ZXGraph, leaf: int) -> list[_CorrelationSurface]:
    """The generators satisfying the closed ports."""
    correlation_surfaces = _find_correlation_surface_generating_set_from_leaf(zx_graph, leaf)

    leaves: dict[Pauli, list[int]] = {pauli: [] for pauli in Pauli.iter_ixzy()}
    for v in zx_graph.leaves():
        leaves[zx_graph.kind(v).flipped()].append(v)
    open_leaves = leaves.pop(Pauli.I)

    if sum(len(ls) for ls in leaves.values()):
        correlation_surfaces = _reform_correlation_surface_generators(
            correlation_surfaces,
            lambda cs: _concat_ints_as_bits((cs._signature_at_nodes(ls, lambda p, pauli=pauli: p not in (Pauli.I, pauli), 1) for pauli, ls in leaves.items()), map(len, leaves.values())),
            stabilizer_basis={},
            basis_surfaces=[],
        )[1]

    if open_leaves:
        correlation_surfaces = [
            _xor_correlation_surfaces([correlation_surfaces[i] for i in indices]) if len(indices := _int_to_bit_indices(mask)) > 1 else correlation_surfaces[indices[0]]
            for _, mask in _normalize_basis(_construct_basis({}, correlation_surfaces, lambda cs: cs._signature_at_nodes(open_leaves))).values()
        ]

    return correlation_surfaces


def _reform_correlation_surface_generators(correlation_surfaces: Iterable[_CorrelationSurface], signature_func: Callable[[_CorrelationSurface], int], stabilizer_basis: dict[int, tuple[int, int]], basis_surfaces: Sequence[_CorrelationSurface], construct_new_surfaces: bool = True, num_new_surfaces_needed: Optional[int] = None, num_basis_surfaces_needed: Optional[int] = None) -> tuple[list[_CorrelationSurface], list[_CorrelationSurface]]:
    basis_surfaces, new_surfaces = list(basis_surfaces), []
    for correlation_surface in correlation_surfaces:
        indices = _solve_linear_system(stabilizer_basis, signature_func(correlation_surface))
        if indices is None:
            basis_surfaces.append(correlation_surface)
            if num_basis_surfaces_needed is not None and len(basis_surfaces) >= num_basis_surfaces_needed:
                break
            continue
        if construct_new_surfaces:
            new_surfaces.append(_xor_correlation_surfaces([*(basis_surfaces[k] for k in indices), correlation_surface]))
            if num_new_surfaces_needed is not None and len(new_surfaces) >= num_new_surfaces_needed:
                break
    return basis_surfaces, new_surfaces


@lru_cache(maxsize=None)
def _generate_valid_local_paulis(node_basis: Pauli, broadcast_pauli: Pauli, passthrough_parity: bool, num_unconnected_neighbors: int, generate_all: bool = False) -> list[list[Pauli]]:
    """The valid Pauli assignments to a node's unconnected edges given its broadcast Pauli and
    passthrough parity."""
    unconnected_neighbors = range(num_unconnected_neighbors)
    combined_pauli = broadcast_pauli ^ node_basis
    if generate_all:
        passthrough_nodes = chain.from_iterable(combinations(unconnected_neighbors, num_passthrough) for num_passthrough in range(passthrough_parity, len(unconnected_neighbors) + 1, 2))
        out_paulis_list = [[combined_pauli if n in nodes else broadcast_pauli for n in unconnected_neighbors] for nodes in passthrough_nodes]
    else:
        passthrough_nodes = ((n,) for n in unconnected_neighbors) if passthrough_parity else pairwise(unconnected_neighbors)
        out_paulis_list = [[combined_pauli if n in m else broadcast_pauli for n in unconnected_neighbors] for m in passthrough_nodes]
        if not passthrough_parity or not num_unconnected_neighbors:
            out_paulis_list.append([broadcast_pauli] * num_unconnected_neighbors)
    return out_paulis_list


def _expand_correlation_surface_to_node(correlation_surface: _CorrelationSurface, broadcast_pauli: Pauli, passthrough_parity: bool, node: int, node_basis: Pauli, unconnected_neighbors: list[int], edges_are_hadamard: list[bool], generate_all: bool = True, always_copy: bool = False) -> Generator[_CorrelationSurface, None, None]:
    """The generating set of the surfaces extended to a newly explored node."""
    for i, out_paulis in enumerate(_generate_valid_local_paulis(node_basis, broadcast_pauli, passthrough_parity, len(unconnected_neighbors), generate_all=generate_all)):
        if i or always_copy:
            new_correlation_surface = copy(correlation_surface)
            new_correlation_surface[node] = copy(correlation_surface[node])
        else:
            new_correlation_surface = correlation_surface
        for n, pauli, edge_is_hadamard in zip(unconnected_neighbors, out_paulis, edges_are_hadamard):
            if (i or always_copy) and n in correlation_surface:
                new_correlation_surface[n] = copy(correlation_surface[n])
            new_correlation_surface._add_pauli_to_edge((node, n), pauli, edge_is_hadamard)
        yield new_correlation_surface


def _find_correlation_surface_generating_set_from_leaf(zx_graph: ZXGraph, leaf: int) -> list[_CorrelationSurface]:
    """A generating set of the surfaces with every port open."""
    neighbor: int = zx_graph.neighbors(leaf)[0]
    correlation_surfaces: Iterator[_CorrelationSurface] = (_CorrelationSurface()._add_pauli_to_edge((leaf, neighbor), pauli, zx_graph.is_hadamard(leaf, neighbor)) for pauli in Pauli.iter_xz())
    if zx_graph.vertex_degree(neighbor) == 1:
        return list(correlation_surfaces)
    frontier = [neighbor]
    explored_leaves = [leaf]
    explored_nodes = {leaf}
    correlation_surface = next(correlation_surfaces)

    while frontier:
        current_node = frontier.pop(0)
        connected_neighbors = list(correlation_surface[current_node].keys())
        unconnected_neighbors = [n for n in zx_graph.neighbors(current_node) if n not in connected_neighbors]
        boundary_nodes = explored_leaves + frontier
        if unconnected_neighbors:
            boundary_nodes.append(current_node)
        generating_set_size = sum(len(correlation_surface[n]) for n in boundary_nodes)
        unexplored_neighbors = [n for n in unconnected_neighbors if n not in correlation_surface]
        passthrough_basis = zx_graph.kind(current_node)

        # Keep the candidates that satisfy the node's rules and are independent of those kept.
        valid_surfaces: list[tuple] = []
        invalid_surfaces: list[_CorrelationSurface] = []
        syndromes: list[int] = []
        vector_basis: dict[int, tuple[int, int]] = {}
        for correlation_surface in chain([correlation_surface], correlation_surfaces):
            constraint_check = correlation_surface._validate_node(current_node, passthrough_basis, bool(unconnected_neighbors))
            if isinstance(constraint_check, int):
                invalid_surfaces.append(correlation_surface)
                syndromes.append(constraint_check)
                continue
            if _solve_linear_system(vector_basis, correlation_surface._signature_at_nodes(boundary_nodes)) is None:
                valid_surfaces.append((correlation_surface, *constraint_check))
                if len(vector_basis) == generating_set_size:
                    break

        # Repair local violations by XOR-ing invalid candidates together.
        all_one = (1 << len(connected_neighbors)) - 1
        syndrome_basis: dict[int, tuple[int, int]] = {}
        basis_surfaces: list[_CorrelationSurface] = []
        for correlation_surface, syndrome in zip(invalid_surfaces, syndromes):
            if len(vector_basis) == generating_set_size:
                break
            for j, target in enumerate((syndrome ^ all_one, syndrome)):
                indices = _solve_linear_system(syndrome_basis, target, update_basis=j == 1)
                if indices is None:
                    if j == 1:
                        basis_surfaces.append(correlation_surface)
                    continue
                new_correlation_surface = _xor_correlation_surfaces([*(basis_surfaces[k] for k in indices), correlation_surface])
                if _solve_linear_system(vector_basis, new_correlation_surface._signature_at_nodes(boundary_nodes)) is None:
                    valid_surfaces.append((new_correlation_surface, *new_correlation_surface._validate_node(current_node, passthrough_basis, bool(unconnected_neighbors))))
                    break

        # The new branches.
        correlation_surfaces = chain.from_iterable(
            starmap(
                partial(_expand_correlation_surface_to_node, node=current_node, node_basis=passthrough_basis, unconnected_neighbors=unconnected_neighbors, edges_are_hadamard=[zx_graph.is_hadamard(current_node, n) for n in unconnected_neighbors]),
                valid_surfaces,
            )
        )
        correlation_surface = next(correlation_surfaces, None)
        if not correlation_surface:
            return []
        frontier.extend(n for n in unexplored_neighbors if n not in explored_nodes and zx_graph.vertex_degree(n) > 1)
        explored_leaves.extend(n for n in unexplored_neighbors if zx_graph.vertex_degree(n) == 1)
        explored_nodes.add(current_node)

    leaves = zx_graph.leaves()
    return _reform_correlation_surface_generators(
        [correlation_surface, *correlation_surfaces],
        lambda cs: cs._signature_at_nodes(leaves),
        stabilizer_basis={},
        basis_surfaces=[],
        construct_new_surfaces=False,
        num_basis_surfaces_needed=len(leaves),
    )[0]


def _concat_ints_as_bits(ints: Iterable[int], bit_length) -> int:
    if isinstance(bit_length, int):
        bit_length = repeat(bit_length)
    return sum(int(x) << shift for x, shift in zip(ints, chain([0], accumulate(bit_length))))


def _solve_linear_system(basis: dict[int, tuple[int, int]], x: int, update_basis: bool = True) -> Optional[tuple[int, ...]]:
    """Gaussian elimination over GF(2): ``None`` when ``x`` is independent of the basis (then
    added to it, when ``update_basis``), else the indices of the basis vectors that sum to it."""
    mask = 1 << len(basis)
    while x:
        highest_bit = x.bit_length() - 1
        if highest_bit not in basis:
            if update_basis:
                basis[highest_bit] = (x, mask)
            return None
        pivot, pivot_mask = basis[highest_bit]
        x ^= pivot
        mask ^= pivot_mask
    return _int_to_bit_indices(mask)[:-1]


def _int_to_bit_indices(x: int) -> tuple[int, ...]:
    return tuple(i for i in range(x.bit_length()) if (x >> i) & 1)


def _construct_basis(basis: dict[int, tuple[int, int]], items: Iterable, func: Callable) -> dict[int, tuple[int, int]]:
    for item in items:
        _solve_linear_system(basis, func(item))
    return basis


def _normalize_basis(basis: dict[int, tuple[int, int]], in_place: bool = True) -> dict[int, tuple[int, int]]:
    normalized_basis = basis if in_place else {}
    highest_bits = sorted(basis, reverse=True)
    for i, key in enumerate(highest_bits):
        vector, mask = basis[key]
        for highest_bit in highest_bits[i + 1 :]:
            if (vector >> highest_bit) & 1:
                pivot, pivot_mask = basis[highest_bit]
                vector ^= pivot
                mask ^= pivot_mask
        normalized_basis[key] = (vector, mask)
    return normalized_basis


def _check_spiders_are_supported(g: ZXGraph) -> None:
    for v in g.vertices():
        pauli = g.kind(v)
        degree = g.vertex_degree(v)
        if pauli is Pauli.I and degree != 1:
            raise TQECError(f"Boundary spider must be dangling, but got {degree} neighbors.")
        if pauli is Pauli.Y and degree != 1:
            raise TQECError(f"S spider must be dangling, but got {degree} neighbors.")

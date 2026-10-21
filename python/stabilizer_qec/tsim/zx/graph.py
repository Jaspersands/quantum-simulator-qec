# Ported from pyzx_param (pyzx_param 0.9.3, https://github.com/rafaelha/pyzx, src pyzx_param/graph/graph_s.py,
# graph/base.py, graph/scalar.py and utils.py; Copyright (C) 2018 - Aleks Kissinger and John van de Wetering;
# Apache License 2.0, see LICENSE in the tsim directory), modified for stabilizer_qec: the simple graph
# backend only (no vertex or edge data, no ground vertices, no phase tracking), a scalar of a power of
# sqrt(2) and a phase (the only terms Tsim's builders and the composition of a graph with its adjoint
# produce), no dependency on pyzx_param, and a conversion to pyzx.
"""The ZX graph Tsim's builders (``tsim.zx.instructions``) write, without pyzx_param.

``ZXGraph`` is the simple backend of pyzx_param's ``GraphS`` with the methods the builders, the
sampling graph, the tensor contraction and the diagrams use, with pyzx_param's semantics (the vertex
numbers a copy, a removal and a composition give; the fusing and Hopf rules of a parallel edge; the
phases a Fraction in units of pi taken mod 2; the phase variables a set of names XORed into the
phase). The same builders run on pyzx_param's ``GraphS`` when it is installed
(``tsim.zx.new_graph("pyzx_param")``), and the two build equal graphs.
"""

from __future__ import annotations

import cmath
import math
from enum import IntEnum
from fractions import Fraction
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

__all__ = ["EdgeType", "Scalar", "VertexType", "ZXGraph", "phase_is_clifford", "toggle_edge"]


class VertexType(IntEnum):
    """Type of a vertex in the graph (pyzx_param's numbering)."""

    BOUNDARY = 0
    Z = 1
    X = 2
    H_BOX = 3
    W_INPUT = 4
    W_OUTPUT = 5
    Z_BOX = 6
    Z_BOLD = 7
    X_BOLD = 8
    DUMMY = 99


class EdgeType(IntEnum):
    """Type of an edge in the graph (pyzx_param's numbering)."""

    SIMPLE = 1
    HADAMARD = 2
    W_IO = 3
    BOLD = 4
    HADAMARD_BOLD = 5


def toggle_edge(ty: int) -> EdgeType:
    """Swap the regular and Hadamard edge types."""
    if ty == EdgeType.BOLD:
        return EdgeType.HADAMARD_BOLD
    if ty == EdgeType.HADAMARD_BOLD:
        return EdgeType.BOLD
    return EdgeType.HADAMARD if ty == EdgeType.SIMPLE else EdgeType.SIMPLE


def _vertex_is_z_like(ty: int) -> bool:
    return ty == VertexType.Z or ty == VertexType.Z_BOX


def _vertex_is_zx_like(ty: int) -> bool:
    return _vertex_is_z_like(ty) or ty == VertexType.X


def phase_is_clifford(phase: Any) -> bool:
    """Whether a phase (in units of pi) is a multiple of 1/2."""
    return phase in [Fraction(i, 2) for i in range(4)]


def _cexp(val: Any) -> complex:
    return cmath.exp(1j * math.pi * val)


class Scalar:
    """The global scalar of a graph: ``sqrt(2)**power2 * exp(i*pi*phase)``.

    pyzx_param's scalar also holds phase nodes and spider pairs, which only its simplifier produces;
    Tsim's builders, ``adjoint`` and ``compose`` use these two terms alone.
    """

    def __init__(self) -> None:
        self.power2: int = 0
        self.phase: Fraction = Fraction(0)

    def __repr__(self) -> str:
        return "Scalar({})".format(str(self))

    def __str__(self) -> str:
        z = self.to_number()
        s = "{0.real:.2f}{0.imag:+.2f}i = ".format(z)
        if self.phase:
            s += "exp({}ipi)".format(str(self.phase))
        s += "sqrt(2)^{:d}".format(self.power2)
        return s

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Scalar):
            return NotImplemented
        return self.power2 == other.power2 and Fraction(self.phase) == Fraction(other.phase)

    def __hash__(self) -> int:
        return hash((self.power2, Fraction(self.phase)))

    def copy(self, conjugate: bool = False) -> "Scalar":
        s = Scalar()
        s.power2 = self.power2
        s.phase = self.phase if not conjugate else -self.phase
        return s

    def conjugate(self) -> "Scalar":
        """A new scalar equal to the complex conjugate."""
        return self.copy(conjugate=True)

    def add_power(self, n: int) -> None:
        """Multiply the scalar by sqrt(2)^n."""
        self.power2 += n

    def add_phase(self, phase: Any) -> None:
        """Multiply the scalar by a complex phase (in units of pi)."""
        self.phase = (self.phase + phase) % 2

    def mult_with_scalar(self, other: "Scalar") -> None:
        """Multiply by another scalar."""
        self.power2 += other.power2
        self.phase = (self.phase + other.phase) % 2

    def to_number(self) -> complex:
        """The scalar as a complex number."""
        return _cexp(self.phase) * math.sqrt(2) ** self.power2

    def to_dict(self) -> Dict[str, Any]:
        return {"power2": self.power2, "phase": str(self.phase)}


def _pack_indices(lst: List[Any]) -> Dict[Any, int]:
    d: Dict[Any, int] = {}
    if len(lst) == 0:
        return d
    lst.sort()
    i = 0
    x = None
    for y in lst:
        if y != x:
            x = y
            d[y] = i
            i += 1
    return d


class ZXGraph:
    """A ZX graph: vertices numbered by integers, each of a type (``VertexType``), a phase (a Fraction
    of pi, mod 2), a set of phase variables (names, XORed into the phase), a qubit and a row; at most
    one edge between two vertices, SIMPLE or HADAMARD (BOLD and HADAMARD_BOLD mark the classical wires
    of ``diagram("pyzx")``); inputs and outputs; and a scalar.

    ``add_edge`` applies the rules for a parallel edge as pyzx_param does: two fusing edges are one;
    a fusing and a Hopf edge leave the fusing one, a pi on one end and a factor of 1/sqrt(2); two Hopf
    edges cancel, with a factor of 1/2. A self-loop of a Z or X spider is dropped (a Hadamard one
    adds a pi).
    """

    backend = "own"

    def __init__(self) -> None:
        self.scalar: Scalar = Scalar()
        self.graph: Dict[int, Dict[int, int]] = {}
        self._vindex: int = 0
        self.nedges: int = 0
        self.ty: Dict[int, int] = {}
        self._phase: Dict[int, Any] = {}
        self._phaseVars: Dict[int, Set[str]] = {}
        self._qindex: Dict[int, Any] = {}
        self._rindex: Dict[int, Any] = {}
        self._inputs: Tuple[int, ...] = ()
        self._outputs: Tuple[int, ...] = ()

    def __repr__(self) -> str:
        return "ZXGraph({} vertices, {} edges)".format(self.num_vertices(), self.num_edges())

    # Inputs and outputs.
    def inputs(self) -> Tuple[int, ...]:
        return self._inputs

    def num_inputs(self) -> int:
        return len(self._inputs)

    def set_inputs(self, inputs: Iterable[int]) -> None:
        self._inputs = tuple(inputs)

    def outputs(self) -> Tuple[int, ...]:
        return self._outputs

    def num_outputs(self) -> int:
        return len(self._outputs)

    def set_outputs(self, outputs: Iterable[int]) -> None:
        self._outputs = tuple(outputs)

    # Vertices.
    def add_vertices(self, amount: int) -> range:
        for i in range(self._vindex, self._vindex + amount):
            self.graph[i] = {}
            self.ty[i] = VertexType.BOUNDARY
            self._phase[i] = Fraction(0)
            self._phaseVars[i] = set()
        self._vindex += amount
        return range(self._vindex - amount, self._vindex)

    def add_vertex_indexed(self, v: int) -> None:
        """Add a vertex with the given number (a ValueError if it exists)."""
        if v in self.graph:
            raise ValueError("Vertex with this index already exists")
        if v >= self._vindex:
            self._vindex = v + 1
        self.graph[v] = {}
        self.ty[v] = VertexType.BOUNDARY
        self._phase[v] = Fraction(0)
        self._phaseVars[v] = set()

    def add_vertex(
        self,
        ty: int = VertexType.BOUNDARY,
        qubit: Any = -1,
        row: Any = -1,
        phase: Any = None,
        index: Optional[int] = None,
        phaseVars: Optional[Set[str]] = None,
    ) -> int:
        """Add a vertex and return its number. A phase that is a str is a phase variable."""
        if index is not None:
            self.add_vertex_indexed(index)
            v = index
        else:
            v = self.add_vertices(1)[0]
        self.set_type(v, ty)
        if phase is None:
            phase = 1 if ty == VertexType.H_BOX else 0
        self.set_qubit(v, qubit)
        self.set_row(v, row)
        if phase:
            self.set_phase(v, phase)
        if phaseVars is not None:
            self.set_params(v, phaseVars)
        return v

    def remove_vertices(self, vertices: Iterable[int]) -> None:
        for v in vertices:
            for v1 in list(self.graph[v]):
                if v1 == v:
                    continue
                self.nedges -= 1
                del self.graph[v][v1]
                del self.graph[v1][v]
            del self.graph[v]
            del self.ty[v]
            del self._phase[v]
            self._phaseVars.pop(v, None)
            if v in self._inputs:
                self._inputs = tuple(u for u in self._inputs if u != v)
            if v in self._outputs:
                self._outputs = tuple(u for u in self._outputs if u != v)
            self._qindex.pop(v, None)
            self._rindex.pop(v, None)
        self._vindex = max(self.vertices(), default=0) + 1

    def remove_vertex(self, vertex: int) -> None:
        self.remove_vertices([vertex])

    def num_vertices(self) -> int:
        return len(self.graph)

    def vertices(self) -> Any:
        return self.graph.keys()

    def vertex_set(self) -> Set[int]:
        return set(self.graph)

    # Edges.
    def add_edges(self, edge_pairs: Iterable[Tuple[int, int]], edgetype: int = EdgeType.SIMPLE) -> None:
        for ep in edge_pairs:
            self.add_edge(ep, edgetype)

    def add_edge(self, edge_pair: Tuple[int, int], edgetype: int = EdgeType.SIMPLE) -> Tuple[int, int]:
        s, t = edge_pair
        t1 = self.ty[s]
        t2 = self.ty[t]
        if s == t:
            if not _vertex_is_zx_like(t1) or not _vertex_is_zx_like(t2):
                raise ValueError("Unexpected vertex type, it should be either z or x because you are trying to add a self-loop")
            if edgetype == EdgeType.SIMPLE:
                return edge_pair
            elif edgetype == EdgeType.HADAMARD:
                self.add_to_phase(s, 1)
                return edge_pair
            else:
                raise ValueError("The edge you are adding is not an accepted type")
        if t not in self.graph[s]:
            self.nedges += 1
            self.graph[s][t] = edgetype
            self.graph[t][s] = edgetype
        else:
            if _vertex_is_zx_like(t1) and _vertex_is_zx_like(t2):
                et1 = self.graph[s][t]
                # the roles of the simple and the Hadamard edge depend on whether the colours match
                if _vertex_is_z_like(t1) == _vertex_is_z_like(t2):
                    fuse, hopf = (EdgeType.SIMPLE, EdgeType.HADAMARD)
                else:
                    fuse, hopf = (EdgeType.HADAMARD, EdgeType.SIMPLE)
                if edgetype == fuse and et1 == fuse:
                    pass
                elif (edgetype == fuse and et1 == hopf) or (edgetype == hopf and et1 == fuse):
                    # the remaining edge is the fusing one, with a pi on one end
                    self.set_edge_type((s, t), fuse)
                    self.add_to_phase(s, 1)
                    self.scalar.add_power(-1)
                elif edgetype == hopf and et1 == hopf:
                    # the edge is removed (reducing mod 2)
                    self.remove_edge((s, t))
                    self.scalar.add_power(-2)
                else:
                    raise ValueError("Got unexpected edge types: {}, {}".format(t1, t2))
            else:
                raise ValueError("Attempted to add unreducible parallel edge {}, types: {}, {}".format(edge_pair, t1, t2))
        return edge_pair

    def remove_edges(self, edges: Iterable[Tuple[int, int]]) -> None:
        for s, t in edges:
            if s == t:
                continue
            self.nedges -= 1
            del self.graph[s][t]
            del self.graph[t][s]

    def remove_edge(self, edge: Tuple[int, int]) -> None:
        self.remove_edges([edge])

    def num_edges(self, s: Optional[int] = None, t: Optional[int] = None, et: Optional[int] = None) -> int:
        if s is not None and t is not None:
            if self.connected(s, t):
                if et is not None:
                    return 1 if self.edge_type((s, t)) == et else 0
                return 1
            return 0
        if s is not None:
            return self.vertex_degree(s)
        return len(list(self.edges()))

    def edges(self, s: Optional[int] = None, t: Optional[int] = None) -> Any:
        """The edges as ``(smaller, larger)`` pairs; those at ``s`` or between ``s`` and ``t`` if given."""
        if s is not None and t is not None:
            if self.connected(s, t):
                yield (s, t) if s < t else (t, s)
        elif s is not None:
            for t in self.graph[s]:
                yield (s, t) if s < t else (t, s)
        else:
            for v0, adj in self.graph.items():
                for v1 in adj:
                    if v1 > v0:
                        yield (v0, v1)

    def edge(self, s: int, t: int, et: int = EdgeType.SIMPLE) -> Tuple[int, int]:
        return (s, t) if s < t else (t, s)

    def edge_set(self) -> Set[Tuple[int, int]]:
        return set(self.edges())

    def edge_st(self, edge: Tuple[int, int]) -> Tuple[int, int]:
        return edge

    def edge_s(self, edge: Tuple[int, int]) -> int:
        return edge[0]

    def edge_t(self, edge: Tuple[int, int]) -> int:
        return edge[1]

    def neighbors(self, vertex: int) -> Any:
        return self.graph[vertex].keys()

    def vertex_degree(self, vertex: int) -> int:
        return len(self.graph[vertex])

    def incident_edges(self, vertex: int) -> List[Tuple[int, int]]:
        return [(vertex, v1) if v1 > vertex else (v1, vertex) for v1 in self.graph[vertex]]

    def connected(self, v1: int, v2: int) -> bool:
        return v2 in self.graph[v1]

    def edge_type(self, e: Tuple[int, int]) -> int:
        v1, v2 = e
        try:
            return self.graph[v1][v2]
        except KeyError:
            return 0

    def set_edge_type(self, e: Tuple[int, int], t: int) -> None:
        v1, v2 = e
        self.graph[v1][v2] = t
        self.graph[v2][v1] = t

    # Types, phases, positions.
    def type(self, vertex: int) -> int:
        return self.ty[vertex]

    def types(self) -> Dict[int, int]:
        return self.ty

    def set_type(self, vertex: int, t: int) -> None:
        self.ty[vertex] = t

    def phase(self, vertex: int) -> Any:
        """The constant part of the vertex's phase, a Fraction of pi."""
        return self._phase.get(vertex, Fraction(1))

    def phases(self) -> Dict[int, Any]:
        return self._phase

    def set_phase(self, vertex: int, phase: Any, clearParams: bool = True) -> None:
        """Set the phase: a str is a phase variable (and the constant part 0), a number the constant
        part (mod 2; the variables are cleared unless ``clearParams`` is False)."""
        if isinstance(phase, str):
            self._phaseVars[vertex].clear()
            self._phase[vertex] = Fraction(0)
            self._phaseVars[vertex].add(phase)
        else:
            if clearParams:
                self._phaseVars[vertex].clear()
            try:
                self._phase[vertex] = Fraction(phase) % 2
            except Exception:
                self._phase[vertex] = phase

    def add_to_phase(self, vertex: int, phase: Any, params: Optional[Set[str]] = None) -> None:
        if params is None:
            params = set()
        self._phaseVars[vertex] = self._phaseVars[vertex].symmetric_difference(params)
        old_phase = self._phase.get(vertex, Fraction(1))
        try:
            self._phase[vertex] = (old_phase + Fraction(phase)) % 2
        except Exception:
            self._phase[vertex] = old_phase + phase

    def get_all_params(self) -> Dict[int, Set[str]]:
        return self._phaseVars

    def get_params(self, vertex: int) -> Set[str]:
        return self._phaseVars[vertex]

    def phase_vars(self, vertex: int) -> Set[str]:
        """The vertex's phase variables (a copy)."""
        return set(self._phaseVars[vertex])

    def set_params(self, vertex: int, params: Iterable[str]) -> None:
        self._phaseVars[vertex].clear()
        self._phaseVars[vertex] = self._phaseVars[vertex].union(params)

    def add_params(self, vertex: int, params: Iterable[str]) -> None:
        self._phaseVars[vertex] = self._phaseVars[vertex].union(params)

    def toggle_var(self, vertex: int, strVar: str) -> None:
        if strVar in self._phaseVars[vertex]:
            self._phaseVars[vertex].remove(strVar)
        else:
            self._phaseVars[vertex].add(strVar)

    def qubit(self, vertex: int) -> Any:
        return self._qindex.get(vertex, -1)

    def qubits(self) -> Dict[int, Any]:
        return self._qindex

    def set_qubit(self, vertex: int, q: Any) -> None:
        self._qindex[vertex] = q

    def row(self, vertex: int) -> Any:
        return self._rindex.get(vertex, -1)

    def rows(self) -> Dict[int, Any]:
        return self._rindex

    def set_row(self, vertex: int, r: Any) -> None:
        self._rindex[vertex] = r

    def set_position(self, vertex: int, q: Any, r: Any) -> None:
        self.set_qubit(vertex, q)
        self.set_row(vertex, r)

    def depth(self) -> Any:
        """The highest row number given to a vertex (-1 if none)."""
        return max(self._rindex.values()) if self._rindex else -1

    # Copies and composition.
    def clone(self) -> "ZXGraph":
        """An identical copy, vertex numbers kept."""
        cpy = ZXGraph()
        for v, d in self.graph.items():
            cpy.graph[v] = d.copy()
        cpy._vindex = self._vindex
        cpy.nedges = self.nedges
        cpy.ty = self.ty.copy()
        cpy._phase = self._phase.copy()
        cpy._phaseVars = {v: set(p) for v, p in self._phaseVars.items()}
        cpy._qindex = self._qindex.copy()
        cpy._rindex = self._rindex.copy()
        cpy.scalar = self.scalar.copy()
        cpy._inputs = tuple(self._inputs)
        cpy._outputs = tuple(self._outputs)
        return cpy

    def copy(self, adjoint: bool = False) -> "ZXGraph":
        """A copy with consecutive vertex numbers (in the order of ``vertices()``). With ``adjoint``,
        the adjoint: inputs and outputs flipped, phases and the scalar's phase negated, rows
        reversed."""
        g = ZXGraph()
        g.scalar = self.scalar.copy(conjugate=adjoint)
        mult = -1 if adjoint else 1
        ty = self.types()
        ph = self.phases()
        pvs = self.get_all_params()
        qs = self.qubits()
        rs = self.rows()
        maxr = self.depth()
        vtab: Dict[int, int] = {}
        for v in self.vertices():
            pv = pvs[v] if v in pvs else set()
            i = g.add_vertex(ty[v], phase=mult * ph[v], phaseVars=pv)
            if v in qs:
                g.set_qubit(i, qs[v])
            if v in rs:
                if adjoint:
                    g.set_row(i, maxr - rs[v])
                else:
                    g.set_row(i, rs[v])
            vtab[v] = i
        new_inputs = tuple(vtab[i] for i in self.inputs())
        new_outputs = tuple(vtab[i] for i in self.outputs())
        if not adjoint:
            g.set_inputs(new_inputs)
            g.set_outputs(new_outputs)
        else:
            g.set_inputs(new_outputs)
            g.set_outputs(new_inputs)
        for e in self.edges():
            s, t = self.edge_st(e)
            g.add_edge((vtab[s], vtab[t]), self.edge_type(e))
        return g

    def adjoint(self) -> "ZXGraph":
        """A new graph equal to the adjoint of this one."""
        return self.copy(adjoint=True)

    def compose(self, other: "ZXGraph") -> None:
        """Insert a graph after this one: its inputs plugged into this one's outputs."""
        other = other.copy()
        outputs = self.outputs()
        inputs = other.inputs()
        if len(outputs) != len(inputs):
            raise TypeError("Outputs of first graph must match inputs of second.")
        plugs: List[Tuple[int, int, int]] = []
        for k in range(len(outputs)):
            o = outputs[k]
            i = inputs[k]
            if len(self.neighbors(o)) != 1:
                raise ValueError("Bad output vertex: " + str(o))
            if len(other.neighbors(i)) != 1:
                raise ValueError("Bad input vertex: " + str(i))
            no = next(iter(self.neighbors(o)))
            ni = next(iter(other.neighbors(i)))
            plugs.append((no, ni, EdgeType.HADAMARD if self.edge_type(self.edge(no, o)) != other.edge_type(other.edge(i, ni)) else EdgeType.SIMPLE))
        self.scalar.mult_with_scalar(other.scalar)
        maxr = max((self.row(v) for v in self.vertices() if self.type(v) != VertexType.BOUNDARY), default=0)
        minr = min((other.row(v) for v in other.vertices() if other.type(v) != VertexType.BOUNDARY), default=0)
        offset = maxr - minr + 1
        for v in outputs:
            self.remove_vertex(v)
        vtab: Dict[int, int] = {}
        for v in other.vertices():
            if v not in inputs:
                w = self.add_vertex(other.type(v), phase=other.phase(v), qubit=other.qubit(v), row=offset + other.row(v))
                self.set_params(w, other.get_params(v))
                vtab[v] = w
        for e in other.edges():
            s, t = other.edge_st(e)
            if s not in inputs and t not in inputs:
                self.add_edge((vtab[s], vtab[t]), edgetype=other.edge_type(e))
        for (no, ni, et) in plugs:
            self.add_edge((no, vtab[ni]), edgetype=et)
        self.set_outputs(tuple(vtab[v] for v in other.outputs()))

    # Layout for the tensor.
    def pack_circuit_rows(self) -> None:
        """Compress the rows so that every index is used."""
        rows = [self.row(v) for v in self.vertices()]
        new_rows = _pack_indices(rows)
        for v in self.vertices():
            self.set_row(v, new_rows[self.row(v)])

    def auto_detect_io(self) -> None:
        """Make every boundary vertex an input or an output: an input if its neighbour is on a later
        row, an output if on an earlier one; each sorted by qubit."""
        ty = self.types()
        inputs: List[int] = []
        outputs: List[int] = []
        for v in self.vertices():
            if ty[v] != VertexType.BOUNDARY:
                continue
            if v in inputs or v in outputs:
                continue
            if self.vertex_degree(v) != 1:
                raise TypeError("Invalid ZX-diagram: Boundary-type vertex does not have unique neighbor")
            w = list(self.neighbors(v))[0]
            if self.row(w) > self.row(v):
                inputs.append(v)
            elif self.row(w) < self.row(v):
                outputs.append(v)
            else:
                raise TypeError("Boundary-type vertex at same horizontal position as neighbor. Can't determine whether it is an input or output.")
        self.set_inputs(sorted((v for v in inputs if v in self.graph), key=self.qubit))
        self.set_outputs(sorted((v for v in outputs if v in self.graph), key=self.qubit))

    def normalize(self) -> None:
        """Put every vertex next to an input or output at the qubit and row of its wire."""
        if self.num_inputs() == 0:
            self.auto_detect_io()
        max_r = self.depth() - 1
        if max_r <= 2:
            for o in self.outputs():
                self.set_row(o, 4)
            max_r = self.depth() - 1
        claimed = []
        for q, i in enumerate(sorted(self.inputs(), key=self.qubit)):
            self.set_row(i, 0)
            self.set_qubit(i, q)
            n = list(self.neighbors(i))[0]
            if self.type(n) in (VertexType.Z, VertexType.X):
                claimed.append(n)
                self.set_row(n, 1)
                self.set_qubit(n, q)
            else:  # directly connected to an output
                e = self.edge(i, n)
                t = self.edge_type(e)
                self.remove_edge(e)
                v = self.add_vertex(VertexType.Z, q, 1)
                self.add_edge((i, v), toggle_edge(t))
                self.add_edge((v, n), EdgeType.HADAMARD)
                claimed.append(v)
        for q, o in enumerate(sorted(self.outputs(), key=self.qubit)):
            self.set_row(o, max_r + 1)
            self.set_qubit(o, q)
            n = list(self.neighbors(o))[0]
            if n not in claimed:
                self.set_row(n, max_r)
                self.set_qubit(n, q)
            else:
                e = self.edge(o, n)
                t = self.edge_type(e)
                self.remove_edge(e)
                v = self.add_vertex(VertexType.Z, q, max_r)
                self.add_edge((o, v), toggle_edge(t))
                self.add_edge((v, n), EdgeType.HADAMARD)
        self.pack_circuit_rows()

    # Tensors.
    def to_tensor(self, preserve_scalar: bool = True) -> Any:
        """The tensor of the graph (outputs, then inputs), its phase variables read as 0
        (``tsim.zx.tensor.tensorfy``)."""
        from .tensor import tensorfy

        return tensorfy(self, preserve_scalar)

    def to_matrix(self, preserve_scalar: bool = True) -> Any:
        """The matrix of the graph, 2^outputs by 2^inputs, its phase variables read as 0."""
        from .tensor import tensor_to_matrix, tensorfy

        return tensor_to_matrix(tensorfy(self, preserve_scalar), self.num_inputs(), self.num_outputs())

    # PyZX.
    def to_pyzx(self) -> Any:
        """The graph as a ``pyzx`` graph (pyzx needed): the phase variables become boolean symbolic
        variables added to the phase (``pyzx.symbolic.new_var(name, is_bool=True)``), the classical
        wires' bold types their plain ones, inputs, outputs and the scalar kept."""
        try:
            import pyzx
            from pyzx.symbolic import new_var
            from pyzx.utils import EdgeType as ZEdge
            from pyzx.utils import VertexType as ZVertex
        except ImportError as exc:
            raise ImportError("ZXGraph.to_pyzx needs pyzx (pip install pyzx)") from exc
        vertex_map = {
            VertexType.BOUNDARY: ZVertex.BOUNDARY,
            VertexType.Z: ZVertex.Z,
            VertexType.X: ZVertex.X,
            VertexType.Z_BOLD: ZVertex.Z,
            VertexType.X_BOLD: ZVertex.X,
            VertexType.H_BOX: ZVertex.H_BOX,
            VertexType.W_INPUT: ZVertex.W_INPUT,
            VertexType.W_OUTPUT: ZVertex.W_OUTPUT,
            VertexType.Z_BOX: ZVertex.Z_BOX,
        }
        edge_map = {
            EdgeType.SIMPLE: ZEdge.SIMPLE,
            EdgeType.BOLD: ZEdge.SIMPLE,
            EdgeType.HADAMARD: ZEdge.HADAMARD,
            EdgeType.HADAMARD_BOLD: ZEdge.HADAMARD,
            EdgeType.W_IO: ZEdge.W_IO,
        }
        g = pyzx.Graph()
        ids: Dict[int, int] = {}
        for v in self.vertices():
            try:
                ty = vertex_map[VertexType(self.ty[v])]
            except (KeyError, ValueError):
                raise ValueError("vertex {} has type {}, which pyzx does not have".format(v, self.ty[v])) from None
            phase = Fraction(self._phase[v])
            for name in sorted(self._phaseVars[v]):
                phase = phase + new_var(name, True, g.var_registry)
            ids[v] = g.add_vertex(ty, qubit=self.qubit(v), row=self.row(v), phase=phase)
        for s, t in self.edges():
            g.add_edge((ids[s], ids[t]), edge_map[EdgeType(self.graph[s][t])])
        g.set_inputs(tuple(ids[v] for v in self._inputs))
        g.set_outputs(tuple(ids[v] for v in self._outputs))
        g.scalar.add_power(self.scalar.power2)
        g.scalar.add_phase(self.scalar.phase)
        return g

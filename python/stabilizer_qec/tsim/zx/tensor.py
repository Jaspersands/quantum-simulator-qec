# Ported from pyzx_param (pyzx_param 0.9.3, https://github.com/rafaelha/pyzx, src pyzx_param/tensor.py;
# Copyright (C) 2018 - Aleks Kissinger and John van de Wetering; Apache License 2.0, see LICENSE in the
# tsim directory), modified for stabilizer_qec: the contraction keeps its legs by name instead of by
# shifted position, runs on any graph with this interface (ZXGraph or pyzx_param's GraphS), and refuses
# a spider or an intermediate tensor too large to hold.
"""The tensor of a ZX graph by contraction in numpy.

A vertex is a Z or X spider (or a boundary: an input or an output), an edge the identity or, if
Hadamard, the Hadamard matrix, and the graph's scalar multiplies the result. The phase variables of
a vertex are read as 0, as Tsim reads them (``Circuit.to_matrix`` of a circuit with measurements
and noise is the map with every measurement outcome and error bit 0): the tensor depends on the
constant part of each phase alone.

The vertices are contracted one at a time, the legs of a vertex to those before it joined to the
tensor so far and its legs to those after it left open, in the order that keeps the fewest legs
open (the earlier row, then the lower number, breaks ties).
"""

from __future__ import annotations

import heapq
import math
from typing import Any, List, Tuple

import numpy as np

__all__ = ["tensor_to_matrix", "tensorfy"]

_BOUNDARY, _Z, _X = 0, 1, 2
_HADAMARD, _HADAMARD_BOLD = 2, 5
_MAX_ARITY = 24  # a spider's tensor has 2^arity entries
_MAX_ENTRIES = 1 << 26  # complex128: 1 GiB, and numpy's tensordot holds the operands' copies besides


def _parity_table(arity: int) -> np.ndarray:
    parity = np.zeros(1, dtype=np.int8)
    for _ in range(arity):
        parity = np.concatenate([parity, 1 - parity])
    return parity


def _z_tensor(arity: int, phase: float) -> np.ndarray:
    value = np.exp(1j * phase)
    m = np.zeros([2] * arity, dtype=complex)
    if arity == 0:
        m[()] = 1 + value
        return m
    m[(0,) * arity] = 1
    m[(1,) * arity] = value
    return m


def _x_tensor(arity: int, phase: float) -> np.ndarray:
    value = np.exp(1j * phase)
    if arity == 0:
        m = np.zeros([2] * arity, dtype=complex)
        m[()] = 1 + value
        return m
    sign = 1 - 2 * _parity_table(arity)  # +1 for an even number of ones, -1 for an odd one
    m = 1 + sign * value
    return np.power(np.sqrt(0.5), arity) * m.reshape([2] * arity)


def tensorfy(g: Any, preserve_scalar: bool = True) -> np.ndarray:
    """The tensor of a graph: a complex array of shape ``(2,) * (outputs + inputs)``, the output
    indices first, each in the order of ``g.outputs()`` and ``g.inputs()``, times the scalar if
    ``preserve_scalar``.

    Raises ``ValueError`` for a vertex that is not a Z or X spider or a boundary, a boundary that is
    not an input or output, an output that is not a leaf, and a spider of more than 2^24 entries (24
    legs) or an intermediate tensor of more than 2^26 (1 GiB of complex128, which ``tensordot``
    copies), instead of allocating it.
    """
    rows = g.rows()
    phases = g.phases()
    types = g.types()
    inputs = tuple(g.inputs())
    outputs = tuple(g.outputs())
    input_set = set(inputs)
    output_index = {v: k for k, v in enumerate(outputs)}
    vertices = list(g.vertices())
    if not inputs and not outputs and any(types[v] == _BOUNDARY for v in vertices):
        raise ValueError("Diagram contains BOUNDARY-type vertices, but has no inputs or outputs set. Perhaps call g.auto_detect_io() first?")
    had = 1 / math.sqrt(2) * np.array([[1, 1], [1, -1]])
    id2 = np.identity(2)

    # The tensor so far and the name of each of its axes: ("in", i) an open input,
    # ("out", k) an open output, (a, b) the end at a of an edge to the later vertex b.
    tensor = np.array(1.0, dtype="complex128")
    legs: List[Tuple[Any, Any]] = []
    for i, u in enumerate(inputs):
        if types[u] != _BOUNDARY:
            raise ValueError("Wrong type for input: {} {}".format(u, types[u]))
        neighbors = list(g.neighbors(u))
        if len(neighbors) != 1:
            raise ValueError("Weird input")
        if tensor.size * 4 > _MAX_ENTRIES:
            raise ValueError("the contraction needs a tensor of more than 2^26 entries")
        tensor = np.tensordot(tensor, id2, axes=0)
        legs += [("in", i), (u, neighbors[0])]
    done = set(inputs)

    # The order: the vertex that leaves the fewest open legs (its legs not yet joined, less those it
    # joins), then the earlier row, then the lower number.
    degree = {v: len(g.neighbors(v)) + (1 if v in output_index else 0) for v in vertices}
    joined = {v: 0 for v in vertices}
    for u in inputs:
        for n in g.neighbors(u):
            joined[n] += 1
    heap = [(degree[v] - 2 * joined[v], rows[v], v) for v in vertices if v not in input_set]
    heapq.heapify(heap)
    while heap:
        net, _, v = heapq.heappop(heap)
        if v in done or net != degree[v] - 2 * joined[v]:
            continue  # stale: the vertex has a better entry
        neighbors = list(g.neighbors(v))
        d = len(neighbors)
        if v in output_index:
            if d != 1:
                raise ValueError("Weird output")
            if types[v] != _BOUNDARY:
                raise ValueError("Wrong type for output: {} {}".format(v, types[v]))
            t = id2
            t_legs: List[Tuple[Any, Any]] = [(v, neighbors[0]), ("out", output_index[v])]
            arity_edges = 1
        else:
            if d > _MAX_ARITY:
                raise ValueError("vertex {} has {} edges: its tensor would have 2^{} entries".format(v, d, d))
            p = phases[v]
            phase = math.pi * p
            if types[v] in (_Z, 7):
                t = _z_tensor(d, phase)
            elif types[v] in (_X, 8):
                t = _x_tensor(d, phase)
            elif types[v] == _BOUNDARY:
                raise ValueError("Boundary vertex {} is neither an input nor an output".format(v))
            else:
                raise ValueError("Vertex {} has type {}, which is not a Z or X spider".format(v, types[v]))
            t_legs = [(v, n) for n in neighbors]
            arity_edges = d
        # Join the legs to the vertices before this one; a Hadamard edge puts the Hadamard matrix on
        # the leg first.
        join_tensor: List[int] = []
        join_t: List[int] = []
        for k in range(arity_edges):
            n = t_legs[k][1]
            if n not in done:
                continue
            if g.edge_type(g.edge(v, n)) in (_HADAMARD, _HADAMARD_BOLD):
                t = np.moveaxis(np.tensordot(t, had, axes=(k, 0)), -1, k)
            join_t.append(k)
            join_tensor.append(legs.index((n, v)))
        if tensor.size * t.size // 4 ** len(join_t) > _MAX_ENTRIES:
            raise ValueError("the contraction needs a tensor of more than 2^26 entries")
        tensor = np.tensordot(tensor, t, axes=(join_tensor, join_t))
        legs = [leg for i, leg in enumerate(legs) if i not in set(join_tensor)] + [leg for k, leg in enumerate(t_legs) if k not in set(join_t)]
        done.add(v)
        for n in neighbors:
            if n not in done:
                joined[n] += 1
                heapq.heappush(heap, (degree[n] - 2 * joined[n], rows[n], n))

    if any(not isinstance(leg[0], str) for leg in legs):
        raise ValueError("the graph has an edge to a vertex that was never contracted")
    position = {leg: i for i, leg in enumerate(legs)}
    perm = [position[("out", k)] for k in range(len(outputs))] + [position[("in", i)] for i in range(len(inputs))]
    tensor = np.transpose(tensor, perm)
    if preserve_scalar:
        tensor = tensor * g.scalar.to_number()
    return tensor


def tensor_to_matrix(t: np.ndarray, inputs: int, outputs: int) -> np.ndarray:
    """A tensor of ``tensorfy`` as a matrix of 2^outputs rows and 2^inputs columns."""
    return np.asarray(t).reshape(2**outputs, 2**inputs)

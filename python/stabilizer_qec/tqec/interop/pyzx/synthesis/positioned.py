# Ported from tqec (tqec/tqec 0.2.0, src/tqec/interop/pyzx/synthesis/positioned.py; Apache
# License 2.0, see LICENSE in the tqec directory), modified for stabilizer_qec: on this
# package's positioned ZX graph (from a block graph or a PyZX graph), so PyZX is not needed.
"""A block graph from a positioned ZX graph, as tqec synthesises it."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from ...._errors import TQECError
from ....block_graph import BlockGraph
from ....cube import Cube, Port, YHalfCube, ZXCube
from ....enums import Basis, Pauli
from ....pipe import PipeKind
from ....position import Direction3D

if TYPE_CHECKING:
    from ....correlation import ZXGraph


def _is_zx_no_phase(pg: "ZXGraph", v: int) -> bool:
    return pg.kind(v) in (Pauli.X, Pauli.Z)


def positioned_block_synthesis(g: "ZXGraph") -> BlockGraph:
    """The block graph of a positioned ZX graph: each vertex a cube whose kind is inferred from
    its neighbours, each edge a pipe. tqec's steps: the cubes of the L, T and X shaped corners;
    the pipes between ports and Y half cubes; pipes inferred greedily from the cubes known;
    then, while vertices are left, an orientation chosen for one and the greedy step again. A
    TQECError when the inferred kinds conflict or the result is not a valid block graph."""
    if not hasattr(g, "kind"):
        from ..positioned import PositionedZX

        g = PositionedZX(g.g, g.positions)
    nodes_to_handle = set(g.vertices())
    edges_to_handle = set(g.edges())

    bg = BlockGraph()
    _handle_corners(g, bg, nodes_to_handle)
    _handle_special_pipes(g, bg, nodes_to_handle, edges_to_handle)
    _greedily_construct_blocks(g, bg, nodes_to_handle, edges_to_handle)
    _handle_leftover_nodes(g, bg, nodes_to_handle, edges_to_handle)
    bg.validate()
    return bg


def _handle_corners(pg: "ZXGraph", bg: BlockGraph, nodes_to_handle: set[int]) -> None:
    for v in pg.vertices():
        directions = {pg.get_direction(u, v) for u in pg.neighbors(v)}
        if len(directions) != 2:
            continue
        normal_direction = set(Direction3D.all_directions()).difference(directions).pop()
        normal_direction_basis = Basis.Z if pg.kind(v) is Pauli.Z else Basis.X
        bases = [normal_direction_basis.flipped() for _ in range(3)]
        bases[normal_direction.value] = normal_direction_basis
        bg.add_cube(pg[v], ZXCube(*bases))
        nodes_to_handle.remove(v)


def _handle_special_pipes(pg: "ZXGraph", bg: BlockGraph, nodes_to_handle: set[int], edges_to_handle: set[tuple[int, int]]) -> None:
    for edge in set(edges_to_handle):
        u, v = edge
        if _is_zx_no_phase(pg, u) or _is_zx_no_phase(pg, v):
            continue
        cube_u = _port_or_y_cube(pg, u)
        cube_v = _port_or_y_cube(pg, v)
        bg.add_cube(cube_u.position, cube_u.kind, cube_u.label)
        bg.add_cube(cube_v.position, cube_v.kind, cube_v.label)
        bg.add_pipe(cube_u.position, cube_v.position, _choose_arbitrary_pipe_kind(pg, edge))
        nodes_to_handle.remove(u)
        nodes_to_handle.remove(v)
        edges_to_handle.remove(edge)


def _greedily_construct_blocks(pg: "ZXGraph", bg: BlockGraph, nodes_to_handle: set[int], edges_to_handle: set[tuple[int, int]]) -> None:
    num_nodes_left = len(nodes_to_handle) + 1
    while len(nodes_to_handle) < num_nodes_left:
        num_nodes_left = len(nodes_to_handle)
        _try_to_handle_edges(pg, bg, nodes_to_handle, edges_to_handle)


def _try_to_handle_edges(pg: "ZXGraph", bg: BlockGraph, nodes_to_handle: set[int], edges_to_handle: set[tuple[int, int]]) -> None:
    for edge in set(edges_to_handle):
        u, v = edge
        if u in nodes_to_handle and v in nodes_to_handle:
            continue
        ut, vt = pg.kind(u), pg.kind(v)
        can_infer_from_u = u not in nodes_to_handle and _is_zx_no_phase(pg, u)
        can_infer_from_v = v not in nodes_to_handle and _is_zx_no_phase(pg, v)
        if not can_infer_from_u and not can_infer_from_v:
            continue
        infer_from, other_node = (u, v) if can_infer_from_u else (v, u)
        ipos, opos = pg[infer_from], pg[other_node]
        cube_kind = cast(ZXCube, bg[ipos].kind)
        pipe_kind = PipeKind._from_cube_kind(cube_kind, pg.get_direction(u, v), can_infer_from_u, pg.is_hadamard(u, v))
        if _is_zx_no_phase(pg, other_node):
            other_cube_kind = _infer_cube_kind_from_pipe(pipe_kind, not can_infer_from_u, vt if can_infer_from_u else ut)
            if other_node not in nodes_to_handle:
                existing_kind = bg[opos].kind
                if not other_cube_kind == existing_kind:
                    raise TQECError(f"Encounter conflicting cube kinds at {opos}: ")
            other_cube = Cube(opos, other_cube_kind)
        else:
            other_cube = _port_or_y_cube(pg, other_node)
        if other_node in nodes_to_handle:
            bg.add_cube(other_cube.position, other_cube.kind, other_cube.label)
            nodes_to_handle.remove(other_node)
        bg.add_pipe(ipos, opos, pipe_kind)
        edges_to_handle.remove(edge)


def _fix_kind_for_one_node(pg: "ZXGraph", bg: BlockGraph, nodes_to_handle: set[int]) -> None:
    sorted_nodes = sorted(nodes_to_handle, key=lambda n: pg[n])
    fix_node = next(n for n in sorted_nodes if _is_zx_no_phase(pg, n))
    fix_pos = pg[fix_node]
    fix_type = pg.kind(fix_node)
    if pg.vertex_degree(fix_node) == 0:
        # A ZX graph of a single node.
        specified_kind = ZXCube.from_str("ZXZ") if fix_type is Pauli.X else ZXCube.from_str("ZXX")
    else:
        # The basis along the edge's direction is the opposite of the node's.
        basis = ["X", "Z"]
        neighbor = next(iter(pg.neighbors(fix_node)))
        basis.insert(pg.get_direction(fix_node, neighbor).value, "X" if fix_type is Pauli.Z else "Z")
        specified_kind = ZXCube.from_str("".join(basis))
    bg.add_cube(fix_pos, specified_kind)
    nodes_to_handle.remove(fix_node)


def _handle_leftover_nodes(pg: "ZXGraph", bg: BlockGraph, nodes_to_handle: set[int], edges_to_handle: set[tuple[int, int]]) -> None:
    while nodes_to_handle:
        _fix_kind_for_one_node(pg, bg, nodes_to_handle)
        _greedily_construct_blocks(pg, bg, nodes_to_handle, edges_to_handle)


def _choose_arbitrary_pipe_kind(pg: "ZXGraph", edge: tuple[int, int]) -> PipeKind:
    bases: list[str] = ["X", "Z"]
    bases.insert(pg.get_direction(*edge).value, "O")
    if pg.is_hadamard(*edge):
        bases.append("H")
    return PipeKind.from_str("".join(bases))


def _infer_cube_kind_from_pipe(pipe_kind: PipeKind, at_pipe_head: bool, vertex_type: Pauli) -> ZXCube:
    bases = [pipe_kind.get_basis_along(direction, at_pipe_head) for direction in Direction3D.all_directions()]
    assert vertex_type in (Pauli.X, Pauli.Z)
    bases[pipe_kind.direction.value] = Basis.Z if vertex_type is Pauli.X else Basis.X
    return ZXCube(*cast(list, bases))


def _port_or_y_cube(pg: "ZXGraph", v: int) -> Cube:
    if pg.kind(v) is Pauli.I:
        source = pg._pyzx
        inputs = tuple(source.inputs()) if source is not None else ()
        outputs = tuple(source.outputs()) if source is not None else ()
        if v in inputs:
            label = f"IN_{inputs.index(v)}"
        elif v in outputs:
            label = f"OUT_{outputs.index(v)}"
        else:
            label = f"P_{v}"
        return Cube(pg[v], Port(), label)
    return Cube(pg[v], YHalfCube())

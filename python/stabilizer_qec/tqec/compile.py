"""Block graphs compiled to circuits on this package's lattice-surgery engine.

A block graph whose cubes are all logical patches of one orientation, joined by time pipes
(memory rounds) and spatial pipes (merges and splits), is a *surgery program* (``src/surgery.rs``):
each cube's tile holds one patch for the cube's lifetime, prepared in the first cube's temporal
basis and read out in the last's; each time slice runs ``rounds`` rounds of syndrome extraction,
with the slice's spatial pipes merged for ``merged`` rounds and split; each correlation surface
of the graph is an observable, its records the final logicals it touches, the outcomes of the
merges it passes along and the seam readouts it crosses.

The engine's patches face their horizontal neighbours with X-type walls (a horizontal merge
measures Z⊗Z) and their vertical neighbours with Z-type walls (X⊗X), so a graph's cubes must
all be of one of the two kinds ``?X?``/``?Z?`` patterns with the X walls along one axis, which
becomes the engine's column axis. What the engine cannot express is refused with tqec named:
Hadamard pipes (CZ), spatial cubes (two spatial directions at one cube: the three CNOTs, move
and rotation, the Steane encoding), Y half cubes, open ports, and two patches on one tile."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional, Sequence, Union

from ._errors import TQECError
from .cube import ZXCube
from .enums import Basis
from .position import Direction3D, Position3D

if TYPE_CHECKING:
    from .block_graph import BlockGraph
    from .correlation import CorrelationSurface

_REFUSED = "tqec (pip install tqec) compiles it"


@dataclass
class CompiledBlockGraph:
    """A block graph's surgery program and circuit: the patches' tiles (engine column, row) per
    cube lifetime, the steps, the observables' terms, and the stabilizers (Pauli strings on the
    graph's leaf cubes, in tqec's order) each observable measures."""

    circuit: object
    distance: int
    tiles: list[tuple[int, int]]
    steps: list[tuple]
    observables: list[list[tuple]]
    stabilizers: list[str]
    surfaces: list["CorrelationSurface"]
    patch_of: dict[Position3D, int] = field(default_factory=dict)

    @property
    def num_patches(self) -> int:
        return len(self.tiles)

    @property
    def num_merges(self) -> int:
        return sum(1 for s in self.steps if s[0] == "merge")


def _basis_name(basis: Basis) -> str:
    return basis.value.lower()


def compile_block_graph(graph: "BlockGraph", distance: int, *, p: float = 0.0, rounds: Optional[int] = None, merged: Optional[int] = None, observables: Union[Sequence["CorrelationSurface"], str, None] = "auto") -> CompiledBlockGraph:
    """The graph as a lattice-surgery circuit at ``distance`` under SD6 noise ``p`` (the engine's
    noise model), ``rounds`` rounds per time slice (default ``distance``) and ``merged`` rounds
    per merge (default ``rounds``); ``observables`` the correlation surfaces to include, as in
    tqec: ``"auto"`` (the default) all the graph has, as ``find_correlation_surfaces`` orders
    them, ``None`` none, or a list (each must be a surface of this graph). The graph's shadowed
    faces are fixed first (``fix_shadowed_faces``), as tqec does, so a pass-through drawn as a
    spatial cube compiles as the patch it is. A surface that runs one patch's logical along two
    of its lines (it passes two seams of that patch) is refused: the surgery engine's
    observables take one line, and the difference of two lines is a band of checks whose signs
    are random."""
    from .correlation import ZXGraph

    if graph.is_open:
        raise TQECError("the graph has open ports; fill them (fill_ports) before compiling")
    graph.validate()
    # As tqec: the faces two pipes in one direction hide take the bases of the pipes in their
    # plane, so a cube drawn spatial but only passed through is the patch it stands for.
    graph = graph.fix_shadowed_faces()
    given = not isinstance(observables, str) and observables is not None
    if isinstance(observables, str):
        if observables != "auto":
            raise TQECError(f"observables must be 'auto', None or a list of correlation surfaces, not {observables!r}")
        observables = graph.find_correlation_surfaces()
    elif observables is None:
        observables = []
    if distance < 3 or distance % 2 == 0:
        raise TQECError(f"lattice surgery needs an odd distance of at least 3, not {distance}")
    rounds = distance if rounds is None else rounds
    merged = rounds if merged is None else merged
    if rounds < 1 or merged < 1:
        raise TQECError("rounds and merged must be at least 1")

    cubes = graph.cubes
    for pipe in graph.pipes:
        if pipe.kind.has_hadamard:
            raise TQECError(f"the pipe {pipe} has a Hadamard; the surgery engine has no Hadamard pipes ({_REFUSED})")
    for cube in cubes:
        if not isinstance(cube.kind, ZXCube):
            raise TQECError(f"the cube at {cube.position} is a {cube.kind}; the surgery engine has no Y half cubes ({_REFUSED})")
        if cube.kind.is_spatial:
            directions = {pipe.direction for pipe in graph.pipes_at(cube.position) if pipe.direction != Direction3D.Z}
            what = "a junction of two spatial directions" if len(directions) == 2 else "its spatial boundaries all of one basis"
            raise TQECError(f"the cube at {cube.position} ({cube.kind}) is spatial ({what}); the surgery engine's patches have an X and a Z boundary and merge one direction at a time ({_REFUSED})")

    # Orientation: the axis whose walls are X-type becomes the engine's column axis.
    x_walls = {cube.kind.get_basis_along(Direction3D.X) for cube in cubes}
    if len(x_walls) != 1:
        raise TQECError("the cubes mix orientations (X walls along x in some, along y in others); the surgery engine keeps one orientation")
    column_axis = Direction3D.X if x_walls.pop() == Basis.X else Direction3D.Y

    def tile_of(pos: Position3D) -> tuple[int, int]:
        return (pos.x, pos.y) if column_axis == Direction3D.X else (pos.y, pos.x)

    min_col = min(tile_of(c.position)[0] for c in cubes)
    min_row = min(tile_of(c.position)[1] for c in cubes)

    def shifted_tile(pos: Position3D) -> tuple[int, int]:
        col, row = tile_of(pos)
        return (col - min_col, row - min_row)

    # Lifetimes: the cubes of a tile, consecutive in time and joined by time pipes.
    by_tile: dict[tuple[int, int], list] = {}
    for cube in cubes:
        by_tile.setdefault(shifted_tile(cube.position), []).append(cube)
    tiles: list[tuple[int, int]] = []
    patch_of: dict[Position3D, int] = {}
    lifetimes: list[list] = []
    for tile in sorted(by_tile):
        column = sorted(by_tile[tile], key=lambda c: c.position.z)
        for a, b in zip(column, column[1:]):
            if b.position.z != a.position.z + 1 or not graph.has_pipe_between(a.position, b.position):
                raise TQECError(f"the tile at {a.position.x, a.position.y} holds two patches (cubes at z = {a.position.z} and {b.position.z} not joined by a time pipe); the surgery engine keeps one patch per tile")
        patch = len(tiles)
        tiles.append(tile)
        lifetimes.append(column)
        for cube in column:
            patch_of[cube.position] = patch

    # The time slices: merges (lines of spatially joined patches) and the steps.
    zs = sorted({c.position.z for c in cubes})
    steps: list[tuple] = []
    merge_index: dict[tuple[int, frozenset], tuple[int, int]] = {}  # (z, {patch, patch}) -> (merge, seam)
    merge_orientation: list[bool] = []
    num_merges = 0
    for z in zs:
        starting: dict[str, list[int]] = {}
        ending: dict[str, list[int]] = {}
        for patch, column in enumerate(lifetimes):
            if column[0].position.z == z:
                starting.setdefault(_basis_name(column[0].kind.get_basis_along(Direction3D.Z)), []).append(patch)
            if column[-1].position.z == z:
                ending.setdefault(_basis_name(column[-1].kind.get_basis_along(Direction3D.Z)), []).append(patch)
        for basis, patches in sorted(starting.items()):
            steps.append(("prepare", patches, basis))
        # Spatial pipes at this slice, grouped into lines.
        adjacency: dict[int, set[int]] = {}
        for pipe in graph.pipes:
            if pipe.direction == Direction3D.Z or pipe.u.position.z != z:
                continue
            a, b = patch_of[pipe.u.position], patch_of[pipe.v.position]
            adjacency.setdefault(a, set()).add(b)
            adjacency.setdefault(b, set()).add(a)
        lines: list[list[int]] = []
        seen: set[int] = set()
        for start in sorted(adjacency):
            if start in seen:
                continue
            component = []
            stack = [start]
            while stack:
                q = stack.pop()
                if q in seen:
                    continue
                seen.add(q)
                component.append(q)
                stack.extend(adjacency[q] - seen)
            cols = {tiles[q][0] for q in component}
            rows = {tiles[q][1] for q in component}
            if len(cols) > 1 and len(rows) > 1:
                raise TQECError(f"at z = {z} the patches at tiles {sorted(tiles[q] for q in component)} merge in both directions at once; the surgery engine merges a straight line ({_REFUSED})")
            horizontal = len(rows) == 1
            component.sort(key=lambda q: tiles[q][0] if horizontal else tiles[q][1])
            for a, b in zip(component, component[1:]):
                ta, tb = tiles[a], tiles[b]
                if (horizontal and tb != (ta[0] + 1, ta[1])) or (not horizontal and tb != (ta[0], ta[1] + 1)):
                    raise TQECError(f"at z = {z} the patches at tiles {ta} and {tb} merge without being neighbours")
            lines.append(component)
        if lines:
            for line in lines:
                for seam, (a, b) in enumerate(zip(line, line[1:])):
                    merge_index[(z, frozenset((a, b)))] = (num_merges, seam)
                steps.append(("merge", line))
                merge_orientation.append(len({tiles[q][1] for q in line}) == 1)
                num_merges += 1
            steps.append(("rounds", merged))
            steps.append(("split",))
        else:
            steps.append(("rounds", rounds))
        for basis, patches in sorted(ending.items()):
            steps.append(("measure", patches, basis))

    # Observables: each correlation surface's records.
    zx = ZXGraph.from_block_graph(graph)
    positions = zx.positions
    surfaces = list(observables)
    if given:
        # A surface of another graph (or a made-up one) would compile into a random observable.
        for k, surface in enumerate(surfaces):
            for edge in surface.span:
                for node in (edge.u, edge.v):
                    if node.id not in positions:
                        raise TQECError(f"observable {k} names vertex {node.id}, which this graph does not have")
                if edge.is_self_loop:
                    continue
                pu, pv = positions[edge.u.id], positions[edge.v.id]
                if not graph.has_pipe_between(pu, pv):
                    raise TQECError(f"observable {k} runs along {pu} -- {pv}, where this graph has no pipe")
                if (edge.u.basis != edge.v.basis) != zx.is_hadamard(edge.u.id, edge.v.id):
                    raise TQECError(f"observable {k} changes basis along {pu} -- {pv}, which is not a Hadamard pipe of this graph")
            # The spider rules at every vertex (tqec's broadcast and passthrough): at a spider
            # of basis B the edges carrying the other basis are all of them or none, and the
            # edges carrying B are even in number unless the spider is a leaf; a leaf spider of
            # type B is a patch prepared or measured in the other basis (|0> is an X spider),
            # which ends a surface in that other basis alone, as tqec's search starts it.
            at: dict[int, dict[int, set]] = {}
            for edge in surface.span:
                if edge.is_self_loop:
                    continue
                at.setdefault(edge.u.id, {}).setdefault(edge.v.id, set()).add(edge.u.basis)
                at.setdefault(edge.v.id, {}).setdefault(edge.u.id, set()).add(edge.v.basis)
            for v, by_neighbour in at.items():
                spider = zx._kind[v]
                if spider.name not in ("X", "Z"):
                    continue
                basis = Basis(spider.name)
                neighbours = zx.neighbors(v)
                opposite = sum(1 for u in neighbours if basis.flipped() in by_neighbour.get(u, set()))
                same = sum(1 for u in neighbours if basis in by_neighbour.get(u, set()))
                if 0 < opposite < len(neighbours):
                    raise TQECError(f"observable {k} is not a correlation surface of this graph: at the {basis.value} spider at {positions[v]} it carries {basis.flipped().value} on some pipes and not others")
                if len(neighbours) > 1 and same % 2:
                    raise TQECError(f"observable {k} is not a correlation surface of this graph: at the {basis.value} spider at {positions[v]} it carries {basis.value} on an odd number of pipes")
                if len(neighbours) == 1 and same:
                    raise TQECError(f"observable {k} is not a correlation surface of this graph: it ends in {basis.value} on the patch at {positions[v]}, which is prepared or measured in {basis.flipped().value}")
    terms_per_surface: list[list[tuple]] = []
    for surface in surfaces:
        terms: list[tuple] = []
        # The line each patch's Z (column) and X (row) logical runs along, fixed by the merge
        # outcomes the surface uses: the lines next to the seam.
        z_line: dict[int, int] = {}
        x_line: dict[int, int] = {}

        def fix(lines: dict[int, int], patch: int, line: int, what: str) -> None:
            if lines.setdefault(patch, line) != line:
                raise TQECError(f"a correlation surface runs the {what} logical of the patch at tile {tiles[patch]} along two lines (it passes two seams of that patch); the surgery engine's observables take one line")

        crossings: list[tuple[int, int, Basis]] = []
        for edge in sorted(surface.span, key=lambda e: (e.u.id, e.v.id, e.u.basis.value, e.v.basis.value)):
            if edge.is_self_loop:
                continue
            pu, pv = positions[edge.u.id], positions[edge.v.id]
            if pu.z != pv.z:
                continue
            a, b = patch_of[pu], patch_of[pv]
            merge, seam = merge_index[(pu.z, frozenset((a, b)))]
            horizontal = merge_orientation[merge]
            basis = edge.u.basis
            parity_basis = Basis.Z if horizontal else Basis.X
            if basis == parity_basis:
                terms.append(("outcome", merge, seam))
                left, right = (a, b) if (tiles[a][0] if horizontal else tiles[a][1]) < (tiles[b][0] if horizontal else tiles[b][1]) else (b, a)
                lines = z_line if horizontal else x_line
                fix(lines, left, distance - 1, basis.value)
                fix(lines, right, 0, basis.value)
            else:
                crossings.append((merge, a, b, basis))
        # A surface crossing a seam orthogonally continues the same logical line on both sides.
        # One seam term covers every seam of its merge along that line, so a surface crossing a
        # merge of three or more patches gets one term per (merge, line), not one per crossing.
        seam_terms: set[tuple[int, int]] = set()
        for merge, a, b, basis in sorted(crossings, key=lambda c: (c[0], c[1], c[2], c[3].value)):
            lines = z_line if basis == Basis.Z else x_line
            la, lb = lines.get(a), lines.get(b)
            if la is not None and lb is not None and la != lb:
                raise TQECError(f"a correlation surface crosses the seam between the patches at tiles {tiles[a]} and {tiles[b]} along different {basis.value} lines on the two sides; the surgery engine's observables take one line")
            line = la if la is not None else (lb if lb is not None else 0)
            lines[a] = line
            lines[b] = line
            if (merge, line) not in seam_terms:
                seam_terms.add((merge, line))
                terms.append(("seam", merge, min(a, b), line))
        for v in sorted(surface.span_vertices):
            pos = positions[v]
            patch = patch_of[pos]
            column = lifetimes[patch]
            if column[-1].position != pos:
                continue
            read = column[-1].kind.get_basis_along(Direction3D.Z)
            if read in surface.bases_at(v):
                lines = z_line if read == Basis.Z else x_line
                terms.append(("logical", patch, _basis_name(read), lines.get(patch, 0)))
        terms_per_surface.append(terms)

    from .. import _core
    from .._circuit import Circuit
    from .._util import call

    text = call(_core.surgery_program, distance, float(p), tiles, steps, terms_per_surface)
    circuit = Circuit(text)
    return CompiledBlockGraph(circuit, distance, tiles, steps, terms_per_surface, [s.external_stabilizer_on_graph(graph) for s in surfaces], surfaces, patch_of)

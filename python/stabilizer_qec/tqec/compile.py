"""Block graphs compiled to circuits, two ways.

tqec's way, its compile pipeline ported (``_compile``: tqec 0.2.0's plaquettes, templates, blocks,
layer tree, detector search and observables, with tqecd 0.2.1): ``compile_block_graph(graph,
convention=..., observables=..., block_temporal_height=...)`` is a
``TopologicalComputationGraph``, whose ``generate_stim_circuit(k, noise_model=...)`` is tqec's
circuit at distance 2k + 1, byte for byte, Hadamard pipes and spatial junctions included. Each
compilation starts from fresh plaquette caches: tqec reschedules the measurements of the
plaquettes it caches in place, so its circuit for a graph can depend on what the process compiled
before; here the first generation of each compilation is the circuit of a fresh process,
including when separate graphs compile concurrently. Later generations of one compilation
retain tqec's in-place rescheduling and can differ from its first generation.

This package's lattice-surgery engine: ``compile_block_graph(graph, distance, p=...)``,
``TopologicalComputationGraph.compile(distance)`` and ``BlockGraph.compile(distance)``. A block
graph whose cubes are all logical patches of one orientation, joined by time pipes (memory
rounds) and spatial pipes (merges and splits), is a *surgery program* (``src/surgery.rs``): each
run of cubes on a tile joined by time pipes is one patch, prepared in the first cube's temporal
basis and read out in the last's, and a tile can hold several such patches one after another (two
CNOTs in a row reuse the ancilla's tile); each time slice runs ``rounds`` rounds of syndrome
extraction, with the slice's spatial pipes merged for ``merged`` rounds and split; each
correlation surface of the graph is an observable, its records the final logicals it touches,
the outcomes of the merges it passes along and the seam readouts it crosses. The engine keeps
one patch per tile, so a program that reuses a tile is written by the engine's writer in Python
(``_program``), which is the engine's text on every program the engine takes. A program's
``circuit`` is in the engine's layout; its ``generate_stim_circuit(k)`` writes the graph in
tqec's, and refuses a program whose rounds tqec's layout cannot honour.

The graph's shadowed faces are fixed first, as tqec does. The engine's patches face their
horizontal neighbours with X-type walls (a horizontal merge measures Z⊗Z) and their vertical
neighbours with Z-type walls (X⊗X), so a graph's cubes must all be of one of the two kinds
``?X?``/``?Z?`` patterns with the X walls along one axis, which becomes the engine's column
axis. What the engine cannot express is refused with tqec's way named: Hadamard pipes (CZ),
spatial cubes (junctions of two spatial directions and one-basis walls: the stability
experiment, the three CNOTs, move and rotation, the Steane encoding), Y half cubes and open
ports."""

from __future__ import annotations

import numbers
import operator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional, Sequence, Union

from ._compile.utils.paths import DEFAULT_DETECTOR_DATABASE_PATH
from ._errors import TQECError
from .cube import ZXCube
from .enums import Basis
from .position import Direction3D, Position3D

if TYPE_CHECKING:
    from .block_graph import BlockGraph
    from .correlation import CorrelationSurface

_REFUSED = "compile_block_graph(graph).generate_stim_circuit(k) compiles it as tqec does"


@dataclass
class CompiledBlockGraph:
    """A block graph's surgery program and circuit: the patches' tiles (engine column, row), one
    per run of cubes joined in time (a tile appears once per patch it holds), the steps, the
    observables' terms, and the stabilizers (Pauli strings on the graph's leaf cubes, in tqec's
    order) each observable measures."""

    circuit: object
    distance: int
    tiles: list[tuple[int, int]]
    steps: list[tuple]
    observables: list[list[tuple]]
    stabilizers: list[str]
    surfaces: list["CorrelationSurface"]
    patch_of: dict[Position3D, int] = field(default_factory=dict)
    computation: Optional["TopologicalComputationGraph"] = field(default=None, repr=False, compare=False)

    @property
    def num_patches(self) -> int:
        return len(self.tiles)

    @property
    def num_merges(self) -> int:
        return sum(1 for s in self.steps if s[0] == "merge")

    def generate_stim_circuit(self, k: Optional[int] = None, noise_model=None, manhattan_radius: int = 2, detector_database=None, database_path=DEFAULT_DETECTOR_DATABASE_PATH, do_not_use_database: bool = False, only_use_database: bool = False, reschedule_measurements: bool = True):
        """tqec's circuit of this program's graph at scale ``k`` (distance 2k + 1; by default
        this program's distance), with ``noise_model`` laid over it:
        ``TopologicalComputationGraph.generate_stim_circuit``, whose parameters these are, with
        tqec's names, order and defaults, apart from ``k``'s: this program's scale, where tqec's
        ``k`` has none (2.1's ``generate_stim_circuit()`` still works). ``database_path``
        defaults, as in tqec, to tqec's database file (``DEFAULT_DETECTOR_DATABASE_PATH``,
        computed as tqec computes it), and is taken and never read or written.

        A program has two layouts. ``circuit`` is the engine's: one patch per tile with its own
        boundaries and ancillas (the CNOT at distance 3 has 113 qubits), each time slice
        ``rounds`` rounds and each merge ``merged``, under the engine's SD6 noise ``p``. This is
        tqec's: its plaquettes, detectors and observables (the CNOT at k = 1 has 81 qubits), each
        time slice, merges included, ``2k + 1`` rounds (``block_temporal_height(k) + 2`` for a
        ``LinearFunction`` height). A program whose rounds differ from tqec's at ``k`` (other
        ``rounds`` or ``merged``, or a number or callable ``block_temporal_height``) is refused
        with a ``ValueError``: tqec's layout cannot honour them, and ``circuit`` is its circuit.
        (In 2.1 this method wrote the engine's layout at distance 2k + 1.)"""
        if self.computation is None:
            raise TQECError("this program was not made by compile_block_graph")
        k = (self.distance - 1) // 2 if k is None else k
        computation = self._in_tqecs_layout(_scale(k))
        return computation.generate_stim_circuit(k, noise_model, manhattan_radius, detector_database, database_path, do_not_use_database, only_use_database, reschedule_measurements)

    def _in_tqecs_layout(self, k: int) -> "TopologicalComputationGraph":
        """The computation that writes this program in tqec's layout at scale ``k``, or a
        ``ValueError`` when tqec's layout cannot honour the program's rounds there."""
        computation = self.computation
        height = computation._height
        if _is_linear_function(height) or (height is None and computation._rounds is None and computation._merged is None):
            # tqec's own height, or the defaults: both layouts run block_temporal_height(k) + 2
            # rounds (2k + 1 by default) in every time slice, merges included.
            return computation
        tqec_rounds = 2 * k + 1
        rounds = computation._rounds_at(k)
        rounds = tqec_rounds if rounds is None else rounds
        merged = rounds if computation._merged is None else computation._merged
        # The program's time slices: each runs `rounds` rounds, or `merged` when it merges.
        slices: set[bool] = set()
        merging = False
        for step in self.steps:
            if step[0] == "merge":
                merging = True
            elif step[0] == "rounds":
                slices.add(merging)
                merging = False
        if (False in slices and rounds != tqec_rounds) or (True in slices and merged != tqec_rounds):
            given = ", ".join(f"{name}={value!r}" for name, value in (("rounds", computation._rounds), ("merged", computation._merged), ("block_temporal_height", height)) if value is not None)
            if slices == {False}:
                runs = f"each time slice runs {rounds} rounds"
            elif slices == {True}:
                runs = f"each merge runs {merged} rounds"
            else:
                runs = f"each time slice runs {rounds} rounds and each merge {merged}"
            raise ValueError(f"this program was compiled with {given}, so at k = {k} {runs}; generate_stim_circuit writes tqec's layout, which runs 2k + 1 = {tqec_rounds} rounds in every time slice, merges included, and cannot honour that. The program's circuit, in the engine's layout, is .circuit (at its distance; compile_block_graph(graph, distance, ...) for another)")
        if height is None:
            return computation
        # A number or callable height whose rounds at k are tqec's default: tqec's layout with its
        # default height (tqec takes a LinearFunction only).
        return TopologicalComputationGraph(computation.block_graph, convention=computation.convention, observables=computation._observables)


def _basis_name(basis: Basis) -> str:
    return basis.value.lower()


def _compile(graph: "BlockGraph", distance: int, *, p: float = 0.0, rounds: Optional[int] = None, merged: Optional[int] = None, observables: Union[Sequence["CorrelationSurface"], str, None] = "auto", write: bool = True) -> CompiledBlockGraph:
    """``compile_block_graph`` at a distance (see there); with ``write`` false, the program
    without its circuit (every check made, nothing written)."""
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

    # Lifetimes: the runs of a tile's cubes joined by time pipes, one patch each. A tile can hold
    # several patches one after another (two CNOTs in a row reuse the ancilla's tile): each is
    # read out before the next is prepared there.
    by_tile: dict[tuple[int, int], list] = {}
    for cube in cubes:
        by_tile.setdefault(shifted_tile(cube.position), []).append(cube)
    tiles: list[tuple[int, int]] = []
    patch_of: dict[Position3D, int] = {}
    lifetimes: list[list] = []
    for tile in sorted(by_tile):
        column = sorted(by_tile[tile], key=lambda c: c.position.z)
        runs = [[column[0]]]
        for a, b in zip(column, column[1:]):
            if b.position.z == a.position.z + 1 and graph.has_pipe_between(a.position, b.position):
                runs[-1].append(b)
            else:
                runs.append([b])
        for run in runs:
            patch = len(tiles)
            tiles.append(tile)
            lifetimes.append(run)
            for cube in run:
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

    stabilizers = [s.external_stabilizer_on_graph(graph) for s in surfaces]
    if not write:
        return CompiledBlockGraph(None, distance, tiles, steps, terms_per_surface, stabilizers, surfaces, patch_of)
    if len(set(tiles)) == len(tiles):
        text = call(_core.surgery_program, distance, float(p), tiles, steps, terms_per_surface)
    else:
        # The engine keeps one patch per tile; its writer in Python takes patches one after
        # another on a tile (and is the engine's text on every program the engine takes).
        from ._program import program_text

        text = program_text(distance, float(p), tiles, steps, terms_per_surface)
    circuit = Circuit(text)
    return CompiledBlockGraph(circuit, distance, tiles, steps, terms_per_surface, stabilizers, surfaces, patch_of)


@dataclass(frozen=True)
class Convention:
    """tqec's conventions for laying blocks out in plaquettes: ``fixed_bulk`` (the bulk's
    stabilizers fixed, each cube's boundaries by its kind; tqec's default) and
    ``fixed_boundary``. ``generate_stim_circuit`` lays them out as tqec does. The surgery engine
    has one layout (the bulk's checks fixed across the grid, each patch's boundaries by its tile)
    and compiles only graphs of one patch orientation, where every cube has the same boundaries:
    there it takes either convention for its layout."""

    name: str

    def __str__(self) -> str:
        return self.name


FIXED_BULK_CONVENTION = Convention("fixed_bulk")
FIXED_BOUNDARY_CONVENTION = Convention("fixed_boundary")
ALL_CONVENTIONS = {c.name: c for c in (FIXED_BULK_CONVENTION, FIXED_BOUNDARY_CONVENTION)}


def _is_convention(value: object) -> bool:
    return isinstance(value, (Convention, str)) or (hasattr(value, "name") and hasattr(value, "triplet"))


def _convention(convention: object) -> Convention:
    """A convention from ours, tqec's (by its name) or a name."""
    name = convention if isinstance(convention, str) else getattr(convention, "name", None)
    if name not in ALL_CONVENTIONS:
        raise TQECError(f"the convention {convention!r} is not one of tqec's ({', '.join(ALL_CONVENTIONS)})")
    return ALL_CONVENTIONS[name]


def _scale(k: object) -> int:
    """``k`` as an ``int`` (a NumPy integer too); tqec takes others and writes nonsense or fails
    in its layers, so they are refused here."""
    if not isinstance(k, bool):
        try:
            value = operator.index(k)
        except TypeError:
            pass
        else:
            if value >= 1:
                return value
    raise TQECError(f"k must be a positive integer (the distance is 2k + 1), not {k!r}")


def _is_linear_function(height: object) -> bool:
    """Whether ``height`` is a ``LinearFunction`` (ours, or tqec's by its slope and offset)."""
    from .scale import LinearFunction

    return isinstance(height, LinearFunction) or (hasattr(height, "slope") and hasattr(height, "offset"))


def _is_number(height: object) -> bool:
    return isinstance(height, numbers.Real) and not isinstance(height, bool)


def _whole_rounds(height: object) -> Optional[int]:
    """``height`` as an ``int`` when it is a whole number at least 0 (an ``int``, a NumPy
    integer, or a float with no fractional part), the heights the engine takes; else None."""
    if isinstance(height, numbers.Integral) and not isinstance(height, bool):
        value = int(height)
    elif isinstance(height, float) and height.is_integer():
        value = int(height)
    else:
        return None
    return value if value >= 0 else None


# Appended to the refusal of a number or a callable as tqec's block_temporal_height, when the
# engine takes it (a callable, or a whole number at least 0: _whole_rounds).
_HEIGHT_HINT = "2.1 took a number or a callable of k here, and its engine ran height + 2 rounds in every time slice (height 5: 7 rounds); the engine still takes them, at a distance: compile_block_graph(graph, distance, block_temporal_height=...)"


def _linear_height(height: object):
    """``block_temporal_height`` as tqec's compile takes it: a ``LinearFunction`` (ours, or
    tqec's by its slope and offset); a number is the constant function, which tqec refuses
    (``TopologicalComputationGraph`` adds to tqec's error)."""
    from .scale import LinearFunction

    if height is None or isinstance(height, LinearFunction):
        return height
    if _is_linear_function(height):
        return LinearFunction(height.slope, height.offset)
    if _is_number(height):
        return LinearFunction(0, height)
    message = f"block_temporal_height must be a LinearFunction (slope * k + offset; tqec's default is LinearFunction(2, -1)), not {height!r}"
    if callable(height):
        message += f". {_HEIGHT_HINT}."
    raise TQECError(message)


def _noisy(noise_model, circuit):
    """The circuit under a noise model: ours, or one that takes Stim's circuits (tqec's own)."""
    from .noise_model import NoiseModel

    if isinstance(noise_model, NoiseModel):
        return noise_model.noisy_circuit(circuit)
    if not hasattr(noise_model, "noisy_circuit"):
        raise TQECError(f"noise_model must be a NoiseModel (NoiseModel.uniform_depolarizing(p), NoiseModel.si1000(p)), not {type(noise_model).__name__}")
    from .._circuit import Circuit

    try:
        import stim
    except ImportError:
        stim = None
    # Another library's model (tqec's own NoiseModel) is written for Stim's circuits: it is
    # given Stim's, when Stim is there, and its result read back.
    noisy = noise_model.noisy_circuit(circuit if stim is None else stim.Circuit(str(circuit)))
    return noisy if isinstance(noisy, Circuit) else Circuit(str(noisy))


class TopologicalComputationGraph:
    """A block graph compiled as tqec's ``compile_block_graph`` compiles it, ready to be written
    at any scale: ``generate_stim_circuit(k, noise_model=...)`` is tqec's circuit at distance
    2k + 1 (its plaquettes, its detectors, its observables), noiseless or with a noise model laid
    over it, and ``compile(distance, p=...)`` the program and circuit of this package's surgery
    engine under its own SD6 noise.

    tqec's checks are made when the graph is compiled, with tqec's errors (open ports, validity,
    a Y half cube's ``NotImplementedError``, a constant ``block_temporal_height``); the engine's
    when ``compile`` is called. ``surfaces`` are the correlation surfaces compiled into
    observables and ``stabilizers`` the Pauli string each measures on the graph's ports (or leaf
    cubes)."""

    def __init__(self, block_graph: "BlockGraph", *, convention: object = FIXED_BULK_CONVENTION, observables: Union[Sequence["CorrelationSurface"], str, None] = "auto", block_temporal_height: object = None, rounds: Optional[int] = None, merged: Optional[int] = None, _engine_only: bool = False) -> None:
        self.convention = _convention(convention)
        if block_temporal_height is not None and (rounds is not None or merged is not None):
            raise TQECError("give block_temporal_height (tqec's) or rounds and merged (the engine's), not both")
        if isinstance(observables, str) and observables != "auto":
            raise TQECError(f"observables must be 'auto', None or a list of correlation surfaces, not {observables!r}")
        self._graph = block_graph
        # One-shot iterables must serve both compilation and public metadata.
        if observables is not None and not isinstance(observables, str):
            observables = list(observables)
        self._observables = observables
        self._height = block_temporal_height
        self._rounds = rounds
        self._merged = merged
        self._surfaces: Optional[list["CorrelationSurface"]] = None
        self._tqec = None
        if not _engine_only:
            self._tqec_graph()

    def _tqec_graph(self):
        """tqec's ``TopologicalComputationGraph`` of the graph (the port's), compiled once."""
        if self._tqec is None:
            from ._compile.compile import compile_block_graph as tqec_compile_block_graph
            from ._compile.convention import ALL_CONVENTIONS as TQEC_CONVENTIONS

            kwargs = {}
            height = _linear_height(self._height)
            if height is not None:
                kwargs["block_temporal_height"] = height
            observables = self._observables
            try:
                self._tqec = tqec_compile_block_graph(self._graph, convention=TQEC_CONVENTIONS[self.convention.name], observables=observables, **kwargs)
            except TQECError as error:
                # A number is the constant height, whose cubes tqec refuses with this message; the
                # hint follows when the engine takes the number.
                if _is_number(self._height) and str(error).startswith("Cannot add the block as a cube."):
                    hint = f" {_HEIGHT_HINT}." if _whole_rounds(self._height) is not None else ""
                    error.args = (f"{error} block_temporal_height={self._height!r} is a constant height, and tqec's is a LinearFunction of k (slope * k + offset, the rounds between a block's initialisation and measurement layers; LinearFunction(2, -1) by default).{hint}",)
                raise
        return self._tqec

    @property
    def block_graph(self) -> "BlockGraph":
        return self._graph

    @property
    def surfaces(self) -> list["CorrelationSurface"]:
        """The correlation surfaces compiled into observables (``"auto"``: all of the graph's, as
        ``find_correlation_surfaces`` orders them on the graph with its shadowed faces fixed)."""
        if self._surfaces is None:
            if self._observables is None:
                self._surfaces = []
            elif isinstance(self._observables, str):
                self._surfaces = self._graph.fix_shadowed_faces().find_correlation_surfaces()
            else:
                self._surfaces = list(self._observables)
        return self._surfaces

    @property
    def stabilizers(self) -> list[str]:
        """The Pauli string each observable's surface measures on the graph's ports (in label
        order), or on its leaf cubes when it has none."""
        graph = self._graph.fix_shadowed_faces()
        return [s.external_stabilizer_on_graph(graph) for s in self.surfaces]

    @property
    def num_observables(self) -> int:
        return len(self.surfaces)

    def _rounds_at(self, k: int) -> Optional[int]:
        if self._height is None:
            return self._rounds
        height = self._height(k) if callable(self._height) else self._height
        rounds = _whole_rounds(height)
        if rounds is None:
            raise TQECError(f"block_temporal_height at k = {k} is {height!r}; it must be a whole number of rounds, at least 0")
        return rounds + 2

    def compile(self, distance: int, *, p: float = 0.0) -> CompiledBlockGraph:
        """The surgery engine's program and circuit at ``distance`` under its SD6 noise ``p``:
        each time slice ``block_temporal_height(k) + 2`` rounds (tqec's layers: an
        initialisation layer, the repeated rounds and a measurement layer; ``2k + 1`` by
        default), or ``rounds`` and ``merged`` as given. Here ``block_temporal_height`` may also
        be a whole number at least 0 (a NumPy integer too) or a callable of ``k`` giving one."""
        rounds = self._rounds_at((distance - 1) // 2) if isinstance(distance, int) and distance >= 3 else self._rounds
        compiled = _compile(self._graph, distance, p=p, rounds=rounds, merged=self._merged, observables=self._observables)
        compiled.computation = self
        return compiled

    def generate_stim_circuit(self, k: int, noise_model=None, manhattan_radius: int = 2, detector_database=None, database_path=DEFAULT_DETECTOR_DATABASE_PATH, do_not_use_database: bool = False, only_use_database: bool = False, reschedule_measurements: bool = True):
        """tqec's circuit at scale ``k`` (distance 2k + 1), with ``noise_model`` (ours, or
        tqec's own) laid over the noiseless circuit, as tqec lays it.

        The detectors are found as tqec finds them, within ``manhattan_radius`` plaquettes (none
        when it is not positive), and ``reschedule_measurements`` is tqec's. tqec keeps the
        detectors it finds in a database file; here a database lives in memory only:
        ``detector_database`` (one from an earlier call's, else an empty one, or none with
        ``do_not_use_database``), and ``database_path`` (by default tqec's database file,
        ``DEFAULT_DETECTOR_DATABASE_PATH``) is taken for tqec's signature and never read or
        written. With ``only_use_database`` a situation missing from a database that has entries is
        an error, as in tqec; an empty database (or none) is filled, as tqec fills it."""
        from ._compile.detectors.database import DetectorDatabase

        if detector_database is not None and not isinstance(detector_database, DetectorDatabase):
            raise TQECError(f"detector_database must be None or this package's DetectorDatabase (stabilizer_qec.tqec._compile.detectors.database), not {type(detector_database).__name__}: tqec's is kept in files, which this package does not read")
        circuit = self._tqec_graph().generate_stim_circuit(_scale(k), None, manhattan_radius, detector_database, database_path, do_not_use_database, only_use_database, reschedule_measurements)
        if noise_model is not None:
            circuit = _noisy(noise_model, circuit)
        return circuit

    def __repr__(self) -> str:
        return f"TopologicalComputationGraph({self._graph.name!r}, {self.convention.name}, {self.num_observables} observables)"


def compile_block_graph(graph: "BlockGraph", distance=None, observables: Union[Sequence["CorrelationSurface"], str, None] = "auto", block_temporal_height: object = None, *, convention: object = FIXED_BULK_CONVENTION, p: float = 0.0, rounds: Optional[int] = None, merged: Optional[int] = None):
    """A block graph compiled to circuits, in either of two shapes.

    tqec's: ``compile_block_graph(graph, convention=..., observables=...,
    block_temporal_height=...)`` (the second argument may be the convention, as tqec places it)
    is a ``TopologicalComputationGraph``, compiled by tqec's pipeline (with tqec's errors), whose
    ``generate_stim_circuit(k, noise_model=...)`` is tqec's circuit at distance 2k + 1.
    ``block_temporal_height`` is a ``LinearFunction`` of ``k``, the rounds between a block's
    initialisation and measurement layers (default ``LinearFunction(2, -1)``).

    This package's engine: ``compile_block_graph(graph, distance, p=..., rounds=...,
    merged=...)`` is the ``CompiledBlockGraph`` at that distance under SD6 noise ``p`` (the
    engine's noise model), with ``rounds`` rounds per time slice (default ``distance``) and
    ``merged`` rounds per merge (default ``rounds``): its circuit, tiles, steps and observables'
    terms.

    ``observables`` are the correlation surfaces to include, as in tqec: ``"auto"`` (the
    default) all the graph has, as ``find_correlation_surfaces`` orders them, ``None`` none, or a
    list. The graph's shadowed faces are fixed first (``fix_shadowed_faces``), as tqec does, so
    a pass-through drawn as a spatial cube compiles as the patch it is. On the engine, a list's
    surfaces must be the graph's, and a surface that runs one patch's logical along two of its
    lines (it passes two seams of that patch) is refused: the surgery engine's observables take
    one line, and the difference of two lines is a band of checks whose signs are random."""
    if distance is not None and _is_convention(distance):
        convention, distance = distance, None
    if distance is None:
        if p:
            raise TQECError("p is the engine's noise at a distance: give the distance, or a NoiseModel to generate_stim_circuit")
        if (rounds is not None or merged is not None) and block_temporal_height is None:
            raise TQECError("rounds and merged are the engine's: give the distance too")
        return TopologicalComputationGraph(graph, convention=convention, observables=observables, block_temporal_height=block_temporal_height, rounds=rounds, merged=merged)
    computation = TopologicalComputationGraph(graph, convention=convention, observables=observables, block_temporal_height=block_temporal_height, rounds=rounds, merged=merged, _engine_only=True)
    return computation.compile(distance, p=p)

# Ported from tqec (tqec/tqec 0.2.0, src/tqec/computation/block_graph.py; Apache License 2.0; see LICENSE in this
# directory), modified for stabilizer_qec: a dict-backed graph in place of networkx, no pyzx, no pycollada
# (the Collada and JSON files are read and written by this package's `collada` module), no HTML viewer.
"""Block graph representation of a logical computation."""

from __future__ import annotations

import json
import pathlib
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Optional, Union

from ._errors import TQECError
from .cube import Cube, CubeKind, Port, YHalfCube, ZXCube, cube_kind_from_string
from .enums import Basis
from .pipe import Pipe, PipeKind
from .position import Direction3D, Position3D, SignedDirection3D

if TYPE_CHECKING:
    from .correlation import CorrelationSurface, ZXGraph
    from .open_graph import FilledGraph

BlockKind = Union[CubeKind, PipeKind]


class BlockGraph:
    """Block graph representation of a logical computation: cubes (the blocks of operations in a
    spacetime volume) as nodes and pipes (the connections that open their boundaries) as edges, as
    tqec draws them, a 3D spacetime diagram of a surface-code computation."""

    def __init__(self, name: str = "") -> None:
        self._name = name
        self._cubes: dict[Position3D, Cube] = {}
        self._edges: dict[Position3D, dict[Position3D, Pipe]] = {}
        self._ports: dict[str, Position3D] = {}

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, name: str) -> None:
        self._name = name

    @property
    def num_cubes(self) -> int:
        """Number of cubes (nodes) in the graph, including the ports."""
        return len(self._cubes)

    @property
    def num_pipes(self) -> int:
        return sum(len(e) for e in self._edges.values()) // 2

    @property
    def num_ports(self) -> int:
        return len([node for node in self.cubes if node.is_port])

    @property
    def num_half_y_cubes(self) -> int:
        return len([node for node in self.cubes if node.is_y_cube])

    @property
    def ordered_ports(self) -> list[str]:
        """The labels of the ports in alphabetical order."""
        return sorted(self._ports.keys())

    @property
    def cubes(self) -> list[Cube]:
        """The cubes (nodes) in the graph, in insertion order."""
        return list(self._cubes.values())

    @property
    def pipes(self) -> list[Pipe]:
        """The pipes (edges) in the graph, in insertion order."""
        seen: dict[tuple[Position3D, Position3D], Pipe] = {}
        for pos, edges in self._edges.items():
            for other, pipe in edges.items():
                key = (min(pos, other), max(pos, other))
                seen.setdefault(key, pipe)
        return list(seen.values())

    @property
    def occupied_positions(self) -> list[Position3D]:
        return list(self._cubes.keys())

    @property
    def is_open(self) -> bool:
        """Whether the graph has ports."""
        return bool(self._ports)

    @property
    def spacetime_volume(self) -> float:
        """Cubes count 1, half Y cubes 0.5, ports and pipes nothing."""
        return self.num_cubes - self.num_ports - self.num_half_y_cubes * 0.5

    def bounding_box_size(self) -> tuple[int, int, int]:
        """The width of the structure along X, Y and Z."""
        positions = self.occupied_positions
        return (
            max(pos.x for pos in positions) - min(pos.x for pos in positions) + 1,
            max(pos.y for pos in positions) - min(pos.y for pos in positions) + 1,
            max(pos.z for pos in positions) - min(pos.z for pos in positions) + 1,
        )

    @property
    def ports(self) -> dict[str, Position3D]:
        """Port labels to their positions."""
        return dict(self._ports)

    def get_degree(self, position: Position3D) -> int:
        return len(self._edges.get(position, {}))

    @property
    def leaf_cubes(self) -> list[Cube]:
        """The cubes with degree 1."""
        return [node for node in self.cubes if self.get_degree(node.position) == 1]

    def _check_cube_exists(self, position: Position3D) -> None:
        if position not in self:
            raise TQECError(f"No cube at position {position}.")

    def _check_pipe_exists(self, pos1: Position3D, pos2: Position3D) -> None:
        if not self.has_pipe_between(pos1, pos2):
            raise TQECError(f"No pipe between {pos1} and {pos2}.")

    def add_cube(self, position: Position3D, kind: Union[CubeKind, str], label: str = "") -> Position3D:
        """Add a cube; its kind a ``CubeKind`` or its string (``"ZXZ"``, ``"P"``, ``"Y"``)."""
        if position in self:
            raise TQECError(f"Cube already exists at position {position}.")
        if isinstance(kind, str):
            kind = cube_kind_from_string(kind)
        if kind == Port() and label in self._ports:
            raise TQECError(f"There is already a port with the same label {label} in the graph.")
        self._cubes[position] = Cube(position, kind, label)
        self._edges.setdefault(position, {})
        if kind == Port():
            self._ports[label] = position
        return position

    def add_pipe(self, pos1: Position3D, pos2: Position3D, kind: Union[PipeKind, str, None] = None) -> None:
        """Add a pipe between two cubes, its kind given or inferred from the cubes. Validity is not
        checked here (``validate`` does)."""
        u, v = self[pos1], self[pos2]
        if self.has_pipe_between(pos1, pos2):
            raise TQECError("There is already a pipe between the given positions in the graph.")
        if kind is None:
            pipe = Pipe.from_cubes(u, v)
        else:
            if isinstance(kind, str):
                kind = PipeKind.from_str(kind)
            pipe = Pipe(u, v, kind)
        self._edges[pos1][pos2] = pipe
        self._edges[pos2][pos1] = pipe

    def remove_cube(self, position: Position3D) -> None:
        """Remove a cube and the pipes connected to it."""
        self._check_cube_exists(position)
        cube = self._cubes.pop(position)
        for other in list(self._edges.pop(position, {})):
            self._edges[other].pop(position, None)
        if cube.is_port:
            self._ports.pop(cube.label)

    def remove_pipe(self, pos1: Position3D, pos2: Position3D) -> None:
        self._check_pipe_exists(pos1, pos2)
        self._edges[pos1].pop(pos2)
        self._edges[pos2].pop(pos1)

    def has_pipe_between(self, pos1: Position3D, pos2: Position3D) -> bool:
        return pos2 in self._edges.get(pos1, {})

    def get_pipe(self, pos1: Position3D, pos2: Position3D) -> Pipe:
        self._check_pipe_exists(pos1, pos2)
        return self._edges[pos1][pos2]

    def pipes_at(self, position: Position3D) -> list[Pipe]:
        """The pipes incident to a position."""
        if position not in self:
            raise TQECError(f"No cube at position {position}.")
        return list(self._edges[position].values())

    def clone(self) -> BlockGraph:
        """A data-independent copy of the graph."""
        graph = BlockGraph(self.name + "_clone")
        graph._cubes = dict(self._cubes)
        graph._edges = {p: dict(e) for p, e in self._edges.items()}
        graph._ports = dict(self._ports)
        return graph

    def __contains__(self, position: object) -> bool:
        return position in self._cubes

    def __getitem__(self, position: Position3D) -> Cube:
        self._check_cube_exists(position)
        return self._cubes[position]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BlockGraph):
            return False
        return self._cubes == other._cubes and {p: dict(e) for p, e in self._edges.items()} == {p: dict(e) for p, e in other._edges.items()} and self._ports == other._ports

    def __hash__(self) -> int:
        return hash(self._name)

    def __repr__(self) -> str:
        return f"BlockGraph({self._name!r}, {self.num_cubes} cubes, {self.num_pipes} pipes, {self.num_ports} ports)"

    def validate(self) -> None:
        """Check the graph can represent a logical computation (Fig. 9 of arXiv:2404.18369's
        conditions a, c, d and f): ports have one pipe; Y half cubes have one, time-like, pipe;
        no cube has pipes in all three directions; the colours of a cube's faces match those of
        the pipes in the same plane."""
        for cube in self.cubes:
            self._validate_locally_at_cube(cube)

    def _validate_locally_at_cube(self, cube: Cube) -> None:
        pipes = self.pipes_at(cube.position)
        if cube.is_port:
            if len(pipes) != 1:
                raise TQECError(f"Port at {cube.position} does not have exactly one pipe connected.")
            return
        if cube.is_y_cube:
            if len(pipes) != 1:
                raise TQECError(f"Y Half Cube at {cube.position} does not have exactly one pipe connected.")
            if not pipes[0].direction == Direction3D.Z:
                raise TQECError(f"Y Half Cube at {cube.position} has non-timelike pipes connected.")
            return
        assert isinstance(cube.kind, ZXCube)
        pipes_by_direction: dict[Direction3D, list[Pipe]] = {}
        for pipe in pipes:
            pipes_by_direction.setdefault(pipe.direction, []).append(pipe)
        for direction in Direction3D.all_directions():
            if len(pipes_by_direction.get(direction, [])) == 2:
                continue
            cube_color = cube.kind.get_basis_along(direction)
            for ortho_dir in direction.orthogonal_directions:
                for pipe in pipes_by_direction.get(ortho_dir, []):
                    pipe_color = pipe.kind.get_basis_along(direction, pipe.at_head(cube.position))
                    if pipe_color != cube_color:
                        raise TQECError(f"Cube {cube} has mismatched colors with pipe {pipe}.")

    def to_zx_graph(self) -> "ZXGraph":
        """The graph as a ZX graph with 3D positions: a cube is a spider (Z for a kind with one Z
        boundary, else X; a Y half cube a Z spider with phase π/2; a port a boundary), a pipe an
        edge, a Hadamard pipe a Hadamard edge."""
        from .correlation import ZXGraph

        return ZXGraph.from_block_graph(self)

    def to_dae_file(self, file_path: Union[str, pathlib.Path], pipe_length: float = 2.0, pop_faces_at_directions: Iterable[Union[SignedDirection3D, str]] = (), show_correlation_surface: Optional["CorrelationSurface"] = None) -> None:
        """Write the block graph to a Collada DAE file, as tqec and SketchUp write it."""
        from .collada import write_block_graph_to_dae_file

        write_block_graph_to_dae_file(self, file_path, pipe_length, pop_faces_at_directions, show_correlation_surface)

    @staticmethod
    def from_dae_file(filename: Union[str, pathlib.Path], graph_name: str = "") -> BlockGraph:
        """A block graph from a Collada DAE file (tqec's or SketchUp's)."""
        from .collada import read_block_graph_from_dae_file

        return read_block_graph_from_dae_file(filename, graph_name)

    def shift_by(self, dx: int = 0, dy: int = 0, dz: int = 0) -> BlockGraph:
        """A copy shifted by the offset."""
        new_graph = BlockGraph(self.name)
        for cube in self.cubes:
            new_graph.add_cube(cube.position.shift_by(dx=dx, dy=dy, dz=dz), cube.kind, cube.label)
        for pipe in self.pipes:
            u, v = pipe.u, pipe.v
            new_graph.add_pipe(u.position.shift_by(dx=dx, dy=dy, dz=dz), v.position.shift_by(dx=dx, dy=dy, dz=dz), pipe.kind)
        return new_graph

    def find_correlation_surfaces(self) -> list["CorrelationSurface"]:
        """The correlation surfaces (Pauli webs, logical observables) of the graph."""
        from .correlation import find_correlation_surfaces

        return find_correlation_surfaces(self.to_zx_graph())

    def fill_ports(self, fill: Union[Mapping[str, CubeKind], CubeKind]) -> None:
        """Fill the ports (by label, or all with one kind) with cubes."""
        if not isinstance(fill, Mapping):
            fill = {label: fill for label in self._ports}
        for label, kind in fill.items():
            if label not in self._ports:
                raise TQECError(f"There is no port with label {label}.")
            pos = self._ports[label]
            fill_node = Cube(pos, kind)
            self._cubes[pos] = fill_node
            # The pipes are removed and added back (so they move to the end of the adjacency
            # lists, as tqec's networkx graph has it: the pipe order is part of the parity).
            for pipe in self.pipes_at(pos):
                other = pipe.u if pipe.v.position == pos else pipe.v
                self._edges[pos].pop(other.position)
                self._edges[other.position].pop(pos)
                new_pipe = Pipe(other, fill_node, pipe.kind)
                self._edges[other.position][pos] = new_pipe
                self._edges[pos][other.position] = new_pipe
            self._ports.pop(label)

    def compile(self, distance: Optional[int] = None, *, k: Optional[int] = None, p: float = 0.0, noise_model=None, rounds: Optional[int] = None, merged: Optional[int] = None, observables="auto", convention="fixed_bulk", block_temporal_height=None):
        """The graph as a lattice-surgery circuit on this package's engine (``compile_block_graph``
        has the program too): patches for the cubes' lifetimes, merges and splits for the spatial
        pipes, the correlation surfaces as observables (``"auto"`` all of them, ``None`` none, or
        a list, as tqec takes them). The size is ``distance`` or tqec's ``k`` (distance 2k + 1);
        the noise the engine's SD6 noise ``p``, or a ``NoiseModel`` laid over the noiseless
        circuit as tqec lays it. Graphs the engine cannot express (Hadamard pipes, spatial cubes,
        Y cubes, open ports) are refused with tqec named."""
        from .compile import TopologicalComputationGraph, _noisy, _scale

        if (distance is None) == (k is None):
            raise TQECError("give the distance or k (the distance is 2k + 1), one of them")
        if k is not None:
            distance = 2 * _scale(k) + 1
        if noise_model is not None and p:
            raise TQECError("give the engine's noise p or a noise_model, not both")
        computation = TopologicalComputationGraph(self, convention=convention, observables=observables, block_temporal_height=block_temporal_height, rounds=rounds, merged=merged)
        circuit = computation.compile(distance, p=p).circuit
        return circuit if noise_model is None else _noisy(noise_model, circuit)

    def fill_ports_for_minimal_simulation(self) -> list["FilledGraph"]:
        """Fill the ports so that the fewest simulation runs cover every logical observable."""
        from .open_graph import fill_ports_for_minimal_simulation

        return fill_ports_for_minimal_simulation(self)

    def compose(self, other: BlockGraph, self_port: str, other_port: str) -> BlockGraph:
        """The graph composed with another at the named ports (the other shifted to meet; other
        overlapping ports glued too)."""
        if self_port not in self.ports:
            raise TQECError(f"Port {self_port} is not in the current graph.")
        if other_port not in other.ports:
            raise TQECError(f"Port {other_port} is not in the other graph.")
        p1, p2 = self.ports[self_port], other.ports[other_port]
        shift = (p1.x - p2.x, p1.y - p2.y, p1.z - p2.z)
        shifted_g = other.shift_by(*shift)
        ports_need_fill: dict[str, Position3D] = {}
        for cube in self.cubes:
            pos = cube.position
            if pos in shifted_g:
                if cube.is_port and shifted_g[pos].is_port:
                    ports_need_fill[cube.label] = pos
                    continue
                raise TQECError(f"Cube at position {cube.position} is overlapping between the two graphs.")
        composed_g = self.clone()
        for label, port in ports_need_fill.items():
            bases: list[Basis] = []
            pipe1 = self.pipes_at(port)[0]
            pipe2 = shifted_g.pipes_at(port)[0]
            for d in Direction3D.all_directions():
                b1 = pipe1.kind.get_basis_along(d, pipe1.at_head(port))
                b2 = pipe2.kind.get_basis_along(d, pipe2.at_head(port))
                if b1 is not None and b2 is not None and b1 != b2:
                    raise TQECError(f"Port at {port} cannot be filled with a cube that has valid boundary conditions.")
                bases.append(b1 or b2 or Basis.Z)
            composed_g.fill_ports({label: ZXCube(*bases)})
        for cube in shifted_g.cubes:
            if cube.position in composed_g:
                continue
            composed_g.add_cube(cube.position, cube.kind, cube.label)
        for pipe in shifted_g.pipes:
            composed_g.add_pipe(pipe.u.position, pipe.v.position, pipe.kind)
        composed_g.name = f"{self.name}_composed_with_{other.name}"
        return composed_g

    def is_single_connected(self) -> bool:
        """Whether the graph has one connected component."""
        if not self._cubes:
            return False
        start = next(iter(self._cubes))
        seen = {start}
        stack = [start]
        while stack:
            p = stack.pop()
            for q in self._edges.get(p, {}):
                if q not in seen:
                    seen.add(q)
                    stack.append(q)
        return len(seen) == len(self._cubes)

    def rotate(self, rotation_axis: Direction3D, counterclockwise: bool = True, num_90_degree_rotation: int = 1) -> BlockGraph:
        """A copy rotated about an axis by a multiple of 90 degrees."""
        from .rotations import get_rotation_matrix, rotate_block_kind_by_matrix, rotate_position_by_matrix

        rotated = BlockGraph(self.name + "_rotated")
        matrix = get_rotation_matrix(rotation_axis, counterclockwise, num_90_degree_rotation)
        pos_map: dict[Position3D, Position3D] = {}
        for cube in self.cubes:
            rotated_pos = rotate_position_by_matrix(cube.position, matrix)
            rotated.add_cube(rotated_pos, rotate_block_kind_by_matrix(cube.kind, matrix), cube.label)
            pos_map[cube.position] = rotated_pos
        for pipe in self.pipes:
            rotated.add_pipe(pos_map[pipe.u.position], pos_map[pipe.v.position], rotate_block_kind_by_matrix(pipe.kind, matrix))
        return rotated

    def fix_shadowed_faces(self) -> BlockGraph:
        """A copy with the bases of the faces two pipes in one direction hide set to match the
        pipes in their plane (a spatial pass-through made a non-spatial cube)."""
        fixed_cubes: dict[Cube, Cube] = {}
        for cube in self.cubes:
            if not isinstance(cube.kind, ZXCube):
                continue
            pipes_by_direction: dict[Direction3D, list[Pipe]] = {}
            for pipe in self.pipes_at(cube.position):
                pipes_by_direction.setdefault(pipe.direction, []).append(pipe)
            if len(pipes_by_direction) in (0, 3):
                continue
            shadowed_directions = {d for d, ps in pipes_by_direction.items() if len(ps) == 2}
            if not shadowed_directions:
                continue
            new_kind = cube.kind
            for shadowed_direction in shadowed_directions:
                if len(pipes_by_direction) == 1 and shadowed_direction != Direction3D.Z and cube.is_spatial:
                    kind = cube.kind
                    basis = kind.get_basis_along(shadowed_direction)
                    new_kind = kind.with_basis_along(shadowed_direction, basis.flipped())
                    fixed_cubes[cube] = Cube(cube.position, new_kind, cube.label)
                elif len(pipes_by_direction) == 2:
                    other_direction = next(d for d in pipes_by_direction if d != shadowed_direction)
                    need_match_pipe = next(iter(pipes_by_direction[other_direction]))
                    pipe_basis = need_match_pipe.kind.get_basis_along(shadowed_direction, need_match_pipe.at_head(cube.position))
                    assert pipe_basis is not None
                    new_kind = new_kind.with_basis_along(shadowed_direction, pipe_basis)
            if new_kind != cube.kind:
                fixed_cubes[cube] = Cube(cube.position, new_kind, cube.label)
        new_graph = BlockGraph(self.name)
        for cube in self.cubes:
            new_cube = fixed_cubes.get(cube, cube)
            new_graph.add_cube(cube.position, new_cube.kind, new_cube.label)
        for pipe in self.pipes:
            new_graph.add_pipe(pipe.u.position, pipe.v.position, pipe.kind)
        return new_graph

    def get_cubes_by_label(self, label: str) -> list[Cube]:
        return [cube for cube in self.cubes if cube.label == label]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "cubes": [cube.to_dict() for cube in self.cubes],
            "pipes": [pipe.to_dict() for pipe in self.pipes],
            "ports": {label: pos.as_tuple() for label, pos in self.ports.items()},
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> BlockGraph:
        graph = BlockGraph(data["name"])
        for cube in data["cubes"]:
            graph.add_cube(position=Position3D(*cube["position"]), kind=cube["kind"], label=cube["label"])
        for pipe in data["pipes"]:
            graph.add_pipe(pos1=Position3D(*pipe["u"]), pos2=Position3D(*pipe["v"]), kind=pipe["kind"])
        return graph

    def to_json(self, file_path: Union[str, pathlib.Path, None] = None, *, indent: Optional[int] = 2) -> Optional[str]:
        """The graph as tqec's JSON (each block with an identity transform), returned or written."""
        obj_dict = self.to_dict()
        eye = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        for cube in obj_dict["cubes"]:
            cube["transform"] = deepcopy(eye)
        for pipe in obj_dict["pipes"]:
            pipe["transform"] = deepcopy(eye)
        if file_path is None:
            return json.dumps(obj_dict, indent=indent)
        with open(file_path, "w") as fp:
            json.dump(obj_dict, fp, indent=indent)
        return None

    @staticmethod
    def from_json(filename: Union[str, pathlib.Path], graph_name: str = "") -> BlockGraph:
        """A block graph from tqec's JSON file."""
        from .collada import read_block_graph_from_json

        return read_block_graph_from_json(filename, graph_name)

    def relabel_cubes(self, label_mapping: Mapping[Union[Position3D, str], str]) -> None:
        """Relabel cubes, by position or by their current label."""
        port_labels = {cube.label for cube in self.cubes if cube.is_port}
        assigned_new_labels: set[str] = set()
        for key, new_label in label_mapping.items():
            if not new_label:
                raise TQECError("New label must be non-empty.")
            if isinstance(key, Position3D):
                matching_cubes = [cube for cube in self.cubes if cube.position == key]
            elif isinstance(key, str):
                matching_cubes = [cube for cube in self.cubes if cube.label == key]
            else:
                raise TQECError(f"Invalid identifier '{key}'. Must be Position3D or str.")
            if not matching_cubes:
                raise TQECError(f"No cube found for identifier '{key}'.")
            is_port_relabel = any(cube.is_port for cube in matching_cubes)
            if is_port_relabel:
                if new_label in port_labels:
                    raise TQECError(f"Port label '{new_label}' is already assigned to another port.")
                if new_label in assigned_new_labels:
                    raise TQECError(f"Port label '{new_label}' is reused multiple times.")
                assigned_new_labels.add(new_label)
                self._ports[new_label] = self._ports.pop(matching_cubes[0].label)
            elif new_label in port_labels:
                raise TQECError(f"The label '{new_label}' belongs to a port and cannot be reused by a non-port cube.")
            for cube in matching_cubes:
                updated = Cube(position=cube.position, kind=cube.kind, label=new_label)
                self._cubes[cube.position] = updated
                for other, pipe in list(self._edges[cube.position].items()):
                    u, v = (updated, pipe.v) if pipe.u.position == cube.position else (pipe.u, updated)
                    new_pipe = Pipe(u, v, pipe.kind)
                    self._edges[cube.position][other] = new_pipe
                    self._edges[other][cube.position] = new_pipe


def block_kind_from_str(string: str) -> BlockKind:
    """A block kind (a cube's or a pipe's) from its string."""
    string = string.upper()
    # The port is tested before the pipes: "PORT" has an O in it (tqec reads it as a pipe and
    # skips ports in its JSON reader for that reason).
    if string in ("P", "PORT"):
        return Port()
    if "O" in string:
        return PipeKind.from_str(string)
    if string == "Y":
        return YHalfCube()
    return ZXCube.from_str(string)

# Ported from tqec (tqec/tqec 0.2.0, src/tqec/interop/collada/read_write.py, _geometry.py, _correlation.py and
# src/tqec/interop/color.py; Apache License 2.0; see LICENSE in this directory), modified for stabilizer_qec: the
# Collada XML read and written with xml.etree in place of pycollada and numpy, the HTML viewer left out.
"""Block graphs as Collada (``.dae``) files, the format tqec and SketchUp share, and tqec's JSON.

A file holds one visual scene whose single root node is named ``SketchUp``; each block is a
child node with a 4×4 ``<matrix>`` (its position, scaled by ``1 + pipe_length``, a pipe also
scaled along its axis) instancing a library node named by the block's kind (``ZXZ``, ``XOZH``,
``Y``); the library nodes carry the faces as rectangles coloured by basis."""

from __future__ import annotations

import datetime as _dt
import json
import math
import pathlib
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Iterable, Optional, Sequence, Union

from ._errors import TQECError
from .cube import Cube, CubeKind, Port, YHalfCube, ZXCube
from .enums import Basis
from .pipe import PipeKind
from .position import Direction3D, FloatPosition3D, Position3D, SignedDirection3D, round_or_fail
from .rotations import adjust_hadamards_direction, get_axes_directions, rotate_on_import

if TYPE_CHECKING:
    from .block_graph import BlockGraph, BlockKind
    from .correlation import CorrelationSurface, ZXEdge

_NS = "http://www.collada.org/2005/11/COLLADASchema"
_ASSET_AUTHOR = "TQEC Community"
_ASSET_AUTHORING_TOOL = "https://github.com/tqec/tqec"
_ASSET_UNIT_NAME = "inch"
_ASSET_UNIT_METER = 0.02539999969303608
_MATERIAL_SYMBOL = "MaterialSymbol"
_CORRELATION_SUFFIX = "_CORRELATION"

Matrix3 = list[list[float]]


# ---------------------------------------------------------------------------------------------
# Colours and faces
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class RGBA:
    r: int
    g: int
    b: int
    a: float

    def as_floats(self) -> tuple[float, float, float, float]:
        return (self.r / 255, self.g / 255, self.b / 255, self.a)

    def to_hex(self) -> str:
        return f"#{self.r:02x}{self.g:02x}{self.b:02x}"


class TQECColor(Enum):
    """tqec's face colours."""

    X = "X"
    Y = "Y"
    Z = "Z"
    H = "H"
    X_CORRELATION = "X_CORRELATION"
    Z_CORRELATION = "Z_CORRELATION"

    @property
    def rgba(self) -> RGBA:
        return {
            TQECColor.X: RGBA(255, 127, 127, 1.0),
            TQECColor.Y: RGBA(99, 198, 118, 1.0),
            TQECColor.Z: RGBA(115, 150, 255, 1.0),
            TQECColor.H: RGBA(255, 255, 101, 1.0),
            TQECColor.X_CORRELATION: RGBA(255, 0, 0, 0.8),
            TQECColor.Z_CORRELATION: RGBA(0, 0, 255, 0.8),
        }[self]

    def with_zx_flipped(self) -> TQECColor:
        return {TQECColor.X: TQECColor.Z, TQECColor.Z: TQECColor.X, TQECColor.X_CORRELATION: TQECColor.Z_CORRELATION, TQECColor.Z_CORRELATION: TQECColor.X_CORRELATION}.get(self, self)


@dataclass(frozen=True)
class Face:
    """A rectangle: its (width, height, normal) axes follow the right-hand rule, (X, Y, Z),
    (Y, Z, X) or (Z, X, Y)."""

    color: TQECColor
    width: float
    height: float
    normal_direction: SignedDirection3D
    position: FloatPosition3D = field(default=FloatPosition3D(0.0, 0.0, 0.0))

    @staticmethod
    def get_triangle_indices() -> list[int]:
        return [0, 0, 2, 2, 1, 1, 2, 2, 0, 0, 3, 3]

    def get_vertices(self) -> list[float]:
        a3 = self.normal_direction.direction.value
        a1, a2 = (a3 + 1) % 3, (a3 + 2) % 3
        out: list[float] = []
        pos = self.position.as_tuple()
        for w, h in ((0.0, 0.0), (self.width, 0.0), (self.width, self.height), (0.0, self.height)):
            v = [0.0, 0.0, 0.0]
            v[a1] = w
            v[a2] = h
            out.extend(v[i] + pos[i] for i in range(3))
        return out

    def get_normal_vectors(self) -> list[float]:
        normal = [0.0, 0.0, 0.0]
        normal[self.normal_direction.direction.value] = 1.0 if self.normal_direction.towards_positive else -1.0
        return normal * 4

    def shift_by(self, dx: float, dy: float, dz: float) -> Face:
        return Face(self.color, self.width, self.height, self.normal_direction, self.position.shift_by(dx, dy, dz))

    def with_negated_normal_direction(self) -> Face:
        return Face(self.color, self.width, self.height, -self.normal_direction, self.position)


class BlockGeometries:
    """The faces of every block kind."""

    def __init__(self) -> None:
        self.geometries: dict["BlockKind", list[Face]] = {}
        self._load_zx_cube_geometries()
        self._load_y_cube_geometry()
        self._load_pipe_without_hadamard_geometries()
        self._load_pipe_with_hadamard_geometries()

    def get_geometry(self, kind: "BlockKind", pop_faces_at_directions: frozenset[SignedDirection3D] = frozenset()) -> list[Face]:
        return [face for face in self.geometries[kind] if face.normal_direction not in pop_faces_at_directions]

    def _load_zx_cube_geometries(self) -> None:
        for kind in ZXCube.all_kinds():
            faces: list[Face] = []
            for direction in Direction3D.all_directions():
                face = Face(TQECColor(kind.get_basis_along(direction).value), 1.0, 1.0, SignedDirection3D(direction, False))
                faces.append(face)
                translation = [0.0, 0.0, 0.0]
                translation[direction.value] = 1.0
                faces.append(face.shift_by(*translation).with_negated_normal_direction())
            self.geometries[kind] = faces

    def _load_y_cube_geometry(self) -> None:
        faces: list[Face] = []
        for direction in Direction3D.all_directions():
            width, height = {Direction3D.X: (1.0, 0.5), Direction3D.Y: (0.5, 1.0), Direction3D.Z: (1.0, 1.0)}[direction]
            face = Face(TQECColor.Y, width, height, SignedDirection3D(direction, False))
            faces.append(face)
            translation = [0.0, 0.0, 0.0]
            translation[direction.value] = 1.0 if direction != Direction3D.Z else 0.5
            faces.append(face.shift_by(*translation).with_negated_normal_direction())
        self.geometries[YHalfCube()] = faces

    def _load_pipe_without_hadamard_geometries(self) -> None:
        for name in ["ozx", "oxz", "xoz", "zox", "xzo", "zxo"]:
            faces: list[Face] = []
            kind = PipeKind.from_str(name)
            for direction in Direction3D.all_directions():
                basis = kind.get_basis_along(direction)
                if basis is None:
                    continue
                width, height = (2.0, 1.0) if direction.value == (kind.direction.value - 1) % 3 else (1.0, 2.0)
                face = Face(TQECColor(basis.value), width, height, SignedDirection3D(direction, False))
                faces.append(face)
                translation = [0.0, 0.0, 0.0]
                translation[direction.value] = 1.0
                faces.append(face.shift_by(*translation).with_negated_normal_direction())
            self.geometries[kind] = faces

    def _load_pipe_with_hadamard_geometries(self) -> None:
        def face_position(shift_direction: Direction3D, shift: float) -> FloatPosition3D:
            tmp = [0.0, 0.0, 0.0]
            tmp[shift_direction.value] = shift
            return FloatPosition3D(*tmp)

        for name in ["ozxh", "oxzh", "zoxh", "xozh", "zxoh", "xzoh"]:
            faces: list[Face] = []
            kind = PipeKind.from_str(name)
            for direction in Direction3D.all_directions():
                basis = kind.get_basis_along(direction)
                if basis is None:
                    continue
                if direction.value == (kind.direction.value - 1) % 3:
                    (w1, h1), (w2, h2), (w3, h3) = (0.9, 1.0), (0.2, 1.0), (0.9, 1.0)
                else:
                    (w1, h1), (w2, h2), (w3, h3) = (1.0, 0.9), (1.0, 0.2), (1.0, 0.9)
                normal_direction = SignedDirection3D(direction, False)
                face1 = Face(TQECColor(basis.value), w1, h1, normal_direction)
                face2 = Face(TQECColor.H, w2, h2, normal_direction, face_position(kind.direction, 0.9))
                face3 = Face(face1.color.with_zx_flipped(), w3, h3, normal_direction, face_position(kind.direction, 1.1))
                faces += [face1, face2, face3]
                translation = [0.0, 0.0, 0.0]
                translation[direction.value] = 1.0
                faces.extend(face.shift_by(*translation).with_negated_normal_direction() for face in (face1, face2, face3))
            self.geometries[kind] = faces


def get_correlation_surface_geometry(basis: Basis) -> Face:
    return Face(TQECColor(basis.value + _CORRELATION_SUFFIX), 1.0, 1.0, SignedDirection3D(Direction3D.Z, True))


# ---------------------------------------------------------------------------------------------
# Transformations
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Transformation:
    """Translation, per-axis scale and rotation of a block, as a 4×4 affine matrix holds them."""

    translation: tuple[float, float, float]
    scale: tuple[float, float, float]
    rotation: Matrix3

    @staticmethod
    def from_4d_affine_matrix(mat: Sequence[Sequence[float]]) -> _Transformation:
        translation = (mat[0][3], mat[1][3], mat[2][3])
        scale = tuple(math.sqrt(sum(mat[i][k] ** 2 for k in range(3))) for i in range(3))
        rotation = [[mat[i][j] / scale[j] for j in range(3)] for i in range(3)]
        return _Transformation(translation, scale, rotation)  # type: ignore[arg-type]

    def to_4d_affine_matrix(self) -> list[list[float]]:
        mat = [[1.0 if i == j else 0.0 for j in range(4)] for i in range(4)]
        for i in range(3):
            for j in range(3):
                mat[i][j] = self.rotation[i][j] * self.scale[j]
            mat[i][3] = self.translation[i]
        return mat


def _is_identity(rotation: Matrix3, atol: float = 1e-9) -> bool:
    return all(abs(rotation[i][j] - (1.0 if i == j else 0.0)) <= atol for i in range(3) for j in range(3))


def _int_position_before_scale(pos: FloatPosition3D, pipe_length: float) -> Position3D:
    return Position3D(round_or_fail(pos.x / (1 + pipe_length), atol=0.35), round_or_fail(pos.y / (1 + pipe_length), atol=0.35), round_or_fail(pos.z / (1 + pipe_length), atol=0.35))


def _offset_y_cube_position(pos: FloatPosition3D, pipe_length: float) -> FloatPosition3D:
    if abs((pos.z - 0.5) - math.floor(pos.z)) <= 1e-9:
        pos = pos.shift_by(dz=-0.5)
    return FloatPosition3D(pos.x, pos.y, pos.z / (1 + pipe_length))


# ---------------------------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------------------------


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _children(element: ET.Element, name: str) -> list[ET.Element]:
    return [child for child in element if _local(child.tag) == name]


def _find(element: ET.Element, name: str) -> Optional[ET.Element]:
    found = _children(element, name)
    return found[0] if found else None


def _parse_matrix(text: str) -> list[list[float]]:
    values = [float(x) for x in text.split()]
    if len(values) != 16:
        raise TQECError("A <matrix> must hold 16 numbers.")
    return [values[4 * i : 4 * i + 4] for i in range(4)]


def read_block_graph_from_dae_file(filepath: Union[str, pathlib.Path], graph_name: str = "") -> "BlockGraph":
    """A block graph from a Collada file written by tqec or SketchUp: the scene's single root
    node ``SketchUp``, each child with a matrix and one ``instance_node`` of a library node named
    by its kind; rotated blocks have their kinds rotated; pipes all of one length; a pipe end
    with no cube becomes a port ``Port<i>``."""
    from .block_graph import BlockGraph, block_kind_from_str

    try:
        root = ET.parse(str(filepath)).getroot()
    except (OSError, ET.ParseError) as e:
        raise TQECError(f"Cannot read the DAE file {filepath}: {e}") from e
    if _local(root.tag) != "COLLADA":
        raise TQECError("Not a COLLADA document.")

    scenes = _find(root, "library_visual_scenes")
    scene_ref = _find(root, "scene")
    scene: Optional[ET.Element] = None
    if scenes is not None:
        visual_scenes = _children(scenes, "visual_scene")
        wanted = None
        if scene_ref is not None and (inst := _find(scene_ref, "instance_visual_scene")) is not None:
            wanted = inst.get("url", "").lstrip("#")
        for vs in visual_scenes:
            if wanted is None or vs.get("id") == wanted:
                scene = vs
                break
    if scene is None:
        raise TQECError("No scene found in the DAE file.")
    scene_nodes = _children(scene, "node")
    if not (len(scene_nodes) == 1 and scene_nodes[0].get("name") == "SketchUp"):
        raise TQECError("The <visual_scene> node must have a single child node with the name 'SketchUp'.")
    sketchup_node = scene_nodes[0]

    library: dict[str, ET.Element] = {}
    if (lib := _find(root, "library_nodes")) is not None:
        for node in _children(lib, "node"):
            if node.get("id"):
                library[node.get("id", "")] = node

    pipe_length: Optional[float] = None
    parsed_cubes: list[tuple[FloatPosition3D, CubeKind, dict[str, int]]] = []
    parsed_pipes: list[tuple[FloatPosition3D, PipeKind, dict[str, int]]] = []
    instance_tags = {"node", "instance_node", "instance_geometry", "instance_camera", "instance_light", "instance_controller"}

    for node in _children(sketchup_node, "node"):
        matrix = _find(node, "matrix")
        children = [child for child in node if _local(child.tag) in instance_tags]
        if not (matrix is not None and matrix.text and len(children) == 1 and _local(children[0].tag) == "instance_node"):
            continue
        url = children[0].get("url", "").lstrip("#")
        library_node = library.get(url)
        if library_node is None:
            raise TQECError(f"The instance node refers to an unknown library node {url!r}.")
        name = library_node.get("name") or library_node.get("id") or ""
        if name.endswith(_CORRELATION_SUFFIX):
            continue

        transformation = _Transformation.from_4d_affine_matrix(_parse_matrix(matrix.text))
        translation = FloatPosition3D(*transformation.translation)
        axes_directions = get_axes_directions(transformation.rotation)
        kind = block_kind_from_str(name)

        if not _is_identity(transformation.rotation):
            translation, kind = rotate_on_import(transformation.rotation, transformation.translation, transformation.scale, kind)

        if isinstance(kind, PipeKind):
            if axes_directions[str(kind.direction)] == -1 and "H" in str(kind):
                kind = adjust_hadamards_direction(kind)

        if isinstance(kind, PipeKind):
            pipe_direction = kind.direction
            scale = transformation.scale[pipe_direction.value]
            if pipe_length is None:
                pipe_length = scale * 2.0
            elif abs(pipe_length - scale * 2.0) > 1e-9:
                raise TQECError("All pipes must have the same length.")
            expected_scale = [1.0, 1.0, 1.0]
            expected_scale[pipe_direction.value] = scale
            if any(abs(a - b) > 1e-9 for a, b in zip(transformation.scale, expected_scale)):
                raise TQECError(f"Only the dimension along the pipe can be scaled, which is not the case at {translation}.")
            parsed_pipes.append((translation, kind, axes_directions))
        else:
            if any(abs(s - 1.0) > 1e-9 for s in transformation.scale):
                raise TQECError(f"Cube at {translation} has a non-identity scale.")
            parsed_cubes.append((translation, kind, axes_directions))

    pipe_length = 2.0 if pipe_length is None else pipe_length
    graph = BlockGraph(graph_name)
    for pos, cube_kind, _ in parsed_cubes:
        if isinstance(cube_kind, YHalfCube):
            graph.add_cube(_int_position_before_scale(_offset_y_cube_position(pos, pipe_length), pipe_length), cube_kind)
        else:
            graph.add_cube(_int_position_before_scale(pos, pipe_length), cube_kind)
    port_index = 0
    for pos, pipe_kind, axes_directions in parsed_pipes:
        directional_multiplier = axes_directions[str(pipe_kind.direction)]
        head_pos = _int_position_before_scale(pos.shift_in_direction(pipe_kind.direction, -1 * directional_multiplier), pipe_length)
        tail_pos = head_pos.shift_in_direction(pipe_kind.direction, 1 * directional_multiplier)
        if head_pos not in graph:
            graph.add_cube(head_pos, Port(), label=f"Port{port_index}")
            port_index += 1
        if tail_pos not in graph:
            graph.add_cube(tail_pos, Port(), label=f"Port{port_index}")
            port_index += 1
        graph.add_pipe(head_pos, tail_pos, pipe_kind)
    return graph


def read_block_graph_from_json(filepath: Union[str, pathlib.Path], graph_name: str = "") -> "BlockGraph":
    """A block graph from tqec's JSON (``cubes`` with ``position``, ``kind``, ``transform``;
    ``pipes`` with ``u``, ``v``, ``kind``, ``transform``); ports in the file are skipped and
    re-created at pipe ends with no cube."""
    from .block_graph import BlockGraph, block_kind_from_str

    try:
        with open(filepath) as f:
            data = json.load(f)
    except Exception:
        raise TQECError("JSON file not found.") from None
    try:
        cubes = data["cubes"]
        pipes = data["pipes"]
        if not len(cubes) > 0 or not len(pipes) > 0:
            raise TQECError("No cubes or pipes found.")
    except Exception:
        raise TQECError("JSON file is not appropriately formatted.") from None

    parsed_cubes: list[tuple[FloatPosition3D, CubeKind, dict[str, int]]] = []
    parsed_pipes: list[tuple[FloatPosition3D, FloatPosition3D, PipeKind, dict[str, int]]] = []
    eye = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

    for cube in cubes:
        if cube["kind"] == "PORT":
            continue
        if not all(isinstance(i, int) for i in cube["position"]):
            raise TQECError(f"Incorrect positioning for cube at {cube['position']}. All positional information must be composed of integer values.")
        if not all(isinstance(i, int) for row in cube["transform"] for i in row):
            raise TQECError(f"Incorrect transformation matrix for cube at {cube['position']}. All elements in transformation matrix need to be integers.")
        translation = FloatPosition3D(*cube["position"])
        axes_directions = get_axes_directions(cube["transform"])
        kind = block_kind_from_str(cube["kind"])
        if cube["transform"] != eye:
            translation, kind = rotate_on_import(cube["transform"], cube["position"], [1.0, 1.0, 1.0], kind)
        if not isinstance(kind, PipeKind):
            parsed_cubes.append((translation, kind, axes_directions))

    for pipe in pipes:
        for pos in (pipe["u"], pipe["v"]):
            if not all(isinstance(i, int) for i in pos):
                raise TQECError(f"Incorrect positioning for pipe at ({pipe['u']}). All positional information must be composed of integer values.")
        if not all(isinstance(i, int) for row in pipe["transform"] for i in row):
            raise TQECError(f"Incorrect transformation for pipe at ({pipe['u']}). All elements in transformation matrix need to be integers.")
        u_pos = FloatPosition3D(*pipe["u"])
        v_pos = FloatPosition3D(*pipe["v"])
        axes_directions = get_axes_directions(pipe["transform"])
        kind = block_kind_from_str(pipe["kind"])
        if pipe["transform"] != eye:
            u_pos, kind = rotate_on_import(pipe["transform"], pipe["u"], [1.0, 1.0, 1.0], kind)
        if isinstance(kind, PipeKind):
            if axes_directions[str(kind.direction)] == -1 and "H" in str(kind):
                kind = adjust_hadamards_direction(kind)
        if isinstance(kind, PipeKind):
            parsed_pipes.append((u_pos, v_pos, kind, axes_directions))

    graph = BlockGraph(graph_name)
    for pos, cube_kind, _ in parsed_cubes:
        if isinstance(cube_kind, YHalfCube):
            graph.add_cube(_int_position_before_scale(_offset_y_cube_position(pos, 0.0), 0.0), cube_kind)
        else:
            graph.add_cube(_int_position_before_scale(pos, 0.0), cube_kind)
    port_index = 0
    for u_pos, v_pos, pipe_kind, _ in parsed_pipes:
        head_pos = _int_position_before_scale(u_pos, 0)
        tail_pos = _int_position_before_scale(v_pos, 0)
        if head_pos not in graph:
            graph.add_cube(head_pos, Port(), label=f"Port{port_index}")
            port_index += 1
        if tail_pos not in graph:
            graph.add_cube(tail_pos, Port(), label=f"Port{port_index}")
            port_index += 1
        graph.add_pipe(head_pos, tail_pos, pipe_kind)
    return graph


# ---------------------------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------------------------


def _fmt(x: float) -> str:
    return format(float(x), "g")


def _fmt_matrix(mat: Sequence[Sequence[float]]) -> str:
    return " ".join(str(float(v)) for row in mat for v in row)


@dataclass(frozen=True)
class _BlockLibraryKey:
    kind: "BlockKind"
    pop_faces_at_directions: frozenset[SignedDirection3D] = frozenset()

    def __str__(self) -> str:
        string = f"{self.kind}"
        if self.pop_faces_at_directions:
            string += "-without-" + "-".join(sorted(str(d) for d in self.pop_faces_at_directions))
        return string


class _ColladaWriter:
    """The document under construction: materials for every colour, one geometry per distinct
    face, a library node per (kind, popped faces), and the scene's instances."""

    def __init__(self, pop_faces_at_directions: Iterable[SignedDirection3D] = ()) -> None:
        self.geometries = BlockGeometries()
        self.root = ET.Element("COLLADA", {"xmlns": _NS, "version": "1.4.1"})
        now = _dt.datetime.now(_dt.timezone.utc).replace(tzinfo=None).isoformat()
        asset = ET.SubElement(self.root, "asset")
        contributor = ET.SubElement(asset, "contributor")
        ET.SubElement(contributor, "author").text = _ASSET_AUTHOR
        ET.SubElement(contributor, "authoring_tool").text = _ASSET_AUTHORING_TOOL
        ET.SubElement(asset, "created").text = now
        ET.SubElement(asset, "modified").text = now
        ET.SubElement(asset, "unit", {"name": _ASSET_UNIT_NAME, "meter": str(_ASSET_UNIT_METER)})
        ET.SubElement(asset, "up_axis").text = "Z_UP"
        self.effects = ET.SubElement(self.root, "library_effects")
        self.geometry_lib = ET.SubElement(self.root, "library_geometries")
        self.materials = ET.SubElement(self.root, "library_materials")
        self.nodes = ET.SubElement(self.root, "library_nodes")
        scenes = ET.SubElement(self.root, "library_visual_scenes")
        visual_scene = ET.SubElement(scenes, "visual_scene", {"id": "scene"})
        self.root_node = ET.SubElement(visual_scene, "node", {"id": "SketchUp", "name": "SketchUp"})
        ET.SubElement(ET.SubElement(self.root, "scene"), "instance_visual_scene", {"url": "#scene"})
        self.face_ids: dict[Face, str] = {}
        self.block_library: dict[_BlockLibraryKey, str] = {}
        self.surface_library: dict[Basis, str] = {}
        self._pop_faces_at_directions = frozenset(pop_faces_at_directions)
        self._num_instances = 0
        self._add_materials()

    def _add_materials(self) -> None:
        for face_color in TQECColor:
            rgba = face_color.rgba.as_floats()
            color_text = " ".join(str(c) for c in rgba)
            effect = ET.SubElement(self.effects, "effect", {"id": f"{face_color.value}_effect", "name": f"{face_color.value}_effect"})
            profile = ET.SubElement(effect, "profile_COMMON")
            technique = ET.SubElement(profile, "technique", {"sid": "common"})
            lambert = ET.SubElement(technique, "lambert")
            ET.SubElement(ET.SubElement(lambert, "diffuse"), "color").text = color_text
            ET.SubElement(ET.SubElement(lambert, "shininess"), "float").text = "0.0"
            ET.SubElement(ET.SubElement(lambert, "reflectivity"), "float").text = "0.0"
            ET.SubElement(ET.SubElement(lambert, "transparent"), "color").text = color_text
            ET.SubElement(ET.SubElement(lambert, "transparency"), "float").text = str(rgba[3])
            extra = ET.SubElement(ET.SubElement(profile, "extra"), "technique", {"profile": "GOOGLEEARTH"})
            ET.SubElement(extra, "double_sided").text = "1"
            material = ET.SubElement(self.materials, "material", {"id": f"{face_color.value}_material", "name": f"{face_color.value}_material"})
            ET.SubElement(material, "instance_effect", {"url": f"#{face_color.value}_effect"})

    def _add_source(self, mesh: ET.Element, source_id: str, values: list[float]) -> None:
        source = ET.SubElement(mesh, "source", {"id": source_id})
        ET.SubElement(source, "float_array", {"count": str(len(values)), "id": f"{source_id}-array"}).text = " ".join(_fmt(v) for v in values)
        accessor = ET.SubElement(ET.SubElement(source, "technique_common"), "accessor", {"count": str(len(values) // 3), "source": f"#{source_id}-array", "stride": "3"})
        for name in "XYZ":
            ET.SubElement(accessor, "param", {"name": name, "type": "float"})

    def _face_geometry(self, face: Face) -> str:
        if face in self.face_ids:
            return self.face_ids[face]
        id_str = f"FaceID{len(self.face_ids)}"
        geometry = ET.SubElement(self.geometry_lib, "geometry", {"id": id_str, "name": id_str})
        mesh = ET.SubElement(geometry, "mesh")
        self._add_source(mesh, f"{id_str}_positions", face.get_vertices())
        self._add_source(mesh, f"{id_str}_normals", face.get_normal_vectors())
        vertices = ET.SubElement(mesh, "vertices", {"id": f"{id_str}_positions-vertices"})
        ET.SubElement(vertices, "input", {"semantic": "POSITION", "source": f"#{id_str}_positions"})
        triangles = ET.SubElement(mesh, "triangles", {"count": "2", "material": _MATERIAL_SYMBOL})
        ET.SubElement(triangles, "input", {"offset": "0", "semantic": "VERTEX", "source": f"#{id_str}_positions-vertices"})
        ET.SubElement(triangles, "input", {"offset": "1", "semantic": "NORMAL", "source": f"#{id_str}_normals"})
        ET.SubElement(triangles, "p").text = " ".join(str(i) for i in Face.get_triangle_indices())
        self.face_ids[face] = id_str
        return id_str

    def _instance_geometry(self, parent: ET.Element, face: Face) -> None:
        geom_id = self._face_geometry(face)
        instance = ET.SubElement(parent, "instance_geometry", {"url": f"#{geom_id}"})
        material = ET.SubElement(ET.SubElement(ET.SubElement(instance, "bind_material"), "technique_common"), "instance_material", {"symbol": _MATERIAL_SYMBOL, "target": f"#{face.color.value}_material"})
        ET.SubElement(material, "bind_vertex_input", {"semantic": "UVSET0", "input_semantic": "TEXCOORD", "input_set": "0"})

    def _block_library_node(self, block_kind: "BlockKind", pop_faces_at_directions: Iterable[SignedDirection3D] = ()) -> str:
        key = _BlockLibraryKey(block_kind, frozenset(pop_faces_at_directions) | self._pop_faces_at_directions)
        if key in self.block_library:
            return self.block_library[key]
        node = ET.SubElement(self.nodes, "node", {"id": str(key), "name": str(key.kind)})
        for face in self.geometries.get_geometry(block_kind, key.pop_faces_at_directions):
            self._instance_geometry(node, face)
        self.block_library[key] = str(key)
        return str(key)

    def _instance(self, node_id: str, matrix: Sequence[Sequence[float]], suffix: str = "") -> None:
        child = ET.SubElement(self.root_node, "node", {"id": f"ID{self._num_instances}", "name": f"instance_{self._num_instances}{suffix}"})
        ET.SubElement(child, "matrix").text = _fmt_matrix(matrix)
        ET.SubElement(child, "instance_node", {"url": f"#{node_id}"})
        self._num_instances += 1

    def add_block_instance(self, matrix: Sequence[Sequence[float]], block_kind: "BlockKind", pop_faces_at_directions: Iterable[SignedDirection3D] = ()) -> None:
        self._instance(self._block_library_node(block_kind, pop_faces_at_directions), matrix)

    def _surface_library_node(self, basis: Basis) -> str:
        if basis in self.surface_library:
            return self.surface_library[basis]
        surface = get_correlation_surface_geometry(basis)
        node = ET.SubElement(self.nodes, "node", {"id": surface.color.value, "name": surface.color.value})
        self._instance_geometry(node, surface)
        self.surface_library[basis] = surface.color.value
        return surface.color.value

    def add_correlation_surface(self, block_graph: "BlockGraph", correlation_surface: "CorrelationSurface", pipe_length: float = 2.0) -> None:
        helper = CorrelationSurfaceTransformationHelper(block_graph, pipe_length)
        for basis, transformation in helper.get_transformations_for_correlation_surface(correlation_surface):
            self._instance(self._surface_library_node(basis), transformation.to_4d_affine_matrix(), "_correlation_surface")

    def write(self, file_like) -> None:
        tree = ET.ElementTree(self.root)
        ET.indent(tree, space="  ")
        if hasattr(file_like, "write"):
            tree.write(file_like, encoding="utf-8", xml_declaration=True)
        else:
            tree.write(str(file_like), encoding="utf-8", xml_declaration=True)


def write_block_graph_to_dae_file(block_graph: "BlockGraph", file_like, pipe_length: float = 2.0, pop_faces_at_directions: Iterable[Union[SignedDirection3D, str]] = (), show_correlation_surface: Optional["CorrelationSurface"] = None) -> None:
    """Write a block graph as tqec writes it: cubes at ``position × (1 + pipe_length)`` with the
    faces a pipe covers popped, pipes one unit past their head cube scaled to ``pipe_length``
    along their axis, a Y half cube under a time pipe raised half a unit, and optionally a
    correlation surface as translucent rectangles."""
    directions = [SignedDirection3D.from_string(d) if isinstance(d, str) else d for d in pop_faces_at_directions]
    base = _ColladaWriter(directions)

    def scale_position(pos: Position3D) -> FloatPosition3D:
        return FloatPosition3D(*(p * (1 + pipe_length) for p in pos.as_tuple()))

    def translation_matrix(pos: FloatPosition3D, scales: Sequence[float] = (1.0, 1.0, 1.0)) -> list[list[float]]:
        mat = [[1.0 if i == j else 0.0 for j in range(4)] for i in range(4)]
        for i, (p, s) in enumerate(zip(pos.as_tuple(), scales)):
            mat[i][i] = s
            mat[i][3] = p
        return mat

    for cube in block_graph.cubes:
        if cube.is_port:
            continue
        scaled_position = scale_position(cube.position)
        if cube.is_y_cube and block_graph.has_pipe_between(cube.position, cube.position.shift_by(dz=1)):
            scaled_position = scaled_position.shift_by(dz=0.5)
        pop_directions = [SignedDirection3D(pipe.direction, cube == pipe.u) for pipe in block_graph.pipes_at(cube.position)]
        base.add_block_instance(translation_matrix(scaled_position), cube.kind, pop_directions)

    for pipe in block_graph.pipes:
        pipe_pos = scale_position(pipe.u.position).shift_in_direction(pipe.direction, 1.0)
        scales = [1.0, 1.0, 1.0]
        scales[pipe.direction.value] = pipe_length / 2.0
        base.add_block_instance(translation_matrix(pipe_pos, scales), pipe.kind)

    if show_correlation_surface is not None:
        base.add_correlation_surface(block_graph, show_correlation_surface, pipe_length)
    base.write(file_like)


# ---------------------------------------------------------------------------------------------
# Correlation surfaces as rectangles
# ---------------------------------------------------------------------------------------------


class CorrelationSurfaceTransformationHelper:
    """The pieces of a correlation surface as transformations of a unit square in the XY plane."""

    def __init__(self, block_graph: "BlockGraph", pipe_length: float) -> None:
        self.block_graph = block_graph
        self.pipe_length = pipe_length
        self.position_map = block_graph.to_zx_graph().positions

    def get_transformations_for_correlation_surface(self, correlation_surface: "CorrelationSurface") -> list[tuple[Basis, _Transformation]]:
        transformations: list[tuple[Basis, _Transformation]] = []
        for edge in correlation_surface.span:
            if edge.is_self_loop:
                continue
            transformations.extend(self._compute_pipe_transformations(edge))
        for v in correlation_surface.span_vertices:
            cube = self._get_cube(v)
            if cube.is_port or cube.is_y_cube:
                continue
            transformations.extend(self._compute_cube_transformations(v, correlation_surface.edges_at(v), correlation_surface.bases_at(v)))
        return transformations

    def _get_position(self, v: int) -> Position3D:
        return self.position_map[v]

    def _get_cube(self, v: int) -> Cube:
        return self.block_graph[self.position_map[v]]

    def _edge_direction(self, edge: "ZXEdge") -> Direction3D:
        p1, p2 = self._get_position(edge.u.id), self._get_position(edge.v.id)
        if p1.x != p2.x:
            return Direction3D.X
        if p1.y != p2.y:
            return Direction3D.Y
        return Direction3D.Z

    def _surface_normal_direction(self, correlation_edge: "ZXEdge") -> Direction3D:
        u, v = correlation_edge
        pipe = self.block_graph.get_pipe(self._get_position(u.id), self._get_position(v.id))
        return next(d for d in Direction3D.all_directions() if pipe.kind.get_basis_along(d) == u.basis.flipped())

    def _scale_position(self, pos: Position3D) -> FloatPosition3D:
        return FloatPosition3D(*(p * (1 + self.pipe_length) for p in pos.as_tuple()))

    def _compute_pipe_transformations(self, edge: "ZXEdge") -> list[tuple[Basis, _Transformation]]:
        transformations: list[tuple[Basis, _Transformation]] = []
        normal_direction = self._surface_normal_direction(edge)
        edge_direction = self._edge_direction(edge)
        surface_position = self._scale_position(self._get_position(edge.u.id)).shift_in_direction(edge_direction, 1).shift_in_direction(normal_direction, 0.5)
        rotation = _rotation_to_plane(normal_direction)
        scale_factor = self.pipe_length / 2 if edge.has_hadamard else self.pipe_length
        scale_direction = edge_direction if edge_direction != Direction3D.Z else normal_direction
        scale = _get_scale(scale_direction, scale_factor)
        transformations.append((edge.u.basis, _Transformation(surface_position.as_tuple(), scale, rotation)))
        if edge.has_hadamard:
            transformations.append((edge.v.basis, _Transformation(surface_position.shift_in_direction(edge_direction, self.pipe_length / 2).as_tuple(), scale, rotation)))
        return transformations

    def _compute_cube_transformations(self, v: int, correlation_edges: set, surface_bases: set[Basis]) -> list[tuple[Basis, _Transformation]]:
        cube = self._get_cube(v)
        kind = cube.kind
        assert isinstance(kind, ZXCube)
        scaled_pos = self._scale_position(cube.position)
        transformations: list[tuple[Basis, _Transformation]] = []
        if kind.normal_basis in surface_bases:
            normal_basis_edges = set()
            for edge in correlation_edges:
                this_node = edge.u if edge.u.id == v else edge.v
                if this_node.basis == kind.normal_basis:
                    normal_basis_edges.add(edge)
            assert len(normal_basis_edges) in {2, 4}, "Even parity constraint violated"
            if len(normal_basis_edges) == 2:
                e1, e2 = sorted(normal_basis_edges)
                if self._edge_direction(e1) == self._edge_direction(e2):
                    normal_direction = self._surface_normal_direction(e1)
                    translation = scaled_pos.shift_in_direction(normal_direction, 0.5)
                    transformations.append((kind.normal_basis, _Transformation(translation.as_tuple(), (1.0, 1.0, 1.0), _rotation_to_plane(normal_direction))))
                else:
                    transformations.extend(self._compute_turn_transformation(scaled_pos, v, (e1, e2)))
            else:
                e1, e2, e3, e4 = sorted(normal_basis_edges)
                transformations.extend(self._compute_turn_transformation(scaled_pos, v, (e1, e2)))
                transformations.extend(self._compute_turn_transformation(scaled_pos, v, (e3, e4)))
        if len(surface_bases) == 2 or kind.normal_basis not in surface_bases:
            translation = scaled_pos.shift_in_direction(kind.normal_direction, 0.5)
            transformations.append((kind.normal_basis.flipped(), _Transformation(translation.as_tuple(), (1.0, 1.0, 1.0), _rotation_to_plane(kind.normal_direction))))
        return transformations

    def _compute_turn_transformation(self, cube_pos: FloatPosition3D, v: int, turn_edges: tuple) -> list[tuple[Basis, _Transformation]]:
        cube_kind = self._get_cube(v).kind
        assert isinstance(cube_kind, ZXCube)
        transformations = []
        turn_dirs = [self._edge_direction(e) for e in turn_edges]
        for i, e in enumerate(turn_edges):
            d, other_d = turn_dirs[i], turn_dirs[1 - i]
            scale_direction = d if d != Direction3D.Z else other_d
            translation = cube_pos.shift_in_direction(other_d, 0.5)
            if v == e.u.id:
                translation = translation.shift_in_direction(d, 0.5)
            transformations.append((cube_kind.normal_basis, _Transformation(translation.as_tuple(), _get_scale(scale_direction, 0.5), _rotation_to_plane(other_d))))
        return transformations


def _rotation_to_plane(plane_normal_direction: Direction3D) -> Matrix3:
    if plane_normal_direction == Direction3D.Z:
        return _rotation_matrix(Direction3D.Z, 0.0)
    if plane_normal_direction == Direction3D.X:
        return _rotation_matrix(Direction3D.Y, 90.0)
    return _rotation_matrix(Direction3D.X, 90.0)


def _rotation_matrix(axis: Direction3D, angle: float = 90.0) -> Matrix3:
    """The right-handed rotation about an axis by an angle in degrees (negated about Y, as tqec
    has it)."""
    if axis == Direction3D.Y:
        angle = -angle
    c, s = math.cos(math.radians(angle)), math.sin(math.radians(angle))
    c, s = float(round(c, 12)) + 0.0, float(round(s, 12)) + 0.0
    if axis == Direction3D.X:
        return [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]]
    if axis == Direction3D.Y:
        return [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]]
    return [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]


def _get_scale(scale_direction: Direction3D, scale_factor: float) -> tuple[float, float, float]:
    scale = [1.0, 1.0, 1.0]
    scale[scale_direction.value] = scale_factor
    return (scale[0], scale[1], scale[2])

# Ported from tqec (tqec/tqec 0.2.0, src/tqec/utils/rotations.py; Apache License 2.0; see LICENSE in this directory),
# modified for stabilizer_qec: integer rotation matrices (no scipy).
"""Rotations of block kinds and positions by multiples of 90 degrees, as Collada files and
SketchUp encode them: a 3×3 matrix of the rotated axes."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Sequence

from ._errors import TQECError
from .position import Direction3D, FloatPosition3D, Position3D, round_or_fail

if TYPE_CHECKING:
    from .block_graph import BlockKind

Matrix = Sequence[Sequence[float]]


def calc_rotation_angles(rotation_matrix: Matrix) -> list[int]:
    """The rotation angle (degrees) of each row of the matrix from the original X/Y/Z axis."""
    out = []
    for i, row in enumerate(rotation_matrix):
        norm = math.sqrt(sum(float(x) ** 2 for x in row))
        cos_theta = float(row[i]) / norm if norm else 1.0
        out.append(round(math.degrees(math.acos(max(-1.0, min(1.0, cos_theta))))))
    return out


def get_axes_directions(rotation_matrix: Matrix) -> dict[str, int]:
    """Up/down multipliers for each row of a rotation matrix."""
    return {"XYZ"[i]: -1 if sum(float(x) for x in row) < 0 else 1 for i, row in enumerate(rotation_matrix)}


def rotate_block_kind_by_matrix(block_kind: "BlockKind", rotation_matrix: Matrix) -> "BlockKind":
    """The kind of a block after the rotation the matrix encodes (the kind's letters permuted as
    the axes are)."""
    from .block_graph import block_kind_from_str

    if str(block_kind) == "PORT":
        return block_kind
    original_name = str(block_kind)[:3] if len(str(block_kind)) > 1 else str(block_kind) + "-!"
    rotated_name = ""
    for row in rotation_matrix:
        entry = ""
        for j, element in enumerate(row):
            entry += abs(int(round(float(element)))) * original_name[j]
        rotated_name += entry
    axes_directions = get_axes_directions(rotation_matrix)
    if "!" in rotated_name and (not rotated_name.endswith("!") or axes_directions["Z"] < 0):
        raise TQECError(
            f"There is an invalid rotation for {rotated_name.replace('!', '').replace('-', '')} block.\nCultivation and Y blocks should only allow rotation around Z axis.",
        )
    elif "!" in rotated_name:
        rotated_name = str(block_kind)
    elif "H" in str(block_kind):
        rotated_name += str(block_kind)[-1]
    return block_kind_from_str(rotated_name)


def get_rotation_matrix(rotation_axis: Direction3D, counterclockwise: bool = True, num_90_degree_rotation: int = 1) -> list[list[int]]:
    """The integer matrix of a rotation about an axis by a multiple of 90 degrees (the right-hand
    rule for counterclockwise)."""
    k = (num_90_degree_rotation if counterclockwise else -num_90_degree_rotation) % 4
    c, s = [1, 0, -1, 0][k], [0, 1, 0, -1][k]
    a = rotation_axis.value
    b, d = (a + 1) % 3, (a + 2) % 3
    m = [[0, 0, 0] for _ in range(3)]
    m[a][a] = 1
    m[b][b], m[b][d] = c, -s
    m[d][b], m[d][d] = s, c
    return m


def rotate_position_by_matrix(position: Position3D, rotation_matrix: Matrix) -> Position3D:
    """A cube's position after the rotation, rotated about its centre (so a cube at (0, 0, 0)
    rotated by 90 degrees around the x-axis lands at (0, −1, 0))."""
    center = [i + 0.5 for i in position.as_tuple()]
    rotated = [sum(float(rotation_matrix[i][j]) * center[j] for j in range(3)) for i in range(3)]
    return Position3D(*[round_or_fail(r - 0.5) for r in rotated])


def rotate_on_import(rotation_matrix: Matrix, translation: Sequence[float], scale: Sequence[float], kind: "BlockKind") -> tuple[FloatPosition3D, "BlockKind"]:
    """A block read from a file with a rotation: its kind rotated and its translation adjusted;
    a rotation that is not a multiple of 90 degrees is refused."""
    angles = calc_rotation_angles(rotation_matrix)
    if any(int(angle) not in (0, 90, 180) for angle in angles) or sum(angles) < 180:
        raise TQECError(f"There is an invalid rotation for {kind} block at position {FloatPosition3D(*[float(t) for t in translation])}.")
    kind = rotate_block_kind_by_matrix(kind, rotation_matrix)
    shifted = [float(translation[i]) + sum(float(rotation_matrix[i][j]) * float(scale[j]) for j in range(3)) for i in range(3)]
    return FloatPosition3D(*shifted), kind


def adjust_hadamards_direction(kind: "BlockKind") -> "BlockKind":
    """A Hadamard pipe read in the negative direction of its axis: its kind's pair."""
    from .block_graph import block_kind_from_str

    equivalences = {"ZXOH": "XZOH", "XOZH": "ZOXH", "OXZH": "OZXH"}
    if str(kind) in equivalences:
        kind = block_kind_from_str(equivalences[str(kind)])
    return kind

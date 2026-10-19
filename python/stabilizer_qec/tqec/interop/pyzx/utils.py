# Ported from tqec (tqec/tqec 0.2.0, src/tqec/interop/pyzx/utils.py; Apache License 2.0, see
# LICENSE in the tqec directory), modified for stabilizer_qec.
"""tqec's helpers on PyZX graphs' spiders (PyZX needed, except for ``cube_kind_to_zx``'s
vertex types)."""

from __future__ import annotations

from fractions import Fraction

from ..._errors import TQECError
from ...cube import CubeKind, Port, ZXCube
from ...enums import Basis, Pauli


def is_zx_no_phase(g, v: int) -> bool:
    """Whether the vertex is a Z or X spider with phase 0."""
    from pyzx.utils import vertex_is_zx

    return vertex_is_zx(g.type(v)) and g.phase(v) == 0


def is_z_no_phase(g, v: int) -> bool:
    """Whether the vertex is a Z spider with phase 0."""
    from pyzx.utils import VertexType

    return g.type(v) is VertexType.Z and g.phase(v) == 0


def is_x_no_phase(g, v: int) -> bool:
    """Whether the vertex is an X spider with phase 0."""
    from pyzx.utils import VertexType

    return g.type(v) is VertexType.X and g.phase(v) == 0


def is_boundary(g, v: int) -> bool:
    """Whether the vertex is a boundary."""
    from pyzx.utils import VertexType

    return g.type(v) is VertexType.BOUNDARY


def is_s(g, v: int) -> bool:
    """Whether the vertex is an S spider (Z with phase 1/2)."""
    from pyzx.utils import VertexType

    return g.type(v) is VertexType.Z and g.phase(v) == Fraction(1, 2)


def is_hadamard(g, edge: tuple[int, int]) -> bool:
    """Whether the edge is a Hadamard edge."""
    from pyzx.utils import EdgeType

    return g.edge_type(edge) is EdgeType.HADAMARD


def cube_kind_to_zx(kind: CubeKind) -> tuple:
    """PyZX's vertex type and phase for a cube kind: a port a boundary, the Y half cube a Z
    spider of phase 1/2, a ZX cube a Z spider if it has one Z boundary, else an X spider."""
    from pyzx.utils import VertexType

    if isinstance(kind, ZXCube):
        if sum(basis == Basis.Z for basis in kind.as_tuple()) == 1:
            return VertexType.Z, 0
        return VertexType.X, 0
    if isinstance(kind, Port):
        return VertexType.BOUNDARY, 0
    return VertexType.Z, Fraction(1, 2)


def vertex_type_to_pauli(vertex_type, phase=0) -> Pauli:
    """The Pauli of a PyZX vertex type and phase (a TQECError for other spiders)."""
    from pyzx.utils import VertexType

    if vertex_type == VertexType.X and phase == 0:
        return Pauli.X
    if vertex_type == VertexType.Z and phase == 0:
        return Pauli.Z
    if vertex_type == VertexType.Z and phase == Fraction(1, 2):
        return Pauli.Y
    if vertex_type == VertexType.BOUNDARY:
        return Pauli.I
    raise TQECError(f"Cannot convert vertex type {vertex_type} and phase {phase} to Pauli operator.")


def zx_to_pauli(g, v: int) -> Pauli:
    """The Pauli of a vertex of a PyZX graph."""
    return vertex_type_to_pauli(g.type(v), g.phase(v))


def vertex_type_to_basis(vertex_type, phase=0) -> Basis:
    """The basis of a PyZX vertex type and phase (a TQECError unless a phase-free Z or X)."""
    try:
        return vertex_type_to_pauli(vertex_type, phase).to_basis()
    except TQECError:
        raise TQECError(f"Cannot convert vertex type {vertex_type} and phase {phase} to Basis.")


def zx_to_basis(g, v: int) -> Basis:
    """The basis of a vertex of a PyZX graph."""
    return vertex_type_to_basis(g.type(v), g.phase(v))

"""Block graphs of logical computations, as tqec draws them (`tqec/tqec <https://github.com/tqec/tqec>`_
0.2.0's ``computation`` package, its gallery and its Collada and JSON files, ported; Apache License
2.0, see ``LICENSE`` in this directory): cubes and pipes in a 3D spacetime diagram, ZX kinds,
validation, port filling, composition, rotation, correlation surfaces (the logical observables)
found on the graph's own ZX graph, and ``.dae`` files SketchUp and tqec exchange, read and
written with the standard library.

>>> from stabilizer_qec import tqec
>>> g = tqec.gallery.cnot(tqec.Basis.Z)
>>> [s.external_stabilizer_on_graph(g) for s in g.find_correlation_surfaces()]
['ZZII', 'IZZZ']
"""

from . import gallery
from ._errors import TQECError, TQECWarning
from .block_graph import BlockGraph, BlockKind, block_kind_from_str
from .collada import read_block_graph_from_dae_file, read_block_graph_from_json, write_block_graph_to_dae_file
from .compile import ALL_CONVENTIONS, FIXED_BOUNDARY_CONVENTION, FIXED_BULK_CONVENTION, CompiledBlockGraph, Convention, TopologicalComputationGraph, compile_block_graph
from .correlation import CorrelationSurface, ZXEdge, ZXGraph, ZXNode, find_correlation_surfaces, reduce_observables_to_minimal_generators
from .cube import Cube, CubeKind, Port, YHalfCube, ZXCube, cube_kind_from_string
from .enums import Basis, Pauli
from .noise_model import NoiseModel, NoiseRule
from .open_graph import FilledGraph, fill_ports_for_minimal_simulation
from .pipe import Pipe, PipeKind
from .position import Direction3D, FloatPosition3D, Position2D, Position3D, SignedDirection3D
from .scale import LinearFunction

__all__ = [
    "ALL_CONVENTIONS",
    "Basis",
    "BlockGraph",
    "BlockKind",
    "CompiledBlockGraph",
    "Convention",
    "CorrelationSurface",
    "compile_block_graph",
    "Cube",
    "CubeKind",
    "Direction3D",
    "FIXED_BOUNDARY_CONVENTION",
    "FIXED_BULK_CONVENTION",
    "FilledGraph",
    "FloatPosition3D",
    "LinearFunction",
    "NoiseModel",
    "NoiseRule",
    "Pauli",
    "Pipe",
    "PipeKind",
    "Port",
    "Position2D",
    "Position3D",
    "SignedDirection3D",
    "TQECError",
    "TQECWarning",
    "TopologicalComputationGraph",
    "YHalfCube",
    "ZXCube",
    "ZXEdge",
    "ZXGraph",
    "ZXNode",
    "block_kind_from_str",
    "cube_kind_from_string",
    "fill_ports_for_minimal_simulation",
    "find_correlation_surfaces",
    "gallery",
    "read_block_graph_from_dae_file",
    "read_block_graph_from_json",
    "reduce_observables_to_minimal_generators",
    "write_block_graph_to_dae_file",
]

"""Interoperability with PyZX's ZX graphs, as tqec's ``tqec.interop.pyzx``: positioned ZX
graphs (``PositionedZX``; ``BlockGraph.to_zx_graph()`` is one, and its ``g`` the PyZX graph),
block synthesis from them, their plots, and the spider helpers.

PyZX is optional: a PyZX graph is needed only to read one (``PositionedZX``,
``block_synthesis`` of a PyZX graph, ``find_correlation_surfaces`` of one,
``CorrelationSurface.to_pauli_web``) or to write one (``ZXGraph.g``). The synthesis and the
plots (matplotlib) work on this package's positioned ZX graph without it."""

from .plot import draw_positioned_zx_graph_on as draw_positioned_zx_graph_on
from .plot import plot_positioned_zx_graph as plot_positioned_zx_graph
from .plot import pyzx_draw_positioned_zx_3d as pyzx_draw_positioned_zx_3d
from .positioned import PositionedZX as PositionedZX
from .synthesis import SynthesisStrategy as SynthesisStrategy
from .synthesis import block_synthesis as block_synthesis
from .synthesis import positioned_block_synthesis as positioned_block_synthesis
from .utils import cube_kind_to_zx as cube_kind_to_zx
from .utils import is_boundary as is_boundary
from .utils import is_zx_no_phase as is_zx_no_phase
from .utils import zx_to_basis as zx_to_basis
from .utils import zx_to_pauli as zx_to_pauli

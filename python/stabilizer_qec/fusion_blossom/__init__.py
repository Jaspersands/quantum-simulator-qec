"""Fusion Blossom 0.2.13 (Yue Wu; MIT, see LICENSE here), whole, on this package's Rust port of
it: ``import stabilizer_qec.fusion_blossom as fb`` and fusion_blossom's code runs unchanged. The
serial and parallel minimum-weight perfect matching solvers (``SolverSerial``,
``SolverDualParallel``, ``SolverParallel``), syndromes with erasures and dynamic weights,
partitions, the example codes with their random errors, the visualizer and its website
(``fb.helper.open_visualizer``), with the same matchings as fusion_blossom's.

Weights are 64-bit (fusion_blossom's wheels use 32-bit ones), and where
fusion_blossom would call the Blossom V library, which its wheels leave out, this package's own
exact matcher runs. The helper and unchanged embedded Bottle live below this package rather than
registering top-level modules. Its server binds when run, uses the latest process-wide data folder,
and checks resolved data-path containment and malformed UTF-8; see ``helper`` for lifecycle,
MIME, logging and filesystem-race limits."""

from .. import _core

fusion_blossom = _core.fusion_blossom

globals().update({name: value for name, value in vars(fusion_blossom).items() if not name.startswith("_")})

from . import helper  # noqa: E402

helper.register(fusion_blossom)


class PyMut:
    """Changes a field of a field: fields come back as copies, so ``obj.a.b = 1`` leaves ``obj``
    alone, and this writes the copy back on leaving the block::

        with fb.PyMut(code, "vertices") as vertices:
            with fb.PyMut(vertices[0], "position") as position:
                position.i = 100
    """

    def __init__(self, object, attr_name):
        self.object = object
        self.attr_name = attr_name
        self.attr_object = None

    def __enter__(self):
        assert self.attr_object is None, "do not enter twice"
        self.attr_object = getattr(self.object, self.attr_name)
        return self.attr_object

    def __exit__(self, exc_type, exc_val, exc_tb):
        setattr(self.object, self.attr_name, self.attr_object)
        self.attr_object = None


PyMut.__module__ = __name__

__version__ = "0.2.13"  # the fusion_blossom release this follows

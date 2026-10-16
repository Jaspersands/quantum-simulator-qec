# Ported from Tsim (QuEraComputing/tsim, bloqade-tsim 0.1.5; Apache License 2.0; see LICENSE in this directory),
# modified for stabilizer_qec.
"""Tsim's encoders on this package's circuits: ``TransversalEncoder`` and its [[7, 1, 3]] Steane
(``SteaneEncoder``) and [[17, 1, 5]] colour-code (``ColorEncoder5``) instances, which encode a
state-preparation program into code blocks and a program into its transversal form (detectors and
observables onto the code's stabilizers and logicals); Tsim's shorthand (``T``, ``T_DAG``,
``TPP``, rotations, ``CCZ``, ...) to and from its tagged Stim (``shorthand_to_stim``,
``stim_to_shorthand``), and that Stim as this package's operations (``tsim_to_native``: Tsim's
``S[T]`` is ``I[T]`` here, its half-turn rotations radians)."""

from ._native import tsim_to_native
from .utils.encoder import ColorEncoder5, SteaneEncoder, TransversalEncoder
from .utils.program_text import shorthand_to_stim, stim_to_shorthand

__all__ = ["ColorEncoder5", "SteaneEncoder", "TransversalEncoder", "shorthand_to_stim", "stim_to_shorthand", "tsim_to_native"]

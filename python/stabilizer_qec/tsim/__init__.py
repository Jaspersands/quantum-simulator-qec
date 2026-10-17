# Ported from Tsim (QuEraComputing/tsim, bloqade-tsim 0.1.5; Apache License 2.0; see LICENSE in this directory),
# modified for stabilizer_qec.
"""Tsim's encoders on this package's circuits: ``TransversalEncoder`` and its [[7, 1, 3]] Steane
(``SteaneEncoder``) and [[17, 1, 5]] colour-code (``ColorEncoder5``) instances, which encode a
state-preparation program into code blocks and a program into its transversal form (detectors and
observables onto the code's stabilizers and logicals); Tsim's shorthand (``T``, ``T_DAG``,
``TPP``, rotations, ``CCZ``, ...) to and from its tagged Stim (``shorthand_to_stim``,
``stim_to_shorthand``), and that Stim as this package's operations (``tsim_to_native``: Tsim's
``S[T]`` is ``I[T]`` here, its half-turn rotations radians); ``Circuit(text)`` reads either
form; ``get_detector_error_model`` is Tsim's model with non-deterministic observables."""

from ._native import tsim_to_native
from .noise.dem import get_detector_error_model
from .utils.encoder import ColorEncoder5, SteaneEncoder, TransversalEncoder
from .utils.program_text import shorthand_to_stim, stim_to_shorthand


def Circuit(text: str = ""):  # noqa: N802 — Tsim's name
    """A ``stabilizer_qec.Circuit`` from Tsim's program text, shorthand or tagged: ``T 0``,
    ``T_DAG 1``, ``TPP X0*Z1``, ``R_Z(0.25) 0`` (half-turns), ``U3(0.3, 0.2, 0.1) 0``,
    ``CCZ 0 1 2``, ``S[T] 0``, ``I[R_X(theta=0.25*pi)] 0``; the rest Stim's. Its samplers run
    the non-Clifford gates (the stabilizer-rank sampler); ``tcount``, ``is_clifford``,
    ``to_matrix`` and ``inverse`` are Tsim's."""
    from .. import Circuit as _Circuit

    return _Circuit(tsim_to_native(shorthand_to_stim(text)))


__all__ = ["Circuit", "ColorEncoder5", "SteaneEncoder", "TransversalEncoder", "get_detector_error_model", "shorthand_to_stim", "stim_to_shorthand", "tsim_to_native"]

# Ported from Tsim (QuEraComputing/tsim, bloqade-tsim 0.1.5; Apache License 2.0; see LICENSE in this directory),
# modified for stabilizer_qec.
"""Tsim's interface on this package's circuits and samplers, with Tsim's conventions.

- ``Circuit``: Tsim's circuit, a subclass of ``stabilizer_qec.Circuit`` (``isinstance`` holds
  for both), read from Tsim's program text (``T 0``, ``T_DAG 1``, ``TPP X0*Z1``, ``R_Z(0.25) 0``
  in half-turns, ``R_ZZ(0.5) 0 1``, ``R_PAULI(0.3) X0*Y1``, ``U3(0.3, 0.2, 0.1) 0``,
  ``CCZ 0 1 2``, Tsim's tagged ``S[T] 0`` and ``I[R_X(theta=0.25*pi)] 0``, and this package's
  ``I[T] 0`` and radians), ``from_stim_program`` and ``from_file`` (shorthand too); ``append``
  takes Tsim's gates with their arguments in half-turns and ``append_from_stim_program_text``
  its text; ``str()`` and ``repr()`` are Tsim's shorthand; ``+`` and ``+=`` take a
  ``stim.Circuit`` or ``stabilizer_qec.Circuit`` (Tsim's tagged Stim read as its gates); what
  Tsim's methods return is a ``tsim.Circuit`` (``copy``, slices, ``*``, ``flattened``,
  ``without_noise``, ``inverse``, ...), and ``stim_circuit`` a plain ``stabilizer_qec.Circuit``.
  ``tcount``, ``is_clifford`` and ``inverse`` are Tsim's; ``to_matrix`` is Tsim's map on the
  qubits the circuit names, a state column when every qubit is reset first, and, of a circuit
  with measurements, resets after a qubit's first operation, feedback or noise on a qubit
  before its reset, Tsim's ZX tensor with every measurement outcome and error bit read as 0;
  ``diagram`` defaults to the timeline SVG, its tagged gates labelled, with Tsim's ``height``,
  ``width`` and ``zoomable`` for the notebook view, and draws the ZX graph for ``"pyzx"``,
  ``"pyzx-dets"`` and ``"pyzx-meas"``.
- ``Circuit.get_graph`` and ``Circuit.get_sampling_graph`` (``stabilizer_qec.tsim.zx``): Tsim's
  ZX graphs, equal to Tsim's vertex for vertex (numbers, types, phases as Fractions, phase
  variables, positions, edges, scalar). They are pyzx_param's ``GraphS`` when pyzx_param is
  installed, else this package's ``ZXGraph`` (``backend="own"`` or ``"pyzx_param"`` chooses;
  ``ZXGraph.to_pyzx()`` gives a pyzx graph). ``zx.tensorfy`` contracts a graph in numpy.
- ``CompiledDetectorSampler`` (``Circuit.compile_detector_sampler``): Tsim's convention, each
  detector's and observable's *value*, the parity of its measurements (``X 0``, ``M 0``,
  ``DETECTOR rec[-1]`` reads True), not Stim's flip from a reference run;
  ``use_detector_reference_sample`` / ``use_observable_reference_sample`` give the flips (one
  noiseless shot each call is the reference, as Tsim's; on a Clifford circuit Stim's events
  exactly). ``CompiledMeasurementSampler`` (``Circuit.compile_sampler``): the records.
  ``batch_size`` and ``postselection_mask`` are checked as Tsim checks them and ``strategy`` is
  accepted (Tsim does not check it either); none changes a sample (they are performance knobs
  of Tsim's JAX programs; every row is a full sample). The shots are this package's samplers'
  (the frame sampler on a Clifford circuit, the stabilizer-rank sampler on a non-Clifford one),
  so they match Tsim's in distribution, not bit for bit. ``stabilizer_qec.Circuit``'s own
  samplers keep Stim's conventions.
- ``sampler.CompiledStateProbs``: P(state | f) for each of a batch of noise draws f, of a
  measurement record or (``sample_detectors``) of the detectors' and observables' values.
- ``Circuit.detector_error_model`` and ``get_detector_error_model``: Tsim's model, which allows
  non-deterministic observables (``error(0.5) D0 L0``); with it ``decompose_errors`` raises.
- Tsim's encoders, ``TransversalEncoder`` and its [[7, 1, 3]] Steane (``SteaneEncoder``) and
  [[17, 1, 5]] colour-code (``ColorEncoder5``) instances, which encode a state-preparation
  program into code blocks and a program into its transversal form (detectors and observables
  onto the code's stabilizers and logicals); their ``circuit`` is a ``tsim.Circuit``.
- ``shorthand_to_stim`` and ``stim_to_shorthand``: Tsim's shorthand to and from its tagged Stim
  (``stim_to_shorthand`` reads this package's tagged forms too), and ``tsim_to_native``: that
  Stim as this package's operations (Tsim's ``S[T]`` is ``I[T]`` here, its half-turn rotations
  radians of exp(−iθP)).

Where it differs from Tsim:

- Shots and noise draws are this package's, equal to Tsim's in distribution (Tsim's are
  JAX's); deterministic values, models, matrices and noiseless probabilities are Tsim's.
- ``R_PAULI(0.25)`` and ``R_XX(0.25)`` (any ±π/8 product rotation) print as ``TPP`` or
  ``TPP_DAG``, the same gate; an angle appended as an int prints as a float (``R_Z(1.0)``,
  Tsim ``R_Z(1)``); a tag on an appended ``T``, ``T_DAG``, ``TPP`` or ``TPP_DAG`` is not kept.
- Two product rotations with the same angle in a row stay two lines (Tsim's Stim text fuses
  them into one, which its shorthand misprints).
- ``tcount`` is the graph's, as Tsim's: a coherent error (``I_ERROR[R_Z(...)]``) is not a gate
  (``stabilizer_qec.Circuit.tcount`` counts its phase).
- A product rotation by ±0.25 half-turns (``R_PAULI(0.25)``, ``R_XX(0.25)``) is the same gate as
  ``TPP`` here, and its ZX graph is ``TPP``'s: Tsim's graph for ``R_PAULI(0.25)`` carries the
  rotation's scalar phase of -1/8 that ``TPP``'s does not, so the graph's scalar (and, through
  the ZX path, ``to_matrix``) differs from Tsim's for ``R_PAULI(0.25)`` by that global phase and
  equals Tsim's for ``TPP``. An angle is read back from the radians this package stores as the
  shortest decimal number of half-turns that gives them, so ``R_Z(0.3)`` is the phase 3/10 of
  Tsim's graph.
- ``diagram("pyzx")`` needs pyzx or pyzx_param to draw; ``"pyzx-dets"`` and ``"pyzx-meas"`` need
  pyzx_param (their graphs are reduced by its ``full_reduce``). ``to_matrix`` of a circuit
  with measurements is at most 2^22 entries, its spiders at most 24 legs, a chain of correlated
  errors at most 24 branches.
- ``circuit[i]`` is this package's instruction (``I[T] 0``), not a ``stim.CircuitInstruction``
  in Tsim's tagged form (``S[T] 0``).
"""

from . import sampler, zx
from ._native import tsim_to_native
from .circuit import Circuit
from .noise.dem import get_detector_error_model
from .sampler import CompiledDetectorSampler, CompiledMeasurementSampler, CompiledStateProbs
from .utils.encoder import ColorEncoder5, SteaneEncoder, TransversalEncoder
from .utils.program_text import shorthand_to_stim, stim_to_shorthand

__all__ = [
    "Circuit",
    "ColorEncoder5",
    "CompiledDetectorSampler",
    "CompiledMeasurementSampler",
    "CompiledStateProbs",
    "SteaneEncoder",
    "TransversalEncoder",
    "get_detector_error_model",
    "sampler",
    "shorthand_to_stim",
    "stim_to_shorthand",
    "tsim_to_native",
    "zx",
]

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
  with measurements, detectors or observables (one that XORs no measurement is a factor of 2),
  resets after a qubit's first operation, feedback, noise on a qubit before its reset or a
  product rotation naming a qubit twice, Tsim's ZX tensor with every measurement outcome and
  error bit read as 0 (a circuit with Pauli products or sweep bits has its ZX graph built first,
  so its errors are Tsim's);
  ``diagram`` defaults to the timeline SVG, its tagged gates labelled, with Tsim's ``height``,
  ``width`` and ``zoomable`` for the notebook view, and draws the ZX graph for ``"pyzx"``,
  ``"pyzx-dets"`` and ``"pyzx-meas"``.
- ``Circuit.get_graph`` and ``Circuit.get_sampling_graph`` (``stabilizer_qec.tsim.zx``): Tsim's
  ZX graphs, equal to Tsim's vertex for vertex (numbers, types, phases as Fractions, phase
  variables, positions, edges, scalar). They are pyzx_param's ``GraphS`` when pyzx_param imports,
  else this package's ``ZXGraph`` (``backend="own"`` or ``"pyzx_param"`` chooses;
  ``ZXGraph.to_pyzx()`` gives a pyzx graph). ``zx.tensorfy`` contracts a graph in numpy. Choosing
  the default backend (and ``zx.pyzx_param_available()``) imports pyzx_param when it is installed,
  and importing pyzx_param calls ``numpy.set_printoptions(suppress=True)`` for the whole process, as
  importing Tsim does; ``backend="own"`` imports nothing. A record entry from before the circuit's
  first measurement (``M 0`` then ``DETECTOR rec[-2]``) raises ``zx.parse.MeasurementRecordError``,
  an ``IndexError`` as Tsim's is and a ``ValueError`` that names the instruction.
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
- An angle appended as an int prints as a float (``R_Z(1.0)``, Tsim ``R_Z(1)``); a tag on an
  appended ``T``, ``T_DAG``, ``TPP`` or ``TPP_DAG`` is not kept, nor the spacing inside a tagged
  rotation's (``SPP[R_PAULI( theta=0.3*pi )]`` is quoted ``SPP[R_PAULI(theta=0.3*pi)]``); the angle
  literals of a ``stim.Circuit`` are read as floats, as Tsim reads its program text (Tsim's
  ``from_stim_program`` keeps the stim circuit's ``0.30000000000000001*pi``, the phase
  30000000000000001/10^17, where this is 3/10). Text that does not parse raises this package's
  ``ValueError`` naming the line (``SPP: bad Pauli product '2'`` for ``R_ZZ(0.25) 0 1 2 3``, Stim's
  "Gate SPP only takes Pauli targets").
- A product rotation is held one product a line, ``I[R_PAULI(theta=t, pauli=XY)] 0 1`` in radians,
  and the rest of Tsim's instruction is kept in how that is written: a factor's letter is lower
  case where it has a ``!`` (``SPP[T] !X0*Y1`` is ``pauli=xY``), the ``SPP_DAG`` of a rotation has
  its arguments in the other order (``SPP_DAG[R_PAULI(theta=0.3*pi)] X0`` is
  ``I[R_PAULI(pauli=X, theta=-0.47123889803846897)] 0``), and ``R_PAULI(±0.25)`` is ±π/8 with
  further digits (``0.392699081698724139``; ``TPP``'s is ``0.39269908169872414``). That is the text
  ``circuit[i]``, ``stim_circuit``, ``to_file`` and this package's own errors show. Read back, the
  instruction is Tsim's: it prints as Tsim's (``SPP_DAG[R_PAULI(theta=0.3*pi)] X0``,
  ``TPP !X0``), and it fuses with its neighbours, counts towards ``R_PAULI``'s 64 qubits and is
  quoted in errors as in Tsim. Text in this form is read as the core reads it: ``_`` and ``I`` are
  identity letters (``I[R_PAULI(theta=0.1, pauli=X_Z)] 0 1 2`` is Tsim's ``R_PAULI(...) X0*Z2``),
  spaces and empty arguments in a tag are skipped (but a pair rotation's empty first argument,
  its ``SPP_DAG``, below), ``I[T()]`` is ``I[T]``, ``R_PAULI`` on ``II`` is
  the product rotation it is on ``I`` (``II[R_PAULI(theta=0.1, pauli=XZ)] 0 1`` is Tsim's
  ``R_PAULI(...) X0*Z1``, its inverse that of the ``I`` form on ``II``), and on ``I_ERROR`` or
  ``II_ERROR`` it is a coherent error, no gate. A product rotation whose letters are all identities
  (``pauli=__``) is no operation and is left out of the print, as the core, the graph and the model
  read it (Tsim would read ``II[R_PAULI(...)] 1 2`` as a plain ``II`` naming qubits 1 and 2). So the
  print, read back by Tsim or by ``tsim.Circuit``, has that instruction fewer, and its ``num_qubits``
  is that of the qubits it names, without one only the product names (``H 0`` then
  ``II[R_PAULI(theta=0.1, pauli=__)] 1 2`` has ``len`` 2 and ``num_qubits`` 3, its print ``H 0`` 1
  and 1), as it is without one only an identity letter names (``I[R_PAULI(theta=0.1, pauli=X_)] 0 1``
  prints ``R_PAULI(...) X0``, ``num_qubits`` 1, not 2): the print and ``repr`` of such a circuit do
  not give it back under ``==``, though what they give has its graph, ``tcount``, matrix, model and
  inverse. A pair
  rotation ``II[R_ZZ(theta=t)] 0 1 2 3`` (``R_XX``, ``R_YY``) is Tsim's ``R_ZZ(a) 0 1`` and
  ``R_ZZ(a) 2 3``; its ``SPP_DAG`` is written with an empty argument first,
  ``II[R_ZZ(, theta=-t)] 3 2 1 0``, as a product rotation's has its arguments in the other order,
  and prints as Tsim's ``SPP_DAG[R_PAULI(theta=a*pi)] Z3*Z2 Z1*Z0``: so the inverse of a pair
  rotation, or of a product rotation on ``II``, is one instruction, as Tsim's, and its inverse the
  circuit again.
- Two product rotations with the same angle in a row stay two lines (Tsim's Stim text fuses
  them into one, which its shorthand misprints), as do a pair rotation's ``TPP``s on several pairs
  (``TPP_DAG X0*X1`` then ``TPP_DAG X2*X3``, the inverse ``TPP X3*X2`` then ``TPP X1*X0``, where
  Tsim prints ``TPP X3*X2 X1*X0``), and a product naming one qubit twice prints as
  ``R_PAULI(0.3) X0*X0`` (Tsim's shorthand ``R_XX(0.3) 0 0`` does not read back). They are one
  instruction to the ZX graph, as in Tsim, so ``R_PAULI``'s limit of 64 qubits is counted over
  them (33 ``R_ZZ(0.3)`` lines in a row are Tsim's error, "got 66").
- ``tcount`` is the graph's, as Tsim's: a coherent error (``I_ERROR[R_Z(...)]``) is not a gate
  (``stabilizer_qec.Circuit.tcount`` counts its phase).
- An angle is kept as it was written, as Tsim keeps it (the radians this package stores carry
  the digits that give the literal back), so ``R_Z(0.3)`` is the phase 3/10 of Tsim's graph and
  ``R_PAULI(0.25)`` is Tsim's ``R_PAULI``, not ``TPP``: its graph has the scalar phase -1/8 (+1/8
  for -0.25) that ``TPP``'s has not, and its errors are ``R_PAULI``'s. An angle whose radians are
  past a float's range (from about 5.7e307 half-turns) is refused; Tsim takes it.
- ``diagram("pyzx")`` needs pyzx or pyzx_param to draw; ``"pyzx-dets"`` and ``"pyzx-meas"`` need
  pyzx_param (their graphs are reduced by its ``full_reduce``). ``to_matrix`` is at most 2^22
  entries and, through the ZX tensor (a circuit with measurements, detectors or observables, ...),
  its spiders at most 24 legs and the tensors of its contraction at most 2^26 entries (1 GiB of
  complex128, which numpy's ``tensordot`` copies). A
  chain of more than 24 correlated errors builds, its graph and matrix Tsim's, but its 2^k
  probabilities (Tsim's ``channel_probs`` entry, an array) are kept as the chain
  (``zx.instructions.CorrelatedChainProbs``): an entry each, and ``ValueError`` read as an array.
- ``circuit[i]`` is this package's instruction (``I[T] 0``), not a ``stim.CircuitInstruction``
  in Tsim's tagged form (``S[T] 0``).
- ``to_matrix`` of a circuit with an instruction Tsim's graph builders cannot read (a tag the core
  leaves inert, ``I[foo(x)] 0``, which ``get_graph`` refuses as Tsim does) is the unitary engine's
  matrix, where Tsim's ``to_matrix`` raises that error (and a refusal of a Pauli product after
  that instruction is not raised either), unless a product rotation names a qubit twice
  (``TPP X0*Y0*X0``, whose sign the engine does not keep): then it raises that error, as Tsim's.
  Where a product leaves a qubit out of Tsim's graph (an identity letter's, ``pauli=X_``, or one
  ``SPP X0*X0`` cancels on), that matrix is Tsim's of the circuit without the instruction up to
  a global phase, as the engine's: its qubits are those Tsim's graph of that circuit has lanes
  for, as every ``to_matrix`` is.
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

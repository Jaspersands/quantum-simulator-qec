"""Loom's Eka model, interpreter, validator and exporters (Entropica Labs' `el-loom
<https://github.com/entropicalabs/el-loom>`_ 0.4.0, ported; Apache License 2.0, see ``LICENSE``
in this directory): blocks of stabilizers on a lattice, lattice-surgery operations on them
(reset, syndrome rounds, logical measurements and Paulis, grow, shrink, merge, split and their
conditionals), an interpreter that turns an experiment into a circuit with syndromes,
detectors and logical observables (gate for gate Loom's own Stim output, except where Loom's
order follows a Python set of strings), a validator that checks a circuit against its input
and output codes on this package's tableau simulator, and exporters to Stim (this package's
``Circuit``, with the noise of Loom's circuit error models), OpenQASM 3, PennyLane, CUDA-Q,
Mimiq and Guppy.

>>> from stabilizer_qec import loom
>>> lattice = loom.Lattice.square_2d((15, 15))
>>> a = loom.RotatedSurfaceCode.create(3, 3, lattice, unique_label="a")
>>> b = loom.RotatedSurfaceCode.create(3, 3, lattice, unique_label="b", position=(4, 0))
>>> eka = loom.Eka(lattice, blocks=[a, b], operations=[
...     loom.ResetAllDataQubits("a"), loom.ResetAllDataQubits("b"),
...     loom.MeasureBlockSyndromes("a"), loom.MeasureBlockSyndromes("b"),
...     loom.Merge(["a", "b"], "m"), loom.MeasureBlockSyndromes("m", n_cycles=2),
...     loom.MeasureLogicalZ("m")])
>>> circuit = loom.eka_to_circuit(loom.interpret_eka(eka))
>>> circuit.num_detectors, circuit.num_observables
(76, 1)

What is not ported: Loom's auxiliary CNOT, transversal Hadamard, move, rotate, Y-wall phase
gate and state injection applicators, its base and logical operations, if-else circuits and
the executor's ``Executor`` factory, run-outcome parsers and legacy cliffordsim export.
"""

from .circuit import Channel, ChannelType, Circuit, SyndromeCircuit, default_circuit
from .eka import (
    CNOT,
    AuxCNOT,
    BaseOperation,
    Block,
    CodeOperation,
    ConditionalLogicalX,
    ConditionalLogicalY,
    ConditionalLogicalZ,
    DiagonalDirection,
    Direction,
    Eka,
    FourBodySchedule,
    Grow,
    Hadamard,
    Lattice,
    LatticeType,
    LogicalMeasurement,
    LogicalOperation,
    LogicalPhaseViaYwall,
    LogicalX,
    LogicalY,
    LogicalZ,
    MeasureBlockSyndromes,
    MeasureLogicalX,
    MeasureLogicalY,
    MeasureLogicalZ,
    MeasureObservable,
    MeasureStabilizerSyndrome,
    Merge,
    MoveBlock,
    Operation,
    Orientation,
    PauliOperator,
    Phase,
    PhaseInverse,
    Reset,
    ResetAllAncillaQubits,
    ResetAllDataQubits,
    ResourceState,
    RotateBlock,
    Shrink,
    SingleQubitPauliEigenstate,
    Split,
    Stabilizer,
    StateInjection,
    T,
    TransversalHadamard,
    X,
    Y,
    Z,
)
from .error_model import ApplicationMode, AsymmetricDepolarizeCEM, CircuitErrorModel, ErrorProbProtocol, ErrorType, HomogeneousTimeDependentCEM, HomogeneousTimeIndependentCEM, noise_annotated_stim_circuit
from .executor import EkaToCudaqConverter, EkaToGuppylangConverter, EkaToMimiqConverter, EkaToPennylaneConverter, EkaToQasmConverter, EkaToStimConverter, StimProgram, eka_to_circuit
from .interpreter import Detector, InterpretationStep, LogicalObservable, Syndrome, SyndromeMissingError, cleanup_final_step, interpret_eka
from .rotated_surface_code import RotatedSurfaceCode
from .validator import (
    AllChecks,
    CodeStabilizerCheck,
    DebugData,
    LogicalOperatorCheck,
    LogicalState,
    StabilizerMeasurementCheck,
    is_circuit_valid,
    is_logical_operation_circuit_valid,
    is_syndrome_extraction_circuit_valid,
    logical_state_transformations_to_check,
    logical_states_to_check,
)

__all__ = [name for name in dir() if not name.startswith("_")]

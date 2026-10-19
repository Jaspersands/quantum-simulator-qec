"""Loom's Eka model, interpreter, validator and exporters (Entropica Labs' `el-loom
<https://github.com/entropicalabs/el-loom>`_ 0.4.0, ported; Apache License 2.0, see ``LICENSE``
in this directory): blocks of stabilizers on a lattice, lattice-surgery operations on them
(reset, syndrome rounds, logical measurements and Paulis, grow, shrink, merge, split and their
conditionals; on the rotated surface code also the auxiliary CNOT, transversal Hadamard, T and S
state injection, move, rotate and the Y-wall phase gate), Loom's other code factories (the
repetition chain, with its own grow, shrink, merge and split; the Shor, Steane and five-qubit
codes), an interpreter that turns an experiment into a circuit with syndromes,
detectors and logical observables (gate for gate Loom's own Stim output, except where Loom's
order follows a Python set: there, under the same ``PYTHONHASHSEED``), a validator that checks a circuit against its input
and output codes on this package's tableau simulator, and exporters to Stim (this package's
``Circuit``, with the noise of Loom's circuit error models), OpenQASM 3, PennyLane, CUDA-Q,
Mimiq and Guppy, with their readings of a run's outcome and Loom's ``Executor``. Loom's module
paths work under this package: ``loom.eka.operations`` is ``stabilizer_qec.loom.eka.operations``,
``loom_repetition_code.code_factory`` is ``stabilizer_qec.loom.repetition_code.code_factory``.

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

What is not ported: Loom's base and logical operations (they are defined, but the interpreter
refuses them), if-else circuits, the Visualizer, cliffordsim and the executor's legacy cliffordsim
export, the Tanner graphs, parity-check matrices and the syndrome-circuit algorithms built on
them, the Pauli-array algebra of ``loom.eka.utilities``, ``dumps``/``loads`` (``asdict`` and
``fromdict`` instead), the converters' operation signatures, and the plugin machinery (entry
points and applicator classes; one dispatch table here).
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
from .executor import Converter, EkaToCudaqConverter, EkaToGuppylangConverter, EkaToMimiqConverter, EkaToPennylaneConverter, EkaToQasmConverter, EkaToStimConverter, Executor, StimProgram, TargetLanguage, eka_to_circuit
from .interpreter import Detector, InterpretationStep, LogicalObservable, Syndrome, SyndromeMissingError, cleanup_final_step, interpret_eka
from .five_qubit_perfect_code import FiveQubitPerfectCode
from .repetition_code import RepetitionCode
from .rotated_surface_code import RotatedSurfaceCode
from .shor_code import ShorCode
from .steane_code import SteaneCode
from .validator import (
    AllChecks,
    CodeStabilizerCheck,
    CodeStabilizerCheckOutput,
    DebugData,
    LogicalOperatorCheck,
    LogicalOperatorCheckOutput,
    LogicalState,
    StabilizerMeasurementCheck,
    StabilizerMeasurementCheckOutput,
    is_circuit_valid,
    is_logical_operation_circuit_valid,
    is_syndrome_extraction_circuit_valid,
    logical_state_transformations_to_check,
    logical_states_to_check,
)

__all__ = [name for name in dir() if not name.startswith("_")]


# Loom's documented module paths, under this package: ``loom.eka.operations`` is
# ``stabilizer_qec.loom.eka.operations``, ``loom_rotated_surface_code.code_factory`` is
# ``stabilizer_qec.loom.rotated_surface_code.code_factory`` (and so for the other code plugins),
# each holding the names of Loom's module that are ported (``loom.eka.circuit`` and
# ``<code>.code_factory.<code>`` are this package's own modules). They are registered in
# sys.modules, which the import system consults before it looks for a file.
def _register(name: str, module) -> None:
    import sys

    sys.modules[f"{__name__}.{name}"] = module
    parent, _, child = name.rpartition(".")
    setattr(sys.modules[f"{__name__}.{parent}"], child, module)


def _namespace(name: str, loom_name: str, names: dict) -> None:
    import types

    module = types.ModuleType(f"{__name__}.{name}", f"The ported names of Loom's ``{loom_name}``.")
    module.__dict__.update(names)
    module.__all__ = sorted(names)
    _register(name, module)


def _register_loom_paths() -> None:
    import sys

    from . import circuit as circuit_module, eka as eka_module, interpreter as interpreter_module

    g = globals()
    # loom.eka also has the circuit IR and the logical states.
    for name in ("Channel", "ChannelType", "Circuit", "SyndromeCircuit", "LogicalState"):
        setattr(eka_module, name, g[name])
    _register("eka.circuit", circuit_module)
    code_operations = ["CodeOperation", "ConditionalLogicalX", "ConditionalLogicalY", "ConditionalLogicalZ", "Direction", "Grow", "LogicalMeasurement", "LogicalX", "LogicalY", "LogicalZ", "MeasureBlockSyndromes", "MeasureLogicalX", "MeasureLogicalY", "MeasureLogicalZ", "Merge", "Operation", "Orientation", "ResetAllAncillaQubits", "ResetAllDataQubits", "ResourceState", "Shrink", "SingleQubitPauliEigenstate", "Split", "StateInjection"]
    operations = ["BaseOperation", "CNOT", "CodeOperation", "ConditionalLogicalX", "ConditionalLogicalY", "ConditionalLogicalZ", "Grow", "Hadamard", "LogicalMeasurement", "LogicalX", "LogicalY", "LogicalZ", "MeasureBlockSyndromes", "MeasureLogicalX", "MeasureLogicalY", "MeasureLogicalZ", "MeasureObservable", "MeasureStabilizerSyndrome", "Merge", "Operation", "Phase", "PhaseInverse", "Reset", "ResetAllAncillaQubits", "ResetAllDataQubits", "Shrink", "Split", "StateInjection", "T", "X", "Y", "Z"]
    _namespace("eka.operations", "loom.eka.operations", {name: g[name] for name in operations})
    _namespace("eka.operations.code_operation", "loom.eka.operations.code_operation", {name: g[name] for name in code_operations})
    _namespace("eka.utilities", "loom.eka.utilities", {"DiagonalDirection": DiagonalDirection, "Direction": Direction, "Orientation": Orientation, "ResourceState": ResourceState, "SingleQubitPauliEigenstate": SingleQubitPauliEigenstate, "SyndromeMissingError": SyndromeMissingError, "apply_to_nested": circuit_module.apply_to_nested})
    _namespace("interpreter.interpreter", "loom.interpreter.interpreter", {"BaseOperation": BaseOperation, "Circuit": Circuit, "CodeOperation": CodeOperation, "Eka": Eka, "InterpretationStep": InterpretationStep, "Operation": Operation, "cleanup_final_step": cleanup_final_step, "interpret_eka": interpret_eka, "interpret_operation": interpreter_module.interpret_operation})
    # The code plugins: loom_<code>.code_factory (and its <code> module), and the rotated
    # surface code's utilities and operations.
    for module_name, class_name in (("rotated_surface_code", "RotatedSurfaceCode"), ("repetition_code", "RepetitionCode"), ("shor_code", "ShorCode"), ("steane_code", "SteaneCode"), ("five_qubit_perfect_code", "FiveQubitPerfectCode")):
        _namespace(f"{module_name}.code_factory", f"loom_{module_name}.code_factory", {class_name: g[class_name]})
        _register(f"{module_name}.code_factory.{module_name}", sys.modules[f"{__name__}.{module_name}"])
    _namespace("rotated_surface_code.utilities", "loom_rotated_surface_code.utilities", {"FourBodySchedule": FourBodySchedule})
    _namespace("rotated_surface_code.operations", "loom_rotated_surface_code.operations", {name: g[name] for name in ("AuxCNOT", "LogicalPhaseViaYwall", "MoveBlock", "RotateBlock", "TransversalHadamard")})


_register_loom_paths()

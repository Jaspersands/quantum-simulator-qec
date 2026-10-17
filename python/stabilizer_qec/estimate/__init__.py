"""Resource-estimation models of other tools, ported and checked against them: Qualtran's
surface-code cost models (`quantumlib/Qualtran <https://github.com/quantumlib/Qualtran>`_ 0.7.0,
``qualtran.surface_code``; Apache 2.0, see ``LICENSE`` in this directory) and Azure's Resource
Estimator (`microsoft/qdk <https://github.com/microsoft/qdk>`_'s ``resource_estimator``: the
physical estimate from logical counts with its exhaustive T-factory search; MIT, see
``LICENSE-qdk``).

>>> from stabilizer_qec.estimate import GateCounts, get_ccz2t_costs_from_error_budget
>>> cost = get_ccz2t_costs_from_error_budget(n_logical_gates=GateCounts(t=10**8, toffoli=10**8), n_algo_qubits=100)
>>> round(cost.failure_prob, 4), cost.footprint, round(cost.duration_hr, 2)
(0.0084, 400204, 7.53)
>>> from stabilizer_qec.estimate import LogicalCounts, azure_estimate
>>> r = azure_estimate(LogicalCounts(num_qubits=100, t_count=100_000, ccz_count=10_000, measurement_count=10_000), "qubit_gate_ns_e4")
>>> r["physicalCounts"]["physicalQubits"], r["physicalCounts"]["runtime"], r["logicalQubit"]["codeDistance"], r["tfactory"]["codeDistancePerRound"]
(58820, 504000000, 9, [7])
"""

from .azure import (
    QUBIT_NAMES,
    AzureEstimator,
    Constraints,
    DistillationUnit,
    DistillationUnitTemplate,
    ErrorBudget,
    EstimationResult,
    FactoryBuildError,
    GateBasedQubit,
    LogicalCounts,
    LogicalPatch,
    MajoranaQubit,
    QECScheme as AzureQECScheme,
    TFactory,
    azure_estimate,
    azure_estimator,
    find_nondominated_tfactories,
    physical_qubit,
    qec_scheme,
)
from .qualtran import (
    AlgorithmSummary,
    BeverlandEtAlRotationCost,
    CCZ2TFactory,
    CompactDataBlock,
    ConstantWithOverheadRotationCost,
    DataBlock,
    FastDataBlock,
    FifteenToOne,
    GateCounts,
    IntermediateDataBlock,
    LogicalErrorModel,
    MagicStateFactory,
    MultiFactory,
    PhysicalCostModel,
    PhysicalCostsSummary,
    PhysicalParameters,
    QECScheme,
    RotationCostModel,
    RotationLogarithmicModel,
    SevenDigitsOfPrecisionConstantCost,
    SimpleDataBlock,
    code_distance,
    get_ccz2t_costs,
    get_ccz2t_costs_from_error_budget,
    get_ccz2t_costs_from_grid_search,
    iter_ccz2t_factories,
    iter_simple_data_blocks,
    minimum_time_steps,
    n_discrete_logical_gates,
    t_states,
)

__all__ = [name for name in dir() if not name.startswith("_")]

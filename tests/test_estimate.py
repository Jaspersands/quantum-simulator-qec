"""The estimator ports (`stabilizer_qec.estimate`): Qualtran's surface-code models against the
numbers Qualtran's own tests pin (and against Qualtran where it is installed), and Azure's
Resource Estimator model against the QDK's results recorded in tests/estimate_port/."""

from __future__ import annotations

import json
import math
import pathlib
import time

import numpy as np
import pytest

from stabilizer_qec import estimate as est
from stabilizer_qec.estimate import azure as az
from stabilizer_qec.estimate import (
    AlgorithmSummary,
    AzureEstimator,
    Constraints,
    DistillationUnitTemplate,
    ErrorBudget,
    GateBasedQubit,
    LogicalCounts,
    MajoranaQubit,
    azure_estimate,
    azure_estimator,
    find_nondominated_tfactories,
    physical_qubit,
    qec_scheme,
)
from stabilizer_qec.estimate import (
    BeverlandEtAlRotationCost,
    CCZ2TFactory,
    FastDataBlock,
    FifteenToOne,
    GateCounts,
    LogicalErrorModel,
    MultiFactory,
    QECScheme,
    RotationLogarithmicModel,
    ConstantWithOverheadRotationCost,
    get_ccz2t_costs_from_error_budget,
    get_ccz2t_costs_from_grid_search,
    iter_ccz2t_factories,
)

try:
    import qualtran.surface_code as their_sc
    from qualtran.resource_counting import GateCounts as TheirGateCounts
except ImportError:  # pragma: no cover
    their_sc = None

needs_qualtran = pytest.mark.skipif(their_sc is None, reason="qualtran is not installed")


# ---------------------------------------------------------------------------------------------
# Qualtran's own pinned values
# ---------------------------------------------------------------------------------------------


def test_qec_scheme():
    assert QECScheme.make_beverland_et_al().logical_error_rate(3, 1e-3) == pytest.approx(3e-4)
    assert QECScheme.make_beverland_et_al().physical_qubits(11) == 242
    scheme = QECScheme.make_gidney_fowler()
    for budget in np.logspace(-1, -18):
        d = scheme.code_distance_from_budget(physical_error=1e-3, budget=budget)
        assert d % 2 == 1 and scheme.logical_error_rate(d, 1e-3) <= budget
        if d > 3:
            assert scheme.logical_error_rate(d - 2, 1e-3) > budget
    assert LogicalErrorModel(qec_scheme=QECScheme.make_beverland_et_al(), physical_error=1e-3)(3) == pytest.approx(3e-4)


@pytest.mark.parametrize("logical_qubits,tiles,data_error", [(100, 230, 0.69), (1318, 2740, 8.22), (12581, 25481, 76.443)])
def test_fast_data_block(logical_qubits, tiles, data_error):
    assert FastDataBlock.get_n_tiles(logical_qubits) == tiles
    model = LogicalErrorModel(qec_scheme=QECScheme.make_gidney_fowler(), physical_error=1e-3)
    assert FastDataBlock(3).data_error(n_algo_qubits=logical_qubits, n_cycles=3, logical_error_model=model) == pytest.approx(data_error)


def test_ccz2t_factory_orders():
    factory, worse = CCZ2TFactory(), CCZ2TFactory(distillation_l1_d=7, distillation_l2_d=15)
    gates = GateCounts(t=10**8, toffoli=10**8)
    model = LogicalErrorModel(qec_scheme=QECScheme.make_gidney_fowler(), physical_error=1e-3)
    assert worse.factory_error(gates, model) > factory.factory_error(gates, model)
    assert worse.n_cycles(gates, model) < factory.n_cycles(gates, model)
    assert worse.n_physical_qubits() < factory.n_physical_qubits()


@pytest.mark.parametrize("d_X,d_Z,d_m,phys_err,p_out,footprint,cycles", [(7, 3, 3, 1e-4, "4.4e-08", 810, 18.1), (9, 3, 3, 1e-4, "9.3e-10", 1150, 18.1), (11, 5, 5, 1e-4, "1.9e-11", 2070, 30), (17, 7, 7, 1e-3, "4.5e-08", 4620, 42.6)])
def test_fifteen_to_one_matches_litinskis_paper(d_X, d_Z, d_m, phys_err, p_out, footprint, cycles):
    factory = FifteenToOne(d_X=d_X, d_Z=d_Z, d_m=d_m)
    model = LogicalErrorModel(qec_scheme=QECScheme.make_gidney_fowler(), physical_error=phys_err)
    assert f"{factory.factory_error(GateCounts(t=1), model):.1e}" == p_out
    assert round(factory.n_physical_qubits(), -1) == footprint
    assert factory.n_cycles(GateCounts(t=1), model) == math.ceil(cycles + 1e-9)
    for bad in ((1, 1, -1), (1, -1, 1), (-1, 1, 1), (5, 1, 1)):
        with pytest.raises(AssertionError):
            FifteenToOne(*bad)


def test_gidney_fowler_spreadsheet():
    cost = get_ccz2t_costs_from_error_budget(n_logical_gates=GateCounts(t=10**8, toffoli=10**8), n_algo_qubits=100, error_budget=0.01, phys_err=1e-3, cycle_time_us=1)
    np.testing.assert_allclose(cost.failure_prob, 0.0084, rtol=1e-3)
    np.testing.assert_allclose(cost.footprint, 4.00e5, rtol=1e-3)
    np.testing.assert_allclose(cost.duration_hr, 7.53, rtol=1e-3)


def test_grid_search():
    cost, factory, block = get_ccz2t_costs_from_grid_search(n_logical_gates=GateCounts(t=10**8, toffoli=10**8), n_algo_qubits=100, phys_err=1e-3, error_budget=0.1, cycle_time_us=1)
    assert isinstance(factory, CCZ2TFactory) and (factory.distillation_l1_d, factory.distillation_l2_d) == (15, 23) and block.data_d == 25
    # Lee et al.'s THC numbers (PRX Quantum 2, 2021, section IV.C), as Qualtran reproduces them.
    best, factory, block = get_ccz2t_costs_from_grid_search(n_logical_gates=GateCounts(toffoli=6665400000), n_algo_qubits=696, error_budget=1e-2, phys_err=1e-3, factory_iter=iter_ccz2t_factories(n_factories=4), cost_function=lambda pc: pc.duration_hr)
    assert best.failure_prob == 0.007725395132201774 and best.footprint == 2933032 and best.duration_hr == 89.1034375
    assert isinstance(factory, MultiFactory) and (factory.base_factory.distillation_l1_d, factory.base_factory.distillation_l2_d) == (17, 29) and block.data_d == 33


BEVERLAND = [
    (AlgorithmSummary(n_algo_qubits=100, n_logical_gates=GateCounts(rotation=30_100, measurement=int(1.4e6)), n_rotation_layers=501), 1e-3, 1.5e6, 1.5e5, 9, 602000),
    (AlgorithmSummary(n_algo_qubits=1318, n_logical_gates=GateCounts(t=int(5.53e7), rotation=int(2.06e8), toffoli=int(1.35e11), measurement=int(1.37e9)), n_rotation_layers=int(2.05e8)), 1e-2, 4.1e11, 4.1e11, 17, 5.44e11),
    (AlgorithmSummary(n_algo_qubits=12581, n_logical_gates=GateCounts(t=12, rotation=12, toffoli=int(3.73e9), measurement=int(1.08e9)), n_rotation_layers=12), 1 / 3, 1.23e10, 1.23e10, 13, 1.49e10),
]


@pytest.mark.parametrize("alg,budget,c_min,time_steps,distance,n_t", BEVERLAND)
def test_beverland_et_al(alg, budget, c_min, time_steps, distance, n_t):
    assert est.minimum_time_steps(error_budget=budget, alg=alg, rotation_model=BeverlandEtAlRotationCost) == pytest.approx(c_min, rel=0.1)
    assert est.code_distance(error_budget=budget, time_steps=time_steps, alg=alg, qec_scheme=QECScheme.make_beverland_et_al(), physical_error=1e-4) == distance
    assert est.t_states(error_budget=budget, alg=alg, rotation_model=BeverlandEtAlRotationCost) == pytest.approx(n_t, rel=0.1)


def test_rotation_cost_models():
    assert BeverlandEtAlRotationCost.rotation_cost(2**-3) == GateCounts(t=7)
    model = ConstantWithOverheadRotationCost(bitsize=13, overhead_rotation_cost=RotationLogarithmicModel(1, 1))
    assert model.rotation_cost(2**-3) == GateCounts(toffoli=11)
    assert BeverlandEtAlRotationCost.preparation_overhead(2**-3) == GateCounts()
    assert model.preparation_overhead(2**-3) == GateCounts(t=104)


def test_physical_cost_model_factories():
    gf = est.PhysicalCostModel.make_gidney_fowler(data_d=25)
    alg = AlgorithmSummary(n_algo_qubits=100, n_logical_gates=GateCounts(t=10**8, toffoli=10**8))
    assert gf.n_phys_qubits(alg) == 150 * 2 * 25**2 + CCZ2TFactory().n_physical_qubits()
    assert gf.duration_hr(alg) == pytest.approx(gf.n_cycles(alg) / 3.6e9)
    bev = est.PhysicalCostModel.make_beverland_et_al(data_d=9, data_block_name="fast")
    assert isinstance(bev.data_block, FastDataBlock) and isinstance(bev.factory, FifteenToOne)
    with pytest.raises(ValueError):
        est.PhysicalCostModel.make_beverland_et_al(data_d=9, data_block_name="odd")


# ---------------------------------------------------------------------------------------------
# Against Qualtran installed
# ---------------------------------------------------------------------------------------------


@needs_qualtran
@pytest.mark.parametrize("d_X,d_Z,d_m,phys_err", [(7, 3, 3, 1e-4), (11, 5, 5, 1e-4), (17, 7, 7, 1e-3), (9, 5, 3, 1e-3)])
def test_fifteen_to_one_against_qualtran(d_X, d_Z, d_m, phys_err):
    ours = FifteenToOne(d_X=d_X, d_Z=d_Z, d_m=d_m)
    theirs = their_sc.FifteenToOne(d_X=d_X, d_Z=d_Z, d_m=d_m)
    model = LogicalErrorModel(qec_scheme=QECScheme.make_gidney_fowler(), physical_error=phys_err)
    their_model = their_sc.LogicalErrorModel(qec_scheme=their_sc.QECScheme.make_gidney_fowler(), physical_error=phys_err)
    assert ours.p_fail(model) == pytest.approx(theirs.p_fail(their_model), rel=1e-9)
    assert ours.p_out(model) == pytest.approx(theirs.p_out(their_model), rel=1e-9)
    assert ours.n_cycles(GateCounts(t=1000), model) == theirs.n_cycles(TheirGateCounts(t=1000), their_model)


@needs_qualtran
def test_grid_search_against_qualtran():
    ours = get_ccz2t_costs_from_grid_search(n_logical_gates=GateCounts(t=10**7, toffoli=10**7, rotation=1000), n_algo_qubits=256, error_budget=0.05)
    theirs = their_sc.get_ccz2t_costs_from_grid_search(n_logical_gates=TheirGateCounts(t=10**7, toffoli=10**7, rotation=1000), n_algo_qubits=256, error_budget=0.05)
    assert (ours[0].failure_prob, ours[0].footprint, ours[0].duration_hr) == (theirs[0].failure_prob, theirs[0].footprint, theirs[0].duration_hr)
    assert (ours[1].distillation_l1_d, ours[1].distillation_l2_d, ours[2].data_d) == (theirs[1].distillation_l1_d, theirs[1].distillation_l2_d, theirs[2].data_d)


# ---------------------------------------------------------------------------------------------
# Azure's Resource Estimator: the QDK's recorded results (qdk 1.33.1, tests/estimate_port/) and the QDK itself
# ---------------------------------------------------------------------------------------------

QDK_CASES = json.loads((pathlib.Path(__file__).parent / "estimate_port" / "qdk_estimates.json").read_text())

try:
    from qdk.estimator import LogicalCounts as QdkLogicalCounts
except ImportError:  # pragma: no cover
    QdkLogicalCounts = None

needs_qdk = pytest.mark.skipif(QdkLogicalCounts is None, reason="qdk is not installed")
SECTIONS = ("physicalCounts", "logicalQubit", "tfactory", "errorBudget", "logicalCounts")


def _ours(job_params: dict, counts: dict) -> dict:
    return azure_estimate(counts, job_params.get("qubitParams"), job_params.get("qecScheme"), job_params.get("errorBudget", 1e-3), job_params.get("constraints"), job_params.get("distillationUnitSpecifications"), job_params.get("estimateType", "singlePoint"))


@pytest.mark.parametrize("key", sorted(QDK_CASES))
def test_azure_against_qdk_recorded(key):
    """Every number the QDK reports (physical qubits, runtime, rqops, the breakdown with its error rates, the
    logical qubit, the factory's rounds, units, distances, qubits, durations and output error, the budget split)
    equals ours exactly, on the 6 qubit presets × 3 schemes × 3 count sets, 26 constrained or custom cases and
    5 frontier estimates (every entry of the frontier, in the QDK's order)."""
    case = QDK_CASES[key]
    if case["status"] != "success":
        with pytest.raises(ValueError):
            _ours(case["jobParams"], case["logicalCounts"])
        return
    ours = _ours(case["jobParams"], case["logicalCounts"])
    if "frontierEntries" in case:
        assert len(ours["frontierEntries"]) == len(case["frontierEntries"])
        for a, b in zip(ours["frontierEntries"], case["frontierEntries"]):
            assert a == b
        return
    for section in SECTIONS:
        assert ours.get(section) == case.get(section), section


QDK_REVIEW2 = json.loads((pathlib.Path(__file__).parent / "estimate_port" / "qdk_review2.json").read_text())


@pytest.mark.parametrize("key", sorted(QDK_REVIEW2))
def test_azure_against_qdk_recorded_review2(key):
    """The second review's cases, recorded from the QDK (1.33.1): the binomial quantile's Newton branch with
    the per-unit failure probability near 1 (where it converges to 0), the knife-edge budgets at 1e15 T gates
    where the QDK adds cycles until the code distance moves, fasteval's folding and function table in scheme
    formulas (a NaN count is 0, `0 / 0` folds to 1), the memory counts the QDK echoes, the u64 wrap of the
    distance power, null fields, a unit whose `name` is not a string."""
    case = QDK_REVIEW2[key]
    if case["status"] != "success":
        with pytest.raises(ValueError):
            _ours(case["jobParams"], case["logicalCounts"])
        return
    ours = _ours(case["jobParams"], case["logicalCounts"])
    for section in ("physicalCounts", "logicalQubit", "tfactory", "errorBudget"):
        assert ours.get(section) == case.get(section), section
    assert ours["logicalCounts"] == case["qdkLogicalCounts"]


@needs_qdk
@pytest.mark.parametrize("counts,params", [
    ({"numQubits": 37, "tCount": 1234, "rotationCount": 77, "rotationDepth": 20, "cczCount": 50, "measurementCount": 999}, {"qubitParams": {"name": "qubit_gate_ns_e4"}, "constraints": {"maxTFactories": 2}}),
    ({"numQubits": 400, "tCount": 10**6, "cczCount": 10**5}, {"qubitParams": {"name": "qubit_maj_ns_e4"}, "qecScheme": {"name": "floquet_code"}, "errorBudget": 0.01}),
    ({"numQubits": 12, "tCount": 5000, "measurementCount": 100}, {"qubitParams": {"instructionSet": "GateBased", "oneQubitMeasurementTime": "1 µs", "oneQubitGateTime": "200 ns", "twoQubitGateTime": "300 ns", "oneQubitGateErrorRate": 1e-4, "twoQubitGateErrorRate": 5e-4, "oneQubitMeasurementErrorRate": 1e-3, "tGateErrorRate": 1e-3}, "constraints": {"maxPhysicalQubits": 50000}}),
])
def test_azure_against_qdk_live(counts, params):
    theirs = json.loads(QdkLogicalCounts(counts).estimate(params).json)
    theirs = theirs[0] if isinstance(theirs, list) else theirs
    ours = _ours(params, counts)
    for section in SECTIONS:
        assert ours.get(section) == theirs.get(section), section


def test_azure_entry_points_agree():
    counts = LogicalCounts(num_qubits=100, t_count=100_000, ccz_count=10_000, measurement_count=10_000)
    a = azure_estimate(counts)
    b = azure_estimate(counts.asdict(), "qubit_gate_ns_e3", "surface_code", 1e-3)
    est = azure_estimator(counts)
    c = est.estimate(ErrorBudget.partition(1e-3, counts)).asdict()
    assert a == b == c
    assert isinstance(est, AzureEstimator) and a["physicalCounts"]["physicalQubits"] == 400060 and a["physicalCounts"]["runtime"] == 1064000000
    assert a["tfactory"]["numUnitsPerRound"] == [18, 1] and a["tfactory"]["codeDistancePerRound"] == [5, 15] and a["logicalQubit"]["codeDistance"] == 19


def test_azure_binomial_quantile_is_the_exact_one_below_a_thousand_units():
    """The `probability` crate sums the mass function below 1000 trials: its quantile is the smallest k with
    CDF(k) ≥ p. Above, its normal expansion lands within a couple of units of the exact quantile."""

    def exact(n, q, p):
        cdf, k = 0.0, 0
        while True:
            cdf += math.exp(math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1) + k * math.log(1 - q) + (n - k) * math.log(q))
            if cdf >= p:
                return k
            k += 1

    for n, q in ((18, 0.3), (23, 0.05), (303, 0.6), (999, 0.001), (640, 0.9)):
        for p in (0.005, 0.0033, 0.5, 0.99):
            assert az._Binomial(n, q).inverse(p) == exact(n, q, p), (n, q, p)
    assert abs(az._Binomial(2000, 0.99).inverse(0.005) - exact(2000, 0.99, 0.005)) <= 2  # Newton branch (n·p·q ≤ 80)
    assert abs(az._Binomial(20137, 0.7856).inverse(0.0025) - exact(20137, 0.7856, 0.0025)) <= 2  # normal branch
    assert az._Binomial(50, 0.2).inverse(0.0) == 0 and az._Binomial(50, 0.2).inverse(1.0) == 50


def test_azure_powi_matches_the_qdk_last_bit():
    assert az._powi(0.1, 10) == 1.0000000000000011e-10 != 0.1**10 and 0.03 * az._powi(0.1, 10) == 3.000000000000003e-12
    assert az._powi(2.0, -3) == 0.125 and az._powi(7.0, 0) == 1.0


def test_azure_expressions_and_times():
    assert az.evaluate_expression("2 * codeDistance ^ 2 + 1", {"codeDistance": 3}) == 19.0
    assert az.evaluate_expression("(4 * twoQubitGateTime + 2 * oneQubitMeasurementTime) * codeDistance", {"twoQubitGateTime": 50, "oneQubitMeasurementTime": 100, "codeDistance": 5}) == 2000.0
    with pytest.raises(ValueError):
        az.evaluate_expression("2 * d", {"codeDistance": 3})
    with pytest.raises(ValueError):
        az.evaluate_expression("__import__('os').getcwd()", {})
    assert az.parse_time("1 µs") == 1000 and az.parse_time("2.5 ms") == 2_500_000 and az.parse_time("1 s") == 10**9 and az.parse_time(100) == 100 and az.parse_time("50ns") == 50
    with pytest.raises(ValueError):
        az.parse_time("fast")


def test_azure_qubit_normalisation():
    q = physical_qubit({"instructionSet": "GateBased", "oneQubitMeasurementTime": "80 ns", "oneQubitGateTime": 30, "oneQubitGateErrorRate": 2e-4, "oneQubitMeasurementErrorRate": 5e-4, "tGateErrorRate": 3e-4})
    assert isinstance(q, GateBasedQubit) and (q.two_qubit_gate_time, q.t_gate_time, q.two_qubit_gate_error_rate, q.idle_error_rate) == (30, 30, 2e-4, 5e-4)
    assert q.clifford_error_rate() == 5e-4 and q.readout_error_rate() == 5e-4
    m = physical_qubit({"name": "qubit_maj_ns_e4", "oneQubitMeasurementErrorRate": {"process": 2e-4, "readout": 1e-4}, "twoQubitJointMeasurementErrorRate": 1.5e-4})
    assert isinstance(m, MajoranaQubit) and m.one_qubit_measurement_error_rate == (2e-4, 1e-4) and m.two_qubit_joint_measurement_error_rate == (1.5e-4, 1.5e-4)
    assert m.clifford_error_rate() == 2e-4 and m.readout_error_rate() == 1.5e-4 and m.t_gate_error_rate == 0.05 and m.t_gate_time == 100
    assert physical_qubit("qubit_gate_us_e3").t_gate_error_rate == 1e-6 and physical_qubit().name == "qubit_gate_ns_e3"
    assert set(az.QUBIT_NAMES) == {"qubit_gate_ns_e3", "qubit_gate_ns_e4", "qubit_gate_us_e3", "qubit_gate_us_e4", "qubit_maj_ns_e4", "qubit_maj_ns_e6"}
    with pytest.raises(ValueError):
        physical_qubit("qubit_gate_ps_e9")
    with pytest.raises(ValueError):
        physical_qubit({"instructionSet": "GateBased", "oneQubitGateTime": 30})


def test_azure_scheme_validation():
    gate, maj = physical_qubit("qubit_gate_ns_e3"), physical_qubit("qubit_maj_ns_e6")
    assert qec_scheme("surface_code", gate).error_correction_threshold == 0.01 and qec_scheme("surface_code", maj).error_correction_threshold == 0.0015
    assert qec_scheme("floquet_code", maj).physical_qubits(5) == 132 and qec_scheme("floquet_code", maj).cycle_time(maj, 5) == 1500
    with pytest.raises(ValueError):
        qec_scheme("floquet_code", gate)
    with pytest.raises(ValueError):
        qec_scheme({"name": "surface_code", "crossingPrefactor": 0.6}, gate)
    with pytest.raises(ValueError):
        qec_scheme({"name": "surface_code", "errorCorrectionThreshold": 1e-3}, gate)
    scheme = qec_scheme("surface_code", gate)
    assert scheme.logical_error_rate(gate, 19) == 3.000000000000003e-12 and scheme.compute_code_distance(gate, 1.5527950310559005e-11) == 19
    with pytest.raises(ValueError):
        scheme.compute_code_distance(gate, 1e-40)


def test_azure_error_budget_partition():
    small = LogicalCounts(num_qubits=10, t_count=5)
    assert ErrorBudget.partition(0.3, small) == ErrorBudget(0.15, 0.15, 0.0)
    assert ErrorBudget.partition(0.3, LogicalCounts(num_qubits=10, t_count=5, rotation_count=1)) == ErrorBudget(0.3 / 3.0, 0.3 / 3.0, 0.3 / 3.0)
    assert ErrorBudget.partition(0.3, LogicalCounts(num_qubits=10, measurement_count=5)) == ErrorBudget(0.3, 0.0, 0.0)
    assert ErrorBudget.partition({"logical": 0.1, "tStates": 0.05, "rotations": 0.0}, small) == ErrorBudget(0.1, 0.05, 0.0)
    for bad in (0.0, 1.0, {"logical": 0.1, "tStates": 0.0, "rotations": 0.0}):
        with pytest.raises(ValueError):
            ErrorBudget.partition(bad, small)


def test_azure_factory_frontier():
    gate = physical_qubit("qubit_gate_ns_e3")
    scheme = qec_scheme("surface_code", gate)
    frontier = find_nondominated_tfactories(scheme, gate, DistillationUnitTemplate.defaults(), 3.5714285714285715e-09, 19)
    points = [(f.normalized_qubits(), f.duration()) for f in frontier]
    assert points == sorted(points, reverse=True) and len(frontier) >= 2
    for a in points:
        for b in points:
            assert not ((a[0] < b[0] and a[1] <= b[1]) or (a[0] <= b[0] and a[1] < b[1]))
    assert all(f.output_error_rate() <= 3.5714285714285715e-09 for f in frontier)
    chosen = min(frontier, key=lambda f: f.normalized_volume())
    assert chosen.asdict()["numUnitsPerRound"] == [18, 1] and chosen.physical_qubits() == 18000 and chosen.duration() == 92000
    trivial = find_nondominated_tfactories(scheme, physical_qubit("qubit_gate_ns_e4"), DistillationUnitTemplate.defaults(), 2.5e-4, 3)
    assert len(trivial) == 1 and trivial[0].unit_names() == ["trivial 1-to-1"] and trivial[0].code_distance_per_round() == [3]


def test_azure_constraints():
    counts = LogicalCounts(num_qubits=100, t_count=100_000, ccz_count=10_000, measurement_count=10_000)
    with pytest.raises(ValueError):
        azure_estimate(counts, constraints={"maxDuration": "3 s", "maxPhysicalQubits": 10**6})
    with pytest.raises(ValueError):
        azure_estimate(counts, constraints={"logicalDepthFactor": 0.5})
    with pytest.raises(ValueError):
        azure_estimate(counts, constraints={"maxPhysicalQubits": 1000})
    assert Constraints.fromspec({"maxDuration": "2 ms"}).max_duration == 2_000_000 and Constraints.fromspec(None).max_distillation_rounds == 3
    r = azure_estimate(counts, constraints={"maxTFactories": 3})
    assert r["physicalCounts"]["breakdown"]["numTfactories"] <= 3 and r["jobParams"]["constraints"]["maxTFactories"] == 3
    assert azure_estimate(LogicalCounts(num_qubits=50, measurement_count=1000))["physicalCounts"]["breakdown"]["numTstates"] == 0
    with pytest.raises(ValueError):
        azure_estimate(LogicalCounts(num_qubits=50))


# ---------------------------------------------------------------------------------------------
# The 2.0 review's regressions (the QDK's parse-time checks, fasteval's reading, its hangs)
# ---------------------------------------------------------------------------------------------


def test_azure_formulas_read_as_fasteval_reads_them():
    ev = az.evaluate_expression
    d = {"codeDistance": 3}
    # Unary minus binds tighter than every binary operator, and a sign is part of a constant.
    assert ev("-2^2", {}) == 4.0 and ev("-2^2^2", {}) == 16.0 and ev("1 - 2^2", {}) == -3.0 and ev("(-2)^3", {}) == -8.0
    assert ev("-(codeDistance^2)", d) == -9.0 and ev("-codeDistance^2", d) == 9.0 and ev("--2^2", {}) == 4.0 and ev("-+2^2", {}) == 4.0 and ev("+-2^2", {}) == 4.0 and ev("2 * -codeDistance", d) == -6.0
    # Each binary operator is its own precedence level, lowest to highest + - * / % ^, and ^ is right-associative.
    assert ev("7 * 3 % 2", {}) == 7.0 and ev("8 / 4 % 3", {}) == 8.0 and ev("codeDistance / 2 % 1", d) == math.inf and ev("7 % 3 + 2 * codeDistance", {"codeDistance": 5}) == 11.0
    assert ev("2^3^2", {}) == 512.0 and ev("2^-2^2", {}) == 16.0 and ev("1 - 2 - 3", {}) == -4.0 and ev("2 / 2 / 2", {}) == 0.5 and ev("1 < 2 < 3", {}) == 1.0
    # Compile-time folding: every constant of a + or * chain is combined first (Sub operands negated, Div operands
    # inverted), a constant within 8 ε of the identity is dropped, and so is a NaN constant.
    assert ev("codeDistance + 0.1 + 0.2", d) == 3.3 and ev("codeDistance + 1e16 - 1e16", d) == 3.0 and ev("0.1 + 0.2", {}) == 0.30000000000000004 and ev("codeDistance * 1.0000000000000002", d) == 3.0
    assert ev("0 / 0", {}) == 1.0 and ev("1 % 0 * codeDistance", d) == 3.0 and ev("10^400 - 10^400 + codeDistance", d) == 3.0
    # The arithmetic's edges.
    assert ev("1 / 0", {}) == math.inf and ev("-1 / 0", {}) == -math.inf and math.isnan(ev("(codeDistance - codeDistance) / (codeDistance - codeDistance)", d))
    assert math.isnan(ev("(-8) ^ 0.5", {})) and math.isnan(ev("1e400 % 2", {})) and ev("(-10)^401", {}) == -math.inf and ev("(-0)^-1", {}) == -math.inf and ev("0^-0.5", {}) == math.inf and ev("0^0", {}) == 1.0
    assert ev("-7 % 3", {}) == -1.0 and ev("7 % -3", {}) == 1.0 and ev("inf", {}) == math.inf and math.isnan(ev("NaN", {})) and ev("1e3", {}) == ev("1E3", {}) == ev("1k", {}) == 1000.0 and ev("2m", {}) == 0.002 and ev("1.", {}) == ev("+1", {}) == 1.0 and ev(".5", {}) == 0.5
    # The function table, comparisons and logic.
    assert ev("min(2, 3)", {}) == 2.0 and ev("max(codeDistance, 1)", d) == 3.0 and ev("abs(-2)", {}) == 2.0 and ev("log(100)", {}) == 2.0 and ev("log(2, 8)", {}) == 3.0 and ev("log(e(), e())", {}) == 1.0 and ev("int(2.5)", {}) == 2.0 and ev("ceil(2.1)", {}) == 3.0 and ev("floor(2.9)", {}) == 2.0
    assert ev("round(2.5)", {}) == 3.0 and ev("round(-2.5)", {}) == -3.0 and ev("round(0.5, 2.3)", {}) == 2.5 and ev("sign(-3)", {}) == -1.0 and ev("sign(0)", {}) == 1.0 and ev("sin(0)", {}) == 0.0 and ev("cos(0)", {}) == 1.0 and ev("tanh(0)", {}) == 0.0 and ev("acosh(1)", {}) == 0.0 and math.isnan(ev("asin(2)", {}))
    assert ev("pi()", {}) == math.pi and ev("e()", {}) == math.e and ev("codeDistance()", d) == 3.0 and ev("[1]", {}) == 1.0
    assert ev("1 < 2", {}) == ev("1 == 1", {}) == ev("1 <= 2", {}) == ev("1 != 2", {}) == ev("2 >= 2", {}) == 1.0 and ev("2 > 1 and 3 > 2", {}) == 1.0 and ev("1 || 0", {}) == 1.0 and ev("2 && 0", {}) == 0.0 and ev("!0", {}) == 1.0 and ev("1 and 2", {}) == 2.0 and ev("1 or 2", {}) == 1.0 and ev("0 or codeDistance", d) == 3.0
    # Refused: Python's own spellings, unknown names, the wrong number of arguments, and fasteval's limits.
    for bad in ("2 ** 3", "1_000 * z", "0x10 + 1", "abs", "sqrt(4)", "pi", "nan", "min()", "pi(1)", "log(1, 2, 3)", "1 // 2", "~1", "2 if 1 else 3", "(1, 2)", "not 1", "2 3", "2(3)", "", "()", "1 +", "1e", "1 + 2 # x", "−2", "print(1)", 100):
        with pytest.raises(ValueError):
            ev(bad, {"z": 1.0})
    with pytest.raises(ValueError, match="unknown variable"):
        ev("2 * d", d)
    with pytest.raises(ValueError, match="TooLong"):
        ev("1" + "+1" * 3000, {})
    assert ev("1" + "+1" * 2000, {}) == 2001.0 and ev("z" + "+z" * 1000, {"z": 1.0}) == 1001.0 and ev("z" + "^z" * 1000, {"z": 1.0}) == 1.0  # long chains need no deep recursion
    for too_deep in ("(" * 33 + "1" + ")" * 33, "-" * 34 + "2"):
        with pytest.raises(ValueError, match="TooDeep"):
            ev(too_deep, {})
    assert ev("(" * 32 + "1" + ")" * 32, {}) == 1.0 and ev("-" * 33 + "2", {}) == -2.0
    with pytest.raises(ValueError, match="SlabOverflow"):
        ev("(1)+" * 64 + "1", {})
    assert ev("(1)+" * 63 + "1", {}) == 64.0
    # The scheme formula the review used: the QDK's patch of 27 qubits, not 19.
    scheme = qec_scheme({"name": "custom", "errorCorrectionThreshold": 0.01, "crossingPrefactor": 0.03, "logicalCycleTime": "4 * twoQubitGateTime * codeDistance", "physicalQubitsPerLogicalQubit": "-2^2 + 10 + codeDistance"}, physical_qubit("qubit_gate_ns_e3"))
    assert scheme.physical_qubits(13) == 27


def test_azure_refuses_what_the_qdk_refuses_at_parse_time():
    base = {"instructionSet": "GateBased", "oneQubitMeasurementTime": "100 ns", "oneQubitGateTime": "50 ns", "twoQubitGateTime": "50 ns", "tGateTime": "50 ns", "oneQubitMeasurementErrorRate": 1e-3, "oneQubitGateErrorRate": 1e-3, "twoQubitGateErrorRate": 1e-3, "tGateErrorRate": 1e-3}
    physical_qubit(base)
    for bad in ({"tGateErrorRate": 0.0}, {"tGateErrorRate": 1.5}, {"tGateErrorRate": 1.0}, {"oneQubitGateErrorRate": True}, {"oneQubitGateErrorRate": "1e-3"}):
        with pytest.raises(ValueError, match="between 0.0 and 1.0"):
            physical_qubit({**base, **bad})
    with pytest.raises(ValueError, match="process"):
        physical_qubit({"name": "qubit_maj_ns_e4", "oneQubitMeasurementErrorRate": {"process": 2e-4}})
    for bad in ({"tCount": -5}, {"tCount": 5.0}, {"tCount": True}, {"numQubits": "10"}):
        with pytest.raises(ValueError, match="non-negative integer"):
            LogicalCounts.fromdict({"numQubits": 10, "tCount": 5, **bad})
    with pytest.raises(ValueError, match="missing"):
        ErrorBudget.partition({"logical": 0.1}, LogicalCounts(num_qubits=10, t_count=5))
    assert ErrorBudget.partition({"logical": 0.1, "tstates": 0.05, "rotations": 0.0}, LogicalCounts(num_qubits=10, t_count=5)) == ErrorBudget(0.1, 0.05, 0.0)
    gate = physical_qubit("qubit_gate_ns_e3")
    with pytest.raises(ValueError, match="non-negative integer"):
        qec_scheme({"name": "surface_code", "distanceCoefficientPower": -1}, gate)
    with pytest.raises(ValueError, match="non-negative integer"):
        qec_scheme({"name": "surface_code", "distanceCoefficientPower": 1.5}, gate)
    # Beyond the QDK (documented): times as numbers of nanoseconds.
    assert physical_qubit({**base, "tGateTime": 50}).t_gate_time == 50


def test_azure_unit_spec_with_a_name_is_the_named_unit():
    custom = {"displayName": "7-to-1 custom", "numInputTs": 7, "numOutputTs": 1, "failureProbabilityFormula": "7 * z + 20 * c", "outputErrorRateFormula": "7 * z^3 + 7 * c", "physicalQubitSpecification": {"numUnitQubits": 7, "durationInQubitCycleTime": 10}}
    assert DistillationUnitTemplate.fromspec(custom).name == "7-to-1 custom"
    named = DistillationUnitTemplate.fromspec({**custom, "name": "15-1 RM"})
    assert named.name == DistillationUnitTemplate.from_name("15-1 RM").name
    assert DistillationUnitTemplate.fromspec({"name": "15-1 RM", "foo": 1}).name == named.name
    with pytest.raises(ValueError, match="Invalid distillation unit"):
        DistillationUnitTemplate.fromspec({**custom, "name": "my-unit"})


def test_azure_hangs_become_errors():
    # The binomial quantile's Newton branch as the probability crate runs it: with the per-unit failure probability
    # near 1 the iterate steps below 0, which the crate's `as usize` reads as 0, and it returns 0 (the first review
    # took this for a hang); only a NaN or infinite iterate never returns, and that raises here.
    assert az._Binomial(2000, 0.9999).inverse(1e-12) == 0 and az._Binomial(1024, 0.9975001068).inverse(0.01) == 0
    assert az._Binomial(2000, 0.01).inverse(0.5) == 1980
    with pytest.raises(ValueError, match="does not converge"):
        az._Binomial(10569, 1.1996456494734822e-05).inverse(1 / 300)  # the live QDK hangs here
    # Every first distillation round fails (15 z >= 1): the QDK adds cycles forever; this raises, and quickly, by
    # jumping to the cycle count where the code distance moves instead of stepping one cycle at a time.
    hopeless = {"instructionSet": "GateBased", "oneQubitMeasurementTime": "100 ns", "oneQubitGateTime": "50 ns", "twoQubitGateTime": "50 ns", "tGateTime": "50 ns", "oneQubitMeasurementErrorRate": 1e-3, "oneQubitGateErrorRate": 1e-3, "twoQubitGateErrorRate": 1e-3, "tGateErrorRate": 0.5}
    t0 = time.perf_counter()
    with pytest.raises(ValueError, match="no T factory"):
        azure_estimate({"numQubits": 10, "tCount": 100}, hopeless)
    with pytest.raises(ValueError, match="no T factory"):
        azure_estimate({"numQubits": 1000, "tCount": 10**12, "measurementCount": 10**12}, hopeless)
    with pytest.raises(ValueError, match="no T factory"):
        azure_estimate({"numQubits": 10, "tCount": 100}, {"name": "qubit_gate_ns_e3", "tGateErrorRate": 0.3})
    assert time.perf_counter() - t0 < 5.0
    with pytest.raises(ValueError, match="frontier is empty"):
        azure_estimate({"numQubits": 10, "tCount": 100}, hopeless, estimate_type="frontier")
    # Inputs the QDK refuses at parse time no longer hang here: a distance power is u32, a count u64; and the
    # distance power is applied in u64 arithmetic, which wraps, as the QDK's does.
    gate = physical_qubit("qubit_gate_ns_e3")
    with pytest.raises(ValueError, match="below 2\\^32"):
        qec_scheme({"name": "surface_code", "distanceCoefficientPower": 2**32}, gate)
    scheme = qec_scheme({"name": "surface_code", "distanceCoefficientPower": 2**32 - 1}, gate)
    assert scheme.logical_error_rate(gate, 3) == 0.03 * float(pow(3, 2**32 - 1, 2**64)) * az._powi(0.1, 2)
    assert qec_scheme({"name": "surface_code", "distanceCoefficientPower": 12}, gate).logical_error_rate(gate, 49) == 0.03 * float(49**12 % 2**64) * az._powi(0.1, 25)
    for key in ("maxCodeDistance",):
        with pytest.raises(ValueError, match="below 2\\^64"):
            qec_scheme({"name": "surface_code", key: 2**64}, gate)
    with pytest.raises(ValueError, match="below 2\\^64"):
        LogicalCounts.fromdict({"numQubits": 10, "tCount": 2**64})
    assert LogicalCounts.fromdict({"numQubits": 10, "tCount": 2**64 - 1}).t_count == 2**64 - 1


def test_azure_rejects_what_the_qdk_rejects_second_review():
    """The second review's rejection gaps, each checked against the live QDK: times, names, budgets as strings,
    booleans where serde wants a number, the constraints' and distillation units' integer fields, the counts the
    QDK echoes, null scheme fields, formulas at even distances, and the casts of NaN and infinite formula values."""
    base = {"instructionSet": "GateBased", "oneQubitMeasurementTime": "100 ns", "oneQubitGateTime": "50 ns", "twoQubitGateTime": "50 ns", "tGateTime": "50 ns", "oneQubitMeasurementErrorRate": 1e-3, "oneQubitGateErrorRate": 1e-3, "twoQubitGateErrorRate": 1e-3, "tGateErrorRate": 1e-3}
    gate = physical_qubit("qubit_gate_ns_e3")
    for bad in (True, None, " 50 ns", "50 ns ", "50\xa0ns", "50 xs", "-50 ns", [50]):
        with pytest.raises(ValueError):
            physical_qubit({**base, "twoQubitGateTime": bad})
    assert physical_qubit({**base, "twoQubitGateTime": "50\tns"}).two_qubit_gate_time == 50 and az.parse_time("1e30 s") == 2**64 - 1 and az.parse_time("0.4 ns") == 0
    for bad in (5, None):
        with pytest.raises(ValueError, match="name"):
            physical_qubit({**base, "name": bad})
    assert physical_qubit({"name": "qubit_maj_ns_e4", "oneQubitMeasurementErrorRate": {"process": 2e-4, "readout": 1e-4, "foo": 1}}).one_qubit_measurement_error_rate == (2e-4, 1e-4)  # serde ignores the extra key
    counts = LogicalCounts(num_qubits=10, t_count=100)
    for bad in ("0.001", [0.001], {"logical": "0.1", "tStates": 0.1, "rotations": 0.0}, {"logical": 0.1, "tStates": True, "rotations": 0.0}):
        with pytest.raises(ValueError):
            ErrorBudget.partition(bad, counts)
    # Schemes: numbers are not booleans or strings, formulas are strings, null is absent, and the formulas are
    # checked at every distance from 3 to the maximum, even ones included.
    for bad in ({"errorCorrectionThreshold": True}, {"crossingPrefactor": "0.03"}, {"logicalCycleTime": 100}, {"physicalQubitsPerLogicalQubit": None, "name": 5}, {"maxCodeDistance": None}, {"physicalQubitsPerLogicalQubit": "codeDistance % 2 * 100"}, {"logicalCycleTime": "codeDistance % 2 * 100"}):
        with pytest.raises(ValueError):
            qec_scheme({"name": "surface_code", **bad}, gate)
    scheme = qec_scheme({"name": "surface_code", "crossingPrefactor": None, "errorCorrectionThreshold": None, "distanceCoefficientPower": None, "physicalQubitsPerLogicalQubit": "(codeDistance % 2) * 100 + 1"}, gate)
    assert (scheme.crossing_prefactor, scheme.error_correction_threshold, scheme.distance_coefficient_power) == (0.03, 0.01, 0) and scheme.physical_qubits(4) == 1
    # A NaN count is 0 and an infinite one 2^64 - 1 (the QDK's `as u64`); `0 / 0` is the constant 1.
    assert qec_scheme({"name": "surface_code", "physicalQubitsPerLogicalQubit": "(codeDistance - codeDistance) / (codeDistance - codeDistance) + 100 * codeDistance"}, gate).physical_qubits(3) == 0
    assert qec_scheme({"name": "surface_code", "logicalCycleTime": "1 / 0"}, gate).cycle_time(gate, 3) == 2**64 - 1
    assert qec_scheme({"name": "surface_code", "logicalCycleTime": "0 / 0 * 400 * codeDistance"}, gate).cycle_time(gate, 3) == 1200
    # Constraints: u64 counts, a float factor (not a bool), a time string; null is absent where the QDK's field is optional.
    for bad in ({"maxTFactories": -1}, {"maxTFactories": 2.5}, {"maxTFactories": 2.0}, {"maxTFactories": True}, {"maxTFactories": 2**64}, {"maxTFactories": "3"}, {"logicalDepthFactor": True}, {"logicalDepthFactor": "2"}, {"maxDuration": None}, {"maxDuration": True}, {"maxDistillationRounds": True}, {"maxDistillationRounds": None}, {"maxDistillationRounds": 1.5}, {"maxPhysicalQubits": 2**64}, []):
        with pytest.raises(ValueError):
            Constraints.fromspec(bad)
    c = Constraints.fromspec({"logicalDepthFactor": None, "maxTFactories": None, "maxPhysicalQubits": None, "maxDistillationRounds": 2})
    assert (c.logical_depth_factor, c.max_t_factories, c.max_physical_qubits, c.max_distillation_rounds) == (None, None, None, 2) and Constraints.fromspec({"logicalDepthFactor": 1}).logical_depth_factor == 1.0
    est = azure_estimator(counts, constraints={"logicalDepthFactor": 1e300})
    assert est.compute_num_cycles(ErrorBudget.partition(1e-3, counts)) == 2**64 - 1  # the QDK's `.ceil() as u64` saturates
    # Distillation units: positive u64 counts, a string name, every field present, a formula that parses.
    custom = {"displayName": "custom 15-1", "numInputTs": 15, "numOutputTs": 1, "failureProbabilityFormula": "15 * z + 356 * c", "outputErrorRateFormula": "35 * z^3 + 7.1 * c", "physicalQubitSpecification": {"numUnitQubits": 31, "durationInQubitCycleTime": 24}}
    for bad in ({"numInputTs": -1}, {"numInputTs": 1.5}, {"numInputTs": 15.0}, {"numInputTs": "15"}, {"numInputTs": True}, {"numInputTs": 2**64}, {"numOutputTs": 2.0}, {"numOutputTs": "1"}, {"numOutputTs": None}, {"displayName": 5}, {"displayName": None}, {"failureProbabilityFormula": None}, {"failureProbabilityFormula": "15 * z ** 1"}, {"physicalQubitSpecification": {"numUnitQubits": 31, "durationInQubitCycleTime": 1.5}}, {"physicalQubitSpecification": {"numUnitQubits": 31}}):
        with pytest.raises(ValueError):
            DistillationUnitTemplate.fromspec({**custom, **bad})
    assert DistillationUnitTemplate.fromspec({**custom, "name": 5}).name == "custom 15-1" and DistillationUnitTemplate.fromspec({**custom, "physicalQubitSpecification": {"numUnitQubits": 31, "durationInQubitCycleTime": 0, "foo": 1}}).physical_qubit_specification == (31, 0)
    with pytest.raises(ValueError):
        azure_estimate(counts, distillation_unit_specifications={})
    # The memory counts: validated as counts, echoed as the QDK echoes them (only when given).
    for bad in ({"numComputeQubits": 1.5}, {"numComputeQubits": "x"}, {"readFromMemoryCount": -1}):
        with pytest.raises(ValueError, match="non-negative integer"):
            LogicalCounts.fromdict({"numQubits": 10, "tCount": 100, **bad})
    echoed = azure_estimate({"numQubits": 10, "tCount": 100, "numComputeQubits": 7, "readFromMemoryCount": 3, "writeToMemoryCount": 4})["logicalCounts"]
    assert (echoed["numComputeQubits"], echoed["readFromMemoryCount"], echoed["writeToMemoryCount"]) == (7, 3, 4) and "numComputeQubits" not in azure_estimate(counts)["logicalCounts"]


def test_qualtran_density_refuses_probabilities_past_one():
    from stabilizer_qec.estimate import qualtran as qt

    d = qt._Density(2)
    with pytest.raises(ValueError, match="between 0 and 1"):
        d.noisy_rotation("ZZ", 1.5, 0.0, 0.0)
    with pytest.raises(ValueError, match="between 0 and 1"):
        d.storage_error("Z", [0.1, -0.2])

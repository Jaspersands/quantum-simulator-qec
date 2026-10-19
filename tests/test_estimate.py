"""The estimator ports (`stabilizer_qec.estimate`): Qualtran's surface-code models against the
numbers Qualtran's own tests pin (and against Qualtran where it is installed), and Azure's
Resource Estimator model against the QDK's results recorded in tests/estimate_port/."""

from __future__ import annotations

import gzip
import json
import math
import pathlib
import sys
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
    EstimatorResult,
    GateBasedQubit,
    LogicalCounts,
    MajoranaQubit,
    azure_estimate,
    azure_estimator,
    find_nondominated_tfactories,
    format_duration,
    format_metric_prefix,
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
    NoisyPauliRotation,
    PauliFlipChannel,
    QECScheme,
    RotationLogarithmicModel,
    ConstantWithOverheadRotationCost,
    TComplexity,
    get_ccz2t_costs_from_error_budget,
    get_ccz2t_costs_from_grid_search,
    iter_ccz2t_factories,
    storage_error,
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
# Qualtran's submodules, its ui, and the details of its classes
# ---------------------------------------------------------------------------------------------

# The names each of Qualtran 0.7.0's submodules of qualtran.surface_code holds that are Qualtran's own (defined
# there or imported from Qualtran), as recorded from Qualtran; ui's Dash layout, inputs and app are not ported.
QUALTRAN_SUBMODULES = {
    "algorithm_summary": ["AlgorithmSummary", "GateCounts"],
    "beverland_et_al_model": ["GateCounts", "code_distance", "minimum_time_steps", "n_discrete_logical_gates", "t_states"],
    "ccz2t_factory": ["CCZ2TFactory", "MagicStateFactory"],
    "data_block": ["CompactDataBlock", "DataBlock", "FastDataBlock", "IntermediateDataBlock", "SimpleDataBlock"],
    "fifteen_to_one_factory": ["FifteenToOne", "MagicStateFactory", "NoisyPauliRotation", "storage_error"],
    "gidney_fowler_model": ["AlgorithmSummary", "CCZ2TFactory", "DataBlock", "MagicStateFactory", "MultiFactory", "PhysicalCostModel", "PhysicalCostsSummary", "PhysicalParameters", "QECScheme", "SimpleDataBlock", "get_ccz2t_costs", "get_ccz2t_costs_from_error_budget", "get_ccz2t_costs_from_grid_search", "iter_ccz2t_factories", "iter_simple_data_blocks"],
    "magic_state_factory": ["MagicStateFactory"],
    "multi_factory": ["MagicStateFactory", "MultiFactory"],
    "physical_cost_model": ["LogicalErrorModel", "PhysicalCostModel"],
    "physical_cost_summary": ["PhysicalCostsSummary"],
    "physical_parameters": ["PhysicalParameters"],
    "qec_scheme": ["LogicalErrorModel", "QECScheme"],
    "rotation_cost_model": ["BeverlandEtAlRotationCost", "ConstantWithOverheadRotationCost", "GateCounts", "RotationCostModel", "RotationLogarithmicModel", "SevenDigitsOfPrecisionConstantCost"],
    "t_factory_utils": ["NoisyPauliRotation", "_rotation_unitary", "storage_error"],
    "ui": ["AlgorithmSummary", "CCZ2TFactory", "FastDataBlock", "FifteenToOne", "GateCounts", "LogicalErrorModel", "MagicStateFactory", "MultiFactory", "PhysicalParameters", "QECScheme", "_BEVERLAND_MODEL", "_GIDNEY_FOWLER_MODEL", "_MAGIC_FACTORIES", "_QEC_SCHEMES", "_ROTATION_MODELS", "_SUPPORTED_ESTIMATION_MODELS", "beverland_et_al_model", "compute_duration", "create_qubit_pie_chart", "create_runtime_plot", "format_duration", "get_objects", "gidney_fowler_model", "min_num_factories", "rotation_cost_model", "total_magic", "update"],
}
# Qualtran's names not ported: the private helpers of its resource counting and Cirq circuit, and the Dash app.
QUALTRAN_NOT_PORTED = {"QECGatesCost", "QubitCount", "get_cost_value", "_QEC_COUNT", "_QUBIT_COUNT", "_build_factory", "_ALGORITHM_INPUTS", "_ALL_INPUTS", "_ERROR_INPUTS", "_ESTIMATION_INPUTS", "_MAGIC_FACTORIES_INPUTS", "_QEC_INPUTS", "_ROTATION_MODELS_INPUTS", "_MAGIC_FACTORIES_components", "algorithm_summary_components", "qec_summary_components", "rotation_cost_model_components", "input_components", "create_ouputs", "app"}


@pytest.mark.parametrize("name", sorted(QUALTRAN_SUBMODULES))
def test_qualtran_submodules(name):
    """Each of Qualtran's 15 submodules imports from ours under its name and holds Qualtran's names, the same
    objects as the package's."""
    import importlib

    import stabilizer_qec.estimate.qualtran as pkg

    mod = importlib.import_module(f"stabilizer_qec.estimate.qualtran.{name}")
    for attr in QUALTRAN_SUBMODULES[name]:
        assert hasattr(mod, attr), attr
        if hasattr(pkg, attr) and not attr.startswith("_"):
            assert getattr(mod, attr) is getattr(pkg, attr), attr
    assert set(mod.__all__) <= set(dir(mod))


def test_qualtran_package_keeps_its_names():
    """The package holds every name the module had, its private helpers too, and the names stabilizer_qec.estimate
    re-exports are the package's."""
    import stabilizer_qec.estimate.qualtran as pkg
    from stabilizer_qec.estimate.qualtran import _Density, _plain_count  # noqa: F401

    assert pkg.GateCounts is est.GateCounts and pkg.FifteenToOne is est.FifteenToOne and pkg.TComplexity is est.TComplexity
    assert {"AlgorithmSummary", "PauliFlipChannel", "TComplexity", "t_states"} <= set(pkg.__all__)
    assert pkg.data_block.SimpleDataBlock is pkg.SimpleDataBlock and pkg.qec_scheme.QECScheme is pkg.QECScheme


@needs_qualtran
@pytest.mark.parametrize("name", sorted(QUALTRAN_SUBMODULES))
def test_qualtran_submodules_against_qualtran(name):
    """The names recorded above are all of Qualtran's own in that submodule, less those not ported."""
    import importlib
    import types

    try:
        theirs = importlib.import_module(f"qualtran.surface_code.{name}")
    except ImportError:  # ui needs dash, plotly and pandas
        pytest.skip(f"qualtran.surface_code.{name} does not import here")
    own = set()
    for attr, value in vars(theirs).items():
        origin = value.__name__ if isinstance(value, types.ModuleType) else getattr(value, "__module__", None)
        if not attr.startswith("__") and (origin or "").startswith("qualtran") or attr in ("_GIDNEY_FOWLER_MODEL", "_BEVERLAND_MODEL", "_SUPPORTED_ESTIMATION_MODELS", "_QEC_SCHEMES", "_MAGIC_FACTORIES", "_ROTATION_MODELS"):
            own.add(attr)
    assert sorted(own - QUALTRAN_NOT_PORTED) == sorted(QUALTRAN_SUBMODULES[name])


QUALTRAN_UI = json.loads(gzip.decompress((pathlib.Path(__file__).parent / "estimate_port" / "qualtran_ui.json.gz").read_bytes()))


def _plain(x):
    if isinstance(x, dict):
        return {k: _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    if isinstance(x, np.ndarray):
        return _plain(x.tolist())
    return x.item() if isinstance(x, np.generic) else x


@pytest.mark.parametrize("model", ["GidneyFowler", "Beverland"])
def test_qualtran_ui_against_recorded(model):
    """Qualtran's `ui.update` (0.7.0, recorded in tests/estimate_port/qualtran_ui.json.gz with numpy's global seed
    set before each call) on 448 inputs: both estimation models, both QEC schemes, the three factories, both
    rotation models, 1 and 4 factories, four algorithms, two error rates and two budgets. Every output equals
    Qualtran's: the pie chart's and the runtime plot's figure dicts (less plotly's template), the styles, the
    magic-state count and its name, the factory count and the duration."""
    from stabilizer_qec.estimate.qualtran import ui

    cases = [c for c in QUALTRAN_UI["cases"] if c["args"][2].startswith(model)]
    assert len(cases) == {"GidneyFowler": 64, "Beverland": 384}[model]
    for case in cases:
        np.random.seed(case["seed"])
        out = list(ui.update(*case["args"]))
        assert isinstance(out[0], ui.Figure) and isinstance(out[2], ui.Figure)
        assert _plain(out) == case["out"], case["args"]


def test_qualtran_ui_helpers():
    from stabilizer_qec.estimate.qualtran import ui

    with pytest.raises(ui.PreventUpdate):
        ui.update(1e-3, None, ui._BEVERLAND_MODEL, [10] * 6, "GidneyFowler", "GidneyFowlerCCZ", 1, "BeverlandEtAlRotationCost")
    with pytest.raises(ui.PreventUpdate):
        ui.update(1e-3, 1e-2, ui._BEVERLAND_MODEL, [10, 10, None, 10, 10, 10], "GidneyFowler", "GidneyFowlerCCZ", 1, "BeverlandEtAlRotationCost")
    assert ui.format_duration([5e11, 1e11]) == ("days", [5e11 / 86400_000_000, 1e11 / 86400_000_000])
    assert ui.format_duration([999.0]) == ("us", [999.0]) and ui.format_duration([2e6])[0] == "s"
    assert sorted(ui._ROTATION_MODELS) == ["BeverlandEtAlRotationCost", "SevenDigitsOfPrecisionConstantCost"]
    assert ui.total_magic(ui._BEVERLAND_MODEL, GateCounts(t=10, toffoli=1)) == (["Total Number of T gates"], "14")
    empty = ui.Figure()
    assert empty == {"data": [], "layout": {}} and empty.data == [] and empty.layout == {}


@needs_qualtran
def test_qualtran_ui_against_qualtran():
    """Qualtran's own ui (where dash, plotly and pandas are installed) on fresh inputs; our figures also make
    plotly figures."""
    try:
        from qualtran.surface_code import ui as their_ui
    except ImportError:
        pytest.skip("Qualtran's ui needs dash, plotly and pandas")
    import base64

    from stabilizer_qec.estimate.qualtran import ui

    def plain_theirs(fig):
        d = fig.to_plotly_json()
        d["layout"].pop("template", None)

        def walk(x):
            if isinstance(x, dict) and {"dtype", "bdata"} <= set(x):
                return np.frombuffer(base64.b64decode(x["bdata"]), dtype=np.dtype(x["dtype"])).tolist()
            if isinstance(x, dict):
                return {k: walk(v) for k, v in x.items()}
            if isinstance(x, (list, tuple)):
                return [walk(v) for v in x]
            return _plain(x)

        return walk(d)

    for seed, args in enumerate([(3e-4, 5e-3, ui._BEVERLAND_MODEL, [250, 77, 10**7, 10**6, 300, 40], "BeverlandEtAl", "FifteenToOne933", 3, "SevenDigitsOfPrecisionConstantCost"), (3e-4, 5e-3, ui._GIDNEY_FOWLER_MODEL, [250, 77, 10**7, 10**6, 300, 40], "BeverlandEtAl", "FifteenToOne933", 2, "BeverlandEtAlRotationCost")]):
        np.random.seed(seed)
        ours = list(ui.update(*args))
        np.random.seed(seed)
        theirs = list(their_ui.update(*args))
        assert _plain(ours[0]) == plain_theirs(theirs[0]) and _plain(ours[2]) == plain_theirs(theirs[2])
        assert _plain(ours[1:2] + ours[3:]) == _plain(theirs[1:2] + theirs[3:])
        assert ours[0].to_plotly().data[0].type == "pie"


def test_qualtran_abstract_classes():
    """As Qualtran's, the data block, factory and rotation-model bases are abstract."""
    from stabilizer_qec.estimate.qualtran import DataBlock, MagicStateFactory, RotationCostModel

    for base in (DataBlock, MagicStateFactory, RotationCostModel):
        with pytest.raises(TypeError, match="abstract"):
            base()

    class Half(MagicStateFactory):
        def n_physical_qubits(self):
            return 1

    with pytest.raises(TypeError, match="abstract"):
        Half()


def test_qualtran_details():
    """Qualtran's reprs (`:g` numbers), the 15-to-1 factory's numpy-float cycles, the private `_qec_scheme`, and
    copies and pickles of the data blocks (whose `data_d` is a slot)."""
    import copy
    import dataclasses
    import pickle

    from stabilizer_qec.estimate.qualtran import CompactDataBlock, PhysicalCostModel, PhysicalParameters, SimpleDataBlock

    assert repr(QECScheme(error_rate_scaler=0.123456789, error_rate_threshold=1e-2)) == "QECScheme(error_rate_scaler=0.123457, error_rate_threshold=0.01)"
    model = LogicalErrorModel(qec_scheme=QECScheme.make_gidney_fowler(), physical_error=1e-3)
    assert repr(model) == "LogicalErrorModel(physical_error=0.001, _qec_scheme=QECScheme(error_rate_scaler=0.1, error_rate_threshold=0.01))"
    assert model.qec_scheme is model._qec_scheme
    assert repr(PhysicalParameters(physical_error=1.23456789e-4, cycle_time_us=0.7)) == "PhysicalParameters(physical_error=0.000123457, cycle_time_us=0.7)"
    assert repr(PhysicalParameters.make_beverland_et_al("ion")) == "PhysicalParameters(physical_error=0.001, cycle_time_us=600.0)"
    assert repr(PhysicalCostModel.make_gidney_fowler(data_d=17)) == "PhysicalCostModel(physical_params=PhysicalParameters(physical_error=0.001, cycle_time_us=1.0), data_block=SimpleDataBlock(data_d=17, routing_overhead=0.5), factory=CCZ2TFactory(distillation_l1_d=15, distillation_l2_d=31), qec_scheme=QECScheme(error_rate_scaler=0.1, error_rate_threshold=0.01))"
    cycles = FifteenToOne(7, 3, 3).n_cycles(GateCounts(t=1000), model)
    assert type(cycles) is np.float64 and cycles == 22338.0
    assert type(CCZ2TFactory().n_cycles(GateCounts(t=1000), model)) is int and type(MultiFactory(FifteenToOne(7, 3, 3), 2).n_cycles(GateCounts(t=1000), model)) is int
    for block in (SimpleDataBlock(data_d=7, routing_overhead=0.25), CompactDataBlock(data_d=9), FastDataBlock(data_d=11)):
        for clone in (copy.copy(block), copy.deepcopy(block), *(pickle.loads(pickle.dumps(block, protocol)) for protocol in range(pickle.HIGHEST_PROTOCOL + 1))):
            assert clone == block and repr(clone) == repr(block) and hash(clone) == hash(block)
        assert dataclasses.replace(block, data_d=13).data_d == 13
        with pytest.raises(dataclasses.FrozenInstanceError):
            block.data_d = 3


def test_qualtran_grid_search_one_shot_iterators():
    """As Qualtran's, the grid search walks the data blocks again for each factory, so one-shot generators of
    blocks are spent on the first factory: here none of its blocks fits, and the search fails as Qualtran's."""
    from stabilizer_qec.estimate.qualtran import iter_simple_data_blocks

    gates = GateCounts(t=10**8, toffoli=10**8)
    with pytest.raises(ValueError, match="^No valid factories found!$"):
        get_ccz2t_costs_from_grid_search(n_logical_gates=gates, n_algo_qubits=100, factory_iter=iter_ccz2t_factories(), data_block_iter=iter_simple_data_blocks())
    with pytest.raises(ValueError, match="^No valid factories found!$"):
        get_ccz2t_costs_from_grid_search(n_logical_gates=gates, n_algo_qubits=100, factory_iter=list(iter_ccz2t_factories()), data_block_iter=iter_simple_data_blocks())
    cost, factory, block = get_ccz2t_costs_from_grid_search(n_logical_gates=gates, n_algo_qubits=100, factory_iter=iter_ccz2t_factories(), data_block_iter=list(iter_simple_data_blocks()))
    assert (factory.distillation_l1_d, factory.distillation_l2_d, block.data_d, cost.footprint) == (15, 25, 29, 378700)


@needs_qualtran
def test_qualtran_details_against_qualtran():
    """Reprs, cycle types, one-shot grid searches, and our models on Qualtran's own `GateCounts` and
    `AlgorithmSummary` against Qualtran."""
    from stabilizer_qec.estimate import qualtran as ours

    def build(m, counts):
        lem = m.LogicalErrorModel(qec_scheme=m.QECScheme.make_gidney_fowler(), physical_error=1e-3)
        objs = [m.QECScheme.make_beverland_et_al(), m.QECScheme(error_rate_scaler=0.123456789, error_rate_threshold=1e-2), lem, m.PhysicalParameters.make_beverland_et_al("majorana", True), m.SimpleDataBlock(data_d=7), m.CompactDataBlock(data_d=9), m.IntermediateDataBlock(data_d=11), m.FastDataBlock(data_d=13), m.CCZ2TFactory(), m.FifteenToOne(7, 3, 3), m.MultiFactory(m.FifteenToOne(9, 3, 3), 2), m.BeverlandEtAlRotationCost, m.SevenDigitsOfPrecisionConstantCost, m.PhysicalCostModel.make_beverland_et_al(data_d=17, data_block_name="fast"), m.PhysicalCostsSummary(failure_prob=0.1, footprint=10, duration_hr=1.5), m.AlgorithmSummary(n_algo_qubits=10, n_logical_gates=counts(t=5, toffoli=3))]
        return objs, lem

    mine, my_lem = build(ours, GateCounts)
    theirs, their_lem = build(their_sc, TheirGateCounts)
    for a, b in zip(mine, theirs):
        assert repr(a) == repr(b)
        if isinstance(b, (their_sc.MagicStateFactory, their_sc.DataBlock)):
            x, y = a.n_cycles(GateCounts(t=10**6, toffoli=10**5), my_lem), b.n_cycles(TheirGateCounts(t=10**6, toffoli=10**5), their_lem)
            assert type(x) is type(y) and x == y
    for kind in ("generators", "a list of factories", "a list of blocks"):
        def args(m, counts):
            factories, blocks = m.iter_ccz2t_factories(), m.iter_simple_data_blocks()
            return dict(n_logical_gates=counts(t=10**8, toffoli=10**8), n_algo_qubits=100, factory_iter=list(factories) if kind == "a list of factories" else factories, data_block_iter=list(blocks) if kind == "a list of blocks" else blocks)

        try:
            expected = their_sc.get_ccz2t_costs_from_grid_search(**args(their_sc, TheirGateCounts))
        except ValueError as ex:
            with pytest.raises(ValueError, match=f"^{ex}$"):
                ours.get_ccz2t_costs_from_grid_search(**args(ours, GateCounts))
            continue
        got = ours.get_ccz2t_costs_from_grid_search(**args(ours, GateCounts))
        assert repr(got) == repr(expected), kind
    their_alg = their_sc.AlgorithmSummary(n_algo_qubits=100, n_logical_gates=TheirGateCounts(t=10**4, toffoli=10, rotation=300, measurement=7), n_rotation_layers=20)
    my_alg = AlgorithmSummary(n_algo_qubits=100, n_logical_gates=TheirGateCounts(t=10**4, toffoli=10, rotation=300, measurement=7), n_rotation_layers=20)
    from qualtran.surface_code import beverland_et_al_model as their_bev

    for alg in (their_alg, my_alg):
        assert est.t_states(error_budget=1e-3, alg=alg, rotation_model=BeverlandEtAlRotationCost) == their_bev.t_states(error_budget=1e-3, alg=their_alg, rotation_model=their_sc.BeverlandEtAlRotationCost) == 14840
        assert est.minimum_time_steps(error_budget=1e-3, alg=alg, rotation_model=BeverlandEtAlRotationCost) == their_bev.minimum_time_steps(error_budget=1e-3, alg=their_alg, rotation_model=their_sc.BeverlandEtAlRotationCost)
    assert GateCounts(t=1) + TheirGateCounts(toffoli=2) == GateCounts(t=1, toffoli=2)


def test_qualtran_duck_typed_counts():
    """Counts that are not this module's `GateCounts` (Qualtran's own, say) are read by their seven fields."""
    from types import SimpleNamespace

    theirs = SimpleNamespace(t=10**4, toffoli=10, cswap=0, and_bloq=0, clifford=0, rotation=300, measurement=7)
    alg = AlgorithmSummary(n_algo_qubits=100, n_logical_gates=theirs, n_rotation_layers=20)
    assert est.t_states(error_budget=1e-3, alg=alg, rotation_model=BeverlandEtAlRotationCost) == 14840
    assert est.n_discrete_logical_gates(eps_syn=1e-3 / 3, alg=alg, rotation_model=BeverlandEtAlRotationCost) == GateCounts(t=14800, toffoli=10, measurement=7)
    assert GateCounts(t=1) + SimpleNamespace(t=1, toffoli=2, cswap=0, and_bloq=0, clifford=0, rotation=0, measurement=0) == GateCounts(t=2, toffoli=2)
    with pytest.raises(TypeError, match="Can only add other `GateCounts` objects"):
        GateCounts(t=1) + 5


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
        with pytest.raises(az.EstimatorError) as e:
            _ours(case["jobParams"], case["logicalCounts"])
        assert str(e.value) == case["message"]
        return
    ours = _ours(case["jobParams"], case["logicalCounts"])
    if "frontierEntries" in case:
        assert len(ours["frontierEntries"]) == len(case["frontierEntries"])
        for a, b in zip(ours["frontierEntries"], case["frontierEntries"]):
            assert {k: v for k, v in a.items() if k != "physicalCountsFormatted"} == b  # the counts in words: test_azure_whole_result_against_qdk_recorded
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
        with pytest.raises(az.EstimatorError) as e:
            _ours(case["jobParams"], case["logicalCounts"])
        assert "EstimatorError: " + str(e.value).replace("\n", " ") == case["message"]  # recorded as `{type}: {str}` on one line
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
    assert json.dumps(ours) == json.dumps(theirs)  # the whole result: the job parameters, the formatted counts, the report


QDK_REPORTS = json.loads(gzip.decompress((pathlib.Path(__file__).parent / "estimate_port" / "qdk_reports.json.gz").read_bytes()))
RECORDED = {"qdk_estimates": QDK_CASES, "qdk_review2": QDK_REVIEW2}


@pytest.mark.parametrize("fixture,key", [(f, k) for f in sorted(QDK_REPORTS) for k in sorted(QDK_REPORTS[f])])
def test_azure_whole_result_against_qdk_recorded(fixture, key):
    """The QDK's whole result (qdk 1.33.1, recorded in tests/estimate_port/qdk_reports.json.gz) for every case of
    both fixtures it solves: the status, the job parameters it echoes, the counts, the counts in words
    (`physicalCountsFormatted`) and the report (`reportData`: the groups' labels, descriptions and explanations
    with their numbers written in, and the assumptions), and a frontier's entries with their counts in words.
    Ours is equal as data and as JSON text, the key order included."""
    case = RECORDED[fixture][key]
    theirs = QDK_REPORTS[fixture][key]
    ours = _ours(case["jobParams"], case["logicalCounts"])
    for section in theirs:
        assert ours[section] == theirs[section], section
    assert json.dumps(ours) == json.dumps(theirs)


def test_azure_report_fixture_covers_every_solved_case():
    for fixture, cases in RECORDED.items():
        assert sorted(QDK_REPORTS[fixture]) == sorted(key for key, case in cases.items() if case["status"] == "success")
    assert sum(len(cases) for cases in QDK_REPORTS.values()) == 97


def test_azure_report_of_the_qdks_own_test():
    """The QDK's own report test (resource_estimator/src/system/tests.rs, `test_report`, against test_report.json):
    100 qubits, 112,110 rotations in 2,001 layers, the default parameters. Its counts in words and its first entry."""
    r = azure_estimate({"numQubits": 100, "rotationCount": 112110, "rotationDepth": 2001})
    f = r["physicalCountsFormatted"]
    assert {k: f[k] for k in ("runtime", "rqops", "physicalQubits", "algorithmicLogicalQubits", "logicalDepth", "numTstates", "numTfactories", "physicalQubitsForTfactoriesPercentage", "requiredLogicalQubitErrorRate", "clockFrequency", "tfactoryRuntime", "numTsPerRotation", "logicalCycleTime", "errorBudget")} == {
        "runtime": "1 secs", "rqops": "30.26M", "physicalQubits": "7.93M", "algorithmicLogicalQubits": "230", "logicalDepth": "152.13k", "numTstates": "2.24M", "numTfactories": "233", "physicalQubitsForTfactoriesPercentage": "97.91 %",
        "requiredLogicalQubitErrorRate": "9.53e-12", "clockFrequency": "131.58k", "tfactoryRuntime": "120 microsecs", "numTsPerRotation": "20", "logicalCycleTime": "8 microsecs", "errorBudget": "1.00e-3"}
    assert len(f) == 43 and list(r) == ["status", "jobParams", "physicalCounts", "physicalCountsFormatted", "logicalQubit", "tfactory", "errorBudget", "logicalCounts", "reportData"]
    groups = r["reportData"]["groups"]
    assert [g["title"] for g in groups] == ["Physical resource estimates", "Resource estimates breakdown", "Logical qubit parameters", "T factory parameters", "Pre-layout logical resources", "Assumed error budget", "Physical qubit parameters", "Constraints"]
    assert groups[0]["alwaysVisible"] and len(r["reportData"]["assumptions"]) == 7
    assert groups[0]["entries"][0] == {"path": "physicalCountsFormatted/runtime", "label": "Runtime", "description": "Total runtime", "explanation": "This is a runtime estimate for the execution time of the algorithm.  In general, the execution time corresponds to the duration of one logical cycle (7,600 nanosecs) multiplied by the 152,130 logical cycles to run the algorithm.  If however the duration of a single T factory (here: 120,000 nanosecs) is larger than the algorithm runtime, we extend the number of logical cycles artificially in order to exceed the runtime of a single T factory."}
    frontier = azure_estimate({"numQubits": 100, "rotationCount": 112110, "rotationDepth": 2001}, estimate_type="frontier")
    assert list(frontier) == ["status", "jobParams", "logicalCounts", "reportData", "frontierEntries"] and all(list(e)[-1] == "physicalCountsFormatted" for e in frontier["frontierEntries"])
    assert frontier["reportData"]["groups"][0]["entries"][0]["explanation"] == groups[0]["entries"][0]["explanation"]  # the report of the first (shortest) entry, here the single-point estimate


def test_azure_number_and_duration_formats():
    """The QDK's formatting (system/data/report.rs). A duration is in the largest unit it reaches, rounded half
    up by the remainder in the unit below; under a microsecond the QDK's rounding test (``rem >= 1 / 2``) always
    passes, so the nanoseconds read one high, as here. Past years it does not round. A count is bare below 1000,
    else to two decimals with a metric prefix (a u64 stops at E). Floats print as Rust prints them: ``{}`` with
    the shortest digits and no exponent, ``{:.2e}`` with a bare exponent, ``{:.2}``; NaN as ``NaN``."""
    durations = [0, 1, 499, 500, 999, 1000, 1499, 1500, 3300, 999_499, 999_500, 59_499_999_999, 59_500_000_000, 1_064_000_000, 86_400 * 10**9 * 365 * 3, 2**64 - 1]
    assert [format_duration(n) for n in durations] == ["1 nanosecs", "2 nanosecs", "500 nanosecs", "501 nanosecs", "1000 nanosecs", "1 microsecs", "1 microsecs", "2 microsecs", "3 microsecs", "999 microsecs", "1000 microsecs", "59 secs", "60 secs", "1 secs", "3 years", "584 years"]
    assert [format_metric_prefix(n) for n in (0, 999, 1000, 400_060, 999_994, 999_995, 38_330_000, 2**64 - 1)] == ["0", "999", "1.00k", "400.06k", "999.99k", "1000.00k", "38.33M", "18.45E"]
    assert [az._display(x) for x in (1.0, 1e-10, 0.1 + 0.2, 1e21, 12345.678, -0.0, float("nan"), float("inf"))] == ["1", "0.0000000001", "0.30000000000000004", "1000000000000000000000", "12345.678", "-0", "NaN", "inf"]
    assert [az._exp2(x) for x in (1.55e-11, 0.0, 2e5, 9.999e-3, float("nan"))] == ["1.55e-11", "0.00e0", "2.00e5", "1.00e-2", "NaN"]
    assert [az._fixed2(x) for x in (0.125, 0.135, 12.0, float("nan"))] == ["0.12", "0.14", "12.00", "NaN"]
    assert (az._thousands(1234567), az._thousands(123), az._thousands(1000), az._thousands_f64(1234567.891)) == ("1,234,567", "123", "1,000", "1,234,567.89")


def test_azure_estimator_result():
    """The QDK's `LogicalCounts.estimate` and `EstimatorResult` as data: one estimate is its result dict, a
    frontier's missing keys come from its first entry, a batch is indexed by item with a failed item holding its
    message and error, and the summary is the report's groups as a text table."""
    counts = LogicalCounts(num_qubits=10, t_count=100, rotation_count=5, rotation_depth=3)
    one = counts.estimate({"qubitParams": {"name": "qubit_gate_ns_e4"}})
    assert isinstance(one, EstimatorResult) and dict(one) == one.data() == azure_estimate(counts, "qubit_gate_ns_e4") and json.loads(one.json) == one.data()
    assert one.error is None and one.logical_counts == LogicalCounts.fromdict(one["logicalCounts"]) and counts.estimate() == azure_estimate(counts)
    lines = one.summary.splitlines()
    assert lines[0] == "Physical resource estimates" and lines[1].split()[:3] == ["Runtime", one["physicalCountsFormatted"]["runtime"].split()[0], one["physicalCountsFormatted"]["runtime"].split()[1]]
    assert sum(len(g["entries"]) for g in one["reportData"]["groups"]) + len(one["reportData"]["groups"]) == len(lines)
    with pytest.raises(ValueError, match="slice"):
        _ = one[0:1]
    with pytest.raises(ValueError, match="idx"):
        one.data(0)
    with pytest.raises(az.EstimatorError, match="MaxDurationTooSmall") as e:
        counts.estimate({"constraints": {"maxDuration": "1 ns"}})
    assert e.value.code == "Qdk.Qsc.Estimates.MaxDurationTooSmall" and str(e.value) == "Error estimating resources (Qdk.Qsc.Estimates.MaxDurationTooSmall):\nNo solution found for the provided maximum duration."
    frontier = counts.estimate({"estimateType": "frontier"})
    assert "physicalCounts" not in dict(frontier) and frontier["physicalCounts"] == frontier["frontierEntries"][0]["physicalCounts"] and frontier["logicalQubit"] == frontier["frontierEntries"][0]["logicalQubit"]
    assert frontier.summary.splitlines()[1].split()[1:3] == frontier["frontierEntries"][0]["physicalCountsFormatted"]["runtime"].split()
    with pytest.raises(KeyError):
        _ = one["frontierEntries"]
    batch = counts.estimate([{"qubitParams": {"name": "qubit_gate_ns_e3"}}, {"constraints": {"maxDuration": "1 ns"}}, {"qubitParams": {"name": "qubit_maj_ns_e4"}, "qecScheme": {"name": "floquet_code"}, "unknownKey": 1}, {"estimateType": "frontier"}])
    assert list(batch) == [0, 1, 2, 3] and batch.data(0) == azure_estimate(counts, "qubit_gate_ns_e3") and batch.data(2) == azure_estimate(counts, "qubit_maj_ns_e4", "floquet_code")
    assert batch[1].data() == {"code": "Qdk.Qsc.Estimates.MaxDurationTooSmall", "message": "No solution found for the provided maximum duration."} and isinstance(batch[1].error, az.EstimatorError) and batch[0].error is None
    assert json.loads(batch.json) == [batch.data(i) for i in range(4)] and batch.logical_counts == one.logical_counts
    table = batch.summary.splitlines()
    assert table[0].split() == ["Item", "0", "1", "2", "3"] and table[1] == "Physical resource estimates" and table[2].split()[:2] == ["Runtime", batch[0]["physicalCountsFormatted"]["runtime"].split()[0]] and "N/A" in table[2]
    assert batch[0:2].splitlines()[0].split() == ["Item", "0", "1"] and batch[:] == batch.summary
    big = counts.estimate([{}] * 7)
    assert big.summary.startswith("(the first 5 of 7 items") and big[:].splitlines()[0].split() == ["Item"] + [str(i) for i in range(7)]
    assert counts.estimate([{"constraints": {"maxDuration": "1 ns"}}] * 2).summary == "None of the jobs succeeded"
    with pytest.raises(az.QSharpError, match=r"^cannot parse JSON: 'invalid type: string \"qubit_gate_ns_e3\", expected struct JobParams at line 1 column 19'$"):
        counts.estimate(["qubit_gate_ns_e3"])


def test_azure_estimator_result_data_frame():
    pytest.importorskip("pandas")
    counts = LogicalCounts(num_qubits=10, t_count=100, rotation_count=5, rotation_depth=3)
    batch = counts.estimate([{"qubitParams": {"name": "qubit_gate_ns_e3"}}, {"constraints": {"maxDuration": "1 ns"}}, {"estimateType": "frontier"}])
    frame = batch.summary_data_frame(labels=["a", "b"])
    assert list(frame.index) == ["a", "b", 2] and list(frame.columns) == ["Logical qubits", "Logical depth", "T states", "Code distance", "T factories", "T factory fraction", "Physical qubits", "rQOPS", "Physical runtime"]
    f = batch[0]["physicalCountsFormatted"]
    assert list(frame.loc["a"]) == [f["algorithmicLogicalQubits"], f["logicalDepth"], f["numTstates"], batch[0]["logicalQubit"]["codeDistance"], f["numTfactories"], f["physicalQubitsForTfactoriesPercentage"], f["physicalQubits"], f["rqops"], f["runtime"]]
    assert list(frame.loc["b"]) == ["No solution found"] * 9 and frame.loc[2, "Physical runtime"] == batch[2]["frontierEntries"][0]["physicalCountsFormatted"]["runtime"]


@needs_qdk
def test_azure_estimator_result_against_qdk_live():
    from qdk.estimator import EstimatorParams

    counts = {"numQubits": 10, "tCount": 100, "rotationCount": 5, "rotationDepth": 3}
    batch = [{"qubitParams": {"name": "qubit_gate_ns_e3"}}, {"constraints": {"maxDuration": "1 ns"}}, {"qubitParams": {"name": "qubit_maj_ns_e4"}, "qecScheme": {"name": "floquet_code"}}]
    theirs, ours = QdkLogicalCounts(counts).estimate(batch), LogicalCounts.fromdict(counts).estimate(batch)
    assert list(theirs) == list(ours) and theirs.data(0) == ours.data(0) and theirs.data(2) == ours.data(2) and theirs[1].error is not None and ours[1].error is not None
    single = {"qubitParams": {"name": "qubit_gate_ns_e4"}, "estimateType": "frontier"}
    assert QdkLogicalCounts(counts).estimate(single).json == LogicalCounts.fromdict(counts).estimate(single).json
    assert QdkLogicalCounts(counts).estimate(single)["logicalQubit"] == LogicalCounts.fromdict(counts).estimate(single)["logicalQubit"]
    params = EstimatorParams(num_items=2)
    params.items[0].qubit_params.name = "qubit_gate_ns_e4"
    params.items[1].qubit_params.name = "qubit_maj_ns_e6"
    params.items[1].qec_scheme.name = "floquet_code"
    a, b = QdkLogicalCounts(counts).estimate(params), LogicalCounts.fromdict(counts).estimate(params)
    assert a.json == b.json
    params = EstimatorParams()
    params.error_budget = 0.01
    params.constraints.max_t_factories = 3
    assert QdkLogicalCounts(counts).estimate(params).json == LogicalCounts.fromdict(counts).estimate(params).json
    try:
        import pandas  # noqa: F401
    except ImportError:  # pragma: no cover
        return
    assert theirs.summary_data_frame(labels=["a"]).equals(ours.summary_data_frame(labels=["a"]))


QDK_ERRORS = json.loads((pathlib.Path(__file__).parent / "estimate_port" / "qdk_errors.json").read_text())["cases"]


def _outcome(counts: dict, params) -> dict:
    """What `LogicalCounts(counts).estimate(params)` gives, as tests/estimate_port/qdk_errors.json records the
    QDK's (the counts are checked when the `LogicalCounts` is made here, when the estimate starts in the QDK)."""
    try:
        r = LogicalCounts(counts).estimate(params)
    except (az.EstimatorError, az.QSharpError) as e:
        return {"ok": False, "cls": type(e).__name__, "code": getattr(e, "code", None), "str": str(e)}
    return _result_outcome(r)


def _result_outcome(r) -> dict:
    d = r.data()
    if isinstance(d, list):
        return {"ok": True, "batch": [{"status": x.get("status"), "code": x.get("code"), "message": x.get("message"), "jobParams": x.get("jobParams")} for x in d], "errors": [str(r[i].error) if r[i].error else None for i in range(len(d))]}
    return {"ok": True, "status": d.get("status"), "jobParams": d.get("jobParams"), "logicalCounts": d.get("logicalCounts")}


@pytest.mark.parametrize("index", range(len(QDK_ERRORS)), ids=[c["name"] for c in QDK_ERRORS])
def test_azure_errors_against_qdk_recorded(index):
    """The QDK's reading of its input and its refusals (qdk 1.33.1, recorded in tests/estimate_port/qdk_errors.json):
    parameters it cannot read raise its `QSharpError` with serde's reason and the column in the JSON text it is
    sent (``cannot parse JSON: 'invalid type: null, expected a map at line 1 column 21'``), a refused estimate its
    `EstimatorError` with its code and message, a batch holds each failed item's ``code`` and ``message``, and an
    estimate it accepts echoes the parameters as it does (the scheme's name as given, a custom scheme's absent
    power as null). Every case: the class, the code and the text are the QDK's."""
    case = QDK_ERRORS[index]
    assert _outcome(case["counts"], case["params"]) == case["qdk"]


LIVE_ERRORS = [i for i in range(len(QDK_ERRORS)) if i % 6 == 0]


@needs_qdk
@pytest.mark.parametrize("index", LIVE_ERRORS, ids=[QDK_ERRORS[i]["name"] for i in LIVE_ERRORS])
def test_azure_errors_against_qdk_live(index):
    case = QDK_ERRORS[index]
    try:
        theirs = _result_outcome(QdkLogicalCounts(case["counts"]).estimate(case["params"]))
    except BaseException as e:  # the QDK's EstimatorError derives from BaseException
        theirs = {"ok": False, "cls": type(e).__name__, "code": getattr(e, "code", None), "str": str(e)}
    assert _outcome(case["counts"], case["params"]) == theirs


def test_azure_errors():
    """`EstimatorError` is a ValueError with the QDK's code and message (and pickles); a single estimate that fails
    raises it, a batch keeps it per item; the building blocks raise the QDK's errors too, a dict they cannot read
    with serde's reason alone (a position means something only in the whole call's JSON text)."""
    import pickle

    counts = LogicalCounts(num_qubits=10, t_count=100)
    with pytest.raises(az.EstimatorError) as e:
        azure_estimate(counts, error_budget=1.5)
    err = e.value
    assert isinstance(err, ValueError) and err.code == "Qdk.Qsc.Estimates.InvalidInputError.InvalidErrorBudget" and err.reason == "The error budget must be between 0.0 and 1.0, provided input was `1.5`"
    assert str(err) == err.message == "Error estimating resources (Qdk.Qsc.Estimates.InvalidInputError.InvalidErrorBudget):\nThe error budget must be between 0.0 and 1.0, provided input was `1.5`"
    again = pickle.loads(pickle.dumps(err))
    assert (type(again), again.code, str(again)) == (az.EstimatorError, err.code, str(err))
    with pytest.raises(az.QSharpError, match=r"^cannot parse JSON: 'invalid type: null, expected a map at line 1 column 44'$"):
        counts.estimate([{"errorBudget": 0.01}, {"qubitParams": None}])
    with pytest.raises(az.QSharpError, match=r"^cannot parse JSON: 'missing field `name or instructionSet`'$"):
        physical_qubit({})
    with pytest.raises(az.EstimatorError, match="missing field `errorCorrectionThreshold`") as e:
        qec_scheme({}, physical_qubit())
    assert e.value.code == "Qdk.Qsc.Estimates.IOError.CannotParseJSON"
    with pytest.raises(az.EstimatorError) as e:
        qec_scheme({"name": "surface_code", "logicalCycleTime": "codeDistance - 4"}, physical_qubit())
    assert str(e.value) == "Error estimating resources (Qdk.Qsc.Estimates.LogicalCycleTimeComputationFailed):\nThe logical cycle time cannot be computed: logicalCycleTime formula yields non-positive value for code distance = 3"
    with pytest.raises(az.QSharpError, match=r"^cannot parse JSON: 'unknown field `bar`, expected one of `logicalDepthFactor`, `maxTFactories`, `maxDuration`, `maxPhysicalQubits`, `maxDistillationRounds`'$"):
        Constraints.fromspec({"bar": 1})
    with pytest.raises(az.EstimatorError, match="Invalid distillation unit specification name: foo.") as e:
        DistillationUnitTemplate.fromspec("foo")
    assert e.value.code == "Qdk.Qsc.Estimates.IOError.CannotParseJSON"
    with pytest.raises(az.EstimatorError, match="provided input was `-0`"):
        ErrorBudget.partition({"logical": 0.001, "tStates": -0.0, "rotations": 0.0}, counts)
    batch = counts.estimate([{"errorBudget": 0.01}, {"errorBudget": 0.0}])
    assert batch.data(1) == {"code": "Qdk.Qsc.Estimates.InvalidInputError.InvalidErrorBudget", "message": "The error budget must be between 0.0 and 1.0, provided input was `0`"} and batch[1].error.code == batch.data(1)["code"]
    with pytest.raises(az.EstimatorError):
        EstimatorResult([{"code": "Qdk.Qsc.Estimates.MaxDurationTooSmall", "message": "No solution found for the provided maximum duration."}])
    assert counts.estimate([]).data() == []


def test_azure_counts_as_the_qdk_takes_them():
    """`LogicalCounts` takes the QDK's dict as the QDK's `LogicalCounts` does: a missing count is 0, an unknown
    key is dropped (the snake_case names too, though `fromdict` reads them), the memory counts stay absent; and
    each count must be a u64, as the QDK reads it."""
    c = LogicalCounts({"numQubits": 5, "tCount": 7, "foo": 1, "t_count": 9})
    assert c == LogicalCounts(num_qubits=5, t_count=7) and isinstance(c.num_qubits, int) and c.num_compute_qubits is None
    assert LogicalCounts({"tCount": 3}).num_qubits == 0 and LogicalCounts.fromdict({"num_qubits": 4, "t_count": 2}) == LogicalCounts(4, 2)
    assert LogicalCounts({"numQubits": 5, "numComputeQubits": None}).asdict() == LogicalCounts(5).asdict()
    with pytest.raises(az.QSharpError, match=r"^cannot parse JSON: 'invalid value: integer `-1`, expected u64 at line 1 column 16'$"):
        LogicalCounts(num_qubits=-1)
    with pytest.raises(az.QSharpError, match="invalid type: boolean `true`, expected u64"):
        LogicalCounts({"numQubits": 5, "readFromMemoryCount": True})


def test_azure_estimate_returns_the_qdks_result():
    """`azure_estimate` gives the QDK's `EstimatorResult`: a frontier's sections come from its first entry, as
    `result["physicalCounts"]` does in the QDK; objects (a `QECScheme`, an `ErrorBudget`, `Constraints`, unit
    templates, a qubit model) stand in the QDK's parameters."""
    counts = LogicalCounts(num_qubits=10, t_count=100, rotation_count=5, rotation_depth=3)
    frontier = azure_estimate(counts, estimate_type="frontier")
    assert isinstance(frontier, EstimatorResult) and frontier["physicalCounts"] == frontier["frontierEntries"][0]["physicalCounts"]
    gate = physical_qubit("qubit_gate_ns_e4")
    a = azure_estimate(counts, gate, qec_scheme("surface_code", gate), ErrorBudget(5e-4, 3e-4, 2e-4), Constraints(max_t_factories=4), [DistillationUnitTemplate.fifteen_to_one_rm_prep(), "15-1 space-efficient"])
    b = azure_estimate(counts, "qubit_gate_ns_e4", "surface_code", {"logical": 5e-4, "tStates": 3e-4, "rotations": 2e-4}, {"maxTFactories": 4}, ["15-to-1 RM prep", "15-1 space-efficient"])
    for section in SECTIONS:
        assert a[section] == b[section], section
    assert a["jobParams"]["qecScheme"] == b["jobParams"]["qecScheme"] and a["jobParams"]["qubitParams"] == b["jobParams"]["qubitParams"]
    custom = azure_estimate(counts, qec_scheme={"errorCorrectionThreshold": 0.01, "crossingPrefactor": 0.03, "logicalCycleTime": "codeDistance", "physicalQubitsPerLogicalQubit": "2 * codeDistance"})
    assert custom["jobParams"]["qecScheme"]["name"] == "" and custom["jobParams"]["qecScheme"]["distanceCoefficientPower"] is None
    assert azure_estimate(counts, qec_scheme="surfaceCode")["jobParams"]["qecScheme"]["name"] == "surfaceCode"


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
    with pytest.raises(az.EstimatorError, match=r"""cannot evaluate expression: 'Undefined\("d"\)'""") as e:
        ev("2 * d", d)
    assert e.value.code == "Qdk.Qsc.Estimates.EvaluationError.CannotEvaluateExpression"
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
        with pytest.raises(az.QSharpError, match="expected value between 0.0 and 1.0|expected f64"):
            physical_qubit({**base, **bad})
    with pytest.raises(az.QSharpError, match="untagged enum MeasurementErrorRate"):
        physical_qubit({"name": "qubit_maj_ns_e4", "oneQubitMeasurementErrorRate": {"process": 2e-4}})
    for bad in ({"tCount": -5}, {"tCount": 5.0}, {"tCount": True}, {"numQubits": "10"}):
        with pytest.raises(az.QSharpError, match="expected u64"):
            LogicalCounts.fromdict({"numQubits": 10, "tCount": 5, **bad})
    with pytest.raises(az.QSharpError, match="untagged enum ErrorBudgetSpecification"):
        ErrorBudget.partition({"logical": 0.1}, LogicalCounts(num_qubits=10, t_count=5))
    assert ErrorBudget.partition({"logical": 0.1, "tstates": 0.05, "rotations": 0.0}, LogicalCounts(num_qubits=10, t_count=5)) == ErrorBudget(0.1, 0.05, 0.0)
    gate = physical_qubit("qubit_gate_ns_e3")
    with pytest.raises(az.QSharpError, match="invalid value: integer `-1`, expected u32"):
        qec_scheme({"name": "surface_code", "distanceCoefficientPower": -1}, gate)
    with pytest.raises(az.QSharpError, match="invalid type: floating point `1.5`, expected u32"):
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
    with pytest.raises(az.QSharpError, match="invalid value: integer `4294967296`, expected u32"):
        qec_scheme({"name": "surface_code", "distanceCoefficientPower": 2**32}, gate)
    scheme = qec_scheme({"name": "surface_code", "distanceCoefficientPower": 2**32 - 1}, gate)
    assert scheme.logical_error_rate(gate, 3) == 0.03 * float(pow(3, 2**32 - 1, 2**64)) * az._powi(0.1, 2)
    assert qec_scheme({"name": "surface_code", "distanceCoefficientPower": 12}, gate).logical_error_rate(gate, 49) == 0.03 * float(49**12 % 2**64) * az._powi(0.1, 25)
    for key in ("maxCodeDistance",):
        with pytest.raises(az.QSharpError, match="floating point `1.8446744073709552e\\+19`, expected u64"):
            qec_scheme({"name": "surface_code", key: 2**64}, gate)
    with pytest.raises(az.QSharpError, match="floating point `1.8446744073709552e\\+19`, expected u64"):
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
        with pytest.raises(az.QSharpError, match="expected a string"):
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
        with pytest.raises(az.QSharpError, match="expected u64"):
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


def test_qualtran_noisy_pauli_rotation():
    """Qualtran's own case (surface_code/t_factory_utils_test.py): the Kraus operators √p (cos θ + i sin θ P) of
    ZIZZ for θ = π/8 × (1, 5, −1, 3) with probabilities 0.4, 0.1, 0.2, 0.3; the channel is their sum."""
    probabilities = [0.4, 0.1, 0.2, 0.3]
    angles = [t * np.pi / 8 for t in (1, 5, -1, 3)]
    i, y, z = np.eye(2), np.array([[0, -1j], [1j, 0]]), np.diag([1.0, -1.0])
    u = np.kron(np.kron(np.kron(z, i), z), z)
    desired = [np.sqrt(p) * (np.cos(t) * np.eye(16) + 1j * np.sin(t) * u) for t, p in zip(angles, probabilities)]
    channel = NoisyPauliRotation("ZIZZ", *probabilities[1:])
    np.testing.assert_allclose(channel.kraus(), desired)
    assert channel.num_qubits() == 4 and [p for p, _ in channel.mixture()] == pytest.approx(probabilities)
    rng = np.random.default_rng(5)
    a = rng.normal(size=(16, 16)) + 1j * rng.normal(size=(16, 16))
    rho = a @ a.conj().T
    rho /= np.trace(rho)
    np.testing.assert_allclose(channel.apply(rho), sum(k @ rho @ k.conj().T for k in desired), atol=1e-15)
    np.testing.assert_allclose(NoisyPauliRotation("XY", 0, 0, 0).kraus()[0], np.cos(np.pi / 8) * np.eye(4) + 1j * np.sin(np.pi / 8) * np.kron(np.array([[0, 1], [1, 0]]), y))
    for bad in (("ZZ", 0.5, 0.4, 0.2), ("ZQ", 0.1, 0.1, 0.1), ("", 0.0, 0.0, 0.0), ("Z", -0.1, 0.0, 0.0)):
        with pytest.raises(ValueError):
            NoisyPauliRotation(*bad)


def test_qualtran_storage_error():
    """Qualtran's own cases (surface_code/t_factory_utils_test.py): a Z or X flip on each qubit with its
    probability, and Y refused."""
    assert storage_error("Z", [0.1, 0.3]) == [PauliFlipChannel("Z", 0.1, 0), PauliFlipChannel("Z", 0.3, 1)]
    assert storage_error("X", [0.2, 0.7], qubits=[3, 5]) == [PauliFlipChannel("X", 0.2, 3), PauliFlipChannel("X", 0.7, 5)]
    with pytest.raises(ValueError, match="Z or X"):
        storage_error("Y", [0.1, 0.3])
    with pytest.raises(ValueError):
        storage_error("Z", [0.1, 0.3], qubits=[0])
    (p0, k0), (p1, k1) = PauliFlipChannel("Z", 0.1, 0).mixture()
    assert (p0, p1) == pytest.approx((0.9, 0.1)) and np.array_equal(k0, np.eye(2)) and np.array_equal(k1, np.diag([1, -1]))
    rho = np.full((4, 4), 0.25, dtype=complex)  # |++⟩
    e = np.kron(np.eye(2), np.diag([1, -1]))
    np.testing.assert_allclose(PauliFlipChannel("Z", 0.25, 1).apply(rho), 0.75 * rho + 0.25 * e @ rho @ e)


@needs_qualtran
def test_qualtran_channels_against_cirq():
    import cirq
    from qualtran.surface_code.t_factory_utils import NoisyPauliRotation as TheirRotation, storage_error as their_storage_error

    rotation, their_rotation = NoisyPauliRotation("ZIZZ", 0.1, 0.2, 0.3), TheirRotation("ZIZZ", 0.1, 0.2, 0.3)
    assert all(np.array_equal(a, b) for a, b in zip(rotation.kraus(), cirq.kraus(their_rotation)))
    assert all(p == q and np.array_equal(a, b) for (p, a), (q, b) in zip(rotation.mixture(), cirq.mixture(their_rotation)))
    rng = np.random.default_rng(7)
    a = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    rho = a @ a.conj().T / np.trace(a @ a.conj().T)
    for kind in ("Z", "X"):
        mine = rho
        for channel in storage_error(kind, [0.2, 0.7]):
            mine = channel.apply(mine)
        theirs = rho
        for qubit, op in enumerate(their_storage_error(kind, [0.2, 0.7], cirq.LineQubit.range(2))):
            ks = [np.kron(k, np.eye(2)) if qubit == 0 else np.kron(np.eye(2), k) for k in cirq.kraus(op)]
            theirs = sum(k @ theirs @ k.conj().T for k in ks)
        np.testing.assert_allclose(mine, theirs, atol=1e-15)


def test_qualtran_legacy_t_complexity():
    """`GateCounts.to_legacy_t_complexity`: a Toffoli and an And are 4 T gates, a CSWAP 7, an And also 9 Cliffords
    (Qualtran's t_complexity of `And()`), a CSWAP 10, and measurements count as Cliffords."""
    g = GateCounts(t=5, toffoli=3, cswap=2, and_bloq=7, clifford=11, rotation=13, measurement=17)
    assert g.to_legacy_t_complexity() == TComplexity(t=59, clifford=111, rotations=13) and g.to_legacy_t_complexity().asdict() == {"t": 59, "rotations": 13, "clifford": 111}
    assert GateCounts(and_bloq=1).to_legacy_t_complexity() == TComplexity(t=4, clifford=9)
    assert g.to_legacy_t_complexity(ts_per_toffoli=1, ts_per_cswap=1, ts_per_and_bloq=1, cliffords_per_and_bloq=0, cliffords_per_cswap=0) == TComplexity(t=17, clifford=28, rotations=13)
    assert TComplexity.rotation_cost(1e-5) == 29 and TComplexity(t=3, rotations=7).t_incl_rotations() == 367 and TComplexity(rotations=1).t_incl_rotations(eps=1e-5) == 29
    assert TComplexity(1, 2, 3) + TComplexity(10, 20, 30) == TComplexity(11, 22, 33) and 2 * TComplexity(1, 2, 3) == TComplexity(1, 2, 3) * 2 == TComplexity(2, 4, 6)
    assert str(TComplexity(t=10**8, clifford=5)) == "T-count:   1e+08\nRotations: 0\nCliffords: 5\n"


@needs_qualtran
def test_qualtran_legacy_t_complexity_against_qualtran():
    from qualtran.cirq_interop.t_complexity_protocol import TComplexity as TheirTComplexity

    for kwargs in ({"t": 5, "toffoli": 3, "cswap": 2, "and_bloq": 7, "clifford": 11, "rotation": 13, "measurement": 17}, {"and_bloq": 1}, {}):
        theirs = TheirGateCounts(**kwargs).to_legacy_t_complexity()
        assert GateCounts(**kwargs).to_legacy_t_complexity().asdict() == {"t": theirs.t, "rotations": theirs.rotations, "clifford": theirs.clifford}
    for eps in (1e-3, 1e-5, 1e-11, 1e-20):
        assert TComplexity.rotation_cost(eps) == TheirTComplexity.rotation_cost(eps)
        assert TComplexity(t=3, clifford=4, rotations=7).t_incl_rotations(eps) == TheirTComplexity(t=3, clifford=4, rotations=7).t_incl_rotations(eps)
    for t, c, r in ((10**8, 5, 0), (0, 0, 0), (123456789, 7, 3)):
        assert str(TComplexity(t=t, clifford=c, rotations=r)) == str(TheirTComplexity(t=t, clifford=c, rotations=r))


@needs_qualtran
def test_qualtran_summary_from_bloq():
    """Qualtran's own cases (surface_code/algorithm_summary_test.py), counted by Qualtran's resource counting."""
    from qualtran.bloqs import basic_gates, mcmt, rotations

    cases = [
        (basic_gates.TGate(is_adjoint=False), 1, GateCounts(t=1)),
        (basic_gates.Toffoli(), 3, GateCounts(toffoli=1)),
        (basic_gates.TwoBitCSwap(), 3, GateCounts(cswap=1)),
        (mcmt.And(), 3, GateCounts(and_bloq=1)),
        (basic_gates.ZPowGate(exponent=0.1, eps=1e-11), 1, GateCounts(rotation=1)),
        (rotations.phase_gradient.PhaseGradientUnitary(bitsize=10, exponent=1, is_controlled=False, eps=1e-10), 10, GateCounts(clifford=2, t=1, rotation=7)),
        (mcmt.MultiControlX(cvs=(1, 1, 1)), 6, GateCounts(and_bloq=2, measurement=2, clifford=3)),
    ]
    for bloq, qubits, gates in cases:
        ours = AlgorithmSummary.from_bloq(bloq)
        assert ours == AlgorithmSummary(n_algo_qubits=qubits, n_logical_gates=gates)
        theirs = their_sc.AlgorithmSummary.from_bloq(bloq)
        assert (theirs.n_algo_qubits, theirs.n_logical_gates.asdict()) == (ours.n_algo_qubits, ours.n_logical_gates.asdict())


def test_qualtran_summary_from_bloq_needs_qualtran(monkeypatch):
    from stabilizer_qec.estimate import qualtran as qt

    monkeypatch.setitem(sys.modules, "qualtran.resource_counting", None)
    with pytest.raises(ImportError, match="pip install qualtran"):
        AlgorithmSummary.from_bloq(object())
    assert qt._plain_count("t", np.int64(3)) == 3
    with pytest.raises(ValueError, match="symbolic"):
        qt._plain_count("t", object())
    with pytest.raises(ValueError, match="not an integer"):
        qt._plain_count("t", 2.5)

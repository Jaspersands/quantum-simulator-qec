"""The QDK's parameter classes (qdk/estimator/_estimator.py, qdk 1.33.1) as `stabilizer_qec.estimate.azure` has
them: every case of tests/estimate_port/qdk_params.json (recorded by tools/qdk_report_fixtures.py from the QDK's
own classes) built from the port's gives the QDK's ``as_dict()`` (or other value) or raises the QDK's exception
with its message, with the same warnings; and the port's estimate takes the objects as the QDK's does."""

from __future__ import annotations

import json
import pathlib
import pickle
import sys
import warnings

import pytest

from stabilizer_qec.estimate import azure as az

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import qdk_report_fixtures as fx  # noqa: E402

FIXTURE = json.loads((pathlib.Path(__file__).parent / "qdk_params.json").read_text(encoding="utf-8"))
CASES = FIXTURE["cases"]
RECORDED_PYTHON = tuple(int(x) for x in FIXTURE["python"].split("."))
QDK_NAMES = ["EstimatorError", "LogicalCounts", "EstimatorResult", "QubitParams", "QECScheme", "MeasurementErrorRate", "EstimatorQubitParams", "EstimatorQecScheme", "ProtocolSpecificDistillationUnitSpecification", "DistillationUnitSpecification", "ErrorBudgetPartition", "EstimatorConstraints", "EstimatorInputParamsItem", "EstimatorParams"]
DEPRECATION = "This version of QRE is deprecated and will be removed in a future release. Please use the new version of QRE in qdk.qre. Refer to aka.ms/qdk.QREv3 for more information."

try:
    import qdk.estimator as qdk_estimator
except ImportError:  # pragma: no cover
    qdk_estimator = None

needs_qdk = pytest.mark.skipif(qdk_estimator is None, reason="qdk is not installed")


def test_fixture_is_the_tools_cases():
    assert FIXTURE["qdk"] == fx.QDK_VERSION and [{k: v for k, v in c.items() if k != "qdk"} for c in CASES] == json.loads(json.dumps(fx.PARAM_CASES))
    assert len(CASES) >= 80 and sum("raised" in c["qdk"] for c in CASES) >= 40


@pytest.mark.parametrize("index", range(len(CASES)), ids=[c["name"] for c in CASES])
def test_params_against_qdk_recorded(index):
    case = CASES[index]
    ours, theirs = fx.run_param_case(case, az), dict(case["qdk"])
    if case.get("py") and sys.version_info[:2] != RECORDED_PYTHON:
        # A TypeError Python raises itself is worded by the Python that runs: 3.9 says `__init__() missing 2 required
        # positional arguments` where 3.10 names the class, and `re.match` of a number says `expected string or
        # bytes-like object` before 3.12, which adds `, got 'int'`. The class is still compared.
        ours.pop("str", None)
        theirs.pop("str", None)
    assert ours == theirs


def test_qdk_names():
    """Everything ``qdk.estimator`` exports, under its name; `QECScheme` is the port's scheme model, which carries
    the QDK's two names."""
    for name in QDK_NAMES:
        assert hasattr(az, name), name
    assert {az.QubitParams.GATE_US_E3, az.QubitParams.GATE_US_E4, az.QubitParams.GATE_NS_E3, az.QubitParams.GATE_NS_E4, az.QubitParams.MAJ_NS_E4, az.QubitParams.MAJ_NS_E6} == set(az.QUBIT_NAMES) and len(az.QUBIT_NAMES) == 6
    assert (az.QECScheme.SURFACE_CODE, az.QECScheme.FLOQUET_CODE) == ("surface_code", "floquet_code")
    assert az.QECScheme.floquet_code().name == az.QECScheme.FLOQUET_CODE and "SURFACE_CODE" not in az.QECScheme.__dataclass_fields__
    from stabilizer_qec import estimate

    for name in QDK_NAMES:
        if name != "QECScheme":  # the package's QECScheme is Qualtran's; the Azure one is AzureQECScheme
            assert getattr(estimate, name) is getattr(az, name), name


@needs_qdk
def test_qdk_names_live():
    assert set(qdk_estimator.__all__) == set(QDK_NAMES)
    for cls in ("QubitParams", "QECScheme"):
        theirs = {k: v for k, v in vars(getattr(qdk_estimator, cls)).items() if k.isupper()}
        assert theirs == {k: getattr(getattr(az, cls), k) for k in theirs}, cls


def test_estimator_error_derives_from_base_exception():
    """As the QDK's: a BaseException, not an Exception, so ``except Exception`` lets it through."""
    assert az.EstimatorError.__bases__ == (BaseException,) and not issubclass(az.EstimatorError, Exception)
    counts = az.LogicalCounts(num_qubits=10, t_count=100)
    with pytest.raises(az.EstimatorError) as e:
        try:
            counts.estimate({"constraints": {"maxDuration": "1 ns"}})
        except Exception:  # pragma: no cover - must not catch it
            pytest.fail("EstimatorError caught as an Exception")
    err = e.value
    assert (err.code, err.message, err.reason) == ("Qdk.Qsc.Estimates.MaxDurationTooSmall", "Error estimating resources (Qdk.Qsc.Estimates.MaxDurationTooSmall):\nNo solution found for the provided maximum duration.", "No solution found for the provided maximum duration.")
    again = pickle.loads(pickle.dumps(err))
    assert (type(again), again.code, str(again)) == (az.EstimatorError, err.code, str(err))
    # The QDK's __init__ does not call BaseException's: its args are the code and reason it was made with.
    assert err.args == ("Qdk.Qsc.Estimates.MaxDurationTooSmall", "No solution found for the provided maximum duration.") and again.args == err.args
    assert repr(az.EstimatorError("c", "m")) == "EstimatorError('c', 'm')"


@needs_qdk
def test_estimator_error_args_and_repr_against_qdk_live():
    for code, message in (("c", "m"), ("Qdk.Qsc.Estimates.MaxDurationTooSmall", "No solution found.\nSecond line")):
        ours, theirs = az.EstimatorError(code, message), qdk_estimator.EstimatorError(code, message)
        assert (ours.args, repr(ours), str(ours), ours.code, ours.message) == (theirs.args, repr(theirs), str(theirs), theirs.code, theirs.message)


def test_estimator_params_warns_its_deprecation():
    with pytest.warns(DeprecationWarning) as caught:
        az.EstimatorParams()
    assert [str(w.message) for w in caught] == [DEPRECATION] and caught[0].filename == __file__  # stacklevel 2: the caller's line
    with pytest.warns(DeprecationWarning, match="deprecated"):
        params = az.EstimatorParams(num_items=3)
    assert params.has_items and len(params.items) == 3 and all(type(i) is az.EstimatorInputParamsItem for i in params.items)


def test_estimator_params_cap_is_a_thousand_items():
    assert az.EstimatorParams.MAX_NUM_ITEMS == 1000
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        assert len(az.EstimatorParams(num_items=1000).as_dict()["items"]) == 1000
        for bad in (0, -1, 1001):
            with pytest.raises(ValueError, match=r"^num_items must be a positive value less or equal to 1000$"):
                az.EstimatorParams(num_items=bad)
        with pytest.raises(Exception, match="^Cannot access items in a non-batching job, call make_params with num_items parameter$"):
            _ = az.EstimatorParams().items


def _params(make):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return make()


def test_estimate_takes_the_parameter_objects():
    """`LogicalCounts.estimate` takes an `EstimatorParams` as the QDK's does (a batch's merged items, else the one
    dict) and refuses an item alone as it does, and `azure_estimate` takes each section as its object."""
    counts = az.LogicalCounts(num_qubits=12, t_count=500, rotation_count=4, rotation_depth=2)

    def batch():
        p = az.EstimatorParams(num_items=3)
        p.qubit_params.name = az.QubitParams.GATE_NS_E4
        p.error_budget = 0.01
        p.items[1].qubit_params.name = az.QubitParams.MAJ_NS_E6
        p.items[1].qec_scheme.name = az.QECScheme.FLOQUET_CODE
        p.items[2].constraints.max_duration = "1 ns"
        return p

    p = _params(batch)
    result = counts.estimate(p)
    assert result.json == counts.estimate(p.as_dict()["items"]).json and result[2].error.code == "Qdk.Qsc.Estimates.MaxDurationTooSmall"
    assert result.data(1)["jobParams"]["qecScheme"]["name"] == "floquet_code" and result.data(0)["jobParams"]["errorBudget"] == 0.01
    single = _params(az.EstimatorParams)
    single.qubit_params = az.EstimatorQubitParams(name=az.QubitParams.GATE_US_E3, one_qubit_measurement_error_rate=1e-4)
    single.error_budget = az.ErrorBudgetPartition(1e-4, 2e-4, 3e-4)
    single.distillation_unit_specifications = [az.DistillationUnitSpecification(name="15-1 RM")]
    assert counts.estimate(single).json == counts.estimate(single.as_dict()).json
    # The QDK takes its EstimatorParams only: an item alone reaches its json.dumps, which refuses it.
    with pytest.raises(TypeError, match="^Object of type EstimatorInputParamsItem is not JSON serializable$"):
        counts.estimate(az.EstimatorInputParamsItem())
    unit = az.DistillationUnitSpecification(display_name="mine", **fx.CUSTOM_UNIT, physical_qubit_specification=az.ProtocolSpecificDistillationUnitSpecification(num_unit_qubits=31, duration_in_qubit_cycle_time=24))
    a = az.azure_estimate(counts, az.EstimatorQubitParams(name="qubit_maj_ns_e4", two_qubit_joint_measurement_error_rate=az.MeasurementErrorRate(1e-4, 2e-4)), az.EstimatorQecScheme(name="floquet_code"), az.ErrorBudgetPartition(), az.EstimatorConstraints(max_t_factories=3), [unit, az.DistillationUnitSpecification(name="15-1 space-efficient")])
    b = az.azure_estimate(counts, {"name": "qubit_maj_ns_e4", "twoQubitJointMeasurementErrorRate": {"process": 1e-4, "readout": 2e-4}}, {"name": "floquet_code"}, {"logical": 0.001 / 3, "tStates": 0.001 / 3, "rotations": 0.001 / 3}, {"maxTFactories": 3}, [dict(displayName="mine", numInputTs=15, numOutputTs=1, failureProbabilityFormula=fx.CUSTOM_UNIT["failure_probability_formula"], outputErrorRateFormula=fx.CUSTOM_UNIT["output_error_rate_formula"], physicalQubitSpecification={"numUnitQubits": 31, "durationInQubitCycleTime": 24}), {"name": "15-1 space-efficient"}])
    assert a.json == b.json and a["jobParams"]["distillationUnitSpecifications"][0]["displayName"] == "mine"
    with pytest.raises(LookupError, match="^one_qubit_measurement_time must be set$"):
        counts.estimate(_set_qubit(az.EstimatorQubitParams(instruction_set="GateBased")))


def _set_qubit(qubit):
    p = _params(az.EstimatorParams)
    p.qubit_params = qubit
    return p


@needs_qdk
def test_estimate_with_parameter_objects_against_qdk_live():
    """The same objects built from the QDK's classes and from the port's give the same estimate, the QDK's."""
    counts = {"numQubits": 12, "tCount": 500, "rotationCount": 4, "rotationDepth": 2, "cczCount": 3}
    for lc, ns in ((qdk_estimator.LogicalCounts(counts), qdk_estimator), (az.LogicalCounts.fromdict(counts), az)):
        with pytest.raises(TypeError, match="^Object of type EstimatorInputParamsItem is not JSON serializable$"):
            lc.estimate(ns.EstimatorInputParamsItem())
    for case in CASES:
        if "value" not in case["qdk"] or not case["make"]["new"].startswith("EstimatorParams") or case.get("call", ["as_dict"]) != ["as_dict"]:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            theirs, ours = fx.build(case["make"], qdk_estimator), fx.build(case["make"], az)
            if ours.has_items and len(ours.items) > 3:
                continue
            expected = qdk_estimator.LogicalCounts(counts).estimate(theirs).json
            # The port takes its own objects and the QDK's alike.
            assert az.LogicalCounts(counts).estimate(ours).json == expected == az.LogicalCounts(counts).estimate(theirs).json, case["name"]

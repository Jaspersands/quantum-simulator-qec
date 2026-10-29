"""Degenerate parameters the QDK accepts, as `stabilizer_qec.estimate.azure` takes them: a crossing prefactor of 0
(or one so small the logical error rate underflows) gives the QDK's estimate, where its f64 division gives an
infinite cycle allowance and the `as u64` cast saturates; and the inputs on which the QDK panics dividing by zero
(a logical cycle time that rounds to 0 ns, a T factory that cannot finish a run, a patch of 0 physical qubits)
raise a ValueError saying so, never a bare ZeroDivisionError or OverflowError. The values are recorded from
qdk 1.33.1; the live tests compare the whole result with the installed QDK."""

import json
import warnings

import pytest

from stabilizer_qec.estimate import azure as az

try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import qdk.estimator as qdk_estimator
except ImportError:  # pragma: no cover
    qdk_estimator = None

needs_qdk = pytest.mark.skipif(qdk_estimator is None, reason="qdk is not installed")

COUNTS = {"numQubits": 12, "tCount": 1000000, "rotationCount": 10, "rotationDepth": 1, "measurementCount": 10}


def _item(qubit, scheme, prefactor, **extra):
    return dict({"qubitParams": {"name": qubit}, "qecScheme": {"name": scheme, "crossingPrefactor": prefactor}}, **extra)


# (physical qubits, runtime ns, code distance, logical cycle time ns) as qdk 1.33.1 gives them, for every prefactor
# of PREFACTORS: 0, a subnormal whose rate underflows past the division, and the smallest subnormal.
PREFACTORS = (0.0, 1e-320, 5e-324)
RECORDED = {
    ("qubit_maj_ns_e4", "surface_code"): (479170, 2000066000, 1, 2000),
    ("qubit_maj_ns_e4", "floquet_code"): (728372, 300009900, 1, 300),
    ("qubit_maj_ns_e6", "surface_code"): (3934, 2000066000, 1, 2000),
    ("qubit_maj_ns_e6", "floquet_code"): (7592, 300009900, 1, 300),
}
# More of the same, with their recorded physical qubits: a frontier, the constraints, a polynomial prefactor.
RECORDED_MORE = [
    ("frontier", _item("qubit_maj_ns_e4", "surface_code", 0.0, estimateType="frontier"), [479170, 460006, 440842, 421678, 402514, 383350, 364186, 345022, 325858, 306694, 287530, 268366, 249202, 230038, 210874, 191710, 172546, 153382, 134218, 115054, 95890, 76726, 57562, 38398, 19234]),
    ("max_physical_qubits", _item("qubit_maj_ns_e4", "surface_code", 0.0, constraints={"maxPhysicalQubits": 10000000}), 9984514),
    ("max_t_factories", _item("qubit_maj_ns_e4", "surface_code", 0.0, constraints={"maxTFactories": 3}), 57562),
    ("distance_coefficient_power", {"qubitParams": {"name": "qubit_maj_ns_e4"}, "qecScheme": {"name": "surface_code", "crossingPrefactor": 0.0, "distanceCoefficientPower": 2}}, 479170),
    ("prefactor_1e-300", _item("qubit_maj_ns_e4", "surface_code", 1e-300), 479170),
]
BATCH = [{"qubitParams": {"name": "qubit_gate_ns_e3"}}, _item("qubit_maj_ns_e4", "surface_code", 0.0)]

LOW = {"errorCorrectionThreshold": 0.01, "crossingPrefactor": 0.03, "logicalCycleTime": "codeDistance / 100", "physicalQubitsPerLogicalQubit": "2 * codeDistance * codeDistance"}
NAN = dict(LOW, logicalCycleTime="NaN")
NO_QUBITS = {"errorCorrectionThreshold": 0.01, "crossingPrefactor": 0.03, "logicalCycleTime": "codeDistance * 100", "physicalQubitsPerLogicalQubit": "codeDistance / 100"}
T_PROGRAM = {"numQubits": 10, "tCount": 100}
MEASUREMENTS = {"numQubits": 10, "measurementCount": 100}
# Inputs the QDK accepts and then panics on (``attempt to divide by zero``, or ``Invalid formatting`` when its
# report formats the infinite clock frequency of a 0 ns cycle), and the panic's text.
PANICS = [
    ("max_t_factories_tiny_program", {"numQubits": 0, "tCount": 1}, {"constraints": {"maxTFactories": 1}}, "attempt to divide by zero"),
    ("cycle_time_below_half", T_PROGRAM, {"qecScheme": LOW}, "attempt to divide by zero"),
    ("cycle_time_nan", T_PROGRAM, {"qecScheme": NAN}, "attempt to divide by zero"),
    ("cycle_time_below_half_max_duration", T_PROGRAM, {"qecScheme": LOW, "constraints": {"maxDuration": "1 s"}}, "attempt to divide by zero"),
    ("cycle_time_below_half_max_physical_qubits", T_PROGRAM, {"qecScheme": LOW, "constraints": {"maxPhysicalQubits": 10000000}}, "attempt to divide by zero"),
    ("cycle_time_below_half_frontier", T_PROGRAM, {"qecScheme": LOW, "estimateType": "frontier"}, "attempt to divide by zero"),
    ("zero_physical_qubits_max_physical_qubits", T_PROGRAM, {"qecScheme": NO_QUBITS, "constraints": {"maxPhysicalQubits": 1000}}, "attempt to divide by zero"),
    ("cycle_time_below_half_no_t_states", MEASUREMENTS, {"qecScheme": LOW}, "Invalid formatting"),
    ("cycle_time_nan_no_t_states", MEASUREMENTS, {"qecScheme": NAN}, "Invalid formatting"),
    ("cycle_time_below_half_no_t_states_max_duration", MEASUREMENTS, {"qecScheme": LOW, "constraints": {"maxDuration": "1 s"}}, "Invalid formatting"),
]


def _json(result):
    return json.loads(result.json)


@pytest.mark.parametrize("prefactor", PREFACTORS)
@pytest.mark.parametrize("qubit,scheme", list(RECORDED))
def test_zero_crossing_prefactor_against_qdk_recorded(qubit, scheme, prefactor):
    r = az.LogicalCounts(COUNTS).estimate(_item(qubit, scheme, prefactor))
    assert (r["physicalCounts"]["physicalQubits"], r["physicalCounts"]["runtime"], r["logicalQubit"]["codeDistance"], r["logicalQubit"]["logicalCycleTime"]) == RECORDED[qubit, scheme]


@pytest.mark.parametrize("name,item,expected", RECORDED_MORE, ids=[c[0] for c in RECORDED_MORE])
def test_zero_crossing_prefactor_paths_against_qdk_recorded(name, item, expected):
    data = _json(az.LogicalCounts(COUNTS).estimate(item))
    got = [e["physicalCounts"]["physicalQubits"] for e in data["frontierEntries"]] if name == "frontier" else data["physicalCounts"]["physicalQubits"]
    assert got == expected


def test_zero_crossing_prefactor_in_a_batch_against_qdk_recorded():
    r = az.LogicalCounts(COUNTS).estimate(BATCH)
    assert [r[i]["status"] for i in range(2)] == ["success", "success"]
    assert [r[i]["physicalCounts"]["physicalQubits"] for i in range(2)] == [525070, 479170]


def test_logical_cycles_for_a_zero_rate_saturate():
    # The QDK's `(budget / (qubits as f64 * rate)).floor() as u64`: a rate of 0, or one whose quotient overflows,
    # allows 2^64 - 1 cycles.
    for prefactor in PREFACTORS:
        est = az._job_estimator(az.LogicalCounts(COUNTS), az._read_params([_item("qubit_maj_ns_e4", "surface_code", prefactor)])[0])
        assert est.logical_cycles_for_code_distance(1e-3, 1) == 2**64 - 1
        assert est.logical_cycles_for_code_distance(1e-3, 3) == 2**64 - 1


@pytest.mark.parametrize("qubit", ["qubit_gate_ns_e3", "qubit_gate_us_e4"])
@pytest.mark.parametrize("prefactor", [0.0, 1e-320])
def test_zero_crossing_prefactor_where_the_qdk_hangs(qubit, prefactor):
    # With gate-based qubits no T factory meets the T budget at distance 1, and the QDK's no-factory loop never
    # ends (checked with a timeout against qdk 1.33.1): the documented ValueError, not an EstimatorError.
    with pytest.raises(ValueError, match="the estimate has no solution") as e:
        az.LogicalCounts(COUNTS).estimate(_item(qubit, "surface_code", prefactor))
    assert not isinstance(e.value, az.EstimatorError)


@pytest.mark.parametrize("name,counts,params,panic", PANICS, ids=[c[0] for c in PANICS])
def test_qdk_panics_are_value_errors(name, counts, params, panic):
    with pytest.raises(ValueError, match=f"the QDK panics here.*{panic}") as e:
        az.LogicalCounts(counts).estimate(params)
    assert not isinstance(e.value, az.EstimatorError)


def test_qdk_panic_fails_the_whole_batch():
    # The QDK's call panics as a whole; so does the port's, rather than holding the item as a failed one.
    with pytest.raises(ValueError, match="the QDK panics here"):
        az.LogicalCounts(T_PROGRAM).estimate([{}, {"qecScheme": NAN}])


def test_divide_by_zero_helpers():
    assert az._ceil_div(7, 2) == 4 and az._floor_div(7, 2) == 3 and az._ceil_div(0, 5) == 0
    for fn in (az._ceil_div, az._floor_div):
        with pytest.raises(ValueError, match="attempt to divide by zero"):
            fn(1, 0)
    assert (az._ceil_as_u64(2.5), az._ceil_as_u64(float("inf")), az._ceil_as_u64(float("nan")), az._ceil_as_u64(1e30)) == (3, 2**64 - 1, 0, 2**64 - 1)


@needs_qdk
def test_zero_crossing_prefactor_against_qdk_live():
    items = [_item(q, s, p) for q, s in RECORDED for p in PREFACTORS] + [c[1] for c in RECORDED_MORE]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for item in items:
            assert _json(az.LogicalCounts(COUNTS).estimate(item)) == _json(qdk_estimator.LogicalCounts(COUNTS).estimate(item))
        assert _json(az.LogicalCounts(COUNTS).estimate(BATCH)) == _json(qdk_estimator.LogicalCounts(COUNTS).estimate(BATCH))


@needs_qdk
@pytest.mark.parametrize("name,counts,params,panic", PANICS, ids=[c[0] for c in PANICS])
def test_qdk_panics_against_qdk_live(name, counts, params, panic):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(BaseException) as e:
            qdk_estimator.LogicalCounts(counts).estimate(params)
    assert type(e.value).__name__ == "PanicException" and panic in str(e.value)

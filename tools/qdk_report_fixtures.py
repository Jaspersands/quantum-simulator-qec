"""The QDK's parameter classes and its report, recorded from the QDK (qdk 1.33.1) for the Azure estimate port.

    python -I tools/qdk_report_fixtures.py [--markdown DIR]

Writes `tests/estimate_port/qdk_params.json`: what `qdk.estimator` exports (``names``: its ``__all__`` and its
classes' constants), and each case of `PARAM_CASES` built from the QDK's own classes and what it gives, its
``as_dict()`` (or another call) or the exception's class and message, with the warnings raised on the way. And
`tests/estimate_port/qdk_reports.json.gz`: the QDK's whole result for every case of `qdk_estimates.json` and
`qdk_review2.json` it solves (re-run here, the same as recorded before), and, under ``renderings``, each of
`REPORT_CASES` (one of those results, a batch of them with failed items, …) fed to the QDK's own
`EstimatorResult`: its ``_repr_html_()``, ``summary``, ``[:]`` and other slices, and the diagrams' HTML, with
what it printed. ``plain`` is drawn without markdown; ``markdown`` with markdown and
python-markdown-math, the QDK's optional renderer of the explanations (`--markdown` names a directory they are
installed in; without it that variant is not recorded). Each HTML is kept as indices into one table of lines
(``lines``; `unpack` joins them again). Under ``widget``, the charts of qsharp_widgets 1.33.1 (the QDK's
notebook widgets; their bundle is read as data and its chart functions run in node, in the en-US locale): color
maps, axis ticks, JavaScript's number strings, each single result's space chart (SpaceChart) and scatter charts
of several (the estimates overview's), as element trees.

`tests/estimate_port/test_qdk_params.py` and `test_qdk_report.py` build the same cases from the port's classes
(`stabilizer_qec.estimate.azure`) with `build`, `run_param_case` and `render_case` here, which need neither the
QDK nor markdown. Run with ``-I``: the QDK is read as data.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import gzip
import io
import json
import pathlib
import platform
import sys
import warnings
from typing import Any, Callable, Dict, List

ROOT = pathlib.Path(__file__).resolve().parent.parent
PORT = ROOT / "tests" / "estimate_port"
QDK_VERSION = "1.33.1"


# ---------------------------------------------------------------------------------------------
# Building parameter objects from a spec, with either module's classes
# ---------------------------------------------------------------------------------------------


def new(cls: str, *args: Any, **kwargs: Any) -> dict:
    """A spec: ``cls(*args, **kwargs)``, arguments themselves specs where they are dicts with ``new``."""
    return {"new": cls, "args": list(args), "kwargs": kwargs}


def const(path: str) -> dict:
    """A spec: a constant of the module, as ``QubitParams.GATE_NS_E3``."""
    return {"const": path}


def build(spec: Any, ns: Any) -> Any:
    """The value a spec stands for, its classes and constants taken from `ns` (``qdk.estimator`` or
    ``stabilizer_qec.estimate.azure``)."""
    if isinstance(spec, list):
        return [build(x, ns) for x in spec]
    if isinstance(spec, dict) and "const" in spec:
        value = ns
        for part in spec["const"].split("."):
            value = getattr(value, part)
        return value
    if isinstance(spec, dict) and "new" in spec:
        obj = getattr(ns, spec["new"])(*build(spec["args"], ns), **{k: build(v, ns) for k, v in spec["kwargs"].items()})
        for path, value in spec.get("set", []):
            *head, last = path.split(".")
            target = obj
            for part in head:
                target = target[int(part)] if part.isdigit() else getattr(target, part)
            if last.isdigit():
                target[int(last)] = build(value, ns)
            else:
                setattr(target, last, build(value, ns))
        return obj
    return spec


def encode(value: Any) -> Any:
    """A value as JSON: an object (a parameter object ``as_dict`` left in place, an item) as its class name and
    attributes."""
    if isinstance(value, dict):
        return {str(k): encode(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [encode(v) for v in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return {"$obj": type(value).__name__, "attrs": encode(vars(value))}


def qdk_names(ns: Any) -> dict:
    """What ``qdk.estimator`` exports: its ``__all__``, and the constants of its classes that have them (the
    upper-case class attributes, as ``QubitParams.GATE_NS_E3``), by class."""
    constants = {}
    for name in ns.__all__:
        found = {k: v for k, v in vars(getattr(ns, name)).items() if k.isupper()}
        if found:
            constants[name] = found
    return {"all": list(ns.__all__), "constants": constants}


def run_param_case(case: dict, ns: Any) -> dict:
    """A case's outcome with `ns`'s classes: the call's value (or the exception's class and message) and the
    warnings raised while building and calling."""
    out: Dict[str, Any] = {}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            obj = build(case["make"], ns)
            call = case.get("call", ["as_dict"])
            if call[0] == "as_dict":
                value = obj.as_dict(**(call[1] if len(call) > 1 else {}))
            elif call[0] == "attr":
                value = getattr(obj, call[1])
            elif call[0] == "repr":
                value = repr(obj)
            elif call[0] == "item_as_dict":  # an item merged with the top level, as EstimatorParams.as_dict does
                value = obj.items[call[1]].as_dict(True, obj)
            else:
                raise AssertionError(call)
            out["value"] = encode(value)
        except Exception as e:  # noqa: BLE001 - the exception is the outcome
            out["raised"] = type(e).__name__
            out["str"] = str(e)
    out["warnings"] = [[w.category.__name__, str(w.message)] for w in caught]
    return out


M = "MeasurementErrorRate"
Q = "EstimatorQubitParams"
S = "EstimatorQecScheme"
P = "ProtocolSpecificDistillationUnitSpecification"
D = "DistillationUnitSpecification"
B = "ErrorBudgetPartition"
C = "EstimatorConstraints"
I = "EstimatorInputParamsItem"
E = "EstimatorParams"
NOVALIDATE = ["as_dict", {"validate": False}]
GATE = {"instruction_set": "GateBased", "one_qubit_measurement_time": "100 ns", "one_qubit_measurement_error_rate": 1e-3, "one_qubit_gate_time": "50 ns"}
CUSTOM_UNIT = {"num_input_ts": 15, "num_output_ts": 1, "failure_probability_formula": "15.0 * inputErrorRate + 356.0 * cliffordErrorRate", "output_error_rate_formula": "35.0 * inputErrorRate ^ 3 + 7.1 * cliffordErrorRate"}


def _set(spec: dict, *pairs: Any) -> dict:
    spec = dict(spec)
    spec["set"] = [list(p) for p in pairs]
    return spec


# Every parameter class: valid, invalid (each validator and each post-validation message), nested, unvalidated,
# and EstimatorParams' batches with their merging. A case whose message comes from Python (a TypeError of a
# comparison or of `re`) is recorded too; its text may differ on another Python (the test says where).
PARAM_CASES: List[dict] = [
    # MeasurementErrorRate
    {"name": "mer valid", "make": new(M, 0.1, 0.2)},
    {"name": "mer keywords", "make": new(M, readout=0.25, process=0.5)},
    {"name": "mer process zero", "make": new(M, 0.0, 0.2)},
    {"name": "mer readout one", "make": new(M, 0.1, 1.0)},
    {"name": "mer readout negative int", "make": new(M, 0.1, -1)},
    {"name": "mer readout string", "make": new(M, 0.1, "0.2")},
    {"name": "mer unvalidated", "make": new(M, 2.0, -1.0), "call": NOVALIDATE},
    {"name": "mer repr", "make": new(M, 0.1, 0.2), "call": ["repr"]},
    {"name": "mer missing arguments", "make": new(M), "py": True},
    # EstimatorQubitParams
    {"name": "qubit empty", "make": new(Q)},
    {"name": "qubit repr", "make": new(Q, name="qubit_gate_ns_e3"), "call": ["repr"]},
] + [
    {"name": f"qubit preset {p}", "make": new(Q, name=const(f"QubitParams.{p}"))} for p in ("GATE_US_E3", "GATE_US_E4", "GATE_NS_E3", "GATE_NS_E4", "MAJ_NS_E4", "MAJ_NS_E6")
] + [
    {"name": "qubit preset with overrides", "make": new(Q, name="qubit_gate_ns_e4", one_qubit_gate_error_rate=1e-4, t_gate_time="60 ns")},
    {"name": "qubit preset with instruction set", "make": new(Q, name="qubit_maj_ns_e4", instruction_set="Majorana")},
    {"name": "qubit preset with bad instruction set", "make": new(Q, name="qubit_maj_ns_e4", instruction_set="photonic")},
    {"name": "qubit custom name only", "make": new(Q, name="my_qubit")},
    {"name": "qubit custom no instruction set", "make": new(Q, one_qubit_gate_time="50 ns")},
    {"name": "qubit custom instruction set only", "make": new(Q, instruction_set="GateBased")},
    {"name": "qubit custom no measurement rate", "make": new(Q, instruction_set="gate_based", one_qubit_measurement_time="100 ns")},
    {"name": "qubit custom no gate time", "make": new(Q, instruction_set="gateBased", one_qubit_measurement_time="100 ns", one_qubit_measurement_error_rate=1e-3)},
    {"name": "qubit custom gate based", "make": new(Q, **GATE)},
    {"name": "qubit custom gate based full", "make": new(Q, name="mine", instruction_set="gate-based", one_qubit_measurement_time="1 µs", two_qubit_joint_measurement_time="1 μs", one_qubit_gate_time="50 ns", two_qubit_gate_time="+.5us", t_gate_time="1e-6 s", one_qubit_measurement_error_rate=1e-3, two_qubit_joint_measurement_error_rate=2e-3, one_qubit_gate_error_rate=1e-4, two_qubit_gate_error_rate=2e-4, t_gate_error_rate=1e-3, idle_error_rate=1e-5)},
    {"name": "qubit custom majorana no gate time needed", "make": new(Q, instruction_set="Majorana", one_qubit_measurement_time="100 ns", one_qubit_measurement_error_rate=new(M, 1e-4, 2e-4))},
    {"name": "qubit custom majorana both rates", "make": new(Q, instruction_set="majorana", one_qubit_measurement_time="100 ns", two_qubit_joint_measurement_time="100 ns", t_gate_time="100 ns", one_qubit_measurement_error_rate=new(M, 1e-4, 2e-4), two_qubit_joint_measurement_error_rate=new(M, 3e-4, 4e-4), t_gate_error_rate=0.05, idle_error_rate=1e-5)},
    {"name": "qubit nested rate invalid", "make": new(Q, name="qubit_maj_ns_e4", one_qubit_measurement_error_rate=new(M, 0.0, 1e-4))},
    {"name": "qubit nested rate unvalidated", "make": new(Q, name="qubit_maj_ns_e4", two_qubit_joint_measurement_error_rate=new(M, 5.0, 1e-4)), "call": NOVALIDATE},
    {"name": "qubit instruction set invalid", "make": new(Q, instruction_set="quantum")},
    {"name": "qubit instruction set lowercase gate", "make": new(Q, instruction_set="gatebased")},
    {"name": "qubit time invalid unit", "make": new(Q, name="qubit_gate_ns_e3", one_qubit_measurement_time="100 parsecs")},
    {"name": "qubit time negative", "make": new(Q, name="qubit_gate_ns_e3", t_gate_time="-5 ns")},
    {"name": "qubit time leading space", "make": new(Q, name="qubit_gate_ns_e3", t_gate_time=" 5 ns")},
    {"name": "qubit time trailing newline", "make": new(Q, name="qubit_gate_ns_e3", t_gate_time="5 ns\n")},
    {"name": "qubit time no space", "make": new(Q, name="qubit_gate_ns_e3", two_qubit_gate_time="5ms")},
    {"name": "qubit time exponent", "make": new(Q, name="qubit_gate_ns_e3", two_qubit_gate_time="2.5E+2 ns")},
    {"name": "qubit time number", "make": new(Q, name="qubit_gate_ns_e3", one_qubit_gate_time=50), "py": True},
    {"name": "qubit measurement rate int", "make": new(Q, name="qubit_gate_ns_e3", one_qubit_measurement_error_rate=1)},
    {"name": "qubit measurement rate one", "make": new(Q, name="qubit_gate_ns_e3", one_qubit_measurement_error_rate=1.0)},
    {"name": "qubit measurement rate string", "make": new(Q, name="qubit_gate_ns_e3", two_qubit_joint_measurement_error_rate="1e-3")},
    {"name": "qubit gate rate zero", "make": new(Q, name="qubit_gate_ns_e3", t_gate_error_rate=0.0)},
    {"name": "qubit idle rate string", "make": new(Q, name="qubit_gate_ns_e3", idle_error_rate="0.1")},
    {"name": "qubit invalid unvalidated", "make": new(Q, instruction_set="quantum", t_gate_error_rate=7.0, one_qubit_measurement_error_rate=new(M, 3.0, 4.0)), "call": NOVALIDATE},
    # EstimatorQecScheme
    {"name": "scheme empty", "make": new(S)},
    {"name": "scheme surface", "make": new(S, name=const("QECScheme.SURFACE_CODE"))},
    {"name": "scheme floquet with distance", "make": new(S, name=const("QECScheme.FLOQUET_CODE"), max_code_distance=31)},
    {"name": "scheme custom", "make": new(S, error_correction_threshold=0.01, crossing_prefactor=0.03, distance_coefficient_power=0, logical_cycle_time="(4 * twoQubitGateTime + 2 * oneQubitMeasurementTime) * codeDistance", physical_qubits_per_logical_qubit="2 * codeDistance * codeDistance", max_code_distance=25)},
    {"name": "scheme threshold one", "make": new(S, name="surface_code", error_correction_threshold=1.0)},
    {"name": "scheme unchecked fields", "make": new(S, error_correction_threshold=0.5, crossing_prefactor="x", distance_coefficient_power=-3)},
    {"name": "scheme repr", "make": new(S, name="surface_code"), "call": ["repr"]},
    # ProtocolSpecificDistillationUnitSpecification
    {"name": "protocol empty", "make": new(P)},
    {"name": "protocol no duration", "make": new(P, num_unit_qubits=20)},
    {"name": "protocol no qubits", "make": new(P, duration_in_qubit_cycle_time=13)},
    {"name": "protocol valid", "make": new(P, num_unit_qubits=20, duration_in_qubit_cycle_time=13)},
    {"name": "protocol empty unvalidated", "make": new(P), "call": NOVALIDATE},
    # DistillationUnitSpecification
    {"name": "unit empty", "make": new(D)},
    {"name": "unit name", "make": new(D, name="15-1 RM")},
    {"name": "unit name and custom", "make": new(D, name="15-1 RM", num_input_ts=15)},
    {"name": "unit display name only", "make": new(D, display_name="mine")},
    {"name": "unit no output count", "make": new(D, num_input_ts=15)},
    {"name": "unit no failure formula", "make": new(D, num_input_ts=15, num_output_ts=1)},
    {"name": "unit no output formula", "make": new(D, num_input_ts=15, num_output_ts=1, failure_probability_formula="15 * inputErrorRate")},
    {"name": "unit custom", "make": new(D, display_name="mine", **CUSTOM_UNIT)},
    {"name": "unit custom with specifications", "make": new(D, **CUSTOM_UNIT, physical_qubit_specification=new(P, num_unit_qubits=31, duration_in_qubit_cycle_time=24), logical_qubit_specification=new(P, num_unit_qubits=15, duration_in_qubit_cycle_time=13), logical_qubit_specification_first_round_override=new(P, num_unit_qubits=20, duration_in_qubit_cycle_time=11))},
    {"name": "unit custom override incomplete", "make": new(D, **CUSTOM_UNIT, logical_qubit_specification_first_round_override=new(P, num_unit_qubits=20))},
    {"name": "unit custom physical incomplete", "make": new(D, **CUSTOM_UNIT, physical_qubit_specification=new(P, duration_in_qubit_cycle_time=3))},
    {"name": "unit name with empty specification", "make": new(D, name="15-1 RM", physical_qubit_specification=new(P))},
    {"name": "unit empty specification unvalidated", "make": new(D, **CUSTOM_UNIT, physical_qubit_specification=new(P)), "call": NOVALIDATE},
    {"name": "unit empty unvalidated", "make": new(D), "call": NOVALIDATE},
    {"name": "unit repr", "make": new(D, name="15-1 space-efficient"), "call": ["repr"]},
    # ErrorBudgetPartition
    {"name": "budget default", "make": new(B)},
    {"name": "budget parts", "make": new(B, 0.01, 0.02, 0.03)},
    {"name": "budget unchecked", "make": new(B, logical=-1.0, t_states=2, rotations="x")},
    {"name": "budget repr", "make": new(B), "call": ["repr"]},
    # EstimatorConstraints
    {"name": "constraints empty", "make": new(C)},
    {"name": "constraints valid", "make": new(C, logical_depth_factor=1.5, max_t_factories=10, max_physical_qubits=10**6)},
    {"name": "constraints duration", "make": new(C, max_duration="1 ms", max_t_factories=3)},
    {"name": "constraints depth factor below one", "make": new(C, logical_depth_factor=0.5)},
    {"name": "constraints no factories", "make": new(C, max_t_factories=0)},
    {"name": "constraints no qubits", "make": new(C, max_physical_qubits=0)},
    {"name": "constraints duration invalid", "make": new(C, max_duration="1 fortnight")},
    {"name": "constraints duration number", "make": new(C, max_duration=1000), "py": True},
    {"name": "constraints duration and qubits", "make": new(C, max_duration="1 s", max_physical_qubits=10**6)},
    {"name": "constraints duration and qubits unvalidated", "make": new(C, max_duration="1 s", max_physical_qubits=0), "call": NOVALIDATE},
    # EstimatorInputParamsItem
    {"name": "item empty", "make": new(I)},
    {"name": "item everything", "make": _set(new(I), ("qubit_params.name", "qubit_gate_ns_e4"), ("qec_scheme.name", "surface_code"), ("error_budget", 0.01), ("estimate_type", "frontier"), ("constraints.max_t_factories", 4), ("distillation_unit_specifications", [new(D, name="15-1 RM"), new(D, display_name="mine", **CUSTOM_UNIT)]))},
    {"name": "item budget zero", "make": _set(new(I), ("error_budget", 0.0))},
    {"name": "item budget one int", "make": _set(new(I), ("error_budget", 1))},
    {"name": "item budget bool", "make": _set(new(I), ("error_budget", True))},
    {"name": "item budget string dropped", "make": _set(new(I), ("error_budget", "0.1"))},
    {"name": "item budget partition", "make": _set(new(I), ("error_budget", new(B, 1e-4, 2e-4, 3e-4)))},
    {"name": "item budget zero unvalidated", "make": _set(new(I), ("error_budget", 0.0)), "call": NOVALIDATE},
    {"name": "item estimate type invalid", "make": _set(new(I), ("estimate_type", "pareto"))},
    {"name": "item estimate type invalid unvalidated", "make": _set(new(I), ("estimate_type", "pareto")), "call": NOVALIDATE},
    {"name": "item unit invalid", "make": _set(new(I), ("distillation_unit_specifications", [new(D, name="15-1 RM"), new(D)]))},
    {"name": "item units empty unvalidated", "make": _set(new(I), ("distillation_unit_specifications", [new(D), new(D)])), "call": NOVALIDATE},
    {"name": "item merged with nothing", "make": _set(new(I), ("qubit_params.name", "qubit_gate_us_e3")), "call": ["as_dict", {"additional_params": None}]},
    # EstimatorParams
    {"name": "params empty", "make": new(E)},
    {"name": "params single", "make": _set(new(E), ("qubit_params.name", "qubit_maj_ns_e6"), ("qec_scheme.name", "floquet_code"), ("error_budget", new(B)), ("constraints.logical_depth_factor", 2.0))},
    {"name": "params single has_items", "make": new(E), "call": ["attr", "has_items"]},
    {"name": "params single items", "make": new(E), "call": ["attr", "items"]},
    {"name": "params zero items", "make": new(E, 0)},
    {"name": "params negative items", "make": new(E, num_items=-1)},
    {"name": "params 1001 items", "make": new(E, num_items=1001)},
    {"name": "params 1000 items", "make": new(E, num_items=1000)},
    {"name": "params cap", "make": new(E), "call": ["attr", "MAX_NUM_ITEMS"]},
    {"name": "params items string", "make": new(E, num_items="2")},
    {"name": "params items float", "make": new(E, num_items=2.5)},
    {"name": "params one item", "make": new(E, num_items=1)},
    {"name": "params batch items", "make": new(E, num_items=2), "call": ["attr", "items"]},
    {"name": "params batch merged", "make": _set(new(E, num_items=3), ("qubit_params.name", "qubit_gate_ns_e3"), ("qec_scheme.name", "surface_code"), ("error_budget", 0.01), ("constraints.max_t_factories", 2), ("items.1.qubit_params.name", "qubit_maj_ns_e4"), ("items.1.qec_scheme.name", "floquet_code"), ("items.2.error_budget", new(B, 1e-4, 2e-4, 3e-4)), ("items.2.constraints.max_physical_qubits", 10**6))},
    {"name": "params batch units only where the item has its own", "make": _set(new(E, num_items=2), ("distillation_unit_specifications", [new(D, name="15-1 space-efficient")]), ("items.1.distillation_unit_specifications", [new(D, name="15-1 RM")]))},
    {"name": "params batch estimate type not inherited", "make": _set(new(E, num_items=2), ("estimate_type", "frontier"), ("items.0.estimate_type", "singlePoint"))},
    {"name": "params batch partition inherited", "make": _set(new(E, num_items=2), ("error_budget", new(B, 0.002, 0.003, 0.004)))},
    {"name": "params batch top budget invalid", "make": _set(new(E, num_items=2), ("error_budget", 1.5))},
    {"name": "params batch item budget invalid", "make": _set(new(E, num_items=2), ("items.1.error_budget", -0.5))},
    {"name": "params batch item qubit invalid", "make": _set(new(E, num_items=2), ("items.1.qubit_params.instruction_set", "GateBased"))},
    {"name": "params batch top qubit invalid", "make": _set(new(E, num_items=2), ("qubit_params.t_gate_error_rate", 2.0))},
    {"name": "params batch unvalidated", "make": _set(new(E, num_items=2), ("error_budget", 1.5), ("items.0.estimate_type", "frontier"), ("items.1.constraints.max_t_factories", 0)), "call": NOVALIDATE},
    {"name": "params batch item merged by hand", "make": _set(new(E, num_items=2), ("qec_scheme.name", "floquet_code"), ("items.0.qubit_params.name", "qubit_maj_ns_e6")), "call": ["item_as_dict", 0]},
    {"name": "params items replaced by a list", "make": _set(new(E, num_items=2), ("items.0", new(I)), ("items.1.qubit_params", new(Q, name="qubit_gate_us_e4")))},
]


# ---------------------------------------------------------------------------------------------
# Rendering recorded results with either module's EstimatorResult
# ---------------------------------------------------------------------------------------------

FAIL_DURATION = {"code": "Qdk.Qsc.Estimates.MaxDurationTooSmall", "message": "No solution found for the provided maximum duration."}
FAIL_BUDGET = {"code": "Qdk.Qsc.Estimates.InvalidInputError.InvalidErrorBudget", "message": "The error budget must be between 0.0 and 1.0, provided input was `0`"}
EST = "qdk_estimates/"
REV = "qdk_review2/"
SINGLES = [EST + k for k in ("big|qubit_gate_ns_e3|surface_code", "big|qubit_maj_ns_e6|floquet_code", "budget_parts", "custom_gate", "custom_maj", "custom_scheme", "custom_unit", "frontier_noT", "frontier_rot_maj_floquet", "frontier_small_gate_ns_e3", "frontier_tiny_us", "logicalDepthFactor", "maxDuration", "maxPhysicalQubits_rot", "noT", "oneT", "one_unit", "rotations|qubit_gate_us_e4|surface_code", "small|qubit_maj_ns_e4|surface_code", "scheme_overrides")] + [REV + k for k in ("echo|memory_counts", "formula|nan_is_zero_qubits", "power|12_wraps_u64", "knife_edge|logical=0.1|tstates=0.005", "unit|name_not_a_string_is_custom")]
# Forty results: 25 recorded estimates, a failed one, and 14 batches (lists) of them: failed items, frontier
# items (their counts in words are "N/A" in the side-by-side table), more than five (the overview is cut off),
# none succeeded (the QDK prints so), a failed first item (with markdown the QDK reads the assumptions of item
# 0 and raises KeyError), items whose groups differ (the QDK reads the first succeeded item's groups by index).
REPORT_CASES: List[dict] = [{"name": key, "data": key} for key in SINGLES] + [
    {"name": "failed", "data": FAIL_DURATION},
    {"name": "batch two qubits", "data": [EST + "big|qubit_gate_ns_e3|surface_code", EST + "big|qubit_maj_ns_e6|floquet_code"], "slices": [[None, None], [1, None]]},
    {"name": "batch failed and frontier", "data": [EST + "custom_gate", FAIL_DURATION, EST + "frontier_small_gate_ns_e3"], "slices": [[None, None], [1, 3], [None, None, -1]]},
    {"name": "batch failed first", "data": [FAIL_BUDGET, EST + "oneT"], "slices": [[None, None], [1, None]]},
    {"name": "batch seven", "data": [EST + "small|qubit_gate_ns_e3|surface_code", EST + "small|qubit_gate_ns_e4|surface_code", FAIL_DURATION, EST + "small|qubit_gate_us_e3|surface_code", EST + "frontier_tiny_us", EST + "small|qubit_maj_ns_e4|floquet_code", EST + "small|qubit_maj_ns_e6|surface_code"], "slices": [[None, None], [5, None], [None, None, 2], [-3, -1]]},
    {"name": "batch none succeeded", "data": [FAIL_DURATION, FAIL_BUDGET], "slices": [[None, None]]},
    {"name": "batch frontier first", "data": [EST + "frontier_rot_maj_floquet", EST + "custom_maj"], "slices": [[None, None]]},
    {"name": "batch units differ", "data": [EST + "custom_unit", EST + "one_unit", EST + "maxTFactories3"], "slices": [[None, None], [0, 2]]},
    {"name": "list of one", "data": [EST + "budget_0.1"]},
    {"name": "list of one failed", "data": [FAIL_DURATION]},
    {"name": "batch six", "data": [EST + "maxDuration", EST + "maxDuration_tight", EST + "maxPhysicalQubits", EST + "maxTFactories1_rot", REV + "power|100", REV + "formula|functions"], "slices": [[None, None], [2, 5], [5, None], [10, None], [None, None, -1]]},
    {"name": "batch two frontiers", "data": [EST + "frontier_noT", EST + "frontier_small_maj_e6_budget"], "slices": [[None, None]]},
    {"name": "batch nan", "data": [REV + "formula|nan_is_zero_qubits", REV + "formula|zero_over_zero_folds_to_one"], "slices": [[None, None]]},
    {"name": "batch fewer groups second", "data": [EST + "big|qubit_gate_us_e4|surface_code", EST + "noT"], "slices": [[None, None], [1, None]]},
    {"name": "batch fewer groups first", "data": [EST + "noT_maxDuration", EST + "tiny_us", EST + "tiny_maj"], "slices": [[None, None]]},
]


def materialize(spec: Any, results: Dict[str, Dict[str, Any]]) -> Any:
    """A case's data: a fixture key (``qdk_estimates/<key>``) is that recorded result, a dict a failed item."""
    if isinstance(spec, list):
        return [materialize(x, results) for x in spec]
    if isinstance(spec, str):
        fixture, key = spec.split("/", 1)
        return copy.deepcopy(results[fixture][key])
    return copy.deepcopy(spec)


def render_case(case: dict, results: Dict[str, Dict[str, Any]], estimator_result: Callable, error: type, diagrams: bool = True) -> dict:
    """What `estimator_result` (the QDK's `EstimatorResult`, or the port's) draws for a case: each step's HTML
    (or its exception's class and message) and what it printed. ``repr`` makes the result and draws
    ``_repr_html_()`` (the QDK draws at construction); a single result's ``summary`` and diagrams, a batch's
    slices."""
    out: Dict[str, Any] = {}
    held: Dict[str, Any] = {}

    def step(name: str, draw: Callable[[], Any]) -> None:
        buf = io.StringIO()
        rec: Dict[str, Any] = {}
        with contextlib.redirect_stdout(buf):
            try:
                rec["html"] = draw()
            except (Exception, error) as e:  # noqa: BLE001 - the QDK's EstimatorError derives from BaseException
                rec["raised"] = type(e).__name__
                rec["str"] = str(e)
        rec["stdout"] = buf.getvalue()
        out[name] = rec

    def make() -> str:
        held["result"] = estimator_result(materialize(case["data"], results))
        return held["result"]._repr_html_()

    step("repr", make)
    result = held.get("result")
    # The QDK draws at construction, so a drawing that fails (KeyError, IndexError) leaves it no result; the
    # port draws when asked, and stops here too. A failed estimate's own error comes from `_repr_html_` in both.
    if result is None or out["repr"].get("raised", error.__name__) != error.__name__:
        return out
    if not isinstance(case["data"], list) or len(case["data"]) == 1:
        step("summary", lambda: result.summary._repr_html_())
        step("[:]", lambda: result[:]._repr_html_())
        if diagrams:
            step("diagram.space", lambda: result.diagram.space._repr_html_())
            step("diagram.time", lambda: result.diagram.time._repr_html_())
    for s in case.get("slices", []):
        step(f"[{':'.join('' if x is None else str(x) for x in s)}]", lambda s=s: result[slice(*s)]._repr_html_())
    return out


def pack(drawn: Dict[str, Dict[str, dict]], lines: Dict[str, int]) -> Dict[str, Dict[str, dict]]:
    """Each HTML as indices into a table of the lines of all of them (`lines`, filled here): the tables repeat
    their template lines thousands of times, which gzip's 32 kB window does not see across results."""
    return {case: {name: dict(rec, html=[lines.setdefault(line, len(lines)) for line in rec["html"].split("\n")]) if isinstance(rec.get("html"), str) else rec for name, rec in steps.items()} for case, steps in drawn.items()}


def unpack(packed: Dict[str, Dict[str, dict]], lines: List[str]) -> Dict[str, Dict[str, dict]]:
    """`pack` undone."""
    return {case: {name: dict(rec, html="\n".join(lines[i] for i in rec["html"])) if isinstance(rec.get("html"), list) else rec for name, rec in steps.items()} for case, steps in packed.items()}


# ---------------------------------------------------------------------------------------------
# The widget's charts: qsharp_widgets' SpaceChart and the estimates overview's scatter, run in node
# ---------------------------------------------------------------------------------------------

# qsharp_widgets 1.33.1's static/index.js bundles microsoft/qdk's source/npm/qsharp/ux (MIT) minified: its
# functions by their names there, each checked by a piece of its text.
WIDGET_FUNCTIONS = {"kh": ("getPieSegment", "Math.PI*n/180"), "so": ("SpaceChart", "qs-widget-spaceChart"), "Zn": ("CreateSingleEstimateResult", "frontierEntries[t]"), "j0": ("ScatterChart", "qs-scatterChart-axis"), "Rh": ("getRanges", "min:r/t"), "cm": ("getMinMaxXYForItems", "Number.MIN_VALUE"), "Lh": ("CreateTimeTicks", ".plural"), "Dh": ("CreateIntegerTicks", "new Intl.NumberFormat"), "Bh": ("ColorMap", "1.4142135623730951"), "xm": ("hslToRgb", "Math.round(n*255)"), "vm": ("rgbToHex", "toUpperCase"), "Mm": ("createRunNames", "sharedRunName"), "ja": ("addIfDifferent", "new Set(n)"), "Sm": ("reDataToRowScatter", "frontierEntries.map(Am)"), "Am": ("frontierEntryToPlotEntry", ", code distance: ")}
WIDGET_VARIABLES = {"yr": "1 nanosecond", "wm": "Runtime", "Em": "Physical qubits"}
NODE_HARNESS = r"""
const ue = (type, props) => ({type, props}), or = "#fragment";
const Er = (v) => ({current: v}), ah = () => ({current: null}), kr = () => {}, Fh = () => {};
%s
function tree(n) {
  if (n == null || n === false || n === true) return [];
  if (typeof n === "string" || typeof n === "number") return [String(n)];
  if (Array.isArray(n)) return n.flatMap(tree);
  if (n.type === or) return tree(n.props.children);
  const attrs = {};
  for (const [k, v] of Object.entries(n.props)) {
    if (k === "children" || k === "ref" || k === "key" || typeof v === "function" || v == null) continue;
    attrs[k === "className" ? "class" : k] = String(v);
  }
  return [{tag: n.type, attrs, children: tree(n.props.children)}];
}
const input = JSON.parse(require("fs").readFileSync(0, "utf8"));
const out = {
  colors: input.colors.map((n) => Bh(n)),
  time_ticks: input.time_ticks.map(([a, b]) => Lh(a, b)),
  integer_ticks: input.integer_ticks.map(([a, b]) => Dh(a, b)),
  numbers: input.numbers.map((s) => { const x = Number(s), r = Math.round(x); return [String(x), x.toLocaleString(), Object.is(r, -0) ? "-0" : String(r)]; }),
  space: input.space.map(([estimate, index]) => tree(so({estimatesData: Zn(estimate, index)}))),
  scatter: input.scatter.map((estimates) => {
    const runNames = Mm(estimates), colors = Bh(estimates.length);
    const data = estimates.map((e, i) => Sm(e, colors[i]));
    return {run_names: runNames, colors, tree: tree(j0({data, xAxis: wm, yAxis: Em, onPointSelected() {}, selectedPoint: undefined, allowSaveImage: false}))};
  }),
};
process.stdout.write(JSON.stringify(out));
"""


def _js_block(source: str, start: int, open_: str, close: str) -> int:
    """The end of the bracketed block whose first `open_` is at or after `start` (strings skipped)."""
    i, depth, quote = source.index(open_, start), 0, None
    while True:
        ch = source[i]
        if quote:
            if ch == "\\":
                i += 1
            elif ch == quote:
                quote = None
        elif ch in "\"'":
            quote = ch
        elif ch == open_:
            depth += 1
        elif ch == close:
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1


def widget_source(bundle: str) -> str:
    """The widget's functions and tables the charts need, cut from the bundle."""
    parts = []
    for name, (what, marker) in WIDGET_FUNCTIONS.items():
        start = bundle.index(f"function {name}(")
        text = bundle[start : _js_block(bundle, bundle.index(")", start), "{", "}")]
        assert bundle.count(f"function {name}(") == 1 and marker in text, (name, what)
        parts.append(text)
    for name, marker in WIDGET_VARIABLES.items():
        start = bundle.index(f"{name}=", bundle.index(marker) - 2000)
        opener = bundle[start + len(name) + 1]
        text = bundle[start : _js_block(bundle, start, opener, {"[": "]", "{": "}"}[opener])]
        assert marker in text, name
        parts.append(f"var {text};")
    return "\n".join(parts)


WIDGET_RANGES = {"time": [[0.5, 2], [1, 1000], [3, 7], [1500, 2500], [5e8, 2e9], [1.5e9, 3.9e9], [30e9, 50e9], [1e12, 5e14], [7e16, 9e16], [4e18, 6e18], [5e19, 9e19], [2e21, 3e21], [7, 3]], "integer": [[1, 1], [3, 7], [25, 95], [120, 180], [512, 4096], [4.5e4, 9.9e4], [1.2e5, 1.9e5], [2e6, 8e7], [3e9, 7e9], [7, 3], [0, 5]]}
# Numbers as JavaScript writes them (`String`, `toLocaleString`, `Math.round`), given as the strings `Number` reads.
WIDGET_NUMBERS = ["0", "-0", "1", "-1", "100", "1000", "0.1", "1234.5678", "-1234567.25", "0.0005", "0.0015", "0.0025", "-0.0001", "999.9995", "123456789.0005", "0.30000000000000004", "1e-6", "1e-7", "123e-20", "5e-324", "0.49999999999999994", "-0.4", "-0.5", "2.5", "-2.5", "184.99999999999994", "9007199254740993", "4099680000000000000", "18446744073709551615", "1e21", "1e+100", "1.5e300", "NaN", "Infinity", "-Infinity"]
WIDGET_SCATTERS = [[EST + "noT"], [EST + "frontier_small_gate_ns_e3"], [EST + "small|qubit_gate_ns_e3|surface_code", EST + "small|qubit_gate_ns_e4|surface_code", EST + "small|qubit_maj_ns_e4|floquet_code"], [EST + "frontier_rot_maj_floquet", EST + "custom_maj", EST + "budget_parts"], [EST + "maxDuration", EST + "maxDuration_tight", EST + "maxPhysicalQubits", EST + "maxTFactories1_rot", REV + "power|100", REV + "formula|functions", EST + "big|qubit_gate_us_e3|surface_code"], [EST + "oneT", EST + "oneT"], [REV + "formula|nan_is_zero_qubits", EST + "tiny_us"]]


def widget_cases(results: Dict[str, Dict[str, Any]]) -> dict:
    """The inputs the widget's charts are drawn for: color maps of 1 to 12 colors, tick ranges, every single
    result's space chart (a frontier's at its first, second, last and an out-of-range entry) and scatters."""
    space = [[key, 0] for key in SINGLES]
    for key in SINGLES:
        if "frontier" in key:
            space += [[key, 1], [key, len(materialize(key, results)["frontierEntries"]) - 1], [key, 99]]
    return {"colors": list(range(1, 13)), "time_ticks": WIDGET_RANGES["time"], "integer_ticks": WIDGET_RANGES["integer"], "numbers": WIDGET_NUMBERS, "space": space, "scatter": WIDGET_SCATTERS}


def widget_drawings(results: Dict[str, Dict[str, Any]], bundle: str) -> dict:
    """The widget's own output for `widget_cases`: colors, ticks, and each chart's element tree (tag, attributes,
    children; the attributes as the DOM gets them), from the bundle's functions run in node, in the en-US locale."""
    import os
    import subprocess

    cases = widget_cases(results)
    data = dict(cases, space=[[materialize(key, results), index] for key, index in cases["space"]], scatter=[materialize(keys, results) for keys in cases["scatter"]])
    env = {"PATH": os.environ["PATH"], "LC_ALL": "en_US.UTF-8", "LANG": "en_US.UTF-8"}
    run = subprocess.run(["node", "-e", NODE_HARNESS % widget_source(bundle)], input=json.dumps(data), capture_output=True, text=True, check=True, env=env, timeout=120)
    node = subprocess.run(["node", "--version"], capture_output=True, text=True, check=True, env=env).stdout.strip()
    return dict(cases, drawn=json.loads(run.stdout), node=node)


def recorded_results() -> Dict[str, Dict[str, Any]]:
    """The QDK's whole result for every case of qdk_estimates.json and qdk_review2.json it solves, re-run."""
    from qdk.estimator import LogicalCounts

    out: Dict[str, Dict[str, Any]] = {}
    for name in ("qdk_estimates", "qdk_review2"):
        cases = json.loads((PORT / f"{name}.json").read_text(encoding="utf-8"))
        out[name] = {}
        for key in sorted(cases):
            if cases[key]["status"] == "success":
                out[name][key] = json.loads(LogicalCounts(cases[key]["logicalCounts"]).estimate(cases[key]["jobParams"]).json)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--markdown", help="a directory with markdown and python-markdown-math installed (pip install --target)")
    args = parser.parse_args()
    if args.markdown:
        sys.path.insert(0, args.markdown)
    import qdk
    import qdk.estimator as ns
    from qdk.estimator import _estimator

    version = getattr(qdk, "__version__", None) or __import__("importlib.metadata").metadata.version("qdk")
    assert version == QDK_VERSION, version
    if args.markdown:
        assert _estimator.has_markdown, "markdown and mdx_math do not import from --markdown"
    python = ".".join(platform.python_version_tuple()[:2])

    params = {"qdk": QDK_VERSION, "python": python, "names": qdk_names(ns), "cases": [dict(case, qdk=run_param_case(case, ns)) for case in PARAM_CASES]}
    (PORT / "qdk_params.json").write_text(json.dumps(params, indent=0, ensure_ascii=False) + "\n", encoding="utf-8")

    results = recorded_results()
    renderings: Dict[str, Any] = {"qdk": QDK_VERSION, "python": python, "cases": REPORT_CASES}
    lines: Dict[str, int] = {}
    has_markdown = _estimator.has_markdown
    try:
        _estimator.has_markdown = False
        renderings["plain"] = pack({case["name"]: render_case(case, results, ns.EstimatorResult, ns.EstimatorError) for case in REPORT_CASES}, lines)
        if has_markdown:
            import markdown
            import mdx_math  # noqa: F401

            _estimator.has_markdown = True
            renderings["markdown_versions"] = {"markdown": markdown.__version__, "python-markdown-math": __import__("importlib.metadata").metadata.version("python-markdown-math")}
            renderings["markdown"] = pack({case["name"]: render_case(case, results, ns.EstimatorResult, ns.EstimatorError, diagrams=False) for case in REPORT_CASES}, lines)
    finally:
        _estimator.has_markdown = has_markdown
    renderings["lines"] = list(lines)
    import importlib.metadata
    import importlib.util

    # Read as data, not imported (it needs anywidget): the bundle the widgets load.
    assert importlib.metadata.version("qsharp_widgets") == QDK_VERSION
    bundle = (pathlib.Path(importlib.util.find_spec("qsharp_widgets").origin).parent / "static" / "index.js").read_text(encoding="utf-8")
    widget = widget_drawings(results, bundle)
    text = json.dumps(dict(results, renderings=renderings, widget=widget), ensure_ascii=False)
    (PORT / "qdk_reports.json.gz").write_bytes(gzip.compress(text.encode(), compresslevel=9, mtime=0))
    print(f"{len(PARAM_CASES)} parameter cases, {len(REPORT_CASES)} results drawn ({'with and without' if has_markdown else 'without'} markdown)")


if __name__ == "__main__":
    main()

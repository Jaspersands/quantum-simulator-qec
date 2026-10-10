"""Stim's whole Python surface exists here: every public name, every method of every class,
and every parameter Stim's signatures name (positional-only dunder arguments aside)."""

from __future__ import annotations

import inspect
import pickle
import re

import pytest

import stabilizer_qec as sq

stim = pytest.importorskip("stim", minversion="1.16")

DUNDERS = {"__init__", "__call__", "__getitem__", "__setitem__", "__len__", "__iter__", "__hash__", "__add__", "__iadd__", "__mul__", "__imul__", "__rmul__", "__pow__", "__ipow__", "__truediv__", "__itruediv__", "__neg__", "__pos__", "__str__", "__repr__", "__contains__"}
# Pickling: Stim's objects define __getstate__/__setstate__; these use __reduce__ (checked below).
PICKLED = {"Circuit", "CompiledMeasurementsToDetectionEventsConverter", "DetectorErrorModel", "PauliString", "Tableau"}
POSITIONAL = {"arg0", "second", "num_qubits"}  # pybind's names for positional-only operands


def stim_params(obj) -> list:
    doc = obj.__doc__ or ""
    sigs = [line for line in doc.splitlines() if line.startswith("@signature def ")]
    line = sigs[-1] if sigs else (doc.splitlines()[0] if doc else "")
    m = re.search(r"\((.*)\)\s*(->.*)?:?\s*$", line)
    if not m:
        return []
    params, depth, cur = [], 0, ""
    for ch in m.group(1):
        depth += ch in "([{"
        depth -= ch in ")]}"
        if ch == "," and depth == 0:
            params.append(cur)
            cur = ""
        else:
            cur += ch
    params.append(cur)
    names = [p.strip().split(":")[0].split("=")[0].strip().lstrip("*") for p in params]
    return [n for n in names if n and n not in ("*", "/") and not n.startswith("self") and n not in POSITIONAL]


def our_params(obj):
    try:
        sig = inspect.signature(obj)
    except (TypeError, ValueError):
        return None
    if any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values()):
        return None
    return list(sig.parameters)


PUBLIC = sorted(n for n in dir(stim) if not n.startswith("_"))


@pytest.mark.parametrize("name", PUBLIC)
def test_name_members_and_parameters(name):
    assert hasattr(sq, name), f"stabilizer_qec has no {name}"
    s, o = getattr(stim, name), getattr(sq, name)
    problems = []
    if inspect.isclass(s):
        for m in dir(s):
            if m.startswith("_") and m not in DUNDERS:
                continue
            if not hasattr(o, m):
                problems.append(f"no {name}.{m}")
                continue
            sm = getattr(s, m)
            if callable(sm) and not isinstance(sm, property):
                ours = our_params(getattr(o, m))
                if ours is not None:
                    problems += [f"{name}.{m} has no parameter {p}" for p in stim_params(sm) if p not in ours]
    elif callable(s):
        ours = our_params(o)
        if ours is not None:
            problems += [f"{name} has no parameter {p}" for p in stim_params(s) if p not in ours]
    assert not problems, problems


def test_the_pickled_classes_round_trip():
    things = [sq.Circuit("H 0\nCX 0 1\nM 0 1\nDETECTOR rec[-1]"), sq.DetectorErrorModel("error(0.1) D0 L0"), sq.PauliString("-iXYZ"), sq.Tableau.random(5), sq.Tableau.from_named_gate("S_DAG")]
    things.append(sq.Circuit("M 0\nDETECTOR rec[-1]").compile_m2d_converter())
    for t in things:
        back = pickle.loads(pickle.dumps(t))
        assert type(back) is type(t) and (repr(back) == repr(t) or back == t)
    assert {type(t).__name__ for t in things} | {"CompiledMeasurementsToDetectionEventsConverter"} >= {"Circuit", "DetectorErrorModel", "PauliString", "Tableau"}

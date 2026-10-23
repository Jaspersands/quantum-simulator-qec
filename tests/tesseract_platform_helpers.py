"""Checks for the recorded Apple-silicon references and native-platform outputs."""

import math
import re
import sys

import pytest


def assert_dem_text(actual, expected):
    # Stim uses long double's digits10 + 1: 16 on Apple silicon/Windows, 19 on
    # x86-64 Unix, 34 on ARM64 Linux (src/dem_program.rs). Keep every instruction,
    # target, tag and coordinate in order; only numeric argument spelling can vary.
    if sys.platform == "darwin":
        assert actual == expected
        return
    args = r"\(([-+0-9.eE, ]+)\)"
    assert re.sub(args, "(ARGS)", actual) == re.sub(args, "(ARGS)", expected)
    a, b = re.findall(args, actual), re.findall(args, expected)
    assert len(a) == len(b)
    for aa, bb in zip(a, b):
        av, bv = [float(x) for x in aa.split(",")], [float(x) for x in bb.split(",")]
        assert len(av) == len(bv)
        for x, y in zip(av, bv):
            # The frozen 16-digit spelling can round a binary64 by a few ulps;
            # circuit analysis can add last-bit arithmetic differences. Zero stays exact.
            assert math.isfinite(x) and math.isfinite(y)
            assert abs(x - y) <= (8 * math.ulp(y) if y else 0), (x, y)


def faults(dem):
    """Read probabilities and parity columns independently of the decoder's search."""
    out = []
    for inst in dem.flattened():
        if inst.type != "error":
            continue
        dets, obs = set(), set()
        for t in inst.targets_copy():
            if t.is_relative_detector_id():
                dets.symmetric_difference_update([t.val])
            elif t.is_logical_observable_id():
                obs.symmetric_difference_update([t.val])
        p = inst.args_copy()[0]
        out.append((math.log((1 - p) / p), dets, obs))
    return out


def correction(columns, errors, fired):
    residual, obs, cost = set(fired), set(), 0.0
    assert len(errors) == len(set(errors))
    for e in errors:
        assert 0 <= e < len(columns)
        c, d, o = columns[e]
        residual.symmetric_difference_update(d)
        obs.symmetric_difference_update(o)
        cost += c
    return residual, sorted(obs), cost


def assert_search_log(actual, expected, dem, fired):
    """Heap ties and generated orders vary with the native C++ library. Check every
    logged chain's parity, each completed correction's cost/observables, and the
    recorded per-order outcomes, without freezing intermediate queue order."""
    columns = faults(dem)
    lines = actual.splitlines()
    summaries = []
    residual = obs = cost = None
    num_dets = None
    completed = False
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("len(pq)"):
            m = re.fullmatch(r"len\(pq\) = (\d+) num_pq_pushed = (\d+)", line)
            assert m and 0 <= int(m[1]) < int(m[2])
            i += 1
            m = re.fullmatch(r"num_dets = (\d+) max_num_dets = (\d+) cost = ([0-9.e+-]+)", lines[i])
            assert m and int(m[1]) <= int(m[2]) and math.isfinite(float(m[3]))
            num_dets = int(m[1])
        elif line.startswith("activated_errors = "):
            assert re.fullmatch(r"activated_errors = (?:\d+, )*", line)
            errors = [int(x) for x in re.findall(r"\d+", line)]
            residual, obs, cost = correction(columns, errors, fired)
            i += 1
            assert lines[i] == "activated_detectors = " + "".join(f"{d}, " for d in sorted(residual))
            if num_dets is not None:
                assert num_dets == len(residual)
                num_dets = None
        elif line.startswith("Decoding complete."):
            m = re.fullmatch(r"Decoding complete\. Cost: ([0-9.e+-]+) num_pq_pushed = (\d+)", line)
            assert m and residual == set() and int(m[2]) > 0
            assert float(m[1]) == pytest.approx(cost, rel=1e-12, abs=0)
            completed = True
        else:
            m = re.fullmatch(r"for detector_order (\d+) beam (\d+) got low confidence (\d+) and cost ([0-9.e+-]+) and obs_mask \[([\d, ]*)\]\. Best cost so far: ([0-9.e+-]+)", line)
            assert m and completed and m[3] == "0", line
            assert [int(x) for x in re.findall(r"\d+", m[5])] == obs
            assert float(m[4]) == pytest.approx(cost, rel=1e-12, abs=0)
            summaries.append(line)
            completed = False
        i += 1
    assert summaries == [s for s in expected.splitlines() if s.startswith("for detector_order ")]
    assert not completed and num_dets is None and summaries

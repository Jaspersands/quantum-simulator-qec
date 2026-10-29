"""The circuit distance of every memory the package builds, three ways: the graph-like search
(`shortest_graphlike_error`, graph-like faults only), Stim's search through hyperedges
(`search_for_undetectable_logical_errors`, an upper bound), and the integer program
(`DetectorErrorModel.distance()`, exact when it finishes within its time limit; otherwise the
best solution it found, a real undetectable logical error, is recorded as an upper bound).

    python tools/distances.py --time-limit 300   # compute, write data/distances.json
    python tools/distances.py --quick    # the small ones only, nothing written

Each memory runs d rounds (bivariate bicycle and CSS memories: their stated rounds) under
circuit noise; the noise strength does not change which faults exist, only their weights.

Stim's search keeps every set of detectors it visits; ours keeps them within a memory budget
(`--max-memory-bytes`, 1 GB by default, a fixed number so the column does not depend on the
machine's memory) and records why it stopped when the budget runs out first. The file is
stamped with the package's version and the platform it ran on: tools/readme_tables.py --check
refuses a recording older than the package's major.minor.
"""

from __future__ import annotations

import argparse
import datetime
import json
import pathlib
import platform
import re
import time

import stabilizer_qec as sq

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "distances.json"
SEARCH = dict(dont_explore_detection_event_sets_with_size_above=6, dont_explore_edges_with_degree_above=6, dont_explore_edges_increasing_symptom_degree=False)


def stim_like(task, d):
    return sq.Circuit.generated(
        task, distance=d, rounds=d, after_clifford_depolarization=0.001, before_round_data_depolarization=0.001, before_measure_flip_probability=0.001, after_reset_flip_probability=0.001
    )


def memories(quick):
    hamming = [[1, 0, 1, 0, 1, 0, 1], [0, 1, 1, 0, 0, 1, 1], [0, 0, 0, 1, 1, 1, 1]]
    yield "surface, rotated (Stim's)", 3, lambda: stim_like("surface_code:rotated_memory_z", 3)
    yield "surface, rotated (Stim's)", 5, lambda: stim_like("surface_code:rotated_memory_z", 5)
    yield "surface, unrotated (Stim's)", 3, lambda: stim_like("surface_code:unrotated_memory_z", 3)
    yield "colour code XYZ (Stim's)", 3, lambda: stim_like("color_code:memory_xyz", 3)
    yield "colour code XYZ (Stim's)", 5, lambda: stim_like("color_code:memory_xyz", 5)
    yield "surface, rotated, SD6", 3, lambda: sq.memory_circuit("rotated", distance=3, rounds=3, p=0.001)
    yield "surface, XZZX, SD6", 3, lambda: sq.memory_circuit("xzzx", distance=3, rounds=3, p=0.001)
    yield "colour code 6.6.6 (CssCode)", 3, lambda: sq.CssCode.color_code(3).memory_circuit(3, 0.001)
    yield "Hamming hypergraph product [[58, 16, 3]] (CssCode)", 3, lambda: sq.CssCode.hypergraph_product(hamming).memory_circuit(3, 0.001)
    yield "bivariate bicycle [[72, 12, 6]]", 6, lambda: sq.BivariateBicycleCode("72").memory_circuit(2, 0.001)
    if quick:
        return
    yield "surface, rotated (Stim's)", 7, lambda: stim_like("surface_code:rotated_memory_z", 7)
    yield "surface, unrotated (Stim's)", 5, lambda: stim_like("surface_code:unrotated_memory_z", 5)
    yield "colour code XYZ (Stim's)", 7, lambda: stim_like("color_code:memory_xyz", 7)
    yield "surface, rotated, SD6", 5, lambda: sq.memory_circuit("rotated", distance=5, rounds=5, p=0.001)
    yield "surface, XZZX, SD6", 5, lambda: sq.memory_circuit("xzzx", distance=5, rounds=5, p=0.001)
    yield "colour code 6.6.6 (CssCode)", 5, lambda: sq.CssCode.color_code(5).memory_circuit(5, 0.001)
    yield "bivariate bicycle [[90, 8, 10]]", 10, lambda: sq.BivariateBicycleCode("90").memory_circuit(2, 0.001)
    yield "bivariate bicycle [[108, 8, 10]]", 10, lambda: sq.BivariateBicycleCode("108").memory_circuit(2, 0.001)
    yield "bivariate bicycle [[144, 12, 12]] (gross)", 12, lambda: sq.BivariateBicycleCode("gross").memory_circuit(2, 0.001)


def measure(circuit, time_limit, max_memory_bytes):
    out = {}
    dem = circuit.detector_error_model(decompose_errors=True, ignore_decomposition_failures=True)
    try:
        out["graphlike"] = dem.shortest_graphlike_error().num_errors
    except ValueError:
        out["graphlike"] = None
    try:
        out["search"] = len(circuit.search_for_undetectable_logical_errors(**SEARCH, max_memory_bytes=max_memory_bytes))
    except ValueError as ex:
        # Nothing found within the limits, or the memory budget ran out first: the message says which.
        out["search"] = None
        out["search_note"] = str(ex)
    t0 = time.perf_counter()
    try:
        out["exact"] = circuit.detector_error_model().distance(time_limit=time_limit)
        out["proven"] = True
    except RuntimeError as ex:
        out["exact"], out["proven"] = None, False
        out["note"] = str(ex)
        # A solution found before the time limit is a real undetectable logical error: a bound.
        found = re.search(r"best found (\d+)", str(ex))
        out["upper"] = int(found.group(1)) if found else None
    out["milp_seconds"] = round(time.perf_counter() - t0, 1)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--time-limit", type=float, default=600.0)
    ap.add_argument("--max-memory-bytes", type=int, default=1_000_000_000, help="the search's memory budget (default 1 GB)")
    args = ap.parse_args()
    rows = []
    for name, d, make in memories(args.quick):
        c = make()
        r = dict(memory=name, code_distance=d, faults=c.detector_error_model().num_errors, **measure(c, args.time_limit, args.max_memory_bytes))
        print(f"{name:52} d={d:2}  graph-like {r['graphlike']}  search {r['search']}  exact {r['exact']}  ({r['milp_seconds']} s)", flush=True)
        rows.append(r)
    if not args.quick:
        doc = dict(version=sq.__version__, platform=f"{platform.system()} {platform.machine()}", generated=datetime.date.today().isoformat(),
                   search_limits=dict(SEARCH, max_memory_bytes=args.max_memory_bytes), time_limit=args.time_limit, memories=rows)
        OUT.write_text(json.dumps(doc, indent=1) + "\n")
        print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

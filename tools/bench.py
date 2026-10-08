"""The benchmarks: this package against Stim and PyMatching, on the same machine and one thread.

Each benchmark times one job here and the same job in the reference, and records the ratio
(this package's time over the reference's; below 1 is faster). Ratios carry from machine to
machine far better than times do, so CI checks every push's ratios against the last recorded
run, and the site's benchmarks page plots the runs recorded at each release.

    python tools/bench.py                       # run, print the table
    python tools/bench.py --record              # run, and append the run to data/bench/runs.json
    python tools/bench.py --check               # run; fail if a ratio is worse than the last
                                                # recorded run's by more than --tolerance
    python tools/bench.py --summary FILE        # also write the table as Markdown (CI's job summary)
    python tools/bench.py --add RUN.json        # record a run made elsewhere (CI's, from --json)
    python tools/bench.py --only error_model    # just these benchmarks (comma-separated keys)

Runs are compared with the last recorded run on the same kind of machine (system and
processor), so CI's Linux runners are held to a Linux run.

Needs the package, numpy, stim and pymatching.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import pathlib
import platform
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
RUNS = ROOT / "data" / "bench" / "runs.json"


def best(fn, repeats):
    """The fastest of `repeats` runs of `fn`, in seconds (after one untimed warm-up)."""
    fn()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return min(times)


def memory(sq, stim, d, p):
    kw = dict(distance=d, rounds=d, after_clifford_depolarization=p, before_round_data_depolarization=p, before_measure_flip_probability=p, after_reset_flip_probability=p)
    return sq.Circuit.generated("surface_code:rotated_memory_z", **kw), stim.Circuit.generated("surface_code:rotated_memory_z", **kw)


def benchmarks(sq, stim, pymatching, quick):
    """[(key, description, unit count, unit, ours(), theirs() or None)]."""
    shots = 2_000 if quick else 20_000
    ours_c, stim_c = memory(sq, stim, 11, 0.001)
    ours_dem = ours_c.detector_error_model(decompose_errors=True)
    stim_dem = stim_c.detector_error_model(decompose_errors=True)
    dets, _ = stim_c.compile_detector_sampler(seed=1).sample(shots, separate_observables=True)
    pm = pymatching.Matching.from_detector_error_model(stim_dem)
    mine = sq.Matching(ours_dem)
    corr_mine = sq.Matching(ours_dem, enable_correlations=True)
    corr_pm = pymatching.Matching.from_detector_error_model(stim_dem, enable_correlations=True)
    measurements = stim_c.compile_sampler(seed=2).sample(shots)
    ours_m2d, stim_m2d = ours_c.compile_m2d_converter(), stim_c.compile_m2d_converter()
    ours_s, stim_s = ours_c.compile_detector_sampler(seed=3), stim_c.compile_detector_sampler(seed=3)
    uf = sq.UnionFind(ours_dem)
    gross = sq.BivariateBicycleCode("gross").memory_circuit(6, 0.003)
    gross_dem = gross.detector_error_model()
    gross_dets, _ = gross.compile_detector_sampler(seed=4).sample(50 if quick else 300, separate_observables=True)
    bposd = sq.BpOsd(gross_dem, max_iter=100, osd_order=7)
    return [
        ("error_model", "Error model of a d = 11 rotated memory, 11 rounds (decomposed)", 1, "model",
         lambda: ours_c.detector_error_model(decompose_errors=True), lambda: stim_c.detector_error_model(decompose_errors=True)),
        ("shortest_graphlike", "Its graph-like distance (shortest_graphlike_error on the model)", 1, "model",
         lambda: ours_dem.shortest_graphlike_error(), lambda: stim_dem.shortest_graphlike_error()),
        ("sample", f"Sampling its detection events, {shots:,} shots", shots, "shot",
         lambda: ours_s.sample(shots, separate_observables=True), lambda: stim_s.sample(shots, separate_observables=True)),
        ("m2d", f"Measurements to detection events, {shots:,} shots", shots, "shot",
         lambda: ours_m2d.convert(measurements=measurements, separate_observables=True),
         lambda: stim_m2d.convert(measurements=measurements, separate_observables=True)),
        ("matching", f"Matching at p = 0.1%, {shots:,} shots (against PyMatching)", shots, "shot",
         lambda: mine.decode_batch(dets), lambda: pm.decode_batch(dets)),
        ("correlated_matching", f"Correlated matching, {shots:,} shots (against PyMatching's)", shots, "shot",
         lambda: corr_mine.decode_batch(dets), lambda: corr_pm.decode_batch(dets, enable_correlations=True)),
        ("union_find", f"Union-find on the same shots", shots, "shot", lambda: uf.decode_batch(dets), None),
        # Stim's sampler and PyMatching's batch decoding run on one thread: these are ours alone.
        ("sample@4", f"Sampling, {shots:,} shots, on 4 threads", shots, "shot", lambda: ours_s.sample(shots, separate_observables=True, threads=4), None),
        ("sample@all", f"Sampling, {shots:,} shots, on every core", shots, "shot", lambda: ours_s.sample(shots, separate_observables=True, threads=0), None),
        ("matching@4", f"Matching, {shots:,} shots, on 4 threads", shots, "shot", lambda: mine.decode_batch(dets, threads=4), None),
        ("matching@all", f"Matching, {shots:,} shots, on every core", shots, "shot", lambda: mine.decode_batch(dets, threads=0), None),
        ("correlated_matching@all", f"Correlated matching, {shots:,} shots, on every core", shots, "shot", lambda: corr_mine.decode_batch(dets, threads=0), None),
        ("bposd_gross", f"BP+OSD on the gross code, 6 cycles at p = 0.3%, {len(gross_dets)} shots", len(gross_dets), "shot",
         lambda: bposd.decode_batch(gross_dets), None),
    ]


def kind():
    return f"{platform.system()} {platform.machine().lower().replace('amd64', 'x86_64')}"


def run(quick, repeats, only=None):
    import pymatching
    import stim

    import stabilizer_qec as sq

    results = {}
    for key, what, units, unit, ours, theirs in benchmarks(sq, stim, pymatching, quick):
        if only and key not in only:
            continue
        t_ours = best(ours, repeats)
        t_ref = best(theirs, repeats) if theirs else None
        results[key] = dict(
            what=what,
            unit=unit,
            ours_us=1e6 * t_ours / units,
            reference_us=None if t_ref is None else 1e6 * t_ref / units,
            ratio=None if t_ref is None else t_ours / t_ref,
        )
        r = results[key]
        ref = f"{r['reference_us']:10.2f}" if t_ref else f"{'':>10}"
        ratio = f"{r['ratio']:6.2f}" if t_ref else f"{'':>6}"
        print(f"{key:24} {r['ours_us']:10.2f} {ref} {ratio}   µs per {unit}", flush=True)
    return dict(
        version=sq.__version__,
        date=datetime.date.today().isoformat(),
        machine=f"{platform.system()} {platform.machine()}, Python {platform.python_version()}",
        cores=os.cpu_count(),
        kind=kind(),
        references=dict(stim=stim.__version__, pymatching=pymatching.__version__),
        quick=quick,
        results=results,
    )


def markdown(run_, last):
    lines = [
        f"### Benchmarks: stabilizer-qec {run_['version']} against Stim {run_['references']['stim']} and PyMatching {run_['references']['pymatching']}",
        "",
        f"{run_['machine']}; one thread; ratio = this package's time / the reference's (below 1 is faster).",
        "",
        "| benchmark | ours (µs) | reference (µs) | ratio | last recorded ratio |",
        "|---|---:|---:|---:|---:|",
    ]
    for key, r in run_["results"].items():
        before = (last or {}).get("results", {}).get(key, {}).get("ratio")
        cell = lambda v, f: "" if v is None else f.format(v)  # noqa: E731
        lines.append(f"| {r['what']} | {cell(r['ours_us'], '{:.2f}')} | {cell(r['reference_us'], '{:.2f}')} | {cell(r['ratio'], '{:.2f}')} | {cell(before, '{:.2f}')} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--record", action="store_true", help="append this run to data/bench/runs.json")
    ap.add_argument("--check", action="store_true", help="fail if a ratio regressed against the last recorded run")
    ap.add_argument("--tolerance", type=float, default=2.0, help="how many times worse a ratio may be (default 2: CI's machines vary)")
    ap.add_argument("--quick", action="store_true", help="fewer shots")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--summary", help="write the table as Markdown here")
    ap.add_argument("--json", help="write this run as JSON here")
    ap.add_argument("--add", help="record this run (a --json file) without running")
    ap.add_argument("--only", help="comma-separated benchmark keys to run (not with --record)")
    args = ap.parse_args()

    runs = json.loads(RUNS.read_text(encoding="utf-8")) if RUNS.exists() else []
    if args.add:
        runs.append(json.loads(pathlib.Path(args.add).read_text(encoding="utf-8")))
        RUNS.parent.mkdir(parents=True, exist_ok=True)
        RUNS.write_text(json.dumps(runs, indent=1) + "\n", encoding="utf-8")
        print(f"recorded {runs[-1]['version']} ({runs[-1]['machine']}) in {RUNS.relative_to(ROOT)}")
        return 0
    same = [r for r in runs if r.get("kind") == kind()]
    last = same[-1] if same else None
    print(f"{'benchmark':24} {'ours':>10} {'reference':>10} {'ratio':>6}")
    if args.only and args.record:
        ap.error("--only runs part of the suite, which is not a run to record")
    this = run(args.quick, args.repeats, set(args.only.split(",")) if args.only else None)
    if args.summary:
        with open(args.summary, "a", encoding="utf-8") as f:
            f.write(markdown(this, last))
    if args.json:
        pathlib.Path(args.json).write_text(json.dumps(this, indent=1) + "\n", encoding="utf-8")
    if args.record:
        runs.append(this)
        RUNS.parent.mkdir(parents=True, exist_ok=True)
        RUNS.write_text(json.dumps(runs, indent=1) + "\n", encoding="utf-8")
        print(f"recorded in {RUNS.relative_to(ROOT)}")
    if args.check:
        if last is None:
            print(f"no run recorded on a {kind()} machine: nothing to check against (record one with --add)")
            return 0
        worse = []
        for key, r in this["results"].items():
            before = last["results"].get(key, {}).get("ratio")
            if r["ratio"] is not None and before is not None and r["ratio"] > args.tolerance * before:
                worse.append(f"{key}: ratio {r['ratio']:.2f}, recorded {before:.2f}")
        if worse:
            print("REGRESSED (more than", args.tolerance, "times the recorded ratio):\n  " + "\n  ".join(worse))
            return 1
        print(f"every ratio within {args.tolerance}× of the run recorded for {last['version']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

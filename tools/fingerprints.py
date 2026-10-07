"""Fingerprints of every result the package computes, so that work on its speed cannot change
one silently: error-model text, shots for a seed (at any thread count), raw measurements,
model samples, measurement conversion, every decoder's predictions and the distance searches,
each hashed (sha256) and recorded in data/fingerprints.json.

    python tools/fingerprints.py            # compute, write data/fingerprints.json
    python tools/fingerprints.py --check    # recompute; exit 1 on any difference

The same file must hold on every platform: CI checks it on Linux, macOS and Windows
(tests/test_fingerprints.py), which also checks that a seed gives the same shots everywhere.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
from typing import Callable, Dict

import numpy as np

import stabilizer_qec as sq

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "fingerprints.json"
SEEDS = (0, 1, 2**63 + 5)
SHOTS = (1, 63, 64, 65, 1000, 4097)


def digest(*parts) -> str:
    h = hashlib.sha256()
    for p in parts:
        if isinstance(p, str):
            h.update(p.encode())
        else:
            a = np.ascontiguousarray(p)
            h.update(str((a.dtype.str, a.shape)).encode())
            h.update(a.tobytes())
    return h.hexdigest()


def surface():
    return sq.Circuit.generated("surface_code:rotated_memory_x", distance=5, rounds=5, after_clifford_depolarization=0.004,
                                before_round_data_depolarization=0.002, before_measure_flip_probability=0.003, after_reset_flip_probability=0.001)


def gross():
    return sq.BivariateBicycleCode("gross").memory_circuit(2, 0.003)


def colour():
    return sq.Circuit.generated("color_code:memory_xyz", distance=5, rounds=5, after_clifford_depolarization=0.001)


def cases() -> Dict[str, Callable[[], str]]:
    """Name -> a function computing that result's fingerprint."""
    out: Dict[str, Callable[[], str]] = {}
    circuits = {"surface": surface, "gross": gross}
    for name, make in circuits.items():
        c = make()
        out[f"{name}/model"] = lambda c=c: digest(str(c.detector_error_model()))
        out[f"{name}/model/flat"] = lambda c=c: digest(str(c.detector_error_model(flatten_loops=True)))
        out[f"{name}/model/disjoint"] = lambda c=c: digest(str(c.detector_error_model(approximate_disjoint_errors=True)))
        if name == "surface":
            out[f"{name}/model/decomposed"] = lambda c=c: digest(str(c.detector_error_model(decompose_errors=True)))
        for seed in SEEDS:
            for n in SHOTS:
                for threads in (1, 3, 0):
                    out[f"{name}/detect/seed{seed}/n{n}/t{threads}"] = lambda c=c, s=seed, n=n, t=threads: digest(
                        *c.compile_detector_sampler(seed=s).sample(n, separate_observables=True, threads=t))
            out[f"{name}/measure/seed{seed}"] = lambda c=c, s=seed: digest(c.compile_sampler(seed=s).sample(1000))
            out[f"{name}/measure/seed{seed}/t0"] = lambda c=c, s=seed: digest(c.compile_sampler(seed=s).sample(1000, threads=0))
            out[f"{name}/dem_sample/seed{seed}"] = lambda c=c, s=seed: digest(
                *c.detector_error_model().compile_sampler(seed=s).sample(1000, return_errors=True))
        out[f"{name}/m2d"] = lambda c=c: digest(*c.compile_m2d_converter().convert(measurements=c.compile_sampler(seed=7).sample(1000), separate_observables=True))
        out[f"{name}/m2d/appended"] = lambda c=c: digest(c.compile_m2d_converter().convert(measurements=c.compile_sampler(seed=7).sample(1000), append_observables=True))

    c = surface()
    dem = c.detector_error_model(decompose_errors=True)
    dets, _ = c.compile_detector_sampler(seed=11).sample(2000, separate_observables=True)
    decoders = {
        "matching": lambda: sq.Matching(dem),
        "correlated": lambda: sq.Matching(dem, enable_correlations=True),
        "union_find": lambda: sq.UnionFind(dem),
        "belief_matching": lambda: sq.BeliefMatching(dem),
        "bposd": lambda: sq.BpOsd(c.detector_error_model(), max_iter=30, osd_order=4),
    }
    for name, make in decoders.items():
        for threads in (1, 0):
            out[f"surface/decode/{name}/t{threads}"] = lambda make=make, t=threads: digest(make().decode_batch(dets, threads=t))
    g = gross()
    gdets, _ = g.compile_detector_sampler(seed=12).sample(200, separate_observables=True)
    out["gross/decode/bposd"] = lambda: digest(sq.BpOsd(g.detector_error_model(), max_iter=100, osd_order=7).decode_batch(gdets, threads=0))

    search = dict(dont_explore_detection_event_sets_with_size_above=6, dont_explore_edges_with_degree_above=6, dont_explore_edges_increasing_symptom_degree=False)
    small = lambda: sq.Circuit.generated("surface_code:rotated_memory_x", distance=3, rounds=3, after_clifford_depolarization=0.004)  # noqa: E731
    for name, make in (("surface_d3", small), ("colour", colour)):
        out[f"{name}/graphlike"] = lambda make=make: digest(str(make().detector_error_model(decompose_errors=True, ignore_decomposition_failures=True).shortest_graphlike_error()))
        out[f"{name}/search"] = lambda make=make: digest(str(make().detector_error_model().search_for_undetectable_logical_errors(**search)))
    return out


def compute(names=None) -> Dict[str, str]:
    cs = cases()
    return {k: cs[k]() for k in (names or cs)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    got = compute()
    if args.check:
        want = json.loads(OUT.read_text(encoding="utf-8"))
        bad = sorted(k for k in set(want) | set(got) if want.get(k) != got.get(k))
        for k in bad:
            print(f"DIFFERS: {k}")
        print(f"{len(got) - len(bad)} of {len(got)} fingerprints unchanged")
        return 1 if bad else 0
    OUT.write_text(json.dumps(got, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {len(got)} fingerprints to {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

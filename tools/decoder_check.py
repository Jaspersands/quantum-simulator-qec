"""The 1.8 decoders against their authors' packages, shot for shot.

Run from the repository root, with the package built and the references installed
(`pip install ldpc chromobius relay-bp` and `pip install --pre tesseract-decoder`):

    python tools/decoder_check.py                 # every check; writes data/decoders/check.json
    python tools/decoder_check.py --quick         # fewer shots, writes nothing (CI)
    python tools/decoder_check.py --only lsd      # some sections (comma-separated)

Sections (each skipped, and said so, when its reference is not installed):
  lsd     BP+LSD against ldpc's BpLsdDecoder on the same check matrix: random matrices, a
          d = 5 surface code's model and the gross code's (2 and 6 cycles at p = 0.3%), both BP
          methods. Every correction must equal ldpc's, except on shots where ours made a choice
          ldpc makes from an order that cannot be reproduced (equal keys in a long std::sort,
          or merging three or more clusters at once, which ldpc orders by hashed pointers);
          those are counted as ties. Also: each untied final cluster's bits, in the order
          ldpc's statistics list them (its robin-set order), must be listed in the same order.

Exits non-zero if any check fails.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np

import stabilizer_qec as sq

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "decoders" / "check.json"


def check_matrix(dem):
    """A model's faults as a dense check matrix and priors (Stim's flattening)."""
    import stim

    flat = stim.DetectorErrorModel(str(dem)).flattened()
    cols, priors = [], []
    for inst in flat:
        if inst.type == "error":
            dets = {}
            for t in inst.targets_copy():
                if t.is_relative_detector_id():
                    dets[t.val] = dets.get(t.val, 0) ^ 1
            cols.append(sorted(d for d, odd in dets.items() if odd))
            priors.append(inst.args_copy()[0])
    pcm = np.zeros((flat.num_detectors, len(cols)), dtype=np.uint8)
    for j, c in enumerate(cols):
        pcm[c, j] = 1
    return pcm, np.array(priors)


def lsd_cases(quick):
    rng = np.random.default_rng(5)
    for k in range(3 if quick else 10):
        pcm = (rng.random((24, 48)) < 0.12).astype(np.uint8)
        pcm[rng.integers(24, size=48), np.arange(48)] = 1
        channel = rng.uniform(0.01, 0.1, 48)
        syndromes = [(pcm @ (rng.random(48) < channel)) % 2 for _ in range(100)]
        yield f"random {k}", pcm, channel, np.array(syndromes, dtype=np.uint8), 3
    circuits = {
        "surface d=5": sq.Circuit.generated("surface_code:rotated_memory_z", distance=5, rounds=5, after_clifford_depolarization=0.005, before_measure_flip_probability=0.005),
        "gross, 2 cycles": sq.BivariateBicycleCode("gross").memory_circuit(2, 0.003),
        "gross, 6 cycles": sq.BivariateBicycleCode("gross").memory_circuit(6, 0.003),
    }
    for name, c in circuits.items():
        pcm, priors = check_matrix(c.detector_error_model())
        n = (100 if quick else 1000) if "gross" in name else (300 if quick else 3000)
        yield name, pcm, priors, c.compile_detector_sampler(seed=1).sample(n).astype(np.uint8), 30


def check_lsd(quick):
    import ldpc

    ok, rows = True, []
    for name, pcm, priors, syndromes, iters in lsd_cases(quick):
        for method in ("minimum_sum", "product_sum"):
            kw = dict(max_iter=iters, bp_method=method, ms_scaling_factor=0.625, lsd_method="lsd_0", lsd_order=0)
            theirs = ldpc.BpLsdDecoder(pcm, error_channel=list(priors), **kw)
            theirs.set_do_stats(True)
            ours = sq.BpLsdDecoder(pcm, error_channel=list(priors), **kw)
            row = dict(case=name, method=method, shots=len(syndromes), agree=0, tied=0, untied=0, clusters_in_order=0, clusters_out_of_order=0, ours_s=0.0, ldpc_s=0.0)
            for s in syndromes:
                t0 = time.perf_counter()
                b = theirs.decode(s)
                t1 = time.perf_counter()
                a = ours.decode(s)
                row["ldpc_s"] += t1 - t0
                row["ours_s"] += time.perf_counter() - t1
                if np.array_equal(a, b):
                    row["agree"] += 1
                elif ours.last_tied:
                    row["tied"] += 1
                else:
                    row["untied"] += 1
                if not theirs.converge:
                    stats = theirs.statistics["individual_cluster_stats"]
                    for cid, bits in ours._cluster_bits(s):
                        if list(stats[cid]["final_bits"]) == list(bits):
                            row["clusters_in_order"] += 1
                        else:
                            row["clusters_out_of_order"] += 1
            good = row["untied"] == 0 and row["clusters_out_of_order"] == 0
            ok &= good
            rows.append(row)
            print(f"  {name:16} {method:12} agree {row['agree']:5}  tied {row['tied']:3}  untied {row['untied']}  "
                  f"clusters in order {row['clusters_in_order']:5} (out {row['clusters_out_of_order']})  "
                  f"{1e3 * row['ours_s'] / len(syndromes):.3f} vs {1e3 * row['ldpc_s'] / len(syndromes):.3f} ms/shot  "
                  f"{'ok' if good else 'FAIL'}", flush=True)
    return ok, rows


SECTIONS = {"lsd": ("BP+LSD against ldpc", "ldpc", check_lsd)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    from importlib.metadata import PackageNotFoundError, version

    only = [x for x in args.only.split(",") if x]
    ok, results = True, {}
    for key, (title, package, run) in SECTIONS.items():
        if only and key not in only:
            continue
        try:
            print(f"{title} ({package} {version(package)})")
        except PackageNotFoundError:
            print(f"{title}: skipped, {package} is not installed")
            continue
        good, rows = run(args.quick)
        ok &= good
        results[key] = dict(reference=f"{package} {version(package)}", rows=rows)
    if not args.quick and not only:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(dict(version=sq.__version__, sections=results), indent=1) + "\n")
    print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

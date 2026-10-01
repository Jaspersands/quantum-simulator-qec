"""The sparse matcher's speed against PyMatching, and a fingerprint of every
prediction and weight, so a faster matcher can be shown to decide nothing
differently.

    .venv-f/bin/python tools/matcher_bench.py            # time and fingerprint -> data/matcher/bench.json
    .venv-f/bin/python tools/matcher_bench.py --check    # fingerprints only, against the recorded ones

Rotated SD6 memories, T = d, shots sampled by Stim with a fixed seed. Times are
single-threaded, the best of --reps runs; ours is decode_b8's own clock (the
decode loop), PyMatching's is decode_batch's wall time.
"""
import argparse, hashlib, json, pathlib, sys, time

import numpy as np
import pymatching
import stim
import stabilizer_qec._core as sq

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from realtime import engine_commit, machine  # noqa: E402

OUT = ROOT / "data" / "matcher" / "bench.json"
POINTS = [(3, 0.003), (3, 0.006), (5, 0.003), (5, 0.006), (7, 0.003), (7, 0.006), (9, 0.004), (11, 0.004)]


def shots_for(d):
    return 100_000 if d <= 7 else 20_000


def point(d, p, reps, with_pm):
    shots = shots_for(d)
    text = sq.generate_circuit("rotated", d, d, "sd6", p, 0.5, "z")
    circuit = stim.Circuit(text)
    dem = circuit.detector_error_model(decompose_errors=True)
    dets, _ = circuit.compile_detector_sampler(seed=7).sample(shots, separate_observables=True)
    packed = np.packbits(dets, axis=1, bitorder="little").tobytes()
    row = dict(d=d, p=p, shots=shots)
    for key, corr in (("plain", False), ("corr", True)):
        best, digest = None, None
        for _ in range(reps):
            pred, w, _, seconds = sq.decode_b8(str(dem), packed, shots, 1, corr)
            h = hashlib.sha256(pred + w).hexdigest()[:16]
            if digest not in (None, h):
                raise SystemExit(f"d={d} p={p} {key}: two runs decided differently")
            digest, best = h, seconds if best is None else min(best, seconds)
        entry = dict(ours_us=best / shots * 1e6, digest=digest)
        if with_pm:
            m = pymatching.Matching.from_detector_error_model(dem, enable_correlations=corr)
            pm = min(_timed(lambda: m.decode_batch(dets, enable_correlations=corr)) for _ in range(reps))
            entry.update(pm_us=pm / shots * 1e6, ratio=best / pm)
        row[key] = entry
    return row


def _timed(fn):
    t = time.perf_counter()
    fn()
    return time.perf_counter() - t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="fingerprints only, against data/matcher/bench.json")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out", type=pathlib.Path, default=OUT)
    args = ap.parse_args()
    if args.check:
        recorded = {(r["d"], r["p"]): r for r in json.loads(OUT.read_text())["points"]}
        bad = 0
        for d, p in POINTS:
            r = point(d, p, 1, False)
            for key in ("plain", "corr"):
                same = r[key]["digest"] == recorded[(d, p)][key]["digest"]
                bad += not same
                print(f"{'ok ' if same else 'BAD'} d={d:<2} p={p} {key:<5} {r[key]['digest']}", flush=True)
        print("ALL FINGERPRINTS MATCH" if not bad else f"{bad} FINGERPRINTS DIFFER")
        return 1 if bad else 0
    points = []
    for d, p in POINTS:
        r = point(d, p, args.reps, True)
        points.append(r)
        print(f"d={d:<2} p={p}  plain {r['plain']['ours_us']:7.2f} us ({r['plain']['ratio']:.2f}x)  "
              f"corr {r['corr']['ours_us']:7.2f} us ({r['corr']['ratio']:.2f}x)", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    doc = dict(generated=time.strftime("%Y-%m-%d"), engine_commit=engine_commit(), machine=machine(),
               pymatching=pymatching.__version__, stim=stim.__version__, points=points)
    args.out.write_text(json.dumps(doc, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

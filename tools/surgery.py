"""Lattice surgery: checked against Stim and PyMatching, and the timing law measured.

Run from the repository root, with the engine built into a virtualenv that
also has stim and pymatching:

    python tools/surgery.py check    # error models against Stim's, matching against PyMatching's
    python tools/surgery.py run      # failure rates against merged rounds -> data/surgery/results.json

The experiment (src/surgery.rs): two rotated distance-d patches in |0>_L, d
rounds apart, the seam prepared in |+> and T rounds of the merged patch, the
split, d rounds apart, the data read out; SD6 noise. L0 is the merge outcome
Z1Z2, L1 and L2 each patch's Z. At T = 1 a single measurement error flips L0
unseen, so the error model refuses the circuit; the run starts at T = 2.
"""

import argparse
import datetime
import math
import pathlib
import sys
import time

import numpy as np
import pymatching
import stim

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import stabilizer_qec as sq  # noqa: E402
from realtime import engine_commit, load_json, machine, write_json  # noqa: E402
from xcheck import graph, mechanisms, rel  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "surgery"
BATCH = 16_384
TARGET = 1500


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, mid - half), min(1.0, mid + half))


def check_cases(quick):
    """(label, Stim text, seed) for every circuit the check covers."""
    cases = []
    for d, merged, basis in [(3, 3, "z"), (3, 2, "x")] + ([] if quick else [(5, 5, "z"), (5, 3, "x")]):
        cases.append((f"Z⊗Z d={d} T={merged} basis {basis}", sq.surgery_circuit(d, merged, 0.003, basis), d * 100 + merged))
    for d in (3,) if quick else (3, 5):
        for inputs in ("z", "x"):
            cases.append((f"CNOT d={d} T={d} inputs {inputs}", sq.surgery_cnot(d, d, 0.003, inputs), d * 100 + 7))
            cases.append((f"X⊗X d={d} T={d} basis {inputs}", sq.surgery_vertical(d, d, 0.003, inputs), d * 100 + 11))
        cases.append((f"Z⊗Z three times d={d} T={d}", sq.surgery_repeated(d, 3, d, 0.003), d * 100 + 13))
        cases.append((f"Z⊗Z⊗Z d={d} T={d}", sq.surgery_product(d, 3, d, 0.003), d * 100 + 17))
    return cases


def cmd_check(args):
    ok = True
    for label, text, seed in check_cases(args.quick):
        circuit = stim.Circuit(text)
        a = mechanisms(stim.DetectorErrorModel(sq.dem_from_circuit(text, False)))
        b = mechanisms(circuit.detector_error_model(decompose_errors=False))
        ea, sa = graph(stim.DetectorErrorModel(sq.dem_from_circuit(text, True)))
        eb, sb = graph(circuit.detector_error_model(decompose_errors=True))
        worst = max((rel(a[k], b[k]) for k in b if k in a), default=0.0)
        splits_differ = sum(1 for k in set(sa) | set(sb) if set(sa.get(k, {})) != set(sb.get(k, {})))
        good = set(a) == set(b) and worst < 1e-9 and set(ea) == set(eb) and splits_differ == 0
        # Matching on Stim's shots: ours against PyMatching, every disagreement a tie.
        shots = 4_000 if args.quick else 20_000
        dem = circuit.detector_error_model(decompose_errors=True)
        dets, obs = circuit.compile_detector_sampler(seed=seed).sample(shots, separate_observables=True)
        truth = sum(obs[:, i].astype(np.uint64) << np.uint64(i) for i in range(obs.shape[1]))
        m = pymatching.Matching.from_detector_error_model(dem)
        pm = m.decode_batch(dets)
        pm = sum(pm[:, i].astype(np.uint64) << np.uint64(i) for i in range(pm.shape[1]))
        raw, w, errors, _ = sq.decode_b8(str(dem), np.packbits(dets, axis=1, bitorder="little").tobytes(), shots, 0, False)
        ours, weights = np.frombuffer(raw, "<u8"), np.frombuffer(w, "<f8")
        disagree = np.nonzero(ours != pm)[0]
        non_ties = sum(1 for i in disagree if abs(m.decode(dets[i], return_weight=True)[1] - weights[i]) > 1e-4)
        good &= non_ties == 0 and errors == 0
        ok &= good
        print(f"  {'ok ' if good else 'BAD'} {label}: {circuit.num_detectors} detectors, {len(b)} "
              f"mechanisms (ours {len(a)}), worst Δp/p {worst:.1e}, {len(eb)} graph edges ({len(set(ea) ^ set(eb))} one-sided), "
              f"{splits_differ} splits differ; PyMatching {int((pm != truth).sum())} failures, ours "
              f"{int((ours != truth).sum())}, {len(disagree)} disagreements, {non_ties} not ties")
    print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
    return 0 if ok else 1


def run_point(d, merged, p, correlated, shot_cap, time_cap):
    text = sq.surgery_circuit(d, merged, p, "z")
    counts = dict(any=0, outcome=0, patches=0)
    shots, t0, batch = 0, time.perf_counter(), 0
    while True:
        dets, obs, _ = sq.sample_b8_batch(text, BATCH, 1_000_003 * d + 10_007 * merged + batch + int(p * 1e5), 0)
        truth = np.frombuffer(obs, np.uint8).reshape(BATCH, -1)[:, 0].astype(np.uint64)
        raw, _, _, _ = sq.decode_b8_own(text, dets, BATCH, 0, correlated)
        pred = np.frombuffer(raw, "<u8")
        # A refused shot is predicted u64::MAX, so it is wrong on every observable: counted once.
        wrong = (pred ^ truth) & np.uint64(0b111)
        counts["any"] += int((wrong != 0).sum())
        counts["outcome"] += int(((wrong & np.uint64(1)) != 0).sum())
        counts["patches"] += int(((wrong & np.uint64(0b110)) != 0).sum())
        shots += BATCH
        batch += 1
        if counts["any"] >= TARGET or shots >= shot_cap or time.perf_counter() - t0 >= time_cap:
            break
    return shots, counts


def cmd_run(args):
    out_path = OUT / "results.json"
    doc = load_json(out_path, dict(points={}))
    doc.update(engine_commit=engine_commit(), machine=machine(),
               note="Z-basis lattice surgery: pre = post = d rounds; L0 the merge outcome, L1/L2 each patch's Z; SD6")
    for p in args.ps:
        for d in (3, 5, 7):
            for merged in range(2, 2 * d + 1):
                for matcher, correlated in (("plain", False), ("correlated", True)):
                    key = f"d{d}/T{merged}/p{p}/{matcher}"
                    if key in doc["points"]:
                        continue
                    t = time.perf_counter()
                    shots, c = run_point(d, merged, p, correlated, args.shot_cap, args.time_cap)
                    rec = dict(d=d, merged=merged, p=p, matcher=matcher, shots=shots, failures=c)
                    for k, v in c.items():
                        rec[f"rate_{k}"] = v / shots
                        rec[f"interval_{k}"] = list(wilson(v, shots))
                    doc["points"][key] = rec
                    doc["generated"] = datetime.date.today().isoformat()
                    write_json(out_path, doc)
                    print(f"d={d} T={merged:>2} p={p} {matcher:<10} {shots:>8,} shots: any {c['any']:>5}  outcome "
                          f"{c['outcome']:>5}  patches {c['patches']:>5}  ({rec['rate_outcome']:.2e} outcome)  "
                          f"{time.perf_counter() - t:.0f} s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["check", "run"])
    ap.add_argument("--ps", type=float, nargs="+", default=[0.003, 0.002])
    ap.add_argument("--quick", action="store_true", help="check: two small cases, fewer shots")
    ap.add_argument("--shot-cap", type=int, default=2_000_000)
    ap.add_argument("--time-cap", type=float, default=300.0)
    args = ap.parse_args()
    return cmd_check(args) if args.command == "check" else cmd_run(args)


if __name__ == "__main__":
    sys.exit(main() or 0)

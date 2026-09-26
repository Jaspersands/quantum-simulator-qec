"""Logical error per cycle of the bivariate bicycle codes under BP+OSD.

Run from the repository root, with the engine built into the virtualenv:

    python tools/gross.py            # both codes, every p -> data/gross/results.json
    python tools/gross.py --quick    # one point each, writes nothing

The experiment is Bravyi et al.'s Z-basis memory: N_c syndrome cycles of the
paper's depth-8 circuit (N_c = 12 for the gross code, 6 for [[72, 12, 6]]),
circuit noise p, the data read out. Shots come from this engine's
bit-parallel sampler, in batches. Each batch is decoded twice on the same
shots:
- BP+OSD-CS of order 7, min-sum BP with adaptive scaling for up to 10,000
  iterations (the paper's decoder);
- BP+OSD-0.
A shot fails if any of the 12 logical qubits is predicted wrongly. A point
stops at 200 failures, a shot cap, or a time cap, whichever comes first.
The logical error per cycle is 1 − (1 − P_L)^(1/N_c), with a Wilson interval
carried through.
"""

import argparse
import datetime
import math
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import stabilizer_qec as sq  # noqa: E402
from realtime import engine_commit, load_json, machine, write_json  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "gross"
CODES = {"gross": 12, "72": 6}
PS = [0.002, 0.003, 0.004, 0.005, 0.006]
TARGET_FAILURES = 200
BATCH = 2_048
DECODERS = {"bposd_cs7": ("osd_cs", 7), "bposd_0": ("osd_0", 0)}


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, mid - half), min(1.0, mid + half))


def per_cycle(pl, cycles):
    return 1 - (1 - pl) ** (1 / cycles)


def run_point(code, cycles, p, shot_cap, time_cap, seed):
    text = sq.bb_memory_circuit(code, cycles, p)
    dem = sq.dem_from_circuit(text, False)
    stats = {k: dict(failures=0, converged=0, seconds=0.0) for k in DECODERS}
    shots, t0, batch = 0, time.perf_counter(), 0
    while True:
        dets, obs, _ = sq.sample_b8_batch(text, BATCH, seed * 1_000_003 + batch, 0)
        truth = np.frombuffer(obs, np.uint8).reshape(BATCH, -1)
        truth = sum(truth[:, i].astype(np.uint64) << np.uint64(8 * i) for i in range(truth.shape[1]))
        for name, (osd, order) in DECODERS.items():
            pred, conv, secs = sq.decode_b8_bposd(dem, dets, BATCH, 10_000, "minimum_sum", 0.0, osd, order, 0)
            pred = np.frombuffer(pred, "<u8")
            stats[name]["failures"] += int((pred != truth).sum())
            stats[name]["converged"] += int(np.frombuffer(conv, np.uint8).sum())
            stats[name]["seconds"] += secs
        shots += BATCH
        batch += 1
        main = stats["bposd_cs7"]["failures"]
        if main >= TARGET_FAILURES or shots >= shot_cap or time.perf_counter() - t0 >= time_cap:
            break
    rec = dict(code=code, cycles=cycles, p=p, shots=shots, results={})
    for name, s in stats.items():
        lo, hi = wilson(s["failures"], shots)
        rec["results"][name] = dict(**s, pl_shot=s["failures"] / shots, pl_cycle=per_cycle(s["failures"] / shots, cycles),
                                    pl_cycle_interval=[per_cycle(lo, cycles), per_cycle(hi, cycles)])
    return rec


def surface_point(d, p, shot_cap, time_cap, seed):
    """A rotated surface code's Z memory beside it: SD6 at the same p, 12 rounds,
    correlated matching. SD6 also puts noise on the Hadamards the surface
    code's X checks use, which the bivariate bicycle circuit does not have, so
    the comparison slightly favours the gross code."""
    text = sq.generate_circuit("rotated", d, 12, "sd6", p, 0.5, "z")
    failures, shots, t0, batch = 0, 0, time.perf_counter(), 0
    while True:
        dets, obs, _ = sq.sample_b8_batch(text, 16_384, seed * 7919 + batch, 0)
        truth = np.frombuffer(obs, np.uint8) & 1
        raw, _, errors, _ = sq.decode_b8_own(text, dets, 16_384, 0, True)
        pred = (np.frombuffer(raw, "<u8") & np.uint64(1)).astype(np.uint8)
        failures += int((pred != truth).sum()) + int(errors)
        shots += 16_384
        batch += 1
        if failures >= TARGET_FAILURES or shots >= shot_cap or time.perf_counter() - t0 >= time_cap:
            break
    lo, hi = wilson(failures, shots)
    return dict(code=f"surface-d{d}", cycles=12, p=p, shots=shots,
                results={"correlated_matching": dict(failures=failures, pl_shot=failures / shots,
                                                     pl_cycle=per_cycle(failures / shots, 12),
                                                     pl_cycle_interval=[per_cycle(lo, 12), per_cycle(hi, 12)])})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--surface", action="store_true", help="the rotated surface code beside it, d = 11 and 13")
    ap.add_argument("--shot-cap", type=int, default=400_000)
    ap.add_argument("--time-cap", type=float, default=1800.0, help="seconds per point")
    args = ap.parse_args()
    out_path = OUT / "results.json"
    doc = load_json(out_path, dict(points={})) if not args.quick else dict(points={})
    doc.update(engine_commit=engine_commit(), machine=machine(), decoders={k: dict(osd=v[0], order=v[1], bp="minimum_sum, adaptive scaling", max_iter=10_000) for k, v in DECODERS.items()},
               note="Z-basis memory, Bravyi et al.'s depth-8 cycle and noise model; a shot fails if any logical qubit does")
    if args.surface:
        for d in (11, 13):
            for p in PS:
                key = f"surface-d{d}/{p}"
                if key in doc.setdefault("surface", {}):
                    continue
                t = time.perf_counter()
                rec = surface_point(d, p, args.shot_cap, args.time_cap, seed=int(p * 1e4) + d)
                doc["surface"][key] = rec
                write_json(out_path, doc)
                r = rec["results"]["correlated_matching"]
                print(f"surface d={d} p={p:.3f}: {rec['shots']:>9,} shots, {r['failures']} failures, p_L/cycle {r['pl_cycle']:.2e}"
                      f"  {time.perf_counter() - t:.0f} s", flush=True)
        return
    for code, cycles in CODES.items():
        for p in ([0.005] if args.quick else PS):
            key = f"{code}/{p}"
            if key in doc["points"]:
                continue
            t = time.perf_counter()
            rec = run_point(code, cycles, p, 4096 if args.quick else args.shot_cap, 60 if args.quick else args.time_cap,
                            seed=int(p * 1e4) + (100 if code == "gross" else 0))
            doc["points"][key] = rec
            doc["generated"] = datetime.date.today().isoformat()
            if not args.quick:
                write_json(out_path, doc)
            r = rec["results"]
            print(f"{code:<5} p={p:.3f}: {rec['shots']:>7,} shots  "
                  + "  ".join(f"{k} {v['failures']} fails, p_L/cycle {v['pl_cycle']:.2e}, BP converged {v['converged']}" for k, v in r.items())
                  + f"  {time.perf_counter() - t:.0f} s", flush=True)


if __name__ == "__main__":
    sys.exit(main() or 0)

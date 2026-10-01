"""Logical error per round of the rotated surface code under uniform circuit noise, measured here.

Run from the repository root, with the engine built into the virtualenv:

    python tools/estimate_noise.py          # every point, then the fit -> data/estimate/noise.json
    python tools/estimate_noise.py --fit    # refit the recorded points only

Section 15's estimate needs the logical error per patch per cycle at the distances a large
algorithm needs, under the uniform noise the papers it is checked against assume (SD6 at
p = 0.1%, as Gidney and Lee et al. assume). The points are the rotated code's Z memory, d rounds,
SD6 noise p, sampled by the bit-parallel sampler and decoded by correlated matching. A point
stops at 200 failures, a shot cap or a time cap. The per-round error is 1 - (1 - P)^(1/d).
The fit is the standard law, eps = A (p / p_th)^((d + 1) / 2), by weighted least squares in
logs over points with at least 20 failures.
"""

import argparse
import datetime
import math
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import stabilizer_qec._core as sq  # noqa: E402
from realtime import engine_commit, load_json, machine, write_json  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "estimate" / "noise.json"
PS = [0.001, 0.002, 0.003, 0.005]
DS = [3, 5, 7, 9, 11]
BATCH = 65_536


def wilson(k, n, z=1.96):
    if n == 0:
        return [0.0, 1.0]
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [max(0.0, mid - half), min(1.0, mid + half)]


per_round = lambda pl, d: 1 - (1 - pl) ** (1 / d)


def point(d, p, seed, shot_cap, time_cap, target):
    text = sq.generate_circuit("rotated", d, d, "sd6", p, 0.5, "z")
    fails = shots = batch = 0
    t0 = time.perf_counter()
    while True:
        dets, obs, _ = sq.sample_b8_batch(text, BATCH, seed * 1_000_003 + batch, 0)
        raw, _, _, _ = sq.decode_b8_own(text, dets, BATCH, 0, True)
        truth = np.frombuffer(obs, np.uint8).reshape(BATCH, -1)[:, 0] & 1
        fails += int(((np.frombuffer(raw, "<u8") & 1) != truth).sum())
        shots += BATCH
        batch += 1
        if fails >= target or shots >= shot_cap or time.perf_counter() - t0 >= time_cap:
            break
    lo, hi = wilson(fails, shots)
    return dict(d=d, p=p, rounds=d, shots=shots, failures=fails, pl=fails / shots,
                eps=per_round(fails / shots, d), eps_interval=[per_round(lo, d), per_round(hi, d)],
                seconds=round(time.perf_counter() - t0, 1))


def fit(points):
    """Weighted least squares of log eps = log A + ((d + 1)/2) (log p - log p_th)."""
    use = [q for q in points if q["failures"] >= 20]
    rows, ys, ws = [], [], []
    for q in use:
        h = (q["d"] + 1) / 2
        rows.append([1.0, h * math.log(q["p"]), -h])
        ys.append(math.log(q["eps"]))
        ws.append(math.sqrt(q["failures"]))  # relative error of a count ~ 1/sqrt(k)
    a, y, w = np.array(rows), np.array(ys), np.array(ws)
    # Unknowns: log A, the slope in log p (1 when the law holds), and log p_th times it.
    coef, *_ = np.linalg.lstsq(a * w[:, None], y * w, rcond=None)
    log_a, slope, c = coef
    # With the slope fixed at 1 the law has two parameters; that is the fit used.
    rows2 = np.array([[1.0, -(q["d"] + 1) / 2] for q in use])
    y2 = np.array([math.log(q["eps"]) - (q["d"] + 1) / 2 * math.log(q["p"]) for q in use])
    (log_a2, log_pth), *_ = np.linalg.lstsq(rows2 * w[:, None], y2 * w, rcond=None)
    A, pth = math.exp(log_a2), math.exp(log_pth)
    resid = [math.log(q["eps"]) - math.log(A * (q["p"] / pth) ** ((q["d"] + 1) / 2)) for q in use]
    return dict(A=A, pth=pth, points_used=len(use), max_log_residual=max(abs(r) for r in resid),
                free_slope=float(slope), residuals={f"d{q['d']}/p{q['p']}": r for q, r in zip(use, resid)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", action="store_true", help="refit the recorded points only")
    ap.add_argument("--shot-cap", type=int, default=20_000_000)
    ap.add_argument("--time-cap", type=float, default=900.0)
    ap.add_argument("--target", type=int, default=200)
    args = ap.parse_args()
    doc = load_json(OUT, dict(points={}))
    if not args.fit:
        doc.update(engine_commit=engine_commit(), machine=machine(),
                   note="Rotated surface code, Z memory of d rounds, SD6 noise p, correlated matching; eps per round")
        for i, (p, d) in enumerate((p, d) for p in PS for d in DS):
            key = f"d{d}/p{p}"
            if key in doc["points"]:
                continue
            q = point(d, p, 9_000_017 * (i + 1), args.shot_cap, args.time_cap, args.target)
            doc["points"][key] = q
            doc["generated"] = datetime.date.today().isoformat()
            write_json(OUT, doc)
            print(f"{key:<12} {q['shots']:>11,} shots {q['failures']:>4} failures  eps {q['eps']:.3e}  {q['seconds']:.0f} s", flush=True)
    doc["fit"] = fit(list(doc["points"].values()))
    write_json(OUT, doc)
    f = doc["fit"]
    print(f"fit: eps = {f['A']:.4f} (p / {f['pth'] * 100:.3f}%)^((d+1)/2) over {f['points_used']} points; "
          f"worst |log residual| {f['max_log_residual']:.2f}; free slope in log p {f['free_slope']:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

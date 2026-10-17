"""Plaquette's study on the lattice-surgery CNOT: leakage added to the circuit noise, and what it
does to the threshold.

    python tools/surgery_leakage.py [--shots N] [--distances 3 5 7] [--fractions 0 0.1 0.5 1]
                                    [--ps 0.002 ... 0.005] [--out data/surgery/leakage.json]

The circuit is `stabilizer_qec.surgery.cnot(d, merged=d, p=p)` (SD6 noise at p: depolarizing after
every gate, flips before every measurement and after every reset) with `LEAKAGE(f·p)` after every
CX on both its qubits, for leakage fractions f, and before it `RELAX(s)` (seepage: a leaked qubit
returns with probability s at each of its two-qubit gates, 0.1 by default, so a data qubit stays
leaked for a few rounds; without it a data qubit, never reset, would stay leaked to the end and
no distance would help). A leaked qubit is read as 1 until it returns, and every two-qubit gate
it meets leaves its partner a uniformly random Pauli (Google's model, Deltakit's `LEAKAGE` and
`RELAX` tags): a leaked ancilla fires its detectors round after round, a leaked data qubit spreads
errors through its four stabilizers. The leakage sampler gives each shot's detection events and
heralds. Each shot is decoded twice: by matching on the leak-free detector error model,
blind to the leakage (Plaquette's experiment: the decoder of the depolarizing model, the noise with
leakage in it), and leakage-aware, the heralds raising the priors of the faults a leak makes likely
(`LeakageAwareDecoder`). The threshold is read two ways per fraction and decoder: the crossing of
the two largest distances' curves (log-linear between grid points), and qec-util's
Wang–Harrington–Preskill fit (`stabilizer_qec.analysis.threshold.get_threshold`, a quadratic in
the rescaled error rate, bootstrapped) on the points within a factor 1.6 of that crossing.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
import time

import numpy as np

import stabilizer_qec as sq
from stabilizer_qec.analysis.threshold import get_threshold

ROOT = pathlib.Path(__file__).resolve().parent.parent
TWO_QUBIT = {"CX", "CNOT", "CZ", "CY", "XCX", "XCY", "XCZ", "YCX", "YCY", "YCZ", "ZCX", "ZCY", "ZCZ", "SWAP", "ISWAP", "CXSWAP", "SWAPCX"}


def leaky_cnot(distance: int, p: float, fraction: float, seepage: float) -> sq.Circuit:
    """The surgery CNOT at `p` with, after every two-qubit gate on both its qubits, `RELAX(seepage)`
    (a leaked qubit returns) then `LEAKAGE(fraction·p)`."""
    lines = []
    for line in str(sq.surgery.cnot(distance, merged=distance, p=p)).splitlines():
        lines.append(line)
        body = line.lstrip()
        name = body.split(" ", 1)[0]
        if fraction > 0 and name in TWO_QUBIT:
            indent = line[: len(line) - len(body)]
            targets = body[len(name):]
            if seepage > 0:
                lines.append(f"{indent}RELAX({seepage:.12g}){targets}")
            lines.append(f"{indent}LEAKAGE({fraction * p:.12g}){targets}")
    return sq.Circuit("\n".join(lines))


def run_point(distance: int, p: float, fraction: float, seepage: float, shots: int, seed: int) -> dict:
    circuit = leaky_cnot(distance, p, fraction, seepage)
    t0 = time.time()
    dets, obs, heralds = circuit.compile_leakage_sampler(seed=seed).sample(shots)
    t1 = time.time()
    if fraction > 0:
        decoder = sq.LeakageAwareDecoder(circuit)
        blind = decoder.decode_batch(dets)
        aware = decoder.decode_batch(dets, heralds=heralds)
    else:  # no leakage: matching on the circuit's model is the same decoder, built in a fraction of the time
        blind = aware = sq.Matching.from_detector_error_model(circuit.detector_error_model(decompose_errors=True)).decode_batch(dets)
    t2 = time.time()
    fails_blind = int((blind != obs).any(axis=1).sum())
    fails_aware = int((aware != obs).any(axis=1).sum())
    return {"d": distance, "p": p, "fraction": fraction, "shots": shots, "fails_blind": fails_blind, "fails_aware": fails_aware, "qubits": circuit.num_qubits, "detectors": circuit.num_detectors, "heralds_per_shot": float(np.asarray(heralds).sum() / shots), "sample_s": t1 - t0, "decode_s": t2 - t1}


def crossing(points: dict, distances: list, which: str) -> float | None:
    """Where the two largest distances' logical error rates cross, log-linear between grid
    points; None without a sign change."""
    a, b = sorted(distances)[-2:]
    ps = sorted({q["p"] for q in points.values()})
    rate = lambda d, p: max(points[f"d{d}/p{p}"][which], 0.5) / points[f"d{d}/p{p}"]["shots"]  # noqa: E731
    diffs = [math.log(rate(b, p)) - math.log(rate(a, p)) for p in ps]
    for (p0, y0), (p1, y1) in zip(zip(ps, diffs), zip(ps[1:], diffs[1:])):
        if y0 <= 0 <= y1 or y0 >= 0 >= y1:
            if y0 == y1:
                return p0
            return math.exp(math.log(p0) + (0 - y0) / (y1 - y0) * (math.log(p1) - math.log(p0)))
    return None


def fitted(points: dict, distances: list, which: str, around: float | None) -> list | None:
    """qec-util's threshold fit on the points within a factor 1.6 of `around`."""
    if around is None:
        return None
    data = {}
    for d in distances:
        rows = sorted((q["p"], q[which], q["shots"]) for q in points.values() if q["d"] == d and around / 1.6 <= q["p"] <= around * 1.6)
        if len(rows) < 3:
            return None
        data[d] = tuple(np.array(col, dtype=float) for col in zip(*rows))
    try:
        guess = np.array([around, 1.0, 0.0, 0.0, 0.0])
        t, lo, hi = get_threshold(data, "poly-2", num_samples_bootstrap=200, params_guess=guess)
    except Exception as e:  # a fit that does not converge is reported as none
        print(f"  fit failed ({which}): {e}", file=sys.stderr)
        return None
    if not (math.isfinite(t) and around / 2 <= t <= around * 2):
        return None
    return [float(t), float(lo), float(hi)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shots", type=int, default=20_000)
    ap.add_argument("--distances", type=int, nargs="+", default=[3, 5, 7])
    ap.add_argument("--fractions", type=float, nargs="+", default=[0.0, 0.1, 0.5, 1.0])
    ap.add_argument("--ps", type=float, nargs="+", default=[0.002, 0.0025, 0.003, 0.0035, 0.004, 0.0045, 0.005])
    ap.add_argument("--seepage", type=float, default=0.1, help="a leaked qubit's probability of returning at each of its two-qubit gates")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default="data/surgery/leakage.json")
    args = ap.parse_args()

    out = {"model": f"surgery.cnot(d, merged=d, p) with RELAX({args.seepage:g}) then LEAKAGE(f·p) after every CX on both qubits; leaked qubits read 1, depolarize their two-qubit partners, return at reset or by RELAX", "shots": args.shots, "distances": args.distances, "ps": args.ps, "fractions": args.fractions, "seepage": args.seepage, "fractions_results": {}, "points": {}}
    seed = args.seed
    for f in args.fractions:
        pts = {}
        for d in args.distances:
            for p in args.ps:
                seed += 1
                q = run_point(d, p, f, args.seepage, args.shots, seed)
                pts[f"d{d}/p{p}"] = q
                out["points"][f"f{f:g}/d{d}/p{p}"] = q
                print(f"f={f:g} d={d} p={p}: blind {q['fails_blind'] / q['shots']:.4f} aware {q['fails_aware'] / q['shots']:.4f} heralds/shot {q['heralds_per_shot']:.2f} ({q['sample_s']:.1f}s + {q['decode_s']:.1f}s)", flush=True)
        res = {}
        for which in ("fails_blind", "fails_aware"):
            cross = crossing(pts, args.distances, which)
            res[which.split("_")[1]] = {"crossing": cross, "fit": fitted(pts, args.distances, which, cross)}
        out["fractions_results"][f"{f:g}"] = res
        print(f"f={f:g}: thresholds {res}", flush=True)
    path = ROOT / args.out
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Belief-matching on Google's Willow and Sycamore data.

Run from the repository root, with the engine built into .venv and the zips
in data/google/:

    .venv/bin/python tools/belief.py sycamore   # every experiment, 50,000 shots -> data/belief/sycamore.json
    .venv/bin/python tools/belief.py willow     # every experiment, first 10,000 shots -> data/belief/willow.json

Sycamore is decoded whole, with the data-fitted pij priors cross-fitted as
Google used them (even shots by the model fitted on odd shots, and the
reverse) and with the circuit's own model. Google's recorded belief-matching
predictions are scored on the same shots, and counted where they agree with
ours shot by shot.

Willow costs about 25 ms a shot at d = 7 and 250 rounds on ten cores, so the
whole dataset at 50,000 shots an experiment would take some 18 hours. Every
experiment is decoded on its first 10,000 shots instead, with Google's SI1000
prior. On those same shots, plain and correlated matching and Google's own
decoders are scored beside it, so every comparison is paired.

Both resume: an experiment already in the output file is skipped.
Spec: docs/superpowers/specs/2026-09-26-belief-matching-design.md.
"""

import argparse
import datetime
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import google as G  # noqa: E402
import stabilizer_qec as sq  # noqa: E402
from realtime import engine_commit, load_json, machine, write_json  # noqa: E402

OUT = G.ROOT / "data" / "belief"
WILLOW_SHOTS = 10_000
MAX_ITER = 20


def belief(dem_text, dets, shots):
    raw, _, conv, errors, seconds = sq.decode_b8_belief(dem_text, dets, shots, MAX_ITER, "product_sum", 1.0, 0)
    return np.frombuffer(raw, "<u8"), np.frombuffer(conv, np.uint8), errors, seconds


def score(raw, truth):
    pred = (raw & np.uint64(1)).astype(np.uint8)
    return pred, int(((pred != truth) | (raw == G.FAILED)).sum())


def cross_fitted(src, dets, shots):
    """Even shots by the model fitted on odd shots, odd by the one fitted on even."""
    even_for_odd, odd_for_even = src
    rows = np.frombuffer(dets, np.uint8).reshape(shots, -1)
    ev, od = rows[0::2], rows[1::2]
    re, ce, xe, se = belief(odd_for_even, ev.tobytes(), len(ev))
    ro, co, xo, so = belief(even_for_odd, od.tobytes(), len(od))
    raw = np.empty(shots, "<u8")
    conv = np.empty(shots, np.uint8)
    raw[0::2], raw[1::2], conv[0::2], conv[1::2] = re, ro, ce, co
    return raw, conv, xe + xo, se + so


def sycamore_one(e):
    dets, truth = G.read(e, "dets"), G.bits(e, "obs")
    rec = dict(d=e.d, patch=e.patch, basis=e.basis, rounds=e.rounds, shots=e.shots, results={}, agree={})
    google = G.bits(e, "pred:belief_matching") if G.exists(e, "pred:belief_matching") else None
    for prior, src in [("pij", (G.text(e, "dem:pij_from_even_for_odd"), G.text(e, "dem:pij_from_odd_for_even"))),
                       ("circuit", G.text(e, "dem:circuit_detector_error_model"))]:
        raw, conv, errors, seconds = cross_fitted(src, dets, e.shots) if prior == "pij" else belief(src, dets, e.shots)
        pred, failures = score(raw, truth)
        rec["results"][f"ours/{prior}/belief"] = dict(failures=failures, converged=int(conv.sum()), errors=int(errors),
                                                      seconds=seconds)
        if google is not None:
            rec["agree"][f"ours/{prior}/belief"] = int((pred == google).sum())
    for name in G.SYCAMORE_DECODERS:
        if G.exists(e, f"pred:{name}"):
            rec["results"][f"google/{name}"] = dict(failures=int((G.bits(e, f"pred:{name}") != truth).sum()))
    return rec


def willow_one(e):
    n = min(WILLOW_SHOTS, e.shots)
    dets = G.read(e, "dets")
    stride = len(dets) // e.shots
    dets = dets[: stride * n]
    truth = G.bits(e, "obs")[:n]
    dem = G.text(e, "dem:correlated_matching_decoder_with_si1000_prior")
    rec = dict(d=e.d, patch=e.patch, basis=e.basis, rounds=e.rounds, shots=n, results={})
    raw, conv, errors, seconds = belief(dem, dets, n)
    _, failures = score(raw, truth)
    rec["results"]["ours/si1000/belief"] = dict(failures=failures, converged=int(conv.sum()), errors=int(errors), seconds=seconds)
    for mode, corr in (("plain", False), ("correlated", True)):
        r, _, x, s = sq.decode_b8(dem, dets, n, 0, corr)
        _, f = score(np.frombuffer(r, "<u8"), truth)
        rec["results"][f"ours/si1000/{mode}"] = dict(failures=f, errors=int(x), seconds=s)
    for name in G.PATHWAYS:
        if G.exists(e, f"pred:{name}"):
            rec["results"][f"google/{name}"] = dict(failures=int((G.bits(e, f"pred:{name}")[:n] != truth).sum()))
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=["sycamore", "willow"])
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    out_path = OUT / f"{args.dataset}.json"
    doc = load_json(out_path, dict(experiments={}))
    doc.update(engine_commit=engine_commit(), machine=machine(), max_iter=MAX_ITER, bp_method="product_sum",
               note=("every experiment, all shots; pij priors cross-fitted" if args.dataset == "sycamore"
                     else f"every experiment, its first {WILLOW_SHOTS:,} shots; SI1000 prior; all decoders on the same shots"))
    exps = list(G.experiments(args.dataset))
    if args.limit:
        exps = exps[: args.limit]
    for e in exps:
        if e.name in doc["experiments"]:
            continue
        t = time.perf_counter()
        rec = sycamore_one(e) if args.dataset == "sycamore" else willow_one(e)
        doc["experiments"][e.name] = rec
        doc["generated"] = datetime.date.today().isoformat()
        write_json(out_path, doc)
        r = rec["results"]
        line = "  ".join(f"{k}: {v['failures']}" for k, v in r.items() if k.startswith("ours") or "belief" in k or "correlated_matching_decoder_with_si1000" in k)
        print(f"{e.name:<44} {line}  {time.perf_counter() - t:.1f} s", flush=True)


if __name__ == "__main__":
    sys.exit(main() or 0)

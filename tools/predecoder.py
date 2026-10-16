"""The AI pre-decoder's weights for the rotated memory, and what they save over matching.

Run from the repository root, after building the engine into .venv:

    VIRTUAL_ENV=$PWD/.venv CARGO_TARGET_DIR=target-py .venv/bin/maturin develop --profile python
    .venv/bin/python tools/predecoder.py train      # -> python/stabilizer_qec/data/predecoder_memory.npz
    .venv/bin/python tools/predecoder.py evaluate   # logical errors -> data/predecoder/evaluate.json

The weights: three hidden layers of 32 filters, 3x3x3 (receptive field 9), trained on
memory_circuit's distance-9, 9-round rotated memory at SD6 strength p = 0.006 (near threshold,
as the paper trains), by Lion at the paper's rate, 12,000 steps of 256 fresh shots each. The
evaluation counts logical errors on the same shots for matching, correlated matching, the
paper's pipeline (hard corrections, then matching) and matching on the network's weights.
"""

import argparse
import json
import pathlib
import platform
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
WEIGHTS = ROOT / "python" / "stabilizer_qec" / "data" / "predecoder_memory.npz"

import stabilizer_qec as sq  # noqa: E402


def model(distance: int, rounds: int, p: float):
    return sq.memory_circuit(distance=distance, rounds=rounds, p=p).detector_error_model(decompose_errors=True)


def train(args: argparse.Namespace) -> None:
    pre = sq.PreDecoder(model(9, 9, 0.006), hidden=(32, 32, 32), seed=1)
    t = time.time()
    losses = pre.train(args.steps, batch=256, optimizer="lion", seed=2026, threads=args.threads)
    print(f"{args.steps} steps in {time.time() - t:.0f} s; loss {losses[:100].mean():.2f} -> {losses[-100:].mean():.2f}")
    WEIGHTS.parent.mkdir(exist_ok=True)
    pre.save(WEIGHTS)


def evaluate(args: argparse.Namespace) -> None:
    rows = []
    for distance in (5, 7, 9):
        for p in (0.003, 0.004, 0.005, 0.006):
            dem = model(distance, distance, p)
            dets, obs, _ = dem.compile_sampler(seed=distance * 1000 + int(p * 1e4)).sample(args.shots)
            fails = lambda pred: int((np.asarray(pred) != obs).any(axis=1).sum())  # noqa: E731
            row = {"distance": distance, "rounds": distance, "p": p, "shots": args.shots}
            row["matching"] = fails(sq.Matching(dem).decode_batch(dets))
            corr = sq.Matching.from_detector_error_model(dem, enable_correlations=True)
            row["correlated matching"] = fails(corr.decode_batch(dets, enable_correlations=True))
            pre = sq.PreDecoder(dem, WEIGHTS)
            left, _, _ = pre.predecode(dets, threads=args.threads)
            row["density reduction"] = round(float(dets.sum() / max(left.sum(), 1)), 2)
            row["paper's pipeline"] = fails(sq.PreDecoder(dem, WEIGHTS, reweight=False).decode_batch(dets, threads=args.threads))
            t = time.time()
            row["reweighted"] = fails(pre.decode_batch(dets, threads=args.threads))
            row["reweighted us/shot"] = round((time.time() - t) / args.shots * 1e6, 1)
            print(row, flush=True)
            rows.append(row)
    out = ROOT / "data" / "predecoder"
    out.mkdir(parents=True, exist_ok=True)
    (out / "evaluate.json").write_text(json.dumps({"machine": platform.platform(), "threads": args.threads, "rows": rows}, indent=1) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--steps", type=int, default=12000)
    t.add_argument("--threads", type=int, default=0)
    e = sub.add_parser("evaluate")
    e.add_argument("--shots", type=int, default=50_000)
    e.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()
    {"train": train, "evaluate": evaluate}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())

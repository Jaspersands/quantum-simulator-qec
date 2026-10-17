"""Sampling and BP throughput on the CPU and the GPU.

    python tools/gpu_bench.py [--shots N] [--distances 5 11 17] [--out data/gpu/bench.json]

For each rotated surface-code memory (Stim's generated circuit, d rounds, all four noise rates 0.1%), shots per second of the word
sampler on one thread and on every core (`device="cpu"`), the lane sampler on every core
(`device="cpu-lanes"`, the GPU's program on the CPU) and the GPU (`device="gpu"`), each timed on
a call of `--shots` shots after a warm-up call; and for min-sum BP on the d = 11 memory's check
matrix and on the [[144,12,12]] code's, shots per second of the f64 decoder (shot by shot, through `decode`), the
f32 decoder (every core) and the GPU. The lane and GPU samplers' bytes are compared on the way
(they must be equal). cuStabilizer's published figure, up to 150× one CPU thread, is the
yardstick the README quotes beside these.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import platform
import sys
import time

import numpy as np

import stabilizer_qec as sq
from stabilizer_qec import BpDecoder
from stabilizer_qec.beliefmatching import detector_error_model_to_check_matrices

ROOT = pathlib.Path(__file__).resolve().parent.parent


def timed(fn, *args, **kwargs):
    fn(*args, **kwargs)  # warm up: the program, the device, the caches
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    return out, time.perf_counter() - t0


def sampling(d: int, shots: int, gpu: bool) -> dict:
    c = sq.Circuit.generated("surface_code:rotated_memory_z", distance=d, rounds=d, after_clifford_depolarization=0.001, before_round_data_depolarization=0.001, before_measure_flip_probability=0.001, after_reset_flip_probability=0.001)
    row = {"distance": d, "qubits": c.num_qubits, "detectors": c.num_detectors, "shots": shots}

    def run(device: str, threads: int):
        s = c.compile_detector_sampler(seed=1, device=device)
        return s.sample(shots, bit_packed=True, threads=threads)

    _, t = timed(run, "cpu", 1)
    row["cpu_1_thread_per_s"] = shots / t
    _, t = timed(run, "cpu", 0)
    row["cpu_all_threads_per_s"] = shots / t
    lanes, t = timed(run, "cpu-lanes", 0)
    row["cpu_lanes_per_s"] = shots / t
    if gpu:
        out, t = timed(run, "gpu", 0)
        row["gpu_per_s"] = shots / t
        row["gpu_equals_lanes"] = bool(np.array_equal(out, lanes))
    return row


def bp(name: str, H, priors, syndromes, gpu: bool) -> dict:
    row = {"problem": name, "checks": int(H.shape[0]), "variables": int(H.shape[1]), "shots": int(len(syndromes)), "max_iter": 20}
    f64 = BpDecoder(H, error_channel=priors, max_iter=20, bp_method="minimum_sum", ms_scaling_factor=0.625)
    f32 = BpDecoder(H, error_channel=priors, max_iter=20, bp_method="minimum_sum", ms_scaling_factor=0.625, device="cpu-f32")
    # The f64 decoder decodes shot by shot through decode; time the batch through sq.BpOsd's
    # sibling path, the ldpc-compatible decoder's batch.
    _, t = timed(f64.decode_batch, syndromes)
    row["cpu_f64_per_s"] = len(syndromes) / t
    out32, t = timed(f32.decode_batch, syndromes)
    row["cpu_f32_per_s"] = len(syndromes) / t
    if gpu:
        g = BpDecoder(H, error_channel=priors, max_iter=20, bp_method="minimum_sum", ms_scaling_factor=0.625, device="gpu")
        outg, t = timed(g.decode_batch, syndromes)
        row["gpu_per_s"] = len(syndromes) / t
        row["gpu_equals_f32"] = bool(np.array_equal(out32, outg))
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shots", type=int, default=1_000_000)
    ap.add_argument("--bp-shots", type=int, default=20_000)
    ap.add_argument("--distances", type=int, nargs="+", default=[5, 11, 17])
    ap.add_argument("--out", default="data/gpu/bench.json")
    args = ap.parse_args()
    gpu = sq.gpu_device()
    print(f"GPU: {gpu}", flush=True)
    out = {"machine": platform.platform(), "processor": platform.processor() or platform.machine(), "cores": os.cpu_count(), "gpu": gpu, "version": sq.__version__, "sampling": [], "bp": []}
    for d in args.distances:
        row = sampling(d, args.shots, gpu is not None)
        out["sampling"].append(row)
        print(json.dumps(row), flush=True)
    c = sq.memory_circuit(distance=11, rounds=11, p=0.001)
    m = detector_error_model_to_check_matrices(c.detector_error_model(decompose_errors=True), allow_undecomposed_hyperedges=True)
    H = m.check_matrix
    H = np.asarray(H.todense() if hasattr(H, "todense") else H, dtype=np.uint8)
    dets = c.compile_detector_sampler(seed=2).sample(args.bp_shots).astype(np.uint8)
    out["bp"].append(bp("surface d=11 memory", H, np.asarray(m.priors, dtype=float), dets, gpu is not None))
    print(json.dumps(out["bp"][-1]), flush=True)
    Hz = np.asarray(sq.BivariateBicycleCode("gross").check_matrices()[1], dtype=np.uint8)
    rng = np.random.default_rng(1)
    errors = rng.random((args.bp_shots, Hz.shape[1])) < 0.003
    synd = (errors.astype(np.uint8) @ Hz.T) % 2
    out["bp"].append(bp("gross [[144,12,12]] code capacity 0.3%", Hz, np.full(Hz.shape[1], 0.003), synd.astype(np.uint8), gpu is not None))
    print(json.dumps(out["bp"][-1]), flush=True)
    path = ROOT / args.out
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Real-time decoding: window decoders' accuracy, latency and throughput.

Run from the repository root, after building the engine into .venv:

    VIRTUAL_ENV=$PWD/.venv CARGO_TARGET_DIR=target-py .venv/bin/maturin develop --profile python
    .venv/bin/python tools/realtime.py accuracy   # windowed against global, Willow and SD6 -> data/realtime/
    .venv/bin/python tools/realtime.py latency    # decode times scheduled at Willow's 1.1 us cycle
    .venv/bin/python tools/realtime.py million    # a million-round stream

Spec: docs/superpowers/specs/2026-09-26-real-time-design.md, Part 2.
"""

import argparse
import datetime
import json
import math
import os
import pathlib
import platform
import subprocess
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import google as G  # noqa: E402
import stabilizer_qec as sq  # noqa: E402

OUT = ROOT / "data" / "realtime"
PRIOR = "dem:correlated_matching_decoder_with_si1000_prior"
CYCLE = 1.1e-6  # Willow's error-correction cycle, seconds
MATCHERS = {"plain": False, "correlated": True}


def machine():
    brand = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
    return dict(cpu=brand or platform.processor(), cores=os.cpu_count())


def engine_commit():
    return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def write_json(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=1) + "\n")
    tmp.replace(p)


def load_json(p, default):
    return json.loads(p.read_text()) if p.exists() else default


def failures(raw, truth):
    raw = np.frombuffer(raw, "<u8")
    return int((((raw & np.uint64(1)).astype(np.uint8) != truth) | (raw == G.FAILED)).sum())


def window_configs(d):
    """C = d, and B = d everywhere; at d = 5 also B = 3 and B = 10, to see where it gives."""
    buffers = [d] + ([3, 10] if d == 5 else [])
    return [(mode, b, m) for b in buffers for mode in ("sliding", "parallel") for m in MATCHERS]


# -- accuracy ---------------------------------------------------------------

def cmd_accuracy(args):
    out_path = OUT / "willow-windows.json"
    doc = load_json(out_path, dict(experiments={}))
    doc.update(engine_commit=engine_commit(), machine=machine(), prior=PRIOR[4:],
               note="C = d; results beside the global decoder's on the same shots")
    global_doc = load_json(G.OUT / "willow.json", dict(experiments={}))["experiments"]
    exps = [e for e in G.experiments("willow") if e.rounds >= 10]
    if args.limit:
        exps = exps[: args.limit]
    for e in exps:
        if e.name in doc["experiments"] and not args.redo:
            continue
        t = time.perf_counter()
        dem = G.text(e, PRIOR)
        dets, truth = G.read(e, "dets"), G.bits(e, "obs")
        rec = dict(d=e.d, patch=e.patch, basis=e.basis, rounds=e.rounds, shots=e.shots, results={})
        glob = global_doc.get(e.name, {}).get("results", {})
        for m in MATCHERS:
            key = f"ours/si1000/{m}"
            if key in glob:
                rec["results"][f"global/{m}"] = dict(failures=glob[key]["failures"])
        for mode, b, m in window_configs(e.d):
            raw, unexplained, _, _ = sq.decode_b8_window(dem, dets, e.shots, e.d, b, mode, MATCHERS[m], 0, False)
            rec["results"][f"window/{mode}/B{b}/{m}"] = dict(failures=failures(raw, truth), unexplained=int(unexplained))
        doc["experiments"][e.name] = rec
        doc["generated"] = datetime.date.today().isoformat()
        write_json(out_path, doc)
        r = rec["results"]
        line = "  ".join(f"{k}: {v['failures']}" for k, v in r.items() if "B%d/" % e.d in k or k.startswith("global"))
        bad = sum(v.get("unexplained", 0) for v in r.values())
        print(f"{e.name:<32} {line}  unexplained {bad}  {time.perf_counter() - t:.1f} s", flush=True)

    # Simulated SD6 memories, T = 50, beside global decoding of the same shots.
    sim_path = OUT / "sd6-windows.json"
    sim = dict(generated=datetime.date.today().isoformat(), engine_commit=engine_commit(), machine=machine(), points=[])
    shots = 20_000
    for d in (3, 5, 7):
        for p in (0.003, 0.005):
            text = sq.generate_circuit("rotated", d, 50, "sd6", p, 0.5, "z")
            dem = sq.dem_from_circuit(text, True)
            dets, obs, _ = sq.sample_b8_batch(text, shots, 1000 * d + int(p * 1e4), 0)
            truth = np.frombuffer(obs, np.uint8) & 1
            point = dict(d=d, p=p, rounds=50, shots=shots, results={})
            for m, corr in MATCHERS.items():
                raw, _, _, _ = sq.decode_b8(dem, dets, shots, 0, corr)
                point["results"][f"global/{m}"] = dict(failures=failures(raw, truth))
                for mode in ("sliding", "parallel"):
                    raw, unexplained, _, _ = sq.decode_b8_window(dem, dets, shots, d, d, mode, corr, 0, False)
                    point["results"][f"window/{mode}/B{d}/{m}"] = dict(failures=failures(raw, truth), unexplained=int(unexplained))
            sim["points"].append(point)
            print(f"SD6 d={d} p={p}: " + "  ".join(f"{k} {v['failures']}" for k, v in point["results"].items()), flush=True)
    write_json(sim_path, sim)


# -- latency ----------------------------------------------------------------

def dependencies(info):
    """Which windows each window waits for. Sliding windows (one per phase)
    wait for the one before. Parallel layer-B windows wait for the layer-A
    windows whose commit regions border theirs."""
    n = len(info)
    phases = [x[4] for x in info]
    sliding = len(set(phases)) == n
    deps = [[] for _ in range(n)]
    for w in range(n):
        if sliding:
            deps[w] = [x for x in range(n) if phases[x] == phases[w] - 1]
        elif phases[w] == 1:
            c0, c1 = info[w][2], info[w][3]
            deps[w] = [a for a in range(n) if phases[a] == 0 and (info[a][3] == c0 or info[a][2] == c1)]
    return deps


def schedule(times, info, workers, deps, cycle=CYCLE):
    """Each window's latency in one stream, at a fixed cadence of rounds.

    `times[w]` is window w's measured decode time, and `info[w]` is (first
    layer, end layer, commit start, commit end, phase). Layer l arrives at
    (l + 1) · cycle, so a window's last layer arrives at end · cycle. A window
    is ready once its last layer has arrived and every window it depends on is
    done; the ready window soonest ready goes to the earliest free of
    `workers`. Latency is completion minus the arrival of its last layer.
    """
    n = len(info)
    arrive = [info[w][1] * cycle for w in range(n)]
    done = [None] * n
    free = [0.0] * workers
    remaining = set(range(n))
    while remaining:
        ready_at = {w: max([arrive[w]] + [done[x] for x in deps[w]])
                    for w in remaining if all(done[x] is not None for x in deps[w])}
        w = min(ready_at, key=ready_at.get)
        k = min(range(workers), key=lambda i: free[i])
        done[w] = max(ready_at[w], free[k]) + times[w]
        free[k] = done[w]
        remaining.remove(w)
    return [done[w] - arrive[w] for w in range(n)]


def latency_stats(times, info, workers):
    """Over every shot (each a stream): mean and 99th-percentile latency, and
    whether the backlog stays bounded (the last quarter's mean latency within
    10% of the second quarter's, over windows ordered by commit)."""
    lat = []
    grow = []
    deps = dependencies(info)
    for row in times:
        ls = schedule(row, info, workers, deps)
        lat.extend(ls)
        order = sorted(range(len(info)), key=lambda w: info[w][2])
        seq = [ls[w] for w in order]
        q = max(1, len(seq) // 4)
        grow.append(np.mean(seq[-q:]) / max(np.mean(seq[q:2 * q]), 1e-12))
    lat = np.array(lat)
    return dict(mean_us=float(lat.mean() * 1e6), p99_us=float(np.percentile(lat, 99) * 1e6),
                max_us=float(lat.max() * 1e6), growth=float(np.median(grow)),
                keeps_up=bool(np.median(grow) < 1.1))


def cmd_latency(args):
    out = dict(generated=datetime.date.today().isoformat(), engine_commit=engine_commit(), machine=machine(),
               cycle_us=CYCLE * 1e6, streams=[])
    shots = args.shots
    willow = {e.d: e for e in G.experiments("willow") if e.rounds == 250 and e.basis == "Z"
              and e.patch in ("d3_at_q6_7", "d5_at_q4_7", "d7_at_q6_7")}
    for d in (3, 5, 7):
        e = willow[d]
        dem = G.text(e, PRIOR)
        dets = G.read(e, "dets")
        stride = len(dets) // e.shots
        sub = dets[: stride * shots]
        for m, corr in MATCHERS.items():
            for mode in ("sliding", "parallel"):
                # One thread: each window's time is one core's, uncontended.
                _, unexplained, times_b, info = sq.decode_b8_window(dem, sub, shots, d, d, mode, corr, 1, True)
                times = np.frombuffer(times_b, "<f8").reshape(shots, len(info))
                entry = dict(source=f"willow/{e.patch}/Z/r250", d=d, rounds=250, matcher=m, mode=mode,
                             shots=shots, windows=len(info), unexplained=int(unexplained),
                             window_us=dict(mean=float(times.mean() * 1e6), p50=float(np.percentile(times, 50) * 1e6),
                                            p99=float(np.percentile(times, 99) * 1e6)),
                             by_workers={})
                for k in ((1,) if mode == "sliding" else (1, 2, 4, 8, 16)):
                    entry["by_workers"][k] = latency_stats(times, info, k)
                out["streams"].append(entry)
                row = "  ".join(f"K={k}: {v['mean_us']:.0f}/{v['p99_us']:.0f} us {'keeps up' if v['keeps_up'] else 'falls behind'}"
                                for k, v in entry["by_workers"].items())
                print(f"d={d} {m:<10} {mode:<8} window {entry['window_us']['mean']:.0f} us  {row}", flush=True)
    write_json(OUT / "latency.json", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["accuracy", "latency", "million"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--redo", action="store_true")
    ap.add_argument("--shots", type=int, default=2000)
    args = ap.parse_args()
    if args.command == "accuracy":
        return cmd_accuracy(args)
    if args.command == "latency":
        return cmd_latency(args)
    return cmd_million(args)


if __name__ == "__main__":
    sys.exit(main() or 0)

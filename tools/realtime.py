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


def buffer_name(d, b):
    """Buffers relative to the distance, so one key spans every distance and Λ can be fitted."""
    return {d: "Bd", 2 * d: "B2d", math.ceil(d / 2): "Bhalf"}.get(b, f"B{b}")


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
            rec["results"][f"window/{mode}/{buffer_name(e.d, b)}/{m}"] = dict(failures=failures(raw, truth), unexplained=int(unexplained))
        doc["experiments"][e.name] = rec
        doc["generated"] = datetime.date.today().isoformat()
        write_json(out_path, doc)
        r = rec["results"]
        line = "  ".join(f"{k}: {v['failures']}" for k, v in r.items() if "/Bd/" in k or k.startswith("global"))
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
                    point["results"][f"window/{mode}/Bd/{m}"] = dict(failures=failures(raw, truth), unexplained=int(unexplained))
            sim["points"].append(point)
            print(f"SD6 d={d} p={p}: " + "  ".join(f"{k} {v['failures']}" for k, v in point["results"].items()), flush=True)
    write_json(sim_path, sim)


# -- latency ----------------------------------------------------------------

def schedule(times, info, workers, deps, cycle=CYCLE):
    """Each window's latency in one stream, at a fixed cadence of rounds.

    `times[w]` is window w's measured decode time, and `info[w]` is (first
    layer, end layer, commit start, commit end, phase, dependencies). Layer l
    arrives at (l + 1) · cycle, so a window's last layer arrives at end ·
    cycle. A window is ready once its last layer has arrived and every window
    it depends on is done; the ready window soonest ready goes to the earliest
    free of `workers` (list scheduling, O(n log n)). Latency is completion
    minus the arrival of its last layer.
    """
    import heapq
    n = len(info)
    arrive = [info[w][1] * cycle for w in range(n)]
    dependents = [[] for _ in range(n)]
    waiting = [len(deps[w]) for w in range(n)]
    for w in range(n):
        for x in deps[w]:
            dependents[x].append(w)
    ready = [(arrive[w], w) for w in range(n) if waiting[w] == 0]
    heapq.heapify(ready)
    free = [0.0] * workers
    done = [0.0] * n
    dep_done = [0.0] * n
    while ready:
        t, w = heapq.heappop(ready)
        f = heapq.heappop(free)
        done[w] = max(t, f) + times[w]
        heapq.heappush(free, done[w])
        for x in dependents[w]:
            dep_done[x] = max(dep_done[x], done[w])
            waiting[x] -= 1
            if waiting[x] == 0:
                heapq.heappush(ready, (max(arrive[x], dep_done[x]), x))
    return [done[w] - arrive[w] for w in range(n)]


def latency_stats(times, info, workers):
    """Over every shot (each a stream): mean and 99th-percentile latency, and
    whether the backlog stays bounded (the last quarter's mean latency within
    10% of the second quarter's, over windows ordered by commit)."""
    lat = []
    grow = []
    # Which windows each waits for, as the engine schedules them.
    deps = [w[5] for w in info]
    order = sorted(range(len(info)), key=lambda w: info[w][2])
    for row in times:
        ls = schedule(row, info, workers, deps)
        lat.extend(ls)
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


def calibrate(d, target, rounds=50, shots=20_000):
    """The SD6 p at which a rotated d memory's detectors fire as often as
    Willow's do (`target`), by bisection on the sampled detection fraction."""
    import stim
    lo, hi = 0.0005, 0.02
    for _ in range(14):
        p = math.sqrt(lo * hi)
        text = sq.generate_circuit("rotated", d, rounds, "sd6", p, 0.5, "z")
        n = stim.Circuit(text).num_detectors
        dets, _, _ = sq.sample_b8_batch(text, shots, 7, 0)
        bits = np.unpackbits(np.frombuffer(dets, np.uint8).reshape(shots, -1), axis=1, bitorder="little")[:, :n]
        lo, hi = (p, hi) if bits.mean() < target else (lo, p)
    return math.sqrt(lo * hi)


def cmd_million(args):
    rounds = args.rounds
    d = 5
    willow = load_json(G.OUT / "willow.json", dict(experiments={}))["experiments"].values()
    fracs = [r["mean_defects"] / r["detectors"] for r in willow if r["d"] == d and r["rounds"] >= 50]
    target = float(np.mean(fracs))
    p = calibrate(d, target)
    out = dict(generated=datetime.date.today().isoformat(), engine_commit=engine_commit(), machine=machine(),
               cycle_us=CYCLE * 1e6, d=d, rounds=rounds, p=p, willow_detection_fraction=target, runs=[])
    print(f"d = {d}: SD6 p = {p:.5f} matches Willow's detection fraction {target:.4f}", flush=True)
    for m, corr in MATCHERS.items():
        for mode in ("sliding", "parallel"):
            # Latency: one thread, so every window's time is one uncontended core's.
            failures, streams, unexplained, times_b, info, wall = sq.stream_decode(
                "rotated", d, p, rounds, d, d, mode, corr, 1, 11, 1)
            times = np.frombuffer(times_b, "<f8")
            # One batch, so one row: lane 0's time for every window of the stream.
            by_k = {k: latency_stats(times.reshape(1, -1), info, k)
                    for k in ((1,) if mode == "sliding" else (1, 2, 4, 8, 16))}
            run = dict(matcher=m, mode=mode, streams=streams, windows=len(info), unexplained=int(unexplained),
                       failures=int(failures), wall_seconds=wall,
                       rounds_per_second_one_thread=streams * rounds / wall,
                       window_us=dict(mean=float(times.mean() * 1e6), p50=float(np.percentile(times, 50) * 1e6),
                                      p99=float(np.percentile(times, 99) * 1e6), max=float(times.max() * 1e6)),
                       by_workers=by_k)
            out["runs"].append(run)
            row = "  ".join(f"K={k}: {v['mean_us']:.0f}/{v['p99_us']:.0f} us {'keeps up' if v['keeps_up'] else 'falls behind'}"
                            for k, v in by_k.items())
            print(f"{m:<10} {mode:<8} {streams} streams x {rounds:,} rounds in {wall:.0f} s, unexplained {unexplained}, "
                  f"window {run['window_us']['mean']:.1f} us  {row}", flush=True)
    # Throughput: every core, independent streams.
    cores = os.cpu_count()
    for m, corr in MATCHERS.items():
        failures, streams, unexplained, _, info, wall = sq.stream_decode(
            "rotated", d, p, rounds // 10, d, d, "parallel", corr, cores, 12, 0)
        out.setdefault("throughput", []).append(dict(matcher=m, streams=streams, rounds=rounds // 10, cores=cores,
                                                     unexplained=int(unexplained), wall_seconds=wall,
                                                     rounds_per_second=streams * (rounds // 10) / wall))
        print(f"throughput {m}: {streams} streams x {rounds // 10:,} rounds on {cores} cores in {wall:.0f} s: "
              f"{streams * (rounds // 10) / wall:,.0f} rounds/s", flush=True)
    write_json(OUT / "million.json", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["accuracy", "latency", "million"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--redo", action="store_true")
    ap.add_argument("--shots", type=int, default=2000)
    ap.add_argument("--rounds", type=int, default=1_000_000)
    args = ap.parse_args()
    if args.command == "accuracy":
        return cmd_accuracy(args)
    if args.command == "latency":
        return cmd_latency(args)
    return cmd_million(args)


if __name__ == "__main__":
    sys.exit(main() or 0)

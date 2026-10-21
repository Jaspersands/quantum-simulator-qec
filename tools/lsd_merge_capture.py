"""ldpc 2.4.1's BP+LSD answers on the QUITS sweep's window decodes, where ldpc does not give one
answer every time or does not give ours: tests/ldpc_lsd_merge/captured.json.

ldpc merges the clusters a growth step meets in the order of a `tsl::robin_set<LsdCluster *>`
(src_cpp/lsd.hpp:47, iterated at :193), which is the order of the clusters' hashed heap
addresses, so one syndrome decoded again in the same process can give another answer. This runs
the QUITS sweep's twelve BP+LSD configurations (HGP n=12, phenomenological sliding windows of 5
rounds committing 3, `lsd_order` 2, 10 BP iterations; both BP methods, both schedules, LSD-0,
LSD-E and LSD-CS) through this package's port of QUITS's sliding window, decodes each window's
syndrome where ldpc's BP does not converge `--reps` times with one ldpc decoder, and keeps the
windows whose answers vary or differ from this package's, with each answer's count and this
package's answer. tests/test_lsd_merge_orders.py checks that every one of ldpc's answers is
among the corrections `BpLsdDecoder._merge_order_outcomes` finds.

    python -I tools/lsd_merge_capture.py DETS [--path DIR] [--reps 50]

DETS is the sweep's detection events: an .npz whose `det` QUITS's notebooks 03-05 save (QUITS
1.2.0's HGP n=12 circuit, 15 rounds at p = 0.1%, 100 shots sampled by Stim with seed 1; the
circuit's gate order follows Python's string hashes, so a run without PYTHONHASHSEED gives other
events). `--path` names a directory to import this package from, for a Python that has ldpc but
not this package installed. Recorded on macOS: the answers are ldpc's there, whose sorts and
hashes are libc++'s.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import itertools
import json
import pathlib
import platform
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "tests" / "ldpc_lsd_merge" / "captured.json"
PCM = ROOT / "tests" / "quits_port" / "parity_check_matrices" / "n=12_dv=3_dc=4_dist=6.txt"
# QUITS's windows, and its phenomenological rate for p = 0.1%: p * (circuit depth 8 + 3).
W, F, EFF = 5, 3, 1e-3 * 11


def window_calls(det, bp_method, schedule, lsd_method):
    """The sweep's window decodes for one configuration, run with this package's decoder:
    the decoders' (matrix, arguments) and each call's (decoder, syndrome, this package's answer)."""
    import stabilizer_qec.quits.decoder.bplsd as port
    from stabilizer_qec.ldpc import BpLsdDecoder
    from stabilizer_qec.quits import qldpc_code

    decoders, calls = [], []

    class Recorder:
        def __init__(self, H, **kw):
            self.id, self.d = len(decoders), BpLsdDecoder(H, **kw)
            decoders.append((np.asarray(H.todense(), dtype=np.uint8), dict(kw)))

        def decode(self, s):
            out = self.d.decode(s)
            calls.append((self.id, np.asarray(s, dtype=np.uint8) % 2, np.asarray(out, dtype=np.uint8)))
            return out

    h = np.loadtxt(PCM, dtype=int)
    code = qldpc_code.HgpCode(h, h)
    kw = dict(eff_error_rate_per_fault=EFF, max_iter=10, lsd_order=2, bp_method=bp_method, schedule=schedule, lsd_method=lsd_method)
    real = port.BpLsdDecoder
    port.BpLsdDecoder = Recorder
    try:
        port.sliding_window_bplsd_phenom_mem(det, code.hz, code.lz, W, F, **kw)
    finally:
        port.BpLsdDecoder = real
    return decoders, calls


def columns(H):
    return [np.flatnonzero(H[:, j]).tolist() for j in range(H.shape[1])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dets")
    ap.add_argument("--path", default="")
    ap.add_argument("--reps", type=int, default=50)
    args = ap.parse_args()
    if args.path:
        sys.path.insert(0, args.path)
    from importlib.metadata import version

    import ldpc

    det = np.load(args.dets)["det"].astype(np.uint8)
    matrices, keys, windows = [], {}, []
    surveyed = unconverged = 0
    t0 = time.perf_counter()
    for lsd_method, schedule, bp_method in itertools.product(["lsd_0", "lsd_cs", "lsd_e"], ["serial", "parallel"], ["product_sum", "minimum_sum"]):
        decoders, calls = window_calls(det, bp_method, schedule, lsd_method)
        theirs = [ldpc.BpLsdDecoder(H, **kw) for H, kw in decoders]
        found = 0
        for did, s, ours in calls:
            surveyed += 1
            d = theirs[did]
            first = np.asarray(d.decode(s), dtype=np.uint8)
            if d.converge:
                continue  # BP's answer: no clusters, no merge order
            unconverged += 1
            answers = collections.Counter([first.tobytes()])
            for _ in range(args.reps - 1):
                answers[np.asarray(d.decode(s), dtype=np.uint8).tobytes()] += 1
            if len(answers) == 1 and ours.tobytes() in answers:
                continue
            H, kw = decoders[did]
            key = hashlib.sha256(H.tobytes() + repr(H.shape).encode()).hexdigest()
            if key not in keys:
                keys[key] = len(matrices)
                matrices.append(dict(rows=int(H.shape[0]), columns=columns(H)))
            windows.append(dict(
                matrix=keys[key], kw=kw, syndrome=np.flatnonzero(s).tolist(), ours=np.flatnonzero(ours).tolist(),
                ldpc=[dict(correction=np.flatnonzero(np.frombuffer(a, dtype=np.uint8)).tolist(), count=n) for a, n in answers.most_common()],
            ))
            found += 1
        print(f"  {bp_method:11} {schedule:8} {lsd_method:6} window decodes {len(calls)}, kept {found}", flush=True)
    doc = dict(
        about="ldpc's BP+LSD answers, over --reps decodes in one process, on the QUITS sweep's window decodes where they vary or differ from ours (tools/lsd_merge_capture.py)",
        ldpc=version("ldpc"), platform=f"{platform.system()}-{platform.machine().lower()}", reps=args.reps,
        dets_sha256=hashlib.sha256(det.tobytes()).hexdigest(), surveyed=surveyed, unconverged=unconverged,
        matrices=matrices, windows=windows,
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, separators=(",", ":")) + "\n")
    print(f"{surveyed} window decodes, {unconverged} where BP did not converge, {len(windows)} kept, {time.perf_counter() - t0:.0f} s: wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

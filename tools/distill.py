"""The 15-to-1 magic-state distillation law, measured here.

Run from the repository root, with the engine built into the virtualenv:

    python tools/distill.py     # -> data/estimate/distill.json

Fifteen faulty T states, their errors twirled to Z, are checked by the [[15, 1, 3]] code's four
X-type stabilizers, whose supports are the rows of the Hamming [15, 11] parity-check matrix
(qubit j is in check k when bit k of j + 1 is set). A pattern of Z errors passes when its
syndrome is zero, and the output is wrong when it passes with odd weight (it anticommutes with
the logical X, all fifteen). The weight-3 codewords of the Hamming code are 35, so the output
error is 35 p^3 to leading order, and a state is kept with probability 1 - 15 p + O(p^2).

Two ways, which must agree:
- exactly, summing over all 2^15 patterns;
- sampled by this engine: 15 qubits prepared in |+>, Z errors, read in X, the four checks as
  detectors and the logical X as the observable; a shot is kept when no detector fires. They
  agree when the exact values lie in the samples' 99.9% Wilson intervals (95% recorded).
This is the law Litinski's factories are built on (their circuits add Clifford noise at
distance-d patches, which this does not model).
"""

import datetime
import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import stabilizer_qec as sq  # noqa: E402
from realtime import engine_commit, machine, write_json  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "estimate" / "distill.json"
PS = [0.001, 0.003, 0.01, 0.03, 0.1]
SHOTS = 50_000_000


def wilson(k, n, z=1.96):
    if n == 0:
        return [0.0, 1.0]
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [max(0.0, mid - half), min(1.0, mid + half)]


def exact(p):
    """(probability kept, probability kept and wrong), over every Z-error pattern."""
    kept = wrong = 0.0
    for e in range(1 << 15):
        s = 0
        for j in range(15):
            if e >> j & 1:
                s ^= j + 1
        if s:
            continue
        w = bin(e).count("1")
        pr = p ** w * (1 - p) ** (15 - w)
        kept += pr
        if w % 2:
            wrong += pr
    return kept, wrong


def circuit(p):
    q = " ".join(str(j) for j in range(15))
    lines = [f"RX {q}", f"Z_ERROR({p}) {q}", f"MX {q}"]
    for k in range(4):
        lines.append("DETECTOR " + " ".join(f"rec[-{15 - j}]" for j in range(15) if (j + 1) >> k & 1))
    lines.append("OBSERVABLE_INCLUDE(0) " + " ".join(f"rec[-{15 - j}]" for j in range(15)))
    return "\n".join(lines) + "\n"


def sampled(p, seed):
    kept = wrong = 0
    batch = 65_536
    for b in range(SHOTS // batch):
        dets, obs, _ = sq.sample_b8_batch(circuit(p), batch, seed + b, 0)
        d = np.frombuffer(dets, np.uint8).reshape(batch, -1)[:, 0] & 0xF
        o = np.frombuffer(obs, np.uint8).reshape(batch, -1)[:, 0] & 1
        keep = d == 0
        kept += int(keep.sum())
        wrong += int((keep & (o == 1)).sum())
    return kept, wrong, (SHOTS // batch) * batch


def main():
    points, ok = [], True
    for i, p in enumerate(PS):
        k_exact, w_exact = exact(p)
        kept, wrong, shots = sampled(p, 5_000_011 * (i + 1))
        out_exact = w_exact / k_exact
        acc_lo, acc_hi = wilson(kept, shots)
        out_lo, out_hi = wilson(wrong, kept)
        # Agreement is judged at 99.9%: five points at 95% would disagree by chance a quarter of the time.
        (a_lo, a_hi), (o_lo, o_hi) = wilson(kept, shots, 3.29), wilson(wrong, kept, 3.29)
        agree = a_lo <= k_exact <= a_hi and o_lo <= out_exact <= o_hi
        ok &= agree
        points.append(dict(p=p, shots=shots, kept=kept, wrong=wrong, acceptance=kept / shots, acceptance_interval=[acc_lo, acc_hi],
                           output=wrong / kept, output_interval=[out_lo, out_hi], exact_acceptance=k_exact,
                           exact_output=out_exact, output_over_35p3=out_exact / (35 * p ** 3),
                           acceptance_vs_1_minus_15p=(1 - k_exact) / (15 * p), agrees=bool(agree)))
        print(f"p {p:<6} kept {kept / shots:.5f} (exact {k_exact:.5f})  output {wrong / kept:.3e} [{out_lo:.2e}, {out_hi:.2e}] "
              f"(exact {out_exact:.3e}, / 35p³ = {out_exact / (35 * p ** 3):.3f})  {'ok' if agree else 'DISAGREES'}", flush=True)
    write_json(OUT, dict(engine_commit=engine_commit(), machine=machine(), generated=datetime.date.today().isoformat(),
                         note="15-to-1 at the logical level: twirled Z errors on fifteen T states, the [[15, 1, 3]] code's "
                              "X checks; exact over all 2^15 patterns and sampled by this engine", points=points))
    print("ALL AGREE" if ok else "SOME DISAGREE")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

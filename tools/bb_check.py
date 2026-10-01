"""The bivariate bicycle codes and BP+OSD, checked against independent references.

Run from the repository root, with the engine built into a virtualenv that
also has stim and ldpc:

    python tools/bb_check.py            # writes data/gross/bb-check.json
    python tools/bb_check.py --quick    # fewer cases, writes nothing

Checks:
  1. The check matrices equal ones built independently here, as Bravyi et al.'s
     published code builds them (Kronecker products of cyclic shifts), and
     the logical operators pair and commute with the checks.
  2. Stim reads the memory circuit, and its error model (undecomposed) equals
     ours fault for fault, symptoms merged, to 1e-9 relative.
  3. Every detector of the noiseless circuit is deterministic (Stim's own
     reference sample agrees with ours: all zero).
  4. BP+OSD equals ldpc's BpOsdDecoder on identical syndromes of the memory
     experiment's model: the same correction on every shot, OSD-0 and OSD-CS.
Exits non-zero if any check fails.
"""

import argparse
import json
import pathlib
import sys

import numpy as np
import stim
from ldpc import BpOsdDecoder
from scipy.sparse import csc_matrix

import stabilizer_qec._core as sq

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "gross"
CODES = {"72": (6, 6), "gross": (12, 6)}


def bravyi_matrices(ell, m):
    """H_X and H_Z exactly as Bravyi et al.'s decoder_setup.py builds them."""
    x = np.kron(np.roll(np.identity(ell, dtype=int), 1, axis=1), np.identity(m, dtype=int))
    y = np.kron(np.identity(ell, dtype=int), np.roll(np.identity(m, dtype=int), 1, axis=1))
    mp = np.linalg.matrix_power
    a = (mp(x, 3) + mp(y, 1) + mp(y, 2)) % 2
    b = (mp(y, 3) + mp(x, 1) + mp(x, 2)) % 2
    return np.hstack((a, b)), np.hstack((b.T, a.T))


def dense(rows, n):
    out = np.zeros((len(rows), n), dtype=int)
    for i, r in enumerate(rows):
        out[i, r] = 1
    return out


def check_code(name):
    ell, m = CODES[name]
    hx_rows, hz_rows, lx_rows, lz_rows = sq.bb_matrices(name)
    n = 2 * ell * m
    hx, hz, lx, lz = dense(hx_rows, n), dense(hz_rows, n), dense(lx_rows, n), dense(lz_rows, n)
    bx, bz = bravyi_matrices(ell, m)
    ok = np.array_equal(hx, bx) and np.array_equal(hz, bz)
    ok &= not ((hx @ hz.T) % 2).any()
    ok &= not ((hx @ lz.T) % 2).any() and not ((hz @ lx.T) % 2).any()
    ok &= np.array_equal((lx @ lz.T) % 2, np.identity(len(lx), dtype=int))
    print(f"  {'ok ' if ok else 'BAD'} [[{n}, {len(lx)}]] ({ell}×{m}): H_X and H_Z equal Bravyi et al.'s construction; "
          f"{len(lx)} logical pairs, paired and commuting with the checks")
    return ok, dict(code=name, n=n, k=len(lx), matches_bravyi=bool(ok))


def mechanisms(dem):
    out = {}
    for inst in dem.flattened():
        if inst.type != "error":
            continue
        key = tuple(sorted(str(t) for t in inst.targets_copy() if not t.is_separator()))
        q = inst.args_copy()[0]
        p = out.get(key, 0.0)
        out[key] = p * (1 - q) + q * (1 - p)
    return out


def check_model(name, cycles, p):
    text = sq.bb_memory_circuit(name, cycles, p)
    circuit = stim.Circuit(text)
    ours = mechanisms(stim.DetectorErrorModel(sq.dem_from_circuit(text, False)))
    theirs = mechanisms(circuit.detector_error_model(decompose_errors=False))
    same_keys = set(ours) == set(theirs)
    worst = max((abs(ours[k] - theirs[k]) / theirs[k] for k in theirs if k in ours), default=0.0)
    noiseless = stim.Circuit(sq.bb_memory_circuit(name, cycles, 0.0))
    deterministic = not noiseless.compile_detector_sampler(seed=1).sample(64).any()
    ok = same_keys and worst < 1e-9 and deterministic
    print(f"  {'ok ' if ok else 'BAD'} {name}, {cycles} cycles, p = {p}: {circuit.num_detectors} detectors, {len(theirs)} "
          f"mechanisms, ours {len(ours)}; {len(set(ours) ^ set(theirs))} one-sided; worst Δp/p {worst:.1e}; "
          f"noiseless detectors all zero: {deterministic}")
    return ok, dict(code=name, cycles=cycles, p=p, detectors=circuit.num_detectors, mechanisms=len(theirs),
                    ours=len(ours), one_sided=len(set(ours) ^ set(theirs)), max_rel=worst, deterministic=bool(deterministic))


def check_bposd(name, cycles, p, shots, osd, order, max_iter):
    text = sq.bb_memory_circuit(name, cycles, p)
    circuit = stim.Circuit(text)
    dem_text = sq.dem_from_circuit(text, False)
    dem = stim.DetectorErrorModel(dem_text)
    # The model's columns, in our model's order.
    columns, priors = [], []
    for inst in dem.flattened():
        if inst.type == "error":
            columns.append([t.val for t in inst.targets_copy() if t.is_relative_detector_id()])
            priors.append(inst.args_copy()[0])
    nd = dem.num_detectors
    r = [i for c in columns for i in c]
    c = [j for j, col in enumerate(columns) for _ in col]
    h = csc_matrix((np.ones(len(r), np.uint8), (r, c)), shape=(nd, len(columns)))
    ref = BpOsdDecoder(h, error_channel=priors, max_iter=max_iter, bp_method="minimum_sum", ms_scaling_factor=0.0,
                       osd_method=osd, osd_order=order, schedule="parallel")
    dets = circuit.compile_detector_sampler(seed=cycles * 1000 + int(p * 1e5)).sample(shots)
    same = conv = 0
    for s in dets:
        theirs = ref.decode(s.astype(np.uint8))
        ours, converged, _ = sq.bposd_decode(nd, columns, priors, s.astype(np.uint8).tolist(), max_iter, "minimum_sum", 0.0,
                                             osd, order)
        same += int(np.array_equal(np.array(ours, dtype=np.uint8), theirs.astype(np.uint8)))
        conv += int(converged)
    ok = same == shots
    print(f"  {'ok ' if ok else 'BAD'} {name}, {cycles} cycles, p = {p}, {osd} {order}, {max_iter} iterations: "
          f"{same} of {shots} corrections equal ldpc's; BP converged on {conv}")
    return ok, dict(code=name, cycles=cycles, p=p, osd=osd, order=order, max_iter=max_iter, shots=shots, same=same,
                    converged=conv)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    ok, report = True, dict(codes=[], models=[], bposd=[])
    print("Check 1: the codes")
    for name in CODES:
        good, r = check_code(name)
        ok &= good
        report["codes"].append(r)
    print("Checks 2 and 3: the circuits' error models against Stim's, and determinism")
    for name, cycles in ([("72", 2)] if args.quick else [("72", 3), ("gross", 2), ("gross", 12)]):
        good, r = check_model(name, cycles, 0.003)
        ok &= good
        report["models"].append(r)
    print("Check 4: BP+OSD against ldpc's BpOsdDecoder")
    cases = ([("72", 2, 0.004, 40, "osd_cs", 7, 100)] if args.quick else
             [("72", 6, 0.004, 300, "osd_0", 0, 1000), ("72", 6, 0.004, 300, "osd_cs", 7, 1000),
              ("gross", 3, 0.004, 100, "osd_cs", 7, 1000)])
    for case in cases:
        good, r = check_bposd(*case)
        ok &= good
        report["bposd"].append(r)
    if not args.quick:
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "bb-check.json").write_text(json.dumps(report, indent=1) + "\n")
    print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

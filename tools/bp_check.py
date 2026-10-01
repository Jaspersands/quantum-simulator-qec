"""Belief propagation and belief-matching, checked against the authors' packages.

Run from the repository root, with the engine built into .venv and the
oracles installed (`.venv/bin/pip install ldpc beliefmatching`):

    .venv/bin/python tools/bp_check.py            # writes data/belief/bp-check.json
    .venv/bin/python tools/bp_check.py --quick    # fewer cases, writes nothing

Checks:
  1. BP against ldpc's BpDecoder (flooding schedule): the same hard decision,
     convergence and iteration count, and posterior log-likelihood ratios equal
     to rounding. Random parity-check matrices, both methods, 1, 3 and 20
     iterations, and the hypergraph of a surface-code error model.
  2. Belief-matching against beliefmatching's BeliefMatching on identical
     shots of SD6 memories at d = 3, 5, 7: the same BP convergence on every
     shot, and the same prediction except where the two matchers tie. A
     disagreement is a tie if our matching's weight on the shot's posterior
     weights equals PyMatching's optimum on the same weights, to within the
     discretisation noise measured on the shots where the two agree.
Exits non-zero if any check fails.
"""

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import pymatching
import stim
from beliefmatching import BeliefMatching, detector_error_model_to_check_matrices
from ldpc import BpDecoder

import stabilizer_qec._core as sq

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "belief"
FAILED = np.uint64(2**64 - 1)


def columns_of(h):
    h = h.tocsc()
    return [h.indices[h.indptr[j]:h.indptr[j + 1]].tolist() for j in range(h.shape[1])]


def compare_bp(h, priors, syndromes, method, scale, iters):
    """Worst differences between ours and ldpc's over the syndromes."""
    cols = columns_of(h)
    ref = BpDecoder(h, error_channel=list(priors), max_iter=iters, bp_method=method, ms_scaling_factor=scale,
                    schedule="parallel", input_vector_type="syndrome")
    worst = dict(llr_abs=0.0, llr_rel=0.0, hard=0, converge=0, iterations=0, cases=0)
    for s in syndromes:
        theirs_hard = ref.decode(s.astype(np.uint8))
        hard, llr, conv, it = sq.bp_decode(h.shape[0], cols, list(priors), s.astype(np.uint8).tolist(), iters, method, scale)
        worst["cases"] += 1
        if not s.any():
            worst["converge"] += int(not conv)
            continue
        llr, theirs = np.array(llr), np.array(ref.log_prob_ratios)
        diff = np.abs(llr - theirs)
        worst["llr_abs"] = max(worst["llr_abs"], float(diff.max()))
        worst["llr_rel"] = max(worst["llr_rel"], float((diff / np.maximum(np.abs(theirs), 1.0)).max()))
        worst["hard"] += int((np.array(hard) != theirs_hard).any())
        worst["converge"] += int(conv != ref.converge)
        worst["iterations"] += int(it != ref.iter)
    return worst


def check_bp(quick):
    rng = np.random.default_rng(7)
    rows = []
    # Random sparse matrices: 40 checks, 80 variables, 2 to 4 checks a column.
    from scipy.sparse import csc_matrix
    for trial in range(2 if quick else 6):
        m, n = 40, 80
        r, c = [], []
        for j in range(n):
            for i in rng.choice(m, size=rng.integers(2, 5), replace=False):
                r.append(i)
                c.append(j)
        h = csc_matrix((np.ones(len(r), np.uint8), (r, c)), shape=(m, n))
        priors = rng.uniform(0.005, 0.15, n)
        errors = (rng.random((60, n)) < priors).astype(np.uint8)
        syndromes = (h @ errors.T % 2).T
        for method, scale in [("product_sum", 1.0), ("minimum_sum", 1.0), ("minimum_sum", 0.625)]:
            for iters in (1, 3, 20):
                w = compare_bp(h, priors, syndromes, method, scale, iters)
                rows.append(dict(source=f"random {trial}", method=method, scale=scale, iters=iters, **w))
    # A surface code's hypergraph.
    circuit = stim.Circuit(sq.generate_circuit("rotated", 5, 5, "sd6", 0.006))
    mats = detector_error_model_to_check_matrices(circuit.detector_error_model(decompose_errors=True))
    dets = circuit.compile_detector_sampler(seed=3).sample(100 if quick else 400)
    for method, scale in [("product_sum", 1.0), ("minimum_sum", 1.0)]:
        w = compare_bp(mats.check_matrix, mats.priors, dets, method, scale, 20)
        rows.append(dict(source="rotated d=5 SD6 p=0.6%", method=method, scale=scale, iters=20, **w))
    ok = all(r["hard"] == 0 and r["converge"] == 0 and r["iterations"] == 0 and r["llr_rel"] < 1e-9 for r in rows)
    for r in rows:
        print(f"  {'ok ' if r['hard'] == r['converge'] == r['iterations'] == 0 and r['llr_rel'] < 1e-9 else 'BAD'} "
              f"{r['source']:<24} {r['method']:<11} x{r['scale']:<5} {r['iters']:>2} iterations: {r['cases']} syndromes, "
              f"LLR differs by at most {r['llr_abs']:.1e} ({r['llr_rel']:.1e} relative); hard {r['hard']}, "
              f"convergence {r['converge']}, iteration counts {r['iterations']} differ")
    return ok, rows


def check_belief(quick):
    rows, ok = [], True
    for d, p in ([(3, 0.006), (5, 0.006)] if quick else [(3, 0.004), (3, 0.008), (5, 0.004), (5, 0.008), (7, 0.004), (7, 0.008)]):
        shots = 1000 if quick else (4000 if d < 7 else 2000)
        circuit = stim.Circuit(sq.generate_circuit("rotated", d, d, "sd6", p))
        dem = circuit.detector_error_model(decompose_errors=True)
        dets, obs = circuit.compile_detector_sampler(seed=10 * d + int(p * 1e3)).sample(shots, separate_observables=True)
        truth = obs[:, 0]
        packed = np.packbits(dets, axis=1, bitorder="little").tobytes()
        raw_b, w_b, conv_b, errors, seconds = sq.decode_b8_belief(str(dem), packed, shots, threads=1)
        raw = np.frombuffer(raw_b, "<u8")
        ours = (raw & np.uint64(1)).astype(np.uint8)
        our_w = np.frombuffer(w_b, "<f8")
        our_conv = np.frombuffer(conv_b, np.uint8).astype(bool)

        bm = BeliefMatching(dem, max_bp_iters=20)
        t = time.perf_counter()
        theirs = bm.decode_batch(dets)[:, 0].astype(np.uint8)
        their_seconds = time.perf_counter() - t

        # BP's convergence on each shot, and PyMatching's optimum on the
        # shot's posterior weights, as beliefmatching computes them.
        mats = bm._matrices

        def their_bp(i):
            bm._bpd.decode(dets[i].astype(np.uint8))
            return bm._bpd.converge

        def their_weight(i):
            bm._bpd.decode(dets[i].astype(np.uint8))
            ps_h = 1 / (1 + np.exp(bm._bpd.log_prob_ratios))
            ps_e = np.clip(mats.hyperedge_to_edge_matrix @ ps_h, 1e-14, 1 - 1e-14)
            m = pymatching.Matching.from_check_matrix(mats.edge_check_matrix, weights=-np.log(ps_e),
                                                      faults_matrix=mats.edge_observables_matrix,
                                                      use_virtual_boundary_node=True)
            return m.decode(dets[i], return_weight=True)[1]

        conv_differ = sum(1 for i in range(shots) if their_bp(i) != our_conv[i])
        agree = np.nonzero((ours == theirs) & ~our_conv & (raw != FAILED))[0][:200]
        noise = max((abs(their_weight(i) - our_w[i]) for i in agree), default=0.0)
        tol = max(10 * noise, 1e-6)
        disagree = np.nonzero(ours != theirs)[0]
        non_ties = sum(1 for i in disagree if our_conv[i] or abs(their_weight(i) - our_w[i]) > tol)
        r = dict(d=d, p=p, shots=shots, ours_failures=int((ours != truth).sum()), theirs_failures=int((theirs != truth).sum()),
                 converged=int(our_conv.sum()), convergence_differs=conv_differ, disagreements=int(len(disagree)),
                 non_ties=int(non_ties), weight_noise=float(noise), errors=int(errors),
                 ours_us=seconds / shots * 1e6, theirs_us=their_seconds / shots * 1e6)
        good = conv_differ == 0 and non_ties == 0 and errors == 0
        ok &= good
        rows.append(r)
        print(f"  {'ok ' if good else 'BAD'} d={d} p={p:.3f} {shots} shots: beliefmatching {r['theirs_failures']}, ours "
              f"{r['ours_failures']} failures; BP converged on {r['converged']} (differs on {conv_differ}); "
              f"{r['disagreements']} disagreements, {non_ties} not ties (weight noise {noise:.1e}); "
              f"{r['ours_us']:.0f} us vs {r['theirs_us']:.0f} us a shot")
    return ok, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    from importlib.metadata import version
    print(f"ldpc {version('ldpc')}, beliefmatching {version('beliefmatching')}, "
          f"pymatching {pymatching.__version__}, stim {stim.__version__}")
    print("Check 1: BP against ldpc")
    ok1, bp_rows = check_bp(args.quick)
    print("Check 2: belief-matching against beliefmatching")
    ok2, bm_rows = check_belief(args.quick)
    if not args.quick:
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "bp-check.json").write_text(json.dumps(dict(bp=bp_rows, belief_matching=bm_rows), indent=1) + "\n")
    print("ALL CHECKS PASSED" if ok1 and ok2 else "SOME CHECKS FAILED")
    return 0 if ok1 and ok2 else 1


if __name__ == "__main__":
    sys.exit(main())

"""Logical operations on the gross code (sub-project J), checked independently.

Run from the repository root, with the engine built into a virtualenv that
also has numpy, scipy, stim and ldpc:

    python tools/gross_ops.py auto        # automorphisms, recomputed here -> data/gross/automorphisms.json
    python tools/gross_ops.py distance    # the deformed codes' distances, exactly -> data/gross/gauging.json
    python tools/gross_ops.py check       # the circuits against Stim and ldpc (--quick: X(f, 0) only)
    python tools/gross_ops.py measure     # failure against merged cycles, beside the memory -> data/gross/logical.json

auto: every shift x^a y^b of both halves of the gross code, with and without
the ZX-duality, is built here with numpy, independently of src/bb_auto.rs,
checked to map the stabilizer group to itself, and its action on the 12
logical qubits computed in the basis `bb_matrices` gives. It must equal the
engine's bit for bit, for all 144 maps. The group order, the shifts acting
trivially, and the two families of translates are recomputed here too.

distance: the distance of the deformed code while each operator is measured,
for Cross et al.'s minimal ancilla system and for the expanded one (edges
added until its Cheeger constant is at least 1), and of the gross code itself
(the method's own check), exactly, by integer programming (scipy's HiGHS). For each logical of the other type in a basis,
the lightest operator commuting with the checks and anticommuting with it;
the least of those is the distance, since every nontrivial logical
anticommutes with some basis element. A solve that hits the time cap is
recorded as a bound.

check: each logical-measurement circuit's error model equals Stim's fault for
fault, Stim finds the noiseless circuit deterministic, and BP+OSD's
corrections equal ldpc's BpOsdDecoder's on Stim's shots, but for shots where
the two differ only by the order of tied BP posteriors (see bposd_matches).

measure: the measurement against merged cycles T (the expanded system, and the
minimal one for comparison), and the memory of the same length, sampled by
this engine and decoded by BP+OSD-CS of order 7.
"""

import argparse
import datetime
import json
import math
import pathlib
import sys
import time

import numpy as np
import stim
from ldpc import BpDecoder, BpOsdDecoder, mod2
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csc_matrix, csr_matrix, hstack, identity

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import stabilizer_qec as sq  # noqa: E402
from bb_check import mechanisms  # noqa: E402
from realtime import engine_commit, load_json, machine, write_json  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "gross"
L, M = 12, 6
H = L * M
OPERATORS = ["f", "gh", "f+gh"]
CONSTRUCTIONS = ["minimal", "expanded"]
BATCH = 2_048


def dense(rows, n):
    out = np.zeros((len(rows), n), np.uint8)
    for i, r in enumerate(rows):
        out[i, r] = 1
    return out


def rank(m):
    return int(mod2.rank(np.asarray(m, np.uint8) % 2))


def code_matrices():
    hx, hz, lx, lz = sq.bb_matrices("gross")
    return dense(hx, 2 * H), dense(hz, 2 * H), dense(lx, 2 * H), dense(lz, 2 * H)


# -- auto --------------------------------------------------------------------

def cell(u, v):
    return (u % L) * M + (v % M)


def data_map(a, b, dual):
    """Where each data qubit goes: cell (u, v) to (u + a, v + b) on both halves;
    for the duality, then left g to right g^-1 and right g to left g^-1."""
    out = np.empty(2 * H, int)
    for side in (0, 1):
        for u in range(L):
            for v in range(M):
                mu, mv = u + a, v + b
                out[side * H + cell(u, v)] = (1 - side) * H + cell(-mu, -mv) if dual else side * H + cell(mu, mv)
    return out


def poly(terms):
    v = np.zeros(H, np.uint8)
    for i, j in terms:
        v[cell(i, j)] ^= 1
    return v


F = [(0, 0), (1, 0), (2, 0), (3, 0), (6, 0), (7, 0), (8, 0), (9, 0), (1, 3), (5, 3), (7, 3), (11, 3)]
G = [(1, 0), (2, 1), (0, 2), (1, 2), (2, 3), (0, 4)]
HP = [(0, 0), (0, 1), (1, 1), (0, 2), (0, 3), (1, 3)]


def operator_vector(name):
    xf = np.concatenate([poly(F), np.zeros(H, np.uint8)])
    xgh = np.concatenate([poly(G), poly(HP)])
    return {"f": xf, "gh": xgh, "f+gh": (xf + xgh) % 2}[name]


def cmd_auto(args):
    HX, HZ, LX, LZ = code_matrices()
    rx, rz = rank(HX), rank(HZ)
    engine = {(a, b, d): dense(rows, 24) for a, b, d, rows in sq.bb_automorphisms("gross")}
    ok = len(engine) == 144
    actions, trivial = {}, []
    for a in range(L):
        for b in range(M):
            for dual in (False, True):
                perm = data_map(a, b, dual)
                moved = lambda m: np.array([np.bincount(perm[np.flatnonzero(r)], minlength=2 * H) % 2 for r in m], np.uint8)
                px, pz = moved(HX), moved(HZ)
                to_x, to_z = (HZ, HX) if dual else (HX, HZ)
                ok &= rank(np.vstack([to_x, px])) == rank(to_x) and rank(np.vstack([to_z, pz])) == rank(to_z)
                act = np.zeros((24, 24), np.uint8)
                for i in range(24):
                    src = LX[i] if i < 12 else LZ[i - 12]
                    img = np.zeros(2 * H, np.uint8)
                    img[perm[np.flatnonzero(src)]] = 1
                    is_x = (i < 12) != dual
                    pair, basis, stab, r, off = (LZ, LX, HX, rx, 0) if is_x else (LX, LZ, HZ, rz, 12)
                    coeff = (pair.astype(int) @ img) % 2
                    act[i, off:off + 12] = coeff
                    residual = (img + coeff @ basis) % 2
                    ok &= rank(np.vstack([stab, residual])) == r
                omega = np.block([[np.zeros((12, 12), int), np.eye(12, dtype=int)], [np.eye(12, dtype=int), np.zeros((12, 12), int)]])
                ok &= np.array_equal((act.astype(int) @ omega @ act.T) % 2, omega)
                ok &= np.array_equal(act, engine[(a, b, dual)])
                actions[(a, b, dual)] = act
                if not dual and np.array_equal(act, np.eye(24, dtype=np.uint8)):
                    trivial.append([a, b])
    keys = {a.tobytes() for a in actions.values()}
    closed = all(((x.astype(int) @ y) % 2).astype(np.uint8).tobytes() in keys for x in actions.values() for y in actions.values())
    ok &= closed
    families = {}
    for name in ("f", "gh"):
        op = operator_vector(name)
        ok &= np.array_equal(np.flatnonzero(op), sq.bb_gauging(name)[0])
        translates = []
        for a in range(L):
            for b in range(M):
                t = np.zeros(2 * H, np.uint8)
                t[data_map(a, b, False)[np.flatnonzero(op)]] = 1
                translates.append(t)
        span = rank(np.vstack([HX] + translates)) - rx
        reps = []
        for t in translates:
            if not any(rank(np.vstack([HX, (t + r) % 2])) == rx for r in reps):
                reps.append(t)
        families[name] = dict(span=span, classes=len(reps))
    doc = dict(engine_commit=engine_commit(), machine=machine(), generated=datetime.date.today().isoformat(),
               note="Automorphisms of the gross code: shifts x^a y^b of both halves, with and without the ZX-duality, "
                    "recomputed with numpy and ldpc's GF(2) rank and compared with the engine's bit for bit",
               maps=144, all_equal_engine=bool(ok), order=len(keys), closed=bool(closed), trivial_shifts=trivial,
               families=families)
    print(f"  {'ok ' if ok else 'BAD'} 144 maps, every one an automorphism with a symplectic action equal to the engine's; "
          f"group order {len(keys)} (closed: {closed}); trivial shifts {trivial}; families {families}")
    write_json(OUT / "automorphisms.json", doc)
    return 0 if ok else 1


# -- distance ----------------------------------------------------------------

def logicals_of(stabilizers, other):
    """Operators commuting with `other`'s checks, outside `stabilizers`' rowspace: a basis."""
    kernel = mod2.nullspace(csr_matrix(other))
    kernel = kernel.toarray() if hasattr(kernel, "toarray") else np.asarray(kernel)
    cur, out = stabilizers.copy(), []
    r = rank(cur)
    for v in kernel % 2:
        t = np.vstack([cur, v])
        if rank(t) > r:
            cur, r = t, r + 1
            out.append(v.astype(np.uint8))
    return np.array(out, np.uint8)


def lightest(checks, dual, cap):
    """The least weight of an operator commuting with every row of `checks` and
    anticommuting with `dual`: (value or None, lower bound, exact)."""
    a = np.vstack([checks, dual]).astype(float)
    n, r = a.shape[1], a.shape[0]
    cons = hstack([csr_matrix(a), -2 * identity(r, format="csr")])
    rhs = np.zeros(r)
    rhs[-1] = 1
    cost = np.concatenate([np.ones(n), np.zeros(r)])
    upper = np.concatenate([np.ones(n), np.full(r, n // 2 + 1)])
    res = milp(cost, constraints=LinearConstraint(cons, rhs, rhs), integrality=np.ones(n + r), bounds=Bounds(0, upper),
               options={"time_limit": cap})
    value = None if res.x is None else int(round(res.fun))
    exact = res.status == 0
    bound = value if exact else int(math.ceil((getattr(res, "mip_dual_bound", 0) or 0) - 1e-6))
    return value, bound, exact


def distance(hx, hz, cap):
    out = {}
    for kind, checks, stabilizers in (("z", hx, hz), ("x", hz, hx)):
        # A Z logical commutes with the X checks and anticommutes with some X logical.
        duals = logicals_of(hx, hz) if kind == "z" else logicals_of(hz, hx)
        t0 = time.perf_counter()
        solves = [lightest(checks, d, cap) for d in duals]
        found = [v for v, _, _ in solves if v is not None]
        out[kind] = dict(value=min(found) if found else None, lower=min(b for _, b, _ in solves),
                         exact=all(e for _, _, e in solves), solves=len(solves),
                         seconds=round(time.perf_counter() - t0, 1))
        print(f"    {kind}-distance {out[kind]}", flush=True)
    return out


def cmd_distance(args):
    path = OUT / "gauging.json"
    doc = load_json(path, dict(operators={}))
    # Entries written before there were two constructions are the minimal ones.
    for name, rec in list(doc["operators"].items()):
        if "distance" in rec:
            doc["operators"][name] = dict(minimal=rec)
    doc.update(engine_commit=engine_commit(), machine=machine(), generated=datetime.date.today().isoformat(),
               note="The gauging ancilla system of each operator (Cross et al.'s mono-layer construction, built from its "
                    "definition), and the deformed code's distance by integer programming (HiGHS), exact unless marked")
    names = [args.operator] if args.operator else ["code"] + OPERATORS
    constructions = [args.construction] if args.construction else CONSTRUCTIONS
    for name in names:
        if name == "code":
            if "code" in doc and not args.redo:
                continue
            print("  code", flush=True)
            hx, hz, _, _ = code_matrices()
            doc["code"] = dict(distance=distance(hx, hz, args.cap))
            write_json(path, doc)
            continue
        for construction in constructions:
            if construction in doc["operators"].get(name, {}) and not args.redo:
                continue
            print(f"  {name}, {construction}", flush=True)
            support, edges, extra, incidence, gauss, flux, hxr, hzr, ticks, cut = sq.bb_gauging(name, construction == "expanded")
            n = 2 * H + len(incidence)
            hx, hz = dense(hxr, n), dense(hzr, n)
            degree = max(int(hx[:, q].sum() + hz[:, q].sum()) for q in range(n))
            degree = max(degree, int(hx.sum(axis=1).max()), int(hz.sum(axis=1).max()))
            rec = dict(weight=len(support), edges=len(incidence), extra=len(extra), gauss=len(support), flux=len(flux),
                       ancillas=len(incidence) + len(support) + len(flux), flux_weights=[len(f) for f in flux],
                       ticks=ticks, max_degree=degree, logical_qubits=n - rank(hx) - rank(hz), worst_cut=list(cut),
                       support=support, edge_checks=edges, extra_edges=[list(e) for e in extra], incidence=incidence,
                       flux_cycles=flux)
            rec["distance"] = distance(hx, hz, args.cap)
            doc["operators"].setdefault(name, {})[construction] = rec
            write_json(path, doc)
    return 0


# -- check -------------------------------------------------------------------

def bposd_matches(text, shots, seed):
    """BP+OSD-CS7 against ldpc's BpOsdDecoder on Stim's shots: (equal, ties, unexplained).

    OSD sorts the columns by BP's posteriors, and min-sum run long leaves many
    exactly equal. ldpc sorts them with C++'s std::sort, whose order among
    equals is the standard library's (libc++ and libstdc++ differ), so where
    tied columns straddle the information set, OSD's answer depends on it:
    ldpc's own answer changes when its columns are permuted. A shot whose
    correction differs is a tie when BP's posteriors are bit-identical to
    ldpc's and ldpc itself returns our exact correction for some permutation
    of the columns; otherwise it is unexplained.
    """
    circuit = stim.Circuit(text)
    dem = stim.DetectorErrorModel(sq.dem_from_circuit(text, False))
    columns, priors = [], []
    for inst in dem.flattened():
        if inst.type == "error":
            columns.append([t.val for t in inst.targets_copy() if t.is_relative_detector_id()])
            priors.append(inst.args_copy()[0])
    nd, n = dem.num_detectors, len(columns)
    priors = np.array(priors)

    def matrix(order):
        cols = [columns[j] for j in order]
        r = [i for c in cols for i in c]
        c = [j for j, col in enumerate(cols) for _ in col]
        return csc_matrix((np.ones(len(r), np.uint8), (r, c)), shape=(nd, n))

    kw = dict(max_iter=1000, bp_method="minimum_sum", ms_scaling_factor=0.0, schedule="parallel")
    ref = BpOsdDecoder(matrix(np.arange(n)), error_channel=list(priors), osd_method="osd_cs", osd_order=7, **kw)
    bp = BpDecoder(matrix(np.arange(n)), error_channel=list(priors), input_vector_type="syndrome", **kw)
    rng = np.random.default_rng(seed)
    same = ties = 0
    for s in circuit.compile_detector_sampler(seed=seed).sample(shots).astype(np.uint8):
        theirs = ref.decode(s).astype(np.uint8)
        ours, _, _ = sq.bposd_decode(nd, columns, list(priors), s.tolist(), 1000, "minimum_sum", 0.0, "osd_cs", 7)
        ours = np.array(ours, np.uint8)
        if np.array_equal(ours, theirs):
            same += 1
            continue
        bp.decode(s)
        _, llr, _, _ = sq.bp_decode(nd, columns, list(priors), s.tolist(), 1000, "minimum_sum", 0.0)
        if not np.array_equal(np.array(llr), np.array(bp.log_prob_ratios)):
            continue
        for _ in range(40):
            order = rng.permutation(n)
            dec = BpOsdDecoder(matrix(order), error_channel=list(priors[order]), osd_method="osd_cs", osd_order=7, **kw)
            again = np.zeros(n, np.uint8)
            again[order] = dec.decode(s).astype(np.uint8)
            if np.array_equal(again, ours):
                ties += 1
                break
    return same, ties, shots - same - ties


def cmd_check(args):
    ok = True
    ops = ["f"] if args.quick else OPERATORS
    shots = 64 if args.quick else 256
    for name in ops:
        for construction, basis in [(c, b) for c in CONSTRUCTIONS for b in ("x", "z")]:
            expanded = construction == "expanded"
            text = sq.bb_logical_measurement_circuit(name, basis, 1, 2, 1, 0.003, expanded)
            circuit = stim.Circuit(text)
            ours = mechanisms(stim.DetectorErrorModel(sq.dem_from_circuit(text, False)))
            theirs = mechanisms(circuit.detector_error_model(decompose_errors=False))
            worst = max((abs(ours[k] - theirs[k]) / theirs[k] for k in theirs if k in ours), default=0.0)
            noiseless = stim.Circuit(sq.bb_logical_measurement_circuit(name, basis, 1, 2, 1, 0.0, expanded))
            det, obs = noiseless.compile_detector_sampler(seed=1).sample(64, separate_observables=True)
            same, ties, unexplained = bposd_matches(text, shots, seed=len(name) * 10 + (basis == "x"))
            good = set(ours) == set(theirs) and worst < 1e-9 and not det.any() and not obs.any() and unexplained == 0
            ok &= good
            print(f"  {'ok ' if good else 'BAD'} {name} {construction} {basis}: {circuit.num_qubits} qubits, {circuit.num_detectors} detectors, "
                  f"{circuit.num_observables} observables, {len(theirs)} mechanisms (ours {len(ours)}, "
                  f"{len(set(ours) ^ set(theirs))} one-sided, worst Δp/p {worst:.1e}); noiseless all zero: "
                  f"{not det.any() and not obs.any()}; BP+OSD-CS7 equals ldpc on {same} of {shots} shots, {ties} differ by the "
                  f"order of tied posteriors (ldpc returns ours with its columns permuted), {unexplained} unexplained", flush=True)
    print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
    return 0 if ok else 1


# -- measure -----------------------------------------------------------------

def wilson(k, n, z=1.96):
    if n == 0:
        return [0.0, 1.0]
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [max(0.0, mid - half), min(1.0, mid + half)]


PRE = POST = 6


def points():
    """(key, metadata, Stim text). The expanded system throughout, and the
    minimal one for X(f, 0), the operator whose minimal system loses distance."""
    pts = []
    p = 0.003
    lengths = set()

    def measure(op, construction, basis, T):
        key = f"{op}/{construction}/{basis}/T{T}/p{p}"
        meta = dict(kind="measure", operator=op, construction=construction, basis=basis, merged=T, pre=PRE, post=POST, p=p)
        lengths.add(PRE + T + POST)
        pts.append((key, meta, lambda: sq.bb_logical_measurement_circuit(op, basis, PRE, T, POST, p, construction == "expanded")))

    for construction in CONSTRUCTIONS:
        for basis in ("x", "z"):
            for T in (2, 4, 7, 12):
                if construction == "expanded" or basis == "x" or T in (2, 7):
                    measure("f", construction, basis, T)
    measure("gh", "expanded", "x", 7)
    for basis in ("x", "z"):
        measure("f+gh", "expanded", basis, 7)
    for basis in ("x", "z"):
        for R in sorted(lengths):
            pts.append((f"memory/{basis}/R{R}/p{p}", dict(kind="memory", basis=basis, cycles=R, p=p),
                        lambda b=basis, R=R: sq.bb_memory_basis_circuit("gross", b, R, p)))
    return pts


def run(text, seed, shot_cap, time_cap, target):
    dem = sq.dem_from_circuit(text, False)
    nobs = stim.Circuit(text).num_observables
    counts = dict(any=0, first=0, rest=0, converged=0)
    shots, t0, batch, secs = 0, time.perf_counter(), 0, 0.0
    while True:
        dets, obs, _ = sq.sample_b8_batch(text, BATCH, seed * 1_000_003 + batch, 0)
        rows = np.frombuffer(obs, np.uint8).reshape(BATCH, -1)
        truth = sum(rows[:, i].astype(np.uint64) << np.uint64(8 * i) for i in range(rows.shape[1]))
        pred, conv, s = sq.decode_b8_bposd(dem, dets, BATCH, 10_000, "minimum_sum", 0.0, "osd_cs", 7, 0)
        wrong = (np.frombuffer(pred, "<u8") ^ truth) & np.uint64((1 << nobs) - 1)
        counts["any"] += int((wrong != 0).sum())
        counts["first"] += int((wrong & np.uint64(1) != 0).sum())
        counts["rest"] += int((wrong >> np.uint64(1) != 0).sum())
        counts["converged"] += int(np.frombuffer(conv, np.uint8).sum())
        secs += s
        shots += BATCH
        batch += 1
        if counts["any"] >= target or shots >= shot_cap or time.perf_counter() - t0 >= time_cap:
            break
    return shots, counts, secs


def cmd_measure(args):
    path = OUT / "logical.json"
    doc = load_json(path, dict(points={}))
    doc.update(engine_commit=engine_commit(), machine=machine(),
               note="The gauging measurement of gross-code logical X operators (src/bb_circuit.rs), Bravyi et al.'s circuit "
                    "noise, sampled by the bit-parallel sampler and decoded by BP+OSD-CS of order 7 (10,000 min-sum "
                    "iterations) on the undecomposed error model. 'first' is L0: the outcome in the X basis; 'rest' the "
                    "other observables")
    for i, (key, meta, make) in enumerate(points()):
        if key in doc["points"]:
            continue
        t = time.perf_counter()
        shots, c, secs = run(make(), 7_000_011 * (i + 1), args.shot_cap, args.time_cap, args.target)
        rec = dict(meta, shots=shots, failures=c, rate_any=c["any"] / shots, interval_any=wilson(c["any"], shots),
                   rate_first=c["first"] / shots, interval_first=wilson(c["first"], shots), decode_seconds=round(secs, 1))
        doc["points"][key] = rec
        doc["generated"] = datetime.date.today().isoformat()
        write_json(path, doc)
        print(f"{key:<24} {shots:>9,} shots  any {c['any']:>5} ({rec['rate_any']:.2e})  first {c['first']:>5}  "
              f"{time.perf_counter() - t:.0f} s", flush=True)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["auto", "distance", "check", "measure"])
    ap.add_argument("--quick", action="store_true", help="check: X(f, 0) only, fewer shots")
    ap.add_argument("--operator", choices=["code"] + OPERATORS, help="distance: one operator only")
    ap.add_argument("--construction", choices=CONSTRUCTIONS, help="distance: one construction only")
    ap.add_argument("--redo", action="store_true", help="distance: recompute entries already recorded")
    ap.add_argument("--cap", type=float, default=7200.0, help="distance: seconds per integer program")
    ap.add_argument("--shot-cap", type=int, default=200_000)
    ap.add_argument("--time-cap", type=float, default=1800.0, help="measure: seconds per point")
    ap.add_argument("--target", type=int, default=100, help="measure: failures per point")
    args = ap.parse_args()
    return {"auto": cmd_auto, "distance": cmd_distance, "check": cmd_check, "measure": cmd_measure}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())

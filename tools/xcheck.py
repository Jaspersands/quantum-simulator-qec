"""Cross-check this engine against Stim and PyMatching.

Run from the repository root, with the engine built into the virtualenv:

    VIRTUAL_ENV=$PWD/.venv CARGO_TARGET_DIR=target-py .venv/bin/maturin develop --profile python
    .venv/bin/python tools/xcheck.py              # full run; writes data/xcheck/
    .venv/bin/python tools/xcheck.py --quick      # a smoke test; writes nothing

Checks, strictest first (spec: docs/superpowers/specs/2026-09-24-stim-foundation-design.md):
  1. Detector error models are identical: every mechanism, to 1e-9 relative.
     1b. So are the matching graphs: every symptom decomposed the same way, every
         edge with the same probability, against Stim's decomposed model.
  2. Decoders agree shot for shot on Stim's samples; every disagreement is a tie.
  3. Samplers agree: per-detector firing rates, and logical error rates.
  4. Speed, reported as measured.
  5. Correlated matching agrees with PyMatching's (enable_correlations=True)
     shot for shot, and every disagreement is a tie: PyMatching's own
     first-pass edges, fed to our second pass, give its answer or tie it.
Exits non-zero if check 1, 1b, 2, 3 or 5 fails.
"""

import argparse
import collections
import datetime
import hashlib
import json
import math
import pathlib
import sys
import time

import numpy as np
import pymatching
import stim

import stabilizer_qec as sq

if not hasattr(sq, "generate_circuit"):
    sys.exit(f"{sq.__file__} is an old build without the cross-check bindings; "
             "run `maturin develop --profile python` into .venv, and run this from the repository root.")

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "xcheck"
P_MODEL = 0.003
TOL = 1e-9


def sha256(text):
    return hashlib.sha256(text.encode()).hexdigest()


def stim_generated(d):
    return stim.Circuit.generated(
        "surface_code:rotated_memory_z", distance=d, rounds=d,
        after_clifford_depolarization=P_MODEL, before_round_data_depolarization=P_MODEL,
        before_measure_flip_probability=P_MODEL, after_reset_flip_probability=P_MODEL)


def pieces_of(inst):
    """An error instruction's pieces, each a sorted tuple of ('D', k) / ('L', k)."""
    pieces = [[]]
    for t in inst.targets_copy():
        if t.is_separator():
            pieces.append([])
        elif t.is_relative_detector_id():
            pieces[-1].append(("D", t.val))
        else:
            pieces[-1].append(("L", t.val))
    return [tuple(sorted(pc)) for pc in pieces]


def xor(p, q):
    return p * (1 - q) + q * (1 - p)


def mechanisms(dem):
    """{full symptom: p}, identical symptoms merged."""
    out = {}
    for inst in dem.flattened():
        if inst.type != "error":
            continue
        full = set()
        for pc in pieces_of(inst):
            full ^= set(pc)
        key = tuple(sorted(full))
        out[key] = xor(out.get(key, 0.0), inst.args_copy()[0])
    return out


def graph(dem):
    """({edge: p}, {full symptom: {decomposition: p}}) for a decomposed model."""
    edges = collections.defaultdict(float)
    splits = collections.defaultdict(dict)
    for inst in dem.flattened():
        if inst.type != "error":
            continue
        p = inst.args_copy()[0]
        pcs = pieces_of(inst)
        full = set()
        for pc in pcs:
            edges[pc] = xor(edges[pc], p)
            full ^= set(pc)
        key = tuple(sorted(pcs))
        sym = tuple(sorted(full))
        splits[sym][key] = xor(splits[sym].get(key, 0.0), p)
    return edges, splits


def rel(a, b):
    return abs(a - b) / max(a, b) if max(a, b) > 0 else 0.0


def check_model(entry, circuit_text):
    """Checks 1 and 1b for one circuit."""
    flat = stim.Circuit(circuit_text).flattened()
    theirs = flat.detector_error_model(decompose_errors=False).flattened()
    ours = stim.DetectorErrorModel(sq.dem_from_circuit(circuit_text, False))
    a, b = mechanisms(ours), mechanisms(theirs)
    rels = [rel(a[k], b[k]) for k in a if k in b]

    ea, sa = graph(stim.DetectorErrorModel(sq.dem_from_circuit(circuit_text, True)))
    eb, sb = graph(flat.detector_error_model(decompose_errors=True))
    edge_rels = [rel(ea[k], eb[k]) for k in ea if k in eb]
    split_differ = sum(
        1 for k in set(sa) | set(sb)
        if set(sa.get(k, {})) != set(sb.get(k, {}))
        or any(rel(sa[k][s], sb[k][s]) > TOL for s in sa.get(k, {}) if s in sb.get(k, {})))

    text = str(theirs)
    entry.update(
        detectors=theirs.num_detectors, mechanisms=len(b), ours=len(a),
        missing=sum(1 for k in b if k not in a), extra=sum(1 for k in a if k not in b),
        differing=sum(1 for r in rels if r > TOL), max_rel=max(rels) if rels else 0.0,
        edges=len(eb), edges_one_sided=len(set(ea) ^ set(eb)),
        edges_max_rel=max(edge_rels) if edge_rels else 0.0, splits_differ=split_differ,
        bytes=len(text.encode()), dem_sha256=sha256(text))
    return entry, text


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (mid - half, mid + half)


def unpack(packed, shots, bits):
    rows = np.frombuffer(packed, dtype=np.uint8).reshape(shots, -1)
    return np.unpackbits(rows, axis=1, bitorder="little")[:, :bits].astype(bool)


FAILED = np.uint64(2**64 - 1)
ONE = np.uint64(1)


def check_decoding(code, d, p, shots, seed):
    """Checks 2, 3 and 4 at one point."""
    text = sq.generate_circuit(code, d, d, "sd6", p, 0.5, "z")
    circuit = stim.Circuit(text)
    dem = circuit.detector_error_model(decompose_errors=True)
    matching = pymatching.Matching.from_detector_error_model(dem)
    dets, obs = circuit.compile_detector_sampler(seed=seed).sample(shots, separate_observables=True)
    actual = obs[:, 0].astype(np.uint64)
    packed = np.packbits(dets, axis=1, bitorder="little").tobytes()

    t = time.perf_counter()
    pm = matching.decode_batch(dets)[:, 0].astype(np.uint64)
    pm_seconds = time.perf_counter() - t

    def ours_with(fn, arg):
        pred_b, w_b, errors, seconds = fn(arg, packed, shots)
        raw = np.frombuffer(pred_b, dtype="<u8")
        return raw & ONE, np.frombuffer(w_b, dtype="<f8"), raw == FAILED, errors, seconds

    ours, our_w, failed, errors, our_seconds = ours_with(sq.decode_b8, str(dem))
    own, own_w, own_failed, own_errors, _ = ours_with(sq.decode_b8_own, text)

    # Check 2: every disagreement must be a tie. The weight noise between two
    # exact matchers that discretise weights differently is measured on shots
    # where they agree, and a disagreement is a tie if it sits inside that noise.
    def ties(pred, weights, bad):
        agree = np.nonzero((pm == pred) & ~bad)[0][:500]
        noise = 0.0
        for i in agree:
            _, w = matching.decode(dets[i], return_weight=True)
            noise = max(noise, abs(w - weights[i]))
        tol = max(10 * noise, 1e-6)
        disagree = np.nonzero((pm != pred) & ~bad)[0]
        non_ties = 0
        for i in disagree:
            _, w = matching.decode(dets[i], return_weight=True)
            if abs(w - weights[i]) > tol:
                non_ties += 1
        return int(len(disagree)), non_ties, noise

    disagreements, non_ties, noise = ties(ours, our_w, failed)
    own_disagreements, own_non_ties, _ = ties(own, own_w, own_failed)

    # Check 3: our sampler on the same circuit text, decoded by our decoder.
    od_b, oo_b = sq.sample_b8(text, shots, seed + 1)
    od = unpack(od_b, shots, circuit.num_detectors)
    oo = unpack(oo_b, shots, 1)[:, 0].astype(np.uint64)
    a, b = dets.sum(axis=0).astype(float), od.sum(axis=0).astype(float)
    mask = (a + b) > 0
    chi2 = float(((a - b) ** 2 / np.where(mask, a + b, 1))[mask].sum())
    dof = int(mask.sum())
    z = (chi2 - dof) / math.sqrt(2 * dof) if dof else 0.0
    s_pred_b, _, s_errors, _ = sq.decode_b8_own(text, np.packbits(od, axis=1, bitorder="little").tobytes(), shots)
    s_raw = np.frombuffer(s_pred_b, dtype="<u8")

    return dict(
        code=code, noise="sd6", d=d, p=p, shots=shots,
        pymatching_failures=int(np.count_nonzero(pm != actual)),
        ours_failures=int(np.count_nonzero((ours != actual) | failed)),
        ours_own_dem_failures=int(np.count_nonzero((own != actual) | own_failed)),
        disagreements=disagreements, non_ties=non_ties, weight_noise=noise,
        own_disagreements=own_disagreements, own_non_ties=own_non_ties,
        decode_errors=int(errors), own_decode_errors=int(own_errors),
        pymatching_us=pm_seconds / shots * 1e6, ours_us=our_seconds / shots * 1e6,
        sampler=dict(chi2=chi2, dof=dof, z=z,
                     our_failures=int(np.count_nonzero(((s_raw & ONE) != oo) | (s_raw == FAILED))),
                     our_decode_errors=int(s_errors)))


def check_correlated(code, d, p, shots, seed):
    """Check 5 at one point: correlated matching, ours against PyMatching's."""
    text = sq.generate_circuit(code, d, d, "sd6", p, 0.5, "z")
    circuit = stim.Circuit(text)
    dem = circuit.detector_error_model(decompose_errors=True)
    matching = pymatching.Matching.from_detector_error_model(dem, enable_correlations=True)
    dets, obs = circuit.compile_detector_sampler(seed=seed).sample(shots, separate_observables=True)
    actual = obs[:, 0].astype(np.uint64)
    packed = np.packbits(dets, axis=1, bitorder="little").tobytes()

    t = time.perf_counter()
    pm = matching.decode_batch(dets, enable_correlations=True)[:, 0].astype(np.uint64)
    pm_seconds = time.perf_counter() - t
    raw_b, _, errors, our_seconds = sq.decode_b8(str(dem), packed, shots, 1, True)
    raw = np.frombuffer(raw_b, dtype="<u8")
    ours, failed = raw & ONE, raw == FAILED
    plain_b, _, _, _ = sq.decode_b8(str(dem), packed, shots, 0, False)
    plain = np.frombuffer(plain_b, dtype="<u8") & ONE

    # A disagreement is explained if PyMatching's own first-pass edges, fed to
    # our second pass, give PyMatching's answer (the two passes traced
    # different, equally short paths), or tie it in weight. The weight noise
    # between the two discretisations is measured on shots where both agree.
    dec = sq.Decoder(str(dem))

    def ours_from_their_edges(i):
        defects = np.flatnonzero(dets[i]).tolist()
        edges = [(int(u), int(v)) for u, v in matching.decode_to_edges_array(dets[i])]
        return dec.pass2(defects, edges)

    def their_weight(i):
        return matching.decode(dets[i], return_weight=True, enable_correlations=True)[1]

    noise = 0.0
    for i in np.nonzero((pm == ours) & ~failed)[0][:300]:
        noise = max(noise, abs(ours_from_their_edges(i)[1] - their_weight(i)))
    tol = max(10 * noise, 1e-6)
    disagree = np.nonzero((pm != ours) & ~failed)[0]
    path_ties = weight_ties = non_ties = 0
    for i in disagree:
        o, w = ours_from_their_edges(i)
        if np.uint64(o & 1) == pm[i]:
            path_ties += 1
        elif abs(w - their_weight(i)) <= tol:
            weight_ties += 1
        else:
            non_ties += 1

    return dict(
        code=code, noise="sd6", d=d, p=p, shots=shots,
        pymatching_failures=int(np.count_nonzero(pm != actual)),
        ours_failures=int(np.count_nonzero((ours != actual) | failed)),
        plain_failures=int(np.count_nonzero(plain != actual)),
        disagreements=int(len(disagree)), path_ties=path_ties, weight_ties=weight_ties,
        non_ties=non_ties, weight_noise=noise, decode_errors=int(errors),
        pymatching_us=pm_seconds / shots * 1e6, ours_us=our_seconds / shots * 1e6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="few shots, d <= 5, write nothing")
    ap.add_argument("--shots", type=int, default=100_000)
    args = ap.parse_args()
    shots = 2_000 if args.quick else args.shots
    ds = (3, 5) if args.quick else (3, 5, 7)
    ok = True

    circuits, files = [], {}
    print("Checks 1 and 1b: detector error models and matching graphs")
    for d in ds:
        name = f"stim-rotated-memory-z-d{d}"
        text = str(stim_generated(d))
        entry, dem_text = check_model(dict(name=name, source="stim", d=d, rounds=d, p=P_MODEL), text)
        circuits.append(entry)
        files[f"{name}.stim.txt"] = text
        files[f"{name}.dem.txt"] = dem_text
    for code in ("rotated", "xzzx"):
        for noise in ("current", "sd6"):
            for d in ds:
                name = f"{code}-{noise}-d{d}"
                text = sq.generate_circuit(code, d, d, noise, P_MODEL, 0.5, "z")
                entry, dem_text = check_model(
                    dict(name=name, source="ours", code=code, noise=noise, d=d, rounds=d, p=P_MODEL,
                         eta=0.5, basis="z", circuit_sha256=sha256(text)), text)
                circuits.append(entry)
                files[f"{name}.dem.txt"] = dem_text
    for c in circuits:
        good = c["missing"] == c["extra"] == c["differing"] == 0
        graph_good = c["edges_one_sided"] == 0 and c["splits_differ"] == 0 and c["edges_max_rel"] <= TOL
        ok &= good and graph_good
        print(f"  {'ok ' if good and graph_good else 'BAD'} {c['name']:<28} {c['detectors']:>5} detectors  "
              f"{c['ours']:>6} / {c['mechanisms']:<6} mechanisms  max rel {c['max_rel']:.1e}  |  "
              f"graph {c['edges']:>5} edges, {c['edges_one_sided']} one-sided, "
              f"{c['splits_differ']} splits differ, max rel {c['edges_max_rel']:.1e}")

    print(f"Checks 2-4: decoding, {shots:,} shots per point")
    decoding = []
    for d in ds:
        for p in (0.003, 0.006):
            r = check_decoding("rotated", d, p, shots, seed=1000 * d + int(p * 1e4))
            decoding.append(r)
            ci_pm = wilson(r["pymatching_failures"], shots)
            ci_us = wilson(r["sampler"]["our_failures"], shots)
            overlap = ci_pm[0] <= ci_us[1] and ci_us[0] <= ci_pm[1]
            good = (r["non_ties"] == 0 and r["own_non_ties"] == 0 and r["decode_errors"] == 0
                    and r["own_decode_errors"] == 0 and r["sampler"]["our_decode_errors"] == 0
                    and abs(r["sampler"]["z"]) < 4 and overlap)
            ok &= good
            print(f"  {'ok ' if good else 'BAD'} d={d} p={p:.3f}  PyMatching {r['pymatching_failures'] / shots:.4%}  "
                  f"ours {r['ours_failures'] / shots:.4%}  own-DEM {r['ours_own_dem_failures'] / shots:.4%}  "
                  f"our sampler {r['sampler']['our_failures'] / shots:.4%}  "
                  f"disagree {r['disagreements']}/{r['own_disagreements']} "
                  f"(non-ties {r['non_ties']}/{r['own_non_ties']}, weight noise {r['weight_noise']:.1e})  "
                  f"chi2 z {r['sampler']['z']:+.2f}  "
                  f"{r['pymatching_us']:.1f} us vs {r['ours_us']:.1f} us")

    print(f"Check 5: correlated matching against PyMatching's, {shots:,} shots per point")
    correlated = []
    for code in ("rotated", "xzzx"):
        for d in ds:
            for p in (0.003, 0.006):
                r = check_correlated(code, d, p, shots, seed=7000 + 1000 * d + int(p * 1e4))
                correlated.append(r)
                ci_pm = wilson(r["pymatching_failures"], shots)
                ci_us = wilson(r["ours_failures"], shots)
                overlap = ci_pm[0] <= ci_us[1] and ci_us[0] <= ci_pm[1]
                good = r["non_ties"] == 0 and r["decode_errors"] == 0 and overlap
                ok &= good
                print(f"  {'ok ' if good else 'BAD'} {code:<7} d={d} p={p:.3f}  "
                      f"PyMatching {r['pymatching_failures'] / shots:.4%}  ours {r['ours_failures'] / shots:.4%}  "
                      f"(plain {r['plain_failures'] / shots:.4%})  disagree {r['disagreements']} "
                      f"(path ties {r['path_ties']}, weight ties {r['weight_ties']}, non-ties {r['non_ties']}, "
                      f"weight noise {r['weight_noise']:.1e})  {r['pymatching_us']:.1f} us vs {r['ours_us']:.1f} us")
    gain_ok = sum(r["ours_failures"] for r in correlated) < sum(r["plain_failures"] for r in correlated)
    ok &= gain_ok
    print(f"  {'ok ' if gain_ok else 'BAD'} correlated fails less often than plain, pooled over every point: "
          f"{sum(r['ours_failures'] for r in correlated):,} against {sum(r['plain_failures'] for r in correlated):,}")

    if not args.quick:
        OUT.mkdir(parents=True, exist_ok=True)
        for name, text in files.items():
            (OUT / name).write_text(text)
        reference = dict(
            generated=datetime.date.today().isoformat(), stim=stim.__version__,
            pymatching=pymatching.__version__, command="python tools/xcheck.py",
            circuits=circuits, decoding=decoding, correlated=correlated)
        (OUT / "reference.json").write_text(json.dumps(reference, indent=1) + "\n")
        print(f"Wrote {OUT.relative_to(ROOT)}/ ({len(files)} files + reference.json)")
    print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

"""The 1.8 decoders against their authors' packages, shot for shot.

Run from the repository root, with the package built and the references installed
(`pip install ldpc chromobius relay-bp` and `pip install --pre tesseract-decoder`):

    python tools/decoder_check.py                 # every check; writes data/decoders/check.json
    python tools/decoder_check.py --quick         # fewer shots, writes nothing (CI)
    python tools/decoder_check.py --only lsd      # some sections (comma-separated)

Sections (each skipped, and said so, when its reference is not installed):
  lsd     BP+LSD against ldpc's BpLsdDecoder on the same check matrix: random matrices, a
          d = 5 surface code's model and the gross code's (2 and 6 cycles at p = 0.3%), LSD-0 with
          both BP methods, LSD-E 6 and LSD-CS 10. Every correction must equal ldpc's, except on shots where ours made a choice
          ldpc makes from an order that cannot be reproduced (equal keys in a long std::sort,
          or merging three or more clusters at once, which ldpc orders by hashed pointers);
          those are counted as ties. Also: each untied final cluster's bits, in the order
          ldpc's statistics list them (its robin-set order), must be listed in the same order.

  relay   Relay-BP against IBM's relay_bp (RelayDecoderF64) on the gross code's check matrix
          (2 and 6 cycles at p = 0.3%): explicit memory strengths, seeded random ones (its sinter
          defaults; every leg), and no memory. Every shot's correction, convergence and
          iteration count must be identical: there is no sort and no tie, and the random
          strengths are drawn with the same generator.

  color   ColorMatching against chromobius on annotated colour-code memories (d = 3 to 9, both
          bases, two noise strengths) and Chromobius's own colour repetition code: the Möbius
          matching's weight equal on every shot (a check of the whole Möbius graph), and the
          predictions equal except where they differ at that equal weight, which is a tie
          between two minimum-weight matchings.

  search  SearchDecoder against tesseract_decoder (Tesseract's defaults: 20 index orders; two
          literal orders; beam 8 with beam climbing) on surface, colour and gross-code models:
          the faults found, shot for shot, must be identical (both break ties as this
          platform's C++ library does).

Exits non-zero if any check fails.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np

import stabilizer_qec as sq

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "decoders" / "check.json"


def check_matrix(dem):
    """A model's faults as a dense check matrix and priors (Stim's flattening)."""
    import stim

    flat = stim.DetectorErrorModel(str(dem)).flattened()
    cols, priors = [], []
    for inst in flat:
        if inst.type == "error":
            dets = {}
            for t in inst.targets_copy():
                if t.is_relative_detector_id():
                    dets[t.val] = dets.get(t.val, 0) ^ 1
            cols.append(sorted(d for d, odd in dets.items() if odd))
            priors.append(inst.args_copy()[0])
    pcm = np.zeros((flat.num_detectors, len(cols)), dtype=np.uint8)
    for j, c in enumerate(cols):
        pcm[c, j] = 1
    return pcm, np.array(priors)


def lsd_cases(quick):
    rng = np.random.default_rng(5)
    for k in range(3 if quick else 10):
        pcm = (rng.random((24, 48)) < 0.12).astype(np.uint8)
        pcm[rng.integers(24, size=48), np.arange(48)] = 1
        channel = rng.uniform(0.01, 0.1, 48)
        syndromes = [(pcm @ (rng.random(48) < channel)) % 2 for _ in range(100)]
        yield f"random {k}", pcm, channel, np.array(syndromes, dtype=np.uint8), 3
    circuits = {
        "surface d=5": sq.Circuit.generated("surface_code:rotated_memory_z", distance=5, rounds=5, after_clifford_depolarization=0.005, before_measure_flip_probability=0.005),
        "gross, 2 cycles": sq.BivariateBicycleCode("gross").memory_circuit(2, 0.003),
        "gross, 6 cycles": sq.BivariateBicycleCode("gross").memory_circuit(6, 0.003),
    }
    for name, c in circuits.items():
        pcm, priors = check_matrix(c.detector_error_model())
        n = (100 if quick else 1000) if "gross" in name else (300 if quick else 3000)
        yield name, pcm, priors, c.compile_detector_sampler(seed=1).sample(n).astype(np.uint8), 30


LSD_CONFIGS = [("minimum_sum", "lsd_0", 0), ("product_sum", "lsd_0", 0), ("minimum_sum", "lsd_e", 6), ("minimum_sum", "lsd_cs", 10)]


def check_lsd(quick):
    import ldpc

    ok, rows = True, []
    for name, pcm, priors, syndromes, iters in lsd_cases(quick):
        for method, lsd, order in LSD_CONFIGS:
            kw = dict(max_iter=iters, bp_method=method, ms_scaling_factor=0.625, lsd_method=lsd, lsd_order=order)
            theirs = ldpc.BpLsdDecoder(pcm, error_channel=list(priors), **kw)
            theirs.set_do_stats(True)
            ours = sq.BpLsdDecoder(pcm, error_channel=list(priors), **kw)
            row = dict(case=name, method=method, lsd=f"{lsd} {order}", shots=len(syndromes), agree=0, tied=0, untied=0, clusters_in_order=0, clusters_out_of_order=0, ours_s=0.0, ldpc_s=0.0)
            for s in syndromes:
                t0 = time.perf_counter()
                b = theirs.decode(s)
                t1 = time.perf_counter()
                a = ours.decode(s)
                row["ldpc_s"] += t1 - t0
                row["ours_s"] += time.perf_counter() - t1
                if np.array_equal(a, b):
                    row["agree"] += 1
                elif ours.last_tied:
                    row["tied"] += 1
                else:
                    row["untied"] += 1
                if not theirs.converge:
                    stats = theirs.statistics["individual_cluster_stats"]
                    for cid, bits in ours._cluster_bits(s):
                        if list(stats[cid]["final_bits"]) == list(bits):
                            row["clusters_in_order"] += 1
                        else:
                            row["clusters_out_of_order"] += 1
            good = row["untied"] == 0 and row["clusters_out_of_order"] == 0
            ok &= good
            rows.append(row)
            print(f"  {name:16} {method:12} {lsd} {order:<3} agree {row['agree']:5}  tied {row['tied']:3}  untied {row['untied']}  "
                  f"clusters in order {row['clusters_in_order']:5} (out {row['clusters_out_of_order']})  "
                  f"{1e3 * row['ours_s'] / len(syndromes):.3f} vs {1e3 * row['ldpc_s'] / len(syndromes):.3f} ms/shot  "
                  f"{'ok' if good else 'FAIL'}", flush=True)
    return ok, rows


def check_relay(quick):
    import scipy.sparse
    import relay_bp

    ok, rows = True, []
    rng = np.random.default_rng(11)
    for cycles in (2, 6):
        c = sq.BivariateBicycleCode("gross").memory_circuit(cycles, 0.003)
        pcm, priors = check_matrix(c.detector_error_model())
        syndromes = c.compile_detector_sampler(seed=2).sample(100 if quick else 1000).astype(np.uint8)
        configs = {
            "explicit strengths": dict(gammas=rng.uniform(-0.24, 0.66, size=(10, pcm.shape[1])), legs=30, solutions=1),
            "sinter defaults, seed 0": dict(seed=0, pre_iterations=60, legs=60, iterations=60, solutions=5),
            "every leg, seed 7": dict(seed=7, legs=20, solutions=None),
            "no memory": dict(gamma0=None, seed=1, legs=10, solutions=1),
        }
        for label, kw in configs.items():
            kw = {**dict(pre_iterations=80, iterations=60, gamma0=0.1, gammas=None, alpha=None, seed=0), **kw}
            ours = sq.RelayBpDecoder(pcm, error_channel=priors, **kw)
            stop = dict(stopping_criterion="all") if kw["solutions"] is None else dict(stop_nconv=kw["solutions"])
            theirs = relay_bp.RelayDecoderF64(scipy.sparse.csr_matrix(pcm), error_priors=priors.astype(np.float64), pre_iter=kw["pre_iterations"], num_sets=kw["legs"],
                                              set_max_iter=kw["iterations"], gamma0=kw["gamma0"], explicit_gammas=kw["gammas"], alpha=kw["alpha"], seed=kw["seed"], **stop)
            row = dict(case=f"gross, {cycles} cycles", config=label, shots=len(syndromes), identical=0, different=0, ours_s=0.0, reference_s=0.0)
            for s in syndromes:
                t0 = time.perf_counter()
                r = theirs.decode_detailed(s)
                t1 = time.perf_counter()
                a = ours.decode(s)
                row["reference_s"] += t1 - t0
                row["ours_s"] += time.perf_counter() - t1
                same = np.array_equal(a, r.decoding) and ours.converge == r.success and ours.iter == r.iterations
                row["identical" if same else "different"] += 1
            ok &= row["different"] == 0
            rows.append(row)
            print(f"  gross, {cycles} cycles  {label:24} identical {row['identical']:5}  different {row['different']}  "
                  f"{1e3 * row['ours_s'] / len(syndromes):.3f} vs {1e3 * row['reference_s'] / len(syndromes):.3f} ms/shot  "
                  f"{'ok' if row['different'] == 0 else 'FAIL'}", flush=True)
    return ok, rows


COLOR_REP_CODE = """
X_ERROR(0.1) 0 1 2 3 4 5 6 7 8
M 0 1 2 3 4 5 6 7 8
DETECTOR(0, 0, 0, 0) rec[-9] rec[-8] rec[-7]
DETECTOR(1, 0, 0, 1) rec[-8] rec[-7] rec[-6]
DETECTOR(2, 0, 0, 2) rec[-7] rec[-6] rec[-5]
DETECTOR(3, 0, 0, 0) rec[-6] rec[-5] rec[-4]
DETECTOR(4, 0, 0, 1) rec[-5] rec[-4] rec[-3]
DETECTOR(5, 0, 0, 2) rec[-4] rec[-3] rec[-2]
DETECTOR(6, 0, 0, 0) rec[-3] rec[-2] rec[-1]
DETECTOR(7, 0, 0, 1) rec[-2] rec[-1]
""" + "".join(f"OBSERVABLE_INCLUDE({k}) rec[-{k + 1}]\n" for k in range(9))


def check_color(quick):
    import chromobius
    import stim

    ok, rows = True, []
    cases = [("Chromobius's colour repetition code (its test)", sq.Circuit(COLOR_REP_CODE))]
    for d in (3, 5, 7) if quick else (3, 5, 7, 9):
        for basis in ("z", "x"):
            for p in (0.001, 0.003):
                cases.append((f"colour code d={d} {basis} p={p}", sq.CssCode.color_code(d).memory_circuit(d, p, basis=basis, annotate_colors=True)))
    for name, c in cases:
        dem = stim.DetectorErrorModel(str(c.detector_error_model()))
        n = 2000 if quick else 20000
        dets, _ = stim.Circuit(str(c)).compile_detector_sampler(seed=3).sample(n, separate_observables=True, bit_packed=True)
        t0 = time.perf_counter()
        theirs, tw = chromobius.compile_decoder_for_dem(dem).predict_weighted_obs_flips_from_dets_bit_packed(dets)
        t1 = time.perf_counter()
        ours, w = sq.ColorMatching(str(dem)).decode_batch(dets, bit_packed_shots=True, bit_packed_predictions=True, return_weights=True)
        t2 = time.perf_counter()
        same = np.all(ours == theirs, axis=1)
        wsame = np.isclose(w, tw, rtol=1e-4, atol=1e-3)
        row = dict(case=name, shots=n, agree=int(same.sum()), tied=int(np.sum(~same & wsame)), untied=int(np.sum(~same & ~wsame)),
                   weights_differ=int(np.sum(~wsame)), ours_s=t2 - t1, chromobius_s=t1 - t0)
        good = row["untied"] == 0 and row["weights_differ"] == 0
        ok &= good
        rows.append(row)
        print(f"  {name:48} agree {row['agree']:6}  tied {row['tied']:4}  untied {row['untied']}  weights differ {row['weights_differ']}  "
              f"{1e6 * row['ours_s'] / n:.1f} vs {1e6 * row['chromobius_s'] / n:.1f} us/shot  {'ok' if good else 'FAIL'}", flush=True)
    return ok, rows


def check_search(quick):
    import stim
    from tesseract_decoder import tesseract

    ok, rows = True, []
    cases = [
        ("surface d=3", sq.Circuit.generated("surface_code:rotated_memory_z", distance=3, rounds=3, after_clifford_depolarization=0.003, before_measure_flip_probability=0.003)),
        ("surface d=5", sq.Circuit.generated("surface_code:rotated_memory_z", distance=5, rounds=5, after_clifford_depolarization=0.002, before_measure_flip_probability=0.002)),
        ("colour d=5", sq.CssCode.color_code(5).memory_circuit(5, 0.002)),
        ("gross, 2 cycles", sq.BivariateBicycleCode("gross").memory_circuit(2, 0.002)),
    ]
    for name, c in cases:
        dem = stim.DetectorErrorModel(str(c.detector_error_model()))
        n = (100 if "gross" in name else 300) if quick else (1000 if "gross" in name else 3000)
        dets = stim.Circuit(str(c)).compile_detector_sampler(seed=4).sample(n)
        nd = dem.num_detectors
        configs = {"defaults (20 index orders)": {}, "two literal orders": dict(det_orders=[list(range(nd)), list(range(nd))[::-1]]), "beam 8, climbing": dict(det_beam=8, beam_climbing=True)}
        for label, kw in configs.items():
            theirs = tesseract.TesseractDecoder(tesseract.TesseractConfig(dem, **kw))
            okw = {("beam" if k == "det_beam" else k): v for k, v in kw.items()}
            ours = sq.SearchDecoder(str(dem), **okw)
            row = dict(case=name, config=label, shots=n, identical=0, different=0, ours_s=0.0, tesseract_s=0.0)
            for d in dets:
                t0 = time.perf_counter()
                theirs.decode_to_errors(d)
                t1 = time.perf_counter()
                faults = ours.decode_to_faults(d)
                row["tesseract_s"] += t1 - t0
                row["ours_s"] += time.perf_counter() - t1
                row["identical" if faults == list(theirs.predicted_errors_buffer) else "different"] += 1
            ok &= row["different"] == 0
            rows.append(row)
            print(f"  {name:16} {label:28} identical {row['identical']:5}  different {row['different']}  "
                  f"{1e3 * row['ours_s'] / n:.3f} vs {1e3 * row['tesseract_s'] / n:.3f} ms/shot  {'ok' if row['different'] == 0 else 'FAIL'}", flush=True)
    return ok, rows


SECTIONS = {
    "lsd": ("BP+LSD against ldpc", "ldpc", check_lsd),
    "relay": ("Relay-BP against IBM's relay_bp", "relay-bp", check_relay),
    "color": ("Colour-code matching against Chromobius", "chromobius", check_color),
    "search": ("The search decoder against Tesseract", "tesseract-decoder", check_search),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    from importlib.metadata import PackageNotFoundError, version

    only = [x for x in args.only.split(",") if x]
    ok, results = True, {}
    for key, (title, package, run) in SECTIONS.items():
        if only and key not in only:
            continue
        try:
            print(f"{title} ({package} {version(package)})")
        except PackageNotFoundError:
            print(f"{title}: skipped, {package} is not installed")
            continue
        good, rows = run(args.quick)
        ok &= good
        results[key] = dict(reference=f"{package} {version(package)}", rows=rows)
    if not args.quick and not only:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(dict(version=sq.__version__, sections=results), indent=1) + "\n")
    print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

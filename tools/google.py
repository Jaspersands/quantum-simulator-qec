"""Google's surface-code memory data, decoded by this engine.

Run from the repository root, after building the engine into .venv:

    VIRTUAL_ENV=$PWD/.venv CARGO_TARGET_DIR=target-py .venv/bin/maturin develop --profile python
    .venv/bin/python tools/google.py check     # parse, m2d and model checks -> data/google-results/checks.json
    .venv/bin/python tools/google.py run       # every prior, both matchers -> data/google-results/{willow,sycamore}.json
    .venv/bin/python tools/google.py summary   # every failed check, decode error and non-optimal disagreement
    .venv/bin/python tools/google.py extract   # the site's live extract -> data/willow-extract/

The datasets are read straight from the zips in data/google/ (see fetch.sh
there), both CC BY 4.0, by Google Quantum AI: Zenodo 13273331 (Willow, "Quantum
error correction below the surface code threshold") and Zenodo 6804040
(Sycamore, "Suppressing quantum errors by scaling a surface code logical
qubit"). Spec: docs/superpowers/specs/2026-09-25-google-data-design.md, Part 2.
"""

import argparse
import dataclasses
import datetime
import gzip
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import time
import zipfile

import numpy as np
import pymatching
import stim

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import stabilizer_qec as sq  # noqa: E402
from xcheck import check_model, graph, rel  # noqa: E402

ZIPS = {
    "willow": ROOT / "data" / "google" / "google_105Q_surface_code_d3_d5_d7.zip",
    "sycamore": ROOT / "data" / "google" / "google_qec3v5_experiment_data.zip",
}
OUT = ROOT / "data" / "google-results"
WILLOW_ROOT = "google_105Q_surface_code_d3_d5_d7/"
PATHWAYS = [
    "correlated_matching_decoder_with_si1000_prior",
    "correlated_matching_decoder_with_rl_optimized_prior",
    "harmony_decoder_with_si1000_prior",
    "harmony_decoder_with_rl_optimized_prior",
    "libra_decoder_with_rl_optimized_prior",
]
SYCAMORE_DECODERS = ["pymatching", "correlated_matching", "belief_matching", "tensor_network_contraction"]
FILES = {
    "willow": dict(ideal="circuit_ideal.stim", noisy="circuit_noisy_si1000.stim", meas="measurements.b8",
                   sweep="sweep_bits.b8", dets="detection_events.b8", obs="obs_flips_actual.b8"),
    "sycamore": dict(ideal="circuit_ideal.stim", noisy="circuit_noisy.stim", meas="measurements.b8",
                     sweep="sweep.b8", dets="detection_events.b8", obs="obs_flips_actual.01"),
}
FAILED = np.uint64(2**64 - 1)
POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint16)


@dataclasses.dataclass
class Exp:
    dataset: str
    name: str
    patch: str
    basis: str
    d: int
    rounds: int
    shots: int
    prefix: str


_zips = {}


def zf(dataset):
    if dataset not in _zips:
        _zips[dataset] = zipfile.ZipFile(ZIPS[dataset])
    return _zips[dataset]


def experiments(dataset):
    z = zf(dataset)
    out = []
    if dataset == "willow":
        for f in z.namelist():
            if f.startswith(WILLOW_ROOT) and f.endswith("/metadata.json"):
                meta = json.loads(z.read(f))
                _, patch, basis, rdir, _ = f.split("/")
                out.append(Exp("willow", f"willow/{patch}/{basis}/{rdir}", patch, basis, meta["distance"],
                               meta["rounds"], meta["shots"], f[: -len("metadata.json")]))
    else:
        for f in z.namelist():
            if f.startswith("surface_code_") and f.endswith("/properties.yml"):
                props = dict(line.split(": ", 1) for line in z.read(f).decode().splitlines() if ": " in line)
                top = f.split("/")[0]
                out.append(Exp("sycamore", f"sycamore/{top}", top[top.index("center_"):], props["basis"].strip(),
                               int(props["distance"]), int(props["rounds"]), int(props["shots"]), top + "/"))
    return sorted(out, key=lambda e: (e.d, e.patch, e.basis, e.rounds))


def path(e, key):
    if key.startswith("pred:"):
        name = key[5:]
        if e.dataset == "willow":
            return f"{e.prefix}decoding_results/{name}/obs_flips_predicted.b8"
        return f"{e.prefix}obs_flips_predicted_by_{name}.01"
    if key.startswith("dem:"):
        name = key[4:]
        if e.dataset == "willow":
            return f"{e.prefix}decoding_results/{name}/error_model.dem"
        return f"{e.prefix}{name}.dem"
    return e.prefix + FILES[e.dataset][key]


_names = {}


def exists(e, key):
    """Not every pathway was run on every Willow experiment: Libra is missing from 56."""
    if e.dataset not in _names:
        _names[e.dataset] = set(zf(e.dataset).namelist())
    return path(e, key) in _names[e.dataset]


def read(e, key):
    return zf(e.dataset).read(path(e, key))


def text(e, key):
    return read(e, key).decode()


def bits(e, key):
    """Observable 0 per shot, as 0/1, from Stim's b8 or 01 format."""
    raw = np.frombuffer(read(e, key), np.uint8)
    out = raw[raw != ord("\n")] - ord("0") if path(e, key).endswith(".01") else raw & 1
    assert len(out) == e.shots, (e.name, key, len(out))
    return out.astype(np.uint8)


def google_prior(e):
    """The model Google decoded with that is closest to ours: SI1000 for Willow, the circuit's for Sycamore."""
    if e.dataset == "willow":
        return "dem:correlated_matching_decoder_with_si1000_prior"
    return "dem:circuit_detector_error_model"


def select(args):
    sets = ["willow", "sycamore"] if args.dataset == "all" else [args.dataset]
    exps = [e for s in sets for e in experiments(s)]
    return exps[: args.limit] if args.limit else exps


def write_json(p, obj):
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=1) + "\n")
    tmp.replace(p)


def load_json(p, default):
    return json.loads(p.read_text()) if p.exists() else default


def engine_commit():
    return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()


# -- check ------------------------------------------------------------------

def check_one(e):
    ideal, noisy = text(e, "ideal"), text(e, "noisy")
    rec = dict(d=e.d, patch=e.patch, basis=e.basis, rounds=e.rounds, shots=e.shots)
    rec["parse_round_trips"] = all(stim.Circuit(sq.circuit_to_stim(t)) == stim.Circuit(t) for t in (ideal, noisy))

    circuit = stim.Circuit(ideal)
    ours_d, ours_o = sq.m2d_b8(ideal, read(e, "meas"), read(e, "sweep"), e.shots)
    theirs = read(e, "dets")
    if ours_d == theirs:
        rec["m2d_detector_shots_differ"] = 0
    elif len(ours_d) != len(theirs):
        rec["m2d_detector_shots_differ"] = e.shots
    else:
        stride = (circuit.num_detectors + 7) // 8
        a = np.frombuffer(ours_d, np.uint8).reshape(e.shots, stride)
        b = np.frombuffer(theirs, np.uint8).reshape(e.shots, stride)
        rec["m2d_detector_shots_differ"] = int((a != b).any(axis=1).sum())
    rec["m2d_observable_shots_differ"] = int(((np.frombuffer(ours_o, np.uint8) & 1) != bits(e, "obs")).sum())

    entry, _ = check_model({}, noisy)
    rec["model_vs_stim"] = {k: entry[k] for k in (
        "detectors", "mechanisms", "ours", "missing", "extra", "differing", "max_rel",
        "edges", "edges_one_sided", "splits_differ", "edges_max_rel")}

    ea, _ = graph(stim.DetectorErrorModel(sq.dem_from_circuit(noisy, True)))
    eb, _ = graph(stim.DetectorErrorModel(text(e, google_prior(e))))
    common = set(ea) & set(eb)
    rels = sorted(rel(ea[k], eb[k]) for k in common)
    rec["model_vs_google"] = dict(
        prior=google_prior(e)[4:], edges_ours=len(ea), edges_theirs=len(eb),
        only_ours=len(set(ea) - set(eb)), only_theirs=len(set(eb) - set(ea)),
        median_rel=rels[len(rels) // 2] if rels else 0.0, max_rel=rels[-1] if rels else 0.0)
    return rec


def model_matches_stim(m):
    return m["missing"] == m["extra"] == m["differing"] == 0 and m["edges_one_sided"] == 0 and m["splits_differ"] == 0


def cmd_check(args):
    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / "checks.json"
    doc = load_json(out_path, dict(experiments={}))
    doc.update(generated=datetime.date.today().isoformat(), stim=stim.__version__, engine_commit=engine_commit())
    for e in select(args):
        if e.name in doc["experiments"] and not args.redo:
            continue
        t = time.perf_counter()
        rec = check_one(e)
        doc["experiments"][e.name] = rec
        write_json(out_path, doc)
        mv, mg = rec["model_vs_stim"], rec["model_vs_google"]
        good = rec["parse_round_trips"] and rec["m2d_detector_shots_differ"] == 0 and rec["m2d_observable_shots_differ"] == 0
        print(f"{'ok ' if good else 'BAD'} {e.name:<58} parse {rec['parse_round_trips']!s:<5} "
              f"m2d {rec['m2d_detector_shots_differ']}/{rec['m2d_observable_shots_differ']} shots differ  "
              f"model vs Stim {'same' if model_matches_stim(mv) else 'DIFFERS'} (max rel {mv['max_rel']:.1e})  "
              f"vs Google's {mg['only_ours']}/{mg['only_theirs']} edges one-sided, median rel {mg['median_rel']:.1e}  "
              f"{time.perf_counter() - t:.1f} s", flush=True)
    recs = doc["experiments"].values()
    print(f"{len(recs)} experiments: {sum(r['parse_round_trips'] for r in recs)} round-trip, "
          f"{sum(r['m2d_detector_shots_differ'] == 0 and r['m2d_observable_shots_differ'] == 0 for r in recs)} "
          f"convert bit for bit, {sum(model_matches_stim(r['model_vs_stim']) for r in recs)} models identical to Stim's")


# -- run --------------------------------------------------------------------

def outcome(raw_bytes, truth):
    raw = np.frombuffer(raw_bytes, "<u8")
    failed = raw == FAILED
    pred = (raw & np.uint64(1)).astype(np.uint8)
    return pred, failed, int(((pred != truth) | failed).sum())


def decode(kind, src, dets, shots, correlated):
    """(raw predictions, weights, errors, seconds) for one prior."""
    if kind == "dem":
        return sq.decode_b8(src, dets, shots, 0, correlated)
    if kind == "circuit":
        return sq.decode_b8_own(src, dets, shots, 0, correlated)
    # Cross-fitted, as Google used the pij models: even shots decoded by the
    # model fitted on odd shots, odd shots by the one fitted on even shots.
    even_for_odd, odd_for_even = src
    stride = len(dets) // shots
    rows = np.frombuffer(dets, np.uint8).reshape(shots, stride)
    ev, od = rows[0::2], rows[1::2]
    pe, we, xe, se = sq.decode_b8(odd_for_even, ev.tobytes(), len(ev), 0, correlated)
    po, wo, xo, so = sq.decode_b8(even_for_odd, od.tobytes(), len(od), 0, correlated)
    pred = np.empty(shots, "<u8")
    pred[0::2], pred[1::2] = np.frombuffer(pe, "<u8"), np.frombuffer(po, "<u8")
    wts = np.empty(shots, "<f8")
    wts[0::2], wts[1::2] = np.frombuffer(we, "<f8"), np.frombuffer(wo, "<f8")
    return pred.tobytes(), wts.tobytes(), xe + xo, se + so


def priors(e):
    if e.dataset == "willow":
        return {
            "si1000": ("dem", text(e, "dem:correlated_matching_decoder_with_si1000_prior")),
            "rl": ("dem", text(e, "dem:correlated_matching_decoder_with_rl_optimized_prior")),
            "ours": ("circuit", text(e, "noisy")),
        }
    return {
        "circuit": ("dem", text(e, "dem:circuit_detector_error_model")),
        "ours": ("circuit", text(e, "noisy")),
        "pij": ("pij", (text(e, "dem:pij_from_even_for_odd"), text(e, "dem:pij_from_odd_for_even"))),
    }


def pymatching_check(e, dem_text, dets, nd, ours_pred, ours_w):
    """Where our plain matcher disagrees with Google's recorded PyMatching, is
    ours optimal? Its weight must equal PyMatching 2.4's optimum, to within the
    discretisation noise measured on shots where the two agree."""
    recorded = bits(e, "pred:pymatching")
    matching = pymatching.Matching.from_detector_error_model(stim.DetectorErrorModel(dem_text))
    rows = np.frombuffer(dets, np.uint8).reshape(e.shots, -1)

    def optimum(i):
        shot = np.unpackbits(rows[i], bitorder="little")[:nd]
        return matching.decode(shot, return_weight=True)[1]

    noise = 0.0
    for i in np.nonzero(ours_pred == recorded)[0][:300]:
        noise = max(noise, abs(ours_w[i] - optimum(i)))
    tol = max(10 * noise, 1e-6)
    disagree = np.nonzero(ours_pred != recorded)[0]
    optimal = sum(1 for i in disagree if abs(ours_w[i] - optimum(i)) <= tol)
    return dict(disagree=int(len(disagree)), ours_optimal=int(optimal),
                ours_not_optimal=int(len(disagree) - optimal), weight_noise=noise)


def run_one(e):
    nd = stim.Circuit(text(e, "ideal")).num_detectors
    dets = read(e, "dets")
    truth = bits(e, "obs")
    rows = np.frombuffer(dets, np.uint8).reshape(e.shots, -1)
    rec = dict(d=e.d, patch=e.patch, basis=e.basis, rounds=e.rounds, shots=e.shots, detectors=nd,
               mean_defects=float(POPCOUNT[rows].sum(axis=1).mean()), results={})
    for prior, (kind, src) in priors(e).items():
        for correlated in (False, True):
            raw, wts, errors, seconds = decode(kind, src, dets, e.shots, correlated)
            pred, _, failures = outcome(raw, truth)
            mode = "correlated" if correlated else "plain"
            rec["results"][f"ours/{prior}/{mode}"] = dict(failures=failures, errors=int(errors), seconds=seconds)
            if e.dataset == "sycamore" and prior == "circuit" and not correlated:
                rec["pymatching_check"] = pymatching_check(e, src, dets, nd, pred, np.frombuffer(wts, "<f8"))
    for name in (PATHWAYS if e.dataset == "willow" else SYCAMORE_DECODERS):
        if exists(e, f"pred:{name}"):
            rec["results"][f"google/{name}"] = dict(failures=int((bits(e, f"pred:{name}") != truth).sum()))
    return rec


def cmd_run(args):
    OUT.mkdir(parents=True, exist_ok=True)
    docs = {}
    for e in select(args):
        out_path = OUT / f"{e.dataset}.json"
        if e.dataset not in docs:
            docs[e.dataset] = load_json(out_path, dict(experiments={}))
            docs[e.dataset].update(engine_commit=engine_commit(), machine=dict(cpus=os.cpu_count()),
                                   stim=stim.__version__, pymatching=pymatching.__version__)
        doc = docs[e.dataset]
        if e.name in doc["experiments"] and not args.redo:
            continue
        t = time.perf_counter()
        rec = run_one(e)
        rec["wall_seconds"] = time.perf_counter() - t
        doc["experiments"][e.name] = rec
        doc["generated"] = datetime.date.today().isoformat()
        write_json(out_path, doc)
        r = rec["results"]
        ours = "  ".join(f"{k[5:]} {v['failures'] / e.shots:.4f}" for k, v in r.items() if k.startswith("ours/"))
        google = "  ".join(f"{k[7:18]} {v['failures'] / e.shots:.4f}" for k, v in r.items() if k.startswith("google/"))
        pm = rec.get("pymatching_check")
        pm_text = f"  PyMatching: {pm['disagree']} differ, {pm['ours_not_optimal']} not optimal" if pm else ""
        print(f"{e.name:<58} defects {rec['mean_defects']:7.1f}  {ours}  |  {google}{pm_text}  "
              f"{rec['wall_seconds']:.1f} s", flush=True)


# -- summary ----------------------------------------------------------------

def cmd_summary(args):
    problems = 0
    checks = load_json(OUT / "checks.json", dict(experiments={}))["experiments"]
    for name, r in checks.items():
        if not r["parse_round_trips"] or r["m2d_detector_shots_differ"] or r["m2d_observable_shots_differ"]:
            problems += 1
            print(f"CHECK {name}: round-trip {r['parse_round_trips']}, m2d differs on "
                  f"{r['m2d_detector_shots_differ']} / {r['m2d_observable_shots_differ']} shots")
        if not model_matches_stim(r["model_vs_stim"]):
            print(f"MODEL {name}: ours differs from Stim's: {r['model_vs_stim']}")
    for dataset in ("willow", "sycamore"):
        doc = load_json(OUT / f"{dataset}.json", dict(experiments={}))
        shots = seconds = 0
        for name, r in doc["experiments"].items():
            shots += r["shots"]
            for k, v in r["results"].items():
                seconds += v.get("seconds", 0.0)
                if v.get("errors"):
                    problems += 1
                    print(f"DECODE {name} {k}: {v['errors']} shots not decoded")
            pm = r.get("pymatching_check")
            if pm and pm["ours_not_optimal"]:
                problems += 1
                print(f"OPTIMAL {name}: {pm['ours_not_optimal']} of {pm['disagree']} disagreements not optimal")
        print(f"{dataset}: {len(doc['experiments'])} experiments, {shots:,} shots, "
              f"{seconds / 60:.1f} min of decoding over every configuration")
    print(f"{len(checks)} experiments checked; {problems} problems")
    return 1 if problems else 0


# -- extract ----------------------------------------------------------------

EXTRACT = ROOT / "data" / "willow-extract"
# d = 7's one patch, and at d = 3 and 5 the patches centred nearest it
# (every d = 5 patch is two sites away; the first is taken). Z basis, 30 rounds.
EXTRACT_PATCHES = {3: "d3_at_q6_7", 5: "d5_at_q4_7", 7: "d7_at_q6_7"}
EXTRACT_ROUNDS = 30
EXTRACT_SHOTS = 2000
CREDIT = ("Google Quantum AI, data for \"Quantum error correction below the surface code threshold\", "
          "Zenodo record 13273331, CC BY 4.0. The first 2,000 shots of each experiment, unmodified; "
          "text files gzipped.")


def cmd_extract(args):
    EXTRACT.mkdir(parents=True, exist_ok=True)
    by_name = {e.name: e for e in experiments("willow")}
    n = EXTRACT_SHOTS
    manifest = dict(credit=CREDIT, licence="CC BY 4.0", zenodo="https://zenodo.org/records/13273331",
                    zip=ZIPS["willow"].name, shots=n, experiments=[])
    for d, patch in EXTRACT_PATCHES.items():
        e = by_name[f"willow/{patch}/Z/r{EXTRACT_ROUNDS}"]
        c = stim.Circuit(text(e, "ideal"))
        out = EXTRACT / f"d{d}"
        out.mkdir(exist_ok=True)
        for old in out.iterdir():
            old.unlink()

        def put(name, data):
            (out / name).write_bytes(data)

        def put_gz(name, data):
            put(name + ".gz", gzip.compress(data, 9, mtime=0))

        put_gz("circuit_ideal.stim", read(e, "ideal"))
        put_gz("circuit_noisy_si1000.stim", read(e, "noisy"))
        put_gz("error_model_si1000.dem", read(e, "dem:correlated_matching_decoder_with_si1000_prior"))
        rows = lambda bits_per_shot: (bits_per_shot + 7) // 8 * n  # noqa: E731
        put("measurements.b8", read(e, "meas")[: rows(c.num_measurements)])
        put("sweep_bits.b8", read(e, "sweep")[: rows(c.num_sweep_bits)])
        put("obs_flips_actual.b8", read(e, "obs")[: rows(c.num_observables)])
        pathways = [p for p in PATHWAYS if exists(e, f"pred:{p}")]
        for pw in pathways:
            put(f"pred_{pw}.b8", read(e, f"pred:{pw}")[: rows(c.num_observables)])
        manifest["experiments"].append(dict(
            d=d, patch=patch, basis="Z", rounds=e.rounds, dir=f"d{d}", shots=n,
            measurements=c.num_measurements, detectors=c.num_detectors, sweep_bits=c.num_sweep_bits,
            detection_events_sha256=hashlib.sha256(read(e, "dets")[: rows(c.num_detectors)]).hexdigest(),
            pathways=pathways, source=e.prefix))
    write_json(EXTRACT / "manifest.json", manifest)
    size = sum(f.stat().st_size for f in EXTRACT.rglob("*") if f.is_file())
    print(f"wrote {EXTRACT.relative_to(ROOT)}/: {len(manifest['experiments'])} experiments, {size / 1e6:.2f} MB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["check", "run", "summary", "extract"])
    ap.add_argument("--dataset", choices=["willow", "sycamore", "all"], default="all")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--redo", action="store_true")
    args = ap.parse_args()
    if args.command == "check":
        return cmd_check(args)
    if args.command == "run":
        return cmd_run(args)
    if args.command == "summary":
        return cmd_summary(args)
    return cmd_extract(args)


if __name__ == "__main__":
    sys.exit(main() or 0)

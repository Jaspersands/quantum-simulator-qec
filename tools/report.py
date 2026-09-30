"""Build the technical report: report/report.md -> report/report.html and report/report.pdf.

Run from the repository root, with Node, pandoc, matplotlib and Google Chrome:

    python3 tools/report.py            # everything
    python3 tools/report.py --no-pdf   # skip the PDF

Every number in the report comes from the committed data, never typed by hand:

- ``{{name}}`` is a value computed here from a file in ``data/`` (or from the
  fits ``tools/lambda.mjs`` makes of them);
- ``{{table:name}}`` is a whole table built the same way;
- ``{{readme:some words}}`` quotes a phrase from README.md, which must contain
  it word for word (for the handful of figures only the README records, such
  as timings of the page in a browser).

The build fails on a placeholder it cannot fill, a value that is not finite,
or a README phrase that is not in the README. Figures are drawn here from the
same data into report/figures/.
"""

import argparse
import datetime
import json
import math
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
REPORT = ROOT / "report"
BUILD = REPORT / "build"
FIGS = REPORT / "figures"
DATA = ROOT / "data"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

PALETTE = dict(d3="#2f5fd0", d5="#c0392b", d7="#2c7a4b", purple="#7b4bb5", amber="#a8620b",
               ink="#1c1d1f", ink2="#4a4c50", ink3="#6b6d71", rule="#ded9cd")


def load(path):
    return json.loads((ROOT / path).read_text())


# -- formatting --------------------------------------------------------------

def pct(x, digits=3):
    return f"{x * 100:.{digits}f}%"


def pct_range(iv, digits=3):
    return f"{iv[0] * 100:.{digits}f}–{iv[1] * 100:.{digits}f}%"


def num(x, digits=0):
    return f"{x:,.{digits}f}"


def us(x):
    return f"{x:.0f} µs" if x >= 20 else f"{x:.1f} µs" if x >= 2 else f"{x:.2f} µs"


def lam(fit):
    return f"{fit['lambda']:.2f} [{fit['interval'][0]:.2f}, {fit['interval'][1]:.2f}]"


def table(header, rows, align=None):
    """A pipe table; `align` is a string of l/r per column."""
    align = align or "l" + "r" * (len(header) - 1)
    rule = ["---:" if a == "r" else ":---" for a in align]
    out = ["| " + " | ".join(header) + " |", "| " + " | ".join(rule) + " |"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


# -- the fits ----------------------------------------------------------------

def fits():
    out = BUILD / "fits.json"
    subprocess.run(["node", "tools/lambda.mjs", "400", "--json", str(out), "data/realtime/willow-windows.json",
                    "data/belief/willow.json", "data/belief/sycamore.json"],
                   cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    return json.loads(out.read_text())


# -- values and tables -------------------------------------------------------

def sweep_mean(name):
    line = (DATA / "sweeps" / name).read_text().strip().splitlines()[-1]
    m = re.search(r"p_th ([\d.]+)% ± ([\d.]+)\s+nu ([\d.]+).*uncorrected ([\d.]+)%", line)
    runs = len(re.findall(r"^sweep \d+:", (DATA / "sweeps" / name).read_text(), re.M))
    return dict(p=float(m.group(1)), err=float(m.group(2)), nu=float(m.group(3)), uncorrected=float(m.group(4)), runs=runs)


def build_values(F):
    v, t = {}, {}
    ref = load("data/xcheck/reference.json")
    v["date"] = datetime.date.today().strftime("%-d %B %Y")
    v["commit"] = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    cargo = (ROOT / "Cargo.toml").read_text()
    v["version"] = re.search(r'^version = "([^"]+)"', cargo, re.M).group(1)
    v["stim"] = ref["stim"]
    v["pymatching"] = ref["pymatching"]

    # Thresholds.
    rot, xzzx = sweep_mean("sd6-rotated.txt"), sweep_mean("sd6-xzzx.txt")
    v["pth.rotated"] = f"{rot['p']:.2f}% ± {rot['err']:.2f}"
    v["pth.xzzx"] = f"{xzzx['p']:.2f}% ± {xzzx['err']:.2f}"
    v["pth.runs"] = str(rot["runs"])
    t["thresholds"] = table(
        ["code", "p_th (mean ± spread)", "ν", "uncorrected crossing", "sweeps"],
        [["rotated", f"{rot['p']:.2f}% ± {rot['err']:.2f}", f"{rot['nu']:.2f}", f"{rot['uncorrected']:.2f}%", rot["runs"]],
         ["XZZX", f"{xzzx['p']:.2f}% ± {xzzx['err']:.2f}", f"{xzzx['nu']:.2f}", f"{xzzx['uncorrected']:.2f}%", xzzx["runs"]]])

    # Stim: the error models.
    circuits = ref["circuits"]
    v["xc.circuits"] = str(len(circuits))
    v["xc.max_rel"] = f"{max(max(c['max_rel'], c['edges_max_rel']) for c in circuits):.1e}"
    v["xc.all_identical"] = "all" if all(c["missing"] == c["extra"] == c["differing"] == c["edges_one_sided"] == c["splits_differ"] == 0 for c in circuits) else "NOT all"
    fam = {}
    for c in circuits:
        key = ("Stim's rotated_memory_z" if c["source"] == "stim"
               else f"{'rotated' if c.get('code') == 'rotated' else 'XZZX'}, {'SD6' if c.get('noise') == 'sd6' else 'engine model'}")
        fam.setdefault(key, []).append(c)
    rows = []
    for key, cs in fam.items():
        cs.sort(key=lambda c: c["d"])
        rows.append([key, " / ".join(num(c["detectors"]) for c in cs), " / ".join(num(c["mechanisms"]) for c in cs),
                     " / ".join(num(c["edges"]) for c in cs), f"{max(max(c['max_rel'], c['edges_max_rel']) for c in cs):.1e}",
                     "identical" if all(c["missing"] == c["extra"] == c["differing"] == c["edges_one_sided"] == c["splits_differ"] == 0 for c in cs) else "DIFFERENT"])
    t["models"] = table(["circuit", "detectors, d = 3 / 5 / 7", "mechanisms", "graph edges", "largest Δp/p", "vs Stim"], rows, "lrrrrl")

    # PyMatching: plain decoding.
    dec = ref["decoding"]
    v["xc.shots"] = num(sum(r["shots"] for r in dec))
    v["xc.plain_disagree"] = str(sum(r["disagreements"] for r in dec))
    v["xc.plain_non_ties"] = str(sum(r["non_ties"] + r["own_non_ties"] for r in dec))
    ratios = [r["ours_us"] / r["pymatching_us"] for r in dec]
    v["xc.plain_ratio"] = f"{min(ratios):.2f} to {max(ratios):.2f}"
    t["decoding"] = table(
        ["d", "p", "PyMatching", "ours", "ours, own model", "disagreements", "non-ties", "PyMatching µs", "ours µs"],
        [[r["d"], pct(r["p"], 1), pct(r["pymatching_failures"] / r["shots"]), pct(r["ours_failures"] / r["shots"]),
          pct(r["ours_own_dem_failures"] / r["shots"]), r["disagreements"], r["non_ties"], f"{r['pymatching_us']:.1f}", f"{r['ours_us']:.1f}"] for r in dec])

    # Correlated matching.
    cor = ref["correlated"]
    v["xc.corr_shots"] = num(sum(r["shots"] for r in cor))
    v["xc.corr_disagree"] = num(sum(r["disagreements"] for r in cor))
    v["xc.corr_path_ties"] = num(sum(r["path_ties"] for r in cor))
    v["xc.corr_weight_ties"] = num(sum(r["weight_ties"] for r in cor))
    v["xc.corr_non_ties"] = num(sum(r["non_ties"] for r in cor))
    corr_f, plain_f = sum(r["ours_failures"] for r in cor), sum(r["plain_failures"] for r in cor)
    v["xc.corr_failures"], v["xc.plain_failures"] = num(corr_f), num(plain_f)
    v["xc.corr_gain"] = f"{(1 - corr_f / plain_f) * 100:.0f}%"
    cr = [r["ours_us"] / r["pymatching_us"] for r in cor]
    v["xc.corr_ratio"] = f"{min(cr):.2f} to {max(cr):.2f}"
    t["correlated"] = table(
        ["code", "d", "p", "PyMatching", "ours", "ours, plain", "disagreements", "path ties", "weight ties", "non-ties"],
        [["XZZX" if r["code"] == "xzzx" else "rotated", r["d"], pct(r["p"], 1), pct(r["pymatching_failures"] / r["shots"]),
          pct(r["ours_failures"] / r["shots"]), pct(r["plain_failures"] / r["shots"]), r["disagreements"], r["path_ties"],
          r["weight_ties"], r["non_ties"]] for r in cor], "lrrrrrrrrr")

    # Sampling speed.
    t["sampling"] = table(
        ["d", "p", "Stim", "frame sampler", "batch sampler", "Stim ÷ batch"],
        [[r["d"], pct(r["p"], 1), us(r["sample_us"]["stim"]), us(r["sample_us"]["frame"]), us(r["sample_us"]["batch"]),
          f"{r['sample_us']['stim'] / r['sample_us']['batch']:.1f}×"] for r in dec])
    d7 = next(r for r in dec if r["d"] == 7 and r["p"] == 0.003)
    v["speed.batch_d7"] = us(d7["sample_us"]["batch"])
    v["speed.stim_d7"] = us(d7["sample_us"]["stim"])
    fr = [r["sample_us"]["frame"] / r["sample_us"]["batch"] for r in dec]
    v["speed.batch_vs_frame"] = f"{min(fr):.0f} to {max(fr):.0f}"

    # Google's data.
    s = load("data/google-results/summary.json")
    v["g.experiments"] = num(s["experiments"])
    v["g.shots"] = f"{s['shots'] / 1e6:.1f} million"
    v["g.m2d_exact"] = num(s["m2d_exact"])
    v["g.models_same"] = num(s["models_same_as_stim"])
    v["g.syc_disagree"] = num(s["sycamore_pymatching"]["disagree"])
    v["g.syc_not_optimal"] = num(s["sycamore_pymatching"]["not_optimal"])
    W, S = F["willow"], F["sycamore"]
    for key, name in [("ours/si1000/plain", "plain"), ("ours/si1000/correlated", "corr"), ("ours/ours/correlated", "own"),
                      ("google/correlated_matching_decoder_with_si1000_prior", "gcorr"),
                      ("google/libra_decoder_with_rl_optimized_prior", "libra"),
                      ("google/harmony_decoder_with_rl_optimized_prior", "harmony")]:
        f = W[key]
        v[f"w.{name}.lambda"] = lam(f)
        v[f"w.{name}.l"] = f"{f['lambda']:.2f}"
        for d in ("3", "5", "7"):
            v[f"w.{name}.e{d}"] = pct(f["eps"][d])
    willow_rows = [
        ("ours/si1000/plain", "ours, plain", "SI1000"), ("ours/si1000/correlated", "ours, correlated", "SI1000"),
        ("ours/rl/correlated", "ours, correlated", "RL-optimised"), ("ours/ours/correlated", "ours, correlated", "ours, from the circuit"),
        ("google/correlated_matching_decoder_with_si1000_prior", "Google, correlated matching", "SI1000"),
        ("google/correlated_matching_decoder_with_rl_optimized_prior", "Google, correlated matching", "RL-optimised"),
        ("google/harmony_decoder_with_rl_optimized_prior", "Google, Harmony", "RL-optimised"),
        ("google/libra_decoder_with_rl_optimized_prior", "Google, Libra", "RL-optimised")]
    t["willow"] = table(["decoder", "prior", "ε, d = 3", "ε, d = 5", "ε, d = 7", "Λ [95%]"],
                        [[n, p] + [pct(W[k]["eps"][d]) for d in ("3", "5", "7")] + [lam(W[k])] for k, n, p in willow_rows], "llrrrr")
    syc_rows = [("ours/circuit/plain", "ours, plain", "circuit"), ("ours/circuit/correlated", "ours, correlated", "circuit"),
                ("ours/pij/correlated", "ours, correlated", "pij, cross-fitted"), ("google/pymatching", "Google, PyMatching", "circuit"),
                ("google/correlated_matching", "Google, correlated matching", "circuit"),
                ("google/belief_matching", "Google, belief matching", "pij"),
                ("google/tensor_network_contraction", "Google, tensor network", "pij")]
    t["sycamore"] = table(["decoder", "prior", "ε, d = 3", "ε, d = 5", "Λ [95%]"],
                          [[n, p] + [pct(S[k]["eps"][d]) for d in ("3", "5")] + [lam(S[k])] for k, n, p in syc_rows], "llrrr")
    v["s.plain.e3"] = pct(S["ours/circuit/plain"]["eps"]["3"])
    v["s.pm.e3"] = pct(S["google/pymatching"]["eps"]["3"])
    val = F["validation"]
    v["val.e3"], v["val.e5"] = pct(val["eps"]["3"]), pct(val["eps"]["5"])
    v["val.pub3"], v["val.pub5"] = pct(val["published"]["3"]), pct(val["published"]["5"])

    # Real time: latency on Willow's syndromes.
    L = load("data/realtime/latency.json")
    v["rt.machine"] = L["machine"]["cpu"]
    v["rt.cycle"] = f"{L['cycle_us']:.1f} µs"

    def keep(by):
        ks = sorted(by, key=int)
        return next((int(k) for k in ks if by[k]["keeps_up"]), None)

    rows = []
    for s_ in L["streams"]:
        k = keep(s_["by_workers"])
        at = s_["by_workers"][str(k)] if k else None
        rows.append([f"d = {s_['d']}", s_["mode"], s_["matcher"], us(s_["window_us"]["mean"]),
                     (f"{k} core{'s' if k > 1 else ''}" if k else ("no (one core by design)" if s_["mode"] == "sliding" else "no")),
                     us(at["mean_us"]) if at else "—", us(at["p99_us"]) if at else "—"])
        if s_["d"] == 5 and s_["mode"] == "parallel" and s_["matcher"] == "correlated":
            v["rt.d5.cores"] = str(k)
            v["rt.d5.mean"] = us(at["mean_us"])
            v["rt.d5.p99"] = us(at["p99_us"])
            v["rt.d5.window"] = us(s_["window_us"]["mean"])
        if s_["d"] == 7 and s_["mode"] == "parallel" and s_["matcher"] == "correlated":
            v["rt.d7.cores"] = str(k)
            v["rt.d7.mean"] = us(at["mean_us"])
        if s_["d"] == 3 and s_["mode"] == "sliding" and s_["matcher"] == "plain":
            v["rt.d3.sliding"] = "keeps up on one core" if k == 1 else "falls behind"
    t["latency"] = table(["stream", "windows", "matcher", "window decode", "keeps up on", "mean latency", "p99"], rows, "lllrlrr")

    # The million-round stream.
    M = load("data/realtime/million.json")
    v["m.p"] = pct(M["p"], 3)
    v["m.frac"] = pct(M["willow_detection_fraction"], 1)
    v["m.rounds"] = num(M["rounds"])
    rows = []
    for r in M["runs"]:
        k = keep(r["by_workers"])
        at = r["by_workers"][str(k)] if k else None
        rows.append([r["mode"], r["matcher"], r["streams"], num(r["windows"]), r["unexplained"], us(r["window_us"]["mean"]),
                     f"{k}" if k else "—", us(at["mean_us"]) if at else "—", us(at["p99_us"]) if at else "—"])
        if r["mode"] == "parallel" and r["matcher"] == "correlated":
            v["m.corr.cores"] = str(k)
            v["m.corr.mean"] = us(at["mean_us"])
            v["m.corr.p99"] = us(at["p99_us"])
            v["m.streams"] = str(r["streams"])
        if r["mode"] == "parallel" and r["matcher"] == "plain":
            v["m.plain.cores"] = str(k)
            v["m.plain.mean"] = us(at["mean_us"])
    v["m.unexplained"] = num(sum(r["unexplained"] for r in M["runs"]) + sum(x["unexplained"] for x in M["throughput"]))
    t["million"] = table(["windows", "matcher", "streams", "windows each", "unexplained", "window decode", "cores", "mean latency", "p99"],
                         rows, "llrrrrrrr")
    for x in M["throughput"]:
        v[f"m.tput.{x['matcher']}"] = f"{x['rounds_per_second'] / 1e6:.1f} million"
        v["m.tput.cores"] = str(x["cores"])
    v["m.willow_rate"] = f"{1 / (L['cycle_us'] * 1e-6) / 1e6:.2f} million"

    # Windows against global decoding.
    Wd = F["files"]["data/realtime/willow-windows.json"]
    order = [("global/plain", "global", "plain", "d"), ("window/sliding/Bd/plain", "sliding", "plain", "d"),
             ("window/parallel/Bd/plain", "parallel", "plain", "d"), ("window/parallel/Bhalf/plain", "parallel", "plain", "d/2 (d = 5 only)"),
             ("window/parallel/B2d/plain", "parallel", "plain", "2d (d = 5 only)"),
             ("global/correlated", "global", "correlated", "d"), ("window/sliding/Bd/correlated", "sliding", "correlated", "d"),
             ("window/parallel/Bd/correlated", "parallel", "correlated", "d"),
             ("window/parallel/Bhalf/correlated", "parallel", "correlated", "d/2 (d = 5 only)"),
             ("window/parallel/B2d/correlated", "parallel", "correlated", "2d (d = 5 only)")]
    rows = []
    for k, sched, m, buf in order:
        f = Wd[k]
        eps = [pct(f["eps"][d]) if d in f["eps"] else "—" for d in ("3", "5", "7")]
        rows.append([sched, m, "—" if sched == "global" else buf] + eps + [lam(f) if "lambda" in f else "—"])
    t["windows"] = table(["decoder", "matcher", "buffer", "ε, d = 3", "ε, d = 5", "ε, d = 7", "Λ [95%]"], rows, "lllrrrr")
    ww = load("data/realtime/willow-windows.json")["experiments"]
    v["win.experiments"] = num(len(ww))
    v["win.shots"] = num(next(iter(ww.values()))["shots"])
    v["win.unexplained"] = num(sum(x.get("unexplained", 0) for e in ww.values() for x in e["results"].values()))
    v["win.corr.lambda"] = lam(Wd["window/parallel/Bd/correlated"])
    v["win.global.lambda"] = lam(Wd["global/correlated"])
    half = Wd["window/parallel/Bhalf/correlated"]["eps"]["5"] / Wd["window/parallel/Bd/correlated"]["eps"]["5"] - 1
    v["win.half_cost"] = f"{half * 100:.1f}%"
    sd = load("data/realtime/sd6-windows.json")
    worst = 0.0
    for pnt in sd["points"]:
        for k, x in pnt["results"].items():
            if k.startswith("window"):
                g = pnt["results"][f"global/{k.split('/')[-1]}"]["failures"]
                worst = max(worst, abs(x["failures"] - g) / g)
    v["win.sd6_worst"] = f"{worst * 100:.1f}%"
    v["win.sd6_shots"] = num(sd["points"][0]["shots"])

    # Belief-matching.
    bc = load("data/belief/bp-check.json")
    v["bp.cases"] = num(sum(r["cases"] for r in bc["bp"]))
    v["bp.configs"] = num(len(bc["bp"]))
    v["bp.max_diff"] = f"{max(r['llr_abs'] for r in bc['bp']):.1f}"
    v["bp.bad"] = num(sum(r["hard"] + r["converge"] + r["iterations"] for r in bc["bp"]))
    v["bm.shots"] = num(sum(r["shots"] for r in bc["belief_matching"]))
    v["bm.disagree"] = num(sum(r["disagreements"] for r in bc["belief_matching"]))
    v["bm.conv_differ"] = num(sum(r["convergence_differs"] for r in bc["belief_matching"]))
    d7 = [r for r in bc["belief_matching"] if r["d"] == 7]
    v["bm.speed"] = f"{sum(r['ours_us'] for r in d7) / len(d7) / 1000:.1f} ms against {sum(r['theirs_us'] for r in d7) / len(d7) / 1000:.1f} ms"
    BS = F["files"]["data/belief/sycamore.json"]
    bsy = load("data/belief/sycamore.json")["experiments"]
    agree = sum(e["agree"]["ours/pij/belief"] for e in bsy.values())
    shots = sum(e["shots"] for e in bsy.values())
    v["bs.agree"] = f"{agree / shots * 100:.1f}%"
    v["bs.shots"] = f"{shots / 1e6:.1f} million"
    ours_f = sum(e["results"]["ours/pij/belief"]["failures"] for e in bsy.values())
    their_f = sum(e["results"]["google/belief_matching"]["failures"] for e in bsy.values())
    v["bs.more"] = f"{(ours_f / their_f - 1) * 100:.1f}%"
    for key, name in [("ours/pij/belief", "ours"), ("google/belief_matching", "google"), ("ours/circuit/belief", "circuit")]:
        v[f"bs.{name}.lambda"] = f"{BS[key]['lambda']:.3f} [{BS[key]['interval'][0]:.2f}, {BS[key]['interval'][1]:.2f}]"
        v[f"bs.{name}.e3"] = pct(BS[key]["eps"]["3"])
    rows = [("ours/pij/belief", "ours, belief-matching", "pij, cross-fitted"), ("ours/circuit/belief", "ours, belief-matching", "circuit"),
            ("google/belief_matching", "Google, belief matching", "pij"), ("google/tensor_network_contraction", "Google, tensor network", "pij"),
            ("google/correlated_matching", "Google, correlated matching", "circuit")]
    t["belief_sycamore"] = table(["decoder", "prior", "ε, d = 3", "ε, d = 5", "Λ [95%]"],
                                 [[n, pr] + [pct(BS[k]["eps"][d]) for d in ("3", "5")] + [lam(BS[k])] for k, n, pr in rows], "llrrr")
    BW = F["files"]["data/belief/willow.json"]
    bw = load("data/belief/willow.json")["experiments"]
    v["bw.experiments"] = num(len(bw))
    v["bw.shots"] = num(next(iter(bw.values()))["shots"])
    rows = [("ours/si1000/belief", "ours, belief-matching"), ("ours/si1000/correlated", "ours, correlated matching"),
            ("ours/si1000/plain", "ours, plain matching"), ("google/correlated_matching_decoder_with_si1000_prior", "Google, correlated matching"),
            ("google/libra_decoder_with_rl_optimized_prior", "Google, Libra (RL prior)")]
    t["belief_willow"] = table(["decoder, same shots", "ε, d = 3", "ε, d = 5", "ε, d = 7", "Λ [95%]"],
                               [[n] + [pct(BW[k]["eps"][d]) for d in ("3", "5", "7")] + [lam(BW[k])] for k, n in rows if "lambda" in BW[k]], "lrrrr")
    conv = sum(e["results"]["ours/si1000/belief"]["converged"] for e in bw.values())
    v["bw.converged"] = f"{conv / sum(e['shots'] for e in bw.values()) * 100:.0f}%"
    it = load("data/belief/iterations.json")["experiments"]["willow/d7_at_q6_7/Z/r50"]
    v["bi.d7.rounds"] = str(it["rounds"])
    v["bi.d7.b20"] = num(it["results"]["belief/20"]["failures"])
    v["bi.d7.b100"] = num(it["results"]["belief/100"]["failures"])
    v["bi.d7.corr"] = num(it["results"]["correlated"]["failures"])
    for key, name in [("ours/si1000/belief", "belief"), ("ours/si1000/correlated", "corr"), ("google/correlated_matching_decoder_with_si1000_prior", "gcorr")]:
        if "lambda" in BW[key]:
            v[f"bw.{name}.lambda"] = lam(BW[key])
            v[f"bw.{name}.e7"] = pct(BW[key]["eps"]["7"])
            v[f"bw.{name}.e3"] = pct(BW[key]["eps"]["3"])
    # The gross code.
    G = load("data/gross/results.json")
    bb = load("data/gross/bb-check.json")
    v["g.model_mechanisms"] = num(next(m["mechanisms"] for m in bb["models"] if m["code"] == "gross" and m["cycles"] == 12))
    v["g.osd_shots"] = num(sum(r["shots"] for r in bb["bposd"]))
    v["g.osd_same"] = num(sum(r["same"] for r in bb["bposd"]))

    def sci(x):
        e = math.floor(math.log10(x))
        return f"{x / 10 ** e:.1f} × 10{str(e).replace('-', '⁻').translate(str.maketrans('0123456789', '⁰¹²³⁴⁵⁶⁷⁸⁹'))}"

    rows = []
    for q in sorted(G["points"].values(), key=lambda q: (q["code"] != "gross", q["p"])):
        cs, o0 = q["results"]["bposd_cs7"], q["results"]["bposd_0"]
        rows.append(["gross [[144, 12, 12]]" if q["code"] == "gross" else "[[72, 12, 6]]", pct(q["p"], 1), num(q["shots"]),
                     num(cs["failures"]), f"{sci(cs['pl_cycle'])}", sci(o0["pl_cycle"]), f"{cs['converged'] / q['shots'] * 100:.1f}%"])
    t["gross"] = table(["code", "p", "shots", "failures", "per cycle, OSD-CS", "per cycle, OSD-0", "BP alone"], rows, "lrrrrrr")
    rows, ratios = [], []
    for p in sorted({q["p"] for q in G["points"].values()}):
        gp = G["points"][f"gross/{p}"]["results"]["bposd_cs7"]["pl_cycle"]
        sp = G["surface"][f"surface-d11/{p}"]["results"]["correlated_matching"]["pl_cycle"]
        twelve = 1 - (1 - sp) ** 12
        ratios.append(twelve / gp)
        rows.append([pct(p, 1), sci(gp), sci(twelve), f"{twelve / gp:.1f}×"])
    t["gross_vs_surface"] = table(["p", "gross code, 288 qubits", "twelve d = 11 patches, 2,892 qubits", "ratio"], rows, "rrrr")
    v["g.ratio_range"] = f"{min(ratios):.1f} to {max(ratios):.1f}"

    # Logical operations on the gross code: automorphisms and the gauging measurement.
    import readme_tables  # noqa: E402  (one definition of these rows)
    cells = lambda line: [c.strip() for c in line.strip("|").split("|")]
    AU = load("data/gross/automorphisms.json")
    GA = load("data/gross/gauging.json")
    GL = load("data/gross/logical.json")["points"]
    v["gl.order"] = str(AU["order"])
    v["gl.span"] = str(AU["families"]["f"]["span"])
    v["gl.classes"] = str(AU["families"]["f"]["classes"])
    ops = GA["operators"]
    fm, fx = ops["f"]["minimal"], ops["f"]["expanded"]
    v["gl.anc_f"], v["gl.anc_fx"], v["gl.extra_f"] = str(fm["ancillas"]), str(fx["ancillas"]), str(fx["extra"])
    v["gl.anc_fgh"], v["gl.anc_fghx"] = str(ops["f+gh"]["minimal"]["ancillas"]), str(ops["f+gh"]["expanded"]["ancillas"])
    v["gl.flux_f"], v["gl.flux_fx"] = ", ".join(map(str, fm["flux_weights"])), str(max(fx["flux_weights"]))
    v["gl.ticks_f"], v["gl.ticks_fx"] = str(fm["ticks"]), str(fx["ticks"])
    v["gl.cut_f"] = f"{fm['worst_cut'][1]} vertices with {fm['worst_cut'][0]} edges leaving them"
    dist = lambda op: f"{readme_tables.distance_text(op['distance']['x'])} in X and {readme_tables.distance_text(op['distance']['z'])} in Z"
    v["gl.dist_f"], v["gl.dist_fx"] = dist(fm), dist(fx)
    v["gl.code_d"] = readme_tables.distance_text(GA["code"]["distance"]["z"])
    t["gross_gauging"] = table(["operator", "system", "ancillas (edges + Gauss + flux)", "heaviest flux", "ticks per cycle",
                                "worst cut", "distance, X", "distance, Z"],
                               [cells(r) for r in readme_tables.gross_gauging()], "llrrrrrr")
    t["gross_logical"] = table(["operator", "system", "basis", "T", "shots", "anything wrong", "outcome wrong", "memory, same length"],
                               [cells(r) for r in readme_tables.gross_logical()], "lllrrrrr")
    for T in (2, 7, 12):
        v[f"gl.outcome_T{T}"] = pct(GL[f"f/expanded/x/T{T}/p0.003"]["rate_first"], 2)
    for b in ("x", "z"):
        m = GL[f"memory/{b}/R19/p0.003"]["rate_any"]
        for c, tag in (("expanded", ""), ("minimal", "m")):
            v[f"gl.ratio_{b}{tag}"] = f"{GL[f'f/{c}/{b}/T7/p0.003']['rate_any'] / m:.1f}"
    grow = lambda a, b, n: (GL[b]["rate_any"] - GL[a]["rate_any"]) / n * 100
    v["gl.grow_merged"] = f"{grow('f/expanded/z/T2/p0.003', 'f/expanded/z/T12/p0.003', 10):.1f}%"
    v["gl.grow_memory"] = f"{grow('memory/z/R14/p0.003', 'memory/z/R24/p0.003', 10):.3f}%"
    q = GL["f/expanded/z/T12/p0.003"]
    v["gl.bp_T12"] = f"{q['failures']['converged'] / q['shots'] * 100:.0f}%"
    q = GL["memory/z/R24/p0.003"]
    v["gl.bp_mem"] = f"{q['failures']['converged'] / q['shots'] * 100:.0f}%"
    FA = load("data/gross/faults.json")["circuits"]
    # A circuit whose search found nothing within its cap gives no bound.
    found = lambda memory: [c["upper"] for k, c in FA.items() if k.startswith("memory") == memory and c["upper"] is not None]
    v["gl.fault_merged"], v["gl.fault_memory"] = str(min(found(False))), str(min(found(True)))

    # Lattice surgery.
    S = load("data/surgery/results.json")["points"]
    rows = []
    for p in (0.003, 0.002):
        for d in (3, 5, 7):
            get = lambda T, m="correlated": S.get(f"d{d}/T{T}/p{p}/{m}")
            cells = [get(2), get(d), get(2 * d), get(d, "plain")]
            if not all(cells):
                continue
            f = lambda q, k="outcome": f"{q['rate_' + k] * 100:.2f}%"
            rows.append([str(d), pct(p, 1), f(cells[0]), f(cells[1]), f(cells[2]), f(cells[3]), f(cells[1], "patches")])
    t["surgery"] = table(["d", "p", "T = 2", "T = d", "T = 2d", "T = d, plain", "either patch, T = d"], rows, "rrrrrrr")

    # Lattice-surgery programs: the CNOT, merges in a row, the three-patch product.
    import readme_tables  # noqa: E402  (one definition of these rows)
    PG = load("data/surgery/programs.json")["points"]
    cells = lambda line: [c.strip() for c in line.strip("|").split("|")]
    t["surgery_cnot"] = table(["d", "p", "Z inputs", "X inputs", "Z inputs, plain", "three idle patches", "CNOT ÷ idle"],
                              [cells(r) for r in readme_tables.surgery_cnot()], "rrrrrrr")
    t["surgery_seq"] = table(["d", "k = 1", "k = 2", "k = 4", "k = 8", "per merge"],
                             [cells(r) for r in readme_tables.surgery_sequences()], "rrrrrr")
    idle = {(d, p): 1 - (1 - PG[f"memory/d{d}/R{4 * d}/p{p}"]["rate_any"]) ** 3 for d in (3, 5, 7) for p in (0.002, 0.003)}
    ratios = [PG[f"cnot/d{d}/T{d}/p{p}/{i}/correlated"]["rate_any"] / idle[(d, p)] for d in (3, 5, 7) for p in (0.002, 0.003) for i in "zx"]
    v["ls.cnot_ratio"] = f"{min(ratios):.2f} to {max(ratios):.2f}"
    v["ls.cnot_d7"] = pct(PG["cnot/d7/T7/p0.002/z/correlated"]["rate_any"], 2)
    v["ls.line3"] = ", ".join(pct(PG[f"line/d{d}/n3/p0.003"]["rate_any"], 2) for d in (3, 5, 7))

    # The resource estimate, by the page's own model (js/estimator.js via tools/estimate.mjs).
    est_path = BUILD / "estimate.json"
    subprocess.run(["node", "tools/estimate.mjs", "--json", str(est_path)], cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    E = json.loads(est_path.read_text())
    labels = {"small": "100 qubits × 10⁶ operations", "medium": "1,000 qubits × 10⁹", "large": "10,000 qubits × 10¹²"}
    names = {"ours": "ours, correlated", "libra": "Google's Libra", "belief": "ours, belief-matching"}

    t["estimate"] = table(["algorithm", "Λ (from)", "d", "physical qubits", "run time", "decoding cores"],
                          [[labels[r["preset"]], f"{r['lambda']:.2f} ({names[r['source']]})", str(r["d"]), r["text"]["physical"],
                            r["text"]["seconds"], r["text"]["cores"]] for r in E["rows"]], "llrrrr")
    medium = next(r for r in E["rows"] if r["preset"] == "medium" and r["source"] == "ours")
    v["est.medium.d"] = str(medium["d"])
    v["est.medium.qubits"] = medium["text"]["physical"]
    v["est.medium.time"] = medium["text"]["seconds"]
    v["est.lever.d"] = str(E["lever"]["d"])
    v["est.lever.qubits"] = E["lever"]["text"]["physical"]
    return v, t


# -- figures -----------------------------------------------------------------

def figures(F):
    import matplotlib
    matplotlib.use("svg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    plain = FuncFormatter(lambda x, _: f"{x:g}")
    plt.rcParams.update({
        "svg.fonttype": "path", "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"], "font.size": 9,
        "axes.edgecolor": PALETTE["ink"], "axes.labelcolor": PALETTE["ink2"], "xtick.color": PALETTE["ink2"],
        "ytick.color": PALETTE["ink2"], "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
        "grid.color": PALETTE["rule"], "grid.linewidth": 0.6, "legend.frameon": False, "figure.dpi": 100,
        "svg.hashsalt": "report",
    })
    # No build date in the SVGs: a rebuild changes only the figures whose data changed.
    STABLE = {"Date": None}
    FIGS.mkdir(parents=True, exist_ok=True)
    ref = load("data/xcheck/reference.json")

    # Figure 1: Willow, ε against d.
    W = F["willow"]
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    series = [("ours/si1000/plain", "ours, plain", PALETTE["ink3"], "o"),
              ("ours/si1000/correlated", "ours, correlated", PALETTE["d5"], "o"),
              ("google/correlated_matching_decoder_with_si1000_prior", "Google, correlated matching", PALETTE["d3"], "s"),
              ("google/libra_decoder_with_rl_optimized_prior", "Google, Libra (RL prior)", PALETTE["d7"], "^")]
    for k, label, color, marker in series:
        ds = [3, 5, 7]
        eps = [W[k]["eps"][str(d)] * 100 for d in ds]
        lo = [W[k]["epsInterval"][str(d)][0] * 100 for d in ds]
        hi = [W[k]["epsInterval"][str(d)][1] * 100 for d in ds]
        ax.errorbar(ds, eps, yerr=[[e - a for e, a in zip(eps, lo)], [b - e for e, b in zip(eps, hi)]], color=color,
                    marker=marker, ms=5, lw=1.4, capsize=2, label=f"{label}: Λ = {W[k]['lambda']:.2f}")
    ax.set_yscale("log")
    ax.set_xticks([3, 5, 7])
    ax.set_yticks([0.2, 0.3, 0.5, 0.7, 1.0])
    ax.set_yticklabels(["0.2%", "0.3%", "0.5%", "0.7%", "1.0%"])
    ax.minorticks_off()
    ax.set_xlabel("code distance d")
    ax.set_ylabel("logical error per cycle, ε")
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIGS / "willow-lambda.svg", metadata=STABLE)
    plt.close(fig)

    # Figure 2: agreement with PyMatching.
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    dec, cor = ref["decoding"], ref["correlated"]
    x = [r["pymatching_failures"] / r["shots"] * 100 for r in dec]
    y = [r["ours_failures"] / r["shots"] * 100 for r in dec]
    ax.scatter(x, y, s=22, color=PALETTE["d3"], label="plain", zorder=3)
    x2 = [r["pymatching_failures"] / r["shots"] * 100 for r in cor]
    y2 = [r["ours_failures"] / r["shots"] * 100 for r in cor]
    ax.scatter(x2, y2, s=22, color=PALETTE["d5"], marker="s", label="correlated", zorder=3)
    lim = [0.4, 13]
    ax.plot(lim, lim, color=PALETTE["ink3"], lw=0.8, zorder=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ticks = [0.5, 1, 2, 5, 10]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels([f"{t:g}%" for t in ticks])
    ax.set_yticklabels([f"{t:g}%" for t in ticks])
    ax.minorticks_off()
    ax.set_xlabel("PyMatching's logical error rate")
    ax.set_ylabel("ours, on the same shots")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGS / "pymatching-agreement.svg", metadata=STABLE)
    plt.close(fig)

    # Figure 3: speed.
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.4, 2.9))
    for rows, key_ours, key_pm, color, label in [(dec, "ours_us", "pymatching_us", PALETTE["d3"], "plain"),
                                                  (cor, "ours_us", "pymatching_us", PALETTE["d5"], "correlated")]:
        pts = sorted((r["d"], r[key_ours], r[key_pm]) for r in rows if r["p"] == 0.003 and r.get("code", "rotated") == "rotated")
        ds = [p[0] for p in pts]
        a1.plot(ds, [p[1] for p in pts], color=color, marker="o", ms=4, lw=1.4, label=f"ours, {label}")
        a1.plot(ds, [p[2] for p in pts], color=color, marker="o", ms=4, lw=1.2, ls="--", mfc="white", label=f"PyMatching, {label}")
    a1.set_yscale("log")
    a1.yaxis.set_major_formatter(plain)
    a1.set_xticks([3, 5, 7])
    a1.set_xlabel("d  (SD6, p = 0.3%, T = d)")
    a1.set_ylabel("decoding, µs per shot")
    a1.legend(fontsize=7)
    pts = sorted((r["d"], r["sample_us"]) for r in dec if r["p"] == 0.003)
    for key, label, color, ls in [("stim", "Stim", PALETTE["ink3"], "--"), ("frame", "frame sampler", PALETTE["amber"], "-"),
                                  ("batch", "batch sampler", PALETTE["d7"], "-")]:
        a2.plot([p[0] for p in pts], [p[1][key] for p in pts], color=color, marker="o", ms=4, lw=1.4, ls=ls, label=label)
    a2.set_yscale("log")
    a2.yaxis.set_major_formatter(plain)
    a2.set_xticks([3, 5, 7])
    a2.set_xlabel("d  (SD6, p = 0.3%, T = d)")
    a2.set_ylabel("sampling, µs per shot")
    a2.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGS / "speed.svg", metadata=STABLE)
    plt.close(fig)

    # The gross code: error per cycle against p.
    G = load("data/gross/results.json")
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    for code, color, label in [("gross", PALETTE["d3"], "gross [[144, 12, 12]]"), ("72", PALETTE["d5"], "[[72, 12, 6]]")]:
        for dec, ls, tag in [("bposd_cs7", "-", "BP+OSD-CS"), ("bposd_0", "--", "BP+OSD-0")]:
            pts = sorted((q["p"], q["results"][dec]) for q in G["points"].values() if q["code"] == code)
            xs = [x * 100 for x, _ in pts]
            ys = [r["pl_cycle"] for _, r in pts]
            ax.plot(xs, ys, color=color, ls=ls, marker="o", ms=4, lw=1.4, label=f"{label}, {tag}")
    ps = sorted({q["p"] for q in G["points"].values()})
    ax.plot([x * 100 for x in ps], [1 - (1 - x) ** 12 for x in ps], color=PALETTE["ink3"], lw=1.2, label="12 unprotected qubits")
    ax.set_yscale("log")
    ax.set_xlabel("physical error rate p, %")
    ax.set_ylabel("logical error per syndrome cycle")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGS / "gross.svg", metadata=STABLE)
    plt.close(fig)

    # Lattice surgery: the merge outcome against merged rounds.
    Sg = load("data/surgery/results.json")["points"]
    fig, ax = plt.subplots(figsize=(5.6, 3.3))
    for d, color in [(3, PALETTE["d3"]), (5, PALETTE["d5"]), (7, PALETTE["d7"])]:
        for p, ls in [(0.003, "-"), (0.002, "--")]:
            pts = sorted((q["merged"], q["rate_outcome"]) for q in Sg.values() if q["d"] == d and q["p"] == p and q["matcher"] == "correlated")
            ax.plot([x for x, _ in pts], [y for _, y in pts], color=color, ls=ls, marker="o", ms=3.5, lw=1.3,
                    label=f"d = {d}, p = {p * 100:.1f}%")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(plain)
    ax.set_xlabel("merged rounds T")
    ax.set_ylabel("merge outcome wrong, per shot")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(FIGS / "surgery.svg", metadata=STABLE)
    plt.close(fig)

    # Figure 4: latency against cores.
    L = load("data/realtime/latency.json")
    fig, ax = plt.subplots(figsize=(5.6, 3.3))
    for d, color in [(3, PALETTE["d3"]), (5, PALETTE["d5"]), (7, PALETTE["d7"])]:
        s = next(x for x in L["streams"] if x["d"] == d and x["mode"] == "parallel" and x["matcher"] == "correlated")
        ks = sorted(s["by_workers"], key=int)
        xs = [int(k) for k in ks]
        mean = [s["by_workers"][k]["mean_us"] for k in ks]
        p99 = [s["by_workers"][k]["p99_us"] for k in ks]
        ax.errorbar(xs, mean, yerr=[[0] * len(xs), [b - a for a, b in zip(mean, p99)]], color=color, marker="o", ms=4,
                    lw=1.4, capsize=2, label=f"d = {d}")
    ax.axhline(63, color=PALETTE["ink"], lw=1, ls="--", label="Google's real-time decoder, d = 5 (published)")
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8, 16])
    ax.set_xticklabels(["1", "2", "4", "8", "16"])
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(plain)
    ax.set_ylim(8, 30000)
    ax.set_xlabel("cores decoding one stream")
    ax.set_ylabel("latency, µs (mean; bar to p99)")
    ax.legend(fontsize=7.5, loc="upper right", ncol=2)
    fig.tight_layout()
    fig.savefig(FIGS / "latency.svg", metadata=STABLE)
    plt.close(fig)


# -- filling, rendering ------------------------------------------------------

def fill(text, v, t, readme):
    missing = []

    def sub(m):
        key = m.group(1).strip()
        if key.startswith("table:"):
            name = key[6:]
            if name not in t:
                missing.append(key)
                return m.group(0)
            return t[name]
        if key.startswith("readme:"):
            phrase = key[7:]
            if phrase not in readme:
                missing.append(key)
            return phrase
        if key not in v:
            missing.append(key)
            return m.group(0)
        if re.search(r"\bnan\b|\binf\b", v[key]):
            missing.append(f"{key} (not finite: {v[key]})")
        return v[key]

    out = re.sub(r"\{\{(.+?)\}\}", sub, text, flags=re.S)
    if missing:
        sys.exit("report: cannot fill " + ", ".join(sorted(set(missing))))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-pdf", action="store_true")
    args = ap.parse_args()
    BUILD.mkdir(parents=True, exist_ok=True)
    F = fits()
    v, t = build_values(F)
    figures(F)
    readme = re.sub(r"\s+", " ", (ROOT / "README.md").read_text())
    source = (REPORT / "report.md").read_text()
    # README phrases are matched with their whitespace collapsed, as they wrap differently.
    source_flat = re.sub(r"\{\{readme:(.+?)\}\}", lambda m: "{{readme:" + re.sub(r"\s+", " ", m.group(1)) + "}}", source, flags=re.S)
    filled = fill(source_flat, v, t, readme)
    (BUILD / "report.md").write_text(filled)
    subprocess.run(["pandoc", str(BUILD / "report.md"), "--from", "markdown+pipe_tables+tex_math_dollars", "--to", "html5",
                    "--standalone", "--template", str(REPORT / "template.html"), "--mathml", "--toc", "--toc-depth", "2",
                    "--number-sections", "--output", str(REPORT / "report.html")], cwd=REPORT, check=True)
    print(f"report/report.html: {len(v)} values, {len(t)} tables")
    if not args.no_pdf:
        subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", "--run-all-compositor-stages-before-draw",
                        "--virtual-time-budget=15000", f"--print-to-pdf={REPORT / 'report.pdf'}",
                        (REPORT / "report.html").as_uri()], check=True, capture_output=True)
        print(f"report/report.pdf: {(REPORT / 'report.pdf').stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()

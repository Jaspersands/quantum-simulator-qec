"""The README's tables, generated from the committed data, so none can fall
behind what was measured.

    python3 tools/readme_tables.py            # print every table
    python3 tools/readme_tables.py --write    # replace them in README.md
    python3 tools/readme_tables.py --check    # exit 1 if any differs (CI)
"""
import json, math, pathlib, re, sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
load = lambda path: json.loads((ROOT / path).read_text())


def sci(x):
    if x == 0:
        return "0"
    e = math.floor(math.log10(x)); m = x / 10 ** e
    if round(m, 1) >= 10:
        m, e = m / 10, e + 1
    return f"{m:.1f} × 10{str(e).replace('-', '⁻').translate(str.maketrans('0123456789', '⁰¹²³⁴⁵⁶⁷⁸⁹'))}"


def pct(k, n):
    return f"{k / n * 100:.3f}%"


def time_us(x):
    if x < 100:
        return f"{x:.1f} µs"
    if x < 1000:
        return f"{x:.0f} µs"
    return f"{x / 1000:.2f} ms" if x < 10_000 else f"{x / 1000:.1f} ms"


def gross():
    g = load("data/gross/results.json")
    rows = []
    for code, name in (("gross", "gross [[144, 12, 12]]"), ("72", "[[72, 12, 6]]")):
        for p in (0.002, 0.003, 0.004, 0.005, 0.006):
            q = g["points"][f"{code}/{p}"]; cs = q["results"]["bposd_cs7"]; o0 = q["results"]["bposd_0"]
            lo, hi = cs["pl_cycle_interval"]
            rows.append(f"| {name} | {p * 100:.1f}% | {q['shots']:,} | {cs['failures']} | {sci(cs['pl_cycle'])} "
                        f"[{sci(lo)}, {sci(hi)}] | {sci(o0['pl_cycle'])} | {cs['converged'] / q['shots'] * 100:.1f}% |")
    return rows


def gross_vs_surface():
    g = load("data/gross/results.json")
    rows = []
    for p in (0.002, 0.003, 0.004, 0.005, 0.006):
        gp = g["points"][f"gross/{p}"]["results"]["bposd_cs7"]["pl_cycle"]
        sp = g["surface"][f"surface-d11/{p}"]["results"]["correlated_matching"]["pl_cycle"]
        twelve = 1 - (1 - sp) ** 12
        rows.append(f"| {p * 100:.1f}% | {sci(gp)} | {sci(twelve)} | {twelve / gp:.1f}× |")
    return rows


def xcheck_plain():
    rows = []
    for r in load("data/xcheck/reference.json")["decoding"]:
        n = r["shots"]
        rows.append(f"| {r['d']} | {r['p'] * 100:.1f}% | {pct(r['pymatching_failures'], n)} | {pct(r['ours_failures'], n)} | "
                    f"{pct(r['ours_own_dem_failures'], n)} | {r['disagreements']} ({r['non_ties']}) | "
                    f"{pct(r['sampler']['our_failures'], n)} | {r['sampler']['z']:+.2f} | {r['pymatching_us']:.1f} µs | {r['ours_us']:.1f} µs |")
    return rows


def xcheck_corr():
    rows = []
    for r in load("data/xcheck/reference.json")["correlated"]:
        n = r["shots"]
        code = "rotated" if r["code"] == "rotated" else "XZZX"
        rows.append(f"| {code} | {r['d']} | {r['p'] * 100:.1f}% | {pct(r['pymatching_failures'], n)} | {pct(r['ours_failures'], n)} | "
                    f"{pct(r['plain_failures'], n)} | {r['disagreements']} | {r['pymatching_us']:.1f} | {r['ours_us']:.1f} |")
    return rows


def speed():
    native = load("data/matcher/native.json")["points"]
    pm = {(r["d"], r["p"]): r["pymatching_us"] for r in load("data/xcheck/reference.json")["decoding"]}
    rows = []
    for r in native:
        m = pm.get((r["d"], r["p"]))
        rows.append(f"| {r['d']} | {r['p'] * 100:.1f}% | {time_us(r['dense_us'])} | {time_us(r['sparse_us'])} | "
                    f"{time_us(m) if m is not None else '—'} |")
    return rows


def latency():
    lat, mil = load("data/realtime/latency.json"), load("data/realtime/million.json")
    us = lambda x: f"{x:.1f} µs" if x < 10 else f"{x:.0f} µs"
    whole = lambda x: f"{x:.0f} µs"
    cores = lambda k: f"{k} core" + ("" if k == 1 else "s")

    def first_k(by):
        # The fewest workers that keep up, or None if none measured does.
        return min((int(k) for k, v in by.items() if v["keeps_up"]), default=None)

    rows = []
    for d, mode, m in ((3, "sliding", "plain"), (3, "parallel", "correlated"), (5, "parallel", "plain"),
                       (5, "parallel", "correlated"), (7, "parallel", "plain"), (7, "parallel", "correlated")):
        s = next(x for x in lat["streams"] if x["d"] == d and x["mode"] == mode and x["matcher"] == m)
        k = first_k(s["by_workers"])
        if k is None:
            rows.append(f"| Willow d = {d} | {mode}, {m} | {us(s['window_us']['mean'])} | falls behind | — | — |")
            continue
        w = s["by_workers"][str(k)]
        rows.append(f"| Willow d = {d} | {mode}, {m} | {us(s['window_us']['mean'])} | {cores(k)} | {whole(w['mean_us'])} | {whole(w['p99_us'])} |")
    for m, k2 in (("plain", 4), ("correlated", 8)):
        r = next(x for x in mil["runs"] if x["mode"] == "parallel" and x["matcher"] == m)
        k = first_k(r["by_workers"])
        if k is None:
            rows.append(f"| SD6 d = 5, 10⁶ rounds | parallel, {m} | {us(r['window_us']['mean'])} | falls behind | — | — |")
            continue
        w = r["by_workers"][str(k)]; w2 = r["by_workers"][str(k2)]
        if k2 > k and w2["mean_us"] < 0.8 * w["mean_us"]:
            rows.append(f"| SD6 d = 5, 10⁶ rounds | parallel, {m} | {us(r['window_us']['mean'])} | {cores(k)} ({k2} for {whole(w2['mean_us'])}) | "
                        f"{whole(w['mean_us'])} ({whole(w2['mean_us'])} on {k2}) | {whole(w['p99_us'])} ({whole(w2['p99_us'])} on {k2}) |")
        else:
            rows.append(f"| SD6 d = 5, 10⁶ rounds | parallel, {m} | {us(r['window_us']['mean'])} | {cores(k)} | {whole(w['mean_us'])} | {whole(w['p99_us'])} |")
    return rows


def surgery_timing():
    pts = load("data/surgery/results.json")["points"]
    rows = []
    for p in (0.003, 0.002):
        for d in (3, 5, 7):
            get = lambda T, m="correlated": pts.get(f"d{d}/T{T}/p{p}/{m}")
            cells = [get(2), get(d), get(2 * d), get(d, "plain")]
            if not all(cells):
                continue
            f = lambda q, k="outcome": f"{q['rate_' + k] * 100:.2f}%"
            rows.append(f"| {d} | {p * 100:.1f}% | {f(cells[0])} | {f(cells[1])} | {f(cells[2])} | {f(cells[3])} | {f(cells[1], 'patches')} |")
    return rows


def surgery_cnot():
    P = load("data/surgery/programs.json")["points"]
    rows = []
    for p in (0.002, 0.003):
        for d in (3, 5, 7):
            z, x = P[f"cnot/d{d}/T{d}/p{p}/z/correlated"], P[f"cnot/d{d}/T{d}/p{p}/x/correlated"]
            plain = P[f"cnot/d{d}/T{d}/p{p}/z/plain"]
            idle = 1 - (1 - P[f"memory/d{d}/R{4 * d}/p{p}"]["rate_any"]) ** 3
            rows.append(f"| {d} | {p * 100:.1f}% | {z['rate_any'] * 100:.2f}% | {x['rate_any'] * 100:.2f}% | "
                        f"{plain['rate_any'] * 100:.2f}% | {idle * 100:.2f}% | {z['rate_any'] / idle:.2f} |")
    return rows


def per_merge(points):
    """The failure each merge adds: a least-squares line through ln(1 - P) against k."""
    xs = [q["k"] for q in points]; ys = [math.log(1 - q["rate_any"]) for q in points]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    return 1 - math.exp(slope)


def surgery_sequences():
    P = load("data/surgery/programs.json")["points"]
    rows = []
    for d in (3, 5):
        pts = [P[f"repeated/d{d}/k{k}/p0.003"] for k in (1, 2, 4, 8)]
        rows.append(f"| {d} | " + " | ".join(f"{q['rate_any'] * 100:.2f}%" for q in pts) + f" | {per_merge(pts) * 100:.2f}% |")
    return rows


GROSS_OPS = [("f", "X(f, 0)"), ("gh", "X(g, h)"), ("f+gh", "X(f, 0) X(g, h)")]


def distance_text(d):
    """12 when exact, "≥ 11" for a bound the integer program stopped at."""
    return str(d["value"]) if d["exact"] else f"≥ {d['lower']}"


def gross_gauging():
    doc = load("data/gross/gauging.json")
    rows = []
    for key, label in GROSS_OPS:
        for construction in ("minimal", "expanded"):
            op = doc["operators"].get(key, {}).get(construction)
            if not op:
                continue
            d = op["distance"]
            rows.append(f"| {label} | {construction} | {op['ancillas']} ({op['edges']} + {op['gauss']} + {op['flux']}) | "
                        f"{max(op['flux_weights'])} | {op['ticks']} | {op['worst_cut'][0]} / {op['worst_cut'][1]} | "
                        f"{distance_text(d['x'])} | {distance_text(d['z'])} |")
    return rows


def gross_logical():
    P = load("data/gross/logical.json")["points"]
    pct = lambda x: f"{x * 100:.2f}%"
    rows = []
    for key, label in GROSS_OPS:
        for construction in ("expanded", "minimal"):
            for basis in ("x", "z"):
                for T in (2, 4, 7, 12):
                    q = P.get(f"{key}/{construction}/{basis}/T{T}/p0.003")
                    if not q:
                        continue
                    m = P.get(f"memory/{basis}/R{q['pre'] + T + q['post']}/p0.003")
                    outcome = pct(q["rate_first"]) if basis == "x" else "—"
                    memory = pct(m["rate_any"]) if m else "—"
                    rows.append(f"| {label} | {construction} | {basis.upper()} | {T} | {q['shots']:,} | {pct(q['rate_any'])} | {outcome} | {memory} |")
    return rows


_ESTIMATE = None


def estimate_json():
    """tools/estimate.mjs's output: the page's own model, run once per invocation."""
    global _ESTIMATE
    if _ESTIMATE is None:
        import json as _json
        import subprocess
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            out = pathlib.Path(tmp) / "estimate.json"
            subprocess.run(["node", "tools/estimate.mjs", "--json", str(out)], cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
            _ESTIMATE = _json.loads(out.read_text())
    return _ESTIMATE


def estimate_full():
    rows = []
    for r in estimate_json()["full"]:
        if "error" in r["text"]:
            rows.append(f"| {r['label']} | {r['variant']} | — | {r['text']['error']} | | | |")
        else:
            t = r["text"]
            rows.append(f"| {r['label']} | {r['variant']} | {r['d']} | {t['qubits']} ({t['block']} + {t['factories']} + {t['storage']}) | "
                        f"{t['perToffoli']} | {t['seconds']} | {'steps' if r['bound'] == 'clifford' else 'reaction'} |")
    return rows


def estimate_validation():
    rows = []
    for group, who in (("rsa", "RSA-2048"), ("femoco", "FeMoco (Reiher)")):
        for r in estimate_json()["validation"][group]:
            if "error" in r["text"]:
                rows.append(f"| {who}: {r['label']} | — | {r['text']['error']} | |")
            else:
                rows.append(f"| {who}: {r['label']} | {r['d']} | {r['text']['qubits']} ({r['qubitRatio']:.2f}×) | "
                            f"{r['text']['seconds']} ({r['timeRatio']:.2f}×) |")
    return rows


def estimate_lambda():
    labels = {"small": "100 qubits × 10⁶ operations", "medium": "1,000 qubits × 10⁹ operations", "large": "10,000 qubits × 10¹² operations"}
    names = {"ours": "ours, correlated matching", "libra": "Google's Libra", "belief": "ours, belief-matching"}
    return [f"| {labels[r['preset']]} | {r['lambda']:.2f} ({names[r['source']]}) | {r['d']} | {r['text']['physical']} | "
            f"{r['text']['seconds']} | {r['text']['cores']} |" for r in estimate_json()["rows"]]


def estimate_noise():
    doc = load("data/estimate/noise.json")
    rows = []
    for p in (0.001, 0.002, 0.003, 0.005):
        cells = []
        for d in (3, 5, 7, 9, 11):
            q = doc["points"].get(f"d{d}/p{p}")
            cells.append("—" if not q else f"{q['eps']:.1e}" + ("" if q["failures"] >= 20 else " *"))
        rows.append(f"| {p * 100:.1f}% | " + " | ".join(cells) + " |")
    return rows


# Each table's header row (as a regex), and the function giving its body.
TABLES = [
    ("gross", r"\| code \| p \| shots \| failures \| per cycle, BP\+OSD-CS \[95%\] \| per cycle, BP\+OSD-0 \| BP alone \|", gross),
    ("gross beside the surface code", r"\| p \| gross code: 12 logical qubits on 288 \| twelve d = 11 surface patches on 2,892 \| ratio \|", gross_vs_surface),
    ("cross-check, plain", r"\| d \| p \| PyMatching \| ours, on Stim's graph \|[^\n]*", xcheck_plain),
    ("cross-check, correlated", r"\| code \| d \| p \| PyMatching correlated \|[^\n]*", xcheck_corr),
    ("matcher speed", r"\| d \| p \| dense \| sparse \| PyMatching \|", speed),
    ("latency", r"\| stream \| decoder \| window decode \| keeps up on \| mean latency \| p99 \|", latency),
    ("surgery timing", r"\| d \| p \| T = 2 \| T = d \| T = 2d \| T = d, plain \| either patch, T = d \|", surgery_timing),
    ("surgery CNOT", r"\| d \| p \| Z inputs \| X inputs \| Z inputs, plain \| three patches idle as long \| CNOT ÷ idle \|", surgery_cnot),
    ("surgery sequences", r"\| d \| k = 1 \| k = 2 \| k = 4 \| k = 8 \| per merge \|", surgery_sequences),
    ("gross gauging", r"\| operator \| system \| ancilla qubits \(edges \+ Gauss \+ flux\) \| heaviest flux check \| ticks per merged cycle \| worst cut \(edges out / vertices\) \| distance, X \| distance, Z \|", gross_gauging),
    ("gross logical", r"\| operator \| system \| basis \| T \| shots \| anything wrong \| outcome wrong \| memory, same length \|", gross_logical),
    ("estimate noise", r"\| p \| d = 3 \| d = 5 \| d = 7 \| d = 9 \| d = 11 \|", estimate_noise),
    ("estimate validation", r"\| case \| d \| physical qubits \(× source\) \| time \(× source\) \|", estimate_validation),
    ("estimate full", r"\| algorithm \| decoder \| d \| physical qubits \(block \+ factories \+ storage\) \| per Toffoli \| run time \| bound \|", estimate_full),
    ("estimate lambda", r"\| algorithm \| Λ \(from\) \| d \| physical qubits \| run time \| decoding cores \|", estimate_lambda),
]


def main():
    readme = (ROOT / "README.md").read_text()
    out, stale = readme, []
    for name, header, fn in TABLES:
        body = "\n".join(fn()) + "\n"
        pattern = re.compile("(" + header + r"\n\|---(?:\|---)*\|\n)((?:\|.*\|\n)+)")
        m = pattern.search(out)
        if not m:
            raise SystemExit(f"{name}: header not found in README.md")
        if m.group(2) != body:
            stale.append(name)
        out = out[:m.start(2)] + body + out[m.end(2):]
        if "--write" not in sys.argv and "--check" not in sys.argv:
            print(f"## {name}\n{body}")
    if "--check" in sys.argv:
        for name in stale:
            print(f"README table out of date: {name}")
        return 1 if stale else 0
    if "--write" in sys.argv:
        (ROOT / "README.md").write_text(out)
        print(f"rewrote {len(stale)} table(s): {', '.join(stale) or 'none'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

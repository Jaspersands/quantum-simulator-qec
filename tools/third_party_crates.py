"""Write THIRD_PARTY_CRATES.md: the Rust crates compiled into each wheel's module
(``stabilizer_qec/_core``), with the licence texts and notices their packages carry.

    python tools/third_party_crates.py            # rewrite THIRD_PARTY_CRATES.md
    python tools/third_party_crates.py --check    # fail if it differs from what Cargo.lock gives

The wheels are built with the features in pyproject.toml's ``[tool.maturin]`` for the targets of
.github/workflows/wheels.yml (TARGETS below). For each target, ``cargo tree -e normal`` lists the
crates the module is built from; this file lists every one with its licence and the wheels that
link it, then the licence files (LICENSE*, COPYING*, COPYRIGHT*, NOTICE*) of each crate's package
as published, each text once with the crates that carry it. A crate whose package carries no
licence file and whose licence allows Apache-2.0 is taken under Apache-2.0 (its text is
LICENSES/Apache-2.0.txt); one that allows only licences needing its copyright line takes it from
NO_FILE below, so that a new such crate stops this tool until someone has looked it up. The
crates of Cargo.lock that no wheel links (build scripts' tools, other targets' crates) are listed
by name, so that tests/test_third_party_notices.py notices a Cargo.lock this file was not
regenerated for.

cargo reads the crates from its registry (``--offline``, the default; ``--online`` lets it fetch).
The tool needs every crate of every wheel target with the wheels' features, which a registry filled
by a build for one host does not hold: when cargo stops offline for a crate or an index entry it has
not got, the tool exits with OFFLINE_UNAVAILABLE (3), not 1, so that the test can tell a registry
that cannot answer from a THIRD_PARTY_CRATES.md that is out of date.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "THIRD_PARTY_CRATES.md"

# The targets of the wheels (wheels.yml): Linux glibc and musl, macOS universal2, Windows.
TARGETS = [
    ("Linux", "x86_64-unknown-linux-gnu"),
    ("Linux", "aarch64-unknown-linux-gnu"),
    ("Linux", "x86_64-unknown-linux-musl"),
    ("Linux", "aarch64-unknown-linux-musl"),
    ("macOS", "x86_64-apple-darwin"),
    ("macOS", "aarch64-apple-darwin"),
    ("Windows", "x86_64-pc-windows-msvc"),
    ("Windows", "aarch64-pc-windows-msvc"),
]

LICENCE_FILE = re.compile(r"^(licen[cs]e|copying|copyright|notice|unlicense)", re.I)

MIT_BODY = """Permission is hereby granted, free of charge, to any person obtaining a copy of this software
and associated documentation files (the "Software"), to deal in the Software without
restriction, including without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or
substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE."""

# Crates under the MIT License alone whose packages carry no licence file: objc2's repository
# (https://github.com/madsmtm/objc2) says in its LICENSE.md that objc2, block2, objc2-foundation
# and objc2-encode are MIT-licensed, and its LICENSE-MIT.txt names Mads Marquart, their author.
_OBJC2 = ("none in the package; the MIT License, with the copyright line of objc2's repository",
          "MIT License\n\nCopyright (c) Mads Marquart\n\n" + MIT_BODY)
NO_FILE = {"objc2": _OBJC2, "block2": _OBJC2, "objc2-encode": _OBJC2, "objc2-foundation": _OBJC2}


# The exit status when cargo, offline, lacks a crate or an index entry the wheels' targets need.
OFFLINE_UNAVAILABLE = 3

# What cargo says when --offline stopped it: a crate's package it would have to download ("but
# --offline was specified"), or a package the index it has does not list (cargo 1.90: "As a reminder,
# you're using offline mode"; cargo 1.99, as on the CI runners: "note: offline mode (via `--offline`)").
# A stale Cargo.lock under --locked says "--offline" too (it suggests using it), so the match is
# on these phrases, not on the word.
_OFFLINE_STOP = re.compile(r"--offline was specified|you're using offline mode|offline mode \(via `--offline`\)")


def offline_stop(stderr):
    """Whether cargo's error is that it could not get, offline, what it needs."""
    return _OFFLINE_STOP.search(stderr) is not None


def cargo(args, online):
    cmd = ["cargo"] + args + ["--locked"] + ([] if online else ["--offline"])
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        msg = "third_party_crates: `{}` failed:\n{}".format(" ".join(cmd), r.stderr)
        if not online and offline_stop(r.stderr):
            sys.stderr.write(msg + "cargo's registry lacks crates the wheels' targets need: run with --online to fetch them\n")
            sys.exit(OFFLINE_UNAVAILABLE)
        sys.exit(msg)
    return r.stdout


def features():
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    m = re.search(r"\[tool\.maturin\][^\[]*?features\s*=\s*\[([^\]]*)\]", text, re.S)
    return ",".join(re.findall(r'"([^"]+)"', m.group(1)))


def lock_packages():
    """(name, version) of every package in Cargo.lock."""
    text = (ROOT / "Cargo.lock").read_text(encoding="utf-8")
    return re.findall(r'\[\[package\]\]\nname = "([^"]+)"\nversion = "([^"]+)"', text)


def linked(online):
    """{(name, version): (proc_macro, set of targets)} from cargo tree, per wheel target."""
    out = {}
    feats = features()
    for _, target in TARGETS:
        text = cargo(["tree", "-e", "normal", "--features", feats, "--target", target, "--prefix", "none",
                      "--format", "{p}"], online)
        for line in text.splitlines():
            m = re.match(r"^(\S+) v(\S+)( \(proc-macro\))?", line)
            if not m or line.find(" (/") >= 0 or line.find(" (" + str(ROOT)) >= 0:
                continue
            key = (m.group(1), m.group(2))
            pm, ts = out.get(key, (False, set()))
            out[key] = (pm or bool(m.group(3)), ts | {target})
    return out


def wheels(ts):
    if len(ts) == len(TARGETS):
        return "all"
    parts = []
    for platform in ("Linux", "macOS", "Windows"):
        mine = [t for p, t in TARGETS if p == platform]
        got = [t for t in mine if t in ts]
        if got == mine:
            parts.append(platform)
        else:
            parts.extend("`{}`".format(t) for t in got)
    return ", ".join(parts)


def norm(text):
    lines = [line.rstrip() for line in text.replace("\r\n", "\n").replace("\r", "\n").expandtabs(4).split("\n")]
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines)


def allows_apache(expr):
    return re.search(r"\bApache-2\.0\b", expr) is not None and " AND " not in expr.replace("(", "").upper()


def build(online=False):
    meta = json.loads(cargo(["metadata", "--format-version", "1", "--features", features()], online))
    pkgs = {(p["name"], p["version"]): p for p in meta["packages"]}
    link = linked(online)
    rows, texts, order = [], {}, []

    def add(text, label):
        # Texts that differ only in white space (line breaks, indentation) are one text.
        t = norm(text)
        k = " ".join(t.split())
        if k not in texts:
            texts[k] = (t, [])
            order.append(k)
        texts[k][1].append(label)

    for key in sorted(link, key=lambda k: (k[0].lower(), [int(x) if x.isdigit() else x for x in re.split(r"[.+-]", k[1])])):
        name, version = key
        p = pkgs[key]
        expr = (p.get("license") or "").replace("/", " OR ")
        pm, ts = link[key]
        d = Path(p["manifest_path"]).parent
        files = sorted(f for f in os.listdir(d) if LICENCE_FILE.match(f) and (d / f).is_file())
        how = ", ".join("`{}`".format(f) for f in files)
        for f in files:
            add((d / f).read_text(encoding="utf-8", errors="replace"), "{} {} (`{}`)".format(name, version, f))
        if not files:
            if name in NO_FILE:
                how, text = NO_FILE[name]
                add(text, "{} {}".format(name, version))
            elif allows_apache(expr):
                how = "none in the package; taken under Apache-2.0 (`LICENSES/Apache-2.0.txt`)"
            else:
                sys.exit("third_party_crates: {} {} ({}) carries no licence file: add it to NO_FILE".format(name, version, expr))
        source = p.get("repository") or "https://crates.io/crates/{}".format(name)
        rows.append("| {}{} | {} | {} | {} | {} | {} |".format(
            name, " (procedural macro)" if pm else "", version, expr, wheels(ts), source, how))

    lock = set(lock_packages())
    root = {(p["name"], p["version"]) for p in meta["packages"] if p["source"] is None}
    unlinked = sorted(lock - set(link) - root, key=lambda k: (k[0].lower(), k[1]))

    out = []
    w = out.append
    w("# Rust crates in the compiled module")
    w("")
    w("<!-- Written by tools/third_party_crates.py from Cargo.lock and the crates' packages; do not edit. -->")
    w("")
    w("Each wheel's compiled module (`stabilizer_qec/_core`, built with the features `{}`) is built".format(features()))
    w("from the Rust crates below, as `cargo tree -e normal` lists them for the wheel's target; every")
    w("wheel carries this one file, and the Wheels column says which link each crate (Linux: glibc and")
    w("musl, x86_64 and aarch64; macOS: x86_64 and arm64; Windows: x64 and arm64). Procedural macros run")
    w("while the module is compiled and are not linked into it; they are listed too. Their licence")
    w("texts and notices follow, as each crate's package carries them, each text once with the crates")
    w("that carry it (texts that differ only in white space count as one). HiGHS, the C++ solver `build.rs` compiles in, and the code ported into this")
    w("package are in `THIRD_PARTY_NOTICES.md`; the Apache License 2.0 is in `LICENSES/Apache-2.0.txt`.")
    w("")
    w("## Crates")
    w("")
    w("| Crate | Version | Licence | Wheels | Source | Licence files |")
    w("|---|---|---|---|---|---|")
    out.extend(rows)
    w("")
    w("## Licence texts and notices")
    for i, k in enumerate(order, 1):
        t, who = texts[k]
        w("")
        w("### Text {}".format(i))
        w("")
        w("Carried by: " + "; ".join(who) + ".")
        w("")
        fence = "~~~~" if "```" in t else "```"
        w(fence + "text")
        w(t)
        w(fence)
    w("")
    w("## Crates of Cargo.lock that no wheel links")
    w("")
    w("Build scripts' tools, development dependencies and crates for other targets:")
    w("")
    for name, version in unlinked:
        w("- {} {}".format(name, version))
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true", help="fail if THIRD_PARTY_CRATES.md differs")
    ap.add_argument("--online", action="store_true", help="let cargo fetch crates it has not got")
    a = ap.parse_args()
    text = build(a.online)
    if a.check:
        if not OUT.exists() or OUT.read_text(encoding="utf-8") != text:
            sys.exit("THIRD_PARTY_CRATES.md is out of date: run python tools/third_party_crates.py")
        print("THIRD_PARTY_CRATES.md is up to date")
        return
    OUT.write_text(text, encoding="utf-8")
    print("wrote {} ({} bytes)".format(OUT.relative_to(ROOT), len(text.encode("utf-8"))))


if __name__ == "__main__":
    main()

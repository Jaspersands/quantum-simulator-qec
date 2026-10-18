"""Write python/stabilizer_qec/_stim_help.json.gz: Stim's help pages (its documentation of each
instruction, result format and command, from Stim 1.16, Apache-2.0; see THIRD_PARTY_NOTICES.md),
which ``stabilizer-qec help <topic>`` prints.

    python tools/stim_help.py --stim path/to/stim            # rewrite the file
    python tools/stim_help.py --stim path/to/stim --check    # fail if it differs

``--stim`` is Stim's own command line (``python -m stim`` of the stim package; a deltakit_stim
install shadows the ``stim`` script). The pages are kept as Stim prints them; the command
pages' ``stim <command>`` reads ``stabilizer-qec <command>`` when printed.
"""

from __future__ import annotations

import argparse
import gzip
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "python" / "stabilizer_qec" / "_stim_help.json.gz"
FORMATS = ["01", "b8", "dets", "hits", "ptb64", "r8"]
INDEXES = ["gates", "formats", "gates_markdown", "formats_markdown", "commands_markdown"]


def build(stim: list) -> bytes:
    def page(topic: str) -> str:
        r = subprocess.run(stim + ["help", topic], capture_output=True, text=True, check=True)
        return r.stdout

    commands = [line.split()[1] for line in page("commands").splitlines() if line.startswith("    stim ")]
    gate_names = sorted({line.strip() for line in page("gates").splitlines() if line.startswith("    ")})
    gates: dict = {}
    aliases: dict = {}
    for name in gate_names:
        text = page(name)
        canonical = text.split("'", 2)[1]  # "### The 'CX' Gate"
        if canonical in gates and gates[canonical] != text:
            raise SystemExit(f"help {name} differs from help {canonical}")
        gates[canonical] = text
        if name != canonical:
            aliases[name] = canonical
    data = {
        "commands": {c: page(c) for c in commands},
        "formats": {f: page(f) for f in FORMATS},
        "gates": gates,
        "aliases": aliases,
        "indexes": {t: page(t) for t in INDEXES},
    }
    raw = json.dumps(data, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    return gzip.compress(raw, compresslevel=9, mtime=0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stim", required=True, help="Stim's command line (a path, or a command with arguments)")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    blob = build(a.stim.split())
    if a.check:
        old = json.loads(gzip.decompress(OUT.read_bytes()))
        if old != json.loads(gzip.decompress(blob)):
            raise SystemExit(f"{OUT} differs from Stim's help; rerun without --check")
        print(f"{OUT.name} matches")
        return
    OUT.write_bytes(blob)
    print(f"wrote {OUT} ({len(blob)} bytes)", file=sys.stderr)


if __name__ == "__main__":
    main()

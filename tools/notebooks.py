"""The tutorials: each written as a Python file in the percent format (`tutorials/*.py`, a
`# %%` line starting each code cell, `# %% [markdown]` each text cell, the text as comments),
built into a notebook with its outputs (`tutorials/*.ipynb`) by running it.

    python tools/notebooks.py            # run every tutorial, write its notebook
    python tools/notebooks.py --check    # run every tutorial; fail if one raises, or if a
                                         # notebook's cells are not its source's

No Jupyter is needed: cells run in one namespace, as a kernel runs them; what they print is
kept, the last expression's value is shown (an SVG diagram as a picture), and each matplotlib
figure becomes a PNG. Needs the package, numpy and matplotlib; Stim and PyMatching where a
tutorial compares with them.
"""

from __future__ import annotations

import argparse
import ast
import base64
import contextlib
import io
import json
import pathlib
import sys
import time
import traceback

ROOT = pathlib.Path(__file__).resolve().parent.parent
TUTORIALS = ROOT / "tutorials"


def cells_of(source: str) -> list:
    """[(kind, text)]: the percent format's cells, markdown cells without their comment marks."""
    cells, kind, lines = [], None, []

    def flush():
        if kind is not None:
            text = "\n".join(lines).strip("\n")
            if kind == "markdown":
                text = "\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in text.splitlines())
            cells.append((kind, text))

    for line in source.splitlines():
        if line.startswith("# %%"):
            flush()
            kind, lines = ("markdown" if "[markdown]" in line else "code"), []
        elif kind is not None:
            lines.append(line)
    flush()
    return cells


def run_cell(code: str, env: dict) -> list:
    """Run one cell; its outputs, as a notebook keeps them."""
    import matplotlib.pyplot as plt

    outputs, out = [], io.StringIO()
    tree = ast.parse(code)
    last = tree.body.pop() if tree.body and isinstance(tree.body[-1], ast.Expr) else None
    with contextlib.redirect_stdout(out):
        exec(compile(tree, "<cell>", "exec"), env)
        value = eval(compile(ast.Expression(last.value), "<cell>", "eval"), env) if last else None
    if out.getvalue():
        outputs.append({"output_type": "stream", "name": "stdout", "text": out.getvalue().splitlines(keepends=True)})
    for num in plt.get_fignums():
        buf = io.BytesIO()
        plt.figure(num).savefig(buf, format="png", dpi=90, bbox_inches="tight")
        png = base64.b64encode(buf.getvalue()).decode()
        outputs.append({"output_type": "display_data", "data": {"image/png": png, "text/plain": ["<Figure>"]}, "metadata": {}})
    plt.close("all")
    if value is not None:
        data = {"text/plain": repr(value).splitlines(keepends=True)}
        svg = getattr(value, "_repr_svg_", lambda: None)()
        if svg:
            data["image/svg+xml"] = svg.splitlines(keepends=True)
        outputs.append({"output_type": "execute_result", "data": data, "metadata": {}, "execution_count": None})
    return outputs


def notebook(cells: list, outputs: list) -> dict:
    nb_cells, count = [], 0
    for (kind, text), outs in zip(cells, outputs):
        src = text.splitlines(keepends=True)
        if kind == "markdown":
            nb_cells.append({"cell_type": "markdown", "metadata": {}, "source": src})
        else:
            count += 1
            for o in outs:
                if o["output_type"] == "execute_result":
                    o["execution_count"] = count
            nb_cells.append({"cell_type": "code", "execution_count": count, "metadata": {}, "outputs": outs, "source": src})
    return {
        "cells": nb_cells,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def run(path: pathlib.Path) -> tuple:
    import matplotlib

    matplotlib.use("Agg")
    cells = cells_of(path.read_text(encoding="utf-8"))
    env = {"__name__": "__main__"}
    outputs = []
    for kind, text in cells:
        outputs.append(run_cell(text, env) if kind == "code" else [])
    return cells, outputs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="run, and compare the notebooks' cells with their sources")
    ap.add_argument("names", nargs="*", help="tutorials to run (default: all)")
    args = ap.parse_args()
    sources = sorted(TUTORIALS.glob("*.py"))
    if args.names:
        sources = [s for s in sources if any(n in s.name for n in args.names)]
    bad = 0
    for src in sources:
        t0 = time.perf_counter()
        try:
            cells, outputs = run(src)
        except Exception:
            traceback.print_exc()
            print(f"FAIL {src.name}: a cell raised")
            bad += 1
            continue
        nb_path = src.with_suffix(".ipynb")
        if args.check:
            have = json.loads(nb_path.read_text(encoding="utf-8")) if nb_path.exists() else {"cells": []}
            mine = [("".join(c["source"]), c["cell_type"]) for c in have["cells"]]
            want = [(text, kind) for kind, text in cells]
            if mine != want:
                print(f"FAIL {src.name}: {nb_path.name} is not built from it (run tools/notebooks.py)")
                bad += 1
                continue
        else:
            nb_path.write_text(json.dumps(notebook(cells, outputs), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"ok   {src.name} ({time.perf_counter() - t0:.1f} s)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

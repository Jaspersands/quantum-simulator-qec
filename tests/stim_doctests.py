"""Stim's own docstring examples, run against this package (``stim`` replaced by
``stabilizer_qec``): the conformance suite for the Stim-compatible API.

Used by tests/test_stim_doctests.py, and runnable directly for a report:

    python tests/stim_doctests.py [Class ...]
"""

from __future__ import annotations

import doctest
import inspect
import re
import sys
from typing import Dict, Iterator, List, Tuple

MEMBERS = ("__init__", "__call__", "__add__", "__iadd__", "__mul__", "__imul__", "__rmul__", "__pow__", "__ipow__", "__truediv__", "__itruediv__", "__neg__", "__pos__", "__len__", "__getitem__", "__setitem__", "__iter__", "__eq__", "__ne__", "__str__", "__repr__")


def translate(text: str) -> str:
    text = re.sub(r"\bimport stim\b", "import stabilizer_qec", text)
    return re.sub(r"\bstim\.", "stabilizer_qec.", text)


def examples(stim) -> Iterator[Tuple[str, doctest.DocTest]]:
    """(qualified name, translated doctest) for every Stim object and member with examples."""
    parser = doctest.DocTestParser()
    for name in sorted(dir(stim)):
        if name.startswith("_"):
            continue
        obj = getattr(stim, name)
        targets = [(name, obj)]
        if inspect.isclass(obj):
            for a in sorted(dir(obj)):
                if a.startswith("_") and a not in MEMBERS:
                    continue
                try:
                    targets.append((f"{name}.{a}", getattr(obj, a)))
                except Exception:
                    continue
        for qual, o in targets:
            doc = getattr(o, "__doc__", None)
            if not doc or ">>>" not in doc:
                continue
            if qual != name and doc == getattr(obj, "__doc__", None):
                continue
            yield qual, parser.get_doctest(translate(doc), {}, qual, None, 0)


def run(only: List[str] = (), verbose: bool = False) -> Dict[str, Tuple[int, int]]:
    import numpy as np
    import stim
    import stabilizer_qec

    results: Dict[str, Tuple[int, int]] = {}
    for qual, test in examples(stim):
        cls = qual.split(".")[0]
        if only and cls not in only and qual not in only:
            continue
        test.globs = {"stabilizer_qec": stabilizer_qec, "np": np, "numpy": np}
        runner = doctest.DocTestRunner(optionflags=doctest.NORMALIZE_WHITESPACE, verbose=False)
        out: List[str] = []
        runner.run(test, out=out.append)
        r = runner.summarize(verbose=False)
        results[qual] = (r.failed, r.attempted)
        if verbose and r.failed:
            print("".join(out)[:3000])
    return results


if __name__ == "__main__":
    only = sys.argv[1:]
    res = run(only, verbose="-v" in only)
    by_class: Dict[str, List[int]] = {}
    for qual, (f, a) in res.items():
        c = by_class.setdefault(qual.split(".")[0], [0, 0, 0])
        c[0] += a - f
        c[1] += a
        c[2] += f == 0
    for c, (ok, total, members_ok) in sorted(by_class.items()):
        print(f"{c:45s} {ok:4d}/{total:4d}")
    print("failing members:", sorted(q for q, (f, a) in res.items() if f))

"""Every result the package computes, against the fingerprints recorded in
data/fingerprints.json (tools/fingerprints.py): the same shots for a seed at any thread count
and the same predictions on every platform, and the same model text on each (recorded per
platform, as Stim's own text differs between them in a last digit). A change to any of them is a
change to a result, which work on speed must never make."""

from __future__ import annotations

import json
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import fingerprints  # noqa: E402

RECORDED = json.loads((ROOT / "data" / "fingerprints.json").read_text(encoding="utf-8"))
CASES = fingerprints.cases()


def test_every_case_is_recorded():
    assert sorted(CASES) == sorted(RECORDED)


@pytest.mark.parametrize("name", sorted(RECORDED))
def test_fingerprint(name):
    got = CASES[name]()
    want = fingerprints.expected(RECORDED, name)
    # A platform with no record fails too, naming what to record (tools/fingerprints.py --add).
    assert got == want, f"{name} on {fingerprints.PLATFORM}: {got} (recorded: {want})"

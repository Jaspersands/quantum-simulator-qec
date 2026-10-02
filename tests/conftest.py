"""Shared fixtures. The reference packages (stim, pymatching, ldpc, beliefmatching) are
optional: tests that compare against one skip where it does not install."""

from __future__ import annotations

import collections
import re

import pytest

import stabilizer_qec as sq


@pytest.fixture(scope="session")
def d5():
    """A d = 5 rotated SD6 memory over 5 rounds at p = 0.4%."""
    return sq.memory_circuit(distance=5, rounds=5, p=0.004)


@pytest.fixture(scope="session")
def d3():
    return sq.memory_circuit(distance=3, rounds=3, p=0.01)


def canonical_dem(text: str) -> dict:
    """A model's faults keyed by their (sorted) pieces and tag, faults with the same pieces
    and tag merged, so models that list the same faults in other orders or other groupings
    compare equal."""
    merged: dict = collections.defaultdict(float)
    for line in text.splitlines():
        m = re.fullmatch(r"error(?:\[([^\]]*)\])?\(([^)]*)\)(.*)", line.strip())
        if not m:
            continue
        tag, p, targets = m.group(1) or "", float(m.group(2)), m.group(3)
        pieces = tuple(sorted(tuple(sorted(x.split())) for x in targets.split("^")))
        key = (tag, pieces) if tag else pieces
        q = merged[key]
        merged[key] = q * (1 - p) + p * (1 - q)
    return dict(merged)


def assert_same_model(ours: str, theirs: str) -> None:
    a, b = canonical_dem(ours), canonical_dem(theirs)
    assert set(a) == set(b)
    for k in a:
        assert a[k] == pytest.approx(b[k], rel=1e-9, abs=1e-15), k

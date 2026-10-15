"""Shared fixtures. The reference packages (stim, pymatching, ldpc, beliefmatching) are
optional: tests that compare against one skip where it does not install."""

from __future__ import annotations

import collections
import os
import re

# Plots render off screen (a CI runner's Tk may be unusable).
os.environ.setdefault("MPLBACKEND", "Agg")

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

# The references the differential tests reproduce, at the versions they reproduce: another
# version (pip picks an older one where the newer has no wheel for the platform) is skipped.
REFERENCE_VERSIONS = {"fusion_blossom": ("fusion-blossom", "0.2.13"), "relay_bp": ("relay-bp", "0.2.2"), "tesseract_decoder": ("tesseract-decoder", "0.1.1.dev20261013012529"), "chromobius": ("chromobius", "1.1.1")}


def reference(module: str):
    """The reference module, or a skip when it is missing or another version."""
    import importlib.metadata

    found = pytest.importorskip(module)
    dist, want = REFERENCE_VERSIONS[module]
    have = importlib.metadata.version(dist)
    if have != want:
        pytest.skip(f"{dist} {have} is installed; the test reproduces {want}")
    return found


# ldpc's own tests (tests/ldpc_port) need scipy, as stabilizer_qec.ldpc does, and so do Tesseract's
# GARI tests: without it they are not collected.
import importlib.util as _importlib_util  # noqa: E402

collect_ignore_glob = [] if _importlib_util.find_spec("scipy") else ["ldpc_port/test_*.py", "tesseract_port/test_tesseract_gari.py", "tesseract_port/test_tesseract_demutil.py"]

# qLDPC's own tests (tests/qldpc_port) need what stabilizer_qec.qldpc does: Python 3.11 and the
# `qldpc` extra.
import sys as _sys  # noqa: E402

if _sys.version_info < (3, 11) or not all(_importlib_util.find_spec(m) for m in ("scipy", "galois", "sympy", "cvxpy", "networkx", "diskcache", "platformdirs")):
    collect_ignore_glob += ["qldpc_port", "qldpc_port/*"]

# qecsim's own tests (tests/qecsim_port) need the `qecsim` extra.
if not all(_importlib_util.find_spec(m) for m in ("scipy", "networkx", "mpmath", "click")):
    collect_ignore_glob += ["qecsim_port", "qecsim_port/*"]

# MQT QECC's own tests (tests/mqt_qecc_port) need the `mqt` extra, and Python 3.10.
if _sys.version_info < (3, 10) or not all(_importlib_util.find_spec(m) for m in ("scipy", "networkx", "mpmath", "click", "qiskit", "z3", "numba", "matplotlib", "multiprocess", "tqdm")):
    collect_ignore_glob += ["mqt_qecc_port", "mqt_qecc_port/*"]

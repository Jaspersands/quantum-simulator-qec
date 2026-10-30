"""tools/bp_check.py compares our BP posteriors with ldpc's; a NaN posterior must fail the comparison,
not read as a zero difference."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("ldpc")
pytest.importorskip("beliefmatching")
pytest.importorskip("pymatching")
pytest.importorskip("stim")
from scipy.sparse import csc_matrix  # noqa: E402


@pytest.fixture
def tool():
    spec = importlib.util.spec_from_file_location("bp_check", Path(__file__).resolve().parents[1] / "tools" / "bp_check.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def small_code():
    h = csc_matrix(np.array([[1, 1, 0, 1], [0, 1, 1, 1]], dtype=np.uint8))
    priors = np.full(4, 0.05)
    syndromes = np.array([[1, 0], [0, 1], [1, 1]], dtype=np.uint8)
    return h, priors, syndromes


def test_compare_bp_passes_on_our_own_decoder(tool):
    h, priors, syndromes = small_code()
    w = tool.compare_bp(h, priors, syndromes, "product_sum", 1.0, 5)
    assert w["cases"] == 3 and w["hard"] == 0 and w["converge"] == 0 and w["iterations"] == 0
    assert w["llr_rel"] < 1e-9


def test_nan_posteriors_are_not_read_as_zero_difference(tool, monkeypatch):
    h, priors, syndromes = small_code()
    decode = tool.sq.bp_decode

    def nan_posteriors(nchecks, cols, pri, s, iters, method, scale):
        hard, llr, conv, it = decode(nchecks, cols, pri, s, iters, method, scale)
        return hard, [math.nan] * len(llr), conv, it

    monkeypatch.setattr(tool.sq, "bp_decode", nan_posteriors)
    w = tool.compare_bp(h, priors, syndromes, "product_sum", 1.0, 5)
    assert math.isnan(w["llr_abs"]) and math.isnan(w["llr_rel"])
    # The check's pass condition is llr_rel < 1e-9, which a NaN does not satisfy.
    assert not (w["llr_rel"] < 1e-9)

"""False-positive controls for the native-platform reference assertions."""

import json
import math
from pathlib import Path
import sys

import pytest

import stabilizer_qec as sq
import tesseract_platform_helpers
from tesseract_platform_helpers import assert_dem_text, assert_search_log


def test_dem_comparison_only_allows_last_bit_numeric_changes(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    expected = "error(0.1) D0 L70\ndetector(1, 2) D0\n"
    assert_dem_text(expected.replace("0.1", "0.1000000000000000056"), expected)
    assert_dem_text(expected.replace("0.1", repr(math.nextafter(0.1, 1))), expected)
    for actual in [expected.replace("0.1", "0.11"), expected.replace("D0", "D1"), expected.replace("L70", "L6"),
                   expected.replace("1, 2", "1, 3"), expected.replace("error(", "error[tag]("), expected.rstrip(),
                   "\n".join(reversed(expected.splitlines())) + "\n"]:
        with pytest.raises(AssertionError):
            assert_dem_text(actual, expected)
    with pytest.raises(AssertionError):
        assert_dem_text("error(1e-300) D0", "error(0) D0")
    # On Apple silicon (16 digits, as the references were recorded), exactly.
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(tesseract_platform_helpers, "STIM_DIGITS", 16)
    with pytest.raises(AssertionError):
        assert_dem_text(expected.replace("0.1", "0.1000000000000000056"), expected)


def test_search_log_rejects_corrupted_chains_costs_masks_and_structure():
    v = json.loads((Path(__file__).parent / "data/tesseract_cli_reference.json").read_text())["verbose"]
    log = "len(pq)" + v["log"].split("len(pq)", 1)[1]
    dem = sq.DetectorErrorModel(v["dem"])
    assert_search_log(log, log, dem, [0, 2])
    for actual in [log.replace("activated_detectors = 0, 2, ", "activated_detectors = 1, 2, ", 1),
                   log.replace("activated_errors = 1, 2, ", "activated_errors = 0, 2, "),
                   log.replace("Decoding complete. Cost: 3.583518938456", "Decoding complete. Cost: 4"),
                   log.replace("obs_mask [1]", "obs_mask [0]"), log.replace("low confidence 0", "low confidence 1"),
                   log.replace("num_dets = 2", "num_dets = 1", 1), "\n".join(log.splitlines()[:-1]) + "\n", log + "unknown\n"]:
        with pytest.raises(AssertionError):
            assert_search_log(actual, log, dem, [0, 2])

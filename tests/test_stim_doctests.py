"""Every example in Stim's own docstrings, run against this package with ``stim`` replaced by
``stabilizer_qec`` (tests/stim_doctests.py). Each class listed in DONE must pass all of its
examples; the 2.0 plan adds a class here as its task lands, until every Stim object is listed.
"""

from __future__ import annotations

import pathlib
import sys

import pytest

stim = pytest.importorskip("stim")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import stim_doctests  # noqa: E402

DONE = [
    "Flow",
    "target_separator",
    "target_relative_detector_id",
    "target_logical_observable_id",
    "DemTarget",
    "DemRepeatBlock",
    "DemInstruction",
    "CircuitInstruction",
    "CircuitRepeatBlock",
    "GateData",
    "GateTarget",
    "PauliString",
    "PauliStringIterator",
    "Tableau",
    "TableauIterator",
    "TableauSimulator",
    "gate_data",
    "target_combined_paulis",
    "target_combiner",
    "target_inv",
    "target_pauli",
    "target_rec",
    "target_sweep_bit",
    "target_x",
    "target_y",
    "target_z",
]

RESULTS = stim_doctests.run(DONE)


@pytest.mark.parametrize("member", sorted(RESULTS))
def test_stims_examples_pass(member):
    failed, attempted = RESULTS[member]
    assert failed == 0, f"{failed} of {attempted} of Stim's examples for {member} fail (python tests/stim_doctests.py {member} -v)"


def test_every_done_class_has_examples():
    seen = {m.split(".")[0] for m in RESULTS}
    assert seen == set(DONE)

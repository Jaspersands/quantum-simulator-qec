"""Every example in Stim's own docstrings, run against this package with ``stim`` replaced by
``stabilizer_qec`` (tests/stim_doctests.py): every public name Stim exports, each of its
examples. ``EXCLUDED`` lists any example that is not expected to pass, with the reason; it is
empty: all of them pass.
"""

from __future__ import annotations

import pathlib
import sys

import pytest

stim = pytest.importorskip("stim", minversion="1.16")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import stim_doctests  # noqa: E402

DONE = sorted(name for name in dir(stim) if not name.startswith("_"))

# member -> why its examples are not expected to pass. None today.
EXCLUDED: dict = {}

RESULTS = stim_doctests.run(DONE)


@pytest.mark.parametrize("member", sorted(RESULTS))
def test_stims_examples_pass(member):
    if member in EXCLUDED:
        pytest.skip(EXCLUDED[member])
    failed, attempted = RESULTS[member]
    assert failed == 0, f"{failed} of {attempted} of Stim's examples for {member} fail (python tests/stim_doctests.py {member} -v)"


def test_the_examples_are_all_there():
    seen = {m.split(".")[0] for m in RESULTS}
    assert seen <= set(DONE)
    # Stim 1.16 has some 2,200 examples across 44 of its names; a much smaller count means
    # the extraction broke.
    assert sum(attempted for _, attempted in RESULTS.values()) > 2000 and len(seen) >= 40

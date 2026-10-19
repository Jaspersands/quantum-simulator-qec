"""Every example in the package's docstrings runs and prints what it says."""

from __future__ import annotations

import doctest

import pytest

import stabilizer_qec
from stabilizer_qec import _circuit, _codes, _decoders, _density, _perturb, _predecoder, _process, clifford, estimate, realtime, surgery
from stabilizer_qec.estimate import azure
from stabilizer_qec.tsim import circuit as tsim_circuit


@pytest.mark.parametrize("module", [stabilizer_qec, _circuit, _codes, _decoders, _density, _perturb, _predecoder, _process, clifford, estimate, azure, realtime, surgery, tsim_circuit], ids=lambda m: m.__name__)
def test_docstring_examples(module):
    result = doctest.testmod(module, optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE)
    assert result.failed == 0
    if module in (stabilizer_qec, _circuit):
        assert result.attempted > 0, "the guide's examples did not run"

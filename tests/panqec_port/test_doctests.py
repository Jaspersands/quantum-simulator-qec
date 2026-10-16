# stabilizer_qec: PanQEC runs its modules' docstring examples (pytest --doctest-modules over
# panqec/); these are the modules that have them.
import doctest
import importlib

import pytest


@pytest.mark.parametrize("name", ["stabilizer_qec.panqec.analysis", "stabilizer_qec.panqec.utils"])
def test_docstring_examples(name):
    result = doctest.testmod(importlib.import_module(name), optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE)
    assert result.attempted > 0 and result.failed == 0

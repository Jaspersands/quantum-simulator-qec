"""sinter's docstring examples, in the ported modules (``stabilizer_qec.sinter``)."""

import doctest
import importlib
import pkgutil

import pytest

import stabilizer_qec.sinter as sinter

MODULES = sorted(m.name for m in pkgutil.walk_packages(sinter.__path__, "stabilizer_qec.sinter.") if not m.name.endswith("__main__"))


@pytest.mark.parametrize("name", MODULES)
def test_examples_pass(name):
    module = importlib.import_module(name)
    result = doctest.testmod(module, optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE)
    assert result.failed == 0, f"{result.failed} of {result.attempted} examples in {name} fail"

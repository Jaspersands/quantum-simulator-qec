"""Every example in the package's docstrings runs and prints what it says: the guide's, and
every one the API reference (``api/``, built by tools/api_docs.py) publishes."""

from __future__ import annotations

import doctest
import importlib
import importlib.util
from pathlib import Path

import pytest

import stabilizer_qec
from stabilizer_qec import _circuit, _codes, _decoders, _density, _matching, _perturb, _predecoder, _process, _stim, clifford, estimate, realtime, surgery
from stabilizer_qec.estimate import azure, azure_params, azure_report, qualtran
from stabilizer_qec.estimate.qualtran import ui as qualtran_ui
from stabilizer_qec.tsim import circuit as tsim_circuit

API = Path(__file__).resolve().parent.parent / "api"

# The modules written here whose examples are the guide's (a check that does not need api/).
_GUIDE = [stabilizer_qec, _circuit, _codes, _decoders, _density, _matching, _perturb, _predecoder, _process, _stim, clifford, estimate, azure, azure_params, azure_report, qualtran, qualtran_ui, realtime, surgery, tsim_circuit]


def _module(name: str):
    """A page's module: imported by name, or (a module object a package holds under that name,
    such as cudaq_qec's ``phase_name``) read from its package."""
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError:
        parent, _, last = name.rpartition(".")
        found = getattr(importlib.import_module(parent), last, None)
        return found if type(found).__name__ == "module" else None


def _published() -> list:
    """Every module whose docstrings the API reference publishes: each module with a page in
    api/, and each module defining a public member of one (``Matching`` is documented on the
    package's page, its docstring written in ``stabilizer_qec._matching``)."""
    if not API.is_dir():
        return []
    found = {}
    for page in sorted(API.rglob("*.html")):
        if page.name == "index.html":
            continue
        module = _module(".".join(page.relative_to(API).with_suffix("").parts))
        if module is None:
            continue
        found[module.__name__] = module
        public = getattr(module, "__all__", None) or [n for n in vars(module) if not n.startswith("_")]
        for name in public:
            owner = getattr(getattr(module, name, None), "__module__", None)
            if isinstance(owner, str) and owner.split(".")[0] == "stabilizer_qec" and owner not in found:
                found[owner] = importlib.import_module(owner)
    return [found[k] for k in sorted(found)]


_PUBLISHED = _published()


# Examples that need scipy, as their docstrings say: without it they are not run.
_NEEDS_SCIPY = {"stabilizer_qec._circuit.DetectorErrorModel.distance"}


def _run(module) -> doctest.TestResults:
    flags = doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE
    if importlib.util.find_spec("scipy"):
        return doctest.testmod(module, optionflags=flags)
    runner = doctest.DocTestRunner(optionflags=flags)
    for test in doctest.DocTestFinder().find(module):
        if test.name not in _NEEDS_SCIPY:
            runner.run(test)
    return runner.summarize(verbose=False)


@pytest.mark.parametrize("module", _GUIDE, ids=lambda m: m.__name__)
def test_docstring_examples(module):
    result = _run(module)
    assert result.failed == 0
    if module in (stabilizer_qec, _circuit, _matching, _stim):
        assert result.attempted > 0, "the guide's examples did not run"


@pytest.mark.parametrize("module", [m for m in _PUBLISHED if m not in _GUIDE], ids=lambda m: m.__name__)
def test_published_docstring_examples(module):
    assert _run(module).failed == 0


def test_the_published_modules_are_found():
    if not API.is_dir():
        pytest.skip("no api/ here (an installed package's tests)")
    names = {m.__name__ for m in _PUBLISHED}
    # The class examples documented on the package's page, and the ports' published examples.
    assert {"stabilizer_qec._matching", "stabilizer_qec._stim", "stabilizer_qec.analysis._dk._lambda", "stabilizer_qec.analysis._dk._leppr", "stabilizer_qec.loom", "stabilizer_qec.sinter"} <= names
    assert len(names) > 100

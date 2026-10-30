"""The bposd port's sources compile without invalid escape sequences: a print string holding a
backslash that is not an escape must be raw or double the backslash, or the module warns on every
compile and fails outright under warnings-as-errors."""

from __future__ import annotations

import pathlib
import warnings

BPOSD = pathlib.Path(__file__).resolve().parents[1] / "python" / "stabilizer_qec" / "bposd"


def test_bposd_sources_have_no_invalid_escape_sequences():
    modules = sorted(BPOSD.glob("*.py"))
    assert modules, "no bposd sources found"
    for path in modules:
        source = path.read_text(encoding="utf-8")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            compile(source, str(path), "exec")
        escapes = [str(w.message) for w in caught if "escape" in str(w.message)]
        assert not escapes, f"{path.name}: {escapes}"

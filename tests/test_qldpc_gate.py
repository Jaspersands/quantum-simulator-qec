"""The gate that keeps qLDPC's own tests out of a partial environment (tests/conftest.py) lists every
package of the `qldpc` extra: without one of them, stabilizer_qec.qldpc does not import, and its
tests must be ignored rather than fail at collection."""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest

CONFTEST = pathlib.Path(__file__).resolve().parent / "conftest.py"


@pytest.mark.parametrize("absent", ["pyperclip", "tqdm"])
def test_qldpc_ports_are_ignored_without_each_extra_package(absent, monkeypatch):
    monkeypatch.setitem(sys.modules, absent, None)  # find_spec then reports the package absent
    spec = importlib.util.spec_from_file_location("_gate_check_conftest", CONFTEST)
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    assert "qldpc_port" in gate.collect_ignore_glob

"""tools/surgery_leakage.py's defaults: the leakage fractions are the grid the committed
data/surgery/leakage.json was made with, and a run with no --out does not overwrite that file."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def tool():
    spec = importlib.util.spec_from_file_location("surgery_leakage", ROOT / "tools" / "surgery_leakage.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_default_run_uses_the_committed_grid_and_leaves_the_committed_file(tool, monkeypatch, tmp_path):
    committed = json.loads((ROOT / "data" / "surgery" / "leakage.json").read_text(encoding="utf-8"))
    monkeypatch.setattr(tool, "ROOT", tmp_path)
    monkeypatch.setattr(tool, "run_point", lambda d, p, f, seepage, shots, seed: dict(shots=shots, fails_blind=0, fails_aware=0,
                                                                                  heralds_per_shot=0.0, sample_s=0.0, decode_s=0.0))
    monkeypatch.setattr(tool, "crossing", lambda *args, **kwargs: None)
    monkeypatch.setattr(tool, "fitted", lambda *args, **kwargs: None)
    monkeypatch.setattr(sys, "argv", ["surgery_leakage.py", "--shots", "1", "--distances", "3", "--ps", "0.002"])
    assert tool.main() == 0
    assert committed["fractions"] == [0.0, 0.03, 0.1, 0.3]
    written = sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*.json"))
    assert written == ["data/surgery/leakage-rerun.json"]
    assert json.loads((tmp_path / written[0]).read_text(encoding="utf-8"))["fractions"] == committed["fractions"]

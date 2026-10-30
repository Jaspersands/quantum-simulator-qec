"""tools/bench.py's options: a benchmark key it does not have is refused (a --check with a typo compared
nothing), and a --quick run is never recorded (README's ratios take the latest recorded run)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


@pytest.fixture
def bench(monkeypatch, tmp_path):
    spec = importlib.util.spec_from_file_location("bench", Path(__file__).resolve().parents[1] / "tools" / "bench.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "RUNS", tmp_path / "runs.json")

    def fake_run(quick, repeats, only=None):
        return dict(version="test", date="2026-10-30", machine="test", cores=1, kind="test", references={},
                    quick=quick, results={"error_model": dict(what="error model", unit="shot", ours_us=1.0, reference_us=1.0, ratio=1.0)})

    monkeypatch.setattr(module, "run", fake_run)
    return module


def test_unknown_only_key_is_refused(bench, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["bench.py", "--only", "error_modle", "--check"])
    with pytest.raises(SystemExit) as refused:
        bench.main()
    assert refused.value.code != 0
    assert "no benchmark named error_modle" in capsys.readouterr().err


def test_known_only_key_runs(bench, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["bench.py", "--only", "error_model"])
    assert bench.main() == 0


def test_quick_run_is_not_recorded(bench, monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(sys, "argv", ["bench.py", "--quick", "--record"])
    with pytest.raises(SystemExit) as refused:
        bench.main()
    assert refused.value.code != 0
    assert "not a run to record" in capsys.readouterr().err
    assert not (tmp_path / "runs.json").exists()

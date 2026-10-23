"""Capability failures must stay separate from decoder comparison failures."""

import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from types import SimpleNamespace

import pytest


@pytest.fixture
def tool():
    spec = importlib.util.spec_from_file_location("decoder_check", Path(__file__).resolve().parents[1] / "tools" / "decoder_check.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def search_cli(monkeypatch, tool, probe, run):
    monkeypatch.setattr(sys, "argv", ["decoder_check.py", "--quick", "--only", "search"])
    monkeypatch.setattr(importlib.metadata, "version", lambda package: "test")
    monkeypatch.setattr(tool, "TESSERACT_PROBE", probe)
    monkeypatch.setattr(tool, "SECTIONS", {"search": ("Search", "tesseract-decoder", run)})


@pytest.mark.skipif(os.name != "posix", reason="POSIX signal exit status")
def test_sigill_unavailable_does_not_run_comparison_or_retain_stale_result(tool, monkeypatch, tmp_path, capsys):
    def must_not_run(quick):
        pytest.fail("an unavailable reference must not run a comparison")

    search_cli(monkeypatch, tool, "import os, signal; os.kill(os.getpid(), signal.SIGILL)", must_not_run)
    # A later section must still run, and the saved result must replace an older success.
    ran = []
    tool.SECTIONS["later"] = ("Later", "later", lambda quick: (ran.append(True) or True, [{"checked": True}]))
    monkeypatch.setattr(sys, "argv", ["decoder_check.py"])
    monkeypatch.setattr(tool, "OUT", tmp_path / "check.json")
    tool.OUT.write_text(json.dumps({"sections": {"search": {"rows": [{"identical": 123}]}}}))
    assert tool.main() == 0
    assert ran == [True]
    saved = json.loads(tool.OUT.read_text())["sections"]
    assert "SIGILL" in saved["search"]["skipped"]
    assert "rows" not in saved["search"]
    assert saved["later"]["rows"] == [{"checked": True}]
    output = capsys.readouterr().out
    assert "search: skipped" in output
    assert "ALL AVAILABLE CHECKS PASSED (skipped: search)" in output
    assert "ALL CHECKS PASSED" not in output

    # The table consumer must display unavailable measurements, without reading old rows or
    # dividing absent timings. Healthy archived sections still render as before.
    spec = importlib.util.spec_from_file_location("decoder_tables", tool.ROOT / "tools" / "readme_tables.py")
    tables = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tables)
    healthy_rows = tables.decoder_checks()
    archived = json.loads((tool.ROOT / "data" / "decoders" / "check.json").read_text())
    archived["sections"]["search"] = saved["search"]
    monkeypatch.setattr(tables, "load", lambda path: archived)
    rows = tables.decoder_checks()
    assert len(rows) == 4
    assert "Skipped: reference-only capability probe terminated by SIGILL" in rows[-1]
    assert rows[-1].endswith("| — | — | — | — | — |")
    assert rows[:-1] == healthy_rows[:-1]


@pytest.mark.parametrize("code, message", [
    ("raise ValueError('reference constructor broke')", "reference constructor broke"),
    ("import sys; sys.exit(132)", "exit 132"),
])
def test_ordinary_probe_failure_is_not_unavailable(tool, monkeypatch, code, message):
    search_cli(monkeypatch, tool, code, lambda quick: pytest.fail("failed probe continued"))
    with pytest.raises(RuntimeError, match=message):
        tool.main()


@pytest.mark.skipif(os.name != "posix", reason="POSIX signal exit status")
def test_unknown_probe_crash_is_failure(tool, monkeypatch):
    search_cli(monkeypatch, tool, "import os, signal; os.kill(os.getpid(), signal.SIGTERM)", lambda quick: pytest.fail("failed probe continued"))
    with pytest.raises(RuntimeError, match=f"exit {-signal.SIGTERM}"):
        tool.main()


def test_probe_timeout_is_failure(tool, monkeypatch):
    search_cli(monkeypatch, tool, "import time; time.sleep(10)", lambda quick: pytest.fail("timed-out probe continued"))
    probe = tool.reference_probe
    monkeypatch.setattr(tool, "reference_probe", lambda code: probe(code, timeout=0.1))
    with pytest.raises(subprocess.TimeoutExpired):
        tool.main()


@pytest.mark.parametrize("error", [ValueError("product error"), RuntimeError("comparison error")])
def test_check_exception_propagates_after_healthy_probe(tool, monkeypatch, error):
    def broken(quick):
        raise error

    search_cli(monkeypatch, tool, "pass", broken)
    with pytest.raises(type(error), match=str(error)):
        tool.main()


@pytest.fixture
def live_tesseract(tool):
    try:
        importlib.metadata.version("tesseract-decoder")
    except importlib.metadata.PackageNotFoundError:
        pytest.skip("Tesseract reference is not installed")
    if not tool.reference_probe(tool.TESSERACT_PROBE):
        pytest.skip("Tesseract reference-only probe terminated by SIGILL")


def test_actual_tesseract_probe_is_healthy(live_tesseract):
    # The fixture used the real isolated import, constructor and decode; errors are failures.
    pass


def test_actual_search_mismatch_keeps_nonzero_status(tool, live_tesseract, monkeypatch, capsys):
    # Keep the real comparison code and native reference, but use a tiny model and inject a
    # wrong product fault. This checks the mismatch-to-exit-status path without a corpus run.
    circuit = tool.sq.Circuit("X_ERROR(0.1) 0\nM 0\nDETECTOR rec[-1]")
    decoder = tool.sq.SearchDecoder

    class WrongSearch:
        def __init__(self, *args, **kwargs):
            self.decoder = decoder(*args, **kwargs)

        def decode_to_faults(self, syndrome):
            return list(self.decoder.decode_to_faults(syndrome)) + [99999]

    tiny_code = SimpleNamespace(memory_circuit=lambda *args: circuit)
    monkeypatch.setattr(tool, "sq", SimpleNamespace(
        Circuit=SimpleNamespace(generated=lambda *args, **kwargs: circuit),
        CssCode=SimpleNamespace(color_code=lambda distance: tiny_code),
        BivariateBicycleCode=lambda name: tiny_code,
        SearchDecoder=WrongSearch,
    ))
    monkeypatch.setattr(sys, "argv", ["decoder_check.py", "--quick", "--only", "search"])
    assert tool.main() == 1
    output = capsys.readouterr().out
    assert "different 300" in output
    assert "SOME CHECKS FAILED" in output
    assert "skipped" not in output

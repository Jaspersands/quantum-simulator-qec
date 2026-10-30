"""tools/report.py finds the Chrome that prints the PDF: $CHROME, then the macOS app, then google-chrome
or chromium on PATH, and says so when none is there."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def tool():
    spec = importlib.util.spec_from_file_location("report", Path(__file__).resolve().parents[1] / "tools" / "report.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_chrome_env_variable_wins(tool, monkeypatch, tmp_path):
    mac = tmp_path / "Chrome"
    mac.write_text("")
    monkeypatch.setattr(tool, "MAC_CHROME", str(mac))
    monkeypatch.setenv("CHROME", "/opt/browser/chrome")
    assert tool.chrome() == "/opt/browser/chrome"


def test_macos_app_is_the_default(tool, monkeypatch, tmp_path):
    mac = tmp_path / "Chrome"
    mac.write_text("")
    monkeypatch.setattr(tool, "MAC_CHROME", str(mac))
    monkeypatch.delenv("CHROME", raising=False)
    monkeypatch.setattr(tool.shutil, "which", lambda name: "/usr/bin/google-chrome")
    assert tool.chrome() == str(mac)


def test_path_is_searched_when_there_is_no_macos_app(tool, monkeypatch, tmp_path):
    monkeypatch.setattr(tool, "MAC_CHROME", str(tmp_path / "missing"))
    monkeypatch.delenv("CHROME", raising=False)
    found = {"chromium": "/usr/bin/chromium"}
    monkeypatch.setattr(tool.shutil, "which", lambda name: found.get(name))
    assert tool.chrome() == "/usr/bin/chromium"


def test_no_chrome_says_how_to_proceed(tool, monkeypatch, tmp_path):
    monkeypatch.setattr(tool, "MAC_CHROME", str(tmp_path / "missing"))
    monkeypatch.delenv("CHROME", raising=False)
    monkeypatch.setattr(tool.shutil, "which", lambda name: None)
    with pytest.raises(SystemExit, match="--no-pdf"):
        tool.chrome()

"""Controlled CLI I/O failures: injected primary/cleanup failures and real bounded EFBIG.

The EFBIG checks assert port safety, not the unrun Tesseract binary's error policy; PyMatching's
native ofstream can silently succeed with truncated frames, which this port deliberately refuses.
"""
from __future__ import annotations

import errno
import os
import pathlib
import subprocess
import sys

import pytest

from stabilizer_qec import pymatching
from stabilizer_qec._cli import main as package_main
from stabilizer_qec.tesseract_decoder import __main__ as tesseract


def _inputs(tmp_path):
    dem, shots = tmp_path / "m.dem", tmp_path / "s.01"
    dem.write_text("error(0.1) D0 L0\n")
    shots.write_text("1\n" * 8)
    return dem, shots


def _args(kind, tmp_path):
    dem, shots = _inputs(tmp_path)
    if kind == "tesseract":
        return ["--dem", str(dem), "--in", str(shots), "--in-format", "01", "--out", str(tmp_path / "pred.01"), "--out-format", "01"]
    return ["animate", "--dem_in", str(dem), "--dets_in", str(shots), "--dets_in_format", "01", "--out_dir", str(tmp_path / "out"), "--held_frames_per_event", "0", "--held_frames_at_start", "0", "--held_frames_at_end", "1", "--max_growth_between_frames", "100", "--pixels_per_unit_length", "4", "--quiet"]


@pytest.mark.parametrize("entry", [pymatching._animate, package_main])
@pytest.mark.parametrize("stage", ["write", "close", "both"])
def test_animate_output_failure_preserves_primary(stage, entry, tmp_path, monkeypatch, capsys):
    args = _args("animate", tmp_path)
    class BrokenWriter:
        def write(self, svg):
            if stage != "close":
                raise OSError(errno.ENOSPC, "primary write failure")
        def close(self):
            if stage != "write":
                raise OSError(errno.EIO, "cleanup close failure")
    monkeypatch.setattr(pymatching, "open", lambda *a, **k: BrokenWriter(), raising=False)
    assert entry(args) == 1
    err = capsys.readouterr().err
    assert "Failed to write" in err and "Traceback" not in err
    assert ("primary write failure" if stage != "close" else "cleanup close failure") in err
    if stage == "both":
        assert "cleanup close failure" not in err


@pytest.mark.parametrize("stage", ["write", "close", "both"])
def test_tesseract_output_failure_preserves_primary(stage, tmp_path, monkeypatch, capsys):
    args = _args("tesseract", tmp_path)
    original_close = tesseract._Writer.close
    def close(writer):
        original_close(writer)
        if stage != "write":
            raise OSError(errno.EIO, "cleanup close failure")
    def write(writer, row):
        raise OSError(errno.ENOSPC, "primary write failure")
    monkeypatch.setattr(tesseract._Writer, "close", close)
    if stage != "close":
        monkeypatch.setattr(tesseract._Writer, "write", write)
    assert tesseract.main(args) == 1
    err = capsys.readouterr().err
    assert "tesseract:" in err and "Traceback" not in err
    assert ("primary write failure" if stage != "close" else "cleanup close failure") in err
    if stage == "both":
        assert "cleanup close failure" not in err


@pytest.mark.skipif(sys.platform == "win32", reason="RLIMIT_FSIZE and SIGXFSZ require POSIX")
@pytest.mark.parametrize("kind", ["animate", "package", "tesseract"])
def test_bounded_real_file_size_failure(kind, tmp_path):
    resource = pytest.importorskip("resource")
    if not hasattr(resource, "RLIMIT_FSIZE"):
        pytest.skip("platform has no RLIMIT_FSIZE")
    args = _args("tesseract" if kind == "tesseract" else "animate", tmp_path)
    # Eight tiny shots; after imports, constrain only child output files to one byte. The
    # timeout bounds execution and only this child is killed by subprocess.run on timeout.
    script = """
import resource, signal, sys
from stabilizer_qec import pymatching
from stabilizer_qec._cli import main as package_main
from stabilizer_qec.tesseract_decoder.__main__ import main as tesseract_main
signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
resource.setrlimit(resource.RLIMIT_FSIZE, (1, resource.getrlimit(resource.RLIMIT_FSIZE)[1]))
entry = {'animate': pymatching._animate, 'package': package_main, 'tesseract': tesseract_main}[sys.argv[1]]
sys.exit(entry(sys.argv[2:]))
"""
    env = dict(os.environ, PYTHONPATH=str(pathlib.Path(pymatching.__file__).parent.parent))
    proc = subprocess.run([sys.executable, "-c", script, kind, *args], env=env, capture_output=True, text=True, timeout=15)
    assert proc.returncode == 1, proc.stderr
    assert ("tesseract:" if kind == "tesseract" else "Failed to write") in proc.stderr
    assert "Traceback" not in proc.stderr
    files = [tmp_path / "pred.01"] if kind == "tesseract" else list((tmp_path / "out").glob("*.svg"))
    assert files and all(p.stat().st_size <= 1 for p in files)

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


@pytest.mark.parametrize("kind", ["chromobius", "tesseract"])
@pytest.mark.parametrize("stage", ["write", "flush"])
def test_help_output_failure_keeps_caller_stream(kind, stage, monkeypatch, capsys):
    from stabilizer_qec.chromobius import __main__ as chromobius
    class BrokenStream:
        closed = False
        buffer = property(lambda self: self)
        def write(self, value):
            if stage == "write":
                raise OSError(errno.EPIPE, "primary write failure")
        def flush(self):
            raise OSError(errno.EIO, "cleanup flush failure")
        def close(self):
            self.closed = True
    stream = BrokenStream()
    monkeypatch.setattr(sys, "stdout", stream)
    entry = chromobius.main if kind == "chromobius" else tesseract.main
    assert entry(["chromobius", "--help"] if kind == "chromobius" else ["--help"]) == 1
    assert sys.stdout is stream and not stream.closed
    err = capsys.readouterr().err
    assert err == f"{kind}: [Errno {errno.EPIPE if stage == 'write' else errno.EIO}] {'primary write failure' if stage == 'write' else 'cleanup flush failure'}\n"


@pytest.mark.parametrize("unbuffered", [False, True])
@pytest.mark.parametrize("kind", ["chromobius", "tesseract_decoder"])
def test_help_closed_reader_process(kind, unbuffered):
    # Close the reader before spawning: deterministic EPIPE at the write (-u) or flush.
    reader, writer = os.pipe()
    os.close(reader)
    try:
        proc = subprocess.run([sys.executable, *(["-u"] if unbuffered else []), "-m", "stabilizer_qec." + kind, "--help"], stdout=writer, stderr=subprocess.PIPE, text=True, timeout=15)
    finally:
        os.close(writer)
    prefix = "chromobius" if kind == "chromobius" else "tesseract"
    assert proc.returncode == 1, proc.stderr
    assert proc.stderr == f"{prefix}: [Errno {errno.EPIPE}] {os.strerror(errno.EPIPE)}\n"


def _color_inputs(tmp_path):
    dem, shots = tmp_path / "color.dem", tmp_path / "color.01"
    dem.write_text("error(0.1) D0 D1 D2 L0\ndetector(0,0,0,0) D0\ndetector(0,0,0,1) D1\ndetector(0,0,0,2) D2\n")
    shots.write_text("0000\n" * 8)
    return dem, shots


def _color_args(mode, tmp_path):
    dem, shots = _color_inputs(tmp_path)
    args = ["chromobius", mode, "--out", str(tmp_path / "color.out")]
    if mode == "describe_decoder":
        return args + ["--in", str(dem)]
    return args + ["--dem", str(dem), "--in", str(shots), "--in_format", "01", "--in_includes_appended_observables"]


@pytest.mark.parametrize("mode", ["predict", "benchmark", "describe_decoder"])
@pytest.mark.parametrize("stage", ["write", "close", "both", "primary"])
def test_chromobius_owned_output_failure(mode, stage, tmp_path, monkeypatch, capsys):
    from stabilizer_qec.chromobius import __main__ as chromobius
    args = _color_args(mode, tmp_path)
    class BrokenWriter:
        closed = False
        def write(self, value):
            if stage in ("write", "both"):
                raise OSError(errno.ENOSPC, "primary write failure")
        def close(self):
            self.closed = True
            if stage != "write":
                raise OSError(errno.EIO, "cleanup close failure")
    stream = BrokenWriter()
    monkeypatch.setattr(chromobius, "_find_output_argument", lambda *a: stream)
    if stage == "primary":
        monkeypatch.setattr(chromobius, "_model", lambda *a: (_ for _ in ()).throw(chromobius._Fail("primary model failure")))
    assert chromobius.main(args) == 1
    err = capsys.readouterr().err
    expected = "primary model failure" if stage == "primary" else "primary write failure" if stage in ("write", "both") else "cleanup close failure"
    assert expected in err and "Traceback" not in err
    if stage in ("both", "primary"):
        assert "cleanup close failure" not in err
    assert stream.closed


@pytest.mark.parametrize("primary,status", [("fail", 1), ("abort", 134)])
def test_chromobius_primary_failure_survives_stdout_flush(primary, status, monkeypatch, capsys):
    from stabilizer_qec.chromobius import __main__ as chromobius
    class BrokenStream:
        buffer = property(lambda self: self)
        def flush(self):
            raise OSError(errno.EIO, "cleanup flush failure")
    stream = BrokenStream()
    monkeypatch.setattr(sys, "stdout", stream)
    exc = chromobius._Fail if primary == "fail" else chromobius._Abort
    def dispatch(*args):
        raise exc("primary dispatch failure")
    monkeypatch.setattr(chromobius, "_describe_decoder", dispatch)
    assert chromobius.main(["chromobius", "describe_decoder"]) == status
    err = capsys.readouterr().err
    assert "primary dispatch failure" in err and "cleanup flush failure" not in err
    assert sys.stdout is stream


@pytest.mark.skipif(sys.platform == "win32", reason="RLIMIT_FSIZE and SIGXFSZ require POSIX")
@pytest.mark.parametrize("mode", ["predict", "benchmark", "describe_decoder"])
def test_chromobius_bounded_real_output_failure(mode, tmp_path):
    resource = pytest.importorskip("resource")
    if not hasattr(resource, "RLIMIT_FSIZE"):
        pytest.skip("platform has no RLIMIT_FSIZE")
    args = _color_args(mode, tmp_path)
    script = """
import resource, signal, sys
from stabilizer_qec.chromobius.__main__ import main
signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
resource.setrlimit(resource.RLIMIT_FSIZE, (1, resource.getrlimit(resource.RLIMIT_FSIZE)[1]))
sys.exit(main(sys.argv[1:]))
"""
    proc = subprocess.run([sys.executable, "-c", script, *args], capture_output=True, text=True, timeout=15)
    assert proc.returncode == 1, proc.stderr
    assert "chromobius:" in proc.stderr and "Traceback" not in proc.stderr
    assert "File too large" in proc.stderr
    assert (tmp_path / "color.out").stat().st_size == 1

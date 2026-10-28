"""The command line's shot data streams, as Stim's does: memory that does not grow with the
shots, and batches that give the bytes one pass over all the shots gives (the same seeded
samples, conversions and predictions, the same messages for bad data). Also PyMatching's
count_mistakes, match graph diagrams of models, and model files with bytes that are not UTF-8."""

from __future__ import annotations

import gc
import pathlib
import re
import shutil
import subprocess
import sys
import tracemalloc

import numpy as np
import pytest

import stabilizer_qec as sq
from stabilizer_qec import _cli
from stabilizer_qec._shots import decode_shots, encode_shots

STIM = shutil.which("stim") or str(pathlib.Path(sys.executable).parent / "stim")
needs_stim = pytest.mark.skipif(not pathlib.Path(STIM).exists(), reason="the stim command is not installed")

SHOTS = 2500  # more than two of the command line's batches, and not a whole number of them


def run(*args, stdin: bytes = b""):
    return subprocess.run([sys.executable, "-m", "stabilizer_qec", *map(str, args)], input=stdin, capture_output=True)


def ours(*args, stdin: bytes = b"") -> bytes:
    r = run(*args, stdin=stdin)
    assert r.returncode == 0, r.stderr.decode()
    return r.stdout


def stims(*args) -> bytes:
    r = subprocess.run([STIM, *map(str, args)], capture_output=True)
    assert r.returncode == 0, r.stderr.decode()
    return r.stdout


@pytest.fixture
def model(tmp_path):
    c = sq.Circuit.generated("surface_code:rotated_memory_x", distance=3, rounds=3, after_clifford_depolarization=0.02, before_measure_flip_probability=0.02)
    (tmp_path / "c.stim").write_text(str(c))
    dem = c.detector_error_model(decompose_errors=True)
    (tmp_path / "c.dem").write_text(str(dem))
    (tmp_path / "u.dem").write_text(str(c.detector_error_model()))
    return tmp_path, c, dem


def test_model_and_circuit_files_with_bytes_that_are_not_utf8(tmp_path):
    """A byte that is not UTF-8 in a comment is passed over, as Stim, PyMatching and the
    Chromobius command line read it, from a file or stdin; CRLF lines read as in text mode."""
    dem = tmp_path / "u.dem"
    dem.write_bytes(b"detector(0,0,0,0) D0\ndetector(1,0,0,1) D1\nerror(0.1) D0 L0 # caf\xe9 \xff\nerror(0.1) D0 D1\r\nerror(0.1) D1\n")
    (tmp_path / "d.01").write_bytes(b"10\n01\n11\n")
    want = b"1\n0\n0\n"
    assert ours("predict", "--dem", dem, "--in", tmp_path / "d.01", "--in_format", "01") == want
    assert ours("decode", "--dem", dem, "--in", tmp_path / "d.01") == want
    (tmp_path / "o.01").write_bytes(want)
    assert ours("count_mistakes", "--dem", dem, "--in", tmp_path / "d.01", "--obs_in", tmp_path / "o.01") == b"0 / 3\n"
    assert len(ours("sample_dem", "--shots", 5, stdin=dem.read_bytes()).splitlines()) == 5
    circuit = b"X_ERROR(0.25) 0 # caf\xe9\r\nM 0 # \xff\nDETECTOR rec[-1]\n"
    (tmp_path / "c.stim").write_bytes(circuit)
    model = b"error(0.25) D0\n"
    assert ours("analyze_errors", "--in", tmp_path / "c.stim").replace(b"\r\n", b"\n") == model
    assert ours("analyze_errors", stdin=circuit).replace(b"\r\n", b"\n") == model
    assert decode_shots(ours("detect", "--in", tmp_path / "c.stim", "--shots", 10), "01", num_detectors=1).shape == (10, 1)

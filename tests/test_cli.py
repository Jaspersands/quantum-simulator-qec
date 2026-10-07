"""The command line against Stim's: the same commands and flags on the same inputs give the same
bytes wherever Stim's output is deterministic, and the right shapes and statistics where it is
sampled."""

from __future__ import annotations

import itertools
import pathlib
import shutil
import subprocess
import sys

import numpy as np
import pytest

import stabilizer_qec as sq

STIM = shutil.which("stim") or str(pathlib.Path(sys.executable).parent / "stim")
needs_stim = pytest.mark.skipif(not pathlib.Path(STIM).exists(), reason="the stim command is not installed")


def ours(*args, stdin: bytes = b"") -> bytes:
    r = subprocess.run([sys.executable, "-m", "stabilizer_qec", *args], input=stdin, capture_output=True)
    assert r.returncode == 0, r.stderr.decode()
    return r.stdout


def theirs(*args, stdin: bytes = b"") -> bytes:
    r = subprocess.run([STIM, *args], input=stdin, capture_output=True)
    assert r.returncode == 0, r.stderr.decode()
    return r.stdout


def both(*args, stdin: bytes = b""):
    assert ours(*args, stdin=stdin) == theirs(*args, stdin=stdin), args


GEN = [("repetition_code", "memory"), ("surface_code", "rotated_memory_x"), ("surface_code", "rotated_memory_z"), ("surface_code", "unrotated_memory_x"), ("surface_code", "unrotated_memory_z"), ("color_code", "memory_xyz")]


@needs_stim
@pytest.mark.parametrize("code,task", GEN)
def test_gen(code, task):
    for d, rounds, noise in itertools.product([3, 5], [2, 3], [[], ["--after_clifford_depolarization", "0.001", "--before_measure_flip_probability", "1e-05"], ["--after_reset_flip_probability", "0.125", "--before_round_data_depolarization", "0.25"]]):
        both("gen", "--code", code, "--task", task, "--distance", str(d), "--rounds", str(rounds), *noise)


@pytest.fixture
def files(tmp_path):
    c = sq.Circuit.generated("surface_code:rotated_memory_x", distance=3, rounds=4, after_clifford_depolarization=0.01, before_measure_flip_probability=0.01)
    (tmp_path / "c.stim").write_text(str(c))
    color = sq.Circuit.generated("color_code:memory_xyz", distance=5, rounds=3, after_clifford_depolarization=0.001)
    (tmp_path / "color.stim").write_text(str(color))
    return tmp_path


@needs_stim
def test_analyze_errors(files):
    c = str(files / "c.stim")
    for flags in ([], ["--decompose_errors"], ["--fold_loops"], ["--decompose_errors", "--fold_loops"], ["--approximate_disjoint_errors"]):
        both("analyze_errors", "--in", c, *flags)
    both("analyze_errors", "--in", str(files / "color.stim"), "--decompose_errors", "--ignore_decomposition_failures")


@needs_stim
def test_m2d_and_convert(files):
    c = str(files / "c.stim")
    meas = files / "m.b8"
    meas.write_bytes(theirs("sample", "--in", c, "--shots", "300", "--seed", "5", "--out_format", "b8"))
    for fmt in ("01", "b8", "dets", "hits"):
        both("m2d", "--circuit", c, "--in", str(meas), "--in_format", "b8", "--out_format", fmt)
        both("m2d", "--circuit", c, "--in", str(meas), "--in_format", "b8", "--out_format", fmt, "--append_observables")
    for o in ("obs_ours", "obs_theirs"):
        (files / o).unlink(missing_ok=True)
    ours("m2d", "--circuit", c, "--in", str(meas), "--in_format", "b8", "--obs_out", str(files / "obs_ours"))
    theirs("m2d", "--circuit", c, "--in", str(meas), "--in_format", "b8", "--obs_out", str(files / "obs_theirs"))
    assert (files / "obs_ours").read_bytes() == (files / "obs_theirs").read_bytes()
    dets = files / "d.01"
    dets.write_bytes(theirs("detect", "--in", c, "--shots", "128", "--seed", "2", "--append_observables"))
    nd, no = sq.Circuit((files / "c.stim").read_text()).num_detectors, 1
    # Stim's convert does not take ptb64 (it prints an error and nothing else); this one does.
    for a, b in itertools.permutations(["01", "b8", "r8", "hits", "dets"], 2):
        src = files / f"x.{a}"
        src.write_bytes(theirs("convert", "--in", str(dets), "--in_format", "01", "--out_format", a, "--num_detectors", str(nd), "--num_observables", str(no)))
        both("convert", "--in", str(src), "--in_format", a, "--out_format", b, "--num_detectors", str(nd), "--num_observables", str(no))
    both("convert", "--in", str(meas), "--in_format", "b8", "--out_format", "dets", "--circuit", c, "--types", "M")


@needs_stim
def test_explain_errors(files):
    c = str(files / "c.stim")
    both("explain_errors", "--in", c, "--single")
    (files / "f.dem").write_text("error(1) D0\nerror(1) D0 D1\n")
    both("explain_errors", "--in", c, "--dem_filter", str(files / "f.dem"))


@needs_stim
def test_deterministic_samples_and_diagrams(files):
    text = "R 0 1\nX 0\nM 0 1\nDETECTOR rec[-1]\nDETECTOR rec[-2]\nOBSERVABLE_INCLUDE(0) rec[-2]\n"
    (files / "det.stim").write_text(text)
    for fmt in ("01", "b8", "dets"):
        both("sample", "--in", str(files / "det.stim"), "--shots", "70", "--out_format", fmt)
        both("detect", "--in", str(files / "det.stim"), "--shots", "70", "--out_format", fmt, "--append_observables")
    assert ours("diagram", "--in", str(files / "c.stim"), "--type", "timeline-svg").startswith(b"<svg")


def test_sample_dem_and_decode(files):
    c = sq.Circuit((files / "c.stim").read_text())
    (files / "c.dem").write_text(str(c.detector_error_model(decompose_errors=True)))
    out = ours("sample_dem", "--in", str(files / "c.dem"), "--shots", "500", "--seed", "1", "--err_out", str(files / "e.01"), "--obs_out", str(files / "o.01"))
    dets = sq.read_shot_data_file(path=files / "e.01", format="01", num_measurements=c.detector_error_model(decompose_errors=True).num_errors)
    assert dets.shape[0] == 500 and len(out.splitlines()) == 500
    replay = ours("sample_dem", "--in", str(files / "c.dem"), "--shots", "500", "--replay_err_in", str(files / "e.01"))
    assert replay == out
    (files / "d.b8").write_bytes(ours("detect", "--in", str(files / "c.stim"), "--shots", "200", "--seed", "3", "--out_format", "b8"))
    predicted = ours("decode", "--dem", str(files / "c.dem"), "--in", str(files / "d.b8"), "--in_format", "b8")
    shots = sq.read_shot_data_file(path=files / "d.b8", format="b8", num_detectors=c.num_detectors)
    want = sq.Matching(c.detector_error_model(decompose_errors=True)).decode_batch(shots).astype(np.uint8)
    assert predicted == "".join("".join(map(str, row)) + "\n" for row in want).encode()


def test_help_and_errors():
    assert b"analyze_errors" in ours("help")
    assert b"--decompose_errors" in ours("help", "analyze_errors")
    r = subprocess.run([sys.executable, "-m", "stabilizer_qec", "frobnicate"], capture_output=True)
    assert r.returncode == 1 and b"Unrecognized command" in r.stderr
    r = subprocess.run([sys.executable, "-m", "stabilizer_qec", "gen", "--code", "surface_code"], capture_output=True)
    assert r.returncode == 1

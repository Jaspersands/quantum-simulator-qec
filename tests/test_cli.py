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


def lines(out: bytes) -> bytes:
    """Printed text with Windows' line endings read as newlines."""
    return out.replace(b"\r\n", b"\n")


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
    both("diagram", "--in", str(files / "c.stim"), "--type", "timeline-text")


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
    # The decoders for any model, on the undecomposed one.
    (files / "u.dem").write_text(str(c.detector_error_model()))
    shots_t = sq.read_shot_data_file(path=files / "d.b8", format="b8", num_detectors=c.num_detectors)
    for name, make in (("bplsd", sq.BpLsd), ("relay_bp", sq.RelayBp), ("search", sq.SearchDecoder)):
        got = ours("decode", "--dem", str(files / "u.dem"), "--in", str(files / "d.b8"), "--in_format", "b8", "--decoder", name)
        want = make(c.detector_error_model()).decode_batch(shots_t).astype(np.uint8)
        assert got == "".join("".join(map(str, row)) + "\n" for row in want).encode(), name


def test_help_and_errors():
    assert b"analyze_errors" in ours("help")
    assert b"--decompose_errors" in ours("help", "analyze_errors")
    r = subprocess.run([sys.executable, "-m", "stabilizer_qec", "frobnicate"], capture_output=True)
    assert r.returncode == 1 and b"Unrecognized command" in r.stderr
    r = subprocess.run([sys.executable, "-m", "stabilizer_qec", "gen", "--code", "surface_code"], capture_output=True)
    assert r.returncode == 1


@needs_stim
def test_repl_matches_stims():
    text = b"M 0\nX 0\nM 0\nX 2 3 9\nM 0 1 2 3 4 5 6 7 8 9\n# a comment\n\nREPEAT 2 {\n    X 0\n    M 0\n}\nMPP X0*X0 Z1\nDETECTOR rec[-1]\n"
    both("repl", stdin=text)


@needs_stim
def test_legacy_modes_and_deprecated_flags(files):
    det = files / "det.stim"
    det.write_text("X_ERROR(1) 0\nX 2\nM 0 1 2\nDETECTOR rec[-3]\nDETECTOR rec[-2]\nOBSERVABLE_INCLUDE(0) rec[-3]\nOBSERVABLE_INCLUDE(1) rec[-1]\n")
    both("--sample=3", "--in", str(det), "--skip_reference_sample")
    both("--sample", "--in", str(det), "--frame0")
    both("--detect=2", "--in", str(det), "--prepend_observables", "--out_format", "dets")
    both("--detect", "--in", str(det), "--prepend_observables")
    both("detect", "--in", str(det), "--out_format", "dets", "--shots", "2")
    for bad in (["--prepend_observables", "--append_observables"], ["--append_observables", "--obs_out", str(files / "o")], ["--out_format", "dets", "--obs_out", str(files / "o")]):
        r = subprocess.run([sys.executable, "-m", "stabilizer_qec", "detect", "--in", str(det), *bad], capture_output=True)
        assert r.returncode == 1 and b"Can't combine" in r.stderr, bad
    both("analyze_errors", "--detector_hypergraph", "--in", str(files / "c.stim"))


def test_help_topics():
    import gzip
    import json
    import re

    from stabilizer_qec import _cli

    data = json.loads(gzip.decompress((pathlib.Path(_cli.__file__).parent / "_stim_help.json.gz").read_bytes()))
    # Stim's pages, as Stim prints them: every instruction (by any name, in any case), format and index.
    names = {g: g for g in data["gates"]} | data["aliases"]
    for name, canonical in names.items():
        assert _cli.help_for(name) == _cli.help_for(name.lower()) == data["gates"][canonical], name
    assert set(names) == {a for g in sq.gate_data().values() for a in g.aliases}
    for f, page in data["formats"].items():
        assert _cli.help_for(f) == page
    for topic, page in data["indexes"].items():
        if topic != "commands_markdown":
            assert _cli.help_for(topic) == page
    # Stim's command pages, its `stim <command>` reading ours; the reference in Stim's layout.
    for c, page in data["commands"].items():
        assert _cli.help_for(c) == re.sub(r"\bstim (?=(?:" + "|".join(data["commands"]) + r")\b)", "stabilizer-qec ", page), c
        assert "stim " + c not in _cli.help_for(c)
        # Every flag Stim documents for the command, this command line takes.
        flags = set(re.findall(r"\[?(--\w+)", page.split("DESCRIPTION")[0]))
        assert flags <= set(_cli._parser(c)._option_string_actions), (c, flags - set(_cli._parser(c)._option_string_actions))
    assert _cli._commands_markdown("stim", data["commands"]) == data["indexes"]["commands_markdown"]
    md = _cli.help_for("commands_markdown")
    assert all(f'<a name="{c}"></a>' in md for c in _cli.COMMANDS) and "stim sample" not in md
    assert "--decoder" in _cli.help_for("decode")  # this package's own commands: their flags
    assert lines(ours("help", "sample")) == lines(ours("sample", "--help")) == _cli.help_for("sample").encode()
    assert ours("--help") == ours("help")
    assert b"### The 'CX' Gate" in ours("help", "cnot") and b"multiple of 64" in ours("help", "ptb64")
    r = subprocess.run([sys.executable, "-m", "stabilizer_qec", "help", "frobnicate"], capture_output=True)
    assert r.returncode == 1 and b"Unrecognized help topic 'frobnicate'" in r.stderr


@pytest.mark.skipif(not __import__("importlib").util.find_spec("stim"), reason="stim is not installed")
def test_help_pages_are_stims():
    """The shipped pages are what Stim prints (tools/stim_help.py --check), for the Stim they were
    taken from."""
    import gzip
    import json

    import stim

    if sys.platform == "win32":
        pytest.skip("Stim's Windows build prints some pages' angles otherwise (-180° for 180°); the pages are Stim's on Linux and macOS")
    if not stim.__version__.startswith("1.16."):
        pytest.skip(f"the pages are Stim 1.16's; stim {stim.__version__} is installed")
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "tools"))
    import stim_help

    stims = [sys.executable, "-c", "import stim, sys; sys.exit(stim.main(command_line_args=sys.argv[1:]))"]
    assert json.loads(gzip.decompress(stim_help.build(stims))) == json.loads(gzip.decompress(stim_help.OUT.read_bytes()))


def test_main_returns_exit_codes(tmp_path, capsys):
    path = tmp_path / "c.stim"
    assert sq.main(command_line_args=["gen", "--code", "repetition_code", "--task", "memory", "--rounds", "2", "--distance", "2", "--out", str(path)]) == 0
    assert path.read_text().startswith("# Generated repetition_code circuit.")
    assert sq.main(command_line_args=["gen", "--code", "nope"]) == 1
    assert sq.main(command_line_args=["sample", "--help"]) == 0


def test_panqec_and_qecsim_command_lines(tmp_path, capsys):
    """`stabilizer-qec panqec ...` and `stabilizer-qec qecsim ...` are PanQEC's `panqec ...` and
    qecsim's `qecsim ...`: the same arguments, outputs and exit codes."""
    import json

    pytest.importorskip("click")
    pytest.importorskip("networkx")

    def run(*args, code=0):
        capsys.readouterr()
        assert sq.main(command_line_args=list(args)) == code
        return capsys.readouterr()

    assert lines(ours("qecsim", "--version")) == b"qecsim, version 1.0b9\n"
    data = json.loads(run("qecsim", "run", "-r5", "-s13", "five_qubit", "generic.depolarizing", "generic.naive", "0.1").out)
    assert [(d["code"], d["n_run"], d["error_probability"]) for d in data] == [("5-qubit", 5, 0.1)]
    out = tmp_path / "ftp.json"
    run("qecsim", "run-ftp", "-r2", "-s5", f"-o{out}", "rotated_planar(3,3)", "3", "generic.bit_phase_flip", "rotated_planar.smwpm", "0.1")
    assert json.loads(out.read_text())[0]["time_steps"] == 3
    err = run("qecsim", "run", code=2).err
    assert "Usage: stabilizer-qec qecsim run " in err and "Missing argument 'CODE'" in err
    assert "No such command 'frobnicate'" in run("qecsim", "frobnicate", code=2).err
    assert "Usage: stabilizer-qec qecsim [OPTIONS] COMMAND" in run("help", "qecsim").out
    assert "stabilizer-qec qecsim run -r10" in run("qecsim", "run", "--help").out
    for module in ("pandas", "tqdm", "psutil", "scipy"):
        pytest.importorskip(module)
    assert run("panqec", "--version").out == "panqec, version 0.1.7\n"
    codes = run("panqec", "ls", "codes").out
    assert codes.startswith("Codes:\n") and "Toric3DCode" in codes
    # The commands its cluster scripts call.
    run("panqec", "run-parallel", "--help")
    run("panqec", "monitor-usage", "--help")
    page = run("panqec", "--help").out
    assert page.startswith("Usage: stabilizer-qec panqec [OPTIONS]") and run("help", "panqec").out == page


@needs_stim
def test_diagram_flags_and_replayed_shot_counts(files):
    c = files / "noisy.stim"
    c.write_text("X_ERROR(0.1) 0\nH 0\nTICK\nH 0\nTICK\nM 0\nDETECTOR rec[-1]\n")
    both("diagram", "--in", str(c), "--type", "timeline-text", "--remove_noise")
    both("diagram", "--in", str(c), "--type", "detslice-text", "--tick", "1:3")
    both("diagram", "--in", str(c), "--type", "timeline-text", "--tick", "1:3")
    assert ours("diagram", "--in", str(c), "--type", "detslice-with-ops-svg").startswith(b"<svg")
    s = files / "surface.stim"
    s.write_text(str(sq.Circuit.generated("surface_code:rotated_memory_x", distance=3, rounds=3, after_clifford_depolarization=0.001)))
    for f in ("2,2:4,2", "D5", "L0", "D2:4,4"):
        both("diagram", "--in", str(s), "--type", "detslice-text", "--tick", "4", "--filter_coords", f)
    assert len(ours("diagram", "--in", str(s), "--type", "detslice-svg", "--tick", "4", "--filter_coords", "D5")) < len(ours("diagram", "--in", str(s), "--type", "detslice-svg", "--tick", "4"))
    assert sq.main(command_line_args=["diagram", "--in", str(s), "--type", "detslice-text", "--filter_coords", "x,1"]) == 1
    dem, replay = files / "r.dem", files / "r.01"
    dem.write_text("error(1) D0 L0\nerror(0) D1\ndetector D2\n")
    replay.write_text("01\n10\n")
    for shots in ("1", "2"):
        both("sample_dem", "--in", str(dem), "--shots", shots, "--replay_err_in", str(replay), "--replay_err_in_format", "01")
    r = subprocess.run([sys.executable, "-m", "stabilizer_qec", "sample_dem", "--in", str(dem), "--shots", "3", "--replay_err_in", str(replay), "--replay_err_in_format", "01"], capture_output=True)
    assert r.returncode != 0 and b"Expected more error data" in r.stderr


@needs_stim
def test_hits_toggle_a_bit_named_twice():
    both("convert", "--in_format", "hits", "--out_format", "01", "--num_measurements", "3", stdin=b"0,0\n1,1,1\n")
    both("convert", "--in_format", "dets", "--out_format", "01", "--num_detectors", "3", stdin=b"shot D0 D0 D1\n")

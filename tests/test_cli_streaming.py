"""The command line's shot data streams, as Stim's does: memory that does not grow with the
shots, and batches that give the bytes one pass over all the shots gives (the same seeded
samples, conversions and predictions, the same messages for bad data). Also PyMatching's
count_mistakes, match graph diagrams of models, and model files with bytes that are not UTF-8."""

from __future__ import annotations

import gc
import os
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


def test_streamed_commands_use_memory_flat_in_the_shots(model):
    """Each shot data command at N and 4N shots: the peak of Python's allocations (numpy's
    arrays among them) stays put, where reading, sampling or writing every shot at once grows
    it fourfold."""
    tmp, c, dem = model
    nd, no = c.num_detectors, c.num_observables
    files = {}
    for n in (64, 8192, 32768):
        meas = c.compile_sampler(seed=1).sample(n)
        dets, obs = c.compile_detector_sampler(seed=1).sample(n, separate_observables=True)
        files[n] = {k: tmp / f"{k}{n}" for k in ("m.b8", "d.01", "o.01", "da.01")}
        files[n]["m.b8"].write_bytes(encode_shots(meas, "b8", num_measurements=c.num_measurements))
        files[n]["d.01"].write_bytes(encode_shots(dets, "01", num_detectors=nd))
        files[n]["o.01"].write_bytes(encode_shots(obs, "01", num_observables=no))
        files[n]["da.01"].write_bytes(encode_shots(np.hstack([dets, obs]), "01", num_detectors=nd, num_observables=no))
    out, obs_out = tmp / "out", tmp / "obs_out"
    commands = {
        "sample": lambda n, f: ["sample", "--in", tmp / "c.stim", "--shots", n, "--out", out],
        "detect": lambda n, f: ["detect", "--in", tmp / "c.stim", "--shots", n, "--out", out, "--obs_out", obs_out],
        "sample_dem": lambda n, f: ["sample_dem", "--in", tmp / "c.dem", "--shots", n, "--out", out, "--obs_out", obs_out],
        "m2d": lambda n, f: ["m2d", "--circuit", tmp / "c.stim", "--in", f["m.b8"], "--in_format", "b8", "--out", out, "--append_observables"],
        "convert": lambda n, f: ["convert", "--in", f["da.01"], "--in_format", "01", "--out_format", "dets", "--num_detectors", nd, "--num_observables", no, "--out", out],
        "decode": lambda n, f: ["decode", "--dem", tmp / "c.dem", "--in", f["d.01"], "--in_format", "01", "--out", out],
        "predict": lambda n, f: ["predict", "--dem", tmp / "c.dem", "--in", f["da.01"], "--in_format", "01", "--in_includes_appended_observables", "--out", out],
        "count_mistakes": lambda n, f: ["count_mistakes", "--dem", tmp / "c.dem", "--in", f["d.01"], "--obs_in", f["o.01"], "--out", out],
    }
    for name, args in commands.items():
        assert _cli.main([str(a) for a in args(64, files[64])]) == 0, name  # caches and imports first
        peaks = []
        for n in (8192, 32768):
            gc.collect()
            tracemalloc.start()
            try:
                assert _cli.main([str(a) for a in args(n, files[n])]) == 0, name
                peaks.append(tracemalloc.get_traced_memory()[1])
            finally:
                tracemalloc.stop()
        assert peaks[1] < 1.5 * peaks[0] + (256 << 10), (name, peaks)


def test_seeded_samples_in_batches_are_one_passs(model):
    tmp, c, dem = model
    nd, no = c.num_detectors, c.num_observables
    whole = c.compile_sampler(seed=7).sample(SHOTS)
    assert ours("sample", "--in", tmp / "c.stim", "--shots", SHOTS, "--seed", 7, "--out_format", "b8") == encode_shots(whole, "b8", num_measurements=c.num_measurements)
    dets, obs = c.compile_detector_sampler(seed=7).sample(SHOTS + 60, separate_observables=True)
    got = ours("detect", "--in", tmp / "c.stim", "--shots", SHOTS + 60, "--seed", 7, "--out_format", "ptb64", "--obs_out", tmp / "o.r8", "--obs_out_format", "r8")
    assert got == encode_shots(dets, "ptb64", num_detectors=nd)
    assert (tmp / "o.r8").read_bytes() == encode_shots(obs, "r8", num_observables=no)
    d, o, e = dem.compile_sampler(seed=7).sample(SHOTS, return_errors=True)
    got = ours("sample_dem", "--in", tmp / "c.dem", "--shots", SHOTS, "--seed", 7, "--obs_out", tmp / "o.01", "--err_out", tmp / "e.b8", "--err_out_format", "b8")
    assert got == encode_shots(d, "01", num_detectors=nd) and (tmp / "o.01").read_bytes() == encode_shots(o, "01", num_observables=no)
    assert (tmp / "e.b8").read_bytes() == encode_shots(e, "b8", num_measurements=dem.num_errors)
    # A replay reads the error data a batch at a time, its extra shots unread.
    assert ours("sample_dem", "--in", tmp / "c.dem", "--shots", SHOTS - 100, "--replay_err_in", tmp / "e.b8", "--replay_err_in_format", "b8") == encode_shots(d[: SHOTS - 100], "01", num_detectors=nd)
    r = run("sample_dem", "--in", tmp / "c.dem", "--shots", SHOTS + 1, "--replay_err_in", tmp / "e.b8", "--replay_err_in_format", "b8")
    assert r.returncode == 1 and b"Expected more error data for the requested number of shots." in r.stderr
    # ptb64 output needs whole blocks of shots: refused before anything is sampled or written.
    for flags in (["--out_format", "ptb64"], ["--obs_out", tmp / "late.01", "--obs_out_format", "ptb64"]):
        r = run("detect", "--in", tmp / "c.stim", "--shots", 100, "--out", tmp / "late", *flags)
        assert r.returncode == 1 and b"shots must be a multiple of 64 to use ptb64 format." in r.stderr
        assert not (tmp / "late").exists() and not (tmp / "late.01").exists()


@pytest.mark.parametrize("src", ["01", "b8", "r8", "ptb64", "hits", "dets"])
def test_conversions_in_batches_are_one_passs(model, src):
    """convert and m2d read every format a batch at a time: the bytes the whole file gives,
    from a file or stdin."""
    tmp, c, dem = model
    nd, no, nm = c.num_detectors, c.num_observables, c.num_measurements
    n = SHOTS + 60 if src == "ptb64" else SHOTS  # whole ptb64 blocks
    meas = c.compile_sampler(seed=3).sample(n)
    meas_file = tmp / f"m.{src}"
    meas_file.write_bytes(encode_shots(meas, src, num_measurements=nm))
    dets, obs = c.compile_m2d_converter().convert(measurements=meas, separate_observables=True)
    both = np.hstack([dets, obs])
    for out in ("01", "b8", "r8", "hits", "dets") + (("ptb64",) if src == "ptb64" else ()):
        got = ours("m2d", "--circuit", tmp / "c.stim", "--in", meas_file, "--in_format", src, "--out_format", out, "--append_observables")
        assert got == encode_shots(both, out, num_detectors=nd + no, names=[f"D{k}" for k in range(nd)] + [f"L{k}" for k in range(no)]), out
        got = ours("convert", "--in_format", src, "--out_format", out, "--num_measurements", nm, stdin=meas_file.read_bytes())
        assert got == encode_shots(meas, out, num_measurements=nm), out
    ours("convert", "--in", meas_file, "--in_format", src, "--out_format", "01", "--circuit", tmp / "c.stim", "--types", "M", "--out", tmp / "split")
    assert (tmp / "split").read_bytes() == encode_shots(meas, "01", num_measurements=nm)


@needs_stim
def test_conversions_in_batches_are_stims(model):
    tmp, c, _ = model
    meas = tmp / "m.b8"
    meas.write_bytes(stims("sample", "--in", tmp / "c.stim", "--shots", SHOTS, "--seed", 5, "--out_format", "b8"))
    for fmt in ("01", "r8", "dets"):
        args = ("m2d", "--circuit", tmp / "c.stim", "--in", meas, "--in_format", "b8", "--out_format", fmt, "--append_observables")
        assert ours(*args) == stims(*args), fmt
        args = ("convert", "--in", meas, "--in_format", "b8", "--out_format", fmt, "--num_measurements", c.num_measurements)
        assert ours(*args) == stims(*args), fmt


def test_bad_shot_data_in_a_later_batch(model, tmp_path):
    """Bad data past the first batch is refused with the message the whole file gets (its
    length, for b8 and ptb64), and m2d's sweep data must end where the measurements do."""
    tmp, c, _ = model
    nm = c.num_measurements
    meas = c.compile_sampler(seed=1).sample(SHOTS)
    good = encode_shots(meas, "b8", num_measurements=nm)
    (tmp / "m.b8").write_bytes(good + b"\x00")
    r = run("convert", "--in", tmp / "m.b8", "--in_format", "b8", "--out_format", "01", "--num_measurements", nm)
    stride = (nm + 7) // 8
    assert r.returncode == 1 and f"b8 data of {len(good) + 1} bytes is not whole shots of {stride} bytes".encode() in r.stderr
    lines = encode_shots(meas, "01", num_measurements=nm).split(b"\n")
    lines[2000] = lines[2000][:-1] + b"2"
    r = run("convert", "--in_format", "01", "--out_format", "b8", "--num_measurements", nm, stdin=b"\n".join(lines))
    assert r.returncode == 1 and b"a 01 line must be" in r.stderr
    r8 = encode_shots(meas, "r8", num_measurements=nm)
    for raw, message in ((r8 + bytes([1]), b"r8 data ends partway through a shot"), (r8 + bytes([nm + 1]), b"r8 data runs past the end of a shot")):
        r = run("convert", "--in_format", "r8", "--out_format", "01", "--num_measurements", nm, stdin=raw)
        assert r.returncode == 1 and message in r.stderr
    # Sweep data of a circuit that has sweep bits: too few and too many shots of it.
    sw = tmp_path / "sw.stim"
    sw.write_text("CX sweep[0] 0\nM 0 1\nDETECTOR rec[-2]\n")
    (tmp_path / "m.01").write_bytes(b"10\n" * SHOTS)
    for k in (SHOTS - 1, SHOTS + 3):
        (tmp_path / "s.01").write_bytes(b"1\n" * k)
        r = run("m2d", "--circuit", sw, "--in", tmp_path / "m.01", "--in_format", "01", "--sweep", tmp_path / "s.01", "--sweep_format", "01")
        assert r.returncode == 1 and f"{k} shots of sweep bits for {SHOTS} shots of measurements".encode() in r.stderr, k
    (tmp_path / "s.01").write_bytes(b"1\n" * SHOTS)
    got = ours("m2d", "--circuit", sw, "--in", tmp_path / "m.01", "--in_format", "01", "--sweep", tmp_path / "s.01", "--sweep_format", "01")
    assert got == b"0\n" * SHOTS


@pytest.mark.parametrize("decoder", ["matching", "relay_bp", "bposd", "search"])
def test_decoding_in_batches_is_one_passs(model, decoder):
    """decode's predictions are the decoder's for all the shots in one call; Relay-BP, whose
    memory strengths depend on each shot's index, included."""
    tmp, c, dem = model
    n = SHOTS if decoder in ("matching", "relay_bp") else 1100
    dets = c.compile_detector_sampler(seed=4).sample(n)
    (tmp / "d.b8").write_bytes(encode_shots(dets, "b8", num_detectors=c.num_detectors))
    model_file = tmp / ("c.dem" if decoder == "matching" else "u.dem")
    whole = {"matching": sq.Matching, "relay_bp": sq.RelayBp, "bposd": sq.BpOsd, "search": sq.SearchDecoder}[decoder](sq.DetectorErrorModel(model_file.read_text())).decode_batch(dets)
    got = ours("decode", "--dem", model_file, "--in", tmp / "d.b8", "--in_format", "b8", "--decoder", decoder)
    assert got == encode_shots(whole.astype(bool), "01", num_observables=c.num_observables)


def _pymatching():
    try:
        import pymatching
    except ImportError:
        return None
    return pymatching if hasattr(pymatching, "cli") and not pymatching.__name__.startswith("stabilizer_qec") else None


def test_count_mistakes_is_pymatchings(model, capfd):
    """PyMatching's count_mistakes: --time on stderr only, --obs_in over appended observables,
    none given compared with zeros, extra --obs_in shots unread, too few its error."""
    tmp, c, dem = model
    dets, obs = c.compile_detector_sampler(seed=9).sample(SHOTS, separate_observables=True)
    nd, no = c.num_detectors, c.num_observables
    (tmp / "d.01").write_bytes(encode_shots(dets, "01", num_detectors=nd))
    (tmp / "da.01").write_bytes(encode_shots(np.hstack([dets, obs]), "01", num_detectors=nd, num_observables=no))
    (tmp / "o.01").write_bytes(encode_shots(obs, "01", num_observables=no))
    (tmp / "o_long.01").write_bytes(encode_shots(np.vstack([obs, ~obs]), "01", num_observables=no))
    (tmp / "o_short.01").write_bytes(encode_shots(obs[:-1], "01", num_observables=no))
    (tmp / "o_flipped.01").write_bytes(encode_shots(~obs, "01", num_observables=no))
    predicted = sq.Matching(dem).decode_batch(dets).astype(bool)
    right = int((predicted != obs).any(axis=1).sum())
    cases = [
        (["--in", tmp / "d.01", "--obs_in", tmp / "o.01"], f"{right} / {SHOTS}\n"),
        (["--in", tmp / "da.01", "--in_includes_appended_observables"], f"{right} / {SHOTS}\n"),
        (["--in", tmp / "d.01", "--obs_in", tmp / "o_long.01"], f"{right} / {SHOTS}\n"),
        (["--in", tmp / "d.01"], f"{int(predicted.any(axis=1).sum())} / {SHOTS}\n"),
        (["--in", tmp / "da.01", "--in_includes_appended_observables", "--obs_in", tmp / "o_flipped.01"], f"{SHOTS - right} / {SHOTS}\n"),
    ]
    reference = _pymatching()
    for flags, want in cases:
        args = ["count_mistakes", "--dem", str(tmp / "c.dem"), *map(str, flags)]
        capfd.readouterr()
        assert _cli.main(args) == 0
        got = capfd.readouterr()
        assert (got.out, got.err) == (want, ""), flags
        if reference is not None:
            assert reference.cli(command_line_args=args + ["--out", str(tmp / "theirs")]) == 0
            assert (tmp / "theirs").read_text() == want, flags
    args = ["count_mistakes", "--dem", str(tmp / "c.dem"), "--in", str(tmp / "d.01"), "--obs_in", str(tmp / "o_short.01")]
    assert _cli.main(args) == 1
    got = capfd.readouterr()
    assert got.out == "" and got.err == "stabilizer-qec count_mistakes: Obs data ended before shot data ended.\n"
    # --time: PyMatching's two lines, on stderr; stdout is the count alone.
    for flags, shots in ((["--in", str(tmp / "d.01"), "--obs_in", str(tmp / "o.01")], SHOTS), (["--in", str(tmp / "empty.01"), "--obs_in", str(tmp / "o.01")], 0)):
        (tmp / "empty.01").write_bytes(b"")
        assert _cli.main(["count_mistakes", "--dem", str(tmp / "c.dem"), *flags, "--time"]) == 0
        got = capfd.readouterr()
        assert got.out == (f"{right} / {SHOTS}\n" if shots else "0 / 0\n")
        assert re.fullmatch(r"Total decoding time: \d+us\nDecoding time per shot: (inf|nan|[0-9.e+-]+)us\n", got.err), got.err
        if not shots:
            assert got.err.split("\n")[1] in ("Decoding time per shot: infus", "Decoding time per shot: nanus")


MATCH_GRAPHS = ["matchgraph-svg", "match-graph-svg", "matchgraph-3d", "matchgraph-3d-html", "match-graph-3d", "match-graph-3d-html"]


@pytest.mark.parametrize("kind", MATCH_GRAPHS)
def test_match_graph_diagrams_of_a_model(model, kind):
    """Every match graph type draws a detector error model as the model's own diagram does,
    and a circuit by its model; --remove_noise is refused for them, as by Stim. The command
    writes a diagram as text, its lines ended as the platform's text mode ends them (\\r\\n on
    Windows), as `_cli._write` writes all text, after Stim's text-mode output."""
    tmp, c, dem = model

    def text(raw):
        assert raw.count(b"\n") == raw.count(os.linesep.encode())
        return raw.decode().replace(os.linesep, "\n").rstrip("\n")

    assert text(ours("diagram", "--in", tmp / "c.dem", "--type", kind)) == str(dem.diagram(kind)).rstrip("\n")
    assert text(ours("diagram", "--in", tmp / "c.stim", "--type", kind)) == str(c.diagram(kind)).rstrip("\n")
    for src in ("c.stim", "c.dem"):
        r = run("diagram", "--in", tmp / src, "--type", kind, "--remove_noise", "--out", tmp / "nothing")
        assert r.returncode == 1 and r.stdout == b"" and not (tmp / "nothing").exists()
        assert r.stderr.decode().strip() == "stabilizer-qec diagram: --remove_noise is incompatible with match graph diagrams, because the noise is needed to produce the match graph."


@needs_stim
@pytest.mark.parametrize("kind", ["matchgraph-3d", "match-graph-3d"])
def test_match_graph_3d_of_a_model_is_stims(model, kind):
    tmp, _, _ = model
    assert ours("diagram", "--in", tmp / "c.dem", "--type", kind) == stims("diagram", "--in", tmp / "c.dem", "--type", kind)


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

"""Chromobius's command line (``python -m stabilizer_qec.chromobius``) against Chromobius 1.1.1's
own program, built from its C++ sources: each case recorded in
tests/data/chromobius_cli_reference.json.gz (tools/chromobius_cli_reference.py) is run again here
in an empty directory holding the same files, with the same arguments and standard input, and
its exit status, output, messages and the files it writes are compared with the program's.

The recording was made on macOS. The predictions, tables and messages are the same on every
platform; describe_decoder's Möbius model prints its numbers as Stim prints a model's on the
machine (16 significant digits on macOS and Windows, 19 on x86-64 Linux, 34 on ARM64 Linux), so
elsewhere those are compared as numbers."""

from __future__ import annotations

import base64
import gzip
import io
import json
import os
import pathlib
import re
import subprocess
import sys
import tempfile

import numpy as np
import pytest

import stabilizer_qec as sq
import stabilizer_qec.chromobius as chromobius
from stabilizer_qec._shots import decode_shots, encode_shots
from stabilizer_qec.chromobius import __main__ as cli

ROOT = pathlib.Path(__file__).resolve().parent.parent
REF = json.loads(gzip.decompress((ROOT / "tests" / "data" / "chromobius_cli_reference.json.gz").read_bytes()))
FILES = {k: base64.b64decode(v) for k, v in REF["files"].items()}
CASES = {c["name"]: c for c in REF["cases"]}

# Where this port knowingly differs (see the module's docstring): a malformed model is refused
# with this package's parser's message, not Stim's; and where Chromobius's program aborts on an
# exception it does not catch, this port prints the same message and exits with status 134
# rather than raising SIGABRT (-6).
PARSER_MESSAGE = {"predict malformed dem", "predict probability above 1", "describe malformed"}
ABORTS = {"predict huge hit", "benchmark huge hit"}


def run(case, tmp_path):
    """The case run in-process in a fresh directory: (exit status, stdout, stderr, the files it
    created or changed)."""
    d = pathlib.Path(tempfile.mkdtemp(prefix=re.sub(r"[^A-Za-z0-9]+", "_", case["name"]), dir=tmp_path))
    for name, data in FILES.items():
        (d / name).write_bytes(data)
    stdin = FILES[case["stdin"]] if case["stdin"] else b""
    saved = sys.stdin, sys.stdout, sys.stderr, os.getcwd()
    # newline="\n": the text written is kept as written. (The default, None, would write each
    # "\n" as os.linesep, so on Windows every message would end CRLF, as the real sys.stderr's
    # do; the recording is macOS's, LF.)
    sys.stdin = io.TextIOWrapper(io.BytesIO(stdin))
    sys.stdout = io.TextIOWrapper(io.BytesIO(), encoding="utf-8", newline="\n")
    sys.stderr = io.TextIOWrapper(io.BytesIO(), encoding="utf-8", newline="\n")
    try:
        os.chdir(d)
        status = cli.main(["chromobius", *case["args"]])
        sys.stdout.flush()
        sys.stderr.flush()
        out, err = sys.stdout.buffer.getvalue(), sys.stderr.buffer.getvalue().decode()
    finally:
        sys.stdin, sys.stdout, sys.stderr = saved[:3]
        os.chdir(saved[3])
    made = {}
    for p in sorted(d.rglob("*")):
        name = p.relative_to(d).as_posix()
        if p.is_file() and (name not in FILES or p.read_bytes() != FILES[name]):
            made[name] = p.read_bytes()
    return status, out, err, made


def recorded(case):
    files = {k: base64.b64decode(v) for k, v in case["files"].items()}
    return case["returncode"], base64.b64decode(case["stdout"]), case["stderr"], files


def mismatches(names, tmp_path, compare):
    bad = []
    for name in names:
        got, want = run(CASES[name], tmp_path), recorded(CASES[name])
        if not compare(name, got, want):
            bad.append((name, got[0], got[2][:300], want[0], want[2][:300]))
    return bad


def test_every_command_is_recorded():
    commands = {c["args"][0] if c["args"] else "" for c in CASES.values()}
    assert {"predict", "benchmark", "describe_decoder", "help", ""} <= commands
    assert len(CASES) > 150


def test_predict_equals_python_api(tmp_path):
    """Every prediction Chromobius's program writes, in every format, from every input format;
    and the same as the Python API's (``compile_decoder_for_dem``) on the same shots."""
    names = [n for n, c in CASES.items() if c["args"][:1] == ["predict"] and c["returncode"] == 0]
    assert len(names) > 40
    assert mismatches(names, tmp_path, lambda name, got, want: got == want) == []
    # The Python API on the d3 shots, written in each format, is what predict writes.
    dem = sq.DetectorErrorModel(FILES["d3.dem"].decode())
    nd, no = dem.num_detectors, dem.num_observables
    dets = decode_shots(FILES["d3.01"], "01", num_detectors=nd)
    packed = np.packbits(dets, axis=1, bitorder="little")
    obs = np.unpackbits(chromobius.compile_decoder_for_dem(dem).predict_obs_flips_from_dets_bit_packed(packed), axis=1, bitorder="little", count=no)
    for fin in ("01", "b8", "r8", "ptb64", "hits", "dets"):
        for fout in ("01", "b8", "r8", "hits", "dets"):
            _, out, _, _ = run(CASES[f"predict {fin} to {fout}"], tmp_path)
            assert out == encode_shots(obs, fout, num_observables=no), (fin, fout)


TIMING = ("setup_seconds", "decoding_seconds", "decoding_microseconds_per_shot", "decoding_microseconds_per_detection_event")


def same_benchmark(got: bytes, want: bytes) -> bool:
    """The same report, its timings aside: the same lines, labels aligned alike, the counts and
    rates equal, each timing a number as C++ prints one."""
    g, w = got.decode().split("\n"), want.decode().split("\n")
    if len(g) != len(w):
        return False
    for a, b in zip(g, w):
        label = b.split(" = ")[0]
        if label.strip() in TIMING:
            if not a.startswith(label + " = ") or not re.fullmatch(r"nan|inf|[0-9]+(\.[0-9]+)?(e[-+][0-9]+)?", a[len(label) + 3:]):
                return False
        elif a != b:
            return False
    return True


def test_benchmark_output_shape(tmp_path):
    names = [n for n, c in CASES.items() if c["args"][:1] == ["benchmark"] and c["returncode"] == 0]
    assert len(names) >= 10

    def compare(name, got, want):
        files_alike = sorted(got[3]) == sorted(want[3]) and all(same_benchmark(got[3][k], want[3][k]) for k in want[3])
        return got[0] == want[0] and got[2] == want[2] and same_benchmark(got[1], want[1]) and files_alike

    assert mismatches(names, tmp_path, compare) == []
    # With no shots, the rates per shot divide by zero as doubles do.
    _, out, _, _ = run(CASES["benchmark no shots"], tmp_path)
    assert "detection_fraction = nan\n" in out.decode()
    assert re.search(r"decoding_microseconds_per_shot = (inf|nan)\n", out.decode())


NUMBER = re.compile(r"-?[0-9]+(?:\.[0-9]+)?(?:e[-+][0-9]+)?")


def same_text_and_numbers(got: str, want: str) -> bool:
    """Equal on macOS; elsewhere the same text with each number equal to 15 digits (the model's
    numbers are printed at the machine's precision, as Stim prints them)."""
    if sys.platform == "darwin" or got == want:
        return got == want
    if NUMBER.split(got) != NUMBER.split(want):
        return False
    return all(np.isclose(float(a), float(b), rtol=1e-15, atol=0) for a, b in zip(NUMBER.findall(got), NUMBER.findall(want)))


def test_describe_decoder_text(tmp_path):
    """describe_decoder's text (``main_describe_decoder.cc``): the charge graph, the
    representatives, the drag graph and the Möbius model with its coordinates, for models read
    from a file, standard input or a circuit."""
    names = [n for n, c in CASES.items() if c["args"][:1] == ["describe_decoder"] and c["returncode"] == 0]
    assert len(names) >= 10

    def compare(name, got, want):
        files_alike = sorted(got[3]) == sorted(want[3]) and all(same_text_and_numbers(got[3][k].decode(), want[3][k].decode()) for k in want[3])
        return got[0] == want[0] and got[2] == want[2] and same_text_and_numbers(got[1].decode(), want[1].decode()) and files_alike

    assert mismatches(names, tmp_path, compare) == []


def test_missing_flags_messages(tmp_path):
    """Misuse gives Chromobius's (Stim's) message and exit status, and leaves the files it
    leaves: missing, empty, unknown, repeated and malformed flags; files that cannot be opened
    (an output file is created before the model is read); records that go wrong part way (the
    predictions before them written); models Chromobius refuses."""
    names = [n for n, c in CASES.items() if c["returncode"] != 0]
    assert len(names) > 70

    def compare(name, got, want):
        if name in PARSER_MESSAGE:
            return got[0] == want[0] == 1 and got[1] == want[1] and got[2] and got[3] == want[3]
        if name in ABORTS:
            return want[0] == -6 and got[0] == 134 and got[1:] == want[1:]
        return got == want

    assert mismatches(names, tmp_path, compare) == []


NO_PERFECT_MATCHING = "No perfect matching could be found. This likely means that the syndrome has odd parity in the support of a connected component without a boundary."


def test_unmatchable_shot_message(tmp_path):
    """A shot the Möbius matching cannot match (a detector no fault touches fired, or a
    component without a boundary has odd parity) gets PyMatching's message, which Chromobius's
    matcher throws, after the predictions of the shots before it: not Chromobius's message for
    a matching it cannot lift."""
    names = [n for n in CASES if "unmatchable" in n or "odd component" in n]
    assert len(names) == 5
    assert all(CASES[n]["returncode"] == 1 and CASES[n]["stderr"] == NO_PERFECT_MATCHING + "\n" for n in names)
    assert mismatches(names, tmp_path, lambda name, got, want: got == want) == []
    # The Python API names the shot, with the same message.
    shots = np.array([[0, 0], [0, 1], [1, 0], [0, 1]], dtype=np.uint8)
    decoder = sq.ColorMatching(sq.DetectorErrorModel(FILES["unmatchable.dem"].decode()))
    for threads in (1, 0):
        with pytest.raises(ValueError, match="^shot 2: " + re.escape(NO_PERFECT_MATCHING) + "$"):
            decoder.decode_batch(shots, threads=threads)
    assert decoder.decode_batch(shots[[0, 1, 3]]).tolist() == [[0], [1], [1]]


def test_runs_as_a_module(tmp_path):
    """``python -m stabilizer_qec.chromobius``, as a program: its exit status and output."""
    for name, data in FILES.items():
        (tmp_path / name).write_bytes(data)
    for name in ("help -h", "predict to a file", "predict from stdin", "predict unknown flag", "describe empty", "predict unmatchable"):
        case = CASES[name]
        stdin = FILES[case["stdin"]] if case["stdin"] else b""
        r = subprocess.run([sys.executable, "-m", "stabilizer_qec.chromobius", *case["args"]], cwd=tmp_path, input=stdin, capture_output=True, timeout=300)
        status, out, err, files = recorded(case)
        stderr = r.stderr
        if sys.platform == "win32":
            # Windows' sys.stderr is a text stream that writes each "\n" as "\r\n" (as is
            # Chromobius's std::cerr there); the recording is macOS's.
            stderr = stderr.replace(b"\r\n", b"\n")
        assert (r.returncode, r.stdout, stderr.decode()) == (status, out, err), name
        for k, v in files.items():
            assert (tmp_path / k).read_bytes() == v

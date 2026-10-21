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
ABORTS = {"predict huge hit", "benchmark huge hit", "predict hit of 2^64"}
# Records with a bit set past the last detector (a b8 record's padding; an observable read both
# appended and from --obs_in). Chromobius's own result for them is undefined: its
# detection_events_to_mobius_detection_events reads node_colors[d] past the end of the table
# for such a bit (decoder.cc), so another build or run may name another node or decode the
# record. The recording (macOS) refuses them with PyMatching's message, which this port gives
# deterministically; should a re-recording differ here, it is Chromobius's undefined behaviour.
UNDEFINED_IN_CHROMOBIUS = {"predict padding bit", "benchmark padding bit", "benchmark both observables"}


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
    predictions before them written); models Chromobius refuses. A record with a bit set past
    the last detector, whose result in Chromobius is undefined (``UNDEFINED_IN_CHROMOBIUS``), is
    refused deterministically with the message the recording shows."""
    names = [n for n, c in CASES.items() if c["returncode"] != 0]
    assert len(names) > 70
    assert UNDEFINED_IN_CHROMOBIUS <= set(names)

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


def test_integers_read_as_stims_uint64(tmp_path):
    """Hit and DETS indices are read as Stim's ``read_uint64`` reads them, in uint64_t
    arithmetic: a value that wraps past 2^64 without dropping below the step before it is read
    wrapped (30000000000000000000 as 11553255926290448384) and refused as too large; only a
    step that drops is the exception Chromobius does not catch."""
    want = {
        "predict wrapped hit": "hit index is too large.\n",
        "benchmark wrapped hit": "hit index is too large.\n",
        "predict largest hit": "hit index is too large.\n",
        "predict wrapped dets": "DETS data had a value larger than expected. Got D11553255926290448384 but expected length of D space to be 18.\n",
        "predict wrapped dets observable": "DETS data had a value larger than expected. Got L11553255926290448384 but expected length of L space to be 1.\n",
    }
    assert {n: CASES[n]["stderr"] for n in want} == want
    assert all(CASES[n]["returncode"] == 1 for n in want)
    assert CASES["predict hit of 2^64"]["returncode"] == -6
    assert mismatches(list(want), tmp_path, lambda name, got, want: got == want) == []
    status, out, err, files = run(CASES["predict hit of 2^64"], tmp_path)
    assert (status, out, err, files) == (134, *recorded(CASES["predict hit of 2^64"])[1:])
    # The reader itself.
    assert cli._Bytes(b"30000000000000000000\n").read_uint64() == (11553255926290448384, 10)
    assert cli._Bytes(b"18446744073709551615,").read_uint64() == (2**64 - 1, 44)
    assert cli._Bytes(b"x").read_uint64() == (None, 120)
    for digits in (b"18446744073709551616", b"99999999999999999999", b"184467440737095516150"):
        with pytest.raises(cli._Abort):
            cli._Bytes(digits).read_uint64()


def test_unliftable_shot_message(tmp_path):
    """A shot no lifting of the Möbius matching explains gets Chromobius's whole message, after
    the predictions of the shots before it: the cycle that failed (its detectors in its order,
    each with its colour and basis, and whether it fired) and the shot's detection events."""
    names = ["predict unliftable", "predict unliftable to a file", "benchmark unliftable"]
    cycle = "The cycle: {\n    D3[RX]\n    D1[BX, triggered]\n    D1[BX, triggered]\n    D2[GX]\n}\n"
    events = "All detection events in the shot: {\n    D0[NEUTRALUNKNOWN_BASIS, triggered]\n    D1[BX, triggered]\n}\n"
    for n in names:
        assert CASES[n]["returncode"] == 1 and CASES[n]["stderr"].startswith("Failed to lift a flattened edge cycle")
        assert CASES[n]["stderr"].endswith(cycle + events)
    assert mismatches(names, tmp_path, lambda name, got, want: got == want) == []
    # The Python API names the shot.
    decoder = sq.ColorMatching(sq.DetectorErrorModel(FILES["unliftable.dem"].decode()))
    with pytest.raises(ValueError, match="^shot 1: no lifting of the matching explains"):
        decoder.decode_batch(np.array([[0, 0, 0, 0], [1, 1, 0, 0]], dtype=np.uint8))


def test_describe_keeps_negative_zero(tmp_path):
    """A coordinate past the shift's length is not shifted, so a negative zero keeps its sign in
    the Möbius model (``collect_nodes.cc`` adds an offset only where the shift has one)."""
    _, out, err, _ = run(CASES["describe negative zero"], tmp_path)
    want = base64.b64decode(CASES["describe negative zero"]["stdout"]).decode()
    for line in ("detector(-0, 0, -0, 0, 2) D0\n", "detector(1, -0, 0, 1, 1) D2\n"):
        assert line in want and line in out.decode()
    assert err == ""


def test_decomposition_failure_coordinates_as_stim_gives_them(tmp_path):
    """The message for a fault Chromobius cannot split names each detector's coordinates as
    Stim's ``get_detector_coordinates`` gives them: its first declaration, shifted only where
    the shift has an entry (a negative zero past it keeps its sign)."""
    names = ["predict sevenfold shifted", "describe sevenfold shifted"]
    for n in names:
        assert CASES[n]["returncode"] == 1
        for line in ("        D0: coords=[0.5, -0, 0, 0] ColorBasis{.color=R, .basis=X}\n", "        D1: coords=[0.5, 1, -0, 1] ColorBasis{.color=G, .basis=X}\n"):
            assert line in CASES[n]["stderr"]
    assert mismatches(names, tmp_path, lambda name, got, want: got == want) == []


def test_error_targets_checked_as_chromobius_checks(tmp_path):
    """Each error's targets, in the model's order, as Chromobius checks them: an observable past
    L63 and a detector no declaration colours are refused with its messages."""
    names = ["predict observable past L63", "describe observable past L63", "predict undeclared detector", "describe undeclared detector"]
    assert [CASES[n]["stderr"] for n in names[::2]] == [
        "Max logical observable is L63 but a larger one appeared in 'error(0.1) D0 L64'\n",
        "Expected all detectors to have at least 4 coordinates, with the 4th identifying the basis and color (RedX=0, GreenX=1, BlueX=2, RedZ=3, GreenZ=4, BlueZ=5), but got error(0.1) D1 L0\n",
    ]
    assert mismatches(names, tmp_path, lambda name, got, want: got == want) == []


def test_runs_as_a_module(tmp_path):
    """``python -m stabilizer_qec.chromobius``, as a program: its exit status and output."""
    for name, data in FILES.items():
        (tmp_path / name).write_bytes(data)
    for name in ("help -h", "predict to a file", "predict from stdin", "predict unknown flag", "describe empty", "predict unmatchable", "predict unliftable"):
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

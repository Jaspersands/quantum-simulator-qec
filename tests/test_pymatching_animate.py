"""PyMatching 2.4's ``animate``: ``stabilizer_qec.pymatching.cli(command_line_args=["animate",
...])``, ``python -m stabilizer_qec.pymatching animate`` and ``stabilizer-qec animate``, against
the frames and messages recorded from PyMatching itself (tools/pymatching_animate_fixtures.py:
PyMatching 2.4.0's macOS arm64 wheel, and its C++ program for exit codes).

PyMatching draws its regions in heap-address order (``std::set<GraphFillRegion *>``), which no
port can reproduce, so a frame is compared as its lines sorted. The frame names, their count,
stderr and the return value must be PyMatching's.

The recorded finite frames are exact on macOS arm64, where they were recorded. PyMatching's own wheels draw
differently elsewhere, and so does this port, as each does on its platform: a C++ build fuses
``a * b + c`` into one rounding on aarch64 but not on x86-64, polygon corners at equal angles
come out in the order the platform's ``std::sort`` leaves them (libc++, libstdc++, Microsoft's),
and ``atan2f`` is the platform's. So off macOS arm64 a frame's lines must be the same elements
with the same colours and corner counts, and its numbers, sorted, the same to within a unit in
the sixth significant digit.
"""

from __future__ import annotations

import base64
import errno
import gzip
import json
import math
import os
import pathlib
import platform
import re
import subprocess
import sys
import threading

import pytest

from stabilizer_qec import DetectorErrorModel, pymatching
from stabilizer_qec._cli import main as package_main

FIXTURES = pathlib.Path(__file__).parent / "pymatching_port" / "animate"
CASES = sorted(p.name[: -len(".json.gz")] for p in FIXTURES.glob("*.json.gz"))
MESSAGES = {m["name"]: m for m in json.loads((FIXTURES / "messages.json").read_text())}
EXACT = sys.platform == "darwin" and platform.machine() == "arm64"
# What the message cases run in (tools/pymatching_animate_fixtures.py's MESSAGE_MODEL).
MESSAGE_MODEL = "error(0.1) D0 D1\nerror(0.1) D1\ndetector(0, 0) D0\ndetector(1, 0) D1\n"


def _case(name: str) -> dict:
    with gzip.open(FIXTURES / f"{name}.json.gz", "rt") as f:
        return json.load(f)


def _inputs(rec: dict, where: pathlib.Path) -> None:
    (where / "model.dem").write_bytes(base64.b64decode(rec["dem_b64"]) if "dem_b64" in rec else rec["dem"].encode())
    (where / rec["shot_name"]).write_bytes(base64.b64decode(rec["shot"]))


def _run(args: list, capsys) -> tuple:
    capsys.readouterr()
    code = pymatching.cli(command_line_args=args)
    return code, capsys.readouterr().err


def _frames(where: pathlib.Path) -> list:
    return sorted(os.listdir(where)) if where.is_dir() else []


_NUMBER = re.compile(r"-?\d+(?:\.\d+)?(?:e[-+]\d+)?")


def _same_frame(ours: list, theirs: list) -> bool:
    if EXACT:
        return ours == theirs
    # Off macOS arm64 (see the module docs): the elements exactly, the numbers to within a unit
    # in the sixth significant digit. Sorting moves no number further than it moved.
    skeletons = [sorted(_NUMBER.sub("#", line) for line in lines) for lines in (ours, theirs)]
    numbers = [sorted(float(x) for line in lines for x in _NUMBER.findall(line)) for lines in (ours, theirs)]
    return skeletons[0] == skeletons[1] and len(numbers[0]) == len(numbers[1]) and all(math.isclose(a, b, rel_tol=2e-5, abs_tol=1e-3) for a, b in zip(*numbers))


@pytest.mark.parametrize("name", CASES)
def test_frames_equal_as_sorted_lines(name, tmp_path, monkeypatch, capsys):
    rec = _case(name)
    _inputs(rec, tmp_path)
    monkeypatch.chdir(tmp_path)
    _run(rec["args"], capsys)
    names = _frames(tmp_path / "out")
    assert len(names) == len(rec["frames"])
    for n, frame in zip(names, rec["frames"]):
        assert _same_frame(sorted((tmp_path / "out" / n).read_text().split("\n")), [rec["lines"][i] for i in frame]), (name, n)


@pytest.mark.parametrize("name", CASES)
def test_frame_names_and_count(name, tmp_path, monkeypatch, capsys):
    rec = _case(name)
    _inputs(rec, tmp_path)
    monkeypatch.chdir(tmp_path)
    code, err = _run(rec["args"], capsys)
    assert (code, err) == (rec["exit"], rec["stderr"])
    assert _frames(tmp_path / "out") == rec["frame_names"]


def _message_case(name: str, tmp_path, monkeypatch, capsys) -> tuple:
    m = MESSAGES[name]
    (tmp_path / "model.dem").write_text(MESSAGE_MODEL)
    (tmp_path / "shot.01").write_text("11\n")
    (tmp_path / "empty.01").write_text("")
    monkeypatch.chdir(tmp_path)
    code, err = _run(["animate"] + m["args"], capsys)
    frames = len(_frames(tmp_path / "out")) + len(_frames(tmp_path / "out2"))
    return m, code, err, frames


def _check_message(name, tmp_path, monkeypatch, capsys) -> None:
    m, code, err, frames = _message_case(name, tmp_path, monkeypatch, capsys)
    assert m["wheel"] == f"returned {code}\n"
    assert m["exit"] == code
    assert (err, frames) == (m["wheel_stderr"], m["wheel_frames"])


REQUIRED = [n for n in MESSAGES if n.startswith(("missing--", "empty--"))]


@pytest.mark.parametrize("name", REQUIRED + ["max_edge_weight"])
def test_required_flags_messages(name, tmp_path, monkeypatch, capsys):
    """Every flag is required, and an int flag without a value says so; ``--max_edge_weight``,
    which PyMatching's usage lists, is refused as an unknown flag, as PyMatching refuses it."""
    _check_message(name, tmp_path, monkeypatch, capsys)
    if name.startswith(("missing--held", "missing--max", "missing--pixels", "empty--")):
        assert MESSAGES[name]["wheel_stderr"] == f"Must specify a value for int flag '--{name.split('--')[1]}'.\n"


@pytest.mark.parametrize("name", ["pixels_3", "pixels_4", "pixels_2048", "pixels_2049"])
def test_pixels_per_unit_length_range(name, tmp_path, monkeypatch, capsys):
    _check_message(name, tmp_path, monkeypatch, capsys)


DIRECTORIES = [n for n in MESSAGES if n.startswith(("dets_in_directory_", "dem_in_directory"))]
OTHER = [n for n in MESSAGES if n not in REQUIRED + DIRECTORIES and not n.startswith("pixels_") and n != "max_edge_weight" and MESSAGES[n]["wheel"].startswith("returned")]


@pytest.mark.parametrize("name", OTHER)
def test_misuse_messages(name, tmp_path, monkeypatch, capsys):
    """Unknown flags and words, values out of range or not integers, a format PyMatching does not
    know, a value for a switch, files that do not open, a file without a shot, ``--out_dir``
    with and without its slash, and ``--`` ending the flags."""
    _check_message(name, tmp_path, monkeypatch, capsys)


@pytest.mark.parametrize("name", DIRECTORIES)
def test_a_directory_reads_as_an_empty_file(name, tmp_path, monkeypatch, capsys):
    """``fopen`` opens a directory on macOS and Linux and its reads fail, so to PyMatching a
    directory is an empty file: as ``--dets_in`` it has no shot (in every format), and as
    ``--dem_in`` it is a model without detectors. On Windows the C runtime's ``fopen`` refuses a
    directory, as Python's ``os.open`` does there, so PyMatching and this say ``Failed to open``."""
    if sys.platform == "win32":
        m, code, err, frames = _message_case(name, tmp_path, monkeypatch, capsys)
        assert (code, err, frames) == (1, "Failed to open '.'\n", 0)
    else:
        _check_message(name, tmp_path, monkeypatch, capsys)


# The two-detector message model's shot ``11`` in each format, as PyMatching reads it.
SHOT_11 = {"01": b"11\n", "b8": b"\x03", "dets": b"shot D0 D1\n", "hits": b"0,1\n", "ptb64": (1).to_bytes(8, "little") * 2, "r8": b"\x00\x00\x00"}
FRAME_FLAGS = ["--held_frames_per_event", "1", "--held_frames_at_start", "1", "--held_frames_at_end", "1", "--max_growth_between_frames", "10", "--pixels_per_unit_length", "10"]


@pytest.mark.skipif(sys.platform == "win32", reason="Windows has no /dev/stdin to name a pipe by")
@pytest.mark.parametrize("fmt", sorted(SHOT_11))
def test_a_pipe_is_read_to_its_first_shot(fmt, tmp_path, monkeypatch, capsys):
    """PyMatching reads ``--dets_in`` only to the end of its first shot, so a writer that sends
    one shot and keeps the pipe open is not waited for; the frames are those of the same shot
    read from a file."""
    (tmp_path / "model.dem").write_text(MESSAGE_MODEL)
    (tmp_path / f"shot.{fmt}").write_bytes(SHOT_11[fmt])
    args = ["animate", "--dem_in", "model.dem", "--dets_in_format", fmt] + FRAME_FLAGS
    monkeypatch.chdir(tmp_path)
    assert _run(args + ["--dets_in", f"shot.{fmt}", "--out_dir", "from_file", "--quiet"], capsys) == (0, "")
    with open(tmp_path / "stderr.txt", "wb") as err:
        proc = subprocess.Popen([sys.executable, "-m", "stabilizer_qec.pymatching"] + args + ["--dets_in", "/dev/stdin", "--out_dir", "from_pipe"], cwd=tmp_path, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=err)
        try:
            proc.stdin.write(SHOT_11[fmt])
            proc.stdin.flush()
            code = proc.wait(timeout=60)  # (the pipe still open)
        finally:
            proc.stdin.close()
            if proc.poll() is None:
                proc.kill()
                proc.wait()
    assert code == 0
    assert (tmp_path / "stderr.txt").read_text().splitlines() == [f"wrote from_pipe/{n}" for n in _frames(tmp_path / "from_file")]
    assert [(tmp_path / "from_pipe" / n).read_bytes() for n in _frames(tmp_path / "from_pipe")] == [(tmp_path / "from_file" / n).read_bytes() for n in _frames(tmp_path / "from_file")]


@pytest.mark.parametrize("fmt", ["01", "b8", "ptb64"])
def test_a_long_file_is_read_only_to_its_first_shot(fmt, tmp_path):
    """Stim's reader stops at the first shot's end; so does this, reading one buffer of a
    file holding a million shots more."""
    path = tmp_path / f"shots.{fmt}"
    path.write_bytes(SHOT_11[fmt] * (1 + 10**6 // len(SHOT_11[fmt])))
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_BINARY", 0))
    try:
        assert pymatching._read_first_shot(pymatching._Stream(fd), fmt, 2, 0) == [0, 1]
        assert os.lseek(fd, 0, os.SEEK_CUR) <= 1 << 16
    finally:
        os.close(fd)


def test_stream_reads_as_stdio_does():
    """``getc`` and ``fread`` over a pipe: ``fread`` waits for its bytes across short reads, and
    the end, once reached, stays the end."""
    r, w = os.pipe()
    os.write(w, b"ab")
    s = pymatching._Stream(r)
    later = threading.Timer(0.2, lambda: (os.write(w, b"cdef"), os.close(w)))
    later.start()
    try:
        assert s.read(4) == b"abcd"
        assert s.getc() == ord("e")
        assert s.read() == b"f"
        assert (s.getc(), s.read(3), s.getc()) == (-1, b"", -1)
    finally:
        later.join()
        os.close(r)


@pytest.mark.parametrize("raw", [b"error(0.1) D0 D1\xa0\nerror(0.1) D1\n", b"error(0.1) D0\x85D1\nerror(0.1) D1\n", "error(0.1) D0 D1\u00a0\nerror(0.1) D1\n".encode()], ids=["nbsp", "nel", "utf8_nbsp"])
def test_a_byte_past_ascii_outside_a_comment_is_refused(raw, tmp_path, monkeypatch, capsys):
    """Stim reads a model's bytes; one past ASCII outside a comment or a tag is no space, so
    PyMatching returns 1 (with Stim's ``Targets must be separated by spacing.`` for these) and
    draws nothing. So does this, with its parser's message (whose wording is the package's, not
    Stim's), the byte shown as its Latin-1 character. (Read as Latin-1, 0x85 and 0xA0 would be
    spaces, and the model accepted.)"""
    (tmp_path / "model.dem").write_bytes(raw)
    (tmp_path / "shot.01").write_text("11\n")
    monkeypatch.chdir(tmp_path)
    code, err = _run(["animate", "--dem_in", "model.dem", "--dets_in", "shot.01", "--dets_in_format", "01", "--out_dir", "out"] + FRAME_FLAGS, capsys)
    bad = raw.split(b"\n")[0].split(b" ")[-1].decode("latin-1")
    assert (code, err, _frames(tmp_path / "out")) == (1, f"line 1: bad target '{bad}'\n", [])


def test_the_graph_is_the_files_text():
    """``digits17``'s probability, printed again at the Apple-silicon model's 16 digits,
    is another number at another weight level, so its frames (equal to PyMatching's) show
    the graph is built from the file's own text. (Its weight, ``log((1 - p) / p) * 999 / log(9)``, is ten ulps below 101.5,
    so it is at level 101 with any platform's ``log``; printed with sixteen digits, it is one ulp
    above, at 102.)"""
    text = _case("digits17")["dem"]
    assert "0.44442025079306513" in text
    # The native DEM writer uses 19/34 digits on Linux, preserving this value.
    # Explicitly model the fixture's 16-digit reserialization on every platform.
    rounded = re.sub(r"error\(([^)]+)\)", lambda m: f"error({float(m[1]):.16g})", text)
    assert "0.44442025079306513" not in rounded
    assert [round(math.log((1 - p) / p) * 999 / math.log(9)) for p in (0.44442025079306513, float("%.16g" % 0.44442025079306513))] == [101, 102]
    if sys.platform == "darwin":
        assert rounded.rstrip("\n") == str(DetectorErrorModel(text))


def test_fixtures_agree_with_pymatchings_program():
    """Every case's frames, recorded from PyMatching's wheel, are what PyMatching's own C++
    program draws when built for arm64 (tools/pymatching_animate_fixtures.py)."""
    assert [n for n in CASES if not _case(n)["program_frames_equal"]] == []


OUT_DIR_CASES = [n for n in MESSAGES if n.startswith("out_dir_") and MESSAGES[n]["wheel"].startswith("RuntimeError")]
# A ``std::filesystem_error`` is worded by the C++ library PyMatching was built with, around the
# reason for the OS's own errno, so the message is the platform's:
# - macOS (libc++), where the fixtures were recorded: exactly the recorded message. There
#   ``mkdir("model.dem/")`` fails ENOTDIR, and libc++ words it
#   ``filesystem error: in create_directory: <strerror> ["<path>"]``.
# - Linux (libstdc++, PyMatching's manylinux wheels): ``create_directory`` throws
#   ``filesystem_error("cannot create directory", p, ec)``, worded
#   ``filesystem error: cannot create directory: <strerror> [<path>]``, and Linux's
#   ``mkdir("model.dem/")`` fails EEXIST, which libstdc++ keeps because the path is not a
#   directory. These are what libstdc++.so.6.0.30 itself throws on Linux (Debian 12 aarch64 and
#   Ubuntu 22.04 x86_64) for PyMatching's ``if (!exists(out_dir)) create_directory(out_dir);``,
#   recorded by calling it through ctypes in this test's two situations.
# - Windows (MSVC's STL, PyMatching's Windows wheel): worded
#   ``create_directory: <Win32 FormatMessage text>: "<path>"``, with Win32 error codes. That was
#   never recorded and the port does not reproduce it, so the test is skipped there, as on any
#   other platform, where neither the library nor the errno was recorded.
OUT_DIR_LIBSTDCXX = {
    "out_dir_is_a_file": "filesystem error: cannot create directory: File exists [model.dem/]",
    "out_dir_parent_missing": "filesystem error: cannot create directory: No such file or directory [missing/out/]",
}


@pytest.mark.skipif(sys.platform != "darwin" and not sys.platform.startswith("linux"), reason="recorded for libc++ on macOS and libstdc++ on Linux only; MSVC's wording on Windows (create_directory: <Win32 message>: \"<path>\") is not reproduced: see OUT_DIR_LIBSTDCXX")
@pytest.mark.parametrize("name", OUT_DIR_CASES)
def test_out_dir_that_cannot_be_made(name, tmp_path, monkeypatch, capsys):
    """PyMatching's ``std::filesystem::create_directory`` throws past its handler: its program
    aborts and its ``cli`` raises RuntimeError with the filesystem's message, worded as the
    platform's C++ library words it (see OUT_DIR_LIBSTDCXX)."""
    with pytest.raises(RuntimeError) as ex:
        _message_case(name, tmp_path, monkeypatch, capsys)
    if sys.platform.startswith("linux"):
        assert str(ex.value) == OUT_DIR_LIBSTDCXX[name]
    else:
        assert f"RuntimeError: {ex.value}\n" == MESSAGES[name]["wheel"]
    assert capsys.readouterr().err == MESSAGES[name]["wheel_stderr"] == ""


@pytest.mark.parametrize("name", OUT_DIR_CASES)
def test_out_dir_wording_on_linux(name, monkeypatch):
    """The Linux wording, checked on every platform: the errno Linux's ``mkdir`` gives in each
    situation (see OUT_DIR_LIBSTDCXX), worded as libstdc++ words it."""
    out_dir = MESSAGES[name]["args"][MESSAGES[name]["args"].index("--out_dir") + 1] + "/"
    err = {"out_dir_is_a_file": errno.EEXIST, "out_dir_parent_missing": errno.ENOENT}[name]
    monkeypatch.setattr(sys, "platform", "linux")
    assert str(pymatching._filesystem_error("create_directory", OSError(err, os.strerror(err)), out_dir)) == OUT_DIR_LIBSTDCXX[name]


@pytest.mark.parametrize("name", OUT_DIR_CASES)
def test_package_command_reports_an_abort_in_one_line(name, tmp_path, monkeypatch, capsys):
    """Where PyMatching's program aborts (its ``cli`` raises, and so does ``pymatching.cli``),
    ``stabilizer-qec animate`` reports it as the package's commands report a failure:
    ``stabilizer-qec animate: <message>``, exit 1, no traceback."""
    with pytest.raises(RuntimeError) as ex:
        _message_case(name, tmp_path, monkeypatch, capsys)
    capsys.readouterr()
    code = package_main(command_line_args=["animate"] + MESSAGES[name]["args"])
    assert (code, capsys.readouterr().err) == (1, f"stabilizer-qec animate: {ex.value}\n")


def test_package_command_reports_a_too_big_index_in_one_line(tmp_path, monkeypatch, capsys):
    """A hit index past 64 bits: PyMatching's ``cli`` raises RuntimeError (``read_uint64``'s
    ``std::runtime_error``), as ``pymatching.cli`` does; ``stabilizer-qec animate`` says it in one
    line."""
    (tmp_path / "model.dem").write_text(MESSAGE_MODEL)
    (tmp_path / "shot.hits").write_text("18446744073709551616\n")
    monkeypatch.chdir(tmp_path)
    args = ["animate", "--dem_in", "model.dem", "--dets_in", "shot.hits", "--dets_in_format", "hits", "--out_dir", "out"] + FRAME_FLAGS
    with pytest.raises(RuntimeError, match="^Integer value read from file was too big$"):
        pymatching.cli(command_line_args=args)
    capsys.readouterr()
    assert (package_main(command_line_args=args), capsys.readouterr().err) == (1, "stabilizer-qec animate: Integer value read from file was too big\n")
    assert not (tmp_path / "out").exists()


def test_unrecognized_command_lists_animate():
    with pytest.raises(ValueError) as ex:
        pymatching.cli(command_line_args=["nope"])
    assert f"ValueError: {ex.value}\n" == MESSAGES["unrecognized_command"]["wheel"]


def test_module_and_package_command_lines(tmp_path):
    """``python -m stabilizer_qec.pymatching animate`` exits with what ``cli`` returns;
    ``stabilizer-qec animate`` is the same command (and has a help page)."""
    rec = _case("no_matching")
    _inputs(rec, tmp_path)
    for prog in (["stabilizer_qec.pymatching"], ["stabilizer_qec"]):
        r = subprocess.run([sys.executable, "-m", *prog] + rec["args"], cwd=tmp_path, capture_output=True)
        # (stderr is text: Windows ends its lines \r\n, as PyMatching's std::cerr does there.)
        assert (r.returncode, r.stderr.decode().replace("\r\n", "\n")) == (1, rec["stderr"])
        assert _frames(tmp_path / "out") == rec["frame_names"]
    r = subprocess.run([sys.executable, "-m", "stabilizer_qec", "help", "animate"], capture_output=True)
    assert r.returncode == 0 and b"--pixels_per_unit_length" in r.stdout


def _edge_args(tmp_path, raw=MESSAGE_MODEL.encode()):
    (tmp_path / "edge.dem").write_bytes(raw)
    (tmp_path / "edge.hits").write_bytes(b"0\n")
    return ["animate", "--dem_in", str(tmp_path / "edge.dem"), "--dets_in", str(tmp_path / "edge.hits"), "--dets_in_format", "hits", "--out_dir", str(tmp_path / "out"), "--quiet"] + FRAME_FLAGS


@pytest.mark.parametrize("raw", [b"error(0.1) D0\vD1\n", b"error(0.1) D0\f\n", b"error(\r0.1) D0\n"])
def test_command_control_whitespace_is_refused(raw, tmp_path, capsys):
    assert _run(_edge_args(tmp_path, raw), capsys)[0] == 1
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("raw", [b"\v\ferror(0.1) D0\n", b"error(0.1)\rD0\n", b"# comment\v\f\r\xff\nerror[ordinary](0.1) D0\n", b"repeat 1 {\v\f\nerror(0.1) D0\n}\v\f\n"])
def test_between_commands_controls_and_tags_are_preserved(raw, tmp_path, capsys):
    assert _run(_edge_args(tmp_path, raw), capsys) == (0, "")
    assert _frames(tmp_path / "out")


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX ENAMETOOLONG semantics; native MSVC wording is not reproduced")
def test_stat_failure_is_distinct_from_create_directory(tmp_path, capsys):
    args = _edge_args(tmp_path)
    args[args.index("--out_dir") + 1] = str(tmp_path / ("x" * 300))
    # Native library/OS wording and long diagnostic truncation are not portable; the operation
    # is status (libstdc++) / posix_stat (libc++), never create_directory.
    with pytest.raises(RuntimeError, match="status|posix_stat") as ex:
        pymatching.cli(command_line_args=args)
    assert "create_directory" not in str(ex.value)
    assert os.strerror(errno.ENAMETOOLONG) in str(ex.value)
    assert package_main(args) == 1
    assert "Traceback" not in capsys.readouterr().err


@pytest.mark.skipif(sys.platform == "win32" or getattr(os, "geteuid", lambda: 0)() == 0, reason="requires POSIX directory permissions and a non-root user")
def test_inaccessible_parent_is_a_stat_failure(tmp_path):
    args = _edge_args(tmp_path)
    parent = tmp_path / "private"
    parent.mkdir()
    args[args.index("--out_dir") + 1] = str(parent / "out")
    parent.chmod(0)
    try:
        with pytest.raises(RuntimeError, match="status|posix_stat") as ex:
            pymatching.cli(command_line_args=args)
        assert os.strerror(errno.EACCES) in str(ex.value)
    finally:
        parent.chmod(0o700)


def test_large_halfway_coordinate_frame(tmp_path, capsys):
    # PyMatching 2.4.0 macOS arm64 audit fixture: this creates the f32 value 1000005.
    raw = b"detector(0,0) D0\ndetector(9896,0) D1\nerror(0.1) D0 D1\nerror(0.1) D0\nerror(0.1) D1\n"
    args = _edge_args(tmp_path, raw)
    args[args.index("--pixels_per_unit_length") + 1] = "101"
    args[args.index("--held_frames_at_start") + 1] = "0"
    args[args.index("--max_growth_between_frames") + 1] = "50"
    (tmp_path / "edge.hits").write_bytes(b"1\n")
    assert _run(args, capsys) == (0, "")
    got = (tmp_path / "out" / "frame_00000.svg").read_text().splitlines()
    want = (FIXTURES / "large_tie_frame.svg").read_text().splitlines()
    assert _same_frame(sorted(got), sorted(want))
    if sys.platform == "darwin":
        assert 'cx="1.00000e+06"' in "\n".join(got)


def test_overflow_coordinates_have_bounded_success_without_byte_parity(tmp_path, capsys):
    # Native non-total NaN set ordering is deliberately not imitated or asserted equal.
    raw = b"error(0.1) D0 D1\nerror(0.1) D0\nerror(0.1) D1\ndetector(-1e39,0) D0\ndetector(1e39,0) D1\n"
    assert _run(_edge_args(tmp_path, raw), capsys) == (0, "")
    frames = _frames(tmp_path / "out")
    assert 0 < len(frames) < 1000
    assert "nan" in (tmp_path / "out" / frames[0]).read_text()

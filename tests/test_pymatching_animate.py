"""PyMatching 2.4's ``animate``: ``stabilizer_qec.pymatching.cli(command_line_args=["animate",
...])``, ``python -m stabilizer_qec.pymatching animate`` and ``stabilizer-qec animate``, against
the frames and messages recorded from PyMatching itself (tools/pymatching_animate_fixtures.py:
PyMatching 2.4.0's macOS arm64 wheel, and its C++ program for exit codes).

PyMatching draws its regions in heap-address order (``std::set<GraphFillRegion *>``), which no
port can reproduce, so a frame is compared as its lines sorted. The frame names, their count,
stderr and the return value must be PyMatching's.

The frames are exact on macOS arm64, where they were recorded. PyMatching's own wheels draw
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

import pytest

from stabilizer_qec import pymatching

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
    (where / "model.dem").write_text(rec["dem"])
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


OTHER = [n for n in MESSAGES if n not in REQUIRED and not n.startswith("pixels_") and n != "max_edge_weight" and MESSAGES[n]["wheel"].startswith("returned")]


@pytest.mark.parametrize("name", OTHER)
def test_misuse_messages(name, tmp_path, monkeypatch, capsys):
    """Unknown flags and words, values out of range or not integers, a format PyMatching does not
    know, a value for a switch, files that do not open, a file without a shot, ``--out_dir``
    with and without its slash, and ``--`` ending the flags."""
    _check_message(name, tmp_path, monkeypatch, capsys)


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

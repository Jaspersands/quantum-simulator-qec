"""PyMatching's module under this package: ``import stabilizer_qec.pymatching as pymatching``
gives ``Matching`` (this package's, which is PyMatching 2.4's API), ``set_seed``,
``randomize`` and ``rand_float``; ``python -m stabilizer_qec.pymatching`` is its command line
(``predict``, ``count_mistakes`` and ``animate``)."""

from __future__ import annotations

import os
import sys
from typing import List, Optional

from . import _matching as matching
from ._matching import Matching, rand_float, randomize, set_seed

# The PyMatching release whose API this is.
__version__ = "2.4.0"

__all__ = ["Matching", "cli", "matching", "rand_float", "randomize", "set_seed"]

_UNRECOGNIZED_COMMAND = (
    "Unrecognized command. Available commands are:\n"
    "    pymatching predict --dem file [--in file] [--out file] [--in_format 01|b8|...] [--out_format 01|b8|...] [--in_includes_appended_observables]\n"
    "    pymatching count_mistakes --dem file [--in file] [--out file] [--in_format 01|b8|...] [--out_format 01|B8|...] [--in_includes_appended_observables] [--obs_in] [--obs_in_format]\n"
    "    pymatching animate --dets_in <file> --dets_in_format 01|b8|... --out_dir <directory> --dem_in <file> --held_frames_per_event # --held_frames_at_start # --held_frames_at_end # --max_growth_between_frames # --max_edge_weight # --pixels_per_unit_length # [--dets_in_includes_appended_observables] [--quiet]"
)


def cli(*, command_line_args: list) -> int:
    """PyMatching's ``cli``: ``predict`` or ``count_mistakes`` with PyMatching's flags, or
    ``animate``, PyMatching's: the decoder's run on a model's first shot drawn as SVG frames
    (``frame_00000.svg``, ...) in ``--out_dir``, with PyMatching's flags, messages and frames.

    ``animate`` returns 1 after printing PyMatching's message to stderr for a refused flag or
    file, a shot it cannot read, a frame it cannot write, or a run with no perfect matching
    (``Unknown tentative event type.``, after the frames before it); an ``--out_dir`` it
    cannot create raises RuntimeError with the filesystem's message, as PyMatching's does.
    (PyMatching's usage line lists ``--max_edge_weight``, which its ``animate`` refuses.)
    """
    args = list(command_line_args)
    if args and args[0] == "animate":
        return _animate(args)
    if not args or args[0] not in ("predict", "count_mistakes"):
        raise ValueError(_UNRECOGNIZED_COMMAND)
    from ._cli import main

    return main(command_line_args=args)


# PyMatching 2.4.0's `animate` (diagram/animation_main.cc), with Stim's flag parsing
# (util_bot/arg_parse.cc, Stim 1320ad7e, the one PyMatching builds with). Copyright 2022
# PyMatching Contributors and Google LLC, Apache-2.0; transliterated to Python.

_ANIMATE_FLAGS = (
    "--dets_in",
    "--dets_in_format",
    "--dets_in_includes_appended_observables",
    "--out_dir",
    "--dem_in",
    "--held_frames_per_event",
    "--held_frames_at_start",
    "--held_frames_at_end",
    "--max_growth_between_frames",
    "--pixels_per_unit_length",
    "--quiet",
)
# Stim's format names, in its map's order.
_FORMATS = ("01", "b8", "dets", "hits", "ptb64", "r8")
_U64 = (1 << 64) - 1


class _InvalidArgument(ValueError):
    """``std::invalid_argument``: PyMatching's ``main`` prints it and returns 1."""


def _is_digit(c: str) -> bool:
    return len(c) == 1 and "0" <= c <= "9"


def _find_argument(name: str, argv: List[str]) -> Optional[str]:
    """Stim's ``find_argument``: the flag's value ('' for a flag alone), or None."""
    flag_count = 1
    while flag_count < len(argv) and argv[flag_count] != "--":  # "--" ends the flags
        flag_count += 1
    n = len(name)
    for i in range(1, flag_count):
        a = argv[i]
        if not a.startswith(name) or (len(a) > n and a[n] != "="):
            continue
        # Alone, and followed by the end or another flag (not a negative number): no value.
        if len(a) == n and (i == len(argv) - 1 or (argv[i + 1][:1] == "-" and not _is_digit(argv[i + 1][1:2]))):
            return ""
        if len(a) > n:
            return a[n + 1 :]
        return argv[i + 1]
    return None


def _check_for_unknown_arguments(known: tuple, mode: str, argv: List[str]) -> None:
    i = 1
    while i < len(argv):
        a = argv[i]
        if i == 1 and a == mode:
            i += 1
            continue
        if a == "--":
            break
        matched = False
        for k in known:
            n = len(k)
            if a.startswith(k) and (len(a) == n or a[n] == "="):
                # Skip a word that is the flag's value.
                if len(a) == n and i < len(argv) - 1 and argv[i + 1][:1] != "-":
                    i += 1
                matched = True
                break
        if not matched:
            msg = f"Unrecognized command line argument {a} for `stim {mode}`.\n"
            msg += f"Recognized command line arguments for `stim {mode}`:\n"
            msg += "".join(f"    {v}\n" for v in sorted(set(known)))
            raise _InvalidArgument(msg)
        i += 1


def _parse_int64(data: str) -> Optional[int]:
    """Stim's ``parse_int64`` (its unsigned accumulator wraps as C++'s does)."""
    if not data:
        return None
    negate = data.startswith("-")
    if negate or data.startswith("+"):
        data = data[1:]
    acc = 0
    for c in data.encode("utf-8", "surrogateescape"):
        if not 48 <= c <= 57:
            return None
        digit = c - 48
        nxt = (acc * 10 + digit) & _U64
        if acc != ((nxt - digit) & _U64) // 10:
            return None  # overflow
        acc = nxt
    if negate and acc == 1 << 63:
        return -(1 << 63)
    if acc > (1 << 63) - 1:
        return None
    return -acc if negate else acc


def _find_int64_argument(name: str, default: int, lo: int, hi: int, argv: List[str]) -> int:
    text = _find_argument(name, argv)
    if not text:
        if default < lo or default > hi:
            raise _InvalidArgument(f"Must specify a value for int flag '{name}'.")
        return default
    i = _parse_int64(text)
    if i is None:
        raise _InvalidArgument(f"Got non-int64 value '{text}' for int64 flag '{name}'.")
    if i < lo or i > hi:
        raise _InvalidArgument(f"Integer value '{text}' for flag '{name}' doesn't satisfy {lo} <= {i} <= {hi}.")
    return i


def _find_bool_argument(name: str, argv: List[str]) -> bool:
    text = _find_argument(name, argv)
    if text is None:
        return False
    if text == "":
        return True
    raise _InvalidArgument(f"Got non-empty value '{text}' for boolean flag '{name}'.")


def _find_open_file_argument(name: str, argv: List[str]) -> bytes:
    """The required file's bytes (Stim's ``find_open_file_argument``, then read whole)."""
    path = _find_argument(name, argv)
    if path is None:
        raise _InvalidArgument(f"Missing command line argument: '{name}'")
    if path == "":
        raise _InvalidArgument(f"Command line argument '{name}' can't be empty. It's supposed to be a file path.")
    try:
        with open(path, "rb") as f:
            return f.read()
    except OSError:
        raise _InvalidArgument(f"Failed to open '{path}'") from None


def _find_format_argument(name: str, argv: List[str]) -> str:
    text = _find_argument(name, argv)
    if text is None:
        raise _InvalidArgument(f"\033[31mMust specify a value for enum flag '{name}'.\n")
    if text not in _FORMATS:
        msg = f"\033[31mUnrecognized value '{text}' for enum flag '{name}'.\nRecognized values are:\n"
        msg += "".join(f"    '{v}'\n" for v in _FORMATS) + "\033[0m"
        raise _InvalidArgument(msg)
    return text


class _Stream:
    """A file's bytes read with ``getc`` (-1 at the end)."""

    def __init__(self, data: bytes) -> None:
        self.data, self.pos = data, 0

    def getc(self) -> int:
        if self.pos >= len(self.data):
            return -1
        self.pos += 1
        return self.data[self.pos - 1]


def _read_uint64(s: _Stream) -> tuple:
    """Stim's ``read_uint64``: (the integer or None, the character after it)."""
    nxt = s.getc()
    if not 48 <= nxt <= 57:
        return None, nxt
    value = 0
    while 48 <= nxt <= 57:
        prev = value
        value = (value * 10 + nxt - 48) & _U64
        if value < prev:
            raise RuntimeError("Integer value read from file was too big")  # std::runtime_error
        nxt = s.getc()
    return value, nxt


def _read_first_shot(data: bytes, fmt: str, nd: int, no: int) -> Optional[List[int]]:
    """Stim's ``MeasureRecordReader::start_and_read_entire_record`` into a ``SparseShot``, once:
    the first shot's detection events (in the file's order in hits and dets), or None for no
    shot. ``no`` is the number of observables appended to each shot."""
    s = _Stream(data)
    n = nd + no
    bits: List[int] = []
    if fmt == "hits":
        first = True
        while True:
            value, nxt = _read_uint64(s)
            if value is None:
                if first and nxt == -1:
                    return None
                if first and nxt == 13:
                    nxt = s.getc()
                if first and nxt == 10:
                    return bits
                raise _InvalidArgument("HITS data wasn't comma-separated integers terminated by a newline.")
            if value >= n:
                raise _InvalidArgument("hit index is too large.")
            if value < nd:
                bits.append(value)
            first = False
            if nxt == 13:
                nxt = s.getc()
                if nxt == 10:
                    return bits
            elif nxt == 10:
                return bits
            if nxt != 44:
                raise _InvalidArgument("HITS data wasn't comma-separated integers terminated by a newline.")
    if fmt == "dets":
        while True:
            c = s.getc()
            if c in (32, 10, 13, 9):
                continue
            if c == -1:
                return None
            if c != 115 or s.getc() != 104 or s.getc() != 111 or s.getc() != 116:
                raise _InvalidArgument("DETS data didn't start with 'shot'")
            break
        c = s.getc()
        while True:
            if c == 13:
                c = s.getc()
            if c in (10, -1):
                return bits
            if c != 32:
                raise _InvalidArgument("DETS data wasn't single-space-separated with no trailing spaces.")
            c = s.getc()
            if c == 77:  # M
                offset, length = 0, 0
            elif c == 68:  # D
                offset, length = 0, nd
            elif c == 76:  # L
                offset, length = nd, no
            else:
                raise _InvalidArgument(f"Unrecognized DETS prefix. Expected M or D or L not '{c}'")
            prefix = chr(c)
            value, c = _read_uint64(s)
            if value is None:
                raise _InvalidArgument("DETS data had a value prefix (M or D or L) not followed by an integer.")
            if value >= length:
                raise _InvalidArgument(f"DETS data had a value larger than expected. Got {prefix}{value} but expected length of {prefix} space to be {length}.")
            if offset + value < nd:
                bits.append(offset + value)
    if fmt == "01":
        ok = True
        for k in range(n):
            b = s.getc()
            if b == 48:
                continue
            if b == 49:
                bits.append(k)
                continue
            if b == -1 and k == 0:
                ok = False
                break
            if b in (-1, 13, 10):
                raise _InvalidArgument(f"01 data ended in middle of record at byte position {k}.\nExpected bits per record was {n}.")
            raise _InvalidArgument(f"Unexpected character in 01 format data: '{b}'.")
        else:
            last = s.getc()
            if n == 0 and last == -1:
                ok = False
            else:
                if last == 13:
                    last = s.getc()
                if last != 10:
                    raise _InvalidArgument(f"01 data didn't end with a newline after the expected data length of '{n}'.")
    elif fmt == "r8":
        c = s.getc()
        ok = c != -1
        pos = 0
        while ok:
            pos += c
            if c != 255:
                if pos < n:
                    bits.append(pos)
                    pos += 1
                elif pos == n:
                    break
                else:
                    raise _InvalidArgument(f"r8 data jumped past expected end of encoded data. Expected to decode {n} bits.")
            c = s.getc()
            if c == -1:
                raise _InvalidArgument(f"End of file before end of r8 data. Expected to decode {n} bits.")
    elif fmt == "b8":
        nb = (n + 7) >> 3
        if n == 0 or not data:
            return None
        if len(data) < nb:
            raise _InvalidArgument(f"b8 data ended in middle of record at byte position {len(data)}.\nExpected bytes per record was {nb} ({n} bits padded).")
        bits = [8 * k + r for k in range(nb) for r in range(8) if data[k] >> r & 1]
        ok = True
    else:  # ptb64: the first group of 64 shots, and its first shot
        nb = n * 8
        if not data[:nb]:
            return None
        if len(data) < nb:
            raise _InvalidArgument(f"ptb64 data ended in middle of 64 record group at byte position {len(data)}.\nExpected bytes per 64 records was {nb} ({n} bits padded).")
        bits = [k for k in range(n) if data[8 * k] & 1]
        ok = True
    # move_obs_in_shots_to_mask_assuming_sorted
    if no > 32:
        raise _InvalidArgument("More than 32 observables. Can't read into SparseShot struct.")
    while bits and bits[-1] >= nd:
        if bits[-1] >= n:
            raise _InvalidArgument("Hit index from data is too large.")
        bits.pop()
    return bits if ok else None


def _filesystem_error(op: str, ex: OSError, path: str) -> RuntimeError:
    """The ``std::filesystem_error`` PyMatching's program aborts on (its ``cli``, RuntimeError),
    worded as the platform's C++ library words it, with the reason from the OS's own errno for the
    same ``mkdir``: libstdc++ on Linux (where ``mkdir("file/")`` fails EEXIST; macOS fails it
    ENOTDIR), libc++ elsewhere. PyMatching's Windows wheel is built with MSVC, whose STL words it
    ``create_directory: <Win32 message>: "<path>"``; that wording is not reproduced, and Windows
    gets libc++'s."""
    reason = os.strerror(ex.errno) if ex.errno else str(ex)
    if sys.platform.startswith("linux"):
        return RuntimeError(f"filesystem error: cannot {op.replace('_', ' ')}: {reason} [{path}]")
    return RuntimeError(f'filesystem error: in {op}: {reason} ["{path}"]')


def _animate(argv: List[str]) -> int:
    """``pymatching animate ...`` (``argv[0]`` is ``animate``): PyMatching's ``main_animation``
    under its ``main``, which prints a refusal and returns 1."""
    try:
        _animate_or_raise(["pymatching"] + list(argv))
    except ValueError as ex:
        print(ex, file=sys.stderr)
        return 1
    return 0


def _animate_or_raise(argv: List[str]) -> None:
    from . import _core
    from ._circuit import DetectorErrorModel
    from ._util import call

    _check_for_unknown_arguments(_ANIMATE_FLAGS, "animate", argv)
    dets_raw = _find_open_file_argument("--dets_in", argv)
    dem_raw = _find_open_file_argument("--dem_in", argv)
    fmt = _find_format_argument("--dets_in_format", argv)
    dets_include_obs = _find_bool_argument("--dets_in_includes_appended_observables", argv)
    quiet = _find_bool_argument("--quiet", argv)
    per_event = _find_int64_argument("--held_frames_per_event", -1, 0, 1000000, argv)
    at_start = _find_int64_argument("--held_frames_at_start", -1, 0, 1000000, argv)
    at_end = _find_int64_argument("--held_frames_at_end", -1, 0, 1000000, argv)
    max_growth = _find_int64_argument("--max_growth_between_frames", -1, 0, 1000000, argv)
    pixels = _find_int64_argument("--pixels_per_unit_length", -1, 4, 2048, argv)
    out_dir = _find_argument("--out_dir", argv)
    if out_dir is None:
        raise _InvalidArgument("\033[31mMissing command line argument: '--out_dir'")
    if not out_dir.endswith("/") and not out_dir.endswith("\\"):
        out_dir += "/"

    dem = DetectorErrorModel(dem_raw.decode("utf-8"))
    nd, no = dem.num_detectors, dem.num_observables
    hits = _read_first_shot(dets_raw, fmt, nd, no if dets_include_obs else 0)
    if hits is None:
        raise _InvalidArgument("No shot in file.")

    coords = dem.get_detector_coordinates()
    animation = call(_core.PmAnimation, str(dem), [[float(c) for c in coords.get(k, [])] for k in range(nd)], pixels)

    if not os.path.exists(out_dir):
        try:
            os.mkdir(out_dir)
        except OSError as ex:
            raise _filesystem_error("create_directory", ex, out_dir) from None

    written = [0]

    def emit(svg: str) -> None:
        path = f"{out_dir}frame_{written[0]:05d}.svg"
        try:
            f = open(path, "w", encoding="utf-8")  # text mode, as PyMatching's ofstream
        except OSError:
            raise _InvalidArgument(f"Failed to open {path} to write.") from None
        with f:
            f.write(svg)
        if not quiet:
            print(f"wrote {path}", file=sys.stderr)
        written[0] += 1

    call(animation.run, hits, per_event, at_start, at_end, max_growth, emit)


if __name__ == "__main__":
    sys.exit(cli(command_line_args=sys.argv[1:]))

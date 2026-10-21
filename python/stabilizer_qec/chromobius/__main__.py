# Ported from Chromobius 1.1.1 (quantumlib/chromobius: src/main.cc, src/chromobius/commands/;
# Copyright 2023 Google LLC, Apache License 2.0) and the Stim it pins (quantumlib/Stim at 1320ad7:
# src/stim/util_bot/arg_parse.cc, src/stim/io/measure_record_reader.inl and
# measure_record_writer.cc; Copyright 2021 Google LLC, Apache License 2.0); see
# THIRD_PARTY_NOTICES.md. Modified for stabilizer_qec: the decoder is this package's port.
"""``python -m stabilizer_qec.chromobius``: Chromobius's command-line program (``chromobius``).

    python -m stabilizer_qec.chromobius predict --dem model.dem --in shots.b8 --out predictions.01
    python -m stabilizer_qec.chromobius benchmark --dem model.dem --in shots.01 --in_includes_appended_observables
    python -m stabilizer_qec.chromobius describe_decoder --circuit circuit.stim

The same commands, flags, defaults, outputs and messages as Chromobius's program, whose flags are
Stim's (a flag's value follows it or ``=``; ``--`` ends the flags; ``--in`` and ``--out`` default
to standard input and output), checked against it in tests/test_chromobius_cli.py. Where it
differs:

- A malformed model or circuit is refused with this package's message, not Stim's parser's.
- Records are read whole before any is decoded (Chromobius streams them); a bad record still
  stops the command with Stim's message after the predictions of the records before it.
- ``predict`` decodes on every core (its predictions are the same at any thread count);
  ``benchmark`` on one, as Chromobius does, so its timings compare.
- A shot no lifting explains (a model with wrong colour annotations) is refused with
  Chromobius's message less its list of the failing cycle's detectors.
- A hit or DETS index past 2⁶⁴ − 1, an exception Chromobius's program does not catch (it
  aborts), prints the C++ library's message and exits with status 134 without raising SIGABRT.
"""

from __future__ import annotations

import os
import re
import sys
import time
from typing import Any, List, Optional, Tuple

import numpy as _np

HELP = r"""

Available chromobius commands:

    # Print usage information.
    chromobius help

    # Predict observable flips from detection event data.
    chromobius predict \
        [--dem FILEPATH] \                     # where to read detector error model from
        [--in] \                               # where to read detection event data (defaults to stdin)
        [--in_format 01|b8|...] \              # format of input detection event data
        [--in_includes_appended_observables] \ # if set, input data includes observables as extra detectors to ignore
        [--out FILEPATH] \                     # where to write predictions to (defaults to stdout)
        [--out_format 01|b8|...]               # format to use when writing predictions

    # Print accuracy and timing statistics collected while decoding.
    chromobius benchmark
        [--dem FILEPATH] \                     # where to read detector error model from
        [--in] \                               # where to read detection event data (defaults to stdin)
        [--in_format 01|b8|...] \              # format of input detection event data
        [--in_includes_appended_observables] \ # if set, observables are extra detectors in detection event data
        [--obs_in FILEPATH] \                  # if set, observables are read from a separate file
        [--obs_in_format 01|b8|...] \          # format of separate observable data
        [--out FILEPATH]                       # where to write results (defaults to stdout)

    # Describes the internal representations used to decode a given dem or circuit.
    chromobius describe_decoder \
        [--in] \           # where to read a detector error model from (defaults to stdin)
        [--circuit] \      # where to read a circuit from (overrides --in)
        [--out FILEPATH]   # where to write output (defaults to stdout)
"""

# Stim's sample formats, by name (``format_name_to_enum_map``, a sorted std::map).
FORMATS = ("01", "b8", "dets", "hits", "ptb64", "r8")


class _Fail(Exception):
    """A std::invalid_argument: Chromobius's program prints it and exits with status 1."""


class _Abort(Exception):
    """An exception Chromobius's program does not catch, so the C++ runtime aborts."""


# -- Stim's command-line flags (arg_parse.cc) -----------------------------------------------


def _is_digit(s: str, k: int) -> bool:
    return k < len(s) and "0" <= s[k] <= "9"


def _find_argument(name: str, argv: List[str]) -> Optional[str]:
    """``find_argument``: the flag's value, "" for a flag given no value, None when absent."""
    flags = 1
    while flags < len(argv) and argv[flags] != "--":
        flags += 1
    n = len(name)
    for i in range(1, flags):
        a = argv[i]
        if not a.startswith(name) or (len(a) > n and a[n] != "="):
            continue
        if len(a) == n:
            # Alone and followed by the end or another flag: no value.
            if i == len(argv) - 1 or (argv[i + 1][:1] == "-" and not _is_digit(argv[i + 1], 1)):
                return ""
            return argv[i + 1]
        return a[n + 1:]
    return None


def _check_for_unknown_arguments(known: Tuple[str, ...], mode: str, argv: List[str]) -> None:
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
            if a.startswith(k) and (len(a) == len(k) or a[len(k)] == "="):
                # A word after a flag given alone is its value.
                if len(a) == len(k) and i < len(argv) - 1 and argv[i + 1][:1] != "-":
                    i += 1
                matched = True
                break
        if not matched:
            lines = "".join(f"    {k}\n" for k in sorted(known))
            raise _Fail(f"Unrecognized command line argument {a} for `stim {mode}`.\nRecognized command line arguments for `stim {mode}`:\n{lines}")
        i += 1


def _find_bool_argument(name: str, argv: List[str]) -> bool:
    text = _find_argument(name, argv)
    if text is None:
        return False
    if text == "":
        return True
    raise _Fail(f"Got non-empty value '{text}' for boolean flag '{name}'.")


def _find_format_argument(name: str, default: str, argv: List[str]) -> str:
    text = _find_argument(name, argv)
    if text is None:
        return default
    if text not in FORMATS:
        values = "".join(f"    '{k}'{' (default)' if k == default else ''}\n" for k in FORMATS)
        raise _Fail(f"\033[31mUnrecognized value '{text}' for enum flag '{name}'.\nRecognized values are:\n{values}\033[0m")
    return text


class _Input:
    """A file opened for reading (``find_open_file_argument`` with mode "rb"), read later; or
    standard input. A directory opens, as C's ``fopen`` opens one, and reads as empty."""

    def __init__(self, path: Optional[str]) -> None:
        self.path, self.file = path, None
        if path is not None and not os.path.isdir(path):
            try:
                self.file = open(path, "rb")
            except OSError:
                raise _Fail(f"Failed to open '{path}'") from None

    def read(self) -> bytes:
        if self.path is None:
            return sys.stdin.buffer.read()
        if self.file is None:
            return b""
        with self.file:
            return self.file.read()


def _find_open_file_argument(name: str, stdin_default: bool, argv: List[str]) -> _Input:
    path = _find_argument(name, argv)
    if path is None:
        if not stdin_default:
            raise _Fail(f"Missing command line argument: '{name}'")
        return _Input(None)
    if path == "":
        raise _Fail(f"Command line argument '{name}' can't be empty. It's supposed to be a file path.")
    return _Input(path)


def _find_output_argument(name: str, argv: List[str]) -> Any:
    """``find_open_file_argument`` with mode "wb" (or ``find_output_stream_argument``): the file,
    created or emptied now, or None for standard output."""
    path = _find_argument(name, argv)
    if path is None:
        return None
    if path == "":
        raise _Fail(f"Command line argument '{name}' can't be empty. It's supposed to be a file path.")
    try:
        return open(path, "wb")
    except OSError:
        raise _Fail(f"Failed to open '{path}'") from None


# -- Stim's record readers (measure_record_reader.inl) ----------------------------------------
#
# Each reads every record it can and returns them (bit-packed little-endian; a b8 record's
# padding bits are what the file holds) with the exception that stopped it after them (None at
# the end of the data).

Records = Tuple[_np.ndarray, Optional[Exception]]
_BAD01 = re.compile(rb"[^01]")


def _pack(bits: _np.ndarray) -> _np.ndarray:
    return _np.packbits(bits.astype(bool), axis=1, bitorder="little")


def _read_01(data: bytes, n: int) -> Records:
    w = n + 1
    if len(data) % w == 0:
        a = _np.frombuffer(data, dtype=_np.uint8).reshape(-1, w)
        if (a[:, n] == 10).all() and ((a[:, :n] == 48) | (a[:, :n] == 49)).all():
            return _pack(a[:, :n] == 49), None
    rows: List[bytes] = []
    pos, end = 0, len(data)
    err: Optional[Exception] = None
    while True:
        chunk = data[pos:pos + n]
        m = _BAD01.search(chunk)
        if m:
            k, b = m.start(), chunk[m.start()]
            if b in (10, 13):
                err = _Fail(f"01 data ended in middle of record at byte position {k}.\nExpected bits per record was {n}.")
            else:
                err = _Fail(f"Unexpected character in 01 format data: '{b}'.")
            break
        if len(chunk) < n:
            if chunk:
                err = _Fail(f"01 data ended in middle of record at byte position {len(chunk)}.\nExpected bits per record was {n}.")
            break
        pos += n
        last = data[pos] if pos < end else -1
        if n == 0 and last == -1:
            break
        pos += 1
        if last == 13:
            last = data[pos] if pos < end else -1
            pos += 1
        if last != 10:
            err = _Fail(f"01 data didn't end with a newline after the expected data length of '{n}'.")
            break
        rows.append(chunk)
    bits = _np.frombuffer(b"".join(rows), dtype=_np.uint8).reshape(len(rows), n) == 49
    return _pack(bits), err


def _read_b8(data: bytes, n: int) -> Records:
    nb = (n + 7) // 8
    if nb == 0:
        return _np.zeros((0, 0), _np.uint8), None
    k, rest = divmod(len(data), nb)
    rows = _np.frombuffer(data[: k * nb], dtype=_np.uint8).reshape(k, nb).copy()
    err = None
    if rest:
        err = _Fail(f"b8 data ended in middle of record at byte position {rest}.\nExpected bytes per record was {nb} ({n} bits padded).")
    return rows, err


def _read_ptb64(data: bytes, n: int) -> Records:
    nb = 8 * n
    if nb == 0:
        return _np.zeros((0, 0), _np.uint8), None
    k, rest = divmod(len(data), nb)
    blocks = _np.frombuffer(data[: k * nb], dtype=_np.uint8).reshape(k, n, 8)
    bits = _np.unpackbits(blocks, axis=2, bitorder="little").transpose(0, 2, 1).reshape(-1, n)
    err = None
    if rest:
        err = _Fail(f"ptb64 data ended in middle of 64 record group at byte position {rest}.\nExpected bytes per 64 records was {nb} ({n} bits padded).")
    return _pack(bits), err


class _Bytes:
    """``getc`` on a byte string: -1 past the end (EOF)."""

    __slots__ = ("data", "pos")

    def __init__(self, data: bytes) -> None:
        self.data, self.pos = data, 0

    def getc(self) -> int:
        p = self.pos
        self.pos = p + 1
        return self.data[p] if p < len(self.data) else -1

    def read_uint64(self) -> Tuple[Optional[int], int]:
        """``read_uint64``: (the integer, the character after it), or (None, the character)."""
        c = self.getc()
        if not 48 <= c <= 57:
            return None, c
        value = 0
        while 48 <= c <= 57:
            value = value * 10 + c - 48
            if value >= 1 << 64:
                raise _Abort("std::runtime_error: Integer value read from file was too big")
            c = self.getc()
        return value, c


def _hits_record(src: _Bytes, n: int, nd: int, no: int, hits: List[int]) -> bool:
    bad = "HITS data wasn't comma-separated integers terminated by a newline."
    first = True
    while True:
        value, c = src.read_uint64()
        if value is None:
            if first and c == -1:
                return False
            if first and c == 13:
                c = src.getc()
            if first and c == 10:
                return True
            raise _Fail(bad)
        if value >= n:
            raise _Fail("hit index is too large.")
        hits.append(value)
        first = False
        if c == 13:
            c = src.getc()
            if c == 10:
                return True
        elif c == 10:
            return True
        if c != 44:
            raise _Fail(bad)


def _dets_record(src: _Bytes, n: int, nd: int, no: int, hits: List[int]) -> bool:
    while True:
        c = src.getc()
        if c in (32, 10, 13, 9):
            continue
        if c == -1:
            return False
        if c != 115 or src.getc() != 104 or src.getc() != 111 or src.getc() != 116:
            raise _Fail("DETS data didn't start with 'shot'")
        break
    c = src.getc()
    while True:
        if c == 13:
            c = src.getc()
        if c in (10, -1):
            return True
        if c != 32:
            raise _Fail("DETS data wasn't single-space-separated with no trailing spaces.")
        c = src.getc()
        if c == 77:
            offset, length = 0, 0
        elif c == 68:
            offset, length = 0, nd
        elif c == 76:
            offset, length = nd, no
        else:
            raise _Fail(f"Unrecognized DETS prefix. Expected M or D or L not '{c}'")
        prefix = chr(c)
        value, c = src.read_uint64()
        if value is None:
            raise _Fail("DETS data had a value prefix (M or D or L) not followed by an integer.")
        if value >= length:
            raise _Fail(f"DETS data had a value larger than expected. Got {prefix}{value} but expected length of {prefix} space to be {length}.")
        hits.append(-1 - (offset + value))  # set, not toggled (see _read_sparse)


def _r8_record(src: _Bytes, n: int, nd: int, no: int, hits: List[int]) -> bool:
    c = src.getc()
    if c == -1:
        return False
    pos = 0
    while True:
        pos += c
        if c != 255:
            if pos < n:
                hits.append(pos)
                pos += 1
            elif pos == n:
                return True
            else:
                raise _Fail(f"r8 data jumped past expected end of encoded data. Expected to decode {n} bits.")
        c = src.getc()
        if c == -1:
            raise _Fail(f"End of file before end of r8 data. Expected to decode {n} bits.")


def _read_sparse(data: bytes, fmt: str, nd: int, no: int) -> Records:
    """The hits, DETS and r8 readers, record by record as Stim's: each record's set bits (a hit
    given twice toggles twice; a DETS index, written ``-1 - k``, sets)."""
    n = nd + no
    src = _Bytes(data)
    rows: List[List[int]] = []
    err: Optional[Exception] = None
    read = {"hits": _hits_record, "dets": _dets_record, "r8": _r8_record}[fmt]
    try:
        while True:
            hits: List[int] = []
            if not read(src, n, nd, no, hits):
                break
            rows.append(hits)
    except (_Fail, _Abort) as e:
        err = e
    out = _np.zeros((len(rows), n), dtype=bool)
    for s, hits in enumerate(rows):
        for b in hits:
            if b < 0:
                out[s, -1 - b] = True
            else:
                out[s, b] ^= True
    return _pack(out), err


def _read_records(data: bytes, fmt: str, nd: int, no: int) -> Records:
    """The records of a ``fmt`` file of ``nd`` detectors then ``no`` observables."""
    if fmt == "01":
        return _read_01(data, nd + no)
    if fmt == "b8":
        return _read_b8(data, nd + no)
    if fmt == "ptb64":
        return _read_ptb64(data, nd + no)
    return _read_sparse(data, fmt, nd, no)


def _clear_bits(rows: _np.ndarray, start: int, stop: int) -> None:
    for k in range(start, stop):
        rows[:, k // 8] &= 0xFF ^ (1 << (k % 8))


def _first64(rows: _np.ndarray) -> _np.ndarray:
    """Each packed row's first 64 bits as a u64."""
    first8 = _np.zeros((len(rows), 8), _np.uint8)
    w = min(8, rows.shape[1])
    first8[:, :w] = rows[:, :w]
    return first8.view("<u8").reshape(-1)


# -- Commands ---------------------------------------------------------------------------------


def _model(data: bytes) -> Any:
    from .._circuit import DetectorErrorModel

    try:
        return DetectorErrorModel(data.decode("utf-8", "replace"))
    except ValueError as e:
        raise _Fail(str(e)) from None


def _decoder(dem: Any) -> Any:
    from .._decoders import ColorMatching

    try:
        return ColorMatching(dem)
    except ValueError as e:
        raise _Fail(str(e)) from None


def _decode(decoder: Any, rows: _np.ndarray, nd: int, threads: int) -> Tuple[_np.ndarray, Optional[_Fail]]:
    """Decode records whose first ``nd`` bits are detectors, as Chromobius's loop decodes them one
    by one: the predictions (u64 masks) of the records before the first that fails, and that
    failure (None if none fails)."""
    from .._decoders import _COLOR_UNMATCHABLE, _NO_PERFECT_MATCHING

    nb = (nd + 7) // 8
    stop, err = len(rows), None
    # A bit set past the last detector (a b8 record's padding, an observable not cleared) reaches
    # PyMatching as two Möbius detection events past its nodes, which it refuses.
    if len(rows) and rows.shape[1] * 8 > nd:
        extra = rows.copy()
        _clear_bits(extra, 0, nd)
        bad = _np.flatnonzero(extra.any(axis=1))
        if bad.size:
            stop = int(bad[0])
            d = int(_np.flatnonzero(_np.unpackbits(extra[stop], bitorder="little"))[0])
            err = _Fail(f"The detection event with index {2 * d} does not correspond to a node in the graph, which only has {2 * nd} nodes.")
    dets = _np.ascontiguousarray(rows[:stop, :nb])
    preds, _weights, status, failed = decoder._x.decode_batch(dets.tobytes(), stop, threads)
    preds = _np.frombuffer(preds, dtype="<u8")
    if failed:
        stop = int(failed[0])
        # PyMatching throws its message where the Möbius matching has no perfect matching (a
        # detector no fault touches fired, or a component without a boundary has odd parity);
        # Chromobius's own message is for a matching it then cannot lift.
        err = _Fail(_NO_PERFECT_MATCHING if status[stop] == _COLOR_UNMATCHABLE else _unliftable(decoder, dets[stop], nd))
    return preds[:stop], err


def _unliftable(decoder: Any, shot: _np.ndarray, nd: int) -> str:
    """Chromobius's message for a shot no lifting explains, less the failing cycle."""
    colors = ["NEUTRALUNKNOWN_BASIS"] * nd
    for d, coords in decoder._made_from[0][0].get_detector_coordinates().items():
        if len(coords) > 3 and coords[3] in (0, 1, 2, 3, 4, 5):
            c = int(coords[3])
            colors[d] = "RGB"[c % 3] + "XZ"[c // 3]
    fired = _np.flatnonzero(_np.unpackbits(shot, bitorder="little")[:nd])
    events = "".join(f"\n    D{d}[{colors[d]}, triggered]" for d in fired)
    return (
        "Failed to lift a flattened edge cycle from the matcher into an explanation of the detection events in the cycle.\n"
        "This error could be due to a coloring error in the model used to configure the decoder, or a bug in the decoder.\n"
        f"All detection events in the shot: {{{events}\n}}"
    )


def _bits(preds: _np.ndarray, width: int) -> _np.ndarray:
    """u64 masks as a (shots, width) bool array."""
    b = _np.unpackbits(preds.astype("<u8").view(_np.uint8).reshape(-1, 8), axis=1, bitorder="little").astype(bool)
    if width <= 64:
        return b[:, :width]
    return _np.hstack([b, _np.zeros((len(preds), width - 64), dtype=bool)])


def _predict(argv: List[str], stdout: Any) -> None:
    _check_for_unknown_arguments(("--in", "--in_format", "--in_includes_appended_observables", "--out", "--out_format", "--dem"), "predict", argv)
    shots_in = _find_open_file_argument("--in", True, argv)
    out = _find_output_argument("--out", argv)
    try:
        dem_in = _find_open_file_argument("--dem", False, argv)
        in_format = _find_format_argument("--in_format", "b8", argv)
        out_format = _find_format_argument("--out_format", "01", argv)
        append = _find_bool_argument("--in_includes_appended_observables", argv)
        dem = _model(dem_in.read())
        decoder = _decoder(dem)
        nd, no = dem.num_detectors, dem.num_observables
        if out_format == "ptb64":
            raise _Fail("SAMPLE_FORMAT_PTB64 incompatible with SingleMeasurementRecord")
        rows, read_err = _read_records(shots_in.read(), in_format, nd, no if append else 0)
        if append:
            _clear_bits(rows, nd, nd + no)
        preds, err = _decode(decoder, rows, nd, 0)
        from .._shots import encode_shots

        (out or stdout).write(encode_shots(_bits(preds, no), out_format, num_observables=no))
        if err or read_err:
            raise err or read_err
    finally:
        if out is not None:
            out.close()


def _benchmark(argv: List[str], stdout: Any) -> None:
    t0 = time.perf_counter()
    _check_for_unknown_arguments(("--in", "--in_format", "--in_includes_appended_observables", "--obs_in", "--obs_in_format", "--out", "--dem"), "benchmark", argv)
    shots_in = _find_open_file_argument("--in", True, argv)
    obs_in = _find_open_file_argument("--obs_in", False, argv) if _find_argument("--obs_in", argv) is not None else None
    out = _find_output_argument("--out", argv)
    try:
        dem_in = _find_open_file_argument("--dem", False, argv)
        in_format = _find_format_argument("--in_format", "01", argv)
        obs_format = _find_format_argument("--obs_in_format", "01", argv)
        append = _find_bool_argument("--in_includes_appended_observables", argv)
        if not append and obs_in is None:
            raise _Fail("Must specify --in_includes_appended_observables or --obs_in.")
        dem = _model(dem_in.read())
        nd, no = dem.num_detectors, dem.num_observables
        decoder = _decoder(dem)
        t1 = time.perf_counter()

        rows, read_err = _read_records(shots_in.read(), in_format, nd, no if append else 0)
        if obs_in is None:
            # The observables appended to each record, then cleared from it.
            actual = _first64(_pack(_np.unpackbits(rows, axis=1, bitorder="little", count=nd + no)[:, nd:]))
            _clear_bits(rows, nd, nd + no)
            num_obs_records, obs_err = len(rows) + 1, None
        else:
            # The separate file's: a mistake is a prediction differing from a record's first 64
            # bits (padding bits included, as Chromobius compares them).
            obs_rows, obs_read_err = _read_records(obs_in.read(), obs_format, 0, no)
            actual = _first64(obs_rows)
            num_obs_records, obs_err = len(obs_rows), obs_read_err or _Fail("Obs data ended before shot data ended.")
        # Record k is read (or its reader fails, or the data ends), then its observables (or
        # their reader fails, or their data ends), then decoded (or its lifting fails).
        preds, decode_err = _decode(decoder, rows[: min(len(rows), num_obs_records)], nd, 1)
        events = [(len(rows), 0, read_err), (num_obs_records, 1, obs_err)]
        if decode_err is not None:
            events.append((len(preds), 2, decode_err))
        first = min(events, key=lambda e: e[:2])[2]
        if first is not None:
            raise first
        num_shots = len(rows)
        num_mistakes = int((actual[:num_shots] != preds).sum())
        num_events = int(_np.unpackbits(rows, axis=1).sum())
        t2 = time.perf_counter()
        decoding_us = float(int((t2 - t1) * 1e6))
        config_us = float(int((t1 - t0) * 1e6))
        text = (
            f"                                num_shots = {num_shots}\n"
            f"                             num_mistakes = {num_mistakes}\n"
            f"                        mistakes_per_shot = {_g(num_mistakes / num_shots if num_shots else 0.0)}\n"
            "\n"
            f"                     num_detection_events = {num_events}\n"
            f"                   num_detectors_per_shot = {nd}\n"
            f"                       detection_fraction = {_g(_div(num_events, nd * num_shots))}\n"
            "\n"
            f"                            setup_seconds = {_g(config_us / 1000000)}\n"
            f"                         decoding_seconds = {_g(decoding_us / 1000000)}\n"
            f"           decoding_microseconds_per_shot = {_g(_div(decoding_us, num_shots))}\n"
            f"decoding_microseconds_per_detection_event = {_g(_div(decoding_us, num_events))}\n"
        )
        (out or stdout).write(text.encode())
    finally:
        if out is not None:
            out.close()


def _div(a: float, b: float) -> float:
    """``a / b`` as doubles divide (b = 0 gives inf, or nan for 0 / 0)."""
    if b:
        return a / b
    return float("nan") if a == 0 else float("inf")


def _g(x: float) -> str:
    """A double as a C++ stream prints one by default (``%g``)."""
    if x != x:
        return "nan"
    if x == float("inf"):
        return "inf"
    return "%g" % x


def _describe_decoder(argv: List[str], stdout: Any) -> None:
    _check_for_unknown_arguments(("--in", "--out", "--circuit"), "describe_decoder", argv)
    out = _find_output_argument("--out", argv)
    try:
        if _find_argument("--circuit", argv) is not None:
            from .._circuit import Circuit

            data = _find_open_file_argument("--circuit", False, argv).read()
            try:
                circuit = Circuit(data.decode("utf-8", "replace"))
                dem = circuit.detector_error_model(decompose_errors=False, flatten_loops=False, allow_gauge_detectors=False, approximate_disjoint_errors=True)
            except ValueError as e:
                raise _Fail(str(e)) from None
        else:
            dem = _model(_find_open_file_argument("--in", True, argv).read())
        text = _decoder(dem)._x.describe()
        (out or stdout).write(text.encode() + b"\n")
    finally:
        if out is not None:
            out.close()


def main(argv: Optional[List[str]] = None) -> int:
    """Chromobius's ``main``: run the command ``argv[1]`` (``argv`` as C's, the program first;
    ``sys.argv`` by default) and return the exit status."""
    argv = ["chromobius", *sys.argv[1:]] if argv is None else list(argv)
    stdout = sys.stdout.buffer
    command = argv[1] if len(argv) >= 2 else ""
    try:
        try:
            if command == "describe_decoder":
                _describe_decoder(argv, stdout)
            elif command == "predict":
                _predict(argv, stdout)
            elif command == "benchmark":
                _benchmark(argv, stdout)
            elif command in ("help", "--help", "-help", "-h"):
                stdout.write(HELP.encode())
            elif command == "":
                raise _Fail("Specify a chromobius command to run.\n" + HELP)
            else:
                raise _Fail(f"Unrecognized chromobius command '{command}'.\n" + HELP)
        finally:
            stdout.flush()
    except _Fail as e:
        sys.stderr.write(f"{e}\n")
        sys.stderr.flush()
        return 1
    except _Abort as e:
        # The C++ runtime's message; Chromobius's program then aborts, which a shell reports as
        # status 134.
        sys.stderr.write(f"libc++abi: terminating due to uncaught exception of type {e}\n")
        sys.stderr.flush()
        return 134
    return 0


if __name__ == "__main__":
    sys.exit(main())

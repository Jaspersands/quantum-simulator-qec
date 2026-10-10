"""Shot data files in Stim's six result formats, read and written as Stim reads and writes them
(``stim.write_shot_data_file`` / ``stim.read_shot_data_file``, the same names and arguments):

- ``01``: a line of ``0``/``1`` characters per shot.
- ``b8``: each shot's bits packed little-endian into ⌈n/8⌉ bytes.
- ``r8``: each shot as the run lengths of zeros before each 1, a 1 implied after its last bit,
  255 meaning "255 zeros, and the run goes on".
- ``ptb64``: shots in blocks of 64, each bit of the block as one 8-byte word of 64 shots.
- ``hits``: a line per shot of the indices of its 1s, comma separated.
- ``dets``: a line per shot, ``shot`` then each 1 named ``M``/``D``/``L`` and its index.

A shot's bits are its measurements (``num_measurements``), or its detectors then its
observables (``num_detectors``, ``num_observables``).
"""

from __future__ import annotations

from os import PathLike
from typing import Any, Union

import numpy as np

from ._util import count

FORMATS = ("01", "b8", "r8", "ptb64", "hits", "dets")


def _format(format: str) -> str:
    if format not in FORMATS:
        raise ValueError(f"format must be one of {', '.join(FORMATS)}, not {format!r}")
    return format


def _width(num_measurements: int, num_detectors: int, num_observables: int) -> int:
    m, d, o = count(num_measurements, "num_measurements"), count(num_detectors, "num_detectors"), count(num_observables, "num_observables")
    if m and (d or o):
        raise ValueError("a shot file holds measurements, or detectors and observables, not both (as in Stim)")
    return m + d + o


def _names(num_measurements: int, num_detectors: int, num_observables: int) -> list:
    return [f"M{k}" for k in range(num_measurements)] + [f"D{k}" for k in range(num_detectors)] + [f"L{k}" for k in range(num_observables)]


def encode_shots(data: np.ndarray, format: str, num_measurements: int = 0, num_detectors: int = 0, num_observables: int = 0, names: Any = None) -> bytes:
    """Shots (a (shots, n) bool array) as the bytes of a ``format`` file. ``names`` overrides
    the ``dets`` format's column names (observables written before detectors, say)."""
    _format(format)
    n = _width(num_measurements, num_detectors, num_observables)
    a = np.asarray(data, dtype=bool)
    if a.ndim != 2 or a.shape[1] != n:
        raise ValueError(f"data must be (shots, {n}) for {num_measurements} measurements, {num_detectors} detectors and {num_observables} observables, not {a.shape}")
    shots = a.shape[0]
    if format == "01":
        lines = np.where(a, ord("1"), ord("0")).astype(np.uint8)
        return np.hstack([lines, np.full((shots, 1), ord("\n"), dtype=np.uint8)]).tobytes()
    if format == "b8":
        return np.packbits(a, axis=1, bitorder="little").tobytes()
    if format == "ptb64":
        if shots % 64:
            raise ValueError("shots must be a multiple of 64 to use ptb64 format.")
        blocks = a.reshape(shots // 64, 64, n).transpose(0, 2, 1)
        return np.packbits(blocks, axis=2, bitorder="little").tobytes()
    out = bytearray()
    if format == "r8":
        for row in a:
            last = -1
            for k in [*np.flatnonzero(row), n]:
                gap = int(k) - last - 1
                while gap >= 255:
                    out.append(255)
                    gap -= 255
                out.append(gap)
                last = int(k)
        return bytes(out)
    if format == "hits":
        return "".join(",".join(map(str, np.flatnonzero(row))) + "\n" for row in a).encode()
    names = names or _names(num_measurements, num_detectors, num_observables)
    return "".join("shot" + "".join(" " + names[k] for k in np.flatnonzero(row)) + "\n" for row in a).encode()


def decode_shots(raw: bytes, format: str, num_measurements: int = 0, num_detectors: int = 0, num_observables: int = 0) -> np.ndarray:
    """The bytes of a ``format`` file as a (shots, n) bool array."""
    _format(format)
    n = _width(num_measurements, num_detectors, num_observables)
    if format == "01":
        lines = [line for line in raw.decode().split("\n")]
        if lines and lines[-1] == "":
            lines.pop()
        rows = []
        for line in lines:
            if len(line) != n or set(line) - {"0", "1"}:
                raise ValueError(f"a 01 line must be {n} characters of 0 and 1, not {line[:40]!r}")
            rows.append([c == "1" for c in line])
        return np.array(rows, dtype=bool).reshape(len(rows), n)
    if format == "b8":
        stride = (n + 7) // 8
        if stride == 0:
            return np.zeros((0, 0), dtype=bool)
        if len(raw) % stride:
            raise ValueError(f"b8 data of {len(raw)} bytes is not whole shots of {stride} bytes")
        packed = np.frombuffer(raw, dtype=np.uint8).reshape(-1, stride)
        return np.unpackbits(packed, axis=1, bitorder="little", count=n).astype(bool)
    if format == "ptb64":
        per_block = 8 * n
        if per_block == 0 or len(raw) % per_block:
            raise ValueError(f"ptb64 data of {len(raw)} bytes is not whole blocks of {per_block} bytes")
        blocks = np.frombuffer(raw, dtype=np.uint8).reshape(-1, n, 8)
        bits = np.unpackbits(blocks, axis=2, bitorder="little").astype(bool)
        return bits.transpose(0, 2, 1).reshape(-1, n)
    rows = []
    if format == "r8":
        k, row = 0, np.zeros(n, dtype=bool)
        for b in raw:
            k += b
            if b == 255:
                continue
            if k == n:
                rows.append(row)
                k, row = 0, np.zeros(n, dtype=bool)
                continue
            if k > n:
                raise ValueError("r8 data runs past the end of a shot")
            row[k] = True
            k += 1
        if k:
            raise ValueError("r8 data ends partway through a shot")
        return np.array(rows, dtype=bool).reshape(len(rows), n)
    lines = raw.decode().split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    if format == "hits":
        for line in lines:
            row = np.zeros(n, dtype=bool)
            for t in filter(None, line.split(",")):
                k = int(t)
                if not 0 <= k < n:
                    raise ValueError(f"hit {k} is out of range for {n} bits")
                row[k] = True
            rows.append(row)
        return np.array(rows, dtype=bool).reshape(len(rows), n)
    offsets = {"M": (0, num_measurements), "D": (num_measurements, num_detectors), "L": (num_measurements + num_detectors, num_observables)}
    for line in lines:
        words = line.split()
        if not words or words[0] != "shot":
            raise ValueError(f"a dets line must start with 'shot', not {line[:40]!r}")
        row = np.zeros(n, dtype=bool)
        for w in words[1:]:
            start, size = offsets.get(w[:1], (0, -1))
            k = int(w[1:]) if w[1:].isdigit() else -1
            if not 0 <= k < size:
                raise ValueError(f"{w!r} is out of range")
            row[start + k] = True
        rows.append(row)
    return np.array(rows, dtype=bool).reshape(len(rows), n)


def write_shot_data_file(
    *, data: np.ndarray, path: Union[str, PathLike], format: str, num_measurements: int = 0, num_detectors: int = 0, num_observables: int = 0
) -> None:
    """Write shots to a file in one of Stim's result formats, as ``stim.write_shot_data_file``.
    ``data`` is (shots, n) bool, or bit-packed uint8 rows (n = measurements + detectors +
    observables)."""
    n = _width(num_measurements, num_detectors, num_observables)
    a = np.asarray(data)
    if a.dtype == np.uint8 and a.ndim == 2:
        a = np.unpackbits(a, axis=1, bitorder="little", count=n).astype(bool)
    with open(path, "wb") as f:
        f.write(encode_shots(a, format, num_measurements, num_detectors, num_observables))


def read_shot_data_file(
    *,
    path: Union[str, PathLike],
    format: str,
    bit_packed: bool = False,
    num_measurements: int = 0,
    num_detectors: int = 0,
    num_observables: int = 0,
    separate_observables: bool = False,
    bit_pack: bool = False,
) -> Any:
    """Read shots from a file in one of Stim's result formats, as ``stim.read_shot_data_file``:
    (shots, n) bool, or bit-packed uint8 rows (``bit_pack`` is Stim's older name). With
    ``separate_observables``, ``(measurements and detectors, observables)``."""
    with open(path, "rb") as f:
        a = decode_shots(f.read(), format, num_measurements, num_detectors, num_observables)
    packed = bit_packed or bit_pack

    def out(x: np.ndarray) -> np.ndarray:
        return np.packbits(x, axis=1, bitorder="little") if packed else x

    if separate_observables:
        n = a.shape[1] - count(num_observables, "num_observables")
        return out(a[:, :n]), out(a[:, n:])
    return out(a)

"""Shared plumbing: calling the extension, and shots between numpy and Stim's b8 rows."""

from __future__ import annotations

import secrets
import sys
from typing import Any, Callable, TypeVar

import numpy as np

T = TypeVar("T")

_REPORT = "https://github.com/Jaspersands/quantum-simulator-qec/issues"


def call(fn: Callable[..., T], *args: Any) -> T:
    """Call into the extension; a Rust panic becomes RuntimeError, never PanicException."""
    try:
        return fn(*args)
    except BaseException as e:  # PanicException derives from BaseException
        if type(e).__name__ == "PanicException":
            raise RuntimeError(f"internal error in stabilizer-qec ({e}); please report it at {_REPORT}") from e
        raise


def text_of(obj: Any, kind: str, ours: type) -> str:
    """Stim text from a str, one of our objects, or a Stim object of the same kind."""
    if isinstance(obj, str):
        return obj
    if isinstance(obj, ours):
        return str(obj)
    if type(obj).__module__.split(".")[0] == "stim" and type(obj).__name__ == kind:
        return str(obj)
    raise TypeError(f"expected a str, stabilizer_qec.{ours.__name__} or stim.{kind}, not {type(obj).__name__}")


def count(value: Any, name: str, least: int = 0) -> int:
    """A non-negative integer argument (shots, threads, ...)."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer, not {type(value).__name__}")
    value = int(value)
    if value < least:
        raise ValueError(f"{name} must be at least {least}, not {value}")
    if value > _SIZE_MAX:
        raise ValueError(f"{name} = {value} is larger than this platform's sizes allow ({_SIZE_MAX})")
    return value


# The largest usize, which the extension's counts are.
_SIZE_MAX = 2 * sys.maxsize + 1


def seed_of(seed: Any) -> int:
    """A 64-bit seed; None draws one from the operating system."""
    if seed is None:
        return secrets.randbits(64)
    seed = count(seed, "seed")
    if seed >= 1 << 64:
        raise ValueError(f"seed must be below 2**64, not {seed}")
    return seed


def real(value: Any, name: str) -> float:
    """A finite number argument."""
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be a number, not {type(value).__name__}")
    try:
        x = float(value)
    except OverflowError:
        raise ValueError(f"{name} is too large for a float") from None
    except (TypeError, ValueError):
        raise TypeError(f"{name} must be a number, not {type(value).__name__}") from None
    if x != x or x in (float("inf"), float("-inf")):
        raise ValueError(f"{name} must be finite, not {x}")
    return x


def probability(p: Any, name: str = "p") -> float:
    p = real(p, name)
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"{name} = {p} is outside [0, 1]")
    return p


def stride(bits: int) -> int:
    return (bits + 7) // 8


def rows_to_b8(data: Any, bits: int, bit_packed: bool, name: str) -> tuple[bytes, int]:
    """A 2-D array of shots (bool or 0/1 per bit, or packed uint8 rows) as b8 bytes and its shots."""
    a = np.asarray(data)
    if a.ndim != 2:
        raise ValueError(f"{name} must be 2-dimensional (shots × bits), not {a.ndim}-dimensional")
    if bit_packed:
        if a.dtype != np.uint8:
            raise TypeError(f"bit-packed {name} must be uint8, not {a.dtype}")
        if a.shape[1] != stride(bits):
            raise ValueError(f"bit-packed {name} rows have {a.shape[1]} bytes, not {stride(bits)} for {bits} bits")
        return np.ascontiguousarray(a).tobytes(), a.shape[0]
    if a.shape[1] != bits:
        raise ValueError(f"{name} rows have {a.shape[1]} bits, not {bits}")
    if a.dtype != np.bool_:
        if not (np.issubdtype(a.dtype, np.integer) or np.issubdtype(a.dtype, np.bool_)):
            raise TypeError(f"{name} must be bool or integer 0/1, not {a.dtype}")
        a = a != 0
    return np.packbits(a, axis=1, bitorder="little").tobytes(), a.shape[0]


def b8_to_rows(raw: bytes, shots: int, bits: int, bit_packed: bool) -> np.ndarray:
    """b8 bytes as (shots, bits) bool, or (shots, ⌈bits/8⌉) uint8 when bit_packed."""
    a = np.frombuffer(raw, dtype=np.uint8).reshape(shots, stride(bits))
    if bit_packed:
        return a.copy()
    return np.unpackbits(a, axis=1, count=bits, bitorder="little").astype(bool)


def u64_to_rows(raw: bytes, shots: int, bits: int, bit_packed: bool) -> np.ndarray:
    """Little-endian u64 masks as (shots, bits) uint8, or packed (shots, ⌈bits/8⌉) uint8."""
    a = np.frombuffer(raw, dtype="<u8").reshape(shots, 1).view(np.uint8)[:, : stride(bits)]
    if bit_packed:
        return a.copy()
    return np.unpackbits(a, axis=1, count=bits, bitorder="little")


def pack_rows(a: np.ndarray, bit_packed: bool) -> np.ndarray:
    return np.packbits(a, axis=1, bitorder="little") if bit_packed else a

"""``FlipSimulator``: Stim's interactive Pauli-frame simulator over a batch of instances.

The state lives in the Rust core as bit-packed rows (one bit per instance); this class gives
it Stim's interface, numpy arrays in and out.
"""

from __future__ import annotations

from typing import Any, List, Optional, Union

import numpy as np

from . import _core, _stim
from ._util import call

_PAULIS = {"I": 0, "_": 0, "X": 1, "Y": 2, "Z": 3, 0: 0, 1: 1, 2: 2, 3: 3}


def _pauli_code(pauli: Any) -> int:
    key = int(pauli) if isinstance(pauli, (int, np.integer)) and not isinstance(pauli, bool) else pauli
    if key not in _PAULIS:
        raise ValueError("Need pauli in ['I', 'X', 'Y', 'Z', 0, 1, 2, 3, '_'].")
    return _PAULIS[key]


def _checked_index(name: str, k: int, n: int, size_name: str) -> int:
    if not -n <= k < n:
        raise IndexError(f"not (-{size_name} <= {name}={k} < {size_name}={n})")
    return k + n if k < 0 else k


class FlipSimulator:
    """Tracks whether things are flipped rather than what they are, for ``batch_size``
    instances at once, as ``stim.FlipSimulator``: errors propagate through Cliffords,
    measurements record the flips they read, detectors and observables the parities of those.

    Unless ``disable_stabilizer_randomization``, every reset and measurement multiplies the
    frame by the new stabilizer with probability 1/2, as Stim does, so a measurement of
    something the state doesn't determine reads as a coin flip."""

    __slots__ = ("_s",)

    def __init__(self, *, batch_size: int, disable_stabilizer_randomization: bool = False, num_qubits: int = 0, seed: Optional[int] = None) -> None:
        self._s = _core.FlipSimCore(int(batch_size), int(num_qubits), not disable_stabilizer_randomization, _stim._seed(seed))

    # Sizes.
    @property
    def batch_size(self) -> int:
        return self._s.batch_size

    @property
    def num_qubits(self) -> int:
        return self._s.counts[0]

    @property
    def num_measurements(self) -> int:
        return self._s.counts[1]

    @property
    def num_detectors(self) -> int:
        return self._s.counts[2]

    @property
    def num_observables(self) -> int:
        return self._s.counts[3]

    # State.
    def _rows(self, which: str, n: int) -> np.ndarray:
        """A table as packed bytes: shape (n, ceil(batch / 8))."""
        words = (self.batch_size + 63) // 64
        raw = np.frombuffer(self._s.table(which), dtype=np.uint8).reshape(n, words * 8)
        return raw[:, : (self.batch_size + 7) // 8].copy()

    def _bools(self, which: str, n: int) -> np.ndarray:
        return np.unpackbits(self._rows(which, n), axis=1, count=self.batch_size, bitorder="little").astype(np.bool_)

    def do(self, obj: Any) -> None:
        """Applies a circuit, an instruction or a repeat block."""
        try:
            call(self._s.do_circuit, _stim._stim_text_of(obj))
        except ValueError as e:
            if "before the beginning of time" in str(e):
                raise IndexError(str(e)) from None
            raise

    def clear(self) -> None:
        """Back to |0> in every instance with empty records (the size is kept)."""
        self._s.clear()

    def copy(self, *, copy_rng: bool = False, seed: Optional[int] = None) -> "FlipSimulator":
        """A simulator with the same state; its random numbers are the original's only with
        ``copy_rng``, otherwise freshly seeded (from ``seed`` if given)."""
        if copy_rng and seed is not None:
            raise ValueError("seed and copy_rng are incompatible")
        out = FlipSimulator.__new__(FlipSimulator)
        out._s = self._s.copy(None if copy_rng else _stim._seed(seed))
        return out

    def peek_pauli_flips(self, *, instance_index: Optional[int] = None) -> Union["_stim.PauliString", List["_stim.PauliString"]]:
        """The Pauli flips of every instance (a list), or of one."""
        n = self.num_qubits
        xs, zs = self._bools("x", n), self._bools("z", n)
        if instance_index is not None:
            k = _checked_index("instance_index", int(instance_index), self.batch_size, "batch_size")
            return _stim.PauliString.from_numpy(xs=xs[:, k], zs=zs[:, k])
        return [_stim.PauliString.from_numpy(xs=xs[:, k], zs=zs[:, k]) for k in range(self.batch_size)]

    def set_pauli_flip(self, pauli: Any, *, qubit_index: int, instance_index: int) -> None:
        """Sets the flip on one qubit of one instance (growing the qubits if needed)."""
        p = _pauli_code(pauli)
        q = int(qubit_index)
        if q < 0:
            raise IndexError("qubit_index")
        k = int(instance_index)
        if k < 0:
            k += self.batch_size
        if not 0 <= k < self.batch_size:
            raise IndexError("instance_index")
        self._s.set_flip(q, k, p in (1, 2), p in (2, 3))

    def broadcast_pauli_errors(self, *, pauli: Any, mask: np.ndarray, p: float = 1) -> None:
        """Applies ``pauli`` to qubit q of instance k wherever ``mask[q, k]``, each with
        probability ``p``."""
        code = _pauli_code(pauli)
        mask = np.asarray(mask)
        if mask.dtype != np.bool_ or mask.ndim != 2 or mask.shape[1] != self.batch_size:
            raise ValueError("Need mask.shape[1] == flip_sim.batch_size")
        self._s.broadcast(code, _pack_rows(mask, self.batch_size), float(p))

    def append_measurement_flips(self, measurement_flip_data: np.ndarray) -> None:
        """Appends measurement flips: one or more rows over the batch, as bools or bytes
        packed little-endian."""
        d = np.asarray(measurement_flip_data)
        b = self.batch_size
        nb = (b + 7) // 8
        if d.dtype == np.bool_:
            if d.ndim == 1:
                if d.shape[0] != b:
                    raise ValueError(f"dtype=np.bool_ and len(shape) == 1 but shape[0]={d.shape[0]} != batch_size={b}")
                d = d[None, :]
            elif d.ndim != 2 or d.shape[1] != b:
                raise ValueError(f"dtype=np.bool_ and len(shape) == 2 but shape[1]={d.shape[1] if d.ndim > 1 else None} != batch_size={b}")
            rows = d
        elif d.dtype == np.uint8:
            if d.ndim == 1:
                if d.shape[0] != nb:
                    raise ValueError(f"dtype=np.uint8 and len(shape) == 1 but shape[0]={d.shape[0]} != math.ceil(batch_size / 8)={nb}")
                d = d[None, :]
            elif d.ndim != 2 or d.shape[1] != nb:
                raise ValueError(f"dtype=np.uint8 and len(shape) == 2 but shape[1]={d.shape[1] if d.ndim > 1 else None} != math.ceil(batch_size / 8)={nb}")
            rows = np.unpackbits(d, axis=1, count=b, bitorder="little").astype(np.bool_)
        else:
            raise ValueError("measurement_flip_data must have dtype np.bool_ or np.uint8")
        self._s.append_measurements(_pack_rows(rows, b))

    # Reading the records.
    def _get(self, which: str, n: int, index: Optional[int], index_name: str, size_name: str, instance_index: Optional[int], bit_packed: bool) -> np.ndarray:
        data = self._bools(which, n)
        if index is not None:
            data = data[_checked_index(index_name, int(index), n, size_name)]
        if instance_index is not None:
            k = _checked_index("instance_index", int(instance_index), self.batch_size, "batch_size")
            data = data[..., k]
        if bit_packed and data.ndim > 0:
            return np.packbits(data, axis=data.ndim - 1, bitorder="little")
        return np.asarray(data)

    def get_measurement_flips(self, *, record_index: Optional[int] = None, instance_index: Optional[int] = None, bit_packed: bool = False) -> np.ndarray:
        """The measurement flips: (num_measurements, batch_size), sliced by the indices."""
        return self._get("m", self.num_measurements, record_index, "record_index", "num_measurements", instance_index, bit_packed)

    def get_detector_flips(self, *, detector_index: Optional[int] = None, instance_index: Optional[int] = None, bit_packed: bool = False) -> np.ndarray:
        """The detector flips: (num_detectors, batch_size), sliced by the indices."""
        return self._get("d", self.num_detectors, detector_index, "detector_index", "num_detectors", instance_index, bit_packed)

    def get_observable_flips(self, *, observable_index: Optional[int] = None, instance_index: Optional[int] = None, bit_packed: bool = False) -> np.ndarray:
        """The observable flips: (num_observables, batch_size), sliced by the indices."""
        return self._get("o", self.num_observables, observable_index, "observable_index", "num_observables", instance_index, bit_packed)

    def to_numpy(
        self,
        *,
        bit_packed: bool = False,
        transpose: bool = False,
        output_xs: Any = False,
        output_zs: Any = False,
        output_measure_flips: Any = False,
        output_detector_flips: Any = False,
        output_observable_flips: Any = False,
    ) -> tuple:
        """The state as numpy arrays ``(xs, zs, ms, ds, os)``, each ``None`` unless asked for
        (``True`` allocates; an array is written into). Rows are qubits / records and columns
        instances, or the reverse with ``transpose``; ``bit_packed`` packs the last axis."""
        wants = [("output_xs", output_xs, "x", self.num_qubits), ("output_zs", output_zs, "z", self.num_qubits), ("output_measure_flips", output_measure_flips, "m", self.num_measurements), ("output_detector_flips", output_detector_flips, "d", self.num_detectors), ("output_observable_flips", output_observable_flips, "o", self.num_observables)]
        if all(isinstance(w, bool) and not w for _, w, _, _ in wants):
            raise ValueError("No outputs requested! Specify at least one output_*= argument.")
        out = []
        for name, want, which, n in wants:
            if isinstance(want, bool) and not want:
                out.append(None)
                continue
            data = self._bools(which, n)
            if transpose:
                data = data.T
            if bit_packed:
                data = np.packbits(data, axis=1, bitorder="little")
            data = np.ascontiguousarray(data)
            if isinstance(want, np.ndarray):
                if want.dtype != data.dtype or want.shape != data.shape:
                    raise ValueError(f"{name} wasn't set to False, True, or a numpy array with dtype={data.dtype.type} and shape={data.shape}")
                want[...] = data
                out.append(want)
            elif want is True:
                out.append(data)
            else:
                raise ValueError(f"{name} wasn't set to False, True, or a numpy array with dtype={data.dtype.type} and shape={data.shape}")
        return tuple(out)

    def generate_bernoulli_samples(self, num_samples: int, *, p: float, bit_packed: bool = False, out: Optional[np.ndarray] = None) -> np.ndarray:
        """Biased coins from the simulator's generator: ``num_samples`` bools (or bytes packed
        little-endian, padding zero), written into ``out`` if given."""
        n = int(num_samples)
        raw = np.frombuffer(self._s.bernoulli(n, float(p)), dtype=np.uint8)
        data = raw.copy() if bit_packed else np.unpackbits(raw, count=n, bitorder="little").astype(np.bool_)
        if out is not None:
            if out.dtype != data.dtype or out.size != data.size:
                raise ValueError(f"Expected output buffer to have size {data.size} but its size is {out.size}.")
            out.reshape(-1)[...] = data
            return out
        return data


def _pack_rows(rows: np.ndarray, batch: int) -> bytes:
    """Bool rows as the core's words: each row padded to whole 64-bit words, little-endian."""
    words = (batch + 63) // 64
    padded = np.zeros((rows.shape[0], words * 64), dtype=np.bool_)
    padded[:, :batch] = rows
    return np.packbits(padded, axis=1, bitorder="little").tobytes()

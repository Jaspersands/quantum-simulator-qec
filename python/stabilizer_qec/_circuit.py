"""Circuits, detector error models, sampling and measurement conversion."""

from __future__ import annotations

from os import PathLike
from typing import Any, Union

import numpy as np

from . import _core
from ._util import b8_to_rows, call, count, pack_rows, rows_to_b8, seed_of, stride, text_of


class Circuit:
    """A stabilizer circuit in Stim's circuit language.

    Build one from Stim text, a ``stim.Circuit``, or another ``Circuit``; ``str()`` gives the
    text back as written. Every Clifford gate of Stim's is read (H, S and CX natively, the
    rest as their exact decompositions), with resets and measurements in all three bases,
    inverted targets (``!q``), Pauli-product measurements and rotations (``MPP``, ``MXX``,
    ``MYY``, ``MZZ``, ``SPP``, ``SPP_DAG``), the Pauli, depolarizing and correlated noise
    channels, ``MPAD``, detectors, observables, coordinates, ``TICK`` and ``REPEAT``.
    Heralded errors and classically controlled gates other than ``CX sweep[k]`` raise
    ``ValueError``.

    >>> c = Circuit("R 0 1\\nH 0\\nCX 0 1\\nM 0 1\\nDETECTOR rec[-1] rec[-2]")
    >>> c.num_qubits, c.num_measurements, c.num_detectors
    (2, 2, 1)
    """

    __slots__ = ("_c",)

    def __init__(self, text: Union[str, "Circuit", Any] = "") -> None:
        self._c = call(_core.Circuit, text_of(text, "Circuit", Circuit))

    @classmethod
    def from_file(cls, path: Union[str, PathLike]) -> "Circuit":
        """Read a ``.stim`` file."""
        with open(path, encoding="utf-8") as f:
            return cls(f.read())

    def __str__(self) -> str:
        return self._c.__str__()

    def __repr__(self) -> str:
        return f"stabilizer_qec.Circuit({str(self)!r})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Circuit):
            return NotImplemented
        return self._c.__eq__(other._c)

    __hash__ = None  # type: ignore[assignment]

    @property
    def num_qubits(self) -> int:
        """One more than the largest qubit index any instruction names."""
        return self._c.num_qubits

    @property
    def num_measurements(self) -> int:
        """Measurements per shot, counted through loops."""
        return self._c.num_measurements

    @property
    def num_detectors(self) -> int:
        return self._c.num_detectors

    @property
    def num_observables(self) -> int:
        return self._c.num_observables

    @property
    def num_sweep_bits(self) -> int:
        return self._c.num_sweep_bits

    def detector_error_model(self, *, decompose_errors: bool = False) -> "DetectorErrorModel":
        """The circuit's detector error model, built by walking it backwards, as Stim's error
        analyzer does. ``decompose_errors`` splits each fault into graph-like pieces (at most
        two detectors each) the way Stim splits them, which matching needs."""
        return DetectorErrorModel._wrap(call(self._c.detector_error_model, bool(decompose_errors)))

    def compile_detector_sampler(self, *, seed: Union[int, None] = None) -> "DetectorSampler":
        """A sampler of detection events and observable flips. The same seed gives the same
        shots on any machine and any number of threads; ``None`` draws a seed."""
        return DetectorSampler(self, seed)

    def compile_m2d_converter(self) -> "MeasurementsToDetectionEventsConverter":
        """A converter from raw measurements (and sweep bits) to detection events, as
        ``stim m2d``. Its reference run keeps a dense tableau: at most 16,384 qubits."""
        return MeasurementsToDetectionEventsConverter(self)


class DetectorErrorModel:
    """A detector error model in Stim's format: independent faults, each with a probability,
    the detectors it flips and the observables it flips.

    Build one from Stim text, a ``stim.DetectorErrorModel``, or ``Circuit.detector_error_model``.
    """

    __slots__ = ("_d",)

    def __init__(self, text: Union[str, "DetectorErrorModel", Any] = "") -> None:
        self._d = call(_core.Dem, text_of(text, "DetectorErrorModel", DetectorErrorModel))

    @classmethod
    def _wrap(cls, inner: Any) -> "DetectorErrorModel":
        dem = cls.__new__(cls)
        dem._d = inner
        return dem

    @classmethod
    def from_file(cls, path: Union[str, PathLike]) -> "DetectorErrorModel":
        """Read a ``.dem`` file."""
        with open(path, encoding="utf-8") as f:
            return cls(f.read())

    def __str__(self) -> str:
        return self._d.__str__()

    def __repr__(self) -> str:
        return f"stabilizer_qec.DetectorErrorModel({str(self)!r})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DetectorErrorModel):
            return NotImplemented
        return str(self) == str(other)

    __hash__ = None  # type: ignore[assignment]

    @property
    def num_detectors(self) -> int:
        return self._d.num_detectors

    @property
    def num_observables(self) -> int:
        return self._d.num_observables

    @property
    def num_errors(self) -> int:
        """The number of faults (``error`` lines)."""
        return self._d.num_errors


class DetectorSampler:
    """Detection events and observable flips, 64 shots to a machine word. Made by
    ``Circuit.compile_detector_sampler``."""

    __slots__ = ("_s",)

    def __init__(self, circuit: Circuit, seed: Union[int, None] = None) -> None:
        if not isinstance(circuit, Circuit):
            circuit = Circuit(circuit)
        self._s = call(circuit._c.sampler, seed_of(seed))

    @property
    def num_detectors(self) -> int:
        return self._s.num_detectors

    @property
    def num_observables(self) -> int:
        return self._s.num_observables

    def sample(
        self,
        shots: int,
        *,
        separate_observables: bool = False,
        append_observables: bool = False,
        bit_packed: bool = False,
        threads: int = 1,
    ) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray]]:
        """``shots`` shots, as Stim's ``DetectorSampler.sample`` gives them: detection events
        of shape (shots, detectors), bool, or (shots, ⌈detectors/8⌉) uint8 when ``bit_packed``
        (bit k of a row in byte k // 8, position k % 8). ``separate_observables`` returns
        ``(detections, observables)``; ``append_observables`` puts the observables after each
        row's detectors. ``threads = 0`` uses every core; the shots do not depend on it.

        Successive calls continue the stream. Shots are drawn 64 at a time, and a call that
        ends partway through 64 discards the rest of them.
        """
        shots = count(shots, "shots")
        threads = count(threads, "threads")
        if separate_observables and append_observables:
            raise ValueError("choose separate_observables or append_observables, not both")
        nd, no = self.num_detectors, self.num_observables
        d, o = call(self._s.sample, shots, threads)
        if append_observables:
            rows = np.concatenate([b8_to_rows(d, shots, nd, False), b8_to_rows(o, shots, no, False)], axis=1)
            return pack_rows(rows, bit_packed)
        dets = b8_to_rows(d, shots, nd, bit_packed)
        if separate_observables:
            return dets, b8_to_rows(o, shots, no, bit_packed)
        return dets


class MeasurementsToDetectionEventsConverter:
    """Raw measurement records to detection events and observable flips, as ``stim m2d``:
    each detector and observable compared with a noiseless reference run. Made by
    ``Circuit.compile_m2d_converter``."""

    __slots__ = ("_m",)

    def __init__(self, circuit: Circuit) -> None:
        if not isinstance(circuit, Circuit):
            circuit = Circuit(circuit)
        self._m = call(circuit._c.m2d)

    @property
    def num_measurements(self) -> int:
        return self._m.num_measurements

    @property
    def num_sweep_bits(self) -> int:
        return self._m.num_sweep_bits

    @property
    def num_detectors(self) -> int:
        return self._m.num_detectors

    @property
    def num_observables(self) -> int:
        return self._m.num_observables

    def convert(
        self,
        *,
        measurements: np.ndarray,
        sweep_bits: Union[np.ndarray, None] = None,
        separate_observables: bool = False,
        append_observables: bool = False,
        bit_packed: bool = False,
    ) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray]]:
        """``measurements`` (and ``sweep_bits``) are (shots, n) bool arrays, or uint8 rows
        bit-packed as Stim packs them. The output takes the same forms as
        ``DetectorSampler.sample``'s, packed when ``bit_packed``."""
        if separate_observables and append_observables:
            raise ValueError("choose separate_observables or append_observables, not both")
        m = np.asarray(measurements)
        packed_in = m.dtype == np.uint8
        meas, shots = rows_to_b8(m, self.num_measurements, packed_in, "measurements")
        ns = self.num_sweep_bits
        if sweep_bits is None:
            sweeps = bytes(stride(ns) * shots)
        else:
            s = np.asarray(sweep_bits)
            sweeps, sweep_shots = rows_to_b8(s, ns, s.dtype == np.uint8, "sweep_bits")
            if sweep_shots != shots:
                raise ValueError(f"{sweep_shots} shots of sweep bits for {shots} shots of measurements")
        d, o = call(self._m.convert, meas, sweeps, shots)
        nd, no = self.num_detectors, self.num_observables
        if append_observables:
            rows = np.concatenate([b8_to_rows(d, shots, nd, False), b8_to_rows(o, shots, no, False)], axis=1)
            return pack_rows(rows, bit_packed)
        dets = b8_to_rows(d, shots, nd, bit_packed)
        if separate_observables:
            return dets, b8_to_rows(o, shots, no, bit_packed)
        return dets

"""A quantum error-correction simulator and decoder, in Rust: Stim's circuit and error-model
formats, a bit-parallel sampler, exact and correlated matching, belief-matching, BP and BP+OSD,
window decoding, IBM's bivariate bicycle codes and lattice surgery. It agrees with Stim,
PyMatching, `ldpc` and `beliefmatching` on every check in its test suite.

    pip install stabilizer-qec

## Getting started

A memory experiment, sampled and decoded:

>>> import stabilizer_qec as sq
>>> circuit = sq.memory_circuit(distance=3, rounds=3, p=0.001)
>>> circuit.num_detectors, circuit.num_observables
(24, 1)
>>> sampler = circuit.compile_detector_sampler(seed=7)
>>> dets, obs = sampler.sample(10_000, separate_observables=True)
>>> dets.shape, dets.dtype
((10000, 24), dtype('bool'))
>>> dem = circuit.detector_error_model(decompose_errors=True)
>>> predictions = sq.Matching(dem).decode_batch(dets)
>>> failures = int((predictions != obs).any(axis=1).sum())
>>> 0 <= failures < 100
True

The names follow the tools this is checked against, so code written for them mostly carries
over: `Circuit`, `compile_detector_sampler` and `separate_observables` are Stim's; `Matching`,
`decode_batch` and `enable_correlations` are PyMatching's; `BpDecoder`, `BpOsdDecoder`,
`error_channel` and `osd_order` are `ldpc`'s.

## Circuits

`Circuit` reads Stim's circuit language: every Clifford gate (H, S and CX natively, the other
46 one- and two-qubit gates as their exact decompositions), resets and measurements in all
three bases with inverted targets (`!q`), Pauli-product measurements and rotations (`MPP`,
`MXX`, `MYY`, `MZZ`, `SPP`), the Pauli, depolarizing and correlated noise channels
(`X_ERROR` ... `PAULI_CHANNEL_2`, `E`, `ELSE_CORRELATED_ERROR`), `MPAD`, detectors,
observables, coordinates, `TICK` and `REPEAT`. Heralded errors and classically controlled
gates other than `CX sweep[k]` are refused with a `ValueError`.

Circuits written for you: `memory_circuit` (rotated and XZZX surface codes),
`BivariateBicycleCode` (the gross code and [[72, 12, 6]], with their logical operations), the
`surgery` module (Z⊗Z and X⊗X merges, a logical CNOT, merges in a row), and `stream_memory`
(a million rounds, window-decoded as they stream).

## Shots

Shots are numpy arrays: one row per shot, `bool` per detector, or bit-packed `uint8` rows
(bit k of a row in byte k // 8, at position k % 8: `numpy.packbits(..., bitorder="little")`),
the layout Stim and PyMatching use. Predictions are `uint8` 0/1 per observable.

## Promises

- **Seeds reproduce.** A sampler's shots depend only on its seed and on how many it has drawn,
  not on the machine or the number of threads (`threads=0` is every core).
- **Errors are errors.** Bad input raises `ValueError` (or `TypeError` for a wrong type); an
  engine bug raises `RuntimeError` asking for a report, never a crash.
- **Limits.** Qubit indices up to 2**24 - 1, as in Stim. An error model or m2d holds a circuit
  unrolled, up to 2**24 instructions and targets; the sampler runs loops without unrolling
  them. Generated circuits are written out round by round, up to 10,000 rounds.
- **Stability.** From 1.0, semantic versioning; until then a minor release may change the API
  and says so in the changelog. The 0.4 functions still work, each warning
  `DeprecationWarning` with its replacement, until 1.0.

The explainer: https://qcompiler.jaspersands.com. The source, the technical report and the
changelog: https://github.com/Jaspersands/quantum-simulator-qec.
"""
from __future__ import annotations

import warnings as _warnings
from importlib.metadata import PackageNotFoundError as _NotFound
from importlib.metadata import version as _version

from . import _core, surgery
from ._circuit import Circuit, DetectorErrorModel, DetectorSampler, MeasurementsToDetectionEventsConverter
from ._codes import Automorphism, BivariateBicycleCode, Gauging, StreamResult, memory_circuit, stream_memory
from ._decoders import BeliefMatching, BpDecoder, BpOsd, BpOsdDecoder, Matching, Window, WindowMatching

try:
    __version__ = _version("stabilizer-qec")
except _NotFound:  # pragma: no cover - running from a source tree without metadata
    __version__ = "0+unknown"

# The 0.4 interface: each name still works through 0.x, warns, and goes in 1.0.
_DEPRECATED = {
    "generate_circuit": "memory_circuit(...)",
    "dem_from_circuit": "Circuit(text).detector_error_model(decompose_errors=...)",
    "circuit_to_stim": "str(Circuit(text))",
    "sample_b8": "Circuit(text).compile_detector_sampler(seed).sample(shots, bit_packed=True)",
    "sample_b8_batch": "Circuit(text).compile_detector_sampler(seed).sample(shots, bit_packed=True, threads=...)",
    "m2d_b8": "Circuit(text).compile_m2d_converter().convert(measurements, sweep_bits, bit_packed=True)",
    "b8_to_01": "numpy.unpackbits(..., bitorder='little')",
    "decode_b8": "Matching(DetectorErrorModel(text)).decode_batch(shots, bit_packed_shots=True)",
    "decode_b8_own": "Matching(Circuit(text).detector_error_model(decompose_errors=True)).decode_batch(...)",
    "decode_b8_window": "WindowMatching(dem, commit=..., buffer=...).decode_batch(...)",
    "stream_decode": "stream_memory(...)",
    "bp_decode": "BpDecoder(pcm, error_channel).decode(syndrome)",
    "decode_b8_belief": "BeliefMatching(dem).decode_batch(...)",
    "bposd_decode": "BpOsdDecoder(pcm, error_channel).decode(syndrome)",
    "decode_b8_bposd": "BpOsd(dem).decode_batch(...)",
    "bb_matrices": "BivariateBicycleCode(name).check_matrices() and .logicals()",
    "bb_memory_circuit": "BivariateBicycleCode(name).memory_circuit(cycles, p)",
    "bb_memory_basis_circuit": "BivariateBicycleCode(name).memory_circuit(cycles, p, basis)",
    "bb_automorphisms": "BivariateBicycleCode(name).automorphisms()",
    "bb_gauging": "BivariateBicycleCode('gross').gauging(operator, expanded)",
    "bb_logical_measurement_circuit": "BivariateBicycleCode('gross').logical_measurement_circuit(...)",
    "surgery_circuit": "surgery.zz_measurement(...)",
    "surgery_vertical": "surgery.xx_measurement(...)",
    "surgery_cnot": "surgery.cnot(...)",
    "surgery_repeated": "surgery.repeated_zz(...)",
    "surgery_line": "surgery.line(...)",
    "Decoder": "Matching(dem)",
    "RotatedSurfaceCode": "memory_circuit(...) with Matching",
}


def __getattr__(name: str):
    replacement = _DEPRECATED.get(name)
    if replacement is None:
        raise AttributeError(f"module 'stabilizer_qec' has no attribute {name!r}")
    _warnings.warn(
        f"stabilizer_qec.{name} is deprecated and will be removed in 1.0; use {replacement}",
        DeprecationWarning,
        stacklevel=2,
    )
    return getattr(_core, name)


def __dir__():
    return sorted(set(globals()) | set(__all__))


__all__ = [
    "Automorphism",
    "BeliefMatching",
    "BivariateBicycleCode",
    "BpDecoder",
    "BpOsd",
    "BpOsdDecoder",
    "Circuit",
    "DetectorErrorModel",
    "DetectorSampler",
    "Gauging",
    "Matching",
    "MeasurementsToDetectionEventsConverter",
    "StreamResult",
    "Window",
    "WindowMatching",
    "__version__",
    "memory_circuit",
    "stream_memory",
    "surgery",
]

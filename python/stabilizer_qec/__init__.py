"""A quantum error-correction simulator and decoder, in Rust.

Circuits and detector error models in Stim's formats, a bit-parallel sampler, exact and
correlated matching, belief-matching, BP and BP+OSD, window decoding, IBM's bivariate bicycle
codes and lattice surgery. See https://qcompiler.jaspersands.com/api/ for the guide and the
reference.
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

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
`MXX`, `MYY`, `MZZ`, `SPP`), every noise channel (`X_ERROR` ... `PAULI_CHANNEL_2`, `E`,
`ELSE_CORRELATED_ERROR`, `HERALDED_ERASE`, `HERALDED_PAULI_CHANNEL_1`), measurement feedback
and sweep bits (`CX rec[-1] q`, `CZ sweep[k] q`), `MPAD`, detectors, observables (of
measurement records, and of Pauli targets: `OBSERVABLE_INCLUDE(0) X0 Z1`), coordinates, `TICK`,
`REPEAT`, and instruction tags (`H[tag] 0`), which reach the error model as in Stim. Channels
whose cases are disjoint enter an error model only with `approximate_disjoint_errors`, as in
Stim.

Or build one in code, as in Stim: `append` an instruction at a time (targets as qubit
indices, strings such as `"rec[-1]"`, or `stim.GateTarget`s), `+` one circuit after another,
and `*` one into a `REPEAT` block.

Circuits written for you: `Circuit.generated` (Stim's generated memories, character for
character), `memory_circuit` (rotated and XZZX surface codes), `BivariateBicycleCode` (every
code of Bravyi et al.'s Table 3 or any other, the gross code's logical operations), `CssCode`
(any CSS code from its checks, hypergraph products and colour codes, with a memory), the
`surgery` module (Z⊗Z and X⊗X merges, a logical CNOT, merges in a row), and `stream_memory`
(a million rounds of a memory, or of any circuit with a loop, window-decoded as they stream).

## Knowing a circuit

Where each fault of the error model comes from, as Stim prints it
(`explain_detector_error_model_errors`, every `ExplainedError` naming the gate, its targets
and the Pauli product or measurement it flips); how few faults defeat the circuit
(`shortest_graphlike_error` and `search_for_undetectable_logical_errors`, Stim's two searches,
on circuits and on models, and `DetectorErrorModel.distance()`, exact by integer programming
with scipy installed):

>>> c = sq.Circuit.generated("repetition_code:memory", distance=3, rounds=2, after_clifford_depolarization=0.01)
>>> len(c.shortest_graphlike_error())
3
>>> print(c.shortest_graphlike_error()[0].circuit_error_locations[0].instruction_targets)
DEPOLARIZE2(0.01) 0 1

Diagrams after Stim's (`Circuit.diagram`): `timeline-text` (Stim's, character for
character), `timeline-svg`, `detslice-svg`, `timeslice-svg` and `detslice-with-ops-svg` (one
panel per tick of a `range`), and the matching graph.

## Shots

Shots are numpy arrays: one row per shot, `bool` per detector, or bit-packed `uint8` rows
(bit k of a row in byte k // 8, at position k % 8: `numpy.packbits(..., bitorder="little")`),
the layout Stim and PyMatching use. Predictions are `uint8` 0/1 per observable. Files in
Stim's six formats (`01`, `b8`, `r8`, `hits`, `dets`, `ptb64`) are read and written by
`read_shot_data_file` and `write_shot_data_file`; raw measurements come from
`Circuit.compile_sampler`, and faults of a model straight from
`DetectorErrorModel.compile_sampler`.

## The command line

`stabilizer-qec` (or `python -m stabilizer_qec`) takes Stim's commands and flags (`gen`,
`sample`, `detect`, `m2d`, `analyze_errors`, `sample_dem`, `convert`, `explain_errors`,
`diagram`) and gives Stim's bytes wherever Stim's output is deterministic, plus `decode`:

    stabilizer-qec gen --code surface_code --task rotated_memory_z --distance 5 --rounds 5 \
        --after_clifford_depolarization 0.001 > c.stim
    stabilizer-qec detect --in c.stim --shots 1000 --out_format b8 > d.b8
    stabilizer-qec analyze_errors --in c.stim --decompose_errors > c.dem
    stabilizer-qec decode --dem c.dem --in d.b8 --in_format b8

## Many processes, and sinter

Circuits, error models, converters and decoders pickle and copy, so they cross into
`multiprocessing` and `concurrent.futures` workers (a decoder is rebuilt there from its model
and options). A sampler does not: send the circuit and a seed.

`stabilizer_qec.sinter` puts the decoders into [sinter](https://pypi.org/project/sinter/)'s
sweeps, decoding the shots Stim samples (`sq_matching`, `sq_correlated_matching`,
`sq_belief_matching`, `sq_bposd`) or running the whole pipeline here (`sq_sim_matching`, ...):

    import sinter
    from stabilizer_qec import sinter as sq_sinter

    stats = sinter.collect(tasks=tasks, decoders=["sq_matching", "pymatching"],
                           custom_decoders=sq_sinter.sinter_decoders(), num_workers=8,
                           max_shots=100_000)

From the command line: `sinter collect ... --decoders sq_matching
--custom_decoders_module_function "stabilizer_qec.sinter:sinter_decoders"`.

## Promises

- **Seeds reproduce.** A sampler's shots depend only on its seed and on how many it has drawn,
  not on the machine or the number of threads (`threads=0` is every core).
- **Errors are errors.** Bad input raises `ValueError` (or `TypeError` for a wrong type); an
  engine bug raises `RuntimeError` asking for a report, never a crash.
- **Limits.** Qubit indices up to 2**24 - 1, as in Stim. An error model folds the loops that
  repeat, as Stim folds them, so a d = 11 memory of 10,000 rounds is analysed in a tenth of a
  second; a decoder unrolls its model, up to 2**24 faults and declarations. The m2d converter
  holds a circuit unrolled, up to 2**24 instructions and targets; the sampler runs loops
  without unrolling them. Generated circuits are written out round by round, up to 10,000
  rounds.
- **Stability.** Semantic versioning: everything importable from `stabilizer_qec` without a
  leading underscore keeps working, with the same meaning, through every 1.x release. A name
  is deprecated (with a `DeprecationWarning` naming its replacement) for at least one minor
  release before a major release removes it. `stabilizer_qec._core` is the engine's
  extension, used by the repository's own tools, and carries no such promise.

The explainer: https://qcompiler.jaspersands.com. The source, the technical report and the
changelog: https://github.com/Jaspersands/quantum-simulator-qec.
"""
from __future__ import annotations

from importlib.metadata import PackageNotFoundError as _NotFound
from importlib.metadata import version as _version

from . import _core, surgery
from ._circuit import Circuit, DemSampler, DetectorErrorModel, DetectorSampler, Diagram, MeasurementSampler, MeasurementsToDetectionEventsConverter
from ._explain import (
    CircuitErrorLocation,
    CircuitErrorLocationStackFrame,
    CircuitTargetsInsideInstruction,
    DemTargetWithCoords,
    ExplainedError,
    FlippedMeasurement,
    GateTargetWithCoords,
)
from ._shots import read_shot_data_file, write_shot_data_file
from ._codes import Automorphism, BivariateBicycleCode, CssCode, Gauging, StreamResult, memory_circuit, stream_memory
from ._decoders import BeliefMatching, BpDecoder, BpLsd, BpLsdDecoder, BpOsd, BpOsdDecoder, ColorMatching, Matching, RelayBp, RelayBpDecoder, UnionFind, Window, WindowMatching

try:
    __version__ = _version("stabilizer-qec")
except _NotFound:  # pragma: no cover - running from a source tree without metadata
    __version__ = "0+unknown"

# The 0.4 interface, removed in 1.0: each old name points to what replaced it. The repository's
# tools use the extension directly, as stabilizer_qec._core, which is not part of the API.
_REMOVED = {
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
    replacement = _REMOVED.get(name)
    if replacement is None:
        raise AttributeError(f"module 'stabilizer_qec' has no attribute {name!r}")
    raise AttributeError(f"stabilizer_qec.{name} was removed in 1.0; use {replacement}")


def __dir__():
    return sorted(set(globals()) | set(__all__))


__all__ = [
    "Automorphism",
    "BeliefMatching",
    "BivariateBicycleCode",
    "CssCode",
    "read_shot_data_file",
    "write_shot_data_file",
    "CircuitErrorLocation",
    "CircuitErrorLocationStackFrame",
    "CircuitTargetsInsideInstruction",
    "DemTargetWithCoords",
    "ExplainedError",
    "FlippedMeasurement",
    "GateTargetWithCoords",
    "BpDecoder",
    "BpLsd",
    "BpLsdDecoder",
    "BpOsd",
    "BpOsdDecoder",
    "Circuit",
    "ColorMatching",
    "DetectorErrorModel",
    "DetectorSampler",
    "DemSampler",
    "MeasurementSampler",
    "Diagram",
    "Gauging",
    "Matching",
    "RelayBp",
    "RelayBpDecoder",
    "MeasurementsToDetectionEventsConverter",
    "StreamResult",
    "UnionFind",
    "Window",
    "WindowMatching",
    "__version__",
    "memory_circuit",
    "stream_memory",
    "surgery",
]

# Changelog

All notable changes to `stabilizer-qec`. The Python package follows
[semantic versioning](https://semver.org) from 1.0. Anything deprecated warns for at least one
minor release before a major release removes it. The Rust crate is versioned on its own.

## 1.0.0 — 2026-10-02

The first stable release. From here, semantic versioning: everything importable from
`stabilizer_qec` without a leading underscore keeps working, with the same meaning, through
every 1.x release; anything to be removed is deprecated for at least one minor release first.
A seed's shots stay the same within 1.x unless a release says otherwise.

### Removed

- The 0.4 functions and classes at the top level (`generate_circuit`, `sample_b8_batch`,
  `decode_b8`, `bb_*`, `surgery_*`, `Decoder`, `RotatedSurfaceCode`, ...), deprecated since 0.5.
  Each now raises an `AttributeError` naming its replacement. The extension keeps them as
  `stabilizer_qec._core`, which the repository's tools use and which is not part of the API.

### Changed

- The Python package is versioned on its own (`pyproject.toml`); the Rust crate stays below
  1.0 (`Cargo.toml`), its modules being the engine's internals.
- Development status: Production/Stable.

## 0.7.0 — 2026-10-02

### Added

- Measurement feedback and sweep-controlled gates: `CX`, `CY`, `CZ`, `XCZ` and `YCZ` with a
  measurement record (`rec[-k]`) or sweep bit as the Z-type control, mixed with ordinary pairs.
- `HERALDED_ERASE` and `HERALDED_PAULI_CHANNEL_1`.
- `Circuit.detector_error_model(approximate_disjoint_errors=...)`, Stim's option (`False`,
  `True` or a threshold): `PAULI_CHANNEL_2`, `ELSE_CORRELATED_ERROR` chains, heralded errors and
  a `PAULI_CHANNEL_1` with no independent equivalent enter an error model approximately, case
  by case, exactly as Stim approximates them.
- A workflow that publishes the Rust crate to crates.io by trusted publishing.

### Changed

- The Rust crate's `python` feature is off by default (maturin turns it on), so a Rust project
  gets the engine alone. The crate is versioned on its own and stays below 1.0; the Python
  package is the stable interface.

### Fixed

- An invalid record target (`CY 2 rec[-0]`) was re-parsed without end and overflowed the stack;
  it is a `ValueError`.

## 0.6.0 — 2026-10-01

### Added

- Stim's whole Clifford language. `S` is a primitive; the other 46 unitary one- and two-qubit
  gates (`S_DAG`, the square roots, the `H_` and `C_` families, `CY` and the `XC`/`YC` family,
  `SWAP`, `ISWAP`, `CXSWAP`, `CZSWAP`, `SQRT_XX`/`YY`/`ZZ`, ...) run as the shortest H/S/CX
  sequences whose tableaus equal Stim's, signs included. Y-basis resets and measurements
  (`RY`, `MY`, `MRY`), inverted targets (`!q`) on every measurement, `MPP`, `MXX`, `MYY`,
  `MZZ`, `SPP`, `SPP_DAG`, `E` / `CORRELATED_ERROR`, `ELSE_CORRELATED_ERROR`,
  `PAULI_CHANNEL_2`, `MPAD`, `I_ERROR`, `II_ERROR` and `II`. Every gate prints as written. As
  in Stim, an error model refuses `ELSE_CORRELATED_ERROR` and `PAULI_CHANNEL_2`; the samplers
  run them.
- A guide and API reference at qcompiler.jaspersands.com/api/, built from the docstrings, whose
  examples the test suite runs.
- The Rust crate is ready for crates.io (metadata, a README, docs.rs settings, Rust 1.87 or
  later, checked in CI).

### Changed

- The m2d reference run and the frame sampler match every instruction by name, so none can be
  skipped silently.

## 0.5.0 — 2026-10-01

The interface 1.0 will keep. The 0.4 functions still work, with a `DeprecationWarning` naming
their replacement, and go in 1.0.

### Added

- A public API on numpy arrays, named after the tools it is checked against:
  - `Circuit` and `DetectorErrorModel` (from Stim text or Stim's own objects; counts taken
    through `REPEAT` blocks without unrolling them), `Circuit.detector_error_model`,
    `compile_detector_sampler` and `compile_m2d_converter`, with Stim's array forms
    (`separate_observables`, `append_observables`, `bit_packed`).
  - Decoders on a detector error model with PyMatching's `decode` / `decode_batch`:
    `Matching` (`enable_correlations`), `BeliefMatching`, `BpOsd` and `WindowMatching`.
  - Decoders on a check matrix with `ldpc`'s names and attributes: `BpDecoder` and
    `BpOsdDecoder` (dense or scipy sparse matrices).
  - `memory_circuit`, `BivariateBicycleCode` (check matrices, logicals, automorphisms,
    memories, the gross code's gauging measurement), the `surgery` module and
    `stream_memory`.
- Reproducible seeding: a sampler's shots depend only on its seed and on how many it has drawn,
  not on the number of threads or the machine.
- A test suite (pytest and Hypothesis) run on Python 3.9 to 3.14 on Linux, macOS and Windows,
  and fuzzing of the parsers and decoders (a randomized Rust test in every build, and
  `cargo fuzz` targets run in CI).

### Changed

- The extension module is now `stabilizer_qec._core`, under a Python package; numpy is a
  dependency.
- A Rust panic surfaces as `RuntimeError` asking for a report, never `PanicException`.

### Fixed

- A detector error model's `repeat` blocks unrolled without limit (`repeat 18446744073709551615`
  hung); a model now refuses past 2²⁴ lines, detectors past index 2²⁴ − 1, offsets that
  overflow, and `logical_observable L64`.
- A subnormal fault probability (`error(1e-320)`) gave an infinite matching weight, which
  overflowed in correlated matching's path tracing and panicked.
- Circuit arguments that are not finite (`DETECTOR(nan)`, `inf`) are refused; a NaN coordinate
  made a circuit unequal to itself.
- Lattice-surgery programs check their size before they are built: distance 2⁶⁴ − 1 wrapped the
  qubit estimate to zero, and huge line or repeated merges allocated before any check. A
  program is held to Stim's qubit indices, 10,000 rounds and 2²⁴ qubit-rounds; a line to 32
  patches. Window schedules check `commit + 2 buffer` without overflow.
- Integers too large for the platform and numbers too large for a double raised
  `OverflowError`; they are `ValueError`s.

### Deprecated

- Every 0.4 function (`generate_circuit`, `sample_b8_batch`, `decode_b8`, `bb_*`, `surgery_*`,
  ...) and the `Decoder` and `RotatedSurfaceCode` classes.

## 0.4.0 — 2026-10-01

- A matcher as fast as PyMatching: correlated matching at 0.91–1.01× its time, plain at
  1.01–1.14×, its decisions unchanged (fingerprinted).
- Lattice-surgery programs: X⊗X merges, a logical CNOT, repeated Z⊗Z and line merges.
- Logical operations on the gross code: automorphisms and the gauging measurement.
- A fuller resource estimate (factories, Litinski's layouts, reaction time, storage).
- Bounds on input size: qubit indices, noise strengths, loop unrolling, loop counts, m2d's
  tableau, written-out rounds; shot-size checks that cannot wrap.

## 0.3.0 — 2026-09-27

- The first release on PyPI: circuits and error models in Stim's formats, sparse-blossom and
  correlated matching, belief-matching, BP+OSD and the bivariate bicycle codes, window
  decoding and streams, lattice surgery.

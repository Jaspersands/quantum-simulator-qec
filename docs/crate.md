# stabilizer_qec

A quantum error-correction simulator and decoder: circuits and detector error models in Stim's
formats, a bit-parallel sampler, exact and correlated matching, belief-matching, BP and BP+OSD,
window decoding, IBM's bivariate bicycle codes and lattice surgery. It is the engine behind the
[`stabilizer-qec`](https://pypi.org/project/stabilizer-qec/) Python package and the
[interactive explainer](https://qcompiler.jaspersands.com), and it agrees with Stim, PyMatching,
`ldpc` and `beliefmatching` on every check in its test suite.

```rust
use stabilizer_qec::{memory_circuit, Basis, DemOptions, Matching, Noise, SurfaceCode};

// A distance-5 surface-code memory under circuit noise, sampled and decoded.
let circuit = memory_circuit(SurfaceCode::Rotated, 5, 5, Noise::Sd6 { p: 0.004 }, Basis::Z)?;
let samples = circuit.detector_sampler(7)?.sample(10_000, 0); // 0 threads: every core
let dem = circuit.detector_error_model(&DemOptions::new().decompose_errors(true))?;
let predictions = Matching::with_correlations(&dem)?.decode_batch(&samples.detectors, 0)?;
let failures = (0..10_000).filter(|&s| predictions[s].flips(0) != samples.observables.get(s, 0)).count();
assert!(failures < 1_000);
# Ok::<(), stabilizer_qec::Error>(())
```

- **Circuits**: [`Circuit`] reads Stim's circuit language (every Clifford gate, all three bases,
  Pauli products, every noise channel, measurement feedback, `REPEAT`, tags, Pauli
  observables) and prints it back, or is built in code ([`Circuit::append`] with [`Target`]s,
  `+`, `*`); [`Circuit::detector_error_model`] builds the error model Stim's analyzer builds,
  fault for fault, its loops folded as Stim folds them (a d = 11 memory of 10,000 rounds in a
  tenth of a second); [`DetectorSampler`] samples 64 shots to a word, reproducibly from a seed on any number
  of threads; [`MeasurementConverter`] turns raw records into detection events.
- **Decoders** on an error model: [`Matching`] (exact, sparse blossom; plain or correlated, as
  fast as PyMatching), [`BeliefMatching`], [`BpOsd`], [`WindowMatching`]. On a check matrix:
  [`BpDecoder`] and [`BpOsdDecoder`], equal to `ldpc`'s. All `Send + Sync`.
- **Codes**: [`memory_circuit`] (rotated and XZZX surface codes), [`stream_memory`] (a million
  rounds, window-decoded as they stream), [`BivariateBicycleCode`] (the gross code, its
  automorphisms' logical action, and the gauging measurement of its logicals),
  [`lattice_surgery`] (Z⊗Z and X⊗X merges, a logical CNOT).

Shots are [`BitTable`]s, bit-packed as Stim's `b8` format. Every fallible call returns
[`Error`]; none panics on bad input.

## Stability

Semantic versioning from 1.0: the items in this documentation keep compiling, with the same
meaning, through every 1.x release (checked by `cargo-semver-checks` in CI). The engine's own
modules are public so the Python bindings, the site and the tools can reach them, but they are
hidden from this documentation and carry no such promise. The Python bindings sit behind the
`python` feature, off by default. The minimum supported Rust is 1.87.

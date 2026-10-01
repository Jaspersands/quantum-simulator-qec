# stabilizer_qec

A quantum error-correction simulator and decoder in Rust, the engine behind the
[`stabilizer-qec`](https://pypi.org/project/stabilizer-qec/) Python package and the
[interactive explainer](https://qcompiler.jaspersands.com).

- Circuits in Stim's language (`circuit`), every Clifford gate included, and detector error
  models built by walking a circuit backwards (`dem`), equal to Stim's fault for fault.
- A bit-parallel sampler, 64 shots to a word, that runs `REPEAT` blocks without unrolling them
  (`batch_sampler`), and raw measurements to detection events (`m2d`).
- Exact matching by sparse blossom and correlated matching (`dem_decoder`, `sparse`), as fast
  as PyMatching; belief propagation and belief-matching (`bp`, `belief`); BP+OSD (`osd`).
- Window decoders and streams for real-time decoding (`window`, `stream`).
- IBM's bivariate bicycle codes and their logical operations (`bb`, `bb_auto`, `bb_gauge`,
  `bb_circuit`), and lattice surgery (`surgery`).

```rust
use stabilizer_qec::{circuit::Circuit, dem::Dem, dem_decoder::DemDecoder};

let circuit = Circuit::parse("R 0 1\nX_ERROR(0.1) 0\nM 0 1\nDETECTOR rec[-2]\nOBSERVABLE_INCLUDE(0) rec[-2]").unwrap();
let dem = Dem::from_circuit(&circuit).unwrap();
let decoder = DemDecoder::new(&dem).unwrap();
assert_eq!(decoder.decode(&[0]).unwrap().observables, 1);
```

The Python package is the supported, stable interface. The crate is versioned on its own and
stays below 1.0: its modules are the engine's internals, and a minor version may change them.
The Python bindings sit behind the `python` feature, off by default.

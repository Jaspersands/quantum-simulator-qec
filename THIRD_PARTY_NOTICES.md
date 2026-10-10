# Third-party notices

stabilizer-qec is MIT-licensed (see `LICENSE`). Parts of it are ported from, or reproduce
material of, other projects under their own licences, which apply to those parts.

## Stim

Copyright 2021 Google LLC. Licensed under the Apache License, Version 2.0; the full text is in
`LICENSES/Apache-2.0.txt`. Source: https://github.com/quantumlib/Stim

Version 2.0 implements Stim's Python API with Stim's semantics and output. Where its behaviour is
defined by Stim's source rather than its documentation, the code here is ported from Stim 1.16.0,
modified to fit this crate. That covers:

- the gate table (`src/gate_data.rs`, generated from Stim by `tools/gen_gate_data.py`);
- the circuit, tableau, Pauli-string, flow, detecting-region and transformation algorithms in
  `src/clifford/` (among them Stim's reverse frame tracker, `with_inlined_feedback`,
  `time_reversed_for_flows`, `flow_generators` and the stabilizer conversions);
- the exports in `src/clifford/export.rs` (OpenQASM, Quirk) and their Python counterparts
  (Crumble URLs, the SAT problems in `python/stabilizer_qec/_circuit.py`);
- the flip simulator's update rules (`src/clifford/flip_sim.rs`);
- the detector-slice text diagram and the 3D (glTF) diagrams (`src/clifford/detslice.rs`,
  `src/clifford/gltf.rs`), including Stim's gate texture image
  (`src/clifford/gltf_texture.b64`), reproduced unmodified.

Stim's own docstring examples are run as a conformance suite (`tests/stim_doctests.py`); they
are read from an installed copy of Stim at test time, not distributed.

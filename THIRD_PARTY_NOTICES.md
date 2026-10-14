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

## sinter

sinter is part of Stim's repository (`glue/sample`): Copyright 2021 Google LLC, licensed under
the Apache License, Version 2.0 (`LICENSES/Apache-2.0.txt`).

`python/stabilizer_qec/sinter/` is sinter 1.16.0's source, modified: it uses this package in
place of Stim and PyMatching, fits lines in closed form in place of scipy's `linregress` and
`leastsq`, adds this package's decoders to the built-in ones, and prints its reprs under
`stabilizer_qec.sinter`. Each file says so at its top. sinter's own tests, modified the same way,
are in `tests/sinter_port/`.

## qec-util

Copyright (c) 2023 MarcSerraPeralta. Licensed under the MIT License:

> Permission is hereby granted, free of charge, to any person obtaining a copy of this software
> and associated documentation files (the "Software"), to deal in the Software without
> restriction, including without limitation the rights to use, copy, modify, merge, publish,
> distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the
> Software is furnished to do so, subject to the following conditions:
>
> The above copyright notice and this permission notice shall be included in all copies or
> substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
> BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
> NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
> DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
> OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

`python/stabilizer_qec/analysis/` (its `circuits`, `dems`, `dem_instrs`, `data_processing`,
`distance`, `mod2`, `performance`, `samplers` and `threshold` modules) is qec-util 0.5.0's
source, modified: it uses this package in place of Stim and PyMatching, and numpy in place of
networkx, galois, lmfit, uncertainties, scipy, gurobipy and pyyaml (each used instead when it is
installed, where qec-util's results depend on it), takes numpy arrays as well as xarray's, and
fixes an index overrun in `mod2.solve_linear_system`. Each file says so at its top.

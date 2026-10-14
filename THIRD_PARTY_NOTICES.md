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

## deltakit

deltakit-explorer and deltakit-decode: Copyright Riverlane. Licensed under the Apache License,
Version 2.0 (`LICENSES/Apache-2.0.txt`). Source: https://github.com/Deltakit/deltakit

`python/stabilizer_qec/analysis/_dk/` is deltakit-explorer 0.9.4's analysis (`_analysis`,
`_binomial_fit`, `_estimate`, `_lambda`, `_leppr`, `_quops`) and deltakit-decode 0.9.2's
`EmpiricalDecodingErrorDistribution`, modified: numpy versions of the scipy.optimize routines
and of uncertainties' propagation stand in when those packages are not installed
(`analysis/_scipy.py`, `analysis/_uncertain.py`), and the code runs on Python 3.9 and numpy 1.x.
Each file says so at its top. `analysis/pij.py`, `analysis/decoding.py`,
`analysis/error_budget.py` and `analysis/plotting.py` follow deltakit's methods and formulas
(the p_ij formulas are deltakit-decode's, reproduced) over this package's models.

## qsample

Copyright Don Winter. Licensed under the Apache License, Version 2.0 (`LICENSES/Apache-2.0.txt`).
Source: https://github.com/dpwinter/qsample

`python/stabilizer_qec/qsample/` is qsample 0.0.2's source, modified: it runs on this package's
tableau simulator (`qsample/sim.py`) in place of chp-sim and projectq, and needs none of
networkx, anytree, simpleeval, scipy, dill, latextools or pydot (`qsample/_graph.py` and
`qsample/_anytree.py` stand in for what it uses of them; matplotlib, tqdm, networkx and pydot are
used to plot, show progress and draw when installed). It also fixes `utils.load`, which opened its
file for writing. Each file says so at its top.

## ldpc

Copyright (c) 2022 Joschka Roffe. Licensed under the MIT License (the notice above, under
qec-util, applies with this copyright line). Source: https://github.com/quantumgizmos/ldpc

The BP, OSD, LSD, soft-information, bit-flipping and union-find decoders in `src/bp.rs`,
`src/osd.rs`, `src/lsd.rs`, `src/flip.rs` and `src/ldpc_uf.rs` follow ldpc 2.4.1's C++ step for
step, `src/mbp.rs` follows `mbp.hpp` from ldpc's main branch (built for the check by
`tools/mbp_reference.cpp`), and `python/stabilizer_qec/_ldpc.py` follows its Python classes'
arguments, defaults and properties, so that their results are ldpc's bit for bit.
`python/stabilizer_qec/ldpc/` is ldpc's Python package ported (each file marked as modified;
`mod2/_mod2.py` reimplements its Cython and C++ linear algebra), and `tests/ldpc_port/` is its
test suite (`python_test/`, with the code matrices in `tests/ldpc_port/pcms/`), run on the port.

## C and C++ standard library algorithms

ldpc's random serial schedule, serial-relative schedule and OSD depend on how the platform's
standard libraries shuffle and sort, so `src/cpp_random.rs` reproduces them:

- libc++'s `std::sort` and `uniform_int_distribution` / `std::shuffle` follow LLVM's libc++
  (`<__algorithm/sort.h>`, `<__random/uniform_int_distribution.h>`): Copyright the LLVM Project
  contributors, licensed under the Apache License, Version 2.0 with LLVM Exceptions
  (`LICENSES/Apache-2.0.txt`).
- The BSD `qsort` (Apple's libc) follows FreeBSD's `lib/libc/stdlib/qsort.c`, under this notice:

  > Copyright (c) 1992, 1993 The Regents of the University of California. All rights reserved.
  >
  > Redistribution and use in source and binary forms, with or without modification, are
  > permitted provided that the following conditions are met: 1. Redistributions of source code
  > must retain the above copyright notice, this list of conditions and the following
  > disclaimer. 2. Redistributions in binary form must reproduce the above copyright notice,
  > this list of conditions and the following disclaimer in the documentation and/or other
  > materials provided with the distribution. 3. Neither the name of the University nor the
  > names of its contributors may be used to endorse or promote products derived from this
  > software without specific prior written permission.
  >
  > THIS SOFTWARE IS PROVIDED BY THE REGENTS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED
  > WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
  > FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE REGENTS OR
  > CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
  > CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
  > SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
  > THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
  > OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
  > POSSIBILITY OF SUCH DAMAGE.
- Microsoft's `std::sort` (and `tools/cpp_msvc_sort.cpp`, which transcribes it to check the
  port) follows microsoft/STL (`stl/inc/algorithm`, `stl/inc/__msvc_heap_algorithms.hpp`):
  Copyright (c) Microsoft Corporation, licensed under the Apache License, Version 2.0 with LLVM
  Exceptions (`LICENSES/Apache-2.0.txt`). Its `std::hash<int>` is FNV-1a (`src/robin.rs`).
- libstdc++'s sort, shuffle and uniform distribution, and Microsoft's shuffle and uniform
  distribution, are reimplemented from their documented algorithms (introsort with a median of
  three and a final insertion sort; Lemire's nearly divisionless method; rejection sampling),
  not copied.

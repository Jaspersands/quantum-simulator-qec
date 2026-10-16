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

## relay_bp

Copyright IBM 2025. Licensed under the Apache License, Version 2.0 (`LICENSES/Apache-2.0.txt`).
Source: https://github.com/trmue/relay (PyPI `relay-bp`)

`src/relay.rs` and `src/relay_num.rs` port relay_bp 0.2.2's Relay-BP and min-sum decoders
(`crates/relay_bp/src/bp/`) with their arithmetic in their order, in each number type relay_bp is
built for; fixed point follows the `fixed` crate's rounding (reimplemented, not copied).
`python/stabilizer_qec/relay_bp/` reproduces its Python interface, and its `stim/` subpackage is
relay_bp's, each file marked as modified.

## PyMatching

Copyright 2022 Oscar Higgott. Licensed under the Apache License, Version 2.0
(`LICENSES/Apache-2.0.txt`). Source: https://github.com/oscarhiggott/PyMatching

The matching graph built by hand (`src/user_graph.rs`), its weight discretisation and the
`pymatching` command line follow PyMatching 2.4, modified to fit this crate.

## Tesseract

Copyright 2025 Google LLC. Licensed under the Apache License, Version 2.0
(`LICENSES/Apache-2.0.txt`). Source: https://github.com/quantumlib/tesseract-decoder

The search decoder (`src/search.rs`) ports Tesseract's A* search (`src/tesseract.cc`,
`src/common.cc`) with its arithmetic in its order, modified to fit this crate.
`python/stabilizer_qec/tesseract_decoder/` ports Tesseract's Python package: its bindings
(`src/*.pybind.h`), the multi-pass decoder (`src/multi_pass/`) and the simplex decoder
(`src/simplex.cc`, solved here by an exact search instead of HiGHS) rewritten in Python, and its
Python modules (`src/py/_tesseract_py_util/` as `demutil/`, `src/py/multi_pass_sinter_decoders.py`)
with their imports pointed at this package. `tests/tesseract_port/` holds Tesseract's Python
tests, modified the same way.

## Chromobius

Copyright 2023 Google LLC. Licensed under the Apache License, Version 2.0
(`LICENSES/Apache-2.0.txt`). Source: https://github.com/quantumlib/chromobius

The colour-code decoder (`src/color/`) ports Chromobius (`src/chromobius`) in its order,
modified to fit this crate.
`python/stabilizer_qec/chromobius/` ports Chromobius's Python bindings
(`src/chromobius/pybind/`) onto that decoder.

## beliefmatching

Copyright Oscar Higgott. Licensed under the Apache License, Version 2.0
(`LICENSES/Apache-2.0.txt`). Source: https://github.com/oscarhiggott/BeliefMatching

`python/stabilizer_qec/beliefmatching/` is beliefmatching 0.2.0's package with its imports
pointed at this package's ldpc, PyMatching, Stim and sinter.

## qLDPC

Copyright 2023-2024 The qLDPC Authors and Infleqtion Inc. (portions authored before June
2024); Copyright 2024-present The qLDPC Authors. Licensed under the Apache License, Version 2.0
(`LICENSES/Apache-2.0.txt`). Source: https://github.com/qLDPCOrg/qLDPC

`python/stabilizer_qec/qldpc/` is qLDPC 0.4.1's package, and `tests/qldpc_port/` its tests, with
their imports pointed at this package's Stim, sinter, ldpc, PyMatching, relay_bp and Tesseract
(and qldpc at stabilizer_qec.qldpc).

## CUDA-Q QEC

Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates. Licensed under the Apache License,
Version 2.0 (`LICENSES/Apache-2.0.txt`). Source: https://github.com/NVIDIA/cudaqx (`libs/qec`)

`python/stabilizer_qec/cudaq_qec/` ports CUDA-Q QEC's decoder interface and its decoders (the
lookup tables, `src/lookup.rs`; the sliding window; the PyMatching plugin; the bundled examples),
its parity-check-matrix tools and its detector error model, from its C++ and Python sources; its
`tensor_network_decoder` keeps CUDA-Q's interface and contracts exactly with `src/tn_ml.rs`.
Its codes (`stabilizer_grid`, the surface code's orientations, Steane, repetition), noise models
and memory experiments (`python/stabilizer_qec/cudaq_qec/_codes.py`) are ported from its C++
sources. `tests/cudaq_qec_port/` holds CUDA-Q QEC's decoder and code tests (the C++ code tests in
Python) with their import pointed at this package.

## Tsim

Copyright 2026 QuEra Computing Inc. Licensed under the Apache License, Version 2.0
(`python/stabilizer_qec/tsim/LICENSE`). Source: https://github.com/QuEraComputing/tsim

`python/stabilizer_qec/tsim/` holds bloqade-tsim 0.1.5's encoders and program-text utilities on
this package's `Circuit`, and `tests/tsim_port/` its program-text tests.

## qecsim

Copyright (c) 2016, qecsim developers. Licensed under the BSD 3-Clause License (its text is in
`python/stabilizer_qec/qecsim/LICENSE`). Source: https://github.com/qecsim/qecsim

`python/stabilizer_qec/qecsim/` is qecsim 1.0b9's package, and `tests/qecsim_port/` its tests,
with qecsim at stabilizer_qec.qecsim; its command line's models come from a table (its entry
points) rather than `pkg_resources`, and without Blossom V's library this package's exact matcher
serves its Blossom V interface.

## bposd

Copyright (c) 2021 Joschka Roffe. Licensed under the MIT License (`python/stabilizer_qec/bposd/LICENSE`).
Source: https://github.com/quantumgizmos/bp_osd

`python/stabilizer_qec/bposd/` is bposd 2.1's package on this package's ldpc.

## MQT QECC

Copyright (c) 2023 - 2026 Chair for Design Automation, TUM. Licensed under the MIT License
(`python/stabilizer_qec/mqt_qecc/LICENSE`). Source: https://github.com/munich-quantum-toolkit/qecc

`python/stabilizer_qec/mqt_qecc/` is MQT QECC 2.0.0's Python package, with its data files, and
`tests/mqt_qecc_port/` its tests, with mqt.qecc at stabilizer_qec.mqt_qecc and its Stim, sinter,
ldpc, PyMatching, qecsim and bposd this package's.

## Fusion Blossom

Copyright (c) 2022 Yue Wu. Licensed under the MIT License (`src/fusion/LICENSE`, and
`python/stabilizer_qec/fusion_blossom/LICENSE`). Source: https://github.com/yuewuo/fusion-blossom

`src/fusion/` is a port of Fusion Blossom 0.2.13's Rust crate (its solvers, example codes,
partitions and visualizer), with the parts of `parking_lot`, `rayon`, `weak_table`,
`priority-queue`, `rand_xoshiro` and `serde_json` it uses written out in `src/fusion/shim.rs`;
`src/py_fusion.rs` and `python/stabilizer_qec/fusion_blossom/` are its Python module, with its
helper and visualizer website files, and `tests/fusion_blossom_port/` its tests and scripts.
`shim.rs`'s `json!` macro is ported from serde_json (Copyright David Tolnay; MIT or Apache-2.0)
and its Xoshiro256** and SplitMix64 follow rand_xoshiro (Copyright the Rust Project Developers
and the Rand Project Developers; MIT or Apache-2.0).

## MWPF

Copyright (c) 2022 Yue Wu. Licensed under the MIT License (`src/mwpf/LICENSE`, and
`python/stabilizer_qec/mwpf/LICENSE`). Source: https://github.com/yuewuo/mwpf

`src/mwpf/` is a port of MWPF 0.2.12's Rust crate (its solvers, relaxers, matrices, example codes,
visualizer, HTML export, BP crate and command line), with the parts of `num-traits`,
`itertools`, `hashbrown`, `lnexp`, `prettytable-rs`, `rand`'s `SmallRng` (Xoshiro256++ seeded by
PCG32, following rand and rand_xoshiro: Copyright the Rust Project Developers and the Rand Project
Developers; MIT or Apache-2.0), `clap`'s parsing and error messages and `serde_json`'s parser
errors (Copyright David Tolnay; MIT or Apache-2.0) it uses written out in `src/mwpf/shim.rs`,
`src/mwpf/cli.rs` and `src/fusion/shim.rs`, and the `highs` crate's calls into HiGHS (Copyright
Ophir Lojkine and rust-or; MIT) in `src/mwpf/lp.rs`. `src/mwpf/cli_help.rs` holds the help texts MWPF's command line prints.
`python/stabilizer_qec/mwpf/` is its Python package and `tests/mwpf_port/` its tests.
`src/mwpf/visualize/standalone.html` is the visualizer MWPF's wheels embed (its `visualize/`
built by Vite), which bundles Vue, three.js, TroisJS and Tweakpane (each under the MIT License).

## HiGHS

Copyright (c) 2024 HiGHS. Licensed under the MIT License (`src/highs/LICENSE.txt`). Source:
https://github.com/ERGO-Code/HiGHS

`src/highs/` is HiGHS 1.8.0's library (its `src/` and `extern/`, as the `highs-sys` 1.6.4 crate
bundles it), which `build.rs` compiles for the `highs` feature that the Python wheels turn on; MWPF's
relaxer optimizer solves its linear programs with it. It includes IPX and BASICLU (Copyright
ERGO-Code; MIT), files from cuPDLP-C 0.3.0 (MIT), zstr 1.0.5 (Copyright Ontario Institute for
Cancer Research; MIT), the LP file reader `filereaderlp` (Copyright (c) 2020 Michael Feldmeier;
MIT, `src/highs/extern/filereaderlp/LICENSE`) and pdqsort (Copyright (c) 2021 Orson Peters; zlib
License, `src/highs/extern/pdqsort/license.txt`).

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

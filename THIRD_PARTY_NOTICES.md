# Third-party notices

The code written for stabilizer-qec is MIT-licensed (see `LICENSE`). Parts of the package are
ported from, or reproduce material of, other projects under their own licences (Apache-2.0, MIT,
BSD, zlib, and Apache-2.0 with LLVM exceptions), which apply to those parts and travel with them; each is listed below with its
copyright notice. The Rust crates the compiled module links are listed, with their licence texts,
in `THIRD_PARTY_CRATES.md` (see "Rust crates" below).

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
  (`src/clifford/gltf_texture.b64`), reproduced unmodified;
- the command line's help pages (`python/stabilizer_qec/_stim_help.json.gz`: what `stim help`
  prints for each instruction, result format and command, taken by `tools/stim_help.py`),
  reproduced unmodified but for `stim <command>` reading `stabilizer-qec <command>`;
- the flag parsing and first-shot reading of `pymatching animate`
  (`python/stabilizer_qec/pymatching.py`), from the Stim that PyMatching 2.4.0 builds with
  (commit 1320ad7e: `util_bot/arg_parse.cc`, `io/measure_record_reader.inl`), with its messages.

Stim's own docstring examples are run as a conformance suite (`tests/stim_doctests.py`); they
are read from an installed copy of Stim at test time, not distributed.
`tests/data/search_corpus.json.gz` (written by `tools/search_corpus.py`) records Stim 1.16.0's
answers to `search_for_undetectable_logical_errors` on 1,200 random models at five sets of
limits, which the tests compare against.

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

deltakit-circuit, deltakit-core, deltakit-decode and deltakit-explorer: Copyright Riverlane.
Licensed under the Apache License, Version 2.0 (`LICENSES/Apache-2.0.txt`). Source:
https://github.com/Deltakit/deltakit

`python/stabilizer_qec/deltakit/` is Deltakit whole (deltakit-circuit and -core 0.9.2, -decode
0.9.2, -explorer 0.9.4) on this package's Stim (`stabilizer_qec.deltakit_stim`, Deltakit's Stim
with its leakage instructions) and PyMatching; its `LICENSE` is beside it, and each modified
file says so at its top.

`python/stabilizer_qec/analysis/_dk/` is deltakit-explorer 0.9.4's analysis (`_analysis`,
`_binomial_fit`, `_estimate`, `_lambda`, `_leppr`, `_quops`) and deltakit-decode 0.9.2's
`EmpiricalDecodingErrorDistribution`, modified: numpy versions of the scipy.optimize routines
and of uncertainties' propagation stand in when those packages are not installed
(`analysis/_scipy.py`, `analysis/_uncertain.py`), and the code runs on Python 3.9 and numpy 1.x.
Each file says so at its top. `analysis/pij.py`, `analysis/decoding.py`,
`analysis/error_budget.py` and `analysis/plotting.py` follow deltakit's methods and formulas
(the p_ij formulas are deltakit-decode's, reproduced) over this package's models.

## PECOS

Copyright 2018 National Technology & Engineering Solutions of Sandia, LLC (NTESS). Licensed under
the Apache License, Version 2.0 (`LICENSES/Apache-2.0.txt`). Source:
https://github.com/PECOS-packages/PECOS

`python/stabilizer_qec/pecos/` is PECOS whole, on this package's engines; its `LICENSE` is
beside it, and each modified file says so at its top.

## PanQEC

Copyright (c) 2022 PanQEC (Eric Huang and contributors). MIT License. Source:
https://github.com/panqec/panqec

`python/stabilizer_qec/panqec/` is PanQEC 0.1.7 whole, on this package's ldpc and PyMatching;
its `LICENSE` is beside it, and each modified file says so at its top.

## QUITS

Copyright (c) 2026 Mingyu Kang (QUITS: Kang, Lin, Yao and Gökduman). MIT License. Source:
https://github.com/mkang315/QUITS

`python/stabilizer_qec/quits/` is QUITS 1.2.0 whole, on this package's Stim and ldpc; its
`LICENSE` is beside it, and each modified file says so at its top.

## qsample

Copyright Don Winter. Licensed under the Apache License, Version 2.0 (`LICENSES/Apache-2.0.txt`).
Source: https://github.com/dpwinter/qsample

`python/stabilizer_qec/qsample/` is qsample 0.0.2's source, modified: it runs on this package's
tableau simulator (`qsample/sim.py`) in place of chp-sim and projectq, and needs none of
networkx, anytree, simpleeval, scipy, dill, latextools or pydot (`qsample/_graph.py` and
`qsample/_anytree.py` stand in for what it uses of them; matplotlib, tqdm, networkx and pydot are
used to plot, show progress and draw when installed, and dill to save, as qsample does, when it
is installed: without it, saving uses pickle, which cannot save lambda or nested check
functions). It also fixes `utils.load`, which opened its file for writing, makes `utils.save`
replace its file only once the new one is written whole, gives each `Protocol` its own
check-function dict, and computes subset probabilities that stay finite past 170 fault
locations. Each file says so at its top.

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
`tests/ldpc_lsd_merge/captured.json` (written by `tools/lsd_merge_capture.py`) records outputs
of ldpc 2.4.1's `BpLsdDecoder` the tests compare against: its answers over 50 decodes in one
process, on the window decodes of QUITS 1.2.0's BP+LSD sweep where they vary or differ from
this package's, with the window check matrices built from QUITS's HGP n=12 code
(`n=12_dv=3_dc=4_dist=6`).

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
`pymatching` command line follow PyMatching 2.4, modified to fit this crate. `src/pm_exact/`
transliterates PyMatching 2.4.0's sparse blossom (`src/pymatching/sparse_blossom/`: its
tracker, flooder, matcher, search flooder and decoding driver) to Rust, and builds its matching
graph in PyMatching 2.4.0's and 2.2.2's neighbour orders (their `driver/user_graph.cc`, and
2.2.2's `driver/io.h`); "Copyright 2022 PyMatching Contributors", its notice in
`src/pm_exact/LICENSE`, each file marked as modified. `src/pm_exact/diagram.rs` and the
`animate` command in `python/stabilizer_qec/pymatching.py` port PyMatching 2.4.0's `animate`
(`diagram/mwpm_diagram.cc`, `diagram/animation_main.cc`), under the same notice.
`tests/pymatching_port/animate/` (written by `tools/pymatching_animate_fixtures.py`) records the
frames, messages and exit codes of PyMatching 2.4.0's `animate` (its macOS arm64 wheel, and its
C++ command line built from its sdist), which the tests compare against.

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
tests, modified the same way. `python/stabilizer_qec/tesseract_decoder/__main__.py` ports its
`tesseract` program (`src/tesseract_main.cc`), reproducing (not copying) the flag parsing,
errors and help of the library it is built with, p-ranav/argparse 3.1 (Copyright (c) 2018
Pranav Srinivas Kumar; MIT License; https://github.com/p-ranav/argparse), and the way the
library it writes its `--stats-out` JSON with, nlohmann/json (Copyright Niels Lohmann; MIT
License; https://github.com/nlohmann/json), writes numbers and objects.
`tests/data/tesseract_cli_reference.json` and `tests/data/tesseract_wide_reference.json` record
outputs of Tesseract's 0.1.1.dev20261013012529 build through its Python API (for its program's
flag sets on stored shots, and on a model of 100 observables), which the tests compare against.

## Chromobius

Copyright 2023 Google LLC. Licensed under the Apache License, Version 2.0
(`LICENSES/Apache-2.0.txt`). Source: https://github.com/quantumlib/chromobius

The colour-code decoder (`src/color/`) ports Chromobius (`src/chromobius`) in its order,
modified to fit this crate.
`python/stabilizer_qec/chromobius/` ports Chromobius's Python bindings
(`src/chromobius/pybind/`) onto that decoder, and its `__main__.py` Chromobius's command-line
program (`src/main.cc`, `src/chromobius/commands/`) with the argument parsing and record
readers and writers it takes from Stim (at 1320ad7: `src/stim/util_bot/arg_parse.cc`,
`src/stim/io/measure_record_reader.inl`, `measure_record_writer.cc`; Stim's notice above).
Its Möbius matching runs on `src/pm_exact/` in PyMatching 2.2.2's neighbour order (PyMatching's
notice above). `tests/data/chromobius_cli_reference.json.gz` (written by
`tools/chromobius_cli_reference.py`) records the outputs and messages of Chromobius 1.1.1's
program, built from source with Stim 1320ad7 and PyMatching 2.2.2, on 184 cases, which the tests
compare against.

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

CUDA-Q QEC 0.8.1 is distributed with a NOTICE file (at the root of cudaqx and in `libs/qec`),
whose attribution is reproduced here as Apache-2.0's section 4(d) asks: "CUDA-QX. This product
includes software developed by NVIDIA corporation and affiliates and includes material from third
parties." Of the third-party works that NOTICE lists, Stim and PyMatching pertain to the parts
ported here, and their notices are above; pybind11, nlohmann/json, {fmt}, BLAS, xtl and xtensor
do not, as nothing of them is in this package (the Python bindings' pybind11 casters, among them
`sparse_matrix_casters.h`, are rewritten in Python, not copied).

`python/stabilizer_qec/cudaq_qec/` ports CUDA-Q QEC's decoder interface and its decoders (the
lookup tables, `src/lookup.rs`; the sliding window; the PyMatching plugin; the bundled examples),
its parity-check-matrix tools and its detector error model, from its C++ and Python sources; its
`tensor_network_decoder` keeps CUDA-Q's interface and contracts exactly with `src/tn_ml.rs`.
Its codes (`stabilizer_grid`, the surface code's orientations, Steane, repetition), noise models
and memory experiments (`python/stabilizer_qec/cudaq_qec/_codes.py`) are ported from its C++
sources. Its real-time decoding configuration (`_config.py`, with YAML read and written by
`_yaml.py`), its host decoding library (`_realtime.py`) and its deprecated typed configurations
(`_compat.py`) are ported from `lib/realtime/`, `lib/decoder.cpp` and its Python bindings. Its
extended detector error model and `dem_chunks` (`_extended_dem.py`) and the DEM of CSS code
matrices (`_css.py`) are ported from `include/cudaq/qec/extended_dem.h`,
`include/cudaq/qec/dem_chunks_memory.h`, `include/cudaq/qec/code_matrices.h`,
`include/cudaq/qec/sparse_binary_matrix.h`, `include/cudaq/qec/dem_construction_code.h`,
`lib/extended_dem.cpp`, `lib/dem_chunks_memory.cpp`, `lib/sparse_binary_matrix.cpp`,
`lib/dem_construction.cpp`, `lib/dem_construction_code.cpp`, `lib/dem_construction_utils.cpp`,
`python/bindings/py_extended_dem.cpp`, `python/bindings/py_dem_construction.cpp` and
`python/bindings/sparse_matrix_casters.h`. Its `trt_decoder` (`_trt.py`) is ported from
`lib/decoders/plugins/trt_decoder/trt_decoder.cpp`, its network run on the CPU by `_onnx.py`
(this package's reader of ONNX models, by the messages and field numbers of `onnx/onnx.proto`,
Copyright (c) ONNX Project Contributors, Apache License 2.0, https://github.com/onnx/onnx, and its
evaluator, whose float32 arithmetic and float8 conversion follow onnxruntime's, below); how
TensorRT's ONNX parser imports quantization (and the messages it refuses with), and the input
types it refuses (uint8 to DequantizeLinear, int32 and int64 to MatMul and Gemm), follow
onnx-tensorrt (Copyright (c) NVIDIA Corporation, Apache License 2.0,
https://github.com/onnx/onnx-tensorrt). `tests/cudaq_qec_port/models/surface_code_decoder.onnx`
is CUDA-Q's own model (`assets/tests/`, a git-lfs object of https://github.com/NVIDIA/cudaq-qec,
as cudaqx is now named); the other models there are built by `tools/cudaq_trt_models.py`, among
them `qdq_int8.onnx`, `qdq_fp8.onnx`, `identity_dynamic.onnx`, `identity_uint8.onnx` and
`cast_uint8_float.onnx` by CUDA-Q's own generators, which that tool mirrors.
`tests/cudaq_qec_port/` holds CUDA-Q QEC's decoder, code and configuration tests (the C++ code
tests in Python) with their import pointed at this package, among them, from CUDA-Q QEC 0.8.1,
its extended-DEM and DEM-construction tests (`test_extended_dem.py`), its chunk-form YAML tests
(`test_decoders_yaml_chunks.py`) and its `trt_decoder` tests (`test_trt_decoder.py`).

## onnxruntime

Copyright (c) Microsoft Corporation. Licensed under the MIT License:

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

`python/stabilizer_qec/cudaq_qec/_onnx.py` uses selected arithmetic from onnxruntime
(https://github.com/microsoft/onnxruntime): a fixed MLAS float32 product order with
once-rounded fused multiply-adds, the MatMul + Add fusion of its default optimizer
(MatMulAddFusion, as its 1.31.0 optimized graphs show it), MLAS's logistic coefficients and the
float8e4m3fn conversion of `core/framework/float8.h`. The 180 recordings on 23 models in
`tests/cudaq_qec_port/models/ort_outputs.npz` come from ORT 1.31.0 CPU on macOS arm64. They do
not establish equality to other optimizer, prepacking, weight-DQ fusion or reduced-precision
paths; threshold differences can change a binary correction.
See [the selected numerical and input contract](python/stabilizer_qec/cudaq_qec/ONNX_CONTRACT.md).

## Tsim

Copyright 2026 QuEra Computing Inc. Licensed under the Apache License, Version 2.0
(`python/stabilizer_qec/tsim/LICENSE`). Source: https://github.com/QuEraComputing/tsim

`python/stabilizer_qec/tsim/` holds bloqade-tsim 0.1.5's encoders, program-text utilities and
its detector error model with non-deterministic observables (`noise/dem.py`) on this package's
`Circuit`, and `tests/tsim_port/` its program-text tests and the gate matrices its tests check
gates against. `python/stabilizer_qec/tsim/zx/` holds its ZX graphs (`core/parse.py`,
`core/instructions.py`, `core/graph.py`, the diagrams of `utils/diagram.py`) with the noise
channels' probability tables (`noise/channels.py`) and `find_basis` (`utils/linalg.py`);
`tests/tsim_port/zx_graphs.json.gz` records its graphs, tensors and T-counts on 40 circuits
(`tools/tsim_zx_fixtures.py`).

`python/stabilizer_qec/tsim/zx/graph.py` and `tensor.py` are ported from pyzx_param 0.9.3
(https://github.com/rafaelha/pyzx, a parametric fork of PyZX): the simple graph backend, the
scalar, the vertex and edge types and the tensor contraction of Copyright (C) 2018 - Aleks
Kissinger and John van de Wetering, licensed under the Apache License, Version 2.0 (the licence
is in `python/stabilizer_qec/tsim/LICENSE`, after Tsim's). pyzx_param and PyZX are otherwise
optional: the graphs are built on this package's own, and `ZXGraph.to_pyzx()` and the diagrams use
them when installed.

## qecsim

Copyright (c) 2016, qecsim developers. Licensed under the BSD 3-Clause License (its text is in
`python/stabilizer_qec/qecsim/LICENSE`). Source: https://github.com/qecsim/qecsim

`python/stabilizer_qec/qecsim/` is qecsim 1.0b9's package, and `tests/qecsim_port/` its tests,
with qecsim at stabilizer_qec.qecsim; its command line's models come from a table (its entry
points) rather than `pkg_resources`, and without Blossom V's library this package's exact matcher
serves its Blossom V interface, which numbers the nodes in order of first appearance (qecsim's in
a set's order, which follows memory addresses for nodes that hash by identity), so a seeded run
repeats. Its logging configuration (`logging_qecsim.ini`, whose handler arguments
`logging.config.fileConfig` evaluates) and Blossom V's library (`clib/libpypm.so`, which ctypes
loads) are read from `$QECSIM_CFG` when it is set (an empty value counting as unset, where qecsim
takes it as the working directory) and from `~/.qecsim` otherwise, as qecsim reads them, but not
the working directory, which qecsim searches first: both files are code, so they come only from
where the user put them, not from whatever directory the command runs in.

## bposd

Copyright (c) 2021 Joschka Roffe. Licensed under the MIT License (`python/stabilizer_qec/bposd/LICENSE`).
Source: https://github.com/quantumgizmos/bp_osd

`python/stabilizer_qec/bposd/` is bposd 2.1's package on this package's ldpc.

## stimbposd

Copyright Oscar Higgott. Licensed under the Apache License, Version 2.0
(`python/stabilizer_qec/stimbposd/LICENSE`, the licence its source distribution carries; its
package metadata's classifier says MIT). Source: https://github.com/oscarhiggott/stimbposd

`python/stabilizer_qec/stimbposd/` is stimbposd 0.2.0's package, and `tests/stimbposd_port/` its
tests and their test data, with their imports pointed at this package's Stim, ldpc and sinter.

## MQT QECC

Copyright (c) 2023 - 2026 Chair for Design Automation, TUM. Licensed under the MIT License
(`python/stabilizer_qec/mqt_qecc/LICENSE`). Source: https://github.com/munich-quantum-toolkit/qecc

`python/stabilizer_qec/mqt_qecc/` is MQT QECC 2.0.0's Python package, with its data files, and
`tests/mqt_qecc_port/` its tests, with mqt.qecc at stabilizer_qec.mqt_qecc and its Stim, sinter,
ldpc, PyMatching, qecsim and bposd this package's.

## tqec

`python/stabilizer_qec/tqec/` is a port of tqec's `computation` package (block graphs, cubes,
pipes, correlation surfaces, port filling), its `utils` (positions, enums, rotations, the
`LinearFunction` of `scale`), its gallery, its Collada and JSON readers and writers, its
`simulation` package (on this package's sinter) and its `interop.pyzx` package (positioned ZX
graphs, block synthesis, plots; PyZX optional) (tqec/tqec 0.2.0,
https://github.com/tqec/tqec), rewritten on the standard library without networkx, PyZX,
pycollada or numpy, with tqec's Steane-encoding Collada file packaged as data.
`python/stabilizer_qec/tqec/_compile/` is a port of tqec's compile pipeline (its `circuit`,
`plaquette`, `templates` and `compile` packages and the `utils` modules they use), on this
package's Stim, with the detector database kept in memory. tqec is the TQEC community's,
licensed under the Apache License, Version 2.0; the licence is in
`python/stabilizer_qec/tqec/LICENSE`. `tests/tqec_port/fixtures/` (written by
`tools/tqec_fixtures.py`) records the circuits tqec 0.2.0 and tqecd 0.2.1 write, and the
refusals they raise, on 57 cases (`manifest.json`), which the tests compare against byte for
byte.

`python/stabilizer_qec/tqec/noise_model.py` is a port of tqec's `utils/noise_model.py`, which
tqec recovered from Gidney, C. (2022), Data for "Inplace Access to the Surface Code Y Basis",
https://doi.org/10.5281/zenodo.7487893, licensed under the Creative Commons Attribution 4.0
International licence (https://creativecommons.org/licenses/by/4.0/legalcode). tqec modified it
(formatting, typing, no depolarizing error after measurements in `uniform_depolarizing` and
`si1000`, no two-body measurement noise, TICKs around REPEAT blocks). This port also changes it,
writing this package's circuits in place of Stim's.

## tqecd

Copyright the TQEC community. Licensed under the Apache License, Version 2.0
(`python/stabilizer_qec/tqec/_compile/tqecd/LICENSE`). Source: https://github.com/tqec/tqecd

`python/stabilizer_qec/tqec/_compile/tqecd/` is tqecd 0.2.1's package (without its
`construction` module, whose work tqec's compile pipeline does itself), on this package's Stim.

## Loom

`python/stabilizer_qec/loom/` is a port of Loom's Eka model (`loom.eka`: lattice, Pauli
operators, stabilizers, blocks, syndrome circuits, the circuit IR and the operations), its
interpreter (`loom.interpreter` and the rotated surface code's grow, shrink, merge and split
applicators), its validator (`loom.validator`), its rotated-surface-code factory
(`loom_rotated_surface_code`, with its move, rotate, move-corners, Y-wall, state-injection,
transversal-Hadamard and auxiliary-CNOT applicators), its other code factories (the repetition
chain with its grow, shrink, merge and split, and the Shor, Steane and five-qubit codes), its
circuit error models (`loom.executor.circuit_error_model`) and its Stim, OpenQASM 3, PennyLane,
CUDA-Q, Mimiq and Guppy executors (`loom.executor`) from el-loom 0.4.0
(https://github.com/entropicalabs/el-loom), rewritten on plain dataclasses and this package's
tableau simulator without pydantic, igraph, plotly, scipy or numpy. Loom is Copyright 2024
Entropica Labs Pte Ltd and licensed under the Apache License, Version 2.0; the licence is in
`python/stabilizer_qec/loom/LICENSE`. `tests/loom_port/fixtures.json`, `fixtures_executor.json`,
`fixtures_gaps.json`, `fixtures_ops_a.json` and `fixtures_ops_b.json` record outputs of el-loom
0.4.0 the tests compare against.

Loom's if/else circuits are not ported. In el-loom 0.4.0 the file that defines them,
`src/loom/eka/ifelse_circuit.py`, carries in place of the Apache-2.0 header a notice that
prohibits its use, distribution and reproduction in source or compiled form without the express
written consent of Entropica Labs Pte Ltd; nothing of it is in this package.

## Qualtran

`python/stabilizer_qec/estimate/qualtran/` is a port of Qualtran's `qualtran.surface_code`
package, the `GateCounts` of `qualtran.resource_counting` and the `TComplexity` of
`qualtran.cirq_interop` (quantumlib/Qualtran 0.7.0, https://github.com/quantumlib/Qualtran),
rewritten on dataclasses, plain numbers and numpy without attrs, sympy or Cirq. Its `ui` is
ported whole: the estimates and figures without Dash, plotly or pandas; the Dash app (its
layout, built without Dash, and its callback), made with Dash where dash is installed; and,
where it is not, a page of this package's own (`_ui_static/ui.html`) that the standard
library's HTTP server sends to run the same app. The page loads plotly.js from the installed
plotly package or from cdn.plot.ly; plotly.js is not distributed here. Qualtran is Copyright
2023–2024 Google LLC and licensed under the Apache License, Version 2.0; the licence is in
`python/stabilizer_qec/estimate/LICENSE`. `tests/estimate_port/qualtran_ui.json.gz` records
outputs of Qualtran 0.7.0's `ui.update`, and `tests/estimate_port/qualtran_layout.json` the
title, layout and callback of its Dash app (with dash 4.4.1), that the tests compare against.

## Azure Quantum Resource Estimator

`python/stabilizer_qec/estimate/azure.py` is a port of the QDK's resource estimator
(microsoft/qdk, `source/resource_estimator/src`, the `estimates` and `system` modules:
physical qubits, fault-tolerance protocols, logical counts, the error budget, distillation
units, round-based factories, the exhaustive T-factory search and the physical estimation;
https://github.com/microsoft/qdk), rewritten in Python. It is Copyright (c) Microsoft Corporation
and licensed under the MIT License; the licence is in `python/stabilizer_qec/estimate/LICENSE-qdk`.
The formulas of its QEC schemes and distillation units (logical cycle time, qubits per logical
qubit, output error rates) are read and computed by fasteval 0.2.4's parser, constant folding and
evaluator, ported to Python: fasteval (https://github.com/likebike/fasteval), the crates.io
expression evaluator the QDK compiles them with, is not Microsoft's but Copyright (c) 2019
Christopher Sebastian, licensed under the MIT License, whose notice is in `LICENSE-qdk` after the
QDK's. The binomial quantile and
mass function in it follow the `probability` crate 0.20 (Ivan Ukhov and contributors,
https://github.com/stainless-steel/probability) and its incomplete beta function the `special`
crate (https://github.com/stainless-steel/special), both Apache-2.0 OR MIT.
`python/stabilizer_qec/estimate/azure_params.py` ports the QDK's parameter classes
(`qdk/estimator/_estimator.py` of the qdk package 1.33.1), and
`python/stabilizer_qec/estimate/azure_report.py` its report (the same file's HTML tables, diagram
and plot) and the space and scatter charts of its notebook widgets (the qsharp_widgets package
1.33.1, built from `source/npm/qsharp/ux` of microsoft/qdk), all Copyright (c) Microsoft
Corporation under the same MIT License (`LICENSE-qdk`).
`tests/estimate_port/qdk_estimates.json` records outputs of qdk 1.33.1 the tests compare
against, and `qdk_reports.json.gz` and `qdk_params.json` there (recorded by
`tools/qdk_report_fixtures.py`) its report's HTML and text, the widgets' charts and its
parameter classes' dictionaries and messages.

## Rust crates

Each wheel's compiled module (`stabilizer_qec/_core`) links Rust crates: PyO3 (the Python
bindings) and rand, and, for the `gpu` feature the wheels turn on, `wgpu`
(https://github.com/gfx-rs/wgpu, the gfx-rs developers), its shader translator naga and `pollster`
(Joshua Barretto), with what those depend on for each platform: objc2 and block2 on macOS, ash and
ordered-float on Linux and Windows, the windows crates on Windows, and libloading, libm, foldhash
and others on all. `THIRD_PARTY_CRATES.md`, which every wheel carries, lists them for each wheel's
target with their licences (MIT, Apache-2.0, ISC, Zlib, BSD-2-Clause and Unicode-3.0, alone or as
alternatives) and reproduces the licence texts and notices their packages carry;
`tools/third_party_crates.py` writes it from Cargo.lock and `cargo tree`. wgpu runs the lane
sampler's and the min-sum decoder's WGSL kernels (`src/gpu/`) on Metal, Vulkan and DX12; the
kernels are this package's own.

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

`python/stabilizer_qec/fusion_blossom/helper/bottle.py`, on which the visualizer's server runs, is
Bottle 0.14-dev as Fusion Blossom 0.2.13 embeds it, reproduced unmodified (Copyright (c)
2009-2024, Marcel Hellkamp; licensed under the MIT License, its notice appended to
`python/stabilizer_qec/fusion_blossom/LICENSE`; source: https://github.com/bottlepy/bottle).
`tests/fusion_blossom_port/helper_server.json` records Fusion Blossom's server's answers, which
the tests compare against.

## MWPF

Copyright (c) 2022 Yue Wu. Licensed under the MIT License (`src/mwpf/LICENSE`, and
`python/stabilizer_qec/mwpf/LICENSE`). Source: https://github.com/yuewuo/mwpf

`src/mwpf/` is a port of MWPF 0.2.12's Rust crate (its solvers, relaxers, matrices, example codes,
visualizer, HTML export, BP crate and command line), with the parts of `num-traits`,
`itertools`, `hashbrown`, `lnexp`, `prettytable-rs`, `rand`'s `SmallRng` (Xoshiro256++ seeded by
PCG32, following rand and rand_xoshiro: Copyright the Rust Project Developers and the Rand Project
Developers; MIT or Apache-2.0), `clap`'s parsing and error messages, `serde_json`'s parser
errors (Copyright David Tolnay; MIT or Apache-2.0) and `miniz_oxide` 0.8.0's deflate compressor
(the level-6 path `flate2`'s `GzEncoder` runs, with which the HTML export gzips its data;
Copyright (c) 2017 Frommi; MIT, Zlib or Apache-2.0) it uses written out in `src/mwpf/shim.rs`,
`src/mwpf/cli.rs` and `src/fusion/shim.rs`, and the `highs` crate's calls into HiGHS (Copyright
Ophir Lojkine and rust-or; MIT) in `src/mwpf/lp.rs`. `src/mwpf/cli_help.rs` holds the help texts MWPF's command line prints.
`python/stabilizer_qec/mwpf/` is its Python package and `tests/mwpf_port/` its tests.
`src/mwpf/visualize/standalone.html` is the visualizer MWPF's macOS wheels embed (its `visualize/`
built by Vite; each platform's wheels embed their own build), which bundles Vue, three.js, TroisJS and Tweakpane (each under the MIT License).

## HiGHS

Copyright (c) 2024 HiGHS. Licensed under the MIT License, quoted below (`src/highs/LICENSE.txt`
in the source). Source: https://github.com/ERGO-Code/HiGHS

`src/highs/` is HiGHS 1.8.0's library (its `src/` and `extern/`, as the `highs-sys` 1.6.4 crate
bundles it), which `build.rs` compiles for the `highs` feature that the Python wheels turn on, so
every wheel's compiled module holds it; MWPF's relaxer optimizer solves its linear programs with
it. It includes IPX and BASICLU (Copyright ERGO-Code; MIT, under HiGHS's licence below), files
from cuPDLP-C 0.3.0 (MIT, under HiGHS's licence below), zstr 1.0.5 (Copyright 2015 Ontario
Institute for Cancer Research; MIT, which its README in HiGHS gives as HiGHS's licence below), the
LP file reader `filereaderlp` (Copyright (c) 2020 Michael Feldmeier; MIT) and pdqsort (Copyright
(c) 2021 Orson Peters; zlib License). The wheels carry none of `src/highs/`, so the three licence
files there are quoted here.

HiGHS's licence (`src/highs/LICENSE.txt`):

> MIT License
>
> Copyright (c) 2024 HiGHS
>
> Permission is hereby granted, free of charge, to any person obtaining a copy
> of this software and associated documentation files (the "Software"), to deal
> in the Software without restriction, including without limitation the rights
> to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
> copies of the Software, and to permit persons to whom the Software is
> furnished to do so, subject to the following conditions:
>
> The above copyright notice and this permission notice shall be included in all
> copies or substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
> IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
> FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
> AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
> LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
> OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
> SOFTWARE.

filereaderlp's licence (`src/highs/extern/filereaderlp/LICENSE`):

> Copyright (c) 2020 Michael Feldmeier
>
> Permission is hereby granted, free of charge, to any person obtaining a copy
> of this software and associated documentation files (the "Software"), to deal
> in the Software without restriction, including without limitation the rights
> to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
> copies of the Software, and to permit persons to whom the Software is
> furnished to do so, subject to the following conditions:
>
> The above copyright notice and this permission notice shall be included in all
> copies or substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
> IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
> FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
> AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
> LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
> OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
> SOFTWARE.

pdqsort's licence (`src/highs/extern/pdqsort/license.txt`):

> Copyright (c) 2021 Orson Peters <orsonpeters@gmail.com>
>
> This software is provided 'as-is', without any express or implied warranty. In no event will the
> authors be held liable for any damages arising from the use of this software.
>
> Permission is granted to anyone to use this software for any purpose, including commercial
> applications, and to alter it and redistribute it freely, subject to the following restrictions:
>
> 1. The origin of this software must not be misrepresented; you must not claim that you wrote the
>    original software. If you use this software in a product, an acknowledgment in the product
>    documentation would be appreciated but is not required.
>
> 2. Altered source versions must be plainly marked as such, and must not be misrepresented as
>    being the original software.
>
> 3. This notice may not be removed or altered from any source distribution.

## QuantumClifford.jl

Copyright (c) 2023 Stefan Krastanov. Licensed under the MIT License (the notice above, under
qec-util, with this copyright line). Source: https://github.com/QuantumSavory/QuantumClifford.jl

`python/stabilizer_qec/clifford.py` ports QuantumClifford.jl 0.11.8's canonical forms (clipped
gauge, partial-trace and Gottesman forms), entanglement entropies, graph-state conversion and
Clifford enumeration, modified to work on this package's Pauli strings and tableaux with qubits
counted from 0. `src/perturb.rs` follows its perturbative expansion (`petrajectories`). Its
tests' pinned values are checked in `tests/test_clifford.py` and `tests/test_perturbative.py`.

## Quantumsim

Quantumsim 0.2.0 (https://gitlab.com/quantumsim/quantumsim) is used only as a reference in
tests (`tests/test_density.py`, when it is installed); none of its code is in this package.

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

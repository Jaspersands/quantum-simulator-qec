# Quantum Error Correction (QEC) Simulator

[![CI](https://github.com/Jaspersands/quantum-simulator-qec/actions/workflows/ci.yml/badge.svg)](https://github.com/Jaspersands/quantum-simulator-qec/actions/workflows/ci.yml)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23212075.svg)](https://doi.org/10.5281/zenodo.23212075)

A Rust stabilizer circuit simulator and decoder. It covers rotated and XZZX surface codes, IBM's
bivariate bicycle codes (the gross code) and lattice surgery between surface-code patches, and it
reads and writes Stim's circuit and error-model formats. It agrees with Stim and PyMatching on every
check in [Checked against Stim and PyMatching](#checked-against-stim-and-pymatching), and it decodes
Google's Willow and Sycamore hardware data. It compiles to WebAssembly for an interactive browser
explainer, and to a typed Python package, [`stabilizer-qec`](https://pypi.org/project/stabilizer-qec/).

The website walks through surface-code error correction in order: errors, syndromes, decoding,
spacetime and the threshold, then how the engine is checked, Google's hardware, real-time decoding,
the gross code, lattice surgery and a resource estimate. Its figures run the engine in the reader's
browser. The few results that need tools a browser lacks, or hours of native compute, are drawn
from the committed data and marked †.

**The technical report** ([web](https://qcompiler.jaspersands.com/report/report.html),
[PDF](report/report.pdf)) covers the engine, how it is checked, and what it measures. Every number
in it is filled in from the committed data (see [Technical report](#technical-report)).

## Install

```bash
pip install stabilizer-qec
```

One abi3 wheel for every CPython from 3.9 on, for Linux (x86_64 and aarch64), macOS (universal2)
and Windows; numpy is its one dependency. The names follow Stim, PyMatching and `ldpc`, so code
written for them mostly carries over:

```python
import stabilizer_qec as sq

circuit = sq.memory_circuit(distance=5, rounds=5, p=0.004)               # a d = 5 SD6 memory
sampler = circuit.compile_detector_sampler(seed=1)                       # same shots on any machine
dets, obs = sampler.sample(100_000, separate_observables=True, threads=0)
dem = circuit.detector_error_model(decompose_errors=True)
pred = sq.Matching(dem, enable_correlations=True).decode_batch(dets, threads=0)
print((pred != obs).any(axis=1).mean())                                  # logical error rate
```

Beyond the surface code, IBM's gross code decoded by BP+OSD, and a logical CNOT by lattice surgery:

```python
gross = sq.BivariateBicycleCode("gross")                                 # [[144, 12, 12]]
c = gross.memory_circuit(12, 0.003)
dets, obs = c.compile_detector_sampler(seed=2).sample(2_000, separate_observables=True)
pred = sq.BpOsd(c.detector_error_model()).decode_batch(dets, threads=0)

cnot = sq.surgery.cnot(5, merged=5, p=0.002)                             # a Circuit, Stim-readable
```

- **In sinter.** `pip install "stabilizer-qec[sinter]"`, then
  `custom_decoders=stabilizer_qec.sinter.sinter_decoders()` puts the decoders into
  [sinter](https://pypi.org/project/sinter/)'s threshold sweeps (`decoders=["sq_matching",
  "sq_belief_matching", "sq_bposd", ...]`), or the whole pipeline with `sq_sim_matching` and
  its kin. Circuits, models and decoders pickle, for `multiprocessing` and sinter's workers.
- **Tutorials.** Three notebooks in [`tutorials/`](tutorials/): [getting started](tutorials/01_getting_started.ipynb)
  (a circuit to a threshold plot), [decoders and real-time decoding](tutorials/02_decoders.ipynb),
  and [codes beyond the surface code](tutorials/03_beyond_the_surface_code.ipynb). Each is
  built from a plain Python file by `tools/notebooks.py`, and CI runs them all.
- **Benchmarks.** Against Stim and PyMatching on the same machine, one thread:
  [qcompiler.jaspersands.com/benchmarks](https://qcompiler.jaspersands.com/benchmarks/).
  `tools/bench.py` runs them; CI fails a push whose ratio to the reference doubles.
- **Docs.** A guide and the full reference at
  [qcompiler.jaspersands.com/api](https://qcompiler.jaspersands.com/api/) (`tools/api_docs.py`
  builds it from the docstrings, whose examples the tests run). The package is typed.
- **Promises.** A seed gives the same shots on any machine and any number of threads. Bad input
  raises `ValueError` or `TypeError`; an engine bug raises `RuntimeError`. From 1.0, semantic
  versioning: the public API (everything in `stabilizer_qec` without a leading underscore) keeps
  working through every 1.x release, and anything removed is deprecated first (see
  [CHANGELOG.md](CHANGELOG.md)). The Rust crate is versioned on its own, with the same promise
  for its documented API.
- **Tests.** `tests/` (pytest and Hypothesis) runs on Python 3.9 to 3.14 on Linux, macOS and
  Windows, against Stim, PyMatching, `ldpc` and `beliefmatching` where they install.
  `python tools/smoke.py`, run from outside the repository, checks an installed wheel end to end.
- **From source:** `pip install maturin && maturin build --out dist && pip install dist/*.whl`.

**From Rust**, the same engine is the crate [`stabilizer_qec`](https://crates.io/crates/stabilizer_qec)
(1.8, its API documented on [docs.rs](https://docs.rs/stabilizer_qec)):

```rust
use stabilizer_qec::{memory_circuit, Basis, DemOptions, Matching, Noise, SurfaceCode};

let circuit = memory_circuit(SurfaceCode::Rotated, 5, 5, Noise::Sd6 { p: 0.004 }, Basis::Z)?;
let samples = circuit.detector_sampler(7)?.sample(10_000, 0);
let dem = circuit.detector_error_model(&DemOptions::new().decompose_errors(true))?;
let predictions = Matching::with_correlations(&dem)?.decode_batch(&samples.detectors, 0)?;
```

**Releases.** On any `v*` tag, `.github/workflows/wheels.yml` builds the wheels and an sdist and
publishes them to PyPI by trusted publishing (workflow `wheels.yml`, environment `pypi`).

## Continuous integration

`.github/workflows/ci.yml` runs on every push and pull request:
- the Rust tests, among them a randomized run of 20,000 generated and mutated circuits and
  models through every parser, model builder, sampler and decoder, which may refuse an input
  but never panic;
- the WebAssembly engine, built fresh and driven in Node through the page's own wrappers
  (`tools/wasm-smoke.mjs`: Willow's raw measurements must hash to Google's detection events, then
  decode, and a stream must window-decode). The committed engine, the one the site serves, runs
  the same check.
- the site's tests, the palette's contrast check, and the README's tables against the data they
  come from (`tools/readme_tables.py --check`);
- the Python package built, installed and smoke-tested against Stim and PyMatching on Linux, macOS
  and Windows. On Linux it also runs the quick forms of the cross-check, BP and belief-matching
  against `ldpc` and `beliefmatching`, the gross code and BP+OSD against Bravyi et al., Stim and
  `ldpc`, lattice surgery against Stim and PyMatching, the gross code's logical measurement
  against Stim and `ldpc`, and the 1.8 decoders against `ldpc`, IBM's `relay_bp`, Chromobius and
  Tesseract (`tools/decoder_check.py`);
- the test suite (`tests/`) against each platform's wheel on Python 3.9, 3.10, 3.11, 3.12, 3.13
  and 3.14;
- `cargo fuzz` on the circuit and model parsers and everything behind them, a minute each
  (`fuzz/`; locally `cargo +nightly fuzz run circuit`).

## Key Features

- **Stabilizer engine**: binary symplectic tableau simulator, $O(N^2)$ Pauli propagation,
  based on the Gottesman–Knill theorem.
- **Codes**: rotated surface code ($d^2$ data qubits, $d^2-1$ ancillas) and XZZX surface code.
- **Noise models**: pure data noise, phenomenological noise, circuit-level noise (the engine's
  own model, and SD6, the standard one), $Z$-bias, located erasure, spatial bursts, and slow
  temporal drift.
- **Decoders**: disjoint-set Union-Find cluster peeling, exact minimum-weight perfect matching,
  a greedy nearest-neighbour baseline, and probability-weighted exact matching over any detector
  error model by sparse blossom: plain, at 0.89 to 0.99 times PyMatching's single-threaded time, and
  correlated (PyMatching 2.4's two-pass reweighting, agreeing with it shot for shot but for ties),
  at 0.75 to 0.94 times (on Apple silicon; on Linux x86_64 about 1.2, see
  [Speed](#speed-against-stim-and-pymatching)).
- **Throughput**: a bit-parallel sampler (64 shots a word, `REPEAT` without flattening) at 15–20×
  the reference sampler's speed, and a pool of workers that puts every core on the page to work.
- **A general circuit path**: circuits and detector error models in Stim's text formats, a
  detector error model built by walking any circuit backwards, a Pauli-frame sampler, and Stim's
  `01`/`b8` shot formats. Checked against Stim and PyMatching, edge for edge. Diagrams after
  Stim's: timelines, detector slices (the same as Stim's at every tick of its generated codes)
  and matching graphs, as text or SVG. Stim's generated memory circuits (`Circuit.generated`),
  character for character.
- **Stim's whole Clifford language**: H, S and CX natively; the other 46 one- and two-qubit
  gates as the shortest H/S/CX sequences whose tableaus equal Stim's, signs included
  (`tools/gen_gates.py` finds and checks them); measurements and resets in all three bases,
  inverted targets, `MPP`, `MXX`/`MYY`/`MZZ`, `SPP`, correlated and heralded errors,
  `PAULI_CHANNEL_2`, measurement feedback (`CX rec[-1] q`) and sweep bits, `MPAD`, and Stim's
  `approximate_disjoint_errors`. Checked against Stim on random circuits that use every gate:
  error models fault for fault, decomposed and not, raw measurements converted bit for bit,
  detection rates within noise.
- **Google's hardware data**: every Willow and Sycamore surface-code memory experiment (27.5
  million shots), rebuilt from raw measurements bit for bit and decoded with plain and correlated
  matching; Λ fitted the way Google fits it, for ours and for every decoder Google published.
- **Belief propagation and belief-matching**: BP whose posteriors equal the `ldpc` library's bit for
  bit, and belief-matching that equals the authors' own package shot for shot. Run on all of
  Google's Sycamore and Willow data, where it makes Sycamore's d = 5 beat d = 3 as Google's does.
- **Real-time decoding**: sliding and parallel window decoders, plain and correlated, checked
  against global decoding on Google's data, with latency measured against Willow's 1.1 µs cycle and
  a million-round stream that keeps up on 4 cores at d = 5.
- **IBM's gross code**: the bivariate bicycle codes [[144, 12, 12]] and [[72, 12, 6]] with Bravyi et
  al.'s depth-8 syndrome cycle, decoded by BP+OSD whose corrections equal `ldpc`'s but for ties. On
  it, every automorphism's logical action computed exactly, and the gauging measurement of a logical
  operator built from Cross et al.'s definition, with its distance proven by integer programming.
- **More decoders**: BP+LSD equal to `ldpc`'s, Relay-BP identical to IBM's `relay_bp`, colour-code
  matching by Chromobius's construction, and Tesseract's search for the most likely error,
  identical to `tesseract_decoder`'s, each held to its authors' package shot for shot (see
  [More decoders](#more-decoders)).
- **More codes**: every bivariate bicycle code of Bravyi et al.'s Table 3 ([[72, 12, 6]] to
  [[288, 12, 18]]) or any other from its polynomials, and any CSS code from its checks:
  hypergraph products of classical codes and the 6.6.6 colour code, each with a memory
  experiment whose error model equals Stim's.
- **Lattice surgery**: surface-code patches merged and split as one circuit, its error model equal
  to Stim's, and programs compiled from such steps: a logical CNOT, merges in a row, and lines of
  patches merged at once.
- **A resource estimate**: magic-state factories, Litinski's floor plans, reaction time from this
  decoder's measured latency, and storage options, priced on noise and decoding measured here and
  checked against Gidney's RSA-2048 and Lee et al.'s FeMoco estimates.
- **Web explainer**: `index.html` plus `css/` and `js/`. No build step, no dependencies. The lattice
  at the top of the page runs the engine live, the interactive figures are driven by it, the
  threshold table is plotted as it is measured, and the bench streams its estimate.
- **Python package** (`stabilizer-qec` on PyPI, one abi3 wheel, typed): circuits, error models,
  sampling, plain, correlated and belief matching, BP+OSD, BP+LSD, Relay-BP, colour-code
  matching, the search decoder, window decoding and streams, the gross code and lattice surgery,
  from Python (see [Install](#install)).

## Speed against Stim and PyMatching

Each benchmark times the same job here and in the reference, on the same machine and one
thread, as a ratio: this package's time over Stim's (or PyMatching's, or the named decoder
package's), below 1 faster. These
are the latest runs recorded for each kind of machine (`tools/bench.py`; Linux on GitHub's
runners, `.github/workflows/bench.yml`); every push is held to them, and the
[benchmarks page](https://qcompiler.jaspersands.com/benchmarks/) plots every release's.

| benchmark | Apple silicon | Linux x86_64 | Linux arm64 |
|---|---|---|---|
| Error model of a d = 11 rotated memory, 11 rounds (decomposed) | 0.61 | 0.42 | 0.53 |
| Its graph-like distance (shortest_graphlike_error on the model) | 0.11 | 0.08 | 0.17 |
| Sampling its detection events, 20,000 shots | 0.92 | 0.69 | 0.57 |
| Measurements to detection events, 20,000 shots | 0.24 | 0.15 | 0.12 |
| Matching at p = 0.1%, 20,000 shots (against PyMatching) | 0.99 | 1.16 | 1.10 |
| Correlated matching, 20,000 shots (against PyMatching's) | 0.99 | 1.09 | 0.88 |
| BP+LSD on the gross code, 6 cycles at p = 0.3%, 200 shots (against ldpc) | 0.60 | — | — |
| Relay-BP on the same shots (against IBM's relay_bp) | 0.87 | — | — |
| Colour-code matching, d = 7 colour code at p = 0.2%, 20,000 shots (against Chromobius) | 0.73 | — | — |
| The search decoder, d = 5 colour code at p = 0.2%, 1,000 shots (against Tesseract) | 0.99 | — | — |
| *recorded with* | 1.8.0 | 1.7.0 | 1.7.0 |

## A note on quoted figures

Unless stated otherwise, the logical error rates below were measured at bias `η = 0.5` (equal
parts X, Y and Z, which is the setting every panel on the site starts from) over `T = d` rounds.
The bias matters more than it might seem. At `d = 7, p = 0.3%` circuit-level, the rotated code
reports **1.07% at η = 1 and 2.37% at η = 100**, while XZZX at the same two settings reports
**1.08% and 0.25%**. Same p, and the ranking between the codes reverses. Shot counts are given
where a number is close enough to the noise floor for it to matter.

## Checked against Stim and PyMatching

Stim and PyMatching are the tools the field uses to simulate and decode surface codes, and the
first question anyone asks of a new simulator is whether it agrees with them. This engine began
checked only against itself. It now has a general circuit path, and on that path it agrees with
both tools on every check below, down to floating-point rounding.

**The general path.**

- `src/circuit.rs` reads and writes the part of Stim's circuit language the engine speaks: resets,
  H, CX, CZ, measurements, the depolarizing and Pauli channels, detectors, observables,
  coordinates, and REPEAT blocks.
- `src/dem.rs` builds a detector error model from any such circuit by walking it backwards once.
  For every qubit it carries the set of detectors an X or a Z error there would flip. That is how
  Stim's error analyzer works, and it is fast enough to run in the browser at d = 7.
- `src/dem_decoder.rs` is exact minimum-weight perfect matching over any error model. Edges are
  weighted ln((1 − p)/p), and the decoder returns the logical observables it predicts flipped.
- `src/frame_sampler.rs` samples a circuit with a Pauli frame. It never looks at the error model,
  so the model and the sampler can disagree.
- `src/memory.rs` writes the rotated and XZZX memory experiments as circuits, under two noise
  models:
  - the engine's own circuit-level model, transcribed gate for gate;
  - SD6, the standard model published thresholds assume (Gidney, Newman, Fowler and Broughton,
    2021): a two-qubit depolarizing channel after every CNOT, single-qubit depolarizing after every
    single-qubit gate and on every idle qubit, and flips on every reset and measurement.

The old per-code paths are untouched, and every other number in this file still comes from them.

**1. The error models are identical.** The comparison covers 15 circuits: Stim's own generated
rotated memory experiment, and ours for both codes under both noise models, each at d = 3, 5 and 7
with p = 0.3%. For every one, this engine's model and Stim's list the same faults with the same
probabilities. This engine parses Stim's circuits, and Stim parses ours.

**1b. So are the matching graphs.** A decoder does not run on that list directly. A fault that trips
more than two detectors must first be split into graph-like pieces, and how it is split sets the
graph's edge weights. The decomposed models agree too: every fault is split the same way, and every
edge has the same probability.

| circuit | d | detectors | mechanisms, ours / Stim's | largest Δp/p | graph edges | graph |
|---|---|---|---|---|---|---|
| Stim's `rotated_memory_z` | 3 | 24 | 219 / 219 | 6.4e-16 | 78 | identical |
| Stim's `rotated_memory_z` | 5 | 120 | 1,677 / 1,677 | 6.4e-16 | 502 | identical |
| Stim's `rotated_memory_z` | 7 | 336 | 5,471 / 5,471 | 6.4e-16 | 1,558 | identical |
| rotated, engine model | 3 | 36 | 223 / 223 | 0.0e+00 | 103 | identical |
| rotated, engine model | 5 | 156 | 1,261 / 1,261 | 0.0e+00 | 581 | identical |
| rotated, engine model | 7 | 408 | 3,739 / 3,739 | 0.0e+00 | 1,723 | identical |
| rotated, SD6 | 3 | 24 | 219 / 219 | 1.1e-15 | 78 | identical |
| rotated, SD6 | 5 | 120 | 1,677 / 1,677 | 1.1e-15 | 502 | identical |
| rotated, SD6 | 7 | 336 | 5,471 / 5,471 | 1.1e-15 | 1,558 | identical |
| XZZX, engine model | 3 | 36 | 223 / 223 | 0.0e+00 | 103 | identical |
| XZZX, engine model | 5 | 156 | 1,261 / 1,261 | 0.0e+00 | 581 | identical |
| XZZX, engine model | 7 | 408 | 3,739 / 3,739 | 0.0e+00 | 1,723 | identical |
| XZZX, SD6 | 3 | 24 | 219 / 219 | 1.1e-15 | 78 | identical |
| XZZX, SD6 | 5 | 120 | 1,677 / 1,677 | 1.1e-15 | 502 | identical |
| XZZX, SD6 | 7 | 336 | 5,471 / 5,471 | 1.1e-15 | 1,558 | identical |

Those two last columns did not match on the first attempt. What caught it was not a comparison with
Stim at all; see *What the checks caught* below.

**2. The decoders agree shot for shot.** At each point, Stim sampled 100,000 shots, and PyMatching and
this engine's decoder (now the sparse matcher described in the next section) decoded the same
detection events. Two exact matchers can disagree only where two corrections tie in weight. There
were 13 disagreements in 600,000 shots, and every one is a tie: the two
matchings' weights differ by less than the discretisation noise measured on shots where the decoders
agree (at most 1e-05).

**3. The samplers agree.** The same circuits were also sampled by this engine's frame sampler and
decoded by its own decoder, and compared with Stim's sampler decoded by PyMatching:

- Per-detector firing rates agree, with χ² z-scores between −2.4 and +0.1 over 24 to 336 detectors.
- The logical error rates lie within each other's intervals.
- At 400,000 shots each, both decoded by PyMatching, the two samplers' rates differ by 0.4σ, 0.4σ
  and 0.1σ at (d, p) = (3, 0.3%), (3, 0.6%) and (5, 0.6%).

| d | p | PyMatching | ours, on Stim's graph | ours, on our graph | disagreements (not ties) | our sampler + decoder | χ² z | PyMatching | ours |
|---|---|---|---|---|---|---|---|---|---|
| 3 | 0.3% | 2.289% | 2.289% | 2.289% | 0 (0) | 2.292% | -0.10 | 0.2 µs | 0.2 µs |
| 3 | 0.6% | 7.603% | 7.603% | 7.603% | 0 (0) | 7.624% | -1.38 | 0.5 µs | 0.4 µs |
| 5 | 0.3% | 1.647% | 1.649% | 1.649% | 2 (0) | 1.631% | -2.38 | 1.9 µs | 1.7 µs |
| 5 | 0.6% | 9.432% | 9.432% | 9.432% | 4 (0) | 9.493% | -0.90 | 4.0 µs | 3.6 µs |
| 7 | 0.3% | 1.124% | 1.124% | 1.124% | 0 (0) | 1.073% | +0.06 | 6.1 µs | 5.5 µs |
| 7 | 0.6% | 10.653% | 10.652% | 10.652% | 7 (0) | 10.686% | -2.35 | 14.6 µs | 13.1 µs |

The last two columns are native decode time per shot.

**4. Speed.** Single-threaded, this engine takes 0.89 to 0.99 times PyMatching's time per
shot, and its correlated matching 0.75 to 0.94 times PyMatching's correlated mode. At d = 7, p = 0.6%
it takes 13.1 µs a shot, native, against PyMatching's 14.6 µs. The first version of this check used
the dense matcher, which was 18 to 136 times slower and fell further behind as the patch grew. The
sparse matcher closed most of that gap, and later passes (below) the rest. It also decodes shots
in parallel, one workspace per thread: across the recording machine's 10 cores, d = 7, p = 0.6% runs
at 2.1 µs a shot.

**5. The old path and the new one describe the same circuit.** `src/equivalence.rs` rebuilds the old
path's error model from its own code:

1. Push every fault the old per-code simulator enumerates through its Pauli frame.
2. Name the detectors each fault trips by plaquette and round.
3. Merge the faults into an error model.

The result is identical to the new path's model of the same memory experiment under the engine's
noise: the same 223, 1,261 and 3,739 mechanisms at d = 3, 5 and 7, for both codes and both bases.
So the engine's circuit-level results and the general path describe the same circuit.

They are not decoded the same way. The old decoder matches one Pauli type at a time, with unit edge
weights. The new one weights every edge by the probability of the faults behind it. The table below
uses the same circuit and the same noise, with 20,000 shots each; the old decoder's column counts the
failures the memory-Z experiment would see.

| d | p | old decoder (unit weights) | new decoder (probability weights) |
|---|---|---|---|
| 3 | 0.2% | 0.265% | 0.310% |
| 3 | 0.3% | 0.760% | 0.495% |
| 3 | 0.4% | 1.230% | 0.910% |
| 5 | 0.2% | 0.195% | 0.100% |
| 5 | 0.3% | 0.540% | 0.385% |
| 5 | 0.4% | 1.395% | 0.865% |
| 7 | 0.2% | 0.100% | 0.030% |
| 7 | 0.3% | 0.410% | 0.170% |
| 7 | 0.4% | 1.200% | 0.665% |

Weighting by probability roughly halves the logical error rate at d = 5 and 7. At d = 3 and the
lowest p the two are within noise. This is a finding about the decoder rather than the physics, and
it suggests the circuit-level thresholds quoted below are partly a property of unit-weight matching.
How much is not measured here: those thresholds were not rerun with the weighted decoder.

**6. Every single fault is corrected.** The exhaustive check from section 7 below now runs on the new
path too: every mechanism of every model, decoded alone, predicts its own logical flip. That covers
the rotated and XZZX codes, both bases and both noise models:

- 446, 2,522 and 7,478 mechanisms at d = 3, 5 and 7 under the engine's model;
- 568, 3,888 and 12,152 under SD6;
- no failures.

**What the checks caught.** The first decomposition kept any fault with at most two detectors whole,
and split wider ones into their X and Z halves. It passed every unit test and the error-model
comparison, since that comparison is about faults, not pieces. Check 6 failed. Two single faults of
the d = 3 memory-X circuit under SD6 decoded into logical errors, and one under XZZX did too.

- **The circuit was sound.** Stim put its graph-like distance at 3, so the faults were correctable,
  and PyMatching corrected them on Stim's decomposition.
- **The graph was the difference.** A Y error on a boundary qubit trips one X-type check and one
  Z-type check. Kept whole, it becomes an edge joining the two matching graphs, and the matcher
  routed a correction through that bridge and across the logical operator.
- **A first fix fell short.** Splitting every fault into its X and Z halves removed the bridges but
  still disagreed with Stim on 66 of 221 symptoms. Stim splits a fault by what the rest of the same
  noise channel can express, not by the fault's own halves, and it keeps faults that share a
  symptom but split differently apart.

The decomposition now reproduces Stim's error analyzer. Each composite channel's combinations are
split using that channel's own single-detector and irreducible two-detector combinations, and
anything left over goes through Stim's global pass. Check 1b is the result. None of this reached
the site: the bug lived only in the new code, and the exhaustive single-fault check caught it on
its first run.

**Reproduce.**

```bash
python3 -m venv .venv && .venv/bin/pip install stim pymatching numpy maturin
VIRTUAL_ENV=$PWD/.venv CARGO_TARGET_DIR=target-py .venv/bin/maturin develop --profile python
.venv/bin/python tools/xcheck.py
cargo test --release --no-default-features equivalence -- --include-ignored --nocapture
cargo test --release --no-default-features every_single_fault -- --include-ignored
```

The recorded run is `data/xcheck/report.txt`. Figure 8 on the site repeats the error-model
comparison live, in the reader's browser.

## An exact sparse matcher

The first general-path decoder was exact and simple:

1. Run Dijkstra from every defect.
2. Build the complete graph of defects.
3. Run Edmonds' blossom algorithm on it, which costs O(n³).

It stopped at 256 defects and, near that limit, took seconds per shot. Google's longest memory
experiments have about 12,000 detectors and on the order of a thousand defects a shot.
`src/sparse/` replaces it with **sparse blossom**, the algorithm of Higgott and Gidney
(arXiv:2303.15933) that PyMatching 2 is built on.

How it works:

- **Regions grow on the detector graph itself.** Every defect grows a region, all at the same rate,
  and a region's radius is its dual variable.
- **Collisions come from the growth.** When two regions touch, Edmonds' alternating trees and
  blossoms take over, but over regions rather than vertices. A collision is found by the growth
  itself, not by a precomputed distance, so nothing is explored beyond where the regions reach.
- **Even weights keep the events on integer times.** Edge weights are discretised to even
  integers, so two regions growing toward each other always meet at an integer time.
- **Only the next event is queued.** A queue of look-at reminders holds only the next event per
  node or region.

The dense matcher stays as `decode_dense`, and it is what the sparse one is checked against first.

**Verification, in five layers.** The dense and sparse matchers minimise exactly the same integer
weights, so their optimal weights must be *equal*, not merely close.

1. **Brute force.** On 3,000 random small graphs, with and without boundaries, including odd
   components that cannot be matched: every weight is optimal, and every prediction is one of the
   optimal ones.
2. **Against the dense matcher**, with equal integer weight on every shot either way:
   - 20,000 random graphs of up to 43 nodes, with the dual's feasibility checked after every event:
     no radius negative, no two regions overlapping across an edge, every node's recorded blossom
     ancestry correct;
   - 4,320 sampled surface-code shots (rotated and XZZX, both bases, both noise models, d = 3, 5, 7,
     p up to 1.2%). Two of them break a tie differently.
3. **Every single fault is corrected**, at d = 3, 5 and 7, in all 24 circuits.
4. **Against PyMatching on identical shots:** 13 disagreements in 600,000 shots, every one a tie
   (check 2 above).
5. **Beyond the dense limit:** d = 7 run for 60 rounds, with more than 256 defects in a shot, decodes
   without error.

It passed every layer on its first complete run. The only change the tests forced was to a test's
own floor on how many unmatchable cases the random generator must produce.

**Speed.** Native (arm64), single-threaded, rotated code under SD6 with T = d, per shot:

| d | p | dense | sparse | PyMatching |
|---|---|---|---|---|
| 3 | 0.3% | 2.0 µs | 0.2 µs | 0.2 µs |
| 3 | 0.6% | 4.0 µs | 0.4 µs | 0.5 µs |
| 5 | 0.3% | 47.8 µs | 1.6 µs | 1.9 µs |
| 5 | 0.6% | 169 µs | 3.7 µs | 4.0 µs |
| 7 | 0.3% | 530 µs | 5.2 µs | 6.1 µs |
| 7 | 0.6% | 2.13 ms | 13.4 µs | 14.6 µs |
| 9 | 0.3% | 3.72 ms | 13.0 µs | — |
| 9 | 0.6% | 17.5 ms | 35.4 µs | — |

The dense and sparse columns come from `cargo run --release --no-default-features --target
aarch64-apple-darwin --example matcher_profile -- timing data/matcher/native.json`, and PyMatching's
column from the cross-check's recorded run. The recording machine's default Rust toolchain is x86_64,
which macOS runs under Rosetta at about half the speed. An earlier version of this table was timed
that way and called native; it was not, and the target is now explicit.

Two later passes took the sparse matcher from 2.5–2.9 times PyMatching's time to 1.01–1.14 plain and
0.91–1.01 correlated, and a third (1.7) to 0.89–0.99 and 0.75–0.94, mostly by passing over reminders
for nodes with nothing ahead without scanning them. None changed one prediction: every change was checked against a fingerprint of
every prediction and weight (`tools/matcher_bench.py --check`: 100,000 shots at each of six points up
to d = 7, and 20,000 at d = 9 and 11, plain and correlated).
1. **No quadratic scans, no allocation per event.** Dissolving a tree and forming a blossom tested
   tree nodes for membership by scanning lists; they now mark nodes. The list of nodes under a region,
   the path to a root and each shot's regions are kept and reused instead of allocated afresh.
2. **Fewer mispredicted branches.** A native profile put two-thirds of the time in one function, the
   one that finds a node's next event by scanning its edges. PyMatching's flooder treats an empty
   neighbour as a radius of zero that does not grow. Taken further here, every edge is the same loads
   and arithmetic: an empty neighbour reads a region that is nobody's, the boundary reads a node past
   the graph's last, and the one branch left is the rarely taken "earlier than the best so far". A
   node of a shrinking region returns at once. Together: 29% off a shot at d = 11.
3. **A radix-heap queue.** Event times never go backwards, so the event queue, and the shortest-path
   search that traces a matching's paths, are radix heaps. They pop in exactly the order the binary
   heap did, which a test pins, so each decode takes the same steps. 9% more.

Tried and dropped, each by the fingerprints or the stopwatch:
- a line-by-line tightening of the same loop, which measured no faster;
- a fast path for shots with no defects, which saved nothing measurable;
- skipping the node look-ups of regions whose growth only slows. This changed one correlated shot's
  optimal weight in 100,000, a missed event, which the fingerprints caught.

The site uses the same matcher in WebAssembly. A whole SD6 sweep window, one shot at every point
across d = 3, 5, 7 and 9, now costs about 0.7 ms instead of more than 100 ms. So the SD6 sweep went
from 800 shots a point to 40,000, and Figure 8's live rates now run 50,000 to 100,000 shots.

## Correlated matching

One fault in a circuit can set off detectors in both halves of the detector graph. A Y error is an X
error and a Z error at once, so it lights the X-type and the Z-type checks together. The error
model knows this and writes such a fault as pieces, `error(p) D0 D1 ^ D7 D8`, one per half. Plain
matching sees only the pieces, as independent edges, and forgets that they come together.
Correlated matching remembers. `src/sparse/correlated.rs` follows PyMatching 2.4's
`enable_correlations=True`, read from its source, so that PyMatching can serve as the oracle:

1. **Pass one.** The sparse matcher, unchanged.
2. **Trace.** A shortest path on the detector graph for each matched pair (`src/sparse/paths.rs`),
   all XORed into one edge set, so an edge two paths share cancels.
3. **Reweight.** From the model's decompositions, each edge c has rules. For every edge a it shares
   an error with, the rule is `p_a = min(0.5, joint(c, a) / marginal(c))`: the probability that a
   fired given that c did. Each edge the set uses lowers every edge its rules name to the smaller
   of its own weight and ln((1 − p_a)/p_a).
4. **Pass two.** The sparse matcher again, on the lowered weights. Its observables are the
   prediction. The weights are then restored, so decoding threads never share mutable state.

**Verification, in five layers.**

1. **Rule tables by hand.** Two- and three-piece decompositions, a piece shared by several errors,
   the cap at one half, errors of probability zero, and PyMatching's quirk of counting a piece
   repeated within one error twice more toward its marginal.
2. **The traced edges.** On 3,600 surface-code shots (rotated and XZZX, d = 3, 5, 7, SD6 at 0.3% and
   0.6%), every traced edge set has exactly the shot's defects as its syndrome and weighs exactly the
   optimum: each is a minimum-weight correction.
3. **Pass two is exact.** On 1,440 shots, the second pass's integer weight equals the dense
   matcher's on a model carrying exactly the reweighted weights.
4. **Against PyMatching's correlated mode on identical shots**, 1.2 million of them (check 5 of the
   cross-check): 696 disagreements, and not one bug among them. For every disagreement,
   PyMatching's own first-pass edges (`decode_to_edges_array`) go into our second pass. In 682
   cases that reproduces PyMatching's answer: the two first passes traced different, equally short
   paths. The other 14 tie PyMatching's second-pass weight to within the discretisation noise
   (below 1e-5).
5. **It helps.** Pooled over all twelve points, correlated matching fails 53,030 times against plain
   matching's 65,316, 19% fewer. At d = 7 it nearly halves the failure rate at p = 0.3%.

| code | d | p | PyMatching correlated | ours correlated | ours plain | disagreements | PyMatching µs | ours µs |
|---|---|---|---|---|---|---|---|---|
| rotated | 3 | 0.3% | 2.061% | 2.061% | 2.332% | 0 | 0.6 | 0.5 |
| rotated | 3 | 0.6% | 6.951% | 6.951% | 7.708% | 0 | 1.3 | 1.1 |
| rotated | 5 | 0.3% | 1.133% | 1.131% | 1.645% | 8 | 5.6 | 4.2 |
| rotated | 5 | 0.6% | 7.735% | 7.738% | 9.335% | 31 | 11.0 | 9.0 |
| rotated | 7 | 0.3% | 0.555% | 0.558% | 1.037% | 29 | 16.1 | 15.2 |
| rotated | 7 | 0.6% | 7.932% | 7.906% | 10.590% | 274 | 39.3 | 33.6 |
| XZZX | 3 | 0.3% | 2.020% | 2.020% | 2.293% | 0 | 0.6 | 0.5 |
| XZZX | 3 | 0.6% | 7.098% | 7.098% | 7.665% | 0 | 1.3 | 1.1 |
| XZZX | 5 | 0.3% | 1.210% | 1.210% | 1.639% | 8 | 5.0 | 4.2 |
| XZZX | 5 | 0.6% | 7.739% | 7.745% | 9.323% | 22 | 11.0 | 9.2 |
| XZZX | 7 | 0.3% | 0.581% | 0.579% | 1.103% | 32 | 15.1 | 13.9 |
| XZZX | 7 | 0.6% | 8.027% | 8.033% | 10.646% | 292 | 37.3 | 32.8 |

100,000 shots per row, sampled by Stim, SD6 noise, T = d. The timing columns are single-threaded,
correlated mode for both: ours takes 0.75 to 0.94 times as long as PyMatching's. Natively (the
table above), the second pass more than doubles the cost of a shot: at d = 9 and p = 0.6%, 35 µs plain
and 86 µs correlated.

It passed every layer on its first complete run. One thing did go wrong along the way, outside
the algorithm. The Python module built with Rust 1.90's default release strip would not load on
this Mac: the arm64 library's symbol string table came out unaligned, and the loader refused it.
The module now builds with its own profile, `python` in `Cargo.toml`: release, unstripped.

## Google's hardware data

Everything above runs on simulated noise. Google publishes what its chips actually recorded, and
this engine now decodes all of it:

- **Willow**, 2024 ("Quantum error correction below the surface code threshold", Zenodo 13273331):
  surface-code memories at d = 3, 5 and 7 (nine, four and one patch), X and Z bases, 1 to 250
  rounds. That is 420 experiments of 50,000 shots, 21.0 million shots.
- **Sycamore**, 2022 ("Suppressing quantum errors by scaling a surface code logical qubit", Zenodo
  6804040): d = 3 (four patches) and d = 5, both bases, 1 to 25 rounds. That is 130 experiments,
  6.5 million shots.

Both datasets are CC BY 4.0, by Google Quantum AI. They stay in their zips in `data/google/`,
gitignored, and are fetched by `data/google/fetch.sh`. What is committed is the results, in
`data/google-results/`, and a 0.95 MB extract for the site, in `data/willow-extract/`.

**Reading the chips' output.**

- **Parser.** Google's circuits need two things the parser lacked:
  - `CX sweep[k] q`, an X applied when the shot's sweep bit k is set, which is how each shot
    prepares its data qubits in a different pattern;
  - the Pauli gates `X`, `Y`, `Z` and `I`.

  Neither changes which detectors a fault sets off, so the error-model builder and the sampler
  ignore both.
- **From raw measurements to detection events** (`src/m2d.rs`). A detector compares a parity of
  measurements with what a noiseless run gives. One noiseless run of the ideal circuit on the
  stabilizer tableau is the reference. Each sweep bit's effect on every detector is fixed, since
  the circuit is Clifford, so one more run per bit finds it. Seven more references with other
  random outcomes must agree, or a detector is refused as nondeterministic.

  This is the tableau reading the circuit, independently of the backward walk the error models
  come from. On synthetic circuits with random sweep bits it matches Stim's own `m2d` converter bit
  for bit.

**Checked against Google's own files** (`tools/google.py check`, all 550 experiments, 27.5 million
shots):

- Every circuit round-trips through the parser and printer, and Stim reads the result back as the
  same circuit.
- The detection events and observable flips rebuilt from the raw `measurements.b8` and sweep bits
  equal Google's `detection_events.b8` and `obs_flips_actual` **bit for bit, on every shot of every
  experiment**.
- This engine's model of every noisy circuit is **identical to Stim's**, mechanism by mechanism and
  edge by edge.
- Against the models Google decoded with:
  - Sycamore's `circuit_detector_error_model.dem` matches ours to 2.5 × 10⁻⁶.
  - Willow's SI1000 prior has exactly our edges, but its probabilities are reweighted: a median
    9.5% from ours, and up to 49%.

  So on Willow our own model is a third prior, not a copy of theirs.

**The run** (`tools/google.py run`) decodes every experiment with every prior, each with plain and
with correlated matching:

- Willow: Google's SI1000 prior, Google's RL-optimised prior, and our model of the noisy circuit.
- Sycamore: the circuit's model; ours; and the data-fitted `pij` models, cross-fitted as Google
  used them, each shot decoded by the model fitted on the other half.

Google's own predictions are scored against the same truth. On ten cores the whole of it took 53
minutes: 48 of decoding and the rest reading the zips.
- The heaviest experiment, d = 7 at 250 rounds, has 12,000 detectors and 936 defects a shot on
  average.
- It decodes at 0.13 ms a shot plain and 0.28 ms correlated.

**One fit for everything** (`js/lambda-fit.js`, run by `node tools/lambda.mjs`; the page runs the
same module):

- **Per patch and basis:** the logical fidelity 1 − 2 P_L is fitted as A (1 − 2ε)^r by weighted
  least squares on ln F. Points within 3σ of zero are left out. The fit starts from round 10 for
  Willow (its first round is not in steady state) and round 3 for Sycamore.
- **Per distance:** ε is the mean over patches and bases.
- **Λ:** from a line through ln ε against d.
- **Intervals:** a parametric bootstrap, each experiment's failures redrawn and everything refitted.
  Which points each fit uses, and how much it weighs them, stay as the observed data set them.
  Letting a redrawn fidelity choose its own weight biased the draws about 0.2% low, which the
  average over eighteen patches turned into visibly lopsided intervals. That was caught and fixed
  before anything was quoted.

**Validation.** Fitted this way, Google's own tensor-network predictions for Sycamore give
**ε₃ = 3.028% and ε₅ = 2.914%, the published numbers exactly** (the paper's are 3.028% ± 0.023%
and 2.914% ± 0.016%), starting from round 2 or 3. So the comparison below is made the way Google
made theirs.

**Willow** (ε per cycle, 95% bootstrap intervals in `data/google-results/lambda.txt`):

| decoder | prior | ε, d = 3 | ε, d = 5 | ε, d = 7 | Λ |
|---|---|---|---|---|---|
| ours, plain | SI1000 | 1.032% | 0.684% | 0.435% | 1.54 |
| ours, correlated | SI1000 | 0.888% | 0.449% | 0.234% | **1.95** [1.94, 1.95] |
| ours, correlated | RL-optimised | 0.808% | 0.425% | 0.222% | 1.91 |
| ours, correlated | our model | 0.884% | 0.468% | 0.241% | 1.91 |
| Google, correlated matching | SI1000 | 0.836% | 0.443% | 0.229% | 1.91 |
| Google, correlated matching | RL-optimised | 0.759% | 0.404% | 0.210% | 1.90 |
| Google, Harmony (51 matchers) | RL-optimised | 0.729% | 0.387% | 0.206% | 1.88 |
| Google, Libra | RL-optimised | 0.712% | 0.349% | 0.171% | 2.04 |
| Google, neural network (published) | — | — | — | 0.143% | 2.14 |

- **Correlated matching is most of the story.** With the same prior, it takes Λ from 1.54 to 1.95
  and nearly halves ε at d = 7.
- **Ours sits within a few per cent of Google's own correlated matcher on the same prior:** 0.234%
  against 0.229% at d = 7. It is further behind at d = 3, 0.888% against 0.836%. Google describes
  theirs as a variant of the two-step reweighting, and small patches evidently reward the variant.
  That relative weakness at d = 3 is why our Λ comes out higher than theirs (1.95 against 1.91):
  Λ rewards improving with size, not being good.
- **Our own model of the noisy circuit**, built here from Google's circuit with no fitting to the
  data, does about as well as their SI1000 prior (0.241% at d = 7).
- **What the published 2.14 is, and isn't.** It is Google's neural-network decoder. Its predictions
  are not in the dataset, so it cannot be refitted here. Libra, the best decoder whose predictions
  are published, reaches 2.04 on this fit.

**Sycamore:**

| decoder | prior | ε, d = 3 | ε, d = 5 | Λ |
|---|---|---|---|---|
| ours, plain | circuit | 4.012% | 4.354% | 0.92 |
| ours, correlated | circuit | 3.507% | 3.558% | 0.99 |
| ours, correlated | pij, cross-fitted | 3.424% | 3.466% | 0.99 |
| Google, PyMatching | circuit | 4.012% | 4.362% | 0.92 |
| Google, correlated matching | circuit | 3.497% | 3.597% | 0.97 |
| Google, belief matching | pij | 3.118% | 3.056% | 1.02 |
| Google, tensor network | pij | 3.028% | 2.914% | 1.04 |

- **Our plain matcher reproduces Google's recorded PyMatching:** 4.012% at d = 3, identical.
- Where the two disagree, on 8,448 shots over the whole dataset, our matching weighs exactly the
  optimum PyMatching 2.4 finds on every one: ties, broken differently by the older PyMatching
  Google used.
- Our correlated matcher lands beside Google's on the same prior.
- Sycamore's Λ ≈ 1 was the point of that paper: it was the first time a larger surface code beat a
  smaller one at all, and only with the best decoders. With matching, d = 5 does not yet beat d = 3.

**On the site.** Section 11 shows two figures.

- **Figure 9 fits** every decoder from these recorded counts, in the reader's tab, with the same
  module.
- **Figure 10 decodes live, from the raw readouts.** The data is 2,000 raw Willow shots each at
  d = 3, 5 and 7 (Z basis, 30 rounds, the first shots of each, unmodified). The worker:
  1. rebuilds their detection events and checks the SHA-256 against Google's;
  2. builds two models, Google's SI1000 prior and ours from the noisy circuit;
  3. decodes with both matchers, next to Google's five decoders on the same shots.

  At d = 7 our correlated matcher on Google's prior fails 123 times in 2,000 shots, against
  Google's correlated matcher's 122, and agrees with it on 97% of shots.

## Belief-matching

Matching weighs every graph edge by its prior alone. **Belief-matching** (Higgott, Bohdanowicz,
Kubica, Flammia and Campbell, PRX 13, 031007, 2023) lets the whole error model speak first:
1. **Belief propagation** runs on the hypergraph: every fault a variable, every detector a check,
   nothing split into pieces.
2. **Its posteriors re-weight the graph.** Each edge gets −ln p, where p sums the posteriors of the
   faults it is part of.
3. **The sparse matcher decodes** on those weights. Where BP converges on its own, its correction is
   the answer.

On Sycamore's data it was one of the two decoders that made d = 5 beat d = 3.

**Built to be checked exactly** (`src/bp.rs`, `src/belief.rs`, `tools/bp_check.py`).
- **BP** is flooding product-sum or scaled min-sum. It is written to reproduce the arithmetic of
  `ldpc` (Roffe et al.), the library the reference implementation runs on, step for step: prefix
  and suffix products along each check, sums down each variable, the same stopping rule.
- **Belief-matching** follows `beliefmatching`, the authors' own package, in every choice:
  - faults keyed by their detectors;
  - the first decomposition of a shared symptom wins, and the last observables;
  - edge probabilities summed, clipped to [1e-14, 1 − 1e-14], and weighted −ln p.

Checked against both packages, installed alongside:
1. **Posteriors equal `ldpc`'s bit for bit.** On random parity-check matrices and on a d = 5 surface
   code's hypergraph, by product-sum and by min-sum (scaled 1 and 0.625), after 1, 3 and 20
   iterations, every posterior log-likelihood ratio equals `ldpc`'s: the largest difference is 0.0.
   Hard decisions, convergence and iteration counts agree too.
2. **Belief-matching equals `beliefmatching`** on every shot of SD6 memories at d = 3, 5, 7: the same
   convergence and the same prediction.
3. **Exact small cases.** On two tree-shaped Tanner graphs product-sum BP is exact, and its
   posteriors equal brute-force marginals to 1e-12. Every single fault of rotated and XZZX circuits
   is corrected. Matching on the model's own weights through the new per-shot weight path is plain
   matching, weight for weight.

Single-threaded it runs at about half the reference's time per shot: 0.9 ms against 1.7 ms at
d = 5 (the reference rebuilds a PyMatching graph for every shot).

**Sycamore, against Google's own belief-matching.** Every experiment, all 6.5 million shots, is
decoded with the data-fitted pij priors, cross-fitted as Google used them, and with the circuit's own
model:

| decoder | prior | ε, d = 3 | ε, d = 5 | Λ [95%] |
|---|---|---|---|---|
| ours, belief-matching | pij, cross-fitted | 3.154% | 3.104% | **1.016** [1.01, 1.02] |
| ours, belief-matching | circuit | 3.249% | 3.190% | 1.019 [1.01, 1.03] |
| Google, belief matching | pij | 3.118% | 3.056% | **1.020** [1.01, 1.03] |
| Google, tensor network | pij | 3.028% | 2.914% | 1.039 [1.03, 1.05] |
| Google, correlated matching | circuit | 3.497% | 3.597% | 0.972 [0.96, 0.98] |
| Google, PyMatching | circuit | 4.012% | 4.362% | 0.920 [0.91, 0.93] |

- **d = 5 beats d = 3**, as it did for Google: Λ = 1.016 [1.01, 1.02] against their 1.020 [1.01,
  1.03].
- **Close, but not the same.** Ours agrees with Google's recorded predictions on 97.2% of shots and
  fails 0.7% more often overall. That is more than chance: ε₃ is 3.154% against 3.118%, outside
  each other's intervals. Google's BP settings are not in the dataset. On one experiment (d = 3,
  25 rounds, 20,000 shots) the gap is not closed by 50 or 200 iterations instead of 20 (95.1% to
  95.3% agreement), nor by dropping the cross-fitting (94.0%). The circuit's own model agrees on
  only 77.8%.

**Willow**, on the first 10,000 shots of each of the 420 experiments, with Google's SI1000 prior.
BP costs about 25 ms a shot on ten cores at d = 7 and 250 rounds, so the full 50,000 would take
some 18 hours. Every decoder below is scored on the same shots:

| decoder, same 10,000 shots each | ε, d = 3 | ε, d = 5 | ε, d = 7 | Λ [95%] |
|---|---|---|---|---|
| ours, belief-matching | 0.824% | 0.452% | 0.251% | 1.81 [1.80, 1.83] |
| ours, correlated matching | 0.888% | 0.448% | 0.231% | 1.96 [1.95, 1.98] |
| ours, plain matching | 1.028% | 0.675% | 0.429% | 1.55 [1.54, 1.56] |
| Google, correlated matching | 0.835% | 0.442% | 0.229% | 1.91 [1.90, 1.92] |
| Google, Harmony (RL prior) | 0.731% | 0.387% | 0.204% | 1.89 [1.88, 1.91] |
| Google, Libra (RL prior) | 0.713% | 0.350% | 0.170% | 2.05 [2.03, 2.06] |

- **Belief-matching wins small and loses large.** At d = 3 it beats every matcher here, correlated
  ones included (0.824% against 0.888% for ours and 0.835% for Google's). At d = 5 it ties
  correlated matching; at d = 7 it is worse (0.251% against 0.231%). So its Λ, 1.81, is lower than
  correlated matching's 1.96.
- **BP rarely settles on these models.** It converges alone on 16% of Willow's shots. Elsewhere its
  posteriors come from a hypergraph full of short loops, where BP is only approximate. These results
  suggest the posteriors help the matcher less as the patch grows; they do not show why.
- **On Sycamore it is among the best.** Those experiments run at most 25 rounds, with priors fitted to
  the data, and belief-matching is one of the two best decoders there, Google's and ours alike.

**Longer BP does not change it** (`tools/belief.py iterations`). On the same shots, 100 iterations
instead of 20 move the failures by at most 4:

| experiment, same 10,000 shots | correlated matching | belief-matching, 20 iterations | 100 iterations | BP alone |
|---|---|---|---|---|
| d = 3, 110 rounds | 4,546 | 4,398 | 4,402 | 9 |
| d = 7, 50 rounds | 902 | 960 | 959 | 0 |
| d = 7, 110 rounds | 1,969 | 2,038 | 2,035 | 0 |

**On the site**, Figure 9 fits both tables from `data/belief/`, and Figure 10 belief-matches its
2,000 raw Willow shots at each distance, in WebAssembly, after the other decoders. At 30 rounds it
beats correlated matching at d = 3 and 5 (452 against 493, 169 against 184) and is within noise at
d = 7 (129 against 123). It takes 5 to 50 ms a shot.

## Throughput

Two changes make the engine produce and decode shots many times faster, natively and in the page.

**A bit-parallel sampler** (`src/batch_sampler.rs`).
- **The frame.** It keeps the Pauli frames of 64 shots in two machine words per qubit, one for X
  and one for Z, so a gate is a couple of word operations for all 64.
- **Noise words.** Noise comes as whole words of Bernoulli bits, drawn by geometric skipping: the
  gap to the next error is `floor(ln U / ln(1 − p))`. A location at p = 10⁻³ costs about one
  random number, not 64.
- **`REPEAT` without flattening.** Measurement records live in a ring buffer sized by the longest
  lookback, and detectors are evaluated from their lookbacks as they are reached. So a
  `REPEAT 1000000` block costs its body's memory, and detection events stream out as they are made.
- **The reference stays.** `FrameSampler`, one shot at a time, is kept as the reference. The two
  share their semantics exactly: disjoint channels, Pauli numbering, and the randomisation that
  makes a nondeterministic detector show itself as a coin flip.

It is checked five ways:
1. **Exactly `FrameSampler`'s shots on deterministic noise.** On rotated and XZZX circuits, both
   bases, d = 3 and 5, every noise channel is made deterministic (a Pauli at p = 0 or 1), 20
   rewrites each. Every lane of every batch equals `FrameSampler`'s shot, detector for detector.
2. **`REPEAT` without flattening** gives words identical to the flattened circuit's from the same
   seed, on Stim's circuits and on nested loops.
3. **Marginals.** Every detector's rate over 200,000 shots is within 5σ of the error model's exact
   prediction.
4. **Against Stim.** The cross-check's check 3 now holds it to Stim's per-detector rates (every
   χ² z-score under 1.2) and to PyMatching's logical error rate, at d = 3, 5, 7 over 100,000 shots
   a point.
5. **A million rounds** of a one-qubit loop sample in constant memory, at the right rate.

**Speed**, single-threaded, per shot, rotated SD6 with T = d (check 4 of the cross-check, 100,000
shots):

| d | p | Stim | `FrameSampler` | batch sampler |
|---|---|---|---|---|
| 3 | 0.3% | 0.12 µs | 1.23 µs | 0.07 µs |
| 5 | 0.3% | 0.49 µs | 5.97 µs | 0.32 µs |
| 7 | 0.3% | 1.31 µs | 17.0 µs | 0.84 µs |
| 7 | 0.6% | 2.27 µs | 17.5 µs | 1.06 µs |

- Stim is timed through its Python API, compiled beforehand and writing packed bits.
- Ours includes transposing batches into Stim's b8 rows.
- On these circuits the batch sampler is 15 to 20 times `FrameSampler`'s speed (15 to 16 in the Rust
  benchmark, 17 to 20 here), and a little
  faster than Stim's sampler as called from Python.
- Decoding, not sampling, is now almost all of a shot's cost.

**A worker pool on the page** (`js/pool.js`).
- **Why a pool.** WebAssembly threads need SharedArrayBuffer, and SharedArrayBuffer needs
  cross-origin isolation headers that GitHub Pages cannot send. So the page's parallelism is a pool
  of ordinary workers instead: one per core but one, at most eight.
- **One download.** The engine is compiled once on the page, and the same module is handed to every
  worker, so the reader downloads it once.
- **How the work is split.** The sweep, the results table, the bias comparison, the bench, Figure
  8's decoding rows and Figure 10's distances split their work across the pool. Every result is a
  sum over independent shots, so splitting changes the wall time and nothing else.
- **Scheduling.** Jobs wait in the pool, not in a worker's queue, so a worker that finishes early
  takes the next one. The sweep hands out its largest distances first.
- **The measured gain.** On an M2 Pro (6 performance and 4 efficiency cores), the section 7 sweep
  (216,000 shots over 36 points) takes 6.6 to 11.2 s over three runs, against 27.8 to 29.6 s with
  one worker: 2.5 to 4.4 times faster. The spread is which cores the workers land on; eight
  workers on this machine include efficiency cores. The times are polled once a second.
  `?workers=N` sets the pool's size, for measuring exactly that.

## Real time

Every decoder above waits for an experiment to end. A quantum computer cannot: Willow runs a round
every 1.1 µs, and a computation waiting on a logical measurement stalls until the decoder catches
up. A decoder slower than the chip falls further behind with every round and never recovers.
Google decoded a distance-5 memory in real time over a million rounds with a 63 µs average latency
(arXiv:2408.13687).

**Window decoders** (`src/window.rs`).
- **The idea.** A real-time decoder matches a few rounds at a time. It decodes a window of
  commit + buffer rounds and commits the edges of its correction that touch the commit region. It
  toggles their far ends, so a correction reaching past the region leaves a defect for the next
  window.
- **Windows are cut from any model by its time coordinates.** An edge leaving a window becomes a
  boundary half-edge of its own, kept apart from the real boundary. That lets a defect near the
  window's end wait for a partner not yet seen.
- **Sliding windows** run one after another, so a stream uses one core.
- **Parallel windows** (Skoric et al., arXiv:2209.08552; Tan et al., arXiv:2209.09219) come in two
  layers:
  - layer A's windows are spaced apart, with buffers and virtual boundaries on both sides;
  - layer B's windows fill the gaps once A has committed.

  Each layer decodes independently, so a stream can use as many cores as it needs.
- **Correlated matching works inside a window.** The second pass's matching is traced on the
  lowered weights before they are restored, and the correlation rules are restricted to the
  window's edges.

It is checked four ways:
1. **One window is global decoding.** With a commit region as long as the stream, the window's
   correction is the global decoder's traced correction exactly, plain and correlated, on rotated
   and XZZX circuits.
2. **Every defect is explained.** On every shot, for every commit size from 1 to 3 and buffer from 0
   (sliding) or 1 (parallel) to 3, in both schedules and with both matchers, the committed edges
   explain the shot's defects exactly. This caught a real hole: parallel windows with no buffer
   have no gaps for layer B, and leave corrections nowhere to land. They are now refused.
3. **Accuracy on Google's data.** Every Willow experiment is decoded with C = B = d, beside the
   global decoder (see the table below).
4. **A template is the full model.** The million-round stream never builds a million-round model.
   Each window takes the graph of the window in the same position in a short template: the ends
   from the template's ends, the bulk from its middle. On a 60-round stream this decodes all 64
   streams exactly as windows cut from the full 60-round model, and deep bulk windows are identical
   graphs, edge for edge.

**Latency.**
- **Method.** Every window's decode time is measured natively, one core at a time. The times are
  then scheduled with rounds arriving every 1.1 µs: a window starts once its last round has arrived
  and, for layer B, both its layer-A neighbours are done.
- **Latency** is the time from a window's last round arriving to its commit. A stream "keeps up" if
  its latency does not grow along it.
- **Streams:**
  - Google's recorded Willow syndromes (2,000 shots of d = 3, 5, 7 at 250 rounds, SI1000 prior);
  - a simulated d = 5 memory **a million rounds long**, 64 streams of rotated SD6 at p = 0.313%,
    the noise at which its detectors fire as often as Willow's (7.5%).

  Commit and buffer are both d rounds, on an Apple M2 Pro:

| stream | decoder | window decode | keeps up on | mean latency | p99 |
|---|---|---|---|---|---|
| Willow d = 3 | sliding, plain | 1.1 µs | 1 core | 1 µs | 5 µs |
| Willow d = 3 | parallel, correlated | 2.6 µs | 1 core | 8 µs | 26 µs |
| Willow d = 5 | parallel, plain | 8.6 µs | 2 cores | 19 µs | 46 µs |
| Willow d = 5 | parallel, correlated | 17 µs | 4 cores | 34 µs | 83 µs |
| Willow d = 7 | parallel, plain | 31 µs | 4 cores | 58 µs | 122 µs |
| Willow d = 7 | parallel, correlated | 65 µs | 8 cores | 114 µs | 236 µs |
| SD6 d = 5, 10⁶ rounds | parallel, plain | 7.7 µs | 1 core (4 for 18 µs) | 34 µs (18 µs on 4) | 129 µs (40 µs on 4) |
| SD6 d = 5, 10⁶ rounds | parallel, correlated | 16 µs | 2 cores (8 for 32 µs) | 82 µs (32 µs on 8) | 318 µs (72 µs on 8) |

- **Sliding windows** keep up only at d = 3, with either matcher. Everywhere else one core is too
  slow, which is exactly why parallel windows exist.
- **Every defect of every stream is explained**, all 64 × 10⁶ rounds of each run. The count is made
  as rounds are let go. An earlier version let rounds go without counting what was left in them,
  so its zero could not have been anything else; review caught it, and these runs are the repeat
  with the count working.
- **At d = 5, correlated matching on 4 cores holds a 32 µs mean latency over a million rounds**,
  and plain matching keeps up on a single core, beside Google's 63 µs for its own real-time decoder. That is on its hardware with its definition
  of latency, so it is context, not a like-for-like comparison.
- **Throughput.** On all 10 cores, independent streams decode at 6.3 million rounds a second plain
  and 3.2 million correlated. Willow's cycle demands 0.91 million per stream.
- **In the browser** (Figure 12), WebAssembly runs window decoding at 0.77 to 0.80, 2.9 to 3.0 and
  6.9 to 7.1 µs per round of one stream at d = 3, 5 and 7 (three runs in headless Chrome). About 1, 3
  and 7 cores would keep up. The timed call is window decoding alone: the streams are
  sampled as they are decoded, as a real stream would arrive, and the copy kept for global
  decoding is drawn outside it.

**Accuracy.** Windowed decoding of every Willow experiment, fitted like section 11 (see
`data/realtime/willow-windows.json`):

| decoder | ε₃ | ε₅ | ε₇ | Λ [95%] |
|---|---|---|---|---|
| global, plain | 1.032% | 0.684% | 0.435% | 1.54 [1.54, 1.55] |
| sliding windows, plain | 1.034% | 0.685% | 0.436% | 1.54 [1.53, 1.54] |
| parallel windows, plain | 1.033% | 0.685% | 0.436% | 1.54 [1.53, 1.54] |
| global, correlated | 0.888% | 0.449% | 0.234% | **1.95** [1.94, 1.95] |
| sliding windows, correlated | 0.892% | 0.454% | 0.236% | **1.94** [1.94, 1.95] |
| parallel windows, correlated | 0.892% | 0.452% | 0.235% | **1.95** [1.94, 1.95] |

- **Windows cost nothing in Λ.** On 392 experiments (every Willow experiment but the 28 single-round
  ones, which have nothing to window), 50,000 shots each, parallel windows with correlated matching
  give the same Λ as global correlated matching. ε rises by under 1% of itself.
- **The buffer matters, and d is enough.** At d = 5 the study also ran buffers of d/2 and 2d. A buffer
  of 3 rounds raises ε₅ to 0.694% plain and 0.460% correlated, about 2.5% worse. A buffer of 10 gives
  0.684% and 0.451%, the same as d = 5.
- **Simulated SD6 agrees.** At p = 0.3% and 0.5%, d = 3, 5, 7, 50 rounds and 20,000 shots, every
  windowed decoder's failures are within 1.6% of the global decoder's on the same shots
  (`data/realtime/sd6-windows.json`).
- **Nothing is left unexplained.** Across all of it, every window decoder explained every defect.

Reproduce with `python tools/realtime.py accuracy` (a few hours on 10 cores) and
`node tools/lambda.mjs 400 data/realtime/willow-windows.json`.

## Beyond the surface code: IBM's gross code

A surface code spends about 2d² physical qubits on each logical qubit. IBM's bivariate bicycle codes
(Bravyi, Cross, Gambetta, Maslov, Rall and Yoder, *Nature* 627, 778, 2024) keep many at once. The
**gross code**, [[144, 12, 12]], stores 12 logical qubits at distance 12 on 144 data and 144 check
qubits; twelve distance-12 surface codes would need about 3,450.
- **The code.** It lives on a 12 × 6 torus. With x and y the shifts along it, A = x³ + y + y² and
  B = y³ + x + x² give H_X = [A | B] and H_Z = [Bᵀ | Aᵀ].
- **The decoder.** Every check has weight six, and a fault sets off several at once, so nothing is
  matchable. The decoder is **BP+OSD**:
  - belief propagation (`src/bp.rs`, the same BP as belief-matching);
  - where BP does not settle, ordered-statistics decoding (`src/osd.rs`). It solves the syndrome
    equation on the faults BP trusts least: OSD-0, or with combinations of other faults tried as
    well (OSD-E, OSD-CS).

**Built** (`src/gf2.rs`, `src/bb.rs`):
- **Logical operators** are computed over GF(2), not typed in. Z logicals are kernel vectors of H_X
  outside the rowspace of H_Z, X logicals likewise, paired by inverting their overlap matrix.
- **The memory experiment** is the paper's depth-8 syndrome cycle, from the schedule in its
  published simulation code, with the paper's noise model and Z-basis memory.
- **Its error model** is built by the same backward walk as every model here, undecomposed, since
  these faults have no graph-like pieces (`Dem::from_circuit_undecomposed`).

**Checked** (`tools/bb_check.py`, and the Rust tests):
- The check matrices equal ones built independently the way Bravyi et al.'s code builds them
  (Kronecker products of cyclic shifts). k = 12, the logical pairs anticommute exactly in pairs,
  and every logical commutes with every check.
- Without noise every detector of the memory circuit is deterministic, which is what says the
  reconstructed schedule measures the stabilizers. The error model equals Stim's fault for fault.
- BP+OSD's corrections equal `ldpc`'s `BpOsdDecoder`'s on every shot tried, on random matrices and
  on the memory's model: OSD-0, OSD-E and OSD-CS, with `ldpc`'s adaptive min-sum scaling.
  Exhaustive OSD finds the brute-force optimum.

**Measured** (`tools/gross.py`; M2 Pro): the Z memory over N_c syndrome cycles (12 for the gross
code, 6 for [[72, 12, 6]]). A shot fails if any logical qubit does, and the logical error per
cycle is 1 − (1 − P_L)^(1/N_c). The decoder is the paper's: BP+OSD-CS of order 7, min-sum BP with
adaptive scaling for up to 10,000 iterations. BP+OSD-0 runs on the same shots.

| code | p | shots | failures | per cycle, BP+OSD-CS [95%] | per cycle, BP+OSD-0 | BP alone |
|---|---|---|---|---|---|---|
| gross [[144, 12, 12]] | 0.2% | 401,408 | 42 | 8.7 × 10⁻⁶ [6.5 × 10⁻⁶, 1.2 × 10⁻⁵] | 2.4 × 10⁻⁵ | 99.7% |
| gross [[144, 12, 12]] | 0.3% | 94,208 | 202 | 1.8 × 10⁻⁴ [1.6 × 10⁻⁴, 2.1 × 10⁻⁴] | 3.7 × 10⁻⁴ | 97.8% |
| gross [[144, 12, 12]] | 0.4% | 12,288 | 230 | 1.6 × 10⁻³ [1.4 × 10⁻³, 1.8 × 10⁻³] | 2.6 × 10⁻³ | 90.5% |
| gross [[144, 12, 12]] | 0.5% | 2,048 | 247 | 1.1 × 10⁻² [9.4 × 10⁻³, 1.2 × 10⁻²] | 1.5 × 10⁻² | 68.7% |
| gross [[144, 12, 12]] | 0.6% | 2,048 | 684 | 3.3 × 10⁻² [3.1 × 10⁻², 3.6 × 10⁻²] | 4.4 × 10⁻² | 39.5% |
| [[72, 12, 6]] | 0.2% | 71,680 | 204 | 4.7 × 10⁻⁴ [4.1 × 10⁻⁴, 5.4 × 10⁻⁴] | 4.9 × 10⁻⁴ | 99.9% |
| [[72, 12, 6]] | 0.3% | 14,336 | 217 | 2.5 × 10⁻³ [2.2 × 10⁻³, 2.9 × 10⁻³] | 2.9 × 10⁻³ | 98.8% |
| [[72, 12, 6]] | 0.4% | 4,096 | 225 | 9.4 × 10⁻³ [8.2 × 10⁻³, 1.1 × 10⁻²] | 1.0 × 10⁻² | 95.1% |
| [[72, 12, 6]] | 0.5% | 2,048 | 295 | 2.6 × 10⁻² [2.3 × 10⁻², 2.9 × 10⁻²] | 2.8 × 10⁻² | 87.4% |
| [[72, 12, 6]] | 0.6% | 2,048 | 565 | 5.2 × 10⁻² [4.8 × 10⁻², 5.7 × 10⁻²] | 5.7 × 10⁻² | 73.2% |

- **Below 12 unprotected qubits everywhere measured.** At p = 0.6% the gross code fails 3.3% of
  cycles, where 12 bare qubits would fail 6.9%.
- **OSD-CS earns its cost.** Against OSD-0 it cuts the gross code's failures 2.7 times at p = 0.2%
  and 2 times at 0.3%, where most shots never reach OSD at all.
- **The gross code falls steeply.** From 0.3% to 0.2% its error per cycle falls twentyfold.

**Beside the surface code.** The same 12 logical qubits as twelve rotated d = 11 surface-code patches
need 12 × 241 = 2,892 qubits, ten times the gross code's 288. Measured the same way (a Z memory of 12
rounds at the same p, sampled here and decoded by correlated matching; SD6, which also puts noise on
the Hadamards the surface code's X checks use), the chance that any of the twelve fails in a cycle
is 1 − (1 − p_L)¹² of a single patch's:

| p | gross code: 12 logical qubits on 288 | twelve d = 11 surface patches on 2,892 | ratio |
|---|---|---|---|
| 0.2% | 8.7 × 10⁻⁶ | 8.2 × 10⁻⁵ | 9.3× |
| 0.3% | 1.8 × 10⁻⁴ | 1.2 × 10⁻³ | 6.8× |
| 0.4% | 1.6 × 10⁻³ | 7.9 × 10⁻³ | 5.0× |
| 0.5% | 1.1 × 10⁻² | 3.3 × 10⁻² | 3.1× |
| 0.6% | 3.3 × 10⁻² | 8.7 × 10⁻² | 2.6× |

With a tenth of the qubits, the gross code fails less often at every p measured, by 2.6 to 9.3 times.

### Computing on the gross code: automorphisms and logical measurement

A memory is half of a computer. IBM's architecture computes on the gross code two ways, and the
engine builds both exactly (`src/bb_auto.rs`, `src/bb_gauge.rs`, `src/bb_circuit.rs`).

**Automorphisms.** Every shift x^a y^b of both halves of the data maps the code to itself, and so
does Bravyi et al.'s ZX-duality (left cell g to right g⁻¹ and back, X and Z exchanged). Each is
therefore a logical Clifford gate that adds no checks.
- For all 144 maps, the engine checks over GF(2) that the stabilizer group maps to itself, and
  computes the action on the 12 logical qubits as a 24 × 24 symplectic matrix, every residual a
  stabilizer. `tools/gross_ops.py auto` recomputes all 144 with numpy and `ldpc`'s GF(2) rank, and
  they agree bit for bit.
- The actions form a **group of order 72**: x⁶ acts as the identity, so the shifts give
  Z₆ × Z₆, doubled by the duality.
- Bravyi et al.'s weight-12 logicals X(f, 0) and X(g, h) each have 72 translates. They fall into
  36 logical classes and span a 6-qubit block that every shift keeps: the two blocks of
  six logical qubits of IBM's layout.

**Logical measurement.** To measure a weight-12 logical X̄, Cross, He, Rall and Yoder
(arXiv:2407.18393) attach an ancilla system, a mono-layer version of Williamson and Yoder's gauging
measurement. Their paper gives the construction but no edge lists, so the engine builds it from the
definition and checks it:
- the Z checks touching the operator are edges between its qubits, one ancilla qubit each;
- a Gauss-law X check per qubit (the qubit and its edges) multiplies to X̄;
- each touched Z check gains its edge qubit;
- flux Z checks on a light basis of the cycles fix the gauge.

The deformed checks commute, and 11 logical qubits remain while merged.

**The minimal system loses distance.** Built on X(f, 0), the code while merged has distance 8 in
X, exactly, not 12. The edges form a 3-regular graph on the operator's 12 qubits made of two
clusters of six joined by two edges. The Gauss-law checks on one cluster multiply to X on its six
qubits and on the two edges leaving it, so a logical that contains the cluster can trade six data
qubits for two edge qubits: 12 − 6 + 2 = 8. The product X(f, 0) X(g, h), the joint measurement a
CNOT needs, is worse: its worst cut is 11 vertices with 3 edges leaving them, and distance 4.
This is for these operators. Cross et al.'s eight operators are written in a notation that need
not give X(f, 0), and their shared system reports distance 12.

Williamson and Yoder's condition is that every set of at most half the vertices has at least as
many edges leaving it as it has vertices (a Cheeger constant of 1). For the product, some edges
are hyperedges of four ends; an edge counts as leaving a set when an odd number of its ends lie in
it, since that is when the Gauss-law checks on the set leave X on its qubit. The engine finds the worst cut
exactly (every set of at most half the vertices) and adds edges across it until the condition
holds. That takes 4 edges for X(f, 0), 2 for X(g, h) and 14 for the product, and every expanded
system has distance 12 in both types, exactly. X(g, h)'s minimal system keeps distance 12 even
though its constant is 2/3: the condition is sufficient, not necessary.

The circuit runs the paper's depth-8 schedule for the original checks. The flux checks read their
edge qubits alongside it, and the Gauss-law checks come last on every qubit they share with a Z
check, which makes every X–Z pair measurable together. The merged cycle takes 12 ticks for X(f, 0)
(13 expanded) against the memory's 8.

| operator | system | ancilla qubits (edges + Gauss + flux) | heaviest flux check | ticks per merged cycle | worst cut (edges out / vertices) | distance, X | distance, Z |
|---|---|---|---|---|---|---|---|
| X(f, 0) | minimal | 37 (18 + 12 + 7) | 8 | 12 | 2 / 6 | 8 | 12 |
| X(f, 0) | expanded | 45 (22 + 12 + 11) | 5 | 13 | 6 / 6 | 12 | 12 |
| X(g, h) | minimal | 37 (18 + 12 + 7) | 6 | 12 | 4 / 6 | 12 | 12 |
| X(g, h) | expanded | 41 (20 + 12 + 9) | 5 | 13 | 6 / 6 | 12 | 12 |
| X(f, 0) X(g, h) | minimal | 63 (31 + 22 + 10) | 11 | 16 | 3 / 11 | 4 | 12 |
| X(f, 0) X(g, h) | expanded | 91 (45 + 22 + 24) | 9 | 15 | 8 / 8 | 12 | 12 |

Distances are exact, by integer programming (`tools/gross_ops.py distance`, HiGHS): for each basis
logical of the other type, the lightest operator that commutes with every check and anticommutes
with it. The same program finds the gross code's own distance, 12 in both types. Cross et al.
report 103 ancilla qubits for one system that measures eight operators, with fault distance 12
from a schedule they optimise; each system here measures one operator.

**Checked** (`tools/gross_ops.py check`, in CI):
- Without noise every detector and observable is deterministic, in both bases, for all three
  operators and both systems, reading out after memory cycles or right at the split.
- A check whose whole support is read in its own basis closes its comparison with that reading:
  the flux checks at the split. A first version left them open, one detector per flux check
  short; the Z-basis measurements were retaken on the corrected circuits.
- The error model equals Stim's fault for fault.
- The cycle writer's plain memory equals `bb_memory_circuit`'s error model, mechanism for
  mechanism.
- BP+OSD's corrections equal `ldpc`'s on Stim's shots, but for ties: 13 of 3,072 shots differed
  over the twelve circuits (three operators, two systems, two bases), 10 of them for the
  product's minimal system. BP's posteriors were bit-identical, but many are exactly equal, and
  `ldpc` orders equal columns with C++'s `std::sort`, whose order among equals is the standard
  library's. `ldpc` returns our exact correction once its columns are permuted, so those shots are
  ties, and the check fails on any other difference.

**Measured** (`tools/gross_ops.py measure`): the data prepared, 6 memory cycles, T merged cycles, 6
memory cycles, the data read. In the X basis the observables are the outcome and the 12 X
logicals; in Z, the 11 Z logicals the measurement keeps, each carried through the edges. Circuit
noise p = 0.3%, Bravyi et al.'s model, decoded by BP+OSD-CS of order 7 on the whole error model.
Beside each point, the memory of the same length:

| operator | system | basis | T | shots | anything wrong | outcome wrong | memory, same length |
|---|---|---|---|---|---|---|---|
| X(f, 0) | expanded | X | 2 | 2,048 | 14.16% | 13.57% | 0.24% |
| X(f, 0) | expanded | X | 4 | 4,096 | 3.61% | 2.71% | 0.29% |
| X(f, 0) | expanded | X | 7 | 4,096 | 2.56% | 0.71% | 0.30% |
| X(f, 0) | expanded | X | 12 | 4,096 | 2.93% | 0.24% | 0.39% |
| X(f, 0) | expanded | Z | 2 | 12,288 | 0.84% | — | 0.30% |
| X(f, 0) | expanded | Z | 4 | 6,144 | 1.74% | — | 0.31% |
| X(f, 0) | expanded | Z | 7 | 4,096 | 3.64% | — | 0.42% |
| X(f, 0) | expanded | Z | 12 | 2,048 | 5.96% | — | 0.37% |
| X(f, 0) | minimal | X | 2 | 2,048 | 13.53% | 13.28% | 0.24% |
| X(f, 0) | minimal | X | 4 | 6,144 | 2.10% | 1.56% | 0.29% |
| X(f, 0) | minimal | X | 7 | 10,240 | 1.22% | 0.31% | 0.30% |
| X(f, 0) | minimal | X | 12 | 6,144 | 1.86% | 0.37% | 0.39% |
| X(f, 0) | minimal | Z | 2 | 14,336 | 0.74% | — | 0.30% |
| X(f, 0) | minimal | Z | 7 | 4,096 | 2.69% | — | 0.42% |
| X(g, h) | expanded | X | 7 | 6,144 | 2.15% | 0.59% | 0.30% |
| X(f, 0) X(g, h) | expanded | X | 7 | 2,048 | 5.71% | 2.69% | 0.30% |
| X(f, 0) X(g, h) | expanded | Z | 7 | 2,048 | 7.13% | — | 0.42% |

- **The outcome needs rounds, as a lattice-surgery merge does.** With two merged cycles the
  Gauss-law checks' last round has nothing after it to compare with, so a measurement error there
  looks like one in the first round. The decoder must guess, and the outcome is wrong 13.6% of the
  time. By seven cycles it is 0.71%, and at twelve 0.24% (expanded system).
- **The measurement costs 4 to 9 times the memory.** At seven merged cycles, a shot has anything
  wrong 4.0 (minimal) to 8.4 (expanded) times as often as the memory of the same 19 cycles in X,
  and 6.4 to 8.7 times in Z. The product operator costs 17 to 19 times. Cross et al. find 5 to 10
  times at p = 0.1%, with a schedule and a decoder they optimise; here p = 0.3%.
- **At this noise, distance is not the limit.** The expanded system fails more often than the
  minimal one in both bases: at p = 0.3% its extra qubits and longer cycle add more faults than
  distance 12 over 8 removes. Which wins at lower noise is not measured here.
- **The merged cycles are the costly part, and why is not settled.** In Z, failures grow by about
  0.5% per merged cycle, against 0.008% per memory cycle. BP settles 78% of shots at twelve merged
  cycles, against 95% for the memory of the same 24 cycles.
- **Fault distance, bounded above only.** An integer-programming search of each error model
  (`tools/gross_ops.py fault`, two minutes per observable) finds undetectable logical faults of 10
  faults in the merged circuits and 12 in the memory. So the fault distance is at most that; the
  search proves no lower bound, and it did not reach the memory's own bound of 10 that Bravyi et al.
  report. Before the flux checks closed their comparisons at the split, the same search found one
  of 6 faults.

**On the site**, section 13 draws the code on its torus (hover a check to see its six qubits). It
plots these measurements, runs BP+OSD on [[72, 12, 6]] in the browser, draws the ancilla system
that measures a logical operator, and plots that measurement against its merged cycles.

## More decoders

Matching needs faults that set off at most two detectors. For everything else the field
compares against four newer decoders, and each is here, ported from its authors' code with its
arithmetic in its order so that its answers can be held to theirs shot for shot:

- **BP+LSD** (`BpLsd`, `BpLsdDecoder`; Hillmann, Berent, Quintavalle, Eisert, Wille and Roffe,
  2024). Where BP does not converge, a cluster grows around each detection event by BP's
  beliefs, merging with the clusters it touches, until its syndrome is in the image of its own
  columns; each cluster is then solved alone (LSD-0) or searched as OSD searches (LSD-E,
  LSD-CS). The clusters iterate their faults in `tsl::robin_set`'s order, which `ldpc` uses and
  which decides the columns' order and so the answer (`src/robin.rs` reproduces it). Two of
  `ldpc`'s choices are not reproducible: equal keys in a long `std::sort`, and the order it
  merges three or more clusters at once, which libc++ takes from hashed pointers. A shot that
  met one is counted as a tie.
- **Relay-BP** (`RelayBp`, `RelayBpDecoder`; Müller et al., IBM, 2025). Min-sum BP whose
  faults remember their last beliefs, run in legs that each draw fresh memory strengths and
  start where the last ended; the lightest of the first few corrections to converge wins.
  The random strengths come from `rand` 0.8's generator, reproduced in `src/chacha.rs`, so for
  a seed the results are IBM's exactly.
- **Colour-code matching** (`ColorMatching`; Gidney and Jones, 2023). Chromobius's Möbius
  construction: each detector doubled into the two sub-graphs that leave out a colour other
  than its own, each fault drawn as edges of the doubled graph, a matching of it found by this
  package's matcher, and the matching lifted back by carrying colour charge around its cycles.
  The detectors need Chromobius's colour annotation (`annotate_colors=True` on the colour-code
  circuits). Where Chromobius's search for a way to drag charge would depend on `unordered_map`
  order and two ways differ, the entry is marked and a shot using it counted as a tie.
- **A search decoder** (`SearchDecoder`; Beni, Higgott and Shutty, Google, 2025). Tesseract's A*
  over sets of faults, cheapest first by weight plus an admissible estimate of what the lit
  detectors still cost, under a beam and a queue bound, over several detector orders. Equally
  promising states leave the queue as the platform's `std::priority_queue` lets them (libc++
  and libstdc++ are both implemented), so its faults are Tesseract's, fault for fault. With no
  beam, no queue bound and revisits allowed it is exact (tested against brute force).

**Checked** (`tools/decoder_check.py`; the quick form runs in CI): the same check matrices,
models and shots through both. BP+LSD on random matrices, a d = 5 surface code and the gross
code (2 and 6 cycles), LSD-0 with both BP methods, LSD-E 6 and LSD-CS 10, and each untied
cluster's faults in `ldpc`'s order; Relay-BP on the gross code with explicit strengths, seeded
strengths and no memory, correction, convergence and iteration count; colour-code matching on
annotated colour codes (d = 3 to 9) and Chromobius's own test code, the Möbius matching's
weight on every shot and the predictions; the search on surface, colour and gross-code models
with Tesseract's defaults, literal orders and beam climbing, the faults found. "Ties" differ
only where the reference's order is not reproducible (BP+LSD) or between two equally light
matchings (colour codes):

| decoder | against | cases | shots | equal | ties | otherwise | time (× theirs) |
|---|---|---|---|---|---|---|---|
| BP+LSD (LSD-0, -E, -CS) | ldpc 2.4.1 | 52 | 24,000 | 23,989 | 11 | 0 | 0.67 |
| Relay-BP | IBM's relay_bp 0.2.2 | 8 | 8,000 | 8,000 | 0 | 0 | 0.77 |
| Colour-code matching | Chromobius 1.1.1 | 17 | 340,000 | 339,829 | 171 | 0 | 0.72 |
| Search decoder | Tesseract 0.1.1.dev20260910235247 | 12 | 30,000 | 30,000 | 0 | 0 | 1.00 |

The time is this package's over the reference's on the same shots, one thread, Apple M2 Pro,
both decoders built beforehand. Tutorial 04 runs every decoder on a colour code and the gross
code. On the gross code BP+LSD fails more often than BP+OSD, as `ldpc`'s own pair does on the
same matrix.

## Lattice surgery

Every experiment above holds logical qubits still. **Lattice surgery** (Horsman, Fowler, Devitt and
Van Meter, 2012) is how surface-code qubits interact on a planar chip:
- two patches are merged across a seam by measuring the merged patch's checks for some rounds;
- the product of the new checks along the seam is the joint parity, here Z₁Z₂;
- the patches are split again.

`src/surgery.rs` writes the whole experiment as one circuit:
1. both patches prepared in |0⟩, then d rounds apart;
2. the seam prepared in |+⟩, then T merged rounds;
3. the seam read out in X, then d rounds apart;
4. the data read out, under SD6.

The observables are the merge outcome and each patch's own Z.

**One rule for detectors.** A check is compared with its previous value, and where its support
changed (the boundary X checks reaching across the seam at the merge, and pulling back at the
split), the qubits it gained were freshly prepared in its basis and the qubits it lost were read out
in it. A check measured for the first time is a detector only if its qubits were all freshly
prepared in its basis.

**Checked:**
- **Determinism.** Without noise, every detector and observable is deterministic at d = 3 and 5,
  in both bases, for 1, 2 and d merged rounds.
- **Stim and PyMatching.** The error models equal Stim's fault for fault, and split for split
  (`tools/surgery.py check`). Every disagreement with PyMatching is a tie.
- **An independent geometry check.** The site's geometry module (`js/surgery-geometry.js`) shows
  that the new seam checks multiply to Z on exactly the two facing columns, Z₁Z₂.
- **The physics of one round.** With a single merged round the error model is refused: one
  measurement error on a seam check flips the outcome with nothing after it to notice. With three,
  every single fault is corrected in both bases.

**The timing law** (`tools/surgery.py run`): how often the merge outcome is wrong, against merged
rounds T, correlated matching.

| d | p | T = 2 | T = d | T = 2d | T = d, plain | either patch, T = d |
|---|---|---|---|---|---|---|
| 3 | 0.3% | 10.99% | 6.38% | 6.15% | 7.53% | 10.93% |
| 5 | 0.3% | 12.87% | 3.48% | 3.19% | 5.12% | 6.25% |
| 7 | 0.3% | 15.34% | 1.62% | 1.60% | 3.21% | 2.99% |
| 3 | 0.2% | 7.31% | 3.23% | 2.90% | 3.59% | 5.65% |
| 5 | 0.2% | 7.61% | 1.02% | 0.94% | 1.65% | 1.85% |
| 7 | 0.2% | 9.97% | 0.28% | 0.27% | 0.71% | 0.54% |

- **T = 2 is a trap that grows with the code.** A pair of measurement errors on one seam check,
  in consecutive rounds, flips the outcome unseen, and a bigger code has more seam checks. At
  p = 0.3%, T = 2 fails 11%, 13% and 15% of the time at d = 3, 5 and 7.
- **By T = d the curve has flattened.** From then on the merge outcome fails about as often as a
  single patch of the same size, and further rounds buy nothing: the textbook "d rounds per
  lattice surgery", measured.
- **Correlated matching matters here too.** At d = 7, p = 0.2%, T = d it fails 0.28% of the time
  against plain matching's 0.71%.

### Programs: a logical CNOT, merges in a row, and three patches merged at once

One measurement is not yet a computation. `src/surgery.rs` compiles lattice-surgery **programs**:
patches on a grid of tiles, each step one of prepare, rounds, merge, split, and measure.
- A horizontal line of patches merges to measure each neighbouring pair's Z⊗Z, its seams prepared
  in |+⟩ and split by reading them in X.
- A vertical line merges to measure each pair's X⊗X, its seams prepared in |0⟩ and split by
  reading them in Z.
- Observables are written as terms: a patch's logical along a chosen line of its final readout, a
  merge's outcome on one seam, and a merge's seam records on a line.

The Z⊗Z experiment above is now a seven-step program, and compiles to the same circuits byte for
byte (28 of them, held to fingerprints taken before the compiler existed).

**The CNOT** (Horsman, Fowler, Devitt and Van Meter; Litinski) puts control C, an ancilla A in |+⟩
and target T in an L.
1. Z_C Z_A is measured (m₁), C and A side by side.
2. X_A X_T is measured (m₂), A above T.
3. A is read out in Z (m₃).

What is left is CNOT from C to T, up to Paulis fixed by the three outcomes, which are tracked, not
applied. Two input pairs measure how often it fails, in Z and in X:
- **|0⟩|0⟩ in:** the observables are Z_C, and Z_T ⊕ Z_A ⊕ m₁ with the A–T seam's Z records on that
  column.
- **|+⟩|+⟩ in:** they are X_T, and X_C X_T ⊕ m₂ with the C–A seam's X record.

**How the programs are checked:**
- **Frames by the reference simulation.** Every detector and observable of every program is
  deterministic without noise; the reference simulation refuses any that is not. It refused the
  first X frame, which took X_C X_A along the ancilla's first row: m₂ reads X on A's last row, and
  the rows differ by checks whose values are not fixed. The frame now uses the last row, and a site
  test holds the geometry to it.
- **A detector-rule case, found by the second merge.** A check measured again after skipping a
  round (a seam check of an earlier merge, whose seam has since been read and prepared again)
  compared itself with its stale value. It now starts over, as if measured for the first time.
- **The frames are exercised.** The CNOT's merge outcomes come out −1 in about half of noiseless
  shots, so a wrong frame could not hide.
- **The CNOT is a CNOT.** |0⟩|0⟩ and |+⟩|+⟩ are left alone by the identity too, so three more
  noiseless programs pin the gate. |+⟩|0⟩ must come out a Bell pair: Z_C Z_T and X_C X_T each fixed
  once the frame is applied. |0⟩|+⟩ must come out as it went in. With the two pairs above, this
  fixes the images of Z_C, Z_T, X_C and X_T as a CNOT's, up to signs. The test has teeth: Z_T alone
  after |+⟩|0⟩, which the identity would leave fixed, is refused as random.
- **A line is not a product.** Three patches merged in a row become one patch holding one logical
  qubit, so the merge measures Z₁Z₂ and Z₂Z₃, two parities, not the product Z₁Z₂Z₃. A first
  version called it a product; review caught it, and each seam now has its own outcome. A true
  product needs an ancilla region along every patch, which a rectangle of tiles cannot be.
- **A seam split and merged again at once.** A check keeping its support while its seam qubits are
  read and prepared again, with no round between, takes those readings into its comparison (or
  starts over if their basis is not its own). Found in review; a test holds it.
- **Against Stim and PyMatching.** Every single fault of every program is corrected at d = 3. Each
  program's error model equals Stim's fault for fault, and PyMatching and ours agree on every shot
  but ties (`tools/surgery.py check`, run in CI).

**The CNOT, measured** (`tools/surgery.py programs`: T = d, correlated matching unless marked;
"any wrong" is a shot with either observable wrong; Z inputs are |0⟩|0⟩, X inputs |+⟩|+⟩). Beside
it, three patches held as memories for the CNOT's 4d rounds:

| d | p | Z inputs | X inputs | Z inputs, plain | three patches idle as long | CNOT ÷ idle |
|---|---|---|---|---|---|---|
| 3 | 0.2% | 10.66% | 11.46% | 11.47% | 13.07% | 0.82 |
| 5 | 0.2% | 3.54% | 3.85% | 5.98% | 4.50% | 0.79 |
| 7 | 0.2% | 1.03% | 1.12% | 2.61% | 1.37% | 0.75 |
| 3 | 0.3% | 19.45% | 22.20% | 21.53% | 24.00% | 0.81 |
| 5 | 0.3% | 11.11% | 12.06% | 16.93% | 14.34% | 0.77 |
| 7 | 0.3% | 5.60% | 6.42% | 11.45% | 7.11% | 0.79 |

- **It falls with distance, below threshold.** At p = 0.2%: 10.7%, 3.5% and 1.0% at d = 3, 5, 7.
- **It costs less than holding its three patches.** It fails 0.75 to 0.93 times as often as three
  patches idle for as long. The ancilla is read out early, and each observable spans fewer patch
  rounds than three whole memories. So the resource estimate's model, every patch exposed for every
  operation, is if anything pessimistic.
- **Correlated matching matters most here.** At d = 7, p = 0.3%, plain matching fails 11.5% against
  correlated's 5.6%.
- **Window decoding loses almost nothing.** Decoded by parallel windows at d = 5, p = 0.3%, it fails
  11.27% and 12.88% against 11.27% and 12.70% decoded whole.

**Merges in a row.** k Z⊗Z measurements on the same two patches, each over d merged rounds:

| d | k = 1 | k = 2 | k = 4 | k = 8 | per merge |
|---|---|---|---|---|---|
| 3 | 12.30% | 16.97% | 27.40% | 42.78% | 5.96% |
| 5 | 6.56% | 8.86% | 12.65% | 20.81% | 2.33% |

ln(1 − P) is a straight line in k: each merge adds the same risk, on top of the fixed rounds before
and after. That is the sum-of-parts law a resource estimate assumes, measured. **Three patches
merged at once**, measuring Z₁Z₂ and Z₂Z₃ together, fail 18.04%, 9.36% and 4.61% at d = 3, 5, 7
and p = 0.3%.

## Circuit distance

A code's distance counts the fewest errors on its data qubits that change a logical value
unnoticed. A circuit that measures the code can do worse: one fault in the middle of a
syndrome cycle can spread to several qubits. The *circuit distance* is the fewest faults of the
whole circuit that flip an observable and set off no detector. It is measured three ways:

- `shortest_graphlike_error`: the shortest such set among faults of at most two detectors, by
  Stim's algorithm, on circuits and on models;
- `search_for_undetectable_logical_errors`: Stim's breadth-first search through every fault,
  hyperedges included, within its limits (here: sets of up to 6 detectors, faults of up to
  6), which gives an upper bound;
- `DetectorErrorModel.distance()`: an integer program over every fault, which proves the
  distance exactly when it finishes (scipy's HiGHS). When it does not, the best solution it
  found is still a real undetectable logical error, and so an upper bound.

The graph-like search is Stim's own algorithm and returns Stim's own faults: its model text and
its explained errors equal Stim's character for character (on Stim's generated codes, 300
random models and 100 random circuits), in about a tenth of Stim's time. The breadth-first
search gives the same number of faults as Stim's on every circuit compared (Stim's surface,
repetition and colour codes, the Steane code and the [[72, 12, 6]] code). Each fault either
returns is explained: the gate it comes from, its targets, and the Pauli product or measurement
it flips, as Stim's `explain_detector_error_model_errors` prints it.
Every memory the package builds, measured by `tools/distances.py` (d rounds; the bivariate
bicycle memories 2):

| memory | code distance | faults | graph-like search | Stim's search | integer program |
|---|---|---|---|---|---|
| surface, rotated (Stim's) | 3 | 219 | 3 | 3 | **3** |
| surface, rotated (Stim's) | 5 | 1,677 | 5 | — | **5** |
| surface, rotated (Stim's) | 7 | 6,023 | 7 | — | ≤ 7 (not proven in 300 s) |
| surface, unrotated (Stim's) | 3 | 423 | 3 | 3 | **3** |
| surface, unrotated (Stim's) | 5 | 3,139 | 5 | — | **5** |
| colour code XYZ (Stim's) | 3 | 72 | 2 | 2 | **2** |
| colour code XYZ (Stim's) | 5 | 1,104 | 3 | 3 | **3** |
| colour code XYZ (Stim's) | 7 | 3,651 | 4 | — | **4** |
| surface, rotated, SD6 | 3 | 219 | 3 | 3 | **3** |
| surface, rotated, SD6 | 5 | 1,677 | 5 | — | **5** |
| surface, XZZX, SD6 | 3 | 219 | 3 | 3 | **3** |
| surface, XZZX, SD6 | 5 | 1,677 | 5 | — | **5** |
| colour code 6.6.6 (CssCode) | 3 | 277 | 2 | 2 | **2** |
| colour code 6.6.6 (CssCode) | 5 | 2,388 | 3 | 3 | **3** |
| Hamming hypergraph product [[58, 16, 3]] (CssCode) | 3 | 3,564 | 3 | 3 | **3** |
| bivariate bicycle [[72, 12, 6]] | 6 | 792 | — | 6 | ≤ 6 (not proven in 300 s) |
| bivariate bicycle [[90, 8, 10]] | 10 | 990 | — | — | ≤ 9 (not proven in 300 s) |
| bivariate bicycle [[108, 8, 10]] | 10 | 1,188 | — | — | ≤ 11 (not proven in 300 s) |
| bivariate bicycle [[144, 12, 12]] (gross) | 12 | 1,584 | — | — | ≤ 12 (not proven in 300 s) |

- **The surface codes keep their distance.** Rotated, unrotated and XZZX, Stim's circuits and
  the SD6 ones: d at d = 3 and 5, proven.
- **The colour code does not.** Stim's generated XYZ colour-code memory has circuit distance 2,
  3 and 4 at code distance 3, 5 and 7, proven, and Stim's own search agrees where it reaches. Two
  or three correlated two-qubit faults do the work of d data errors. The generic `CssCode`
  schedule loses the colour code's distance the same way, so the loss belongs to the code's
  schedules, not only to one circuit.
- **The bivariate bicycle codes were not settled.** Five minutes proved none of them. For
  [[90, 8, 10]] the integer program found 9 faults that flip a logical qubit unnoticed, so this
  2-round memory has circuit distance at most 9, below the code's 10. Bravyi et al. (their
  Table 1) bound the circuit distance at 6, 8, 8 and 10 for [[72, 12, 6]], [[90, 8, 10]],
  [[108, 8, 10]] and the gross code. Ours are 6, 9, 11 and 12, so at this time limit the
  larger two are loose and say nothing new. The graph-like search does not apply (their
  faults are not graph-like), and Stim's search finds nothing within its limits.

## What it would take: a resource estimate

Every section above measures one ingredient of a quantum computer's cost. `js/estimator.js`
(section 15 of the site, and `node tools/estimate.mjs`) puts them together with what the papers
that price real algorithms report. Every number read from a paper is in
`data/estimate/sources.json` with its page or table.

**The model:**
- **Algorithms.**
  - RSA-2048: 1,409 logical qubits (1,280 of them idle input), 6.5 × 10⁹ Toffolis, from Gidney
    (arXiv:2505.15917, Tables 4 and 5).
  - FeMoco by tensor hypercontraction, from Lee et al. (arXiv:2011.03494, Table III): 2,142
    qubits and 5.3 × 10⁹ Toffolis (Reiher), and 2,196 and 3.2 × 10¹⁰ (Li).
  - Three illustrative sizes.
- **Logical error per patch per cycle.** Measured here under uniform SD6 noise, or as measured on
  Willow (section 11). The uniform law is fitted to this engine's own simulations of the rotated
  code (`tools/estimate_noise.py`: d = 3 to 11, p = 0.1% to 0.5%, correlated matching; ε per
  round, * where fewer than 20 failures):

| p | d = 3 | d = 5 | d = 7 | d = 9 | d = 11 |
|---|---|---|---|---|---|
| 0.1% | 9.5e-04 | 9.8e-05 | 8.5e-06 | 8.3e-07 | 7.4e-08 |
| 0.2% | 3.6e-03 | 6.8e-04 | 1.5e-04 | 3.5e-05 | 6.4e-06 |
| 0.3% | 6.7e-03 | 2.5e-03 | 7.8e-04 | 2.9e-04 | 1.2e-04 |
| 0.5% | 1.7e-02 | 1.0e-02 | 6.1e-03 | 4.1e-03 | 2.8e-03 |

- **Magic states.** Either Gidney's cultivation factory, or the cheapest of Litinski's
  distillation protocols. Cultivation makes T states to 10⁻⁷ and CCZs from them at 28 p_T², in a
  3 × 4-patch factory making a CCZ every 150 rounds at d = 25. Its figures are Gidney's at
  p = 0.1%, used for any lower p. Scaling the 150 rounds with d is this model's assumption.
  Litinski's protocols come from Quantum 3, 205, Table 1: the cheapest (Quantum 3, 205, Table 1) that
  meets the error each Toffoli may spend. Half the budget goes to magic states, and there are as
  many factories as keep up.
- **The floor plan.** One of Litinski's data blocks (Quantum 3, 128, Sec. 2): compact (1.5n + 3
  tiles, 9 steps per magic state), intermediate (2n + 4, 5) or fast (2n + √(8n) + 1, 1). A tile
  is 2d² physical qubits, and every tile is priced as an idle patch for the whole run. That errs
  on the safe side: the logical CNOT measured above fails less often than its patches held idle.
- **Reaction.** Each Toffoli waits for the longer of its lattice-surgery steps and its reaction.
  The reaction is this project's decoder latency (section 12's p99 for one window, extrapolated
  as a power law) plus a control delay (10 µs by default).
- **Storage** for idle qubits: surface-code patches, Gidney's yoked surface codes (430 qubits
  each at 10⁻¹⁵ per round), or gross-code modules (12 logical qubits on 288, plus Cross et al.'s
  103 ancillas, at the error per cycle section 13 measured). Storage is priced idle. Reaching a
  stored qubit is not priced: for the gross code that is a logical measurement, which section 13
  measured at 4 to 9 times a memory's failures.
- **Our own check of the distillation law** (`tools/distill.py`). The 15-to-1 protocol, twirled
  T-state errors against the [[15, 1, 3]] code's X checks, sampled by this engine (50 million
  shots per p) agrees with an exact sum over all 2¹⁵ error patterns. At p = 0.1% the output
  error is 1.003 × 35p³, and the acceptance is 1 − 15p to first order.

**Checked against the sources.** The first case is entirely Gidney's: his ε (10⁻¹⁵ per round at
d = 25), his schedule (3.7 Toffolis per 25 µs lattice-surgery step), yoked storage, and a 6.7%
failure budget per shot. Each case after it switches one option to this project's.

| case | d | physical qubits (× source) | time (× source) |
|---|---|---|---|
| RSA-2048: Gidney's assumptions, ε and schedule | 27 | 1.4 million (1.57×) | 13 h (1.08×) |
| RSA-2048: this engine's measured ε instead | — | yoked storage at 1e-15 per round would fail 6.9% on its own even over the shortest run it could need (15 h), past the 6.7% budget | |
| RSA-2048: one Toffoli per step instead | — | yoked storage at 1e-15 per round would fail 8.3% on its own even over the shortest run it could need (18 h), past the 6.7% budget | |
| RSA-2048: surface-code storage instead of yoked | 27 | 4.7 million (5.23×) | 13 h (1.08×) |
| RSA-2048: this project's decoder latency instead | — | yoked storage at 1e-15 per round would fail 13% on its own even over the shortest run it could need (27 h), past the 6.7% budget | |
| FeMoco (Reiher): Lee et al.'s noise, distillation | 35 | 11 million (2.73×) | 8.6 days (2.86×) |
| FeMoco (Reiher): cultivation instead | 35 | 11 million (2.75×) | 2.1 days (0.72×) |

- **The model reproduces Gidney's estimate.** On his assumptions, it lands within 1.6 times his
  897,864 qubits and 1.1 times his 12.07 hours per shot.
- **Storage is the biggest single choice.** Surface-code storage instead of yoked costs over 5
  times the qubits.
- **Everything that lengthens the run breaks his storage.** This engine's ε at p = 0.1%
  (extrapolated to d = 25, about 60 times his 10⁻¹⁵) forces a larger d, and larger steps. So does
  one Toffoli per step, and so does this project's decoder. Yoked storage, fixed at 10⁻¹⁵ per
  round, then exceeds the budget on its own.
- **FeMoco lands within 3 times Lee et al.'s** four million qubits and three days. The fast
  block's 2n tiles and this engine's ε put it above them.

**Real algorithms**, with the page's defaults (uniform 0.1% by this engine's fit, cultivation, the
fast block, surface storage, a 10 µs control delay), and with this project's decoder or with
decoding inside the control delay:

| algorithm | decoder | d | physical qubits (block + factories + storage) | per Toffoli | run time | bound |
|---|---|---|---|---|---|---|
| RSA-2048 (Gidney 2025) | measured decoder | 41 | 9.9 million (9.8 million + 42,336 + 0) | 42.5 ms | 8.8 years | reaction |
| RSA-2048 (Gidney 2025) | decoder within the control delay | 33 | 6.5 million (6.4 million + 166,464 + 0) | 33.0 µs | 2.5 days | steps |
| FeMoco, Reiher Hamiltonian (Lee et al. 2021) | measured decoder | 41 | 15 million (15 million + 42,336 + 0) | 42.5 ms | 7.1 years | reaction |
| FeMoco, Reiher Hamiltonian (Lee et al. 2021) | decoder within the control delay | 35 | 11 million (11 million + 186,624 + 0) | 35.0 µs | 2.1 days | steps |
| FeMoco, Li Hamiltonian (Lee et al. 2021) | measured decoder | — | cultivation's CCZ error, 2.8 × 10⁻¹³, is more than the 1.6 × 10⁻¹³ per Toffoli this budget allows | | | |
| FeMoco, Li Hamiltonian (Lee et al. 2021) | decoder within the control delay | — | cultivation's CCZ error, 2.8 × 10⁻¹³, is more than the 1.6 × 10⁻¹³ per Toffoli this budget allows | | | |
| Illustration: 100 qubits, 10⁶ Toffolis | measured decoder | 29 | 408,460 (386,860 + 21,600 + 0) | 13.9 ms | 3.8 h | reaction |
| Illustration: 100 qubits, 10⁶ Toffolis | decoder within the control delay | 23 | 326,284 (243,340 + 82,944 + 0) | 23.0 µs | 23 s | steps |
| Illustration: 1,000 qubits, 10⁹ Toffolis | measured decoder | 39 | 6.4 million (6.4 million + 38,400 + 0) | 36.1 ms | 1.1 years | reaction |
| Illustration: 1,000 qubits, 10⁹ Toffolis | decoder within the control delay | 31 | 4.2 million (4.0 million + 147,456 + 0) | 31.0 µs | 8.6 h | steps |
| Illustration: 10,000 qubits, 10¹² Toffolis | measured decoder | — | cultivation's CCZ error, 2.8 × 10⁻¹³, is more than the 5.0 × 10⁻¹⁵ per Toffoli this budget allows | | | |
| Illustration: 10,000 qubits, 10¹² Toffolis | decoder within the control delay | — | cultivation's CCZ error, 2.8 × 10⁻¹³, is more than the 5.0 × 10⁻¹⁵ per Toffoli this budget allows | | | |

- **Decoding is the other lever.** This project's decoder, extrapolated to the distances these
  algorithms need, takes tens of milliseconds per window. That makes every Toffoli wait for it,
  and RSA-2048 would take years. A decoder that finishes inside the control delay brings it to
  days. The extrapolation is long (measured at d ≤ 7), but so is the gap.
- **Cultivation has a floor.** Its 2.8 × 10⁻¹³ per CCZ is more than a 3.2 × 10¹⁰-Toffoli run can
  afford at a 1% budget, so the Li FeMoco and the largest illustration need distillation.

**Λ is the lever**, in the simple model of before: N patches of 2d² − 1 with a routing overhead of
2, every operation d merged rounds of Willow's 1.1 µs cycle, and ε falling by the Λ measured on
Willow. The sizes are illustrations of scale:

| algorithm | Λ (from) | d | physical qubits | run time | decoding cores |
|---|---|---|---|---|---|
| 100 qubits × 10⁶ operations | 1.95 (ours, correlated matching) | 71 | 2.0 million | 78 s | 94,586 |
| 100 qubits × 10⁶ operations | 2.04 (Google's Libra) | 67 | 1.8 million | 74 s | 83,936 |
| 100 qubits × 10⁶ operations | 1.81 (ours, belief-matching) | 79 | 2.5 million | 87 s | 117,855 |
| 1,000 qubits × 10⁹ operations | 1.95 (ours, correlated matching) | 101 | 41 million | 31 h | 2.0 million |
| 1,000 qubits × 10⁹ operations | 2.04 (Google's Libra) | 93 | 35 million | 28 h | 1.6 million |
| 1,000 qubits × 10⁹ operations | 1.81 (ours, belief-matching) | 111 | 49 million | 34 h | 2.4 million |
| 10,000 qubits × 10¹² operations | 1.95 (ours, correlated matching) | 129 | 666 million | 4.5 years | 32 million |
| 10,000 qubits × 10¹² operations | 2.04 (Google's Libra) | 119 | 566 million | 4.1 years | 27 million |
| 10,000 qubits × 10¹² operations | 1.81 (ours, belief-matching) | 143 | 818 million | 5.0 years | 40 million |

- **Λ ≈ 2 is not enough for anything large.** Even the small size needs d ≈ 70, because every
  factor of 10 in ε costs about 7 more steps of distance at Λ = 2.
- **Doubling Λ cuts the qubits about fourfold.** At Λ = 4 the medium size would need d = 51 and
  10 million physical qubits.

## Technical report

`report/report.md` is the source. `python3 tools/report.py` builds `report/report.html`, which
is served with the site, and `report/report.pdf`, printed from it by headless Chrome. It needs
Node, pandoc, matplotlib and Chrome.

- **Numbers.** None is typed by hand. `{{name}}` is a value computed from a file in `data/`, or from
  the fits `tools/lambda.mjs --json` makes of them. `{{table:name}}` is a whole table built the same
  way. `{{readme:…}}` quotes a phrase the README must contain word for word, for the few figures
  only the README records.
- **Failure.** The build stops on a value it cannot fill or one that is not finite.
- **Figures** are drawn from the same data into `report/figures/`.

## Citing

`CITATION.cff` gives the citation (GitHub's **Cite this repository** reads it, in APA and
BibTeX). Every release from 1.6.0 on is archived on Zenodo with a DOI of its own; the concept
DOI [10.5281/zenodo.23212075](https://doi.org/10.5281/zenodo.23212075) always resolves to the latest. `.zenodo.json` holds the
record's metadata, and [docs/zenodo.md](docs/zenodo.md) how the archiving is set up.

## Engine defects found and fixed

Seventeen bugs surfaced while making the site report live data. All seventeen are fixed, and the
fixes are in `src/` and in the committed `stabilizer_qec.wasm`. They are written up in twelve
sections below. Section 6 covers two bugs that had to be fixed together before the XZZX code
worked at all, and section 5 covers two, the second being the discovery that the first fix had
only been applied to a third of the cases it claimed to cover.

### 1. The WASM Monte Carlo was not random

Six `simulate_*` functions seeded from a constant on the non-Python cfg branch
(`Xorshift::new(12345)`, and `54321` in two). Each of those functions is one shot and
`wasm_run_benchmark` loops over them, so every shot in a batch was bit-identical and the reported
rate collapsed to a step function: exactly 0 below a cutoff, a plateau, exactly 1 above.

Fixed with SplitMix64 over a counter, plus a `wasm_seed(lo, hi)` export seeded from
`crypto.getRandomValues`. Seeding each shot from the previous shot's xorshift state is not
enough, because that hands shot N+1 shot N's stream offset by one draw, and batch variance comes
out several times binomial. Verified: twelve repeats at d=5, p=5%, N=4000 give observed σ 0.00224
against binomial 0.00222 (ratio 1.01).

### 2. Phenomenological noise lied in the final round

Defects are time differences, so a lie in the last round has no partner and leaves an unpaired
defect the decoder must match somewhere. More checks means more last-round lies, so logical error
rose with distance far below threshold (1.9% at d=3 to 10.9% at d=7, at p=0.5%). The final round
is now noiseless.

Thresholds at bias `η = 0.5`, `T = d`. The collapse ansatz is the d → ∞ limit and these patches are
small, so the fit carries the leading correction to scaling, `+ D·d^(-ω)`, and fits it over the
widest part of the sweep the scaling form actually describes. That window is chosen by reduced χ²,
not by hand. Figures are the mean and spread of **four independent sweeps**:

| code | noise | p_th | ν | ω | uncorrected fit |
|---|---|---|---|---|---|
| rotated | data | **14.65% ± 0.57** | 1.63 ± 0.05 | 0.94 | 12.2% |
| XZZX | data | **14.36% ± 0.59** | 1.59 ± 0.06 | 1.06 | 12.4% |
| rotated | phenomenological | **3.29% ± 0.14** | 0.95 ± 0.11 | 2.63 | 2.93% |
| XZZX | phenomenological | **3.25% ± 0.07** | 0.97 ± 0.06 | 2.56 | 2.94% |
| rotated | circuit-level | **0.41% ± 0.03** | *not determined* | 4.31 | 0.37% |
| XZZX | circuit-level | **0.42% ± 0.04** | *not determined* | 3.88 | 0.34% |
| rotated | circuit-level, SD6 (per basis) | **0.54% ± 0.01** | 1.11 ± 0.05 † | 1.06 | 0.48% |
| XZZX | circuit-level, SD6 (per basis) | **0.51% ± 0.01** | 1.22 ± 0.05 † | 2.19 | 0.47% |

The two circuit-level models are different physics.

- **The engine's own model** puts an independent error on each qubit of every CNOT, so a gate
  fails about 2p of the time, and it never lets an idle qubit err.
- **SD6** gives each CNOT one two-qubit depolarizing error of total probability p, and
  depolarizes every idle qubit in every layer. It is the standard model, and the one published
  thresholds assume.

The SD6 rows come from the general path (see *Checked against Stim and PyMatching*) and its
probability-weighted sparse matcher, and they are scored the way experiments are, one logical basis
at a time. That barely moves where the curves cross, but it roughly halves the rate below threshold.

- **Stim's own generated circuit crosses higher**, near 0.65–0.7% with PyMatching, because it
  depolarizes idle data only once a round.
- **These sweeps run 40,000 shots a point**, affordable since the sparse matcher replaced the dense
  one, which allowed 800. The same fit at 800 shots gave 0.48% ± 0.05 and 0.53% ± 0.07; the new
  values sit inside those ranges, with the spread five times smaller.
- **† ν is provisional.** At these shot counts the fit reports ν as determined for SD6, for the
  first time at circuit level. But the synthetic-data check in section 9, which established which
  sweeps can recover ν, was run at the old shot counts and has not been repeated at these.
- **The correction term is loosely constrained under SD6.** Across these sweeps ω ranged from 0.5 to
  3.0. In two of the four XZZX sweeps the corrected point estimate sits at or just outside its own
  bootstrap interval, and a single run of the page's own rotated sweep gave 0.51% [0.50–0.54%],
  just below the four sweeps' spread. The uncorrected fit is the steadier number, at 0.47–0.48% for
  both codes. Read the corrected values as good to a few hundredths of a percent, not to the ±0.01
  the four-sweep spread alone suggests.

The runs are in `data/sweeps/`, from `node tools/sweep.mjs 3 0` and `node tools/sweep.mjs 3 1`.

**Confirmed against an independent implementation.** An L×L toric code written from scratch in
JavaScript (periodic lattice, no boundaries, with its own noise, syndrome, graph and scoring, and
sharing no physics with the Rust engine) gives **ν = 0.86** for phenomenological noise, against
the engine's **0.89 ± 0.04**. Two implementations that share nothing but the arithmetic agree on
the exponent.

Getting there required fixing the harness rather than the engine. Its decoder was
nearest-pair-plus-2-opt, which is not minimum-weight, and that was a confound rather than a control:
a weak decoder moves the threshold, and at these sizes drags the fit with it. The harness now uses
the same exact matcher the engine does. That is the one thing deliberately shared, since
minimum-weight matching is a generic graph problem with a single right answer, and the matcher has
been checked against brute force on 1,500 instances. With it, the toric threshold moves from
**1.51% to 3.04%**, close to the engine's 3.36%, and reduced χ² settles near 1. Exact agreement on
the threshold is not expected, since these are different codes. ν is the universal quantity, and it
agrees.

Reproduce with `node tools/toric_exponent.mjs 7000`, which runs both matchers over the same lattices,
rates and seeds and prints both exponents. An earlier revision of this file quoted 1.41 ± 0.28 from
this harness. That number came from an uncommitted script whose settings are lost, and it does not
reproduce: the committed driver gives 0.94 with the approximate matcher and 0.86 with the exact one.
The driver is committed now so that cannot happen again.

An earlier revision of this file also compared the phenomenological ν against 1.46 and called it
low. That was the wrong comparison. 1.46 is the 2D value, which belongs to data noise.
Phenomenological noise is a 2+1-dimensional problem in a different universality class, where the
expected exponent is near 1.0.

**ν is quoted only where it was verified recoverable, and circuit-level is not.** Fed synthetic data
with the exponent fixed at 1.46 in advance, under each model's real sweep conditions, the fit returns
it as 1.46 ± 0.02 from a data-noise sweep, 1.47 ± 0.20 from a phenomenological one, and **0.63 ± 0.36
from a circuit-level one**, a −57% bias at the shot count that model can afford. It also reports a
tight-looking interval while doing so: measured coverage of the true value is 80%, 85% and 46%
respectively. The bootstrap cannot catch this, because the failure is bias and a bootstrap resamples
around its own answer. So the gate is on the statistics the sweep actually carries (roughly 900k,
216k and 43k shots in total), which is what predicts recoverability.

The same check says the phenomenological exponent is real: had the truth been 1.46, the fit would
have returned about 1.47, not 0.95. It really is below the 2D textbook value, and that is a result
rather than an artefact, since the 2D value does not apply to a 2+1-dimensional problem. The
independent toric code above agrees with it.

The spread across sweeps matches the bootstrap interval each sweep reports on its own, which is the
check that the interval means what it says. The two codes agree within it on all three models. That
is the expected answer at `η = 0.5`, which is depolarizing noise; XZZX's advantage is a biased-noise
effect, and it appears once the bias is turned up (see the figure in section 8 of the site).

### 3. The stabilizer tableau replayed identical measurements

`Tableau::new` hardcoded `rng_state: 0xdeadbeef12345678`, so every `StabilizerSimulator` ever
constructed drew the same sequence of random measurement outcomes. Circuit-level runs were entirely
deterministic: a noiseless circuit reported exactly 0% or exactly 100% logical error depending only
on the distance. `Tableau::with_seed` now threads a seed from the shot's own generator. This one
also affected the Python build.

### 4. The two extraction circuits did not commute

X and Z ancillas walked their plaquettes in the same order, and boundary plaquettes compressed
their two CNOTs into the first two time slots. The X and Z stabilizer measurements therefore
interfered where plaquettes overlap, and each disturbed the other. Ancillas now interleave in
opposite orders (the classic N and Z schedules), scheduled by direction rather than by index into
a variable-length neighbour list. A noiseless circuit now never fails at d = 3, 5 and 7, and the
curve rises monotonically with p.

### 5. Logical tomography returned something outside the Bloch sphere

`wasm_estimate_logical_fidelity` ran three identical simulations (for data and phenomenological
noise the three calls differed in nothing but the variable each was assigned to) and reported
`1 - 2*failure_rate` for each as though they were Bloch components. At zero noise that gave
`(1, 1, 1)`, a "state" of length √3.

Under Pauli noise and a Pauli decoder the logical channel is itself a Pauli channel, which shrinks
each Bloch axis independently. The simulators now return which logical Pauli class survived rather
than a bare pass/fail, so the four channel probabilities can be counted and the diagonal of the
Pauli transfer matrix computed properly. Verified: `(1, 1, 1)` at zero noise now correctly means
the channel shrinks nothing, every factor stays in [−1, 1], and the site draws the sphere's image
as an ellipsoid.

**This fix was at first only half applied**, which a later audit of the printed numbers caught: four
of the six code-and-noise pairings still returned a bare pass/fail, and the channel built from them
came back with `r_x` pinned at exactly 1, meaning every failure was counted as the same kind. There
were two causes. The XZZX simulators asked their stabilizer group only whether the residual was
logical, not which one; they now derive a pair of anticommuting logical representatives (the null
space of the commutation map, reduced modulo the stabilizer group) and read the class off by
commutation, so nothing is hard-coded. And the circuit-level simulator asked the tableau, which can
only answer the question its basis poses: prepared in |0_L> it sees a logical X and is blind to a
logical Z, an operator it commutes with. A Pauli frame now shadows the tableau through the same
circuit and yields both.

Two consequences. The derived representatives are checked against the rotated code's own
independently known logicals (a column of X, a row of Z) over 4,000 random residuals at d = 3 and
5. And the circuit-level failure rate roughly doubled, because it now counts logical X and Z where
before it counted only whichever the preparation could see. Data and phenomenological noise had
always counted both, so this makes the three models comparable rather than changing the physics.

### 6. XZZX matched one defect set twice, and checked the wrong logical operator

Two bugs, and both had to go before the code worked at all.

The decoder derived its defects once then matched them twice, on two graphs with different
edge-to-qubit mappings, because the second pass reused the first's defects verbatim
(`let defects_x = defects_z.clone();`). XZZX has one syndrome and two edge families over the same
nodes (an X error flips the stabilizers on one diagonal, a Z error those on the other), so this
explained every defect twice and applied both corrections. A single X error returned the correct
one-qubit X correction plus two invented Z ones, and their count grew with the lattice, so the code
degraded as it grew: 5.4% at d=3 up to 21.0% at d=7, at p=2%. `build_combined_graph` now emits both
families into one graph with a per-edge type tag and a single set of timelike edges; the defect set
is matched once and each chosen edge routes to X or Z by its tag.

That left the rate flat in distance rather than falling, which turned out to be a second, unrelated
bug: the logical-operator check compared the residual against a hard-coded alternating string of
Paulis that is not a logical operator of this lattice. It fired on residuals that were not logical
operators at all, including weight-one residuals, which cannot be logical errors in any code of
distance 3 or more, and it did so at every distance, hence the flat curve. The check now reduces
the residual against a row-reduced basis of the stabilizer group and asks whether anything is left,
which requires no convention about representatives.

Brute-force search over the actual stabilizer group confirms the construction was always sound:
minimum logical weight 3 at d=3, above 4 at d=5, for both codes. The bugs were entirely in the
decoding and scoring.

Verified after both: no single X, Z or Y error causes a logical failure at d = 3, 5 or 7; the rate
falls with distance at every bias (η=64, p=2%: 0.48% → 0.05% → 0.03%); and XZZX beats the rotated
code under bias as the literature says it should. At d=7, p=3%, η=64, data noise, 20,000 shots:
**0.04% against 0.47%**.

### 7. The decoder had no model of the circuit

Circuit-level noise is qualitatively harder than the other two models. A fault on an ancilla partway
through its four CNOTs propagates onto every data qubit it has yet to touch, so one fault becomes a
correlated multi-qubit data error, known as a hook error. The phenomenological spacetime graph has
no edge for that, so the matcher explained it with two unrelated edges and could walk the correction
into a logical operator. Roughly one single fault in thirty did exactly that, and the code got worse
with distance instead of better.

`src/circuit_model.rs` derives the decoding graph from the circuit instead of assuming one. Every
elementary fault is propagated as a Pauli frame (H swaps x and z, CNOT sends `x_target ^= x_control`
and `z_control ^= z_target`, a Z-basis measurement flips exactly when the frame carries x), and the
detectors it fires, together with the data error it leaves, become one edge. Edges may correct
several qubits at once, which is what the old graph could not express.

Two things made it tractable:

- **The circuit is defined once.** `round_program` returns the round as a list of instructions, and
  both the stabilizer simulation and the fault propagation consume that same list. A detector error
  model describing a subtly different circuit from the one being run is worse than none, and two
  hand-written copies would not stay in step.
- **Decomposition is free.** The usual hard part is splitting a fault that fires three or more
  detectors into graph-like pieces. The decoder already matches twice, once per Pauli type, so a
  fault splits into its X part and its Z part and each is graph-like alone. Measured over the whole
  enumeration: no fault fires more than two detectors in either graph.

**Validation is exhaustive, not statistical.** A distance-d code must survive any one fault, so
every fault the circuit admits is propagated, decoded, corrected and checked:
**0 failures out of 600 / 3,240 / 9,408 faults** at d = 3 / 5 / 7, for Union-Find and exact MWPM
alike. That test caught two more bugs:

1. It showed the two CNOT schedules must be transposes the other way round from the pairing first
   tried. The wrong pairing passes cleanly at d = 5 and d = 7 and fails 44 times out of 600 at
   d = 3, so sampling would very likely have missed it.
2. It caught state-preparation noise being applied before the baseline round, where an error sits in
   both readings a detector compares, never fires it, and lands in the residual uncorrectable at any
   distance. It had been showing up as a stubborn p¹ term.

With both fixed, the logical error rate at d = 3 scales as p² as theory demands (ratios 3.63, 3.73
and 3.70 on successive doublings of p from 0.1% to 0.8%, against 4.0 for a clean p², at 200,000
shots per point), and a threshold appears where it should: **0.41% ± 0.03** over four sweeps (table
above). The uncorrected collapse puts it at 0.37%; the gap is the finite-size bias section 9
describes. Of the three models this one is fitted worst, and its exponent is not determined,
because it has the most fault mechanisms per round and the narrowest usable window of p.

### 8. XZZX had no model of its circuit either

The detector error model above is a CSS construction. A rotated plaquette measures either X or Z, so
the two error families light disjoint detectors, the decoder matches twice (once per family), and
that split is exactly what makes the decomposition free. An XZZX plaquette reads `X Z Z X` from a
single ancilla: one syndrome bit, both families firing the same detectors, no split available.

So the XZZX + circuit-level pairing never received any of the three XZZX fixes. It still built two
graphs, matched the same defect set against both, and scored against the hard-coded logical string.
Bugs 6 and 7 were still alive in a path the bench UI let you select. Its logical error rate rose
steeply with distance, from 4.9% at d=3 to 40.6% at d=7 at p=0.2%, where the rotated code at the
same settings falls from 0.23% to 0.03%.

`build_combined` now derives one graph over a single node set, each edge naming the X part and the Z
part of the correction it implies. Only X and Z faults are enumerated, never Y: propagation is
linear over the Pauli frame, so a Y fault is exactly the composition of the two, and letting the
matcher pick both graphlike edges reconstructs it. That is the decomposition the CSS model got for
free, made explicit. The circuit holds every ancilla in |+> and reads an X leg with a CNOT, a Z leg
with a CZ.

**The two sublattices must walk their neighbours in transposed orders.** This is the same lesson
bug 4 taught, and it was no more guessable the second time. With any single order shared by both
sublattices, all 24 permutations leave 10 to 12 of 672 single faults uncorrectable at d=3 while d=5
stays clean. Searching the two sublattices independently over all 576 combinations, scored on
commutation against the tableau and on the exhaustive single-fault check, leaves exactly 6 that
pass, all of them transposes. The search is kept as an ignored test (`xzzx_search_schedules`).

Two properties need separate checks here, because the simulation runs on the Pauli frame rather than
the tableau. The frame is exact for a Clifford circuit under Pauli noise, and it keeps the scoring
sound: an XZZX logical operator is a mixed X/Z string, so there is no row of qubits to measure the
way the rotated code can, and the residual goes to the stabilizer group instead. But a frame
presumes the circuit really is a valid simultaneous measurement; if the schedule made neighbouring
plaquettes disturb each other, the frame would stay self-consistent while the device produced noise.
That property is tested against the tableau: after the first round has projected, every later
noiseless round must reproduce its outcomes exactly, at d = 3, 5 and 7 across seeds.

Verified: **0 failures out of 672 / 3,600 / 10,416 faults** at d = 3 / 5 / 7, Union-Find and exact
MWPM alike; the rate now falls with distance (0.80% / 0.53% / 0.24% at p=0.2%); and the threshold
fits at **0.42% ± 0.04**, indistinguishable from the rotated code's 0.41% ± 0.03. The raw
crossovers agree with each other: both codes go flat around p = 0.33% and are rising by 0.40%,
which sits below the fitted value by the finite-size margin section 9 describes.

The extra noise does show up, but below threshold rather than in the threshold. Every XZZX ancilla
needs an H where the rotated code rotates only its X-type ones (72 noise locations per round at
d = 3 against 64), and at p = 0.26%, d = 7 that costs about 30% in logical error rate: 0.74% against
0.58% over 40,000 shots, which is roughly the size 12% extra locations would predict. An earlier
draft put this at a factor of three; that was the class-counting bug of section 5, which had the
rotated code reporting only half its failures. Threshold is set by where the curves cross, which the
extra locations barely move; the rate below it is not.

**The bias advantage survives the circuit**, which an earlier draft of this file denied. At d=7,
p=0.3%, going from η=1 to η=100 takes XZZX from 1.08% to 0.25% while the rotated code goes the other
way, 1.07% to 2.37%. That is better than nine-fold apart, where at η=1 the two are level (15,000
shots each).

That earlier "honest negative result" was an artifact of the bug in section 5. The rotated
circuit-level simulator counted only the logical error its preparation could see, and under strong
Z-bias the failures it was missing are exactly the ones bias produces, so it looked as though the
rotated code improved to 0.00% while XZZX did not. Counting both classes shows the opposite. A
measurement that cannot see half its outcomes does not just lose precision; it can invert the
conclusion.

### 9. The threshold fit ignored corrections to scaling

Every threshold this project quoted was low, by 15 to 20%, and the reason was in the ansatz rather
than the engine. Finite-size collapse says the curves for every distance fall onto one universal
curve against `x = (p - p_th)·d^(1/ν)`. That is the `d → ∞` statement. The patches here start at
nine qubits, and the approach to the limit is not a rounding error.

Diagnosed on synthetic data with the threshold fixed in advance: the bare collapse comes back **27%
high and stays there** as shots increase, which is the signature of bias rather than noise. Adding
the leading correction `+ D·d^(-ω)` recovers the true value to within 2%, with its error falling
from 40% to 13% as shots go from 1,200 to 20,000. Separating that term from a shift in the threshold
needs a fourth distance, so the sweeps now run to `d = 9`, and to `d = 11` for data noise where a
point costs 0.06s. With three distances the panel says so and reports the uncorrected number
instead.

What makes it checkable without any fitting at all is that the pairwise crossings drift. For
rotated data noise they climb monotonically from 10.5% (d = 3 against 5) to 13.7% (9 against 11) and
are still climbing. So the threshold is above 13.7%, and the uncorrected fit's 12.3% lies below
every crossing involving `d ≥ 5`.

The fit window is chosen the same way, and for the same reason. The scaling form is an expansion
about the threshold and stops describing points far from it, so some of the sweep has to be
excluded. But a fraction picked by hand makes the threshold depend on whoever picked it. One fixed
choice of 75% gave a reduced χ² of 2.3 for data noise and **13.2 for phenomenological**: the same
number fitting one model acceptably and rejecting another outright, with the rejected one's
parameters reported as though they meant something. It is also what drove ω to the end of its range,
since with the shape wrong the correction term is free to absorb the misfit. The fit now takes the
widest window the form actually fits, and reports which. All six sweeps now land at reduced χ²
between 0.8 and 1.9, and ω between 0.9 and 4.3 with none against a boundary.

Two more bugs fell out of verifying the interval rather than the estimate. The bootstrap's local
search used a grid coarser than the spread it was measuring, so every replica landed on the same
cell and the interval collapsed. Against synthetic data with a known threshold it covered the truth
**5% of the time**, which is worse than reporting no interval at all. A two-stage search fixed it;
coverage is now 100% for data noise, 80% for phenomenological and 69% for circuit-level. And ν is no
longer printed for sweeps that cannot determine it, per the check above.

Two side-fixes also came out of this. The variance floor gave zero-failure points 228 times the
weight of a 5% point, so two points out of twenty-seven carried 84% of the fit; that is now
Jeffreys-smoothed. And confidence intervals are a bootstrap over shots rather than the spread of
repeated sweeps, because a fit biased by its own window reproduces that bias on every repeat, so
the old interval was tight and wrong.

An extrapolation of the crossings to infinite `d` was tried as a third estimator and dropped: over 40
synthetic realizations it came out biased −35% with an rms error of 49%, against −2% and 13% for the
corrected fit. Three crossings and three parameters is an exact fit, and the drift exponent runs away
with the intercept. The raw crossings are still displayed, since they are where the curves visibly
cross, but they are not an estimate.

### 10. Derived logical operators overflowed their word from d = 9

Found while extending the sweeps, and latent until then. `find_logical_pair` searches the null space
of the commutation map by packing a Pauli's 2n bits into a `u128`. At `d = 9` that is 162 bits. The
shift silently wrapped, and the "logical operators" it returned **anticommuted with the stabilizers
they were supposed to commute with**, while the logical error rates they produced looked entirely
plausible, falling with distance exactly as they should.

The row is now split along the seam the Pauli already has, one word for the X half and one for the Z
half, so every shift stays inside a word. Checked directly at `d = 3, 5, 7, 9, 11`: both
representatives commute with every stabilizer, anticommute with each other, and are not products of
stabilizers.

### 11. The Union-Find decoder was not a function of its inputs

Called twice with byte-identical arguments it returned 35 edges, then 37. The active cluster roots
were collected into a `HashSet`, and Rust seeds each `HashSet`'s hasher randomly, so iterating one
visits its elements in a different order every time. That order decides which cluster grows first and
therefore which correction comes out. Collected in node order instead.

Monte Carlo averages wash this out, so it did not bias any threshold, but it made every individual
result unreproducible, in the default decoder.

### 12. "Exact MWPM" silently ran the greedy decoder instead

Two undisclosed limits: above sixteen defects it called `decode_greedy` and returned, and above
50,000 backtracking steps it did the same. At `d = 9` phenomenological both are exceeded on nearly
every shot, so asking for the best of the three decoders quietly gave you the worst.

It is visible once looked for. Exact MWPM reported a phenomenological threshold near **0.75% against
Union-Find's 3.3%**, and an exact decoder cannot be beaten by an approximate one. At `d = 9, p = 2%`
its logical error rate was 16.25% against greedy's 16.78%, i.e. the same decoder.

The first fix replaced the fallbacks with a real starting matching and a branch-and-bound search
that could be cut off without collapsing, but it kept a cap of its own, at twenty defects, above
which the result was an approximation. "Exact" was still not quite true where the problem was hard.

**It is now exact at every defect count.** `src/blossom.rs` implements Edmonds' blossom algorithm,
which handles the odd cycles that make general-graph matching hard by contracting them, searching the
contracted graph, and expanding them again. The boundary is modelled as `m` interchangeable copies:
pairing a defect with any copy costs its distance to the boundary, and copies pair with each other
for nothing. That turns "match these defects, and any may instead run to the boundary" into a plain
perfect matching with every weight finite.

Measured against Union-Find over 8,000 shots at each of 18 code × noise × distance combinations,
**MWPM is now lighter in every one**, by 2.8σ to 10σ. Before the cap was removed it lost outright on
XZZX circuit-level.

Cost: 0.8 ms per solve at sixty vertices, 4.5 ms at a hundred and twenty.

**This retracts an earlier claim.** A previous commit found MWPM scoring worse than Union-Find on
XZZX circuit-level by 5.7σ, and attributed it to matching being unable to express the correlation
between the two edges a Y fault needs. The bias-dependence seemed to confirm it. The real cause was
the fallback: strong bias means fewer errors, fewer defects, and so a genuine MWPM rather than a
silent greedy. With the decoder fixed, MWPM is better than Union-Find at η = 1, 10 and 1000 alike.
The earlier explanation was plausible and consistent with the evidence, but wrong.

### Located loss under circuit-level noise

An erasure is a qubit the hardware knows it lost. The Pauli is uniform and unknown, but the location
is known, and that makes it far easier to correct: every decoding edge the location could have
produced costs nothing, so the matcher routes through it freely.

Under circuit-level noise the location is a point in the circuit rather than a qubit, and an erasure
on an ancilla partway through its CNOTs frees everything the loss goes on to touch, by the same
propagation that produces a hook error. The detector error model already records which edges each
circuit location produces, so it answers that question directly: `DetectorGraph::site_edges` maps an
erasure site to the edges it makes free.

Measured at d = 5, p = 0.8%: logical error falls from 17.9% with no erasure, to 5.6% when half the
faults are located, to 0.02% when all of them are. It is not quite zero because p = 0.8% still sits
below the located-noise threshold rather than nowhere near it. Fully located noise has its own
threshold near p = 2%, roughly four times the Pauli threshold. The fact that it has a threshold at
all, rather than being perfect everywhere, is the check that the information is being used rather
than assumed.

## Repository Structure

```
src/tableau.rs        symplectic tableau and Clifford operations
src/simulator.rs      a stabilizer simulator on the tableau
src/surface_code.rs   rotated and XZZX lattices, noise models, error generation
src/decoder.rs        Union-Find cluster growth and peeling
src/blossom.rs        exact minimum-weight perfect matching (Edmonds' blossom), with hard ceilings
src/circuit_model.rs  detector error model derived from the extraction circuit, CSS and non-CSS
src/circuit.rs        circuits in Stim's text format: parse, emit, flatten, resolve records
src/gates.rs          Stim's other Clifford gates as exact H/S/CX sequences (tools/gen_gates.py)
src/dem.rs            detector error models: built backwards from any circuit, decomposed as
                      Stim decomposes, read and written in Stim's .dem format, compared
src/dem_decoder.rs    probability-weighted exact matching over any detector error model
src/frame_sampler.rs  Pauli-frame sampling of any circuit, independent of the model
src/batch_sampler.rs  the same, 64 shots per word, REPEAT without flattening
src/shots.rs          Stim's 01 and b8 detection-event formats
src/memory.rs         rotated and XZZX memory experiments as circuits, engine noise and SD6
src/equivalence.rs    test: the old per-code circuit and the general one are the same circuit
src/fixtures.rs       circuit texts shared by several modules' tests
src/sparse/           sparse blossom: exact matching by growing regions on the detector graph,
                      and correlated matching's two passes on top of it
src/m2d.rs            raw measurements and sweep bits to detection events, by noiseless tableau runs
src/diagram.rs        timelines, detector slices and matching graphs, as text and SVG
src/generated.rs      Stim's generated memory circuits, written as Stim writes them
src/css.rs            CSS codes from their checks: hypergraph products, colour codes, a memory for any
src/bp.rs             belief propagation on a Tanner graph, reproducing ldpc's arithmetic
src/belief.rs         belief-matching, as the authors' beliefmatching package does it
src/gf2.rs            linear algebra over GF(2): rank, kernel, inverse, on bit-packed rows
src/bb.rs             bivariate bicycle codes (the gross code), their logicals, the depth-8 memory
src/bb_auto.rs        their automorphisms, and each one's action on the logical qubits, exactly
src/bb_gauge.rs       the gauging ancilla system that measures a logical operator, minimal or expanded
src/bb_circuit.rs     circuits on them beyond the memory: any checks on any schedule, and on it
                      the logical measurement
src/osd.rs            BP+OSD: OSD-0, OSD-E and OSD-CS on BP's posteriors
src/lsd.rs            BP+LSD: clusters grown by BP's posteriors and solved alone, as ldpc does
src/plu.rs            dense GF(2) elimination column by column, ldpc's, for LSD's clusters
src/robin.rs          a set iterating as tsl::robin_set does, for LSD's order
src/relay.rs          Relay-BP: memory BP in legs, as IBM's relay_bp computes it
src/chacha.rs         rand 0.8's StdRng (ChaCha12) and Uniform<f64>, for Relay-BP's strengths
src/color/            colour-code matching: Chromobius's Möbius model, tables and lifting
src/search.rs         the search decoder: Tesseract's A*, with libc++'s and libstdc++'s heaps
src/surgery.rs        lattice surgery: patches on a grid of tiles merged and split by a program of
                      steps, compiled to one circuit; Z⊗Z and X⊗X, the logical CNOT, lines of patches
src/window.rs         window decoders: models cut by time, sliding and parallel schedules
src/stream.rs         streams too long to model, decoded window by window from a template
src/api/              the crate's stable Rust API, re-exported at its root
src/batch.rs          the loops over shots that the Python bindings and the Rust API share
src/wasm_session.rs   the WebAssembly session behind the explainer's first sections
src/py_objects.rs     the compiled objects the public Python API wraps
src/py_api.rs         the extension's own functions (stabilizer_qec._core), which the tools use
src/parallel.rs       work over shots, one contiguous range per thread
src/fuzzing.rs        what the fuzzers drive: any text through every stage that reads it
src/wasm_xc.rs        WASM exports for Figure 8 and SD6
src/wasm_hw.rs        WASM exports for Figure 10: raw readouts to predictions
src/wasm_rt.rs        WASM exports for Figure 12: streams window-decoded and globally decoded
src/wasm_bb.rs        WASM exports for Figure 15: the [[72, 12, 6]] memory decoded by BP+OSD
src/wasm_ls.rs        WASM exports for Figure 22: lattice-surgery programs sampled and matched
src/lib.rs            PyO3 module and the WASM C-ABI interface
examples/matcher_profile.rs  the sparse matcher run natively, for a profiler or a timing table

index.html            the explainer (structure only)
css/styles.css        design tokens, then base, then components
js/main.js            wiring: the main-thread engine for the lattice figures, the pool for the rest
js/engine.js          typed wrapper over the WASM exports
js/dom.js             small DOM helpers, and running a figure's work when it scrolls near
js/nav.js             the contents rail and its scroll-spy
js/opener.js          the live patch under the hero; js/opener-math.js its pure helpers
js/ambient.js         a slow heartbeat for read-only figures, while on screen
js/patterns.js        error patterns the interactive figures click in
js/channel.js         the noise channel, shared by the figures and the engine
js/channel-view.js    the logical channel drawn as the Bloch sphere's image
js/worker.js          Monte Carlo worker (own engine instance)
js/pool.js            a pool of workers, one per core but one; js/pool-merge.js combines their parts
js/stream.js          chunking for a streamed Monte Carlo run
js/meter.js           a strip chart of a streamed run's estimate settling
js/compute.js         worker RPC, Wilson intervals, threshold collapse fit
js/lattice.js         canvas renderer for a code patch, 2D and spacetime
js/plot.js            canvas plotting primitive
js/sections/*.js      one module per section of the page
js/sweep-config.js    what the threshold sweep measures, shared with tools/sweep.mjs
js/xcheck-format.js   pure formatting for Figure 8
js/lambda-fit.js      logical error per cycle and Λ, for the README and section 11
js/hardware-format.js names and formats for section 11
js/realtime-format.js names and formats for section 12
js/bb-geometry.js     the gross code's torus, for section 13's drawing
js/surgery-geometry.js lattice surgery's layout, for section 14's drawing
js/estimator.js       the resource estimate's model and its inputs, for section 15

tools/xcheck.py         the cross-check against Stim and PyMatching; writes data/xcheck/
tools/matcher_bench.py  the matcher's speed against PyMatching, and a fingerprint of its decisions
tools/sweep.mjs         repeated threshold sweeps in Node, for the figures quoted here
tools/independent_toric.mjs  an independent toric-code simulator in plain JS, sharing no engine code
tools/toric_exponent.mjs     the critical exponent on it, with an approximate and an exact matcher
tools/google.py         Google's Willow and Sycamore data: check, decode, summarise, extract
tools/lambda.mjs        the fits for every decoder, printed; the README's tables come from it
tools/realtime.py       window decoders' accuracy, latency, and the million-round stream
tools/bp_check.py       BP and belief-matching against ldpc and beliefmatching, exactly
tools/belief.py         belief-matching on all of Sycamore and Willow; BP iterations
tools/bb_check.py       the bivariate bicycle codes and BP+OSD against Bravyi et al., Stim and ldpc
tools/decoder_check.py  BP+LSD, Relay-BP, colour-code matching and the search decoder against
                        ldpc, relay_bp, Chromobius and Tesseract, shot for shot
tools/gross.py          the gross code's logical error per cycle, and the surface code beside it
tools/gross_ops.py      the gross code's automorphisms, gauging distances and logical measurement
tools/surgery.py        lattice surgery against Stim and PyMatching, its timing law, and programs
tools/estimate_noise.py the surface code's error per round under uniform noise, fitted
tools/distill.py        the 15-to-1 distillation law, measured
tools/estimate.mjs      the resource estimate from the committed data
tools/readme_tables.py  the README's tables, generated from the data (--check in CI)
data/xcheck/            Stim's circuits and models, the recorded reference, and the run's report
data/matcher/           the matcher's timings and fingerprints
data/sweeps/            the SD6 threshold sweeps
data/belief/            belief-matching's results and the oracle check
data/gross/             the gross code's results and checks, automorphisms, gauging, measurement
data/decoders/          the 1.8 decoders' check against their authors' packages
data/surgery/           lattice surgery's golden circuit, timing law and programs
data/estimate/          the estimate's noise fit, distillation law and sources
data/realtime/          latency, accuracy and million-round results
data/google-results/    per-experiment checks and failure counts, and the fits (lambda.txt)
data/willow-extract/    2,000 raw shots at each of d = 3, 5, 7, for Figure 10 (CC BY 4.0, Google)

tools/smoke.py          an installed wheel checked end to end against Stim and PyMatching
tools/wasm-smoke.mjs    the WASM engine driven in Node through the page's wrappers
tools/site-tests.mjs    Node tests for the site's pure modules
tools/contrast.mjs      WCAG contrast for every text-on-wash pairing the page draws
tools/report.py         the technical report: values, tables and figures from data/, then HTML and PDF
report/                 the report's source, template, figures, and the built HTML and PDF
tools/notebooks.py      the tutorials: each .py run and built into its notebook, or checked
tutorials/              three notebooks, each built from the Python file beside it
tools/bench.py          the benchmarks against Stim and PyMatching, recorded and checked
benchmarks/             the benchmarks page; data/bench/runs.json the recorded runs
pyproject.toml          the Python package (maturin; one abi3 wheel); python/README.md its PyPI page
python/stabilizer_qec/  the Python package: the public API over the extension, _core, and its stub
tests/                  the package's test suite (pytest, Hypothesis), and every gate against Stim
api/                    the Python guide and reference, built by tools/api_docs.py (pdoc)
docs/crate.md           the Rust crate's README
fuzz/                   cargo-fuzz targets and their seed inputs
CHANGELOG.md            what changed in each release
.github/workflows/      CI on every push; wheels and PyPI publishing on a tag

run_benchmarks.py       phenomenological threshold benchmarks
run_data_benchmarks.py  data-noise threshold benchmarks
```

## Building & Running

### The website

It is a static site with no build step, but it does need to be served over HTTP, since ES modules
and the `.wasm` fetch both fail from `file://`.

```bash
python3 -m http.server 8080
```

Then open `http://localhost:8080`.

The site's pure modules have Node tests, and the palette has a contrast check:

```bash
node tools/site-tests.mjs
node tools/contrast.mjs
```

### Rust library

```bash
cargo build --release
```

### Rebuilding the WASM module

```bash
cargo build --release --target wasm32-unknown-unknown --no-default-features
```

Copy the resulting `target/wasm32-unknown-unknown/release/stabilizer_qec.wasm` to the repository
root. The committed `.wasm` is built this way and includes the fixes above.

Note for Apple Silicon: if `cargo` reports `bad CPU type in executable`, the toolchain is the
x86_64 build and Rosetta is not available to the shell. `softwareupdate --install-rosetta` fixes
it; installing a native `aarch64-apple-darwin` toolchain is the better long-term answer.

### Python bindings & benchmarks

```bash
python3 -m venv .venv && .venv/bin/pip install stim pymatching numpy maturin
VIRTUAL_ENV=$PWD/.venv CARGO_TARGET_DIR=target-py .venv/bin/maturin develop --profile python
.venv/bin/python run_benchmarks.py
.venv/bin/python run_data_benchmarks.py
.venv/bin/python tools/xcheck.py
```

If an older `stabilizer_qec.so` sits at the repository root (git ignores it), delete it. Python looks
in a script's own directory first, so for the two benchmark scripts it would shadow the fresh build.
`tools/xcheck.py` refuses to run against a build that lacks the cross-check bindings.

## License

MIT

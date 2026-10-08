---
title: "stabilizer-qec: a quantum error-correction simulator and decoder checked against the field's reference tools"
tags:
  - Rust
  - Python
  - WebAssembly
  - quantum error correction
  - surface code
  - decoding
  - stabilizer simulation
authors:
  - name: Jasper Sands
    affiliation: 1
affiliations:
  - name: Independent researcher
    index: 1
date: 7 October 2026
bibliography: paper.bib
---

# Summary

A quantum error-correcting code protects logical qubits by measuring checks on many physical
qubits, round after round, and handing the outcomes to a decoder that infers which errors
occurred. Studying a code means three steps: simulating its circuits under noise, turning each
circuit into a detector error model (the list of independent faults and the checks each one
flips), and decoding the sampled syndromes. `stabilizer-qec` does all three in one Rust engine.
The same engine is a Rust crate, a typed Python package with one abi3 wheel for every CPython
from 3.9 on, and a WebAssembly module that drives an interactive explainer in the browser
(<https://qcompiler.jaspersands.com>).

It reads and writes Stim's circuit and error-model formats [@gidney2021stim], so circuits move
freely between the two. It provides a bit-parallel Pauli-frame sampler; an error-model builder
that walks a circuit backwards; exact minimum-weight perfect matching by sparse blossom
[@higgott2025sparse], plain and correlated; belief propagation, BP+OSD
[@roffe2020decoding; @panteleev2021degenerate] and belief-matching [@higgott2023improved];
union-find [@delfosse2021almost]; and sliding and parallel window decoders for real-time decoding
[@skoric2023parallel; @tan2022scalable]. Beyond the rotated and XZZX surface codes
[@bonilla2021xzzx], it builds IBM's bivariate bicycle codes [@bravyi2024high], any CSS code from
its checks (hypergraph products [@tillich2014quantum] and colour codes among them), and lattice
surgery between surface-code patches [@horsman2012surface] compiled into whole-program circuits.
It also explains errors: each fault in an error model is traced back to the gates that cause
it, and a circuit's distance is found by Stim's two searches or proven exactly by integer
programming.

# Statement of need

Two tools define how the field simulates and decodes stabilizer codes: Stim for circuits, error
models and sampling, and PyMatching for matching [@higgott2022pymatching; @higgott2025sparse].
Belief-propagation decoders live in a third package, `ldpc` [@roffe2020decoding], and newer
decoders are spread across further packages, each with its own conventions. A researcher who
wants to test a new code, noise model or decoder must glue these together. When the glue is
wrong (a mis-decomposed fault, an edge weight from the wrong probability), the logical error
rate is quietly wrong, and nothing flags it.

`stabilizer-qec` is for researchers and students who want that whole pipeline in one place,
with every stage checked against the reference implementation of that stage. It is also for
readers who learn by running code: the website's figures execute the same engine in the
reader's browser, and three tutorials take a circuit through to a threshold plot, real-time
decoding, and codes beyond the surface code.

# State of the field

Stim [@gidney2021stim] is the standard stabilizer circuit simulator, and its error-model
builder is the reference this project matches. PyMatching 2 [@higgott2025sparse] is the
standard matching decoder; `ldpc` provides BP and BP+OSD; `beliefmatching`
[@higgott2023improved] combines them; Chromobius [@gidney2023new] and Tesseract
[@beni2025tesseract] decode colour codes and general error models. These are excellent tools,
and `stabilizer-qec` does not claim to replace them. Its contribution is different: one engine
that implements each stage independently, so that each stage can be checked against the
reference tool for that stage. The same engine also runs where they do not. A browser explainer
and a real-time decoding study both needed a sampler, an error-model builder and a decoder in
the same process, compiled to WebAssembly, with no Python available. Building that engine was
therefore needed; it was not a choice made instead of contributing to Stim or PyMatching. Its
agreement with those tools is itself a result: two independent implementations that agree
fault for fault and shot for shot are evidence that both are right.

# Software design

The engine is one Rust crate with no required dependencies. Thin bindings expose it to Python
(PyO3) and to the browser (a plain WebAssembly ABI, no build step for the site). Three choices
shape it.

*Reference tools as oracles.* Every stage has a test that compares it with an independent
implementation, not just a test of its own outputs. Error models must equal Stim's fault for
fault, decomposed and not. On 15 circuits the largest relative probability difference is
$1.1\times10^{-15}$. They are also checked on random circuits that use all 49 of Stim's one-
and two-qubit gates. Matchings must tie PyMatching's: there were 13 disagreements in 600,000
shots, and every one was a tie between equal-weight corrections. BP posteriors equal `ldpc`'s bit
for bit, and BP+OSD corrections equal `ldpc`'s except where tied corrections are chosen
differently. Explained errors and the text timeline diagram match Stim's output character for
character. The command line is byte-identical to Stim's wherever Stim's output is
deterministic.

*One engine, three targets.* Sampling, model building and decoding run in the same process, so
the browser, Python and Rust users all get the same results from a seed, on any machine and
any number of threads. The sampler packs 64 shots into a machine word and runs `REPEAT` blocks
without flattening them, taking 0.89 µs a shot at d = 7 against Stim's 1.44 µs on the recording
machine (Apple silicon). There, sparse blossom runs at 0.89 to 0.99 times PyMatching's
single-threaded time, and correlated matching at 0.75 to 0.94 times PyMatching's; on Linux x86_64
both take about 1.2 times. Error models are built in about half of Stim's time.

*Real data, not only simulation.* The engine rebuilds the detection events of every published
Willow and Sycamore surface-code experiment from Google's raw measurements, bit for bit (550
experiments, 27.5 million shots) [@google2023suppressing; @google2025quantum], and decodes them.
That anchors its decoders to hardware rather than only to its own noise model.

Correctness is enforced beyond the comparisons. A randomized run of 20,000 generated and
mutated circuits and models drives every parser, builder, sampler and decoder; inputs may be
refused, but nothing may panic. Fuzzing covers the parsers. The Python test suite runs on
Python 3.9 to 3.14 on Linux, macOS and Windows. Benchmarks against Stim and PyMatching run in
continuous integration, which fails a change that doubles a speed ratio.

# Research impact statement

The package is published on PyPI and crates.io under the MIT licence. It plugs into sinter's
threshold sweeps as a set of custom decoders. Its results are reproducible: a technical report
in the repository fills every number from committed data, which the scripts in `tools/`
regenerate. The report contains findings of its own:

- *Google's hardware.* On Willow's data, with Google's SI1000 prior, the correlated matcher
  reaches Λ = 1.95; Google's own correlated matcher on the same prior reaches 1.91. Belief-matching
  makes Sycamore's d = 5 beat d = 3, as Google's decoder does.
- *Real-time decoding.* Parallel windows decode Willow's d = 5 syndromes within its 1.1 µs
  cycle on four cores.
- *The gross code.* Decoded by BP+OSD, it keeps 12 logical qubits failing 2.6 to 9.3 times less
  often than twelve d = 11 surface-code patches.
- *Logical measurement.* A gauging measurement of one of its logical operators
  [@cross2024improved; @williamson2024gauging], built from the definition, loses distance while
  merged until four added edges restore it.
- *Circuit distance.* Stim's generated colour-code memory has circuit distance 2, 3 and 4 at
  code distance 3, 5 and 7. This is proven by integer programming and agrees with Stim's own
  search.
- *Resource estimates.* An estimate built on the engine's measured noise and decoding latency,
  with Litinski's floor plans and magic-state factories [@litinski2019game], comes within 1.6
  times the qubit count of Gidney's RSA-2048 estimate [@gidney2025factor] on his assumptions.
  It is also checked against Lee et al.'s FeMoco estimate [@lee2021even].

These are benchmarks and reproducible materials rather than citations. The software is new, and
this paper is its first publication.

# AI usage disclosure

Generative AI was used extensively. Most of the source code, tests, documentation and the first
draft of this paper were written with Anthropic's Claude models, through Claude Code. The author
directed the work, chose its scope, reviewed the results, and is responsible for the content.
Because generated code can be plausible and wrong, the project's verification does not rely on
the code reviewing itself. Each stage is compared with an independent reference implementation
(Stim, PyMatching, `ldpc`, `beliefmatching`) or with Google's published hardware data, as
described above. The figures in this paper all come from those comparisons.

# Acknowledgements

The author thanks the authors of Stim, PyMatching, `ldpc` and `beliefmatching`, whose tools
serve as this project's references, and Google Quantum AI for publishing the raw data of its
surface-code experiments.

# References

# Changelog

All notable changes to `stabilizer-qec`. The Python package follows
[semantic versioning](https://semver.org) from 1.0. Anything deprecated warns for at least one
minor release before a major release removes it. The Rust crate is versioned on its own.

## 1.9.0 — unreleased

The Python package and the Rust crate both at 1.9.0, "beyond Pauli noise". The headline is
something no other tool does: sampling and decoding any Stim circuit under coherent errors at
circuit level, at sizes a state vector cannot reach, checked exactly against a state vector
where it can. API additions only.

### Added

- **Non-Pauli operations as Stim-readable tags** on identities, which Stim and every Clifford
  engine here read as identities:
  - `I_ERROR[R_X|R_Y|R_Z(theta=…)]`, `II_ERROR[R_XX|R_YY|R_ZZ(theta=…)]`,
    `I_ERROR[R_PAULI(theta=…, pauli=…)]` (exp(−iθP));
  - `I[T]`, `I[T_DAG]`, `I[U3(theta=…, phi=…, lambda=…)]`;
  - `I_ERROR[AMPLITUDE_DAMPING(gamma=…)]`;
  - `I_ERROR[LEAK(p=…)]`, `I_ERROR[SEEP(p=…)]`, `II_ERROR[LEAK_TRANSPORT(p=…)]`.

  A malformed known tag is an error naming its line; an unknown tag stays inert.
- **The coherent sampler** (`Circuit.compile_coherent_sampler`, `CoherentSampler`; Rust
  `Circuit::coherent_sampler`, `CoherentOptions`, `WeightedSamples`). It gives shots of the
  Pauli-twirled circuit, each weighted by its class's coherent over twirled probability.
  Weighted, the shots are distributed as the coherent circuit's.
  - The kernel of indistinguishable fault sets counts random measurement outcomes as gauges.
  - Equivalent rotations are drawn as one.
  - The rest of the kernel is found locally.
  - Exact on every branch of 150 random small circuits. Within error bars of the state
    vector's exact answers, where the twirl is 5–10× off.
- **The state vector** (`Circuit.compile_exact_sampler`, `ExactSampler`,
  `Circuit.exact_distribution`; Rust `Circuit::exact_sampler`): up to 24 qubits in use, every
  instruction plus the tags above, noise as trajectories or followed branch by branch.
  Checked against a dense density-matrix oracle.
- **`Circuit.twirled(merge=False)`** (Rust `Circuit::twirled`): the twirl as Pauli channels,
  or with `merge` the coherence-aware model, which adds equivalent rotations' angles before
  twirling. Decoding on it fails less often on coherent shots.
- **`weighted_rate`, `weighted_logical_error_rate`, `effective_sample_size`** for weighted
  shots.
- **Leakage** in a frame sampler (`Circuit.compile_leakage_sampler`, `LeakageSampler`). It
  reports a herald per measurement and uses Google's partner-depolarising model. The same
  model in the state vector agrees with it.
- `tools/coherent.py`, `data/coherent/`, site section 16 "Beyond Pauli noise" (Figures
  26–29), and tutorial 05.

## 1.8.0 — 2026-10-08

The Python package and the Rust crate both at 1.8.0, "more decoders": four decoders the field
compares against, each ported from its authors' code and held to it shot for shot
(`tools/decoder_check.py`, run in CI). API additions only.

### Added

- **BP+LSD** (`BpLsd` on a model, `BpLsdDecoder` on a check matrix with `ldpc`'s arguments;
  Rust `BpLsd`, `BpLsdDecoder`, `LsdOptions`): belief propagation, then localized statistics
  decoding (Hillmann et al., 2024), LSD-0, LSD-E and LSD-CS. Corrections equal to `ldpc`
  2.4.1's on every shot checked; the only differences possible are where `ldpc`'s own order
  is not reproducible (equal keys in a long `std::sort`, or several clusters merged at once,
  which it orders by hashed pointers), and those shots are reported as ties. Each cluster's
  faults iterate in `ldpc`'s `tsl::robin_set` order, which is reproduced. 0.6 of `ldpc`'s time
  on the gross code.
- **Relay-BP** (`RelayBp`, `RelayBpDecoder`; Rust `RelayBp`, `RelayBpDecoder`,
  `RelayOptions`): min-sum BP with memory, run in legs of disordered memory strengths (Müller
  et al., IBM, 2025). Identical to IBM's `relay_bp` 0.2.2 on every shot (correction,
  convergence and iteration count), for explicit strengths and for a seed: its random
  strengths come from `rand` 0.8's generator, reproduced without the dependency. A model's
  shot k draws from the seed and k, so results do not depend on the thread count.
- **Colour-code matching** (`ColorMatching`; Rust `ColorMatching`): Chromobius's Möbius
  construction (Gidney and Jones, 2023) on this package's matcher. The Möbius matching weighs
  the same as Chromobius 1.1.1's on every shot checked, and predictions differ only between
  equally light matchings. 0.74 of Chromobius's time at d = 7.
- **A search decoder** (`SearchDecoder`; Rust `SearchDecoder`, `SearchOptions`,
  `DetectorOrder`): Tesseract's A* for the most likely error (Beni, Higgott and Shutty,
  Google, 2025), for any model. The faults it finds are Tesseract's, fault for fault, on the
  same platform (its ties are broken as the platform's C++ library breaks them). With no
  beam, no queue bound and revisits allowed it is exact.
- **Colour annotations**: `CssCode.memory_circuit(..., annotate_colors=True)` (Rust
  `memory_circuit_with_colors`) gives each detector Chromobius's 4th coordinate;
  `Circuit.generated("color_code:memory_xyz", ..., annotate_colors=True)` (Rust
  `generated_with_colors`) flattens Stim's colour code and annotates it as Chromobius's
  authors do.
- The four in `stabilizer_qec.sinter` (`"bplsd"`, `"relay_bp"`, `"color_matching"`,
  `"search"`) and in `stabilizer-qec decode --decoder`; fuzzed with Hypothesis and
  fingerprinted; benchmarked against their references; tutorial 04 compares every decoder on a
  colour code and the gross code.

## 1.7.0 — 2026-10-08

The Python package and the Rust crate both at 1.7.0, "faster": error models in 0.4 to 0.6 of
Stim's time, measurement conversion in 0.1 to 0.25, the graph-like distance in 0.1 to 0.2 and
now as Stim's own answer, and matching at or below PyMatching's time on Apple silicon (about 1.1
times on Linux arm64 and 1.2 on x86_64), with every result unchanged and fingerprinted on every
platform. Benchmarks run on Linux arm64 as well as x86_64, and at 4 threads and every core.

### Changed

- `shortest_graphlike_error`, on circuits and on models, now runs Stim's own search (one
  breadth-first search over pairs of detection events from every edge that flips an
  observable) and so returns Stim's own faults, in Stim's order: its model text, and its
  explained errors with and without `canonicalize_circuit_errors`, equal Stim's character for
  character (checked on Stim's generated codes, 300 random models and 100 random circuits;
  1.6 matched Stim's length only). It takes 0.11 to 0.12 of Stim's time from d = 11 to 19,
  where 1.6 took 0.71 to 1.82.
- Error models are built in 0.6 of Stim's time on Apple silicon (1.6: 2.0): the walk no longer
  allocates for every combination of a noise channel's errors, and the pass that looks ahead
  for a loop's period skips noise, which it never collects. The models are unchanged (their
  text is fingerprinted on each platform).
- Measurements are converted to detection events 64 shots at a time, as Stim converts them
  (each detector the XOR of its records' words, rows moved in and out by 64×64 bit
  transposes): 0.24 of Stim's time on Apple silicon (1.6: 1.26), the output unchanged.
- Matching is 14% faster plain and 12% correlated, every decision unchanged (the matcher's
  fingerprints): a reminder for a node with nothing ahead is passed over without a scan while
  nothing that could change that has happened, no scan at all when no region grows, the scan
  reads one array with no selects, and reminders compare as integers. Against PyMatching,
  plain matching now takes 0.98 of its time on Apple silicon, 1.08 on Linux arm64 and about
  1.2 on Linux x86_64 (whose spare registers the loop exhausts); correlated matching 0.91 to
  0.98, 0.85 and about 1.2.
- `tools/bench.py` adds sampling and decoding at 4 threads and on every core (no reference:
  Stim's sampler and PyMatching's batch decoding are single-threaded), and records the core
  count.

### Added

- `tools/fingerprints.py` and `tests/test_fingerprints.py`: a hash of 152 results (error models,
  shots for a seed at any thread count, raw measurements, model samples, m2d, every decoder,
  the distance searches), checked on Linux, macOS and Windows, so work on speed cannot change
  a result unnoticed. `examples/hot.rs` runs each benchmark's job natively for a profiler;
  benchmarks run on Linux arm64 as well as x86_64 (`.github/workflows/bench.yml`).

## 1.6.0 — 2026-10-07

The Python package and the Rust crate both at 1.6.0, "know your circuit": where each fault of
an error model comes from and how few faults defeat a circuit, both as Stim reports them, and
the exact circuit distance by integer programming; Stim's result formats, samplers and command
line; Stim's text timeline and time-slice diagrams; and a paper for the Journal of Open Source
Software. The first release archived on Zenodo.

### Added

- The text timeline (`diagram("timeline-text")`, and the command line's) is now Stim's,
  character for character: Stim's moments, labels and boxes, loops between `|` columns headed
  `/REP n` with records, detectors and coordinates in terms of `iter`, detectors drawn at their
  qubit's coordinates. Checked on hand-made circuits pinning each rule, Stim's generated
  memories and 200 random circuits using every gate.
- `diagram("timeslice-svg")` and `diagram("detslice-with-ops-svg")`, and `tick` as a `range`
  (with `rows`) for these and `detslice-svg`: a panel per tick, each the qubits at their
  coordinates with that tick's operations and/or the detector slice after them (Rust:
  `DiagramKind::TimeSliceSvg`, `DetectorSliceWithOpsSvg`, `DetectorSlicesSvg`).
- `Circuit.explain_detector_error_model_errors(*, dem_filter=None,
  reduce_to_one_representative_error=False)` (Rust: `Circuit::explain_errors`): where each
  fault of a circuit's error model comes from: the instruction (numbered as Stim numbers
  them, adjacent identical instructions joined), its targets, the Pauli product or the
  measurement it flips, the `TICK`s before it and the loop passes around it. The results
  (`ExplainedError`, `CircuitErrorLocation` and their parts, Stim's names) print as Stim's
  character for character, each part on its own as well (Rust: `Display` for
  `TargetWithCoords` and `CircuitErrorLocation`, and `CircuitErrorLocation::instruction_text`),
  for every location and for the representative Stim picks, on
  Stim's generated memories and on 1,000 random circuits using every gate, noise channel and
  tag.
- `shortest_graphlike_error` on circuits and on detector error models (Rust:
  `Circuit::shortest_graphlike_error`, `DetectorErrorModel::shortest_graphlike_error`): the
  graph-like circuit distance, the fewest faults of at most two detectors that together flip
  an observable and no detector, found as Stim finds it (decomposed faults left out unless
  `ignore_ungraphlike_errors=False`). Its length equals Stim's on Stim's generated surface,
  repetition and colour codes up to d = 15. On a model it takes Stim's time at d = 11 (3.5 ms)
  and 2.5 times it at d = 15; `tools/bench.py` holds it to Stim's.
- `search_for_undetectable_logical_errors` on circuits and models: Stim's breadth-first
  search through hyperedges, with its three limits, giving the same number of faults as
  Stim's on its colour codes, surface codes, the Steane code and the [[72, 12, 6]] code.
- `DetectorErrorModel.distance()`: the exact distance of a model by integer programming (scipy's
  HiGHS), hyperedges included, proven optimal or refused; `method="graphlike"` or `"search"`
  for the two searches. It finds that Stim's generated XYZ colour-code memory has circuit
  distance 2 at d = 3 and 3 at d = 5 (two or three correlated two-qubit faults; Stim's own
  search agrees given room), and that the generic `CssCode` schedule loses the colour code's
  distance likewise, while the surface codes and the Hamming hypergraph product keep theirs.
  `tools/distances.py` measures every memory the package builds.
- `write_shot_data_file` and `read_shot_data_file`, Stim's functions with Stim's arguments, in
  all six of its result formats (`01`, `b8`, `r8`, `ptb64`, `hits`, `dets`): files byte for
  byte as Stim writes them, and Stim's read back to the same shots.
- The command line, `stabilizer-qec` (or `python -m stabilizer_qec`), after Stim's: `gen`,
  `sample`, `detect`, `m2d`, `analyze_errors`, `explain_errors`, `diagram`, `convert`,
  `sample_dem` and `help`, with Stim's flags and formats, and `decode` after PyMatching's
  `predict` (any of the package's decoders). Tested end to end against the `stim` command: the
  same bytes from `gen` (its header and qubit layout included), `analyze_errors`, `m2d`,
  `convert`, `explain_errors`, and from `sample` and `detect` on deterministic circuits.
- `Circuit.reference_sample()`: the noiseless run's measurement record, as Stim's.
- `Circuit.compile_sampler(*, skip_reference_sample=False, seed=None)`: raw measurement
  records, as Stim's measurement sampler: a noiseless reference run with each shot's flips,
  equal to Stim's on deterministic circuits and to its rates otherwise.
- `DetectorErrorModel.compile_sampler(seed=...)` (Rust: `DetectorErrorModel::sampler`): shots
  drawn from a model's faults directly, as Stim's `CompiledDemSampler`, with the faults that
  fired (`return_errors`) and recorded faults replayed (`recorded_errors_to_replay`); rates
  and pairwise correlations agree with Stim's sampler, and a seed's shots do not depend on the
  number of threads.
- `detector_error_model(..., ignore_decomposition_failures=True)` (Rust:
  `DemOptions::ignore_decomposition_failures`), Stim's option: a fault that cannot be split
  into graph-like pieces is kept whole. The text equals Stim's on its colour codes.
- Documentation: the guide's "Knowing a circuit" and "The command line"; tutorial 01's "How
  far from failing"; the README's and the report's "Circuit distance", every memory the package
  builds measured three ways (`tools/distances.py`, `data/distances.json`), where the
  integer program's best solution is kept as an upper bound when it does not finish. It finds
  the 2-round [[90, 8, 10]] memory has circuit distance at most 9 (checked: 9 faults, no
  detector, five logical qubits flipped).
- A paper for the Journal of Open Source Software (`paper/`, built by the Paper workflow) and
  its submission checklist (`docs/joss.md`); `CONTRIBUTING.md` says where to ask for help.

### Changed

- An explicit zero argument on `M`, `MX`, `MR`, `MRX` and `MPAD` (`M(0) 0`) is kept and
  written back, as Stim keeps it.

## 1.5.0 — 2026-10-04

The Python package and the Rust crate both at 1.5.0: Stim's generated circuits character for
character, every bivariate bicycle code of Bravyi et al.'s table and any CSS code from its
checks, tutorials, and benchmarks against Stim and PyMatching tracked in CI.

### Added

- `Circuit.generated(code_task, *, distance, rounds, ...)` (Rust: `Circuit::generated` with
  `GeneratedNoise`): Stim's generated memory experiments, the repetition code, the rotated
  and unrotated surface codes in either basis and the colour code's XYZ memory, with Stim's
  four noise parameters. The text is Stim's character for character (every task at distances
  2 to 7 and 25, one to seven rounds and 50, with and without each kind of noise), and the
  arguments Stim refuses are refused.
- Every bivariate bicycle code of Bravyi et al.'s Table 3: `BivariateBicycleCode("90")`
  ([[90, 8, 10]]), `"108"` ([[108, 8, 10]]) and `"288"` ([[288, 12, 18]]) beside `"72"` and
  `"144"`/`"gross"` (Rust: `bb90`, `bb108`, `bb288`, `named`), and any other from its
  polynomials, `BivariateBicycleCode.from_polynomials(l, m, a, b)` (Rust: `from_polynomials`),
  checked for weight-six checks and at least one logical qubit. Each comes with the paper's
  depth-8 syndrome cycle, its memories in both bases, its logicals and its automorphisms; a
  noiseless memory of each is checked to fire no detector. `polynomials` gives a code's torus
  and monomials.
- `CssCode` (Python and Rust): any CSS code from its check matrices, checked to commute and
  to encode a qubit, with paired logical operators and a memory experiment correct for any
  such code (every X check measured, then every Z check, in layers no qubit is used twice in;
  rounds as a loop). Two families come built: `CssCode.hypergraph_product(h1, h2)` of any two
  classical codes (the Hamming code's with itself is [[58, 16, 3]], two repetition codes'
  the unrotated surface code), and `CssCode.color_code(d)`, the triangular 6.6.6 colour code
  on Stim's layout. Their memories' error models equal Stim's character for character, and
  decode with `BpOsd`.
- Tutorials: three notebooks in `tutorials/` (getting started, from a circuit to a threshold
  plot; decoders and real-time decoding; codes beyond the surface code), each built from a
  plain Python file by `tools/notebooks.py`, and run on every push.
- Benchmarks against Stim and PyMatching (`tools/bench.py`): error models, sampling,
  measurement conversion, matching and correlated matching, each as a ratio to the
  reference's time on the same machine and thread. Runs are recorded in
  `data/bench/runs.json` and shown on the site's new benchmarks page; every push is checked
  against the last run recorded on the same kind of machine.
- Zenodo metadata (`.zenodo.json`) and the steps to link the repository
  (`docs/zenodo.md`), so each release can be archived with a DOI; a Citing section in the
  README.

### Changed

- Instruction arguments are written as Stim writes them (`1e-05`, not `0.00001`) whenever
  Stim's six significant digits hold the value exactly; a value they would round is still
  written in full, so circuit text never loses precision.

## 1.4.0 — 2026-10-04

The Python package and the Rust crate both at 1.4.0: window decoding of models too long to
unroll, streaming any circuit with a loop, and circuit diagrams after Stim's.

### Added

- Window decoding of models too long to unroll. `WindowMatching` takes its windows' graphs
  from a short template of the model's longest loop, whose middle windows serve every window
  away from the model's ends, shifted by whole passes of the loop; shots are fed in layer by
  layer. A d = 11 rotated memory of 10,000 rounds (27.6 million faults) builds its model in
  0.2 s and its decoder in about a second, and decodes from the folded model. It is the
  default for a model of more than a million faults with a loop; `template=True` (Rust:
  `WindowOptions::with_template`) asks for it, `template=False` for the whole model, and both
  give the same predictions (tested on Stim's rotated, unrotated and repetition memories,
  sliding and parallel, with and without correlations). `WindowMatching.streamed` says which.
- `stream_memory(circuit=...)` (Rust: `stream_circuit`): any circuit with a loop, sampled
  round by round and window-decoded as it streams, its windows from a template of its folded
  model. The template's middle spans a whole number of window periods that is also a whole
  number of the loop's passes, so loops Stim folds two rounds at a time are served.
- Diagrams, after Stim's `diagram`: `Circuit.diagram(type, tick=...)` (Rust:
  `Circuit::diagram(DiagramKind)`) draws `"timeline-text"` and `"timeline-svg"`, every
  operation in its column with `TICK` groups bracketed, loops drawn once with their count and
  measurements numbered; `"detslice-text"` and `"detslice-svg"`, what each detector compares
  after `tick` `TICK`s, its Paulis drawn as shapes over the qubits at their coordinates (the
  same slices as Stim's at every tick of its generated surface, colour and repetition
  codes); and `"matchgraph-svg"`, a decomposed model's matching graph
  (`DetectorErrorModel.diagram`). The result is a `Diagram`: it prints as its text, shows as
  a picture in a notebook, and `save`s to a file. The pictures follow the reader's light or
  dark theme.

## 1.3.0 — 2026-10-04

The Python package and the Rust crate both at 1.3.0: error models whose text is Stim's,
character for character; a union-find decoder; and a measurement converter that never unrolls.


### Added

- `UnionFind` (Python and Rust), weighted union-find decoding (Delfosse and Nickerson, with
  Huang, Newman and Brown's weighted growth) on the matching graph, and `"union_find"` in the
  sinter adapter. Its corrections always explain the detection events, every single fault is
  corrected, and its logical error rate is a little above matching's (d = 9 at p = 0.005:
  0.70% against 0.65%). It is the standard baseline rather than the fast option here: this
  engine's matcher, near-linear too, is about twice as fast at every noise strength measured.

- The measurement converter no longer unrolls a circuit: its reference is one tableau run
  through the loops as written, determinism and the sweep bits' effects come from the error
  model's backward walk (folding loops; seven extra random runs, and one per sweep bit, are
  gone), and shots are read by a program of the circuit's measurements and detectors with its
  loops kept. A memory of millions of rounds converts; the 2²⁴-instruction limit is gone.

### Fixed

- Building an error model, or a converter, for a circuit whose walk never settles into a
  period (a qubit read every round and never reset) is refused once its work passes the limit,
  rather than running for hours: the limit now counts the sensitivity sets merged, not only
  the instructions.

### Changed

- Error models print as Stim prints them, character for character (515 of 515 random,
  disjoint, tagged, Pauli-observable and looped circuits, and Stim's generated memories). Fault
  classes are Stim's: a fault's pieces in the order they arose, so a fault Stim lists once per
  order is listed so here too, and `num_errors` equals Stim's. Probabilities are accumulated in
  Stim's order and rounded as Stim's build for the machine rounds them (its ARM64 build fuses a
  multiply into an add), and printed to Stim's 16 significant digits.
- `PAULI_CHANNEL_1` is converted to independent errors as Stim converts it: with the cases
  relabelled so that identity is the likeliest, and by Newton's method where no exact solution
  exists, so channels Stim treats as independent need no `approximate_disjoint_errors` here
  either. Disjoint cases that cannot be told apart are merged in Stim's order.

## 1.2.0 — 2026-10-02

The Python package and the Rust crate both at 1.2.0: circuits built in code, error models
with their loops folded as Stim folds them (long memories that 1.1 refused now build in a
tenth of a second), the crate level with the Python package, and wheels for free-threaded
Python, musl Linux and Windows on ARM.


### Added

- Circuits built in code, as in Stim, in Python and in the crate. `Circuit.append(name,
  targets, arg, tag=...)` takes an instruction a time: targets as qubit indices, Stim's target
  text (`"rec[-1]"`, `"!3"`, `"X0"`, `"*"`) or `stim.GateTarget`s, and whole Stim circuits,
  instructions and repeat blocks; `+` and `+=` join circuits, `*` and `*=` make a `REPEAT`
  block (multiplying the count of a circuit that is one loop, as Stim does);
  `append_from_stim_program_text` and `copy`. In Rust: `Circuit::new`, `append` with the new
  `Target` and `Pauli` types, `append_tagged`, `append_text`, `append_circuit`, `repeated`, and
  the operators `+`, `+=`, `*`, `*=`. A failed append leaves the circuit unchanged.

- Error models with their loops folded, as Stim folds them. The model is built by walking the
  circuit's structure backwards with sparse sensitivity sets, finding where a loop's state
  repeats (tortoise and hare, as Stim's error analyzer), and writing one period as a `repeat`
  block. A d = 11 rotated memory of 10,000 rounds, which 1.0 refused, now builds in about 0.1 s
  and prints in 15,000 lines, its structure (the `repeat` blocks, the declarations, and the
  faults between them) equal to Stim's. Checked against Stim on its generated memories
  (rotated, unrotated, repetition and color codes), nested and tagged loops, and random
  circuits in loops.
- `DetectorErrorModel.flattened()` (Python and Rust), Stim's: the model without `repeat`
  blocks or `shift_detectors`. `Circuit.detector_error_model(flatten_loops=True)` (and
  `DemOptions::flatten_loops`) walks every pass instead of folding.
- Models are held as written, `repeat` blocks and all: Stim's folded models read without
  unrolling, and are counted through their loops. A decoder unrolls its model once, up to
  2²⁴ faults and declarations.

- The crate catches up with the Python package: `BivariateBicycleCode::automorphisms` (each
  shift's and duality's action on the logical qubits, as `Automorphism`s),
  `BivariateBicycleCode::gauging` (the ancilla system measuring a gross-code logical, as a
  `Gauging`), and `stream_memory` (a memory streamed round by round and window-decoded, as a
  `StreamResult`). The Python package and the crate now share the streaming loop.

- Wheels for free-threaded CPython 3.14 (cp314t), musl Linux (x86_64 and aarch64) and
  Windows on ARM. The module declares it does not need the GIL, and every object can be
  shared between threads: decoders decode from many threads at once, and threads sharing a
  sampler draw disjoint batches of its stream. CI runs the suite on free-threaded 3.14.
- `CITATION.cff`, `SECURITY.md`, `CONTRIBUTING.md`, and Dependabot for the workflows and the
  crate's dependencies.

### Fixed

- Two threads sampling from one `DetectorSampler` no longer raise "Already mutably borrowed":
  each call reserves its batches before it samples.

### Changed

- PyO3 0.29 (from 0.22), for Python 3.14 and free threading.
- A built model prints as Stim prints one: errors first in each stretch, then detector
  declarations with coordinates, `logical_observable` only where needed, and loops folded.
  Decoders take a model's faults in the order it is written, so a model, its text, and its
  pickle decode alike. Within a fault, pieces are listed in sorted order, where Stim lists
  them in the order they arose and so lists one fault once per order; the faults are the
  same, and `num_errors` counts ours.
- A circuit that reads a measurement record before its first measurement can now be made (it
  may be a piece of a larger circuit, a round comparing with the last); sampling it, building
  its error model or converting its measurements is the error, as in Stim. Counts are combined
  piece by piece, so building a circuit an instruction at a time stays linear.

## 1.1.0 — 2026-10-02

The Python package and the Rust crate both at 1.1.0. The crate gains the circuit-language
additions below (its API is unchanged); the rest is the Python package's.


### Added

- Stim's instruction tags: `H[tag] 0`, `REPEAT[tag] 5 {`, `DETECTOR[tag](1, 2) rec[-1]`, on
  every instruction. Tags change nothing a circuit does, print back as written, and reach the
  error model as Stim carries them: `error[tag](p)` for faults of tagged noise (faults with
  different tags kept apart), `detector[tag]` and `logical_observable[tag]`. Error models read
  tags too. A `#` inside a tag is not a comment.
- Pauli targets in `OBSERVABLE_INCLUDE` (`OBSERVABLE_INCLUDE(0) X0 !Z3 rec[-1]`): the observable
  takes in the Pauli's value at that point, so errors before it that anticommute with it flip
  it. In error models and both samplers, as in Stim; a Pauli observable the state does not fix
  is refused as Stim refuses it, and raw measurements convert as Stim converts them.

- `stabilizer_qec.sinter`: the decoders in [sinter](https://pypi.org/project/sinter/)'s
  Monte Carlo sweeps. `sinter_decoders()` names them for `custom_decoders` (and for
  `sinter collect --custom_decoders_module_function "stabilizer_qec.sinter:sinter_decoders"`):
  `sq_matching`, `sq_correlated_matching`, `sq_belief_matching` and `sq_bposd` decode the shots
  Stim samples; `sq_sim_matching` and its kin are sinter samplers running the whole pipeline
  here, postselection included. `Decoder(kind, **options)` and `Sampler(kind, **options)` make
  others. A `sinter` extra installs sinter and Stim.
- Pickling and copying (`pickle`, `copy`, `multiprocessing`, `concurrent.futures`):
  `Circuit` and `DetectorErrorModel` as their text; the converter and every decoder as what
  they were made from, rebuilt on the other side (`BpDecoder` and `BpOsdDecoder` keep their
  last run's results). A `DetectorSampler` refuses with a `TypeError` saying why: a copy would
  restart its stream, so send the circuit and a seed instead.

### Fixed

- A printed error model declares observables no fault flips (`logical_observable L1`), so it
  reads back with the same number of observables.

## Rust crate 1.0.0 — 2026-10-02

The crate (`stabilizer_qec` on crates.io, released on `crate-v*` tags) gets a designed, stable
API, re-exported at its root and mirroring the Python package's: `Circuit` and
`DetectorErrorModel` (`FromStr` / `Display`), `DemOptions`, `DetectorSampler` and
`MeasurementConverter` producing bit-packed `BitTable`s, the decoders `Matching`,
`BeliefMatching`, `BpOsd`, `WindowMatching`, `BpDecoder` and `BpOsdDecoder` (all `Send + Sync`),
`memory_circuit`, `BivariateBicycleCode`, the `lattice_surgery` module, and an `Error` type.
Semantic versioning from here, checked by `cargo-semver-checks` in CI. The engine's modules stay
public for the bindings and tools but are hidden from the documentation and outside the promise.
Rust 1.87 or later.

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

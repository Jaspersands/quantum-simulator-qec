# Changelog

All notable changes to `stabilizer-qec`. The Python package follows
[semantic versioning](https://semver.org) from 1.0, under the
[compatibility policy](CONTRIBUTING.md#compatibility-policy): public names keep working through
a major version, and anything deprecated warns for at least one minor release before a major
release removes it; behaviour changes in a minor release only to correct it, and each such change
is named under **Changed** or **Fixed**. The Rust crate is versioned on its own, under the same
policy for its documented items.

## 2.2.0 — 2026-10-24

The coherent sampler sums the selected completed kernel in floating-point arithmetic when
its contraction plan fits the entry budget, where 2.1 summed each shot's whole class only
up to twelve. PyMatching's own sparse blossom, transliterated line for line, is now the matcher
behind PyMatching's API and behind Chromobius. tqec's compile pipeline, PyMatching's `animate`,
Tsim's ZX graphs, CUDA-Q QEC's `dem_chunks` and `trt_decoder`, the QDK's parameter classes and
report, Qualtran's Dash app, Fusion Blossom's Bottle server, and Tesseract's and Chromobius's
command lines are ported, each checked against what its original records or its own tests. What is
still not ported, and why, is at the end. The Python package and the Rust crate both at 2.2.0.
Behaviour changes are under **Changed**.

**Exact class sums, by contraction.** A coherent-sampler shot's weight sums the amplitudes of
every fault set no record tells apart from the shot's: its class, a coset of the circuit's
spacetime kernel, 2^k sets for k generators. 2.1 summed the whole class up to twelve generators
and truncated it past that. 2.2 writes the sum over c ∈ GF(2)^k as a product of small factors
(one for each group of generators that share locations, a phase and a sign for each generator,
and (−1)^(c_i c_j) for each pair whose operators anticommute: a quadratic form) and contracts it
one generator at a time along a min-fill elimination order, in O(k·2^w) a shot, w the
contraction width (`CoherentSampler.contraction_width`).

- **When it is exact.** Whenever the contraction's tables fit
  `compile_coherent_sampler(max_contraction=65536)`, table entries in all. 0 turns the
  contraction off, a budget past 2^31 is 2^31, and no table spans 32 or more generators at any
  budget. Past the budget, a kernel of at most twelve generators is still listed whole, as in
  2.1; a larger one is truncated as 2.1 truncated it (`exact_weights` is False), and the sampler
  now says so with a `UserWarning`, where 2.1 was silent. Its entry lower bound is not a
  guaranteed sufficient budget. The warning describes the chosen basis/elimination heuristic: "the circuit's K interference generators contract past max_contraction=B table
  entries (their tables would hold N or more): ...; a max_contraction below N cannot sum the
  whole class, and N or more may"; or, when no budget would, "... interact too widely to
  contract at any max_contraction: their tables would hold more than 2^31 = 2147483648 entries
  in all, the most a budget allows; ..."; or "the contraction is off (max_contraction=0), ...".
  Unexpected internal refusal warns and asks for the circuit. Width is not optimized over
  equivalent circuit representations. Successful planning can dominate compilation: audited
  20/40-round crosstalk memories took 0.3060–0.3078 / 2.1947–2.2014 CPU seconds, versus 2.1.0's
  0.00648–0.00812 / 0.02049–0.02076 (three runs, shared load; different exact/truncated work).
  Sampling scratch per active worker is `24*(arena+largest)` bytes before factors/overhead
  (`arena=entries/2`, at most `36*entries`), so the entry budget is not a byte limit.
- **Checked** against the whole kernel listed (a random circuit for each k from 13 to 20, every
  one with couplings between its generators, to 1e-10), against a sum over every one of the 2^k
  products (eight random circuits for each k from 13 to 16, to 1e-11), and against the state
  vector.
  These are finite controls, not a proof over all numerical ranges. Contraction can change
  last bits even at ≤12 generators; corrected proposal/group arithmetic can change seeded
  outcomes with `max_contraction=0` too. Historical sample-time ratios require remeasurement.
- **The joint sampler** keeps 2.1's rule, the whole class up to twelve generators and the
  truncation past that, which `exact_weights=False` reports without a warning: its cross terms
  are overlaps of the exact engine's states, not products of factors, so `max_contraction` has
  no effect on it.
- **Measurement provenance.** All 8 validation and 28 capacity rows in `data/coherent/`, and
  the 6 d = 3 sweep and 4 d = 3 crosstalk rows, were remeasured with the 2.2.0 wheel and carry
  `generation_revision="coherent-merged-proposal-v1"`, which the current table and chart
  generators require. This marker records provenance, separately from `coherent.exact` and
  validation; it is never inferred from a version or added to an old row. Only the 12
  larger-distance sweep and 5 crosstalk rows, measured by 1.8.0 before the inherited
  merged-proposal correction, remain, as archive data excluded from the current tables; their
  ratios, ESS/error-bar claims and distance-scaling conclusions are not current validation.
  Earlier development-2.2 measurements remain in Git history and the preserved raw evidence.

### Added

- **PyMatching's sparse blossom, transliterated.** `src/pm_exact/` is PyMatching 2.4.0's
  `sparse_blossom` line for line: its radix heap and the order it breaks ties in, its arenas'
  reuse of freed slots, and its graph's neighbour order. `Matching`, PyMatching's class here
  (`sq.Matching`, `stabilizer_qec.pymatching.Matching`, sinter's `pymatching` decoder), decodes
  on it: `decode`, `decode_batch`, `decode_to_edges_array` and `decode_to_matched_dets_array`
  agree in the recorded plain-decoding corpus, ties and orders included. Checked shot by shot on
  950,000 shots of 19 models (surface codes d = 3 to 7 at p = 0.001 to 0.02, colour-code
  Möbius graphs, repetition codes, unit and negative weights) and 600,000 of 3,000 hand-built
  graphs: 0 differences. It takes 1.00 to 1.04 times PyMatching's time (`decode_batch` with
  weights, d = 5 memory over 5 rounds, 100,000 shots, one thread). The tie tolerances in the
  PyMatching, sinter, PanQEC, beliefmatching (min-sum BP) and folding comparisons are gone:
  those finite comparisons are exact. Correlated facade decoding uses the separate matcher and
  requires an unedited DEM/circuit loaded with correlations, with actual observable IDs 0–63;
  ensured wider output is zero-padded. Edited or hand-built graphs refuse correlated decoding,
  even where upstream accepts them. Two further divergences are deliberate. When a bit-packed
  batch fails on a padding bit, PyMatching's later decodes raise "!mwpm.flooder.queue.empty()"
  if the failing shot had earlier detection events at non-boundary nodes; this decoder refuses
  the shot before it makes any region, and keeps decoding. A failed batch still names its shot
  ("shot k: " and PyMatching's message). And a region's radius offset is 64-bit: PyMatching
  2.4.0's `compute_wrapped_radius` returns `int32_t`, so where a region's scaled radius passes
  what that holds (a uniform chain of 131 or more nodes at weight 1.5 or 16,777,214.5) its decoder never
  finishes, and this one decodes the path (`tests/test_matching_radius.py`). A fault id at or
  past the decoder's count, which only a `replace` or `smallest-weight` merge can put on an
  edge (merges do not extend the count), is ignored when the count is 64 or fewer, as it is
  above 64: PyMatching's `1 << id` is undefined there, and its arm64 wheel drops such ids eight
  at a time and wraps the rest modulo 64, so no rule matches it, and this one never reports a
  counted fault id for an uncounted one.
- **PyMatching's `animate`**: `stabilizer-qec animate`, `python -m stabilizer_qec.pymatching
  animate` and `pymatching.cli(command_line_args=["animate", ...])`, with PyMatching's flags,
  messages and exit codes. Its frames are those of PyMatching 2.4.0's macOS arm64 wheel: all
  1,080 frames of 11 recorded cases equal as sorted lines, the arithmetic fused as that wheel
  fuses it on aarch64 (nothing is fused on x86-64), and all 44 recorded misuses give the
  wheel's messages. Of 600 random command lines, 590 behave identically; the other 10 name a
  model that does not parse (see the Tesseract entry). These recordings do not establish
  general byte parity for NaN/overflow geometry or locale-dependent high-byte parsing. All
  coordinate axes contribute to projection. Write/close failures produce controlled exit 1,
  deliberately differing from observed native silent truncated success. `--dets_in` is read only to the first
  shot's end, as C's stdio reads it, and a model's bytes past ASCII in comments and tags are
  passed over, as Stim passes over them. Where PyMatching's program aborts (an `--out_dir` it
  cannot make, a hit index past 64 bits), `stabilizer-qec animate` prints `stabilizer-qec
  animate: <message>` and exits 1, and `pymatching.cli` raises the RuntimeError PyMatching's
  does. Out-directory messages follow local host/library policies; exact probes below
  cover macOS arm64, not universal native wording. MSVC's wording is not reproduced. Fresh exact ENAMETOOLONG/EACCES probes were macOS
  arm64 only; a trailing diagnostic punctuation difference is retained. The six-digit halfway
  formatter correction was checked on 20,374 finite f32 values on Apple libc, not all runtimes.
- **Chromobius on PyMatching 2.2.2's matcher, and its command line.** The Möbius matching is now
  `pm_exact` in the neighbour order of PyMatching 2.2.2, the PyMatching Chromobius 1.1.1 links.
  The lifting gets PyMatching's edges in PyMatching's order. Recorded macOS predictions agree
  on every shot of a 65,000-shot sweep (colour memories d = 3, 5 and 7
  over 3 to 7 rounds, p = 0.001 to 0.01, X and Z; 240,000 shots in an extended run) and on all
  340,000 shots of `tools/decoder_check.py`'s colour cases (0 ties, at 0.72 times Chromobius's
  time), with the weights equal as float32. This finite evidence does not ensure equal
  observable choices on other inputs: tied lifting can differ at equal matching weight on
  any platform, including macOS. Facade cases permit 95% agreement; the sweep requires 99%. `python -m stabilizer_qec.chromobius
  predict|benchmark|describe_decoder|help` is Chromobius's program, with Stim's flag parsing
  and record formats, and its outputs and messages, checked against 184 cases recorded from
  Chromobius 1.1.1's own binary. Where it differs: a `--circuit` that Stim's error analysis
  refuses gets this package's analysis message; a fault of probability 1 is refused
  (Chromobius describes and decodes it, with NaN weights); a model that only declares an
  observable past L63 gets this package's 64-observable message (Chromobius decodes it, its
  predictions past L63 undefined); records are read whole before decoding, and sparse hits/dets
  additionally expand to a shots-by-detectors bool array; `predict` decodes on every core; an index whose digits overflow Stim's 64-bit read
  prints pinned macOS libc++abi text on every host and exits 134 without raising SIGABRT; a bit past the
  last detector, where Chromobius's result is undefined, is refused with PyMatching's message
  every time; and the matcher's weight scale counts a self-loop given more than once at its
  merged weight alone, where PyMatching 2.2.2's also counts its first weight (no Möbius graph
  has a self-loop). Invalid UTF-8 model bytes use replacement; native byte preservation
  was not freshly verified. Precision policy is 16 significant digits on macOS arm64/Windows,
  19 on non-Windows x86-64, 34 on other ARM64; exact text fixtures are macOS arm64 only.
  Lifting failures use the port's shot-indexed message and preserve later valid calls rather
  than imitating native state damage. Finite-coordinate overflow now yields controlled exit 1.
- **Tesseract's command line, and any number of observables.** `python -m
  stabilizer_qec.tesseract_decoder` is Tesseract's `tesseract` program: its flags (`--circuit`,
  `--dem`, `--sample-num-shots`, `--in`, `--out`, the detector-order flags, `--beam`,
  `--pqlimit`, `--multipass`, `--stats-out`, ...), its argparse library's errors and help, its
  output lines, `--dem-out` and the `--stats-out` JSON. Checked against Tesseract's Python API
  run as its program runs it (7 flag sets on 120 and 60 recorded shots) and its source:
  predictions, counts and `--dem-out` equal, and the cost lines equal on the same model. With
  no shot to count from (an empty `--in`, or every shot a logical error), `--dem-out` holds
  `error(nan)` lines, as Tesseract's program writes them, and the stats and final line follow.
  Model and circuit bytes that are not UTF-8 are replaced, as in the Chromobius program, so a
  comment holding one reads as Stim reads it. Where it differs: `--sample-num-shots` draws this package's shots, not Stim's; an exception that
  ends Tesseract's `main` prints the same message and exits 134 without a core dump; an `--out`
  it cannot open stops with `Failed to open <path>` (Tesseract's program writes through a null
  file); `--max-errors` stops exactly at the limit (Tesseract's program keeps the shots its
  threads had already decoded, so its `num_errors` can pass it); and `--threads` shares one
  decoder (and runs on one thread with `--verbose` or `--multipass`). The search decoder now
  takes any number of observables, as Tesseract's does (`SearchDecoder`, Tesseract's
  decoders, `build_det_orders`, `build_detector_graph`): a 100-observable model recorded from
  Tesseract's wheel decodes as it does there, the verbose log included. The Rust API's
  `SearchDecoder` stays at 64, since its `Prediction.observables` is a `u64`, and refuses more
  with its own message. Prediction write/close errors now yield controlled exit 1 and preserve
  the primary error. Native Tesseract I/O-failure policy was not independently executed.
  In all three programs, as everywhere in this package, a model or
  circuit that does not parse is refused with this package's parser's message, not Stim's
  (`line 2: unsupported instruction 'FOO'`, where Stim raises `Unrecognized instruction name:
  foo`). The exception is a model fault that Stim's parser throws `std::out_of_range` for:
  an unrecognized instruction name, a number of 2^60 or more, a `{` never closed or a `}` never
  opened. Chromobius's program does not catch it, so the Chromobius program aborts as
  Chromobius's does, with Stim's message in the pinned libc++abi text and status 134.
- **tqec's compile pipeline, byte for byte.** tqec 0.2.0's `circuit`, `plaquette`, `templates`
  and `compile` packages are ported, with tqecd 0.2.1's detector search
  (`stabilizer_qec/tqec/_compile/`, on numpy alone, Python 3.9 up).
  `compile_block_graph(graph, convention=..., observables=..., block_temporal_height=...)`
  builds tqec's graph, and its `generate_stim_circuit(k, noise_model, manhattan_radius, ...)`
  writes tqec's circuit from tqec's own plaquettes, templates and detectors. Hadamard pipes,
  spatial junction cubes and the whole gallery compile (stability, move/rotation, three CNOTs,
  Steane encoding, CZ). All 51 circuits recorded from tqec 0.2.0, one fresh process each, are
  byte-identical, and the 6 recorded refusals raise tqec's type and message. It is faster than
  tqec in wall time on the cases timed (`compile_block_graph(graph,
  convention=...).generate_stim_circuit(1)`, three runs a side in separate processes at a load
  average of 5 to 11: 1.5 to 1.6 times on the Steane encoding, 2.8 to 3.1 on the CNOT, where
  tqec takes 9 to 10 s, 5.9 to 6.6 on the memory; repeated at load averages of 38 to 184 the
  ratios were 1.9 to 5.0 on the CNOT and 5.0 to 8.8 on the memory, higher in some runs and lower
  in others, with no trend with load). Where it differs, on purpose: each compilation starts from fresh caches, so its
  first generated circuit is a fresh tqec process's (tqec's caches hand out shared plaquettes whose schedules a
  compilation changes, so tqec's circuit for a graph can depend on what the process compiled
  before); the detector database is kept in memory (`database_path` is taken and never read or
  written); there are no worker processes (the output is the same); `k` must be a positive
  integer (tqec raises IndexError for 0 and writes a circuit for 1.5); and a
  `block_temporal_height` that is not a `LinearFunction` raises a TQECError (for a number, the
  one tqec raises for the constant height the number stands for), where tqec 0.2.0 fails with
  an AttributeError (a TypeError for an unhashable value). tqec 0.2.0's own bugs come with its
  circuits: `cz(["XI -> XZ", "IZ -> IZ"])` and `cz(["ZI -> ZI", "IX -> ZX"])`
  have non-deterministic observables at k = 1 (tqec issue #1034); `cz("ZZ -> ZZ")` in the
  fixed-boundary convention at k = 2 has a non-deterministic detector; and so do two L-shaped
  spatial-junction memories stacked with no pipe between them (#1062, fixed after 0.2.0). The
  fixtures record examples, not an exhaustive list: spatial Hadamards at spatial junctions
  can also produce non-deterministic detectors/observables. These inherited circuits are
  reproduced for parity, not certified as valid error-correction experiments. A spatial
  Hadamard pipe between two non-spatial cubes, along x or y, requires `fixed_boundary`;
  `fixed_bulk` lacks the template (one touching a spatial cube, as in the CZ, compiles in both).
  Y cubes and unfilled ports are unsupported. The lattice-surgery
  engine's layout from `compile_block_graph(graph, distance)` and `BlockGraph.compile`
  remains available at `.circuit`; `generate_stim_circuit` writes tqec's layout (see Changed).
- **Tsim's ZX graphs.** `tsim.Circuit.get_graph(backend=)` and
  `get_sampling_graph(sample_detectors, backend=)` build Tsim 0.1.5's ZX graphs, vertex for
  vertex and edge for edge (`stabilizer_qec.tsim.zx`: Tsim's instruction builders and parser,
  pyzx_param's simple graph with its vertex numbering and rewrite rules, and the sampling
  graph's transforms), on pyzx_param's `GraphS` where it imports and on this package's pure-Python
  `ZXGraph` otherwise (tensor contraction uses NumPy). `diagram("pyzx")`, `"pyzx-dets"` and `"pyzx-meas"` draw them, and
  `to_matrix` and `to_tensor` of a circuit with measurements, detectors, observables, feedback,
  a reset after a qubit's first operation or noise before a reset are Tsim's ZX tensor, where
  2.1 raised (measurements, feedback, later resets) or gave the wrong matrix (see Fixed).
  Checked against 40 recorded circuits exercising selected gates, channels and annotations
  (I_ERROR/II_ERROR are absent from this corpus) on
  both backends, and against live Tsim on 4,200 random circuits (five graphs each), 8,000
  random `to_matrix` (902 of them equal up to a global phase, the unitary engine's, below) and
  60 circuits' diagrams: 0 differences. CI runs a bounded sample of these live comparisons on
  every push (one Linux job with Tsim installed). Where it differs: the shots stay this package's (see
  Not ported); an angle appended as an int prints as a float, the spacing inside a tag is not
  kept, and a `stim.Circuit`'s angle literals are read as floats; two product rotations with
  the same angle in a row print as two lines, and so do a pair rotation's `TPP`s on several
  pairs and an undaggered pair rotation on several pairs (the inverse of an `SPP_DAG` one
  prints `R_ZZ(a) 3 2` then `R_ZZ(a) 1 0`), where Tsim prints each as one line
  (`R_ZZ(0.3) 0 1 Z2*Z3`, `TPP X3*X2 X1*X0`, `R_ZZ(a) 3 2 Z1*Z0`); and a product naming a qubit
  twice prints as `R_PAULI(0.3) X0*X0` (Tsim's `R_XX(0.3) 0 0` does not read back). Tsim reads
  each of these prints as its own circuit. An angle whose radians are past a float's range
  (from about 5.7e307 half-turns) is refused, where Tsim takes it. A record entry from before
  the first measurement (`M 0` then `DETECTOR rec[-2]`) raises `MeasurementRecordError`, an
  `IndexError` as Tsim's is and a `ValueError` naming the instruction. `to_matrix` of a circuit
  that does not need the ZX tensor is the unitary engine's, Tsim's up to a global phase
  (`SQRT_X_DAG 0` gives Tsim's times e^(iπ/4)), and so is that of such a circuit with an
  instruction Tsim's graph builders cannot read (a tag the core leaves inert, `I[foo(x)] 0`),
  where Tsim's raises its parse error (as it is raised here too when the circuit needs the ZX
  tensor or a product rotation names a qubit twice). `to_matrix` refuses more than 2^22
  entries, a spider of more than 24 legs and an intermediate tensor of more than 2^26 entries,
  where Tsim allocates.
- **CUDA-Q QEC: `dem_chunks` and `trt_decoder`.** CUDA-Q QEC 0.8.1's extended detector error
  models (`SeamId`, `ExtendedDem`, `DemChunksSpec` and every other name of `py_extended_dem.cpp`;
  `dem_stitch`, `dem_close`, `dem_chunks_to_pcm`, ...) and its CSS model construction
  (`dem_from_css_matrices`) are ported. `decoder_config.dem_chunks` is read from and written to
  CUDA-Q's YAML as CUDA-Q writes it, and `expand_dem_chunks` closes the chunks into the model the
  decoder gets; `create_realtime_decoder` expands them. Errors are CUDA-Q's types and messages.
  All 101 tests of CUDA-Q's `test_extended_dem.cpp` and 70 of its DEM and YAML tests (Python and
  C++) are ported, and the closed chunks equal, mechanism for mechanism, the model this package
  derives from the same circuit. `trt_decoder` runs its ONNX model on the CPU: a zero-dependency
  ONNX reader and a numpy evaluator of the operators the decoder needs (Gemm, MatMul, Add, Relu,
  Sigmoid, Identity, Cast, Reshape, Flatten, and QuantizeLinear and DequantizeLinear in int8,
  uint8 and float8), using deterministic typed operations and selected MLAS arithmetic: each
  float32 multiply-add rounded once, as MLAS's FMA rounds it, and a MatMul followed by an Add
  fused into one Gemm where onnxruntime's default optimizer fuses them. Its outputs match
  onnxruntime 1.31.0's bit for bit on all 180 cases recorded on 23 models (macOS arm64; CUDA-Q
  0.8.1's three models among them, equal to its git-lfs pointers' sha256). The selected
  numerical contract also checks CUDA-Q's 30 PyTorch surface-model outputs at absolute
  tolerance 1e-4. Other ORT optimizer/prepacking paths, computed MatMul weights, weight-only DQ
  fusion and reduced-precision paths can round differently on the same CPU;
  near 0.5 this can change a binary correction. This is not a native TensorRT/GPU comparison.
  [ONNX_CONTRACT.md](python/stabilizer_qec/cudaq_qec/ONNX_CONTRACT.md) specifies the targeted
  IR 7–10/opset 13–21 eleven-operator subset, types, tolerances and limits; it is not a full
  schema validator or resource bound. Numeric/strided/Fortran arrays, rectangular lists and
  sparse `O` inputs are intentionally broader than CUDA-Q's row-major uint8 tensor binding.
- **Azure: the QDK's parameter classes and its report.** qdk 1.33.1's `EstimatorParams` and its
  sections (`EstimatorQubitParams`, `EstimatorQecScheme`, `DistillationUnitSpecification`,
  `ErrorBudgetPartition`, `EstimatorConstraints`, ...), the QDK's code for Python 3.9, are taken
  by `LogicalCounts.estimate` and `azure_estimate`, and raise the QDK's own errors: a
  `LookupError` for a custom qubit's or unit's missing field, a name given with a custom unit, or
  both maximum constraints at once, and a bare `Exception` from `EstimatorParams().items` without
  `num_items`. `EstimatorResult` draws the QDK's report: `_repr_html_`, `summary` and slices as
  text with their HTML, `diagram`, a batch's `plot`, and qsharp_widgets 1.33.1's space and scatter
  charts as SVG (`space_chart()`, `scatter_chart()`). They are checked against 40 recorded
  results, plain and with markdown, and the charts against the widgets' own functions run in node,
  to 4 ulps. `QECScheme.SURFACE_CODE` and `FLOQUET_CODE` are in. Kept from 2.1: `EstimatorError`
  and `QSharpError` are `ValueError`s (the QDK's `EstimatorError` derives from BaseException
  alone), and `LogicalCounts.estimate` takes any object with `as_dict` (a batch when it
  `has_items`, else its one dictionary; the QDK refuses an `EstimatorInputParamsItem` with a
  TypeError). A batch keeps its `summary` and single results their `summary_data_frame`, which
  the QDK lacks, and the tables are drawn when first asked for.
- **Qualtran's Dash app.** `stabilizer_qec.estimate.qualtran.ui` has Qualtran 0.7.0's layout
  builders and callback. `ui.make_app()` (and `ui.app`) is Qualtran's Dash app: its layout and
  callback match the recorded Qualtran cases through Flask's test client. Very large gate
  counts can differ in last bits of 15-to-1 factory cycles/durations. `ui.serve()`
  and `python -m stabilizer_qec.estimate.qualtran.ui` run it, on Dash where dash is installed
  (the new `dash` extra, `pip install "stabilizer-qec[dash]"`; CI installs dash 4.4.1) and
  otherwise on the standard library's HTTP server, with a page of this package's own that draws
  the component layout and runs the same callback, with template-free figures whose colors,
  backgrounds and fonts can differ from Dash's installed-Plotly template. Public builders are
  lightweight component/dependency mirrors, not Dash's full mutation/pickling/callback API;
  `make_app` converts them. Only the fallback guarantees ephemeral binding at port 0; Dash
  receives 0 unchanged (Dash 4.4.1 uses PORT or 8050). The fallback accepts JSON media types
  and bounds body reads at 1 MiB, with explicit length/method errors; this is not a bound on
  callback work, connection count or client duration, nor a Host/CORS security guarantee. That page loads the installed plotly's
  plotly.js, or without plotly cdn.plot.ly's plotly.js 3.7.0, pinned by its hash. Importing
  `ui` imports neither dash nor plotly (Qualtran's `ui` does not import without dash). Without
  dash, `ui.app` raises `make_app`'s ImportError, and so does `hasattr(ui, "app")`, where
  `hasattr` gives False for a missing attribute.
- **Fusion Blossom's `helper.bottle`.** fusion_blossom 0.2.13's vendored Bottle (0.14-dev) is
  `stabilizer_qec.fusion_blossom.helper.bottle`, byte for byte, and `helper.serve` is
  fusion_blossom's server on it: 55 recorded requests (18 paths by GET, HEAD and POST, and a
  range request) retain exact recorded headers and bodies, `index.js` after taking out the one
  change made to it: the page imports a `patch_url` only from its own origin, where
  fusion_blossom 0.2.13's imports any URL and so runs any web page's module in the data
  server's origin, where it can read the data folder (`serve`, `open_visualizer` and
  `peek_code` serve the working directory by default). Raw live parity also has two platform limits:
  GET/HEAD `//index.js` keeps the recorded 200 on older HTTP handlers whose upstream returns
  404; that exact 404 is checked separately, and the port must match the same live upstream's
  canonical `/index.js`. Windows upstream website CRLF is compared as LF after raw body/length
  checks, while installed resources must contain LF. It is imported with
  `stabilizer_qec.fusion_blossom`, as fusion_blossom imports its own.
  LF bytes and the raw upstream SHA are enforced for the vendored path; a Windows wheel
  byte comparison remains unexecuted. Data serving now validates UTF-8 and stable-filesystem
  containment of Bottle's strip+abspath candidate while preserving original stripped spelling
  for both static-file calls (including Range preflight). Concurrent filesystem replacement
  is outside this guarantee. JSON uses application/json; extensionless files default to HTML.
  Quiet suppresses request logs, not all syntax/error logging.
  fusion_blossom also installs its Bottle and its helper as the top-level modules `bottle` and
  `helper` (so `import bottle` gives fusion_blossom's); this package installs neither.
- **Stim: a memory budget for the undetectable-error search.**
  `search_for_undetectable_logical_errors(..., max_memory_bytes=None)`, on `Circuit` and
  `DetectorErrorModel` (in Rust, the `*_with_budget` methods), keeps the sets it visits in
  columns, within the budget even while they grow: half the machine's physical memory by
  default, or on Linux half the cgroup's memory limit if that is less (Stim has no limit). A set
  of k detectors takes 24 + 4k bytes, plus 5 to 11 bytes of the table. When the sets fill the
  budget they hold between half and all of it, so a set takes one to two times its bytes in
  budget: about 40 to 90 bytes for the usual sets of two to four detectors. A larger budget
  never keeps fewer sets. Out of budget, the search raises Stim's message for a search that
  finds nothing, then "(memory budget of N bytes reached after M states)". Limits of (2, 2)
  take Stim's graph-like search, which the budget does not cover. The answers are unchanged:
  6,000 recorded searches give Stim 1.16.0's. On QUITS's [[72, 8, 8]] (PYTHONHASHSEED 0, the
  whole process, three runs each) the search takes 0.64 to 0.66 s of user time and a 144 to
  148 MB peak footprint, against 2.1.0's 0.79 to 0.81 s and 228 to 241 MB, and Stim's 4.25 to
  4.54 s and 530 MB. A gigabyte holds about twice the sets it did: on the gross code's memory
  at limits (6, 6), 2.1 gave up at its 20-million-set cap with a 2.45 GB footprint, and 2.2 with
  a 1 GB budget keeps 12.7 million sets in 763 MB.
- **ldpc: BP+LSD's merge orders.** When LSD merges three or more clusters at once, ldpc takes
  the order from a hash map keyed by heap addresses, so its answer to a syndrome can change
  from one decode to the next in the same process. `BpLsdDecoder._merge_order_outcomes(syndrome,
  max_leaves=4096, target=None)` walks the merge orders this decoder can take, fewest
  departures from `decode`'s order first, and returns the corrections they reach. On the 15
  windows of QUITS's sweep where ldpc's answer varied (recorded in `tests/ldpc_lsd_merge/`), the
  corrections reached are exactly two, this decoder's and ldpc's other. The QUITS differential
  test now replays ldpc's window answers and requires equal predictions, where it let a shot or
  two differ, and `tools/decoder_check.py` counts a BP+LSD shot as tied only when ldpc's answer
  is one of the reachable ones: in every run, none is not (how many differ varies from run to
  run with ldpc's heap). `decode` is unchanged. A random or relative serial schedule moves on
  at every decode; the search after `decode(syndrome)` runs that decode's BP again, from the
  schedule's state it started from, so `decode`'s answer is always among the corrections. The
  search of any other syndrome is the next decode's, and no search moves the schedule on.

### Changed

- **The compatibility promise.** 2.1's README promised that the public API "keeps working
  through every 2.x release", that a seed gives the same shots "on any machine and any number
  of threads", and that an engine bug raises `RuntimeError` asking for a report, "never a
  crash". 2.2 states the promise as the
  [compatibility policy](CONTRIBUTING.md#compatibility-policy), and narrows it. Names and
  signatures keep working and are deprecated before removal, as before, but what a call returns
  or raises can change in a minor release to correct it: to make a port do what its upstream
  does, or to fix a defect, each change named here. 2.2's tqec, Chromobius, Tsim, Fusion
  Blossom and coherent-sampler changes below are such corrections, and some 2.1 calls now raise:
  `compile_block_graph(g, block_temporal_height=5)` (a number or a callable without a
  distance) raises a TQECError, and `generate_stim_circuit` on a program whose `rounds` tqec's
  layout cannot honour (`compile_block_graph(g, 3, rounds=5).generate_stim_circuit()`) a
  ValueError, each saying what to call instead (see tqec below). Seeds reproduce within a version, on any number of threads, not across
  machines or versions: a sampler correction can change seeded shots, as 2.2's coherent
  corrections do. Crashes and hangs are bugs to report, not a supported response. Names a
  module merely imports (fusion_blossom's `helper.ThreadingHTTPServer`, say) are not part of
  the API.
- **The coherent sampler.** With more than twelve generators, a Clifford circuit's coherent
  sampler now sums each shot's whole class when the contraction fits `max_contraction`, so its
  weights and their spread can differ from 2.1's truncated ones. `max_contraction=0` keeps
  truncation, but does not undo corrected proposals or normalized group arithmetic. Past the budget it warns where 2.1 was silent, so code run under `-W error` stops
  there. `exact_weights` is True for a completed contracted kernel, and `contraction_width` is new.
- **The coherent data.** Newly constructed rows record actual `version` and the explicit
  generation revision above. `coherent.exact` describes whole-class coverage independently.
  Saving historical rows never upgrades their provenance. Repeated `--only` selectors for
  one field now union their values; selectors on different fields intersect.
- **PyMatching.** Among equally light matchings, `Matching`'s predictions, weights, edges and
  matched pairs are now PyMatching's (2.1 could pick another, and listed edges in its own
  order). `Matching.from_detector_error_model` and `from_stim_circuit` keep each fault's
  detectors in the order written, as PyMatching does, where 2.1 sorted them. A `Matching`
  loaded with `enable_correlations=True` is loaded as PyMatching loads it: it decodes plainly as
  PyMatching's does (2.1 used the plain load's weight scale, so ties and, rarely, predictions
  differed), its `edges()` list each edge low node first, and it refuses what PyMatching
  refuses, with PyMatching's messages (2.1 gave its correlated matcher's messages and took
  `L0 ^ D1` pieces). `ensure_num_fault_ids` keeps a decoder already built, as PyMatching's
  does: predictions are as long as the new count, and until the graph changes (or
  `decode_to_edges_array` or a correlated call rebuilds it) the decoder fills the fault ids it
  was built with. A correlated decode on a load whose implied weights name a self-loop refuses
  with PyMatching's "Failed to find neighbor.", and `decode_to_edges_array` with
  `enable_correlations=True` checks the syndrome first. Rust: `UserDecoder`'s 2.1 decoding
  methods (`scratch`, `decode`, `decode_to_edges`, `decode_to_matched_pairs`, the four-argument
  `decode_packed`, and `Decoded`) are deprecated, in favour of `decode_with`,
  `decode_to_edges_with`, `decode_to_matched_pairs_with` and `decode_packed_with`, each taking a
  workspace from `mwpm()`.
- **Chromobius.** A colour-code model is refused as Chromobius refuses it, with its messages: a
  detector declared without a fourth coordinate that is a whole number from −1 to 5
  (`detector D0` among them), which 2.1 decoded as uncoloured; an error naming a detector no
  declaration colours ("Expected all detectors to have at least 4 coordinates, ... but got
  error(0.1) D1 L0", where 2.1 said "detector D1 has no colour annotation ...") or an
  observable past L63 ("Max logical observable is L63 but a larger one appeared in ...", where
  2.1 gave its 64-observable message); and a fault it cannot decompose (Chromobius's message,
  with the instruction, its pieces and the detectors' data). The model is read as written, as
  Chromobius reads it: a detector declared twice takes its last declaration. A shot the
  Möbius matching cannot match is refused by `ColorMatching.decode_batch` with PyMatching's
  "No perfect matching could be found ...", where 2.1 said "no lifting of the matching explains
  the detection events".
- **tqec.** `compile_block_graph(graph)` and its `generate_stim_circuit` are tqec's pipeline now,
  so their circuits are tqec's: the CNOT at k = 1 has 81 qubits, where 2.1's engine layout had
  113.
  - `compile_block_graph(g, block_temporal_height=5)` without a distance, or with a callable
    of k, ran the engine's rounds in 2.1 (height + 2 in every time slice). It now raises a
    TQECError (tqec's height is a `LinearFunction` of k): for a number, the one tqec raises for
    that constant height, and for a callable, one saying the height must be a `LinearFunction`.
    Each is followed by a sentence pointing at the engine, which still takes numbers and
    callables at a distance: `compile_block_graph(g, distance, block_temporal_height=...)` and
    `BlockGraph.compile`. The engine now also takes NumPy integers, which 2.1 refused, as
    heights and as the distance (see Fixed); a float or bool distance is a TQECError (2.1
    raised a TypeError for a float).
  - `CompiledBlockGraph.generate_stim_circuit` (on `compile_block_graph(g, distance)`) wrote
    the engine's layout at distance 2k + 1 with the program's `rounds` and `merged`. It now
    writes tqec's layout and takes tqec's keyword parameters (an unknown keyword is a
    TypeError, where 2.1 ignored it), and raises a ValueError pointing at `.circuit`, the
    engine's layout, for a program whose rounds tqec's layout cannot honour.
  - Separate compilations isolate the cache initialization/build transaction across threads.
    First generation matches a fresh reference process; repeated generation of one compilation
    retains upstream mutable rescheduling. Concurrent mutation/generation of one object is not
    guaranteed. Built-in compilation pickle restores schedules and the next generated circuit;
    observable iterables are materialized once. Direct graph rounds/merged controls are checked
    against tqec's 2k + 1 layout on the graph's time slices, on every graph tqec compiles
    (spatial cubes and Hadamard pipes too): matching values pass through to tqec's circuit,
    others receive the same layout ValueError as the distance wrapper.
  - `LinearFunction.unambiguous_max_on_positives` inherits tqec's single-pass-iterator caveat
    (unlike 2.1): candidate checks can consume it and yield an incorrect maximum; pass a list.
  - `database_path` (on `generate_stim_circuit`, `generate_stim_circuits_with_detectors`,
    `generate_sinter_tasks` and `start_simulation_using_sinter`) defaults to tqec's
    `DEFAULT_DETECTOR_DATABASE_PATH`, where it was None. The path is never read or written.
- **Tsim.** `R_PAULI(±0.25)` and `R_XX`, `R_YY` and `R_ZZ(±0.25)` print as written and are
  Tsim's `R_PAULI`: its graph's scalar, its matrix through the ZX path and its errors. 2.1
  stored and printed them as `TPP`. An angle is kept as written: 2.1 printed back the shortest
  decimal that gave the stored radians, so `R_Z(1.9967044602602857)` came back as
  1.996704460260286. `inverse` keeps the literals too (a zero as `0.0`, as Tsim's). `tcount` is
  Tsim's, the graph's count, in which a coherent error is not a gate
  (`stabilizer_qec.Circuit.tcount` still counts its phase). The unrolled Python graph incurs
  memory/time and can refuse sweep targets, malformed tags, unsupported products and invalid
  record lookbacks where 2.1 returned its fast count. The two identified repeated scans were
  removed without a general linear-cost promise. Aliases/case are resolved before tags, and
  measurement-only/feedback/measured-I[RL] matrices use the ZX tensor. Lazy-chain integer
  indexing works beyond sys.maxsize outcomes; len still overflows and full materialization
  remains refused. T/T_DAG/TPP suffix metadata is not retained (`S[T:x] 0` prints `T 0`). A circuit with Pauli products or
  sweep bits has its ZX graph built before `to_matrix`, so its errors are Tsim's. Choosing the
  default ZX backend imports pyzx_param where it is installed, which calls
  `numpy.set_printoptions(suppress=True)` for the whole process, as importing Tsim does.
- **Azure.** `EstimatorError.args` is the QDK's `(code, reason)`, where it was `(message,)`,
  and its repr follows (`EstimatorError('<code>', '<reason>')`); `str(e)`, `.message`, `.code`,
  `.reason` and pickling are unchanged (read `e.message` where code read `e.args[0]`). A batch's
  side-by-side table shows "N/A" for a frontier item's counts, as the QDK's does, where 2.1
  read the item's first frontier entry (`result[i].summary` has it). A failed result's
  `summary` raises an error that is both 2.1's `EstimatorError`, with the estimate's code,
  reason and message, and the QDK's `AttributeError`: `hasattr(failed, "summary")` is False and
  `getattr(failed, "summary", d)` gives `d`, where 2.1 raised from both, and its `str()` is the
  QDK's "'EstimatorResult' object has no attribute 'summary'" (read `e.message` for the
  estimate's). `summary` and `result[i:j]` are `Report`s, a `str` subclass whose
  `_repr_html_` holds the HTML.
- **Fusion Blossom.** `helper.serve` is fusion_blossom's Bottle server, so it answers as
  fusion_blossom's does where 2.1's standard-library server answered otherwise: `icon.svg` is
  served as `text/html; charset=UTF-8`, `/data/../x` is a 403 "Access denied.", errors are
  Bottle's pages, data files carry ETag, Last-Modified and Accept-Ranges and answer range and
  conditional requests, HEAD works, and POST is a 405. The server is wsgiref's, one thread, and
  leaves its listening socket open after `srv.shutdown()`; a later `serve` call changes the
  data folder of servers already running. Binding happens when the runner starts;
  `server_close()` releases the socket after shutdown. `open_visualizer` propagates bind
  failure before browser/input work and cleans up its owned worker/socket on input errors;
  synchronous startup and in-flight requests are not universally time-bounded. `import stabilizer_qec.fusion_blossom` now loads
  Bottle and appends Bottle's import hook to `sys.meta_path`, as
  `import fusion_blossom` does. `import stabilizer_qec` and `import stabilizer_qec.sinter` load
  neither, as `import sinter` does not: sinter's Fusion Blossom decoder imports Fusion Blossom
  when it decodes, as sinter's does.
- **Stim's undetectable-error search.** The 20-million-set cap and its message ("the search
  visited 20000000 sets of detectors without finding an undetectable logical error; lower its
  limits") are gone: the search runs to its memory budget.
- **Models past 64 observables.** The refusal now names what takes at most 64 (the matching,
  BP, clustering and window decoders, the distance searches, matching-graph diagrams and the
  model sampler) and what takes any number (the Python search decoder, the text, counts,
  circuit sampling, the leakage samplers among it (`compile_leakage_sampler`, and
  `deltakit_stim`'s samplers of a circuit with leakage, which 2.1 refused past 64), and
  measurement conversion).
- **CUDA-Q QEC.** A configuration with `dem_chunks`, which 2.1 refused, is expanded, and
  `dem_chunks` with `stim_dem_path` is refused, as in CUDA-Q. `trt_decoder`, which 2.1 lacked
  (as a CUDA-Q built without TensorRT lacks it), is a registered decoder with CUDA-Q's
  parameter schema.
- **CUDA-Q QEC's `pymatching` decodes to the observables when its model has them**, as CUDA-Q
  0.8.1's plugin does: given `O`, or a detector error model's text with observables, a decode
  returns the observable flips (2.1 returned an error frame unless asked with
  `output="observables"`), and parallel edges merge as independent errors unless
  `merge_strategy` says otherwise (2.1 refused model text whose mechanisms shared a detector
  pair). `output="errors"`, this package's keyword, still asks for the error frame. With model
  text, an `O` or `error_rate_vec` the caller gives replaces the text's, as CUDA-Q's
  `make_pcm_decoder` lets the caller's keys win (2.1 refused them), and a wrongly shaped `O` gets
  CUDA-Q's message ("O must be of shape (num_observables, block_size); ..."). CUDA-Q's two
  PyMatching decoding tests of `test_dem.py` are ported.
- **Stim's shot writer, and sinter's predictions.** `write_shot_data_file` takes what Stim's
  takes: a 2-D numpy array, bool with one column per bit or uint8 bit-packed with ⌈n/8⌉
  columns, and refuses anything else with Stim's ValueError ("Expected 24 bits per shot. Got
  bit packed data (dtype=np.uint8) but data.shape[1]=24 != math.ceil(24 / 8)=3", "data must be
  a 2-dimensional numpy array with dtype=np.uint8 or dtype=np.bool_"). 2.1 read any uint8 array
  as packed and cut or padded it to the width, so `sinter.predict_observables(dets=)` with 0/1
  uint8 detection events, or `predict_observables_bit_packed` with rows of the wrong width,
  decoded other shots without an error; both raise now, as sinter does.
- **Fusion Blossom on a syndrome with no perfect matching** (an odd number of defects in a part
  of the graph with no virtual vertex, say, or a sinter model that does not match its data):
  `perfect_matching()`, `subgraph()` and the Stim and sinter decoders on every solver raise
  fusion_blossom's `PanicException` ("cannot compute final matching with unmatched outer node
  0") at once. 2.1 spun forever there, holding the GIL and deaf to Ctrl-C, as formatting the
  unmatched node for that message waited on a lock its own thread held.
- **MWPF's HTML export.** `HTMLExport.compress_content`, `Visualizer.generate_html` and the
  command line's `--visualizer-filename` gzip the visualizer data as MWPF does, with
  miniz_oxide 0.8.0's deflate at flate2's default level, byte for byte: a d=15 planar code's
  page is MWPF's 493,740 bytes, where 2.1 wrote stored (uncompressed) deflate blocks and
  1,141,236 bytes.
- **The command line streams shots.** `sample`, `detect`, `m2d`, `convert`, `sample_dem`,
  `decode`, `predict` and `count_mistakes` read, sample, decode and write 1,024 shots at a
  time, as Stim's command line does, so their memory no longer grows with `--shots` or the
  input's length (2.1 held every shot, about 10 bytes per output bit in `01`). The bytes are
  those of one pass: seeded samples are the samplers' own, and `decode --decoder relay_bp`
  seeds each batch as the whole run's. A ptb64 output whose `--shots` is not a multiple of 64
  is refused before anything is written. Bad data, or `m2d` sweep data that ends early, is
  refused when its batch is reached, after the batches before it are written.
- **`count_mistakes` is PyMatching's.** `--time` writes its two lines to stderr in PyMatching's
  format (`Decoding time per shot: infus` for no shots); 2.1 wrote them to stdout after the
  count. `--obs_in` is used whenever it is given, even with
  `--in_includes_appended_observables`. Without either, the predictions are compared with zero
  observables, as PyMatching's program compares them (2.1 refused). Extra `--obs_in` shots are
  not read, and too few is PyMatching's `Obs data ended before shot data ended.`; 2.1 broadcast
  mismatched files, printing impossible counts when either had one shot. Malformed shot data
  is still refused with this package's reader's message, not Stim's.
- **`diagram` draws a detector error model for every match graph type** (`matchgraph-3d`,
  `matchgraph-3d-html` and the `match-graph-*` names, as well as `matchgraph-svg`); 2.1 parsed
  a model as a circuit there and failed. `--remove_noise` with a match graph type is refused
  with Stim's message, where 2.1 drew a noiseless circuit's empty graph.
- **Model and circuit files are read as bytes.** A byte that is not UTF-8 (in a comment, say)
  becomes U+FFFD rather than refusing the file, in every `stabilizer-qec` command, whatever the
  locale (stdin too).
- **Licence notices.** The wheels and the sdist carry `THIRD_PARTY_CRATES.md`, the licence texts
  and notices of the Rust crates the compiled module links (PyO3, rand, wgpu, naga, objc2,
  libloading, libm and the rest, about 90, listed for each wheel's target; written from
  Cargo.lock by `tools/third_party_crates.py`), and `THIRD_PARTY_NOTICES.md` quotes HiGHS's,
  filereaderlp's and pdqsort's licences, where it cited files the wheels do not carry. It also
  reproduces the attribution of CUDA-Q QEC's NOTICE file, and names fasteval 0.2.4 (Christopher
  Sebastian, MIT), whose parser and evaluator the Azure estimator's formulas run on, with its
  notice in `estimate/LICENSE-qdk`. `CITATION.cff` gives `license: MIT` and says in its abstract
  that the package is MIT AND Apache-2.0, as CFF reads a list of licences as a choice.

### Fixed

- **Coherent arithmetic.** Corrected the inherited merged-proposal odd probability (previously
  odd/even odds) and normalized long-group parity probabilities/weights. Demonstrated valid
  long groups now remain finite in deterministic and state-vector controls. The recurrence is
  still f64 with limiting regularization; no proof over every numerical range is claimed.
  Seeds and cost can change: three shared-load CPU runs on 100,000 shots of twenty merged
  rotations took 0.261004–0.262286 s versus 0.138538–0.142096 before; the measured surface
  case's overlapping spread is not evidence of a speedup. Early provably-wide refusal is
  faster; it does not remove successful-plan compilation costs.
- **Matching state.** Stable shared-object decode/search calls lease independent mutable
  scratch; failed calls drop it. No-op merges retain configuration; edited graphs invalidate.
  Numeric batch coercion casts to uint8 before nonzero; NaN readback/noise validity follows
  the reference predicates. Correlated packed output/copies preserve ensured width. Allocation
  checks do not promise recoverable true OOM, and arbitrary concurrent mutation/load is not
  atomic. Arm64 FMA can still change probability readback in last bits; no universal immunity
  of weights or tie choices is claimed.
- **QDK rendering.** Untrusted table/report fields are escaped, optional Markdown retains a
  fixed balanced formatting subset and HTTP(S) links, and diagrams embed quoted escaped JSON
  preserving caller data. This deliberately differs from unsafe QDK 1.33.1 output. Explicit
  developer HTML wrappers remain HTML. Finite output/semantic checks are not browser/CDN safety
  certification.
- **CI and helpers.** PyMatching 2.4.0/Stim 1.16.0 references are pinned. One bounded extended
  Ubuntu/Python 3.12 job covers three designated slow corpus files, and a second runs what the
  matrix cannot install, none of it skipped: the Tsim tests against live Tsim, the sinter 1.16.0
  and relay-bp 0.2.2 differentials (both built from their sources), and PanQEC's slow tests
  (`PANQEC_RUNSLOW=1`); other matrices use reduced cases. The matrix no longer names
  `relay-bp==0.2.2` or `qldpc==0.4.1`, which it could never install. This source change does not establish remote CI success. Valid scalar qsample comb
  results again retain numpy.float64 rather than Python float.

- **Tsim.**
  - `to_matrix` gave the wrong shape for noise before a reset (`Z_ERROR(0.1) 1` then `R 1`
    gave a 2×1 matrix) and for `II` (`II 0 1` gave 1×1), and, in a circuit without
    measurements, a factor of 2 too few for each target-less `DETECTOR` and each of
    `OBSERVABLE_INCLUDE(k)`'s k + 1 observables (since 2.1.0); `OBSERVABLE_INCLUDE(0) X0` gave
    the identity, where Tsim raises.
  - A qubit named only by noise and a product's identity letter, or by noise and a product that
    cancels on it (`SPP X0*Z1*Z1` with noise on qubit 1), was dropped from `to_matrix` (2×2
    where Tsim's is 4×4).
  - Hand-written held forms are read as the core reads them: `pauli=X_Z` (the core's `_`
    identity), a tag's empty arguments (`theta=0.1,`) and `I[T()]` gave a wrong detector error
    model silently and broke `to_matrix`, `get_graph`, `tcount` and printing, and
    `II[R_PAULI(theta=t, pauli=...)]` was dropped, a wrong model and matrix, silently.
  - `SPP_DAG[R_PAULI(theta=a*pi)]` and `!` factors are kept as Tsim keeps them: in printing, in
    `inverse` (Tsim's `SPP_DAG[...]` text), in fusion with neighbours and in `R_PAULI`'s
    limits and messages. A native `II[R_ZZ(theta=pi/8)]` prints as `TPP Z0*Z1`; it printed
    `R_ZZ(0.25) 0 1`, which reads back as `R_PAULI`.
  - An all-identity product rotation is left out of `str()`; Tsim read its raw line as a plain
    `II`, another matrix, or refused it.
  - The inverse of a pair rotation (`II[R_ZZ(theta=t)] ...`, and `R_XX`, `R_YY`) or of
    `II[R_PAULI(...)]` is one instruction, printed as Tsim's inverse:
    `c.inverse().inverse() == c`, and `len(c.inverse())` is Tsim's.
  - `to_matrix` raises Tsim's errors for an `R_PAULI` naming a qubit twice or more than 64
    qubits, an anti-Hermitian product and sweep bits, and is Tsim's for `TPP Z0*Z0` ([[1]]) and
    for products naming a qubit twice. `SPP[T] X0 * Y1` no longer raises `IndexError`.
  - `CompiledStateProbs` conditions on every record bit (since 2.1.0). Before, the
    stabilizer-rank engine ignored `MPAD` bits and herald bits and matched each later
    measurement against the wrong bit, so probabilities over all outcomes summed to 2 and
    impossible records got probability 1 (`MPAD 0` gave 1 for both outcomes). An `MPAD` flip and
    a herald are now noise drawn with the rest, as Tsim draws them, and a draw's probabilities
    sum to 1. A reset's unread outcome is summed over, as in Tsim, where the qubit is entangled
    with the rest. A reset of a qubit on its own, such as a fresh qubit's `RX`, leaves the same
    state either way, so it costs one run and is never drawn. Before, the outcome was drawn
    once, so a noiseless circuit that resets an entangled qubit (`H 0`, `CX 0 1`, `R 0`, `M 1`)
    gave 0 or 1 depending on the seed, where the truth and Tsim's answer are 0.5.
  - `compile_sampler`, `compile_detector_sampler` and `CompiledStateProbs` refuse what Tsim's
    graph builder refuses, with Tsim's messages, as `to_matrix` and `tcount` already did: an
    anti-Hermitian product (`TPP Y0*X0`), an `R_PAULI` that names a qubit twice or more than 64
    qubits (33 `R_ZZ` lines in a row), and sweep bits. Before, they sampled such circuits. A
    record entry from before the first measurement raises `MeasurementRecordError` (an
    `IndexError`, as in Tsim) from every sampler. Before, the samplers raised a plain
    `ValueError`.
  - Parametric tags are read with Tsim's grammar. A tag of the form `name(...)` that Tsim
    refuses raises Tsim's message, prefixed with the line number. Examples: an extra or unknown
    key (`I[R_Z(theta=0.3*pi, phi=0.1*pi)]`, `U3(..., mu=0.1*pi)`), a missing key
    (`I[U3(theta=0.1*pi)]`, which gave a bare `KeyError`), and a literal that is not a decimal
    number times pi (`theta=0.3*pi*pi`, `theta=1/3*pi`, `theta=abc*pi`, `theta = 0.3*pi`).
    Before, these were read as rotations, or crashed with `KeyError` or "could not convert
    string to float". A half-turn tag that Tsim does not read as a rotation (`I[R_Z (theta=0.5*pi)] 0`,
    `I[ R_Z(...)] 0`, `I[R_Z(...) ] 0`, `I[R_Z(theta=0.3*pi)#x] 0`) is the identity, as in
    Tsim, and `SPP[ R_PAULI(...)] X0` is Tsim's `SPP`. Before, both were rotations. A `#` inside
    a tag's brackets is part of the tag, as Stim reads it, not the start of a comment.
- **Stim.** `Circuit.inverse()` of `II[R_PAULI(theta=t, pauli=...)]` keeps each letter on its
  qubit. It reversed the letters whole while reversing II's targets pair by pair, so
  `c + c.inverse()` was not the identity.
- **Coherent twirl.** `Circuit.twirled()` (merge=False) repeated a tagged gate of several
  targets once per target: `I[T] 0 1` came out as `I[T] 0 1` twice (S on each qubit) and
  `I[R_Z(theta=0.1)] 0 1 2` three times, silently (also `I[T] 0` and `I[T] 1` on separate lines,
  which the parser fuses, and a tagged channel such as `I_ERROR[LEAK(p=0.1)] 0 1`). Each gate is
  now kept once, beside its errors twirled target by target, so a circuit with no coherent error
  twirls to itself (since 2.0.0).
- **Merged twirl.** `twirled(merge=True)` read a tagged gate at a Clifford angle
  (`I[U3(theta=π/2, phi=0, lambda=π)]`, `I[R_Z(theta=π/4)]`) as a non-Clifford location rather
  than the Clifford gate it is, so the rotations either side of it that are one fault were not
  merged and the "coherence-aware" model was the plain twirl. It now reads them as the coherent
  samplers do, and gives the plain-gate circuit's merged errors. A non-Clifford gate between
  errors of its own axis was counted twice: `R_X(0.1)`, `I[R_X(0.3)]`, `R_X(0.2)` merged to one
  error of sin²(0.6) while the gate also ran. The errors now merge to sin²(0.3) beside the gate.
  Errors either side of a gate they anticommute with (Z either side of R_X, X either side of T)
  stay apart, and a gate on other qubits (a T layer between a fault's places) does not stop the
  merge.
- **Time reversal.** `Circuit.time_reversed_for_flows` reversed a tagged `I`/`II` gate's
  targets but kept its tag, so `I[R_PAULI(theta=π/4, pauli=XZ)] 0 1` became `... pauli=XZ)] 1 0`
  (the letters on other qubits) and `I[T]`, rotation angles and `U3` were not inverted. The tag is
  now inverted as `Circuit.inverse()` inverts it (`I[R_PAULI(theta=-π/4, pauli=ZX)] 1 0`,
  `I[T_DAG]`); a coherent error is negated as `inverse()` negates it
  (`I_ERROR[R_X(theta=-0.1)]`, the same twirl), and a channel with no inverse (damping,
  leakage) and a tag that is not ours are kept. This text differs from Stim's, which treats
  every tag as inert.
- **Coherent sampler order.** `compile_coherent_sampler(order=0)` (and the joint sampler's)
  enumerated every connected product of each cluster, exponential in its size, and hung on a
  truncated kernel (d = 3, 40 rounds, `max_contraction=0`). The core now takes 0 as 1, as the
  Rust builder's `order` already did.
- **Copies keep their class.** `pickle`, `copy` and `deepcopy` rebuild a `Circuit`,
  `DetectorErrorModel`, `PauliString`, `Tableau` or `Matching` as its own class, as Stim's
  and PyMatching's do. A subclass stays one, rebuilt without calling its constructor (as
  Stim's), so a subclass whose constructor takes other arguments than the text copies too. A
  `Matching` subclass keeps its overrides and its attributes; the other classes keep none of
  the instance's own attributes, as in Stim. Deltakit's circuit
  (`stabilizer_qec.deltakit_stim`) keeps its leakage, so its copies sample and model it.
  Before, every copy was the base class. A Deltakit circuit sent to `RunAllAnalysisEngine`'s
  process pool (its default of 16 processes) therefore lost its leakage there, and the engine
  counted 0 failures where a serial run counts about 3%. Pickles written by 2.1 still load.
- **Reference signs of tagged Clifford-angle gates.** `reference_detector_and_observable_signs`
  and `count_determined_measurements` read a tagged gate at a Clifford angle
  (`I[R_X(theta=π/2)] 0`) as the Clifford gate it is, as `reference_sample` and the samplers
  do. `compile_m2d_converter(skip_reference_sample=True)` on such a circuit now gives raw
  parities, where it gave the reference-relative events.
- **A `#` in a model's tag.** `DetectorErrorModel` reads `error[t#g](0.1) D0` as Stim does,
  with the tag whole. Before, `len`, iteration, indexing, `repr`, `rounded`, `+`,
  `approx_equals` and `get_detector_coordinates` raised `Failed to parse as a stim.DemTarget`
  on such a model, and so `pymatching animate` refused it.
- **Indexing is linear.** A `Circuit` or `DetectorErrorModel` parses its instructions once and
  keeps them until it changes, so `len(c)` and `c[i]` in a loop are O(1) each, as Stim's.
  Before, each call printed and re-parsed the whole object. Indexing all 1,600 items took
  2.1 s for a circuit and 9.5 s for a model; it now takes under 0.01 s. QUITS's
  `check_overlapping_CX` and MQT's `collect_circuit_layers` loop this way.
- **A stray byte in a file.** `Circuit.from_file` and `DetectorErrorModel.from_file` (and
  through them `Matching.from_detector_error_model_file` and `from_stim_circuit_file`) read a
  byte that is not UTF-8 (`caf\xe9` in a comment) as U+FFFD, not refusing the file. Stim skips
  such bytes in comments. They also take a file opened in binary mode.
- **WDIMACS with a certain error.** `likeliest_error_sat_problem` writes Stim 1.16's problem for
  a model with a probability-1 error, where it raised `math domain error`.
- **The exact distance.** `DetectorErrorModel.distance()` (the integer program) leaves out
  faults of probability 0, as the graph-like and search distances and Stim do. It reads the
  model's own faults, so a tag holding `)` no longer breaks it. Before, it could prove a
  distance below the true one (1 for `error(0) L0` beside a distance-2 model), or raise.
- **Heavy graph edges.** `Matching` built from a networkx or rustworkx graph leaves out an edge
  whose weight is past 2^24 − 1, with PyMatching's `UserWarning`, and decodes the rest. Before,
  it kept the edge, and then every decode raised. The warning's weight is printed as
  PyMatching prints it (`Weight 20000000.000000 of edge (1, 2) ...`), for `add_edge` too.
- **Docstring examples.** Every example the API reference publishes runs in
  `tests/test_docs.py`, found from `api/`'s pages and the modules defining their members. The
  `Matching` example now prints `array([0, 1, 0], dtype=uint8)` and the `PauliString` one
  `stabilizer_qec.PauliString("+i_ZZ")`, the answers PyMatching and Stim give. The `analysis`
  examples import what they use.
- **Deltakit's Stim.** Deltakit's `Circuit.from_stim_circuit` (and `parse_stim_circuit`
  through it) reads this package's own `stabilizer_qec.Circuit`, which a pickled Deltakit
  circuit comes back as, where it read an empty circuit and an empty decoding graph without
  error. A `deltakit_stim` circuit's diagrams draw Deltakit's instructions (`LEAKAGE(0.1)`,
  `RL`, a herald as `HERALD_LEAKAGE_EVENT(0.01):rec[k]` on the qubit it reads, a transport
  gate's four arguments on both its ends), where they drew the lowered `I_ERROR` and `MPAD`
  lines, a second qubit's herald on qubit 0; `flattened_operations` lists `("LEAKAGE", [0],
  0.1)`, not `("I_ERROR", [0], 0)`; `to_tableau` and `time_reversed_for_flows` refuse
  `LEAKAGE`, `RELAX`, `RL` and heralds with deltakit_stim's messages; `approx_equals` compares
  their probabilities; and `explain_detector_error_model_errors` explains only Stim's own
  noise, as deltakit_stim does, so Deltakit's `parse_explained_dem` reads it (it raised
  `TypeError: Unsupported gate type: MPAD`). A tag on Deltakit's own instructions
  (`LEAKAGE[site](0.1) 0`, which deltakit_stim and `deltakit.circuit`'s tagged `Leakage` and
  `Relax` write) is still refused with `ValueError`: the lowered line's tag is where the
  instruction is held.
- **Tesseract.** An empty answer's cost is 0, as Tesseract's `cost_from_errors` gives it, not
  −0.
- **ldpc.**
  - A pickled or copied `BpDecoder`, `BpOsdDecoder`, `BpLsdDecoder` or `BeliefFindDecoder`
    decoded with its constructor's settings where one had been changed
    since (`lsd_method`, `lsd_order`, `osd_order`, `serial_schedule_order`,
    `random_schedule_seed`, `set_do_stats`), while it reported the new ones; a copied
    `LsdDecoder` lost its `lsd_method` and `lsd_order` (since 2.1.0). A copy now has them, and
    the serial schedule's state as the decodes so far left it (a random schedule's
    generator, a relative schedule's order), so it decodes on as the original does.
  - One decoder decoding from several threads could answer a decode with another thread's
    correction, silently (8 threads at a 1 µs switch interval: about 13% of `BpDecoder`'s
    decodes, 4% of `BeliefFindDecoder`'s; since 2.1.0). Each decode keeps its results on the
    decoder and answered from them. Decodes on one decoder of the ldpc family now take turns,
    as ldpc's do under the GIL.
  - On Linux, OSD's column sort could panic ("user-provided comparison function does not
    correctly implement a total order"), or order the columns otherwise than glibc's `qsort`
    (ldpc's there), when BP's posteriors held NaN: product-sum BP gives them on a model with
    `error(0)` or `error(1)` faults (since 2.1.0). The sort is glibc's merge sort (msort.c)
    now, which takes any comparison.
- **relay_bp.**
  - A decoder or `ObservableDecoderRunner` shared between threads raised "Already borrowed"
    when two calls overlapped (since 2.1.0). Calls now wait their turn, as relay_bp's do.
  - The sinter decoders (`relay-bp`, `mem-bp`, `msl-bp`) on a model with a fault of
    probability 1 predicted wrongly on nearly every shot, silently (logical error rate 0.9985
    where 0.0015 is right; since 2.1.0). The default pruning built the biases as `np.matrix`
    values, and the observables' from the check matrix's rows. Both are flat arrays now, the
    observables' from the observables' matrix. relay_bp 0.2.2 raises `TypeError` there.
  - A `uint8` CSR or CSC check matrix storing an entry twice was read with that entry set
    (1 + 1 taken as 1, not 0), so `success` could be True for a correction whose real checks
    are not the syndrome (since 2.1.0). The entry is now the matrix's sum, 0 mod 2, as with
    any stored values; relay_bp 0.2.2 panics on such a matrix.
- **PreDecoder.** A pickled or copied pre-decoder had its constructor's `threshold` (0.5 by
  default) where one had been set since, so it applied other faults (about 4 times as many
  on the distance-5 memory at threshold 0.95) and decoded some shots otherwise. It also lost
  the optimizer's state and `steps`, so trained on, it took other steps (since 2.1.0). A
  copy now has the threshold and either engine's optimizer state.
- **Real-time decoding** (since 2.0.0). `realtime.make_decoder(name, circuit=...)`, and so
  `stabilizer-qec serve --circuit` and `DecoderClient.spawn(circuit=...)`, builds the circuit's
  model as `stabilizer_qec.sinter.Sampler(name)` does: decomposed only for the decoders that
  match on a graph, with `approximate_disjoint_errors=True`. It decomposed it for
  `color_matching` and `ambiguity_clustering` too, so a distance-5 colour code could not be
  served with `color_matching` (its faults do not split into graph-like pieces) and ambiguity
  clustering decoded another model than sinter's (31 of 3,000 shots differed on a d=3 surface
  code). A `DecoderQueue` batch whose decode raised is collected by raising its error once: it
  stayed in the queue, so every later `get_corrections()` raised the same error, `pending()`
  never fell, and the decoder server kept each rejected `DECODE`'s syndromes (200 rejected
  1 MiB frames grew it from 62 to 565 MB; it now levels off at 155 MB). A `DecoderClient`
  request interrupted between sending and reading its reply (Ctrl-C) drops the connection, and
  later calls raise `ConnectionError`; the next call of the same size returned the interrupted
  batch's predictions. `stabilizer-qec serve`'s default `--threads 0` reaches the decoder as 0,
  every core, as its help says; it was left out, so each batch decoded on one thread.
- **qecsim, PanQEC, PECOS and Loom ports.**
  - qecsim's Blossom V interface numbered its nodes in a set's order. The planar CMWPM and the
    rotated planar and rotated toric SMWPM decoders hash their nodes by identity, so the choice
    among equal-weight matchings followed memory addresses, and a seeded `stabilizer-qec qecsim
    run` or `run-ftp` gave a different result in each process (`rotated_planar(4,4)`,
    `rotated_planar.smwpm`, `-s5`: n_fail 113 to 126). The nodes are now numbered in order of
    first appearance, so the same seed gives the same result every time. Ties can still be
    broken differently from qecsim's NetworkX matching.
  - `stabilizer-qec qecsim`, and any decode through qecsim's `graphtools.mwpm`, read
    `logging_qecsim.ini` and `clib/libpypm.so` from the working directory first, as qecsim
    1.0b9 does. Python evaluates the first file's handler arguments, and the second is loaded as
    a library in place of the matcher, so running in a directory someone else supplied ran that
    directory's code. Both files are now read from `$QECSIM_CFG` or `~/.qecsim` only, and an
    empty `QECSIM_CFG` (which qecsim took as the working directory, and which a CI `env` entry
    for an undefined variable gives) counts as unset. A relative `QECSIM_CFG` is still read
    relative to the working directory, as in qecsim (SECURITY.md).
  - PanQEC's BP-OSD decoder took each sector's correction from ldpc's `osdw_decoding`, which
    ldpc 2 leaves holding the previous shot's correction whenever a sector's syndrome is zero. A
    decoder used for many shots (`DirectSimulation`, `generate-input`'s default decoder, and the
    X-cube decoder's Z part) therefore reapplied a stale correction that did not explain the
    syndrome: 216 of 300 logical failures on `Toric3DCode(3)` at p = 0.01, against 0 now. It
    now uses `decode`'s return value. PanQEC 0.1.7 has the same bug on ldpc 2. Its own
    single-error test for this decoder, which was marked as an expected failure, now passes.
  - PECOS's `SparseSim.refactor` (both the default simulator and `pySparseSim`) ignored the
    -1s that come from reordering the generators it multiplies together. In about one call in
    five this flipped the new generator's sign, and later measurements then gave outcomes the
    state cannot produce. The signed stabilizer group is now unchanged by `refactor`.
  - Loom's `Eka` is a frozen dataclass, as Loom's is: `Eka.fromdict(e.asdict()) == e`, its
    fields cannot be assigned (assigning `blocks` used to skip its lattice and label checks),
    and it has Loom's repr. `AuxCNOT` accepts exactly two block names, as Loom does. With
    three names it used to drop the third without warning, and with one it failed later with
    `IndexError`.
- **qsample.**
  - Each `Protocol` built without check functions has its own dict. qsample's default was one
    dict shared by them all, so a second `NoisyDFTStatePrepSimulator` rebound the first one's
    decoder and checks: a |+> simulator's p_L at p = 0.02 went from 0.022 to 0.078 once a |0>
    one was built.
  - Subset probabilities stay finite for any number of fault locations. A group of 1030 or more
    raised OverflowError (qsample gives NaN from 171); terms whose coefficient is beyond a
    float, or whose power falls below the normal floats (`binom(200, 1029, 0.02)` was 0, not
    3.3e-129), come from Loader's saddle-point method, as R's `dbinom`. `comb` gives inf where
    the coefficient is beyond a float.
  - `save` (and `DirectSampler.save`, `SubsetSampler.save`) uses dill when it is installed, as
    qsample does, so samplers with lambda or nested check functions save; without dill it uses
    pickle, which takes only top-level functions. It writes a new file beside the target and
    replaces the target only once written whole: a failed save no longer leaves an existing
    file at 0 bytes.
- **MQT QECC.** Every `circuit_utils` function that takes a circuit, each noise model's
  `apply`, `CatStatePreparationExperiment` and the NDFT simulators take Stim's own circuits;
  they raised AssertionError or TypeError, though 2.1.0 said every port took them. Adding this
  package's `Circuit` and Stim's with `+` still raises TypeError: convert with
  `stabilizer_qec.Circuit(str(c))` first.
- **qLDPC.**
  - `as_noiseless_circuit`, `get_state_stabilizers`, `get_logical_state_stabilizers`,
    `get_state_prep_diagnostic_circuit`/`_tasks`, `QuditCode.deformed`/`CSSCode.deformed`,
    `get_logical_tableau`, `get_encoder_and_decoder`, `get_pauli_product_measurements` and
    `get_transversal_circuit(s)` take Stim's own `Circuit`, `Tableau` and `PauliString`; they
    raised, though 2.1.0 said every port took Stim's objects. Adding this package's `Circuit`
    and Stim's with `+` still raises TypeError: convert with `stabilizer_qec.Circuit(str(c))`
    first.
  - `CSSCode.reduce_logical_op(s)` and the SL, PSL, GL and PGL lifts run on NumPy before 2.1
    (qLDPC's `np.reshape(..., copy=False)` needs 2.1, its `np.vstack(..., dtype=int)` 1.24); the
    port's suite passes on NumPy 1.23.5. The `qldpc` extra now asks for qLDPC's own NumPy floor
    (`numpy>=1.24` on Python 3.11 and later).
- **qec-util's tools** (`stabilizer_qec.analysis`).
  - `dems.detectors_to_observables` and `dem_instrs.detectors_to_observables` apply a mapping
    keyed by Stim's own `DemTarget`s, as qec-util does. Stim's targets neither equal nor hash
    like this package's, so such a mapping matched nothing and the model came back unchanged.
  - `samplers.merge_batches_in_file` keeps every extra metric. It wrote `seconds` into the
    header a second time (as qec-util 0.5.0 does), so the merged row was one value short and,
    read back, a metric named after `seconds` lost its count to `seconds`.
  - `distance.get_upper_bound_circuit_distance` decodes over dem_decoders' check matrix (errors
    flipping the same detectors merged, their probabilities combined), so its bound is
    qec-util's (3 against qec-util's 2 on a decomposed d = 3 color code before), and returns
    the decoded errors, which flip observables and no detector. It (like qec-util) took the
    fault indices as indices of every instruction of the model, so it could return detector
    lines or errors that flip detectors.
  - Without scipy, threshold fits are MINPACK's Levenberg-Marquardt, as scipy's `curve_fit`:
    they stop where scipy's do (poly-4's threshold was 0.042 of a standard error off, and
    tanh took 400,000 evaluations against scipy's 7,000), and report scipy's infinite error,
    with its warning, where the fit cannot fix the threshold (poly-9 gave a finite interval,
    all-zero failures NaN).
  - Without lmfit, `LogicalErrorProbDecayModel.fit` fits as lmfit does (MINPACK, with lmfit's
    options and standard errors) and takes lmfit's arguments: `weights` and `max_nfev` were
    ignored, as were unknown keywords (lmfit warns), and too few points or NaN data gave a
    fit or a LinAlgError where lmfit raises TypeError or ValueError.
- **Azure estimates on degenerate parameters.** A crossing prefactor of 0 (or one whose logical
  error rate underflows) gives the QDK's estimate: its f64 quotient and saturating cast allow
  every cycle. A division by zero (a logical cycle time that rounds to 0 ns, a T factory that
  cannot finish a run under `maxTFactories`, a patch of 0 physical qubits) and a report that
  cannot format the infinite clock frequency of a 0 ns cycle are each a `ValueError`, where the
  QDK panics (a batch fails as a whole, as the QDK's call does). 2.1 raised a bare
  `ZeroDivisionError` or `OverflowError` on all of these.
- **tqec's engine at a NumPy-integer distance.** `compile_block_graph(g, np.int64(3),
  block_temporal_height=5)` and `BlockGraph.compile` at such a distance ran the default rounds,
  dropping the height (and a callable or `LinearFunction` one), silently; the distance is now
  read as an integer, and the height honoured.
- **2.1.0's "Not ported" list** said QUITS's distance search used more memory and time than
  Stim's: 456 MB and 8.7 s against 139 MB and 1.6 s on [[72, 8, 8]]. That was the search before
  it took Stim's steps (f3dc7ac, which is in 2.1.0). 2.1.0's own search used a 228 to 241 MB
  peak footprint and 0.79 to 0.81 s of user time on that circuit, against Stim's 530 MB and
  4.25 to 4.54 s (PYTHONHASHSEED 0, the whole process, three runs each).

### Not ported

These remain, each in the README and the inventory:

- **Loom's if/else circuits** (`IfElseCircuit` and its `BoolOp` conditions). The file that
  defines them in el-loom 0.4.0, `src/loom/eka/ifelse_circuit.py`, alone among Loom's files
  carries a notice that prohibits its use, distribution and reproduction without Entropica Labs'
  express written consent, in place of the Apache-2.0 header. A port, and fixtures recorded by
  running it, need that consent.
- **tqec's Y cubes.** tqec 0.2.0 has none: it raises `NotImplementedError("Y cube is not
  implemented.")`, and so does this package.
- **Tsim's shots, bit for bit.** Tsim draws its noise with numpy's PCG64, channel by simplified
  channel, and its outcomes with JAX's threefry, from probabilities whose last bits follow
  pyzx_param's rewrite order and XLA's complex64 sums; Tsim itself is reproducible only at a
  fixed `batch_size`. The shots here are this package's, equal in distribution.
- **The QDK's live widgets.** The space and scatter charts are drawn as static SVG from
  qsharp_widgets' own functions, not by the widget bundle: no watermark, the overview table
  static (its first five columns, no sorting or selection), and a batch's failed items left out
  of the scatter.
- **CUDA-Q QEC's TensorRT engines, and `dispatch` and `transport`.** `trt_decoder` runs the ONNX
  model on the CPU and refuses `engine_load_path` and `engine_save_path` by name; `dispatch` and
  `transport` are kept and written, and ignored on the host, as CUDA-Q's host library ignores
  them.
- **PyMatching's ties outside its API.** Correlated matching (`enable_correlations=True`),
  `BeliefMatching`, `WindowMatching` and `stream_memory`, `LeakageAwareDecoder`,
  `ArgumentReweighting` and `PreDecoder` (unless `reweight=False`) with `decoder="matching"`,
  their default, and the Rust crate's `Matching` decode on this package's own sparse blossom
  (`src/sparse`), whose answer among equally light matchings can differ from PyMatching's (the
  README lists what decodes on which).
- **Chromobius tied lifting.** Chromobius walks a table in `std::unordered_map` order, which is the
  C++ library's own: with libstdc++ it may drag charge another way on some shots. Off macOS
  the sweep requires 99% agreement and some facade cases require 95%; equal-weight lifting can
  differ on macOS too.
- **The coherent sampler past its budget, and the joint sampler past twelve generators**, which
  truncate the class sums as 2.1 did. The sweep's d ≥ 5 rows and the crosstalk's d = 5 and 7
  rows keep 1.8.0's truncated measurements as pre-correction archive data, not current validation.

## 2.1.0 — 2026-10-20

An audit of every port against its original's own documented examples, and the gaps it found,
closed. Each port was run on the examples, tutorials and command lines its original documents,
on Stim's own objects as well as this package's, and on Python 3.9 to 3.14 and Windows; what
differed or was missing is fixed below, and what is left is listed in the README and the
inventory (`docs/superpowers/notes/2026-10-09-parity-inventory.md`). The Python package and the
Rust crate both at 2.1.0. Behaviour changes are under **Changed**.

### Added

- **Stim.** `search_for_undetectable_logical_errors` now takes Stim's own steps, and finds Stim's
  faults exactly: the same terms on 3,804 random models, the colour, surface and bivariate
  bicycle codes and BPC72, and Stim's messages where both fail. It is also faster than Stim:
  0.07 s on BPC72 against Stim's 0.25 s (2.0 took 8.1 s), and 0.6 s against 4.0 s on the d = 5
  surface code at limits (6, 6). Limits of (2, 2) go to the graph-like search, as in Stim. The
  CLI's `diagram` takes `--filter_coords`, and `help <topic>` prints Stim's pages (gates,
  formats, commands, indexes, markdown). `matchgraph-svg` draws a model's undecomposed
  hyperedges.
- **PyMatching.** The CLI's `predict` reads b8 by default, as PyMatching's does.
  `stabilizer_qec.pymatching.__version__` is "2.4.0".
- **sinter.** `mw_parity_factor` and `hypergraph_union_find` run on this package's MWPF.
- **Stim's objects in every port.** The qec-util, qLDPC, MQT QECC, MWPF, Deltakit, sinter and
  Relay-BP ports take Stim's own `Circuit`, `DetectorErrorModel` and `DemInstruction`, read
  exactly. Before, they raised errors, and sinter's Fusion Blossom and MWPF decoders silently
  dropped every error of a Stim model.
  MWPF's `HeraldedDetectorErrorModel.of`, `RefCircuit.of` and `with_circuit` take them too, and
  the Tesseract port gives Stim's models back where it was given Stim's.
- **stimbposd.** `stabilizer_qec.stimbposd` is stimbposd 0.2.0 whole: `BPOSD`, `BPLSD`, the
  sinter decoders, `sinter_decoders()` (eight variants) and
  `detector_error_model_to_check_matrices`, bit-identical on ldpc 2.4.1. Use it with
  `--custom_decoders_module_function stabilizer_qec.stimbposd:sinter_decoders`.
- **Relay-BP.** `ObservableDecoderRunner(parallel=True)` and `SinterDecoder_RelayBP(parallel=True)`
  decode on threads without the GIL: 4.5× serial on the gross code, where 2.0's parallel run
  was slower than serial. Results are bit-identical to relay_bp. With `logging=True` the runner
  and sinter decoders write `relay_logging.out`. `insert_uniform_academic_circuit_noise` takes
  real Stim circuits with `REPEAT`.
- **ldpc.** `BpOsdDecoder` and `BpLsdDecoder` take ldpc's `schedule=` and
  `random_schedule_seed=`. `lsd_method` and `lsd_order` can be set after construction,
  `OSD_OFF` included. `input_vector_type="received_vector"` on BP+OSD, BP+LSD and BeliefFind
  matches ldpc's decodings and iteration counts. ldpc's exceptions match in type and message,
  and the v1 classes enforce Cython's argument types.
- **CUDA-Q QEC real-time decoding.**
  - Configuration: `decoder_config`, `multi_decoder_config`, CUDA-Q's YAML (read without a new
    dependency), `configure_decoders` (and `_from_file`, `_from_str`), `finalize_decoders` and
    status codes.
  - Decoding: `enqueue_syndromes(id, syndromes, tag)` and `get_corrections(id, return_size,
    reset)`, sliding windows and `stim_dem_path`. The ticket API is kept.
  - Schemas: `decoder_param_schema`, `registered_decoder_schemas`, `validate_custom_args`,
    JSON Schema export, and the deprecated typed configuration classes.
  - Decoders under CUDA-Q's names: nv-qldpc-decoder, chromobius and nv-fusion-decoder. A
    setting that isn't supported is refused by name.
  - Helpers: `dem_sampling` (bit for bit with CUDA-Q's CPU sampler), `relay_solutions`, `patch`,
    `pcm_from_sparse_vec` and `generate_timelike_sparse_detector_matrix`.
  - `KernelNotRunWarning` for Python `@code` with registered kernels.
- **Tsim.** `tsim.Circuit` is a real subclass of `sq.Circuit`. It has:
  - Tsim's `append` (in half-turns), `append_from_stim_program_text`, `from_file`, `+` with Stim's
    circuits and this package's;
  - `from_stim_program`, `stim_circuit`, `cast_to_stim` and `without_annotations`;
  - `CompiledStateProbs`, and `to_matrix` by Tsim's rules;
  - diagrams that label tagged gates, with Tsim's height, width and zoomable HTML wrapper.
- **tqec.** tqec's call shape: `compile_block_graph(g, convention=...).generate_stim_circuit(k=...,
  noise_model=NoiseModel.uniform_depolarizing(p))` at distance 2k + 1. `NoiseModel`
  (`uniform_depolarizing`, `si1000`) gives tqec's noisy circuits instruction for instruction,
  and `BlockGraph.compile` takes `k=` and `noise_model=`. Block graphs that reuse a tile at
  different times now compile (two CNOTs in a row), and so do pass-through cubes drawn as
  spatial. New: `stabilizer_qec.tqec.simulation` (`start_simulation_using_sinter`,
  `binary_search_threshold`, plots, CSV) on this package's sinter, and
  `stabilizer_qec.tqec.interop.pyzx` (PyZX graphs in and out, `PositionedZX`, `block_synthesis`,
  Pauli webs; PyZX optional). `sq.tqec` loads on first use.
- **Loom.** Loom's operations: transversal Hadamard, state injection, auxiliary CNOT, move
  block, move corners, the Y-wall phase gate and rotate block. Its error models
  (`CircuitErrorModel`, `HomogeneousTimeIndependentCEM`, `HomogeneousTimeDependentCEM`,
  `AsymmetricDepolarizeCEM`) and their noise in the Stim export; the Mimiq and Guppy exporters.
  The repetition, Shor, Steane and five-qubit code factories, with the repetition chain's grow,
  shrink, merge and split. The `Executor` and `TargetLanguage`; the Mimiq, CUDA-Q and Guppy
  run-outcome parsers; `stim_polygons` and the crumble printer; Loom's module paths. Each is
  checked against fixtures recorded from Loom 0.4.0.
- **MQT QECC.**
  - The analog-information simulators take `decoding_method="bposd"` again.
  - The colour-code decoder runs as `stabilizer-qec mqt.qecc.cc-decoder` or
    `python -m stabilizer_qec.mqt_qecc.cc_decoder`.
  - `NoisyDFTStatePrepSimulator` runs on this package's qsample at fault-tolerant rates. CNOTs
    are read as (control, target), decoding measures each stabilizer, and a wrong verification
    order is refused. Upstream's simulator never ran.
- **PanQEC and qecsim.** `stabilizer-qec panqec ...` and `stabilizer-qec qecsim ...` run their
  command lines; PanQEC's generated cluster scripts call them, and `stabilizer-qec qecsim run`
  picks up qecsim extension packages' entry points. qecsim's Blossom V stand-in stays exact past
  512 nodes by matching with Fusion Blossom: the 13×13 `run-ftp` example takes 40 s instead of
  timing out.
- **PECOS.** `SparseSim.print_stabs(print_destabs=, print_y=)`, `copy`, `refactor`, `find_stab`,
  `run_direct` and `'G2'`.
- **Tesseract.** Its binding's behaviour: constructor overloads (`TesseractConfig(7)`),
  pybind11's enums, the binding's exception types, and output where Tesseract's goes
  (`ostream_redirect`, `Visualizer.write`). The decoder's config is read at every decode, as
  Tesseract's is, except the sparsify settings.
- **Chromobius.** Faults of probability above 1/2 decode as Chromobius's negative-weight
  edges, with its predictions and weights; probability 1 is refused, as Chromobius's weights are
  NaN there. The sinter decoder takes any integer data, and `decode_via_files` reads short
  files, partial records and other record widths as Chromobius does.
- **Deltakit.** `deltakit.decode` carries Deltakit's cloud decoders' names.
- **Resource estimates.**
  - Azure: the report (`status`, the `jobParams` echo, `physicalCountsFormatted`, `reportData`),
    `format_metric_prefix`, `format_duration` and `LogicalCounts.estimate`.
    `azure_estimate` returns an `EstimatorResult`; a frontier's `result["physicalCounts"]` is
    its first entry's.
  - Azure: `EstimatorError`, a `ValueError` with the QDK's `.code` and `.message`, for every
    refusal, in the QDK's order. Unreadable input raises `QSharpError` with serde's reason and
    column. Both match the QDK on 286 recorded inputs.
  - Azure: `LogicalCounts({...})` takes the QDK's dictionary. A `qecScheme` without a name is a
    custom scheme.
  - Qualtran: `stabilizer_qec.estimate.qualtran` is a package with Qualtran's fifteen
    submodules, so Qualtran's module paths work. Its UI helpers run without Dash, plotly or
    pandas, with figures as plotly dictionaries. They equal Qualtran 0.7.0 on 448 recorded
    inputs. `AlgorithmSummary.from_bloq`, `NoisyPauliRotation`, `storage_error`,
    `GateCounts.to_legacy_t_complexity` and `TComplexity` are also in.
- **The exact and stabilizer-rank samplers** have a `.device`.
- **qsample.** Drawing and plotting raise an `ImportError` that names the missing optional
  dependency.

### Changed

- **The coherent and joint samplers complete the circuit's spacetime kernel.** One elimination
  over the rotations finds what the local search misses: a logical error at one time with the
  same logical at another, whose amplitudes add.
  - A d = 3 repetition memory with one over-rotation per round was read at the twirl's rate of
    undetected logical errors: 0.00103 against the exact 0.00203, −44σ. It now reads
    0.00206 ± 0.00005.
  - With at most twelve generators the coherent sampler sums the whole class, so its weights
    are exact (`exact_weights`). Past that, a shot whose faults reach a non-local generator
    sums its elements; `num_nonlocal_generators` counts those generators.
  - `compile_coherent_sampler(max_complete=50_000)` bounds the completion. Past it the sampler
    warns that interference between distant faults is left out, and `kernel_complete` is False.
  - Circuits with an over-rotation after every tick (the README's sweeps, d = 3 to 11) and the
    distillation circuits already had complete local kernels. Their results are unchanged.
- **ldpc's BP+OSD now equals ldpc's on every shot checked.** OSD weighs ln(1/p) in ldpc's order,
  and macOS's qsort is emulated exactly. On the gross code, 13 of 3,072 shots differed; none do
  now (the README paragraph and `tools/gross_ops.py` are corrected). The audit's tie hunt went
  from 118 of 16,200 to none, and the QUITS sweep from 4 to none. One BP+LSD shot in 600 still
  differs, from ldpc's cluster merge order, which follows hashed pointers.
- **tqec.** `compile_block_graph(..., observables=None)` now means no observables, as in tqec.
  The new default `"auto"` includes every correlation surface. A CZ flow error names the flow
  as Stim writes it, and `gallery.cz` takes Stim's sparse flow syntax.
- **Tsim.** `tsim.Circuit`'s detector sampler reports values as Tsim's does
  (`use_detector_reference_sample` and `use_observable_reference_sample` give flips).
  `str` and `repr` use Tsim's shorthand. `detector_error_model` reads T as S, as Tsim's does.
- **PECOS.** `SparseSim.print_stabs()` alone prints and returns only the stabilizers, as PECOS
  documents. `fault_tolerance_check` decodes an ideal round after the faulty one.
- **Azure.** `jobParams` echoes the scheme's name as given, and a custom scheme's absent power
  as null.

### Fixed

- **Stim.**
  - `OBSERVABLE_INCLUDE` targets print in the order written (a Pauli before a record).
  - `explain_detector_error_model_errors(dem_filter=)` lists in Stim's order.
  - A search that finds no logical error raises Stim's message with its warnings.
  - A negative repetition count on a circuit or model raises Stim's `TypeError`.
- **Tsim.** Two adjacent identical `I[R_PAULI(...)]` lines were fused into one that didn't
  parse.
- **PECOS 0.2.0's own bugs.** Its phenomenological error models (`DepolarGen`, `XGen`,
  `ZGen`, `XZGen`), mixed two-qubit error sets on NumPy 2, `TimingRunner.total_time` and
  `run_gates`, `fault_tolerance_check` on the shipped codes, instruction plots on networkx 3,
  and the Python simulator's `copy` and `refactor`. `'init |-i>'` prepares |−i>, and Y prints
  with its sign.
- **qecsim.** Centre-slice labels `lim=(0.2, 0.8, 0.0)` work under NumPy 2, and its
  missing-argument message under click 8.2.
- **PanQEC and qecsim.** `--version` names panqec and qecsim again.
- **Relay-BP.** Check matrices with stored zeros or even entries decode as relay_bp's do.
- **Tesseract.** The two-pass decoder on an annotated model ("DEM error index out of range"),
  repeated detection events (a panic), merging zero-probability errors ("bad argument 'nan'") and
  beam climbing with a negative beam now give Tesseract's answers.
- **Python 3.9 and 3.10, and Windows.** Loom's operations ran only on 3.11+ (`zip(strict=)`);
  they now run everywhere. The tests read Windows' line endings and paths, Stim's Windows help
  pages and click 8.1, and skip a port's tests where its Python or extras are missing.

### Not ported

These remain, each listed in the README and the inventory:

- tqec's Hadamard pipes, spatial junction cubes and Y cubes, and its plaquette templates and
  detector search.
- PyMatching's `animate`.
- Tsim's PyZX diagrams and ZX graphs, its bit-for-bit (JAX) random numbers, and `to_matrix`
  with measurements.
- CUDA-Q QEC's `dem_chunks`, `expand_dem_chunks` and `trt_decoder`.
- The QDK's parameter classes (their dictionaries are accepted), and Qualtran's Dash layout.
- The coherent sampler's truncation past twelve generators. On a d = 3 surface-code memory over
  three rounds (fourteen generators) it moves the silent logical rate from 2.0's 0.00100 to
  0.00161 ± 0.00013, against the exact 0.00258. Part of the interference between rounds is
  still left out.
- Chromobius's predictions differ on some shots, always at the same matching weight. Its
  lifting depends on the order of the matched edges, and this package's matcher does not list
  them in PyMatching's order. Handed PyMatching's edges, the lifting equals Chromobius's on all
  but 1 of about 18,500 shots checked. The weights are bit for bit.
- Fusion Blossom's `helper.bottle`, a vendored copy of the Bottle web framework (the
  visualiser is served by the standard library). Tesseract and Chromobius have no command
  line, and the Tesseract port's A* decoder takes at most 64 observables.
- QUITS's distance search uses more memory and time than Stim's: 456 MB and 8.7 s against
  139 MB and 1.6 s on [[72, 8, 8]]. (Corrected in 2.2.0.)

## 2.0.0 — 2026-10-18

The features of twenty-five other QEC tools — inventoried, feature by feature, in
`docs/superpowers/notes/2026-10-09-parity-inventory.md`, which says what is here and what is
not — under the names those tools use, each port checked against its original; and one thing
none of them has: Clifford+T circuits with a
coherent error on every gate, sampled at sizes no state vector reaches. Thirteen workstreams:
Stim's objects and PyMatching's graph API, sinter and analysis, ldpc whole and twelve more
decoders, codes and code tools, noise and hardware models, the stabilizer-rank and joint
samplers, tqec and Loom and the two resource estimators, real time and the GPU, the editor in
the site, and these notes. Breaking changes are under **Changed**; `str(circuit)` and the
decoders' defaults are now Stim's and ldpc's.

**The release's adversarial review** (seven independent reviews of the 2.0 work against the
references, their scripts in the session's scratchpad; fixed before tagging):

- *Clifford+T.* A tagged gate at a Clifford angle (`I[R_Z(theta=π/2)] 0`, `I[U3(π, 0, π)] 0`) is
  the Clifford gate it is in every engine — the frame samplers, the reference run, the detector
  error model and the coherent samplers (`stabrank::frame_view`); Stim reads the tag as nothing,
  and so did the engines the circuit was routed to when it had no other non-Clifford gate. A
  coherent error at a Clifford angle (`I_ERROR[R_Z(theta=π/4)]`, a fixed S) is run as the gate it
  is, exactly, by the joint sampler (which `compile_coherent_sampler` now picks for such a
  circuit), not drawn as a twirl; the reference run ignores it, as Stim's ignores every error. The truncated joint sampler's generators
  are now a basis of the kernel's projection onto the error locations with every single-error
  element present: the basis over every location (the exact gates' among them) could project
  `{e₁, T}` and `{e₁, e₂}` and leave `{e₂}` to a product no shot formed, and its sums missed
  first-order terms on small circuits (60σ on a 1-qubit case); `compile_coherent_sampler(pairs=)`
  exposes the pairwise products as a check. The frame sampler warns when a circuit's coherent
  errors are read as nothing (`twirled()` and `compile_coherent_sampler` run them; with a
  non-Clifford gate present the stabilizer-rank sampler runs them exactly, as before).
  `I_ERROR[T]` is refused (T is a gate). `tcount`'s docstring says where it differs from Tsim.
- *tqec.* A correlation surface crossing two seams of one merge got two seam terms that
  cancelled (the compiled observable then depended on the seam qubits' resets); one term covers
  a merge's seams on a line. Merges are indexed per time slice, so the same pair merging twice is
  two merges. `adjust_hadamards_direction` has both directions of each Hadamard pair.
  `ZXGraph.from_block_graph` checks tqec's preconditions. Terms are written in a fixed order.
- *The editor.* `MPP X0*Z1` parses and prints as Stim writes it; `CNOT`, `MZ`, `SQRT_Z`, `H_XZ`
  and the other aliases read as their canonical gates; `MPAD`, `MXX`/`MYY`/`MZZ` and the
  heralded channels make records; `rec[-k]`-controlled gates apply their Pauli when a marker
  flipped the record; an observable with several `OBSERVABLE_INCLUDE` lines is their parity;
  `SPP`/`SPP_DAG` follow the product rule; every Clifford gate Stim has is in the tables, which
  `tools/editor_fixtures.py` now writes from the package's gate data (`js/clifford-tables.js`).
  Markers in empty layers survive the URL; a `+` in a URL is not a space; `%g` rounds before it
  picks the exponent; stray braces and bad targets are errors; a million operations is the most
  a REPEAT block unrolls to; placing or removing a measurement, or deleting a layer, moves every
  later `rec[-k]` with its record and drops the ones whose record went; coordinates keep every
  argument.
- *The GPU.* The pacer's clamp panicked on a graph with fewer than 64 shots per dispatch budget
  and poisoned the GPU lock; the lock recovers. Dispatches start at one word (or at a billion edge
  updates' worth of BP shots) and grow fourfold a step while one finishes within a second for
  sampling and ten for BP (a d = 17 word's own latency is half a second, a d = 11 BP shot's a
  third, and the GPU runs hundreds or thousands in that same time), shrinking in proportion past
  it, within 96 MiB of buffers; a GPU that never signals is an error after ten minutes; a word writes a sentinel
  the host checks, so a dispatch the driver drops (Metal did so now and then past
  100 MB, with no error) is retried at half the size and, if it still fails, is an error and
  never zeros; wgpu's validation and allocation errors are errors too. The lane sampler's per-word
  generator is xoshiro128** (128 bits of state from the word's seed; the 32-bit xorshift paired
  words every 2³³ and repeated 32-shot blocks), so **the lane devices' shots for a seed differ
  from 2.0's development builds** (the cpu device's are unchanged). A REPEAT nested deeper than
  the GPU's 16-level loop stack, or counting past 2³² − 1, is refused rather than skipped; a
  program too long for the GPU's buffers is refused. The parser refuses REPEAT blocks nested
  more than 100 deep rather than overflowing its stack (a thousand overflowed a small thread's).
- *`BpDecoder` on a device.* `decode_batch` means the same on every device: rows are syndromes,
  a zero row is the zero error, the decoder's per-shot state is left alone, and `_details`
  returns the flags on the cpu device too; a single shot on the gpu device runs on the cpu-f32
  path (the same bits, without an upload); a scale single precision cannot hold is refused; a
  device without a method asks for min-sum; `threads=-1` is a `ValueError`. The f32 claim is
  bounded to damped min-sum, with a test of what holds at scale 1 and 0.
- *The estimates.* The QDK's parse-time checks: error rates strictly between 0 and 1, counts
  non-negative integers, both parts of a Majorana measurement error, all three parts of an error
  budget, `distanceCoefficientPower` a non-negative integer; a distillation-unit spec with a
  `name` is the named unit whatever else it carries, as the QDK's untagged union reads it;
  formulas read as fasteval 0.2.4 reads them (its parser, its constant-folding compile pass and
  its function table, ported: unary minus binds tightest, each operator its own level with `^`
  right-associative, `codeDistance + 0.1 + 0.2` is 3.3, `min`/`max`/`log`/trig, `inf` and
  `NaN`, its length and depth limits; Python's `**` refused); the binomial quantile's Newton
  branch is the `probability` crate's loop verbatim (saturating casts and all) and raises only
  on the iterate the crate never leaves, the no-factory loop jumps to the cycle count that moves
  the code distance where the QDK's `+1` stalls and raises only when none does, the QDK's
  wrapping `u64::pow` in the distance coefficient is wrapped the same way, and an empty frontier
  is an error where the QDK panics; `Constraints` and every count, time and rate are validated
  as the QDK's parser validates them (45 more recorded QDK cases in
  `tests/estimate_port/qdk_review2.json`). Qualtran's `_Density` refuses probabilities outside
  [0, 1].
- *Real time.* The protocol caps a frame at 256 MiB and a decode at 2²⁰ shots, drops a stalled
  connection after `timeout`, and a reset forgets in-flight results; shared queues hold their
  decoder weakly; `connect` and `spawn` fail fast with the server's error.
- *A second pass*, after those fixes: the editor reads marks where Stim and Crumble write them
  (mid-layer, splitting the layer there; after a layer's gates), takes pair-target gates and
  channels out by the pair, prints numbers as the package does (half to even, big integers
  exact), reads lowercase Pauli targets, keeps `MPAD`'s bits out of the qubits, and is strict
  about braces, `rec[-0]` and non-decimal numbers; tqec's compile checks explicit
  `observables=` against the graph's pipes and spider rules (another graph's surfaces compiled
  into random observables) and names a lone cube; the realtime client shuts the peer down
  before closing its files (a silent server hung `connect`/`spawn` past their timeout), the
  server's listen backlog is the system's (five reset most of a burst), a timed-out
  `get_corrections` keeps the batch, batches past the server's `max_shots` (now in HELLO) go in
  pieces, a shared queue holds its decoder by a plain weak reference, decoders of plain rows
  and `**kwargs` signatures are served, the median is a median, one syndrome enqueues as one
  shot, and the CLI checks its port; the joint sampler sums every product of its generators on
  small kernels (twelve or fewer: the whole projected span, the same set from every shot) and
  `pairs=True` takes every pair, touched or not, so the second-order terms cancel as they
  should; a `U3` written as a gate expands for the frame engines only when all three of its
  rotations are Clifford; `I_ERROR[U3(...)]` is refused; the frame sampler's warning covers
  coherent errors at Clifford angles. **The 1.9 coherent sampler's detection events on
  nondeterministic measurements were complemented** whenever its own seeded reference run had
  drawn a 1 there (its weights were exact; the exhaustive tests compared raw parities): its
  reference is now the converter's, every random outcome 0, as every other engine's, and the
  exhaustive tests compare detection events. The GPU pacer holds a dispatch to the larger of
  its target and four unit latencies (held to the target alone, a word slower than a second
  shrank every dispatch to one word for good), keeps what it learned on the program or decoder
  for the next call, refuses to run in a child forked after the device opened (Metal crashes
  there) and returns an error for a shot count that does not fit in memory; a `BpDecoder` on
  the gpu device runs batches under 256 shots on the cpu-f32 path (the same bits, faster).
- *The text.* The README's benchmark table is re-recorded with every reference installed; the
  leakage, herald, twirl and f32 claims are bounded to what the runs show; third-party notices
  for PECOS, PanQEC, QUITS and the whole of Deltakit; the licence statement names the licences
  involved; "every decoder" is every detector-error-model decoder.

### Changed

- **`str(circuit)` is Stim's text exactly**: compatible neighbouring instructions are fused
  (`H 0` then `H 1` is `H 0 1`), arguments print to 6 significant digits as Stim prints them,
  and there is no trailing newline. The arguments themselves stay exact: equality, pickling and
  copies use them, and `circuit[k].gate_args_copy()` returns them.
- **A circuit's instructions are Stim's list**: the parser and `append` fuse an instruction
  onto the one before it when they can be one, as Stim's do, while `pop`, slicing and REPEAT
  bodies leave neighbours apart, and `insert` fuses only at its seams; `len`, indexing,
  equality and the text follow (`c.pop(1)` of `RX 0`, `TICK`, `RX 1` leaves two instructions,
  not `RX 0 1`). This holds in Rust too: `Circuit::parse` and `append_circuit` fuse.
- **`BpDecoder`, `BpOsdDecoder` and `BpLsdDecoder` take ldpc's defaults**: `max_iter=0` (the
  block length), product-sum BP unless `bp_method` is given (ldpc's effective default, whatever
  its signatures say), `ms_scaling_factor=1.0`, and for BP+OSD `osd_method="osd_0"`. 1.x's
  defaults were 20 iterations, min-sum, adaptive scaling and OSD-CS of order 7; pass them to
  keep 1.x's results. An OSD order of 0 now means OSD-0 for every method, as in ldpc.
- The `stabilizer-qec[sinter]` extra installs matplotlib (also `[plot]`); sinter and Stim are
  no longer needed.
- **An error's targets stay as written**, in their order and with repeats, in a model's text
  and when it is flattened, as Stim keeps them (`error(0.1) D1 D0` was printed `D0 D1`, and a
  target written twice was dropped); a repeated target still cancels in the error's meaning.
  `error` instructions starting or ending with `^`, or with `^ ^`, are refused, as in Stim.
  `DemInstruction` is hashable, as Stim's is.
- `repr(DetectorErrorModel)` prints numbers at 6 significant digits, as Stim's repr does on
  every platform (`str` keeps full precision: 19 digits on x86-64, as Stim's).
- Text shot files (`01`, `hits`, `dets`) may end their lines `\r\n`, as files written in text
  mode on Windows do.
- `REPEAT 0 { ... }` is an error, as in Stim ("Repeating 0 times is not supported.").
- **Errors as Stim words them**: an unknown instruction is `Gate not found: 'NAME'` (was
  `unsupported instruction 'NAME'`), and a file that cannot be opened raises `ValueError`
  ("Failed to open 'path' for reading." or "... to write."), not `OSError`. `REPEAT 5 {}` on one
  line is an empty block, as Stim reads it.
- **ldpc's OSD on Windows sorts as Microsoft's `qsort` does**: its unstable order among columns of
  equal reliability (was a stable sort there), so `BpOsdDecoder` (and PanQEC's and QUITS' BP+OSD
  on it) decodes as ldpc does on Windows as well as on macOS and Linux.
- `stabilizer-qec gen` and `analyze_errors` end their output with a newline, as Stim's do;
  `str(DetectorErrorModel)` has no final newline and its `repr` is Stim's.
- **A fault that flips observables and no detector is kept in the model** as `error(p) L0`, as
  Stim keeps it; 1.x refused the circuit. `Dem::refuse_undetectable` (Rust) gives the old
  refusal where a caller wants it.
- **Any number of observables**: circuits, model text, model building, the detector sampler and
  measurement conversion take observables past 64 (`OBSERVABLE_INCLUDE(99)`, `L399`), as Stim
  does. The decoders, the model sampler and the per-shot samplers still take at most 64 and say
  so.

- **`detslice-text` is Stim's diagram**, character for character (qubit lines with each
  detector's Paulis at its tick), where 1.x printed a line per detector. Slice diagrams show
  every detector and no observable unless `filter_coords` asks, as in Stim.
- A correlated error may name a qubit twice (`E(0.1) X0 Z0`), and a Pauli product may repeat
  a qubit (`MPP X0*X0`, `SPP X0*X1*X0`), as in Stim: the Paulis multiply, a product that cancels
  to ±1 is measured as that sign every time. 1.x refused both. Products Stim parses but refuses
  to run (anti-Hermitian ones such as `X0*Z0`; edits to the measurement record such as
  `CX 0 rec[-1]`) are refused when the circuit is built, with Stim's message.
- Integer targets are read as Stim reads them: digits only (`+3`, `rec[-+1]`, `D+1` are
  refused, as in Stim; 1.x took a leading `+`).
- **`Matching` is PyMatching's `Matching`** (2.4), graph API and all. Its first positional
  parameter is `graph` (a model, check matrix, networkx or rustworkx graph), as PyMatching's;
  1.x's `Matching(model)` still works. **Correlated decoding is asked for twice, as in PyMatching**:
  `enable_correlations=True` when building from a model and again when decoding
  (`decode(..., enable_correlations=True)`, `decode_batch(...)`). In 1.x, building with it was
  enough; 1.x code that built with it now decodes without correlations unless the decode call
  asks too. `decode_batch` casts shots of any integer or bool
  dtype as PyMatching does, where 1.x refused values other than 0 and 1.
- **A reference sample takes every random measurement as 0** (it was a seeded draw), as
  Stim's does where the measured qubit is not entangled with another; where it is, Stim's
  tableau can make another choice (`H 0`, `CX 0 1`, `CZ 1 0`, `MX 1`, `MX 0` gives 1, 0 in
  Stim's reference and 0, 1 here): `reference_sample()`, the samplers' references and the
  measurement-to-detection converter, which now takes detectors and observables that are not deterministic (gauge
  detectors) and compares them with that reference, as Stim's converter does. A seed's raw
  measurement shots change with it; detection events do not.
- `num_qubits` counts targets as written, as Stim does: an `II`'s or `I_ERROR`'s, an `MPAD`'s
  values and the qubits of products that cancel; a circuit's tableau (`Tableau.from_circuit`)
  is that wide.
- Correlated matching merges parallel edges as PyMatching does (their probabilities combined,
  the first edge's observables kept); it kept the likelier edge's observables.
- The density matrix and the perturbative expansion refuse a chain of correlated errors broken by
  another instruction (`E`, `X 0`, `ELSE_CORRELATED_ERROR`); an `ELSE_CORRELATED_ERROR` with no
  chain before it starts one, as in Stim.
- The text timeline has no trailing newline (an empty circuit's is `''`), as Stim's.
- Parsing as Stim parses: spaces around a product's `*` (`MPP X0 * Y1`), products in `E`
  (`E(0.1) X0*Y1`, its factors), `CZ` on two classical bits (no gate), lowercase `d0`/`l0` in
  a model; arguments must follow the name directly and a space must follow them
  (`X_ERROR (0.1) 0` and `X_ERROR(0.1)0` are refused), and a model's `detector` and
  `logical_observable` take one target.
- `Tableau.iter_all` runs in Stim's order; `str(GateData)` prints no flows, as Stim's;
  `PauliString.after`/`before` refuse an operation on a qubit past the string, naming it as
  Stim does; `PauliString([True])` is `+X`; `DemTarget("^").val` raises, as Stim's.
- `set_density_matrix` takes only density matrices (finite, Hermitian, trace 1, no negative
  eigenvalue). The ldpc decoders and `Matching` refuse check matrices with entries other than 0
  and 1 (they read 2 as 0 or as 1), with ldpc's and PyMatching's messages, and take float 0/1
  matrices, as ldpc does; `BpOsdDecoder` refuses an order above 0 for OSD-0, as ldpc does.
- `DemSampler.sample` refuses a replay whose row count is not `shots`; `sample_write` and the
  CLI's `sample_dem` replay the first `shots` rows and refuse fewer, as Stim's.
- The package and crate are licensed `MIT AND Apache-2.0`: the parts ported from Stim keep
  Stim's Apache-2.0 licence (`THIRD_PARTY_NOTICES.md`, `LICENSES/`), shipped in the wheels,
  the sdist and the crate.

### Added

- **PyMatching's graph API** (`stabilizer_qec.Matching`, and `stabilizer_qec.pymatching`):
  graphs built edge by edge (`add_edge`, `add_boundary_edge`, `set_boundary_nodes`, the five
  merge strategies, fault-id sets of any size), from check matrices (dense or scipy, with
  `repetitions`, `timelike_weights`, `measurement_error_probabilities`, `faults_matrix` and a
  virtual boundary), from networkx and rustworkx graphs, models, circuits and files; every
  query (`edges`, `get_edge_data`, `has_edge`, `num_fault_ids`, ...); `decode`,
  `decode_batch` (bit-packed in and out, weights), `decode_to_edges_array`,
  `decode_to_matched_dets_array` / `_dict`; `to_networkx` / `to_rustworkx`; `add_noise` with
  `set_seed` / `randomize` / `rand_float`; `draw`; pickling. Weights are discretised as
  PyMatching discretises them, negative weights handled as PyMatching handles them:
  predictions and solution weights are PyMatching's own on random hand-built graphs, check
  matrices and models (`tests/test_pymatching_differential.py`, which also checks every
  public name and parameter). Rust: `stabilizer_qec::user_graph`.
- **sinter, whole** (`stabilizer_qec.sinter`): a port of sinter 1.16 that needs no Stim,
  PyMatching or scipy (matplotlib only to plot). `Task`, `TaskStats`, `AnonTaskStats`,
  `CollectionOptions`, `Progress`; `collect` / `iter_collect` with worker processes, saving
  and resuming, existing data, post-selection, custom counts and error keys; the CSV format and
  `read_stats_from_csv_files`; the fits; `plot_error_rate`, `plot_discard_rate`,
  `plot_custom`, `group_by`; `predict_*`; and the command line (`stabilizer-qec sinter
  collect|combine|plot|predict`, `python -m stabilizer_qec.sinter`). The built-in decoders are
  sinter's, with `pymatching` this package's PyMatching, and this package's (`matching`,
  `correlated_matching`, `belief_matching`, `union_find`, `bposd`, `bplsd`, `relay_bp`,
  `search`). sinter's own unit tests and docstring examples pass on the port
  (`tests/sinter_port/`), and its CSV text, strong ids, fits, combined files and predictions are
  sinter's (`tests/test_sinter_differential.py`). 1.x's `Decoder(kind)`, `Sampler(kind)` and
  `sinter_decoders()` work as before, with this sinter or with sinter itself.
- **qec-util's tools** (`stabilizer_qec.analysis`: `circuits`, `dems`, `dem_instrs`,
  `data_processing`, `distance`, `mod2`, `performance`, `samplers`, `threshold`), a port of
  qec-util 0.5 that needs only numpy: gauge and detector removal, observables as detectors,
  rec-target formatting, model differences, hyperedge decomposition, disjoint graphs, p_ij
  matrices (exact and approximate), syndromes and defects (numpy or xarray), GF(2) solving,
  the logical-error decay fit, Wilson intervals, failure sampling to CSV, threshold fits with
  bootstrapped intervals, and exact circuit distances (scipy's HiGHS for qec-util's Gurobi).
  Its results are qec-util's on generated circuits and models
  (`tests/test_analysis_qec_util.py`): the same text, matrices and numbers, fits to within
  10⁻³ of a standard error, and decompositions up to the order of their pieces.
- **deltakit's analysis** (`stabilizer_qec.analysis`): Λ fits (`calculate_lambda_and_lambda_stddev`
  by three methods, `calculate_lambda_asymmetric` with profile-likelihood intervals), the
  logical error per round with SPAM (`compute_logical_error_per_round`,
  `fit_logical_error_per_round_asymmetric`, `simulate_different_round_numbers_*`), binomial
  fits, quops prediction (`predict_quops_at_distance`, `predict_quops_interval`,
  `predict_distance_for_quops`); `pij` (expectation data, p_ij of edges and hyperedges up to
  four detectors, models from p_ij, comparisons, the correlation matrix and its plot);
  `decoding` (`run_decoding_on_circuit`, `EmpiricalDecodingErrorDistribution`);
  `error_budget` (each noise parameter's contribution to 1/Λ by Google's method, with
  `standard_noise_model`, whose noise is that of Stim's generated circuits); and `plotting`
  (Λ, logical error per round, defect rates, detection probability on the patch). Only numpy
  is needed: scipy's optimisers and uncertainties' propagation are used when installed, and
  numpy versions of them otherwise. Results are deltakit's (`tests/test_analysis_deltakit.py`,
  run where deltakit installs): fits to 10⁻⁷, intervals to 10⁻⁶, p_ij to 10⁻¹², and to
  10⁻⁴ without scipy and uncertainties.
- **Subset sampling**: `stabilizer_qec.qsample` is qsample (dynamical subset sampling of
  fault-tolerant protocols: `Protocol` graphs of circuits whose transitions check measurement
  history, its error models, `DirectSampler` and `SubsetSampler` with the statistical error and
  the upper bound from subsets not reached), ported onto this package's tableau simulator with
  no dependency beyond numpy; its estimates agree with qsample's and with direct sampling
  (`tests/test_qsample.py`). `stabilizer_qec.analysis.subset_sampling.SubsetSampler` does the
  same for any Stim-format circuit and decoder: the circuit's noise channels are its fault
  locations, each fault count's failure rate is sampled with exactly that many faults
  injected through the flip simulator, and one run gives the logical error rate, its standard
  deviation and both bounds at every scale of the noise. It agrees with direct sampling at
  every scale (`tests/test_analysis.py`).
- **ldpc's BP options**, bit for bit against ldpc 2.4.1 (`tests/test_ldpc_bp_options.py`):
  serial and serial-relative schedules and the random serial schedule
  (`random_serial_schedule`, `random_schedule_seed`, `serial_schedule_order`), on
  `BpDecoder`, `BpOsdDecoder` and `BpLsdDecoder`; `input_vector_type="received_vector"`;
  `SoftInfoBpDecoder` (soft syndromes, `cutoff`, `sigma`); and every property ldpc's classes
  have (`error_rate`, `error_channel`, `channel_probs`, `update_channel_probs`,
  `log_prob_ratios`, `decoding`, `bp_decoding`, `osd0_decoding`, `osdw_decoding`,
  `osd_method`/`osd_order` and `lsd_method`/`lsd_order` setters, `omp_thread_count`, ...).
  Where ldpc's result depends on the C++ standard library (shuffles, and the order of ties in
  `std::sort` and `qsort`), this package reproduces libc++ and Apple's libc on macOS and
  libstdc++ and glibc on Linux, and Microsoft's shuffle and `std::sort` on Windows (its `qsort`,
  which OSD's ties depend on, is approximated by a stable sort).
- **ldpc's `BpFlipDecoder` and `LsdDecoder`**, bit for bit against ldpc 2.4.1
  (`tests/test_ldpc_decoders.py`): bit flipping with seeded random flips (`flip_iterations`,
  `pflip_frequency`, `pflip_seed`) followed by BP; and LSD alone on given bit weights. LSD's
  sorts (candidates and free columns by weight, invalid clusters by size) are now the platform's
  `std::sort`, so `BpLsdDecoder` and `LsdDecoder` agree with ldpc on ties too, leaving only
  merges of three or more clusters at once, whose order ldpc takes from addresses. On Windows,
  `tsl::robin_set`'s order now follows Microsoft's `std::hash<int>` (FNV-1a), as ldpc's Windows
  wheels do.
- **ldpc's `UnionFindDecoder` and `BeliefFindDecoder`** on any check matrix, bit for bit against
  ldpc 2.4.1 (`tests/test_ldpc_decoders.py`): clusters grown around the fired checks, all at
  once or `bits_per_step` at a time by weight (BP's posteriors in `BeliefFindDecoder`), solved
  by inversion (any matrix) or by peeling a spanning tree (columns of one or two checks). Where
  ldpc's peeling goes round forever, the cluster is solved by inversion (`last_stuck`); where
  no correction exists (ldpc loops), `decode` raises `ValueError`. `UnionFindDecoder`'s
  `uf_method` takes `True`/`False` or a name (ldpc's reads any non-empty string, `"peeling"`
  included, as inversion, and refuses its own default).
- **`stabilizer_qec.ldpc`: ldpc 2.4.1, whole** (`import stabilizer_qec.ldpc as ldpc`; extra `ldpc`
  for scipy). The decoders under ldpc's names and module paths (`ldpc.bp_decoder`,
  `ldpc.bposd_decoder._bposd_decoder`, ...), ldpc v1's `bp_decoder` and `bposd_decoder`, and its
  pure-Python modules ported: `code_util`, `codes`, `alist`, `protograph`, `noise_models`,
  `monte_carlo_simulation` (numba and tqdm optional), `ckt_noise` (circuit check matrices,
  overlapping-window decoders, memory circuits) and `sinter_decoders`, on this package's
  Stim, PyMatching and sinter. `mod2` (Cython and C++ in ldpc) is reimplemented with ldpc's
  answers entry for entry: both eliminations (gf2dense's PLU and gf2sparse's `RowReduce`,
  whose pivot choice follows its linked lists' order after row swaps), kernels, row bases and
  spans, distances and `PluDecomposition`'s factors (`tests/test_ldpc_package.py`). ldpc's own
  test suite passes on the port (`tests/ldpc_port`; two of its tests fail on ldpc 2.4.1 too).
  `BpLsdDecoder` keeps ldpc's statistics (`set_do_stats`, `statistics`,
  `set_additional_stat_fields`, `reset_cluster_stats`), equal to ldpc's; `SoftInfoBpOsdDecoder`,
  an empty class in ldpc, decodes as it describes. Wrongly typed arguments to the BP decoders
  raise `TypeError`, as ldpc's do.
- **`stabilizer_qec.relay_bp`: IBM's relay_bp 0.2.2, whole** (`import stabilizer_qec.relay_bp as
  relay_bp`), bit for bit against relay_bp (`tests/test_relay_bp_package.py`): Relay-BP in f32, f64,
  i32 and i64 (`RelayDecoderF32`…) and min-sum (Mem-)BP in f32, f64, i8, i16, i32, i64 and 64-bit
  fixed point with 0–31 fractional bits (`MinSumBPDecoderF32`…`MinSumBPDecoderFixed`), with
  `data_scale_value`, `max_data_value`, explicit memory strengths, the three stopping criteria
  and relay_bp's `relay_logging.out`; `decode_detailed` and `DecodeResult`;
  `ObservableDecoderRunner` (sequential and parallel batches, decoding from known errors with
  relay_bp's error classification) and `ObservableDecodeResult`; and its `stim` subpackage
  (`CheckMatrices`, the sinter decoders, the noise tools) on this package's Stim and sinter.
  Rust: `RelayCore<N>` over `RelayNum` (`Relay` is `RelayCore<f64>`); `RelayConfig` gains
  `stop_after_first`, `data_scale`, `max_data` and `logging` and implements `Default`.
- **`stabilizer_qec.tesseract_decoder`: Tesseract, whole** (`import stabilizer_qec.tesseract_decoder
  as tesseract_decoder`), against Tesseract's 2026-10-13 build (`tests/test_tesseract_package.py`)
  and passing its own Python tests (`tests/tesseract_port/`): `TesseractConfig` and
  `TesseractDecoder` with every setting (beams and `INF_DET_BEAM`, beam climbing, revisits,
  `pqlimit`, `det_penalty`, merging, sparsified error activation, literal or generated detector
  orders), giving Tesseract's errors, flags, observables and costs; its verbose log and
  visualizer file line for line; `common`, `utils` and `viz`; the sinter decoders
  (`TesseractSinterDecoder`, pickling in all three layouts, `decode_via_files`,
  `make_tesseract_sinter_decoders_dict`); Tesseract's new **multi-pass decoder** (two detector
  components, reweighted by correlations between passes; `SchedulingStrategy`,
  `multi_pass_sinter_decoders`); the **simplex decoder**, whose integer programs are solved by an
  exact search (errors outside a time window, which cost nothing, solved by linear algebra), the
  same optimal cost as Tesseract's HiGHS and 30–100 times faster; and `demutil` (error
  decomposition, detector-basis classifiers, GARI). Rust: the search keeps Tesseract's verbose
  log, takes probabilities changed for one decode (`Search::decode_with_probabilities`) and
  adds beams with Tesseract's `size_t` wrap-around.
- **`stabilizer_qec.qldpc`: qLDPC 0.4.1 whole** (`import stabilizer_qec.qldpc as qldpc`), its
  Stim, sinter, ldpc, PyMatching, relay_bp and Tesseract this package's, and its own 953 tests
  passing (`tests/qldpc_port/`): its classical and quantum codes, distance, groups and rings,
  objects, circuits and memory experiments, noise models, surgery gadgets, and its decoders (lookup
  plain, weighted and observable, ILP, GUF, composite, direct, frontier, subgraph, sequential and
  sliding windows, trivial, erasure bits, and its adapters for every decoder above). Needs Python
  3.11 and the `qldpc` extra (galois, sympy, cvxpy, networkx and others), as qLDPC does.
- **`stabilizer_qec.cudaq_qec`: CUDA-Q QEC's decoders** (`import stabilizer_qec.cudaq_qec as qec`),
  passing CUDA-Q's own decoder tests (`tests/cudaq_qec_port/`): `get_decoder` (dense or scipy
  matrices, or a Stim model's text, with `O`, `error_rate_vec` and `output`), `Decoder`,
  `DecoderResult`, `BatchDecoderResult`, `decode_async`, the `@decoder` registry; the
  **single- and multi-error lookup tables** (Rust, `stabilizer_qec::lookup`), the **sliding
  window** around any decoder (whole or streamed round by round, with boundary rounds), the
  PyMatching plugin, the **tensor-network decoder** (exact maximum likelihood of a logical given a
  soft syndrome, by contraction, `stabilizer_qec::tn_ml`; CUDA-Q's published values to 1e-15)
  and the bundled examples; `dem_from_stim_text` and its `DetectorErrorModel` (round
  canonicalization); and its matrix tools (`sort_pcm_columns`, `get_pcm_for_rounds`,
  `pcm_extend_to_n_rounds`, `simplify_pcm`, `generate_random_pcm`, `shuffle_pcm_columns`, ...),
  the random ones drawing as CUDA-Q's C++ does. CUDA-Q's quantum kernels, GPU and real-time
  decoding need CUDA-Q itself.
- **CUDA-Q QEC's codes and memory experiments** in `stabilizer_qec.cudaq_qec`, passing CUDA-Q's
  code tests (`tests/cudaq_qec_port/test_qec_codes.py`, from its C++ unit tests, and
  `test_surface_code.py`): `get_code` (`"steane"`, `"repetition"`, `"surface_code"`, `stabilizers=`
  overrides, the `@code` registry for Python codes), `Code`'s parity, observable and schedule
  accessors; the surface code's **four orientations** `XV`, `XH`, `ZV`, `ZH` (NVIDIA Ising's
  `O1`-`O4`) with `stabilizer_grid`, its roles, coordinates, stabilizers, observables and CNOT
  schedules (each hook along the logical it cannot shorten), all as CUDA-Q's; `memory_circuit`,
  `sample_memory_circuit`, `dem_from_memory_circuit` (and their `x_`/`z_` forms) on a code with
  CUDA-Q's `NoiseModel` channels, detectors laid out as its kernels lay them out; and
  `sample_code_capacity` drawing as CUDA-Q's `std::mt19937` does (its seeded test data, equal).
- **`memory_circuit(..., orientation=)`**: the rotated SD6 memory in any of the four
  orientations (`"ZH"`, the default layout, is the circuit `memory_circuit` has always built).
- **`stabilizer_qec.tsim`: Tsim's encoders** (bloqade-tsim 0.1.5's `TransversalEncoder`,
  `SteaneEncoder` and `ColorEncoder5`, and its program-text shorthand, passing its tests in
  `tests/tsim_port/`) on this package's `Circuit`, each encoding checked by `Tableau` (the
  encoded stabilizers and logicals the inputs' images) and each encoded circuit's logical
  outcomes against the unencoded one's. `tsim_to_native` reads Tsim's tagged Stim (`S[T]`,
  rotations in half-turns, `SPP[T]`) as this package's non-Clifford operations, checked against
  a state vector of Tsim's definitions.
- **`stabilizer_qec.qecsim`: qecsim 1.0b9 whole**, passing its own tests (`tests/qecsim_port/`;
  2,902, including its command-line tests, which qecsim itself can no longer load without
  `pkg_resources`): planar, rotated planar, toric, rotated toric, colour 6.6.6 and small codes,
  its error models, and its decoders, among them the **MPS tensor-network decoders** (exact and
  by bond dimension; planar, rotated planar, colour 6.6.6) and the fault-tolerant (time-repeated)
  runs. Its Blossom V interface works without Blossom V's library, on this package's exact matcher.
  `stabilizer_qec.bposd`: bposd 2.1 on this package's ldpc.
- **`stabilizer_qec.mqt_qecc`: MQT QECC 2.0.0 whole** on this package's Stim, sinter, ldpc,
  PyMatching, qecsim and bposd, passing its own tests (`tests/mqt_qecc_port/`): the **colour-code
  decoder by MaxSAT** (LightsOut, `cc_decoder`), **analog information decoding** (single-shot and
  quasi-single-shot with analog syndromes), **fault-tolerant state preparation** (synthesis,
  verification and correction circuits, cat states, deterministic and heuristic, by SAT; encoders
  of optimal depth or gate count), stabilizer and CSS codes with their constructions (bivariate
  bicycle, hypergraph product, rotated surface, concatenated and colour codes, and its library
  of instances), colour-code routing and compilation (`cococo`) and code switching. The `mqt` extra
  installs what it needs (Python 3.10).
- **`stabilizer_qec.quits`: QUITS 1.2.0 whole** on this package's Stim and ldpc, passing its own
  126 tests (`tests/quits_port/`): bivariate bicycle, balanced product cyclic, hypergraph product,
  lift-connected surface, Mitten and (polynomial) quasi-cyclic lifted product codes, or any
  parity-check matrices; circuits by ZX colouration, cardinal and cardinal NS-merge schedules and
  each family's custom one (Mitten's hook-error-free too); the **sliding-window BP+OSD and BP+LSD
  decoders**, phenomenological and circuit-level; space-time distance estimates; layouts; GF(2)
  linear algebra, groups and lifts, and its LDPC generators. Against QUITS itself
  (`tests/test_quits_differential.py`) every family's matrices, every strategy's circuit text and
  every window decode are equal. The `quits` extra installs what it needs (Python 3.10).
- **`stabilizer_qec.panqec`: PanQEC 0.1.7 whole** on this package's ldpc and PyMatching, passing
  its own tests (`tests/panqec_port/` and its doctests; as of 2.2, 795 pass with its slow ones,
  which run only with `PANQEC_RUNSLOW=1`, as CI's live-references job runs them, and 12 are
  PanQEC's own expected failures): 2D and 3D toric,
  planar and rotated surface codes, rhombic and hollow codes, the X-cube fracton model, 6.6.6,
  4.8.8 and 3D colour codes, their Clifford deformations (XZZX, XY and others) and biased Pauli
  noise; its sweep-matching, rotated sweep-matching, X-cube matching, matching, BP+OSD, memory BP
  and union-find decoders; direct, batch and splitting (MCMC) simulations, the finite-size-scaling
  threshold analysis, its command line and the 2D/3D **GUI** (`python -m stabilizer_qec.panqec
  start-gui`; Flask, the `panqec-gui` extra). Against PanQEC itself
  (`tests/test_panqec_differential.py`): every registered code's stabilizers, logicals and
  coordinates, plain and deformed, the noise drawn on the same seeds, every decoder's corrections
  on the same errors (on PyMatching's matcher; on this package's, whose ties between equally good
  matchings fall otherwise, the same syndromes and weights), direct simulations and the GUI's
  answers, all equal. Memory BP takes the syndrome as an integer, where PanQEC's raises on NumPy
  2. The `panqec` extra installs what it needs.
- **`stabilizer_qec.pecos`: PECOS 0.2.0 whole** (`quantum-pecos`), on NumPy 2 and networkx 3,
  passing its own tests (`tests/pecos_port/`): the 4.4.4.4 and medial 4.4.4.4 surface codes and
  the 4.8.8 colour code with their logical gates and instructions, logical circuits, error
  generators (depolarizing, X, Z, XZ and gatewise, at the code-capacity, phenomenological and
  circuit levels), the standard and timing circuit runners, MWPM2D, the fault-tolerance checks
  (`t_errors_check`, `fault_check`, `distance_check`, `fault_tolerance_check`), logical error
  rates and pseudo-thresholds. Its C++ sparse stabilizer simulator (`cySparseSim`, the default
  `SparseSim` here, as in PECOS's source; PECOS 0.2.0 installed from PyPI has no compiled C++,
  so its `SparseSim` is the Python one, `pySparseSim` here) is ported to Rust: on random circuits of every gate, initialization and
  measurement it ends in PECOS's C++ simulator's stabilizer group with its outcomes. Against
  PECOS itself (`tests/test_pecos_differential.py`, from `tests/data/pecos_reference.json`, as
  PECOS needs NumPy 1): both simulators, every code's every instruction, errors and outputs on
  the same seeds, and logical error rates are equal, MWPM2D's recoveries too, or among equally
  good matchings (networkx 3 breaks ties otherwise) the same syndromes and weights. Where PECOS
  0.2.0 itself raises, the port works: the fault-tolerance checks (on the instructions'
  `final_logical_ops`, not a `logical_stabs` none has, and `ErrorCircuits.add_circuits`), the
  medial and colour codes' initializations (their circuits' `metadata`, once `params`), the
  colour code's initialized logical operators (PECOS labels its Z `X`), MWPM2D with no syndrome,
  and measurement outcomes left to the default, -1 (drawn at random, as in PECOS's C++). The
  `pecos` extra installs what it needs; its density- and process-matrix simulators are below.
- **`stabilizer_qec.deltakit`: Deltakit 0.10 whole** (Riverlane's toolkit: deltakit-circuit,
  -core and -decode 0.9.2, -explorer 0.9.4), passing its own tests (`tests/deltakit_port/`):
  `deltakit.circuit` (its circuit model: gates, noise channels, annotations, layers, noise
  factories, Stim conversion both ways), `deltakit.core` (decoding graphs and hypergraphs, DEM
  parsing, windowing, b8/01/csv data formats), `deltakit.decode` (its matching decoders, noise
  sources for code-capacity matching, decoder managers and analysis engines) and
  `deltakit.explorer` (rotated and unrotated planar, toric, repetition, bivariate bicycle and
  generic CSS codes with their schedules, memory experiments, QPUs with native gate sets and
  times, compilation to them, the SD6, SI1000 (with leakage), toy, phenomenological and
  T1/T2-derived physical noise models, analysis, error budgets, types and plots). Its cloud
  client is Deltakit's, calling Riverlane's server only with the user's token, for what runs
  only there in Deltakit too (stability circuits, its cloud decoders). Against Deltakit itself
  (`tests/test_deltakit_differential.py`): every code's memory circuits in both bases, compiled to
  native gate sets under every noise model, equal as text and as detector error models. The
  `deltakit` extra (Python 3.11) installs what it needs.
- **`stabilizer_qec.deltakit_stim`: Deltakit's Stim** (`deltakit_stim` 0.2.7), which Deltakit
  imports as `stim`: this package's Stim with Deltakit's leakage instructions native (`LEAKAGE`,
  `RELAX`, `RL`, `HERALD_LEAKAGE_EVENT`, and leakage spreading and mobility as the four arguments
  of `CX`, `CY`, `CZ`, `SQRT_XX` and `SQRT_XX_DAG`). It prints, iterates and indexes them as
  Deltakit's does; its samplers run Deltakit's leakage model, and its detector error models
  are Deltakit's, with each herald's leakage and each transport gate's depolarized partner, in
  Deltakit's single-precision arithmetic. `Circuit.generated` takes Deltakit's
  `after_reset_leakage` and `after_clifford_leakage_and_relaxation`. Against `deltakit_stim`
  itself (`tests/test_deltakit_stim_differential.py`): text, instructions, gate data, noise
  removal, decomposition, generated circuits and detector error models (heralds, transport,
  loops, decomposition) equal, and sampled rates equal within sampling error. One difference is
  kept on purpose: Deltakit's frame simulator stops moving leakage (mobility) for the rest of a
  batch once a move lands on an already leaked partner, which here, as the probability says,
  every shot draws anew. In the base `Circuit` these lines stay inert tags, as Stim reads them.
- **Density matrices**: a density-matrix simulator in Rust (up to 12 qubits) that runs everything
  the exact sampler runs, noise channels applied whole and measurements branching only on what
  they record. `Circuit.density_matrix()` (the final state, outcomes averaged away),
  `Circuit.exact_distribution(method="density_matrix")` (far fewer branches on a noisy circuit),
  and `DensityMatrixSimulator`, acted on circuit by circuit as `TableauSimulator` is: unitaries
  and Kraus channels on any qubits, forced measurements in any basis, expectations, purity,
  reduced states, entropy and fidelity (with `reduced_density_matrix`, `fidelity` and
  `von_neumann_entropy`). Checked against Quantumsim 0.2.0's density matrices (to 1e-10), the
  state vector's exact distributions and trajectories, and the tableau.
- **Phase damping and thermal relaxation**: `I_ERROR[PHASE_DAMPING(lambda=…)]` and
  `I_ERROR[THERMAL_RELAXATION(t=…, t1=…, t2=…)]` (amplitude damping then phase damping, as
  Quantumsim's T1/T2 channel), in the density matrix and the state vector's exact distributions
  and trajectories.
- **Process matrices**: `process_matrix(circuit, kind)` and `Circuit.process_matrix` give a
  channel on up to 6 qubits as its Choi matrix, superoperator, Pauli transfer matrix or χ
  matrix; `convert_process` between them, `process_fidelity`, `average_gate_fidelity`,
  `choi_of_unitary` and `pauli_basis`. Checked against the closed forms of Clifford gates,
  depolarizing and damping channels.
- **PECOS's density- and process-matrix simulators**, which PECOS 0.2.0 declares and does not
  ship: `stabilizer_qec.pecos.simulators.DensityMatrix` and `ProcessMatrix` on its simulator
  interface and gate symbols (and ProjectQ's further gates: T, rotations, controlled rotations,
  Toffoli), checked gate by gate and on random circuits against PECOS's stabilizer simulators,
  and on PECOS's own logical runs. An initialization is a reset (PECOS's stabilizer simulators
  force a measurement outcome instead; the same on a qubit not entangled with others).
- **The perturbative expansion** (QuantumClifford.jl's `petrajectories`, on detection events):
  `Circuit.perturbative_distribution(order)` sums the outcome distribution over every set of at
  most `order` faults, each weighted by its exact probability, so the weights fall short of the
  exact distribution by O(p^(order+1)) and equal it at full order; `symbolic=True` gives each
  outcome's probability as a polynomial in a noise scale x, exact to x^order (QuantumClifford's
  symbolic expansion with `degree="full"`). Each fault's effect is found once, by a Pauli frame
  per fault carried through the circuit together, and the sum is a dynamic program, so it runs
  on circuits of any size (a distance-3 surface-code memory to third order in 0.2 seconds for
  two rounds, 4 seconds for three). `perturbative_logical_error_rate(circuit, order, decoder=)` decodes each outcome once:
  a decoder's failure rate with no sampling noise, however small, as a number or a series.
  QuantumClifford's pinned Bell-purification polynomials are reproduced.
- **`stabilizer_qec.clifford`: QuantumClifford.jl's stabilizer-state structure and Clifford
  enumeration**, re-exported at the top level: the clipped gauge (`canonicalize_clip`, `bigram`),
  the partial-trace form (`canonicalize_rref`) and Gottesman's standard form
  (`canonicalize_gott`); `entanglement_entropy` three ways (clipped gauge, partial trace, graph
  state; pure and mixed states) and `mutual_information`; `graph_state` (a stabilizer state's
  graph and the local Cliffords to it), `graph_gate_sequence`, `graph_gate` and
  `Tableau.from_graph`; `enumerate_cliffords` in QuantumClifford's symplectic order,
  `symplectic_gram_schmidt`, `clifford_cardinality`, `enumerate_single_qubit_gates` and
  `enumerate_phases`. Passes QuantumClifford's own tests and pinned values; the entropies
  equal the state vector's Schmidt entropies and the density matrix's, and the enumeration is a
  bijection onto `Tableau.iter_all`.
- **`stabilizer_qec.fusion_blossom`: Fusion Blossom 0.2.13 whole**, a Rust port of its solvers
  (`src/fusion/`): the **serial and parallel minimum-weight perfect matching** solvers
  (`SolverSerial`, `SolverDualParallel`, `SolverParallel`, with partitions and fusion), erasures
  and dynamic weights, the `max_tree_size` blend of union-find and MWPM, its example codes and
  their random errors, the visualizer with its website (`fb.helper.open_visualizer`), JSON and
  Stim's bit-packed prediction. Matchings, subgraphs, dual sums, snapshots, random errors and
  visualizer files are fusion_blossom's exactly; its own tests pass (`tests/fusion_blossom_port/`).
  Weights are 64-bit (its wheels' are 32-bit), and where it would call Blossom V this package's
  exact matcher runs. sinter's `fusion_blossom` decoder uses it. Fixed on the way: with a tree-size
  limit, a tree that grew by absorbing a matched pair collapsed while its root was still locked,
  which deadlocks when, as here, nodes sit behind locks (fusion_blossom's wheels use raw pointers).
- **`stabilizer_qec.mwpf`: MWPF 0.2.12 whole**, a Rust port (`src/mwpf/`) of the **hypergraph
  minimum-weight parity factor** decoder (Wu et al., arXiv 2508.04969): `SolverSerialJointSingleHair`,
  `SolverSerialSingleHair`, `SolverSerialUnionFind` and the BP wrapper, with erasures, heralds and
  override weights; its example codes and their random errors, parity and echelon matrices, the
  visualizer and its HTML export, benchmark suites, its sinter decoder (`SinterMWPFDecoder`),
  heralded error models and reference circuits, and its **command line** (`mwpf benchmark`,
  `matrix-speed`, `test ...`; help, errors and seeded output as MWPF's, run in process by
  `stabilizer_qec.mwpf.run_cli`). Subgraphs, weight bounds, snapshots, random errors, configs and
  their errors are MWPF's (`tests/test_mwpf_package.py`, `tests/test_mwpf_cli.py`); its own Python
  tests pass (`tests/mwpf_port/`), and on circuits its sinter decoder predicts as MWPF's. Its
  linear programs are solved by **HiGHS 1.8.0**, compiled in as MWPF's wheels compile it
  (`src/highs/`, the `highs` feature, on in the wheels): the optimum HiGHS picks among several
  steers MWPF's relaxers, and another solver's choice made some decodes run thousands of times
  longer. A Rust build without the feature solves them with this crate's simplex, which perturbs
  its ratio test so that degenerate programs cannot make it cycle. QEC-Playground's
  codes (MWPF's `QECPlaygroundCode`, on the qecp crate) are not included. The serde_json parser
  shared with the Fusion Blossom port now reports serde_json's errors at serde_json's positions.
- **`stabilizer_qec.conformance`** (Shen and Zhong, arXiv 2609.07035): what can be checked about
  a decoder's answers without knowing the right ones. Each correction must explain its syndrome
  in the caller's indices, with any observables and weight returned being its own; and the same
  instance presented otherwise (faults reordered, detectors renumbered, an extra or shifted
  observable, decoded fresh or again, faults split in parallel halves, weights scaled) must get
  as good an answer. Two feasible answers of different weight prove the heavier is not
  minimum-weight with no solver: a `Certificate`, which `verify` and the standalone
  `tools/verify_certificates.py` re-derive from the model's text. Against an exhaustive oracle
  (up to 22 faults) answers are held to the optimum and to the code's radius. Verdicts follow
  declarations as conformance testing has them: a declared capability is held to, a declined
  one skipped, one the documentation is silent on measured but never failed. `population()` is
  the paper's 4260 syndromes; `panel()` its nine configurations of PyMatching, fusion-blossom,
  ldpc, MWPF and Tesseract, on this package's ports, held to their upstream documentation as the
  paper audited it; `subjects()` this package's decoders, each declaring every capability. The
  paper's findings reproduce (MWPF's union-find returns corrections up to 50% heavier under a
  different detector numbering), and `audit()` of all 19 (1.35 million judged answers, 45 s)
  finds no conformance failure, once three defects it found here were fixed (see Fixed; and
  Relay-BP's declaration, which said its answers depend on the syndrome alone, when its seeded
  stream runs across decodes).
- **`ArgumentReweighting`** (Xie, Yoshioka, Tsubouchi and Li, arXiv 2601.17757): post-selection
  by asking the decoder again. It decodes, makes the correction's faults less likely (each prior
  p to p^b, the paper's ratio test; `test="exact_ratio"` or `"gap"`, its appendices' variants),
  decodes again, and keeps the shot if the second correction is the first (`criterion="pec"`),
  has its logical effect (`"2r-lec"`), or still has it after a third decode with the second
  suppressed too (`"3r-lec"`, the default and the paper's best). Around matching, union-find,
  BP+OSD, BP+LSD or Relay-BP, each thread on its own copy, a shot's changed priors put back after
  it. On a d = 5 surface-code memory at p = 0.2%, 3R-LEC at b = 1.2 rejects 3.7% of shots and
  cuts the kept shots' logical error rate 5.5 times, which PEC needs 18% rejection for; on
  [[72,12,6]] at p = 0.3% BP+OSD's falls 16 times at 11% rejection.
- **`PreDecoder`, the AI pre-decoder** (NVIDIA's, Chamberland, Olle, Li, Thornton and Baratta,
  arXiv 2604.12841, built from the paper, with our own weights): a fully convolutional 3D
  network reads the detection events on the grid of the detectors' coordinates, gives each of
  the model's graph-like faults a probability, applies those above the threshold, and a global
  decoder decodes what is left. It runs on any decomposed model whose detectors have (x, y, t)
  coordinates: inputs per sector (a CSS memory's X and Z detectors) and outputs per kind of
  fault (sector and offset, or boundary), so weights trained on one distance run on any. The
  network, its training (binary cross-entropy against the faults that fired, Lion or Adam, the
  paper's schedule) and inference are in Rust, the same on any number of threads; training is
  in numpy too, step for step the same but for rounding. Two pipelines: the paper's (hard
  corrections, then any decoder on its own weights), and, by default for matching and
  union-find, ours: matching the rest on the network's probabilities (the faults applied at
  1 − q), which at threshold 1/2 is matching the whole syndrome on the network's signed
  weights. `PreDecoder.pretrained` loads weights trained on `memory_circuit`'s rotated memory
  (d = 9, p = 0.006, 3.1M shots; `tools/predecoder.py` trains them again). Over matching alone,
  on 50,000 shots of d × d × d memories at p = 0.003 to 0.006 (`data/predecoder/evaluate.json`):
  the syndrome 7 to 19 times sparser; the paper's pipeline 1.06 to 1.12 times fewer logical
  errors; ours 1.27 to 1.48 times at d = 5, 1.47 to 1.83 at d = 7 and 1.67 to 2.47 at d = 9,
  better than correlated matching from p = 0.004 up at every distance (at p = 0.006, d = 9: 3533
  against its 3984 and matching's 5896), below it at p = 0.003 for d = 7 and 9.
- **Deltakit's leakage dialect and leakage-aware decoding** (Riverlane's `LEAKAGE`, `RELAX`,
  `HERALD_LEAKAGE_EVENT` and `LeakageFlags`). Circuits read the dialect as tagged lines Stim runs
  (`I_ERROR[LEAKAGE(p=…)]`, `I_ERROR[RELAX(p=…)]`, `MPAD[HERALD_LEAKAGE_EVENT(qubits=…)](p)`,
  a record that is 0 when nothing leaks) and `to_deltakit_text()` writes it back. The leakage
  sampler runs Deltakit's model (a leaked qubit is depolarized and read as it stands; a herald
  reads 1 while its qubit is leaked), and `LeakageSampler.sample_measurements` gives Deltakit's
  measurements and leakage flags; `Circuit.without_leakage()` is Deltakit's deinstrumentation,
  loops kept (measurements converted on it are the sampler's detection events, shot for shot).
  **`LeakageAwareDecoder`** takes heralds, leakage flags or herald detectors and, around
  matching, union-find, BP+OSD, BP+LSD or Relay-BP, raises for the shot the priors of the faults
  a leak makes likely (each partner of the leaked qubit since its reset, graded by when the leak
  may have come); on a d = 5 memory with leakage at 0.3% per CX it cuts matching's logical
  errors 4 times, BP+OSD's 7 times and Relay-BP's 3 times. `decoder="local_clustering"` is
  `LocalClustering.from_circuit`, which now also takes heralds' records as triggers.
- **`CompositeDecoder` and `SubgraphDecoder`**, built in (qLDPC's are in `stabilizer_qec.qldpc`):
  `CompositeDecoder` splits a model into the independent parts its faults connect (separate
  patches or blocks, and a decomposed model's X and Z sectors) and decodes each alone with any
  built-in decoder, by name, or a callable; exact, so with matching it is matching, and BP+OSD
  on two blocks is BP+OSD on each. `SubgraphDecoder` decodes chosen detector subgraphs, each
  owning chosen observables, and warns, as qLDPC's does, where two subgraphs' predictions would
  cancel or a detector is in none.
- **`CollisionClustering` and `LocalClustering`** (Barber et al., arXiv 2309.05558, and Ziad et
  al., arXiv 2411.10343: Riverlane's hardware union-finds, built from their papers). Collision
  Clustering grows a ball per detection event and merges two clusters when a pair of their balls
  overlaps, every pair compared each round; the Local Clustering Decoder runs the paper's
  growing, merging, picking and syncing stages on per-vertex radii and cluster indices. Their
  clusters are union-find's, and without pre-grown edges their predictions are the same, shot for
  shot. Unweighted by default, as the hardware is (under SD6 the curves cross at 0.55%, the LCD
  paper's figure); `weighted=True` scales the weights to steps (crossing near 0.68%, close to
  matching's). `LocalClustering` takes an adaptivity map of triggers to edges it pre-grows, fired
  by `decode_batch(heralds=...)`, and `LocalClustering.from_circuit` builds the paper's leakage
  map, one trigger per measurement: with `compile_leakage_sampler`'s heralds it cuts failures 2 to
  12 times on leaky surface-code memories. `return_stats` gives the rounds, the pairs compared or
  stage passes run, and the edges pre-grown; sinter and the command line know them as
  `collision_clustering` and `local_clustering`.
- **`AmbiguityClustering`** (Wolanski and Barber, arXiv 2406.14527; Riverlane's, built from the
  paper): BP, then an elimination that goes only as far as the syndrome needs, grown by `kappa`
  times the fault count further columns into blocks of the matrix decoded apart, ambiguous blocks
  searched over every choice of at most two non-pivot columns and each observable set by summed
  probability. At `kappa=0` it is BP+OSD-0 on the same BP, shot for shot; at the default 0.05 it
  beats BP+OSD-CS(7) on the bivariate bicycle codes' memories (on the gross code at p = 0.3%, 26
  failures in 20,000 shots against 32) and runs 8 to 100 times faster. `return_stats` gives each
  shot's blocks, ambiguous blocks and solutions searched; sinter and the command line know it as
  `ambiguity_clustering`.
- **`stabilizer_qec.chromobius`: Chromobius 1.1.1's API** (`CompiledDecoder.from_dem`,
  `compile_decoder_for_dem`, `predict_obs_flips_from_dets_bit_packed`,
  `predict_weighted_obs_flips_from_dets_bit_packed`, `sinter_decoders`) on `ColorMatching`.
  The weights are now Chromobius's bit for bit: PyMatching's discretized Möbius matching
  weight (2²⁴ levels, divided by its normalising constant) as float32. `ColorMatching`'s
  `return_weights` gives the same weights, which were the matcher's own scale and differed in
  the last digits.
- **`stabilizer_qec.beliefmatching`: beliefmatching 0.2.0 whole**, on this package's ldpc and
  PyMatching (`BeliefMatching`, `detector_error_model_to_check_matrices`, `DemMatrices`,
  `BeliefMatchingSinterDecoder`); `BeliefMatching` gains `from_stim_circuit` and
  `from_detector_error_model`, and takes a circuit (decomposing its model), as beliefmatching's.
- **`MbpDecoder`** (also `mbp_decoder`): memory belief propagation over GF(4) (Kuo and Lai), as
  ldpc's `mbp_decoder`, whose C++ ldpc 2.4.1 does not ship (it is on ldpc's main branch and no
  longer builds). Ported as written and checked bit for bit, log-likelihood ratios included,
  against that C++ built here (`tools/mbp_reference.cpp`, `tests/test_mbp.py`): X, Y and Z
  errors decoded together on a GF(4) matrix or on `HX_CSS` and `HZ_CSS`; `alpha_parameter`
  (inhibition, per Pauli and qubit), `beta_parameter`, product-sum or min-sum with
  `gamma_parameter`; `uf_decode` falls back on union-find weighted by MBP's posteriors.
- **PyMatching's command line**: `stabilizer-qec predict` and `stabilizer-qec count_mistakes`
  (and `python -m stabilizer_qec.pymatching`, `stabilizer_qec.pymatching.cli`), with
  PyMatching's flags and output (error messages are this package's: prefixed with
  `stabilizer-qec <command>: `, and this package's reader's and parser's for malformed shot
  data and models; the timings are measurements); `predict` also takes
  `--in_includes_appended_observables`, which PyMatching 2.4's fails on.
- **Conformance**: all 2,238 examples in Stim 1.16's docstrings pass (`tests/test_stim_doctests.py`),
  and every public Stim name, method and parameter exists here (`tests/test_stim_api_surface.py`).
  `PauliString` and `Tableau` pickle, as Stim's do; `Circuit`, `DetectorErrorModel` and
  `append_from_stim_program_text` take Stim's keyword names (`stim_program_text`,
  `detector_error_model_text`; 1.x's `text=` still works); `read_shot_data_file` has
  `separate_observables` and `bit_pack`.
- **The command line is Stim's in full**: `repl` (byte for byte as Stim's), `help gates`,
  `help formats`, `help <gate>`, `help <format>` and the `*_markdown` topics (our own text,
  from the gate table); `<command> --help`; Stim's older invocations (`--sample=10`,
  `--detect`, `--help`, ...) and deprecated flags (`--prepend_observables`, `--frame0`,
  `--detector_hypergraph`), with Stim's rules for combining them; `m2d --ran_without_feedback`;
  `analyze_errors --allow_gauge_detectors` and `--block_decompose_from_introducing_remnant_edges`.
  `stabilizer_qec.main(command_line_args=...)` runs it from Python, as `stim.main`.
- **`allow_gauge_detectors` and `block_decomposition_from_introducing_remnant_edges`** on
  `Circuit.detector_error_model` (and `DemOptions` in Rust): a gauge detector becomes Stim's 50%
  error, removed from the walk as Stim removes it, with the collapse checked at each
  measurement as well as each reset (the model is Stim's, character for character, on random
  circuits with gauges).
- **The explained-error classes are Stim's**: `GateTargetWithCoords.gate_target` is a
  `GateTarget` and `DemTargetWithCoords.dem_target` a `DemTarget` (1.x: strings); every class
  has Stim's keyword constructor, text, repr, equality and hash; `flipped_measurement` is None
  when nothing is measured; a location's text is computed from its fields, as Stim's.
- **Stim's diagram surface**: `timeline-3d` and `matchgraph-3d` (circuits and models) as glTF,
  byte for byte as Stim writes them; every `-html` type (a viewer page); `interactive` (a page
  opening the circuit in Crumble); `filter_coords` (detectors, observables, coordinate
  prefixes); Stim's type aliases; the diagram object's notebook display as Stim's.

- **Stim's stabilizer objects**: `PauliString` (and `PauliStringIterator`), `Tableau` (and
  `TableauIterator`), `TableauSimulator`, `GateData` / `gate_data`, `GateTarget` and every
  `target_*` helper, `CircuitInstruction`, `CircuitRepeatBlock`, `Flow`. Every example in Stim's
  own docstrings for these passes (`tests/test_stim_doctests.py`, 1,269 examples).
- **Flows**: `Flow` (Stim's text form and product), `Circuit.has_flow`, `has_all_flows`,
  `flow_generators`, `solve_flow_measurements`, `time_reversed_for_flows`,
  `missing_detectors`. The signed flow check runs once, exactly (each input qubit entangled
  with a reference), where Stim samples 256 random inputs.
- **Circuit transformations** as Stim writes them: `decomposed`, `flattened`,
  `flattened_operations`, `without_noise`, `without_tags`, `inverse`, `with_inlined_feedback`;
  and `num_ticks`, `get_detector_coordinates`, `get_final_qubit_coordinates`,
  `count_determined_measurements`, `reference_detector_and_observable_signs`, `to_tableau`,
  `to_file`.
- **The detector error model's objects**: `DemTarget`, `DemInstruction`, `DemRepeatBlock`,
  `target_relative_detector_id`, `target_logical_observable_id`, `target_separator`; a model is
  a list (indexing, slicing, `append`, `clear`, `copy`) with Stim's `+`, `*`, `approx_equals`,
  `rounded`, `get_detector_coordinates`, `without_tags`, `to_file`.
- **Stim's sampler surface**: the `Compiled*` class names (`CompiledMeasurementSampler`,
  `CompiledDetectorSampler`, `CompiledDemSampler`,
  `CompiledMeasurementsToDetectionEventsConverter`) as aliases; `sample_bit_packed`;
  `sample_write` on every sampler (every Stim format; observables, errors and replays to their
  own files) and `convert_file` on the converter, each writing the bytes Stim writes;
  `compile_sampler(reference_sample=...)`; `compile_m2d_converter(skip_reference_sample=True)`;
  `prepend_observables`, `dets_out` and `obs_out` on the detector sampler; Stim's reprs.
- **`FlipSimulator`**, Stim's interactive Pauli-frame simulator over a batch of instances, with
  every method (`do`, `peek_pauli_flips`, `set_pauli_flip`, `broadcast_pauli_errors`,
  `append_measurement_flips`, the flip getters, `to_numpy`, `generate_bernoulli_samples`,
  `copy`, `clear`), its state as bit-packed rows in Rust (`clifford::flip_sim`). With
  stabilizer randomization off, flips propagate bit for bit as in Stim (resets, measurements,
  MPP, SPP and feedback included); noise matches Stim's statistics.
- **`CliffordString`**: a single-qubit Clifford per qubit, with Stim's products, powers,
  concatenation, slicing, `x_outputs` / `y_outputs` / `z_outputs`, conversion from circuits
  and Pauli strings, `random` and `all_cliffords_string`; whole-string operations are table
  lookups over a numpy array.
- **Stim's exports and searches**, each written as Stim writes it: `Circuit.to_qasm`
  (OpenQASM 2 and 3), `to_quirk_url`, `to_crumble_url` (with `mark` drawing explained errors),
  `detecting_regions` (target filters by name, kind or coordinate prefix; tick selection),
  `shortest_error_sat_problem` and `likeliest_error_sat_problem` (WDIMACS max-SAT).
  `CircuitErrorLocationStackFrame.iteration_index` is now Stim's: the pass of the loop enclosing
  the frame, so the outermost frame's is 0.
- `Circuit` as a list: `len`, indexing and slicing (instructions and `CircuitRepeatBlock`s),
  `insert`, `pop`, `clear`, `append_operation`, `approx_equals`.
- In Rust: `clifford::{PauliString, Tableau, TableauSimulator}`, Stim's conversions
  (`clifford::convert`: stabilizers, state vectors, unitaries, circuits, graph states, random
  Cliffords) and its instruction-level circuit (`clifford::ir`).
- **The stabilizer-rank sampler** (`Circuit.compile_stabilizer_rank_sampler`, Rust
  `StabilizerRankSampler`): Clifford+T circuits at any qubit count, exactly, the cost growing
  with the number of non-Clifford gates (`I[T]`, `I[T_DAG]`, rotations and `U3` written as
  gates) rather than of qubits, as Tsim samples them. The state is a sum of Clifford branches,
  one per coset of the reference run's stabilizer group the non-Clifford gates reach (at most
  2^(T count); seven T gates on the Steane code's |0⟩ are eight terms, fifteen on the
  [[15,1,3]] code's thirty-two), on a reference tableau that is the same in every shot;
  Pauli noise, measurement flips, heralds and feedback run as the frame sampler's.
  `sample` (detection events), `sample_measurements`, `probability_of` (a record's exact
  probability), `peak_terms`; `exact_distribution(method="stabilizer_rank")`. Equal to the
  state vector's distribution bit for bit on random Clifford+T circuits with noise,
  measurements in every basis, resets and feedback (`tests/test_stabrank.py`). **A circuit's
  samplers run its gates**: `compile_sampler` and `compile_detector_sampler` on a circuit with
  non-Clifford gates now give this sampler's shots (1.9's ran Stim's reading, the identity;
  `without_tags()` keeps it). Coherent errors (`I_ERROR[R_Z(…)]`) are still not gates:
  `compile_coherent_sampler` is theirs. `Circuit.num_non_clifford_gates()` counts the gates.
- **Tsim's circuit surface**: `Circuit.tcount()` (Tsim's count: every non-Clifford phase),
  `Circuit.is_clifford`, `Circuit.to_matrix(endian="big")` and `to_tensor()` (the unitary of a
  circuit of gates, Clifford or not, up to 10 qubits, by the state vector), `Circuit.inverse()`
  of tagged gates (`I[T]` ↔ `I[T_DAG]`, angles negated, `U3(θ, φ, λ)⁻¹ = U3(−θ, −λ, −φ)`),
  `Circuit.detector_error_model(allow_non_deterministic_observables=True)` (Tsim's
  conversion of an observable a non-Clifford gate leaves random, ported in
  `stabilizer_qec.tsim.get_detector_error_model`), and `stabilizer_qec.tsim.Circuit(text)`,
  which reads Tsim's shorthand (`T 0`, `TPP X0*Z1`, `R_Z(0.25) 0`, `CCZ 0 1 2`) or tagged text
  as a circuit of this package. Every gate matrix Tsim tests its gates against (its
  `gate_matrices.py`, in `tests/tsim_port/`) is `to_matrix`'s up to a global phase, T's and the
  rotations' exactly; Tsim's `test_dem.py` and the `test_circuit.py` cases of these methods are
  ported (`tests/test_tsim_surface.py`). Against Tsim itself where it is installed
  (`tests/test_tsim_differential.py`, `tools/clifford_t.py check`): its demo circuits' records
  within the error bars of our exact probabilities, `tcount`, `is_clifford`, `to_matrix` and
  `inverse` agreeing on random shorthand circuits.
- **Clifford+T circuits with coherent noise on every gate** — the 2.0 headline. The joint
  sampler (`compile_coherent_sampler` on a circuit with non-Clifford gates; Rust
  `JointSampler`) runs the non-Clifford *gates* (`I[T]`, `I[R_Z(theta=…)]`, `I[U3(…)]`)
  exactly, as the stabilizer-rank sampler does, draws the coherent *errors*
  (`I_ERROR[R_Z(…)]`, `II_ERROR[R_ZZ(…)]`) as their Pauli twirl, and weights each shot by the
  interference the twirl drops: 1 + Σ_G ρ_G ⟨Ψ_{S⊕G}|Ψ_S⟩/‖Ψ_S‖² over the error sets G the
  spacetime kernel relates to the shot's, every non-Clifford gate a location in the kernel's
  search so that the interference running through the T gates is found, each member's state a
  re-run of the exact engine with its errors inserted and the shot's outcomes forced, the
  overlaps dot products of the two runs' terms in one Pauli frame (the frame carries its
  phase). Neither Tsim (cost exponential in every rotation) nor 1.9's coherent sampler (a
  Clifford reference circuit) covers such circuits. The identity is proved as 1.9's was: with
  the whole kernel, the weighted twirl summed over every branch equals the state vector's
  distribution on random circuits of T gates, U3, rotations, coherent errors, Pauli noise,
  resets and mid-circuit measurements (`coherent::joint::tests`, and
  `exact_distribution(method="coherent")` from Python); the truncation (the kernel's local
  generators, products of up to `order` touched ones per cluster, products within each group
  of equal-flip locations) is 1.9's, as a pairwise sum rather than 1.9's product over
  clusters, since members of different clusters overlap through a T gate. The samplers'
  reading of the tags: a rotation written on `I`/`II` is a gate (exact), on `I_ERROR`/`II_ERROR`
  an error (twirled and reweighted); `twirled()` now twirls the errors alone and keeps the
  gates; 1.9 treated `I[R_Z(…)]` as an error too. The experiments (`tools/clifford_t.py
  inject|distill`, `data/clifford_t/`, the README's tables): under an over-rotation after every
  gate, the twirl under-reads the infidelity of a |T⟩ state injected into the Steane and
  [[17,1,5]] colour codes by two to three times (Steane, ε = 0.02: 0.0015 against the exact
  0.0043; ε = 0.1: 0.040 against 0.094) and the joint sampler reads it within sampling error;
  on Tsim's 5-qubit distillation protocol the twirl turns a working protocol into a broken one
  (ε = 0.01: acceptance 11.6% and output infidelity 2 × 10⁻² against the exact 16.8% and
  2 × 10⁻⁵), the joint sampler puts the acceptance back (16.6%) and the infidelity within a
  factor of 25 at ε ≤ 0.02 and three orders of magnitude at ε = 0.05 (a cancellation-dominated
  quantity the truncation does not reach), and in
  the protocol encoded in the colour code (85 qubits, 801 rotations) it is the only estimate
  there is, at 130 ms per shot.
- **`stabilizer_qec.tqec`: tqec's block graphs** (tqec/tqec 0.2.0's `computation` package, its
  gallery and its Collada and JSON files, ported on the standard library without networkx, PyZX,
  pycollada or numpy; Apache 2.0): `BlockGraph` with cubes, pipes and ports, the ZX kinds,
  `validate`, `fill_ports`, `fill_ports_for_minimal_simulation`, `compose`, `shift_by`,
  `rotate`, `fix_shadowed_faces`, `to_zx_graph` on a ZX graph of this package's own and
  `find_correlation_surfaces` (tqec's search, generating set for generating set — the PyZX
  adjacency order it depends on is reproduced), `to_json`/`from_json`, `to_dae_file`/
  `from_dae_file` (tqec's node names, scales, colours and correlation-surface rectangles),
  the seven gallery builders (the Steane encoding from tqec's Collada file, packaged). On every
  gallery graph, open and filled, the cubes, pipes, ports, volumes, correlation surfaces,
  stabilizers, JSON text and Collada files (read both ways) equal tqec's. `BlockGraph.compile`
  (and `compile_block_graph`, with the program) runs a graph on this package's surgery engine
  through the new `_core.surgery_program`: each cube lifetime a patch prepared and read in its
  temporal bases, each spatial pipe a merge and split, each correlation surface an observable
  from the final logicals, merge outcomes and seam readouts it touches; memory and the CNOT
  compile (the CNOT's decoded logical error rate equals the engine's own CNOT's), and graphs
  with Hadamard pipes, spatial cubes, Y half cubes or open ports are refused with tqec named.
  tqec's plaquette templates and detector search are not ported.
- **`stabilizer_qec.loom`: Loom's Eka model, interpreter, validator and exporters** (Entropica
  Labs' el-loom 0.4.0, ported on plain dataclasses without pydantic, igraph, plotly or scipy;
  Apache 2.0): `Lattice`, `PauliOperator`, `Stabilizer`, `Block` with Loom's validation,
  `SyndromeCircuit` and Loom's circuit IR (channels, nested circuits with its timing and
  unrolling), the operations (reset, syndrome rounds, logical measurements and Paulis,
  conditional Paulis, grow, shrink, merge, split; Loom's other operations are defined for
  interchange and refused), `Eka` with Loom's dict form (its JSON reads here and back),
  `RotatedSurfaceCode.create` (stabilizers, ancillas, schedules, syndrome circuits, logicals,
  charges, boundaries and corners equal to Loom's on 72 configurations), `interpret_eka` with
  Loom's syndromes, detectors, logical observables, stabilizer and logical evolutions and
  corrections, `EkaToStimConverter` (this package's `Circuit`) and `eka_to_circuit`, and the
  `EkaToQasmConverter`, `EkaToCudaqConverter` and `EkaToPennylaneConverter` text exporters.
  On fourteen recorded experiments (memories, merges horizontal and vertical and at distance
  5, merge and split, a conditional Pauli after a split, grow, shrink, parallel operations) the
  Stim text equals Loom's byte for byte, except the shrink, where Loom orders a block's
  stabilizers by a Python set of strings (hash-randomised run to run) and the gates, detectors
  and observables are the same; the QASM and CUDA-Q texts equal Loom's, PennyLane's up to the
  wire-register ids. The validator (`is_circuit_valid`, `is_syndrome_extraction_circuit_valid`,
  `is_logical_operation_circuit_valid`) runs on `TableauSimulator` with Loom's three checks
  and gives Loom's verdicts. Not ported: the auxiliary CNOT, transversal Hadamard, move,
  rotate, Y-wall phase gate and state injection applicators, base and logical operations,
  if-else circuits, error models, and the Mimiq and Guppy exporters.
- **`stabilizer_qec.estimate`: Qualtran's and Azure's resource estimators, to the bit.**
  `estimate.qualtran` is `qualtran.surface_code` (Qualtran 0.7.0, Apache 2.0) on dataclasses
  and numpy: `GateCounts`, `AlgorithmSummary`, `QECScheme` (Gidney–Fowler, Beverland et al.),
  `LogicalErrorModel`, `PhysicalParameters`, the simple, compact, intermediate and fast data
  blocks, `CCZ2TFactory`, `FifteenToOne` with Litinski's density-matrix error model,
  `MultiFactory`, the rotation cost models, `PhysicalCostModel` with its `make_gidney_fowler`
  and `make_beverland_et_al`, `get_ccz2t_costs`, `get_ccz2t_costs_from_error_budget`,
  `get_ccz2t_costs_from_grid_search`, `iter_ccz2t_factories`, `iter_simple_data_blocks`, and
  Beverland et al.'s `minimum_time_steps`, `code_distance` and `t_states`; every number
  Qualtran's tests pin (Litinski's 15-to-1 table, the Gidney–Fowler spreadsheet, Lee et al.'s
  THC grid search, Beverland's three algorithms) and, with Qualtran installed, its outputs,
  equal. `estimate.azure` is the QDK's resource estimator (microsoft/qdk `resource_estimator`,
  MIT): `azure_estimate(counts, qubit_params, qec_scheme, error_budget, constraints,
  distillation_unit_specifications)` takes the QDK's parameters — the six qubit presets or a
  custom gate-based or Majorana qubit with the QDK's defaulting, `surface_code` and
  `floquet_code` or a custom scheme's threshold, prefactor, cycle-time and qubit formulas, a
  total error budget or its three parts, the 15-to-1 RM-prep and space-efficient units or
  custom units with their formulas, `logicalDepthFactor`, `maxTFactories`, `maxDuration`,
  `maxPhysicalQubits`, `maxDistillationRounds` — and returns the QDK's result dict
  (`physicalCounts` with its breakdown, `logicalQubit`, `tfactory`, `errorBudget`,
  `logicalCounts`), or with `estimate_type="frontier"` the QDK's `frontierEntries` along the
  qubits–runtime Pareto frontier. Inside are the PSSPC layout, the code-distance inverse, the
  round-based factory with the `probability` crate's binomial quantile (its mass-function
  sums, Newton steps and Cornish–Fisher expansion), the exhaustive search over unit sequences
  and non-decreasing code distances with the QDK's bisection, its iteration order and its
  Pareto population, the three estimates (unrestricted, under a duration cap, under a qubit
  cap) and the frontier. On 55 configurations — the presets by scheme by three count sets, the
  constraints, custom qubits, schemes, units and budgets, the no-T and one-T algorithms, two
  refusals, five frontiers — every reported number equals the QDK's
  (`tests/estimate_port/qdk_estimates.json`, qdk 1.33.1): physical qubits, runtime, rqops,
  the required error rates, the factory's rounds, units, distances, qubits, durations and
  output error, every frontier entry in order; and live against the installed QDK. Not
  ported: the profiling.
- **The lattice-surgery CNOT's threshold under leakage** (Plaquette's experiment;
  `tools/surgery_leakage.py` → `data/surgery/leakage.json`, a README table): SD6 noise at p
  with `LEAKAGE(f·p)` after every CX on both qubits and `RELAX(0.1)` seepage, d = 3, 5, 7,
  20,000 shots a point, decoded by matching blind to the leakage and leakage-aware with the
  sampler's heralds; the threshold by the crossing of the two largest distances and by
  qec-util's Wang–Harrington–Preskill fit with a bootstrap interval. Leakage at 3%, 10% and
  30% of the error rate takes the threshold from 0.49% to 0.45%, 0.37% and 0.23%; at 50% an
  unrecorded 3,000-shot run found the distance-5 CNOT no better than the distance-3 one at any
  p (too few shots to place a crossing); heralds cut the logical error by up to a tenth (two to
  five per cent at d = 7, p = 0.3%) and move the threshold by less than its interval.
- **`stabilizer_qec.realtime`: decoding in the background, a decoder queue with statistics,
  and a decoder server** (CUDA-Q QEC's `decode_async`, `enqueue_syndromes`, `get_corrections`,
  `reset_decoder` and decoder statistics; its real-time host API). Every detector-error-model
  decoder has `decode_async(shots)`: the batch runs on a worker thread with the GIL released in the engine
  and returns an `AsyncResult` (`ready`, `wait`, `get`, its latency and service time).
  `DecoderQueue(decoder, workers=n)` takes `enqueue_syndromes` (a ticket back),
  `get_corrections` (by ticket, or the oldest uncollected batch, in order), `pending`, `reset`
  and `stats()`: a `DecoderStats` with shots and batches decoded, latency from enqueue to
  result (mean, median, 99th percentile, maximum), the decode's own time, shots per second and
  the deepest the queue got — fed faster than it decodes, the latency shows it.
  `cudaq_qec.configure_decoders`, `enqueue_syndromes`, `get_corrections`, `reset_decoder` and
  `decoder_stats` are the host API over it, for this package's decoders and CUDA-Q QEC's
  `Decoder` objects alike. `stabilizer-qec serve --dem model.dem --decoder matching --tcp
  host:port` (or `--stdio`, `--circuit`, `--workers`) serves a decoder over a framed protocol
  (`realtime.Protocol`: a 20-byte header and b8 syndromes in, b8 predictions out; `STATS`,
  `RESET`, `ERROR` frames), and `DecoderClient.connect(host, port)` or
  `DecoderClient.spawn(dem, decoder=...)` (a subprocess on its pipes) decodes through it with
  `decode_batch`, `decode`, `stats` and `reset`. Over TCP and over the pipes the predictions
  equal the local decoder's shot for shot (Relay-BP's per batch: it seeds a shot's memory
  strengths from its place in its batch).
- **Sampling on the GPU, bit for bit the CPU's** (cuStabilizer's place; `device="gpu"`).
  `compile_detector_sampler(device=...)` and `compile_sampler(device=...)` take `"cpu"` (the
  word sampler as before), `"cpu-lanes"` or `"gpu"`. The lane sampler (`src/lanes.rs`) is the
  batch sampler's program flattened into a stream of 32-bit words — 32 shots a word, every
  random number an integer compared with a threshold, a per-word xorshift32 seeded from the
  sampler's seed and the word — and the GPU (`src/gpu/sample.wgsl`, through wgpu on Metal,
  Vulkan or DX12; Cargo feature `gpu`) interprets the very same stream, one invocation per
  word, REPEAT blocks on a loop stack, so its bytes equal the CPU lane sampler's for every seed,
  shot count and batching (checked on memories, the colour code, the repetition code, the
  surgery CNOT and a circuit with every channel, feedback, heralds and nested loops). The lane
  devices draw different shots from the same seed than `"cpu"` does (the same distribution;
  every detector's rate and the decoded logical error rate agree within sampling error).
  `stabilizer_qec.gpu_device()` names the device or returns None (no `gpu` feature in the
  build, or no adapter); `device="gpu"` then raises. Leakage and the non-Clifford samplers stay
  on the CPU.
- **Min-sum BP on the GPU.** `BpDecoder(..., device="cpu-f32")` runs the parallel min-sum
  schedule in single precision (`src/bp32.rs`: ldpc's operation order, the adaptive scale
  tabulated per iteration), `device="gpu"` the same on the GPU (`src/gpu/bp.wgsl`), equal bit
  for bit — hard decisions, convergence and iteration counts — over thousands of syndromes at
  every scale; `BpDecoder.decode_batch(syndromes)` decodes rows at once on any device. The f32
  decoder gives the f64 decoder's decision on over 98% of a d = 5 memory's syndromes; the f64
  decoder (`device="cpu"`, every schedule, product-sum too) is unchanged. `tools/gpu_bench.py`
  records shots per second of each sampler and decoder in `data/gpu/bench.json`.

- **The site edits circuits** (section 17, Figure 30; Crumble's job in the page): qubits on a
  grid at their coordinates and a timeline of layers, a palette of gates placed by clicking
  (two-qubit gates control first), layers added and removed, Pauli markers dropped before a
  layer and propagated live by Clifford conjugation through H, S, √X, CX, CZ, SWAP and the rest
  to the measurements they flip and so to the detectors and observables, which the table names
  per marker; DETECTOR and OBSERVABLE placed on a qubit's newest measurement; the Stim text
  rewritten on every edit and loaded back; the Crumble URL (`Circuit.to_crumble_url`'s format,
  markers between TICKs as Stim writes them) written and read, and opened in Crumble. Checked in
  CI by `tools/site-tests.mjs` against `tests/editor_port/crumble.json`, which
  `tools/editor_fixtures.py` writes from the package: the page's URL equals the package's for
  every example (a Bell pair, repetition, surface and colour code memories, a noisy memory) and
  reads back to the same text, and every marker on every qubit of every layer flips exactly the
  detectors and observables the engine's sampler flips with the same error inserted (about
  700 cases). Noise and anything the editor does not draw pass through the text untouched.

### Fixed

- `Circuit.num_non_clifford_gates()`, `tcount()` and `is_clifford` count through REPEAT blocks
  by multiplication instead of unrolling the circuit, so `compile_detector_sampler` (which asks
  first) no longer refuses a memory too long to unroll (a 10,000-round d = 11 memory: 28 million
  instructions, past the engine's 2²⁴).
- A circuit takes an anti-Hermitian Pauli product (`MPP X0*Z0`), prints it and counts its record,
  and refuses to simulate or analyse it, as Stim does; building it was the error.
- A model of a circuit with a non-deterministic detector or observable is refused with Stim's
  opening lines ("The circuit contains non-deterministic observables."), then ours and every
  detector and observable the collapse randomises.
- relay_bp: `ObservableDecoderRunner` takes an observable matrix of any width, as relay_bp's does
  (computing observables with a wrong one is the error); the signatures show `...` for the two
  defaults relay_bp's do (`gamma_dist_interval`, `stopping_criterion`).
- ldpc: `BpLsdDecoder(random_serial_schedule=True)` keeps it, as ldpc's does; it was dropped.
- Editing a `DetectorErrorModel` (`append`, `insert`, indexing, `copy`, pickling) keeps its
  numbers exact; each edit went through its printed text, which rounds them to Stim's 16 digits,
  so a merged probability could lose its last bit. `==` compares the numbers exactly, as Stim's.
- `DetectorErrorModel.append` parses only what it appends; it re-read the whole model each
  time, so building one instruction by instruction was quadratic (Tesseract's error
  decomposition of a distance-5, 15-round surface code took 7 minutes; now half a second, as
  on Stim).
- The `ldpc`-style decoders (`BpDecoder`, `BpOsdDecoder`, `BpOsd`, `BeliefMatching`) take
  ldpc's aliases for `bp_method` (`"ps"`, `"ms"`, ...), as `BpLsd` already did.
- Correlated matching, `BeliefMatching` and `UnionFind` decode models with a fault that flips
  observables and no detector (`error(p) L0`, which 2.0 keeps as Stim does), ignoring it as
  PyMatching does; they refused such models.
- Found by `stabilizer_qec.conformance`: `UnionFindDecoder`'s peeling (and `BeliefFindDecoder`'s)
  peeled the boundary through the last boundary bit it met, whether its spanning tree took it or
  not, as ldpc 2.4.1 does, which leaves corrections that do not explain the syndrome (78 of the
  distance-5 rotated surface code's 4096); there it now peels through the tree's boundary bit
  (`last_stuck`), and elsewhere still equals ldpc's. `Matching.decode_to_edges_array(...,
  enable_correlations=True)` returned the uncorrelated matching's edges; it returns the
  correlated matching's, whose observables are `decode`'s, as PyMatching's (they differed on 77
  of 2000 circuit-level shots).
- `LeakageSampler.sample`'s heralds have one entry per measurement record, padded records and
  heralded errors' included (they had one per measurement only, out of step with the record);
  `LocalClustering.from_circuit` no longer reads an `MPAD`'s values as qubits.

- Found by a system-wide adversarial pass after W8 (eight agents, each against a reference):
  - Leakage: two heralds in a row fused into a crash, and `append` and `+` never merged them;
    an inverted herald target (`!q`) read 1 when the qubit was not leaked; a probability kept
    its spelling in the lowered tag, so equal circuits compared unequal; `LEAKAGE` with no
    targets printed lowered; `deltakit_stim`'s `without_tags` removed Deltakit's instructions
    (its Stim keeps them). An empty `repeat` block in a model prints as Stim prints it.
  - The density matrix: a chain of correlated errors was grouped wrongly after a stray
    `ELSE_CORRELATED_ERROR`; small seeds were biased; feedback in one `do` could not read the
    records of the previous; a circuit with feedback branched on every record, not only those
    fed back, and observables blocked `density_matrix()`. A `stim.PauliString`'s sign was
    dropped, an imaginary observable read 0 (Stim's error now), and the fidelity lost about
    1e-8 to the square roots of round-off (a unitary's process fidelity with itself came out
    above 1).
  - PECOS's simulators: the density and process matrices took NumPy angles badly (the gate text
    held `np.float64(…)`), qubits past the simulator (and a process matrix's reference qubits),
    NumPy integer qubits not at all, a one-qubit gate on two qubits, and `forced_outcome=2`;
    unseeded, they ignored the runner's seed. `cySparseSim.logical_sign` crashed, and
    `pySparseSim`'s (PECOS's own) dropped the −1s of reordering X and Z when multiplying
    generators, so `Y0 Y1` on a Bell-like state read −1 and `Y0` on |+i⟩ returned 2. A Choi
    matrix given to `convert_process` was not shape-checked.
  - The perturbative expansion: a stray `ELSE_CORRELATED_ERROR` panicked, `DEPOLARIZE1` above
    3/4 and more than 64 observables were refused, a known tag's positional arguments were
    dropped, and a huge `degree` overflowed. The CHANGELOG's timing for it was for two rounds.
  - Stim's surface: `TableauSimulator.do` panicked on an odd number of pair targets and took
    `CX 0 0`; `flow_generators` refused `I_ERROR`; the text timeline drew `MPAD`'s arguments,
    `QUBIT_COORDS`, shifted coordinates and where detectors and observables sit unlike Stim; a
    model printed `detector(-0)` as `0`; a replay returned its padding bits.
  - Decoders: `beliefmatching.BeliefMatching(stim.Circuit)` crashed; `BeliefFindDecoder` skipped
    BP on a zero syndrome (ldpc's `iter` and `log_prob_ratios` come from that run); a received
    vector set after construction was decoded as a syndrome by BP+OSD, BP+LSD and BeliefFind;
    correlated matching refused `error(0)` and `D0 D1 ^ D0 D1`.
  - Robustness: `copy.copy` of a `TableauSimulator`, `FlipSimulator` or
    `DensityMatrixSimulator` shared its state with the original (it copies now; Stim's
    raises); a `FlipSimulator`, `DensityMatrixSimulator` or `PreDecoder` used from two threads
    raised "Already borrowed" (calls wait their turn now); `detslice-with-ops-svg` failed past
    the last tick; `DetectorErrorModel.get_detector_coordinates` walked every pass of a loop
    (a `repeat` of 10^12 hung; it skips passes declaring no wanted detector) and kept a
    detector's last declaration, where Stim keeps its first; `Matching` of a model naming a
    detector past the largest index panicked; the CLI's `diagram` ignored `--remove_noise` and
    refused `--tick A:B` (read now as Stim's command line reads it: a `detslice-text` at the
    range's first tick, a `timeline-text` ignoring it); the `hits` format set a bit named twice,
    where Stim toggles it.

## 1.9.0 — 2026-10-09

The Python package and the Rust crate both at 1.9.0, "beyond Pauli noise". The headline is
sampling and decoding any Stim circuit with coherent errors on every gate, at sizes a state
vector cannot reach, checked exactly against a state vector where it can. (Tsim, arXiv
2604.01059, samples circuits with a few non-Clifford gates at any size, at a cost exponential in
their number; noise on every gate is out of its reach. An earlier wording of this entry said no
other tool samples coherent errors beyond a state vector, which overlooked Tsim.) API additions
only.

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

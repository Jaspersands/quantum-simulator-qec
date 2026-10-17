# Changelog

All notable changes to `stabilizer-qec`. The Python package follows
[semantic versioning](https://semver.org) from 1.0. Anything deprecated warns for at least one
minor release before a major release removes it. The Rust crate is versioned on its own.

## 2.0.0 — unreleased

Everything every other QEC tool has (the inventory in
`docs/superpowers/notes/2026-10-09-parity-inventory.md`), and Clifford+T circuits with coherent
noise on every gate. In progress on `feature/2.0`.

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
- **A reference sample takes every random measurement as 0, as Stim's does** (it was a seeded
  draw): `reference_sample()`, the samplers' references and the measurement-to-detection
  converter, which now takes detectors and observables that are not deterministic (gauge
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
  its own tests (`tests/panqec_port/`, 789 with its slow ones and its doctests): 2D and 3D toric,
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
  `SparseSim`) is ported to Rust: on random circuits of every gate, initialization and
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
  PyMatching's flags and output byte for byte; `predict` also takes
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
  2 × 10⁻⁵), the joint sampler puts the acceptance back and the infidelity within one to two
  orders of magnitude (a cancellation-dominated quantity the truncation does not reach), and in
  the protocol encoded in the colour code (85 qubits, 801 rotations) it is the only estimate
  there is, at about a second per shot.
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
  ported: the report text and the profiling.
- **The lattice-surgery CNOT's threshold under leakage** (Plaquette's experiment;
  `tools/surgery_leakage.py` → `data/surgery/leakage.json`, a README table): SD6 noise at p
  with `LEAKAGE(f·p)` after every CX on both qubits and `RELAX(0.1)` seepage, d = 3, 5, 7,
  20,000 shots a point, decoded by matching blind to the leakage and leakage-aware with the
  sampler's heralds; the threshold by the crossing of the two largest distances and by
  qec-util's Wang–Harrington–Preskill fit with a bootstrap interval. Leakage at 3%, 10% and
  30% of the error rate takes the threshold from 0.49% to 0.45%, 0.37% and 0.23%; at 50% the
  distance-5 CNOT is worse than the distance-3 one at every p (no threshold); heralds cut the
  logical error by one to two parts in twenty and move the threshold by less than its interval.
- **`stabilizer_qec.realtime`: decoding in the background, a decoder queue with statistics,
  and a decoder server** (CUDA-Q QEC's `decode_async`, `enqueue_syndromes`, `get_corrections`,
  `reset_decoder` and decoder statistics; its real-time host API). Every decoder has
  `decode_async(shots)`: the batch runs on a worker thread with the GIL released in the engine
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

### Fixed

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

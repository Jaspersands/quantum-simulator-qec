# Parity inventory for 2.0 (2026-10-09)

What every other QEC tool can do that `stabilizer-qec` 1.9.0 cannot. This is the checklist
2.0 closes (spec: `docs/superpowers/specs/2026-10-09-2.0-parity-design.md`). Each line has an
ID, which the spec's workstreams and the tests refer to.

## How it was gathered

- **Installed reference packages, introspected.** These are Stim 1.16.0, sinter 1.16.0,
  PyMatching 2.4.0, ldpc 2.4.1, relay-bp 0.2.2, tesseract-decoder 0.1.1.dev20260910,
  chromobius 1.1.1 and beliefmatching 0.2.0 (all current on PyPI on this date).
  - Every public class, method, function and parameter was diffed against
    `stabilizer_qec` 1.9.0 (`surface.py` dumps in the session scratchpad).
  - The CLIs were compared: `stim help`, `sinter help/collect/plot`, `pymatching`.
- **Packages not installed: wheels downloaded, never imported or run.** Their public API was
  listed statically, from the source's syntax tree:
  - qldpc 0.4.1, the deltakit 0.10.1 family, bloqade-tsim 0.1.5, mqt.qecc 2.0.0 and
    cudaq-qec;
  - mwpf 0.2.12, fusion-blossom 0.2.13, stimbposd 0.2.0, quits 1.2.0 and qec-util 0.5.0;
  - panqec 0.1.7, qecsim 1.0b9, tqec 0.2.0, quantum-pecos 0.2.0 and qualtran 0.7.0
    (`surface_code`);
  - el-loom 0.4.0 and qsample 0.0.2.
- **Web, for what isn't on PyPI or is newer:**
  - CUDA-Q QEC 0.1–0.8 release notes and API index; cuQuantum cuStabilizer;
  - NVIDIA Ising pre-decoders; Microsoft QDK QEC modules; Azure Resource Estimator;
  - Loom/Entwine; Plaquette; Crumble; qsample's paper; the 2026 decoder-conformance paper
    (arXiv 2609.07035).
- **Not open, so not inventoried:** deltakit's LCD, AC and CC decoders run only on
  Riverlane's cloud (`deltakit_explorer.enums.DecoderType`), and nv-qldpc-decoder and the
  NV Fusion decoder are closed source. They are listed here from their papers and docs, and
  2.0 implements them from the papers.

Status: **have** (1.9.0 has it), **part** (has some of it; the gap is named), **gap** (missing).

---

## S — Stim 1.16

### S.1 Object model (every class is a gap)

| ID | Feature | Status |
|---|---|---|
| S1 | `PauliString` (28 members: products, commutes, after/before through circuits, to/from numpy, unitary, iter_all, random, weight, sign) | done (2.0 W2: Stim's surface, its API equal to Stim 1.16's, member for member) |
| S2 | `PauliStringIterator` | done (2.0 W2: Stim's surface, its API equal to Stim 1.16's, member for member) |
| S3 | `Tableau` (44: from_circuit/conjugated_generators/named_gate/numpy/stabilizers/state_vector/unitary_matrix, to_*, inverse, then, append/prepend, x/y/z_output(_pauli), inverse_*_output, iter_all, random, `__call__` on PauliString) | done (2.0 W2: Stim's surface, its API equal to Stim 1.16's, member for member) |
| S4 | `TableauIterator` | done (2.0 W2: Stim's surface, its API equal to Stim 1.16's, member for member) |
| S5 | `TableauSimulator` (70: every gate method, measure/measure_kickback/measure_many/measure_observable, peek_x/y/z/bloch/observable_expectation, postselect_*, canonical_stabilizers, state_vector, set_state_from_*, do/do_circuit/do_pauli_string/do_tableau, current_inverse_tableau, noise channels) | done (2.0 W2: Stim's surface, its API equal to Stim 1.16's, member for member) |
| S6 | `CliffordString` (single-qubit Clifford arrays: products, powers, x/y/z_outputs, random, all_cliffords_string) | done (2.0 W2: Stim's surface, its API equal to Stim 1.16's, member for member) |
| S7 | `FlipSimulator` (batch frame simulator you drive by hand: do, broadcast_pauli_errors, set_pauli_flip, peek_pauli_flips, get_*_flips, append_measurement_flips, generate_bernoulli_samples, to_numpy) | done (2.0 W2: Stim's surface, its API equal to Stim 1.16's, member for member) |
| S8 | `Flow` and `Circuit.has_flow / has_all_flows / flow_generators / solve_flow_measurements / time_reversed_for_flows` | done (2.0 W2: Stim's surface, its API equal to Stim 1.16's, member for member) |
| S9 | `GateData` / `stim.gate_data()` (aliases, flows, inverse, generalized_inverse, hadamard_conjugated, tableau, unitary_matrix, predicates) | done (2.0 W2: Stim's surface, its API equal to Stim 1.16's, member for member) |
| S10 | `GateTarget` plus `target_rec/inv/x/y/z/pauli/combiner/combined_paulis/sweep_bit` | done (2.0 W2: Stim's surface, its API equal to Stim 1.16's, member for member) |
| S11 | `CircuitInstruction`, `CircuitRepeatBlock` (name, tag, targets_copy, gate_args_copy, target_groups, num_measurements, body_copy, repeat_count) | done (2.0 W2: Stim's surface, its API equal to Stim 1.16's, member for member) |
| S12 | `DemInstruction`, `DemRepeatBlock`, `DemTarget` plus `target_relative_detector_id/logical_observable_id/separator` | done (2.0 W2: Stim's surface, its API equal to Stim 1.16's, member for member) |
| S13 | Fields on `ExplainedError`, `CircuitErrorLocation`, `CircuitErrorLocationStackFrame`, `CircuitTargetsInsideInstruction`, `FlippedMeasurement`, `GateTargetWithCoords`, `DemTargetWithCoords` matching Stim's attribute names (ours are dataclass fields; check all names) | done (2.0 W2: Stim's surface, its API equal to Stim 1.16's, member for member) |

### S.2 `Circuit` methods

| ID | Feature | Status |
|---|---|---|
| S20 | `__getitem__` (index, slice), `__len__`, `insert`, `pop`, `clear`, `append_operation`, `append(CircuitInstruction/RepeatBlock)` | done (2.0 W2: Stim's surface, its API equal to Stim 1.16's, member for member) |
| S21 | `approx_equals` | done (2.0 W2: Stim's surface, its API equal to Stim 1.16's, member for member) |
| S22 | `flattened`, `flattened_operations`, `decomposed`, `inverse`, `without_noise`, `without_tags`, `with_inlined_feedback` | done (2.0 W2: Stim's surface, its API equal to Stim 1.16's, member for member) |
| S23 | `num_ticks`, `get_detector_coordinates`, `get_final_qubit_coordinates`, `count_determined_measurements`, `reference_detector_and_observable_signs`, `missing_detectors` | done (2.0 W2: Stim's surface, its API equal to Stim 1.16's, member for member) |
| S24 | `detecting_regions` | done (2.0 W2: Stim's surface, its API equal to Stim 1.16's, member for member) |
| S25 | `to_tableau`, `to_file`, `to_qasm` (open_qasm_version 2/3, skip_dets_and_obs), `to_quirk_url`, `to_crumble_url` | done (2.0 W2: Stim's surface, its API equal to Stim 1.16's, member for member) |
| S26 | `likeliest_error_sat_problem`, `shortest_error_sat_problem` (WCNF text) | done (2.0 W2: Stim's surface, its API equal to Stim 1.16's, member for member) |
| S27 | `detector_error_model(allow_gauge_detectors=, block_decomposition_from_introducing_remnant_edges=)` | done (2.0 W2: Stim's surface, its API equal to Stim 1.16's, member for member) |
| S28 | `compile_sampler(reference_sample=)`, `compile_m2d_converter(skip_reference_sample=)` | done (2.0 W2: Stim's surface, its API equal to Stim 1.16's, member for member) |
| S29 | `diagram(filter_coords=)` | done (2.0 W2: Stim's surface, its API equal to Stim 1.16's, member for member) |

### S.3 `DetectorErrorModel` methods

| ID | Feature | Status |
|---|---|---|
| S30 | `+`, `+=`, `*`, `*=`, `__getitem__`, `__len__`, `append`, `clear`, `copy` | done (2.0 W2: Stim's surface, its API equal to Stim 1.16's, member for member) |
| S31 | `approx_equals`, `rounded`, `get_detector_coordinates`, `to_file`, `without_tags` | done (2.0 W2: Stim's surface, its API equal to Stim 1.16's, member for member) |

### S.4 Samplers and converters

| ID | Feature | Status |
|---|---|---|
| S40 | `sample_bit_packed` on measurement and detector samplers | done (2.0 W2: Stim's surface, its API equal to Stim 1.16's, member for member) |
| S41 | `sample_write` (to a file in any format) on measurement, detector and DEM samplers | done (2.0 W2: Stim's surface, its API equal to Stim 1.16's, member for member) |
| S42 | Detector `sample(dets_out=, obs_out=, prepend_observables=)` | done (2.0 W2: Stim's surface, its API equal to Stim 1.16's, member for member) |
| S43 | m2d `convert(bit_pack_result=)`, `convert_file` | done (2.0 W2: Stim's surface, its API equal to Stim 1.16's, member for member) |
| S44 | Stim's class names `CompiledMeasurementSampler`, `CompiledDetectorSampler`, `CompiledDemSampler`, `CompiledMeasurementsToDetectionEventsConverter` (aliases) | done (2.0 W2: Stim's surface, its API equal to Stim 1.16's, member for member) |

### S.5 Diagrams

| ID | Feature | Status |
|---|---|---|
| S50 | `timeline-3d`, `matchgraph-3d` (glTF) | done (2.0 W2: 3D glTF, HTML and interactive diagrams) |
| S51 | `interactive` (Crumble embedded) | done (2.0 W2: 3D glTF, HTML and interactive diagrams) |
| S52 | `*-html` variants of every diagram | done (2.0 W2: 3D glTF, HTML and interactive diagrams) |
| S53 | DEM `matchgraph-3d` | done (2.0 W2: 3D glTF, HTML and interactive diagrams) |

### S.6 CLI

| ID | Feature | Status |
|---|---|---|
| S60 | `repl` | done (2.0 W2: Stim's surface, its API equal to Stim 1.16's, member for member) |
| S61 | `help gates`, `help formats`, `help <gate>`, `help <format>` | done (2.0 W2: Stim's surface, its API equal to Stim 1.16's, member for member; 2.1: Stim's own help pages, every instruction, format, command and index, `tools/stim_help.py`) |
| S62 | Every flag of every subcommand matches Stim (re-audit against `stim help <cmd>`) | done (2.0 W2: Stim's surface, its API equal to Stim 1.16's, member for member; 2.1: `diagram --filter_coords`, and a test that every flag Stim's pages document is taken) |

### S.7 Crumble and Quirk

| ID | Feature | Status |
|---|---|---|
| S70 | Crumble-equivalent editor: edit a circuit in 2D and propagate Pauli markers live (`#!pragma MARK_X0`), view and edit detectors and observables, import from and export to Stim text and URLs | done (2.0 W12: site section 17, `js/editor.js` + `js/crumble-format.js` + `js/sections/editor.js` — grid and timeline views, the gate palette, markers propagated by Clifford conjugation to the measurements, detectors and observables they flip, detectors/observables placed on measurements, Stim text and Crumble URLs both ways; the URL equals `to_crumble_url`'s on every example and every marker's flips equal the engine's on ~700 (qubit, layer, Pauli) cases, `tools/site-tests.mjs` in CI. Not Crumble's: polygons, the `#!pragma` text form, undo) |
| S71 | Quirk export | done (2.0 W2: `to_quirk_url`) |

## P — PyMatching 2.4

| ID | Feature | Status |
|---|---|---|
| P1 | Graph construction: `add_edge(node1,node2,fault_ids,weight,error_probability,merge_strategy)`, `add_boundary_edge`, `set_boundary_nodes`, `ensure_num_fault_ids` | done (2.0 W3: PyMatching's graph API and CLI, member for member) |
| P2 | `from_check_matrix(H, weights, error_probabilities, repetitions, timelike_weights, measurement_error_probabilities, faults_matrix, merge_strategy, use_virtual_boundary_node)` and `load_from_check_matrix` | done (2.0 W3: PyMatching's graph API and CLI, member for member) |
| P3 | `from_networkx/to_networkx`, `from_rustworkx`/`load_from_rustworkx`/`to_rustworkx` (and retworkx) | done (2.0 W3: PyMatching's graph API and CLI, member for member) |
| P4 | `from_stim_circuit(_file)`, `from_detector_error_model_file` | done (2.0 W3: PyMatching's graph API and CLI, member for member) |
| P5 | Queries: `edges()`, `get_edge_data`, `get_boundary_edge_data`, `has_edge`, `has_boundary_edge`, `num_nodes/num_edges/num_fault_ids/boundary` | done (2.0 W3: PyMatching's graph API and CLI, member for member) |
| P6 | `decode(return_weight=)`, `decode_to_edges_array`, `decode_to_matched_dets_array`, `decode_to_matched_dets_dict` | done (2.0 W3: PyMatching's graph API and CLI, member for member; 2.2: decoded on PyMatching 2.4.0's sparse blossom transliterated (`src/pm_exact/`), so among equally light matchings the one listed, and its order, are PyMatching's: edges, matched pairs, predictions and weights equal on 950,000 shots of 19 models and 600,000 of 3,000 hand-built graphs, and so for everything built on `Matching`. Not ported: PyMatching's choice among equally light matchings outside that: correlated matching (`enable_correlations=True`), `BeliefMatching`, `WindowMatching`, `stream_memory`, `LeakageAwareDecoder`, `ArgumentReweighting` and `PreDecoder` (unless `reweight=False`) with matching, and the Rust crate's `Matching` decode on this package's own sparse blossom (`src/sparse/`), whose choice can differ) |
| P7 | `add_noise` (sample errors on the graph's own probabilities) | done (2.0 W3: PyMatching's graph API and CLI, member for member) |
| P8 | `draw` | done (2.0 W3: PyMatching's graph API and CLI, member for member) |
| P9 | Constructor `Matching(H or graph, weights, …)` | done (2.0 W3: PyMatching's graph API and CLI, member for member) |
| P10 | `pymatching predict` and `pymatching count_mistakes` CLI (we have `decode`, after predict) | done (2.0 W3: PyMatching's graph API and CLI, member for member; 2.1: `predict` reads b8 by default, as PyMatching's does; 2.2: `animate` too (`stabilizer-qec animate`, `python -m stabilizer_qec.pymatching animate`, `pymatching.cli`), with PyMatching's flags, messages and exit codes: all 1,080 frames of 11 cases recorded from PyMatching 2.4.0's macOS arm64 wheel equal as sorted lines, and its 44 recorded misuses' messages) |
| P11 | `set_seed`, `randomize`, `rand_float` (tie-breaking randomness) | done (2.0 W3: PyMatching's graph API and CLI, member for member) |

## N — sinter 1.16

| ID | Feature | Status |
|---|---|---|
| N1 | `Task`, `TaskStats`, `AnonTaskStats`, `CollectionOptions`, `Progress`, `Fit` (strong_id, CSV lines, with_edits, `+`) | done (2.0 W4: `stabilizer_qec.sinter`, sinter whole, with its CLI) |
| N2 | `collect` / `iter_collect` (processes, max_shots/max_errors, batch sizing, save/resume, existing data, progress callback, hint_num_tasks, count_observable_error_combos, count_detection_events, custom_error_count_key, allowed_cpu_affinity_ids) | done (2.0 W4: `stabilizer_qec.sinter`, sinter whole, with its CLI) |
| N3 | Post-selection: by detector 4th coordinate, detector and observable predicates, discards | done (2.0 W4: `stabilizer_qec.sinter`, sinter whole, with its CLI) |
| N4 | `read_stats_from_csv_files`, `stats_from_csv_files`, CSV output format | done (2.0 W4: `stabilizer_qec.sinter`, sinter whole, with its CLI) |
| N5 | `fit_binomial`, `fit_line_slope`, `fit_line_y_at_x`, `log_binomial`, `log_factorial`, `shot_error_rate_to_piece_error_rate` | done (2.0 W4: `stabilizer_qec.sinter`, sinter whole, with its CLI) |
| N6 | `plot_error_rate`, `plot_discard_rate`, `plot_custom`, `group_by`, `better_sorted_str_terms`, `comma_separated_key_values` | done (2.0 W4: `stabilizer_qec.sinter`, sinter whole, with its CLI) |
| N7 | `predict_observables(_bit_packed)`, `predict_discards_bit_packed`, `predict_on_disk`, `post_selection_mask_from_4th_coord` | done (2.0 W4: `stabilizer_qec.sinter`, sinter whole, with its CLI) |
| N8 | `Sampler`/`CompiledSampler` and `Decoder`/`CompiledDecoder` plug-in protocols | done (2.0 W4: `stabilizer_qec.sinter`, sinter whole, with its CLI; 2.1: the built-in `mw_parity_factor` and `hypergraph_union_find` run on this package's MWPF) |
| N9 | CLI `collect`, `combine`, `plot` with all flags | done (2.0 W4: `stabilizer_qec.sinter`, sinter whole, with its CLI) |

## L — ldpc 2.4

| ID | Feature | Status |
|---|---|---|
| L1 | BP schedules: `parallel`, `serial`, `serial_relative`, random serial (`random_serial_schedule`, `random_schedule_seed`), `serial_schedule_order` | done (2.0 W5: `stabilizer_qec.ldpc`, ldpc 2 whole, every module) |
| L2 | `input_vector_type="received_vector"` (decode a codeword, not a syndrome) | done (2.0 W5: `stabilizer_qec.ldpc`, ldpc 2 whole, every module); 2.1: on BP+OSD, BP+LSD and BeliefFind too, decoding, iterations and convergence equal to ldpc's (the audit's 44 differing shots of 900 are 0) |
| L3 | Attributes: `log_prob_ratios`, `converge`, `iter`, `channel_probs`, `update_channel_probs`, `decoding`, `osd0_decoding`, `osdw_decoding`, `bp_decoding`, `omp_thread_count` | done (2.0 W5: `stabilizer_qec.ldpc`, ldpc 2 whole, every module) |
| L4 | `BeliefFindDecoder` (BP then union-find, `uf_method` peeling/inversion, `bits_per_step`) | done (2.0 W5: `stabilizer_qec.ldpc`, ldpc 2 whole, every module); 2.1: ldpc keeps `uf_method` in a C bool, so its "peeling" is inversion and only `""` peels; ours runs the name given (documented), equal to ldpc's matching algorithm shot for shot |
| L5 | `SoftInfoBpDecoder` (soft syndromes: cutoff, sigma) | done (2.0 W5: `stabilizer_qec.ldpc`, ldpc 2 whole, every module) |
| L6 | `BpFlipDecoder` (BP plus bit flipping) | done (2.0 W5: `stabilizer_qec.ldpc`, ldpc 2 whole, every module) |
| L7 | `mbp_decoder` (memory BP) | done (2.0 W5: `stabilizer_qec.ldpc`, ldpc 2 whole, every module) |
| L8 | `UnionFindDecoder` on any parity-check matrix (peeling and inversion) | done (2.0 W5: `stabilizer_qec.ldpc`, ldpc 2 whole, every module); 2.1: with LLRs too: ours' inversion equals ldpc's `uf_method="peeling"` (inversion) on every shot; ours' peeling equals ldpc's `uf_method=""` but where `last_stuck` flags a shot on which ldpc 2.4.1's peeling hangs, crashes or leaves the syndrome unexplained. ldpc's default (`False`) is a TypeError there; here it peels |
| L9 | `LsdDecoder` (standalone, given bit weights) | done (2.0 W5: `stabilizer_qec.ldpc`, ldpc 2 whole, every module); 2.1: `lsd_method` and `lsd_order` settable on LsdDecoder and BpLsdDecoder, and decoding with them as ldpc's does (OSD_OFF reachable through the setter, as in ldpc); the audit's 95 differing shots of 300 are 0 |
| L10 | Overlapping-window decoders: BP+OSD, BP+LSD and PyMatching over windows of a DEM, with sinter wrappers (`ckt_noise`) | done (2.0 W5: `stabilizer_qec.ldpc`, ldpc 2 whole, every module); 2.1: BP+OSD windows at OSD-CS 3 equal to ldpc's on every shot (3 circuits, 4,000 shots each): OSD sums a candidate's ln(1/p) in bit order and sorts columns with the platform's C qsort (Apple's now exact, its bounded insertion sort and heapsort fallback included), as ldpc does. The BP+LSD windows are still held to more than 97% of shots equal to ldpc's (`tests/test_ldpc_ckt_noise.py`), since ldpc's answer can vary with the order it merges three or more clusters in; 2.2's search of those orders is under U2 |
| L11 | `mod2`: rank, kernel, nullspace, row_echelon, reduced_row_echelon, row_basis, row_complement_basis, row_span, inverse, pivot_rows, PluDecomposition.lu_solve, estimate/compute_exact_code_distance | done (2.0 W5: `stabilizer_qec.ldpc`, ldpc 2 whole, every module) |
| L12 | `code_util`: code parameters, dimension, generator matrix, distance (exact and estimated), average Hamming weights, cycle search (girth) | done (2.0 W5: `stabilizer_qec.ldpc`, ldpc 2 whole, every module) |
| L13 | `codes`: rep_code, ring_code, hamming_code, random_binary_code; `alist` I/O; protograph RingOfCirculantsF2 | done (2.0 W5: `stabilizer_qec.ldpc`, ldpc 2 whole, every module) |
| L14 | Monte Carlo: BSC simulation, single-shot and quasi-single-shot memory with analog syndromes | done (2.0 W5: `stabilizer_qec.ldpc`, ldpc 2 whole, every module) |
| L15 | sinter wrappers: SinterBpOsdDecoder, SinterLsdDecoder, SinterBeliefFindDecoder | done (2.0 W5: `stabilizer_qec.ldpc`, ldpc 2 whole, every module) |

## R — Relay-BP 0.2.2, Tesseract, Chromobius, belief-matching, stimbposd

| ID | Feature | Status |
|---|---|---|
| R1 | `MinSumBPDecoder` in every precision: F32, F64, Fixed, I8, I16, I32, I64; Relay in F32/F64/I32/I64 | done (2.0 W5: `stabilizer_qec.relay_bp`, every precision) |
| R2 | `decode_detailed(_batch)` → iterations, success, posterior ratios | done (2.0 W5: `stabilizer_qec.relay_bp`, every precision) |
| R3 | `ObservableDecoderRunner` (observables directly, detailed results: converged, error_detected, error_mismatch_detected, better/worse quality, from_errors_* helpers) | done (2.0 W5: `stabilizer_qec.relay_bp`, every precision; 2.1: `parallel=True` decodes on threads without the GIL, one decoder state per thread, where 2.0's was serial; the runner writes the decoder's `relay_logging.out`; check matrices with stored zeros read as relay_bp reads them) |
| R4 | Tesseract `decode_to_errors`, `cost_from_errors`, `get_observables_from_errors`, `low_confidence_flag`, visualizer | done (2.0 W5: `stabilizer_qec.tesseract_decoder`, Tesseract whole, `demutil` too. 2.1: the two-pass decoder on annotated models, repeated detection events, merged zero-probability faults and negative beams decode as Tesseract's; the decoder's config is read at every decode as Tesseract's is (not its sparsify settings); Stim's models in give Stim's out; pybind11's enums, overloads and exception types. Difference: a bad detector order raises at the next decode where Tesseract reads past it. 2.2: any number of observables, as Tesseract's (the Rust API's `SearchDecoder` stays at 64, its `Prediction.observables` a `u64`), a 100-observable model recorded from Tesseract's wheel decoding as there; Tesseract's `tesseract` program (`python -m stabilizer_qec.tesseract_decoder`), its flags, argparse errors and help, output lines, `--dem-out` and `--stats-out`, checked against Tesseract's Python API run as its program runs it; an empty answer's cost 0, not −0) |
| R5 | Tesseract `SimplexDecoder` (exact LP/ILP decoding with windows) | done (2.0 W5: `stabilizer_qec.tesseract_decoder`; each program solved by an exact search, not HiGHS: the same cost without windows, the errors can differ among equally likely ones; with windows a different pick can make the total cost higher or lower; `verbose` prints window bounds, not HiGHS's log) |
| R6 | Tesseract utilities `dem_from_counts`, `merge_indistinguishable_errors`, `remove_zero_probability_errors`, `build_det_orders` (BFS, coordinate, index), `build_detector_graph`, sparsify options | done (2.0 W5: `stabilizer_qec.tesseract_decoder`, Tesseract whole, `demutil` too; 2.1: each gives back the kind of model it was given, Stim's or this package's; 2.2: `build_det_orders` and `build_detector_graph` take models of any number of observables, as Tesseract's do) |
| R7 | Chromobius `predict_weighted_obs_flips_from_dets_bit_packed` (weights out) | done (2.0 W5: `stabilizer_qec.chromobius`; weights bit for bit. 2.1: faults of probability above 1/2 as Chromobius's negative-weight edges (probability 1 refused); the sinter decoder's integer data and `decode_via_files` widths, short files and partial records as Chromobius's. Chromobius itself reads records wider than the model inconsistently, naming different detectors from run to run or decoding them, and off macOS can decode a shot with a padding bit set; this package refuses both. 2.2: the Möbius matching is PyMatching 2.2.2's sparse blossom, the one Chromobius 1.1.1 links (`src/pm_exact/` in 2.2.2's neighbour order), so the predictions are Chromobius's: on macOS on every shot of a 65,000-shot sweep and of `tools/decoder_check.py`'s 340,000 colour shots; off macOS Chromobius walks some tables in the C++ library's `std::unordered_map` order, and the tests hold the predictions to 99%. Models refused as Chromobius refuses them, with its messages. `python -m stabilizer_qec.chromobius` is Chromobius's command line, checked against 184 cases recorded from its binary) |
| R8 | BeliefMatching `from_stim_circuit` | done (2.0 W5: `stabilizer_qec.beliefmatching`) |
| R9 | stimbposd BPOSD/BPLSD on a DEM | done (2.1: `stabilizer_qec.stimbposd`, stimbposd 0.2.0 whole: `BPOSD`, `BPLSD`, its sinter decoders and `sinter_decoders()`' eight variants, `detector_error_model_to_check_matrices`; 2.0 had marked it done by `stabilizer_qec.bposd`, which is quantumgizmos' bposd 2.1, another package; the native `BpOsd`/`BpLsd` take its `schedule=` and `random_schedule_seed=` too) |

## D — Deltakit 0.10 (Riverlane)

| ID | Feature | Status |
|---|---|---|
| D1 | Collision Clustering decoder (CC, arXiv 2309.05558) | done (2.0 W6: from the paper, in Rust, `src/clustering.rs`) |
| D2 | Local Clustering Decoder (LCD, arXiv 2411.10343): adaptive, leakage-aware | done (2.0 W6: from the paper, in Rust, `src/clustering.rs`) |
| D3 | Ambiguity Clustering (BP-AC, arXiv 2406.14527) | done (2.0 W6: from the paper, in Rust, `src/ambiguity.rs`) |
| D4 | Leakage flags in data and leakage-aware decoding (`LeakageFlags`, `HERALD_LEAKAGE_EVENT`) | done (2.0 W6: dialect, sampler, `without_leakage`, `LeakageAwareDecoder`) |
| D5 | Noise models: SD6, SI1000, phenomenological, toy, `PhysicalNoise` (T1/T2-derived), noise profiles per gate, `Leakage` and `Relax` channels | done (2.0 W7: `stabilizer_qec.deltakit.explorer.qpu`, Deltakit's own, its circuits and DEMs equal to Deltakit's) |
| D6 | QPU model: native gate sets with times, `compile_circuit_to_native_gates` (CZ / CZSWAP), parallelise circuits, merge layers, remove identities | done (2.0 W7: `stabilizer_qec.deltakit.explorer.qpu`, Deltakit's own, its compiled circuits equal to Deltakit's) |
| D7 | Codes: rotated/unrotated planar, unrotated toric, repetition, BB, generic `CSSCode`, `StabiliserCode`, schedules (orders, types) | done (2.0 W7: `stabilizer_qec.deltakit.explorer.codes`, Deltakit's own, every code's memory circuits equal to Deltakit's) |
| D8 | Stability experiments (`css_code_stability_circuit`, `QECExperimentType.STABILITY`) | done (2.0 W7: as in Deltakit, through its cloud client, `stabilizer_qec.deltakit.Client`; Deltakit builds none locally either) |
| D9 | Correlation matrix p_ij from data, `create_dem_from_pij`, DEM vs p_ij comparison | done (2.0 W4: `stabilizer_qec.analysis`, deltakit's analysis) |
| D10 | Λ fits (`calculate_lambda_*`, asymmetric), logical error per round fits, `simulate_different_round_numbers`, `predict_quops_*`, `predict_distance_for_quops` | done (2.0 W4: `stabilizer_qec.analysis`, deltakit's analysis) |
| D11 | Error budget (`get_error_budget`: sensitivity of Λ to each noise parameter, plot) | done (2.0 W4: `stabilizer_qec.analysis`, deltakit's analysis) |
| D12 | Plots: defect diagram, defect rates, detection probability on patch, Λ, LEPPR, correlation matrix | done (2.0 W4: `stabilizer_qec.analysis`, deltakit's analysis) |
| D13 | Decoding graphs: hypergraph objects, graph distance, logicals extraction, window ids, JSON I/O | done (2.0 W7: `stabilizer_qec.deltakit.core.decoding_graphs`, Deltakit's own) |
| D14 | Data formats: b8/01/csv for syndromes, logical flips, measurements; c64 | done (2.0 W7: `stabilizer_qec.deltakit.core.data_formats`, Deltakit's own) |
| D15 | Decoder managers and run_decoding_on_circuit; empirical decoding error distribution | done (2.0 W4: `stabilizer_qec.analysis`, deltakit's analysis) |
| D16 | Noise sources for code-capacity matching: exhaustive, fixed weight, uniform erasure | done (2.0 W7: `stabilizer_qec.deltakit.decode.noise_sources`, Deltakit's own) |

deltakit-compile 0.1.0 and deltakit-visualise 0.4.0 (its xDSL compiler and visualiser) are W10's.

## Q — qLDPC 0.4.1 (Infleqtion)

| ID | Feature | Status |
|---|---|---|
| Q1 | Classical codes: repetition, ring, cyclic, Golay, Hamming, extended Hamming, Reed–Muller, Reed–Solomon, BCH, simplex, Tanner | done (2.0 W6: `stabilizer_qec.qldpc`, qLDPC 0.4.1 whole; W7 cross-checks) |
| Q2 | Quantum codes: five-qubit, Steane, quantum Hamming, quantum Reed–Muller, tetrahedral, quantum Golay, iceberg, C4, C6, many-hypercube, TB, GALA, QC, BB (general, toric layouts), HGP, CHGP, CRC, SHP, Bacon–Shor, SHYPS, LP, SLP, QT, surface, toric, generalized surface, T4 | done (2.0 W6: `stabilizer_qec.qldpc`, qLDPC 0.4.1 whole; W7 cross-checks) |
| Q3 | Qudit codes over GF(q) (`QuditCode`, `QuditPauli`, fields) | done (2.0 W6: `stabilizer_qec.qldpc`, qLDPC 0.4.1 whole; W7 cross-checks) |
| Q4 | Non-CSS stabilizer codes (symplectic), subsystem codes, conversion to CSS/SWEL | done (2.0 W6: `stabilizer_qec.qldpc`, qLDPC 0.4.1 whole; W7 cross-checks) |
| Q5 | Logical operators (canonical basis, symplectic Gram–Schmidt), dual basis | done (2.0 W6: `stabilizer_qec.qldpc`, qLDPC 0.4.1 whole; W7 cross-checks) |
| Q6 | Distance: exact (Brouwer–Zimmermann for CSS; brute force), bounds by random search, upper bounds | done (2.0 W6: `stabilizer_qec.qldpc`, qLDPC 0.4.1 whole; W7 cross-checks) |
| Q7 | Code equivalence, canonical forms, `forget_distance`, code parameters | done (2.0 W6: `stabilizer_qec.qldpc`, qLDPC 0.4.1 whole; W7 cross-checks) |
| Q8 | Groups and rings: cyclic, dihedral, alternating, symmetric, quaternion, small groups, SL/PSL/GL/PGL, group rings, lifts, Wedderburn–Artin | done (2.0 W6: `stabilizer_qec.qldpc`, qLDPC 0.4.1 whole; W7 cross-checks) |
| Q9 | Encoding circuits and tableaus, logical tableaus, encoder/decoder pairs | done (2.0 W6: `stabilizer_qec.qldpc`, qLDPC 0.4.1 whole; W7 cross-checks) |
| Q10 | Transversal gates: transversal S, transversal ops, automorphism group, transversal circuits for given logical Cliffords | done (2.0 W6: `stabilizer_qec.qldpc`, qLDPC 0.4.1 whole; W7 cross-checks) |
| Q11 | Memory experiments for any code: syndrome-measurement strategies (edge colouring, EdgeColoringXZ), AlphaSyndrome (MCTS schedule search), Bell-state logical prep, observables | done (2.0 W6: `stabilizer_qec.qldpc`, qLDPC 0.4.1 whole; W7 cross-checks) |
| Q12 | State-prep diagnostics (flags, post-selection, discard rate) | done (2.0 W6: `stabilizer_qec.qldpc`, qLDPC 0.4.1 whole; W7 cross-checks) |
| Q13 | Noise models: `NoiseModel` rules, depolarizing, SI1000, Pauli channels, `as_noiseless_circuit` | done (2.0 W6: `stabilizer_qec.qldpc`, qLDPC 0.4.1 whole; W7 cross-checks) |
| Q14 | Decoders: lookup (plain, weighted, observable), ILP, GUF, composite, direct, frontier, subgraph, sequential and sliding windows, trivial; erasure flags | done (2.0 W6: `stabilizer_qec.qldpc`) |
| Q15 | `DetectorErrorModelArrays`: simplified, without_detectors, post_selected_on, with_decomposed_errors, to_circuit | done (2.0 W6: `stabilizer_qec.qldpc`, qLDPC 0.4.1 whole; W7 cross-checks) |
| Q16 | Code-capacity DEMs and decoders; Monte Carlo with weight-truncated error-rate functions (`ErrorRateFunc`, Jeffreys variance, sample allocation by weight) | done (2.0 W6: `stabilizer_qec.qldpc`, qLDPC 0.4.1 whole; W7 cross-checks) |
| Q17 | Experimental surgery: bridges between codes, Cheeger constant, gadget boosting, single and joint Pauli-product measurement circuits | done (2.0 W6: `stabilizer_qec.qldpc`, qLDPC 0.4.1 whole; W7 cross-checks) |
| Q18 | External: qecdb / QLDPC-challenge code fetch, GAP/Magma groups, sqetch distance bounds | done (2.0 W6: `stabilizer_qec.qldpc`, qLDPC 0.4.1 whole; W7 cross-checks) |

## T — Tsim (QuEra, bloqade-tsim 0.1.5)

| ID | Feature | Status |
|---|---|---|
| T1 | Clifford+T sampling by stabilizer rank (ZX), cost exponential in non-Clifford count, 80+ qubits | done (2.0 W9: `Circuit.compile_stabilizer_rank_sampler`, Rust `StabilizerRankSampler`; Pauli-coset terms on a shot-independent reference tableau, exact; `compile_sampler`/`compile_detector_sampler` run it on a circuit with non-Clifford gates; equal to the state vector bit for bit, within error bars of Tsim on its demos; 2.1: `tsim.Circuit`'s samplers are Tsim's `CompiledMeasurementSampler`/`CompiledDetectorSampler`, the detector sampler reporting each detector's and observable's value as Tsim's does, `use_detector_reference_sample`/`use_observable_reference_sample` the flips; `batch_size` and `postselection_mask` checked as Tsim's, `strategy` accepted; a `stabilizer_qec.Circuit` keeps Stim's flips. Not ported: Tsim's shots bit for bit (its noise is drawn with numpy's PCG64 and its outcomes with JAX's threefry, and Tsim is reproducible only at a fixed `batch_size`); the shots here are this package's, equal in distribution) |
| T2 | Gates T, T_DAG, R_X/Y/Z, U3, R_PAULI, TPP, R_XX/YY/ZZ (plain gates, not only tags) | done (2.0 W9: Tsim's shorthand and tagged text through `stabilizer_qec.tsim.Circuit`, every gate's matrix Tsim's up to a global phase; 2.1: `tsim.Circuit` is a class, a `Circuit` subclass whose results (slices, `+`, `copy`, `inverse`, ...) are `tsim.Circuit`s, `str`/`repr` Tsim's shorthand, `append` of Tsim's gates in half-turns, `append_from_stim_program_text`, `from_file`, `+` with Stim's and this package's circuits; `stim_to_shorthand` reads this package's tagged form too) |
| T3 | CCZ, CCX | done (2.0 W9: Tsim's Clifford+T decomposition through the shorthand, 7 T gates and 8 terms in the sampler) |
| T4 | `CompiledStateProbs.probability_of` (exact output probabilities) | done (2.0 W9: `StabilizerRankSampler.probability_of`, exact without noise, an average over drawn noise with it; 2.1: `tsim.CompiledStateProbs(circuit, sample_detectors=...)`, Tsim's: one exact P(state given the draw) per noise draw, of a record or of the detectors' and observables' values (summed over their records, up to 2^16), equal to Tsim's without noise) |
| T5 | `to_matrix`, `to_tensor`, `tcount`, `is_clifford`, `cast_to_stim`, `inverse` with non-Clifford gates | done (2.0 W9: on `Circuit`; 2.1: `tsim.Circuit.to_matrix` is Tsim's: unused qubits dropped, a qubit reset first a state input (a state column of up to 22 qubits), noise skipped; `cast_to_stim` returns the circuit, `stim_circuit` a plain `Circuit`; 2.2: `to_matrix` and `to_tensor` of a circuit with measurements, detectors, observables, feedback, a reset after a qubit's first operation or noise before a reset are Tsim's ZX tensor, where 2.1 raised (measurements, feedback, later resets) or gave the wrong matrix (the wrong shape for noise before a reset and, without measurements, a factor of 2 too few for each target-less detector and each observable); `tcount` is the graph's, as Tsim's (a coherent error is not a gate); `R_PAULI(±0.25)` and the pair rotations at ±0.25 are Tsim's `R_PAULI`, angles kept as written, and `inverse` Tsim's) |
| T6 | DEM of a non-Clifford circuit (Clifford part, non-deterministic observables allowed) | done (2.0 W9: `detector_error_model(allow_non_deterministic_observables=True)`, Tsim's conversion ported; 2.1: `tsim.Circuit.detector_error_model` is Tsim's model, Stim's analysis of its tagged form — T read as S, a rotation as I — equal to Tsim's text) |
| T7 | Encoders: TransversalEncoder, Steane encoder, colour-code encoder | done (2.0 W7: `stabilizer_qec.tsim`, on `Circuit`, verified by `Tableau`; 2.1: the encoder's circuit is a `tsim.Circuit`, its text and model Tsim's) |
| T8 | GPU (JAX, CUDA) | done (2.0 W11: frame sampling and min-sum BP on any GPU wgpu drives — Metal, Vulkan, DX12 — through `device="gpu"`, see C10; no CUDA or JAX dependency) |
| T9 | Diagrams of non-Clifford circuits (timeline SVG with `height`/`width`/`zoomable`; pyzx ZX graphs) | done (2.1: the timeline SVG and text, and the time-slice SVG, label T, T_DAG, rotations, U3 and TPP by their half-turns; an untagged circuit's drawing is unchanged (the text Stim's byte for byte); `tsim.Circuit.diagram` gives Tsim's HTML wrapper and sizes; 2.2: `diagram("pyzx")`, `"pyzx-dets"` and `"pyzx-meas"` draw Tsim 0.1.5's ZX graphs, which `get_graph` and `get_sampling_graph` build vertex for vertex and edge for edge (`stabilizer_qec.tsim.zx`; pyzx_param's `GraphS` where it imports, this package's numpy `ZXGraph` otherwise), checked on 40 recorded circuits on both backends and against live Tsim on 4,200 random circuits) |

## C — CUDA-Q QEC 0.8, cuStabilizer

| ID | Feature | Status |
|---|---|---|
| C1 | Tensor-network decoder (exact maximum likelihood by contraction) | done (2.0 W6: `stabilizer_qec.cudaq_qec`) |
| C2 | Single-error and multi-error lookup-table decoders | done (2.0 W6: `stabilizer_qec.cudaq_qec`) |
| C3 | Sliding-window decoder around any inner decoder (boundary-aware) | done (2.0 W6: `stabilizer_qec.cudaq_qec`) |
| C4 | Relay-BP gamma ensembles, `relay_solutions` post-processing; BP+OSD `osd_init_method="min_llr"`; sum-product BP variants | partly done (2.0 W5: Relay-BP and BP variants on the ldpc/relay_bp ports. 2.1: `cudaq_qec.relay_solutions` (CUDA-Q's `unpack`, `stop_nconv_sweep`) works on CUDA-Q's recorded payloads, but this package's `nv-qldpc-decoder` does not record `relay_solutions` (its relay engine keeps no per-convergence records: needs Rust) and refuses gamma ensembles (`gamma_ensemble_size`), `bp_method` 4/5, `clip_value`, `iter_per_check` and `bp_llr_history`) |
| C5 | AI pre-decoder pipeline: neural network clears local errors, residual goes to matching (NVIDIA Ising CNN, arXiv 2604.12841) | done (2.0 W6: `PreDecoder`, Rust and numpy training, shipped weights, matching on the network's weights) |
| C6 | Async and streaming API: `decode_async`, `enqueue_syndromes`, `get_corrections`, `reset_decoder`, decoder stats (latency) | done (2.0 W11: `stabilizer_qec.realtime` — `decode_async` on every decoder, `DecoderQueue` with `enqueue_syndromes`, `get_corrections` by ticket or in order, `reset`, `DecoderStats`; `cudaq_qec.configure_decoders({id: decoder})`/`enqueue_syndromes(id, batch)`/`get_corrections(id, ticket)` over it. 2.1: CUDA-Q's real-time host library itself — `decoder_config`, `multi_decoder_config` and their YAML (zero-dependency reader, CUDA-Q's LLVM format), `configure_decoders`/`_from_file`/`_from_str` with CUDA-Q's status codes, `finalize_decoders`, `enqueue_syndromes(id, syndromes, tag)` mapping measurements to detectors through `D_sparse`, `get_corrections(id, return_size, reset)` accumulating observable flips until reset, the sliding window a layer a round, `stim_dem_path`; the parameter schemas and JSON Schema export; the deprecated typed configs. 2.2: `dem_chunks` and `expand_dem_chunks`, on CUDA-Q QEC 0.8.1's extended detector error models (every name of `py_extended_dem.cpp`, `dem_from_css_matrices`; its 101 extended-model tests and 70 DEM and YAML tests ported; the closed chunks equal, mechanism for mechanism, the model derived from the same circuit), and `trt_decoder`, its ONNX model run on the CPU in numpy, equal to onnxruntime 1.31.0 bit for bit on 157 recorded cases of 17 models (TensorRT's `engine_load_path` and `engine_save_path` refused by name). Not ported: `dispatch`/`transport` (kept and written, ignored on the host as in CUDA-Q's host library)) |
| C7 | Decoder server (real-time, RPC) | done (2.0 W11: `stabilizer-qec serve --dem/--circuit --decoder NAME --tcp host:port | --stdio [--workers N]`, a framed protocol (`realtime.Protocol`), `DecoderClient.connect`/`spawn` with `decode_batch`, `decode`, `stats`, `reset`; over TCP and over a subprocess's pipes the predictions equal the local decoder's) |
| C8 | PCM utilities: `get_pcm_for_rounds`, `pcm_extend_to_n_rounds`, sorting/simplifying/shuffling columns, `generate_random_pcm`, timelike sparse detector matrix | done (2.0 W6: `stabilizer_qec.cudaq_qec`; 2.1: `generate_timelike_sparse_detector_matrix`, `pcm_from_sparse_vec`, `pcm_to_sparse_vec`, `patch`) |
| C9 | `sample_code_capacity`, `sample_memory_circuit` (x/z), `dem_from_memory_circuit` (x/z), `dem_sampling` | done (2.0 W6: `stabilizer_qec.cudaq_qec`; 2.1: `dem_sampling`, CUDA-Q's CPU draws bit for bit (`std::mt19937`); kernels registered for a Python `@code` are not run and raise `KernelNotRunWarning`) |
| C10 | GPU frame sampling (cuStabilizer: X/Z/leakage tables, up to 150× one CPU thread) and GPU decoders | done (2.0 W11: `compile_detector_sampler(device="gpu")` and `compile_sampler(device="gpu")` run the lane program (`src/lanes.rs`: 32 shots a word, integer draws) on wgpu (`src/gpu/sample.wgsl`; Metal, Vulkan, DX12; Cargo feature `gpu`), bit for bit the CPU lane sampler's bytes (`device="cpu-lanes"`); `BpDecoder(device="gpu")` runs f32 min-sum BP (`src/gpu/bp.wgsl`) equal to `device="cpu-f32"` (`src/bp32.rs`) bit for bit; `stabilizer_qec.gpu_device()` names the device; `tools/gpu_bench.py` → `data/gpu/bench.json`. Leakage runs on the CPU samplers only) |
| C11 | Surface-code orientations XV, XH, ZV, ZH | done (2.0 W7: `cudaq_qec.surface_code`/`stabilizer_grid` as CUDA-Q's, `memory_circuit(orientation=)`) |
| C12 | CUDA-Q QEC's GPU decoder plugins under their names: `nv-qldpc-decoder` (BP, BP+OSD, Relay-BP; also as a sliding-window inner decoder), `chromobius`, `nv-fusion-decoder` | done (2.1: on this package's ldpc/relay_bp, Chromobius and fusion_blossom ports, checked against ldpc, chromobius and fusion-blossom; settings the ports lack are refused by name, not ignored; 2.2: `chromobius` decodes on the Chromobius port's PyMatching 2.2.2 matcher, its predictions Chromobius's on macOS) |

## M — MQT QECC 2.0

| ID | Feature | Status |
|---|---|---|
| M1 | Encoding-circuit synthesis: Gottesman, heuristic, gate-optimal and depth-optimal (SAT), for CSS and non-CSS | done (2.0 W6: `stabilizer_qec.mqt_qecc`, MQT QECC 2.0.0 whole; W7 cross-checks) |
| M2 | Fault-tolerant state preparation: verification circuits (gate-optimal), deterministic verification with corrections, flags, hooks | done (2.0 W6: `stabilizer_qec.mqt_qecc`, MQT QECC 2.0.0 whole; W7 cross-checks; 2.1: `NoisyDFTStatePrepSimulator` on `stabilizer_qec.qsample`, with two faults of MQT's fixed (CNOTs as qsample's (control, target), and the final decoding measuring each stabilizer, where MQT's corrected nothing and gave slope 1), MQT's skipped simulation tests run and pass (slope ≈ 2), and verifications of the wrong Pauli type refused) |
| M3 | Cat-state preparation (balanced tree, line, pruned, fused) and experiments | done (2.0 W6: `stabilizer_qec.mqt_qecc`, MQT QECC 2.0.0 whole; W7 cross-checks) |
| M4 | Clifford synthesis (exact gate count and depth, SAT), CNOT-circuit synthesis (greedy and rollout) | done (2.0 W6: `stabilizer_qec.mqt_qecc`, MQT QECC 2.0.0 whole; W7 cross-checks; 2.1: Stim's own circuits and models taken too, read from their text, where it took only this package's) |
| M5 | Code switching compiler (minimal switching, min cut) | done (2.0 W6: `stabilizer_qec.mqt_qecc.code_switching`) |
| M6 | Colour-code MaxSAT decoder (LightsOut) | done (2.0 W6: `stabilizer_qec.mqt_qecc`, MQT QECC whole; 2.1: its `mqt.qecc.cc-decoder` script as `stabilizer-qec mqt.qecc.cc-decoder` and `python -m stabilizer_qec.mqt_qecc.cc_decoder`, no console script of that name; its `detector_error_model_to_check_matrices` takes Stim's own models too) |
| M7 | Analog Tanner-graph decoding (soft syndromes), single-shot and quasi-single-shot simulators | done (2.0 W6: `stabilizer_qec.mqt_qecc`, MQT QECC whole) |
| M8 | Codes: concatenated (CSS), quantum Hamming, iceberg, many-hypercube, 3D/4D (sparse) hypergraph product, hexagonal and square-octagon colour codes, rotated surface | done (2.0 W6: `stabilizer_qec.mqt_qecc`, MQT QECC 2.0.0 whole; W7 cross-checks) |
| M9 | Stabilizer tableau objects (`StabilizerTableau`, `Pauli`, symplectic vectors and matrices), `CliffordIsometry` | done (= S3/S5, 2.0 W2) |
| M10 | Routing for lattice surgery (cococo: hill climbing, snake builders, teleportation router, Steiner trees) | done (2.0 W6: `stabilizer_qec.mqt_qecc.cococo`) |
| M11 | Fault sets, coset leaders, undetectable-fault search on CNOT circuits | done (2.0 W6: `stabilizer_qec.mqt_qecc`, MQT QECC 2.0.0 whole; W7 cross-checks) |

## U — QUITS, qec-util, PanQEC, qecsim, PECOS, Fusion Blossom, MWPF, qsample

| ID | Feature | Status |
|---|---|---|
| U1 | QUITS codes: BB, BPC, HGP, LCS, Mitten, QLP (and polynomial QLP); circuit builders (cardinal, cardinal NS-merge, ZX colouration, custom); layouts (toric, BB toric, transversal); stabilizer-flow schedule validation | done (2.0 W7: `stabilizer_qec.quits`, QUITS 1.2.0 whole; its matrices and circuits equal to QUITS') |
| U2 | QUITS sliding-window BP+OSD and BP+LSD for phenomenological and circuit memory; spacetime distance estimate; girth/cycle analysis and LDPC optimisation | done (2.0 W7: `stabilizer_qec.quits`; window decodes equal to QUITS' shot for shot; 2.1: with OSD-E/OSD-CS of order 2 and up too, where OSD's candidate weights and C qsort's order among equal posteriors decide: in QUITS' options sweep (phenomenological, every BP method, schedule and OSD/LSD method at order 2, 100 shots each) BP+OSD went from 4 differing shots to 0, and QUITS' OSD tie repro from 1 of 1,500 to 0; its `examples/circuit_distance_search.py` finds Stim's errors now that `search_for_undetectable_logical_errors` takes Stim's steps. 2.2: BP+LSD equal too, with ldpc's answers replayed. ldpc's answer to a window can vary from decode to decode in one process (it merges three or more clusters in an order that follows their heap addresses). `BpLsdDecoder._merge_order_outcomes` walks the merge orders this decoder can take, fewest departures from `decode`'s order first: on each of the 15 windows of QUITS's sweep where ldpc's answer varied (`tests/ldpc_lsd_merge/`), it reaches exactly two corrections within its default 4,096 orders, this decoder's and ldpc's other. `tests/test_quits_differential.py` requires each of ldpc's BP+LSD window answers to be this decoder's or one it reaches, then replays ldpc's answers in this package's sliding windows and requires the predictions equal exactly. The distance search takes less time than Stim's and under a third of its memory (at PYTHONHASHSEED 0, the whole process: 0.64 to 0.66 s user and a 144 to 148 MB peak footprint against Stim's 4.25 to 4.54 s and 530 MB), keeping its sets within `max_memory_bytes` (half the machine's memory by default) where 2.1 stopped at 20 million) |
| U3 | qec-util circuit tools: remove gauge detectors, remove detectors/observables, observables_to_detectors, redefine/merge observables, move observables to end, format rec targets | done (2.0 W4: `stabilizer_qec.analysis`, qec-util whole; 2.1: Stim's own circuits and models taken too, read from their text, where it took only this package's) |
| U4 | qec-util DEM tools: dem_difference, hyperedge decomposition to edges, disjoint graphs, flippable detectors/observables, errors triggering detectors, merge instructions | done (2.0 W4: `stabilizer_qec.analysis`, qec-util whole; 2.1: Stim's own circuits and models taken too, read from their text, where it took only this package's) |
| U5 | qec-util analysis: p_ij matrix (exact and approximate), syndromes/defects/defect probabilities, logical error decay fit (`LogicalErrorProbDecayModel`), threshold estimation and plots, circuit distance upper bound | done (2.0 W4: `stabilizer_qec.analysis`, qec-util whole; 2.1: Stim's own circuits and models taken too, read from their text, where it took only this package's) |
| U6 | qec-util samplers: sample_failures to file, merge batches | done (2.0 W4: `stabilizer_qec.analysis`, qec-util whole; 2.1: Stim's own circuits and models taken too, read from their text, where it took only this package's) |
| U7 | PanQEC: 3D codes (3D toric, rhombic, XCube fracton, Haah-like), deformed (XZZX, biased) codes, BP-OSD, MBP, union-find, sweep-matching, threshold FSS analysis, GUI | done (2.0 W7: `stabilizer_qec.panqec`, PanQEC 0.1.7 whole; codes, noise, decodes, simulations and GUI equal to PanQEC's. 2.1: its command line as `stabilizer-qec panqec`, which the cluster scripts `generate-cluster-script` writes call) |
| U8 | qecsim: MPS tensor-network decoders (planar, colour 6.6.6, toric; exact and approximate by bond dimension), fault-tolerant (time-repeated) runs, Blossom V | done (2.0 W6: `stabilizer_qec.qecsim`, qecsim whole, its MPS decoders too. 2.1: its command line as `stabilizer-qec qecsim`, extensions' `qecsim.cli.run*` entry points read as qecsim reads them; past 512 nodes the Blossom V stand-in matches with Fusion Blossom, exactly — qecsim's `run-ftp` 13×13 example in 40 s, where NetworkX ran past five minutes; center-slice labels print as under NumPy 1) |
| U9 | PECOS: fault-tolerance checks (`t_errors_check`, `fault_check`, `distance_check`), pseudo-threshold tools, error generators, 4.4.4.4 surface and 4.8.8 colour codes, density-matrix and process-matrix simulators | done (2.0 W7: `stabilizer_qec.pecos`, PECOS 0.2.0 whole, its C++ simulator in Rust; simulators, codes, runs and rates equal to PECOS's, its broken checks fixed. W8: its `DensityMatrix` and `ProcessMatrix` simulators, which PECOS declares and does not ship, checked against its stabilizer simulators and logical runs. 2.1: the default `SparseSim` has the Python one's `print_stabs(print_destabs, print_y)` (PECOS's docs' output exactly), `copy`, `refactor`, `find_stab`, `run_direct` and `'G2'`; more PECOS bugs fixed — `'init \|-i>'` on the C++ simulator, Y's printed sign, phenomenological error models, mixed two-qubit error sets on NumPy ≥ 1.24, `TimingRunner.total_time`, `fault_tolerance_check` (now passing the shipped codes), instruction plots on networkx 3; listed in `stabilizer_qec.pecos`'s docstring) |
| U10 | Fusion Blossom: parallel MWPM by partitioned primal/dual, visualiser | done (2.0 W6: `stabilizer_qec.fusion_blossom`, Fusion Blossom whole in Rust; 2.2: `helper.bottle` too, fusion_blossom 0.2.13's vendored Bottle (0.14-dev) byte for byte, imported with the package as fusion_blossom imports its own, and `helper.serve` is fusion_blossom's server on it: 55 recorded requests (18 paths by GET, HEAD and POST, and a range request) answered as fusion_blossom's server answers them, headers and bodies) |
| U11 | MWPF: hypergraph minimum-weight parity factor decoder, hypergraph union-find (HUF), heralded-erasure DEMs | done (2.0 W6: `stabilizer_qec.mwpf`, MWPF whole in Rust; not `QECPlaygroundCode`, an integration with QEC-Playground's simulator, see the spec's out of scope. 2.1: its sinter decoder, `HeraldedDetectorErrorModel.of`, `RefCircuit.of` and `with_circuit` take Stim's own circuits and models, which once decoded every shot to 0) |
| U12 | qsample: dynamical subset sampling (importance sampling by fault count, upper and lower bounds), protocol graphs with feed-forward | done (2.0 W4: `stabilizer_qec.qsample`; 2.1: its drawing dependencies, which qsample requires and this package does not install (latextools, matplotlib, networkx, pydot), are named with their `pip install` when missing) |

## F — Fault-tolerant compilation and estimation (tqec, Loom, Qualtran, Azure RE, Plaquette)

| ID | Feature | Status |
|---|---|---|
| F1 | tqec block graphs: cubes, pipes, ports; compile to Stim circuits with detectors (fixed bulk and fixed boundary conventions); correlation surfaces (ZX) → observables | done (2.0 W10: `stabilizer_qec.tqec`, tqec's `computation` package ported without networkx or PyZX — `BlockGraph`, kinds, validation, `fill_ports`, `fill_ports_for_minimal_simulation`, `compose`, `rotate`, `fix_shadowed_faces`, correlation surfaces on its own ZX graph, equal to tqec's on every gallery graph; `BlockGraph.compile` through the surgery engine for graphs of one patch orientation with time and single-direction spatial pipes (memory, CNOT), the correlation surfaces its observables. 2.1: tqec's call shape, `compile_block_graph(g, convention=...)` a `TopologicalComputationGraph` with `generate_stim_circuit(k=..., noise_model=...)` (`block_temporal_height` and `LinearFunction` as tqec's); `NoiseModel.uniform_depolarizing` and `si1000` equal to tqec's instruction for instruction (Gidney's, CC BY 4.0); `fix_shadowed_faces` first, so a pass-through drawn XXZ compiles; `observables="auto"`/`None` as tqec's; a tile reused at different times (two CNOTs in a row) on the engine; `tqec.simulation` (`start_simulation_using_sinter`, `binary_search_threshold`, the split statistics, the inset plots, the CSV writer) on `stabilizer_qec.sinter`; `tqec.interop.pyzx`: `ZXGraph.g`, `PositionedZX`, `block_synthesis` (without PyZX on this package's ZX graph), Pauli webs and PyZX graphs into `find_correlation_surfaces`, equal to tqec's on every gallery graph. 2.2: tqec 0.2.0's compile pipeline (`circuit`, `plaquette`, `templates`, `compile`) with tqecd 0.2.1's detector search, on numpy alone: `compile_block_graph(graph, convention=...)`'s `generate_stim_circuit` writes tqec's circuit from tqec's own plaquettes, templates and detectors, Hadamard pipes and spatial junction cubes included (the CNOT at k = 1 has tqec's 81 qubits); all 51 circuits recorded from tqec 0.2.0 byte-identical and its 6 recorded refusals with tqec's type and message; 6 to 16 times faster than tqec. Each compilation starts from fresh caches (a fresh tqec process's circuit), the detector database is kept in memory, there are no worker processes, and `k` must be a positive integer. Y cubes raise tqec's `NotImplementedError("Y cube is not implemented.")`, as tqec 0.2.0 has none. The engine's route, `compile_block_graph(graph, distance)` and `BlockGraph.compile`, is unchanged) |
| F2 | tqec gallery: memory, stability, CNOT, CZ, move/rotation, three CNOTs, Steane encoding | done (2.0 W10: `stabilizer_qec.tqec.gallery`, the seven builders, Steane encoding from tqec's Collada file packaged. 2.1: compositions that reuse a tile (`cnot().compose(cnot(), "Out_Control", "In_Control")`, and through the targets) compile here; `cz` takes flows in Stim's syntax, sparse too (`"X0 -> X0*Z1"`, placed by qubit index where tqec 0.2.0 misplaces it), its errors worded as tqec's (`X_ -> XX is not a valid flow for the CZ gate.`); 2.2: every gallery graph compiles by tqec's pipeline (F1), stability, move/rotation, three CNOTs, Steane encoding and CZ included, each circuit tqec's; tqec 0.2.0's own bugs come with its circuits, and the fixtures record which: two `cz` flows' observables are non-deterministic at k = 1 (tqec issue #1034), and a non-deterministic detector is in `cz("ZZ -> ZZ")`'s fixed-boundary circuit at k = 2 and in two L-shaped spatial-junction memories stacked with no pipe between them (#1062, fixed after 0.2.0)) |
| F3 | tqec `.dae` (Collada) import/export, 3D visualisation, SketchUp interop | done (2.0 W10: `to_dae_file`/`from_dae_file` and `to_json`/`from_json` with the standard library, tqec's node names, scales, colours and correlation-surface rectangles; each reads the other's files; no 3D viewer) |
| F4 | Loom code operations: grow, shrink, merge, split, move block, rotate block, move corners, Y-wall (phase), state injection, transversal Hadamard, auxiliary CNOT | done (2.1: move block, rotate block, move corners, the Y-wall phase gate, T and S state injection, transversal Hadamard and the auxiliary CNOT on the rotated surface code, gate for gate Loom's own; with Loom's other code factories — the repetition chain with its grow, shrink, merge and split, the Shor, Steane and five-qubit codes — its circuit error models, the OpenQASM 3, PennyLane, CUDA-Q, Mimiq and Guppy exporters, the run-outcome parsers and `Executor`. 2.0 W10: `stabilizer_qec.loom`, Loom 0.4.0's Eka model and interpreter ported — blocks, `RotatedSurfaceCode.create` equal to Loom's on 72 configurations, reset, syndrome rounds, logical measurements and Paulis, conditional Paulis, grow, shrink, merge, split, parallel operations; the Stim text equals Loom's byte for byte on fourteen recorded experiments (the shrink up to a set order Loom itself does not fix).) |
| F5 | Loom validator (stabilizer, logical and syndrome-measurement checks), exporters to QASM, PennyLane, CUDA-Q, Mimiq | done (2.0 W10: `is_circuit_valid`, `is_syndrome_extraction_circuit_valid`, `is_logical_operation_circuit_valid` on `TableauSimulator`, Loom's verdicts; `EkaToStimConverter` to this package's `Circuit`, `EkaToQasmConverter` and `EkaToCudaqConverter` equal to Loom's text, `EkaToPennylaneConverter` up to the register ids; 2.1: the Mimiq and Guppy exporters too (F4). Not ported: Loom's if/else circuits (`IfElseCircuit` and its `BoolOp` conditions). In el-loom 0.4.0 the file that defines them, `src/loom/eka/ifelse_circuit.py`, alone among Loom's files carries a notice that prohibits its use, distribution and reproduction without Entropica Labs' express written consent, in place of the Apache-2.0 header; a port, and fixtures recorded by running it, need that consent (2.2's Task 15, declined)) |
| F6 | Qualtran surface-code costs: Gidney–Fowler model, CCZ2T and 15-to-1 factories, data blocks (simple, compact, intermediate, fast), rotation cost models, Beverland et al. model, multi-factory | done (2.0 W10: `stabilizer_qec.estimate.qualtran`, `qualtran.surface_code` whole on dataclasses and numpy — `QECScheme`, `LogicalErrorModel`, `PhysicalParameters`, the four data blocks, `CCZ2TFactory`, `FifteenToOne` with Litinski's density-matrix model, `MultiFactory`, the rotation cost models, `PhysicalCostModel`, the Gidney–Fowler spreadsheet and grid search, Beverland et al.'s `minimum_time_steps`/`code_distance`/`t_states`; Qualtran's pinned numbers and, where installed, Qualtran's own outputs equal. 2.1: a package with Qualtran's 15 submodules under their names, `ui`'s estimates and figures (`update` and its helpers, figures as plotly figure dicts) equal to Qualtran's on 448 recorded inputs, `AlgorithmSummary.from_bloq` (through Qualtran's resource counting), the 15-to-1 channels, `TComplexity`, Qualtran's reprs and abstract bases; the 15-to-1 `p_fail`/`p_out` differ from Cirq's simulation in the last bits. 2.2: the Dash app, Qualtran 0.7.0's layout builders and callback (`ui.make_app()`, `ui.app`): its layout and callback equal Qualtran's, and so do its answers through Flask's test client; `ui.serve()` and `python -m stabilizer_qec.estimate.qualtran.ui` run it on Dash where dash is installed (the `dash` extra) and otherwise on the standard library's HTTP server, with a page of this package's own that draws the same layout and runs the same callback) |
| F7 | Azure RE: custom qubit parameters (separate readout and idle errors), surface and Floquet schemes, custom QEC formulas, distillation units, error budget split, constraints, batch configurations, frontier (Pareto) | done (2.0 W10: `stabilizer_qec.estimate.azure`, the QDK's `estimates` and `system` crates ported — the six presets and custom gate-based or Majorana qubits with their normalisation, surface and Floquet schemes and custom formulas, the PSSPC layout, the budget split, the 15-to-1 units and custom units, `RoundBasedFactory` with the `probability` crate's binomial quantile, the exhaustive search with its code-distance iterators and Pareto population, the unrestricted, max-duration and max-qubits estimates and the frontier, `maxTFactories`, `logicalDepthFactor`, `maxDistillationRounds`; equal to the QDK on 55 recorded configurations (five of them frontiers, entry for entry) and live. 2.1: the whole result as the QDK gives it (`EstimatorResult`, the report and the counts in words, batches with `{code, message}` items), its errors (`EstimatorError` with its codes and messages, `QSharpError` with serde_json's reason and column) and its echo of the parameters, equal on 286 recorded inputs. 2.2: the QDK's parameter classes (qdk 1.33.1's `EstimatorParams` and its sections, the QDK's code), taken by `LogicalCounts.estimate` and `azure_estimate` and raising the QDK's own errors (a `LookupError` for a missing field of a custom qubit or unit, a name with a custom unit, or both maximum constraints; a bare `Exception` from `EstimatorParams().items` without `num_items`); and the report: `_repr_html_`, `summary` and slices as text with their HTML, `diagram`, a batch's `plot`, and qsharp_widgets 1.33.1's space and scatter charts as static SVG, checked against 40 recorded results and the charts against the widgets' own functions run in node, to 4 ulps. `EstimatorError` stays a `ValueError` (the QDK's derives from BaseException alone); the widgets' live behaviour is not ported) |
| F8 | Plaquette: leakage noise in lattice-surgery CX thresholds | done (2.0 W10: `tools/surgery_leakage.py` → `data/surgery/leakage.json`, README table — the surgery CNOT under SD6 with `LEAKAGE(f·p)` and `RELAX(0.1)` after every CX, d = 3/5/7, 20k shots per point, decoded blind and leakage-aware with heralds; thresholds by crossing and by qec-util's WHP fit: 0.49% → 0.45% / 0.37% / 0.23% at f = 0.03 / 0.1 / 0.3, no crossing found at f = 0.5 in a 3,000-shot run; heralds move it by less than its interval) |

## X — Other features named in 2026 tools

| ID | Feature | Status |
|---|---|---|
| X1 | Density-matrix simulation (Quantumsim, PECOS) with amplitude/phase damping | done (2.0 W8: `DensityMatrixSimulator`, `Circuit.density_matrix`, `exact_distribution(method="density_matrix")`, phase damping and `THERMAL_RELAXATION`, process matrices; equal to Quantumsim 0.2.0's to 1e-10) |
| X2 | Perturbative symbolic error rates (QuantumClifford.jl `petrajectories`) | done (2.0 W8: `Circuit.perturbative_distribution`, numeric or symbolic, and `perturbative_logical_error_rate`; QuantumClifford's pinned polynomials) |
| X3 | Graph states, entanglement entropy, Clifford enumeration (QuantumClifford.jl) | done (2.0 W8: `stabilizer_qec.clifford`, QuantumClifford's algorithms and tests) |
| X4 | Post-selection decoding by soft output (arXiv 2601.17757) | done (2.0 W6: argument reweighting's post-selection) |
| X5 | Decoder conformance checks: syndrome consistency, presentation sensitivity, min-weight check (arXiv 2609.07035) | done (2.0 W6: `stabilizer_qec.conformance`) |
| X6 | Microsoft QDK QEC: validating and debugging encoded programs | gap (= F5) |

---

## What 1.9 has that none of them do (keep, and keep ahead)

- the coherent sampler at circuit level, beyond state-vector size, its weights exact by contracting each shot's class sum (2.2) wherever the contraction fits its budget (past it, and in the joint sampler, a kernel of more than twelve generators is truncated as in 2.1);
- Stim-identical DEMs with correlated matching ≤ 1.0× PyMatching time;
- bit-identical ports of ldpc's OSD/LSD (LSD's merge orders walked where ldpc's own vary), Relay-BP and Tesseract's search in one package, PyMatching's sparse blossom transliterated (2.2), and on it Chromobius's construction with its weights bit for bit and its predictions (on macOS on every shot checked; off it the C++ library's `std::unordered_map` order can lift a shot another way);
- gross-code logical measurement by gauging with exact ILP distances;
- the browser site that runs everything in WASM.

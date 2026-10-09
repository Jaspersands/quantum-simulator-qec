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
| S1 | `PauliString` (28 members: products, commutes, after/before through circuits, to/from numpy, unitary, iter_all, random, weight, sign) | gap |
| S2 | `PauliStringIterator` | gap |
| S3 | `Tableau` (44: from_circuit/conjugated_generators/named_gate/numpy/stabilizers/state_vector/unitary_matrix, to_*, inverse, then, append/prepend, x/y/z_output(_pauli), inverse_*_output, iter_all, random, `__call__` on PauliString) | gap |
| S4 | `TableauIterator` | gap |
| S5 | `TableauSimulator` (70: every gate method, measure/measure_kickback/measure_many/measure_observable, peek_x/y/z/bloch/observable_expectation, postselect_*, canonical_stabilizers, state_vector, set_state_from_*, do/do_circuit/do_pauli_string/do_tableau, current_inverse_tableau, noise channels) | gap (engine has a tableau; not exposed) |
| S6 | `CliffordString` (single-qubit Clifford arrays: products, powers, x/y/z_outputs, random, all_cliffords_string) | gap |
| S7 | `FlipSimulator` (batch frame simulator you drive by hand: do, broadcast_pauli_errors, set_pauli_flip, peek_pauli_flips, get_*_flips, append_measurement_flips, generate_bernoulli_samples, to_numpy) | gap |
| S8 | `Flow` and `Circuit.has_flow / has_all_flows / flow_generators / solve_flow_measurements / time_reversed_for_flows` | gap |
| S9 | `GateData` / `stim.gate_data()` (aliases, flows, inverse, generalized_inverse, hadamard_conjugated, tableau, unitary_matrix, predicates) | gap |
| S10 | `GateTarget` plus `target_rec/inv/x/y/z/pauli/combiner/combined_paulis/sweep_bit` | gap |
| S11 | `CircuitInstruction`, `CircuitRepeatBlock` (name, tag, targets_copy, gate_args_copy, target_groups, num_measurements, body_copy, repeat_count) | gap |
| S12 | `DemInstruction`, `DemRepeatBlock`, `DemTarget` plus `target_relative_detector_id/logical_observable_id/separator` | gap |
| S13 | Fields on `ExplainedError`, `CircuitErrorLocation`, `CircuitErrorLocationStackFrame`, `CircuitTargetsInsideInstruction`, `FlippedMeasurement`, `GateTargetWithCoords`, `DemTargetWithCoords` matching Stim's attribute names (ours are dataclass fields; check all names) | part |

### S.2 `Circuit` methods

| ID | Feature | Status |
|---|---|---|
| S20 | `__getitem__` (index, slice), `__len__`, `insert`, `pop`, `clear`, `append_operation`, `append(CircuitInstruction/RepeatBlock)` | gap |
| S21 | `approx_equals` | gap |
| S22 | `flattened`, `flattened_operations`, `decomposed`, `inverse`, `without_noise`, `without_tags`, `with_inlined_feedback` | gap |
| S23 | `num_ticks`, `get_detector_coordinates`, `get_final_qubit_coordinates`, `count_determined_measurements`, `reference_detector_and_observable_signs`, `missing_detectors` | gap |
| S24 | `detecting_regions` | gap |
| S25 | `to_tableau`, `to_file`, `to_qasm` (open_qasm_version 2/3, skip_dets_and_obs), `to_quirk_url`, `to_crumble_url` | gap |
| S26 | `likeliest_error_sat_problem`, `shortest_error_sat_problem` (WCNF text) | gap |
| S27 | `detector_error_model(allow_gauge_detectors=, block_decomposition_from_introducing_remnant_edges=)` | part |
| S28 | `compile_sampler(reference_sample=)`, `compile_m2d_converter(skip_reference_sample=)` | part |
| S29 | `diagram(filter_coords=)` | part |

### S.3 `DetectorErrorModel` methods

| ID | Feature | Status |
|---|---|---|
| S30 | `+`, `+=`, `*`, `*=`, `__getitem__`, `__len__`, `append`, `clear`, `copy` | gap |
| S31 | `approx_equals`, `rounded`, `get_detector_coordinates`, `to_file`, `without_tags` | gap |

### S.4 Samplers and converters

| ID | Feature | Status |
|---|---|---|
| S40 | `sample_bit_packed` on measurement and detector samplers | gap |
| S41 | `sample_write` (to a file in any format) on measurement, detector and DEM samplers | gap |
| S42 | Detector `sample(dets_out=, obs_out=, prepend_observables=)` | part |
| S43 | m2d `convert(bit_pack_result=)`, `convert_file` | part |
| S44 | Stim's class names `CompiledMeasurementSampler`, `CompiledDetectorSampler`, `CompiledDemSampler`, `CompiledMeasurementsToDetectionEventsConverter` (aliases) | gap |

### S.5 Diagrams

| ID | Feature | Status |
|---|---|---|
| S50 | `timeline-3d`, `matchgraph-3d` (glTF) | gap |
| S51 | `interactive` (Crumble embedded) | gap |
| S52 | `*-html` variants of every diagram | gap |
| S53 | DEM `matchgraph-3d` | gap |

### S.6 CLI

| ID | Feature | Status |
|---|---|---|
| S60 | `repl` | gap |
| S61 | `help gates`, `help formats`, `help <gate>`, `help <format>` | gap |
| S62 | Every flag of every subcommand matches Stim (re-audit against `stim help <cmd>`) | part |

### S.7 Crumble and Quirk

| ID | Feature | Status |
|---|---|---|
| S70 | Crumble-equivalent editor: edit a circuit in 2D and propagate Pauli markers live (`#!pragma MARK_X0`), view and edit detectors and observables, import from and export to Stim text and URLs | gap (site has viewers only) |
| S71 | Quirk export | gap (= S25) |

## P — PyMatching 2.4

| ID | Feature | Status |
|---|---|---|
| P1 | Graph construction: `add_edge(node1,node2,fault_ids,weight,error_probability,merge_strategy)`, `add_boundary_edge`, `set_boundary_nodes`, `ensure_num_fault_ids` | gap |
| P2 | `from_check_matrix(H, weights, error_probabilities, repetitions, timelike_weights, measurement_error_probabilities, faults_matrix, merge_strategy, use_virtual_boundary_node)` and `load_from_check_matrix` | gap |
| P3 | `from_networkx/to_networkx`, `from_rustworkx`/`load_from_rustworkx`/`to_rustworkx` (and retworkx) | gap |
| P4 | `from_stim_circuit(_file)`, `from_detector_error_model_file` | part (DEM only) |
| P5 | Queries: `edges()`, `get_edge_data`, `get_boundary_edge_data`, `has_edge`, `has_boundary_edge`, `num_nodes/num_edges/num_fault_ids/boundary` | gap |
| P6 | `decode(return_weight=)`, `decode_to_edges_array`, `decode_to_matched_dets_array`, `decode_to_matched_dets_dict` | gap |
| P7 | `add_noise` (sample errors on the graph's own probabilities) | gap |
| P8 | `draw` | gap |
| P9 | Constructor `Matching(H or graph, weights, …)` | gap |
| P10 | `pymatching predict` and `pymatching count_mistakes` CLI (we have `decode`, after predict) | part |
| P11 | `set_seed`, `randomize`, `rand_float` (tie-breaking randomness) | gap |

## N — sinter 1.16

| ID | Feature | Status |
|---|---|---|
| N1 | `Task`, `TaskStats`, `AnonTaskStats`, `CollectionOptions`, `Progress`, `Fit` (strong_id, CSV lines, with_edits, `+`) | gap |
| N2 | `collect` / `iter_collect` (processes, max_shots/max_errors, batch sizing, save/resume, existing data, progress callback, hint_num_tasks, count_observable_error_combos, count_detection_events, custom_error_count_key, allowed_cpu_affinity_ids) | gap |
| N3 | Post-selection: by detector 4th coordinate, detector and observable predicates, discards | gap |
| N4 | `read_stats_from_csv_files`, `stats_from_csv_files`, CSV output format | gap |
| N5 | `fit_binomial`, `fit_line_slope`, `fit_line_y_at_x`, `log_binomial`, `log_factorial`, `shot_error_rate_to_piece_error_rate` | gap |
| N6 | `plot_error_rate`, `plot_discard_rate`, `plot_custom`, `group_by`, `better_sorted_str_terms`, `comma_separated_key_values` | gap |
| N7 | `predict_observables(_bit_packed)`, `predict_discards_bit_packed`, `predict_on_disk`, `post_selection_mask_from_4th_coord` | gap |
| N8 | `Sampler`/`CompiledSampler` and `Decoder`/`CompiledDecoder` plug-in protocols | have (`stabilizer_qec.sinter`) |
| N9 | CLI `collect`, `combine`, `plot` with all flags | gap |

## L — ldpc 2.4

| ID | Feature | Status |
|---|---|---|
| L1 | BP schedules: `parallel`, `serial`, `serial_relative`, random serial (`random_serial_schedule`, `random_schedule_seed`), `serial_schedule_order` | part (parallel only on BpDecoder/BpOsd; check BpLsd) |
| L2 | `input_vector_type="received_vector"` (decode a codeword, not a syndrome) | gap |
| L3 | Attributes: `log_prob_ratios`, `converge`, `iter`, `channel_probs`, `update_channel_probs`, `decoding`, `osd0_decoding`, `osdw_decoding`, `bp_decoding`, `omp_thread_count` | part |
| L4 | `BeliefFindDecoder` (BP then union-find, `uf_method` peeling/inversion, `bits_per_step`) | gap |
| L5 | `SoftInfoBpDecoder` (soft syndromes: cutoff, sigma) | gap |
| L6 | `BpFlipDecoder` (BP plus bit flipping) | gap |
| L7 | `mbp_decoder` (memory BP) | gap |
| L8 | `UnionFindDecoder` on any parity-check matrix (peeling and inversion) | part (ours decodes DEMs) |
| L9 | `LsdDecoder` (standalone, given bit weights) | gap |
| L10 | Overlapping-window decoders: BP+OSD, BP+LSD and PyMatching over windows of a DEM, with sinter wrappers (`ckt_noise`) | part (matching windows only) |
| L11 | `mod2`: rank, kernel, nullspace, row_echelon, reduced_row_echelon, row_basis, row_complement_basis, row_span, inverse, pivot_rows, PluDecomposition.lu_solve, estimate/compute_exact_code_distance | gap (internal gf2.rs, not exposed) |
| L12 | `code_util`: code parameters, dimension, generator matrix, distance (exact and estimated), average Hamming weights, cycle search (girth) | gap |
| L13 | `codes`: rep_code, ring_code, hamming_code, random_binary_code; `alist` I/O; protograph RingOfCirculantsF2 | gap |
| L14 | Monte Carlo: BSC simulation, single-shot and quasi-single-shot memory with analog syndromes | gap |
| L15 | sinter wrappers: SinterBpOsdDecoder, SinterLsdDecoder, SinterBeliefFindDecoder | part (ours: BP+OSD, BP+LSD) |

## R — Relay-BP 0.2.2, Tesseract, Chromobius, belief-matching, stimbposd

| ID | Feature | Status |
|---|---|---|
| R1 | `MinSumBPDecoder` in every precision: F32, F64, Fixed, I8, I16, I32, I64; Relay in F32/F64/I32/I64 | part (F64 Relay only) |
| R2 | `decode_detailed(_batch)` → iterations, success, posterior ratios | gap |
| R3 | `ObservableDecoderRunner` (observables directly, detailed results: converged, error_detected, error_mismatch_detected, better/worse quality, from_errors_* helpers) | gap |
| R4 | Tesseract `decode_to_errors`, `cost_from_errors`, `get_observables_from_errors`, `low_confidence_flag`, visualizer | part (`decode_to_faults`) |
| R5 | Tesseract `SimplexDecoder` (exact LP/ILP decoding with windows) | gap |
| R6 | Tesseract utilities `dem_from_counts`, `merge_indistinguishable_errors`, `remove_zero_probability_errors`, `build_det_orders` (BFS, coordinate, index), `build_detector_graph`, sparsify options | part |
| R7 | Chromobius `predict_weighted_obs_flips_from_dets_bit_packed` (weights out) | gap |
| R8 | BeliefMatching `from_stim_circuit` | gap |
| R9 | stimbposd BPOSD/BPLSD on a DEM | have |

## D — Deltakit 0.10 (Riverlane)

| ID | Feature | Status |
|---|---|---|
| D1 | Collision Clustering decoder (CC, arXiv 2309.05558) | gap |
| D2 | Local Clustering Decoder (LCD, arXiv 2411.10343): adaptive, leakage-aware | gap |
| D3 | Ambiguity Clustering (BP-AC, arXiv 2406.14527) | gap |
| D4 | Leakage flags in data and leakage-aware decoding (`LeakageFlags`, `HERALD_LEAKAGE_EVENT`) | part (heralds sampled, not decoded) |
| D5 | Noise models: SD6, SI1000, phenomenological, toy, `PhysicalNoise` (T1/T2-derived), noise profiles per gate, `Leakage` and `Relax` channels | part (SD6, SI1000-like in data; no T1/T2) |
| D6 | QPU model: native gate sets with times, `compile_circuit_to_native_gates` (CZ / CZSWAP), parallelise circuits, merge layers, remove identities | gap |
| D7 | Codes: rotated/unrotated planar, unrotated toric, repetition, BB, generic `CSSCode`, `StabiliserCode`, schedules (orders, types) | part |
| D8 | Stability experiments (`css_code_stability_circuit`, `QECExperimentType.STABILITY`) | gap |
| D9 | Correlation matrix p_ij from data, `create_dem_from_pij`, DEM vs p_ij comparison | gap |
| D10 | Λ fits (`calculate_lambda_*`, asymmetric), logical error per round fits, `simulate_different_round_numbers`, `predict_quops_*`, `predict_distance_for_quops` | part (Λ in tools, not API) |
| D11 | Error budget (`get_error_budget`: sensitivity of Λ to each noise parameter, plot) | gap |
| D12 | Plots: defect diagram, defect rates, detection probability on patch, Λ, LEPPR, correlation matrix | gap |
| D13 | Decoding graphs: hypergraph objects, graph distance, logicals extraction, window ids, JSON I/O | part |
| D14 | Data formats: b8/01/csv for syndromes, logical flips, measurements; c64 | part |
| D15 | Decoder managers and run_decoding_on_circuit; empirical decoding error distribution | gap |
| D16 | Noise sources for code-capacity matching: exhaustive, fixed weight, uniform erasure | gap |

## Q — qLDPC 0.4.1 (Infleqtion)

| ID | Feature | Status |
|---|---|---|
| Q1 | Classical codes: repetition, ring, cyclic, Golay, Hamming, extended Hamming, Reed–Muller, Reed–Solomon, BCH, simplex, Tanner | gap |
| Q2 | Quantum codes: five-qubit, Steane, quantum Hamming, quantum Reed–Muller, tetrahedral, quantum Golay, iceberg, C4, C6, many-hypercube, TB, GALA, QC, BB (general, toric layouts), HGP, CHGP, CRC, SHP, Bacon–Shor, SHYPS, LP, SLP, QT, surface, toric, generalized surface, T4 | part (surface, colour, BB, HGP) |
| Q3 | Qudit codes over GF(q) (`QuditCode`, `QuditPauli`, fields) | gap |
| Q4 | Non-CSS stabilizer codes (symplectic), subsystem codes, conversion to CSS/SWEL | gap |
| Q5 | Logical operators (canonical basis, symplectic Gram–Schmidt), dual basis | part (CSS logicals) |
| Q6 | Distance: exact (Brouwer–Zimmermann for CSS; brute force), bounds by random search, upper bounds | part (MILP) |
| Q7 | Code equivalence, canonical forms, `forget_distance`, code parameters | gap |
| Q8 | Groups and rings: cyclic, dihedral, alternating, symmetric, quaternion, small groups, SL/PSL/GL/PGL, group rings, lifts, Wedderburn–Artin | part (BB automorphisms) |
| Q9 | Encoding circuits and tableaus, logical tableaus, encoder/decoder pairs | gap |
| Q10 | Transversal gates: transversal S, transversal ops, automorphism group, transversal circuits for given logical Cliffords | part (BB automorphisms) |
| Q11 | Memory experiments for any code: syndrome-measurement strategies (edge colouring, EdgeColoringXZ), AlphaSyndrome (MCTS schedule search), Bell-state logical prep, observables | part (CSS memory, one schedule) |
| Q12 | State-prep diagnostics (flags, post-selection, discard rate) | gap |
| Q13 | Noise models: `NoiseModel` rules, depolarizing, SI1000, Pauli channels, `as_noiseless_circuit` | part |
| Q14 | Decoders: lookup (plain, weighted, observable), ILP, GUF, composite, direct, frontier, subgraph, sequential and sliding windows, trivial; erasure flags | part |
| Q15 | `DetectorErrorModelArrays`: simplified, without_detectors, post_selected_on, with_decomposed_errors, to_circuit | gap |
| Q16 | Code-capacity DEMs and decoders; Monte Carlo with weight-truncated error-rate functions (`ErrorRateFunc`, Jeffreys variance, sample allocation by weight) | gap |
| Q17 | Experimental surgery: bridges between codes, Cheeger constant, gadget boosting, single and joint Pauli-product measurement circuits | part (gross gauging) |
| Q18 | External: qecdb / QLDPC-challenge code fetch, GAP/Magma groups, sqetch distance bounds | gap (network or third-party; offline equivalents only) |

## T — Tsim (QuEra, bloqade-tsim 0.1.5)

| ID | Feature | Status |
|---|---|---|
| T1 | Clifford+T sampling by stabilizer rank (ZX), cost exponential in non-Clifford count, 80+ qubits | gap (state vector ≤ 24 qubits) |
| T2 | Gates T, T_DAG, R_X/Y/Z, U3, R_PAULI, TPP, R_XX/YY/ZZ (plain gates, not only tags) | part (tags) |
| T3 | CCZ, CCX | gap |
| T4 | `CompiledStateProbs.probability_of` (exact output probabilities) | part (exact_distribution ≤ 24) |
| T5 | `to_matrix`, `to_tensor`, `tcount`, `is_clifford`, `cast_to_stim`, `inverse` with non-Clifford gates | gap |
| T6 | DEM of a non-Clifford circuit (Clifford part, non-deterministic observables allowed) | part |
| T7 | Encoders: TransversalEncoder, Steane encoder, colour-code encoder | gap |
| T8 | GPU (JAX, CUDA) | gap |

## C — CUDA-Q QEC 0.8, cuStabilizer

| ID | Feature | Status |
|---|---|---|
| C1 | Tensor-network decoder (exact maximum likelihood by contraction) | gap |
| C2 | Single-error and multi-error lookup-table decoders | gap |
| C3 | Sliding-window decoder around any inner decoder (boundary-aware) | part (matching only) |
| C4 | Relay-BP gamma ensembles, `relay_solutions` post-processing; BP+OSD `osd_init_method="min_llr"`; sum-product BP variants | part |
| C5 | AI pre-decoder pipeline: neural network clears local errors, residual goes to matching (NVIDIA Ising CNN, arXiv 2604.12841) | gap |
| C6 | Async and streaming API: `decode_async`, `enqueue_syndromes`, `get_corrections`, `reset_decoder`, decoder stats (latency) | part (`stream_memory`) |
| C7 | Decoder server (real-time, RPC) | gap |
| C8 | PCM utilities: `get_pcm_for_rounds`, `pcm_extend_to_n_rounds`, sorting/simplifying/shuffling columns, `generate_random_pcm`, timelike sparse detector matrix | gap |
| C9 | `sample_code_capacity`, `sample_memory_circuit` (x/z), `dem_from_memory_circuit` (x/z), `dem_sampling` | part |
| C10 | GPU frame sampling (cuStabilizer: X/Z/leakage tables, up to 150× one CPU thread) and GPU decoders | gap |
| C11 | Surface-code orientations XV, XH, ZV, ZH | gap |

## M — MQT QECC 2.0

| ID | Feature | Status |
|---|---|---|
| M1 | Encoding-circuit synthesis: Gottesman, heuristic, gate-optimal and depth-optimal (SAT), for CSS and non-CSS | gap |
| M2 | Fault-tolerant state preparation: verification circuits (gate-optimal), deterministic verification with corrections, flags, hooks | gap |
| M3 | Cat-state preparation (balanced tree, line, pruned, fused) and experiments | gap |
| M4 | Clifford synthesis (exact gate count and depth, SAT), CNOT-circuit synthesis (greedy and rollout) | gap |
| M5 | Code switching compiler (minimal switching, min cut) | gap |
| M6 | Colour-code MaxSAT decoder (LightsOut) | gap |
| M7 | Analog Tanner-graph decoding (soft syndromes), single-shot and quasi-single-shot simulators | gap |
| M8 | Codes: concatenated (CSS), quantum Hamming, iceberg, many-hypercube, 3D/4D (sparse) hypergraph product, hexagonal and square-octagon colour codes, rotated surface | part |
| M9 | Stabilizer tableau objects (`StabilizerTableau`, `Pauli`, symplectic vectors and matrices), `CliffordIsometry` | gap (= S3/S5) |
| M10 | Routing for lattice surgery (cococo: hill climbing, snake builders, teleportation router, Steiner trees) | gap |
| M11 | Fault sets, coset leaders, undetectable-fault search on CNOT circuits | part |

## U — QUITS, qec-util, PanQEC, qecsim, PECOS, Fusion Blossom, MWPF, qsample

| ID | Feature | Status |
|---|---|---|
| U1 | QUITS codes: BB, BPC, HGP, LCS, Mitten, QLP (and polynomial QLP); circuit builders (cardinal, cardinal NS-merge, ZX colouration, custom); layouts (toric, BB toric, transversal); stabilizer-flow schedule validation | part |
| U2 | QUITS sliding-window BP+OSD and BP+LSD for phenomenological and circuit memory; spacetime distance estimate; girth/cycle analysis and LDPC optimisation | gap |
| U3 | qec-util circuit tools: remove gauge detectors, remove detectors/observables, observables_to_detectors, redefine/merge observables, move observables to end, format rec targets | gap |
| U4 | qec-util DEM tools: dem_difference, hyperedge decomposition to edges, disjoint graphs, flippable detectors/observables, errors triggering detectors, merge instructions | part |
| U5 | qec-util analysis: p_ij matrix (exact and approximate), syndromes/defects/defect probabilities, logical error decay fit (`LogicalErrorProbDecayModel`), threshold estimation and plots, circuit distance upper bound | part |
| U6 | qec-util samplers: sample_failures to file, merge batches | gap (= N2) |
| U7 | PanQEC: 3D codes (3D toric, rhombic, XCube fracton, Haah-like), deformed (XZZX, biased) codes, BP-OSD, MBP, union-find, sweep-matching, threshold FSS analysis, GUI | part (XZZX, bias in memory_circuit) |
| U8 | qecsim: MPS tensor-network decoders (planar, colour 6.6.6, toric; exact and approximate by bond dimension), fault-tolerant (time-repeated) runs, Blossom V | gap (MPS decoders) |
| U9 | PECOS: fault-tolerance checks (`t_errors_check`, `fault_check`, `distance_check`), pseudo-threshold tools, error generators, 4.4.4.4 surface and 4.8.8 colour codes, density-matrix and process-matrix simulators | gap |
| U10 | Fusion Blossom: parallel MWPM by partitioned primal/dual, visualiser | part (window parallelism) |
| U11 | MWPF: hypergraph minimum-weight parity factor decoder, hypergraph union-find (HUF), heralded-erasure DEMs | gap |
| U12 | qsample: dynamical subset sampling (importance sampling by fault count, upper and lower bounds), protocol graphs with feed-forward | gap |

## F — Fault-tolerant compilation and estimation (tqec, Loom, Qualtran, Azure RE, Plaquette)

| ID | Feature | Status |
|---|---|---|
| F1 | tqec block graphs: cubes, pipes, ports; compile to Stim circuits with detectors (fixed bulk and fixed boundary conventions); correlation surfaces (ZX) → observables | part (surgery programs) |
| F2 | tqec gallery: memory, stability, CNOT, CZ, move/rotation, three CNOTs, Steane encoding | part (CNOT, merges) |
| F3 | tqec `.dae` (Collada) import/export, 3D visualisation, SketchUp interop | gap |
| F4 | Loom code operations: grow, shrink, merge, split, move block, rotate block, move corners, Y-wall (phase), state injection, transversal Hadamard, auxiliary CNOT | part (merge/split) |
| F5 | Loom validator (stabilizer, logical and syndrome-measurement checks), exporters to QASM, PennyLane, CUDA-Q, Mimiq | gap |
| F6 | Qualtran surface-code costs: Gidney–Fowler model, CCZ2T and 15-to-1 factories, data blocks (simple, compact, intermediate, fast), rotation cost models, Beverland et al. model, multi-factory | part (our estimator) |
| F7 | Azure RE: custom qubit parameters (separate readout and idle errors), surface and Floquet schemes, custom QEC formulas, distillation units, error budget split, constraints, batch configurations, frontier (Pareto) | part |
| F8 | Plaquette: leakage noise in lattice-surgery CX thresholds | part |

## X — Other features named in 2026 tools

| ID | Feature | Status |
|---|---|---|
| X1 | Density-matrix simulation (Quantumsim, PECOS) with amplitude/phase damping | part (state vector with trajectories) |
| X2 | Perturbative symbolic error rates (QuantumClifford.jl `petrajectories`) | gap |
| X3 | Graph states, entanglement entropy, Clifford enumeration (QuantumClifford.jl) | gap |
| X4 | Post-selection decoding by soft output (arXiv 2601.17757) | gap |
| X5 | Decoder conformance checks: syndrome consistency, presentation sensitivity, min-weight check (arXiv 2609.07035) | gap |
| X6 | Microsoft QDK QEC: validating and debugging encoded programs | gap (= F5) |

---

## What 1.9 has that none of them do (keep, and keep ahead)

- the coherent sampler at circuit level, beyond state-vector size;
- Stim-identical DEMs with correlated matching ≤ 1.0× PyMatching time;
- bit-identical ports of ldpc's OSD/LSD, Relay-BP, Chromobius and Tesseract in one package;
- gross-code logical measurement by gauging with exact ILP distances;
- the browser site that runs everything in WASM.

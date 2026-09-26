# Sub-project B2 — correlated matching, and Google's hardware data

**Date:** 2026-09-25
**Status:** Approved (Jasper approved the framing, the site, the ordering and all five design sections interactively)

## Where this sits

| | Sub-project | State |
|---|---|---|
| A | General circuit path, checked against Stim and PyMatching | Done, merged and deployed |
| B1 | Exact sparse matcher | Done on `feature/willow-data`, not merged |
| **B2** | **Correlated matching, and Google's Willow and Sycamore data (this spec)** | — |
| C | Real-time decoding | — |
| — | Launch kit: pip wheels, CI, written report | — |
| D | Belief-matching and BP, evaluated on the Willow data | — |
| F | IBM's gross code with BP+OSD | — |
| E | Lattice surgery | — |
| — | Resource estimator driven by measured Λ | Needs B2 |

B2 was to decode Google's data with the plain matcher and compare Λ with Google's published decoders.
Every Willow decoder Google released is correlated matching or an ensemble of correlated matchers,
so Jasper chose to pull correlated matching into B2 and make the headline comparison like for like.

## Decisions

| Question | Decision |
|---|---|
| Framing against Google's decoders | Build correlated matching now, so ours can be compared with theirs like for like. |
| Correlated algorithm | PyMatching 2.4's two-pass edge reweighting, reproduced from its source, so PyMatching's `enable_correlations` is an exact oracle. |
| Structure | One spec, two plans in sequence: Plan 1 the correlated matcher, Plan 2 the data, fits, site and README. The full run happens once, with both decoders. |
| Priors | Willow: Google's SI1000 and RL-optimised models, and ours if it differs from their SI1000. Sycamore: the circuit model and the cross-fitted `pij` models. Each with plain and correlated matching. |
| Fitting | One implementation, in JavaScript, used by a Node tool for the README and by the page. Applied identically to our predictions and Google's. |
| Site | A new section 11: the recorded run, marked †, and a live panel that decodes 2,000 real shots per distance from Google's raw readouts. |
| Data on disk | Read straight from the zips in `data/google/`; nothing is unpacked. |
| Licence | Both datasets are CC BY 4.0 (checked on Zenodo, records 13273331 and 6804040). The site's extract credits Google Quantum AI. |

## The datasets

**Willow** (`google_105Q_surface_code_d3_d5_d7.zip`, 12.8 GB unpacked): 14 patches (nine at d = 3,
four at d = 5, one at d = 7), X and Z bases, 15 round counts (1, 10, 13, 30, 50, …, 250), 50,000
shots each: 420 experiments, 21.0 million shots. Each experiment has `circuit_ideal.stim`,
`circuit_noisy_si1000.stim`, `measurements.b8`, `sweep_bits.b8`, `detection_events.b8`,
`obs_flips_actual.b8`, `metadata.json`, and per decoding pathway an `error_model.dem` and
`obs_flips_predicted.b8`. The pathways are correlated matching with the SI1000 prior, correlated
matching with the RL-optimised prior, Harmony with each prior, and Libra with the RL-optimised prior.

**Sycamore 2022** (`google_qec3v5_experiment_data.zip`): d = 3 (four patches) and d = 5 (one patch),
X and Z bases, rounds 1–25 (odd), 50,000 shots each: 130 surface-code experiments, 6.5 million
shots. Each has `circuit_ideal.stim`, `circuit_noisy.stim`, `measurements.b8`, `sweep.b8`,
`detection_events.b8`, `obs_flips_actual.01`, `circuit_detector_error_model.dem`, the two
cross-fitted models `pij_from_even_for_odd.dem` and `pij_from_odd_for_even.dem`, and predictions
from PyMatching, correlated matching, belief-matching and tensor-network contraction. Google's
PyMatching and correlated matching used `circuit_detector_error_model.dem` for every shot; belief
matching and tensor-network contraction used `pij_from_even_for_odd` on odd shots and
`pij_from_odd_for_even` on even shots.

**What the engine lacks.** A scan of all 1,102 circuit files finds exactly four things the parser
does not yet take: `sweep[k]` targets on `CX`, and the gates `X`, `Y` and `I`. The model files use
only `error` and `detector`, and 73% of their errors carry `^` decompositions.

**Published numbers**, from the abstracts:
- Willow (arXiv:2408.13687): Λ = 2.14 ± 0.02, and d = 7 at 0.143% ± 0.003% error per cycle. This
  is Google's neural-network decoder, whose predictions are not in the dataset.
- Sycamore (arXiv:2207.06431): d = 5 at 2.914% ± 0.016% per cycle against d = 3 at 3.028% ± 0.023%.

---

# Part 1 — the correlated matcher (Plan 1)

## The algorithm, as PyMatching 2.4 implements it

Read from PyMatching's source (`user_graph.{h,cc}`, `flooder/graph.{h,cc}`,
`driver/mwpm_decoding.cc`).

**Graph.** The same merged edges as the plain matcher: every graph-like piece of every error,
decomposed pieces included, merged as independent events.

**Joint probabilities.** For each error of probability `p` with pieces `c_0 … c_k`:
- for every unordered pair `k0 < k1`: `joint[c_k0][c_k1] ⊕= p` and `joint[c_k1][c_k0] ⊕= p`, where
  `a ⊕ p = a(1 − p) + p(1 − a)`;
- for every piece: `joint[c][c] ⊕= p`, the edge's marginal.

Errors with `p = 0` are skipped, as are errors with no detectors outside a decomposition.

**Rewrite rules.** For each edge `c` with marginal `m = joint[c][c] > 0`, and every other edge `a`
in `joint[c]`: `p_a = min(0.5, joint[c][a] / m)`, implied weight `w_a = int_weight(ln((1 − p_a)/p_a))`
(even, non-negative).

**Decode, two passes.**
1. **Pass 1.** The sparse matcher, unchanged. Extraction also returns the matched pairs: two defects,
   or a defect and the boundary.
2. **Paths.** For each pair, a shortest path on the detector graph in the integer weights. The
   edges of all paths are XORed together, so an edge used twice cancels.
3. **Reweight.** For each edge in that set, apply each of its rules: the affected edge's weight
   becomes `min(current, w_a)`. Both half-edges change together.
4. **Pass 2.** The sparse matcher on the reweighted graph. Its observables are the prediction, and
   its weight is reported in the reweighted weights. The weights are then restored.

## Components

| File | Responsibility |
|---|---|
| `src/sparse/correlated.rs` | `Correlations::from_dem(&Dem, &SparseGraph)`: the joint table and the rules, as a CSR from edge to `(affected edge, implied weight)`. Reweighting, restoring, and the two-pass decode. |
| `src/sparse/paths.rs` | Pass 1's edge set: a Dijkstra shortest path per matched pair, with early stop and touched-list reset, XORed into one set. |
| `src/sparse/graph.rs` | Edge ids: each undirected edge's index in the merged edges names both half-edges, and `edge_id(u, v)` finds it. |
| `src/sparse/extract.rs` | Also returns the matched pairs, for pass 1. |
| `src/sparse/state.rs` | `Scratch` gains its own copy of the edge weights, which the flooder reads, and an undo log. Reweighting never touches the shared graph, so threads stay independent. |
| `src/dem_decoder.rs` | `DemDecoder::decode_correlated(defects)`, and `new` builds the rules. A model whose decomposition the rules cannot use is refused with PyMatching's reason. |
| `src/py_api.rs` | `decode_b8(..., correlated=False)`, `decode_b8_own(..., correlated=False)`; a `Decoder(dem_text)` class with `edges(defects)`, our pass-1 edge set, and `pass2(defects, edges)`, pass 2 from an edge set supplied from outside, per shot. |
| `tools/xcheck.py` | Check 5, correlated matching against PyMatching, and the correlated timing column. |

## Verification

Five layers, each pass or fail.

1. **Rule tables.** Hand-built models with the rules worked by hand: two-piece and three-piece
   decompositions, boundary pieces, repeated pieces, the 0.5 cap, and `p = 0`.
2. **Paths.** On surface-code shots at d = 3–7: the traced edge set has exactly the shot's defects
   as its syndrome, and its integer weight equals pass 1's optimal weight, so it is a minimum-weight
   correction. Its observables are not compared with pass 1's: two equally short paths can differ
   by a logical operator (a defect midway between two boundaries), and either is a correct trace.
3. **Pass 2 is exact.** On the reweighted graph, the sparse matcher's integer weight equals the dense
   oracle's, on thousands of shots.
4. **Against PyMatching `enable_correlations`**, on identical shots of the SD6 circuits (rotated and
   XZZX, d = 3, 5, 7, p = 0.3% and 0.6%). For every shot where the predictions differ, PyMatching's
   own pass-1 edges (`decode_to_edges_array`, without correlations, which is what its correlated
   decode computes first) go into our pass 2, which must then give PyMatching's prediction or tie it
   in weight. This separates an equally short path traced differently from a bug.
5. **It helps.** Correlated matching's logical error rate is below plain matching's on the SD6
   circuits. The gain is reported as measured.

**Speed.** µs per shot, ours against PyMatching's correlated mode, single-threaded, in check 4's
table.

Plan 1 changes the harness and the README, not the site.

---

# Part 2 — the data, the fits and the site (Plan 2)

## Parser

- `CX sweep[k] q`: an X on `q` when sweep bit `k` is set, as in Stim. Only the control may be a
  sweep bit.
- `X`, `Y`, `Z`, `I`: Pauli gates.
- Neither changes a Pauli frame, so the error-model builder and the frame sampler ignore them. The
  noiseless reference run applies them.
- `to_stim` writes them back, and every Google circuit round-trips through parse and print.

## From raw measurements to detection events

`m2d` (Rust, with `m2d_b8` in Python and an export in WASM):
- One noiseless tableau run of the ideal circuit, with every sweep bit 0, gives the reference
  measurement record.
- Sweep bits act linearly: for each bit, the measurements it flips come from pushing one X through
  the circuit with the frame machinery.
- A shot's detector fires when the parity of its records differs from the reference's, after XORing
  the flips of the shot's set sweep bits. Observables likewise.

**Check:** equal, bit for bit, to Google's `detection_events.b8` and `obs_flips_actual`, on every
one of the 550 experiments: 27.5 million shots.

## Error-model check

For each noisy circuit, our model is compared with Stim's (A's comparison, mechanism by mechanism
and on the merged graph) and with Google's SI1000-prior `error_model.dem` on the merged graph. If
ours matches Google's, one prior is decoded for both; otherwise the differences are reported and
ours is decoded as its own prior.

## The full run

`tools/google.py run`, multithreaded and resumable, experiment by experiment:
1. read the files from the zip;
2. run the measurement check;
3. decode with every prior, each with plain and correlated matching (Sycamore's `pij` models
   cross-fitted as Google used them);
4. score Google's own prediction files against the same `obs_flips_actual`;
5. on Sycamore, check that every shot where our plain matcher on `circuit_detector_error_model.dem`
   disagrees with Google's recorded PyMatching is a tie, by weight from PyMatching 2.4;
6. write one record to `data/google-results/` (committed, small): failures per decoder, shots, mean
   defects per shot, µs per shot and the thread count, and each check's result.

`tools/google.py summary` prints every failed check, unmatchable shot and model mismatch. None of
these may stand unexplained under the README's numbers.

**Cost.** Estimated at tens of minutes per decoder configuration on all cores, dominated by d = 7
at 250 rounds (about 12,000 detectors and 1,000 defects a shot). Measured and reported.

## Fits

`js/lambda-fit.js`, run by `tools/lambda.mjs` for the README and by the page:

- **Per experiment:** `P_L = failures / shots`, fidelity `F = 1 − 2 P_L`, `σ_F = 2 √(P_L(1 − P_L)/N)`.
- **Per patch, basis and decoder:** a weighted least-squares fit of `ln F = ln A + r · ln(1 − 2ε)`.
  The 1-round experiment is excluded (its errors differ from the steady state), as is any point with
  `F < 3 σ_F`. The start round is stated, and the fit with one alternative start is reported as a
  robustness line.
- **Per distance:** `ε_d`, the mean over patches and both bases.
- **Λ.** Willow: `exp(−2 · slope)` of `ln ε_d` against d = 3, 5, 7, plus pairwise Λ₃/₅ and Λ₅/₇.
  Sycamore: `ε₃ / ε₅`.
- **Uncertainty:** a parametric bootstrap, each point's failures redrawn binomially and everything
  refitted.

**Validation.** Fitting Google's own Sycamore tensor-network predictions must reproduce the published
3.028% ± 0.023% and 2.914% ± 0.016%. If it does not, the protocol is brought into line with the
paper's supplement before any comparison is quoted.

## Site: section 11, "Real hardware"

After "Under the hood", so the page ends on Google's chips.

**The recorded run (†, with the run date and engine commit):**
- Willow: logical error per cycle `ε_d` against d = 3, 5, 7 on a log axis, one line per decoder: our
  plain and correlated matchers, and Google's five.
- A table of `ε₃`, `ε₅`, `ε₇`, Λ₃/₅, Λ₅/₇ and Λ with bootstrap intervals, and Google's published
  Λ = 2.14 ± 0.02 on its own row, labelled as their neural-network decoder.
- Sycamore: `ε₃`, `ε₅` and Λ, ours beside Google's four decoders and the published figures.
- Check lines: detection events re-derived from the raw measurements match on N of N shots; our
  plain matcher agrees with Google's PyMatching but for K ties.

**Live, from raw readouts:**
- Three Willow experiments: one patch at each of d = 3, 5, 7, Z basis, 30 rounds, the first 2,000
  shots of each (so nothing is chosen).
- Ships about 2 MB, credited under CC BY 4.0: `circuit_ideal.stim`, `circuit_noisy_si1000.stim`,
  the raw measurements and sweep bits, `obs_flips_actual`, the five pathways' predictions, and a
  SHA-256 of Google's detection events for those shots.
- In the worker, the engine derives the detection events from the raw measurements and shows that
  their SHA-256 matches Google's, builds the error model from the noisy circuit (or loads Google's
  SI1000 model, shipped instead, if the full run found ours differs, and says so), and decodes with
  both our matchers.
- Per distance: failures for our plain and correlated matchers and for each of Google's decoders on
  the same shots, shot-by-shot agreement with Google's correlated matcher, and µs per shot in
  WebAssembly.
- The panel says that 2,000 shots cannot pin down Λ and that the recorded run does.

**WASM:** exports for `m2d`, correlated decoding and loading a model, and a byte buffer beside the
text buffer. They run in the existing worker through `compute.call`.

## README

A section "Google's hardware data": the priors, the fit and its start round, the bootstrap; both
tables; the checks with their counts; timing; and what the published 2.14 is and is not.

---

## Error handling

- **Engine:** an unknown instruction or target is refused with its line. A model the correlated
  matcher cannot use (an undecomposed error on three or more detectors, or a decomposed piece with
  no detectors) is refused with PyMatching's reason.
- **Run tool:** a failed check, an unmatchable shot or a model mismatch is recorded in the
  experiment's entry and printed by `summary`; the README's numbers wait until it is explained.
- **Page:** a failed fetch or a WASM error shows in the panel's status line, as in Figure 8.

## Risks

1. **The fit may not reproduce Sycamore's published numbers.** Then the protocol follows the paper's
   supplement before anything is quoted.
2. **Our model may differ from Google's SI1000 prior** (float rounding, or a different
   decomposition). Then ours is decoded as its own prior, and the page ships Google's model file.
3. **The full run's cost is an estimate.** If a configuration runs far beyond it, it is measured,
   reported and given more threads; no data is dropped.
4. **Correlated matching may disagree with PyMatching beyond ties.** Layer 4 exists for this, and it
   is fixed before Plan 2 starts.

## Out of scope

- Sycamore's d = 25 repetition code (not a surface code).
- Neural-network decoding.
- Belief-matching and BP (D).
- Real-time decoding and latency (C).
- The IBM gross code (F).

Both plans build on `feature/willow-data`. Nothing reaches `master` without asking.

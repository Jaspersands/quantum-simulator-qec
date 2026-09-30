# K — A fuller resource estimate: implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Section 15 prices real algorithms (RSA-2048, FeMoco) and illustrative sizes with:
- magic-state factories (cultivation or distillation);
- Litinski's floor plans;
- a reaction time from this project's measured decoder latency;
- an idle-storage option (surface, yoked, gross-code modules).

The model is checked against the sources' own worked examples and totals.

**Architecture:** `js/estimator.js` stays pure and tested, and gains the new model beside the old one.
- `epsilonUniform` gives logical error under uniform circuit noise, from a fit to this project's own SD6 simulations.
- `BLOCKS` holds Litinski's data blocks.
- `factoryFor` gives cultivation (Gidney 2025) or distillation (Litinski 2019, Table 1).
- `estimateFull` chooses the smallest distance that meets the budget. Run time is the largest of the Clifford, reaction and factory bounds.

The inputs are read from papers, with page and table references in `data/estimate/*.json`, and every number the page shows is either measured here or read from one of them. `tools/estimate.mjs` prints every preset and the validation, and generates the README and report tables.

**Tech Stack:** vanilla JS modules (site and Node), Python tools with the engine (`stabilizer_qec`), the existing Plot and site-test harness.

## Global constraints

- Inputs are measured by this project or read from a cited paper; the page says which. Sources and page/table references are in `docs/superpowers/notes/2026-09-30-full-estimate-sources.md` and go into `data/estimate/sources.json`.
- A Toffoli is 1 CCZ state where the source uses CCZ (cultivation, Litinski's 8-to-CCZ row), else 4 T states.
- Validation (spec): set to Gidney's assumptions (uniform p = 0.1%, 1 µs cycle, 10 µs reaction, cultivation), the model must land within a factor of 2 of his RSA-2048 physical qubits (897,864) and run time (12.07 h per shot). If it does not, the reason is found and stated on the page and in the README before shipping. Nothing is tuned to hit the target.
- Commit per task, messages ending `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. Build the Python engine with `--profile python`.

## File structure

| File | Responsibility |
|---|---|
| `tools/estimate_noise.py` (create) | SD6 rotated memory at p ∈ {0.1, 0.2, 0.3, 0.5}%, d = 3–11, correlated matching; fit ε = A (p/p_th)^((d+1)/2) → `data/estimate/noise.json`. |
| `tools/distill.py` (create) | 15-to-1 at the logical level: exact enumeration over the 2¹⁵ Z-error patterns, and a circuit sampled by this engine → `data/estimate/distill.json`. |
| `data/estimate/sources.json` (create) | Algorithms, factory tables and reference totals, each with its citation. |
| `js/estimator.js` (modify) | `epsilonUniform`, `BLOCKS`, `factoryFor`, `latencyAt`, `estimateFull`. |
| `tools/site-tests.mjs` (modify) | Unit tests against the sources' worked examples. |
| `tools/estimate.mjs` (modify) | Presets, validation and generated tables (`--json`). |
| `js/sections/estimator.js`, `index.html`, `css/styles.css` (modify) | Section 15 rebuilt. |
| `README.md`, `tools/readme_tables.py`, `report/report.md`, `tools/report.py` (modify) | "What it would take" regenerated. |

---

### Task 1: ε under uniform noise, measured here

- [ ] **Step 1: `tools/estimate_noise.py`.**
  - For p in (0.001, 0.002, 0.003, 0.005) and d in (3, 5, 7, 9, 11): text = `sq.generate_circuit("rotated", d, d, "sd6", p, 0.5, "z")`, sampled by `sq.sample_b8_batch` and decoded by correlated matching (`sq.decode_b8_own(text, dets, n, 0, True)`), as `tools/surgery.py` does.
  - A point stops at 200 failures, 2×10⁷ shots or 900 s. Per-round ε = 1 − (1 − P)^(1/d), with a Wilson interval.
  - Fit log ε = log A + ((d + 1)/2)(log p − log p_th) by weighted least squares over points with ≥ 20 failures. Write `data/estimate/noise.json` with the points, `A`, `pth`, and the fit's residuals.
- [ ] **Step 2:** `export function epsilonUniform(d, p, { A, pth }) { return A * (p / pth) ** ((d + 1) / 2); }` in `js/estimator.js`. Site test: the fit reproduces every fitted point within 3× (log residual < ln 3), and ε falls with d below p_th.
- [ ] **Step 3:** Run the tool (background, about an hour), commit the tool, data and function: `feat(estimate): logical error under uniform noise, fitted to this engine's SD6 simulations`.

### Task 2: sources, blocks, factories

- [ ] **Step 1: `data/estimate/sources.json`**, copied from the notes file with citations:
  - `algorithms`:
    - rsa2048: 1,409 logical qubits (1,280 cold + 129 hot), 6.5e9 Toffolis. Reference: 897,864 qubits, 12.07 h per shot, 6 factories, d = 25. Gidney 2025, Tables 4 and 5 and §costs.
    - femoco_reiher: 2,142 logical qubits, 5.3e9 Toffolis. Reference: ≈4e6 qubits at d = 31, ≈3 days with 4 factories. Lee et al., Table III, Fig. 10 and p. 31.
    - femoco_li: 2,196 logical qubits, 3.2e10 Toffolis. Table III.
    - Three illustrations: 100 × 1e6, 1,000 × 1e9 and 10,000 × 1e12 Toffolis, labelled as illustrations.
  - `litinski_factories`: every row of Table 1 at p = 1e-3 and 1e-4: name, p_phys, p_out, qubits, cycles, qubitcycles per output, and whether its output is a CCZ.
  - `cultivation`: T error 1e-7 at p = 1e-3; 30,000 qubit·rounds per T; CCZ error 28 p_T²; factory 3 × 4 patches; 150 rounds per CCZ at d = 25. Valid for p ≤ 1e-3; Gidney 2025.
  - `yoked`: 430 physical qubits per cold logical at 1e-15 per round; Gidney 2025, at p = 1e-3.
  - `blocks`: compact (1.5n + 3 tiles, 9 steps), intermediate (2n + 4, 5) and fast (2n + √(8n) + 1, 1); Litinski 2019, Sec. 2. Each tile is 2d² physical qubits, as his examples count them.
- [ ] **Step 2: Failing site tests.**
  - `BLOCKS.intermediate.tiles(100) === 204`.
  - `Math.round(BLOCKS.fast.tiles(100)) === 231`.
  - Litinski's intermediate example: (204 + 22) tiles × 2 · 13² ≈ 76,400.
  - `factoryFor('distillation', 1e-3, …, target 1e-10)` picks the cheapest row meeting the target, and returns null above 1e-3.
  - `factoryFor('cultivation', 1e-3, 25, …)` gives 12 · 2 · 26² qubits and 150 rounds per CCZ.
- [ ] **Step 3: Implement** `BLOCKS`, `factoryFor(kind, p, d, target, sources)` and `latencyAt(d, measured, override)`. The latency is the measured p99 window time at d, extrapolated as a power law through the two largest measured d, as `decodingCores` extrapolates cores; or an override in µs.
- [ ] **Step 4: Implement `estimateFull(algorithm, options, sources)`.**
  - **Options:** noise (`{kind: 'uniform', p, fit}` or `{kind: 'measured', model}`), `budget`, `block`, `factory`, `storage` ('surface' | 'yoked' | 'gross'), `cycleSeconds`, `controlSeconds`, `latency` ({d: µs}, or `latencyOverrideUs`), `parallel`.
  - **Split.** hot = `algorithm.hot` when storage is not 'surface', else all. Half the budget goes to magic states, which fixes the target error per state; the factory is chosen for that target.
  - **For each odd d from 3:**
    - steps per Toffoli = block steps × states per Toffoli; Clifford time = steps · d · cycle;
    - reaction = latency(d) + control;
    - time per Toffoli = max(Clifford, reaction) / parallel; total cycles = Toffolis · time per Toffoli / cycle;
    - memory failure = tiles · total cycles · ε(d) + storage failure; the smallest d with total failure ≤ budget.
  - **Factories.** The count keeps up with the consumption rate.
  - **Qubits.** Tiles · 2d², plus factories · factory qubits, plus storage (yoked: cold · 430; gross: ⌈cold / 12⌉ · (288 + 103), with ε per module-cycle from `data/gross/results.json` at the chosen p when measured there, else unavailable).
  - **Returns** the distance; qubits by part; seconds and which bound set them; failure by part; the factory and its count; and decoding cores.
- [ ] **Step 5:** Run the site tests → pass. Commit `feat(estimate): magic-state factories, Litinski's floor plans, reaction time and storage in the model`.

### Task 3: the 15-to-1 law, measured here

- [ ] **Step 1: `tools/distill.py`.**
  - **The circuit:** 15 qubits RX; Z_ERROR(p) on each (twirled faulty T states); MX all. Four detectors, each the parity of one Hamming [15, 11] parity check's support (column j is the binary expansion of j + 1). One observable, the parity of all 15.
  - **Sampling.** `sq.sample_b8_batch` gives the output error given acceptance, and the acceptance rate, at p in (0.001, 0.003, 0.01, 0.03, 0.1).
  - **Exact.** Sum over all 2¹⁵ patterns (accepted when the syndrome is zero; wrong when accepted and of odd weight).
  - **Assertions:** the samples agree with the exact values within their Wilson intervals, the exact output error / (35 p³) → 1 as p → 0, and the acceptance is 1 − 15p + O(p²).
  - Write `data/estimate/distill.json`.
- [ ] **Step 2:** Run and commit `data(estimate): the 15-to-1 law, sampled by this engine and summed exactly`.

### Task 4: validation and generated numbers

- [ ] **Step 1: `tools/estimate.mjs`** (rewritten), for every preset and noise model:
  - the page's default estimate;
  - the validation block: RSA-2048 under Gidney's assumptions (uniform 0.1% by our fit, 1 µs, 10 µs control and decoder latency, cultivation, fast block, yoked cold storage), with qubits and time against his and the ratio; the same with surface storage, and with our measured decoder latency; FeMoco (Reiher) against Lee et al.
  - `--json` for the report; `--markdown` for the README.
- [ ] **Step 2: Decide the stated reasons.** Where a ratio exceeds 2, name the reason measured by switching exactly one option:
  - yoked storage versus surface storage;
  - parallel Toffolis: Gidney's arithmetic runs several per lattice-surgery period;
  - decoder latency.

  Put each in the output text.
- [ ] **Step 3:** Commit `feat(estimate): presets and validation against Gidney's RSA-2048 and Lee et al.'s FeMoco`.

### Task 5: section 15 rebuilt

- [ ] **Controls:**
  - algorithm (the presets and custom);
  - noise (Willow as measured, three decoders, or uniform p ∈ {0.05, 0.1, 0.2}%);
  - magic states (cultivation or distillation);
  - data block;
  - storage;
  - budget, cycle time, control delay, decoder (measured or a latency in µs), and parallel Toffolis.
- [ ] **Outputs:** a table (d, qubits by part, time and its binding bound, failure by part, factories, decoding cores); a stacked bar of qubits (block, factories, storage); a bar of time per Toffoli (Clifford, reaction, factory).
- [ ] **Figure 24:** the validation table (RSA and FeMoco against their sources, ratios and stated reasons) and the presets grid. The 15-to-1 law goes in the note.
- [ ] **Verify:** site tests, contrast, and a headless check at 1440 and 390 px. Commit `feat(estimate): section 15 rebuilt around factories, floor plans and reaction time`.

### Task 6: README and report

- [ ] "What it would take", regenerated from `tools/estimate.mjs`: the model's parts with their sources, the validation and its reasons, the 15-to-1 check, the uniform-noise fit. Report section likewise, from `--json`. `readme_tables.py` checks the tables. Commit.

### Task 7: review, merge, push, CI

- [ ] `/code-review --effort high master..feature/full-estimate`; fix; merge; push; CI; memory.

## Self-review

- **Spec coverage:**
  - K1 (Task 2 sources and presets);
  - K2 both ways and our check (Tasks 2 and 3);
  - K3 (Task 2 blocks; the CNOT law as conservative idle pricing, cited from I);
  - K4 (latency and control in `estimateFull`);
  - K5 (storage options including gross modules);
  - validation (Task 4);
  - outputs (Tasks 5 and 6).
- **Deviation:** the spec's "cultivation figures against p" are, in the sources, only at p = 1e-3 (and 5e-4 for the T error). The model offers cultivation for p ≤ 1e-3, with the 1e-3 numbers, and says so.

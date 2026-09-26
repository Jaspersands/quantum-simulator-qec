# Throughput Implementation Plan (C1)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A bit-parallel frame sampler that runs `REPEAT` blocks without flattening, and a pool of browser workers, so the engine produces and decodes shots many times faster, natively and on the page.

**Architecture:** Plan C1 of sub-project C (spec: `docs/superpowers/specs/2026-09-26-real-time-design.md`, Part 1).
- `src/batch_sampler.rs` compiles a circuit's instruction tree once, with noise probabilities pre-processed for geometric skipping, and runs 64 shots per machine word. Measurement records live in a ring buffer, so `REPEAT` bodies iterate in constant memory, and detectors reach a sink as they are evaluated.
- `js/pool.js` runs N copies of the existing worker and spreads independent Monte Carlo jobs over them.

**Tech Stack:** Rust 2021 (no new crates), PyO3 (`--profile python`), plain ES modules and workers.

## Global Constraints

- `FrameSampler` stays, unchanged, as the reference the batch sampler is checked against.
- The noise semantics are exactly `FrameSampler`'s: disjoint channels, gauge randomisation at resets and measurements, and the same Pauli index mapping for `DEPOLARIZE1` (1 + r mod 3) and `DEPOLARIZE2` (1 + r mod 15, low two bits on the first qubit).
- Every speed number is measured, and quoted with the machine.
- Tests: `cargo test --release --no-default-features`; `cargo build --release` and the wasm32 build compile; `node tools/site-tests.mjs`; the quick cross-check passes.
- Commit per task on `feature/real-time` with the trailer. Merge into `master` only once C1 and C2 are complete and verified.

---

### Task 1: The batch sampler

**Files:** Create `src/batch_sampler.rs`; add `pub mod batch_sampler;` to `src/lib.rs`.

**Interfaces:**
- `BatchSampler::new(&Circuit) -> Result<BatchSampler, String>`;
- fields `num_qubits`, `num_detectors`, `num_observables`, `num_measurements` (all `usize`, counted through repeats);
- `run(&self, rng: &mut Xorshift, sink: &mut dyn FnMut(usize, u64)) -> [u64; 64]`: each detector's word is sent to `sink(index, word)` as it is evaluated, and the observable words are returned, where word k is observable k;
- `sample(&self, rng) -> Batch { detectors: Vec<u64>, observables: Vec<u64> }`;
- `Batch::lane_defects(&self, lane) -> Vec<u32>`;
- `Batch::lane_observables(&self, lane) -> u64`.

**Semantics** (each op over whole words):
- `H` swaps x and z.
- `CX`: `x[t] ^= x[c]; z[c] ^= z[t]`.
- `CZ`: `z[b] ^= x[a]; z[a] ^= x[b]`.
- Resets: Z sets x = 0 and z = a random word; X sets z = 0 and x = a random word.
- Measurements:
  1. record `x` (Z basis) or `z` (X basis), XOR a Bernoulli(flip) word;
  2. set the other component to a random word;
  3. then reset if it's `MR`.
- `X_ERROR`, `Y_ERROR`, `Z_ERROR`: XOR a Bernoulli(p) word into x, both, or z.
- `DEPOLARIZE1`: a Bernoulli(p) word, then for each set lane the Pauli `1 + next_u64() % 3`.
- `DEPOLARIZE2`: likewise, with `r = 1 + next_u64() % 15`; `r & 3` goes on the first qubit and `r >> 2` on the second.
- `PAULI_CHANNEL_1`: a Bernoulli(px + py + pz) word; per set lane, `u = next_f64() · total` picks X below px, Y below px + py, otherwise Z.
- Pauli gates and `SweepX` do nothing to the frame.
- `DETECTOR`: the XOR of the ring entries at its lookbacks. `OBSERVABLE_INCLUDE`: XOR into that observable's word.
- `Repeat` iterates its compiled body.
- **Bernoulli words:** p ≤ 0 gives 0 and p ≥ 1 gives !0. For p < 1/4, geometric skipping with the precomputed `ln(1 − p)`: position `i += 1 + floor(ln(1 − u) / ln(1 − p))`, while `i < 64`. Otherwise 64 comparisons.

**Tests** (in the module):
- `matches_frame_sampler_on_deterministic_noise`. Generated SD6 memory circuits (rotated and XZZX, Z and X basis, d = 3, 5) have every noise line rewritten to `X_ERROR(b)`, `Z_ERROR(b)` or `Y_ERROR(b)` on the same qubits (pairs flattened), and `M(p)` to `M(b)`, with b = 1 on about 3% of lines. Across 20 random rewrites per circuit, every lane of a batch equals `FrameSampler::sample`'s shot, detectors and observables.
- `repeat_blocks_run_without_flattening`. For Stim's d = 3 circuit from `data/xcheck/` (which has a `REPEAT` block) and a nested-`REPEAT` circuit, the sampler on the circuit and the sampler on `circuit.flattened()`, with the same seed, give identical batches.
- `marginals_match_the_error_model`. The `frame_sampler` marginals test, on `REP3` at p = 0.05 and on a rotated SD6 d = 3 circuit at p = 0.01, with 200,000 shots (3,125 batches): every detector within 5σ of `(1 − Π(1 − 2p))/2`, and observable 0 too.
- `noiseless_circuits_never_fire`.
- `bernoulli_words_have_the_right_rate`. Over 200,000 words at p = 0.001, 0.1, 0.3 and 0.9, the set fraction is within 5σ of p, and bit positions are uniform (each lane within 5σ).
- `batch_timing` (ignored): µs per shot for `FrameSampler` and the batch sampler, on rotated SD6 at d = 3, 5, 7, 9 with T = d.

- [ ] Step 1: Write the tests (failing).
- [ ] Step 2: Implement.
- [ ] Step 3: Run `cargo test --release --no-default-features batch`, then everything, then `batch_timing -- --ignored --nocapture`.
- [ ] Step 4: Commit `feat(sampler): a bit-parallel frame sampler, 64 shots a word, REPEAT without flattening`.

### Task 2: The batch sampler in the harness

**Files:** `src/py_api.rs`, `tools/xcheck.py`.
- `sq.sample_b8_batch(circuit_text, num_shots, seed, threads=1) -> (dets b8, obs b8)`. Batches are transposed into b8 rows. Threads each take a contiguous range of batches, with seeds `seed + thread`.
- xcheck check 3 gains the batch sampler: a χ² z-score of per-detector rates against Stim's, and its logical error rate under our decoder, in the same Wilson-overlap test. Check 4 reports sampling µs per shot for Stim, `FrameSampler` and the batch sampler.
- [ ] Implement, run `xcheck.py --quick`, commit `feat(xcheck): the batch sampler against Stim's`.

### Task 3: The batch sampler in the browser

**Files:** `src/wasm_xc.rs`.
- `wasm_xc_run` samples with `BatchSampler` in batches of 64, and decodes the lanes it needs (the last batch is partial). `decode == 0` still times sampling alone.
- The SD6 sweep's statistics are unchanged, since the sampler differs in speed but not in distribution: verified with `node tools/sweep.mjs 3 0 2`, whose fit must fall within the recorded interval.
- [ ] Implement, rebuild the WASM, run the Node sweep and the headless Figure 8, commit `feat(wasm): the general path samples 64 shots a word`.

### Task 4: The worker pool

**Files:** Create `js/pool.js`. Modify `js/compute.js` (a `Compute` can front a pool), `js/main.js`, `js/sections/threshold.js` (the sweep), `js/sections/bench.js` (the stream), `js/sections/xcheck.js` (the decoding rows) and `js/sections/hardware.js` (the three distances). `tools/site-tests.mjs` gains tests of the pure combining functions (`js/pool-merge.js`).
- `Pool(size)` holds `size` `Compute` instances (workers), with:
  - `call(op, payload, onProgress)` on the least-busy worker;
  - `map(op, payloads, onProgress)` spreading jobs FIFO over free workers and resolving to results in input order;
  - `cancel(promise)` routed to the worker running it;
  - `size`.
- Size: `Math.max(1, Math.min(8, (navigator.hardwareConcurrency || 2) - 1))`, leaving the main thread a core.
- Sweep: the sweep op already takes a list of distances × ps. The page splits it by point into `map` jobs and merges results in order, and progress sums across jobs.
- Bench stream: split the requested runs into `size` streams; progress combines counts (`mergeStream`); cancel cancels all.
- The vitals line reports the pool size.
- [ ] Implement, add the site tests, run the headless checks of Figures 6 (sweep), 8, 9 and 10 and the bench, commit `feat(site): a worker pool; sweeps and the bench use every core`.

### Task 5: README, verification, review
- README: a section "Throughput" covering the batch sampler (design, five checks, speed table against `FrameSampler` and Stim) and the pool (sizes and measured speed-ups); Repository Structure.
- Full verification as in B2's Task 9, then the `code-review` skill at `high` over C1's commits. Fix, re-verify, commit.

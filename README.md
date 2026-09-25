# Quantum Error Correction (QEC) Simulator

A Rust stabilizer circuit simulator and decoder for rotated surface codes and XZZX codes.
Compiles to WebAssembly for an interactive browser explainer, and to PyO3 Python bindings
(`stabilizer_qec`) for Monte Carlo threshold benchmarking. It reads and writes Stim's circuit and
error-model formats, and it agrees with Stim and PyMatching on every check in
[Checked against Stim and PyMatching](#checked-against-stim-and-pymatching).

The website walks through surface-code error correction in order: errors, syndromes, decoding,
spacetime, threshold. Each interactive figure is driven by the real engine running locally, and
every number on the page is computed in the reader's browser on load.

## Key Features

- **Stabilizer engine**: binary symplectic tableau simulator, $O(N^2)$ Pauli propagation,
  based on the Gottesman–Knill theorem.
- **Codes**: rotated surface code ($d^2$ data qubits, $d^2-1$ ancillas) and XZZX surface code.
- **Noise models**: pure data noise, phenomenological noise, circuit-level noise (the engine's
  own model, and SD6, the standard one), $Z$-bias, located erasure, spatial bursts, and slow
  temporal drift.
- **Decoders**: disjoint-set Union-Find cluster peeling, exact minimum-weight perfect matching,
  a greedy nearest-neighbour baseline, and probability-weighted exact matching over any detector
  error model by sparse blossom, within 2.3 to 2.9 times PyMatching's speed single-threaded.
- **A general circuit path**: circuits and detector error models in Stim's text formats, a
  detector error model built by walking any circuit backwards, a Pauli-frame sampler, and Stim's
  `01`/`b8` shot formats. Checked against Stim and PyMatching, edge for edge.
- **Web explainer**: `index.html` plus `css/` and `js/`. No build step, no dependencies. The lattice
  at the top of the page runs the engine live, every figure is driven by it, the threshold table is
  plotted as it is measured, and the bench streams its estimate.
- **Python extension** (`stabilizer_qec.so`): PyO3 bindings for offline threshold benchmarking.

## A note on quoted figures

Unless stated otherwise, the logical error rates below were measured at bias `η = 0.5` (equal
parts X, Y and Z, which is the setting every panel on the site starts from) over `T = d` rounds.
The bias matters more than it might seem. At `d = 7, p = 0.3%` circuit-level, the rotated code
reports **1.07% at η = 1 and 2.37% at η = 100**, while XZZX at the same two settings reports
**1.08% and 0.25%**. Same p, and the ranking between the codes reverses. Shot counts are given
where a number is close enough to the noise floor for it to matter.

## Checked against Stim and PyMatching

Stim and PyMatching are the tools the field uses to simulate and decode surface codes, and the
first question anyone asks of a new simulator is whether it agrees with them. Until now this
engine had only been checked against itself. It now has a general circuit path, and on that path
it agrees with both tools on every check below, down to floating-point rounding.

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
| 3 | 0.3% | 2.289% | 2.289% | 2.289% | 0 (0) | 2.292% | -0.10 | 0.2 µs | 0.6 µs |
| 3 | 0.6% | 7.603% | 7.603% | 7.603% | 0 (0) | 7.624% | -1.38 | 0.4 µs | 1.3 µs |
| 5 | 0.3% | 1.647% | 1.649% | 1.649% | 2 (0) | 1.631% | -2.38 | 1.8 µs | 4.7 µs |
| 5 | 0.6% | 9.432% | 9.432% | 9.432% | 4 (0) | 9.493% | -0.90 | 5.1 µs | 11.5 µs |
| 7 | 0.3% | 1.124% | 1.124% | 1.124% | 0 (0) | 1.073% | +0.06 | 6.1 µs | 15.6 µs |
| 7 | 0.6% | 10.653% | 10.652% | 10.652% | 7 (0) | 10.686% | -2.35 | 16.0 µs | 40.8 µs |

The last two columns are native decode time per shot.

**4. Speed.** Single-threaded, this engine takes 2.3 to 2.9 times PyMatching's time per
shot. At d = 7, p = 0.6% it takes 41 µs a shot, native, against PyMatching's
16 µs. The first version of this check used the dense matcher, which was 18 to
136 times slower and fell further behind as the patch grew. The sparse matcher closed that gap. It
also decodes shots in parallel, one workspace per thread: across every core of the recording machine,
d = 7, p = 0.6% runs at 5.6 µs a shot.

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
How much is a question for the next decoder work.

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
VIRTUAL_ENV=$PWD/.venv CARGO_TARGET_DIR=target-py .venv/bin/maturin develop --release
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

**Speed.** Native, single-threaded, rotated code under SD6 with T = d, per shot:

| d | p | dense | sparse | PyMatching |
|---|---|---|---|---|
| 3 | 0.3% | 6.0 µs | 1.0 µs | 0.2 µs |
| 3 | 0.6% | 13.4 µs | 2.1 µs | 0.4 µs |
| 5 | 0.3% | 75.5 µs | 7.6 µs | 1.8 µs |
| 5 | 0.6% | 210 µs | 15.8 µs | 5.1 µs |
| 7 | 0.3% | 656 µs | 23.0 µs | 6.1 µs |
| 7 | 0.6% | 2.58 ms | 55.5 µs | 16.0 µs |
| 9 | 0.3% | 4.42 ms | 53.4 µs | — |
| 9 | 0.6% | 20.5 ms | 137 µs | — |

The dense and sparse columns come from `cargo test --release --no-default-features sparse::tests::timing
-- --ignored --nocapture`. PyMatching's column is from the cross-check's recorded run.

The site uses the same matcher in WebAssembly. A whole SD6 sweep window, one shot at every point
across d = 3, 5, 7 and 9, now costs about 1.7 ms instead of more than 100 ms. So the SD6 sweep went
from 800 shots a point to 40,000, and Figure 8's live rates now run 50,000 to 100,000 shots.

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
src/surface_code.rs   rotated and XZZX lattices, noise models, error generation
src/decoder.rs        Union-Find cluster growth and peeling
src/circuit_model.rs  detector error model derived from the extraction circuit, CSS and non-CSS
src/circuit.rs        circuits in Stim's text format: parse, emit, flatten, resolve records
src/dem.rs            detector error models: built backwards from any circuit, decomposed as
                      Stim decomposes, read and written in Stim's .dem format, compared
src/dem_decoder.rs    probability-weighted exact matching over any detector error model
src/frame_sampler.rs  Pauli-frame sampling of any circuit, independent of the model
src/shots.rs          Stim's 01 and b8 detection-event formats
src/memory.rs         rotated and XZZX memory experiments as circuits, engine noise and SD6
src/equivalence.rs    test: the old per-code circuit and the general one are the same circuit
src/sparse/           sparse blossom: exact matching by growing regions on the detector graph
src/py_api.rs         PyO3 bindings for the cross-check
src/wasm_xc.rs        WASM exports for Figure 8 and SD6
src/lib.rs            PyO3 module and the WASM C-ABI interface

index.html            the explainer (structure only)
css/styles.css        design tokens, then base, then components
js/engine.js          typed wrapper over the WASM exports
js/channel.js         the noise channel, shared by the figures and the engine
js/channel-view.js    the logical channel drawn as the Bloch sphere's image
js/worker.js          Monte Carlo worker (own engine instance)
js/compute.js         worker RPC, Wilson intervals, threshold collapse fit
js/lattice.js         canvas renderer for a code patch, 2D and spacetime
js/plot.js            canvas plotting primitive
js/sections/*.js      one module per section of the page
js/sweep-config.js    what the threshold sweep measures, shared with tools/sweep.mjs
js/xcheck-format.js   pure formatting for Figure 8

tools/xcheck.py         the cross-check against Stim and PyMatching; writes data/xcheck/
tools/sweep.mjs         repeated threshold sweeps in Node, for the figures quoted here
data/xcheck/            Stim's circuits and models, the recorded reference, and the run's report

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
VIRTUAL_ENV=$PWD/.venv CARGO_TARGET_DIR=target-py .venv/bin/maturin develop --release
.venv/bin/python run_benchmarks.py
.venv/bin/python run_data_benchmarks.py
.venv/bin/python tools/xcheck.py
```

If an older `stabilizer_qec.so` sits at the repository root (git ignores it), delete it. Python looks
in a script's own directory first, so for the two benchmark scripts it would shadow the fresh build.
`tools/xcheck.py` refuses to run against a build that lacks the cross-check bindings.

## License

MIT

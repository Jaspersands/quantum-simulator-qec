# Site Rework — Guided Explainer

**Date:** 2026-08-11
**Status:** Approved

## Problem

`index.html` is 1,736 lines containing seven independent tools stacked vertically
with no navigation and no argument connecting them. Specific failures:

1. **No spine.** Each section assumes the reader already knows why it matters.
   The visualizer, benchmark panel, Bloch sphere, and threshold fitter have no
   stated relationship to each other.
2. **Two visual languages.** The page is built in a hard-bordered cream
   editorial style, but the threshold-fitting section uses GitHub-blue and green
   rounded buttons (`#0969da`, `#22c55e`, `border-radius: 4px`). It was
   evidently bolted on later. There is no token system, so there was no
   vocabulary to reach for.
3. **Everything is an inline style.** ~200 `style="..."` attributes. No reuse.
4. **Dishonest data.** Four metric cards hardcode numbers (`75,700 runs/s`). A
   stats table hardcodes logical error rates. Two static PNGs
   (`threshold_plot.png`, `data_threshold_plot.png`) duplicate what the live
   fitter already computes.
5. **The stats table reads as broken.** It shows logical error rate *rising*
   with distance (5% → 11.9% → 18.1% at p=1% phenomenological). That is correct
   behaviour above threshold, but the table never labels the regime, so it looks
   like a bug in the simulator.
6. **Heavy sweeps freeze the page.** `wasm_run_benchmark` runs on the main
   thread behind a `setTimeout(…, 50)`.

## Toolchain

The first attempt at `cargo` failed with `bad CPU type in executable` — the toolchain is
`stable-x86_64-apple-darwin` on an M2 Pro, and Rosetta was not reachable from the shell at that
moment. It became available later in the session, at which point the engine could be rebuilt and
two of the three defects below were fixed at the source. `softwareupdate --install-rosetta` is the
quick fix; a native `aarch64-apple-darwin` toolchain is the better one.

## Seven engine defects found during the rework

All surfaced while replacing fabricated numbers with live ones. Six are fixed at the source and the
module rebuilt; one gap remains.

| # | Defect | Status |
|---|---|---|
| 1 | Batch Monte Carlo seeded from a constant inside every shot | FIXED — SplitMix64 over a counter, `wasm_seed` export |
| 2 | Phenomenological noise applied a faulty readout in the final round | FIXED — final round noiseless |
| 3 | `Tableau::new` hardcoded its RNG state, so circuit runs were deterministic | FIXED — `Tableau::with_seed` |
| 4 | X and Z extraction circuits used the same CNOT order, so they did not commute | FIXED — N/Z schedules, by direction |
| 5 | Tomography ran three identical sims and called the result a Bloch vector | FIXED — Pauli transfer matrix diagonal |
| 6 | XZZX matched one defect set twice, on two different graphs | FIXED — one combined graph, one matching |
| 7 | XZZX scored logical errors against a hard-coded string that was not a logical operator | FIXED — stabilizer-group membership |
| 8 | The decoder had no model of the circuit, so hook errors were mis-paired | FIXED — detector error model in `src/circuit_model.rs` |

Headline verifications: batch variance matches binomial (σ 0.00224 against 0.00222 over twelve
repeats); a noiseless circuit fails never at any distance; zero-noise tomography returns (1, 1, 1)
meaning *the channel shrinks nothing*; no single X, Z or Y error causes an XZZX logical failure at
d = 3, 5 or 7; XZZX beats the rotated code under bias (d=7, p=3%, η=64: 0.07% against 0.43%); and
**every single circuit fault is corrected — 0 failures out of 600 / 3,240 / 9,408 at d = 3 / 5 / 7**,
for Union-Find and exact MWPM alike.

Thresholds now land where the literature puts them for all three noise models: ~11% data noise,
~2.7% phenomenological, and 0.45% circuit-level (ν 1.48, reduced χ² 1.05).

### On the detector error model

The circuit is defined once — `round_program` — and both the stabilizer simulation and the fault
propagation consume it, so the model cannot drift from the circuit it describes. The decomposition
that usually makes this hard is free: the decoder already matches once per Pauli type, so a fault
splits into its X and Z parts and each is graph-like alone. No fault in the whole enumeration fires
more than two detectors in either graph.

The exhaustive single-fault check is what made it possible to get right. It is deterministic and
complete, and it caught two things sampling would have missed or misattributed: the CNOT schedules
had to be transposes *the other way round* from the pairing first tried (which passes at d = 5 and
d = 7 and fails 44/600 at d = 3), and state-preparation noise was being applied before the baseline
round, where it sits in both readings a detector compares and so is never detected at all.

Not modelled: located erasure under circuit-level noise, which needs its own fault family and
per-shot edge reweighting.

## Decisions

| Question | Decision |
|---|---|
| What the site is | A guided explainer you can play with |
| Aesthetic | Keep the warm-paper editorial look, applied consistently via tokens |
| New functionality | Only: replace fabricated data with live-computed data |
| Layout | One continuous scroll with sticky section nav + scroll-spy |
| Load behaviour | Progressive — page renders instantly, numbers fill in from a worker |

## Narrative spine

Each existing tool is placed at the point in the argument where it explains
something. Nothing is discarded.

| § | Beat | Carried by |
|---|---|---|
| 1 | You cannot copy a qubit and cannot look at one | Prose |
| 2 | Measure *parity*, not state | New pure-JS single-plaquette demo |
| 3 | Tile it — anatomy of the rotated surface code | Lattice renderer, read-only, hover a stabilizer to light its support |
| 4 | Errors light up defects at the **endpoints of a chain** | Lattice + click-to-inject |
| 5 | **The syndrome does not determine the error** | Lattice + decode, rendering error ⊕ correction |
| 6 | So decode: Union-Find / Greedy / MWPM | Decoder panel |
| 7 | Measurements lie too → repeat *d* rounds, decode in spacetime | Lattice in spacetime mode |
| 8 | Does it work? Below threshold more qubits help; above it they hurt | Live sweep + fit + live results table |
| 9 | Real noise is Z-biased → XZZX | Bias slider + code select |
| 10 | The bench — run your own experiment | Full parameter console + tomography |
| 11 | Under the hood | Engine internals |

§5 is the conceptual crux the current site never states. It is built from
existing exports: inject a chain, let the decoder produce its own chain, render
the residual `error ⊕ correction`. A closed loop is success; a chain spanning
the lattice is a logical failure. `wasm_decode` already returns that verdict.

§8 reframes the embarrassing table as the payoff: the crossover *is* the
threshold.

## Architecture

```
index.html            structure only, zero inline style
css/styles.css        design tokens first, then base, then components
js/engine.js          typed wrapper over the exports; the only module that
                      touches raw pointers or wasm memory
js/channel.js         the noise channel, shared by the figures and the engine
js/channel-view.js    the logical channel drawn as the Bloch sphere's image
js/worker.js          worker-side: own wasm instance, sweeps and tables
js/compute.js         promise-based RPC client for the worker; Wilson intervals;
                      threshold scaling fit
js/lattice.js         one canvas renderer — 2D and spacetime — used by §3,4,5,6
js/plot.js            one plotting primitive — axes, series, error bars
js/patterns.js        error patterns (straight chains, scatter) for the figures
js/dom.js             small DOM helpers
js/nav.js             sticky nav + scroll-spy
js/sections/*.js      vitals, parity, anatomy, syndrome, decode, spacetime,
                      threshold, bias, bench
js/main.js            wiring
```

Two load-bearing pieces:

- **The token system** is what actually removes the disjointed feeling. Every
  colour, border, radius, shadow, and type size is defined once. Stray widgets
  become impossible because there is a vocabulary to reach for.
- **The worker** removes the freeze. The `.wasm` has no imports
  (`WebAssembly.instantiate(buffer, {})`), so a second instance in a worker is
  trivial. The main thread keeps an instance for interactive lattice work;
  the worker owns all Monte Carlo.

ES modules require HTTP, not `file://`. The `.wasm` fetch already required a
server, and GitHub Pages serves over HTTP. No build step, no dependencies.

## Engine API (fixed)

```
wasm_create_session(d, code_type) -> ptr
wasm_free_session(ptr)
wasm_set_num_rounds(ptr, T)
wasm_get_data_qubit_count(ptr) / _x(ptr,i) / _y(ptr,i)
wasm_get_stabilizer_count(ptr) / _x / _y / _type(ptr,i)
wasm_get_physical_x_ptr / _z_ptr / _erased_ptr(ptr)   -> u8[num_data * T]
wasm_get_syndrome(ptr)                                -> u8[num_stab * T]
wasm_get_correction_x_ptr / _z_ptr(ptr)               -> u8[num_data * T]
wasm_toggle_error(ptr, q, type, t)      type: 0=X 1=Z
wasm_toggle_erasure(ptr, q, t)
wasm_toggle_measurement_error(ptr, s, t)
wasm_clear_errors(ptr)
wasm_decode(ptr, decoder) -> u8          1 = logical failure
wasm_run_benchmark(d, code, decoder, p, bias, T, runs, noise, erasure, corr) -> f64
wasm_estimate_logical_fidelity(d, code, decoder, p, bias, noise, T, runs,
                               erasure, corr) -> *f64 (3 values)
```

Enumerations: `code_type` 0=Rotated 1=XZZX · `decoder` 0=Union-Find 1=Greedy
2=Exact MWPM · `noise_mode` 0=Data 1=Phenomenological 2=Circuit-level ·
`stabilizer type` 0=Z 1=X 2=XZZX · `correlated` 0=None 1=Bursts 2=Drift 3=Both.

Note: the current Bloch-sphere call passes `noise_mode = 2` (circuit-level)
hardcoded while its UI implies phenomenological. The rework exposes `noise_mode`
as a real control instead.

## Honest data

- **Vitals cards** — throughput measured on load with a real timed run, not
  literals.
- **Results table** — computed live per cell via the worker, showing a Wilson
  95% interval and the run count. Cells label whether they sit below or above
  the fitted threshold.
- **Static PNGs** — `threshold_plot.png` and `data_threshold_plot.png` deleted;
  the live plot replaces them.
- Every displayed number must be traceable to a run performed in the browser.

## Design tokens

Warm paper and ink, with accents muted enough to sit on cream:

```
--paper #f6f5f0   --surface #fff   --surface-sunk #faf9f6
--ink #1c1d1f     --ink-2 #4a4c50  --ink-3 #77797d
--rule #1c1d1f    --rule-soft #ddd9cf
--x #2f5fd0 (X)   --z #c0392b (Z)  --y #7b4bb5 (Y)
--defect #d68910  --ok #2c7a4b     --fail #b23a2e
```

Type: Lora (display), Inter (body), JetBrains Mono (data). Fonts move from a
render-blocking `@import` to `<link rel=preconnect>` + `<link rel=stylesheet>`.

One button vocabulary (`.btn`, `.btn--primary`, `.btn--ghost`) and one control
vocabulary (`.field`, `.field__label`). The GitHub-blue and green widgets are
removed.

## Out of scope

- Rebuilding the Rust/WASM engine (toolchain is broken; see Constraint)
- New decoders, codes, or noise models
- A threshold-collapse explorer, decoder head-to-head, or scripted story mode —
  considered and explicitly declined
- Any build step, bundler, or dependency

## Verification

- Page renders with no console errors; all sections reachable from the nav
- Every interactive control from the old page still functions
- No literal simulation numbers remain in the markup
- Heavy sweeps leave the main thread responsive
- Layout holds at 375 px, 768 px, and 1440 px

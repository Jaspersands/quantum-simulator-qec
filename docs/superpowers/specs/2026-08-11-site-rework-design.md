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

## Constraint

`cargo` and `rustup` on this machine are x86-64 binaries and fail with
`bad CPU type in executable` on ARM. **`stabilizer_qec.wasm` cannot be rebuilt.**
Its 25 `extern "C"` exports are fixed. All work is HTML/CSS/JS against the
existing engine. No new simulation capability is in scope.

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
js/engine.js          typed wrapper over all 25 exports; the only module that
                      touches raw pointers or wasm memory
js/worker.js          worker-side: own wasm instance, Monte Carlo + sweeps
js/compute.js         promise-based RPC client for the worker; Wilson intervals;
                      threshold scaling fit
js/lattice.js         one canvas renderer — 2D and spacetime — used by §3,4,5,7
js/plot.js            one plotting primitive — axes, series, error bars
js/nav.js             sticky nav + scroll-spy
js/sections/*.js      parity, anatomy, syndrome, degeneracy, spacetime,
                      threshold, bias, bench, vitals
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

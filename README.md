# Quantum Error Correction (QEC) Simulator

A Rust stabilizer circuit simulator and decoder for rotated surface codes and XZZX codes.
Compiles to WebAssembly for an interactive browser explainer, and to PyO3 Python bindings
(`stabilizer_qec`) for Monte Carlo threshold benchmarking.

The website is a guided explanation of surface-code error correction — errors, syndromes,
decoding, spacetime, threshold — where each interactive figure is driven by the real engine
running locally. Every number on the page is computed in the reader's browser on load.

## Key Features

- **Stabilizer engine**: binary symplectic tableau simulator, $O(N^2)$ Pauli propagation,
  based on the Gottesman–Knill theorem.
- **Codes**: rotated surface code ($d^2$ data qubits, $d^2-1$ ancillas) and XZZX surface code.
- **Noise models**: pure data noise, phenomenological noise, circuit-level noise, $Z$-bias,
  located erasure, spatial bursts, and slow temporal drift.
- **Decoders**: disjoint-set Union-Find cluster peeling, exact minimum-weight perfect matching,
  and a greedy nearest-neighbour baseline.
- **Web explainer**: `index.html` plus `css/` and `js/` — no build step, no dependencies.
- **Python extension** (`stabilizer_qec.so`): PyO3 bindings for offline threshold benchmarking.

## Known engine defects

Two bugs in the compiled WebAssembly module were found while making the site report live data.
Both need a rebuild to fix; the website works around the first and reports the second.

### The WASM build's Monte Carlo is not random

In `src/surface_code.rs`, six `simulate_*` functions seed their generator from a constant on the
non-Python cfg branch:

```rust
#[cfg(feature = "python")]
let mut rng = Xorshift::new(rand::random());
#[cfg(not(feature = "python"))]
let mut rng = Xorshift::new(12345);
```

The seed is taken fresh inside each shot, so every shot of a `wasm_run_benchmark` batch is
identical and the returned rate collapses to a step function — exactly 0 below a cutoff, exactly
1 above it. `wasm_run_benchmark` and `wasm_estimate_logical_fidelity` are therefore unusable from
the browser. **The Python build is unaffected**, since it takes the `rand::random()` branch.

The site works around this in `js/montecarlo.js`: noise is sampled in JavaScript and applied
through the per-shot session API, so the lattice, syndrome extraction, decoders, and
logical-failure verdict all still come from Rust. The resulting thresholds match the literature
(~10.5% data noise, ~2.6% phenomenological). Logical-state tomography and circuit-level noise
cannot be reached through that API and are absent from the site until the module is rebuilt.

**Fix**: replace the constant with a seed threaded in from the caller, or expose a
`wasm_seed(u64)` entry point, and rebuild.

### The XZZX decoder reuses the wrong defect set

In the XZZX branch of `wasm_decode` (`src/lib.rs`), the second decoding pass does not derive its
own defects:

```rust
let graph_x = code.build_syndrome_graph(num_rounds, false);
let defects_x = defects_z.clone();     // <-- the Z pass's defects, on the X graph
```

`graph_x` has a different `edge_to_qubit` mapping, so the `correction_z` it produces bears no
relation to any Z error that occurred. Injecting a single X error on a d=3 XZZX patch returns the
correct one-qubit X correction plus **two spurious Z corrections**, each a fresh error on the patch.
The number of them scales with the graph, so the code degrades as it grows: measured at $p = 2\%$
unbiased over 4,000 shots, the rotated code improves from 0.65% at $d=3$ to 0.18% at $d=5$ while
XZZX degrades from 5.3% to 12.8%. The bias response is correspondingly flat across
$\eta = 0.5$ to $\eta = 1000$.

The rotated branch derives `defects_x` and `defects_z` independently and is unaffected, as is the
lattice geometry — the figures in sections 3 and 5 draw XZZX correctly.

**Fix**: build `defects_x` from the X-stabilizer outcomes the way the rotated branch does, and
rebuild.

## Repository Structure

```
src/tableau.rs        symplectic tableau and Clifford operations
src/surface_code.rs   rotated and XZZX lattices, noise models, error generation
src/decoder.rs        Union-Find cluster growth and peeling
src/lib.rs            PyO3 module and the WASM C-ABI interface

index.html            the explainer — structure only
css/styles.css        design tokens, then base, then components
js/engine.js          typed wrapper over the WASM exports
js/montecarlo.js      JS noise sampling driving the engine's decoders
js/worker.js          Monte Carlo worker (own engine instance)
js/compute.js         worker RPC, Wilson intervals, threshold collapse fit
js/lattice.js         canvas renderer for a code patch, 2D and spacetime
js/plot.js            canvas plotting primitive
js/sections/*.js      one module per section of the page

run_benchmarks.py       phenomenological threshold benchmarks
run_data_benchmarks.py  data-noise threshold benchmarks
```

## Building & Running

### The website

It is a static site with no build step, but it does need to be served over HTTP — ES modules and
the `.wasm` fetch both fail from `file://`.

```bash
python3 -m http.server 8080
```

Then open `http://localhost:8080`.

### Rust library

```bash
cargo build --release
```

### Rebuilding the WASM module

```bash
cargo build --release --target wasm32-unknown-unknown --no-default-features
```

Copy the resulting `target/wasm32-unknown-unknown/release/stabilizer_qec.wasm` to the repository
root. Fixing the seeding defect above before rebuilding would let the site drop its JavaScript
sampling workaround and restore the tomography panel.

### Python bindings & benchmarks

```bash
pip install maturin
maturin develop --release
python3 run_benchmarks.py
python3 run_data_benchmarks.py
```

## License

MIT

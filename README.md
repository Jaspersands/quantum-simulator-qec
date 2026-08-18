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

## Engine defects found and fixed

Six bugs surfaced while making the site report live data. Five are fixed; one XZZX issue and one
circuit-level gap remain, both characterised below. Everything here is reflected in `src/` and in
the committed `stabilizer_qec.wasm`.

### 1. FIXED — the WASM Monte Carlo was not random

Six `simulate_*` functions seeded from a constant on the non-Python cfg branch
(`Xorshift::new(12345)`, and `54321` in two). Each of those functions *is* one shot and
`wasm_run_benchmark` loops over them, so every shot in a batch was bit-identical and the reported
rate collapsed to a step function — exactly 0 below a cutoff, a plateau, exactly 1 above.

Fixed with SplitMix64 over a counter, plus a `wasm_seed(lo, hi)` export seeded from
`crypto.getRandomValues`. Seeding each shot from the previous shot's xorshift state is *not*
enough — that hands shot N+1 shot N's stream offset by one draw, and batch variance comes out
several times binomial. Verified: twelve repeats at d=5, p=5%, N=4000 give observed σ 0.00224
against binomial 0.00222 (ratio 1.01).

### 2. FIXED — phenomenological noise lied in the final round

Defects are time differences, so a last-round lie has no partner and leaves an unpaired defect the
decoder must match somewhere. More checks means more last-round lies, so logical error *rose* with
distance far below threshold (1.9% at d=3 to 10.9% at d=7, at p=0.5%). The final round is now
noiseless.

Thresholds after 1 and 2: ~11% data noise, ~2.7% phenomenological, both with reduced χ² near 1,
agreeing point-by-point with an independent JavaScript Monte Carlo written against the per-shot
session API.

### 3. FIXED — the stabilizer tableau replayed identical measurements

`Tableau::new` hardcoded `rng_state: 0xdeadbeef12345678`, so every `StabilizerSimulator` ever
constructed drew the same sequence of random measurement outcomes. Circuit-level runs were entirely
deterministic — a *noiseless* circuit reported exactly 0% or exactly 100% logical error depending
only on the distance. `Tableau::with_seed` now threads a seed from the shot's own generator. This
one also affected the Python build.

### 4. FIXED — the two extraction circuits did not commute

X and Z ancillas walked their plaquettes in the same order, and boundary plaquettes compressed
their two CNOTs into the first two time slots. The X and Z stabilizer measurements therefore
interfered where plaquettes overlap, and each disturbed the other. Ancillas now interleave in
opposite orders — the classic N and Z schedules — scheduled by *direction* rather than by index
into a variable-length neighbour list. A noiseless circuit now fails **never**, at d = 3, 5 and 7,
and the curve rises monotonically with p.

### 5. FIXED — logical tomography returned something outside the Bloch sphere

`wasm_estimate_logical_fidelity` ran three *identical* simulations — for data and phenomenological
noise the three calls differed in nothing but the variable each was assigned to — and reported
`1 - 2*failure_rate` for each as though they were Bloch components. At zero noise that gave
`(1, 1, 1)`, a "state" of length √3.

Under Pauli noise and a Pauli decoder the logical channel is itself a Pauli channel, which shrinks
each Bloch axis independently. The simulators now return which logical Pauli class survived rather
than a bare pass/fail, so the four channel probabilities can be counted and the diagonal of the
Pauli transfer matrix computed properly. Verified: `(1, 1, 1)` at zero noise now correctly means
*the channel shrinks nothing*, every factor stays in [−1, 1], and the site draws the sphere's image
as an ellipsoid.

## Still open

### XZZX: matched once now, but flat in distance

The double-matching bug is fixed — see `build_combined_graph`. Both edge families go into one graph
with a per-edge type tag, the defect set is matched once, and each chosen edge routes to X or Z by
its tag. A single error of either type is now corrected exactly, with no spurious operators, at
d = 3, 5 and 7, and the old runaway (5.4% at d=3 climbing to 21.0% at d=7) is gone.

What remains: the curve is flat in distance rather than falling. At the boundary a lone defect is
explainable by either error family at equal weight, and when the matcher picks wrong it leaves a
weight-one residual that the logical-operator check counts as a failure — at any distance.
Reproduce with a single Z error on qubit 0: one defect fires and the decoder answers with one *X*
correction. Settling it means fixing this lattice's boundary conventions and logical-operator
representatives.

### Circuit-level noise: no threshold

With bugs 3 and 4 fixed, circuit-level noise is a real measurement — noiseless is clean, and the
curve is monotonic. But the decoder still matches on the phenomenological spacetime graph, which
carries no edges for the correlated two-qubit errors a CNOT failing mid-extraction produces.
Without those hook-error edges the decoder mis-corrects in a way that grows with the patch, so
larger distances cost more than they buy and no threshold appears — measured down to p = 2×10⁻⁵.
Building the circuit-level detector graph is new work rather than a bug fix.

## Repository Structure

```
src/tableau.rs        symplectic tableau and Clifford operations
src/surface_code.rs   rotated and XZZX lattices, noise models, error generation
src/decoder.rs        Union-Find cluster growth and peeling
src/lib.rs            PyO3 module and the WASM C-ABI interface

index.html            the explainer — structure only
css/styles.css        design tokens, then base, then components
js/engine.js          typed wrapper over the WASM exports
js/channel.js         the noise channel, shared by the figures and the engine
js/channel-view.js    the logical channel drawn as the Bloch sphere's image
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
root. The committed `.wasm` is built this way and includes the two fixes above.

Note for Apple Silicon: if `cargo` reports `bad CPU type in executable`, the toolchain is the
x86_64 build and Rosetta is not available to the shell. `softwareupdate --install-rosetta` fixes
it; installing a native `aarch64-apple-darwin` toolchain is the better long-term answer.

### Python bindings & benchmarks

```bash
pip install maturin
maturin develop --release
python3 run_benchmarks.py
python3 run_data_benchmarks.py
```

## License

MIT

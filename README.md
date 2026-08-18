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

Three bugs surfaced while making the site report live data. All three are fixed in `src/` and in
the committed `stabilizer_qec.wasm`.

### 1. The WASM build's Monte Carlo was not random

Six `simulate_*` functions seeded their generator from a constant on the non-Python cfg branch:

```rust
#[cfg(not(feature = "python"))]
let mut rng = Xorshift::new(12345);   // and 54321 in two of them
```

Each of those functions *is* one shot, and `wasm_run_benchmark` calls them in a loop — so every
shot in a batch was bit-identical and the returned rate collapsed to a step function: exactly 0
below a cutoff, a plateau, exactly 1 above. Nothing the browser reported was a measurement. The
Python build takes the `rand::random()` branch and was unaffected.

Fixed by keeping one stream alive across shots and drawing seeds from it with **SplitMix64 over a
counter**. Advancing with the same xorshift recurrence the shots use is *not* sufficient — that
gives shot N+1 shot N's stream offset by one draw, and batch variance comes out several times
binomial. Verified: twelve repeats at d=5, p=5%, N=4000 give observed σ 0.00224 against a binomial
0.00222 (ratio 1.01).

A new export, `wasm_seed(lo, hi)`, lets the caller seed from `crypto.getRandomValues` so results
vary across page loads.

### 2. Phenomenological noise applied a faulty readout in the final round

Defects are time differences, so a lie in the last round has no following round to cancel against
and leaves an unpaired defect the decoder must match somewhere — injecting a correction for an
error that never happened. A larger patch has more checks to misreport on, so logical error *rose*
with distance far below threshold (1.9% at d=3 to 10.9% at d=7, at p=0.5%). The final round is now
noiseless, which is both the standard convention and what a real experiment does by ending with a
transversal data readout.

After both fixes the engine reproduces the literature: threshold near 10% for pure data noise and
near 2.6% phenomenological, and it agrees with an independent JavaScript Monte Carlo written
against the per-shot session API to within sampling noise at every point tested.

## Known engine defects (still open)

### The XZZX decoder reuses the wrong defect set

In the XZZX branch of `wasm_decode` (`src/lib.rs`), the second decoding pass does not derive its
own defects:

```rust
let graph_x = code.build_syndrome_graph(num_rounds, false);
let defects_x = defects_z.clone();     // <-- the Z pass's defects, on the X graph
```

XZZX has one syndrome and two edge families over the same nodes — one for the qubit diagonal an X
error flips, one for the Z diagonal. Decoding the full defect set independently on both explains
every defect twice, once with X's and once with Z's. Injecting a single X error on a d=3 patch
returns the correct one-qubit X correction plus **two spurious Z corrections**, each a fresh error.
Their count scales with the graph, so the code degrades as it grows: at p=2% unbiased over 4,000
shots, the rotated code improves from 0.9% at d=3 to 0.05% at d=7 while XZZX degrades from 5.4% to
21.0%, with a flat bias response from η=0.5 to η=64.

**Fix**: build one graph carrying both edge families with a per-edge type tag, decode the defect
set once, and route each matched edge's correction to X or Z by its tag. This is a decoder change
rather than a one-line correction, and it needs validating against the obvious test — logical error
must fall with distance below threshold — before it can be trusted.

### Circuit-level noise

`simulate_circuit_noise` returns a logical error rate that *falls* as the physical rate rises —
measured at d=7: 89% at p=0.1%, 82% at 0.2%, 62% at 0.5%, 51% at 1% — and is far worse at larger
distances. Not exposed on the site until diagnosed.

### Logical-state tomography

`wasm_estimate_logical_fidelity` returns `(1, 1, 1)` for a noiseless run. That vector has length
√3, outside the Bloch sphere, so it is not a Bloch vector — it is three independent survival
probabilities presented as one. Not exposed on the site until reworked.

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

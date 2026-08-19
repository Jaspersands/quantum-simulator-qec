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

Nine bugs surfaced while making the site report live data. All nine are fixed. Everything here is
reflected in `src/` and in the committed `stabilizer_qec.wasm`. They are written up in eight
sections below — section 6 covers two, which had to be fixed together before the XZZX code worked
at all.

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

Thresholds: ~11% data noise, ~2.7% phenomenological, ~0.45% circuit-level, each with reduced χ² near
1, agreeing point-by-point with an independent JavaScript Monte Carlo written against the per-shot
session API. The XZZX code has its own, measured separately: ~10% data noise and ~0.40%
circuit-level.

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

### 6. FIXED — XZZX matched one defect set twice, and checked the wrong logical operator

Two bugs, and both had to go before the code worked at all.

The decoder derived its defects once then matched them *twice*, on two graphs with different
edge-to-qubit mappings, because the second pass reused the first's defects verbatim
(`let defects_x = defects_z.clone();`). XZZX has one syndrome and two edge families over the same
nodes — an X error flips the stabilizers on one diagonal, a Z error those on the other — so this
explained every defect twice and applied both corrections. A single X error returned the correct
one-qubit X correction plus two invented Z ones, and their count grew with the lattice, so the code
degraded as it grew: 5.4% at d=3 up to 21.0% at d=7, at p=2%. `build_combined_graph` now emits both
families into one graph with a per-edge type tag and a single set of timelike edges; the defect set
is matched once and each chosen edge routes to X or Z by its tag.

That left the rate flat in distance rather than falling, which turned out to be a second, unrelated
bug: the logical-operator check compared the residual against a hard-coded alternating string of
Paulis that is not a logical operator of this lattice. It fired on residuals that were not logical
operators at all — including *weight-one* residuals, which cannot be logical errors in any code of
distance 3 or more — and it did so at every distance, hence the flat curve. The check now reduces
the residual against a row-reduced basis of the stabilizer group and asks whether anything is left,
which requires no convention about representatives.

Brute-force search over the actual stabilizer group confirms the construction was always sound:
minimum logical weight 3 at d=3, above 4 at d=5, for both codes. The bugs were entirely in the
decoding and scoring.

Verified after both: no single X, Z or Y error causes a logical failure at d = 3, 5 or 7; the rate
falls with distance at every bias (η=64, p=2%: 0.48% → 0.05% → 0.03%); and XZZX beats the rotated
code under bias as the literature says it should — at d=7, p=3%, η=64: **0.07% against 0.43%**.

### 7. FIXED — the decoder had no model of the circuit

Circuit-level noise is qualitatively harder than the other two models. A fault on an ancilla partway
through its four CNOTs propagates onto every data qubit it has yet to touch, so one fault becomes a
correlated multi-qubit data error — a *hook error*. The phenomenological spacetime graph has no edge
for that, so the matcher explained it with two unrelated edges and could walk the correction into a
logical operator. Roughly one single fault in thirty did exactly that, and the code got worse with
distance instead of better.

`src/circuit_model.rs` derives the decoding graph from the circuit instead of assuming one. Every
elementary fault is propagated as a Pauli frame — H swaps x and z, CNOT sends `x_target ^= x_control`
and `z_control ^= z_target`, a Z-basis measurement flips exactly when the frame carries x — and the
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
alike. That test paid for itself twice:

1. It showed the two CNOT schedules must be transposes *the other way round* from the pairing first
   tried. The wrong pairing passes cleanly at d = 5 and d = 7 and fails 44 times out of 600 at
   d = 3 — sampling would very likely have missed it.
2. It caught state-preparation noise being applied before the baseline round, where an error sits in
   both readings a detector compares, never fires it, and lands in the residual uncorrectable at any
   distance. It had been showing up as a stubborn p¹ term.

With both fixed, the logical error rate at d = 3 scales as p² as theory demands (ratios 3.1, 3.6,
3.4 on successive doublings of p, against 4 for a clean p²), and a threshold appears where it should. The live sweep
fits p_th = 0.45% with ν = 1.48 and reduced χ² = 1.05 — the closest to the textbook exponent of
~1.46 of any of the three noise models.

### 8. FIXED — XZZX had no model of its circuit either

The detector error model above is a CSS construction. A rotated plaquette measures either X or Z, so
the two error families light disjoint detectors, the decoder matches twice — once per family — and
that split is exactly what makes the decomposition free. An XZZX plaquette reads `X Z Z X` from a
single ancilla: one syndrome bit, both families firing the same detectors, no split available.

So the XZZX + circuit-level pairing never received any of the three XZZX fixes. It still built two
graphs, matched the same defect set against both, and scored against the hard-coded logical string —
bugs 6 and 7 alive in a path the bench UI let you select. Its logical error rate *rose* steeply with
distance, from 4.9% at d=3 to 40.6% at d=7 at p=0.2%, where the rotated code at the same settings
falls from 0.23% to 0.03%.

`build_combined` now derives one graph over a single node set, each edge naming the X part and the Z
part of the correction it implies. Only X and Z faults are enumerated, never Y: propagation is
linear over the Pauli frame, so a Y fault is exactly the composition of the two, and letting the
matcher pick both graphlike edges reconstructs it. That is the decomposition the CSS model got for
free, made explicit. The circuit holds every ancilla in |+> and reads an X leg with a CNOT, a Z leg
with a CZ.

**The two sublattices must walk their neighbours in transposed orders** — the same lesson bug 4
taught, and no more guessable the second time. With any single order shared by both sublattices, all
24 permutations leave 10 to 12 of 672 single faults uncorrectable at d=3 while d=5 stays clean.
Searching the two sublattices independently over all 576 combinations, scored on commutation against
the tableau and on the exhaustive single-fault check, leaves exactly 6 that pass — all of them
transposes. The search is kept as an ignored test (`xzzx_search_schedules`).

Two properties need separate checks here, because the simulation runs on the Pauli frame rather than
the tableau. The frame is exact for a Clifford circuit under Pauli noise, and it makes the scoring
honest — an XZZX logical operator is a mixed X/Z string, so there is no row of qubits to measure the
way the rotated code can, and the residual goes to the stabilizer group instead. But a frame
presumes the circuit really is a valid *simultaneous* measurement; if the schedule made neighbouring
plaquettes disturb each other the frame would stay self-consistent while the device produced noise.
That property is tested against the tableau: after the first round has projected, every later
noiseless round must reproduce its outcomes exactly, at d = 3, 5 and 7 across seeds.

Verified: **0 failures out of 672 / 3,600 / 10,416 faults** at d = 3 / 5 / 7, Union-Find and exact
MWPM alike; the rate now falls with distance (0.60% / 0.43% / 0.07% at p=0.2%); and a threshold fits
at ~0.40%. That sits just below the rotated code's 0.45%, and the gap is very nearly the extra
noise: every XZZX ancilla needs an H where the rotated code rotates only its X-type ones, giving 72
noise locations per round at d=3 against 64.

One honest negative result: **the bias advantage does not survive the circuit.** At d=7, p=0.3%,
going from η=1 to η=100 takes XZZX from 0.83% to 0.35% while the rotated code goes from 0.40% to
0.00%. Under circuit-level noise the dominant failures are ancilla faults propagating through a CNOT
and a CZ alike, which no amount of dephasing bias suppresses. The XZZX advantage this project does
reproduce (0.07% against 0.43% at d=7, p=3%, η=64) is a data-noise result, and the bias figure on
the site measures it in that regime.

### Located loss under circuit-level noise

An erasure is a qubit the hardware knows it lost — the Pauli is uniform and unknown, the location is
not. That makes it far easier to correct: every decoding edge the location could have produced costs
nothing, so the matcher routes through it freely.

Under circuit-level noise the location is a point in the circuit rather than a qubit, and an erasure
on an ancilla partway through its CNOTs frees everything the loss goes on to touch — the same
propagation that produces a hook error. The detector error model already records which edges each
circuit location produces, so it answers that question directly: `DetectorGraph::site_edges` maps an
erasure site to the edges it makes free.

Measured at d = 5: logical error at p = 0.8% falls from 8.2% with no erasure, to 3.4% when half the
faults are located, to nothing when all of them are. Fully located noise has its own threshold near
p = 2%, roughly four times the Pauli threshold — and the fact that it *has* a threshold, rather than
being perfect everywhere, is the check that the information is being used rather than assumed.

## Repository Structure

```
src/tableau.rs        symplectic tableau and Clifford operations
src/surface_code.rs   rotated and XZZX lattices, noise models, error generation
src/decoder.rs        Union-Find cluster growth and peeling
src/circuit_model.rs  detector error model derived from the extraction circuit, CSS and non-CSS
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

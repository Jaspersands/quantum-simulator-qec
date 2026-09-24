# Sub-project A — a general circuit path, checked against Stim

**Date:** 2026-09-24
**Status:** Approved (Jasper approved sections 1–3 interactively and "all your planning" for the rest,
then asked for the plan and the build to run overnight)

## Where this sits

This is the first of seven sub-projects agreed on 2026-09-24:

| | Sub-project | Needs |
|---|---|---|
| **A** | **General circuit path + Stim/PyMatching cross-check (this spec)** | — |
| B | Decode Google's Sycamore / Willow data, measure Λ | A; permission to download the dataset |
| C | Bit-parallel frame simulation, threaded WASM, circuit-level ν | — |
| D | Belief propagation, belief-matching / correlated matching | A |
| E | Lattice surgery | A |
| F | The [[144,12,12]] gross code with BP+OSD | A, D |
| G | Resource estimator; streaming decoder | A, B/C |

Everything after A takes a circuit, or a detector error model, that this engine did not write
itself. A is the machinery that makes that possible, and the proof that the machinery is right.

## Problem

1. **The engine can only decode circuits it wrote itself.** `circuit_model.rs` derives its decoding
   graph from each code's own `round_program`. Google's data ships as Stim circuits and detector
   error models; lattice surgery and the gross code are circuits the per-code builders cannot express.
2. **Corrections are data-qubit bitmasks.** Every edge carries a `u128` of data qubits it corrects.
   That caps a patch at 128 data qubits and ties decoding to this engine's qubit numbering. The
   standard formulation, and the one external data uses, is that each error mechanism flips a set
   of detectors and a set of logical observables.
3. **The circuit-level noise model is not the literature's.** After every CNOT each qubit takes an
   independent biased Pauli with probability *p*, so a gate fails about 2*p* of the time, and idle
   qubits never err. The standard model (SD6: one two-qubit depolarizing channel of total *p* per
   CNOT, plus idle noise) is what published thresholds assume. The site's 0.41% may be a correct
   threshold for a harsher model rather than a low one, and nothing on the site can currently tell.
4. **Nothing has been checked against the field's reference tools.** The engine has been checked
   against itself (exhaustive single faults, a toric code sharing no physics). It has never been
   checked against Stim and PyMatching, which is what anyone in the field would ask first.

## Decisions

| Question | Decision |
|---|---|
| Order | Foundation first: A → B → C → D → E → F → G. |
| Approach | A general path **alongside** the old per-code paths. The old paths and every number they produce are untouched. |
| Noise models | Both. The engine's model stays, named **Current**; **SD6** is added. Both thresholds are quoted and the site explains why they differ. |
| On the site | A live check plus marked references: our detector error model is derived live and compared with Stim's shipped file; our rates run live beside PyMatching's recorded ones, which are visibly marked. Section 10's opening line becomes "Nothing of ours is precomputed." |
| Tests | The DEM matches Stim's to 1e-9 relative on every mechanism; both decoders see identical shots; the exhaustive single-fault check runs on the new path. |

## Components

All new Rust is additive. No existing function changes signature or behaviour.

### 1. `src/circuit.rs` — the circuit

A flat instruction list over numbered qubits, read from and written to Stim's text format.

Supported instructions, which cover Stim's generated surface-code circuits and everything
`to_circuit` emits:

- Gates: `R`, `RX`, `H`, `CX` (alias `CNOT`), `CZ`, `M`, `MX`, `MR`, `MRX`.
  Measurements may carry a flip probability, as in `M(0.01) 3`.
- Noise: `DEPOLARIZE1`, `DEPOLARIZE2`, `X_ERROR`, `Z_ERROR`, `Y_ERROR`, `PAULI_CHANNEL_1`.
- Annotations: `DETECTOR` (with coordinates, `rec[-k]` targets), `OBSERVABLE_INCLUDE(k)`,
  `QUBIT_COORDS`, `SHIFT_COORDS`, `TICK`.
- Blocks: `REPEAT n { … }`, kept as a block in the structure and flattened on demand.

Anything else is a parse error naming the instruction and the line. There is no silent skip.
Round trip: `parse(emit(c)) == c`, and Stim accepts everything `emit` writes.

### 2. `src/dem.rs` — the detector error model

`Dem { num_detectors, num_observables, detector_coords, mechanisms: Vec<Mechanism> }`, where a
`Mechanism` is `{ p: f64, detectors: Vec<u32> (sorted), observables: u64, pieces: Vec<Piece> }`
and each `Piece` is a graph-like component (at most two detectors, plus an observable mask).

**`Dem::from_circuit(&Circuit)` uses backward sensitivity analysis**, the way Stim's error
analyzer works. It walks the flattened circuit in reverse, keeping two bitsets per qubit over
detectors and observables: `sx[q]` holds what an X error on `q` at this point would flip, and
`sz[q]` what a Z error would. Gates conjugate the pair: H swaps them; CX sends `sx[c] ^= sx[t]`
and `sz[t] ^= sz[c]`; CZ sends `sx[a] ^= sz[b]` and `sx[b] ^= sz[a]`. A Z-basis measurement
feeding record *k* adds the detectors and observables that include *k* to `sx[q]`. A Z-basis
reset clears both. At each noise instruction every Pauli component's symptom is read directly: X
gives `sx`, Z gives `sz`, Y gives `sx ^ sz`, and two-qubit Paulis XOR their two sides.

The cost is O(instructions × bitset words) rather than O(faults × instructions). It makes d = 7
cheap enough to derive in the browser while the reader watches.

**Determinism is checked during the same pass.** A detector's value depends on a random outcome in
two cases:

- a Z-basis reset (`R`, `MR`), or the start of the circuit, where every qubit begins in |0⟩, is
  reached with `sz[q]` non-empty;
- an X-basis reset (`RX`, `MRX`) is reached with `sx[q]` non-empty.

Either way the builder fails with the detector's index and coordinates, which is the check Stim
makes. Measurement flip probabilities (`M(p)`) contribute an independent mechanism whose symptom
is exactly the detectors and observables that include that record.

**Channel conversion follows Stim exactly.** Disjoint channels become independent Pauli
components:

- `DEPOLARIZE1(p)` → X, Y, Z each with q = ½(1 − √(1 − 4p/3)).
- `DEPOLARIZE2(p)` → the 15 two-qubit Paulis each with q = ½(1 − (1 − 16p/15)^⅛).
- `PAULI_CHANNEL_1(px, py, pz)` → independent qx, qy, qz with
  (1 − 2qx) = √(λY·λZ/λX), and cyclic, where λX = 1 − 2(py + pz), λY = 1 − 2(px + pz),
  λZ = 1 − 2(px + py). A non-positive λ is an error naming the channel.
- `X_ERROR`, `Y_ERROR`, `Z_ERROR` and measurement flips are already independent.

These formulas were checked against Stim 1.16 while writing this spec. `DEPOLARIZE1(0.01)` before
`M` gives `error(0.006666…) D0`, and `DEPOLARIZE2(0.01)` gives `error(0.0026738…)` on each of its
three symptoms. Components with identical symptoms merge by XOR probability, p ← p₁(1 − p₂) +
p₂(1 − p₁). Components with empty symptoms are dropped. A component with observables but no
detectors is an undetectable logical error, and the builder fails on it.

**Decomposition** (revised during the build; see the amendment below). Faults are decomposed the
way Stim's error analyzer decomposes them, so that this engine's matching graph is the one
PyMatching builds from Stim's model:

1. Each composite channel instance (`DEPOLARIZE1`, `DEPOLARIZE2`, `PAULI_CHANNEL_1`) splits every
   combination of its basis errors using only that channel's own single-detector combinations and
   its irreducible two-detector ones, as in Stim's `decompose_helper_add_error_combinations`.
   Stim's basis order is kept.
2. Anything still wider than two detectors goes through Stim's global pass: a backtracking
   partition into known one- and two-detector pieces with matching observables, then a greedy
   fallback that allows one remnant edge.
3. Faults that share a symptom but decompose differently remain separate mechanisms, each with
   its own probability.

If none of this works, building fails and names the fault. Invariants: every piece has one or two
detectors, and the pieces XOR back to the fault exactly, observables included. Stim assumes that
last property; this engine checks it.

**Amendment (build night).** The first decomposition kept any fault with at most two detectors
whole. A Y fault on a boundary qubit then became an edge joining the X-check and Z-check graphs,
and two single faults of the d = 3 memory-X circuit under SD6 decoded into logical errors, which
PyMatching on Stim's model corrected. The exhaustive single-fault check caught it. Mirroring
Stim's analyzer fixed it, and a new check, 1b, compares the decomposed graphs edge for edge.

**Stim's `.dem` text is read and written**, covering `error`, `detector`, `logical_observable`,
`shift_detectors` and `repeat`, with `^` separators read as pieces. Reading does not re-derive
pieces. A mechanism Stim did not decompose, which has more than two detectors and no `^`, has
no pieces, and handing that to the decoder is an error.

### 3. `src/dem_decoder.rs` — weighted matching over any DEM

- **Graph:** one node per detector, plus the boundary. Each piece is an edge with weight
  `ln((1 − p)/p)`, scaled to integers at 2²⁰ per unit. A one-detector piece is a boundary edge.
- **Parallel edges:** with the same observable mask they XOR-combine, which is what PyMatching 2.4
  does (checked). With different masks the more probable edge is kept and the event is counted in
  `DemDecoder::conflicts`. It is expected to be zero for any code of distance ≥ 3, since a
  conflict means a weight-two logical.
- **Decode:** Dijkstra from each defect, carrying the XOR of observable masks along each path, then
  the existing `blossom::min_weight_perfect_matching` with the existing m-boundary-copies
  construction. It returns the predicted observable flips as a `u64`.
- **Failures are explicit, never fallbacks** (the lesson of bug 12):
  - More than 512 matching vertices returns `Err(TooManyDefects)`.
  - A piece with p > 0.5 rejects the model at construction.
  - An odd component with no route to the boundary returns `Err(Unmatchable)`.
  - A published rate counts every shot it attempted, and failed decodes are reported as a count
    of their own.

**Known limit, owned by B:** the decoder is dense, so Google's long Willow runs, with over 1,000
defects per shot, exceed it. B will add a sparse or windowed matcher.

### 4. `src/frame_sampler.rs` — sampling that never reads the DEM

A scalar Pauli frame walked forward through the circuit. Noise channels are sampled with their
true disjoint semantics, not the DEM's independent approximation. Resets clear the frame. A
measurement records whether the frame flips it; a Z-basis measurement is flipped by an x
component. Detection events and observable flips are XORs of recorded flips. That is exact for
deterministic detectors, which the builder has already verified.

The frame gets its randomness from the existing SplitMix64/xorshift RNG and `wasm_seed`. Because
the sampler never touches the DEM, a DEM that is wrong about the circuit shows up as a
disagreement between the two, rather than being checked against itself. C will make this sampler
bit-parallel.

### 5. `src/shots.rs` — detection-event files

Read and write Stim's `01` and `b8` formats for detectors and observables. B reads Google's data
with this. In A, it carries shots from Stim to our decoder in the cross-check.

### 6. `src/memory.rs` — codes as circuits (additive)

It lives in its own module rather than `surface_code.rs`, which is already 1,842 lines.
`generate(CodeKind, d, rounds, NoiseModel, Basis) -> Circuit` covers the rotated and XZZX codes, with
`NoiseModel::{ Current { p, eta }, Sd6 { p } }` and `Basis::{ Z, X }`.

- **Schedules:** the CNOT schedules are exactly those in `round_program` and
  `round_program_ordered`, which the exhaustive single-fault checks settled. They are read from the
  same constants, not copied.
- **Memory experiment:** data prepared in the logical basis, and ancillas prepared in |0⟩, or |+⟩
  through `RX` and `H` as `round_program` does. After *T* rounds, data is measured transversally.
  Detectors come from consecutive ancilla readings and from the final data readout against the last
  round. The observable is the logical operator's support in the final readout.
- **XZZX:** the XZZX code is the rotated code with a Hadamard on a subset of data qubits.
  `xzzx_hadamard_pattern()` derives that subset by comparing the two codes' plaquettes at the same
  coordinates, and fails loudly if any qubit is inconsistent. Each data qubit is then prepared and
  measured in the basis the pattern gives it. First-round detectors cover exactly the stabilizers
  that preparation makes deterministic. The observable is the rotated logical conjugated by the
  pattern, and it is checked to commute with every XZZX stabilizer.
- **`Current { p, eta }`** reproduces the old path gate for gate:
  - `PAULI_CHANNEL_1` biased by η at every `Op::Noise` location.
  - `X_ERROR(p)` before each ancilla measurement.
  - The baseline round and the final round noiseless, with a noiseless data readout.
- **`Sd6 { p }`**, from Gidney, Newman, Fowler and Broughton (2021):
  - `DEPOLARIZE2(p)` after every two-qubit gate and `DEPOLARIZE1(p)` after every single-qubit gate.
  - `DEPOLARIZE1(p)` on every qubit idle in a TICK layer.
  - `X_ERROR(p)` after `R`, `Z_ERROR(p)` after `RX`, and the same flips before `M` / `MX`.
  - Every round noisy, including the first and the data readout.
- Every emitted circuit carries `QUBIT_COORDS` and detector coordinates `(x, y, t)`, so detectors
  can be matched across paths by coordinate.

**What counts as a failure.** The old path's frame sees both logical classes, so it counts any
logical error. A real circuit measures one basis, as Stim does and as the literature reports.
The SD6 numbers are `memory_z` and are labelled "per basis" everywhere they appear. The crossing,
which is what a threshold is, is insensitive to this, but the rates below threshold are not, and
the label says so.

### 7. Bindings

- **PyO3** (in `lib.rs`, additive):
  - `generate_circuit(code, d, rounds, noise, p, eta, basis) -> str`
  - `dem_from_circuit(text) -> str`, with a flag for undecomposed or with pieces
  - `decode_dem(dem_text, dets: numpy bool[shots, n]) -> numpy uint64[shots]`, plus the
    failure count
  - `sample_circuit(text, shots) -> (dets, obs)`
  - timing helpers
- **WASM**:
  - Text goes in through a shared byte buffer (`wasm_text_buf(len) -> *mut u8`), following the
    `wasm_match_cost_ptr` pattern, and results come back as JSON text through the same buffer.
  - Exports: `wasm_xc_generate`, `wasm_xc_compare_dem` (build ours from a circuit, parse Stim's,
    return a comparison summary), `wasm_xc_run` (sample and decode *n* shots on the new path,
    returning failures and failed decodes) and `wasm_xc_single_fault`.
  - `js/engine.js` stays the only module that touches raw pointers. It gains
    `NOISE.SD6 = 3` and typed wrappers around the new exports.

## The cross-check

### Checks, strictest first

**1. The error models are identical.** For every circuit in the matrix:

- Our undecomposed DEM and Stim's (`detector_error_model(decompose_errors=False).flattened()`)
  have the same set of symptoms.
- Every probability agrees to 1e-9 relative.

The matrix:

- `stim.Circuit.generated("surface_code:rotated_memory_z")` at d = 3, 5, 7 and T = d, with all four
  noise parameters set to 0.003. Stim writes these circuits and ours parses them.
- Our `to_circuit` for {rotated, XZZX} × {Current(p = 0.003, η = 0.5), SD6(p = 0.003)} × d = 3, 5, 7
  × memory_z. We write these and Stim parses them.

**1b. The matching graphs are identical.** Our decomposed model and Stim's
(`decompose_errors=True`, on the flattened circuit) decompose every symptom the same way, and every
edge has the same probability, to 1e-9 relative.

**2. The decoders agree shot for shot.** On 100,000 Stim-sampled shots per circuit, at SD6 d = 3,
5, 7 and p ∈ {0.003, 0.006}, our decoder and PyMatching each decode identical detection events
from Stim's decomposed DEM. Both are exact MWPM with the same weights, so they may disagree only on
ties and integer rounding.

- Reported: disagreements per shot, and each decoder's failure rate on those shared shots.
- Pass: **every disagreement is a tie.** Our total matching weight and PyMatching's
  (`decode(..., return_weight=True)`) agree to within the two decoders' weight discretisation.
  A rate threshold is not used, because near threshold genuine ties are common and a rate says
  nothing about whether both answers are minimum-weight. `DemDecoder` therefore also returns the
  total weight of its matching in the original float weights.

**3. The samplers agree.** At the same points, our frame sampler on our circuit and Stim's sampler
on the same circuit text give per-detector firing rates that agree (χ² over detectors), and
logical error rates (our sampler + our decoder against Stim's sampler + PyMatching) whose Wilson
intervals overlap.

**4. Speed.** Microseconds per shot for decoding, ours against PyMatching, at d = 3, 5, 7. It is
reported as measured. PyMatching's sparse blossom is expected to win by a large factor, and the
page says so.

**5. Old path and new path describe the same circuit.** In Rust, not against Stim:

- For rotated and XZZX at d = 3, 5, 7 under Current, every elementary fault the old
  `CircuitLayout::propagate` enumerates maps to a component of the new DEM with the same detector
  set. Detectors are matched through their (stabilizer, round) coordinates.
- The merged probabilities agree.

That is exact at the level of symptoms, which makes a statistical comparison of the two samplers'
detector rates redundant. The new sampler is checked statistically anyway, against the DEM's
exact marginals in a unit test and against Stim's sampler in check 3.

The old and new decoders differ, since the old one is unit-weight and the new one is weighted by
probability. So their logical rates are *reported side by side, not asserted equal*. If proper
weights move the Current threshold, that is a finding for the README.

**6. Single faults, exhaustively.** Every mechanism of the new DEM, applied alone and decoded by the
new decoder, gives no logical failure. This runs for rotated and XZZX at d = 3, 5, 7 under both
noise models. It is the project's signature test, carried over to the new path.

### The harness

- `tools/xcheck.py` runs checks 1–4 in `.venv`, which holds `stim`, `pymatching`, `numpy` and
  `maturin`, and imports the engine built with `maturin develop --release`.
- It prints a report and writes `data/xcheck/reference.json`, which records:
  - the Stim and PyMatching versions, the date and the command;
  - for each circuit: the mechanism count and the SHA-256 of Stim's DEM text;
  - for each decoding point: shots, both failure counts, disagreements, and µs per shot.
- It writes Stim's undecomposed flattened DEMs to `data/xcheck/<name>.dem.txt` (`.txt` so GitHub
  Pages serves them compressed) and Stim's generated circuits to `data/xcheck/<name>.stim.txt`.
- Checks 5 and 6 are `cargo test`. The d = 7 cases are `#[ignore]` if they take more than a few
  seconds, and run explicitly in the verification step.

## The site

### Section 10: "Checked against Stim", a new figure

It lives in the figure vocabulary (caption line, un-boxed, ink on paper). It sits directly after
section 10's opening prose, which it is evidence for, and before the list of engine entries. A
short paragraph introduces it. Two small tables:

1. **The error model, derived live.** A row per circuit showing: circuit, detectors, mechanisms
   (ours / Stim's), largest relative Δp, and a verdict: *identical*, or the count that differ.
   - Our side is built in the worker from our own circuit, or, for Stim's generated circuits, from
     the shipped `.stim.txt`.
   - Stim's side comes from the shipped `.dem.txt`.
   - The files are fetched lazily when the figure nears the viewport. d = 3 and 5 run on arrival,
     and d = 7 (266 KB) runs on a "check d = 7" control.
2. **Decoding.** A row per (d, p) showing:
   - our live rate, from our sampler and decoder, with its interval;
   - PyMatching's rate †;
   - shot-for-shot disagreements †;
   - µs per shot, ours live and PyMatching's †.
   - Each † value is recorded and set in the muted mono style with the dagger. The footnote names
     the Stim and PyMatching versions, the date, and `tools/xcheck.py` as the way to reproduce them.

The opening line of section 10 changes from "Nothing on this page is precomputed" to "**Nothing of
ours is precomputed.**" A sentence then says what the recorded references are and why: PyMatching
is a Python/C++ library and does not run in the browser.

### Section 07: SD6 as a second circuit-level model

- The sweep's noise selector gains "Circuit-level (SD6, per basis)", run through the new path. Its
  sweep window is centred on the crossing measured for SD6. Stim's own circuit under PyMatching
  crosses near 0.65–0.7%, so the window starts around 0.3%–1.0% and is re-centred on what our SD6
  circuit actually shows.
- Its distances follow the circuit-level ones (3, 5, 7, 9) and its shots are set by measured cost,
  as the existing comment block requires.
- One paragraph explains why the two circuit-level thresholds differ. Current fails a CNOT with
  about 2*p* and has no idle noise; SD6 fails it with *p* and does. The numbers themselves are
  quoted from the fits.

### Section 09: the bench

The noise selector gains SD6, using the same path.

### README

- A new section, "Checked against Stim and PyMatching", with the results of checks 1–6.
- SD6 rows in the threshold table, measured with four independent sweeps like the other rows.
- The side-by-side Current rates for the old and new decoders from check 5.
- Anything found along the way is written up in the style of the defect sections.

## Error handling, summarised

| Situation | Behaviour |
|---|---|
| Unknown circuit instruction | Parse error naming the instruction and line |
| Non-deterministic detector | Build error naming the detector and its coordinates |
| Non-positive λ in a Pauli channel | Build error naming the channel |
| Undetectable logical mechanism | Build error naming the fault (the code has distance < 2 against it) |
| Mechanism that cannot be decomposed | Build error naming the fault |
| Piece with p > 0.5 | Decoder construction error |
| More than 512 matching vertices | `Err(TooManyDefects)`, counted, never a fallback |
| Unmatchable odd component | `Err(Unmatchable)`, counted |
| Reference file fails to load on the site | The row reads "reference unavailable". Nothing is fabricated and no stale number is shown |

## Testing

- **Rust unit tests** (`cargo test`):
  - Circuit parse/emit round trip.
  - Stim's two channel examples above, as exact numbers.
  - Determinism failures on a deliberately bad circuit.
  - The decomposition invariants.
  - The decoder against brute force (every edge subset) on random small DEMs.
  - The sampler against the DEM, where per-detector marginals follow from the DEM exactly
    as (1 − Π(1 − 2pᵢ))/2.
  - Checks 5 and 6.
- **The harness** (`tools/xcheck.py`) runs checks 1–4. Its report is the evidence for the README.
- **Site:**
  - `node tools/site-tests.mjs` gains the pure pieces (the reference-file parsing and the
    formatting of the comparison summary).
  - `node tools/contrast.mjs` still passes, since the dagger style is a token.
  - A headless-browser check loads the page, waits for the figure, and reads its rows.
- **No regressions:** every existing test still passes. The WASM is rebuilt and committed, and the
  existing sections' numbers come from unchanged code.

## Out of scope

- Sparse or windowed matching, Google's data formats beyond `01`/`b8`/`.dem`, and SI1000 noise
  (all B).
- A bit-parallel sampler and threads (C).
- Correlated or belief-propagation decoding (D).
- Moving the existing sections onto the new path. The old paths stay until a later sub-project
  retires them on the strength of check 5.
- Union-Find over a general DEM. The new path is MWPM, which is what PyMatching is.

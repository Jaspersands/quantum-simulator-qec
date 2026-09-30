# I — Lattice-Surgery Programs: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compile lattice-surgery *programs* (patches on a grid of tiles, merges along lines, splits, measurements) into one circuit each, and measure:
- a logical CNOT end to end at circuit level;
- sequences of Z⊗Z measurements;
- a three-patch Z⊗Z⊗Z measurement.

**Architecture:** `src/surgery.rs` gains a grid `Layout`, a `Program` of `Step`s, and explicit observables built from `Term`s: final logical strings, merge outcomes and seam records. The existing `Surgery` experiment becomes a program, and must compile byte for byte to the same circuit. Detectors keep the one rule that section 14 already proved. Observables' determinism is checked by the engine's reference simulation (`M2d`), which refuses any nondeterministic one. That check is the test of every hand-derived Pauli frame.

**Tech Stack:** Rust (the crate), PyO3 bindings, `tools/surgery.py` (stim 1.16, pymatching 2.4 in `.venv-f`), the site's WebAssembly (`wasm_ls.rs`, `js/sections/surgery.js`).

## Global Constraints

- **Byte-identical back-compatibility.** `Surgery { d, pre, merged, post, basis, p }.circuit()` must produce exactly the same Stim text as before for every d in {3, 5, 7}, merged in {1, 2, 3, d, 2d} and both bases. Golden SHA-256 values are recorded before any change (Task 1).
- **Geometry follows `RotatedSurfaceCode`.** Data sits at odd (x, y) and checks at even (x, y); a check is X-type by the parity of (x + y)/2; CNOT orders as today.
- **Boundaries.** Left and right boundaries are X-type, and top and bottom Z-type.
- **Logicals.** A patch's logical Z is its first data column (smallest x), and its logical X its first data row (smallest y).
- **Tiles.** They are 2d + 2 apart in both directions, so one seam column or row of data lies between neighbours.
- **Merges.**
  - A horizontal line of patches measures Z⊗…⊗Z: the seam is prepared in |+⟩ and split by measuring X.
  - A vertical line measures X⊗…⊗X: the seam is prepared in |0⟩ and split by measuring Z.
  - The outcome is the product of the merged code's new checks of the measured type in the merge's first round.
- **SD6 noise** at p, exactly as the `Writer` applies it today.
- **Build the Python engine** with `VIRTUAL_ENV=$PWD/.venv-f CARGO_TARGET_DIR=target-f .venv-f/bin/maturin develop --profile python`.
- **Timing and threads.** Native Rust timings use `--target aarch64-apple-darwin`. Decoding runs are Python and multi-threaded.

---

### Task 1: Golden circuits, then a grid layout

**Files:**
- Modify: `src/surgery.rs` (`Layout`)
- Create: `data/surgery/golden.json` (the SHA-256 of today's circuits)

**Interfaces:**
- Produces:
  - `Layout::new(d, cols, rows)`;
  - `Layout::tile(&self, (i, j)) -> TileBox { x0, x1, y0, y1 }`, the data columns x0..=x1 and rows y0..=y1, odd;
  - `Layout::code(&self, b: TileBox) -> Vec<Check>`;
  - `Layout::data(&self, b: TileBox) -> Vec<u32>` (sorted);
  - `TileBox::union(a, b)`.

- [ ] **Step 1: Record the golden hashes, with today's code.** Add a test that prints `sha256(circuit.to_stim())` for d ∈ {3, 5, 7}, merged ∈ {1, 2, 3, d, 2d} and both bases. Write the values to `data/surgery/golden.json` from a Python one-off using `sq.surgery_circuit`:

```bash
.venv-f/bin/python - <<'EOF'
import hashlib, json, stabilizer_qec as sq
out = {}
for d in (3, 5, 7):
    for merged in sorted({1, 2, 3, d, 2 * d}):
        for basis in ("z", "x"):
            text = sq.surgery_circuit(d, merged, 0.002, basis)
            out[f"d{d}/T{merged}/{basis}"] = hashlib.sha256(text.encode()).hexdigest()
json.dump(out, open("data/surgery/golden.json", "w"), indent=1)
print(len(out), "circuits")
EOF
```

- [ ] **Step 2: Write the failing test** (in `surgery.rs`'s tests). It compares today's `Surgery` circuits with the golden file:

```rust
    /// The grid layout and the program compiler change nothing: every
    /// experiment section 14 measured compiles to the same circuit, byte for
    /// byte, as before either existed.
    #[test]
    fn the_z_z_experiment_is_unchanged() {
        let golden = include_str!("../data/surgery/golden.json");
        for d in [3usize, 5, 7] {
            let mut ts = vec![1usize, 2, 3, d, 2 * d];
            ts.sort_unstable();
            ts.dedup();
            for merged in ts {
                for (basis, name) in [(Basis::Z, "z"), (Basis::X, "x")] {
                    let text = Surgery { d, pre: d, merged, post: d, basis, p: 0.002 }.circuit().unwrap().to_stim();
                    let key = format!("\"d{d}/T{merged}/{name}\": \"{}\"", crate::sha256_hex(text.as_bytes()));
                    assert!(golden.contains(&key), "d = {d}, T = {merged}, {name}: the circuit changed");
                }
            }
        }
    }
```

This needs `crate::sha256_hex`. If the crate has no SHA-256, add a 60-line `sha256_hex(&[u8]) -> String` to `src/lib.rs`. It is test-only (`#[cfg(test)]`) and is checked against the NIST vector for "abc" in its own test.

- [ ] **Step 3: Run it.** Expected: PASS against today's code. It is the contract for Steps 4–5.

- [ ] **Step 4: Generalise `Layout`**

```rust
/// Data columns x0..=x1 and rows y0..=y1 (odd) of one patch, or of patches merged.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct TileBox { x0: i32, x1: i32, y0: i32, y1: i32 }

impl TileBox {
    fn union(a: TileBox, b: TileBox) -> TileBox {
        TileBox { x0: a.x0.min(b.x0), x1: a.x1.max(b.x1), y0: a.y0.min(b.y0), y1: a.y1.max(b.y1) }
    }
}

struct Layout {
    d: i32,
    positions: Vec<(i32, i32)>,
    index: HashMap<(i32, i32), u32>,
}

impl Layout {
    /// A grid of `cols` × `rows` tiles, 2d + 2 apart: data first, row by row,
    /// then every check position, row by row (for 2 × 1 tiles, exactly the
    /// order of the two-patch experiment).
    fn new(d: usize, cols: i32, rows: i32) -> Layout {
        let d = d as i32;
        let (w, h) = (cols * (2 * d + 2) - 2, rows * (2 * d + 2) - 2);
        let (mut positions, mut index) = (Vec::new(), HashMap::new());
        for y in (1..h).step_by(2) {
            for x in (1..w).step_by(2) {
                index.insert((x, y), positions.len() as u32);
                positions.push((x, y));
            }
        }
        for y in (0..=h).step_by(2) {
            for x in (0..=w).step_by(2) {
                index.insert((x, y), positions.len() as u32);
                positions.push((x, y));
            }
        }
        Layout { d, positions, index }
    }

    fn tile(&self, (i, j): (i32, i32)) -> TileBox {
        let (x0, y0) = (1 + i * (2 * self.d + 2), 1 + j * (2 * self.d + 2));
        TileBox { x0, x1: x0 + 2 * self.d - 2, y0, y1: y0 + 2 * self.d - 2 }
    }

    fn data(&self, b: TileBox) -> Vec<u32> {
        let mut v: Vec<u32> = (b.y0..=b.y1)
            .step_by(2)
            .flat_map(|y| (b.x0..=b.x1).step_by(2).map(move |x| (x, y)))
            .map(|p| self.index[&p])
            .collect();
        v.sort_unstable();
        v
    }

    /// The rotated code on box `b`: X-type boundaries left and right, Z-type
    /// top and bottom.
    fn code(&self, b: TileBox) -> Vec<Check> {
        let inside = |x: i32, y: i32| x >= b.x0 && x <= b.x1 && y >= b.y0 && y <= b.y1;
        let mut checks = Vec::new();
        for y in (b.y0 - 1..=b.y1 + 1).step_by(2) {
            for x in (b.x0 - 1..=b.x1 + 1).step_by(2) {
                let x_type = ((x + y) / 2).rem_euclid(2) == 1;
                // Z checks stay off the left and right edges, X checks off the top and bottom.
                let allowed = if x_type { y > b.y0 && y < b.y1 } else { x > b.x0 && x < b.x1 };
                if !allowed {
                    continue;
                }
                let order = if x_type { RotatedSurfaceCode::X_ORDER } else { RotatedSurfaceCode::Z_ORDER };
                let mut at_step = [None; 4];
                for (k, &(dx, dy)) in order.iter().enumerate() {
                    if inside(x + dx, y + dy) {
                        at_step[k] = Some(self.index[&(x + dx, y + dy)]);
                    }
                }
                if at_step.iter().flatten().count() >= 2 {
                    checks.push(Check { pos: (x, y), x_type, anc: self.index[&(x, y)], at_step });
                }
            }
        }
        checks
    }
}
```

The old `y >= 2 && y <= 2 * d - 2` for X checks is `y > y0 && y < y1` with y0 = 1 and y1 = 2d − 1, and the old `x > x0 && x < x1` is unchanged. Update `Surgery::circuit` to use `Layout::new(d, 2, 1)`, `layout.tile((0, 0))` and `layout.tile((1, 0))`. The seam box is `TileBox { x0: 2d + 1, x1: 2d + 1, y0: 1, y1: 2d − 1 }`. Update `the_codes_have_the_right_number_of_checks` to the new signatures, adding a vertical merge: two tiles stacked have d(2d + 1) − 1 checks, as two side by side do.

- [ ] **Step 5: Run the whole suite.** The golden test passes (the circuits are unchanged), as do the existing surgery tests.
- [ ] **Step 6: Commit:** `refactor(surgery): a grid of tiles, the two-patch experiment unchanged byte for byte`.

---

### Task 2: The program compiler, and the Z⊗Z experiment as a program

**Files:**
- Modify: `src/surgery.rs`

**Interfaces:**
- Produces (public):

```rust
pub struct Program {
    pub d: usize,
    pub p: f64,
    /// Each patch's tile, (column, row).
    pub tiles: Vec<(i32, i32)>,
    pub steps: Vec<Step>,
    /// Each observable, as the terms whose records it multiplies.
    pub observables: Vec<Vec<Term>>,
}

pub enum Step {
    /// Prepare the patches' data in `basis`, as one reset.
    Prepare { patches: Vec<usize>, basis: Basis },
    /// Rounds of syndrome extraction on every live patch and merge.
    Rounds(usize),
    /// Merge a line of adjacent patches: a horizontal line measures Z⊗…⊗Z, a
    /// vertical one X⊗…⊗X. The seam is prepared at once; rounds follow.
    Merge { patches: Vec<usize> },
    /// Split every current merge: each seam measured, X for a horizontal line, Z for a vertical one.
    Split,
    /// Measure the patches' data in `basis`, as one measurement; they are done.
    Measure { patches: Vec<usize>, basis: Basis },
}

#[derive(Clone, Copy, Debug)]
pub enum Term {
    /// A patch's logical in `basis`, from its final measurement in that basis:
    /// Z its first column, X its first row.
    Logical { patch: usize, basis: Basis },
    /// Merge `merge`'s outcome (merges numbered in program order).
    Outcome { merge: usize },
    /// Merge `merge`'s seam records, from its split, on the line of `patch`'s
    /// logical that crosses it: its first row for a horizontal line, its first
    /// column for a vertical one.
    Seam { merge: usize, patch: usize },
}

impl Program { pub fn circuit(&self) -> Result<Circuit, String> }
```

- `Surgery::circuit` builds a `Program` and compiles it:
  - tiles `[(0,0), (1,0)]`;
  - steps: `Prepare{[0,1], basis}`, `Rounds(pre)`, `Merge{[0,1]}`, `Rounds(merged)`, `Split`, `Rounds(post)`, `Measure{[0,1], basis}`;
  - observables for Z: `[[Outcome 0], [Logical 0 Z], [Logical 1 Z]]`; for X: `[[Logical 0 X, Logical 1 X, Seam{0, 0}]]`.

- [ ] **Step 1: Write the compiler.** It lives in `surgery.rs`, below `Writer`:
  - **Coordinates:** it emits `QubitCoords` for every position.
  - **`Prepare`:** one `w.prepare` over the patches' data, concatenated in the order given, each patch's sorted.
  - **`Rounds`:** each round runs `w.round(checks, data)` over the *units*:
    - every live patch not in a merge, in patch order: its own code and data;
    - every current merge, in order of its lowest patch: its merged code and merged data, sorted.
  - **Unit order:** units are ordered by their lowest patch index. Checks and data are concatenated in unit order.
  - **Merge outcome:** the first round after each merge records that merge's outcome records, the records of its new checks.
  - **`Merge`:**
    - validates that the patches are live, not merged, and a line of consecutive tiles, in order;
    - emits `Tick`, then prepares the seam: the merged box's data less the patches' data, sorted, in X for a horizontal line and Z for a vertical one;
    - its new checks are the merged code's checks of the measured type whose position is in no patch's code.
  - **`Split`:** one `Tick`, then one `w.measure` per current merge's seam (X for horizontal, Z for vertical), in merge order.
  - **`Measure`:**
    - one `Tick`, then one `w.measure(basis, data)` over the patches' data;
    - then, for each measured patch's checks of that basis's type, in patch order, a detector:
      - the check's final support records, in support order;
      - then its last record;
      - then the records of qubits its last measurement read that its patch code does not (seam qubits measured at a split in the same basis).
    - It refuses a check whose last support lost a qubit not measured in its basis, as `round` does.
  - **Observables:** after the steps, each observable's terms are resolved to records in term order, and an `Observable { index, recs }` is emitted for each, in order.

- [ ] **Step 2: Rewrite `Surgery::circuit`** on it, keeping its two validation errors verbatim.
- [ ] **Step 3: Run the suite.** `the_z_z_experiment_is_unchanged` passes: the program compiles to the same bytes. The determinism and single-fault tests pass too.
- [ ] **Step 4: Commit:** `feat(surgery): lattice-surgery programs; the Z⊗Z experiment is one, byte for byte`.

---

### Task 3: Vertical merges (X⊗X), the CNOT, sequences and the product

**Files:**
- Modify: `src/surgery.rs` (builders and tests)

**Interfaces:**
- Produces:
  - `pub fn cnot(d, merged, p, inputs: Basis) -> Program`. Tiles: control (0,0), ancilla (1,0), target (1,1). Steps:
    1. `Prepare{[C,T], inputs}` and `Prepare{[A], X}`, then `Rounds(d)`;
    2. `Merge{[C,A]}`, `Rounds(merged)`, `Split`;
    3. `Merge{[A,T]}`, `Rounds(merged)`, `Split`;
    4. `Measure{[A], Z}`, `Rounds(d)`, `Measure{[C,T], inputs}`.

    Observables for Z inputs:
    - `[Logical C Z]`;
    - `[Logical T Z, Logical A Z, Outcome 0, Seam{1, T}]`.

    Observables for X inputs:
    - `[Logical T X]`;
    - `[Logical C X, Logical T X, Outcome 1, Seam{0, C}]`.
  - `pub fn repeated(d, k, merged, p) -> Program`: two patches in |0⟩|0⟩, then k Z⊗Z merges of `merged` rounds, each split and followed by one round apart. Observables: each merge's outcome, then Z₁ and Z₂.
  - `pub fn product(d, n, merged, p) -> Program`: n patches in a row in |0⟩ merged at once. Observables: the outcome, then each patch's Z.
  - `pub fn vertical(d, merged, p, basis) -> Program`: the X⊗X mirror of the Z⊗Z experiment, two patches stacked. For X inputs, the outcome and each patch's X; for Z inputs, Z₁Z₂ with the seam records on patch 0's column.

- [ ] **Step 1: Write the failing tests**

```rust
    /// Every program's detectors and observables are deterministic without
    /// noise. For the CNOT this checks the hand-derived Pauli frames: a
    /// frame missing a record would leave its observable random, and the
    /// reference simulation refuses it.
    #[test]
    fn programs_are_deterministic() {
        for d in [3usize, 5] {
            for basis in [Basis::Z, Basis::X] {
                for prog in [cnot(d, d, 0.0, basis), vertical(d, d, 0.0, basis)] {
                    let c = prog.circuit().unwrap();
                    M2d::new(&c).unwrap_or_else(|e| panic!("d = {d} {basis:?}: {e}"));
                }
            }
            for prog in [repeated(d, 3, d, 0.0), product(d, 3, d, 0.0)] {
                M2d::new(&prog.circuit().unwrap()).unwrap_or_else(|e| panic!("d = {d}: {e}"));
            }
        }
    }

    /// The frame is exercised: without noise the merge outcomes the CNOT's
    /// observables use come out -1 in about half of shots, so a wrong frame
    /// could not hide behind outcomes that happened to be +1.
    #[test]
    fn the_cnot_s_outcomes_are_random() {
        use crate::frame_sampler::FrameSampler;
        let prog = cnot(3, 3, 0.0, Basis::Z);
        let c = prog.circuit().unwrap();
        let sampler = FrameSampler::new(&c).unwrap();
        let mut rng = crate::surface_code::Xorshift::new(4);
        let first = prog.outcome_records(0);
        let mut ones = 0;
        for _ in 0..400 {
            let shot = sampler.sample(&mut rng);
            ones += first.iter().fold(false, |a, &r| a ^ shot.measurements[r]) as usize;
        }
        assert!((120..=280).contains(&ones), "{ones} of 400");
    }

    /// With d merged rounds, every single fault of the CNOT is corrected.
    #[test]
    fn the_cnot_corrects_every_single_fault() {
        for basis in [Basis::Z, Basis::X] {
            let dem = Dem::from_circuit(&cnot(3, 3, 0.001, basis).circuit().unwrap()).unwrap();
            let dec = DemDecoder::new(&dem).unwrap();
            let failed = dem.mechanisms.iter()
                .filter(|m| dec.decode(&m.detectors).map(|p| p.observables != m.observables).unwrap_or(true))
                .count();
            assert_eq!(failed, 0, "{basis:?}: {failed} of {} single faults", dem.mechanisms.len());
        }
    }
```

`Program::outcome_records(merge) -> Vec<usize>` is a test helper that compiles and returns the measurement indices of a merge's outcome. `FrameSampler`'s shot needs its measurement bits. If the sample type has no `measurements`, use `M2d`'s reference path to read records instead.

- [ ] **Step 2: Run them and see them fail** (the builders do not exist).
- [ ] **Step 3: Implement the builders.** If `programs_are_deterministic` refuses a CNOT observable, the error names it. Derive the missing records by propagating the observable's Pauli back through the program:
  - at each split, for seam qubits on its line;
  - at each merge, for the outcome.

  Record the derivation in the builder's doc comment.
- [ ] **Step 4: Run the suite.** All pass.
- [ ] **Step 5: Commit:** `feat(surgery): X⊗X merges, a logical CNOT, repeated Z⊗Z and a three-patch product, with their frames checked`.

---

### Task 4: Bindings, the Stim cross-check, and CI

**Files:**
- Modify: `src/py_api.rs`, `stabilizer_qec.pyi`, `tools/surgery.py`, `tools/smoke.py`, `.github/workflows/ci.yml` (already runs `surgery.py check --quick`)

**Interfaces:**
- Produces:
  - `surgery_cnot(d, merged, p, inputs="z") -> str`;
  - `surgery_repeated(d, k, merged, p) -> str`;
  - `surgery_product(d, n, merged, p) -> str`;
  - `surgery_vertical(d, merged, p, basis="x") -> str`.

  All return Stim text.

- [ ] **Step 1: Bindings and stubs,** following `surgery_circuit`'s pattern: parse the basis string, build, then `circuit()?.to_stim()`.
- [ ] **Step 2: Extend `tools/surgery.py check`.** For:
  - the CNOT, both inputs, d = 3 (and d = 5 in the full check);
  - `repeated` with k = 3;
  - `product` with n = 3;
  - `vertical`, both bases,

  run the same two checks as the Z⊗Z circuits: the error model against Stim's, fault for fault (mechanisms, graph edges, splits), and PyMatching on Stim's model against ours on ours (every disagreement a tie). `--quick` covers d = 3 only.
- [ ] **Step 3: A smoke test** in `tools/smoke.py`: `surgery_cnot(3, 3, 0.001)` builds, its model decodes, and one merged round is refused.
- [ ] **Step 4: Run** `.venv-f/bin/python tools/surgery.py check --quick` and `tools/smoke.py`. Expected: `ALL CHECKS PASSED` and `all checks passed`.
- [ ] **Step 5: Commit:** `feat(surgery): the programs from Python, checked against Stim and PyMatching`.

---

### Task 5: Measurements

**Files:**
- Modify: `tools/surgery.py` (add `programs`)
- Create: `data/surgery/programs.json`

- [ ] **Step 1: Add `tools/surgery.py programs`.** Each point samples by the batch sampler, decodes with correlated and plain matching, and records failures per observable and "any observable wrong", with Wilson intervals. It reuses `run`'s point function, with a target of 1,500 failures of the "any" event, a shot cap of 2,000,000 and a time cap of 900 s. The points:
  - **CNOT:** d ∈ {3, 5, 7} × p ∈ {0.002, 0.003} × inputs ∈ {z, x}, T = d; and T ∈ {2, …, 2d} at d ∈ {3, 5}, p = 0.003, inputs z.
  - **Repeated:** d ∈ {3, 5}, p = 0.003, k ∈ {1, 2, 4, 8}, T = d.
  - **Product:** d ∈ {3, 5, 7}, p = 0.003, n = 3, T = d.
  - **Windowed:** the CNOT at d = 5, p = 0.003, both inputs, parallel windows (commit and buffer d, correlated), against global decoding of the same shots.
- [ ] **Step 2: Run it** in the background (a few hours at most), then commit the data with its `engine_commit`.
- [ ] **Step 3: Check the sum-of-parts law.** Fit `P(k) = 1 − (1 − q)^k` to the repeated points and report q against the single measurement's rate. The CNOT is compared with the model the estimator uses: 3 patches × 2d merged rounds × ε_d, where ε_d is the memory's error per round at the same p from `data/surgery/results.json`. The ratio goes in the data file as `model_ratio`.

---

### Task 6: The site, the README and the report

**Files:**
- Modify: `js/surgery-geometry.js` (tiles), `js/sections/surgery.js`, `index.html` (section 14), `src/wasm_ls.rs` (programs), `README.md`, `report/report.md`, `tools/report.py`, `tools/readme_tables.py`

- [ ] **Step 1: WebAssembly.** `wasm_ls.rs` gains a `kind` argument: 0 is today's Z⊗Z, 1 the CNOT with Z inputs, 2 the CNOT with X inputs. The live figure decodes the chosen program at d = 3.
- [ ] **Step 2: Figure 16 gains a CNOT diagram.** Three tiles in an L, two seams, drawn with the section's palette; a control steps through the program (prepare, merge C–A, split, merge A–T, split, measure A).
- [ ] **Step 3: Two new figures,** from `programs.json`: the CNOT's failure against d at both p (each "any wrong" with its interval, beside 2 × the Z⊗Z measurement's rate), and the sequence law (P(k) against k, with 1 − (1 − q)^k drawn).
- [ ] **Step 4: README and report.** A "Programs" subsection under Lattice surgery: the CNOT construction and its frames, the checks, and the tables. Those tables are generated by `readme_tables.py`, which gains a `surgery_programs` table. The report's section is filled from the same data.
- [ ] **Step 5: The estimator (section 15)** notes the measured CNOT against its per-operation model, `model_ratio`. If the ratio lies outside [0.5, 2], the model's per-operation failure is scaled by it, and the page says so.
- [ ] **Step 6: Verify.**
  - `cargo test`;
  - `tools/surgery.py check`, full;
  - `tools/smoke.py`;
  - the WebAssembly build and smoke test;
  - the site tests, `readme_tables.py --check` and the report build;
  - a headless check of section 14 at 1440 and 390 px.
- [ ] **Step 7: Review at high effort, fix, merge, push, and check CI.**

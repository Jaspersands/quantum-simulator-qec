# I — Lattice-surgery programs: a logical CNOT and sequences of operations

**Date:** 2026-09-27
**Status:** Proposed. Questions answered under Jasper's standing instruction ("do both or the
biggest"), marked where they were; for his review before a plan is written.

## Why

Section 14 measures one operation: a Z⊗Z measurement between two patches, merged and split once. An
algorithm is a sequence of such operations, and the first one anyone asks about is the logical
CNOT, which lattice surgery builds from two joint measurements and an ancilla patch (Horsman et
al., arXiv:1111.4022; Litinski, arXiv:1808.02892):

1. Prepare an ancilla patch A in |+⟩.
2. Measure Z_C Z_A between the control and A.
3. Measure X_A X_T between A and the target.
4. Measure A in Z.
5. Apply Pauli corrections fixed by the three outcomes, tracked in software as a Pauli frame, never
   applied physically.

Simulating that end to end at circuit level answers three questions:
- how often a logical CNOT fails;
- whether a sequence of operations fails as the sum of its parts, which is what the resource
  estimator assumes;
- whether the detector rule written for one surgery holds for any program.

## Design

### A program, not an experiment

`src/surgery.rs` writes one hard-wired experiment. It becomes a small **program compiler**:

- **Layout.** Patches sit on a grid of tiles. A patch is a rotated distance-d code at a tile
  position, oriented with its Z boundary vertical or horizontal. Seams between adjacent tiles are
  single columns or rows of data qubits.
- **Instructions:**
  - `prepare(patch, basis)`;
  - `idle(rounds)`;
  - `merge(patches, pauli, rounds)`, where `pauli` is Z⊗Z or X⊗X, and a line of more than two
    patches measures the product;
  - `split`;
  - `measure(patch, basis)`.
- **Detectors** follow the rule the one-shot experiment already proved. A check measured in
  consecutive rounds is compared across them. Where its support changed, the qubits it gained were
  freshly prepared in its basis and the qubits it lost were measured in its basis, so their records
  join the comparison. The rule does not care how many merges come before.
- **The Pauli frame.** Each measurement outcome is a set of records. The compiler tracks, per
  logical qubit, which outcome records flip its X and Z frame. A logical observable is written as
  the final data readout XOR its frame records. That makes it deterministic, and so checkable.

The existing Z⊗Z experiment becomes a four-line program. The compiler must produce its circuit
exactly as before: byte-identical Stim text, so section 14's data stands.

### Experiments

1. **Logical CNOT, both input bases.** A CNOT is fixed by what it does to Z_C, Z_T, X_C and X_T:
   - inputs |0⟩|0⟩ check Z_C → Z_C and Z_T → Z_C Z_T;
   - inputs |+⟩|+⟩ check X_C → X_C X_T and X_T → X_T.

   Two programs cover the four. Each has two final logical observables plus the three measurement
   outcomes, and the CNOT succeeds when all are decoded right.
2. **Sequences.** k Z⊗Z measurements back to back on the same pair, for k = 1 to 8, each over d
   merged rounds. Does the failure probability grow as k times one measurement's, or faster?
3. **Multi-patch product.** Z⊗Z⊗Z across three patches in a row (biggest). This is the Pauli-product
   measurement that Litinski's compilation reduces every gate to.

Decoding uses correlated and plain matching on the whole program's error model. Every program is
graph-like: X and Z faults stay in their own graphs, as for memories.

## Decisions

| Question | Decision |
|---|---|
| Which CNOT construction | **The ancilla-patch one** (Z⊗Z then X⊗X). The X⊗X merge needs a horizontal seam, so the layout is an L of three tiles. This also exercises both seam orientations. |
| Rounds | Merges of **T = d** (section 14's measured knee), with T swept 2 to 2d for the CNOT at d = 3 and 5 (biggest). |
| Distances and noise | d = 3, 5, 7; SD6 at p = 0.2% and 0.3%, as section 14. |
| Real-time decoding of a program | **Also windowed**, reusing section 12's parallel windows over the program's rounds. The outcome of step 2 must be known before step 4's correction is fixed, so this is the decode that sets a program's reaction time. Global decoding stays the reference. |

## Verification

1. **Determinism.** Every detector and observable of every program is deterministic without noise.
   The engine's reference simulation refuses any that is not.
2. **Stim.** Each program's error model equals Stim's fault for fault. PyMatching on Stim's model
   and ours on ours agree on every shot but ties.
3. **The frame is exercised.** Without noise the merge outcomes are random: each comes out −1 in
   about half of sampled shots, which is checked. The frame-corrected observables are nonetheless
   deterministic, so a wrong frame record could not hide behind an outcome that happened to be +1.
4. **Back compatibility.** The Z⊗Z program's circuit is byte-identical to the current experiment's,
   and `tools/surgery.py check` still passes.
5. **The sum-of-parts law.** Experiment 2's failure probability is compared with k times the single
   measurement's, with Wilson intervals. The estimator's per-operation model is then either confirmed
   or corrected from data.

## Outputs

- **Section 14 grows.** A drawing of the CNOT's three tiles and its merges, the CNOT's failure
  against d and p, and the sequence law.
- **Live.** The CNOT sampled and decoded in the browser.
- **Python.** `surgery_program(...)` in the bindings, `tools/surgery.py cnot`, and
  `data/surgery/cnot.json`.
- **Estimator.** A measured CNOT and per-operation cost for section 15. Its "N · d · ε_d per
  operation" becomes the measured law.

## Out of scope

Twists, S gates by surgery, and the T gate. Magic states are sub-project K. Routing on a real
floor plan is K's too.

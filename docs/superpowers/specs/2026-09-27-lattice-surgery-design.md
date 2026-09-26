# E — Lattice surgery: measuring Z⊗Z between two patches

**Date:** 2026-09-27
**Status:** Decided under Jasper's standing instruction of 2026-09-26 ("do both or the biggest";
questions answered that way and marked).

## Why

Everything so far is memory, one logical qubit sitting still. A computer needs logical qubits to
interact. On a surface-code chip the standard way is **lattice surgery** (Horsman, Fowler, Devitt and
Van Meter, 2012):
1. **Merge.** Two patches are merged into one by measuring the stabilizers across the seam between
   them for some rounds.
2. **Read the product.** The product of the new stabilizers is the joint logical parity, Z₁Z₂
   here.
3. **Split.** The patches are split again.

It is how Clifford+T algorithms are laid out, and its timing (how many merged rounds a
measurement needs) sets the clock of every resource estimate. This sub-project builds the
experiment as a circuit, checks it as strictly as the memories, and measures what the merged rounds
buy.

## The experiment

Two rotated distance-d patches sit side by side, one column of data qubits (the seam) between them.
Each patch's logical Z runs down a column, so the patches face each other with the boundaries whose
weight-two checks are X-type. Coordinates follow `RotatedSurfaceCode`: data at odd (x, y), checks
at even (x, y), and the same CNOT orders.

1. **Prepare.** Both patches in |0⟩_L (data in |0⟩), then R₀ rounds of separate memory.
2. **Merge.** The seam is prepared in |+⟩, and the merged patch (2d + 1 columns × d rows) is measured
   for T rounds.
   - The old boundary X checks each extend across the seam onto qubits in |+⟩, so they stay
     deterministic.
   - The new Z checks along the seam are each random, but their product is Z₁Z₂, which is +1 for
     |0⟩|0⟩.
3. **Split.** The seam is measured in X. The patches' boundary X checks resume, and each is predicted
   by its last merged value times its two seam qubits.
4. **Finish.** R₁ rounds of separate memory, then the data are measured in Z.

Three observables:
- **L₀** is the merge outcome: the product of the new Z checks in the first merged round, which must
  come out +1.
- **L₁ and L₂** are each patch's final logical Z, which the ZZ measurement leaves alone.

The surgery succeeds when all three are decoded right. Noise is SD6, as for every memory.

## Decisions

| Question | Decision |
|---|---|
| Which parity | **Z⊗Z** by a vertical seam. X⊗X is the same by symmetry: a horizontal seam, or Hadamard-rotated patches. |
| Rounds | R₀ = R₁ = d; **T swept from 1 to 2d** (biggest), at d = 3, 5, 7. |
| Decoder | The sparse matcher, **plain and correlated**, on this engine's error model of the whole experiment. The model is graph-like, as for memories. |
| A second check of the physics | **Also the X-basis version** (both patches in \|+⟩_L). There the merge outcome is random, but X₁X₂ must survive it, so X₁X₂ is the observable. |

## Verification

1. **Determinism.** Every detector and observable is deterministic in the noiseless circuit. The
   engine's own references refuse any that is not, so a wrongly placed detector or observable cannot
   pass.
2. **Stim.** Our error model equals Stim's fault for fault, decomposed graph and all. PyMatching and
   our matcher agree on sampled shots, and every disagreement is a tie.
3. **Every single fault is corrected when T ≥ 2.** At T = 1, a single measurement error in the merged
   round flips L₀ unseen: nothing after it repeats the new checks. The check asserts both, since the
   first failure is the physics this experiment is about.
4. **The timing law.** The merge outcome's failure rate falls with T until the space-like failures
   of the patches dominate. By T ≈ d it has flattened, which is the textbook "d rounds per lattice
   surgery". Measured at p = 0.1% and 0.3%.

## Site and write-up

- **Section 14, lattice surgery.**
  - **A diagram:** the two patches, the seam, the checks of the merged patch, and the new Z checks
    whose product is the outcome. A slider steps through prepare, merge, split and finish.
  - **The recorded curve:** the merge outcome's failure rate against T at d = 3, 5, 7.
  - **A live panel:** the whole experiment sampled and decoded in the browser at d = 3 and 5.
- **README section, and a report subsection.** The measured timing feeds the resource estimator
  (the last sub-project).

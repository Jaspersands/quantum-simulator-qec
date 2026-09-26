# F — IBM's gross code, decoded by BP+OSD

**Date:** 2026-09-27
**Status:** Decided under Jasper's standing instruction of 2026-09-26 ("do both or the biggest";
questions answered that way and marked).

## Why

The surface code spends about $d^2$ physical qubits for each logical qubit. Bravyi, Cross,
Gambetta, Maslov, Rall and Yoder ("High-threshold and low-overhead fault-tolerant quantum memory",
Nature 627, 778, 2024; arXiv:2308.07915) showed a family of *bivariate bicycle* codes that does
far better. The flagship, the **gross code** $[[144, 12, 12]]$, keeps 12 logical qubits at distance
12 on 288 physical qubits. The surface code would need nearly 3,500 to match it.

Such codes are not matchable: a fault sets off many checks at once, and the decoder of choice is
**BP+OSD**. BP runs first; when it fails to converge, ordered-statistics decoding solves for the
most reliable consistent correction.

This engine builds any stabilizer circuit's error model and now has BP (sub-project D). The gross
code is the natural test of both beyond the surface code, and IBM's roadmap builds on it.

## Decisions

| Question | Decision |
|---|---|
| Code | **Bivariate bicycle codes in general**, from $\ell, m$ and the monomials of $A$ and $B$. The gross code is $\ell = 12$, $m = 6$, $A = x^3 + y + y^2$, $B = y^3 + x + x^2$. $[[72, 12, 6]]$ ($\ell = 6$) is checked too, as the paper's smaller code. |
| Circuit | **The paper's depth-8 syndrome cycle**, as Bravyi et al. published it with their simulation code. Each check qubit's six CNOTs are ordered `sX = [idle, 1, 4, 3, 5, 0, 2]` and `sZ = [3, 5, 0, 1, 2, 4, idle]` over its neighbours: $A_1, A_2, A_3$ on the left data and $B_1, B_2, B_3$ on the right, transposed for Z checks. X checks are prepared in round 0 and measured in round 7; Z checks are measured in round 6 and re-prepared in round 7. |
| Noise | **The paper's circuit-level model**, one parameter $p$: two-qubit depolarizing after every CNOT, single-qubit depolarizing on every idle data qubit in every round, preparation and measurement flips. |
| Experiment | **A Z-basis memory**, as the paper simulates: data prepared in $\lvert 0\rangle$, $N_c = 12$ syndrome cycles, then the data measured. The detectors are Z checks compared round to round, closed by the final readout; the observables are all 12 logical Z operators. The logical error per cycle is $p_L = 1 - (1 - P_L)^{1/N_c}$, as the paper defines it. |
| Decoder | **BP+OSD, both OSD variants** (biggest): OSD-0, and OSD-CS (combination sweep) of order 7. BP is min-sum with adaptive scaling, up to 10,000 iterations, the paper's settings. BP runs on the merged error model (faults with the same symptom merged), and OSD orders columns by BP's posteriors. |
| Oracle | **`ldpc`'s `BpOsdDecoder`**, from the same library whose BP ours already reproduces bit for bit. On identical syndromes, the corrections must be equal, or, where OSD's column order ties, equally weighted. |
| Logical operators | Computed, not typed in. Over GF(2), the logical Z operators are a basis of $\ker H_X$ modulo the rowspace of $H_Z$, paired with logical X operators so that the pairing matrix is the identity. The code checks $k = 12$ and that every logical commutes with every stabilizer. |

## Verification

1. **The code.** $n = 144$, $k = 12$; $H_X H_Z^T = 0$; the 12 logical pairs anticommute exactly in
   pairs. For $[[72,12,6]]$ as well. The distance is not recomputed (it is the paper's theorem), but
   a random search for low-weight logicals must find none below 12.
2. **The circuit.** Every detector is deterministic in the noiseless circuit (the engine's own m2d
   reference runs, which refuse nondeterministic detectors). Stim reads the circuit, and its error
   model equals ours fault for fault, as in the cross-check. Stim's
   `search_for_undetectable_logical_errors` bounds the circuit-level distance. The paper states
   that this circuit's distance is at most 10.
3. **The decoder.** BP+OSD against `ldpc` on identical syndromes, at several $p$: the same
   corrections, or ties.
4. **The result.** Logical error per cycle against $p$ for the gross code, with Wilson intervals,
   beside the paper's published curve, and beside a surface code of similar size decoded by
   matching.

## Site and write-up

- **A new section 13** on the site: the code's Tanner graph drawn on its torus, the syndrome
  cycle, and the recorded logical error per cycle.
- **A live panel** runs BP+OSD in the browser on the $[[72,12,6]]$ code, which is cheap enough.
- **A README section,** and a report subsection.

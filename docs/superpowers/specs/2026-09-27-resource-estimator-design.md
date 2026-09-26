# The resource estimator — from measured Λ to qubits and hours

**Date:** 2026-09-27
**Status:** Decided under Jasper's standing instruction of 2026-09-26 ("do both or the biggest";
questions answered that way and marked).

## Why

Every sub-project so far measures something a resource estimate needs:
- **Λ and ε**, how fast logical error falls with distance, on Google's own hardware data (B2, D);
- **real-time decoding**: how many cores keep up at each distance, and at what latency (C);
- **lattice surgery**: how many merged rounds one logical operation needs, measured (E);
- **the gross code**: what a denser memory buys (F).

The estimator puts them together to answer the question the field actually asks: how many physical
qubits, and how long, does an algorithm of a given size need? Each input is taken from this
project's own measurements, and each is shown next to what it came from.

## The model

An algorithm is N logical qubits and D logical operations (each a lattice-surgery step), with a
total failure budget δ.
1. **Logical error per cycle** at distance d is $\varepsilon_d = \varepsilon_{d_0}\,
   \Lambda^{-(d-d_0)/2}$, extrapolated from the measured ε at the largest measured distance.
2. **Each operation** takes $T = d$ merged rounds (measured in E), and every patch is exposed for
   those rounds. The failure per operation is about $N \cdot d \cdot \varepsilon_d$.
3. **The distance** is the smallest odd d with $D \cdot N \cdot d \cdot \varepsilon_d \le \delta$.
4. **Qubits** are $N \cdot (2d^2 - 1)$ for the data patches, times a routing overhead (default 2,
   for the bus that lattice surgery needs; adjustable).
5. **Time** is $D \cdot d$ rounds of 1.1 µs (Willow's cycle; adjustable).
6. **Decoding**:
   - cores per patch at that d, from C's measured latency, extrapolated by the measured growth of
     window cost with d;
   - the total cores for N patches.

The gross code enters as an alternative memory. Its measured error per cycle at the chosen p
stands against the surface code's at that d, for qubits that sit idle.

## Decisions

| Question | Decision |
|---|---|
| Where it runs | **An interactive section on the site** (biggest), and a table in the README and the report. |
| Which Λ | **Every measured one, selectable:** ours correlated on Google's prior (1.95), Google's Libra (2.04), our belief-matching on Willow, and a custom value. |
| Presets | Three illustrative sizes, labelled as illustrations and not literature claims: 100 qubits × 10⁶ operations, 1,000 × 10⁹, and 10,000 × 10¹². |
| Sensitivity | **Shown:** how the qubit count moves with Λ from 1.5 to 4, since Λ is the lever that matters. |

## Verification

- **The formulas as pure functions,** with site tests. At the measured distances the model
  reproduces the measured ε exactly; distances are odd and monotonic in D; qubits are monotonic
  in d.
- **The inputs,** read from the committed data files by the same code that draws the figures
  above, so the estimator cannot drift from them.

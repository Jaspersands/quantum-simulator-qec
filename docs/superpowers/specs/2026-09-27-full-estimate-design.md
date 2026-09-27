# K — A fuller resource estimate: magic states, routing, and reaction time

**Date:** 2026-09-27
**Status:** Proposed. Questions answered under Jasper's standing instruction ("do both or the
biggest"), marked where they were; for his review before a plan is written. Built last: it consumes
H, I and J.

## Why

Section 15 prices an algorithm with a model that fits in a sentence:
- N logical qubits and D operations, each a lattice-surgery step of d rounds;
- failure N · d · ε_d per operation;
- qubits N(2d² − 1) with a flat routing overhead of 2.

It shows how Λ drives the answer, and it leaves out what dominates real estimates:
- the non-Clifford gates, and the factories that feed them;
- the floor plan the Pauli-product measurements run on;
- the classical reaction time that each T gate's correction waits for.

Every serious estimate prices these. Gidney's RSA-2048 estimate (arXiv:2505.15917: under a million
noisy qubits, under a week, at 0.1% gate error, a 1 µs cycle and a 10 µs reaction time) is the
reference point a company reader has in mind.

## Design

The model stays in `js/estimator.js`, pure and tested, and gains three parts. Each input is either
measured by this project or read from a cited paper, and the page says which.

### K1. Algorithms

Presets give each algorithm's logical qubits and Toffoli (or T) count, read from the papers' own
tables and recorded with table references in `data/estimate/algorithms.json`:
- RSA-2048 (Gidney, arXiv:2505.15917);
- FeMoco by tensor hypercontraction (Lee et al., arXiv:2011.03494);
- the section's current illustrative sizes (100, 1,000 and 10,000 logical qubits), labelled as
  illustrations, with a T count per operation the reader sets;
- a custom row.

A Toffoli is priced as 4 T states, or 1 CCZ state, as the source does.

### K2. Magic states, both ways (biggest)

- **Distillation:** Litinski's 15-to-1 factories ("Magic state distillation: not as costly as you
  think", Quantum 3, 205, 2019). Each factory's footprint, cycles and output error at the chosen
  p come from his tables.
- **Cultivation:** Gidney, Shutty and Jones (arXiv:2409.17595), which reaches 2 × 10⁻⁹ at 0.1%
  noise for about the cost of a lattice-surgery CNOT. Its cost and output error against p come from
  their figures.
- **Our own check.** The 15-to-1 protocol is simulated at the logical level with this engine's
  stabilizer simulator: injected T states' errors, twirled to Z faults, through the [[15, 1, 3]]
  circuit. This reproduces the output law (35 p³ to leading order) and the acceptance rate, so the
  factory's error model is measured here, not only quoted.
- **Throughput.** The estimator places enough factories to feed the algorithm's T rate. It reports
  the factory share of qubits, and the T states' share of the failure budget.

### K3. Routing: a floor plan instead of a factor of 2

Litinski's data blocks ("A game of surface codes", Quantum 3, 128, 2019):
- compact: fewest tiles, slowest Pauli-product measurements;
- intermediate;
- fast: most tiles, one step per measurement.

Tiles and time steps per measurement come from the paper. A tile's qubits come from the measured
patch, 2d² − 1 plus its share of the routing space.

The CNOT and per-operation failure law measured in sub-project I replace the assumed "N · d · ε_d".

### K4. Reaction time, from real-time decoding

A T gate by teleportation waits for the decoder before its correction is known. The wait per T
layer is the measured windowed-decoding latency at the chosen d (section 12's p99, extrapolated as
the cores are), plus a control-system delay the reader sets (default 10 µs, Gidney's assumption).
Run time is then the larger of two bounds:
- the Clifford bound: operations × steps × d cycles;
- the reaction bound: T layers × reaction time.

This ties section 12 into the estimate: decoding speed becomes a line in the budget, and
sub-project H moves it.

### K5. A qLDPC-memory option

IBM's architecture keeps idle logical qubits in gross-code modules, and computes through logical
measurements (sub-project J's measured cost, and Yoder et al.'s instruction set). The estimate
offers that layout beside the surface code's. It uses our measured gross-code error per cycle and
J's measured logical-measurement cost, and says plainly where the model extrapolates.

## Decisions

| Question | Decision |
|---|---|
| Distillation or cultivation | **Both**, selectable, with the page defaulting to cultivation (the current state of the art). |
| Routing layouts | **All three of Litinski's blocks**, selectable. |
| Reaction time | **Measured decode latency plus a settable control delay**, not a fixed number. |
| Numbers from papers | Read from tables and figures, and recorded with their page, table or figure references in the data file. Where a figure must be digitised, the page says so. |
| Validation | The model, set to Gidney's assumptions (0.1%, 1 µs cycle, 10 µs reaction, cultivation), must land within a factor of 2 of his RSA-2048 physical-qubit count and run time. If it does not, the reason is found and stated before the section ships. |

## Verification

1. **Unit tests** in `tools/site-tests.mjs`: each part of the model reproduces the source's own
   worked examples (Litinski's factories, Gidney's RSA totals under his assumptions) to the stated
   precision.
2. The 15-to-1 simulation's output error follows 35 p³ at small p, with Wilson intervals, and its
   acceptance rate matches 1 − 15p to first order.
3. `tools/estimate.mjs` prints every preset's full breakdown, and the README and report tables are
   generated from it.

## Outputs

- **Section 15 rebuilt.** Algorithm presets, magic-state and layout choices, and a stacked bar of
  where the qubits go (data, routing, factories) and where the time goes (Clifford steps against
  reaction).
- The README's "What it would take" and the report's section regenerated.

## Out of scope

Compiling real circuits (the Toffoli counts are the papers'), yoked surface codes, and error-rate
drift over a long run.

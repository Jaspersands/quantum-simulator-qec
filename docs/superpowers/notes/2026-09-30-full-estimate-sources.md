# K sources (gathered 2026-09-30; verify page/table refs before shipping)

## Gidney 2025, arXiv:2505.15917 (RSA-2048)
- Max logical qubits 1,409 in loop4 (Table 4): m = 1,280 cold (yoked) + 131 hot; idle hot patches 7×18 = 126.
- Toffolis: 6.5e9 expected (Table 5, n = 2048).
- Physical: 897,864 = cold 1,280 × 430 (550,400) + hot 131 × 1,352 (177,112) + compute 7×18 patches (170,352); "< 1 million".
- Run time: 12.07 h per shot × 9.2 expected shots → 4.63 days; / 93.3% → 4.96 days ("less than a week").
- Hot storage d = 25, 2(d+1)² = 1,352 per logical; yoked cold 430 per logical; both 1e-15 per round.
- 6 CCZ factories (8T-to-CCZ via cultivation), each 3×4 hot patches; 150 rounds per CCZ; lattice surgery period 25 µs.
- Assumptions: 0.1% uniform depolarizing, 1 µs cycle, 10 µs reaction.

## Litinski 2019, "Magic state distillation: not as costly as you think", Quantum 3, 205 (arXiv:1905.06903), Table 1 (p. 2)
protocol | p_phys | p_out | qubits | cycles | qubitcycles
(15-to-1)7,3,3 | 1e-4 | 4.4e-8 | 810 | 18.1 | 14,600
(15-to-1)9,3,3 | 1e-4 | 9.3e-10 | 1,150 | 18.1 | 20,700
(15-to-1)11,5,5 | 1e-4 | 1.9e-11 | 2,070 | 30.0 | 62,000
(15-to-1)^4_9,3,3 x (20-to-4)15,7,9 | 1e-4 | 2.4e-15 | 16,400 | 90.3 | 371,000 (per output state; 4 outputs)
(15-to-1)^4_9,3,3 x (15-to-1)25,9,9 | 1e-4 | 6.3e-25 | 18,600 | 67.8 | 1,260,000
(15-to-1)17,7,7 | 1e-3 | 4.5e-8 | 4,620 | 42.6 | 197,000
(15-to-1)^6_13,5,5 x (20-to-4)23,11,13 | 1e-3 | 1.4e-10 | 43,300 | 130 | 1,410,000
(15-to-1)^4_13,5,5 x (20-to-4)27,13,15 | 1e-3 | 2.6e-11 | 46,800 | 157 | 1,840,000
(15-to-1)^6_11,5,5 x (15-to-1)25,11,11 | 1e-3 | 2.7e-12 | 30,700 | 82.5 | 2,540,000
(15-to-1)^6_13,5,5 x (15-to-1)29,11,13 | 1e-3 | 3.3e-14 | 39,100 | 97.5 | 3,810,000
(15-to-1)^6_17,7,7 x (15-to-1)41,17,17 | 1e-3 | 4.5e-20 | 73,400 | 128 | 9,370,000
(15-to-1)^6_13,7,7 x (8-to-CCZ)25,15,15 | 1e-3 | 5.2e-11 (per CCZ) | 47,000 | 60.0 | 2,820,000
Notes: qubitcycles are per output state; the "Qubits" include ancillas. A (20-to-4) protocol outputs 4 states per run.
Output law (one-level 15-to-1, faulty-T error p): leading order 35 p^3 (text: lowest achievable p_out for one level 10.37 p^3 with his faulty-T measurement model).

## Gidney, Shutty, Jones 2024, cultivation (arXiv:2409.17595)
- p_out 2e-9 at p = 1e-3, 4e-11 at p = 5e-4 (abstract); "roughly the same number of physical gates as a lattice surgery CNOT".
- Fig. 14: at p = 1e-3, d1 = 3: ~20 attempts per kept shot (75% discard?); d1 = 5: ~100 attempts (99% discard); at 5e-4, d1 = 5: 10 attempts.
- No explicit qubit-round table in text (Fig. 1 volumes, digitised if used). Use Gidney 2025's factory spec for the model: 3×4 hot-patch factory, 150 rounds per CCZ (8 cultivated T -> CCZ), at p = 1e-3.

## Litinski 2019, "A Game of Surface Codes", Quantum 3, 128 (arXiv:1808.02892), Sec. 2 (pp. 7-9), Figs. 9, 13
- Compact block: 1.5n + 3 tiles; up to 9 time steps (1 step = d code cycles) to consume a magic state.
- Intermediate block: 2n + 4 tiles (text, p. 8; Fig. 13a caption says 2.5n + 4 — text and the 100-qubit example, 204 tiles, agree on 2n + 4); up to 5 steps.
- Fast block: 2n + sqrt(8n) + 1 tiles (100 qubits: 231); 1 step.
- Worked examples (Sec. 4, pp. 17-19; 100 qubits, 1e8 T, p = 1e-4): compact 153 + 11 distillation tiles; intermediate 204 + 22 = 226 tiles, 76,400 qubits; fast 231 + 132 = 363 tiles, 123,000 qubits.
- Abstract: compact 55,000 qubits, 4 h; intermediate 120,000, 22 min; fast (time-optimal units) 3.3e8 qubits, 1 s.

## Lee et al. 2021, THC FeMoco (arXiv:2011.03494; PRX Quantum 2, 030305)
- Table III (p. 7): THC, Reiher et al. FeMoco: 2,142 logical qubits, 5.3e9 Toffolis; Li et al. FeMoco: 2,196 logical qubits, 3.2e10 Toffolis.
- Sec. (p. 31) and Fig. 10 (p. 28), Reiher Hamiltonian: floor plan 53 × 36 = 1,908 logical patches at d = 31, 1,908 · 2 · 32² ≈ 4e6 physical qubits;
  CCZ factories from Gidney-Fowler (level-1 d = 19, level-2 d = 31), four factories at a total 25 kHz; 483,000 inner loops × 13,880 Toffolis = 6.7e9 Toffolis;
  "3 days" of Toffolis; abstract: "about four million physical qubits and under four days", 1 µs cycles, 0.1% gate error.

## Gidney 2025, magic states and run time (arXiv:2505.15917, HTML)
- Cultivation: "cultivating a T state with a logical error rate of 1e-7 uses 30000 physical qubit·rounds" (at p = 1e-3).
- 8T-to-CCZ distillation: error suppression 28 p²: CCZ error 28·(1e-7)² < 1e-12.
- Six factories, 150 rounds per CCZ state per factory; factory = 6 layers of lattice surgery at 2/3 d rounds each; 3×4 hot patches.
- Run time is lattice-surgery-throughput limited, not reaction limited: lattice surgery period 25 µs (d = 25) = CCZ period (150/6 = 25 rounds).
  2 ms per addition, 2 ms per lookup, 1 ms per phaseup → 12.07 h per shot.
- Shots succeed with 93.3% (6.7% logical failure budget per shot); 4.63 days / 0.933 = 4.96 days.
- Hot: 131 logical at 2(d+1)² = 1,352 each (d = 25); cold: 1,280 logical at 430 each (yoked); both 1e-15 per patch per round.

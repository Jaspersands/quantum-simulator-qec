# J — Logical operations on the gross code

**Date:** 2026-09-27
**Status:** Proposed. Questions answered under Jasper's standing instruction ("do both or the
biggest"), marked where they were; for his review before a plan is written. The riskiest of the
four new sub-projects.

## Why

Section 13 shows the gross code [[144, 12, 12]] as a memory. It stores 12 logical qubits in a
tenth of the qubits twelve surface patches need, and fails less often. A memory is half the story.
IBM's architecture (Bravyi et al., arXiv:2308.07915; Yoder et al., "Tour de gross",
arXiv:2506.03094) computes on those qubits two ways:

- **Automorphisms.** Permutations of the qubits that map the code to itself, and so act on the 12
  logical qubits as Clifford gates.
- **Logical measurements** through an attached ancilla system, measuring a chosen logical Pauli
  operator. Cross et al. (arXiv:2407.18393) do it on the gross code with 103 ancilla qubits,
  enough for every logical Clifford. The construction generalises Williamson and Yoder's gauging
  measurement (arXiv:2410.02213).

The claim to make measurable is IBM's: that computing on a qLDPC memory costs little beyond the
memory itself.

## Design

### J1. The automorphism group, exactly

The gross code's checks are polynomials in two commuting shifts x (order 12) and y (order 6). So
every shift x^a y^b, applied to both halves of the data, is an automorphism.

- **Enumerate** the 72 shifts, and the ZX-duality of Bravyi et al. (swapping the halves and
  exchanging X with Z).
- **Check** each one exactly over GF(2): it maps the stabilizer group to itself, and maps logical
  operators to logical operators. Its action on the 12 logical qubits is a symplectic 24 × 24
  matrix in the paired logical basis `bb.rs` already computes.
- **Report:**
  - the group the actions generate, including its order;
  - which actions are logical permutations and which are not;
  - which logical Cliffords are reachable by automorphisms alone.

  Everything here is exact computation, not sampling.

### J2. Measuring a logical operator: the gauging construction

For a logical operator L of weight w (w = 12 at the gross code's distance):

1. **A graph G** on L's support is chosen with enough expansion that the deformed code keeps
   distance 12. Cross et al. give the gross code's. G is taken from their paper, and its expansion
   (Cheeger constant) and the deformed code's distance are checked here.
2. **One ancilla qubit per edge.** The new checks are:
   - a Gauss-law check per vertex, the vertex's qubit times its edges' qubits;
   - a flux check per cycle of a cycle basis;
   - the original checks, deformed to commute with them.

   The product of the Gauss-law checks is L, so measuring them measures L.
3. **A circuit** measures the deformed code for T rounds between memory rounds of the plain code.
   Detectors follow the lattice-surgery rule, as in sub-project I. The syndrome schedule for the new
   checks is searched for, as the memory's depth-8 cycle was, so that no single fault flips the
   outcome.
4. **Decoding** by BP+OSD-CS of order 7, the memory's decoder, on the whole experiment's
   undecomposed error model. The outcome's failure rate is measured against T and p, beside the
   memory's.

## Decisions

| Question | Decision |
|---|---|
| Which logical operator | **A weight-12 logical X̄ of one logical qubit, and a product across two** (biggest). One tests the construction; the other is the joint measurement a CNOT needs. |
| Where the ancilla graph comes from | **Cross et al.'s**, read from their paper and checked here: expansion, commutation, and distance. Designing a new graph is out of scope. |
| Distance of the deformed code | **Checked two ways.** A BP+OSD search for low-weight logicals gives an upper bound. An exact SAT search up to weight 11 must find none, within a 2-hour cap per operator; if the cap is hit, the result is reported as a bound. |
| Automorphisms in circuits | **Not simulated.** How hardware realises the shifts (routing, or rewiring the syndrome schedule) is device-specific. J1 reports exact logical actions, and makes no claim about their noise. |

## Verification

1. **J1:** exact GF(2) checks for all 144 maps (72 shifts, with and without the duality), with the
   group structure cross-checked against a Python implementation using `ldpc`'s GF(2) tools.
2. **J2 algebra:**
   - every deformed check commutes with every other;
   - the Gauss-law product equals L;
   - the flux checks span G's cycle space;
   - the deformed code encodes 11 logical qubits during the merge (L is now a stabilizer).
3. **J2 circuit:** noiseless determinism; the error model equals Stim's fault for fault; BP+OSD's
   corrections equal `ldpc`'s on sampled shots, as in section 13.
4. **Literature:** the ancilla count equals Cross et al.'s for the same operator, and the measured
   failure rate is compared with any rates they report.

## Outputs

- **Section 13 grows.** The automorphism group drawn on the torus as shifts, the gauging graph
  drawn on L's support, and the logical measurement's failure against T beside the memory's.
- **Python:** `bb_automorphisms(code)`, `bb_logical_measurement_circuit(...)`, and
  `tools/gross.py measure`.
- **Estimator (K):** the measured cost of a logical measurement on a gross-code module.

## Risks

The distance of the deformed code, and a syndrome schedule that keeps it, are research questions
Cross et al. settle for their graph. If the SAT check does not finish, the result is reported as a
bound. If the schedule search finds only schedules with a weight-one fault on the outcome, that is
what gets reported.

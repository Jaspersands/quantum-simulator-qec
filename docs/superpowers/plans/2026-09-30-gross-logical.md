# J — Logical operations on the gross code: implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compute the gross code's automorphism group and its exact logical action. Then build Cross et al.'s gauging measurement of a weight-12 logical X̄ (and of a product of two) as a circuit, check it against Stim and `ldpc`, and measure how often it fails against the number of merged cycles, beside the memory.

**Architecture:** Three Rust modules:
- `src/bb_auto.rs`: shifts and the ZX-duality as qubit maps, and their 24 × 24 symplectic logical actions.
- `src/bb_gauge.rs`: the gauging ancilla system of a logical X operator, and the deformed code.
- `src/bb_circuit.rs`: a cycle writer that runs any checks on any tick schedule, with the lattice-surgery detector rule. It reproduces `BbCode::memory_z`'s error model exactly, and on top of it builds the logical-measurement experiment.

Python tooling (`tools/gross_ops.py`) checks the algebra independently. It computes the deformed code's distance exactly by integer programming, checks the circuits against Stim and `ldpc`, and records measurements for section 13, the README and the report.

**Tech Stack:** Rust (engine, PyO3 bindings), Python 3 with numpy, scipy (`scipy.optimize.milp`, HiGHS), stim, ldpc; the site's vanilla JS modules.

## Global constraints

- Build the Python engine with `VIRTUAL_ENV=$PWD/.venv-f CARGO_TARGET_DIR=target-f .venv-f/bin/maturin develop --profile python`, never `--release`.
- Rust tests: `cargo test --release --no-default-features`.
- Commit per task, messages ending `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Everything in J1 is exact computation, not sampling. Automorphisms are not simulated as circuits (spec decision).
- The logical operators are Bravyi et al.'s, written as polynomials on the 12 × 6 torus:
  - f = 1 + x + x² + x³ + x⁶ + x⁷ + x⁸ + x⁹ + (x + x⁵ + x⁷ + x¹¹)y³;
  - g = x + x²y + (1 + x)y² + x²y³ + y⁴;
  - h = 1 + (1 + x)y + y² + (1 + x)y³.

  X(f, 0) and X(g, h) are weight-12 logicals, checked in the scratch session of 2026-09-30.
- Operators measured:
  - "f" = X(f, 0), weight 12;
  - "gh" = X(g, h), weight 12;
  - "f+gh" = their product, weight 22 (the joint measurement a CNOT needs).
- **Deviations from the spec, recorded:**
  - Cross et al. publish no edge lists. Their construction is an algorithm (Section 3 of arXiv:2407.18393, mono-layer, L = 1), so the graph is built by that algorithm here and checked, not transcribed:
    - vertices: the logical's support;
    - hyperedges: the Z checks touching it;
    - one ancilla qubit per hyperedge;
    - Gauss-law checks per vertex;
    - gauge-fixing flux checks from the null space of the restricted check matrix.
  - Distance is exact by integer programming (HiGHS through scipy), not SAT. Both are exact; the ILP was prototyped at about 30 s per solve on the deformed code of X(f, 0).
  - "Which actions are logical permutations" depends on the basis. It is replaced by basis-free facts: the group order, the shifts acting trivially, and whether each family of translates spans a subspace the shifts keep.
- Noise is Bravyi et al.'s circuit model, as in `BbCode::memory_z`:
  - DEPOLARIZE2(p) after every CNOT;
  - DEPOLARIZE1(p) on every data or edge qubit idle in a tick;
  - X_ERROR/Z_ERROR(p) after preparations and before measurements;
  - no idle noise on ancillas.

## File structure

| File | Responsibility |
|---|---|
| `src/bb.rs` (modify) | Make `shift`, `SX`, `SZ` `pub(crate)`; add `invert`, `poly_x`, `gross_operator`. |
| `src/bb_auto.rs` (create) | `Automorphism`, `BbCode::data_map`, `BbCode::logical_action`, `automorphism_report`. |
| `src/bb_gauge.rs` (create) | `Gauging::new`, `Gauging::deformed`, `Gauging::edges_for`, minimum-weight flux basis. |
| `src/bb_circuit.rs` (create) | `Check`, `Tick`, `Cycle`, `Writer`; `memory_cycle`, `merged_cycle`; `memory`, `logical_measurement`. |
| `src/lib.rs` (modify) | Register the three modules. |
| `src/py_api.rs`, `stabilizer_qec.pyi` (modify) | `bb_automorphisms`, `bb_gauging`, `bb_memory_basis_circuit`, `bb_logical_measurement_circuit`. |
| `tools/gross_ops.py` (create) | `auto`, `distance`, `check`, `measure` commands. |
| `data/gross/automorphisms.json`, `gauging.json`, `logical.json` (create) | Results. |
| `.github/workflows/ci.yml`, `tools/smoke.py` (modify) | `gross_ops.py check --quick` in CI; smoke check 11. |
| `js/sections/gross.js`, `index.html`, `css/styles.css` (modify) | Figures 16 and 17; later figures renumber. |
| `README.md`, `tools/readme_tables.py`, `report/report.md`, `tools/report.py` (modify) | The section, generated tables, the report subsection. |

---

### Task 1: The weight-12 logicals and the automorphism group (J1)

**Files:**
- Modify: `src/bb.rs` (visibility; `invert`, `poly_x`, `gross_operator`)
- Create: `src/bb_auto.rs`
- Modify: `src/lib.rs` (`pub mod bb_auto;`)

**Interfaces:**
- Produces:
  - `BbCode::poly_x(&self, left: &[Monomial], right: &[Monomial]) -> Vec<usize>`: the support of X(left, right), data indices 0..2ℓm.
  - `pub fn gross_operator(name: &str) -> Result<Vec<usize>, String>`: "f", "gh" or "f+gh".
  - `pub struct Automorphism { pub shift: Monomial, pub dual: bool }`
  - `BbCode::data_map(&self, a: Automorphism) -> Vec<usize>`
  - `BbCode::logical_action(&self, a: Automorphism) -> Result<BitMatrix, String>`: 24 × 24; row i is the image of basis element i, where the basis is X̄₀..X̄₁₁ then Z̄₀..Z̄₁₁ from `logicals()`.
  - `pub struct AutoReport { pub order: usize, pub trivial_shifts: Vec<Monomial>, pub families_invariant: [bool; 2], pub classes: [usize; 2] }` and `BbCode::automorphism_report(&self) -> Result<AutoReport, String>`.

- [ ] **Step 1: Write the failing tests** (end of `src/bb_auto.rs`)

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::bb::gross_operator;

    fn omega(k: usize) -> BitMatrix {
        let mut o = BitMatrix::zeros(2 * k, 2 * k);
        for i in 0..k {
            o.set(i, k + i, true);
            o.set(k + i, i, true);
        }
        o
    }

    /// Bravyi et al.'s operators are logicals of weight 12 (22 for the product),
    /// commuting with every Z check and outside the X stabilizers.
    #[test]
    fn the_papers_operators_are_weight_twelve_logicals() {
        let code = BbCode::gross();
        let (hx, hz) = (code.hx(), code.hz());
        for (name, weight) in [("f", 12), ("gh", 12), ("f+gh", 22)] {
            let s = gross_operator(name).unwrap();
            assert_eq!(s.len(), weight, "{name}");
            let row = BitMatrix::from_rows(code.num_data(), &[s.clone()]);
            assert!(hz.mul(&row.transpose()).is_zero(), "{name} commutes with the Z checks");
            assert_eq!(hx.stack(&row).rank(), hx.rank() + 1, "{name} is not a stabilizer");
        }
    }

    /// Every shift, with and without the duality, maps the stabilizer group to
    /// itself (the duality exchanging X and Z), and acts on the 12 logical
    /// qubits symplectically, with every residual a stabilizer.
    #[test]
    fn all_144_maps_are_automorphisms_with_symplectic_actions() {
        let code = BbCode::gross();
        let (hx, hz) = (code.hx(), code.hz());
        let permute = |m: &BitMatrix, map: &[usize]| {
            let rows: Vec<Vec<usize>> = (0..m.rows).map(|r| m.row_ones(r).iter().map(|&q| map[q]).collect()).collect();
            BitMatrix::from_rows(m.cols, &rows)
        };
        for a in 0..code.l {
            for b in 0..code.m {
                for dual in [false, true] {
                    let auto = Automorphism { shift: (a, b), dual };
                    let map = code.data_map(auto);
                    let (px, pz) = (permute(&hx, &map), permute(&hz, &map));
                    let (to_x, to_z) = if dual { (&hz, &hx) } else { (&hx, &hz) };
                    assert_eq!(to_x.stack(&px).rank(), to_x.rank(), "{auto:?}: X checks");
                    assert_eq!(to_z.stack(&pz).rank(), to_z.rank(), "{auto:?}: Z checks");
                    let m = code.logical_action(auto).unwrap();
                    assert_eq!(m.mul(&omega(12)).mul(&m.transpose()), omega(12), "{auto:?}: symplectic");
                }
            }
        }
    }

    /// The identity map acts as the identity, and actions compose: shifting
    /// by x then by y is shifting by xy.
    #[test]
    fn actions_compose() {
        let code = BbCode::gross();
        let act = |a, b, dual| code.logical_action(Automorphism { shift: (a, b), dual }).unwrap();
        assert_eq!(act(0, 0, false), BitMatrix::identity(24));
        assert_eq!(act(1, 0, false).mul(&act(0, 1, false)), act(1, 1, false));
        assert_eq!(act(0, 0, true).mul(&act(0, 0, true)), BitMatrix::identity(24), "the duality is an involution");
    }

    /// The 144 actions are closed under composition, so they are the group.
    #[test]
    fn the_report_is_a_group() {
        let r = BbCode::gross().automorphism_report().unwrap();
        assert!(r.order >= 2 && 144 % r.order == 0, "order {} divides 144", r.order);
        assert_eq!(72 / r.trivial_shifts.len() * 2, r.order, "the order is the shifts' quotient, doubled by the duality");
    }
}
```

The last assertion encodes the expectation that the duality's action is not a shift's. If the computed group says otherwise, the test's message shows the order: replace the assertion with the computed fact and say so in the commit message.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test --release --no-default-features bb_auto`
Expected: compile errors (`gross_operator`, `Automorphism` not defined).

- [ ] **Step 3: Implement** in `src/bb.rs`:

  - `const SX`/`SZ` → `pub(crate) const`
  - `fn shift` → `pub(crate) fn shift`
  - add:

```rust
    /// The cell of g⁻¹: (−u mod ℓ, −v mod m).
    pub(crate) fn invert(&self, cell: usize) -> usize {
        let (u, v) = (cell / self.m, cell % self.m);
        ((self.l - u) % self.l) * self.m + (self.m - v) % self.m
    }

    /// The support of X(p, q): the monomials of p on the left data, of q on the right.
    pub fn poly_x(&self, left: &[Monomial], right: &[Monomial]) -> Vec<usize> {
        let h = self.half();
        let mut on = vec![false; 2 * h];
        for &mono in left {
            on[self.shift(mono, 0)] ^= true;
        }
        for &mono in right {
            on[h + self.shift(mono, 0)] ^= true;
        }
        (0..2 * h).filter(|&q| on[q]).collect()
    }
}

/// Bravyi et al.'s weight-12 X logicals of the gross code, X(f, 0) and X(g, h)
/// (Nature 627, 778, 2024), and their product.
pub fn gross_operator(name: &str) -> Result<Vec<usize>, String> {
    const F: [Monomial; 12] = [(0, 0), (1, 0), (2, 0), (3, 0), (6, 0), (7, 0), (8, 0), (9, 0), (1, 3), (5, 3), (7, 3), (11, 3)];
    const G: [Monomial; 6] = [(1, 0), (2, 1), (0, 2), (1, 2), (2, 3), (0, 4)];
    const H: [Monomial; 6] = [(0, 0), (0, 1), (1, 1), (0, 2), (0, 3), (1, 3)];
    let code = BbCode::gross();
    let xor = |a: Vec<usize>, b: Vec<usize>| {
        let mut on = vec![false; code.num_data()];
        for q in a.into_iter().chain(b) {
            on[q] ^= true;
        }
        (0..code.num_data()).filter(|&q| on[q]).collect::<Vec<_>>()
    };
    match name {
        "f" => Ok(code.poly_x(&F, &[])),
        "gh" => Ok(code.poly_x(&G, &H)),
        "f+gh" => Ok(xor(code.poly_x(&F, &[]), code.poly_x(&G, &H))),
        other => Err(format!("unknown gross-code operator '{other}' (f, gh or f+gh)")),
    }
}
```

(`poly_x` goes inside `impl BbCode`; move the closing brace accordingly.) Then `src/bb_auto.rs`:

```rust
//! Automorphisms of bivariate bicycle codes, and what they do to the logical
//! qubits, exactly.
//!
//! WHY THIS EXISTS
//! ---------------
//! A memory is half of a computer. The gross code's checks are polynomials
//! in two commuting shifts x and y, so every shift x^a y^b of both halves of
//! the data maps the code to itself. So does the ZX-duality of Bravyi et
//! al.: left cell g to right cell g⁻¹ and back, with X and Z exchanged. Each
//! map permutes qubits (with Hadamards for the duality), so it is a logical
//! Clifford gate at no cost in noise beyond the moves. This module computes
//! each one's action on the 12 logical qubits as a 24 × 24 symplectic matrix
//! over GF(2), and checks it: every residual is a stabilizer.

use std::collections::HashSet;

use crate::bb::{BbCode, Monomial};
use crate::gf2::BitMatrix;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Automorphism {
    pub shift: Monomial,
    /// Followed by the ZX-duality.
    pub dual: bool,
}

#[derive(Clone, Debug)]
pub struct AutoReport {
    /// Distinct logical actions among the 144 maps; they form a group.
    pub order: usize,
    /// Shifts that act on the logical qubits as the identity.
    pub trivial_shifts: Vec<Monomial>,
    /// Whether the translates of X(f, 0), and of X(g, h), span a subspace every shift keeps.
    pub families_invariant: [bool; 2],
    /// Distinct logical classes among the 72 translates of each.
    pub classes: [usize; 2],
}

fn parity(a: &[usize], b: &[usize]) -> bool {
    a.iter().filter(|q| b.contains(q)).count() % 2 == 1
}

impl BbCode {
    /// Where each data qubit goes: shifted by x^a y^b on both halves, then,
    /// for the duality, left g to right g⁻¹ and right g to left g⁻¹.
    pub fn data_map(&self, a: Automorphism) -> Vec<usize> {
        let h = self.half();
        (0..2 * h)
            .map(|q| {
                let (side, moved) = (q / h, self.shift(a.shift, q % h));
                if a.dual { (1 - side) * h + self.invert(moved) } else { side * h + moved }
            })
            .collect()
    }

    /// The map's action on the logical qubits: row i is the image of basis
    /// element i (X̄₀..X̄ₖ₋₁, then Z̄₀..Z̄ₖ₋₁) in that basis. An X-type image
    /// is read off by its overlaps with the Z logicals, a Z-type one by its
    /// overlaps with the X logicals; what is left must be a stabilizer.
    pub fn logical_action(&self, a: Automorphism) -> Result<BitMatrix, String> {
        let (lx, lz) = self.logicals();
        let (hx, hz) = (self.hx(), self.hz());
        let (rx, rz) = (hx.rank(), hz.rank());
        let k = lx.rows;
        let map = self.data_map(a);
        let mut out = BitMatrix::zeros(2 * k, 2 * k);
        for i in 0..2 * k {
            let (from, was_x) = if i < k { (lx.row_ones(i), true) } else { (lz.row_ones(i - k), false) };
            let image: Vec<usize> = from.iter().map(|&q| map[q]).collect();
            let is_x = was_x != a.dual;
            let (pair, basis, stabilizers, rank, offset) = if is_x { (&lz, &lx, &hx, rx, 0) } else { (&lx, &lz, &hz, rz, k) };
            let mut residual = BitMatrix::from_rows(self.num_data(), &[image.clone()]);
            for j in 0..k {
                if parity(&image, &pair.row_ones(j)) {
                    out.set(i, offset + j, true);
                    for q in basis.row_ones(j) {
                        residual.flip(0, q);
                    }
                }
            }
            if stabilizers.stack(&residual).rank() != rank {
                return Err(format!("{a:?}: the image of logical {i} is not a logical in the basis plus a stabilizer"));
            }
        }
        Ok(out)
    }

    pub fn automorphism_report(&self) -> Result<AutoReport, String> {
        let key = |m: &BitMatrix| (0..m.rows).map(|r| m.row_ones(r)).collect::<Vec<_>>();
        let mut actions = Vec::new();
        let mut trivial_shifts = Vec::new();
        for a in 0..self.l {
            for b in 0..self.m {
                for dual in [false, true] {
                    let m = self.logical_action(Automorphism { shift: (a, b), dual })?;
                    if !dual && m == BitMatrix::identity(m.rows) {
                        trivial_shifts.push((a, b));
                    }
                    actions.push(m);
                }
            }
        }
        let distinct: HashSet<Vec<Vec<usize>>> = actions.iter().map(key).collect();
        for x in &actions {
            for y in &actions {
                if !distinct.contains(&key(&x.mul(y))) {
                    return Err("the actions are not closed under composition".into());
                }
            }
        }
        let (hx, rank_hx) = (self.hx(), self.hx().rank());
        let family = |gen: &[usize]| {
            let rows: Vec<Vec<usize>> = (0..self.l)
                .flat_map(|a| (0..self.m).map(move |b| (a, b)))
                .map(|s| {
                    let map = self.data_map(Automorphism { shift: s, dual: false });
                    gen.iter().map(|&q| map[q]).collect()
                })
                .collect();
            BitMatrix::from_rows(self.num_data(), &rows)
        };
        let mut families_invariant = [false; 2];
        let mut classes = [0; 2];
        for (i, name) in ["f", "gh"].iter().enumerate() {
            let fam = family(&crate::bb::gross_operator(name)?);
            // Its span modulo the X stabilizers; the translates are all in it by construction.
            let span = hx.stack(&fam).rank() - rank_hx;
            families_invariant[i] = span < 12;
            // Distinct classes: translates equal modulo stabilizers count once.
            let mut reps: Vec<Vec<usize>> = Vec::new();
            for r in 0..fam.rows {
                let row = fam.row_ones(r);
                let same = reps.iter().any(|s| {
                    let mut diff = BitMatrix::from_rows(self.num_data(), &[row.clone()]);
                    for &q in s {
                        diff.flip(0, q);
                    }
                    hx.stack(&diff).rank() == rank_hx
                });
                if !same {
                    reps.push(row);
                }
            }
            classes[i] = reps.len();
        }
        Ok(AutoReport { order: distinct.len(), trivial_shifts, families_invariant, classes })
    }
}
```

`families_invariant[i]` is true when a family's span is a proper invariant subspace. Its span is shift-invariant by construction: a shift maps a translate to a translate. The informative number is the span's dimension, so `AutoReport` also carries it. Add `pub spans: [usize; 2]`, set `spans[i] = span`, and assert in `the_report_is_a_group` that `r.spans.iter().all(|&s| s >= 1 && s <= 12)`.

- [ ] **Step 4: Run the tests** — `cargo test --release --no-default-features bb_auto`, expected PASS. Record the report in the commit message: `cargo test --release --no-default-features bb_auto -- --nocapture` after adding `eprintln!("{r:?}")` to `the_report_is_a_group`.

- [ ] **Step 5: Commit** — `git add src/bb.rs src/bb_auto.rs src/lib.rs && git commit` with message `feat(gross): the automorphism group and its exact logical action`.

---

### Task 2: The gauging ancilla system and the deformed code

**Files:**
- Create: `src/bb_gauge.rs`; modify `src/lib.rs`.

**Interfaces:**
- Consumes: `gross_operator`, `BbCode::{hx, hz, num_data, logicals}`.
- Produces:
```rust
pub struct Gauging {
    pub support: Vec<usize>,      // V₀, sorted data indices
    pub edges: Vec<usize>,        // C₀: the Z checks touching V₀, sorted; edge qubit i belongs to Z check edges[i]
    pub incidence: Vec<Vec<usize>>, // per edge, its vertices (indices into support)
    pub gauss: Vec<Vec<usize>>,   // per vertex, its edges (indices into edges)
    pub flux: Vec<Vec<usize>>,    // a minimum-weight basis of the cycles, edge indices
}
impl Gauging {
    pub fn new(code: &BbCode, support: &[usize]) -> Result<Gauging, String>;
    pub fn deformed(&self, code: &BbCode) -> (BitMatrix, BitMatrix); // on num_data + edges.len() qubits; edge qubit i = num_data + i
    pub fn edges_for(&self, z: &[usize]) -> Result<Vec<usize>, String>; // edge set whose boundary on V₀ is z ∩ V₀
    pub fn ancillas(&self) -> usize; // edges + gauss + flux
}
```

- [ ] **Step 1: Failing tests** (in `src/bb_gauge.rs`)

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::bb::gross_operator;

    fn gauged(name: &str) -> (BbCode, Gauging, Vec<usize>) {
        let code = BbCode::gross();
        let l = gross_operator(name).unwrap();
        let g = Gauging::new(&code, &l).unwrap();
        (code, g, l)
    }

    /// The deformed code's checks commute, the Gauss-law checks multiply to
    /// the operator (every edge qubit twice or four times), the flux checks
    /// span the cycles, and it encodes 11 logical qubits: L is now a stabilizer.
    #[test]
    fn the_deformed_code_measures_the_operator() {
        for name in ["f", "gh", "f+gh"] {
            let (code, g, l) = gauged(name);
            let (hx, hz) = g.deformed(&code);
            assert!(hx.mul(&hz.transpose()).is_zero(), "{name}: the checks commute");
            let n = hx.cols;
            assert_eq!(n - hx.rank() - hz.rank(), 11, "{name}: 11 logical qubits while merged");
            let gauss_rows = code.half()..code.half() + g.support.len();
            let mut product = vec![false; n];
            for r in gauss_rows {
                for q in hx.row_ones(r) {
                    product[q] ^= true;
                }
            }
            let got: Vec<usize> = (0..n).filter(|&q| product[q]).collect();
            assert_eq!(got, l, "{name}: the Gauss-law checks multiply to L");
            let f = g.restricted(&code);
            assert_eq!(g.flux.len(), g.edges.len() - f.rank(), "{name}: one flux check per independent cycle");
        }
    }

    /// X(f, 0) and X(g, h) give a 3-regular graph on 12 vertices: 18 edges,
    /// 7 independent cycles, 37 ancilla qubits. The product gives hyperedges.
    #[test]
    fn the_ancilla_systems_have_the_expected_sizes() {
        for (name, edges, flux) in [("f", 18, 7), ("gh", 18, 7), ("f+gh", 31, 10)] {
            let (_, g, _) = gauged(name);
            assert_eq!((g.edges.len(), g.flux.len()), (edges, flux), "{name}");
            assert_eq!(g.ancillas(), edges + g.support.len() + flux);
        }
    }

    /// The flux basis is minimum-weight: no cycle of the basis can be
    /// swapped for a lighter one outside the span of the lighter ones.
    #[test]
    fn the_flux_basis_is_light() {
        let (_, g, _) = gauged("f");
        let w: Vec<usize> = g.flux.iter().map(Vec::len).collect();
        assert!(w.windows(2).all(|p| p[0] <= p[1]), "sorted by weight: {w:?}");
        assert!(*w.last().unwrap() <= 8, "cycles of length at most 8: {w:?}");
    }

    /// A Z logical commuting with L passes through the merge along an edge path
    /// whose boundary is its overlap with L's support.
    #[test]
    fn z_logicals_route_through_the_edges() {
        let (code, g, l) = gauged("f");
        let (_, lz) = code.logicals();
        for j in 0..lz.rows {
            let z = lz.row_ones(j);
            let overlap = z.iter().filter(|q| l.contains(q)).count();
            let got = g.edges_for(&z);
            assert_eq!(got.is_ok(), overlap % 2 == 0, "Z logical {j}: overlap {overlap}");
        }
    }
}
```

- [ ] **Step 2:** `cargo test --release --no-default-features bb_gauge` → fails to compile.

- [ ] **Step 3: Implement**

```rust
//! The gauging measurement of a logical operator on a bivariate bicycle code
//! (Williamson and Yoder, arXiv:2410.02213; mono-layer, as Cross, He, Rall
//! and Yoder build it for the gross code, arXiv:2407.18393, Section 3).
//!
//! To measure a logical X̄ with support V₀: every Z check touching V₀ (the
//! set C₀) meets it in an even number of qubits, so C₀ is a hypergraph on
//! V₀, a graph when every check meets it in two. One new qubit per edge,
//! prepared in |0⟩. A Gauss-law X check per vertex: its data qubit and its
//! edges' qubits; their product is X̄, since every edge has an even number
//! of ends. Each Z check of C₀ gains its edge qubit, so it commutes with
//! them. Flux Z checks on each cycle (a set of edges with no boundary)
//! fix the gauge. Measuring the Gauss-law checks measures X̄.

use crate::bb::BbCode;
use crate::gf2::BitMatrix;

pub struct Gauging { /* fields as in Interfaces */ }

impl Gauging {
    pub fn new(code: &BbCode, support: &[usize]) -> Result<Gauging, String> {
        let hz = code.hz();
        let mut support = support.to_vec();
        support.sort_unstable();
        let edges: Vec<usize> = (0..hz.rows).filter(|&c| hz.row_ones(c).iter().any(|q| support.contains(q))).collect();
        let incidence: Vec<Vec<usize>> = edges
            .iter()
            .map(|&c| hz.row_ones(c).iter().filter_map(|q| support.iter().position(|v| v == q)).collect())
            .collect();
        if let Some(i) = incidence.iter().position(|e| e.len() % 2 == 1) {
            return Err(format!("Z check {} meets the operator oddly: it is not a logical X", edges[i]));
        }
        let gauss = (0..support.len()).map(|v| (0..edges.len()).filter(|&e| incidence[e].contains(&v)).collect()).collect();
        let mut g = Gauging { support, edges, incidence, gauss, flux: Vec::new() };
        g.flux = g.min_weight_cycles()?;
        Ok(g)
    }

    /// F: edges × vertices, the Z checks of C₀ restricted to V₀.
    pub fn restricted(&self, _code: &BbCode) -> BitMatrix {
        BitMatrix::from_rows(self.support.len(), &self.incidence)
    }

    /// A minimum-weight basis of the cycles {s : sᵀF = 0}: every vector of
    /// the cycle space, lightest first, kept when independent of those kept
    /// (the greedy basis of a linear matroid is a minimum-weight one).
    fn min_weight_cycles(&self) -> Result<Vec<Vec<usize>>, String> {
        let f = BitMatrix::from_rows(self.support.len(), &self.incidence);
        let basis = f.transpose().kernel(); // rows: vectors over edges
        if basis.rows > 16 {
            return Err(format!("{} independent cycles is too many to enumerate", basis.rows));
        }
        let mut all: Vec<Vec<usize>> = (1u32..(1 << basis.rows))
            .map(|mask| {
                let mut on = vec![false; self.edges.len()];
                for r in (0..basis.rows).filter(|r| mask >> r & 1 == 1) {
                    for e in basis.row_ones(r) {
                        on[e] ^= true;
                    }
                }
                (0..self.edges.len()).filter(|&e| on[e]).collect()
            })
            .collect();
        all.sort_by(|a, b| a.len().cmp(&b.len()).then(a.cmp(b)));
        let mut kept: Vec<Vec<usize>> = Vec::new();
        for cand in all {
            let mut rows = kept.clone();
            rows.push(cand.clone());
            if BitMatrix::from_rows(self.edges.len(), &rows).rank() == rows.len() {
                kept.push(cand);
                if kept.len() == basis.rows {
                    break;
                }
            }
        }
        Ok(kept)
    }

    /// (H_X', H_Z') on num_data + edges qubits. H_X' rows: the code's X
    /// checks, then one Gauss-law check per vertex. H_Z' rows: the code's Z
    /// checks (those of C₀ with their edge qubit), then the flux checks.
    pub fn deformed(&self, code: &BbCode) -> (BitMatrix, BitMatrix) {
        let (hx, hz, n) = (code.hx(), code.hz(), code.num_data());
        let cols = n + self.edges.len();
        let mut x: Vec<Vec<usize>> = (0..hx.rows).map(|r| hx.row_ones(r)).collect();
        for (v, es) in self.gauss.iter().enumerate() {
            x.push(std::iter::once(self.support[v]).chain(es.iter().map(|e| n + e)).collect());
        }
        let mut z: Vec<Vec<usize>> = (0..hz.rows).map(|r| hz.row_ones(r)).collect();
        for (i, &c) in self.edges.iter().enumerate() {
            z[c].push(n + i);
        }
        for cycle in &self.flux {
            z.push(cycle.iter().map(|e| n + e).collect());
        }
        (BitMatrix::from_rows(cols, &x), BitMatrix::from_rows(cols, &z))
    }

    /// Edges whose ends, counted mod 2, are exactly z's qubits in V₀: a
    /// solution s of Fᵀ s = z|V₀.
    pub fn edges_for(&self, z: &[usize]) -> Result<Vec<usize>, String> {
        let (nv, ne) = (self.support.len(), self.edges.len());
        // Augmented [Fᵀ | b], row-reduced.
        let mut rows: Vec<Vec<usize>> = (0..nv).map(|v| (0..ne).filter(|&e| self.incidence[e].contains(&v)).collect()).collect();
        for (v, row) in rows.iter_mut().enumerate() {
            if z.contains(&self.support[v]) {
                row.push(ne);
            }
        }
        let mut m = BitMatrix::from_rows(ne + 1, &rows);
        let pivots = m.row_reduce();
        if pivots.contains(&ne) {
            return Err("no edge set has that boundary".into());
        }
        let mut s = Vec::new();
        for (r, &p) in pivots.iter().enumerate() {
            if m.get(r, ne) {
                s.push(p);
            }
        }
        s.sort_unstable();
        Ok(s)
    }

    pub fn ancillas(&self) -> usize {
        self.edges.len() + self.support.len() + self.flux.len()
    }
}
```

Check `BitMatrix::row_reduce`'s contract before relying on it: read `src/gf2.rs:149`. It must return pivot columns in row order, with reduced rows above and zero rows below. If it returns something else, adapt `edges_for`, which must set each free variable to 0 and read pivot variables from the last column.

- [ ] **Step 4:** `cargo test --release --no-default-features bb_gauge` → PASS. If the flux-weight assertion fails, print `w` and set the bound to the computed maximum. Report it in the commit message; it is a fact, not a target.

- [ ] **Step 5: Commit** `feat(gross): the gauging ancilla system and the deformed code`.

---

### Task 3: A cycle writer that reproduces the memory exactly

**Files:**
- Create: `src/bb_circuit.rs`; modify `src/lib.rs`.

**Interfaces:**
- Consumes: `BbCode::{neighbours, half, logicals}`, `SX`, `SZ`.
- Produces:
```rust
pub struct Check { pub key: usize, pub anc: u32, pub x_type: bool, pub support: Vec<u32>, pub pos: (f64, f64) }
#[derive(Default, Clone)] pub struct Tick { pub measure_z: Vec<usize>, pub prepare_x: Vec<usize>, pub cnots: Vec<(usize, u32)>, pub measure_x: Vec<usize>, pub prepare_z: Vec<usize> }
pub struct Cycle { pub checks: Vec<Check>, pub ticks: Vec<Tick> }
pub struct Writer { pub c: Vec<Instr>, /* … */ }
impl Writer {
    pub fn new(p: f64, detect: Basis, data: Vec<u32>) -> Writer;
    pub fn prepare(&mut self, basis: Basis, qubits: &[u32]);
    pub fn prepare_ancillas_z(&mut self, qubits: &[u32]); // R + X_ERROR, no freshness
    pub fn measure(&mut self, basis: Basis, qubits: &[u32]) -> Vec<usize>;
    pub fn cycle(&mut self, cyc: &Cycle) -> Vec<Option<usize>>;
    pub fn final_detectors(&mut self, cyc: &Cycle, basis: Basis, readout: &std::collections::HashMap<u32, usize>) -> Result<(), String>;
    pub fn observable(&mut self, index: u32, recs: &[usize]);
    pub fn add_data(&mut self, qubits: &[u32]); pub fn remove_data(&mut self, qubits: &[u32]);
}
pub fn memory_cycle(code: &BbCode) -> Cycle;
pub fn memory(code: &BbCode, basis: Basis, cycles: usize, p: f64) -> Circuit;
```
Qubit numbering is `memory_z`'s: X ancillas 0..h, data h..3h, Z ancillas 3h..4h. Check keys: X check c → c, Z check c → h + c.

- [ ] **Step 1: Failing tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::circuit::Circuit;
    use crate::dem::Dem;
    use crate::m2d::M2d;

    fn mechanisms(c: &Circuit) -> Vec<(Vec<u32>, u64, f64)> {
        let dem = Dem::from_circuit_undecomposed(c).unwrap();
        let mut v: Vec<_> = dem.mechanisms.iter().map(|m| (m.detectors.clone(), m.observables, m.p)).collect();
        v.sort_by(|a, b| (&a.0, a.1).cmp(&(&b.0, b.1)));
        v
    }

    /// The writer's Z-basis memory is the paper's, fault for fault: the same
    /// detectors in the same order, the same observables, the same priors.
    #[test]
    fn the_writers_memory_is_the_papers() {
        for code in [BbCode::bb72(), BbCode::gross()] {
            let ours = memory(&code, Basis::Z, 3, 0.001);
            let theirs = Circuit::parse(&code.memory_z(3, 0.001)).unwrap();
            let (a, b) = (mechanisms(&ours), mechanisms(&theirs));
            assert_eq!(a.len(), b.len(), "{}×{}", code.l, code.m);
            for (x, y) in a.iter().zip(&b) {
                assert_eq!((&x.0, x.1), (&y.0, y.1));
                assert!((x.2 - y.2).abs() <= 1e-12 * y.2.max(1e-300), "{x:?} vs {y:?}");
            }
        }
    }

    /// The X-basis memory is deterministic without noise, with X detectors
    /// from the first cycle and the 12 X logicals as observables.
    #[test]
    fn the_x_basis_memory_is_deterministic() {
        let code = BbCode::gross();
        let m = M2d::new(&memory(&code, Basis::X, 3, 0.0)).unwrap();
        assert_eq!((m.num_detectors, m.num_observables), (code.half() * 4, 12));
    }
}
```

- [ ] **Step 2:** `cargo test --release --no-default-features bb_circuit` → fails to compile.

- [ ] **Step 3: Implement.** Write the module doc, then the types above. Rules:
  - **Detectors.** Keyed by `Check::key`, `Last { rec, support, cycle }`. A check of type `detect` measured in cycle t compares with its last measurement only if that was in cycle t − 1. Its gained qubits, and kept qubits read and prepared again since, must be fresh in its basis; the kept ones' readings join. Lost qubits must have been measured in its basis since (record > last.rec), and their records join. With no usable last measurement, it is a detector alone iff every support qubit is fresh in its basis. `fresh` is cleared at the end of each cycle. This is `src/surgery.rs`'s rule, on cycles.
  - **One tick.** In order: `Tick`; the `measure_z` ancillas (X_ERROR(p), M), with their detectors emitted right after, in check order, coords (pos.0, pos.1, cycle); the `prepare_x` ancillas (RX, Z_ERROR(p)); the CNOTs (an X check's ancilla controls, a Z check's ancilla is the target), then DEPOLARIZE2(p) on them; DEPOLARIZE1(p) on every data qubit in play not in a CNOT this tick (skipped when empty); the `measure_x` ancillas (Z_ERROR(p), MX), with their detectors; the `prepare_z` ancillas (R, X_ERROR(p)). Noise instructions are omitted when p = 0, as `surgery.rs` does.
  - **`memory_cycle(code)`.** Eight ticks:
    - tick 0: `prepare_x` all X checks, CNOTs SZ[0];
    - ticks 1–5: SX[r] and SZ[r];
    - tick 6: `measure_z` all Z checks, CNOTs SX[6];
    - tick 7: `measure_x` all X checks, `prepare_z` all Z checks.

    Check positions: X check (2u, 2v), Z check (2u + 1, 2v + 1).
  - **`memory(code, basis, cycles, p)`.** QUBIT_COORDS as `memory_z` writes them, then:
    - prepare the data in `basis`;
    - `prepare_ancillas_z` on the Z ancillas;
    - `cycles` memory cycles;
    - `Tick`, and the data measured in `basis`;
    - final detectors on the checks of type `basis`: readout records of the support, the last record, and any lost qubits' records; coords (pos, cycles);
    - observables: `logicals()`'s rows of the basis type, over the readout.
  - Match `memory_z`'s instruction order where noise acts on the same qubit, so that the priors agree to 1e-12.

- [ ] **Step 4:** `cargo test --release --no-default-features bb_circuit` → PASS. If the mechanism test fails, print the first differing pair: the detector list names the cycle and the check, and the fix is the order of noise within that tick.

- [ ] **Step 5: Commit** `feat(gross): a cycle writer, equal to the paper's memory fault for fault`.

---

### Task 4: The logical-measurement experiment

**Files:** modify `src/bb_circuit.rs`.

**Interfaces:**
- Consumes: `Gauging`, `Writer`, `memory_cycle`.
- Produces:
```rust
pub fn merged_cycle(code: &BbCode, g: &Gauging) -> Cycle;
/// Data in `basis`; `pre` memory cycles; the edges prepared in |0⟩ and
/// `merged` cycles of the deformed code; the edges read in Z; `post` memory
/// cycles; the data read in `basis`. Observables, X basis: L0 the outcome
/// (the Gauss-law checks' first merged records), then the 12 X logicals.
/// Z basis: the 11 Z logicals that commute with L, each with its edge path's
/// split records.
pub fn logical_measurement(code: &BbCode, g: &Gauging, basis: Basis, pre: usize, merged: usize, post: usize, p: f64) -> Result<Circuit, String>;
```
Qubits: edges 4h..4h + E (edge i), Gauss ancillas 4h + E + v, flux ancillas 4h + E + V + j. Keys: Gauss vertex v → 2h + v, flux j → 2h + V + j.

- [ ] **Step 1: Failing tests**

```rust
    fn gauged(name: &str) -> (BbCode, Gauging) {
        let code = BbCode::gross();
        let g = Gauging::new(&code, &crate::bb::gross_operator(name).unwrap()).unwrap();
        (code, g)
    }

    /// Each tick of the merged cycle uses every qubit at most once.
    #[test]
    fn the_merged_cycle_is_a_schedule() {
        let (code, g) = gauged("f+gh");
        let cyc = merged_cycle(&code, &g);
        for (t, tick) in cyc.ticks.iter().enumerate() {
            let mut seen = std::collections::HashSet::new();
            for &(ch, q) in &tick.cnots {
                assert!(seen.insert(cyc.checks[ch].anc), "tick {t}: ancilla twice");
                assert!(seen.insert(q), "tick {t}: qubit {q} twice");
            }
        }
    }

    /// Without noise every detector and observable is deterministic, in both
    /// bases, for all three operators: the schedule measures what it should
    /// and the frames are right.
    #[test]
    fn the_logical_measurement_is_deterministic() {
        for name in ["f", "gh", "f+gh"] {
            let (code, g) = gauged(name);
            for (basis, obs) in [(Basis::X, 13), (Basis::Z, 11)] {
                let c = logical_measurement(&code, &g, basis, 1, 2, 1, 0.0).unwrap();
                let m = M2d::new(&c).unwrap_or_else(|e| panic!("{name} {basis:?}: {e}"));
                assert_eq!(m.num_observables, obs, "{name} {basis:?}");
            }
        }
    }

    /// With one merged cycle, one measurement error on a Gauss-law ancilla
    /// flips the outcome unseen; with two, no single fault flips any
    /// observable without a detector.
    #[test]
    fn the_outcome_needs_two_merged_cycles() {
        let (code, g) = gauged("f");
        let unseen = |merged| {
            let c = logical_measurement(&code, &g, Basis::X, 1, merged, 1, 0.001).unwrap();
            let dem = Dem::from_circuit_undecomposed(&c).unwrap();
            dem.mechanisms.iter().filter(|m| m.detectors.is_empty() && m.observables != 0).count()
        };
        assert!(unseen(1) > 0);
        assert_eq!(unseen(2), 0);
    }
```

- [ ] **Step 2:** run → fails to compile.

- [ ] **Step 3: Implement `merged_cycle`.** Checks:
  - the code's X checks (unchanged);
  - one Gauss-law X check per vertex: support = its data qubit, then its edge qubits; position (data pos) + (0.5, 0);
  - the code's Z checks, those of C₀ with their edge qubit appended;
  - one flux Z check per cycle: position (−1 − j, −1).

  Ticks:
  - ticks 0–5 as `memory_cycle`, with `prepare_x` also covering the Gauss-law ancillas at tick 0;
  - tick 6: SX[6], plus each C₀ Z check's CNOT with its edge qubit;
  - from tick 7, first-fit: every (flux check, edge) pair, plus each Gauss-law check's CNOT with its data qubit, and `measure_x` of the code's X checks at the end of tick 7;
  - the tick after the last flux CNOT: `measure_z` of every Z-type check (the code's and the flux), and from that tick, first-fit, every (Gauss-law check, edge) pair;
  - last tick: `measure_x` of the Gauss-law checks, `prepare_z` of every Z-type check.

  This order makes every X–Z pair sharing qubits agree:
  - Z checks meet data before the Gauss-law checks do (ticks ≤ 5 against ≥ 7);
  - C₀ checks meet their edge qubit at tick 6, before any Gauss-law check;
  - flux checks meet every edge before any Gauss-law check.

  In each case the count of "X before Z" is zero. The determinism test is the proof.

  `first_fit(pairs, from, cyc)` places each pair at the first tick ≥ `from` where neither its ancilla nor its qubit already has a CNOT, appending empty ticks as needed.

- [ ] **Step 4: Implement `logical_measurement`.**
  - QUBIT_COORDS for all qubits.
  - `detect = basis`; data in play = the code's data.
  - Prepare the data in `basis` and `prepare_ancillas_z` on the Z ancillas; run `pre` memory cycles.
  - `Tick`; `prepare(Z, edges)`; `prepare_ancillas_z(flux ancillas)`; `add_data(edges)`.
  - `merged` merged cycles. The outcome is the XOR of the Gauss-law checks' records in the first of them; error if `merged == 0` and basis is X.
  - `Tick`; `measure(Z, edges)` → split records; `remove_data(edges)`.
  - `post` memory cycles; `Tick`; data measured in `basis`; final detectors with `memory_cycle`'s checks.
  - Observables:
    - **X basis:** L0 = outcome, then Lx rows.
    - **Z basis:** span(Lz) ∩ L⊥. Pick the first row j₀ with odd overlap with L; each other row j becomes row j + [overlap odd]·row j₀. For each resulting z, the edges `g.edges_for(&z)`: readout records of z plus the split records of those edges.

- [ ] **Step 5:** `cargo test --release --no-default-features bb_circuit` → PASS. A nondeterministic detector names its coordinates, which name the check and the cycle; fix the schedule or the rule there.

- [ ] **Step 6: Commit** `feat(gross): the gauging measurement of a logical operator, as a circuit`.

---

### Task 5: Python bindings, and the independent checks

**Files:**
- Modify: `src/py_api.rs`, `stabilizer_qec.pyi`, `tools/smoke.py`, `.github/workflows/ci.yml`
- Create: `tools/gross_ops.py`, `data/gross/automorphisms.json`, `data/gross/gauging.json`

**Interfaces:**
- Produces (Python):
  - `bb_automorphisms(code: str) -> list[tuple[int, int, bool, list[list[int]]]]`: (a, b, dual, action rows as column lists).
  - `bb_gauging(code: str, operator: str) -> tuple[list[int], list[int], list[list[int]], list[list[int]], list[list[int]], list[list[int]]]`: (support, edges, gauss, flux, H_X' rows, H_Z' rows).
  - `bb_memory_basis_circuit(code: str, basis: str, cycles: int, p: float) -> str`
  - `bb_logical_measurement_circuit(code: str, operator: str, basis: str, pre: int, merged: int, post: int, p: float) -> str`

- [ ] **Step 1:** Add the four `#[pyfunction]`s: parse the code with `bb_code`, the basis with the existing basis helper, map `String` errors with `err`. Register them, and add `.pyi` stubs with docstrings. Rebuild with maturin (`--profile python`).

- [ ] **Step 2: `tools/gross_ops.py`** (module docstring lists the commands):
  - **`auto`.** An independent numpy implementation of the data maps and the logical actions: translate each basis row, read coefficients by overlap parity with the paired basis, and check the residual's rank with `ldpc.mod2.rank`. Compare bit for bit with `bb_automorphisms("gross")` for all 144 maps. Compute the group order by closure of the 144 matrices under multiplication mod 2. Write `data/gross/automorphisms.json`: order, trivial shifts, spans, classes, engine commit.
  - **`distance [--operator f|gh|f+gh|code] [--cap SECONDS]`.** Exact distance by `scipy.optimize.milp`: one ILP per dual logical j, minimize weight subject to `H z − 2 s = 0` and `⟨z, dual_j⟩ − 2 s' = 1`, binary z, integer s. The distance is the minimum over j, for each type (X and Z).
    - "code" is the plain gross code, the method's own check: it must give 12 and 12.
    - A solve that hits the cap is recorded as a lower bound (`"exact": false`).
    - Write `data/gross/gauging.json`: per operator, the sizes (edges, gauss, flux, ancillas), the flux weights, the degrees (max data-qubit degree in the merged Tanner graph), the distances and seconds.
  - **`check [--quick]`.**
    - For the X- and Z-basis logical measurement of "f" (quick) or all three (full), at pre = post = 1, merged = 2, p = 0.003: our error model against Stim's, mechanism for mechanism (the `mechanisms` helper of `tools/bb_check.py`, imported), and Stim's noiseless detectors all zero.
    - BP+OSD-CS7 against `ldpc`'s `BpOsdDecoder` on 64 (quick) or 256 shots sampled by Stim: identical corrections, as `tools/bb_check.py` checks the memory. Exit 1 on any difference.

- [ ] **Step 3:** Run `.venv-f/bin/python tools/gross_ops.py auto`, then `distance --operator code`, then `distance` for the three operators. Commit the two JSON files. If a deformed distance is below 12, that is a finding: record it, and report it in the site text in Task 7 as measured. Do not tune the construction to hide it.

- [ ] **Step 4:** Run `.venv-f/bin/python tools/gross_ops.py check`. Expected: every line `ok`.

- [ ] **Step 5: CI and smoke.**
  - `ci.yml`, after the lattice-surgery step: `- name: Gross-code logical measurement against Stim and ldpc` / `if: matrix.os == 'ubuntu-latest'` / `run: python tools/gross_ops.py check --quick`.
  - `smoke.py` check 11: `bb_logical_measurement_circuit("gross", "f", "x", 1, 2, 1, 0.001)` parses in Stim with 13 observables, and 64 Stim shots decode with `decode_b8_bposd` without error.

- [ ] **Step 6: Commit** `feat(gross): automorphisms and the gauging measurement in Python, checked against ldpc and Stim, distance by ILP`.

---

### Task 6: The measurements

**Files:** `tools/gross_ops.py` (`measure`), `data/gross/logical.json`.

- [ ] **Step 1: Implement `measure`.**
  - Points:
    - operator "f", both bases, merged T ∈ {2, 3, 4, 7, 12}, pre = post = 6, p = 0.003;
    - operators "gh" (X basis) and "f+gh" (both bases) at T = 7;
    - the memory of the same total length (12 + T cycles) in each basis, from `bb_memory_basis_circuit`, as the reference.
  - Shots from `sample_b8_batch` in batches of 2,048, decoded by `decode_b8_bposd` (osd_cs, order 7, max_iter 10,000, as `tools/gross.py`).
  - Counts per point: shots; failures of any observable; in the X basis, the outcome alone and the logicals alone; Wilson intervals; seconds.
  - A point stops at 200 failures, a 400,000-shot cap, or a 1,800 s cap. Resumable: existing keys are skipped. Keys: `f/x/T7/p0.003`, `memory/x/R19/p0.003`, …

- [ ] **Step 2:** Run it in the background: `.venv-f/bin/python tools/gross_ops.py measure`, a few hours in total. Commit `data/gross/logical.json` with `data(gross): the logical measurement against T, beside the memory`.

---

### Task 7: The page, the README and the report

**Files:** `index.html`, `js/sections/gross.js`, `css/styles.css` (if needed), `README.md`, `tools/readme_tables.py`, `report/report.md`, `tools/report.py`, `tools/site-tests.mjs`.

- [ ] **Step 1: Figure 16, "measuring a logical operator".**
  - Reuse `drawTorus` (js/sections/gross.js:54) with an overlay drawn from `data/gross/gauging.json`:
    - the 12 data qubits of X(f, 0) filled in the defect colour;
    - each edge as a line between its two vertices through its Z check's position;
    - the flux cycles listed below.
  - A three-row table (f, gh, f+gh): ancilla qubits (edges + Gauss + flux), the largest flux check, and the deformed code's X and Z distance, marked exact or a bound.
  - A sentence on the automorphisms from `automorphisms.json`: the group's order, which shifts act trivially, the duality.
  - The JSON gains, per edge, its two vertices' cells, written by `distance` in Task 5, so the page needs no engine.
- [ ] **Step 2: Figure 17, "the logical measurement, measured".**
  - A plot of failure against T (X basis: any, and the outcome alone; Z basis: any), with the memory of the same length as a dashed line.
  - A table per point, and a foot note naming the machine and engine commit, as `runCnotRecorded` does.
- [ ] **Step 3: Renumber.** Surgery figures become 18–22 and the estimator's 23–24. Update the figure numbers in `index.html`, the doc comments in `js/sections/gross.js` and `js/sections/surgery.js`, and the README's module map lines naming figures.
- [ ] **Step 4: Site test.** In `tools/site-tests.mjs`, hold the drawing to the data: every edge joins two vertices of the operator's support, and every vertex has degree 3 for f and gh.
- [ ] **Step 5: README.** Add a subsection "Logical operations on the gross code" under the gross-code section:
  - the automorphism facts;
  - the construction;
  - the sizes and distances table;
  - the measurement table, generated: `readme_tables.py` gains `gross_logical` with header `| operator | basis | T | shots | any wrong | outcome wrong | memory, same length |`;
  - what was checked (Stim, `ldpc`, determinism, ILP);
  - the comparison with Cross et al.: 103 ancillas for eight operators against ours per operator; their 7 merged rounds at p = 0.1%.
- [ ] **Step 6: Report.** A subsection in the gross-code section with the same content. Values come from the JSON files through `tools/report.py` (`gl.*`), and the table through `readme_tables`. Rebuild with `python3 tools/report.py`.
- [ ] **Step 7: Verify.**
  - `node tools/site-tests.mjs`, `node tools/contrast.mjs` and `python3 tools/readme_tables.py --check`;
  - the report builds with no `{{`;
  - a headless check at 1440 and 390 px with no console errors and the figures drawn;
  - bump `?v=` in `index.html`.
- [ ] **Step 8: Commit** `feat(gross): logical operations on the page, in the README and in the report`.

### Task 8: Review, merge, push

- [ ] Run `/code-review --effort high master..feature/gross-logical`, fix what it finds, and re-report the outcomes.
- [ ] Run the full cargo test suite, `tools/gross_ops.py check`, `tools/smoke.py`, the wasm build and smoke, the site tests and the README check.
- [ ] Merge `--no-ff` to master, push, and confirm CI and the Pages deploy are green.
- [ ] Update the roadmap memory.

---

## Self-review

- **Spec coverage:**
  - J1: enumerate, check, and report (Task 1), with the Python cross-check (Task 5).
  - J2 algebra (Task 2), distance (Task 5), circuit (Tasks 3–4), and decoding and measurement (Task 6).
  - Verification 1–4 (Tasks 1, 2, 5, 6, 7).
  - Outputs (Task 7). The estimator hook is K's, which reads `data/gross/logical.json`.
  - Deviations are recorded under Global constraints.
- **Placeholders:** none. Numbers the tools measure are read from data, not typed in.
- **Types:** `Gauging` fields, and the `Writer`/`Cycle`/`Tick` names, are the same in Tasks 2–5; the Python names are the same in Tasks 5–7.

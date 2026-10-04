//! Automorphisms of bivariate bicycle codes, and what they do to the logical
//! qubits, exactly.
//!
//! WHY THIS EXISTS
//! ---------------
//! A memory is half of a computer. The gross code's checks are polynomials
//! in two commuting shifts x and y, so every shift x^a y^b of both halves of
//! the data maps the code to itself. So does the ZX-duality of Bravyi et
//! al.: left cell g to right cell g⁻¹ and back, with X and Z exchanged. Each
//! map moves qubits (with Hadamards for the duality), so it is a logical
//! Clifford gate that adds no checks. This module computes each one's action
//! on the 12 logical qubits as a 24 × 24 symplectic matrix over GF(2), and
//! checks it: whatever the basis does not account for is a stabilizer.

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
    /// Distinct logical actions among the 144 maps; they are closed under
    /// composition, so they are the group the automorphisms generate.
    pub order: usize,
    /// Shifts that act on the logical qubits as the identity.
    pub trivial_shifts: Vec<Monomial>,
    /// The dimension, modulo stabilizers, of the span of the 72 translates
    /// of X(f, 0), and of X(g, h): a subspace every shift keeps.
    pub spans: [usize; 2],
    /// Distinct logical classes among those 72 translates.
    pub classes: [usize; 2],
}

fn parity(a: &[usize], b: &[usize]) -> bool {
    a.iter().filter(|q| b.contains(q)).count() % 2 == 1
}

impl BbCode {
    /// Where each data qubit goes: shifted by x^a y^b on both halves, then,
    /// for the duality, left g to right g⁻¹ and right g to left g⁻¹. The
    /// duality maps the X check at g to the Z check at g⁻¹ and back.
    pub fn data_map(&self, a: Automorphism) -> Vec<usize> {
        let h = self.half();
        (0..2 * h)
            .map(|q| {
                let (side, moved) = (q / h, self.shift(a.shift, q % h));
                if a.dual {
                    (1 - side) * h + self.invert(moved)
                } else {
                    side * h + moved
                }
            })
            .collect()
    }

    /// The map's action on the logical qubits: row i is the image of basis
    /// element i (X̄₀..X̄ₖ₋₁, then Z̄₀..Z̄ₖ₋₁, from `logicals`) in that basis.
    /// An X-type image is read off by its overlaps with the Z logicals, a
    /// Z-type one by its overlaps with the X logicals; what is left must be a
    /// stabilizer, or the map is not an automorphism.
    pub fn logical_action(&self, a: Automorphism) -> Result<BitMatrix, String> {
        let (lx, lz) = self.logicals();
        let (hx, hz) = (self.hx(), self.hz());
        let (rx, rz) = (hx.rank(), hz.rank());
        let k = lx.rows;
        let map = self.data_map(a);
        let mut out = BitMatrix::zeros(2 * k, 2 * k);
        for i in 0..2 * k {
            let (from, was_x) = if i < k {
                (lx.row_ones(i), true)
            } else {
                (lz.row_ones(i - k), false)
            };
            let image: Vec<usize> = from.iter().map(|&q| map[q]).collect();
            let is_x = was_x != a.dual;
            let (pair, basis, stabilizers, rank, offset) = if is_x {
                (&lz, &lx, &hx, rx, 0)
            } else {
                (&lx, &lz, &hz, rz, k)
            };
            let mut residual = BitMatrix::from_rows(self.num_data(), std::slice::from_ref(&image));
            for j in 0..k {
                if parity(&image, &pair.row_ones(j)) {
                    out.set(i, offset + j, true);
                    for q in basis.row_ones(j) {
                        residual.flip(0, q);
                    }
                }
            }
            if stabilizers.stack(&residual).rank() != rank {
                return Err(format!(
                    "{a:?}: logical {i}'s image is not the basis's plus a stabilizer"
                ));
            }
        }
        Ok(out)
    }

    /// The group the 144 maps generate, and how the paper's two families of
    /// logical operators sit in it.
    pub fn automorphism_report(&self) -> Result<AutoReport, String> {
        let key = |m: &BitMatrix| (0..m.rows).map(|r| m.row_ones(r)).collect::<Vec<_>>();
        let mut actions = Vec::new();
        let mut trivial_shifts = Vec::new();
        for a in 0..self.l {
            for b in 0..self.m {
                for dual in [false, true] {
                    let m = self.logical_action(Automorphism {
                        shift: (a, b),
                        dual,
                    })?;
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
        let hx = self.hx();
        let rank_hx = hx.rank();
        let mut spans = [0; 2];
        let mut classes = [0; 2];
        for (i, name) in ["f", "gh"].iter().enumerate() {
            let op = crate::bb::gross_operator(name)?;
            let translates: Vec<Vec<usize>> = (0..self.l)
                .flat_map(|a| (0..self.m).map(move |b| (a, b)))
                .map(|s| {
                    let map = self.data_map(Automorphism {
                        shift: s,
                        dual: false,
                    });
                    op.iter().map(|&q| map[q]).collect()
                })
                .collect();
            spans[i] = hx
                .stack(&BitMatrix::from_rows(self.num_data(), &translates))
                .rank()
                - rank_hx;
            // Translates equal modulo the X stabilizers count once.
            let mut reps: Vec<&Vec<usize>> = Vec::new();
            for t in &translates {
                let same = reps.iter().any(|r| {
                    let mut diff = BitMatrix::from_rows(self.num_data(), std::slice::from_ref(t));
                    for &q in r.iter() {
                        diff.flip(0, q);
                    }
                    hx.stack(&diff).rank() == rank_hx
                });
                if !same {
                    reps.push(t);
                }
            }
            classes[i] = reps.len();
        }
        Ok(AutoReport {
            order: distinct.len(),
            trivial_shifts,
            spans,
            classes,
        })
    }
}

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

    /// Bravyi et al.'s operators are logicals of weight 12 (22 for the
    /// product), commuting with every Z check and outside the X stabilizers.
    #[test]
    fn the_papers_operators_are_weight_twelve_logicals() {
        let code = BbCode::gross();
        let (hx, hz) = (code.hx(), code.hz());
        for (name, weight) in [("f", 12), ("gh", 12), ("f+gh", 22)] {
            let s = gross_operator(name).unwrap();
            assert_eq!(s.len(), weight, "{name}");
            let row = BitMatrix::from_rows(code.num_data(), std::slice::from_ref(&s));
            assert!(
                hz.mul(&row.transpose()).is_zero(),
                "{name} commutes with the Z checks"
            );
            assert_eq!(
                hx.stack(&row).rank(),
                hx.rank() + 1,
                "{name} is not a stabilizer"
            );
        }
    }

    /// Every shift, with and without the duality, maps the stabilizer group
    /// to itself (the duality exchanging X and Z), and acts on the 12 logical
    /// qubits symplectically, every residual a stabilizer.
    #[test]
    fn all_144_maps_are_automorphisms_with_symplectic_actions() {
        let code = BbCode::gross();
        let (hx, hz) = (code.hx(), code.hz());
        let permute = |m: &BitMatrix, map: &[usize]| {
            let rows: Vec<Vec<usize>> = (0..m.rows)
                .map(|r| m.row_ones(r).iter().map(|&q| map[q]).collect())
                .collect();
            BitMatrix::from_rows(m.cols, &rows)
        };
        for a in 0..code.l {
            for b in 0..code.m {
                for dual in [false, true] {
                    let auto = Automorphism {
                        shift: (a, b),
                        dual,
                    };
                    let map = code.data_map(auto);
                    let (px, pz) = (permute(&hx, &map), permute(&hz, &map));
                    let (to_x, to_z) = if dual { (&hz, &hx) } else { (&hx, &hz) };
                    assert_eq!(to_x.stack(&px).rank(), to_x.rank(), "{auto:?}: X checks");
                    assert_eq!(to_z.stack(&pz).rank(), to_z.rank(), "{auto:?}: Z checks");
                    let m = code.logical_action(auto).unwrap();
                    assert_eq!(
                        m.mul(&omega(12)).mul(&m.transpose()),
                        omega(12),
                        "{auto:?}: symplectic"
                    );
                }
            }
        }
    }

    /// The identity acts as the identity, and actions compose: x then y is xy.
    #[test]
    fn actions_compose() {
        let code = BbCode::gross();
        let act = |a, b, dual| {
            code.logical_action(Automorphism {
                shift: (a, b),
                dual,
            })
            .unwrap()
        };
        assert_eq!(act(0, 0, false), BitMatrix::identity(24));
        assert_eq!(act(1, 0, false).mul(&act(0, 1, false)), act(1, 1, false));
        assert_eq!(
            act(0, 0, true).mul(&act(0, 0, true)),
            BitMatrix::identity(24),
            "the duality is an involution"
        );
    }

    /// The 144 actions are closed under composition, so they are the group.
    #[test]
    fn the_report_is_a_group() {
        let r = BbCode::gross().automorphism_report().unwrap();
        eprintln!("{r:?}");
        assert!(
            r.order >= 2 && 144 % r.order == 0,
            "order {} divides 144",
            r.order
        );
        assert_eq!(
            72 / r.trivial_shifts.len() * 2,
            r.order,
            "the shifts' quotient, doubled by the duality"
        );
        assert!(
            r.spans.iter().all(|&s| (1..=12).contains(&s)),
            "{:?}",
            r.spans
        );
    }
}

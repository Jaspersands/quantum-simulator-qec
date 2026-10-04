//! Bivariate bicycle codes (Bravyi, Cross, Gambetta, Maslov, Rall and Yoder,
//! Nature 627, 778, 2024), and the paper's depth-8 syndrome cycle as a circuit.
//!
//! WHY THIS EXISTS
//! ---------------
//! A surface code spends about d² qubits on each logical qubit. The gross code
//! [[144, 12, 12]] keeps twelve at distance 12 on 144 data qubits and 144 check
//! qubits; twelve distance-12 surface codes would need nearly 3,500. Its checks
//! are weight six and each fault sets off several at once, so it cannot be
//! matched: it is decoded by BP+OSD (`osd`). This module builds the code, its
//! logical operators, and the memory experiment the paper simulates, as a
//! circuit this engine's error-model builder and samplers already understand.
//!
//! The code lives on an ℓ × m torus. x and y are the cyclic shifts along its
//! two directions, A and B sums of three monomials in them, and
//! H_X = [A | B], H_Z = [Bᵀ | Aᵀ], which commute because A and B do.

use crate::gf2::{complement_rows, BitMatrix};

/// A monomial x^i y^j.
pub type Monomial = (usize, usize);

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BbCode {
    pub l: usize,
    pub m: usize,
    pub a: [Monomial; 3],
    pub b: [Monomial; 3],
}

/// The per-round order in which each check qubit meets its six neighbours
/// (0-2: A₁-A₃ on the left data; 3-5: B₁-B₃ on the right), from the paper's
/// published simulation code. `None` is a round the check spends idle.
pub(crate) const SX: [Option<usize>; 7] = [None, Some(1), Some(4), Some(3), Some(5), Some(0), Some(2)];
pub(crate) const SZ: [Option<usize>; 7] = [Some(3), Some(5), Some(0), Some(1), Some(2), Some(4), None];


/// A bivariate bicycle circuit is written out cycle by cycle: at least `least` cycles of the
/// part that needs them, and no more in all (`total`) than a flat memory's rounds.
pub fn check_cycles(total: usize, least: usize) -> Result<(), String> {
    let most = crate::memory::MAX_FLAT_ROUNDS;
    if least == 0 {
        Err("a memory needs at least one cycle".into())
    } else if total > most {
        Err(format!("{total} cycles: at most {most} are written out cycle by cycle"))
    } else {
        Ok(())
    }
}

impl BbCode {
    /// The gross code, [[144, 12, 12]]: A = x³ + y + y², B = y³ + x + x² on a 12 × 6 torus.
    pub fn gross() -> BbCode {
        BbCode { l: 12, m: 6, a: [(3, 0), (0, 1), (0, 2)], b: [(0, 3), (1, 0), (2, 0)] }
    }

    /// [[72, 12, 6]]: the same polynomials on a 6 × 6 torus.
    pub fn bb72() -> BbCode {
        BbCode { l: 6, m: 6, a: [(3, 0), (0, 1), (0, 2)], b: [(0, 3), (1, 0), (2, 0)] }
    }

    /// [[90, 8, 10]]: A = x⁹ + y + y², B = 1 + x² + x⁷ on a 15 × 3 torus.
    pub fn bb90() -> BbCode {
        BbCode { l: 15, m: 3, a: [(9, 0), (0, 1), (0, 2)], b: [(0, 0), (2, 0), (7, 0)] }
    }

    /// [[108, 8, 10]]: the gross code's polynomials on a 9 × 6 torus.
    pub fn bb108() -> BbCode {
        BbCode { l: 9, m: 6, a: [(3, 0), (0, 1), (0, 2)], b: [(0, 3), (1, 0), (2, 0)] }
    }

    /// [[288, 12, 18]]: A = x³ + y² + y⁷, B = y³ + x + x² on a 12 × 12 torus.
    pub fn bb288() -> BbCode {
        BbCode { l: 12, m: 12, a: [(3, 0), (0, 2), (0, 7)], b: [(0, 3), (1, 0), (2, 0)] }
    }

    /// The codes of Bravyi et al.'s Table 3 by their number of data qubits: "72", "90", "108",
    /// "144" (also "gross") and "288".
    pub fn named(name: &str) -> Result<BbCode, String> {
        match name {
            "72" => Ok(BbCode::bb72()),
            "90" => Ok(BbCode::bb90()),
            "108" => Ok(BbCode::bb108()),
            "144" | "gross" => Ok(BbCode::gross()),
            "288" => Ok(BbCode::bb288()),
            other => Err(format!("unknown bivariate bicycle code '{other}': 72, 90, 108, 144 (gross) or 288")),
        }
    }

    /// The code of A = Σ x^i y^j over `a` and B over `b` on an ℓ × m torus. Each polynomial's
    /// three monomials must differ (the checks are weight six, as the syndrome cycle needs),
    /// and the code must encode at least one qubit.
    pub fn new(l: usize, m: usize, a: [Monomial; 3], b: [Monomial; 3]) -> Result<BbCode, String> {
        if l == 0 || m == 0 || l * m > 4096 {
            return Err(format!("a {l} × {m} torus: each side at least 1, and at most 4096 cells"));
        }
        let reduce = |p: [Monomial; 3]| p.map(|(i, j)| (i % l, j % m));
        let (a, b) = (reduce(a), reduce(b));
        for (name, p) in [("A", a), ("B", b)] {
            if p[0] == p[1] || p[0] == p[2] || p[1] == p[2] {
                return Err(format!("{name}'s three monomials must differ on the {l} × {m} torus, not {p:?}"));
            }
        }
        let code = BbCode { l, m, a, b };
        let (hx, hz) = (code.hx(), code.hz());
        if hx.rank() + hz.rank() == code.num_data() {
            return Err("this code encodes no logical qubits".into());
        }
        Ok(code)
    }

    /// Qubits of one kind (checks of one type, or data on one side): ℓm.
    pub fn half(&self) -> usize {
        self.l * self.m
    }

    pub fn num_data(&self) -> usize {
        2 * self.half()
    }

    /// The column where row `r` of the monomial's permutation matrix is set.
    pub(crate) fn shift(&self, (i, j): Monomial, r: usize) -> usize {
        let (u, v) = (r / self.m, r % self.m);
        ((u + i) % self.l) * self.m + (v + j) % self.m
    }

    /// The cell of g⁻¹: (−u mod ℓ, −v mod m).
    pub(crate) fn invert(&self, cell: usize) -> usize {
        let (u, v) = (cell / self.m, cell % self.m);
        ((self.l - u) % self.l) * self.m + (self.m - v) % self.m
    }

    /// The support of X(p, q): the monomials of p on the left data, of q on
    /// the right (a monomial named twice cancels).
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

    /// The row where column `c` of the monomial's permutation matrix is set.
    fn unshift(&self, (i, j): Monomial, c: usize) -> usize {
        let (u, v) = (c / self.m, c % self.m);
        ((u + self.l - i % self.l) % self.l) * self.m + (v + self.m - j % self.m) % self.m
    }

    /// Check `c`'s six data neighbours, in the order `SX`/`SZ` index: for an
    /// X check, A₁ A₂ A₃ on the left then B₁ B₂ B₃ on the right; for a Z check,
    /// B₁ᵀ B₂ᵀ B₃ᵀ on the left then A₁ᵀ A₂ᵀ A₃ᵀ on the right. Data qubits are
    /// numbered left then right.
    pub fn neighbours(&self, c: usize, x_check: bool) -> [usize; 6] {
        let h = self.half();
        let mut out = [0; 6];
        for k in 0..3 {
            if x_check {
                out[k] = self.shift(self.a[k], c);
                out[k + 3] = h + self.shift(self.b[k], c);
            } else {
                out[k] = self.unshift(self.b[k], c);
                out[k + 3] = h + self.unshift(self.a[k], c);
            }
        }
        out
    }

    pub fn hx(&self) -> BitMatrix {
        let rows: Vec<Vec<usize>> = (0..self.half()).map(|c| self.neighbours(c, true).to_vec()).collect();
        BitMatrix::from_rows(self.num_data(), &rows)
    }

    pub fn hz(&self) -> BitMatrix {
        let rows: Vec<Vec<usize>> = (0..self.half()).map(|c| self.neighbours(c, false).to_vec()).collect();
        BitMatrix::from_rows(self.num_data(), &rows)
    }

    /// Logical operators (X, Z), one per row, paired so that row i of X
    /// anticommutes with row i of Z and commutes with every other.
    pub fn logicals(&self) -> (BitMatrix, BitMatrix) {
        let (hx, hz) = (self.hx(), self.hz());
        // Z logicals commute with every X check, and are not Z stabilizers.
        let lz = complement_rows(&hz, &hx.kernel());
        let lx = complement_rows(&hx, &hz.kernel());
        // Pair them: with P = Lx Lzᵀ, replacing Lx by P⁻¹ Lx makes the pairing the identity.
        let p = lx.mul(&lz.transpose());
        let inv = p.inverse().expect("logical X and Z operators pair nondegenerately");
        (inv.mul(&lx), lz)
    }

    /// The paper's Z-basis memory experiment as a circuit in Stim's text:
    /// data and checks prepared in |0⟩, `cycles` rounds of the depth-8
    /// syndrome cycle, the data measured. Noise is the paper's, one parameter
    /// `p`: depolarizing after every CNOT and on every idle data qubit,
    /// preparation and measurement flips. Detectors compare Z checks round to
    /// round and with the final readout; the observables are the code's logical
    /// Z operators.
    pub fn memory_z(&self, cycles: usize, p: f64) -> String {
        let h = self.half();
        // Qubits: X checks, left data, right data, Z checks, as the paper orders them.
        let xq = |c: usize| c;
        let dq = |q: usize| h + q;
        let zq = |c: usize| 3 * h + c;
        let xn: Vec<[usize; 6]> = (0..h).map(|c| self.neighbours(c, true)).collect();
        let zn: Vec<[usize; 6]> = (0..h).map(|c| self.neighbours(c, false)).collect();
        let list = |v: &[usize]| v.iter().map(|q| q.to_string()).collect::<Vec<_>>().join(" ");
        let all_data: Vec<usize> = (0..2 * h).map(dq).collect();
        let x_checks: Vec<usize> = (0..h).map(xq).collect();
        let z_checks: Vec<usize> = (0..h).map(zq).collect();

        let mut out = String::new();
        let mut line = |s: String| {
            out.push_str(&s);
            out.push('\n');
        };
        for c in 0..h {
            let (u, v) = (c / self.m, c % self.m);
            line(format!("QUBIT_COORDS({}, {}) {}", 2 * u, 2 * v, xq(c)));
            line(format!("QUBIT_COORDS({}, {}) {}", 2 * u + 1, 2 * v, dq(c)));
            line(format!("QUBIT_COORDS({}, {}) {}", 2 * u, 2 * v + 1, dq(h + c)));
            line(format!("QUBIT_COORDS({}, {}) {}", 2 * u + 1, 2 * v + 1, zq(c)));
        }
        line(format!("R {}", list(&all_data)));
        line(format!("X_ERROR({p}) {}", list(&all_data)));
        line(format!("R {}", list(&z_checks)));
        line(format!("X_ERROR({p}) {}", list(&z_checks)));
        let mut measured = 0usize;
        let mut last_z: Option<usize> = None;
        for cycle in 0..cycles {
            for round in 0..8 {
                line("TICK".into());
                let mut touched = vec![false; 2 * h];
                let mut pairs: Vec<(usize, usize)> = Vec::new();
                if round == 0 {
                    line(format!("RX {}", list(&x_checks)));
                    line(format!("Z_ERROR({p}) {}", list(&x_checks)));
                }
                if round < 7 {
                    if let Some(k) = SX[round] {
                        for c in 0..h {
                            pairs.push((xq(c), dq(xn[c][k])));
                            touched[xn[c][k]] = true;
                        }
                    }
                    if let Some(k) = SZ[round] {
                        for c in 0..h {
                            pairs.push((dq(zn[c][k]), zq(c)));
                            touched[zn[c][k]] = true;
                        }
                    }
                }
                if round == 6 {
                    line(format!("X_ERROR({p}) {}", list(&z_checks)));
                    line(format!("M {}", list(&z_checks)));
                    let first = measured;
                    measured += h;
                    for c in 0..h {
                        let now = measured - (first + c);
                        let (u, v) = (c / self.m, c % self.m);
                        match last_z {
                            None => line(format!("DETECTOR({}, {}, {cycle}) rec[-{now}]", 2 * u + 1, 2 * v + 1)),
                            Some(prev) => line(format!(
                                "DETECTOR({}, {}, {cycle}) rec[-{now}] rec[-{}]",
                                2 * u + 1,
                                2 * v + 1,
                                measured - (prev + c)
                            )),
                        }
                    }
                    last_z = Some(first);
                }
                if !pairs.is_empty() {
                    let flat: Vec<usize> = pairs.iter().flat_map(|&(a, b)| [a, b]).collect();
                    line(format!("CX {}", list(&flat)));
                    line(format!("DEPOLARIZE2({p}) {}", list(&flat)));
                }
                if round == 0 || round >= 6 {
                    let idle: Vec<usize> = (0..2 * h).filter(|&q| !touched[q]).map(dq).collect();
                    if !idle.is_empty() {
                        line(format!("DEPOLARIZE1({p}) {}", list(&idle)));
                    }
                }
                if round == 7 {
                    line(format!("Z_ERROR({p}) {}", list(&x_checks)));
                    line(format!("MX {}", list(&x_checks)));
                    measured += h;
                    line(format!("R {}", list(&z_checks)));
                    line(format!("X_ERROR({p}) {}", list(&z_checks)));
                }
            }
        }
        // The data, measured: each Z check's parity closes its last comparison,
        // and each logical Z operator's parity is an observable.
        line("TICK".into());
        line(format!("X_ERROR({p}) {}", list(&all_data)));
        line(format!("M {}", list(&all_data)));
        let data_start = measured;
        measured += 2 * h;
        let rec = |q: usize| measured - (data_start + q);
        let prev = last_z.expect("at least one cycle");
        for c in 0..h {
            let (u, v) = (c / self.m, c % self.m);
            let mut targets: Vec<String> = zn[c].iter().map(|&q| format!("rec[-{}]", rec(q))).collect();
            targets.push(format!("rec[-{}]", measured - (prev + c)));
            line(format!("DETECTOR({}, {}, {cycles}) {}", 2 * u + 1, 2 * v + 1, targets.join(" ")));
        }
        let (_, lz) = self.logicals();
        for k in 0..lz.rows {
            let targets: Vec<String> = lz.row_ones(k).iter().map(|&q| format!("rec[-{}]", rec(q))).collect();
            line(format!("OBSERVABLE_INCLUDE({k}) {}", targets.join(" ")));
        }
        out
    }
}

/// Bravyi et al.'s weight-12 X logicals of the gross code, X(f, 0) and
/// X(g, h) (Nature 627, 778, 2024, "logical operators"), and their product
/// "f+gh", the joint measurement a CNOT between them needs. Supports as data
/// indices (left 0..72, right 72..144).
pub fn gross_operator(name: &str) -> Result<Vec<usize>, String> {
    // f = 1 + x + x² + x³ + x⁶ + x⁷ + x⁸ + x⁹ + (x + x⁵ + x⁷ + x¹¹)y³
    const F: [Monomial; 12] = [(0, 0), (1, 0), (2, 0), (3, 0), (6, 0), (7, 0), (8, 0), (9, 0), (1, 3), (5, 3), (7, 3), (11, 3)];
    // g = x + x²y + (1 + x)y² + x²y³ + y⁴
    const G: [Monomial; 6] = [(1, 0), (2, 1), (0, 2), (1, 2), (2, 3), (0, 4)];
    // h = 1 + (1 + x)y + y² + (1 + x)y³
    const H: [Monomial; 6] = [(0, 0), (0, 1), (1, 1), (0, 2), (0, 3), (1, 3)];
    let code = BbCode::gross();
    match name {
        "f" => Ok(code.poly_x(&F, &[])),
        "gh" => Ok(code.poly_x(&G, &H)),
        "f+gh" => {
            let mut on = vec![false; code.num_data()];
            for q in code.poly_x(&F, &[]).into_iter().chain(code.poly_x(&G, &H)) {
                on[q] ^= true;
            }
            Ok((0..code.num_data()).filter(|&q| on[q]).collect())
        }
        other => Err(format!("unknown gross-code operator '{other}' (f, gh or f+gh)")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::circuit::Circuit;
    use crate::dem::Dem;
    use crate::m2d::M2d;

    #[test]
    fn the_checks_commute_and_encode_twelve_qubits() {
        for code in [BbCode::gross(), BbCode::bb72()] {
            let (hx, hz) = (code.hx(), code.hz());
            assert!(hx.mul(&hz.transpose()).is_zero(), "H_X H_Zᵀ = 0");
            let n = code.num_data();
            let k = n - hx.rank() - hz.rank();
            assert_eq!(k, 12, "{}×{}: k", code.l, code.m);
            for r in 0..hx.rows {
                assert_eq!(hx.row_ones(r).len(), 6, "weight-six checks");
            }
        }
    }

    #[test]
    fn logical_operators_pair_and_commute_with_the_checks() {
        for code in [BbCode::gross(), BbCode::bb72()] {
            let (lx, lz) = code.logicals();
            assert_eq!((lx.rows, lz.rows), (12, 12));
            assert_eq!(lx.mul(&lz.transpose()), BitMatrix::identity(12));
            assert!(code.hx().mul(&lz.transpose()).is_zero(), "Z logicals commute with X checks");
            assert!(code.hz().mul(&lx.transpose()).is_zero(), "X logicals commute with Z checks");
            // Not stabilizers: each adds to the rank of the checks of its type.
            assert_eq!(code.hz().stack(&lz).rank(), code.hz().rank() + 12);
            assert_eq!(code.hx().stack(&lx).rank(), code.hx().rank() + 12);
        }
    }

    /// Each syndrome-cycle round meets every data qubit at most once, and
    /// the middle five meet every data qubit exactly once, which is what
    /// makes the cycle depth eight.
    #[test]
    fn every_round_uses_each_data_qubit_at_most_once() {
        let code = BbCode::gross();
        let h = code.half();
        for round in 0..7 {
            let mut count = vec![0; 2 * h];
            for c in 0..h {
                if let Some(k) = SX[round] {
                    count[code.neighbours(c, true)[k]] += 1;
                }
                if let Some(k) = SZ[round] {
                    count[code.neighbours(c, false)[k]] += 1;
                }
            }
            assert!(count.iter().all(|&n| n <= 1), "round {round}");
            if (1..6).contains(&round) {
                assert!(count.iter().all(|&n| n == 1), "round {round} uses every data qubit");
            }
        }
    }

    /// The circuit parses, and without noise every detector is deterministic
    /// (the m2d references refuse any that is not) and every observable is too.
    #[test]
    fn the_memory_circuit_measures_the_stabilizers() {
        for code in [BbCode::bb72(), BbCode::gross()] {
            let text = code.memory_z(3, 0.0);
            let circuit = Circuit::parse(&text).unwrap();
            let m = M2d::new(&circuit).unwrap();
            assert_eq!(m.num_detectors, code.half() * 4);
            assert_eq!(m.num_observables, 12);
        }
    }

    /// With noise, the error model has every detector and observable, and
    /// every fault's symptom is consistent: a fault flipping an observable
    /// flips some detector (no undetectable logical from a single fault).
    #[test]
    fn no_single_fault_is_an_undetectable_logical_error() {
        let code = BbCode::bb72();
        let circuit = Circuit::parse(&code.memory_z(3, 0.001)).unwrap();
        let dem = Dem::from_circuit_undecomposed(&circuit).unwrap();
        assert_eq!(dem.num_observables, 12);
        for m in &dem.mechanisms {
            assert!(!(m.detectors.is_empty() && m.observables != 0), "a single fault flips a logical undetected");
        }
    }
}

//! Memory belief propagation over GF(4) (MBP): `ldpc`'s `mbp_decoder` (src_cpp/mbp.hpp),
//! after Kuo and Lai, "Exploiting degeneracy in belief propagation decoding of quantum codes"
//! (npj Quantum Information 8, 111, 2022).
//!
//! WHY THIS EXISTS
//! ---------------
//! Binary BP decodes a quantum code's X and Z errors apart, so it cannot see that a Y error
//! flips both; quaternary BP keeps the three Pauli errors of each qubit together. Each check
//! sends each qubit one number (the check's belief that the qubit's error anticommutes with
//! it), and each qubit holds three log-likelihood ratios, one per Pauli. MBP scales the
//! messages a qubit adds up by `1/alpha` before deciding but subtracts only the unscaled
//! message it sends back (its "memory"), which with `alpha < 1` sharpens beliefs and helps BP
//! choose among degenerate errors. `beta` weighs the messages of checks that commute with a
//! Pauli (0 in Kuo and Lai), and min-sum's messages are scaled by `gamma`.
//!
//! `ldpc` 2.4.1 ships the Python half of this decoder but not the C++ (`mbp.hpp` is on its main
//! branch, written for a sparse-matrix interface it has since replaced, so it cannot be built).
//! This is that C++, step for step: a serial schedule over the qubits in order, product-sum
//! with its numerical guards (kept as they are, including a clamp that turns a message of -1
//! into NaN) or min-sum, and its hard decision and stopping rule. A reference program built
//! from `mbp.hpp` gives the same decodings and log-likelihood ratios bit for bit
//! (`tests/test_mbp.py`, with its recorded outputs).

/// One nonzero of the check matrix.
#[derive(Clone, Debug)]
struct Entry {
    row: usize,
    col: usize,
    /// 1 = X, 2 = Y, 3 = Z.
    pauli: u8,
    to_check: [f64; 3],
    to_qubit: f64,
}

/// The decoder and its state (which carries over between decodes, as `ldpc`'s does).
#[derive(Clone, Debug)]
pub struct Mbp {
    num_checks: usize,
    num_qubits: usize,
    entries: Vec<Entry>,
    /// Each qubit's entries, by row.
    by_col: Vec<Vec<usize>>,
    /// Each check's entries, by column.
    by_row: Vec<Vec<usize>>,
    /// `channel[w][q]`: the probability of Pauli `w + 1` on qubit `q`.
    pub channel: [Vec<f64>; 3],
    pub alpha: [Vec<f64>; 3],
    pub beta: f64,
    pub gamma: f64,
    pub min_sum: bool,
    pub max_iter: usize,
    /// 0 = I, 1 = X, 2 = Y, 3 = Z per qubit.
    pub decoding: Vec<u8>,
    pub log_prob_ratios: [Vec<f64>; 3],
    pub iterations: usize,
    pub converge: bool,
}

impl Mbp {
    /// `entries`: (check, qubit, Pauli 1–3) for each nonzero of the matrix.
    #[allow(clippy::too_many_arguments)]
    pub fn new(num_checks: usize, num_qubits: usize, entries: &[(usize, usize, u8)], channel: [Vec<f64>; 3], alpha: [Vec<f64>; 3], beta: f64, gamma: f64, min_sum: bool, max_iter: usize) -> Result<Mbp, String> {
        let mut es: Vec<(usize, usize, u8)> = Vec::with_capacity(entries.len());
        for &(r, c, p) in entries {
            if r >= num_checks || c >= num_qubits {
                return Err(format!("entry ({r}, {c}) is outside the {num_checks}x{num_qubits} matrix"));
            }
            if !(1..=3).contains(&p) {
                return Err(format!("entry ({r}, {c}) is {p}; a GF(4) check matrix holds 1 (X), 2 (Y) or 3 (Z)"));
            }
            es.push((r, c, p));
        }
        // ldpc's insert_entry keeps one entry per place, with the last value inserted there.
        es.sort_by_key(|&(r, c, _)| (r, c)); // stable: insertion order among equals
        let mut entries: Vec<Entry> = Vec::new();
        for (r, c, p) in es {
            match entries.last_mut() {
                Some(e) if e.row == r && e.col == c => e.pauli = p,
                _ => entries.push(Entry { row: r, col: c, pauli: p, to_check: [0.0; 3], to_qubit: 0.0 }),
            }
        }
        for v in channel.iter().chain(alpha.iter()) {
            if v.len() != num_qubits {
                return Err(format!("the channel and alpha need {num_qubits} values per Pauli"));
            }
        }
        let mut by_col = vec![Vec::new(); num_qubits];
        let mut by_row = vec![Vec::new(); num_checks];
        for (k, e) in entries.iter().enumerate() {
            by_row[e.row].push(k);
        }
        let mut order: Vec<usize> = (0..entries.len()).collect();
        order.sort_by_key(|&k| (entries[k].col, entries[k].row));
        for k in order {
            by_col[entries[k].col].push(k);
        }
        Ok(Mbp {
            num_checks,
            num_qubits,
            entries,
            by_col,
            by_row,
            channel,
            alpha,
            beta,
            gamma,
            min_sum,
            max_iter,
            decoding: vec![0; num_qubits],
            log_prob_ratios: [vec![0.0; num_qubits], vec![0.0; num_qubits], vec![0.0; num_qubits]],
            iterations: 0,
            converge: false,
        })
    }

    pub fn num_checks(&self) -> usize {
        self.num_checks
    }

    pub fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    /// The syndrome of a GF(4) error: whether each check anticommutes with it.
    pub fn pauli_syndrome(&self, error: &[u8]) -> Vec<u8> {
        self.by_row.iter().map(|row| (row.iter().filter(|&&k| {
            let e = &self.entries[k];
            error[e.col] != 0 && error[e.col] != e.pauli
        }).count() % 2) as u8).collect()
    }

    /// The incoming message of entry `g`, as a log ratio of commuting to anticommuting.
    fn ratio(g: &Entry) -> (f64, bool) {
        let p = (g.pauli - 1) as usize;
        let numerator = 1.0 + (-g.to_check[p]).exp();
        let mut denominator = 0.0;
        for w in 0..3 {
            if w != p {
                denominator += (-g.to_check[w]).exp();
            }
        }
        ((1e-12 + numerator / denominator).ln(), denominator <= 0.0)
    }

    /// Decode a syndrome (one value per check, compared as given); the decoding.
    pub fn decode(&mut self, syndrome: &[u8]) -> &[u8] {
        let n = self.num_qubits;
        self.converge = false;
        for q in 0..n {
            for &k in &self.by_col[q] {
                let e = &mut self.entries[k];
                for w in 0..3 {
                    e.to_check[w] = if e.pauli as usize != w + 1 { ((1.0 - self.channel[w][q]) / self.channel[w][q]).ln() } else { 0.0 };
                }
            }
        }
        let mut iter = 1;
        while iter <= self.max_iter {
            for q in 0..n {
                for &k in &self.by_col[q] {
                    let r = self.entries[k].row;
                    let message = if !self.min_sum {
                        let mut product = 1.0f64;
                        for &g in &self.by_row[r] {
                            if g != k {
                                let (t, reset) = Self::ratio(&self.entries[g]);
                                if reset {
                                    product = 1e-12;
                                }
                                product *= (t / 2.0).tanh();
                            }
                        }
                        if product >= 1.0 {
                            product = 1.0 - 1e-8;
                        } else if product <= -1.0 {
                            product = 1.0 + 1e-8; // (sic)
                        }
                        let sign = if syndrome[r] % 2 == 1 { -1.0 } else { 1.0 };
                        sign * ((1.0 + product) / (1.0 - product)).ln()
                    } else {
                        let mut min = f64::MAX;
                        let mut sgn = syndrome[r] as u32;
                        for &g in &self.by_row[r] {
                            if g != k {
                                let (t, _) = Self::ratio(&self.entries[g]);
                                if t.abs() < min {
                                    min = t.abs();
                                }
                                if t <= 0.0 {
                                    sgn += 1;
                                }
                            }
                        }
                        let sign = if sgn % 2 == 1 { -1.0 } else { 1.0 };
                        sign * self.gamma * min
                    };
                    self.entries[k].to_qubit = message;
                }
                for w in 0..3 {
                    self.log_prob_ratios[w][q] = ((1.0 - self.channel[w][q]) / self.channel[w][q]).ln();
                }
                for &k in &self.by_col[q] {
                    let e = &self.entries[k];
                    for w in 0..3 {
                        if e.pauli as usize != w + 1 {
                            self.log_prob_ratios[w][q] += (1.0 / self.alpha[w][q]) * e.to_qubit;
                        } else {
                            self.log_prob_ratios[w][q] += self.beta * e.to_qubit;
                        }
                    }
                }
                let mut all_positive = true;
                let mut min = f64::INFINITY;
                for w in 0..3 {
                    let l = self.log_prob_ratios[w][q];
                    if l < min {
                        min = l;
                        self.decoding[q] = w as u8 + 1;
                    }
                    if l < 0.0 {
                        all_positive = false;
                    }
                }
                if all_positive {
                    self.decoding[q] = 0;
                }
                for &k in &self.by_col[q] {
                    let e = &mut self.entries[k];
                    for w in 0..3 {
                        e.to_check[w] = self.log_prob_ratios[w][q];
                        if e.pauli as usize != w + 1 {
                            e.to_check[w] -= e.to_qubit;
                        }
                    }
                }
            }
            let candidate = self.pauli_syndrome(&self.decoding);
            if candidate.iter().zip(syndrome).all(|(a, b)| a == b) {
                self.iterations = iter;
                self.converge = true;
                return &self.decoding;
            }
            iter += 1;
        }
        self.converge = false;
        self.iterations = iter - 1;
        &self.decoding
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The five-qubit code's stabilisers XZZXI and its cyclic shifts, in GF(4).
    fn five_qubit() -> Vec<(usize, usize, u8)> {
        let row = [1u8, 3, 3, 1, 0];
        (0..4).flat_map(|r| (0..5).filter_map(move |c| {
            let p = row[(c + 5 - r) % 5];
            (p != 0).then_some((r, c, p))
        })).collect()
    }

    fn uniform(n: usize, v: f64) -> [Vec<f64>; 3] {
        [vec![v; n], vec![v; n], vec![v; n]]
    }

    #[test]
    fn it_decodes_the_five_qubit_codes_single_qubit_errors() {
        for (alpha, min_sum) in [(1.0, false), (0.8, false), (1.0, true)] {
            let mut mbp = Mbp::new(4, 5, &five_qubit(), uniform(5, 0.03), uniform(5, alpha), 0.0, 1.0, min_sum, 20).unwrap();
            let mut converged = 0;
            for q in 0..5 {
                for p in 1..=3u8 {
                    let mut e = vec![0u8; 5];
                    e[q] = p;
                    let s = mbp.pauli_syndrome(&e);
                    let d = mbp.decode(&s).to_vec();
                    // Converged means the decoding explains the syndrome.
                    assert_eq!(mbp.converge, mbp.pauli_syndrome(&d) == s);
                    converged += mbp.converge as usize;
                }
            }
            assert!(converged >= 8, "{alpha} {min_sum}: {converged} of 15");
        }
    }

    #[test]
    fn its_quirks_are_kept() {
        // A repeated entry keeps the last value; Paulis outside 1..3 are refused.
        let m = Mbp::new(1, 2, &[(0, 0, 1), (0, 0, 3), (0, 1, 3)], uniform(2, 0.1), uniform(2, 1.0), 0.0, 1.0, false, 3).unwrap();
        assert_eq!((m.pauli_syndrome(&[3, 0]), m.pauli_syndrome(&[1, 0])), (vec![0], vec![1]));
        assert!(Mbp::new(1, 1, &[(0, 0, 4)], uniform(1, 0.1), uniform(1, 1.0), 0.0, 1.0, false, 3).is_err());
        // A zero syndrome converges at once, at iteration 1.
        let mut m = m;
        m.decode(&[0]);
        assert!(m.converge && m.iterations == 1);
    }
}

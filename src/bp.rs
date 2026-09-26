//! Belief propagation on a parity-check (Tanner) graph.
//!
//! WHY THIS EXISTS
//! ---------------
//! Matching sees an error model as a graph: every fault splits into pieces
//! that set off at most two detectors, and the pieces are weighed as if they
//! were independent. BP works on the model as it is, a hypergraph with every
//! fault a variable and every detector a check. It passes messages between
//! them until each fault's posterior probability, given this shot's syndrome,
//! settles or the iterations run out. Belief-matching (`belief`) feeds those
//! posteriors to the matcher; BP+OSD, for codes matching cannot decode, will
//! feed them to ordered-statistics decoding.
//!
//! The schedule is the parallel (flooding) one of `ldpc` (Roffe et al.), the
//! library `beliefmatching` runs on, and it is written to reproduce that
//! library's arithmetic step for step:
//!
//! - each check's messages come from prefix and suffix products (or minima)
//!   along its row, variables in increasing order;
//! - each variable's outgoing messages come from prefix and suffix sums down
//!   its column, checks in increasing order;
//! - a variable is set when its log-likelihood ratio is at most zero;
//! - BP stops as soon as the set variables reproduce the syndrome.
//!
//! So the posteriors can be held to `ldpc`'s to rounding, not just to within
//! noise (tools/bp_check.py).

/// How checks combine their incoming messages.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Method {
    /// The exact rule on a tree: tanh products.
    ProductSum,
    /// The min-sum approximation, scaled; a scale of zero means `ldpc`'s
    /// adaptive scaling, 1 − 2⁻ⁱ at iteration i.
    MinSum { scale: f64 },
}

/// A parity-check matrix as a Tanner graph. Edges are numbered in column
/// order (variable by variable, checks ascending within each); each check
/// lists its edges in variable order.
pub struct Bp {
    pub num_checks: usize,
    pub num_vars: usize,
    var_start: Vec<u32>,
    edge_check: Vec<u32>,
    edge_var: Vec<u32>,
    chk_start: Vec<u32>,
    chk_edges: Vec<u32>,
    prior_llr: Vec<f64>,
}

/// One decode's messages and results; one per thread.
pub struct BpWork {
    b2c: Vec<f64>,
    c2b: Vec<f64>,
    /// tanh(b2c / 2), computed once per edge and iteration. `ldpc` computes
    /// it twice, in its forward and backward passes; the value is the same.
    half_tanh: Vec<f64>,
    /// Each variable's posterior log-likelihood ratio, ln(P(0)/P(1)).
    pub llr: Vec<f64>,
    /// The hard decision: 1 where a variable's LLR is at most zero.
    pub hard: Vec<u8>,
    candidate: Vec<u8>,
}

/// How a decode ended.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BpOutcome {
    pub converged: bool,
    pub iterations: usize,
}

impl Bp {
    /// `columns[v]` lists the checks variable `v` touches; `priors[v]` is its
    /// probability.
    pub fn new(num_checks: usize, columns: &[Vec<u32>], priors: &[f64]) -> Result<Bp, String> {
        if columns.len() != priors.len() {
            return Err(format!("{} columns but {} priors", columns.len(), priors.len()));
        }
        let mut var_start = vec![0u32];
        let (mut edge_check, mut edge_var) = (Vec::new(), Vec::new());
        let mut degree = vec![0u32; num_checks];
        for (v, col) in columns.iter().enumerate() {
            let mut col = col.clone();
            col.sort_unstable();
            col.dedup();
            for &c in &col {
                if c as usize >= num_checks {
                    return Err(format!("variable {v} touches check {c}, beyond {num_checks}"));
                }
                edge_check.push(c);
                edge_var.push(v as u32);
                degree[c as usize] += 1;
            }
            var_start.push(edge_check.len() as u32);
        }
        let mut chk_start = vec![0u32; num_checks + 1];
        for c in 0..num_checks {
            chk_start[c + 1] = chk_start[c] + degree[c];
        }
        let mut fill: Vec<u32> = chk_start[..num_checks].to_vec();
        let mut chk_edges = vec![0u32; edge_check.len()];
        // Edges are visited in variable order, so each check's list comes out
        // in variable order too.
        for (e, &c) in edge_check.iter().enumerate() {
            chk_edges[fill[c as usize] as usize] = e as u32;
            fill[c as usize] += 1;
        }
        let prior_llr = priors.iter().map(|&p| ((1.0 - p) / p).ln()).collect();
        Ok(Bp { num_checks, num_vars: columns.len(), var_start, edge_check, edge_var, chk_start, chk_edges, prior_llr })
    }

    pub fn work(&self) -> BpWork {
        let m = self.edge_check.len();
        BpWork {
            b2c: vec![0.0; m],
            c2b: vec![0.0; m],
            half_tanh: vec![0.0; m],
            llr: self.prior_llr.clone(),
            hard: vec![0; self.num_vars],
            candidate: vec![0; self.num_checks],
        }
    }

    pub fn num_edges(&self) -> usize {
        self.edge_check.len()
    }

    /// The prior log-likelihood ratios, ln((1 − p)/p).
    pub fn prior_llr(&self) -> &[f64] {
        &self.prior_llr
    }

    /// Run BP on `syndrome` (one byte per check, nonzero where it fired) for
    /// at most `max_iter` iterations. A zero syndrome converges at once to the
    /// empty correction, as `ldpc` does, without iterating.
    pub fn decode(&self, syndrome: &[u8], method: Method, max_iter: usize, w: &mut BpWork) -> BpOutcome {
        self.iterate(syndrome, method, max_iter, true, w)
    }

    /// `decode`, or with `stop` false, every iteration whatever the decision:
    /// the posteriors after exactly `max_iter` iterations.
    pub(crate) fn iterate(&self, syndrome: &[u8], method: Method, max_iter: usize, stop: bool, w: &mut BpWork) -> BpOutcome {
        assert_eq!(syndrome.len(), self.num_checks, "one syndrome bit per check");
        if syndrome.iter().all(|&s| s == 0) {
            w.hard.fill(0);
            w.llr.copy_from_slice(&self.prior_llr);
            return BpOutcome { converged: true, iterations: 0 };
        }
        for v in 0..self.num_vars {
            for e in self.var_start[v] as usize..self.var_start[v + 1] as usize {
                w.b2c[e] = self.prior_llr[v];
            }
        }
        for it in 1..=max_iter {
            self.check_update(syndrome, method, it, w);
            // Posteriors and the hard decision, with each edge's outgoing
            // message primed by the prefix sum down its column.
            for v in 0..self.num_vars {
                let mut temp = self.prior_llr[v];
                for e in self.var_start[v] as usize..self.var_start[v + 1] as usize {
                    w.b2c[e] = temp;
                    temp += w.c2b[e];
                }
                w.llr[v] = temp;
                w.hard[v] = u8::from(temp <= 0.0);
            }
            w.candidate.fill(0);
            for (e, &c) in self.edge_check.iter().enumerate() {
                w.candidate[c as usize] ^= w.hard[self.edge_var[e] as usize];
            }
            let converged = w.candidate.iter().zip(syndrome).all(|(&a, &b)| a == u8::from(b != 0));
            if converged && (stop || it == max_iter) {
                return BpOutcome { converged: true, iterations: it };
            }
            // Finish the outgoing messages with the suffix sums.
            for v in 0..self.num_vars {
                let mut temp = 0.0;
                for e in (self.var_start[v] as usize..self.var_start[v + 1] as usize).rev() {
                    w.b2c[e] += temp;
                    temp += w.c2b[e];
                }
            }
        }
        BpOutcome { converged: false, iterations: max_iter }
    }

    fn check_update(&self, syndrome: &[u8], method: Method, it: usize, w: &mut BpWork) {
        for c in 0..self.num_checks {
            let row = &self.chk_edges[self.chk_start[c] as usize..self.chk_start[c + 1] as usize];
            match method {
                Method::ProductSum => {
                    let mut temp = 1.0f64;
                    for &e in row {
                        let e = e as usize;
                        w.c2b[e] = temp;
                        w.half_tanh[e] = (w.b2c[e] / 2.0).tanh();
                        temp *= w.half_tanh[e];
                    }
                    temp = 1.0;
                    let sign = if syndrome[c] != 0 { -1.0 } else { 1.0 };
                    for &e in row.iter().rev() {
                        let e = e as usize;
                        let x = w.c2b[e] * temp;
                        w.c2b[e] = sign * ((1.0 + x) / (1.0 - x)).ln();
                        temp *= w.half_tanh[e];
                    }
                }
                Method::MinSum { scale } => {
                    let scale = if scale == 0.0 { 1.0 - 2f64.powi(-(it as i32)) } else { scale };
                    let mut total_sign = u32::from(syndrome[c] != 0);
                    let mut temp = f64::INFINITY;
                    for &e in row {
                        let e = e as usize;
                        w.c2b[e] = temp;
                        if w.b2c[e] <= 0.0 {
                            total_sign += 1;
                        }
                        temp = temp.min(w.b2c[e].abs());
                    }
                    temp = f64::INFINITY;
                    for &e in row.iter().rev() {
                        let e = e as usize;
                        let mut sign = total_sign;
                        if w.b2c[e] <= 0.0 {
                            sign += 1;
                        }
                        if temp < w.c2b[e] {
                            w.c2b[e] = temp;
                        }
                        let s = if sign % 2 == 0 { 1.0 } else { -1.0 };
                        w.c2b[e] *= s * scale;
                        temp = temp.min(w.b2c[e].abs());
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Exact posteriors by summing over every error pattern.
    fn brute_force(num_checks: usize, columns: &[Vec<u32>], priors: &[f64], syndrome: &[u8]) -> Vec<f64> {
        let n = columns.len();
        let mut p1 = vec![0.0; n];
        let mut total = 0.0;
        for pattern in 0u32..(1 << n) {
            let mut s = vec![0u8; num_checks];
            let mut weight = 1.0;
            for v in 0..n {
                if pattern >> v & 1 == 1 {
                    weight *= priors[v];
                    for &c in &columns[v] {
                        s[c as usize] ^= 1;
                    }
                } else {
                    weight *= 1.0 - priors[v];
                }
            }
            if s == syndrome {
                total += weight;
                for v in 0..n {
                    if pattern >> v & 1 == 1 {
                        p1[v] += weight;
                    }
                }
            }
        }
        p1.iter().map(|x| x / total).collect()
    }

    /// On a Tanner graph with no cycles, product-sum BP is exact: run long
    /// enough, every posterior equals the true marginal. Two trees: a path (a
    /// repetition code) and a branching one with checks of degree three.
    #[test]
    fn product_sum_is_exact_on_trees() {
        let path: (usize, Vec<Vec<u32>>, Vec<f64>) =
            (4, vec![vec![0], vec![0, 1], vec![1, 2], vec![2, 3], vec![3]], vec![0.1, 0.05, 0.2, 0.15, 0.3]);
        let branch: (usize, Vec<Vec<u32>>, Vec<f64>) =
            (3, vec![vec![0], vec![0], vec![0, 1], vec![1, 2], vec![2], vec![2]], vec![0.1, 0.2, 0.05, 0.3, 0.15, 0.25]);
        for (checks, columns, priors) in [path, branch] {
            let bp = Bp::new(checks, &columns, &priors).unwrap();
            let mut w = bp.work();
            for pattern in 1u32..(1 << checks) {
                let syndrome: Vec<u8> = (0..checks).map(|c| (pattern >> c & 1) as u8).collect();
                let exact = brute_force(checks, &columns, &priors, &syndrome);
                bp.iterate(&syndrome, Method::ProductSum, 30, false, &mut w);
                for v in 0..columns.len() {
                    let p = 1.0 / (1.0 + w.llr[v].exp());
                    assert!((p - exact[v]).abs() < 1e-12, "{syndrome:?} var {v}: {p} vs {}", exact[v]);
                }
            }
        }
    }

    /// A converged decision explains the syndrome.
    #[test]
    fn a_converged_decision_explains_the_syndrome() {
        let columns: Vec<Vec<u32>> = vec![vec![0], vec![0, 1], vec![1, 2], vec![2, 3], vec![3]];
        let bp = Bp::new(4, &columns, &[0.1, 0.05, 0.2, 0.15, 0.3]).unwrap();
        let mut w = bp.work();
        for pattern in 1u32..16 {
            let syndrome: Vec<u8> = (0..4).map(|c| (pattern >> c & 1) as u8).collect();
            if bp.decode(&syndrome, Method::ProductSum, 20, &mut w).converged {
                let mut s = vec![0u8; 4];
                for v in 0..5 {
                    if w.hard[v] == 1 {
                        for &c in &columns[v] {
                            s[c as usize] ^= 1;
                        }
                    }
                }
                assert_eq!(s, syndrome);
            }
        }
    }

    #[test]
    fn a_zero_syndrome_converges_at_once() {
        let bp = Bp::new(2, &[vec![0], vec![0, 1], vec![1]], &[0.1, 0.1, 0.1]).unwrap();
        let mut w = bp.work();
        let out = bp.decode(&[0, 0], Method::ProductSum, 20, &mut w);
        assert_eq!(out, BpOutcome { converged: true, iterations: 0 });
        assert_eq!(w.hard, vec![0, 0, 0]);
    }

    /// A single likely fault is found in one iteration, by both methods.
    #[test]
    fn one_fault_is_found() {
        let columns: Vec<Vec<u32>> = vec![vec![0], vec![0, 1], vec![1, 2], vec![2]];
        let bp = Bp::new(3, &columns, &[0.01, 0.01, 0.01, 0.01]).unwrap();
        let mut w = bp.work();
        for method in [Method::ProductSum, Method::MinSum { scale: 1.0 }, Method::MinSum { scale: 0.625 }] {
            let out = bp.decode(&[0, 1, 1], method, 20, &mut w);
            assert!(out.converged, "{method:?}");
            assert_eq!(w.hard, vec![0, 0, 1, 0], "{method:?}");
        }
    }

    /// Min-sum's messages are min-sum's: on a single check, a variable hears
    /// the scaled minimum of the others' magnitudes, signed by the syndrome
    /// and their signs.
    #[test]
    fn min_sum_messages_are_scaled_minima() {
        let columns: Vec<Vec<u32>> = vec![vec![0], vec![0], vec![0]];
        let priors = [0.1, 0.2, 0.3];
        let bp = Bp::new(1, &columns, &priors).unwrap();
        let mut w = bp.work();
        let llr: Vec<f64> = priors.iter().map(|p: &f64| ((1.0 - p) / p).ln()).collect();
        bp.decode(&[1], Method::MinSum { scale: 0.5 }, 1, &mut w);
        for v in 0..3 {
            let others = (0..3).filter(|&u| u != v).map(|u| llr[u]).fold(f64::INFINITY, f64::min);
            let want = llr[v] - 0.5 * others;
            assert!((w.llr[v] - want).abs() < 1e-12, "var {v}: {} vs {want}", w.llr[v]);
        }
    }
}

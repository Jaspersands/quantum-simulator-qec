//! Relay-BP (Müller, Alexander, Beverland, Bühler, Johnson, Maurer and Vandeth, IBM,
//! arXiv:2506.01779): chained runs of belief propagation with memory.
//!
//! WHY THIS EXISTS
//! ---------------
//! Plain BP fails on quantum LDPC codes mostly by oscillating: degenerate errors pull a fault's
//! belief back and forth, and the messages never settle. BP+OSD and BP+LSD follow a failure
//! with an elimination, which is slow and hard to put in hardware. Relay-BP stays BP:
//!
//! - *memory*: each fault's prior at an iteration mixes its true prior with its own last
//!   posterior, `(1 − γ)·λ + γ·Λ`, which damps oscillation (γ > 0) or pushes a stuck fault
//!   out (γ < 0);
//! - *disorder*: after a first leg with every γ the same, each further leg draws a γ for every
//!   fault at random, so different legs break different symmetries;
//! - *relay*: a leg starts from the posteriors the last one ended with (its messages reset);
//! - the first `solutions` legs whose hard decision explains the syndrome are kept, and the
//!   one of least weight `Σ ln((1 − p)/p)` is returned.
//!
//! Every step is min-sum, so it suits hardware, and on the gross code it matches BP+OSD's
//! logical error rate in a few hundred iterations.
//!
//! This is IBM's implementation (`relay_bp` 0.2.2, crates/relay_bp/src/bp/{min_sum,relay}.rs,
//! its `f64` instantiation) ported with its arithmetic in its order, so its results are
//! reproduced bit for bit: for explicit memory strengths, and for a seed (the strengths come
//! from `chacha`, which reproduces the `rand` 0.8 generator it draws them with).

use crate::chacha::{StdRng, Uniform};

/// How Relay-BP runs (IBM's `RelayDecoderConfig` and `MinSumDecoderConfig`).
#[derive(Clone, Debug, PartialEq)]
pub struct RelayConfig {
    /// Iterations of the first leg.
    pub pre_iter: usize,
    /// Legs after the first, at most.
    pub legs: usize,
    /// Iterations of each later leg.
    pub leg_iter: usize,
    /// Stop after this many legs have converged; `None` runs every leg.
    pub solutions: Option<usize>,
    /// The first leg's memory strength; `None` turns memory off (plain min-sum throughout).
    pub gamma0: Option<f64>,
    /// The interval later legs draw memory strengths from.
    pub gamma_range: (f64, f64),
    /// Explicit memory strengths instead, one row of one per fault for each leg (reused
    /// cyclically).
    pub gammas: Option<Vec<Vec<f64>>>,
    /// The min-sum scaling: `None` is 1, `Some(0.0)` is `1 − 2^−(t/alpha_scaling)` at
    /// iteration t (from 1).
    pub alpha: Option<f64>,
    pub alpha_scaling: f64,
    pub seed: u64,
}

/// The check matrix and the per-fault constants.
pub struct Relay {
    pub num_checks: usize,
    pub num_vars: usize,
    var_start: Vec<u32>,
    edge_var: Vec<u32>,
    chk_start: Vec<u32>,
    chk_edges: Vec<u32>,
    /// ln((1 − p)/p), `f64::MAX` where p = 0 (as `N::max_value()`).
    prior: Vec<f64>,
    /// ln((1 − p)/p) where finite, for weighing solutions.
    weight: Vec<f64>,
    pub config: RelayConfig,
    uniform: Uniform,
}

/// One decoder's mutable state. Like IBM's decoder, the random memory strengths continue from
/// shot to shot: a decode depends on how many came before it on the same work.
pub struct RelayWork {
    v2c: Vec<f64>,
    c2v: Vec<f64>,
    posterior: Vec<f64>,
    gamma: Vec<f64>,
    hard: Vec<u8>,
    decoded: Vec<u8>,
    rng: StdRng,
    /// The returned correction.
    pub correction: Vec<u8>,
}

/// How a decode ended.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RelayOutcome {
    /// Whether the correction explains the syndrome.
    pub converged: bool,
    /// Iterations over every leg run.
    pub iterations: usize,
    /// Legs run, the first included.
    pub legs: usize,
    /// The correction's weight Σ ln((1 − p)/p) (where it converged).
    pub weight: f64,
}

impl Relay {
    /// `columns[v]` lists the checks fault `v` sets off; `priors[v]` is its probability.
    pub fn new(num_checks: usize, columns: &[Vec<u32>], priors: &[f64], config: RelayConfig) -> Result<Relay, String> {
        if columns.len() != priors.len() {
            return Err(format!("{} columns but {} priors", columns.len(), priors.len()));
        }
        let (lo, hi) = config.gamma_range;
        if !(lo < hi && lo.is_finite() && hi.is_finite()) {
            return Err(format!("the memory strengths' range must be finite and increasing, not ({lo}, {hi})"));
        }
        if let Some(g) = &config.gammas {
            if g.is_empty() || g.iter().any(|row| row.len() != columns.len()) {
                return Err(format!("explicit memory strengths need rows of {} (one per fault)", columns.len()));
            }
        }
        let mut var_start = vec![0u32];
        let mut edge_check = Vec::new();
        let mut edge_var = Vec::new();
        let mut degree = vec![0u32; num_checks];
        for (v, col) in columns.iter().enumerate() {
            let mut col = col.clone();
            col.sort_unstable();
            col.dedup();
            for &c in &col {
                if c as usize >= num_checks {
                    return Err(format!("fault {v} sets off check {c}, beyond {num_checks}"));
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
        for (e, &c) in edge_check.iter().enumerate() {
            chk_edges[fill[c as usize] as usize] = e as u32;
            fill[c as usize] += 1;
        }
        let llr: Vec<f64> = priors.iter().map(|&p| ((1.0 - p) / p).ln()).collect();
        let prior = llr.iter().map(|&x| if x == f64::INFINITY { f64::MAX } else { x }).collect();
        let weight = llr.iter().map(|&x| if x.is_finite() { x } else { 0.0 }).collect();
        let uniform = Uniform::new(lo, hi);
        Ok(Relay { num_checks, num_vars: columns.len(), var_start, edge_var, chk_start, chk_edges, prior, weight, config, uniform })
    }

    pub fn work(&self) -> RelayWork {
        self.work_seeded(self.config.seed)
    }

    pub(crate) fn work_seeded(&self, seed: u64) -> RelayWork {
        let m = self.edge_var.len();
        RelayWork {
            v2c: vec![0.0; m],
            c2v: vec![0.0; m],
            posterior: vec![0.0; self.num_vars],
            gamma: vec![0.0; self.num_vars],
            hard: vec![0; self.num_vars],
            decoded: vec![0; self.num_checks],
            rng: StdRng::seed_from_u64(seed),
            correction: vec![0; self.num_vars],
        }
    }

    /// Restart the random memory strengths from `seed`.
    pub(crate) fn reseed(&self, w: &mut RelayWork, seed: u64) {
        w.rng = StdRng::seed_from_u64(seed);
    }

    fn alpha(&self, iteration: usize) -> f64 {
        let mut a = match self.config.alpha {
            Some(0.0) => 1.0 - 2f64.powf(-((iteration + 1) as f64 / self.config.alpha_scaling)),
            Some(v) => v,
            None => 1.0,
        };
        if a < 0.0 {
            a = 1.0;
        }
        a
    }

    fn reset_messages(&self, w: &mut RelayWork) {
        w.c2v.fill(0.0);
        for v in 0..self.num_vars {
            for e in self.var_start[v] as usize..self.var_start[v + 1] as usize {
                w.v2c[e] = self.prior[v];
            }
        }
    }

    /// One iteration: checks, then variables and posteriors, then the hard decision.
    fn iterate(&self, syndrome: &[u8], iteration: usize, w: &mut RelayWork) {
        let alpha = self.alpha(iteration);
        for c in 0..self.num_checks {
            let row = &self.chk_edges[self.chk_start[c] as usize..self.chk_start[c + 1] as usize];
            let mut sign = syndrome[c] == 1;
            let (mut min_var, mut min, mut second) = (0u32, f64::MAX, f64::MAX);
            for &e in row {
                let m = w.v2c[e as usize];
                sign ^= m.is_sign_negative();
                let a = m.abs();
                if a <= min {
                    second = min;
                    min = a;
                    min_var = self.edge_var[e as usize];
                } else if a <= second {
                    second = a;
                }
            }
            for &e in row {
                let e = e as usize;
                let s = sign ^ w.v2c[e].is_sign_negative();
                let m = if self.edge_var[e] != min_var { min } else { second };
                let msg = alpha * m;
                w.c2v[e] = if s { -msg } else { msg };
            }
        }
        let memory = self.config.gamma0.is_some();
        for v in 0..self.num_vars {
            let mut sum = if !memory || self.prior[v] == f64::MAX {
                self.prior[v]
            } else {
                let g = w.gamma[v];
                (self.prior[v] / 1.0) * (1.0 - g) + (w.posterior[v] / 1.0) * g
            };
            let edges = self.var_start[v] as usize..self.var_start[v + 1] as usize;
            for e in edges.clone() {
                w.v2c[e] = sum;
                sum += w.c2v[e];
            }
            w.posterior[v] = sum;
            let mut sum = 0.0;
            for e in edges.rev() {
                w.v2c[e] += sum;
                sum += w.c2v[e];
            }
        }
        for v in 0..self.num_vars {
            w.hard[v] = u8::from(w.posterior[v] <= 0.0);
        }
    }

    /// Whether the hard decision sets off exactly the syndrome.
    fn explains(&self, syndrome: &[u8], w: &mut RelayWork) -> bool {
        for c in 0..self.num_checks {
            let mut x = 0u8;
            for &e in &self.chk_edges[self.chk_start[c] as usize..self.chk_start[c + 1] as usize] {
                x ^= w.hard[self.edge_var[e as usize] as usize];
            }
            w.decoded[c] = x;
        }
        w.decoded.iter().zip(syndrome).all(|(&a, &b)| a == b)
    }

    /// A leg of at most `max_iter` iterations: (converged, iterations).
    fn leg(&self, syndrome: &[u8], max_iter: usize, w: &mut RelayWork) -> (bool, usize) {
        for it in 0..max_iter {
            self.iterate(syndrome, it, w);
            if self.explains(syndrome, w) {
                return (true, it + 1);
            }
        }
        (false, max_iter)
    }

    fn weight_of(&self, hard: &[u8]) -> f64 {
        hard.iter().zip(&self.weight).filter(|(&h, _)| h == 1).map(|(_, &x)| x).sum()
    }

    /// Decode one syndrome (one byte per check, 0 or 1); the correction is left in
    /// `w.correction`. Where no leg converges, it is the first leg's last hard decision.
    pub fn decode(&self, syndrome: &[u8], w: &mut RelayWork) -> RelayOutcome {
        assert_eq!(syndrome.len(), self.num_checks, "one syndrome bit per check");
        let gamma0 = self.config.gamma0.unwrap_or(0.0);
        w.gamma.fill(gamma0);
        self.reset_messages(w);
        if self.config.gamma0.is_some() {
            w.posterior.copy_from_slice(&self.prior);
        }
        let (ok, its) = self.leg(syndrome, self.config.pre_iter, w);
        w.correction.copy_from_slice(&w.hard);
        let mut best = if ok { self.weight_of(&w.hard) } else { f64::MAX };
        let mut converged = usize::from(ok);
        let mut out = RelayOutcome { converged: ok, iterations: its, legs: 1, weight: best };
        if ok && self.config.solutions.is_some_and(|k| converged >= k) {
            return out;
        }
        let mut total = its;
        for leg in 1..=self.config.legs {
            match &self.config.gammas {
                Some(rows) => w.gamma.copy_from_slice(&rows[leg % rows.len()]),
                None => {
                    for g in w.gamma.iter_mut() {
                        *g = self.uniform.sample(&mut w.rng);
                    }
                }
            }
            self.reset_messages(w);
            let (ok, its) = self.leg(syndrome, self.config.leg_iter, w);
            total += its;
            out.legs += 1;
            if ok {
                converged += 1;
                let q = self.weight_of(&w.hard);
                if q < best {
                    best = q;
                    w.correction.copy_from_slice(&w.hard);
                    out.converged = true;
                    out.weight = q;
                }
                if self.config.solutions.is_some_and(|k| converged >= k) {
                    break;
                }
            }
        }
        out.iterations = total;
        out
    }
}

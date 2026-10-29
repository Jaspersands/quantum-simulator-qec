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
//! This is IBM's implementation (`relay_bp` 0.2.2, crates/relay_bp/src/bp/{min_sum,relay}.rs)
//! ported with its arithmetic in its order, in each number type it is built for (`RelayNum`:
//! `f32`, `f64`, `i8`–`i64`, fixed point), with its data scaling and magnitude bounds, its
//! stopping criteria and its plain min-sum decoder (`RelayCore::decode_min_sum`, Mem-BP when
//! `gamma0` is set), so its results are reproduced bit for bit: for explicit memory strengths,
//! and for a seed (the strengths come from `chacha`, which reproduces the `rand` 0.8 generator
//! it draws them with). `Relay` is the `f64` instantiation.

use crate::chacha::{StdRng, Uniform};
use crate::relay_num::RelayNum;

/// How Relay-BP runs (IBM's `RelayDecoderConfig` and `MinSumDecoderConfig`).
#[derive(Clone, Debug, PartialEq)]
pub struct RelayConfig {
    /// Iterations of the first leg (of the whole run, for plain min-sum).
    pub pre_iter: usize,
    /// Legs after the first, at most.
    pub legs: usize,
    /// Iterations of each later leg.
    pub leg_iter: usize,
    /// Stop after this many legs have converged; `None` runs every leg.
    pub solutions: Option<usize>,
    /// IBM's `pre_iter` stopping criterion instead: stop if the first leg converges, otherwise
    /// run every leg (`solutions` is then ignored).
    pub stop_after_first: bool,
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
    /// Every value scaled by this (IBM's `data_scale_value`, for integer types).
    pub data_scale: Option<f64>,
    /// Messages and posteriors kept within ±this (IBM's `max_data_value`).
    pub max_data: Option<f64>,
    /// Keep each leg's iterations and outcome for IBM's log (`RelayWork::log_lines`).
    pub logging: bool,
    pub seed: u64,
}

impl Default for RelayConfig {
    /// IBM's defaults.
    fn default() -> Self {
        RelayConfig {
            pre_iter: 80,
            legs: 300,
            leg_iter: 60,
            solutions: Some(1),
            stop_after_first: false,
            gamma0: Some(0.1),
            gamma_range: (-0.24, 0.66),
            gammas: None,
            alpha: None,
            alpha_scaling: 1.0,
            data_scale: None,
            max_data: None,
            logging: false,
            seed: 0,
        }
    }
}

/// The check matrix and the per-fault constants, in number type `N`.
#[derive(Clone)]
pub struct RelayCore<N: RelayNum> {
    pub num_checks: usize,
    pub num_vars: usize,
    var_start: Vec<u32>,
    edge_var: Vec<u32>,
    chk_start: Vec<u32>,
    chk_edges: Vec<u32>,
    /// Per edge, 1 where its stored value is odd (empty: every edge's is). Messages pass along
    /// every stored entry, but the checks a hard decision sets off are IBM's
    /// `check_matrix * decoding % 2`, which reads the values: an even one (an explicit 0 left
    /// by `H.data %= 2`) is an edge that sets nothing off.
    parity: Vec<u8>,
    /// ln((1 − p)/p), scaled and bounded, `N::max_value()` where p = 0.
    prior: Vec<N>,
    /// ln((1 − p)/p) where finite, for weighing solutions.
    weight: Vec<f64>,
    data_scale: Option<N>,
    max_data: Option<N>,
    pub config: RelayConfig,
    uniform: Uniform,
}

/// Relay-BP in `f64`.
pub type Relay = RelayCore<f64>;
/// Its state.
pub type RelayWork = RelayCoreWork<f64>;

/// One decoder's mutable state. Like IBM's decoder, the random memory strengths continue from
/// shot to shot (a decode depends on how many came before it on the same work), and so do
/// values a decode does not overwrite.
#[derive(Clone)]
pub struct RelayCoreWork<N: RelayNum> {
    v2c: Vec<N>,
    c2v: Vec<N>,
    posterior: Vec<N>,
    gamma: Vec<N>,
    hard: Vec<u8>,
    decoded: Vec<u8>,
    rng: StdRng,
    /// The returned correction.
    pub correction: Vec<u8>,
    /// The returned run's checks set off by the correction (zeros if it ran no iteration).
    pub decoded_detectors: Vec<u8>,
    /// The returned run's posterior log-likelihood ratios, unscaled, as `f64`.
    pub posterior_ratios: Vec<f64>,
    /// For IBM's log: each leg's iterations, whether it converged, and whether it gave the
    /// unique best solution, kept from decode to decode as IBM's are; the legs run last.
    log_iter: Vec<usize>,
    log_conv: Vec<bool>,
    log_best: Vec<bool>,
    log_legs: usize,
}

/// How a decode ended.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RelayOutcome {
    /// Whether the correction explains the syndrome.
    pub converged: bool,
    /// Iterations over every leg run (the first leg's alone if it converged and stopped).
    pub iterations: usize,
    /// Legs run, the first included.
    pub legs: usize,
    /// The correction's weight Σ ln((1 − p)/p) (where it converged; `f64::MAX` otherwise).
    pub weight: f64,
    /// The iteration limit of the leg the correction came from.
    pub max_iter: usize,
}

impl<N: RelayNum> RelayCore<N> {
    /// `columns[v]` lists the checks fault `v` sets off (a check listed twice, none);
    /// `priors[v]` is its probability.
    pub fn new(num_checks: usize, columns: &[Vec<u32>], priors: &[f64], config: RelayConfig) -> Result<RelayCore<N>, String> {
        Self::with_values(num_checks, columns, None, priors, config)
    }

    /// As `new`, with the check matrix's stored value for each entry of `columns`
    /// (`values[v][k]` for `columns[v][k]`): every entry is an edge, and one whose values sum
    /// to an even number sets off nothing, as in IBM's decoder.
    pub fn with_values(num_checks: usize, columns: &[Vec<u32>], values: Option<&[Vec<u8>]>, priors: &[f64], config: RelayConfig) -> Result<RelayCore<N>, String> {
        if columns.len() != priors.len() {
            return Err(format!("{} columns but {} priors", columns.len(), priors.len()));
        }
        if let Some(vals) = values {
            if vals.len() != columns.len() || vals.iter().zip(columns).any(|(a, b)| a.len() != b.len()) {
                return Err("one stored value per check-matrix entry".to_string());
            }
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
        let mut parity = Vec::new();
        for (v, col) in columns.iter().enumerate() {
            // (check, value parity), sorted, a repeated entry once with its values' parity:
            // without values each stored entry is a 1, so a check stored twice sums to 2 and sets
            // off nothing (it stays an edge, as one with an even value does).
            let mut entries: Vec<(u32, u8)> = match values {
                Some(vals) => col.iter().copied().zip(vals[v].iter().map(|&x| x & 1)).collect(),
                None => col.iter().map(|&c| (c, 1)).collect(),
            };
            entries.sort_unstable_by_key(|&(c, _)| c);
            let mut merged: Vec<(u32, u8)> = Vec::with_capacity(entries.len());
            for (c, p) in entries {
                match merged.last_mut() {
                    Some(last) if last.0 == c => last.1 ^= p,
                    _ => merged.push((c, p)),
                }
            }
            for &(c, p) in &merged {
                if c as usize >= num_checks {
                    return Err(format!("fault {v} sets off check {c}, beyond {num_checks}"));
                }
                edge_check.push(c);
                edge_var.push(v as u32);
                parity.push(p);
                degree[c as usize] += 1;
            }
            var_start.push(edge_check.len() as u32);
        }
        if parity.iter().all(|&p| p == 1) {
            parity.clear();
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
        let convert = |x: f64, what: &str| N::from_f64(x).ok_or_else(|| format!("{what} {x} does not fit this decoder's number type"));
        let max_data = config.max_data.and_then(N::from_f64);
        let data_scale = config.data_scale.and_then(N::from_f64);
        let llr: Vec<f64> = priors.iter().map(|&p| ((1.0 - p) / p).ln()).collect();
        let mut prior = Vec::with_capacity(llr.len());
        for &x in &llr {
            let v = if x == f64::INFINITY { N::max_value() } else { convert(config.data_scale.map_or(x, |s| s * x), "the log prior ratio")? };
            prior.push(max_data.map_or(v, |m| bound(v, m)));
        }
        let g0 = config.gamma0.unwrap_or(0.0);
        convert(config.data_scale.map_or(g0, |s| s * g0), "the memory strength")?;
        let weight = llr.iter().map(|&x| if x.is_finite() { x } else { 0.0 }).collect();
        let uniform = Uniform::new(lo, hi);
        Ok(RelayCore { num_checks, num_vars: columns.len(), var_start, edge_var, chk_start, chk_edges, parity, prior, weight, data_scale, max_data, config, uniform })
    }

    /// The faults' priors (0 where a prior was 0).
    pub fn priors(&self) -> Vec<f64> {
        self.prior.iter().zip(&self.weight).map(|(p, &w)| if *p == N::max_value() { 0.0 } else { 1.0 / (1.0 + w.exp()) }).collect()
    }

    /// One fault's new prior, scaled and bounded as the others were.
    pub fn set_prior(&mut self, v: usize, p: f64) {
        let x = ((1.0 - p) / p).ln();
        let scaled = if x == f64::INFINITY { N::max_value() } else { N::from_f64(self.config.data_scale.map_or(x, |s| s * x)).unwrap_or_else(N::max_value) };
        self.prior[v] = self.max_data.map_or(scaled, |m| bound(scaled, m));
        self.weight[v] = if x.is_finite() { x } else { 0.0 };
    }

    pub fn work(&self) -> RelayCoreWork<N> {
        self.work_seeded(self.config.seed)
    }

    pub(crate) fn work_seeded(&self, seed: u64) -> RelayCoreWork<N> {
        let m = self.edge_var.len();
        let sets = if self.config.logging { self.config.legs + 1 } else { 0 };
        RelayCoreWork {
            v2c: vec![N::zero(); m],
            c2v: vec![N::zero(); m],
            // (IBM's starts at the priors with memory, at zero without.)
            posterior: if self.config.gamma0.is_some() { self.prior.clone() } else { vec![N::zero(); self.num_vars] },
            gamma: vec![N::zero(); self.num_vars],
            hard: vec![0; self.num_vars],
            decoded: vec![0; self.num_checks],
            rng: StdRng::seed_from_u64(seed),
            correction: vec![0; self.num_vars],
            decoded_detectors: vec![0; self.num_checks],
            posterior_ratios: vec![0.0; self.num_vars],
            log_iter: vec![self.config.leg_iter; sets],
            log_conv: vec![false; sets],
            log_best: vec![false; sets],
            log_legs: 0,
        }
    }

    /// Restart the random memory strengths from `seed`.
    pub(crate) fn reseed(&self, w: &mut RelayCoreWork<N>, seed: u64) {
        w.rng = StdRng::seed_from_u64(seed);
    }

    /// `N::from_f64`, which IBM's decoder unwraps.
    fn num(&self, x: f64) -> N {
        N::from_f64(x).unwrap_or_else(|| panic!("{x} does not fit this decoder's number type"))
    }

    fn scaled(&self, x: f64) -> N {
        self.num(self.config.data_scale.map_or(x, |s| s * x))
    }

    fn alpha(&self, iteration: usize) -> N {
        let mut a = match self.config.alpha {
            Some(0.0) => 1.0 - 2f64.powf(-((iteration + 1) as f64 / self.config.alpha_scaling)),
            Some(v) => v,
            None => 1.0,
        };
        if a < 0.0 {
            a = 1.0;
        }
        self.scaled(a)
    }

    fn reset_messages(&self, w: &mut RelayCoreWork<N>) {
        w.c2v.fill(N::zero());
        for v in 0..self.num_vars {
            for e in self.var_start[v] as usize..self.var_start[v + 1] as usize {
                w.v2c[e] = self.prior[v];
            }
        }
    }

    /// One iteration: checks, then variables and posteriors, then the hard decision.
    fn iterate(&self, syndrome: &[u8], iteration: usize, w: &mut RelayCoreWork<N>) {
        let alpha = self.alpha(iteration);
        for c in 0..self.num_checks {
            let row = &self.chk_edges[self.chk_start[c] as usize..self.chk_start[c + 1] as usize];
            let mut sign = syndrome[c] == 1;
            let (mut min_var, mut min, mut second) = (0u32, N::max_value(), N::max_value());
            for &e in row {
                let m = w.v2c[e as usize];
                sign ^= m.is_negative();
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
                let s = sign ^ w.v2c[e].is_negative();
                let m = if self.edge_var[e] != min_var { min } else { second };
                let msg = alpha.mul(m);
                w.c2v[e] = if s { msg.neg() } else { msg };
            }
        }
        if let Some(scale) = self.data_scale {
            for x in w.c2v.iter_mut() {
                *x = x.div(scale);
            }
        }
        let memory = self.config.gamma0.is_some();
        let one = self.data_scale.unwrap_or(N::one());
        for v in 0..self.num_vars {
            let mut sum = if !memory || self.prior[v] == N::max_value() {
                self.prior[v]
            } else {
                let g = w.gamma[v];
                self.prior[v].div(one).mul(one.add(g.neg())).add(w.posterior[v].div(one).mul(g))
            };
            let edges = self.var_start[v] as usize..self.var_start[v + 1] as usize;
            for e in edges.clone() {
                w.v2c[e] = sum;
                sum = sum.add(w.c2v[e]);
            }
            w.posterior[v] = sum;
            let mut sum = N::zero();
            for e in edges.rev() {
                w.v2c[e] = w.v2c[e].add(sum);
                sum = sum.add(w.c2v[e]);
            }
        }
        if let Some(m) = self.max_data {
            for x in w.v2c.iter_mut() {
                *x = bound(*x, m);
            }
            for x in w.posterior.iter_mut() {
                *x = bound(*x, m);
            }
        }
        for v in 0..self.num_vars {
            w.hard[v] = u8::from(w.posterior[v] <= N::zero());
        }
    }

    /// Whether the hard decision sets off exactly the syndrome.
    fn explains(&self, syndrome: &[u8], w: &mut RelayCoreWork<N>) -> bool {
        for c in 0..self.num_checks {
            let mut x = 0u8;
            let row = &self.chk_edges[self.chk_start[c] as usize..self.chk_start[c + 1] as usize];
            if self.parity.is_empty() {
                for &e in row {
                    x ^= w.hard[self.edge_var[e as usize] as usize];
                }
            } else {
                for &e in row {
                    x ^= w.hard[self.edge_var[e as usize] as usize] & self.parity[e as usize];
                }
            }
            w.decoded[c] = x;
        }
        w.decoded.iter().zip(syndrome).all(|(&a, &b)| a == b)
    }

    /// A leg of at most `max_iter` iterations: (converged, iterations). The checks the hard
    /// decision sets off are left in `w.decoded` (zeros if no iteration ran).
    fn leg(&self, syndrome: &[u8], max_iter: usize, w: &mut RelayCoreWork<N>) -> (bool, usize) {
        if max_iter == 0 {
            w.decoded.fill(0);
            return (false, 0);
        }
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

    /// IBM's `get_decoding_quality` of any error: Σ ln((1 − p)/p) over its faults.
    pub fn quality(&self, error: &[u8]) -> f64 {
        self.weight_of(error)
    }

    /// The leg just run becomes the result.
    fn keep(&self, w: &mut RelayCoreWork<N>) {
        w.correction.copy_from_slice(&w.hard);
        w.decoded_detectors.copy_from_slice(&w.decoded);
        let unscale = self.config.data_scale;
        for (out, &p) in w.posterior_ratios.iter_mut().zip(&w.posterior) {
            let x = p.to_f64();
            *out = unscale.map_or(x, |s| x / s);
        }
    }

    fn start(&self, w: &mut RelayCoreWork<N>) {
        let gamma0 = self.scaled(self.config.gamma0.unwrap_or(0.0));
        w.gamma.fill(gamma0);
        self.reset_messages(w);
        if self.config.gamma0.is_some() {
            w.posterior.copy_from_slice(&self.prior);
        }
    }

    /// Plain min-sum (IBM's `MinSumBPDecoder`; Mem-BP with `gamma0`): one run of `pre_iter`
    /// iterations, the result left in the work as for `decode`.
    pub fn decode_min_sum(&self, syndrome: &[u8], w: &mut RelayCoreWork<N>) -> RelayOutcome {
        assert_eq!(syndrome.len(), self.num_checks, "one syndrome bit per check");
        self.start(w);
        let (ok, its) = self.leg(syndrome, self.config.pre_iter, w);
        self.keep(w);
        let weight = if ok { self.weight_of(&w.hard) } else { f64::MAX };
        RelayOutcome { converged: ok, iterations: its, legs: 1, weight, max_iter: self.config.pre_iter }
    }

    /// Decode one syndrome (one byte per check, 0 or 1); the correction is left in
    /// `w.correction`. Where no leg converges, it is the first leg's last hard decision.
    pub fn decode(&self, syndrome: &[u8], w: &mut RelayCoreWork<N>) -> RelayOutcome {
        assert_eq!(syndrome.len(), self.num_checks, "one syndrome bit per check");
        let logging = self.config.logging;
        self.start(w);
        let (ok, its) = self.leg(syndrome, self.config.pre_iter, w);
        self.keep(w);
        w.log_legs = 1;
        if logging {
            w.log_iter[0] = its;
            w.log_conv[0] = ok;
        }
        let mut best = if ok { self.weight_of(&w.hard) } else { f64::MAX };
        let mut converged = usize::from(ok);
        let mut best_count = usize::from(ok);
        let mut best_leg = 0;
        let mut out = RelayOutcome { converged: ok, iterations: its, legs: 1, weight: best, max_iter: self.config.pre_iter };
        if ok && (self.config.stop_after_first || self.config.solutions.is_some_and(|k| converged >= k)) {
            if logging {
                w.log_best[0] = true;
            }
            return out;
        }
        let stop = if self.config.stop_after_first { None } else { self.config.solutions };
        let mut total = its;
        for leg in 1..=self.config.legs {
            match &self.config.gammas {
                Some(rows) => {
                    let row = &rows[leg % rows.len()];
                    for (g, &x) in w.gamma.iter_mut().zip(row) {
                        *g = self.scaled(x);
                    }
                }
                None => {
                    for i in 0..self.num_vars {
                        let x = self.uniform.sample(&mut w.rng);
                        w.gamma[i] = self.scaled(x);
                    }
                }
            }
            self.reset_messages(w);
            let (ok, its) = self.leg(syndrome, self.config.leg_iter, w);
            total += its;
            out.legs += 1;
            w.log_legs += 1;
            if ok {
                converged += 1;
                let q = self.weight_of(&w.hard);
                if logging {
                    w.log_conv[leg] = true;
                    w.log_iter[leg] = its;
                }
                if q == best {
                    best_count += 1;
                }
                if q < best {
                    best_count = 1;
                    best_leg = leg;
                    best = q;
                    self.keep(w);
                    out.converged = true;
                    out.weight = q;
                    out.max_iter = self.config.leg_iter;
                }
                if stop.is_some_and(|k| converged >= k) {
                    break;
                }
            }
        }
        if logging && best_count == 1 {
            w.log_best[best_leg] = true;
        }
        out.iterations = total;
        out
    }
}

impl<N: RelayNum> RelayCoreWork<N> {
    /// Become a copy of `other` (a decoder's state as it was), keeping this one's buffers: what
    /// a parallel batch does before each row, so that every row decodes from the same state on
    /// one work per thread.
    pub fn copy_from(&mut self, other: &RelayCoreWork<N>) {
        let RelayCoreWork { v2c, c2v, posterior, gamma, hard, decoded, rng, correction, decoded_detectors, posterior_ratios, log_iter, log_conv, log_best, log_legs } = other;
        self.v2c.clone_from(v2c);
        self.c2v.clone_from(c2v);
        self.posterior.clone_from(posterior);
        self.gamma.clone_from(gamma);
        self.hard.clone_from(hard);
        self.decoded.clone_from(decoded);
        self.rng.clone_from(rng);
        self.correction.clone_from(correction);
        self.decoded_detectors.clone_from(decoded_detectors);
        self.posterior_ratios.clone_from(posterior_ratios);
        self.log_iter.clone_from(log_iter);
        self.log_conv.clone_from(log_conv);
        self.log_best.clone_from(log_best);
        self.log_legs = *log_legs;
    }

    /// IBM's log of the last decode, one line per leg: "leg (−1 for the first), iterations,
    /// converged, unique best".
    pub fn log_lines(&self) -> Vec<String> {
        (0..self.log_legs.min(self.log_iter.len())).map(|s| format!("{}, {}, {}, {}", s as i64 - 1, self.log_iter[s], self.log_conv[s] as u8, self.log_best[s] as u8)).collect()
    }
}

/// IBM's `bound_value_magnitude`.
fn bound<N: RelayNum>(value: N, max: N) -> N {
    if value < max.neg() {
        max.neg()
    } else if value > max {
        max
    } else {
        value
    }
}

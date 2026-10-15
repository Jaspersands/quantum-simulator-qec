//! Argument reweighting (Xie, Yoshioka, Tsubouchi and Li, arXiv 2601.17757): post-selection by
//! decoding again under a model that makes the first correction less likely.
//!
//! WHY THIS EXISTS
//! ---------------
//! A shot whose most likely correction is far more likely than any other is one a decoder gets
//! right; one with a close rival is one to discard. Argument reweighting asks the decoder itself:
//! it decodes, suppresses the faults of its correction c (each prior p to p^b, the ratio test;
//! or so that c as a whole loses a factor e^-b, the gap test), and decodes again. The shot is
//! kept if the second correction is c (the physical-error criterion), or has c's logical effect
//! (the two-round logical criterion); or if, suppressing the second correction too, a third
//! decode still has c's logical effect (the three-round criterion, the paper's best at low
//! rejection). Raising b rejects more and keeps fewer logical errors. It needs only a decoder
//! that weighs faults by their priors and returns its correction: here matching, union-find,
//! BP+OSD, BP+LSD and Relay-BP, the paper's.
//!
//! Each thread decodes on its own copy of the decoder, and a shot changes only the priors of
//! the faults it suppresses (the exact ratio test, which raises every other fault's prior, changes
//! them all), putting them back after.

use crate::lsd::{BpLsd, BpLsdWork};
use crate::osd::{BpOsd, BpOsdWork};
use crate::relay::{Relay, RelayWork};
use crate::sparse::union_find::UfScratch;
use crate::sparse::{Scratch, SparseGraph};

/// How the first correction is suppressed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Test {
    /// Each of its faults' priors p to p^b (the paper's Eq. 6, its practical ratio test).
    Ratio,
    /// The same, and every other fault's p to 1 − (1 − p)^b, so that c's likelihood is exactly
    /// its b-th power (Appendix F).
    ExactRatio,
    /// Its faults' priors lowered in proportion to their share of ln Pr(c), so c loses e^-b
    /// in all (Appendix G).
    Gap,
}

/// When a shot is kept.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Criterion {
    /// The second correction is the first.
    Physical,
    /// The second correction has the first's logical effect.
    TwoRound,
    /// So does a third, decoded with the second suppressed too.
    ThreeRound,
}

/// A decoder that weighs faults by their priors and returns its correction.
pub enum Inner {
    /// Faults are the matching graph's edges.
    Matching(SparseGraph),
    UnionFind(SparseGraph),
    /// Faults are the model's, each with its observables.
    BpOsd(BpOsd, Vec<u64>),
    BpLsd(BpLsd, Vec<u64>),
    Relay(Relay, Vec<u64>, u64),
}

pub struct ArgumentReweighting {
    inner: Inner,
    priors: Vec<f64>,
    pub b: f64,
    pub test: Test,
    pub criterion: Criterion,
}

/// One shot's outcome.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ArOutcome {
    /// The first correction's observables.
    pub observables: u64,
    pub accepted: bool,
    /// The decodes run.
    pub rounds: u8,
    /// No correction explained the shot.
    pub failed: bool,
}

enum State {
    Matching(Box<Scratch>),
    UnionFind(Box<UfScratch>),
    BpOsd(Box<BpOsd>, Box<BpOsdWork>),
    BpLsd(Box<BpLsd>, Box<BpLsdWork>),
    Relay(Box<Relay>, Box<RelayWork>),
}

/// One thread's copy of the decoder, and a shot's changed priors.
pub struct ArWork {
    state: State,
    p: Vec<f64>,
    changed: Vec<bool>,
    changed_list: Vec<u32>,
    patch: Vec<(u32, i64)>,
}

/// The smallest prior kept: below it a weight or log-likelihood ratio would be infinite.
const TINY: f64 = 1e-300;

impl ArgumentReweighting {
    pub fn new(inner: Inner, b: f64, test: Test, criterion: Criterion) -> Result<ArgumentReweighting, String> {
        let ok = match test {
            Test::Gap => b > 0.0,
            _ => b > 1.0,
        };
        if !ok || !b.is_finite() {
            return Err(format!("b must be {} for the {} test, not {b}", if test == Test::Gap { "positive" } else { "above 1" }, if test == Test::Gap { "gap" } else { "ratio" }));
        }
        let priors: Vec<f64> = match &inner {
            Inner::Matching(g) | Inner::UnionFind(g) => (0..g.num_edges() as u32).map(|id| 1.0 / (1.0 + g.float_weight(id).exp())).collect(),
            Inner::BpOsd(d, _) => d.bp.prior_llr().iter().map(|&l| 1.0 / (1.0 + l.exp())).collect(),
            Inner::BpLsd(d, _) => d.bp.prior_llr().iter().map(|&l| 1.0 / (1.0 + l.exp())).collect(),
            Inner::Relay(d, _, _) => d.priors(),
        };
        Ok(ArgumentReweighting { inner, priors, b, test, criterion })
    }

    pub fn work(&self) -> ArWork {
        let state = match &self.inner {
            Inner::Matching(g) => State::Matching(Box::new(Scratch::new(g))),
            Inner::UnionFind(g) => State::UnionFind(Box::new(g.union_find_scratch())),
            Inner::BpOsd(d, _) => State::BpOsd(Box::new(d.clone()), Box::new(d.work())),
            Inner::BpLsd(d, _) => State::BpLsd(Box::new(d.clone()), Box::new(d.work())),
            Inner::Relay(d, _, _) => State::Relay(Box::new(d.clone()), Box::new(d.work())),
        };
        let n = self.priors.len();
        ArWork { state, p: self.priors.clone(), changed: vec![false; n], changed_list: Vec::new(), patch: Vec::new() }
    }

    /// Fault `q`'s prior is now `p` (for the rest of the shot).
    fn set(&self, w: &mut ArWork, q: u32, p: f64) {
        let p = p.clamp(TINY, 0.5);
        w.p[q as usize] = p;
        if !w.changed[q as usize] {
            w.changed[q as usize] = true;
            w.changed_list.push(q);
        }
        match &mut w.state {
            State::BpOsd(d, _) => d.set_prior(q as usize, p),
            State::BpLsd(d, _) => d.bp.set_prior(q as usize, p),
            State::Relay(d, _) => d.set_prior(q as usize, p),
            State::Matching(_) | State::UnionFind(_) => {}
        }
    }

    /// Suppress correction `c` under the current priors.
    fn reweight(&self, w: &mut ArWork, c: &[u32]) {
        match self.test {
            Test::Ratio => {
                for &q in c {
                    let p = w.p[q as usize];
                    self.set(w, q, p.powf(self.b));
                }
            }
            Test::ExactRatio => {
                let mut in_c = vec![false; w.p.len()];
                for &q in c {
                    in_c[q as usize] = true;
                }
                for q in 0..w.p.len() as u32 {
                    let p = w.p[q as usize];
                    let to = if in_c[q as usize] { p.powf(self.b) } else { 1.0 - (1.0 - p).powf(self.b) };
                    self.set(w, q, to);
                }
            }
            Test::Gap => {
                let ln_pc: f64 = c.iter().map(|&q| w.p[q as usize].ln()).sum();
                for &q in c {
                    let p = w.p[q as usize];
                    self.set(w, q, (-(p.ln() / ln_pc) * self.b).exp() * p);
                }
            }
        }
    }

    /// One decode on the current priors: its observables and correction (sorted fault ids).
    fn round(&self, w: &mut ArWork, defects: &[u32], syndrome: &[u8], seed: u64) -> Option<(u64, Vec<u32>)> {
        let faults = |correction: &[u8]| -> Vec<u32> { correction.iter().enumerate().filter(|x| *x.1 != 0).map(|x| x.0 as u32).collect() };
        let fold = |c: &[u32], obs: &[u64]| c.iter().fold(0u64, |a, &q| a ^ obs[q as usize]);
        match (&self.inner, &mut w.state) {
            (Inner::Matching(g), State::Matching(s)) => {
                w.patch.clear();
                w.patch.extend(w.changed_list.iter().map(|&q| (q, crate::dem_decoder::edge_weight(w.p[q as usize]).1)));
                let (pred, mut c) = g.decode_edge_ids_patched(s, defects, &w.patch).ok()?;
                c.sort_unstable();
                Some((pred.observables, c))
            }
            (Inner::UnionFind(g), State::UnionFind(s)) => {
                w.patch.clear();
                w.patch.extend(w.changed_list.iter().map(|&q| (q, crate::dem_decoder::edge_weight(w.p[q as usize]).1)));
                let obs = g.decode_union_find_patched(s, defects, &w.patch)?;
                let mut c = s.correction.clone();
                c.sort_unstable();
                Some((obs, c))
            }
            (Inner::BpOsd(_, obs), State::BpOsd(d, work)) => {
                d.decode(syndrome, work);
                let c = faults(&work.correction);
                Some((fold(&c, obs), c))
            }
            (Inner::BpLsd(_, obs), State::BpLsd(d, work)) => {
                d.decode(syndrome, work);
                let c = faults(&work.correction);
                Some((fold(&c, obs), c))
            }
            (Inner::Relay(_, obs, _), State::Relay(d, work)) => {
                d.reseed(work, seed);
                d.decode(syndrome, work);
                let c = faults(&work.correction);
                Some((fold(&c, obs), c))
            }
            _ => unreachable!("a work serves the decoder it was made for"),
        }
    }

    /// Decode one shot (its detection events as sorted `defects` and as one byte per detector)
    /// and decide whether to keep it. `shot` seeds Relay-BP's memory strengths, the same in
    /// every round.
    pub fn decode(&self, defects: &[u32], syndrome: &[u8], shot: u64, w: &mut ArWork) -> ArOutcome {
        let seed = match &self.inner {
            Inner::Relay(_, _, seed) => crate::batch::shot_seed(*seed, shot),
            _ => 0,
        };
        let mut out = ArOutcome::default();
        let outcome = (|| {
            out.rounds = 1;
            let (l, c) = self.round(w, defects, syndrome, seed)?;
            out.observables = l;
            if c.is_empty() {
                return Some(true);
            }
            self.reweight(w, &c);
            out.rounds = 2;
            let (l2, c2) = self.round(w, defects, syndrome, seed)?;
            Some(match self.criterion {
                Criterion::Physical => c2 == c,
                Criterion::TwoRound => l2 == l,
                Criterion::ThreeRound => {
                    if l2 != l {
                        false
                    } else {
                        self.reweight(w, &c2);
                        out.rounds = 3;
                        self.round(w, defects, syndrome, seed)?.0 == l
                    }
                }
            })
        })();
        match outcome {
            Some(a) => out.accepted = a,
            None => out.failed = true,
        }
        // Put back the priors the shot changed.
        let changed = std::mem::take(&mut w.changed_list);
        for &q in &changed {
            let p = self.priors[q as usize];
            w.p[q as usize] = p;
            w.changed[q as usize] = false;
            match &mut w.state {
                State::BpOsd(d, _) => d.set_prior(q as usize, p),
                State::BpLsd(d, _) => d.bp.set_prior(q as usize, p),
                State::Relay(d, _) => d.set_prior(q as usize, p),
                State::Matching(_) | State::UnionFind(_) => {}
            }
        }
        w.changed_list = changed;
        w.changed_list.clear();
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dem::Dem;
    use crate::dem_decoder::DemDecoder;

    /// B - D0 - D1 - B with the left boundary edge flipping L0: events on both detectors are
    /// explained by the middle edge (p = 0.1), or by both boundary edges (0.01, the other class).
    fn line(make: fn(SparseGraph) -> Inner, b: f64, test: Test, criterion: Criterion) -> ArgumentReweighting {
        let dem = Dem::parse("error(0.1) D0 L0\nerror(0.1) D0 D1\nerror(0.1) D1").unwrap();
        let (g, _) = DemDecoder::new(&dem).unwrap().into_parts(false);
        ArgumentReweighting::new(make(g), b, test, criterion).unwrap()
    }

    #[test]
    fn a_shot_is_kept_while_its_correction_stays_the_more_likely() {
        // Raised to p^b the middle edge stays above the pair's 0.01 below b = 2 (and, suppressed
        // again in a third round when the second correction is it too, p^(b²) below b = √2).
        for make in [Inner::Matching as fn(SparseGraph) -> Inner, Inner::UnionFind] {
            for criterion in [Criterion::Physical, Criterion::TwoRound, Criterion::ThreeRound] {
                for (b, kept) in [(1.3, true), (2.5, false)] {
                    let ar = line(make, b, Test::Ratio, criterion);
                    let mut w = ar.work();
                    let out = ar.decode(&[0, 1], &[1, 1], 0, &mut w);
                    assert_eq!((out.observables, out.accepted, out.failed), (0, kept, false), "b {b}, {criterion:?}");
                    assert_eq!(out.rounds, if criterion == Criterion::ThreeRound && kept { 3 } else { 2 });
                    // The priors are back: the same shot again decides the same.
                    assert!(w.changed_list.is_empty() && w.p == ar.priors && !w.changed.iter().any(|&c| c));
                    assert_eq!(ar.decode(&[0, 1], &[1, 1], 0, &mut w), out);
                }
            }
        }
    }

    #[test]
    fn the_gap_test_takes_e_to_the_minus_b_off_the_correction() {
        // The likelihood ratio is 10: a gap below ln 10 keeps, above rejects.
        for (b, kept) in [(2.0, true), (2.5, false)] {
            let ar = line(Inner::Matching, b, Test::Gap, Criterion::Physical);
            assert_eq!(ar.decode(&[0, 1], &[1, 1], 0, &mut ar.work()).accepted, kept, "b {b}");
        }
        // A shot with nothing to correct is kept after one decode; one nothing explains fails.
        let ar = line(Inner::Matching, 1.5, Test::ExactRatio, Criterion::ThreeRound);
        let out = ar.decode(&[], &[0, 0], 0, &mut ar.work());
        assert!(out.accepted && out.rounds == 1);
        let dem = Dem::parse("error(0.1) D0 D1 L0").unwrap();
        let (g, _) = DemDecoder::new(&dem).unwrap().into_parts(false);
        let ar = ArgumentReweighting::new(Inner::Matching(g), 1.5, Test::Ratio, Criterion::Physical).unwrap();
        assert!(ar.decode(&[0], &[1, 0], 0, &mut ar.work()).failed);
    }

    #[test]
    fn b_is_checked() {
        let dem = Dem::parse("error(0.1) D0").unwrap();
        let g = || DemDecoder::new(&dem).unwrap().into_parts(false).0;
        assert!(ArgumentReweighting::new(Inner::Matching(g()), 1.0, Test::Ratio, Criterion::Physical).is_err());
        assert!(ArgumentReweighting::new(Inner::Matching(g()), f64::NAN, Test::ExactRatio, Criterion::Physical).is_err());
        assert!(ArgumentReweighting::new(Inner::Matching(g()), 0.0, Test::Gap, Criterion::Physical).is_err());
        assert!(ArgumentReweighting::new(Inner::Matching(g()), 0.5, Test::Gap, Criterion::Physical).is_ok());
    }
}

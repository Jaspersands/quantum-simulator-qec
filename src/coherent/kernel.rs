//! The spacetime kernel of a circuit: its low-weight generators, found by hashing, completed to
//! a basis of the whole kernel, and per shot the classes of the faults that fired, summed over
//! products of the generators those faults touch.
//!
//! A set of locations is in the kernel when the gauges can match what it flips, which the gauge
//! solver's reduction tests exactly: reduce each location's flips by the gauges and the residues
//! of a kernel element sum to zero. The local generators sought are
//!
//! - single locations whose residue is zero (a rotation the state does not see);
//! - pairs with equal residues (two places the same fault can sit, say either side of a gate,
//!   or the two qubits of a weight-two stabiliser);
//! - three and four locations whose residues sum to zero, built from pairs of locations that
//!   share a detector (a stabiliser's worth of faults in one time step).
//!
//! They miss every element that is not a sum of such small pieces, and the one that matters
//! most is the commonest: a logical operator at one time times the same logical at another (in
//! a repetition memory, XXX in round 1 with XXX in round 2: weight 2d, no syndrome, no logical
//! effect). Its two halves give the same outcome, so their amplitudes add — logical errors at
//! different times accumulate coherently — and a kernel without it reads the twirl's rate (a d =
//! 3, 2-round memory at θ = 0.3: 0.00103 against the exact 0.00203). So the local generators
//! are completed: the whole kernel's null space modulo their span (`complete`), each new
//! direction shortened against the generators, appended as a non-local generator. That costs
//! one elimination over the locations' residues and is done up to `max_complete` locations;
//! past it the kernel is the local one and `complete` is false.
//!
//! With the whole kernel spanned, a shot sums every product of the generators (and the groups):
//! the class sum of `Program::weight`, exact. The sum is contracted (`contract`), at a cost
//! exponential in the width of the generators' interactions rather than in their number; with
//! the contraction past `max_contraction` table entries, and at most `SMALL_KERNEL` generators,
//! every product is listed once and summed. Otherwise the weight is truncated, as in 2.1. The
//! local generators go by clusters as before 2.1: those holding a fault that
//! fired and their neighbours, joined when they share a location, each cluster's sum over the
//! products of up to `order` of its generators, each other generator's 1 + r on its own, the
//! weight |Π S|² / Π N of the sums S and their squared sizes N. A non-local generator holding a
//! fault adds to Π S its connected products of up to `order` generators with its neighbours,
//! each times the factors it does not overlap (and to Π N their sizes): first order in the
//! non-local generators, so products of two that share no location, and products with
//! generators no fault reached that it overlaps, are left out. The estimate is
//! self-normalised.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use super::contract::{ContractionPlan, Refusal, ShotFactors};
use super::{Element, GaugeSolver, ParitySums, Program, Shot, Slot};
use crate::statevec::C;

/// Options for the coherent sampler's kernel.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CoherentOptions {
    /// The most generators multiplied together in a shot's class sum; 0 is taken as 1.
    pub order: usize,
    /// The most generators in a cluster before products are cut to pairs.
    pub max_cluster: usize,
    /// Whether to look for generators of three and four locations.
    pub quads: bool,
    /// How many steps of shared locations a shot's clusters reach past the generators its
    /// faults touch.
    pub hops: usize,
    /// The most locations whose kernel is completed past the local generators (one Gaussian
    /// elimination over their residues); 0 never completes it.
    pub max_complete: usize,
    /// The most table entries, in all, of the contraction that sums a shot's whole class; 0
    /// never contracts, and past `contract::MAX_ENTRIES` (2^31) it is 2^31.
    pub max_contraction: usize,
}

impl Default for CoherentOptions {
    fn default() -> Self {
        CoherentOptions { order: 3, max_cluster: 12, quads: true, hops: 1, max_complete: MAX_COMPLETE, max_contraction: MAX_CONTRACTION }
    }
}

/// The default `max_complete`: a d = 11 surface-code memory with a rotation after every tick
/// (about 9,000 locations) completes in about a second.
pub const MAX_COMPLETE: usize = 50_000;

/// The default `max_contraction`: 2^16 entries (a d = 7 surface memory over seven rounds, of
/// width 9, takes about 4,700).
pub const MAX_CONTRACTION: usize = 1 << 16;

/// With the kernel complete, not contracted and this many generators or fewer, a shot sums
/// every product of them, listed once (2^12 elements at most): exact, and the same set from
/// every shot. The joint sampler's limit too.
pub const SMALL_KERNEL: usize = 12;

/// The kernel's generators (the local ones first, then those completing them to a basis of the
/// whole kernel), and each location's generators.
pub struct Kernel {
    pub generators: Vec<Element>,
    /// How many of `generators`, at the end, are not local: the completion.
    pub num_nonlocal: usize,
    /// Whether the generators and the groups span the whole kernel (false past `max_complete`
    /// locations: interference between distant faults, a logical error at one time with one at
    /// another, is then left out).
    pub complete: bool,
    /// With a complete kernel: the contraction of its class sums, within `max_contraction`.
    plan: Option<ContractionPlan>,
    /// With a complete kernel and no plan: why.
    refusal: Option<Refusal>,
    /// With a complete kernel not contracted, of at most `SMALL_KERNEL` generators: every
    /// nonzero product of them, which each shot sums.
    span: Option<Vec<Element>>,
    /// Groups of locations that flip exactly the same outcomes: the same fault in different
    /// places. Their class sums have a closed form.
    pub groups: Vec<Group>,
    /// Each location's group's first member (itself if in none).
    rep_of: Vec<usize>,
    /// A random word per location: a set's key is the XOR of its locations'.
    zobrist: Vec<u64>,
    by_location: Vec<Vec<usize>>,
    options: CoherentOptions,
}

/// Locations that flip exactly the same outcomes, with each one's operator relative to the
/// first's: toggling an even set T of them has ratio Π_{i∈T} τ_i, τ_i the location's own toggle
/// factor times `mu[i]`.
#[derive(Clone, Debug)]
pub struct Group {
    pub members: Vec<usize>,
    pub mu: Vec<C>,
}

/// A residue as a hashable key.
type Key = Vec<u64>;

fn xor(a: &[u64], b: &[u64]) -> Key {
    a.iter().zip(b).map(|(x, y)| x ^ y).collect()
}

/// Each location's residue: its flips reduced by the gauges', zero when the gauges match them.
pub(crate) fn residues(p: &Program, solver: &GaugeSolver) -> Vec<Key> {
    let words = p.slots.len().div_ceil(64);
    (0..p.locations.len())
        .map(|l| {
            let mut v = vec![0u64; words];
            for &s in &solver.flips[l] {
                v[s / 64] ^= 1 << (s % 64);
            }
            solver.reduce(&mut v);
            v
        })
        .collect()
}

impl Kernel {
    pub fn new(p: &Program, solver: &GaugeSolver, options: CoherentOptions) -> Kernel {
        let nl = p.locations.len();
        let words = p.slots.len().div_ceil(64);
        let residue: Vec<Key> = residues(p, solver);
        let zero = vec![0u64; words];
        // Groups: equal flips, operators equal up to a phase that composes consistently.
        let mut by_flips: HashMap<&Vec<usize>, Vec<usize>> = HashMap::new();
        for l in 0..nl {
            if !solver.flips[l].is_empty() {
                by_flips.entry(&solver.flips[l]).or_default().push(l);
            }
        }
        let mut groups: Vec<Group> = Vec::new();
        let mut is_rep = vec![true; nl];
        let phase = |e: &Element| {
            let v = [C::ONE, C::I, C::new(-1.0, 0.0), C::new(0.0, -1.0)][e.k as usize % 4];
            if e.offset { C::ZERO - v } else { v }
        };
        let mut lists: Vec<&Vec<usize>> = by_flips.values().filter(|v| v.len() > 1).collect();
        lists.sort();
        for list in lists {
            // Runs of the list with no coherent location between that anticommutes with their
            // operator: such a location's fault would change the run's proposal.
            let mut runs: Vec<Vec<usize>> = Vec::new();
            let mut blockers: Vec<usize> = Vec::new();
            for &l in list.iter() {
                let start_new = match runs.last() {
                    None => true,
                    Some(run) => blockers.iter().any(|&b| b > run[0] && b < l),
                };
                if start_new {
                    runs.push(vec![l]);
                    blockers = p.anticommuting_after(l);
                } else {
                    runs.last_mut().unwrap().push(l);
                }
            }
            for run in runs {
                let rep = run[0];
                let mut members = vec![rep];
                let mut mu = vec![C::ONE];
                for &l in &run[1..] {
                    let Some(e) = p.element(&[rep, l], solver) else { continue };
                    if !e.z.is_empty() {
                        continue;
                    }
                    let m = phase(&e);
                    // Consistent with every member taken so far.
                    let ok = members.iter().zip(&mu).skip(1).all(|(&j, &mj)| {
                        let pair = if j < l { [j, l] } else { [l, j] };
                        p.element(&pair, solver).is_some_and(|e2| e2.z.is_empty() && (phase(&e2) - mj * m).norm2() < 1e-12)
                    });
                    if ok {
                        members.push(l);
                        mu.push(m);
                    }
                }
                if members.len() > 1 {
                    for &l in &members[1..] {
                        is_rep[l] = false;
                    }
                    groups.push(Group { members, mu });
                }
            }
        }
        let mut candidates: BTreeSet<Vec<usize>> = BTreeSet::new();
        // Singles, and pairs of equal residue (a chain through each group).
        let mut by_residue: HashMap<&Key, Vec<usize>> = HashMap::new();
        for (l, r) in residue.iter().enumerate() {
            if !is_rep[l] {
                continue;
            }
            if *r == zero {
                candidates.insert(vec![l]);
            } else {
                by_residue.entry(r).or_default().push(l);
            }
        }
        for group in by_residue.values() {
            for w in group.windows(2) {
                candidates.insert(vec![w[0], w[1]]);
            }
        }
        if options.quads {
            // Pairs of locations sharing a detector (or observable), keyed by their residues' sum.
            let record_of: Vec<Option<usize>> = p.slots.iter().map(|s| if let Slot::Record(k) = s { Some(*k) } else { None }).collect();
            let mut checks_of_record: Vec<Vec<usize>> = vec![Vec::new(); p.num_records];
            for (d, recs) in p.detectors.iter().chain(&p.observables).enumerate() {
                for &r in recs {
                    checks_of_record[r].push(d);
                }
            }
            let nchecks = p.detectors.len() + p.observables.len();
            let mut at_check: Vec<Vec<usize>> = vec![Vec::new(); nchecks];
            for l in 0..nl {
                let mut fired: Vec<usize> = Vec::new();
                for &s in &solver.flips[l] {
                    if let Some(k) = record_of[s] {
                        fired.extend(&checks_of_record[k]);
                    }
                }
                fired.sort_unstable();
                let mut odd = Vec::new();
                for d in fired {
                    if odd.last() == Some(&d) {
                        odd.pop();
                    } else {
                        odd.push(d);
                    }
                }
                if is_rep[l] {
                    for d in odd {
                        at_check[d].push(l);
                    }
                }
            }
            let mut pairs: HashMap<Key, Vec<(usize, usize)>> = HashMap::new();
            let mut seen: BTreeSet<(usize, usize)> = BTreeSet::new();
            for ls in &at_check {
                if ls.len() > 64 {
                    continue;
                }
                for (i, &a) in ls.iter().enumerate() {
                    for &b in &ls[i + 1..] {
                        let (a, b) = (a.min(b), a.max(b));
                        if residue[a] != residue[b] && seen.insert((a, b)) {
                            pairs.entry(xor(&residue[a], &residue[b])).or_default().push((a, b));
                        }
                    }
                }
            }
            for (key, list) in &pairs {
                // Triples: a pair summing to a single location's residue.
                if let Some(ls) = by_residue.get(key) {
                    for &(a, b) in list.iter().take(8) {
                        for &c in ls.iter().take(8) {
                            if c != a && c != b {
                                let mut g = vec![a, b, c];
                                g.sort_unstable();
                                candidates.insert(g);
                            }
                        }
                    }
                }
                // Quads: two disjoint pairs with the same sum.
                for (i, &(a, b)) in list.iter().enumerate().take(16) {
                    for &(c, d) in list[i + 1..].iter().take(16) {
                        if a != c && a != d && b != c && b != d {
                            let mut g = vec![a, b, c, d];
                            g.sort_unstable();
                            candidates.insert(g);
                        }
                    }
                }
            }
        }
        // Smallest first, each kept only if independent of those kept (over GF(2), by locations).
        let mut candidates: Vec<Vec<usize>> = candidates.into_iter().collect();
        candidates.sort_by_key(|g| g.len());
        let lw = nl.div_ceil(64);
        let mut basis: Vec<(usize, Vec<u64>)> = Vec::new();
        let mut generators: Vec<Element> = Vec::new();
        for g in candidates {
            let mut v = vec![0u64; lw];
            for &l in &g {
                v[l / 64] ^= 1 << (l % 64);
            }
            for (pivot, row) in &basis {
                if v[pivot / 64] >> (pivot % 64) & 1 == 1 {
                    v.iter_mut().zip(row).for_each(|(a, b)| *a ^= b);
                }
            }
            let Some(w) = v.iter().position(|&w| w != 0) else { continue };
            let Some(e) = p.element(&g, solver) else { continue };
            let pivot = w * 64 + v[w].trailing_zeros() as usize;
            for (_, row) in basis.iter_mut() {
                if row[pivot / 64] >> (pivot % 64) & 1 == 1 {
                    row.iter_mut().zip(&v).for_each(|(a, b)| *a ^= b);
                }
            }
            basis.push((pivot, v));
            generators.push(e);
        }
        // The rest of the kernel: what the local generators and the groups leave out. Reduced
        // by the groups' pairs (each member past a group's first replaced by the first) and then
        // by the generators' basis, a kernel element is off every location those pin (the members
        // past the first, the basis's pivots), and a nonzero element of their span is on one of
        // them: so the kernel modulo the local span is the null space of the free locations'
        // residues, and each vector of that null space is a new direction.
        let num_local = generators.len();
        let mut complete = nl <= options.max_complete;
        if complete {
            let mut pinned: Vec<bool> = is_rep.iter().map(|&r| !r).collect();
            for (pivot, _) in &basis {
                pinned[*pivot] = true;
            }
            let free: Vec<(Key, Vec<usize>)> = (0..nl).filter(|&l| !pinned[l]).map(|l| (residue[l].clone(), vec![l])).collect();
            let local: Vec<Vec<usize>> = generators.iter().map(|e| e.members.clone()).collect();
            for members in shorten(null_space(free), &local, nl, &|_| 1) {
                match p.element(&members, solver) {
                    Some(e) => generators.push(e),
                    // A kernel element always carries through; were one not to, say so.
                    None => complete = false,
                }
            }
        }
        let mut by_location = vec![Vec::new(); nl];
        for (i, g) in generators.iter().enumerate() {
            for &l in &g.members {
                by_location[l].push(i);
            }
        }
        let mut rep_of: Vec<usize> = (0..nl).collect();
        for g in &groups {
            for &l in &g.members {
                rep_of[l] = g.members[0];
            }
        }
        let zobrist = (0..nl as u64).map(|l| crate::batch_sampler::splitmix64(l.wrapping_add(0x9e37_79b9_7f4a_7c15))).collect();
        let num_nonlocal = generators.len() - num_local;
        let mut kernel = Kernel { generators, num_nonlocal, complete, plan: None, refusal: None, span: None, groups, rep_of, zobrist, by_location, options };
        // The whole kernel: contracted, or (a small one past the budget) every product of the
        // generators, once for every shot.
        if complete {
            match ContractionPlan::try_build(p, solver, &kernel, options.max_contraction) {
                Ok(plan) => kernel.plan = Some(plan),
                Err(why) => {
                    kernel.refusal = Some(why);
                    if kernel.generators.len() <= SMALL_KERNEL {
                        kernel.span = Some(whole_span(p, solver, &kernel.generators));
                    }
                }
            }
        }
        kernel
    }

    /// Whether every shot sums the whole kernel (complete, and contracted or of at most
    /// `SMALL_KERNEL` generators): the weights are then exact.
    pub fn is_whole(&self) -> bool {
        self.plan.is_some() || self.span.is_some()
    }

    /// With the class sums contracted, the most generators a table holds besides the one it
    /// sums out.
    pub fn contraction_width(&self) -> Option<usize> {
        self.plan.as_ref().map(ContractionPlan::width)
    }

    /// With the kernel complete and its class sums not contracted, why: the contraction off
    /// (`Refusal::Off`), past the budget (`Refusal::Budget`, with the entries a budget needs at
    /// least), past every budget (`Refusal::Width`), or a bug (`Refusal::Internal`, which release
    /// builds survive by listing a small kernel's span or truncating, and which the Python
    /// sampler warns of).
    pub fn contraction_refusal(&self) -> Option<Refusal> {
        self.refusal
    }

    /// The contraction of the class sums, if any.
    #[cfg(test)]
    pub(crate) fn plan(&self) -> Option<&ContractionPlan> {
        self.plan.as_ref()
    }

    /// The whole kernel's dimension: locations less the rank of their residues.
    pub fn dimension(&self, p: &Program, solver: &GaugeSolver) -> usize {
        let words = p.slots.len().div_ceil(64);
        let mut rows: Vec<(usize, Vec<u64>)> = Vec::new();
        for l in 0..p.locations.len() {
            let mut v = vec![0u64; words];
            for &s in &solver.flips[l] {
                v[s / 64] ^= 1 << (s % 64);
            }
            solver.reduce(&mut v);
            for (pivot, r) in &rows {
                if v[pivot / 64] >> (pivot % 64) & 1 == 1 {
                    v.iter_mut().zip(r).for_each(|(a, b)| *a ^= b);
                }
            }
            if let Some(w) = v.iter().position(|&w| w != 0) {
                rows.push((w * 64 + v[w].trailing_zeros() as usize, v));
            }
        }
        p.locations.len() - rows.len()
    }

    /// The dimension the generators and groups span.
    pub fn local_span(&self) -> usize {
        let n = self.rep_of.len();
        let lw = n.div_ceil(64);
        let mut rows: Vec<(usize, Vec<u64>)> = Vec::new();
        let vecs = self.generators.iter().map(|g| g.members.clone()).chain(self.groups.iter().flat_map(|g| g.members[1..].iter().map(move |&l| vec![g.members[0], l])));
        for members in vecs {
            let mut v = vec![0u64; lw];
            for &l in &members {
                v[l / 64] ^= 1 << (l % 64);
            }
            for (pivot, r) in &rows {
                if v[pivot / 64] >> (pivot % 64) & 1 == 1 {
                    v.iter_mut().zip(r).for_each(|(a, b)| *a ^= b);
                }
            }
            if let Some(w) = v.iter().position(|&w| w != 0) {
                rows.push((w * 64 + v[w].trailing_zeros() as usize, v));
            }
        }
        rows.len()
    }

    /// The groups, as `Program::set_merge` takes them.
    pub fn merge_groups(&self) -> Vec<(Vec<usize>, Vec<C>)> {
        self.groups.iter().map(|g| (g.members.clone(), g.mu.clone())).collect()
    }

    /// A shot's weight: with the whole kernel, its class sum over every product of the
    /// generators; otherwise by clusters of the generators its faults touch. `cache` holds the
    /// products already carried through the circuit (by their locations).
    pub fn weight(&self, p: &Program, shot: &Shot, solver: &GaugeSolver, cache: &mut HashMap<u64, Option<Element>>) -> f64 {
        let (w, eff) = self.groups_of(p, shot);
        if let Some(plan) = &self.plan {
            let (sum, norm) = plan.class_sums(&self.factors(p, shot, &eff));
            return w * sum.norm2() / norm;
        }
        self.listed_or_truncated(p, shot, solver, cache, w, &eff)
    }

    /// A shot's factors for the contraction, as `weight` reads them: (the groups' own factor of
    /// the weight, the factors).
    #[cfg(test)]
    pub(crate) fn shot_factors(&self, p: &Program, shot: &Shot) -> (f64, ShotFactors) {
        let (w, eff) = self.groups_of(p, shot);
        (w, self.factors(p, shot, &eff))
    }

    fn factors(&self, p: &Program, shot: &Shot, eff: &[Option<(C, f64)>]) -> ShotFactors {
        let (toggle, size) = eff
            .iter()
            .enumerate()
            .map(|(l, e)| match *e {
                Some(f) => f,
                None => {
                    let t = p.toggle(shot, l);
                    (t, t.norm2())
                }
            })
            .unzip();
        let sign = self.generators.iter().map(|g| g.z.iter().fold(false, |a, &s| a ^ shot.branch[s])).collect();
        ShotFactors { toggle, size, sign }
    }

    /// Each group as one location: its sums over the even and the odd sets of its members to
    /// toggle, in closed form. The even sum is the group's own factor of the weight (the first
    /// value); in any element holding the group's first member, odd over even takes that
    /// member's place (the second, by location).
    fn groups_of(&self, p: &Program, shot: &Shot) -> (f64, Vec<Option<(C, f64)>>) {
        let mut w = 1.0;
        let mut eff: Vec<Option<(C, f64)>> = vec![None; self.rep_of.len()];
        for g in &self.groups {
            let sums = ParitySums::new(g.members.iter().zip(&g.mu).map(|(&l, &mu)| p.toggle(shot, l) * mu));
            let (even, odd) = (sums.even, sums.odd);
            // A class that cancels exactly (even = 0) has weight 0 unless an element moves it
            // to the odd sum; a tiny stand-in keeps that limit finite, as for the raw sums.
            let even = if even.norm2() < 1e-200 { C::new(1e-100, 0.0) } else { even };
            let (mut neven, mut nodd) = (sums.neven, sums.nodd);
            if !p.merge.is_empty() {
                let taus = g.members.iter().zip(&g.mu).map(|(&l, &mu)| p.unfired_toggle(l, mu, shot.anti[l]));
                let odd_p = ParitySums::new(taus).proposal();
                let last = *g.members.last().unwrap();
                let fired = shot.fired[last];
                let (rho, rho_other) = if fired { (odd_p, 1.0 - odd_p) } else { (1.0 - odd_p, odd_p) };
                // Raw sums use neven = rho/q_drawn and nodd = rho_other/q_drawn.
                // For the canonical draw (only the last member may fire),
                // q_drawn * N = 1, N = product(1 + |toggle|²). Dividing *both*
                // amplitude norms and twirl masses by N therefore leaves rho and
                // rho_other directly. No underflowing product of cos² is needed.
                neven = rho;
                nodd = rho_other;
            }
            // In unmerged shots neven is the twirl's parity mass divided by N;
            // in merged shots it is rho. The shared amplitude scale cancels out
            // of odd/even but remains squared in this group's weight factor.
            w *= (2.0 * sums.log_scale + even.norm2().ln() - neven.ln()).exp();
            let inv = even.conj().scale(1.0 / even.norm2());
            eff[g.members[0]] = Some((odd * inv, nodd / neven));
        }
        (w, eff)
    }

    /// The weight past the contraction: every product of a small kernel's generators, listed;
    /// otherwise truncated.
    fn listed_or_truncated(&self, p: &Program, shot: &Shot, solver: &GaugeSolver, cache: &mut HashMap<u64, Option<Element>>, w: f64, eff: &[Option<(C, f64)>]) -> f64 {
        // An element's ratio and its squared size, groups standing in for their first members.
        let term = |e: &Element| -> (C, f64) {
            let (mut r, mut n) = (p.prefactor(shot, e), 1.0);
            for &l in &e.members {
                match eff[l] {
                    Some((c, m)) => {
                        r = r * c;
                        n *= m;
                    }
                    None => {
                        let t = p.toggle(shot, l);
                        r = r * t;
                        n *= t.norm2();
                    }
                }
            }
            (r, n)
        };
        // The whole kernel: |1 + Σ r|² / (1 + Σ |r|²) over every element (`Program::weight`'s
        // class sum, the groups in closed form), exact.
        if let Some(span) = &self.span {
            let (mut sum, mut norm) = (C::ONE, 1.0);
            for e in span {
                let (r, n) = term(e);
                sum = sum + r;
                norm += n;
            }
            return w * sum.norm2() / norm;
        }
        // Truncated: the local generators by clusters, as before 2.1, and the non-local ones on
        // top. Every factor below is a class sum S over some generators' products with its
        // squared size N, the weight |Π S|² / Π N; the non-local terms add to Π S and Π N.
        let nloc = self.generators.len() - self.num_nonlocal;
        let mut fired = vec![false; self.rep_of.len()];
        for (l, &f) in shot.fired.iter().enumerate() {
            if f {
                fired[self.rep_of[l]] = true;
            }
        }
        // The local generators the faults touch, with their local neighbours (generators sharing
        // a location), whose products with them move the faults on.
        let touched = self.touched(&self.generators, &self.by_location, &fired, self.options.hops, &|g| g < nloc);
        // The non-local generators the faults reach: holding one, or sharing a location with a
        // touched generator (a logical at another time whose representative here is not the
        // faults' own, one product of local generators away).
        let mut reached = self.touched(&self.generators, &self.by_location, &fired, 0, &|g| g >= nloc);
        for &t in &touched {
            for &l in &self.generators[t].members {
                reached.extend(self.by_location[l].iter().copied().filter(|&g| g >= nloc));
            }
        }
        reached.sort_unstable();
        reached.dedup();
        // Every other generator on its own: 1 + r, 1 + |r|².
        let mut alone: Vec<Option<(C, f64)>> = vec![None; self.generators.len()];
        let (mut pu, mut qu) = (C::ONE, 1.0);
        let (mut t, mut s) = (touched.iter().peekable(), reached.iter().peekable());
        for (g, e) in self.generators.iter().enumerate() {
            if t.peek() == Some(&&g) {
                t.next();
                continue;
            }
            if s.peek() == Some(&&g) {
                s.next();
                continue;
            }
            let (r, n) = term(e);
            alone[g] = Some((C::ONE + r, 1.0 + n));
            pu = pu * (C::ONE + r);
            qu *= 1.0 + n;
        }
        let mut element_of = |members: &[usize]| p.element(members, solver);
        // Clusters of touched local generators (joined when they share a location): the sum over
        // their connected products of up to `order` generators.
        let clusters = self.clusters(&self.generators, &touched);
        let mut cluster_of: HashMap<usize, usize> = HashMap::new();
        let mut sums: Vec<(C, f64)> = Vec::with_capacity(clusters.len());
        for (k, gens) in clusters.iter().enumerate() {
            let (mut sum, mut norm) = (C::ONE, 1.0);
            for (_, e) in self.products(&self.generators, gens, &self.options, cache, &mut element_of, &|_| true, None) {
                let (r, m) = term(&e);
                sum = sum + r;
                norm += m;
            }
            for &g in gens {
                cluster_of.insert(g, k);
            }
            sums.push((sum, norm));
        }
        let (mut total, mut size) = (pu, qu);
        for &(sum, norm) in &sums {
            total = total * sum;
            size *= norm;
        }
        if reached.is_empty() {
            return w * total.norm2() / size;
        }
        // A reached non-local generator g (a logical at one time with the same logical at
        // another): every element g ⊕ k, k a product of the generators around g, in g's frame.
        // Those sharing a location with g (whole touched clusters, untouched generators, reached
        // ones after g) fall into clusters D of their own, joined when they share a location (g
        // left out: it would join them all). Within each, the connected products k of up to
        // `order` of D's generators give A_D = Σ r(g ⊕ k) / r(g), with 1 for k = ∅: exact in the
        // rotations' toggles where the D are apart (in a memory, g's two ends). g's term is
        // r(g) Π A_D times every factor around it does not take the place of. An element is
        // counted with the first reached generator it holds: those before g are left out here.
        let mut position = vec![usize::MAX; self.generators.len()];
        for (i, &g) in reached.iter().enumerate() {
            position[g] = i;
        }
        let (mut extra, mut extra_size) = (C::ZERO, 0.0);
        for (i, &g) in reached.iter().enumerate() {
            let ge = &self.generators[g];
            let (rg, ng) = term(ge);
            if rg.norm2() < 1e-280 || ng < 1e-280 {
                continue;
            }
            let inv = rg.conj().scale(1.0 / rg.norm2());
            let mut around: Vec<usize> = Vec::new();
            let mut taken = vec![false; sums.len()];
            for &l in &ge.members {
                for &h in &self.by_location[l] {
                    if h == g {
                        continue;
                    }
                    if let Some(&k) = cluster_of.get(&h) {
                        if !taken[k] {
                            taken[k] = true;
                            around.extend_from_slice(&clusters[k]);
                        }
                    } else if position[h] == usize::MAX || position[h] > i {
                        around.push(h);
                    }
                }
            }
            around.sort_unstable();
            around.dedup();
            // And the non-local generators chained to those around g, up to `order` steps
            // further: the same logical at further times.
            let mut frontier: Vec<usize> = around.iter().copied().filter(|&h| h >= nloc).collect();
            for _ in 1..self.options.order {
                let mut next: Vec<usize> = frontier.iter().flat_map(|&h| self.generators[h].members.iter().flat_map(|&l| self.by_location[l].iter().copied())).filter(|&x| x >= nloc && x != g && (position[x] == usize::MAX || position[x] > i) && around.binary_search(&x).is_err()).collect();
                next.sort_unstable();
                next.dedup();
                if next.is_empty() {
                    break;
                }
                around.extend_from_slice(&next);
                around.sort_unstable();
                frontier = next;
            }
            let (mut prod, mut norm) = (rg, ng);
            for d in self.clusters(&self.generators, &around) {
                let (mut a, mut b) = (C::ONE, 1.0);
                for (_, e) in self.products(&self.generators, &d, &self.options, cache, &mut element_of, &|_| true, Some(&ge.members)) {
                    let (r, m) = term(&e);
                    a = a + r * inv;
                    b += m / ng;
                }
                prod = prod * a;
                norm *= b;
            }
            // The touched clusters g does not overlap.
            for (k, &(sum, n)) in sums.iter().enumerate() {
                if !taken[k] {
                    prod = prod * sum;
                    norm *= n;
                }
            }
            // The untouched generators not around g: their product with those around divided
            // out (each near 1; directly if one is not).
            let around_has = |h: usize| around.binary_search(&h).is_ok();
            if around.iter().any(|&h| alone[h].is_some_and(|(a, _)| a.norm2() < 0.25)) {
                for (h, a) in alone.iter().enumerate() {
                    if let Some((a, n)) = a {
                        if !around_has(h) {
                            prod = prod * *a;
                            norm *= n;
                        }
                    }
                }
            } else {
                prod = prod * pu;
                norm *= qu;
                for &h in &around {
                    if let Some((a, n)) = alone[h] {
                        prod = (prod * a.conj()).scale(1.0 / a.norm2());
                        norm /= n;
                    }
                }
            }
            // The reached generators after g, not around it, on their own.
            for &h in &reached[i + 1..] {
                if !around_has(h) {
                    let (r, n) = term(&self.generators[h]);
                    prod = prod * (C::ONE + r);
                    norm *= 1.0 + n;
                }
            }
            extra = extra + prod;
            extra_size += norm;
        }
        w * (total + extra).norm2() / (size + extra_size)
    }
}

impl Kernel {
    /// The generators (of `generators`, indexed by `by_location`, those `include` takes)
    /// holding a fired location, with `hops` steps of such neighbours sharing a location;
    /// sorted. The joint sampler's generator list carries each group's pairs itself.
    pub(crate) fn touched(&self, generators: &[Element], by_location: &[Vec<usize>], fired: &[bool], hops: usize, include: &dyn Fn(usize) -> bool) -> Vec<usize> {
        let mut touched: Vec<usize> = fired.iter().enumerate().filter(|(_, &f)| f).flat_map(|(l, _)| by_location[l].iter().copied()).filter(|&g| include(g)).collect();
        touched.sort_unstable();
        touched.dedup();
        for _ in 0..hops {
            let neighbours: Vec<usize> = touched.iter().flat_map(|&g| generators[g].members.iter().flat_map(|&l| by_location[l].iter().copied())).filter(|&g| include(g)).collect();
            touched.extend(neighbours);
            touched.sort_unstable();
            touched.dedup();
        }
        touched
    }

    /// The touched generators in clusters: joined when they share a location.
    pub(crate) fn clusters(&self, generators: &[Element], touched: &[usize]) -> Vec<Vec<usize>> {
        let n = touched.len();
        let mut parent: Vec<usize> = (0..n).collect();
        fn find(parent: &mut [usize], mut i: usize) -> usize {
            while parent[i] != i {
                parent[i] = parent[parent[i]];
                i = parent[i];
            }
            i
        }
        let mut owner: HashMap<usize, usize> = HashMap::new();
        for (i, &g) in touched.iter().enumerate() {
            for &l in &generators[g].members {
                if let Some(&j) = owner.get(&l) {
                    let (a, b) = (find(&mut parent, i), find(&mut parent, j));
                    parent[a] = b;
                } else {
                    owner.insert(l, i);
                }
            }
        }
        let mut clusters: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        for i in 0..n {
            let root = find(&mut parent, i);
            clusters.entry(root).or_default().push(touched[i]);
        }
        clusters.into_values().collect()
    }

    /// A cluster's elements: every connected set of up to `order` of its generators (each
    /// sharing a location with another; pairs at most past `max_cluster` generators) that
    /// `keep` takes, by Wernicke's enumeration, each resulting location set once, as
    /// `element_of` makes it, with the generators (ids in `generators`) it is the product of.
    /// With `shift`, each product times that element (its locations), outside the cluster.
    #[allow(clippy::type_complexity, clippy::too_many_arguments)]
    pub(crate) fn products(&self, generators: &[Element], gens: &[usize], options: &CoherentOptions, cache: &mut HashMap<u64, Option<Element>>, element_of: &mut dyn FnMut(&[usize]) -> Option<Element>, keep: &dyn Fn(&[usize]) -> bool, shift: Option<&[usize]>) -> Vec<(Vec<usize>, Element)> {
        // At least single generators (an order of 0 is 1, as the builder's `order` takes it): the
        // enumeration stops at `order`, so 0 would enumerate every connected product.
        let order = options.order.max(1);
        let order = if gens.len() > options.max_cluster { order.min(2) } else { order };
        let n = gens.len();
        let mut at: HashMap<usize, Vec<usize>> = HashMap::new();
        for (i, &g) in gens.iter().enumerate() {
            for &l in &generators[g].members {
                at.entry(l).or_default().push(i);
            }
        }
        let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
        for list in at.values() {
            for &a in list {
                for &b in list {
                    if a != b {
                        adj[a].push(b);
                    }
                }
            }
        }
        for a in adj.iter_mut() {
            a.sort_unstable();
            a.dedup();
        }
        let mut out = Vec::new();
        let mut seen: std::collections::HashSet<u64> = std::collections::HashSet::new();
        let mut visit = |subset: &[usize], seen: &mut std::collections::HashSet<u64>, out: &mut Vec<(Vec<usize>, Element)>| {
            let ids: Vec<usize> = subset.iter().map(|&i| gens[i]).collect();
            if !keep(&ids) {
                return;
            }
            let mut members: Vec<usize> = shift.map(|s| s.to_vec()).unwrap_or_default();
            for &i in subset {
                members = symmetric_difference(&members, &generators[gens[i]].members);
            }
            if members.is_empty() {
                return;
            }
            let key = members.iter().fold(0u64, |h, &l| h ^ self.zobrist[l]);
            if !seen.insert(key) {
                return;
            }
            let e = if subset.len() == 1 && shift.is_none() { Some(generators[gens[subset[0]]].clone()) } else { cache.entry(key).or_insert_with(|| element_of(&members)).clone() };
            if let Some(e) = e {
                out.push((ids, e));
            }
        };
        for v in 0..n {
            let ext: Vec<usize> = adj[v].iter().copied().filter(|&u| u > v).collect();
            let mut stack: Vec<(Vec<usize>, Vec<usize>)> = vec![(vec![v], ext)];
            while let Some((sub, mut ext)) = stack.pop() {
                visit(&sub, &mut seen, &mut out);
                if sub.len() == order {
                    continue;
                }
                while let Some(w) = ext.pop() {
                    let mut next = ext.clone();
                    for &u in &adj[w] {
                        if u > v && !sub.contains(&u) && !next.contains(&u) && !sub.iter().any(|&s| adj[s].binary_search(&u).is_ok()) {
                            next.push(u);
                        }
                    }
                    let mut sub2 = sub.clone();
                    sub2.push(w);
                    stack.push((sub2, next));
                }
            }
        }
        out
    }
}

impl Kernel {
    /// A location's random word; a set's key is the XOR of its locations'.
    pub(crate) fn zobrist(&self, l: usize) -> u64 {
        self.zobrist[l]
    }
}

/// Every nonzero product of the generators (2^n − 1 of them), each once.
fn whole_span(p: &Program, solver: &GaugeSolver, generators: &[Element]) -> Vec<Element> {
    let n = generators.len();
    let mut out = Vec::with_capacity((1usize << n) - 1);
    for mask in 1u32..(1u32 << n) {
        if mask.count_ones() == 1 {
            out.push(generators[mask.trailing_zeros() as usize].clone());
            continue;
        }
        let mut members: Vec<usize> = Vec::new();
        for (g, e) in generators.iter().enumerate() {
            if mask >> g & 1 == 1 {
                members = symmetric_difference(&members, &e.members);
            }
        }
        out.extend(p.element(&members, solver));
    }
    out
}

/// The null space of `items` (each a residue and the sorted locations it stands for): every
/// combination of items whose residues sum to zero, as the symmetric difference of their
/// locations, a basis of them. Forward elimination in the items' order (rows in echelon form,
/// each zero at the pivots before it, so one pass reduces).
pub(crate) fn null_space(items: Vec<(Key, Vec<usize>)>) -> Vec<Vec<usize>> {
    let mut rows: Vec<(usize, Key, Vec<usize>)> = Vec::new();
    let mut out = Vec::new();
    for (mut v, mut combo) in items {
        for (pivot, rv, rc) in &rows {
            if v[pivot / 64] >> (pivot % 64) & 1 == 1 {
                v.iter_mut().zip(rv).for_each(|(a, b)| *a ^= b);
                combo = symmetric_difference(&combo, rc);
            }
        }
        match v.iter().position(|&w| w != 0) {
            Some(w) => rows.push((w * 64 + v[w].trailing_zeros() as usize, v, combo)),
            None if !combo.is_empty() => out.push(combo),
            None => {}
        }
    }
    out
}

/// Kernel vectors (sorted locations among `n`) made light: each, greedily, plus whichever of the
/// generators `gens` (or of the vectors before it, already made light) lowers its cost most, the
/// sum of `cost` over its locations, until none does. Adding a generator keeps a vector's class
/// modulo their span, and adding an earlier vector keeps the set a basis of the same quotient.
pub(crate) fn shorten(mut found: Vec<Vec<usize>>, gens: &[Vec<usize>], n: usize, cost: &dyn Fn(usize) -> u64) -> Vec<Vec<usize>> {
    let mut by_location: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (i, g) in gens.iter().enumerate() {
        for &l in g {
            by_location[l].push(i);
        }
    }
    let weight = |v: &[usize]| v.iter().map(|&l| cost(l)).sum::<u64>();
    for i in 0..found.len() {
        let mut v = std::mem::take(&mut found[i]);
        let mut held = vec![false; n];
        for &l in &v {
            held[l] = true;
        }
        loop {
            let mut best: (i64, Option<usize>) = (0, None);
            let mut seen: std::collections::HashSet<usize> = std::collections::HashSet::new();
            for &l in &v {
                for &g in &by_location[l] {
                    if seen.insert(g) {
                        let gain: i64 = gens[g].iter().map(|&m| if held[m] { cost(m) as i64 } else { -(cost(m) as i64) }).sum();
                        if gain > best.0 {
                            best = (gain, Some(g));
                        }
                    }
                }
            }
            let next = match best.1 {
                Some(g) => symmetric_difference(&v, &gens[g]),
                None => match found[..i].iter().map(|f| symmetric_difference(&v, f)).min_by_key(|w| weight(w)) {
                    Some(w) if weight(&w) < weight(&v) => w,
                    _ => break,
                },
            };
            for &l in &v {
                held[l] = false;
            }
            for &l in &next {
                held[l] = true;
            }
            v = next;
        }
        found[i] = v;
    }
    found
}

pub(crate) fn symmetric_difference(a: &[usize], b: &[usize]) -> Vec<usize> {
    let (mut i, mut j, mut out) = (0, 0, Vec::with_capacity(a.len() + b.len()));
    while i < a.len() || j < b.len() {
        if j == b.len() || (i < a.len() && a[i] < b[j]) {
            out.push(a[i]);
            i += 1;
        } else if i == a.len() || b[j] < a[i] {
            out.push(b[j]);
            j += 1;
        } else {
            i += 1;
            j += 1;
        }
    }
    out
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::circuit::Circuit;

    #[test]
    fn long_group_weights_and_factor_ratios_match_both_parities() {
        let cases = [
            vec![1.6_f64; 110], vec![1.6; 150], vec![-1.6; 150],
            [1.6, -0.2, 0.03].repeat(50), [1.6, -1.6].repeat(75), vec![0.001; 150],
        ];
        for angles in cases {
            let text = format!("R 0\n{}M 0\nDETECTOR rec[-1]", angles.iter().map(|a| format!("I_ERROR[R_X(theta={a})] 0\n")).collect::<String>());
            let mut p = Program::new(&Circuit::parse(&text).unwrap()).unwrap();
            let solver = GaugeSolver::new(&p);
            let kernel = Kernel::new(&p, &solver, CoherentOptions::default());
            assert!(kernel.generators.is_empty() && kernel.groups.len() == 1);
            let total_angle = angles.iter().sum::<f64>();
            let physical = [total_angle.cos().powi(2), total_angle.sin().powi(2)];
            let twirl_odd = 0.5 * (1.0 - angles.iter().map(|a| (2.0 * a).cos()).product::<f64>());
            for merged in [false, true] {
                if merged {
                    p.set_merge(&kernel.merge_groups());
                }
                let q_odd = if merged { super::super::COHERENT_SHARE * physical[1] + (1.0 - super::super::COHERENT_SHARE) * twirl_odd } else { twirl_odd };
                let q = [1.0 - q_odd, q_odd];
                let mut normalization = 0.0;
                for parity in 0..2 {
                    let mut shot = Shot { fired: vec![false; angles.len()], anti: vec![false; angles.len()], branch: p.reference.clone(), ..Shot::default() };
                    // Exercise distributed twirl faults as well as the canonical merged draw.
                    if !merged {
                        for i in (0..angles.len() - 1).step_by(3) {
                            shot.fired[i] = true;
                        }
                    }
                    let prior_parity = shot.fired.iter().filter(|&&b| b).count() % 2;
                    shot.fired[angles.len() - 1] = prior_parity != parity;
                    let (weight, eff) = kernel.groups_of(&p, &shot);
                    let expected = physical[parity] / q[parity];
                    assert!(weight.is_finite() && weight >= 0.0);
                    assert!((weight - expected).abs() < 2e-11 * expected.max(1.0), "{} {merged} {parity}: {weight} != {expected}", angles.len());
                    normalization += q[parity] * weight;
                    let (toggle, size) = eff[kernel.groups[0].members[0]].unwrap();
                    assert!((size - q[1 - parity] / q[parity]).abs() < 2e-11 * size.max(1.0));
                    if physical.iter().all(|v| *v > 1e-20) {
                        let expected_toggle = if parity == 0 { total_angle.tan() } else { -1.0 / total_angle.tan() };
                        assert!(toggle.re.abs() < 1e-12);
                        assert!((toggle.im - expected_toggle).abs() < 2e-11 * expected_toggle.abs().max(1.0));
                    }
                }
                assert!((normalization - 1.0).abs() < 2e-11, "{} {merged}: {normalization}", angles.len());
            }
        }
    }

    /// The kernel's exact distribution against the state vector's: the bias, with no sampling
    /// noise.
    pub(crate) fn bias(text: &str, options: CoherentOptions) -> f64 {
        let c = Circuit::parse(text).unwrap();
        let p = Program::new(&c).unwrap();
        let solver = GaugeSolver::new(&p);
        let kernel = Kernel::new(&p, &solver, options);
        let mut cache = HashMap::new();
        let ours = p.exact_distribution_by(&mut |shot| kernel.weight(&p, shot, &solver, &mut cache), true, 1 << 22).unwrap();
        let total: f64 = ours.values().sum();
        let sv = crate::statevec::Program::new(&c).unwrap();
        let reference = sv.reference_parities().to_vec();
        let nd = sv.num_detectors();
        let mut theirs = std::collections::BTreeMap::new();
        for ((dets, obs), q) in sv.distribution(1 << 22).unwrap() {
            let d: Vec<bool> = dets.iter().zip(&reference).map(|(a, b)| a ^ b).collect();
            let o = (0..sv.num_observables()).fold(0u64, |acc, k| acc | (((obs >> k & 1 == 1) ^ reference[nd + k]) as u64) << k);
            *theirs.entry((d, o)).or_insert(0.0) += q;
        }
        let keys: BTreeSet<_> = ours.keys().chain(theirs.keys()).cloned().collect();
        keys.iter().map(|k| (ours.get(k).copied().unwrap_or(0.0) / total - theirs.get(k).copied().unwrap_or(0.0)).abs()).fold(0.0, f64::max)
    }

    /// The sampler as built (`CoherentSampler::new`: Clifford-angle gates expanded, groups
    /// merged, the local kernel's weight), its detection events summed exactly over every
    /// branch, against the state vector's events: the largest difference in any outcome's
    /// probability. Events, not raw parities: a detector on a random outcome is a flip from
    /// the reference run, which must be the state vector's.
    fn events_bias(text: &str) -> f64 {
        let c = Circuit::parse(text).unwrap();
        let s = super::super::sampler::CoherentSampler::new(&c, CoherentOptions::default()).unwrap();
        let solver = GaugeSolver::new(&s.program);
        let mut cache = HashMap::new();
        let ours = s.program.exact_distribution_by(&mut |shot| s.kernel.weight(&s.program, shot, &solver, &mut cache), false, 1 << 22).unwrap();
        let total: f64 = ours.values().sum();
        let theirs = crate::statevec::Program::new(&c).unwrap().distribution(1 << 22).unwrap();
        let keys: BTreeSet<_> = ours.keys().chain(theirs.keys()).cloned().collect();
        keys.iter().map(|k| (ours.get(k).copied().unwrap_or(0.0) / total - theirs.get(k).copied().unwrap_or(0.0)).abs()).fold(0.0, f64::max)
    }

    /// The adversarial review's circuits: a detector on a random outcome after random outcomes
    /// on other qubits, which the 1.9 sampler reported complemented (its reference run was a
    /// seeded one, not the converter's). Through the sampler's own construction.
    #[test]
    fn detectors_on_random_outcomes_are_the_state_vectors() {
        for text in [
            "R 0 1 2\nI_ERROR[R_Y(theta=0.24)] 0\nMX 1 2\nMX 0\nDETECTOR rec[-1]",
            "R 0 1 2\nI_ERROR[R_Y(theta=0.24)] 0\nMX 1\nMX 0\nDETECTOR rec[-1]",
            "R 0 1 2 3 4\nI_ERROR[R_Y(theta=0.24)] 0\nMX 1 2 3 4\nMX 0\nDETECTOR rec[-1]",
            "R 0 1 2 3\nI_ERROR[R_X(theta=0.15)] 0 2\nCX 0 1 2 1\nM 1\nI_ERROR[R_Y(theta=0.2)] 3\nMX 0 2\nMX 3\nDETECTOR rec[-1]",
            "RX 0 1\nI_ERROR[R_Z(theta=0.3)] 0\nMX 1\nMY 0\nDETECTOR rec[-1]",
            "RX 0 1 2 3\nI_ERROR[R_Z(theta=0.3)] 0\nMX 1 2 3\nMY 0\nDETECTOR rec[-1]",
            "R 0 1 2\nRX 1 2\nI_ERROR[R_Y(theta=0.24)] 0\nMX 0\nDETECTOR rec[-1]",
            "R 0 1 2\nI_ERROR[R_Y(theta=0.24)] 0\nMRX 1 2\nMX 0\nDETECTOR rec[-1]\nDETECTOR rec[-2] rec[-3]\nOBSERVABLE_INCLUDE(0) rec[-1] rec[-2]",
        ] {
            let b = events_bias(text);
            assert!(b < 1e-10, "{text}\nbias {b:.3e}");
        }
    }

    #[test]
    fn local_generators_are_found() {
        // Z before and after a CZ on its control is one fault in two places.
        let p = Program::new(&Circuit::parse("RX 0\nR 1\nI_ERROR[R_Z(theta=0.1)] 0\nCZ 0 1\nI_ERROR[R_Z(theta=0.1)] 0\nMX 0\nM 1").unwrap()).unwrap();
        let k = Kernel::new(&p, &GaugeSolver::new(&p), CoherentOptions::default());
        assert!(k.groups.iter().any(|g| g.members == vec![0, 1]), "{:?}", k.groups);
    }

    #[test]
    fn small_circuits_are_exact_or_nearly() {
        let rep = "R 0 1 2\nI_ERROR[R_X(theta=0.2)] 0 2\nCX 0 1\nCX 2 1\nMR 1\nI_ERROR[R_X(theta=0.15)] 0 2\nCX 0 1\nCX 2 1\nMR 1\n\
                   DETECTOR rec[-2]\nDETECTOR rec[-1] rec[-2]\nM 0 2\nDETECTOR rec[-1] rec[-2] rec[-3]\nOBSERVABLE_INCLUDE(0) rec[-1]";
        let b = bias(rep, CoherentOptions::default());
        assert!(b < 1e-3, "{b}");
        let two = "RX 0\nR 1\nI_ERROR[R_Z(theta=0.3)] 0\nCZ 0 1\nI_ERROR[R_Z(theta=0.2)] 0\nMX 0\nM 1\nDETECTOR rec[-2]";
        assert!(bias(two, CoherentOptions::default()) < 1e-10);
    }
}

#[cfg(test)]
mod survey {
    use super::*;
    use crate::circuit::Circuit;

    /// The local kernel with merged groups, exhaustively, over random small circuits: as exact
    /// as the whole kernel.
    #[test]
    fn exact_over_random_circuits() {
        let mut rng = crate::surface_code::Xorshift::new(2026);
        let ops1 = ["H", "S", "X", "SQRT_X", "I_ERROR[R_X(theta=0.3)]", "I_ERROR[R_Y(theta=-0.25)]", "I_ERROR[R_Z(theta=0.4)]", "X_ERROR(0.1)", "DEPOLARIZE1(0.1)", "R", "RX", "M", "MX", "MR"];
        let ops2 = ["CX", "CZ", "SWAP", "II_ERROR[R_ZZ(theta=0.35)]"];
        let mut worst: f64 = 0.0;
        for _ in 0..80 {
            let n = 1 + (rng.next_u64() % 3) as usize;
            let mut lines = Vec::new();
            let (mut rot, mut rand, mut noise, mut measured) = (0, 0, 0, 0usize);
            while lines.len() < 12 {
                let two = n > 1 && rng.next_u64().is_multiple_of(3);
                let op = if two { ops2[(rng.next_u64() % ops2.len() as u64) as usize] } else { ops1[(rng.next_u64() % ops1.len() as u64) as usize] };
                if op.contains("theta") { if rot == 4 { continue; } rot += 1; }
                if op.contains("ERROR(") || op.contains("DEPOL") { if noise == 2 { continue; } noise += 1; }
                if op.starts_with('M') || (op.starts_with('R') && !op.starts_with("R_")) { if rand == 4 { continue; } rand += 1; measured += op.starts_with('M') as usize; }
                let a = (rng.next_u64() % n as u64) as usize;
                if two { let b = (a + 1 + (rng.next_u64() % (n as u64 - 1)) as usize) % n; lines.push(format!("{op} {a} {b}")); } else { lines.push(format!("{op} {a}")); }
            }
            lines.push(format!("M {}", (0..n).map(|q| q.to_string()).collect::<Vec<_>>().join(" ")));
            let total = measured + n;
            for _ in 0..2 { lines.push(format!("DETECTOR rec[-{}]", 1 + rng.next_u64() % total as u64)); }
            lines.push(format!("OBSERVABLE_INCLUDE(0) rec[-{}]", 1 + rng.next_u64() % total as u64));
            let text = lines.join("\n");
            let c = Circuit::parse(&text).unwrap();
            let p = Program::new(&c).unwrap();
            let solver = GaugeSolver::new(&p);
            let kernel = Kernel::new(&p, &solver, CoherentOptions::default());
            let mut cache = HashMap::new();
            let mut pm = p.clone();
            pm.set_merge(&kernel.merge_groups());
            let ours = pm.exact_distribution_by(&mut |shot| kernel.weight(&pm, shot, &solver, &mut cache), true, 1 << 22).unwrap();
            let full = p.exact_distribution(&p.full_kernel().unwrap(), true, 1 << 22).unwrap();
            let total_w: f64 = ours.values().sum();
            let b = full.iter().map(|(k, v)| (ours.get(k).copied().unwrap_or(0.0) / total_w - v).abs()).fold(0.0, f64::max);
            if b > worst {
                worst = b;
                println!("bias {b:.2e} (kernel {} generators, full {})\n{text}\n", kernel.generators.len(), p.full_kernel().unwrap().len());
            }
        }
        assert!(worst < 1e-9, "worst {worst:.2e}");
    }
}

#[cfg(test)]
pub(crate) mod nonlocal {
    use super::*;
    use crate::circuit::Circuit;

    /// A repetition memory of distance d: an over-rotation exp(−iθX) on every data qubit at
    /// each round's start, nothing else (the 2.1 audit's circuit at d = 3, 2 rounds).
    pub(crate) fn repetition(d: usize, rounds: usize, theta: f64) -> String {
        let data: Vec<String> = (0..d).map(|i| (2 * i).to_string()).collect();
        let ancillas: Vec<String> = (0..d - 1).map(|i| (2 * i + 1).to_string()).collect();
        let pairs = |off: usize| (0..d - 1).map(|i| format!("{} {}", 2 * i + off, 2 * i + 1)).collect::<Vec<_>>().join(" ");
        let mut s = format!("R {}\n", (0..2 * d - 1).map(|q| q.to_string()).collect::<Vec<_>>().join(" "));
        for r in 0..rounds {
            s += &format!("I_ERROR[R_X(theta={theta})] {}\nCX {}\nCX {}\nMR {}\n", data.join(" "), pairs(0), pairs(2), ancillas.join(" "));
            for k in 0..d - 1 {
                let a = d - 1 - k;
                s += &if r == 0 { format!("DETECTOR rec[-{a}]\n") } else { format!("DETECTOR rec[-{a}] rec[-{}]\n", a + d - 1) };
            }
        }
        s += &format!("M {}\n", data.join(" "));
        for k in 0..d - 1 {
            s += &format!("DETECTOR rec[-{}] rec[-{}] rec[-{}]\n", d - k, d - k - 1, 2 * d - 1 - k);
        }
        s + "OBSERVABLE_INCLUDE(0) rec[-1]\n"
    }

    /// Stim's rotated surface-code X memory with an over-rotation exp(−iθZ) on every data qubit
    /// at each round's start (before the round's first Hadamard on the X ancillas).
    pub(crate) fn surface(d: u32, rounds: u64, theta: f64) -> String {
        let text = crate::generated::generate("surface_code:rotated_memory_x", d, rounds, &crate::generated::Noise::default()).unwrap();
        let lines: Vec<&str> = text.lines().collect();
        let data = lines.iter().rev().find(|l| l.trim_start().starts_with("MX ")).unwrap().trim_start()[3..].to_string();
        let mut out = Vec::new();
        for (i, line) in lines.iter().enumerate() {
            if line.trim_start().starts_with("H ") && lines.get(i + 1).is_some_and(|l| l.trim() == "TICK") && lines.get(i + 2).is_some_and(|l| l.trim_start().starts_with("CX ")) {
                out.push(format!("I_ERROR[R_Z(theta={theta})] {data}"));
            }
            out.push(line.to_string());
        }
        out.join("\n")
    }

    /// The largest difference (relative past 1), over `shots` shots, between the kernel's weight
    /// and the whole kernel's class sum (`Program::weight`, which `exact` proves).
    fn against_whole(k: &Kernel, p: &Program, solver: &GaugeSolver, shots: u64) -> f64 {
        against_whole_if(k, p, solver, shots, &|_| true)
    }

    /// The same over the shots `take` takes.
    fn against_whole_if(k: &Kernel, p: &Program, solver: &GaugeSolver, shots: u64, take: &dyn Fn(&Shot) -> bool) -> f64 {
        let full = p.full_kernel().unwrap();
        let mut cache = HashMap::new();
        let mut worst = 0.0f64;
        for seed in 0..shots {
            let mut rng = crate::surface_code::Xorshift::new(seed);
            let shot = p.run_shot(&mut super::super::shot::Stream(&mut rng));
            if !take(&shot) {
                continue;
            }
            let (a, b) = (k.weight(p, &shot, solver, &mut cache), p.weight(&shot, &full));
            worst = worst.max((a - b).abs() / b.max(1.0));
        }
        worst
    }

    fn kernel(text: &str, options: CoherentOptions) -> (Program, GaugeSolver, Kernel) {
        let p = Program::new(&Circuit::parse(text).unwrap()).unwrap();
        let solver = GaugeSolver::new(&p);
        let k = Kernel::new(&p, &solver, options);
        (p, solver, k)
    }

    /// An order of 0 is 1, as the builder's `order` takes it: single generators, not every
    /// connected product of the cluster (exponential in its size, a hang on a large one).
    #[test]
    fn order_zero_is_order_one() {
        let (p, solver, k) = kernel(&surface(3, 3, 0.2), CoherentOptions::default());
        let gens: Vec<usize> = (0..k.generators.len() - k.num_nonlocal).collect();
        assert!(gens.len() > 3);
        let ids = |order: usize| -> Vec<Vec<usize>> {
            let mut cache = HashMap::new();
            let mut element_of = |members: &[usize]| p.element(members, &solver);
            let options = CoherentOptions { order, ..CoherentOptions::default() };
            k.products(&k.generators, &gens, &options, &mut cache, &mut element_of, &|_| true, None).into_iter().map(|(ids, _)| ids).collect()
        };
        let (zero, one, two) = (ids(0), ids(1), ids(2));
        assert_eq!(zero, one);
        assert!(one.iter().all(|g| g.len() == 1) && two.len() > one.len());
    }

    /// XXX in round 1 with XXX in round 2: no syndrome, no logical effect, weight six, and the
    /// whole kernel; the local search finds nothing.
    #[test]
    fn the_cross_round_logical_completes_the_kernel() {
        let (_, _, k) = kernel(&repetition(3, 2, 0.3), CoherentOptions::default());
        assert_eq!((k.generators.len(), k.num_nonlocal, k.complete, k.is_whole()), (1, 1, true, true));
        assert_eq!(k.generators[0].members, vec![0, 1, 2, 3, 4, 5]);
        let (_, _, local) = kernel(&repetition(3, 2, 0.3), CoherentOptions { max_complete: 0, ..CoherentOptions::default() });
        assert_eq!((local.generators.len(), local.complete, local.is_whole()), (0, false, false));
    }

    /// The completion's directions are the shortest there are: a logical in one round with the
    /// same logical in another, 2d locations (the surface code's Z_L is a line of three; its
    /// stabilisers, which the local search finds, are shorter still).
    #[test]
    fn nonlocal_generators_are_short() {
        for (text, rounds, weight) in [(repetition(5, 3, 0.2), 3, 10), (repetition(3, 4, 0.2), 4, 6)] {
            let (_, _, k) = kernel(&text, CoherentOptions::default());
            assert!(k.complete);
            assert_eq!(k.num_nonlocal, rounds - 1, "{text}");
            for e in &k.generators[k.generators.len() - k.num_nonlocal..] {
                assert_eq!(e.members.len(), weight, "{:?}", e.members);
            }
        }
        let (p, solver, k) = kernel(&surface(3, 3, 0.2), CoherentOptions::default());
        assert!(k.complete && k.num_nonlocal >= 2);
        assert_eq!(k.generators.len(), k.dimension(&p, &solver), "a basis of the whole kernel");
        for e in &k.generators[k.generators.len() - k.num_nonlocal..] {
            assert!(e.members.len() <= 6, "{:?}", e.members);
        }
    }

    /// With the kernel complete and small, a shot's weight is the whole kernel's class sum:
    /// exact, per shot, with no sampling noise. At θ = 0.6 a logical error fires often.
    #[test]
    fn small_complete_kernels_are_the_whole_kernels_sums() {
        for text in [repetition(3, 2, 0.6), repetition(3, 3, 0.6), repetition(5, 2, 0.6), surface(3, 2, 0.6)] {
            let (p, solver, k) = kernel(&text, CoherentOptions::default());
            assert!(k.is_whole(), "{text}");
            let worst = against_whole(&k, &p, &solver, 3000);
            assert!(worst < 1e-9, "{text}\nworst {worst:.3e}");
        }
    }

    /// Truncated (neither contracted nor listed), a shot whose faults reach a non-local
    /// generator sums its elements in its frame: on the audit's circuit (one generator) that is
    /// the whole kernel, and with three rounds (a logical in round 1, 2 or 3 with the same
    /// logical in the others) it is the whole kernel for every shot with a fault. With none, the
    /// untouched generators' product leaves out that the two overlap: a term of order tan⁶ θ.
    #[test]
    fn truncated_sums_take_the_nonlocal_generators() {
        let (p, solver, mut k) = kernel(&repetition(3, 2, 0.6), CoherentOptions::default());
        (k.plan, k.span) = (None, None);
        let worst = against_whole(&k, &p, &solver, 3000);
        assert!(worst < 1e-9, "worst {worst:.3e}");
        let (p, solver, mut k) = kernel(&repetition(3, 3, 0.6), CoherentOptions::default());
        (k.plan, k.span) = (None, None);
        let worst = against_whole_if(&k, &p, &solver, 3000, &|shot| shot.fired.iter().any(|&f| f));
        assert!(worst < 1e-9, "worst {worst:.3e}");
        let (p, solver, mut k) = kernel(&repetition(3, 3, 0.2), CoherentOptions::default());
        (k.plan, k.span) = (None, None);
        let worst = against_whole(&k, &p, &solver, 3000);
        assert!(worst < 1e-3, "worst {worst:.3e}");
    }
}

#[cfg(test)]
mod merged {
    use super::*;
    use crate::circuit::Circuit;

    fn compare(text: &str) {
        let c = Circuit::parse(text).unwrap();
        let p = Program::new(&c).unwrap();
        let solver = GaugeSolver::new(&p);
        let kernel = Kernel::new(&p, &solver, CoherentOptions::default());
        let mut pm = p.clone();
        pm.set_merge(&kernel.merge_groups());
        let mut cache = HashMap::new();
        let ours = pm.exact_distribution_by(&mut |shot| kernel.weight(&pm, shot, &solver, &mut cache), true, 1 << 22).unwrap();
        let full = p.exact_distribution(&p.full_kernel().unwrap(), true, 1 << 22).unwrap();
        assert!(!kernel.groups.is_empty());
        for (k, v) in &full {
            assert!((ours.get(k).copied().unwrap_or(0.0) - v).abs() < 1e-12, "{text}\n{k:?}");
        }
    }

    #[test]
    fn merged_groups_are_exact() {
        compare("RX 0\nR 1\nI_ERROR[R_Z(theta=0.3)] 0\nCZ 0 1\nI_ERROR[R_Z(theta=0.2)] 0\nMX 0\nM 1\nDETECTOR rec[-2]");
        compare("RX 0\nI_ERROR[R_Z(theta=0.3)] 0\nZ_ERROR(0.2) 0\nI_ERROR[R_Z(theta=0.2)] 0\nMX 0\nDETECTOR rec[-1]");
    }
}

#[cfg(test)]
mod diagnose {
    use super::*;
    use crate::circuit::Circuit;

    /// Per-shot weights of the local kernel against the whole kernel's, on a circuit from
    /// `COH_CIRCUIT` (run with `--ignored --nocapture`).
    #[test]
    #[ignore]
    fn local_against_full() {
        let c = Circuit::parse(&std::fs::read_to_string(std::env::var("COH_CIRCUIT").unwrap()).unwrap()).unwrap();
        let p = Program::new(&c).unwrap();
        let solver = GaugeSolver::new(&p);
        let order = std::env::var("COH_ORDER").map_or(3, |v| v.parse().unwrap());
        let k = Kernel::new(&p, &solver, CoherentOptions { order, ..CoherentOptions::default() });
        println!("kernel dimension {} local span {}", k.dimension(&p, &solver), k.local_span());
        if p.locations.len() > 20 {
            return;
        }
        let full = p.full_kernel().unwrap();
        println!("locations {} full kernel {} generators {} groups {:?}", p.locations.len(), full.len(), k.generators.len(), k.groups.iter().map(|g| g.members.clone()).collect::<Vec<_>>());
        for g in &k.generators {
            println!("  gen {:?}", g.members);
        }
        // Full kernel dimension and whether the local generators and groups span it.
        let nl = p.locations.len();
        let mut basis: Vec<u64> = Vec::new();
        let add = |v: u64, basis: &mut Vec<u64>| {
            let mut v = v;
            for &b in basis.iter() {
                v = v.min(v ^ b);
            }
            if v != 0 {
                basis.push(v);
                basis.sort_unstable_by(|a, b| b.cmp(a));
                true
            } else {
                false
            }
        };
        for g in &k.generators {
            add(g.members.iter().fold(0u64, |a, &l| a | 1 << l), &mut basis);
        }
        for g in &k.groups {
            for &l in &g.members[1..] {
                add((1u64 << g.members[0]) | (1 << l), &mut basis);
            }
        }
        let local_dim = basis.len();
        let mut missing = Vec::new();
        for e in &full {
            let v = e.members.iter().fold(0u64, |a, &l| a | 1 << l);
            if add(v, &mut basis) {
                missing.push(e.members.clone());
            }
        }
        println!("local span dim {local_dim}, full dim {}, missing {:?}", basis.len(), missing);
        let _ = nl;
        let mut cache = HashMap::new();
        let (mut worst, mut sum_l, mut sum_f) = (0.0f64, 0.0, 0.0);
        for seed in 0..20000u64 {
            let mut rng = crate::surface_code::Xorshift::new(seed);
            let shot = p.run_shot(&mut super::super::shot::Stream(&mut rng));
            let wl = k.weight(&p, &shot, &solver, &mut cache);
            let wf = p.weight(&shot, &full);
            sum_l += wl;
            sum_f += wf;
            if (wl - wf).abs() > worst {
                worst = (wl - wf).abs();
                let fired: Vec<usize> = (0..nl).filter(|&l| shot.fired[l]).collect();
                println!("seed {seed}: local {wl:.4} full {wf:.4} fired {fired:?}");
            }
        }
        println!("mean local {:.4} full {:.4}", sum_l / 20000.0, sum_f / 20000.0);
    }
}

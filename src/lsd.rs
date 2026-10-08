//! BP+LSD: belief propagation, and where it does not converge, localized statistics decoding
//! (Hillmann, Berent, Quintavalle, Eisert, Wille and Roffe, arXiv:2406.18655).
//!
//! WHY THIS EXISTS
//! ---------------
//! OSD (`osd`) solves the whole syndrome equation at once: one elimination of the full check
//! matrix per shot, cubic in its size, which is what makes BP+OSD slow on large codes. LSD
//! solves it in pieces. Each fired check starts a cluster; clusters grow one fault at a time,
//! most likely first by BP's posteriors, and merge when they touch; a cluster stops growing
//! once its syndrome is in the image of its own columns, which an elimination kept up to date
//! column by column (`plu`) answers after every step. Each finished cluster is solved alone,
//! and the pieces together explain the syndrome. Clusters stay small at useful error rates, so
//! the eliminations are small too.
//!
//! This is `ldpc`'s `LsdDecoder` (src_cpp/lsd.hpp), ported so that its corrections are
//! `ldpc`'s: the same growth and merge order, the same per-cluster elimination, and its
//! clusters' sets iterated in `tsl::robin_set`'s order (`robin`). Three of its choices cannot
//! be reproduced, because they depend on the order of equal keys under `std::sort` or on
//! pointer values, and are made here as follows:
//!
//! - candidates of equal weight are taken in set order, and invalid clusters of equal size
//!   grow in creation order: a stable sort. `std::sort` is stable too on short inputs (it
//!   insertion-sorts up to 16 elements in libstdc++ and 30 in libc++), so only a sort of more
//!   than 16 with equal keys where they matter can differ;
//! - clusters to merge into one are merged in the order they were met. robin-map orders them by
//!   the hash of their address, which libc++ scrambles (libstdc++ does not), so when a step
//!   merges three or more clusters into one, the order ldpc merges them in is not
//!   reproducible.
//!
//! A decode that met one of these where the order could differ counts it in
//! `BpLsdWork::ties`, so a check against `ldpc` can tell a tie from a bug.
//!
//! Higher orders (LSD-E, LSD-CS): each cluster grows on until it has `order` free columns, and
//! ordered-statistics search runs inside it on BP's posteriors (`ldpc`'s `apply_lsdw` and
//! `DenseOsdDecoder`).

use crate::bp::{Bp, BpWork, Method};
use crate::osd::OsdMethod;
use crate::plu::Plu;
use crate::robin::RobinSet;
use std::collections::HashMap;

const NONE: u32 = u32::MAX;

/// One cluster: `ldpc`'s `LsdCluster`.
struct Cluster {
    id: u32,
    active: bool,
    valid: bool,
    /// Whether a choice ldpc might make otherwise went into this cluster.
    tied: bool,
    bits: RobinSet,
    checks: RobinSet,
    boundary: RobinSet,
    candidates: RobinSet,
    enclosed: RobinSet,
    /// Clusters met this step, in the order they were met.
    merge_list: Vec<u32>,
    /// The cluster's columns, over its local check indices, in the order its bits joined.
    pcm: Vec<Vec<u32>>,
    syndrome: Vec<u8>,
    local_to_check: Vec<u32>,
    check_to_local: HashMap<u32, u32>,
    local_to_bit: Vec<u32>,
    plu: Plu,
}

impl Cluster {
    fn new(id: u32) -> Cluster {
        let mut c = Cluster {
            id,
            active: true,
            valid: false,
            tied: false,
            bits: RobinSet::new(),
            checks: RobinSet::new(),
            boundary: RobinSet::new(),
            candidates: RobinSet::new(),
            enclosed: RobinSet::new(),
            merge_list: Vec::new(),
            pcm: Vec::new(),
            syndrome: Vec::new(),
            local_to_check: vec![id],
            check_to_local: HashMap::from([(id, 0)]),
            local_to_bit: Vec::new(),
            plu: Plu::new(1, 0, Vec::new()),
        };
        c.boundary.insert(id);
        c.enclosed.insert(id);
        c.checks.insert(id);
        c
    }
}

/// The check matrix both ways and the per-shot state shared by every cluster.
struct Lsd<'a> {
    rows: &'a [Vec<u32>],
    columns: &'a [Vec<u32>],
    clusters: Vec<Cluster>,
    bit_owner: Vec<u32>,
    check_owner: Vec<u32>,
    ties: u32,
}

impl Lsd<'_> {
    fn note_merge(&mut self, me: usize, other: u32) {
        let c = &mut self.clusters[me];
        if !c.merge_list.contains(&other) {
            c.merge_list.push(other);
        }
    }

    /// `add_check`: the check's local index, adding it (and to the boundary when asked).
    fn add_check(&mut self, me: usize, check: u32, boundary: bool) -> u32 {
        let c = &mut self.clusters[me];
        if boundary {
            c.boundary.insert(check);
        }
        self.check_owner[check as usize] = me as u32;
        if !c.checks.insert(check) {
            return c.check_to_local[&check];
        }
        c.local_to_check.push(check);
        let local = c.local_to_check.len() as u32 - 1;
        c.check_to_local.insert(check, local);
        local
    }

    /// `add_column_to_cluster_pcm`.
    fn add_column(&mut self, me: usize, bit: u32) {
        let mut col = Vec::with_capacity(self.columns[bit as usize].len());
        for &check in &self.columns[bit as usize] {
            let owner = self.check_owner[check as usize];
            if owner == me as u32 {
                col.push(self.clusters[me].check_to_local[&check]);
                continue;
            }
            if owner != NONE {
                self.note_merge(me, owner);
            }
            col.push(self.add_check(me, check, true));
        }
        self.clusters[me].pcm.push(col);
    }

    /// `add_bit_node_to_cluster`: whether the bit was this cluster's to consider.
    fn add_bit_node(&mut self, me: usize, bit: u32, in_merge: bool) -> bool {
        let owner = self.bit_owner[bit as usize];
        if owner == me as u32 {
            return false;
        }
        if owner == NONE || in_merge {
            let c = &mut self.clusters[me];
            if c.bits.insert(bit) {
                c.local_to_bit.push(bit);
                self.bit_owner[bit as usize] = me as u32;
            }
            self.add_column(me, bit);
        } else {
            self.note_merge(me, owner);
        }
        true
    }

    /// `compute_growth_candidate_bit_nodes`.
    fn candidates(&mut self, me: usize) {
        let c = &mut self.clusters[me];
        c.candidates.clear();
        let mut spent = Vec::new();
        for check in c.boundary.iter() {
            let mut erase = true;
            for &bit in &self.rows[check as usize] {
                if self.bit_owner[bit as usize] != me as u32 {
                    c.candidates.insert(bit);
                    erase = false;
                }
            }
            if erase {
                spent.push(check);
            }
        }
        for check in spent {
            c.boundary.remove(check);
        }
    }

    /// `grow_cluster`: add the `per_step` lightest candidates, then merge with whatever the
    /// cluster touched. Whether anything changed.
    fn grow(&mut self, me: usize, weights: &[f64], per_step: usize) -> bool {
        if !self.clusters[me].active {
            return false;
        }
        self.candidates(me);
        self.clusters[me].merge_list.clear();
        let cands: Vec<u32> = self.clusters[me].candidates.iter().collect();
        let mut order: Vec<usize> = (0..cands.len()).collect();
        let w = |i: usize| weights[cands[i] as usize];
        order.sort_by(|&a, &b| w(a).partial_cmp(&w(b)).unwrap_or(std::cmp::Ordering::Equal));
        let take = per_step.min(order.len());
        // Equal weights among the taken or at the cut, in a sort long enough for std::sort to
        // be unstable.
        if order.len() > 16 && (1..(take + 1).min(order.len())).any(|k| w(order[k]) == w(order[k - 1])) {
            self.ties += 1;
            self.clusters[me].tied = true;
        }
        for &i in &order[..take] {
            self.add_bit_node(me, cands[i], false);
        }
        self.merge_intersecting(me);
        take > 0
    }

    /// `merge_with_intersecting_clusters`, with on-the-fly elimination.
    fn merge_intersecting(&mut self, me: usize) {
        let mut larger = me;
        let mut k = 0;
        let met = self.clusters[me].merge_list.len();
        while k < self.clusters[me].merge_list.len() {
            if k == met {
                // A cluster met while merging: robin-map may or may not visit it.
                self.ties += 1;
                self.clusters[larger].tied = true;
            }
            let other = self.clusters[me].merge_list[k] as usize;
            larger = self.merge(larger, other);
            k += 1;
        }
        if met > 1 {
            // Several clusters merged at once, in an order ldpc takes from pointer values.
            self.ties += 1;
            self.clusters[larger].tied = true;
        }
        let valid = self.on_the_fly(larger);
        self.clusters[larger].valid = valid;
    }

    /// `merge_clusters`: the smaller into the larger (by bits; ties keep `a`); the survivor.
    fn merge(&mut self, a: usize, b: usize) -> usize {
        let (smaller, larger) = if self.clusters[a].bits.len() < self.clusters[b].bits.len() { (a, b) } else { (b, a) };
        let bits: Vec<u32> = self.clusters[smaller].bits.iter().collect();
        for bit in bits {
            self.add_bit_node(larger, bit, true);
        }
        let boundary: Vec<u32> = self.clusters[smaller].boundary.iter().collect();
        let enclosed: Vec<u32> = self.clusters[smaller].enclosed.iter().collect();
        let l = &mut self.clusters[larger];
        for c in boundary {
            l.boundary.insert(c);
        }
        for s in enclosed {
            l.enclosed.insert(s);
        }
        self.clusters[smaller].active = false;
        self.clusters[larger].tied |= self.clusters[smaller].tied;
        larger
    }

    /// `sort_non_pivot_cols`: the cluster's free columns, lightest first by BP's posteriors.
    fn sort_non_pivot_cols(&mut self, me: usize, weights: &[f64]) {
        let c = &mut self.clusters[me];
        if c.plu.not_pivot_cols.len() < 2 {
            return;
        }
        let wt = |col: usize| weights[c.local_to_bit[col] as usize];
        let mut cols = std::mem::take(&mut c.plu.not_pivot_cols);
        cols.sort_by(|&a, &b| wt(a).partial_cmp(&wt(b)).unwrap_or(std::cmp::Ordering::Equal));
        if cols.len() > 16 && cols.windows(2).any(|p| wt(p[0]) == wt(p[1])) {
            self.ties += 1;
            c.tied = true;
        }
        c.plu.not_pivot_cols = cols;
    }

    /// `apply_on_the_fly_elimination`: whether the cluster's syndrome is in its image.
    fn on_the_fly(&mut self, me: usize) -> bool {
        let c = &mut self.clusters[me];
        for idx in c.plu.col_count..c.bits.len() {
            c.plu.add_column(&c.pcm[idx]);
        }
        c.syndrome.resize(c.checks.len(), 0);
        for s in c.enclosed.iter() {
            c.syndrome[c.check_to_local[&s] as usize] = 1;
        }
        let start = c.plu.cols_eliminated;
        c.plu.rref_with_y_image_check(&c.syndrome, start)
    }
}

/// `DenseOsdDecoder::osd_decode` on one cluster: its LSD-0 solution, or the lightest (by
/// Hamming weight, as ldpc weighs them here; the first of equals) of the candidates that set
/// free columns and solve for the pivots. Exhaustive search tries every pattern of the first
/// `order` free columns; combination sweep tries each free column alone and every pair among
/// the first `order`. (ldpc builds patterns of `order` bits in vectors of the free-column
/// count, so patterns beyond it repeat earlier ones; they cannot win and are not tried.)
fn dense_osd(c: &Cluster, method: OsdMethod, order: usize) -> Vec<u8> {
    let syndrome = &c.syndrome;
    let best0 = c.plu.lu_solve(syndrome);
    let free = &c.plu.not_pivot_cols;
    let k = free.len();
    if k == 0 {
        return best0;
    }
    let mut best_weight = best0.iter().filter(|&&x| x == 1).count();
    let mut best = best0;
    let mut try_flips = |flips: &[usize]| {
        let mut t = syndrome.clone();
        for &i in flips {
            for &e in &c.pcm[free[i]] {
                t[e as usize] ^= 1;
            }
        }
        let mut sol = c.plu.lu_solve(&t);
        for &col in free {
            sol[col] = 0;
        }
        for &i in flips {
            sol[free[i]] = 1;
        }
        let mut decoded = vec![0u8; syndrome.len()];
        let mut weight = 0;
        for (j, &x) in sol.iter().enumerate() {
            if x == 1 {
                for &e in &c.pcm[j] {
                    decoded[e as usize] ^= 1;
                }
                weight += 1;
            }
        }
        if decoded == *syndrome && weight < best_weight {
            best_weight = weight;
            best = sol;
        }
    };
    match method {
        OsdMethod::Osd0 => {}
        OsdMethod::Exhaustive(_) => {
            let bits = order.min(k).min(24);
            for pattern in 1u32..1 << bits {
                let flips: Vec<usize> = (0..bits).filter(|&i| pattern >> i & 1 == 1).collect();
                try_flips(&flips);
            }
        }
        OsdMethod::CombinationSweep(_) => {
            for i in 0..k {
                try_flips(&[i]);
            }
            let first = order.min(k);
            for i in 0..first {
                for j in i + 1..first {
                    try_flips(&[i, j]);
                }
            }
        }
    }
    best
}

/// BP+LSD on a check matrix given by its columns.
pub struct BpLsd {
    pub bp: Bp,
    columns: Vec<Vec<u32>>,
    rows: Vec<Vec<u32>>,
    pub method: Method,
    pub max_iter: usize,
    pub lsd: OsdMethod,
    pub bits_per_step: usize,
    pub always_run: bool,
}

/// One decode's state; one per thread.
pub struct BpLsdWork {
    pub bp: BpWork,
    /// The correction: one byte per fault.
    pub correction: Vec<u8>,
    /// Choices made where ldpc's order is not reproducible (see the module comment).
    pub ties: u32,
    /// When set, each final untied cluster's bits in set order, by cluster id: a test hook.
    pub record: bool,
    pub cluster_bits: Vec<(u32, Vec<u32>)>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BpLsdOutcome {
    pub converged: bool,
    pub iterations: usize,
    /// Whether the correction explains the syndrome (LSD fails only on a syndrome no set of
    /// faults explains).
    pub solved: bool,
}

impl BpLsd {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        num_checks: usize,
        columns: Vec<Vec<u32>>,
        priors: &[f64],
        method: Method,
        max_iter: usize,
        lsd: OsdMethod,
        bits_per_step: usize,
        always_run: bool,
    ) -> Result<BpLsd, String> {
        let columns: Vec<Vec<u32>> = columns
            .into_iter()
            .map(|mut c| {
                c.sort_unstable();
                c.dedup();
                c
            })
            .collect();
        let bp = Bp::new(num_checks, &columns, priors)?;
        let mut rows = vec![Vec::new(); num_checks];
        for (v, col) in columns.iter().enumerate() {
            for &c in col {
                rows[c as usize].push(v as u32);
            }
        }
        let bits_per_step = if bits_per_step == 0 { columns.len().max(1) } else { bits_per_step };
        Ok(BpLsd { bp, columns, rows, method, max_iter, lsd, bits_per_step, always_run })
    }

    pub fn work(&self) -> BpLsdWork {
        BpLsdWork { bp: self.bp.work(), correction: vec![0; self.columns.len()], ties: 0, record: false, cluster_bits: Vec::new() }
    }

    /// Decode a syndrome (one byte per check); the correction is left in `w.correction`.
    pub fn decode(&self, syndrome: &[u8], w: &mut BpLsdWork) -> BpLsdOutcome {
        w.ties = 0;
        w.cluster_bits.clear();
        if syndrome.iter().all(|&s| s == 0) {
            w.correction.fill(0);
            return BpLsdOutcome { converged: true, iterations: 0, solved: true };
        }
        let out = self.bp.decode(syndrome, self.method, self.max_iter, &mut w.bp);
        if out.converged && !self.always_run {
            w.correction.copy_from_slice(&w.bp.hard);
            return BpLsdOutcome { converged: true, iterations: out.iterations, solved: true };
        }
        let solved = self.lsd_decode(syndrome, w);
        BpLsdOutcome { converged: out.converged, iterations: out.iterations, solved }
    }

    /// `lsd_decode` on BP's posteriors.
    fn lsd_decode(&self, syndrome: &[u8], w: &mut BpLsdWork) -> bool {
        // Borrowed out of the work for the decode, and put back.
        let llr = std::mem::take(&mut w.bp.llr);
        let solved = self.lsd_on(syndrome, &llr, w);
        w.bp.llr = llr;
        solved
    }

    fn lsd_on(&self, syndrome: &[u8], weights: &[f64], w: &mut BpLsdWork) -> bool {
        let mut s = Lsd {
            rows: &self.rows,
            columns: &self.columns,
            clusters: Vec::new(),
            bit_owner: vec![NONE; self.columns.len()],
            check_owner: vec![NONE; self.rows.len()],
            ties: 0,
        };
        for (i, &b) in syndrome.iter().enumerate() {
            if b != 0 {
                s.check_owner[i] = s.clusters.len() as u32;
                s.clusters.push(Cluster::new(i as u32));
            }
        }
        let mut invalid: Vec<usize> = (0..s.clusters.len()).collect();
        let mut solved = true;
        while !invalid.is_empty() {
            let mut grew = false;
            for &cl in &invalid {
                if s.clusters[cl].active {
                    grew |= s.grow(cl, weights, self.bits_per_step);
                }
            }
            invalid = (0..s.clusters.len()).filter(|&c| s.clusters[c].active && !s.clusters[c].valid).collect();
            if !grew && !invalid.is_empty() {
                // Nothing left to add: no set of faults explains this syndrome. (ldpc loops.)
                solved = false;
                break;
            }
            // Equal sizes keep creation order; std::sort may not, past 16 clusters.
            invalid.sort_by_key(|&c| s.clusters[c].bits.len());
            if invalid.len() > 16 && invalid.windows(2).any(|p| s.clusters[p[0]].bits.len() == s.clusters[p[1]].bits.len()) {
                s.ties += 1;
                for &c in &invalid {
                    s.clusters[c].tied = true;
                }
            }
        }
        let order = match self.lsd {
            OsdMethod::Osd0 => 0,
            OsdMethod::Exhaustive(k) | OsdMethod::CombinationSweep(k) => k,
        };
        w.correction.fill(0);
        if order > 0 {
            self.apply_lsdw(&mut s, order, weights, w);
        } else {
            for c in s.clusters.iter().filter(|c| c.active) {
                let solution = c.plu.lu_solve(&c.syndrome);
                for (i, &x) in solution.iter().enumerate() {
                    if x == 1 {
                        w.correction[c.local_to_bit[i] as usize] = 1;
                    }
                }
            }
        }
        if w.record {
            w.cluster_bits = s.clusters.iter().filter(|c| c.active && !c.tied).map(|c| (c.id, c.bits.iter().collect())).collect();
        }
        w.ties += s.ties;
        solved
    }

    /// `apply_lsdw`: grow each cluster until it has `order` free columns (or `order` more
    /// steps, or its initial size in steps, or the whole matrix), then search each.
    fn apply_lsdw(&self, s: &mut Lsd, order: usize, weights: &[f64], w: &mut BpLsdWork) {
        let n = self.columns.len();
        for cl in 0..s.clusters.len() {
            if !s.clusters[cl].active {
                continue;
            }
            let initial = s.clusters[cl].bits.len();
            let mut grown = 0;
            while s.clusters[cl].plu.not_pivot_cols.len() < order && grown < order && s.clusters[cl].bits.len() < n && grown <= initial {
                s.grow(cl, weights, 1);
                grown += 1;
            }
        }
        for cl in 0..s.clusters.len() {
            if !s.clusters[cl].active {
                continue;
            }
            s.sort_non_pivot_cols(cl, weights);
            let c = &s.clusters[cl];
            let solution = dense_osd(c, self.lsd, order);
            for (i, &x) in solution.iter().enumerate() {
                if x == 1 {
                    w.correction[c.local_to_bit[i] as usize] = 1;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::surface_code::Xorshift;

    fn random_code(m: usize, n: usize, rng: &mut Xorshift) -> Vec<Vec<u32>> {
        (0..n)
            .map(|_| {
                let mut col: Vec<u32> = Vec::new();
                while col.len() < 3 {
                    let c = (rng.next_u64() % m as u64) as u32;
                    if !col.contains(&c) {
                        col.push(c);
                    }
                }
                col
            })
            .collect()
    }

    fn syndrome_of(columns: &[Vec<u32>], m: usize, e: &[u8]) -> Vec<u8> {
        let mut s = vec![0u8; m];
        for (v, &x) in e.iter().enumerate() {
            if x != 0 {
                for &c in &columns[v] {
                    s[c as usize] ^= 1;
                }
            }
        }
        s
    }

    /// Whatever BP does, LSD's correction explains the syndrome, and LSD ran on most shots.
    #[test]
    fn every_correction_explains_its_syndrome() {
        let mut rng = Xorshift::new(21);
        let (m, n) = (30, 60);
        let columns = random_code(m, n, &mut rng);
        let priors = vec![0.05; n];
        for per_step in [1, 3, 0] {
            let dec = BpLsd::new(m, columns.clone(), &priors, Method::MinSum { scale: 0.625 }, 2, OsdMethod::Osd0, per_step, false).unwrap();
            let mut w = dec.work();
            let mut lsd_runs = 0;
            for _ in 0..300 {
                let e: Vec<u8> = (0..n).map(|_| u8::from(rng.next_f64() < 0.08)).collect();
                let s = syndrome_of(&columns, m, &e);
                let out = dec.decode(&s, &mut w);
                lsd_runs += usize::from(!out.converged);
                assert!(out.solved);
                assert_eq!(syndrome_of(&columns, m, &w.correction), s, "bits per step {per_step}");
            }
            assert!(lsd_runs > 50, "LSD ran on {lsd_runs} of 300");
        }
    }

    /// Higher orders explain the syndrome too, and grow clusters only where LSD runs.
    #[test]
    fn higher_orders_explain_the_syndrome() {
        let mut rng = Xorshift::new(22);
        let (m, n) = (30, 60);
        let columns = random_code(m, n, &mut rng);
        let priors: Vec<f64> = (0..n).map(|i| 0.02 + 0.001 * i as f64).collect();
        for lsd in [OsdMethod::Exhaustive(5), OsdMethod::CombinationSweep(8), OsdMethod::Exhaustive(10)] {
            let dec = BpLsd::new(m, columns.clone(), &priors, Method::MinSum { scale: 0.625 }, 2, lsd, 1, false).unwrap();
            let mut w = dec.work();
            for _ in 0..200 {
                let e: Vec<u8> = (0..n).map(|_| u8::from(rng.next_f64() < 0.08)).collect();
                let s = syndrome_of(&columns, m, &e);
                assert!(dec.decode(&s, &mut w).solved);
                assert_eq!(syndrome_of(&columns, m, &w.correction), s, "{lsd:?}");
            }
        }
    }

    /// A syndrome no set of faults explains ends (ldpc would loop) and says so.
    #[test]
    fn an_unexplainable_syndrome_is_reported() {
        let columns: Vec<Vec<u32>> = vec![vec![0, 1], vec![1, 2]];
        let dec = BpLsd::new(3, columns, &[0.1, 0.1], Method::ProductSum, 1, OsdMethod::Osd0, 1, true).unwrap();
        let mut w = dec.work();
        assert!(!dec.decode(&[1, 0, 0], &mut w).solved);
    }

    /// One likely fault is found when LSD always runs.
    #[test]
    fn one_fault_is_found() {
        let columns: Vec<Vec<u32>> = vec![vec![0], vec![0, 1], vec![1, 2], vec![2]];
        let dec = BpLsd::new(3, columns, &[0.01; 4], Method::MinSum { scale: 1.0 }, 20, OsdMethod::Osd0, 1, true).unwrap();
        let mut w = dec.work();
        dec.decode(&[0, 1, 1], &mut w);
        assert_eq!(w.correction, vec![0, 0, 1, 0]);
    }
}

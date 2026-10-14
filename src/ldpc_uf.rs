//! Union-find decoding on any check matrix, as `ldpc`'s `UnionFindDecoder` and the second
//! stage of its `BeliefFindDecoder` (src_cpp/union_find.hpp).
//!
//! WHY THIS EXISTS
//! ---------------
//! The union-find decoder of Delfosse and Nickerson grows a cluster around each fired check
//! until every cluster can explain its own checks, then solves each cluster alone. `ldpc`
//! generalises it beyond matching graphs (Delfosse, Londe and Beverland): a cluster is valid
//! once its syndrome is in the image of its own columns, and is solved by elimination
//! (`matrix_decode`, "inversion"). On a matrix whose columns have at most two checks it also
//! has the original: a cluster is valid once its parity is even or it touches the boundary
//! (a column with one check), and is solved by peeling a spanning tree (`peel_decode`). Growth
//! takes every neighbouring column at once, or, given weights (BP's LLRs in
//! `BeliefFindDecoder`), the `bits_per_step` lightest.
//!
//! The port gives `ldpc`'s corrections: its clusters' sets are walked in `tsl::robin_set`'s
//! order (`robin`), its sorts are the platform's `std::sort` (`cpp_random`), and a cluster's
//! elimination is `RowReduce::fast_solve`, which stops at the first prefix of the cluster's
//! columns (in their order) whose span holds the syndrome and returns the unique solution on
//! that prefix's independent columns. One choice cannot be reproduced: when a cluster meets two
//! or more others in one step, `ldpc` merges them in the order of their addresses' hashes, and
//! the order the merged sets are filled in can move later choices. A decode that met that (or,
//! on Windows, equal keys in a long sort, where `std::sort` is not reproduced) counts it in
//! `UfOutcome::ties`.
//!
//! Where `ldpc` would loop forever, this does not: a cluster that cannot grow and is still
//! invalid (a syndrome nothing explains) is reported unsolved, and a cluster whose peeling goes
//! round without end (its spanning tree's leaves run out, or come back, before its syndrome is
//! peeled) is solved by inversion and counted in `UfOutcome::stuck`.

use crate::cpp_random::Library;
use crate::robin::RobinSet;
use std::collections::HashMap;

const NONE: u32 = u32::MAX;
/// The virtual boundary check, `-1` in `ldpc` (whose hash, all ones, this matches).
const BOUNDARY: u32 = u32::MAX;
const UNSURE_SORT: bool = matches!(Library::NATIVE, Library::Msvc);

/// A check matrix to decode on.
#[derive(Clone, Debug)]
pub struct LdpcUf {
    num_checks: usize,
    columns: Vec<Vec<u32>>,
    rows: Vec<Vec<u32>>,
    /// Columns with one check: the planar code's boundary, for peeling.
    boundary_bits: Vec<bool>,
    max_degree_2: bool,
}

/// A decode's result.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UfOutcome {
    pub decoding: Vec<u8>,
    /// Whether every cluster became valid.
    pub solved: bool,
    /// Steps whose result may differ from `ldpc`'s (see the module notes).
    pub ties: u32,
    /// Clusters whose peeling did not finish (where `ldpc`'s loops forever), solved by
    /// inversion instead.
    pub stuck: u32,
}

struct Cluster {
    active: bool,
    valid: bool,
    contains_boundary_bits: bool,
    bits: RobinSet,
    checks: RobinSet,
    boundary_checks: RobinSet,
    candidates: Vec<u32>,
    enclosed: usize,
    merge_list: Vec<u32>,
    decoding: Vec<u32>,
}

impl Cluster {
    fn new(check: u32) -> Cluster {
        let mut c = Cluster { active: true, valid: false, contains_boundary_bits: false, bits: RobinSet::new(), checks: RobinSet::new(), boundary_checks: RobinSet::new(), candidates: Vec::new(), enclosed: 1, merge_list: Vec::new(), decoding: Vec::new() };
        c.boundary_checks.insert(check);
        c.checks.insert(check);
        c
    }
}

struct State<'a> {
    uf: &'a LdpcUf,
    peel: bool,
    clusters: Vec<Cluster>,
    bit_owner: Vec<u32>,
    check_owner: Vec<u32>,
    ties: u32,
}

impl State<'_> {
    fn candidates(&mut self, me: usize) {
        let mut spent = Vec::new();
        let mut cands = Vec::new();
        for check in self.clusters[me].boundary_checks.iter() {
            let mut erase = true;
            for &bit in &self.uf.rows[check as usize] {
                if self.bit_owner[bit as usize] != me as u32 {
                    cands.push(bit);
                    erase = false;
                }
            }
            if erase {
                spent.push(check);
            }
        }
        let c = &mut self.clusters[me];
        for check in spent {
            c.boundary_checks.remove(check);
        }
        c.candidates = cands;
    }

    fn note_merge(list: &mut Vec<u32>, other: u32) {
        if !list.contains(&other) {
            list.push(other);
        }
    }

    /// `add_bit_node_to_cluster`.
    fn add_bit(&mut self, me: usize, bit: u32) {
        let owner = self.bit_owner[bit as usize];
        if owner == me as u32 {
            return;
        }
        if owner == NONE {
            self.clusters[me].bits.insert(bit);
        } else {
            Self::note_merge(&mut self.clusters[me].merge_list, owner);
        }
        self.bit_owner[bit as usize] = me as u32;
        if self.peel && self.uf.boundary_bits[bit as usize] {
            self.clusters[me].contains_boundary_bits = true;
        }
        for &check in &self.uf.columns[bit as usize] {
            let owner = self.check_owner[check as usize];
            if owner == me as u32 {
                continue;
            }
            let c = &mut self.clusters[me];
            c.checks.insert(check);
            c.boundary_checks.insert(check);
            if owner != NONE {
                Self::note_merge(&mut c.merge_list, owner);
            }
            self.check_owner[check as usize] = me as u32;
        }
    }

    /// `merge_with_cluster`: `other` into `me`.
    fn merge(&mut self, me: usize, other: usize) {
        let bits: Vec<u32> = self.clusters[other].bits.iter().collect();
        let checks: Vec<u32> = self.clusters[other].checks.iter().collect();
        let boundary: Vec<u32> = self.clusters[other].boundary_checks.iter().collect();
        let (enclosed, has_boundary) = (self.clusters[other].enclosed, self.clusters[other].contains_boundary_bits);
        self.clusters[other].active = false;
        let c = &mut self.clusters[me];
        for b in bits {
            c.bits.insert(b);
            self.bit_owner[b as usize] = me as u32;
        }
        for k in checks {
            c.checks.insert(k);
            self.check_owner[k as usize] = me as u32;
        }
        for k in boundary {
            c.boundary_checks.insert(k);
        }
        c.contains_boundary_bits |= has_boundary;
        c.enclosed += enclosed;
    }

    /// `grow_cluster`; whether it had anything to add.
    fn grow(&mut self, me: usize, weights: Option<&[f64]>, per_step: usize) -> bool {
        if !self.clusters[me].active {
            return false;
        }
        self.candidates(me);
        self.clusters[me].merge_list.clear();
        let cands = std::mem::take(&mut self.clusters[me].candidates);
        match weights {
            None => {
                for &bit in &cands {
                    self.add_bit(me, bit);
                }
            }
            Some(w) => {
                let mut order: Vec<usize> = (0..cands.len()).collect();
                let wt = |i: usize| w[cands[i] as usize];
                Library::NATIVE.sort_by(&mut order, &mut |&a, &b| wt(a) < wt(b));
                let take = per_step.min(order.len());
                if UNSURE_SORT && order.len() > 16 && (1..(take + 1).min(order.len())).any(|k| wt(order[k]) == wt(order[k - 1])) {
                    self.ties += 1;
                }
                for &i in &order[..take] {
                    self.add_bit(me, cands[i]);
                }
            }
        }
        let merges = std::mem::take(&mut self.clusters[me].merge_list);
        if merges.len() > 1 {
            // Merged in an order ldpc takes from the clusters' addresses.
            self.ties += 1;
        }
        for &other in &merges {
            self.merge(me, other as usize);
        }
        self.clusters[me].merge_list = merges;
        self.clusters[me].candidates = cands;
        !self.clusters[me].candidates.is_empty()
    }

    /// `invert_decode`: solve the cluster on its own columns, as `RowReduce::fast_solve` does;
    /// sets `valid` and the cluster's decoding.
    fn invert(&mut self, me: usize, syndrome: &[u8], weights: Option<&[f64]>) {
        let c = &self.clusters[me];
        let mut cols: Vec<u32> = c.bits.iter().collect();
        if let Some(w) = weights {
            let unsorted = cols.clone();
            let mut order: Vec<usize> = (0..cols.len()).collect();
            Library::NATIVE.sort_by(&mut order, &mut |&a, &b| w[unsorted[a] as usize] < w[unsorted[b] as usize]);
            if UNSURE_SORT && order.len() > 16 && order.windows(2).any(|p| w[unsorted[p[0]] as usize] == w[unsorted[p[1]] as usize]) {
                self.ties += 1;
            }
            cols = order.iter().map(|&i| unsorted[i]).collect();
        }
        let local: HashMap<u32, usize> = c.checks.iter().enumerate().map(|(i, k)| (k, i)).collect();
        let words = local.len().div_ceil(64).max(1);
        let vec_of = |rows: &mut dyn Iterator<Item = usize>| {
            let mut v = vec![0u64; words];
            for r in rows {
                v[r / 64] ^= 1 << (r % 64);
            }
            v
        };
        let mut y = vec_of(&mut c.checks.iter().enumerate().filter(|&(_, k)| syndrome[k as usize] != 0).map(|(i, _)| i));
        // Reduced basis: (vector, pivot row, which pivot columns sum to it).
        let mut basis: Vec<(Vec<u64>, usize, Vec<u64>)> = Vec::new();
        let mut pivot_cols: Vec<u32> = Vec::new();
        let mut y_combo: Vec<u64> = Vec::new();
        let max_rank = local.len().min(cols.len());
        let lowest = |v: &[u64]| v.iter().enumerate().find(|(_, &x)| x != 0).map(|(i, &x)| i * 64 + x.trailing_zeros() as usize);
        let has = |v: &[u64], r: usize| v[r / 64] >> (r % 64) & 1 == 1;
        let xor = |a: &mut Vec<u64>, b: &[u64]| {
            if a.len() < b.len() {
                a.resize(b.len(), 0);
            }
            for (x, y) in a.iter_mut().zip(b) {
                *x ^= y;
            }
        };
        let mut in_image = false;
        for &bit in &cols {
            if pivot_cols.len() == max_rank {
                break;
            }
            let mut v = vec_of(&mut self.uf.columns[bit as usize].iter().filter_map(|k| local.get(k).copied()));
            let mut combo: Vec<u64> = Vec::new();
            for (bv, p, bc) in &basis {
                if has(&v, *p) {
                    xor(&mut v, bv);
                    xor(&mut combo, bc);
                }
            }
            let Some(p) = lowest(&v) else { continue };
            let k = pivot_cols.len();
            if combo.len() <= k / 64 {
                combo.resize(k / 64 + 1, 0);
            }
            combo[k / 64] ^= 1 << (k % 64);
            pivot_cols.push(bit);
            if has(&y, p) {
                xor(&mut y, &v);
                xor(&mut y_combo, &combo);
            }
            basis.push((v, p, combo));
            if lowest(&y).is_none() {
                in_image = true;
                break;
            }
        }
        let c = &mut self.clusters[me];
        c.valid = in_image;
        c.decoding = if in_image { pivot_cols.iter().enumerate().filter(|&(k, _)| y_combo.get(k / 64).is_some_and(|w| w >> (k % 64) & 1 == 1)).map(|(_, &b)| b).collect() } else { Vec::new() };
    }

    /// `peel_decode` on one cluster: peel a spanning tree of its checks; the erasure, or
    /// `None` where peeling goes round forever.
    fn peel(&mut self, me: usize, syndrome: &[u8]) -> Option<Vec<u32>> {
        let uf = self.uf;
        let c = &mut self.clusters[me];
        let mut synds = RobinSet::new();
        for k in c.checks.iter() {
            if syndrome[k as usize] != 0 {
                synds.insert(k);
            }
        }
        if c.contains_boundary_bits && c.enclosed % 2 == 1 {
            synds.insert(BOUNDARY);
        }
        // find_spanning_tree
        let mut tree_bits = RobinSet::new();
        for b in c.bits.iter() {
            tree_bits.insert(b);
        }
        if c.contains_boundary_bits {
            c.checks.insert(BOUNDARY);
        }
        let mut roots: HashMap<u32, u32> = c.checks.iter().map(|k| (k, k)).collect();
        let find = |roots: &mut HashMap<u32, u32>, mut k: u32| loop {
            let parent = *roots.entry(k).or_insert(0);
            if parent == k {
                return k;
            }
            k = parent;
        };
        let mut boundary_bit = NONE;
        for b in c.bits.iter() {
            let col = &uf.columns[b as usize];
            let (first, last) = (col[col.len() - 1], col[0]);
            let second = if first == last {
                boundary_bit = b;
                BOUNDARY
            } else {
                last
            };
            let (r0, r1) = (find(&mut roots, first), find(&mut roots, second));
            if r0 != r1 {
                roots.insert(r1, r0);
            } else {
                tree_bits.remove(b);
            }
        }
        let connectivity = |tree_bits: &RobinSet, k: u32| uf.rows[k as usize].iter().filter(|&&b| tree_bits.contains(b)).count();
        let mut leaves = RobinSet::new();
        for k in c.checks.iter() {
            if k == BOUNDARY || connectivity(&tree_bits, k) == 1 {
                leaves.insert(k);
            }
        }
        let mut erasure = Vec::new();
        let mut budget = 4 * (c.bits.len() + c.checks.len()) + 64;
        while synds.len() > 0 && budget > 0 {
            budget -= 1;
            let Some(leaf) = leaves.iter().next() else { return None };
            let bit = if leaf == BOUNDARY {
                boundary_bit
            } else {
                let row = &uf.rows[leaf as usize];
                row.iter().copied().find(|&b| tree_bits.contains(b)).unwrap_or(row[row.len() - 1])
            };
            if bit == NONE {
                return None;
            }
            let check2 = uf.columns[bit as usize].iter().copied().filter(|&k| k != leaf).last().unwrap_or(BOUNDARY);
            leaves.remove(leaf);
            tree_bits.remove(bit);
            if synds.contains(leaf) {
                erasure.push(bit);
                if !synds.remove(check2) {
                    synds.insert(check2);
                }
                synds.remove(leaf);
            }
            if check2 == BOUNDARY || connectivity(&tree_bits, check2) == 1 {
                leaves.insert(check2);
            }
        }
        (synds.len() == 0).then_some(erasure)
    }
}

impl LdpcUf {
    /// The decoder for a matrix of `num_checks` rows given by its columns' rows. Every column
    /// needs a check (as in `ldpc`).
    pub fn new(num_checks: usize, columns: Vec<Vec<u32>>) -> Result<LdpcUf, String> {
        let mut rows = vec![Vec::new(); num_checks];
        let mut max_degree_2 = true;
        let mut boundary_bits = vec![false; columns.len()];
        for (j, col) in columns.iter().enumerate() {
            match col.len() {
                0 => return Err("Invalid parity check matrix. Column weight is zero.".into()),
                1 => boundary_bits[j] = true,
                2 => {}
                _ => max_degree_2 = false,
            }
            for &r in col {
                if r as usize >= num_checks {
                    return Err(format!("check {r} is out of range"));
                }
                rows[r as usize].push(j as u32);
            }
        }
        Ok(LdpcUf { num_checks, columns, rows, boundary_bits, max_degree_2 })
    }

    /// Whether every column has at most two checks (so peeling applies).
    pub fn max_degree_2(&self) -> bool {
        self.max_degree_2
    }

    /// `matrix_decode` (`peel` false) or `peel_decode` (`peel` true; needs `max_degree_2`).
    /// `weights` (lower: likelier) order growth `per_step` columns at a time; without them,
    /// each step takes every neighbouring column.
    pub fn decode(&self, syndrome: &[u8], weights: Option<&[f64]>, per_step: usize, peel: bool) -> Result<UfOutcome, String> {
        let n = self.columns.len();
        if syndrome.len() != self.num_checks {
            return Err(format!("the syndrome has {} entries; the matrix has {} checks", syndrome.len(), self.num_checks));
        }
        if weights.is_some_and(|w| w.len() != n) {
            return Err(format!("the weights must have length {n}"));
        }
        if peel && !self.max_degree_2 {
            return Err("Peel decoder only works for planar codes. Use the matrix_decode method for more general codes.".into());
        }
        let mut s = State { uf: self, peel, clusters: Vec::new(), bit_owner: vec![NONE; n], check_owner: vec![NONE; self.num_checks], ties: 0 };
        for (i, &v) in syndrome.iter().enumerate() {
            if v != 0 {
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
                    grew |= s.grow(cl, weights, per_step);
                    if !peel {
                        s.invert(cl, syndrome, weights);
                    }
                }
            }
            invalid = (0..s.clusters.len()).filter(|&i| {
                let c = &s.clusters[i];
                c.active && if peel { c.enclosed % 2 == 1 && !c.contains_boundary_bits } else { !c.valid }
            }).collect();
            if !grew && !invalid.is_empty() {
                solved = false;
                break;
            }
            Library::NATIVE.sort_by(&mut invalid, &mut |&a, &b| s.clusters[a].bits.len() < s.clusters[b].bits.len());
            if UNSURE_SORT && invalid.len() > 16 && invalid.windows(2).any(|p| s.clusters[p[0]].bits.len() == s.clusters[p[1]].bits.len()) {
                s.ties += 1;
            }
        }
        let mut decoding = vec![0u8; n];
        let mut stuck = 0;
        if solved {
            for cl in 0..s.clusters.len() {
                if !s.clusters[cl].active {
                    continue;
                }
                let bits = if peel {
                    match s.peel(cl, syndrome) {
                        Some(bits) => bits,
                        None => {
                            stuck += 1;
                            s.clusters[cl].checks.remove(BOUNDARY);
                            s.invert(cl, syndrome, weights);
                            s.clusters[cl].decoding.clone()
                        }
                    }
                } else {
                    s.clusters[cl].decoding.clone()
                };
                for b in bits {
                    decoding[b as usize] = 1;
                }
            }
        }
        Ok(UfOutcome { decoding, solved, ties: s.ties, stuck })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::surface_code::Xorshift;

    fn syndrome_of(columns: &[Vec<u32>], m: usize, e: &[u8]) -> Vec<u8> {
        let mut s = vec![0u8; m];
        for (j, col) in columns.iter().enumerate() {
            if e[j] != 0 {
                for &r in col {
                    s[r as usize] ^= 1;
                }
            }
        }
        s
    }

    /// A ring of `n` checks, each fault between neighbours, plus one boundary fault on check 0.
    fn ring(n: u32) -> Vec<Vec<u32>> {
        let mut cols: Vec<Vec<u32>> = (0..n).map(|i| vec![i, (i + 1) % n]).map(|mut v| { v.sort(); v }).collect();
        cols.push(vec![0]);
        cols
    }

    #[test]
    fn inversion_explains_every_syndrome_in_the_image() {
        let mut rng = Xorshift::new(5);
        for _ in 0..200 {
            let (m, n) = (8 + (rng.next_u64() % 10) as usize, 20 + (rng.next_u64() % 20) as usize);
            let cols: Vec<Vec<u32>> = (0..n).map(|_| {
                let mut c: Vec<u32> = (0..m as u32).filter(|_| (rng.next_u64() % 4) == 0).collect();
                if c.is_empty() {
                    c.push((rng.next_u64() % m as u64) as u32);
                }
                c
            }).collect();
            let uf = LdpcUf::new(m, cols.clone()).unwrap();
            let e: Vec<u8> = (0..n).map(|_| ((rng.next_u64() % 8) == 0) as u8).collect();
            let s = syndrome_of(&cols, m, &e);
            let w: Vec<f64> = (0..n).map(|_| (rng.next_u64() % 5) as f64).collect();
            for weights in [None, Some(&w[..])] {
                let out = uf.decode(&s, weights, 1 + (rng.next_u64() % 3) as usize, false).unwrap();
                assert!(out.solved);
                assert_eq!(syndrome_of(&cols, m, &out.decoding), s);
            }
        }
    }

    #[test]
    fn peeling_corrects_a_ring_with_a_boundary() {
        let cols = ring(9);
        let uf = LdpcUf::new(9, cols.clone()).unwrap();
        assert!(uf.max_degree_2());
        let mut rng = Xorshift::new(2);
        for _ in 0..200 {
            let e: Vec<u8> = (0..cols.len()).map(|_| ((rng.next_u64() % 6) == 0) as u8).collect();
            let s = syndrome_of(&cols, 9, &e);
            let out = uf.decode(&s, None, usize::MAX, true).unwrap();
            assert!(out.solved);
            assert_eq!(syndrome_of(&cols, 9, &out.decoding), s);
        }
    }

    #[test]
    fn errors_as_ldpc_raises_them() {
        assert!(LdpcUf::new(2, vec![vec![0], vec![]]).is_err());
        let uf = LdpcUf::new(2, vec![vec![0, 1], vec![0, 1], vec![0, 1]]).unwrap();
        assert!(uf.max_degree_2());
        let tri = LdpcUf::new(3, vec![vec![0, 1, 2]]).unwrap();
        assert!(tri.decode(&[1, 0, 0], None, 1, true).is_err());
        // Nothing explains a single fired check here: reported, where ldpc would loop.
        let out = uf.decode(&[1, 0], None, 1, false).unwrap();
        assert!(!out.solved);
    }
}

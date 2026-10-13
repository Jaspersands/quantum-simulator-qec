//! PyMatching's `UserGraph` (`user_graph.cc`, PyMatching 2.4): a matching graph built by hand.
//! Nodes, edges carrying sets of fault ids, weights and error probabilities, boundary nodes, and
//! the five ways a parallel edge merges. It decodes on this crate's sparse blossom, built with
//! PyMatching's own discretisation, so the weights it minimises are PyMatching's to the bit:
//! integer weights as they are, otherwise scaled so the heaviest edge is 2^24 − 1, rounded, and
//! doubled; a negative weight's edge pre-applied (its endpoints toggled, its fault ids flipped,
//! its weight counted) and matched at its absolute weight; edges to boundary nodes made boundary
//! edges, the lightest kept where several meet one node.

use std::collections::BTreeSet;

use crate::dem_decoder::DecodeError;
use crate::dem_program::{DemInstr, DemProgram, OBS};
use crate::sparse::{Scratch, SparseGraph};

/// PyMatching's `NUM_DISTINCT_WEIGHTS - 1`: the heaviest edge after scaling, and the largest
/// weight a user edge may have.
pub const MAX_USER_EDGE_WEIGHT: f64 = ((1u32 << 24) - 1) as f64;

/// How an edge joins one already between the same nodes (PyMatching's `merge_strategy`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Merge {
    Disallow,
    Independent,
    SmallestWeight,
    KeepOriginal,
    Replace,
}

impl Merge {
    pub fn parse(name: &str) -> Result<Merge, String> {
        Ok(match name {
            "disallow" => Merge::Disallow,
            "independent" => Merge::Independent,
            "smallest-weight" => Merge::SmallestWeight,
            "keep-original" => Merge::KeepOriginal,
            "replace" => Merge::Replace,
            _ => return Err(format!("Merge strategy \"{name}\" not recognised.")),
        })
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct UserEdge {
    pub node1: usize,
    /// None for the boundary.
    pub node2: Option<usize>,
    pub fault_ids: Vec<usize>,
    pub weight: f64,
    /// Outside [0, 1] when unknown.
    pub error_probability: f64,
}

/// PyMatching's merge of two independent mechanisms' weights (log-likelihood ratios), computed
/// without overflow.
pub fn merge_weights(a: f64, b: f64) -> f64 {
    let sgn = 1f64.copysign(a) * 1f64.copysign(b);
    let signed_min = sgn * a.abs().min(b.abs());
    signed_min + (1.0 + (-(a + b).abs()).exp()).ln() - (1.0 + (-(a - b).abs()).exp()).ln()
}

/// A graph built for decoding (rebuilt after any change).
struct Built {
    graph: SparseGraph,
    /// The weight scale: a reported weight is the integer total over this.
    normalising: f64,
    /// Nodes whose detection is toggled by the negative-weight edges, sorted.
    neg_dets: Vec<u32>,
    /// Fault ids the negative-weight edges flip (each listed once if flipped an odd number of times).
    neg_faults: Vec<usize>,
    neg_sum: i64,
    /// Per sparse edge, its fault ids (for more than 64) and its user endpoints (-1: boundary).
    faults: Vec<Vec<usize>>,
    /// The negative-weight edges, as user endpoints.
    neg_edges: Vec<(i64, i64)>,
    wide: bool,
}

#[derive(Clone, Debug, Default)]
pub struct UserGraph {
    /// Per node, the ids of the edges touching it, in the order they were added.
    adjacency: Vec<Vec<usize>>,
    pub edges: Vec<UserEdge>,
    boundary: BTreeSet<usize>,
    num_fault_ids: usize,
    all_have_probabilities: bool,
}

/// One shot's decoding: the fault ids flipped, and the weight.
#[derive(Clone, Debug, PartialEq)]
pub struct Decoded {
    pub faults: Vec<bool>,
    pub weight: f64,
}

pub struct UserDecoder {
    built: Built,
    num_nodes: usize,
    num_fault_ids: usize,
    is_boundary: Vec<bool>,
    has_boundary_nodes: bool,
}

impl UserGraph {
    pub fn new(num_nodes: usize, num_fault_ids: usize) -> UserGraph {
        UserGraph { adjacency: vec![Vec::new(); num_nodes], edges: Vec::new(), boundary: BTreeSet::new(), num_fault_ids, all_have_probabilities: true }
    }

    pub fn num_nodes(&self) -> usize {
        self.adjacency.len()
    }

    pub fn num_edges(&self) -> usize {
        self.edges.len()
    }

    pub fn num_fault_ids(&self) -> usize {
        self.num_fault_ids
    }

    pub fn num_detectors(&self) -> usize {
        self.num_nodes() - self.boundary.len()
    }

    pub fn boundary(&self) -> &BTreeSet<usize> {
        &self.boundary
    }

    pub fn all_edges_have_error_probabilities(&self) -> bool {
        self.all_have_probabilities
    }

    pub fn ensure_num_fault_ids(&mut self, n: usize) {
        self.num_fault_ids = self.num_fault_ids.max(n);
    }

    fn ensure_nodes(&mut self, n: usize) {
        if self.adjacency.len() < n {
            self.adjacency.resize(n, Vec::new());
        }
    }

    pub fn set_boundary(&mut self, boundary: BTreeSet<usize>) {
        if let Some(&m) = boundary.iter().next_back() {
            self.ensure_nodes(m + 1);
        }
        self.boundary = boundary;
    }

    pub fn is_boundary(&self, node: Option<usize>) -> bool {
        node.is_none_or(|n| self.boundary.contains(&n))
    }

    /// The edge between `node` and `other` (None: the boundary edge), if any.
    pub fn find(&self, node: usize, other: Option<usize>) -> Option<usize> {
        let adj = self.adjacency.get(node)?;
        adj.iter().copied().find(|&e| {
            let ed = &self.edges[e];
            let across = if ed.node1 == node { ed.node2 } else { Some(ed.node1) };
            across == other
        })
    }

    fn note_new(&mut self, fault_ids: &[usize], p: f64) {
        if let Some(&m) = fault_ids.iter().max() {
            self.num_fault_ids = self.num_fault_ids.max(m + 1);
        }
        if !(0.0..=1.0).contains(&p) {
            self.all_have_probabilities = false;
        }
    }

    fn merge(&mut self, e: usize, fault_ids: &[usize], weight: f64, p: f64, how: Merge) -> Result<(), String> {
        let old = &self.edges[e];
        match how {
            Merge::Disallow => {
                let n2 = old.node2.map_or(usize::MAX, |v| v);
                Err(format!(
                    "Edge ({}, {n2}) already exists in the graph. Parallel edges not permitted with the provided `disallow` `merge_strategy`. Please provide a different `merge_strategy`.",
                    old.node1
                ))
            }
            Merge::KeepOriginal => Ok(()),
            Merge::SmallestWeight if weight >= old.weight => Ok(()),
            Merge::Replace | Merge::SmallestWeight => {
                let ed = &mut self.edges[e];
                ed.weight = weight;
                ed.error_probability = p;
                ed.fault_ids = fault_ids.to_vec();
                if !(0.0..=1.0).contains(&p) {
                    self.all_have_probabilities = false;
                }
                Ok(())
            }
            Merge::Independent => {
                let ed = &mut self.edges[e];
                let q = ed.error_probability;
                let valid = |x: f64| (0.0..=1.0).contains(&x);
                ed.weight = merge_weights(weight, ed.weight);
                ed.error_probability = if valid(q) && valid(p) { p * (1.0 - q) + q * (1.0 - p) } else { -1.0 };
                if !valid(ed.error_probability) {
                    self.all_have_probabilities = false;
                }
                Ok(())
            }
        }
    }

    /// PyMatching's `add_or_merge_edge`.
    pub fn add_edge(&mut self, node1: usize, node2: usize, fault_ids: &[usize], weight: f64, p: f64, how: Merge) -> Result<(), String> {
        self.ensure_nodes(node1.max(node2) + 1);
        match self.find(node1, Some(node2)) {
            Some(e) => self.merge(e, fault_ids, weight, p, how),
            None => {
                let id = self.edges.len();
                self.edges.push(UserEdge { node1, node2: Some(node2), fault_ids: fault_ids.to_vec(), weight, error_probability: p });
                self.adjacency[node1].push(id);
                if node1 != node2 {
                    self.adjacency[node2].push(id);
                }
                self.note_new(fault_ids, p);
                Ok(())
            }
        }
    }

    /// PyMatching's `add_or_merge_boundary_edge`.
    pub fn add_boundary_edge(&mut self, node: usize, fault_ids: &[usize], weight: f64, p: f64, how: Merge) -> Result<(), String> {
        self.ensure_nodes(node + 1);
        match self.find(node, None) {
            Some(e) => self.merge(e, fault_ids, weight, p, how),
            None => {
                let id = self.edges.len();
                self.edges.push(UserEdge { node1: node, node2: None, fault_ids: fault_ids.to_vec(), weight, error_probability: p });
                self.adjacency[node].push(id);
                self.note_new(fault_ids, p);
                Ok(())
            }
        }
    }

    /// PyMatching's graph of a detector error model: each graph-like piece an edge (one
    /// detector: to the boundary) of weight ln((1 − p)/p), parallel pieces merged as
    /// independent mechanisms.
    pub fn from_dem(dem: &DemProgram) -> Result<UserGraph, String> {
        let stats = dem.stats();
        let mut g = UserGraph::new(stats.num_detectors as usize, stats.num_observables);
        for i in &dem.flattened()?.instrs {
            let DemInstr::Error { p, pieces, .. } = i else { continue };
            let p = *p;
            if p <= 0.0 {
                continue; // as PyMatching
            }
            for piece in pieces {
                let dets: Vec<usize> = piece.iter().filter(|&&t| t & OBS == 0).map(|&t| t as usize).collect();
                let obs: Vec<usize> = piece.iter().filter(|&&t| t & OBS != 0).map(|&t| (t & !OBS) as usize).collect();
                let w = ((1.0 - p) / p).ln();
                match dets.as_slice() {
                    [a, b] => g.add_edge(*a, *b, &obs, w, p, Merge::Independent)?,
                    [a] => g.add_boundary_edge(*a, &obs, w, p, Merge::Independent)?,
                    _ => {}
                }
            }
        }
        Ok(g)
    }

    /// PyMatching's `get_edge_weight_normalising_constant` (before doubling).
    fn normalising_constant(&self) -> Result<f64, String> {
        let mut max = 0f64;
        let mut integral = true;
        for e in &self.edges {
            max = max.max(e.weight.abs());
            integral &= e.weight.round() == e.weight;
        }
        if max > MAX_USER_EDGE_WEIGHT {
            return Err(format!("maximum absolute edge weight of {} exceeded.", MAX_USER_EDGE_WEIGHT as u32));
        }
        Ok(if integral { 1.0 } else { MAX_USER_EDGE_WEIGHT / max })
    }

    /// The decoder for the graph as it is now.
    pub fn decoder(&self) -> Result<UserDecoder, String> {
        let n = self.num_nodes();
        let c = self.normalising_constant()?;
        let wide = self.num_fault_ids > 64;
        let mask = |f: &[usize]| -> u64 { if wide { 0 } else { f.iter().filter(|&&k| k < 64).fold(0u64, |m, &k| m ^ (1u64 << k)) } };
        let mut toggled = vec![false; n];
        let mut neg_faults: BTreeSet<usize> = BTreeSet::new();
        let mut neg_sum = 0i64;
        let mut neg_edges = Vec::new();
        let mut negative = |u: usize, v: Option<usize>, w: i64, f: &[usize], toggled: &mut Vec<bool>| {
            for k in f {
                if !neg_faults.remove(k) {
                    neg_faults.insert(*k);
                }
            }
            toggled[u] ^= true;
            if let Some(v) = v {
                toggled[v] ^= true;
            }
            neg_sum += w;
            neg_edges.push((u as i64, v.map_or(-1, |v| v as i64)));
        };
        let mut sparse: Vec<(u32, u32, f64, i64, u64)> = Vec::new();
        let mut faults: Vec<Vec<usize>> = Vec::new();
        // Boundary edges are held back, the lightest per node, then added in node order.
        let mut boundary_edge: Vec<Option<(i64, usize)>> = vec![None; n];
        for (id, e) in self.edges.iter().enumerate() {
            let w = ((e.weight * c).round() as i32 as i64) * 2;
            let b1 = self.is_boundary(Some(e.node1));
            let b2 = self.is_boundary(e.node2);
            if b2 && !b1 || b1 && !b2 {
                let u = if b1 { e.node2.expect("a non-boundary endpoint") } else { e.node1 };
                if boundary_edge[u].is_none_or(|(old, _)| old > w) {
                    boundary_edge[u] = Some((w, id));
                }
            } else if !b1 {
                let v = e.node2.expect("not the boundary");
                if w < 0 {
                    negative(e.node1, Some(v), w, &e.fault_ids, &mut toggled);
                }
                if e.node1 != v {
                    sparse.push((e.node1 as u32, v as u32, w.abs() as f64 / (2.0 * c), w.abs(), mask(&e.fault_ids)));
                    faults.push(e.fault_ids.clone());
                }
            }
        }
        for (u, be) in boundary_edge.iter().enumerate() {
            let Some((w, id)) = *be else { continue };
            let f = &self.edges[id].fault_ids;
            if w < 0 {
                negative(u, None, w, f, &mut toggled);
            }
            sparse.push((u as u32, n as u32, w.abs() as f64 / (2.0 * c), w.abs(), mask(f)));
            faults.push(f.clone());
        }
        let neg_dets: Vec<u32> = toggled.iter().enumerate().filter(|(_, &t)| t).map(|(k, _)| k as u32).collect();
        let graph = SparseGraph::from_weighted(n, &sparse);
        let mut is_boundary = vec![false; n];
        for &b in &self.boundary {
            is_boundary[b] = true;
        }
        Ok(UserDecoder {
            built: Built { graph, normalising: 2.0 * c, neg_dets, neg_faults: neg_faults.into_iter().collect(), neg_sum, faults, neg_edges, wide },
            num_nodes: n,
            num_fault_ids: self.num_fault_ids,
            has_boundary_nodes: !self.boundary.is_empty(),
            is_boundary,
        })
    }
}

const NO_MATCHING: &str = "No perfect matching could be found. This likely means that the syndrome has odd parity in the support of a connected component without a boundary.";

impl UserDecoder {
    pub fn scratch(&self) -> Scratch {
        Scratch::new(&self.built.graph)
    }

    pub fn num_fault_ids(&self) -> usize {
        self.num_fault_ids
    }

    pub fn num_nodes(&self) -> usize {
        self.num_nodes
    }

    /// The defects the matcher sees: detection events off the boundary nodes, toggled by the
    /// negative-weight edges, sorted.
    fn defects(&self, events: &[u64]) -> Result<Vec<u32>, String> {
        // The usual case, quickly: sorted distinct events, no boundary nodes, no negative weights.
        if self.built.neg_dets.is_empty() && !self.has_boundary_nodes && events.windows(2).all(|w| w[0] < w[1]) {
            if let Some(&last) = events.last() {
                if last as usize >= self.num_nodes {
                    return Err(format!("The detection event with index {last} does not correspond to a node in the graph, which only has {} nodes.", self.num_nodes));
                }
            }
            return Ok(events.iter().map(|&d| d as u32).collect());
        }
        let mut on = BTreeSet::new();
        for &d in events {
            let d = d as usize;
            if d >= self.num_nodes {
                return Err(format!("The detection event with index {d} does not correspond to a node in the graph, which only has {} nodes.", self.num_nodes));
            }
            if self.is_boundary[d] {
                continue;
            }
            if !on.remove(&(d as u32)) {
                on.insert(d as u32);
            }
        }
        for &d in &self.built.neg_dets {
            if !on.remove(&d) {
                on.insert(d);
            }
        }
        Ok(on.into_iter().collect())
    }

    fn err(e: DecodeError) -> String {
        match e {
            DecodeError::Unmatchable => NO_MATCHING.into(),
            other => format!("{other:?}"),
        }
    }

    /// One shot from its detection events as sorted distinct node indices, into `out` (the
    /// fault ids flipped, bit-packed, zeroed first): the weight. Allocation-free on the usual
    /// path (no boundary nodes, no negative weights, at most 64 fault ids); `buf` is reused.
    pub fn decode_packed(&self, scratch: &mut Scratch, sorted_events: &[u32], buf: &mut Vec<u32>, out: &mut [u8]) -> Result<f64, String> {
        out.iter_mut().for_each(|b| *b = 0);
        let b = &self.built;
        if b.wide || !b.neg_dets.is_empty() || self.has_boundary_nodes {
            let events: Vec<u64> = sorted_events.iter().map(|&e| e as u64).collect();
            let r = self.decode(scratch, &events)?;
            for (k, &f) in r.faults.iter().enumerate() {
                if f {
                    out[k / 8] |= 1 << (k % 8);
                }
            }
            return Ok(r.weight);
        }
        if let Some(&last) = sorted_events.last() {
            if last as usize >= self.num_nodes {
                return Err(format!("The detection event with index {last} does not correspond to a node in the graph, which only has {} nodes.", self.num_nodes));
            }
        }
        buf.clear();
        buf.extend_from_slice(sorted_events);
        let p = b.graph.decode(scratch, buf).map_err(Self::err)?;
        let mut obs = p.observables;
        for &k in &b.neg_faults {
            obs ^= 1u64 << k;
        }
        for (k, byte) in out.iter_mut().enumerate().take(8) {
            *byte = (obs >> (8 * k)) as u8;
        }
        // Bits past the fault count stay clear.
        let nf = self.num_fault_ids;
        if !nf.is_multiple_of(8) && nf / 8 < out.len() {
            out[nf / 8] &= (1u8 << (nf % 8)) - 1;
        }
        Ok((p.iweight + b.neg_sum) as f64 / b.normalising)
    }

    /// One shot: the fault ids flipped and the matching's weight (PyMatching's `decode`).
    pub fn decode(&self, scratch: &mut Scratch, events: &[u64]) -> Result<Decoded, String> {
        let b = &self.built;
        let defects = self.defects(events)?;
        let mut faults = vec![false; self.num_fault_ids];
        let iweight = if b.wide {
            let (p, ids) = b.graph.decode_with_edge_ids(scratch, &defects).map_err(Self::err)?;
            for id in ids {
                for &k in &b.faults[id as usize] {
                    if let Some(f) = faults.get_mut(k) {
                        *f ^= true;
                    }
                }
            }
            p.iweight
        } else {
            let p = b.graph.decode(scratch, &defects).map_err(Self::err)?;
            for (k, f) in faults.iter_mut().enumerate().take(64) {
                *f = p.observables >> k & 1 != 0;
            }
            p.iweight
        };
        // Fault ids past the count (a merge can bring them, as in PyMatching) are dropped.
        for &k in &b.neg_faults {
            if let Some(f) = faults.get_mut(k) {
                *f ^= true;
            }
        }
        Ok(Decoded { faults, weight: (iweight + b.neg_sum) as f64 / b.normalising })
    }

    /// The edges the correction flips, as user endpoints (-1: the boundary), the negative-weight
    /// edges included (PyMatching's `decode_to_edges_array`).
    pub fn decode_to_edges(&self, scratch: &mut Scratch, events: &[u64]) -> Result<Vec<(i64, i64)>, String> {
        let b = &self.built;
        let defects = self.defects(events)?;
        let ids = b.graph.decode_edge_ids(scratch, &defects).map_err(Self::err)?;
        let n = self.num_nodes as u32;
        let user = |(u, v): (u32, u32)| -> (i64, i64) { (u as i64, if v == n { -1 } else { v as i64 }) };
        let key = |(u, v): (i64, i64)| if v != -1 && v < u { (v, u) } else { (u, v) };
        let mut set: BTreeSet<(i64, i64)> = BTreeSet::new();
        for &e in &b.neg_edges {
            if e.0 != e.1 && !set.remove(&key(e)) {
                set.insert(key(e));
            }
        }
        let mut out: Vec<(i64, i64)> = Vec::new();
        for id in ids {
            let e = key(user(b.graph.edge_ends(id)));
            if !set.remove(&e) {
                out.push(e);
            }
        }
        out.extend(set);
        Ok(out)
    }

    /// The matched pairs of detection events (-1: the boundary), PyMatching's
    /// `decode_to_matched_dets_array`. Refused for graphs with negative weights, as PyMatching.
    pub fn decode_to_matched_pairs(&self, scratch: &mut Scratch, events: &[u64]) -> Result<Vec<(i64, i64)>, String> {
        let b = &self.built;
        if b.neg_sum != 0 {
            return Err("Decoding to matched detection events not supported for graphs containing edges with negative weights.".into());
        }
        let defects = self.defects(events)?;
        let (_, pairs) = b.graph.decode_pairs(scratch, &defects).map_err(Self::err)?;
        let n = self.num_nodes as u32;
        Ok(pairs.into_iter().map(|(a, c)| (a as i64, if c == n { -1 } else { c as i64 })).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn line(n: usize) -> UserGraph {
        let mut g = UserGraph::new(n, 0);
        for k in 0..n - 1 {
            g.add_edge(k, k + 1, &[k], 1.0, -1.0, Merge::Disallow).unwrap();
        }
        g.add_boundary_edge(0, &[], 1.0, -1.0, Merge::Disallow).unwrap();
        g.add_boundary_edge(n - 1, &[n - 1], 1.0, -1.0, Merge::Disallow).unwrap();
        g
    }

    #[test]
    fn a_repetition_line_decodes() {
        let g = line(5);
        let d = g.decoder().unwrap();
        let mut s = d.scratch();
        let r = d.decode(&mut s, &[1, 3]).unwrap();
        assert_eq!(r.faults, vec![false, true, true, false, false]);
        assert_eq!(r.weight, 2.0);
        assert!(d.decode(&mut s, &[9]).is_err());
    }

    #[test]
    fn merges() {
        let mut g = line(3);
        assert!(g.add_edge(0, 1, &[], 5.0, -1.0, Merge::Disallow).is_err());
        g.add_edge(0, 1, &[7], 5.0, -1.0, Merge::SmallestWeight).unwrap();
        assert_eq!(g.edges[0].fault_ids, vec![0]);
        g.add_edge(1, 0, &[7], 0.5, -1.0, Merge::SmallestWeight).unwrap();
        assert_eq!((g.edges[0].weight, g.edges[0].fault_ids.clone()), (0.5, vec![7]));
        g.add_edge(0, 1, &[1], 2.0, 0.1, Merge::Replace).unwrap();
        g.add_edge(0, 1, &[2], 2.0, 0.2, Merge::Independent).unwrap();
        assert!((g.edges[0].error_probability - (0.1 * 0.8 + 0.2 * 0.9)).abs() < 1e-12);
        assert_eq!(g.edges[0].fault_ids, vec![1]);
        // As PyMatching: only a new edge counts its fault ids, not a merge.
        assert_eq!(g.num_fault_ids(), 3);
    }

    #[test]
    fn negative_weights_are_pre_applied() {
        let mut g = UserGraph::new(2, 0);
        g.add_edge(0, 1, &[0], -2.0, -1.0, Merge::Disallow).unwrap();
        g.add_boundary_edge(0, &[1], 3.0, -1.0, Merge::Disallow).unwrap();
        g.add_boundary_edge(1, &[], 3.0, -1.0, Merge::Disallow).unwrap();
        let d = g.decoder().unwrap();
        let mut s = d.scratch();
        // Its endpoints fire: the negative edge alone explains them, at weight -2.
        let r = d.decode(&mut s, &[0, 1]).unwrap();
        assert_eq!((r.faults.clone(), r.weight), (vec![true, false], -2.0));
        assert_eq!(d.decode_to_edges(&mut s, &[0, 1]).unwrap(), vec![(0, 1)]);
        // Nothing fires: nothing is flipped (taking it and coming back costs nothing).
        let r = d.decode(&mut s, &[]).unwrap();
        assert_eq!((r.faults, r.weight), (vec![false, false], 0.0));
        assert!(d.decode_to_edges(&mut s, &[]).unwrap().is_empty());
    }
}

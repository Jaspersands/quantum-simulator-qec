//! PyMatching's `UserGraph` (`user_graph.cc`, PyMatching 2.4): a matching graph built by hand.
//! Nodes, edges carrying sets of fault ids, weights and error probabilities, boundary nodes, and
//! the five ways a parallel edge merges. It decodes on `pm_exact`, PyMatching's sparse blossom
//! transliterated, on the matching graph PyMatching builds from it (`UserGraph::to_mwpm`): integer
//! weights as they are, otherwise scaled so the heaviest edge is 2^24 − 1, rounded, and doubled;
//! a negative weight's edge pre-applied (its endpoints toggled, its fault ids flipped, its weight
//! counted) and matched at its absolute weight; edges to boundary nodes made boundary edges, the
//! lightest kept where several meet one node. So its answers are PyMatching's, ties included.

use std::collections::BTreeSet;
use std::sync::Arc;

use crate::dem_program::{DemInstr, DemProgram, OBS};
use crate::pm_exact::{Graph, Mwpm, NeighbourOrder, NUM_DISTINCT_WEIGHTS};

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

#[derive(Clone, Debug, Default)]
pub struct UserGraph {
    /// Per node, the ids of the edges touching it, in the order they were added.
    adjacency: Vec<Vec<usize>>,
    pub edges: Vec<UserEdge>,
    boundary: BTreeSet<usize>,
    num_fault_ids: usize,
    all_have_probabilities: bool,
}

/// The decoder of a graph as it was: PyMatching's matching graph, shared between threads; each
/// thread decodes on its own `Mwpm` over it.
#[derive(Clone, Debug)]
pub struct UserDecoder {
    graph: Arc<Graph>,
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

    /// The ids of the edges touching `node`, in the order they were added (a self-loop once).
    pub fn adjacency(&self, node: usize) -> &[usize] {
        &self.adjacency[node]
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

    /// PyMatching's graph of a detector error model (`detector_error_model_to_user_graph`): each
    /// piece of each error with probability above 0, its targets as written (repeats kept), is an
    /// edge if it has two detectors (one: to the boundary; otherwise nothing) of weight
    /// ln((1 − p)/p), parallel pieces merged as independent mechanisms.
    pub fn from_dem(dem: &DemProgram) -> Result<UserGraph, String> {
        let stats = dem.stats();
        let max = crate::dem_program::MAX_DETECTOR;
        if stats.num_detectors > max + 1 {
            return Err(format!("the model names detector {}, past the largest index, {max}", stats.num_detectors - 1));
        }
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
    pub(crate) fn normalising_constant(&self) -> Result<f64, String> {
        crate::pm_exact::normalising_constant_v2_4(self, NUM_DISTINCT_WEIGHTS)
    }

    /// The decoder for the graph as it is now (PyMatching's `to_mwpm`).
    pub fn decoder(&self) -> Result<UserDecoder, String> {
        Ok(UserDecoder { graph: Arc::new(Graph::try_from_user_graph(self, NeighbourOrder::V2_4, NUM_DISTINCT_WEIGHTS)?) })
    }
}

impl UserDecoder {
    pub fn graph(&self) -> &Arc<Graph> {
        &self.graph
    }

    /// A fresh `Mwpm` over the graph (one per thread).
    pub fn mwpm(&self) -> Mwpm {
        Mwpm::new(self.graph.clone())
    }

    pub fn num_fault_ids(&self) -> usize {
        self.graph.num_observables
    }

    pub fn num_nodes(&self) -> usize {
        self.graph.num_nodes
    }

    /// One shot, from its detection events (distinct node indices), into `out`: the fault ids
    /// flipped, bit-packed (`ceil(num_fault_ids / 8)` bytes, zeroed first). Returns the weight.
    pub fn decode_packed(&self, mwpm: &mut Mwpm, events: &[u64], out: &mut [u8]) -> Result<f64, String> {
        out.iter_mut().for_each(|b| *b = 0);
        let g = &self.graph;
        if g.num_observables <= 64 {
            let (mask, w) = mwpm.decode_to_mask(events)?;
            for (k, byte) in out.iter_mut().enumerate().take(8) {
                *byte = (mask >> (8 * k)) as u8;
            }
            Ok(w as f64 / g.normalising_constant)
        } else {
            let mut bytes = vec![0u8; g.num_observables];
            let w = mwpm.decode_into(events, &mut bytes)?;
            for (k, &b) in bytes.iter().enumerate() {
                out[k / 8] |= (b & 1) << (k % 8);
            }
            Ok(w as f64 / g.normalising_constant)
        }
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
        let mut m = d.mwpm();
        let (faults, w) = m.decode(&[1, 3]);
        assert_eq!(faults, vec![0, 1, 1, 0, 0]);
        assert_eq!(w, 2.0);
        assert!(m.try_decode(&[9]).is_err());
        let mut out = [0u8];
        assert_eq!(d.decode_packed(&mut m, &[1, 3], &mut out).unwrap(), 2.0);
        assert_eq!(out, [0b110]);
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
    fn a_model_keeps_its_targets_in_order() {
        let dem = DemProgram::parse("error(0.1) D2 D0 L1\nerror(0.2) D1 D1\nerror(0.3) D0 ^ D1 D2").unwrap();
        let g = UserGraph::from_dem(&dem).unwrap();
        assert_eq!((g.edges[0].node1, g.edges[0].node2), (2, Some(0)));
        assert_eq!((g.edges[1].node1, g.edges[1].node2), (1, Some(1)));
        assert_eq!((g.edges[2].node1, g.edges[2].node2), (0, None));
        assert_eq!((g.edges[3].node1, g.edges[3].node2), (1, Some(2)));
    }
}

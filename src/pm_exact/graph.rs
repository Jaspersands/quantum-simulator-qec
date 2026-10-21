// Copyright 2022 PyMatching Contributors
// Licensed under the Apache License, Version 2.0 (see src/pm_exact/LICENSE). Modified: transliterated
// to Rust for this crate.

//! The matching graph and its search twin, as PyMatching builds them from a `UserGraph`
//! (`driver/user_graph.{h,cc}`: `iter_discretized_edges`, `to_matching_or_search_graph_helper`,
//! `get_edge_weight_normalising_constant`; `flooder/graph.cc`; `search/search_graph.cc`), or as
//! PyMatching 2.2.2 built them from a detector error model (`driver/io.{h,cc}`,
//! `IntermediateWeightedGraph`), which is what Chromobius links. Both graphs share one
//! adjacency: PyMatching adds the same edges to each in the same order.

use std::collections::BTreeSet;

use crate::user_graph::UserGraph;

/// `pm::NUM_DISTINCT_WEIGHTS`: 2^24 weight levels.
pub const NUM_DISTINCT_WEIGHTS: u32 = 1 << 24;
/// `pm::MAX_USER_EDGE_WEIGHT`.
pub const MAX_USER_EDGE_WEIGHT: f64 = ((1u32 << 24) - 1) as f64;
/// No node, region or tree node (PyMatching's `nullptr`); as a neighbour, the boundary.
pub const NONE: u32 = u32::MAX;

/// Which PyMatching's neighbour order (and weight scale) to build.
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum NeighbourOrder {
    /// PyMatching 2.4 (`UserGraph::to_matching_graph`): edges in the graph's edge order, each
    /// node's boundary edge (the lightest, where several meet it) first. Integer weights are
    /// kept as they are; otherwise the heaviest edge is scaled to `num_distinct_weights - 1`.
    /// Weights above 2^24 − 1 are refused.
    V2_4,
    /// PyMatching 2.2.2 (`IntermediateWeightedGraph::iter_discretized_edges`): node by node, each
    /// node's entries in the order its edges were added, an edge emitted from its lower node; the
    /// heaviest edge always scaled to `num_distinct_weights - 1`. What Chromobius links.
    V2_2,
}

/// PyMatching's `MatchingGraph` and `SearchGraph` in one: per node, its neighbours (`NONE`: the
/// boundary, always first), their discretised weights (even integers) and observables, as
/// compressed rows; and what the negative-weight edges toggle.
#[derive(Clone, Debug)]
pub struct Graph {
    pub num_nodes: usize,
    /// `num_observables` (the user graph's fault ids): masks are kept only up to 64.
    pub num_observables: usize,
    /// The weights' scale, doubled: a reported weight is the integer total over this.
    pub normalising_constant: f64,
    pub num_distinct_weights: u32,
    pub order: NeighbourOrder,
    first: Vec<u32>,
    neighbors: Vec<u32>,
    weights: Vec<u32>,
    observables: Vec<u64>,
    /// Per slot, `index_of_neighbor` of the way back (first match), or `NONE` at the boundary.
    reverse: Vec<u32>,
    obs_first: Vec<u32>,
    obs_indices: Vec<u32>,
    /// Filled only when the user graph has boundary nodes.
    pub is_user_graph_boundary_node: Vec<bool>,
    /// The nodes a negative-weight edge toggles, sorted (`negative_weight_detection_events`).
    pub negative_weight_detection_events: Vec<u64>,
    /// The observables they flip, sorted.
    pub negative_weight_observables: Vec<usize>,
    pub negative_weight_obs_mask: u64,
    pub negative_weight_sum: i64,
    /// The search graph's negative-weight edges in the order added (`NONE`: the boundary).
    pub negative_weight_edges: Vec<(u32, u32)>,
}

struct Slot {
    nbr: u32,
    w: u32,
    mask: u64,
    obs: u32,
}

struct Builder {
    num_observables: usize,
    lists: Vec<Vec<Slot>>,
    boundary: Vec<Option<Slot>>,
    obs_lists: Vec<Vec<u32>>,
    neg_dets: BTreeSet<u64>,
    neg_obs: BTreeSet<usize>,
    neg_sum: i64,
    neg_edges: Vec<(u32, u32)>,
}

fn toggle<T: Ord>(set: &mut BTreeSet<T>, x: T) {
    if !set.remove(&x) {
        set.insert(x);
    }
}

impl Builder {
    fn new(n: usize, num_observables: usize) -> Builder {
        Builder {
            num_observables,
            lists: (0..n).map(|_| Vec::new()).collect(),
            boundary: (0..n).map(|_| None).collect(),
            obs_lists: Vec::new(),
            neg_dets: BTreeSet::new(),
            neg_obs: BTreeSet::new(),
            neg_sum: 0,
            neg_edges: Vec::new(),
        }
    }

    fn mask(&self, observables: &[usize]) -> u64 {
        let mut m = 0u64;
        if self.num_observables <= 64 {
            // As the hardware shifts: an id past 63 (brought by a merge) wraps.
            for &o in observables {
                m ^= 1u64.wrapping_shl(o as u32);
            }
        }
        m
    }

    fn obs_ref(&mut self, observables: &[usize]) -> u32 {
        self.obs_lists.push(observables.iter().map(|&o| o as u32).collect());
        (self.obs_lists.len() - 1) as u32
    }

    /// `MatchingGraph::add_edge` and `SearchGraph::add_edge`.
    fn add_edge(&mut self, u: usize, v: usize, w: i32, observables: &[usize]) {
        if w < 0 {
            for &o in observables {
                toggle(&mut self.neg_obs, o);
            }
            toggle(&mut self.neg_dets, u as u64);
            toggle(&mut self.neg_dets, v as u64);
            self.neg_sum += w as i64;
        }
        if u == v {
            return;
        }
        if w < 0 {
            self.neg_edges.push((u as u32, v as u32));
        }
        let mask = self.mask(observables);
        let obs = self.obs_ref(observables);
        let aw = w.unsigned_abs();
        self.lists[u].push(Slot { nbr: v as u32, w: aw, mask, obs });
        self.lists[v].push(Slot { nbr: u as u32, w: aw, mask, obs });
    }

    /// `MatchingGraph::add_boundary_edge` and `SearchGraph::add_boundary_edge`.
    fn add_boundary_edge(&mut self, u: usize, w: i32, observables: &[usize]) -> Result<(), String> {
        let mask = self.mask(observables);
        if w < 0 {
            for &o in observables {
                toggle(&mut self.neg_obs, o);
            }
            toggle(&mut self.neg_dets, u as u64);
            self.neg_sum += w as i64;
            self.neg_edges.push((u as u32, NONE));
        }
        if self.boundary[u].is_some() {
            return Err("Max one boundary edge.".into());
        }
        let obs = self.obs_ref(observables);
        self.boundary[u] = Some(Slot { nbr: NONE, w: w.unsigned_abs(), mask, obs });
        Ok(())
    }

    fn finish(self, num_nodes: usize, normalising_constant: f64, num_distinct_weights: u32, order: NeighbourOrder, is_boundary: Vec<bool>) -> Graph {
        let n = num_nodes;
        let total: usize = self.lists.iter().map(|l| l.len()).sum::<usize>() + self.boundary.iter().filter(|b| b.is_some()).count();
        let mut first = Vec::with_capacity(n + 1);
        let mut neighbors = Vec::with_capacity(total);
        let mut weights = Vec::with_capacity(total);
        let mut observables = Vec::with_capacity(total);
        let mut obs_first = Vec::with_capacity(total + 1);
        let mut obs_indices = Vec::new();
        obs_first.push(0u32);
        let mut owner = Vec::with_capacity(total);
        for (u, (b, list)) in self.boundary.iter().zip(self.lists.iter()).enumerate() {
            first.push(neighbors.len() as u32);
            for s in b.iter().chain(list.iter()) {
                neighbors.push(s.nbr);
                weights.push(s.w);
                observables.push(s.mask);
                obs_indices.extend_from_slice(&self.obs_lists[s.obs as usize]);
                obs_first.push(obs_indices.len() as u32);
                owner.push(u as u32);
            }
        }
        first.push(neighbors.len() as u32);
        // The way back along each edge: the first slot of the neighbour's that points here
        // (`index_of_neighbor`).
        let mut incoming: Vec<Vec<u32>> = vec![Vec::new(); n];
        for (s, &v) in neighbors.iter().enumerate() {
            if v != NONE {
                incoming[v as usize].push(s as u32);
            }
        }
        let mut reverse = vec![NONE; neighbors.len()];
        let mut pos = vec![NONE; n];
        for v in 0..n {
            let (a, b) = (first[v] as usize, first[v + 1] as usize);
            for k in a..b {
                let u = neighbors[k];
                if u != NONE && pos[u as usize] == NONE {
                    pos[u as usize] = (k - a) as u32;
                }
            }
            for &s in &incoming[v] {
                reverse[s as usize] = pos[owner[s as usize] as usize];
            }
            for k in a..b {
                let u = neighbors[k];
                if u != NONE {
                    pos[u as usize] = NONE;
                }
            }
        }
        let negative_weight_observables: Vec<usize> = self.neg_obs.into_iter().collect();
        let mut negative_weight_obs_mask = 0u64;
        if self.num_observables <= 64 {
            for &o in &negative_weight_observables {
                negative_weight_obs_mask ^= 1u64.wrapping_shl(o as u32);
            }
        }
        Graph {
            num_nodes: n,
            num_observables: self.num_observables,
            normalising_constant,
            num_distinct_weights,
            order,
            first,
            neighbors,
            weights,
            observables,
            reverse,
            obs_first,
            obs_indices,
            is_user_graph_boundary_node: is_boundary,
            negative_weight_detection_events: self.neg_dets.into_iter().collect(),
            negative_weight_observables,
            negative_weight_obs_mask,
            negative_weight_sum: self.neg_sum,
            negative_weight_edges: self.neg_edges,
        }
    }
}

/// `UserGraph::get_edge_weight_normalising_constant` (PyMatching 2.4), before doubling: the
/// weights a correlated load implied count too, and each must name an edge and keep its sign.
pub fn normalising_constant_v2_4(g: &UserGraph, num_distinct_weights: u32) -> Result<f64, String> {
    let mut max_abs_weight = 0f64;
    let mut all_integral_weight = true;
    for (id, e) in g.edges.iter().enumerate() {
        if e.weight.abs() > max_abs_weight {
            max_abs_weight = e.weight.abs();
        }
        if e.weight.round() != e.weight {
            all_integral_weight = false;
        }
        for implied in g.implied_weights(id) {
            if implied.weight.abs() > max_abs_weight {
                max_abs_weight = implied.weight.abs();
            }
            if implied.weight.round() != implied.weight {
                all_integral_weight = false;
            }
            let Some(other) = g.find(implied.node1, implied.node2) else {
                return Err(format!("Edge rewrite rule refers to non-existent edge ({}, {})", implied.node1, implied.node2.unwrap_or(usize::MAX)));
            };
            let same_sign = g.edges[other].weight * implied.weight >= 0.0;
            if !same_sign {
                return Err("Edge weight rewrite rules that change the sign of an edge weight are not currently supported.".into());
            }
        }
    }
    if max_abs_weight > MAX_USER_EDGE_WEIGHT {
        return Err(format!("maximum absolute edge weight of {} exceeded.", MAX_USER_EDGE_WEIGHT as u32));
    }
    if all_integral_weight {
        Ok(1.0)
    } else {
        Ok(num_distinct_weights.wrapping_sub(1) as f64 / max_abs_weight)
    }
}

/// PyMatching 2.2.2's scale: the heaviest edge always to `num_distinct_weights - 1`.
///
/// 2.2.2's `IntermediateWeightedGraph::max_abs_weight` reads every node's entries. A self-loop
/// has two entries at its node (`add_or_merge_edge` pushes one per end), and a merge into it
/// updates only the first (both of its searches find the first), so the second keeps the loop's
/// first weight and counts toward the maximum; here a merged self-loop counts with its merged
/// weight only (the user graph keeps no other). (The loop itself is never an edge of the graph:
/// `iter_discretized_edges` keeps only `j > i`.) Only a model with a self-loop given more than
/// once (`error(p) D0 D0`, then `error(q) D0 D0`) whose first weight is heavier, in absolute
/// value, than every edge and the loop's merged weight scales differently; Chromobius's Möbius
/// models have no self-loops.
fn normalising_constant_v2_2(g: &UserGraph, num_distinct_weights: u32) -> f64 {
    let mut max_abs_weight = 0f64;
    for e in &g.edges {
        if e.weight.abs() > max_abs_weight {
            max_abs_weight = e.weight.abs();
        }
    }
    num_distinct_weights.wrapping_sub(1) as f64 / max_abs_weight
}

/// `(signed_weight_int)round(weight * c) * 2`.
#[inline]
fn discretise(weight: f64, c: f64) -> i32 {
    ((weight * c).round() as i32).wrapping_mul(2)
}

impl Graph {
    /// The graph PyMatching would decode `g` on, with 2^24 weight levels. Panics where PyMatching
    /// raises (a weight above 2^24 − 1 in `V2_4`); `try_from_user_graph` returns the message.
    pub fn from_user_graph(g: &UserGraph, order: NeighbourOrder) -> Graph {
        Graph::from_user_graph_with(g, order, NUM_DISTINCT_WEIGHTS)
    }

    /// As `from_user_graph`, with `num_distinct_weights` levels (PyMatching's animation uses 1000).
    pub fn from_user_graph_with(g: &UserGraph, order: NeighbourOrder, num_distinct_weights: u32) -> Graph {
        Graph::try_from_user_graph(g, order, num_distinct_weights).unwrap_or_else(|e| panic!("{e}"))
    }

    pub fn try_from_user_graph(g: &UserGraph, order: NeighbourOrder, num_distinct_weights: u32) -> Result<Graph, String> {
        let n = g.num_nodes();
        let mut is_boundary = Vec::new();
        if !g.boundary().is_empty() {
            is_boundary = vec![false; n];
            for &b in g.boundary() {
                is_boundary[b] = true;
            }
        }
        let bnd = |x: Option<usize>| x.is_none_or(|v| g.boundary().contains(&v));
        let mut b = Builder::new(n, g.num_fault_ids());
        // The user edges that became edges of the graph, whose implied weights PyMatching converts.
        let mut in_graph: Vec<usize> = Vec::new();
        let c = match order {
            NeighbourOrder::V2_4 => {
                let c = normalising_constant_v2_4(g, num_distinct_weights)?;
                // Boundary edges are held back, the lightest per node (the first among equals),
                // then added in node order.
                let mut held: Vec<Option<(i32, usize)>> = vec![None; n];
                for (id, e) in g.edges.iter().enumerate() {
                    let w = discretise(e.weight, c);
                    let (b1, b2) = (bnd(Some(e.node1)), bnd(e.node2));
                    let at = if b2 && !b1 {
                        Some(e.node1)
                    } else if b1 && !b2 {
                        e.node2
                    } else {
                        if !b1 {
                            let v = e.node2.expect("both ends are nodes");
                            b.add_edge(e.node1, v, w, &e.fault_ids);
                            if v != e.node1 {
                                in_graph.push(id);
                            }
                        }
                        None
                    };
                    if let Some(u) = at {
                        if held[u].is_none_or(|(old, _)| old > w) {
                            held[u] = Some((w, id));
                        }
                    }
                }
                for (u, h) in held.iter().enumerate() {
                    if let Some((w, id)) = *h {
                        b.add_boundary_edge(u, w, &g.edges[id].fault_ids)?;
                        in_graph.push(id);
                    }
                }
                c
            }
            NeighbourOrder::V2_2 => {
                let c = normalising_constant_v2_2(g, num_distinct_weights);
                // Boundary nodes are not in PyMatching 2.2.2's graphs: an edge to one is taken as
                // a boundary edge, the lightest per node kept (the first among equals).
                let mut chosen: Vec<Option<(i32, usize)>> = vec![None; n];
                for (id, e) in g.edges.iter().enumerate() {
                    let (b1, b2) = (bnd(Some(e.node1)), bnd(e.node2));
                    let at = if b2 && !b1 {
                        Some(e.node1)
                    } else if b1 && !b2 {
                        e.node2
                    } else {
                        None
                    };
                    if let Some(u) = at {
                        let w = discretise(e.weight, c);
                        if chosen[u].is_none_or(|(old, _)| old > w) {
                            chosen[u] = Some((w, id));
                        }
                    }
                }
                for i in 0..n {
                    if bnd(Some(i)) {
                        continue;
                    }
                    for &id in g.adjacency(i) {
                        let e = &g.edges[id];
                        let w = discretise(e.weight, c);
                        let j = if e.node1 == i { e.node2 } else { Some(e.node1) };
                        if bnd(j) {
                            if chosen[i].is_some_and(|(_, k)| k == id) {
                                b.add_boundary_edge(i, w, &e.fault_ids)?;
                            }
                        } else if let Some(j) = j {
                            if j > i {
                                b.add_edge(i, j, w, &e.fault_ids);
                            }
                        }
                    }
                }
                c
            }
        };
        let graph = b.finish(n, 2.0 * c, num_distinct_weights, order, is_boundary);
        // `convert_implied_weights`: each implied weight of an edge in the graph must name an edge
        // in the graph (not a self-loop, nor an edge to a boundary node), or PyMatching refuses
        // to build its decoder.
        for id in in_graph {
            for r in g.implied_weights(id) {
                let target = r.node2.map_or(NONE, |v| v as u32);
                if r.node1 >= n || r.node2.is_some_and(|v| v >= n) || graph.index_of_neighbor(r.node1 as u32, target).is_none() {
                    return Err("Failed to find neighbor.".into());
                }
            }
        }
        Ok(graph)
    }

    /// A node's neighbours (`NONE`: the boundary, only ever first).
    #[inline]
    pub fn neighbors(&self, node: u32) -> &[u32] {
        &self.neighbors[self.first[node as usize] as usize..self.first[node as usize + 1] as usize]
    }

    #[inline]
    pub fn neighbor_weights(&self, node: u32) -> &[u32] {
        &self.weights[self.first[node as usize] as usize..self.first[node as usize + 1] as usize]
    }

    #[inline]
    pub fn neighbor_observables(&self, node: u32) -> &[u64] {
        &self.observables[self.first[node as usize] as usize..self.first[node as usize + 1] as usize]
    }

    /// The observables an edge crosses, as listed (the search graph's
    /// `neighbor_observable_indices`).
    #[inline]
    pub fn neighbor_observable_indices(&self, node: u32, k: usize) -> &[u32] {
        let s = self.first[node as usize] as usize + k;
        &self.obs_indices[self.obs_first[s] as usize..self.obs_first[s + 1] as usize]
    }

    #[inline]
    pub(crate) fn slot(&self, node: u32, k: usize) -> usize {
        self.first[node as usize] as usize + k
    }

    #[inline]
    pub(crate) fn nbr_at(&self, s: usize) -> u32 {
        self.neighbors[s]
    }

    #[inline]
    pub(crate) fn weight_at(&self, s: usize) -> u32 {
        self.weights[s]
    }

    #[inline]
    pub(crate) fn obs_at(&self, s: usize) -> u64 {
        self.observables[s]
    }

    #[inline]
    pub(crate) fn degree(&self, node: u32) -> usize {
        (self.first[node as usize + 1] - self.first[node as usize]) as usize
    }

    /// `index_of_neighbor` from the far end of `node`'s `k`th edge back to `node`.
    #[inline]
    pub(crate) fn reverse_index(&self, node: u32, k: usize) -> usize {
        self.reverse[self.first[node as usize] as usize + k] as usize
    }

    /// `DetectorNode::index_of_neighbor`: the first slot of `node` pointing at `target`
    /// (`NONE`: the boundary).
    pub fn index_of_neighbor(&self, node: u32, target: u32) -> Option<usize> {
        self.neighbors(node).iter().position(|&x| x == target)
    }

    pub fn num_slots(&self) -> usize {
        self.neighbors.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::user_graph::Merge;

    #[test]
    fn neighbour_orders() {
        // Edges in model order: (2, 1), (0, 1), boundary at 1, (1, 3).
        let mut g = UserGraph::new(4, 0);
        g.add_edge(2, 1, &[0], 1.5, -1.0, Merge::Disallow).unwrap();
        g.add_edge(0, 1, &[], 2.0, -1.0, Merge::Disallow).unwrap();
        g.add_boundary_edge(1, &[1], 3.0, -1.0, Merge::Disallow).unwrap();
        g.add_edge(1, 3, &[], 1.0, -1.0, Merge::Disallow).unwrap();
        let a = Graph::from_user_graph(&g, NeighbourOrder::V2_4);
        assert_eq!(a.neighbors(1), &[NONE, 2, 0, 3]);
        assert_eq!(a.neighbors(2), &[1]);
        let b = Graph::from_user_graph(&g, NeighbourOrder::V2_2);
        // Node 0 emits (0, 1) first; node 1 then emits its boundary and (1, 2), (1, 3).
        assert_eq!(b.neighbors(1), &[NONE, 0, 2, 3]);
        assert_eq!(b.neighbors(0), &[1]);
        for gr in [&a, &b] {
            for u in 0..4u32 {
                for (k, &v) in gr.neighbors(u).iter().enumerate() {
                    if v != NONE {
                        assert_eq!(gr.neighbors(v)[gr.reverse_index(u, k)], u);
                    }
                }
            }
        }
        // Not integral: the heaviest edge is 2^24 - 1, doubled.
        assert_eq!(a.neighbor_weights(1)[0], 2 * ((1u32 << 24) - 1));
        assert_eq!(a.normalising_constant, 2.0 * ((1u32 << 24) - 1) as f64 / 3.0);
        assert_eq!(a.neighbor_observables(1), &[2, 1, 0, 0]);
    }

    #[test]
    fn negative_weights_and_boundary_nodes() {
        let mut g = UserGraph::new(4, 0);
        g.add_edge(0, 1, &[0], -2.0, -1.0, Merge::Disallow).unwrap();
        g.add_edge(1, 3, &[1], 4.0, -1.0, Merge::Disallow).unwrap();
        g.add_boundary_edge(1, &[2], 5.0, -1.0, Merge::Disallow).unwrap();
        g.add_edge(2, 2, &[3], -1.0, -1.0, Merge::Disallow).unwrap();
        g.set_boundary([3].into_iter().collect());
        let a = Graph::from_user_graph(&g, NeighbourOrder::V2_4);
        // Integral: weights doubled. Node 3 is a boundary node: (1, 3) is 1's boundary edge, the
        // lighter of the two.
        assert_eq!(a.neighbors(1), &[NONE, 0]);
        assert_eq!(a.neighbor_weights(1), &[8, 4]);
        assert_eq!(a.negative_weight_detection_events, vec![0, 1]);
        assert_eq!(a.negative_weight_observables, vec![0, 3]);
        assert_eq!(a.negative_weight_obs_mask, 0b1001);
        assert_eq!(a.negative_weight_sum, -6);
        assert_eq!(a.negative_weight_edges, vec![(0, 1)]);
        assert_eq!(a.is_user_graph_boundary_node, vec![false, false, false, true]);
        assert!(Graph::try_from_user_graph(&{
            let mut h = UserGraph::new(2, 0);
            h.add_edge(0, 1, &[], 2e7, -1.0, Merge::Disallow).unwrap();
            h
        }, NeighbourOrder::V2_4, NUM_DISTINCT_WEIGHTS)
        .is_err());
    }
}

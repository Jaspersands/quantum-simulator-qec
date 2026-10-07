//! The detector graph the regions grow on.

use crate::dem_decoder::edge_weight;

use super::state::{Scratch, BOUNDARY, NONE};

/// Adjacency in compressed rows. The boundary is not a node: an edge to it has
/// target `BOUNDARY`. Every undirected edge also has an id, its index in the
/// sorted merged edges it was built from, which names both of its half-edges.
pub struct SparseGraph {
    pub(crate) num_nodes: usize,
    offsets: Vec<u32>,
    pub(crate) to: Vec<u32>,
    /// `to`, with the boundary as `num_nodes`: the node a scan reads, past the graph's last.
    pub(crate) reach: Vec<u32>,
    /// Even integer weights, from `int_weight`.
    pub(crate) w: Vec<i64>,
    pub(crate) obs: Vec<u64>,
    /// Per half-edge, the id of its edge.
    pub(crate) edge_of: Vec<u32>,
    /// Per edge, its half-edges; the second is NONE for a boundary edge.
    pub(crate) halves: Vec<[u32; 2]>,
    /// Per edge, its endpoints, with `num_nodes` standing for the boundary.
    pub(crate) ends: Vec<(u32, u32)>,
    /// Per edge, its weight ln((1 − p)/p) as a float, which the dense
    /// reference matcher sums for the weight it reports.
    wf: Vec<f64>,
}

impl SparseGraph {
    /// From merged edges `(u, v, p, observables)`; `v == num_nodes` is the boundary.
    pub fn from_edges(num_nodes: usize, edges: &[(u32, u32, f64, u64)]) -> SparseGraph {
        let mut degree = vec![0u32; num_nodes];
        for &(u, v, _, _) in edges {
            degree[u as usize] += 1;
            if (v as usize) < num_nodes {
                degree[v as usize] += 1;
            }
        }
        let mut offsets = vec![0u32; num_nodes + 1];
        for i in 0..num_nodes {
            offsets[i + 1] = offsets[i] + degree[i];
        }
        let m = offsets[num_nodes] as usize;
        let mut to = vec![0u32; m];
        let mut w = vec![0i64; m];
        let mut obs = vec![0u64; m];
        let mut edge_of = vec![0u32; m];
        let mut halves = vec![[NONE, NONE]; edges.len()];
        let mut ends = Vec::with_capacity(edges.len());
        let mut wf = Vec::with_capacity(edges.len());
        let mut fill: Vec<u32> = offsets[..num_nodes].to_vec();
        for (i, &(u, v, p, o)) in edges.iter().enumerate() {
            let (fw, iw) = edge_weight(p);
            wf.push(fw);
            let boundary = (v as usize) >= num_nodes;
            let pairs: &[(u32, u32)] = if boundary { &[(u, BOUNDARY)] } else { &[(u, v), (v, u)] };
            for (k, &(a, b)) in pairs.iter().enumerate() {
                let slot = fill[a as usize] as usize;
                fill[a as usize] += 1;
                to[slot] = b;
                w[slot] = iw;
                obs[slot] = o;
                edge_of[slot] = i as u32;
                halves[i][k] = slot as u32;
            }
            ends.push((u, if boundary { num_nodes as u32 } else { v }));
        }
        let reach = to.iter().map(|&b| if b == BOUNDARY { num_nodes as u32 } else { b }).collect();
        SparseGraph { num_nodes, offsets, to, reach, w, obs, edge_of, halves, ends, wf }
    }

    pub(crate) fn edges(&self, v: u32) -> std::ops::Range<usize> {
        self.offsets[v as usize] as usize..self.offsets[v as usize + 1] as usize
    }

    /// Node `v`'s edges in order, each as (the node across it, or None for the
    /// boundary; its integer weight; its observables; its float weight).
    pub(crate) fn arcs(&self, v: u32) -> impl Iterator<Item = (Option<u32>, i64, u64, f64)> + '_ {
        self.edges(v).map(move |e| {
            let to = self.to[e];
            (if to == BOUNDARY { None } else { Some(to) }, self.w[e], self.obs[e], self.wf[self.edge_of[e] as usize])
        })
    }

    pub fn num_edges(&self) -> usize {
        self.halves.len()
    }

    /// The id of the edge between `u` and `v`, where `num_nodes` in either
    /// place stands for the boundary.
    pub fn edge_id(&self, u: u32, v: u32) -> Option<u32> {
        let n = self.num_nodes as u32;
        let (u, v) = if u == n { (v, u) } else { (u, v) };
        if u >= n || v > n {
            return None;
        }
        let target = if v == n { BOUNDARY } else { v };
        self.edges(u).find(|&e| self.to[e] == target).map(|e| self.edge_of[e])
    }

    /// An edge's endpoints, with `num_nodes` standing for the boundary.
    pub fn edge_ends(&self, id: u32) -> (u32, u32) {
        self.ends[id as usize]
    }

    /// An edge's integer weight as built, before any reweighting.
    #[cfg(test)]
    pub(crate) fn weight_of(&self, id: u32) -> i64 {
        self.w[self.halves[id as usize][0] as usize]
    }

    /// A `Scratch` serves the graph it was built for, and comes back with its
    /// weights restored.
    pub(crate) fn check_scratch(&self, scratch: &Scratch) {
        assert!(
            scratch.nodes.len() == self.num_nodes + 1 && scratch.w.len() == self.w.len(),
            "a Scratch serves the graph it was built for"
        );
        debug_assert!(scratch.undo.is_empty(), "a Scratch's weights are restored after every decode");
    }
}

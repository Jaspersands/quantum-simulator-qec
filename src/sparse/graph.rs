//! The detector graph the regions grow on.

use crate::dem_decoder::int_weight;

use super::state::BOUNDARY;

/// Adjacency in compressed rows. The boundary is not a node: an edge to it has
/// target `BOUNDARY`.
pub struct SparseGraph {
    pub(crate) num_nodes: usize,
    offsets: Vec<u32>,
    pub(crate) to: Vec<u32>,
    /// Even integer weights, from `int_weight`.
    pub(crate) w: Vec<i64>,
    pub(crate) obs: Vec<u64>,
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
        let mut fill: Vec<u32> = offsets[..num_nodes].to_vec();
        for &(u, v, p, o) in edges {
            let iw = int_weight(((1.0 - p) / p).ln());
            let boundary = (v as usize) >= num_nodes;
            let ends: &[(u32, u32)] = if boundary { &[(u, BOUNDARY)] } else { &[(u, v), (v, u)] };
            for &(a, b) in ends {
                let slot = fill[a as usize] as usize;
                fill[a as usize] += 1;
                to[slot] = b;
                w[slot] = iw;
                obs[slot] = o;
            }
        }
        SparseGraph { num_nodes, offsets, to, w, obs }
    }

    pub(crate) fn edges(&self, v: u32) -> std::ops::Range<usize> {
        self.offsets[v as usize] as usize..self.offsets[v as usize + 1] as usize
    }
}

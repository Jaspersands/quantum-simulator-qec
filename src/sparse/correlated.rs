//! Correlated matching: two passes of the sparse matcher, the second on weights
//! lowered by what the first found.
//!
//! WHY
//! ---
//! One fault in a circuit can set off detectors in both halves of the detector
//! graph: a Y error is an X error and a Z error at once. The error model knows,
//! and writes such a fault as pieces, `D0 D1 ^ D7 D8`, one per half. Plain
//! matching sees only the pieces, as independent edges, and forgets that they
//! come together. Correlated matching remembers. It matches once, finds the
//! edges that matching used, and for each makes the edges that come with it
//! cheaper, by the probability they fired given that it did. Then it matches
//! again.
//!
//! This follows PyMatching 2.4 (`enable_correlations=True`) exactly, as read
//! from its source, so that PyMatching serves as the oracle:
//! - joint probabilities: for an error of probability p with pieces c₀ … c_k,
//!   every pair of pieces takes p, combined as independent events, and so does
//!   every piece alone, which gives its marginal;
//! - the rule for edge c and another edge a: p_a = min(0.5, joint(c, a) /
//!   marginal(c)), with weight ln((1 − p_a)/p_a) in the shared even integers;
//! - decoding: match, trace one shortest path per matched pair (`paths`), XOR
//!   the paths' edges, lower each edge a rule names to the smaller of its
//!   weight and the rule's, match again, and restore.

use std::collections::HashMap;

use crate::dem::{xor_prob, Dem};
use crate::dem_decoder::edge_weight;

use super::graph::SparseGraph;

/// Every edge's rules, in compressed rows: for edge c, the edges it makes
/// cheaper and the weights it implies for them.
pub struct Correlations {
    start: Vec<u32>,
    pub(crate) affected: Vec<u32>,
    pub(crate) weight: Vec<i64>,
    /// The implied probabilities, kept for the tests and for reading.
    pub(crate) prob: Vec<f64>,
}

impl Correlations {
    pub fn from_dem(dem: &Dem, graph: &SparseGraph) -> Result<Correlations, String> {
        let n = graph.num_edges();
        let boundary = graph.num_nodes as u32;
        let mut marginal = vec![0.0f64; n];
        let mut joint: HashMap<(u32, u32), f64> = HashMap::new();
        let mut ids: Vec<u32> = Vec::new();
        for m in &dem.mechanisms {
            // PyMatching skips an error that cannot happen.
            if m.p == 0.0 {
                continue;
            }
            ids.clear();
            for piece in &m.pieces {
                let (u, v) = match piece.detectors.as_slice() {
                    [a] => (*a, boundary),
                    [a, b] => (*a.min(b), *a.max(b)),
                    other => return Err(format!("piece with {} detectors cannot be an edge", other.len())),
                };
                ids.push(graph.edge_id(u, v).ok_or_else(|| format!("piece ({u}, {v}) is not an edge of the graph"))?);
            }
            if ids.len() > 1 {
                for k0 in 0..ids.len() {
                    for k1 in k0 + 1..ids.len() {
                        let (a, b) = (ids[k0], ids[k1]);
                        if a == b {
                            // PyMatching's two entries are then one, the
                            // marginal, and it takes p twice.
                            let e = &mut marginal[a as usize];
                            *e = xor_prob(xor_prob(*e, m.p), m.p);
                        } else {
                            for key in [(a, b), (b, a)] {
                                let e = joint.entry(key).or_insert(0.0);
                                *e = xor_prob(*e, m.p);
                            }
                        }
                    }
                }
            }
            for &c in &ids {
                let e = &mut marginal[c as usize];
                *e = xor_prob(*e, m.p);
            }
        }
        let mut rules: Vec<(u32, u32, f64)> = joint
            .into_iter()
            .filter(|&((c, _), _)| marginal[c as usize] > 0.0)
            .map(|((c, a), p)| (c, a, (p / marginal[c as usize]).min(0.5)))
            .collect();
        rules.sort_unstable_by(|x, y| (x.0, x.1).cmp(&(y.0, y.1)));
        let mut start = vec![0u32; n + 1];
        for &(c, _, _) in &rules {
            start[c as usize + 1] += 1;
        }
        for i in 0..n {
            start[i + 1] += start[i];
        }
        Ok(Correlations {
            start,
            affected: rules.iter().map(|r| r.1).collect(),
            weight: rules.iter().map(|r| edge_weight(r.2).1).collect(),
            prob: rules.iter().map(|r| r.2).collect(),
        })
    }

    pub(crate) fn rules(&self, c: u32) -> std::ops::Range<usize> {
        self.start[c as usize] as usize..self.start[c as usize + 1] as usize
    }

    /// Edge c's rules as (affected edge, implied probability, implied weight).
    pub fn rules_of(&self, c: u32) -> Vec<(u32, f64, i64)> {
        self.rules(c).map(|r| (self.affected[r], self.prob[r], self.weight[r])).collect()
    }

    pub fn num_rules(&self) -> usize {
        self.affected.len()
    }

    pub(crate) fn num_edges(&self) -> usize {
        self.start.len() - 1
    }
}

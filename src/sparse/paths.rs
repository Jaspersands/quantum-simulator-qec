//! The edges a matching uses: for each matched pair, one shortest path on the
//! detector graph, all XORed together, as PyMatching's `decode_to_edges_array`
//! gives them. An edge two paths share cancels. Correlated matching reweights
//! from this set.
//!
//! When two shortest paths join a pair, which one is traced is a tie-break,
//! and it may differ from PyMatching's. Either way the set is a minimum-weight
//! correction of the shot.

use crate::dem_decoder::{DecodeError, Prediction};

use super::state::{Scratch, BOUNDARY, NONE, NO_TIME};
use super::Solver;

fn flip(s: &mut Scratch, id: u32) {
    let f = &mut s.flipped[id as usize];
    *f = !*f;
    if *f {
        s.edge_set.push(id);
    }
}

fn clear_search(s: &mut Scratch) {
    for &v in &s.seen {
        s.dist[v as usize] = NO_TIME;
    }
    s.seen.clear();
    s.heap.clear();
}

impl<'a> Solver<'a> {
    /// Match, then trace the matching's edges into `s.edge_set`.
    pub(crate) fn pass_one(&mut self, defects: &[u32]) -> Result<Prediction, DecodeError> {
        self.reset();
        self.run(defects, false)?;
        let prediction = self.extract();
        self.trace_pairs()?;
        Ok(prediction)
    }

    /// One shortest path per matched pair of the last extraction, on the
    /// current weights, XORed into `s.edge_set`.
    pub(crate) fn trace_pairs(&mut self) -> Result<(), DecodeError> {
        self.s.ensure_paths(self.g);
        self.s.edge_set.clear();
        for i in 0..self.s.pairs.len() {
            let (a, b) = self.s.pairs[i];
            if let Err(e) = self.trace(a, b) {
                // Leave no flag raised for the next shot.
                let s = &mut *self.s;
                for &id in &s.edge_set {
                    s.flipped[id as usize] = false;
                }
                s.edge_set.clear();
                return Err(e);
            }
        }
        // Keep each edge that ended flipped, once, and lower every flag.
        let s = &mut *self.s;
        let flipped = &mut s.flipped;
        s.edge_set
            .retain(|&e| std::mem::replace(&mut flipped[e as usize], false));
        Ok(())
    }

    /// Flip the edges of one shortest path from defect `a` to `b`, a defect or
    /// BOUNDARY: Dijkstra from `a`, stopping once nothing can come closer.
    fn trace(&mut self, a: u32, b: u32) -> Result<(), DecodeError> {
        let g = self.g;
        let s = &mut *self.s;
        s.dist[a as usize] = 0;
        s.seen.push(a);
        s.heap.push(0, a);
        // The best way out to the boundary so far: distance, node, edge.
        let mut exit = (NO_TIME, NONE, NONE);
        while let Some((d, u)) = s.heap.pop() {
            if d > s.dist[u as usize] {
                continue;
            }
            if u == b || (b == BOUNDARY && exit.0 <= d) {
                break;
            }
            for e in g.edges(u) {
                let v = g.to[e];
                let nd = d + s.w[e];
                if v == BOUNDARY {
                    if nd < exit.0 {
                        exit = (nd, u, g.edge_of[e]);
                    }
                    continue;
                }
                if nd < s.dist[v as usize] {
                    if s.dist[v as usize] == NO_TIME {
                        s.seen.push(v);
                    }
                    s.dist[v as usize] = nd;
                    s.pred[v as usize] = (u, g.edge_of[e]);
                    s.heap.push(nd, v);
                }
            }
        }
        let mut v = if b == BOUNDARY {
            if exit.1 == NONE {
                clear_search(s);
                return Err(DecodeError::MatcherDeclined);
            }
            flip(s, exit.2);
            exit.1
        } else {
            if s.dist[b as usize] == NO_TIME {
                clear_search(s);
                return Err(DecodeError::MatcherDeclined);
            }
            b
        };
        while v != a {
            let (u, id) = s.pred[v as usize];
            flip(s, id);
            v = u;
        }
        clear_search(s);
        Ok(())
    }
}

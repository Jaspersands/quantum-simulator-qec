//! Exact minimum-weight perfect matching on the detector graph, by sparse
//! blossom: the algorithm of Higgott and Gidney (arXiv:2303.15933), which is
//! the core of PyMatching 2.
//!
//! WHY THIS EXISTS
//! ---------------
//! The dense decoder in `dem_decoder` runs Dijkstra from every defect and an
//! O(n³) blossom over the complete graph of defects. That is exact and simple,
//! and it stops at 256 defects. Google's d = 7 memory experiments run for 250
//! rounds: about 12,000 detectors and on the order of a thousand defects a shot.
//!
//! Sparse blossom never builds the complete graph. Every defect grows a region
//! directly on the detector graph, all at the same rate, and a region's radius
//! is its dual variable. Regions that touch are handled by Edmonds' alternating
//! trees and blossoms, but over regions instead of vertices, and a collision is
//! found by the growth itself instead of by a precomputed distance. Nothing is
//! explored beyond where regions actually reach, and at useful error rates they
//! reach only their neighbours.
//!
//! The module follows the paper's own seams: `graph` (the detector graph, with
//! even-integer weights), `state` (regions, tree nodes, detector nodes),
//! `tracker` (the reminder queue), `flooder` (growth and the events it raises),
//! `matcher` (the seven cases an event can be) and `extract` (the matching).

mod correlated;
#[cfg(test)]
mod correlated_tests;
mod extract;
mod flooder;
mod graph;
mod matcher;
mod paths;
mod state;
#[cfg(test)]
mod tests;
mod tracker;
mod union_find;

pub use correlated::Correlations;
pub(crate) use correlated::FaultEdges;
pub use graph::SparseGraph;
pub use state::Scratch;
pub use union_find::UfScratch;

use crate::dem_decoder::{DecodeError, Prediction};

/// A graph and the mutable state of one decode.
pub(crate) struct Solver<'a> {
    pub(crate) g: &'a SparseGraph,
    pub(crate) s: &'a mut Scratch,
}

impl SparseGraph {
    /// Decode one shot. `defects` must be sorted and distinct (checked in debug
    /// builds), and `scratch` built for this graph.
    pub fn decode(
        &self,
        scratch: &mut Scratch,
        defects: &[u32],
    ) -> Result<Prediction, DecodeError> {
        self.check_scratch(scratch);
        let mut solver = Solver {
            g: self,
            s: scratch,
        };
        solver.reset();
        solver.run(defects, false)?;
        Ok(solver.extract())
    }

    /// Decode one shot on weights given for this shot alone, `edge_w[id]` for
    /// each edge (even integers, as `int_weight` makes them). The scratch's
    /// weights are the graph's again afterwards.
    pub fn decode_with_weights(
        &self,
        scratch: &mut Scratch,
        defects: &[u32],
        edge_w: &[i64],
    ) -> Result<Prediction, DecodeError> {
        self.check_scratch(scratch);
        assert_eq!(edge_w.len(), self.num_edges(), "one weight per edge");
        for (halves, &wt) in self.halves.iter().zip(edge_w) {
            for &slot in halves.iter().filter(|&&h| h != state::NONE) {
                scratch.w[slot as usize] = wt;
            }
        }
        let mut solver = Solver {
            g: self,
            s: scratch,
        };
        solver.reset();
        let result = solver.run(defects, false).map(|()| solver.extract());
        scratch.w.copy_from_slice(&self.w);
        result
    }

    /// The edges one shot's matching uses (see `paths`), as endpoints with
    /// `num_nodes` standing for the boundary.
    pub fn decode_to_edges(
        &self,
        scratch: &mut Scratch,
        defects: &[u32],
    ) -> Result<Vec<(u32, u32)>, DecodeError> {
        self.check_scratch(scratch);
        let mut solver = Solver {
            g: self,
            s: scratch,
        };
        solver.pass_one(defects)?;
        Ok(solver
            .s
            .edge_set
            .iter()
            .map(|&id| self.ends[id as usize])
            .collect())
    }

    /// Pass one's edge set, as edge ids: what a window decoder commits from.
    pub fn decode_edge_ids(
        &self,
        scratch: &mut Scratch,
        defects: &[u32],
    ) -> Result<Vec<u32>, DecodeError> {
        self.check_scratch(scratch);
        let mut solver = Solver {
            g: self,
            s: scratch,
        };
        solver.pass_one(defects)?;
        Ok(solver.s.edge_set.clone())
    }

    /// Correlated matching's edge set, as edge ids: pass two's matching traced
    /// on the lowered weights.
    pub fn decode_correlated_edge_ids(
        &self,
        corr: &Correlations,
        scratch: &mut Scratch,
        defects: &[u32],
    ) -> Result<Vec<u32>, DecodeError> {
        self.check_scratch(scratch);
        assert_eq!(
            corr.num_edges(),
            self.num_edges(),
            "Correlations serve the graph they were built for"
        );
        let mut solver = Solver {
            g: self,
            s: scratch,
        };
        solver.pass_one(defects)?;
        solver.pass_two_edges(corr, defects)?;
        Ok(solver.s.edge_set.clone())
    }

    /// Correlated matching (see `correlated`): match, lower the weights the
    /// used edges' rules name, match again. `corr` must be this graph's.
    pub fn decode_correlated(
        &self,
        corr: &Correlations,
        scratch: &mut Scratch,
        defects: &[u32],
    ) -> Result<Prediction, DecodeError> {
        self.check_scratch(scratch);
        assert_eq!(
            corr.num_edges(),
            self.num_edges(),
            "Correlations serve the graph they were built for"
        );
        let mut solver = Solver {
            g: self,
            s: scratch,
        };
        solver.pass_one(defects)?;
        solver.pass_two(corr, defects)
    }

    /// Pass two alone, from an edge set given by edge id: how PyMatching's own
    /// first pass is fed into ours, to tell a tie in the traced paths from a bug.
    pub fn decode_pass2(
        &self,
        corr: &Correlations,
        scratch: &mut Scratch,
        defects: &[u32],
        edge_ids: &[u32],
    ) -> Result<Prediction, DecodeError> {
        self.check_scratch(scratch);
        assert_eq!(
            corr.num_edges(),
            self.num_edges(),
            "Correlations serve the graph they were built for"
        );
        let mut solver = Solver {
            g: self,
            s: scratch,
        };
        solver.s.edge_set.clear();
        solver.s.edge_set.extend_from_slice(edge_ids);
        solver.pass_two(corr, defects)
    }

    /// The same, checking the dual's feasibility after every event.
    #[cfg(test)]
    pub(crate) fn decode_checked(
        &self,
        scratch: &mut Scratch,
        defects: &[u32],
    ) -> Result<Prediction, DecodeError> {
        self.check_scratch(scratch);
        let mut solver = Solver {
            g: self,
            s: scratch,
        };
        solver.reset();
        solver.run(defects, true)?;
        Ok(solver.extract())
    }
}

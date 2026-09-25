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

mod extract;
mod flooder;
mod graph;
mod matcher;
mod state;
mod tracker;
#[cfg(test)]
mod tests;

pub use graph::SparseGraph;
pub use state::Scratch;

use crate::dem_decoder::{DecodeError, Prediction};

/// A graph and the mutable state of one decode.
pub(crate) struct Solver<'a> {
    pub(crate) g: &'a SparseGraph,
    pub(crate) s: &'a mut Scratch,
}

impl SparseGraph {
    /// Decode one shot. `defects` must be sorted and distinct (checked in debug
    /// builds), and `scratch` built for this graph.
    pub fn decode(&self, scratch: &mut Scratch, defects: &[u32]) -> Result<Prediction, DecodeError> {
        self.check_scratch(scratch);
        let mut solver = Solver { g: self, s: scratch };
        solver.reset();
        solver.run(defects, false)?;
        Ok(solver.extract())
    }

    /// The same, checking the dual's feasibility after every event.
    #[cfg(test)]
    pub(crate) fn decode_checked(&self, scratch: &mut Scratch, defects: &[u32]) -> Result<Prediction, DecodeError> {
        self.check_scratch(scratch);
        let mut solver = Solver { g: self, s: scratch };
        solver.reset();
        solver.run(defects, true)?;
        Ok(solver.extract())
    }
}

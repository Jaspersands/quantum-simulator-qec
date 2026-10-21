// Copyright 2022 PyMatching Contributors
// Licensed under the Apache License, Version 2.0 (see src/pm_exact/LICENSE). Modified: transliterated
// to Rust for this crate.

//! PyMatching 2.4.0's sparse blossom (`src/pymatching/sparse_blossom/`), line for line: the
//! same graph in the same neighbour order, the same radix-heap queue with its last-in-first-out
//! ties, the same flooder, alternating trees and blossom cycles, the same extraction detection
//! event by detection event, and the same two-ended path search and edge cancellation, so the
//! answers are PyMatching's to the edge and to the order: `decode_to_edges` lists PyMatching's
//! edges in PyMatching's order, `decode_to_match_edges` its pairs in its order, and `decode` its
//! observables and weight. The PyMatching API's `Matching` (`src/user_graph.rs`) decodes here.
//! The package's own decoders (`src/sparse`) are untouched.
//!
//! Files: `tracker` (time, the queue, event trackers), `graph` (the matching and search graph
//! built from a `UserGraph`, in PyMatching 2.4's neighbour order or 2.2.2's), `flooder`
//! (`graph_flooder.cc`, nodes and regions), `matcher` (`mwpm.cc`, `alternating_tree.cc`),
//! `search` (`search_flooder.cc`), `decode` (`mwpm_decoding.cc`, the step-by-step run),
//! `diagram` (`diagram/mwpm_diagram.cc`, `animate`'s frames drawn from that run).
//!
//! Pointers are `u32` indices, `NONE` for null: nodes index the graph; regions and
//! alternating-tree nodes index arenas that reuse the most recently freed slot first, as
//! PyMatching's `Arena` does, so a stale queue entry meets the region PyMatching's would.
//!
//! # The step-by-step run (for drawing, as PyMatching's `animate`)
//!
//! `Mwpm::step` advances the run one action at a time, exactly as
//! `write_animated_decoding_svg_frames` does (dequeue, discard or process, hand to the matcher),
//! and reports it as an `MwpmEvent`:
//!
//! - `Dequeued(ev)`: a flood-check event left the queue and the clock (`cur_time`) moved to its
//!   time; nothing else changed. This is where PyMatching draws its in-between growth frames,
//!   by setting the clock back (`set_cur_time`) to each frame's time and restoring it after.
//! - `Discarded(ev)`: the event was stale; its tracker dropped it (or requeued it later).
//! - `Flooded(ev)`: the flooder processed it (a region grew onto a node or shrank off one, or
//!   rescheduled) with nothing for the matcher.
//! - `Found(m)`: the flooder found a `MatcherEvent` (`RegionHitRegion`, `RegionHitBoundary`,
//!   `BlossomShatter`); the state is still as before the matcher acts. PyMatching draws its
//!   `frames_per_mwpm_event` frames here, with `m` as the focused event.
//! - `Handled(m)`: the matcher handled it.
//!
//! `step` returns `None` once the queue is dry and nothing is pending. Start with `begin(dets)`
//! (the decoder's set-up) or `begin_animation(dets)` (the animation's: a region at every
//! listed node, nothing else), stop when `all_frozen(dets)` (the animation's `check_if_done`) or
//! at `None`, then `finish()` and `extract_match_edges(dets)` for the decoder's answer.
//!
//! PyMatching's animation does not stop at a dry queue. Its `dequeue` then returns a
//! `NO_FLOOD_CHECK_EVENT` with the clock unchanged (`radix_heap_queue.h:124-125`),
//! `dequeue_decision` accepts it, and `process_tentative_event_returning_mwpm_event` reaches its
//! default case and throws "Unknown tentative event type." (`mwpm_diagram.cc:477-499`,
//! `graph_flooder.cc:254-267, 353-363`). So `None` before `all_frozen` (a detection event that
//! can reach neither a partner nor the boundary) is where `pymatching animate` prints that
//! message to stderr and stops: the frames before it are written, the end frames are not, and
//! `pymatching.cli` returns 1 (its `pymatching` console script drops that and exits 0). An
//! animation that copies PyMatching's fails there with that message, or says why it differs.
//!
//! The state to draw at any step, all read-only:
//!
//! - the graph: `graph()` (`neighbors`, `neighbor_weights`, `NONE` for the boundary, first);
//! - each node: `node(n)` (`region_that_arrived`, `region_that_arrived_top`,
//!   `reached_from_source` (a node is a detection event when it reached itself),
//!   `radius_of_arrival`, `wrapped_radius_cached`), `local_radius(n)`,
//!   `compute_local_radius_at_time_bounded_by_region`,
//!   `compute_stitch_radius_at_time_bounded_by_region_towards_neighbor`;
//! - each region: `live_regions()`, `region(r)` (`radius` as a `Varying`: growing, shrinking or
//!   frozen; `blossom_parent`, `blossom_parent_top`, `blossom_children` with the cycle's edges,
//!   `shell_area`, `matched` (`region == NONE` with a `loc_from`: the boundary),
//!   `alt_tree_node`), `total_area(r)`, `region_is_within` (PyMatching's region `<=`),
//!   `tree_equal` (its region `==`, structural);
//! - the alternating trees: `live_alt_tree_nodes()`, `alt_tree_node(a)` (`inner_region`,
//!   `outer_region`, `inner_to_outer_edge`, `parent`, `children`);
//! - pending collisions: `next_event_at_node(n)` (neighbour index and time, as PyMatching's
//!   `find_next_event_at_node_returning_neighbor_index_and_time`), `queued_events()`;
//! - the clock: `cur_time()`, `set_cur_time(t)`.

mod decode;
mod diagram;
mod flooder;
mod graph;
mod matcher;
mod search;
mod tracker;

use std::sync::Arc;

pub use decode::{NEGATIVE_WEIGHTS_REFUSED, NO_PERFECT_MATCHING};
pub use diagram::{pick_coords_for_drawing, write_animated_decoding_svg_frames, write_decoder_state_as_svg, AnimationError, FrameSchedule, Point, UNKNOWN_TENTATIVE_EVENT};
pub use flooder::{Arena, CompressedEdge, DetectorNode, GraphFillRegion, Match, MatcherEvent, RegionEdge};
pub use graph::{normalising_constant_v2_4, Graph, NeighbourOrder, MAX_USER_EDGE_WEIGHT, NONE, NUM_DISTINCT_WEIGHTS};
pub use matcher::{AltTreeEdge, AltTreeNode, MatchingResult};
pub use search::{SearchDetectorNode, SearchFlooder, TargetType, FLIPPED};
pub use tracker::{cyclic_gt, cyclic_lt, widen_from_nearby_reference, FloodEvent, QueuedEventTracker, RadixHeapQueue, Varying};

/// One step of the run (see the module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MwpmEvent {
    Dequeued(FloodEvent),
    Discarded(FloodEvent),
    Flooded(FloodEvent),
    Found(MatcherEvent),
    Handled(MatcherEvent),
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum Pending {
    #[default]
    None,
    Flood(FloodEvent),
    Matcher(MatcherEvent),
}

/// PyMatching's `Mwpm`: the flooder, the alternating trees and the search flooder over one
/// shared graph. Decoding leaves it ready for the next shot; each thread keeps its own.
#[derive(Clone, Debug)]
pub struct Mwpm {
    g: Arc<Graph>,
    nodes: Vec<DetectorNode>,
    queue: RadixHeapQueue,
    regions: Arena<GraphFillRegion>,
    alt: Arena<AltTreeNode>,
    match_edges: Vec<CompressedEdge>,
    search: SearchFlooder,
    area: Vec<u32>,
    stack: Vec<u32>,
    stack2: Vec<u32>,
    pending: Pending,
}

impl Mwpm {
    /// A decoder on `graph` (a `Graph` or an `Arc<Graph>` shared between threads).
    pub fn new(graph: impl Into<Arc<Graph>>) -> Mwpm {
        let g: Arc<Graph> = graph.into();
        let search = SearchFlooder::new(&g);
        Mwpm {
            nodes: vec![DetectorNode::default(); g.num_nodes],
            g,
            queue: RadixHeapQueue::default(),
            regions: Arena::default(),
            alt: Arena::default(),
            match_edges: Vec::new(),
            search,
            area: Vec::new(),
            stack: Vec::new(),
            stack2: Vec::new(),
            pending: Pending::None,
        }
    }

    pub fn graph(&self) -> &Graph {
        &self.g
    }

    pub fn graph_arc(&self) -> Arc<Graph> {
        self.g.clone()
    }

    pub fn node(&self, n: u32) -> &DetectorNode {
        &self.nodes[n as usize]
    }

    pub fn region(&self, r: u32) -> &GraphFillRegion {
        &self.regions.items[r as usize]
    }

    pub fn alt_tree_node(&self, a: u32) -> &AltTreeNode {
        &self.alt.items[a as usize]
    }

    /// The regions in use, by slot.
    pub fn live_regions(&self) -> Vec<u32> {
        self.regions.live_slots()
    }

    /// The alternating-tree nodes in use, by slot.
    pub fn live_alt_tree_nodes(&self) -> Vec<u32> {
        self.alt.live_slots()
    }

    pub fn cur_time(&self) -> i64 {
        self.queue.cur_time
    }

    /// Moves the clock (PyMatching's animation draws in-between frames this way, then restores it).
    pub fn set_cur_time(&mut self, t: i64) {
        self.queue.cur_time = t;
    }

    /// A node's local radius (`DetectorNode::local_radius`).
    pub fn local_radius(&self, n: u32) -> Varying {
        flooder::local_radius(&self.nodes, &self.regions.items, n)
    }

    /// The node's next event: (neighbour index, time), or `None`.
    pub fn next_event_at_node(&self, n: u32) -> Option<(usize, i64)> {
        let (k, t) = self.find_next_event_at_node(n);
        (k != usize::MAX).then_some((k, t))
    }

    /// The nodes of a region and its descendants, in `do_op_for_each_node_in_total_area`'s order.
    pub fn total_area(&self, r: u32) -> Vec<u32> {
        let (mut out, mut stack) = (Vec::new(), Vec::new());
        flooder::total_area(&self.regions.items, r, &mut out, &mut stack);
        out
    }

    /// The match edges of the last extraction.
    pub fn match_edges(&self) -> &[CompressedEdge] {
        &self.match_edges
    }

    pub fn search_flooder(&self) -> &SearchFlooder {
        &self.search
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::user_graph::{Merge, UserGraph};

    fn repetition(n: usize, w: f64) -> UserGraph {
        let mut g = UserGraph::new(n, 0);
        g.add_boundary_edge(0, &[0], w, -1.0, Merge::Disallow).unwrap();
        for k in 0..n - 1 {
            g.add_edge(k, k + 1, &[k + 1], w, -1.0, Merge::Disallow).unwrap();
        }
        g.add_boundary_edge(n - 1, &[n], w, -1.0, Merge::Disallow).unwrap();
        g
    }

    #[test]
    fn a_repetition_code() {
        let mut m = Mwpm::new(Graph::from_user_graph(&repetition(9, 1.0), NeighbourOrder::V2_4));
        let (obs, w) = m.decode(&[1, 3]);
        assert_eq!(w, 2.0);
        assert_eq!(obs, vec![0, 0, 1, 1, 0, 0, 0, 0, 0, 0]);
        assert_eq!(m.decode_to_match_edges(&[1, 3]), vec![(1, 3)]);
        // PyMatching's answers (2.4.0): the path traced back from the collision at node 2.
        assert_eq!(m.decode_to_edges(&[1, 3]), vec![(2, 1), (2, 3)]);
        assert_eq!(m.decode_to_match_edges(&[0, 8]), vec![(0, -1), (8, -1)]);
        assert_eq!(m.decode_to_edges(&[0]), vec![(0, -1)]);
        assert!(m.try_decode(&[9]).is_err());
        // The state is clean after every shot.
        assert_eq!(m.decode(&[2, 6]).1, 4.0);
    }

    #[test]
    fn no_perfect_matching_is_refused_and_forgotten() {
        let mut g = UserGraph::new(3, 0);
        g.add_edge(0, 1, &[0], 1.0, -1.0, Merge::Disallow).unwrap();
        g.add_edge(1, 2, &[1], 1.0, -1.0, Merge::Disallow).unwrap();
        let mut m = Mwpm::new(Graph::from_user_graph(&g, NeighbourOrder::V2_4));
        assert_eq!(m.try_decode(&[0]).unwrap_err(), NO_PERFECT_MATCHING);
        assert_eq!(m.decode(&[0, 2]), (vec![1, 1], 2.0));
    }

    #[test]
    fn negative_weights() {
        let mut g = UserGraph::new(2, 0);
        g.add_edge(0, 1, &[0], -2.0, -1.0, Merge::Disallow).unwrap();
        g.add_boundary_edge(0, &[1], 3.0, -1.0, Merge::Disallow).unwrap();
        g.add_boundary_edge(1, &[], 3.0, -1.0, Merge::Disallow).unwrap();
        let mut m = Mwpm::new(Graph::from_user_graph(&g, NeighbourOrder::V2_4));
        assert_eq!(m.decode(&[0, 1]), (vec![1, 0], -2.0));
        assert_eq!(m.decode_to_edges(&[0, 1]), vec![(0, 1)]);
        assert_eq!(m.decode(&[]), (vec![0, 0], 0.0));
        assert!(m.decode_to_edges(&[]).is_empty());
        assert_eq!(m.try_decode_to_match_edges(&[0, 1]).unwrap_err(), NEGATIVE_WEIGHTS_REFUSED);
    }

    #[test]
    fn many_observables_go_through_the_search_graph() {
        let mut g = repetition(70, 1.0);
        g.ensure_num_fault_ids(80);
        let mut m = Mwpm::new(Graph::from_user_graph(&g, NeighbourOrder::V2_4));
        let (obs, w) = m.decode(&[10, 20]);
        assert_eq!(w, 10.0);
        assert_eq!(obs.iter().enumerate().filter(|(_, &b)| b == 1).map(|(k, _)| k).collect::<Vec<_>>(), (11..=20).collect::<Vec<_>>());
    }

    #[test]
    fn stepping_gives_the_same_matches() {
        let g = Arc::new(Graph::from_user_graph(&repetition(30, 1.5), NeighbourOrder::V2_4));
        let dets = [2u64, 3, 7, 11, 12, 20, 25];
        let mut a = Mwpm::new(g.clone());
        let want = a.decode_to_match_edges(&dets);
        let mut b = Mwpm::new(g);
        b.begin(&dets).unwrap();
        let (mut found, mut handled) = (0, 0);
        while let Some(ev) = b.step() {
            match ev {
                MwpmEvent::Found(_) => found += 1,
                MwpmEvent::Handled(_) => handled += 1,
                _ => {}
            }
        }
        assert!(found > 0 && found == handled);
        // At the end every top region is frozen, matched, and equal (PyMatching's structural
        // `==`) to itself only; a node's local radius bounded by its own top region is its own.
        let tops: Vec<u32> = b.live_regions().into_iter().filter(|&r| b.region(r).blossom_parent == NONE).collect();
        assert!(tops.iter().all(|&r| b.region(r).radius.is_frozen()));
        for &r in &tops {
            assert_eq!(tops.iter().filter(|&&s| b.tree_equal(r, s)).count(), 1);
        }
        for &d in &dets {
            let n = b.node(d as u32);
            let t = b.cur_time();
            assert_eq!(b.compute_local_radius_at_time_bounded_by_region(d as u32, t, n.region_that_arrived_top), b.local_radius(d as u32).get_distance_at_time(t));
        }
        b.finish().unwrap();
        let got: Vec<(i64, i64)> = b.extract_match_edges(&dets).iter().map(|e| (e.loc_from as i64, if e.loc_to == NONE { -1 } else { e.loc_to as i64 })).collect();
        assert_eq!(got, want);
    }

    #[test]
    fn regions_and_trees_are_freed_after_each_shot() {
        let mut m = Mwpm::new(Graph::from_user_graph(&repetition(40, 2.5), NeighbourOrder::V2_4));
        for dets in [vec![1u64, 2, 9, 30], vec![0, 5, 6, 7, 8, 39], vec![]] {
            m.decode(&dets);
            assert!(m.live_regions().is_empty());
            assert!(m.live_alt_tree_nodes().is_empty());
            assert!((0..40).all(|n| m.node(n).region_that_arrived == NONE));
        }
    }
}

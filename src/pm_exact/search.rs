// Copyright 2022 PyMatching Contributors
// Licensed under the Apache License, Version 2.0 (see src/pm_exact/LICENSE). Modified: transliterated
// to Rust for this crate.

//! The search flooder (`search/search_flooder.{h,cc}`, `search/search_detector_node.cc`): a
//! two-ended Dijkstra search between a matched pair (or from a detection event to the boundary)
//! on its own radix heap, whose collision edge is traced back to both ends. Its ties are
//! PyMatching's: last in, first out, then the first strictly earlier neighbour.

use super::graph::{Graph, NONE};
use super::tracker::{FloodEvent, QueuedEventTracker, RadixHeapQueue};

/// `search_detector_node.h`'s `FLIPPED` marker.
pub const FLIPPED: u8 = 1;

/// `SearchDetectorNode`'s ephemeral fields.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SearchDetectorNode {
    pub reached_from_source: u32,
    pub index_of_predecessor: u32,
    pub distance_from_source: i64,
    pub node_event_tracker: QueuedEventTracker,
}

impl Default for SearchDetectorNode {
    fn default() -> Self {
        SearchDetectorNode { reached_from_source: NONE, index_of_predecessor: NONE, distance_from_source: 0, node_event_tracker: QueuedEventTracker::default() }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TargetType {
    DetectorNode,
    Boundary,
    NoTarget,
}

/// `SearchFlooder`, with the search graph's per-edge markers (`neighbor_markers`).
#[derive(Clone, Debug)]
pub struct SearchFlooder {
    pub nodes: Vec<SearchDetectorNode>,
    pub queue: RadixHeapQueue,
    pub reached_nodes: Vec<u32>,
    pub target_type: TargetType,
    /// Per slot of the graph: `FLIPPED` while an edge is flipped an odd number of times.
    pub markers: Vec<u8>,
}

impl SearchFlooder {
    pub fn new(g: &Graph) -> SearchFlooder {
        SearchFlooder { nodes: vec![SearchDetectorNode::default(); g.num_nodes], queue: RadixHeapQueue::default(), reached_nodes: Vec::new(), target_type: TargetType::NoTarget, markers: vec![0; g.num_slots()] }
    }

    fn find_next_event_at_node(&self, g: &Graph, n: u32) -> (usize, i64) {
        let d = &self.nodes[n as usize];
        let now = self.queue.cur_time;
        let mut best_time = i64::MAX;
        let mut best_neighbor = usize::MAX;
        let a = g.slot(n, 0);
        let deg = g.degree(n);
        let mut start = 0;
        if deg > 0 && g.nbr_at(a) == NONE {
            if self.target_type == TargetType::Boundary {
                let collision_time = now + g.weight_at(a) as i64 - (now - d.distance_from_source);
                if collision_time < best_time {
                    best_time = collision_time;
                    best_neighbor = 0;
                }
            }
            start = 1;
        }
        for i in start..deg {
            let w = g.weight_at(a + i) as i64;
            let nb = &self.nodes[g.nbr_at(a + i) as usize];
            let collision_time = if nb.reached_from_source == d.reached_from_source {
                continue;
            } else if nb.reached_from_source == NONE {
                now + w - (now - d.distance_from_source)
            } else {
                let covered_from_this_node = now - d.distance_from_source;
                let covered_from_neighbor = now - nb.distance_from_source;
                now + (w - covered_from_this_node - covered_from_neighbor) / 2
            };
            if collision_time < best_time {
                best_time = collision_time;
                best_neighbor = i;
            }
        }
        (best_neighbor, best_time)
    }

    fn reschedule(&mut self, g: &Graph, n: u32) {
        let (k, t) = self.find_next_event_at_node(g, n);
        let tracker = &mut self.nodes[n as usize].node_event_tracker;
        if k == usize::MAX {
            tracker.set_no_desired_event();
        } else {
            tracker.set_desired_event(FloodEvent::node(n, t as u32), &mut self.queue);
        }
    }

    fn start_at(&mut self, g: &Graph, src: u32) {
        let d = &mut self.nodes[src as usize];
        d.reached_from_source = src;
        d.index_of_predecessor = NONE;
        d.distance_from_source = 0;
        self.reached_nodes.push(src);
        self.reschedule(g, src);
    }

    fn explore(&mut self, g: &Graph, empty: u32, empty_to_from_index: usize) {
        let s = g.slot(empty, empty_to_from_index);
        let from = g.nbr_at(s);
        let f = self.nodes[from as usize];
        let d = &mut self.nodes[empty as usize];
        d.reached_from_source = f.reached_from_source;
        d.index_of_predecessor = empty_to_from_index as u32;
        d.distance_from_source = g.weight_at(s) as i64 + f.distance_from_source;
        self.reached_nodes.push(empty);
        self.reschedule(g, empty);
    }

    fn look_at_node(&mut self, g: &Graph, n: u32) -> Option<(u32, usize)> {
        let (k, t) = self.find_next_event_at_node(g, n);
        let now = self.queue.cur_time;
        if t == now {
            let dst = g.nbr_at(g.slot(n, k));
            if dst == NONE {
                return Some((n, k));
            } else if self.nodes[n as usize].reached_from_source != NONE && self.nodes[dst as usize].reached_from_source == NONE {
                self.explore(g, dst, g.reverse_index(n, k));
                self.nodes[n as usize].node_event_tracker.set_desired_event(FloodEvent::node(n, now as u32), &mut self.queue);
                return None;
            } else {
                return Some((n, k));
            }
        } else if k != usize::MAX {
            self.nodes[n as usize].node_event_tracker.set_desired_event(FloodEvent::node(n, t as u32), &mut self.queue);
        }
        None
    }

    /// `run_until_collision` (`dst == NONE`: to the boundary): the collision edge.
    pub fn run_until_collision(&mut self, g: &Graph, src: u32, dst: u32) -> Option<(u32, usize)> {
        if dst == NONE {
            self.target_type = TargetType::Boundary;
        } else {
            self.target_type = TargetType::DetectorNode;
            self.start_at(g, dst);
        }
        self.start_at(g, src);
        while let Some(ev) = self.queue.dequeue() {
            let n = ev.item;
            if self.nodes[n as usize].node_event_tracker.dequeue_decision(ev, &mut self.queue) {
                if let Some(c) = self.look_at_node(g, n) {
                    return Some(c);
                }
            }
        }
        None
    }

    /// `iter_edges_on_shortest_path_from_middle`: the edges traced back from the collision to
    /// the first end, then to the second, then the collision edge, as (node, neighbour index).
    pub fn iter_edges_on_shortest_path_from_middle(&mut self, g: &Graph, src: u32, dst: u32, mut handle_edge: impl FnMut(&mut SearchFlooder, u32, usize)) -> Result<(), String> {
        let Some((cn, ck)) = self.run_until_collision(g, src, dst) else {
            self.reset();
            return Err(format!("no path from node {src} to {}", if dst == NONE { "the boundary".to_string() } else { format!("node {dst}") }));
        };
        let mut cur = cn;
        while self.nodes[cur as usize].index_of_predecessor != NONE {
            let pred = self.nodes[cur as usize].index_of_predecessor as usize;
            handle_edge(self, cur, pred);
            cur = g.nbr_at(g.slot(cur, pred));
        }
        let other = g.nbr_at(g.slot(cn, ck));
        if other != NONE {
            let mut cur = other;
            while self.nodes[cur as usize].index_of_predecessor != NONE {
                let pred = self.nodes[cur as usize].index_of_predecessor as usize;
                handle_edge(self, cur, pred);
                cur = g.nbr_at(g.slot(cur, pred));
            }
        }
        handle_edge(self, cn, ck);
        self.reset();
        Ok(())
    }

    /// `flip_edge`: both directions of an edge.
    #[inline]
    pub fn flip_edge(&mut self, g: &Graph, n: u32, k: usize) {
        self.markers[g.slot(n, k)] ^= FLIPPED;
        let nb = g.nbr_at(g.slot(n, k));
        if nb != NONE {
            self.markers[g.slot(nb, g.reverse_index(n, k))] ^= FLIPPED;
        }
    }

    /// `SearchFlooder::reset`.
    pub fn reset(&mut self) {
        for &n in &self.reached_nodes {
            let d = &mut self.nodes[n as usize];
            d.reached_from_source = NONE;
            d.index_of_predecessor = NONE;
            d.node_event_tracker.clear();
        }
        self.reached_nodes.clear();
        self.queue.reset();
    }
}

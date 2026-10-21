// Copyright 2022 PyMatching Contributors
// Licensed under the Apache License, Version 2.0 (see src/pm_exact/LICENSE). Modified: transliterated
// to Rust for this crate.

//! The driver (`driver/mwpm_decoding.cc`): the timeline run to completion, the matches
//! extracted detection event by detection event, and the three ways out (fault ids and weight,
//! edges, matched detection events); and the step-by-step run PyMatching's animation makes
//! (`diagram/mwpm_diagram.cc`'s `write_animated_decoding_svg_frames`).

use super::flooder::{CompressedEdge, MatcherEvent};
use super::graph::NONE;
use super::matcher::MatchingResult;
use super::search::FLIPPED;
use super::tracker::FloodEvent;
use super::{Mwpm, MwpmEvent, Pending};

pub const NO_PERFECT_MATCHING: &str = "No perfect matching could be found. This likely means that the syndrome has odd parity in the support of a connected component without a boundary.";
pub const NEGATIVE_WEIGHTS_REFUSED: &str = "Decoding to matched detection events not supported for graphs containing edges with negative weights.";

impl Mwpm {
    /// `process_timeline_until_completion`'s set-up: the clock at 0 and a region at every
    /// detection event (skipping the user graph's boundary nodes), the negative-weight edges'
    /// toggles applied. Detection events must be distinct. Out-of-range events are refused
    /// before anything is created (PyMatching first creates the regions of the earlier events,
    /// so when there are earlier (non-boundary) detection events, its later decodes raise
    /// "!mwpm.flooder.queue.empty()"; deliberately not copied: see `decode_batch` in
    /// `src/py_pm.rs`).
    pub fn begin(&mut self, detection_events: &[u64]) -> Result<(), String> {
        if !self.queue.is_empty() {
            return Err("!mwpm.flooder.queue.empty()".into());
        }
        self.queue.cur_time = 0;
        self.pending = Pending::None;
        let g = self.g.clone();
        let n = g.num_nodes as u64;
        let is_boundary = |d: u64| (d as usize) < g.is_user_graph_boundary_node.len() && g.is_user_graph_boundary_node[d as usize];
        if g.negative_weight_detection_events.is_empty() {
            if let Some(&d) = detection_events.iter().find(|&&d| d >= n) {
                return Err(format!("The detection event with index {d} does not correspond to a node in the graph, which only has {n} nodes."));
            }
            for &d in detection_events {
                if !is_boundary(d) {
                    self.create_detection_event(d as u32);
                }
            }
        } else {
            if let Some(&d) = detection_events.iter().find(|&&d| d >= n) {
                return Err(format!("Detection event index `{d}` is larger than any detector node index in the graph."));
            }
            for &d in &g.negative_weight_detection_events {
                self.nodes[d as usize].radius_of_arrival = 1;
            }
            for &d in detection_events {
                if self.nodes[d as usize].radius_of_arrival == 0 {
                    if !is_boundary(d) {
                        self.create_detection_event(d as u32);
                    }
                } else {
                    self.nodes[d as usize].radius_of_arrival = 0;
                }
            }
            for &d in &g.negative_weight_detection_events {
                if self.nodes[d as usize].radius_of_arrival != 0 {
                    self.nodes[d as usize].radius_of_arrival = 0;
                    self.create_detection_event(d as u32);
                }
            }
        }
        Ok(())
    }

    /// The end of `process_timeline_until_completion`: refused (and the state reset) if some
    /// alternating tree is left.
    pub fn finish(&mut self) -> Result<(), String> {
        if self.alt.live() != 0 {
            self.reset();
            return Err(NO_PERFECT_MATCHING.into());
        }
        Ok(())
    }

    /// `process_timeline_until_completion`.
    pub fn process_timeline_until_completion(&mut self, detection_events: &[u64]) -> Result<(), String> {
        self.begin(detection_events)?;
        while let Some(ev) = self.run_until_next_mwpm_notification() {
            self.process_event(ev);
        }
        self.finish()
    }

    /// `shatter_blossoms_for_all_detection_events_and_extract_*`: the region at each listed
    /// node (whether or not it was a detection event), shattered into matches.
    pub fn shatter_blossoms_for_all_detection_events(&mut self, detection_events: &[u64], res: &mut MatchingResult, edges: bool) {
        for &i in detection_events {
            let d = &self.nodes[i as usize];
            if d.region_that_arrived != NONE {
                let top = d.region_that_arrived_top;
                self.shatter_blossom_and_extract(top, res, edges);
            }
        }
    }

    fn shatter_all(&mut self, detection_events: &[u64], edges: bool) -> MatchingResult {
        let mut res = MatchingResult::default();
        self.match_edges.clear();
        self.shatter_blossoms_for_all_detection_events(detection_events, &mut res, edges);
        if !self.g.negative_weight_detection_events.is_empty() {
            let g = self.g.clone();
            self.shatter_blossoms_for_all_detection_events(&g.negative_weight_detection_events, &mut res, edges);
        }
        res
    }

    /// `decode_detection_events`: XORs the observables the matching flips into `obs` (one byte
    /// each, `num_observables` long), and returns its weight (integer; over
    /// `normalising_constant` for PyMatching's). Over 64 observables, through the search graph.
    pub fn decode_into(&mut self, detection_events: &[u64], obs: &mut [u8]) -> Result<i64, String> {
        self.process_timeline_until_completion(detection_events)?;
        let g = self.g.clone();
        if g.num_observables > 64 {
            self.shatter_all(detection_events, true);
            let mut weight = 0i64;
            let edges = std::mem::take(&mut self.match_edges);
            for e in &edges {
                self.search.iter_edges_on_shortest_path_from_middle(&g, e.loc_from, e.loc_to, |_, n, k| {
                    for &i in g.neighbor_observable_indices(n, k) {
                        // Ids past the count (a merge can bring them) are dropped.
                        if let Some(b) = obs.get_mut(i as usize) {
                            *b ^= 1;
                        }
                    }
                    weight += g.neighbor_weights(n)[k] as i64;
                })?;
            }
            self.match_edges = edges;
            for &o in &g.negative_weight_observables {
                if let Some(b) = obs.get_mut(o) {
                    *b ^= 1;
                }
            }
            Ok(weight + g.negative_weight_sum)
        } else {
            let mut res = self.shatter_all(detection_events, false);
            res.obs_mask ^= g.negative_weight_obs_mask;
            for (i, b) in obs.iter_mut().enumerate().take(g.num_observables) {
                *b ^= ((res.obs_mask >> i) & 1) as u8;
            }
            Ok(res.weight + g.negative_weight_sum)
        }
    }

    /// The observables as a mask (at most 64) and the integer weight, without a byte per
    /// observable (`decode_detection_events_for_up_to_64_observables`, masked to the count).
    pub fn decode_to_mask(&mut self, detection_events: &[u64]) -> Result<(u64, i64), String> {
        let g = self.g.clone();
        if g.num_observables > 64 {
            return Err("Too many observables".into());
        }
        self.process_timeline_until_completion(detection_events)?;
        let mut res = self.shatter_all(detection_events, false);
        res.obs_mask ^= g.negative_weight_obs_mask;
        let keep = if g.num_observables == 64 { u64::MAX } else { (1u64 << g.num_observables) - 1 };
        Ok((res.obs_mask & keep, res.weight + g.negative_weight_sum))
    }

    /// PyMatching's `decode`: the observables flipped (a byte each) and the weight.
    pub fn try_decode(&mut self, detection_events: &[u64]) -> Result<(Vec<u8>, f64), String> {
        let mut obs = vec![0u8; self.g.num_observables];
        let w = self.decode_into(detection_events, &mut obs)?;
        Ok((obs, w as f64 / self.g.normalising_constant))
    }

    /// As `try_decode`, panicking with PyMatching's message where it raises.
    pub fn decode(&mut self, detection_events: &[u64]) -> (Vec<u8>, f64) {
        self.try_decode(detection_events).unwrap_or_else(|e| panic!("{e}"))
    }

    /// `decode_detection_events_to_match_edges`: the matched pairs (`-1`: the boundary) in
    /// PyMatching's order. Refused for graphs with negative weights, as there.
    pub fn try_decode_to_match_edges(&mut self, detection_events: &[u64]) -> Result<Vec<(i64, i64)>, String> {
        if self.g.negative_weight_sum != 0 {
            return Err(NEGATIVE_WEIGHTS_REFUSED.into());
        }
        self.process_timeline_until_completion(detection_events)?;
        self.match_edges.clear();
        let mut res = MatchingResult::default();
        self.shatter_blossoms_for_all_detection_events(detection_events, &mut res, true);
        Ok(self.match_edges.iter().map(|e| (e.loc_from as i64, if e.loc_to == NONE { -1 } else { e.loc_to as i64 })).collect())
    }

    pub fn decode_to_match_edges(&mut self, detection_events: &[u64]) -> Vec<(i64, i64)> {
        self.try_decode_to_match_edges(detection_events).unwrap_or_else(|e| panic!("{e}"))
    }

    /// `decode_detection_events_to_edges`: the negative-weight edges, then each match's shortest
    /// path traced from the middle, each edge flipped as it is listed; then the list compacted
    /// (an edge no longer flipped swapped out for the last, a flipped one kept and unflipped).
    pub fn try_decode_to_edges(&mut self, detection_events: &[u64]) -> Result<Vec<(i64, i64)>, String> {
        self.process_timeline_until_completion(detection_events)?;
        self.shatter_all(detection_events, true);
        let g = self.g.clone();
        let mut edges: Vec<i64> = Vec::with_capacity(detection_events.len() / 2);
        for &(u, v) in &g.negative_weight_edges {
            let k = g.index_of_neighbor(u, v).expect("a negative-weight edge is in the graph");
            self.search.flip_edge(&g, u, k);
            edges.push(u as i64);
            edges.push(if v == NONE { -1 } else { v as i64 });
        }
        let match_edges = std::mem::take(&mut self.match_edges);
        for e in &match_edges {
            let r = self.search.iter_edges_on_shortest_path_from_middle(&g, e.loc_from, e.loc_to, |s, n, k| {
                s.flip_edge(&g, n, k);
                edges.push(n as i64);
                let nb = g.neighbors(n)[k];
                edges.push(if nb == NONE { -1 } else { nb as i64 });
            });
            if let Err(err) = r {
                // (Unreachable: a match is always joined by a path. Leave no edge flipped.)
                self.search.markers.iter_mut().for_each(|m| *m = 0);
                self.match_edges = match_edges;
                return Err(err);
            }
        }
        self.match_edges = match_edges;
        let mut i = 0;
        while i < edges.len() / 2 {
            let (u, v) = (edges[2 * i] as u32, edges[2 * i + 1]);
            let idx = if v == -1 { 0 } else { g.index_of_neighbor(u, v as u32).expect("an edge of the graph") };
            if self.search.markers[g.slot(u, idx)] & FLIPPED == 0 {
                let len = edges.len();
                edges[2 * i] = edges[len - 2];
                edges[2 * i + 1] = edges[len - 1];
                edges.truncate(len - 2);
            } else {
                self.search.flip_edge(&g, u, idx);
                i += 1;
            }
        }
        Ok(edges.chunks_exact(2).map(|p| (p[0], p[1])).collect())
    }

    pub fn decode_to_edges(&mut self, detection_events: &[u64]) -> Vec<(i64, i64)> {
        self.try_decode_to_edges(detection_events).unwrap_or_else(|e| panic!("{e}"))
    }

    /// The animation's set-up (`write_animated_decoding_svg_frames`): the clock at 0 and a
    /// region at every detection event, nothing else.
    pub fn begin_animation(&mut self, detection_events: &[u64]) {
        self.queue.cur_time = 0;
        self.pending = Pending::None;
        for &d in detection_events {
            self.create_detection_event(d as u32);
        }
    }

    /// The animation's `check_if_done`: every detection event's top region frozen.
    pub fn all_frozen(&self, detection_events: &[u64]) -> bool {
        detection_events.iter().all(|&d| {
            let top = self.nodes[d as usize].region_that_arrived_top;
            top != NONE && self.regions.items[top as usize].radius.is_frozen()
        })
    }

    /// One step of the run, as PyMatching's animation takes it (see the module docs): `None` once
    /// the queue is dry and nothing is pending.
    pub fn step(&mut self) -> Option<MwpmEvent> {
        match std::mem::replace(&mut self.pending, Pending::None) {
            Pending::None => {
                let ev = self.queue.dequeue()?;
                self.pending = Pending::Flood(ev);
                Some(MwpmEvent::Dequeued(ev))
            }
            Pending::Flood(ev) => {
                if !self.dequeue_decision(ev) {
                    return Some(MwpmEvent::Discarded(ev));
                }
                match self.process_tentative_event(ev) {
                    None => Some(MwpmEvent::Flooded(ev)),
                    Some(m) => {
                        self.pending = Pending::Matcher(m);
                        Some(MwpmEvent::Found(m))
                    }
                }
            }
            Pending::Matcher(m) => {
                self.process_event(m);
                Some(MwpmEvent::Handled(m))
            }
        }
    }

    /// The matches extracted (after a run to completion, by `step` or otherwise), in PyMatching's
    /// order, as `try_decode_to_match_edges` lists them; the regions are freed.
    pub fn extract_match_edges(&mut self, detection_events: &[u64]) -> Vec<CompressedEdge> {
        self.shatter_all(detection_events, true);
        self.match_edges.clone()
    }

    /// `Mwpm::reset`: every node cleared, the queue emptied, the arenas forgotten.
    pub fn reset(&mut self) {
        for n in self.nodes.iter_mut() {
            n.reset();
        }
        for m in self.search.nodes.iter_mut() {
            *m = Default::default();
        }
        self.search.reached_nodes.clear();
        self.search.queue.reset();
        self.queue.clear();
        self.alt.clear();
        self.regions.clear();
        self.match_edges.clear();
        self.pending = Pending::None;
    }

    /// The events in the queue in time order, for drawing.
    pub fn queued_events(&self) -> Vec<FloodEvent> {
        self.queue.to_vec()
    }

    /// The matcher event found but not yet handled, if any.
    pub fn pending_matcher_event(&self) -> Option<MatcherEvent> {
        match self.pending {
            Pending::Matcher(m) => Some(m),
            _ => None,
        }
    }
}

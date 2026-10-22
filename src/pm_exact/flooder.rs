// Copyright 2022 PyMatching Contributors
// Licensed under the Apache License, Version 2.0 (see src/pm_exact/LICENSE). Modified: transliterated
// to Rust for this crate.

//! The flooder: `DetectorNode` (`flooder/detector_node.{h,cc}`), `GraphFillRegion`
//! (`flooder/graph_fill_region.{h,cc}`), the arena (`arena.h`) and `GraphFlooder`
//! (`flooder/graph_flooder.cc`), as methods on `Mwpm`. Pointers are indices (`NONE` for null);
//! the arena hands back the most recently freed slot first, as PyMatching's does, so a stale
//! queue entry meets the same region it would meet there.

use super::graph::NONE;
use super::tracker::{FloodEvent, QueuedEventTracker, Varying};
use super::Mwpm;

/// `CompressedEdge`: a path between two detection events (`loc_to == NONE`: the boundary) and
/// the observables it crosses.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CompressedEdge {
    pub loc_from: u32,
    pub loc_to: u32,
    pub obs_mask: u64,
}

impl CompressedEdge {
    pub const EMPTY: CompressedEdge = CompressedEdge { loc_from: NONE, loc_to: NONE, obs_mask: 0 };

    #[inline]
    pub fn reversed(self) -> CompressedEdge {
        CompressedEdge { loc_from: self.loc_to, loc_to: self.loc_from, obs_mask: self.obs_mask }
    }
}

impl Default for CompressedEdge {
    fn default() -> Self {
        CompressedEdge::EMPTY
    }
}

/// `RegionEdge`: a blossom child and the edge to the next child in the cycle.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct RegionEdge {
    pub region: u32,
    pub edge: CompressedEdge,
}

/// `Match`: what a region is matched to (`region == NONE`: the boundary, or nothing).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Match {
    pub region: u32,
    pub edge: CompressedEdge,
}

impl Match {
    pub const EMPTY: Match = Match { region: NONE, edge: CompressedEdge::EMPTY };
}

/// PyMatching's `MwpmEvent`: what the flooder tells the matcher.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MatcherEvent {
    RegionHitRegion { region1: u32, region2: u32, edge: CompressedEdge },
    RegionHitBoundary { region: u32, edge: CompressedEdge },
    BlossomShatter { blossom_region: u32, in_parent_region: u32, in_child_region: u32 },
}

/// A detector node's state while matching (`DetectorNode`'s ephemeral fields).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DetectorNode {
    pub region_that_arrived: u32,
    pub region_that_arrived_top: u32,
    pub wrapped_radius_cached: i64,
    pub reached_from_source: u32,
    pub observables_crossed_from_source: u64,
    pub radius_of_arrival: i64,
    pub node_event_tracker: QueuedEventTracker,
}

impl Default for DetectorNode {
    fn default() -> Self {
        DetectorNode {
            region_that_arrived: NONE,
            region_that_arrived_top: NONE,
            wrapped_radius_cached: 0,
            reached_from_source: NONE,
            observables_crossed_from_source: 0,
            radius_of_arrival: 0,
            node_event_tracker: QueuedEventTracker::default(),
        }
    }
}

impl DetectorNode {
    /// `DetectorNode::reset`.
    #[inline]
    pub fn reset(&mut self) {
        *self = DetectorNode::default();
    }
}

/// `GraphFillRegion`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GraphFillRegion {
    pub blossom_parent: u32,
    pub blossom_parent_top: u32,
    pub alt_tree_node: u32,
    /// For a blossom, its growth since it formed (the extra radius).
    pub radius: Varying,
    pub shrink_event_tracker: QueuedEventTracker,
    pub matched: Match,
    pub blossom_children: Vec<RegionEdge>,
    pub shell_area: Vec<u32>,
}

impl GraphFillRegion {
    fn reset(&mut self, me: u32) {
        self.blossom_parent = NONE;
        self.blossom_parent_top = me;
        self.alt_tree_node = NONE;
        self.radius = Varying(1);
        self.shrink_event_tracker.clear();
        self.matched = Match::EMPTY;
        self.blossom_children.clear();
        self.shell_area.clear();
    }
}

/// `Arena<T>`: slots never move; freed slots are reused last freed, first used.
#[derive(Clone, Debug)]
pub struct Arena<T> {
    pub items: Vec<T>,
    free: Vec<u32>,
    is_free: Vec<bool>,
}

impl<T> Default for Arena<T> {
    fn default() -> Self {
        Arena { items: Vec::new(), free: Vec::new(), is_free: Vec::new() }
    }
}

impl<T> Arena<T> {
    #[inline]
    pub(crate) fn alloc(&mut self, make: impl FnOnce() -> T) -> u32 {
        match self.free.pop() {
            Some(i) => {
                self.is_free[i as usize] = false;
                i
            }
            None => {
                self.items.push(make());
                self.is_free.push(false);
                (self.items.len() - 1) as u32
            }
        }
    }

    #[inline]
    pub(crate) fn del(&mut self, i: u32) {
        debug_assert!(!self.is_free[i as usize]);
        self.is_free[i as usize] = true;
        self.free.push(i);
    }

    /// How many slots are in use.
    #[inline]
    pub fn live(&self) -> usize {
        self.items.len() - self.free.len()
    }

    pub fn is_live(&self, i: u32) -> bool {
        (i as usize) < self.items.len() && !self.is_free[i as usize]
    }

    /// The slots in use, in slot order.
    pub fn live_slots(&self) -> Vec<u32> {
        (0..self.items.len() as u32).filter(|&i| !self.is_free[i as usize]).collect()
    }

    /// `~Arena`: everything forgotten.
    pub(crate) fn clear(&mut self) {
        self.items.clear();
        self.free.clear();
        self.is_free.clear();
    }
}

/// `region_that_arrived_top->radius + wrapped_radius_cached`, or 0 off any region.
#[inline]
pub(crate) fn local_radius(nodes: &[DetectorNode], regions: &[GraphFillRegion], n: u32) -> Varying {
    let d = &nodes[n as usize];
    if d.region_that_arrived_top == NONE {
        return Varying(0);
    }
    regions[d.region_that_arrived_top as usize].radius.plus(d.wrapped_radius_cached)
}

/// The radius offset relative to the top region, in the same 64-bit units as arrivals.
/// Long paths can exceed i32 even when every individual edge weight fits.
#[inline]
pub(crate) fn compute_wrapped_radius(nodes: &[DetectorNode], regions: &[GraphFillRegion], n: u32) -> i64 {
    let d = &nodes[n as usize];
    if d.reached_from_source == NONE {
        return 0;
    }
    let mut total: i64 = 0;
    let mut r = d.region_that_arrived;
    while r != d.region_that_arrived_top {
        total += regions[r as usize].radius.y_intercept();
        r = regions[r as usize].blossom_parent;
    }
    total - d.radius_of_arrival
}

/// `do_op_for_each_node_in_total_area`'s order: the shell newest first, then each child's area.
pub(crate) fn total_area(regions: &[GraphFillRegion], root: u32, out: &mut Vec<u32>, stack: &mut Vec<u32>) {
    stack.clear();
    stack.push(root);
    while let Some(r) = stack.pop() {
        let reg = &regions[r as usize];
        out.extend(reg.shell_area.iter().rev());
        stack.extend(reg.blossom_children.iter().rev().map(|c| c.region));
    }
}

/// `do_op_for_each_descendant_and_self`'s order (pre-order, children in order).
pub(crate) fn descendants_and_self(regions: &[GraphFillRegion], root: u32, out: &mut Vec<u32>, stack: &mut Vec<u32>) {
    stack.clear();
    stack.push(root);
    while let Some(r) = stack.pop() {
        out.push(r);
        stack.extend(regions[r as usize].blossom_children.iter().rev().map(|c| c.region));
    }
}

impl Mwpm {
    /// `find_next_event_at_node_returning_neighbor_index_and_time`: the neighbour index (or
    /// `usize::MAX`) and time of the node's next event.
    pub(crate) fn find_next_event_at_node(&self, n: u32) -> (usize, i64) {
        let g = &*self.g;
        let nodes = &self.nodes[..];
        let regions = &self.regions.items[..];
        let rad1 = local_radius(nodes, regions, n);
        let a = g.slot(n, 0);
        let deg = g.degree(n);
        let mut best_time = i64::MAX;
        let mut best_neighbor = usize::MAX;
        let mut start = 0;
        if rad1.is_growing() {
            if deg > 0 && g.nbr_at(a) == NONE {
                let collision_time = g.weight_at(a) as i64 - rad1.y_intercept();
                if collision_time < best_time {
                    best_time = collision_time;
                    best_neighbor = 0;
                }
                start = 1;
            }
            let top = nodes[n as usize].region_that_arrived_top;
            for i in start..deg {
                let nb = g.nbr_at(a + i);
                if nodes[nb as usize].region_that_arrived_top == top {
                    continue;
                }
                let rad2 = local_radius(nodes, regions, nb);
                if rad2.is_shrinking() {
                    continue;
                }
                let mut collision_time = g.weight_at(a + i) as i64 - rad1.y_intercept() - rad2.y_intercept();
                if rad2.is_growing() {
                    collision_time >>= 1;
                }
                if collision_time < best_time {
                    best_time = collision_time;
                    best_neighbor = i;
                }
            }
        } else {
            if deg > 0 && g.nbr_at(a) == NONE {
                start = 1;
            }
            for i in start..deg {
                let nb = g.nbr_at(a + i);
                let rad2 = local_radius(nodes, regions, nb);
                if rad2.is_growing() {
                    let collision_time = g.weight_at(a + i) as i64 - rad1.y_intercept() - rad2.y_intercept();
                    if collision_time < best_time {
                        best_time = collision_time;
                        best_neighbor = i;
                    }
                }
            }
        }
        (best_neighbor, best_time)
    }

    #[inline]
    pub(crate) fn reschedule_events_at_detector_node(&mut self, n: u32) {
        let (k, t) = self.find_next_event_at_node(n);
        let tracker = &mut self.nodes[n as usize].node_event_tracker;
        if k == usize::MAX {
            tracker.set_no_desired_event();
        } else {
            tracker.set_desired_event(FloodEvent::node(n, t as u32), &mut self.queue);
        }
    }

    /// Reschedules every node in a region's total area, in PyMatching's order.
    pub(crate) fn reschedule_total_area(&mut self, region: u32) {
        let mut area = std::mem::take(&mut self.area);
        let mut stack = std::mem::take(&mut self.stack);
        area.clear();
        total_area(&self.regions.items, region, &mut area, &mut stack);
        for &n in &area {
            self.reschedule_events_at_detector_node(n);
        }
        self.area = area;
        self.stack = stack;
    }

    pub(crate) fn schedule_tentative_shrink_event(&mut self, region: u32) {
        let r = &self.regions.items[region as usize];
        let t = match r.shell_area.last() {
            None => r.radius.time_of_x_intercept_for_shrinking(),
            Some(&n) => local_radius(&self.nodes, &self.regions.items, n).time_of_x_intercept_for_shrinking(),
        };
        self.regions.items[region as usize].shrink_event_tracker.set_desired_event(FloodEvent::region(region, t as u32), &mut self.queue);
    }

    pub(crate) fn do_region_created_at_empty_detector_node(&mut self, region: u32, n: u32) {
        let d = &mut self.nodes[n as usize];
        d.reached_from_source = n;
        d.radius_of_arrival = 0;
        d.region_that_arrived = region;
        d.region_that_arrived_top = region;
        d.wrapped_radius_cached = 0;
        self.regions.items[region as usize].shell_area.push(n);
        self.reschedule_events_at_detector_node(n);
    }

    pub(crate) fn do_region_arriving_at_empty_detector_node(&mut self, region: u32, empty: u32, from: u32, from_to_empty_index: usize) {
        let obs = self.nodes[from as usize].observables_crossed_from_source ^ self.g.obs_at(self.g.slot(from, from_to_empty_index));
        let reached = self.nodes[from as usize].reached_from_source;
        let r = &self.regions.items[region as usize];
        let radius_of_arrival = r.radius.get_distance_at_time(self.queue.cur_time);
        let top = r.blossom_parent_top;
        let d = &mut self.nodes[empty as usize];
        d.observables_crossed_from_source = obs;
        d.reached_from_source = reached;
        d.radius_of_arrival = radius_of_arrival;
        d.region_that_arrived = region;
        d.region_that_arrived_top = top;
        self.nodes[empty as usize].wrapped_radius_cached = compute_wrapped_radius(&self.nodes, &self.regions.items, empty);
        self.regions.items[region as usize].shell_area.push(empty);
        self.reschedule_events_at_detector_node(empty);
    }

    pub(crate) fn do_region_shrinking(&mut self, region: u32) -> Option<MatcherEvent> {
        let r = &self.regions.items[region as usize];
        if r.shell_area.is_empty() {
            return Some(self.do_blossom_shattering(region));
        }
        if r.shell_area.len() == 1 && r.blossom_children.is_empty() {
            return Some(self.do_degenerate_implosion(region));
        }
        let leaving = self.regions.items[region as usize].shell_area.pop().expect("not empty");
        let d = &mut self.nodes[leaving as usize];
        d.region_that_arrived = NONE;
        d.region_that_arrived_top = NONE;
        d.wrapped_radius_cached = 0;
        d.reached_from_source = NONE;
        d.radius_of_arrival = 0;
        d.observables_crossed_from_source = 0;
        self.reschedule_events_at_detector_node(leaving);
        self.schedule_tentative_shrink_event(region);
        None
    }

    pub(crate) fn do_neighbor_interaction(&mut self, src: u32, src_to_dst_index: usize, dst: u32) -> Option<MatcherEvent> {
        let (s, d) = (self.nodes[src as usize], self.nodes[dst as usize]);
        if s.region_that_arrived != NONE && d.region_that_arrived == NONE {
            self.do_region_arriving_at_empty_detector_node(s.region_that_arrived_top, dst, src, src_to_dst_index);
            None
        } else if d.region_that_arrived != NONE && s.region_that_arrived == NONE {
            let back = self.g.reverse_index(src, src_to_dst_index);
            self.do_region_arriving_at_empty_detector_node(d.region_that_arrived_top, src, dst, back);
            None
        } else {
            Some(MatcherEvent::RegionHitRegion {
                region1: s.region_that_arrived_top,
                region2: d.region_that_arrived_top,
                edge: CompressedEdge {
                    loc_from: s.reached_from_source,
                    loc_to: d.reached_from_source,
                    obs_mask: s.observables_crossed_from_source ^ d.observables_crossed_from_source ^ self.g.obs_at(self.g.slot(src, src_to_dst_index)),
                },
            })
        }
    }

    pub(crate) fn do_region_hit_boundary_interaction(&self, n: u32) -> MatcherEvent {
        let d = &self.nodes[n as usize];
        MatcherEvent::RegionHitBoundary {
            region: d.region_that_arrived_top,
            edge: CompressedEdge { loc_from: d.reached_from_source, loc_to: NONE, obs_mask: d.observables_crossed_from_source ^ self.g.obs_at(self.g.slot(n, 0)) },
        }
    }

    pub(crate) fn do_degenerate_implosion(&self, region: u32) -> MatcherEvent {
        let a = &self.alt.items[self.regions.items[region as usize].alt_tree_node as usize];
        let p = &self.alt.items[a.parent.alt_tree_node as usize];
        MatcherEvent::RegionHitRegion {
            region1: p.outer_region,
            region2: a.outer_region,
            edge: CompressedEdge { loc_from: a.parent.edge.loc_to, loc_to: a.inner_to_outer_edge.loc_to, obs_mask: a.inner_to_outer_edge.obs_mask ^ a.parent.edge.obs_mask },
        }
    }

    pub(crate) fn do_blossom_shattering(&self, region: u32) -> MatcherEvent {
        let a = &self.alt.items[self.regions.items[region as usize].alt_tree_node as usize];
        MatcherEvent::BlossomShatter {
            blossom_region: region,
            in_parent_region: self.heir_region_on_shatter(a.parent.edge.loc_from),
            in_child_region: self.heir_region_on_shatter(a.inner_to_outer_edge.loc_from),
        }
    }

    /// `DetectorNode::heir_region_on_shatter`: the child of the node's top region that owns it.
    pub fn heir_region_on_shatter(&self, n: u32) -> u32 {
        let d = &self.nodes[n as usize];
        let mut r = d.region_that_arrived;
        loop {
            let p = self.regions.items[r as usize].blossom_parent;
            if p == d.region_that_arrived_top {
                return r;
            }
            r = p;
        }
    }

    /// A fresh `GraphFillRegion` (`alloc_default_constructed`).
    pub(crate) fn alloc_region(&mut self) -> u32 {
        let r = self.regions.alloc(|| GraphFillRegion {
            blossom_parent: NONE,
            blossom_parent_top: NONE,
            alt_tree_node: NONE,
            radius: Varying(1),
            shrink_event_tracker: QueuedEventTracker::default(),
            matched: Match::EMPTY,
            blossom_children: Vec::new(),
            shell_area: Vec::new(),
        });
        self.regions.items[r as usize].reset(r);
        r
    }

    pub(crate) fn create_blossom(&mut self, contained_regions: Vec<RegionEdge>) -> u32 {
        let b = self.alloc_region();
        let t = self.queue.cur_time;
        self.regions.items[b as usize].radius = Varying::growing_varying_with_zero_distance_at_time(t);
        self.regions.items[b as usize].blossom_children = contained_regions;
        for k in 0..self.regions.items[b as usize].blossom_children.len() {
            let child = self.regions.items[b as usize].blossom_children[k].region;
            let c = &mut self.regions.items[child as usize];
            c.radius = c.radius.then_frozen_at_time(t);
            self.wrap_into_blossom(child, b);
            self.regions.items[child as usize].shrink_event_tracker.set_no_desired_event();
        }
        self.reschedule_total_area(b);
        b
    }

    /// Sets every descendant's top (and its nodes' tops, and maybe their wrapped radii).
    fn set_tops(&mut self, region: u32, top: u32, wrapped: bool) {
        let mut list = std::mem::take(&mut self.stack2);
        let mut stack = std::mem::take(&mut self.stack);
        list.clear();
        descendants_and_self(&self.regions.items, region, &mut list, &mut stack);
        self.stack = stack;
        for &desc in &list {
            self.regions.items[desc as usize].blossom_parent_top = top;
            for k in 0..self.regions.items[desc as usize].shell_area.len() {
                let n = self.regions.items[desc as usize].shell_area[k];
                self.nodes[n as usize].region_that_arrived_top = top;
                if wrapped {
                    self.nodes[n as usize].wrapped_radius_cached = compute_wrapped_radius(&self.nodes, &self.regions.items, n);
                }
            }
        }
        self.stack2 = list;
    }

    pub(crate) fn wrap_into_blossom(&mut self, region: u32, new_parent: u32) {
        self.regions.items[region as usize].blossom_parent = new_parent;
        self.set_tops(region, new_parent, true);
    }

    pub(crate) fn clear_blossom_parent(&mut self, region: u32) {
        self.regions.items[region as usize].blossom_parent = NONE;
        self.set_tops(region, region, true);
    }

    pub(crate) fn clear_blossom_parent_ignoring_wrapped_radius(&mut self, region: u32) {
        self.regions.items[region as usize].blossom_parent = NONE;
        self.set_tops(region, region, false);
    }

    pub(crate) fn dequeue_decision(&mut self, ev: FloodEvent) -> bool {
        if ev.is_region() {
            self.regions.items[ev.index() as usize].shrink_event_tracker.dequeue_decision(ev, &mut self.queue)
        } else {
            self.nodes[ev.index() as usize].node_event_tracker.dequeue_decision(ev, &mut self.queue)
        }
    }

    /// `dequeue_valid`: `None` is PyMatching's `NO_FLOOD_CHECK_EVENT` (the queue ran dry).
    #[inline]
    pub(crate) fn dequeue_valid(&mut self) -> Option<FloodEvent> {
        loop {
            let ev = self.queue.dequeue()?;
            if self.dequeue_decision(ev) {
                return Some(ev);
            }
        }
    }

    pub(crate) fn set_region_growing(&mut self, region: u32) {
        let t = self.queue.cur_time;
        let r = &mut self.regions.items[region as usize];
        r.radius = r.radius.then_growing_at_time(t);
        r.shrink_event_tracker.set_no_desired_event();
        self.reschedule_total_area(region);
    }

    pub(crate) fn set_region_frozen(&mut self, region: u32) {
        let t = self.queue.cur_time;
        let r = &mut self.regions.items[region as usize];
        let was_shrinking = r.radius.is_shrinking();
        r.radius = r.radius.then_frozen_at_time(t);
        r.shrink_event_tracker.set_no_desired_event();
        if was_shrinking {
            self.reschedule_total_area(region);
        }
    }

    pub(crate) fn set_region_shrinking(&mut self, region: u32) {
        let t = self.queue.cur_time;
        let r = &mut self.regions.items[region as usize];
        r.radius = r.radius.then_shrinking_at_time(t);
        self.schedule_tentative_shrink_event(region);
        let mut area = std::mem::take(&mut self.area);
        let mut stack = std::mem::take(&mut self.stack);
        area.clear();
        total_area(&self.regions.items, region, &mut area, &mut stack);
        for &n in &area {
            self.nodes[n as usize].node_event_tracker.set_no_desired_event();
        }
        self.area = area;
        self.stack = stack;
    }

    pub(crate) fn do_look_at_node_event(&mut self, n: u32) -> Option<MatcherEvent> {
        let (k, t) = self.find_next_event_at_node(n);
        let now = self.queue.cur_time;
        if t == now {
            self.nodes[n as usize].node_event_tracker.set_desired_event(FloodEvent::node(n, now as u32), &mut self.queue);
            let nb = self.g.nbr_at(self.g.slot(n, k));
            if nb == NONE {
                return Some(self.do_region_hit_boundary_interaction(n));
            }
            return self.do_neighbor_interaction(n, k, nb);
        } else if k != usize::MAX {
            self.nodes[n as usize].node_event_tracker.set_desired_event(FloodEvent::node(n, t as u32), &mut self.queue);
        }
        None
    }

    pub(crate) fn process_tentative_event(&mut self, ev: FloodEvent) -> Option<MatcherEvent> {
        if ev.is_region() {
            self.do_region_shrinking(ev.index())
        } else {
            self.do_look_at_node_event(ev.index())
        }
    }

    /// `run_until_next_mwpm_notification`: `None` once the queue runs dry.
    pub(crate) fn run_until_next_mwpm_notification(&mut self) -> Option<MatcherEvent> {
        loop {
            let ev = self.dequeue_valid()?;
            if let Some(m) = self.process_tentative_event(ev) {
                return Some(m);
            }
        }
    }

    /// `GraphFillRegion::cleanup_shell_area`.
    pub(crate) fn cleanup_shell_area(&mut self, region: u32) {
        for k in 0..self.regions.items[region as usize].shell_area.len() {
            let n = self.regions.items[region as usize].shell_area[k];
            self.nodes[n as usize].reset();
        }
    }

    /// `GraphFillRegion::add_match`.
    pub(crate) fn add_match(&mut self, region: u32, other: u32, edge: CompressedEdge) {
        self.regions.items[region as usize].matched = Match { region: other, edge };
        self.regions.items[other as usize].matched = Match { region, edge: edge.reversed() };
    }

    /// `rhs <= lhs` in PyMatching's region order: whether `ancestor` is `region` or one of its
    /// blossom ancestors.
    pub fn region_is_within(&self, region: u32, ancestor: u32) -> bool {
        let mut r = region;
        while r != NONE && r != ancestor {
            r = self.regions.items[r as usize].blossom_parent;
        }
        r == ancestor
    }

    /// `GraphFillRegion::tree_equal`, which is PyMatching's region `==`: the same alternating-tree
    /// node, radius and shell area, and children equal edge by edge, recursively.
    pub fn tree_equal(&self, a: u32, b: u32) -> bool {
        let (x, y) = (&self.regions.items[a as usize], &self.regions.items[b as usize]);
        if x.alt_tree_node != y.alt_tree_node || x.radius != y.radius || x.blossom_children.len() != y.blossom_children.len() || x.shell_area != y.shell_area {
            return false;
        }
        x.blossom_children.iter().zip(&y.blossom_children).all(|(c, d)| c.edge == d.edge && self.tree_equal(c.region, d.region))
    }

    /// `DetectorNode::compute_local_radius_at_time_bounded_by_region` (for drawing).
    pub fn compute_local_radius_at_time_bounded_by_region(&self, n: u32, time: i64, bounding_region: u32) -> i64 {
        let d = &self.nodes[n as usize];
        if d.region_that_arrived == NONE {
            return 0;
        }
        // `*region_that_arrived > bounding_region`: the arriving region strictly contains it.
        let gt = |a: u32, b: u32| a != b && self.region_is_within(b, a);
        if gt(d.region_that_arrived, bounding_region) {
            return 0;
        }
        let mut container = d.region_that_arrived;
        let mut container_radius = 0i64;
        loop {
            if container == NONE {
                break;
            }
            if gt(container, bounding_region) {
                break;
            }
            container_radius += self.regions.items[container as usize].radius.get_distance_at_time(time);
            // (PyMatching's region `==`: `tree_equal`, not identity.)
            if self.tree_equal(container, bounding_region) {
                break;
            }
            container = self.regions.items[container as usize].blossom_parent;
        }
        container_radius - d.radius_of_arrival
    }

    /// `DetectorNode::compute_stitch_radius_at_time_bounded_by_region_towards_neighbor` (for
    /// drawing): where along the node's `k`th edge ownership changes, or `None` inside one
    /// source's area.
    pub fn compute_stitch_radius_at_time_bounded_by_region_towards_neighbor(&self, n: u32, time: i64, bounding_region: u32, k: usize) -> Option<f32> {
        let neighbor = self.g.neighbors(n)[k];
        let max_w = self.g.neighbor_weights(n)[k] as i64;
        let r1 = self.compute_local_radius_at_time_bounded_by_region(n, time, bounding_region);
        if neighbor == NONE {
            return Some(max_w.min(r1) as u32 as f32);
        }
        let r2 = self.compute_local_radius_at_time_bounded_by_region(neighbor, time, bounding_region);
        let (d, e) = (&self.nodes[n as usize], &self.nodes[neighbor as usize]);
        if r1 + r2 < max_w || e.region_that_arrived_top != d.region_that_arrived_top {
            return Some(r1 as u32 as f32);
        }
        let gt = |a: u32, b: u32| a != b && self.region_is_within(b, a);
        if r1 == max_w && gt(e.region_that_arrived, d.region_that_arrived) {
            return Some(max_w as f32);
        }
        if d.reached_from_source == e.reached_from_source {
            return None;
        }
        let growth_past_collision = (r1 + r2 - max_w) as f32;
        let growth_past_collision_of_r1 = growth_past_collision / 2.0;
        Some(r1 as f32 - growth_past_collision_of_r1)
    }
}

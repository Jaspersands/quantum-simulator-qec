//! Growth on the detector graph, and the events it raises.
//!
//! A node's local radius is how far the regions owning it have grown past it:
//! the top region's radius plus a constant kept in `wrapped`. Regions that grow
//! toward each other across an edge of weight w meet when the two local radii
//! sum to w. Weights are even and every region starts at radius 0 at time 0, so
//! the local radius of every node in a tree has the parity of the time, and two
//! growing regions always meet at an integer time.

use crate::dem_decoder::DecodeError;

use super::state::{AltNode, CEdge, NodeState, Radius, Region, BOUNDARY, NOBODY, NONE, NO_TIME};
use super::tracker::Item;
use super::Solver;

/// Guard against a bug looping forever: events allowed per defect, and the
/// floor for small shots. Real shots take about five events a defect (at most
/// 5.7 measured, on SD6 memory circuits up to d = 11 with 2,247 defects), so
/// this is thousands of times any real need, and a loop is caught in seconds.
const EVENTS_PER_DEFECT: u64 = 1 << 16;
const EVENT_FLOOR: u64 = 1 << 20;

#[derive(Clone, Copy, Debug)]
pub(crate) enum NodeEvent {
    /// The region at node `from` reaches the empty node `to` across edge `e`.
    Arrive { from: u32, to: u32, e: usize },
    /// The regions at `v` and `u` meet across edge `e`.
    Collide { v: u32, u: u32, e: usize },
    /// The region at `v` reaches the boundary across edge `e`.
    Boundary { v: u32, e: usize },
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum RegionEvent {
    /// A shrinking region gives up the last node it reached.
    Leave(u32),
    /// A shrinking blossom reaches radius zero.
    Shatter(u32),
    /// A shrinking trivial region reaches radius zero.
    Implode(u32),
}

impl<'a> Solver<'a> {
    pub(crate) fn reset(&mut self) {
        for &v in &self.s.touched {
            self.s.nodes[v as usize] = NodeState::EMPTY;
            self.s.queued[v as usize] = NO_TIME;
            self.s.dirty[v as usize] = false;
        }
        self.s.touched.clear();
        // Keep the regions' and tree nodes' vectors for the next shot's.
        let s = &mut *self.s;
        // Region 0, nobody's, stays.
        for r in s.regions.drain(super::state::NOBODY + 1..) {
            if r.shell.capacity() > 0 {
                let mut v = r.shell;
                v.clear();
                s.spare_u32.push(v);
            }
            if r.children.capacity() > 0 {
                let mut v = r.children;
                v.clear();
                s.spare_cycles.push(v);
            }
        }
        for a in s.alt.drain(..) {
            if a.children.capacity() > 0 {
                let mut v = a.children;
                v.clear();
                s.spare_u32.push(v);
            }
        }
        self.s.queue.clear();
        self.s.now = 0;
        self.s.events = 0;
    }

    fn touch(&mut self, v: u32) {
        if !self.s.dirty[v as usize] {
            self.s.dirty[v as usize] = true;
            self.s.touched.push(v);
        }
    }

    pub(crate) fn new_region(&mut self, region: Region) -> u32 {
        self.s.regions.push(region);
        (self.s.regions.len() - 1) as u32
    }

    pub(crate) fn new_alt(&mut self, node: AltNode) -> u32 {
        self.s.alt.push(node);
        (self.s.alt.len() - 1) as u32
    }

    pub(crate) fn local_radius(&self, v: u32) -> i64 {
        let n = &self.s.nodes[v as usize];
        self.s.regions[n.top as usize].radius.at(self.s.now) + n.wrapped
    }

    fn slope_at(&self, v: u32) -> i64 {
        self.s.regions[self.s.nodes[v as usize].top as usize].radius.slope
    }

    /// Every node owned by `r` or by any region inside it, in a buffer the
    /// caller hands back with `done_with_nodes` once it has walked them.
    pub(crate) fn nodes_under(&mut self, r: u32) -> Vec<u32> {
        let mut out = std::mem::take(&mut self.s.under);
        let mut stack = std::mem::take(&mut self.s.under_stack);
        out.clear();
        stack.clear();
        stack.push(r);
        while let Some(x) = stack.pop() {
            let reg = &self.s.regions[x as usize];
            out.extend_from_slice(&reg.shell);
            stack.extend(reg.children.iter().map(|c| c.0));
        }
        self.s.under_stack = stack;
        out
    }

    pub(crate) fn done_with_nodes(&mut self, buf: Vec<u32>) {
        self.s.under = buf;
    }

    fn create_trivial(&mut self, d: u32) {
        let none = CEdge { a: NONE, b: NONE, obs: 0 };
        let mut shell = self.s.spare_u32();
        shell.push(d);
        let r = self.new_region(Region {
            radius: Radius { y0: -self.s.now, slope: 1 },
            blossom_parent: NONE,
            children: Vec::new(),
            shell,
            tree: NONE,
            matched: None,
            queued: NO_TIME,
            dead: false,
        });
        let children = self.s.spare_u32();
        let a = self.new_alt(AltNode {
            inner: NONE,
            outer: r,
            inner_to_outer: none,
            parent: NONE,
            parent_edge: none,
            children,
            alive: true,
            mark: 0,
        });
        self.s.regions[r as usize].tree = a;
        self.touch(d);
        let n = &mut self.s.nodes[d as usize];
        n.region = r;
        n.top = r;
        n.source = d;
        n.obs = 0;
        n.wrapped = 0;
        n.own = r;
    }

    /// The next thing that happens across one of `v`'s edges, seen from `v`:
    /// the earliest, and among equals the first edge in adjacency order.
    ///
    /// Every edge is one formula. An empty node, and the boundary, count as a
    /// radius of zero that does not grow; two sides whose radii grow at a
    /// combined rate of 1 or 2 meet when they cover the edge's weight. What the
    /// winning edge's event is (reaching an empty node, being reached, a
    /// collision, the boundary) is decided only for the winner.
    pub(crate) fn next_node_event(&self, v: u32) -> Option<(i64, NodeEvent)> {
        let now = self.s.now;
        let nodes = &self.s.nodes;
        let regions = &self.s.regions;
        let nv = &nodes[v as usize];
        let (lv, sv) = if nv.top == NONE {
            (0, 0)
        } else {
            let r = &regions[nv.top as usize].radius;
            (r.at(now) + nv.wrapped, r.slope)
        };
        // A node of a shrinking region meets nothing: no neighbour grows faster
        // than its region gives ground.
        if sv < 0 {
            return None;
        }
        let range = self.g.edges(v);
        let first = range.start;
        let to = &self.g.to[range.clone()];
        let w = &self.s.w[range];
        let mut best_dt = i64::MAX;
        let mut best_k = usize::MAX;
        // The boundary reads the node past the graph's last, which is never
        // reached; an empty node reads nobody's region. So every edge is the
        // same loads and arithmetic, and the one branch left is the rarely
        // taken "earlier than the best so far".
        let beyond = nodes.len() - 1;
        for (k, (&u, &wt)) in to.iter().zip(w).enumerate() {
            let nu = &nodes[if u == BOUNDARY { beyond } else { u as usize }];
            let r = &regions[if nu.top == NONE { NOBODY } else { nu.top as usize }].radius;
            let lu = r.at(now) + nu.wrapped;
            let rate = sv + r.slope;
            let gap = wt - lv - lu;
            // One region, or two empty nodes, meet nothing; nor do sides whose
            // radii do not close.
            let meets = nu.top != nv.top && rate > 0;
            debug_assert!(!meets || (gap >= 0 && gap % rate == 0), "gap {gap} at rate {rate}");
            // Slopes are -1, 0 or 1, so a positive rate is 1 or 2.
            let dt = if meets { gap.max(0) >> u32::from(rate == 2) } else { i64::MAX };
            if dt < best_dt {
                best_dt = dt;
                best_k = k;
            }
        }
        if best_k == usize::MAX {
            return None;
        }
        let (u, e) = (to[best_k], first + best_k);
        let ev = if u == BOUNDARY {
            NodeEvent::Boundary { v, e }
        } else if nodes[u as usize].top == NONE {
            NodeEvent::Arrive { from: v, to: u, e }
        } else if nv.top == NONE {
            NodeEvent::Arrive { from: u, to: v, e }
        } else {
            NodeEvent::Collide { v, u, e }
        };
        Some((now + best_dt, ev))
    }

    pub(crate) fn look_at_node(&mut self, v: u32) {
        if let Some((t, _)) = self.next_node_event(v) {
            self.schedule_node(v, t);
        }
    }

    fn schedule_node(&mut self, v: u32, t: i64) {
        if t < self.s.queued[v as usize] {
            self.touch(v);
            self.s.queued[v as usize] = t;
            self.s.queue.push(t, Item::Node(v));
        }
    }

    /// The next thing that happens to a shrinking top region.
    pub(crate) fn next_region_event(&self, r: u32) -> Option<(i64, RegionEvent)> {
        let reg = &self.s.regions[r as usize];
        if reg.dead || reg.blossom_parent != NONE || reg.radius.slope >= 0 {
            return None;
        }
        let now = self.s.now;
        // A trivial region keeps its own defect, which is always shell[0].
        let keep = usize::from(reg.children.is_empty());
        if reg.shell.len() > keep {
            let v = *reg.shell.last().expect("non-empty shell");
            Some((now + self.local_radius(v).max(0), RegionEvent::Leave(r)))
        } else if reg.children.is_empty() {
            Some((now + reg.radius.at(now).max(0), RegionEvent::Implode(r)))
        } else {
            Some((now + reg.radius.at(now).max(0), RegionEvent::Shatter(r)))
        }
    }

    pub(crate) fn look_at_region(&mut self, r: u32) {
        if let Some((t, _)) = self.next_region_event(r) {
            self.schedule_region(r, t);
        }
    }

    fn schedule_region(&mut self, r: u32, t: i64) {
        let reg = &mut self.s.regions[r as usize];
        if t < reg.queued {
            reg.queued = t;
            self.s.queue.push(t, Item::Region(r));
        }
    }

    fn arrive(&mut self, from: u32, to: u32, e: usize) {
        let nf = self.s.nodes[from as usize];
        let top = nf.top;
        let y = self.s.regions[top as usize].radius.at(self.s.now);
        let obs = nf.obs ^ self.g.obs[e];
        self.touch(to);
        let n = &mut self.s.nodes[to as usize];
        n.region = top;
        n.top = top;
        n.source = nf.source;
        n.obs = obs;
        n.wrapped = -y;
        self.s.regions[top as usize].shell.push(to);
        self.look_at_node(to);
        self.look_at_node(from);
    }

    pub(crate) fn leave(&mut self, r: u32) {
        let v = self.s.regions[r as usize].shell.pop().expect("a leave event has a node to give up");
        let n = self.s.nodes[v as usize];
        // Its reminder and touch flag live apart, and stay as they are.
        self.s.nodes[v as usize] = NodeState { own: n.own, ..NodeState::EMPTY };
        self.look_at_node(v);
        self.look_at_region(r);
    }

    /// Change a top region's growth rate now, keeping its radius continuous.
    pub(crate) fn set_slope(&mut self, r: u32, slope: i64) {
        let now = self.s.now;
        let rad = &mut self.s.regions[r as usize].radius;
        let y = rad.at(now);
        *rad = Radius { y0: y - slope * now, slope };
    }

    /// Recompute every reminder a region's change can have moved.
    pub(crate) fn reschedule(&mut self, r: u32) {
        let nodes = self.nodes_under(r);
        for &v in &nodes {
            self.look_at_node(v);
        }
        self.done_with_nodes(nodes);
        self.look_at_region(r);
    }

    fn dispatch_node(&mut self, ev: NodeEvent) {
        match ev {
            NodeEvent::Arrive { from, to, e } => self.arrive(from, to, e),
            NodeEvent::Collide { v, u, e } => {
                let (nv, nu) = (self.s.nodes[v as usize], self.s.nodes[u as usize]);
                let edge = CEdge { a: nv.source, b: nu.source, obs: nv.obs ^ nu.obs ^ self.g.obs[e] };
                self.on_collide(nv.top, nu.top, edge);
            }
            NodeEvent::Boundary { v, e } => {
                let nv = self.s.nodes[v as usize];
                self.on_boundary(nv.top, CEdge { a: nv.source, b: BOUNDARY, obs: nv.obs ^ self.g.obs[e] });
            }
        }
    }

    fn dispatch_region(&mut self, ev: RegionEvent) {
        match ev {
            RegionEvent::Leave(r) => self.leave(r),
            RegionEvent::Shatter(b) => self.shatter(b),
            RegionEvent::Implode(r) => self.implode(r),
        }
    }

    pub(crate) fn run(&mut self, defects: &[u32], check: bool) -> Result<(), DecodeError> {
        debug_assert!(defects.windows(2).all(|w| w[0] < w[1]), "defects must be sorted and distinct");
        let limit = EVENT_FLOOR.max(EVENTS_PER_DEFECT * defects.len() as u64);
        for &d in defects {
            self.create_trivial(d);
        }
        for &d in defects {
            self.look_at_node(d);
        }
        while let Some((t, item)) = self.s.queue.pop() {
            debug_assert!(t >= self.s.now, "time went backwards");
            self.s.now = t;
            self.s.events += 1;
            if self.s.events > limit {
                return Err(DecodeError::MatcherDeclined);
            }
            match item {
                Item::Node(v) => {
                    if self.s.queued[v as usize] != t {
                        continue;
                    }
                    self.s.queued[v as usize] = NO_TIME;
                    match self.next_node_event(v) {
                        Some((te, ev)) if te == t => {
                            self.dispatch_node(ev);
                            self.look_at_node(v);
                        }
                        Some((te, _)) => self.schedule_node(v, te),
                        None => {}
                    }
                }
                Item::Region(r) => {
                    if self.s.regions[r as usize].queued != t {
                        continue;
                    }
                    self.s.regions[r as usize].queued = NO_TIME;
                    match self.next_region_event(r) {
                        Some((te, ev)) if te == t => {
                            self.dispatch_region(ev);
                            self.look_at_region(r);
                        }
                        Some((te, _)) => self.schedule_region(r, te),
                        None => {}
                    }
                }
            }
            if check {
                self.check_invariants();
            }
        }
        if self.s.alt.iter().any(|a| a.alive) {
            return Err(DecodeError::Unmatchable);
        }
        Ok(())
    }

    /// Dual feasibility, in full. Every radius is non-negative, every node's top
    /// is its region's outermost ancestor, and no two regions (nor a region and
    /// an empty node, nor a region and the boundary) overlap across an edge.
    pub(crate) fn check_invariants(&self) {
        let now = self.s.now;
        for (i, r) in self.s.regions.iter().enumerate() {
            if !r.dead {
                assert!(r.radius.at(now) >= 0, "region {i} has radius {} at {now}", r.radius.at(now));
            }
        }
        for v in 0..self.g.num_nodes as u32 {
            let n = self.s.nodes[v as usize];
            if n.top == NONE {
                continue;
            }
            let mut r = n.region;
            while self.s.regions[r as usize].blossom_parent != NONE {
                r = self.s.regions[r as usize].blossom_parent;
            }
            assert_eq!(r, n.top, "node {v}: recorded top is not its region's outermost ancestor");
            let lv = self.local_radius(v);
            assert!(lv >= 0, "node {v}: local radius {lv}");
            for e in self.g.edges(v) {
                let (u, w) = (self.g.to[e], self.s.w[e]);
                if u == BOUNDARY {
                    assert!(lv <= w, "node {v} overlaps the boundary");
                    continue;
                }
                let nu = self.s.nodes[u as usize];
                if nu.top == NONE {
                    assert!(lv <= w, "node {v} overlaps empty node {u}");
                } else if nu.top != n.top {
                    assert!(lv + self.local_radius(u) <= w, "nodes {v} and {u} overlap");
                }
            }
        }
    }
}

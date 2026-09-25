# Sparse Matcher Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Exact minimum-weight perfect matching that scales to thousands of defects per shot: Higgott and Gidney's sparse blossom, in Rust, verified against our dense blossom, brute force and PyMatching, and made the default decoder of the general path.

**Architecture:** A new `src/sparse/` module. Regions grow on the detector graph with even-integer weights, events are driven by a lazy reminder queue, and alternating trees and blossoms work over regions. `DemDecoder` keeps its API, decodes with the sparse matcher by default, and keeps the dense blossom as `decode_dense`, the oracle. Both matchers solve the identical integer problem, so their total integer weights must be *equal*, which is the sharpest test available.

**Tech Stack:** Rust 2021 with no new crates (`std::thread` for parallel shots), PyO3 in `.venv`, the wasm32 build, Node for site tests.

**Spec:** `docs/superpowers/specs/2026-09-25-sparse-matcher-design.md`

## Global Constraints

- Exactness is non-negotiable. The sparse matcher's integer weight must equal the dense matcher's on every shot both can decode.
- No fallback anywhere. Failures are `DecodeError`s that callers count.
- No new crates.
- The old per-code decoders, and every number they produce, stay untouched.
- Branch `feature/willow-data`. Commit after every task with the trailer `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Tests run as `cargo test --release --no-default-features`. `cargo build --release` (the Python cfg) and the wasm32 build must compile.

---

## File map

| File | Responsibility |
|---|---|
| `src/dem_decoder.rs` | Shared `int_weight` (even integers); `Prediction.iweight`; `merged_edges`; `DemDecoder` holds a `SparseGraph` and a `Scratch`; `decode` uses the sparse matcher, `decode_dense` the dense one |
| `src/sparse/mod.rs` | Module doc; `Solver`; `SparseGraph::decode`, `decode_checked` |
| `src/sparse/graph.rs` | `SparseGraph::from_edges`: compressed adjacency, even weights, observables, boundary |
| `src/sparse/state.rs` | `CEdge`, `Radius`, `Region`, `AltNode`, `NodeState`, `Scratch`, sentinels |
| `src/sparse/tracker.rs` | The reminder heap |
| `src/sparse/flooder.rs` | Local radii, node and region events, arrive and leave, rates, the event loop, invariant checks |
| `src/sparse/matcher.rs` | The seven cases; blossom formation and shattering; dissolving trees |
| `src/sparse/extract.rs` | Matched blossoms shattered recursively; observables; weight |
| `src/sparse/tests.rs` | Brute force, random-graph and surface-code differential tests, large shots, timing |
| `src/py_api.rs` | `decode_b8` and `decode_b8_own` take `threads`, one `Scratch` per thread |
| `tools/xcheck.py` | Speed measured single-threaded; full re-run; reference regenerated |
| `js/sweep-config.js`, `js/sections/xcheck.js` | SD6 shot counts from new measured cost |
| `README.md` | "An exact sparse matcher" section |

---

### Task 1: Shared integer weights, `iweight`, merged edges

**Files:** Modify `src/dem_decoder.rs`.

**Interfaces:**
- Produces: `pub const HALF_SCALE: f64 = 524_288.0`; `pub fn int_weight(wf: f64) -> i64`, returning `2 * round(wf · HALF_SCALE)`; `Prediction { observables: u64, weight: f64, iweight: i64 }`; `pub(crate) fn merged_edges(dem: &Dem) -> Result<(Vec<(u32, u32, f64, u64)>, usize), String>`, where `v == num_detectors` is the boundary and the `usize` counts conflicts.

- [ ] **Step 1: Test.** Add to `dem_decoder::tests`:

```rust
    #[test]
    fn integer_weights_are_even_and_reported() {
        assert_eq!(int_weight(1.0) % 2, 0);
        assert_eq!(int_weight(0.0), 0);
        let dec = DemDecoder::new(&Dem::parse("error(0.1) D0 D1 L0\nerror(0.2) D1").unwrap()).unwrap();
        let pred = dec.decode_dense(&[0, 1]).unwrap();
        assert_eq!(pred.iweight, int_weight((0.9f64 / 0.1).ln()));
        assert_eq!(pred.iweight % 2, 0);
    }
```

- [ ] **Step 2: Implement.**
  - Add `HALF_SCALE` and `int_weight` beside `SCALE`, with a doc comment: even, so that regions growing from both ends of an edge meet at integer times; shared, so that both matchers solve the same integer problem.
  - Add `pub iweight: i64` to `Prediction`.
  - Move the edge-merging loop out of `DemDecoder::new` into `merged_edges`, which returns the sorted `(u, v, p, obs)` list plus the conflict count, and checks `0 < p <= 0.5`.
  - `new` builds `adj` from that list, with `w = int_weight(wf)`.
  - Rename the current `decode` to `decode_dense`, keeping its duplicate-cancelling preamble, and accumulate `iweight += d` for each matched pair's integer distance `d`.
  - Add a `decode` that, for now, forwards to `decode_dense`; Task 3 switches it.
  - Every `Prediction { .. }` literal gains `iweight`.

- [ ] **Step 3:** Run `cargo test --release --no-default-features`. Everything passes, 60 tests including the new one.

- [ ] **Step 4:** Commit: `refactor(decoder): even integer weights shared by both matchers; report the integer weight`.

---

### Task 2: The sparse matcher

**Files:** Create `src/sparse/{mod,graph,state,tracker,flooder,matcher,extract,tests}.rs`. Modify `src/lib.rs` (`pub mod sparse;`) and `src/dem_decoder.rs` (add `pub fn graph(&self) -> &SparseGraph` and the `sparse` field, built in `new` from `merged_edges`).

**Interfaces:**
- Produces:
  - `SparseGraph::from_edges(num_nodes, &[(u32, u32, f64, u64)]) -> SparseGraph`
  - `SparseGraph::decode(&self, &mut Scratch, &[u32]) -> Result<Prediction, DecodeError>` (defects sorted and distinct)
  - `#[cfg(test)] SparseGraph::decode_checked(..)`, the same with invariants checked after every event
  - `Scratch::new(&SparseGraph)`
  - `DemDecoder::graph()`

- [ ] **Step 1: `src/sparse/mod.rs`**

```rust
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
    /// Decode one shot. `defects` must be sorted and distinct.
    pub fn decode(&self, scratch: &mut Scratch, defects: &[u32]) -> Result<Prediction, DecodeError> {
        let mut solver = Solver { g: self, s: scratch };
        solver.reset();
        solver.run(defects, false)?;
        Ok(solver.extract())
    }

    /// The same, checking the dual's feasibility after every event.
    #[cfg(test)]
    pub(crate) fn decode_checked(&self, scratch: &mut Scratch, defects: &[u32]) -> Result<Prediction, DecodeError> {
        let mut solver = Solver { g: self, s: scratch };
        solver.reset();
        solver.run(defects, true)?;
        Ok(solver.extract())
    }
}
```

- [ ] **Step 2: `src/sparse/graph.rs`**

```rust
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
```

- [ ] **Step 3: `src/sparse/state.rs`**

```rust
//! The state of one decode: regions, alternating-tree nodes, detector nodes.

use super::graph::SparseGraph;
use super::tracker::Tracker;

pub(crate) const NONE: u32 = u32::MAX;
/// The boundary, as an edge target and as a match partner.
pub(crate) const BOUNDARY: u32 = u32::MAX - 1;
pub(crate) const NO_TIME: i64 = i64::MAX;

/// A path between two defects, or a defect and the boundary, compressed to its
/// endpoints and the observables it crosses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct CEdge {
    pub a: u32,
    pub b: u32,
    pub obs: u64,
}

impl CEdge {
    pub fn rev(self) -> CEdge {
        CEdge { a: self.b, b: self.a, obs: self.obs }
    }
}

/// A radius moving at slope +1, 0 or -1: y(t) = y0 + slope · t.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Radius {
    pub y0: i64,
    pub slope: i64,
}

impl Radius {
    pub fn at(&self, t: i64) -> i64 {
        self.y0 + self.slope * t
    }
}

pub(crate) struct Region {
    pub radius: Radius,
    pub blossom_parent: u32,
    /// Blossom cycle: child j, and the edge from child j to child j + 1.
    pub children: Vec<(u32, CEdge)>,
    /// Nodes this region owns directly, in arrival order.
    pub shell: Vec<u32>,
    /// Alternating-tree node, or NONE.
    pub tree: u32,
    /// Partner (a region or BOUNDARY), and the edge from this region to it.
    pub matched: Option<(u32, CEdge)>,
    pub queued: i64,
    pub dead: bool,
}

pub(crate) struct AltNode {
    /// Shrinking region; NONE at a root.
    pub inner: u32,
    /// Growing region.
    pub outer: u32,
    pub inner_to_outer: CEdge,
    pub parent: u32,
    /// From the parent's outer region to this node's inner region.
    pub parent_edge: CEdge,
    pub children: Vec<u32>,
    pub alive: bool,
}

#[derive(Clone, Copy)]
pub(crate) struct NodeState {
    /// The region that arrived here, or NONE.
    pub region: u32,
    /// Its outermost blossom ancestor: the region whose radius moves.
    pub top: u32,
    /// The defect the growth came from.
    pub source: u32,
    /// Observables crossed on the way from `source`.
    pub obs: u64,
    /// Local radius = the top's radius + `wrapped`.
    pub wrapped: i64,
    /// For a defect, its own trivial region.
    pub own: u32,
    pub queued: i64,
    pub dirty: bool,
}

impl NodeState {
    pub const EMPTY: NodeState = NodeState {
        region: NONE,
        top: NONE,
        source: NONE,
        obs: 0,
        wrapped: 0,
        own: NONE,
        queued: NO_TIME,
        dirty: false,
    };
}

/// Reusable workspace for decoding shots on one graph, one per thread.
pub struct Scratch {
    pub(crate) nodes: Vec<NodeState>,
    pub(crate) touched: Vec<u32>,
    pub(crate) regions: Vec<Region>,
    pub(crate) alt: Vec<AltNode>,
    pub(crate) queue: Tracker,
    pub(crate) now: i64,
    pub(crate) events: u64,
}

impl Scratch {
    pub fn new(graph: &SparseGraph) -> Scratch {
        Scratch {
            nodes: vec![NodeState::EMPTY; graph.num_nodes],
            touched: Vec::new(),
            regions: Vec::new(),
            alt: Vec::new(),
            queue: Tracker::default(),
            now: 0,
            events: 0,
        }
    }
}
```

- [ ] **Step 4: `src/sparse/tracker.rs`**

```rust
//! The event queue: reminders to look at a node or a region at a given time.
//!
//! Only the earliest reminder per entity is queued, and a reminder made stale
//! by a change of growth rate is not removed: it fires, the flooder recomputes,
//! and finds nothing due. That is the paper's tracker, and it keeps the queue
//! small.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum Item {
    Node(u32),
    Region(u32),
}

#[derive(Default)]
pub(crate) struct Tracker {
    heap: BinaryHeap<Reverse<(i64, Item)>>,
}

impl Tracker {
    pub fn clear(&mut self) {
        self.heap.clear();
    }
    pub fn push(&mut self, t: i64, item: Item) {
        self.heap.push(Reverse((t, item)));
    }
    pub fn pop(&mut self) -> Option<(i64, Item)> {
        self.heap.pop().map(|Reverse(x)| x)
    }
}
```

- [ ] **Step 5: `src/sparse/flooder.rs`**

```rust
//! Growth on the detector graph, and the events it raises.
//!
//! A node's local radius is how far the regions owning it have grown past it:
//! the top region's radius plus a constant kept in `wrapped`. Regions that grow
//! toward each other across an edge of weight w meet when the two local radii
//! sum to w. Weights are even and every region starts at radius 0 at time 0, so
//! the local radius of every node in a tree has the parity of the time, and two
//! growing regions always meet at an integer time.

use crate::dem_decoder::DecodeError;

use super::state::{AltNode, CEdge, NodeState, Radius, Region, BOUNDARY, NONE, NO_TIME};
use super::tracker::Item;
use super::Solver;

/// Guard against a bug looping forever; far beyond any real shot.
const EVENT_LIMIT: u64 = 1 << 34;

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
        }
        self.s.touched.clear();
        self.s.regions.clear();
        self.s.alt.clear();
        self.s.queue.clear();
        self.s.now = 0;
        self.s.events = 0;
    }

    fn touch(&mut self, v: u32) {
        let n = &mut self.s.nodes[v as usize];
        if !n.dirty {
            n.dirty = true;
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

    /// Every node owned by `r` or by any region inside it.
    pub(crate) fn nodes_under(&self, r: u32) -> Vec<u32> {
        let mut out = Vec::new();
        let mut stack = vec![r];
        while let Some(x) = stack.pop() {
            let reg = &self.s.regions[x as usize];
            out.extend_from_slice(&reg.shell);
            stack.extend(reg.children.iter().map(|c| c.0));
        }
        out
    }

    fn create_trivial(&mut self, d: u32) {
        let none = CEdge { a: NONE, b: NONE, obs: 0 };
        let r = self.new_region(Region {
            radius: Radius { y0: -self.s.now, slope: 1 },
            blossom_parent: NONE,
            children: Vec::new(),
            shell: vec![d],
            tree: NONE,
            matched: None,
            queued: NO_TIME,
            dead: false,
        });
        let a = self.new_alt(AltNode {
            inner: NONE,
            outer: r,
            inner_to_outer: none,
            parent: NONE,
            parent_edge: none,
            children: Vec::new(),
            alive: true,
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

    /// The next thing that happens across one of `v`'s edges, seen from `v`.
    pub(crate) fn next_node_event(&self, v: u32) -> Option<(i64, NodeEvent)> {
        let now = self.s.now;
        let nv = self.s.nodes[v as usize];
        let v_owned = nv.top != NONE;
        let (lv, sv) = if v_owned { (self.local_radius(v), self.slope_at(v)) } else { (0, 0) };
        let mut best: Option<(i64, NodeEvent)> = None;
        let mut offer = |t: i64, ev: NodeEvent| {
            if best.map_or(true, |(bt, _)| t < bt) {
                best = Some((t, ev));
            }
        };
        for e in self.g.edges(v) {
            let u = self.g.to[e];
            let w = self.g.w[e];
            if u == BOUNDARY {
                if v_owned && sv > 0 {
                    offer(now + (w - lv).max(0), NodeEvent::Boundary { v, e });
                }
                continue;
            }
            let nu = self.s.nodes[u as usize];
            match (v_owned, nu.top != NONE) {
                (false, false) => {}
                (true, false) => {
                    if sv > 0 {
                        offer(now + (w - lv).max(0), NodeEvent::Arrive { from: v, to: u, e });
                    }
                }
                (false, true) => {
                    if self.slope_at(u) > 0 {
                        offer(now + (w - self.local_radius(u)).max(0), NodeEvent::Arrive { from: u, to: v, e });
                    }
                }
                (true, true) => {
                    if nu.top == nv.top {
                        continue;
                    }
                    let rate = sv + self.slope_at(u);
                    if rate <= 0 {
                        continue;
                    }
                    let gap = w - lv - self.local_radius(u);
                    debug_assert!(gap >= 0 && gap % rate == 0, "gap {gap} at rate {rate}");
                    offer(now + gap.max(0) / rate, NodeEvent::Collide { v, u, e });
                }
            }
        }
        best
    }

    pub(crate) fn look_at_node(&mut self, v: u32) {
        if let Some((t, _)) = self.next_node_event(v) {
            self.schedule_node(v, t);
        }
    }

    fn schedule_node(&mut self, v: u32, t: i64) {
        if t < self.s.nodes[v as usize].queued {
            self.touch(v);
            self.s.nodes[v as usize].queued = t;
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
        self.s.nodes[v as usize] = NodeState { own: n.own, queued: n.queued, dirty: true, ..NodeState::EMPTY };
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
        for v in self.nodes_under(r) {
            self.look_at_node(v);
        }
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
            if self.s.events > EVENT_LIMIT {
                return Err(DecodeError::MatcherDeclined);
            }
            match item {
                Item::Node(v) => {
                    if self.s.nodes[v as usize].queued != t {
                        continue;
                    }
                    self.s.nodes[v as usize].queued = NO_TIME;
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
                let (u, w) = (self.g.to[e], self.g.w[e]);
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
```

- [ ] **Step 6: `src/sparse/matcher.rs`**

```rust
//! What an event means: the seven cases of the paper, over alternating trees
//! of regions.

use super::state::{AltNode, CEdge, Radius, Region, BOUNDARY, NONE, NO_TIME};
use super::Solver;

impl<'a> Solver<'a> {
    fn set_match(&mut self, r: u32, partner: u32, e: CEdge) {
        self.s.regions[r as usize].matched = Some((partner, e));
        if partner != BOUNDARY {
            self.s.regions[partner as usize].matched = Some((r, e.rev()));
        }
    }

    fn path_to_root(&self, mut n: u32) -> Vec<u32> {
        let mut out = vec![n];
        while self.s.alt[n as usize].parent != NONE {
            n = self.s.alt[n as usize].parent;
            out.push(n);
        }
        out
    }

    /// Index of the blossom child of `b` that holds `defect`.
    pub(crate) fn child_index(&self, b: u32, children: &[(u32, CEdge)], defect: u32) -> usize {
        let mut r = self.s.nodes[defect as usize].own;
        while self.s.regions[r as usize].blossom_parent != b {
            r = self.s.regions[r as usize].blossom_parent;
        }
        children.iter().position(|c| c.0 == r).expect("the defect lies in a child of the blossom")
    }

    /// Two top regions touch; `e` runs from `r1` to `r2`.
    pub(crate) fn on_collide(&mut self, r1: u32, r2: u32, e: CEdge) {
        let n1 = self.s.regions[r1 as usize].tree;
        let n2 = self.s.regions[r2 as usize].tree;
        if n1 == NONE {
            assert!(n2 != NONE, "a collision needs a growing region");
            return self.on_collide(r2, r1, e.rev());
        }
        if n2 == NONE {
            let (partner, _) = self.s.regions[r2 as usize].matched.expect("a region outside every tree is matched");
            if partner == BOUNDARY {
                self.take_from_boundary(n1, r2, e)
            } else {
                self.grow_tree(n1, r2, e)
            }
        } else if self.path_to_root(n1).last() == self.path_to_root(n2).last() {
            self.form_blossom(n1, n2, e)
        } else {
            self.augment(n1, n2, e)
        }
    }

    /// (f) A growing region reaches the boundary.
    pub(crate) fn on_boundary(&mut self, r: u32, e: CEdge) {
        let n = self.s.regions[r as usize].tree;
        assert!(n != NONE, "only a growing region reaches the boundary");
        self.set_match(r, BOUNDARY, e);
        self.dissolve(n);
    }

    /// (a) A growing region reaches a matched pair; both join its tree.
    fn grow_tree(&mut self, n: u32, m: u32, e: CEdge) {
        let (partner, me) = self.s.regions[m as usize].matched.take().expect("matched");
        self.s.regions[partner as usize].matched = None;
        let child = self.new_alt(AltNode {
            inner: m,
            outer: partner,
            inner_to_outer: me,
            parent: n,
            parent_edge: e,
            children: Vec::new(),
            alive: true,
        });
        self.s.alt[n as usize].children.push(child);
        self.s.regions[m as usize].tree = child;
        self.s.regions[partner as usize].tree = child;
        self.set_slope(m, -1);
        self.set_slope(partner, 1);
        self.reschedule(m);
        self.reschedule(partner);
    }

    /// (g) A growing region reaches a region matched to the boundary. It takes
    /// that region as its partner, and its tree dissolves.
    fn take_from_boundary(&mut self, n: u32, m: u32, e: CEdge) {
        let r = self.s.alt[n as usize].outer;
        self.set_match(r, m, e);
        self.dissolve(n);
    }

    /// (b) Two trees meet: augment along both paths to their roots.
    fn augment(&mut self, n1: u32, n2: u32, e: CEdge) {
        let (r1, r2) = (self.s.alt[n1 as usize].outer, self.s.alt[n2 as usize].outer);
        self.set_match(r1, r2, e);
        self.dissolve(n1);
        self.dissolve(n2);
    }

    /// Turn the tree holding `n` into matched pairs. `n`'s outer region has just
    /// been matched outside the tree. Along the path to the root each inner
    /// region matches its parent's outer region; every other node's pair matches
    /// along its own edge.
    fn dissolve(&mut self, n: u32) {
        let path = self.path_to_root(n);
        for &m in &path {
            let (parent, inner, pe) = {
                let a = &self.s.alt[m as usize];
                (a.parent, a.inner, a.parent_edge)
            };
            if parent != NONE {
                let pouter = self.s.alt[parent as usize].outer;
                self.set_match(inner, pouter, pe.rev());
            }
        }
        let root = *path.last().expect("path has a root");
        let mut stack = vec![root];
        let mut regions = Vec::new();
        while let Some(x) = stack.pop() {
            let (inner, outer, io, children) = {
                let a = &self.s.alt[x as usize];
                (a.inner, a.outer, a.inner_to_outer, a.children.clone())
            };
            stack.extend(children);
            if inner != NONE {
                if !path.contains(&x) {
                    self.set_match(inner, outer, io);
                }
                regions.push(inner);
            }
            regions.push(outer);
            self.s.alt[x as usize].alive = false;
        }
        for &r in &regions {
            self.s.regions[r as usize].tree = NONE;
            self.set_slope(r, 0);
        }
        for r in regions {
            self.reschedule(r);
        }
    }

    /// (c) Two growing regions of one tree meet. The cycle through their common
    /// ancestor becomes a blossom, which takes the ancestor's outer place.
    pub(crate) fn form_blossom(&mut self, n1: u32, n2: u32, e: CEdge) {
        let path1 = self.path_to_root(n1);
        let path2 = self.path_to_root(n2);
        let a = *path2.iter().find(|x| path1.contains(x)).expect("same tree");
        let p1: Vec<u32> = path1.iter().copied().take_while(|&x| x != a).collect();
        let p2: Vec<u32> = path2.iter().copied().take_while(|&x| x != a).collect();

        let mut cycle: Vec<(u32, CEdge)> = Vec::new();
        let mut cur = self.s.alt[a as usize].outer;
        for &m in p1.iter().rev() {
            let (pe, inner, io, outer) = {
                let am = &self.s.alt[m as usize];
                (am.parent_edge, am.inner, am.inner_to_outer, am.outer)
            };
            cycle.push((cur, pe));
            cycle.push((inner, io));
            cur = outer;
        }
        cycle.push((cur, e));
        for &m in &p2 {
            let (outer, io, inner, pe) = {
                let am = &self.s.alt[m as usize];
                (am.outer, am.inner_to_outer, am.inner, am.parent_edge)
            };
            cycle.push((outer, io.rev()));
            cycle.push((inner, pe.rev()));
        }

        let b = self.new_region(Region {
            radius: Radius { y0: -self.s.now, slope: 1 },
            blossom_parent: NONE,
            children: cycle.clone(),
            shell: Vec::new(),
            tree: a,
            matched: None,
            queued: NO_TIME,
            dead: false,
        });
        for &(c, _) in &cycle {
            self.enclose(c, b);
        }

        let on: Vec<u32> = p1.iter().chain(p2.iter()).copied().collect();
        let mut orphans = Vec::new();
        for &m in &on {
            let children = self.s.alt[m as usize].children.clone();
            orphans.extend(children.into_iter().filter(|c| !on.contains(c)));
            self.s.alt[m as usize].alive = false;
        }
        let kept: Vec<u32> = self.s.alt[a as usize].children.iter().copied().filter(|c| !on.contains(c)).collect();
        self.s.alt[a as usize].children = kept;
        for o in orphans {
            self.s.alt[o as usize].parent = a;
            self.s.alt[a as usize].children.push(o);
        }
        self.s.alt[a as usize].outer = b;
        self.reschedule(b);
    }

    /// Put top region `c` inside blossom `b`, which starts at radius zero now,
    /// keeping every node's local radius continuous.
    fn enclose(&mut self, c: u32, b: u32) {
        let now = self.s.now;
        for v in self.nodes_under(c) {
            let l = self.local_radius(v);
            let n = &mut self.s.nodes[v as usize];
            n.top = b;
            n.wrapped = l;
        }
        let reg = &mut self.s.regions[c as usize];
        let y = reg.radius.at(now);
        reg.radius = Radius { y0: y, slope: 0 };
        reg.blossom_parent = b;
        reg.tree = NONE;
        reg.matched = None;
    }

    /// (d) A shrinking blossom reaches radius zero and comes apart. The even
    /// path around its cycle, between the children its two tree edges attach
    /// to, rejoins the tree; the rest of the cycle pairs off.
    pub(crate) fn shatter(&mut self, b: u32) {
        let n = self.s.regions[b as usize].tree;
        let (p, pe, iot) = {
            let a = &self.s.alt[n as usize];
            (a.parent, a.parent_edge, a.inner_to_outer)
        };
        let children = self.s.regions[b as usize].children.clone();
        let k = children.len();
        let i_in = self.child_index(b, &children, pe.b);
        let i_out = self.child_index(b, &children, iot.a);

        let now = self.s.now;
        for &(c, _) in &children {
            let yc = self.s.regions[c as usize].radius.at(now);
            for v in self.nodes_under(c) {
                let l = self.local_radius(v);
                let nd = &mut self.s.nodes[v as usize];
                nd.top = c;
                nd.wrapped = l - yc;
            }
        }
        for &(c, _) in &children {
            self.s.regions[c as usize].blossom_parent = NONE;
        }
        self.s.regions[b as usize].dead = true;
        self.s.regions[b as usize].children.clear();

        let forward = ((i_out + k - i_in) % k) % 2 == 0;
        let mut path = vec![i_in];
        let mut edges = Vec::new();
        let mut j = i_in;
        while j != i_out {
            if forward {
                edges.push(children[j].1);
                j = (j + 1) % k;
            } else {
                let prev = (j + k - 1) % k;
                edges.push(children[prev].1.rev());
                j = prev;
            }
            path.push(j);
        }
        let mut on_path = vec![false; k];
        for &i in &path {
            on_path[i] = true;
        }

        let mut changed = Vec::new();
        let mut j = if forward { (i_out + 1) % k } else { (i_in + 1) % k };
        while !on_path[j] {
            let next = (j + 1) % k;
            let (c1, c2, ce) = (children[j].0, children[next].0, children[j].1);
            self.set_match(c1, c2, ce);
            self.set_slope(c1, 0);
            self.set_slope(c2, 0);
            changed.push(c1);
            changed.push(c2);
            j = (next + 1) % k;
        }

        let q: Vec<u32> = path.iter().map(|&i| children[i].0).collect();
        let (mut parent, mut pedge, mut first) = (p, pe, NONE);
        let mut i = 0;
        while i + 1 < q.len() {
            let node = self.new_alt(AltNode {
                inner: q[i],
                outer: q[i + 1],
                inner_to_outer: edges[i],
                parent,
                parent_edge: pedge,
                children: Vec::new(),
                alive: true,
            });
            if parent == p {
                first = node;
            } else {
                self.s.alt[parent as usize].children.push(node);
            }
            self.s.regions[q[i] as usize].tree = node;
            self.s.regions[q[i + 1] as usize].tree = node;
            self.set_slope(q[i], -1);
            self.set_slope(q[i + 1], 1);
            changed.push(q[i]);
            changed.push(q[i + 1]);
            parent = node;
            pedge = edges[i + 1];
            i += 2;
        }
        let last = *q.last().expect("the path has an end");
        {
            let a = &mut self.s.alt[n as usize];
            a.inner = last;
            a.parent = parent;
            a.parent_edge = pedge;
        }
        if parent != p {
            self.s.alt[parent as usize].children.push(n);
        }
        if first != NONE {
            let pc = &mut self.s.alt[p as usize].children;
            let slot = pc.iter().position(|&c| c == n).expect("n is a child of p");
            pc[slot] = first;
        }
        self.s.regions[last as usize].tree = n;
        self.set_slope(last, -1);
        changed.push(last);
        for r in changed {
            self.reschedule(r);
        }
    }

    /// (e) A trivial shrinking region reaches radius zero. Its tree parent and
    /// child now touch through it, and the three form a blossom.
    pub(crate) fn implode(&mut self, r: u32) {
        let n = self.s.regions[r as usize].tree;
        let (p, pe, iot) = {
            let a = &self.s.alt[n as usize];
            (a.parent, a.parent_edge, a.inner_to_outer)
        };
        let e = CEdge { a: pe.a, b: iot.b, obs: pe.obs ^ iot.obs };
        self.form_blossom(p, n, e);
    }
}
```

- [ ] **Step 7: `src/sparse/extract.rs`**

```rust
//! The matching, read off the final regions.
//!
//! Every top region is matched to another or to the boundary. A matched
//! blossom comes apart recursively: the child holding the matched edge's
//! endpoint takes the match, and the rest of the cycle pairs off along its own
//! edges. The prediction is the XOR of every matched edge's observables. The
//! integer weight is the sum of every region's radius, which by LP duality is
//! the weight of the matching.

use crate::dem_decoder::{Prediction, SCALE};

use super::state::{BOUNDARY, NONE};
use super::Solver;

impl<'a> Solver<'a> {
    pub(crate) fn extract(&self) -> Prediction {
        let now = self.s.now;
        let mut iweight = 0i64;
        let mut observables = 0u64;
        for (r, reg) in self.s.regions.iter().enumerate() {
            if reg.dead {
                continue;
            }
            iweight += reg.radius.at(now);
            if reg.blossom_parent != NONE {
                continue;
            }
            let (partner, e) = reg.matched.expect("every top region is matched at the end");
            if partner == BOUNDARY || (r as u32) < partner {
                observables ^= e.obs;
                self.expand(r as u32, e.a, &mut observables);
                if partner != BOUNDARY {
                    self.expand(partner, e.b, &mut observables);
                }
            }
        }
        Prediction { observables, weight: iweight as f64 / SCALE, iweight }
    }

    fn expand(&self, r: u32, a: u32, observables: &mut u64) {
        let children = &self.s.regions[r as usize].children;
        if children.is_empty() {
            return;
        }
        let k = children.len();
        let i = self.child_index(r, children, a);
        let mut j = (i + 1) % k;
        while j != i {
            let next = (j + 1) % k;
            let e = children[j].1;
            *observables ^= e.obs;
            self.expand(children[j].0, e.a, observables);
            self.expand(children[next].0, e.b, observables);
            j = (next + 1) % k;
        }
        self.expand(children[i].0, a, observables);
    }
}
```

- [ ] **Step 8: `src/sparse/tests.rs`** (layers 1, 2a and 5)

```rust
use super::*;
use crate::dem::Dem;
use crate::dem_decoder::{DecodeError, DemDecoder};
use crate::surface_code::Xorshift;
use std::collections::HashSet;

type TestEdge = (u32, Option<u32>, f64, u64);

fn decoder(text: &str) -> DemDecoder {
    DemDecoder::new(&Dem::parse(text).unwrap()).unwrap()
}

/// Exhaustive minimum over edge subsets with the given boundary, in the same
/// integer weights both matchers use.
fn brute_force(edges: &[TestEdge], defects: &[u32]) -> Option<(i64, Vec<u64>)> {
    let target: u64 = defects.iter().fold(0, |acc, &d| acc ^ (1 << d));
    let mut best: Option<i64> = None;
    let mut masks = Vec::new();
    for mask in 0u32..(1 << edges.len()) {
        let (mut syn, mut w, mut obs) = (0u64, 0i64, 0u64);
        for (i, e) in edges.iter().enumerate() {
            if (mask >> i) & 1 == 1 {
                syn ^= 1 << e.0;
                if let Some(b) = e.1 {
                    syn ^= 1 << b;
                }
                w += crate::dem_decoder::int_weight(((1.0 - e.2) / e.2).ln());
                obs ^= e.3;
            }
        }
        if syn != target {
            continue;
        }
        match best {
            Some(b) if w > b => {}
            Some(b) if w == b => masks.push(obs),
            _ => {
                best = Some(w);
                masks = vec![obs];
            }
        }
    }
    best.map(|b| (b, masks))
}

fn random_small(rng: &mut Xorshift) -> (usize, Vec<TestEdge>, String) {
    let nd = 3 + (rng.next_u64() % 6) as usize;
    let mut edges: Vec<TestEdge> = Vec::new();
    let mut seen = HashSet::new();
    let target = (2 * nd).min(14);
    let mut tries = 0;
    while edges.len() < target && tries < 1000 {
        tries += 1;
        let a = (rng.next_u64() % nd as u64) as u32;
        let b = if rng.next_u64() % 3 == 0 { None } else { Some((rng.next_u64() % nd as u64) as u32) };
        if b == Some(a) {
            continue;
        }
        let key = match b {
            Some(b) => (a.min(b), Some(a.max(b))),
            None => (a, None),
        };
        if !seen.insert(key) {
            continue;
        }
        edges.push((key.0, key.1, 0.01 + 0.3 * rng.next_f64(), rng.next_u64() % 2));
    }
    let mut text = String::new();
    for d in 0..nd {
        text.push_str(&format!("detector D{d}\n"));
    }
    for &(a, b, p, obs) in &edges {
        text.push_str(&format!("error({p}) D{a}"));
        if let Some(b) = b {
            text.push_str(&format!(" D{b}"));
        }
        if obs == 1 {
            text.push_str(" L0");
        }
        text.push('\n');
    }
    (nd, edges, text)
}

#[test]
fn matches_brute_force_on_small_graphs() {
    let mut rng = Xorshift::new(17);
    let (mut compared, mut unmatchable) = (0, 0);
    for trial in 0..3000 {
        let (nd, edges, text) = random_small(&mut rng);
        let dec = decoder(&text);
        let defects: Vec<u32> = (0..nd as u32).filter(|_| rng.next_u64() % 2 == 0).collect();
        let mut scratch = Scratch::new(dec.graph());
        match (brute_force(&edges, &defects), dec.graph().decode_checked(&mut scratch, &defects)) {
            (None, Err(DecodeError::Unmatchable)) => unmatchable += 1,
            (Some((w, masks)), Ok(pred)) => {
                assert_eq!(pred.iweight, w, "trial {trial}");
                assert!(masks.contains(&pred.observables), "trial {trial}: {} not in {masks:?}", pred.observables);
                compared += usize::from(!defects.is_empty());
            }
            (b, s) => panic!("trial {trial}: brute force {b:?}, sparse {s:?}\n{text}defects {defects:?}"),
        }
    }
    assert!(compared > 1500 && unmatchable > 50, "{compared} compared, {unmatchable} unmatchable");
}

fn random_graph(rng: &mut Xorshift, nodes: usize, boundary: f64) -> String {
    let mut seen = HashSet::new();
    let mut text = String::new();
    for d in 0..nodes {
        text.push_str(&format!("detector D{d}\n"));
    }
    let obs = |rng: &mut Xorshift| if rng.next_u64() % 4 == 0 { " L0" } else { "" };
    for u in 0..nodes {
        for _ in 0..2 {
            let v = (rng.next_u64() % nodes as u64) as usize;
            if v == u || !seen.insert((u.min(v), u.max(v))) {
                continue;
            }
            let p = 0.001 + 0.3 * rng.next_f64();
            let o = obs(rng);
            text.push_str(&format!("error({p}) D{u} D{v}{o}\n"));
        }
        if rng.next_f64() < boundary {
            let p = 0.001 + 0.3 * rng.next_f64();
            let o = obs(rng);
            text.push_str(&format!("error({p}) D{u}{o}\n"));
        }
    }
    text
}

#[test]
fn agrees_with_the_dense_matcher_on_random_graphs() {
    let mut rng = Xorshift::new(2024);
    let (mut compared, mut unmatchable) = (0, 0);
    for trial in 0..20_000 {
        let nodes = 4 + (rng.next_u64() % 40) as usize;
        let boundary = [0.0, 0.1, 0.5][(rng.next_u64() % 3) as usize];
        let text = random_graph(&mut rng, nodes, boundary);
        let dec = decoder(&text);
        let defects: Vec<u32> = (0..nodes as u32).filter(|_| rng.next_u64() % 3 == 0).collect();
        let mut scratch = Scratch::new(dec.graph());
        match (dec.graph().decode_checked(&mut scratch, &defects), dec.decode_dense(&defects)) {
            (Ok(s), Ok(d)) => {
                assert_eq!(s.iweight, d.iweight, "trial {trial}\n{text}defects {defects:?}");
                compared += 1;
            }
            (Err(DecodeError::Unmatchable), Err(DecodeError::Unmatchable)) => unmatchable += 1,
            (s, d) => panic!("trial {trial}: sparse {s:?}, dense {d:?}\n{text}defects {defects:?}"),
        }
    }
    assert!(compared > 10_000 && unmatchable > 100, "{compared} compared, {unmatchable} unmatchable");
}

#[test]
fn one_scratch_decodes_many_shots() {
    let mut rng = Xorshift::new(5);
    let text = random_graph(&mut rng, 30, 0.5);
    let dec = decoder(&text);
    let mut scratch = Scratch::new(dec.graph());
    for _ in 0..500 {
        let defects: Vec<u32> = (0..30).filter(|_| rng.next_u64() % 3 == 0).collect();
        let a = dec.graph().decode(&mut scratch, &defects).map(|p| p.iweight);
        let b = dec.graph().decode(&mut Scratch::new(dec.graph()), &defects).map(|p| p.iweight);
        assert_eq!(a, b);
    }
}
```

- [ ] **Step 9: Wire up.** In `src/lib.rs`, add `pub mod sparse;` after `pub mod dem_decoder;`. In `DemDecoder`:
  - add the field `sparse: crate::sparse::SparseGraph`;
  - build it in `new` with `SparseGraph::from_edges(nd, &edges)`;
  - add `pub fn graph(&self) -> &SparseGraph`.

- [ ] **Step 10: Run.** `cargo test --release --no-default-features sparse::`

Expected: 3 passed. Debugging method, if not:
- **A brute-force or dense mismatch prints the graph and the defects.** Shrink it by deleting error lines while it still fails, then trace the events with `decode_checked` and a `println!` of each dispatched event.
- **An invariant panic names the node or region.** The event just before it is the bug.
- **Never loosen an assertion.** Record anything found for the README.

- [ ] **Step 11:** Commit: `feat(sparse): sparse blossom: exact matching that grows regions on the detector graph`.

---

### Task 3: Sparse by default; surface-code differential; single faults; big shots

**Files:** Modify `src/dem_decoder.rs` and `src/sparse/tests.rs`.

**Interfaces:**
- Produces:
  - `DemDecoder::decode` uses the sparse matcher, through `scratch: RefCell<Scratch>`;
  - `DemDecoder::decode_dense` is the oracle;
  - `decode_bools` is unchanged.

- [ ] **Step 1: Switch.** `DemDecoder` gains `scratch: std::cell::RefCell<Scratch>`, built in `new`. `decode` keeps the duplicate-cancelling preamble, then returns `self.sparse.decode(&mut self.scratch.borrow_mut(), defects)`. Update the dense-decoder tests:
  - `too_many_defects_is_an_error_not_a_fallback`: `decode_dense(&defects)` is `Err(TooManyDefects(257))`, and `decode(&defects)` is `Ok`.
  - `parallel_edges_merge_like_pymatching`: through `decode_dense`, the weight holds to 1e-12; through `decode`, to 1e-5, since the sparse weight is `iweight / SCALE`.
  - Every other existing test runs through `decode` (sparse) unchanged, and a copy of `matches_brute_force_on_small_graphs` runs through `decode_dense`.

- [ ] **Step 2: Layer 2b** (append to `src/sparse/tests.rs`)

```rust
#[test]
fn agrees_with_the_dense_matcher_on_surface_code_shots() {
    use crate::circuit::Basis;
    use crate::frame_sampler::FrameSampler;
    use crate::memory::{generate, CodeKind, NoiseModel};
    let mut rng = Xorshift::new(99);
    let (mut compared, mut obs_differ) = (0, 0);
    for kind in [CodeKind::Rotated, CodeKind::Xzzx] {
        for basis in [Basis::Z, Basis::X] {
            for d in [3usize, 5, 7] {
                for &p in &[0.002, 0.006, 0.012] {
                    for noise in [NoiseModel::Sd6 { p }, NoiseModel::Current { p, eta: 0.5 }] {
                        let c = generate(kind, d, d, noise, basis).unwrap();
                        let dec = DemDecoder::new(&Dem::from_circuit(&c).unwrap()).unwrap();
                        let sampler = FrameSampler::new(&c).unwrap();
                        let mut scratch = Scratch::new(dec.graph());
                        for shot_i in 0..60 {
                            let shot = sampler.sample(&mut rng);
                            let defects: Vec<u32> =
                                shot.detectors.iter().enumerate().filter(|x| *x.1).map(|x| x.0 as u32).collect();
                            let dense = match dec.decode_dense(&defects) {
                                Ok(x) => x,
                                Err(DecodeError::TooManyDefects(_)) => continue,
                                Err(e) => panic!("{e:?}"),
                            };
                            let sparse = if shot_i < 5 && d <= 5 {
                                dec.graph().decode_checked(&mut scratch, &defects)
                            } else {
                                dec.graph().decode(&mut scratch, &defects)
                            }
                            .unwrap();
                            assert_eq!(sparse.iweight, dense.iweight, "{kind:?} {basis:?} {noise:?} d = {d}: {defects:?}");
                            compared += 1;
                            obs_differ += usize::from(sparse.observables != dense.observables);
                        }
                    }
                }
            }
        }
    }
    println!("{compared} shots, equal weight on all; {obs_differ} tie-broken differently");
    assert!(compared > 3000, "{compared}");
}

#[test]
fn decodes_shots_far_beyond_the_dense_limit() {
    use crate::circuit::Basis;
    use crate::frame_sampler::FrameSampler;
    use crate::memory::{generate, CodeKind, NoiseModel};
    let c = generate(CodeKind::Rotated, 7, 60, NoiseModel::Sd6 { p: 0.006 }, Basis::Z).unwrap();
    let dec = DemDecoder::new(&Dem::from_circuit(&c).unwrap()).unwrap();
    let sampler = FrameSampler::new(&c).unwrap();
    let mut rng = Xorshift::new(7);
    let (mut most, mut failures) = (0, 0);
    for _ in 0..100 {
        let shot = sampler.sample(&mut rng);
        most = most.max(shot.detectors.iter().filter(|&&b| b).count());
        let pred = dec.decode_bools(&shot.detectors).expect("the sparse matcher has no defect ceiling");
        failures += ((pred.observables ^ shot.observables) & 1) as usize;
    }
    assert!(most > 256, "the test should exceed the dense limit; most was {most}");
    assert!(failures < 100, "every shot failed");
}

#[test]
#[ignore] // timing, for the README and the site
fn timing() {
    use crate::circuit::Basis;
    use crate::frame_sampler::FrameSampler;
    use crate::memory::{generate, CodeKind, NoiseModel};
    for d in [3usize, 5, 7, 9] {
        for &p in &[0.003, 0.006] {
            let c = generate(CodeKind::Rotated, d, d, NoiseModel::Sd6 { p }, Basis::Z).unwrap();
            let dec = DemDecoder::new(&Dem::from_circuit(&c).unwrap()).unwrap();
            let sampler = FrameSampler::new(&c).unwrap();
            let mut rng = Xorshift::new(1);
            let shots: Vec<Vec<u32>> = (0..2000)
                .map(|_| sampler.sample(&mut rng).detectors.iter().enumerate().filter(|x| *x.1).map(|x| x.0 as u32).collect())
                .collect();
            let mut scratch = Scratch::new(dec.graph());
            let t = std::time::Instant::now();
            for s in &shots {
                dec.graph().decode(&mut scratch, s).unwrap();
            }
            let sparse_us = t.elapsed().as_secs_f64() * 1e6 / shots.len() as f64;
            let t = std::time::Instant::now();
            let mut dense_n = 0;
            for s in shots.iter().take(300) {
                if dec.decode_dense(s).is_ok() {
                    dense_n += 1;
                }
            }
            let dense_us = t.elapsed().as_secs_f64() * 1e6 / dense_n.max(1) as f64;
            println!("d = {d}, p = {p}: sparse {sparse_us:.1} us/shot, dense {dense_us:.1} us/shot");
        }
    }
}
```

- [ ] **Step 3: Run.**
  - `cargo test --release --no-default-features`: everything passes. That includes `memory::tests::every_single_fault_is_corrected_d3_d5` (layer 3) and the equivalence tests, now running through the sparse matcher.
  - `cargo test --release --no-default-features -- --ignored every_single_fault_is_corrected_d7 --nocapture`: 0 failures on all 8 circuits.
  - `cargo test --release --no-default-features sparse::tests::timing -- --ignored --nocapture`: keep the output for the README.

- [ ] **Step 4:** Commit: `feat(decoder): the sparse matcher decodes by default; the dense blossom stays as its oracle`.

---

### Task 4: Parallel shots in Python; the harness re-run

**Files:** Modify `src/py_api.rs` and `tools/xcheck.py`.

**Interfaces:**
- Produces: `decode_b8(dem_text, packed, num_shots, threads=1)` and `decode_b8_own(circuit_text, packed, num_shots, threads=1)`. `threads=0` means every core.

- [ ] **Step 1: `decode_packed`** takes `threads: usize` and decodes chunks of shots on scoped threads, one `Scratch` per thread, sharing `decoder.graph()`:

```rust
fn decode_packed<'py>(py: Python<'py>, dem: &Dem, packed: &[u8], num_shots: usize, threads: usize) -> PyResult<Decoded<'py>> {
    let decoder = DemDecoder::new(dem).map_err(err)?;
    let graph = decoder.graph();
    let nd = dem.num_detectors;
    let stride = nd.div_ceil(8);
    if packed.len() != stride * num_shots {
        return Err(err(format!("{} bytes is not {num_shots} shots of {nd} detectors", packed.len())));
    }
    let threads = if threads == 0 { std::thread::available_parallelism().map_or(1, |n| n.get()) } else { threads }
        .clamp(1, num_shots.max(1));
    let chunk = num_shots.div_ceil(threads);
    let start = Instant::now();
    let parts: Vec<Vec<(u64, f64)>> = py.allow_threads(|| {
        std::thread::scope(|scope| {
            let handles: Vec<_> = (0..threads)
                .map(|t| {
                    let (lo, hi) = (t * chunk, ((t + 1) * chunk).min(num_shots));
                    scope.spawn(move || {
                        let mut scratch = Scratch::new(graph);
                        let mut defects = Vec::new();
                        let mut out = Vec::with_capacity(hi.saturating_sub(lo));
                        for s in lo..hi {
                            defects.clear();
                            let row = &packed[s * stride..(s + 1) * stride];
                            for i in 0..nd {
                                if (row[i / 8] >> (i % 8)) & 1 == 1 {
                                    defects.push(i as u32);
                                }
                            }
                            out.push(match graph.decode(&mut scratch, &defects) {
                                Ok(p) => (p.observables, p.weight),
                                Err(_) => (u64::MAX, f64::NAN),
                            });
                        }
                        out
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().expect("decoder thread panicked")).collect()
        })
    });
    let seconds = start.elapsed().as_secs_f64();
    let mut preds = Vec::with_capacity(8 * num_shots);
    let mut weights = Vec::with_capacity(8 * num_shots);
    let mut errors = 0usize;
    for (o, w) in parts.into_iter().flatten() {
        errors += usize::from(o == u64::MAX);
        preds.extend_from_slice(&o.to_le_bytes());
        weights.extend_from_slice(&w.to_le_bytes());
    }
    Ok((PyBytes::new_bound(py, &preds), PyBytes::new_bound(py, &weights), errors, seconds))
}
```

The `#[pyo3(signature = (..., threads=1))]` on both `decode_b8` and `decode_b8_own` forward `threads`. Imports: `crate::sparse::Scratch`.

- [ ] **Step 2: Build and re-run the harness.**

```bash
VIRTUAL_ENV=$PWD/.venv CARGO_TARGET_DIR=target-py .venv/bin/maturin develop --release
.venv/bin/python tools/xcheck.py --quick
.venv/bin/python tools/xcheck.py > <scratchpad>/xcheck-full.txt 2>&1   # background
```

Expected: "ALL CHECKS PASSED". Check 2's "ours" columns are now the sparse matcher, since `decode_b8` goes through `DemDecoder`, and every disagreement is still a tie. Check 4 reports single-threaded µs per shot. Copy the run into `data/xcheck/report.txt`.

- [ ] **Step 3:** Commit `src/py_api.rs`, `tools/xcheck.py` and `data/xcheck/` with the message `feat(python): parallel decoding; the cross-check re-run with the sparse matcher`.

---

### Task 5: WASM, site, README, verification

**Files:** Modify `stabilizer_qec.wasm`, `js/sweep-config.js`, `js/sections/xcheck.js` and `README.md`.

- [ ] **Step 1: Rebuild the WASM** (`cargo build --release --target wasm32-unknown-unknown --no-default-features` and copy). Then measure SD6 cost in Node, with the Task 11 Step 2 command from the A plan, at d = 3, 5, 7, 9 and p = 0.3%–0.7%.

- [ ] **Step 2: Shot counts.**
  - Set `SWEEP_RUNS[NOISE.SD6]` so the SD6 sweep costs about the circuit-level sweep's wall-clock (~80–110 s), and rewrite its comment with the new per-shot costs.
  - Raise `LIVE_RUNS` in `js/sections/xcheck.js` to what finishes in under about 30 s.
  - If SD6 shots rose by 3× or more, re-run `node tools/sweep.mjs 3 0` and `node tools/sweep.mjs 3 1` into `data/sweeps/`, and update the README's SD6 rows and section 07's sentence from the new means.

- [ ] **Step 3: README.** Add a section, "An exact sparse matcher", after "Checked against Stim and PyMatching":
  - what sparse blossom is, in one paragraph;
  - the five verification layers and their counts (brute-force trials, random graphs, surface-code shots with equal integer weight, single faults at d ≤ 7, PyMatching ties);
  - the timing table, sparse against dense against PyMatching;
  - anything the checks caught, written up like the defect sections.

  Update "Repository Structure" with `src/sparse/`.

- [ ] **Step 4: Verify everything.**

```bash
cargo test --release --no-default-features
cargo test --release --no-default-features -- --ignored every_single_fault_is_corrected_d7 old_and_new_paths_agree_d7
cargo build --release
cargo build --release --target wasm32-unknown-unknown --no-default-features && cmp target/wasm32-unknown-unknown/release/stabilizer_qec.wasm stabilizer_qec.wasm
node tools/site-tests.mjs && node tools/contrast.mjs
.venv/bin/python tools/xcheck.py --quick
```

Then check the page in headless Chrome: Figure 8 finishes, with the decoding rows faster than before, and there are no console errors.

- [ ] **Step 5: Review.** Run the `code-review` skill at `high` over `master...feature/willow-data`. Fix confirmed findings, re-run Step 4, and commit.

//! The state of one decode: regions, alternating-tree nodes, detector nodes.

use super::graph::SparseGraph;
use super::tracker::{RadixHeap, Tracker};

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

/// Region 0 is nobody's: a dead region of radius zero that does not grow,
/// which an empty node's neighbours read in place of a branch.
pub(crate) const NOBODY: usize = 0;

/// The `top` of an empty node: nobody's region, so that reading an empty node's radius needs
/// no branch.
pub(crate) const NO_TOP: u32 = NOBODY as u32;

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
    /// Marks the node as one of a set being built (see `Scratch::stamp`):
    /// equal to the current stamp while it is in the set.
    pub mark: u64,
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
}

// Two nodes to a 64-byte cache line: the hot loop reads a neighbour's node
// for every edge it scans.
const _: () = assert!(std::mem::size_of::<NodeState>() == 32);

impl NodeState {
    pub const EMPTY: NodeState = NodeState {
        region: NONE,
        top: NO_TOP,
        source: NONE,
        obs: 0,
        wrapped: 0,
        own: NONE,
    };
}

/// Reusable workspace for decoding shots on one graph, one per thread.
pub struct Scratch {
    pub(crate) nodes: Vec<NodeState>,
    /// Per node: the time of its queued reminder (NO_TIME if none), and
    /// whether this decode has touched it. Kept apart from `nodes`, which the
    /// hot loop reads, so a node there is 32 bytes, two to a cache line.
    pub(crate) queued: Vec<i64>,
    pub(crate) dirty: Vec<bool>,
    pub(crate) touched: Vec<u32>,
    pub(crate) regions: Vec<Region>,
    /// How many regions grow (positive slope). With none, nothing can meet anything, which
    /// `next_node_event` knows without a scan.
    pub(crate) growing: u32,
    pub(crate) alt: Vec<AltNode>,
    pub(crate) queue: Tracker,
    pub(crate) now: i64,
    pub(crate) events: u64,
    /// This decode's edge weights, per half-edge. They are the graph's, except
    /// while correlated matching's second pass has lowered some; `undo` holds
    /// the old values until they are restored.
    pub(crate) w: Vec<i64>,
    pub(crate) undo: Vec<(u32, i64)>,
    /// The last decode's matched pairs: two defects, or a defect and BOUNDARY.
    pub(crate) pairs: Vec<(u32, u32)>,
    /// Shortest-path search: per node the distance (NO_TIME when unreached)
    /// and the (node, edge) it was reached by; the nodes reached; the queue.
    pub(crate) dist: Vec<i64>,
    pub(crate) pred: Vec<(u32, u32)>,
    pub(crate) seen: Vec<u32>,
    pub(crate) heap: RadixHeap<u32>,
    /// The edge set being built: a flag per edge, and every edge whose flag
    /// was raised at some point.
    pub(crate) flipped: Vec<bool>,
    pub(crate) edge_set: Vec<u32>,
    /// The last stamp handed out. A tree node is in a set being built while
    /// its `mark` equals the set's stamp, so a set is made in one pass and
    /// tested in constant time, and forgetting it costs nothing.
    pub(crate) stamp: u64,
    /// Buffers kept between events and shots, so the hot loop does not
    /// allocate: paths to a tree root, a depth-first stack, regions to
    /// reschedule, and the nodes under a region.
    pub(crate) path_a: Vec<u32>,
    pub(crate) path_b: Vec<u32>,
    pub(crate) stack: Vec<u32>,
    pub(crate) touched_regions: Vec<u32>,
    pub(crate) under: Vec<u32>,
    pub(crate) under_stack: Vec<u32>,
    /// Emptied vectors from the last shot's regions and tree nodes, reused by
    /// the next shot's. Every new shell and child list draws from here, so
    /// the pool never holds more than one shot's worth.
    pub(crate) spare_u32: Vec<Vec<u32>>,
    pub(crate) spare_cycles: Vec<Vec<(u32, CEdge)>>,
}

impl Scratch {
    pub fn new(graph: &SparseGraph) -> Scratch {
        Scratch {
            // One more than the graph's: the boundary's, never reached, which
            // the hot loop reads for a boundary edge in place of a branch.
            nodes: vec![NodeState::EMPTY; graph.num_nodes + 1],
            queued: vec![NO_TIME; graph.num_nodes],
            dirty: vec![false; graph.num_nodes],
            touched: Vec::new(),
            regions: vec![Region {
                radius: Radius { y0: 0, slope: 0 },
                blossom_parent: NONE,
                children: Vec::new(),
                shell: Vec::new(),
                tree: NONE,
                matched: None,
                queued: NO_TIME,
                dead: true,
            }],
            growing: 0,
            alt: Vec::new(),
            queue: Tracker::default(),
            now: 0,
            events: 0,
            w: graph.w.clone(),
            undo: Vec::new(),
            pairs: Vec::new(),
            // Sized on first use (`ensure_paths`): plain matching never traces.
            dist: Vec::new(),
            pred: Vec::new(),
            seen: Vec::new(),
            heap: RadixHeap::default(),
            flipped: Vec::new(),
            edge_set: Vec::new(),
            stamp: 0,
            path_a: Vec::new(),
            path_b: Vec::new(),
            stack: Vec::new(),
            touched_regions: Vec::new(),
            under: Vec::new(),
            under_stack: Vec::new(),
            spare_u32: Vec::new(),
            spare_cycles: Vec::new(),
        }
    }

    /// Size the shortest-path state for `graph`, once: tracing a matching's
    /// paths and correlated matching need it, plain matching does not.
    pub(crate) fn ensure_paths(&mut self, graph: &SparseGraph) {
        if self.dist.len() != graph.num_nodes {
            self.dist = vec![NO_TIME; graph.num_nodes];
            self.pred = vec![(NONE, NONE); graph.num_nodes];
        }
        if self.flipped.len() != graph.num_edges() {
            self.flipped = vec![false; graph.num_edges()];
        }
    }

    /// A fresh stamp: no tree node carries it yet.
    pub(crate) fn next_stamp(&mut self) -> u64 {
        self.stamp += 1;
        self.stamp
    }

    /// An empty vector, reusing a spare one's allocation when there is one.
    pub(crate) fn spare_u32(&mut self) -> Vec<u32> {
        self.spare_u32.pop().unwrap_or_default()
    }

    /// Keep an emptied vector's allocation for later.
    pub(crate) fn keep_u32(&mut self, mut v: Vec<u32>) {
        if v.capacity() > 0 {
            v.clear();
            self.spare_u32.push(v);
        }
    }
}

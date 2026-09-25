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

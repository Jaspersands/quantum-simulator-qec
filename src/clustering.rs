//! Collision Clustering (Barber et al., arXiv 2309.05558, Riverlane's ASIC decoder) and the
//! Local Clustering Decoder (Ziad et al., arXiv 2411.10343, Riverlane's adaptive FPGA decoder),
//! from their papers, on the matching graph.
//!
//! WHY THIS EXISTS
//! ---------------
//! Both are union-find decoders laid out for hardware, and both papers claim their clusters are
//! union-find's. Collision Clustering keeps no cluster boundary at all: each detection event has a
//! ball whose radius grows while its cluster is odd and off the boundary (the Cluster Growth
//! Stack), and two clusters merge when some pair of their events' balls overlap, r_i + r_j ≥
//! d(i, j), every pair compared each round (the Parent table and its Parity and Boundary
//! registers). The Local Clustering Decoder instead gives every vertex a radius and a cluster
//! index, and runs a controller through growing, merging, picking and syncing stages whose
//! kernels each vertex runs on its own and its neighbours' fields alone; its adaptivity engine
//! pre-grows the edges a heralded leakage event makes likely, so the clusters form around them
//! first.
//!
//! A step of growth is half an edge, as union-find grows: unweighted, every edge is 2 steps long
//! (the papers' hardware), and weighted, each edge's weight ln((1 − p)/p) is scaled to at most
//! `resolution` steps (Collision Clustering's distances are then Dijkstra's, and the Local
//! Clustering Decoder's vertices grow to their longest edge, its `wmax`). Both decoders find
//! their correction by peeling a spanning forest of the edges their clusters cover. The
//! clusters are round-synchronous union-find's (Delfosse and Nickerson's), and the two decoders
//! cover the same edges, so without pre-grown edges their predictions are the same shot for
//! shot: the tests check both.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};

use crate::sparse::SparseGraph;

const NONE: u32 = u32::MAX;

/// The decoding graph both decoders grow on: the matching graph's detectors, then the boundary
/// as node `n`, each edge a length in steps of growth.
pub struct ClusterGraph {
    pub n: usize,
    offsets: Vec<u32>,
    /// Per node (the boundary's too), its edges as (the node across, the edge's id).
    adj: Vec<(u32, u32)>,
    pub ends: Vec<(u32, u32)>,
    pub len: Vec<u32>,
    pub obs: Vec<u64>,
    /// Per node, its longest edge.
    wmax: Vec<u32>,
}

impl ClusterGraph {
    /// From a matching graph: every edge 2 steps long (`resolution` None), or its weight scaled
    /// so that the heaviest edge is `resolution` steps, and none shorter than one.
    pub fn new(g: &SparseGraph, resolution: Option<u32>) -> ClusterGraph {
        let n = g.num_nodes;
        let m = g.num_edges();
        let ends: Vec<(u32, u32)> = (0..m as u32).map(|id| g.edge_ends(id)).collect();
        let obs: Vec<u64> = (0..m).map(|id| g.obs[g.halves[id][0] as usize]).collect();
        let len: Vec<u32> = match resolution {
            None => vec![2; m],
            Some(r) => {
                let r = r.max(1);
                let top = (0..m as u32).map(|id| g.float_weight(id)).fold(0.0f64, f64::max);
                (0..m as u32)
                    .map(|id| {
                        let w = g.float_weight(id);
                        if top > 0.0 && w > 0.0 { ((f64::from(r) * w / top).round() as u32).clamp(1, r) } else { 1 }
                    })
                    .collect()
            }
        };
        Self::from_parts(n, ends, len, obs)
    }

    pub fn from_parts(n: usize, ends: Vec<(u32, u32)>, len: Vec<u32>, obs: Vec<u64>) -> ClusterGraph {
        let mut degree = vec![0u32; n + 1];
        for &(a, b) in &ends {
            degree[a as usize] += 1;
            degree[b as usize] += 1;
        }
        let mut offsets = vec![0u32; n + 2];
        for v in 0..=n {
            offsets[v + 1] = offsets[v] + degree[v];
        }
        let mut fill = offsets.clone();
        let mut adj = vec![(0, 0); offsets[n + 1] as usize];
        let mut wmax = vec![0u32; n + 1];
        for (id, &(a, b)) in ends.iter().enumerate() {
            for (u, v) in [(a, b), (b, a)] {
                adj[fill[u as usize] as usize] = (v, id as u32);
                fill[u as usize] += 1;
                wmax[u as usize] = wmax[u as usize].max(len[id]);
            }
        }
        ClusterGraph { n, offsets, adj, ends, len, obs, wmax }
    }

    fn adj(&self, v: u32) -> &[(u32, u32)] {
        &self.adj[self.offsets[v as usize] as usize..self.offsets[v as usize + 1] as usize]
    }

    /// The id of the edge between `u` and `v` (`n` is the boundary).
    pub fn edge_id(&self, u: u32, v: u32) -> Option<u32> {
        if u as usize > self.n || v as usize > self.n {
            return None;
        }
        self.adj(u).iter().find(|&&(to, _)| to == v).map(|&(_, e)| e)
    }

    /// Peel a spanning forest of the grown edges from its leaves: a leaf holding a detection
    /// event takes the edge to its parent into the correction and passes the event up. The
    /// boundary roots its own tree, so an odd cluster touching it passes its last event to it.
    /// `None` when a tree's root is left holding one (an odd cluster with no boundary).
    fn peel(&self, p: &mut Peel, grown: &[bool], grown_list: &[u32], defects: &[u32], correction: &mut Vec<u32>) -> Option<u64> {
        let n = self.n as u32;
        correction.clear();
        p.order.clear();
        for &d in defects {
            p.parity[d as usize] ^= true;
        }
        // The boundary's tree first, from its grown edges.
        p.visited[n as usize] = true;
        p.order.push(n);
        for &e in grown_list {
            let (a, b) = self.ends[e as usize];
            let v = if b == n { a } else if a == n { b } else { continue };
            if !p.visited[v as usize] {
                p.visited[v as usize] = true;
                p.via[v as usize] = e;
                p.order.push(v);
            }
        }
        let mut k = 1;
        let mut roots = defects.iter();
        loop {
            while k < p.order.len() {
                let v = p.order[k];
                k += 1;
                for &(to, e) in self.adj(v) {
                    if grown[e as usize] && !p.visited[to as usize] {
                        p.visited[to as usize] = true;
                        p.via[to as usize] = e;
                        p.order.push(to);
                    }
                }
            }
            match roots.find(|&&d| !p.visited[d as usize]) {
                Some(&d) => {
                    p.visited[d as usize] = true;
                    p.order.push(d);
                }
                None => break,
            }
        }
        let mut obs = 0u64;
        let mut stuck = false;
        for k in (1..p.order.len()).rev() {
            let v = p.order[k] as usize;
            if !p.parity[v] {
                continue;
            }
            p.parity[v] = false;
            let e = p.via[v];
            if e == NONE {
                stuck = true;
                continue;
            }
            obs ^= self.obs[e as usize];
            correction.push(e);
            let (a, b) = self.ends[e as usize];
            let up = if a as usize == v { b } else { a };
            p.parity[up as usize] ^= true;
        }
        for &v in &p.order {
            p.visited[v as usize] = false;
            p.via[v as usize] = NONE;
            p.parity[v as usize] = false;
        }
        (!stuck).then_some(obs)
    }
}

/// Peeling's working memory.
struct Peel {
    visited: Vec<bool>,
    via: Vec<u32>,
    parity: Vec<bool>,
    order: Vec<u32>,
}

impl Peel {
    fn new(nodes: usize) -> Peel {
        Peel { visited: vec![false; nodes], via: vec![NONE; nodes], parity: vec![false; nodes], order: Vec::new() }
    }
}

/// The detection events of a shot, each once: an event listed twice cancels.
fn unique_defects(defects: &[u32], n: usize, seen: &mut [bool], out: &mut Vec<u32>) -> bool {
    out.clear();
    for &d in defects {
        if d as usize >= n {
            for &v in out.iter() {
                seen[v as usize] = false;
            }
            out.clear();
            return false;
        }
        seen[d as usize] ^= true;
    }
    for &d in defects {
        if seen[d as usize] {
            seen[d as usize] = false;
            out.push(d);
        }
    }
    out.sort_unstable();
    true
}

/// What a decode did, besides its prediction.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ClusterStats {
    /// Rounds of growth.
    pub rounds: u32,
    /// Collision Clustering: the pairs of events the Match unit compared. The Local Clustering
    /// Decoder: the passes its stages ran, the hardware's time steps (merging and syncing
    /// repeat while a vertex is busy).
    pub steps: u64,
}

// ---------------------------------------------------------------------------------------------
// Collision Clustering

/// One event's ball: Dijkstra from its node, every node within `reach` settled (none yet when
/// it is -1); the boundary settles but is never passed through.
#[derive(Default)]
struct Ball {
    heap: BinaryHeap<Reverse<(u32, u32)>>,
    dist: HashMap<u32, u32>,
    order: Vec<(u32, u32)>,
    reach: i64,
}

pub struct CcWork {
    peel: Peel,
    seen: Vec<bool>,
    defects: Vec<u32>,
    balls: Vec<Ball>,
    radius: Vec<u32>,
    parent: Vec<u32>,
    parity: Vec<bool>,
    boundary: Vec<bool>,
    valid: Vec<bool>,
    merges: Vec<(u32, u32)>,
    /// Per node, one more than its cover (how far past it the furthest ball reaches), 0 when
    /// no ball reaches it.
    cover: Vec<u32>,
    covered: Vec<u32>,
    grown: Vec<bool>,
    grown_list: Vec<u32>,
    /// The last decode's correction, as edge ids.
    pub correction: Vec<u32>,
}

pub struct CollisionClustering {
    pub graph: ClusterGraph,
}

impl CollisionClustering {
    pub fn new(graph: ClusterGraph) -> CollisionClustering {
        CollisionClustering { graph }
    }

    pub fn work(&self) -> CcWork {
        let nodes = self.graph.n + 1;
        CcWork {
            peel: Peel::new(nodes),
            seen: vec![false; nodes],
            defects: Vec::new(),
            balls: Vec::new(),
            radius: Vec::new(),
            parent: Vec::new(),
            parity: Vec::new(),
            boundary: Vec::new(),
            valid: Vec::new(),
            merges: Vec::new(),
            cover: vec![0; nodes],
            covered: Vec::new(),
            grown: vec![false; self.graph.len.len()],
            grown_list: Vec::new(),
            correction: Vec::new(),
        }
    }

    /// Per event of the last decode (in node order), the lowest event in its cluster, or
    /// `u32::MAX` for a cluster that touches the boundary.
    pub fn cluster_labels(&self, w: &mut CcWork) -> Vec<u32> {
        (0..w.defects.len() as u32)
            .map(|i| {
                let r = find(&mut w.parent, i) as usize;
                if w.boundary[r] { NONE } else { w.defects[r] }
            })
            .collect()
    }

    /// Settle a ball out to `r`.
    fn extend(&self, ball: &mut Ball, r: u32) {
        if ball.reach >= i64::from(r) {
            return;
        }
        let n = self.graph.n as u32;
        while let Some(&Reverse((d, v))) = ball.heap.peek() {
            if d > r {
                break;
            }
            ball.heap.pop();
            if ball.dist.contains_key(&v) {
                continue;
            }
            ball.dist.insert(v, d);
            ball.order.push((v, d));
            if v == n {
                continue;
            }
            for &(to, e) in self.graph.adj(v) {
                if !ball.dist.contains_key(&to) {
                    ball.heap.push(Reverse((d + self.graph.len[e as usize], to)));
                }
            }
        }
        ball.reach = i64::from(r);
    }

    /// Whether events `i` and `j` are within `r` of each other, from whichever ball has
    /// settled further, extended to `r` if neither has.
    fn within(&self, w: &mut CcWork, i: usize, j: usize, r: u32) -> bool {
        let (a, b) = if w.balls[j].reach > w.balls[i].reach { (j, i) } else { (i, j) };
        self.extend(&mut w.balls[a], r);
        w.balls[a].dist.get(&w.defects[b]).is_some_and(|&d| d <= r)
    }

    /// The observables the correction flips (`None` for an odd cluster with nowhere to grow),
    /// and what the decode did.
    pub fn decode(&self, defects: &[u32], w: &mut CcWork) -> (Option<u64>, ClusterStats) {
        let n = self.graph.n;
        let mut stats = ClusterStats::default();
        let mut events = std::mem::take(&mut w.defects);
        if !unique_defects(defects, n, &mut w.seen, &mut events) {
            w.defects = events;
            return (None, stats);
        }
        w.defects = events;
        let s = w.defects.len();
        while w.balls.len() < s {
            w.balls.push(Ball::default());
        }
        w.radius.clear();
        w.radius.resize(s, 0);
        w.parent.clear();
        w.parent.extend(0..s as u32);
        w.parity.clear();
        w.parity.resize(s, true);
        w.boundary.clear();
        w.boundary.resize(s, false);
        w.valid.clear();
        w.valid.resize(s, true);
        for i in 0..s {
            let ball = &mut w.balls[i];
            ball.heap.clear();
            ball.dist.clear();
            ball.order.clear();
            ball.heap.push(Reverse((0, w.defects[i])));
            ball.reach = -1;
        }
        let mut stuck = false;
        while w.valid.iter().any(|&v| v) {
            // A growing ball that has settled its whole component, and covers it, can meet
            // nothing new: its cluster is odd with no boundary.
            if (0..s).any(|i| w.valid[i] && exhausted(&mut w.balls[i], w.radius[i])) {
                stuck = true;
                break;
            }
            stats.rounds += 1;
            // Grow: the valid entries of the Cluster Growth Stack.
            for i in 0..s {
                if w.valid[i] {
                    w.radius[i] += 1;
                }
            }
            // Match: every pair from two clusters with a ball that grew, and every growing ball
            // against the boundary.
            w.merges.clear();
            for i in 0..s {
                for j in i + 1..s {
                    if !(w.valid[i] || w.valid[j]) || find(&mut w.parent, i as u32) == find(&mut w.parent, j as u32) {
                        continue;
                    }
                    stats.steps += 1;
                    let r = w.radius[i] + w.radius[j];
                    if self.within(w, i, j, r) {
                        w.merges.push((i as u32, j as u32));
                    }
                }
            }
            for i in 0..s {
                if w.valid[i] {
                    let r = w.radius[i];
                    self.extend(&mut w.balls[i], r);
                    if w.balls[i].dist.get(&(n as u32)).is_some_and(|&d| d <= r) {
                        w.merges.push((i as u32, NONE));
                    }
                }
            }
            // Union: the Parent table and the Parity and Boundary registers.
            for k in 0..w.merges.len() {
                let (i, j) = w.merges[k];
                let ri = find(&mut w.parent, i);
                if j == NONE {
                    w.boundary[ri as usize] = true;
                    continue;
                }
                let rj = find(&mut w.parent, j);
                if ri != rj {
                    let (keep, gone) = (ri.min(rj), ri.max(rj));
                    w.parent[gone as usize] = keep;
                    w.parity[keep as usize] ^= w.parity[gone as usize];
                    w.boundary[keep as usize] |= w.boundary[gone as usize];
                }
            }
            for i in 0..s {
                let r = find(&mut w.parent, i as u32) as usize;
                w.valid[i] = w.parity[r] && !w.boundary[r];
            }
        }
        if stuck {
            return (None, stats);
        }
        // The edges the balls cover, for peeling.
        for i in 0..s {
            let r = w.radius[i];
            self.extend(&mut w.balls[i], r);
            for k in 0..w.balls[i].order.len() {
                let (v, d) = w.balls[i].order[k];
                if d > r || v as usize == n {
                    continue;
                }
                if w.cover[v as usize] == 0 {
                    w.covered.push(v);
                }
                w.cover[v as usize] = w.cover[v as usize].max(r - d + 1);
            }
        }
        for k in 0..w.covered.len() {
            let v = w.covered[k];
            let cv = w.cover[v as usize] - 1;
            for &(to, e) in self.graph.adj(v) {
                let ct = w.cover[to as usize].saturating_sub(1);
                if !w.grown[e as usize] && cv + ct >= self.graph.len[e as usize] {
                    w.grown[e as usize] = true;
                    w.grown_list.push(e);
                }
            }
        }
        w.grown_list.sort_unstable();
        let mut correction = std::mem::take(&mut w.correction);
        let out = self.graph.peel(&mut w.peel, &w.grown, &w.grown_list, &w.defects, &mut correction);
        w.correction = correction;
        for &v in &w.covered {
            w.cover[v as usize] = 0;
        }
        w.covered.clear();
        for &e in &w.grown_list {
            w.grown[e as usize] = false;
        }
        w.grown_list.clear();
        (out, stats)
    }
}

/// Whether a ball has settled every node it can reach, all within `r`.
fn exhausted(ball: &mut Ball, r: u32) -> bool {
    while let Some(&Reverse((_, v))) = ball.heap.peek() {
        if !ball.dist.contains_key(&v) {
            return false;
        }
        ball.heap.pop();
    }
    ball.reach >= 0 && ball.order.iter().all(|&(_, d)| d <= r)
}

fn find(parent: &mut [u32], mut v: u32) -> u32 {
    while parent[v as usize] != v {
        let up = parent[parent[v as usize] as usize];
        parent[v as usize] = up;
        v = up;
    }
    v
}

// ---------------------------------------------------------------------------------------------
// The Local Clustering Decoder

/// The controller's stages (the paper's Fig. 4).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Stage {
    Growing,
    Merging,
    Picking,
    Syncing,
}

pub struct LcdWork {
    peel: Peel,
    seen: Vec<bool>,
    defects: Vec<u32>,
    /// Per vertex, the paper's fields: its cluster's index (the lowest key in the cluster, the
    /// boundary's 0 below every vertex's own index + 1), its parent, radius, parity and
    /// activity; and whether it is in a cluster (a defect, or on a fully grown edge).
    cindex: Vec<u32>,
    parent: Vec<u32>,
    radius: Vec<u32>,
    parity: Vec<bool>,
    active: Vec<bool>,
    member: Vec<bool>,
    members: Vec<u32>,
    pregrown: Vec<bool>,
    pregrown_list: Vec<u32>,
    grown: Vec<bool>,
    grown_list: Vec<u32>,
    /// The last decode's correction, as edge ids.
    pub correction: Vec<u32>,
}

impl LcdWork {
    /// How many edges the last decode pre-grew.
    pub fn pregrown_count(&self) -> u32 {
        self.pregrown_list.len() as u32
    }
}

pub struct LocalClustering {
    pub graph: ClusterGraph,
    /// The adaptivity map: per trigger (a measurement's herald), the edges it pre-grows.
    pub adaptivity: Vec<Vec<u32>>,
}

impl LocalClustering {
    pub fn new(graph: ClusterGraph, adaptivity: Vec<Vec<u32>>) -> LocalClustering {
        LocalClustering { graph, adaptivity }
    }

    pub fn work(&self) -> LcdWork {
        let nodes = self.graph.n + 1;
        let m = self.graph.len.len();
        let mut cindex: Vec<u32> = (1..=nodes as u32).collect();
        cindex[self.graph.n] = 0;
        LcdWork {
            peel: Peel::new(nodes),
            seen: vec![false; nodes],
            defects: Vec::new(),
            cindex,
            parent: (0..nodes as u32).collect(),
            radius: vec![0; nodes],
            parity: vec![false; nodes],
            active: vec![false; nodes],
            member: vec![false; nodes],
            members: Vec::new(),
            pregrown: vec![false; m],
            pregrown_list: Vec::new(),
            grown: vec![false; m],
            grown_list: Vec::new(),
            correction: Vec::new(),
        }
    }

    /// Whether `u`'s neighbour across edge `e` is accessible: their radii cover the edge, or
    /// the adaptivity engine pre-grew it.
    fn accessible(&self, w: &LcdWork, u: u32, v: u32, e: u32) -> bool {
        w.pregrown[e as usize] || w.radius[u as usize] + w.radius[v as usize] >= self.graph.len[e as usize]
    }

    fn join(&self, w: &mut LcdWork, v: u32) {
        if !w.member[v as usize] {
            w.member[v as usize] = true;
            if v as usize != self.graph.n {
                w.members.push(v);
            }
        }
    }

    /// The observables the correction flips (`None` for an odd cluster with nowhere to grow),
    /// and what the decode did. `triggers` are the adaptivity map's entries that fired (the
    /// heralded measurements).
    pub fn decode(&self, defects: &[u32], triggers: &[u32], w: &mut LcdWork) -> (Option<u64>, ClusterStats) {
        let n = self.graph.n as u32;
        self.reset(w);
        let mut stats = ClusterStats::default();
        let mut events = std::mem::take(&mut w.defects);
        let ok = unique_defects(defects, self.graph.n, &mut w.seen, &mut events);
        w.defects = events;
        if !ok {
            return (None, stats);
        }
        // Initing: the defects, and the pre-grown edges' ends.
        for k in 0..w.defects.len() {
            let d = w.defects[k];
            w.parity[d as usize] = true;
            w.active[d as usize] = true;
            self.join(w, d);
        }
        for &t in triggers {
            for &e in self.adaptivity.get(t as usize).map_or(&[][..], |x| &x[..]) {
                if !w.pregrown[e as usize] {
                    w.pregrown[e as usize] = true;
                    w.pregrown_list.push(e);
                    let (a, b) = self.graph.ends[e as usize];
                    self.join(w, a);
                    self.join(w, b);
                }
            }
        }
        let mut stage = if w.pregrown_list.is_empty() { Stage::Growing } else { Stage::Merging };
        let mut stuck = false;
        loop {
            match stage {
                Stage::Growing => {
                    let (mut grew, mut any) = (false, false);
                    for k in 0..w.members.len() {
                        let u = w.members[k];
                        any |= w.active[u as usize];
                        if w.active[u as usize] && w.radius[u as usize] <= self.graph.wmax[u as usize] {
                            w.radius[u as usize] += 1;
                            grew = true;
                        }
                    }
                    if !grew {
                        // Nothing active (no events at all), or active vertices all grown out.
                        stuck = any;
                        break;
                    }
                    stats.steps += 1;
                    stats.rounds += 1;
                    // The vertices across newly grown edges join the clusters.
                    for k in 0..w.members.len() {
                        let u = w.members[k];
                        if !w.active[u as usize] {
                            continue;
                        }
                        for &(v, e) in self.graph.adj(u) {
                            if !w.member[v as usize] && self.accessible(w, u, v, e) {
                                self.join(w, v);
                            }
                        }
                    }
                    stage = Stage::Merging;
                }
                Stage::Merging => {
                    // Passes in serial until no vertex is busy.
                    loop {
                        stats.steps += 1;
                        let mut busy = false;
                        for k in 0..w.members.len() {
                            let u = w.members[k];
                            for &(v, e) in self.graph.adj(u) {
                                if w.cindex[v as usize] < w.cindex[u as usize] && self.accessible(w, u, v, e) {
                                    w.cindex[u as usize] = w.cindex[v as usize];
                                    w.parent[u as usize] = v;
                                    busy = true;
                                }
                            }
                            let p = w.parent[u as usize];
                            if w.parity[u as usize] && p != u {
                                // The boundary absorbs parity.
                                if p != n {
                                    w.parity[p as usize] ^= true;
                                }
                                w.parity[u as usize] = false;
                                busy = true;
                            }
                        }
                        if !busy {
                            break;
                        }
                    }
                    stage = Stage::Picking;
                }
                Stage::Picking => {
                    stats.steps += 1;
                    for k in 0..w.members.len() {
                        let u = w.members[k] as usize;
                        w.active[u] = w.parity[u];
                    }
                    stage = Stage::Syncing;
                }
                Stage::Syncing => {
                    loop {
                        stats.steps += 1;
                        let mut busy = false;
                        for k in 0..w.members.len() {
                            let u = w.members[k];
                            if w.active[u as usize] {
                                continue;
                            }
                            if self.graph.adj(u).iter().any(|&(v, e)| w.active[v as usize] && self.accessible(w, u, v, e)) {
                                w.active[u as usize] = true;
                                busy = true;
                            }
                        }
                        if !busy {
                            break;
                        }
                    }
                    if !w.members.iter().any(|&u| w.active[u as usize]) {
                        break;
                    }
                    stage = Stage::Growing;
                }
            }
        }
        let mut out = None;
        if !stuck {
            for k in 0..w.members.len() {
                let u = w.members[k];
                for &(v, e) in self.graph.adj(u) {
                    if !w.grown[e as usize] && self.accessible(w, u, v, e) {
                        w.grown[e as usize] = true;
                        w.grown_list.push(e);
                    }
                }
            }
            w.grown_list.sort_unstable();
            let mut correction = std::mem::take(&mut w.correction);
            out = self.graph.peel(&mut w.peel, &w.grown, &w.grown_list, &w.defects, &mut correction);
            w.correction = correction;
        }
        (out, stats)
    }

    /// Put back what the last decode touched, which stays until the next for its clusters to
    /// be read.
    fn reset(&self, w: &mut LcdWork) {
        for &u in &w.members {
            let u = u as usize;
            w.cindex[u] = u as u32 + 1;
            w.parent[u] = u as u32;
            w.radius[u] = 0;
            w.parity[u] = false;
            w.active[u] = false;
            w.member[u] = false;
        }
        w.member[self.graph.n] = false;
        w.members.clear();
        for &e in &w.pregrown_list {
            w.pregrown[e as usize] = false;
        }
        w.pregrown_list.clear();
        for &e in &w.grown_list {
            w.grown[e as usize] = false;
        }
        w.grown_list.clear();
    }

    /// Per event of the last decode (in node order), the lowest event in its cluster, or
    /// `u32::MAX` for a cluster that touches the boundary.
    pub fn cluster_labels(&self, w: &LcdWork) -> Vec<u32> {
        let mut lowest: HashMap<u32, u32> = HashMap::new();
        for &d in &w.defects {
            lowest.entry(w.cindex[d as usize]).or_insert(d);
        }
        w.defects.iter().map(|&d| if w.cindex[d as usize] == 0 { NONE } else { lowest[&w.cindex[d as usize]] }).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::circuit::Circuit;
    use crate::dem::Dem;
    use crate::dem_decoder::DemDecoder;
    use crate::frame_sampler::FrameSampler;
    use crate::surface_code::Xorshift;

    fn surface(d: usize, rounds: usize, p: f64) -> (Circuit, Dem, SparseGraph) {
        use crate::memory::{generate, CodeKind, NoiseModel};
        let c = generate(CodeKind::Rotated, d, rounds, NoiseModel::Sd6 { p }, crate::circuit::Basis::Z).unwrap();
        let dem = Dem::from_circuit(&c).unwrap();
        let (g, _) = DemDecoder::new(&dem).unwrap().into_parts(false);
        (c, dem, g)
    }

    fn events(dets: &[bool]) -> Vec<u32> {
        dets.iter().enumerate().filter(|x| *x.1).map(|x| x.0 as u32).collect()
    }

    /// The correction's edges flip exactly the detection events.
    fn explains(g: &ClusterGraph, correction: &[u32], defects: &[u32]) -> bool {
        let mut flipped = vec![false; g.n + 1];
        for &e in correction {
            let (a, b) = g.ends[e as usize];
            flipped[a as usize] ^= true;
            flipped[b as usize] ^= true;
        }
        for &d in defects {
            flipped[d as usize] ^= true;
        }
        !flipped[..g.n].iter().any(|&x| x)
    }

    /// Delfosse and Nickerson's union-find, round by round: every odd cluster off the boundary
    /// grows each edge leaving it by a step, then the clusters an edge grown whole joins merge.
    /// Per event, the lowest event of its cluster, or `NONE` for a cluster on the boundary;
    /// `None` when an odd cluster cannot grow.
    fn reference_clusters(g: &ClusterGraph, defects: &[u32]) -> Option<Vec<u32>> {
        let n = g.n;
        let mut parent: Vec<u32> = (0..n as u32).collect();
        let mut odd = vec![false; n];
        let mut boundary = vec![false; n];
        let mut inside = vec![false; n];
        for &d in defects {
            odd[d as usize] = true;
            inside[d as usize] = true;
        }
        let mut support = vec![0u32; g.len.len()];
        loop {
            let mut grows = vec![false; n];
            for v in 0..n as u32 {
                let r = find(&mut parent, v) as usize;
                grows[v as usize] = inside[v as usize] && odd[r] && !boundary[r];
            }
            if !grows.iter().any(|&x| x) {
                break;
            }
            let mut changed = false;
            for (e, &(a, b)) in g.ends.iter().enumerate() {
                if support[e] >= g.len[e] {
                    continue;
                }
                let ra = find(&mut parent, a);
                let rb = if b as usize == n { u32::MAX } else { find(&mut parent, b) };
                if ra == rb {
                    continue;
                }
                for x in [a, b] {
                    if (x as usize) < n && grows[x as usize] {
                        support[e] += 1;
                        changed = true;
                    }
                }
            }
            if !changed {
                return None;
            }
            for (e, &(a, b)) in g.ends.iter().enumerate() {
                if support[e] < g.len[e] {
                    continue;
                }
                inside[a as usize] = true;
                let ra = find(&mut parent, a);
                if b as usize == n {
                    boundary[ra as usize] = true;
                    continue;
                }
                inside[b as usize] = true;
                let rb = find(&mut parent, b);
                if ra != rb {
                    let (keep, gone) = (ra.min(rb), ra.max(rb));
                    parent[gone as usize] = keep;
                    odd[keep as usize] ^= odd[gone as usize];
                    boundary[keep as usize] |= boundary[gone as usize];
                }
            }
        }
        let mut lowest: HashMap<u32, u32> = HashMap::new();
        let mut out = Vec::new();
        for &d in defects {
            let r = find(&mut parent, d);
            out.push(if boundary[r as usize] { NONE } else { *lowest.entry(r).or_insert(d) });
        }
        Some(out)
    }

    #[test]
    fn clusters_are_union_finds_and_the_two_decoders_agree_shot_for_shot() {
        for (resolution, p) in [(None, 0.006), (Some(12), 0.006), (None, 0.012)] {
            let (c, _, sg) = surface(5, 5, p);
            let cc = CollisionClustering::new(ClusterGraph::new(&sg, resolution));
            let lcd = LocalClustering::new(ClusterGraph::new(&sg, resolution), Vec::new());
            let (mut wc, mut wl) = (cc.work(), lcd.work());
            let sampler = FrameSampler::new(&c).unwrap();
            let mut rng = Xorshift::new(3);
            let mut merged = 0;
            for _ in 0..600 {
                let defects = events(&sampler.sample(&mut rng).detectors);
                let (a, sa) = cc.decode(&defects, &mut wc);
                let (b, sb) = lcd.decode(&defects, &[], &mut wl);
                let want = reference_clusters(&cc.graph, &defects).unwrap();
                assert_eq!(cc.cluster_labels(&mut wc), want, "{resolution:?} {defects:?}");
                assert_eq!(lcd.cluster_labels(&wl), want, "{resolution:?} {defects:?}");
                assert_eq!(a, b, "{resolution:?} {defects:?}");
                assert_eq!(wc.correction.iter().copied().collect::<std::collections::BTreeSet<_>>(), wl.correction.iter().copied().collect(), "{defects:?}");
                assert!(a.is_some() && explains(&cc.graph, &wc.correction, &defects));
                assert_eq!(sa.rounds, sb.rounds);
                merged += want.iter().filter(|&&l| l != NONE).count();
            }
            assert!(merged > 500, "{merged}");
        }
    }

    #[test]
    fn every_single_fault_is_corrected() {
        let (_, dem, sg) = surface(5, 5, 0.003);
        for resolution in [None, Some(16)] {
            let cc = CollisionClustering::new(ClusterGraph::new(&sg, resolution));
            let lcd = LocalClustering::new(ClusterGraph::new(&sg, resolution), Vec::new());
            let (mut wc, mut wl) = (cc.work(), lcd.work());
            for m in &dem.mechanisms {
                assert_eq!(cc.decode(&m.detectors, &mut wc).0, Some(m.observables), "{resolution:?} {:?}", m.detectors);
                assert_eq!(lcd.decode(&m.detectors, &[], &mut wl).0, Some(m.observables), "{resolution:?} {:?}", m.detectors);
            }
        }
    }

    /// A path B - 0 - 1 - 2 - 3 - 4 - B whose left boundary edge flips the observable.
    fn path() -> ClusterGraph {
        let ends = vec![(0, 5), (0, 1), (1, 2), (2, 3), (3, 4), (4, 5)];
        ClusterGraph::from_parts(5, ends, vec![2; 6], vec![1, 0, 0, 0, 0, 0])
    }

    #[test]
    fn pre_grown_edges_cluster_first() {
        let lcd = LocalClustering::new(path(), vec![vec![1, 2, 3, 4], vec![]]);
        let mut w = lcd.work();
        // Alone, each end's event meets its own boundary: the correction crosses the left one.
        assert_eq!(lcd.decode(&[0, 4], &[], &mut w).0, Some(1));
        assert_eq!(w.correction.len(), 2);
        // Heralded, the pre-grown path pairs them.
        let (obs, stats) = lcd.decode(&[0, 4], &[0], &mut w);
        assert_eq!(obs, Some(0));
        assert_eq!(w.correction.len(), 4);
        assert_eq!(stats.rounds, 0);
        assert_eq!(lcd.cluster_labels(&w), vec![0, 0]);
        // A trigger with nothing to pre-grow changes nothing.
        assert_eq!(lcd.decode(&[0, 4], &[1], &mut w).0, Some(1));
    }

    #[test]
    fn odd_clusters_with_nowhere_to_grow_fail_and_work_is_reset() {
        // Two nodes, no boundary.
        let g = || ClusterGraph::from_parts(2, vec![(0, 1)], vec![2], vec![1]);
        let cc = CollisionClustering::new(g());
        let lcd = LocalClustering::new(g(), Vec::new());
        let (mut wc, mut wl) = (cc.work(), lcd.work());
        for _ in 0..2 {
            assert_eq!(cc.decode(&[0], &mut wc).0, None);
            assert_eq!(lcd.decode(&[0], &[], &mut wl).0, None);
            assert_eq!(cc.decode(&[0, 1], &mut wc).0, Some(1));
            assert_eq!(lcd.decode(&[0, 1], &[], &mut wl).0, Some(1));
            assert_eq!(cc.decode(&[], &mut wc).0, Some(0));
            assert_eq!(lcd.decode(&[1, 1], &[], &mut wl).0, Some(0));
            assert_eq!(cc.decode(&[7], &mut wc).0, None);
        }
    }
}

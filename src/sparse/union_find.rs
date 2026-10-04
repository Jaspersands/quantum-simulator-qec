//! Weighted union-find decoding (Delfosse and Nickerson, Quantum 5, 595, 2021, with the
//! weighted growth of Huang, Newman and Brown, PRA 102, 012419, 2020) on the matching graph.
//!
//! WHY THIS EXISTS
//! ---------------
//! Union-find is the standard fast approximate decoder, and the baseline other decoders are
//! measured against. Each cluster with an odd number of detection events grows along all its
//! edges at once, edge weight by edge weight, until it meets another cluster or the boundary and
//! its events can pair up inside it; a spanning tree of each cluster is then peeled from its
//! leaves for the correction. Its corrections always explain every detection event, and its
//! logical error rate is a little above matching's.
//!
//! Growth is simulated event by event: each edge's completion is scheduled in a radix heap (its
//! times only rise), and rescheduled only when a cluster at one of its ends starts or stops
//! growing. Measured against this engine's matcher, which is itself near-linear, union-find is
//! about half its speed at every noise strength tried: in this engine, it is the baseline, not
//! the fast option.

use super::graph::SparseGraph;
use super::state::BOUNDARY;

/// Reusable working memory for one graph, reset between decodes in time proportional to what
/// the decode touched.
pub struct UfScratch {
    parent: Vec<u32>,
    size: Vec<u32>,
    /// Per root: an odd number of detection events.
    odd: Vec<bool>,
    /// Per root: the cluster reaches the boundary.
    boundary: Vec<bool>,
    /// Per root: the edges leaving the cluster that may still grow (some may have gone stale).
    frontier: Vec<Vec<u32>>,
    /// Per node: in some cluster.
    clustered: Vec<bool>,
    /// Per edge: its support at `since`, the rate it grows at, and the version of its event.
    /// Support and time are in half-weight units, so that an edge growing from both ends
    /// (rate 2) finishes at a whole time.
    support: Vec<u64>,
    since: Vec<u64>,
    rate: Vec<u8>,
    version: Vec<u32>,
    grown: Vec<bool>,
    events: RadixHeap,
    /// Edges to re-rate after a merge, kept to save allocating.
    changed: Vec<u32>,
    touched_nodes: Vec<u32>,
    node_touched: Vec<bool>,
    touched_edges: Vec<u32>,
    edge_touched: Vec<bool>,
    /// Peeling.
    defect: Vec<bool>,
    visited: Vec<bool>,
    via: Vec<u32>,
    order: Vec<u32>,
    /// The last decode's correction, as edge ids.
    pub(crate) correction: Vec<u32>,
}

const NO_EDGE: u32 = u32::MAX;

impl SparseGraph {
    pub fn union_find_scratch(&self) -> UfScratch {
        let n = self.num_nodes + 1;
        let m = self.num_edges();
        let mut boundary = vec![false; n];
        boundary[self.num_nodes] = true;
        UfScratch {
            parent: (0..n as u32).collect(),
            size: vec![1; n],
            odd: vec![false; n],
            boundary,
            frontier: vec![Vec::new(); n],
            clustered: vec![false; n],
            support: vec![0; m],
            since: vec![0; m],
            rate: vec![0; m],
            version: vec![0; m],
            grown: vec![false; m],
            events: RadixHeap::default(),
            changed: Vec::new(),
            touched_nodes: Vec::new(),
            node_touched: vec![false; n],
            touched_edges: Vec::new(),
            edge_touched: vec![false; m],
            defect: vec![false; n],
            visited: vec![false; n],
            via: vec![NO_EDGE; n],
            order: Vec::new(),
            correction: Vec::new(),
        }
    }

    /// An edge's weight in half units.
    fn uf_weight(&self, id: u32) -> u64 {
        2 * self.w[self.halves[id as usize][0] as usize].max(0) as u64
    }

    fn uf_obs(&self, id: u32) -> u64 {
        self.obs[self.halves[id as usize][0] as usize]
    }

    /// Bring node `v` into the decode: a cluster of its own, its edges its frontier.
    fn uf_join(&self, s: &mut UfScratch, v: u32) {
        touch(s, v);
        if s.clustered[v as usize] || v as usize == self.num_nodes {
            return;
        }
        s.clustered[v as usize] = true;
        for e in self.edges(v) {
            s.frontier[v as usize].push(self.edge_of[e]);
        }
    }

    /// Recompute edge `id`'s growth rate at time `now`: one for each end in a cluster still
    /// growing, none once both ends are in one cluster; and schedule when it will be whole.
    fn uf_rerate(&self, s: &mut UfScratch, id: u32, now: u64) {
        let i = id as usize;
        if s.grown[i] {
            return;
        }
        if !s.edge_touched[i] {
            s.edge_touched[i] = true;
            s.touched_edges.push(id);
        }
        let (a, b) = self.ends[i];
        let n = self.num_nodes as u32;
        let ra = find(&mut s.parent, a);
        let rb = find(&mut s.parent, b);
        let grows = |s: &UfScratch, v: u32, r: u32| v != n && s.clustered[v as usize] && s.odd[r as usize] && !s.boundary[r as usize];
        let rate = if ra == rb { 0 } else { u8::from(grows(s, a, ra)) + u8::from(grows(s, b, rb)) };
        s.support[i] += u64::from(s.rate[i]) * (now - s.since[i]);
        s.since[i] = now;
        if rate == s.rate[i] && rate > 0 {
            return;
        }
        s.rate[i] = rate;
        s.version[i] = s.version[i].wrapping_add(1);
        if rate > 0 {
            let done = now + self.uf_weight(id).saturating_sub(s.support[i]).div_ceil(u64::from(rate));
            s.events.push(done, id, s.version[i]);
        }
    }

    /// Re-rate every edge on cluster `root`'s frontier, dropping those gone stale.
    fn uf_rerate_cluster(&self, s: &mut UfScratch, root: u32, now: u64) {
        let mut frontier = std::mem::take(&mut s.frontier[root as usize]);
        self.uf_rerate_edges(s, &mut frontier, now);
        s.frontier[root as usize] = frontier;
    }

    /// Re-rate `edges`, dropping those grown or now inside one cluster: clusters only merge,
    /// so an edge inside one stays inside, and never grows again.
    fn uf_rerate_edges(&self, s: &mut UfScratch, edges: &mut Vec<u32>, now: u64) {
        edges.retain(|&id| {
            if s.grown[id as usize] {
                return false;
            }
            let (a, b) = self.ends[id as usize];
            if find(&mut s.parent, a) == find(&mut s.parent, b) {
                // Stop any growth it had scheduled.
                s.version[id as usize] = s.version[id as usize].wrapping_add(1);
                s.rate[id as usize] = 0;
                return false;
            }
            true
        });
        for &id in edges.iter() {
            self.uf_rerate(s, id, now);
        }
    }

    /// The observables the union-find correction of `defects` flips. Every detection event is
    /// explained: each cluster with an odd number of events grows, all its edges at once, until
    /// it meets another cluster or the boundary; the one case it cannot explain is a component
    /// holding an odd number of events and no boundary (`None`).
    pub fn decode_union_find(&self, s: &mut UfScratch, defects: &[u32]) -> Option<u64> {
        let n = self.num_nodes as u32;
        for &d in defects {
            if d >= n {
                return None;
            }
            self.uf_join(s, d);
            s.odd[d as usize] ^= true;
            s.defect[d as usize] ^= true;
        }
        touch(s, n);
        for &d in defects {
            if find(&mut s.parent, d) == d {
                self.uf_rerate_cluster(s, d, 0);
            }
        }
        let mut stuck = false;
        while let Some((now, id, version)) = s.events.pop() {
            let i = id as usize;
            if version != s.version[i] || s.grown[i] {
                continue;
            }
            s.support[i] += u64::from(s.rate[i]) * (now - s.since[i]);
            s.since[i] = now;
            s.grown[i] = true;
            let (a, b) = self.ends[i];
            for v in [a, b] {
                self.uf_join(s, v);
            }
            let (ra, rb) = (find(&mut s.parent, a), find(&mut s.parent, b));
            if ra == rb {
                continue;
            }
            // Only the edges of a side whose growth stops or starts change rate.
            let grows = |s: &UfScratch, r: u32| s.odd[r as usize] && !s.boundary[r as usize];
            let merged = (s.odd[ra as usize] ^ s.odd[rb as usize]) && !(s.boundary[ra as usize] || s.boundary[rb as usize]);
            let mut changed = std::mem::take(&mut s.changed);
            for r in [ra, rb] {
                if grows(s, r) != merged {
                    changed.append(&mut s.frontier[r as usize]);
                }
            }
            let root = union(s, a, b);
            self.uf_rerate_edges(s, &mut changed, now);
            // (A node the edge just reached is a cluster of its own that was not growing, so its
            // edges are among those re-rated.)
            s.frontier[root as usize].append(&mut changed);
            s.changed = changed;
        }
        // A cluster still odd and growing has run out of edges.
        for &d in defects {
            let r = find(&mut s.parent, d) as usize;
            if s.odd[r] && !s.boundary[r] {
                stuck = true;
            }
        }
        let out = (!stuck).then(|| self.peel(s));
        reset(s);
        out
    }

    /// Peel a spanning forest of the grown edges from its leaves: a leaf holding a detection
    /// event takes the edge to its parent into the correction and passes the event up. Trees
    /// touching the boundary are rooted there, so an odd one's last event goes to it.
    fn peel(&self, s: &mut UfScratch) -> u64 {
        let n = self.num_nodes as u32;
        s.order.clear();
        s.correction.clear();
        let starts: Vec<u32> = std::iter::once(n).chain(s.touched_nodes.iter().copied()).collect();
        for root in starts {
            if s.visited[root as usize] {
                continue;
            }
            s.visited[root as usize] = true;
            let first = s.order.len();
            s.order.push(root);
            let mut k = first;
            while k < s.order.len() {
                let v = s.order[k];
                k += 1;
                if v == n {
                    // The boundary's grown edges.
                    for &id in &s.touched_edges {
                        let (a, b) = self.ends[id as usize];
                        if s.grown[id as usize] && b == n && !s.visited[a as usize] {
                            s.visited[a as usize] = true;
                            s.via[a as usize] = id;
                            s.order.push(a);
                        }
                    }
                    continue;
                }
                for e in self.edges(v) {
                    let id = self.edge_of[e];
                    if !s.grown[id as usize] {
                        continue;
                    }
                    let to = self.to[e];
                    let to = if to == BOUNDARY { n } else { to };
                    if !s.visited[to as usize] {
                        s.visited[to as usize] = true;
                        s.via[to as usize] = id;
                        s.order.push(to);
                    }
                }
            }
        }
        let mut obs = 0u64;
        for k in (0..s.order.len()).rev() {
            let v = s.order[k] as usize;
            let id = s.via[v];
            if id == NO_EDGE || !s.defect[v] {
                continue;
            }
            s.defect[v] = false;
            obs ^= self.uf_obs(id);
            s.correction.push(id);
            let (a, b) = self.ends[id as usize];
            let parent = if a as usize == v { b } else { a } as usize;
            s.defect[parent] ^= true;
        }
        obs
    }
}

/// A monotone priority queue (a radix heap): every key pushed is at least the last popped, as
/// event times are, which lets each push and pop cost amortized constant time where a binary
/// heap costs a logarithm.
struct RadixHeap {
    last: u64,
    len: usize,
    buckets: [Vec<(u64, u32, u32)>; 65],
}

impl Default for RadixHeap {
    fn default() -> RadixHeap {
        RadixHeap { last: 0, len: 0, buckets: std::array::from_fn(|_| Vec::new()) }
    }
}

impl RadixHeap {
    fn bucket(&self, key: u64) -> usize {
        (64 - (key ^ self.last).leading_zeros()) as usize
    }

    fn push(&mut self, key: u64, id: u32, version: u32) {
        debug_assert!(key >= self.last, "a radix heap's keys never fall below the last popped");
        let b = self.bucket(key);
        self.buckets[b].push((key, id, version));
        self.len += 1;
    }

    fn pop(&mut self) -> Option<(u64, u32, u32)> {
        if self.len == 0 {
            return None;
        }
        if self.buckets[0].is_empty() {
            let b = (1..65).find(|&b| !self.buckets[b].is_empty()).expect("a nonempty bucket");
            let moved = std::mem::take(&mut self.buckets[b]);
            self.last = moved.iter().map(|e| e.0).min().expect("a nonempty bucket");
            for e in &moved {
                let nb = self.bucket(e.0);
                self.buckets[nb].push(*e);
            }
            // Keep the emptied bucket's memory for later pushes.
            let mut moved = moved;
            moved.clear();
            self.buckets[b] = moved;
        }
        self.len -= 1;
        self.buckets[0].pop()
    }

    fn clear(&mut self) {
        for b in &mut self.buckets {
            b.clear();
        }
        self.len = 0;
        self.last = 0;
    }
}

fn find(parent: &mut [u32], mut v: u32) -> u32 {
    while parent[v as usize] != v {
        let up = parent[parent[v as usize] as usize];
        parent[v as usize] = up;
        v = up;
    }
    v
}

/// Note `v` for the reset.
fn touch(s: &mut UfScratch, v: u32) {
    if !s.node_touched[v as usize] {
        s.node_touched[v as usize] = true;
        s.touched_nodes.push(v);
    }
}

fn union(s: &mut UfScratch, a: u32, b: u32) -> u32 {
    let (ra, rb) = (find(&mut s.parent, a), find(&mut s.parent, b));
    if ra == rb {
        return ra;
    }
    let (big, small) = if s.size[ra as usize] >= s.size[rb as usize] { (ra, rb) } else { (rb, ra) };
    s.parent[small as usize] = big;
    s.size[big as usize] += s.size[small as usize];
    s.odd[big as usize] ^= s.odd[small as usize];
    s.boundary[big as usize] |= s.boundary[small as usize];
    // Append, so the smaller cluster's list keeps its memory for the next decode.
    let mut moved = std::mem::take(&mut s.frontier[small as usize]);
    s.frontier[big as usize].append(&mut moved);
    s.frontier[small as usize] = moved;
    big
}

/// Put back everything the decode touched.
fn reset(s: &mut UfScratch) {
    let boundary = s.parent.len() - 1;
    for &v in &s.touched_nodes {
        let v = v as usize;
        s.parent[v] = v as u32;
        s.size[v] = 1;
        s.odd[v] = false;
        s.boundary[v] = v == boundary;
        s.frontier[v].clear();
        s.clustered[v] = false;
        s.defect[v] = false;
        s.visited[v] = false;
        s.via[v] = NO_EDGE;
        s.node_touched[v] = false;
    }
    s.touched_nodes.clear();
    for &id in &s.touched_edges {
        let i = id as usize;
        s.support[i] = 0;
        s.since[i] = 0;
        s.rate[i] = 0;
        s.grown[i] = false;
        s.edge_touched[i] = false;
    }
    s.touched_edges.clear();
    s.events.clear();
}

#[cfg(test)]
mod tests {
    use crate::circuit::Circuit;
    use crate::dem::Dem;
    use crate::dem_decoder::DemDecoder;
    use crate::frame_sampler::FrameSampler;
    use crate::sparse::SparseGraph;
    use crate::surface_code::Xorshift;

    fn graph(d: usize, rounds: usize, p: f64) -> (Circuit, Dem, SparseGraph) {
        use crate::memory::{generate, CodeKind, NoiseModel};
        let c = generate(CodeKind::Rotated, d, rounds, NoiseModel::Sd6 { p }, crate::circuit::Basis::Z).unwrap();
        let dem = Dem::from_circuit(&c).unwrap();
        let (g, _) = DemDecoder::new(&dem).unwrap().into_parts(false);
        (c, dem, g)
    }

    /// The correction's edges flip exactly the detection events.
    fn explains(g: &SparseGraph, correction: &[u32], defects: &[u32]) -> bool {
        let mut flipped = vec![false; g.num_nodes + 1];
        for &id in correction {
            let (a, b) = g.edge_ends(id);
            flipped[a as usize] ^= true;
            flipped[b as usize] ^= true;
        }
        let mut want = vec![false; g.num_nodes + 1];
        for &d in defects {
            want[d as usize] ^= true;
        }
        flipped[..g.num_nodes] == want[..g.num_nodes]
    }

    #[test]
    fn corrections_explain_every_detection_event() {
        let (c, _, g) = graph(5, 5, 0.01);
        let sampler = FrameSampler::new(&c).unwrap();
        let mut rng = Xorshift::new(7);
        let mut s = g.union_find_scratch();
        for _ in 0..2000 {
            let shot = sampler.sample(&mut rng);
            let defects: Vec<u32> = shot.detectors.iter().enumerate().filter(|x| *x.1).map(|x| x.0 as u32).collect();
            g.decode_union_find(&mut s, &defects).expect("a surface code's syndromes always have a correction");
            assert!(explains(&g, &s.correction, &defects), "{defects:?}");
        }
    }

    #[test]
    fn every_single_fault_is_corrected() {
        let (_, dem, g) = graph(5, 5, 0.003);
        let mut s = g.union_find_scratch();
        for m in &dem.mechanisms {
            let got = g.decode_union_find(&mut s, &m.detectors).unwrap();
            assert_eq!(got, m.observables, "{:?}", m.detectors);
        }
    }

    #[test]
    fn nearly_as_good_as_matching_and_better_with_distance() {
        let rate = |d: usize, p: f64, shots: usize| {
            let (c, dem, g) = graph(d, d, p);
            let dec = DemDecoder::new(&dem).unwrap();
            let sampler = FrameSampler::new(&c).unwrap();
            let mut rng = Xorshift::new(11);
            let mut s = g.union_find_scratch();
            let (mut uf, mut mwpm) = (0, 0);
            for _ in 0..shots {
                let shot = sampler.sample(&mut rng);
                let defects: Vec<u32> = shot.detectors.iter().enumerate().filter(|x| *x.1).map(|x| x.0 as u32).collect();
                uf += (g.decode_union_find(&mut s, &defects).unwrap() != shot.observables) as usize;
                mwpm += (dec.decode(&defects).unwrap().observables != shot.observables) as usize;
            }
            (uf as f64 / shots as f64, mwpm as f64 / shots as f64)
        };
        let (uf3, m3) = rate(3, 0.003, 20_000);
        let (uf5, m5) = rate(5, 0.003, 20_000);
        assert!(uf3 < 1.6 * m3 + 0.002 && uf5 < 1.6 * m5 + 0.002, "{uf3} {m3} {uf5} {m5}");
        assert!(uf5 < uf3, "{uf5} {uf3}");
    }

    #[test]
    fn scratch_is_reset_between_decodes() {
        let (_, dem, g) = graph(3, 3, 0.01);
        let mut s = g.union_find_scratch();
        let first: Vec<u64> = dem.mechanisms.iter().map(|m| g.decode_union_find(&mut s, &m.detectors).unwrap()).collect();
        let again: Vec<u64> = dem.mechanisms.iter().map(|m| g.decode_union_find(&mut s, &m.detectors).unwrap()).collect();
        assert_eq!(first, again);
        assert_eq!(g.decode_union_find(&mut s, &[]), Some(0));
        assert_eq!(g.decode_union_find(&mut s, &[u32::MAX]), None);
    }
}

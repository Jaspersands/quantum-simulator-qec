//! Decoding one shot: the Möbius matching, its Euler tours, and each tour lifted back to the
//! code by carrying colour charge around it (Chromobius's `decode/decoder.cc` and
//! `graph/euler_tours.cc`).

use std::sync::Arc;

use super::{next_charge, Lifted, Options, NEUTRAL};
use crate::dem::Dem;
use crate::pm_exact::{Graph, Mwpm, NeighbourOrder, NUM_DISTINCT_WEIGHTS};

/// A colour-code decoder: the lifted model and the matching graph of its Möbius model.
pub struct ColorDecoder {
    lifted: Lifted,
    /// The Möbius model's graph as Chromobius's matcher builds it: PyMatching 2.2.2's
    /// (`pm::detector_error_model_to_mwpm(dem, 1 << 24, true)`), 2²⁴ weight levels, each node's
    /// neighbours in 2.2.2's order. Each thread decodes on its own `Mwpm` over it.
    graph: Arc<Graph>,
    /// The Möbius model, for inspection.
    pub mobius: Dem,
}

/// One decode's state; one per thread.
pub struct ColorWork {
    mwpm: Mwpm,
    euler: Vec<EulerNode>,
    cycle: Vec<u32>,
    cycle2: Vec<u32>,
    used: Vec<u32>,
    events: Vec<u64>,
    edges: Vec<(u32, u32)>,
    /// Whether the last decode lifted through a table entry Chromobius might have filled
    /// otherwise (see the module comment).
    pub tied: bool,
}

#[derive(Clone, Default)]
struct EulerNode {
    /// (neighbour, the index of this node in the neighbour's list); a used edge's far end is
    /// `BOUNDARY`.
    neighbors: Vec<(u32, u32)>,
    next: usize,
}

impl EulerNode {
    fn look_next(&mut self) -> Option<usize> {
        while self.next < self.neighbors.len() {
            if self.neighbors[self.next].0 != super::BOUNDARY {
                return Some(self.next);
            }
            self.next += 1;
        }
        None
    }
}

/// How a decode failed.
#[derive(Clone, Debug, PartialEq)]
pub enum ColorError {
    /// The Möbius matcher found no perfect matching: some component of the Möbius graph
    /// without a boundary holds an odd number of detection events (a detector no fault
    /// touches fired, say). Chromobius's matcher then throws PyMatching's
    /// [`NO_PERFECT_MATCHING`](crate::pm_exact::NO_PERFECT_MATCHING), the only refusal a
    /// shot can meet here (its events are always nodes of the graph).
    Unmatchable,
    /// The matched edges did not fall into closed tours.
    NotEulerian,
    /// A tour no start charge explains (wrong annotations, or a model Chromobius cannot
    /// decode). The tour's detectors.
    Unliftable(Vec<u32>),
}

impl ColorDecoder {
    /// The decoder for a model, given unrolled (`dem`) and as written (`program`, whose
    /// declarations give the detectors' colours as Chromobius reads them).
    pub fn from_dem(dem: &Dem, program: &crate::dem_program::DemProgram, ignore_decomposition_failures: bool) -> Result<ColorDecoder, String> {
        let options = Options { ignore_decomposition_failures, ..Options::default() };
        let lifted = Lifted::from_dem(dem, program, options)?;
        let mobius = Dem::parse(&lifted.mobius_text)?;
        // The Möbius model's edges, merged as PyMatching merges them. One of probability above
        // 1/2 has a negative weight, which PyMatching matches at its absolute value with the
        // edge pre-applied.
        let user = crate::user_graph::UserGraph::from_dem(&crate::dem_program::DemProgram::parse(&lifted.mobius_text)?)?;
        for e in &user.edges {
            let p = e.error_probability;
            if !(p > 0.0 && p < 1.0) {
                // (At probability 1 PyMatching's weights, and so Chromobius's, are NaN.)
                return Err(format!("Möbius edge ({}, {}) has probability {p}; Chromobius's matcher needs 0 < p < 1", e.node1, e.node2.map_or(-1, |v| v as i64)));
            }
        }
        let graph = Arc::new(Graph::try_from_user_graph(&user, NeighbourOrder::V2_2, NUM_DISTINCT_WEIGHTS)?);
        Ok(ColorDecoder { lifted, graph, mobius })
    }

    pub fn num_detectors(&self) -> usize {
        self.lifted.colors.len()
    }

    /// The Möbius graph's nodes; this index stands for the boundary in its edges.
    pub fn num_mobius_nodes(&self) -> usize {
        self.graph.num_nodes
    }

    /// The Möbius model as Stim's text.
    pub fn mobius_text(&self) -> &str {
        &self.lifted.mobius_text
    }

    /// The decoder's tables and Möbius model as `chromobius describe_decoder` prints them.
    pub fn describe(&self) -> Result<String, String> {
        self.lifted.describe()
    }

    pub fn work(&self) -> ColorWork {
        ColorWork {
            mwpm: Mwpm::new(self.graph.clone()),
            euler: vec![EulerNode::default(); 2 * self.lifted.colors.len()],
            cycle: Vec::new(),
            cycle2: Vec::new(),
            used: Vec::new(),
            events: Vec::new(),
            edges: Vec::new(),
            tied: false,
        }
    }

    /// The Möbius detection events of a shot (`detection_events_to_mobius_detection_events`):
    /// both copies of each fired detector not ignored, in order.
    fn mobius_events(&self, defects: &[u32], w: &mut ColorWork) {
        w.events.clear();
        for &d in defects {
            if !self.lifted.colors[d as usize].ignored {
                w.events.push(2 * d as u64);
                w.events.push(2 * d as u64 + 1);
            }
        }
    }

    /// Decode one shot given as the fired detectors' bits (`fired(d)`) and their indices,
    /// ascending: the observables, and the Möbius matching's weight as Chromobius reports it.
    ///
    /// As Chromobius's `decode_detection_events`: PyMatching's
    /// `decode_detection_events_to_edges` lists the matching's edges in its order (the
    /// negative-weight ones it leaves flipped among them), and the edges, with each event's two
    /// copies joined, are walked into tours in that order and lifted. The weight is
    /// `PymatchingMatcher::match_edges`'s: each listed edge's discretised weight as its second
    /// node holds it (the absolute value, as the search graph keeps it), summed, over the
    /// graph's normalising constant.
    pub fn decode(&self, defects: &[u32], fired: &dyn Fn(u32) -> bool, w: &mut ColorWork) -> Result<(u64, f64), ColorError> {
        w.tied = false;
        self.mobius_events(defects, w);
        if w.events.is_empty() && self.graph.negative_weight_edges.is_empty() {
            return Ok((0, 0.0));
        }
        // (The matcher's only refusal here is PyMatching's "No perfect matching": every Möbius
        // event is a node of the graph, and every match is joined by a path.)
        let listed = w.mwpm.try_decode_to_edges(&w.events).map_err(|e| {
            debug_assert_eq!(e, crate::pm_exact::NO_PERFECT_MATCHING);
            ColorError::Unmatchable
        })?;
        let n = self.graph.num_nodes as u32;
        let mut edges = std::mem::take(&mut w.edges);
        edges.clear();
        let mut total = 0i64;
        for &(a, b) in &listed {
            // (Never the boundary, `-1`: every Möbius error joins two copies.)
            let (a, b) = (a as u32, if b < 0 { n } else { b as u32 });
            if b != n {
                let k = self.graph.index_of_neighbor(b, a).expect("a listed edge is in the graph");
                total += self.graph.neighbor_weights(b)[k] as i64;
            }
            edges.push((a, b));
        }
        let weight = total as f64 / self.graph.normalising_constant;
        let result = self.lift(&edges, fired, w).map(|solution| (solution, weight));
        w.edges = edges;
        result
    }

    /// The lifting alone, of a Möbius matching given as its edges in the order a matcher
    /// gives them (Chromobius's tours, and so its predictions, depend on that order): the
    /// observables. `defects` are the shot's fired detectors, as for `decode`.
    pub fn decode_given_edges(&self, defects: &[u32], edges: &[(u32, u32)], fired: &dyn Fn(u32) -> bool, w: &mut ColorWork) -> Result<u64, ColorError> {
        w.tied = false;
        self.mobius_events(defects, w);
        if w.events.is_empty() && edges.is_empty() {
            return Ok(0);
        }
        let n = self.graph.num_nodes as u32;
        if edges.iter().any(|&(a, b)| a > n || b > n) {
            return Err(ColorError::NotEulerian);
        }
        self.lift(edges, fired, w)
    }

    /// The matched edges (and each event's two copies) walked into tours, each lifted.
    fn lift(&self, edges: &[(u32, u32)], fired: &dyn Fn(u32) -> bool, w: &mut ColorWork) -> Result<u64, ColorError> {
        let boundary = self.graph.num_nodes as u32;
        let mut flat: Vec<u32> = Vec::with_capacity(2 * edges.len());
        for &(a, b) in edges {
            if a == boundary || b == boundary {
                return Err(ColorError::NotEulerian);
            }
            flat.push(a);
            flat.push(b);
        }
        for k in (0..flat.len()).step_by(2) {
            add_edge(&mut w.euler, flat[k], flat[k + 1]);
        }
        for k in (0..w.events.len()).step_by(2) {
            add_edge(&mut w.euler, w.events[k] as u32, w.events[k + 1] as u32);
        }
        let mut solution = 0u64;
        let mut result = Ok(());
        for &n in &flat {
            match self.burn(n, fired, w) {
                Ok(Some(obs)) => solution ^= obs,
                Ok(None) => {}
                Err(e) => {
                    result = Err(e);
                    break;
                }
            }
        }
        // Clear what this shot touched (Chromobius resets the matched nodes; the event copies
        // are all among them, since every event is matched).
        for n in flat.iter().map(|&n| n as usize).chain(w.events.iter().map(|&n| n as usize)) {
            let node = &mut w.euler[n];
            node.neighbors.clear();
            node.next = 0;
        }
        w.cycle.clear();
        w.cycle2.clear();
        result.map(|()| solution)
    }

    /// `burn_component_at`: the tour through `n`'s unused edges, lifted.
    fn burn(&self, n: u32, fired: &dyn Fn(u32) -> bool, w: &mut ColorWork) -> Result<Option<u64>, ColorError> {
        if w.euler[n as usize].look_next().is_none() {
            return Ok(None);
        }
        w.cycle.push(n);
        loop {
            extend_cycle(&mut w.euler, &mut w.cycle);
            if !rotate_cycle(&mut w.euler, &mut w.cycle, &mut w.cycle2)? {
                break;
            }
        }
        let obs = self.discharge(fired, w)?;
        w.cycle.clear();
        Ok(Some(obs))
    }

    /// `discharge_cycle`: the first start charge whose walk around the tour comes back to it.
    fn discharge(&self, fired: &dyn Fn(u32) -> bool, w: &mut ColorWork) -> Result<u64, ColorError> {
        for start in 0..4 {
            if let Some(obs) = self.walk(start, fired, w) {
                return Ok(obs);
            }
        }
        Err(ColorError::Unliftable(w.cycle.iter().map(|&n| n >> 1).collect()))
    }

    /// `discharge_cycle_helper_single_start_charge_many_cur_charge`.
    fn walk(&self, start: u8, fired: &dyn Fn(u32) -> bool, w: &mut ColorWork) -> Option<u64> {
        let colors = &self.lifted.colors;
        let reps = &self.lifted.rgb_reps;
        w.used.clear();
        let mut cur: [Option<u64>; 4] = [None; 4];
        cur[start as usize] = Some(0);
        let mut loc = *w.cycle.last().unwrap() >> 1;
        for k in 0..w.cycle.len() {
            let next = w.cycle[k] >> 1;
            if next == loc && fired(loc) && !w.used.contains(&loc) {
                // Pick up the detection event.
                w.used.push(loc);
                let det = colors[loc as usize].color;
                let mut after: [Option<u64>; 4] = [None; 4];
                after[det as usize] = cur[NEUTRAL as usize];
                after[NEUTRAL as usize] = cur[det as usize];
                let r = reps[loc as usize];
                if r.weight() == 3 {
                    let c1 = next_charge(det);
                    let c2 = next_charge(c1);
                    if let Some(f) = cur[c1 as usize] {
                        after[c2 as usize] = Some(f ^ r.obs);
                    }
                    if let Some(f) = cur[c2 as usize] {
                        after[c1 as usize] = Some(f ^ r.obs);
                    }
                }
                cur = after;
            } else {
                // Drag the charge to near the next location, perhaps changing its colour.
                let mut after: [Option<u64>; 4] = [None; 4];
                if let Some(table) = self.lifted.drag.step(loc, next) {
                    for (c, state) in cur.iter().enumerate() {
                        let Some(f) = state else { continue };
                        for (nc, slot) in after.iter_mut().enumerate() {
                            if let Some((flip, ambiguous)) = table[4 * c + nc] {
                                *slot = Some(f ^ flip);
                                w.tied |= ambiguous;
                            }
                        }
                    }
                }
                cur = after;
            }
            loc = next;
        }
        cur[start as usize]
    }
}

fn add_edge(nodes: &mut [EulerNode], a: u32, b: u32) {
    let na = nodes[a as usize].neighbors.len() as u32;
    let nb = nodes[b as usize].neighbors.len() as u32;
    nodes[a as usize].neighbors.push((b, nb));
    nodes[b as usize].neighbors.push((a, na));
}

/// `extend_cycle_depth_first`: walk unused edges from the cycle's end until stuck.
fn extend_cycle(nodes: &mut [EulerNode], cycle: &mut Vec<u32>) {
    loop {
        let at = *cycle.last().unwrap() as usize;
        let Some(k) = nodes[at].look_next() else { return };
        nodes[at].next += 1;
        let (to, back) = nodes[at].neighbors[k];
        cycle.push(to);
        nodes[to as usize].neighbors[back as usize].0 = super::BOUNDARY;
    }
}

/// `rotate_cycle_to_end_with_unfinished_node`: a closed walk rotated to end at a node with
/// unused edges, to extend again; false when none is left.
fn rotate_cycle(nodes: &mut [EulerNode], cycle: &mut Vec<u32>, buf: &mut Vec<u32>) -> Result<bool, ColorError> {
    if cycle.last() != cycle.first() {
        return Err(ColorError::NotEulerian);
    }
    cycle.pop();
    let mut k = 1;
    while k < cycle.len() && nodes[cycle[k] as usize].look_next().is_none() {
        k += 1;
    }
    if k < cycle.len() {
        buf.extend_from_slice(&cycle[k..]);
        buf.extend_from_slice(&cycle[..=k]);
        std::mem::swap(cycle, buf);
        buf.clear();
        Ok(true)
    } else {
        Ok(false)
    }
}

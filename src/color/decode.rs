//! Decoding one shot: the Möbius matching, its Euler tours, and each tour lifted back to the
//! code by carrying colour charge around it (Chromobius's `decode/decoder.cc` and
//! `graph/euler_tours.cc`).

use super::{next_charge, Lifted, Options, NEUTRAL};
use crate::dem::Dem;
use crate::sparse::{Scratch, SparseGraph};

/// A colour-code decoder: the lifted model and the matcher of its Möbius graph.
pub struct ColorDecoder {
    lifted: Lifted,
    graph: SparseGraph,
    /// The Möbius model, for inspection.
    pub mobius: Dem,
    /// Each Möbius edge's weight as PyMatching discretizes it for Chromobius (2²⁴ levels),
    /// keyed by its nodes in order, and the constant a sum of them is divided by.
    pm_weights: std::collections::HashMap<(u32, u32), i64>,
    pm_scale: f64,
    /// The Möbius edges of probability above 1/2 (negative weight), sorted, and the nodes
    /// they toggle, sorted: PyMatching matches such an edge at its absolute weight with its
    /// endpoints' detection toggled and the edge itself counted in (see `decode`).
    neg_edges: Vec<(u32, u32)>,
    neg_nodes: Vec<u32>,
}

/// One decode's state; one per thread.
pub struct ColorWork {
    scratch: Scratch,
    euler: Vec<EulerNode>,
    cycle: Vec<u32>,
    cycle2: Vec<u32>,
    used: Vec<u32>,
    events: Vec<u32>,
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
    /// The Möbius matching found no matching (a detection event with no partner).
    Unmatchable,
    /// The matched edges did not fall into closed tours.
    NotEulerian,
    /// A tour no start charge explains (wrong annotations, or a model Chromobius cannot
    /// decode). The tour's detectors.
    Unliftable(Vec<u32>),
}

impl ColorDecoder {
    pub fn from_dem(dem: &Dem, ignore_decomposition_failures: bool) -> Result<ColorDecoder, String> {
        let options = Options { ignore_decomposition_failures, ..Options::default() };
        let lifted = Lifted::from_dem(dem, options)?;
        let mobius = Dem::parse(&lifted.mobius_text)?;
        // The Möbius graph's edges, merged; one of probability above 1/2 has a negative weight,
        // which PyMatching (Chromobius's matcher) takes at its absolute value with the edge
        // pre-applied.
        let (mut edges, _) = crate::dem_decoder::merged_edges_any(&mobius)?;
        let boundary = mobius.num_detectors as u32;
        let mut neg_edges = Vec::new();
        let mut toggled = std::collections::BTreeSet::new();
        for e in edges.iter_mut() {
            if !(e.2 > 0.0 && e.2 < 1.0) {
                // (At probability 1 PyMatching's weights, and so Chromobius's, are NaN.)
                return Err(format!("Möbius edge ({}, {}) has probability {}; Chromobius's matcher needs 0 < p < 1", e.0, e.1, e.2));
            }
            if e.2 > 0.5 {
                neg_edges.push((e.0.min(e.1), e.0.max(e.1)));
                for n in [e.0, e.1] {
                    if n != boundary && !toggled.remove(&n) {
                        toggled.insert(n);
                    }
                }
                e.2 = 1.0 - e.2;
            }
        }
        neg_edges.sort_unstable();
        let graph = SparseGraph::from_edges(mobius.num_detectors, &edges);
        // Chromobius's weight is PyMatching's: the matched edges' weights as PyMatching's
        // search graph holds them (`pm::detector_error_model_to_mwpm(dem, 1 << 24, true)`),
        // summed and divided by the graph's normalising constant.
        let user = crate::user_graph::UserGraph::from_dem(&crate::dem_program::DemProgram::parse(&lifted.mobius_text)?)?;
        let c = user.normalising_constant()?;
        let mut pm_weights = std::collections::HashMap::new();
        for e in &user.edges {
            if let Some(v) = e.node2 {
                let w = ((e.weight * c).round() as i32 as i64 * 2).abs();
                pm_weights.insert((e.node1.min(v) as u32, e.node1.max(v) as u32), w);
            }
        }
        Ok(ColorDecoder { lifted, graph, mobius, pm_weights, pm_scale: 2.0 * c, neg_edges, neg_nodes: toggled.into_iter().collect() })
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

    pub fn work(&self) -> ColorWork {
        ColorWork {
            scratch: Scratch::new(&self.graph),
            euler: vec![EulerNode::default(); 2 * self.lifted.colors.len()],
            cycle: Vec::new(),
            cycle2: Vec::new(),
            used: Vec::new(),
            events: Vec::new(),
            tied: false,
        }
    }

    /// Decode one shot given as the fired detectors' bits (`fired(d)`) and their indices,
    /// ascending: the observables, and the Möbius matching's weight.
    pub fn decode(&self, defects: &[u32], fired: &dyn Fn(u32) -> bool, w: &mut ColorWork) -> Result<(u64, f64), ColorError> {
        w.tied = false;
        w.events.clear();
        for &d in defects {
            if !self.lifted.colors[d as usize].ignored {
                w.events.push(2 * d);
                w.events.push(2 * d + 1);
            }
        }
        if w.events.is_empty() && self.neg_edges.is_empty() {
            return Ok((0, 0.0));
        }
        let (prediction, edges) = if self.neg_edges.is_empty() {
            self.graph.decode_with_edges(&mut w.scratch, &w.events).map_err(|_| ColorError::Unmatchable)?
        } else {
            self.match_with_negative_edges(w)?
        };
        let pm_sum: Option<i64> = edges.iter().map(|&(a, b)| self.pm_weights.get(&(a.min(b), a.max(b))).copied()).sum();
        let weight = pm_sum.map_or(prediction.weight, |w| w as f64 / self.pm_scale);
        self.lift(&edges, fired, w).map(|solution| (solution, weight))
    }

    /// PyMatching's matching where some edges have negative weights: the detection events
    /// toggled at those edges' ends, matched at absolute weights, and the edges the correction
    /// flips (`decode_detection_events_to_edges`): the matched ones those edges do not cancel,
    /// then the negative-weight edges left, in order.
    fn match_with_negative_edges(&self, w: &mut ColorWork) -> Result<(crate::dem_decoder::Prediction, Vec<(u32, u32)>), ColorError> {
        let mut defects: Vec<u32> = Vec::with_capacity(w.events.len() + self.neg_nodes.len());
        let (mut i, mut j) = (0, 0);
        while i < w.events.len() || j < self.neg_nodes.len() {
            let (a, b) = (w.events.get(i).copied(), self.neg_nodes.get(j).copied());
            match (a, b) {
                (Some(x), Some(y)) if x == y => {
                    i += 1;
                    j += 1;
                }
                (Some(x), Some(y)) if x < y => {
                    defects.push(x);
                    i += 1;
                }
                (Some(x), None) => {
                    defects.push(x);
                    i += 1;
                }
                (_, Some(y)) => {
                    defects.push(y);
                    j += 1;
                }
                (None, None) => break,
            }
        }
        let (prediction, matched) = if defects.is_empty() {
            (crate::dem_decoder::Prediction { observables: 0, weight: 0.0, iweight: 0 }, Vec::new())
        } else {
            self.graph.decode_with_edges(&mut w.scratch, &defects).map_err(|_| ColorError::Unmatchable)?
        };
        let mut left: std::collections::BTreeSet<(u32, u32)> = self.neg_edges.iter().copied().collect();
        let mut edges = Vec::with_capacity(matched.len() + left.len());
        for (a, b) in matched {
            if !left.remove(&(a.min(b), a.max(b))) {
                edges.push((a, b));
            }
        }
        edges.extend(left);
        Ok((prediction, edges))
    }

    /// The lifting alone, of a Möbius matching given as its edges in the order a matcher
    /// gives them (Chromobius's tours, and so its predictions, depend on that order): the
    /// observables. `defects` are the shot's fired detectors, as for `decode`.
    pub fn decode_given_edges(&self, defects: &[u32], edges: &[(u32, u32)], fired: &dyn Fn(u32) -> bool, w: &mut ColorWork) -> Result<u64, ColorError> {
        w.tied = false;
        w.events.clear();
        for &d in defects {
            if !self.lifted.colors[d as usize].ignored {
                w.events.push(2 * d);
                w.events.push(2 * d + 1);
            }
        }
        if w.events.is_empty() {
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
            add_edge(&mut w.euler, w.events[k], w.events[k + 1]);
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
        for &n in flat.iter().chain(&w.events) {
            let node = &mut w.euler[n as usize];
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

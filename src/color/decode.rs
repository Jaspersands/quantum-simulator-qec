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
        let (graph, _) = crate::dem_decoder::DemDecoder::new(&mobius)?.into_parts(false);
        Ok(ColorDecoder { lifted, graph, mobius })
    }

    pub fn num_detectors(&self) -> usize {
        self.lifted.colors.len()
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
        if w.events.is_empty() {
            return Ok((0, 0.0));
        }
        let (prediction, edges) = self.graph.decode_with_edges(&mut w.scratch, &w.events).map_err(|_| ColorError::Unmatchable)?;
        let weight = prediction.weight;
        let boundary = self.graph.num_nodes as u32;
        let mut flat: Vec<u32> = Vec::with_capacity(2 * edges.len());
        for &(a, b) in &edges {
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
        result.map(|()| (solution, weight))
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

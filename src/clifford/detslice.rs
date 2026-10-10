//! Stim's detector slices (`detector_slice_set.cc`): what each detector and observable is
//! sensitive to at chosen ticks, found by running the circuit backwards, and the text diagram
//! of them, drawn with a port of Stim's `AsciiDiagram`.

use std::collections::{BTreeMap, BTreeSet};

use super::ir::{self, GateTarget, Instruction, Item};
use super::rev_tracker::{dem_target_str, is_observable, RevTracker};
use super::transform::{count_detectors, detector_coordinates, final_qubit_coordinates};
use crate::dem_program::fmt_g;

/// One entry of `filter_coords`: a detector or observable, or a coordinate prefix (NaN matches
/// anything) that picks detectors.
#[derive(Clone, Debug, Default)]
pub struct CoordFilter {
    pub coords: Vec<f64>,
    pub target: Option<u64>,
}

impl CoordFilter {
    pub fn matches(&self, coords: &[f64], target: u64) -> bool {
        if let Some(t) = self.target {
            return t == target;
        }
        if is_observable(target) {
            return false;
        }
        self.coords.iter().enumerate().all(|(k, &c)| c.is_nan() || (k < coords.len() && coords[k] == c))
    }
}

type Key = (u64, u64);

#[derive(Clone, Debug, Default)]
pub struct SliceSet {
    pub num_qubits: usize,
    pub min_tick: u64,
    pub num_ticks: u64,
    pub coordinates: BTreeMap<u64, Vec<f64>>,
    pub detector_coordinates: BTreeMap<u64, Vec<f64>>,
    /// (tick, detector or observable) to its Paulis.
    pub slices: BTreeMap<Key, Vec<GateTarget>>,
    pub anticommutations: BTreeMap<Key, Vec<GateTarget>>,
}

struct Computer<'a> {
    tracker: RevTracker,
    tick_cur: u64,
    first: u64,
    num: u64,
    used_qubits: BTreeSet<u32>,
    out: &'a mut SliceSet,
}

impl Computer<'_> {
    fn process_anticommutations(&mut self, out_tick: u64) {
        let n = self.tracker.xs.len();
        let found: Vec<(u64, u32)> = self.tracker.anticommutations.iter().copied().collect();
        for (d, g) in found {
            self.out.anticommutations.entry((out_tick, d)).or_default().push(GateTarget(g));
            // Stop propagating it backwards once it broke.
            for q in 0..n {
                if self.tracker.xs[q].0.contains(&d) {
                    self.tracker.xs[q].xor_item(d);
                }
                if self.tracker.zs[q].0.contains(&d) {
                    self.tracker.zs[q].xor_item(d);
                }
            }
        }
        self.tracker.anticommutations.clear();
    }

    fn on_tick(&mut self) {
        self.process_anticommutations(self.tick_cur + 1);
        for q in 0..self.tracker.xs.len() {
            let mut xs: BTreeSet<u64> = self.tracker.xs[q].0.clone();
            let mut ys = BTreeSet::new();
            let mut zs = BTreeSet::new();
            for &t in self.tracker.zs[q].0.iter() {
                if xs.remove(&t) {
                    ys.insert(t);
                } else {
                    zs.insert(t);
                }
            }
            for (set, p) in [(xs, 1u8), (ys, 2), (zs, 3)] {
                for t in set {
                    self.out.slices.entry((self.tick_cur, t)).or_default().push(GateTarget::pauli(q as u32, p, false));
                }
            }
        }
    }

    fn process_tick(&mut self) -> bool {
        if self.tick_cur >= self.first && self.tick_cur < self.first.saturating_add(self.num) {
            self.on_tick();
        }
        self.tick_cur = self.tick_cur.wrapping_sub(1);
        self.tick_cur.wrapping_add(1) < self.first
    }

    fn block(&mut self, c: &ir::Circuit) -> Result<bool, String> {
        for it in c.items.iter().rev() {
            if self.item(it)? {
                return Ok(true);
            }
        }
        Ok(false)
    }

    fn item(&mut self, it: &Item) -> Result<bool, String> {
        match it {
            Item::Op(op) => self.op(op),
            Item::Repeat { count, body, .. } => {
                let stop = self.first.saturating_add(self.num);
                let max_skip = self.tick_cur.max(stop) - stop;
                let mut reps = *count;
                let per = body.count_ticks();
                let skipped = if max_skip == 0 {
                    0
                } else if per == 0 {
                    reps
                } else {
                    reps.min(max_skip / per)
                };
                if skipped > 0 {
                    self.tracker.undo_loop(body, skipped)?;
                    reps -= skipped;
                    self.tick_cur -= per * skipped;
                }
                while reps > 0 {
                    if self.block(body)? {
                        return Ok(true);
                    }
                    reps -= 1;
                }
                Ok(false)
            }
        }
    }

    fn op(&mut self, op: &Instruction) -> Result<bool, String> {
        if op.gate.name == "TICK" {
            return Ok(self.process_tick());
        }
        for t in &op.targets {
            if t.has_qubit_value() {
                self.used_qubits.insert(t.value());
            }
        }
        self.tracker.undo_gate(op)?;
        Ok(false)
    }
}

impl SliceSet {
    /// Stim's `DetectorSliceSet::from_circuit_ticks`: the slices at ticks `start_tick` ..
    /// `start_tick + num_ticks` (tick 0 is the start; tick k is just after the k-th TICK).
    pub fn from_circuit_ticks(c: &ir::Circuit, start_tick: u64, num_ticks: u64, filters: &[CoordFilter]) -> Result<SliceSet, String> {
        let ticks = c.count_ticks();
        let num_ticks = num_ticks.min(ticks.wrapping_sub(start_tick).wrapping_add(1)).max(1);
        let nq = c.count_qubits() as usize;
        let mut out = SliceSet { num_qubits: nq, min_tick: start_tick, num_ticks, ..Default::default() };
        {
            let mut h = Computer { tracker: RevTracker::new(nq, c.count_measurements(), count_detectors(c), false), tick_cur: ticks + 1, first: start_tick, num: num_ticks, used_qubits: BTreeSet::new(), out: &mut out };
            if !h.process_tick() && !h.block(c)? {
                h.tracker.undo_implicit_rzs_at_start_of_circuit()?;
                h.process_anticommutations(1);
            }
            let used = std::mem::take(&mut h.used_qubits);
            drop(h);
            out.coordinates = final_qubit_coordinates(c);
            for q in used {
                out.coordinates.entry(q as u64).or_default();
            }
        }
        let included: BTreeSet<u64> = out.slices.keys().filter(|k| !is_observable(k.1)).map(|k| k.1).collect();
        out.detector_coordinates = detector_coordinates(c, &included)?;
        let keep = |t: u64, dc: &BTreeMap<u64, Vec<f64>>| -> bool {
            let coords: &[f64] = if is_observable(t) { &[] } else { dc.get(&t).map(|v| v.as_slice()).unwrap_or(&[]) };
            filters.iter().any(|f| f.matches(coords, t))
        };
        let removed: Vec<Key> = out.slices.keys().filter(|k| !keep(k.1, &out.detector_coordinates)).copied().collect();
        for k in removed {
            out.slices.remove(&k);
            if !is_observable(k.1) {
                out.detector_coordinates.remove(&k.1);
            }
        }
        Ok(out)
    }

    /// Stim's `write_text_diagram_to`.
    pub fn text(&self) -> String {
        let mut d = Ascii::default();
        let mut cur_moment = 0usize;
        let mut used = vec![false; self.num_qubits];
        let m2x = |m: usize| m * 3 + 2;
        let q2y = |q: usize| q * 2 + 1;
        let reserve = |targets: &[GateTarget], d: &mut Ascii, cur_moment: &mut usize, used: &mut Vec<bool>| {
            let qs: Vec<usize> = targets.iter().filter(|t| !(t.is_combiner() || t.is_record() || t.is_sweep())).map(|t| t.value() as usize).collect();
            let (Some(&lo), Some(&hi)) = (qs.iter().min(), qs.iter().max()) else { return };
            if (lo..=hi).any(|q| used.get(q).copied().unwrap_or(false)) {
                *cur_moment += 1;
                used.iter_mut().for_each(|u| *u = false);
            }
            for q in lo..=hi {
                if q < used.len() {
                    used[q] = true;
                }
            }
            if lo < hi {
                d.lines.push(((m2x(*cur_moment), q2y(lo), 0.0, 0.5), (m2x(*cur_moment), q2y(hi), 0.0, 0.5)));
            }
        };
        for ((_, t), gs) in &self.anticommutations {
            reserve(gs, &mut d, &mut cur_moment, &mut used);
            for g in gs {
                d.add((m2x(cur_moment + 1), q2y(g.value() as usize), 0.0, 0.5), format!("ANTICOMMUTED:{}", dem_target_str(*t)));
            }
        }
        for ((_, t), gs) in &self.slices {
            reserve(gs, &mut d, &mut cur_moment, &mut used);
            for g in gs {
                let p = if g.is_x() {
                    "X"
                } else if g.is_y() {
                    "Y"
                } else if g.is_z() {
                    "Z"
                } else {
                    "?"
                };
                d.add((m2x(cur_moment), q2y(g.value() as usize), 0.0, 0.5), format!("{p}:{}", dem_target_str(*t)));
            }
        }
        let mut lines: Vec<Line> = (0..self.num_qubits).map(|q| ((0, q2y(q), 1.0, 0.5), (m2x(cur_moment) + 1, q2y(q), 1.0, 0.5))).collect();
        lines.append(&mut d.lines);
        d.lines = lines;
        for q in 0..self.num_qubits {
            let mut label = format!("q{q}:");
            if let Some(c) = self.coordinates.get(&(q as u64)) {
                if !c.is_empty() {
                    label.push_str(&format!("({})", c.iter().map(|&v| fmt_g(v, 6)).collect::<Vec<_>>().join(", ")));
                }
            }
            label.push(' ');
            d.add((0, q2y(q), 1.0, 0.5), label);
        }
        d.render()
    }
}

/// A position: column, row, and the alignment within the cell.
type Pos = (usize, usize, f32, f32);
type Line = (Pos, Pos);

/// Stim's `AsciiDiagram`: labels in cells of variable-width columns, and lines between cells.
#[derive(Default)]
pub struct Ascii {
    /// By (column, row); the first label put in a cell stays.
    cells: BTreeMap<(usize, usize), (f32, f32, String)>,
    lines: Vec<Line>,
}

impl Ascii {
    fn add(&mut self, p: Pos, label: String) {
        self.cells.entry((p.0, p.1)).or_insert((p.2, p.3, label));
    }

    fn render(&self) -> String {
        let (mut nx, mut ny) = (0usize, 0usize);
        for &(x, y) in self.cells.keys() {
            nx = nx.max(x + 1);
            ny = ny.max(y + 1);
        }
        for (a, b) in &self.lines {
            for p in [a, b] {
                nx = nx.max(p.0 + 1);
                ny = ny.max(p.1 + 1);
            }
        }
        let mut xs = vec![1usize; nx];
        let ys = vec![1usize; ny];
        for (&(x, _), (_, _, label)) in &self.cells {
            xs[x] = xs[x].max(label.len());
        }
        let offsets = |spans: &[usize]| -> Vec<usize> {
            let mut o = vec![0];
            for s in spans {
                o.push(o.last().unwrap() + s);
            }
            o
        };
        let (xo, yo) = (offsets(&xs), offsets(&ys));
        let mut out: Vec<Vec<u8>> = vec![vec![b' '; *xo.last().unwrap()]; *yo.last().unwrap()];
        let align = |c0: usize, cn: usize, a: f32| -> usize {
            let cn = if a == 0.5 { cn - 1 } else { cn };
            c0 + (a * cn as f32).floor() as usize
        };
        for (p1, p2) in &self.lines {
            let (mut x1, mut x2) = (align(xo[p1.0], xs[p1.0], p1.2), align(xo[p2.0], xs[p2.0], p2.2));
            let (mut y1, mut y2) = (align(yo[p1.1], ys[p1.1], p1.3), align(yo[p2.1], ys[p2.1], p2.3));
            if x1 > x2 {
                std::mem::swap(&mut x1, &mut x2);
            }
            if y1 > y2 {
                std::mem::swap(&mut y1, &mut y2);
            }
            let bx = x1 != x2;
            while x1 < x2 {
                out[y1][x1] = b'-';
                x1 += 1;
            }
            let mut next = if bx { b'.' } else { b'|' };
            while y1 < y2 {
                out[y1][x1] = next;
                next = b'|';
                y1 += 1;
            }
        }
        for (&(cx, cy), (ax, ay, label)) in &self.cells {
            let x = xo[cx] + (ax * (xs[cx] - label.len()) as f32).floor() as usize;
            let y = yo[cy] + (ay * (ys[cy] - 1) as f32).floor() as usize;
            out[y][x..x + label.len()].copy_from_slice(label.as_bytes());
        }
        let mut lines: Vec<String> = out.into_iter().map(|l| String::from_utf8(l).unwrap().trim_end_matches(' ').to_string()).collect();
        while lines.last().is_some_and(|l| l.is_empty()) {
            lines.pop();
        }
        let start = lines.iter().position(|l| !l.is_empty()).unwrap_or(lines.len());
        let lines = &lines[start..];
        let indent = lines.iter().map(|l| l.len() - l.trim_start_matches(' ').len()).min().unwrap_or(0);
        lines.iter().map(|l| &l[indent..]).collect::<Vec<_>>().join("\n")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stims_example() {
        let c = ir::Circuit::parse("H 0\nCNOT 0 1\nTICK\nM 0 1\nDETECTOR rec[-1] rec[-2]").unwrap();
        let s = SliceSet::from_circuit_ticks(&c, 1, 1, &[CoordFilter::default()]).unwrap();
        assert_eq!(s.text(), "q0: -Z:D0-\n     |\nq1: -Z:D0-");
    }
}

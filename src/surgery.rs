//! Lattice surgery: two rotated surface-code patches merged across a seam to
//! measure Z⊗Z, then split again, as one circuit.
//!
//! WHY THIS EXISTS
//! ---------------
//! A memory holds one logical qubit still; a computer needs logical qubits to
//! interact. On a planar chip the standard way is lattice surgery (Horsman,
//! Fowler, Devitt and Van Meter, 2012). The stabilizers across the seam
//! between two patches are measured for some rounds, which merges them into
//! one patch; the product of the new checks along the seam is the joint parity
//! of the two logical qubits; then the patches are split. How many merged rounds a
//! measurement needs is the clock of every resource estimate. This module
//! writes the whole experiment as a circuit, so the error-model builder, the
//! samplers and the matchers see it exactly as they see a memory.
//!
//! Geometry is `RotatedSurfaceCode`'s: data at odd (x, y), checks at even
//! (x, y), X or Z by the parity of (x + y)/2, the same CNOT orders. Patch 1's
//! data columns are x = 1 … 2d − 1, the seam is x = 2d + 1, and patch 2 is
//! patch 1 moved 2d + 2 to the right. Each patch's logical Z runs down a
//! column, so the two face each other with their X-type boundaries.
//!
//! Detectors follow one rule. A check measured in two consecutive rounds is
//! compared across them, and if its support changed in between (the boundary X
//! checks reaching across the seam at the merge and pulling back at the
//! split), the qubits it gained were freshly prepared in its basis and the
//! qubits it lost were measured in its basis, so their records join the
//! comparison. A check measured for the first time is a detector alone only
//! if every qubit it touches was freshly prepared in its basis.

use std::collections::HashMap;

use crate::circuit::{Basis, Circuit, Instr};
use crate::surface_code::RotatedSurfaceCode;

/// The experiment.
#[derive(Clone, Copy, Debug)]
pub struct Surgery {
    pub d: usize,
    /// Rounds of separate memory before the merge, and after the split.
    pub pre: usize,
    pub merged: usize,
    pub post: usize,
    /// Z: both patches in |0⟩, the merge outcome and each patch's Z are the
    /// observables. X: both in |+⟩, X₁X₂ (which the Z⊗Z measurement keeps) is.
    pub basis: Basis,
    pub p: f64,
}

#[derive(Clone, Debug)]
struct Check {
    pos: (i32, i32),
    x_type: bool,
    anc: u32,
    /// The data qubit at each of the four CNOT steps, if there is one.
    at_step: [Option<u32>; 4],
}

impl Check {
    fn support(&self) -> Vec<u32> {
        let mut s: Vec<u32> = self.at_step.iter().flatten().copied().collect();
        s.sort_unstable();
        s
    }
}

/// Data columns x0..=x1 and rows y0..=y1 (odd) of one patch, or of patches merged.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct TileBox {
    x0: i32,
    x1: i32,
    y0: i32,
    y1: i32,
}

impl TileBox {
    fn union(a: TileBox, b: TileBox) -> TileBox {
        TileBox { x0: a.x0.min(b.x0), x1: a.x1.max(b.x1), y0: a.y0.min(b.y0), y1: a.y1.max(b.y1) }
    }
}

struct Layout {
    d: i32,
    positions: Vec<(i32, i32)>,
    index: HashMap<(i32, i32), u32>,
}

impl Layout {
    /// A grid of `cols` × `rows` tiles, 2d + 2 apart, one seam column or row
    /// of data between neighbours: data first, row by row, then every check
    /// position, row by row (for 2 × 1 tiles, the order of the two-patch
    /// experiment as it was first written).
    fn new(d: usize, cols: i32, rows: i32) -> Layout {
        let d = d as i32;
        let (w, h) = (cols * (2 * d + 2) - 2, rows * (2 * d + 2) - 2);
        let (mut positions, mut index) = (Vec::new(), HashMap::new());
        for y in (1..h).step_by(2) {
            for x in (1..w).step_by(2) {
                index.insert((x, y), positions.len() as u32);
                positions.push((x, y));
            }
        }
        for y in (0..=h).step_by(2) {
            for x in (0..=w).step_by(2) {
                index.insert((x, y), positions.len() as u32);
                positions.push((x, y));
            }
        }
        Layout { d, positions, index }
    }

    /// The data of tile (column i, row j).
    fn tile(&self, (i, j): (i32, i32)) -> TileBox {
        let (x0, y0) = (1 + i * (2 * self.d + 2), 1 + j * (2 * self.d + 2));
        TileBox { x0, x1: x0 + 2 * self.d - 2, y0, y1: y0 + 2 * self.d - 2 }
    }

    fn data(&self, b: TileBox) -> Vec<u32> {
        let mut v: Vec<u32> = (b.y0..=b.y1)
            .step_by(2)
            .flat_map(|y| (b.x0..=b.x1).step_by(2).map(move |x| (x, y)))
            .map(|p| self.index[&p])
            .collect();
        v.sort_unstable();
        v
    }

    /// The rotated code on box `b`: X-type boundaries left and right, Z-type
    /// top and bottom.
    fn code(&self, b: TileBox) -> Vec<Check> {
        let inside = |x: i32, y: i32| x >= b.x0 && x <= b.x1 && y >= b.y0 && y <= b.y1;
        let mut checks = Vec::new();
        for y in (b.y0 - 1..=b.y1 + 1).step_by(2) {
            for x in (b.x0 - 1..=b.x1 + 1).step_by(2) {
                let x_type = ((x + y) / 2).rem_euclid(2) == 1;
                // Z checks stay off the left and right edges, X checks off the top and bottom.
                let allowed = if x_type { y > b.y0 && y < b.y1 } else { x > b.x0 && x < b.x1 };
                if !allowed {
                    continue;
                }
                let order = if x_type { RotatedSurfaceCode::X_ORDER } else { RotatedSurfaceCode::Z_ORDER };
                let mut at_step = [None; 4];
                for (k, &(dx, dy)) in order.iter().enumerate() {
                    if inside(x + dx, y + dy) {
                        at_step[k] = Some(self.index[&(x + dx, y + dy)]);
                    }
                }
                if at_step.iter().flatten().count() >= 2 {
                    checks.push(Check { pos: (x, y), x_type, anc: self.index[&(x, y)], at_step });
                }
            }
        }
        checks
    }
}

/// What a detector needs to know about each check's last measurement.
struct Last {
    rec: usize,
    support: Vec<u32>,
}

struct Writer {
    c: Vec<Instr>,
    m: usize,
    p: f64,
    last: HashMap<(i32, i32), Last>,
    /// Data qubits freshly prepared this round, with their basis.
    fresh: HashMap<u32, Basis>,
    /// Data qubits measured since their checks last read them: record and basis.
    measured: HashMap<u32, (usize, Basis)>,
    round: usize,
}

impl Writer {
    fn lookback(&self, rec: usize) -> u32 {
        (self.m - rec) as u32
    }

    fn noise(&mut self, instr: Instr) {
        if self.p > 0.0 {
            self.c.push(instr);
        }
    }

    fn depol1(&mut self, qubits: Vec<u32>) {
        if !qubits.is_empty() {
            let p = self.p;
            self.noise(Instr::Depolarize1 { p, qubits });
        }
    }

    fn prepare(&mut self, basis: Basis, qubits: &[u32]) {
        if qubits.is_empty() {
            return;
        }
        self.c.push(Instr::Reset { basis, qubits: qubits.to_vec() });
        let p = self.p;
        self.noise(Instr::PauliError { pauli: if basis == Basis::Z { 1 } else { 2 }, p, qubits: qubits.to_vec() });
        for &q in qubits {
            self.fresh.insert(q, basis);
        }
    }

    /// Measure data qubits; returns their records in the order given.
    fn measure(&mut self, basis: Basis, qubits: &[u32]) -> Vec<usize> {
        let p = self.p;
        self.noise(Instr::PauliError { pauli: if basis == Basis::Z { 1 } else { 2 }, p, qubits: qubits.to_vec() });
        self.c.push(Instr::Measure { basis, reset: false, flip: 0.0, qubits: qubits.to_vec() });
        let recs: Vec<usize> = (0..qubits.len()).map(|i| self.m + i).collect();
        self.m += qubits.len();
        for (&q, &r) in qubits.iter().zip(&recs) {
            self.measured.insert(q, (r, basis));
        }
        recs
    }

    /// One round of syndrome extraction on `checks`, the SD6 way. `data` is
    /// every data qubit in play; those not freshly prepared this round idle
    /// through the reset. Returns each check's record.
    fn round(&mut self, checks: &[Check], data: &[u32]) -> Vec<usize> {
        self.round += 1;
        let anc: Vec<u32> = checks.iter().map(|c| c.anc).collect();
        let had: Vec<u32> = checks.iter().filter(|c| c.x_type).map(|c| c.anc).collect();
        let active: Vec<u32> = data.iter().copied().chain(anc.iter().copied()).collect();
        let idle = |busy: &[u32]| -> Vec<u32> { active.iter().copied().filter(|q| !busy.contains(q)).collect() };

        self.c.push(Instr::Tick);
        self.c.push(Instr::Reset { basis: Basis::Z, qubits: anc.clone() });
        let p = self.p;
        self.noise(Instr::PauliError { pauli: 1, p, qubits: anc.clone() });
        let resting: Vec<u32> = data.iter().copied().filter(|q| !self.fresh.contains_key(q)).collect();
        self.depol1(resting);

        for pass in 0..2 {
            self.c.push(Instr::Tick);
            if !had.is_empty() {
                self.c.push(Instr::H(had.clone()));
                self.depol1(had.clone());
            }
            let rest = idle(&had);
            self.depol1(rest);
            if pass == 1 {
                break;
            }
            for step in 0..4 {
                self.c.push(Instr::Tick);
                let mut pairs = Vec::new();
                for ch in checks {
                    if let Some(q) = ch.at_step[step] {
                        pairs.push(if ch.x_type { (ch.anc, q) } else { (q, ch.anc) });
                    }
                }
                if !pairs.is_empty() {
                    self.c.push(Instr::Cx(pairs.clone()));
                    self.noise(Instr::Depolarize2 { p, pairs: pairs.clone() });
                }
                let busy: Vec<u32> = pairs.iter().flat_map(|&(a, b)| [a, b]).collect();
                let rest = idle(&busy);
                self.depol1(rest);
            }
        }

        self.c.push(Instr::Tick);
        self.noise(Instr::PauliError { pauli: 1, p, qubits: anc.clone() });
        self.c.push(Instr::Measure { basis: Basis::Z, reset: false, flip: 0.0, qubits: anc.clone() });
        let recs: Vec<usize> = (0..anc.len()).map(|i| self.m + i).collect();
        self.m += anc.len();
        self.depol1(data.to_vec());

        for (ch, &rec) in checks.iter().zip(&recs) {
            let basis = if ch.x_type { Basis::X } else { Basis::Z };
            let support = ch.support();
            let coords = vec![ch.pos.0 as f64, ch.pos.1 as f64, self.round as f64];
            let mut targets = vec![rec];
            let deterministic = match self.last.get(&ch.pos) {
                Some(prev) => {
                    targets.push(prev.rec);
                    let gained = support.iter().filter(|q| !prev.support.contains(q));
                    let lost = prev.support.iter().filter(|q| !support.contains(q));
                    let gained_ok = gained.clone().all(|q| self.fresh.get(q) == Some(&basis));
                    let mut lost_ok = true;
                    for q in lost {
                        match self.measured.get(q) {
                            Some(&(r, b)) if b == basis => targets.push(r),
                            _ => lost_ok = false,
                        }
                    }
                    gained_ok && lost_ok
                }
                None => support.iter().all(|q| self.fresh.get(q) == Some(&basis)),
            };
            if deterministic {
                let recs = targets.iter().map(|&r| self.lookback(r)).collect();
                self.c.push(Instr::Detector { coords, recs });
            }
            self.last.insert(ch.pos, Last { rec, support });
        }
        self.fresh.clear();
        recs
    }
}

impl Surgery {
    pub fn circuit(&self) -> Result<Circuit, String> {
        if self.d < 3 || self.d % 2 == 0 {
            return Err(format!("lattice surgery needs an odd distance of at least 3, not {}", self.d));
        }
        if self.merged == 0 {
            return Err("the merge needs at least one round".into());
        }
        let layout = Layout::new(self.d, 2, 1);
        let d = self.d as i32;
        let (b1, b2) = (layout.tile((0, 0)), layout.tile((1, 0)));
        let (p1, seam, p2) = ((b1.x0, b1.x1), 2 * d + 1, (b2.x0, b2.x1));
        let patches: Vec<Check> = layout.code(b1).into_iter().chain(layout.code(b2)).collect();
        let merged = layout.code(TileBox::union(b1, b2));
        let patch_data: Vec<u32> = layout.data(b1).into_iter().chain(layout.data(b2)).collect();
        let seam_data = layout.data(TileBox { x0: seam, x1: seam, ..b1 });
        let all_data: Vec<u32> = layout.data(TileBox::union(b1, b2));

        let mut w = Writer { c: Vec::new(), m: 0, p: self.p, last: HashMap::new(), fresh: HashMap::new(), measured: HashMap::new(), round: 0 };
        for (q, &(x, y)) in layout.positions.iter().enumerate() {
            w.c.push(Instr::QubitCoords { coords: vec![x as f64, y as f64], qubits: vec![q as u32] });
        }

        // Prepare, and hold each patch separately.
        w.prepare(self.basis, &patch_data);
        for _ in 0..self.pre {
            w.round(&patches, &patch_data);
        }
        // Merge: the seam in |+⟩, the merged patch measured.
        w.c.push(Instr::Tick);
        w.prepare(Basis::X, &seam_data);
        let new_z: Vec<usize> = merged.iter().enumerate().filter(|(_, c)| !c.x_type && !patches.iter().any(|q| q.pos == c.pos)).map(|(i, _)| i).collect();
        let mut first_merge: Option<Vec<usize>> = None;
        for _ in 0..self.merged {
            let recs = w.round(&merged, &all_data);
            first_merge.get_or_insert(recs);
        }
        // Split: the seam read out in X.
        w.c.push(Instr::Tick);
        let seam_recs = w.measure(Basis::X, &seam_data);
        for _ in 0..self.post {
            w.round(&patches, &patch_data);
        }
        // Finish: the patches read out in the experiment's basis.
        w.c.push(Instr::Tick);
        let final_recs = w.measure(self.basis, &patch_data);
        let rec_of: HashMap<u32, usize> = patch_data.iter().copied().zip(final_recs).collect();
        let final_basis_x = self.basis == Basis::X;
        for ch in patches.iter().filter(|c| c.x_type == final_basis_x) {
            let mut recs: Vec<u32> = ch.support().iter().map(|q| w.lookback(rec_of[q])).collect();
            recs.push(w.lookback(w.last[&ch.pos].rec));
            let coords = vec![ch.pos.0 as f64, ch.pos.1 as f64, (w.round + 1) as f64];
            w.c.push(Instr::Detector { coords, recs });
        }
        let column = |w: &Writer, x: i32| -> Vec<u32> {
            (1..2 * d).step_by(2).map(|y| w.lookback(rec_of[&layout.index[&(x, y)]])).collect()
        };
        match self.basis {
            Basis::Z => {
                let first = first_merge.expect("at least one merged round");
                let outcome: Vec<u32> = new_z.iter().map(|&i| w.lookback(first[i])).collect();
                let (z1, z2) = (column(&w, p1.0), column(&w, p2.0));
                w.c.push(Instr::Observable { index: 0, recs: outcome });
                w.c.push(Instr::Observable { index: 1, recs: z1 });
                w.c.push(Instr::Observable { index: 2, recs: z2 });
            }
            Basis::X => {
                // X₁X₂ along the bottom row, with the seam qubit of that row read at the split.
                let mut recs: Vec<u32> = (p1.0..=p1.1).step_by(2).chain((p2.0..=p2.1).step_by(2)).map(|x| w.lookback(rec_of[&layout.index[&(x, 1)]])).collect();
                let seam_q = layout.index[&(seam, 1)];
                let seam_rec = seam_data.iter().position(|&q| q == seam_q).map(|i| seam_recs[i]).expect("seam qubit");
                recs.push(w.lookback(seam_rec));
                w.c.push(Instr::Observable { index: 0, recs });
            }
        }
        Ok(Circuit { instrs: w.c })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dem::Dem;
    use crate::dem_decoder::DemDecoder;
    use crate::m2d::M2d;

    fn surgery(d: usize, merged: usize, basis: Basis, p: f64) -> Surgery {
        Surgery { d, pre: d, merged, post: d, basis, p }
    }

    /// FNV-1a, 64 bits: a fingerprint that is the same in every Rust version.
    fn fnv1a64(bytes: &[u8]) -> String {
        let h = bytes.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &b| (h ^ u64::from(b)).wrapping_mul(0x100_0000_01b3));
        format!("{h:016x}")
    }

    /// The grid layout and the program compiler change nothing: every
    /// experiment section 14 measured compiles to the same circuit, byte for
    /// byte, as before either existed (fingerprints taken before, in
    /// data/surgery/golden.json).
    #[test]
    fn the_z_z_experiment_is_unchanged() {
        let golden = include_str!("../data/surgery/golden.json");
        let mut n = 0;
        for d in [3usize, 5, 7] {
            let mut ts = vec![1usize, 2, 3, d, 2 * d];
            ts.sort_unstable();
            ts.dedup();
            for merged in ts {
                for (basis, name) in [(Basis::Z, "z"), (Basis::X, "x")] {
                    let text = Surgery { d, pre: d, merged, post: d, basis, p: 0.002 }.circuit().unwrap().to_stim();
                    let key = format!("\"d{d}/T{merged}/{name}\": \"{}\"", fnv1a64(text.as_bytes()));
                    assert!(golden.contains(&key), "d = {d}, T = {merged}, {name}: the circuit changed");
                    n += 1;
                }
            }
        }
        assert_eq!(n, 28);
    }

    /// Without noise every detector and observable is deterministic: the
    /// engine's references refuse any that is not.
    #[test]
    fn every_detector_and_observable_is_deterministic() {
        for d in [3usize, 5] {
            for basis in [Basis::Z, Basis::X] {
                for merged in [1usize, 2, d] {
                    let c = surgery(d, merged, basis, 0.0).circuit().unwrap();
                    let m = M2d::new(&c).unwrap_or_else(|e| panic!("d = {d} {basis:?} T = {merged}: {e}"));
                    assert_eq!(m.num_observables, if basis == Basis::Z { 3 } else { 1 });
                }
            }
        }
    }

    /// The patches are rotated codes: 2(d² − 1) checks between them, and the
    /// merged patch has d(2d + 1) − 1.
    #[test]
    fn the_codes_have_the_right_number_of_checks() {
        for d in [3usize, 5, 7] {
            let layout = Layout::new(d, 2, 2);
            let (a, b, c) = (layout.tile((0, 0)), layout.tile((1, 0)), layout.tile((0, 1)));
            assert_eq!(layout.code(a).len(), d * d - 1);
            assert_eq!(layout.code(b).len(), d * d - 1);
            // Merged side by side, and one above the other.
            assert_eq!(layout.code(TileBox::union(a, b)).len(), d * (2 * d + 1) - 1);
            assert_eq!(layout.code(TileBox::union(a, c)).len(), d * (2 * d + 1) - 1);
        }
    }

    /// With one merged round, a single measurement error on a new seam check
    /// flips the merge outcome with nothing after it to notice: the error-model
    /// builder refuses the circuit, naming an undetectable flip of L0 alone.
    /// With three, every single fault of the model is corrected.
    #[test]
    fn single_faults_need_more_than_one_merged_round() {
        let one = surgery(3, 1, Basis::Z, 0.001).circuit().unwrap();
        let err = Dem::from_circuit(&one).unwrap_err();
        assert!(err.contains("undetectable logical error") && err.contains("flips observables 0b1 "), "{err}");

        for basis in [Basis::Z, Basis::X] {
            let c = surgery(3, 3, basis, 0.001).circuit().unwrap();
            let dem = Dem::from_circuit(&c).unwrap();
            let dec = DemDecoder::new(&dem).unwrap();
            let failed = dem
                .mechanisms
                .iter()
                .filter(|m| dec.decode(&m.detectors).map(|p| p.observables != m.observables).unwrap_or(true))
                .count();
            assert_eq!(failed, 0, "{basis:?}, T = 3: {failed} of {} single faults uncorrected", dem.mechanisms.len());
        }
    }
}

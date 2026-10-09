//! Stim's `SparseUnsignedRevFrameTracker`: runs a circuit backwards, tracking for each qubit
//! which detectors and observables are sensitive to an X or Z error there. Detecting regions,
//! feedback inlining and anticommutation checks are built on it. Ported from
//! `sparse_rev_frame_tracker.cc`, loop period detection included.

use std::collections::{BTreeMap, BTreeSet};

use super::ir::{self, GateTarget, Instruction, Item};
use super::tableau_sim::{decompose_mpp, decompose_spp, disjoint_pair_segments};

/// A detector (its index) or observable (the index with the top bit set), as Stim's
/// `DemTarget` orders them: detectors first, by index, then observables.
pub type DemTarget = u64;
pub const OBSERVABLE_BIT: u64 = 1 << 63;

pub fn is_observable(t: DemTarget) -> bool {
    t & OBSERVABLE_BIT != 0
}

/// A sorted set under symmetric difference (Stim's `SparseXorVec`).
#[derive(Clone, Default, Debug, PartialEq, Eq)]
pub struct XorSet(pub BTreeSet<u64>);

impl XorSet {
    pub fn xor_item(&mut self, v: u64) {
        if !self.0.remove(&v) {
            self.0.insert(v);
        }
    }
    pub fn xor(&mut self, other: &XorSet) {
        for &v in &other.0 {
            self.xor_item(v);
        }
    }
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
    pub fn clear(&mut self) {
        self.0.clear();
    }
    pub fn iter(&self) -> impl Iterator<Item = &u64> {
        self.0.iter()
    }
}

#[derive(Clone, Debug)]
pub struct RevTracker {
    pub xs: Vec<XorSet>,
    pub zs: Vec<XorSet>,
    pub rec_bits: BTreeMap<u64, XorSet>,
    pub num_measurements_in_past: u64,
    pub num_detectors_in_past: u64,
    pub fail_on_anticommute: bool,
    /// (detector or observable, the Pauli target it anticommuted with).
    pub anticommutations: BTreeSet<(DemTarget, u32)>,
}

fn rec_offset(t: GateTarget) -> i64 {
    -(t.value() as i64)
}

impl RevTracker {
    pub fn new(num_qubits: usize, num_measurements: u64, num_detectors: u64, fail_on_anticommute: bool) -> RevTracker {
        RevTracker { xs: vec![XorSet::default(); num_qubits], zs: vec![XorSet::default(); num_qubits], rec_bits: BTreeMap::new(), num_measurements_in_past: num_measurements, num_detectors_in_past: num_detectors, fail_on_anticommute, anticommutations: BTreeSet::new() }
    }

    fn fail(&self, inst: &Instruction) -> Result<(), String> {
        let mut s = format!("While running backwards through the circuit, during reverse-execution of the instruction\n    {inst}\nthe following detecting region vs dissipation anticommutations occurred\n");
        for (d, g) in &self.anticommutations {
            s.push_str(&format!("    {} vs {}\n", dem_target_str(*d), GateTarget(*g)));
        }
        s.push_str("Therefore invalid detectors/observables are present in the circuit.\n");
        Err(s)
    }

    fn handle_gauge(&mut self, set: &XorSet, inst: &Instruction, location: GateTarget) -> Result<(), String> {
        if set.is_empty() {
            return Ok(());
        }
        for &d in set.iter() {
            self.anticommutations.insert((d, location.0));
        }
        if self.fail_on_anticommute {
            return self.fail(inst);
        }
        Ok(())
    }

    fn handle_xor_gauge(&mut self, a: &XorSet, b: &XorSet, inst: &Instruction, location: GateTarget) -> Result<(), String> {
        if a == b {
            return Ok(());
        }
        let mut dif = a.clone();
        dif.xor(b);
        for &d in dif.iter() {
            self.anticommutations.insert((d, location.0));
        }
        if self.fail_on_anticommute {
            return self.fail(inst);
        }
        Ok(())
    }

    fn gauges(&mut self, inst: &Instruction, basis: u8) -> Result<(), String> {
        for t in inst.targets.iter().rev() {
            let q = t.value() as usize;
            match basis {
                1 => {
                    let s = self.xs[q].clone();
                    self.handle_gauge(&s, inst, GateTarget::pauli(q as u32, 1, false))?
                }
                2 => {
                    let (a, b) = (self.xs[q].clone(), self.zs[q].clone());
                    self.handle_xor_gauge(&a, &b, inst, GateTarget::pauli(q as u32, 2, false))?
                }
                _ => {
                    let s = self.zs[q].clone();
                    self.handle_gauge(&s, inst, GateTarget::pauli(q as u32, 3, false))?
                }
            }
        }
        Ok(())
    }

    /// The anticommutation check of the implicit resets at the start of the circuit.
    pub fn undo_implicit_rzs_at_start_of_circuit(&mut self) -> Result<(), String> {
        for q in 0..self.xs.len() {
            for &d in self.xs[q].0.iter() {
                self.anticommutations.insert((d, q as u32));
            }
        }
        if !self.anticommutations.is_empty() && self.fail_on_anticommute {
            let mut s = String::from("While running backwards through the circuit,\nduring reverse-execution of the implicit resets at the beginning of the circuit,\nthe following detecting region vs dissipation anticommutations occurred\n");
            for (d, g) in &self.anticommutations {
                s.push_str(&format!("    {} vs {}\n", dem_target_str(*d), GateTarget(*g)));
            }
            s.push_str("Therefore invalid detectors/observables are present in the circuit.\n");
            return Err(s);
        }
        Ok(())
    }

    fn undo_classical_pauli(&mut self, control: GateTarget, target: GateTarget) {
        if control.is_sweep() {
            return;
        }
        let m = (self.num_measurements_in_past as i64 + rec_offset(control)) as u64;
        let q = target.value() as usize;
        let mut dst = self.rec_bits.remove(&m).unwrap_or_default();
        if target.0 & ir::TARGET_PAULI_X_BIT != 0 {
            dst.xor(&self.zs[q]);
        }
        if target.0 & ir::TARGET_PAULI_Z_BIT != 0 {
            dst.xor(&self.xs[q]);
        }
        if !dst.is_empty() {
            self.rec_bits.insert(m, dst);
        }
    }

    fn xor_into(&mut self, dst_x: bool, dst: usize, src_x: bool, src: usize) {
        let s = if src_x { self.xs[src].clone() } else { self.zs[src].clone() };
        if dst_x {
            self.xs[dst].xor(&s)
        } else {
            self.zs[dst].xor(&s)
        }
    }

    fn zcx_single(&mut self, c: GateTarget, t: GateTarget) -> Result<(), String> {
        let (cd, td) = (c.0 & !ir::TARGET_INVERTED_BIT, t.0 & !ir::TARGET_INVERTED_BIT);
        if (cd | td) & (ir::TARGET_RECORD_BIT | ir::TARGET_SWEEP_BIT) == 0 {
            self.xor_into(false, cd as usize, false, td as usize);
            self.xor_into(true, td as usize, true, cd as usize);
        } else if !t.is_qubit() {
            return Err(format!("CX gate had '{t}' as its target, but its target must be a qubit."));
        } else {
            self.undo_classical_pauli(c, GateTarget::pauli(td, 1, false));
        }
        Ok(())
    }

    fn zcy_single(&mut self, c: GateTarget, t: GateTarget) -> Result<(), String> {
        let (cd, td) = (c.0 & !ir::TARGET_INVERTED_BIT, t.0 & !ir::TARGET_INVERTED_BIT);
        if (cd | td) & (ir::TARGET_RECORD_BIT | ir::TARGET_SWEEP_BIT) == 0 {
            let (c, t) = (cd as usize, td as usize);
            self.xor_into(false, c, false, t);
            self.xor_into(false, c, true, t);
            self.xor_into(true, t, true, c);
            self.xor_into(false, t, true, c);
        } else if !t.is_qubit() {
            return Err(format!("CY gate had '{t}' as its target, but its target must be a qubit."));
        } else {
            self.undo_classical_pauli(c, GateTarget::pauli(td, 2, false));
        }
        Ok(())
    }

    fn zcz_single(&mut self, c: GateTarget, t: GateTarget) {
        let (cd, td) = (c.0 & !ir::TARGET_INVERTED_BIT, t.0 & !ir::TARGET_INVERTED_BIT);
        let bits = ir::TARGET_RECORD_BIT | ir::TARGET_SWEEP_BIT;
        if (cd | td) & bits == 0 {
            self.xor_into(false, cd as usize, true, td as usize);
            self.xor_into(false, td as usize, true, cd as usize);
        } else if td & bits == 0 {
            self.undo_classical_pauli(c, GateTarget::pauli(td, 3, false));
        } else if cd & bits == 0 {
            self.undo_classical_pauli(t, GateTarget::pauli(cd, 3, false));
        }
    }

    fn take_rec(&mut self) -> Option<XorSet> {
        self.num_measurements_in_past -= 1;
        self.rec_bits.remove(&self.num_measurements_in_past)
    }

    fn undo_measure(&mut self, inst: &Instruction, basis: u8, reset: bool) -> Result<(), String> {
        self.gauges(inst, match basis {
            1 => 3,
            2 => 2,
            _ => 1,
        })?;
        for t in inst.targets.iter().rev() {
            let q = t.value() as usize;
            let r = self.take_rec();
            if reset {
                self.xs[q].clear();
                self.zs[q].clear();
            }
            if let Some(r) = r {
                if basis != 3 {
                    self.xs[q].xor(&r);
                }
                if basis != 1 {
                    self.zs[q].xor(&r);
                }
            }
        }
        Ok(())
    }

    fn undo_reset(&mut self, inst: &Instruction, basis: u8) -> Result<(), String> {
        self.gauges(inst, match basis {
            1 => 3,
            2 => 2,
            _ => 1,
        })?;
        for t in inst.targets.iter().rev() {
            let q = t.value() as usize;
            self.xs[q].clear();
            self.zs[q].clear();
        }
        Ok(())
    }

    fn pairs_rev(inst: &Instruction) -> Vec<(GateTarget, GateTarget)> {
        inst.targets.chunks(2).rev().map(|p| (p[0], p[1])).collect()
    }

    pub fn undo_gate(&mut self, inst: &Instruction) -> Result<(), String> {
        let name = inst.gate.name;
        match name {
            "DETECTOR" => {
                self.num_detectors_in_past -= 1;
                let det = self.num_detectors_in_past;
                for t in &inst.targets {
                    let index = rec_offset(*t) + self.num_measurements_in_past as i64;
                    if index < 0 {
                        return Err("Referred to a measurement result before the beginning of time.".into());
                    }
                    self.rec_bits.entry(index as u64).or_default().xor_item(det);
                }
            }
            "OBSERVABLE_INCLUDE" => {
                let obs = inst.args[0] as u32 as u64 | OBSERVABLE_BIT;
                for t in &inst.targets {
                    if t.is_record() {
                        let index = rec_offset(*t) + self.num_measurements_in_past as i64;
                        if index < 0 {
                            return Err("Referred to a measurement result before the beginning of time.".into());
                        }
                        self.rec_bits.entry(index as u64).or_default().xor_item(obs);
                    } else if t.is_pauli() {
                        if t.0 & ir::TARGET_PAULI_X_BIT != 0 {
                            self.xs[t.value() as usize].xor_item(obs);
                        }
                        if t.0 & ir::TARGET_PAULI_Z_BIT != 0 {
                            self.zs[t.value() as usize].xor_item(obs);
                        }
                    } else {
                        return Err(format!("Unexpected target for OBSERVABLE_INCLUDE: {t}"));
                    }
                }
            }
            "MX" => self.undo_measure(inst, 1, false)?,
            "MY" => self.undo_measure(inst, 2, false)?,
            "M" => self.undo_measure(inst, 3, false)?,
            "MRX" => self.undo_measure(inst, 1, true)?,
            "MRY" => self.undo_measure(inst, 2, true)?,
            "MR" => self.undo_measure(inst, 3, true)?,
            "RX" => self.undo_reset(inst, 1)?,
            "RY" => self.undo_reset(inst, 2)?,
            "R" => self.undo_reset(inst, 3)?,
            "MPP" => {
                let mut rev = inst.clone();
                rev.targets.reverse();
                for sub in decompose_mpp(&rev, self.xs.len())? {
                    if sub.gate.name == "M" {
                        let mut m = sub.clone();
                        m.targets.reverse();
                        self.undo_measure(&m, 3, false)?;
                    } else {
                        self.undo_gate(&sub)?;
                    }
                }
            }
            "SPP" | "SPP_DAG" => {
                let mut rev = inst.clone();
                rev.targets.reverse();
                for sub in decompose_spp(&rev)? {
                    self.undo_gate(&sub)?;
                }
            }
            "MXX" | "MYY" | "MZZ" => {
                let mut rev = inst.clone();
                rev.targets.reverse();
                let (pre, basis) = match name {
                    "MXX" => ("CX", 1u8),
                    "MYY" => ("CY", 2),
                    _ => ("XCZ", 3),
                };
                for seg in disjoint_pair_segments(&rev.targets) {
                    let conj = Instruction::new(pre, vec![], seg.clone(), "")?;
                    self.undo_gate(&conj)?;
                    for pair in seg.chunks(2) {
                        let m = Instruction::new(["", "MX", "MY", "M"][basis as usize], inst.args.clone(), vec![pair[0]], "")?;
                        self.undo_measure(&m, basis, false)?;
                    }
                    self.undo_gate(&conj)?;
                }
            }
            "XCX" => {
                for (a, b) in Self::pairs_rev(inst) {
                    let (a, b) = (a.value() as usize, b.value() as usize);
                    self.xor_into(true, a, false, b);
                    self.xor_into(true, b, false, a);
                }
            }
            "XCY" | "YCX" => {
                for (a, b) in Self::pairs_rev(inst) {
                    let (tx, ty) = if name == "XCY" { (a.value() as usize, b.value() as usize) } else { (b.value() as usize, a.value() as usize) };
                    self.xor_into(true, tx, true, ty);
                    self.xor_into(true, tx, false, ty);
                    self.xor_into(true, ty, false, tx);
                    self.xor_into(false, ty, false, tx);
                }
            }
            "CY" => {
                for (c, t) in Self::pairs_rev(inst) {
                    self.zcy_single(c, t)?;
                }
            }
            "YCZ" => {
                for (t, c) in Self::pairs_rev(inst) {
                    self.zcy_single(c, t)?;
                }
            }
            "YCY" => {
                for (a, b) in Self::pairs_rev(inst) {
                    let (a, b) = (a.value() as usize, b.value() as usize);
                    self.xor_into(false, a, true, b);
                    self.xor_into(false, a, false, b);
                    self.xor_into(true, a, true, b);
                    self.xor_into(true, a, false, b);
                    self.xor_into(false, b, true, a);
                    self.xor_into(false, b, false, a);
                    self.xor_into(true, b, true, a);
                    self.xor_into(true, b, false, a);
                }
            }
            "CX" => {
                for (c, t) in Self::pairs_rev(inst) {
                    self.zcx_single(c, t)?;
                }
            }
            "XCZ" => {
                for (t, c) in Self::pairs_rev(inst) {
                    self.zcx_single(c, t)?;
                }
            }
            "CZ" => {
                for (a, b) in Self::pairs_rev(inst) {
                    self.zcz_single(a, b);
                }
            }
            "C_XYZ" | "C_NXYZ" | "C_XNYZ" | "C_XYNZ" => {
                for t in inst.targets.iter().rev() {
                    let q = t.value() as usize;
                    self.xor_into(false, q, true, q);
                    self.xor_into(true, q, false, q);
                }
            }
            "C_ZYX" | "C_NZYX" | "C_ZNYX" | "C_ZYNX" => {
                for t in inst.targets.iter().rev() {
                    let q = t.value() as usize;
                    self.xor_into(true, q, false, q);
                    self.xor_into(false, q, true, q);
                }
            }
            "SWAP" => {
                for (a, b) in Self::pairs_rev(inst) {
                    let (a, b) = (a.value() as usize, b.value() as usize);
                    self.xs.swap(a, b);
                    self.zs.swap(a, b);
                }
            }
            "CXSWAP" => {
                for (a, b) in Self::pairs_rev(inst) {
                    let (a, b) = (a.value() as usize, b.value() as usize);
                    self.xor_into(false, a, false, b);
                    self.xor_into(false, b, false, a);
                    self.xor_into(true, b, true, a);
                    self.xor_into(true, a, true, b);
                }
            }
            "CZSWAP" => {
                for (a, b) in Self::pairs_rev(inst) {
                    let (a, b) = (a.value() as usize, b.value() as usize);
                    self.xor_into(false, a, true, b);
                    self.xor_into(false, b, true, a);
                    self.xs.swap(a, b);
                    self.zs.swap(a, b);
                }
            }
            "SWAPCX" => {
                for (a, b) in Self::pairs_rev(inst) {
                    let (a, b) = (a.value() as usize, b.value() as usize);
                    self.xor_into(false, b, false, a);
                    self.xor_into(false, a, false, b);
                    self.xor_into(true, a, true, b);
                    self.xor_into(true, b, true, a);
                }
            }
            "SQRT_XX" | "SQRT_XX_DAG" => {
                for (a, b) in Self::pairs_rev(inst) {
                    let (a, b) = (a.value() as usize, b.value() as usize);
                    self.xor_into(true, a, false, a);
                    self.xor_into(true, a, false, b);
                    self.xor_into(true, b, false, a);
                    self.xor_into(true, b, false, b);
                }
            }
            "SQRT_YY" | "SQRT_YY_DAG" => {
                for (a, b) in Self::pairs_rev(inst) {
                    let (a, b) = (a.value() as usize, b.value() as usize);
                    self.xor_into(false, a, true, a);
                    self.xor_into(false, b, true, b);
                    self.xor_into(true, a, false, a);
                    self.xor_into(true, a, false, b);
                    self.xor_into(true, b, false, a);
                    self.xor_into(true, b, false, b);
                    self.xor_into(false, a, true, a);
                    self.xor_into(false, b, true, b);
                }
            }
            "SQRT_ZZ" | "SQRT_ZZ_DAG" => {
                for (a, b) in Self::pairs_rev(inst) {
                    let (a, b) = (a.value() as usize, b.value() as usize);
                    self.xor_into(false, a, true, a);
                    self.xor_into(false, a, true, b);
                    self.xor_into(false, b, true, a);
                    self.xor_into(false, b, true, b);
                }
            }
            "SQRT_X" | "SQRT_X_DAG" | "H_YZ" | "H_NYZ" => {
                for t in inst.targets.iter().rev() {
                    let q = t.value() as usize;
                    self.xor_into(true, q, false, q);
                }
            }
            "SQRT_Y" | "SQRT_Y_DAG" | "H" | "H_NXZ" => {
                for t in inst.targets.iter().rev() {
                    let q = t.value() as usize;
                    let x = std::mem::take(&mut self.xs[q]);
                    let z = std::mem::replace(&mut self.zs[q], x);
                    self.xs[q] = z;
                }
            }
            "S" | "S_DAG" | "H_XY" | "H_NXY" => {
                for t in inst.targets.iter().rev() {
                    let q = t.value() as usize;
                    self.xor_into(false, q, true, q);
                }
            }
            "ISWAP" | "ISWAP_DAG" => {
                for (a, b) in Self::pairs_rev(inst) {
                    let (a, b) = (a.value() as usize, b.value() as usize);
                    self.xor_into(false, a, true, a);
                    self.xor_into(false, a, true, b);
                    self.xor_into(false, b, true, a);
                    self.xor_into(false, b, true, b);
                    self.xs.swap(a, b);
                    self.zs.swap(a, b);
                }
            }
            "TICK" | "QUBIT_COORDS" | "SHIFT_COORDS" | "DEPOLARIZE1" | "DEPOLARIZE2" | "X_ERROR" | "Y_ERROR" | "Z_ERROR" | "PAULI_CHANNEL_1" | "PAULI_CHANNEL_2" | "E" | "ELSE_CORRELATED_ERROR" | "X" | "Y" | "Z" | "I" | "II" | "I_ERROR" | "II_ERROR" => {}
            "MPAD" | "HERALDED_ERASE" | "HERALDED_PAULI_CHANNEL_1" => {
                for _ in 0..inst.targets.len() {
                    self.take_rec();
                }
            }
            _ => return Err(format!("Not implemented by SparseUnsignedRevFrameTracker::undo_gate: {name}")),
        }
        Ok(())
    }

    pub fn undo_circuit(&mut self, c: &ir::Circuit) -> Result<(), String> {
        for it in c.items.iter().rev() {
            match it {
                Item::Op(op) => self.undo_gate(op)?,
                Item::Repeat { count, body, .. } => self.undo_loop(body, *count)?,
            }
        }
        Ok(())
    }

    fn is_shifted_copy(&self, other: &RevTracker) -> bool {
        let mo = other.num_measurements_in_past as i64 - self.num_measurements_in_past as i64;
        let dof = other.num_detectors_in_past as i64 - self.num_detectors_in_past as i64;
        let shift = |s: &XorSet| -> Vec<u64> { s.0.iter().map(|&d| if is_observable(d) { d } else { (d as i64 + dof) as u64 }).collect() };
        if self.rec_bits.len() != other.rec_bits.len() {
            return false;
        }
        for (k, v) in &self.rec_bits {
            match other.rec_bits.get(&((*k as i64 + mo) as u64)) {
                Some(w) if shift(v) == w.0.iter().copied().collect::<Vec<_>>() => {}
                _ => return false,
            }
        }
        let same = |a: &[XorSet], b: &[XorSet]| a.len() == b.len() && a.iter().zip(b).all(|(x, y)| shift(x) == y.0.iter().copied().collect::<Vec<_>>());
        same(&self.xs, &other.xs) && same(&self.zs, &other.zs)
    }

    fn shift(&mut self, mo: i64, dof: i64) {
        self.num_measurements_in_past = (self.num_measurements_in_past as i64 + mo) as u64;
        self.num_detectors_in_past = (self.num_detectors_in_past as i64 + dof) as u64;
        let sh = |s: &XorSet| XorSet(s.0.iter().map(|&d| if is_observable(d) { d } else { (d as i64 + dof) as u64 }).collect());
        self.rec_bits = self.rec_bits.iter().map(|(k, v)| (((*k as i64) + mo) as u64, sh(v))).collect();
        for x in self.xs.iter_mut() {
            *x = sh(x);
        }
        for z in self.zs.iter_mut() {
            *z = sh(z);
        }
    }

    pub fn undo_loop(&mut self, body: &ir::Circuit, iterations: u64) -> Result<(), String> {
        if iterations < 5 {
            for _ in 0..iterations {
                self.undo_circuit(body)?;
            }
            return Ok(());
        }
        let mut tortoise = self.clone();
        let mut hare_steps = 0u64;
        let mut tortoise_steps = 0u64;
        loop {
            self.undo_circuit(body)?;
            hare_steps += 1;
            if self.is_shifted_copy(&tortoise) {
                break;
            }
            if hare_steps > iterations - hare_steps {
                for _ in 0..iterations - hare_steps {
                    self.undo_circuit(body)?;
                }
                return Ok(());
            }
            if hare_steps % 2 == 0 {
                tortoise.undo_circuit(body)?;
                tortoise_steps += 1;
                if self.is_shifted_copy(&tortoise) {
                    break;
                }
            }
        }
        let period = hare_steps - tortoise_steps;
        let skipped = (iterations - hare_steps) / period;
        let dets = tortoise.num_detectors_in_past - self.num_detectors_in_past;
        let meas = tortoise.num_measurements_in_past - self.num_measurements_in_past;
        self.shift(-((meas * skipped) as i64), -((dets * skipped) as i64));
        hare_steps += skipped * period;
        for _ in 0..iterations - hare_steps {
            self.undo_circuit(body)?;
        }
        Ok(())
    }
}

/// Stim's text of a DEM target: `D5` or `L2`.
pub fn dem_target_str(d: DemTarget) -> String {
    if is_observable(d) {
        format!("L{}", d & !OBSERVABLE_BIT)
    } else {
        format!("D{d}")
    }
}

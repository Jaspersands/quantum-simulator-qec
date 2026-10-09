//! Circuit transformations with Stim's exact output: `decomposed` (Stim's `simplified_circuit`),
//! `flattened`, `without_noise`, `without_tags`, `inverse`, `with_inlined_feedback`, and the
//! queries `get_final_qubit_coordinates`, `get_detector_coordinates`,
//! `count_determined_measurements` and `reference_detector_and_observable_signs`.

use std::collections::{BTreeMap, BTreeSet};

use super::ir::{self, GateTarget, Instruction, Item};
use super::pauli_string::PauliString;
use super::rev_tracker::{is_observable, RevTracker, XorSet, OBSERVABLE_BIT};
use super::tableau_sim::{decompose_mpp, decompose_spp, TableauSimulator};
use crate::gate_data::{self, FLAG_CAN_TARGET_BITS, FLAG_IS_NOISY, FLAG_IS_RESET, FLAG_IS_SINGLE_QUBIT_GATE, FLAG_PRODUCES_RESULTS, FLAG_TARGETS_PAIRS};

fn inst(name: &str, args: Vec<f64>, targets: Vec<GateTarget>, tag: &str) -> Instruction {
    Instruction::new(name, args, targets, tag).expect("known gate")
}

// ---------------------------------------------------------------------------------------------
// decomposed

struct Simplifier<'a> {
    out: &'a mut ir::Circuit,
    num_qubits: usize,
}

impl Simplifier<'_> {
    fn y(&mut self, g: &str, t: &[GateTarget], tag: &str) {
        self.out.push_op(inst(g, vec![], t.to_vec(), tag));
    }

    fn seq(&mut self, gates: &[&str], t: &[GateTarget], tag: &str) {
        for g in gates {
            self.y(g, t, tag);
        }
    }

    fn xcz(&mut self, t: &[GateTarget], tag: &str) {
        if t.is_empty() {
            return;
        }
        let swapped: Vec<GateTarget> = t.chunks(2).flat_map(|p| [p[1], p[0]]).collect();
        self.y("CX", &swapped, tag);
    }

    fn disjoint_1q(&mut self, op: &Instruction) -> Result<(), String> {
        let (t, tag) = (&op.targets[..], op.tag.as_str());
        let s = match op.gate.name {
            "I" => &[][..],
            "X" => &["H", "S", "S", "H"][..],
            "Y" => &["H", "S", "S", "H", "S", "S"][..],
            "Z" => &["S", "S"][..],
            "C_XYZ" => &["S", "S", "S", "H"][..],
            "C_NXYZ" => &["S", "S", "S", "H", "S", "S"][..],
            "C_XNYZ" => &["S", "H"][..],
            "C_XYNZ" => &["S", "H", "S", "S"][..],
            "C_ZYX" => &["H", "S"][..],
            "C_ZYNX" => &["S", "S", "H", "S"][..],
            "C_ZNYX" => &["H", "S", "S", "S"][..],
            "C_NZYX" => &["S", "S", "H", "S", "S", "S"][..],
            "H" => &["H"][..],
            "H_XY" => &["H", "S", "S", "H", "S"][..],
            "H_YZ" => &["H", "S", "H", "S", "S"][..],
            "H_NXY" => &["S", "H", "S", "S", "H"][..],
            "H_NXZ" => &["S", "S", "H", "S", "S"][..],
            "H_NYZ" => &["S", "S", "H", "S", "H"][..],
            "S" => &["S"][..],
            "SQRT_X" => &["H", "S", "H"][..],
            "SQRT_X_DAG" => &["H", "S", "S", "S", "H"][..],
            "SQRT_Y" => &["S", "S", "H"][..],
            "SQRT_Y_DAG" => &["H", "S", "S"][..],
            "S_DAG" => &["S", "S", "S"][..],
            "MX" => &["H", "M", "H"][..],
            "MY" => &["S", "S", "S", "H", "M", "H", "S"][..],
            "M" => &["M"][..],
            "MRX" => &["H", "M", "R", "H"][..],
            "MRY" => &["S", "S", "S", "H", "M", "R", "H", "S"][..],
            "MR" => &["M", "R"][..],
            "RX" => &["R", "H"][..],
            "RY" => &["R", "H", "S"][..],
            "R" => &["R"][..],
            _ => return Err(format!("Unhandled in Simplifier::simplify_disjoint_1q_instruction: {op}")),
        };
        self.seq(s, t, tag);
        Ok(())
    }

    fn disjoint_2q(&mut self, op: &Instruction) -> Result<(), String> {
        let (ts, tag) = (&op.targets[..], op.tag.as_str());
        let (mut q1, mut q2, mut qs) = (Vec::new(), Vec::new(), Vec::new());
        for p in ts.chunks(2) {
            if p[0].has_qubit_value() {
                let t = GateTarget::qubit(p[0].value(), false);
                q1.push(t);
                qs.push(t);
            }
            if p[1].has_qubit_value() {
                let t = GateTarget::qubit(p[1].value(), false);
                q2.push(t);
                qs.push(t);
            }
        }
        match op.gate.name {
            "CX" => self.y("CX", ts, tag),
            "XCZ" => self.xcz(ts, tag),
            "XCX" => {
                self.y("H", &q1, tag);
                self.y("CX", ts, tag);
                self.y("H", &q1, tag);
            }
            "XCY" => {
                self.seq(&["S", "S", "S"], &q2, tag);
                self.y("H", &q1, tag);
                self.y("CX", ts, tag);
                self.y("H", &q1, tag);
                self.y("S", &q2, tag);
            }
            "YCX" => {
                self.seq(&["S", "S", "S", "H"], &q1, tag);
                self.y("CX", ts, tag);
                self.seq(&["H", "S"], &q1, tag);
            }
            "YCY" => {
                self.seq(&["S", "S", "S"], &qs, tag);
                self.y("H", &q1, tag);
                self.y("CX", ts, tag);
                self.y("H", &q1, tag);
                self.y("S", &qs, tag);
            }
            "YCZ" => {
                self.seq(&["S", "S", "S"], &q1, tag);
                self.xcz(ts, tag);
                self.y("S", &q1, tag);
            }
            "CY" => {
                self.seq(&["S", "S", "S"], &q2, tag);
                self.y("CX", ts, tag);
                self.y("S", &q2, tag);
            }
            "CZ" => {
                self.y("H", &q2, tag);
                self.y("CX", ts, tag);
                self.y("H", &q2, tag);
            }
            "SQRT_XX" => {
                self.y("H", &q1, tag);
                self.y("CX", ts, tag);
                self.y("H", &q2, tag);
                self.seq(&["S", "H"], &qs, tag);
            }
            "SQRT_XX_DAG" => {
                self.y("H", &q1, tag);
                self.y("CX", ts, tag);
                self.y("H", &q2, tag);
                self.seq(&["S", "S", "S", "H"], &qs, tag);
            }
            "SQRT_YY" => {
                self.seq(&["S", "S", "S"], &qs, tag);
                self.y("H", &q1, tag);
                self.y("CX", ts, tag);
                self.y("H", &q2, tag);
                self.seq(&["S", "H", "S"], &qs, tag);
            }
            "SQRT_YY_DAG" => {
                self.seq(&["S", "S"], &q1, tag);
                self.y("S", &qs, tag);
                self.y("H", &q1, tag);
                self.y("CX", ts, tag);
                self.y("H", &q2, tag);
                self.seq(&["S", "H", "S"], &qs, tag);
                self.seq(&["S", "S"], &q2, tag);
            }
            "SQRT_ZZ" => {
                self.y("H", &q2, tag);
                self.y("CX", ts, tag);
                self.y("H", &q2, tag);
                self.y("S", &qs, tag);
            }
            "SQRT_ZZ_DAG" => {
                self.y("H", &q2, tag);
                self.y("CX", ts, tag);
                self.y("H", &q2, tag);
                self.seq(&["S", "S", "S"], &qs, tag);
            }
            "SWAP" => {
                self.y("CX", ts, tag);
                self.xcz(ts, tag);
                self.y("CX", ts, tag);
            }
            // Stim's XCZ helper reuses the buffer of both qubits, leaving it holding the pairs
            // swapped; the S layer after it in ISWAP follows that order.
            "ISWAP" => {
                self.y("H", &q1, tag);
                self.y("CX", ts, tag);
                self.xcz(ts, tag);
                let qs: Vec<GateTarget> = ts.chunks(2).flat_map(|p| [p[1], p[0]]).collect();
                self.y("H", &q2, tag);
                self.y("S", &qs, tag);
            }
            "ISWAP_DAG" => {
                self.y("H", &q1, tag);
                self.y("CX", ts, tag);
                self.xcz(ts, tag);
                let qs: Vec<GateTarget> = ts.chunks(2).flat_map(|p| [p[1], p[0]]).collect();
                self.y("H", &q2, tag);
                self.seq(&["S", "S", "S"], &qs, tag);
            }
            "CXSWAP" => {
                self.xcz(ts, tag);
                self.y("CX", ts, tag);
            }
            "SWAPCX" => {
                self.y("CX", ts, tag);
                self.xcz(ts, tag);
            }
            "CZSWAP" => {
                self.y("H", &q1, tag);
                self.y("CX", ts, tag);
                self.xcz(ts, tag);
                self.y("H", &q2, tag);
            }
            "MXX" => {
                self.y("CX", ts, tag);
                self.seq(&["H", "M", "H"], &q1, tag);
                self.y("CX", ts, tag);
            }
            "MYY" => {
                self.y("S", &qs, tag);
                self.y("CX", ts, tag);
                self.seq(&["S", "S"], &q2, tag);
                self.seq(&["H", "M", "H"], &q1, tag);
                self.y("CX", ts, tag);
                self.y("S", &qs, tag);
            }
            "MZZ" => {
                self.y("CX", ts, tag);
                self.y("M", &q2, tag);
                self.y("CX", ts, tag);
            }
            _ => return Err(format!("Unhandled in Simplifier::simplify_instruction: {op}")),
        }
        Ok(())
    }

    fn overlapping(&mut self, op: &Instruction, width: usize) -> Result<(), String> {
        let mut used = BTreeSet::new();
        let mut start = 0;
        let mut k = 0;
        while k < op.targets.len() {
            let group = &op.targets[k..(k + width).min(op.targets.len())];
            if group.iter().any(|t| t.has_qubit_value() && used.contains(&t.value())) {
                let piece = Instruction { targets: op.targets[start..k].to_vec(), ..op.clone() };
                self.disjoint(&piece, width)?;
                used.clear();
                start = k;
            }
            for t in group {
                if t.has_qubit_value() {
                    used.insert(t.value());
                }
            }
            k += width;
        }
        let piece = Instruction { targets: op.targets[start..].to_vec(), ..op.clone() };
        self.disjoint(&piece, width)
    }

    fn disjoint(&mut self, op: &Instruction, width: usize) -> Result<(), String> {
        if width == 1 {
            self.disjoint_1q(op)
        } else {
            self.disjoint_2q(op)
        }
    }

    fn instruction(&mut self, op: &Instruction) -> Result<(), String> {
        match op.gate.name {
            "I" | "II" => Ok(()),
            "MPP" => {
                for sub in decompose_mpp(op, self.num_qubits)? {
                    self.instruction(&sub)?;
                }
                Ok(())
            }
            "SPP" | "SPP_DAG" => {
                for sub in decompose_spp(op)? {
                    self.instruction(&sub)?;
                }
                Ok(())
            }
            "MPAD" | "DETECTOR" | "OBSERVABLE_INCLUDE" | "TICK" | "QUBIT_COORDS" | "SHIFT_COORDS" | "DEPOLARIZE1" | "DEPOLARIZE2" | "X_ERROR" | "Y_ERROR" | "Z_ERROR" | "I_ERROR" | "II_ERROR" | "PAULI_CHANNEL_1" | "PAULI_CHANNEL_2" | "E" | "ELSE_CORRELATED_ERROR" | "HERALDED_ERASE" | "HERALDED_PAULI_CHANNEL_1" => {
                self.out.push_op(op.clone());
                Ok(())
            }
            _ if op.has_flag(FLAG_IS_SINGLE_QUBIT_GATE) => self.overlapping(op, 1),
            _ if op.has_flag(FLAG_TARGETS_PAIRS) => self.overlapping(op, 2),
            _ => Err(format!("Unhandled in simplify_potentially_overlapping_instruction: {op}")),
        }
    }
}

/// Stim's `simplified_circuit` (`Circuit.decomposed`): the circuit in H, S, CX, M and R.
pub fn decomposed(c: &ir::Circuit) -> Result<ir::Circuit, String> {
    let n = c.count_qubits() as usize;
    let mut out = ir::Circuit::new();
    for it in &c.items {
        match it {
            Item::Repeat { count, body, tag } => out.push_repeat(*count, decomposed(body)?, tag),
            Item::Op(op) => Simplifier { out: &mut out, num_qubits: n }.instruction(op)?,
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------------------------
// flattened, without_noise, without_tags

fn flatten_into(c: &ir::Circuit, shift: &mut Vec<f64>, out: &mut ir::Circuit) {
    for it in &c.items {
        match it {
            Item::Repeat { count, body, .. } => {
                for _ in 0..*count {
                    flatten_into(body, shift, out);
                }
            }
            Item::Op(op) if op.gate.name == "SHIFT_COORDS" => {
                if shift.len() < op.args.len() {
                    shift.resize(op.args.len(), 0.0);
                }
                for (k, a) in op.args.iter().enumerate() {
                    shift[k] += a;
                }
            }
            Item::Op(op) => {
                let mut o = op.clone();
                if matches!(op.gate.name, "QUBIT_COORDS" | "DETECTOR") {
                    for (k, a) in o.args.iter_mut().enumerate() {
                        if k < shift.len() {
                            *a += shift[k];
                        }
                    }
                }
                out.push_op(o);
            }
        }
    }
}

/// Loops unrolled, coordinate shifts folded into the coordinates.
pub fn flattened(c: &ir::Circuit) -> ir::Circuit {
    let mut out = ir::Circuit::new();
    flatten_into(c, &mut Vec::new(), &mut out);
    out
}

/// Noise channels removed, measurement flip probabilities dropped, heralds made `MPAD 0`.
pub fn without_noise(c: &ir::Circuit) -> ir::Circuit {
    let mut out = ir::Circuit::new();
    for it in &c.items {
        match it {
            Item::Repeat { count, body, tag } => out.push_raw_repeat(*count, without_noise(body), tag),
            Item::Op(op) => {
                let f = op.gate.flags;
                if f & FLAG_PRODUCES_RESULTS != 0 {
                    if matches!(op.gate.name, "HERALDED_ERASE" | "HERALDED_PAULI_CHANNEL_1") {
                        out.push_op(inst("MPAD", vec![], vec![GateTarget::qubit(0, false); op.targets.len()], &op.tag));
                    } else {
                        out.push_op(Instruction { args: vec![], ..op.clone() });
                    }
                } else if f & FLAG_IS_NOISY == 0 {
                    out.push_op(op.clone());
                }
            }
        }
    }
    out
}

/// Every tag removed.
pub fn without_tags(c: &ir::Circuit) -> ir::Circuit {
    let mut out = ir::Circuit::new();
    for it in &c.items {
        match it {
            Item::Repeat { count, body, .. } => out.push_raw_repeat(*count, without_tags(body), ""),
            Item::Op(op) => out.push_op(Instruction { tag: String::new(), ..op.clone() }),
        }
    }
    out
}

// ---------------------------------------------------------------------------------------------
// inverse

/// Stim's `Circuit::inverse`: operations inverted and in reverse; with `allow_weak_inverse`,
/// noise, measurements and resets are kept in place and detectors dropped.
pub fn inverse(c: &ir::Circuit, allow_weak_inverse: bool) -> Result<ir::Circuit, String> {
    let mut items: Vec<Item> = Vec::new();
    let mut skip = 0usize;
    for (k, it) in c.items.iter().enumerate() {
        let op = match it {
            Item::Repeat { count, body, tag } => {
                items.push(Item::Repeat { count: *count, body: inverse(body, allow_weak_inverse)?, tag: tag.clone() });
                continue;
            }
            Item::Op(op) => op,
        };
        let f = op.gate.flags;
        let mut args = op.args.clone();
        if op.gate.is_unitary || op.gate.name == "TICK" {
        } else if f & FLAG_IS_NOISY != 0 {
            if !allow_weak_inverse || op.gate.name == "ELSE_CORRELATED_ERROR" {
                return Err(format!("The circuit has no well-defined inverse because it contains noise.\nFor example it contains a '{op}' instruction."));
            }
        } else if f & (FLAG_IS_RESET | FLAG_PRODUCES_RESULTS) != 0 {
            if !allow_weak_inverse {
                return Err(format!("The circuit has no well-defined inverse because it contains resets or measurements.\nFor example it contains a '{op}' instruction."));
            }
        } else if op.gate.name == "QUBIT_COORDS" {
            if k > skip {
                return Err("Inverting QUBIT_COORDS is not implemented except at the start of the circuit.".into());
            }
            skip += 1;
        } else if op.gate.name == "SHIFT_COORDS" {
            args = args.iter().map(|a| -a).collect();
        } else if matches!(op.gate.name, "DETECTOR" | "OBSERVABLE_INCLUDE") {
            if allow_weak_inverse {
                continue;
            }
            return Err(format!("Inverse not implemented: {op}"));
        } else {
            return Err(format!("Inverse not implemented: {op}"));
        }
        let pairs = op.has_flag(FLAG_TARGETS_PAIRS);
        let mut targets = Vec::with_capacity(op.targets.len());
        if pairs {
            for p in op.targets.chunks(2).rev() {
                targets.extend_from_slice(p);
            }
        } else {
            targets.extend(op.targets.iter().rev());
        }
        let name = gate_inverse_candidate(op.gate.name);
        let new = inst(name, args, targets, &op.tag);
        // Stim's safe_append_reversed_targets: fuse with the previous instruction if possible.
        if let Some(Item::Op(last)) = items.last_mut() {
            if last.can_fuse(&new) {
                last.targets.extend(new.targets);
                continue;
            }
        }
        items.push(Item::Op(new));
    }
    items[skip..].reverse();
    Ok(ir::Circuit { items })
}

/// Stim's `best_candidate_inverse_id`.
pub fn gate_inverse_candidate(name: &str) -> &'static str {
    let g = gate_data::info(name).expect("gate");
    g.inverse_candidate
}

// ---------------------------------------------------------------------------------------------
// coordinates

fn pad_add_mul(target: &mut Vec<f64>, offset: &[f64], mul: u64) {
    if target.len() < offset.len() {
        target.resize(offset.len(), 0.0);
    }
    for (k, o) in offset.iter().enumerate() {
        target[k] += o * mul as f64;
    }
}

fn final_coord_shift(c: &ir::Circuit) -> Vec<f64> {
    let mut s = Vec::new();
    for it in &c.items {
        match it {
            Item::Op(op) if op.gate.name == "SHIFT_COORDS" => pad_add_mul(&mut s, &op.args, 1),
            Item::Repeat { count, body, .. } => pad_add_mul(&mut s, &final_coord_shift(body), *count),
            _ => {}
        }
    }
    s
}

fn final_qubit_coords_helper(c: &ir::Circuit, reps: u64, shift: &mut Vec<f64>, out: &mut BTreeMap<u64, Vec<f64>>) {
    let initial = shift.clone();
    let mut new: BTreeMap<u64, Vec<f64>> = BTreeMap::new();
    for it in &c.items {
        match it {
            Item::Repeat { count, body, .. } => final_qubit_coords_helper(body, *count, shift, &mut new),
            Item::Op(op) if op.gate.name == "SHIFT_COORDS" => pad_add_mul(shift, &op.args, 1),
            Item::Op(op) if op.gate.name == "QUBIT_COORDS" => {
                if shift.len() < op.args.len() {
                    shift.resize(op.args.len(), 0.0);
                }
                for t in &op.targets {
                    if t.is_qubit() {
                        let v = new.entry(t.value() as u64).or_default();
                        for (k, a) in op.args.iter().enumerate() {
                            v.push(a + shift[k]);
                        }
                    }
                }
            }
            _ => {}
        }
    }
    if reps > 1 && *shift != initial {
        let mut gain = shift.clone();
        for (k, v) in initial.iter().enumerate() {
            gain[k] -= v;
        }
        for qc in new.values_mut() {
            for (k, v) in qc.iter_mut().enumerate() {
                *v += gain[k] * (reps - 1) as f64;
            }
        }
        pad_add_mul(shift, &gain, reps - 1);
    }
    for (k, v) in new {
        out.insert(k, v);
    }
}

/// Every qubit's coordinates at the end of the circuit.
pub fn final_qubit_coordinates(c: &ir::Circuit) -> BTreeMap<u64, Vec<f64>> {
    let mut out = BTreeMap::new();
    final_qubit_coords_helper(c, 1, &mut Vec::new(), &mut out);
    out
}

fn count_detectors(c: &ir::Circuit) -> u64 {
    c.items
        .iter()
        .map(|it| match it {
            Item::Op(op) => (op.gate.name == "DETECTOR") as u64,
            Item::Repeat { count, body, .. } => count.saturating_mul(count_detectors(body)),
        })
        .fold(0u64, |a, b| a.saturating_add(b))
}

fn detector_coords_helper(c: &ir::Circuit, wanted: &[u64], pos: &mut usize, initial: &[f64], next: &mut u64, out: &mut BTreeMap<u64, Vec<f64>>) {
    if *pos >= wanted.len() {
        return;
    }
    let mut shift = initial.to_vec();
    for it in &c.items {
        match it {
            Item::Op(op) if op.gate.name == "SHIFT_COORDS" => pad_add_mul(&mut shift, &op.args, 1),
            Item::Repeat { count, body, .. } => {
                let block_shift = final_coord_shift(body);
                let per = count_detectors(body);
                let reps = *count;
                let mut used = 0u64;
                while used < reps {
                    let skip = if per == 0 { reps } else { reps.min((wanted[*pos] - *next) / per) };
                    used += skip;
                    *next += per * skip;
                    pad_add_mul(&mut shift, &block_shift, skip);
                    if used < reps {
                        detector_coords_helper(body, wanted, pos, &shift, next, out);
                        used += 1;
                        pad_add_mul(&mut shift, &block_shift, 1);
                        if *pos >= wanted.len() {
                            return;
                        }
                    }
                }
            }
            Item::Op(op) if op.gate.name == "DETECTOR" => {
                if *next == wanted[*pos] {
                    let coords = op.args.iter().enumerate().map(|(k, a)| a + shift.get(k).copied().unwrap_or(0.0)).collect();
                    out.insert(*next, coords);
                    *pos += 1;
                    if *pos >= wanted.len() {
                        return;
                    }
                }
                *next += 1;
            }
            _ => {}
        }
    }
}

/// The coordinates of the given detectors (Stim's `get_detector_coordinates`).
pub fn detector_coordinates(c: &ir::Circuit, wanted: &BTreeSet<u64>) -> Result<BTreeMap<u64, Vec<f64>>, String> {
    let w: Vec<u64> = wanted.iter().copied().collect();
    let mut out = BTreeMap::new();
    let mut pos = 0;
    let mut next = 0;
    detector_coords_helper(c, &w, &mut pos, &[], &mut next, &mut out);
    if pos < w.len() {
        return Err(format!("Detector index {} is too big. The circuit has {} detectors)", w[pos], count_detectors(c)));
    }
    Ok(out)
}

// ---------------------------------------------------------------------------------------------
// reference signs and determined measurements

/// Stim's reference sample: the noiseless circuit run with every random outcome +1.
pub fn reference_sample(c: &ir::Circuit) -> Result<Vec<bool>, String> {
    let quiet = without_noise(c);
    let mut sim = TableauSimulator::new(c.count_qubits() as usize, 0);
    sim.sign_bias = 1;
    sim.do_circuit(&quiet)?;
    Ok(sim.measurement_record)
}

/// The noiseless parity of each detector's and observable's measurement set.
pub fn reference_detector_and_observable_signs(c: &ir::Circuit, num_observables: usize) -> Result<(Vec<bool>, Vec<bool>), String> {
    let r = reference_sample(c)?;
    let mut dets = Vec::new();
    let mut obs = vec![false; num_observables];
    let mut offset: i64 = 0;
    c.for_each_operation(&mut |op| {
        if matches!(op.gate.name, "DETECTOR" | "OBSERVABLE_INCLUDE") {
            let mut v = false;
            for t in &op.targets {
                if t.is_record() {
                    v ^= r[(offset - t.value() as i64) as usize];
                }
            }
            if op.gate.name == "DETECTOR" {
                dets.push(v);
            } else {
                obs[op.args[0] as usize] ^= v;
            }
        } else {
            offset += op.count_measurement_results() as i64;
        }
    });
    Ok((dets, obs))
}

/// How many measurements have a deterministic outcome (Stim's `count_determined_measurements`).
pub fn count_determined_measurements(c: &ir::Circuit, unknown_input: bool) -> Result<u64, String> {
    let mut n = c.count_qubits() as usize;
    let mut sim = TableauSimulator::new(n, 0);
    if unknown_input {
        sim.ensure_large_enough_for_qubits(2 * n);
        for k in 0..n {
            sim.do_gate(&inst("XCX", vec![], vec![GateTarget::qubit(k as u32, false), GateTarget::qubit((k + n) as u32, false)], ""))?;
        }
        n *= 2;
    }
    let mut result = 0u64;
    let mut err = Ok(());
    c.for_each_operation(&mut |op| {
        if err.is_err() {
            return;
        }
        let r = (|| -> Result<(), String> {
            if op.gate.flags & FLAG_PRODUCES_RESULTS == 0 {
                return sim.do_gate(op);
            }
            match op.gate.name {
                "M" | "MR" | "MX" | "MRX" | "MY" | "MRY" => {
                    for t in &op.targets {
                        let q = t.value() as usize;
                        let peek = match op.gate.name {
                            "M" | "MR" => sim.peek_z(q),
                            "MX" | "MRX" => sim.peek_x(q),
                            _ => sim.peek_y(q),
                        };
                        result += (peek != 0) as u64;
                        sim.do_gate(&inst(op.gate.name, vec![], vec![*t], ""))?;
                    }
                }
                "MXX" | "MYY" | "MZZ" => {
                    let x = op.gate.name != "MZZ";
                    let z = op.gate.name != "MXX";
                    for p in op.targets.chunks(2) {
                        let mut obs = PauliString::new(n);
                        for t in p {
                            obs.set_x(t.value() as usize, x);
                            obs.set_z(t.value() as usize, z);
                        }
                        result += (sim.peek_observable_expectation(&obs) != 0) as u64;
                        sim.do_gate(&inst(op.gate.name, vec![], p.to_vec(), ""))?;
                    }
                }
                "MPP" => {
                    let t = &op.targets;
                    let mut start = 0;
                    while start < t.len() {
                        let mut end = start + 1;
                        while end < t.len() && t[end].is_combiner() {
                            end += 2;
                        }
                        let mut obs = PauliString::new(n);
                        for k in (start..end).step_by(2) {
                            obs.set_x(t[k].value() as usize, t[k].0 & ir::TARGET_PAULI_X_BIT != 0);
                            obs.set_z(t[k].value() as usize, t[k].0 & ir::TARGET_PAULI_Z_BIT != 0);
                        }
                        result += (sim.peek_observable_expectation(&obs) != 0) as u64;
                        sim.do_gate(&inst("MPP", vec![], t[start..end].to_vec(), ""))?;
                        start = end;
                    }
                }
                _ => return Err(format!("count_determined_measurements unhandled measurement type {op}")),
            }
            Ok(())
        })();
        if r.is_err() {
            err = r;
        }
    });
    err?;
    Ok(result)
}

// ---------------------------------------------------------------------------------------------
// with_inlined_feedback

struct FeedbackHelper {
    out: ir::Circuit,
    tracker: RevTracker,
    obs_changes: BTreeMap<u64, BTreeSet<u32>>,
    det_changes: BTreeMap<u64, XorSet>,
}

impl FeedbackHelper {
    fn sensitivity(&self, q: usize, x: bool, z: bool) -> XorSet {
        if x && !z {
            self.tracker.zs[q].clone()
        } else if z && !x {
            self.tracker.xs[q].clone()
        } else {
            let mut s = self.tracker.xs[q].clone();
            s.xor(&self.tracker.zs[q]);
            s
        }
    }

    fn single(&mut self, rec: GateTarget, q: usize, x: bool, z: bool) {
        for d in self.sensitivity(q, x, z).0 {
            if is_observable(d) {
                let set = self.obs_changes.entry(d & !OBSERVABLE_BIT).or_default();
                if !set.remove(&rec.0) {
                    set.insert(rec.0);
                }
            } else {
                let m = (self.tracker.num_measurements_in_past as i64 - rec.value() as i64) as u64;
                self.det_changes.entry(d).or_default().xor_item(m);
            }
        }
    }

    fn undo_feedback_op(&mut self, op: &Instruction) -> Result<(), String> {
        let mut k = op.targets.len();
        while k > 0 {
            k -= 2;
            let piece = Instruction { targets: op.targets[k..k + 2].to_vec(), ..op.clone() };
            let (t1, t2) = (op.targets[k], op.targets[k + 1]);
            let (b1, b2) = (t1.is_record(), t2.is_record());
            let xz = |n: &str| match n {
                "CX" => Ok((true, false)),
                "CY" => Ok((true, true)),
                "CZ" => Ok((false, true)),
                _ => Err("Unknown feedback gate.".to_string()),
            };
            if b1 && !b2 {
                let (x, z) = xz(op.gate.name)?;
                self.single(t1, t2.value() as usize, x, z);
            } else if b2 && !b1 {
                let (x, z) = xz(op.gate.name)?;
                self.single(t2, t1.value() as usize, x, z);
            } else if !b1 && !b2 {
                self.out.items.push(Item::Op(piece.clone()));
            }
            self.tracker.undo_gate(&piece)?;
        }
        let changes = std::mem::take(&mut self.obs_changes);
        for (obs, recs) in changes {
            if !recs.is_empty() {
                let targets = recs.into_iter().map(GateTarget).collect();
                self.out.items.push(Item::Op(inst("OBSERVABLE_INCLUDE", vec![obs as f64], targets, &op.tag)));
            }
        }
        Ok(())
    }

    fn undo_circuit(&mut self, c: &ir::Circuit) -> Result<(), String> {
        for it in c.items.iter().rev() {
            match it {
                Item::Repeat { count, body, tag } => {
                    let saved = std::mem::take(&mut self.out);
                    let mut tmp = saved;
                    for _ in 0..*count {
                        self.out = ir::Circuit::new();
                        self.undo_circuit(body)?;
                        let piece = std::mem::take(&mut self.out);
                        tmp.items.push(Item::Repeat { count: 1, body: piece, tag: tag.clone() });
                    }
                    self.out = tmp;
                }
                Item::Op(op) => {
                    if op.has_flag(FLAG_CAN_TARGET_BITS) {
                        self.undo_feedback_op(op)?;
                    } else {
                        self.out.items.push(Item::Op(op.clone()));
                        self.tracker.undo_gate(op)?;
                    }
                }
            }
        }
        Ok(())
    }

    fn build(&mut self, reversed: &ir::Circuit) -> ir::Circuit {
        let mut result = ir::Circuit::new();
        for it in reversed.items.iter().rev() {
            match it {
                Item::Repeat { count, body, tag } => {
                    let b = self.build(body);
                    result.push_raw_repeat(*count, b, tag);
                }
                Item::Op(op) => {
                    self.tracker.num_measurements_in_past += op.count_measurement_results();
                    if op.gate.name == "DETECTOR" {
                        let idx = self.tracker.num_detectors_in_past;
                        self.tracker.num_detectors_in_past += 1;
                        if let Some(changes) = self.det_changes.get_mut(&idx) {
                            for t in &op.targets {
                                changes.xor_item((self.tracker.num_measurements_in_past as i64 - t.value() as i64) as u64);
                            }
                            let m = self.tracker.num_measurements_in_past as i64;
                            let targets = changes.iter().map(|&x| GateTarget::rec((m - x as i64) as u32)).collect();
                            result.push_op(Instruction { targets, ..op.clone() });
                            continue;
                        }
                    }
                    result.push_op(op.clone());
                }
            }
        }
        result
    }
}

fn fuse_identical_adjacent_loops(c: &ir::Circuit) -> ir::Circuit {
    let mut result = ir::Circuit::new();
    let mut growing: Option<(ir::Circuit, u64, String)> = None;
    let flush = |result: &mut ir::Circuit, g: &mut Option<(ir::Circuit, u64, String)>| {
        if let Some((body, reps, tag)) = g.take() {
            let body = fuse_identical_adjacent_loops(&body);
            if reps > 1 {
                result.push_raw_repeat(reps, body, &tag);
            } else {
                result.extend(&body);
            }
        }
    };
    for it in &c.items {
        match it {
            Item::Repeat { count, body, tag } => {
                if let Some((g, reps, _)) = growing.as_mut() {
                    if g == body {
                        *reps += count;
                        continue;
                    }
                }
                flush(&mut result, &mut growing);
                growing = Some((body.clone(), *count, tag.clone()));
            }
            Item::Op(op) => {
                flush(&mut result, &mut growing);
                result.push_op(op.clone());
            }
        }
    }
    flush(&mut result, &mut growing);
    result
}

/// The circuit without feedback: each classically controlled Pauli removed, the detectors and
/// observables it affected rewritten to include the controlling measurement instead.
pub fn with_inlined_feedback(c: &ir::Circuit) -> Result<ir::Circuit, String> {
    let n = c.count_qubits() as usize;
    let mut h = FeedbackHelper { out: ir::Circuit::new(), tracker: RevTracker::new(n, c.count_measurements(), count_detectors(c), false), obs_changes: BTreeMap::new(), det_changes: BTreeMap::new() };
    h.undo_circuit(c)?;
    let reversed = std::mem::take(&mut h.out);
    let out = h.build(&reversed);
    Ok(fuse_identical_adjacent_loops(&out))
}

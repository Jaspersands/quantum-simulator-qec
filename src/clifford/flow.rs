//! Stabilizer flows, as `stim.Flow`: `P -> Q xor rec[...] xor obs[...]`, with Stim's text form
//! and product; and the circuit methods built on them (`has_flow`, `flow_generators`,
//! `solve_flow_measurements`), ported from Stim's `flow.inl`, `has_flow.inl` and
//! `circuit_flow_generators.inl`.

use std::cmp::Ordering;
use std::collections::BTreeSet;
use std::fmt;

use super::ir::{self, GateTarget, Instruction, Item};
use super::pauli_string::PauliString;
use super::rev_tracker::{RevTracker, XorSet};
use super::tableau_sim::{decompose_mpp, decompose_spp, gate_tableaus, TableauSimulator};
use super::transform::without_noise;

/// A flow. The Pauli strings have real signs; measurements are record indices (negative from
/// the end, or non-negative from the start), sorted with duplicates cancelled.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Flow {
    pub input: PauliString,
    pub output: PauliString,
    pub measurements: Vec<i32>,
    pub observables: Vec<u32>,
}

/// Sort and cancel pairs (Stim's `inplace_xor_sort`).
pub fn xor_sort<T: Ord + Copy>(v: &mut Vec<T>) {
    v.sort();
    let mut out: Vec<T> = Vec::with_capacity(v.len());
    for x in v.drain(..) {
        if out.last() == Some(&x) {
            out.pop();
        } else {
            out.push(x);
        }
    }
    *v = out;
}

fn xor_merge(a: &[i32], b: &[i32]) -> Vec<i32> {
    let mut v: Vec<i32> = a.iter().chain(b).copied().collect();
    xor_sort(&mut v);
    v
}

/// Stim's order on Pauli strings: qubit by qubit I < X < Y < Z, then length, then sign.
pub fn pauli_cmp(a: &PauliString, b: &PauliString) -> Ordering {
    let key = |p: &PauliString, q: usize| (p.x(q) ^ p.z(q)) as u8 + 2 * p.z(q) as u8;
    let n = a.num_qubits().min(b.num_qubits());
    for q in 0..n {
        let (p1, p2) = (key(a, q), key(b, q));
        if p1 != p2 {
            return p1.cmp(&p2);
        }
    }
    a.num_qubits().cmp(&b.num_qubits()).then(a.sign().cmp(&b.sign()))
}

impl Flow {
    pub fn new(input: PauliString, output: PauliString, mut measurements: Vec<i32>, mut observables: Vec<u32>) -> Flow {
        xor_sort(&mut measurements);
        xor_sort(&mut observables);
        Flow { input, output, measurements, observables }
    }

    /// Parse Stim's text: `X_ -> -XZ xor rec[-1] xor obs[2]`, `1 -> Z`, sparse `X2*Y5`.
    pub fn from_text(text: &str) -> Result<Flow, String> {
        let bad = || format!("Invalid stabilizer flow text: '{text}'.");
        let parts: Vec<&str> = text.split('>').collect();
        if parts.len() != 2 || !parts[0].ends_with('-') {
            return Err(bad());
        }
        let left = parts[0][..parts[0].len() - 1].trim_end_matches(' ');
        let pauli = |t: &str| -> Result<(PauliString, bool), String> {
            match t {
                "+1" | "1" => Ok((PauliString::new(0), false)),
                "-1" => {
                    let mut p = PauliString::new(0);
                    p.phase = 2;
                    Ok((p, false))
                }
                "" => Err("Got an ambiguously blank pauli string. Use '1' for the empty Pauli string.".into()),
                _ => {
                    let p = PauliString::from_text(t).map_err(|_| bad())?;
                    let imag = p.is_imaginary();
                    let mut r = p;
                    r.phase &= 2;
                    Ok((r, imag))
                }
            }
        };
        let (input, imag_in) = pauli(left)?;
        let words: Vec<&str> = parts[1].split(' ').collect();
        let mut k = 0;
        while k < words.len() && words[k].is_empty() {
            k += 1;
        }
        if k >= words.len() {
            return Err(bad());
        }
        let mut output = PauliString::new(0);
        let mut imag_out = false;
        let mut measurements = Vec::new();
        let mut observables = Vec::new();
        let rec = |w: &str| -> Option<i32> { w.strip_prefix("rec[")?.strip_suffix(']')?.parse::<i64>().ok().filter(|v| *v >= i32::MIN as i64 && *v <= i32::MAX as i64).map(|v| v as i32) };
        let obs = |w: &str| -> Option<u32> { w.strip_prefix("obs[")?.strip_suffix(']')?.parse::<u32>().ok() };
        let mut first = words[k];
        let flip = first.starts_with('-');
        if flip {
            first = &first[1..];
        }
        if !first.is_empty() && !first.starts_with('r') && !first.starts_with('o') {
            let (p, i) = pauli(first)?;
            output = p;
            imag_out = i;
        } else if first.starts_with('r') {
            measurements.push(rec(first).ok_or_else(bad)?);
        } else if first.starts_with('o') {
            observables.push(obs(first).ok_or_else(bad)?);
        } else {
            return Err(bad());
        }
        if flip {
            output.phase ^= 2;
        }
        k += 1;
        while k < words.len() {
            if words[k] != "xor" || k + 1 == words.len() {
                return Err(bad());
            }
            let w = words[k + 1];
            if w.starts_with('r') {
                measurements.push(rec(w).ok_or_else(bad)?);
            } else if w.starts_with('o') {
                observables.push(obs(w).ok_or_else(bad)?);
            } else {
                return Err(bad());
            }
            k += 2;
        }
        if imag_in != imag_out {
            return Err("Anti-Hermitian flows aren't allowed.".into());
        }
        Ok(Flow::new(input, output, measurements, observables))
    }

    /// The product of two flows (an error if their inputs and outputs make it anti-Hermitian).
    pub fn mul(&self, rhs: &Flow) -> Result<Flow, String> {
        let mut input = self.input.clone();
        let mut output = self.output.clone();
        input.ensure_num_qubits(rhs.input.num_qubits());
        output.ensure_num_qubits(rhs.output.num_qubits());
        let mut ri = rhs.input.clone();
        let mut ro = rhs.output.clone();
        ri.ensure_num_qubits(input.num_qubits());
        ro.ensure_num_qubits(output.num_qubits());
        let ip = input.phase;
        let op = output.phase;
        input.mul_assign(&ri);
        output.mul_assign(&ro);
        // Stim tallies each side's power of i with the right side's sign folded in, subtracts
        // the input's from the output's, and keeps the left input's sign.
        let li = (input.phase + 4 - ip) & 3;
        let lo = (output.phase + 4 - op) & 3;
        let log_i = (lo + 4 - li) & 3;
        input.phase = ip;
        output.phase = op;
        if log_i & 1 != 0 {
            return Err(format!("{self} anticommutes with {rhs}"));
        }
        if log_i & 2 != 0 {
            output.phase ^= 2;
        }
        let mut m = self.measurements.clone();
        m.extend(&rhs.measurements);
        let mut o = self.observables.clone();
        o.extend(&rhs.observables);
        Ok(Flow::new(input, output, m, o))
    }
}

impl PartialOrd for Flow {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Flow {
    fn cmp(&self, o: &Self) -> Ordering {
        pauli_cmp(&self.input, &o.input).then_with(|| pauli_cmp(&self.output, &o.output)).then_with(|| self.measurements.cmp(&o.measurements)).then_with(|| self.observables.cmp(&o.observables))
    }
}

impl fmt::Display for Flow {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let sparse_ok = |p: &PauliString| p.num_qubits() > 8 && p.weight() * 8 <= p.num_qubits();
        let last_empty = |p: &PauliString| p.num_qubits() > 0 && !p.x(p.num_qubits() - 1) && !p.z(p.num_qubits() - 1);
        let mut use_sparse = sparse_ok(&self.input) || sparse_ok(&self.output);
        if last_empty(&self.input) || last_empty(&self.output) {
            use_sparse = false;
        }
        let write = |p: &PauliString, s: &mut String| -> bool {
            if p.sign() {
                s.push('-');
            }
            let mut any = false;
            if use_sparse {
                for q in 0..p.num_qubits() {
                    let v = p.x(q) as usize + 2 * p.z(q) as usize;
                    if v != 0 {
                        if any {
                            s.push('*');
                        }
                        s.push(['_', 'X', 'Z', 'Y'][v]);
                        s.push_str(&q.to_string());
                        any = true;
                    }
                }
            } else {
                for q in 0..p.num_qubits() {
                    s.push(['_', 'X', 'Z', 'Y'][p.x(q) as usize + 2 * p.z(q) as usize]);
                    any = true;
                }
            }
            any
        };
        let mut s = String::new();
        if !write(&self.input, &mut s) {
            s.push('1');
        }
        s.push_str(" -> ");
        let mut has = write(&self.output, &mut s);
        for m in &self.measurements {
            if has {
                s.push_str(" xor ");
            }
            has = true;
            s.push_str(&format!("rec[{m}]"));
        }
        for o in &self.observables {
            if has {
                s.push_str(" xor ");
            }
            has = true;
            s.push_str(&format!("obs[{o}]"));
        }
        if !has {
            s.push('1');
        }
        f.write_str(&s)
    }
}

/// Conjugate `p` backwards through an instruction (U† P U), as Stim's `undo_instruction`.
pub fn undo_instruction(p: &mut PauliString, op: &Instruction) -> Result<(), String> {
    if op.gate.name == "SPP" || op.gate.name == "SPP_DAG" {
        let mut subs = decompose_spp(op)?;
        subs.reverse();
        for s in &subs {
            undo_instruction(p, s)?;
        }
        return Ok(());
    }
    if matches!(op.gate.name, "I" | "II" | "I_ERROR" | "II_ERROR") {
        return Ok(());
    }
    for t in &op.targets {
        if t.has_qubit_value() && t.value() as usize >= p.num_qubits() {
            return Err(format!("The instruction '{op}' targets qubits outside the pauli string '{p}'."));
        }
    }
    let (_, inv) = gate_tableaus(op.gate.name).ok_or_else(|| format!("Not a unitary instruction: {op}"))?;
    let w = inv.num_qubits();
    for g in op.targets.chunks(w).rev() {
        let q: Vec<usize> = g.iter().map(|t| t.value() as usize).collect();
        inv.apply_within(p, &q);
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// The flow-generator solver.

struct Solver {
    table: Vec<Flow>,
    imag: BTreeSet<usize>,
    num_qubits: usize,
    num_measurements: u64,
    in_past: i64,
    measurements_only: Vec<Flow>,
}

impl Solver {
    fn new(num_qubits: usize, num_measurements: u64) -> Result<Solver, String> {
        if num_measurements > i32::MAX as u64 {
            return Err(format!("Circuit is too large. Max flow measurement index is {}", i32::MAX));
        }
        Ok(Solver { table: Vec::new(), imag: BTreeSet::new(), num_qubits, num_measurements, in_past: num_measurements as i64, measurements_only: Vec::new() })
    }

    fn add_row(&mut self) -> &mut Flow {
        self.table.push(Flow { input: PauliString::new(self.num_qubits), output: PauliString::new(self.num_qubits), measurements: Vec::new(), observables: Vec::new() });
        self.table.last_mut().unwrap()
    }

    fn rows_with(&self, pred: impl Fn(&Flow) -> bool) -> Vec<usize> {
        (0..self.table.len()).filter(|&r| pred(&self.table[r])).collect()
    }

    fn rows_anticommuting(&self, q: usize, x: bool, z: bool) -> Vec<usize> {
        self.rows_with(|f| (f.input.x(q) & z) ^ (f.input.z(q) & x))
    }

    fn mult_row_into(&mut self, src: usize, dst: usize) {
        let (si, so, sm) = (self.table[src].input.clone(), self.table[src].output.clone(), self.table[src].measurements.clone());
        let d = &mut self.table[dst];
        let (ip, op) = (d.input.phase, d.output.phase);
        d.input.mul_assign(&si);
        d.output.mul_assign(&so);
        let li = (d.input.phase + 4 - ip) & 3;
        let lo = (d.output.phase + 4 - op) & 3;
        let log_i = (li + 4 - lo) & 3;
        d.input.phase = ip;
        d.output.phase = op;
        if log_i & 1 != 0 {
            if !self.imag.remove(&dst) {
                self.imag.insert(dst);
            }
        }
        if log_i & 2 != 0 {
            d.input.phase ^= 2;
        }
        d.measurements = xor_merge(&d.measurements, &sm);
    }

    fn handle_anticommutations(&mut self, set: &[usize]) {
        if set.is_empty() {
            return;
        }
        for &k in &set[1..] {
            self.mult_row_into(set[0], k);
        }
        self.table.remove(set[0]);
    }

    fn check_1q(&mut self, op: &Instruction, x: bool, z: bool) -> Result<(), String> {
        for t in &op.targets {
            if !t.is_qubit() && !(t.is_inverted() && t.0 & (ir::TARGET_PAULI_X_BIT | ir::TARGET_PAULI_Z_BIT | ir::TARGET_RECORD_BIT | ir::TARGET_SWEEP_BIT | ir::TARGET_COMBINER) == 0) {
                return Err(format!("Bad target in {op}"));
            }
            let s = self.rows_anticommuting(t.value() as usize, x, z);
            self.handle_anticommutations(&s);
        }
        Ok(())
    }

    fn remove_resets(&mut self, op: &Instruction) {
        for t in &op.targets {
            let q = t.value() as usize;
            for row in self.table.iter_mut() {
                row.input.set(q, 0);
            }
        }
    }

    fn add_1q_measure(&mut self, op: &Instruction, x: bool, z: bool) {
        for t in op.targets.iter().rev() {
            self.in_past -= 1;
            let m = self.in_past as i32;
            let q = t.value() as usize;
            let inv = t.is_inverted();
            let row = self.add_row();
            row.measurements.push(m);
            row.input.set_x(q, x);
            row.input.set_z(q, z);
            if inv {
                row.input.phase ^= 2;
            }
        }
    }

    fn undo(&mut self, op: &Instruction) -> Result<(), String> {
        if self.table.len() > self.num_qubits * 3 {
            let n = self.table.len();
            self.canonicalize_over_qubits(n);
        }
        let name = op.gate.name;
        // Stim: x unless the gate is the Z-basis one, z unless it is the X-basis one.
        let xz = |n: &str| -> (bool, bool) { (!matches!(n, "MR" | "M" | "R"), !matches!(n, "MRX" | "MX" | "RX")) };
        match name {
            "MRX" | "MRY" | "MR" => {
                let (x, z) = xz(name);
                self.check_1q(op, x, z)?;
                self.remove_resets(op);
                self.add_1q_measure(op, x, z);
            }
            "MX" | "MY" | "M" => {
                let (x, z) = xz(name);
                self.check_1q(op, x, z)?;
                self.add_1q_measure(op, x, z);
            }
            "RX" | "RY" | "R" => {
                let (x, z) = xz(name);
                self.check_1q(op, x, z)?;
                self.remove_resets(op);
            }
            "MXX" | "MYY" | "MZZ" => {
                let (x, z) = (name != "MZZ", name != "MXX");
                let mut k = op.targets.len();
                while k > 0 {
                    k -= 2;
                    let (q1, q2) = (op.targets[k].value() as usize, op.targets[k + 1].value() as usize);
                    let s = self.rows_with(|f| (f.input.x(q1) & z) ^ (f.input.z(q1) & x) ^ (f.input.x(q2) & z) ^ (f.input.z(q2) & x));
                    self.handle_anticommutations(&s);
                }
                let mut k = op.targets.len();
                while k > 0 {
                    k -= 2;
                    self.in_past -= 1;
                    let m = self.in_past as i32;
                    let (t1, t2) = (op.targets[k], op.targets[k + 1]);
                    let row = self.add_row();
                    row.measurements.push(m);
                    for t in [t1, t2] {
                        row.input.set_x(t.value() as usize, x);
                        row.input.set_z(t.value() as usize, z);
                    }
                    if t1.is_inverted() ^ t2.is_inverted() {
                        row.input.phase ^= 2;
                    }
                }
            }
            "DETECTOR" | "OBSERVABLE_INCLUDE" | "TICK" | "QUBIT_COORDS" | "SHIFT_COORDS" | "DEPOLARIZE1" | "DEPOLARIZE2" | "X_ERROR" | "Y_ERROR" | "Z_ERROR" | "PAULI_CHANNEL_1" | "PAULI_CHANNEL_2" | "E" | "ELSE_CORRELATED_ERROR" => {}
            "HERALDED_ERASE" | "HERALDED_PAULI_CHANNEL_1" => {
                for _ in &op.targets {
                    self.in_past -= 1;
                    let m = self.in_past as i32;
                    self.add_row().measurements.push(m);
                }
            }
            "MPAD" => {
                for t in &op.targets {
                    self.in_past -= 1;
                    let m = self.in_past as i32;
                    let one = t.value() != 0;
                    let row = self.add_row();
                    row.measurements.push(m);
                    if one {
                        row.output.phase = 2;
                    }
                }
            }
            "CX" | "XCZ" | "CY" | "YCZ" | "CZ" => {
                let (x, z) = match name {
                    "CX" | "XCZ" => (true, false),
                    "CY" | "YCZ" => (true, true),
                    _ => (false, true),
                };
                let mut k = op.targets.len();
                while k > 0 {
                    k -= 2;
                    let (t1, t2) = (op.targets[k], op.targets[k + 1]);
                    let (m1, m2) = (t1.is_record(), t2.is_record());
                    let (f1, f2) = (t1.is_qubit(), t2.is_qubit());
                    if (m1 && f2) || (m2 && f1) {
                        let q = if f1 { t1.value() } else { t2.value() } as usize;
                        let lookback = if m1 { t1.value() } else { t2.value() } as i64;
                        let t = self.in_past - lookback;
                        if t < 0 {
                            return Err(format!("Referred to measurement before start of time in {op}"));
                        }
                        for r in self.rows_anticommuting(q, x, z) {
                            let mut v = self.table[r].measurements.clone();
                            v.push(t as i32);
                            xor_sort(&mut v);
                            self.table[r].measurements = v;
                        }
                    } else if f1 && f2 {
                        let sub = Instruction { targets: vec![t1, t2], args: vec![], ..op.clone() };
                        for row in self.table.iter_mut() {
                            undo_instruction(&mut row.input, &sub)?;
                        }
                    }
                }
            }
            "MPP" => {
                let mut rev = op.clone();
                rev.targets.reverse();
                rev.args.clear();
                for mut sub in decompose_mpp(&rev, self.num_qubits)? {
                    if sub.gate.name == "M" {
                        sub.targets.reverse();
                    }
                    self.undo(&sub)?;
                }
            }
            _ if op.gate.is_unitary => {
                for row in self.table.iter_mut() {
                    undo_instruction(&mut row.input, op)?;
                }
            }
            _ => return Err(format!("Not handled by circuit flow generators method: {op}")),
        }
        Ok(())
    }

    fn elimination_step(&mut self, set: Vec<usize>, eliminated: &mut usize, available: usize) {
        let Some(&pivot) = set.iter().find(|&&p| p >= *eliminated && p < available) else { return };
        for &p in &set {
            if p != pivot {
                self.mult_row_into(pivot, p);
            }
        }
        self.table.swap(pivot, *eliminated);
        *eliminated += 1;
    }

    fn eliminate_input(&mut self, eliminated: &mut usize, available: usize) {
        for q in 0..self.num_qubits {
            let s = self.rows_with(|f| f.input.x(q));
            self.elimination_step(s, eliminated, available);
            let s = self.rows_with(|f| f.input.z(q));
            self.elimination_step(s, eliminated, available);
        }
    }

    fn eliminate_output(&mut self, eliminated: &mut usize, available: usize) {
        for q in 0..self.num_qubits {
            let s = self.rows_with(|f| f.output.x(q));
            self.elimination_step(s, eliminated, available);
            let s = self.rows_with(|f| f.output.z(q));
            self.elimination_step(s, eliminated, available);
        }
    }

    fn canonicalize_over_qubits(&mut self, available: usize) {
        let mut e = 0;
        for q in 0..self.num_qubits {
            for which in 0..4 {
                let s = self.rows_with(|f| match which {
                    0 => f.input.x(q),
                    1 => f.input.z(q),
                    2 => f.output.x(q),
                    _ => f.output.z(q),
                });
                self.elimination_step(s, &mut e, available);
            }
        }
        let mut r = 0;
        while r < self.table.len() {
            if self.table[r].input.weight() == 0 && self.table[r].output.weight() == 0 {
                let row = self.table.swap_remove(r);
                self.measurements_only.push(row);
            } else {
                r += 1;
            }
        }
    }

    fn with_circuit(c: &ir::Circuit, min_qubits: usize) -> Result<Solver, String> {
        let n = (c.count_qubits() as usize).max(min_qubits);
        let mut s = Solver::new(n, c.count_measurements())?;
        for q in 0..n {
            let x = s.add_row();
            x.input.set_x(q, true);
            x.output.set_x(q, true);
            let z = s.add_row();
            z.input.set_z(q, true);
            z.output.set_z(q, true);
        }
        let mut ops = Vec::new();
        c.for_each_operation(&mut |op| ops.push(op.clone()));
        for op in ops.iter().rev() {
            s.undo(op)?;
        }
        Ok(s)
    }
}

/// Every flow of the circuit is a product of these (Stim's `flow_generators`).
pub fn flow_generators(c: &ir::Circuit) -> Result<Vec<Flow>, String> {
    let mut s = Solver::with_circuit(c, 0)?;
    if !s.imag.is_empty() {
        return Err("Unexpected anticommutation while solving for flow generators.".into());
    }
    let only = std::mem::take(&mut s.measurements_only);
    s.table.extend(only);
    let mut e = 0;
    let n = s.table.len();
    s.eliminate_input(&mut e, n);
    s.eliminate_output(&mut e, n);
    for m in 0..s.num_measurements {
        let set = s.rows_with(|f| f.measurements.contains(&(m as i32)));
        s.elimination_step(set, &mut e, n);
    }
    for row in s.table.iter_mut() {
        row.output.phase ^= row.input.phase & 2;
        row.input.phase = 0;
        if row.input.weight() == 0 {
            row.input = PauliString::new(0);
        }
        if row.output.weight() == 0 {
            let sign = row.output.phase;
            row.output = PauliString::new(0);
            row.output.phase = sign;
        }
    }
    s.table.sort();
    Ok(s.table)
}

/// For each flow, the measurements that make it a flow of the circuit, or None.
pub fn solve_flow_measurements(c: &ir::Circuit, flows: &[Flow]) -> Result<Vec<Option<Vec<i32>>>, String> {
    let mut nq = 0;
    for f in flows {
        nq = nq.max(f.input.num_qubits()).max(f.output.num_qubits());
        if f.input.weight() == 0 && f.output.weight() == 0 {
            return Err("Given a 1 -> 1 flow (empty input, empty output). Only solving non-empty flows is supported by this method.".into());
        }
    }
    let mut s = Solver::with_circuit(c, nq)?;
    let num_circuit_flows = s.table.len();
    for f in flows {
        let n = s.num_qubits;
        let row = s.add_row();
        for q in 0..f.input.num_qubits().min(n) {
            row.input.set(q, f.input.get(q));
        }
        for q in 0..f.output.num_qubits().min(n) {
            row.output.set(q, f.output.get(q));
        }
    }
    let mut e = 0;
    s.eliminate_input(&mut e, num_circuit_flows);
    s.eliminate_output(&mut e, num_circuit_flows);
    for k in e..num_circuit_flows {
        if s.table[k].input.weight() == 0 && s.table[k].output.weight() == 0 {
            let src = s.table[k].measurements.clone();
            for k2 in num_circuit_flows..s.table.len() {
                let dst = &s.table[k2].measurements;
                if dst.len() >= src.len() * 2 {
                    continue;
                }
                let merged = xor_merge(dst, &src);
                if merged.len() < dst.len() {
                    s.table[k2].measurements = merged;
                }
            }
        }
    }
    let mut out = Vec::new();
    for k in 0..flows.len() {
        let solved = &s.table[k + num_circuit_flows];
        if s.imag.contains(&k) || solved.input.weight() != 0 || solved.output.weight() != 0 {
            out.push(None);
        } else {
            out.push(Some(solved.measurements.clone()));
        }
    }
    Ok(out)
}

fn measurement_target(m: i32, num_measurements: u64, flow: &Flow) -> Result<GateTarget, String> {
    if (m >= 0 && m as u64 >= num_measurements) || (m < 0 && (-(m as i64)) as u64 > num_measurements) {
        return Err(format!(
            "The flow '{flow}' is malformed for the given circuit. The flow mentions a measurement index '{m}', but this index out of range because the circuit only has {num_measurements} measurements."
        ));
    }
    let lookback = if m >= 0 { num_measurements as i64 - m as i64 } else { -(m as i64) };
    Ok(GateTarget::rec(lookback as u32))
}

/// Whether each flow is a flow of the circuit up to sign (Stim's
/// `check_if_circuit_has_unsigned_stabilizer_flows`).
pub fn has_unsigned_flows(c: &ir::Circuit, flows: &[Flow]) -> Result<Vec<bool>, String> {
    let mut nq = c.count_qubits() as usize;
    for f in flows {
        nq = nq.max(f.input.num_qubits()).max(f.output.num_qubits());
    }
    let nm = c.count_measurements();
    let mut rev = RevTracker::new(nq, nm, flows.len() as u64, false);
    for (fi, f) in flows.iter().enumerate() {
        for q in 0..f.output.num_qubits() {
            if f.output.x(q) {
                rev.xs[q].xor_item(fi as u64);
            }
            if f.output.z(q) {
                rev.zs[q].xor_item(fi as u64);
            }
        }
    }
    let mut obs_effects: std::collections::BTreeMap<u32, XorSet> = std::collections::BTreeMap::new();
    for (fi, f) in flows.iter().enumerate() {
        for &o in &f.observables {
            obs_effects.entry(o).or_default().xor_item(fi as u64);
        }
    }
    for (fi, f) in flows.iter().enumerate().rev() {
        let mut targets = Vec::new();
        for &m in &f.measurements {
            targets.push(measurement_target(m, nm, f)?);
        }
        let _ = fi;
        rev.undo_gate(&Instruction::new("DETECTOR", vec![], targets, "")?)?;
    }
    let mut ops = Vec::new();
    c.for_each_operation(&mut |op| ops.push(op.clone()));
    for op in ops.iter().rev() {
        match op.gate.name {
            "DETECTOR" => {}
            "OBSERVABLE_INCLUDE" => {
                let Some(effects) = obs_effects.get(&(op.args[0] as u32)).cloned() else { continue };
                for t in &op.targets {
                    if t.is_record() {
                        let index = rev.num_measurements_in_past as i64 - t.value() as i64;
                        if index < 0 {
                            return Err("Referred to a measurement result before the beginning of time.".into());
                        }
                        rev.rec_bits.entry(index as u64).or_default().xor(&effects);
                    } else if t.is_pauli() {
                        if t.0 & ir::TARGET_PAULI_X_BIT != 0 {
                            rev.xs[t.value() as usize].xor(&effects);
                        }
                        if t.0 & ir::TARGET_PAULI_Z_BIT != 0 {
                            rev.zs[t.value() as usize].xor(&effects);
                        }
                    } else {
                        return Err(format!("Unexpected target for OBSERVABLE_INCLUDE: {t}"));
                    }
                }
            }
            _ => rev.undo_gate(op)?,
        }
    }
    for (fi, f) in flows.iter().enumerate() {
        for q in 0..f.input.num_qubits() {
            if f.input.x(q) {
                rev.xs[q].xor_item(fi as u64);
            }
            if f.input.z(q) {
                rev.zs[q].xor_item(fi as u64);
            }
        }
    }
    let mut result = vec![true; flows.len()];
    for s in rev.xs.iter().chain(rev.zs.iter()) {
        for &t in s.iter() {
            result[t as usize] = false;
        }
    }
    for (d, _) in &rev.anticommutations {
        result[*d as usize] = false;
    }
    Ok(result)
}

/// Whether each flow is a flow of the circuit, signs included. Stim samples 256 random inputs;
/// here each input qubit is entangled with a reference qubit, so one run decides it exactly.
pub fn has_signed_flows(c: &ir::Circuit, flows: &[Flow]) -> Result<Vec<bool>, String> {
    let quiet = without_noise(c);
    let unsigned = has_unsigned_flows(&quiet, flows)?;
    let mut out = Vec::with_capacity(flows.len());
    for (f, ok) in flows.iter().zip(unsigned) {
        if !ok {
            out.push(false);
            continue;
        }
        let nq = (quiet.count_qubits() as usize).max(f.input.num_qubits()).max(f.output.num_qubits());
        let nm = quiet.count_measurements();
        let anc = 2 * nq;
        let mut aug = ir::Circuit::new();
        let q = |k: usize| GateTarget::qubit(k as u32, false);
        let op = |n: &str, t: Vec<GateTarget>| Instruction::new(n, vec![], t, "").expect("gate");
        for k in 0..nq {
            aug.push_op(op("H", vec![q(nq + k)]));
            aug.push_op(op("CX", vec![q(nq + k), q(k)]));
        }
        let controlled = |p: &PauliString, aug: &mut ir::Circuit| {
            for k in 0..p.num_qubits() {
                match p.x(k) as u8 + 2 * p.z(k) as u8 {
                    1 => aug.push_op(op("XCX", vec![q(k), q(anc)])),
                    2 => aug.push_op(op("CX", vec![q(k), q(anc)])),
                    3 => aug.push_op(op("YCX", vec![q(k), q(anc)])),
                    _ => {}
                }
            }
            if p.sign() {
                aug.push_op(op("X", vec![q(anc)]));
            }
        };
        controlled(&f.input, &mut aug);
        let obs: BTreeSet<u32> = f.observables.iter().copied().collect();
        aug.extend(&flow_test_block(&quiet, q(anc), &obs)?);
        for &m in &f.measurements {
            aug.push_op(op("CX", vec![measurement_target(m, nm, f)?, q(anc)]));
        }
        controlled(&f.output, &mut aug);
        let mut sim = TableauSimulator::new(anc + 1, 0x51ed);
        sim.do_circuit(&aug)?;
        out.push(sim.peek_z(anc) == 1);
    }
    Ok(out)
}

fn flow_test_block(c: &ir::Circuit, anc: GateTarget, obs: &BTreeSet<u32>) -> Result<ir::Circuit, String> {
    let mut out = ir::Circuit::new();
    for it in &c.items {
        match it {
            Item::Repeat { count, body, tag } => out.push_repeat(*count, flow_test_block(body, anc, obs)?, tag),
            Item::Op(op) if op.gate.name == "OBSERVABLE_INCLUDE" && obs.contains(&(op.args[0] as u32)) => {
                for t in &op.targets {
                    if t.is_inverted() {
                        out.push_op(Instruction::new("X", vec![], vec![anc], &op.tag)?);
                    }
                    let qt = GateTarget::qubit(t.value(), false);
                    if t.is_record() {
                        out.push_op(Instruction::new("CX", vec![], vec![*t, anc], &op.tag)?);
                    } else if t.is_x() {
                        out.push_op(Instruction::new("XCX", vec![], vec![qt, anc], &op.tag)?);
                    } else if t.is_y() {
                        out.push_op(Instruction::new("YCX", vec![], vec![qt, anc], &op.tag)?);
                    } else if t.is_z() {
                        out.push_op(Instruction::new("CX", vec![], vec![qt, anc], &op.tag)?);
                    } else {
                        return Err(format!("Not handled: {op}"));
                    }
                }
            }
            Item::Op(op) => out.push_op(op.clone()),
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fl(t: &str) -> Flow {
        Flow::from_text(t).unwrap()
    }

    #[test]
    fn text_and_products_are_stims() {
        assert_eq!(fl("X2 -> -Y2*Z4 xor rec[-1]").to_string(), "__X -> -__Y_Z xor rec[-1]");
        assert_eq!(fl("X2 -> Y2*Y2 xor rec[-2] xor rec[-2]").to_string(), "__X -> ___");
        assert_eq!(fl("X -> X").mul(&fl("Z -> Z")).unwrap().to_string(), "Y -> Y");
        assert_eq!(fl("1 -> XX").mul(&fl("1 -> ZZ")).unwrap().to_string(), "1 -> -YY");
        assert_eq!(fl("X -> rec[-1]").mul(&fl("X -> rec[-2]")).unwrap().to_string(), "_ -> rec[-2] xor rec[-1]");
        assert!(fl("1 -> X").mul(&fl("1 -> Y")).is_err());
    }

    #[test]
    fn generators_and_checks() {
        let c = ir::Circuit::parse("H 0").unwrap();
        let g: Vec<String> = flow_generators(&c).unwrap().iter().map(|f| f.to_string()).collect();
        assert_eq!(g, vec!["X -> Z", "Z -> X"]);
        let c = ir::Circuit::parse("M 0").unwrap();
        let g: Vec<String> = flow_generators(&c).unwrap().iter().map(|f| f.to_string()).collect();
        assert_eq!(g, vec!["1 -> Z xor rec[0]", "Z -> rec[0]"]);
        let c = ir::Circuit::parse("CNOT 2 4").unwrap();
        assert_eq!(has_unsigned_flows(&c, &[fl("__X__ -> __X_X"), fl("Z4 -> Z4")]).unwrap(), vec![true, false]);
        assert_eq!(has_signed_flows(&c, &[fl("X2*X4 -> X2"), fl("X2*X4 -> -X2")]).unwrap(), vec![true, false]);
    }
}

// ---------------------------------------------------------------------------------------------
// Time reversal (Stim's circuit_inverse_qec).

struct Reverser {
    num_observables: u64,
    num_qubits: usize,
    dont_turn_measurements_into_resets: bool,
    rev: RevTracker,
    num_new: u64,
    out: ir::Circuit,
    d2tag: std::collections::BTreeMap<u64, String>,
    d2coords: std::collections::BTreeMap<u64, Vec<f64>>,
    coord_shifts: Vec<f64>,
    qubit_coords: Vec<Instruction>,
    d2ms: std::collections::BTreeMap<u64, BTreeSet<u64>>,
}

use super::rev_tracker::OBSERVABLE_BIT;

/// Stim's `safe_append_reversed_targets`.
fn append_reversed(out: &mut ir::Circuit, name: &str, args: Vec<f64>, targets: &[GateTarget], tag: &str, pairs: bool) -> Result<(), String> {
    let mut t = Vec::with_capacity(targets.len());
    if pairs {
        for p in targets.chunks(2).rev() {
            t.extend_from_slice(p);
        }
    } else {
        t.extend(targets.iter().rev());
    }
    out.push_op(Instruction::new(name, args, t, tag)?);
    Ok(())
}

/// Stim's `for_each_disjoint_target_segment_in_instruction_reversed`: runs of targets, from the
/// end, in which no qubit repeats.
fn disjoint_segments_reversed(targets: &[GateTarget]) -> Vec<std::ops::Range<usize>> {
    let mut out = Vec::new();
    let mut used = BTreeSet::new();
    let (mut end, mut start) = (targets.len(), targets.len());
    while start > 0 {
        let t = targets[start - 1];
        if t.has_qubit_value() {
            if used.contains(&t.value()) {
                out.push(start..end);
                used.clear();
                end = start;
            }
            used.insert(t.value());
        }
        start -= 1;
    }
    if end > 0 {
        out.push(0..end);
    }
    out
}

impl Reverser {
    fn active_terms(&self) -> BTreeSet<u64> {
        let mut a = BTreeSet::new();
        for s in self.rev.xs.iter().chain(self.rev.zs.iter()) {
            a.extend(s.iter().copied());
        }
        for s in self.rev.rec_bits.values() {
            a.extend(s.iter().copied());
        }
        a
    }

    fn flush(&mut self) -> Result<(), String> {
        let active = self.active_terms();
        let mut erase = Vec::new();
        let entries: Vec<(u64, BTreeSet<u64>)> = self.d2ms.iter().map(|(k, v)| (*k, v.clone())).collect();
        for (d, ms) in entries {
            let (gate, args) = if d & OBSERVABLE_BIT != 0 {
                let id = d & !OBSERVABLE_BIT;
                if id >= self.num_observables {
                    continue;
                }
                ("OBSERVABLE_INCLUDE", vec![id as f64])
            } else if active.contains(&d) {
                continue;
            } else {
                ("DETECTOR", self.d2coords.get(&d).cloned().unwrap_or_default())
            };
            let targets = ms.iter().map(|&e| GateTarget::rec((self.num_new - e) as u32)).collect();
            let tag = self.d2tag.get(&d).cloned().unwrap_or_default();
            self.out.push_op(Instruction::new(gate, args, targets, &tag)?);
            erase.push(d);
        }
        for d in erase {
            self.d2coords.remove(&d);
            self.d2ms.remove(&d);
            self.d2tag.remove(&d);
        }
        Ok(())
    }

    fn simple(&mut self, op: &Instruction) -> Result<(), String> {
        self.rev.undo_gate(op)?;
        let inv = super::transform::gate_inverse_candidate(op.gate.name);
        append_reversed(&mut self.out, inv, op.args.clone(), &op.targets, &op.tag, op.has_flag(crate::gate_data::FLAG_TARGETS_PAIRS))
    }

    fn do_instruction(&mut self, op: &Instruction) -> Result<(), String> {
        let name = op.gate.name;
        match name {
            "DETECTOR" => {
                self.rev.undo_gate(op)?;
                let d = self.rev.num_detectors_in_past;
                self.d2tag.insert(d, op.tag.clone());
                let v = self.d2coords.entry(d).or_default();
                for (k, a) in op.args.iter().enumerate() {
                    v.push(a + self.coord_shifts.get(k).copied().unwrap_or(0.0));
                }
            }
            "OBSERVABLE_INCLUDE" => {
                let paulis: Vec<GateTarget> = op.targets.iter().copied().filter(|t| t.is_pauli()).collect();
                if paulis.is_empty() {
                    self.d2tag.insert(op.args[0] as u64 | OBSERVABLE_BIT, op.tag.clone());
                } else {
                    self.out.items.push(Item::Op(Instruction { targets: paulis, ..op.clone() }));
                }
                self.rev.undo_gate(op)?;
            }
            "TICK" | "I" | "II" | "I_ERROR" | "II_ERROR" | "X" | "Y" | "Z" | "C_XYZ" | "C_NXYZ" | "C_XNYZ" | "C_XYNZ" | "C_ZYX" | "C_NZYX" | "C_ZNYX" | "C_ZYNX" | "SQRT_X" | "SQRT_X_DAG" | "SQRT_Y" | "SQRT_Y_DAG" | "S" | "S_DAG" | "SQRT_XX" | "SQRT_XX_DAG" | "SQRT_YY" | "SQRT_YY_DAG" | "SQRT_ZZ" | "SQRT_ZZ_DAG" | "SPP" | "SPP_DAG" | "SWAP" | "ISWAP" | "CXSWAP" | "SWAPCX" | "CZSWAP" | "ISWAP_DAG" | "XCX" | "XCY" | "YCX" | "YCY" | "H" | "H_XY" | "H_YZ" | "H_NXZ" | "H_NXY" | "H_NYZ" | "DEPOLARIZE1" | "DEPOLARIZE2" | "X_ERROR" | "Y_ERROR" | "Z_ERROR" | "PAULI_CHANNEL_1" | "PAULI_CHANNEL_2" | "E" | "HERALDED_ERASE" | "HERALDED_PAULI_CHANNEL_1" => {
                self.simple(op)?;
            }
            "XCZ" | "YCZ" | "CX" | "CY" | "CZ" => {
                if op.targets.iter().any(|t| t.is_record()) {
                    return Err(format!("Time-reversing feedback isn't supported yet. Found feedback in: {op}"));
                }
                self.simple(op)?;
            }
            "MRX" | "MRY" | "MR" | "RX" | "RY" | "R" => {
                let inv = super::transform::gate_inverse_candidate(name);
                for seg in disjoint_segments_reversed(&op.targets) {
                    // Stim walks every target of the instruction here, for each segment.
                    for k in (0..op.targets.len()).rev() {
                        let q = op.targets[k].value() as usize;
                        let ds: Vec<u64> = self.rev.xs[q].iter().chain(self.rev.zs[q].iter()).copied().collect();
                        for d in ds {
                            self.d2ms.entry(d).or_default().insert(self.num_new);
                        }
                        self.num_new += 1;
                    }
                    let segment = Instruction { targets: op.targets[seg.clone()].to_vec(), ..op.clone() };
                    self.rev.undo_gate(&segment)?;
                    append_reversed(&mut self.out, inv, vec![], &segment.targets, &op.tag, false)?;
                    if !op.args.is_empty() {
                        let ejected = match name {
                            "MRX" | "MRY" => "Z_ERROR",
                            "MR" => "X_ERROR",
                            _ => return Err(format!("Don't know how to invert {op}")),
                        };
                        append_reversed(&mut self.out, ejected, segment.args.clone(), &segment.targets, &op.tag, false)?;
                    }
                }
                self.flush()?;
            }
            "MX" | "MY" | "M" => {
                let reset = match name {
                    "MX" => "RX",
                    "MY" => "RY",
                    _ => "R",
                };
                let inv = super::transform::gate_inverse_candidate(name);
                for k in (0..op.targets.len()).rev() {
                    let t = op.targets[k];
                    let q = t.value() as usize;
                    let last = self.rev.num_measurements_in_past - 1;
                    if !self.dont_turn_measurements_into_resets && self.rev.xs[q].is_empty() && self.rev.zs[q].is_empty() && self.rev.rec_bits.contains_key(&last) && op.args.is_empty() {
                        self.out.push_op(Instruction::new(reset, op.args.clone(), vec![t], &op.tag)?);
                    } else {
                        if let Some(f) = self.rev.rec_bits.get(&last) {
                            for &d in f.iter() {
                                self.d2ms.entry(d).or_default().insert(self.num_new);
                            }
                        }
                        self.num_new += 1;
                        self.out.push_op(Instruction::new(inv, op.args.clone(), vec![t], &op.tag)?);
                    }
                    self.rev.undo_gate(&Instruction::new(name, vec![], vec![t], &op.tag)?)?;
                }
                self.flush()?;
            }
            "MPAD" | "MPP" | "MXX" | "MYY" | "MZZ" => {
                let m = op.count_measurement_results();
                for k in 0..m {
                    if let Some(f) = self.rev.rec_bits.get(&(self.rev.num_measurements_in_past - k - 1)) {
                        for &d in f.iter() {
                            self.d2ms.entry(d).or_default().insert(self.num_new);
                        }
                    }
                    self.num_new += 1;
                }
                let inv = super::transform::gate_inverse_candidate(name);
                append_reversed(&mut self.out, inv, op.args.clone(), &op.targets, &op.tag, op.has_flag(crate::gate_data::FLAG_TARGETS_PAIRS))?;
                self.rev.undo_gate(op)?;
                self.flush()?;
            }
            "QUBIT_COORDS" => {
                let args = op.args.iter().enumerate().map(|(k, a)| a + self.coord_shifts.get(k).copied().unwrap_or(0.0)).collect();
                self.qubit_coords.push(Instruction { args, ..op.clone() });
            }
            "SHIFT_COORDS" => {
                if self.coord_shifts.len() < op.args.len() {
                    self.coord_shifts.resize(op.args.len(), 0.0);
                }
                for (k, a) in op.args.iter().enumerate() {
                    self.coord_shifts[k] += a;
                }
            }
            _ => return Err(format!("Don't know how to invert {op}")),
        }
        Ok(())
    }

    fn sensitivity(&self, d: u64) -> PauliString {
        let mut p = PauliString::new(self.num_qubits);
        for q in 0..self.num_qubits {
            p.set_x(q, self.rev.xs[q].0.contains(&d));
            p.set_z(q, self.rev.zs[q].0.contains(&d));
        }
        p
    }
}

/// The circuit run backwards, with its detectors and observables re-expressed in the reversed
/// circuit's measurements, and the given flows turned around (Stim's `time_reversed_for_flows`).
pub fn time_reversed_for_flows(c: &ir::Circuit, flows: &[Flow], dont_turn_measurements_into_resets: bool) -> Result<(ir::Circuit, Vec<Flow>), String> {
    let mut nq = c.count_qubits() as usize;
    for f in flows {
        nq = nq.max(f.input.num_qubits() + 1).max(f.output.num_qubits() + 1);
    }
    let num_measurements = c.count_measurements();
    let mut num_detectors = 0u64;
    let mut num_observables = 0u64;
    c.for_each_operation(&mut |op| match op.gate.name {
        "DETECTOR" => num_detectors += 1,
        "OBSERVABLE_INCLUDE" => num_observables = num_observables.max(op.args[0] as u64 + 1),
        _ => {}
    });
    let mut r = Reverser {
        num_observables,
        num_qubits: nq,
        dont_turn_measurements_into_resets,
        rev: RevTracker::new(nq, num_measurements, num_detectors, true),
        num_new: 0,
        out: ir::Circuit::new(),
        d2tag: Default::default(),
        d2coords: Default::default(),
        coord_shifts: Vec::new(),
        qubit_coords: Vec::new(),
        d2ms: Default::default(),
    };
    let target = |k: usize| (num_observables + k as u64) | OBSERVABLE_BIT;
    for (k, f) in flows.iter().enumerate() {
        for q in 0..f.output.num_qubits() {
            if f.output.x(q) {
                r.rev.xs[q].xor_item(target(k));
            }
            if f.output.z(q) {
                r.rev.zs[q].xor_item(target(k));
            }
        }
    }
    for (k, f) in flows.iter().enumerate() {
        for &m in &f.measurements {
            let mm = if m < 0 { m as i64 + num_measurements as i64 } else { m as i64 };
            if mm < 0 || mm as u64 >= num_measurements {
                return Err(format!("Out of range measurement in one of the flows: {f}"));
            }
            r.rev.rec_bits.entry(mm as u64).or_default().0.insert(target(k));
        }
    }
    let mut ops = Vec::new();
    c.for_each_operation(&mut |op| ops.push(op.clone()));
    for op in ops.iter().rev() {
        r.do_instruction(op)?;
    }
    for (k, f) in flows.iter().enumerate() {
        for q in 0..f.input.num_qubits() {
            if f.input.x(q) {
                r.rev.xs[q].xor_item(target(k));
            }
            if f.input.z(q) {
                r.rev.zs[q].xor_item(target(k));
            }
        }
    }
    let mut example = None;
    for q in 0..nq {
        for &e in r.rev.xs[q].iter().chain(r.rev.zs[q].iter()) {
            example = Some(e);
        }
    }
    if let Some(e) = example {
        let name = super::rev_tracker::dem_target_str(e);
        if e & OBSERVABLE_BIT == 0 || (e & !OBSERVABLE_BIT) < num_observables {
            return Err(format!(
                "The detecting region of {name} reached the start of the circuit.\nOnly flows given as arguments are permitted to touch the start or end of the circuit.\nThere are four potential ways to fix this issue, depending on what's wrong:\n- If {name} was relying on implicit initialization into |0> at the start of the circuit, add explicit resets to the circuit.\n- If {name} shouldn't be reaching the start of the circuit, fix its declaration.\n- If {name} isn't needed, delete it from the circuit.\n- If the given circuit is a partial circuit, and {name} is reaching outside of it, refactor {name}into a flow argument."
            ));
        }
        let f = &flows[((e & !OBSERVABLE_BIT) - num_observables) as usize];
        let mut v = r.sensitivity(e);
        let mut fi = f.input.clone();
        fi.ensure_num_qubits(v.num_qubits());
        for q in 0..v.num_qubits() {
            v.set_x(q, v.x(q) ^ fi.x(q));
            v.set_z(q, v.z(q) ^ fi.z(q));
        }
        let fixed = Flow { input: v, output: f.output.clone(), measurements: f.measurements.clone(), observables: Vec::new() };
        return Err(format!("The circuit didn't satisfy one of the given flows (ignoring sign): {f}\nChanging the flow to '{fixed}' would make it a valid flow."));
    }
    let mut inverted = Vec::new();
    for (k, f) in flows.iter().enumerate() {
        let mut input = f.output.clone();
        let mut output = f.input.clone();
        input.phase = 0;
        output.phase = 0;
        let ms = r.d2ms.get(&target(k)).map(|s| s.iter().map(|&m| m as i32 - r.num_new as i32).collect()).unwrap_or_default();
        inverted.push(Flow { input, output, measurements: ms, observables: Vec::new() });
    }
    let mut out = ir::Circuit::new();
    for op in r.qubit_coords.iter().rev() {
        out.items.push(Item::Op(op.clone()));
    }
    out.extend(&r.out);
    Ok((out, inverted))
}

/// Detectors the circuit could declare but doesn't (Stim's `missing_detectors`).
pub fn missing_detectors(c: &ir::Circuit, unknown_input: bool) -> Result<ir::Circuit, String> {
    let circuit = if unknown_input {
        c.clone()
    } else {
        let n = c.count_qubits();
        let mut w = ir::Circuit::new();
        w.items.push(Item::Op(Instruction::new("R", vec![], (0..n).map(|q| GateTarget::qubit(q as u32, false)).collect(), "")?));
        w.extend(c);
        w
    };
    let nm = circuit.count_measurements() as usize;
    let mut rows: Vec<(Vec<bool>, bool)> = Vec::new();
    let mut logicals: Vec<Vec<bool>> = Vec::new();
    let mut ignored = BTreeSet::new();
    let mut offset: i64 = 0;
    circuit.for_each_operation(&mut |op| {
        offset += op.count_measurement_results() as i64;
        if matches!(op.gate.name, "DETECTOR" | "OBSERVABLE_INCLUDE") {
            let row: &mut Vec<bool> = if op.gate.name == "DETECTOR" {
                rows.push((vec![false; nm], false));
                &mut rows.last_mut().unwrap().0
            } else {
                let i = op.args[0] as usize;
                while logicals.len() <= i {
                    logicals.push(vec![false; nm]);
                }
                &mut logicals[i]
            };
            for t in &op.targets {
                if t.is_record() {
                    let k = (offset - t.value() as i64) as usize;
                    row[k] ^= true;
                } else if t.is_pauli() && op.gate.name == "OBSERVABLE_INCLUDE" {
                    ignored.insert(op.args[0] as usize);
                }
            }
        }
    });
    for (k, l) in logicals.into_iter().enumerate() {
        if !ignored.contains(&k) {
            rows.push((l, false));
        }
    }
    let originals: Vec<Vec<bool>> = rows.iter().map(|r| r.0.clone()).collect();
    for g in flow_generators(&circuit)? {
        if g.input.weight() == 0 && g.output.weight() == 0 && g.observables.is_empty() {
            let mut r = vec![false; nm];
            for &e in &g.measurements {
                let k = if e < 0 { (e as i64 + nm as i64) as usize } else { e as usize };
                r[k] ^= true;
            }
            rows.push((r, true));
        }
    }
    let mut solved = 0;
    for k in 0..nm {
        let mut pivot = (solved..rows.len()).find(|&r| rows[r].0[k] && !rows[r].1);
        if pivot.is_none() {
            pivot = (solved..rows.len()).find(|&r| rows[r].0[k]);
        }
        let Some(p) = pivot else { continue };
        let pr = rows[p].0.clone();
        for (r, row) in rows.iter_mut().enumerate() {
            if row.0[k] && r != p {
                for (a, b) in row.0.iter_mut().zip(&pr) {
                    *a ^= b;
                }
            }
        }
        rows.swap(p, solved);
        solved += 1;
    }
    let mut out = ir::Circuit::new();
    for (r, generated) in rows.iter_mut() {
        if *generated && r.iter().any(|&b| b) {
            for det in &originals {
                if r.iter().zip(det).all(|(&a, &d)| !a || d) {
                    for (a, d) in r.iter_mut().zip(det) {
                        *a ^= d;
                    }
                }
            }
            let targets = (0..nm).filter(|&b| r[b]).map(|b| GateTarget::rec((nm - b) as u32)).collect();
            out.items.push(Item::Op(Instruction::new("DETECTOR", vec![], targets, "")?));
        }
    }
    Ok(out)
}

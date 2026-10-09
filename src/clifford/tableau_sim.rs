//! Stim's `TableauSimulator`: a stabilizer state kept as the inverse of the Clifford that
//! prepares it from |0…0⟩, with Stim's collapse, kickback, postselection and peeks, ported
//! from `tableau_simulator.inl`. Random outcomes come from our own generator, so they differ
//! from Stim's stream; everything deterministic is Stim's exactly.

use std::collections::{BTreeSet, HashMap};
use std::sync::OnceLock;

use super::ir::{self, GateTarget, Instruction};
use super::pauli_string::PauliString;
use super::tableau::Tableau;
use crate::gate_data;

/// A unitary gate's tableau and its inverse, cached by name.
pub fn gate_tableaus(name: &str) -> Option<&'static (Tableau, Tableau)> {
    static CACHE: OnceLock<HashMap<&'static str, (Tableau, Tableau)>> = OnceLock::new();
    let map = CACHE.get_or_init(|| {
        let mut m = HashMap::new();
        for g in gate_data::GATE_INFO {
            if let Ok(t) = Tableau::from_named_gate(g.name) {
                let inv = t.inverse(false);
                m.insert(g.name, (t, inv));
            }
        }
        m
    });
    let g = gate_data::info(name)?;
    map.get(g.name)
}

/// xorshift64*: small, fast, seedable; the stream is ours, not Stim's.
#[derive(Clone, Debug)]
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Rng {
        // SplitMix the seed so nearby seeds give unrelated streams.
        let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        Rng(if z == 0 { 0x2545_F491_4F6C_DD1D } else { z })
    }
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }
    pub fn bernoulli(&mut self, p: f64) -> bool {
        p > 0.0 && self.next_f64() < p
    }
}

#[derive(Clone, Debug)]
pub struct TableauSimulator {
    pub inv_state: Tableau,
    pub rng: Rng,
    /// 0: random outcomes; +1: always +1 (0); −1: always −1 (1).
    pub sign_bias: i8,
    pub measurement_record: Vec<bool>,
    last_correlated_error_occurred: bool,
}

fn pz(p: &PauliString) -> (Vec<u64>, bool) {
    (p.xs_words().to_vec(), p.xs_words().iter().any(|&w| w != 0))
}

impl TableauSimulator {
    pub fn new(num_qubits: usize, seed: u64) -> TableauSimulator {
        TableauSimulator { inv_state: Tableau::identity(num_qubits), rng: Rng::new(seed), sign_bias: 0, measurement_record: Vec::new(), last_correlated_error_occurred: false }
    }

    pub fn num_qubits(&self) -> usize {
        self.inv_state.num_qubits()
    }

    pub fn ensure_large_enough_for_qubits(&mut self, n: usize) {
        if n > self.num_qubits() {
            let extra = Tableau::identity(n - self.num_qubits());
            self.inv_state = self.inv_state.tensor(&extra);
        }
    }

    fn has_x(p: &PauliString) -> bool {
        pz(p).1
    }

    pub fn is_deterministic_x(&self, q: usize) -> bool {
        !Self::has_x(&self.inv_state.xs[q])
    }
    pub fn is_deterministic_y(&self, q: usize) -> bool {
        self.inv_state.xs[q].xs_words() == self.inv_state.zs[q].xs_words()
    }
    pub fn is_deterministic_z(&self, q: usize) -> bool {
        !Self::has_x(&self.inv_state.zs[q])
    }

    /// The inverse tableau's image of Y_q (Stim's `eval_y_obs`).
    fn eval_y_obs(&self, q: usize) -> PauliString {
        self.inv_state.y_output(q)
    }

    fn prepend_gate(&mut self, name: &str, targets: &[usize]) {
        let (_, inv) = gate_tableaus(name).expect("unitary gate");
        self.inv_state.prepend(inv, targets);
    }

    fn append_gate(&mut self, name: &str, targets: &[usize]) {
        let (t, _) = gate_tableaus(name).expect("unitary gate");
        self.inv_state.append(t, targets);
    }

    fn read_record(&self, t: GateTarget) -> bool {
        if t.is_sweep() {
            return false;
        }
        let k = t.value() as usize;
        let n = self.measurement_record.len();
        k <= n && self.measurement_record[n - k]
    }

    /// Run one instruction.
    pub fn do_gate(&mut self, inst: &Instruction) -> Result<(), String> {
        let name = inst.gate.name;
        match name {
            "DETECTOR" | "OBSERVABLE_INCLUDE" | "TICK" | "QUBIT_COORDS" | "SHIFT_COORDS" | "REPEAT" | "I" | "II" | "I_ERROR" | "II_ERROR" => {}
            "M" => self.do_mz(inst),
            "MX" => self.do_mx(inst),
            "MY" => self.do_my(inst),
            "MR" => self.do_mr(inst, 3),
            "MRX" => self.do_mr(inst, 1),
            "MRY" => self.do_mr(inst, 2),
            "R" => self.do_reset(inst, 3),
            "RX" => self.do_reset(inst, 1),
            "RY" => self.do_reset(inst, 2),
            "MPP" => self.do_mpp(inst)?,
            "SPP" | "SPP_DAG" => self.do_spp(inst)?,
            "MXX" | "MYY" | "MZZ" => self.do_pair_measure(inst),
            "MPAD" => {
                for t in &inst.targets {
                    self.measurement_record.push(t.value() != 0);
                }
                self.noisify(&inst.args, inst.targets.len());
            }
            "CX" | "CY" | "CZ" | "XCZ" | "YCZ" => {
                for pair in inst.targets.chunks(2) {
                    let (a, b) = (pair[0], pair[1]);
                    // Normalise to (control, target) with a Z control.
                    let (c, t, pauli) = match name {
                        "CX" => (a, b, 1u8),
                        "CY" => (a, b, 2),
                        "CZ" => (a, b, 3),
                        "XCZ" => (b, a, 1),
                        _ => (b, a, 2),
                    };
                    let cbit = c.is_classical_bit();
                    let tbit = t.is_classical_bit();
                    if !cbit && !tbit {
                        let g = match pauli {
                            1 => "CX",
                            2 => "CY",
                            _ => "CZ",
                        };
                        self.prepend_gate(g, &[c.value() as usize, t.value() as usize]);
                    } else if pauli == 3 && cbit && tbit {
                    } else if pauli == 3 && tbit {
                        if self.read_record(t) {
                            self.prepend_gate("Z", &[c.value() as usize]);
                        }
                    } else if tbit {
                        return Err("Measurement record editing is not supported.".into());
                    } else if self.read_record(c) {
                        let g = ["I", "X", "Y", "Z"][pauli as usize];
                        self.prepend_gate(g, &[t.value() as usize]);
                    }
                }
            }
            "DEPOLARIZE1" => {
                for t in &inst.targets {
                    if self.rng.bernoulli(inst.args[0]) {
                        let p = 1 + self.rng.next_u64() % 3;
                        self.pauli_on(t.value() as usize, p & 1 != 0, p & 2 != 0);
                    }
                }
            }
            "DEPOLARIZE2" => {
                for pair in inst.targets.chunks(2) {
                    if self.rng.bernoulli(inst.args[0]) {
                        let p = 1 + self.rng.next_u64() % 15;
                        self.pauli_on(pair[0].value() as usize, p & 1 != 0, p & 2 != 0);
                        self.pauli_on(pair[1].value() as usize, p & 4 != 0, p & 8 != 0);
                    }
                }
            }
            "X_ERROR" | "Y_ERROR" | "Z_ERROR" => {
                for t in &inst.targets {
                    if self.rng.bernoulli(inst.args[0]) {
                        let q = t.value() as usize;
                        match name {
                            "X_ERROR" => self.inv_state.zs[q].phase ^= 2,
                            "Z_ERROR" => self.inv_state.xs[q].phase ^= 2,
                            _ => {
                                self.inv_state.xs[q].phase ^= 2;
                                self.inv_state.zs[q].phase ^= 2;
                            }
                        }
                    }
                }
            }
            "PAULI_CHANNEL_1" | "PAULI_CHANNEL_2" => {
                let width = if name == "PAULI_CHANNEL_1" { 1 } else { 2 };
                let saved = self.last_correlated_error_occurred;
                for group in inst.targets.chunks(width) {
                    self.last_correlated_error_occurred = false;
                    let mut used = 0.0;
                    for pauli in 1..(1usize << (2 * width)) {
                        let p = inst.args[pauli - 1];
                        if p == 0.0 {
                            continue;
                        }
                        let remaining = 1.0 - used;
                        let cond = if remaining <= 0.0 { 0.0 } else if remaining <= p { 1.0 } else { p / remaining };
                        used += p;
                        let mut targets = Vec::new();
                        for q in 0..width {
                            let z = (pauli >> (2 * (width - q - 1))) & 2 != 0;
                            let y = (pauli >> (2 * (width - q - 1))) & 1 != 0;
                            targets.push(GateTarget::pauli_xz(group[q].value(), z ^ y, z, false));
                        }
                        self.else_correlated(cond, &targets);
                    }
                }
                self.last_correlated_error_occurred = saved;
            }
            "E" => {
                self.last_correlated_error_occurred = false;
                self.else_correlated(inst.args[0], &inst.targets);
            }
            "ELSE_CORRELATED_ERROR" => self.else_correlated(inst.args[0], &inst.targets),
            "HERALDED_ERASE" => {
                for t in &inst.targets {
                    let fired = self.rng.bernoulli(inst.args[0]);
                    if fired {
                        let r = self.rng.next_u64();
                        let q = t.value() as usize;
                        if r & 1 != 0 {
                            self.inv_state.xs[q].phase ^= 2;
                        }
                        if r & 2 != 0 {
                            self.inv_state.zs[q].phase ^= 2;
                        }
                    }
                    self.measurement_record.push(fired);
                }
            }
            "HERALDED_PAULI_CHANNEL_1" => {
                let (hi, hx, hy, hz) = (inst.args[0], inst.args[1], inst.args[2], inst.args[3]);
                let ht = (hi + hx + hy + hz).min(1.0);
                let cond = if ht != 0.0 { [hx / ht, hy / ht, hz / ht] } else { [hx, hy, hz] };
                for t in &inst.targets {
                    let fired = self.rng.bernoulli(ht);
                    self.measurement_record.push(fired);
                    if fired {
                        let pc = Instruction::new("PAULI_CHANNEL_1", cond.to_vec(), vec![*t], "")?;
                        self.do_gate(&pc)?;
                    }
                }
            }
            _ if inst.gate.is_unitary => {
                let n = if inst.has_flag(gate_data::FLAG_TARGETS_PAIRS) { 2 } else { 1 };
                for group in inst.targets.chunks(n) {
                    let q: Vec<usize> = group.iter().map(|t| t.value() as usize).collect();
                    self.prepend_gate(name, &q);
                }
            }
            _ => return Err(format!("Not implemented by TableauSimulator::do_gate: {name}")),
        }
        Ok(())
    }

    fn pauli_on(&mut self, q: usize, x: bool, z: bool) {
        // Stim flips the X image's sign for a Z-type component and the Z image's for X.
        if x {
            self.inv_state.zs[q].phase ^= 2;
        }
        if z {
            self.inv_state.xs[q].phase ^= 2;
        }
    }

    fn else_correlated(&mut self, p: f64, targets: &[GateTarget]) {
        if self.last_correlated_error_occurred {
            return;
        }
        self.last_correlated_error_occurred = self.rng.bernoulli(p);
        if !self.last_correlated_error_occurred {
            return;
        }
        for t in targets {
            let q = t.value() as usize;
            if t.0 & ir::TARGET_PAULI_X_BIT != 0 {
                self.prepend_gate("X", &[q]);
            }
            if t.0 & ir::TARGET_PAULI_Z_BIT != 0 {
                self.prepend_gate("Z", &[q]);
            }
        }
    }

    fn noisify(&mut self, args: &[f64], n: usize) {
        if args.is_empty() {
            return;
        }
        let last = self.measurement_record.len();
        for k in 0..n {
            if self.rng.bernoulli(args[0]) {
                let i = last - 1 - k;
                self.measurement_record[i] ^= true;
            }
        }
    }

    /// Collapse qubit `target`'s Z observable (Stim's `collapse_qubit_z`); returns the pivot, or
    /// None when the outcome was already deterministic.
    fn collapse_qubit_z(&mut self, target: usize) -> Option<usize> {
        let n = self.num_qubits();
        let zrow = self.inv_state.zs[target].clone();
        let pivot = (0..n).find(|&q| zrow.x(q))?;
        for k in pivot + 1..n {
            if zrow.x(k) {
                self.append_gate("CX", &[pivot, k]);
            }
        }
        if self.inv_state.zs[target].z(pivot) {
            self.append_gate("H_YZ", &[pivot]);
        } else {
            self.append_gate("H", &[pivot]);
        }
        let result = if self.sign_bias == 0 { self.rng.next_u64() & 1 == 1 } else { self.sign_bias < 0 };
        if self.inv_state.zs[target].sign() != result {
            self.append_gate("X", &[pivot]);
        }
        Some(pivot)
    }

    fn collapse_isolate_qubit_z(&mut self, target: usize) {
        let n = self.num_qubits();
        self.collapse_qubit_z(target);
        // In the transposed view, zs.zt[q][target] is inv_state.zs[target].z(q).
        let q = (0..n).find(|&q| self.inv_state.zs[target].z(q)).expect("Z image has a Z");
        if q != target {
            self.append_gate("SWAP", &[q, target]);
        }
        for q in 0..n {
            if q != target && self.inv_state.zs[target].z(q) {
                self.append_gate("CX", &[q, target]);
            }
        }
        if self.inv_state.xs[target].z(target) {
            self.append_gate("S", &[target]);
        }
        for q in 0..n {
            if q != target {
                let x = &self.inv_state.xs[target];
                let p = x.x(q) as u8 + 2 * x.z(q) as u8;
                match p {
                    1 => self.append_gate("CX", &[target, q]),
                    2 => self.append_gate("CZ", &[target, q]),
                    3 => self.append_gate("CY", &[target, q]),
                    _ => {}
                }
            }
        }
    }

    fn unique_nondeterministic(&self, targets: &[GateTarget], stride: usize, basis: u8) -> Vec<usize> {
        let mut set = BTreeSet::new();
        let mut ordered = Vec::new();
        for t in targets.iter().step_by(stride) {
            let q = t.value() as usize;
            let det = match basis {
                1 => self.is_deterministic_x(q),
                2 => self.is_deterministic_y(q),
                _ => self.is_deterministic_z(q),
            };
            if !det {
                if basis == 3 {
                    ordered.push(q);
                } else {
                    set.insert(q);
                }
            }
        }
        if basis == 3 {
            ordered
        } else {
            set.into_iter().collect()
        }
    }

    pub fn collapse(&mut self, targets: &[GateTarget], stride: usize, basis: u8) {
        let qs = self.unique_nondeterministic(targets, stride, basis);
        if qs.is_empty() {
            return;
        }
        let change = match basis {
            1 => Some("H"),
            2 => Some("H_YZ"),
            _ => None,
        };
        if let Some(g) = change {
            for &q in &qs {
                self.prepend_gate(g, &[q]);
            }
        }
        for &q in &qs {
            self.collapse_qubit_z(q);
        }
        if let Some(g) = change {
            for &q in &qs {
                self.prepend_gate(g, &[q]);
            }
        }
    }

    fn sign_of(&self, q: usize, basis: u8) -> bool {
        match basis {
            1 => self.inv_state.xs[q].sign(),
            2 => self.eval_y_obs(q).sign(),
            _ => self.inv_state.zs[q].sign(),
        }
    }

    fn do_mz(&mut self, inst: &Instruction) {
        self.do_measure(inst, 3);
    }
    fn do_mx(&mut self, inst: &Instruction) {
        self.do_measure(inst, 1);
    }
    fn do_my(&mut self, inst: &Instruction) {
        self.do_measure(inst, 2);
    }

    fn do_measure(&mut self, inst: &Instruction, basis: u8) {
        self.collapse(&inst.targets, 1, basis);
        for t in &inst.targets {
            let b = self.sign_of(t.value() as usize, basis) ^ t.is_inverted();
            self.measurement_record.push(b);
        }
        self.noisify(&inst.args, inst.targets.len());
    }

    fn do_mr(&mut self, inst: &Instruction, basis: u8) {
        self.collapse(&inst.targets, 1, basis);
        for t in &inst.targets {
            let q = t.value() as usize;
            let s = self.sign_of(q, basis);
            self.measurement_record.push(s ^ t.is_inverted());
            if basis == 2 {
                if s {
                    self.inv_state.zs[q].phase ^= 2;
                }
            } else {
                self.inv_state.xs[q].phase &= !2;
                self.inv_state.zs[q].phase &= !2;
            }
        }
        self.noisify(&inst.args, inst.targets.len());
    }

    fn do_reset(&mut self, inst: &Instruction, basis: u8) {
        self.collapse(&inst.targets, 1, basis);
        for t in &inst.targets {
            let q = t.value() as usize;
            self.inv_state.xs[q].phase &= !2;
            self.inv_state.zs[q].phase &= !2;
            if basis == 2 && self.eval_y_obs(q).sign() {
                self.inv_state.zs[q].phase ^= 2;
            }
        }
    }

    fn do_pair_measure(&mut self, inst: &Instruction) {
        let (pre, basis) = match inst.gate.name {
            "MXX" => ("CX", 1u8),
            "MYY" => ("CY", 2),
            _ => ("XCZ", 3),
        };
        for seg in disjoint_pair_segments(&inst.targets) {
            let pairs: Vec<GateTarget> = seg.iter().map(|t| GateTarget::qubit(t.value(), false)).collect();
            let conj = Instruction::new(pre, vec![], pairs, "").unwrap();
            let _ = self.do_gate(&conj);
            self.collapse(&seg, 2, basis);
            for pair in seg.chunks(2) {
                let q = pair[0].value() as usize;
                let flipped = pair[0].is_inverted() ^ pair[1].is_inverted();
                let b = self.sign_of(q, basis) ^ flipped;
                self.measurement_record.push(b);
            }
            self.noisify(&inst.args, seg.len() / 2);
            let _ = self.do_gate(&conj);
        }
    }

    fn do_mpp(&mut self, inst: &Instruction) -> Result<(), String> {
        let n = self.num_qubits();
        for sub in decompose_mpp(inst, n)? {
            self.do_gate(&sub)?;
        }
        Ok(())
    }

    fn do_spp(&mut self, inst: &Instruction) -> Result<(), String> {
        for sub in decompose_spp(inst)? {
            self.do_gate(&sub)?;
        }
        Ok(())
    }

    /// Measure a Pauli string (with a flip probability), as Stim's `measure_pauli_string`.
    pub fn measure_pauli_string(&mut self, p: &PauliString, flip_probability: f64) -> Result<bool, String> {
        if !(0.0..=1.0).contains(&flip_probability) {
            return Err("Need 0 <= flip_probability <= 1".into());
        }
        self.ensure_large_enough_for_qubits(p.num_qubits());
        let mut targets = Vec::new();
        for k in 0..p.num_qubits() {
            if p.x(k) || p.z(k) {
                targets.push(GateTarget::pauli_xz(k as u32, p.x(k), p.z(k), false));
                targets.push(GateTarget::combiner());
            }
        }
        let mut prob = flip_probability;
        if p.sign() {
            prob = 1.0 - prob;
        }
        if targets.is_empty() {
            let b = self.rng.bernoulli(prob);
            self.measurement_record.push(b);
        } else {
            targets.pop();
            let inst = Instruction::new("MPP", vec![prob], targets, "")?;
            self.do_mpp(&inst)?;
        }
        Ok(*self.measurement_record.last().unwrap())
    }

    pub fn peek_x(&self, q: usize) -> i8 {
        let x = &self.inv_state.xs[q];
        if Self::has_x(x) {
            0
        } else if x.sign() {
            -1
        } else {
            1
        }
    }
    pub fn peek_y(&self, q: usize) -> i8 {
        let y = self.eval_y_obs(q);
        if Self::has_x(&y) {
            0
        } else if y.sign() {
            -1
        } else {
            1
        }
    }
    pub fn peek_z(&self, q: usize) -> i8 {
        let z = &self.inv_state.zs[q];
        if Self::has_x(z) {
            0
        } else if z.sign() {
            -1
        } else {
            1
        }
    }

    /// The qubit's state as a single-qubit Pauli: +X, -Z, ... or + identity when entangled.
    pub fn peek_bloch(&self, q: usize) -> PauliString {
        let (x, z) = (&self.inv_state.xs[q], &self.inv_state.zs[q]);
        let mut r = PauliString::new(1);
        if !Self::has_x(x) {
            r.phase = x.phase & 2;
            r.set_x(0, true);
        } else if !Self::has_x(z) {
            r.phase = z.phase & 2;
            r.set_z(0, true);
        } else if x.xs_words() == z.xs_words() {
            let y = self.eval_y_obs(q);
            r.phase = y.phase & 2;
            r.set_x(0, true);
            r.set_z(0, true);
        }
        r
    }

    pub fn canonical_stabilizers(&self) -> Vec<PauliString> {
        self.inv_state.inverse(false).stabilizers(true)
    }

    pub fn current_inverse_tableau(&self) -> &Tableau {
        &self.inv_state
    }

    /// The expectation (+1, −1 or 0) of a Pauli observable, without changing the state.
    pub fn peek_observable_expectation(&self, obs: &PauliString) -> i8 {
        let mut state = self.clone();
        let n = state.num_qubits().max(obs.num_qubits());
        state.ensure_large_enough_for_qubits(n + 1);
        let anc = n;
        if obs.sign() {
            state.prepend_gate("X", &[anc]);
        }
        for q in 0..obs.num_qubits() {
            let p = obs.x(q) as u8 + 2 * obs.z(q) as u8;
            let g = match p {
                1 => "XCX",
                3 => "YCX",
                2 => "CX",
                _ => continue,
            };
            state.prepend_gate(g, &[q, anc]);
        }
        if !state.is_deterministic_z(anc) {
            return 0;
        }
        if state.inv_state.zs[anc].sign() {
            -1
        } else {
            1
        }
    }

    fn try_isolate_observable_to_qubit_z(&mut self, obs: &PauliString, undo: bool) -> Option<usize> {
        let mut pivot: Option<usize> = None;
        for q in 0..obs.num_qubits() {
            let p = obs.x(q) as u8 + 2 * obs.z(q) as u8;
            if p == 0 {
                continue;
            }
            match pivot {
                None => {
                    pivot = Some(q);
                    if !undo {
                        if p == 1 {
                            self.inv_state.prepend(&gate_tableaus("H").unwrap().0, &[q]);
                        } else if p == 3 {
                            self.inv_state.prepend(&gate_tableaus("H_YZ").unwrap().0, &[q]);
                        }
                        if obs.sign() {
                            self.inv_state.prepend(&gate_tableaus("X").unwrap().0, &[q]);
                        }
                    }
                }
                Some(pv) => {
                    // Stim's p numbering here is x + 2z: 1 X, 2 Z, 3 Y.
                    let g = match p {
                        1 => "XCX",
                        2 => "XCZ",
                        _ => "XCY",
                    };
                    self.inv_state.prepend(&gate_tableaus(g).unwrap().0, &[pv, q]);
                }
            }
        }
        if undo {
            if let Some(pv) = pivot {
                let p = obs.x(pv) as u8 + 2 * obs.z(pv) as u8;
                if obs.sign() {
                    self.inv_state.prepend(&gate_tableaus("X").unwrap().0, &[pv]);
                }
                if p == 1 {
                    self.inv_state.prepend(&gate_tableaus("H").unwrap().0, &[pv]);
                } else if p == 3 {
                    self.inv_state.prepend(&gate_tableaus("H_YZ").unwrap().0, &[pv]);
                }
            }
        }
        pivot
    }

    /// Force qubits into a basis state (`basis` 1 X, 2 Y, 3 Z; `desired` true for −1).
    pub fn postselect(&mut self, qubits: &[usize], desired: bool, basis: u8) -> Result<(), String> {
        let (change, names) = match basis {
            1 => ("H", ("+", "-")),
            2 => ("H_YZ", ("i", "-i")),
            _ => ("I", ("0", "1")),
        };
        let unique: Vec<usize> = qubits.iter().copied().collect::<BTreeSet<_>>().into_iter().collect();
        for &q in &unique {
            self.prepend_gate(change, &[q]);
        }
        let old_bias = self.sign_bias;
        self.sign_bias = if desired { -1 } else { 1 };
        let mut finished = 0;
        while finished < qubits.len() {
            let q = qubits[finished];
            self.collapse_qubit_z(q);
            if self.inv_state.zs[q].sign() != desired {
                break;
            }
            finished += 1;
        }
        self.sign_bias = old_bias;
        for &q in &unique {
            self.prepend_gate(change, &[q]);
        }
        if finished < qubits.len() {
            let (want, other) = if desired { (names.1, names.0) } else { (names.0, names.1) };
            let mut msg = format!("The requested postselection was impossible.\nDesired state: |{want}>\nQubit {} is in the perpendicular state |{other}>\n", qubits[finished]);
            if finished > 0 {
                msg.push_str(&format!("{finished} of the requested postselections were finished ("));
                for q in &qubits[..finished] {
                    msg.push_str(&format!("qubit {q}, "));
                }
                msg.push_str("[failed here])\n");
            }
            return Err(msg);
        }
        Ok(())
    }

    pub fn postselect_observable(&mut self, obs: &PauliString, desired: bool) -> Result<(), String> {
        self.ensure_large_enough_for_qubits(obs.num_qubits());
        let pivot = self.try_isolate_observable_to_qubit_z(obs, false);
        let mut expected: i8 = match pivot {
            Some(p) => self.peek_z(p),
            None => {
                if obs.sign() {
                    -1
                } else {
                    1
                }
            }
        };
        if desired {
            expected = -expected;
        }
        if expected != -1 {
            if let Some(p) = pivot {
                self.postselect(&[p], desired, 3)?;
            }
        }
        self.try_isolate_observable_to_qubit_z(obs, true);
        if expected == -1 {
            return Err(format!(
                "It's impossible to postselect into the {} eigenstate of {} because the system is deterministically in the {} eigenstate.",
                if desired { "-1" } else { "+1" },
                obs,
                if desired { "+1" } else { "-1" }
            ));
        }
        Ok(())
    }

    /// Measure Z of `q` and return the outcome and the kickback (Stim's `measure_kickback_z`).
    pub fn measure_kickback(&mut self, q: usize, basis: u8, inverted: bool) -> (bool, Option<PauliString>) {
        let change = match basis {
            1 => Some("H"),
            2 => Some("H_YZ"),
            _ => None,
        };
        if let Some(g) = change {
            self.prepend_gate(g, &[q]);
        }
        let has_kickback = !self.is_deterministic_z(q);
        let mut kickback = None;
        if has_kickback {
            let pivot = self.collapse_qubit_z(q).unwrap();
            kickback = Some(self.inv_state.inverse_output(1, pivot, true));
        }
        let result = self.inv_state.zs[q].sign() ^ inverted;
        self.measurement_record.push(result);
        self.collapse_isolate_qubit_z(q);
        if let Some(g) = change {
            self.prepend_gate(g, &[q]);
        }
        if let Some(k) = kickback.as_mut() {
            match basis {
                1 => {
                    let (x, z) = (k.x(q), k.z(q));
                    k.set_x(q, z);
                    k.set_z(q, x);
                }
                2 => {
                    let v = k.x(q) ^ k.z(q);
                    k.set_x(q, v);
                }
                _ => {}
            }
        }
        (result, kickback)
    }

    /// Shrink or grow the qubit count; qubits removed are collapsed and decoupled first.
    pub fn set_num_qubits(&mut self, n: usize) {
        if n >= self.num_qubits() {
            self.ensure_large_enough_for_qubits(n);
            return;
        }
        for q in n..self.num_qubits() {
            self.collapse_isolate_qubit_z(q);
        }
        let old = &self.inv_state;
        let mut t = Tableau::identity(n);
        for q in 0..n {
            let cut = |p: &PauliString| PauliString::from_fn(n, p.phase & 2, |k| p.get(k));
            t.xs[q] = cut(&old.xs[q]);
            t.zs[q] = cut(&old.zs[q]);
        }
        self.inv_state = t;
    }

    /// Run every instruction of a circuit `reps` times.
    pub fn do_circuit(&mut self, c: &ir::Circuit) -> Result<(), String> {
        self.ensure_large_enough_for_qubits(c.count_qubits() as usize);
        let mut err = Ok(());
        c.for_each_operation(&mut |op| {
            if err.is_ok() {
                err = self.do_gate(op);
            }
        });
        err
    }
}

/// Split a pair instruction into runs whose qubits don't repeat (Stim's
/// `decompose_pair_instruction_into_disjoint_segments`).
pub fn disjoint_pair_segments(targets: &[GateTarget]) -> Vec<Vec<GateTarget>> {
    let mut out = Vec::new();
    let mut used = BTreeSet::new();
    let mut start = 0;
    let mut cur = 0;
    while cur < targets.len() {
        let (q0, q1) = (targets[cur].value(), targets[cur + 1].value());
        if used.contains(&q0) || used.contains(&q1) {
            out.push(targets[start..cur].to_vec());
            used.clear();
            start = cur;
        }
        used.insert(q0);
        used.insert(q1);
        cur += 2;
    }
    if start < targets.len() {
        out.push(targets[start..].to_vec());
    }
    out
}

/// The Pauli products of an MPP/SPP instruction, each with its classical-bit targets (Stim's
/// `accumulate_next_obs_terms_to_pauli_string_helper`, terms multiplied on the left).
pub fn product_terms(inst: &Instruction, n: usize, allow_imaginary: bool) -> Result<Vec<(PauliString, Vec<GateTarget>)>, String> {
    let mut out = Vec::new();
    let t = &inst.targets;
    let mut start = 0;
    while start < t.len() {
        let mut end = start + 1;
        while end < t.len() && t[end].is_combiner() {
            end += 2;
        }
        let mut obs = PauliString::new(n);
        let mut bits = Vec::new();
        for k in (start..end).step_by(2) {
            let g = t[k];
            if g.is_pauli() {
                let mut single = PauliString::new(n.max(g.value() as usize + 1));
                single.set(g.value() as usize, g.pauli_index());
                if g.is_inverted() {
                    single.phase = 2;
                }
                obs.ensure_num_qubits(single.num_qubits());
                single.ensure_num_qubits(obs.num_qubits());
                obs = single.mul(&obs);
            } else if g.is_classical_bit() {
                bits.push(g);
            } else {
                return Err(format!("Found an unsupported target `{g}` in {inst}"));
            }
        }
        if obs.is_imaginary() && !allow_imaginary {
            return Err(format!("Acted on an anti-Hermitian operator (e.g. X0*Z0 instead of Y0) in {inst}"));
        }
        out.push((obs, bits));
        start = end;
    }
    Ok(out)
}

/// MPP as H, H_YZ, CX, M and MPAD (Stim's `decompose_mpp_operation`).
pub fn decompose_mpp(inst: &Instruction, n: usize) -> Result<Vec<Instruction>, String> {
    let mut out = Vec::new();
    let (mut h_xz, mut h_yz, mut cnot, mut meas) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let mut merged = BTreeSet::new();
    let tag = &inst.tag;
    let mk = |name: &str, args: Vec<f64>, t: Vec<GateTarget>| Instruction::new(name, args, t, tag).unwrap();
    let flush = |out: &mut Vec<Instruction>, h_xz: &mut Vec<GateTarget>, h_yz: &mut Vec<GateTarget>, cnot: &mut Vec<GateTarget>, meas: &mut Vec<GateTarget>, merged: &mut BTreeSet<usize>| {
        if meas.is_empty() {
            return;
        }
        let pre: Vec<Instruction> = [("H", h_xz.clone()), ("H_YZ", h_yz.clone()), ("CX", cnot.clone())].into_iter().filter(|(_, t)| !t.is_empty()).map(|(g, t)| mk(g, vec![], t)).collect();
        out.extend(pre.iter().cloned());
        out.push(mk("M", inst.args.clone(), meas.clone()));
        out.extend(pre.into_iter().rev());
        h_xz.clear();
        h_yz.clear();
        cnot.clear();
        meas.clear();
        merged.clear();
    };
    for (cur, _) in product_terms(inst, n, false)? {
        if cur.weight() == 0 {
            flush(&mut out, &mut h_xz, &mut h_yz, &mut cnot, &mut meas, &mut merged);
            out.push(mk("MPAD", inst.args.clone(), vec![GateTarget::qubit(cur.sign() as u32, false)]));
            continue;
        }
        let support: Vec<usize> = (0..cur.num_qubits()).filter(|&q| cur.x(q) || cur.z(q)).collect();
        if support.iter().any(|q| merged.contains(q)) {
            flush(&mut out, &mut h_xz, &mut h_yz, &mut cnot, &mut meas, &mut merged);
        }
        merged.extend(support.iter().copied());
        let mut first = true;
        for &q in &support {
            let (x, z) = (cur.x(q), cur.z(q));
            if x {
                if z {
                    h_yz.push(GateTarget::qubit(q as u32, false));
                } else {
                    h_xz.push(GateTarget::qubit(q as u32, false));
                }
            }
            if first {
                meas.push(GateTarget::qubit(q as u32, cur.sign()));
                first = false;
            } else {
                cnot.push(GateTarget::qubit(q as u32, false));
                cnot.push(GateTarget::qubit(meas.last().unwrap().value(), false));
            }
        }
    }
    flush(&mut out, &mut h_xz, &mut h_yz, &mut cnot, &mut meas, &mut merged);
    Ok(out)
}

/// SPP / SPP_DAG as H, H_YZ, CX and S / S_DAG (Stim's `decompose_spp_or_spp_dag_operation`).
pub fn decompose_spp(inst: &Instruction) -> Result<Vec<Instruction>, String> {
    let invert = inst.gate.name == "SPP_DAG";
    let n = inst.targets.iter().filter(|t| t.has_qubit_value()).map(|t| t.value() as usize + 1).max().unwrap_or(0);
    let mut out = Vec::new();
    let tag = &inst.tag;
    for (obs, bits) in product_terms(inst, n, false)? {
        let (mut h_xz, mut h_yz, mut cnot) = (Vec::new(), Vec::new(), Vec::new());
        let mut focus: Option<u32> = None;
        for q in 0..obs.num_qubits() {
            let (x, z) = (obs.x(q), obs.z(q));
            if !x && !z {
                continue;
            }
            if x {
                if z {
                    h_yz.push(GateTarget::qubit(q as u32, false));
                } else {
                    h_xz.push(GateTarget::qubit(q as u32, false));
                }
            }
            match focus {
                None => focus = Some(q as u32),
                Some(f) => {
                    cnot.push(GateTarget::qubit(q as u32, false));
                    cnot.push(GateTarget::qubit(f, false));
                }
            }
        }
        let Some(f) = focus else { continue };
        for b in &bits {
            cnot.push(*b);
            cnot.push(GateTarget::qubit(f, false));
        }
        let sign = invert ^ obs.sign();
        let pre: Vec<Instruction> = [("H", h_xz), ("H_YZ", h_yz), ("CX", cnot)].into_iter().filter(|(_, t)| !t.is_empty()).map(|(g, t)| Instruction::new(g, vec![], t, tag).unwrap()).collect();
        out.extend(pre.iter().cloned());
        out.push(Instruction::new(if sign { "S_DAG" } else { "S" }, vec![], vec![GateTarget::qubit(f, false)], tag).unwrap());
        out.extend(pre.into_iter().rev());
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(text: &str) -> TableauSimulator {
        let c = ir::Circuit::parse(text).unwrap();
        let mut s = TableauSimulator::new(c.count_qubits() as usize, 5);
        s.do_circuit(&c).unwrap();
        s
    }

    #[test]
    fn bell_pairs_agree_and_peeks_work() {
        for seed in 0..20 {
            let c = ir::Circuit::parse("H 0\nCX 0 1\nM 0 1").unwrap();
            let mut s = TableauSimulator::new(2, seed);
            s.do_circuit(&c).unwrap();
            assert_eq!(s.measurement_record[0], s.measurement_record[1]);
        }
        let s = run("H 0");
        assert_eq!((s.peek_x(0), s.peek_z(0)), (1, 0));
        assert_eq!(s.peek_bloch(0).to_string(), "+X");
        let s = run("X 0\nH 1\nS 1");
        assert_eq!(s.peek_z(0), -1);
        assert_eq!(s.peek_y(1), 1);
        let s = run("H 0\nCX 0 1");
        let mut zz = PauliString::from_text("ZZ").unwrap();
        assert_eq!(s.peek_observable_expectation(&zz), 1);
        zz.phase = 2;
        assert_eq!(s.peek_observable_expectation(&zz), -1);
        let st: Vec<String> = s.canonical_stabilizers().iter().map(|p| p.to_string()).collect();
        assert_eq!(st, vec!["+XX", "+ZZ"]);
    }

    #[test]
    fn mpp_and_pair_measurements_are_consistent() {
        for seed in 0..20 {
            let c = ir::Circuit::parse("H 0\nCX 0 1\nMPP X0*X1 Z0*Z1 !Y0*Y1\nMXX 0 1\nMZZ 0 1").unwrap();
            let mut s = TableauSimulator::new(2, seed);
            s.do_circuit(&c).unwrap();
            assert_eq!(s.measurement_record, vec![false, false, false, false, false]);
        }
    }

    #[test]
    fn postselection() {
        let mut s = TableauSimulator::new(1, 1);
        s.do_gate(&Instruction::new("H", vec![], vec![GateTarget::qubit(0, false)], "").unwrap()).unwrap();
        s.postselect(&[0], true, 3).unwrap();
        assert_eq!(s.peek_z(0), -1);
        assert!(s.postselect(&[0], false, 3).is_err());
    }
}

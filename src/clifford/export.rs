//! Stim's exports of a circuit: OpenQASM 2 and 3 (`export_qasm.cc`), a Quirk URL
//! (`export_quirk_url.cc`), and its detecting regions (`circuit_to_detecting_regions.cc`). Each
//! writes what Stim writes, character for character, quirks included.

use std::collections::{BTreeMap, BTreeSet};

use super::ir::{self, GateTarget, Instruction};
use super::rev_tracker::{RevTracker, OBSERVABLE_BIT};
use super::tableau_sim::{decompose_mpp, decompose_spp};
use crate::gate_data::{self, FLAG_IS_RESET, FLAG_IS_SINGLE_QUBIT_GATE, FLAG_IS_UNITARY, FLAG_PRODUCES_RESULTS, FLAG_TARGETS_COMBINERS, FLAG_TARGETS_PAIRS};

/// Counts Stim's `CircuitStats` keeps that the exports use.
struct Stats {
    num_qubits: u64,
    num_measurements: u64,
    num_detectors: u64,
    num_observables: u64,
    num_sweep_bits: u64,
    num_ticks: u64,
}

fn stats(c: &ir::Circuit) -> Stats {
    let (mut dets, mut obs, mut sweeps) = (0u64, 0u64, 0u64);
    c.for_each_operation(&mut |op| {
        match op.gate.name {
            "DETECTOR" => dets += 1,
            "OBSERVABLE_INCLUDE" => obs = obs.max(op.args.first().map(|&a| a as u64 + 1).unwrap_or(0)),
            _ => {}
        }
        for t in &op.targets {
            if t.is_sweep() {
                sweeps = sweeps.max(t.value() as u64 + 1);
            }
        }
    });
    Stats { num_qubits: c.count_qubits(), num_measurements: c.count_measurements(), num_detectors: dets, num_observables: obs, num_sweep_bits: sweeps, num_ticks: c.count_ticks() }
}

// ---------------------------------------------------------------------------------------------
// OpenQASM

/// The single-qubit gates QASM defines through `U`, with the angles Stim's `to_euler_angles`
/// rounds to (its output for each gate, which is what the export writes).
const QASM_EULER: &[(&str, &str, &str)] = &[
    ("C_XYZ", "cxyz", "pi/2, 0, pi/2"),
    ("C_ZYX", "czyx", "pi/2, pi/2, pi/2"),
    ("C_NXYZ", "cnxyz", "pi/2, pi/2, pi/2"),
    ("C_XNYZ", "cxnyz", "pi/2, 0, -pi/2"),
    ("C_XYNZ", "cxynz", "pi/2, pi/2, -pi/2"),
    ("C_NZYX", "cnzyx", "pi/2, -pi/2, 0"),
    ("C_ZNYX", "cznyx", "pi/2, -pi/2, pi/2"),
    ("C_ZYNX", "czynx", "pi/2, pi/2, 0"),
    ("H_XY", "hxy", "pi/2, 0, pi/2"),
    ("H_YZ", "hyz", "pi/2, pi/2, pi/2"),
    ("H_NXY", "hnxy", "pi/2, 0, -pi/2"),
    ("H_NXZ", "hnxz", "pi/2, pi/2, 0"),
    ("H_NYZ", "hnyz", "pi/2, -pi/2, -pi/2"),
    ("SQRT_Y", "sy", "pi/2, 0, 0"),
    ("SQRT_Y_DAG", "sydg", "pi/2, pi/2, pi/2"),
];

/// The gates QASM defines from their H/S/CX/M/R decomposition, in Stim's order.
const QASM_DECOMPOSED: &[(&str, &str)] = &[
    ("CXSWAP", "cxswap"),
    ("CZSWAP", "czswap"),
    ("ISWAP", "iswap"),
    ("ISWAP_DAG", "iswapdg"),
    ("SQRT_XX", "sxx"),
    ("SQRT_XX_DAG", "sxxdg"),
    ("SQRT_YY", "syy"),
    ("SQRT_YY_DAG", "syydg"),
    ("SQRT_ZZ", "szz"),
    ("SQRT_ZZ_DAG", "szzdg"),
    ("SWAPCX", "swapcx"),
    ("XCX", "xcx"),
    ("XCY", "xcy"),
    ("XCZ", "xcz"),
    ("YCX", "ycx"),
    ("YCY", "ycy"),
    ("YCZ", "ycz"),
    ("MR", "mr"),
    ("MRX", "mrx"),
    ("MRY", "mry"),
    ("MX", "mx"),
    ("MXX", "mxx"),
    ("MY", "my"),
    ("MYY", "myy"),
    ("MZZ", "mzz"),
    ("RX", "rx"),
    ("RY", "ry"),
];

const QASM_STANDARD: &[(&str, &str)] = &[("I", "id"), ("X", "x"), ("Y", "y"), ("Z", "z"), ("SQRT_X", "sx"), ("SQRT_X_DAG", "sxdg"), ("S", "s"), ("S_DAG", "sdg"), ("CX", "cx"), ("CY", "cy"), ("CZ", "cz"), ("SWAP", "swap"), ("H", "h")];

const NOISE: &[&str] = &["DEPOLARIZE1", "DEPOLARIZE2", "X_ERROR", "Y_ERROR", "Z_ERROR", "PAULI_CHANNEL_1", "PAULI_CHANNEL_2", "E", "ELSE_CORRELATED_ERROR", "HERALDED_ERASE", "HERALDED_PAULI_CHANNEL_1"];

struct Qasm {
    out: String,
    version: u8,
    skip_dets_and_obs: bool,
    stats: Stats,
    reference: Vec<bool>,
    measurement_offset: u64,
    detector_offset: u64,
    names: BTreeMap<&'static str, &'static str>,
    used: BTreeSet<&'static str>,
}

impl Qasm {
    fn measurement(&mut self, invert: bool, q: &str, m: &str) {
        if invert {
            if self.version == 3 {
                self.out.push_str(&format!("measure {q} -> {m};{m} = {m} ^ 1;"));
            } else {
                self.out.push_str(&format!("x {q};measure {q} -> {m};x {q};"));
            }
        } else {
            self.out.push_str(&format!("measure {q} -> {m};"));
        }
    }

    fn decomposed_operation(&mut self, invert: bool, gate: &str, q0: &str, q1: &str, m: &str) -> Result<(), String> {
        let info = gate_data::info(gate).unwrap();
        let c = ir::Circuit::parse(info.decomposition)?;
        let q2n = |t: GateTarget| if t.value() == 0 { q0 } else { q1 };
        let mut first = true;
        for it in &c.items {
            let ir::Item::Op(op) = it else { continue };
            let step = if op.gate.name == "CX" { 2 } else { 1 };
            for p in op.targets.chunks(step) {
                if !first {
                    self.out.push(' ');
                }
                first = false;
                match op.gate.name {
                    "S" => self.out.push_str(&format!("s {};", q2n(p[0]))),
                    "H" => self.out.push_str(&format!("h {};", q2n(p[0]))),
                    "R" => self.out.push_str(&format!("reset {};", q2n(p[0]))),
                    "CX" => self.out.push_str(&format!("cx {}, {};", q2n(p[0]), q2n(p[1]))),
                    "M" => self.measurement(invert, q2n(p[0]), m),
                    _ => return Err(format!("Unhandled: {op}")),
                }
            }
        }
        Ok(())
    }

    fn define_single(&mut self, gate: &'static str, name: &'static str, angles: &str) {
        self.names.insert(gate, name);
        if self.used.contains(gate) {
            self.out.push_str(&format!("gate {name} q0 {{ U({angles}) q0; }}\n"));
        }
    }

    fn define_decomposed(&mut self, gate: &'static str, name: &'static str) -> Result<(), String> {
        self.names.insert(gate, name);
        if !self.used.contains(gate) {
            return Ok(());
        }
        let info = gate_data::info(gate).unwrap();
        let c = ir::Circuit::parse(info.decomposition)?;
        let mut is_unitary = true;
        c.for_each_operation(&mut |op| is_unitary &= op.has_flag(FLAG_IS_UNITARY));
        let pairs = info.flags & FLAG_TARGETS_PAIRS != 0;
        let num_measurements = c.count_measurements();
        if is_unitary {
            self.out.push_str(&format!("gate {name} q0{} {{ ", if pairs { ", q1" } else { "" }));
        } else {
            if self.version == 2 {
                return Ok(());
            }
            self.out.push_str(&format!("def {name}(qubit q0{})", if pairs { ", qubit q1" } else { "" }));
            if num_measurements > 1 {
                return Err("Multiple measurement gates not supported.".into());
            }
            self.out.push_str(if num_measurements == 1 { " -> bit { bit b; " } else { " { " });
        }
        self.decomposed_operation(false, gate, "q0", "q1", "b")?;
        if num_measurements > 0 {
            self.out.push_str(" return b;");
        }
        self.out.push_str(" }\n");
        Ok(())
    }

    fn declarations(&mut self) -> Result<(), String> {
        self.out.push_str(if self.version == 2 { "include \"qelib1.inc\";\n" } else { "include \"stdgates.inc\";\n" });
        for &(g, n) in QASM_STANDARD {
            self.names.insert(g, n);
        }
        for &(g, n, a) in QASM_EULER {
            self.define_single(g, n, a);
        }
        for &(g, n) in QASM_DECOMPOSED {
            self.define_decomposed(g, n)?;
        }
        self.out.push('\n');
        Ok(())
    }

    fn storage(&mut self) {
        let s = &self.stats;
        let mut o = String::new();
        if s.num_qubits > 0 {
            o.push_str(&format!("qreg q[{}];\n", s.num_qubits));
        }
        if s.num_measurements > 0 {
            o.push_str(&format!("creg rec[{}];\n", s.num_measurements));
        }
        if s.num_detectors > 0 && !self.skip_dets_and_obs {
            o.push_str(&format!("creg dets[{}];\n", s.num_detectors));
        }
        if s.num_observables > 0 && !self.skip_dets_and_obs {
            o.push_str(&format!("creg obs[{}];\n", s.num_observables));
        }
        if s.num_sweep_bits > 0 {
            o.push_str(&format!("creg sweep[{}];\n", s.num_sweep_bits));
        }
        o.push('\n');
        self.out.push_str(&o);
    }

    fn name(&self, gate: &str) -> &'static str {
        self.names.get(gate).copied().unwrap_or("")
    }

    fn decomposable(&mut self, inst: &Instruction, inline: bool) -> Result<(), String> {
        let f = inst.gate.flags;
        let step = if f & FLAG_TARGETS_PAIRS != 0 { 2 } else { 1 };
        let mut m = String::new();
        for p in inst.targets.chunks(step) {
            let (t0, t1) = (p[0], p[step - 1]);
            let mut invert = t0.is_inverted();
            if step == 2 {
                invert ^= t1.is_inverted();
            }
            if inline {
                let (q1, q2) = (format!("q[{}]", t0.value()), format!("q[{}]", t1.value()));
                if f & FLAG_PRODUCES_RESULTS != 0 {
                    m = format!("rec[{}]", self.measurement_offset);
                    self.measurement_offset += 1;
                }
                self.decomposed_operation(invert, inst.gate.name, &q1, &q2, &m)?;
                self.out.push_str(&format!(" // decomposed {}\n", inst.gate.name));
            } else {
                if f & FLAG_PRODUCES_RESULTS != 0 {
                    self.out.push_str(&format!("rec[{}] = ", self.measurement_offset));
                    self.measurement_offset += 1;
                }
                self.out.push_str(&format!("{}(q[{}]", self.name(inst.gate.name), t0.value()));
                if step == 2 {
                    self.out.push_str(&format!(", q[{}]", t1.value()));
                }
                self.out.push(')');
                if f & FLAG_PRODUCES_RESULTS != 0 && invert {
                    self.out.push_str(" ^ 1");
                }
                self.out.push_str(";\n");
            }
        }
        Ok(())
    }

    fn two_qubit(&mut self, inst: &Instruction) -> Result<(), String> {
        for p in inst.targets.chunks(2) {
            let (t1, t2) = (p[0], p[1]);
            if t1.is_qubit() && t2.is_qubit() {
                self.out.push_str(&format!("{} q[{}], q[{}];\n", self.name(inst.gate.name), t1.value(), t2.value()));
            } else if t1.is_qubit() || t2.is_qubit() {
                let (basis, control, target) = match inst.gate.name {
                    "CX" => ('X', t1, t2),
                    "CY" => ('Y', t1, t2),
                    "CZ" => {
                        if t1.is_qubit() {
                            ('Z', t2, t1)
                        } else {
                            ('Z', t1, t2)
                        }
                    }
                    "XCZ" => ('X', t2, t1),
                    "YCZ" => ('Y', t2, t1),
                    _ => return Err(format!("Not implemented in output_two_qubit_unitary_instruction_with_possible_feedback: {inst}")),
                };
                self.out.push_str("if (");
                if control.is_record() {
                    if self.version == 2 {
                        return Err("The circuit contains feedback, but OPENQASM 2 doesn't support feedback.\nYou can use `stim.Circuit.with_inlined_feedback` to drop feedback operations.\nAlternatively, pass the argument `open_qasm_version=3`.".into());
                    }
                    self.out.push_str(&format!("ms[{}]", self.measurement_offset.wrapping_sub(control.value() as u64)));
                } else if control.is_sweep() {
                    if self.version == 2 {
                        return Err("The circuit contains sweep operation, but OPENQASM 2 doesn't support feedback.\nRemove these operations, or pass the argument `open_qasm_version=3`.".into());
                    }
                    self.out.push_str(&format!("sweep[{}]", control.value()));
                } else {
                    return Err(format!("Not implemented in output_two_qubit_unitary_instruction_with_possible_feedback: {inst}"));
                }
                self.out.push_str(&format!(") {{\n    {basis} q[{}];\n}}\n", target.value()));
            }
        }
        Ok(())
    }

    fn instruction(&mut self, inst: &Instruction) -> Result<(), String> {
        let f = inst.gate.flags;
        let name = inst.gate.name;
        match name {
            "QUBIT_COORDS" | "SHIFT_COORDS" | "II" | "I_ERROR" | "II_ERROR" => return Ok(()),
            "MPAD" => {
                for t in &inst.targets {
                    if self.version == 3 {
                        self.out.push_str(&format!("rec[{}] = {};\n", self.measurement_offset, t.value()));
                    } else if t.value() != 0 {
                        return Err("The circuit contains a vacuous measurement with a non-zero result (like MPAD 1 or MPP !X1*X1) but OPENQASM 2 doesn't support classical assignment.\nPass the argument `open_qasm_version=3` to fix this.".into());
                    }
                    self.measurement_offset += 1;
                }
                return Ok(());
            }
            "TICK" => {
                self.out.push_str("barrier q;\n\n");
                return Ok(());
            }
            "M" => {
                for t in &inst.targets {
                    let (q, m) = (t.value(), self.measurement_offset);
                    if t.is_inverted() {
                        if self.version == 3 {
                            self.out.push_str(&format!("measure q[{q}] -> rec[{m}];rec[{m}] = rec[{m}] ^ 1;"));
                        } else {
                            self.out.push_str(&format!("x q[{q}];measure q[{q}] -> rec[{m}];x q[{q}];"));
                        }
                    } else {
                        self.out.push_str(&format!("measure q[{q}] -> rec[{m}];"));
                    }
                    self.out.push('\n');
                    self.measurement_offset += 1;
                }
                return Ok(());
            }
            "R" => {
                for t in &inst.targets {
                    self.out.push_str(&format!("reset q[{}];\n", t.value()));
                }
                return Ok(());
            }
            "DETECTOR" | "OBSERVABLE_INCLUDE" => {
                if self.skip_dets_and_obs {
                    return Ok(());
                }
                if self.version == 2 {
                    return Err("The circuit contains detectors or observables, but OPENQASM 2 doesn't support the operations needed for accumulating detector and observable values.\nTo simply ignore detectors and observables, pass the argument `skip_dets_and_obs=True`.\nAlternatively, pass the argument `open_qasm_version=3`.".into());
                }
                if name == "DETECTOR" {
                    self.out.push_str(&format!("dets[{}] = ", self.detector_offset));
                    self.detector_offset += 1;
                } else {
                    let k = inst.args[0] as i32;
                    self.out.push_str(&format!("obs[{k}] = obs[{k}] ^ "));
                }
                let mut reference = false;
                let mut had_paulis = false;
                for t in &inst.targets {
                    if t.is_record() {
                        let i = self.measurement_offset.wrapping_sub(t.value() as u64);
                        reference ^= self.reference.get(i as usize).copied().unwrap_or(false);
                        self.out.push_str(&format!("rec[{i}] ^ "));
                    } else if t.is_pauli() {
                        had_paulis = true;
                    } else {
                        return Err(format!("Unexpected target for OBSERVABLE_INCLUDE: {t}"));
                    }
                }
                self.out.push_str(&format!("{};\n", reference as u8));
                if had_paulis {
                    self.out.push_str(&format!("// Warning: ignored pauli terms in {inst}\n"));
                }
                return Ok(());
            }
            n if NOISE.contains(&n) => {
                return Err("The circuit contains noise, but OPENQASM 2 doesn't support noise operations.\nUse `stim.Circuit.without_noise` to get a version of the circuit without noise.".into());
            }
            "MPP" => {
                self.out.push_str(&format!("// --- begin decomposed {inst}\n"));
                for op in decompose_mpp(inst, self.stats.num_qubits as usize)? {
                    self.instruction(&op)?;
                }
                self.out.push_str("// --- end decomposed MPP\n");
                return Ok(());
            }
            "SPP" | "SPP_DAG" => {
                self.out.push_str(&format!("// --- begin decomposed {inst}\n"));
                for op in decompose_spp(inst)? {
                    self.instruction(&op)?;
                }
                self.out.push_str("// --- end decomposed SPP\n");
                return Ok(());
            }
            _ => {}
        }
        if f & (FLAG_IS_RESET | FLAG_PRODUCES_RESULTS) != 0 {
            return self.decomposable(inst, self.version == 2);
        }
        if f & FLAG_IS_UNITARY != 0 {
            if f & FLAG_IS_SINGLE_QUBIT_GATE != 0 {
                for t in &inst.targets {
                    self.out.push_str(&format!("{} q[{}];\n", self.name(name), t.value()));
                }
                return Ok(());
            }
            if f & FLAG_TARGETS_PAIRS != 0 {
                return self.two_qubit(inst);
            }
        }
        Err(format!("Not implemented in QasmExporter::output_instruction: {inst}"))
    }
}

fn collect_used(c: &ir::Circuit, used: &mut BTreeSet<&'static str>) {
    for it in &c.items {
        match it {
            ir::Item::Op(op) => {
                used.insert(op.gate.name);
            }
            ir::Item::Repeat { body, .. } => {
                used.insert("REPEAT");
                collect_used(body, used);
            }
        }
    }
}

/// The circuit as OpenQASM 2 or 3 (Stim's `Circuit.to_qasm`).
pub fn to_qasm(c: &ir::Circuit, version: i64, skip_dets_and_obs: bool) -> Result<String, String> {
    if version != 2 && version != 3 {
        return Err("Only open_qasm_version=2 and open_qasm_version=3 are supported.".into());
    }
    let st = stats(c);
    let reference = if st.num_detectors > 0 || st.num_observables > 0 { super::transform::reference_sample(c)? } else { vec![false; st.num_measurements as usize] };
    let mut used = BTreeSet::new();
    collect_used(c, &mut used);
    let mut q = Qasm { out: String::new(), version: version as u8, skip_dets_and_obs, stats: st, reference, measurement_offset: 0, detector_offset: 0, names: BTreeMap::new(), used };
    q.out.push_str(if version == 2 { "OPENQASM 2.0;\n" } else { "OPENQASM 3.0;\n" });
    q.declarations()?;
    q.storage();
    let mut result = Ok(());
    c.for_each_operation(&mut |op| {
        if result.is_ok() {
            result = q.instruction(op);
        }
    });
    result?;
    Ok(q.out)
}

// ---------------------------------------------------------------------------------------------
// Quirk

/// Quirk's definitions of the gates it lacks, in Stim's gate order.
const QUIRK_CUSTOM: &[(&str, &str)] = &[
    ("H_XY", r#"{"id":"~Hxy","name":"Hxy","matrix":"{{0,-√½-√½i},{√½-√½i,0}}"}"#),
    ("H_YZ", r#"{"id":"~Hyz","name":"Hyz","matrix":"{{-√½i,-√½},{√½,√½i}}"}"#),
    ("H_NXY", r#"{"id":"~Hnxy","name":"Hnxy","matrix":"{{0,√½+√½i},{√½-√½i,0}}"}"#),
    ("H_NXZ", r#"{"id":"~Hnxz","name":"Hnxz","matrix":"{{-√½,√½},{√½,√½}}"}"#),
    ("H_NYZ", r#"{"id":"~Hnyz","name":"Hnyz","matrix":"{{-√½,-√½i},{√½i,√½}}"}"#),
    ("C_XYZ", r#"{"id":"~Cxyz","name":"Cxyz","matrix":"{{½-½i,-½-½i},{½-½i,½+½i}}"}"#),
    ("C_ZYX", r#"{"id":"~Czyx","name":"Czyx","matrix":"{{½+½i,½+½i},{-½+½i,½-½i}}"}"#),
    ("C_NXYZ", r#"{"id":"~Cnxyz","name":"Cnxyz","matrix":"{{½+½i,½-½i},{-½-½i,½-½i}}"}"#),
    ("C_XNYZ", r#"{"id":"~Cxnyz","name":"Cxnyz","matrix":"{{½+½i,-½+½i},{½+½i,½-½i}}"}"#),
    ("C_XYNZ", r#"{"id":"~Cxynz","name":"Cxynz","matrix":"{{½-½i,½+½i},{-½+½i,½+½i}}"}"#),
    ("C_NZYX", r#"{"id":"~Cnzyx","name":"Cnzyx","matrix":"{{½+½i,-½-½i},{½-½i,½-½i}}"}"#),
    ("C_ZNYX", r#"{"id":"~Cznyx","name":"Cznyx","matrix":"{{½-½i,½-½i},{-½-½i,½+½i}}"}"#),
    ("C_ZYNX", r#"{"id":"~Czynx","name":"Czynx","matrix":"{{½-½i,-½+½i},{½+½i,½+½i}}"}"#),
];

fn quirk_name(g: &str) -> &'static str {
    match g {
        "H" => "H",
        "H_XY" => "~Hxy",
        "H_YZ" => "~Hyz",
        "H_NXY" => "~Hnxy",
        "H_NXZ" => "~Hnxz",
        "H_NYZ" => "~Hnyz",
        "I" => "…",
        "X" => "X",
        "Y" => "Y",
        "Z" => "Z",
        "C_XYZ" => "~Cxyz",
        "C_NXYZ" => "~Cnxyz",
        "C_XNYZ" => "~Cxnyz",
        "C_XYNZ" => "~Cxynz",
        "C_ZYX" => "~Czyx",
        "C_NZYX" => "~Cnzyx",
        "C_ZNYX" => "~Cznyx",
        "C_ZYNX" => "~Czynx",
        "SQRT_X" => "X^½",
        "SQRT_X_DAG" => "X^-½",
        "SQRT_Y" => "Y^½",
        "SQRT_Y_DAG" => "Y^-½",
        "S" => "Z^½",
        "S_DAG" => "Z^-½",
        "MX" => "XDetector",
        "MY" => "YDetector",
        "M" => "ZDetector",
        "MRX" | "RX" => "XDetectControlReset",
        "MRY" | "RY" => "YDetectControlReset",
        "MR" | "R" => "ZDetectControlReset",
        _ => "",
    }
}

fn quirk_control_target(g: &str) -> (&'static str, &'static str) {
    let (x, y, z) = ("⊖", "(/)", "•");
    match g {
        "XCX" => (x, "X"),
        "XCY" => (x, "Y"),
        "XCZ" => (x, "Z"),
        "YCX" => (y, "X"),
        "YCY" => (y, "Y"),
        "YCZ" => (y, "Z"),
        "CX" => (z, "X"),
        "CY" => (z, "Y"),
        "CZ" => (z, "Z"),
        "SWAPCX" => (z, "X"),
        "CXSWAP" => (x, "Z"),
        "CZSWAP" => (z, "Z"),
        "ISWAP" | "ISWAP_DAG" | "SQRT_ZZ" | "SQRT_ZZ_DAG" => ("zpar", "zpar"),
        "SQRT_XX" | "SQRT_XX_DAG" | "MXX" | "MYY" | "MZZ" => ("xpar", "xpar"),
        "SQRT_YY" | "SQRT_YY_DAG" => ("ypar", "ypar"),
        _ => ("", ""),
    }
}

fn quirk_phase(g: &str) -> &'static str {
    match g {
        "SQRT_XX" | "SQRT_YY" | "SQRT_ZZ" | "SPP" | "ISWAP" => "i",
        "SQRT_XX_DAG" | "SQRT_YY_DAG" | "SQRT_ZZ_DAG" | "SPP_DAG" | "ISWAP_DAG" => "-i",
        _ => "",
    }
}

struct Quirk {
    num_qubits: usize,
    col_offset: usize,
    used: BTreeSet<&'static str>,
    cols: BTreeMap<usize, BTreeMap<usize, String>>,
}

impl Quirk {
    fn set(&mut self, col: usize, q: usize, v: &str) {
        self.cols.entry(col).or_default().insert(q, v.to_string());
    }

    fn has(&mut self, col: usize, q: usize) -> bool {
        self.cols.entry(col).or_default().contains_key(&q)
    }

    fn pauli_par_controls(&mut self, g: &str, col: usize, targets: &[GateTarget]) {
        for t in targets {
            if t.has_qubit_value() {
                let x = t.0 & ir::TARGET_PAULI_X_BIT != 0;
                let z = t.0 & ir::TARGET_PAULI_Z_BIT != 0;
                let p = x as usize + 2 * z as usize;
                let v = if p == 0 { quirk_control_target(g).0 } else { ["", "xpar", "zpar", "ypar"][p] };
                self.set(col, t.value() as usize, v);
            }
        }
    }

    fn pick_free_qubit(&self, targets: &[GateTarget]) -> usize {
        if self.num_qubits <= 16 {
            return self.num_qubits;
        }
        let qs: BTreeSet<usize> = targets.iter().filter(|t| t.has_qubit_value()).map(|t| t.value() as usize).collect();
        let mut q = 0;
        while qs.contains(&q) {
            q += 1;
        }
        q
    }

    fn pick_merge_qubit(&self, targets: &[GateTarget]) -> usize {
        if self.num_qubits <= 16 {
            return self.num_qubits;
        }
        // Stim tests `pauli_type()`, which is never zero ('I' for a plain qubit), so any qubit counts.
        for t in targets {
            if t.has_qubit_value() && t.value() <= 16 {
                return t.value() as usize;
            }
        }
        self.num_qubits
    }

    fn single(&mut self, g: &str, t: GateTarget) {
        if !t.has_qubit_value() {
            return;
        }
        let q = t.value() as usize;
        let c = self.col_offset;
        if self.has(c, q) || self.has(c + 1, q) || self.has(c + 2, q) {
            self.col_offset += 3;
        }
        let c = self.col_offset;
        let n = quirk_name(g);
        match n {
            "XDetectControlReset" => {
                self.set(c, q, n);
                self.set(c + 1, q, "H");
            }
            "YDetectControlReset" => {
                self.set(c, q, n);
                self.set(c + 1, q, "~Hyz");
                self.used.insert("H_YZ");
            }
            "ZDetectControlReset" => self.set(c, q, n),
            _ => self.set(c + 1, q, n),
        }
    }

    fn multi_phase(&mut self, g: &str, group: &[GateTarget]) {
        self.col_offset += 3;
        let q = self.pick_free_qubit(group);
        let c = self.col_offset;
        self.pauli_par_controls(g, c, group);
        self.set(c, q, quirk_phase(g));
        self.col_offset += 3;
    }

    fn multi_measure(&mut self, g: &str, group: &[GateTarget]) {
        self.col_offset += 3;
        let q = self.pick_merge_qubit(group);
        let c = self.col_offset;
        self.pauli_par_controls(g, c, group);
        if q == self.num_qubits {
            self.set(c, q, "X");
            self.set(c + 1, q, "ZDetectControlReset");
        } else {
            self.pauli_par_controls(g, c + 2, group);
            let r = self.cols.entry(c).or_default().entry(q).or_default().clone();
            let (a, b) = match r.as_str() {
                "xpar" => ("Z", "XDetector"),
                "ypar" => ("X", "YDetector"),
                _ => ("X", "ZDetector"),
            };
            self.set(c, q, a);
            self.set(c + 1, q, b);
            self.set(c + 2, q, a);
        }
        self.col_offset += 3;
    }

    fn controlled(&mut self, g: &str, t1: GateTarget, t2: GateTarget) {
        if t1.has_qubit_value() && t2.has_qubit_value() {
            self.col_offset += 3;
            let (c, t) = quirk_control_target(g);
            let col = self.col_offset;
            self.set(col, t1.value() as usize, c);
            self.set(col, t2.value() as usize, t);
            self.col_offset += 3;
        }
    }

    fn swap_plus(&mut self, g: &str, t1: GateTarget, t2: GateTarget) {
        if t1.has_qubit_value() && t2.has_qubit_value() {
            self.col_offset += 3;
            let col = self.col_offset;
            self.set(col, t1.value() as usize, "Swap");
            self.set(col, t2.value() as usize, "Swap");
            if g == "ISWAP" || g == "ISWAP_DAG" {
                self.multi_phase(g, &[t1, t2]);
            } else {
                self.controlled(g, t1, t2);
            }
            self.col_offset += 3;
        }
    }

    fn group(&mut self, g: &'static str, group: &[GateTarget], full: &Instruction) -> Result<(), String> {
        match g {
            "DETECTOR" | "OBSERVABLE_INCLUDE" | "QUBIT_COORDS" | "SHIFT_COORDS" | "MPAD" | "DEPOLARIZE1" | "DEPOLARIZE2" | "X_ERROR" | "Y_ERROR" | "Z_ERROR" | "PAULI_CHANNEL_1" | "PAULI_CHANNEL_2" | "E" | "ELSE_CORRELATED_ERROR" | "HERALDED_ERASE" | "HERALDED_PAULI_CHANNEL_1" | "II" | "I_ERROR" | "II_ERROR" => {}
            "TICK" => self.col_offset += 3,
            "MX" | "MY" | "M" | "MRX" | "MRY" | "MR" | "RX" | "RY" | "R" | "H" | "H_XY" | "H_YZ" | "H_NXY" | "H_NXZ" | "H_NYZ" | "I" | "X" | "Y" | "Z" | "C_XYZ" | "C_NXYZ" | "C_XNYZ" | "C_XYNZ" | "C_ZYX" | "C_NZYX" | "C_ZNYX" | "C_ZYNX" | "SQRT_X" | "SQRT_X_DAG" | "SQRT_Y" | "SQRT_Y_DAG" | "S" | "S_DAG" => self.single(g, group[0]),
            "SQRT_XX" | "SQRT_YY" | "SQRT_ZZ" | "SQRT_XX_DAG" | "SQRT_YY_DAG" | "SQRT_ZZ_DAG" | "SPP" | "SPP_DAG" => self.multi_phase(g, group),
            "XCX" | "XCY" | "XCZ" | "YCX" | "YCY" | "YCZ" | "CX" | "CY" | "CZ" => self.controlled(g, group[0], group[1]),
            "SWAP" | "ISWAP" | "CXSWAP" | "SWAPCX" | "CZSWAP" | "ISWAP_DAG" => self.swap_plus(g, group[0], group[1]),
            "MXX" | "MYY" | "MZZ" | "MPP" => self.multi_measure(g, group),
            _ => return Err(format!("Not supported in export_quirk_url: {full}")),
        }
        Ok(())
    }
}

/// An instruction's targets as Stim's `for_each_combined_targets_group` splits them: each
/// product of targets joined by combiners.
pub fn combined_groups(targets: &[GateTarget]) -> Vec<Vec<GateTarget>> {
    let mut out: Vec<Vec<GateTarget>> = Vec::new();
    let mut k = 0;
    while k < targets.len() {
        let mut g = vec![targets[k]];
        k += 1;
        while k + 1 < targets.len() && targets[k].is_combiner() {
            g.push(targets[k + 1]);
            k += 2;
        }
        out.push(g);
    }
    out
}

/// A URL that opens the circuit in Quirk (Stim's `Circuit.to_quirk_url`).
pub fn to_quirk_url(c: &ir::Circuit) -> Result<String, String> {
    let mut e = Quirk { num_qubits: c.count_qubits() as usize, col_offset: 0, used: BTreeSet::new(), cols: BTreeMap::new() };
    let mut result = Ok(());
    c.for_each_operation(&mut |op| {
        if result.is_err() {
            return;
        }
        e.used.insert(op.gate.name);
        let f = op.gate.flags;
        let groups: Vec<Vec<GateTarget>> = if f & FLAG_TARGETS_COMBINERS != 0 {
            combined_groups(&op.targets)
        } else if f & FLAG_TARGETS_PAIRS != 0 {
            op.targets.chunks(2).map(|p| p.to_vec()).collect()
        } else if f & FLAG_IS_SINGLE_QUBIT_GATE != 0 {
            op.targets.iter().map(|t| vec![*t]).collect()
        } else {
            vec![op.targets.clone()]
        };
        for g in groups {
            if let Err(x) = e.group(op.gate.name, &g, op) {
                result = Err(x);
                return;
            }
        }
    });
    result?;
    e.col_offset += 3;
    let mut out = String::from(r#"https://algassert.com/quirk#circuit={"cols":["#);
    let mut has_col = false;
    for k in 0..e.col_offset {
        let Some(col) = e.cols.get(&k) else { continue };
        if col.is_empty() {
            continue;
        }
        let mut entries: Vec<&str> = Vec::new();
        for (&q, v) in col {
            while entries.len() <= q {
                entries.push("");
            }
            entries[q] = v;
        }
        if has_col {
            out.push(',');
        }
        has_col = true;
        out.push('[');
        for (q, v) in entries.iter().enumerate() {
            if q > 0 {
                out.push(',');
            }
            if v.is_empty() {
                out.push('1');
            } else {
                out.push('"');
                out.push_str(v);
                out.push('"');
            }
        }
        out.push(']');
    }
    out.push(']');
    let mut has_custom = false;
    for &(g, def) in QUIRK_CUSTOM {
        if e.used.contains(g) {
            out.push_str(if has_custom { "," } else { r#","gates":["# });
            has_custom = true;
            out.push_str(def);
        }
    }
    if has_custom {
        out.push(']');
    }
    out.push('}');
    Ok(out)
}

// ---------------------------------------------------------------------------------------------
// Detecting regions

/// For each wanted detector or observable (rev_tracker's encoding), at each wanted tick, the
/// qubits where it is sensitive to X errors and those where it is sensitive to Z errors (Stim's
/// `circuit_to_detecting_regions`).
#[allow(clippy::type_complexity)]
pub fn detecting_regions(c: &ir::Circuit, targets: &BTreeSet<u64>, ticks: &BTreeSet<u64>, ignore_anticommutation_errors: bool) -> Result<BTreeMap<u64, BTreeMap<u64, (Vec<bool>, Vec<bool>)>>, String> {
    let st = stats(c);
    let n = st.num_qubits as usize;
    let mut tick = st.num_ticks;
    let mut tracker = RevTracker::new(n, st.num_measurements, st.num_detectors, !ignore_anticommutation_errors);
    let mut out: BTreeMap<u64, BTreeMap<u64, (Vec<bool>, Vec<bool>)>> = BTreeMap::new();
    let mut ops: Vec<Instruction> = Vec::new();
    c.for_each_operation(&mut |op| ops.push(op.clone()));
    for op in ops.iter().rev() {
        if op.gate.name == "TICK" {
            tick -= 1;
            if ticks.contains(&tick) {
                for q in 0..n {
                    for (which, set) in [(0, &tracker.xs[q]), (1, &tracker.zs[q])] {
                        for &t in set.iter() {
                            if targets.contains(&t) {
                                let e = out.entry(t).or_default().entry(tick).or_insert_with(|| (vec![false; n], vec![false; n]));
                                if which == 0 {
                                    e.0[q] ^= true;
                                } else {
                                    e.1[q] ^= true;
                                }
                            }
                        }
                    }
                }
            }
        }
        tracker.undo_gate(op)?;
    }
    tracker.undo_implicit_rzs_at_start_of_circuit()?;
    Ok(out)
}

/// The circuit's detector and observable counts (for the detecting-regions target filter).
pub fn count_detectors_and_observables(c: &ir::Circuit) -> (u64, u64) {
    let s = stats(c);
    (s.num_detectors, s.num_observables)
}

pub const OBS_BIT: u64 = OBSERVABLE_BIT;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qasm_of_a_bell_pair() {
        let c = ir::Circuit::parse("H 0\nCX 0 1\nM 0 1\nDETECTOR rec[-1] rec[-2]").unwrap();
        let q = to_qasm(&c, 3, false).unwrap();
        assert!(q.starts_with("OPENQASM 3.0;\ninclude \"stdgates.inc\";\n\nqreg q[2];\ncreg rec[2];\ncreg dets[1];\n\nh q[0];\ncx q[0], q[1];\n"));
        assert!(q.ends_with("dets[0] = rec[1] ^ rec[0] ^ 0;\n"));
        assert!(to_qasm(&c, 2, false).is_err());
        assert!(to_qasm(&c, 4, false).is_err());
    }

    #[test]
    fn quirk_of_a_bell_pair() {
        let c = ir::Circuit::parse("H 0\nCX 0 1").unwrap();
        assert_eq!(to_quirk_url(&c).unwrap(), r#"https://algassert.com/quirk#circuit={"cols":[["H"],["•","X"]]}"#);
    }

    #[test]
    fn regions_of_a_repetition_check() {
        let c = ir::Circuit::parse("R 0 1 2\nTICK\nCX 0 1 2 1\nTICK\nM 1\nDETECTOR rec[-1]").unwrap();
        let r = detecting_regions(&c, &[0].into_iter().collect(), &[0, 1].into_iter().collect(), false).unwrap();
        let d = &r[&0];
        assert_eq!(d[&1], (vec![false; 3], vec![false, true, false]));
        assert_eq!(d[&0], (vec![false; 3], vec![true, true, true]));
    }
}

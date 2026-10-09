//! Conversions between tableaus, stabilizers, state vectors, unitary matrices and circuits,
//! ported from Stim (`stabilizers_to_tableau.inl`, `circuit_vs_tableau.inl`,
//! `stabilizers_vs_amplitudes.inl`, `circuit_vs_amplitudes.cc`, `graph_simulator.cc`,
//! `tableau.inl`'s random sampler) so that every choice Stim makes (which destabilizers, which
//! gates, which global phase) comes out the same.

use super::ir::{self, GateTarget, Instruction, Item};
use super::pauli_string::PauliString;
use super::tableau::Tableau;
use super::tableau_sim::{gate_tableaus, Rng, TableauSimulator};
use crate::gate_data::{self, FLAG_IS_NOISY, FLAG_IS_RESET, FLAG_IS_SINGLE_QUBIT_GATE, FLAG_PRODUCES_RESULTS, FLAG_TARGETS_PAIRS};

pub type C64 = (f64, f64);

fn cmul(a: C64, b: C64) -> C64 {
    (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0)
}
fn cdiv(a: C64, b: C64) -> C64 {
    let d = b.0 * b.0 + b.1 * b.1;
    ((a.0 * b.0 + a.1 * b.1) / d, (a.1 * b.0 - a.0 * b.1) / d)
}
fn cabs(a: C64) -> f64 {
    (a.0 * a.0 + a.1 * a.1).sqrt()
}

/// A plain state-vector simulator (little-endian), for Stim's conversions.
struct VectorSim {
    state: Vec<C64>,
}

impl VectorSim {
    fn apply(&mut self, m: &[C64], qubits: &[usize]) {
        let k = qubits.len();
        let dim = 1usize << k;
        let masks: Vec<usize> = (0..dim).map(|j| (0..k).filter(|&q| j >> q & 1 == 1).map(|q| 1usize << qubits[q]).sum()).collect();
        let all = masks[dim - 1];
        for base in 0..self.state.len() {
            if base & all != 0 {
                continue;
            }
            let input: Vec<C64> = masks.iter().map(|&m| self.state[base | m]).collect();
            for r in 0..dim {
                let mut v = (0.0, 0.0);
                for c in 0..dim {
                    let p = cmul(m[r * dim + c], input[c]);
                    v.0 += p.0;
                    v.1 += p.1;
                }
                self.state[base | masks[r]] = v;
            }
        }
    }

    fn apply_gate(&mut self, name: &str, qubits: &[usize]) {
        let g = gate_data::info(name).expect("gate");
        let m: Vec<C64> = g.unitary.to_vec();
        self.apply(&m, qubits);
    }

    /// state ← (state + P state) / 2.
    fn project(&mut self, p: &PauliString) {
        let n = p.num_qubits();
        let (mut xm, mut zm, mut ys) = (0usize, 0usize, 0u32);
        for q in 0..n {
            if p.x(q) {
                xm |= 1 << q;
            }
            if p.z(q) {
                zm |= 1 << q;
            }
            if p.x(q) && p.z(q) {
                ys += 1;
            }
        }
        let mut out = self.state.clone();
        for j in 0..self.state.len() {
            let ph = (p.phase as u32 + ys + 2 * (j & zm).count_ones()) & 3;
            let c = [(1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0)][ph as usize];
            let v = cmul(c, self.state[j]);
            out[j ^ xm].0 += v.0;
            out[j ^ xm].1 += v.1;
        }
        for v in out.iter_mut() {
            v.0 *= 0.5;
            v.1 *= 0.5;
        }
        self.state = out;
    }

    /// Stim's `canonicalize_assuming_stabilizer_state`.
    fn canonicalize(&mut self, norm2: f64) -> Result<(), String> {
        let mut nz = 0;
        for k in 1..self.state.len() {
            if cabs(self.state[k]) > cabs(self.state[nz]) * 2.0 {
                nz = k;
            }
        }
        let big = self.state[nz];
        let mut nonzero = 0usize;
        for v in self.state.iter_mut() {
            let r = cdiv(*v, big);
            if cabs(r) < 0.1 {
                *v = (0.0, 0.0);
                continue;
            }
            nonzero += 1;
            *v = [(1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0)]
                .into_iter()
                .find(|&u| cabs((r.0 - u.0, r.1 - u.1)) < 0.1)
                .ok_or("State vector extraction failed. This shouldn't occur.")?;
        }
        let scale = (norm2 / nonzero as f64).sqrt();
        for v in self.state.iter_mut() {
            v.0 *= scale;
            v.1 *= scale;
        }
        Ok(())
    }

    /// Stim's `smooth_stabilizer_state`: every amplitude a ratio of the base in {0, ±1, ±i}.
    fn smooth(&mut self, base: C64) -> Result<(), String> {
        for v in self.state.iter_mut() {
            let r = cdiv(*v, base);
            let opts = [(0.0, 0.0), (1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)];
            let mut solved = false;
            for u in opts {
                let d = (r.0 - u.0, r.1 - u.1);
                if d.0 * d.0 + d.1 * d.1 < 0.125 {
                    *v = u;
                    solved = true;
                }
            }
            if !solved {
                return Err("The state vector wasn't a stabilizer state.".into());
            }
        }
        Ok(())
    }
}

/// The state stabilized by `stabilizers` (all on the same n qubits), little-endian, with
/// Stim's phase: the first sizeable amplitude real and positive, norm² `norm2`.
pub fn state_vector_from_stabilizers(stabilizers: &[PauliString], n: usize, norm2: f64) -> Result<Vec<C64>, String> {
    let mut rng = Rng::new(0x5eed);
    let mut sim = VectorSim { state: (0..1usize << n).map(|_| (rng.next_f64() * 2.0 - 1.0, rng.next_f64() * 2.0 - 1.0)).collect() };
    for p in stabilizers {
        sim.project(p);
    }
    if stabilizers.is_empty() {
        sim.project(&PauliString::new(0));
    }
    sim.canonicalize(norm2)?;
    Ok(sim.state)
}

fn reverse_bits(state: &[C64], n: usize) -> Vec<C64> {
    let mut out = vec![(0.0, 0.0); state.len()];
    for (j, v) in state.iter().enumerate() {
        let mut r = 0;
        for q in 0..n {
            if j >> q & 1 == 1 {
                r |= 1 << (n - 1 - q);
            }
        }
        out[r] = *v;
    }
    out
}

/// The state C|0…0⟩ (Stim's `Tableau.to_state_vector`).
pub fn tableau_to_state_vector(t: &Tableau, little_endian: bool) -> Result<Vec<C64>, String> {
    let n = t.num_qubits();
    let v = state_vector_from_stabilizers(&t.zs, n, 1.0)?;
    Ok(if little_endian { v } else { reverse_bits(&v, n) })
}

/// The unitary matrix, row-major, with Stim's global phase (Stim's `to_flat_unitary_matrix`).
pub fn tableau_to_unitary(t: &Tableau, little_endian: bool) -> Result<Vec<C64>, String> {
    let n = t.num_qubits();
    let mut ps = Vec::with_capacity(2 * n);
    for (k, row) in t.xs.iter().enumerate() {
        let mut p = PauliString::from_fn(2 * n, row.phase, |q| if q < n { row.get(q) } else { 0 });
        p.set_x(n + k, !p.x(n + k));
        ps.push(p);
    }
    for (k, row) in t.zs.iter().enumerate() {
        let mut p = PauliString::from_fn(2 * n, row.phase, |q| if q < n { row.get(q) } else { 0 });
        p.set_z(n + k, !p.z(n + k));
        ps.push(p);
    }
    for p in ps.iter_mut() {
        if !little_endian {
            let mut q = 0;
            while q < n - q - 1 {
                let q2 = n - q - 1;
                for off in [0, n] {
                    let (a, b) = (p.get(q + off), p.get(q2 + off));
                    p.set(q + off, b);
                    p.set(q2 + off, a);
                }
                q += 1;
            }
        }
        for q in 0..n {
            let (a, b) = (p.get(q), p.get(q + n));
            p.set(q, b);
            p.set(q + n, a);
        }
    }
    state_vector_from_stabilizers(&ps, 2 * n, (1usize << n) as f64)
}

/// The tableau of a Clifford unitary (row-major, dimension 2^n), or an error if it isn't one.
pub fn unitary_to_tableau(m: &[C64], little_endian: bool) -> Result<Tableau, String> {
    let dim = (m.len() as f64).sqrt().round() as usize;
    if dim * dim != m.len() || !dim.is_power_of_two() {
        return Err(format!("Matrix width and height must be a power of 2. Height was {dim}"));
    }
    let n = dim.trailing_zeros() as usize;
    let not_clifford = || "The given unitary matrix wasn't a Clifford operation.".to_string();
    let conj = |p: &PauliString| -> Result<PauliString, String> {
        let pm = p.to_unitary_matrix(little_endian);
        // U P U†.
        let mut up = vec![(0.0, 0.0); dim * dim];
        for r in 0..dim {
            for c in 0..dim {
                let mut s = (0.0, 0.0);
                for k in 0..dim {
                    let v = cmul(m[r * dim + k], pm[k * dim + c]);
                    s.0 += v.0;
                    s.1 += v.1;
                }
                up[r * dim + c] = s;
            }
        }
        let mut out = vec![(0.0, 0.0); dim * dim];
        for r in 0..dim {
            for c in 0..dim {
                let mut s = (0.0, 0.0);
                for k in 0..dim {
                    let u = m[c * dim + k];
                    let v = cmul(up[r * dim + k], (u.0, -u.1));
                    s.0 += v.0;
                    s.1 += v.1;
                }
                out[r * dim + c] = s;
            }
        }
        let image = PauliString::from_unitary_matrix(&out, little_endian, false).map_err(|_| not_clifford())?;
        if image.is_imaginary() {
            return Err(not_clifford());
        }
        Ok(image)
    };
    let mut t = Tableau::identity(n);
    for k in 0..n {
        let mut x = PauliString::new(n);
        x.set_x(k, true);
        let mut z = PauliString::new(n);
        z.set_z(k, true);
        t.xs[k] = conj(&x)?;
        t.zs[k] = conj(&z)?;
    }
    if !t.satisfies_invariants() {
        return Err(not_clifford());
    }
    Ok(t)
}

/// The tableau of a circuit's unitary part (Stim's `circuit_to_tableau`); `inverse` returns the
/// inverse tableau instead.
pub fn circuit_to_tableau(c: &ir::Circuit, ignore_noise: bool, ignore_measurement: bool, ignore_reset: bool, inverse: bool) -> Result<Tableau, String> {
    let n = c.count_qubits() as usize;
    let mut sim = TableauSimulator::new(n, 0);
    let mut err: Result<(), String> = Ok(());
    c.for_each_operation(&mut |op| {
        if err.is_err() {
            return;
        }
        let f = op.gate.flags;
        if !ignore_measurement && f & FLAG_PRODUCES_RESULTS != 0 {
            err = Err(format!("The circuit has no well-defined tableau because it contains measurement operations.\nTo ignore measurement operations, pass the argument ignore_measurement=True.\nThe first measurement operation is: {op}"));
            return;
        }
        if !ignore_reset && f & FLAG_IS_RESET != 0 {
            err = Err(format!("The circuit has no well-defined tableau because it contains reset operations.\nTo ignore reset operations, pass the argument ignore_reset=True.\nThe first reset operation is: {op}"));
            return;
        }
        if !ignore_noise && f & FLAG_IS_NOISY != 0 && op.args.iter().any(|&a| a > 0.0) {
            err = Err(format!("The circuit has no well-defined tableau because it contains noisy operations.\nTo ignore noisy operations, pass the argument ignore_noise=True.\nThe first noisy operation is: {op}"));
            return;
        }
        if op.gate.is_unitary {
            err = sim.do_gate(op);
        }
    });
    err?;
    Ok(if inverse { sim.inv_state } else { sim.inv_state.inverse(false) })
}

/// The inverse of a unitary circuit (Stim's `circuit_inverse_unitary`).
pub fn circuit_inverse_unitary(c: &ir::Circuit) -> Result<ir::Circuit, String> {
    let mut ops = Vec::new();
    c.for_each_operation(&mut |op| ops.push(op.clone()));
    let mut out = ir::Circuit::new();
    for op in ops.iter().rev() {
        if !op.gate.is_unitary {
            return Err(format!("Not unitary: {op}"));
        }
        let step = if op.has_flag(FLAG_TARGETS_PAIRS) { 2 } else { 1 };
        let inv = op.gate.inverse.unwrap_or(op.gate.name);
        let mut k = op.targets.len();
        while k > 0 {
            out.push_op(Instruction::new(inv, op.args.clone(), op.targets[k - step..k].to_vec(), &op.tag)?);
            k -= step;
        }
    }
    Ok(out)
}

fn q(t: usize) -> GateTarget {
    GateTarget::qubit(t as u32, false)
}

fn op(name: &str, targets: Vec<GateTarget>) -> Instruction {
    Instruction::new(name, vec![], targets, "").expect("known gate")
}

/// Stim's `stabilizers_to_tableau`: a tableau whose Z images are the stabilizers (and whose X
/// images are Stim's choice of destabilizers).
pub fn stabilizers_to_tableau(stabilizers: &[PauliString], allow_redundant: bool, allow_underconstrained: bool, invert: bool) -> Result<Tableau, String> {
    let n = stabilizers.iter().map(|p| p.num_qubits()).max().unwrap_or(0);
    let mut buf: Vec<PauliString> = stabilizers
        .iter()
        .map(|p| {
            let mut c = p.clone();
            c.ensure_num_qubits(n);
            c
        })
        .collect();
    let mut elim = ir::Circuit::new();
    let anticommute_error = |stabs: &[PauliString]| -> String {
        for k1 in 0..stabs.len() {
            for k2 in k1 + 1..stabs.len() {
                if !stabs[k1].commutes(&stabs[k2]) {
                    return format!("Some of the given stabilizers anticommute.\nFor example:\n    stabilizers[{k1}] = {}\nanticommutes with\n    stabilizers[{k2}] = {}", stabs[k1], stabs[k2]);
                }
            }
        }
        "The given stabilizers commute but the solver failed in a way that suggests they anticommute. Please report this as a bug.".into()
    };
    let apply = |buf: &mut Vec<PauliString>, elim: &mut ir::Circuit, name: &str, targets: &[usize]| {
        let (t, _) = gate_tableaus(name).expect("gate");
        for p in buf.iter_mut() {
            t.apply_within(p, targets);
        }
        elim.push_op(op(name, targets.iter().map(|&t| q(t)).collect()));
    };
    let redundant_parts = |k: usize, elim: &ir::Circuit| -> String {
        let mut target = stabilizers[k].clone();
        target.ensure_num_qubits(n);
        let mut e = elim.clone();
        if n > 0 {
            e.push_op(op("X", vec![q(n - 1)]));
            e.push_op(op("X", vec![q(n - 1)]));
        }
        let fwd = circuit_to_tableau(&e, false, false, false, false).expect("unitary");
        let target = fwd.apply(&target).expect("sizes");
        let inverse = circuit_to_tableau(&e, false, false, false, true).expect("unitary");
        let mut s = String::new();
        for qq in 0..n {
            if !(target.x(qq) || target.z(qq)) {
                continue;
            }
            s.push_str("\n    ");
            let hit = (0..stabilizers.len()).find(|&j| {
                let mut p = stabilizers[j].clone();
                p.ensure_num_qubits(n);
                p == inverse.zs[qq]
            });
            match hit {
                Some(j) => s.push_str(&format!("stabilizers[{j}] = {}", stabilizers[j])),
                None => s.push_str(&inverse.zs[qq].to_string()),
            }
        }
        s
    };
    let mut used = 0;
    for k in 0..buf.len() {
        for qq in 0..used {
            if buf[k].x(qq) {
                return Err(anticommute_error(stabilizers));
            }
        }
        let pivot = (used..n).find(|&p| buf[k].x(p) || buf[k].z(p));
        let Some(pivot) = pivot else {
            if buf[k].sign() {
                return Err(format!("Some of the given stabilizers contradict each other.\nFor example:\n    stabilizers[{k}] = {}\nis the negation of the product of the following stabilizers: {{{}\n}}", stabilizers[k], redundant_parts(k, &elim)));
            }
            if !allow_redundant {
                return Err(format!("Some of the given stabilizers are redundant.\nTo allow redundant stabilizers, pass the argument allow_redundant=True.\n\nFor example:\n    stabilizers[{k}] = {}\nis the product of the following stabilizers: {{{}\n}}", stabilizers[k], redundant_parts(k, &elim)));
            }
            continue;
        };
        if buf[k].x(pivot) {
            let g = if buf[k].z(pivot) { "H_YZ" } else { "H" };
            apply(&mut buf, &mut elim, g, &[pivot]);
        }
        for qq in 0..n {
            let p = buf[k].x(qq) as u8 + 2 * buf[k].z(qq) as u8;
            if p != 0 && qq != pivot {
                let g = match p {
                    1 => "XCX",
                    2 => "XCZ",
                    _ => "XCY",
                };
                apply(&mut buf, &mut elim, g, &[pivot, qq]);
            }
        }
        if pivot != used {
            apply(&mut buf, &mut elim, "SWAP", &[pivot, used]);
        }
        if buf[k].sign() {
            apply(&mut buf, &mut elim, "X", &[used]);
        }
        used += 1;
    }
    if used < n && !allow_underconstrained {
        return Err("There weren't enough stabilizers to uniquely specify the state. To allow underspecifying the state, pass the argument allow_underconstrained=True.".into());
    }
    if n > 0 {
        elim.push_op(op("X", vec![q(n - 1)]));
        elim.push_op(op("X", vec![q(n - 1)]));
    }
    if invert {
        circuit_to_tableau(&circuit_inverse_unitary(&elim)?, false, false, false, true)
    } else {
        circuit_to_tableau(&elim, false, false, false, true)
    }
}

/// A circuit preparing the given stabilizer state from |0…0⟩ (Stim's
/// `stabilizer_state_vector_to_circuit`).
pub fn stabilizer_state_vector_to_circuit(state: &[C64], little_endian: bool) -> Result<ir::Circuit, String> {
    let len = state.len();
    if len == 0 || !len.is_power_of_two() {
        return Err(format!("Expected number of amplitudes to be a power of 2. The given state vector had {len} amplitudes."));
    }
    let n = len.trailing_zeros() as usize;
    // Stim works in single precision.
    let state: Vec<C64> = state.iter().map(|v| (v.0 as f32 as f64, v.1 as f32 as f64)).collect();
    let mut sim = VectorSim { state: state.clone() };
    let mut rec = ir::Circuit::new();
    let tq = |t: usize| q(if little_endian { t } else { n - t - 1 });
    let apply = |sim: &mut VectorSim, rec: &mut ir::Circuit, g: &str, t: &[usize]| {
        sim.apply_gate(g, t);
        rec.push_op(op(g, t.iter().map(|&x| tq(x)).collect()));
    };
    let norm = |v: C64| v.0 * v.0 + v.1 * v.1;
    let mut pivot = 0;
    for k in 1..len {
        if norm(state[k]) > norm(state[pivot]) {
            pivot = k;
        }
    }
    for qq in 0..n {
        if pivot >> qq & 1 == 1 {
            apply(&mut sim, &mut rec, "X", &[qq]);
        }
    }
    let base = sim.state[0];
    sim.smooth(base)?;
    let occ = |s: &VectorSim| s.state.iter().filter(|v| v.0 != 0.0 || v.1 != 0.0).count();
    let mut occupation = occ(&sim);
    if !occupation.is_power_of_two() {
        return Err("State vector isn't a stabilizer state.".into());
    }
    while occupation > 1 {
        let Some(k) = (1..len).find(|&k| sim.state[k].0 != 0.0 || sim.state[k].1 != 0.0) else { break };
        let mut base_qubit = usize::MAX;
        for qq in 0..n {
            if k >> qq & 1 == 1 {
                if base_qubit == usize::MAX {
                    base_qubit = qq;
                } else {
                    apply(&mut sim, &mut rec, "CX", &[base_qubit, qq]);
                }
            }
        }
        let s = sim.state[1 << base_qubit];
        if s == (-1.0, 0.0) {
            apply(&mut sim, &mut rec, "Z", &[base_qubit]);
        } else if s == (0.0, 1.0) {
            apply(&mut sim, &mut rec, "S_DAG", &[base_qubit]);
        } else if s == (0.0, -1.0) {
            apply(&mut sim, &mut rec, "S", &[base_qubit]);
        }
        apply(&mut sim, &mut rec, "H", &[base_qubit]);
        let base = sim.state[0];
        sim.smooth(base)?;
        if occ(&sim) * 2 != occupation {
            return Err("State vector isn't a stabilizer state.".into());
        }
        occupation >>= 1;
    }
    let mut rec = circuit_inverse_unitary(&rec)?;
    if (rec.count_qubits() as usize) < n {
        rec.push_op(op("I", vec![q(n - 1)]));
    }
    Ok(rec)
}

/// Stim's `Tableau.from_state_vector`.
pub fn state_vector_to_tableau(state: &[C64], little_endian: bool) -> Result<Tableau, String> {
    let c = stabilizer_state_vector_to_circuit(state, little_endian)?;
    circuit_to_tableau(&c, false, false, false, false)
}

/// Stim's `tableau_to_circuit` with method "elimination", "graph_state", "mpp_state" or
/// "mpp_state_unsigned".
pub fn tableau_to_circuit(t: &Tableau, method: &str) -> Result<ir::Circuit, String> {
    match method {
        "elimination" => Ok(elimination_circuit(t)),
        "graph_state" => {
            let mut g = GraphSimulator::new(t.num_qubits());
            g.do_circuit(&elimination_circuit(t))?;
            Ok(g.to_circuit(true))
        }
        "mpp_state" => Ok(mpp_circuit(t, false)),
        "mpp_state_unsigned" => Ok(mpp_circuit(t, true)),
        _ => Err(format!("Unknown method: '{method}'. Known methods:\n    - 'elimination'\n    - 'graph_state'\n    - 'mpp_state'\n    - 'mpp_state_unsigned'\n")),
    }
}

fn mpp_circuit(t: &Tableau, skip_sign: bool) -> ir::Circuit {
    let n = t.num_qubits();
    let mut out = ir::Circuit::new();
    for k in 0..n {
        let s = &t.zs[k];
        let mut need_sign = s.sign();
        let mut targets = Vec::new();
        for qq in 0..n {
            let (x, z) = (s.x(qq), s.z(qq));
            if x || z {
                targets.push(GateTarget::pauli_xz(qq as u32, x, z, need_sign));
                targets.push(GateTarget::combiner());
                need_sign = false;
            }
        }
        targets.pop();
        out.push_op(op("MPP", targets));
    }
    if !skip_sign {
        let (mut tx, mut ty, mut tz) = (Vec::new(), Vec::new(), Vec::new());
        for k in 0..n {
            let d = &t.xs[k];
            for qq in 0..n {
                let dest = match (d.x(qq), d.z(qq)) {
                    (true, false) => Some(&mut tx),
                    (false, true) => Some(&mut tz),
                    (true, true) => Some(&mut ty),
                    _ => None,
                };
                if let Some(v) = dest {
                    v.push(GateTarget::rec((n - k) as u32));
                    v.push(q(qq));
                }
            }
        }
        for (g, v) in [("CX", tx), ("CY", ty), ("CZ", tz)] {
            if !v.is_empty() {
                out.push_op(op(g, v));
            }
        }
    }
    out
}

fn elimination_circuit(t: &Tableau) -> ir::Circuit {
    let mut rem = t.inverse(false);
    let mut rec = ir::Circuit::new();
    let n = rem.num_qubits();
    let apply = |rem: &mut Tableau, rec: &mut ir::Circuit, g: &str, targets: &[usize]| {
        let (gt, _) = gate_tableaus(g).expect("gate");
        rem.append(gt, targets);
        rec.push_op(op(g, targets.iter().map(|&x| q(x)).collect()));
    };
    let xo = |rem: &Tableau, i: usize, o: usize| rem.xs[i].x(o) as u8 + 2 * rem.xs[i].z(o) as u8;
    let zo = |rem: &Tableau, i: usize, o: usize| rem.zs[i].x(o) as u8 + 2 * rem.zs[i].z(o) as u8;
    for col in 0..n {
        let mut pivot_row = col;
        while pivot_row < n {
            let (px, pz) = (xo(&rem, col, pivot_row), zo(&rem, col, pivot_row));
            if px != 0 && pz != 0 && px != pz {
                break;
            }
            pivot_row += 1;
        }
        if pivot_row != col {
            apply(&mut rem, &mut rec, "CX", &[pivot_row, col]);
            apply(&mut rem, &mut rec, "CX", &[col, pivot_row]);
            apply(&mut rem, &mut rec, "CX", &[pivot_row, col]);
        }
        if zo(&rem, col, col) == 3 {
            apply(&mut rem, &mut rec, "S", &[col]);
        }
        if zo(&rem, col, col) != 2 {
            apply(&mut rem, &mut rec, "H", &[col]);
        }
        if xo(&rem, col, col) != 1 {
            apply(&mut rem, &mut rec, "S", &[col]);
        }
        for row in col + 1..n {
            if xo(&rem, col, row) == 3 {
                apply(&mut rem, &mut rec, "S", &[row]);
            }
        }
        for row in col + 1..n {
            if xo(&rem, col, row) == 2 {
                apply(&mut rem, &mut rec, "H", &[row]);
            }
        }
        for row in col + 1..n {
            if xo(&rem, col, row) != 0 {
                apply(&mut rem, &mut rec, "CX", &[col, row]);
            }
        }
        for row in col + 1..n {
            if zo(&rem, col, row) == 3 {
                apply(&mut rem, &mut rec, "S", &[row]);
            }
        }
        for row in col + 1..n {
            if zo(&rem, col, row) == 1 {
                apply(&mut rem, &mut rec, "H", &[row]);
            }
        }
        for row in col + 1..n {
            if zo(&rem, col, row) != 0 {
                apply(&mut rem, &mut rec, "CX", &[row, col]);
            }
        }
    }
    let signs: Vec<bool> = (0..n).map(|c| rem.zs[c].sign()).collect();
    for col in 0..n {
        if signs[col] {
            apply(&mut rem, &mut rec, "H", &[col]);
        }
    }
    for col in 0..n {
        if signs[col] {
            apply(&mut rem, &mut rec, "S", &[col]);
            apply(&mut rem, &mut rec, "S", &[col]);
        }
    }
    for col in 0..n {
        if signs[col] {
            apply(&mut rem, &mut rec, "H", &[col]);
        }
    }
    for col in 0..n {
        if rem.xs[col].sign() {
            apply(&mut rem, &mut rec, "S", &[col]);
            apply(&mut rem, &mut rec, "S", &[col]);
        }
    }
    if (rec.count_qubits() as usize) < n {
        apply(&mut rem, &mut rec, "H", &[n - 1]);
        apply(&mut rem, &mut rec, "H", &[n - 1]);
    }
    rec
}

/// Stim's `GraphSimulator`: a graph state under a layer of single-qubit Cliffords and Paulis.
pub struct GraphSimulator {
    n: usize,
    adj: Vec<Vec<bool>>,
    paulis: PauliString,
    x2outs: PauliString,
    z2outs: PauliString,
}

impl GraphSimulator {
    pub fn new(n: usize) -> GraphSimulator {
        let mut x2outs = PauliString::new(n);
        let mut z2outs = PauliString::new(n);
        for k in 0..n {
            x2outs.set_z(k, true);
            z2outs.set_x(k, true);
        }
        GraphSimulator { n, adj: vec![vec![false; n]; n], paulis: PauliString::new(n), x2outs, z2outs }
    }

    fn do_1q_gate(&mut self, gate: &str, qubit: usize) {
        let (t, _) = gate_tableaus(gate).expect("gate");
        t.apply_within(&mut self.x2outs, &[qubit]);
        t.apply_within(&mut self.z2outs, &[qubit]);
        let px = self.paulis.x(qubit) ^ self.z2outs.sign();
        let pz = self.paulis.z(qubit) ^ self.x2outs.sign();
        self.paulis.set_x(qubit, px);
        self.paulis.set_z(qubit, pz);
        self.x2outs.phase = 0;
        self.z2outs.phase = 0;
    }

    fn after2inside(&self, qubit: usize, x: bool, z: bool) -> (bool, bool, bool) {
        let (xx, xz, zx, zz) = (self.x2outs.x(qubit), self.x2outs.z(qubit), self.z2outs.x(qubit), self.z2outs.z(qubit));
        let out_x = (x & zz) ^ (z & zx);
        let out_z = (x & xz) ^ (z & xx);
        let mut sign = false;
        sign ^= self.paulis.x(qubit) & out_z;
        sign ^= self.paulis.z(qubit) & out_x;
        sign ^= out_x == out_z && !(xx ^ zz) && !(xx ^ xz ^ zx);
        (out_x, out_z, sign)
    }

    fn cz(&mut self, a: usize, b: usize) {
        self.adj[a][b] ^= true;
        self.adj[b][a] ^= true;
    }

    fn cx(&mut self, c: usize, t: usize) {
        for k in 0..self.n {
            let v = self.adj[c][k] ^ self.adj[t][k];
            self.adj[c][k] = v;
        }
        for k in 0..self.n {
            self.adj[k][c] = self.adj[c][k];
        }
        let z = self.paulis.z(c) ^ self.adj[c][c];
        self.paulis.set_z(c, z);
        self.adj[c][c] = false;
    }

    fn sqrt_z(&mut self, q: usize) {
        let (x2x, x2z, z2x, z2z) = (self.x2outs.x(q), self.x2outs.z(q), self.z2outs.x(q), self.z2outs.z(q));
        let mut z = self.paulis.z(q) ^ self.paulis.x(q);
        z ^= !(x2x ^ z2z) && !(x2x ^ x2z ^ z2x);
        self.paulis.set_z(q, z);
        self.x2outs.set_x(q, x2x ^ z2x);
        self.x2outs.set_z(q, x2z ^ z2z);
    }

    fn sqrt_x_dag(&mut self, q: usize) {
        let (x2x, x2z, z2x, z2z) = (self.x2outs.x(q), self.x2outs.z(q), self.z2outs.x(q), self.z2outs.z(q));
        let mut x = self.paulis.x(q) ^ self.paulis.z(q);
        x ^= !(x2x ^ z2z) && !(x2x ^ x2z ^ z2x);
        self.paulis.set_x(q, x);
        self.z2outs.set_x(q, z2x ^ x2x);
        self.z2outs.set_z(q, z2z ^ x2z);
    }

    fn cy(&mut self, c: usize, t: usize) {
        self.cz(c, t);
        self.cx(c, t);
        self.sqrt_z(c);
    }

    fn complementation(&mut self, q: usize) {
        let buffer: Vec<usize> = (0..self.n).filter(|&nb| self.adj[q][nb]).collect();
        for &nb in &buffer {
            self.sqrt_z(nb);
        }
        for k1 in 0..buffer.len() {
            for k2 in k1 + 1..buffer.len() {
                self.cz(buffer[k1], buffer[k2]);
            }
        }
        self.sqrt_x_dag(q);
    }

    fn flip_z(&mut self, q: usize) {
        let v = !self.paulis.z(q);
        self.paulis.set_z(q, v);
    }
    fn flip_x(&mut self, q: usize) {
        let v = !self.paulis.x(q);
        self.paulis.set_x(q, v);
    }

    fn ycx(&mut self, q1: usize, q2: usize) {
        if self.adj[q1][q2] {
            self.complementation(q1);
            self.cy(q1, q2);
            self.flip_z(q1);
        } else {
            self.complementation(q1);
            self.cx(q1, q2);
        }
    }

    fn ycy(&mut self, q1: usize, q2: usize) {
        if self.adj[q1][q2] {
            self.complementation(q1);
            self.cx(q1, q2);
        } else {
            self.complementation(q1);
            self.cy(q1, q2);
        }
    }

    fn xcx(&mut self, q1: usize, q2: usize) {
        if self.adj[q1][q2] {
            self.complementation(q2);
            self.complementation(q1);
            self.cy(q1, q2);
            self.flip_z(q1);
            self.flip_x(q2);
            self.flip_z(q2);
        } else {
            for q3 in 0..self.n {
                if self.adj[q1][q3] {
                    self.complementation(q3);
                    if self.adj[q2][q3] {
                        self.flip_x(q1);
                        self.flip_z(q1);
                        self.flip_x(q2);
                        self.flip_z(q2);
                        self.ycy(q1, q2);
                    } else {
                        self.flip_x(q2);
                        self.ycx(q1, q2);
                    }
                    return;
                }
            }
        }
    }

    fn inside_pauli_interaction(&mut self, x1: bool, z1: bool, x2: bool, z2: bool, q1: usize, q2: usize) {
        let p1 = x1 as i32 + 2 * z1 as i32 - 1;
        let p2 = x2 as i32 + 2 * z2 as i32 - 1;
        match p1 + p2 * 3 {
            0 => self.xcx(q1, q2),
            1 => self.cx(q1, q2),
            2 => self.ycx(q1, q2),
            3 => self.cx(q2, q1),
            4 => self.cz(q1, q2),
            5 => self.cy(q2, q1),
            6 => self.ycx(q2, q1),
            7 => self.cy(q1, q2),
            _ => self.ycy(q1, q2),
        }
    }

    fn pauli_interaction(&mut self, x1: bool, z1: bool, x2: bool, z2: bool, q1: usize, q2: usize) {
        let (x1i, z1i, s1) = self.after2inside(q1, x1, z1);
        let (x2i, z2i, s2) = self.after2inside(q2, x2, z2);
        if s1 {
            let (a, b) = (self.paulis.x(q2) ^ x2i, self.paulis.z(q2) ^ z2i);
            self.paulis.set_x(q2, a);
            self.paulis.set_z(q2, b);
        }
        if s2 {
            let (a, b) = (self.paulis.x(q1) ^ x1i, self.paulis.z(q1) ^ z1i);
            self.paulis.set_x(q1, a);
            self.paulis.set_z(q1, b);
        }
        self.inside_pauli_interaction(x1i, z1i, x2i, z2i, q1, q2);
    }

    fn do_instruction(&mut self, inst: &Instruction) -> Result<(), String> {
        let f = inst.gate.flags;
        if inst.gate.is_unitary {
            if f & FLAG_IS_SINGLE_QUBIT_GATE != 0 {
                for t in &inst.targets {
                    self.do_1q_gate(inst.gate.name, t.value() as usize);
                }
                return Ok(());
            }
            if f & FLAG_TARGETS_PAIRS != 0 {
                let (p1, p2): (u8, u8) = match inst.gate.name {
                    "XCX" => (1, 1),
                    "XCY" => (1, 3),
                    "XCZ" => (1, 2),
                    "YCX" => (3, 1),
                    "YCY" => (3, 3),
                    "YCZ" => (3, 2),
                    "CX" => (2, 1),
                    "CY" => (2, 3),
                    "CZ" => (2, 2),
                    _ => return self.by_decomposition(inst),
                };
                for pair in inst.targets.chunks(2) {
                    if !pair[0].is_qubit() || !pair[1].is_qubit() {
                        return Err(format!("Unsupported operation: {inst}"));
                    }
                    self.pauli_interaction(p1 & 1 != 0, p1 & 2 != 0, p2 & 1 != 0, p2 & 2 != 0, pair[0].value() as usize, pair[1].value() as usize);
                }
                return Ok(());
            }
        }
        match inst.gate.name {
            "TICK" | "QUBIT_COORDS" | "SHIFT_COORDS" => Ok(()),
            _ => Err(format!("Unsupported operation: {inst}")),
        }
    }

    fn by_decomposition(&mut self, inst: &Instruction) -> Result<(), String> {
        let d = ir::Circuit::parse(inst.gate.decomposition)?;
        for pair in inst.targets.chunks(2) {
            let (a, b) = (pair[0].value() as usize, pair[1].value() as usize);
            for it in &d.items {
                let Item::Op(o) = it else { continue };
                let map = |t: GateTarget| if t.value() == 0 { a } else { b };
                if o.gate.name == "CX" {
                    for p in o.targets.chunks(2) {
                        self.pauli_interaction(false, true, true, false, map(p[0]), map(p[1]));
                    }
                } else {
                    for t in &o.targets {
                        self.do_1q_gate(o.gate.name, map(*t));
                    }
                }
            }
        }
        Ok(())
    }

    pub fn do_circuit(&mut self, c: &ir::Circuit) -> Result<(), String> {
        let mut err = Ok(());
        c.for_each_operation(&mut |op| {
            if err.is_ok() {
                err = self.do_instruction(op);
            }
        });
        err
    }

    fn output_pauli_layer(&self, out: &mut ir::Circuit, to_hs_xyz: bool) {
        let mut groups: [Vec<GateTarget>; 4] = Default::default();
        for qq in 0..self.n {
            let x = self.paulis.x(qq);
            let mut z = self.paulis.z(qq);
            if to_hs_xyz {
                let (xx, xz, zx, zz) = (self.x2outs.x(qq), self.x2outs.z(qq), self.z2outs.x(qq), self.z2outs.z(qq));
                z ^= xx && xz && zx && !zz;
            }
            groups[x as usize + 2 * z as usize].push(q(qq));
        }
        for (g, k) in [("X", 1), ("Y", 3), ("Z", 2)] {
            if !groups[k].is_empty() {
                out.push_op(op(g, groups[k].clone()));
            }
        }
    }

    pub fn to_circuit(&self, to_hs_xyz: bool) -> ir::Circuit {
        let mut out = ir::Circuit::new();
        let all: Vec<GateTarget> = (0..self.n).map(q).collect();
        if !all.is_empty() {
            out.push_op(op("RX", all));
        }
        out.push_op(op("TICK", vec![]));
        let mut has_cz = false;
        for a in 0..self.n {
            let mut t = Vec::new();
            for b in a + 1..self.n {
                if self.adj[a][b] {
                    t.push(q(a));
                    t.push(q(b));
                }
            }
            if !t.is_empty() {
                has_cz = true;
                out.push_op(op("CZ", t));
            }
        }
        if has_cz {
            out.push_op(op("TICK", vec![]));
        }
        self.output_pauli_layer(&mut out, to_hs_xyz);
        let mut groups: Vec<Vec<GateTarget>> = vec![Vec::new(); 16];
        for qq in 0..self.n {
            let (xx, xz, zx, zz) = (self.x2outs.x(qq), self.x2outs.z(qq), self.z2outs.x(qq), self.z2outs.z(qq));
            groups[xx as usize + 2 * xz as usize + 4 * zx as usize + 8 * zz as usize].push(q(qq));
        }
        let mut shs: [Vec<GateTarget>; 3] = Default::default();
        for (g, k, use_shs) in [("C_XYZ", 0b0111, [true, true, false]), ("C_ZYX", 0b1110, [false, true, true]), ("H", 0b0110, [false, true, false]), ("S", 0b1011, [true, false, false]), ("SQRT_X_DAG", 0b1101, [true, true, true])] {
            if to_hs_xyz {
                for j in 0..3 {
                    if use_shs[j] {
                        shs[j].extend(groups[k].iter().copied());
                    }
                }
            } else if !groups[k].is_empty() {
                out.push_op(op(g, groups[k].clone()));
            }
        }
        for (k, v) in shs.iter_mut().enumerate() {
            if !v.is_empty() {
                v.sort();
                out.push_op(op(if k == 1 { "H" } else { "S" }, v.clone()));
            }
        }
        out
    }
}

type Bits = Vec<Vec<bool>>;

fn identity_bits(n: usize) -> Bits {
    (0..n).map(|r| (0..n).map(|c| r == c).collect()).collect()
}

fn mat_mul(a: &Bits, b: &Bits, n: usize) -> Bits {
    (0..n).map(|r| (0..n).map(|c| (0..n).fold(false, |acc, k| acc ^ (a[r][k] & b[k][c]))).collect()).collect()
}

fn inverse_lower(m: &Bits, n: usize) -> Bits {
    let mut result = identity_bits(n);
    for target in 0..n {
        let mut row = m[target].clone();
        for pivot in 0..target {
            if row[pivot] {
                for c in 0..n {
                    row[c] ^= m[pivot][c];
                }
                let rp = result[pivot].clone();
                for c in 0..n {
                    result[target][c] ^= rp[c];
                }
            }
        }
    }
    result
}

fn randomize(row: &mut [bool], count: usize, rng: &mut Rng) {
    for b in row.iter_mut().take(count) {
        *b = rng.next_u64() & 1 == 1;
    }
}

/// A uniformly random Clifford tableau (Bravyi and Maslov, arXiv 2003.09412), as Stim samples it.
pub fn random_tableau(n: usize, rng: &mut Rng) -> Tableau {
    // Quantum Mallows sample.
    let mut hada = Vec::new();
    let mut perm = Vec::new();
    let mut remaining: Vec<usize> = (0..n).collect();
    for _ in 0..n {
        let m = remaining.len();
        let u = rng.next_f64();
        let eps = 4f64.powi(-(m as i32));
        let mut k = (-(u + (1.0 - u) * eps).log2().ceil()) as usize;
        hada.push(k < m);
        if k >= m {
            k = 2 * m - k - 1;
        }
        perm.push(remaining[k]);
        remaining.remove(k);
    }
    let mut sym = vec![vec![false; n]; n];
    for row in 0..n {
        randomize(&mut sym[row], row + 1, rng);
        for col in 0..row {
            sym[col][row] = sym[row][col];
        }
    }
    let mut sym_m = vec![vec![false; n]; n];
    for row in 0..n {
        randomize(&mut sym_m[row], row + 1, rng);
        sym_m[row][row] &= hada[row];
        for col in 0..row {
            let mut b = hada[row] && hada[col];
            b |= hada[row] && !hada[col] && perm[row] < perm[col];
            b |= !hada[row] && hada[col] && perm[row] > perm[col];
            sym_m[row][col] &= b;
            sym_m[col][row] = sym_m[row][col];
        }
    }
    let mut lower = identity_bits(n);
    for row in 0..n {
        randomize(&mut lower[row], row, rng);
    }
    let mut lower_m = identity_bits(n);
    for row in 0..n {
        randomize(&mut lower_m[row], row, rng);
        for col in 0..row {
            let mut b = !hada[row] && hada[col];
            b |= hada[row] && hada[col] && perm[row] > perm[col];
            b |= !hada[row] && !hada[col] && perm[row] < perm[col];
            lower_m[row][col] &= b;
        }
    }
    let prod = mat_mul(&sym, &lower, n);
    let prod_m = mat_mul(&sym_m, &lower_m, n);
    let transpose = |m: &Bits| -> Bits { (0..n).map(|r| (0..n).map(|c| m[c][r]).collect()).collect() };
    let inv = transpose(&inverse_lower(&lower, n));
    let inv_m = transpose(&inverse_lower(&lower_m, n));
    let quad = |a: &Bits, b: &Bits, c: &Bits, d: &Bits| -> Bits {
        let mut m = vec![vec![false; 2 * n]; 2 * n];
        for r in 0..n {
            for col in 0..n {
                m[r][col] = a[r][col];
                m[r][col + n] = b[r][col];
                m[r + n][col] = c[r][col];
                m[r + n][col + n] = d[r][col];
            }
        }
        m
    };
    let zero = vec![vec![false; n]; n];
    let fused = quad(&lower, &zero, &prod, &inv);
    let fused_m = quad(&lower_m, &zero, &prod_m, &inv_m);
    let mut u = vec![vec![false; 2 * n]; 2 * n];
    for row in 0..n {
        u[row] = fused[perm[row]].clone();
        u[row + n] = fused[perm[row] + n].clone();
    }
    for row in 0..n {
        if hada[row] {
            u.swap(row, row + n);
        }
    }
    let raw = mat_mul(&fused_m, &u, 2 * n);
    let mut t = Tableau::identity(n);
    for row in 0..n {
        let mut x = PauliString::new(n);
        let mut z = PauliString::new(n);
        for col in 0..n {
            x.set_x(col, raw[row][col]);
            x.set_z(col, raw[row][col + n]);
            z.set_x(col, raw[row + n][col]);
            z.set_z(col, raw[row + n][col + n]);
        }
        t.xs[row] = x;
        t.zs[row] = z;
    }
    for row in 0..n {
        if rng.next_u64() & 1 == 1 {
            t.xs[row].phase = 2;
        }
    }
    for row in 0..n {
        if rng.next_u64() & 1 == 1 {
            t.zs[row].phase = 2;
        }
    }
    t
}

#[cfg(test)]
mod tests {
    use super::*;

    fn circ(t: &str) -> ir::Circuit {
        ir::Circuit::parse(t).unwrap()
    }

    #[test]
    fn state_vectors_and_unitaries_have_stims_phase() {
        let h = Tableau::from_named_gate("H").unwrap();
        let v = tableau_to_state_vector(&h, true).unwrap();
        let r = 0.5f64.sqrt();
        assert!((v[0].0 - r).abs() < 1e-9 && (v[1].0 - r).abs() < 1e-9);
        let u = tableau_to_unitary(&Tableau::from_named_gate("SQRT_X").unwrap(), true).unwrap();
        // The first entry is made real and positive: [[1, -i], [-i, 1]]/√2.
        assert!((u[0].0 - r).abs() < 1e-9 && u[0].1.abs() < 1e-9);
        assert!((u[1].1 + r).abs() < 1e-9);
        let back = unitary_to_tableau(&u, true).unwrap();
        assert_eq!(back, Tableau::from_named_gate("SQRT_X").unwrap());
    }

    #[test]
    fn round_trips() {
        let mut rng = Rng::new(3);
        for n in 0..6 {
            for _ in 0..10 {
                let t = random_tableau(n, &mut rng);
                assert!(t.satisfies_invariants());
                let c = tableau_to_circuit(&t, "elimination").unwrap();
                let back = circuit_to_tableau(&c, false, false, false, false).unwrap();
                assert_eq!(back.num_qubits(), n.max(back.num_qubits()));
                if n > 0 {
                    assert_eq!(back, t);
                    let st = stabilizers_to_tableau(&t.zs, false, false, false).unwrap();
                    assert_eq!(st.stabilizers(true), t.stabilizers(true));
                    let g = tableau_to_circuit(&t, "graph_state").unwrap();
                    let mut sim = TableauSimulator::new(n, 0);
                    sim.do_circuit(&g).unwrap();
                    assert_eq!(sim.canonical_stabilizers(), t.stabilizers(true));
                    if n <= 4 {
                        let v = tableau_to_state_vector(&t, true).unwrap();
                        let t2 = state_vector_to_tableau(&v, true).unwrap();
                        assert_eq!(t2.stabilizers(true), t.stabilizers(true));
                        let u = tableau_to_unitary(&t, false).unwrap();
                        assert_eq!(unitary_to_tableau(&u, false).unwrap(), t);
                    }
                }
            }
        }
        let c = circ("H 0\nCX 0 1\nS 1");
        assert_eq!(circuit_inverse_unitary(&c).unwrap().to_string(), "S_DAG 1\nCX 0 1\nH 0");
    }
}

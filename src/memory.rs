//! Surface-code memory experiments, written out as circuits.
//!
//! The same patch the rest of the engine simulates, in the general form: a
//! circuit with detectors and an observable, which Stim can read and this
//! engine can build a model of, sample and decode like any other.
//!
//! Two noise models:
//!
//! - **Current** reproduces the engine's own circuit-level model gate for gate:
//!   a biased Pauli at every `Op::Noise` location of `round_program` (so each
//!   qubit of a CNOT errs independently, and a gate fails about 2p of the
//!   time), flipped readouts, no idle noise, and a noiseless first and last
//!   round. `equivalence.rs` proves it is the old path's circuit.
//! - **SD6** is the standard model of Gidney et al. (2021), the one published
//!   thresholds assume: DEPOLARIZE2(p) after every two-qubit gate,
//!   DEPOLARIZE1(p) after every single-qubit gate and on every idle qubit in a
//!   layer, flips of p after resets and before measurements, every round noisy.
//!
//! The CNOT schedules are the ones the exhaustive single-fault checks settled,
//! read from the same constants `round_program` uses, never copied.

use std::collections::HashMap;

use crate::circuit::{Basis, Circuit, Instr};
use crate::circuit_model::Op;
use crate::surface_code::{RotatedSurfaceCode, XZZXSurfaceCode};

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum NoiseModel {
    Current { p: f64, eta: f64 },
    Sd6 { p: f64 },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CodeKind {
    Rotated,
    Xzzx,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Gate2 {
    Cx(u32, u32),
    Cz(u32, u32),
}

impl Gate2 {
    fn qubits(&self) -> (u32, u32) {
        match *self {
            Gate2::Cx(a, b) | Gate2::Cz(a, b) => (a, b),
        }
    }
}

/// One syndrome-extraction round, as layers.
pub struct RoundLayers {
    /// Ancillas, in measurement order.
    pub ancillas: Vec<u32>,
    /// Ancillas rotated into the X basis before and after the entangling layers.
    pub hadamard: Vec<u32>,
    pub steps: [Vec<Gate2>; 4],
}

/// Everything a memory experiment needs to know about a patch.
pub struct Patch {
    pub d: usize,
    pub num_qubits: usize,
    pub num_data: usize,
    /// Data qubits, then ancillas in `layers.ancillas` order.
    pub qubit_coords: Vec<(usize, usize)>,
    pub layers: RoundLayers,
    /// `round_program`: where the Current model puts its noise.
    pub program: Vec<Op>,
    /// Data qubits of each ancilla's plaquette, by ancilla position.
    pub support: Vec<Vec<u32>>,
    /// Preparation and readout basis of each data qubit.
    pub data_basis: Vec<Basis>,
    /// Ancilla positions whose first reading the preparation fixes.
    pub deterministic: Vec<usize>,
    /// Data qubits whose readouts multiply to the logical observable.
    pub observable: Vec<u32>,
}

fn other(b: Basis) -> Basis {
    match b {
        Basis::X => Basis::Z,
        Basis::Z => Basis::X,
    }
}

/// Logical Z of the rotated code runs down a column: qubits 0, d, 2d, ...
/// (Z-type boundaries are top and bottom, so a Z string ends there.)
fn column(d: usize) -> Vec<u32> {
    (0..d).map(|i| (i * d) as u32).collect()
}

/// Logical X runs along a row: qubits 0..d.
fn row(d: usize) -> Vec<u32> {
    (0..d).map(|i| i as u32).collect()
}

pub fn rotated_layers(code: &RotatedSurfaceCode) -> RoundLayers {
    let n = code.data_qubits.len();
    let nx = code.x_stabilizers.len();
    let nz = code.z_stabilizers.len();
    let x_anc = |j: usize| (n + j) as u32;
    let z_anc = |k: usize| (n + nx + k) as u32;
    let mut steps: [Vec<Gate2>; 4] = Default::default();
    for (step, gates) in steps.iter_mut().enumerate() {
        let (dx, dy) = RotatedSurfaceCode::X_ORDER[step];
        for j in 0..nx {
            if let Some(q) = code.neighbor_at(&code.x_stabilizers[j], dx, dy) {
                gates.push(Gate2::Cx(x_anc(j), q as u32));
            }
        }
        let (dx, dy) = RotatedSurfaceCode::Z_ORDER[step];
        for k in 0..nz {
            if let Some(q) = code.neighbor_at(&code.z_stabilizers[k], dx, dy) {
                gates.push(Gate2::Cx(q as u32, z_anc(k)));
            }
        }
    }
    RoundLayers {
        ancillas: (0..nx).map(x_anc).chain((0..nz).map(z_anc)).collect(),
        hadamard: (0..nx).map(x_anc).collect(),
        steps,
    }
}

pub fn xzzx_layers(code: &XZZXSurfaceCode) -> RoundLayers {
    let n = code.data_qubits.len();
    let ns = code.stabilizers.len();
    let split = code.z_stabilizers.len();
    let anc = |s: usize| (n + s) as u32;
    let mut steps: [Vec<Gate2>; 4] = Default::default();
    for (step, gates) in steps.iter_mut().enumerate() {
        for s in 0..ns {
            let (dx, dy, is_x) =
                if s < split { XZZXSurfaceCode::SCHEDULE_A[step] } else { XZZXSurfaceCode::SCHEDULE_B[step] };
            let (sx, sy) = code.stabilizers[s];
            if let Some(q) = code.get_neighbor_idx(sx as i32 + dx, sy as i32 + dy) {
                gates.push(if is_x { Gate2::Cx(anc(s), q as u32) } else { Gate2::Cz(anc(s), q as u32) });
            }
        }
    }
    RoundLayers { ancillas: (0..ns).map(anc).collect(), hadamard: (0..ns).map(anc).collect(), steps }
}

pub fn rotated_patch(code: &RotatedSurfaceCode, basis: Basis) -> Patch {
    let d = code.d;
    let n = code.data_qubits.len();
    let nx = code.x_stabilizers.len();
    let nz = code.z_stabilizers.len();
    let stab_coords: Vec<(usize, usize)> =
        code.x_stabilizers.iter().chain(code.z_stabilizers.iter()).copied().collect();
    let support = stab_coords.iter().map(|s| code.get_neighbors(s).into_iter().map(|q| q as u32).collect()).collect();
    let (deterministic, observable) = match basis {
        Basis::Z => ((nx..nx + nz).collect(), column(d)),
        Basis::X => ((0..nx).collect(), row(d)),
    };
    Patch {
        d,
        num_qubits: n + nx + nz,
        num_data: n,
        qubit_coords: code.data_qubits.iter().copied().chain(stab_coords).collect(),
        layers: rotated_layers(code),
        program: code.round_program(),
        support,
        data_basis: vec![basis; n],
        deterministic,
        observable,
    }
}

/// Which data qubits carry a Hadamard when the XZZX code is read as the
/// rotated code conjugated by single-qubit Hadamards.
///
/// Derived, not assumed: at every plaquette the rotated code has Z (or X) on
/// every leg and XZZX has X on the NW/SE diagonal and Z on NE/SW. A leg differs
/// exactly where a Hadamard sits, and each qubit must get the same answer from
/// every plaquette it belongs to, or the equivalence does not hold and this
/// fails rather than guessing.
pub fn xzzx_hadamard_pattern(code: &XZZXSurfaceCode) -> Result<Vec<bool>, String> {
    let n = code.data_qubits.len();
    let split = code.z_stabilizers.len();
    let mut h: Vec<Option<bool>> = vec![None; n];
    for (s, &(sx, sy)) in code.stabilizers.iter().enumerate() {
        let rotated_is_z = s < split;
        for &(dx, dy, is_x) in &XZZXSurfaceCode::SCHEDULE_A {
            if let Some(q) = code.get_neighbor_idx(sx as i32 + dx, sy as i32 + dy) {
                let differs = is_x == rotated_is_z;
                match h[q] {
                    None => h[q] = Some(differs),
                    Some(prev) if prev != differs => {
                        return Err(format!("data qubit {q} needs a Hadamard for one plaquette and not another"));
                    }
                    _ => {}
                }
            }
        }
    }
    h.into_iter().enumerate().map(|(q, v)| v.ok_or_else(|| format!("data qubit {q} is in no plaquette"))).collect()
}

pub fn xzzx_patch(code: &XZZXSurfaceCode, basis: Basis) -> Result<Patch, String> {
    let d = code.d;
    let n = code.data_qubits.len();
    let ns = code.stabilizers.len();
    let split = code.z_stabilizers.len();
    let h = xzzx_hadamard_pattern(code)?;
    let support =
        code.stabilizers.iter().map(|s| code.get_neighbors(s).into_iter().map(|q| q as u32).collect()).collect();
    let (deterministic, observable) = match basis {
        Basis::Z => ((0..split).collect(), column(d)),
        Basis::X => ((split..ns).collect(), row(d)),
    };
    Ok(Patch {
        d,
        num_qubits: n + ns,
        num_data: n,
        qubit_coords: code.data_qubits.iter().chain(code.stabilizers.iter()).copied().collect(),
        layers: xzzx_layers(code),
        program: code.round_program(),
        support,
        data_basis: h.iter().map(|&flip| if flip { other(basis) } else { basis }).collect(),
        deterministic,
        observable,
    })
}

pub fn patch_for(kind: CodeKind, d: usize, basis: Basis) -> Result<Patch, String> {
    if !(2..=11).contains(&d) {
        return Err(format!("distance {d} is outside 2..=11"));
    }
    match kind {
        CodeKind::Rotated => Ok(rotated_patch(&RotatedSurfaceCode::new(d), basis)),
        CodeKind::Xzzx => xzzx_patch(&XZZXSurfaceCode::new(d), basis),
    }
}

pub fn generate(kind: CodeKind, d: usize, rounds: usize, noise: NoiseModel, basis: Basis) -> Result<Circuit, String> {
    if rounds == 0 {
        return Err("a memory experiment needs at least one round".into());
    }
    if rounds > MAX_FLAT_ROUNDS {
        return Err(format!(
            "{rounds} rounds: at most {MAX_FLAT_ROUNDS} are written out round by round"
        ));
    }
    match noise {
        NoiseModel::Current { p, eta } => {
            probability(p)?;
            if !(eta.is_finite() && eta >= 0.0) {
                return Err(format!("the bias {eta} is not a non-negative number"));
            }
        }
        NoiseModel::Sd6 { p } => probability(p)?,
    }
    Ok(memory_circuit(&patch_for(kind, d, basis)?, rounds, noise))
}

/// The most rounds `generate` writes out without a loop.
pub const MAX_FLAT_ROUNDS: usize = 10_000;

/// A noise strength must be a probability (NaN is not).
pub fn probability(p: f64) -> Result<(), String> {
    if (0.0..=1.0).contains(&p) {
        Ok(())
    } else {
        Err(format!("the noise strength {p} is outside [0, 1]"))
    }
}

pub fn memory_circuit(patch: &Patch, rounds: usize, noise: NoiseModel) -> Circuit {
    match noise {
        NoiseModel::Current { p, eta } => current_circuit(patch, rounds, p, eta),
        NoiseModel::Sd6 { p } => sd6_circuit(patch, rounds, p, false),
    }
}

/// The SD6 memory experiment with rounds 2 to `rounds` written once, as a
/// `REPEAT` block whose detectors move forward in time by `SHIFT_COORDS`.
/// Flattened, it is `generate`'s circuit exactly; unflattened, a million rounds
/// cost one round of instructions, which is how the batch sampler streams them.
pub fn generate_repeat(kind: CodeKind, d: usize, rounds: usize, p: f64, basis: Basis) -> Result<Circuit, String> {
    if rounds == 0 {
        return Err("a memory experiment needs at least one round".into());
    }
    probability(p)?;
    Ok(sd6_circuit(&patch_for(kind, d, basis)?, rounds, p, true))
}

/* -- Shared pieces --------------------------------------------------------- */

fn lookback(now: usize, abs: usize) -> u32 {
    (now - abs) as u32
}

fn coords_header(patch: &Patch, c: &mut Vec<Instr>) {
    for (q, &(x, y)) in patch.qubit_coords.iter().enumerate() {
        c.push(Instr::QubitCoords { coords: vec![x as f64, y as f64], qubits: vec![q as u32] });
    }
}

fn data_by_basis(patch: &Patch) -> (Vec<u32>, Vec<u32>) {
    let mut zs = Vec::new();
    let mut xs = Vec::new();
    for (q, &b) in patch.data_basis.iter().enumerate() {
        match b {
            Basis::Z => zs.push(q as u32),
            Basis::X => xs.push(q as u32),
        }
    }
    (zs, xs)
}

/// Reset the data in their bases, with a flip of `p` after (none when p = 0).
fn reset_data(patch: &Patch, c: &mut Vec<Instr>, p: f64) {
    let (zs, xs) = data_by_basis(patch);
    if !zs.is_empty() {
        c.push(Instr::Reset { basis: Basis::Z, qubits: zs.clone() });
        if p > 0.0 {
            c.push(Instr::PauliError { pauli: 1, p, qubits: zs });
        }
    }
    if !xs.is_empty() {
        c.push(Instr::Reset { basis: Basis::X, qubits: xs.clone() });
        if p > 0.0 {
            c.push(Instr::PauliError { pauli: 2, p, qubits: xs });
        }
    }
}

/// The transversal readout, its detectors against the last round, and the observable.
fn final_readout(patch: &Patch, c: &mut Vec<Instr>, m: &mut usize, last: &[usize], t: f64, p: f64) {
    let (zs, xs) = data_by_basis(patch);
    c.push(Instr::Tick);
    let mut at = vec![usize::MAX; patch.num_data];
    for (qs, basis, flip) in [(&zs, Basis::Z, 1u8), (&xs, Basis::X, 2u8)] {
        if qs.is_empty() {
            continue;
        }
        if p > 0.0 {
            c.push(Instr::PauliError { pauli: flip, p, qubits: qs.clone() });
        }
        c.push(Instr::Measure { basis, reset: false, flip: 0.0, qubits: qs.clone() });
        for &q in qs {
            at[q as usize] = *m;
            *m += 1;
        }
    }
    for &a in &patch.deterministic {
        let (x, y) = patch.qubit_coords[patch.layers.ancillas[a] as usize];
        let mut recs: Vec<u32> = patch.support[a].iter().map(|&q| lookback(*m, at[q as usize])).collect();
        recs.push(lookback(*m, last[a]));
        c.push(Instr::Detector { coords: vec![x as f64, y as f64, t], recs });
    }
    let recs = patch.observable.iter().map(|&q| lookback(*m, at[q as usize])).collect();
    c.push(Instr::Observable { index: 0, recs });
}

/* -- Current: the engine's own model, gate for gate ------------------------ */

fn current_circuit(patch: &Patch, rounds: usize, p: f64, eta: f64) -> Circuit {
    // sample_pauli in surface_code.rs: Z with eta/(eta+1), X and Y with
    // 1/(2(eta+1)) each, all scaled by p.
    let pz = p * eta / (eta + 1.0);
    let px = p / (2.0 * (eta + 1.0));
    let py = px;
    let na = patch.layers.ancillas.len();
    let pos: HashMap<u32, usize> = patch.layers.ancillas.iter().enumerate().map(|(i, &a)| (a, i)).collect();

    let mut c = Vec::new();
    coords_header(patch, &mut c);
    reset_data(patch, &mut c, 0.0);
    let mut m = 0usize;
    let mut prev: Option<Vec<usize>> = None;
    let total = rounds + 2;
    for r in 0..total {
        c.push(Instr::Tick);
        let noisy = r >= 1 && r <= rounds;
        let mut this = vec![usize::MAX; na];
        for &op in &patch.program {
            match op {
                Op::Reset(q) => c.push(Instr::Reset { basis: Basis::Z, qubits: vec![q as u32] }),
                Op::H(q) => c.push(Instr::H(vec![q as u32])),
                Op::Cnot(a, b) => c.push(Instr::Cx(vec![(a as u32, b as u32)])),
                Op::Cz(a, b) => c.push(Instr::Cz(vec![(a as u32, b as u32)])),
                Op::Noise(q) => {
                    if noisy && p > 0.0 {
                        c.push(Instr::PauliChannel1 { px, py, pz, qubits: vec![q as u32] });
                    }
                }
                Op::Measure(q, _, _) => {
                    let flip = if noisy { p } else { 0.0 };
                    c.push(Instr::Measure { basis: Basis::Z, reset: false, flip, qubits: vec![q as u32] });
                    this[pos[&(q as u32)]] = m;
                    m += 1;
                }
            }
        }
        if let Some(prev) = &prev {
            for a in 0..na {
                let (x, y) = patch.qubit_coords[patch.layers.ancillas[a] as usize];
                c.push(Instr::Detector {
                    coords: vec![x as f64, y as f64, r as f64],
                    recs: vec![lookback(m, this[a]), lookback(m, prev[a])],
                });
            }
        }
        prev = Some(this);
    }
    let last = prev.expect("at least one round");
    final_readout(patch, &mut c, &mut m, &last, total as f64, 0.0);
    Circuit { instrs: c }
}

/* -- SD6: the standard model ----------------------------------------------- */

fn sd6_circuit(patch: &Patch, rounds: usize, p: f64, fold: bool) -> Circuit {
    // Folded, rounds 2..rounds are one loop body: every round after the first
    // reads the same relative records, so writing round 2 once and repeating it
    // is exact. Its detectors sit at t = 2, moved on by SHIFT_COORDS each time.
    let fold = fold && rounds >= 2;
    let na = patch.layers.ancillas.len();
    let nq = patch.num_qubits;
    let data: Vec<u32> = (0..patch.num_data as u32).collect();
    let anc = patch.layers.ancillas.clone();
    let idle = |busy: &[u32]| -> Vec<u32> {
        let mut used = vec![false; nq];
        for &q in busy {
            used[q as usize] = true;
        }
        (0..nq as u32).filter(|&q| !used[q as usize]).collect()
    };
    let depol1 = |c: &mut Vec<Instr>, qubits: Vec<u32>| {
        if p > 0.0 && !qubits.is_empty() {
            c.push(Instr::Depolarize1 { p, qubits });
        }
    };
    let hadamard_layer = |c: &mut Vec<Instr>| {
        c.push(Instr::Tick);
        if !patch.layers.hadamard.is_empty() {
            c.push(Instr::H(patch.layers.hadamard.clone()));
            depol1(c, patch.layers.hadamard.clone());
        }
        depol1(c, idle(&patch.layers.hadamard));
    };

    let mut head = Vec::new();
    coords_header(patch, &mut head);
    let mut body = Vec::new();
    let mut m = 0usize;
    let mut prev: Option<Vec<usize>> = None;
    for r in 1..=(if fold { 2 } else { rounds }) {
        let c = if fold && r == 2 { &mut body } else { &mut head };
        // Reset: ancillas every round, data in the first; idle data otherwise.
        c.push(Instr::Tick);
        c.push(Instr::Reset { basis: Basis::Z, qubits: anc.clone() });
        if p > 0.0 {
            c.push(Instr::PauliError { pauli: 1, p, qubits: anc.clone() });
        }
        if r == 1 {
            reset_data(patch, c, p);
        } else {
            depol1(c, data.clone());
        }

        hadamard_layer(c);

        for step in &patch.layers.steps {
            c.push(Instr::Tick);
            let cx: Vec<(u32, u32)> = step
                .iter()
                .filter_map(|g| match *g {
                    Gate2::Cx(a, b) => Some((a, b)),
                    _ => None,
                })
                .collect();
            let cz: Vec<(u32, u32)> = step
                .iter()
                .filter_map(|g| match *g {
                    Gate2::Cz(a, b) => Some((a, b)),
                    _ => None,
                })
                .collect();
            if !cx.is_empty() {
                c.push(Instr::Cx(cx));
            }
            if !cz.is_empty() {
                c.push(Instr::Cz(cz));
            }
            let pairs: Vec<(u32, u32)> = step.iter().map(|g| g.qubits()).collect();
            if p > 0.0 && !pairs.is_empty() {
                c.push(Instr::Depolarize2 { p, pairs: pairs.clone() });
            }
            let busy: Vec<u32> = pairs.iter().flat_map(|&(a, b)| [a, b]).collect();
            depol1(c, idle(&busy));
        }

        hadamard_layer(c);

        // Measure the ancillas; the data idle meanwhile.
        c.push(Instr::Tick);
        if p > 0.0 {
            c.push(Instr::PauliError { pauli: 1, p, qubits: anc.clone() });
        }
        c.push(Instr::Measure { basis: Basis::Z, reset: false, flip: 0.0, qubits: anc.clone() });
        let this: Vec<usize> = (0..na).map(|a| m + a).collect();
        m += na;
        depol1(c, data.clone());

        let coords = |a: usize| {
            let (x, y) = patch.qubit_coords[patch.layers.ancillas[a] as usize];
            vec![x as f64, y as f64, r as f64]
        };
        match &prev {
            None => {
                for &a in &patch.deterministic {
                    c.push(Instr::Detector { coords: coords(a), recs: vec![lookback(m, this[a])] });
                }
            }
            Some(prev) => {
                for a in 0..na {
                    c.push(Instr::Detector {
                        coords: coords(a),
                        recs: vec![lookback(m, this[a]), lookback(m, prev[a])],
                    });
                }
            }
        }
        prev = Some(this);
    }
    let mut c = head;
    let (last, t_final) = if fold {
        body.push(Instr::ShiftCoords(vec![0.0, 0.0, 1.0]));
        c.push(Instr::Repeat { count: (rounds - 1) as u64, body });
        // Every round measured; the final readout's detectors are written at
        // t = 2, which the rounds - 1 shifts carry to rounds + 1.
        m = na * rounds;
        ((m - na..m).collect::<Vec<usize>>(), 2.0)
    } else {
        (prev.expect("at least one round"), (rounds + 1) as f64)
    };
    final_readout(patch, &mut c, &mut m, &last, t_final, p);
    Circuit { instrs: c }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::dem::Dem;
    use crate::dem_decoder::DemDecoder;

    /// Per-qubit projection of a round: for each qubit, the ops touching it, in
    /// order. Two rounds with equal projections are the same circuit up to
    /// reordering gates on disjoint qubits.
    fn projection(ops: &[String], touches: &[Vec<usize>], nq: usize) -> Vec<Vec<String>> {
        let mut out = vec![Vec::new(); nq];
        for (op, qs) in ops.iter().zip(touches) {
            for &q in qs {
                out[q].push(op.clone());
            }
        }
        out
    }

    fn describe_program(program: &[Op]) -> (Vec<String>, Vec<Vec<usize>>) {
        let mut ops = Vec::new();
        let mut touches = Vec::new();
        for op in program {
            let (s, t) = match *op {
                Op::Reset(q) => ("R".to_string(), vec![q]),
                Op::H(q) => ("H".to_string(), vec![q]),
                Op::Cnot(a, b) => (format!("CX {a} {b}"), vec![a, b]),
                Op::Cz(a, b) => (format!("CZ {a} {b}"), vec![a, b]),
                Op::Measure(q, _, _) => ("M".to_string(), vec![q]),
                Op::Noise(_) => continue,
            };
            ops.push(s);
            touches.push(t);
        }
        (ops, touches)
    }

    fn describe_layers(l: &RoundLayers) -> (Vec<String>, Vec<Vec<usize>>) {
        let mut ops = Vec::new();
        let mut touches = Vec::new();
        for &a in &l.ancillas {
            ops.push("R".to_string());
            touches.push(vec![a as usize]);
        }
        for &a in &l.hadamard {
            ops.push("H".to_string());
            touches.push(vec![a as usize]);
        }
        for step in &l.steps {
            for g in step {
                let (s, a, b) = match *g {
                    Gate2::Cx(a, b) => (format!("CX {a} {b}"), a, b),
                    Gate2::Cz(a, b) => (format!("CZ {a} {b}"), a, b),
                };
                ops.push(s);
                touches.push(vec![a as usize, b as usize]);
            }
        }
        for &a in &l.hadamard {
            ops.push("H".to_string());
            touches.push(vec![a as usize]);
        }
        for &a in &l.ancillas {
            ops.push("M".to_string());
            touches.push(vec![a as usize]);
        }
        (ops, touches)
    }

    #[test]
    fn layers_are_the_round_program() {
        for d in [3, 5, 7] {
            for kind in [CodeKind::Rotated, CodeKind::Xzzx] {
                let patch = patch_for(kind, d, Basis::Z).unwrap();
                let (a, ta) = describe_program(&patch.program);
                let (b, tb) = describe_layers(&patch.layers);
                assert_eq!(
                    projection(&a, &ta, patch.num_qubits),
                    projection(&b, &tb, patch.num_qubits),
                    "{kind:?} d = {d}"
                );
            }
        }
    }

    #[test]
    fn xzzx_is_the_rotated_code_under_a_checkerboard_of_hadamards() {
        for d in [3, 5, 7, 9] {
            let code = XZZXSurfaceCode::new(d);
            let h = xzzx_hadamard_pattern(&code).unwrap();
            for (q, &(x, y)) in code.data_qubits.iter().enumerate() {
                assert_eq!(h[q], ((x + y) / 2) % 2 == 1, "d = {d}, qubit {q} at ({x}, {y})");
            }
        }
    }

    pub(crate) fn all_circuits(ds: &[usize]) -> Vec<(String, Circuit)> {
        let mut out = Vec::new();
        for &d in ds {
            for kind in [CodeKind::Rotated, CodeKind::Xzzx] {
                for basis in [Basis::Z, Basis::X] {
                    for noise in [NoiseModel::Current { p: 0.003, eta: 0.5 }, NoiseModel::Sd6 { p: 0.003 }] {
                        let name = format!("{kind:?} {basis:?} {noise:?} d = {d}");
                        out.push((name, generate(kind, d, d, noise, basis).unwrap()));
                    }
                }
            }
        }
        out
    }

    #[test]
    fn every_memory_circuit_has_a_deterministic_decomposable_model() {
        for (name, c) in all_circuits(&[3, 5]) {
            let dem = Dem::from_circuit(&c).unwrap_or_else(|e| panic!("{name}: {e}"));
            for m in &dem.mechanisms {
                assert!(!m.pieces.is_empty(), "{name}: {:?} undecomposed", m.detectors);
                let mut dets: Vec<u32> = Vec::new();
                let mut obs = 0u64;
                for piece in &m.pieces {
                    assert!((1..=2).contains(&piece.detectors.len()), "{name}: piece {:?}", piece.detectors);
                    for &d in &piece.detectors {
                        match dets.iter().position(|&x| x == d) {
                            Some(i) => {
                                dets.swap_remove(i);
                            }
                            None => dets.push(d),
                        }
                    }
                    obs ^= piece.observables;
                }
                dets.sort_unstable();
                assert_eq!((dets, obs), (m.detectors.clone(), m.observables), "{name}: pieces do not XOR back");
            }
            assert_eq!(Circuit::parse(&c.to_stim()).unwrap(), c, "{name}: text round trip");
        }
    }

    #[test]
    fn detector_counts_follow_the_conventions() {
        let d = 3;
        let na = 8; // 4 X plaquettes, 4 Z plaquettes
        let det = 4;
        let current = generate(CodeKind::Rotated, d, d, NoiseModel::Current { p: 0.003, eta: 0.5 }, Basis::Z).unwrap();
        assert_eq!(current.resolve().unwrap().detectors.len(), (d + 1) * na + det);
        let sd6 = generate(CodeKind::Rotated, d, d, NoiseModel::Sd6 { p: 0.003 }, Basis::Z).unwrap();
        // Stim's own rotated_memory_z at d = 3, rounds = 3 also has 24.
        assert_eq!(sd6.resolve().unwrap().detectors.len(), 2 * det + (d - 1) * na);
    }

    /// A CSS code's X-type and Z-type checks see disjoint error families, so no
    /// graph-like piece may join them. One that does is a bridge between the two
    /// matching graphs, and the matcher will route through it: a Y fault on a
    /// boundary qubit, kept whole as an edge from an X check to a Z check, let the
    /// d = 3 memory-X circuit under SD6 decode two of its single faults into
    /// logical errors, where PyMatching on Stim's decomposition decoded none.
    #[test]
    fn rotated_pieces_never_join_x_and_z_checks() {
        for d in [3, 5] {
            for basis in [Basis::Z, Basis::X] {
                for noise in [NoiseModel::Current { p: 0.003, eta: 0.5 }, NoiseModel::Sd6 { p: 0.003 }] {
                    let code = RotatedSurfaceCode::new(d);
                    let dem = Dem::from_circuit(&generate(CodeKind::Rotated, d, d, noise, basis).unwrap()).unwrap();
                    let is_x = |det: u32| {
                        let c = &dem.detector_coords[det as usize];
                        code.x_stabilizers.contains(&(c[0] as usize, c[1] as usize))
                    };
                    for m in &dem.mechanisms {
                        for piece in &m.pieces {
                            let kinds: Vec<bool> = piece.detectors.iter().map(|&x| is_x(x)).collect();
                            assert!(kinds.windows(2).all(|w| w[0] == w[1]), "{basis:?} {noise:?} d = {d}: piece {:?} joins X and Z checks", piece.detectors);
                        }
                    }
                }
            }
        }
    }

    /// Check 6 of the spec: a distance-d circuit must survive any one fault.
    /// Every mechanism of the model, decoded alone, must predict its own flip.
    pub(crate) fn single_fault_failures(c: &Circuit) -> (usize, usize) {
        let dem = Dem::from_circuit(c).unwrap();
        let dec = DemDecoder::new(&dem).unwrap();
        let mut failures = 0;
        for m in &dem.mechanisms {
            match dec.decode(&m.detectors) {
                Ok(pred) if pred.observables == m.observables => {}
                _ => failures += 1,
            }
        }
        (dem.mechanisms.len(), failures)
    }

    #[test]
    fn every_single_fault_is_corrected_d3_d5() {
        for (name, c) in all_circuits(&[3, 5]) {
            let (tested, failed) = single_fault_failures(&c);
            assert_eq!(failed, 0, "{name}: {failed} of {tested} single faults fail");
        }
    }

    #[test]
    #[ignore] // ~1 min; run in the verification step.
    fn every_single_fault_is_corrected_d7() {
        for (name, c) in all_circuits(&[7]) {
            let (tested, failed) = single_fault_failures(&c);
            println!("{name}: {tested} mechanisms, {failed} fail");
            assert_eq!(failed, 0, "{name}");
        }
    }

    #[test]
    fn the_folded_circuit_is_the_flat_one() {
        for kind in [CodeKind::Rotated, CodeKind::Xzzx] {
            for basis in [Basis::Z, Basis::X] {
                for d in [3usize, 5] {
                    for rounds in [1usize, 2, 3, 7] {
                        let folded = generate_repeat(kind, d, rounds, 0.004, basis).unwrap();
                        let flat = generate(kind, d, rounds, NoiseModel::Sd6 { p: 0.004 }, basis).unwrap();
                        assert_eq!(folded.flattened(), flat, "{kind:?} {basis:?} d = {d}, {rounds} rounds");
                        if rounds >= 3 {
                            assert!(folded.instrs.len() < flat.instrs.len());
                        }
                    }
                }
            }
        }
    }
}


//! Sampling circuits with coherent errors: the Pauli twirl as an importance distribution,
//! reweighted shot by shot by the interference the twirl throws away.
//!
//! A coherent location applies exp(−iθP) = cos θ − i sin θ P. Expanding every location, the
//! circuit is a sum over fault sets F with amplitudes a_F = Π_{k∈F}(−i sin θ_k) Π_{k∉F} cos θ_k.
//! The twirl keeps only the diagonal |a_F|² — exactly what the frame sampler draws when location
//! k fires P_k with probability sin²θ_k. The true probability of a measurement record also has
//! cross terms between fault sets F and F Δ G that no measurement can tell apart: G must flip
//! no measurement, counting a reset and the end of the circuit as measurements nobody reads.
//! Such sets G form a linear space over GF(2), the circuit's spacetime kernel. Each cross term
//! is, relative to the twirl's probability of the shot,
//!
//! ```text
//!     Π_{g∈G∖F} (i tan θ_g) · Π_{g∈G∩F} (−i / tan θ_g) · (−1)^{Σ_g a_g} · conj(σ_G) · (−1)^{m′·z_G}
//! ```
//!
//! where a_g is whether P_g anticommutes with the shot's faults before g, m′ is the shot's
//! noiseless-branch outcomes (its record with the faults' flips taken out, plus the outcomes
//! nobody reads) and σ_G, z_G come from carrying the operator of G through the circuit: what is
//! left of it is σ_G times Z on the outcomes in z_G. The shot's weight is 1 plus the cross terms
//! (each pair counted from its likelier side), and the weighted shots are distributed as the
//! coherent circuit's. `exact` checks that identity exhaustively on small circuits.

pub mod exact;
pub mod kernel;
pub mod sampler;
mod shot;

pub use shot::Shot;

use crate::circuit::{Basis, Circuit, Instr};
use crate::nonpauli::NonPauli;
use crate::simulator::StabilizerSimulator;
use crate::statevec::C;

/// A Pauli code: X 1, Z 2, Y 3.
type Code = u8;

/// One step of a compiled circuit, one target each.
#[derive(Clone, Debug)]
pub(crate) enum Step {
    H(usize),
    S(usize),
    /// A Pauli gate, which only signs an operator carried through it.
    Pauli(usize, Code),
    Cx(usize, usize),
    Cz(usize, usize),
    /// A measurement into outcome `slot`; with `reset`, then a reset (whose unread outcome is
    /// the measurement's).
    Measure { q: usize, basis: Basis, reset: bool, flip: f64, slot: usize },
    /// A reset, its unread outcome `slot`.
    Reset { q: usize, basis: Basis, slot: usize },
    PauliError { q: usize, pauli: Code, p: f64 },
    Depolarize1 { q: usize, p: f64 },
    PauliChannel1 { q: usize, px: f64, py: f64, pz: f64 },
    Depolarize2 { a: usize, b: usize, p: f64 },
    PauliChannel2 { a: usize, b: usize, probs: Vec<f64> },
    Correlated { p: f64, paulis: Vec<(usize, Code)>, chained: bool },
    Herald { q: usize, probs: [f64; 4], slot: usize },
    Pad { flip: f64, value: bool, slot: usize },
    /// Coherent location `loc`.
    Coherent(usize),
}

/// A coherent rotation exp(−iθP).
#[derive(Clone, Debug, PartialEq)]
pub struct Location {
    pub paulis: Vec<(usize, Code)>,
    pub theta: f64,
}

/// What an outcome slot is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Slot {
    /// Measurement record `k`.
    Record(usize),
    /// An outcome nobody reads: a reset's, or a qubit's at the end of the circuit.
    Hidden,
}

/// A circuit compiled for the coherent sampler.
#[derive(Clone, Debug)]
pub struct Program {
    pub(crate) steps: Vec<Step>,
    pub(crate) num_qubits: usize,
    pub(crate) slots: Vec<Slot>,
    pub(crate) num_records: usize,
    pub locations: Vec<Location>,
    /// With merging on, each location's place in its group: the groups' members are drawn as
    /// one, the parity at the group's last location (see `set_merge`).
    pub(crate) merge: Vec<Option<Merge>>,
    pub(crate) merge_groups: Vec<Vec<usize>>,
    pub(crate) detectors: Vec<Vec<usize>>,
    pub(crate) observables: Vec<Vec<usize>>,
    pub(crate) gauges: Vec<Gauge>,
    /// Per step, the gauge right after its measurement and the one right after its reset.
    pub(crate) gauge_at: Vec<[Option<usize>; 2]>,
    /// A noiseless reference run's value of every slot (the last `num_qubits` are the end).
    pub(crate) reference: Vec<bool>,
}

impl Program {
    pub fn new(circuit: &Circuit) -> Result<Program, String> {
        let res = circuit.resolve()?;
        let nq = res.num_qubits.max(
            res.instrs
                .iter()
                .filter_map(|i| match i {
                    Instr::NonPauli(ops) => ops.iter().filter_map(|o| match o {
                        NonPauli::Rotation { pauli, .. } => pauli.iter().map(|&(q, _)| q as usize + 1).max(),
                        _ => None,
                    }).max(),
                    _ => None,
                })
                .max()
                .unwrap_or(0),
        );
        let mut steps = Vec::new();
        let mut slots = Vec::new();
        let mut locations = Vec::new();
        let mut records = 0usize;
        let mut record = |slots: &mut Vec<Slot>| {
            slots.push(Slot::Record(records));
            records += 1;
            slots.len() - 1
        };
        for ins in &res.instrs {
            match ins {
                Instr::Reset { basis, qubits } => {
                    for &q in qubits {
                        slots.push(Slot::Hidden);
                        steps.push(Step::Reset { q: q as usize, basis: *basis, slot: slots.len() - 1 });
                    }
                }
                Instr::H(qs) => steps.extend(qs.iter().map(|&q| Step::H(q as usize))),
                Instr::S(qs) => steps.extend(qs.iter().map(|&q| Step::S(q as usize))),
                Instr::Cx(ps) => steps.extend(ps.iter().map(|&(a, b)| Step::Cx(a as usize, b as usize))),
                Instr::Cz(ps) => steps.extend(ps.iter().map(|&(a, b)| Step::Cz(a as usize, b as usize))),
                Instr::Pauli { pauli, qubits } => {
                    if *pauli != 0 {
                        steps.extend(qubits.iter().map(|&q| Step::Pauli(q as usize, *pauli)));
                    }
                }
                Instr::Measure { basis, reset, flip, qubits } => {
                    for &q in qubits {
                        let slot = record(&mut slots);
                        steps.push(Step::Measure { q: q as usize, basis: *basis, reset: *reset, flip: *flip, slot });
                    }
                }
                Instr::PauliError { pauli, p, qubits } => steps.extend(qubits.iter().map(|&q| Step::PauliError { q: q as usize, pauli: *pauli, p: *p })),
                Instr::Depolarize1 { p, qubits } => steps.extend(qubits.iter().map(|&q| Step::Depolarize1 { q: q as usize, p: *p })),
                Instr::PauliChannel1 { px, py, pz, qubits } => {
                    steps.extend(qubits.iter().map(|&q| Step::PauliChannel1 { q: q as usize, px: *px, py: *py, pz: *pz }))
                }
                Instr::Depolarize2 { p, pairs } => steps.extend(pairs.iter().map(|&(a, b)| Step::Depolarize2 { a: a as usize, b: b as usize, p: *p })),
                Instr::PauliChannel2 { probs, pairs } => {
                    steps.extend(pairs.iter().map(|&(a, b)| Step::PauliChannel2 { a: a as usize, b: b as usize, probs: probs.clone() }))
                }
                Instr::Correlated { p, paulis, chained } => {
                    steps.push(Step::Correlated { p: *p, paulis: paulis.iter().map(|&(q, c)| (q as usize, c)).collect(), chained: *chained })
                }
                Instr::Heralded { probs, qubits, .. } => {
                    for &q in qubits {
                        let slot = record(&mut slots);
                        steps.push(Step::Herald { q: q as usize, probs: *probs, slot });
                    }
                }
                Instr::Pad { flip, values } => {
                    for &value in values {
                        let slot = record(&mut slots);
                        steps.push(Step::Pad { flip: *flip, value, slot });
                    }
                }
                Instr::NonPauli(ops) => {
                    for op in ops {
                        match op {
                            NonPauli::Rotation { pauli, theta } => {
                                if pauli.is_empty() {
                                    continue;
                                }
                                steps.push(Step::Coherent(locations.len()));
                                locations.push(Location { paulis: pauli.iter().map(|&(q, c)| (q as usize, c)).collect(), theta: *theta });
                            }
                            other => return Err(format!("the coherent sampler takes rotations only, not '{}'; the state vector runs it", other.line())),
                        }
                    }
                }
                Instr::Feedback { .. } | Instr::SweepX(_) => {
                    return Err("the coherent sampler does not take feedback or sweep bits".into());
                }
                Instr::Observable { paulis, .. } if !paulis.is_empty() => {
                    return Err("the coherent sampler does not take observables of Pauli targets".into());
                }
                Instr::Detector { .. } | Instr::Observable { .. } | Instr::QubitCoords { .. } | Instr::ShiftCoords(_) | Instr::Tick => {}
                Instr::Repeat { .. } | Instr::Gate { .. } => unreachable!("resolve flattens loops and gates"),
            }
        }
        for _ in 0..nq {
            slots.push(Slot::Hidden);
        }
        let mut p = Program {
            steps,
            num_qubits: nq,
            slots,
            num_records: records,
            locations,
            detectors: res.detectors,
            observables: res.observables,
            gauges: Vec::new(),
            gauge_at: Vec::new(),
            merge: Vec::new(),
            merge_groups: Vec::new(),
            reference: Vec::new(),
        };
        p.reference = p.reference_run();
        p.build_gauges();
        Ok(p)
    }

    pub fn num_detectors(&self) -> usize {
        self.detectors.len()
    }

    pub fn num_observables(&self) -> usize {
        self.observables.len()
    }

    /// The slot of qubit q's outcome at the end.
    pub(crate) fn end_slot(&self, q: usize) -> usize {
        self.slots.len() - self.num_qubits + q
    }

    /// A noiseless run's value of every slot.
    fn reference_run(&self) -> Vec<bool> {
        let mut sim = StabilizerSimulator::with_seed(self.num_qubits.max(1), 0x5eed_c0e7);
        let mut out = vec![false; self.slots.len()];
        let measure = |sim: &mut StabilizerSimulator, q: usize, basis: Basis| match basis {
            Basis::Z => sim.measure_z(q) == 1,
            Basis::X => sim.measure_x(q) == 1,
        };
        let fix = |sim: &mut StabilizerSimulator, q: usize, basis: Basis, v: bool| {
            if v {
                match basis {
                    Basis::Z => sim.apply_x(q),
                    Basis::X => sim.apply_z(q),
                }
            }
        };
        for step in &self.steps {
            match *step {
                Step::H(q) => sim.apply_h(q),
                Step::S(q) => sim.apply_s(q),
                Step::Pauli(q, c) => match c {
                    1 => sim.apply_x(q),
                    2 => sim.apply_z(q),
                    _ => sim.apply_y(q),
                },
                Step::Cx(a, b) => sim.apply_cnot(a, b),
                Step::Cz(a, b) => {
                    sim.apply_h(b);
                    sim.apply_cnot(a, b);
                    sim.apply_h(b);
                }
                Step::Measure { q, basis, reset, slot, .. } => {
                    let v = measure(&mut sim, q, basis);
                    out[slot] = v;
                    if reset {
                        fix(&mut sim, q, basis, v);
                    }
                }
                Step::Reset { q, basis, slot } => {
                    let v = measure(&mut sim, q, basis);
                    out[slot] = v;
                    fix(&mut sim, q, basis, v);
                }
                Step::Pad { value, slot, .. } => out[slot] = value,
                _ => {}
            }
        }
        for q in 0..self.num_qubits {
            out[self.end_slot(q)] = sim.measure_z(q) == 1;
        }
        out
    }

    /// The frame's randomisation points, in the order the frame sampler draws them: a Z on
    /// every qubit at the start, and after each measurement and each reset the measured basis's
    /// Pauli. Each is a stabiliser of the noiseless state (after a measurement, times the
    /// outcome's Z), so the noiseless branches of a shot are the reference branch moved by them.
    fn build_gauges(&mut self) {
        let mut gauges = Vec::new();
        let mut at = vec![[None, None]; self.steps.len()];
        for q in 0..self.num_qubits {
            gauges.push(Gauge { q, pauli: 2, slot: None });
        }
        let code = |b: Basis| if b == Basis::Z { 2 } else { 1 };
        for (i, step) in self.steps.iter().enumerate() {
            match *step {
                Step::Measure { q, basis, reset, slot, .. } => {
                    at[i][0] = Some(gauges.len());
                    gauges.push(Gauge { q, pauli: code(basis), slot: Some(slot) });
                    if reset {
                        at[i][1] = Some(gauges.len());
                        gauges.push(Gauge { q, pauli: code(basis), slot: None });
                    }
                }
                Step::Reset { q, basis, .. } => {
                    at[i][1] = Some(gauges.len());
                    gauges.push(Gauge { q, pauli: code(basis), slot: None });
                }
                _ => {}
            }
        }
        self.gauges = gauges;
        self.gauge_at = at;
    }

    /// The slots each insertion flips: locations first (by index), then gauges, 64 at a time.
    pub(crate) fn flips(&self) -> Vec<Vec<usize>> {
        let (nl, ng) = (self.locations.len(), self.gauges.len());
        let n = nl + ng;
        let mut out = vec![Vec::new(); n];
        for chunk in (0..n).step_by(64) {
            let mut x = vec![0u64; self.num_qubits];
            let mut z = vec![0u64; self.num_qubits];
            let lane = |id: usize| if id >= chunk && id < chunk + 64 { 1u64 << (id - chunk) } else { 0 };
            let insert = |x: &mut [u64], z: &mut [u64], q: usize, c: u8, bit: u64| {
                if c & 1 != 0 {
                    x[q] ^= bit;
                }
                if c & 2 != 0 {
                    z[q] ^= bit;
                }
            };
            let note = |words: u64, slot: usize, out: &mut Vec<Vec<usize>>| {
                let mut w = words;
                while w != 0 {
                    out[chunk + w.trailing_zeros() as usize].push(slot);
                    w &= w - 1;
                }
            };
            for q in 0..self.num_qubits {
                insert(&mut x, &mut z, q, 2, lane(nl + q));
            }
            for (i, step) in self.steps.iter().enumerate() {
                match *step {
                    Step::H(q) => std::mem::swap(&mut x[q], &mut z[q]),
                    Step::S(q) => z[q] ^= x[q],
                    Step::Cx(c, t) => {
                        x[t] ^= x[c];
                        z[c] ^= z[t];
                    }
                    Step::Cz(a, b) => {
                        z[b] ^= x[a];
                        z[a] ^= x[b];
                    }
                    Step::Measure { q, basis, slot, .. } | Step::Reset { q, basis, slot } => {
                        note(if basis == Basis::Z { x[q] } else { z[q] }, slot, &mut out);
                        if let Some(g) = self.gauge_at[i][0] {
                            insert(&mut x, &mut z, q, self.gauges[g].pauli, lane(nl + g));
                        }
                        if reset_of(step) {
                            x[q] = 0;
                            z[q] = 0;
                        }
                        if let Some(g) = self.gauge_at[i][1] {
                            insert(&mut x, &mut z, q, self.gauges[g].pauli, lane(nl + g));
                        }
                    }
                    Step::Coherent(loc) => {
                        let bit = lane(loc);
                        if bit != 0 {
                            for &(q, c) in &self.locations[loc].paulis {
                                insert(&mut x, &mut z, q, c, bit);
                            }
                        }
                    }
                    _ => {}
                }
            }
            for q in 0..self.num_qubits {
                note(x[q], self.end_slot(q), &mut out);
            }
        }
        out
    }

    /// The locations after `l` whose Pauli anticommutes with `l`'s carried to them.
    pub(crate) fn anticommuting_after(&self, l: usize) -> Vec<usize> {
        let (mut x, mut z) = (vec![false; self.num_qubits], vec![false; self.num_qubits]);
        let mut out = Vec::new();
        let mut started = false;
        for step in &self.steps {
            match *step {
                Step::Coherent(loc) if loc == l => {
                    for &(q, c) in &self.locations[l].paulis {
                        x[q] ^= c & 1 != 0;
                        z[q] ^= c & 2 != 0;
                    }
                    started = true;
                }
                _ if !started => {}
                Step::H(q) => std::mem::swap(&mut x[q], &mut z[q]),
                Step::S(q) => z[q] ^= x[q],
                Step::Cx(c, t) => {
                    let (a, b) = (x[c], z[t]);
                    x[t] ^= a;
                    z[c] ^= b;
                }
                Step::Cz(a, b) => {
                    let (xa, xb) = (x[a], x[b]);
                    z[b] ^= xa;
                    z[a] ^= xb;
                }
                Step::Measure { q, reset: true, .. } | Step::Reset { q, .. } => {
                    x[q] = false;
                    z[q] = false;
                }
                Step::Coherent(loc) => {
                    let anti = self.locations[loc].paulis.iter().fold(false, |a, &(q, c)| a ^ (c & 1 != 0 && z[q]) ^ (c & 2 != 0 && x[q]));
                    if anti {
                        out.push(loc);
                    }
                }
                _ => {}
            }
        }
        out
    }

    /// The kernel element of the locations `g` (sorted): `None` unless what they flip is what
    /// some set of gauges flips. Carrying the locations' operator and those gauges' through the
    /// circuit, nothing is flipped and what is left is a phase times Z on unread outcomes; with
    /// the reordering that puts the gauges first, that gives the element's factor.
    pub(crate) fn element(&self, g: &[usize], solver: &GaugeSolver) -> Option<Element> {
        let mut flipped = vec![0u64; self.slots.len().div_ceil(64)];
        for &l in g {
            for &s in &solver.flips[l] {
                flipped[s / 64] ^= 1 << (s % 64);
            }
        }
        let j = solver.solve(&flipped)?;
        let mut in_j = vec![false; self.gauges.len()];
        for &k in &j {
            in_j[k] = true;
        }
        let mut e = Signed::new(self.num_qubits);
        // The gauges alone, unsigned, to count the swaps that move them before the locations.
        let (mut jx, mut jz) = (vec![false; self.num_qubits], vec![false; self.num_qubits]);
        let mut swaps = false;
        let mut zs: Vec<usize> = Vec::new();
        let gauge = |k: usize, e: &mut Signed, jx: &mut [bool], jz: &mut [bool], zs: &mut Vec<usize>| {
            if in_j[k] {
                let ga = &self.gauges[k];
                e.left_multiply(&[(ga.q, ga.pauli)]);
                jx[ga.q] ^= ga.pauli & 1 != 0;
                jz[ga.q] ^= ga.pauli & 2 != 0;
                if let Some(s) = ga.slot {
                    zs.push(s);
                }
            }
        };
        for q in 0..self.num_qubits {
            gauge(q, &mut e, &mut jx, &mut jz, &mut zs);
        }
        let mut next = 0usize;
        for (i, step) in self.steps.iter().enumerate() {
            match *step {
                Step::H(q) => {
                    e.h(q);
                    std::mem::swap(&mut jx[q], &mut jz[q]);
                }
                Step::S(q) => {
                    e.s(q);
                    jz[q] ^= jx[q];
                }
                Step::Pauli(q, c) => e.sign_by(q, c),
                Step::Cx(c, t) => {
                    e.cx(c, t);
                    let (a, b) = (jx[c], jz[t]);
                    jx[t] ^= a;
                    jz[c] ^= b;
                }
                Step::Cz(a, b) => {
                    e.cz(a, b);
                    let (xa, xb) = (jx[a], jx[b]);
                    jz[b] ^= xa;
                    jz[a] ^= xb;
                }
                Step::Measure { q, basis, slot, .. } | Step::Reset { q, basis, slot } => {
                    if e.flips(q, basis) {
                        return None;
                    }
                    if let Some(k) = self.gauge_at[i][0] {
                        gauge(k, &mut e, &mut jx, &mut jz, &mut zs);
                    }
                    if reset_of(step) {
                        if e.absorb(q, basis) {
                            zs.push(slot);
                        }
                        jx[q] = false;
                        jz[q] = false;
                    }
                    if let Some(k) = self.gauge_at[i][1] {
                        gauge(k, &mut e, &mut jx, &mut jz, &mut zs);
                    }
                }
                Step::Coherent(loc) => {
                    if next < g.len() && g[next] == loc {
                        let paulis = &self.locations[loc].paulis;
                        swaps ^= paulis.iter().fold(false, |a, &(q, c)| a ^ (c & 1 != 0 && jz[q]) ^ (c & 2 != 0 && jx[q]));
                        e.left_multiply(paulis);
                        next += 1;
                    }
                }
                _ => {}
            }
        }
        for q in 0..self.num_qubits {
            if e.flips(q, Basis::Z) {
                return None;
            }
            if e.absorb(q, Basis::Z) {
                zs.push(self.end_slot(q));
            }
        }
        // Gauges first: W = (−1)^swaps i^k Z_zs. The cross term needs conj of ⟨ψ_{m'}|E|ψ_{m''}⟩,
        // evaluated from the partner's branch m'' = m' ⊕ flips: conj(λ) (−1)^{m'·z} (−1)^{flips·z}.
        zs.sort_unstable();
        let mut z: Vec<usize> = Vec::new();
        for s in zs {
            if z.last() == Some(&s) {
                z.pop();
            } else {
                z.push(s);
            }
        }
        let k = (4 - e.k + if swaps { 2 } else { 0 }) % 4;
        let offset = z.iter().fold(false, |a, &s| a ^ (flipped[s / 64] >> (s % 64) & 1 == 1));
        Some(Element { members: g.to_vec(), k, z, offset })
    }
}

/// A location's place in a merged group.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Merge {
    pub group: usize,
    /// Its operator relative to the group's first member's.
    pub mu: C,
    pub last: bool,
}

/// The share of the coherent parity in a merged group's proposal; the rest is the twirl's, so
/// no parity is ever impossible.
const COHERENT_SHARE: f64 = 0.9;

impl Program {
    /// Draw each group (locations that flip the same outcomes, `groups[i].0` with phases
    /// `groups[i].1`) as one: its parity from the coherent sum of its rotations, the fault (if
    /// any) at its last location. A far better proposal than the twirl when rotations add up.
    pub fn set_merge(&mut self, groups: &[(Vec<usize>, Vec<C>)]) {
        self.merge = vec![None; self.locations.len()];
        self.merge_groups = groups.iter().map(|g| g.0.clone()).collect();
        for (i, (members, mu)) in groups.iter().enumerate() {
            for (k, (&l, &m)) in members.iter().zip(mu).enumerate() {
                self.merge[l] = Some(Merge { group: i, mu: m, last: k + 1 == members.len() });
            }
        }
    }

    /// A group member's toggle factor with nothing of the group fired: i t μ, signed.
    pub(crate) fn unfired_toggle(&self, l: usize, mu: C, anti: bool) -> C {
        let v = C::new(0.0, self.locations[l].theta.tan()) * mu;
        if anti { C::ZERO - v } else { v }
    }

    /// The proposal's probability of odd parity for a group, from its members' toggles with
    /// nothing fired: the coherent share |S₁|² / (|S₀|² + |S₁|²), mixed with the twirl's.
    pub(crate) fn odd_proposal(taus: &[C]) -> f64 {
        let (mut plus, mut minus, mut nplus, mut nminus) = (C::ONE, C::ONE, 1.0, 1.0);
        for &t in taus {
            plus = plus * (C::ONE + t);
            minus = minus * (C::ONE - t);
            nplus *= 1.0 + t.norm2();
            nminus *= 1.0 - t.norm2();
        }
        let (s0, s1) = ((plus + minus).scale(0.5).norm2(), (plus - minus).scale(0.5).norm2());
        let coherent = if s0 + s1 > 0.0 { s1 / (s0 + s1) } else { 0.0 };
        let twirl = (nplus - nminus) / (nplus + nminus);
        COHERENT_SHARE * coherent + (1.0 - COHERENT_SHARE) * twirl
    }
}

/// Whether a step resets its qubit.
fn reset_of(step: &Step) -> bool {
    matches!(step, Step::Reset { .. } | Step::Measure { reset: true, .. })
}

/// A point where the frame sampler randomises: the Pauli `pauli` on qubit `q`, and for one
/// right after a measurement, that measurement's outcome slot (whose Z it carries).
#[derive(Clone, Debug)]
pub(crate) struct Gauge {
    pub q: usize,
    pub pauli: Code,
    pub slot: Option<usize>,
}

/// Which gauges flip what a set of locations flips: elimination over the gauges' flips.
pub struct GaugeSolver {
    /// Each insertion's flips (locations, then gauges).
    pub flips: Vec<Vec<usize>>,
    /// Reduced rows: (pivot slot, flips as words, the gauges summed).
    rows: Vec<(usize, Vec<u64>, Vec<u64>)>,
}

impl GaugeSolver {
    pub fn new(p: &Program) -> GaugeSolver {
        let flips = p.flips();
        let words = p.slots.len().div_ceil(64);
        let gw = p.gauges.len().div_ceil(64);
        let nl = p.locations.len();
        let mut rows: Vec<(usize, Vec<u64>, Vec<u64>)> = Vec::new();
        for g in 0..p.gauges.len() {
            let mut v = vec![0u64; words];
            for &s in &flips[nl + g] {
                v[s / 64] ^= 1 << (s % 64);
            }
            let mut c = vec![0u64; gw];
            c[g / 64] |= 1 << (g % 64);
            for (pivot, rv, rc) in &rows {
                if v[pivot / 64] >> (pivot % 64) & 1 == 1 {
                    v.iter_mut().zip(rv).for_each(|(a, b)| *a ^= b);
                    c.iter_mut().zip(rc).for_each(|(a, b)| *a ^= b);
                }
            }
            if let Some(w) = v.iter().position(|&w| w != 0) {
                let pivot = w * 64 + v[w].trailing_zeros() as usize;
                // Keep the rows reduced at every pivot.
                for (_, rv, rc) in rows.iter_mut() {
                    if rv[pivot / 64] >> (pivot % 64) & 1 == 1 {
                        rv.iter_mut().zip(&v).for_each(|(a, b)| *a ^= b);
                        rc.iter_mut().zip(&c).for_each(|(a, b)| *a ^= b);
                    }
                }
                rows.push((pivot, v, c));
            }
        }
        GaugeSolver { flips, rows }
    }

    /// `v` reduced by the gauges' flips: zero exactly when the gauges can match it.
    pub fn reduce(&self, v: &mut [u64]) {
        for (pivot, rv, _) in &self.rows {
            if v[pivot / 64] >> (pivot % 64) & 1 == 1 {
                v.iter_mut().zip(rv).for_each(|(a, b)| *a ^= b);
            }
        }
    }

    /// Gauges whose flips sum to `target`, if any.
    pub fn solve(&self, target: &[u64]) -> Option<Vec<usize>> {
        let mut v = target.to_vec();
        let mut c = vec![0u64; self.rows.first().map_or(0, |r| r.2.len())];
        for (pivot, rv, rc) in &self.rows {
            if v[pivot / 64] >> (pivot % 64) & 1 == 1 {
                v.iter_mut().zip(rv).for_each(|(a, b)| *a ^= b);
                c.iter_mut().zip(rc).for_each(|(a, b)| *a ^= b);
            }
        }
        if v.iter().any(|&w| w != 0) {
            return None;
        }
        Some((0..c.len() * 64).filter(|&g| c[g / 64] >> (g % 64) & 1 == 1).collect())
    }
}

/// An element of the spacetime kernel: its locations (sorted), and what carrying their
/// operator through the circuit leaves, i^k times Z on the unread outcomes `z`.
#[derive(Clone, Debug, PartialEq)]
pub struct Element {
    pub members: Vec<usize>,
    /// The factor's power of i.
    pub k: u8,
    /// The outcomes whose noiseless-branch values sign it.
    pub z: Vec<usize>,
    /// A fixed sign besides.
    pub offset: bool,
}

impl Program {
    /// Kernel element `e`'s fault set's amplitude relative to the shot's own, as the shot sees
    /// it: the cross term over the twirl's probability of the shot.
    pub fn ratio(&self, shot: &Shot, e: &Element) -> C {
        e.members.iter().fold(self.prefactor(shot, e), |acc, &g| acc * self.toggle(shot, g))
    }

    /// Location `l`'s own factor in a ratio: −i/t to take its fault away, i t to add one,
    /// signed by whether it anticommutes with the faults before it.
    pub fn toggle(&self, shot: &Shot, l: usize) -> C {
        let t = self.locations[l].theta.tan();
        let v = if shot.fired[l] { C::new(0.0, -1.0 / t) } else { C::new(0.0, t) };
        if shot.anti[l] { C::ZERO - v } else { v }
    }

    /// Kernel element `e`'s factor besides its locations' own: its phase, and the signs of the
    /// unread outcomes it ends on.
    pub fn prefactor(&self, shot: &Shot, e: &Element) -> C {
        let sign = e.z.iter().fold(e.offset, |a, &s| a ^ shot.branch[s]);
        let v = [C::ONE, C::I, C::new(-1.0, 0.0), C::new(0.0, -1.0)][(e.k % 4) as usize];
        if sign { C::ZERO - v } else { v }
    }

    /// A shot's weight: the coherent probability of its class (every fault set the kernel
    /// relates to it, which no record tells apart) over the twirl's, |1 + Σ r|² / (1 + Σ |r|²)
    /// for the ratios r of `kernel`'s elements. Never negative; exact (an unbiased weight) when
    /// `kernel` is the whole kernel.
    pub fn weight(&self, shot: &Shot, kernel: &[Element]) -> f64 {
        let (mut sum, mut norm) = (C::ONE, 1.0);
        for e in kernel {
            let r = self.ratio(shot, e);
            sum = sum + r;
            norm += r.norm2();
        }
        sum.norm2() / norm
    }
}

/// i^k Π_q X^{x_q} Z^{z_q}: a Pauli operator with its phase.
#[derive(Clone, Debug)]
pub(crate) struct Signed {
    x: Vec<bool>,
    z: Vec<bool>,
    k: u8,
}

impl Signed {
    fn new(n: usize) -> Signed {
        Signed { x: vec![false; n], z: vec![false; n], k: 0 }
    }

    fn turn(&mut self, quarter: u8) {
        self.k = (self.k + quarter) % 4;
    }

    /// P · self for the Hermitian Pauli product P: on each qubit, (i^y X^a Z^b)(X^x Z^z) =
    /// i^y (−1)^{bx} X^{a⊕x} Z^{b⊕z}.
    fn left_multiply(&mut self, paulis: &[(usize, Code)]) {
        for &(q, c) in paulis {
            let (a, b) = (c & 1 != 0, c & 2 != 0);
            if c == 3 {
                self.turn(1);
            }
            if b && self.x[q] {
                self.turn(2);
            }
            self.x[q] ^= a;
            self.z[q] ^= b;
        }
    }

    fn h(&mut self, q: usize) {
        if self.x[q] && self.z[q] {
            self.turn(2);
        }
        std::mem::swap(&mut self.x[q], &mut self.z[q]);
    }

    fn s(&mut self, q: usize) {
        if self.x[q] {
            self.turn(1);
            self.z[q] ^= true;
        }
    }

    fn cx(&mut self, c: usize, t: usize) {
        let xc = self.x[c];
        self.x[t] ^= xc;
        let zt = self.z[t];
        self.z[c] ^= zt;
    }

    fn cz(&mut self, a: usize, b: usize) {
        let (xa, xb) = (self.x[a], self.x[b]);
        if xa && xb {
            self.turn(2);
        }
        self.z[a] ^= xb;
        self.z[b] ^= xa;
    }

    /// Conjugation by the Pauli gate `c` on q: a sign where they anticommute.
    fn sign_by(&mut self, q: usize, c: Code) {
        let anti = (c & 1 != 0 && self.z[q]) ^ (c & 2 != 0 && self.x[q]);
        if anti {
            self.turn(2);
        }
    }

    /// Whether a measurement of q in `basis` would see this operator flip it.
    fn flips(&self, q: usize, basis: Basis) -> bool {
        match basis {
            Basis::Z => self.x[q],
            Basis::X => self.z[q],
        }
    }

    /// Takes the measured basis's Pauli off qubit q (it is the outcome's ±1); whether there
    /// was one. Only called when `flips` is false, so the Pauli factors out with no phase.
    fn absorb(&mut self, q: usize, basis: Basis) -> bool {
        let slot = match basis {
            Basis::Z => &mut self.z[q],
            Basis::X => &mut self.x[q],
        };
        std::mem::replace(slot, false)
    }
}

/// The circuit with each coherent rotation exp(−iθP) replaced by its Pauli twirl, P with
/// probability sin²θ: what Stim and every decoder assume. With `merge`, rotations that are the
/// same fault in different places (`kernel::Group`) become one, at the group's last place, with
/// their angles added (sin²(Σ ±θ)): the coherence-aware model. Merging flattens loops.
pub fn twirled(circuit: &Circuit, merge: bool) -> Result<Circuit, String> {
    let pauli_error = |paulis: &[(u32, u8)], p: f64| -> Instr {
        match paulis {
            [(q, c)] => Instr::PauliError { pauli: *c, p, qubits: vec![*q] },
            _ => Instr::Correlated { p, paulis: paulis.to_vec(), chained: false },
        }
    };
    if !merge {
        type Twirl<'a> = &'a dyn Fn(&[(u32, u8)], f64) -> Instr;
        fn walk(instrs: &[Instr], f: Twirl) -> Vec<Instr> {
            let mut out = Vec::new();
            for ins in instrs {
                match ins {
                    Instr::Repeat { count, body, tag } => out.push(Instr::Repeat { count: *count, body: walk(body, f), tag: tag.clone() }),
                    Instr::Gate { body, .. } if matches!(body.as_slice(), [Instr::NonPauli(_)]) => {
                        let Instr::NonPauli(ops) = &body[0] else { unreachable!() };
                        for op in ops {
                            match op {
                                NonPauli::Rotation { pauli, theta } => out.push(f(pauli, theta.sin().powi(2))),
                                _ => out.push(ins.clone()),
                            }
                        }
                    }
                    other => out.push(other.clone()),
                }
            }
            out
        }
        return Ok(Circuit { instrs: walk(&circuit.instrs, &pauli_error) });
    }
    let program = Program::new(circuit)?;
    let solver = GaugeSolver::new(&program);
    let kernel = kernel::Kernel::new(&program, &solver, kernel::CoherentOptions::default());
    // Each location's probability: its own, or its group's merged angle at the group's last.
    let mut prob: Vec<Option<f64>> = program.locations.iter().map(|l| Some(l.theta.sin().powi(2))).collect();
    for g in &kernel.groups {
        let angle: f64 = g.members.iter().zip(&g.mu).map(|(&l, mu)| program.locations[l].theta * if mu.re < 0.0 { -1.0 } else { 1.0 }).sum();
        for &l in &g.members {
            prob[l] = None;
        }
        prob[*g.members.last().unwrap()] = Some(angle.sin().powi(2));
    }
    let res = circuit.resolve()?;
    let mut out = Vec::new();
    let mut loc = 0usize;
    for ins in &res.instrs {
        match ins {
            Instr::NonPauli(ops) => {
                for op in ops {
                    if let NonPauli::Rotation { pauli, .. } = op {
                        if pauli.is_empty() {
                            continue;
                        }
                        if let Some(p) = prob[loc] {
                            out.push(pauli_error(pauli, p));
                        }
                        loc += 1;
                    }
                }
            }
            other => out.push(other.clone()),
        }
    }
    Ok(Circuit { instrs: out })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn prog(text: &str) -> Program {
        Program::new(&Circuit::parse(text).unwrap()).unwrap()
    }

    #[test]
    fn locations_are_numbered_in_order() {
        let p = prog("R 0 1\nI_ERROR[R_Z(theta=0.1)] 0 1\nII_ERROR[R_XX(theta=0.2)] 0 1\nM 0 1");
        assert_eq!(p.locations.len(), 3);
        assert_eq!(p.locations[2], Location { paulis: vec![(0, 1), (1, 1)], theta: 0.2 });
    }

    #[test]
    fn flips_follow_the_frame() {
        // X on qubit 0 before CX 0 1 flips both measurements and both ends; Z on 0 flips nothing.
        let p = prog("R 0 1\nI_ERROR[R_X(theta=0.1)] 0\nI_ERROR[R_Z(theta=0.1)] 0\nCX 0 1\nM 0 1");
        let f = p.flips();
        assert_eq!(f[0], vec![2, 3, p.end_slot(0), p.end_slot(1)]);
        assert_eq!(f[1], Vec::<usize>::new());
    }

    #[test]
    fn elements_of_small_circuits() {
        let element = |text: &str, g: &[usize]| {
            let p = prog(text);
            let solver = GaugeSolver::new(&p);
            p.element(g, &solver)
        };
        // Z on |0⟩ is +|0⟩: compatible, factor 1, signed by qubit 0's end.
        let e = element("R 0\nI_ERROR[R_Z(theta=0.1)] 0\nM 0", &[0]).unwrap();
        assert_eq!((e.k, e.offset), (0, false));
        // X on |0⟩ flips a deterministic measurement: no gauge flips it.
        assert!(element("R 0\nI_ERROR[R_X(theta=0.1)] 0\nM 0", &[0]).is_none());
        // X on |+⟩ is +|+⟩, though it flips the final Z measurement (a random one).
        assert!(element("R 0\nH 0\nI_ERROR[R_X(theta=0.1)] 0\nM 0", &[0]).is_some());
    }
}

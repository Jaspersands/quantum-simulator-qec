//! An exact simulator: the full state vector, for circuits of up to 24 qubits in use. It runs
//! everything the circuit language says, Clifford or not: every Stim gate, measurement, reset,
//! noise channel (as a quantum trajectory, one Kraus branch drawn per shot) and feedback, and
//! the tagged operations of `nonpauli` (rotations, T, U3, amplitude damping). It is the oracle
//! the coherent sampler is checked against.
//!
//! `exact_distribution` does not sample: it follows every branch (each measurement outcome,
//! each noise channel's every Pauli) with its probability, so on a small circuit it gives the
//! distribution of detection events and observable flips exactly.
//!
//! Detection events and observable flips are, as in Stim, the parities compared with the
//! noiseless reference run's (the Clifford circuit with every tag inert).

use std::collections::BTreeMap;

use crate::circuit::{Basis, Circuit, Control, Instr};
use crate::nonpauli::NonPauli;
use crate::surface_code::Xorshift;

/// The most qubits in use the simulator takes (2^24 amplitudes, 256 MB).
pub const MAX_QUBITS: usize = 24;

/// A complex number.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct C {
    pub re: f64,
    pub im: f64,
}

impl C {
    pub const ZERO: C = C { re: 0.0, im: 0.0 };
    pub const ONE: C = C { re: 1.0, im: 0.0 };
    pub const I: C = C { re: 0.0, im: 1.0 };

    pub fn new(re: f64, im: f64) -> C {
        C { re, im }
    }

    /// e^{iφ}.
    pub fn phase(phi: f64) -> C {
        C { re: phi.cos(), im: phi.sin() }
    }

    pub fn conj(self) -> C {
        C { re: self.re, im: -self.im }
    }

    pub fn norm2(self) -> f64 {
        self.re * self.re + self.im * self.im
    }

    pub fn scale(self, s: f64) -> C {
        C { re: self.re * s, im: self.im * s }
    }
}

impl std::ops::Add for C {
    type Output = C;
    fn add(self, o: C) -> C {
        C { re: self.re + o.re, im: self.im + o.im }
    }
}

impl std::ops::Sub for C {
    type Output = C;
    fn sub(self, o: C) -> C {
        C { re: self.re - o.re, im: self.im - o.im }
    }
}

impl std::ops::Mul for C {
    type Output = C;
    fn mul(self, o: C) -> C {
        C { re: self.re * o.re - self.im * o.im, im: self.re * o.im + self.im * o.re }
    }
}

/// A single-qubit matrix, rows then columns.
pub type M2 = [[C; 2]; 2];

/// The state of `n` qubits: amplitude k is the basis state whose bit q is qubit q.
#[derive(Clone, Debug)]
pub struct StateVector {
    n: usize,
    amp: Vec<C>,
}

impl StateVector {
    /// |0…0⟩ on `n` qubits.
    pub fn new(n: usize) -> StateVector {
        assert!(n <= MAX_QUBITS, "at most {MAX_QUBITS} qubits");
        let mut amp = vec![C::ZERO; 1 << n];
        amp[0] = C::ONE;
        StateVector { n, amp }
    }

    pub fn num_qubits(&self) -> usize {
        self.n
    }

    pub fn amplitudes(&self) -> &[C] {
        &self.amp
    }

    pub fn apply_1q(&mut self, q: usize, m: &M2) {
        let bit = 1usize << q;
        for k in 0..self.amp.len() {
            if k & bit == 0 {
                let (a, b) = (self.amp[k], self.amp[k | bit]);
                self.amp[k] = m[0][0] * a + m[0][1] * b;
                self.amp[k | bit] = m[1][0] * a + m[1][1] * b;
            }
        }
    }

    pub fn h(&mut self, q: usize) {
        let r = std::f64::consts::FRAC_1_SQRT_2;
        let bit = 1usize << q;
        for k in 0..self.amp.len() {
            if k & bit == 0 {
                let (a, b) = (self.amp[k], self.amp[k | bit]);
                self.amp[k] = (a + b).scale(r);
                self.amp[k | bit] = (a - b).scale(r);
            }
        }
    }

    /// diag(1, e^{iφ}).
    pub fn phase(&mut self, q: usize, phi: C) {
        let bit = 1usize << q;
        for (k, a) in self.amp.iter_mut().enumerate() {
            if k & bit != 0 {
                *a = *a * phi;
            }
        }
    }

    pub fn cx(&mut self, c: usize, t: usize) {
        let (cb, tb) = (1usize << c, 1usize << t);
        for k in 0..self.amp.len() {
            if k & cb != 0 && k & tb == 0 {
                self.amp.swap(k, k | tb);
            }
        }
    }

    pub fn cz(&mut self, a: usize, b: usize) {
        let m = (1usize << a) | (1usize << b);
        for (k, v) in self.amp.iter_mut().enumerate() {
            if k & m == m {
                *v = C::ZERO - *v;
            }
        }
    }

    /// The Pauli product `paulis` (qubit, code X 1, Z 2, Y 3) applied: P|k⟩ = i^{#Y}
    /// (−1)^{|k ∧ z|} |k ⊕ x⟩.
    pub fn pauli(&mut self, paulis: &[(usize, u8)]) {
        self.rotate_or_apply(paulis, None);
    }

    /// exp(−iθP) = cos θ − i sin θ P.
    pub fn rotate(&mut self, paulis: &[(usize, u8)], theta: f64) {
        self.rotate_or_apply(paulis, Some(theta));
    }

    fn rotate_or_apply(&mut self, paulis: &[(usize, u8)], theta: Option<f64>) {
        let (mut x, mut z, mut ny) = (0usize, 0usize, 0u32);
        for &(q, p) in paulis {
            if p & 1 != 0 {
                x ^= 1 << q;
            }
            if p & 2 != 0 {
                z ^= 1 << q;
            }
            if p == 3 {
                ny += 1;
            }
        }
        let iy = [C::ONE, C::I, C::new(-1.0, 0.0), C::new(0.0, -1.0)][(ny % 4) as usize];
        // (P|ψ⟩)[k ⊕ x] = iy (−1)^{|k ∧ z|} ψ[k].
        let coef = |k: usize| if (k & z).count_ones() % 2 == 1 { C::ZERO - iy } else { iy };
        let (c, s) = match theta {
            Some(t) => (t.cos(), t.sin()),
            None => (0.0, 1.0),
        };
        // The factor on P: −i sin θ for a rotation, 1 for the Pauli itself.
        let f = if theta.is_some() { C::new(0.0, -s) } else { C::ONE };
        if x == 0 {
            for (k, a) in self.amp.iter_mut().enumerate() {
                *a = a.scale(c) + f * coef(k) * *a;
            }
            return;
        }
        for k in 0..self.amp.len() {
            let j = k ^ x;
            if k < j {
                let (a, b) = (self.amp[k], self.amp[j]);
                // new[j] = c b + f coef(k) a; new[k] = c a + f coef(j) b.
                self.amp[j] = b.scale(c) + f * coef(k) * a;
                self.amp[k] = a.scale(c) + f * coef(j) * b;
            }
        }
    }

    /// The probability that qubit `q` measures 1 in the Z basis.
    pub fn prob_one(&self, q: usize) -> f64 {
        let bit = 1usize << q;
        self.amp.iter().enumerate().filter(|(k, _)| k & bit != 0).map(|(_, a)| a.norm2()).sum()
    }

    /// Projects qubit `q` onto `outcome` and renormalises by `p`, that outcome's probability.
    pub fn collapse(&mut self, q: usize, outcome: bool, p: f64) {
        let bit = 1usize << q;
        let s = 1.0 / p.sqrt();
        for (k, a) in self.amp.iter_mut().enumerate() {
            *a = if (k & bit != 0) == outcome { a.scale(s) } else { C::ZERO };
        }
    }

    /// A Kraus operator applied without renormalising; the squared norm after it.
    fn kraus(&mut self, q: usize, m: &M2) -> f64 {
        self.apply_1q(q, m);
        self.amp.iter().map(|a| a.norm2()).sum()
    }

    fn renormalise(&mut self, norm2: f64) {
        let s = 1.0 / norm2.sqrt();
        for a in &mut self.amp {
            *a = a.scale(s);
        }
    }
}

/// OpenQASM's U3(θ, φ, λ).
pub fn u3(theta: f64, phi: f64, lambda: f64) -> M2 {
    let (c, s) = ((theta / 2.0).cos(), (theta / 2.0).sin());
    [[C::new(c, 0.0), C::ZERO - C::phase(lambda).scale(s)], [C::phase(phi).scale(s), C::phase(phi + lambda).scale(c)]]
}

/// One step of the program: each instruction split into one step per target, qubits renumbered
/// to the ones in use.
#[derive(Clone, Debug)]
enum Step {
    H(usize),
    S(usize),
    Paulis(Vec<(usize, u8)>),
    Cx(usize, usize),
    Cz(usize, usize),
    Measure { q: usize, basis: Basis, reset: bool, flip: f64 },
    Reset { q: usize, basis: Basis },
    /// One Pauli product from `branches` (probability, product), or none with the rest.
    Channel(Vec<(f64, Vec<(usize, u8)>)>),
    Correlated { p: f64, paulis: Vec<(usize, u8)>, chained: bool },
    Pad { flip: f64, value: bool },
    /// A herald record, then I, X, Y or Z with probabilities `probs`.
    Herald { q: usize, probs: [f64; 4] },
    Feedback { pauli: u8, rec: usize, q: usize },
    Rotation { paulis: Vec<(usize, u8)>, theta: f64 },
    Unitary(usize, M2),
    Damp { q: usize, gamma: f64 },
    Leak { q: usize, p: f64 },
    Seep { q: usize, p: f64 },
    Transport { a: usize, b: usize, p: f64 },
}

/// A circuit ready for the state vector.
#[derive(Clone, Debug)]
pub struct Program {
    steps: Vec<Step>,
    num_qubits: usize,
    num_measurements: usize,
    detectors: Vec<Vec<usize>>,
    observables: Vec<Vec<usize>>,
    /// The noiseless reference's parity of each detector, then each observable.
    reference: Vec<bool>,
    /// Whether the circuit has leakage, run as the frame sampler's model.
    leaky: bool,
    /// Whether measuring a leaked qubit reads 1 (else a coin flip).
    pub leaked_reads_one: bool,
}

const CODE: [u8; 4] = [0, 1, 3, 2]; // Stim's letter order I, X, Y, Z as the engine's codes.

impl Program {
    pub fn new(circuit: &Circuit) -> Result<Program, String> {
        let res = circuit.resolve()?;
        // Qubits in use, renumbered in order.
        let mut used = std::collections::BTreeSet::new();
        for ins in &res.instrs {
            used.extend(ins.qubits());
            if let Instr::NonPauli(ops) = ins {
                for op in ops {
                    match op {
                        NonPauli::Rotation { pauli, .. } => used.extend(pauli.iter().map(|&(q, _)| q)),
                        NonPauli::T { qubit, .. } | NonPauli::U3 { qubit, .. } | NonPauli::AmplitudeDamping { qubit, .. } | NonPauli::Leak { qubit, .. } | NonPauli::Seep { qubit, .. } => {
                            used.insert(*qubit);
                        }
                        NonPauli::LeakTransport { a, b, .. } => used.extend([*a, *b]),
                    }
                }
            }
        }
        if used.len() > MAX_QUBITS {
            return Err(format!("the circuit uses {} qubits; the state vector holds at most {MAX_QUBITS}", used.len()));
        }
        let index: BTreeMap<u32, usize> = used.iter().enumerate().map(|(i, &q)| (q, i)).collect();
        let ix = |q: &u32| index[q];
        let mut steps = Vec::new();
        let mut m = 0usize;
        for ins in &res.instrs {
            match ins {
                Instr::Reset { basis, qubits } => steps.extend(qubits.iter().map(|q| Step::Reset { q: ix(q), basis: *basis })),
                Instr::H(qs) => steps.extend(qs.iter().map(|q| Step::H(ix(q)))),
                Instr::S(qs) => steps.extend(qs.iter().map(|q| Step::S(ix(q)))),
                Instr::Cx(ps) => steps.extend(ps.iter().map(|(a, b)| Step::Cx(ix(a), ix(b)))),
                Instr::Cz(ps) => steps.extend(ps.iter().map(|(a, b)| Step::Cz(ix(a), ix(b)))),
                Instr::Pauli { pauli, qubits } => {
                    if *pauli != 0 {
                        steps.extend(qubits.iter().map(|q| Step::Paulis(vec![(ix(q), *pauli)])))
                    }
                }
                // Sweep bits are all 0 here.
                Instr::SweepX(_) => {}
                Instr::Measure { basis, reset, flip, qubits } => {
                    m += qubits.len();
                    steps.extend(qubits.iter().map(|q| Step::Measure { q: ix(q), basis: *basis, reset: *reset, flip: *flip }));
                }
                Instr::PauliError { pauli, p, qubits } => {
                    steps.extend(qubits.iter().map(|q| Step::Channel(vec![(*p, vec![(ix(q), *pauli)])])));
                }
                Instr::Depolarize1 { p, qubits } => {
                    steps.extend(qubits.iter().map(|q| Step::Channel((1..4).map(|c| (p / 3.0, vec![(ix(q), c)])).collect())));
                }
                Instr::PauliChannel1 { px, py, pz, qubits } => {
                    steps.extend(qubits.iter().map(|q| Step::Channel(vec![(*px, vec![(ix(q), 1)]), (*py, vec![(ix(q), 3)]), (*pz, vec![(ix(q), 2)])])));
                }
                Instr::Depolarize2 { p, pairs } => {
                    for (a, b) in pairs {
                        let branches = (1..16u8).map(|r| (p / 15.0, vec![(ix(a), r & 3), (ix(b), r >> 2)])).collect();
                        steps.push(Step::Channel(branches));
                    }
                }
                Instr::PauliChannel2 { probs, pairs } => {
                    for (a, b) in pairs {
                        let branches = probs.iter().enumerate().map(|(k, &p)| (p, vec![(ix(a), CODE[(k + 1) / 4]), (ix(b), CODE[(k + 1) % 4])])).collect();
                        steps.push(Step::Channel(branches));
                    }
                }
                Instr::Correlated { p, paulis, chained } => {
                    steps.push(Step::Correlated { p: *p, paulis: paulis.iter().map(|(q, c)| (ix(q), *c)).collect(), chained: *chained });
                }
                Instr::Pad { flip, values } => {
                    m += values.len();
                    steps.extend(values.iter().map(|&v| Step::Pad { flip: *flip, value: v }));
                }
                Instr::Heralded { probs, qubits, .. } => {
                    m += qubits.len();
                    steps.extend(qubits.iter().map(|q| Step::Herald { q: ix(q), probs: *probs }));
                }
                Instr::Feedback { pauli, control, qubit } => match control {
                    Control::Rec(k) => steps.push(Step::Feedback { pauli: *pauli, rec: m - *k as usize, q: ix(qubit) }),
                    Control::Sweep(_) => {}
                },
                Instr::Observable { paulis, .. } if !paulis.is_empty() => {
                    return Err("the state vector does not take observables of Pauli targets".into());
                }
                Instr::NonPauli(ops) => {
                    for op in ops {
                        steps.push(match op {
                            NonPauli::Rotation { pauli, theta } => Step::Rotation { paulis: pauli.iter().map(|(q, c)| (ix(q), *c)).collect(), theta: *theta },
                            NonPauli::T { qubit, dagger } => {
                                let phi = if *dagger { -std::f64::consts::FRAC_PI_4 } else { std::f64::consts::FRAC_PI_4 };
                                Step::Unitary(ix(qubit), [[C::ONE, C::ZERO], [C::ZERO, C::phase(phi)]])
                            }
                            NonPauli::U3 { qubit, theta, phi, lambda } => Step::Unitary(ix(qubit), u3(*theta, *phi, *lambda)),
                            NonPauli::AmplitudeDamping { gamma, qubit } => Step::Damp { q: ix(qubit), gamma: *gamma },
                            NonPauli::Leak { p, qubit } => Step::Leak { q: ix(qubit), p: *p },
                            NonPauli::Seep { p, qubit } => Step::Seep { q: ix(qubit), p: *p },
                            NonPauli::LeakTransport { p, a, b } => Step::Transport { a: ix(a), b: ix(b), p: *p },
                        });
                    }
                }
                Instr::Detector { .. } | Instr::Observable { .. } | Instr::QubitCoords { .. } | Instr::ShiftCoords(_) | Instr::Tick => {}
                Instr::Repeat { .. } | Instr::Gate { .. } => unreachable!("resolve flattens loops and gates"),
            }
        }
        let reference = {
            // The converter's reference run, so detection events are the converter's.
            let rec = crate::m2d::run(circuit, crate::batch_sampler::Counts::of(&circuit.instrs)?.qubits, &[], 1);
            let parity = |recs: &Vec<usize>| recs.iter().fold(false, |a, &r| a ^ rec[r]);
            res.detectors.iter().chain(&res.observables).map(parity).collect()
        };
        let leaky = steps.iter().any(|s| matches!(s, Step::Leak { .. } | Step::Seep { .. } | Step::Transport { .. }));
        Ok(Program { steps, num_qubits: used.len(), num_measurements: m, detectors: res.detectors, observables: res.observables, reference, leaky, leaked_reads_one: true })
    }

    pub fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    /// The reference run's parity of each detector, then each observable.
    pub fn reference_parities(&self) -> &[bool] {
        &self.reference
    }

    pub fn num_detectors(&self) -> usize {
        self.detectors.len()
    }

    pub fn num_observables(&self) -> usize {
        self.observables.len()
    }

    fn ctx(&self) -> Ctx {
        Ctx { chain: false, leaked: if self.leaky { vec![false; self.num_qubits] } else { Vec::new() }, reads_one: self.leaked_reads_one }
    }

    /// A record's detection events and observable flips.
    fn outcome(&self, rec: &[bool]) -> (Vec<bool>, u64) {
        let parity = |recs: &Vec<usize>| recs.iter().fold(false, |a, &r| a ^ rec[r]);
        let nd = self.detectors.len();
        let dets = self.detectors.iter().enumerate().map(|(i, d)| parity(d) ^ self.reference[i]).collect();
        let obs = self.observables.iter().enumerate().fold(0u64, |acc, (i, o)| acc | ((parity(o) ^ self.reference[nd + i]) as u64) << i);
        (dets, obs)
    }

    /// One shot: its detection events and observable flips.
    pub fn sample(&self, rng: &mut Xorshift) -> (Vec<bool>, u64) {
        let mut sv = StateVector::new(self.num_qubits);
        let mut rec = Vec::with_capacity(self.num_measurements);
        let mut ctx = self.ctx();
        for step in &self.steps {
            run_step(step, &mut sv, &mut rec, &mut ctx, &mut |branches: &[f64]| {
                let u = rng.next_f64();
                let mut acc = 0.0;
                for (k, &p) in branches.iter().enumerate() {
                    acc += p;
                    if u < acc {
                        return k;
                    }
                }
                branches.len() - 1
            });
        }
        self.outcome(&rec)
    }

    /// Every outcome's exact probability, following every branch (outcomes below 1e-15 are
    /// dropped). At most `max_branches` leaves.
    pub fn distribution(&self, max_branches: usize) -> Result<BTreeMap<(Vec<bool>, u64), f64>, String> {
        let mut out = BTreeMap::new();
        let mut leaves = 0usize;
        let sv = StateVector::new(self.num_qubits);
        self.branch(0, sv, Vec::new(), self.ctx(), 1.0, &mut out, &mut leaves, max_branches)?;
        Ok(out)
    }

    #[allow(clippy::too_many_arguments)]
    fn branch(&self, mut pc: usize, mut sv: StateVector, mut rec: Vec<bool>, mut ctx: Ctx, prob: f64, out: &mut BTreeMap<(Vec<bool>, u64), f64>, leaves: &mut usize, max: usize) -> Result<(), String> {
        while pc < self.steps.len() {
            let step = &self.steps[pc];
            // The branch probabilities of this step, given the state.
            let probs = branch_probs(step, &sv, &ctx);
            if probs.len() > 1 {
                for (k, &p) in probs.iter().enumerate() {
                    if p * prob < 1e-15 {
                        continue;
                    }
                    let (mut sv2, mut rec2, mut ctx2) = (sv.clone(), rec.clone(), ctx.clone());
                    run_step(step, &mut sv2, &mut rec2, &mut ctx2, &mut |_| k);
                    self.branch(pc + 1, sv2, rec2, ctx2, prob * p, out, leaves, max)?;
                }
                return Ok(());
            }
            run_step(step, &mut sv, &mut rec, &mut ctx, &mut |_| 0);
            pc += 1;
        }
        *leaves += 1;
        if *leaves > max {
            return Err(format!("more than {max} branches"));
        }
        *out.entry(self.outcome(&rec)).or_insert(0.0) += prob;
        Ok(())
    }
}

/// A trajectory's classical state besides its amplitudes: whether the current chain of
/// correlated errors has fired, and which qubits are leaked.
#[derive(Clone, Debug)]
struct Ctx {
    chain: bool,
    leaked: Vec<bool>,
    reads_one: bool,
}

impl Ctx {
    fn leaked(&self, q: usize) -> bool {
        self.leaked.get(q).copied().unwrap_or(false)
    }

    /// The Pauli product without its factors on leaked qubits (which hold no state).
    fn live(&self, paulis: &[(usize, u8)]) -> Vec<(usize, u8)> {
        paulis.iter().copied().filter(|&(q, _)| !self.leaked(q)).collect()
    }
}

/// The probabilities of a step's branches in the order `run_step` numbers them; one entry for
/// a step that does not branch.
fn branch_probs(step: &Step, sv: &StateVector, ctx: &Ctx) -> Vec<f64> {
    let chain = ctx.chain;
    match step {
        Step::Measure { q, flip, .. } if ctx.leaked(*q) => {
            if ctx.reads_one {
                vec![0.0, 0.0, 1.0 - flip, *flip]
            } else {
                vec![0.5 * (1.0 - flip), 0.5 * flip, 0.5 * (1.0 - flip), 0.5 * flip]
            }
        }
        Step::Cx(a, b) | Step::Cz(a, b) if ctx.leaked(*a) != ctx.leaked(*b) => vec![0.25; 4],
        Step::Leak { q, p } if !ctx.leaked(*q) => {
            let p1 = sv.prob_one(*q);
            vec![1.0 - p, p * (1.0 - p1), p * p1]
        }
        Step::Seep { q, p } if ctx.leaked(*q) => vec![1.0 - p, p / 2.0, p / 2.0],
        Step::Transport { a, b, p } if ctx.leaked(*a) != ctx.leaked(*b) => {
            let other = if ctx.leaked(*a) { *b } else { *a };
            let p1 = sv.prob_one(other);
            vec![1.0 - p, p * (1.0 - p1) / 2.0, p * (1.0 - p1) / 2.0, p * p1 / 2.0, p * p1 / 2.0]
        }
        Step::Leak { .. } | Step::Seep { .. } | Step::Transport { .. } => vec![1.0],
        Step::Measure { q, basis, flip, .. } => {
            let p1 = match basis {
                Basis::Z => sv.prob_one(*q),
                Basis::X => {
                    let mut t = sv.clone();
                    t.h(*q);
                    t.prob_one(*q)
                }
            };
            // Outcome, then flip: (0, kept), (0, flipped), (1, kept), (1, flipped).
            vec![(1.0 - p1) * (1.0 - flip), (1.0 - p1) * flip, p1 * (1.0 - flip), p1 * flip]
        }
        Step::Reset { q, basis } => {
            let p1 = match basis {
                Basis::Z => sv.prob_one(*q),
                Basis::X => {
                    let mut t = sv.clone();
                    t.h(*q);
                    t.prob_one(*q)
                }
            };
            vec![1.0 - p1, p1]
        }
        Step::Channel(branches) => {
            let mut v: Vec<f64> = branches.iter().map(|b| b.0).collect();
            v.push(1.0 - v.iter().sum::<f64>());
            v
        }
        Step::Correlated { p, chained, .. } => {
            if *chained && chain {
                vec![1.0]
            } else {
                vec![*p, 1.0 - p]
            }
        }
        Step::Pad { flip, .. } => vec![1.0 - flip, *flip],
        Step::Herald { probs, .. } => {
            let mut v = probs.to_vec();
            v.push(1.0 - probs.iter().sum::<f64>());
            v
        }
        Step::Damp { q, gamma } => {
            let p1 = gamma * sv.prob_one(*q);
            vec![1.0 - p1, p1]
        }
        _ => vec![1.0],
    }
}

/// Runs one step, `choose` picking a branch (given the branch probabilities for a sampler; an
/// enumerator ignores them and names the branch).
fn run_step(step: &Step, sv: &mut StateVector, rec: &mut Vec<bool>, ctx: &mut Ctx, choose: &mut dyn FnMut(&[f64]) -> usize) {
    let probs = || branch_probs(step, sv, ctx);
    match step {
        // Leaked qubits take no part: their gates do nothing, a two-qubit gate with one leaves
        // the other a random Pauli, and measuring one reads 1 (or a coin).
        Step::H(q) | Step::S(q) | Step::Unitary(q, _) | Step::Damp { q, .. } if ctx.leaked(*q) => {}
        Step::Cx(a, b) | Step::Cz(a, b) if ctx.leaked(*a) || ctx.leaked(*b) => {
            if ctx.leaked(*a) != ctx.leaked(*b) {
                let other = if ctx.leaked(*a) { *b } else { *a };
                let k = choose(&probs());
                if k > 0 {
                    sv.pauli(&[(other, [0, 1, 3, 2][k])]);
                }
            }
        }
        Step::Measure { q, reset, .. } if ctx.leaked(*q) => {
            let k = choose(&probs());
            rec.push((k >= 2) ^ (k % 2 == 1));
            if *reset {
                ctx.leaked[*q] = false;
            }
        }
        Step::Reset { q, .. } if ctx.leaked(*q) => {
            ctx.leaked[*q] = false;
            if matches!(step, Step::Reset { basis: Basis::X, .. }) {
                sv.h(*q);
            }
        }
        Step::Leak { q, .. } => {
            if probs().len() > 1 {
                let k = choose(&probs());
                if k > 0 {
                    // The qubit's state goes with it: its Z outcome, then |0⟩ held in place.
                    let p1 = sv.prob_one(*q);
                    sv.collapse(*q, k == 2, if k == 2 { p1 } else { 1.0 - p1 });
                    if k == 2 {
                        sv.pauli(&[(*q, 1)]);
                    }
                    ctx.leaked[*q] = true;
                }
            }
        }
        Step::Seep { q, .. } => {
            if probs().len() > 1 {
                let k = choose(&probs());
                if k > 0 {
                    ctx.leaked[*q] = false;
                    if k == 2 {
                        sv.pauli(&[(*q, 1)]);
                    }
                }
            }
        }
        Step::Transport { a, b, .. } => {
            if probs().len() > 1 {
                let k = choose(&probs());
                if k > 0 {
                    let (back, other) = if ctx.leaked(*a) { (*a, *b) } else { (*b, *a) };
                    let gone = k >= 3;
                    let p1 = sv.prob_one(other);
                    sv.collapse(other, gone, if gone { p1 } else { 1.0 - p1 });
                    if gone {
                        sv.pauli(&[(other, 1)]);
                    }
                    ctx.leaked[other] = true;
                    ctx.leaked[back] = false;
                    if k.is_multiple_of(2) {
                        sv.pauli(&[(back, 1)]);
                    }
                }
            }
        }
        Step::H(q) => sv.h(*q),
        Step::S(q) => sv.phase(*q, C::I),
        Step::Paulis(p) => sv.pauli(&ctx.live(p)),
        Step::Cx(c, t) => sv.cx(*c, *t),
        Step::Cz(a, b) => sv.cz(*a, *b),
        Step::Rotation { paulis, theta } => {
            if paulis.iter().all(|&(q, _)| !ctx.leaked(q)) {
                sv.rotate(paulis, *theta)
            }
        }
        Step::Unitary(q, m) => sv.apply_1q(*q, m),
        Step::Measure { q, basis, reset, flip } => {
            if *basis == Basis::X {
                sv.h(*q);
            }
            let probs = branch_probs(&Step::Measure { q: *q, basis: Basis::Z, reset: false, flip: *flip }, sv, ctx);
            let k = choose(&probs);
            let outcome = k >= 2;
            let p = if outcome { probs[2] + probs[3] } else { probs[0] + probs[1] };
            sv.collapse(*q, outcome, p);
            rec.push(outcome ^ (k % 2 == 1));
            if *reset && outcome {
                sv.pauli(&[(*q, 1)]);
            }
            if *basis == Basis::X {
                sv.h(*q);
            }
        }
        Step::Reset { q, basis } => {
            if *basis == Basis::X {
                sv.h(*q);
            }
            let p1 = sv.prob_one(*q);
            let k = choose(&[1.0 - p1, p1]);
            sv.collapse(*q, k == 1, if k == 1 { p1 } else { 1.0 - p1 });
            if k == 1 {
                sv.pauli(&[(*q, 1)]);
            }
            if *basis == Basis::X {
                sv.h(*q);
            }
        }
        Step::Channel(branches) => {
            let mut probs: Vec<f64> = branches.iter().map(|b| b.0).collect();
            probs.push(1.0 - probs.iter().sum::<f64>());
            let k = choose(&probs);
            if k < branches.len() {
                sv.pauli(&ctx.live(&branches[k].1));
            }
        }
        Step::Correlated { p, paulis, chained } => {
            let fires = if *chained && ctx.chain { false } else { choose(&[*p, 1.0 - p]) == 0 };
            ctx.chain = if *chained { ctx.chain || fires } else { fires };
            if fires {
                sv.pauli(&ctx.live(paulis));
            }
        }
        Step::Pad { flip, value } => {
            let k = choose(&[1.0 - flip, *flip]);
            rec.push(*value ^ (k == 1));
        }
        Step::Herald { q, probs } => {
            let mut v = probs.to_vec();
            v.push(1.0 - probs.iter().sum::<f64>());
            let k = choose(&v);
            rec.push(k < 4);
            if (1..4).contains(&k) && !ctx.leaked(*q) {
                sv.pauli(&[(*q, [0, 1, 3, 2][k])]);
            }
        }
        Step::Feedback { pauli, rec: r, q } => {
            if rec[*r] {
                sv.pauli(&ctx.live(&[(*q, *pauli)]));
            }
        }
        Step::Damp { q, gamma } => {
            let p1 = gamma * sv.prob_one(*q);
            let k = choose(&[1.0 - p1, p1]);
            let g = *gamma;
            let m: M2 = if k == 1 { [[C::ZERO, C::new(g.sqrt(), 0.0)], [C::ZERO, C::ZERO]] } else { [[C::ONE, C::ZERO], [C::ZERO, C::new((1.0 - g).sqrt(), 0.0)]] };
            let n2 = sv.kraus(*q, &m);
            sv.renormalise(n2);
        }
    }
}

/// Exact samples, `shots` of them: detection events and observable flips as b8 rows, the
/// shot with index s drawn from its own stream (`batch_seed(seed, s)`), so threads do not
/// change them.
pub fn sample_seeded(program: &Program, seed: u64, first: u64, shots: usize, threads: usize) -> (Vec<u8>, Vec<u8>) {
    let (nd, no) = (program.num_detectors(), program.num_observables());
    let (dw, ow) = (nd.div_ceil(8), no.div_ceil(8));
    let rows = crate::parallel::parallel(shots, threads, |range| {
        range
            .map(|s| {
                let mut rng = Xorshift::new(crate::batch_sampler::batch_seed(seed, first + s as u64));
                program.sample(&mut rng)
            })
            .collect()
    });
    let (mut d, mut o) = (vec![0u8; shots * dw], vec![0u8; shots * ow]);
    for (s, (dets, obs)) in rows.into_iter().enumerate() {
        for (k, &b) in dets.iter().enumerate() {
            if b {
                d[s * dw + k / 8] |= 1 << (k % 8);
            }
        }
        for k in 0..no {
            if obs >> k & 1 == 1 {
                o[s * ow + k / 8] |= 1 << (k % 8);
            }
        }
    }
    (d, o)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dist(text: &str) -> BTreeMap<(Vec<bool>, u64), f64> {
        Program::new(&Circuit::parse(text).unwrap()).unwrap().distribution(1 << 20).unwrap()
    }

    #[test]
    fn noiseless_detectors_never_fire() {
        let d = dist("R 0 1\nH 0\nCX 0 1\nM 0 1\nDETECTOR rec[-1] rec[-2]\nOBSERVABLE_INCLUDE(0) rec[-1] rec[-2]");
        assert_eq!(d.len(), 1);
        let ((dets, obs), p) = d.into_iter().next().unwrap();
        assert_eq!((dets, obs), (vec![false], 0));
        assert!((p - 1.0).abs() < 1e-12);
    }

    #[test]
    fn a_rotation_flips_with_sin_squared() {
        let d = dist("R 0\nI_ERROR[R_X(theta=0.3)] 0\nM 0\nDETECTOR rec[-1]");
        let p1 = d[&(vec![true], 0)];
        assert!((p1 - 0.3f64.sin().powi(2)).abs() < 1e-12, "{p1}");
        // Two rotations add as angles, not as probabilities.
        let d = dist("R 0\nI_ERROR[R_X(theta=0.3)] 0\nI_ERROR[R_X(theta=0.2)] 0\nM 0\nDETECTOR rec[-1]");
        assert!((d[&(vec![true], 0)] - 0.5f64.sin().powi(2)).abs() < 1e-12);
    }

    #[test]
    fn t_gates_rotate_the_plus_state() {
        // H T T H = H S H = √X up to phase: P(1) = 1/2 from |0⟩; H T H: P(1) = sin²(π/8).
        let d = dist("R 0\nH 0\nI[T] 0\nH 0\nM 0\nDETECTOR rec[-1]");
        assert!((d[&(vec![true], 0)] - (std::f64::consts::PI / 8.0).sin().powi(2)).abs() < 1e-12);
    }

    #[test]
    fn noise_matches_its_channel() {
        let d = dist("R 0\nX_ERROR(0.1) 0\nM(0.05) 0\nDETECTOR rec[-1]");
        assert!((d[&(vec![true], 0)] - (0.1 * 0.95 + 0.9 * 0.05)).abs() < 1e-12);
        let d = dist("R 0 1\nDEPOLARIZE2(0.3) 0 1\nM 0 1\nDETECTOR rec[-2]\nDETECTOR rec[-1]");
        // Each qubit flips under 8 of the 15 Paulis.
        assert!((d[&(vec![true, false], 0)] - 0.3 * 4.0 / 15.0).abs() < 1e-12);
        assert!((d[&(vec![true, true], 0)] - 0.3 * 4.0 / 15.0).abs() < 1e-12);
    }

    #[test]
    fn amplitude_damping_decays_one() {
        let d = dist("R 0\nX 0\nI_ERROR[AMPLITUDE_DAMPING(gamma=0.25)] 0\nM 0\nDETECTOR rec[-1]");
        assert!((d[&(vec![true], 0)] - 0.25).abs() < 1e-12);
    }

    #[test]
    fn too_many_qubits_is_an_error() {
        let text: String = (0..25).map(|q| format!("R {q}\n")).collect();
        assert!(Program::new(&Circuit::parse(&text).unwrap()).unwrap_err().contains("at most 24"));
    }

    #[test]
    fn samples_follow_the_distribution() {
        let text = "R 0 1\nH 0\nI_ERROR[R_X(theta=0.4)] 1\nCX 0 1\nM 0 1\nDETECTOR rec[-1] rec[-2]";
        let prog = Program::new(&Circuit::parse(text).unwrap()).unwrap();
        let p = prog.distribution(1000).unwrap()[&(vec![true], 0)];
        let mut rng = Xorshift::new(7);
        let n = 20000;
        let hits = (0..n).filter(|_| prog.sample(&mut rng).0[0]).count() as f64;
        let sigma = (p * (1.0 - p) / n as f64).sqrt();
        assert!((hits / n as f64 - p).abs() < 5.0 * sigma, "{} vs {p}", hits / n as f64);
    }
}

//! A sampler for Clifford+T circuits at any qubit count, its cost growing with the number of
//! non-Clifford gates rather than the number of qubits: the stabilizer-rank sampler.
//!
//! The state is kept as a sum over Clifford branches in the Pauli-coset form the frame sampler
//! already lives in. A noiseless reference run — an Aaronson–Gottesman tableau of signed
//! stabilizer and destabilizer rows, random outcomes taken as 0, as Stim's reference sample —
//! is the same in every shot. The state is Σ_k c_k D(k)|ref⟩ over *keys* k ∈ F₂ⁿ: a key is a
//! Pauli's pattern of anticommutation with the stabilizer generators, which names its coset
//! modulo the stabilizer group (two Paulis with the same key differ by a stabilizer and a
//! phase), and D(k) is the coset's canonical representative, the product of the destabilizers
//! k selects. Distinct keys are orthogonal states. A Clifford gate conjugates the rows and
//! changes neither keys nor coefficients; a Pauli P takes D(k)|ref⟩ to λ·D(k ⊕ key(P))|ref⟩
//! for a phase λ found by one stabilizer-membership computation (`Tableau::express`), memoised
//! per step and key across shots.
//!
//! A non-Clifford gate is a sum of Paulis — exp(−iθP) = cos θ − i sin θ P; T is e^{iπ/8}
//! exp(−iπ/8 Z); U3 is three rotations — and splits every term in two (a Clifford angle is a
//! tableau operation and splits nothing). Terms landing on one key add: that is the
//! interference, and the number of terms is the number of distinct cosets, at most 2^(T count)
//! and far fewer on code states. A measurement that is deterministic on the reference keeps
//! the terms whose coset has the outcome; a random one collapses the reference as the tableau
//! does and re-expresses each term in the new destabilizers, the outcome's probability being
//! the squared norm of what remains. Pauli noise, feedback and reset corrections live in a
//! per-shot Pauli frame outside the terms, with the frame sampler's rules: an outcome flips
//! where the frame anticommutes with the measured Pauli, and a gate U becomes G U G, each of
//! its Pauli terms signed by whether it anticommutes with the frame. The frame carries its
//! phase, so two shots' states compare exactly (the joint coherent sampler's overlaps).
//!
//! Global phases of the circuit's gates are not kept (no probability depends on them).

use std::borrow::Cow;
use std::collections::{BTreeMap, HashMap};
use std::f64::consts::{FRAC_PI_4, FRAC_PI_8};

use crate::circuit::{Basis, Circuit, Control, Instr};
use crate::nonpauli::NonPauli;
use crate::statevec::C;
use crate::surface_code::Xorshift;

/// A Pauli's coset modulo the stabilizer group: its anticommutation with each generator,
/// bit-packed.
pub type Key = Vec<u64>;

fn bit(v: &[u64], q: usize) -> bool {
    v[q / 64] >> (q % 64) & 1 == 1
}

fn set_bit(v: &mut [u64], q: usize, b: bool) {
    if b {
        v[q / 64] |= 1 << (q % 64);
    } else {
        v[q / 64] &= !(1 << (q % 64));
    }
}

fn flip_bit(v: &mut [u64], q: usize) {
    v[q / 64] ^= 1 << (q % 64);
}

fn parity_and(a: &[u64], b: &[u64]) -> bool {
    a.iter().zip(b).fold(0u32, |acc, (x, y)| acc ^ (x & y).count_ones()) & 1 == 1
}

/// i^k X^x Z^z over the tableau's qubits: the operator is the product over qubits of
/// X^{x_q} Z^{z_q} (X before Z on each qubit), times i^k. Y is i·XZ.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Pauli {
    x: Vec<u64>,
    z: Vec<u64>,
    k: u8,
}

impl Pauli {
    pub fn identity(w: usize) -> Pauli {
        Pauli { x: vec![0; w], z: vec![0; w], k: 0 }
    }

    /// The Hermitian Pauli product with code 1 X, 2 Z, 3 Y on each listed qubit.
    pub fn from_codes(w: usize, paulis: &[(usize, u8)]) -> Pauli {
        let mut p = Pauli::identity(w);
        for &(q, c) in paulis {
            if c & 1 != 0 {
                flip_bit(&mut p.x, q);
            }
            if c & 2 != 0 {
                flip_bit(&mut p.z, q);
            }
            if c == 3 {
                p.k = (p.k + 1) % 4;
            }
        }
        p
    }

    pub fn is_identity(&self) -> bool {
        self.x.iter().all(|&w| w == 0) && self.z.iter().all(|&w| w == 0)
    }

    pub fn anticommutes(&self, o: &Pauli) -> bool {
        parity_and(&self.x, &o.z) ^ parity_and(&self.z, &o.x)
    }

    /// self · o: Z^{z} X^{x'} = (−1)^{|z ∧ x'|} X^{x'} Z^{z}.
    pub fn mul_assign(&mut self, o: &Pauli) {
        let swaps = self.z.iter().zip(&o.x).fold(0u32, |acc, (a, b)| acc + (a & b).count_ones());
        self.k = (self.k as u32 + o.k as u32 + 2 * swaps).rem_euclid(4) as u8;
        for (a, b) in self.x.iter_mut().zip(&o.x) {
            *a ^= b;
        }
        for (a, b) in self.z.iter_mut().zip(&o.z) {
            *a ^= b;
        }
    }

    /// self becomes p · self.
    pub fn left_mul(&mut self, p: &Pauli) {
        let mut q = p.clone();
        q.mul_assign(self);
        *self = q;
    }

    /// The adjoint: (i^k X^x Z^z)† = i^{−k} Z^z X^x = i^{−k} (−1)^{|x ∧ z|} X^x Z^z.
    pub fn dagger(&self) -> Pauli {
        let yy = self.x.iter().zip(&self.z).fold(0u32, |acc, (a, b)| acc + (a & b).count_ones());
        Pauli { x: self.x.clone(), z: self.z.clone(), k: (4 - self.k as u32 + 2 * yy).rem_euclid(4) as u8 }
    }

    // Conjugation by the Clifford gates, as the tableau's rows are conjugated.

    fn h(&mut self, q: usize) {
        let (xq, zq) = (bit(&self.x, q), bit(&self.z, q));
        if xq && zq {
            self.k = (self.k + 2) % 4;
        }
        set_bit(&mut self.x, q, zq);
        set_bit(&mut self.z, q, xq);
    }

    fn s(&mut self, q: usize) {
        if bit(&self.x, q) {
            self.k = (self.k + 1) % 4;
            flip_bit(&mut self.z, q);
        }
    }

    fn cx(&mut self, c: usize, t: usize) {
        if bit(&self.x, c) {
            flip_bit(&mut self.x, t);
        }
        if bit(&self.z, t) {
            flip_bit(&mut self.z, c);
        }
    }

    fn cz(&mut self, a: usize, b: usize) {
        let (xa, xb) = (bit(&self.x, a), bit(&self.x, b));
        if xa && xb {
            self.k = (self.k + 2) % 4;
        }
        if xb {
            flip_bit(&mut self.z, a);
        }
        if xa {
            flip_bit(&mut self.z, b);
        }
    }

    /// Conjugation by the Pauli `p`: a sign where they anticommute.
    fn conjugate_by_pauli(&mut self, p: &Pauli) {
        if self.anticommutes(p) {
            self.k = (self.k + 2) % 4;
        }
    }

    /// Conjugation by exp(−iπ/4 P) (by exp(+iπ/4 P) with `inverse`): Q anticommuting with P
    /// becomes i Q P (−i Q P).
    fn conjugate_by_rotation(&mut self, p: &Pauli, inverse: bool) {
        if self.anticommutes(p) {
            self.mul_assign(p);
            self.k = (self.k + if inverse { 3 } else { 1 }) % 4;
        }
    }
}

/// The reference run's tableau: destabilizer rows 0..n, stabilizer rows n..2n, each a signed
/// Pauli `i^k X^x Z^z`; the state is the +1 eigenstate of every stabilizer row.
#[derive(Clone, Debug)]
pub struct Tableau {
    n: usize,
    w: usize,
    x: Vec<u64>,
    z: Vec<u64>,
    k: Vec<u8>,
}

impl Tableau {
    pub fn new(n: usize) -> Tableau {
        let w = n.div_ceil(64).max(1);
        let mut t = Tableau { n, w, x: vec![0; 2 * n * w], z: vec![0; 2 * n * w], k: vec![0; 2 * n] };
        for i in 0..n {
            flip_bit(&mut t.x[i * w..(i + 1) * w], i);
            flip_bit(&mut t.z[(n + i) * w..(n + i + 1) * w], i);
        }
        t
    }

    fn row(&self, i: usize) -> Pauli {
        Pauli { x: self.x[i * self.w..(i + 1) * self.w].to_vec(), z: self.z[i * self.w..(i + 1) * self.w].to_vec(), k: self.k[i] }
    }

    fn row_x(&self, i: usize) -> &[u64] {
        &self.x[i * self.w..(i + 1) * self.w]
    }

    fn row_z(&self, i: usize) -> &[u64] {
        &self.z[i * self.w..(i + 1) * self.w]
    }

    fn row_anticommutes(&self, i: usize, p: &Pauli) -> bool {
        parity_and(self.row_x(i), &p.z) ^ parity_and(self.row_z(i), &p.x)
    }

    /// Row i becomes row_i · p.
    fn row_mul(&mut self, i: usize, p: &Pauli) {
        let w = self.w;
        let swaps = self.z[i * w..(i + 1) * w].iter().zip(&p.x).fold(0u32, |acc, (a, b)| acc + (a & b).count_ones());
        self.k[i] = (self.k[i] as u32 + p.k as u32 + 2 * swaps).rem_euclid(4) as u8;
        for (a, b) in self.x[i * w..(i + 1) * w].iter_mut().zip(&p.x) {
            *a ^= b;
        }
        for (a, b) in self.z[i * w..(i + 1) * w].iter_mut().zip(&p.z) {
            *a ^= b;
        }
    }

    fn h(&mut self, q: usize) {
        let w = self.w;
        for i in 0..2 * self.n {
            let (xq, zq) = (bit(&self.x[i * w..], q), bit(&self.z[i * w..], q));
            if xq && zq {
                self.k[i] = (self.k[i] + 2) % 4;
            }
            set_bit(&mut self.x[i * w..(i + 1) * w], q, zq);
            set_bit(&mut self.z[i * w..(i + 1) * w], q, xq);
        }
    }

    fn s(&mut self, q: usize) {
        let w = self.w;
        for i in 0..2 * self.n {
            if bit(&self.x[i * w..], q) {
                self.k[i] = (self.k[i] + 1) % 4;
                flip_bit(&mut self.z[i * w..(i + 1) * w], q);
            }
        }
    }

    fn cx(&mut self, c: usize, t: usize) {
        let w = self.w;
        for i in 0..2 * self.n {
            if bit(&self.x[i * w..], c) {
                flip_bit(&mut self.x[i * w..(i + 1) * w], t);
            }
            if bit(&self.z[i * w..], t) {
                flip_bit(&mut self.z[i * w..(i + 1) * w], c);
            }
        }
    }

    fn cz(&mut self, a: usize, b: usize) {
        let w = self.w;
        for i in 0..2 * self.n {
            let (xa, xb) = (bit(&self.x[i * w..], a), bit(&self.x[i * w..], b));
            if xa && xb {
                self.k[i] = (self.k[i] + 2) % 4;
            }
            if xb {
                flip_bit(&mut self.z[i * w..(i + 1) * w], a);
            }
            if xa {
                flip_bit(&mut self.z[i * w..(i + 1) * w], b);
            }
        }
    }

    /// Conjugation by the Pauli `p`: a sign on every row that anticommutes with it.
    fn conjugate_by(&mut self, p: &Pauli) {
        for i in 0..2 * self.n {
            if self.row_anticommutes(i, p) {
                self.k[i] = (self.k[i] + 2) % 4;
            }
        }
    }

    /// Conjugation by exp(−iπ/4 P) (by exp(+iπ/4 P) with `inverse`): a row Q anticommuting
    /// with P becomes i Q P (−i Q P).
    fn clifford_rotation(&mut self, p: &Pauli, inverse: bool) {
        for i in 0..2 * self.n {
            if self.row_anticommutes(i, p) {
                self.row_mul(i, p);
                self.k[i] = (self.k[i] + if inverse { 3 } else { 1 }) % 4;
            }
        }
    }

    /// The anticommutation of `p` with the stabilizer rows: its key.
    fn key_of(&self, p: &Pauli) -> Key {
        let mut key = vec![0u64; self.w];
        for i in 0..self.n {
            if self.row_anticommutes(self.n + i, p) {
                flip_bit(&mut key, i);
            }
        }
        key
    }

    /// The anticommutation of `p` with the destabilizer rows: which stabilizers make it up when
    /// it is in the group.
    fn dkey_of(&self, p: &Pauli) -> Key {
        let mut key = vec![0u64; self.w];
        for i in 0..self.n {
            if self.row_anticommutes(i, p) {
                flip_bit(&mut key, i);
            }
        }
        key
    }

    /// The product of the rows `base + i` over the bits i of `sel`, in increasing i.
    fn product(&self, base: usize, sel: &[u64]) -> Pauli {
        let mut p = Pauli::identity(self.w);
        for i in 0..self.n {
            if bit(sel, i) {
                p.mul_assign(&self.row(base + i));
            }
        }
        p
    }

    /// The coset's canonical representative D(key), the destabilizers the key selects.
    fn representative(&self, key: &[u64]) -> Pauli {
        self.product(0, key)
    }

    /// P|ref⟩ as λ·D(key)|ref⟩: the key of P and the phase λ = i^m, as (key, m). D(key)† P
    /// commutes with every stabilizer, so it is μ times a product of stabilizer rows, the ones
    /// its anticommutation with the destabilizers selects; μ is what is left after multiplying
    /// that product back in, and λ = μ.
    pub fn express(&self, p: &Pauli) -> (Key, u8) {
        let key = self.key_of(p);
        let mut r = self.representative(&key).dagger();
        r.mul_assign(p);
        let b = self.dkey_of(&r);
        r.mul_assign(&self.product(self.n, &b));
        debug_assert!(r.is_identity(), "a Pauli commuting with every stabilizer is in the group up to phase");
        (key, r.k)
    }

    /// The stabilizer row anticommuting with `m`, if any: a measurement of `m` is random.
    fn pivot(&self, m: &Pauli) -> Option<usize> {
        (0..self.n).find(|&i| self.row_anticommutes(self.n + i, m))
    }

    /// The reference's outcome of a deterministic measurement of the Hermitian `m` (±m is a
    /// stabilizer): 1 when −m is.
    fn deterministic_outcome(&self, m: &Pauli) -> bool {
        let prod = self.product(self.n, &self.dkey_of(m));
        debug_assert!(prod.x == m.x && prod.z == m.z);
        (prod.k + 4 - m.k) % 4 == 2
    }

    /// A random measurement of `m`, pivot `p`, collapsed to the reference's outcome 0: rows
    /// anticommuting with m take the pivot's stabilizer, which becomes the pivot's destabilizer,
    /// and +m the pivot's stabilizer.
    fn collapse(&mut self, m: &Pauli, p: usize) {
        let g = self.row(self.n + p);
        for i in 0..2 * self.n {
            if i != self.n + p && self.row_anticommutes(i, m) {
                self.row_mul(i, &g);
            }
        }
        let w = self.w;
        self.x[p * w..(p + 1) * w].copy_from_slice(&g.x);
        self.z[p * w..(p + 1) * w].copy_from_slice(&g.z);
        self.k[p] = g.k;
        self.x[(self.n + p) * w..(self.n + p + 1) * w].copy_from_slice(&m.x);
        self.z[(self.n + p) * w..(self.n + p + 1) * w].copy_from_slice(&m.z);
        self.k[self.n + p] = m.k;
    }
}

/// One step of the program.
#[derive(Clone, Debug)]
enum Step {
    H(usize),
    S(usize),
    Cx(usize, usize),
    Cz(usize, usize),
    /// A Pauli gate, or a rotation by a multiple of π/2: conjugates the reference.
    PauliGate(Pauli),
    /// exp(∓iπ/4 P): a Clifford rotation.
    CliffordRotation { pauli: Pauli, inverse: bool },
    /// A non-Clifford gate as a sum of Paulis: Σ amplitude · Pauli (the identity among them).
    Branch(Vec<(C, Pauli)>),
    /// A coherent error drawn as its Pauli twirl: location `twirl[i]` fires (its Pauli into
    /// the frame) with probability sin²θ.
    Twirl(usize),
    Measure { q: usize, basis: Basis, reset: bool, flip: f64 },
    Reset { q: usize, basis: Basis },
    /// One Pauli product from `branches` (probability, product), or none with the rest.
    Channel(Vec<(f64, Vec<(usize, u8)>)>),
    Correlated { p: f64, paulis: Vec<(usize, u8)>, chained: bool },
    Pad { flip: f64, value: bool },
    Herald { q: usize, probs: [f64; 4] },
    Feedback { pauli: u8, rec: usize, q: usize },
}

/// A circuit compiled for the stabilizer-rank sampler.
#[derive(Debug)]
pub struct Program {
    steps: Vec<Step>,
    num_qubits: usize,
    w: usize,
    num_measurements: usize,
    detectors: Vec<Vec<usize>>,
    observables: Vec<Vec<usize>>,
    /// The noiseless reference's parity of each detector, then each observable.
    reference: Vec<bool>,
    /// Whether any step draws at random besides measurements (a noise channel, a flip).
    noisy: bool,
    /// The twirled coherent errors (`with_twirl`): each one's Pauli and angle.
    twirl: Vec<(Pauli, f64)>,
    /// The reference tableau at the end of the circuit, the same in every shot.
    final_tab: Tableau,
}

const CODE: [u8; 4] = [0, 1, 3, 2]; // Stim's letter order I, X, Y, Z as the engine's codes.

/// Steps for exp(−iθP): nothing, a Pauli or a Clifford rotation at a multiple of π/4 (within
/// 1e-9 of one), else the two-term sum.
fn rotation_steps(w: usize, pauli: &Pauli, theta: f64) -> Vec<Step> {
    if pauli.is_identity() {
        return Vec::new();
    }
    let r = theta / FRAC_PI_4;
    let k = r.round();
    if (r - k).abs() < 1e-9 {
        return match (k as i64).rem_euclid(8) {
            0 | 4 => Vec::new(),
            2 | 6 => vec![Step::PauliGate(pauli.clone())],
            1 | 5 => vec![Step::CliffordRotation { pauli: pauli.clone(), inverse: false }],
            _ => vec![Step::CliffordRotation { pauli: pauli.clone(), inverse: true }],
        };
    }
    vec![Step::Branch(vec![(C::new(theta.cos(), 0.0), Pauli::identity(w)), (C::new(0.0, -theta.sin()), pauli.clone())])]
}

/// The reference tableau after every step: Clifford gates applied, random measurements
/// collapsed to 0.
fn end_tableau(steps: &[Step], n: usize, w: usize) -> Tableau {
    let mut tab = Tableau::new(n);
    for step in steps {
        match step {
            Step::H(q) => tab.h(*q),
            Step::S(q) => tab.s(*q),
            Step::Cx(c, t) => tab.cx(*c, *t),
            Step::Cz(a, b) => tab.cz(*a, *b),
            Step::PauliGate(p) => tab.conjugate_by(p),
            Step::CliffordRotation { pauli, inverse } => tab.clifford_rotation(pauli, *inverse),
            Step::Measure { q, basis, .. } | Step::Reset { q, basis } => {
                let m = Pauli::from_codes(w, &[(*q, if *basis == Basis::Z { 2 } else { 1 })]);
                if let Some(p) = tab.pivot(&m) {
                    tab.collapse(&m, p);
                }
            }
            _ => {}
        }
    }
    tab
}

impl Program {
    /// The circuit with every operation exact, coherent errors included.
    pub fn new(circuit: &Circuit) -> Result<Program, String> {
        Program::build(circuit, false)
    }

    /// The circuit with its coherent errors (`I_ERROR[R_Z(theta=…)]`, rotations written as
    /// errors) drawn as their Pauli twirl, for the joint coherent sampler; the non-Clifford
    /// gates stay exact.
    pub fn with_twirl(circuit: &Circuit) -> Result<Program, String> {
        Program::build(circuit, true)
    }

    fn build(circuit: &Circuit, twirl_errors: bool) -> Result<Program, String> {
        circuit.check_runnable()?;
        let res = circuit.resolve()?;
        let mut n = res.num_qubits;
        for ins in &res.instrs {
            n = n.max(ins.qubits().into_iter().map(|q| q as usize + 1).max().unwrap_or(0));
            if let Instr::NonPauli(ops) = ins {
                for op in ops {
                    n = n.max(op.qubits().into_iter().map(|q| q as usize + 1).max().unwrap_or(0));
                }
            }
        }
        let w = n.div_ceil(64).max(1);
        let ix = |q: &u32| *q as usize;
        let mut steps = Vec::new();
        let mut twirl = Vec::new();
        let mut m = 0usize;
        let mut noisy = false;
        for ins in &res.instrs {
            match ins {
                Instr::Fail(_) => {}
                Instr::Reset { basis, qubits } => steps.extend(qubits.iter().map(|q| Step::Reset { q: ix(q), basis: *basis })),
                Instr::H(qs) => steps.extend(qs.iter().map(|q| Step::H(ix(q)))),
                Instr::S(qs) => steps.extend(qs.iter().map(|q| Step::S(ix(q)))),
                Instr::Cx(ps) => steps.extend(ps.iter().map(|(a, b)| Step::Cx(ix(a), ix(b)))),
                Instr::Cz(ps) => steps.extend(ps.iter().map(|(a, b)| Step::Cz(ix(a), ix(b)))),
                Instr::Pauli { pauli, qubits } => {
                    if *pauli != 0 {
                        steps.extend(qubits.iter().map(|q| Step::PauliGate(Pauli::from_codes(w, &[(ix(q), *pauli)]))));
                    }
                }
                // Sweep bits are all 0 here.
                Instr::SweepX(_) => {}
                Instr::Measure { basis, reset, flip, qubits } => {
                    m += qubits.len();
                    noisy |= *flip > 0.0;
                    steps.extend(qubits.iter().map(|q| Step::Measure { q: ix(q), basis: *basis, reset: *reset, flip: *flip }));
                }
                Instr::PauliError { pauli, p, qubits } => {
                    noisy = true;
                    steps.extend(qubits.iter().map(|q| Step::Channel(vec![(*p, vec![(ix(q), *pauli)])])));
                }
                Instr::Depolarize1 { p, qubits } => {
                    noisy = true;
                    steps.extend(qubits.iter().map(|q| Step::Channel((1..4).map(|c| (p / 3.0, vec![(ix(q), c)])).collect())));
                }
                Instr::PauliChannel1 { px, py, pz, qubits } => {
                    noisy = true;
                    steps.extend(qubits.iter().map(|q| Step::Channel(vec![(*px, vec![(ix(q), 1)]), (*py, vec![(ix(q), 3)]), (*pz, vec![(ix(q), 2)])])));
                }
                Instr::Depolarize2 { p, pairs } => {
                    noisy = true;
                    for (a, b) in pairs {
                        steps.push(Step::Channel((1..16u8).map(|r| (p / 15.0, vec![(ix(a), r & 3), (ix(b), r >> 2)])).collect()));
                    }
                }
                Instr::PauliChannel2 { probs, pairs } => {
                    noisy = true;
                    for (a, b) in pairs {
                        steps.push(Step::Channel(probs.iter().enumerate().map(|(k, &p)| (p, vec![(ix(a), CODE[(k + 1) / 4]), (ix(b), CODE[(k + 1) % 4])])).collect()));
                    }
                }
                Instr::Correlated { p, paulis, chained } => {
                    noisy = true;
                    steps.push(Step::Correlated { p: *p, paulis: paulis.iter().map(|(q, c)| (ix(q), *c)).collect(), chained: *chained });
                }
                Instr::Pad { flip, values } => {
                    m += values.len();
                    noisy |= *flip > 0.0;
                    steps.extend(values.iter().map(|&v| Step::Pad { flip: *flip, value: v }));
                }
                Instr::Heralded { probs, qubits, .. } => {
                    m += qubits.len();
                    noisy = true;
                    steps.extend(qubits.iter().map(|q| Step::Herald { q: ix(q), probs: *probs }));
                }
                Instr::Feedback { pauli, control, qubit } => match control {
                    Control::Rec(k) => steps.push(Step::Feedback { pauli: *pauli, rec: m - *k as usize, q: ix(qubit) }),
                    Control::Sweep(_) => {}
                },
                Instr::Observable { paulis, .. } if !paulis.is_empty() => {
                    return Err("the stabilizer-rank sampler does not take observables of Pauli targets".into());
                }
                Instr::NonPauli(ops) => {
                    for op in ops {
                        match op {
                            NonPauli::Rotation { pauli, theta, noise } => {
                                let p = Pauli::from_codes(w, &pauli.iter().map(|&(q, c)| (q as usize, c)).collect::<Vec<_>>());
                                // A coherent error at a Clifford angle is a fixed Clifford: run as the gate
                                // it is, not drawn as a twirl (I or Z, 50/50) that puts all of it in the
                                // interference the twirl leaves out. The reference run ignores it, as
                                // Stim's ignores every error.
                                if twirl_errors && *noise && !p.is_identity() && !is_clifford_angle(*theta) {
                                    steps.push(Step::Twirl(twirl.len()));
                                    twirl.push((p, *theta));
                                } else {
                                    steps.extend(rotation_steps(w, &p, *theta));
                                }
                            }
                            NonPauli::T { qubit, dagger } => {
                                let z = Pauli::from_codes(w, &[(*qubit as usize, 2)]);
                                steps.extend(rotation_steps(w, &z, if *dagger { -FRAC_PI_8 } else { FRAC_PI_8 }));
                            }
                            // U3(θ, φ, λ) = e^{i(φ+λ)/2} R_Z(φ) R_Y(θ) R_Z(λ), R_P(α) = exp(−iαP/2).
                            NonPauli::U3 { qubit, theta, phi, lambda } => {
                                let q = *qubit as usize;
                                steps.extend(rotation_steps(w, &Pauli::from_codes(w, &[(q, 2)]), lambda / 2.0));
                                steps.extend(rotation_steps(w, &Pauli::from_codes(w, &[(q, 3)]), theta / 2.0));
                                steps.extend(rotation_steps(w, &Pauli::from_codes(w, &[(q, 2)]), phi / 2.0));
                            }
                            NonPauli::AmplitudeDamping { .. } | NonPauli::PhaseDamping { .. } => {
                                return Err(format!("the stabilizer-rank sampler does not run '{}'; the state vector and the density matrix do", op.line()));
                            }
                            other => {
                                return Err(format!("the stabilizer-rank sampler does not run leakage ('{}'); the leakage sampler does", other.line()));
                            }
                        }
                    }
                }
                Instr::Detector { .. } | Instr::Observable { .. } | Instr::QubitCoords { .. } | Instr::ShiftCoords(_) | Instr::Tick => {}
                Instr::Repeat { .. } | Instr::Gate { .. } => unreachable!("resolve flattens loops and gates"),
            }
        }
        if res.observables.len() > 64 {
            return Err(format!("{} observables: the stabilizer-rank sampler takes at most 64", res.observables.len()));
        }
        let reference = {
            let rec = crate::m2d::reference(circuit, crate::batch_sampler::Counts::of(&circuit.instrs)?.qubits);
            let parity = |recs: &Vec<usize>| recs.iter().fold(false, |a, &r| a ^ rec[r]);
            res.detectors.iter().chain(&res.observables).map(parity).collect()
        };
        let final_tab = end_tableau(&steps, n, w);
        Ok(Program { steps, num_qubits: n, w, num_measurements: m, detectors: res.detectors, observables: res.observables, reference, noisy, twirl, final_tab })
    }

    pub fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    pub fn num_measurements(&self) -> usize {
        self.num_measurements
    }

    pub fn num_detectors(&self) -> usize {
        self.detectors.len()
    }

    pub fn num_observables(&self) -> usize {
        self.observables.len()
    }

    /// The twirled coherent errors.
    pub fn num_twirl(&self) -> usize {
        self.twirl.len()
    }

    /// Whether anything besides measurement outcomes is drawn at random.
    pub fn is_noisy(&self) -> bool {
        self.noisy
    }

    /// How many of the circuit's gates split the state (non-Clifford gates, rotations at a
    /// Clifford angle not among them).
    pub fn num_non_clifford(&self) -> usize {
        self.steps.iter().filter(|s| matches!(s, Step::Branch(_))).count()
    }

    /// A record's detection events and observable flips, compared with the reference run's.
    pub fn outcome(&self, rec: &[bool]) -> (Vec<bool>, u64) {
        let parity = |recs: &Vec<usize>| recs.iter().fold(false, |a, &r| a ^ rec[r]);
        let nd = self.detectors.len();
        let dets = self.detectors.iter().enumerate().map(|(i, d)| parity(d) ^ self.reference[i]).collect();
        let obs = self.observables.iter().enumerate().fold(0u64, |acc, (i, o)| acc | ((parity(o) ^ self.reference[nd + i]) as u64) << i);
        (dets, obs)
    }
}

/// Where a shot's random choices come from.
pub trait Source {
    /// The branch taken among `probs` (which sum to 1).
    fn choose(&mut self, probs: &[f64]) -> usize;
    /// A measurement whose outcome is 1 with probability `p1` and whose record is flipped with
    /// probability `flip`: (outcome, record).
    fn outcome(&mut self, p1: f64, flip: f64) -> (bool, bool);
    /// Whether twirled location `loc` fires (with probability `p`, the twirl's).
    fn fire(&mut self, loc: usize, p: f64) -> bool;
}

/// A random stream.
pub struct Stream<'a>(pub &'a mut Xorshift);

impl Source for Stream<'_> {
    fn choose(&mut self, probs: &[f64]) -> usize {
        let u = self.0.next_f64();
        let mut acc = 0.0;
        for (k, &p) in probs.iter().enumerate() {
            acc += p;
            if u < acc {
                return k;
            }
        }
        probs.len() - 1
    }

    fn outcome(&mut self, p1: f64, flip: f64) -> (bool, bool) {
        let k = self.choose(&[(1.0 - p1) * (1.0 - flip), (1.0 - p1) * flip, p1 * (1.0 - flip), p1 * flip]);
        (k >= 2, (k >= 2) ^ (k % 2 == 1))
    }

    fn fire(&mut self, _loc: usize, p: f64) -> bool {
        self.0.next_f64() < p
    }
}

/// A shot's draws, kept so that the shot can be run again with other errors inserted: every
/// branch chosen and every measurement's (outcome, record).
#[derive(Clone, Debug, Default)]
pub struct Trace {
    pub choices: Vec<usize>,
    pub outcomes: Vec<(bool, bool)>,
}

/// A source whose draws are recorded.
pub struct Recording<'a> {
    inner: &'a mut dyn Source,
    pub trace: Trace,
    pub fired: Vec<bool>,
}

impl<'a> Recording<'a> {
    pub fn new(inner: &'a mut dyn Source, num_twirl: usize) -> Recording<'a> {
        Recording { inner, trace: Trace::default(), fired: vec![false; num_twirl] }
    }

    pub fn into_trace(self) -> Trace {
        self.trace
    }
}

impl Source for Recording<'_> {
    fn choose(&mut self, probs: &[f64]) -> usize {
        let k = self.inner.choose(probs);
        self.trace.choices.push(k);
        k
    }

    fn outcome(&mut self, p1: f64, flip: f64) -> (bool, bool) {
        let o = self.inner.outcome(p1, flip);
        self.trace.outcomes.push(o);
        o
    }

    fn fire(&mut self, loc: usize, p: f64) -> bool {
        let f = self.inner.fire(loc, p);
        self.fired[loc] = f;
        f
    }
}

/// A recorded shot run again, with a given set of errors fired.
pub struct Replay<'a> {
    trace: &'a Trace,
    fired: &'a [bool],
    ci: usize,
    oi: usize,
}

impl<'a> Replay<'a> {
    pub fn new(trace: &'a Trace, fired: &'a [bool]) -> Replay<'a> {
        Replay { trace, fired, ci: 0, oi: 0 }
    }
}

impl Source for Replay<'_> {
    fn choose(&mut self, _probs: &[f64]) -> usize {
        self.ci += 1;
        self.trace.choices[self.ci - 1]
    }

    fn outcome(&mut self, _p1: f64, _flip: f64) -> (bool, bool) {
        self.oi += 1;
        self.trace.outcomes[self.oi - 1]
    }

    fn fire(&mut self, loc: usize, _p: f64) -> bool {
        self.fired[loc]
    }
}

/// Memoised phases: for a branch step's Pauli applied to a key, and for a term re-expressed
/// after a random measurement, with or without the pivot's old stabilizer.
#[derive(Default)]
pub struct Memo {
    branch: HashMap<(usize, usize, Key), (Key, u8)>,
    measure: HashMap<(usize, bool, Key), (Key, u8)>,
}

const PHASES: [C; 4] = [C::ONE, C::I, C { re: -1.0, im: 0.0 }, C { re: 0.0, im: -1.0 }];

/// A shot run through the circuit: its records, which twirled errors fired, its final terms
/// (keys to amplitudes, normalised) and Pauli frame — the state is frame · Σ_k c_k D(k)|ref⟩ —
/// the probability of its trajectory's measurement and reset outcomes, and its draws.
#[derive(Clone, Debug)]
pub struct Run {
    pub records: Vec<bool>,
    pub fired: Vec<bool>,
    pub terms: HashMap<Key, C>,
    pub frame: Pauli,
    pub prob: f64,
    /// Whether a forced outcome was impossible: the state is zero.
    pub dead: bool,
    /// The most terms held at once.
    pub peak: usize,
    pub trace: Trace,
    chain: bool,
}

impl Run {
    fn renormalise(&mut self, norm2: f64) {
        if norm2 <= 1e-300 || self.terms.is_empty() {
            self.dead = true;
            self.prob = 0.0;
            self.terms.clear();
            return;
        }
        self.prob *= norm2;
        let s = 1.0 / norm2.sqrt();
        for c in self.terms.values_mut() {
            *c = c.scale(s);
        }
        self.peak = self.peak.max(self.terms.len());
    }
}

/// Terms below this amplitude are dropped (their probability is below 1e-26).
const EPS: f64 = 1e-13;

fn add_term(map: &mut HashMap<Key, C>, key: Key, c: C) {
    let e = map.entry(key).or_insert(C::ZERO);
    *e = *e + c;
}

fn prune(map: &mut HashMap<Key, C>) {
    map.retain(|_, c| c.norm2() > EPS * EPS);
}

fn norm2(map: &HashMap<Key, C>) -> f64 {
    map.values().map(|c| c.norm2()).sum()
}

impl Program {
    /// One shot, in full.
    pub fn run_full(&self, src: &mut dyn Source, memo: &mut Memo) -> Run {
        let (n, w) = (self.num_qubits, self.w);
        let mut tab = Tableau::new(n);
        let mut shot = Run {
            records: Vec::with_capacity(self.num_measurements),
            fired: vec![false; self.twirl.len()],
            terms: HashMap::from([(vec![0u64; w], C::ONE)]),
            frame: Pauli::identity(w),
            prob: 1.0,
            dead: false,
            peak: 1,
            trace: Trace::default(),
            chain: false,
        };
        for (i, step) in self.steps.iter().enumerate() {
            if shot.dead {
                break;
            }
            match step {
                Step::H(q) => {
                    tab.h(*q);
                    shot.frame.h(*q);
                }
                Step::S(q) => {
                    tab.s(*q);
                    shot.frame.s(*q);
                }
                Step::Cx(c, t) => {
                    tab.cx(*c, *t);
                    shot.frame.cx(*c, *t);
                }
                Step::Cz(a, b) => {
                    tab.cz(*a, *b);
                    shot.frame.cz(*a, *b);
                }
                Step::PauliGate(p) => {
                    tab.conjugate_by(p);
                    shot.frame.conjugate_by_pauli(p);
                }
                Step::CliffordRotation { pauli, inverse } => {
                    tab.clifford_rotation(pauli, *inverse);
                    shot.frame.conjugate_by_rotation(pauli, *inverse);
                }
                Step::Branch(terms) => self.branch(i, &tab, &mut shot, terms, memo),
                Step::Twirl(loc) => {
                    let (p, theta) = &self.twirl[*loc];
                    let f = src.fire(*loc, theta.sin().powi(2));
                    shot.fired[*loc] = f;
                    if f {
                        shot.frame.left_mul(p);
                    }
                }
                Step::Measure { q, basis, reset, flip } => {
                    let r = self.measure(i, &mut tab, &mut shot, *q, *basis, *reset, Some(*flip), src, memo);
                    shot.records.push(r);
                }
                Step::Reset { q, basis } => {
                    self.measure(i, &mut tab, &mut shot, *q, *basis, true, None, src, memo);
                }
                Step::Channel(branches) => {
                    let mut probs: Vec<f64> = branches.iter().map(|b| b.0).collect();
                    probs.push(1.0 - probs.iter().sum::<f64>());
                    let k = src.choose(&probs);
                    if k < branches.len() {
                        shot.frame.left_mul(&Pauli::from_codes(w, &branches[k].1));
                    }
                }
                Step::Correlated { p, paulis, chained } => {
                    let fires = if *chained && shot.chain { false } else { src.choose(&[*p, 1.0 - p]) == 0 };
                    shot.chain = if *chained { shot.chain || fires } else { fires };
                    if fires {
                        shot.frame.left_mul(&Pauli::from_codes(w, paulis));
                    }
                }
                Step::Pad { flip, value } => {
                    let k = src.choose(&[1.0 - flip, *flip]);
                    shot.records.push(*value ^ (k == 1));
                }
                Step::Herald { q, probs } => {
                    let mut v = probs.to_vec();
                    v.push(1.0 - probs.iter().sum::<f64>());
                    let k = src.choose(&v);
                    shot.records.push(k < 4);
                    if (1..4).contains(&k) {
                        shot.frame.left_mul(&Pauli::from_codes(w, &[(*q, [0, 1, 3, 2][k])]));
                    }
                }
                Step::Feedback { pauli, rec, q } => {
                    if shot.records[*rec] {
                        shot.frame.left_mul(&Pauli::from_codes(w, &[(*q, *pauli)]));
                    }
                }
            }
        }
        shot
    }

    /// One shot: its records, and the most terms it held at once.
    pub(crate) fn run_shot(&self, src: &mut dyn Source, memo: &mut Memo) -> (Vec<bool>, usize) {
        let run = self.run_full(src, memo);
        (run.records, run.peak)
    }

    /// A run's final terms with its state brought to the frame `frame` — the state is
    /// frame · Σ c′_k D(k)|ref⟩ — and the trajectory's probability in their norm, so that two
    /// runs' overlap is the dot product of their terms. Empty for a dead run.
    pub fn in_frame(&self, run: &Run, frame: &Pauli) -> HashMap<Key, C> {
        if run.dead {
            return HashMap::new();
        }
        let scale = C::new(run.prob.sqrt(), 0.0);
        let mut q = frame.dagger();
        q.mul_assign(&run.frame);
        let mut out: HashMap<Key, C> = HashMap::with_capacity(run.terms.len());
        if q.is_identity() {
            let phase = PHASES[q.k as usize] * scale;
            for (k, c) in &run.terms {
                out.insert(k.clone(), *c * phase);
            }
            return out;
        }
        for (k, c) in &run.terms {
            let mut p = q.clone();
            p.mul_assign(&self.final_tab.representative(k));
            let (k2, m) = self.final_tab.express(&p);
            add_term(&mut out, k2, *c * PHASES[m as usize] * scale);
        }
        out
    }

    /// A non-Clifford gate on the terms: U = Σ a_j P_j, conjugated by the frame (a_j signed by
    /// whether P_j anticommutes with it), each Pauli taking a term to its coset.
    fn branch(&self, step: usize, tab: &Tableau, shot: &mut Run, terms: &[(C, Pauli)], memo: &mut Memo) {
        let mut out: HashMap<Key, C> = HashMap::with_capacity(shot.terms.len() * 2);
        let signs: Vec<bool> = terms.iter().map(|(_, p)| shot.frame.anticommutes(p)).collect();
        for (key, c) in &shot.terms {
            for (j, ((a, p), &flip)) in terms.iter().zip(&signs).enumerate() {
                let amp = if flip { C::ZERO - *a } else { *a } * *c;
                if p.is_identity() {
                    add_term(&mut out, key.clone(), amp);
                    continue;
                }
                let (k2, m) = memo
                    .branch
                    .entry((step, j, key.clone()))
                    .or_insert_with(|| {
                        let mut q = p.clone();
                        q.mul_assign(&tab.representative(key));
                        tab.express(&q)
                    })
                    .clone();
                add_term(&mut out, k2, amp * PHASES[m as usize]);
            }
        }
        prune(&mut out);
        shot.terms = out;
        shot.peak = shot.peak.max(shot.terms.len());
    }

    /// Each term's eigenvalue of the measured Pauli, 0 or 1, as the represented state sees it,
    /// given the destabilizers' anticommutation `delta` and the reference's own outcome.
    fn term_outcome(key: &[u64], delta: &[u64], reference: bool) -> bool {
        reference ^ parity_and(key, delta)
    }

    /// The terms after a random measurement collapses the reference, those with eigenvalue `v`:
    /// Π_v D(k)|ref⟩ = D(k) g^{v ⊕ a_k} |ref′⟩ / √2 for the pivot's old stabilizer g, re-expressed
    /// in the new destabilizers.
    #[allow(clippy::too_many_arguments)]
    fn rebase(&self, step: usize, terms: &HashMap<Key, C>, v: bool, delta: &[u64], old: &Tableau, g: &Pauli, tab: &Tableau, memo: &mut Memo) -> HashMap<Key, C> {
        let mut out: HashMap<Key, C> = HashMap::with_capacity(terms.len());
        let s2 = std::f64::consts::FRAC_1_SQRT_2;
        for (key, c) in terms {
            let s = v ^ parity_and(key, delta);
            let (k2, m) = memo
                .measure
                .entry((step, s, key.clone()))
                .or_insert_with(|| {
                    let mut p = old.representative(key);
                    if s {
                        p.mul_assign(g);
                    }
                    tab.express(&p)
                })
                .clone();
            add_term(&mut out, k2, c.scale(s2) * PHASES[m as usize]);
        }
        prune(&mut out);
        out
    }

    /// A measurement (recorded with its flip probability when `flip` is given) or a reset
    /// (`reset`, and nothing recorded) of qubit `q` in `basis`: the record (or the unread
    /// outcome).
    #[allow(clippy::too_many_arguments)]
    fn measure(&self, step: usize, tab: &mut Tableau, shot: &mut Run, q: usize, basis: Basis, reset: bool, flip: Option<f64>, src: &mut dyn Source, memo: &mut Memo) -> bool {
        let code = match basis {
            Basis::Z => 2,
            Basis::X => 1,
        };
        let m = Pauli::from_codes(self.w, &[(q, code)]);
        // Whether the frame flips this outcome.
        let f = shot.frame.anticommutes(&m);
        let delta = tab.dkey_of(&m);
        let draw = |src: &mut dyn Source, p1_rep: f64| -> (bool, bool) {
            // The real outcome is the represented one flipped by the frame.
            let p1 = if f { 1.0 - p1_rep } else { p1_rep };
            match flip {
                Some(fl) => src.outcome(p1.clamp(0.0, 1.0), fl),
                None => {
                    let o = src.choose(&[1.0 - p1, p1]) == 1;
                    (o, o)
                }
            }
        };
        let (outcome, record) = match tab.pivot(&m) {
            None => {
                let r = tab.deterministic_outcome(&m);
                let p1_rep: f64 = shot.terms.iter().filter(|(k, _)| Self::term_outcome(k, &delta, r)).map(|(_, c)| c.norm2()).sum();
                let (outcome, record) = draw(src, p1_rep);
                let v = outcome ^ f;
                shot.terms.retain(|k, _| Self::term_outcome(k, &delta, r) == v);
                shot.renormalise(if v { p1_rep } else { 1.0 - p1_rep });
                (outcome, record)
            }
            Some(p) => {
                let old = tab.clone();
                let g = tab.row(tab.n + p);
                tab.collapse(&m, p);
                let zero = self.rebase(step, &shot.terms, false, &delta, &old, &g, tab, memo);
                let p0 = norm2(&zero);
                let (outcome, record) = draw(src, 1.0 - p0);
                let v = outcome ^ f;
                let (map, prob) = if v { (self.rebase(step, &shot.terms, true, &delta, &old, &g, tab, memo), 1.0 - p0) } else { (zero, p0) };
                shot.terms = map;
                shot.renormalise(prob);
                (outcome, record)
            }
        };
        // After a reset the real state is |0⟩ (|+⟩): the frame takes up the outcome's flip.
        if reset && outcome {
            shot.frame.left_mul(&Pauli::from_codes(self.w, &[(q, if basis == Basis::Z { 1 } else { 2 })]));
        }
        record
    }

    /// One shot's detection events and observable flips.
    pub fn sample(&self, rng: &mut Xorshift, memo: &mut Memo) -> (Vec<bool>, u64) {
        let (rec, _) = self.run_shot(&mut Stream(rng), memo);
        self.outcome(&rec)
    }

    /// One shot's measurement record, and the most terms it held at once.
    pub fn sample_record(&self, rng: &mut Xorshift, memo: &mut Memo) -> (Vec<bool>, usize) {
        self.run_shot(&mut Stream(rng), memo)
    }

    /// The probability of a measurement record: exact for a circuit without noise channels or
    /// flips; with them, an average over `samples` draws of the noise.
    pub fn probability_of(&self, record: &[bool], samples: usize, seed: u64) -> Result<f64, String> {
        if record.len() != self.num_measurements {
            return Err(format!("the record has {} bits; the circuit has {} measurements", record.len(), self.num_measurements));
        }
        let mut memo = Memo::default();
        let runs = if self.noisy { samples.max(1) } else { 1 };
        let mut total = 0.0;
        for s in 0..runs {
            let mut rng = Xorshift::new(crate::batch_sampler::batch_seed(seed, s as u64));
            let mut src = Forced { record, next: 0, prob: 1.0, rng: &mut rng };
            let run = self.run_full(&mut src, &mut memo);
            if !run.dead {
                total += src.prob;
            }
        }
        Ok(total / runs as f64)
    }

    /// Every outcome's exact probability, following every branch: each noise draw, each
    /// measurement outcome and flip, each reset's unread outcome. At most `max_branches`.
    pub fn distribution(&self, max_branches: usize) -> Result<BTreeMap<(Vec<bool>, u64), f64>, String> {
        let mut out = BTreeMap::new();
        self.each_branch(max_branches, &mut |rec, p| *out.entry(self.outcome(rec)).or_insert(0.0) += p)?;
        Ok(out)
    }

    /// The same for measurement records: {record: probability}.
    pub fn record_distribution(&self, max_branches: usize) -> Result<BTreeMap<Vec<bool>, f64>, String> {
        let mut out = BTreeMap::new();
        self.each_branch(max_branches, &mut |rec, p| *out.entry(rec.to_vec()).or_insert(0.0) += p)?;
        Ok(out)
    }

    /// Every branch of the circuit (noise draws, outcomes, flips, twirled errors firing), with
    /// its probability, to `visit(records, probability)`.
    pub fn each_branch(&self, max_branches: usize, visit: &mut dyn FnMut(&[bool], f64)) -> Result<(), String> {
        let mut memo = Memo::default();
        let mut odo = Odometer::default();
        let mut branches = 0usize;
        loop {
            odo.restart();
            let (rec, _) = self.run_shot(&mut odo, &mut memo);
            if odo.p > 1e-15 {
                visit(&rec, odo.p);
            }
            branches += 1;
            if branches > max_branches {
                return Err(format!("more than {max_branches} branches"));
            }
            if !odo.advance() {
                return Ok(());
            }
        }
    }
}

/// Forced measurement outcomes (a record's), noise drawn at random; `prob` is the record's
/// probability along the draw.
struct Forced<'a> {
    record: &'a [bool],
    next: usize,
    prob: f64,
    rng: &'a mut Xorshift,
}

impl Source for Forced<'_> {
    fn choose(&mut self, probs: &[f64]) -> usize {
        Stream(self.rng).choose(probs)
    }

    fn outcome(&mut self, p1: f64, flip: f64) -> (bool, bool) {
        let want = self.record[self.next];
        self.next += 1;
        let flipped = flip > 0.0 && self.rng.next_f64() < flip;
        let outcome = want ^ flipped;
        self.prob *= if outcome { p1 } else { 1.0 - p1 };
        (outcome, want)
    }

    fn fire(&mut self, _loc: usize, p: f64) -> bool {
        self.rng.next_f64() < p
    }
}

/// Every branch of a shot, by replaying it with its choices counted like an odometer.
#[derive(Default)]
pub struct Odometer {
    choices: Vec<usize>,
    probs: Vec<Vec<f64>>,
    pos: usize,
    /// The branch's probability so far.
    pub p: f64,
}

impl Odometer {
    /// At the start of a branch.
    pub fn restart(&mut self) {
        self.pos = 0;
        self.p = 1.0;
    }

    fn pick(&mut self, probs: Vec<f64>) -> usize {
        if self.pos == self.choices.len() {
            let k = probs.iter().position(|&p| p > 0.0).unwrap_or(0);
            self.choices.push(k);
            self.probs.push(probs);
        }
        let k = self.choices[self.pos];
        self.p *= self.probs[self.pos][k];
        self.pos += 1;
        k
    }

    /// The next branch; false when every one has run.
    pub fn advance(&mut self) -> bool {
        while let Some(&k) = self.choices.last() {
            let probs = self.probs.last().unwrap();
            if let Some(next) = (k + 1..probs.len()).find(|&j| probs[j] > 0.0) {
                *self.choices.last_mut().unwrap() = next;
                return true;
            }
            self.choices.pop();
            self.probs.pop();
        }
        false
    }
}

impl Source for Odometer {
    fn choose(&mut self, probs: &[f64]) -> usize {
        self.pick(probs.to_vec())
    }

    fn outcome(&mut self, p1: f64, flip: f64) -> (bool, bool) {
        let k = self.pick(vec![(1.0 - p1) * (1.0 - flip), (1.0 - p1) * flip, p1 * (1.0 - flip), p1 * flip]);
        (k >= 2, (k >= 2) ^ (k % 2 == 1))
    }

    fn fire(&mut self, _loc: usize, p: f64) -> bool {
        self.pick(vec![1.0 - p, p]) == 1
    }
}

/// Shots `first..first + shots`, shot s from its own stream (`batch_seed(seed, s)`), so threads
/// never change them: each shot's record, and the most terms any shot held.
pub fn records_seeded(program: &Program, seed: u64, first: u64, shots: usize, threads: usize) -> (Vec<Vec<bool>>, usize) {
    let rows = crate::parallel::parallel(shots, threads, |range| {
        let mut memo = Memo::default();
        range
            .map(|s| {
                let mut rng = Xorshift::new(crate::batch_sampler::batch_seed(seed, first + s as u64));
                program.sample_record(&mut rng, &mut memo)
            })
            .collect()
    });
    let peak = rows.iter().map(|r| r.1).max().unwrap_or(0);
    (rows.into_iter().map(|r| r.0).collect(), peak)
}

fn pack(rows: impl Iterator<Item = Vec<bool>>, shots: usize, bits: usize) -> Vec<u8> {
    let stride = bits.div_ceil(8);
    let mut out = vec![0u8; shots * stride];
    for (s, row) in rows.enumerate() {
        for (k, &b) in row.iter().enumerate() {
            if b {
                out[s * stride + k / 8] |= 1 << (k % 8);
            }
        }
    }
    out
}

/// Detection events and observable flips as b8 rows, and the peak term count.
pub fn sample_seeded(program: &Program, seed: u64, first: u64, shots: usize, threads: usize) -> (Vec<u8>, Vec<u8>, usize) {
    let (records, peak) = records_seeded(program, seed, first, shots, threads);
    let (nd, no) = (program.num_detectors(), program.num_observables());
    let outcomes: Vec<(Vec<bool>, u64)> = records.iter().map(|r| program.outcome(r)).collect();
    let d = pack(outcomes.iter().map(|o| o.0.clone()), shots, nd);
    let o = pack(outcomes.iter().map(|o| (0..no).map(|k| o.1 >> k & 1 == 1).collect()), shots, no);
    (d, o, peak)
}

/// Measurement records as b8 rows, and the peak term count.
pub fn measurements_seeded(program: &Program, seed: u64, first: u64, shots: usize, threads: usize) -> (Vec<u8>, usize) {
    let (records, peak) = records_seeded(program, seed, first, shots, threads);
    (pack(records.into_iter(), shots, program.num_measurements()), peak)
}

/// Whether exp(−iθP) is a Clifford: θ a multiple of π/4 (within 1e-9), as the sampler reads it.
pub fn is_clifford_angle(theta: f64) -> bool {
    let r = theta / FRAC_PI_4;
    (r - r.round()).abs() < 1e-9
}

/// `f` summed over every non-Pauli op of the circuit, through its loops without unrolling them
/// (a REPEAT body counts once per repetition; a million-round memory is not expanded).
fn sum_non_pauli(instrs: &[Instr], f: &dyn Fn(&NonPauli) -> usize) -> usize {
    let mut n = 0usize;
    for ins in instrs {
        match ins {
            Instr::NonPauli(ops) => n = n.saturating_add(ops.iter().map(f).sum()),
            Instr::Repeat { count, body, .. } => n = n.saturating_add(sum_non_pauli(body, f).saturating_mul(*count as usize)),
            Instr::Gate { body, .. } => n = n.saturating_add(sum_non_pauli(body, f)),
            _ => {}
        }
    }
    n
}

/// How many non-Clifford gates a circuit has: its T gates, and the rotations and U3 half-angle
/// rotations written as gates (on `I`, `II`) at angles that are not Clifford. Coherent errors
/// (`I_ERROR`, `II_ERROR`) are not gates and are not counted.
pub fn non_clifford_gates(circuit: &Circuit) -> Result<usize, String> {
    Ok(sum_non_pauli(&circuit.instrs, &|op| match op {
        NonPauli::T { .. } => 1,
        NonPauli::Rotation { pauli, theta, noise: false } if !pauli.is_empty() && !is_clifford_angle(*theta) => 1,
        NonPauli::U3 { theta, phi, lambda, .. } => [lambda, theta, phi].iter().filter(|a| !is_clifford_angle(**a / 2.0)).count(),
        _ => 0,
    }))
}

/// The circuit's T-count as Tsim (pyzx) counts it: its non-Clifford phases, one for each T
/// and T_DAG gate and each rotation (a gate or a coherent error, each of a `U3`'s three) by an
/// angle that is not a multiple of π/2 in half-turns (of π/4 in exp(−iθP)), whatever the
/// angle. Coherent errors count here and not in Tsim, which reads `I_ERROR[R_Z(...)]` as
/// nothing: the count is of the phases the circuit's exact simulation pays for.
pub fn t_count(circuit: &Circuit) -> Result<usize, String> {
    Ok(sum_non_pauli(&circuit.instrs, &|op| match op {
        NonPauli::T { .. } => 1,
        NonPauli::Rotation { pauli, theta, .. } if !pauli.is_empty() && !is_clifford_angle(*theta) => 1,
        NonPauli::U3 { theta, phi, lambda, .. } => [lambda, theta, phi].iter().filter(|a| !is_clifford_angle(**a / 2.0)).count(),
        _ => 0,
    }))
}

/// Whether every gate and coherent error of the circuit is Clifford: no T gate, and every
/// rotation (a `U3`'s three among them) by a multiple of π/2 in half-turns, of π/4 in
/// exp(−iθP). Noise channels, damping and leakage are not gates and do not count.
pub fn is_clifford(circuit: &Circuit) -> Result<bool, String> {
    Ok(sum_non_pauli(&circuit.instrs, &|op| match op {
        NonPauli::T { .. } => 1,
        NonPauli::Rotation { pauli, theta, .. } => usize::from(!(pauli.is_empty() || is_clifford_angle(*theta))),
        NonPauli::U3 { theta, phi, lambda, .. } => usize::from(![lambda, theta, phi].iter().all(|a| is_clifford_angle(**a / 2.0))),
        _ => 0,
    }) == 0)
}

/// Whether the circuit has a tagged gate at a Clifford angle (`I[R_Z(theta=π/4)] 0` is `S 0`,
/// `I[U3(theta=π, phi=0, lambda=π)] 0` is `X 0`): a Clifford gate Stim reads as nothing, which
/// every engine here runs as the gate it is (`frame_view`).
pub fn has_clifford_tagged_gates(circuit: &Circuit) -> bool {
    sum_non_pauli(&circuit.instrs, &|op| match op {
        NonPauli::Rotation { pauli, theta, noise: false } => usize::from(!pauli.is_empty() && is_clifford_angle(*theta)),
        NonPauli::U3 { theta, phi, lambda, .. } => usize::from([lambda, theta, phi].iter().any(|a| is_clifford_angle(**a / 2.0))),
        _ => 0,
    }) > 0
}

/// The circuit as the frame engines (the detector and measurement samplers, the reference run,
/// the detector error model) run it: with its tagged gates at Clifford angles written as the
/// Clifford gates they are (`expand_clifford_rotations`), so that `I[R_Z(theta=π/4)] 0` is
/// `S 0` in every engine and not, as Stim reads it, nothing. The circuit itself when it has
/// none, which costs one pass over its instructions.
pub fn frame_view(circuit: &Circuit) -> Result<Cow<'_, Circuit>, String> {
    if has_clifford_tagged_gates(circuit) {
        Ok(Cow::Owned(expand_clifford_rotations(circuit)?))
    } else {
        Ok(Cow::Borrowed(circuit))
    }
}

/// How many coherent errors (`I_ERROR`, `II_ERROR` rotations) are at Clifford angles: fixed
/// Cliffords (`I_ERROR[R_Z(theta=π/4)] 0` is an S), which the joint sampler runs exactly and the
/// coherent sampler, a twirl-and-reweight over a Clifford reference, cannot take.
pub fn clifford_angle_errors(circuit: &Circuit) -> usize {
    sum_non_pauli(&circuit.instrs, &|op| match op {
        NonPauli::Rotation { pauli, theta, noise: true } => usize::from(!pauli.is_empty() && is_clifford_angle(*theta)),
        _ => 0,
    })
}

/// The circuit with every tagged gate at a Clifford angle written as the Clifford gate it is
/// (`I[R_Z(theta=π/4)] 0` is `S 0`, `I[R_PAULI(theta=π/4, pauli=XZ)] 0 1` is `SPP X0*Z1`, a
/// multiple of π/2 a Pauli gate), and every `U3` as its three rotations, each so written when
/// Clifford; the other gates and the coherent errors as they are (an error at a Clifford angle
/// stays an error: the reference run must not see it). Loops are kept.
pub fn expand_clifford_rotations(circuit: &Circuit) -> Result<Circuit, String> {
    fn lines_of(pauli: &[(u32, u8)], theta: f64, out: &mut Vec<String>) {
        if pauli.is_empty() {
            return;
        }
        let letters = |c: u8| ['I', 'X', 'Z', 'Y'][c as usize];
        if !is_clifford_angle(theta) {
            let qs: Vec<String> = pauli.iter().map(|&(q, _)| q.to_string()).collect();
            let ls: String = pauli.iter().map(|&(_, c)| letters(c)).collect();
            out.push(format!("I[R_PAULI(theta={theta:?}, pauli={ls})] {}", qs.join(" ")));
            return;
        }
        let k = (theta / FRAC_PI_4).round() as i64;
        match k.rem_euclid(8) {
            0 | 4 => {}
            2 | 6 => {
                for &(q, c) in pauli {
                    out.push(format!("{} {q}", letters(c)));
                }
            }
            odd => {
                let product: Vec<String> = pauli.iter().map(|&(q, c)| format!("{}{q}", letters(c))).collect();
                out.push(format!("{} {}", if odd == 1 || odd == 5 { "SPP" } else { "SPP_DAG" }, product.join("*")));
            }
        }
    }
    fn walk(instrs: &[Instr]) -> Result<Vec<Instr>, String> {
        let mut out = Vec::new();
        for ins in instrs {
            match ins {
                Instr::Repeat { count, body, tag } => out.push(Instr::Repeat { count: *count, body: walk(body)?, tag: tag.clone() }),
                Instr::Gate { body, .. } if matches!(body.as_slice(), [Instr::NonPauli(_)]) => {
                    let Instr::NonPauli(ops) = &body[0] else { unreachable!() };
                    let gate = ops.iter().all(|op| matches!(op, NonPauli::Rotation { noise: false, .. } | NonPauli::U3 { .. }));
                    if !gate {
                        out.push(ins.clone());
                        continue;
                    }
                    let mut lines = Vec::new();
                    for op in ops {
                        match op {
                            NonPauli::Rotation { pauli, theta, .. } => lines_of(pauli, *theta, &mut lines),
                            NonPauli::U3 { qubit, theta, phi, lambda } => {
                                lines_of(&[(*qubit, 2)], lambda / 2.0, &mut lines);
                                lines_of(&[(*qubit, 3)], theta / 2.0, &mut lines);
                                lines_of(&[(*qubit, 2)], phi / 2.0, &mut lines);
                            }
                            _ => unreachable!(),
                        }
                    }
                    out.extend(Circuit::parse(&lines.join("\n"))?.instrs);
                }
                other => out.push(other.clone()),
            }
        }
        Ok(out)
    }
    Ok(Circuit { instrs: walk(&circuit.instrs)? })
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    fn dist(text: &str) -> BTreeMap<(Vec<bool>, u64), f64> {
        Program::new(&Circuit::parse(text).unwrap()).unwrap().distribution(1 << 20).unwrap()
    }

    pub(crate) fn close(a: &BTreeMap<(Vec<bool>, u64), f64>, b: &BTreeMap<(Vec<bool>, u64), f64>, text: &str) {
        let keys: std::collections::BTreeSet<_> = a.keys().chain(b.keys()).collect();
        for k in keys {
            let (x, y) = (a.get(k).copied().unwrap_or(0.0), b.get(k).copied().unwrap_or(0.0));
            assert!((x - y).abs() < 1e-10, "{text}\n{k:?}: {x} vs {y}");
        }
    }

    #[test]
    fn pauli_algebra() {
        let w = 1;
        let x = Pauli::from_codes(w, &[(0, 1)]);
        let z = Pauli::from_codes(w, &[(0, 2)]);
        let y = Pauli::from_codes(w, &[(0, 3)]);
        // XZ is the basis element X^1 Z^1 itself (−iY: Y is i·XZ, k = 1); ZX = −XZ.
        let mut xz = x.clone();
        xz.mul_assign(&z);
        assert_eq!((xz.x, xz.z, xz.k), (y.x.clone(), y.z.clone(), 0));
        assert_eq!(y.k, 1);
        let mut zx = z.clone();
        zx.mul_assign(&x);
        assert_eq!(zx.k, 2);
        let mut yz = y.clone();
        yz.mul_assign(&z); // YZ = iX
        assert_eq!((yz.x, yz.z, yz.k), (x.x.clone(), x.z.clone(), 1));
        let mut yy = y.clone();
        yy.mul_assign(&y);
        assert!(yy.is_identity() && yy.k == 0);
        assert_eq!(y.dagger(), y);
        assert!(x.anticommutes(&z) && !x.anticommutes(&x));
        // Conjugations: S X S† = Y; H Y H = −Y; CX takes X⊗I to X⊗X; CZ takes X⊗X to Y⊗Y.
        let mut p = x.clone();
        p.s(0);
        assert_eq!(p, y);
        let mut p = y.clone();
        p.h(0);
        let mut my = y.clone();
        my.k = (my.k + 2) % 4;
        assert_eq!(p, my);
        let mut p = Pauli::from_codes(1, &[(0, 1)]);
        p.cx(0, 1);
        assert_eq!(p, Pauli::from_codes(1, &[(0, 1), (1, 1)]));
        let mut p = Pauli::from_codes(1, &[(0, 1), (1, 1)]);
        p.cz(0, 1);
        assert_eq!(p, Pauli::from_codes(1, &[(0, 3), (1, 3)]));
    }

    #[test]
    fn t_on_zero_is_one_term_and_h_t_h_rotates() {
        let p = Program::new(&Circuit::parse("R 0\nI[T] 0\nM 0\nDETECTOR rec[-1]").unwrap()).unwrap();
        let mut memo = Memo::default();
        let (rec, peak) = p.sample_record(&mut Xorshift::new(1), &mut memo);
        assert_eq!((rec, peak), (vec![false], 1));
        let d = dist("R 0\nH 0\nI[T] 0\nH 0\nM 0\nDETECTOR rec[-1]");
        assert!((d[&(vec![true], 0)] - FRAC_PI_8.sin().powi(2)).abs() < 1e-12);
        // T T = S, exactly a Clifford: H S H |0⟩ measures 1 with probability 1/2.
        let d = dist("R 0\nH 0\nI[T] 0\nI[T] 0\nH 0\nM 0\nDETECTOR rec[-1]");
        assert!((d[&(vec![true], 0)] - 0.5).abs() < 1e-12);
    }

    #[test]
    fn clifford_angles_split_nothing() {
        let text = "R 0 1\nI[R_X(theta=0.7853981633974483)] 0\nII[R_ZZ(theta=1.5707963267948966)] 0 1\nI[R_Y(theta=-2.356194490192345)] 1\nM 0 1\nDETECTOR rec[-1]\nDETECTOR rec[-2]";
        let p = Program::new(&Circuit::parse(text).unwrap()).unwrap();
        assert_eq!(p.num_non_clifford(), 0);
        let sv = crate::statevec::Program::new(&Circuit::parse(text).unwrap()).unwrap().distribution(1 << 10).unwrap();
        close(&dist(text), &sv, "clifford angles");
        // Expanded, the same circuit is written in Stim's gates.
        let e = expand_clifford_rotations(&Circuit::parse(text).unwrap()).unwrap();
        assert_eq!(non_clifford_gates(&e).unwrap(), 0);
        assert!(!format!("{e:?}").contains("NonPauli"));
        close(&Program::new(&e).unwrap().distribution(1 << 10).unwrap(), &sv, "expanded");
    }

    #[test]
    fn noise_and_frame_rules() {
        let d = dist("R 0\nX_ERROR(0.1) 0\nM(0.05) 0\nDETECTOR rec[-1]");
        assert!((d[&(vec![true], 0)] - (0.1 * 0.95 + 0.9 * 0.05)).abs() < 1e-12);
        // A Z error before a T on |+⟩: the rotation's sense flips, P(1) = cos².
        let d = dist("R 0\nH 0\nZ_ERROR(1) 0\nI[T] 0\nH 0\nM 0\nDETECTOR rec[-1]");
        assert!((d[&(vec![true], 0)] - FRAC_PI_8.cos().powi(2)).abs() < 1e-12, "{d:?}");
    }

    /// A random circuit of Clifford gates, T gates, rotations, U3, noise, measurements in both
    /// bases, resets and feedback; with `errors`, coherent errors too.
    pub(crate) fn random_circuit(rng: &mut Xorshift, n: usize, len: usize, errors: bool) -> String {
        let mut lines = vec![format!("R {}", (0..n).map(|q| q.to_string()).collect::<Vec<_>>().join(" "))];
        let mut nm = 0usize;
        let q = |rng: &mut Xorshift| rng.next_u64() as usize % n;
        for _ in 0..len {
            let u = rng.next_f64();
            let a = q(rng);
            let line = if u < 0.30 {
                format!("{} {a}", ["H", "S", "X", "Y", "Z", "S_DAG", "SQRT_X", "H_YZ", "C_XYZ"][rng.next_u64() as usize % 9])
            } else if u < 0.50 && n > 1 {
                let mut b = q(rng);
                while b == a {
                    b = q(rng);
                }
                format!("{} {a} {b}", ["CX", "CZ", "CY", "SWAP", "ISWAP", "SQRT_XX"][rng.next_u64() as usize % 6])
            } else if u < 0.66 {
                match rng.next_u64() % 5 {
                    0 => format!("I[T] {a}"),
                    1 => format!("I[T_DAG] {a}"),
                    2 => format!("I[R_{}(theta={:?})] {a}", ['X', 'Y', 'Z'][rng.next_u64() as usize % 3], rng.next_f64() * 3.0 - 1.5),
                    3 => format!("I[U3(theta={:?}, phi={:?}, lambda={:?})] {a}", rng.next_f64() * 6.0, rng.next_f64() * 6.0, rng.next_f64() * 6.0),
                    _ if n > 1 => {
                        let mut b = q(rng);
                        while b == a {
                            b = q(rng);
                        }
                        format!("II[R_{}(theta={:?})] {a} {b}", ["XX", "YY", "ZZ"][rng.next_u64() as usize % 3], rng.next_f64() * 3.0 - 1.5)
                    }
                    _ => format!("I[T] {a}"),
                }
            } else if u < 0.78 {
                match rng.next_u64() % 5 {
                    0 => format!("DEPOLARIZE1(0.2) {a}"),
                    1 => format!("X_ERROR(0.15) {a}"),
                    2 if n > 1 => {
                        let mut b = q(rng);
                        while b == a {
                            b = q(rng);
                        }
                        format!("DEPOLARIZE2(0.1) {a} {b}")
                    }
                    3 if errors => format!("I_ERROR[R_{}(theta={:?})] {a}", ['X', 'Y', 'Z'][rng.next_u64() as usize % 3], rng.next_f64() * 0.8 - 0.4),
                    _ => format!("PAULI_CHANNEL_1(0.1, 0.05, 0.02) {a}"),
                }
            } else if u < 0.92 {
                nm += 1;
                match rng.next_u64() % 5 {
                    0 => format!("M {a}"),
                    1 => format!("MX {a}"),
                    2 => format!("MR {a}"),
                    3 => format!("M(0.1) {a}"),
                    _ => format!("MRX {a}"),
                }
            } else if u < 0.97 {
                format!("{} {a}", ["R", "RX"][rng.next_u64() as usize % 2])
            } else if nm > 0 {
                format!("CX rec[-{}] {a}", 1 + rng.next_u64() as usize % nm.min(3))
            } else {
                format!("H {a}")
            };
            lines.push(line);
        }
        for a in 0..n {
            lines.push(format!("M {a}"));
            nm += 1;
        }
        for k in 1..=nm.min(3) {
            lines.push(format!("DETECTOR rec[-{k}]"));
        }
        lines.push(format!("OBSERVABLE_INCLUDE(0) rec[-1] rec[-{}]", nm.min(2)));
        lines.join("\n")
    }

    #[test]
    fn random_circuits_equal_the_state_vector() {
        let mut rng = Xorshift::new(2026);
        for case in 0..120 {
            let n = 1 + case % 3;
            let text = random_circuit(&mut rng, n, 6 + case % 7, case % 2 == 1);
            let c = Circuit::parse(&text).unwrap();
            let sv = crate::statevec::Program::new(&c).unwrap().distribution(1 << 22).unwrap();
            let ours = Program::new(&c).unwrap().distribution(1 << 22).unwrap();
            close(&ours, &sv, &text);
            // Expanded, the same distribution.
            let e = expand_clifford_rotations(&c).unwrap();
            close(&Program::new(&e).unwrap().distribution(1 << 22).unwrap(), &sv, &text);
        }
    }

    #[test]
    fn samples_follow_the_distribution() {
        let text = "R 0 1\nH 0\nI[T] 0\nI[R_X(theta=0.4)] 1\nCX 0 1\nI[T] 1\nH 1\nM 0 1\nDETECTOR rec[-1] rec[-2]";
        let prog = Program::new(&Circuit::parse(text).unwrap()).unwrap();
        let p = prog.distribution(1000).unwrap()[&(vec![true], 0)];
        let mut rng = Xorshift::new(7);
        let mut memo = Memo::default();
        let n = 20000;
        let hits = (0..n).filter(|_| prog.sample(&mut rng, &mut memo).0[0]).count() as f64;
        let sigma = (p * (1.0 - p) / n as f64).sqrt();
        assert!((hits / n as f64 - p).abs() < 5.0 * sigma, "{} vs {p}", hits / n as f64);
    }

    #[test]
    fn probability_of_a_record() {
        let text = "R 0 1\nH 0\nI[T] 0\nCX 0 1\nI[T_DAG] 1\nH 1\nM 0 1";
        let prog = Program::new(&Circuit::parse(text).unwrap()).unwrap();
        let d = prog.record_distribution(1000).unwrap();
        let mut total = 0.0;
        for (rec, p) in &d {
            assert!((prog.probability_of(rec, 1, 0).unwrap() - p).abs() < 1e-12);
            total += p;
        }
        assert!((total - 1.0).abs() < 1e-12);
    }

    #[test]
    fn t_gates_on_a_code_state_merge() {
        // Seven T gates on the Steane code's |0_L⟩ (three X stabilizers): at most 2^3 terms.
        let text = "R 0 1 2 3 4 5 6\nH 4 5 6\nCX 4 0 4 1 4 2\nCX 5 0 5 1 5 3\nCX 6 0 6 2 6 3\nI[T] 0 1 2 3 4 5 6\nMX 0 1 2 3 4 5 6";
        let prog = Program::new(&Circuit::parse(text).unwrap()).unwrap();
        let mut memo = Memo::default();
        let (_, peak) = prog.sample_record(&mut Xorshift::new(3), &mut memo);
        assert!(peak <= 8, "{peak}");
    }

    #[test]
    fn many_qubits_few_t_gates() {
        let n = 200;
        let mut text = format!("R {}\n", (0..n).map(|q| q.to_string()).collect::<Vec<_>>().join(" "));
        text += "H 0\n";
        for q in 1..n {
            text += &format!("CX {} {q}\n", q - 1);
        }
        text += "I[T] 0 50 100 150\nH 0 50 100 150\n";
        text += &format!("M {}", (0..n).map(|q| q.to_string()).collect::<Vec<_>>().join(" "));
        let prog = Program::new(&Circuit::parse(&text).unwrap()).unwrap();
        let mut memo = Memo::default();
        let (rec, peak) = prog.sample_record(&mut Xorshift::new(5), &mut memo);
        assert_eq!(rec.len(), n);
        assert!(peak <= 16);
    }

    #[test]
    fn twirled_errors_and_frames_compare() {
        // With the error twirled and the firing summed with the twirl's probabilities, the
        // records are the twirled circuit's.
        let text = "R 0\nH 0\nI_ERROR[R_Z(theta=0.4)] 0\nI[T] 0\nH 0\nM 0\nDETECTOR rec[-1]";
        let c = Circuit::parse(text).unwrap();
        let p = Program::with_twirl(&c).unwrap();
        assert_eq!(p.num_twirl(), 1);
        let twirl = crate::statevec::Program::new(&crate::coherent::twirled(&c, false).unwrap()).unwrap().distribution(1 << 10).unwrap();
        close(&p.distribution(1 << 10).unwrap(), &twirl, text);
        // Two runs (the error fired or not), outcome 0 forced, brought to one frame: their
        // overlap is the states' overlap, ⟨0|H T H|+⟩* ⟨0|H T Z H|+⟩ in modulus.
        let mut memo = Memo::default();
        // The reset's unread outcome is the first draw, then the measurement's.
        let trace = Trace { choices: vec![0], outcomes: vec![(false, false)] };
        let a = p.run_full(&mut Replay::new(&trace, &[false]), &mut memo);
        let b = p.run_full(&mut Replay::new(&trace, &[true]), &mut memo);
        let (ta, tb) = (p.in_frame(&a, &a.frame), p.in_frame(&b, &a.frame));
        let dot: C = ta.iter().fold(C::ZERO, |acc, (k, c)| acc + c.conj() * tb.get(k).copied().unwrap_or(C::ZERO));
        let amp = |z: bool| {
            let t = Circuit::parse(&format!("H 0\n{}I[T] 0\nH 0", if z { "Z 0\n" } else { "" })).unwrap();
            let (_, m) = crate::statevec::unitary_matrix(&t, true).unwrap();
            m[0]
        };
        let expect = amp(false).conj() * amp(true);
        assert!((dot.norm2().sqrt() - expect.norm2().sqrt()).abs() < 1e-12, "{dot:?} vs {expect:?}");
        assert!((a.prob - amp(false).norm2()).abs() < 1e-12 && (b.prob - amp(true).norm2()).abs() < 1e-12);
    }

    #[test]
    fn gate_counts_walk_loops_without_unrolling() {
        // A million rounds: counted by multiplication, not by expanding the loop (which the
        // engine refuses past 2^24 instructions). The Clifford angle is not a gate to count; the
        // coherent error is a rotation for the T-count's purposes and not a gate for the other.
        let c = Circuit::parse("R 0\nREPEAT 1000000 {\n    H 0\n    REPEAT 40 {\n        I[T] 0\n        I[R_Z(theta=0.3)] 0\n        I[R_Z(theta=0.7853981633974483)] 0\n        I_ERROR[R_Z(theta=0.1)] 0\n    }\n    M 0\n}").unwrap();
        assert_eq!(non_clifford_gates(&c).unwrap(), 2 * 40 * 1_000_000);
        assert_eq!(t_count(&c).unwrap(), 3 * 40 * 1_000_000);
        assert!(!is_clifford(&c).unwrap());
        let clifford = Circuit::parse("R 0\nREPEAT 1000000 {\n    REPEAT 40 {\n        I[R_Z(theta=0.7853981633974483)] 0\n        X_ERROR(0.1) 0\n    }\n    M 0\n}").unwrap();
        assert_eq!(non_clifford_gates(&clifford).unwrap(), 0);
        assert!(is_clifford(&clifford).unwrap());
    }
}

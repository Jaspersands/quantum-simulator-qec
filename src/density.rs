//! A density-matrix simulator, for circuits of up to 12 qubits. It runs what the state vector
//! runs (every Stim gate, measurement, reset, noise channel and feedback, and the tagged
//! rotations, T, U3, amplitude and phase damping), noise as channels rather than drawn
//! trajectories: ρ → Σ_k K_k ρ K_k†.
//!
//! ρ is held vectorised, as the state vector of 2n qubits whose amplitude `i | j << n` is
//! `ρ[i][j]`: an operation K on the ket qubits and K̄ on the bra qubits n..2n is K ρ K†, so every
//! gate is the state vector's own.
//!
//! Measurements branch only on what they record. `distribution` follows each record (an
//! outcome and its flip merged into the bit they write) with its probability, every noise
//! channel applied whole on the way; `final_state` averages the records away, branching only
//! when feedback reads them.

use std::collections::BTreeMap;

use crate::circuit::{Basis, Circuit};
use crate::statevec::{Outcomes, Program, StateVector, Step, C, M2};
use crate::surface_code::Xorshift;

/// The most qubits the density matrix holds (4^12 entries, 256 MB).
pub const MAX_QUBITS: usize = 12;

fn conj(m: &M2) -> M2 {
    [[m[0][0].conj(), m[0][1].conj()], [m[1][0].conj(), m[1][1].conj()]]
}

fn re(x: f64) -> C {
    C::new(x, 0.0)
}

/// The Kraus operators of amplitude damping.
pub fn amplitude_damping(gamma: f64) -> [M2; 2] {
    [[[C::ONE, C::ZERO], [C::ZERO, re((1.0 - gamma).sqrt())]], [[C::ZERO, re(gamma.sqrt())], [C::ZERO, C::ZERO]]]
}

/// The Kraus operators of phase damping.
pub fn phase_damping(lambda: f64) -> [M2; 2] {
    [[[C::ONE, C::ZERO], [C::ZERO, re((1.0 - lambda).sqrt())]], [[C::ZERO, C::ZERO], [C::ZERO, re(lambda.sqrt())]]]
}

/// The Kraus operators of a Z-basis reset, |0⟩⟨0| and |0⟩⟨1|.
const RESET: [M2; 2] = [[[C::ONE, C::ZERO], [C::ZERO, C::ZERO]], [[C::ZERO, C::ONE], [C::ZERO, C::ZERO]]];

/// A density matrix on `n` qubits: `ρ[i][j]` with bit q of i (and j) qubit q.
#[derive(Clone, Debug)]
pub struct DensityMatrix {
    n: usize,
    v: StateVector,
}

impl DensityMatrix {
    /// |0…0⟩⟨0…0| on `n` qubits.
    pub fn new(n: usize) -> DensityMatrix {
        assert!(n <= MAX_QUBITS, "at most {MAX_QUBITS} qubits");
        DensityMatrix { n, v: StateVector::new(2 * n) }
    }

    /// ρ from its entries, rows then columns.
    pub fn from_matrix(n: usize, entries: &[C]) -> Result<DensityMatrix, String> {
        if n > MAX_QUBITS {
            return Err(format!("at most {MAX_QUBITS} qubits"));
        }
        let d = 1usize << n;
        if entries.len() != d * d {
            return Err(format!("{} entries for {n} qubits; expected {}", entries.len(), d * d));
        }
        let mut v = StateVector::zeros(2 * n);
        let a = v.amplitudes_mut();
        for i in 0..d {
            for j in 0..d {
                a[i | j << n] = entries[i * d + j];
            }
        }
        Ok(DensityMatrix { n, v })
    }

    pub fn num_qubits(&self) -> usize {
        self.n
    }

    /// `ρ[i][j]`.
    pub fn get(&self, i: usize, j: usize) -> C {
        self.v.amplitudes()[i | j << self.n]
    }

    /// The entries, rows then columns.
    pub fn matrix(&self) -> Vec<C> {
        let d = 1usize << self.n;
        let mut out = Vec::with_capacity(d * d);
        for i in 0..d {
            for j in 0..d {
                out.push(self.get(i, j));
            }
        }
        out
    }

    /// ρ ⊗ |0⟩⟨0| on qubits up to `n`.
    pub fn grow(&mut self, n: usize) {
        if n <= self.n {
            return;
        }
        assert!(n <= MAX_QUBITS, "at most {MAX_QUBITS} qubits");
        let d = 1usize << self.n;
        let mut v = StateVector::zeros(2 * n);
        let a = v.amplitudes_mut();
        for i in 0..d {
            for j in 0..d {
                a[i | j << n] = self.get(i, j);
            }
        }
        *self = DensityMatrix { n, v };
    }

    pub fn trace(&self) -> f64 {
        (0..1usize << self.n).map(|i| self.get(i, i).re).sum()
    }

    /// Tr ρ².
    pub fn purity(&self) -> f64 {
        self.v.amplitudes().iter().map(|a| a.norm2()).sum()
    }

    /// The probability of each basis state, ρ's diagonal.
    pub fn probabilities(&self) -> Vec<f64> {
        (0..1usize << self.n).map(|i| self.get(i, i).re).collect()
    }

    fn shifted(&self, paulis: &[(usize, u8)]) -> Vec<(usize, u8)> {
        paulis.iter().map(|&(q, p)| (q + self.n, p)).collect()
    }

    pub fn unitary_1q(&mut self, q: usize, m: &M2) {
        self.v.apply_1q(q, m);
        self.v.apply_1q(q + self.n, &conj(m));
    }

    pub fn h(&mut self, q: usize) {
        self.v.h(q);
        self.v.h(q + self.n);
    }

    pub fn s(&mut self, q: usize) {
        self.v.phase(q, C::I);
        self.v.phase(q + self.n, C::new(0.0, -1.0));
    }

    pub fn cx(&mut self, c: usize, t: usize) {
        self.v.cx(c, t);
        self.v.cx(c + self.n, t + self.n);
    }

    pub fn cz(&mut self, a: usize, b: usize) {
        self.v.cz(a, b);
        self.v.cz(a + self.n, b + self.n);
    }

    /// P ρ P for the Pauli product `paulis` (qubit, code X 1, Z 2, Y 3). P̄ = (−1)^{#Y} P.
    pub fn pauli(&mut self, paulis: &[(usize, u8)]) {
        self.v.pauli(paulis);
        self.v.pauli(&self.shifted(paulis));
        if paulis.iter().filter(|&&(_, p)| p == 3).count() % 2 == 1 {
            for a in self.v.amplitudes_mut() {
                *a = C::ZERO - *a;
            }
        }
    }

    /// exp(−iθP) ρ exp(iθP).
    pub fn rotate(&mut self, paulis: &[(usize, u8)], theta: f64) {
        self.v.rotate(paulis, theta);
        let sign = if paulis.iter().filter(|&&(_, p)| p == 3).count() % 2 == 1 { 1.0 } else { -1.0 };
        self.v.rotate(&self.shifted(paulis), sign * theta);
    }

    fn add_scaled(&mut self, other: &DensityMatrix, w: f64) {
        for (a, b) in self.v.amplitudes_mut().iter_mut().zip(other.v.amplitudes()) {
            *a = *a + b.scale(w);
        }
    }

    fn scale(&mut self, w: f64) {
        for a in self.v.amplitudes_mut() {
            *a = a.scale(w);
        }
    }

    /// Σ_k K_k ρ K_k† for single-qubit Kraus operators.
    pub fn kraus_1q(&mut self, q: usize, ks: &[M2]) {
        let mut out = DensityMatrix { n: self.n, v: StateVector::zeros(2 * self.n) };
        for k in ks {
            let mut t = self.clone();
            t.unitary_1q(q, k);
            out.add_scaled(&t, 1.0);
        }
        *self = out;
    }

    /// (1 − Σp) ρ + Σ p_k P_k ρ P_k.
    pub fn pauli_channel(&mut self, branches: &[(f64, Vec<(usize, u8)>)]) {
        let rest = 1.0 - branches.iter().map(|b| b.0).sum::<f64>();
        let mut out = self.clone();
        out.scale(rest);
        for (p, paulis) in branches {
            if *p == 0.0 {
                continue;
            }
            let mut t = self.clone();
            t.pauli(paulis);
            out.add_scaled(&t, *p);
        }
        *self = out;
    }

    /// U on `qubits` (U a 2^k by 2^k matrix, rows then columns, `qubits[0]` the least
    /// significant bit of its index): U ρ U†.
    pub fn apply_matrix(&mut self, qubits: &[usize], u: &[C]) -> Result<(), String> {
        let k = qubits.len();
        let d = 1usize << k;
        if u.len() != d * d {
            return Err(format!("{} entries for a {k}-qubit matrix; expected {}", u.len(), d * d));
        }
        if qubits.iter().any(|&q| q >= self.n) || (1..k).any(|i| qubits[..i].contains(&qubits[i])) {
            return Err(format!("qubits must be distinct and below {}", self.n));
        }
        let conj_u: Vec<C> = u.iter().map(|c| c.conj()).collect();
        let n = self.n;
        let bra: Vec<usize> = qubits.iter().map(|&q| q + n).collect();
        apply_dense(self.v.amplitudes_mut(), qubits, u);
        apply_dense(self.v.amplitudes_mut(), &bra, &conj_u);
        Ok(())
    }

    /// Σ_k K_k ρ K_k† for Kraus operators on `qubits` (as `apply_matrix` takes them).
    pub fn apply_kraus(&mut self, qubits: &[usize], ops: &[Vec<C>]) -> Result<(), String> {
        let mut out = DensityMatrix { n: self.n, v: StateVector::zeros(2 * self.n) };
        for k in ops {
            let mut t = self.clone();
            t.apply_matrix(qubits, k)?;
            out.add_scaled(&t, 1.0);
        }
        *self = out;
        Ok(())
    }

    /// Measures qubit `q` in the Z basis: `forced` if given (unless its probability is 0),
    /// else drawn; ρ collapses onto it.
    pub fn measure(&mut self, q: usize, forced: Option<bool>, rng: &mut Xorshift) -> bool {
        let p1 = self.prob_one(q);
        let outcome = match forced {
            Some(true) if p1 > 1e-12 => true,
            Some(false) if p1 < 1.0 - 1e-12 => false,
            Some(_) => p1 > 0.5,
            None => rng.next_f64() < p1,
        };
        self.project(q, outcome);
        self.scale(1.0 / if outcome { p1 } else { 1.0 - p1 });
        outcome
    }

    /// The probability that qubit `q` reads 1 in the Z basis.
    pub fn prob_one(&self, q: usize) -> f64 {
        (0..1usize << self.n).filter(|i| i >> q & 1 == 1).map(|i| self.get(i, i).re).sum()
    }

    /// The projection of qubit `q` onto `outcome` in the Z basis, unnormalised.
    fn project(&mut self, q: usize, outcome: bool) {
        let n = self.n;
        let mask = (1usize << n) - 1;
        for (k, a) in self.v.amplitudes_mut().iter_mut().enumerate() {
            let (i, j) = (k & mask, k >> n);
            if (i >> q & 1 == 1) != outcome || (j >> q & 1 == 1) != outcome {
                *a = C::ZERO;
            }
        }
    }

    /// Qubit `q` measured in the Z basis and the outcome forgotten.
    pub fn dephase(&mut self, q: usize) {
        let n = self.n;
        let mask = (1usize << n) - 1;
        for (k, a) in self.v.amplitudes_mut().iter_mut().enumerate() {
            if ((k & mask) >> q & 1) != ((k >> n) >> q & 1) {
                *a = C::ZERO;
            }
        }
    }

    /// Tr(ρ P) for the Pauli product `paulis`.
    pub fn expectation(&self, paulis: &[(usize, u8)]) -> C {
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
        // P|j⟩ = iy (−1)^{|j ∧ z|} |j ⊕ x⟩, so Tr(ρP) = Σ_j ⟨j|ρP|j⟩ = Σ_j ρ[j][j ⊕ x] iy (−1)^{|j ∧ z|}.
        let mut t = C::ZERO;
        for j in 0..1usize << self.n {
            let c = if (j & z).count_ones() % 2 == 1 { C::ZERO - iy } else { iy };
            t = t + self.get(j, j ^ x) * c;
        }
        t
    }

    /// The state after one step, the record bits it writes merged as a mixture, or (for a
    /// step that records) each record bit's weight and state: `Ok(None)` when it writes none.
    fn step(&mut self, step: &Step, rec: &[bool]) -> Result<Option<Vec<(bool, f64, DensityMatrix)>>, String> {
        match step {
            Step::H(q) => self.h(*q),
            Step::S(q) => self.s(*q),
            Step::Paulis(p) => self.pauli(p),
            Step::Cx(c, t) => self.cx(*c, *t),
            Step::Cz(a, b) => self.cz(*a, *b),
            Step::Rotation { paulis, theta } => self.rotate(paulis, *theta),
            Step::Unitary(q, m) => self.unitary_1q(*q, m),
            Step::Damp { q, gamma } => self.kraus_1q(*q, &amplitude_damping(*gamma)),
            Step::Dephase { q, lambda } => self.kraus_1q(*q, &phase_damping(*lambda)),
            Step::Channel(branches) => self.pauli_channel(branches),
            Step::Correlated { .. } => unreachable!("grouped by `run`"),
            Step::Feedback { pauli, rec: r, q } => {
                if rec[*r] {
                    self.pauli(&[(*q, *pauli)]);
                }
            }
            Step::Reset { q, basis } => {
                if *basis == Basis::X {
                    self.h(*q);
                }
                self.kraus_1q(*q, &RESET);
                if *basis == Basis::X {
                    self.h(*q);
                }
            }
            Step::Measure { q, basis, reset, flip } => {
                if *basis == Basis::X {
                    self.h(*q);
                }
                let mut out = Vec::new();
                let p1 = self.prob_one(*q);
                let (mut zero, mut one) = (self.clone(), self.clone());
                zero.project(*q, false);
                one.project(*q, true);
                for bit in [false, true] {
                    // The bit is the outcome, kept, or the other outcome, flipped.
                    let (wk, wf) = (if bit { p1 } else { 1.0 - p1 } * (1.0 - flip), if bit { 1.0 - p1 } else { p1 } * flip);
                    let w = wk + wf;
                    if w <= 0.0 {
                        continue;
                    }
                    let (kept, flipped) = if bit { (&one, &zero) } else { (&zero, &one) };
                    let mut s = DensityMatrix { n: self.n, v: StateVector::zeros(2 * self.n) };
                    if wk > 0.0 {
                        s.add_scaled(kept, (1.0 - flip) / w);
                    }
                    if wf > 0.0 {
                        s.add_scaled(flipped, *flip / w);
                    }
                    if *reset {
                        s.kraus_1q(*q, &RESET);
                    }
                    if *basis == Basis::X {
                        s.h(*q);
                    }
                    out.push((bit, w, s));
                }
                return Ok(Some(out));
            }
            Step::Pad { flip, value, leak } => {
                if leak.is_some() {
                    return Err("the density matrix does not run leakage".into());
                }
                let mut out = Vec::new();
                for (bit, w) in [(*value, 1.0 - flip), (!value, *flip)] {
                    if w > 0.0 {
                        out.push((bit, w, self.clone()));
                    }
                }
                return Ok(Some(out));
            }
            Step::Herald { q, probs } => {
                let fired: f64 = probs.iter().sum();
                let mut out = Vec::new();
                if fired > 0.0 {
                    let mut s = self.clone();
                    let branches: Vec<(f64, Vec<(usize, u8)>)> = (1..4).map(|k| (probs[k] / fired, vec![(*q, [0, 1, 3, 2][k])])).collect();
                    s.pauli_channel(&branches);
                    out.push((true, fired, s));
                }
                if fired < 1.0 {
                    out.push((false, 1.0 - fired, self.clone()));
                }
                return Ok(Some(out));
            }
            Step::Leak { .. } | Step::Seep { .. } | Step::Transport { .. } => return Err("the density matrix does not run leakage".into()),
        }
        Ok(None)
    }
}

/// The dense k-qubit matrix `u` applied to the amplitudes, `qubits[0]` the least significant
/// bit of its index.
fn apply_dense(amp: &mut [C], qubits: &[usize], u: &[C]) {
    let d = 1usize << qubits.len();
    let mask: usize = qubits.iter().map(|&q| 1usize << q).sum();
    let offsets: Vec<usize> = (0..d).map(|r| qubits.iter().enumerate().filter(|(b, _)| r >> b & 1 == 1).map(|(_, &q)| 1usize << q).sum()).collect();
    let mut gathered = vec![C::ZERO; d];
    for base in 0..amp.len() {
        if base & mask != 0 {
            continue;
        }
        for (r, &o) in offsets.iter().enumerate() {
            gathered[r] = amp[base | o];
        }
        for (r, &o) in offsets.iter().enumerate() {
            amp[base | o] = (0..d).fold(C::ZERO, |acc, c| acc + u[r * d + c] * gathered[c]);
        }
    }
}

/// The steps with each chain of correlated errors (`E`, then its `ELSE_CORRELATED_ERROR`s) one
/// Pauli channel: the k-th fires with probability p_k Π_{j<k} (1 − p_j). A chain must be
/// unbroken: an `ELSE_CORRELATED_ERROR` after other operations fires on whether its chain did,
/// which no channel says (the state vector, which follows each run, takes it). One before any
/// `E` fires as an `E` would, as in Stim.
pub(crate) fn grouped(steps: &[Step], who: &str) -> Result<Vec<Step>, String> {
    let mut out: Vec<Step> = Vec::new();
    let mut none = 1.0;
    // Whether the last step is a chain's channel, and whether any chain has begun.
    let (mut open, mut begun) = (false, false);
    for step in steps {
        match step {
            Step::Correlated { p, paulis, chained } => {
                if *chained && !open && begun {
                    return Err(format!("{who} takes a chain of correlated errors unbroken: an ELSE_CORRELATED_ERROR right after its E or another ELSE_CORRELATED_ERROR (the state vector takes any)"));
                }
                if !*chained || !open {
                    none = 1.0;
                    out.push(Step::Channel(Vec::new()));
                }
                let Some(Step::Channel(branches)) = out.last_mut() else { unreachable!("pushed above") };
                branches.push((p * none, paulis.clone()));
                none *= 1.0 - p;
                (open, begun) = (true, true);
            }
            other => {
                out.push(other.clone());
                open = false;
            }
        }
    }
    Ok(out)
}

/// A circuit ready for the density matrix.
pub struct DensityProgram {
    program: Program,
    steps: Vec<Step>,
}

impl DensityProgram {
    /// Qubits renumbered to the ones in use, for the outcomes' distribution.
    pub fn new(circuit: &Circuit) -> Result<DensityProgram, String> {
        DensityProgram::with_layout(circuit, false, Outcomes::Packed)
    }

    /// Qubits renumbered (`whole` false) or as the circuit numbers them; `outcomes` what the
    /// results carry (`StateOnly` for the state alone, any observables then allowed).
    pub fn with_layout(circuit: &Circuit, whole: bool, outcomes: Outcomes) -> Result<DensityProgram, String> {
        let program = Program::with_layout_for(circuit, whole, MAX_QUBITS, outcomes)?;
        let steps = grouped(&program.steps, "the density matrix")?;
        if steps.iter().any(|s| matches!(s, Step::Leak { .. } | Step::Seep { .. } | Step::Transport { .. })) {
            return Err("the density matrix does not run leakage; the leakage sampler and the state vector do".into());
        }
        Ok(DensityProgram { program, steps })
    }

    pub fn num_qubits(&self) -> usize {
        self.program.num_qubits()
    }

    /// Every outcome's exact probability: detection events and observable flips, as the
    /// state vector's `distribution`. At most `max_branches` records.
    pub fn distribution(&self, max_branches: usize) -> Result<BTreeMap<(Vec<bool>, u64), f64>, String> {
        let mut out = BTreeMap::new();
        let mut leaves = 0usize;
        let rho = DensityMatrix::new(self.num_qubits());
        self.branch(0, rho, Vec::new(), 1.0, None, &mut |rec, _, p| {
            leaves += 1;
            if leaves > max_branches {
                return Err(format!("more than {max_branches} branches"));
            }
            *out.entry(self.program.outcome(rec)).or_insert(0.0) += p;
            Ok(())
        })?;
        Ok(out)
    }

    /// The final state, starting from `rho` (|0…0⟩ if `None`), records averaged away
    /// (followed one by one only when feedback reads them, at most `max_branches`).
    pub fn final_state(&self, rho: Option<DensityMatrix>, max_branches: usize) -> Result<DensityMatrix, String> {
        let mut rho = rho.unwrap_or_else(|| DensityMatrix::new(self.num_qubits()));
        rho.grow(self.num_qubits());
        if !self.steps.iter().any(|s| matches!(s, Step::Feedback { .. })) {
            for step in &self.steps {
                if let Some(parts) = rho.step(step, &[])? {
                    let mut sum = DensityMatrix { n: rho.n, v: StateVector::zeros(2 * rho.n) };
                    for (_, w, s) in parts {
                        sum.add_scaled(&s, w);
                    }
                    rho = sum;
                }
            }
            return Ok(rho);
        }
        let mut sum = DensityMatrix { n: rho.n, v: StateVector::zeros(2 * rho.n) };
        let mut leaves = 0usize;
        // Only the records feedback reads are followed; the rest are averaged as they come.
        let mut read = vec![false; self.program.num_measurements];
        for step in &self.steps {
            if let Step::Feedback { rec, .. } = step {
                read[*rec] = true;
            }
        }
        self.branch(0, rho, Vec::new(), 1.0, Some(&read), &mut |_, s, p| {
            leaves += 1;
            if leaves > max_branches {
                return Err(format!("more than {max_branches} branches"));
            }
            sum.add_scaled(s, p);
            Ok(())
        })?;
        Ok(sum)
    }

    /// One run from `rho`, each record bit drawn: the final state and the record.
    pub fn sample(&self, mut rho: DensityMatrix, rng: &mut Xorshift) -> Result<(DensityMatrix, Vec<bool>), String> {
        rho.grow(self.num_qubits());
        let mut rec = Vec::with_capacity(self.program.num_measurements);
        for step in &self.steps {
            if let Some(parts) = rho.step(step, &rec)? {
                let u = rng.next_f64();
                let mut acc = 0.0;
                let last = parts.len() - 1;
                for (k, (bit, w, s)) in parts.into_iter().enumerate() {
                    acc += w;
                    if u < acc || k == last {
                        rec.push(bit);
                        rho = s;
                        break;
                    }
                }
            }
        }
        Ok((rho, rec))
    }

    /// Each run on from step `pc`, branching on its records: on every one, or (with `read`)
    /// only on those `read` marks, the others averaged (their bits then left unset).
    fn branch(&self, mut pc: usize, mut rho: DensityMatrix, mut rec: Vec<bool>, prob: f64, read: Option<&[bool]>, leaf: &mut dyn FnMut(&[bool], &DensityMatrix, f64) -> Result<(), String>) -> Result<(), String> {
        while pc < self.steps.len() {
            if let Some(parts) = rho.step(&self.steps[pc], &rec)? {
                let mut parts = parts;
                if parts.len() == 1 {
                    let (bit, _, s) = parts.pop().unwrap();
                    rec.push(bit);
                    rho = s;
                } else if read.is_some_and(|r| !r[rec.len()]) {
                    let mut sum = DensityMatrix { n: rho.n, v: StateVector::zeros(2 * rho.n) };
                    for (_, w, s) in parts {
                        sum.add_scaled(&s, w);
                    }
                    rec.push(false);
                    rho = sum;
                } else {
                    for (bit, w, s) in parts {
                        if w * prob < 1e-15 {
                            continue;
                        }
                        let mut r = rec.clone();
                        r.push(bit);
                        self.branch(pc + 1, s, r, prob * w, read, leaf)?;
                    }
                    return Ok(());
                }
            }
            pc += 1;
        }
        leaf(&rec, &rho, prob)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: C, b: C) -> bool {
        (a.re - b.re).abs() < 1e-12 && (a.im - b.im).abs() < 1e-12
    }

    /// ρ of a pure state vector.
    fn pure(sv: &StateVector) -> DensityMatrix {
        let a = sv.amplitudes();
        let n = sv.num_qubits();
        let entries: Vec<C> = (0..a.len()).flat_map(|i| (0..a.len()).map(move |j| (i, j))).map(|(i, j)| a[i] * a[j].conj()).collect();
        DensityMatrix::from_matrix(n, &entries).unwrap()
    }

    #[test]
    fn unitaries_agree_with_the_state_vector() {
        let mut sv = StateVector::new(3);
        let mut dm = DensityMatrix::new(3);
        let m = crate::statevec::u3(0.3, 1.1, -0.4);
        sv.h(0);
        dm.h(0);
        sv.apply_1q(1, &m);
        dm.unitary_1q(1, &m);
        sv.cx(0, 2);
        dm.cx(0, 2);
        sv.phase(1, C::I);
        dm.s(1);
        sv.rotate(&[(0, 3), (1, 1), (2, 2)], 0.37);
        dm.rotate(&[(0, 3), (1, 1), (2, 2)], 0.37);
        sv.pauli(&[(1, 3), (2, 1)]);
        dm.pauli(&[(1, 3), (2, 1)]);
        sv.cz(1, 2);
        dm.cz(1, 2);
        let want = pure(&sv);
        for (a, b) in dm.matrix().iter().zip(want.matrix()) {
            assert!(close(*a, b));
        }
        assert!((dm.purity() - 1.0).abs() < 1e-12 && (dm.trace() - 1.0).abs() < 1e-12);
        // Tr(ρP) = ⟨ψ|P|ψ⟩, Y's sign included (|+i⟩ has ⟨Y⟩ = +1).
        for p in [vec![(0, 3), (2, 2)], vec![(1, 3)], vec![(0, 1), (1, 3)], vec![(2, 3)]] {
            let mut t = sv.clone();
            t.pauli(&p);
            let e = sv.amplitudes().iter().zip(t.amplitudes()).fold(C::ZERO, |acc, (a, b)| acc + a.conj() * *b);
            assert!(close(dm.expectation(&p), e), "{p:?}");
        }
        let mut plus_i = DensityMatrix::new(1);
        plus_i.h(0);
        plus_i.s(0);
        assert!(close(plus_i.expectation(&[(0, 3)]), C::ONE));
    }

    #[test]
    fn dense_matrices_agree_with_gates() {
        let mut a = DensityMatrix::new(3);
        let mut b = DensityMatrix::new(3);
        a.h(0);
        b.h(0);
        a.cx(0, 2);
        // CX with control qubit 0 and target 2, as a matrix on [2, 0]: index bit 0 is qubit 2,
        // bit 1 qubit 0, so |10⟩ and |11⟩ (indices 2, 3) swap.
        let o = C::ONE;
        let z = C::ZERO;
        let cx = [o, z, z, z, z, o, z, z, z, z, z, o, z, z, o, z];
        b.apply_matrix(&[2, 0], &cx).unwrap();
        for (x, y) in a.matrix().iter().zip(b.matrix()) {
            assert!(close(*x, y));
        }
        let mut rng = Xorshift::new(1);
        assert!(b.measure(2, Some(true), &mut rng));
        assert!((b.get(5, 5).re - 1.0).abs() < 1e-12);
    }

    #[test]
    fn damping_decays_and_dephases() {
        let mut dm = DensityMatrix::new(1);
        dm.h(0);
        dm.kraus_1q(0, &amplitude_damping(0.2));
        dm.kraus_1q(0, &phase_damping(0.3));
        // ρ11 = 0.5 (1 − γ), ρ01 = 0.5 √(1 − γ) √(1 − λ).
        assert!((dm.get(1, 1).re - 0.4).abs() < 1e-12);
        assert!((dm.get(0, 1).re - 0.5 * (0.8f64 * 0.7).sqrt()).abs() < 1e-12);
    }

    #[test]
    fn distribution_equals_the_state_vector() {
        let text = "R 0 1 2\nH 0\nI_ERROR[AMPLITUDE_DAMPING(gamma=0.1)] 0\nCX 0 1\nDEPOLARIZE2(0.05) 0 1\nI_ERROR[PHASE_DAMPING(lambda=0.2)] 1\nE(0.1) X2\nELSE_CORRELATED_ERROR(0.2) Z0\nM(0.02) 0 1\nCX rec[-1] 2\nMX 2\nDETECTOR rec[-2] rec[-3]\nOBSERVABLE_INCLUDE(0) rec[-1]";
        let c = Circuit::parse(text).unwrap();
        let a = DensityProgram::new(&c).unwrap().distribution(1 << 20).unwrap();
        let b = Program::new(&c).unwrap().distribution(1 << 20).unwrap();
        assert_eq!(a.keys().collect::<Vec<_>>(), b.keys().collect::<Vec<_>>());
        for (k, p) in &a {
            assert!((p - b[k]).abs() < 1e-12, "{k:?}: {p} vs {}", b[k]);
        }
    }

    #[test]
    fn final_state_with_and_without_feedback() {
        // Measure-and-correct leaves |0⟩: the feedback branches; without it the qubit is mixed.
        let c = Circuit::parse("H 0\nM 0\nCX rec[-1] 0").unwrap();
        let rho = DensityProgram::with_layout(&c, true, Outcomes::StateOnly).unwrap().final_state(None, 1 << 10).unwrap();
        assert!((rho.get(0, 0).re - 1.0).abs() < 1e-12);
        let c = Circuit::parse("H 0\nM 0").unwrap();
        let rho = DensityProgram::with_layout(&c, true, Outcomes::StateOnly).unwrap().final_state(None, 1 << 10).unwrap();
        assert!((rho.get(0, 0).re - 0.5).abs() < 1e-12 && rho.get(0, 1).norm2() < 1e-24);
    }

    #[test]
    fn only_records_feedback_reads_are_followed() {
        // 30 random records, one read: two branches, not 2^30.
        let text = format!("{}H 1\nM 1\nCX rec[-1] 0", "H 1\nM 1\n".repeat(30));
        let c = Circuit::parse(&text).unwrap();
        let rho = DensityProgram::with_layout(&c, true, Outcomes::StateOnly).unwrap().final_state(None, 4).unwrap();
        // Qubit 0 corrected to the last outcome, which qubit 1 holds: |00⟩ and |11⟩, half each.
        assert!((rho.get(0, 0).re - 0.5).abs() < 1e-12 && (rho.get(3, 3).re - 0.5).abs() < 1e-12);
    }

    #[test]
    fn chains_of_correlated_errors_are_unbroken() {
        // An ELSE before any E fires as an E would, as in Stim.
        let c = Circuit::parse("ELSE_CORRELATED_ERROR(0.25) X0\nM 0").unwrap();
        let a = DensityProgram::new(&c).unwrap().distribution(16).unwrap();
        let b = Program::new(&c).unwrap().distribution(16).unwrap();
        assert_eq!(a.len(), b.len());
        // One after other operations (here a channel, which it must not join) is refused.
        for between in ["X_ERROR(0.1) 1", "H 0", "M 1"] {
            let c = Circuit::parse(&format!("E(0.5) X0\n{between}\nELSE_CORRELATED_ERROR(0.5) X1\nM 0 1")).unwrap();
            assert!(DensityProgram::new(&c).err().unwrap().contains("unbroken"), "{between}");
        }
        // Within one chain, the state vector's numbers.
        let c = Circuit::parse("E(0.5) X0\nELSE_CORRELATED_ERROR(0.5) X1\nELSE_CORRELATED_ERROR(0.2) X0 X1\nM 0 1\nDETECTOR rec[-1]\nDETECTOR rec[-2]").unwrap();
        let a = DensityProgram::new(&c).unwrap().distribution(16).unwrap();
        let b = Program::new(&c).unwrap().distribution(16).unwrap();
        for (k, p) in &a {
            assert!((p - b[k]).abs() < 1e-12);
        }
    }
}

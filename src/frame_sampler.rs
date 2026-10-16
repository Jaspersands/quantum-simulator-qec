//! Sampling a circuit with a Pauli frame, independently of any error model.
//!
//! The frame records how this shot differs from a noiseless run. For a Clifford
//! circuit under Pauli noise that is all a detector needs: it reads a parity the
//! noiseless circuit fixes, so it fires exactly when the frame flips an odd
//! number of the records it reads. Noise is sampled with each channel's true
//! (disjoint) semantics, not the error model's independent approximation, and
//! nothing here looks at the model, so a model that is wrong about the circuit
//! shows up as disagreement between the two rather than agreeing with itself.
//!
//! Resets and measurements randomise the component of the frame that the state
//! is insensitive to, as Stim's frame simulator does. It costs a draw and changes
//! nothing for a deterministic detector; for a non-deterministic one it makes the
//! failure visible as a coin flip instead of hiding it.

use crate::circuit::{Basis, Circuit, Instr};
use crate::nonpauli::NonPauli;
use crate::surface_code::Xorshift;

pub struct Shot {
    pub detectors: Vec<bool>,
    pub observables: u64,
    /// For a circuit with leakage, per measurement record, whether a measurement found its
    /// qubit leaked (false for other records); empty otherwise.
    pub heralds: Vec<bool>,
}

/// How a qubit is leaked: not, by `LEAK` (measuring it reads 1, or a coin), or by Deltakit's
/// `LEAKAGE` (it was depolarized, and measuring it reads its state).
const NOT_LEAKED: u8 = 0;
const LEAKED: u8 = 1;
const DELTAKIT_LEAKED: u8 = 2;

pub struct FrameSampler {
    /// Whether the circuit has leakage (`I_ERROR[LEAK(p=…)]` and the rest): a leaked flag per
    /// qubit, set by LEAK and LEAK_TRANSPORT, cleared by SEEP and by a reset. A two-qubit gate
    /// with a leaked partner leaves the other qubit a uniformly random Pauli (Google's model),
    /// and measuring a leaked qubit reads 1 (or a coin flip, `leaked_reads_one = false`).
    /// Deltakit's model is its own: `LEAKAGE` leaks a qubit and depolarizes it, after which it
    /// is measured as it stands; `RELAX` returns it as it is; a reset returns it too, and a
    /// leaked partner depolarizes as above. A herald's record is 1 while its qubit is leaked.
    leaky: bool,
    pub leaked_reads_one: bool,
    /// The noiseless reference record, which a leaked qubit's 1 is a flip from.
    reference: Vec<bool>,
    instrs: Vec<Instr>,
    num_qubits: usize,
    num_measurements: usize,
    detectors: Vec<Vec<usize>>,
    observables: Vec<Vec<usize>>,
}

/// After a two-qubit gate, a leaked partner leaves the other qubit a uniformly random Pauli.
fn leaked_partner(leaked: &[u8], a: usize, b: usize, x: &mut [bool], z: &mut [bool], rng: &mut Xorshift) {
    for (l, o) in [(a, b), (b, a)] {
        if leaked[l] != NOT_LEAKED && leaked[o] == NOT_LEAKED {
            apply(x, z, o, [0, 1, 3, 2][(rng.next_u64() % 4) as usize]);
        }
    }
}

fn apply(x: &mut [bool], z: &mut [bool], q: usize, pauli: u8) {
    if pauli & 1 != 0 {
        x[q] ^= true;
    }
    if pauli & 2 != 0 {
        z[q] ^= true;
    }
}

impl FrameSampler {
    pub fn new(circuit: &Circuit) -> Result<FrameSampler, String> {
        circuit.check_runnable()?;
        let res = circuit.resolve()?;
        if res.observables.len() > 64 {
            return Err(format!("{} observables: the per-shot sampler takes at most 64 (the detector sampler takes any number)", res.observables.len()));
        }
        let leaky = res.instrs.iter().any(|i| matches!(i, Instr::NonPauli(ops) if ops.iter().any(|o| o.is_leakage())));
        let reference = if leaky { crate::m2d::run(circuit, crate::batch_sampler::Counts::of(&circuit.instrs)?.qubits, &[], 1) } else { Vec::new() };
        Ok(FrameSampler {
            leaky,
            leaked_reads_one: true,
            reference,
            instrs: res.instrs,
            num_qubits: res.num_qubits,
            num_measurements: res.num_measurements,
            detectors: res.detectors,
            observables: res.observables,
        })
    }

    pub fn num_detectors(&self) -> usize {
        self.detectors.len()
    }

    pub fn num_observables(&self) -> usize {
        self.observables.len()
    }

    pub fn num_measurements(&self) -> usize {
        self.num_measurements
    }

    pub fn sample(&self, rng: &mut Xorshift) -> Shot {
        let (rec, pauli_obs, heralds) = self.run(rng);
        let parity = |recs: &[usize]| recs.iter().fold(false, |acc, &m| acc ^ rec[m]);
        let detectors = self.detectors.iter().map(|r| parity(r)).collect();
        let mut observables = pauli_obs;
        for (k, r) in self.observables.iter().enumerate() {
            if parity(r) {
                observables |= 1u64 << k;
            }
        }
        Shot { detectors, observables, heralds }
    }

    /// The same sampler, ready for `sample_measurements` (a leaky one already is).
    pub fn with_reference(mut self, circuit: &Circuit) -> Result<FrameSampler, String> {
        if self.reference.len() != self.num_measurements {
            self.reference = crate::m2d::run(circuit, crate::batch_sampler::Counts::of(&circuit.instrs)?.qubits, &[], 1);
        }
        Ok(self)
    }

    /// One shot's measurement record, and its heralds as `sample`'s.
    pub fn sample_measurements(&self, rng: &mut Xorshift) -> (Vec<bool>, Vec<bool>) {
        let (mut rec, _, heralds) = self.run(rng);
        for (r, &reference) in rec.iter_mut().zip(&self.reference) {
            *r ^= reference;
        }
        (rec, heralds)
    }

    /// One shot: its record as flips from the reference, its Pauli targets' observable flips,
    /// and its heralds.
    fn run(&self, rng: &mut Xorshift) -> (Vec<bool>, u64, Vec<bool>) {
        let nq = self.num_qubits;
        let mut x = vec![false; nq];
        let mut z: Vec<bool> = (0..nq).map(|_| rng.next_u64() & 1 == 1).collect();
        let mut rec = Vec::with_capacity(self.num_measurements);
        // Whether the current chain of correlated errors has fired.
        let mut chain = false;
        let mut pauli_obs = 0u64;
        let mut leaked = vec![NOT_LEAKED; if self.leaky { nq } else { 0 }];
        let mut heralds = Vec::new();
        // A herald's readings, for the pads that follow its markers.
        let mut pending = std::collections::VecDeque::new();
        for ins in &self.instrs {
            match ins {
                Instr::Fail(_) => {}
                Instr::Reset { basis, qubits } => {
                    for &q in qubits {
                        let q = q as usize;
                        if self.leaky {
                            leaked[q] = NOT_LEAKED;
                        }
                        match basis {
                            Basis::Z => {
                                x[q] = false;
                                z[q] = rng.next_u64() & 1 == 1;
                            }
                            Basis::X => {
                                z[q] = false;
                                x[q] = rng.next_u64() & 1 == 1;
                            }
                        }
                    }
                }
                Instr::H(qubits) => {
                    for &q in qubits {
                        let q = q as usize;
                        std::mem::swap(&mut x[q], &mut z[q]);
                    }
                }
                Instr::Cx(pairs) => {
                    for &(c, t) in pairs {
                        let (c, t) = (c as usize, t as usize);
                        if x[c] {
                            x[t] ^= true;
                        }
                        if z[t] {
                            z[c] ^= true;
                        }
                        if self.leaky {
                            leaked_partner(&leaked, c, t, &mut x, &mut z, rng);
                        }
                    }
                }
                Instr::Cz(pairs) => {
                    for &(a, b) in pairs {
                        let (a, b) = (a as usize, b as usize);
                        if x[a] {
                            z[b] ^= true;
                        }
                        if x[b] {
                            z[a] ^= true;
                        }
                        if self.leaky {
                            leaked_partner(&leaked, a, b, &mut x, &mut z, rng);
                        }
                    }
                }
                Instr::Measure { basis, reset, flip, qubits } => {
                    for &q in qubits {
                        let q = q as usize;
                        let mut r = match basis {
                            Basis::Z => x[q],
                            Basis::X => z[q],
                        };
                        if self.leaky {
                            heralds.push(leaked[q] != NOT_LEAKED);
                            if leaked[q] == LEAKED {
                                // A leaked qubit reads 1: a flip from the reference's value.
                                r = if self.leaked_reads_one { !self.reference[rec.len()] } else { rng.next_u64() & 1 == 1 };
                            }
                            if *reset {
                                leaked[q] = NOT_LEAKED;
                            }
                        }
                        if *flip > 0.0 && rng.next_f64() < *flip {
                            r = !r;
                        }
                        rec.push(r);
                        match basis {
                            Basis::Z => z[q] = rng.next_u64() & 1 == 1,
                            Basis::X => x[q] = rng.next_u64() & 1 == 1,
                        }
                        if *reset {
                            match basis {
                                Basis::Z => {
                                    x[q] = false;
                                    z[q] = rng.next_u64() & 1 == 1;
                                }
                                Basis::X => {
                                    z[q] = false;
                                    x[q] = rng.next_u64() & 1 == 1;
                                }
                            }
                        }
                    }
                }
                Instr::PauliError { pauli, p, qubits } => {
                    for &q in qubits {
                        if rng.next_f64() < *p {
                            apply(&mut x, &mut z, q as usize, *pauli);
                        }
                    }
                }
                Instr::Depolarize1 { p, qubits } => {
                    for &q in qubits {
                        if rng.next_f64() < *p {
                            apply(&mut x, &mut z, q as usize, 1 + (rng.next_u64() % 3) as u8);
                        }
                    }
                }
                Instr::PauliChannel1 { px, py, pz, qubits } => {
                    for &q in qubits {
                        let u = rng.next_f64();
                        let pauli = if u < *px {
                            1
                        } else if u < px + py {
                            3
                        } else if u < px + py + pz {
                            2
                        } else {
                            0
                        };
                        apply(&mut x, &mut z, q as usize, pauli);
                    }
                }
                Instr::Depolarize2 { p, pairs } => {
                    for &(a, b) in pairs {
                        if rng.next_f64() < *p {
                            let r = 1 + rng.next_u64() % 15;
                            apply(&mut x, &mut z, a as usize, (r & 3) as u8);
                            apply(&mut x, &mut z, b as usize, (r >> 2) as u8);
                        }
                    }
                }
                Instr::S(qubits) => {
                    for &q in qubits {
                        let q = q as usize;
                        z[q] ^= x[q];
                    }
                }
                Instr::Correlated { p, paulis, chained } => {
                    let fires = rng.next_f64() < *p && !(*chained && chain);
                    chain = if *chained { chain || fires } else { fires };
                    if fires {
                        for &(q, pauli) in paulis {
                            apply(&mut x, &mut z, q as usize, pauli);
                        }
                    }
                }
                Instr::PauliChannel2 { probs, pairs } => {
                    for &(a, b) in pairs {
                        let u = rng.next_f64();
                        let mut acc = 0.0;
                        if let Some(k) = probs.iter().position(|p| {
                            acc += p;
                            u < acc
                        }) {
                            const CODE: [u8; 4] = [0, 1, 3, 2];
                            apply(&mut x, &mut z, a as usize, CODE[(k + 1) / 4]);
                            apply(&mut x, &mut z, b as usize, CODE[(k + 1) % 4]);
                        }
                    }
                }
                // Relative to the noiseless run, a record's flip flips its Pauli; a sweep bit
                // moves only the reference.
                Instr::Feedback { pauli, control, qubit } => {
                    if let crate::circuit::Control::Rec(k) = control {
                        if rec[rec.len() - *k as usize] {
                            apply(&mut x, &mut z, *qubit as usize, *pauli);
                        }
                    }
                }
                Instr::Heralded { probs, qubits, .. } => {
                    for &q in qubits {
                        let u = rng.next_f64();
                        let mut acc = 0.0;
                        let case = probs.iter().position(|p| {
                            acc += p;
                            u < acc
                        });
                        rec.push(case.is_some());
                        if self.leaky {
                            heralds.push(false);
                        }
                        if let Some(k) = case {
                            apply(&mut x, &mut z, q as usize, [0, 1, 3, 2][k]);
                        }
                    }
                }
                Instr::Pad { flip, values } => {
                    for _ in values {
                        // A herald's pad reads its qubit's leakage, a flip from the pad's value.
                        let leaked_now = pending.pop_front().unwrap_or(false);
                        rec.push(leaked_now ^ (*flip > 0.0 && rng.next_f64() < *flip));
                        if self.leaky {
                            heralds.push(false);
                        }
                    }
                }
                // Signs, annotations and ticks leave a frame alone; loops and gates are
                // flattened away.
                // A Pauli target reads the frame: X anticommutes with its Z part, Z with its X.
                Instr::Observable { index, paulis, .. } => {
                    for &(q, p, _) in paulis {
                        let q = q as usize;
                        if (p & 1 != 0 && z[q]) != (p & 2 != 0 && x[q]) {
                            pauli_obs ^= 1u64 << index;
                        }
                    }
                }
                Instr::NonPauli(ops) if self.leaky => {
                    for op in ops {
                        match *op {
                            NonPauli::Leak { p, qubit } => {
                                if rng.next_f64() < p && leaked[qubit as usize] == NOT_LEAKED {
                                    leaked[qubit as usize] = LEAKED;
                                }
                            }
                            NonPauli::Leakage { p, qubit } => {
                                let q = qubit as usize;
                                if rng.next_f64() < p {
                                    // Deltakit's Pauli mixture: I, X, Y or Z, a quarter each.
                                    leaked[q] = DELTAKIT_LEAKED;
                                    x[q] ^= rng.next_u64() & 1 == 1;
                                    z[q] ^= rng.next_u64() & 1 == 1;
                                }
                            }
                            NonPauli::Relax { p, qubit } => {
                                let q = qubit as usize;
                                if rng.next_f64() < p && leaked[q] != NOT_LEAKED {
                                    // A qubit LEAK took returns at random; LEAKAGE's already is.
                                    if leaked[q] == LEAKED {
                                        x[q] ^= rng.next_u64() & 1 == 1;
                                        z[q] = rng.next_u64() & 1 == 1;
                                    }
                                    leaked[q] = NOT_LEAKED;
                                }
                            }
                            NonPauli::Herald { qubit } => pending.push_back(leaked[qubit as usize] != NOT_LEAKED),
                            NonPauli::Seep { p, qubit } => {
                                let q = qubit as usize;
                                if rng.next_f64() < p && leaked[q] != NOT_LEAKED {
                                    leaked[q] = NOT_LEAKED;
                                    // Back in |0⟩ or |1⟩ at random.
                                    x[q] ^= rng.next_u64() & 1 == 1;
                                    z[q] = rng.next_u64() & 1 == 1;
                                }
                            }
                            NonPauli::LeakTransport { p, a, b } => {
                                let (a, b) = (a as usize, b as usize);
                                if rng.next_f64() < p && (leaked[a] == NOT_LEAKED) != (leaked[b] == NOT_LEAKED) {
                                    let back = if leaked[a] != NOT_LEAKED { a } else { b };
                                    leaked.swap(a, b);
                                    x[back] ^= rng.next_u64() & 1 == 1;
                                    z[back] = rng.next_u64() & 1 == 1;
                                }
                            }
                            _ => {}
                        }
                    }
                }
                Instr::Pauli { .. }
                | Instr::SweepX(_)
                | Instr::Detector { .. }
                | Instr::QubitCoords { .. }
                | Instr::ShiftCoords(_)
                | Instr::Tick
                | Instr::NonPauli(_)
                | Instr::Repeat { .. }
                | Instr::Gate { .. } => {}
            }
        }
        (rec, pauli_obs, heralds)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dem::Dem;
    use crate::fixtures::REP3;

    /// Marginals follow from the model exactly: a detector fires when an odd
    /// number of its independent mechanisms do, (1 − Π(1 − 2p))/2. The sampler
    /// never reads the model, so agreement is a check on both.
    #[test]
    fn detector_marginals_match_the_error_model() {
        let circuit = Circuit::parse(&REP3.replace("0.01", "0.05")).unwrap();
        let dem = Dem::from_circuit(&circuit).unwrap();
        let sampler = FrameSampler::new(&circuit).unwrap();
        let n = 200_000usize;
        let mut counts = vec![0usize; sampler.num_detectors()];
        let mut obs = 0usize;
        let mut rng = Xorshift::new(11);
        for _ in 0..n {
            let shot = sampler.sample(&mut rng);
            for (i, &b) in shot.detectors.iter().enumerate() {
                counts[i] += b as usize;
            }
            obs += (shot.observables & 1) as usize;
        }
        let predict = |pick: &dyn Fn(&crate::dem::Mechanism) -> bool| -> f64 {
            let prod: f64 = dem.mechanisms.iter().filter(|m| pick(m)).map(|m| 1.0 - 2.0 * m.p).product();
            (1.0 - prod) / 2.0
        };
        for (d, &c) in counts.iter().enumerate() {
            let q = predict(&|m| m.detectors.contains(&(d as u32)));
            let sigma = (q * (1.0 - q) / n as f64).sqrt();
            let rate = c as f64 / n as f64;
            assert!((rate - q).abs() < 5.0 * sigma, "D{d}: sampled {rate}, model {q}");
        }
        let q = predict(&|m| m.observables & 1 == 1);
        let rate = obs as f64 / n as f64;
        assert!((rate - q).abs() < 5.0 * (q * (1.0 - q) / n as f64).sqrt(), "L0: sampled {rate}, model {q}");
    }

    /// Deltakit's model: LEAKAGE(1) leaks qubit 0 every shot and a herald reads it; a CX with
    /// it depolarizes qubit 1, whose measurement is a coin; RELAX(1) returns it, and an inverted
    /// herald then reads its pad's value. A record per measurement and pad in `heralds`.
    #[test]
    fn deltakit_leakage_and_heralds() {
        let c = Circuit::parse("R 0 1\nLEAKAGE(1) 0\nHERALD_LEAKAGE_EVENT 0 1\nDETECTOR rec[-2]\nDETECTOR rec[-1]\nCX 0 1\nM 1\nDETECTOR rec[-1]\nRELAX(1) 0\nHERALD_LEAKAGE_EVENT !0\nDETECTOR rec[-1]").unwrap();
        let sampler = FrameSampler::new(&c).unwrap();
        let mut rng = Xorshift::new(5);
        let mut coin = 0;
        for _ in 0..2000 {
            let shot = sampler.sample(&mut rng);
            assert_eq!((shot.detectors[0], shot.detectors[1], shot.detectors[3]), (true, false, false));
            assert_eq!(shot.heralds, vec![false, false, false, false]);
            coin += shot.detectors[2] as usize;
        }
        assert!((850..1150).contains(&coin), "{coin}");
        // A reset returns a leaked qubit; a measurement reads a LEAKAGE-leaked qubit's state,
        // where LEAK's reads 1.
        let c = Circuit::parse("R 0\nLEAKAGE(1) 0\nR 0\nHERALD_LEAKAGE_EVENT 0\nDETECTOR rec[-1]\nLEAKAGE(1) 0\nM 0\nDETECTOR rec[-1]").unwrap();
        let sampler = FrameSampler::new(&c).unwrap();
        let fired: usize = (0..2000).map(|_| sampler.sample(&mut rng)).inspect(|s| assert!(!s.detectors[0] && s.heralds == vec![false, true])).map(|s| s.detectors[1] as usize).sum();
        assert!((850..1150).contains(&fired), "{fired}");
    }

    #[test]
    fn noiseless_circuits_never_fire() {
        let circuit = Circuit::parse(&REP3.replace("0.01", "0")).unwrap();
        let sampler = FrameSampler::new(&circuit).unwrap();
        let mut rng = Xorshift::new(3);
        for _ in 0..2000 {
            let shot = sampler.sample(&mut rng);
            assert!(shot.detectors.iter().all(|&b| !b));
            assert_eq!(shot.observables, 0);
        }
    }
}

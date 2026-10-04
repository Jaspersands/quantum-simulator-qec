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
use crate::surface_code::Xorshift;

pub struct Shot {
    pub detectors: Vec<bool>,
    pub observables: u64,
}

pub struct FrameSampler {
    instrs: Vec<Instr>,
    num_qubits: usize,
    num_measurements: usize,
    detectors: Vec<Vec<usize>>,
    observables: Vec<Vec<usize>>,
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
        let res = circuit.resolve()?;
        Ok(FrameSampler {
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

    pub fn sample(&self, rng: &mut Xorshift) -> Shot {
        let nq = self.num_qubits;
        let mut x = vec![false; nq];
        let mut z: Vec<bool> = (0..nq).map(|_| rng.next_u64() & 1 == 1).collect();
        let mut rec = Vec::with_capacity(self.num_measurements);
        // Whether the current chain of correlated errors has fired.
        let mut chain = false;
        let mut pauli_obs = 0u64;
        for ins in &self.instrs {
            match ins {
                Instr::Reset { basis, qubits } => {
                    for &q in qubits {
                        let q = q as usize;
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
                    }
                }
                Instr::Measure { basis, reset, flip, qubits } => {
                    for &q in qubits {
                        let q = q as usize;
                        let mut r = match basis {
                            Basis::Z => x[q],
                            Basis::X => z[q],
                        };
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
                        if let Some(k) = case {
                            apply(&mut x, &mut z, q as usize, [0, 1, 3, 2][k]);
                        }
                    }
                }
                Instr::Pad { flip, values } => {
                    for _ in values {
                        rec.push(*flip > 0.0 && rng.next_f64() < *flip);
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
                Instr::Pauli { .. }
                | Instr::SweepX(_)
                | Instr::Detector { .. }
                | Instr::QubitCoords { .. }
                | Instr::ShiftCoords(_)
                | Instr::Tick
                | Instr::Repeat { .. }
                | Instr::Gate { .. } => {}
            }
        }
        let parity = |recs: &[usize]| recs.iter().fold(false, |acc, &m| acc ^ rec[m]);
        let detectors = self.detectors.iter().map(|r| parity(r)).collect();
        let mut observables = pauli_obs;
        for (k, r) in self.observables.iter().enumerate() {
            if parity(r) {
                observables |= 1u64 << k;
            }
        }
        Shot { detectors, observables }
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

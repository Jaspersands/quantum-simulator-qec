//! The coherent sampler's identity, checked exhaustively: every random choice a shot can make
//! (the frame's randomisation, every noise branch, every location firing or not), each with its
//! twirl probability, weighted by the whole spacetime kernel. The weighted outcomes must be the
//! state vector's to rounding, for any small circuit. This is where the signs are proved.

use std::collections::BTreeMap;

use super::shot::Source;
use super::{Element, Program, Slot};

/// Every branch of a shot, by replaying it with its choices counted like an odometer.
struct Odometer {
    choices: Vec<usize>,
    probs: Vec<Vec<f64>>,
    pos: usize,
    p: f64,
}

impl Odometer {
    fn choose(&mut self, probs: Vec<f64>) -> usize {
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
    fn advance(&mut self) -> bool {
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
    fn bern(&mut self, p: f64) -> bool {
        self.choose(vec![1.0 - p, p]) == 1
    }

    fn pick(&mut self, probs: &[f64]) -> Option<usize> {
        let mut v = probs.to_vec();
        v.push(1.0 - probs.iter().sum::<f64>());
        let k = self.choose(v);
        (k < probs.len()).then_some(k)
    }

    fn index(&mut self, p: f64, n: u64) -> Option<u64> {
        let mut v = vec![1.0 - p];
        v.extend((0..n).map(|_| p / n as f64));
        let k = self.choose(v);
        (k > 0).then(|| k as u64 - 1)
    }

    fn coin(&mut self) -> bool {
        self.choose(vec![0.5, 0.5]) == 1
    }
}

impl Program {
    /// The whole spacetime kernel, by trying every set of locations (a few dozen locations at
    /// most).
    pub fn full_kernel(&self) -> Result<Vec<Element>, String> {
        let n = self.locations.len();
        if n > 20 {
            return Err(format!("{n} locations: the whole kernel is for small circuits"));
        }
        let solver = super::GaugeSolver::new(self);
        let mut out = Vec::new();
        for mask in 1u32..(1 << n) {
            let members: Vec<usize> = (0..n).filter(|&g| mask >> g & 1 == 1).collect();
            out.extend(self.element(&members, &solver));
        }
        Ok(out)
    }

    /// Every outcome's probability by the weighted twirl, summed over every branch. With `raw`,
    /// detectors and observables are the record's parities themselves rather than compared with
    /// the reference run's.
    pub fn exact_distribution(&self, kernel: &[Element], raw: bool, max_branches: usize) -> Result<BTreeMap<(Vec<bool>, u64), f64>, String> {
        let reference: Vec<bool> = self.slots.iter().zip(&self.reference).filter(|(s, _)| matches!(s, Slot::Record(_))).map(|(_, &v)| v).collect();
        let mut out = BTreeMap::new();
        let mut odo = Odometer { choices: Vec::new(), probs: Vec::new(), pos: 0, p: 1.0 };
        let mut branches = 0usize;
        loop {
            odo.pos = 0;
            odo.p = 1.0;
            let shot = self.run_shot(&mut odo);
            branches += 1;
            if branches > max_branches {
                return Err(format!("more than {max_branches} branches"));
            }
            let rec: Vec<bool> = if raw { shot.records.iter().zip(&reference).map(|(a, b)| a ^ b).collect() } else { shot.records.clone() };
            let w = self.weight(&shot, kernel);
            *out.entry(self.outcome(&rec)).or_insert(0.0) += odo.p * w;
            if !odo.advance() {
                break;
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::circuit::Circuit;

    /// The state vector's distribution of raw parities.
    fn truth(c: &Circuit) -> BTreeMap<(Vec<bool>, u64), f64> {
        let sv = crate::statevec::Program::new(c).unwrap();
        let reference = sv.reference_parities().to_vec();
        let nd = sv.num_detectors();
        let mut out = BTreeMap::new();
        for ((dets, obs), p) in sv.distribution(1 << 22).unwrap() {
            let d: Vec<bool> = dets.iter().zip(&reference).map(|(a, b)| a ^ b).collect();
            let o = (0..sv.num_observables()).fold(0u64, |acc, k| acc | (((obs >> k & 1 == 1) ^ reference[nd + k]) as u64) << k);
            *out.entry((d, o)).or_insert(0.0) += p;
        }
        out
    }

    fn check(text: &str) {
        let c = Circuit::parse(text).unwrap();
        let p = Program::new(&c).unwrap();
        let kernel = p.full_kernel().unwrap();
        let ours = p.exact_distribution(&kernel, true, 1 << 22).unwrap();
        let theirs = truth(&c);
        let keys: std::collections::BTreeSet<_> = ours.keys().chain(theirs.keys()).cloned().collect();
        for k in keys {
            let (a, b) = (ours.get(&k).copied().unwrap_or(0.0), theirs.get(&k).copied().unwrap_or(0.0));
            assert!((a - b).abs() < 1e-10, "{text}\n{k:?}: weighted twirl {a}, state vector {b}");
        }
    }

    #[test]
    fn one_rotation() {
        check("R 0\nI_ERROR[R_X(theta=0.3)] 0\nM 0\nDETECTOR rec[-1]");
    }

    #[test]
    fn rotations_that_add() {
        check("R 0\nI_ERROR[R_X(theta=0.3)] 0\nI_ERROR[R_X(theta=0.2)] 0\nM 0\nDETECTOR rec[-1]");
        check("R 0\nI_ERROR[R_X(theta=0.3)] 0\nI_ERROR[R_X(theta=-0.2)] 0\nM 0\nDETECTOR rec[-1]");
    }

    #[test]
    fn rotations_through_gates() {
        check("R 0\nH 0\nI_ERROR[R_Z(theta=0.3)] 0\nH 0\nI_ERROR[R_X(theta=0.2)] 0\nM 0\nDETECTOR rec[-1]");
        check("R 0 1\nI_ERROR[R_Y(theta=0.3)] 0\nCX 0 1\nI_ERROR[R_X(theta=0.25)] 1\nS 1\nI_ERROR[R_Y(theta=0.1)] 1\nM 0 1\nDETECTOR rec[-1]\nDETECTOR rec[-2]");
        check("R 0 1\nH 0 1\nII_ERROR[R_ZZ(theta=0.4)] 0 1\nCZ 0 1\nI_ERROR[R_Z(theta=0.2)] 1\nH 0 1\nM 0 1\nDETECTOR rec[-1]\nDETECTOR rec[-2]");
    }

    #[test]
    fn rotations_with_noise_and_mid_circuit_measurement() {
        check("R 0 1\nH 0\nI_ERROR[R_X(theta=0.3)] 1\nCX 0 1\nDEPOLARIZE1(0.1) 1\nM(0.05) 1\nI_ERROR[R_X(theta=0.2)] 1\nM 1\nDETECTOR rec[-1] rec[-2]\nOBSERVABLE_INCLUDE(0) rec[-1]");
        check("R 0\nI_ERROR[R_X(theta=0.3)] 0\nMR 0\nI_ERROR[R_X(theta=0.2)] 0\nX_ERROR(0.1) 0\nM 0\nDETECTOR rec[-1]\nDETECTOR rec[-2]");
        check("RX 0\nI_ERROR[R_Z(theta=0.3)] 0\nMX 0\nRX 0\nI_ERROR[R_Z(theta=0.2)] 0\nMX 0\nDETECTOR rec[-1]\nDETECTOR rec[-2]");
    }

    #[test]
    fn repetition_code_rounds() {
        check(
            "R 0 1 2\nI_ERROR[R_X(theta=0.2)] 0 2\nCX 0 1\nCX 2 1\nMR 1\nI_ERROR[R_X(theta=0.15)] 0 2\nCX 0 1\nCX 2 1\nMR 1\n\
             DETECTOR rec[-2]\nDETECTOR rec[-1] rec[-2]\nM 0 2\nDETECTOR rec[-1] rec[-2] rec[-3]\nOBSERVABLE_INCLUDE(0) rec[-1]",
        );
        // The same in X: rotations about Z on |+⟩ data, X-type checks.
        check(
            "RX 0 2\nR 1\nI_ERROR[R_Z(theta=0.2)] 0 2\nH 1\nCZ 1 0\nCZ 1 2\nH 1\nMR 1\nI_ERROR[R_Z(theta=-0.3)] 0 2\nH 1\nCZ 1 0\nCZ 1 2\nH 1\nMR 1\n\
             DETECTOR rec[-2]\nDETECTOR rec[-1] rec[-2]\nMX 0 2\nDETECTOR rec[-1] rec[-2] rec[-3]\nOBSERVABLE_INCLUDE(0) rec[-1]",
        );
    }

    /// Random small circuits of every kind of step.
    #[test]
    fn random_circuits() {
        let mut rng = crate::surface_code::Xorshift::new(2026);
        let ops1 = ["H", "S", "X", "Y", "Z", "S_DAG", "SQRT_X", "I_ERROR[R_X(theta=0.3)]", "I_ERROR[R_Y(theta=-0.25)]", "I_ERROR[R_Z(theta=0.4)]", "X_ERROR(0.1)", "Z_ERROR(0.15)", "DEPOLARIZE1(0.1)", "R", "RX", "M", "MX", "MR", "M(0.1)"];
        let ops2 = ["CX", "CZ", "CY", "SWAP", "II_ERROR[R_ZZ(theta=0.35)]", "II_ERROR[R_XX(theta=-0.2)]", "DEPOLARIZE2(0.1)"];
        for case in 0..150 {
            let n = 1 + (rng.next_u64() % 3) as usize;
            let mut lines = Vec::new();
            let (mut rotations, mut randomness, mut measured, mut noise) = (0, 0, 0usize, 0);
            while lines.len() < 12 {
                let two = n > 1 && rng.next_u64().is_multiple_of(3);
                let op = if two { ops2[(rng.next_u64() % ops2.len() as u64) as usize] } else { ops1[(rng.next_u64() % ops1.len() as u64) as usize] };
                if op.contains("ERROR(") || op.contains("DEPOLARIZE") || op.contains("M(") {
                    if noise == 2 {
                        continue;
                    }
                    noise += 1;
                }
                if op.contains("theta") {
                    if rotations == 4 {
                        continue;
                    }
                    rotations += 1;
                }
                if op.starts_with('M') || op.starts_with('R') && !op.starts_with("R_") {
                    if randomness == 4 {
                        continue;
                    }
                    randomness += 1;
                    measured += op.starts_with('M') as usize;
                }
                let a = (rng.next_u64() % n as u64) as usize;
                if two {
                    let b = (a + 1 + (rng.next_u64() % (n as u64 - 1)) as usize) % n;
                    lines.push(format!("{op} {a} {b}"));
                } else {
                    lines.push(format!("{op} {a}"));
                }
            }
            let finals: Vec<String> = (0..n).map(|q| q.to_string()).collect();
            lines.push(format!("M {}", finals.join(" ")));
            let total = measured + n;
            for k in 0..3usize.min(total) {
                let a = 1 + (rng.next_u64() % total as u64) as usize;
                let b = 1 + (rng.next_u64() % total as u64) as usize;
                lines.push(if a == b || k == 0 { format!("DETECTOR rec[-{a}]") } else { format!("DETECTOR rec[-{a}] rec[-{b}]") });
            }
            lines.push(format!("OBSERVABLE_INCLUDE(0) rec[-{}]", 1 + (rng.next_u64() % total as u64)));
            let text = lines.join("\n");
            let _ = case;
            check(&text);
        }
    }
}

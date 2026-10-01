//! From raw measurements to detection events, as `stim m2d` computes them.
//!
//! A detector compares a parity of measurements with what a noiseless run of
//! the circuit gives. One noiseless run, with every sweep bit clear, is the
//! reference. A sweep bit (`CX sweep[k] q`) is a classically controlled X, set
//! per shot to prepare the data qubits in a pattern of 0s and 1s. The circuit
//! is Clifford and its detectors deterministic, so each bit's effect on every
//! detector is fixed: one more noiseless run with that bit set finds it. A
//! shot's detector then fires when its records' parity differs from the
//! reference's, after the fixed flips of the shot's set sweep bits.
//!
//! The runs use the stabilizer tableau, not the frame machinery the error model
//! is built on, so this is a second, independent reading of the circuit.
//! Google's own detection events are its oracle.

use crate::circuit::{Basis, Circuit, Instr, Resolved};
use crate::simulator::StabilizerSimulator;

pub struct M2d {
    pub num_measurements: usize,
    pub num_sweep_bits: usize,
    pub num_detectors: usize,
    pub num_observables: usize,
    detectors: Vec<Vec<usize>>,
    observables: Vec<Vec<usize>>,
    ref_det: Vec<bool>,
    ref_obs: u64,
    /// Per sweep bit, the detectors it flips, and the observables.
    sweep_det: Vec<Vec<u32>>,
    sweep_obs: Vec<u64>,
}

fn reset(sim: &mut StabilizerSimulator, basis: Basis, q: usize) {
    match basis {
        Basis::Z => {
            if sim.measure_z(q) == 1 {
                sim.apply_x(q);
            }
        }
        Basis::X => {
            if sim.measure_x(q) == 1 {
                sim.apply_z(q);
            }
        }
    }
}

/// SplitMix64's finaliser. The tableau draws its random outcomes from an LCG,
/// whose first bits barely move between small consecutive seeds, so each run's
/// seed is mixed first: otherwise eight "independent" references can all flip
/// the same coin the same way.
fn mix(k: u64) -> u64 {
    let mut z = k.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// One noiseless run of the circuit with the given sweep bits: its measurement record.
fn run(res: &Resolved, sweeps: &[bool], seed: u64) -> Vec<bool> {
    let mut sim = StabilizerSimulator::with_seed(res.num_qubits.max(1), mix(seed));
    let mut rec = Vec::with_capacity(res.num_measurements);
    for ins in &res.instrs {
        match ins {
            Instr::Reset { basis, qubits } => {
                for &q in qubits {
                    reset(&mut sim, *basis, q as usize);
                }
            }
            Instr::H(qubits) => qubits.iter().for_each(|&q| sim.apply_h(q as usize)),
            Instr::Cx(pairs) => pairs.iter().for_each(|&(c, t)| sim.apply_cnot(c as usize, t as usize)),
            Instr::Cz(pairs) => {
                for &(a, b) in pairs {
                    sim.apply_h(b as usize);
                    sim.apply_cnot(a as usize, b as usize);
                    sim.apply_h(b as usize);
                }
            }
            Instr::Pauli { pauli, qubits } => {
                for &q in qubits {
                    match pauli {
                        1 => sim.apply_x(q as usize),
                        2 => sim.apply_z(q as usize),
                        3 => sim.apply_y(q as usize),
                        _ => {}
                    }
                }
            }
            Instr::SweepX(pairs) => {
                for &(k, q) in pairs {
                    if sweeps[k as usize] {
                        sim.apply_x(q as usize);
                    }
                }
            }
            Instr::Measure { basis, reset: then_reset, qubits, .. } => {
                for &q in qubits {
                    let q = q as usize;
                    let bit = match basis {
                        Basis::Z => sim.measure_z(q),
                        Basis::X => sim.measure_x(q),
                    };
                    rec.push(bit == 1);
                    if *then_reset {
                        reset(&mut sim, *basis, q);
                    }
                }
            }
            // Noise channels, annotations and ticks: the run is noiseless.
            _ => {}
        }
    }
    rec
}

fn parity(recs: &[usize], rec: &[bool]) -> bool {
    recs.iter().fold(false, |acc, &m| acc ^ rec[m])
}

fn evaluate(detectors: &[Vec<usize>], observables: &[Vec<usize>], rec: &[bool]) -> (Vec<bool>, u64) {
    let dets = detectors.iter().map(|r| parity(r, rec)).collect();
    let mut obs = 0u64;
    for (k, r) in observables.iter().enumerate() {
        if parity(r, rec) {
            obs |= 1 << k;
        }
    }
    (dets, obs)
}

/// A dense tableau of n qubits takes about n² / 2 bytes: 128 MiB at this size.
const MAX_QUBITS: usize = 1 << 14;

impl M2d {
    pub fn new(circuit: &Circuit) -> Result<M2d, String> {
        let res = circuit.resolve()?;
        if res.num_qubits > MAX_QUBITS {
            return Err(format!(
                "{} qubits: the reference run keeps a dense tableau, which holds at most {MAX_QUBITS}",
                res.num_qubits
            ));
        }
        let (detectors, observables) = (res.detectors.clone(), res.observables.clone());
        let clear = vec![false; res.num_sweep_bits];
        let (ref_det, ref_obs) = evaluate(&detectors, &observables, &run(&res, &clear, 1));
        // Seven more references, each with other random outcomes, must agree:
        // a detector that reads a coin flip survives all seven with
        // probability 1/128.
        for seed in 2..=8 {
            let (det, obs) = evaluate(&detectors, &observables, &run(&res, &clear, seed));
            if let Some(d) = (0..ref_det.len()).find(|&d| ref_det[d] != det[d]) {
                return Err(format!("detector D{d} is not deterministic"));
            }
            if ref_obs != obs {
                return Err("an observable is not deterministic".into());
            }
        }
        let mut sweep_det = Vec::with_capacity(res.num_sweep_bits);
        let mut sweep_obs = Vec::with_capacity(res.num_sweep_bits);
        for k in 0..res.num_sweep_bits {
            let mut bits = clear.clone();
            bits[k] = true;
            let (det, obs) = evaluate(&detectors, &observables, &run(&res, &bits, 9 + k as u64));
            sweep_det.push((0..det.len()).filter(|&d| det[d] != ref_det[d]).map(|d| d as u32).collect());
            sweep_obs.push(obs ^ ref_obs);
        }
        Ok(M2d {
            num_measurements: res.num_measurements,
            num_sweep_bits: res.num_sweep_bits,
            num_detectors: detectors.len(),
            num_observables: observables.len(),
            detectors,
            observables,
            ref_det,
            ref_obs,
            sweep_det,
            sweep_obs,
        })
    }

    /// One shot's detection events and observable flips. `meas` holds every
    /// measurement and `sweeps` every sweep bit of the circuit.
    pub fn convert(&self, meas: &[bool], sweeps: &[bool]) -> (Vec<bool>, u64) {
        assert!(
            meas.len() == self.num_measurements && sweeps.len() == self.num_sweep_bits,
            "a shot of {} measurements and {} sweep bits, for a circuit of {} and {}",
            meas.len(),
            sweeps.len(),
            self.num_measurements,
            self.num_sweep_bits
        );
        let (mut det, mut obs) = evaluate(&self.detectors, &self.observables, meas);
        for (d, &r) in det.iter_mut().zip(&self.ref_det) {
            *d ^= r;
        }
        obs ^= self.ref_obs;
        for (k, &set) in sweeps.iter().enumerate() {
            if set {
                for &d in &self.sweep_det[k] {
                    det[d as usize] ^= true;
                }
                obs ^= self.sweep_obs[k];
            }
        }
        (det, obs)
    }

    /// Many shots in Stim's b8 layout, rows padded to whole bytes: detection
    /// events, and observable flips.
    pub fn convert_b8(&self, meas: &[u8], sweeps: &[u8], num_shots: usize) -> Result<(Vec<u8>, Vec<u8>), String> {
        let (ms, ss) = (self.num_measurements.div_ceil(8), self.num_sweep_bits.div_ceil(8));
        if meas.len() != ms * num_shots || sweeps.len() != ss * num_shots {
            return Err(format!(
                "{} + {} bytes is not {num_shots} shots of {} measurements and {} sweep bits",
                meas.len(),
                sweeps.len(),
                self.num_measurements,
                self.num_sweep_bits
            ));
        }
        let unpack = |row: &[u8], n: usize| -> Vec<bool> { (0..n).map(|i| (row[i / 8] >> (i % 8)) & 1 == 1).collect() };
        let (mut dets, mut obs) = (Vec::new(), Vec::new());
        for s in 0..num_shots {
            let m = unpack(&meas[s * ms..(s + 1) * ms], self.num_measurements);
            let w = unpack(&sweeps[s * ss..(s + 1) * ss], self.num_sweep_bits);
            let (d, o) = self.convert(&m, &w);
            crate::shots::pack_row(&d, &mut dets);
            let ob: Vec<bool> = (0..self.num_observables).map(|k| (o >> k) & 1 == 1).collect();
            crate::shots::pack_row(&ob, &mut obs);
        }
        Ok((dets, obs))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::circuit::Circuit;

    #[test]
    fn a_sweep_bit_moves_the_reference() {
        let c = Circuit::parse(
            "R 0 1\nCX sweep[0] 0\nX 1\nM 0 1\nDETECTOR rec[-2]\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]\n",
        )
        .unwrap();
        let m = M2d::new(&c).unwrap();
        assert_eq!((m.num_measurements, m.num_sweep_bits, m.num_detectors), (2, 1, 2));
        // A noiseless shot with the bit set reads 1 1: nothing fired.
        assert_eq!(m.convert(&[true, true], &[true]), (vec![false, false], 0));
        // The same readout with the bit clear: qubit 0 flipped.
        assert_eq!(m.convert(&[true, true], &[false]), (vec![true, false], 0));
        // Qubit 1 read 0 against the reference's 1: D1 and the observable fire.
        assert_eq!(m.convert(&[false, false], &[false]), (vec![false, true], 1));
    }

    #[test]
    fn noiseless_runs_convert_to_silence() {
        use crate::circuit::Basis;
        use crate::memory::{generate, CodeKind, NoiseModel};
        // A memory circuit with a sweep bit on each of its nine data qubits,
        // right after their reset, as Google's circuits have them.
        let text = generate(CodeKind::Rotated, 3, 3, NoiseModel::Sd6 { p: 0.0 }, Basis::Z).unwrap().to_stim();
        let reset = "R 0 1 2 3 4 5 6 7 8\n";
        let sweep: String = (0..9).map(|k| format!(" sweep[{k}] {k}")).collect();
        let with = text.replacen(reset, &format!("{reset}CX{sweep}\n"), 1);
        assert_ne!(with, text, "the data qubits' reset line moved");
        let c = Circuit::parse(&with).unwrap();
        let m = M2d::new(&c).unwrap();
        assert_eq!(m.num_sweep_bits, 9);
        let res = c.resolve().unwrap();
        let mut rng = crate::surface_code::Xorshift::new(5);
        let mut fired_before = 0;
        for seed in 0..40u64 {
            let sweeps: Vec<bool> = (0..9).map(|_| rng.next_u64() & 1 == 1).collect();
            let meas = run(&res, &sweeps, 100 + seed);
            let (dets, obs) = m.convert(&meas, &sweeps);
            assert!(dets.iter().all(|&b| !b), "seed {seed}: {dets:?}");
            assert_eq!(obs, 0);
            // Not vacuous: ignoring the sweep bits would have fired detectors.
            fired_before += usize::from(m.convert(&meas, &[false; 9]).0.iter().any(|&b| b));
        }
        assert!(fired_before > 30, "{fired_before}");
    }

    #[test]
    fn circuits_too_wide_for_the_tableau_are_refused() {
        let err = M2d::new(&Circuit::parse("M 16384").unwrap()).err().unwrap_or_default();
        assert!(err.contains("dense tableau"), "{err}");
    }

    #[test]
    fn nondeterministic_detectors_are_refused() {
        let c = Circuit::parse("RX 0\nM 0\nDETECTOR rec[-1]\n").unwrap();
        assert!(M2d::new(&c).err().unwrap().contains("not deterministic"));
    }

    #[test]
    fn b8_conversion_matches_the_bool_path() {
        let c = Circuit::parse("R 0 1\nCX sweep[0] 0\nM 0 1\nDETECTOR rec[-2]\nDETECTOR rec[-1] rec[-2]\n").unwrap();
        let m = M2d::new(&c).unwrap();
        // Shots: meas (1, 0) with the bit set; meas (1, 1) with it clear.
        let (d, o) = m.convert_b8(&[0b01, 0b11], &[1, 0], 2).unwrap();
        assert_eq!(d, vec![0b00, 0b01]);
        assert_eq!(o, Vec::<u8>::new());
        assert!(m.convert_b8(&[0], &[0], 2).is_err());
    }
}

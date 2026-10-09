//! From raw measurements to detection events, as `stim m2d` computes them.
//!
//! A detector compares a parity of measurements with what a noiseless run of the circuit gives.
//! One noiseless run, with every sweep bit clear, is the reference. A sweep bit (`CX sweep[k]
//! q`) is a classically controlled X, set per shot to prepare the data qubits in a pattern of 0s
//! and 1s; the circuit is Clifford and its detectors deterministic, so each bit's effect on
//! every detector is fixed. A shot's detector then fires when its records' parity differs from
//! the reference's, after the fixed flips of the shot's set sweep bits.
//!
//! Nothing is unrolled. The reference is one run of the stabilizer tableau through the circuit
//! as written, its loops run pass by pass; that every detector is deterministic, and what each
//! sweep bit flips, come from the error model's backward walk (`dem_build::sweep_effects`), with
//! loops folded; and each shot is read by a program of the circuit's measurements, detectors and
//! observables alone, its loops kept. The tableau is not the frame machinery the error model is
//! built on, so this is a second, independent reading of the circuit; Google's own detection
//! events are its oracle.

use crate::obsbits::ObsBits;
use crate::batch_sampler::Counts;
use crate::circuit::{Basis, Circuit, Control, Instr};
use crate::simulator::StabilizerSimulator;

/// The circuit as the conversion reads it: how many records each step makes, and which records
/// each detector and observable reads (lookbacks, 1 the latest).
#[derive(Debug)]
enum Op {
    Records(u64),
    Detector(Vec<u32>),
    Observable(u32, Vec<u32>),
    Repeat(u64, Vec<Op>),
}

fn compile(instrs: &[Instr], out: &mut Vec<Op>) {
    for ins in instrs {
        let records = match ins {
            Instr::Measure { qubits, .. } | Instr::Heralded { qubits, .. } => qubits.len() as u64,
            Instr::Pad { values, .. } => values.len() as u64,
            Instr::Detector { recs, .. } => {
                out.push(Op::Detector(recs.clone()));
                continue;
            }
            Instr::Observable { index, recs, .. } => {
                if !recs.is_empty() {
                    out.push(Op::Observable(*index, recs.clone()));
                }
                continue;
            }
            Instr::Gate { body, .. } => {
                compile(body, out);
                continue;
            }
            Instr::Repeat { count, body, .. } => {
                let mut inner = Vec::new();
                compile(body, &mut inner);
                if *count > 0 && !inner.is_empty() {
                    out.push(Op::Repeat(*count, inner));
                }
                continue;
            }
            _ => continue,
        };
        match out.last_mut() {
            Some(Op::Records(n)) => *n += records,
            _ => out.push(Op::Records(records)),
        }
    }
}

/// Walk the program over one shot's records, `bit(i)` the i-th: each detector's parity in turn
/// to `detector`, the observables' to `obs`.
fn walk(ops: &[Op], m: &mut u64, bit: &impl Fn(u64) -> bool, detector: &mut impl FnMut(bool), obs: &mut ObsBits) {
    for op in ops {
        match op {
            Op::Records(n) => *m += n,
            Op::Detector(recs) => detector(recs.iter().fold(false, |acc, &k| acc ^ bit(*m - u64::from(k)))),
            Op::Observable(i, recs) => {
                if recs.iter().fold(false, |acc, &k| acc ^ bit(*m - u64::from(k))) {
                    obs.flip(*i as usize);
                }
            }
            Op::Repeat(count, body) => {
                for _ in 0..*count {
                    walk(body, m, bit, detector, obs);
                }
            }
        }
    }
}

/// `walk` 64 shots at a time: `records[i]` holds record i of each shot (bit s, shot s), each
/// detector's word goes to `detector`, and the observables' to `obs` (one word each).
fn walk_words(ops: &[Op], m: &mut u64, records: &[u64], detector: &mut impl FnMut(u64), obs: &mut [u64]) {
    for op in ops {
        match op {
            Op::Records(n) => *m += n,
            Op::Detector(recs) => detector(recs.iter().fold(0, |acc, &k| acc ^ records[(*m - u64::from(k)) as usize])),
            Op::Observable(i, recs) => obs[*i as usize] ^= recs.iter().fold(0, |acc, &k| acc ^ records[(*m - u64::from(k)) as usize]),
            Op::Repeat(count, body) => {
                for _ in 0..*count {
                    walk_words(body, m, records, detector, obs);
                }
            }
        }
    }
}

/// Transpose a 64×64 bit matrix in place (row r's bit c to row c's bit r), by swapping ever
/// smaller off-diagonal blocks.
fn transpose64(a: &mut [u64; 64]) {
    let (mut j, mut m) = (32, 0x0000_0000_FFFF_FFFFu64);
    while j != 0 {
        for k in (0..64).filter(|k| k & j == 0) {
            let t = ((a[k] >> j) ^ a[k | j]) & m;
            a[k] ^= t << j;
            a[k | j] ^= t;
        }
        j >>= 1;
        m ^= m << j;
    }
}

/// Up to 64 rows of `width` bytes (bit i of a row: byte i / 8, bit i % 8) as words, one per
/// column: `words[i]` bit s is row s's bit i.
fn to_words(rows: &[u8], width: usize, num_rows: usize, words: &mut [u64]) {
    let mut block = [0u64; 64];
    for (c, chunk) in words.chunks_mut(64).enumerate() {
        for (s, b) in block.iter_mut().enumerate() {
            *b = 0;
            if s < num_rows {
                let row = &rows[s * width..(s + 1) * width];
                for (byte, &v) in row.iter().skip(c * 8).take(8).enumerate() {
                    *b |= u64::from(v) << (8 * byte);
                }
            }
        }
        transpose64(&mut block);
        chunk.copy_from_slice(&block);
    }
}

/// The inverse: words (one per column, bit s row s) back to up to 64 rows of `width` bytes.
fn from_words(words: &mut [u64], rows: &mut [u8], width: usize, num_rows: usize) {
    let mut block = [0u64; 64];
    for (c, chunk) in words.chunks_mut(64).enumerate() {
        block[..chunk.len()].copy_from_slice(chunk);
        block[chunk.len()..].fill(0);
        transpose64(&mut block);
        for (s, &b) in block.iter().enumerate().take(num_rows) {
            let row = &mut rows[s * width..(s + 1) * width];
            for (byte, v) in row.iter_mut().skip(c * 8).take(8).enumerate() {
                *v = (b >> (8 * byte)) as u8;
            }
        }
    }
}

pub struct M2d {
    pub num_measurements: usize,
    pub num_sweep_bits: usize,
    pub num_detectors: usize,
    pub num_observables: usize,
    ops: Vec<Op>,
    ref_det: Vec<bool>,
    ref_obs: ObsBits,
    /// Per sweep bit, the detectors it flips, and the observables.
    sweep_det: Vec<Vec<u32>>,
    sweep_obs: Vec<ObsBits>,
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

/// SplitMix64's finaliser. The tableau draws its random outcomes from an LCG, whose first bits
/// barely move between small consecutive seeds, so each run's seed is mixed first.
fn mix(k: u64) -> u64 {
    let mut z = k.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// One noiseless run of the circuit with the given sweep bits, its loops run pass by pass: its
/// measurement record.
pub(crate) fn run(circuit: &Circuit, num_qubits: usize, sweeps: &[bool], seed: u64) -> Vec<bool> {
    let mut sim = StabilizerSimulator::with_seed(num_qubits.max(1), mix(seed));
    let mut rec = Vec::new();
    run_block(&circuit.instrs, &mut sim, sweeps, &mut rec);
    rec
}

fn run_block(instrs: &[Instr], sim: &mut StabilizerSimulator, sweeps: &[bool], rec: &mut Vec<bool>) {
    for ins in instrs {
        match ins {
            Instr::Reset { basis, qubits } => {
                for &q in qubits {
                    reset(sim, *basis, q as usize);
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
                    if sweeps.get(k as usize).copied().unwrap_or(false) {
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
                        reset(sim, *basis, q);
                    }
                }
            }
            Instr::S(qubits) => qubits.iter().for_each(|&q| sim.apply_s(q as usize)),
            Instr::Pad { values, .. } => rec.extend(values.iter().copied()),
            Instr::Feedback { pauli, control, qubit } => {
                let on = match control {
                    Control::Rec(k) => rec[rec.len() - *k as usize],
                    Control::Sweep(k) => sweeps.get(*k as usize).copied().unwrap_or(false),
                };
                if on {
                    match pauli {
                        1 => sim.apply_x(*qubit as usize),
                        2 => sim.apply_z(*qubit as usize),
                        _ => sim.apply_y(*qubit as usize),
                    }
                }
            }
            // A herald never fires in the noiseless run.
            Instr::Heralded { qubits, .. } => rec.extend(qubits.iter().map(|_| false)),
            Instr::Repeat { count, body, .. } => {
                for _ in 0..*count {
                    run_block(body, sim, sweeps, rec);
                }
            }
            Instr::Gate { body, .. } => run_block(body, sim, sweeps, rec),
            // Noise channels, annotations and ticks: the run is noiseless.
            Instr::PauliError { .. }
            | Instr::Depolarize1 { .. }
            | Instr::Depolarize2 { .. }
            | Instr::PauliChannel1 { .. }
            | Instr::PauliChannel2 { .. }
            | Instr::Correlated { .. }
            | Instr::Detector { .. }
            | Instr::Observable { .. }
            | Instr::QubitCoords { .. }
            | Instr::ShiftCoords(_)
            | Instr::NonPauli(_)
            | Instr::Tick => {}
        }
    }
}

/// A dense tableau of n qubits takes about n² / 2 bytes: 128 MiB at this size.
const MAX_QUBITS: usize = 1 << 14;

/// The most instructions and targets the reference run may take, its loops counted pass by
/// pass: 2³² is minutes of tableau work.
const MAX_REFERENCE: u64 = 1 << 32;

impl M2d {
    pub fn new(circuit: &Circuit) -> Result<M2d, String> {
        let counts = Counts::of(&circuit.instrs)?;
        if counts.qubits > MAX_QUBITS {
            return Err(format!(
                "{} qubits: the reference run keeps a dense tableau, which holds at most {MAX_QUBITS}",
                counts.qubits
            ));
        }
        let size = crate::circuit::unrolled_size(&circuit.instrs);
        if size > MAX_REFERENCE {
            return Err(format!(
                "the circuit runs to {size} instructions and targets, more than the {MAX_REFERENCE} its reference run takes"
            ));
        }
        let num_measurements = usize::try_from(counts.measurements).map_err(|_| "too many measurements for this machine")?;
        let num_detectors = usize::try_from(counts.detectors).map_err(|_| "too many detectors for this machine")?;
        // Deterministic detectors and observables, and the sweep bits' effects, by the backward
        // walk; it also refuses records read before the first measurement.
        let effects = crate::dem_build::sweep_effects(circuit)?;
        let mut ops = Vec::new();
        compile(&circuit.instrs, &mut ops);
        let reference = run(circuit, counts.qubits, &[], 1);
        let (mut ref_det, mut ref_obs) = (Vec::with_capacity(num_detectors), ObsBits::new());
        walk(&ops, &mut 0, &|i| reference[i as usize], &mut |b| ref_det.push(b), &mut ref_obs);
        let (sweep_det, sweep_obs) = effects.into_iter().unzip();
        Ok(M2d {
            num_measurements,
            num_sweep_bits: counts.sweep_bits,
            num_detectors,
            num_observables: counts.observables,
            ops,
            ref_det,
            ref_obs,
            sweep_det,
            sweep_obs,
        })
    }

    /// One shot's detection events and observable flips. `meas` holds every measurement and
    /// `sweeps` every sweep bit of the circuit.
    pub fn convert(&self, meas: &[bool], sweeps: &[bool]) -> (Vec<bool>, u64) {
        let (det, obs) = self.convert_wide(meas, sweeps);
        (det, obs.low())
    }

    /// As `convert`, with every observable.
    pub fn convert_wide(&self, meas: &[bool], sweeps: &[bool]) -> (Vec<bool>, ObsBits) {
        assert!(
            meas.len() == self.num_measurements && sweeps.len() == self.num_sweep_bits,
            "a shot of {} measurements and {} sweep bits, for a circuit of {} and {}",
            meas.len(),
            sweeps.len(),
            self.num_measurements,
            self.num_sweep_bits
        );
        let mut det = Vec::with_capacity(self.num_detectors);
        let mut obs = self.ref_obs.clone();
        walk(&self.ops, &mut 0, &|i| meas[i as usize], &mut |b| det.push(b ^ self.ref_det[det.len()]), &mut obs);
        for (k, &set) in sweeps.iter().enumerate() {
            if set {
                for &d in &self.sweep_det[k] {
                    det[d as usize] ^= true;
                }
                obs.xor_with(&self.sweep_obs[k]);
            }
        }
        (det, obs)
    }

    /// Many shots in Stim's b8 layout, rows padded to whole bytes: detection events, and
    /// observable flips.
    pub fn convert_b8(&self, meas: &[u8], sweeps: &[u8], num_shots: usize) -> Result<(Vec<u8>, Vec<u8>), String> {
        self.check_sizes(meas, sweeps, num_shots)?;
        let (ms, ss) = (self.num_measurements.div_ceil(8), self.num_sweep_bits.div_ceil(8));
        let (ds, os) = (self.num_detectors.div_ceil(8), self.num_observables.div_ceil(8));
        let mut dets = vec![0u8; ds * num_shots];
        let mut obs_out = vec![0u8; os * num_shots];
        // 64 shots at a time, as words: one per measurement (bit s the block's shot s), so each
        // detector is the XOR of its records' words, as Stim converts.
        let mut records = vec![0u64; self.num_measurements.div_ceil(64) * 64];
        let mut sweep_words = vec![0u64; self.num_sweep_bits.div_ceil(64) * 64];
        let mut det_words = vec![0u64; self.num_detectors.div_ceil(64) * 64];
        let mut obs_words = vec![0u64; self.num_observables.div_ceil(64).max(1) * 64];
        for first in (0..num_shots).step_by(64) {
            let shots = (num_shots - first).min(64);
            to_words(&meas[first * ms..(first + shots) * ms], ms, shots, &mut records);
            to_words(&sweeps[first * ss..(first + shots) * ss], ss, shots, &mut sweep_words);
            for (k, w) in obs_words.iter_mut().enumerate() {
                *w = if self.ref_obs.bit(k) { u64::MAX } else { 0 };
            }
            let mut d = 0usize;
            walk_words(&self.ops, &mut 0, &records, &mut |w| {
                det_words[d] = if self.ref_det[d] { !w } else { w };
                d += 1;
            }, &mut obs_words);
            for k in 0..self.num_sweep_bits {
                let w = sweep_words[k];
                if w != 0 {
                    for &d in &self.sweep_det[k] {
                        det_words[d as usize] ^= w;
                    }
                    for (j, o) in obs_words.iter_mut().enumerate() {
                        if self.sweep_obs[k].bit(j) {
                            *o ^= w;
                        }
                    }
                }
            }
            from_words(&mut det_words, &mut dets[first * ds..(first + shots) * ds], ds, shots);
            from_words(&mut obs_words[..], &mut obs_out[first * os..(first + shots) * os], os, shots);
        }
        Ok((dets, obs_out))
    }

    /// The same, shot by shot: the conversion the word-wide one is checked against.
    #[cfg(test)]
    fn convert_b8_by_shot(&self, meas: &[u8], sweeps: &[u8], num_shots: usize) -> Result<(Vec<u8>, Vec<u8>), String> {
        self.check_sizes(meas, sweeps, num_shots)?;
        let (ms, ss) = (self.num_measurements.div_ceil(8), self.num_sweep_bits.div_ceil(8));
        let (ds, os) = (self.num_detectors.div_ceil(8), self.num_observables.div_ceil(8));
        let mut dets = vec![0u8; ds * num_shots];
        let mut obs_out = vec![0u8; os * num_shots];
        for s in 0..num_shots {
            let row = &meas[s * ms..(s + 1) * ms];
            let out = &mut dets[s * ds..(s + 1) * ds];
            let mut d = 0usize;
            let mut obs = self.ref_obs.clone();
            walk(
                &self.ops,
                &mut 0,
                &|i| (row[(i / 8) as usize] >> (i % 8)) & 1 == 1,
                &mut |b| {
                    if b ^ self.ref_det[d] {
                        out[d / 8] |= 1 << (d % 8);
                    }
                    d += 1;
                },
                &mut obs,
            );
            let sw = &sweeps[s * ss..(s + 1) * ss];
            for k in 0..self.num_sweep_bits {
                if (sw[k / 8] >> (k % 8)) & 1 == 1 {
                    for &d in &self.sweep_det[k] {
                        out[d as usize / 8] ^= 1 << (d % 8);
                    }
                    obs.xor_with(&self.sweep_obs[k]);
                }
            }
            for k in 0..self.num_observables {
                if obs.bit(k) {
                    obs_out[s * os + k / 8] |= 1 << (k % 8);
                }
            }
        }
        Ok((dets, obs_out))
    }

    fn check_sizes(&self, meas: &[u8], sweeps: &[u8], num_shots: usize) -> Result<(), String> {
        let (ms, ss) = (self.num_measurements.div_ceil(8), self.num_sweep_bits.div_ceil(8));
        if ms.checked_mul(num_shots) != Some(meas.len()) || ss.checked_mul(num_shots) != Some(sweeps.len()) {
            return Err(format!(
                "{} + {} bytes is not {num_shots} shots of {} measurements and {} sweep bits",
                meas.len(),
                sweeps.len(),
                self.num_measurements,
                self.num_sweep_bits
            ));
        }
        Ok(())
    }

}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::circuit::Circuit;

    #[test]
    fn transpose64_transposes() {
        let mut rng = crate::surface_code::Xorshift::new(9);
        let mut a = [0u64; 64];
        for x in a.iter_mut() {
            *x = rng.next_u64();
        }
        let mut b = a;
        transpose64(&mut b);
        for r in 0..64 {
            for c in 0..64 {
                assert_eq!((b[c] >> r) & 1, (a[r] >> c) & 1);
            }
        }
    }

    #[test]
    fn word_wide_conversion_equals_shot_by_shot() {
        // Random circuits with loops, feedback, sweeps and many measurements; shot counts
        // around the 64-shot blocks; random (padded) rows.
        let mut rng = crate::surface_code::Xorshift::new(4);
        let mut inputs = crate::fuzzing::Inputs::new(5);
        let mut checked = 0;
        for _ in 0..4000 {
            let text = inputs.circuit();
            let Ok(c) = Circuit::parse(&text) else { continue };
            let Ok(m) = M2d::new(&c) else { continue };
            let (ms, ss) = (m.num_measurements.div_ceil(8), m.num_sweep_bits.div_ceil(8));
            for shots in [0usize, 1, 63, 64, 65, 130] {
                let meas: Vec<u8> = (0..ms * shots).map(|_| rng.next_u64() as u8).collect();
                let sweeps: Vec<u8> = (0..ss * shots).map(|_| rng.next_u64() as u8).collect();
                assert_eq!(m.convert_b8(&meas, &sweeps, shots), m.convert_b8_by_shot(&meas, &sweeps, shots), "{text}");
            }
            checked += 1;
        }
        assert!(checked >= 100, "only {checked} circuits converted");
        let big = crate::generated::generate("surface_code:rotated_memory_x", 5, 5, &Default::default()).unwrap();
        let m = M2d::new(&Circuit::parse(&big).unwrap()).unwrap();
        let meas: Vec<u8> = (0..m.num_measurements.div_ceil(8) * 200).map(|_| rng.next_u64() as u8).collect();
        assert_eq!(m.convert_b8(&meas, &[], 200), m.convert_b8_by_shot(&meas, &[], 200));
    }

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
        let nq = crate::batch_sampler::Counts::of(&c.instrs).unwrap().qubits;
        let mut rng = crate::surface_code::Xorshift::new(5);
        let mut fired_before = 0;
        for seed in 0..40u64 {
            let sweeps: Vec<bool> = (0..9).map(|_| rng.next_u64() & 1 == 1).collect();
            let meas = run(&c, nq, &sweeps, 100 + seed);
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

    #[test]
    fn long_loops_convert_without_unrolling() {
        // 2 million rounds of a repetition-code-like check: 2 million detectors, far past what
        // unrolling allowed.
        let c = Circuit::parse(
            "R 0 1\nCX sweep[0] 0\nMR 1\nREPEAT 2000000 {\n CX 0 1\n MR 1\n DETECTOR rec[-1] rec[-2]\n}\nM 0\nOBSERVABLE_INCLUDE(0) rec[-1]",
        )
        .unwrap();
        let m = M2d::new(&c).unwrap();
        assert_eq!((m.num_measurements, m.num_detectors, m.num_sweep_bits), (2_000_002, 2_000_000, 1));
        // The sweep bit flips qubit 0: the first round's detector, and the observable.
        assert_eq!((m.sweep_det[0].clone(), m.sweep_obs[0].low()), (vec![0], 1));
        let mut meas = vec![true; 2_000_002];
        meas[0] = false;
        let (d, o) = m.convert(&meas, &[true]);
        assert!(d.iter().all(|&b| !b) && o == 0);
        let (d, o) = m.convert(&meas, &[false]);
        assert!(d[0] && d[1..].iter().all(|&b| !b) && o == 1);
    }

    #[test]
    fn a_walk_that_never_settles_is_refused_not_endless() {
        // Qubit 0 is never reset and every round reads it, so the walk's state grows by a
        // detector a pass and never repeats.
        let c = Circuit::parse("R 0 1\nREPEAT 2000000 {\n CX 0 1\n MR 1\n DETECTOR rec[-1]\n}").unwrap();
        let t = std::time::Instant::now();
        assert!(M2d::new(&c).err().unwrap_or_default().contains("too large"));
        assert!(t.elapsed().as_secs() < 60);
    }
}

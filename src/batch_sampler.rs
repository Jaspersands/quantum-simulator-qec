//! Sampling 64 shots at once, one bit per shot in every machine word.
//!
//! WHY THIS EXISTS
//! ---------------
//! `FrameSampler` carries one shot's Pauli frame as a vector of booleans and
//! walks the whole flattened circuit per shot. That is simple and it is the
//! reference, but a gate costs the same for one shot as a word-wide XOR costs
//! for 64, and a flattened million-round circuit does not fit in memory. This
//! sampler keeps the X and Z parts of 64 frames in two words per qubit, so each
//! gate is a couple of word operations; it draws noise words by geometric
//! skipping, so a location at p = 10⁻³ costs about one random number rather
//! than 64; and it runs `REPEAT` blocks by iterating, with measurement records
//! in a ring buffer and detectors evaluated from their lookbacks as they are
//! reached, so a million rounds cost the loop body's memory and the detection
//! events stream out as they are made.
//!
//! The semantics are `FrameSampler`'s exactly: the same disjoint channels, the
//! same Pauli numbering for the composite channels, and the same randomisation
//! of the component a reset or measurement makes irrelevant, so that a
//! nondeterministic detector shows as a coin flip rather than hiding.

use crate::circuit::{Basis, Circuit, Instr};
use crate::surface_code::Xorshift;

/// A probability, with what geometric skipping needs precomputed.
#[derive(Clone, Copy, Debug)]
struct Noise {
    p: f64,
    log1mp: f64,
}

impl Noise {
    fn new(p: f64) -> Noise {
        // ln_1p keeps ln(1 − p) nonzero for p far below 1e-16, where 1 − p
        // rounds to 1; a p it still cannot resolve is treated as zero.
        let log1mp = if p > 0.0 && p < 1.0 { (-p).ln_1p() } else { 0.0 };
        Noise { p: if log1mp == 0.0 && p < 1.0 { 0.0 } else { p }, log1mp }
    }
}

/// A word whose bits are independently set with probability `n.p`.
#[inline]
fn bernoulli(rng: &mut Xorshift, n: Noise) -> u64 {
    if n.p <= 0.0 {
        return 0;
    }
    if n.p >= 1.0 {
        return !0;
    }
    if n.p < 0.25 {
        // The gap before the next set bit is geometric: floor(ln U / ln(1 − p)).
        let mut w = 0u64;
        let mut i: i64 = -1;
        loop {
            let u = rng.next_f64();
            // The cast saturates, and a gap of 64 or more ends the word anyway.
            i += 1 + (((1.0 - u).ln() / n.log1mp) as i64).min(64);
            if i >= 64 {
                return w;
            }
            w |= 1u64 << i;
        }
    }
    let mut w = 0u64;
    for k in 0..64 {
        if rng.next_f64() < n.p {
            w |= 1u64 << k;
        }
    }
    w
}

#[derive(Debug)]
enum Op {
    Reset { basis: Basis, qubits: Vec<u32> },
    H(Vec<u32>),
    Cx(Vec<(u32, u32)>),
    Cz(Vec<(u32, u32)>),
    Measure { basis: Basis, reset: bool, flip: Noise, qubits: Vec<u32> },
    PauliError { pauli: u8, noise: Noise, qubits: Vec<u32> },
    Depolarize1 { noise: Noise, qubits: Vec<u32> },
    Depolarize2 { noise: Noise, pairs: Vec<(u32, u32)> },
    PauliChannel1 { any: Noise, px: f64, pxy: f64, total: f64, qubits: Vec<u32> },
    /// Lookbacks (1 is the latest measurement), and the detector's last
    /// coordinate as written, with its index: its time, before shifts.
    Detector(Vec<u32>, Option<(usize, f64)>),
    ShiftCoords(Vec<f64>),
    /// An observable's records (lookbacks), and its Pauli targets: the frame's Pauli there
    /// flips it where it anticommutes with one.
    Observable(u32, Vec<u32>, Vec<(u32, u8)>),
    Repeat(u64, Vec<Op>),
    S(Vec<u32>),
    /// `E` / `ELSE_CORRELATED_ERROR`: the product on the lanes that fire; a chained one fires
    /// only on lanes where its chain has not.
    Correlated { noise: Noise, paulis: Vec<(u32, u8)>, chained: bool },
    /// `PAULI_CHANNEL_2`: the lanes any case fires on, then one case each by its share.
    PauliChannel2 { any: Noise, cumulative: Vec<f64>, pairs: Vec<(u32, u32)> },
    /// `MPAD`: records that, relative to the noiseless run, hold only their flips.
    Pad { flip: Noise, count: usize },
    /// A Pauli on a qubit where a record (lookback) flipped relative to the noiseless run.
    Feedback { pauli: u8, lookback: u32, qubit: u32 },
    /// Heralded errors: per qubit, the herald record, and on the lanes it fires a Pauli by
    /// its share (I, X, Y, Z).
    Heralded { any: Noise, cumulative: [f64; 4], qubits: Vec<u32> },
}

fn compile(instrs: &[Instr]) -> Vec<Op> {
    let mut out = Vec::new();
    for ins in instrs {
        out.push(match ins {
            Instr::Reset { basis, qubits } => Op::Reset { basis: *basis, qubits: qubits.clone() },
            Instr::H(q) => Op::H(q.clone()),
            Instr::Cx(p) => Op::Cx(p.clone()),
            Instr::Cz(p) => Op::Cz(p.clone()),
            Instr::Measure { basis, reset, flip, qubits } => {
                Op::Measure { basis: *basis, reset: *reset, flip: Noise::new(*flip), qubits: qubits.clone() }
            }
            Instr::PauliError { pauli, p, qubits } => {
                Op::PauliError { pauli: *pauli, noise: Noise::new(*p), qubits: qubits.clone() }
            }
            Instr::Depolarize1 { p, qubits } => Op::Depolarize1 { noise: Noise::new(*p), qubits: qubits.clone() },
            Instr::Depolarize2 { p, pairs } => Op::Depolarize2 { noise: Noise::new(*p), pairs: pairs.clone() },
            Instr::PauliChannel1 { px, py, pz, qubits } => {
                let total = px + py + pz;
                Op::PauliChannel1 { any: Noise::new(total), px: *px, pxy: px + py, total, qubits: qubits.clone() }
            }
            Instr::Detector { recs, coords } => {
                Op::Detector(recs.clone(), coords.last().map(|&t| (coords.len() - 1, t)))
            }
            Instr::ShiftCoords(shift) => Op::ShiftCoords(shift.clone()),
            Instr::Observable { index, recs, paulis } => {
                Op::Observable(*index, recs.clone(), paulis.iter().map(|&(q, p, _)| (q, p)).collect())
            }
            Instr::Repeat { count, body, .. } => Op::Repeat(*count, compile(body)),
            Instr::S(q) => Op::S(q.clone()),
            Instr::Gate { body, .. } => {
                out.extend(compile(body));
                continue;
            }
            Instr::Correlated { p, paulis, chained } => {
                Op::Correlated { noise: Noise::new(*p), paulis: paulis.clone(), chained: *chained }
            }
            Instr::PauliChannel2 { probs, pairs } => {
                let total: f64 = probs.iter().sum();
                let cumulative = probs.iter().scan(0.0, |acc, p| {
                    *acc += p;
                    Some(*acc)
                });
                Op::PauliChannel2 { any: Noise::new(total), cumulative: cumulative.collect(), pairs: pairs.clone() }
            }
            Instr::Pad { flip, values } => Op::Pad { flip: Noise::new(*flip), count: values.len() },
            Instr::Feedback { pauli, control: crate::circuit::Control::Rec(k), qubit } => {
                Op::Feedback { pauli: *pauli, lookback: *k, qubit: *qubit }
            }
            // A sweep bit moves only the noiseless reference, which a frame is relative to.
            Instr::Feedback { control: crate::circuit::Control::Sweep(_), .. } => continue,
            Instr::Heralded { probs, qubits, .. } => {
                let mut cumulative = [0.0; 4];
                let mut acc = 0.0;
                for (c, p) in cumulative.iter_mut().zip(probs) {
                    acc += p;
                    *c = acc;
                }
                Op::Heralded { any: Noise::new(acc), cumulative, qubits: qubits.clone() }
            }
            // Pauli gates and sweep-controlled X only flip signs, which a frame
            // relative to the noiseless run does not carry; annotations and
            // ticks do nothing.
            Instr::Pauli { .. } | Instr::SweepX(_) | Instr::QubitCoords { .. } | Instr::Tick => {
                continue
            }
        });
    }
    out
}

/// Counts through repeats, the largest lookback, and a check that no lookback
/// reaches before the first measurement (the first pass through a loop is the
/// only one that could), unless `lenient`.
#[derive(Default)]
struct Shape {
    measurements: u64,
    detectors: u64,
    observables: usize,
    qubits: usize,
    lookback: u32,
    lenient: bool,
}

/// What a circuit holds, counted through its loops without unrolling them.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Counts {
    pub qubits: usize,
    pub measurements: u64,
    pub detectors: u64,
    pub observables: usize,
    pub sweep_bits: usize,
}

impl Counts {
    /// A circuit's counts. Its lookbacks are not checked: a piece of a circuit may read records
    /// made before it, and sampling or analysing the whole is what refuses one that reaches
    /// before the first measurement.
    pub fn of(instrs: &[Instr]) -> Result<Counts, String> {
        let mut s = Shape { lenient: true, ..Shape::default() };
        shape(&compile(instrs), &mut s)?;
        Ok(Counts {
            qubits: instrs.iter().flat_map(Instr::qubits).map(|q| q as usize + 1).max().unwrap_or(0),
            measurements: s.measurements,
            detectors: s.detectors,
            observables: s.observables,
            sweep_bits: max_sweep_bit(instrs),
        })
    }

    /// The counts of this circuit followed by `next`.
    pub fn then(self, next: Counts) -> Result<Counts, String> {
        let overflow = || "the circuit's measurements or detectors overflow a count".to_string();
        Ok(Counts {
            qubits: self.qubits.max(next.qubits),
            measurements: self.measurements.checked_add(next.measurements).ok_or_else(overflow)?,
            detectors: self.detectors.checked_add(next.detectors).ok_or_else(overflow)?,
            observables: self.observables.max(next.observables),
            sweep_bits: self.sweep_bits.max(next.sweep_bits),
        })
    }

    /// The counts of `REPEAT count { this }`.
    pub fn times(self, count: u64) -> Result<Counts, String> {
        let overflow = || format!("REPEAT {count}: the loop's measurements or detectors overflow a count");
        Ok(Counts {
            measurements: self.measurements.checked_mul(count).ok_or_else(overflow)?,
            detectors: self.detectors.checked_mul(count).ok_or_else(overflow)?,
            // As `of` counts a loop never run: its qubits and sweep bits named, its observables not.
            observables: if count == 0 { 0 } else { self.observables },
            ..self
        })
    }
}

/// One more than the largest sweep bit the instructions read.
fn max_sweep_bit(instrs: &[Instr]) -> usize {
    instrs
        .iter()
        .map(|i| match i {
            Instr::SweepX(pairs) => pairs.iter().map(|&(k, _)| k as usize + 1).max().unwrap_or(0),
            Instr::Feedback { control: crate::circuit::Control::Sweep(k), .. } => *k as usize + 1,
            Instr::Repeat { body, .. } | Instr::Gate { body, .. } => max_sweep_bit(body),
            _ => 0,
        })
        .max()
        .unwrap_or(0)
}

fn shape(ops: &[Op], s: &mut Shape) -> Result<(), String> {
    for op in ops {
        let touch = |s: &mut Shape, q: u32| s.qubits = s.qubits.max(q as usize + 1);
        match op {
            Op::Reset { qubits, .. }
            | Op::H(qubits)
            | Op::PauliError { qubits, .. }
            | Op::Depolarize1 { qubits, .. }
            | Op::PauliChannel1 { qubits, .. } => qubits.iter().for_each(|&q| touch(s, q)),
            Op::Cx(pairs) | Op::Cz(pairs) | Op::Depolarize2 { pairs, .. } => {
                pairs.iter().for_each(|&(a, b)| {
                    touch(s, a);
                    touch(s, b)
                })
            }
            Op::Measure { qubits, .. } => {
                qubits.iter().for_each(|&q| touch(s, q));
                s.measurements = s.measurements.saturating_add(qubits.len() as u64);
            }
            Op::S(qubits) => qubits.iter().for_each(|&q| touch(s, q)),
            Op::Correlated { paulis, .. } => paulis.iter().for_each(|&(q, _)| touch(s, q)),
            Op::PauliChannel2 { pairs, .. } => pairs.iter().for_each(|&(a, b)| {
                touch(s, a);
                touch(s, b)
            }),
            Op::Pad { count, .. } => s.measurements = s.measurements.saturating_add(*count as u64),
            Op::Heralded { qubits, .. } => {
                qubits.iter().for_each(|&q| touch(s, q));
                s.measurements = s.measurements.saturating_add(qubits.len() as u64);
            }
            Op::Feedback { lookback, qubit, .. } => {
                touch(s, *qubit);
                if (*lookback == 0 || u64::from(*lookback) > s.measurements) && !s.lenient {
                    return Err(format!("rec[-{lookback}] reaches before the first measurement"));
                }
                s.lookback = s.lookback.max(*lookback);
            }
            Op::ShiftCoords(_) => {}
            Op::Detector(recs, _) | Op::Observable(_, recs, _) => {
                for &k in recs {
                    if (k == 0 || u64::from(k) > s.measurements) && !s.lenient {
                        return Err(format!("rec[-{k}] reaches before the first measurement"));
                    }
                    s.lookback = s.lookback.max(k);
                }
                match op {
                    Op::Detector(..) => s.detectors = s.detectors.saturating_add(1),
                    Op::Observable(i, _, paulis) => {
                        paulis.iter().for_each(|&(q, _)| touch(s, q));
                        if *i >= 64 {
                            return Err(format!("OBSERVABLE_INCLUDE({i}): at most 64 observables are supported"));
                        }
                        s.observables = s.observables.max(*i as usize + 1)
                    }
                    _ => {}
                }
            }
            Op::Repeat(0, _) => {}
            Op::Repeat(count, body) => {
                let (m0, d0) = (s.measurements, s.detectors);
                shape(body, s)?;
                let (dm, dd) = (s.measurements - m0, s.detectors - d0);
                let too_many = || format!("REPEAT {count}: the loop's measurements or detectors overflow a count");
                s.measurements = dm.checked_mul(*count).and_then(|n| n.checked_add(m0)).ok_or_else(too_many)?;
                s.detectors = dd.checked_mul(*count).and_then(|n| n.checked_add(d0)).ok_or_else(too_many)?;
            }
        }
    }
    Ok(())
}

pub struct BatchSampler {
    ops: Vec<Op>,
    pub num_qubits: usize,
    pub num_measurements: usize,
    pub num_detectors: usize,
    pub num_observables: usize,
    ring_mask: usize,
}

/// One batch of 64 shots: word d holds detector d for every shot, bit k being shot k.
pub struct Batch {
    pub detectors: Vec<u64>,
    pub observables: Vec<u64>,
}

impl Batch {
    /// The detectors shot `lane` fired, in order.
    pub fn lane_defects(&self, lane: usize) -> Vec<u32> {
        let bit = 1u64 << lane;
        (0..self.detectors.len()).filter(|&d| self.detectors[d] & bit != 0).map(|d| d as u32).collect()
    }

    /// The observables shot `lane` flipped, as a mask.
    pub fn lane_observables(&self, lane: usize) -> u64 {
        let mut out = 0u64;
        for (k, &w) in self.observables.iter().enumerate() {
            out |= ((w >> lane) & 1) << k;
        }
        out
    }

    /// The first `lanes` shots appended as Stim b8 rows: detectors to `dets`,
    /// observables to `obs`, each row padded to whole bytes.
    pub fn write_b8(&self, lanes: usize, dets: &mut Vec<u8>, obs: &mut Vec<u8>) {
        let (ds, os) = (self.detectors.len().div_ceil(8), self.observables.len().div_ceil(8));
        let (d0, o0) = (dets.len(), obs.len());
        dets.resize(d0 + lanes * ds, 0);
        obs.resize(o0 + lanes * os, 0);
        for (rows, start, stride, words) in [(&mut *dets, d0, ds, &self.detectors), (&mut *obs, o0, os, &self.observables)] {
            for (k, &w) in words.iter().enumerate() {
                let mut w = w;
                while w != 0 {
                    let lane = w.trailing_zeros() as usize;
                    w &= w - 1;
                    if lane < lanes {
                        rows[start + lane * stride + k / 8] |= 1 << (k % 8);
                    }
                }
            }
        }
    }

    /// Every shot's defects at once, reading only the set bits.
    pub fn all_defects(&self) -> Vec<Vec<u32>> {
        let mut out = vec![Vec::new(); 64];
        for (d, &w) in self.detectors.iter().enumerate() {
            let mut w = w;
            while w != 0 {
                out[w.trailing_zeros() as usize].push(d as u32);
                w &= w - 1;
            }
        }
        out
    }
}

struct State {
    x: Vec<u64>,
    z: Vec<u64>,
    ring: Vec<u64>,
    mask: usize,
    m: usize,
    det: usize,
    obs: Vec<u64>,
    shift: Vec<f64>,
    /// The lanes where the current chain of correlated errors has fired.
    chain: u64,
}

impl State {
    fn reset(&mut self, rng: &mut Xorshift, basis: Basis, q: usize) {
        match basis {
            Basis::Z => {
                self.x[q] = 0;
                self.z[q] = rng.next_u64();
            }
            Basis::X => {
                self.z[q] = 0;
                self.x[q] = rng.next_u64();
            }
        }
    }

    #[inline]
    fn pauli_on_lane(&mut self, q: usize, lane: u32, pauli: u64) {
        let bit = 1u64 << lane;
        if pauli & 1 != 0 {
            self.x[q] ^= bit;
        }
        if pauli & 2 != 0 {
            self.z[q] ^= bit;
        }
    }
}

fn exec(ops: &[Op], st: &mut State, rng: &mut Xorshift, sink: &mut dyn FnMut(usize, f64, u64)) {
    for op in ops {
        match op {
            Op::Reset { basis, qubits } => {
                for &q in qubits {
                    st.reset(rng, *basis, q as usize);
                }
            }
            Op::H(qubits) => {
                for &q in qubits {
                    let q = q as usize;
                    std::mem::swap(&mut st.x[q], &mut st.z[q]);
                }
            }
            Op::Cx(pairs) => {
                for &(c, t) in pairs {
                    let (c, t) = (c as usize, t as usize);
                    st.x[t] ^= st.x[c];
                    st.z[c] ^= st.z[t];
                }
            }
            Op::Cz(pairs) => {
                for &(a, b) in pairs {
                    let (a, b) = (a as usize, b as usize);
                    st.z[b] ^= st.x[a];
                    st.z[a] ^= st.x[b];
                }
            }
            Op::Measure { basis, reset, flip, qubits } => {
                for &q in qubits {
                    let q = q as usize;
                    let rec = match basis {
                        Basis::Z => st.x[q],
                        Basis::X => st.z[q],
                    } ^ bernoulli(rng, *flip);
                    st.ring[st.m & st.mask] = rec;
                    st.m += 1;
                    match basis {
                        Basis::Z => st.z[q] = rng.next_u64(),
                        Basis::X => st.x[q] = rng.next_u64(),
                    }
                    if *reset {
                        st.reset(rng, *basis, q);
                    }
                }
            }
            Op::PauliError { pauli, noise, qubits } => {
                for &q in qubits {
                    let w = bernoulli(rng, *noise);
                    if pauli & 1 != 0 {
                        st.x[q as usize] ^= w;
                    }
                    if pauli & 2 != 0 {
                        st.z[q as usize] ^= w;
                    }
                }
            }
            Op::Depolarize1 { noise, qubits } => {
                for &q in qubits {
                    let mut w = bernoulli(rng, *noise);
                    while w != 0 {
                        let lane = w.trailing_zeros();
                        w &= w - 1;
                        st.pauli_on_lane(q as usize, lane, 1 + rng.next_u64() % 3);
                    }
                }
            }
            Op::Depolarize2 { noise, pairs } => {
                for &(a, b) in pairs {
                    let mut w = bernoulli(rng, *noise);
                    while w != 0 {
                        let lane = w.trailing_zeros();
                        w &= w - 1;
                        let r = 1 + rng.next_u64() % 15;
                        st.pauli_on_lane(a as usize, lane, r & 3);
                        st.pauli_on_lane(b as usize, lane, r >> 2);
                    }
                }
            }
            Op::PauliChannel1 { any, px, pxy, total, qubits } => {
                for &q in qubits {
                    let mut w = bernoulli(rng, *any);
                    while w != 0 {
                        let lane = w.trailing_zeros();
                        w &= w - 1;
                        let u = rng.next_f64() * total;
                        let pauli = if u < *px {
                            1
                        } else if u < *pxy {
                            3
                        } else {
                            2
                        };
                        st.pauli_on_lane(q as usize, lane, pauli);
                    }
                }
            }
            Op::Detector(recs, time) => {
                let mut w = 0u64;
                for &k in recs {
                    w ^= st.ring[(st.m - k as usize) & st.mask];
                }
                let t = time.map_or(f64::NAN, |(i, t)| t + st.shift.get(i).copied().unwrap_or(0.0));
                sink(st.det, t, w);
                st.det += 1;
            }
            Op::ShiftCoords(shift) => {
                if st.shift.len() < shift.len() {
                    st.shift.resize(shift.len(), 0.0);
                }
                for (a, b) in st.shift.iter_mut().zip(shift) {
                    *a += b;
                }
            }
            Op::Observable(i, recs, paulis) => {
                for &k in recs {
                    st.obs[*i as usize] ^= st.ring[(st.m - k as usize) & st.mask];
                }
                // An X target anticommutes with the frame's Z part, a Z target with its X part.
                for &(q, p) in paulis {
                    let q = q as usize;
                    if p & 1 != 0 {
                        st.obs[*i as usize] ^= st.z[q];
                    }
                    if p & 2 != 0 {
                        st.obs[*i as usize] ^= st.x[q];
                    }
                }
            }
            Op::Repeat(count, body) => {
                for _ in 0..*count {
                    exec(body, st, rng, sink);
                }
            }
            Op::S(qubits) => {
                for &q in qubits {
                    let q = q as usize;
                    st.z[q] ^= st.x[q];
                }
            }
            Op::Correlated { noise, paulis, chained } => {
                let mut w = bernoulli(rng, *noise);
                if *chained {
                    w &= !st.chain;
                    st.chain |= w;
                } else {
                    st.chain = w;
                }
                for &(q, pauli) in paulis {
                    if pauli & 1 != 0 {
                        st.x[q as usize] ^= w;
                    }
                    if pauli & 2 != 0 {
                        st.z[q as usize] ^= w;
                    }
                }
            }
            Op::PauliChannel2 { any, cumulative, pairs } => {
                let total = *cumulative.last().unwrap_or(&0.0);
                for &(a, b) in pairs {
                    let mut w = bernoulli(rng, *any);
                    while w != 0 {
                        let lane = w.trailing_zeros();
                        w &= w - 1;
                        let u = rng.next_f64() * total;
                        // Case k + 1 in Stim's order: first qubit's Pauli k / 4, second's k % 4,
                        // each I, X, Y, Z.
                        let k = cumulative.iter().position(|&c| u < c).unwrap_or(14) + 1;
                        const CODE: [u64; 4] = [0, 1, 3, 2];
                        st.pauli_on_lane(a as usize, lane, CODE[k / 4]);
                        st.pauli_on_lane(b as usize, lane, CODE[k % 4]);
                    }
                }
            }
            Op::Feedback { pauli, lookback, qubit } => {
                let w = st.ring[(st.m - *lookback as usize) & st.mask];
                if pauli & 1 != 0 {
                    st.x[*qubit as usize] ^= w;
                }
                if pauli & 2 != 0 {
                    st.z[*qubit as usize] ^= w;
                }
            }
            Op::Heralded { any, cumulative, qubits } => {
                for &q in qubits {
                    let w = bernoulli(rng, *any);
                    st.ring[st.m & st.mask] = w;
                    st.m += 1;
                    let mut lanes = w;
                    while lanes != 0 {
                        let lane = lanes.trailing_zeros();
                        lanes &= lanes - 1;
                        let u = rng.next_f64() * cumulative[3];
                        let k = cumulative.iter().position(|&c| u < c).unwrap_or(3);
                        st.pauli_on_lane(q as usize, lane, [0, 1, 3, 2][k]);
                    }
                }
            }
            Op::Pad { flip, count } => {
                for _ in 0..*count {
                    st.ring[st.m & st.mask] = bernoulli(rng, *flip);
                    st.m += 1;
                }
            }
        }
    }
}

impl BatchSampler {
    pub fn new(circuit: &Circuit) -> Result<BatchSampler, String> {
        let ops = compile(&circuit.instrs);
        let mut s = Shape::default();
        shape(&ops, &mut s)?;
        let ring = (s.lookback.max(1) as usize).next_power_of_two();
        Ok(BatchSampler {
            ops,
            num_qubits: s.qubits,
            num_measurements: usize::try_from(s.measurements).map_err(|_| "too many measurements for this machine")?,
            num_detectors: usize::try_from(s.detectors).map_err(|_| "too many detectors for this machine")?,
            num_observables: s.observables,
            ring_mask: ring - 1,
        })
    }

    /// Run 64 shots, handing each detector's word to `sink(index, word)` as it
    /// is evaluated; returns the observables' words.
    pub fn run(&self, rng: &mut Xorshift, sink: &mut dyn FnMut(usize, u64)) -> Vec<u64> {
        self.run_timed(rng, &mut |d, _, w| sink(d, w))
    }

    /// The same, with each detector's time coordinate (its last coordinate,
    /// shifts applied; NaN if it has none): how a stream is cut into rounds as
    /// it is made.
    pub fn run_timed(&self, rng: &mut Xorshift, sink: &mut dyn FnMut(usize, f64, u64)) -> Vec<u64> {
        let mut st = State {
            x: vec![0; self.num_qubits],
            // Every qubit starts in |0>, to which Z is invisible.
            z: (0..self.num_qubits).map(|_| rng.next_u64()).collect(),
            ring: vec![0; self.ring_mask + 1],
            mask: self.ring_mask,
            m: 0,
            det: 0,
            obs: vec![0; self.num_observables],
            shift: Vec::new(),
            chain: 0,
        };
        exec(&self.ops, &mut st, rng, sink);
        st.obs
    }

    /// `shots` shots as b8 rows (detectors, observables), batch `first + k` drawn from a stream
    /// of its own, seeded by `seed` and the batch's index alone: the shots do not depend on how
    /// many threads drew them, and a later call continues where an earlier one stopped.
    pub fn sample_seeded(&self, seed: u64, first: u64, shots: usize, threads: usize) -> (Vec<u8>, Vec<u8>) {
        let parts = crate::parallel::parallel(shots.div_ceil(64), threads, |range| {
            let (mut dets, mut obs) = (Vec::new(), Vec::new());
            for b in range {
                let mut rng = Xorshift::new(batch_seed(seed, first + b as u64));
                self.sample(&mut rng).write_b8((shots - b * 64).min(64), &mut dets, &mut obs);
            }
            vec![(dets, obs)]
        });
        let (mut dets, mut obs) = (Vec::new(), Vec::new());
        for (d, o) in parts {
            dets.extend(d);
            obs.extend(o);
        }
        (dets, obs)
    }

    /// 64 shots' detectors and observables.
    pub fn sample(&self, rng: &mut Xorshift) -> Batch {
        // Capped: a loop of a trillion detectors may still be run through `run`.
        let mut detectors = Vec::with_capacity(self.num_detectors.min(1 << 24));
        let observables = self.run(rng, &mut |_, w| detectors.push(w));
        Batch { detectors, observables }
    }
}

fn splitmix64(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// The seed of batch `b`'s stream, from the sampler's seed and the batch alone.
fn batch_seed(seed: u64, b: u64) -> u64 {
    splitmix64(seed ^ splitmix64(b ^ 0x632B_E59B_D9B4_E019))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame_sampler::FrameSampler;

    /// Every noise line made deterministic: each channel becomes X_ERROR,
    /// Y_ERROR or Z_ERROR on the same qubits at p = 0 or 1, and each
    /// measurement flip 0 or 1. Then every shot is the same shot, and the two
    /// samplers must agree on it exactly.
    fn deterministic(text: &str, rng: &mut Xorshift) -> String {
        let mut out = String::new();
        for line in text.lines() {
            let head = line.split(['(', ' ']).next().unwrap_or("");
            let b = u8::from(rng.next_u64().is_multiple_of(32));
            let args_end = line.find(')').map(|i| i + 1);
            let targets = args_end.map(|i| line[i..].trim()).unwrap_or("");
            match head {
                "DEPOLARIZE1" | "DEPOLARIZE2" | "X_ERROR" | "Y_ERROR" | "Z_ERROR" | "PAULI_CHANNEL_1" => {
                    let name = ["X_ERROR", "Y_ERROR", "Z_ERROR"][(rng.next_u64() % 3) as usize];
                    out.push_str(&format!("{name}({b}) {targets}\n"));
                }
                "M" | "MX" | "MR" | "MRX" | "MZ" if args_end.is_some() => {
                    out.push_str(&format!("{head}({b}) {targets}\n"));
                }
                _ => {
                    out.push_str(line);
                    out.push('\n');
                }
            }
        }
        out
    }

    #[test]
    fn matches_frame_sampler_on_deterministic_noise() {
        use crate::memory::{generate, CodeKind, NoiseModel};
        let mut rng = Xorshift::new(12);
        let mut compared = 0;
        for kind in [CodeKind::Rotated, CodeKind::Xzzx] {
            for basis in [Basis::Z, Basis::X] {
                for d in [3usize, 5] {
                    let text = generate(kind, d, d, NoiseModel::Sd6 { p: 0.01 }, basis).unwrap().to_stim();
                    for _ in 0..20 {
                        let c = Circuit::parse(&deterministic(&text, &mut rng)).unwrap();
                        let shot = FrameSampler::new(&c).unwrap().sample(&mut rng);
                        let want: Vec<u32> =
                            shot.detectors.iter().enumerate().filter(|x| *x.1).map(|x| x.0 as u32).collect();
                        let batch = BatchSampler::new(&c).unwrap().sample(&mut rng);
                        for lane in [0, 17, 63] {
                            assert_eq!(batch.lane_defects(lane), want, "{kind:?} {basis:?} d = {d}");
                            assert_eq!(batch.lane_observables(lane), shot.observables);
                        }
                        compared += usize::from(!want.is_empty());
                    }
                }
            }
        }
        assert!(compared > 100, "{compared} rewrites fired anything");
    }

    #[test]
    fn repeat_blocks_run_without_flattening() {
        let stim = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/data/xcheck/stim-rotated-memory-z-d3.stim.txt"))
            .unwrap();
        let nested = "R 0 1\nREPEAT 3 {\n    REPEAT 2 {\n        X_ERROR(0.2) 0\n        CX 0 1\n        DEPOLARIZE1(0.1) 1\n        M 1\n        DETECTOR rec[-1]\n    }\n    M 0\n    DETECTOR rec[-1] rec[-2]\n}\nOBSERVABLE_INCLUDE(0) rec[-1]\n";
        for text in [stim.as_str(), nested] {
            let c = Circuit::parse(text).unwrap();
            assert!(c.instrs.iter().any(|i| matches!(i, Instr::Repeat { .. })));
            let (a, b) = (BatchSampler::new(&c).unwrap(), BatchSampler::new(&c.flattened()).unwrap());
            assert_eq!((a.num_detectors, a.num_measurements), (b.num_detectors, b.num_measurements));
            assert_eq!(a.num_detectors, c.resolve().unwrap().detectors.len());
            let (x, y) = (a.sample(&mut Xorshift::new(9)), b.sample(&mut Xorshift::new(9)));
            assert_eq!(x.detectors, y.detectors);
            assert_eq!(x.observables, y.observables);
        }
    }

    #[test]
    fn a_million_rounds_cost_the_loop_body() {
        let c = Circuit::parse("R 0\nREPEAT 1000000 {\n    X_ERROR(0.001) 0\n    MR 0\n    DETECTOR rec[-1]\n}\n").unwrap();
        let s = BatchSampler::new(&c).unwrap();
        assert_eq!((s.num_detectors, s.ring_mask), (1_000_000, 0));
        let mut fired = 0u64;
        s.run(&mut Xorshift::new(1), &mut |_, w| fired += u64::from(w.count_ones()));
        // 64 million Bernoulli(0.001) draws.
        assert!((fired as f64 - 64_000.0).abs() < 5.0 * 64_000f64.sqrt(), "{fired}");
    }

    fn marginals(circuit: &Circuit, batches: usize) {
        use crate::dem::Dem;
        let dem = Dem::from_circuit(circuit).unwrap();
        let s = BatchSampler::new(circuit).unwrap();
        let mut rng = Xorshift::new(31);
        let mut counts = vec![0u64; s.num_detectors];
        let mut obs = 0u64;
        for _ in 0..batches {
            let b = s.sample(&mut rng);
            for (c, w) in counts.iter_mut().zip(&b.detectors) {
                *c += u64::from(w.count_ones());
            }
            obs += u64::from(b.observables.first().copied().unwrap_or(0).count_ones());
        }
        let n = (batches * 64) as f64;
        let predict = |pick: &dyn Fn(&crate::dem::Mechanism) -> bool| -> f64 {
            let prod: f64 = dem.mechanisms.iter().filter(|m| pick(m)).map(|m| 1.0 - 2.0 * m.p).product();
            (1.0 - prod) / 2.0
        };
        for (d, &c) in counts.iter().enumerate() {
            let q = predict(&|m| m.detectors.contains(&(d as u32)));
            let rate = c as f64 / n;
            assert!((rate - q).abs() < 5.0 * (q * (1.0 - q) / n).sqrt().max(1e-9), "D{d}: sampled {rate}, model {q}");
        }
        let q = predict(&|m| m.observables & 1 == 1);
        assert!((obs as f64 / n - q).abs() < 5.0 * (q * (1.0 - q) / n).sqrt().max(1e-9), "L0");
    }

    #[test]
    fn marginals_match_the_error_model() {
        use crate::memory::{generate, CodeKind, NoiseModel};
        marginals(&Circuit::parse(&crate::fixtures::REP3.replace("0.01", "0.05")).unwrap(), 3125);
        marginals(&generate(CodeKind::Rotated, 3, 3, NoiseModel::Sd6 { p: 0.01 }, Basis::Z).unwrap(), 3125);
    }

    #[test]
    fn noiseless_circuits_never_fire() {
        let c = Circuit::parse(&crate::fixtures::REP3.replace("0.01", "0")).unwrap();
        let s = BatchSampler::new(&c).unwrap();
        let mut rng = Xorshift::new(3);
        for _ in 0..100 {
            let b = s.sample(&mut rng);
            assert!(b.detectors.iter().all(|&w| w == 0));
            assert!(b.observables.iter().all(|&w| w == 0));
        }
    }

    #[test]
    fn bernoulli_words_have_the_right_rate() {
        let mut rng = Xorshift::new(77);
        let words = 200_000usize;
        for &p in &[0.001, 0.1, 0.3, 0.9] {
            let n = Noise::new(p);
            let mut lanes = [0u64; 64];
            for _ in 0..words {
                let w = bernoulli(&mut rng, n);
                for (k, l) in lanes.iter_mut().enumerate() {
                    *l += (w >> k) & 1;
                }
            }
            let total: u64 = lanes.iter().sum();
            let n_all = (words * 64) as f64;
            let rate = total as f64 / n_all;
            assert!((rate - p).abs() < 5.0 * (p * (1.0 - p) / n_all).sqrt(), "p = {p}: {rate}");
            let per = words as f64;
            for (k, &l) in lanes.iter().enumerate() {
                assert!((l as f64 / per - p).abs() < 5.0 * (p * (1.0 - p) / per).sqrt(), "p = {p}, lane {k}");
            }
        }
        assert_eq!(bernoulli(&mut rng, Noise::new(0.0)), 0);
        assert_eq!(bernoulli(&mut rng, Noise::new(1.0)), !0);
    }

    #[test]
    fn vanishing_probabilities_never_fire() {
        let mut rng = Xorshift::new(4);
        for &p in &[1e-20, 1e-300, f64::MIN_POSITIVE] {
            for _ in 0..10_000 {
                assert_eq!(bernoulli(&mut rng, Noise::new(p)), 0, "p = {p}");
            }
        }
        // Small but resolvable: the rate is still right.
        let n = Noise::new(1e-7);
        let fired: u32 = (0..2_000_000).map(|_| bernoulli(&mut rng, n).count_ones()).sum();
        assert!(fired < 40, "{fired} of 128e6 at p = 1e-7");
    }

    #[test]
    fn a_loop_that_never_runs_is_not_checked() {
        assert!(BatchSampler::new(&Circuit::parse("REPEAT 0 {\n DETECTOR rec[-1]\n}\nM 0").unwrap()).is_ok());
    }

    #[test]
    fn lookbacks_before_the_first_measurement_are_errors() {
        assert!(BatchSampler::new(&Circuit::parse("M 0\nDETECTOR rec[-2]").unwrap()).is_err());
        assert!(BatchSampler::new(&Circuit::parse("REPEAT 3 {\n M 0\n DETECTOR rec[-2]\n}").unwrap()).is_err());
        assert!(BatchSampler::new(&Circuit::parse("M 0\nREPEAT 3 {\n M 0\n DETECTOR rec[-2]\n}").unwrap()).is_ok());
        // Counts that overflow are errors, not wrapped numbers.
        let nested = "REPEAT 4000000000 {\n REPEAT 4000000000 {\n REPEAT 4000000000 {\n M 0\n }\n }\n}";
        assert!(BatchSampler::new(&Circuit::parse(nested).unwrap()).err().unwrap_or_default().contains("overflow"));
    }

    #[test]
    #[ignore] // timing, for the README
    fn batch_timing() {
        use crate::memory::{generate, CodeKind, NoiseModel};
        for d in [3usize, 5, 7, 9] {
            let c = generate(CodeKind::Rotated, d, d, NoiseModel::Sd6 { p: 0.003 }, Basis::Z).unwrap();
            let (old, new) = (FrameSampler::new(&c).unwrap(), BatchSampler::new(&c).unwrap());
            let mut rng = Xorshift::new(1);
            let shots = 20_000usize;
            let t = std::time::Instant::now();
            for _ in 0..shots {
                std::hint::black_box(old.sample(&mut rng));
            }
            let old_us = t.elapsed().as_secs_f64() * 1e6 / shots as f64;
            let t = std::time::Instant::now();
            for _ in 0..shots / 64 {
                std::hint::black_box(new.sample(&mut rng));
            }
            let new_us = t.elapsed().as_secs_f64() * 1e6 / (shots / 64 * 64) as f64;
            println!("d = {d}: FrameSampler {old_us:.2} us/shot, batch {new_us:.3} us/shot, {:.0}x", old_us / new_us);
        }
    }

    fn sd6() -> BatchSampler {
        use crate::memory::{generate, CodeKind, NoiseModel};
        BatchSampler::new(&generate(CodeKind::Rotated, 3, 3, NoiseModel::Sd6 { p: 0.02 }, Basis::Z).unwrap()).unwrap()
    }

    #[test]
    fn seeded_shots_do_not_depend_on_threads() {
        let s = sd6();
        let one = s.sample_seeded(7, 0, 1000, 1);
        for threads in [2, 3, 8] {
            assert_eq!(s.sample_seeded(7, 0, 1000, threads), one, "{threads} threads");
        }
        assert_ne!(s.sample_seeded(8, 0, 1000, 1), one, "another seed draws other shots");
        assert_eq!(one.0.len(), 1000 * s.num_detectors.div_ceil(8));
    }

    #[test]
    fn seeded_calls_continue_batch_by_batch() {
        let s = sd6();
        let whole = s.sample_seeded(3, 0, 128, 4);
        let (a, b) = (s.sample_seeded(3, 0, 64, 1), s.sample_seeded(3, 1, 64, 1));
        assert_eq!([a.0, b.0].concat(), whole.0);
        assert_eq!([a.1, b.1].concat(), whole.1);
    }
}

//! The lane sampler: the batch sampler's ops flattened into a stream of 32-bit words and run
//! 32 shots per word with every random draw an integer compared with a threshold, so a GPU
//! can run the very same stream (`gpu`) and give the very same bits. This CPU executor is the
//! reference the GPU is held to.
//!
//! WHY A SECOND SAMPLER
//! --------------------
//! `BatchSampler` draws its noise words by geometric skipping with f64 logarithms, which WGSL
//! has no f64 for and whose rounding no two GPUs repeat. Here a noise location draws one
//! 32-bit number per lane and compares it with a threshold — 32 draws where the word sampler
//! makes about one, which a GPU pays for in parallel across thousands of words. The frames,
//! the ring of records, detectors and observables are the word sampler's with u32 words, and
//! every choice a composite channel makes is an integer compare too. The two samplers draw
//! different numbers from the same seed (their rates agree, their shots do not); within this
//! one, the same seed gives the same bytes on any device, any thread count and any batching.
//!
//! THE PROGRAM
//! -----------
//! A `Vec<u32>`: an opcode, its operands, the next opcode. A probability is two words, a kind
//! (0 never, 1 always, 2 compare) and a threshold t: a lane fires when its draw u satisfies
//! u <= t, so P(fire) = (t + 1) / 2^32 = round(p · 2^32) / 2^32 (the generator's draws cover
//! every u32). A channel's case cutoffs are shares of 2^32 compared with `u < c`. REPEAT
//! carries its count (a u32) and its body's length; the GPU runs it with a loop stack
//! `MAX_REPEAT_DEPTH` deep, this executor recursively, and a program nested deeper is refused
//! rather than run differently on the two.

use crate::batch_sampler::{batch_seed, compile, splitmix64, BatchSampler, Noise, Op};
use crate::circuit::{Basis, Circuit};

pub const LANES: usize = 32;

/// How deep the GPU's loop stack goes (`sample.wgsl`'s `STACK`): the lane program's limit.
pub const MAX_REPEAT_DEPTH: usize = 16;

pub const OP_RESET: u32 = 1;
pub const OP_H: u32 = 2;
pub const OP_S: u32 = 3;
pub const OP_CX: u32 = 4;
pub const OP_CZ: u32 = 5;
pub const OP_MEASURE: u32 = 6;
pub const OP_PAULI_ERROR: u32 = 7;
pub const OP_DEPOLARIZE1: u32 = 8;
pub const OP_DEPOLARIZE2: u32 = 9;
pub const OP_PAULI_CHANNEL1: u32 = 10;
pub const OP_DETECTOR: u32 = 11;
pub const OP_OBSERVABLE: u32 = 12;
pub const OP_REPEAT: u32 = 13;
pub const OP_CORRELATED: u32 = 14;
pub const OP_PAULI_CHANNEL2: u32 = 15;
pub const OP_PAD: u32 = 16;
pub const OP_FEEDBACK: u32 = 17;
pub const OP_HERALDED: u32 = 18;

pub const PROB_NEVER: u32 = 0;
pub const PROB_ALWAYS: u32 = 1;
pub const PROB_COMPARE: u32 = 2;

/// A probability as the program carries it: (kind, threshold).
pub fn encode_probability(p: f64) -> (u32, u32) {
    if !(p > 0.0) {
        return (PROB_NEVER, 0);
    }
    if p >= 1.0 {
        return (PROB_ALWAYS, u32::MAX);
    }
    let scaled = (p * 4_294_967_296.0).round();
    if scaled <= 0.0 {
        (PROB_NEVER, 0)
    } else if scaled >= 4_294_967_296.0 {
        (PROB_ALWAYS, u32::MAX)
    } else {
        (PROB_COMPARE, scaled as u32 - 1)
    }
}

/// A share of the draws (0 ≤ share ≤ 1) as a cutoff for `u < cutoff`.
fn cutoff(share: f64) -> u32 {
    if !(share > 0.0) {
        0
    } else {
        (share * 4_294_967_296.0).round().min(4_294_967_295.0) as u32
    }
}

fn noise(code: &mut Vec<u32>, n: &Noise) {
    let (kind, thr) = encode_probability(n.p);
    code.push(kind);
    code.push(thr);
}

fn emit(ops: &[Op], code: &mut Vec<u32>) {
    for op in ops {
        match op {
            Op::Reset { basis, qubits } => {
                code.extend([OP_RESET, basis_code(*basis), qubits.len() as u32]);
                code.extend(qubits.iter().copied());
            }
            Op::H(qubits) => {
                code.extend([OP_H, qubits.len() as u32]);
                code.extend(qubits.iter().copied());
            }
            Op::S(qubits) => {
                code.extend([OP_S, qubits.len() as u32]);
                code.extend(qubits.iter().copied());
            }
            Op::Cx(pairs) => {
                code.extend([OP_CX, pairs.len() as u32]);
                code.extend(pairs.iter().flat_map(|&(a, b)| [a, b]));
            }
            Op::Cz(pairs) => {
                code.extend([OP_CZ, pairs.len() as u32]);
                code.extend(pairs.iter().flat_map(|&(a, b)| [a, b]));
            }
            Op::Measure { basis, reset, flip, qubits } => {
                code.extend([OP_MEASURE, basis_code(*basis), u32::from(*reset)]);
                noise(code, flip);
                code.push(qubits.len() as u32);
                code.extend(qubits.iter().copied());
            }
            Op::PauliError { pauli, noise: n, qubits } => {
                code.extend([OP_PAULI_ERROR, u32::from(*pauli)]);
                noise(code, n);
                code.push(qubits.len() as u32);
                code.extend(qubits.iter().copied());
            }
            Op::Depolarize1 { noise: n, qubits } => {
                code.push(OP_DEPOLARIZE1);
                noise(code, n);
                code.push(qubits.len() as u32);
                code.extend(qubits.iter().copied());
            }
            Op::Depolarize2 { noise: n, pairs } => {
                code.push(OP_DEPOLARIZE2);
                noise(code, n);
                code.push(pairs.len() as u32);
                code.extend(pairs.iter().flat_map(|&(a, b)| [a, b]));
            }
            Op::PauliChannel1 { any, px, pxy, total, qubits } => {
                code.push(OP_PAULI_CHANNEL1);
                noise(code, any);
                code.extend([cutoff(px / total), cutoff(pxy / total), qubits.len() as u32]);
                code.extend(qubits.iter().copied());
            }
            Op::Detector(recs, _) => {
                code.extend([OP_DETECTOR, recs.len() as u32]);
                code.extend(recs.iter().copied());
            }
            Op::ShiftCoords(_) => {}
            Op::Observable(i, recs, paulis) => {
                code.extend([OP_OBSERVABLE, *i, recs.len() as u32]);
                code.extend(recs.iter().copied());
                code.push(paulis.len() as u32);
                code.extend(paulis.iter().flat_map(|&(q, p)| [q, u32::from(p)]));
            }
            Op::Repeat(count, body) => {
                let mut inner = Vec::new();
                emit(body, &mut inner);
                code.extend([OP_REPEAT, *count as u32, inner.len() as u32]);
                code.extend(inner);
            }
            Op::Correlated { noise: n, paulis, chained } => {
                code.push(OP_CORRELATED);
                noise(code, n);
                code.extend([u32::from(*chained), paulis.len() as u32]);
                code.extend(paulis.iter().flat_map(|&(q, p)| [q, u32::from(p)]));
            }
            Op::PauliChannel2 { any, cumulative, pairs } => {
                code.push(OP_PAULI_CHANNEL2);
                noise(code, any);
                let total = *cumulative.last().unwrap_or(&0.0);
                for k in 0..15 {
                    code.push(cumulative.get(k).map_or(u32::MAX, |&c| if total > 0.0 { cutoff(c / total) } else { u32::MAX }));
                }
                code.push(pairs.len() as u32);
                code.extend(pairs.iter().flat_map(|&(a, b)| [a, b]));
            }
            Op::Pad { flip, count } => {
                code.push(OP_PAD);
                noise(code, flip);
                code.push(*count as u32);
            }
            Op::Feedback { pauli, lookback, qubit } => {
                code.extend([OP_FEEDBACK, u32::from(*pauli), *lookback, *qubit]);
            }
            Op::Heralded { any, cumulative, qubits } => {
                code.push(OP_HERALDED);
                noise(code, any);
                let total = cumulative[3];
                for k in 0..3 {
                    code.push(if total > 0.0 { cutoff(cumulative[k] / total) } else { u32::MAX });
                }
                code.push(qubits.len() as u32);
                code.extend(qubits.iter().copied());
            }
        }
    }
}

fn basis_code(b: Basis) -> u32 {
    match b {
        Basis::Z => 0,
        Basis::X => 1,
    }
}

/// What a GPU learned about pacing a program's dispatches, kept with the program so that a
/// later call starts where the last one left off: the dispatch size that worked, and one
/// unit's own latency (f64 bits; 0 while unknown).
#[derive(Debug, Default)]
pub struct PaceHint {
    pub next: std::sync::atomic::AtomicUsize,
    pub latency_bits: std::sync::atomic::AtomicU64,
}

impl Clone for PaceHint {
    fn clone(&self) -> PaceHint {
        use std::sync::atomic::Ordering::Relaxed;
        PaceHint { next: self.next.load(Relaxed).into(), latency_bits: self.latency_bits.load(Relaxed).into() }
    }
}

/// A circuit as a lane program.
#[derive(Clone, Debug)]
pub struct LaneProgram {
    pub code: Vec<u32>,
    pub num_qubits: usize,
    pub num_measurements: usize,
    pub num_detectors: usize,
    pub num_observables: usize,
    /// The ring of records: a power of two minus one.
    pub ring_mask: u32,
    /// The GPU's pacing, learned across calls.
    pub pace: PaceHint,
}

/// 32 shots' results: one word per detector, per observable and (when kept) per measurement.
pub struct LaneWord {
    pub detectors: Vec<u32>,
    pub observables: Vec<u32>,
    pub measurements: Vec<u32>,
}

/// xoshiro128** (Blackman and Vigna), one per word: 128 bits of state from the word's seed, so
/// that no two words of a run share a stream (a 32-bit state would pair words every 2^33), and
/// draws that cover every u32.
#[derive(Clone, Copy)]
pub struct Rng32(pub [u32; 4]);

impl Rng32 {
    /// The state splitmix64 makes of a 64-bit seed (never all zero).
    pub fn from_seed(seed: u64) -> Rng32 {
        let a = splitmix64(seed);
        let b = splitmix64(a ^ 0x9E37_79B9_7F4A_7C15);
        let mut s = [a as u32, (a >> 32) as u32, b as u32, (b >> 32) as u32];
        if s == [0; 4] {
            s[0] = 0x9E37_79B9;
        }
        Rng32(s)
    }

    #[inline]
    pub fn next(&mut self) -> u32 {
        let s = &mut self.0;
        let result = s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = s[1] << 9;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = s[3].rotate_left(11);
        result
    }
}

/// The state of word `w`'s stream, from the sampler's seed and the word alone.
pub fn word_seed(seed: u64, w: u64) -> [u32; 4] {
    Rng32::from_seed(batch_seed(seed, w ^ 0x4C61_6E65_5361_6D70)).0
}

/// The deepest REPEAT nesting and the largest REPEAT count of a program's ops.
fn repeat_extent(ops: &[Op]) -> (usize, u64) {
    let mut depth = 0;
    let mut count = 0;
    for op in ops {
        if let Op::Repeat(n, body) = op {
            let (d, c) = repeat_extent(body);
            depth = depth.max(d + 1);
            count = count.max(*n).max(c);
        }
    }
    (depth, count)
}

struct State {
    x: Vec<u32>,
    z: Vec<u32>,
    ring: Vec<u32>,
    mask: usize,
    m: usize,
    obs: Vec<u32>,
    chain: u32,
    dets: Vec<u32>,
    all: Option<Vec<u32>>,
}

#[inline]
fn bernoulli(rng: &mut Rng32, kind: u32, thr: u32) -> u32 {
    match kind {
        PROB_NEVER => 0,
        PROB_ALWAYS => u32::MAX,
        _ => {
            let mut w = 0u32;
            for lane in 0..LANES {
                if rng.next() <= thr {
                    w |= 1 << lane;
                }
            }
            w
        }
    }
}

impl State {
    #[inline]
    fn reset(&mut self, rng: &mut Rng32, basis: u32, q: usize) {
        if basis == 0 {
            self.x[q] = 0;
            self.z[q] = rng.next();
        } else {
            self.z[q] = 0;
            self.x[q] = rng.next();
        }
    }

    #[inline]
    fn pauli_on_lane(&mut self, q: usize, lane: u32, pauli: u32) {
        let bit = 1u32 << lane;
        if pauli & 1 != 0 {
            self.x[q] ^= bit;
        }
        if pauli & 2 != 0 {
            self.z[q] ^= bit;
        }
    }

    #[inline]
    fn record(&mut self, w: u32) {
        self.ring[self.m & self.mask] = w;
        if let Some(all) = self.all.as_mut() {
            all.push(w);
        }
        self.m += 1;
    }

    #[inline]
    fn lookback(&self, k: u32) -> u32 {
        self.ring[(self.m - k as usize) & self.mask]
    }
}

fn exec(code: &[u32], st: &mut State, rng: &mut Rng32) {
    let mut pc = 0usize;
    while pc < code.len() {
        let op = code[pc];
        pc += 1;
        match op {
            OP_RESET => {
                let (basis, n) = (code[pc], code[pc + 1] as usize);
                pc += 2;
                for &q in &code[pc..pc + n] {
                    st.reset(rng, basis, q as usize);
                }
                pc += n;
            }
            OP_H => {
                let n = code[pc] as usize;
                pc += 1;
                for &q in &code[pc..pc + n] {
                    st.x.swap(q as usize, q as usize); // no-op to keep borrowck simple
                    let q = q as usize;
                    let (x, z) = (st.x[q], st.z[q]);
                    st.x[q] = z;
                    st.z[q] = x;
                }
                pc += n;
            }
            OP_S => {
                let n = code[pc] as usize;
                pc += 1;
                for &q in &code[pc..pc + n] {
                    let q = q as usize;
                    st.z[q] ^= st.x[q];
                }
                pc += n;
            }
            OP_CX => {
                let n = code[pc] as usize;
                pc += 1;
                for k in 0..n {
                    let (c, t) = (code[pc + 2 * k] as usize, code[pc + 2 * k + 1] as usize);
                    st.x[t] ^= st.x[c];
                    st.z[c] ^= st.z[t];
                }
                pc += 2 * n;
            }
            OP_CZ => {
                let n = code[pc] as usize;
                pc += 1;
                for k in 0..n {
                    let (a, b) = (code[pc + 2 * k] as usize, code[pc + 2 * k + 1] as usize);
                    st.z[b] ^= st.x[a];
                    st.z[a] ^= st.x[b];
                }
                pc += 2 * n;
            }
            OP_MEASURE => {
                let (basis, reset, kind, thr, n) = (code[pc], code[pc + 1] != 0, code[pc + 2], code[pc + 3], code[pc + 4] as usize);
                pc += 5;
                for &q in &code[pc..pc + n] {
                    let q = q as usize;
                    let rec = if basis == 0 { st.x[q] } else { st.z[q] } ^ bernoulli(rng, kind, thr);
                    st.record(rec);
                    if basis == 0 {
                        st.z[q] = rng.next();
                    } else {
                        st.x[q] = rng.next();
                    }
                    if reset {
                        st.reset(rng, basis, q);
                    }
                }
                pc += n;
            }
            OP_PAULI_ERROR => {
                let (pauli, kind, thr, n) = (code[pc], code[pc + 1], code[pc + 2], code[pc + 3] as usize);
                pc += 4;
                for &q in &code[pc..pc + n] {
                    let w = bernoulli(rng, kind, thr);
                    if pauli & 1 != 0 {
                        st.x[q as usize] ^= w;
                    }
                    if pauli & 2 != 0 {
                        st.z[q as usize] ^= w;
                    }
                }
                pc += n;
            }
            OP_DEPOLARIZE1 => {
                let (kind, thr, n) = (code[pc], code[pc + 1], code[pc + 2] as usize);
                pc += 3;
                for &q in &code[pc..pc + n] {
                    let mut w = bernoulli(rng, kind, thr);
                    while w != 0 {
                        let lane = w.trailing_zeros();
                        w &= w - 1;
                        let pauli = 1 + rng.next() % 3;
                        st.pauli_on_lane(q as usize, lane, pauli);
                    }
                }
                pc += n;
            }
            OP_DEPOLARIZE2 => {
                let (kind, thr, n) = (code[pc], code[pc + 1], code[pc + 2] as usize);
                pc += 3;
                for k in 0..n {
                    let (a, b) = (code[pc + 2 * k] as usize, code[pc + 2 * k + 1] as usize);
                    let mut w = bernoulli(rng, kind, thr);
                    while w != 0 {
                        let lane = w.trailing_zeros();
                        w &= w - 1;
                        let r = 1 + rng.next() % 15;
                        st.pauli_on_lane(a, lane, r & 3);
                        st.pauli_on_lane(b, lane, r >> 2);
                    }
                }
                pc += 2 * n;
            }
            OP_PAULI_CHANNEL1 => {
                let (kind, thr, cx, cxy, n) = (code[pc], code[pc + 1], code[pc + 2], code[pc + 3], code[pc + 4] as usize);
                pc += 5;
                for &q in &code[pc..pc + n] {
                    let mut w = bernoulli(rng, kind, thr);
                    while w != 0 {
                        let lane = w.trailing_zeros();
                        w &= w - 1;
                        let u = rng.next();
                        let pauli = if u < cx {
                            1
                        } else if u < cxy {
                            3
                        } else {
                            2
                        };
                        st.pauli_on_lane(q as usize, lane, pauli);
                    }
                }
                pc += n;
            }
            OP_DETECTOR => {
                let n = code[pc] as usize;
                pc += 1;
                let mut w = 0u32;
                for &k in &code[pc..pc + n] {
                    w ^= st.lookback(k);
                }
                st.dets.push(w);
                pc += n;
            }
            OP_OBSERVABLE => {
                let (i, n) = (code[pc] as usize, code[pc + 1] as usize);
                pc += 2;
                let mut w = 0u32;
                for &k in &code[pc..pc + n] {
                    w ^= st.lookback(k);
                }
                pc += n;
                let m = code[pc] as usize;
                pc += 1;
                for k in 0..m {
                    let (q, p) = (code[pc + 2 * k] as usize, code[pc + 2 * k + 1]);
                    if p & 1 != 0 {
                        w ^= st.z[q];
                    }
                    if p & 2 != 0 {
                        w ^= st.x[q];
                    }
                }
                pc += 2 * m;
                st.obs[i] ^= w;
            }
            OP_REPEAT => {
                let (count, len) = (code[pc], code[pc + 1] as usize);
                pc += 2;
                let body = &code[pc..pc + len];
                for _ in 0..count {
                    exec(body, st, rng);
                }
                pc += len;
            }
            OP_CORRELATED => {
                let (kind, thr, chained, n) = (code[pc], code[pc + 1], code[pc + 2] != 0, code[pc + 3] as usize);
                pc += 4;
                let mut w = bernoulli(rng, kind, thr);
                if chained {
                    w &= !st.chain;
                    st.chain |= w;
                } else {
                    st.chain = w;
                }
                for k in 0..n {
                    let (q, p) = (code[pc + 2 * k] as usize, code[pc + 2 * k + 1]);
                    if p & 1 != 0 {
                        st.x[q] ^= w;
                    }
                    if p & 2 != 0 {
                        st.z[q] ^= w;
                    }
                }
                pc += 2 * n;
            }
            OP_PAULI_CHANNEL2 => {
                let (kind, thr) = (code[pc], code[pc + 1]);
                pc += 2;
                let cuts = &code[pc..pc + 15];
                pc += 15;
                let n = code[pc] as usize;
                pc += 1;
                const CODE: [u32; 4] = [0, 1, 3, 2];
                for k in 0..n {
                    let (a, b) = (code[pc + 2 * k] as usize, code[pc + 2 * k + 1] as usize);
                    let mut w = bernoulli(rng, kind, thr);
                    while w != 0 {
                        let lane = w.trailing_zeros();
                        w &= w - 1;
                        let u = rng.next();
                        let case = cuts.iter().position(|&c| u < c).unwrap_or(14) + 1;
                        st.pauli_on_lane(a, lane, CODE[case / 4]);
                        st.pauli_on_lane(b, lane, CODE[case % 4]);
                    }
                }
                pc += 2 * n;
            }
            OP_PAD => {
                let (kind, thr, count) = (code[pc], code[pc + 1], code[pc + 2]);
                pc += 3;
                for _ in 0..count {
                    let w = bernoulli(rng, kind, thr);
                    st.record(w);
                }
            }
            OP_FEEDBACK => {
                let (pauli, lookback, q) = (code[pc], code[pc + 1], code[pc + 2] as usize);
                pc += 3;
                let w = st.lookback(lookback);
                if pauli & 1 != 0 {
                    st.x[q] ^= w;
                }
                if pauli & 2 != 0 {
                    st.z[q] ^= w;
                }
            }
            OP_HERALDED => {
                let (kind, thr, c0, c1, c2, n) = (code[pc], code[pc + 1], code[pc + 2], code[pc + 3], code[pc + 4], code[pc + 5] as usize);
                pc += 6;
                for &q in &code[pc..pc + n] {
                    let w = bernoulli(rng, kind, thr);
                    st.record(w);
                    let mut lanes = w;
                    while lanes != 0 {
                        let lane = lanes.trailing_zeros();
                        lanes &= lanes - 1;
                        let u = rng.next();
                        let pauli = if u < c0 {
                            0
                        } else if u < c1 {
                            1
                        } else if u < c2 {
                            3
                        } else {
                            2
                        };
                        st.pauli_on_lane(q as usize, lane, pauli);
                    }
                }
                pc += n;
            }
            other => unreachable!("lane program: unknown opcode {other} at {}", pc - 1),
        }
    }
}

impl LaneProgram {
    pub fn new(circuit: &Circuit) -> Result<LaneProgram, String> {
        let sampler = BatchSampler::new(circuit)?;
        Self::from_sampler(&sampler)
    }

    pub fn from_sampler(sampler: &BatchSampler) -> Result<LaneProgram, String> {
        let (depth, count) = repeat_extent(&sampler.ops);
        if depth > MAX_REPEAT_DEPTH {
            return Err(format!("the circuit nests REPEAT blocks {depth} deep; the lane sampler runs them {MAX_REPEAT_DEPTH} deep: flatten the circuit (`flattened()`) or sample on the cpu device"));
        }
        if count > u64::from(u32::MAX) {
            return Err(format!("a REPEAT block runs {count} times; the lane sampler counts to 2^32 - 1: split the block or sample on the cpu device"));
        }
        let mut code = Vec::new();
        emit(&sampler.ops, &mut code);
        if code.len() > (1 << 30) {
            return Err("the circuit's lane program is too long (over 2^30 words)".into());
        }
        Ok(LaneProgram {
            code,
            pace: PaceHint::default(),
            num_qubits: sampler.num_qubits,
            num_measurements: sampler.num_measurements,
            num_detectors: sampler.num_detectors,
            num_observables: sampler.num_observables,
            ring_mask: u32::try_from(sampler.ring_mask).map_err(|_| "the circuit's lookbacks are too long for the lane sampler")?,
        })
    }

    /// The program straight from instructions (a piece of a circuit, for the tests).
    pub fn from_instrs(instrs: &[crate::circuit::Instr], ring_mask: u32) -> LaneProgram {
        let ops = compile(instrs);
        let mut code = Vec::new();
        emit(&ops, &mut code);
        LaneProgram { code, num_qubits: 0, num_measurements: 0, num_detectors: 0, num_observables: 0, ring_mask, pace: PaceHint::default() }
    }

    /// 32 shots from a word's stream.
    pub fn run_word(&self, seed: [u32; 4], keep_measurements: bool) -> LaneWord {
        let mut rng = Rng32(seed);
        let mut st = State {
            x: vec![0; self.num_qubits],
            // Every qubit starts in |0>, to which Z is invisible.
            z: (0..self.num_qubits).map(|_| rng.next()).collect(),
            ring: vec![0; self.ring_mask as usize + 1],
            mask: self.ring_mask as usize,
            m: 0,
            obs: vec![0; self.num_observables],
            chain: 0,
            dets: Vec::with_capacity(self.num_detectors.min(1 << 24)),
            all: keep_measurements.then(|| Vec::with_capacity(self.num_measurements.min(1 << 24))),
        };
        exec(&self.code, &mut st, &mut rng);
        LaneWord { detectors: st.dets, observables: st.obs, measurements: st.all.unwrap_or_default() }
    }

    /// `shots` shots from word `first` on as b8 rows of detectors and of observables, `threads`
    /// wide; the same bytes however many threads, and a later call continues where an earlier
    /// one stopped.
    pub fn sample_seeded(&self, seed: u64, first: u64, shots: usize, threads: usize) -> (Vec<u8>, Vec<u8>) {
        let (ds, os) = (self.num_detectors.div_ceil(8), self.num_observables.div_ceil(8));
        let parts = crate::parallel::parallel(shots.div_ceil(LANES), threads, |range| {
            let (mut dets, mut obs) = (Vec::new(), Vec::new());
            for w in range {
                let word = self.run_word(word_seed(seed, first + w as u64), false);
                let lanes = (shots - w * LANES).min(LANES);
                write_b8(&word.detectors, lanes, ds, &mut dets);
                write_b8(&word.observables, lanes, os, &mut obs);
            }
            vec![(dets, obs)]
        });
        let mut dets = Vec::with_capacity(shots * ds);
        let mut obs = Vec::with_capacity(shots * os);
        for (d, o) in parts {
            dets.extend(d);
            obs.extend(o);
        }
        (dets, obs)
    }

    /// `shots` measurement records as b8 rows: the noiseless `reference` with each shot's
    /// flips (every flip as is when the reference is empty).
    pub fn sample_measurements_seeded(&self, seed: u64, first: u64, shots: usize, threads: usize, reference: &[bool]) -> Vec<u8> {
        let stride = self.num_measurements.div_ceil(8);
        let mut base = vec![0u8; stride];
        for (k, &b) in reference.iter().enumerate() {
            if b {
                base[k / 8] |= 1 << (k % 8);
            }
        }
        let parts = crate::parallel::parallel(shots.div_ceil(LANES), threads, |range| {
            let mut rows = Vec::new();
            for w in range {
                let word = self.run_word(word_seed(seed, first + w as u64), true);
                let lanes = (shots - w * LANES).min(LANES);
                let start = rows.len();
                write_b8(&word.measurements, lanes, stride, &mut rows);
                for lane in 0..lanes {
                    for (k, &b) in base.iter().enumerate() {
                        rows[start + lane * stride + k] ^= b;
                    }
                }
            }
            vec![rows]
        });
        parts.into_iter().flatten().collect()
    }
}

/// The first `lanes` shots of `words` (bit k of word j is shot k's bit j) appended as b8 rows of
/// `stride` bytes.
pub(crate) fn write_b8(words: &[u32], lanes: usize, stride: usize, rows: &mut Vec<u8>) {
    let start = rows.len();
    rows.resize(start + lanes * stride, 0);
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

#[cfg(test)]
mod tests {
    use super::*;

    fn circuit(text: &str) -> Circuit {
        Circuit::parse(text).expect("the test circuit parses")
    }

    #[test]
    fn probabilities_round_trip() {
        assert_eq!(encode_probability(0.0), (PROB_NEVER, 0));
        assert_eq!(encode_probability(1.0), (PROB_ALWAYS, u32::MAX));
        assert_eq!(encode_probability(0.5), (PROB_COMPARE, (1u64 << 31) as u32 - 1));
        let (kind, thr) = encode_probability(1e-3);
        assert_eq!(kind, PROB_COMPARE);
        assert!(((thr as f64 + 1.0) / 4_294_967_296.0 - 1e-3).abs() < 1e-9);
        assert_eq!(encode_probability(1e-12), (PROB_NEVER, 0));
    }

    #[test]
    fn same_seed_same_bytes_any_threads_and_batching() {
        let c = circuit("R 0 1 2\nH 0\nCX 0 1\nDEPOLARIZE2(0.1) 0 1\nX_ERROR(0.05) 2\nM 0 1 2\nDETECTOR rec[-1]\nDETECTOR rec[-2] rec[-3]\nOBSERVABLE_INCLUDE(0) rec[-1]");
        let p = LaneProgram::new(&c).unwrap();
        let one = p.sample_seeded(7, 0, 1000, 1);
        let many = p.sample_seeded(7, 0, 1000, 8);
        assert_eq!(one, many);
        let (a, b) = (p.sample_seeded(7, 0, 320, 3), p.sample_seeded(7, 10, 680, 3));
        assert_eq!([a.0, b.0].concat(), one.0);
        assert_eq!([a.1, b.1].concat(), one.1);
    }

    #[test]
    fn rates_match_the_word_sampler() {
        let noise = crate::generated::Noise {
            after_clifford_depolarization: 0.01,
            before_round_data_depolarization: 0.0,
            before_measure_flip_probability: 0.01,
            after_reset_flip_probability: 0.0,
            after_reset_leakage: 0.0,
            after_clifford_leakage_and_relaxation: 0.0,
        };
        let c = Circuit::parse(&crate::generated::generate("surface_code:rotated_memory_z", 3, 3, &noise).unwrap()).unwrap();
        let lanes = LaneProgram::new(&c).unwrap();
        let words = BatchSampler::new(&c).unwrap();
        let n = 64_000usize;
        let (ld, _) = lanes.sample_seeded(1, 0, n, 4);
        let (wd, _) = words.sample_seeded(1, 0, n, 4);
        let stride = lanes.num_detectors.div_ceil(8);
        let count = |rows: &[u8]| rows.iter().map(|b| b.count_ones() as usize).sum::<usize>() as f64 / (n * lanes.num_detectors) as f64;
        let (a, b) = (count(&ld), count(&wd));
        assert_eq!(ld.len(), n * stride);
        assert!((a - b).abs() < 0.1 * b, "detector rates {a} vs {b}");
    }

    #[test]
    fn deterministic_circuits_agree_with_the_frame_sampler() {
        // Every channel at p = 1 and no random outcome: both samplers must give the same shot
        // (the lane sampler's random Pauli choices aside, which X_ERROR, Z_ERROR and flips do
        // not make).
        let text = "R 0 1\nCX 0 1\nX_ERROR(1) 0\nZ_ERROR(1) 1\nM 0 1\nDETECTOR rec[-1]\nDETECTOR rec[-2]\nX_ERROR(1) 1\nMR 1\nDETECTOR rec[-1]\nREPEAT 3 {\n    X_ERROR(1) 0\n    M 0\n    DETECTOR rec[-1] rec[-2]\n}\nOBSERVABLE_INCLUDE(0) rec[-1]";
        let c = circuit(text);
        let lanes = LaneProgram::new(&c).unwrap();
        let word = lanes.run_word(word_seed(5, 0), false);
        let frame = crate::frame_sampler::FrameSampler::new(&c).unwrap();
        let shot = frame.sample(&mut crate::surface_code::Xorshift::new(9));
        for (k, &w) in word.detectors.iter().enumerate() {
            assert!(w == 0 || w == u32::MAX, "detector {k} is not the same on every lane");
            assert_eq!(w != 0, shot.detectors[k], "detector {k}");
        }
    }
}

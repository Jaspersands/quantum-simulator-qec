//! One shot of the twirled circuit, as the frame sampler draws it, keeping what the weight
//! needs: which locations fired, whether each anticommutes with the faults before it, and the
//! noiseless-branch value of every outcome.
//!
//! Two frames run side by side. The combined frame is the frame sampler's, randomised where the
//! state is insensitive, so the records are exactly its (the same random stream gives the same
//! shot when no rotation fires). The fault frame holds the faults alone: Pauli noise and fired
//! locations, never randomised, and cleared on a reset (whatever it held there went with the old
//! qubit). The two differ by the randomisation, which is what takes the reference run to the
//! shot's noiseless branch.

use super::{Program, Step};
use crate::statevec::C;
use crate::circuit::Basis;

/// Where a shot's random choices come from: a random stream for the sampler, an enumeration of
/// every branch for the exact check. Each method is one of the frame sampler's draws.
pub(crate) trait Source {
    /// True with probability p (one uniform draw).
    fn bern(&mut self, p: f64) -> bool;
    /// The first k with u < p_0 + … + p_k for one uniform draw u; None past them all.
    fn pick(&mut self, probs: &[f64]) -> Option<usize>;
    /// With probability p, one of 0..n uniformly (a uniform draw, then a word mod n).
    fn index(&mut self, p: f64, n: u64) -> Option<u64>;
    /// A fair coin (one word's low bit).
    fn coin(&mut self) -> bool;
}

/// What one shot leaves for its weight and its outcome.
#[derive(Clone, Debug, Default)]
pub struct Shot {
    /// Each record's flip relative to the reference run.
    pub records: Vec<bool>,
    /// Which locations fired.
    pub fired: Vec<bool>,
    /// Whether each location's Pauli anticommutes with the faults before it.
    pub anti: Vec<bool>,
    /// Each outcome slot's value in the shot's noiseless branch.
    pub branch: Vec<bool>,
}

fn apply(x: &mut [bool], z: &mut [bool], q: usize, c: u8) {
    x[q] ^= c & 1 != 0;
    z[q] ^= c & 2 != 0;
}

impl Program {
    /// One shot, its choices from `src`.
    pub(crate) fn run_shot(&self, src: &mut dyn Source) -> Shot {
        let nq = self.num_qubits;
        let (mut cx, mut cz) = (vec![false; nq], vec![false; nq]);
        let (mut fx, mut fz) = (vec![false; nq], vec![false; nq]);
        for z in cz.iter_mut() {
            *z = src.coin();
        }
        let nl = self.locations.len();
        let mut shot = Shot { records: Vec::with_capacity(self.num_records), fired: vec![false; nl], anti: vec![false; nl], branch: self.reference.clone() };
        let mut chain = false;
        // Merged groups' toggles so far.
        let mut taus: Vec<Vec<C>> = vec![Vec::new(); self.merge_groups.len()];
        // A Pauli on both frames.
        macro_rules! both {
            ($q:expr, $c:expr) => {{
                let (q, c) = ($q, $c);
                apply(&mut cx, &mut cz, q, c);
                apply(&mut fx, &mut fz, q, c);
            }};
        }
        for step in &self.steps {
            match *step {
                Step::H(q) => {
                    std::mem::swap(&mut cx[q], &mut cz[q]);
                    std::mem::swap(&mut fx[q], &mut fz[q]);
                }
                Step::S(q) => {
                    cz[q] ^= cx[q];
                    fz[q] ^= fx[q];
                }
                Step::Pauli(..) => {}
                Step::Cx(c, t) => {
                    if cx[c] {
                        cx[t] ^= true;
                    }
                    if cz[t] {
                        cz[c] ^= true;
                    }
                    if fx[c] {
                        fx[t] ^= true;
                    }
                    if fz[t] {
                        fz[c] ^= true;
                    }
                }
                Step::Cz(a, b) => {
                    let (ca, cb, fa, fb) = (cx[a], cx[b], fx[a], fx[b]);
                    cz[b] ^= ca;
                    cz[a] ^= cb;
                    fz[b] ^= fa;
                    fz[a] ^= fb;
                }
                Step::Measure { q, basis, reset, flip, slot } => {
                    let (c, f) = match basis {
                        Basis::Z => (cx[q], fx[q]),
                        Basis::X => (cz[q], fz[q]),
                    };
                    let mut r = c;
                    if flip > 0.0 && src.bern(flip) {
                        r = !r;
                    }
                    shot.records.push(r);
                    // The randomisation is the combined frame less the faults.
                    shot.branch[slot] ^= c ^ f;
                    match basis {
                        Basis::Z => cz[q] = src.coin(),
                        Basis::X => cx[q] = src.coin(),
                    }
                    if reset {
                        match basis {
                            Basis::Z => {
                                cx[q] = false;
                                cz[q] = src.coin();
                            }
                            Basis::X => {
                                cz[q] = false;
                                cx[q] = src.coin();
                            }
                        }
                        fx[q] = false;
                        fz[q] = false;
                    }
                }
                Step::Reset { q, basis, slot } => {
                    let (c, f) = match basis {
                        Basis::Z => (cx[q], fx[q]),
                        Basis::X => (cz[q], fz[q]),
                    };
                    shot.branch[slot] ^= c ^ f;
                    match basis {
                        Basis::Z => {
                            cx[q] = false;
                            cz[q] = src.coin();
                        }
                        Basis::X => {
                            cz[q] = false;
                            cx[q] = src.coin();
                        }
                    }
                    fx[q] = false;
                    fz[q] = false;
                }
                Step::PauliError { q, pauli, p } => {
                    if src.bern(p) {
                        both!(q, pauli);
                    }
                }
                Step::Depolarize1 { q, p } => {
                    if let Some(k) = src.index(p, 3) {
                        both!(q, 1 + k as u8);
                    }
                }
                Step::PauliChannel1 { q, px, py, pz } => {
                    if let Some(k) = src.pick(&[px, py, pz]) {
                        both!(q, [1, 3, 2][k]);
                    }
                }
                Step::Depolarize2 { a, b, p } => {
                    if let Some(k) = src.index(p, 15) {
                        let r = 1 + k as u8;
                        both!(a, r & 3);
                        both!(b, r >> 2);
                    }
                }
                Step::PauliChannel2 { a, b, ref probs } => {
                    if let Some(k) = src.pick(probs) {
                        const CODE: [u8; 4] = [0, 1, 3, 2];
                        both!(a, CODE[(k + 1) / 4]);
                        both!(b, CODE[(k + 1) % 4]);
                    }
                }
                Step::Correlated { p, ref paulis, chained } => {
                    let fires = src.bern(p) && !(chained && chain);
                    chain = if chained { chain || fires } else { fires };
                    if fires {
                        for &(q, c) in paulis {
                            both!(q, c);
                        }
                    }
                }
                Step::Herald { q, probs, slot } => {
                    let case = src.pick(&probs);
                    shot.records.push(case.is_some());
                    let _ = slot;
                    if let Some(k) = case {
                        both!(q, [0, 1, 3, 2][k]);
                    }
                }
                Step::Pad { flip, .. } => {
                    let r = flip > 0.0 && src.bern(flip);
                    shot.records.push(r);
                }
                Step::Coherent(loc) => {
                    let l = &self.locations[loc];
                    shot.anti[loc] = l.paulis.iter().fold(false, |a, &(q, c)| a ^ (c & 1 != 0 && fz[q]) ^ (c & 2 != 0 && fx[q]));
                    let p = match self.merge.get(loc).copied().flatten() {
                        None => l.theta.sin().powi(2),
                        Some(m) => {
                            taus[m.group].push(self.unfired_toggle(loc, m.mu, shot.anti[loc]));
                            if m.last { Program::odd_proposal(&taus[m.group]) } else { 0.0 }
                        }
                    };
                    if p != 0.0 && src.bern(p) {
                        shot.fired[loc] = true;
                        for &(q, c) in &l.paulis {
                            both!(q, c);
                        }
                    }
                }
            }
        }
        for q in 0..nq {
            let s = self.end_slot(q);
            shot.branch[s] ^= cx[q] ^ fx[q];
        }
        shot
    }

    /// A shot's detection events and observable flips.
    pub fn outcome(&self, records: &[bool]) -> (Vec<bool>, u64) {
        let parity = |r: &Vec<usize>| r.iter().fold(false, |a, &m| a ^ records[m]);
        let dets = self.detectors.iter().map(parity).collect();
        let obs = self.observables.iter().enumerate().fold(0u64, |acc, (k, r)| acc | (parity(r) as u64) << k);
        (dets, obs)
    }
}

/// The frame sampler's draws from its generator, in its order.
pub(crate) struct Stream<'a>(pub &'a mut crate::surface_code::Xorshift);

impl Source for Stream<'_> {
    fn bern(&mut self, p: f64) -> bool {
        self.0.next_f64() < p
    }

    fn pick(&mut self, probs: &[f64]) -> Option<usize> {
        let u = self.0.next_f64();
        let mut acc = 0.0;
        probs.iter().position(|p| {
            acc += p;
            u < acc
        })
    }

    fn index(&mut self, p: f64, n: u64) -> Option<u64> {
        if self.0.next_f64() < p {
            Some(self.0.next_u64() % n)
        } else {
            None
        }
    }

    fn coin(&mut self) -> bool {
        self.0.next_u64() & 1 == 1
    }
}

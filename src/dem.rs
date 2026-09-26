//! Detector error models: which detectors and logical observables each
//! elementary fault flips, and how likely it is.
//!
//! BUILT BACKWARDS
//! ---------------
//! The obvious construction pushes every fault forward through the rest of the
//! circuit, which costs faults × instructions. This walks the circuit once, in
//! reverse, carrying for every qubit the set of detectors and observables an X
//! error there would flip (`sx`) and the set a Z error would flip (`sz`). At a
//! noise instruction the symptom of every Pauli it can apply is then read off
//! directly. That is how Stim's error analyzer works, and it is what lets a
//! d = 7 model be derived in the browser while the reader watches.
//!
//! Going backwards through a gate conjugates the pair: H swaps them; CX sends
//! `sx[c] ^= sx[t]` and `sz[t] ^= sz[c]`; CZ sends `sx[a] ^= sz[b]` and
//! `sx[b] ^= sz[a]`. A Z-basis measurement adds what reads its record to `sx`;
//! a reset clears both, since an error before a reset is erased by it.
//!
//! The same walk checks determinism for free. Reaching a Z-basis reset with
//! `sz` non-empty means some detector anticommutes with the state the reset
//! prepares, so its value is a coin flip. That is an error, not a warning.

use std::collections::HashMap;
use std::fmt::Write as _;

use crate::circuit::{fmt_args, split_instruction, Basis, Circuit, Instr};

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct Piece {
    pub detectors: Vec<u32>,
    pub observables: u64,
}

#[derive(Clone, Debug)]
pub struct Mechanism {
    pub p: f64,
    /// Sorted.
    pub detectors: Vec<u32>,
    pub observables: u64,
    /// Graph-like components (at most two detectors each) that XOR back to the
    /// mechanism. Empty for a hyperedge nobody decomposed. Faults that share a
    /// symptom but split differently are separate mechanisms, each with its own
    /// probability, as in Stim's decomposed models.
    pub pieces: Vec<Piece>,
}

#[derive(Clone, Debug, Default)]
pub struct Dem {
    pub num_detectors: usize,
    pub num_observables: usize,
    pub detector_coords: Vec<Vec<f64>>,
    pub mechanisms: Vec<Mechanism>,
}

/* -- Channel conversion ---------------------------------------------------- */

/// Independent X, Y and Z probability equivalent to `DEPOLARIZE1(p)`.
///
/// A Pauli channel is fixed by how much it shrinks each Pauli. Depolarizing
/// shrinks X, Y and Z alike by 1 − 4p/3; three independent channels of
/// probability q shrink each by (1 − 2q)², since each Pauli anticommutes with
/// two of the three. Equating the two is exact, which is why the model loses
/// nothing by treating the components as independent.
pub fn depolarize1_component(p: f64) -> f64 {
    0.5 - 0.5 * (1.0 - 4.0 * p / 3.0).sqrt()
}

/// The same for `DEPOLARIZE2(p)`: fifteen components, each non-identity Pauli
/// anticommuting with eight of them, so (1 − 2q)⁸ = 1 − 16p/15.
pub fn depolarize2_component(p: f64) -> f64 {
    0.5 - 0.5 * (1.0 - 16.0 * p / 15.0).powf(0.125)
}

/// Independent (qx, qy, qz) equivalent to `PAULI_CHANNEL_1(px, py, pz)`.
pub fn pauli_channel_1_independent(px: f64, py: f64, pz: f64) -> Result<(f64, f64, f64), String> {
    let lx = 1.0 - 2.0 * (py + pz);
    let ly = 1.0 - 2.0 * (px + pz);
    let lz = 1.0 - 2.0 * (px + py);
    if lx <= 0.0 || ly <= 0.0 || lz <= 0.0 {
        return Err(format!("PAULI_CHANNEL_1({px}, {py}, {pz}) has no equivalent set of independent errors"));
    }
    let q = |a: f64| 0.5 * (1.0 - a);
    Ok((q((ly * lz / lx).sqrt()), q((lx * lz / ly).sqrt()), q((lx * ly / lz).sqrt())))
}

/// Probability that exactly one of two independent events happens.
pub fn xor_prob(a: f64, b: f64) -> f64 {
    a * (1.0 - b) + b * (1.0 - a)
}

/* -- Symptom bitsets ------------------------------------------------------- */

/// Detectors occupy bits 0..nd; observable k sits at bit nd + k.
#[derive(Clone, Copy)]
struct Space {
    nd: usize,
    words: usize,
}

impl Space {
    fn new(nd: usize, no: usize) -> Self {
        Space { nd, words: (nd + no).div_ceil(64).max(1) }
    }

    fn zero(&self) -> Vec<u64> {
        vec![0; self.words]
    }

    fn split(&self, s: &[u64]) -> (Vec<u32>, u64) {
        let mut dets = Vec::new();
        let mut obs = 0u64;
        for (w, &word) in s.iter().enumerate() {
            let mut bits = word;
            while bits != 0 {
                let b = bits.trailing_zeros() as usize;
                bits &= bits - 1;
                let i = w * 64 + b;
                if i < self.nd {
                    dets.push(i as u32);
                } else {
                    obs |= 1u64 << (i - self.nd);
                }
            }
        }
        (dets, obs)
    }
}

fn toggle(s: &mut [u64], i: usize) {
    s[i / 64] ^= 1u64 << (i % 64);
}

fn xor_into(a: &mut [u64], b: &[u64]) {
    for (x, y) in a.iter_mut().zip(b) {
        *x ^= y;
    }
}

fn is_zero(a: &[u64]) -> bool {
    a.iter().all(|&w| w == 0)
}

/* -- The builder ----------------------------------------------------------- */

/// Where a fault came from, for error messages.
#[derive(Clone, Copy)]
struct Origin {
    instr: usize,
    name: &'static str,
    a: u32,
    b: Option<u32>,
    pauli: (u8, u8),
}

fn pauli_name(p: u8) -> &'static str {
    match p {
        0 => "I",
        1 => "X",
        2 => "Z",
        _ => "Y",
    }
}

impl Origin {
    fn describe(&self) -> String {
        match self.b {
            Some(b) => format!(
                "{} {}⊗{} on qubits {}, {} (instruction {})",
                self.name,
                pauli_name(self.pauli.0),
                pauli_name(self.pauli.1),
                self.a,
                b,
                self.instr
            ),
            None => format!("{} {} on qubit {} (instruction {})", self.name, pauli_name(self.pauli.0), self.a, self.instr),
        }
    }
}

struct Entry {
    sym: Vec<u64>,
    p: f64,
    /// Each distinct way a fault with this symptom was split into pieces, with
    /// its own probability. Faults can share a symptom without sharing a
    /// decomposition, and each contributes its own pieces to the matching graph,
    /// as they do in Stim's decomposed models.
    variants: Vec<(Vec<Vec<u64>>, f64)>,
    origin: Origin,
}

struct Builder {
    space: Space,
    index: HashMap<Vec<u64>, usize>,
    entries: Vec<Entry>,
}

impl Builder {
    /// Record a fault of probability `p` whose symptom is the XOR of `pieces`.
    fn add(&mut self, p: f64, pieces: Vec<Vec<u64>>, origin: Origin) {
        if p <= 0.0 {
            return;
        }
        let mut pieces: Vec<Vec<u64>> = pieces.into_iter().filter(|x| !is_zero(x)).collect();
        pieces.sort();
        let mut sym = self.space.zero();
        for x in &pieces {
            xor_into(&mut sym, x);
        }
        if is_zero(&sym) {
            return;
        }
        match self.index.get(&sym) {
            Some(&i) => {
                let e = &mut self.entries[i];
                e.p = xor_prob(e.p, p);
                match e.variants.iter_mut().find(|v| v.0 == pieces) {
                    Some(v) => v.1 = xor_prob(v.1, p),
                    None => e.variants.push((pieces, p)),
                }
            }
            None => {
                self.index.insert(sym.clone(), self.entries.len());
                self.entries.push(Entry { sym, p, variants: vec![(pieces, p)], origin });
            }
        }
    }
}

fn nondeterministic(space: &Space, coords: &[Vec<f64>], sym: &[u64], q: usize, what: &str) -> String {
    let (dets, obs) = space.split(sym);
    let subject = match dets.first() {
        Some(&d) => {
            let c = coords.get(d as usize).map(|c| fmt_args(c)).unwrap_or_default();
            format!("detector D{d} ({c})")
        }
        None => format!("observable L{}", obs.trailing_zeros()),
    };
    format!("{subject} is not deterministic: it depends on the random outcome of {what} on qubit {q}")
}

impl Dem {
    pub fn from_circuit(circuit: &Circuit) -> Result<Dem, String> {
        Dem::build(circuit, true)
    }

    /// The model without splitting faults into graph-like pieces: one
    /// mechanism per symptom, faults with the same symptom merged. What BP and
    /// BP+OSD decode, and the only model a code whose faults fire three or
    /// more checks has (a fault of one or two detectors keeps itself as its
    /// one piece).
    pub fn from_circuit_undecomposed(circuit: &Circuit) -> Result<Dem, String> {
        Dem::build(circuit, false)
    }

    fn build(circuit: &Circuit, decompose: bool) -> Result<Dem, String> {
        let res = circuit.resolve()?;
        let nd = res.detectors.len();
        let no = res.observables.len();
        let space = Space::new(nd, no);

        // Which detectors and observables read each measurement record.
        let mut rec_sym = vec![space.zero(); res.num_measurements];
        for (d, recs) in res.detectors.iter().enumerate() {
            for &m in recs {
                toggle(&mut rec_sym[m], d);
            }
        }
        for (o, recs) in res.observables.iter().enumerate() {
            for &m in recs {
                toggle(&mut rec_sym[m], nd + o);
            }
        }

        let nq = res.num_qubits;
        let mut sx = vec![space.zero(); nq];
        let mut sz = vec![space.zero(); nq];
        let mut b = Builder { space, index: HashMap::new(), entries: Vec::new() };
        let mut m = res.num_measurements;
        let coords = &res.detector_coords;

        let check_reset = |sx: &[Vec<u64>], sz: &[Vec<u64>], q: usize, basis: Basis, what: &str| -> Result<(), String> {
            let sensitive = match basis {
                Basis::Z => &sz[q],
                Basis::X => &sx[q],
            };
            if is_zero(sensitive) {
                Ok(())
            } else {
                Err(nondeterministic(&space, coords, sensitive, q, what))
            }
        };
        let reset_name = |basis: Basis| if basis == Basis::Z { "a Z-basis reset" } else { "an X-basis reset" };

        for (idx, ins) in res.instrs.iter().enumerate().rev() {
            match ins {
                Instr::Reset { basis, qubits } => {
                    for &q in qubits.iter().rev() {
                        let q = q as usize;
                        check_reset(&sx, &sz, q, *basis, reset_name(*basis))?;
                        sx[q].fill(0);
                        sz[q].fill(0);
                    }
                }
                Instr::Measure { basis, reset, flip, qubits } => {
                    for &q in qubits.iter().rev() {
                        let qi = q as usize;
                        m -= 1;
                        if *reset {
                            check_reset(&sx, &sz, qi, *basis, reset_name(*basis))?;
                            sx[qi].fill(0);
                            sz[qi].fill(0);
                        }
                        match basis {
                            Basis::Z => xor_into(&mut sx[qi], &rec_sym[m]),
                            Basis::X => xor_into(&mut sz[qi], &rec_sym[m]),
                        }
                        if *flip > 0.0 {
                            let origin = Origin { instr: idx, name: "measurement flip", a: q, b: None, pauli: (1, 0) };
                            b.add(*flip, vec![rec_sym[m].clone()], origin);
                        }
                    }
                }
                Instr::H(qubits) => {
                    for &q in qubits.iter().rev() {
                        std::mem::swap(&mut sx[q as usize], &mut sz[q as usize]);
                    }
                }
                Instr::Cx(pairs) => {
                    for &(c, t) in pairs.iter().rev() {
                        let (c, t) = (c as usize, t as usize);
                        let from_t = sx[t].clone();
                        xor_into(&mut sx[c], &from_t);
                        let from_c = sz[c].clone();
                        xor_into(&mut sz[t], &from_c);
                    }
                }
                Instr::Cz(pairs) => {
                    for &(a, bq) in pairs.iter().rev() {
                        let (a, bq) = (a as usize, bq as usize);
                        xor_into(&mut sx[a], &sz[bq]);
                        xor_into(&mut sx[bq], &sz[a]);
                    }
                }
                // X_ERROR, Y_ERROR and Z_ERROR are single faults and stay whole
                // here, a Y included, as Stim leaves them; anything wider than a
                // pair is split by the global pass below.
                Instr::PauliError { pauli, p, qubits } => {
                    for &q in qubits {
                        let q = q as usize;
                        let mut sym = space.zero();
                        if pauli & 1 != 0 {
                            xor_into(&mut sym, &sx[q]);
                        }
                        if pauli & 2 != 0 {
                            xor_into(&mut sym, &sz[q]);
                        }
                        let origin = Origin { instr: idx, name: "Pauli error", a: q as u32, b: None, pauli: (*pauli, 0) };
                        b.add(*p, vec![sym], origin);
                    }
                }
                // The composite channels are split combination by combination,
                // with Stim's basis order: Z then X for DEPOLARIZE1, X then Z for
                // PAULI_CHANNEL_1, and Z_a, X_a, Z_b, X_b for DEPOLARIZE2.
                Instr::Depolarize1 { p, qubits } => {
                    if *p > 0.75 {
                        return Err(format!("DEPOLARIZE1({p}) exceeds 3/4 (instruction {idx})"));
                    }
                    let q1 = depolarize1_component(*p);
                    for &q in qubits {
                        let qi = q as usize;
                        let combos = channel_combinations(&space, &[sz[qi].clone(), sx[qi].clone()]);
                        for (k, pieces) in combos.into_iter().enumerate() {
                            let pauli = [2u8, 1, 3][k];
                            let origin = Origin { instr: idx, name: "DEPOLARIZE1", a: q, b: None, pauli: (pauli, 0) };
                            b.add(q1, pieces, origin);
                        }
                    }
                }
                Instr::PauliChannel1 { px, py, pz, qubits } => {
                    let (qx, qy, qz) =
                        pauli_channel_1_independent(*px, *py, *pz).map_err(|e| format!("{e} (instruction {idx})"))?;
                    for &q in qubits {
                        let qi = q as usize;
                        let combos = channel_combinations(&space, &[sx[qi].clone(), sz[qi].clone()]);
                        for (k, pieces) in combos.into_iter().enumerate() {
                            let (pauli, prob) = [(1u8, qx), (2, qz), (3, qy)][k];
                            let origin = Origin { instr: idx, name: "PAULI_CHANNEL_1", a: q, b: None, pauli: (pauli, 0) };
                            b.add(prob, pieces, origin);
                        }
                    }
                }
                Instr::Depolarize2 { p, pairs } => {
                    if *p > 15.0 / 16.0 {
                        return Err(format!("DEPOLARIZE2({p}) exceeds 15/16 (instruction {idx})"));
                    }
                    let q2 = depolarize2_component(*p);
                    for &(qa, qb) in pairs {
                        let (a, bq) = (qa as usize, qb as usize);
                        let basis = [sz[a].clone(), sx[a].clone(), sz[bq].clone(), sx[bq].clone()];
                        for (i, pieces) in channel_combinations(&space, &basis).into_iter().enumerate() {
                            let k = i + 1;
                            let pa = ((k >> 1) & 1) as u8 | (((k & 1) as u8) << 1);
                            let pb = ((k >> 3) & 1) as u8 | ((((k >> 2) & 1) as u8) << 1);
                            let origin = Origin { instr: idx, name: "DEPOLARIZE2", a: qa, b: Some(qb), pauli: (pa, pb) };
                            b.add(q2, pieces, origin);
                        }
                    }
                }
                Instr::Detector { .. }
                | Instr::Observable { .. }
                | Instr::QubitCoords { .. }
                | Instr::ShiftCoords(_)
                | Instr::Tick
                | Instr::Pauli { .. }
                | Instr::SweepX(_)
                | Instr::Repeat { .. } => {}
            }
        }

        // Every qubit starts in |0>, which is a Z-basis reset at time zero.
        for q in 0..nq {
            check_reset(&sx, &sz, q, Basis::Z, "the initial |0>")?;
        }

        for e in &b.entries {
            let (dets, obs) = space.split(&e.sym);
            if dets.is_empty() {
                return Err(format!(
                    "{} flips observables {obs:#b} while firing no detector: an undetectable logical error",
                    e.origin.describe()
                ));
            }
        }

        if !decompose {
            let mut mechanisms: Vec<Mechanism> = b
                .entries
                .iter()
                .map(|e| {
                    let (detectors, observables) = space.split(&e.sym);
                    let pieces = if (1..=2).contains(&detectors.len()) {
                        vec![Piece { detectors: detectors.clone(), observables }]
                    } else {
                        Vec::new()
                    };
                    Mechanism { p: e.p, detectors, observables, pieces }
                })
                .collect();
            mechanisms.sort_by(|a, b| a.detectors.cmp(&b.detectors).then(a.observables.cmp(&b.observables)));
            return Ok(Dem { num_detectors: nd, num_observables: no, detector_coords: res.detector_coords, mechanisms });
        }

        // The global pass, as Stim runs it when the circuit is done: every
        // one- or two-detector piece is a known edge, and any fault still holding
        // a wider piece is rewritten into known edges.
        let mut known: HashMap<Vec<u32>, Vec<u64>> = HashMap::new();
        for e in &b.entries {
            for (pieces, _) in &e.variants {
                for x in pieces {
                    let (dets, _) = space.split(x);
                    if (1..=2).contains(&dets.len()) {
                        known.entry(dets).or_insert_with(|| x.clone());
                    }
                }
            }
        }

        let mut mechanisms = Vec::with_capacity(b.entries.len());
        for e in &b.entries {
            let (detectors, observables) = space.split(&e.sym);
            let mut grouped: Vec<(Vec<Piece>, f64)> = Vec::new();
            for (pieces, p) in &e.variants {
                let graphlike = pieces.iter().all(|x| (1..=2).contains(&space.split(x).0.len()));
                let rewritten: Vec<Vec<u64>> = if graphlike {
                    pieces.clone()
                } else {
                    let mut out = Vec::new();
                    for x in pieces {
                        let found = brute_force_known(&space, x, &known).or_else(|| greedy_known(&space, x, &known));
                        match found {
                            Some(mut parts) => out.append(&mut parts),
                            None => {
                                return Err(format!(
                                    "cannot split {} into graph-like pieces: it fires detectors {:?}",
                                    e.origin.describe(),
                                    detectors
                                ))
                            }
                        }
                    }
                    out
                };
                let mut final_pieces = Vec::with_capacity(rewritten.len());
                for x in &rewritten {
                    let (dets, obs) = space.split(x);
                    if dets.is_empty() || dets.len() > 2 {
                        return Err(format!(
                            "cannot split {} into graph-like pieces: a piece fires detectors {:?}",
                            e.origin.describe(),
                            dets
                        ));
                    }
                    final_pieces.push(Piece { detectors: dets, observables: obs });
                }
                final_pieces.sort();
                match grouped.iter_mut().find(|g| g.0 == final_pieces) {
                    Some(g) => g.1 = xor_prob(g.1, *p),
                    None => grouped.push((final_pieces, *p)),
                }
            }
            for (pieces, p) in grouped {
                mechanisms.push(Mechanism { p, detectors: detectors.clone(), observables, pieces });
            }
        }
        mechanisms.sort_by(|a, b| {
            a.detectors.cmp(&b.detectors).then(a.observables.cmp(&b.observables)).then(a.pieces.cmp(&b.pieces))
        });

        Ok(Dem { num_detectors: nd, num_observables: no, detector_coords: res.detector_coords, mechanisms })
    }
}

/* -- Decomposition --------------------------------------------------------- */
//
// Stim's decomposition, reproduced so that this engine's matching graph is the
// one PyMatching builds from Stim's model, edge for edge. A first attempt split
// each fault into its own X and Z halves; it disagreed with Stim on dozens of
// edges, and that was not cosmetic. Kept whole, a Y fault on a boundary qubit is
// an edge joining the X-check and Z-check graphs, and the matcher routes
// through such bridges: two single faults of the d = 3 memory-X circuit under
// SD6 decoded into logical errors, where PyMatching on Stim's model decoded
// none.

/// Symptom with the observable bits cleared.
fn det_mask(space: &Space, s: &[u64]) -> Vec<u64> {
    let mut out = s.to_vec();
    for (w, word) in out.iter_mut().enumerate() {
        let lo = w * 64;
        if lo >= space.nd {
            *word = 0;
        } else if lo + 64 > space.nd {
            *word &= (1u64 << (space.nd - lo)) - 1;
        }
    }
    out
}

fn popcount(s: &[u64]) -> u32 {
    s.iter().map(|w| w.count_ones()).sum()
}

fn subset(a: &[u64], b: &[u64]) -> bool {
    a.iter().zip(b).all(|(x, y)| x & !y == 0)
}

fn or(a: &[u64], b: &[u64]) -> Vec<u64> {
    a.iter().zip(b).map(|(x, y)| x | y).collect()
}

fn and_not(a: &[u64], b: &[u64]) -> Vec<u64> {
    a.iter().zip(b).map(|(x, y)| x & !y).collect()
}

/// Stim's `decompose_helper_add_error_combinations`. For a channel with basis
/// errors b_0..b_{s-1} (their symptoms), returns the pieces of every
/// combination k = 1..2^s, in order. A combination is split using only the
/// channel's own single-detector combinations and its irreducible
/// two-detector ones; one that cannot be is left whole for the global pass.
fn channel_combinations(space: &Space, basis: &[Vec<u64>]) -> Vec<Vec<Vec<u64>>> {
    let s = basis.len();
    let n = 1usize << s;
    let mut sym = vec![space.zero(); n];
    for (k, v) in sym.iter_mut().enumerate().skip(1) {
        for (i, b) in basis.iter().enumerate() {
            if (k >> i) & 1 == 1 {
                xor_into(v, b);
            }
        }
    }
    let mask: Vec<Vec<u64>> = sym.iter().map(|v| det_mask(space, v)).collect();
    let count: Vec<u32> = mask.iter().map(|m| popcount(m)).collect();

    let mut solved = vec![false; n];
    let mut single_union = space.zero();
    for k in 1..n {
        if count[k] == 1 {
            single_union = or(&single_union, &mask[k]);
            solved[k] = true;
        }
    }
    let mut irreducible = Vec::new();
    for k in 1..n {
        if count[k] == 2 && !subset(&mask[k], &single_union) {
            irreducible.push(k);
            solved[k] = true;
        }
    }

    let mut out: Vec<Vec<Vec<u64>>> = Vec::with_capacity(n - 1);
    for k in 1..n {
        if count[k] == 0 || solved[k] {
            out.push(vec![sym[k].clone()]);
            continue;
        }
        let goal = &mask[k];
        let mut pieces: Vec<Vec<u64>> = Vec::new();
        let mut remnants;
        if subset(goal, &single_union) {
            remnants = goal.clone();
        } else if let Some(&kp) = irreducible
            .iter()
            .find(|&&kp| subset(&mask[kp], goal) && subset(goal, &or(&single_union, &mask[kp])))
        {
            pieces.push(sym[kp].clone());
            remnants = and_not(goal, &mask[kp]);
        } else {
            let mut found = None;
            'pairs: for (i1, &k1) in irreducible.iter().enumerate() {
                for &k2 in &irreducible[i1 + 1..] {
                    let both = or(&mask[k1], &mask[k2]);
                    let disjoint = mask[k1].iter().zip(&mask[k2]).all(|(a, b)| a & b == 0);
                    if disjoint && subset(goal, &or(&single_union, &both)) {
                        found = Some((k1, k2, both));
                        break 'pairs;
                    }
                }
            }
            match found {
                Some((k1, k2, both)) => {
                    pieces.push(sym[k1].clone());
                    pieces.push(sym[k2].clone());
                    remnants = and_not(goal, &both);
                }
                None => {
                    pieces.push(sym[k].clone());
                    remnants = space.zero();
                }
            }
        }
        for k2 in 1..n {
            if is_zero(&remnants) {
                break;
            }
            if count[k2] == 1 && subset(&mask[k2], &remnants) {
                remnants = and_not(&remnants, &mask[k2]);
                pieces.push(sym[k2].clone());
            }
        }
        // Stim trusts this construction; check it. The pieces must XOR back to
        // the combination, observables included, or the model would be wrong.
        let mut check = space.zero();
        for x in &pieces {
            xor_into(&mut check, x);
        }
        if check != sym[k] {
            pieces = vec![sym[k].clone()];
        }
        out.push(pieces);
    }
    out
}

/// Stim's `brute_force_decomposition_into_known_graphlike_errors`: partition a
/// wide piece's detectors into known singles and pairs whose observables XOR to
/// its own. The first unused detector is paired with each later one in turn,
/// then tried alone, exactly in Stim's order.
fn brute_force_known(space: &Space, x: &[u64], known: &HashMap<Vec<u32>, Vec<u64>>) -> Option<Vec<Vec<u64>>> {
    let (dets, obs) = space.split(x);
    fn go(
        space: &Space,
        dets: &[u32],
        used: &mut [bool],
        remaining: u64,
        known: &HashMap<Vec<u32>, Vec<u64>>,
        out: &mut Vec<Vec<u64>>,
    ) -> bool {
        let Some(start) = (0..dets.len()).find(|&i| !used[i]) else {
            return remaining == 0;
        };
        used[start] = true;
        for k in start + 1..=dets.len() {
            let key = if k < dets.len() {
                if used[k] {
                    continue;
                }
                used[k] = true;
                vec![dets[start], dets[k]]
            } else {
                vec![dets[start]]
            };
            if let Some(m) = known.get(&key) {
                out.push(m.clone());
                if go(space, dets, used, remaining ^ space.split(m).1, known, out) {
                    return true;
                }
                out.pop();
            }
            if k < dets.len() {
                used[k] = false;
            }
        }
        used[start] = false;
        false
    }
    let mut used = vec![false; dets.len()];
    let mut out = Vec::new();
    go(space, &dets, &mut used, obs, known, &mut out).then_some(out)
}

/// Stim's `decompose_and_append_component_to_tail`: greedily take known pairs,
/// then known singles, and let whatever is left (at most two detectors) stand
/// as an edge of its own.
fn greedy_known(space: &Space, x: &[u64], known: &HashMap<Vec<u32>, Vec<u64>>) -> Option<Vec<Vec<u64>>> {
    let (dets, _) = space.split(x);
    if dets.len() <= 2 {
        return Some(vec![x.to_vec()]);
    }
    let mut done = vec![false; dets.len()];
    let mut rest = x.to_vec();
    let mut out = Vec::new();
    for k in 0..dets.len() {
        if done[k] {
            continue;
        }
        for k2 in k + 1..dets.len() {
            if done[k2] {
                continue;
            }
            if let Some(m) = known.get(&vec![dets[k], dets[k2]]) {
                done[k] = true;
                done[k2] = true;
                xor_into(&mut rest, m);
                out.push(m.clone());
                break;
            }
        }
    }
    let mut missed = 0;
    for k in 0..dets.len() {
        if !done[k] {
            if let Some(m) = known.get(&vec![dets[k]]) {
                done[k] = true;
                xor_into(&mut rest, m);
                out.push(m.clone());
            }
        }
        missed += !done[k] as usize;
    }
    if missed > 2 {
        return None;
    }
    if !is_zero(&rest) {
        out.push(rest);
    }
    Some(out)
}

/* -- Stim's .dem text ------------------------------------------------------ */

fn push_targets(s: &mut String, dets: &[u32], obs: u64) {
    for d in dets {
        let _ = write!(s, " D{d}");
    }
    for k in 0..64 {
        if (obs >> k) & 1 == 1 {
            let _ = write!(s, " L{k}");
        }
    }
}

impl Dem {
    pub fn to_stim(&self, with_pieces: bool) -> String {
        let mut s = String::new();
        for (i, c) in self.detector_coords.iter().enumerate() {
            if c.is_empty() {
                let _ = writeln!(s, "detector D{i}");
            } else {
                let _ = writeln!(s, "detector({}) D{i}", fmt_args(c));
            }
        }
        for m in &self.mechanisms {
            let _ = write!(s, "error({})", m.p);
            if with_pieces && !m.pieces.is_empty() {
                for (k, piece) in m.pieces.iter().enumerate() {
                    if k > 0 {
                        s.push_str(" ^");
                    }
                    push_targets(&mut s, &piece.detectors, piece.observables);
                }
            } else {
                push_targets(&mut s, &m.detectors, m.observables);
            }
            s.push('\n');
        }
        s
    }

    pub fn parse(text: &str) -> Result<Dem, String> {
        let lines: Vec<&str> = text.lines().collect();
        let mut dem = Dem::default();
        let mut pos = 0usize;
        let mut offset = 0u64;
        let mut shift = Vec::new();
        parse_dem_block(&lines, &mut pos, false, &mut dem, &mut offset, &mut shift)?;
        dem.detector_coords.resize(dem.num_detectors, Vec::new());
        Ok(dem)
    }
}

fn toggle_det(v: &mut Vec<u32>, d: u32) {
    match v.iter().position(|&x| x == d) {
        Some(i) => {
            v.swap_remove(i);
        }
        None => v.push(d),
    }
}

fn parse_dem_block(
    lines: &[&str],
    pos: &mut usize,
    nested: bool,
    dem: &mut Dem,
    offset: &mut u64,
    shift: &mut Vec<f64>,
) -> Result<(), String> {
    while *pos < lines.len() {
        let lineno = *pos + 1;
        let line = lines[*pos].split('#').next().unwrap_or("").trim();
        *pos += 1;
        if line.is_empty() {
            continue;
        }
        if line == "}" {
            if nested {
                return Ok(());
            }
            return Err(format!("line {lineno}: unmatched '}}'"));
        }
        if let Some(rest) = line.strip_suffix('{') {
            let mut parts = rest.split_whitespace();
            if !parts.next().unwrap_or("").eq_ignore_ascii_case("repeat") {
                return Err(format!("line {lineno}: only repeat opens a block"));
            }
            let count: u64 = parts
                .next()
                .and_then(|s| s.parse().ok())
                .ok_or_else(|| format!("line {lineno}: repeat needs a count"))?;
            let start = *pos;
            if count == 0 {
                // Walk the body to find its end without applying it.
                let mut skip = Dem::default();
                let (mut o, mut s) = (0u64, Vec::new());
                parse_dem_block(lines, pos, true, &mut skip, &mut o, &mut s)?;
            }
            for _ in 0..count {
                *pos = start;
                parse_dem_block(lines, pos, true, dem, offset, shift)?;
            }
            continue;
        }
        let (name, args, tokens) = split_instruction(line).map_err(|e| format!("line {lineno}: {e}"))?;
        let bad = |t: &str| format!("line {lineno}: bad target '{t}'");
        match name.as_str() {
            "ERROR" => {
                if args.len() != 1 || !(0.0..=1.0).contains(&args[0]) {
                    return Err(format!("line {lineno}: error takes one probability"));
                }
                let mut pieces = vec![Piece::default()];
                for t in &tokens {
                    if *t == "^" {
                        pieces.push(Piece::default());
                    } else if let Some(d) = t.strip_prefix('D') {
                        let d = d.parse::<u64>().map_err(|_| bad(t))? + *offset;
                        dem.num_detectors = dem.num_detectors.max(d as usize + 1);
                        toggle_det(&mut pieces.last_mut().unwrap().detectors, d as u32);
                    } else if let Some(l) = t.strip_prefix('L') {
                        let l = l.parse::<u32>().map_err(|_| bad(t))?;
                        if l >= 64 {
                            return Err(format!("line {lineno}: at most 64 observables are supported"));
                        }
                        dem.num_observables = dem.num_observables.max(l as usize + 1);
                        pieces.last_mut().unwrap().observables ^= 1u64 << l;
                    } else {
                        return Err(bad(t));
                    }
                }
                let mut detectors = Vec::new();
                let mut observables = 0u64;
                for piece in &mut pieces {
                    piece.detectors.sort_unstable();
                    for &d in &piece.detectors {
                        toggle_det(&mut detectors, d);
                    }
                    observables ^= piece.observables;
                }
                detectors.sort_unstable();
                let pieces = if pieces.len() > 1 {
                    pieces.into_iter().filter(|p| !p.detectors.is_empty() || p.observables != 0).collect()
                } else if !detectors.is_empty() && detectors.len() <= 2 {
                    vec![Piece { detectors: detectors.clone(), observables }]
                } else {
                    Vec::new()
                };
                dem.mechanisms.push(Mechanism { p: args[0], detectors, observables, pieces });
            }
            "DETECTOR" => {
                for t in &tokens {
                    let d = t.strip_prefix('D').and_then(|d| d.parse::<u64>().ok()).ok_or_else(|| bad(t))? + *offset;
                    let d = d as usize;
                    dem.num_detectors = dem.num_detectors.max(d + 1);
                    if dem.detector_coords.len() <= d {
                        dem.detector_coords.resize(d + 1, Vec::new());
                    }
                    dem.detector_coords[d] =
                        args.iter().enumerate().map(|(i, v)| v + shift.get(i).copied().unwrap_or(0.0)).collect();
                }
            }
            "LOGICAL_OBSERVABLE" => {
                for t in &tokens {
                    let l = t.strip_prefix('L').and_then(|l| l.parse::<usize>().ok()).ok_or_else(|| bad(t))?;
                    dem.num_observables = dem.num_observables.max(l + 1);
                }
            }
            "SHIFT_DETECTORS" => {
                if shift.len() < args.len() {
                    shift.resize(args.len(), 0.0);
                }
                for (a, b) in shift.iter_mut().zip(&args) {
                    *a += b;
                }
                for t in &tokens {
                    *offset += t.parse::<u64>().map_err(|_| bad(t))?;
                }
            }
            _ => return Err(format!("line {lineno}: unsupported instruction '{name}'")),
        }
    }
    if nested {
        return Err("unterminated repeat block".into());
    }
    Ok(())
}

/* -- Comparison ------------------------------------------------------------ */

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Comparison {
    pub ours: usize,
    pub theirs: usize,
    /// In theirs, not in ours.
    pub missing: usize,
    /// In ours, not in theirs.
    pub extra: usize,
    /// In both, with relative probability difference above the tolerance.
    pub differing: usize,
    pub max_rel: f64,
}

fn merged(d: &Dem) -> HashMap<(Vec<u32>, u64), f64> {
    let mut out: HashMap<(Vec<u32>, u64), f64> = HashMap::new();
    for m in &d.mechanisms {
        let e = out.entry((m.detectors.clone(), m.observables)).or_insert(0.0);
        *e = xor_prob(*e, m.p);
    }
    out
}

/// Mechanism-by-mechanism comparison, after merging identical symptoms on each side.
pub fn compare(ours: &Dem, theirs: &Dem, tol: f64) -> Comparison {
    let a = merged(ours);
    let b = merged(theirs);
    let mut c = Comparison { ours: a.len(), theirs: b.len(), missing: 0, extra: 0, differing: 0, max_rel: 0.0 };
    for (k, &pa) in &a {
        match b.get(k) {
            Some(&pb) => {
                let rel = (pa - pb).abs() / pa.abs().max(pb.abs());
                c.max_rel = c.max_rel.max(rel);
                if rel > tol {
                    c.differing += 1;
                }
            }
            None => c.extra += 1,
        }
    }
    c.missing = b.keys().filter(|k| !a.contains_key(*k)).count();
    c
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixtures::{REP3, SAMPLE};

    fn dem(text: &str) -> Dem {
        Dem::from_circuit(&Circuit::parse(text).unwrap()).unwrap()
    }

    fn p_of(d: &Dem, dets: &[u32], obs: u64) -> f64 {
        d.mechanisms.iter().find(|m| m.detectors == dets && m.observables == obs).map(|m| m.p).unwrap()
    }

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() <= 1e-12 * a.abs().max(b.abs())
    }

    /// Numbers from Stim 1.16, `circuit.detector_error_model()`.
    #[test]
    fn depolarizing_channels_match_stim() {
        let d = dem("R 0\nDEPOLARIZE1(0.01) 0\nM 0\nDETECTOR rec[-1]");
        assert_eq!(d.mechanisms.len(), 1);
        assert!(close(d.mechanisms[0].p, 0.006666666666666613), "{}", d.mechanisms[0].p);

        let d = dem("R 0 1\nDEPOLARIZE2(0.01) 0 1\nM 0 1\nDETECTOR rec[-1]\nDETECTOR rec[-2]");
        assert_eq!(d.mechanisms.len(), 3);
        for m in &d.mechanisms {
            assert!(close(m.p, 0.002673815958446298), "{}", m.p);
        }
    }

    #[test]
    fn pauli_channel_and_flips_match_stim() {
        let d = dem("R 0\nPAULI_CHANNEL_1(0.01, 0.02, 0.03) 0\nM 0\nDETECTOR rec[-1]\n\
                     RX 1\nPAULI_CHANNEL_1(0.01, 0.02, 0.03) 1\nMX 1\nDETECTOR rec[-1]");
        assert!(close(p_of(&d, &[0], 0), 0.03) && close(p_of(&d, &[1], 0), 0.05));

        let d = dem("R 0\nX_ERROR(0.01) 0\nM(0.02) 0\nDETECTOR rec[-1]\nRX 1\nY_ERROR(0.1) 1\nMRX 1\nDETECTOR rec[-1]");
        assert!(close(p_of(&d, &[0], 0), 0.0296) && close(p_of(&d, &[1], 0), 0.1));
    }

    #[test]
    fn gates_propagate_errors_backwards_correctly() {
        // X on a control before CX spreads to the target.
        let d = dem("R 0 1\nX_ERROR(0.1) 0\nCX 0 1\nM 0 1\nDETECTOR rec[-2]\nDETECTOR rec[-1]");
        assert!(close(p_of(&d, &[0, 1], 0), 0.1));
        // Z on a target before CX spreads to the control.
        let d = dem("RX 0 1\nZ_ERROR(0.2) 1\nCX 0 1\nMX 0 1\nDETECTOR rec[-2]\nDETECTOR rec[-1]");
        assert!(close(p_of(&d, &[0, 1], 0), 0.2));
        // X before CZ becomes X on its own qubit and Z on the other.
        let d = dem("R 0\nRX 1\nX_ERROR(0.1) 0\nCZ 0 1\nM 0\nMX 1\nDETECTOR rec[-2]\nDETECTOR rec[-1]");
        assert!(close(p_of(&d, &[0, 1], 0), 0.1));
        // H swaps: X before H flips an X-basis readout, Z before H does not.
        let d = dem("R 0\nX_ERROR(0.1) 0\nZ_ERROR(0.2) 0\nH 0\nMX 0\nDETECTOR rec[-1]");
        assert_eq!(d.mechanisms.len(), 1);
        assert!(close(p_of(&d, &[0], 0), 0.1));
    }

    #[test]
    fn nondeterministic_detectors_are_errors() {
        let e = Dem::from_circuit(&Circuit::parse("RX 0\nM 0\nDETECTOR rec[-1]").unwrap()).unwrap_err();
        assert!(e.contains("not deterministic"), "{e}");
        let e = Dem::from_circuit(&Circuit::parse("M 0\nMX 0\nDETECTOR rec[-1]").unwrap()).unwrap_err();
        assert!(e.contains("not deterministic"), "{e}");
    }

    #[test]
    fn undetectable_logical_errors_are_errors() {
        let e = Dem::from_circuit(&Circuit::parse("R 0\nX_ERROR(0.1) 0\nM 0\nOBSERVABLE_INCLUDE(0) rec[-1]").unwrap())
            .unwrap_err();
        assert!(e.contains("undetectable"), "{e}");
    }

    #[test]
    fn sample_and_rep3_build() {
        let d = dem(SAMPLE);
        assert_eq!(d.num_detectors, 4);
        let d = dem(REP3);
        assert_eq!((d.num_detectors, d.num_observables), (6, 1));
        assert!(d.mechanisms.iter().any(|m| m.observables == 1));
    }

    #[test]
    fn stim_text_round_trips() {
        let d = dem(REP3);
        let back = Dem::parse(&d.to_stim(false)).unwrap();
        let c = compare(&d, &back, 1e-12);
        assert_eq!((c.missing, c.extra, c.differing), (0, 0, 0));
        assert_eq!(back.num_detectors, d.num_detectors);
        assert_eq!(back.detector_coords, d.detector_coords);
    }

    #[test]
    fn parse_reads_pieces_repeats_and_shifts() {
        let d = Dem::parse(
            "error(0.1) D0 D1 ^ D2\nrepeat 2 {\n  error(0.2) D0\n  shift_detectors 1\n}\ndetector(1, 2) D0\n",
        )
        .unwrap();
        assert_eq!(d.mechanisms[0].detectors, vec![0, 1, 2]);
        assert_eq!(d.mechanisms[0].pieces.len(), 2);
        assert_eq!(d.mechanisms[1].detectors, vec![0]);
        assert_eq!(d.mechanisms[2].detectors, vec![1]);
        assert_eq!(d.detector_coords[2], vec![1.0, 2.0]);
        assert_eq!(d.num_detectors, 3);
    }

    #[test]
    fn comparison_counts_differences() {
        let a = Dem::parse("error(0.1) D0\nerror(0.2) D1").unwrap();
        let b = Dem::parse("error(0.1) D0\nerror(0.25) D1\nerror(0.1) D0 D1").unwrap();
        let c = compare(&a, &b, 1e-9);
        assert_eq!((c.ours, c.theirs, c.missing, c.extra, c.differing), (2, 3, 1, 0, 1));
        assert!((c.max_rel - 0.2).abs() < 1e-12);
    }

    #[test]
    fn pauli_gates_and_sweep_bits_leave_the_model_alone() {
        // They flip signs, never which detectors a fault sets off.
        let base = Circuit::parse(crate::fixtures::REP3).unwrap();
        let text = crate::fixtures::REP3.replacen("TICK\n", "TICK\nX 0 1\nY 2\nCX sweep[0] 1\nI 0\n", 1);
        let with = Circuit::parse(&text).unwrap();
        assert_ne!(with, base);
        assert_eq!(Dem::from_circuit(&with).unwrap().to_stim(true), Dem::from_circuit(&base).unwrap().to_stim(true));
    }
}

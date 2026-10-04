//! Building a circuit's detector error model, folding its loops as Stim does.
//!
//! BUILT BACKWARDS, SPARSELY, AND FOLDED
//! --------------------------------------
//! The circuit is walked once, in reverse, carrying for every qubit the detectors and
//! observables an X error there would flip (`sx`) and those a Z error would flip (`sz`), and
//! for every measurement still to be reached, those its result feeds. At a noise instruction
//! the symptom of each Pauli it can apply is read off directly (see `dem.rs` for the
//! conjugation rules). The sets are sorted lists of targets, as in Stim's tracker: a qubit's
//! set holds the few detectors near it, so a gate costs the size of two small lists, not the
//! number of detectors in the circuit.
//!
//! A `REPEAT` block is not unrolled. As in Stim's error analyzer, a second walker (the hare)
//! runs ahead through the loop's iterations without collecting errors, while the first (the
//! tortoise) follows at half its pace; once the hare's state is the tortoise's with every
//! detector shifted by the detectors between them, every further period of the loop yields the
//! same errors shifted, so one period's errors are collected and written as a `repeat` block,
//! and the walk jumps to the loop's start. Errors are merged, and wide ones split by the pieces
//! already known, within each stretch between such jumps, exactly where Stim flushes its own.
//! A d = 11 memory of 10,000 rounds is analysed in milliseconds and printed in a few hundred
//! lines.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;

use crate::batch_sampler::Counts;
use crate::circuit::{Basis, Circuit, Control, Instr};
use crate::dem::{depolarize1_component, depolarize2_component, fused, pauli_channel_1_independent};
use crate::dem_program::{DemInstr, DemProgram, OBS};

/// Sorted, distinct targets: detectors by index, observables with the `OBS` bit.
type Sym = Vec<u64>;

/// The most instructions and targets the walk may undo, loops counted pass by pass: what a
/// loop costs when it never settles into a period and is walked in full.
const MAX_WORK: u64 = 1 << 26;

/* -- Sparse symptoms ------------------------------------------------------- */

thread_local! {
    /// Targets merged by `xor` on this thread: the walk's real work, which `Analyzer::spend`
    /// bounds alongside its instructions.
    static MERGED: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

fn xor(a: &[u64], b: &[u64]) -> Sym {
    MERGED.with(|m| m.set(m.get().wrapping_add((a.len() + b.len()) as u64)));
    let mut out = Vec::with_capacity(a.len() + b.len());
    let (mut i, mut j) = (0, 0);
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Less => {
                out.push(a[i]);
                i += 1;
            }
            std::cmp::Ordering::Greater => {
                out.push(b[j]);
                j += 1;
            }
            std::cmp::Ordering::Equal => {
                i += 1;
                j += 1;
            }
        }
    }
    out.extend_from_slice(&a[i..]);
    out.extend_from_slice(&b[j..]);
    out
}

fn xor_into(a: &mut Sym, b: &[u64]) {
    if !b.is_empty() {
        *a = xor(a, b);
    }
}

fn toggle(a: &mut Sym, t: u64) {
    match a.binary_search(&t) {
        Ok(i) => {
            a.remove(i);
        }
        Err(i) => a.insert(i, t),
    }
}

/// The detectors of a symptom.
fn dets(s: &[u64]) -> Vec<u64> {
    s.iter().copied().filter(|t| t & OBS == 0).collect()
}

/// The observables of a symptom, as a mask.
fn obs_mask(s: &[u64]) -> u64 {
    s.iter().filter(|t| *t & OBS != 0).fold(0, |m, t| m ^ (1u64 << (t & !OBS)))
}

fn subset(a: &[u64], b: &[u64]) -> bool {
    let mut j = 0;
    for &x in a {
        while j < b.len() && b[j] < x {
            j += 1;
        }
        if j == b.len() || b[j] != x {
            return false;
        }
    }
    true
}

fn or(a: &[u64], b: &[u64]) -> Sym {
    let mut out: Sym = a.iter().chain(b).copied().collect();
    out.sort_unstable();
    out.dedup();
    out
}

fn and_not(a: &[u64], b: &[u64]) -> Sym {
    a.iter().copied().filter(|x| b.binary_search(x).is_err()).collect()
}

fn disjoint(a: &[u64], b: &[u64]) -> bool {
    a.iter().all(|x| b.binary_search(x).is_err())
}

/// `a` with every detector moved by `by` equals `b`.
fn shifted_eq(a: &[u64], b: &[u64], by: i128) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(&x, &y)| if x & OBS == 0 { x as i128 + by == y as i128 } else { x == y })
}

fn shift_sym(s: &mut Sym, by: i128) {
    for t in s.iter_mut() {
        if *t & OBS == 0 {
            *t = (*t as i128 + by) as u64;
        }
    }
}

/* -- The tracker ----------------------------------------------------------- */

/// The walk's state: what an X or Z error on each qubit would flip, what each measurement yet
/// to be reached feeds, and how many measurements and detectors lie before the current point.
#[derive(Clone)]
struct Tracker {
    sx: Vec<Sym>,
    sz: Vec<Sym>,
    rec: BTreeMap<u64, Sym>,
    m: u64,
    d: u64,
}

impl Tracker {
    /// Whether `other` is this state with every measurement and detector moved by the
    /// measurements and detectors between them (Stim's `is_shifted_copy`).
    fn is_shifted_copy(&self, other: &Tracker) -> bool {
        let dm = other.m as i128 - self.m as i128;
        let dd = other.d as i128 - self.d as i128;
        self.rec.len() == other.rec.len()
            && self.rec.iter().zip(&other.rec).all(|((ka, a), (kb, b))| *ka as i128 + dm == *kb as i128 && shifted_eq(a, b, dd))
            && self.sx.iter().zip(&other.sx).all(|(a, b)| shifted_eq(a, b, dd))
            && self.sz.iter().zip(&other.sz).all(|(a, b)| shifted_eq(a, b, dd))
    }

    fn shift(&mut self, dm: i128, dd: i128) {
        self.m = (self.m as i128 + dm) as u64;
        self.d = (self.d as i128 + dd) as u64;
        let rec = std::mem::take(&mut self.rec);
        self.rec = rec
            .into_iter()
            .map(|(k, mut v)| {
                shift_sym(&mut v, dd);
                ((k as i128 + dm) as u64, v)
            })
            .collect();
        for s in self.sx.iter_mut().chain(self.sz.iter_mut()) {
            shift_sym(s, dd);
        }
    }

    fn take_record(&mut self) -> Result<Sym, String> {
        self.m = self.m.checked_sub(1).ok_or("a measurement before the first measurement")?;
        Ok(self.rec.remove(&self.m).unwrap_or_default())
    }

    /// The record `lookback` measurements before the current point, toggled by `with`.
    fn feed(&mut self, lookback: u32, with: &[u64]) -> Result<(), String> {
        if lookback == 0 || u64::from(lookback) > self.m {
            return Err(format!("rec[-{lookback}] reaches before the first measurement"));
        }
        let k = self.m - u64::from(lookback);
        let entry = self.rec.entry(k).or_default();
        xor_into(entry, with);
        if entry.is_empty() {
            self.rec.remove(&k);
        }
        Ok(())
    }
}

/* -- Error classes --------------------------------------------------------- */

/// Where a fault came from, for error messages.
#[derive(Clone, Copy)]
struct Origin {
    name: &'static str,
    a: u32,
    b: Option<u32>,
    pauli: (u8, u8),
}

fn pauli_name(p: u8) -> &'static str {
    ["I", "X", "Z", "Y"][p as usize & 3]
}

impl Origin {
    fn describe(&self) -> String {
        match self.b {
            Some(b) => format!("{} {}⊗{} on qubits {}, {}", self.name, pauli_name(self.pauli.0), pauli_name(self.pauli.1), self.a, b),
            None => format!("{} {} on qubit {}", self.name, pauli_name(self.pauli.0), self.a),
        }
    }
}

/// Stim's separator between a fault's pieces, which sorts after every target.
const SEP: u64 = u64::MAX;

/// A fault class: its probability, accumulated as faults with its key arrive, and where its
/// first fault came from.
struct Class {
    p: f64,
    origin: Origin,
}

/// The fault classes of one stretch of the walk, as Stim keeps them: keyed by the pieces in
/// the order they arose (joined by `SEP`) and the tag, in sorted order, each probability
/// combined fault by fault in the order the walk meets them.
#[derive(Default)]
struct Window {
    classes: BTreeMap<(Vec<u64>, String), Class>,
}

impl Window {
    fn add(&mut self, p: f64, pieces: Vec<Sym>, origin: Origin, tag: &Option<Arc<str>>, decompose: bool) {
        if p <= 0.0 {
            return;
        }
        let key = if decompose {
            let mut key = Vec::new();
            for x in pieces.iter().filter(|x| !x.is_empty()) {
                if !key.is_empty() {
                    key.push(SEP);
                }
                key.extend_from_slice(x);
            }
            key
        } else {
            pieces.iter().fold(Sym::new(), |acc, x| xor(&acc, x))
        };
        if key.is_empty() {
            return;
        }
        let tag = tag.as_deref().unwrap_or("").to_string();
        let c = self.classes.entry((key, tag)).or_insert(Class { p: 0.0, origin });
        c.p = join(c.p, p);
    }
}

/// A class's probability after another independent fault of probability `p` joins it, as
/// Stim computes `old·(1 − p) + (1 − old)·p` (rounded as its build rounds, see `fused`), so
/// every printed digit equals Stim's.
fn join(old: f64, p: f64) -> f64 {
    fused(old, 1.0 - p, (1.0 - old) * p)
}

/// A key's pieces.
fn components(key: &[u64]) -> impl Iterator<Item = &[u64]> {
    key.split(|&t| t == SEP)
}

/// Every piece fires at most two detectors.
fn graphlike(key: &[u64]) -> bool {
    components(key).all(|c| c.iter().filter(|&&t| t & OBS == 0).count() <= 2)
}


/* -- The walk -------------------------------------------------------------- */

struct Analyzer {
    t: Tracker,
    /// The hare walks without collecting errors.
    accumulate: bool,
    /// Fold loops that repeat (Stim's default), or walk every pass (`flatten_loops`).
    fold: bool,
    decompose: bool,
    approximate: Option<f64>,
    window: Window,
    /// Stim's flushed, reversed model: everything walked so far, last first, detectors
    /// absolute.
    reversed: Vec<DemInstr>,
    work: u64,
    /// When set, what each sweep bit's X flips, gathered as the walk passes it.
    sweeps: Option<Vec<Sym>>,
}

impl Analyzer {
    fn spend(&mut self, units: u64) -> Result<(), String> {
        self.work = self.work.saturating_add(units);
        let merged = MERGED.with(|m| m.replace(0));
        self.work = self.work.saturating_add(merged / 8);
        if self.work > MAX_WORK {
            return Err(format!(
                "the circuit's model is too large to build: its loops do not settle into a repeating pattern within {MAX_WORK} units of work"
            ));
        }
        Ok(())
    }

    fn add(&mut self, p: f64, pieces: Vec<Sym>, origin: Origin, tag: &Option<Arc<str>>) {
        if self.accumulate {
            self.window.add(p, pieces, origin, tag, self.decompose);
        }
    }

    /// A channel of disjoint cases is allowed only approximately, as in Stim; one with at most
    /// one case of nonzero probability needs no approximation when `single` allows it.
    fn approximated(&self, name: &str, args: &[f64], single: bool) -> Result<(), String> {
        if single && args.iter().filter(|&&p| p > 0.0).count() <= 1 {
            return Ok(());
        }
        match self.approximate {
            None => Err(format!(
                "{name}: its cases are disjoint, not independent faults; an error model holds it only with approximate_disjoint_errors"
            )),
            Some(t) => match args.iter().find(|&&p| p > t) {
                Some(p) => Err(format!("{name} has a probability ({p}) above the approximate_disjoint_errors threshold ({t})")),
                None => Ok(()),
            },
        }
    }

    fn check_reset(&self, q: usize, basis: Basis, what: &str) -> Result<(), String> {
        let sensitive = match basis {
            Basis::Z => &self.t.sz[q],
            Basis::X => &self.t.sx[q],
        };
        match sensitive.first() {
            None => Ok(()),
            Some(&t) if t & OBS == 0 => Err(format!("detector D{t} is not deterministic: it depends on the random outcome of {what} on qubit {q}")),
            Some(&t) => Err(format!("observable L{} is not deterministic: it depends on the random outcome of {what} on qubit {q}", t & !OBS)),
        }
    }

    fn sym_of(&self, q: usize, pauli: u8) -> Sym {
        match pauli & 3 {
            1 => self.t.sx[q].clone(),
            2 => self.t.sz[q].clone(),
            3 => xor(&self.t.sx[q], &self.t.sz[q]),
            _ => Sym::new(),
        }
    }

    fn undo_block(&mut self, instrs: &[Instr], outer: &Option<Arc<str>>) -> Result<(), String> {
        // The ELSE_CORRELATED_ERRORs met walking backwards, until their E.
        let mut chain: Vec<(f64, Sym, Origin)> = Vec::new();
        for ins in instrs.iter().rev() {
            // A tagged instruction is a gate holding it untagged.
            let (ins, tag) = match ins {
                Instr::Gate { body, tag, .. } if !tag.is_empty() && body.len() == 1 => (&body[0], Some(Arc::<str>::from(tag.as_str()))),
                _ => (ins, outer.clone()),
            };
            self.spend(1 + ins.qubits().len() as u64)?;
            if let Instr::Correlated { p, paulis, chained } = ins {
                let mut sym = Sym::new();
                for &(q, pauli) in paulis {
                    xor_into(&mut sym, &self.sym_of(q as usize, pauli));
                }
                let (a, pa) = paulis.first().copied().unwrap_or((0, 0));
                let (b, pb) = paulis.get(1).copied().map_or((None, 0), |(q, p)| (Some(q), p));
                let name = if *chained { "ELSE_CORRELATED_ERROR" } else { "correlated error" };
                let origin = Origin { name, a, b, pauli: (pa, pb) };
                if *chained {
                    chain.push((*p, sym, origin));
                    continue;
                }
                if !chain.is_empty() {
                    // Stim checks the threshold against each case's actual probability.
                    self.approximated("ELSE_CORRELATED_ERROR", &[], false)?;
                }
                let mut none_yet = 1.0 - p;
                let mut cases = vec![(*p, sym, origin)];
                for (q, sym, origin) in chain.drain(..).rev() {
                    cases.push((q * none_yet, sym, origin));
                    none_yet *= 1.0 - q;
                }
                let t = self.approximate.unwrap_or(1.0);
                if let Some((actual, ..)) = cases.iter().find(|c| cases.len() > 1 && c.0 > t) {
                    return Err(format!(
                        "an E / ELSE_CORRELATED_ERROR chain has a case of probability {actual}, above the approximate_disjoint_errors threshold ({t})"
                    ));
                }
                for (p, sym, origin) in cases {
                    self.add(p, vec![sym], origin, &tag);
                }
                continue;
            }
            if !chain.is_empty() {
                return Err("ELSE_CORRELATED_ERROR is not preceded by E or another ELSE_CORRELATED_ERROR".into());
            }
            self.undo(ins, &tag)?;
        }
        if !chain.is_empty() {
            return Err("ELSE_CORRELATED_ERROR is not preceded by E or another ELSE_CORRELATED_ERROR".into());
        }
        Ok(())
    }

    fn undo(&mut self, ins: &Instr, tag: &Option<Arc<str>>) -> Result<(), String> {
        let reset_name = |basis: Basis| if basis == Basis::Z { "a Z-basis reset" } else { "an X-basis reset" };
        match ins {
            Instr::Repeat { count, body, tag: own } => self.run_loop(body, *count, own, tag)?,
            Instr::Gate { body, tag: own, .. } => {
                let inner = if own.is_empty() { tag.clone() } else { Some(Arc::<str>::from(own.as_str())) };
                self.undo_block(body, &inner)?;
            }
            Instr::Reset { basis, qubits } => {
                for &q in qubits.iter().rev() {
                    let q = q as usize;
                    self.check_reset(q, *basis, reset_name(*basis))?;
                    self.t.sx[q].clear();
                    self.t.sz[q].clear();
                }
            }
            Instr::Measure { basis, reset, flip, qubits } => {
                for &q in qubits.iter().rev() {
                    let qi = q as usize;
                    let r = self.t.take_record()?;
                    if *reset {
                        self.check_reset(qi, *basis, reset_name(*basis))?;
                        self.t.sx[qi].clear();
                        self.t.sz[qi].clear();
                    }
                    match basis {
                        Basis::Z => xor_into(&mut self.t.sx[qi], &r),
                        Basis::X => xor_into(&mut self.t.sz[qi], &r),
                    }
                    if *flip > 0.0 {
                        let origin = Origin { name: "measurement flip", a: q, b: None, pauli: (1, 0) };
                        self.add(*flip, vec![r], origin, tag);
                    }
                }
            }
            Instr::H(qubits) => {
                for &q in qubits.iter().rev() {
                    let q = q as usize;
                    std::mem::swap(&mut self.t.sx[q], &mut self.t.sz[q]);
                }
            }
            // S takes X to Y and keeps Z: an X before it is a Y after.
            Instr::S(qubits) => {
                for &q in qubits.iter().rev() {
                    let q = q as usize;
                    let from_z = self.t.sz[q].clone();
                    xor_into(&mut self.t.sx[q], &from_z);
                }
            }
            Instr::Cx(pairs) => {
                for &(c, t) in pairs.iter().rev() {
                    let (c, t) = (c as usize, t as usize);
                    let from_t = self.t.sx[t].clone();
                    xor_into(&mut self.t.sx[c], &from_t);
                    let from_c = self.t.sz[c].clone();
                    xor_into(&mut self.t.sz[t], &from_c);
                }
            }
            Instr::Cz(pairs) => {
                for &(a, b) in pairs.iter().rev() {
                    let (a, b) = (a as usize, b as usize);
                    let (za, zb) = (self.t.sz[a].clone(), self.t.sz[b].clone());
                    xor_into(&mut self.t.sx[a], &zb);
                    xor_into(&mut self.t.sx[b], &za);
                }
            }
            Instr::Pad { flip, values } => {
                for _ in values.iter().rev() {
                    let r = self.t.take_record()?;
                    if *flip > 0.0 {
                        let origin = Origin { name: "MPAD flip", a: 0, b: None, pauli: (1, 0) };
                        self.add(*flip, vec![r], origin, tag);
                    }
                }
            }
            Instr::Detector { coords, recs } => {
                self.t.d = self.t.d.checked_sub(1).ok_or("a detector before the first detector")?;
                let id = self.t.d;
                for &k in recs {
                    self.t.feed(k, &[id])?;
                }
                let tag = tag.as_deref().unwrap_or("").to_string();
                self.reversed.push(DemInstr::Detector { coords: coords.clone(), targets: vec![id], tag });
            }
            // A Pauli target: errors before it that anticommute with it flip the observable,
            // a Z component an X target and an X component a Z target.
            Instr::Observable { index, recs, paulis } => {
                if *index >= 64 {
                    return Err(format!("OBSERVABLE_INCLUDE({index}): at most 64 observables are supported"));
                }
                let target = OBS | u64::from(*index);
                for &k in recs {
                    self.t.feed(k, &[target])?;
                }
                for &(q, pauli, _) in paulis {
                    let q = q as usize;
                    if pauli & 1 != 0 {
                        toggle(&mut self.t.sz[q], target);
                    }
                    if pauli & 2 != 0 {
                        toggle(&mut self.t.sx[q], target);
                    }
                }
                let tag = tag.as_deref().unwrap_or("").to_string();
                self.reversed.push(DemInstr::Observable { targets: vec![*index], tag });
            }
            Instr::ShiftCoords(c) => {
                let tag = tag.as_deref().unwrap_or("").to_string();
                self.reversed.push(DemInstr::Shift { coords: c.clone(), by: 0, tag });
            }
            // An error flipping the controlling record also flips the Pauli here.
            Instr::Feedback { pauli, control: Control::Rec(k), qubit } => {
                let sym = self.sym_of(*qubit as usize, *pauli);
                self.t.feed(*k, &sym)?;
            }
            // A sweep bit moves only the noiseless reference: what it moves, where asked.
            Instr::Feedback { pauli, control: Control::Sweep(k), qubit } => {
                let sym = self.sym_of(*qubit as usize, *pauli);
                if let Some(sweeps) = self.sweeps.as_mut() {
                    xor_into(&mut sweeps[*k as usize], &sym);
                }
            }
            Instr::SweepX(pairs) => {
                if let Some(sweeps) = self.sweeps.as_mut() {
                    for &(k, q) in pairs {
                        xor_into(&mut sweeps[k as usize], &self.t.sx[q as usize]);
                    }
                }
            }
            Instr::Correlated { .. } => unreachable!("undo_block takes correlated errors"),
            Instr::PauliChannel2 { probs, pairs } => {
                self.approximated("PAULI_CHANNEL_2", probs, true)?;
                for &(qa, qb) in pairs {
                    let (a, b) = (qa as usize, qb as usize);
                    // Stim's basis: the second qubit's X and Z errors, then the first's.
                    let t = &self.t;
                    let combos = channel_combinations(&[t.sx[b].clone(), t.sz[b].clone(), t.sx[a].clone(), t.sz[a].clone()]);
                    // Case k + 1 in Stim's order: the first qubit's Pauli (k + 1) / 4, the
                    // second's (k + 1) % 4, each I, X, Y, Z, as bits (X 1, Z 2).
                    const BITS: [usize; 4] = [0b00, 0b01, 0b11, 0b10];
                    let mut by_combo = vec![0.0; 16];
                    for (k, &p) in probs.iter().enumerate() {
                        by_combo[BITS[(k + 1) % 4] | (BITS[(k + 1) / 4] << 2)] = p;
                    }
                    let origin = Origin { name: "PAULI_CHANNEL_2", a: qa, b: Some(qb), pauli: (0, 0) };
                    self.add_disjoint(by_combo, &combos, origin, tag);
                }
            }
            Instr::Heralded { erase, args, probs, qubits } => {
                let name = if *erase { "HERALDED_ERASE" } else { "HERALDED_PAULI_CHANNEL_1" };
                self.approximated(name, args, !*erase)?;
                for &q in qubits.iter().rev() {
                    let herald = self.t.take_record()?;
                    let qi = q as usize;
                    // Stim's basis: the Z error, the X error, the herald.
                    let combos = channel_combinations(&[self.t.sz[qi].clone(), self.t.sx[qi].clone(), herald]);
                    // The herald with I, X, Y or Z.
                    let mut by_combo = vec![0.0; 8];
                    for (p, k) in [(probs[0], 0b100), (probs[1], 0b110), (probs[2], 0b111), (probs[3], 0b101)] {
                        by_combo[k] = p;
                    }
                    let origin = Origin { name: "heralded error", a: q, b: None, pauli: (0, 0) };
                    self.add_disjoint(by_combo, &combos, origin, tag);
                }
            }
            // X_ERROR, Y_ERROR and Z_ERROR are single faults and stay whole, a Y included, as
            // Stim leaves them; anything wider than a pair is split at the flush.
            Instr::PauliError { pauli, p, qubits } => {
                for &q in qubits {
                    let sym = self.sym_of(q as usize, *pauli);
                    let origin = Origin { name: "Pauli error", a: q, b: None, pauli: (*pauli, 0) };
                    self.add(*p, vec![sym], origin, tag);
                }
            }
            // The composite channels are split combination by combination, with Stim's basis
            // order: Z then X for DEPOLARIZE1, X then Z for PAULI_CHANNEL_1, and Z_a, X_a, Z_b,
            // X_b for DEPOLARIZE2.
            Instr::Depolarize1 { p, qubits } => {
                if *p > 0.75 {
                    return Err(format!("DEPOLARIZE1({p}) exceeds 3/4"));
                }
                let q1 = depolarize1_component(*p);
                for &q in qubits {
                    let qi = q as usize;
                    let combos = channel_combinations(&[self.t.sz[qi].clone(), self.t.sx[qi].clone()]);
                    for (k, pieces) in combos.into_iter().enumerate() {
                        let origin = Origin { name: "DEPOLARIZE1", a: q, b: None, pauli: ([2u8, 1, 3][k], 0) };
                        self.add(q1, pieces, origin, tag);
                    }
                }
            }
            Instr::PauliChannel1 { px, py, pz, qubits } => match pauli_channel_1_independent(*px, *py, *pz) {
                Err(_) => {
                    // No independent equivalent: approximately, each case its own fault.
                    self.approximated("PAULI_CHANNEL_1", &[*px, *py, *pz], true)?;
                    for &q in qubits {
                        let qi = q as usize;
                        let combos = channel_combinations(&[self.t.sx[qi].clone(), self.t.sz[qi].clone()]);
                        let origin = Origin { name: "PAULI_CHANNEL_1", a: q, b: None, pauli: (0, 0) };
                        self.add_disjoint(vec![0.0, *px, *pz, *py], &combos, origin, tag);
                    }
                }
                Ok((qx, qy, qz)) => {
                    for &q in qubits {
                        let qi = q as usize;
                        let combos = channel_combinations(&[self.t.sx[qi].clone(), self.t.sz[qi].clone()]);
                        for (k, pieces) in combos.into_iter().enumerate() {
                            let (pauli, prob) = [(1u8, qx), (2, qz), (3, qy)][k];
                            let origin = Origin { name: "PAULI_CHANNEL_1", a: q, b: None, pauli: (pauli, 0) };
                            self.add(prob, pieces, origin, tag);
                        }
                    }
                }
            },
            Instr::Depolarize2 { p, pairs } => {
                if *p > 15.0 / 16.0 {
                    return Err(format!("DEPOLARIZE2({p}) exceeds 15/16"));
                }
                let q2 = depolarize2_component(*p);
                for &(qa, qb) in pairs {
                    let (a, b) = (qa as usize, qb as usize);
                    let t = &self.t;
                    let basis = [t.sz[a].clone(), t.sx[a].clone(), t.sz[b].clone(), t.sx[b].clone()];
                    for (i, pieces) in channel_combinations(&basis).into_iter().enumerate() {
                        let k = i + 1;
                        let pa = ((k >> 1) & 1) as u8 | (((k & 1) as u8) << 1);
                        let pb = ((k >> 3) & 1) as u8 | ((((k >> 2) & 1) as u8) << 1);
                        let origin = Origin { name: "DEPOLARIZE2", a: qa, b: Some(qb), pauli: (pa, pb) };
                        self.add(q2, pieces, origin, tag);
                    }
                }
            }
            Instr::QubitCoords { .. } | Instr::Tick | Instr::Pauli { .. } => {}
        }
        Ok(())
    }

    /// The cases of one disjoint channel, approximated as independent faults as Stim
    /// approximates them: `by_combo[k]` is the probability of combination `k` of the channel's
    /// basis errors, `combos[k - 1]` its pieces. Cases that cannot be told apart (whose XOR fires
    /// nothing) are summed into the lowest-numbered, in Stim's order, and each combination is
    /// then recorded in turn.
    fn add_disjoint(&mut self, mut by_combo: Vec<f64>, combos: &[Vec<Sym>], origin: Origin, tag: &Option<Arc<str>>) {
        let n = by_combo.len();
        for k in 1..n {
            if combos[k - 1].iter().fold(Sym::new(), |acc, x| xor(&acc, x)).is_empty() {
                for dst in 0..n {
                    let src = dst ^ k;
                    if src > dst {
                        by_combo[dst] += by_combo[src];
                        by_combo[src] = 0.0;
                    }
                }
            }
        }
        for k in 1..n {
            self.add(by_combo[k], combos[k - 1].clone(), origin, tag);
        }
    }

    /// Stim's `run_loop`: find the loop's period by tortoise and hare, and fold whole periods
    /// into a `repeat` block.
    fn run_loop(&mut self, body: &[Instr], iterations: u64, loop_tag: &str, tag: &Option<Arc<str>>) -> Result<(), String> {
        if !self.fold {
            for _ in 0..iterations {
                self.undo_block(body, tag)?;
            }
            return Ok(());
        }
        let mut hare = Analyzer {
            t: self.t.clone(),
            accumulate: false,
            fold: true,
            decompose: false,
            approximate: self.approximate,
            window: Window::default(),
            reversed: Vec::new(),
            work: self.work,
            sweeps: None,
        };
        let (mut hare_iter, mut tortoise_iter) = (0u64, 0u64);
        while hare_iter < iterations {
            if hare.undo_block(body, tag).is_err() {
                // Abandon folding; the walk below meets the error itself.
                hare_iter = iterations;
                break;
            }
            hare.reversed.clear();
            hare_iter += 1;
            if hare.t.is_shifted_copy(&self.t) {
                break;
            }
            if hare_iter % 2 == 0 {
                self.undo_block(body, tag)?;
                tortoise_iter += 1;
                if hare.t.is_shifted_copy(&self.t) {
                    break;
                }
            }
        }
        self.work = self.work.max(hare.work);
        if hare_iter < iterations {
            let period = hare_iter - tortoise_iter;
            let period_iterations = (iterations - tortoise_iter) / period;
            let detectors_per_period = self.t.d - hare.t.d;
            let measurements_per_period = self.t.m - hare.t.m;
            // Don't bother folding a single iteration into a repeated block.
            if period_iterations > 1 {
                self.flush()?;
                let before = std::mem::take(&mut self.reversed);
                // The state as if the loop had run all but its first period.
                let skipped = period_iterations - 1;
                self.t.shift(
                    -((skipped as i128) * measurements_per_period as i128),
                    -((skipped as i128) * detectors_per_period as i128),
                );
                tortoise_iter += skipped * period;
                for _ in 0..period {
                    self.undo_block(body, tag)?;
                    tortoise_iter += 1;
                }
                self.flush()?;
                let mut folded = std::mem::take(&mut self.reversed);
                // The body ends (it begins, reversed) by shifting past its detectors.
                let remaining = detectors_per_period - total_shift(&folded);
                if remaining > 0 {
                    match folded.first_mut() {
                        Some(DemInstr::Shift { by, .. }) => *by += remaining,
                        _ => folded.insert(0, DemInstr::Shift { coords: Vec::new(), by: remaining, tag: String::new() }),
                    }
                }
                self.reversed = before;
                self.reversed.push(DemInstr::Repeat { count: period_iterations, body: folded, tag: loop_tag.to_string() });
            }
        }
        // The iterations left after whole periods.
        while tortoise_iter < iterations {
            self.undo_block(body, tag)?;
            tortoise_iter += 1;
        }
        Ok(())
    }

    /// Stim's `flush`: split the stretch's wide errors by the pieces it knows
    /// (`do_global_error_decomposition_pass`), and write its fault classes to the model in
    /// their sorted order (reversed here, as the model is).
    fn flush(&mut self) -> Result<(), String> {
        let mut classes = std::mem::take(&mut self.window).classes;
        for ((key, _), c) in &classes {
            let sym = components(key).fold(Sym::new(), |acc, x| xor(&acc, x));
            if !sym.is_empty() && sym.iter().all(|t| t & OBS != 0) {
                return Err(format!(
                    "{} flips observables {:#b} while firing no detector: an undetectable logical error",
                    c.origin.describe(),
                    obs_mask(&sym)
                ));
            }
        }
        if self.decompose && classes.keys().any(|(k, _)| !graphlike(k)) {
            // Every one- and two-detector piece of every class, in the classes' order: where two
            // share detectors, the later class's stands.
            let mut known: HashMap<Vec<u64>, Sym> = HashMap::new();
            for ((key, _), c) in &classes {
                if c.p == 0.0 {
                    continue;
                }
                for comp in components(key) {
                    let d = dets(comp);
                    if (1..=2).contains(&d.len()) {
                        known.insert(d, comp.to_vec());
                    }
                }
            }
            let mut rewrites = Vec::new();
            for ((key, tag), c) in &classes {
                if c.p == 0.0 || graphlike(key) {
                    continue;
                }
                let mut out = Vec::new();
                for comp in components(key) {
                    let parts = brute_force_known(comp, &known).or_else(|| greedy_known(comp, &known)).ok_or_else(|| {
                        format!("cannot split {} into graph-like pieces: it fires detectors {:?}", c.origin.describe(), dets(comp))
                    })?;
                    for part in parts {
                        if dets(&part).len() > 2 {
                            return Err(format!(
                                "cannot split {} into graph-like pieces: a piece fires detectors {:?}",
                                c.origin.describe(),
                                dets(&part)
                            ));
                        }
                        out.extend_from_slice(&part);
                        out.push(SEP);
                    }
                }
                out.pop();
                rewrites.push(((key.clone(), tag.clone()), out));
            }
            for (old, new) in rewrites {
                let c = classes.remove(&old).expect("a class just listed");
                let slot = classes.entry((new, old.1)).or_insert(Class { p: 0.0, origin: c.origin });
                slot.p = join(slot.p, c.p);
            }
        }
        for ((key, tag), c) in classes.into_iter().rev() {
            if c.p > 0.0 && !key.is_empty() {
                let pieces = components(&key).map(<[u64]>::to_vec).collect();
                self.reversed.push(DemInstr::Error { p: c.p, pieces, tag });
            }
        }
        Ok(())
    }
}

/// The detectors a reversed model's shifts move past, loops included.
fn total_shift(instrs: &[DemInstr]) -> u64 {
    instrs
        .iter()
        .map(|i| match i {
            DemInstr::Shift { by, .. } => *by,
            DemInstr::Repeat { count, body, .. } => count * total_shift(body),
            _ => 0,
        })
        .sum()
}

/// Stim's `unreversed`: the walk's model in the circuit's order, detectors relative to the
/// shifts before them, and declarations dropped that say nothing an error has not already.
fn unreversed(rev: &[DemInstr], base: &mut u64, seen: &mut HashSet<u64>) -> Vec<DemInstr> {
    let mut out = Vec::new();
    let relative = |t: u64, base: u64| if t & OBS == 0 { t - base } else { t };
    for e in rev.iter().rev() {
        match e {
            DemInstr::Shift { by, .. } => {
                *base += by;
                out.push(e.clone());
            }
            DemInstr::Error { p, pieces, tag } => {
                seen.extend(pieces.iter().flatten().copied());
                let pieces = pieces.iter().map(|x| x.iter().map(|&t| relative(t, *base)).collect()).collect();
                out.push(DemInstr::Error { p: *p, pieces, tag: tag.clone() });
            }
            DemInstr::Detector { coords, targets, tag } => {
                if !coords.is_empty() || !tag.is_empty() || !seen.contains(&targets[0]) {
                    let targets = targets.iter().map(|&t| relative(t, *base)).collect();
                    out.push(DemInstr::Detector { coords: coords.clone(), targets, tag: tag.clone() });
                }
            }
            DemInstr::Observable { targets, tag } => {
                if !tag.is_empty() || !seen.contains(&(OBS | u64::from(targets[0]))) {
                    out.push(e.clone());
                }
            }
            DemInstr::Repeat { count, body, tag } => {
                if *count > 0 {
                    let old = *base;
                    let body = unreversed(body, base, seen);
                    out.push(DemInstr::Repeat { count: *count, body, tag: tag.clone() });
                    let loop_shift = *base - old;
                    *base += loop_shift * (count - 1);
                }
            }
        }
    }
    out
}

/// A circuit's model as a program, its loops folded where they repeat (`fold`), or walked in
/// full. `decompose` splits faults into graph-like pieces as Stim does; `approximate` is Stim's
/// `approximate_disjoint_errors` threshold.
pub fn build(circuit: &Circuit, decompose: bool, approximate: Option<f64>, fold: bool) -> Result<DemProgram, String> {
    let counts = Counts::of(&circuit.instrs)?;
    let nq = counts.qubits;
    let mut a = Analyzer {
        t: Tracker { sx: vec![Sym::new(); nq], sz: vec![Sym::new(); nq], rec: BTreeMap::new(), m: counts.measurements, d: counts.detectors },
        accumulate: true,
        fold,
        decompose,
        approximate,
        window: Window::default(),
        reversed: Vec::new(),
        work: 0,
        sweeps: None,
    };
    a.undo_block(&circuit.instrs, &None)?;
    // Every qubit starts in |0>, which is a Z-basis reset at time zero.
    for q in 0..nq {
        a.check_reset(q, Basis::Z, "the initial |0>")?;
    }
    a.flush()?;
    let (mut base, mut seen) = (0u64, HashSet::new());
    Ok(DemProgram { instrs: unreversed(&a.reversed, &mut base, &mut seen) })
}

/// What each sweep bit's X flips, as (detectors, observables), and a check that every detector
/// and observable is deterministic: the backward walk, collecting no faults. Loops are folded
/// unless a sweep bit is read inside one, whose every pass counts.
pub fn sweep_effects(circuit: &Circuit) -> Result<Vec<(Vec<u32>, u64)>, String> {
    fn sweeps_in_loops(instrs: &[Instr], inside: bool) -> bool {
        instrs.iter().any(|i| match i {
            Instr::Repeat { body, .. } => sweeps_in_loops(body, true),
            Instr::Gate { body, .. } => sweeps_in_loops(body, inside),
            Instr::SweepX(_) | Instr::Feedback { control: Control::Sweep(_), .. } => inside,
            _ => false,
        })
    }
    let counts = Counts::of(&circuit.instrs)?;
    let nq = counts.qubits;
    let mut a = Analyzer {
        t: Tracker { sx: vec![Sym::new(); nq], sz: vec![Sym::new(); nq], rec: BTreeMap::new(), m: counts.measurements, d: counts.detectors },
        accumulate: false,
        fold: !sweeps_in_loops(&circuit.instrs, false),
        decompose: false,
        // Faults are not collected, so no channel needs approximating.
        approximate: Some(f64::INFINITY),
        window: Window::default(),
        reversed: Vec::new(),
        work: 0,
        sweeps: Some(vec![Sym::new(); counts.sweep_bits]),
    };
    a.undo_block(&circuit.instrs, &None)?;
    for q in 0..nq {
        a.check_reset(q, Basis::Z, "the initial |0>")?;
    }
    Ok(a.sweeps.unwrap_or_default().iter().map(|s| (dets(s).into_iter().map(|d| d as u32).collect(), obs_mask(s))).collect())
}

/* -- Decomposition --------------------------------------------------------- */
//
// Stim's decomposition, reproduced so that this engine's matching graph is the one PyMatching
// builds from Stim's model, edge for edge (see `dem.rs` for why it matters).

/// Stim's `decompose_helper_add_error_combinations`: the pieces of every combination k =
/// 1..2^s of a channel's basis errors, in order. A combination is split using only the
/// channel's own single-detector combinations and its irreducible two-detector ones; one that
/// cannot be is left whole for the flush.
fn channel_combinations(basis: &[Sym]) -> Vec<Vec<Sym>> {
    let s = basis.len();
    let n = 1usize << s;
    let mut sym = vec![Sym::new(); n];
    for (k, v) in sym.iter_mut().enumerate().skip(1) {
        for (i, b) in basis.iter().enumerate() {
            if (k >> i) & 1 == 1 {
                xor_into(v, b);
            }
        }
    }
    let mask: Vec<Sym> = sym.iter().map(|v| dets(v)).collect();
    let count: Vec<usize> = mask.iter().map(|m| m.len()).collect();

    let mut solved = vec![false; n];
    let mut single_union = Sym::new();
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

    let mut out: Vec<Vec<Sym>> = Vec::with_capacity(n - 1);
    for k in 1..n {
        if count[k] == 0 || solved[k] {
            out.push(vec![sym[k].clone()]);
            continue;
        }
        let goal = &mask[k];
        let mut pieces: Vec<Sym> = Vec::new();
        let mut remnants;
        if subset(goal, &single_union) {
            remnants = goal.clone();
        } else if let Some(&kp) =
            irreducible.iter().find(|&&kp| subset(&mask[kp], goal) && subset(goal, &or(&single_union, &mask[kp])))
        {
            pieces.push(sym[kp].clone());
            remnants = and_not(goal, &mask[kp]);
        } else {
            let mut found = None;
            'pairs: for (i1, &k1) in irreducible.iter().enumerate() {
                for &k2 in &irreducible[i1 + 1..] {
                    let both = or(&mask[k1], &mask[k2]);
                    if disjoint(&mask[k1], &mask[k2]) && subset(goal, &or(&single_union, &both)) {
                        found = Some((k1, k2, both));
                        break 'pairs;
                    }
                }
            }
            match found {
                Some((k1, k2, both)) => {
                    // Stim appends the pair whose targets sort first first.
                    let (k1, k2) = if sym[k2] < sym[k1] { (k2, k1) } else { (k1, k2) };
                    pieces.push(sym[k1].clone());
                    pieces.push(sym[k2].clone());
                    remnants = and_not(goal, &both);
                }
                None => {
                    pieces.push(sym[k].clone());
                    remnants = Sym::new();
                }
            }
        }
        for k2 in 1..n {
            if remnants.is_empty() {
                break;
            }
            if count[k2] == 1 && subset(&mask[k2], &remnants) {
                remnants = and_not(&remnants, &mask[k2]);
                pieces.push(sym[k2].clone());
            }
        }
        // Stim trusts this construction; check it. The pieces must XOR back to the
        // combination, observables included, or the model would be wrong.
        let mut check = Sym::new();
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

/// Stim's `brute_force_decomposition_into_known_graphlike_errors`: partition a wide piece's
/// detectors into known singles and pairs whose observables XOR to its own, trying the first
/// unused detector with each later one in turn, then alone.
fn brute_force_known(x: &[u64], known: &HashMap<Vec<u64>, Sym>) -> Option<Vec<Sym>> {
    let d = dets(x);
    fn go(d: &[u64], used: &mut [bool], remaining: u64, known: &HashMap<Vec<u64>, Sym>, out: &mut Vec<Sym>) -> bool {
        let Some(start) = (0..d.len()).find(|&i| !used[i]) else {
            return remaining == 0;
        };
        used[start] = true;
        for k in start + 1..=d.len() {
            let key = if k < d.len() {
                if used[k] {
                    continue;
                }
                used[k] = true;
                vec![d[start], d[k]]
            } else {
                vec![d[start]]
            };
            if let Some(m) = known.get(&key) {
                out.push(m.clone());
                if go(d, used, remaining ^ obs_mask(m), known, out) {
                    return true;
                }
                out.pop();
            }
            if k < d.len() {
                used[k] = false;
            }
        }
        used[start] = false;
        false
    }
    let mut used = vec![false; d.len()];
    let mut out = Vec::new();
    go(&d, &mut used, obs_mask(x), known, &mut out).then_some(out)
}

/// Stim's `decompose_and_append_component_to_tail`: greedily take known pairs, then known
/// singles, and let whatever is left (at most two detectors) stand as an edge of its own.
fn greedy_known(x: &[u64], known: &HashMap<Vec<u64>, Sym>) -> Option<Vec<Sym>> {
    let d = dets(x);
    if d.len() <= 2 {
        return Some(vec![x.to_vec()]);
    }
    let mut done = vec![false; d.len()];
    let mut rest = x.to_vec();
    let mut out = Vec::new();
    for k in 0..d.len() {
        if done[k] {
            continue;
        }
        for k2 in k + 1..d.len() {
            if done[k2] {
                continue;
            }
            if let Some(m) = known.get(&vec![d[k], d[k2]]) {
                done[k] = true;
                done[k2] = true;
                xor_into(&mut rest, m);
                out.push(m.clone());
                break;
            }
        }
    }
    let mut missed = 0;
    for k in 0..d.len() {
        if !done[k] {
            if let Some(m) = known.get(&vec![d[k]]) {
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
    if !rest.is_empty() {
        out.push(rest);
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dem::Dem;

    /// Folded and walked pass by pass, the model unrolls to the same faults.
    fn same_as_flat(text: &str) {
        let c = Circuit::parse(text).unwrap();
        for decompose in [false, true] {
            let old = build(&c, decompose, Some(1.0), false).and_then(|p| p.to_dem_merged());
            for fold in [true] {
                let new = build(&c, decompose, Some(1.0), fold).and_then(|p| p.to_dem_merged());
                match (&old, &new) {
                    (Ok(a), Ok(b)) => {
                        let norm = |d: &Dem| {
                            let mut v: Vec<(String, f64)> = d
                                .mechanisms
                                .iter()
                                .map(|m| (format!("{:?} {} {:?} {}", m.detectors, m.observables, m.pieces, m.tag), m.p))
                                .collect();
                            v.sort_by(|x, y| x.0.cmp(&y.0));
                            v
                        };
                        let (na, nb) = (norm(a), norm(b));
                        let keys = |v: &[(String, f64)]| v.iter().map(|x| x.0.clone()).collect::<Vec<_>>();
                        assert_eq!(keys(&na), keys(&nb), "decompose={decompose} fold={fold}\n{text}");
                        for (x, y) in na.iter().zip(&nb) {
                            assert!((x.1 - y.1).abs() <= 1e-12 * x.1.max(1e-300), "{} {} {}", x.0, x.1, y.1);
                        }
                        assert_eq!((a.num_detectors, a.num_observables), (b.num_detectors, b.num_observables));
                    }
                    (Err(_), Err(_)) => {}
                    (a, b) => panic!("decompose={decompose} fold={fold}: {:?} vs {:?}\n{text}", a.as_ref().err(), b.as_ref().err()),
                }
            }
        }
    }

    #[test]
    fn folding_agrees_with_walking_every_pass() {
        use crate::memory::{generate, CodeKind, NoiseModel};
        for kind in [CodeKind::Rotated, CodeKind::Xzzx] {
            for basis in [Basis::X, Basis::Z] {
                for (d, rounds) in [(3, 1), (3, 2), (3, 5), (5, 7)] {
                    let c = generate(kind, d, rounds, NoiseModel::Sd6 { p: 0.003 }, basis).unwrap();
                    same_as_flat(&c.to_stim());
                }
            }
        }
        same_as_flat(crate::fixtures::SAMPLE);
        same_as_flat(crate::fixtures::REP3);
    }
}

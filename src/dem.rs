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

use std::collections::{HashMap, HashSet};
use std::fmt::Write as _;

use crate::circuit::{fmt_args, split_instruction, Basis, Circuit, Instr};

#[derive(Clone, Debug, PartialEq, Default)]
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
    /// mechanism. Empty for a hyperedge nobody decomposed.
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

/// One single-qubit, single-type part of a fault. Kind 1 comes from an X
/// component, 2 from a Z component, 0 from a classical readout flip.
#[derive(Clone, PartialEq)]
struct Atom {
    sym: Vec<u64>,
    kind: u8,
}

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
    /// Up to four distinct ways the merged faults split into atoms; any one that
    /// decomposes will do.
    atom_options: Vec<Vec<Atom>>,
    origin: Origin,
}

struct Builder {
    space: Space,
    index: HashMap<Vec<u64>, usize>,
    entries: Vec<Entry>,
}

impl Builder {
    fn add(&mut self, p: f64, atoms: Vec<Atom>, origin: Origin) {
        if p <= 0.0 {
            return;
        }
        let mut sym = self.space.zero();
        for a in &atoms {
            xor_into(&mut sym, &a.sym);
        }
        if is_zero(&sym) {
            return;
        }
        match self.index.get(&sym) {
            Some(&i) => {
                let e = &mut self.entries[i];
                e.p = xor_prob(e.p, p);
                if e.atom_options.len() < 4 && !e.atom_options.contains(&atoms) {
                    e.atom_options.push(atoms);
                }
            }
            None => {
                self.index.insert(sym.clone(), self.entries.len());
                self.entries.push(Entry { sym, p, atom_options: vec![atoms], origin });
            }
        }
    }
}

fn atoms_for(pauli: u8, q: usize, sx: &[Vec<u64>], sz: &[Vec<u64>]) -> Vec<Atom> {
    let mut out = Vec::new();
    if pauli & 1 != 0 {
        out.push(Atom { sym: sx[q].clone(), kind: 1 });
    }
    if pauli & 2 != 0 {
        out.push(Atom { sym: sz[q].clone(), kind: 2 });
    }
    out
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
                            b.add(*flip, vec![Atom { sym: rec_sym[m].clone(), kind: 0 }], origin);
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
                Instr::PauliError { pauli, p, qubits } => {
                    for &q in qubits {
                        let origin = Origin { instr: idx, name: "Pauli error", a: q, b: None, pauli: (*pauli, 0) };
                        b.add(*p, atoms_for(*pauli, q as usize, &sx, &sz), origin);
                    }
                }
                Instr::Depolarize1 { p, qubits } => {
                    if *p > 0.75 {
                        return Err(format!("DEPOLARIZE1({p}) exceeds 3/4 (instruction {idx})"));
                    }
                    let q1 = depolarize1_component(*p);
                    for &q in qubits {
                        for pauli in [1u8, 3, 2] {
                            let origin = Origin { instr: idx, name: "DEPOLARIZE1", a: q, b: None, pauli: (pauli, 0) };
                            b.add(q1, atoms_for(pauli, q as usize, &sx, &sz), origin);
                        }
                    }
                }
                Instr::PauliChannel1 { px, py, pz, qubits } => {
                    let (qx, qy, qz) =
                        pauli_channel_1_independent(*px, *py, *pz).map_err(|e| format!("{e} (instruction {idx})"))?;
                    for &q in qubits {
                        for (pauli, prob) in [(1u8, qx), (3, qy), (2, qz)] {
                            let origin = Origin { instr: idx, name: "PAULI_CHANNEL_1", a: q, b: None, pauli: (pauli, 0) };
                            b.add(prob, atoms_for(pauli, q as usize, &sx, &sz), origin);
                        }
                    }
                }
                Instr::Depolarize2 { p, pairs } => {
                    if *p > 15.0 / 16.0 {
                        return Err(format!("DEPOLARIZE2({p}) exceeds 15/16 (instruction {idx})"));
                    }
                    let q2 = depolarize2_component(*p);
                    for &(qa, qb) in pairs {
                        for pa in 0..4u8 {
                            for pb in 0..4u8 {
                                if pa == 0 && pb == 0 {
                                    continue;
                                }
                                let mut atoms = atoms_for(pa, qa as usize, &sx, &sz);
                                atoms.extend(atoms_for(pb, qb as usize, &sx, &sz));
                                let origin =
                                    Origin { instr: idx, name: "DEPOLARIZE2", a: qa, b: Some(qb), pauli: (pa, pb) };
                                b.add(q2, atoms, origin);
                            }
                        }
                    }
                }
                Instr::Detector { .. }
                | Instr::Observable { .. }
                | Instr::QubitCoords { .. }
                | Instr::ShiftCoords(_)
                | Instr::Tick
                | Instr::Repeat { .. } => {}
            }
        }

        // Every qubit starts in |0>, which is a Z-basis reset at time zero.
        for q in 0..nq {
            check_reset(&sx, &sz, q, Basis::Z, "the initial |0>")?;
        }

        // Graph-like mechanisms, indexed by detector, for decomposition step 3.
        let mut graphlike: HashSet<Vec<u64>> = HashSet::new();
        let mut by_det: HashMap<u32, Vec<Vec<u64>>> = HashMap::new();
        for e in &b.entries {
            let (dets, obs) = space.split(&e.sym);
            if dets.is_empty() {
                return Err(format!(
                    "{} flips observables {obs:#b} while firing no detector: an undetectable logical error",
                    e.origin.describe()
                ));
            }
            if dets.len() <= 2 {
                graphlike.insert(e.sym.clone());
                for d in dets {
                    by_det.entry(d).or_default().push(e.sym.clone());
                }
            }
        }

        let mut mechanisms = Vec::with_capacity(b.entries.len());
        for e in &b.entries {
            let (detectors, observables) = space.split(&e.sym);
            let pieces = decompose(&space, e, &graphlike, &by_det).ok_or_else(|| {
                format!("cannot split {} into graph-like pieces: it fires detectors {:?}", e.origin.describe(), detectors)
            })?;
            mechanisms.push(Mechanism { p: e.p, detectors, observables, pieces });
        }
        mechanisms.sort_by(|a, b| a.detectors.cmp(&b.detectors).then(a.observables.cmp(&b.observables)));

        Ok(Dem { num_detectors: nd, num_observables: no, detector_coords: res.detector_coords, mechanisms })
    }
}

/* -- Decomposition --------------------------------------------------------- */

/// Parts to pieces, or None if any part is too wide or is a bare logical.
fn to_pieces(space: &Space, parts: &[Vec<u64>]) -> Option<Vec<Piece>> {
    let mut out = Vec::new();
    for part in parts {
        if is_zero(part) {
            continue;
        }
        let (detectors, observables) = space.split(part);
        if detectors.is_empty() || detectors.len() > 2 {
            return None;
        }
        out.push(Piece { detectors, observables });
    }
    Some(out)
}

/// Two existing graph-like mechanisms whose symptoms XOR to `s`.
fn split_existing(
    space: &Space,
    s: &[u64],
    graphlike: &HashSet<Vec<u64>>,
    by_det: &HashMap<u32, Vec<Vec<u64>>>,
) -> Option<(Vec<u64>, Vec<u64>)> {
    let (dets, _) = space.split(s);
    if dets.is_empty() || dets.len() > 4 {
        return None;
    }
    for g in by_det.get(&dets[0])? {
        let mut h = s.to_vec();
        xor_into(&mut h, g);
        if graphlike.contains(&h) {
            return Some((g.clone(), h));
        }
    }
    None
}

/// Split a mechanism into graph-like pieces, most natural split first:
/// 1. its X part and its Z part (propagation is linear, and a Y is X·Z);
/// 2. atom by atom (a two-qubit Pauli gives up to four single-type atoms);
/// 3. any part still too wide, into two graph-like mechanisms that exist.
fn decompose(
    space: &Space,
    e: &Entry,
    graphlike: &HashSet<Vec<u64>>,
    by_det: &HashMap<u32, Vec<Vec<u64>>>,
) -> Option<Vec<Piece>> {
    let (dets, _) = space.split(&e.sym);
    if dets.len() <= 2 {
        return to_pieces(space, std::slice::from_ref(&e.sym));
    }
    for atoms in &e.atom_options {
        let mut xpart = space.zero();
        let mut zpart = space.zero();
        let mut parts = Vec::new();
        for a in atoms {
            match a.kind {
                1 => xor_into(&mut xpart, &a.sym),
                2 => xor_into(&mut zpart, &a.sym),
                _ => parts.push(a.sym.clone()),
            }
        }
        parts.push(xpart);
        parts.push(zpart);
        if let Some(p) = to_pieces(space, &parts) {
            return Some(p);
        }

        let mut singles: Vec<Vec<u64>> = Vec::new();
        for a in atoms {
            match singles.iter().position(|s| *s == a.sym) {
                Some(i) => {
                    singles.swap_remove(i);
                }
                None => singles.push(a.sym.clone()),
            }
        }
        if let Some(p) = to_pieces(space, &singles) {
            return Some(p);
        }

        let mut out = Vec::new();
        let mut ok = true;
        for s in &singles {
            if is_zero(s) {
                continue;
            }
            if let Some(mut p) = to_pieces(space, std::slice::from_ref(s)) {
                out.append(&mut p);
                continue;
            }
            match split_existing(space, s, graphlike, by_det).and_then(|(g, h)| to_pieces(space, &[g, h])) {
                Some(mut p) => out.append(&mut p),
                None => {
                    ok = false;
                    break;
                }
            }
        }
        if ok {
            return Some(out);
        }
    }
    split_existing(space, &e.sym, graphlike, by_det).and_then(|(g, h)| to_pieces(space, &[g, h]))
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
}

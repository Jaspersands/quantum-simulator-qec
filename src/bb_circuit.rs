//! Circuits on bivariate bicycle codes beyond the memory: a writer that runs
//! any checks on any schedule of ticks, with lattice surgery's detector rule,
//! and on it the gauging measurement of a logical operator (`bb_gauge`).
//!
//! WHY THIS EXISTS
//! ---------------
//! `BbCode::memory_z` writes the paper's memory as text, one schedule, one
//! basis. A logical measurement changes the code for a while: new qubits,
//! new checks, checks that gain a qubit and later lose it. The writer here
//! takes each cycle as a list of checks and ticks, and decides detectors by
//! one rule, the one `surgery.rs` proved for lattice surgery: a check
//! compares with its measurement in the previous cycle; the qubits it gained
//! since were freshly prepared in its basis, the qubits it lost were
//! measured in its basis, and their records join. Its plain memory equals
//! `memory_z`'s error model fault for fault, which is what makes its
//! logical measurement trustworthy.

use std::collections::HashMap;

use crate::bb::{BbCode, SX, SZ};
use crate::bb_gauge::Gauging;
use crate::circuit::{Basis, Circuit, Instr};

/// A check a cycle measures: its ancilla, its type, the qubits it reads in
/// CNOT order, its position (for detector coordinates), and the key that
/// matches it with its own measurements in other cycles.
#[derive(Clone, Debug)]
pub struct Check {
    pub key: usize,
    pub anc: u32,
    pub x_type: bool,
    pub support: Vec<u32>,
    pub pos: (f64, f64),
}

/// One tick. In order: Z-type ancillas measured; X-type ancillas prepared;
/// the CNOTs (check index, qubit); idle noise on data not in a CNOT; X-type
/// ancillas measured; Z-type ancillas prepared for the next cycle.
#[derive(Clone, Debug, Default)]
pub struct Tick {
    pub measure_z: Vec<usize>,
    pub prepare_x: Vec<usize>,
    pub cnots: Vec<(usize, u32)>,
    pub measure_x: Vec<usize>,
    pub prepare_z: Vec<usize>,
}

/// One syndrome cycle: the checks it measures and its ticks.
#[derive(Clone, Debug)]
pub struct Cycle {
    pub checks: Vec<Check>,
    pub ticks: Vec<Tick>,
}

struct Last {
    rec: usize,
    support: Vec<u32>,
    cycle: usize,
}

/// Writes instructions, keeps the measurement count, and decides detectors.
pub struct Writer {
    pub c: Vec<Instr>,
    m: usize,
    p: f64,
    /// Which checks give detectors: those of this basis's type.
    detect: Basis,
    /// Data qubits in play, which idle noise reaches.
    data: Vec<u32>,
    last: HashMap<usize, Last>,
    /// Qubits prepared since the last cycle ended, with their basis.
    fresh: HashMap<u32, Basis>,
    /// Qubits measured, with their latest record and basis.
    measured: HashMap<u32, (usize, Basis)>,
    cycle: usize,
}

fn flip(basis: Basis) -> u8 {
    // An X error flips a Z-basis state or reading, a Z error an X-basis one.
    if basis == Basis::Z {
        1
    } else {
        2
    }
}

impl Writer {
    pub fn new(p: f64, detect: Basis, data: Vec<u32>) -> Writer {
        Writer { c: Vec::new(), m: 0, p, detect, data, last: HashMap::new(), fresh: HashMap::new(), measured: HashMap::new(), cycle: 0 }
    }

    fn lookback(&self, rec: usize) -> u32 {
        (self.m - rec) as u32
    }

    fn noise(&mut self, instr: Instr) {
        if self.p > 0.0 {
            self.c.push(instr);
        }
    }

    /// Prepare qubits in `basis`, with a flip after, and mark them fresh.
    pub fn prepare(&mut self, basis: Basis, qubits: &[u32]) {
        if qubits.is_empty() {
            return;
        }
        self.c.push(Instr::Reset { basis, qubits: qubits.to_vec() });
        let p = self.p;
        self.noise(Instr::PauliError { pauli: flip(basis), p, qubits: qubits.to_vec() });
        for &q in qubits {
            self.fresh.insert(q, basis);
        }
    }

    /// Prepare Z-type ancillas in |0⟩ (not data, so not fresh).
    pub fn prepare_ancillas_z(&mut self, qubits: &[u32]) {
        if qubits.is_empty() {
            return;
        }
        self.c.push(Instr::Reset { basis: Basis::Z, qubits: qubits.to_vec() });
        let p = self.p;
        self.noise(Instr::PauliError { pauli: 1, p, qubits: qubits.to_vec() });
    }

    /// Measure qubits in `basis`, with a flip before; their records, in order.
    pub fn measure(&mut self, basis: Basis, qubits: &[u32]) -> Vec<usize> {
        let p = self.p;
        self.noise(Instr::PauliError { pauli: flip(basis), p, qubits: qubits.to_vec() });
        self.c.push(Instr::Measure { basis, reset: false, flip: 0.0, qubits: qubits.to_vec() });
        let recs: Vec<usize> = (self.m..self.m + qubits.len()).collect();
        self.m += qubits.len();
        for (&q, &r) in qubits.iter().zip(&recs) {
            self.measured.insert(q, (r, basis));
        }
        recs
    }

    pub fn add_data(&mut self, qubits: &[u32]) {
        self.data.extend_from_slice(qubits);
    }

    pub fn remove_data(&mut self, qubits: &[u32]) {
        self.data.retain(|q| !qubits.contains(q));
    }

    pub fn observable(&mut self, index: u32, recs: &[usize]) {
        let recs = recs.iter().map(|&r| self.lookback(r)).collect();
        self.c.push(Instr::Observable { index, recs, paulis: Vec::new() });
    }

    /// Measure checks' ancillas in the checks' basis, and give each its detector.
    fn read(&mut self, cyc: &Cycle, which: &[usize], basis: Basis, out: &mut [Option<usize>]) {
        if which.is_empty() {
            return;
        }
        let ancs: Vec<u32> = which.iter().map(|&i| cyc.checks[i].anc).collect();
        let p = self.p;
        self.noise(Instr::PauliError { pauli: flip(basis), p, qubits: ancs.clone() });
        self.c.push(Instr::Measure { basis, reset: false, flip: 0.0, qubits: ancs });
        // The whole measurement first, so every detector's lookbacks count from its end.
        let first = self.m;
        self.m += which.len();
        for (k, &i) in which.iter().enumerate() {
            out[i] = Some(first + k);
            self.detect_check(&cyc.checks[i], first + k);
        }
    }

    /// The detector rule (as `surgery.rs`'s Writer::round, on cycles).
    fn detect_check(&mut self, ch: &Check, rec: usize) {
        let basis = if ch.x_type { Basis::X } else { Basis::Z };
        if basis == self.detect {
            let mut targets = vec![rec];
            let deterministic = match self.last.get(&ch.key).filter(|prev| prev.cycle + 1 == self.cycle) {
                Some(prev) => {
                    targets.push(prev.rec);
                    let read_since = |q: &u32| match self.measured.get(q) {
                        Some(&(r, b)) if b == basis && r > prev.rec => Some(r),
                        _ => None,
                    };
                    let mut ok = true;
                    for q in ch.support.iter().filter(|q| self.fresh.contains_key(q)) {
                        ok &= self.fresh.get(q) == Some(&basis);
                        if prev.support.contains(q) {
                            match read_since(q) {
                                Some(r) => targets.push(r),
                                None => ok = false,
                            }
                        }
                    }
                    ok &= ch.support.iter().filter(|q| !prev.support.contains(q)).all(|q| self.fresh.contains_key(q));
                    for q in prev.support.iter().filter(|q| !ch.support.contains(q)) {
                        match read_since(q) {
                            Some(r) => targets.push(r),
                            None => ok = false,
                        }
                    }
                    ok
                }
                None => ch.support.iter().all(|q| self.fresh.get(q) == Some(&basis)),
            };
            if deterministic {
                let recs = targets.iter().map(|&r| self.lookback(r)).collect();
                self.c.push(Instr::Detector { coords: vec![ch.pos.0, ch.pos.1, self.cycle as f64], recs });
            }
        }
        self.last.insert(ch.key, Last { rec, support: ch.support.clone(), cycle: self.cycle });
    }

    /// One cycle; returns each check's record.
    pub fn cycle(&mut self, cyc: &Cycle) -> Vec<Option<usize>> {
        let mut out = vec![None; cyc.checks.len()];
        let p = self.p;
        for tick in &cyc.ticks {
            self.c.push(Instr::Tick);
            self.read(cyc, &tick.measure_z, Basis::Z, &mut out);
            if !tick.prepare_x.is_empty() {
                let ancs: Vec<u32> = tick.prepare_x.iter().map(|&i| cyc.checks[i].anc).collect();
                self.c.push(Instr::Reset { basis: Basis::X, qubits: ancs.clone() });
                self.noise(Instr::PauliError { pauli: 2, p, qubits: ancs });
            }
            let pairs: Vec<(u32, u32)> = tick
                .cnots
                .iter()
                .map(|&(i, q)| {
                    let ch = &cyc.checks[i];
                    if ch.x_type {
                        (ch.anc, q)
                    } else {
                        (q, ch.anc)
                    }
                })
                .collect();
            if !pairs.is_empty() {
                self.c.push(Instr::Cx(pairs.clone()));
                self.noise(Instr::Depolarize2 { p, pairs });
            }
            let busy: Vec<u32> = tick.cnots.iter().map(|&(_, q)| q).collect();
            let idle: Vec<u32> = self.data.iter().copied().filter(|q| !busy.contains(q)).collect();
            if !idle.is_empty() {
                self.noise(Instr::Depolarize1 { p, qubits: idle });
            }
            self.read(cyc, &tick.measure_x, Basis::X, &mut out);
            let ancs: Vec<u32> = tick.prepare_z.iter().map(|&i| cyc.checks[i].anc).collect();
            self.prepare_ancillas_z(&ancs);
        }
        self.fresh.clear();
        self.cycle += 1;
        out
    }

    /// Checks that end here, their whole support just read out in their own
    /// basis (a flux check when its edge qubits are read in Z at the split):
    /// each of `detect`'s type measured in the last cycle is compared with
    /// the parity of those readings.
    pub fn retire(&mut self, checks: &[Check]) {
        for ch in checks {
            let basis = if ch.x_type { Basis::X } else { Basis::Z };
            if basis != self.detect {
                continue;
            }
            let Some(last) = self.last.get(&ch.key).filter(|l| l.cycle + 1 == self.cycle) else { continue };
            let reads: Option<Vec<usize>> = ch
                .support
                .iter()
                .map(|q| match self.measured.get(q) {
                    Some(&(r, b)) if b == basis && r > last.rec => Some(r),
                    _ => None,
                })
                .collect();
            if let Some(mut targets) = reads {
                targets.push(last.rec);
                let recs = targets.iter().map(|&r| self.lookback(r)).collect();
                self.c.push(Instr::Detector { coords: vec![ch.pos.0, ch.pos.1, self.cycle as f64], recs });
            }
        }
    }

    /// After the data are read out in `basis` (`readout`: qubit to record),
    /// each check of that type in `cyc` measured in the last cycle is
    /// compared with the parity of its support, and with the records of any
    /// qubit its last measurement read that it does not (read in its basis
    /// since).
    pub fn final_detectors(&mut self, cyc: &Cycle, basis: Basis, readout: &HashMap<u32, usize>) -> Result<(), String> {
        let want_x = basis == Basis::X;
        for ch in cyc.checks.iter().filter(|c| c.x_type == want_x) {
            let Some(last) = self.last.get(&ch.key).filter(|l| l.cycle + 1 == self.cycle) else { continue };
            let mut targets: Vec<usize> = Vec::new();
            for q in &ch.support {
                targets.push(*readout.get(q).ok_or(format!("qubit {q} of check {} was not read out", ch.key))?);
            }
            targets.push(last.rec);
            for q in last.support.iter().filter(|q| !ch.support.contains(q)) {
                match self.measured.get(q) {
                    Some(&(r, b)) if b == basis && r > last.rec => targets.push(r),
                    _ => return Err(format!("check {} lost qubit {q}, not read in its basis", ch.key)),
                }
            }
            let recs = targets.iter().map(|&r| self.lookback(r)).collect();
            self.c.push(Instr::Detector { coords: vec![ch.pos.0, ch.pos.1, self.cycle as f64], recs });
        }
        Ok(())
    }
}

/// `memory_z`'s qubits: X ancillas 0..h, left data h..2h, right data 2h..3h,
/// Z ancillas 3h..4h; cell (u, v) draws them at (2u, 2v), (2u + 1, 2v),
/// (2u, 2v + 1), (2u + 1, 2v + 1).
fn memory_coords(code: &BbCode, w: &mut Writer) {
    let h = code.half();
    for c in 0..h {
        let (u, v) = ((c / code.m) as f64, (c % code.m) as f64);
        for (q, x, y) in [(c, 2.0 * u, 2.0 * v), (h + c, 2.0 * u + 1.0, 2.0 * v), (2 * h + c, 2.0 * u, 2.0 * v + 1.0), (3 * h + c, 2.0 * u + 1.0, 2.0 * v + 1.0)] {
            w.c.push(Instr::QubitCoords { coords: vec![x, y], qubits: vec![q as u32] });
        }
    }
}

/// The code's checks: X checks (index and key c), then Z checks (index and
/// key h + c), each reading its six neighbours in `neighbours` order.
fn code_checks(code: &BbCode) -> Vec<Check> {
    let h = code.half();
    let pos = |c: usize, dx: f64, dy: f64| (2.0 * (c / code.m) as f64 + dx, 2.0 * (c % code.m) as f64 + dy);
    let mut checks = Vec::with_capacity(2 * h);
    for c in 0..h {
        let support = code.neighbours(c, true).iter().map(|&q| (h + q) as u32).collect();
        checks.push(Check { key: c, anc: c as u32, x_type: true, support, pos: pos(c, 0.0, 0.0) });
    }
    for c in 0..h {
        let support = code.neighbours(c, false).iter().map(|&q| (h + q) as u32).collect();
        checks.push(Check { key: h + c, anc: (3 * h + c) as u32, x_type: false, support, pos: pos(c, 1.0, 1.0) });
    }
    checks
}

/// The paper's depth-8 cycle: X ancillas prepared at tick 0 and read after
/// tick 7; Z checks meet their neighbours in ticks 0-5 and are read at tick
/// 6; X checks in ticks 1-6.
pub fn memory_cycle(code: &BbCode) -> Cycle {
    let h = code.half();
    let checks = code_checks(code);
    let mut ticks = vec![Tick::default(); 8];
    ticks[0].prepare_x = (0..h).collect();
    for (t, tick) in ticks.iter_mut().enumerate().take(7) {
        if let Some(k) = SX[t] {
            tick.cnots.extend((0..h).map(|c| (c, checks[c].support[k])));
        }
        if let Some(k) = SZ[t] {
            tick.cnots.extend((0..h).map(|c| (h + c, checks[h + c].support[k])));
        }
    }
    ticks[6].measure_z = (h..2 * h).collect();
    ticks[7].measure_x = (0..h).collect();
    ticks[7].prepare_z = (h..2 * h).collect();
    Cycle { checks, ticks }
}

/// A memory in either basis: the data prepared in it, `cycles` cycles, the
/// data read in it; detectors on the checks of its type; the logical
/// operators of its type as observables. In Z it is the paper's memory.
pub fn memory(code: &BbCode, basis: Basis, cycles: usize, p: f64) -> Circuit {
    let h = code.half() as u32;
    let data: Vec<u32> = (h..3 * h).collect();
    let mut w = Writer::new(p, basis, data.clone());
    memory_coords(code, &mut w);
    w.prepare(basis, &data);
    w.prepare_ancillas_z(&(3 * h..4 * h).collect::<Vec<_>>());
    let cyc = memory_cycle(code);
    for _ in 0..cycles {
        w.cycle(&cyc);
    }
    w.c.push(Instr::Tick);
    let recs = w.measure(basis, &data);
    let readout: HashMap<u32, usize> = data.iter().copied().zip(recs).collect();
    w.final_detectors(&cyc, basis, &readout).expect("a memory's checks keep their support");
    let (lx, lz) = code.logicals();
    let logicals = if basis == Basis::X { lx } else { lz };
    for k in 0..logicals.rows {
        let recs: Vec<usize> = logicals.row_ones(k).iter().map(|&q| readout[&(h + q as u32)]).collect();
        w.observable(k as u32, &recs);
    }
    Circuit { instrs: w.c }
}

/// Qubits the gauging adds after `memory_z`'s 4h: edge qubits, then the
/// Gauss-law ancillas, then the flux ancillas.
struct Added {
    base: u32,
    edges: u32,
    vertices: u32,
}

impl Added {
    fn new(code: &BbCode, g: &Gauging) -> Added {
        Added { base: 4 * code.half() as u32, edges: g.num_edges() as u32, vertices: g.support.len() as u32 }
    }
    fn edge(&self, i: usize) -> u32 {
        self.base + i as u32
    }
    fn gauss(&self, v: usize) -> u32 {
        self.base + self.edges + v as u32
    }
    fn flux(&self, j: usize) -> u32 {
        self.base + self.edges + self.vertices + j as u32
    }
}

/// Where a data qubit is drawn (left data of cell (u, v) at (2u + 1, 2v), right at (2u, 2v + 1)).
fn data_pos(code: &BbCode, d: usize) -> (f64, f64) {
    let h = code.half();
    let (u, v) = (((d % h) / code.m) as f64, ((d % h) % code.m) as f64);
    if d < h {
        (2.0 * u + 1.0, 2.0 * v)
    } else {
        (2.0 * u, 2.0 * v + 1.0)
    }
}

/// Place each (check, qubit) CNOT at the first tick from `from` where
/// neither the check's ancilla nor the qubit has one yet; ticks are added as
/// needed. Returns the last tick used, if any.
fn first_fit(ticks: &mut Vec<Tick>, checks: &[Check], pairs: &[(usize, u32)], from: usize) -> Option<usize> {
    let mut last = None;
    for &(ci, q) in pairs {
        let mut t = from;
        loop {
            if ticks.len() <= t {
                ticks.resize(t + 1, Tick::default());
            }
            let clash = ticks[t].cnots.iter().any(|&(c2, q2)| q2 == q || checks[c2].anc == checks[ci].anc);
            if !clash {
                ticks[t].cnots.push((ci, q));
                last = Some(last.map_or(t, |l: usize| l.max(t)));
                break;
            }
            t += 1;
        }
    }
    last
}

/// A cycle of the deformed code. Checks: the code's X checks (0..h), its Z
/// checks (h..2h; those of C₀ read their edge qubit last), the Gauss-law
/// checks (2h + v), the flux checks (2h + V + j). Ticks 0-6 are the paper's
/// cycle, and alongside them the flux checks read their edge qubits (from
/// tick 0) and each C₀ check reads its edge qubit (tick 6); each Gauss-law
/// check reads its data qubit from tick 7; every Z-type check is read out;
/// then the Gauss-law checks read their edges.
///
/// That order is what makes the checks measurable together: on every qubit
/// an X check and a Z check share, the Z check acts first (data: ticks 0-5
/// against 7 on; edges: tick 6 and the flux ticks against the Gauss ticks
/// after them), so each pair's CNOTs commute past each other as a whole.
/// The determinism tests are the proof.
pub fn merged_cycle(code: &BbCode, g: &Gauging) -> Cycle {
    let h = code.half();
    let nv = g.support.len();
    let q = Added::new(code, g);
    let mut checks = code_checks(code);
    for (i, &c) in g.edges.iter().enumerate() {
        checks[h + c].support.push(q.edge(i));
    }
    for (v, es) in g.gauss.iter().enumerate() {
        let d = g.support[v];
        let (x, y) = data_pos(code, d);
        let support = std::iter::once((h + d) as u32).chain(es.iter().map(|&e| q.edge(e))).collect();
        checks.push(Check { key: 2 * h + v, anc: q.gauss(v), x_type: true, support, pos: (x + 0.25, y + 0.25) });
    }
    for (j, cycle) in g.flux.iter().enumerate() {
        let support = cycle.iter().map(|&e| q.edge(e)).collect();
        checks.push(Check { key: 2 * h + nv + j, anc: q.flux(j), x_type: false, support, pos: (-1.0 - j as f64, -1.0) });
    }
    let gauss_checks: Vec<usize> = (2 * h..2 * h + nv).collect();
    let z_type: Vec<usize> = (h..2 * h).chain(2 * h + nv..checks.len()).collect();

    let mut ticks = vec![Tick::default(); 8];
    ticks[0].prepare_x = (0..h).chain(gauss_checks.iter().copied()).collect();
    for (t, tick) in ticks.iter_mut().enumerate().take(7) {
        if let Some(k) = SX[t] {
            tick.cnots.extend((0..h).map(|c| (c, checks[c].support[k])));
        }
        if let Some(k) = SZ[t] {
            tick.cnots.extend((0..h).map(|c| (h + c, checks[h + c].support[k])));
        }
    }
    ticks[6].cnots.extend(g.edges.iter().enumerate().map(|(i, &c)| (h + c, q.edge(i))));
    ticks[7].measure_x = (0..h).collect();
    let mut early: Vec<(usize, u32)> = Vec::new();
    for (j, cycle) in g.flux.iter().enumerate() {
        early.extend(cycle.iter().map(|&e| (2 * h + nv + j, q.edge(e))));
    }
    early.extend((0..nv).map(|v| (2 * h + v, (h + g.support[v]) as u32)));
    let flux_pairs = g.flux.iter().map(Vec::len).sum::<usize>();
    // Flux checks touch only edge qubits, idle until tick 6, so they start
    // at tick 0; Z-type checks need no order among themselves.
    let last_flux = first_fit(&mut ticks, &checks, &early[..flux_pairs], 0);
    first_fit(&mut ticks, &checks, &early[flux_pairs..], 7);
    let read_z = last_flux.map_or(7, |t| (t + 1).max(7));
    if ticks.len() <= read_z {
        ticks.resize(read_z + 1, Tick::default());
    }
    ticks[read_z].measure_z = z_type.clone();
    let mut late: Vec<(usize, u32)> = Vec::new();
    for (v, es) in g.gauss.iter().enumerate() {
        late.extend(es.iter().map(|&e| (2 * h + v, q.edge(e))));
    }
    let last = first_fit(&mut ticks, &checks, &late, read_z).unwrap_or(read_z).max(7);
    ticks.truncate(last + 1);
    ticks[last].measure_x.extend(gauss_checks);
    ticks[last].prepare_z = z_type;
    Cycle { checks, ticks }
}

/// Measure the logical X operator `g` gauges. The data are prepared in
/// `basis` and held for `pre` memory cycles; the edge qubits are prepared in
/// |0⟩ and the deformed code measured for `merged` cycles; the edge qubits
/// are read in Z; `post` memory cycles follow and the data are read in
/// `basis`. Detectors are the checks of `basis`'s type.
///
/// Observables. X basis: L0 the outcome, the product of the Gauss-law
/// checks in the first merged cycle, which with the data in |+⟩ is +1; then
/// the code's 12 X logicals, which the measurement keeps. Z basis: the 11 Z
/// logicals that commute with the operator (the one that does not is made
/// random), each with its route through the edges read at the split.
pub fn logical_measurement(code: &BbCode, g: &Gauging, basis: Basis, pre: usize, merged: usize, post: usize, p: f64) -> Result<Circuit, String> {
    if merged == 0 {
        return Err("a logical measurement needs at least one merged cycle".into());
    }
    let h = code.half();
    let q = Added::new(code, g);
    let data: Vec<u32> = (h as u32..3 * h as u32).collect();
    let mut w = Writer::new(p, basis, data.clone());
    memory_coords(code, &mut w);
    let hz_pos = |c: usize| (2.0 * (c / code.m) as f64 + 1.0, 2.0 * (c % code.m) as f64 + 1.0);
    for (i, &c) in g.edges.iter().enumerate() {
        let (x, y) = hz_pos(c);
        w.c.push(Instr::QubitCoords { coords: vec![x + 0.5, y + 0.5], qubits: vec![q.edge(i)] });
    }
    // An added edge sits between its two vertices (as drawn, not across the torus).
    for (k, &(a, b)) in g.extra.iter().enumerate() {
        let ((xa, ya), (xb, yb)) = (data_pos(code, g.support[a]), data_pos(code, g.support[b]));
        w.c.push(Instr::QubitCoords { coords: vec![(xa + xb) / 2.0, (ya + yb) / 2.0], qubits: vec![q.edge(g.edges.len() + k)] });
    }
    for (v, &d) in g.support.iter().enumerate() {
        let (x, y) = data_pos(code, d);
        w.c.push(Instr::QubitCoords { coords: vec![x + 0.25, y + 0.25], qubits: vec![q.gauss(v)] });
    }
    for j in 0..g.flux.len() {
        w.c.push(Instr::QubitCoords { coords: vec![-1.0 - j as f64, -1.0], qubits: vec![q.flux(j)] });
    }

    w.prepare(basis, &data);
    w.prepare_ancillas_z(&(3 * h as u32..4 * h as u32).collect::<Vec<_>>());
    let memory = memory_cycle(code);
    let merged_c = merged_cycle(code, g);
    for _ in 0..pre {
        w.cycle(&memory);
    }
    w.c.push(Instr::Tick);
    let edges: Vec<u32> = (0..g.num_edges()).map(|i| q.edge(i)).collect();
    w.prepare(Basis::Z, &edges);
    w.prepare_ancillas_z(&(0..g.flux.len()).map(|j| q.flux(j)).collect::<Vec<_>>());
    w.add_data(&edges);
    let mut outcome = Vec::new();
    for k in 0..merged {
        let recs = w.cycle(&merged_c);
        if k == 0 {
            outcome = (0..g.support.len()).map(|v| recs[2 * h + v].expect("every Gauss-law check is read")).collect();
        }
    }
    w.c.push(Instr::Tick);
    let split = w.measure(Basis::Z, &edges);
    // The flux checks end with their edges read: their last value is those readings' parity.
    w.retire(&merged_c.checks[2 * h + g.support.len()..]);
    w.remove_data(&edges);
    for _ in 0..post {
        w.cycle(&memory);
    }
    w.c.push(Instr::Tick);
    let recs = w.measure(basis, &data);
    let readout: HashMap<u32, usize> = data.iter().copied().zip(recs).collect();
    w.final_detectors(&memory, basis, &readout)?;

    let on_data = |support: &[usize]| support.iter().map(|&d| readout[&((h + d) as u32)]).collect::<Vec<usize>>();
    let (lx, lz) = code.logicals();
    match basis {
        Basis::X => {
            w.observable(0, &outcome);
            for k in 0..lx.rows {
                w.observable(1 + k as u32, &on_data(&lx.row_ones(k)));
            }
        }
        Basis::Z => {
            let odd = |z: &[usize]| z.iter().filter(|d| g.support.contains(d)).count() % 2 == 1;
            let rows: Vec<Vec<usize>> = (0..lz.rows).map(|k| lz.row_ones(k)).collect();
            let pivot = rows.iter().position(|z| odd(z)).ok_or("every Z logical commutes with the operator: it is a stabilizer")?;
            for (index, (k, z)) in rows.iter().enumerate().filter(|&(k, _)| k != pivot).enumerate() {
                let z: Vec<usize> = if odd(z) {
                    let mut on = vec![false; 2 * h];
                    for &d in z.iter().chain(&rows[pivot]) {
                        on[d] ^= true;
                    }
                    (0..2 * h).filter(|&d| on[d]).collect()
                } else {
                    z.clone()
                };
                let route = g.edges_for(&z).map_err(|e| format!("Z logical {k}: {e}"))?;
                let mut recs = on_data(&z);
                recs.extend(route.iter().map(|&e| split[e]));
                w.observable(index as u32, &recs);
            }
        }
    }
    Ok(Circuit { instrs: w.c })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::circuit::Circuit;
    use crate::dem::Dem;
    use crate::m2d::M2d;

    fn mechanisms(c: &Circuit) -> Vec<(Vec<u32>, u64, f64)> {
        let dem = Dem::from_circuit_undecomposed(c).unwrap();
        let mut v: Vec<_> = dem.mechanisms.iter().map(|m| (m.detectors.clone(), m.observables, m.p)).collect();
        v.sort_by(|a, b| (&a.0, a.1).cmp(&(&b.0, b.1)));
        v
    }

    /// The writer's Z-basis memory is the paper's, fault for fault: the same
    /// detectors in the same order, the same observables, the same priors.
    #[test]
    fn the_writers_memory_is_the_papers() {
        for code in [BbCode::bb72(), BbCode::gross()] {
            let ours = memory(&code, Basis::Z, 3, 0.001);
            let theirs = Circuit::parse(&code.memory_z(3, 0.001)).unwrap();
            let (a, b) = (mechanisms(&ours), mechanisms(&theirs));
            assert_eq!(a.len(), b.len(), "{}×{}", code.l, code.m);
            for (x, y) in a.iter().zip(&b) {
                assert_eq!((&x.0, x.1), (&y.0, y.1));
                assert!((x.2 - y.2).abs() <= 1e-12 * y.2, "{x:?} vs {y:?}");
            }
        }
    }

    /// The X-basis memory is deterministic without noise, with X detectors
    /// from the first cycle and the 12 X logicals as observables.
    #[test]
    fn the_x_basis_memory_is_deterministic() {
        let code = BbCode::gross();
        let m = M2d::new(&memory(&code, Basis::X, 3, 0.0)).unwrap();
        assert_eq!((m.num_detectors, m.num_observables), (code.half() * 4, 12));
    }
    fn gauged(name: &str) -> (BbCode, Gauging) {
        let code = BbCode::gross();
        let g = Gauging::new(&code, &crate::bb::gross_operator(name).unwrap()).unwrap();
        (code, g)
    }

    /// Each tick of the merged cycle uses every qubit at most once.
    #[test]
    fn the_merged_cycle_is_a_schedule() {
        for name in ["f", "f+gh"] {
            let (code, g) = gauged(name);
            let cyc = merged_cycle(&code, &g);
            eprintln!("{name}: {} ticks", cyc.ticks.len());
            for (t, tick) in cyc.ticks.iter().enumerate() {
                let mut seen = std::collections::HashSet::new();
                for &(ch, q) in &tick.cnots {
                    assert!(seen.insert(cyc.checks[ch].anc), "{name}, tick {t}: an ancilla twice");
                    assert!(seen.insert(q), "{name}, tick {t}: qubit {q} twice");
                }
            }
        }
    }

    /// Without noise every detector and observable is deterministic, in both
    /// bases, for all three operators and both constructions: the schedule
    /// measures what it should, the outcome is X̄, and each Z logical's route
    /// through the edges (added edges included) is right.
    #[test]
    fn the_logical_measurement_is_deterministic() {
        let code = BbCode::gross();
        let systems = ["f", "gh", "f+gh"].into_iter().flat_map(|name| {
            let l = crate::bb::gross_operator(name).unwrap();
            [(name, Gauging::new(&code, &l).unwrap()), (name, Gauging::expanded(&code, &l).unwrap())]
        });
        for (name, g) in systems {
            let name = format!("{name} (+{} edges)", g.extra.len());
            let code = BbCode::gross();
            for (basis, obs) in [(Basis::X, 13), (Basis::Z, 11)] {
                let c = logical_measurement(&code, &g, basis, 1, 2, 1, 0.0).unwrap();
                let m = M2d::new(&c).unwrap_or_else(|e| panic!("{name} {basis:?}: {e}"));
                assert_eq!(m.num_observables, obs, "{name} {basis:?}");
            }
            // Read out right after the split too: the checks that lost their
            // edge qubit take its reading into their last comparison.
            for basis in [Basis::X, Basis::Z] {
                M2d::new(&logical_measurement(&code, &g, basis, 1, 2, 0, 0.0).unwrap()).unwrap_or_else(|e| panic!("{name} {basis:?}, post 0: {e}"));
            }
        }
    }

    /// At the split, each flux check's last value is compared with its edge
    /// qubits' readings: one more detector per flux check in the Z basis
    /// than the checks that go on being measured give, none in X.
    #[test]
    fn retired_flux_checks_close_their_comparisons() {
        let (code, g) = gauged("f");
        let count = |basis, merged| M2d::new(&logical_measurement(&code, &g, basis, 1, merged, 1, 0.0).unwrap()).unwrap().num_detectors;
        let h = code.half();
        // Z basis: each merged cycle adds the Z checks and the flux checks; the split adds the flux checks once more.
        assert_eq!(count(Basis::Z, 3) - count(Basis::Z, 2), h + g.flux.len());
        assert_eq!(count(Basis::Z, 2), 5 * h + 2 * g.flux.len() + g.flux.len());
        // X basis: each merged cycle adds the X checks and the Gauss-law checks after the first; nothing at the split.
        assert_eq!(count(Basis::X, 3) - count(Basis::X, 2), h + g.support.len());
    }

    /// With one merged cycle, one measurement error on a Gauss-law ancilla
    /// flips the outcome unseen, and the error-model builder refuses the
    /// circuit; with two, no single fault flips an observable unseen.
    #[test]
    fn the_outcome_needs_two_merged_cycles() {
        let (code, g) = gauged("f");
        let build = |merged| Dem::from_circuit_undecomposed(&logical_measurement(&code, &g, Basis::X, 1, merged, 1, 0.001).unwrap());
        assert!(build(1).unwrap_err().contains("undetectable"));
        let dem = build(2).unwrap();
        assert_eq!(dem.num_observables, 13);
        let (code, g) = gauged("f");
        let z = Dem::from_circuit_undecomposed(&logical_measurement(&code, &g, Basis::Z, 1, 2, 1, 0.001).unwrap()).unwrap();
        assert_eq!(z.num_observables, 11);
    }
}

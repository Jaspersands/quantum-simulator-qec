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
        self.c.push(Instr::Observable { index, recs });
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
}

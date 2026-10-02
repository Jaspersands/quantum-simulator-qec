//! Lattice surgery: two rotated surface-code patches merged across a seam to
//! measure Z⊗Z, then split again, as one circuit.
//!
//! WHY THIS EXISTS
//! ---------------
//! A memory holds one logical qubit still; a computer needs logical qubits to
//! interact. On a planar chip the standard way is lattice surgery (Horsman,
//! Fowler, Devitt and Van Meter, 2012). The stabilizers across the seam
//! between two patches are measured for some rounds, which merges them into
//! one patch; the product of the new checks along the seam is the joint parity
//! of the two logical qubits; then the patches are split. How many merged rounds a
//! measurement needs is the clock of every resource estimate. This module
//! writes the whole experiment as a circuit, so the error-model builder, the
//! samplers and the matchers see it exactly as they see a memory.
//!
//! Geometry is `RotatedSurfaceCode`'s: data at odd (x, y), checks at even
//! (x, y), X or Z by the parity of (x + y)/2, the same CNOT orders. Patch 1's
//! data columns are x = 1 … 2d − 1, the seam is x = 2d + 1, and patch 2 is
//! patch 1 moved 2d + 2 to the right. Each patch's logical Z runs down a
//! column, so the two face each other with their X-type boundaries.
//!
//! Detectors follow one rule. A check measured in two consecutive rounds is
//! compared across them, and if its support changed in between (the boundary X
//! checks reaching across the seam at the merge and pulling back at the
//! split), the qubits it gained were freshly prepared in its basis and the
//! qubits it lost were measured in its basis, so their records join the
//! comparison. A check measured for the first time is a detector alone only
//! if every qubit it touches was freshly prepared in its basis.

use std::collections::HashMap;

use crate::circuit::{Basis, Circuit, Instr};
use crate::surface_code::RotatedSurfaceCode;

/// The experiment.
#[derive(Clone, Copy, Debug)]
pub struct Surgery {
    pub d: usize,
    /// Rounds of separate memory before the merge, and after the split.
    pub pre: usize,
    pub merged: usize,
    pub post: usize,
    /// Z: both patches in |0⟩, the merge outcome and each patch's Z are the
    /// observables. X: both in |+⟩, X₁X₂ (which the Z⊗Z measurement keeps) is.
    pub basis: Basis,
    pub p: f64,
}

#[derive(Clone, Debug)]
struct Check {
    pos: (i32, i32),
    x_type: bool,
    anc: u32,
    /// The data qubit at each of the four CNOT steps, if there is one.
    at_step: [Option<u32>; 4],
}

impl Check {
    fn support(&self) -> Vec<u32> {
        let mut s: Vec<u32> = self.at_step.iter().flatten().copied().collect();
        s.sort_unstable();
        s
    }
}

/// Data columns x0..=x1 and rows y0..=y1 (odd) of one patch, or of patches merged.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct TileBox {
    x0: i32,
    x1: i32,
    y0: i32,
    y1: i32,
}

impl TileBox {
    fn union(a: TileBox, b: TileBox) -> TileBox {
        TileBox { x0: a.x0.min(b.x0), x1: a.x1.max(b.x1), y0: a.y0.min(b.y0), y1: a.y1.max(b.y1) }
    }
}

struct Layout {
    d: i32,
    positions: Vec<(i32, i32)>,
    index: HashMap<(i32, i32), u32>,
}

impl Layout {
    /// A grid of `cols` × `rows` tiles, 2d + 2 apart, one seam column or row
    /// of data between neighbours: data first, row by row, then every check
    /// position, row by row (for 2 × 1 tiles, the order of the two-patch
    /// experiment as it was first written).
    fn new(d: usize, cols: i32, rows: i32) -> Layout {
        let d = d as i32;
        let (w, h) = (cols * (2 * d + 2) - 2, rows * (2 * d + 2) - 2);
        let (mut positions, mut index) = (Vec::new(), HashMap::new());
        for y in (1..h).step_by(2) {
            for x in (1..w).step_by(2) {
                index.insert((x, y), positions.len() as u32);
                positions.push((x, y));
            }
        }
        for y in (0..=h).step_by(2) {
            for x in (0..=w).step_by(2) {
                index.insert((x, y), positions.len() as u32);
                positions.push((x, y));
            }
        }
        Layout { d, positions, index }
    }

    /// The data of tile (column i, row j).
    fn tile(&self, (i, j): (i32, i32)) -> TileBox {
        let (x0, y0) = (1 + i * (2 * self.d + 2), 1 + j * (2 * self.d + 2));
        TileBox { x0, x1: x0 + 2 * self.d - 2, y0, y1: y0 + 2 * self.d - 2 }
    }

    fn data(&self, b: TileBox) -> Vec<u32> {
        let mut v: Vec<u32> = (b.y0..=b.y1)
            .step_by(2)
            .flat_map(|y| (b.x0..=b.x1).step_by(2).map(move |x| (x, y)))
            .map(|p| self.index[&p])
            .collect();
        v.sort_unstable();
        v
    }

    /// The rotated code on box `b`: X-type boundaries left and right, Z-type
    /// top and bottom.
    fn code(&self, b: TileBox) -> Vec<Check> {
        let inside = |x: i32, y: i32| x >= b.x0 && x <= b.x1 && y >= b.y0 && y <= b.y1;
        let mut checks = Vec::new();
        for y in (b.y0 - 1..=b.y1 + 1).step_by(2) {
            for x in (b.x0 - 1..=b.x1 + 1).step_by(2) {
                let x_type = ((x + y) / 2).rem_euclid(2) == 1;
                // Z checks stay off the left and right edges, X checks off the top and bottom.
                let allowed = if x_type { y > b.y0 && y < b.y1 } else { x > b.x0 && x < b.x1 };
                if !allowed {
                    continue;
                }
                let order = if x_type { RotatedSurfaceCode::X_ORDER } else { RotatedSurfaceCode::Z_ORDER };
                let mut at_step = [None; 4];
                for (k, &(dx, dy)) in order.iter().enumerate() {
                    if inside(x + dx, y + dy) {
                        at_step[k] = Some(self.index[&(x + dx, y + dy)]);
                    }
                }
                if at_step.iter().flatten().count() >= 2 {
                    checks.push(Check { pos: (x, y), x_type, anc: self.index[&(x, y)], at_step });
                }
            }
        }
        checks
    }
}

/// What a detector needs to know about each check's last measurement.
struct Last {
    rec: usize,
    support: Vec<u32>,
    /// The round it was measured in.
    round: usize,
}

struct Writer {
    c: Vec<Instr>,
    m: usize,
    p: f64,
    last: HashMap<(i32, i32), Last>,
    /// Data qubits freshly prepared this round, with their basis.
    fresh: HashMap<u32, Basis>,
    /// Data qubits measured since their checks last read them: record and basis.
    measured: HashMap<u32, (usize, Basis)>,
    round: usize,
}

impl Writer {
    fn lookback(&self, rec: usize) -> u32 {
        (self.m - rec) as u32
    }

    fn noise(&mut self, instr: Instr) {
        if self.p > 0.0 {
            self.c.push(instr);
        }
    }

    fn depol1(&mut self, qubits: Vec<u32>) {
        if !qubits.is_empty() {
            let p = self.p;
            self.noise(Instr::Depolarize1 { p, qubits });
        }
    }

    fn prepare(&mut self, basis: Basis, qubits: &[u32]) {
        if qubits.is_empty() {
            return;
        }
        self.c.push(Instr::Reset { basis, qubits: qubits.to_vec() });
        let p = self.p;
        self.noise(Instr::PauliError { pauli: if basis == Basis::Z { 1 } else { 2 }, p, qubits: qubits.to_vec() });
        for &q in qubits {
            self.fresh.insert(q, basis);
        }
    }

    /// Measure data qubits; returns their records in the order given.
    fn measure(&mut self, basis: Basis, qubits: &[u32]) -> Vec<usize> {
        let p = self.p;
        self.noise(Instr::PauliError { pauli: if basis == Basis::Z { 1 } else { 2 }, p, qubits: qubits.to_vec() });
        self.c.push(Instr::Measure { basis, reset: false, flip: 0.0, qubits: qubits.to_vec() });
        let recs: Vec<usize> = (0..qubits.len()).map(|i| self.m + i).collect();
        self.m += qubits.len();
        for (&q, &r) in qubits.iter().zip(&recs) {
            self.measured.insert(q, (r, basis));
        }
        recs
    }

    /// One round of syndrome extraction on `checks`, the SD6 way. `data` is
    /// every data qubit in play; those not freshly prepared this round idle
    /// through the reset. Returns each check's record.
    fn round(&mut self, checks: &[Check], data: &[u32]) -> Vec<usize> {
        self.round += 1;
        let anc: Vec<u32> = checks.iter().map(|c| c.anc).collect();
        let had: Vec<u32> = checks.iter().filter(|c| c.x_type).map(|c| c.anc).collect();
        let active: Vec<u32> = data.iter().copied().chain(anc.iter().copied()).collect();
        let idle = |busy: &[u32]| -> Vec<u32> { active.iter().copied().filter(|q| !busy.contains(q)).collect() };

        self.c.push(Instr::Tick);
        self.c.push(Instr::Reset { basis: Basis::Z, qubits: anc.clone() });
        let p = self.p;
        self.noise(Instr::PauliError { pauli: 1, p, qubits: anc.clone() });
        let resting: Vec<u32> = data.iter().copied().filter(|q| !self.fresh.contains_key(q)).collect();
        self.depol1(resting);

        for pass in 0..2 {
            self.c.push(Instr::Tick);
            if !had.is_empty() {
                self.c.push(Instr::H(had.clone()));
                self.depol1(had.clone());
            }
            let rest = idle(&had);
            self.depol1(rest);
            if pass == 1 {
                break;
            }
            for step in 0..4 {
                self.c.push(Instr::Tick);
                let mut pairs = Vec::new();
                for ch in checks {
                    if let Some(q) = ch.at_step[step] {
                        pairs.push(if ch.x_type { (ch.anc, q) } else { (q, ch.anc) });
                    }
                }
                if !pairs.is_empty() {
                    self.c.push(Instr::Cx(pairs.clone()));
                    self.noise(Instr::Depolarize2 { p, pairs: pairs.clone() });
                }
                let busy: Vec<u32> = pairs.iter().flat_map(|&(a, b)| [a, b]).collect();
                let rest = idle(&busy);
                self.depol1(rest);
            }
        }

        self.c.push(Instr::Tick);
        self.noise(Instr::PauliError { pauli: 1, p, qubits: anc.clone() });
        self.c.push(Instr::Measure { basis: Basis::Z, reset: false, flip: 0.0, qubits: anc.clone() });
        let recs: Vec<usize> = (0..anc.len()).map(|i| self.m + i).collect();
        self.m += anc.len();
        self.depol1(data.to_vec());

        for (ch, &rec) in checks.iter().zip(&recs) {
            let basis = if ch.x_type { Basis::X } else { Basis::Z };
            let support = ch.support();
            let coords = vec![ch.pos.0 as f64, ch.pos.1 as f64, self.round as f64];
            let mut targets = vec![rec];
            // A check compares with its last measurement only if that was the
            // round before: one that skipped a round (a seam check of an
            // earlier merge, say, whose seam was since read and prepared
            // again) starts over, as if measured for the first time.
            let deterministic = match self.last.get(&ch.pos).filter(|prev| prev.round + 1 == self.round) {
                Some(prev) => {
                    targets.push(prev.rec);
                    // A reading since `prev`, in the check's basis, of a qubit it read then.
                    let read_since = |q: &u32| match self.measured.get(q) {
                        Some(&(r, b)) if b == basis && r > prev.rec => Some(r),
                        _ => None,
                    };
                    let mut ok = true;
                    // Qubits gained, and qubits kept but read and prepared again
                    // since (a seam split and merged again at once), must be
                    // fresh in its basis; the kept ones' readings join.
                    for q in support.iter().filter(|q| self.fresh.contains_key(q)) {
                        ok &= self.fresh.get(q) == Some(&basis);
                        if prev.support.contains(q) {
                            match read_since(q) {
                                Some(r) => targets.push(r),
                                None => ok = false,
                            }
                        }
                    }
                    ok &= support.iter().filter(|q| !prev.support.contains(q)).all(|q| self.fresh.contains_key(q));
                    for q in prev.support.iter().filter(|q| !support.contains(q)) {
                        match read_since(q) {
                            Some(r) => targets.push(r),
                            None => ok = false,
                        }
                    }
                    ok
                }
                None => support.iter().all(|q| self.fresh.get(q) == Some(&basis)),
            };
            if deterministic {
                let recs = targets.iter().map(|&r| self.lookback(r)).collect();
                self.c.push(Instr::Detector { coords, recs });
            }
            self.last.insert(ch.pos, Last { rec, support, round: self.round });
        }
        self.fresh.clear();
        recs
    }
}

/// A lattice-surgery program: patches of distance d on a grid of tiles, and
/// what is done to them, in order.
///
/// Every step is one of five. `Prepare` resets patches' data in a basis;
/// `Rounds` measures every live patch and merge for some rounds; `Merge` joins
/// a line of adjacent patches, which a horizontal line does by measuring each
/// neighbouring pair's Z⊗Z (its seam prepared in |+⟩) and a vertical line
/// each pair's X⊗X (in |0⟩); `Split` measures every current merge's seam, X
/// or Z, to part them again;
/// `Measure` reads patches' data out in a basis, which ends them. Detectors
/// follow the one rule the Writer applies, whatever the program. Observables
/// are given as terms (`Term`), and their determinism is not assumed: the
/// engine's reference simulation refuses any observable the terms leave
/// random, so a program with a wrong Pauli frame fails to compile into a
/// usable model rather than decoding nonsense.
#[derive(Clone, Debug)]
pub struct Program {
    pub d: usize,
    pub p: f64,
    /// Each patch's tile, (column, row).
    pub tiles: Vec<(i32, i32)>,
    pub steps: Vec<Step>,
    /// Each observable, as the terms whose records it multiplies.
    pub observables: Vec<Vec<Term>>,
}

#[derive(Clone, Debug)]
pub enum Step {
    /// Prepare the patches' data in `basis`, as one reset.
    Prepare { patches: Vec<usize>, basis: Basis },
    /// Rounds of syndrome extraction on every live patch and merge.
    Rounds(usize),
    /// Merge a line of adjacent patches: horizontal measures each neighbouring
    /// pair's Z⊗Z, vertical each pair's X⊗X. A line of n is one patch holding
    /// one logical qubit, so it measures n − 1 parities, not the n-body
    /// product (which needs an ancilla region along every patch). The seams
    /// are prepared at once; rounds follow.
    Merge { patches: Vec<usize> },
    /// Split every current merge: each seam measured, X for a horizontal
    /// line, Z for a vertical one.
    Split,
    /// Measure the patches' data in `basis`, as one measurement; they are done.
    Measure { patches: Vec<usize>, basis: Basis },
}

#[derive(Clone, Copy, Debug)]
pub enum Term {
    /// A patch's logical in `basis`, from its final measurement in that
    /// basis, along data line `line` (0 the first): Z a column, X a row.
    Logical { patch: usize, basis: Basis, line: usize },
    /// Merge `merge`'s outcome on seam `seam` (merges numbered in program
    /// order, seams along the line, 0 between its first two patches): the
    /// product of that seam's new checks of the measured type in the merge's
    /// first round, the parity of the two patches either side.
    Outcome { merge: usize, seam: usize },
    /// Merge `merge`'s seam records, from its split, on `patch`'s data line
    /// `line` where it crosses the seams: a row for a horizontal line
    /// (vertical seams), a column for a vertical line.
    Seam { merge: usize, patch: usize, line: usize },
}

struct MergeState {
    patches: Vec<usize>,
    horizontal: bool,
    code: Vec<Check>,
    data: Vec<u32>,
    /// The seams' data qubits, sorted.
    seam: Vec<u32>,
    /// Per seam, the merged code's new checks of the measured type, as indices into `code`.
    new_checks: Vec<Vec<usize>>,
    /// Per seam, those checks' records in the first merged round.
    outcomes: Option<Vec<Vec<usize>>>,
    seam_recs: Option<Vec<usize>>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Life {
    Unprepared,
    Live,
    Done,
}

/// The most rounds (and other steps) a program is written out for.
const MAX_ROUNDS: usize = 10_000;

/// About how many qubits `tiles` tiles at distance `d` take: 2(d + 1)² each, saturating.
fn qubits_for(d: usize, tiles: u64) -> u64 {
    tiles.saturating_mul(2).saturating_mul((d as u64).saturating_add(1).saturating_pow(2))
}

/// Refuse, before a program is built, sizes it could never be written for: more qubits than
/// Stim can index, or more rounds than are written out. The program constructors allocate in
/// proportion to their patches and steps, so the bindings call this first.
pub fn check_size(d: usize, tiles: usize, rounds: usize) -> Result<(), String> {
    let qubits = qubits_for(d, tiles as u64);
    if qubits > 1 << 24 {
        return Err(format!("{tiles} patches at distance {d} need more qubits than Stim can index"));
    }
    if rounds > MAX_ROUNDS {
        return Err(format!("{rounds} rounds and steps: at most {MAX_ROUNDS} are written out"));
    }
    // Every program holds its patches d rounds before and after, besides `rounds`.
    check_volume(qubits, rounds.saturating_add(d.saturating_mul(2)))
}

/// Z⊗Z on two patches side by side (see `Surgery`), checked for size before it is built.
pub fn zz_circuit(d: usize, pre: usize, merged: usize, post: usize, p: f64, basis: Basis) -> Result<Circuit, String> {
    check_size(d, 2, merged.saturating_add(pre).saturating_add(post))?;
    Surgery { d, pre, merged, post, basis, p }.circuit()
}

/// X⊗X (see `vertical`), checked for size before it is built.
pub fn xx_circuit(d: usize, merged: usize, p: f64, basis: Basis) -> Result<Circuit, String> {
    check_size(d, 2, merged)?;
    vertical(d, merged, p, basis).circuit()
}

/// The logical CNOT (see `cnot`), checked for size before it is built.
pub fn cnot_circuit(d: usize, merged: usize, p: f64, inputs: Basis) -> Result<Circuit, String> {
    check_size(d, 4, merged.saturating_mul(2))?;
    cnot(d, merged, p, inputs).circuit()
}

/// `k` Z⊗Z measurements in a row (see `repeated`), checked for size before it is built.
pub fn repeated_circuit(d: usize, k: usize, merged: usize, p: f64) -> Result<Circuit, String> {
    check_size(d, 2, k.saturating_mul(merged.saturating_add(2)))?;
    repeated(d, k, merged, p).circuit()
}

/// `n` patches merged in a row (see `line`), checked before it is built: at least two patches,
/// and at most 32, since each seam and each patch is an observable.
pub fn line_circuit(d: usize, n: usize, merged: usize, p: f64) -> Result<Circuit, String> {
    if n < 2 {
        return Err(format!("a line merge needs at least two patches, not {n}"));
    }
    if n > 32 {
        return Err(format!("{n} patches have {} observables (each seam and each patch); at most 64 are supported", 2 * n - 1));
    }
    check_size(d, n, merged)?;
    line(d, n, merged, p).circuit()
}

/// The most qubit-rounds a program is written out for: past this an error model, which holds
/// a circuit unrolled, would refuse it anyway.
const MAX_VOLUME: u64 = 1 << 24;

fn check_volume(qubits: u64, rounds: usize) -> Result<(), String> {
    if qubits.saturating_mul(rounds as u64) > MAX_VOLUME {
        return Err(format!(
            "about {qubits} qubits over {rounds} rounds: more than the {MAX_VOLUME} qubit-rounds a program is written out for"
        ));
    }
    Ok(())
}

impl Program {
    /// The program as a circuit, with its detectors and observables.
    pub fn circuit(&self) -> Result<Circuit, String> {
        let d = self.d;
        if d < 3 || d.is_multiple_of(2) {
            return Err(format!("lattice surgery needs an odd distance of at least 3, not {d}"));
        }
        if self.tiles.is_empty() || self.tiles.iter().any(|&(i, j)| i < 0 || j < 0) {
            return Err("a program needs patches on tiles of non-negative column and row".into());
        }
        crate::memory::probability(self.p)?;
        // The grid's qubits, about 2(d + 1)² a tile, must fit Stim's qubit
        // indices, and the circuit, written out round by round, memory.
        let grid = |t: fn(&(i32, i32)) -> i32| self.tiles.iter().map(t).max().unwrap() as u64 + 1;
        let qubits = qubits_for(d, grid(|t| t.0).saturating_mul(grid(|t| t.1)));
        if qubits > 1 << 24 {
            return Err(format!("the program's grid of tiles at distance {d} needs more qubits than Stim can index"));
        }
        let rounds = self.steps.iter().fold(0usize, |n, s| n.saturating_add(if let Step::Rounds(k) = s { *k } else { 1 }));
        if rounds > MAX_ROUNDS {
            return Err(format!("the program runs {rounds} rounds and steps, more than the {MAX_ROUNDS} written out"));
        }
        check_volume(qubits, rounds)?;
        let n = self.tiles.len();
        for a in 0..n {
            if self.tiles[a + 1..].contains(&self.tiles[a]) {
                return Err(format!("two patches share tile {:?}", self.tiles[a]));
            }
        }
        let cols = self.tiles.iter().map(|t| t.0).max().unwrap() + 1;
        let rows = self.tiles.iter().map(|t| t.1).max().unwrap() + 1;
        let layout = Layout::new(d, cols, rows);
        let boxes: Vec<TileBox> = self.tiles.iter().map(|&t| layout.tile(t)).collect();
        let codes: Vec<Vec<Check>> = boxes.iter().map(|&b| layout.code(b)).collect();
        let datas: Vec<Vec<u32>> = boxes.iter().map(|&b| layout.data(b)).collect();

        let mut w = Writer { c: Vec::new(), m: 0, p: self.p, last: HashMap::new(), fresh: HashMap::new(), measured: HashMap::new(), round: 0 };
        for (q, &(x, y)) in layout.positions.iter().enumerate() {
            w.c.push(Instr::QubitCoords { coords: vec![x as f64, y as f64], qubits: vec![q as u32] });
        }
        let mut life = vec![Life::Unprepared; n];
        let mut in_merge: Vec<Option<usize>> = vec![None; n];
        let mut merges: Vec<MergeState> = Vec::new();
        let mut active: Vec<usize> = Vec::new();
        let mut final_basis: Vec<Option<Basis>> = vec![None; n];
        let mut final_rec: HashMap<u32, usize> = HashMap::new();
        let check = |q: usize| if q < n { Ok(()) } else { Err(format!("there is no patch {q}")) };
        // A step's patches: at least one, each once.
        let listed = |patches: &[usize]| -> Result<(), String> {
            if patches.is_empty() {
                return Err("a step names no patch".into());
            }
            for (i, &q) in patches.iter().enumerate() {
                check(q)?;
                if patches[..i].contains(&q) {
                    return Err(format!("patch {q} is named twice in one step"));
                }
            }
            Ok(())
        };

        for step in &self.steps {
            match step {
                Step::Prepare { patches, basis } => {
                    listed(patches)?;
                    for &q in patches {
                        if life[q] != Life::Unprepared {
                            return Err(format!("patch {q} is prepared twice"));
                        }
                        life[q] = Life::Live;
                    }
                    let data: Vec<u32> = patches.iter().flat_map(|&q| datas[q].iter().copied()).collect();
                    w.prepare(*basis, &data);
                }
                Step::Rounds(k) => {
                    // Units in order of their lowest patch: a patch alone, or a merge.
                    let mut units: Vec<(usize, Option<usize>)> = (0..n)
                        .filter(|&q| life[q] == Life::Live && in_merge[q].is_none())
                        .map(|q| (q, None))
                        .chain(active.iter().map(|&m| (merges[m].patches[0], Some(m))))
                        .collect();
                    units.sort_unstable_by_key(|u| u.0);
                    if units.is_empty() {
                        return Err("rounds with no live patch".into());
                    }
                    let mut checks: Vec<Check> = Vec::new();
                    let mut data: Vec<u32> = Vec::new();
                    let mut offsets = Vec::new();
                    for &(q, m) in &units {
                        offsets.push(checks.len());
                        match m {
                            None => {
                                checks.extend(codes[q].iter().cloned());
                                data.extend(&datas[q]);
                            }
                            Some(m) => {
                                checks.extend(merges[m].code.iter().cloned());
                                data.extend(&merges[m].data);
                            }
                        }
                    }
                    for _ in 0..*k {
                        let recs = w.round(&checks, &data);
                        for (&(_, m), &off) in units.iter().zip(&offsets) {
                            if let Some(m) = m {
                                let ms = &mut merges[m];
                                if ms.outcomes.is_none() {
                                    ms.outcomes = Some(ms.new_checks.iter().map(|seam| seam.iter().map(|&i| recs[off + i]).collect()).collect());
                                }
                            }
                        }
                    }
                }
                Step::Merge { patches } => {
                    listed(patches)?;
                    if patches.len() < 2 {
                        return Err("a merge needs at least two patches".into());
                    }
                    for &q in patches {
                        if life[q] != Life::Live || in_merge[q].is_some() {
                            return Err(format!("patch {q} is not live and alone, so it cannot merge"));
                        }
                    }
                    let t: Vec<(i32, i32)> = patches.iter().map(|&q| self.tiles[q]).collect();
                    let horizontal = t.windows(2).all(|p| p[1] == (p[0].0 + 1, p[0].1));
                    let vertical = t.windows(2).all(|p| p[1] == (p[0].0, p[0].1 + 1));
                    if !horizontal && !vertical {
                        return Err(format!("patches {patches:?} are not a line of neighbouring tiles, in order"));
                    }
                    let bbox = patches.iter().map(|&q| boxes[q]).reduce(TileBox::union).unwrap();
                    let code = layout.code(bbox);
                    let data = layout.data(bbox);
                    let seam: Vec<u32> = data.iter().copied().filter(|q| !patches.iter().any(|&p| datas[p].contains(q))).collect();
                    // Horizontal measures Z⊗Z across each seam, by new Z checks;
                    // vertical X⊗X, by new X checks. A seam's checks lie between
                    // the two patches either side of it.
                    let across = |c: &Check| if horizontal { c.pos.0 } else { c.pos.1 };
                    let span = |q: usize| if horizontal { (boxes[q].x0, boxes[q].x1) } else { (boxes[q].y0, boxes[q].y1) };
                    let new_checks = patches
                        .windows(2)
                        .map(|pair| {
                            let (after, before) = (span(pair[0]).1, span(pair[1]).0);
                            code.iter()
                                .enumerate()
                                .filter(|(_, c)| c.x_type != horizontal && after < across(c) && across(c) < before)
                                .filter(|(_, c)| !patches.iter().any(|&q| codes[q].iter().any(|pc| pc.pos == c.pos)))
                                .map(|(i, _)| i)
                                .collect()
                        })
                        .collect();
                    w.c.push(Instr::Tick);
                    w.prepare(if horizontal { Basis::X } else { Basis::Z }, &seam);
                    let m = merges.len();
                    for &q in patches {
                        in_merge[q] = Some(m);
                    }
                    merges.push(MergeState { patches: patches.clone(), horizontal, code, data, seam, new_checks, outcomes: None, seam_recs: None });
                    active.push(m);
                }
                Step::Split => {
                    if active.is_empty() {
                        return Err("a split with nothing merged".into());
                    }
                    w.c.push(Instr::Tick);
                    for m in std::mem::take(&mut active) {
                        let basis = if merges[m].horizontal { Basis::X } else { Basis::Z };
                        let seam = merges[m].seam.clone();
                        merges[m].seam_recs = Some(w.measure(basis, &seam));
                        for &q in &merges[m].patches {
                            in_merge[q] = None;
                        }
                    }
                }
                Step::Measure { patches, basis } => {
                    listed(patches)?;
                    for &q in patches {
                        if life[q] != Life::Live || in_merge[q].is_some() {
                            return Err(format!("patch {q} is not live and alone, so it cannot be measured"));
                        }
                    }
                    w.c.push(Instr::Tick);
                    let data: Vec<u32> = patches.iter().flat_map(|&q| datas[q].iter().copied()).collect();
                    let recs = w.measure(*basis, &data);
                    final_rec.extend(data.iter().copied().zip(recs));
                    let want_x = *basis == Basis::X;
                    for &q in patches {
                        life[q] = Life::Done;
                        final_basis[q] = Some(*basis);
                        for ch in codes[q].iter().filter(|c| c.x_type == want_x) {
                            // Only a check measured in the last round predicts the readout.
                            let Some(last) = w.last.get(&ch.pos).filter(|l| l.round == w.round) else { continue };
                            let support = ch.support();
                            let mut targets: Vec<usize> = support.iter().map(|x| final_rec[x]).collect();
                            targets.push(last.rec);
                            // Qubits its last measurement read and its patch does not:
                            // seam qubits, read at a split in the same basis.
                            for lost in last.support.iter().filter(|x| !support.contains(x)) {
                                match w.measured.get(lost) {
                                    Some(&(r, b)) if b == *basis && r > last.rec => targets.push(r),
                                    _ => return Err(format!("the check at {:?} lost a qubit not measured in its basis", ch.pos)),
                                }
                            }
                            let coords = vec![ch.pos.0 as f64, ch.pos.1 as f64, (w.round + 1) as f64];
                            let recs = targets.iter().map(|&r| w.lookback(r)).collect();
                            w.c.push(Instr::Detector { coords, recs });
                        }
                    }
                }
            }
        }

        for (index, terms) in self.observables.iter().enumerate() {
            let mut recs = Vec::new();
            for term in terms {
                match *term {
                    Term::Logical { patch, basis, line } => {
                        check(patch)?;
                        if final_basis[patch] != Some(basis) {
                            return Err(format!("patch {patch}'s logical {basis:?} needs it measured in {basis:?}"));
                        }
                        if line >= d {
                            return Err(format!("a patch has {d} lines, not {}", line + 1));
                        }
                        let b = boxes[patch];
                        let at = 2 * line as i32;
                        let qubits: Vec<u32> = if basis == Basis::Z {
                            (b.y0..=b.y1).step_by(2).map(|y| layout.index[&(b.x0 + at, y)]).collect()
                        } else {
                            (b.x0..=b.x1).step_by(2).map(|x| layout.index[&(x, b.y0 + at)]).collect()
                        };
                        recs.extend(qubits.iter().map(|q| w.lookback(final_rec[q])));
                    }
                    Term::Outcome { merge, seam } => {
                        let ms = merges.get(merge).ok_or(format!("there is no merge {merge}"))?;
                        let out = ms.outcomes.as_ref().ok_or(format!("merge {merge} had no rounds"))?;
                        let out = out.get(seam).ok_or(format!("merge {merge} has no seam {seam}"))?;
                        recs.extend(out.iter().map(|&r| w.lookback(r)));
                    }
                    Term::Seam { merge, patch, line } => {
                        check(patch)?;
                        if line >= d {
                            return Err(format!("a patch has {d} lines, not {}", line + 1));
                        }
                        let ms = merges.get(merge).ok_or(format!("there is no merge {merge}"))?;
                        if !ms.patches.contains(&patch) {
                            return Err(format!("patch {patch} is not in merge {merge}"));
                        }
                        let seam_recs = ms.seam_recs.as_ref().ok_or(format!("merge {merge} was never split"))?;
                        let b = boxes[patch];
                        let at = 2 * line as i32;
                        for (&q, &r) in ms.seam.iter().zip(seam_recs) {
                            let (x, y) = layout.positions[q as usize];
                            if (ms.horizontal && y == b.y0 + at) || (!ms.horizontal && x == b.x0 + at) {
                                recs.push(w.lookback(r));
                            }
                        }
                    }
                }
            }
            w.c.push(Instr::Observable { index: index as u32, recs });
        }
        Ok(Circuit { instrs: w.c })
    }
}

impl Surgery {
    /// The experiment as a program: two patches side by side, prepared, held
    /// apart, merged to measure Z₁Z₂, split, held apart, read out.
    pub fn program(&self) -> Program {
        let observables = match self.basis {
            Basis::Z => vec![
                vec![Term::Outcome { merge: 0, seam: 0 }],
                vec![Term::Logical { patch: 0, basis: Basis::Z, line: 0 }],
                vec![Term::Logical { patch: 1, basis: Basis::Z, line: 0 }],
            ],
            // X₁X₂ along the first row, with that row's seam qubit read at the split.
            Basis::X => vec![vec![
                Term::Logical { patch: 0, basis: Basis::X, line: 0 },
                Term::Logical { patch: 1, basis: Basis::X, line: 0 },
                Term::Seam { merge: 0, patch: 0, line: 0 },
            ]],
        };
        Program {
            d: self.d,
            p: self.p,
            tiles: vec![(0, 0), (1, 0)],
            steps: vec![
                Step::Prepare { patches: vec![0, 1], basis: self.basis },
                Step::Rounds(self.pre),
                Step::Merge { patches: vec![0, 1] },
                Step::Rounds(self.merged),
                Step::Split,
                Step::Rounds(self.post),
                Step::Measure { patches: vec![0, 1], basis: self.basis },
            ],
            observables,
        }
    }

    pub fn circuit(&self) -> Result<Circuit, String> {
        // With X inputs no observable reads the outcome, so the program alone would not notice.
        if self.merged == 0 {
            return Err("the merge needs at least one round".into());
        }
        self.program().circuit()
    }
}

/// The CNOT's patches: control, ancilla, target.
const CNOT: [usize; 3] = [0, 1, 2];

/// A logical CNOT by lattice surgery (Horsman, Fowler, Devitt and Van Meter,
/// arXiv:1111.4022; Litinski, arXiv:1808.02892): control C at tile (0, 0),
/// an ancilla A at (1, 0) in |+⟩, target T at (1, 1). Z_C Z_A is measured
/// (outcome m₁, merging C and A side by side), then X_A X_T (m₂, merging A
/// and T one above the other), then A is read out in Z (m₃). The result is
/// CNOT from C to T, up to Pauli corrections tracked in software.
///
/// The frames, derived by following the stabilizers through the three
/// measurements. With Z inputs, |0⟩|0⟩: Z_C stays +1, and Z_T comes out
/// (−1)^(m₁ + m₃), so Z_T ⊕ Z_A ⊕ m₁ is +1. Z_A and Z_T are columns that the
/// A–T merge joined through its seam, which the split read in Z, so the seam
/// qubits on that column join them. With X inputs, |+⟩|+⟩: X_T stays +1, and
/// X_C X_T comes out (−1)^m₂. m₂ is the product of the A–T seam's new X checks,
/// which is X on A's last row and T's first; X_C X_A must be taken along that
/// same last row (a different row of A differs by A's X checks, which are not
/// all fixed), and it crossed the C–A seam, read in X at its split, so the
/// seam qubit on that row joins it. `inputs` picks the pair. Both pairs are
/// left alone by a CNOT, and by the identity too: they measure how often the
/// CNOT fails in Z and in X, and a test with inputs that a CNOT does change
/// (`the_cnot_is_a_cnot`) pins its action.
pub fn cnot(d: usize, merged: usize, p: f64, inputs: Basis) -> Program {
    let [c, a, t] = CNOT;
    let observables = match inputs {
        Basis::Z => vec![
            vec![Term::Logical { patch: c, basis: Basis::Z, line: 0 }],
            vec![
                Term::Logical { patch: t, basis: Basis::Z, line: 0 },
                Term::Logical { patch: a, basis: Basis::Z, line: 0 },
                Term::Outcome { merge: 0, seam: 0 },
                Term::Seam { merge: 1, patch: t, line: 0 },
            ],
        ],
        // m₂ reads X on A's last row and T's first, the rows that touch the
        // A–T seam, so X_C X_A is taken along the last row too.
        Basis::X => vec![
            vec![Term::Logical { patch: t, basis: Basis::X, line: 0 }],
            vec![
                Term::Logical { patch: c, basis: Basis::X, line: d.saturating_sub(1) },
                Term::Logical { patch: t, basis: Basis::X, line: 0 },
                Term::Outcome { merge: 1, seam: 0 },
                Term::Seam { merge: 0, patch: c, line: d.saturating_sub(1) },
            ],
        ],
    };
    cnot_with(d, merged, p, [inputs; 2], [inputs; 2], observables)
}

/// The CNOT's program with control and target prepared in `prep` and read
/// out in `read` (each [C, T]), and the given observables.
fn cnot_with(d: usize, merged: usize, p: f64, prep: [Basis; 2], read: [Basis; 2], observables: Vec<Vec<Term>>) -> Program {
    let [c, a, t] = CNOT;
    // Each pair read or prepared alike is one step, as the recorded circuits were written.
    let pair = |basis: [Basis; 2], step: fn(Vec<usize>, Basis) -> Step| {
        if basis[0] == basis[1] {
            vec![step(vec![c, t], basis[0])]
        } else {
            vec![step(vec![c], basis[0]), step(vec![t], basis[1])]
        }
    };
    let mut steps = pair(prep, |patches, basis| Step::Prepare { patches, basis });
    steps.extend([
        Step::Prepare { patches: vec![a], basis: Basis::X },
        Step::Rounds(d),
        Step::Merge { patches: vec![c, a] },
        Step::Rounds(merged),
        Step::Split,
        Step::Merge { patches: vec![a, t] },
        Step::Rounds(merged),
        Step::Split,
        Step::Measure { patches: vec![a], basis: Basis::Z },
        Step::Rounds(d),
    ]);
    steps.extend(pair(read, |patches, basis| Step::Measure { patches, basis }));
    Program { d, p, tiles: vec![(0, 0), (1, 0), (1, 1)], steps, observables }
}

/// k Z⊗Z measurements in a row on two patches in |0⟩|0⟩, each of `merged`
/// rounds, split, and followed by one round apart; d rounds apart before the
/// first and after the last. Observables: each outcome (all +1), then each
/// patch's Z.
pub fn repeated(d: usize, k: usize, merged: usize, p: f64) -> Program {
    let mut steps = vec![Step::Prepare { patches: vec![0, 1], basis: Basis::Z }, Step::Rounds(d)];
    for _ in 0..k {
        steps.extend([Step::Merge { patches: vec![0, 1] }, Step::Rounds(merged), Step::Split, Step::Rounds(1)]);
    }
    steps.extend([Step::Rounds(d.saturating_sub(1)), Step::Measure { patches: vec![0, 1], basis: Basis::Z }]);
    let mut observables: Vec<Vec<Term>> = (0..k).map(|i| vec![Term::Outcome { merge: i, seam: 0 }]).collect();
    observables.push(vec![Term::Logical { patch: 0, basis: Basis::Z, line: 0 }]);
    observables.push(vec![Term::Logical { patch: 1, basis: Basis::Z, line: 0 }]);
    Program { d, p, tiles: vec![(0, 0), (1, 0)], steps, observables }
}

/// n patches in a row, all in |0⟩, merged at once and split: each
/// neighbouring pair's Z⊗Z measured together. The merged patch holds one
/// logical qubit, so this takes n − 1 parities, not the n-body product
/// Z⊗…⊗Z of a Pauli-based computation, which needs an ancilla region along
/// every patch. Observables: each seam's outcome (+1), then each patch's Z.
pub fn line(d: usize, n: usize, merged: usize, p: f64) -> Program {
    let all: Vec<usize> = (0..n).collect();
    let mut observables: Vec<Vec<Term>> = (0..n.saturating_sub(1)).map(|seam| vec![Term::Outcome { merge: 0, seam }]).collect();
    observables.extend((0..n).map(|q| vec![Term::Logical { patch: q, basis: Basis::Z, line: 0 }]));
    Program {
        d,
        p,
        tiles: (0..n as i32).map(|i| (i, 0)).collect(),
        steps: vec![
            Step::Prepare { patches: all.clone(), basis: Basis::Z },
            Step::Rounds(d),
            Step::Merge { patches: all.clone() },
            Step::Rounds(merged),
            Step::Split,
            Step::Rounds(d),
            Step::Measure { patches: all, basis: Basis::Z },
        ],
        observables,
    }
}

/// The X⊗X mirror of the Z⊗Z experiment: two patches one above the other,
/// merged across a row of seam qubits prepared in |0⟩ and split by reading
/// them in Z. With X inputs the observables are the outcome and each patch's
/// X; with Z inputs, Z₁Z₂ (which the X⊗X measurement keeps), with the seam
/// qubit on patch 0's column read at the split.
pub fn vertical(d: usize, merged: usize, p: f64, basis: Basis) -> Program {
    let observables = match basis {
        Basis::X => vec![
            vec![Term::Outcome { merge: 0, seam: 0 }],
            vec![Term::Logical { patch: 0, basis: Basis::X, line: 0 }],
            vec![Term::Logical { patch: 1, basis: Basis::X, line: 0 }],
        ],
        Basis::Z => vec![vec![
            Term::Logical { patch: 0, basis: Basis::Z, line: 0 },
            Term::Logical { patch: 1, basis: Basis::Z, line: 0 },
            Term::Seam { merge: 0, patch: 0, line: 0 },
        ]],
    };
    Program {
        d,
        p,
        tiles: vec![(0, 0), (0, 1)],
        steps: vec![
            Step::Prepare { patches: vec![0, 1], basis },
            Step::Rounds(d),
            Step::Merge { patches: vec![0, 1] },
            Step::Rounds(merged),
            Step::Split,
            Step::Rounds(d),
            Step::Measure { patches: vec![0, 1], basis },
        ],
        observables,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dem::Dem;
    use crate::dem_decoder::DemDecoder;
    use crate::m2d::M2d;

    fn surgery(d: usize, merged: usize, basis: Basis, p: f64) -> Surgery {
        Surgery { d, pre: d, merged, post: d, basis, p }
    }

    /// FNV-1a, 64 bits: a fingerprint that is the same in every Rust version.
    fn fnv1a64(bytes: &[u8]) -> String {
        let h = bytes.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &b| (h ^ u64::from(b)).wrapping_mul(0x100_0000_01b3));
        format!("{h:016x}")
    }

    /// The grid layout and the program compiler change nothing: every
    /// experiment section 14 measured compiles to the same circuit, byte for
    /// byte, as before either existed (fingerprints taken before, in
    /// data/surgery/golden.json).
    #[test]
    fn the_z_z_experiment_is_unchanged() {
        let golden = include_str!("../data/surgery/golden.json");
        let mut n = 0;
        for d in [3usize, 5, 7] {
            let mut ts = vec![1usize, 2, 3, d, 2 * d];
            ts.sort_unstable();
            ts.dedup();
            for merged in ts {
                for (basis, name) in [(Basis::Z, "z"), (Basis::X, "x")] {
                    let text = Surgery { d, pre: d, merged, post: d, basis, p: 0.002 }.circuit().unwrap().to_stim();
                    let key = format!("\"d{d}/T{merged}/{name}\": \"{}\"", fnv1a64(text.as_bytes()));
                    assert!(golden.contains(&key), "d = {d}, T = {merged}, {name}: the circuit changed");
                    n += 1;
                }
            }
        }
        assert_eq!(n, 28);
    }

    /// Without noise every detector and observable is deterministic: the
    /// engine's references refuse any that is not.
    #[test]
    fn every_detector_and_observable_is_deterministic() {
        for d in [3usize, 5] {
            for basis in [Basis::Z, Basis::X] {
                for merged in [1usize, 2, d] {
                    let c = surgery(d, merged, basis, 0.0).circuit().unwrap();
                    let m = M2d::new(&c).unwrap_or_else(|e| panic!("d = {d} {basis:?} T = {merged}: {e}"));
                    assert_eq!(m.num_observables, if basis == Basis::Z { 3 } else { 1 });
                }
            }
        }
    }

    /// The patches are rotated codes: 2(d² − 1) checks between them, and the
    /// merged patch has d(2d + 1) − 1.
    #[test]
    fn the_codes_have_the_right_number_of_checks() {
        for d in [3usize, 5, 7] {
            let layout = Layout::new(d, 2, 2);
            let (a, b, c) = (layout.tile((0, 0)), layout.tile((1, 0)), layout.tile((0, 1)));
            assert_eq!(layout.code(a).len(), d * d - 1);
            assert_eq!(layout.code(b).len(), d * d - 1);
            // Merged side by side, and one above the other.
            assert_eq!(layout.code(TileBox::union(a, b)).len(), d * (2 * d + 1) - 1);
            assert_eq!(layout.code(TileBox::union(a, c)).len(), d * (2 * d + 1) - 1);
        }
    }

    /// With one merged round, a single measurement error on a new seam check
    /// flips the merge outcome with nothing after it to notice: the error-model
    /// builder refuses the circuit, naming an undetectable flip of L0 alone.
    /// With three, every single fault of the model is corrected.
    #[test]
    fn single_faults_need_more_than_one_merged_round() {
        let one = surgery(3, 1, Basis::Z, 0.001).circuit().unwrap();
        let err = Dem::from_circuit(&one).unwrap_err();
        assert!(err.contains("undetectable logical error") && err.contains("flips observables 0b1 "), "{err}");

        for basis in [Basis::Z, Basis::X] {
            let (failed, of) = uncorrected(&surgery(3, 3, basis, 0.001).circuit().unwrap());
            assert_eq!(failed, 0, "{basis:?}, T = 3: {failed} of {of} single faults uncorrected");
        }
    }
    /// Every program's detectors and observables are deterministic without
    /// noise. For the CNOT this checks the hand-derived Pauli frames: a frame
    /// missing a record leaves its observable random, and the reference
    /// simulation refuses it.
    #[test]
    fn programs_are_deterministic() {
        for d in [3usize, 5] {
            for basis in [Basis::Z, Basis::X] {
                for (name, prog) in [("cnot", cnot(d, d, 0.0, basis)), ("vertical", vertical(d, d, 0.0, basis))] {
                    let c = prog.circuit().unwrap();
                    M2d::new(&c).unwrap_or_else(|e| panic!("{name}, d = {d}, {basis:?}: {e}"));
                }
            }
            for (name, prog) in [("repeated", repeated(d, 3, d, 0.0)), ("line", line(d, 3, d, 0.0))] {
                M2d::new(&prog.circuit().unwrap()).unwrap_or_else(|e| panic!("{name}, d = {d}: {e}"));
            }
        }
    }

    /// The frame is exercised: without noise the two merge outcomes the CNOT's
    /// observables use each come out -1 in about half of shots, so a wrong
    /// frame could not hide behind outcomes that happened to be +1.
    #[test]
    fn the_cnot_s_outcomes_are_random() {
        use crate::frame_sampler::FrameSampler;
        for basis in [Basis::Z, Basis::X] {
            let mut prog = cnot(3, 3, 0.0, basis);
            let k = prog.observables.len();
            prog.observables.push(vec![Term::Outcome { merge: 0, seam: 0 }]);
            prog.observables.push(vec![Term::Outcome { merge: 1, seam: 0 }]);
            let sampler = FrameSampler::new(&prog.circuit().unwrap()).unwrap();
            let mut rng = crate::surface_code::Xorshift::new(4);
            let (mut ones0, mut ones1) = (0, 0);
            for _ in 0..400 {
                let shot = sampler.sample(&mut rng);
                ones0 += ((shot.observables >> k) & 1) as usize;
                ones1 += ((shot.observables >> (k + 1)) & 1) as usize;
            }
            assert!((120..=280).contains(&ones0) && (120..=280).contains(&ones1), "{basis:?}: {ones0}, {ones1} of 400");
        }
    }

    /// With d merged rounds, every single fault of every program is
    /// corrected: the CNOT in both input bases, the X⊗X mirror, repeated
    /// Z⊗Z and the three-patch product.
    #[test]
    fn every_program_corrects_every_single_fault() {
        let programs = [
            ("cnot, Z inputs", cnot(3, 3, 0.001, Basis::Z)),
            ("cnot, X inputs", cnot(3, 3, 0.001, Basis::X)),
            ("vertical, X", vertical(3, 3, 0.001, Basis::X)),
            ("vertical, Z", vertical(3, 3, 0.001, Basis::Z)),
            ("repeated, k = 3", repeated(3, 3, 3, 0.001)),
            ("line, n = 3", line(3, 3, 3, 0.001)),
        ];
        for (name, prog) in programs {
            let (failed, of) = uncorrected(&prog.circuit().unwrap());
            assert_eq!(failed, 0, "{name}: {failed} of {of} single faults uncorrected");
        }
    }

    /// Single faults of a circuit's model its matcher gets wrong, and how many there are.
    fn uncorrected(circuit: &Circuit) -> (usize, usize) {
        let dem = Dem::from_circuit(circuit).unwrap();
        let dec = DemDecoder::new(&dem).unwrap();
        let failed = dem
            .mechanisms
            .iter()
            .filter(|m| dec.decode(&m.detectors).map(|p| p.observables != m.observables).unwrap_or(true))
            .count();
        (failed, dem.mechanisms.len())
    }

    /// The CNOT is a CNOT. |0⟩|0⟩ and |+⟩|+⟩, which the error rates use, are
    /// left alone by the identity too, so three more noiseless programs pin
    /// the logical action. |+⟩|0⟩ must come out a Bell pair, Z_C Z_T and
    /// X_C X_T each fixed once the frame is applied, and |0⟩|+⟩ as it went in.
    /// The reference simulation accepts an observable only if it is
    /// deterministic. With the two input pairs above, this fixes the images
    /// of Z_C, Z_T, X_C and X_T as a CNOT's, up to signs. And the test has
    /// teeth: Z_T alone, which the identity would leave fixed, is refused.
    #[test]
    fn the_cnot_is_a_cnot() {
        use Basis::{X, Z};
        let [c, a, t] = CNOT;
        let logical = |patch, basis, line| Term::Logical { patch, basis, line };
        for d in [3usize, 5] {
            // Z_T and its frame (m₁, A's Z, the A–T seam's Z records on T's column).
            let z_t = vec![logical(t, Z, 0), logical(a, Z, 0), Term::Outcome { merge: 0, seam: 0 }, Term::Seam { merge: 1, patch: t, line: 0 }];
            // X_C X_T and its frame (m₂, the C–A seam's X record on C's last row).
            let x_cx_t = vec![logical(c, X, d - 1), logical(t, X, 0), Term::Outcome { merge: 1, seam: 0 }, Term::Seam { merge: 0, patch: c, line: d - 1 }];
            // Z_C along its last column, the one m₁ reads: with C in |+⟩ its
            // columns differ by Z checks that are not all fixed.
            let z_cz_t: Vec<Term> = std::iter::once(logical(c, Z, d - 1)).chain(z_t.iter().copied()).collect();
            let cases = [
                ("|+0⟩ read in Z", [X, Z], [Z, Z], vec![z_cz_t]),
                ("|+0⟩ read in X", [X, Z], [X, X], vec![x_cx_t]),
                ("|0+⟩", [Z, X], [Z, X], vec![vec![logical(c, Z, 0)], vec![logical(t, X, 0)]]),
            ];
            for (name, prep, read, observables) in cases {
                let circuit = cnot_with(d, d, 0.0, prep, read, observables).circuit().unwrap();
                M2d::new(&circuit).unwrap_or_else(|e| panic!("{name}, d = {d}: {e}"));
            }
            let circuit = cnot_with(d, d, 0.0, [X, Z], [Z, Z], vec![z_t]).circuit().unwrap();
            let err = M2d::new(&circuit).err().unwrap_or_default();
            assert!(err.contains("not deterministic"), "d = {d}: after |+0⟩, Z_T alone should be random: {err}");
        }
    }

    /// Patches merged in a row measure each neighbouring pair's parity, not
    /// one product: the merged patch of three holds one logical qubit, so
    /// two parities were taken from three. Each seam's outcome is fixed on
    /// its own with every patch in |0⟩.
    #[test]
    fn a_line_of_three_measures_each_seam() {
        for d in [3usize, 5] {
            let layout = Layout::new(d, 3, 1);
            let row = TileBox::union(layout.tile((0, 0)), layout.tile((2, 0)));
            assert_eq!(layout.code(row).len(), layout.data(row).len() - 1, "d = {d}: one logical qubit");
            let m = M2d::new(&line(d, 3, d, 0.0).circuit().unwrap()).unwrap();
            assert_eq!(m.num_observables, 2 + 3);
        }
    }

    /// A seam split and merged again at once, with no round between. The
    /// patches' boundary checks keep their support, but their seam qubits
    /// were read and prepared again, so those readings join the comparison;
    /// the seam's own checks, read in X and measuring Z, start over. Every
    /// detector stays deterministic and every single fault is corrected.
    #[test]
    fn a_seam_split_and_merged_again_at_once() {
        let d = 3;
        let both = || vec![0usize, 1];
        let prog = |p| Program {
            d,
            p,
            tiles: vec![(0, 0), (1, 0)],
            steps: vec![
                Step::Prepare { patches: both(), basis: Basis::Z },
                Step::Rounds(d),
                Step::Merge { patches: both() },
                Step::Rounds(d),
                Step::Split,
                Step::Merge { patches: both() },
                Step::Rounds(d),
                Step::Split,
                Step::Rounds(d),
                Step::Measure { patches: both(), basis: Basis::Z },
            ],
            observables: vec![
                vec![Term::Outcome { merge: 0, seam: 0 }],
                vec![Term::Outcome { merge: 1, seam: 0 }],
                vec![Term::Logical { patch: 0, basis: Basis::Z, line: 0 }],
                vec![Term::Logical { patch: 1, basis: Basis::Z, line: 0 }],
            ],
        };
        M2d::new(&prog(0.0).circuit().unwrap()).unwrap();
        let (failed, of) = uncorrected(&prog(0.001).circuit().unwrap());
        assert_eq!(failed, 0, "{failed} of {of} single faults uncorrected");
    }

    /// Programs that name nothing, name a patch twice, or ask for records
    /// that do not exist are refused, not compiled into something else.
    #[test]
    fn malformed_programs_are_errors() {
        type Spoil = fn(&mut Program);
        let cases: [(&str, Spoil); 5] = [
            ("lines", |p| p.observables.push(vec![Term::Seam { merge: 1, patch: 2, line: 3 }])),
            ("not in merge", |p| p.observables.push(vec![Term::Seam { merge: 1, patch: 0, line: 0 }])),
            ("no seam 1", |p| p.observables.push(vec![Term::Outcome { merge: 0, seam: 1 }])),
            ("twice", |p| p.steps[0] = Step::Prepare { patches: vec![0, 0, 2], basis: Basis::Z }),
            ("names no patch", |p| p.steps.push(Step::Measure { patches: vec![], basis: Basis::Z })),
        ];
        for (want, spoil) in cases {
            let mut prog = cnot(3, 3, 0.0, Basis::Z);
            spoil(&mut prog);
            let err = prog.circuit().err().unwrap_or_default();
            assert!(err.contains(want), "expected an error naming '{want}', got '{err}'");
        }
        assert!(cnot(0, 3, 0.0, Basis::X).circuit().unwrap_err().contains("odd distance"));
    }
}

//! Where a circuit's faults come from (Stim's `explain_detector_error_model_errors`): the error
//! model's backward walk, run in full and undecomposed, records for each fault the instruction
//! it came from (numbered as Stim numbers them, adjacent identical instructions joined), its
//! targets, the Pauli product or measurement it flips, the `TICK`s before it and the loop
//! passes around it. Faults are grouped by what they set off and written as Stim writes them.

use std::collections::HashMap;
use std::rc::Rc;

use crate::circuit::{fmt_args, instr_line, split_instruction, Circuit, Instr};
use crate::dem_program::OBS;

/// What a fault site knows: the targets it covers in its line, the Pauli product it applies
/// (qubit, 1 X, 2 Z, 3 Y, in target order), and the measurement it flips (record index and the
/// observable measured, inverted targets marked by bit 4 of the Pauli).
#[derive(Clone)]
pub(crate) struct Detail {
    pub range: (u32, u32),
    pub pauli: Vec<(u32, u8)>,
    pub meas: Option<(u64, Vec<(u32, u8)>)>,
    /// Stim's number for this case of its channel (cases of one application are listed in it).
    pub rank: u32,
    /// The instruction it belongs to, when not the one being undone (an `ELSE_CORRELATED_ERROR`
    /// case, added at its `E`).
    pub item: Option<Rc<Item>>,
}

/// Stim's number for a case of a channel: each qubit's Pauli (I 0, X 1, Y 2, Z 3), the first
/// target's the most significant (the engine's Paulis are 1 X, 2 Z, 3 Y).
pub(crate) fn rank(paulis: &[u8]) -> u32 {
    paulis.iter().fold(0, |acc, &p| acc * 4 + [0, 1, 3, 2][(p & 3) as usize])
}

/// A target range in an instruction, `[start, end)`.
pub(crate) type Range = (u32, u32);
/// Paulis on qubits (1 X, 2 Z, 3 Y; bit 4 an inverted target).
pub(crate) type Paulis = Vec<(u32, u8)>;

/// One instruction as Stim holds it: its number in its block (adjacent identical instructions
/// joined), where this line's targets start in it, and its text.
pub(crate) struct Item {
    index: u64,
    offset: u32,
    name: String,
    head: String,
    tag: String,
    args: Vec<f64>,
    tokens: Vec<String>,
}

impl Item {
    /// The measurement groups of a measuring line: each product of `MPP`, each pair of
    /// `MXX`/`MYY`/`MZZ`, each target otherwise; as (start, end) token ranges.
    pub(crate) fn groups(&self) -> Vec<(u32, u32)> {
        let n = self.tokens.len() as u32;
        match self.name.as_str() {
            "MPP" | "SPP" | "SPP_DAG" => {
                let mut out = Vec::new();
                let mut start = 0;
                let mut k = 0;
                while k < n {
                    if k + 1 < n && self.tokens[k as usize + 1] == "*" {
                        k += 2;
                        continue;
                    }
                    out.push((start, k + 1));
                    k += 1;
                    start = k;
                }
                out
            }
            "MXX" | "MYY" | "MZZ" => (0..n / 2).map(|k| (2 * k, 2 * k + 2)).collect(),
            _ => (0..n).map(|k| (k, k + 1)).collect(),
        }
    }

    /// The observable a measurement group measures, as Paulis on qubits (bit 4: inverted).
    pub(crate) fn measured(&self, (a, b): (u32, u32)) -> Vec<(u32, u8)> {
        let basis = match self.name.as_str() {
            "MX" | "MRX" | "MXX" => 1,
            "MY" | "MRY" | "MYY" => 3,
            _ => 2,
        };
        let mut out = Vec::new();
        for t in &self.tokens[a as usize..b as usize] {
            if t == "*" {
                continue;
            }
            let (inv, t) = match t.strip_prefix('!') {
                Some(rest) => (16u8, rest),
                None => (0, t.as_str()),
            };
            let (p, q) = match t.chars().next() {
                Some('X') => (1, &t[1..]),
                Some('Z') => (2, &t[1..]),
                Some('Y') => (3, &t[1..]),
                _ => (basis, t),
            };
            if let Ok(q) = q.parse::<u32>() {
                out.push((q, p | inv));
            }
        }
        out
    }
}

const UNJOINED: [&str; 8] = ["DETECTOR", "E", "ELSE_CORRELATED_ERROR", "OBSERVABLE_INCLUDE", "QUBIT_COORDS", "SHIFT_COORDS", "TICK", "REPEAT"];

/// A block's instructions as Stim holds them, one item per engine instruction.
fn block_items(instrs: &[Instr]) -> Vec<Rc<Item>> {
    let mut out: Vec<Rc<Item>> = Vec::with_capacity(instrs.len());
    let mut next = 0u64;
    for ins in instrs {
        let line = instr_line(ins);
        let (name, tag, args, tokens) = split_instruction(&line).unwrap_or_default();
        let head = format!("{name}{}{}", if tag.is_empty() { String::new() } else { format!("[{tag}]") }, if args.is_empty() { String::new() } else { format!("({})", fmt_args(&args)) });
        let tokens = split_products(&tokens);
        let joined = out.last().filter(|p| p.head == head && !UNJOINED.contains(&name.as_str()));
        let (index, offset) = match joined {
            Some(p) => (p.index, p.offset + p.tokens.len() as u32),
            None => {
                next += 1;
                (next - 1, 0)
            }
        };
        out.push(Rc::new(Item { index, offset, name, head, tag: tag.to_string(), args, tokens }));
    }
    out
}

/// Targets as Stim counts them: `X0*Z1` is three (the combiner counts).
fn split_products(tokens: &[&str]) -> Vec<String> {
    let mut out = Vec::new();
    for t in tokens {
        let mut first = true;
        for part in t.split('*') {
            if !first {
                out.push("*".to_string());
            }
            first = false;
            out.push(part.to_string());
        }
    }
    out
}

/// One place a fault arises.
#[derive(Clone, Debug, PartialEq)]
pub struct Location {
    pub tick_offset: u64,
    pub pauli: Vec<(u32, u8)>,
    pub measurement: Option<(u64, Vec<(u32, u8)>)>,
    pub gate: String,
    pub args: Vec<f64>,
    pub head: String,
    /// The targets this fault covers, as written, and their range in the instruction.
    pub targets: Vec<String>,
    pub range: (u32, u32),
    /// (instruction offset, iteration, repetitions) from the outermost block in.
    pub stack: Vec<(u64, u64, u64)>,
    pub tag: String,
}

/// A fault class (what it sets off) and every place in the circuit such a fault arises.
#[derive(Clone, Debug, PartialEq)]
pub struct Explained {
    /// Detectors, then observables as `OBS | k`.
    pub terms: Vec<u64>,
    pub locations: Vec<Location>,
}

struct GateCtx {
    groups: Vec<(u32, u32)>,
    seen: usize,
}

/// The walk's record of where it is and what it has met.
pub(crate) struct Recorder {
    pub(crate) ticks: u64,
    frames: Vec<(u64, u64, u64)>,
    current: Option<Rc<Item>>,
    gates: Vec<GateCtx>,
    views: HashMap<usize, Rc<Vec<Rc<Item>>>>,
    pub(crate) records: Vec<(Vec<u64>, Location, u32)>,
}

impl Recorder {
    pub(crate) fn new(ticks: u64) -> Recorder {
        Recorder { ticks, frames: Vec::new(), current: None, gates: Vec::new(), views: HashMap::new(), records: Vec::new() }
    }

    /// The items of a block (cached: loops visit their bodies again and again).
    pub(crate) fn view(&mut self, instrs: &[Instr]) -> Rc<Vec<Rc<Item>>> {
        let key = instrs.as_ptr() as usize ^ (instrs.len() << 48);
        self.views.entry(key).or_insert_with(|| Rc::new(block_items(instrs))).clone()
    }

    /// The instruction being undone (not inside a decomposed gate, whose line stays current).
    pub(crate) fn enter(&mut self, item: &Rc<Item>) {
        if self.gates.is_empty() {
            self.current = Some(item.clone());
        }
    }

    pub(crate) fn current_index(&self) -> u64 {
        self.current.as_ref().map_or(0, |c| c.index)
    }

    pub(crate) fn push_frame(&mut self, index: u64, iteration: u64, repetitions: u64) {
        self.frames.push((index, iteration, repetitions));
    }

    pub(crate) fn pop_frame(&mut self) {
        self.frames.pop();
    }

    /// Entering a decomposed gate: its measurements are its line's groups, met last first.
    pub(crate) fn enter_gate(&mut self) {
        let groups = self.current.as_ref().map_or_else(Vec::new, |c| c.groups());
        self.gates.push(GateCtx { groups, seen: 0 });
    }

    pub(crate) fn leave_gate(&mut self) {
        self.gates.pop();
    }

    pub(crate) fn in_gate(&self) -> bool {
        !self.gates.is_empty()
    }

    /// Inside a decomposed gate, the next measurement met (walking backwards): its group's
    /// range and observable.
    pub(crate) fn gate_measurement(&mut self) -> Option<(Range, Paulis)> {
        let current = self.current.clone()?;
        let g = self.gates.last_mut()?;
        g.seen += 1;
        let k = g.groups.len().checked_sub(g.seen)?;
        let range = g.groups[k];
        Some((range, current.measured(range)))
    }

    /// The instruction being undone.
    pub(crate) fn current(&self) -> Option<Rc<Item>> {
        self.current.clone()
    }

    /// The current line's whole target range.
    pub(crate) fn whole(&self) -> (u32, u32) {
        (0, self.current.as_ref().map_or(0, |c| c.tokens.len() as u32))
    }

    pub(crate) fn record(&mut self, symptom: Vec<u64>, d: Detail) {
        if symptom.is_empty() {
            return;
        }
        let Some(item) = d.item.clone().or_else(|| self.current.clone()) else { return };
        let (a, b) = (d.range.0.min(item.tokens.len() as u32), d.range.1.min(item.tokens.len() as u32));
        let loc = Location {
            tick_offset: self.ticks,
            pauli: d.pauli,
            measurement: d.meas,
            gate: item.name.clone(),
            args: item.args.clone(),
            head: item.head.clone(),
            targets: item.tokens[a as usize..b as usize].to_vec(),
            range: (item.offset + a, item.offset + b),
            stack: {
                let mut s = self.frames.clone();
                s.push((item.index, 0, 0));
                s
            },
            tag: item.tag.clone(),
        };
        self.records.push((symptom, loc, d.rank));
    }
}

/// Coordinates for printing: each qubit's last `QUBIT_COORDS`, each detector's.
#[derive(Clone, Debug, Default)]
pub struct Coords {
    pub qubits: HashMap<u32, Vec<f64>>,
    pub detectors: Vec<Vec<f64>>,
}

impl Coords {
    pub fn of(circuit: &Circuit) -> Result<Coords, String> {
        let mut qubits = HashMap::new();
        for ins in &circuit.flattened().instrs {
            let mut visit = |i: &Instr| {
                if let Instr::QubitCoords { coords, qubits: qs } = i {
                    for &q in qs {
                        qubits.insert(q, coords.clone());
                    }
                }
            };
            match ins {
                Instr::Gate { body, .. } => body.iter().for_each(&mut visit),
                other => visit(other),
            }
        }
        Ok(Coords { qubits, detectors: circuit.resolve()?.detector_coords })
    }

    pub(crate) fn at(c: &[f64]) -> String {
        c.iter().map(|&x| fmt_args(&[x])).collect::<Vec<_>>().join(",")
    }

    /// A qubit's coordinates (empty for none).
    pub fn of_qubit(&self, q: u32) -> Vec<f64> {
        self.qubits.get(&q).cloned().unwrap_or_default()
    }

    /// A detector's coordinates (empty for none).
    pub fn of_detector(&self, d: u64) -> Vec<f64> {
        self.detectors.get(d as usize).cloned().unwrap_or_default()
    }

    fn qubit(&self, q: u32) -> String {
        match self.qubits.get(&q).filter(|c| !c.is_empty()) {
            Some(c) => format!("{q}[coords {}]", Coords::at(c)),
            None => q.to_string(),
        }
    }

    fn term(&self, t: u64) -> String {
        if t & OBS != 0 {
            return format!("L{}", t & !OBS);
        }
        match self.detectors.get(t as usize).filter(|c| !c.is_empty()) {
            Some(c) => format!("D{t}[coords {}]", Coords::at(c)),
            None => format!("D{t}"),
        }
    }

    fn paulis(&self, ps: &[(u32, u8)]) -> String {
        ps.iter()
            .map(|&(q, p)| format!("{}{}{}", if p & 16 != 0 { "!" } else { "" }, ["I", "X", "Z", "Y"][(p & 3) as usize], self.qubit(q)))
            .collect::<Vec<_>>()
            .join("*")
    }

    /// A target token with its coordinates (`X3` → `X3[coords 1,2]`).
    fn token(&self, t: &str) -> String {
        let body = t.trim_start_matches('!');
        let lead = &t[..t.len() - body.len()];
        let (p, q) = match body.chars().next() {
            Some(c @ ('X' | 'Y' | 'Z')) if body[1..].parse::<u32>().is_ok() => (c.to_string(), &body[1..]),
            _ => (String::new(), body),
        };
        match q.parse::<u32>() {
            Ok(q) => format!("{lead}{p}{}", self.qubit(q)),
            Err(_) => t.to_string(),
        }
    }
}

impl Location {
    /// Stim's text for it (`str(stim.CircuitErrorLocation)`).
    pub fn to_stim(&self, coords: &Coords) -> String {
        let mut s = String::from("CircuitErrorLocation {\n");
        if !self.tag.is_empty() {
            s += &format!("    noise_tag: {}\n", self.tag);
        }
        if !self.pauli.is_empty() {
            s += &format!("    flipped_pauli_product: {}\n", coords.paulis(&self.pauli));
        }
        if let Some((index, observable)) = &self.measurement {
            s += &format!("    flipped_measurement.measurement_record_index: {index}\n");
            if !observable.is_empty() {
                s += &format!("    flipped_measurement.measured_observable: {}\n", coords.paulis(observable));
            }
        }
        s += "    Circuit location stack trace:\n";
        s += &format!("        (after {} TICKs)\n", self.tick_offset);
        for (depth, &(index, iteration, reps)) in self.stack.iter().enumerate() {
            let place = if depth == 0 { "the circuit" } else { "the REPEAT block" };
            let what = if depth + 1 < self.stack.len() { format!("a REPEAT {reps} block") } else { self.gate.clone() };
            s += &format!("        at instruction #{} ({what}) in {place}\n", index + 1);
            if depth + 1 < self.stack.len() {
                s += &format!("        after {iteration} completed iterations\n");
            }
        }
        let (a, b) = self.range;
        if b == a + 1 {
            s += &format!("        at target #{} of the instruction\n", a + 1);
        } else {
            s += &format!("        at targets #{} to #{} of the instruction\n", a + 1, b);
        }
        s += &format!("        resolving to {}\n", self.instruction_text(coords));
        s += "}";
        s
    }

    /// The instruction and the targets it covers, with coordinates
    /// (`str(stim.CircuitTargetsInsideInstruction)`).
    pub fn instruction_text(&self, coords: &Coords) -> String {
        let mut shown = String::new();
        for (k, t) in self.targets.iter().enumerate() {
            if t != "*" && k > 0 && self.targets[k - 1] != "*" {
                shown.push(' ');
            }
            shown += &if t == "*" { "*".to_string() } else { coords.token(t) };
        }
        format!("{} {}", self.head, shown)
    }
}

impl Explained {
    /// Stim's text for it.
    pub fn to_stim(&self, coords: &Coords) -> String {
        let mut s = String::from("ExplainedError {\n");
        let terms: Vec<String> = self.terms.iter().map(|&t| coords.term(t)).collect();
        s += &format!("    dem_error_terms: {}\n", terms.join(" "));
        if self.locations.is_empty() {
            s += "    [no single circuit error had these exact symptoms]\n";
        }
        for loc in &self.locations {
            for line in loc.to_stim(coords).lines() {
                s += &format!("    {line}\n");
            }
        }
        s += "}";
        s
    }
}

/// The circuit's faults explained: for each fault class of its (undecomposed, unfolded) error
/// model, or of `filter`'s faults when given, every place it arises, or with `reduce` one.
pub fn explain(circuit: &Circuit, filter: Option<&[Vec<u64>]>, reduce: bool) -> Result<Vec<Explained>, String> {
    let mut records = crate::dem_build::provenance(circuit)?;
    // One instruction's faults in Stim's order of its targets, each target's cases in Stim's
    // numbering of them.
    let mut start = 0;
    while start < records.len() {
        let mut end = start + 1;
        while end < records.len() && records[end].1.stack == records[start].1.stack && records[end].1.gate == records[start].1.gate {
            end += 1;
        }
        // Stim meets a pair channel pair by pair, first to last, each pair's cases in turn;
        // a one-qubit channel case by case, each case's targets last first.
        let pairs = matches!(records[start].1.gate.as_str(), "DEPOLARIZE2" | "PAULI_CHANNEL_2");
        if pairs {
            records[start..end].sort_by_key(|r| (r.1.range.0, r.2));
        } else {
            records[start..end].sort_by_key(|r| (r.2, std::cmp::Reverse(r.1.range.0)));
        }
        start = end;
    }
    let mut groups: Vec<(Vec<u64>, Vec<Location>)> = Vec::new();
    let mut at: HashMap<Vec<u64>, usize> = HashMap::new();
    for (symptom, loc, _) in records {
        let k = *at.entry(symptom.clone()).or_insert_with(|| {
            groups.push((symptom, Vec::new()));
            groups.len() - 1
        });
        groups[k].1.push(loc);
    }
    let mut out: Vec<Explained> = match filter {
        Some(terms) => terms
            .iter()
            .map(|t| {
                let mut key = t.clone();
                key.sort_unstable();
                let locations = at.get(&key).map(|&k| groups[k].1.clone()).unwrap_or_default();
                Explained { terms: key, locations }
            })
            .collect(),
        None => {
            let mut all: Vec<Explained> = groups.into_iter().map(|(terms, locations)| Explained { terms, locations }).collect();
            all.sort_by(|a, b| a.terms.cmp(&b.terms));
            all
        }
    };
    if reduce {
        for e in &mut out {
            if let Some(best) = representative(&e.locations) {
                e.locations = vec![best];
            }
        }
    }
    Ok(out)
}

/// Stim's representative of a fault class: its simplest location (fewest qubits and flipped
/// measurements), then the earliest, then the least Pauli product, the targets' place in the
/// instruction, the least targets, arguments and gate type in Stim's order; among equals with one tag the first in
/// the circuit, among tags the last.
fn representative(locs: &[Location]) -> Option<Location> {
    let size = |l: &Location| l.pauli.len() + usize::from(l.measurement.is_some());
    // Stim compares targets by their bits: a Z flag below an X flag below both (Y), then qubit.
    let product = |l: &Location| l.pauli.iter().map(|&(q, p)| ([0u8, 2, 1, 3][(p & 3) as usize], q)).collect::<Vec<_>>();
    // Arguments compare as numbers (they are never negative, so their bits order alike).
    let args = |l: &Location| l.args.iter().map(|a| a.to_bits()).collect::<Vec<_>>();
    // Targets compare as Stim stores them: Pauli flags above the qubit (Z < X < Y), `!` above all.
    let targets = |l: &Location| {
        l.targets
            .iter()
            .map(|t| {
                let inv = u8::from(t.starts_with('!'));
                let t = t.trim_start_matches('!');
                let (flag, q) = match t.as_bytes().first() {
                    Some(b'Z') => (1u8, &t[1..]),
                    Some(b'X') => (2, &t[1..]),
                    Some(b'Y') => (3, &t[1..]),
                    _ => (0, t),
                };
                (inv, flag, q.parse::<u64>().unwrap_or(u64::MAX))
            })
            .collect::<Vec<_>>()
    };
    let key = |l: &Location| (size(l), l.tick_offset, product(l), l.range, targets(l), args(l), gate_rank(&l.gate));
    // Stim reduces each tag's faults on their own (the earliest of equals), and a later tag's
    // choice replaces an equal earlier one.
    let mut tags: Vec<&str> = locs.iter().map(|l| l.tag.as_str()).collect();
    tags.sort_unstable();
    tags.dedup();
    let per_tag = tags.iter().filter_map(|t| locs.iter().filter(|l| l.tag == *t).min_by_key(|l| (key(l), l.stack.clone(), l.range)));
    per_tag.min_by_key(|l| (key(l), std::cmp::Reverse(l.stack.clone()))).cloned()
}

/// Stim's order of its gate types (its `GateType` enumeration), for the gates that carry faults.
fn gate_rank(gate: &str) -> u32 {
    const ORDER: [&str; 26] = [
        "MPAD", "MX", "MY", "M", "MRX", "MRY", "MR", "RX", "RY", "R", "DEPOLARIZE1", "DEPOLARIZE2", "X_ERROR", "Y_ERROR", "Z_ERROR", "I_ERROR", "II_ERROR",
        "PAULI_CHANNEL_1", "PAULI_CHANNEL_2", "E", "ELSE_CORRELATED_ERROR", "HERALDED_ERASE", "HERALDED_PAULI_CHANNEL_1", "MPP", "MXX", "MYY",
    ];
    ORDER.iter().position(|&g| g == gate).map_or(u32::MAX - 1, |k| k as u32)
}

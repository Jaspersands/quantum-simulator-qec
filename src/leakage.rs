//! Deltakit's leakage dialect (Riverlane's fork of Stim): `LEAKAGE(p)`, `RELAX(p)` and
//! `HERALD_LEAKAGE_EVENT(p)`, read as tagged instructions that Stim runs, and the circuit
//! without any leakage.
//!
//! WHY THIS EXISTS
//! ---------------
//! Deltakit writes leakage into circuits with three instructions of its own. `LEAKAGE(p) q`
//! marks q leaked with probability p and fully depolarizes it; `RELAX(p) q` returns a leaked q
//! to the computational space with probability p; `HERALD_LEAKAGE_EVENT(p) q` writes whether
//! q is leaked into the measurement record (each result flipped with probability p, `!q`
//! inverted), where a `DETECTOR` can read it. Plain Stim reads none of them. Here each line is
//! lowered as it is read into a tagged line that Stim does read: `I_ERROR[LEAKAGE(p=…)] q`,
//! `I_ERROR[RELAX(p=…)] q` and `MPAD[HERALD_LEAKAGE_EVENT(qubits=q …)](p) 0 …`, a padded
//! record that reads 0 when nothing leaks, as Deltakit's herald does. Stim, and this package's
//! Clifford engines and error models, run the circuit as if nothing leaked; the leakage sampler
//! runs Deltakit's model; `to_deltakit` writes the dialect back.
//!
//! `without_leakage` is Deltakit's deinstrumentation (`Measurements._deinstrument_leakage_circuit`,
//! which turns its measurements into detection events): no leakage noise, no herald and no
//! detector reading one, every other lookback renumbered. Deltakit's works line by line on the
//! text, so it renumbers a loop's lookbacks as if the loop ran once; this one keeps loops,
//! peeling the first iterations off a loop whose lookbacks reach before it, and leaves an
//! observable its other records where Deltakit drops the line.
//!
//! `LeakageAware` decodes with the heralds: around any decoder that weighs faults by their
//! priors (the machinery argument reweighting uses), a shot's fired triggers raise the priors
//! of the faults their qubits' leakage makes likely, and the shot is decoded once.

use std::collections::VecDeque;

use crate::clifford::ir::{self, GateTarget, Instruction, Item};
use crate::nonpauli::NonPauli;

const NAMES: [&str; 3] = ["LEAKAGE", "RELAX", "HERALD_LEAKAGE_EVENT"];

/// The tag of a lowered herald.
pub const HERALD: &str = "HERALD_LEAKAGE_EVENT";

/// A line of Deltakit's dialect (its comment stripped) as the tagged line it is read as; `None`
/// for any other line.
pub fn lower_line(line: &str) -> Result<Option<String>, String> {
    let trimmed = line.trim_start();
    let end = trimmed.find(|c: char| !(c.is_ascii_alphanumeric() || c == '_')).unwrap_or(trimmed.len());
    let name = trimmed[..end].to_ascii_uppercase();
    if !NAMES.contains(&name.as_str()) {
        return Ok(None);
    }
    let indent = &line[..line.len() - trimmed.len()];
    let mut rest = &trimmed[end..];
    if rest.starts_with('[') {
        return Err(format!("{name}: Deltakit's instruction takes no tag"));
    }
    let mut args: Vec<&str> = Vec::new();
    if let Some(inner) = rest.trim_start().strip_prefix('(') {
        let close = inner.find(')').ok_or_else(|| format!("{name}: unclosed '('"))?;
        args = inner[..close].split(',').map(str::trim).filter(|a| !a.is_empty()).collect();
        rest = &inner[close + 1..];
    } else if !rest.is_empty() && !rest.starts_with(char::is_whitespace) {
        return Err(format!("{name}: targets must be separated by spacing"));
    }
    for a in &args {
        match a.parse::<f64>() {
            Ok(p) if (0.0..=1.0).contains(&p) => {}
            _ => return Err(format!("{name}: '{a}' is not a probability")),
        }
    }
    let targets: Vec<&str> = rest.split_whitespace().collect();
    let qubit = |t: &str| -> Result<(u32, bool), String> {
        let (q, inverted) = match t.strip_prefix('!') {
            Some(q) => (q, true),
            None => (t, false),
        };
        match crate::circuit::uint::<u32>(q) {
            Some(q) if q < 1 << 24 => Ok((q, inverted)),
            _ => Err(format!("{name}: '{t}' is not a qubit")),
        }
    };
    let qubits = targets.iter().map(|t| qubit(t)).collect::<Result<Vec<_>, _>>()?;
    let list = |f: &dyn Fn(&(u32, bool)) -> String| qubits.iter().map(f).collect::<Vec<_>>().join(" ");
    if name == "HERALD_LEAKAGE_EVENT" {
        if args.len() > 1 {
            return Err(format!("{name} takes at most one argument, the probability of flipping its result"));
        }
        let flip = args.first().map(|a| format!("({a})")).unwrap_or_default();
        let values = list(&|&(_, inv)| if inv { "1".into() } else { "0".into() });
        return Ok(Some(format!("{indent}MPAD[{HERALD}(qubits={})]{flip} {values}", list(&|&(q, _)| q.to_string())).trim_end().to_string()));
    }
    if args.len() != 1 {
        return Err(format!("{name} takes one argument, its probability"));
    }
    if qubits.iter().any(|q| q.1) {
        return Err(format!("{name}: a noise channel's target is not inverted"));
    }
    Ok(Some(format!("{indent}I_ERROR[{name}(p={})] {}", args[0], list(&|&(q, _)| q.to_string())).trim_end().to_string()))
}

/// The qubits a herald's tag names, if `tag` is one.
pub fn herald_qubits(tag: &str) -> Option<Result<Vec<u32>, String>> {
    let inner = tag.trim().strip_prefix(HERALD)?.trim();
    let inner = inner.strip_prefix('(').and_then(|s| s.strip_suffix(')'))?;
    let qubits = inner.trim().strip_prefix("qubits")?.trim_start().strip_prefix('=')?;
    Some(qubits.split_whitespace().map(|q| crate::circuit::uint::<u32>(q).filter(|&q| q < 1 << 24).ok_or_else(|| format!("{HERALD}: '{q}' is not a qubit"))).collect())
}

/// A herald `MPAD`'s markers: one per padded record, the qubit it reads. `values` are the
/// pads (1 for an inverted target).
pub fn herald_ops(tag: &str, values: usize) -> Result<Option<Vec<NonPauli>>, String> {
    let Some(qubits) = herald_qubits(tag) else {
        return Ok(None);
    };
    let qubits = qubits?;
    if qubits.len() != values {
        return Err(format!("MPAD[{tag}]: {} qubits for {values} records", qubits.len()));
    }
    Ok(Some(qubits.into_iter().map(|qubit| NonPauli::Herald { qubit }).collect()))
}

/// Text with every lowered line written back in Deltakit's dialect.
pub fn to_deltakit(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for line in text.lines() {
        out.push_str(&raise_line(line).unwrap_or_else(|| line.to_string()));
        out.push('\n');
    }
    out
}

fn raise_line(line: &str) -> Option<String> {
    let trimmed = line.trim_start();
    let indent = &line[..line.len() - trimmed.len()];
    let code = crate::circuit::strip_comment(trimmed).trim_end();
    let (name, tag, args, targets) = crate::circuit::split_instruction(code).ok()?;
    let arg = |v: f64| format!("{v:?}");
    match name.as_str() {
        "I_ERROR" => {
            let ops = crate::nonpauli::parse("I_ERROR", tag, &targets.iter().map(|t| t.parse().ok()).collect::<Option<Vec<u32>>>()?).ok()??;
            let (word, p) = match ops.first()? {
                NonPauli::Leakage { p, .. } => ("LEAKAGE", *p),
                NonPauli::Relax { p, .. } => ("RELAX", *p),
                _ => return None,
            };
            Some(format!("{indent}{word}({}) {}", arg(p), targets.join(" ")))
        }
        "MPAD" => {
            let qubits = herald_qubits(tag)?.ok()?;
            if qubits.len() != targets.len() {
                return None;
            }
            let flip = args.first().map(|&p| format!("({})", arg(p))).unwrap_or_default();
            let t: Vec<String> = qubits.iter().zip(&targets).map(|(q, v)| format!("{}{q}", if *v == "1" { "!" } else { "" })).collect();
            Some(format!("{indent}HERALD_LEAKAGE_EVENT{flip} {}", t.join(" ")).trim_end().to_string())
        }
        _ => None,
    }
}

fn targets_u32(op: &Instruction) -> Option<Vec<u32>> {
    op.targets.iter().map(|t| t.is_qubit().then(|| t.value())).collect()
}

/// Whether `op` is leakage noise (LEAK, SEEP, LEAK_TRANSPORT, LEAKAGE, RELAX).
fn is_leakage_noise(op: &Instruction) -> bool {
    if op.tag.is_empty() || !matches!(op.gate.name, "I" | "II" | "I_ERROR" | "II_ERROR") {
        return false;
    }
    let Some(t) = targets_u32(op) else {
        return false;
    };
    matches!(crate::nonpauli::parse(op.gate.name, &op.tag, &t), Ok(Some(ops)) if !ops.is_empty() && ops.iter().all(|o| o.is_leakage()))
}

fn is_herald(op: &Instruction) -> bool {
    op.gate.name == "MPAD" && herald_qubits(&op.tag).is_some()
}

/// The last records made, each whether it is a herald, as far back as any lookback reads.
struct Window {
    flags: VecDeque<bool>,
    cap: usize,
}

impl Window {
    fn push(&mut self, herald: bool, n: u64) {
        for _ in 0..n.min(self.cap as u64 + 1) {
            if self.flags.len() == self.cap {
                self.flags.pop_front();
            }
            self.flags.push_back(herald);
        }
    }

    /// Lookback `k` after removing heralds: `None` if it reads one.
    fn remap(&self, k: u32) -> Result<Option<u32>, String> {
        let k = k as usize;
        if k == 0 || k > self.flags.len() {
            return Err(format!("rec[-{k}] reaches before the first measurement"));
        }
        let at = self.flags.len() - k;
        if self.flags[at] {
            return Ok(None);
        }
        Ok(Some(self.flags.range(at..).filter(|&&h| !h).count() as u32))
    }
}

fn max_lookback(c: &ir::Circuit) -> u32 {
    c.items
        .iter()
        .map(|it| match it {
            Item::Op(op) => op.targets.iter().filter(|t| t.is_record()).map(|t| t.value()).max().unwrap_or(0),
            Item::Repeat { body, .. } => max_lookback(body),
        })
        .max()
        .unwrap_or(0)
}

fn records(c: &ir::Circuit) -> u64 {
    c.count_measurements()
}

fn strip(c: &ir::Circuit, w: &mut Window) -> Result<ir::Circuit, String> {
    let mut out = ir::Circuit::new();
    for it in &c.items {
        match it {
            Item::Op(op) if is_leakage_noise(op) => {}
            Item::Op(op) if is_herald(op) => w.push(true, op.count_measurement_results()),
            Item::Op(op) => {
                if let Some(op) = renumbered(op, w)? {
                    out.items.push(Item::Op(op));
                }
                w.push(false, op.count_measurement_results());
            }
            Item::Repeat { count, body, tag } => {
                let r = records(body);
                // From iteration `stable` on, every lookback stays inside the loop, so each
                // iteration is the same.
                let stable = if r == 0 { 1 } else { 1 + (w.cap as u64).div_ceil(r) };
                let runs = (*count).min(stable);
                let bodies = (0..runs).map(|_| strip(body, w)).collect::<Result<Vec<_>, _>>()?;
                let last = bodies.last().expect("a loop runs at least once");
                let first_same = bodies.iter().rposition(|b| b != last).map_or(0, |i| i + 1);
                for b in &bodies[..first_same] {
                    out.items.extend(b.items.iter().cloned());
                }
                let rest = count - first_same as u64;
                if !last.items.is_empty() {
                    if rest == 1 && tag.is_empty() {
                        out.items.extend(last.items.iter().cloned());
                    } else {
                        out.push_raw_repeat(rest, last.clone(), tag);
                    }
                }
            }
        }
    }
    Ok(out)
}

/// `op` with its lookbacks renumbered: a detector reading a herald is dropped, as are an
/// observable's herald records and a classically controlled pair on one.
fn renumbered(op: &Instruction, w: &Window) -> Result<Option<Instruction>, String> {
    if !op.targets.iter().any(|t| t.is_record()) {
        return Ok(Some(op.clone()));
    }
    let mut targets = Vec::with_capacity(op.targets.len());
    match op.gate.name {
        "DETECTOR" => {
            for t in &op.targets {
                match t.is_record() {
                    true => match w.remap(t.value())? {
                        Some(k) => targets.push(GateTarget::rec(k)),
                        None => return Ok(None),
                    },
                    false => targets.push(*t),
                }
            }
        }
        "OBSERVABLE_INCLUDE" => {
            for t in &op.targets {
                if !t.is_record() {
                    targets.push(*t);
                } else if let Some(k) = w.remap(t.value())? {
                    targets.push(GateTarget::rec(k));
                }
            }
        }
        _ => {
            // Classically controlled gates: pairs, one of them a record.
            for pair in op.targets.chunks(2) {
                let mut kept = Vec::with_capacity(2);
                let mut drop = false;
                for t in pair {
                    if t.is_record() {
                        match w.remap(t.value())? {
                            Some(k) => kept.push(GateTarget::rec(k)),
                            None => drop = true,
                        }
                    } else {
                        kept.push(*t);
                    }
                }
                if !drop {
                    targets.extend(kept);
                }
            }
            if targets.is_empty() {
                return Ok(None);
            }
        }
    }
    Ok(Some(Instruction { targets, ..op.clone() }))
}

/// The circuit with no leakage noise, no herald and no detector reading one (Deltakit's
/// deinstrumentation, loops kept).
pub fn without_leakage(c: &ir::Circuit) -> Result<ir::Circuit, String> {
    let mut w = Window { flags: VecDeque::new(), cap: max_lookback(c) as usize };
    strip(c, &mut w)
}

/// Where a circuit's heralds are, by absolute index.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct HeraldMap {
    /// The heralds' records, in order (Deltakit's leakage flags are these columns).
    pub herald_records: Vec<u64>,
    /// The detectors `without_leakage` keeps, in order.
    pub kept_detectors: Vec<u64>,
    /// Detectors that read exactly one herald and nothing else: (detector, its record).
    pub herald_detectors: Vec<(u64, u64)>,
}

/// The circuit's herald records and which of its detectors read them.
pub fn herald_map(c: &ir::Circuit) -> Result<HeraldMap, String> {
    let mut map = HeraldMap::default();
    let mut herald = Vec::<bool>::new();
    let mut detector = 0u64;
    let mut err = None;
    c.for_each_operation(&mut |op| {
        if err.is_some() {
            return;
        }
        let n = op.count_measurement_results();
        if is_herald(op) {
            for _ in 0..n {
                map.herald_records.push(herald.len() as u64);
                herald.push(true);
            }
            return;
        }
        if op.gate.name == "DETECTOR" {
            let mut recs = Vec::new();
            for t in op.targets.iter().filter(|t| t.is_record()) {
                let k = t.value() as usize;
                if k == 0 || k > herald.len() {
                    err = Some(format!("rec[-{k}] reaches before the first measurement"));
                    return;
                }
                recs.push(herald.len() - k);
            }
            let heralds: Vec<usize> = recs.iter().copied().filter(|&r| herald[r]).collect();
            if heralds.is_empty() {
                map.kept_detectors.push(detector);
            } else if recs.len() == 1 {
                map.herald_detectors.push((detector, heralds[0] as u64));
            }
            detector += 1;
        }
        herald.extend(std::iter::repeat_n(false, n as usize));
    });
    match err {
        Some(e) => Err(e),
        None => Ok(map),
    }
}

/// Leakage-aware decoding from heralds, around any decoder that weighs faults by their priors
/// (matching, union-find, BP+OSD, BP+LSD, Relay-BP): each trigger (a measurement that found its
/// qubit leaked, or a herald) names the faults its qubit's leakage makes likely, with their
/// probability given it; a shot whose triggers fire has those faults' priors raised to that
/// (combined with their own as independent causes), and is decoded once, the priors put back
/// after.
pub struct LeakageAware {
    pub dec: crate::reweight::PriorDecoder,
    triggers: Vec<Vec<(u32, f64)>>,
}

impl LeakageAware {
    pub fn new(dec: crate::reweight::PriorDecoder, triggers: Vec<Vec<(u32, f64)>>) -> Result<LeakageAware, String> {
        let n = dec.num_faults();
        for (t, faults) in triggers.iter().enumerate() {
            for &(f, q) in faults {
                if f as usize >= n {
                    return Err(format!("trigger {t} names fault {f}; the decoder has {n}"));
                }
                if !(0.0..=1.0).contains(&q) {
                    return Err(format!("trigger {t}: probability {q} is outside [0, 1]"));
                }
            }
        }
        Ok(LeakageAware { dec, triggers })
    }

    pub fn num_triggers(&self) -> usize {
        self.triggers.len()
    }

    /// The faults of each trigger.
    pub fn trigger_sizes(&self) -> Vec<usize> {
        self.triggers.iter().map(Vec::len).collect()
    }

    /// One shot: its detection events as sorted `defects` and one byte per detector, and the
    /// triggers that fired. The predicted observables (`None` where no correction explains the
    /// shot) and how many priors were raised.
    pub fn decode(&self, defects: &[u32], syndrome: &[u8], fired: &[u32], shot: u64, w: &mut crate::reweight::ArWork) -> (Option<u64>, u32) {
        let mut raised = 0u32;
        for &t in fired {
            for &(f, q) in &self.triggers[t as usize] {
                let p = self.dec.prior(w, f);
                self.dec.set(w, f, p * (1.0 - q) + q * (1.0 - p));
                raised += 1;
            }
        }
        let out = self.dec.round(w, defects, syndrome, self.dec.seed(shot)).map(|r| r.0);
        self.dec.restore(w);
        (out, raised)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ir(text: &str) -> ir::Circuit {
        ir::Circuit::parse(text).unwrap()
    }

    #[test]
    fn deltakit_lines_lower_and_raise() {
        assert_eq!(lower_line("LEAKAGE(0.01) 0 3").unwrap().unwrap(), "I_ERROR[LEAKAGE(p=0.01)] 0 3");
        assert_eq!(lower_line("  relax(0.5) 2").unwrap().unwrap(), "  I_ERROR[RELAX(p=0.5)] 2");
        assert_eq!(lower_line("HERALD_LEAKAGE_EVENT 0 !4").unwrap().unwrap(), "MPAD[HERALD_LEAKAGE_EVENT(qubits=0 4)] 0 1");
        assert_eq!(lower_line("HERALD_LEAKAGE_EVENT(0.02) 1").unwrap().unwrap(), "MPAD[HERALD_LEAKAGE_EVENT(qubits=1)](0.02) 0");
        assert_eq!(lower_line("M 0").unwrap(), None);
        assert_eq!(lower_line("LEAKAGES 0").unwrap(), None);
        assert!(lower_line("LEAKAGE 0").unwrap_err().contains("one argument"));
        assert!(lower_line("LEAKAGE(2) 0").unwrap_err().contains("probability"));
        assert!(lower_line("RELAX(0.1) !0").unwrap_err().contains("inverted"));
        assert!(lower_line("HERALD_LEAKAGE_EVENT[t] 0").unwrap_err().contains("no tag"));
        let text = "LEAKAGE(0.01) 0 3\nRELAX(0.5) 2\nHERALD_LEAKAGE_EVENT(0.02) 0 !4\nM 0\n";
        let lowered: String = text.lines().map(|l| lower_line(l).unwrap().unwrap_or(l.to_string()) + "\n").collect();
        assert_eq!(to_deltakit(&lowered), text);
    }

    #[test]
    fn without_leakage_renumbers_and_drops_heralds() {
        let c = ir("R 0 1\nI_ERROR[LEAKAGE(p=0.1)] 0\nM 0\nMPAD[HERALD_LEAKAGE_EVENT(qubits=1)] 0\nDETECTOR rec[-1]\nM 1\nDETECTOR rec[-1] rec[-3]\nOBSERVABLE_INCLUDE(0) rec[-1] rec[-2]\nCX rec[-2] 0 rec[-1] 1");
        let out = without_leakage(&c).unwrap();
        assert_eq!(out.to_string(), "R 0 1\nM 0\nM 1\nDETECTOR rec[-1] rec[-2]\nOBSERVABLE_INCLUDE(0) rec[-1]\nCX rec[-1] 1");
        let map = herald_map(&c).unwrap();
        assert_eq!(map, HeraldMap { herald_records: vec![1], kept_detectors: vec![1], herald_detectors: vec![(0, 1)] });
    }

    #[test]
    fn loops_are_kept_and_peeled_where_lookbacks_leave_them() {
        // Each round: a measurement, a herald, a detector against the last round's measurement.
        // The first round's reaches two measurements back before the loop, where no herald
        // sits between: rec[-4] is rec[-3] there, rec[-2] after; then one loop of the rest.
        let c = ir("M 0 0\nREPEAT 100 {\n    M 0\n    MPAD[HERALD_LEAKAGE_EVENT(qubits=0)] 0\n    DETECTOR rec[-2] rec[-4]\n}");
        let out = without_leakage(&c).unwrap();
        assert_eq!(out.to_string(), "M 0 0\nM 0\nDETECTOR rec[-1] rec[-3]\nREPEAT 99 {\n    M 0\n    DETECTOR rec[-1] rec[-2]\n}");
        // With a herald before the loop too, nothing is peeled.
        let c = ir("M 0\nMPAD[HERALD_LEAKAGE_EVENT(qubits=0)] 0\nREPEAT 100 {\n    M 0\n    MPAD[HERALD_LEAKAGE_EVENT(qubits=0)] 0\n    DETECTOR rec[-2] rec[-4]\n}");
        assert_eq!(without_leakage(&c).unwrap().to_string(), "M 0\nREPEAT 100 {\n    M 0\n    DETECTOR rec[-1] rec[-2]\n}");
        // Loops with nothing to renumber stay loops.
        let c = ir("REPEAT 5 {\n    M 0\n    DETECTOR rec[-1]\n    I_ERROR[RELAX(p=0.1)] 0\n}");
        assert_eq!(without_leakage(&c).unwrap().to_string(), "REPEAT 5 {\n    M 0\n    DETECTOR rec[-1]\n}");
        assert_eq!(herald_map(&ir("REPEAT 3 {\n    MPAD[HERALD_LEAKAGE_EVENT(qubits=0)] 0\n    M 0\n    DETECTOR rec[-2]\n}")).unwrap().herald_detectors, vec![(0, 0), (1, 2), (2, 4)]);
    }
}

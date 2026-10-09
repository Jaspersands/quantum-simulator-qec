//! Stim's circuit as Stim holds it: a list of instructions (gate, parens arguments, targets,
//! tag) and `REPEAT` blocks. Text parses and prints exactly as `stim.Circuit` does: names
//! canonical, compatible neighbours fused (`H 0` then `H 1` is `H 0 1`), arguments in C++'s
//! default format (6 significant digits) while the values stay exact.

use std::fmt;

use crate::dem_program::fmt_g;
use crate::gate_data::{self, GateInfo, FLAG_IS_NOT_FUSABLE, FLAG_TARGETS_PAIRS};

pub const TARGET_VALUE_MASK: u32 = (1 << 24) - 1;
pub const TARGET_INVERTED_BIT: u32 = 1 << 31;
pub const TARGET_PAULI_X_BIT: u32 = 1 << 30;
pub const TARGET_PAULI_Z_BIT: u32 = 1 << 29;
pub const TARGET_RECORD_BIT: u32 = 1 << 28;
pub const TARGET_COMBINER: u32 = 1 << 27;
pub const TARGET_SWEEP_BIT: u32 = 1 << 26;

/// A target as Stim encodes it in 32 bits: the value and flag bits above.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct GateTarget(pub u32);

impl GateTarget {
    pub fn qubit(q: u32, inverted: bool) -> GateTarget {
        GateTarget(q | if inverted { TARGET_INVERTED_BIT } else { 0 })
    }
    /// A Pauli target: `pauli` 1 = X, 2 = Y, 3 = Z (Stim's numbering).
    pub fn pauli(q: u32, pauli: u8, inverted: bool) -> GateTarget {
        let (x, z) = crate::clifford::pauli_string::xz_from_pauli(pauli);
        GateTarget::pauli_xz(q, x, z, inverted)
    }
    pub fn pauli_xz(q: u32, x: bool, z: bool, inverted: bool) -> GateTarget {
        let mut d = q;
        if x {
            d |= TARGET_PAULI_X_BIT;
        }
        if z {
            d |= TARGET_PAULI_Z_BIT;
        }
        if inverted {
            d |= TARGET_INVERTED_BIT;
        }
        GateTarget(d)
    }
    /// `rec[-lookback]`, lookback ≥ 1.
    pub fn rec(lookback: u32) -> GateTarget {
        GateTarget(lookback | TARGET_RECORD_BIT)
    }
    pub fn sweep(k: u32) -> GateTarget {
        GateTarget(k | TARGET_SWEEP_BIT)
    }
    pub fn combiner() -> GateTarget {
        GateTarget(TARGET_COMBINER)
    }
    pub fn value(self) -> u32 {
        self.0 & TARGET_VALUE_MASK
    }
    pub fn is_combiner(self) -> bool {
        self.0 == TARGET_COMBINER
    }
    pub fn is_inverted(self) -> bool {
        self.0 & TARGET_INVERTED_BIT != 0
    }
    pub fn is_x(self) -> bool {
        self.0 & TARGET_PAULI_X_BIT != 0 && self.0 & TARGET_PAULI_Z_BIT == 0
    }
    pub fn is_y(self) -> bool {
        self.0 & TARGET_PAULI_X_BIT != 0 && self.0 & TARGET_PAULI_Z_BIT != 0
    }
    pub fn is_z(self) -> bool {
        self.0 & TARGET_PAULI_X_BIT == 0 && self.0 & TARGET_PAULI_Z_BIT != 0
    }
    pub fn is_pauli(self) -> bool {
        self.0 & (TARGET_PAULI_X_BIT | TARGET_PAULI_Z_BIT) != 0
    }
    pub fn is_record(self) -> bool {
        self.0 & TARGET_RECORD_BIT != 0
    }
    pub fn is_sweep(self) -> bool {
        self.0 & TARGET_SWEEP_BIT != 0
    }
    pub fn is_classical_bit(self) -> bool {
        self.0 & (TARGET_RECORD_BIT | TARGET_SWEEP_BIT) != 0
    }
    /// A plain qubit (possibly inverted), not a Pauli, record, sweep bit or combiner.
    pub fn is_qubit(self) -> bool {
        self.0 & (TARGET_PAULI_X_BIT | TARGET_PAULI_Z_BIT | TARGET_RECORD_BIT | TARGET_SWEEP_BIT | TARGET_COMBINER) == 0
    }
    /// Whether the target names a qubit (plain or Pauli).
    pub fn has_qubit_value(self) -> bool {
        self.0 & (TARGET_RECORD_BIT | TARGET_SWEEP_BIT | TARGET_COMBINER) == 0
    }
    /// Stim's `pauli_type`: 'I', 'X', 'Y' or 'Z'.
    pub fn pauli_type(self) -> char {
        ['I', 'Z', 'X', 'Y'][((self.0 >> 29) & 3) as usize]
    }
    /// The Pauli as Stim numbers it (0 none, 1 X, 2 Y, 3 Z).
    pub fn pauli_index(self) -> u8 {
        crate::clifford::pauli_string::pauli_from_xz(self.0 & TARGET_PAULI_X_BIT != 0, self.0 & TARGET_PAULI_Z_BIT != 0)
    }
}

impl fmt::Display for GateTarget {
    /// Stim's succinct form: `5`, `!5`, `X5`, `!Y5`, `rec[-2]`, `sweep[3]`, `*`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_combiner() {
            return f.write_str("*");
        }
        if self.is_inverted() {
            f.write_str("!")?;
        }
        if self.is_pauli() {
            let x = self.0 & TARGET_PAULI_X_BIT != 0;
            let z = self.0 & TARGET_PAULI_Z_BIT != 0;
            write!(f, "{}", ['I', 'X', 'Z', 'Y'][x as usize + 2 * z as usize])?;
        }
        if self.is_record() {
            write!(f, "rec[-{}]", self.value())
        } else if self.is_sweep() {
            write!(f, "sweep[{}]", self.value())
        } else {
            write!(f, "{}", self.value())
        }
    }
}

/// One instruction: a gate (canonical), its parens arguments, its targets, its tag.
#[derive(Clone, Debug)]
pub struct Instruction {
    pub gate: &'static GateInfo,
    pub args: Vec<f64>,
    pub targets: Vec<GateTarget>,
    pub tag: String,
}

impl PartialEq for Instruction {
    fn eq(&self, o: &Self) -> bool {
        self.gate.name == o.gate.name && self.args == o.args && self.targets == o.targets && self.tag == o.tag
    }
}

impl Instruction {
    pub fn new(name: &str, args: Vec<f64>, targets: Vec<GateTarget>, tag: &str) -> Result<Instruction, String> {
        let gate = gate_data::info(name).ok_or_else(|| format!("Gate not found: '{name}'"))?;
        Ok(Instruction { gate, args, targets, tag: tag.to_string() })
    }

    pub fn has_flag(&self, flag: u32) -> bool {
        self.gate.flags & flag != 0
    }

    /// Whether `other` can be fused onto the end of this one (Stim's `can_fuse`).
    pub fn can_fuse(&self, other: &Instruction) -> bool {
        self.gate.name == other.gate.name && self.args == other.args && self.tag == other.tag && !self.has_flag(FLAG_IS_NOT_FUSABLE)
    }

    /// How many measurement results the instruction records.
    pub fn count_measurement_results(&self) -> u64 {
        if !self.gate.produces_measurements {
            return 0;
        }
        let n = self.targets.len() as u64;
        if self.has_flag(gate_data::FLAG_TARGETS_COMBINERS) {
            // One result per product: targets minus combiners, minus the targets they join.
            let combiners = self.targets.iter().filter(|t| t.is_combiner()).count() as u64;
            return n - 2 * combiners;
        }
        if self.has_flag(FLAG_TARGETS_PAIRS) {
            return n / 2;
        }
        n
    }

    /// Approximate equality: the same but for arguments within `atol`.
    pub fn approx_equals(&self, other: &Instruction, atol: f64) -> bool {
        self.gate.name == other.gate.name
            && self.targets == other.targets
            && self.tag == other.tag
            && self.args.len() == other.args.len()
            && self.args.iter().zip(&other.args).all(|(a, b)| (a - b).abs() <= atol)
    }

    /// The text with arguments written exactly (round-trippable), for pickling.
    pub fn exact_line(&self) -> String {
        self.line_with(|v| exact_arg(v))
    }

    fn line_with(&self, fmt_arg: impl Fn(f64) -> String) -> String {
        let mut s = String::from(self.gate.name);
        if !self.tag.is_empty() {
            s.push('[');
            s.push_str(&escape_tag(&self.tag));
            s.push(']');
        }
        if !self.args.is_empty() {
            s.push('(');
            s.push_str(&self.args.iter().map(|&v| fmt_arg(v)).collect::<Vec<_>>().join(", "));
            s.push(')');
        }
        let mut skip_space = false;
        for t in &self.targets {
            if t.is_combiner() {
                skip_space = true;
            } else if !skip_space {
                s.push(' ');
            } else {
                skip_space = false;
            }
            s.push_str(&t.to_string());
        }
        s
    }
}

impl fmt::Display for Instruction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.line_with(stim_arg))
    }
}

/// An argument as Stim's `operator<<` writes it: an integer as an integer, otherwise C++'s
/// default stream format (6 significant digits).
pub fn stim_arg(v: f64) -> String {
    if v > i64::MIN as f64 && v < i64::MAX as f64 && (v as i64) as f64 == v {
        format!("{}", v as i64)
    } else {
        fmt_g(v, 6)
    }
}

/// An argument written so it parses back to the same double.
pub fn exact_arg(v: f64) -> String {
    if v > i64::MIN as f64 && v < i64::MAX as f64 && (v as i64) as f64 == v {
        return format!("{}", v as i64);
    }
    let g = fmt_g(v, 6);
    if g.parse::<f64>() == Ok(v) {
        g
    } else {
        format!("{v}")
    }
}

/// Stim's tag escapes: newline `\n`, carriage return `\r`, backslash `\B`, `]` as `\C`.
pub fn escape_tag(tag: &str) -> String {
    let mut s = String::new();
    for c in tag.chars() {
        match c {
            '\n' => s.push_str("\\n"),
            '\r' => s.push_str("\\r"),
            '\\' => s.push_str("\\B"),
            ']' => s.push_str("\\C"),
            c => s.push(c),
        }
    }
    s
}

fn unescape_tag(tag: &str) -> Result<String, String> {
    let mut s = String::new();
    let mut it = tag.chars();
    while let Some(c) = it.next() {
        if c == '\\' {
            match it.next() {
                Some('n') => s.push('\n'),
                Some('r') => s.push('\r'),
                Some('B') => s.push('\\'),
                Some('C') => s.push(']'),
                other => return Err(format!("Unrecognized escape sequence '\\{}' in tag.", other.map(String::from).unwrap_or_default())),
            }
        } else {
            s.push(c);
        }
    }
    Ok(s)
}

/// An item of a circuit: an instruction or a repeated block.
#[derive(Clone, Debug, PartialEq)]
pub enum Item {
    Op(Instruction),
    Repeat { count: u64, body: Circuit, tag: String },
}

/// A circuit: Stim's list of instructions and blocks.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Circuit {
    pub items: Vec<Item>,
}

impl Circuit {
    pub fn new() -> Circuit {
        Circuit::default()
    }

    /// Parse Stim's text (structure only; the engine's parser validates gates and targets).
    pub fn parse(text: &str) -> Result<Circuit, String> {
        let lines: Vec<&str> = text.lines().collect();
        let mut pos = 0;
        let c = parse_block(&lines, &mut pos, false)?;
        Ok(c)
    }

    /// Append an instruction, fusing it with the last one when Stim would.
    pub fn push_op(&mut self, op: Instruction) {
        if let Some(Item::Op(last)) = self.items.last_mut() {
            if last.can_fuse(&op) {
                last.targets.extend(op.targets);
                return;
            }
        }
        self.items.push(Item::Op(op));
    }

    pub fn push_repeat(&mut self, count: u64, body: Circuit, tag: &str) {
        self.items.push(Item::Repeat { count, body, tag: tag.to_string() });
    }

    /// The same: a block never fuses (Stim's `append_repeat_block`).
    pub fn push_raw_repeat(&mut self, count: u64, body: Circuit, tag: &str) {
        self.push_repeat(count, body, tag);
    }

    /// Append another circuit's items, fusing at the seam.
    pub fn extend(&mut self, other: &Circuit) {
        for it in &other.items {
            match it {
                Item::Op(op) => self.push_op(op.clone()),
                r => self.items.push(r.clone()),
            }
        }
    }

    /// Every instruction in order, loops unrolled.
    pub fn for_each_operation(&self, f: &mut dyn FnMut(&Instruction)) {
        for it in &self.items {
            match it {
                Item::Op(op) => f(op),
                Item::Repeat { count, body, .. } => {
                    for _ in 0..*count {
                        body.for_each_operation(f);
                    }
                }
            }
        }
    }

    /// The number of qubits: one more than the largest qubit any target names.
    pub fn count_qubits(&self) -> u64 {
        let mut n = 0u64;
        for it in &self.items {
            match it {
                Item::Op(op) => {
                    for t in &op.targets {
                        if t.has_qubit_value() && !op.gate.name.eq("MPAD") {
                            n = n.max(t.value() as u64 + 1);
                        }
                    }
                }
                Item::Repeat { body, .. } => n = n.max(body.count_qubits()),
            }
        }
        n
    }

    pub fn count_measurements(&self) -> u64 {
        self.items
            .iter()
            .map(|it| match it {
                Item::Op(op) => op.count_measurement_results(),
                Item::Repeat { count, body, .. } => count.saturating_mul(body.count_measurements()),
            })
            .fold(0u64, |a, b| a.saturating_add(b))
    }

    pub fn count_ticks(&self) -> u64 {
        self.items
            .iter()
            .map(|it| match it {
                Item::Op(op) => (op.gate.name == "TICK") as u64,
                Item::Repeat { count, body, .. } => count.saturating_mul(body.count_ticks()),
            })
            .fold(0u64, |a, b| a.saturating_add(b))
    }

    /// The text with every argument exact: parses back to an equal circuit.
    pub fn exact_text(&self) -> String {
        let mut out = String::new();
        write_items(&self.items, "", &mut out, &|op| op.exact_line());
        out
    }

    pub fn approx_equals(&self, other: &Circuit, atol: f64) -> bool {
        self.items.len() == other.items.len()
            && self.items.iter().zip(&other.items).all(|(a, b)| match (a, b) {
                (Item::Op(x), Item::Op(y)) => x.approx_equals(y, atol),
                (Item::Repeat { count: c1, body: b1, tag: t1 }, Item::Repeat { count: c2, body: b2, tag: t2 }) => c1 == c2 && t1 == t2 && b1.approx_equals(b2, atol),
                _ => false,
            })
    }
}

fn write_items(items: &[Item], indent: &str, out: &mut String, line: &dyn Fn(&Instruction) -> String) {
    for (k, it) in items.iter().enumerate() {
        if k > 0 {
            out.push('\n');
        }
        match it {
            Item::Op(op) => {
                out.push_str(indent);
                out.push_str(&line(op));
            }
            Item::Repeat { count, body, tag } => {
                out.push_str(indent);
                out.push_str("REPEAT");
                if !tag.is_empty() {
                    out.push('[');
                    out.push_str(&escape_tag(tag));
                    out.push(']');
                }
                out.push_str(&format!(" {count} {{\n"));
                write_items(&body.items, &format!("{indent}    "), out, line);
                out.push('\n');
                out.push_str(indent);
                out.push('}');
            }
        }
    }
}

impl fmt::Display for Circuit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut out = String::new();
        write_items(&self.items, "", &mut out, &|op| op.to_string());
        f.write_str(&out)
    }
}

fn parse_block(lines: &[&str], pos: &mut usize, nested: bool) -> Result<Circuit, String> {
    let mut c = Circuit::new();
    while *pos < lines.len() {
        let raw = lines[*pos];
        *pos += 1;
        let line = strip_comment(raw).trim();
        if line.is_empty() {
            continue;
        }
        if line == "}" {
            if !nested {
                return Err("Uninitiated block end.".into());
            }
            return Ok(c);
        }
        let (name, tag, args, rest) = split_head(line)?;
        if name.eq_ignore_ascii_case("REPEAT") {
            let rest = rest.trim();
            let count_text = rest.strip_suffix('{').ok_or("Missing '{' at start of REPEAT block.")?.trim();
            let count: u64 = count_text.parse().map_err(|_| format!("Not a valid repetition count: '{count_text}'."))?;
            if count == 0 {
                return Err("Repeating 0 times is not supported.".into());
            }
            let body = parse_block(lines, pos, true)?;
            c.push_repeat(count, body, &tag);
            continue;
        }
        let gate = gate_data::info(&name).ok_or_else(|| format!("Gate not found: '{name}'"))?;
        let targets = parse_targets(rest)?;
        c.push_op(Instruction { gate, args, targets, tag });
    }
    if nested {
        return Err("Unterminated block. Got a '{' without an eventual '}'.".into());
    }
    Ok(c)
}

/// The line without its `#` comment; a `#` inside a `[tag]` is part of the tag.
fn strip_comment(raw: &str) -> &str {
    let mut in_tag = false;
    for (k, c) in raw.char_indices() {
        match c {
            '[' if !in_tag => in_tag = true,
            ']' if in_tag => in_tag = false,
            '#' if !in_tag => return &raw[..k],
            _ => {}
        }
    }
    raw
}

/// (name, tag, args, rest-of-line targets).
fn split_head(line: &str) -> Result<(String, String, Vec<f64>, &str), String> {
    let bytes = line.as_bytes();
    let mut i = 0;
    while i < bytes.len() && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_') {
        i += 1;
    }
    let name = line[..i].to_string();
    if name.is_empty() {
        return Err(format!("Expected a gate name in '{line}'."));
    }
    let mut tag = String::new();
    if i < bytes.len() && bytes[i] == b'[' {
        let end = line[i..].find(']').ok_or("Tag didn't end with ']'.")? + i;
        tag = unescape_tag(&line[i + 1..end])?;
        i = end + 1;
    }
    let mut args = Vec::new();
    if i < bytes.len() && bytes[i] == b'(' {
        let end = line[i..].find(')').ok_or_else(|| format!("Parens arguments for '{name}' didn't end with a ')'."))? + i;
        let inner = line[i + 1..end].trim();
        if !inner.is_empty() {
            for a in inner.split(',') {
                let a = a.trim();
                let v: f64 = a.parse().map_err(|_| format!("Parens arguments for '{name}' didn't end with a ')'."))?;
                if !v.is_finite() {
                    return Err(format!("Parens arguments for '{name}' didn't end with a ')'."));
                }
                args.push(v);
            }
        }
        i = end + 1;
    }
    let rest = &line[i..];
    if !rest.is_empty() && !rest.starts_with(|c: char| c.is_whitespace()) {
        return Err("Targets must be separated by spacing.".into());
    }
    Ok((name, tag, args, rest))
}

fn parse_targets(rest: &str) -> Result<Vec<GateTarget>, String> {
    let mut out = Vec::new();
    for word in rest.split_whitespace() {
        let mut first = true;
        for piece in word.split('*') {
            if !first {
                out.push(GateTarget::combiner());
            }
            first = false;
            if piece.is_empty() {
                continue;
            }
            out.push(parse_target(piece)?);
        }
    }
    Ok(out)
}

/// One target's text: `5`, `!5`, `X5`, `!y5`, `rec[-2]`, `sweep[3]`.
pub fn parse_target(t: &str) -> Result<GateTarget, String> {
    let bad = || format!("Unrecognized target '{t}'.");
    let (inv, s) = match t.strip_prefix('!') {
        Some(r) => (true, r),
        None => (false, t),
    };
    if let Some(r) = s.strip_prefix("rec[-").and_then(|r| r.strip_suffix(']')) {
        let k: u32 = r.parse().map_err(|_| bad())?;
        if k == 0 || k > TARGET_VALUE_MASK || inv {
            return Err(bad());
        }
        return Ok(GateTarget::rec(k));
    }
    if let Some(r) = s.strip_prefix("sweep[").and_then(|r| r.strip_suffix(']')) {
        let k: u32 = r.parse().map_err(|_| bad())?;
        if k > TARGET_VALUE_MASK || inv {
            return Err(bad());
        }
        return Ok(GateTarget::sweep(k));
    }
    let (pauli, digits) = match s.chars().next() {
        Some('X') | Some('x') => (1, &s[1..]),
        Some('Y') | Some('y') => (2, &s[1..]),
        Some('Z') | Some('z') => (3, &s[1..]),
        _ => (0, s),
    };
    if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return Err(bad());
    }
    let q: u64 = digits.parse().map_err(|_| bad())?;
    if q > TARGET_VALUE_MASK as u64 {
        return Err(format!("Qubit target {q} is too large."));
    }
    Ok(if pauli == 0 { GateTarget::qubit(q as u32, inv) } else { GateTarget::pauli(q as u32, pauli, inv) })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rt(t: &str) -> String {
        Circuit::parse(t).unwrap().to_string()
    }

    #[test]
    fn prints_as_stim_does() {
        assert_eq!(rt("H 0\nH 1"), "H 0 1");
        assert_eq!(rt("X_ERROR(0.1) 0\nX_ERROR(0.1) 1\nX_ERROR(0.2) 2"), "X_ERROR(0.1) 0 1\nX_ERROR(0.2) 2");
        assert_eq!(rt("H[a] 0\nH[a] 1\nH 2"), "H[a] 0 1\nH 2");
        assert_eq!(rt("DETECTOR rec[-1]\nDETECTOR rec[-2]"), "DETECTOR rec[-1]\nDETECTOR rec[-2]");
        assert_eq!(rt("MPP X0*Z1\nMPP Y2"), "MPP X0*Z1 Y2");
        assert_eq!(rt("MPP X0 * Y1"), "MPP X0*Y1");
        assert_eq!(rt("mpp x0*y1"), "MPP X0*Y1");
        assert_eq!(rt("DEPOLARIZE1(0.30000000000000004) 0"), "DEPOLARIZE1(0.3) 0");
        assert_eq!(rt("X_ERROR(0.123456789) 0"), "X_ERROR(0.123457) 0");
        assert_eq!(rt("QUBIT_COORDS(0.5,1e10) 0"), "QUBIT_COORDS(0.5, 10000000000) 0");
        assert_eq!(rt("CNOT 0 1"), "CX 0 1");
        assert_eq!(rt("REPEAT 2 {\n}"), "REPEAT 2 {\n\n}");
        assert_eq!(rt("REPEAT[t] 2 {\nH 0\n}"), "REPEAT[t] 2 {\n    H 0\n}");
        assert_eq!(rt("TICK\nREPEAT 2 {\nTICK\n}\nTICK"), "TICK\nREPEAT 2 {\n    TICK\n}\nTICK");
        assert_eq!(rt("H[a\\Cb] 0"), "H[a\\Cb] 0");
        assert_eq!(rt("H[a#b] 0 # comment"), "H[a#b] 0");
        assert_eq!(rt("X_ERROR(1e-5) 0"), "X_ERROR(1e-05) 0");
        assert!(Circuit::parse("X_ERROR(0.1)0").is_err());
        assert!(Circuit::parse("REPEAT 0 {\nH 0\n}").is_err());
        let c = Circuit::parse("X_ERROR(0.123456789) 0").unwrap();
        assert_eq!(Circuit::parse(&c.exact_text()).unwrap(), c);
    }

    #[test]
    fn counts() {
        let c = Circuit::parse("M 0 1\nMPP X0*Y1 Z2\nMXX 0 1 2 3\nREPEAT 3 {\nMR 4\nTICK\n}").unwrap();
        assert_eq!(c.count_measurements(), 2 + 2 + 2 + 3);
        assert_eq!(c.count_qubits(), 5);
        assert_eq!(c.count_ticks(), 3);
    }
}

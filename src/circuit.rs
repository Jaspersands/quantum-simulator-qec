//! A circuit, in the part of Stim's text format this engine speaks.
//!
//! WHY THIS EXISTS
//! ---------------
//! Every circuit the engine ran used to be one it wrote itself: each code emits
//! a `round_program`, and `circuit_model` derives its decoding graph from that.
//! Circuits from anywhere else (Stim's generated surface codes, published
//! experiments, lattice surgery) had no way in. This is the way in and the way
//! out: the text Stim reads and writes, so one circuit can be handed to both
//! tools and they can be asked about exactly the same thing.
//!
//! Anything outside the supported subset is an error naming the instruction
//! and its line. A circuit that half-parses and then runs answers a question
//! nobody asked.

use std::fmt::Write as _;
use std::sync::Arc;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Basis {
    X,
    Z,
}

/// A single-qubit Pauli as a bitmask: 1 = X, 2 = Z, 3 = Y, matching
/// `circuit_model::Pauli`.
pub type Pauli = u8;

#[derive(Clone, Debug, PartialEq)]
pub enum Instr {
    Reset { basis: Basis, qubits: Vec<u32> },
    H(Vec<u32>),
    Cx(Vec<(u32, u32)>),
    Cz(Vec<(u32, u32)>),
    /// `I`, `X`, `Y`, `Z`: Pauli gates, which only flip signs. 0 = I.
    Pauli { pauli: Pauli, qubits: Vec<u32> },
    /// `CX sweep[k] q`: an X on `q` when sweep bit `k` of the shot is set.
    SweepX(Vec<(u32, u32)>),
    /// `M`, `MX`, `MR`, `MRX`. `flip` is the classical flip probability of `M(p)`.
    Measure { basis: Basis, reset: bool, flip: f64, qubits: Vec<u32> },
    /// `X_ERROR`, `Y_ERROR`, `Z_ERROR`.
    PauliError { pauli: Pauli, p: f64, qubits: Vec<u32> },
    Depolarize1 { p: f64, qubits: Vec<u32> },
    Depolarize2 { p: f64, pairs: Vec<(u32, u32)> },
    PauliChannel1 { px: f64, py: f64, pz: f64, qubits: Vec<u32> },
    /// `recs` are lookbacks: 1 is `rec[-1]`.
    Detector { coords: Vec<f64>, recs: Vec<u32> },
    /// `recs` as for a detector; `paulis` are Pauli targets (`X3`, `!Z0`): the observable
    /// takes in that Pauli's value at this point of the circuit, which errors before it that
    /// anticommute with it flip. `true` marks an inverted target.
    Observable { index: u32, recs: Vec<u32>, paulis: Vec<(u32, Pauli, bool)> },
    QubitCoords { coords: Vec<f64>, qubits: Vec<u32> },
    ShiftCoords(Vec<f64>),
    Tick,
    /// `tag` is Stim's instruction tag (`REPEAT[tag] 5 {`), empty for none.
    Repeat { count: u64, body: Vec<Instr>, tag: String },
    /// `S`: the phase gate. With H and CX it generates every Clifford gate.
    S(Vec<u32>),
    /// A gate of Stim's that the engine runs as its exact decomposition into its own
    /// instructions (`gates.rs` for the unitary ones; the Y basis, Pauli products and inverted
    /// targets built here): printed as `line`, run as `body`, which records exactly as many
    /// measurements as the gate does. A tagged instruction (`H[tag] 0`, Stim's tags, which
    /// change nothing it does) is a gate too: `tag` holds the tag, carried to the error model,
    /// and `body` the instruction untagged.
    Gate { line: String, body: Vec<Instr>, tag: String },
    /// `E` (`CORRELATED_ERROR`), and `ELSE_CORRELATED_ERROR` when `chained`: with probability
    /// `p`, the Pauli product `paulis`; a chained one only where no earlier error of its chain
    /// fired.
    Correlated { p: f64, paulis: Vec<(u32, Pauli)>, chained: bool },
    /// `PAULI_CHANNEL_2`: on each pair, one of the 15 non-identity two-qubit Paulis, with
    /// probabilities in Stim's order (IX, IY, IZ, XI, XX, ..., ZZ; the first letter is the
    /// first qubit's).
    PauliChannel2 { probs: Vec<f64>, pairs: Vec<(u32, u32)> },
    /// `MPAD`: measurement records of fixed values, each flipped with probability `flip`.
    Pad { flip: f64, values: Vec<bool> },
    /// A classically controlled Pauli (`CX rec[-1] 3`, `CZ sweep[0] 2`, `XCZ 1 rec[-2]`):
    /// `pauli` on `qubit` where the measurement record or sweep bit is 1.
    Feedback { pauli: Pauli, control: Control, qubit: u32 },
    /// `HERALDED_ERASE` and `HERALDED_PAULI_CHANNEL_1`: per qubit, a herald record that is 1
    /// when the error fires, and then I, X, Y or Z with probabilities `probs` (which sum to
    /// the herald's). `args` are the instruction's own, for printing.
    Heralded { erase: bool, args: Vec<f64>, probs: [f64; 4], qubits: Vec<u32> },
    /// Operations beyond Clifford gates and Pauli noise, from a tagged identity
    /// (`I_ERROR[R_Z(theta=0.1)] 0`, see `nonpauli`). Only inside the `Gate` that tag writes;
    /// Clifford engines run it as the identity it is in Stim.
    NonPauli(Vec<crate::nonpauli::NonPauli>),
}

/// What classically controls a `Feedback`: a measurement record (lookback, 1 is the latest)
/// or a sweep bit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Control {
    Rec(u32),
    Sweep(u32),
}

impl std::fmt::Display for Control {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Control::Rec(k) => write!(f, "rec[-{k}]"),
            Control::Sweep(k) => write!(f, "sweep[{k}]"),
        }
    }
}

impl Instr {
    /// Every qubit the instruction names.
    pub fn qubits(&self) -> Vec<u32> {
        match self {
            Instr::Reset { qubits, .. }
            | Instr::H(qubits)
            | Instr::Measure { qubits, .. }
            | Instr::PauliError { qubits, .. }
            | Instr::Depolarize1 { qubits, .. }
            | Instr::PauliChannel1 { qubits, .. }
            | Instr::QubitCoords { qubits, .. }
            | Instr::Pauli { qubits, .. } => qubits.clone(),
            Instr::SweepX(pairs) => pairs.iter().map(|&(_, q)| q).collect(),
            Instr::Cx(pairs) | Instr::Cz(pairs) | Instr::Depolarize2 { pairs, .. } => {
                pairs.iter().flat_map(|&(a, b)| [a, b]).collect()
            }
            Instr::Repeat { body, .. } | Instr::Gate { body, .. } => body.iter().flat_map(|i| i.qubits()).collect(),
            Instr::S(qubits) | Instr::Heralded { qubits, .. } => qubits.clone(),
            Instr::Feedback { qubit, .. } => vec![*qubit],
            Instr::Correlated { paulis, .. } => paulis.iter().map(|&(q, _)| q).collect(),
            Instr::Observable { paulis, .. } => paulis.iter().map(|&(q, ..)| q).collect(),
            Instr::PauliChannel2 { pairs, .. } => pairs.iter().flat_map(|&(a, b)| [a, b]).collect(),
            Instr::NonPauli(ops) => ops.iter().flat_map(|op| op.qubits()).collect(),
            _ => Vec::new(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Default)]
pub struct Circuit {
    pub instrs: Vec<Instr>,
}

/// A flattened circuit with every measurement record resolved to an absolute
/// index. It is what the error-model builder and the sampler both walk.
#[derive(Debug)]
pub struct Resolved {
    pub instrs: Vec<Instr>,
    pub num_qubits: usize,
    pub num_measurements: usize,
    /// One more than the highest sweep bit any `CX sweep[k]` reads.
    pub num_sweep_bits: usize,
    /// Absolute measurement indices each detector reads.
    pub detectors: Vec<Vec<usize>>,
    pub detector_coords: Vec<Vec<f64>>,
    /// Absolute measurement indices each observable accumulates.
    pub observables: Vec<Vec<usize>>,
    /// The tagged instructions of `instrs`, by index, in order.
    pub tags: Vec<(usize, Arc<str>)>,
    /// Each tagged detector's tag, by detector, in order.
    pub detector_tags: Vec<(usize, Arc<str>)>,
    /// Each observable's tag: the first `OBSERVABLE_INCLUDE` of it with one.
    pub observable_tags: Vec<String>,
}

impl Resolved {
    /// Instruction `idx`'s tag, if it has one.
    pub fn tag(&self, idx: usize) -> Option<&Arc<str>> {
        self.tags.binary_search_by_key(&idx, |t| t.0).ok().map(|i| &self.tags[i].1)
    }
}

impl Circuit {
    pub fn parse(text: &str) -> Result<Circuit, String> {
        let lines: Vec<&str> = text.lines().collect();
        let mut pos = 0usize;
        let instrs = parse_block(&lines, &mut pos, false)?;
        Ok(Circuit { instrs })
    }

    /// One instruction, `name[tag](args) targets`, read as its line would be. The name is one
    /// word, the tag one line without `]`, and each target one token (`3`, `!3`, `rec[-1]`,
    /// `sweep[0]`, `X3`, or `*` joining the Pauli targets either side), so that no part can
    /// smuggle in another instruction.
    pub fn instruction(name: &str, tag: &str, args: &[f64], targets: &[String]) -> Result<Circuit, String> {
        if name.is_empty() || !name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
            return Err(format!("'{name}' is not an instruction name"));
        }
        if name.eq_ignore_ascii_case("REPEAT") {
            return Err("a REPEAT block is appended as a circuit repeated, not as an instruction".into());
        }
        if tag.contains([']', '\n', '\r']) {
            return Err(format!("{name}: a tag holds no ']' and no line break, got '{tag}'"));
        }
        if let Some(a) = args.iter().find(|a| !a.is_finite()) {
            return Err(format!("{name}: argument {a} is not finite"));
        }
        let mut line = name.to_string();
        if !tag.is_empty() {
            let _ = write!(line, "[{tag}]");
        }
        if !args.is_empty() {
            let _ = write!(line, "({})", fmt_args(args));
        }
        let mut glue = true;
        for t in targets {
            if t.is_empty() || t.contains(|c: char| c.is_whitespace() || "#{}".contains(c)) {
                return Err(format!("{name}: '{t}' is not a target"));
            }
            if t == "*" {
                line.push('*');
                glue = true;
            } else {
                line.push_str(if glue && line.ends_with('*') { "" } else { " " });
                line.push_str(t);
                glue = false;
            }
        }
        if line.ends_with('*') || line.contains(" *") {
            return Err(format!("{name}: a '*' joins two targets"));
        }
        let c = Circuit::parse(&line)?;
        if c.instrs.len() != 1 {
            return Err(format!("'{line}' is not one instruction"));
        }
        Ok(c)
    }

    /// `REPEAT count { self }`, as Stim's `circuit * count`: nothing for 0, the circuit itself
    /// for 1, and a circuit that is one untagged loop has its count multiplied.
    pub fn repeated(&self, count: u64) -> Circuit {
        match (count, self.instrs.as_slice()) {
            (0, _) => Circuit::default(),
            (1, _) => self.clone(),
            (_, [Instr::Repeat { count: inner, body, tag }]) if tag.is_empty() && inner.checked_mul(count).is_some() => {
                Circuit { instrs: vec![Instr::Repeat { count: inner * count, body: body.clone(), tag: String::new() }] }
            }
            _ => Circuit { instrs: vec![Instr::Repeat { count, body: self.instrs.clone(), tag: String::new() }] },
        }
    }

    pub fn to_stim(&self) -> String {
        let mut s = String::new();
        emit(&self.instrs, "", &mut s);
        s
    }

    /// REPEAT blocks expanded, SHIFT_COORDS folded into the coordinates they move. Gates stay
    /// whole, as in Stim's `flattened`.
    pub fn flattened(&self) -> Circuit {
        let mut out = Vec::new();
        let mut shift = Vec::new();
        flatten_into(&self.instrs, &mut out, &mut shift);
        Circuit { instrs: out }
    }

    pub fn resolve(&self) -> Result<Resolved, String> {
        let size = unrolled_size(&self.instrs);
        if size > MAX_UNROLLED {
            return Err(format!(
                "the circuit unrolls to {size} instructions and targets, more than the {MAX_UNROLLED} \
                 this engine holds: it expands every REPEAT block"
            ));
        }
        let mut flat = Vec::new();
        let mut tags = Vec::new();
        expand_gates(&self.flattened().instrs, None, &mut flat, &mut tags);
        let mut detector_tags = Vec::new();
        let mut observable_tags: Vec<String> = Vec::new();
        let mut next_tag = 0usize;
        let mut num_qubits = 0usize;
        let mut m = 0usize;
        let mut sweeps = 0usize;
        let mut detectors = Vec::new();
        let mut detector_coords = Vec::new();
        let mut observables: Vec<Vec<usize>> = Vec::new();
        let absolute = |k: u32, m: usize| -> Result<usize, String> {
            if k == 0 || k as usize > m {
                Err(format!("rec[-{k}] reaches before the first measurement"))
            } else {
                Ok(m - k as usize)
            }
        };
        for (idx, ins) in flat.iter().enumerate() {
            for q in ins.qubits() {
                num_qubits = num_qubits.max(q as usize + 1);
            }
            let tag = match tags.get(next_tag) {
                Some((i, t)) if *i == idx => {
                    next_tag += 1;
                    Some(t)
                }
                _ => None,
            };
            match ins {
                Instr::Measure { qubits, .. } => m += qubits.len(),
                Instr::Pad { values, .. } => m += values.len(),
                Instr::Heralded { qubits, .. } => m += qubits.len(),
                Instr::Feedback { control: Control::Rec(k), .. } => {
                    absolute(*k, m)?;
                }
                Instr::Feedback { control: Control::Sweep(k), .. } => sweeps = sweeps.max(*k as usize + 1),
                Instr::SweepX(pairs) => {
                    for &(k, _) in pairs {
                        sweeps = sweeps.max(k as usize + 1);
                    }
                }
                Instr::Detector { coords, recs } => {
                    let abs = recs.iter().map(|&k| absolute(k, m)).collect::<Result<Vec<_>, _>>()?;
                    if let Some(t) = tag {
                        detector_tags.push((detectors.len(), Arc::clone(t)));
                    }
                    detectors.push(abs);
                    detector_coords.push(coords.clone());
                }
                Instr::Observable { index, recs, .. } => {
                    let i = *index as usize;
                    if observables.len() <= i {
                        observables.resize(i + 1, Vec::new());
                        observable_tags.resize(i + 1, String::new());
                    }
                    if let Some(t) = tag.filter(|_| observable_tags[i].is_empty()) {
                        observable_tags[i] = t.to_string();
                    }
                    for &k in recs {
                        observables[i].push(absolute(k, m)?);
                    }
                }
                _ => {}
            }
        }
        Ok(Resolved {
            instrs: flat,
            num_qubits,
            num_measurements: m,
            num_sweep_bits: sweeps,
            detectors,
            detector_coords,
            observables,
            tags,
            detector_tags,
            observable_tags,
        })
    }
}

fn parse_block(lines: &[&str], pos: &mut usize, nested: bool) -> Result<Vec<Instr>, String> {
    let mut out = Vec::new();
    while *pos < lines.len() {
        let lineno = *pos + 1;
        let line = strip_comment(lines[*pos]).trim();
        *pos += 1;
        if line.is_empty() {
            continue;
        }
        if line == "}" {
            if nested {
                return Ok(out);
            }
            return Err(format!("line {lineno}: unmatched '}}'"));
        }
        if let Some(rest) = line.strip_suffix('{') {
            let (name, tag, args, parts) = split_instruction(rest).map_err(|e| format!("line {lineno}: {e}"))?;
            if name != "REPEAT" {
                return Err(format!("line {lineno}: only REPEAT opens a block, got '{name}'"));
            }
            let count: u64 = match (args.is_empty(), parts.as_slice()) {
                (true, [n]) => uint(n),
                _ => None,
            }
            .ok_or_else(|| format!("line {lineno}: REPEAT needs a count"))?;
            if count == 0 {
                return Err(format!("line {lineno}: Repeating 0 times is not supported."));
            }
            let body = parse_block(lines, pos, true)?;
            out.push(Instr::Repeat { count, body, tag: tag.to_string() });
            continue;
        }
        out.push(parse_line(line).map_err(|e| format!("line {lineno}: {e}"))?);
    }
    if nested {
        return Err("unterminated REPEAT block".into());
    }
    Ok(out)
}

/// A line without its comment: from the first `#` outside brackets (a tag may hold one).
pub(crate) fn strip_comment(line: &str) -> &str {
    let mut depth = 0usize;
    for (i, c) in line.char_indices() {
        match c {
            '[' => depth += 1,
            ']' => depth = depth.saturating_sub(1),
            '#' if depth == 0 => return &line[..i],
            _ => {}
        }
    }
    line
}

/// Where a line's instruction name ends.
fn name_end(line: &str) -> usize {
    line.find(|c: char| !(c.is_ascii_alphanumeric() || c == '_')).unwrap_or(line.len())
}

/// An instruction line's name, tag, arguments and targets.
pub(crate) type Split<'a> = (String, &'a str, Vec<f64>, Vec<&'a str>);

/// Split `NAME[tag](a, b) t1 t2` into its name (upper-cased), tag (empty for none), arguments
/// and targets. The tag is kept as written, Stim's escapes (`\B`, `\C`, `\n`, `\r`) included.
/// Shared with the `.dem` reader, whose lines have the same shape.
pub(crate) fn split_instruction(line: &str) -> Result<Split<'_>, String> {
    let name_end = name_end(line);
    if name_end == 0 {
        return Err(format!("expected an instruction name in '{line}'"));
    }
    let name = line[..name_end].to_ascii_uppercase();
    let mut rest = &line[name_end..];
    let mut tag = "";
    if let Some(inner) = rest.strip_prefix('[') {
        let close = inner.find(']').ok_or_else(|| format!("{name}: unclosed '[' in its tag"))?;
        if inner[..close].contains(['\n', '\r']) {
            return Err(format!("{name}: a tag cannot span lines"));
        }
        tag = &inner[..close];
        rest = &inner[close + 1..];
        if rest.starts_with(|c: char| !(c.is_whitespace() || c == '(')) {
            return Err(format!("{name}[{tag}]: the tag must be followed by arguments or a space"));
        }
    }
    let mut rest = rest.trim_start();
    let mut args = Vec::new();
    if let Some(inner) = rest.strip_prefix('(') {
        let close = inner.find(')').ok_or_else(|| format!("{name}: unclosed '('"))?;
        for a in inner[..close].split(',') {
            let a = a.trim();
            if a.is_empty() {
                continue;
            }
            args.push(a.parse::<f64>().ok().filter(|v| v.is_finite()).ok_or_else(|| format!("{name}: bad argument '{a}'"))?);
        }
        rest = inner[close + 1..].trim_start();
    }
    Ok((name, tag, args, rest.split_whitespace().collect()))
}

/// The largest qubit index Stim accepts (its targets keep 24 bits for it).
const MAX_QUBIT: u32 = (1 << 24) - 1;

/// An unsigned integer as Stim reads one: decimal digits only (no sign, no spaces).
pub(crate) fn uint<T: std::str::FromStr>(s: &str) -> Option<T> {
    if s.is_empty() || !s.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    s.parse().ok()
}

fn qubit(t: &str, name: &str) -> Result<u32, String> {
    match uint::<u32>(t).ok_or(()) {
        Ok(q) if q <= MAX_QUBIT => Ok(q),
        Ok(q) => Err(format!("{name}: qubit {q} is beyond the largest index, {MAX_QUBIT}")),
        Err(_) => Err(format!("{name}: bad qubit target '{t}'")),
    }
}

fn qubit_targets(tokens: &[&str], name: &str) -> Result<Vec<u32>, String> {
    tokens.iter().map(|t| qubit(t, name)).collect()
}

fn pair_targets(tokens: &[&str], name: &str) -> Result<Vec<(u32, u32)>, String> {
    let q = qubit_targets(tokens, name)?;
    if q.len() % 2 != 0 {
        return Err(format!("{name}: targets must come in pairs, got {}", q.len()));
    }
    q.chunks(2)
        .map(|c| {
            if c[0] == c[1] {
                Err(format!("{name}: a pair cannot act twice on qubit {}", c[0]))
            } else {
                Ok((c[0], c[1]))
            }
        })
        .collect()
}

/// `sweep[k] q` pairs: a sweep bit controlling an X on a qubit.
fn sweep_pairs(tokens: &[&str], name: &str) -> Result<Vec<(u32, u32)>, String> {
    if !tokens.len().is_multiple_of(2) {
        return Err(format!("{name}: targets must come in pairs, got {}", tokens.len()));
    }
    tokens
        .chunks(2)
        .map(|c| {
            let bit = c[0]
                .strip_prefix("sweep[")
                .and_then(|s| s.strip_suffix(']'))
                .and_then(uint::<u32>)
                .filter(|&k| k <= MAX_QUBIT)
                .ok_or_else(|| format!("{name}: a sweep-controlled pair needs 'sweep[k] q', got '{} {}'", c[0], c[1]))?;
            Ok((bit, qubit(c[1], name)?))
        })
        .collect()
}

fn rec_targets(tokens: &[&str], name: &str) -> Result<Vec<u32>, String> {
    tokens
        .iter()
        .map(|t| {
            t.strip_prefix("rec[-")
                .and_then(|s| s.strip_suffix(']'))
                .and_then(uint::<u32>)
                .filter(|&k| k >= 1)
                .ok_or_else(|| format!("{name}: bad record target '{t}'"))
        })
        .collect()
}

fn parse_line(line: &str) -> Result<Instr, String> {
    let (name, tag, args, t) = split_instruction(line)?;
    let ins = parse_instruction(&name, &args, &t)?;
    let ins = tagged(ins, tag);
    // A tag that asks for a non-Pauli operation on an identity (`I_ERROR[R_Z(theta=0.1)] 0`)
    // runs it in the engines that honour it, and nothing in the rest.
    if let (Instr::Gate { line, tag, .. }, "I" | "II" | "I_ERROR" | "II_ERROR") = (&ins, name.as_str()) {
        if let Some(ops) = crate::nonpauli::parse(&name, tag, &qubit_targets(&t, &name)?)? {
            return Ok(Instr::Gate { line: line.clone(), tag: tag.clone(), body: vec![Instr::NonPauli(ops)] });
        }
    }
    Ok(ins)
}

/// `inner` with Stim's tag `tag` (none if empty): printed with `[tag]` after its name, run as
/// itself.
pub(crate) fn tagged(inner: Instr, tag: &str) -> Instr {
    if tag.is_empty() {
        return inner;
    }
    let splice = |line: &str| {
        let end = name_end(line);
        format!("{}[{tag}]{}", &line[..end], &line[end..])
    };
    match inner {
        Instr::Gate { line, body, .. } => Instr::Gate { line: splice(&line), body, tag: tag.to_string() },
        Instr::Repeat { count, body, .. } => Instr::Repeat { count, body, tag: tag.to_string() },
        other => {
            let mut line = String::new();
            emit(std::slice::from_ref(&other), "", &mut line);
            Instr::Gate { line: splice(line.trim_end()), body: vec![other], tag: tag.to_string() }
        }
    }
}

fn parse_instruction(name: &str, args: &[f64], t: &[&str]) -> Result<Instr, String> {
    let name = name.to_string();
    let args = args.to_vec();
    let t = t.to_vec();
    let none = || -> Result<(), String> {
        if args.is_empty() {
            Ok(())
        } else {
            Err(format!("{name} takes no arguments"))
        }
    };
    let exactly = |n: usize| -> Result<(), String> {
        if args.len() == n {
            Ok(())
        } else {
            Err(format!("{name} takes {n} argument(s), got {}", args.len()))
        }
    };
    let prob = |i: usize| -> Result<f64, String> {
        let p = args[i];
        if (0.0..=1.0).contains(&p) {
            Ok(p)
        } else {
            Err(format!("{name}: probability {p} is outside [0, 1]"))
        }
    };
    Ok(match name.as_str() {
        "R" | "RZ" => {
            none()?;
            Instr::Reset { basis: Basis::Z, qubits: qubit_targets(&t, &name)? }
        }
        "RX" => {
            none()?;
            Instr::Reset { basis: Basis::X, qubits: qubit_targets(&t, &name)? }
        }
        "H" => {
            none()?;
            Instr::H(qubit_targets(&t, &name)?)
        }
        // Every pair sweep-controlled, or none classical: the engine's own CX instructions.
        "CX" | "CNOT" | "ZCX"
            if t.chunks(2).all(|p| p[0].starts_with("sweep[")) || !t.iter().any(|x| x.starts_with("rec[") || x.starts_with("sweep[")) =>
        {
            none()?;
            if t.iter().any(|x| x.starts_with("sweep[")) {
                Instr::SweepX(sweep_pairs(&t, &name)?)
            } else {
                Instr::Cx(pair_targets(&t, &name)?)
            }
        }
        "CX" | "CNOT" | "ZCX" | "CY" | "ZCY" | "CZ" | "ZCZ" | "XCZ" | "YCZ"
            if t.iter().any(|x| x.starts_with("rec[") || x.starts_with("sweep[")) =>
        {
            none()?;
            Instr::Gate { line: gate_line(&name, &args, &t), tag: String::new(), body: controlled_pairs(&name, &t)? }
        }
        "I" | "X" | "Y" | "Z" => {
            none()?;
            let pauli = match name.as_str() {
                "X" => 1,
                "Z" => 2,
                "Y" => 3,
                _ => 0,
            };
            Instr::Pauli { pauli, qubits: qubit_targets(&t, &name)? }
        }
        "CZ" | "ZCZ" => {
            none()?;
            Instr::Cz(pair_targets(&t, &name)?)
        }
        "M" | "MZ" | "MX" | "MR" | "MRZ" | "MRX" => {
            if args.len() > 1 {
                return Err(format!("{name} takes at most one argument"));
            }
            let flip = explicit_zero(if args.is_empty() { 0.0 } else { prob(0)? }, &args);
            let basis = if name.contains('X') { Basis::X } else { Basis::Z };
            let reset = name.starts_with("MR");
            if !t.iter().any(|x| x.starts_with('!')) {
                Instr::Measure { basis, reset, flip, qubits: qubit_targets(&t, &name)? }
            } else {
                // An inverted result: the Pauli that flips this basis's outcome before the
                // measurement, and again after it unless a reset follows.
                let flipper = if basis == Basis::Z { 1 } else { 2 };
                let mut body = Vec::new();
                for tok in &t {
                    let (q, inverted) = match tok.strip_prefix('!') {
                        Some(q) => (qubit(q, &name)?, true),
                        None => (qubit(tok, &name)?, false),
                    };
                    if inverted {
                        body.push(Instr::Pauli { pauli: flipper, qubits: vec![q] });
                    }
                    body.push(Instr::Measure { basis, reset, flip, qubits: vec![q] });
                    if inverted && !reset {
                        body.push(Instr::Pauli { pauli: flipper, qubits: vec![q] });
                    }
                }
                Instr::Gate { line: gate_line(&name, &args, &t), tag: String::new(), body }
            }
        }
        "X_ERROR" | "Y_ERROR" | "Z_ERROR" => {
            exactly(1)?;
            let pauli = match name.as_bytes()[0] {
                b'X' => 1,
                b'Z' => 2,
                _ => 3,
            };
            Instr::PauliError { pauli, p: prob(0)?, qubits: qubit_targets(&t, &name)? }
        }
        "DEPOLARIZE1" => {
            exactly(1)?;
            Instr::Depolarize1 { p: prob(0)?, qubits: qubit_targets(&t, &name)? }
        }
        "DEPOLARIZE2" => {
            exactly(1)?;
            Instr::Depolarize2 { p: prob(0)?, pairs: pair_targets(&t, &name)? }
        }
        "PAULI_CHANNEL_1" => {
            exactly(3)?;
            let (px, py, pz) = (prob(0)?, prob(1)?, prob(2)?);
            if px + py + pz > 1.0 + 1e-12 {
                return Err(format!("{name}: probabilities sum to more than 1"));
            }
            Instr::PauliChannel1 { px, py, pz, qubits: qubit_targets(&t, &name)? }
        }
        "DETECTOR" => Instr::Detector { coords: args.clone(), recs: rec_targets(&t, &name)? },
        "OBSERVABLE_INCLUDE" => {
            exactly(1)?;
            let i = args[0];
            if i < 0.0 || i.fract() != 0.0 {
                return Err(format!("{name}: observable index must be a non-negative integer"));
            }
            let mut recs = Vec::new();
            let mut paulis = Vec::new();
            for tok in &t {
                if tok.starts_with("rec[") {
                    recs.extend(rec_targets(&[tok], &name)?);
                    continue;
                }
                let (body, inverted) = match tok.strip_prefix('!') {
                    Some(b) => (b, true),
                    None => (*tok, false),
                };
                let code = body.chars().next().and_then(pauli_code).filter(|&c| c != 0);
                let (Some(code), Some(q)) = (code, body.get(1..)) else {
                    return Err(format!("{name}: bad target '{tok}': a record (rec[-k]) or a Pauli (X3, !Z0)"));
                };
                paulis.push((qubit(q, &name)?, code, inverted));
            }
            Instr::Observable { index: i as u32, recs, paulis }
        }
        "QUBIT_COORDS" => Instr::QubitCoords { coords: args.clone(), qubits: qubit_targets(&t, &name)? },
        "SHIFT_COORDS" => {
            if !t.is_empty() {
                return Err(format!("{name} takes no targets"));
            }
            Instr::ShiftCoords(args.clone())
        }
        "TICK" => {
            none()?;
            if !t.is_empty() {
                return Err(format!("{name} takes no targets"));
            }
            Instr::Tick
        }
        "S" | "SQRT_Z" => {
            none()?;
            Instr::S(qubit_targets(&t, &name)?)
        }
        "MY" | "MRY" | "MPP" | "MXX" | "MYY" | "MZZ" => {
            if args.len() > 1 {
                return Err(format!("{name} takes at most one argument"));
            }
            let flip = if args.is_empty() { 0.0 } else { prob(0)? };
            Instr::Gate { line: gate_line(&name, &args, &t), tag: String::new(), body: pauli_measurements(&name, flip, &t)? }
        }
        "RY" => {
            none()?;
            let body = qubit_targets(&t, &name)?.into_iter().flat_map(reset_y).collect();
            Instr::Gate { line: gate_line(&name, &args, &t), tag: String::new(), body }
        }
        "SPP" | "SPP_DAG" => {
            none()?;
            let mut body = Vec::new();
            for (product, inverted) in pauli_products(&t, &name)? {
                if product.is_empty() {
                    continue; // ±1: only a global phase
                }
                // The -1 eigenspace phased by i (SPP), or by -i; a negated product swaps them.
                let dag = (name == "SPP_DAG") != inverted;
                let q0 = product[0].0;
                let phase = if dag { vec![Instr::S(vec![q0]), Instr::S(vec![q0]), Instr::S(vec![q0])] } else { vec![Instr::S(vec![q0])] };
                body.extend(around_product(&product, phase));
            }
            Instr::Gate { line: gate_line(&name, &args, &t), tag: String::new(), body }
        }
        "E" | "CORRELATED_ERROR" | "ELSE_CORRELATED_ERROR" => {
            exactly(1)?;
            let mut paulis = Vec::new();
            for tok in &t {
                let code = tok.chars().next().and_then(pauli_code).filter(|&c| c != 0);
                let (Some(code), Some(q)) = (code, tok.get(1..)) else {
                    return Err(format!("{name}: bad Pauli target '{tok}'"));
                };
                let q = qubit(q, &name)?;
                // A qubit may be named twice, as in Stim: its Paulis multiply (every engine
                // applies them in turn, by XOR), and the targets stay as written.
                paulis.push((q, code));
            }
            Instr::Correlated { p: prob(0)?, paulis, chained: name == "ELSE_CORRELATED_ERROR" }
        }
        "PAULI_CHANNEL_2" => {
            exactly(15)?;
            let probs = (0..15).map(prob).collect::<Result<Vec<_>, _>>()?;
            if probs.iter().sum::<f64>() > 1.0 + 1e-12 {
                return Err(format!("{name}: probabilities sum to more than 1"));
            }
            Instr::PauliChannel2 { probs, pairs: pair_targets(&t, &name)? }
        }
        "MPAD" => {
            if args.len() > 1 {
                return Err(format!("{name} takes at most one argument"));
            }
            let flip = explicit_zero(if args.is_empty() { 0.0 } else { prob(0)? }, &args);
            let values = t
                .iter()
                .map(|v| match *v {
                    "0" => Ok(false),
                    "1" => Ok(true),
                    other => Err(format!("{name}: a padded record is 0 or 1, not '{other}'")),
                })
                .collect::<Result<_, _>>()?;
            Instr::Pad { flip, values }
        }
        "HERALDED_ERASE" | "HERALDED_PAULI_CHANNEL_1" => {
            let erase = name == "HERALDED_ERASE";
            exactly(if erase { 1 } else { 4 })?;
            let probs = if erase {
                let p = prob(0)?;
                [p / 4.0; 4]
            } else {
                let v = [prob(0)?, prob(1)?, prob(2)?, prob(3)?];
                if v.iter().sum::<f64>() > 1.0 + 1e-12 {
                    return Err(format!("{name}: probabilities sum to more than 1"));
                }
                v
            };
            Instr::Heralded { erase, args: args.clone(), probs, qubits: qubit_targets(&t, &name)? }
        }
        "I_ERROR" | "II_ERROR" | "II" => {
            for i in 0..args.len() {
                prob(i)?;
            }
            if name == "II" || name == "II_ERROR" {
                pair_targets(&t, &name)?;
            } else {
                qubit_targets(&t, &name)?;
            }
            Instr::Gate { line: gate_line(&name, &args, &t), tag: String::new(), body: Vec::new() }
        }
        _ => match crate::gates::find(&name) {
            Some(def) => {
                none()?;
                let qs = qubit_targets(&t, &name)?;
                if def.arity == 2 {
                    pair_targets(&t, &name)?;
                }
                let mut body = Vec::new();
                for slots in qs.chunks(def.arity as usize) {
                    body.extend(def.steps.iter().map(|step| match *step {
                        crate::gates::Step::H(i) => Instr::H(vec![slots[i as usize]]),
                        crate::gates::Step::S(i) => Instr::S(vec![slots[i as usize]]),
                        crate::gates::Step::Cx(i, j) => Instr::Cx(vec![(slots[i as usize], slots[j as usize])]),
                    }));
                }
                Instr::Gate { line: gate_line(&name, &args, &t), tag: String::new(), body }
            }
            None => return Err(format!("unsupported instruction '{name}'")),
        },
    })
}

fn control(t: &str) -> Option<Control> {
    if let Some(k) = t.strip_prefix("rec[-").and_then(|s| s.strip_suffix(']')).and_then(uint::<u32>) {
        return (k >= 1).then_some(Control::Rec(k));
    }
    let k = t.strip_prefix("sweep[").and_then(|s| s.strip_suffix(']')).and_then(uint::<u32>)?;
    (k <= MAX_QUBIT).then_some(Control::Sweep(k))
}

/// A controlled gate's pairs, some with a measurement record or sweep bit as the control:
/// those become `Feedback`, the rest the gate itself. Z-type controls only, as in Stim: the
/// first of a pair for CX, CY and CZ, the second for XCZ and YCZ (either for CZ).
fn controlled_pairs(name: &str, t: &[&str]) -> Result<Vec<Instr>, String> {
    if !t.len().is_multiple_of(2) {
        return Err(format!("{name}: targets must come in pairs, got {}", t.len()));
    }
    let (pauli, classical_first): (Pauli, bool) = match name {
        "CX" | "CNOT" | "ZCX" => (1, true),
        "CY" | "ZCY" => (3, true),
        "CZ" | "ZCZ" => (2, true),
        "XCZ" => (1, false),
        _ => (3, false), // YCZ
    };
    let mut body = Vec::new();
    for pair in t.chunks(2) {
        let (a, b) = (control(pair[0]), control(pair[1]));
        let fed = match (a, b) {
            (Some(_), Some(_)) => return Err(format!("Measurement record editing is not supported. ({name} {} {}: a pair cannot be two classical bits; Stim parses this but refuses to run it)", pair[0], pair[1])),
            (Some(c), None) if classical_first || pauli == 2 => Some((c, qubit(pair[1], name)?)),
            (None, Some(c)) if !classical_first || pauli == 2 => Some((c, qubit(pair[0], name)?)),
            (None, None) if pair.iter().any(|x| x.starts_with("rec[") || x.starts_with("sweep[")) => {
                return Err(format!("{name}: bad classical target in '{} {}'", pair[0], pair[1]));
            }
            (None, None) => None,
            _ => return Err(format!("Measurement record editing is not supported. ({name} {} {}: a classical bit can only be a Z-type control; Stim parses this but refuses to run it)", pair[0], pair[1])),
        };
        match fed {
            Some((control, qubit)) => body.push(Instr::Feedback { pauli, control, qubit }),
            None => {
                let line = format!("{name} {} {}", pair[0], pair[1]);
                match parse_line(&line)? {
                    Instr::Gate { body: inner, .. } => body.extend(inner),
                    other => body.push(other),
                }
            }
        }
    }
    Ok(body)
}

/// A line as written, for printing a gate the engine runs decomposed.
fn gate_line(name: &str, args: &[f64], t: &[&str]) -> String {
    format!("{} {}", with_args(name, args), t.join(" ")).trim_end().to_string()
}

/// Stim's letter for a Pauli as the engine's code: X 1, Z 2, Y 3 (I 0).
fn pauli_code(c: char) -> Option<Pauli> {
    match c.to_ascii_uppercase() {
        'I' => Some(0),
        'X' => Some(1),
        'Z' => Some(2),
        'Y' => Some(3),
        _ => None,
    }
}

/// `MPP`-style targets: products such as `X0*!Y1*Z2`, each with its qubits and whether an odd
/// number of its factors are negated.
/// A Pauli product: each factor's qubit and Pauli, and whether the product is negated.
type Product = (Vec<(u32, Pauli)>, bool);

fn pauli_products(t: &[&str], name: &str) -> Result<Vec<Product>, String> {
    let mut out = Vec::new();
    for tok in t {
        let mut product: Vec<(u32, Pauli)> = Vec::new();
        let mut inverted = false;
        let mut phase = 0u8;
        for factor in tok.split('*') {
            let factor = match factor.strip_prefix('!') {
                Some(f) => {
                    inverted = !inverted;
                    f
                }
                None => factor,
            };
            let code = factor.chars().next().and_then(pauli_code).filter(|&c| c != 0);
            let (Some(code), Some(q)) = (code, factor.get(1..)) else {
                return Err(format!("{name}: bad Pauli product '{tok}'"));
            };
            let q = qubit(q, name)?;
            // A qubit named twice: its Paulis multiply, as in Stim, with the phase kept.
            match product.iter().position(|&(o, _)| o == q) {
                Some(k) => {
                    let a: Pauli = product[k].1;
                    if a != 0 && a != code {
                        // X·Y = iZ, Y·Z = iX, Z·X = iY; the other order gives -i.
                        let cyclic = matches!((a, code), (1, 3) | (3, 2) | (2, 1));
                        phase += if cyclic { 1 } else { 3 };
                    }
                    product[k].1 = a ^ code;
                }
                None => product.push((q, code)),
            }
        }
        if phase % 2 == 1 {
            return Err(format!("Acted on an anti-Hermitian operator (e.g. X0*Z0 instead of Y0) in {name} {}", t.join(" ")));
        }
        if phase % 4 == 2 {
            inverted = !inverted;
        }
        product.retain(|&(_, c)| c != 0);
        out.push((product, inverted));
    }
    Ok(out)
}

/// The basis change taking a Pauli on `q` to Z (X: H; Y: S† then H), as instructions.
fn to_z(q: u32, p: Pauli) -> Vec<Instr> {
    match p {
        1 => vec![Instr::H(vec![q])],
        3 => vec![Instr::S(vec![q]), Instr::S(vec![q]), Instr::S(vec![q]), Instr::H(vec![q])],
        _ => Vec::new(),
    }
}

/// Its inverse (X: H; Y: H then S).
fn from_z(q: u32, p: Pauli) -> Vec<Instr> {
    match p {
        1 => vec![Instr::H(vec![q])],
        3 => vec![Instr::H(vec![q]), Instr::S(vec![q])],
        _ => Vec::new(),
    }
}

/// `middle` (acting on the product's first qubit, as Z) conjugated into the product: each
/// factor turned to Z, the others' parity gathered onto the first qubit by CX, `middle`, and
/// all of it undone. Measuring Z there measures the product; phasing it phases the product.
fn around_product(product: &[(u32, Pauli)], middle: Vec<Instr>) -> Vec<Instr> {
    let q0 = product[0].0;
    let mut body: Vec<Instr> = product.iter().flat_map(|&(q, p)| to_z(q, p)).collect();
    body.extend(product[1..].iter().map(|&(q, _)| Instr::Cx(vec![(q, q0)])));
    body.extend(middle);
    body.extend(product[1..].iter().rev().map(|&(q, _)| Instr::Cx(vec![(q, q0)])));
    body.extend(product.iter().rev().flat_map(|&(q, p)| from_z(q, p)));
    body
}

/// A reset to |+i⟩, Y's +1 eigenstate: |0⟩, then H, then S.
fn reset_y(q: u32) -> Vec<Instr> {
    vec![Instr::Reset { basis: Basis::Z, qubits: vec![q] }, Instr::H(vec![q]), Instr::S(vec![q])]
}

/// The Y-basis and Pauli-product measurements, each one record: `MY` and `MRY` on qubits,
/// `MPP` on products, `MXX`, `MYY` and `MZZ` on pairs; `!` inverts a result.
fn pauli_measurements(name: &str, flip: f64, t: &[&str]) -> Result<Vec<Instr>, String> {
    let products: Vec<Product> = match name {
        "MPP" => pauli_products(t, name)?,
        "MY" | "MRY" => t
            .iter()
            .map(|tok| {
                let (q, inverted) = match tok.strip_prefix('!') {
                    Some(q) => (q, true),
                    None => (*tok, false),
                };
                Ok((vec![(qubit(q, name)?, 3)], inverted))
            })
            .collect::<Result<_, String>>()?,
        _ => {
            let p = pauli_code(name.as_bytes()[1] as char).unwrap_or(2);
            if !t.len().is_multiple_of(2) {
                return Err(format!("{name}: targets must come in pairs, got {}", t.len()));
            }
            t.chunks(2)
                .map(|pair| {
                    let mut inverted = false;
                    let mut qs = Vec::new();
                    for tok in pair {
                        let q = match tok.strip_prefix('!') {
                            Some(q) => {
                                inverted = !inverted;
                                q
                            }
                            None => tok,
                        };
                        qs.push(qubit(q, name)?);
                    }
                    if qs[0] == qs[1] {
                        return Err(format!("{name}: a pair cannot act twice on qubit {}", qs[0]));
                    }
                    Ok((vec![(qs[0], p), (qs[1], p)], inverted))
                })
                .collect::<Result<_, String>>()?
        }
    };
    let reset = name == "MRY";
    let mut body = Vec::new();
    for (product, inverted) in products {
        if product.is_empty() {
            // A product that cancels to ±1 reads its sign every time (as Stim).
            body.push(Instr::Pad { flip, values: vec![inverted] });
            continue;
        }
        let q0 = product[0].0;
        let mut middle = Vec::new();
        if inverted {
            middle.push(Instr::Pauli { pauli: 1, qubits: vec![q0] });
        }
        middle.push(Instr::Measure { basis: Basis::Z, reset, flip, qubits: vec![q0] });
        if inverted && !reset {
            middle.push(Instr::Pauli { pauli: 1, qubits: vec![q0] });
        }
        if reset {
            // MRY: the record taken, the qubit goes to |+i⟩ (|0⟩ from the reset, then H, S).
            body.extend(to_z(q0, 3));
            body.extend(middle);
            body.extend([Instr::H(vec![q0]), Instr::S(vec![q0])]);
        } else {
            body.extend(around_product(&product, middle));
        }
    }
    Ok(body)
}

/// The most instructions plus targets `resolve` will unroll a circuit into.
const MAX_UNROLLED: u64 = 1 << 24;

/// Instructions plus targets once every REPEAT block is expanded, saturating.
pub(crate) fn unrolled_size(instrs: &[Instr]) -> u64 {
    instrs.iter().fold(0u64, |n, ins| {
        n.saturating_add(match ins {
            Instr::Repeat { count, body, .. } => count.saturating_mul(unrolled_size(body)),
            Instr::Detector { recs, .. } => 1 + recs.len() as u64,
            Instr::Observable { recs, paulis, .. } => 1 + (recs.len() + paulis.len()) as u64,
            other => 1 + other.qubits().len() as u64,
        })
    })
}

fn flatten_into(instrs: &[Instr], out: &mut Vec<Instr>, shift: &mut Vec<f64>) {
    let shifted = |c: &[f64], s: &[f64]| -> Vec<f64> {
        c.iter().enumerate().map(|(i, v)| v + s.get(i).copied().unwrap_or(0.0)).collect()
    };
    for ins in instrs {
        match ins {
            Instr::Repeat { count, body, .. } => {
                for _ in 0..*count {
                    flatten_into(body, out, shift);
                }
            }
            // A tagged annotation moves with the coordinates, and keeps its tag.
            Instr::Gate { body, tag, .. }
                if !tag.is_empty()
                    && matches!(body.as_slice(), [Instr::Detector { .. } | Instr::QubitCoords { .. } | Instr::ShiftCoords(_)]) =>
            {
                let mut inner = Vec::new();
                flatten_into(body, &mut inner, shift);
                out.extend(inner.into_iter().map(|i| tagged(i, tag)));
            }
            Instr::ShiftCoords(s) => {
                if shift.len() < s.len() {
                    shift.resize(s.len(), 0.0);
                }
                for (a, b) in shift.iter_mut().zip(s) {
                    *a += b;
                }
            }
            Instr::Detector { coords, recs } => {
                out.push(Instr::Detector { coords: shifted(coords, shift), recs: recs.clone() })
            }
            Instr::QubitCoords { coords, qubits } => {
                out.push(Instr::QubitCoords { coords: shifted(coords, shift), qubits: qubits.clone() })
            }
            other => out.push(other.clone()),
        }
    }
}

/// Gates replaced by their bodies, each instruction from a tagged one noted in `tags` by its
/// index in `out`.
fn expand_gates(instrs: &[Instr], tag: Option<&Arc<str>>, out: &mut Vec<Instr>, tags: &mut Vec<(usize, Arc<str>)>) {
    for ins in instrs {
        match ins {
            Instr::Gate { body, tag: own, .. } => {
                let own = (!own.is_empty()).then(|| Arc::<str>::from(own.as_str()));
                expand_gates(body, own.as_ref().or(tag), out, tags);
            }
            other => {
                if let Some(t) = tag {
                    tags.push((out.len(), Arc::clone(t)));
                }
                out.push(other.clone());
            }
        }
    }
}

pub(crate) fn fmt_args(a: &[f64]) -> String {
    a.iter().map(|&v| fmt_arg(v)).collect::<Vec<_>>().join(", ")
}

/// A number as Stim writes an instruction's argument (whole numbers in full, others as `%g` to
/// six digits) whenever that holds it exactly; otherwise its shortest exact form, so the text
/// never loses precision where Stim's would.
fn fmt_arg(v: f64) -> String {
    if v.is_finite() && v == v.trunc() && v.abs() < 9.007_199_254_740_992e15 {
        return format!("{}", v as i64);
    }
    let g = crate::dem_program::fmt_g(v, 6);
    if g.parse::<f64>() == Ok(v) {
        g
    } else {
        format!("{v}")
    }
}

fn with_args(name: &str, args: &[f64]) -> String {
    if args.is_empty() {
        name.to_string()
    } else {
        format!("{name}({})", fmt_args(args))
    }
}

fn join_q(q: &[u32]) -> String {
    q.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(" ")
}

fn join_pairs(p: &[(u32, u32)]) -> String {
    p.iter().map(|(a, b)| format!("{a} {b}")).collect::<Vec<_>>().join(" ")
}

fn join_recs(r: &[u32]) -> String {
    r.iter().map(|k| format!("rec[-{k}]")).collect::<Vec<_>>().join(" ")
}

/// A flip probability written as `(0)` is kept as −0.0: zero everywhere it is used, but written
/// back as Stim writes it (Stim keeps an argument it was given).
fn explicit_zero(p: f64, args: &[f64]) -> f64 {
    if p == 0.0 && !args.is_empty() {
        -0.0
    } else {
        p
    }
}

/// Whether a measurement's flip probability is written: positive, or an explicit zero.
fn written(p: f64) -> bool {
    p > 0.0 || (p == 0.0 && p.is_sign_negative())
}

/// One instruction as its line of Stim's text (a loop as its `REPEAT n {` line).
pub(crate) fn instr_line(ins: &Instr) -> String {
    if let Instr::Repeat { count, tag, .. } = ins {
        let tag = if tag.is_empty() { String::new() } else { format!("[{tag}]") };
        return format!("REPEAT{tag} {count} {{");
    }
    let mut s = String::new();
    emit(std::slice::from_ref(ins), "", &mut s);
    s.lines().next().unwrap_or("").to_string()
}

fn emit(instrs: &[Instr], indent: &str, s: &mut String) {
    for ins in instrs {
        let line = match ins {
            Instr::Reset { basis, qubits } => {
                format!("{} {}", if *basis == Basis::X { "RX" } else { "R" }, join_q(qubits))
            }
            Instr::H(q) => format!("H {}", join_q(q)),
            Instr::Cx(p) => format!("CX {}", join_pairs(p)),
            Instr::Cz(p) => format!("CZ {}", join_pairs(p)),
            Instr::Pauli { pauli, qubits } => format!("{} {}", ["I", "X", "Z", "Y"][*pauli as usize], join_q(qubits)),
            Instr::SweepX(pairs) => format!(
                "CX {}",
                pairs.iter().map(|(k, q)| format!("sweep[{k}] {q}")).collect::<Vec<_>>().join(" ")
            ),
            Instr::Measure { basis, reset, flip, qubits } => {
                let name = match (reset, basis) {
                    (false, Basis::Z) => "M",
                    (false, Basis::X) => "MX",
                    (true, Basis::Z) => "MR",
                    (true, Basis::X) => "MRX",
                };
                let args: &[f64] = if written(*flip) { std::slice::from_ref(flip) } else { &[] };
                format!("{} {}", with_args(name, args), join_q(qubits))
            }
            Instr::PauliError { pauli, p, qubits } => {
                let name = match pauli {
                    1 => "X_ERROR",
                    2 => "Z_ERROR",
                    _ => "Y_ERROR",
                };
                format!("{} {}", with_args(name, &[*p]), join_q(qubits))
            }
            Instr::Depolarize1 { p, qubits } => format!("{} {}", with_args("DEPOLARIZE1", &[*p]), join_q(qubits)),
            Instr::Depolarize2 { p, pairs } => format!("{} {}", with_args("DEPOLARIZE2", &[*p]), join_pairs(pairs)),
            Instr::PauliChannel1 { px, py, pz, qubits } => {
                format!("{} {}", with_args("PAULI_CHANNEL_1", &[*px, *py, *pz]), join_q(qubits))
            }
            Instr::Detector { coords, recs } => format!("{} {}", with_args("DETECTOR", coords), join_recs(recs)),
            Instr::Observable { index, recs, paulis } => {
                let mut targets = join_recs(recs);
                for &(q, p, inverted) in paulis {
                    let _ = write!(targets, " {}{}{q}", if inverted { "!" } else { "" }, ["I", "X", "Z", "Y"][p as usize]);
                }
                format!("OBSERVABLE_INCLUDE({index}) {}", targets.trim_start())
            }
            Instr::QubitCoords { coords, qubits } => {
                format!("{} {}", with_args("QUBIT_COORDS", coords), join_q(qubits))
            }
            Instr::ShiftCoords(c) => with_args("SHIFT_COORDS", c),
            Instr::Tick => "TICK".to_string(),
            Instr::S(q) => format!("S {}", join_q(q)),
            Instr::Gate { line, .. } => line.clone(),
            Instr::Correlated { p, paulis, chained } => {
                let name = if *chained { "ELSE_CORRELATED_ERROR" } else { "E" };
                let targets: Vec<String> = paulis.iter().map(|&(q, c)| format!("{}{q}", ["I", "X", "Z", "Y"][c as usize])).collect();
                format!("{} {}", with_args(name, &[*p]), targets.join(" "))
            }
            Instr::PauliChannel2 { probs, pairs } => format!("{} {}", with_args("PAULI_CHANNEL_2", probs), join_pairs(pairs)),
            Instr::Feedback { pauli, control, qubit } => {
                format!("{} {control} {qubit}", match pauli { 1 => "CX", 3 => "CY", _ => "CZ" })
            }
            Instr::Heralded { erase, args, qubits, .. } => {
                let name = if *erase { "HERALDED_ERASE" } else { "HERALDED_PAULI_CHANNEL_1" };
                format!("{} {}", with_args(name, args), join_q(qubits))
            }
            Instr::Pad { flip, values } => {
                let args: &[f64] = if written(*flip) { std::slice::from_ref(flip) } else { &[] };
                let v: Vec<&str> = values.iter().map(|&b| if b { "1" } else { "0" }).collect();
                format!("{} {}", with_args("MPAD", args), v.join(" "))
            }
            Instr::NonPauli(ops) => {
                for op in ops {
                    let _ = writeln!(s, "{indent}{}", op.line());
                }
                continue;
            }
            Instr::Repeat { count, body, tag } => {
                let tag = if tag.is_empty() { String::new() } else { format!("[{tag}]") };
                let _ = writeln!(s, "{indent}REPEAT{tag} {count} {{");
                emit(body, &format!("{indent}    "), s);
                let _ = writeln!(s, "{indent}}}");
                continue;
            }
        };
        let _ = writeln!(s, "{indent}{}", line.trim_end());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixtures::SAMPLE;

    #[test]
    fn parses_every_supported_construct() {
        let c = Circuit::parse(SAMPLE).unwrap();
        assert!(matches!(c.instrs[3], Instr::Reset { basis: Basis::Z, .. }));
        assert!(c.instrs.iter().any(|i| matches!(i, Instr::Repeat { count: 2, .. })));
        let r = c.resolve().unwrap();
        assert_eq!(r.num_qubits, 3);
        assert_eq!(r.num_measurements, 5);
        assert_eq!(r.detectors, vec![vec![0], vec![1, 0], vec![2, 1], vec![4, 3, 2]]);
        assert_eq!(
            r.detector_coords,
            vec![vec![1.0, 0.0, 0.0], vec![1.0, 0.0, 1.0], vec![1.0, 0.0, 2.0], vec![1.0, 0.0, 3.0]]
        );
        assert_eq!(r.observables, vec![vec![4]]);
    }

    #[test]
    fn text_round_trips() {
        let c = Circuit::parse(SAMPLE).unwrap();
        assert_eq!(Circuit::parse(&c.to_stim()).unwrap(), c);
        let flat = c.flattened();
        assert_eq!(Circuit::parse(&flat.to_stim()).unwrap(), flat);
    }

    #[test]
    fn measurement_flips_and_aliases_parse() {
        let c = Circuit::parse("CNOT 0 1\nMZ(0.25) 0\nMRX 1\nRZ 0\nZCZ 0 1").unwrap();
        assert_eq!(c.instrs[0], Instr::Cx(vec![(0, 1)]));
        assert_eq!(c.instrs[1], Instr::Measure { basis: Basis::Z, reset: false, flip: 0.25, qubits: vec![0] });
        assert_eq!(c.instrs[2], Instr::Measure { basis: Basis::X, reset: true, flip: 0.0, qubits: vec![1] });
        assert_eq!(c.instrs[3], Instr::Reset { basis: Basis::Z, qubits: vec![0] });
        assert_eq!(c.instrs[4], Instr::Cz(vec![(0, 1)]));
    }

    #[test]
    fn unsupported_instructions_are_named_with_their_line() {
        let e = Circuit::parse("R 0\nSQRT_W 0\n").unwrap_err();
        assert!(e.contains("line 2") && e.contains("'SQRT_W'"), "{e}");
    }

    #[test]
    fn malformed_targets_are_errors() {
        assert!(Circuit::parse("CX 0 1 2").unwrap_err().contains("pairs"));
        assert!(Circuit::parse("CX 0 0").is_err());
        assert!(Circuit::parse("M !!0").is_err());
        assert!(Circuit::parse("M !0").is_ok());
        assert!(Circuit::parse("X_ERROR(1.5) 0").is_err());
        assert!(Circuit::parse("REPEAT 2 {\nH 0\n").is_err());
    }

    #[test]
    fn sweep_controls_and_pauli_gates_parse_and_round_trip() {
        let text = "R 0 1 2\nCX sweep[0] 1 sweep[3] 2\nX 0 1\nY 2\nZ 0\nI 1 2\nM 0 1 2\nDETECTOR rec[-1]\n";
        let c = Circuit::parse(text).unwrap();
        assert_eq!(c.instrs[1], Instr::SweepX(vec![(0, 1), (3, 2)]));
        assert_eq!(c.instrs[2], Instr::Pauli { pauli: 1, qubits: vec![0, 1] });
        assert_eq!(c.instrs[3], Instr::Pauli { pauli: 3, qubits: vec![2] });
        assert_eq!(c.instrs[4], Instr::Pauli { pauli: 2, qubits: vec![0] });
        assert_eq!(c.instrs[5], Instr::Pauli { pauli: 0, qubits: vec![1, 2] });
        assert_eq!(Circuit::parse(&c.to_stim()).unwrap(), c);
        let r = c.resolve().unwrap();
        assert_eq!(r.num_sweep_bits, 4);
        assert_eq!(r.num_qubits, 3);
    }

    #[test]
    fn classical_bits_are_z_type_controls() {
        assert!(Circuit::parse("CX 0 sweep[1]").is_err());
        assert!(Circuit::parse("CX sweep[0] 1 2 3").is_ok());
        assert!(Circuit::parse("CX sweep[0] sweep[1]").is_err());
        assert!(Circuit::parse("CZ sweep[0] 1").is_ok());
        assert!(Circuit::parse("CZ 1 sweep[0]").is_ok());
        assert!(Circuit::parse("XCZ sweep[0] 1").is_err());
        assert!(Circuit::parse("M 0\nCY rec[-1] 1").is_ok());
        assert!(Circuit::parse("M 0\nCY rec[-2] 1").unwrap().resolve().is_err());
        // An invalid record (rec[-0]) is an error, not a qubit to parse again forever.
        assert!(Circuit::parse("CY 2 rec[-0]").is_err());
        assert!(Circuit::parse("CZ sweep[x] 1").is_err());
        assert!(Circuit::parse("CX sweep[x] 1").is_err());
        assert!(Circuit::parse("X(0.1) 0").is_err());
    }

    #[test]
    fn records_before_the_first_measurement_are_errors() {
        let c = Circuit::parse("M 0\nDETECTOR rec[-2]").unwrap();
        assert!(c.resolve().unwrap_err().contains("rec[-2]"));
        assert!(Circuit::parse("M 16777215").is_ok());
        assert!(Circuit::parse("M 4000000000").unwrap_err().contains("beyond the largest index"));
        assert!(Circuit::parse("CX sweep[4000000000] 0").is_err());
        for arg in ["nan", "inf", "-inf", "1e400"] {
            assert!(Circuit::parse(&format!("DETECTOR({arg}) rec[-1]")).is_err(), "{arg}");
        }
    }

    #[test]
    fn loops_too_large_to_unroll_are_errors() {
        // Nested counts that overflow u64 when multiplied must saturate, not wrap.
        let huge = "REPEAT 4000000000 {\n REPEAT 4000000000 {\n REPEAT 4000000000 {\n TICK\n }\n }\n}";
        for text in ["REPEAT 4000000000 {\n TICK\n}", "REPEAT 100000000 {\n M 0\n}", huge] {
            let err = Circuit::parse(text).unwrap().resolve().unwrap_err();
            assert!(err.contains("REPEAT"), "{text}: {err}");
        }
        let fits = Circuit::parse("REPEAT 1000 {\n M 0 1\n DETECTOR rec[-1] rec[-2]\n}").unwrap();
        assert_eq!(fits.resolve().unwrap().detectors.len(), 1000);
    }
}

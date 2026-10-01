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
    Observable { index: u32, recs: Vec<u32> },
    QubitCoords { coords: Vec<f64>, qubits: Vec<u32> },
    ShiftCoords(Vec<f64>),
    Tick,
    Repeat { count: u64, body: Vec<Instr> },
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
            Instr::Repeat { body, .. } => body.iter().flat_map(|i| i.qubits()).collect(),
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
}

impl Circuit {
    pub fn parse(text: &str) -> Result<Circuit, String> {
        let lines: Vec<&str> = text.lines().collect();
        let mut pos = 0usize;
        let instrs = parse_block(&lines, &mut pos, false)?;
        Ok(Circuit { instrs })
    }

    pub fn to_stim(&self) -> String {
        let mut s = String::new();
        emit(&self.instrs, "", &mut s);
        s
    }

    /// REPEAT blocks expanded, SHIFT_COORDS folded into the coordinates they move.
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
        let flat = self.flattened().instrs;
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
        for ins in &flat {
            for q in ins.qubits() {
                num_qubits = num_qubits.max(q as usize + 1);
            }
            match ins {
                Instr::Measure { qubits, .. } => m += qubits.len(),
                Instr::SweepX(pairs) => {
                    for &(k, _) in pairs {
                        sweeps = sweeps.max(k as usize + 1);
                    }
                }
                Instr::Detector { coords, recs } => {
                    let abs = recs.iter().map(|&k| absolute(k, m)).collect::<Result<Vec<_>, _>>()?;
                    detectors.push(abs);
                    detector_coords.push(coords.clone());
                }
                Instr::Observable { index, recs } => {
                    let i = *index as usize;
                    if i >= 64 {
                        return Err(format!("OBSERVABLE_INCLUDE({i}): at most 64 observables are supported"));
                    }
                    if observables.len() <= i {
                        observables.resize(i + 1, Vec::new());
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
        })
    }
}

fn parse_block(lines: &[&str], pos: &mut usize, nested: bool) -> Result<Vec<Instr>, String> {
    let mut out = Vec::new();
    while *pos < lines.len() {
        let lineno = *pos + 1;
        let line = lines[*pos].split('#').next().unwrap_or("").trim();
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
            let mut parts = rest.split_whitespace();
            let name = parts.next().unwrap_or("");
            if !name.eq_ignore_ascii_case("REPEAT") {
                return Err(format!("line {lineno}: only REPEAT opens a block, got '{name}'"));
            }
            let count: u64 = parts
                .next()
                .and_then(|s| s.parse().ok())
                .ok_or_else(|| format!("line {lineno}: REPEAT needs a count"))?;
            let body = parse_block(lines, pos, true)?;
            out.push(Instr::Repeat { count, body });
            continue;
        }
        out.push(parse_line(line).map_err(|e| format!("line {lineno}: {e}"))?);
    }
    if nested {
        return Err("unterminated REPEAT block".into());
    }
    Ok(out)
}

/// Split `NAME(a, b) t1 t2` into its name (upper-cased), arguments and targets.
/// Shared with the `.dem` reader, whose lines have the same shape.
pub(crate) fn split_instruction(line: &str) -> Result<(String, Vec<f64>, Vec<&str>), String> {
    let name_end = line
        .find(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
        .unwrap_or(line.len());
    if name_end == 0 {
        return Err(format!("expected an instruction name in '{line}'"));
    }
    let name = line[..name_end].to_ascii_uppercase();
    let mut rest = line[name_end..].trim_start();
    let mut args = Vec::new();
    if let Some(inner) = rest.strip_prefix('(') {
        let close = inner.find(')').ok_or_else(|| format!("{name}: unclosed '('"))?;
        for a in inner[..close].split(',') {
            let a = a.trim();
            if a.is_empty() {
                continue;
            }
            args.push(a.parse::<f64>().map_err(|_| format!("{name}: bad argument '{a}'"))?);
        }
        rest = inner[close + 1..].trim_start();
    }
    Ok((name, args, rest.split_whitespace().collect()))
}

/// The largest qubit index Stim accepts (its targets keep 24 bits for it).
const MAX_QUBIT: u32 = (1 << 24) - 1;

fn qubit(t: &str, name: &str) -> Result<u32, String> {
    match t.parse::<u32>() {
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
                .and_then(|s| s.parse::<u32>().ok())
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
                .and_then(|s| s.parse::<u32>().ok())
                .filter(|&k| k >= 1)
                .ok_or_else(|| format!("{name}: bad record target '{t}'"))
        })
        .collect()
}

fn parse_line(line: &str) -> Result<Instr, String> {
    let (name, args, t) = split_instruction(line)?;
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
        "CX" | "CNOT" | "ZCX" => {
            none()?;
            if t.iter().any(|x| x.starts_with("sweep[")) {
                Instr::SweepX(sweep_pairs(&t, &name)?)
            } else {
                Instr::Cx(pair_targets(&t, &name)?)
            }
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
            let flip = if args.is_empty() { 0.0 } else { prob(0)? };
            let basis = if name.contains('X') { Basis::X } else { Basis::Z };
            Instr::Measure { basis, reset: name.starts_with("MR"), flip, qubits: qubit_targets(&t, &name)? }
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
            Instr::Observable { index: i as u32, recs: rec_targets(&t, &name)? }
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
        _ => return Err(format!("unsupported instruction '{name}'")),
    })
}

/// The most instructions plus targets `resolve` will unroll a circuit into.
const MAX_UNROLLED: u64 = 1 << 24;

/// Instructions plus targets once every REPEAT block is expanded, saturating.
fn unrolled_size(instrs: &[Instr]) -> u64 {
    instrs.iter().fold(0u64, |n, ins| {
        n.saturating_add(match ins {
            Instr::Repeat { count, body } => count.saturating_mul(unrolled_size(body)),
            Instr::Detector { recs, .. } | Instr::Observable { recs, .. } => 1 + recs.len() as u64,
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
            Instr::Repeat { count, body } => {
                for _ in 0..*count {
                    flatten_into(body, out, shift);
                }
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

pub(crate) fn fmt_args(a: &[f64]) -> String {
    a.iter().map(|v| format!("{v}")).collect::<Vec<_>>().join(", ")
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
                let args: &[f64] = if *flip > 0.0 { std::slice::from_ref(flip) } else { &[] };
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
            Instr::Observable { index, recs } => format!("OBSERVABLE_INCLUDE({index}) {}", join_recs(recs)),
            Instr::QubitCoords { coords, qubits } => {
                format!("{} {}", with_args("QUBIT_COORDS", coords), join_q(qubits))
            }
            Instr::ShiftCoords(c) => with_args("SHIFT_COORDS", c),
            Instr::Tick => "TICK".to_string(),
            Instr::Repeat { count, body } => {
                let _ = writeln!(s, "{indent}REPEAT {count} {{");
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
        let e = Circuit::parse("R 0\nS 0\n").unwrap_err();
        assert!(e.contains("line 2") && e.contains("'S'"), "{e}");
    }

    #[test]
    fn malformed_targets_are_errors() {
        assert!(Circuit::parse("CX 0 1 2").unwrap_err().contains("pairs"));
        assert!(Circuit::parse("CX 0 0").is_err());
        assert!(Circuit::parse("M !0").is_err());
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
    fn sweep_bits_may_only_control_a_cx() {
        assert!(Circuit::parse("CX 0 sweep[1]").is_err());
        assert!(Circuit::parse("CX sweep[0] 1 2 3").is_err());
        assert!(Circuit::parse("CX sweep[0] sweep[1]").is_err());
        assert!(Circuit::parse("CZ sweep[0] 1").is_err());
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

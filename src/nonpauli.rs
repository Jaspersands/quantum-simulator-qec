//! Operations beyond Clifford gates and Pauli noise, written as Stim instruction tags on the
//! identity instructions (`I[T] 0`, `I_ERROR[R_Z(theta=0.01)] 0 1`, `II_ERROR[R_ZZ(theta=0.02)]
//! 0 1`, `I_ERROR[LEAK(p=0.001)] 3`). Stim reads them as identities, so a circuit stays one
//! Stim runs; this package's Clifford engines do the same, and the coherent sampler and the
//! state vector honour them.
//!
//! A tag names one operation with `key=value` arguments. An unknown tag is left inert, as Stim
//! leaves it; a known one with the wrong arguments or on the wrong instruction is an error.

use crate::circuit::Pauli;

/// One operation a tag asks for, on its qubits.
#[derive(Clone, Debug, PartialEq)]
pub enum NonPauli {
    /// exp(−iθP) for the Pauli product `pauli` (qubit, Pauli code: X 1, Z 2, Y 3).
    Rotation { pauli: Vec<(u32, Pauli)>, theta: f64 },
    /// The T gate, diag(1, e^{iπ/4}), or its inverse.
    T { qubit: u32, dagger: bool },
    /// OpenQASM's U3(θ, φ, λ).
    U3 { qubit: u32, theta: f64, phi: f64, lambda: f64 },
    /// With probability `p`, the qubit leaks out of the computational space.
    Leak { p: f64, qubit: u32 },
    /// With probability `p`, a leaked qubit returns (to |0⟩ or |1⟩ at random).
    Seep { p: f64, qubit: u32 },
    /// With probability `p`, leakage moves from one qubit of the pair to the other.
    LeakTransport { p: f64, a: u32, b: u32 },
    /// Amplitude damping with decay probability `gamma`.
    AmplitudeDamping { gamma: f64, qubit: u32 },
}

impl NonPauli {
    /// The qubits it acts on.
    pub fn qubits(&self) -> Vec<u32> {
        match self {
            NonPauli::Rotation { pauli, .. } => pauli.iter().map(|&(q, _)| q).collect(),
            NonPauli::T { qubit, .. } | NonPauli::U3 { qubit, .. } | NonPauli::Leak { qubit, .. } | NonPauli::Seep { qubit, .. } | NonPauli::AmplitudeDamping { qubit, .. } => vec![*qubit],
            NonPauli::LeakTransport { a, b, .. } => vec![*a, *b],
        }
    }

    /// Whether this is a rotation, the one kind the coherent sampler takes.
    pub fn is_rotation(&self) -> bool {
        matches!(self, NonPauli::Rotation { .. })
    }

    /// The operation as one tagged identity line that reads back as itself.
    pub fn line(&self) -> String {
        let f = |v: f64| format!("{v:?}");
        match self {
            NonPauli::Rotation { pauli, theta } => {
                let letters: String = pauli.iter().map(|&(_, p)| ['I', 'X', 'Z', 'Y'][p as usize]).collect();
                let qs: Vec<String> = pauli.iter().map(|&(q, _)| q.to_string()).collect();
                format!("I_ERROR[R_PAULI(theta={}, pauli={letters})] {}", f(*theta), qs.join(" "))
            }
            NonPauli::T { qubit, dagger } => format!("I[{}] {qubit}", if *dagger { "T_DAG" } else { "T" }),
            NonPauli::U3 { qubit, theta, phi, lambda } => format!("I[U3(theta={}, phi={}, lambda={})] {qubit}", f(*theta), f(*phi), f(*lambda)),
            NonPauli::Leak { p, qubit } => format!("I_ERROR[LEAK(p={})] {qubit}", f(*p)),
            NonPauli::Seep { p, qubit } => format!("I_ERROR[SEEP(p={})] {qubit}", f(*p)),
            NonPauli::LeakTransport { p, a, b } => format!("II_ERROR[LEAK_TRANSPORT(p={})] {a} {b}", f(*p)),
            NonPauli::AmplitudeDamping { gamma, qubit } => format!("I_ERROR[AMPLITUDE_DAMPING(gamma={})] {qubit}", f(*gamma)),
        }
    }

    /// Whether this is one of the leakage operations.
    pub fn is_leakage(&self) -> bool {
        matches!(self, NonPauli::Leak { .. } | NonPauli::Seep { .. } | NonPauli::LeakTransport { .. })
    }
}

/// The tag's name and its `key=value` arguments: `R_Z(theta=0.1)` is ("R_Z", [("theta", "0.1")]).
fn split(tag: &str) -> Result<(String, Vec<(String, String)>), String> {
    let tag = tag.trim();
    let Some(open) = tag.find('(') else {
        return Ok((tag.to_string(), Vec::new()));
    };
    let name = tag[..open].trim().to_string();
    let rest = tag[open + 1..].trim_end();
    let Some(inner) = rest.strip_suffix(')') else {
        return Err(format!("{name}: its arguments are not closed by ')'"));
    };
    let mut args = Vec::new();
    for part in inner.split(',').map(str::trim).filter(|p| !p.is_empty()) {
        let (k, v) = part.split_once('=').ok_or_else(|| format!("{name}: argument '{part}' is not key=value"))?;
        args.push((k.trim().to_string(), v.trim().to_string()));
    }
    Ok((name, args))
}

/// The names this module knows; any other tag is inert.
const KNOWN: &[&str] = &["R_X", "R_Y", "R_Z", "R_XX", "R_YY", "R_ZZ", "R_PAULI", "T", "T_DAG", "U3", "LEAK", "SEEP", "LEAK_TRANSPORT", "AMPLITUDE_DAMPING"];

/// The operations a tagged `I`, `II`, `I_ERROR` or `II_ERROR` asks for on `targets`; `None`
/// for an untagged or unknown one (inert, as in Stim).
pub fn parse(gate: &str, tag: &str, targets: &[u32]) -> Result<Option<Vec<NonPauli>>, String> {
    if tag.is_empty() || !matches!(gate, "I" | "II" | "I_ERROR" | "II_ERROR") {
        return Ok(None);
    }
    let Ok((name, args)) = split(tag) else {
        // A tag that is not of our form at all is someone else's, and inert.
        return Ok(None);
    };
    if !KNOWN.contains(&name.as_str()) {
        return Ok(None);
    }
    let here = format!("{gate}[{tag}]");
    let mut seen = vec![false; args.len()];
    let mut number = |key: &str| -> Result<f64, String> {
        let i = args.iter().position(|(k, _)| k == key).ok_or_else(|| format!("{here}: missing argument '{key}'"))?;
        seen[i] = true;
        let v: f64 = args[i].1.parse().map_err(|_| format!("{here}: '{key}' must be a number, got '{}'", args[i].1))?;
        if !v.is_finite() {
            return Err(format!("{here}: '{key}' must be finite"));
        }
        Ok(v)
    };
    let pair_gate = gate.starts_with("II");
    let single = |what: &str| -> Result<(), String> {
        if pair_gate {
            Err(format!("{here}: {what} acts on single qubits; write it on I or I_ERROR"))
        } else {
            Ok(())
        }
    };
    let pairs = |what: &str| -> Result<(), String> {
        if pair_gate {
            Ok(())
        } else {
            Err(format!("{here}: {what} acts on pairs; write it on II or II_ERROR"))
        }
    };
    let prob = |v: f64, key: &str| -> Result<f64, String> {
        if (0.0..=1.0).contains(&v) {
            Ok(v)
        } else {
            Err(format!("{here}: '{key}' must be a probability, got {v}"))
        }
    };
    let code = |c: char| match c {
        'X' => Some(1u8),
        'Z' => Some(2),
        'Y' => Some(3),
        _ => None,
    };
    let ops: Vec<NonPauli> = match name.as_str() {
        "R_X" | "R_Y" | "R_Z" => {
            single(&name)?;
            let theta = number("theta")?;
            let p = code(name.chars().last().unwrap()).unwrap();
            targets.iter().map(|&q| NonPauli::Rotation { pauli: vec![(q, p)], theta }).collect()
        }
        "R_XX" | "R_YY" | "R_ZZ" => {
            pairs(&name)?;
            let theta = number("theta")?;
            let p = code(name.chars().last().unwrap()).unwrap();
            targets.chunks(2).map(|c| NonPauli::Rotation { pauli: vec![(c[0], p), (c[1], p)], theta }).collect()
        }
        "R_PAULI" => {
            let theta = number("theta")?;
            let i = args.iter().position(|(k, _)| k == "pauli").ok_or_else(|| format!("{here}: missing argument 'pauli'"))?;
            seen[i] = true;
            let s = args[i].1.to_ascii_uppercase();
            if s.chars().count() != targets.len() {
                return Err(format!("{here}: 'pauli' has {} letters for {} targets", s.chars().count(), targets.len()));
            }
            let mut pauli = Vec::new();
            for (c, &q) in s.chars().zip(targets) {
                if c == 'I' || c == '_' {
                    continue;
                }
                pauli.push((q, code(c).ok_or_else(|| format!("{here}: '{c}' is not a Pauli"))?));
            }
            vec![NonPauli::Rotation { pauli, theta }]
        }
        "T" | "T_DAG" => {
            single(&name)?;
            targets.iter().map(|&q| NonPauli::T { qubit: q, dagger: name == "T_DAG" }).collect()
        }
        "U3" => {
            single(&name)?;
            let (theta, phi, lambda) = (number("theta")?, number("phi")?, number("lambda")?);
            targets.iter().map(|&q| NonPauli::U3 { qubit: q, theta, phi, lambda }).collect()
        }
        "LEAK" | "SEEP" => {
            single(&name)?;
            let p = prob(number("p")?, "p")?;
            targets.iter().map(|&q| if name == "LEAK" { NonPauli::Leak { p, qubit: q } } else { NonPauli::Seep { p, qubit: q } }).collect()
        }
        "LEAK_TRANSPORT" => {
            pairs(&name)?;
            let p = prob(number("p")?, "p")?;
            targets.chunks(2).map(|c| NonPauli::LeakTransport { p, a: c[0], b: c[1] }).collect()
        }
        "AMPLITUDE_DAMPING" => {
            single(&name)?;
            let gamma = prob(number("gamma")?, "gamma")?;
            targets.iter().map(|&q| NonPauli::AmplitudeDamping { gamma, qubit: q }).collect()
        }
        _ => unreachable!(),
    };
    if let Some(i) = seen.iter().position(|s| !s) {
        return Err(format!("{here}: unknown argument '{}'", args[i].0));
    }
    Ok(Some(ops))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rotations_parse() {
        let ops = parse("I_ERROR", "R_Z(theta=0.1)", &[0, 1]).unwrap().unwrap();
        assert_eq!(ops, vec![NonPauli::Rotation { pauli: vec![(0, 2)], theta: 0.1 }, NonPauli::Rotation { pauli: vec![(1, 2)], theta: 0.1 }]);
        let ops = parse("II_ERROR", "R_ZZ(theta=-0.25)", &[3, 4]).unwrap().unwrap();
        assert_eq!(ops, vec![NonPauli::Rotation { pauli: vec![(3, 2), (4, 2)], theta: -0.25 }]);
        let ops = parse("I_ERROR", "R_PAULI(theta=0.5, pauli=XIY)", &[0, 1, 2]).unwrap().unwrap();
        assert_eq!(ops, vec![NonPauli::Rotation { pauli: vec![(0, 1), (2, 3)], theta: 0.5 }]);
        let ops = parse("I", "T_DAG", &[5]).unwrap().unwrap();
        assert_eq!(ops, vec![NonPauli::T { qubit: 5, dagger: true }]);
        let ops = parse("I", "U3(theta=1, phi=2, lambda=3)", &[0]).unwrap().unwrap();
        assert_eq!(ops, vec![NonPauli::U3 { qubit: 0, theta: 1.0, phi: 2.0, lambda: 3.0 }]);
        let ops = parse("II_ERROR", "LEAK_TRANSPORT(p=0.5)", &[0, 1]).unwrap().unwrap();
        assert_eq!(ops, vec![NonPauli::LeakTransport { p: 0.5, a: 0, b: 1 }]);
    }

    #[test]
    fn unknown_tags_are_inert_and_bad_known_ones_are_errors() {
        assert_eq!(parse("I_ERROR", "my-own-tag", &[0]).unwrap(), None);
        assert_eq!(parse("I_ERROR", "FOO(x=1)", &[0]).unwrap(), None);
        assert_eq!(parse("H", "T", &[0]).unwrap(), None);
        assert!(parse("I_ERROR", "R_Z(angle=0.1)", &[0]).unwrap_err().contains("missing argument 'theta'"));
        assert!(parse("I_ERROR", "R_Z(theta=0.1, phi=1)", &[0]).unwrap_err().contains("unknown argument 'phi'"));
        assert!(parse("I_ERROR", "R_Z(theta=x)", &[0]).unwrap_err().contains("must be a number"));
        assert!(parse("II_ERROR", "R_Z(theta=0.1)", &[0, 1]).unwrap_err().contains("single qubits"));
        assert!(parse("I_ERROR", "R_ZZ(theta=0.1)", &[0, 1]).unwrap_err().contains("pairs"));
        assert!(parse("I_ERROR", "LEAK(p=2)", &[0]).unwrap_err().contains("probability"));
        assert!(parse("I_ERROR", "R_PAULI(theta=1, pauli=XX)", &[0]).unwrap_err().contains("letters"));
    }
}

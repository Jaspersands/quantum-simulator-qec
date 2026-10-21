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
    /// exp(−iθP) for the Pauli product `pauli` (qubit, Pauli code: X 1, Z 2, Y 3). `noise` is
    /// whether it was written as an error (`I_ERROR`, `II_ERROR`: a coherent error, which the
    /// coherent sampler draws as its twirl and reweights) rather than a gate (`I`, `II`: run
    /// exactly, as the Clifford+T sampler runs it).
    Rotation { pauli: Vec<(u32, Pauli)>, theta: f64, noise: bool },
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
    /// Phase damping (Nielsen and Chuang's): Kraus operators diag(1, √(1−λ)) and diag(0, √λ),
    /// coherences shrinking by √(1−λ).
    PhaseDamping { lambda: f64, qubit: u32 },
    /// Deltakit's `LEAKAGE(p)`: with probability `p`, the qubit is marked leaked and fully
    /// depolarized (as Deltakit's, whether or not it was leaked); measuring it reads its state.
    Leakage { p: f64, qubit: u32 },
    /// Deltakit's `RELAX(p)`: with probability `p`, a leaked qubit is no longer leaked.
    Relax { p: f64, qubit: u32 },
    /// Deltakit's `HERALD_LEAKAGE_EVENT`: the padded record this marker comes before (in the
    /// `MPAD[HERALD_LEAKAGE_EVENT(qubits=…)]` it is read from) is 1 when the qubit is leaked.
    Herald { qubit: u32 },
    /// Deltakit's `RL` (written `I[RL] q`): the qubit is no longer leaked; its state is left
    /// as it is.
    LeakageReset { qubit: u32 },
    /// After a pair of one of Deltakit's transport gates (`CX`, `CY`, `CZ`, `SQRT_XX`,
    /// `SQRT_XX_DAG`), as its frame simulator runs them: a leaked qubit depolarizes the other,
    /// with `p` its leakage spreading (first to second, back) and mobility (first to second,
    /// back). Never read from text: `deltakit::sampler_instrs` places it.
    Transport { a: u32, b: u32, p: [f64; 4] },
}

impl NonPauli {
    /// The qubits it acts on.
    pub fn qubits(&self) -> Vec<u32> {
        match self {
            NonPauli::Rotation { pauli, .. } => pauli.iter().map(|&(q, _)| q).collect(),
            NonPauli::T { qubit, .. }
            | NonPauli::U3 { qubit, .. }
            | NonPauli::Leak { qubit, .. }
            | NonPauli::Seep { qubit, .. }
            | NonPauli::AmplitudeDamping { qubit, .. }
            | NonPauli::PhaseDamping { qubit, .. }
            | NonPauli::Leakage { qubit, .. }
            | NonPauli::Relax { qubit, .. }
            | NonPauli::Herald { qubit }
            | NonPauli::LeakageReset { qubit } => vec![*qubit],
            NonPauli::LeakTransport { a, b, .. } | NonPauli::Transport { a, b, .. } => vec![*a, *b],
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
            NonPauli::Rotation { pauli, theta, noise } => {
                let letters: String = pauli.iter().map(|&(_, p)| ['I', 'X', 'Z', 'Y'][p as usize]).collect();
                let qs: Vec<String> = pauli.iter().map(|&(q, _)| q.to_string()).collect();
                format!("{}[R_PAULI(theta={}, pauli={letters})] {}", if *noise { "I_ERROR" } else { "I" }, f(*theta), qs.join(" "))
            }
            NonPauli::T { qubit, dagger } => format!("I[{}] {qubit}", if *dagger { "T_DAG" } else { "T" }),
            NonPauli::U3 { qubit, theta, phi, lambda } => format!("I[U3(theta={}, phi={}, lambda={})] {qubit}", f(*theta), f(*phi), f(*lambda)),
            NonPauli::Leak { p, qubit } => format!("I_ERROR[LEAK(p={})] {qubit}", f(*p)),
            NonPauli::Seep { p, qubit } => format!("I_ERROR[SEEP(p={})] {qubit}", f(*p)),
            NonPauli::LeakTransport { p, a, b } => format!("II_ERROR[LEAK_TRANSPORT(p={})] {a} {b}", f(*p)),
            NonPauli::AmplitudeDamping { gamma, qubit } => format!("I_ERROR[AMPLITUDE_DAMPING(gamma={})] {qubit}", f(*gamma)),
            NonPauli::PhaseDamping { lambda, qubit } => format!("I_ERROR[PHASE_DAMPING(lambda={})] {qubit}", f(*lambda)),
            NonPauli::Leakage { p, qubit } => format!("I_ERROR[LEAKAGE(p={})] {qubit}", f(*p)),
            NonPauli::Relax { p, qubit } => format!("I_ERROR[RELAX(p={})] {qubit}", f(*p)),
            NonPauli::Herald { qubit } => format!("MPAD[{}(qubits={qubit})] 0", crate::leakage::HERALD),
            NonPauli::LeakageReset { qubit } => format!("I[RL] {qubit}"),
            NonPauli::Transport { a, b, p } => format!("CX[{}(spread={}, spread_back={}, mobility={}, mobility_back={})] {a} {b}", crate::leakage::TRANSPORT, f(p[0]), f(p[1]), f(p[2]), f(p[3])),
        }
    }

    /// Whether this is one of the leakage operations.
    pub fn is_leakage(&self) -> bool {
        matches!(self, NonPauli::Leak { .. } | NonPauli::Seep { .. } | NonPauli::LeakTransport { .. } | NonPauli::Leakage { .. } | NonPauli::Relax { .. } | NonPauli::Herald { .. } | NonPauli::LeakageReset { .. } | NonPauli::Transport { .. })
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
const KNOWN: &[&str] = &["R_X", "R_Y", "R_Z", "R_XX", "R_YY", "R_ZZ", "R_PAULI", "T", "T_DAG", "U3", "LEAK", "SEEP", "LEAK_TRANSPORT", "AMPLITUDE_DAMPING", "PHASE_DAMPING", "THERMAL_RELAXATION", "LEAKAGE", "RELAX", "RL"];

/// The operations a tagged `I`, `II`, `I_ERROR` or `II_ERROR` asks for on `targets`; `None`
/// for an untagged or unknown one (inert, as in Stim).
pub fn parse(gate: &str, tag: &str, targets: &[u32]) -> Result<Option<Vec<NonPauli>>, String> {
    if tag.is_empty() || !matches!(gate, "I" | "II" | "I_ERROR" | "II_ERROR") {
        return Ok(None);
    }
    let (name, args) = match split(tag) {
        Ok(parsed) => parsed,
        // A known name written wrongly (`AMPLITUDE_DAMPING(0.1)`, its argument unnamed) is an
        // error, not someone else's tag to ignore.
        Err(e) if KNOWN.contains(&tag.split('(').next().unwrap_or("").trim()) => return Err(format!("{gate}[{tag}]: {e}")),
        // A tag that is not of our form at all is someone else's, and inert.
        Err(_) => return Ok(None),
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
    let noise = gate.ends_with("_ERROR");
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
            targets.iter().map(|&q| NonPauli::Rotation { pauli: vec![(q, p)], theta, noise }).collect()
        }
        "R_XX" | "R_YY" | "R_ZZ" => {
            pairs(&name)?;
            let theta = number("theta")?;
            let p = code(name.chars().last().unwrap()).unwrap();
            targets.chunks(2).map(|c| NonPauli::Rotation { pauli: vec![(c[0], p), (c[1], p)], theta, noise }).collect()
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
            vec![NonPauli::Rotation { pauli, theta, noise }]
        }
        "T" | "T_DAG" => {
            single(&name)?;
            if noise {
                return Err(format!("{here}: {name} is a gate, written I[{name}]; a coherent error is a rotation (I_ERROR[R_Z(theta=0.7853981633974483)] for a T-sized over-rotation)"));
            }
            targets.iter().map(|&q| NonPauli::T { qubit: q, dagger: name == "T_DAG" }).collect()
        }
        "U3" => {
            single(&name)?;
            if noise {
                return Err(format!("{here}: U3 is a gate, written I[U3(...)]; a coherent error is a rotation (I_ERROR[R_Z(theta=...)] and the rest)"));
            }
            let (theta, phi, lambda) = (number("theta")?, number("phi")?, number("lambda")?);
            targets.iter().map(|&q| NonPauli::U3 { qubit: q, theta, phi, lambda }).collect()
        }
        "LEAK" | "SEEP" => {
            single(&name)?;
            let p = prob(number("p")?, "p")?;
            targets.iter().map(|&q| if name == "LEAK" { NonPauli::Leak { p, qubit: q } } else { NonPauli::Seep { p, qubit: q } }).collect()
        }
        "LEAKAGE" | "RELAX" => {
            single(&name)?;
            let p = prob(number("p")?, "p")?;
            targets.iter().map(|&q| if name == "LEAKAGE" { NonPauli::Leakage { p, qubit: q } } else { NonPauli::Relax { p, qubit: q } }).collect()
        }
        "RL" => {
            single(&name)?;
            targets.iter().map(|&q| NonPauli::LeakageReset { qubit: q }).collect()
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
        "PHASE_DAMPING" => {
            single(&name)?;
            let lambda = prob(number("lambda")?, "lambda")?;
            targets.iter().map(|&q| NonPauli::PhaseDamping { lambda, qubit: q }).collect()
        }
        // Idling for `t` with relaxation time T1 and dephasing time T2 (T2 ≤ 2 T1), as
        // Quantumsim's AmpPhDamp: amplitude damping γ = 1 − e^{−t/T1}, then phase damping
        // λ = 1 − e^{−2t/Tφ}, 1/Tφ = 1/T2 − 1/(2 T1), so coherences decay as e^{−t/T2}.
        "THERMAL_RELAXATION" => {
            single(&name)?;
            let (t, t1, t2) = (number("t")?, number("t1")?, number("t2")?);
            if t < 0.0 || t1 <= 0.0 || t2 <= 0.0 {
                return Err(format!("{here}: 't' must be at least 0 and 't1', 't2' positive"));
            }
            if t2 > 2.0 * t1 * (1.0 + 1e-12) {
                return Err(format!("{here}: 't2' must not be greater than 2 t1"));
            }
            let gamma = 1.0 - (-t / t1).exp();
            let rate = (1.0 / t2 - 1.0 / (2.0 * t1)).max(0.0);
            let lambda = 1.0 - (-2.0 * t * rate).exp();
            targets.iter().flat_map(|&q| [NonPauli::AmplitudeDamping { gamma, qubit: q }, NonPauli::PhaseDamping { lambda, qubit: q }]).collect()
        }
        _ => unreachable!(),
    };
    if let Some(i) = seen.iter().position(|s| !s) {
        return Err(format!("{here}: unknown argument '{}'", args[i].0));
    }
    Ok(Some(ops))
}

/// The tag of the inverse of a tagged `I`/`II` gate: T ↔ T_DAG, a rotation's angle negated
/// (an `R_PAULI`'s letters reversed as the inverse writes its targets: in reverse, or on a gate
/// of pairs (`pairs`: `II`, `II_ERROR`) pair by pair, each pair's order kept, so each letter
/// stays on its qubit), U3(θ, φ, λ)⁻¹ = U3(−θ, −λ, −φ). `None` for a tag that is not one of this
/// module's operations (inert, its own inverse); an error for an operation with no inverse
/// (damping, leakage).
pub fn inverse_tag(tag: &str, pairs: bool) -> Result<Option<String>, String> {
    let Ok((name, args)) = split(tag) else {
        return Ok(None);
    };
    if !KNOWN.contains(&name.as_str()) {
        return Ok(None);
    }
    let neg = |v: &str| -> Result<String, String> {
        let x: f64 = v.parse().map_err(|_| format!("{tag}: '{v}' is not a number"))?;
        Ok(format!("{:?}", -x))
    };
    let joined = |args: &[(String, String)]| format!("{name}({})", args.iter().map(|(k, v)| format!("{k}={v}")).collect::<Vec<_>>().join(", "));
    Ok(Some(match name.as_str() {
        "T" => "T_DAG".to_string(),
        "T_DAG" => "T".to_string(),
        "R_X" | "R_Y" | "R_Z" | "R_XX" | "R_YY" | "R_ZZ" | "R_PAULI" => {
            let mut a = args;
            for (k, v) in a.iter_mut() {
                if k == "theta" {
                    *v = neg(v)?;
                } else if k == "pauli" {
                    let letters: Vec<char> = v.chars().collect();
                    *v = if pairs { letters.chunks(2).rev().flatten().collect() } else { letters.into_iter().rev().collect() };
                }
            }
            joined(&a)
        }
        "U3" => {
            let get = |key: &str| args.iter().find(|(k, _)| k == key).map(|(_, v)| v.clone()).ok_or_else(|| format!("{tag}: missing argument '{key}'"));
            let (t, p, l) = (get("theta")?, get("phi")?, get("lambda")?);
            format!("U3(theta={}, phi={}, lambda={})", neg(&t)?, neg(&l)?, neg(&p)?)
        }
        _ => return Err(format!("The circuit has no well-defined inverse because it contains '{tag}', which is not unitary.")),
    }))
}

/// A number as `%g` writes it with 6 significant digits (`0.1`, `-0.25`, `1`, `1e-07`).
fn short(x: f64) -> String {
    if x == 0.0 || !x.is_finite() {
        return if x == 0.0 { "0".to_string() } else { format!("{x}") };
    }
    let trim = |s: String| if s.contains('.') { s.trim_end_matches('0').trim_end_matches('.').to_string() } else { s };
    let e = x.abs().log10().floor() as i32;
    // The exponent after rounding to 6 digits (0.9999995 is 1).
    let rounded: f64 = format!("{x:.5e}").parse().unwrap_or(x);
    let e = if rounded.abs() >= 10f64.powi(e + 1) { e + 1 } else { e };
    if (-4..6).contains(&e) {
        trim(format!("{x:.*}", (5 - e) as usize))
    } else {
        let s = format!("{x:.5e}");
        let (m, exp) = s.split_once('e').unwrap_or((&s, "0"));
        let exp: i32 = exp.parse().unwrap_or(0);
        format!("{}e{}{:02}", trim(m.to_string()), if exp < 0 { '-' } else { '+' }, exp.abs())
    }
}

/// An angle in radians as a multiple of π (`0.1π`).
fn turns(radians: f64) -> String {
    format!("{}π", short(radians / std::f64::consts::PI))
}

/// How a diagram labels a tagged gate on `I` or `II` (`I[T]`, `I[R_X(theta=…)]`,
/// `II[R_ZZ(theta=…)]`, `I[R_PAULI(…)]`, `I[U3(…)]`): groups of (qubit, label) cells, a group
/// of several qubits drawn joined. Rotations are written as Tsim writes them, by their angle
/// as a multiple of π (`R_X(0.1π)` is exp(−i 0.1π X / 2), the tag's θ = 0.05π), Pauli
/// products factor by factor (`R_PAULI[X](0.3π)`, and `TPP[X]` / `TPP_DAG[X]` at ±π/8, Tsim's
/// T-like product rotations). `None` for anything else — coherent errors, leakage, damping,
/// other gates — which keeps its drawing.
pub fn diagram_labels(gate: &str, tag: &str, ops: &[NonPauli]) -> Option<Vec<Vec<(u32, String)>>> {
    if !matches!(gate, "I" | "II") {
        return None;
    }
    let name = tag.split('(').next().unwrap_or("").trim();
    let letter = |p: u8| ['I', 'X', 'Z', 'Y'][p as usize & 3];
    let mut out = Vec::new();
    for op in ops {
        match op {
            NonPauli::T { qubit, dagger } => out.push(vec![(*qubit, if *dagger { "T_DAG" } else { "T" }.to_string())]),
            NonPauli::U3 { qubit, theta, phi, lambda } => out.push(vec![(*qubit, format!("U3({},{},{})", turns(*theta), turns(*phi), turns(*lambda)))]),
            NonPauli::Rotation { pauli, theta, noise: false } => {
                // The rotation angle is 2θ: exp(−iθP) = R_P(2θ).
                let angle = turns(2.0 * theta);
                if pauli.is_empty() {
                    continue;
                }
                let same_pair = pauli.len() == 2 && pauli[0].1 == pauli[1].1;
                let cells: Vec<(u32, String)> = match name {
                    "R_X" | "R_Y" | "R_Z" | "R_XX" | "R_YY" | "R_ZZ" => {
                        let label = format!("{name}({angle})");
                        pauli.iter().map(|&(q, _)| (q, label.clone())).collect()
                    }
                    _ if theta.abs() == std::f64::consts::FRAC_PI_8 => {
                        let head = if *theta > 0.0 { "TPP" } else { "TPP_DAG" };
                        pauli.iter().map(|&(q, p)| (q, format!("{head}[{}]", letter(p)))).collect()
                    }
                    _ if same_pair => {
                        let p = letter(pauli[0].1);
                        let label = format!("R_{p}{p}({angle})");
                        pauli.iter().map(|&(q, _)| (q, label.clone())).collect()
                    }
                    _ => pauli.iter().map(|&(q, p)| (q, format!("R_PAULI[{}]({angle})", letter(p)))).collect(),
                };
                out.push(cells);
            }
            _ => return None,
        }
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diagram_labels_of_tagged_gates() {
        let labels = |gate: &str, tag: &str, targets: &[u32]| diagram_labels(gate, tag, &parse(gate, tag, targets).unwrap().unwrap());
        assert_eq!(labels("I", "T", &[0, 2]).unwrap(), vec![vec![(0, "T".to_string())], vec![(2, "T".to_string())]]);
        assert_eq!(labels("I", "T_DAG", &[1]).unwrap(), vec![vec![(1, "T_DAG".to_string())]]);
        assert_eq!(labels("I", "R_X(theta=0.15707963267948966)", &[0]).unwrap(), vec![vec![(0, "R_X(0.1π)".to_string())]]);
        assert_eq!(labels("II", "R_ZZ(theta=-0.7853981633974483)", &[0, 3]).unwrap(), vec![vec![(0, "R_ZZ(-0.5π)".to_string()), (3, "R_ZZ(-0.5π)".to_string())]]);
        assert_eq!(labels("I", "R_PAULI(theta=0.39269908169872414, pauli=XY)", &[0, 1]).unwrap(), vec![vec![(0, "TPP[X]".to_string()), (1, "TPP[Y]".to_string())]]);
        assert_eq!(labels("I", "R_PAULI(theta=-0.39269908169872414, pauli=Z)", &[4]).unwrap(), vec![vec![(4, "TPP_DAG[Z]".to_string())]]);
        assert_eq!(labels("I", "R_PAULI(theta=0.7853981633974483, pauli=ZZ)", &[0, 2]).unwrap(), vec![vec![(0, "R_ZZ(0.5π)".to_string()), (2, "R_ZZ(0.5π)".to_string())]]);
        assert_eq!(labels("I", "R_PAULI(theta=0.4712388980384689, pauli=XIZ)", &[0, 1, 2]).unwrap(), vec![vec![(0, "R_PAULI[X](0.3π)".to_string()), (2, "R_PAULI[Z](0.3π)".to_string())]]);
        assert_eq!(labels("I", "U3(theta=0.3141592653589793, phi=0.6283185307179586, lambda=0.9424777960769379)", &[0]).unwrap(), vec![vec![(0, "U3(0.1π,0.2π,0.3π)".to_string())]]);
        assert_eq!(labels("I_ERROR", "R_Z(theta=0.1)", &[0]), None);
        assert_eq!(labels("I_ERROR", "LEAK(p=0.1)", &[0]), None);
        assert_eq!(short(1e-7), "1e-07");
        assert_eq!(short(1e-5), "1e-05");
        assert_eq!(short(0.0001), "0.0001");
        assert_eq!(short(123456.7), "123457");
        assert_eq!(short(1234567.0), "1.23457e+06");
        assert_eq!(short(0.9999999), "1");
        assert_eq!(short(-0.25), "-0.25");
    }

    #[test]
    fn inverse_tags() {
        assert_eq!(inverse_tag("T", false).unwrap().unwrap(), "T_DAG");
        assert_eq!(inverse_tag("T_DAG", false).unwrap().unwrap(), "T");
        assert_eq!(inverse_tag("R_Z(theta=0.25)", false).unwrap().unwrap(), "R_Z(theta=-0.25)");
        assert_eq!(inverse_tag("R_PAULI(theta=-0.5, pauli=XZY)", false).unwrap().unwrap(), "R_PAULI(theta=0.5, pauli=YZX)");
        // On II the targets are reversed pair by pair (0 1 2 3 is 2 3 0 1), and so are the letters.
        assert_eq!(inverse_tag("R_PAULI(theta=0.1, pauli=XZYy)", true).unwrap().unwrap(), "R_PAULI(theta=-0.1, pauli=YyXZ)");
        assert_eq!(inverse_tag("R_PAULI(pauli=X_Z_Yx, theta=0.1)", true).unwrap().unwrap(), "R_PAULI(pauli=YxZ_X_, theta=-0.1)");
        assert_eq!(inverse_tag("R_ZZ(theta=0.1)", true).unwrap().unwrap(), "R_ZZ(theta=-0.1)");
        assert_eq!(inverse_tag("U3(theta=1.0, phi=2.0, lambda=3.0)", false).unwrap().unwrap(), "U3(theta=-1.0, phi=-3.0, lambda=-2.0)");
        assert_eq!(inverse_tag("mine", false).unwrap(), None);
        assert!(inverse_tag("LEAK(p=0.1)", false).unwrap_err().contains("not unitary"));
    }

    #[test]
    fn rotations_parse() {
        let ops = parse("I_ERROR", "R_Z(theta=0.1)", &[0, 1]).unwrap().unwrap();
        assert_eq!(ops, vec![NonPauli::Rotation { pauli: vec![(0, 2)], theta: 0.1, noise: true }, NonPauli::Rotation { pauli: vec![(1, 2)], theta: 0.1, noise: true }]);
        let ops = parse("II_ERROR", "R_ZZ(theta=-0.25)", &[3, 4]).unwrap().unwrap();
        assert_eq!(ops, vec![NonPauli::Rotation { pauli: vec![(3, 2), (4, 2)], theta: -0.25, noise: true }]);
        let ops = parse("I", "R_PAULI(theta=0.5, pauli=XIY)", &[0, 1, 2]).unwrap().unwrap();
        assert_eq!(ops, vec![NonPauli::Rotation { pauli: vec![(0, 1), (2, 3)], theta: 0.5, noise: false }]);
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

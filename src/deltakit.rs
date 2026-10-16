//! Deltakit's Stim (`deltakit_stim` 0.2.7, Riverlane's fork of Stim 1.16): how its leakage runs,
//! for the leakage sampler and the error model in its mode.
//!
//! WHAT DELTAKIT'S STIM DOES
//! ------------------------
//! Its frame simulator keeps a leaked flag per qubit and shot. `LEAKAGE(p) q` leaks q with
//! probability p and depolarizes it; `RELAX(p) q` returns a leaked q with probability p, its
//! state as it is; `RL q` and every reset return q. A leaked qubit is measured as it stands.
//! `HERALD_LEAKAGE_EVENT(p) q` records whether q is leaked, a leaked one read as unleaked with
//! probability p (never the other way). After each pair of a `CX`, `CY`, `CZ`, `SQRT_XX` or
//! `SQRT_XX_DAG` (only these), a leaked qubit depolarizes its partner, and the gate's four
//! optional arguments spread leakage to the partner (`CX(p1, p2, p3, p4)`: spreading from the
//! first qubit to the second and back, then mobility, which moves the leakage to an unleaked
//! partner, from the first to the second and back), the first qubit's effects before the
//! second's.
//!
//! Its error analyzer adds leakage only where a herald reads it. A herald on q, met walking
//! backwards, remembers its record's detectors and how much leakage reached q since its last
//! reset (the sum of the `LEAKAGE` probabilities, in single precision as Deltakit sums them),
//! and adds q's depolarization with the herald firing (a 50% error on the herald and what q's
//! X, Z or Y flips, pieces that are the herald alone left out). Each transport gate before it
//! on q depolarizes the partner the same way, with the share of q's leakage that came before
//! the gate (`LEAKAGE` met further back takes its probability off), and a reset forgets the
//! herald. These terms join the model's errors as Stim's do; Deltakit's `RELAX` adds nothing.
//!
//! Here the circuit holds Deltakit's instructions as the tagged lines it reads them as (see
//! `leakage`): `I_ERROR[LEAKAGE(p=…)]`, `I_ERROR[RELAX(p=…)]`, `I[RL]`,
//! `MPAD[HERALD_LEAKAGE_EVENT(qubits=…)](p)` and `CX[LEAKAGE_TRANSPORT(spread=…, …)]`. This
//! module tells, for each circuit-level instruction, what it is to Deltakit.

use std::collections::HashMap;

use crate::circuit::{Circuit, Instr};
use crate::nonpauli::NonPauli;

/// The gates leakage passes through, by canonical name.
const TRANSPORT_GATES: [&str; 5] = ["CX", "CY", "CZ", "SQRT_XX", "SQRT_XX_DAG"];

/// The resets of Deltakit's Stim besides `RL` (its gates flagged as resets), by canonical name.
const RESETS: [&str; 6] = ["R", "RX", "RY", "MR", "MRX", "MRY"];

/// What a circuit-level instruction is to Deltakit's leakage.
pub(crate) enum Role {
    /// A transport gate: its qubit pairs (pairs with a record or sweep bit left out), its four
    /// probabilities (zero without arguments), and its own Stim tag, if any.
    Transport { name: String, targets: Vec<String>, pairs: Vec<(u32, u32)>, p: [f64; 4], tag: Option<String> },
    /// A reset of these qubits.
    Reset(Vec<u32>),
    /// A herald: per record, its qubit and whether its target is inverted, and the records'
    /// flip probability.
    Herald { qubits: Vec<(u32, bool)>, flip: f64 },
    Other,
}

fn canonical(name: &str) -> String {
    crate::gate_data::info(name).map_or_else(|| name.to_string(), |g| g.name.to_string())
}

fn qubit(t: &str) -> Option<u32> {
    crate::circuit::uint::<u32>(t.strip_prefix('!').unwrap_or(t))
}

/// The role of `ins`, an instruction at the circuit's level (not inside a gate's body).
pub(crate) fn role(ins: &Instr) -> Role {
    match ins {
        Instr::Cx(pairs) => Role::Transport { name: "CX".into(), targets: pair_targets(pairs), pairs: pairs.clone(), p: [0.0; 4], tag: None },
        Instr::Cz(pairs) => Role::Transport { name: "CZ".into(), targets: pair_targets(pairs), pairs: pairs.clone(), p: [0.0; 4], tag: None },
        Instr::Reset { qubits, .. } | Instr::Measure { reset: true, qubits, .. } => Role::Reset(qubits.clone()),
        Instr::Gate { line, tag, body } => {
            if let [Instr::NonPauli(ops), Instr::Pad { flip, values }] = body.as_slice() {
                if ops.iter().all(|o| matches!(o, NonPauli::Herald { .. })) && ops.len() == values.len() && crate::leakage::herald_qubits(tag).is_some() {
                    let qubits = ops.iter().zip(values).map(|(o, &v)| (o.qubits()[0], v)).collect();
                    return Role::Herald { qubits, flip: *flip };
                }
            }
            let Ok((name, _, _, targets)) = crate::circuit::split_instruction(line.trim()) else {
                return Role::Other;
            };
            let name = canonical(&name);
            if TRANSPORT_GATES.contains(&name.as_str()) {
                let pairs = targets.chunks(2).filter_map(|c| Some((qubit(c[0])?, qubit(*c.get(1)?)?))).collect();
                let p = crate::leakage::transport_args(tag).and_then(Result::ok).unwrap_or([0.0; 4]);
                let own = (!tag.is_empty() && crate::leakage::transport_args(tag).is_none()).then(|| tag.clone());
                return Role::Transport { name, targets: targets.iter().map(|t| t.to_string()).collect(), pairs, p, tag: own };
            }
            if RESETS.contains(&name.as_str()) {
                return Role::Reset(targets.iter().filter_map(|t| qubit(t)).collect());
            }
            Role::Other
        }
        _ => Role::Other,
    }
}

fn pair_targets(pairs: &[(u32, u32)]) -> Vec<String> {
    pairs.iter().flat_map(|&(a, b)| [a.to_string(), b.to_string()]).collect()
}

/// The circuit's instructions as Deltakit's frame simulator runs them: a transport gate's pairs
/// each on their own, each qubit pair followed by its transport (`NonPauli::Transport`).
pub(crate) fn sampler_instrs(instrs: &[Instr]) -> Result<Vec<Instr>, String> {
    let mut out = Vec::with_capacity(instrs.len());
    for ins in instrs {
        if let Instr::Repeat { count, body, tag } = ins {
            out.push(Instr::Repeat { count: *count, body: sampler_instrs(body)?, tag: tag.clone() });
            continue;
        }
        let Role::Transport { name, targets, p, .. } = role(ins) else {
            out.push(ins.clone());
            continue;
        };
        for pair in targets.chunks(2) {
            let piece = match ins {
                Instr::Cx(_) | Instr::Cz(_) => {
                    let (a, b) = (qubit(&pair[0]).expect("a qubit"), qubit(&pair[1]).expect("a qubit"));
                    vec![if matches!(ins, Instr::Cx(_)) { Instr::Cx(vec![(a, b)]) } else { Instr::Cz(vec![(a, b)]) }]
                }
                _ => Circuit::parse(&format!("{name} {} {}", pair[0], pair[1]))?.instrs,
            };
            out.extend(piece);
            if let (Some(a), Some(b)) = (qubit(&pair[0]), qubit(&pair[1])) {
                out.push(Instr::NonPauli(vec![NonPauli::Transport { a, b, p }]));
            }
        }
    }
    Ok(out)
}

/// The records an instruction makes (a loop's not counted).
fn records(ins: &Instr) -> u64 {
    match ins {
        Instr::Measure { qubits, .. } | Instr::Heralded { qubits, .. } => qubits.len() as u64,
        Instr::Pad { values, .. } => values.len() as u64,
        Instr::Gate { body, .. } => body.iter().map(records).sum(),
        _ => 0,
    }
}

fn has_heralds(instrs: &[Instr]) -> bool {
    instrs.iter().any(|i| match i {
        Instr::Repeat { body, .. } => has_heralds(body),
        i => matches!(role(i), Role::Herald { .. }),
    })
}

/// For each herald record (by its index among the circuit's records): the leakage that reached
/// its qubit since the qubit's last reset, `LEAKAGE` probabilities summed in single precision,
/// in Deltakit's forward pass (`get_num_meas_before_her_to_pl`): a reset that also measures
/// (`MR`) counts as a measurement there, not a reset, and an inverted herald target reads none.
/// Empty for a circuit without heralds, as Deltakit's.
pub(crate) fn herald_leakage(instrs: &[Instr]) -> HashMap<u64, f32> {
    let mut out = HashMap::new();
    if !has_heralds(instrs) {
        return out;
    }
    fn walk(instrs: &[Instr], index: &mut u64, seen: &mut HashMap<u32, f32>, out: &mut HashMap<u64, f32>) {
        for ins in instrs {
            if let Instr::Repeat { count, body, .. } = ins {
                for _ in 0..*count {
                    walk(body, index, seen, out);
                }
                continue;
            }
            let role = role(ins);
            if let Role::Herald { qubits, .. } = &role {
                for &(q, inverted) in qubits {
                    let pl = if inverted { 0.0 } else { seen.get(&q).copied().unwrap_or(0.0) };
                    out.entry(*index).or_insert(pl);
                    *index += 1;
                }
                continue;
            }
            let n = records(ins);
            if n > 0 {
                *index += n;
                continue;
            }
            let ops = nonpauli_ops(ins);
            if ops.iter().any(|o| matches!(o, NonPauli::Leakage { .. })) {
                for o in ops {
                    if let NonPauli::Leakage { p, qubit } = o {
                        let s = seen.entry(*qubit).or_insert(0.0);
                        *s = (f64::from(*s) + p) as f32;
                    }
                }
            } else if let Role::Reset(qubits) = role {
                for q in qubits {
                    seen.insert(q, 0.0);
                }
            } else if ops.iter().any(|o| matches!(o, NonPauli::LeakageReset { .. })) {
                for o in ops {
                    seen.insert(o.qubits()[0], 0.0);
                }
            }
        }
    }
    walk(instrs, &mut 0, &mut HashMap::new(), &mut out);
    out
}

/// The non-Pauli operations a circuit-level instruction is (a tagged identity's).
pub(crate) fn nonpauli_ops(ins: &Instr) -> &[NonPauli] {
    match ins {
        Instr::NonPauli(ops) => ops,
        Instr::Gate { body, .. } => match body.as_slice() {
            [Instr::NonPauli(ops)] => ops,
            _ => &[],
        },
        _ => &[],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roles_of_the_dialect() {
        let c = Circuit::parse("CX 0 1\nCY(0.1, 0.2, 0.3, 0.4) 2 3\nCX rec[-1] 4 5 6\nSWAP 0 1\nMR 2\nRY 3\nRL 4\nHERALD_LEAKAGE_EVENT(0.01) 1 !2").unwrap_or_else(|e| panic!("{e}"));
        let roles: Vec<String> = c
            .instrs
            .iter()
            .map(|i| match role(i) {
                Role::Transport { name, pairs, p, .. } => format!("{name} {pairs:?} {p:?}"),
                Role::Reset(q) => format!("reset {q:?}"),
                Role::Herald { qubits, flip } => format!("herald {qubits:?} {flip}"),
                Role::Other => "other".into(),
            })
            .collect();
        assert_eq!(
            roles,
            [
                "CX [(0, 1)] [0.0, 0.0, 0.0, 0.0]",
                "CY [(2, 3)] [0.1, 0.2, 0.3, 0.4]",
                "CX [(5, 6)] [0.0, 0.0, 0.0, 0.0]",
                "other",
                "reset [2]",
                "reset [3]",
                "other",
                "herald [(1, false), (2, true)] 0.01"
            ]
        );
    }

    #[test]
    fn herald_leakage_sums_since_the_last_reset() {
        let c = Circuit::parse("R 0 1\nLEAKAGE(0.1) 0 1\nLEAKAGE(0.2) 0\nHERALD_LEAKAGE_EVENT 0 1 !0\nM 0\nRL 1\nMR 0\nREPEAT 2 {\n    LEAKAGE(0.05) 0 1\n    HERALD_LEAKAGE_EVENT 0 1\n}").unwrap();
        let m = herald_leakage(&c.instrs);
        let mut got: Vec<(u64, f32)> = m.into_iter().collect();
        got.sort_by_key(|x| x.0);
        let f = |x: f64| x as f32;
        let sum = |a: f32, b: f64| (f64::from(a) + b) as f32;
        // MR measures (Deltakit's pass counts it, and not as a reset); RL resets qubit 1.
        assert_eq!(got, vec![(0, sum(f(0.1), 0.2)), (1, f(0.1)), (2, 0.0), (5, sum(sum(f(0.1), 0.2), 0.05)), (6, f(0.05)), (7, sum(sum(sum(f(0.1), 0.2), 0.05), 0.05)), (8, sum(f(0.05), 0.05))]);
        assert!(herald_leakage(&Circuit::parse("LEAKAGE(0.1) 0\nM 0").unwrap().instrs).is_empty());
    }
}

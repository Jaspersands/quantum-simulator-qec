# Stim Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the engine a general circuit path — Stim-format circuits and detector error models in and out, a DEM built by backward sensitivity analysis, weighted MWPM over any DEM, an independent frame sampler, SD6 noise — and prove it against Stim and PyMatching, live on the site.

**Architecture:** New Rust modules sit beside the old per-code paths and never change them: `circuit.rs` (IR + Stim text), `dem.rs` (error model), `dem_decoder.rs` (matching), `frame_sampler.rs`, `shots.rs`, `memory.rs` (surface-code memory experiments as circuits). PyO3 bindings feed a Python harness that runs Stim and PyMatching; WASM exports feed a new section-10 figure and an SD6 option in sections 07 and 09.

**Tech Stack:** Rust 2021 (no new crates), PyO3 0.22 + maturin in `.venv` (stim 1.16.0, pymatching 2.4.0, numpy), wasm32-unknown-unknown C-ABI build, vanilla ES modules, Node 24 for pure-module tests, the built-in browser pane for page checks.

**Spec:** `docs/superpowers/specs/2026-09-24-stim-foundation-design.md`

## Global Constraints

- Old paths untouched: no existing function changes behaviour or signature, except moving the rotated code's two schedule arrays from function-local `const` to associated `pub const` (a pure refactor, covered by the existing tests).
- No new crates. No new site dependencies; no build step; the page runs from `python3 -m http.server`.
- "Every colour, size, and border in this file comes from a token" — `css/styles.css` header rule. Light only; no dark theme.
- Every failure is explicit: parse errors name the instruction and line; builder errors name the fault; decoder errors are returned, counted and shown, never replaced by a fallback.
- Rust tests: `cargo test --release --no-default-features` (the WASM-side cfg) must pass, and `cargo build --release` (the Python cfg) must compile.
- Every number that reaches the README or the page is produced by a committed tool (`tools/xcheck.py`, `tools/sweep.mjs`, or an ignored `cargo test`), and the command is written next to it.
- Work on branch `feature/stim-foundation`. Commit after every task with the trailer `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. Do not merge to `master` or push: `master` deploys qcompiler.jaspersands.com.

---

## File map

| File | Responsibility |
|---|---|
| `src/circuit.rs` (new) | `Circuit`/`Instr`/`Basis`, Stim text parse and emit, `flattened`, `resolve` |
| `src/dem.rs` (new) | `Dem`/`Mechanism`/`Piece`, `from_circuit` (backward analysis, channel conversion, merging, determinism, decomposition), Stim `.dem` parse/emit, `compare` |
| `src/dem_decoder.rs` (new) | `DemDecoder`: weighted graph from a DEM, Dijkstra + blossom, `Prediction {observables, weight}`, `DecodeError` |
| `src/frame_sampler.rs` (new) | `FrameSampler`: forward Pauli frame over a circuit with true channel semantics |
| `src/shots.rs` (new) | Stim `01` and `b8` shot formats |
| `src/memory.rs` (new) | `NoiseModel`, `CodeKind`, `RoundLayers`, `Patch`, `xzzx_hadamard_pattern`, `memory_circuit`, `generate` |
| `src/fixtures.rs` (new, test-only) | Shared circuit texts for tests |
| `src/equivalence.rs` (new, test-only) | Check 5: old path and new path describe the same circuit |
| `src/py_api.rs` (new, python cfg) | PyO3 functions for the harness |
| `src/wasm_xc.rs` (new, wasm cfg) | Text buffer, `wasm_xc_*` exports, SD6 runs |
| `src/surface_code.rs` | `X_ORDER`/`Z_ORDER` become associated consts; `next_shot_rng` becomes `pub(crate)` |
| `src/lib.rs` | module declarations; PyO3 registration; `wasm_run_benchmark` routes noise mode 3 |
| `tools/xcheck.py` (new) | Checks 1–4 against Stim and PyMatching; writes `data/xcheck/` |
| `data/xcheck/` (new) | `reference.json`, Stim's DEMs (`*.dem.txt`) and circuits (`*.stim.txt`) |
| `js/engine.js` | `NOISE.SD6`, text buffer helpers, `xc*` wrappers, `decodeErrors` |
| `js/worker.js` | `xcheck` and `xctiming` ops |
| `js/xcheck-format.js` (new) | Pure formatting for the figure |
| `js/sections/xcheck.js` (new) | Figure 8 |
| `js/sweep-config.js` (new) | Sweep distances, windows, shot counts — shared by section 07 and `tools/sweep.mjs` |
| `js/sections/threshold.js`, `js/sections/bench.js`, `js/main.js` | SD6 option; boot the figure |
| `index.html`, `css/styles.css` | Figure 8; SD6 options; section 10 opening line; prose |
| `tools/site-tests.mjs` | tests for `xcheck-format.js` and `sweep-config.js` |
| `tools/sweep.mjs` (new) | Four-sweep thresholds for the README |
| `README.md` | "Checked against Stim and PyMatching"; SD6 rows; findings |

---

### Task 1: The circuit

**Files:**
- Create: `src/circuit.rs`, `src/fixtures.rs`
- Modify: `src/lib.rs` (module declarations)

**Interfaces:**
- Produces: `circuit::{Basis, Pauli, Instr, Circuit, Resolved}`; `Circuit::parse(&str) -> Result<Circuit, String>`, `Circuit::to_stim(&self) -> String`, `Circuit::flattened(&self) -> Circuit`, `Circuit::resolve(&self) -> Result<Resolved, String>`; `pub(crate) fn split_instruction(line) -> Result<(String, Vec<f64>, Vec<&str>), String>`; `pub(crate) fn fmt_args(&[f64]) -> String`. `Resolved { instrs, num_qubits, num_measurements, detectors: Vec<Vec<usize>>, detector_coords: Vec<Vec<f64>>, observables: Vec<Vec<usize>> }`.
- Fixtures: `fixtures::SAMPLE` (a small circuit using every supported construct), `fixtures::REP3` (distance-3 repetition code, two rounds, circuit noise at 0.01).

- [ ] **Step 1: Write the fixtures and the failing tests**

`src/fixtures.rs`:

```rust
//! Circuit texts shared by the tests of several modules.

/// Every construct the parser supports, in a circuit that is also a valid
/// (deterministic) two-qubit parity check.
pub const SAMPLE: &str = "\
QUBIT_COORDS(0, 0) 0
QUBIT_COORDS(1, 0) 1
QUBIT_COORDS(2, 0) 2
R 0 1 2
X_ERROR(0.001) 0 1 2
TICK
CX 0 1
DEPOLARIZE2(0.001) 0 1
CX 2 1
DEPOLARIZE2(0.001) 2 1
X_ERROR(0.001) 1
MR 1
DETECTOR(1, 0, 0) rec[-1]
REPEAT 2 {
    TICK
    DEPOLARIZE1(0.001) 0 2
    CX 0 1 2 1
    MR(0.001) 1
    SHIFT_COORDS(0, 0, 1)
    DETECTOR(1, 0, 0) rec[-1] rec[-2]
}
M 0 2
DETECTOR(1, 0, 1) rec[-1] rec[-2] rec[-3]
OBSERVABLE_INCLUDE(0) rec[-1]
";

/// A distance-3 repetition code (data 0, 2, 4; ancillas 1, 3), two rounds of
/// circuit noise at 0.01, then a transversal readout. Distance 3 against X
/// errors, so every single fault must be corrected.
pub const REP3: &str = "\
R 0 1 2 3 4
TICK
CX 0 1 2 3
DEPOLARIZE2(0.01) 0 1 2 3
TICK
CX 2 1 4 3
DEPOLARIZE2(0.01) 2 1 4 3
TICK
X_ERROR(0.01) 1 3
MR 1 3
DETECTOR(1, 0, 0) rec[-2]
DETECTOR(3, 0, 0) rec[-1]
TICK
DEPOLARIZE1(0.01) 0 2 4
CX 0 1 2 3
DEPOLARIZE2(0.01) 0 1 2 3
TICK
CX 2 1 4 3
DEPOLARIZE2(0.01) 2 1 4 3
TICK
X_ERROR(0.01) 1 3
MR 1 3
DETECTOR(1, 0, 1) rec[-2] rec[-4]
DETECTOR(3, 0, 1) rec[-1] rec[-3]
TICK
X_ERROR(0.01) 0 2 4
M 0 2 4
DETECTOR(1, 0, 2) rec[-3] rec[-2] rec[-5]
DETECTOR(3, 0, 2) rec[-2] rec[-1] rec[-4]
OBSERVABLE_INCLUDE(0) rec[-1]
";
```

Tests at the bottom of `src/circuit.rs`:

```rust
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
        assert_eq!(r.detector_coords, vec![
            vec![1.0, 0.0, 0.0], vec![1.0, 0.0, 1.0], vec![1.0, 0.0, 2.0], vec![1.0, 0.0, 3.0],
        ]);
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
    fn records_before_the_first_measurement_are_errors() {
        let c = Circuit::parse("M 0\nDETECTOR rec[-2]").unwrap();
        assert!(c.resolve().unwrap_err().contains("rec[-2]"));
    }
}
```

`src/lib.rs`, after `pub mod circuit_model;`:

```rust
pub mod circuit;
#[cfg(test)]
mod fixtures;
```

- [ ] **Step 2: Run the tests to see them fail**

Run: `cargo test --release --no-default-features circuit::`
Expected: compile failure, `circuit.rs` has no items.

- [ ] **Step 3: Implement `src/circuit.rs`**

```rust
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
            | Instr::QubitCoords { qubits, .. } => qubits.clone(),
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
pub struct Resolved {
    pub instrs: Vec<Instr>,
    pub num_qubits: usize,
    pub num_measurements: usize,
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
        let flat = self.flattened().instrs;
        let mut num_qubits = 0usize;
        let mut m = 0usize;
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
        Ok(Resolved { instrs: flat, num_qubits, num_measurements: m, detectors, detector_coords, observables })
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

fn qubit_targets(tokens: &[&str], name: &str) -> Result<Vec<u32>, String> {
    tokens
        .iter()
        .map(|t| t.parse::<u32>().map_err(|_| format!("{name}: bad qubit target '{t}'")))
        .collect()
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
        if args.is_empty() { Ok(()) } else { Err(format!("{name} takes no arguments")) }
    };
    let exactly = |n: usize| -> Result<(), String> {
        if args.len() == n { Ok(()) } else { Err(format!("{name} takes {n} argument(s), got {}", args.len())) }
    };
    let prob = |i: usize| -> Result<f64, String> {
        let p = args[i];
        if (0.0..=1.0).contains(&p) { Ok(p) } else { Err(format!("{name}: probability {p} is outside [0, 1]")) }
    };
    Ok(match name.as_str() {
        "R" | "RZ" => { none()?; Instr::Reset { basis: Basis::Z, qubits: qubit_targets(&t, &name)? } }
        "RX" => { none()?; Instr::Reset { basis: Basis::X, qubits: qubit_targets(&t, &name)? } }
        "H" => { none()?; Instr::H(qubit_targets(&t, &name)?) }
        "CX" | "CNOT" | "ZCX" => { none()?; Instr::Cx(pair_targets(&t, &name)?) }
        "CZ" | "ZCZ" => { none()?; Instr::Cz(pair_targets(&t, &name)?) }
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
            let pauli = match name.as_bytes()[0] { b'X' => 1, b'Z' => 2, _ => 3 };
            Instr::PauliError { pauli, p: prob(0)?, qubits: qubit_targets(&t, &name)? }
        }
        "DEPOLARIZE1" => { exactly(1)?; Instr::Depolarize1 { p: prob(0)?, qubits: qubit_targets(&t, &name)? } }
        "DEPOLARIZE2" => { exactly(1)?; Instr::Depolarize2 { p: prob(0)?, pairs: pair_targets(&t, &name)? } }
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
    if args.is_empty() { name.to_string() } else { format!("{name}({})", fmt_args(args)) }
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
                let name = match pauli { 1 => "X_ERROR", 2 => "Z_ERROR", _ => "Y_ERROR" };
                format!("{} {}", with_args(name, &[*p]), join_q(qubits))
            }
            Instr::Depolarize1 { p, qubits } => format!("{} {}", with_args("DEPOLARIZE1", &[*p]), join_q(qubits)),
            Instr::Depolarize2 { p, pairs } => format!("{} {}", with_args("DEPOLARIZE2", &[*p]), join_pairs(pairs)),
            Instr::PauliChannel1 { px, py, pz, qubits } => {
                format!("{} {}", with_args("PAULI_CHANNEL_1", &[*px, *py, *pz]), join_q(qubits))
            }
            Instr::Detector { coords, recs } => format!("{} {}", with_args("DETECTOR", coords), join_recs(recs)),
            Instr::Observable { index, recs } => format!("OBSERVABLE_INCLUDE({index}) {}", join_recs(recs)),
            Instr::QubitCoords { coords, qubits } => format!("{} {}", with_args("QUBIT_COORDS", coords), join_q(qubits)),
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
```

- [ ] **Step 4: Run the tests**

Run: `cargo test --release --no-default-features circuit::`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add src/circuit.rs src/fixtures.rs src/lib.rs
git commit -m "feat(circuit): read and write the part of Stim's circuit language the engine speaks"
```

---

### Task 2: The detector error model

**Files:**
- Create: `src/dem.rs`
- Modify: `src/lib.rs` (`pub mod dem;`)

**Interfaces:**
- Consumes: `circuit::{Circuit, Instr, Basis, Resolved, split_instruction, fmt_args}`.
- Produces:
  - `Piece { detectors: Vec<u32>, observables: u64 }`, `Mechanism { p: f64, detectors: Vec<u32>, observables: u64, pieces: Vec<Piece> }`, `Dem { num_detectors, num_observables, detector_coords: Vec<Vec<f64>>, mechanisms: Vec<Mechanism> }`.
  - `Dem::from_circuit(&Circuit) -> Result<Dem, String>`; `Dem::parse(&str) -> Result<Dem, String>`; `Dem::to_stim(&self, with_pieces: bool) -> String`.
  - `pub fn depolarize1_component(p) -> f64`, `depolarize2_component(p) -> f64`, `pauli_channel_1_independent(px, py, pz) -> Result<(f64, f64, f64), String>` (returns qx, qy, qz), `xor_prob(a, b) -> f64`.
  - `Comparison { ours, theirs, missing, extra, differing: usize, max_rel: f64 }`; `pub fn compare(ours: &Dem, theirs: &Dem, tol: f64) -> Comparison`.

- [ ] **Step 1: Write the failing tests**

At the bottom of `src/dem.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixtures::{REP3, SAMPLE};

    fn dem(text: &str) -> Dem {
        Dem::from_circuit(&Circuit::parse(text).unwrap()).unwrap()
    }

    fn p_of(d: &Dem, dets: &[u32], obs: u64) -> f64 {
        d.mechanisms.iter().find(|m| m.detectors == dets && m.observables == obs).map(|m| m.p).unwrap()
    }

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() <= 1e-12 * a.abs().max(b.abs())
    }

    /// Numbers from Stim 1.16, `circuit.detector_error_model()`.
    #[test]
    fn depolarizing_channels_match_stim() {
        let d = dem("R 0\nDEPOLARIZE1(0.01) 0\nM 0\nDETECTOR rec[-1]");
        assert_eq!(d.mechanisms.len(), 1);
        assert!(close(d.mechanisms[0].p, 0.006666666666666613), "{}", d.mechanisms[0].p);

        let d = dem("R 0 1\nDEPOLARIZE2(0.01) 0 1\nM 0 1\nDETECTOR rec[-1]\nDETECTOR rec[-2]");
        assert_eq!(d.mechanisms.len(), 3);
        for m in &d.mechanisms {
            assert!(close(m.p, 0.002673815958446298), "{}", m.p);
        }
    }

    #[test]
    fn pauli_channel_and_flips_match_stim() {
        let d = dem("R 0\nPAULI_CHANNEL_1(0.01, 0.02, 0.03) 0\nM 0\nDETECTOR rec[-1]\n\
                     RX 1\nPAULI_CHANNEL_1(0.01, 0.02, 0.03) 1\nMX 1\nDETECTOR rec[-1]");
        assert!(close(p_of(&d, &[0], 0), 0.03) && close(p_of(&d, &[1], 0), 0.05));

        let d = dem("R 0\nX_ERROR(0.01) 0\nM(0.02) 0\nDETECTOR rec[-1]\nRX 1\nY_ERROR(0.1) 1\nMRX 1\nDETECTOR rec[-1]");
        assert!(close(p_of(&d, &[0], 0), 0.0296) && close(p_of(&d, &[1], 0), 0.1));
    }

    #[test]
    fn gates_propagate_errors_backwards_correctly() {
        // X on a control before CX spreads to the target.
        let d = dem("R 0 1\nX_ERROR(0.1) 0\nCX 0 1\nM 0 1\nDETECTOR rec[-2]\nDETECTOR rec[-1]");
        assert!(close(p_of(&d, &[0, 1], 0), 0.1));
        // Z on a target before CX spreads to the control.
        let d = dem("RX 0 1\nZ_ERROR(0.2) 1\nCX 0 1\nMX 0 1\nDETECTOR rec[-2]\nDETECTOR rec[-1]");
        assert!(close(p_of(&d, &[0, 1], 0), 0.2));
        // X before CZ becomes X on its own qubit and Z on the other.
        let d = dem("R 0\nRX 1\nX_ERROR(0.1) 0\nCZ 0 1\nM 0\nMX 1\nDETECTOR rec[-2]\nDETECTOR rec[-1]");
        assert!(close(p_of(&d, &[0, 1], 0), 0.1));
        // H swaps: X before H flips an X-basis readout, Z before H does not.
        let d = dem("R 0\nX_ERROR(0.1) 0\nZ_ERROR(0.2) 0\nH 0\nMX 0\nDETECTOR rec[-1]");
        assert_eq!(d.mechanisms.len(), 1);
        assert!(close(p_of(&d, &[0], 0), 0.1));
    }

    #[test]
    fn nondeterministic_detectors_are_errors() {
        let e = Dem::from_circuit(&Circuit::parse("RX 0\nM 0\nDETECTOR rec[-1]").unwrap()).unwrap_err();
        assert!(e.contains("not deterministic"), "{e}");
        let e = Dem::from_circuit(&Circuit::parse("M 0\nMX 0\nDETECTOR rec[-1]").unwrap()).unwrap_err();
        assert!(e.contains("not deterministic"), "{e}");
    }

    #[test]
    fn undetectable_logical_errors_are_errors() {
        let e = Dem::from_circuit(&Circuit::parse("R 0\nX_ERROR(0.1) 0\nM 0\nOBSERVABLE_INCLUDE(0) rec[-1]").unwrap())
            .unwrap_err();
        assert!(e.contains("undetectable"), "{e}");
    }

    #[test]
    fn sample_and_rep3_build() {
        let d = dem(SAMPLE);
        assert_eq!(d.num_detectors, 4);
        let d = dem(REP3);
        assert_eq!((d.num_detectors, d.num_observables), (6, 1));
        assert!(d.mechanisms.iter().any(|m| m.observables == 1));
    }

    #[test]
    fn stim_text_round_trips() {
        let d = dem(REP3);
        let back = Dem::parse(&d.to_stim(false)).unwrap();
        let c = compare(&d, &back, 1e-12);
        assert_eq!((c.missing, c.extra, c.differing), (0, 0, 0));
        assert_eq!(back.num_detectors, d.num_detectors);
        assert_eq!(back.detector_coords, d.detector_coords);
    }

    #[test]
    fn parse_reads_pieces_repeats_and_shifts() {
        let d = Dem::parse("error(0.1) D0 D1 ^ D2\nrepeat 2 {\n  error(0.2) D0\n  shift_detectors 1\n}\ndetector(1, 2) D0\n")
            .unwrap();
        assert_eq!(d.mechanisms[0].detectors, vec![0, 1, 2]);
        assert_eq!(d.mechanisms[0].pieces.len(), 2);
        assert_eq!(d.mechanisms[1].detectors, vec![0]);
        assert_eq!(d.mechanisms[2].detectors, vec![1]);
        assert_eq!(d.detector_coords[2], vec![1.0, 2.0]);
        assert_eq!(d.num_detectors, 3);
    }

    #[test]
    fn comparison_counts_differences() {
        let a = Dem::parse("error(0.1) D0\nerror(0.2) D1").unwrap();
        let b = Dem::parse("error(0.1) D0\nerror(0.25) D1\nerror(0.1) D0 D1").unwrap();
        let c = compare(&a, &b, 1e-9);
        assert_eq!((c.ours, c.theirs, c.missing, c.extra, c.differing), (2, 3, 1, 0, 1));
        assert!((c.max_rel - 0.2).abs() < 1e-12);
    }
}
```

- [ ] **Step 2: Run the tests to see them fail**

Run: `cargo test --release --no-default-features dem::`
Expected: compile failure.

- [ ] **Step 3: Implement `src/dem.rs`**

```rust
//! Detector error models: which detectors and logical observables each
//! elementary fault flips, and how likely it is.
//!
//! BUILT BACKWARDS
//! ---------------
//! The obvious construction pushes every fault forward through the rest of the
//! circuit, which costs faults × instructions. This walks the circuit once, in
//! reverse, carrying for every qubit the set of detectors and observables an X
//! error there would flip (`sx`) and the set a Z error would flip (`sz`). At a
//! noise instruction the symptom of every Pauli it can apply is then read off
//! directly. That is how Stim's error analyzer works, and it is what lets a
//! d = 7 model be derived in the browser while the reader watches.
//!
//! Going backwards through a gate conjugates the pair: H swaps them; CX sends
//! `sx[c] ^= sx[t]` and `sz[t] ^= sz[c]`; CZ sends `sx[a] ^= sz[b]` and
//! `sx[b] ^= sz[a]`. A Z-basis measurement adds what reads its record to `sx`;
//! a reset clears both — an error before a reset is erased by it.
//!
//! The same walk checks determinism for free. Reaching a Z-basis reset with
//! `sz` non-empty means some detector anticommutes with the state the reset
//! prepares, so its value is a coin flip. That is an error, not a warning.

use std::collections::{HashMap, HashSet};
use std::fmt::Write as _;

use crate::circuit::{fmt_args, split_instruction, Basis, Circuit, Instr};

#[derive(Clone, Debug, PartialEq, Default)]
pub struct Piece {
    pub detectors: Vec<u32>,
    pub observables: u64,
}

#[derive(Clone, Debug)]
pub struct Mechanism {
    pub p: f64,
    /// Sorted.
    pub detectors: Vec<u32>,
    pub observables: u64,
    /// Graph-like components (at most two detectors each) that XOR back to the
    /// mechanism. Empty for a hyperedge nobody decomposed.
    pub pieces: Vec<Piece>,
}

#[derive(Clone, Debug, Default)]
pub struct Dem {
    pub num_detectors: usize,
    pub num_observables: usize,
    pub detector_coords: Vec<Vec<f64>>,
    pub mechanisms: Vec<Mechanism>,
}

/* -- Channel conversion ---------------------------------------------------- */

/// Independent X, Y and Z probability equivalent to `DEPOLARIZE1(p)`.
///
/// A Pauli channel is fixed by how much it shrinks each Pauli. Depolarizing
/// shrinks X, Y and Z alike by 1 − 4p/3; three independent channels of
/// probability q shrink each by (1 − 2q)², since each Pauli anticommutes with
/// two of the three. Equating the two is exact, which is why the model loses
/// nothing by treating the components as independent.
pub fn depolarize1_component(p: f64) -> f64 {
    0.5 - 0.5 * (1.0 - 4.0 * p / 3.0).sqrt()
}

/// The same for `DEPOLARIZE2(p)`: fifteen components, each non-identity Pauli
/// anticommuting with eight of them, so (1 − 2q)⁸ = 1 − 16p/15.
pub fn depolarize2_component(p: f64) -> f64 {
    0.5 - 0.5 * (1.0 - 16.0 * p / 15.0).powf(0.125)
}

/// Independent (qx, qy, qz) equivalent to `PAULI_CHANNEL_1(px, py, pz)`.
pub fn pauli_channel_1_independent(px: f64, py: f64, pz: f64) -> Result<(f64, f64, f64), String> {
    let lx = 1.0 - 2.0 * (py + pz);
    let ly = 1.0 - 2.0 * (px + pz);
    let lz = 1.0 - 2.0 * (px + py);
    if lx <= 0.0 || ly <= 0.0 || lz <= 0.0 {
        return Err(format!(
            "PAULI_CHANNEL_1({px}, {py}, {pz}) has no equivalent set of independent errors"
        ));
    }
    let q = |a: f64| 0.5 * (1.0 - a);
    Ok((q((ly * lz / lx).sqrt()), q((lx * lz / ly).sqrt()), q((lx * ly / lz).sqrt())))
}

/// Probability that exactly one of two independent events happens.
pub fn xor_prob(a: f64, b: f64) -> f64 {
    a * (1.0 - b) + b * (1.0 - a)
}

/* -- Symptom bitsets ----------------------------------------------------------- */

/// Detectors occupy bits 0..nd; observable k sits at bit nd + k.
#[derive(Clone, Copy)]
struct Space {
    nd: usize,
    words: usize,
}

impl Space {
    fn new(nd: usize, no: usize) -> Self {
        Space { nd, words: (nd + no).div_ceil(64).max(1) }
    }
    fn zero(&self) -> Vec<u64> {
        vec![0; self.words]
    }
    fn split(&self, s: &[u64]) -> (Vec<u32>, u64) {
        let mut dets = Vec::new();
        let mut obs = 0u64;
        for (w, &word) in s.iter().enumerate() {
            let mut bits = word;
            while bits != 0 {
                let b = bits.trailing_zeros() as usize;
                bits &= bits - 1;
                let i = w * 64 + b;
                if i < self.nd {
                    dets.push(i as u32);
                } else {
                    obs |= 1u64 << (i - self.nd);
                }
            }
        }
        (dets, obs)
    }
}

fn toggle(s: &mut [u64], i: usize) {
    s[i / 64] ^= 1u64 << (i % 64);
}

fn xor_into(a: &mut [u64], b: &[u64]) {
    for (x, y) in a.iter_mut().zip(b) {
        *x ^= y;
    }
}

fn is_zero(a: &[u64]) -> bool {
    a.iter().all(|&w| w == 0)
}

/* -- The builder ----------------------------------------------------------------- */

/// One single-qubit, single-type part of a fault. Kind 1 comes from an X
/// component, 2 from a Z component, 0 from a classical readout flip.
#[derive(Clone, PartialEq)]
struct Atom {
    sym: Vec<u64>,
    kind: u8,
}

/// Where a fault came from, for error messages.
#[derive(Clone, Copy)]
struct Origin {
    instr: usize,
    name: &'static str,
    a: u32,
    b: Option<u32>,
    pauli: (u8, u8),
}

fn pauli_name(p: u8) -> &'static str {
    match p { 0 => "I", 1 => "X", 2 => "Z", _ => "Y" }
}

impl Origin {
    fn describe(&self) -> String {
        match self.b {
            Some(b) => format!(
                "{} {}⊗{} on qubits {}, {} (instruction {})",
                self.name, pauli_name(self.pauli.0), pauli_name(self.pauli.1), self.a, b, self.instr
            ),
            None => format!("{} {} on qubit {} (instruction {})", self.name, pauli_name(self.pauli.0), self.a, self.instr),
        }
    }
}

struct Entry {
    sym: Vec<u64>,
    p: f64,
    /// Up to four distinct ways the merged faults split into atoms; any one that
    /// decomposes will do.
    atom_options: Vec<Vec<Atom>>,
    origin: Origin,
}

struct Builder {
    space: Space,
    index: HashMap<Vec<u64>, usize>,
    entries: Vec<Entry>,
}

impl Builder {
    fn add(&mut self, p: f64, atoms: Vec<Atom>, origin: Origin) {
        if p <= 0.0 {
            return;
        }
        let mut sym = self.space.zero();
        for a in &atoms {
            xor_into(&mut sym, &a.sym);
        }
        if is_zero(&sym) {
            return;
        }
        match self.index.get(&sym) {
            Some(&i) => {
                let e = &mut self.entries[i];
                e.p = xor_prob(e.p, p);
                if e.atom_options.len() < 4 && !e.atom_options.contains(&atoms) {
                    e.atom_options.push(atoms);
                }
            }
            None => {
                self.index.insert(sym.clone(), self.entries.len());
                self.entries.push(Entry { sym, p, atom_options: vec![atoms], origin });
            }
        }
    }
}

fn atoms_for(pauli: u8, q: usize, sx: &[Vec<u64>], sz: &[Vec<u64>]) -> Vec<Atom> {
    let mut out = Vec::new();
    if pauli & 1 != 0 {
        out.push(Atom { sym: sx[q].clone(), kind: 1 });
    }
    if pauli & 2 != 0 {
        out.push(Atom { sym: sz[q].clone(), kind: 2 });
    }
    out
}

fn nondeterministic(space: &Space, coords: &[Vec<f64>], sym: &[u64], q: u32, what: &str) -> String {
    let (dets, obs) = space.split(sym);
    let subject = match dets.first() {
        Some(&d) => {
            let c = coords.get(d as usize).map(|c| fmt_args(c)).unwrap_or_default();
            format!("detector D{d} ({c})")
        }
        None => format!("observable L{}", obs.trailing_zeros()),
    };
    format!("{subject} is not deterministic: it depends on the random outcome of {what} on qubit {q}")
}

impl Dem {
    pub fn from_circuit(circuit: &Circuit) -> Result<Dem, String> {
        let res = circuit.resolve()?;
        let nd = res.detectors.len();
        let no = res.observables.len();
        let space = Space::new(nd, no);

        // Which detectors and observables read each measurement record.
        let mut rec_sym = vec![space.zero(); res.num_measurements];
        for (d, recs) in res.detectors.iter().enumerate() {
            for &m in recs {
                toggle(&mut rec_sym[m], d);
            }
        }
        for (o, recs) in res.observables.iter().enumerate() {
            for &m in recs {
                toggle(&mut rec_sym[m], nd + o);
            }
        }

        let nq = res.num_qubits;
        let mut sx = vec![space.zero(); nq];
        let mut sz = vec![space.zero(); nq];
        let mut b = Builder { space, index: HashMap::new(), entries: Vec::new() };
        let mut m = res.num_measurements;
        let coords = &res.detector_coords;

        let check_reset = |sx: &[Vec<u64>], sz: &[Vec<u64>], q: usize, basis: Basis| -> Result<(), String> {
            let sensitive = match basis { Basis::Z => &sz[q], Basis::X => &sx[q] };
            if is_zero(sensitive) {
                Ok(())
            } else {
                let what = if basis == Basis::Z { "a Z-basis reset" } else { "an X-basis reset" };
                Err(nondeterministic(&space, coords, sensitive, q as u32, what))
            }
        };

        for (idx, ins) in res.instrs.iter().enumerate().rev() {
            match ins {
                Instr::Reset { basis, qubits } => {
                    for &q in qubits.iter().rev() {
                        let q = q as usize;
                        check_reset(&sx, &sz, q, *basis)?;
                        sx[q].fill(0);
                        sz[q].fill(0);
                    }
                }
                Instr::Measure { basis, reset, flip, qubits } => {
                    for &q in qubits.iter().rev() {
                        let qi = q as usize;
                        m -= 1;
                        if *reset {
                            check_reset(&sx, &sz, qi, *basis)?;
                            sx[qi].fill(0);
                            sz[qi].fill(0);
                        }
                        match basis {
                            Basis::Z => xor_into(&mut sx[qi], &rec_sym[m]),
                            Basis::X => xor_into(&mut sz[qi], &rec_sym[m]),
                        }
                        let origin = Origin { instr: idx, name: "measurement flip", a: q, b: None, pauli: (1, 0) };
                        b.add(*flip, vec![Atom { sym: rec_sym[m].clone(), kind: 0 }], origin);
                    }
                }
                Instr::H(qubits) => {
                    for &q in qubits.iter().rev() {
                        std::mem::swap(&mut sx[q as usize], &mut sz[q as usize]);
                    }
                }
                Instr::Cx(pairs) => {
                    for &(c, t) in pairs.iter().rev() {
                        let (c, t) = (c as usize, t as usize);
                        let from_t = sx[t].clone();
                        xor_into(&mut sx[c], &from_t);
                        let from_c = sz[c].clone();
                        xor_into(&mut sz[t], &from_c);
                    }
                }
                Instr::Cz(pairs) => {
                    for &(a, bq) in pairs.iter().rev() {
                        let (a, bq) = (a as usize, bq as usize);
                        xor_into(&mut sx[a], &sz[bq]);
                        xor_into(&mut sx[bq], &sz[a]);
                    }
                }
                Instr::PauliError { pauli, p, qubits } => {
                    for &q in qubits {
                        let origin = Origin { instr: idx, name: "Pauli error", a: q, b: None, pauli: (*pauli, 0) };
                        b.add(*p, atoms_for(*pauli, q as usize, &sx, &sz), origin);
                    }
                }
                Instr::Depolarize1 { p, qubits } => {
                    if *p > 0.75 {
                        return Err(format!("DEPOLARIZE1({p}) exceeds 3/4 (instruction {idx})"));
                    }
                    let q1 = depolarize1_component(*p);
                    for &q in qubits {
                        for pauli in [1u8, 3, 2] {
                            let origin = Origin { instr: idx, name: "DEPOLARIZE1", a: q, b: None, pauli: (pauli, 0) };
                            b.add(q1, atoms_for(pauli, q as usize, &sx, &sz), origin);
                        }
                    }
                }
                Instr::PauliChannel1 { px, py, pz, qubits } => {
                    let (qx, qy, qz) = pauli_channel_1_independent(*px, *py, *pz)
                        .map_err(|e| format!("{e} (instruction {idx})"))?;
                    for &q in qubits {
                        for (pauli, prob) in [(1u8, qx), (3, qy), (2, qz)] {
                            let origin = Origin { instr: idx, name: "PAULI_CHANNEL_1", a: q, b: None, pauli: (pauli, 0) };
                            b.add(prob, atoms_for(pauli, q as usize, &sx, &sz), origin);
                        }
                    }
                }
                Instr::Depolarize2 { p, pairs } => {
                    if *p > 15.0 / 16.0 {
                        return Err(format!("DEPOLARIZE2({p}) exceeds 15/16 (instruction {idx})"));
                    }
                    let q2 = depolarize2_component(*p);
                    for &(qa, qb) in pairs {
                        for pa in 0..4u8 {
                            for pb in 0..4u8 {
                                if pa == 0 && pb == 0 {
                                    continue;
                                }
                                let mut atoms = atoms_for(pa, qa as usize, &sx, &sz);
                                atoms.extend(atoms_for(pb, qb as usize, &sx, &sz));
                                let origin = Origin { instr: idx, name: "DEPOLARIZE2", a: qa, b: Some(qb), pauli: (pa, pb) };
                                b.add(q2, atoms, origin);
                            }
                        }
                    }
                }
                Instr::Detector { .. }
                | Instr::Observable { .. }
                | Instr::QubitCoords { .. }
                | Instr::ShiftCoords(_)
                | Instr::Tick
                | Instr::Repeat { .. } => {}
            }
        }

        // Every qubit starts in |0>, which is a Z-basis reset at time zero.
        for q in 0..nq {
            check_reset(&sx, &sz, q, Basis::Z).map_err(|e| e.replace("a Z-basis reset", "the initial |0>"))?;
        }

        // Graph-like mechanisms, indexed by detector, for decomposition step 3.
        let mut graphlike: HashSet<Vec<u64>> = HashSet::new();
        let mut by_det: HashMap<u32, Vec<Vec<u64>>> = HashMap::new();
        for e in &b.entries {
            let (dets, obs) = space.split(&e.sym);
            if dets.is_empty() {
                return Err(format!(
                    "{} flips observables {obs:#b} while firing no detector: an undetectable logical error",
                    e.origin.describe()
                ));
            }
            if dets.len() <= 2 {
                graphlike.insert(e.sym.clone());
                for d in dets {
                    by_det.entry(d).or_default().push(e.sym.clone());
                }
            }
        }

        let mut mechanisms = Vec::with_capacity(b.entries.len());
        for e in &b.entries {
            let (detectors, observables) = space.split(&e.sym);
            let pieces = decompose(&space, e, &graphlike, &by_det).ok_or_else(|| {
                format!(
                    "cannot split {} into graph-like pieces: it fires detectors {:?}",
                    e.origin.describe(),
                    detectors
                )
            })?;
            mechanisms.push(Mechanism { p: e.p, detectors, observables, pieces });
        }
        mechanisms.sort_by(|a, b| a.detectors.cmp(&b.detectors).then(a.observables.cmp(&b.observables)));

        Ok(Dem { num_detectors: nd, num_observables: no, detector_coords: res.detector_coords, mechanisms })
    }
}

/* -- Decomposition ------------------------------------------------------------ */

/// Parts to pieces, or None if any part is too wide or is a bare logical.
fn to_pieces(space: &Space, parts: &[Vec<u64>]) -> Option<Vec<Piece>> {
    let mut out = Vec::new();
    for part in parts {
        if is_zero(part) {
            continue;
        }
        let (detectors, observables) = space.split(part);
        if detectors.is_empty() || detectors.len() > 2 {
            return None;
        }
        out.push(Piece { detectors, observables });
    }
    Some(out)
}

/// Two existing graph-like mechanisms whose symptoms XOR to `s`.
fn split_existing(
    space: &Space,
    s: &[u64],
    graphlike: &HashSet<Vec<u64>>,
    by_det: &HashMap<u32, Vec<Vec<u64>>>,
) -> Option<(Vec<u64>, Vec<u64>)> {
    let (dets, _) = space.split(s);
    if dets.is_empty() || dets.len() > 4 {
        return None;
    }
    for g in by_det.get(&dets[0])? {
        let mut h = s.to_vec();
        xor_into(&mut h, g);
        if graphlike.contains(&h) {
            return Some((g.clone(), h));
        }
    }
    None
}

/// Split a mechanism into graph-like pieces, most natural split first:
/// 1. its X part and its Z part (propagation is linear, and a Y is X·Z);
/// 2. atom by atom (a two-qubit Pauli gives up to four single-type atoms);
/// 3. any part still too wide, into two graph-like mechanisms that exist.
fn decompose(
    space: &Space,
    e: &Entry,
    graphlike: &HashSet<Vec<u64>>,
    by_det: &HashMap<u32, Vec<Vec<u64>>>,
) -> Option<Vec<Piece>> {
    let (dets, _) = space.split(&e.sym);
    if dets.len() <= 2 {
        return to_pieces(space, std::slice::from_ref(&e.sym));
    }
    for atoms in &e.atom_options {
        let mut xpart = space.zero();
        let mut zpart = space.zero();
        let mut parts = Vec::new();
        for a in atoms {
            match a.kind {
                1 => xor_into(&mut xpart, &a.sym),
                2 => xor_into(&mut zpart, &a.sym),
                _ => parts.push(a.sym.clone()),
            }
        }
        parts.push(xpart);
        parts.push(zpart);
        if let Some(p) = to_pieces(space, &parts) {
            return Some(p);
        }

        let mut singles: Vec<Vec<u64>> = Vec::new();
        for a in atoms {
            match singles.iter().position(|s| *s == a.sym) {
                Some(i) => {
                    singles.swap_remove(i);
                }
                None => singles.push(a.sym.clone()),
            }
        }
        if let Some(p) = to_pieces(space, &singles) {
            return Some(p);
        }

        let mut out = Vec::new();
        let mut ok = true;
        for s in &singles {
            if is_zero(s) {
                continue;
            }
            if let Some(mut p) = to_pieces(space, std::slice::from_ref(s)) {
                out.append(&mut p);
                continue;
            }
            match split_existing(space, s, graphlike, by_det).and_then(|(g, h)| to_pieces(space, &[g, h])) {
                Some(mut p) => out.append(&mut p),
                None => {
                    ok = false;
                    break;
                }
            }
        }
        if ok {
            return Some(out);
        }
    }
    split_existing(space, &e.sym, graphlike, by_det).and_then(|(g, h)| to_pieces(space, &[g, h]))
}

/* -- Stim's .dem text --------------------------------------------------------- */

fn push_targets(s: &mut String, dets: &[u32], obs: u64) {
    for d in dets {
        let _ = write!(s, " D{d}");
    }
    for k in 0..64 {
        if (obs >> k) & 1 == 1 {
            let _ = write!(s, " L{k}");
        }
    }
}

impl Dem {
    pub fn to_stim(&self, with_pieces: bool) -> String {
        let mut s = String::new();
        for (i, c) in self.detector_coords.iter().enumerate() {
            if c.is_empty() {
                let _ = writeln!(s, "detector D{i}");
            } else {
                let _ = writeln!(s, "detector({}) D{i}", fmt_args(c));
            }
        }
        for m in &self.mechanisms {
            let _ = write!(s, "error({})", m.p);
            if with_pieces && !m.pieces.is_empty() {
                for (k, piece) in m.pieces.iter().enumerate() {
                    if k > 0 {
                        s.push_str(" ^");
                    }
                    push_targets(&mut s, &piece.detectors, piece.observables);
                }
            } else {
                push_targets(&mut s, &m.detectors, m.observables);
            }
            s.push('\n');
        }
        s
    }

    pub fn parse(text: &str) -> Result<Dem, String> {
        let lines: Vec<&str> = text.lines().collect();
        let mut dem = Dem::default();
        let mut pos = 0usize;
        let mut offset = 0u64;
        let mut shift = Vec::new();
        parse_dem_block(&lines, &mut pos, false, &mut dem, &mut offset, &mut shift)?;
        dem.detector_coords.resize(dem.num_detectors, Vec::new());
        Ok(dem)
    }
}

fn toggle_det(v: &mut Vec<u32>, d: u32) {
    match v.iter().position(|&x| x == d) {
        Some(i) => {
            v.swap_remove(i);
        }
        None => v.push(d),
    }
}

fn parse_dem_block(
    lines: &[&str],
    pos: &mut usize,
    nested: bool,
    dem: &mut Dem,
    offset: &mut u64,
    shift: &mut Vec<f64>,
) -> Result<(), String> {
    while *pos < lines.len() {
        let lineno = *pos + 1;
        let line = lines[*pos].split('#').next().unwrap_or("").trim();
        *pos += 1;
        if line.is_empty() {
            continue;
        }
        if line == "}" {
            if nested {
                return Ok(());
            }
            return Err(format!("line {lineno}: unmatched '}}'"));
        }
        if let Some(rest) = line.strip_suffix('{') {
            let mut parts = rest.split_whitespace();
            if !parts.next().unwrap_or("").eq_ignore_ascii_case("repeat") {
                return Err(format!("line {lineno}: only repeat opens a block"));
            }
            let count: u64 = parts
                .next()
                .and_then(|s| s.parse().ok())
                .ok_or_else(|| format!("line {lineno}: repeat needs a count"))?;
            let start = *pos;
            for _ in 0..count {
                *pos = start;
                parse_dem_block(lines, pos, true, dem, offset, shift)?;
            }
            if count == 0 {
                // Skip the body without applying it.
                let mut skip = Dem::default();
                let (mut o, mut s) = (0u64, Vec::new());
                parse_dem_block(lines, pos, true, &mut skip, &mut o, &mut s)?;
            }
            continue;
        }
        let (name, args, tokens) = split_instruction(line).map_err(|e| format!("line {lineno}: {e}"))?;
        let bad = |t: &str| format!("line {lineno}: bad target '{t}'");
        match name.as_str() {
            "ERROR" => {
                if args.len() != 1 || !(0.0..=1.0).contains(&args[0]) {
                    return Err(format!("line {lineno}: error takes one probability"));
                }
                let mut pieces = vec![Piece::default()];
                for t in &tokens {
                    if *t == "^" {
                        pieces.push(Piece::default());
                    } else if let Some(d) = t.strip_prefix('D') {
                        let d = d.parse::<u64>().map_err(|_| bad(t))? + *offset;
                        dem.num_detectors = dem.num_detectors.max(d as usize + 1);
                        toggle_det(&mut pieces.last_mut().unwrap().detectors, d as u32);
                    } else if let Some(l) = t.strip_prefix('L') {
                        let l = l.parse::<u32>().map_err(|_| bad(t))?;
                        if l >= 64 {
                            return Err(format!("line {lineno}: at most 64 observables are supported"));
                        }
                        dem.num_observables = dem.num_observables.max(l as usize + 1);
                        pieces.last_mut().unwrap().observables ^= 1u64 << l;
                    } else {
                        return Err(bad(t));
                    }
                }
                let mut detectors = Vec::new();
                let mut observables = 0u64;
                for piece in &mut pieces {
                    piece.detectors.sort_unstable();
                    for &d in &piece.detectors {
                        toggle_det(&mut detectors, d);
                    }
                    observables ^= piece.observables;
                }
                detectors.sort_unstable();
                let pieces = if pieces.len() > 1 {
                    pieces.into_iter().filter(|p| !p.detectors.is_empty() || p.observables != 0).collect()
                } else if !detectors.is_empty() && detectors.len() <= 2 {
                    vec![Piece { detectors: detectors.clone(), observables }]
                } else {
                    Vec::new()
                };
                dem.mechanisms.push(Mechanism { p: args[0], detectors, observables, pieces });
            }
            "DETECTOR" => {
                for t in &tokens {
                    let d = t.strip_prefix('D').and_then(|d| d.parse::<u64>().ok()).ok_or_else(|| bad(t))? + *offset;
                    let d = d as usize;
                    dem.num_detectors = dem.num_detectors.max(d + 1);
                    if dem.detector_coords.len() <= d {
                        dem.detector_coords.resize(d + 1, Vec::new());
                    }
                    dem.detector_coords[d] =
                        args.iter().enumerate().map(|(i, v)| v + shift.get(i).copied().unwrap_or(0.0)).collect();
                }
            }
            "LOGICAL_OBSERVABLE" => {
                for t in &tokens {
                    let l = t.strip_prefix('L').and_then(|l| l.parse::<usize>().ok()).ok_or_else(|| bad(t))?;
                    dem.num_observables = dem.num_observables.max(l + 1);
                }
            }
            "SHIFT_DETECTORS" => {
                if shift.len() < args.len() {
                    shift.resize(args.len(), 0.0);
                }
                for (a, b) in shift.iter_mut().zip(&args) {
                    *a += b;
                }
                for t in &tokens {
                    *offset += t.parse::<u64>().map_err(|_| bad(t))?;
                }
            }
            _ => return Err(format!("line {lineno}: unsupported instruction '{name}'")),
        }
    }
    if nested {
        return Err("unterminated repeat block".into());
    }
    Ok(())
}

/* -- Comparison ---------------------------------------------------------------- */

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Comparison {
    pub ours: usize,
    pub theirs: usize,
    /// In theirs, not in ours.
    pub missing: usize,
    /// In ours, not in theirs.
    pub extra: usize,
    /// In both, with relative probability difference above the tolerance.
    pub differing: usize,
    pub max_rel: f64,
}

fn merged(d: &Dem) -> HashMap<(Vec<u32>, u64), f64> {
    let mut out: HashMap<(Vec<u32>, u64), f64> = HashMap::new();
    for m in &d.mechanisms {
        let e = out.entry((m.detectors.clone(), m.observables)).or_insert(0.0);
        *e = xor_prob(*e, m.p);
    }
    out
}

/// Mechanism-by-mechanism comparison, after merging identical symptoms on each side.
pub fn compare(ours: &Dem, theirs: &Dem, tol: f64) -> Comparison {
    let a = merged(ours);
    let b = merged(theirs);
    let mut c = Comparison { ours: a.len(), theirs: b.len(), missing: 0, extra: 0, differing: 0, max_rel: 0.0 };
    for (k, &pa) in &a {
        match b.get(k) {
            Some(&pb) => {
                let rel = (pa - pb).abs() / pa.abs().max(pb.abs());
                c.max_rel = c.max_rel.max(rel);
                if rel > tol {
                    c.differing += 1;
                }
            }
            None => c.extra += 1,
        }
    }
    c.missing = b.keys().filter(|k| !a.contains_key(*k)).count();
    c
}
```

`src/lib.rs`: add `pub mod dem;` after `pub mod circuit;`.

- [ ] **Step 4: Run the tests**

Run: `cargo test --release --no-default-features dem::`
Expected: 9 passed. If `depolarizing_channels_match_stim` fails by more than rounding, the conversion is wrong — do not loosen the tolerance.

- [ ] **Step 5: Commit**

```bash
git add src/dem.rs src/lib.rs
git commit -m "feat(dem): derive detector error models backwards through any circuit"
```

---

### Task 3: Weighted matching over any DEM

**Files:**
- Create: `src/dem_decoder.rs`
- Modify: `src/lib.rs` (`pub mod dem_decoder;`)

**Interfaces:**
- Consumes: `dem::{Dem, xor_prob}`, `blossom::{min_weight_perfect_matching, MAX_VERTICES}`.
- Produces: `DemDecoder::new(&Dem) -> Result<DemDecoder, String>`; `DemDecoder::decode(&self, defects: &[u32]) -> Result<Prediction, DecodeError>`; `DemDecoder::decode_bools(&self, dets: &[bool]) -> Result<Prediction, DecodeError>`; `pub conflicts: usize`; `Prediction { observables: u64, weight: f64 }`; `enum DecodeError { TooManyDefects(usize), Unmatchable, MatcherDeclined }`.

- [ ] **Step 1: Write the failing tests** (bottom of `src/dem_decoder.rs`)

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::circuit::Circuit;
    use crate::fixtures::REP3;
    use crate::surface_code::Xorshift;
    use std::collections::HashSet;

    type TestEdge = (u32, Option<u32>, f64, u64);

    /// Minimum-weight edge set with the given boundary, by exhaustion. Returns
    /// the weight and every observable mask that attains it (within 1e-4).
    fn brute_force(edges: &[TestEdge], defects: &[u32]) -> Option<(f64, Vec<u64>)> {
        let target: u64 = defects.iter().fold(0, |acc, &d| acc ^ (1 << d));
        let mut found: Vec<(f64, u64)> = Vec::new();
        for mask in 0u32..(1 << edges.len()) {
            let (mut syn, mut w, mut obs) = (0u64, 0.0, 0u64);
            for (i, e) in edges.iter().enumerate() {
                if (mask >> i) & 1 == 1 {
                    syn ^= 1 << e.0;
                    if let Some(b) = e.1 {
                        syn ^= 1 << b;
                    }
                    w += ((1.0 - e.2) / e.2).ln();
                    obs ^= e.3;
                }
            }
            if syn == target {
                found.push((w, obs));
            }
        }
        let best = found.iter().map(|f| f.0).fold(f64::INFINITY, f64::min);
        if !best.is_finite() {
            return None;
        }
        let mut masks: Vec<u64> = found.iter().filter(|f| f.0 <= best + 1e-4).map(|f| f.1).collect();
        masks.dedup();
        Some((best, masks))
    }

    #[test]
    fn matches_brute_force_on_small_graphs() {
        let mut rng = Xorshift::new(7);
        for trial in 0..300 {
            let nd = 3 + (rng.next_u64() % 5) as usize;
            let mut edges: Vec<TestEdge> = Vec::new();
            let mut seen = HashSet::new();
            while edges.len() < 2 * nd {
                let a = (rng.next_u64() % nd as u64) as u32;
                let b = if rng.next_u64() % 3 == 0 { None } else { Some((rng.next_u64() % nd as u64) as u32) };
                if b == Some(a) {
                    continue;
                }
                let key = match b { Some(b) => (a.min(b), Some(a.max(b))), None => (a, None) };
                if !seen.insert(key) {
                    continue;
                }
                edges.push((key.0, key.1, 0.01 + 0.3 * rng.next_f64(), rng.next_u64() % 2));
            }
            let mut text = String::new();
            for &(a, b, p, obs) in &edges {
                text.push_str(&format!("error({p}) D{a}"));
                if let Some(b) = b {
                    text.push_str(&format!(" D{b}"));
                }
                if obs == 1 {
                    text.push_str(" L0");
                }
                text.push('\n');
            }
            for d in 0..nd {
                text.push_str(&format!("detector D{d}\n"));
            }
            let dec = DemDecoder::new(&Dem::parse(&text).unwrap()).unwrap();
            let defects: Vec<u32> = (0..nd as u32).filter(|_| rng.next_u64() % 2 == 0).collect();
            match (brute_force(&edges, &defects), dec.decode(&defects)) {
                (None, Err(DecodeError::Unmatchable)) => {}
                (Some((w, masks)), Ok(pred)) => {
                    assert!((pred.weight - w).abs() < 1e-4, "trial {trial}: weight {} vs brute {w}", pred.weight);
                    assert!(masks.contains(&pred.observables), "trial {trial}: obs {} not in {masks:?}", pred.observables);
                }
                (b, d) => panic!("trial {trial}: brute {b:?}, decoder {d:?}"),
            }
        }
    }

    #[test]
    fn corrects_every_single_fault_of_the_repetition_code() {
        let dem = Dem::from_circuit(&Circuit::parse(REP3).unwrap()).unwrap();
        let dec = DemDecoder::new(&dem).unwrap();
        for m in &dem.mechanisms {
            let pred = dec.decode(&m.detectors).unwrap();
            assert_eq!(pred.observables, m.observables, "mechanism {:?}", m.detectors);
        }
        assert_eq!(dec.conflicts, 0);
    }

    #[test]
    fn empty_syndrome_predicts_nothing() {
        let dem = Dem::from_circuit(&Circuit::parse(REP3).unwrap()).unwrap();
        let pred = DemDecoder::new(&dem).unwrap().decode(&[]).unwrap();
        assert_eq!((pred.observables, pred.weight), (0, 0.0));
    }

    #[test]
    fn too_many_defects_is_an_error_not_a_fallback() {
        let text: String = (0..300).map(|d| format!("error(0.1) D{d}\n")).collect();
        let dec = DemDecoder::new(&Dem::parse(&text).unwrap()).unwrap();
        let defects: Vec<u32> = (0..257).collect();
        assert_eq!(dec.decode(&defects), Err(DecodeError::TooManyDefects(257)));
    }

    #[test]
    fn unmatchable_syndromes_are_errors() {
        let dec = DemDecoder::new(&Dem::parse("error(0.1) D0 D1\ndetector D2").unwrap()).unwrap();
        assert_eq!(dec.decode(&[2]), Err(DecodeError::Unmatchable));
        assert_eq!(dec.decode(&[0]), Err(DecodeError::Unmatchable));
        assert!(dec.decode(&[0, 1]).is_ok());
    }

    #[test]
    fn rejects_probabilities_above_one_half() {
        assert!(DemDecoder::new(&Dem::parse("error(0.6) D0").unwrap()).is_err());
    }

    #[test]
    fn parallel_edges_merge_like_pymatching() {
        // PyMatching 2.4 gives this pair one edge of p = 0.26, weight 1.0459685551826876.
        let dec = DemDecoder::new(&Dem::parse("error(0.1) D0 D1 L0\nerror(0.2) D0 D1 L0").unwrap()).unwrap();
        let pred = dec.decode(&[0, 1]).unwrap();
        assert_eq!(pred.observables, 1);
        assert!((pred.weight - 1.0459685551826876).abs() < 1e-12);
        assert_eq!(dec.conflicts, 0);
    }

    #[test]
    fn conflicting_parallel_edges_are_counted() {
        let dec = DemDecoder::new(&Dem::parse("error(0.1) D0 D1 L0\nerror(0.2) D0 D1").unwrap()).unwrap();
        assert_eq!(dec.conflicts, 1);
        assert_eq!(dec.decode(&[0, 1]).unwrap().observables, 0);
    }

    #[test]
    fn undecomposed_hyperedges_are_refused() {
        assert!(DemDecoder::new(&Dem::parse("error(0.1) D0 D1 D2").unwrap()).is_err());
    }
}
```

- [ ] **Step 2: Run to see them fail**

Run: `cargo test --release --no-default-features dem_decoder::`
Expected: compile failure.

- [ ] **Step 3: Implement `src/dem_decoder.rs`**

```rust
//! Minimum-weight perfect matching over any detector error model.
//!
//! The per-code decoders match on a graph whose edges carry data-qubit
//! corrections and unit weights. This one takes the standard formulation: each
//! edge is a graph-like piece of an error mechanism, weighted ln((1 − p)/p), and
//! carrying the logical observables it flips. What comes out is a prediction of
//! the observables, not a correction, which is all a memory experiment needs and
//! all an external dataset can be scored against.
//!
//! Shortest paths are Dijkstra from each defect; the matching is the existing
//! Edmonds blossom, with the boundary modelled as one interchangeable copy per
//! defect. Nothing here falls back to anything: a syndrome too large for the
//! dense matcher, or one with no consistent explanation, is an error the caller
//! counts. Bug 12 in the README is why.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};

use crate::blossom::{min_weight_perfect_matching, MAX_VERTICES};
use crate::dem::{xor_prob, Dem};

/// Integer weight resolution: 2^20 per unit of ln((1 − p)/p). Fine enough that
/// rounding cannot reorder paths that differ by more than a few parts in 10^6.
pub const SCALE: f64 = 1_048_576.0;

/// Cost of a pairing with no path. Dominates any real path, and stays far enough
/// below the blossom's own infinity that sums of 512 of them cannot overflow.
const UNREACHABLE: i64 = 1 << 44;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Prediction {
    pub observables: u64,
    /// Total weight of the chosen matching, in the float weights.
    pub weight: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub enum DecodeError {
    /// More defects than the dense matcher accepts (2k vertices > MAX_VERTICES).
    TooManyDefects(usize),
    /// A defect with no path to any partner or to the boundary.
    Unmatchable,
    /// The blossom hit one of its internal ceilings.
    MatcherDeclined,
}

struct Arc {
    to: u32,
    w: i64,
    wf: f64,
    obs: u64,
}

pub struct DemDecoder {
    num_detectors: usize,
    adj: Vec<Vec<Arc>>,
    /// Parallel edges whose observable masks disagreed. Zero for any code of
    /// distance three or more: a conflict is a weight-two logical operator.
    pub conflicts: usize,
}

impl DemDecoder {
    pub fn new(dem: &Dem) -> Result<DemDecoder, String> {
        let nd = dem.num_detectors;
        let boundary = nd as u32;
        let mut edges: HashMap<(u32, u32), (f64, u64)> = HashMap::new();
        let mut conflicts = 0usize;
        for m in &dem.mechanisms {
            if m.pieces.is_empty() {
                return Err(format!(
                    "mechanism on detectors {:?} fires more than two detectors and has no decomposition",
                    m.detectors
                ));
            }
            for piece in &m.pieces {
                let key = match piece.detectors.as_slice() {
                    [a] => (*a, boundary),
                    [a, b] => (*a.min(b), *a.max(b)),
                    other => return Err(format!("piece with {} detectors cannot be an edge", other.len())),
                };
                match edges.get_mut(&key) {
                    None => {
                        edges.insert(key, (m.p, piece.observables));
                    }
                    Some(e) if e.1 == piece.observables => e.0 = xor_prob(e.0, m.p),
                    Some(e) => {
                        conflicts += 1;
                        if m.p > e.0 {
                            *e = (m.p, piece.observables);
                        }
                    }
                }
            }
        }
        let mut keys: Vec<(u32, u32)> = edges.keys().copied().collect();
        keys.sort_unstable();
        let mut adj: Vec<Vec<Arc>> = (0..=nd).map(|_| Vec::new()).collect();
        for (u, v) in keys {
            let (p, obs) = edges[&(u, v)];
            if !(p > 0.0 && p <= 0.5) {
                return Err(format!("edge ({u}, {v}) has probability {p}; weights need 0 < p <= 0.5"));
            }
            let wf = ((1.0 - p) / p).ln();
            let w = (wf * SCALE).round() as i64;
            adj[u as usize].push(Arc { to: v, w, wf, obs });
            adj[v as usize].push(Arc { to: u, w, wf, obs });
        }
        Ok(DemDecoder { num_detectors: nd, adj, conflicts })
    }

    pub fn decode_bools(&self, dets: &[bool]) -> Result<Prediction, DecodeError> {
        let defects: Vec<u32> = dets.iter().enumerate().filter(|(_, &b)| b).map(|(i, _)| i as u32).collect();
        self.decode(&defects)
    }

    pub fn decode(&self, defects: &[u32]) -> Result<Prediction, DecodeError> {
        let k = defects.len();
        if k == 0 {
            return Ok(Prediction { observables: 0, weight: 0.0 });
        }
        if 2 * k > MAX_VERTICES {
            return Err(DecodeError::TooManyDefects(k));
        }
        let nd = self.num_detectors;
        let boundary = nd;
        let mut slot = vec![u32::MAX; nd + 1];
        for (i, &d) in defects.iter().enumerate() {
            slot[d as usize] = i as u32;
        }

        // Row i: distance, observable parity and float weight from defect i to
        // every defect j > i, with column k the boundary.
        let mut dist = vec![vec![UNREACHABLE; k + 1]; k];
        let mut obs = vec![vec![0u64; k + 1]; k];
        let mut wsum = vec![vec![0f64; k + 1]; k];

        let mut d_node = vec![i64::MAX; nd + 1];
        let mut o_node = vec![0u64; nd + 1];
        let mut f_node = vec![0f64; nd + 1];
        let mut touched: Vec<usize> = Vec::new();
        let mut heap = BinaryHeap::new();

        for (i, &src) in defects.iter().enumerate() {
            for &t in &touched {
                d_node[t] = i64::MAX;
            }
            touched.clear();
            heap.clear();
            let src = src as usize;
            d_node[src] = 0;
            o_node[src] = 0;
            f_node[src] = 0.0;
            touched.push(src);
            heap.push(Reverse((0i64, src)));
            // Targets still to settle: defects after i, and the boundary.
            let mut remaining = (k - 1 - i) + 1;
            while let Some(Reverse((du, u))) = heap.pop() {
                if du > d_node[u] {
                    continue;
                }
                if u == boundary {
                    dist[i][k] = du;
                    obs[i][k] = o_node[u];
                    wsum[i][k] = f_node[u];
                    remaining -= 1;
                    if remaining == 0 {
                        break;
                    }
                    // The boundary is not a node paths may pass through.
                    continue;
                }
                let j = slot[u];
                if j != u32::MAX && (j as usize) > i {
                    dist[i][j as usize] = du;
                    obs[i][j as usize] = o_node[u];
                    wsum[i][j as usize] = f_node[u];
                    remaining -= 1;
                    if remaining == 0 {
                        break;
                    }
                }
                for arc in &self.adj[u] {
                    let v = arc.to as usize;
                    let nd2 = du + arc.w;
                    if nd2 < d_node[v] {
                        if d_node[v] == i64::MAX {
                            touched.push(v);
                        }
                        d_node[v] = nd2;
                        o_node[v] = o_node[u] ^ arc.obs;
                        f_node[v] = f_node[u] + arc.wf;
                        heap.push(Reverse((nd2, v)));
                    }
                }
            }
        }

        // Defects 0..k, then one boundary copy per defect; copies pair for free.
        let n = 2 * k;
        let mut cost = vec![vec![0i64; n]; n];
        for i in 0..k {
            for j in (i + 1)..k {
                cost[i][j] = dist[i][j];
                cost[j][i] = dist[i][j];
            }
            for c in 0..k {
                cost[i][k + c] = dist[i][k];
                cost[k + c][i] = dist[i][k];
            }
        }
        let mate = min_weight_perfect_matching(n, &cost).ok_or(DecodeError::MatcherDeclined)?;

        let mut prediction = Prediction { observables: 0, weight: 0.0 };
        for i in 0..k {
            let j = mate[i];
            let (d, o, w) = if j >= k {
                (dist[i][k], obs[i][k], wsum[i][k])
            } else if i < j {
                (dist[i][j], obs[i][j], wsum[i][j])
            } else {
                continue;
            };
            if d >= UNREACHABLE {
                return Err(DecodeError::Unmatchable);
            }
            prediction.observables ^= o;
            prediction.weight += w;
        }
        Ok(prediction)
    }
}
```

`src/lib.rs`: `pub mod dem_decoder;`. Confirm `blossom::MAX_VERTICES` is `pub` (it is: `pub const MAX_VERTICES: usize = 512;`).

- [ ] **Step 4: Run the tests**

Run: `cargo test --release --no-default-features dem_decoder::`
Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
git add src/dem_decoder.rs src/lib.rs
git commit -m "feat(decoder): weighted MWPM over any detector error model, predicting observables"
```

---

### Task 4: The frame sampler and shot formats

**Files:**
- Create: `src/frame_sampler.rs`, `src/shots.rs`
- Modify: `src/lib.rs` (`pub mod frame_sampler; pub mod shots;`)

**Interfaces:**
- Consumes: `circuit::{Circuit, Instr, Basis}`, `surface_code::Xorshift`.
- Produces: `FrameSampler::new(&Circuit) -> Result<FrameSampler, String>`, `num_detectors()`, `num_observables()`, `sample(&self, &mut Xorshift) -> Shot`; `Shot { detectors: Vec<bool>, observables: u64 }`. `shots::{write_01, read_01, write_b8, read_b8, pack_row}` with `pack_row(bits: &[bool], out: &mut Vec<u8>)`.

- [ ] **Step 1: Failing tests**

`src/frame_sampler.rs` tests:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::dem::Dem;
    use crate::fixtures::REP3;

    /// Marginals follow from the model exactly: a detector fires when an odd
    /// number of its independent mechanisms do, (1 − Π(1 − 2p))/2. The sampler
    /// never reads the model, so agreement is a check on both.
    #[test]
    fn detector_marginals_match_the_error_model() {
        let circuit = Circuit::parse(&REP3.replace("0.01", "0.05")).unwrap();
        let dem = Dem::from_circuit(&circuit).unwrap();
        let sampler = FrameSampler::new(&circuit).unwrap();
        let n = 200_000usize;
        let mut counts = vec![0usize; sampler.num_detectors()];
        let mut obs = 0usize;
        let mut rng = Xorshift::new(11);
        for _ in 0..n {
            let shot = sampler.sample(&mut rng);
            for (i, &b) in shot.detectors.iter().enumerate() {
                counts[i] += b as usize;
            }
            obs += (shot.observables & 1) as usize;
        }
        let predict = |pick: &dyn Fn(&crate::dem::Mechanism) -> bool| -> f64 {
            let prod: f64 = dem.mechanisms.iter().filter(|m| pick(m)).map(|m| 1.0 - 2.0 * m.p).product();
            (1.0 - prod) / 2.0
        };
        for (d, &c) in counts.iter().enumerate() {
            let q = predict(&|m| m.detectors.contains(&(d as u32)));
            let sigma = (q * (1.0 - q) / n as f64).sqrt();
            let rate = c as f64 / n as f64;
            assert!((rate - q).abs() < 5.0 * sigma, "D{d}: sampled {rate}, model {q}");
        }
        let q = predict(&|m| m.observables & 1 == 1);
        let rate = obs as f64 / n as f64;
        assert!((rate - q).abs() < 5.0 * (q * (1.0 - q) / n as f64).sqrt(), "L0: sampled {rate}, model {q}");
    }

    #[test]
    fn noiseless_circuits_never_fire() {
        let circuit = Circuit::parse(&REP3.replace("0.01", "0")).unwrap();
        let sampler = FrameSampler::new(&circuit).unwrap();
        let mut rng = Xorshift::new(3);
        for _ in 0..2000 {
            let shot = sampler.sample(&mut rng);
            assert!(shot.detectors.iter().all(|&b| !b));
            assert_eq!(shot.observables, 0);
        }
    }
}
```

`src/shots.rs` tests:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Vec<Vec<bool>> {
        vec![
            vec![true, false, false, false, false, false, false, false, true, true],
            vec![false; 10],
            vec![true; 10],
        ]
    }

    #[test]
    fn b8_round_trips_and_is_little_endian_within_bytes() {
        let bytes = write_b8(&sample(), 10);
        assert_eq!(bytes.len(), 6);
        assert_eq!(bytes[0], 0b0000_0001);
        assert_eq!(bytes[1], 0b0000_0011);
        assert_eq!(read_b8(&bytes, 10).unwrap(), sample());
    }

    #[test]
    fn zero_one_round_trips() {
        let text = write_01(&sample());
        assert_eq!(text.lines().next().unwrap(), "1000000011");
        assert_eq!(read_01(&text, 10).unwrap(), sample());
    }

    #[test]
    fn malformed_input_is_an_error() {
        assert!(read_b8(&[0u8; 5], 10).is_err());
        assert!(read_01("101\n", 10).is_err());
        assert!(read_01("10x0000000\n", 10).is_err());
    }
}
```

- [ ] **Step 2: Run to see them fail**

Run: `cargo test --release --no-default-features frame_sampler:: shots::`
Expected: compile failure.

- [ ] **Step 3: Implement**

`src/frame_sampler.rs`:

```rust
//! Sampling a circuit with a Pauli frame, independently of any error model.
//!
//! The frame records how this shot differs from a noiseless run. For a Clifford
//! circuit under Pauli noise that is all a detector needs: it reads a parity the
//! noiseless circuit fixes, so it fires exactly when the frame flips an odd
//! number of the records it reads. Noise is sampled with each channel's true
//! (disjoint) semantics, not the error model's independent approximation, and
//! nothing here looks at the model — so a model that is wrong about the circuit
//! shows up as disagreement between the two rather than agreeing with itself.
//!
//! Resets and measurements randomise the component of the frame that the state
//! is insensitive to, as Stim's frame simulator does. It costs a draw and changes
//! nothing for a deterministic detector; for a non-deterministic one it makes the
//! failure visible as a coin flip instead of hiding it.

use crate::circuit::{Basis, Circuit, Instr};
use crate::surface_code::Xorshift;

pub struct Shot {
    pub detectors: Vec<bool>,
    pub observables: u64,
}

pub struct FrameSampler {
    instrs: Vec<Instr>,
    num_qubits: usize,
    num_measurements: usize,
    detectors: Vec<Vec<usize>>,
    observables: Vec<Vec<usize>>,
}

impl FrameSampler {
    pub fn new(circuit: &Circuit) -> Result<FrameSampler, String> {
        let res = circuit.resolve()?;
        Ok(FrameSampler {
            instrs: res.instrs,
            num_qubits: res.num_qubits,
            num_measurements: res.num_measurements,
            detectors: res.detectors,
            observables: res.observables,
        })
    }

    pub fn num_detectors(&self) -> usize {
        self.detectors.len()
    }

    pub fn num_observables(&self) -> usize {
        self.observables.len()
    }

    pub fn sample(&self, rng: &mut Xorshift) -> Shot {
        let nq = self.num_qubits;
        let mut x = vec![false; nq];
        let mut z: Vec<bool> = (0..nq).map(|_| rng.next_u64() & 1 == 1).collect();
        let mut rec = Vec::with_capacity(self.num_measurements);
        let apply = |x: &mut [bool], z: &mut [bool], q: usize, pauli: u8| {
            if pauli & 1 != 0 {
                x[q] ^= true;
            }
            if pauli & 2 != 0 {
                z[q] ^= true;
            }
        };
        for ins in &self.instrs {
            match ins {
                Instr::Reset { basis, qubits } => {
                    for &q in qubits {
                        let q = q as usize;
                        match basis {
                            Basis::Z => { x[q] = false; z[q] = rng.next_u64() & 1 == 1; }
                            Basis::X => { z[q] = false; x[q] = rng.next_u64() & 1 == 1; }
                        }
                    }
                }
                Instr::H(qubits) => {
                    for &q in qubits {
                        let q = q as usize;
                        std::mem::swap(&mut x[q], &mut z[q]);
                    }
                }
                Instr::Cx(pairs) => {
                    for &(c, t) in pairs {
                        let (c, t) = (c as usize, t as usize);
                        if x[c] { x[t] ^= true; }
                        if z[t] { z[c] ^= true; }
                    }
                }
                Instr::Cz(pairs) => {
                    for &(a, b) in pairs {
                        let (a, b) = (a as usize, b as usize);
                        if x[a] { z[b] ^= true; }
                        if x[b] { z[a] ^= true; }
                    }
                }
                Instr::Measure { basis, reset, flip, qubits } => {
                    for &q in qubits {
                        let q = q as usize;
                        let mut r = match basis { Basis::Z => x[q], Basis::X => z[q] };
                        if *flip > 0.0 && rng.next_f64() < *flip {
                            r = !r;
                        }
                        rec.push(r);
                        match basis {
                            Basis::Z => z[q] = rng.next_u64() & 1 == 1,
                            Basis::X => x[q] = rng.next_u64() & 1 == 1,
                        }
                        if *reset {
                            match basis {
                                Basis::Z => { x[q] = false; z[q] = rng.next_u64() & 1 == 1; }
                                Basis::X => { z[q] = false; x[q] = rng.next_u64() & 1 == 1; }
                            }
                        }
                    }
                }
                Instr::PauliError { pauli, p, qubits } => {
                    for &q in qubits {
                        if rng.next_f64() < *p {
                            apply(&mut x, &mut z, q as usize, *pauli);
                        }
                    }
                }
                Instr::Depolarize1 { p, qubits } => {
                    for &q in qubits {
                        if rng.next_f64() < *p {
                            apply(&mut x, &mut z, q as usize, 1 + (rng.next_u64() % 3) as u8);
                        }
                    }
                }
                Instr::PauliChannel1 { px, py, pz, qubits } => {
                    for &q in qubits {
                        let u = rng.next_f64();
                        let pauli = if u < *px { 1 } else if u < px + py { 3 } else if u < px + py + pz { 2 } else { 0 };
                        apply(&mut x, &mut z, q as usize, pauli);
                    }
                }
                Instr::Depolarize2 { p, pairs } => {
                    for &(a, b) in pairs {
                        if rng.next_f64() < *p {
                            let r = 1 + rng.next_u64() % 15;
                            apply(&mut x, &mut z, a as usize, (r & 3) as u8);
                            apply(&mut x, &mut z, b as usize, (r >> 2) as u8);
                        }
                    }
                }
                _ => {}
            }
        }
        let parity = |recs: &[usize]| recs.iter().fold(false, |acc, &m| acc ^ rec[m]);
        let detectors = self.detectors.iter().map(|r| parity(r)).collect();
        let mut observables = 0u64;
        for (k, r) in self.observables.iter().enumerate() {
            if parity(r) {
                observables |= 1u64 << k;
            }
        }
        Shot { detectors, observables }
    }
}
```

`src/shots.rs`:

```rust
//! Stim's shot formats: `01` (one line of '0'/'1' per shot) and `b8` (each shot
//! packed into ceil(n/8) bytes, bit i at byte i/8, bit i%8 — little-endian
//! within each byte). Google's published detection events use these.

pub fn pack_row(bits: &[bool], out: &mut Vec<u8>) {
    let start = out.len();
    out.resize(start + bits.len().div_ceil(8), 0);
    for (i, &b) in bits.iter().enumerate() {
        if b {
            out[start + i / 8] |= 1 << (i % 8);
        }
    }
}

pub fn write_b8(shots: &[Vec<bool>], num_bits: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(shots.len() * num_bits.div_ceil(8));
    for s in shots {
        debug_assert_eq!(s.len(), num_bits);
        pack_row(s, &mut out);
    }
    out
}

pub fn read_b8(bytes: &[u8], num_bits: usize) -> Result<Vec<Vec<bool>>, String> {
    let stride = num_bits.div_ceil(8);
    if stride == 0 || bytes.len() % stride != 0 {
        return Err(format!("{} bytes is not a whole number of {num_bits}-bit shots", bytes.len()));
    }
    Ok(bytes
        .chunks(stride)
        .map(|row| (0..num_bits).map(|i| (row[i / 8] >> (i % 8)) & 1 == 1).collect())
        .collect())
}

pub fn write_01(shots: &[Vec<bool>]) -> String {
    let mut s = String::with_capacity(shots.iter().map(|r| r.len() + 1).sum());
    for row in shots {
        s.extend(row.iter().map(|&b| if b { '1' } else { '0' }));
        s.push('\n');
    }
    s
}

pub fn read_01(text: &str, num_bits: usize) -> Result<Vec<Vec<bool>>, String> {
    text.lines()
        .filter(|l| !l.is_empty())
        .enumerate()
        .map(|(i, line)| {
            if line.len() != num_bits {
                return Err(format!("shot {i} has {} bits, expected {num_bits}", line.len()));
            }
            line.chars()
                .map(|c| match c {
                    '0' => Ok(false),
                    '1' => Ok(true),
                    _ => Err(format!("shot {i}: unexpected character '{c}'")),
                })
                .collect()
        })
        .collect()
}
```

- [ ] **Step 4: Run the tests**

Run: `cargo test --release --no-default-features frame_sampler:: shots::`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/frame_sampler.rs src/shots.rs src/lib.rs
git commit -m "feat: a frame sampler that never reads the error model, and Stim's shot formats"
```

---

### Task 5: Surface-code memory experiments as circuits

**Files:**
- Create: `src/memory.rs`
- Modify: `src/surface_code.rs` (`X_ORDER`/`Z_ORDER` → `pub const` on `RotatedSurfaceCode`; `next_shot_rng` → `pub(crate)`), `src/lib.rs` (`pub mod memory;`)

**Interfaces:**
- Consumes: `RotatedSurfaceCode::{new, X_ORDER, Z_ORDER, neighbor_at, get_neighbors, round_program}`, `XZZXSurfaceCode::{new, SCHEDULE_A, SCHEDULE_B, get_neighbor_idx, get_neighbors, round_program}`, `circuit_model::{Op, StabKind}`, `circuit::{Circuit, Instr, Basis}`.
- Produces: `NoiseModel::{Current { p, eta }, Sd6 { p }}`, `CodeKind::{Rotated, Xzzx}`, `Gate2`, `RoundLayers`, `Patch` (fields below), `rotated_patch(&RotatedSurfaceCode, Basis) -> Patch`, `xzzx_patch(&XZZXSurfaceCode, Basis) -> Result<Patch, String>`, `xzzx_hadamard_pattern(&XZZXSurfaceCode) -> Result<Vec<bool>, String>`, `memory_circuit(&Patch, rounds, NoiseModel) -> Circuit`, `generate(CodeKind, d, rounds, NoiseModel, Basis) -> Result<Circuit, String>`, `patch_for(CodeKind, d, Basis) -> Result<Patch, String>`.

**Conventions fixed by this task** (checked by its tests, relied on by Tasks 6–8):
- Qubits: data `0..n` at their lattice coordinates, then ancillas in `RoundLayers::ancillas` order (rotated: X ancillas then Z ancillas, as `round_program` numbers them; XZZX: `stabilizers` order).
- Logical operators of the rotated code: Z runs down a column (qubits `i·d`), X along a row (qubits `0..d`). `memory_z` measures data in Z and its observable is the column; `memory_x` measures in X and its observable is the row. (This matches the existing `simulate_circuit_noise_with_model`, which detects a logical X by parity against the column.)
- Detector coordinates `(x, y, t)`: the plaquette's coordinates and the round index. Current: rounds `t = 1..=T+1` compare against round `t − 1` (round 0 is the noiseless baseline); the final readout is `t = T+2`. SD6: round 1 carries only the deterministic plaquettes, rounds `2..=T` all, final readout `t = T+1`.

- [ ] **Step 1: The refactor the generator needs**

In `src/surface_code.rs`, inside `impl RotatedSurfaceCode` (before `round_program`), add — moving the comment block that currently sits above the two local consts:

```rust
    /// X ancillas walk their plaquette column-major, Z ancillas row-major —
    /// the two orders are transposes, which is what makes the extraction
    /// circuits commute where plaquettes overlap. The pairing is not
    /// arbitrary and not guessable: with the two orders swapped, every single
    /// fault is still corrected at d = 5 and d = 7 but 44 of 600 fail at
    /// d = 3. The exhaustive single-fault check settled which way round it
    /// goes. `memory.rs` reads these same two arrays.
    pub const X_ORDER: [(i32, i32); 4] = [(-1, -1), (-1, 1), (1, -1), (1, 1)];
    pub const Z_ORDER: [(i32, i32); 4] = [(-1, -1), (1, -1), (-1, 1), (1, 1)];
```

and in `round_program` delete the two local `const` lines and their comment, replacing uses with `Self::X_ORDER[step]` / `Self::Z_ORDER[step]`. Change `fn next_shot_rng()` to `pub(crate) fn next_shot_rng()`.

Run: `cargo test --release --no-default-features`
Expected: 22 passed, 3 ignored — unchanged.

- [ ] **Step 2: Write the failing tests** (bottom of `src/memory.rs`)

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::dem::Dem;
    use crate::dem_decoder::DemDecoder;

    /// Per-qubit projection of a round: for each qubit, the ops touching it, in
    /// order. Two rounds with equal projections are the same circuit up to
    /// reordering gates on disjoint qubits.
    fn projection(ops: &[String], touches: &[Vec<usize>], nq: usize) -> Vec<Vec<String>> {
        let mut out = vec![Vec::new(); nq];
        for (op, qs) in ops.iter().zip(touches) {
            for &q in qs {
                out[q].push(op.clone());
            }
        }
        out
    }

    fn describe_program(program: &[Op]) -> (Vec<String>, Vec<Vec<usize>>) {
        let mut ops = Vec::new();
        let mut touches = Vec::new();
        for op in program {
            let (s, t) = match *op {
                Op::Reset(q) => ("R".to_string(), vec![q]),
                Op::H(q) => ("H".to_string(), vec![q]),
                Op::Cnot(a, b) => (format!("CX {a} {b}"), vec![a, b]),
                Op::Cz(a, b) => (format!("CZ {a} {b}"), vec![a, b]),
                Op::Measure(q, _, _) => ("M".to_string(), vec![q]),
                Op::Noise(_) => continue,
            };
            ops.push(s);
            touches.push(t);
        }
        (ops, touches)
    }

    fn describe_layers(l: &RoundLayers) -> (Vec<String>, Vec<Vec<usize>>) {
        let mut ops = Vec::new();
        let mut touches = Vec::new();
        for &a in &l.ancillas {
            ops.push("R".to_string());
            touches.push(vec![a as usize]);
        }
        for &a in &l.hadamard {
            ops.push("H".to_string());
            touches.push(vec![a as usize]);
        }
        for step in &l.steps {
            for g in step {
                let (s, a, b) = match *g {
                    Gate2::Cx(a, b) => (format!("CX {a} {b}"), a, b),
                    Gate2::Cz(a, b) => (format!("CZ {a} {b}"), a, b),
                };
                ops.push(s);
                touches.push(vec![a as usize, b as usize]);
            }
        }
        for &a in &l.hadamard {
            ops.push("H".to_string());
            touches.push(vec![a as usize]);
        }
        for &a in &l.ancillas {
            ops.push("M".to_string());
            touches.push(vec![a as usize]);
        }
        (ops, touches)
    }

    #[test]
    fn layers_are_the_round_program() {
        for d in [3, 5, 7] {
            for kind in [CodeKind::Rotated, CodeKind::Xzzx] {
                let patch = patch_for(kind, d, Basis::Z).unwrap();
                let (a, ta) = describe_program(&patch.program);
                let (b, tb) = describe_layers(&patch.layers);
                assert_eq!(
                    projection(&a, &ta, patch.num_qubits),
                    projection(&b, &tb, patch.num_qubits),
                    "{kind:?} d = {d}"
                );
            }
        }
    }

    #[test]
    fn xzzx_is_the_rotated_code_under_a_checkerboard_of_hadamards() {
        for d in [3, 5, 7, 9] {
            let code = XZZXSurfaceCode::new(d);
            let h = xzzx_hadamard_pattern(&code).unwrap();
            for (q, &(x, y)) in code.data_qubits.iter().enumerate() {
                assert_eq!(h[q], ((x + y) / 2) % 2 == 1, "d = {d}, qubit {q} at ({x}, {y})");
            }
        }
    }

    fn all_circuits(ds: &[usize]) -> Vec<(String, Circuit)> {
        let mut out = Vec::new();
        for &d in ds {
            for kind in [CodeKind::Rotated, CodeKind::Xzzx] {
                for basis in [Basis::Z, Basis::X] {
                    for noise in [NoiseModel::Current { p: 0.003, eta: 0.5 }, NoiseModel::Sd6 { p: 0.003 }] {
                        let name = format!("{kind:?} {basis:?} {noise:?} d = {d}");
                        out.push((name, generate(kind, d, d, noise, basis).unwrap()));
                    }
                }
            }
        }
        out
    }

    #[test]
    fn every_memory_circuit_has_a_deterministic_decomposable_model() {
        for (name, c) in all_circuits(&[3, 5]) {
            let dem = Dem::from_circuit(&c).unwrap_or_else(|e| panic!("{name}: {e}"));
            for m in &dem.mechanisms {
                assert!(!m.pieces.is_empty(), "{name}: {:?} undecomposed", m.detectors);
                let mut dets: Vec<u32> = Vec::new();
                let mut obs = 0u64;
                for piece in &m.pieces {
                    assert!((1..=2).contains(&piece.detectors.len()), "{name}: piece {:?}", piece.detectors);
                    for &d in &piece.detectors {
                        match dets.iter().position(|&x| x == d) { Some(i) => { dets.swap_remove(i); } None => dets.push(d) }
                    }
                    obs ^= piece.observables;
                }
                dets.sort_unstable();
                assert_eq!((dets, obs), (m.detectors.clone(), m.observables), "{name}: pieces do not XOR back");
            }
            assert_eq!(Circuit::parse(&c.to_stim()).unwrap(), c, "{name}: text round trip");
        }
    }

    #[test]
    fn detector_counts_follow_the_conventions() {
        let d = 3;
        let na = 8; // 4 X plaquettes, 4 Z plaquettes
        let det = 4;
        let current = generate(CodeKind::Rotated, d, d, NoiseModel::Current { p: 0.003, eta: 0.5 }, Basis::Z).unwrap();
        assert_eq!(current.resolve().unwrap().detectors.len(), (d + 1) * na + det);
        let sd6 = generate(CodeKind::Rotated, d, d, NoiseModel::Sd6 { p: 0.003 }, Basis::Z).unwrap();
        // Stim's own rotated_memory_z at d = 3, rounds = 3 also has 24.
        assert_eq!(sd6.resolve().unwrap().detectors.len(), 2 * det + (d - 1) * na);
    }

    /// Check 6 of the spec: a distance-d circuit must survive any one fault.
    /// Every mechanism of the model, decoded alone, must predict its own flip.
    pub(crate) fn single_fault_failures(c: &Circuit) -> (usize, usize) {
        let dem = Dem::from_circuit(c).unwrap();
        let dec = DemDecoder::new(&dem).unwrap();
        let mut failures = 0;
        for m in &dem.mechanisms {
            match dec.decode(&m.detectors) {
                Ok(pred) if pred.observables == m.observables => {}
                _ => failures += 1,
            }
        }
        (dem.mechanisms.len(), failures)
    }

    #[test]
    fn every_single_fault_is_corrected_d3_d5() {
        for (name, c) in all_circuits(&[3, 5]) {
            let (tested, failed) = single_fault_failures(&c);
            assert_eq!(failed, 0, "{name}: {failed} of {tested} single faults fail");
        }
    }

    #[test]
    #[ignore] // ~1 min; run in the verification step.
    fn every_single_fault_is_corrected_d7() {
        for (name, c) in all_circuits(&[7]) {
            let (tested, failed) = single_fault_failures(&c);
            println!("{name}: {tested} mechanisms, {failed} fail");
            assert_eq!(failed, 0, "{name}");
        }
    }
}
```

- [ ] **Step 3: Run to see them fail**

Run: `cargo test --release --no-default-features memory::`
Expected: compile failure.

- [ ] **Step 4: Implement `src/memory.rs`**

```rust
//! Surface-code memory experiments, written out as circuits.
//!
//! The same patch the rest of the engine simulates, in the general form: a
//! circuit with detectors and an observable, which Stim can read and this
//! engine can build a model of, sample and decode like any other.
//!
//! Two noise models:
//!
//! - **Current** reproduces the engine's own circuit-level model gate for gate:
//!   a biased Pauli at every `Op::Noise` location of `round_program` (so each
//!   qubit of a CNOT errs independently, and a gate fails about 2p of the
//!   time), flipped readouts, no idle noise, and a noiseless first and last
//!   round. `equivalence.rs` proves it is the old path's circuit.
//! - **SD6** is the standard model of Gidney et al. (2021), the one published
//!   thresholds assume: DEPOLARIZE2(p) after every two-qubit gate,
//!   DEPOLARIZE1(p) after every single-qubit gate and on every idle qubit in a
//!   layer, flips of p after resets and before measurements, every round noisy.
//!
//! The CNOT schedules are the ones the exhaustive single-fault checks settled,
//! read from the same constants `round_program` uses, never copied.

use std::collections::HashMap;

use crate::circuit::{Basis, Circuit, Instr};
use crate::circuit_model::Op;
use crate::surface_code::{RotatedSurfaceCode, XZZXSurfaceCode};

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum NoiseModel {
    Current { p: f64, eta: f64 },
    Sd6 { p: f64 },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CodeKind {
    Rotated,
    Xzzx,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Gate2 {
    Cx(u32, u32),
    Cz(u32, u32),
}

impl Gate2 {
    fn qubits(&self) -> (u32, u32) {
        match *self {
            Gate2::Cx(a, b) | Gate2::Cz(a, b) => (a, b),
        }
    }
}

/// One syndrome-extraction round, as layers.
pub struct RoundLayers {
    /// Ancillas, in measurement order.
    pub ancillas: Vec<u32>,
    /// Ancillas rotated into the X basis before and after the entangling layers.
    pub hadamard: Vec<u32>,
    pub steps: [Vec<Gate2>; 4],
}

/// Everything a memory experiment needs to know about a patch.
pub struct Patch {
    pub d: usize,
    pub num_qubits: usize,
    pub num_data: usize,
    /// Data qubits, then ancillas in `layers.ancillas` order.
    pub qubit_coords: Vec<(usize, usize)>,
    pub layers: RoundLayers,
    /// `round_program`: where the Current model puts its noise.
    pub program: Vec<Op>,
    /// Data qubits of each ancilla's plaquette, by ancilla position.
    pub support: Vec<Vec<u32>>,
    /// Preparation and readout basis of each data qubit.
    pub data_basis: Vec<Basis>,
    /// Ancilla positions whose first reading the preparation fixes.
    pub deterministic: Vec<usize>,
    /// Data qubits whose readouts multiply to the logical observable.
    pub observable: Vec<u32>,
}

fn other(b: Basis) -> Basis {
    match b { Basis::X => Basis::Z, Basis::Z => Basis::X }
}

fn column(d: usize) -> Vec<u32> {
    (0..d).map(|i| (i * d) as u32).collect()
}

fn row(d: usize) -> Vec<u32> {
    (0..d).map(|i| i as u32).collect()
}

pub fn rotated_layers(code: &RotatedSurfaceCode) -> RoundLayers {
    let n = code.data_qubits.len();
    let nx = code.x_stabilizers.len();
    let nz = code.z_stabilizers.len();
    let x_anc = |j: usize| (n + j) as u32;
    let z_anc = |k: usize| (n + nx + k) as u32;
    let mut steps: [Vec<Gate2>; 4] = Default::default();
    for (step, gates) in steps.iter_mut().enumerate() {
        let (dx, dy) = RotatedSurfaceCode::X_ORDER[step];
        for j in 0..nx {
            if let Some(q) = code.neighbor_at(&code.x_stabilizers[j], dx, dy) {
                gates.push(Gate2::Cx(x_anc(j), q as u32));
            }
        }
        let (dx, dy) = RotatedSurfaceCode::Z_ORDER[step];
        for k in 0..nz {
            if let Some(q) = code.neighbor_at(&code.z_stabilizers[k], dx, dy) {
                gates.push(Gate2::Cx(q as u32, z_anc(k)));
            }
        }
    }
    RoundLayers {
        ancillas: (0..nx).map(x_anc).chain((0..nz).map(z_anc)).collect(),
        hadamard: (0..nx).map(x_anc).collect(),
        steps,
    }
}

pub fn xzzx_layers(code: &XZZXSurfaceCode) -> RoundLayers {
    let n = code.data_qubits.len();
    let ns = code.stabilizers.len();
    let split = code.z_stabilizers.len();
    let anc = |s: usize| (n + s) as u32;
    let mut steps: [Vec<Gate2>; 4] = Default::default();
    for (step, gates) in steps.iter_mut().enumerate() {
        for s in 0..ns {
            let (dx, dy, is_x) =
                if s < split { XZZXSurfaceCode::SCHEDULE_A[step] } else { XZZXSurfaceCode::SCHEDULE_B[step] };
            let (sx, sy) = code.stabilizers[s];
            if let Some(q) = code.get_neighbor_idx(sx as i32 + dx, sy as i32 + dy) {
                gates.push(if is_x { Gate2::Cx(anc(s), q as u32) } else { Gate2::Cz(anc(s), q as u32) });
            }
        }
    }
    RoundLayers { ancillas: (0..ns).map(anc).collect(), hadamard: (0..ns).map(anc).collect(), steps }
}

pub fn rotated_patch(code: &RotatedSurfaceCode, basis: Basis) -> Patch {
    let d = code.d;
    let n = code.data_qubits.len();
    let nx = code.x_stabilizers.len();
    let nz = code.z_stabilizers.len();
    let stab_coords: Vec<(usize, usize)> =
        code.x_stabilizers.iter().chain(code.z_stabilizers.iter()).copied().collect();
    let support = stab_coords
        .iter()
        .map(|s| code.get_neighbors(s).into_iter().map(|q| q as u32).collect())
        .collect();
    let (deterministic, observable) = match basis {
        Basis::Z => ((nx..nx + nz).collect(), column(d)),
        Basis::X => ((0..nx).collect(), row(d)),
    };
    Patch {
        d,
        num_qubits: n + nx + nz,
        num_data: n,
        qubit_coords: code.data_qubits.iter().copied().chain(stab_coords).collect(),
        layers: rotated_layers(code),
        program: code.round_program(),
        support,
        data_basis: vec![basis; n],
        deterministic,
        observable,
    }
}

/// Which data qubits carry a Hadamard when the XZZX code is read as the
/// rotated code conjugated by single-qubit Hadamards.
///
/// Derived, not assumed: at every plaquette the rotated code has Z (or X) on
/// every leg and XZZX has X on the NW/SE diagonal and Z on NE/SW. A leg differs
/// exactly where a Hadamard sits, and each qubit must get the same answer from
/// every plaquette it belongs to — or the equivalence does not hold and this
/// fails rather than guessing.
pub fn xzzx_hadamard_pattern(code: &XZZXSurfaceCode) -> Result<Vec<bool>, String> {
    let n = code.data_qubits.len();
    let split = code.z_stabilizers.len();
    let mut h: Vec<Option<bool>> = vec![None; n];
    for (s, &(sx, sy)) in code.stabilizers.iter().enumerate() {
        let rotated_is_z = s < split;
        for &(dx, dy, is_x) in &XZZXSurfaceCode::SCHEDULE_A {
            if let Some(q) = code.get_neighbor_idx(sx as i32 + dx, sy as i32 + dy) {
                let differs = is_x == rotated_is_z;
                match h[q] {
                    None => h[q] = Some(differs),
                    Some(prev) if prev != differs => {
                        return Err(format!("data qubit {q} needs a Hadamard for one plaquette and not another"));
                    }
                    _ => {}
                }
            }
        }
    }
    h.into_iter()
        .enumerate()
        .map(|(q, v)| v.ok_or_else(|| format!("data qubit {q} is in no plaquette")))
        .collect()
}

pub fn xzzx_patch(code: &XZZXSurfaceCode, basis: Basis) -> Result<Patch, String> {
    let d = code.d;
    let n = code.data_qubits.len();
    let ns = code.stabilizers.len();
    let split = code.z_stabilizers.len();
    let h = xzzx_hadamard_pattern(code)?;
    let support = code
        .stabilizers
        .iter()
        .map(|s| code.get_neighbors(s).into_iter().map(|q| q as u32).collect())
        .collect();
    let (deterministic, observable) = match basis {
        Basis::Z => ((0..split).collect(), column(d)),
        Basis::X => ((split..ns).collect(), row(d)),
    };
    Ok(Patch {
        d,
        num_qubits: n + ns,
        num_data: n,
        qubit_coords: code.data_qubits.iter().chain(code.stabilizers.iter()).copied().collect(),
        layers: xzzx_layers(code),
        program: code.round_program(),
        support,
        data_basis: h.iter().map(|&flip| if flip { other(basis) } else { basis }).collect(),
        deterministic,
        observable,
    })
}

pub fn patch_for(kind: CodeKind, d: usize, basis: Basis) -> Result<Patch, String> {
    if !(2..=11).contains(&d) {
        return Err(format!("distance {d} is outside 2..=11"));
    }
    match kind {
        CodeKind::Rotated => Ok(rotated_patch(&RotatedSurfaceCode::new(d), basis)),
        CodeKind::Xzzx => xzzx_patch(&XZZXSurfaceCode::new(d), basis),
    }
}

pub fn generate(kind: CodeKind, d: usize, rounds: usize, noise: NoiseModel, basis: Basis) -> Result<Circuit, String> {
    if rounds == 0 {
        return Err("a memory experiment needs at least one round".into());
    }
    Ok(memory_circuit(&patch_for(kind, d, basis)?, rounds, noise))
}

pub fn memory_circuit(patch: &Patch, rounds: usize, noise: NoiseModel) -> Circuit {
    match noise {
        NoiseModel::Current { p, eta } => current_circuit(patch, rounds, p, eta),
        NoiseModel::Sd6 { p } => sd6_circuit(patch, rounds, p),
    }
}

/* -- Shared pieces ---------------------------------------------------------- */

fn lookback(now: usize, abs: usize) -> u32 {
    (now - abs) as u32
}

fn coords_header(patch: &Patch, c: &mut Vec<Instr>) {
    for (q, &(x, y)) in patch.qubit_coords.iter().enumerate() {
        c.push(Instr::QubitCoords { coords: vec![x as f64, y as f64], qubits: vec![q as u32] });
    }
}

fn data_by_basis(patch: &Patch) -> (Vec<u32>, Vec<u32>) {
    let mut zs = Vec::new();
    let mut xs = Vec::new();
    for (q, &b) in patch.data_basis.iter().enumerate() {
        match b {
            Basis::Z => zs.push(q as u32),
            Basis::X => xs.push(q as u32),
        }
    }
    (zs, xs)
}

/// Reset the data in their bases, with a flip of `p` after (none when p = 0).
fn reset_data(patch: &Patch, c: &mut Vec<Instr>, p: f64) {
    let (zs, xs) = data_by_basis(patch);
    if !zs.is_empty() {
        c.push(Instr::Reset { basis: Basis::Z, qubits: zs.clone() });
        if p > 0.0 {
            c.push(Instr::PauliError { pauli: 1, p, qubits: zs });
        }
    }
    if !xs.is_empty() {
        c.push(Instr::Reset { basis: Basis::X, qubits: xs.clone() });
        if p > 0.0 {
            c.push(Instr::PauliError { pauli: 2, p, qubits: xs });
        }
    }
}

/// The transversal readout, its detectors against the last round, and the observable.
fn final_readout(patch: &Patch, c: &mut Vec<Instr>, m: &mut usize, last: &[usize], t: f64, p: f64) {
    let (zs, xs) = data_by_basis(patch);
    c.push(Instr::Tick);
    let mut at = vec![usize::MAX; patch.num_data];
    for (qs, basis, flip) in [(&zs, Basis::Z, 1u8), (&xs, Basis::X, 2u8)] {
        if qs.is_empty() {
            continue;
        }
        if p > 0.0 {
            c.push(Instr::PauliError { pauli: flip, p, qubits: qs.clone() });
        }
        c.push(Instr::Measure { basis, reset: false, flip: 0.0, qubits: qs.clone() });
        for &q in qs {
            at[q as usize] = *m;
            *m += 1;
        }
    }
    for &a in &patch.deterministic {
        let (x, y) = patch.qubit_coords[patch.layers.ancillas[a] as usize];
        let mut recs: Vec<u32> = patch.support[a].iter().map(|&q| lookback(*m, at[q as usize])).collect();
        recs.push(lookback(*m, last[a]));
        c.push(Instr::Detector { coords: vec![x as f64, y as f64, t], recs });
    }
    let recs = patch.observable.iter().map(|&q| lookback(*m, at[q as usize])).collect();
    c.push(Instr::Observable { index: 0, recs });
}

/* -- Current: the engine's own model, gate for gate -------------------------- */

fn current_circuit(patch: &Patch, rounds: usize, p: f64, eta: f64) -> Circuit {
    // sample_pauli in surface_code.rs: Z with eta/(eta+1), X and Y with
    // 1/(2(eta+1)) each, all scaled by p.
    let pz = p * eta / (eta + 1.0);
    let px = p / (2.0 * (eta + 1.0));
    let py = px;
    let na = patch.layers.ancillas.len();
    let pos: HashMap<u32, usize> = patch.layers.ancillas.iter().enumerate().map(|(i, &a)| (a, i)).collect();

    let mut c = Vec::new();
    coords_header(patch, &mut c);
    reset_data(patch, &mut c, 0.0);
    let mut m = 0usize;
    let mut prev: Option<Vec<usize>> = None;
    let total = rounds + 2;
    for r in 0..total {
        c.push(Instr::Tick);
        let noisy = r >= 1 && r <= rounds;
        let mut this = vec![usize::MAX; na];
        for &op in &patch.program {
            match op {
                Op::Reset(q) => c.push(Instr::Reset { basis: Basis::Z, qubits: vec![q as u32] }),
                Op::H(q) => c.push(Instr::H(vec![q as u32])),
                Op::Cnot(a, b) => c.push(Instr::Cx(vec![(a as u32, b as u32)])),
                Op::Cz(a, b) => c.push(Instr::Cz(vec![(a as u32, b as u32)])),
                Op::Noise(q) => {
                    if noisy && p > 0.0 {
                        c.push(Instr::PauliChannel1 { px, py, pz, qubits: vec![q as u32] });
                    }
                }
                Op::Measure(q, _, _) => {
                    let flip = if noisy { p } else { 0.0 };
                    c.push(Instr::Measure { basis: Basis::Z, reset: false, flip, qubits: vec![q as u32] });
                    this[pos[&(q as u32)]] = m;
                    m += 1;
                }
            }
        }
        if let Some(prev) = &prev {
            for a in 0..na {
                let (x, y) = patch.qubit_coords[patch.layers.ancillas[a] as usize];
                c.push(Instr::Detector {
                    coords: vec![x as f64, y as f64, r as f64],
                    recs: vec![lookback(m, this[a]), lookback(m, prev[a])],
                });
            }
        }
        prev = Some(this);
    }
    let last = prev.expect("at least one round");
    final_readout(patch, &mut c, &mut m, &last, total as f64, 0.0);
    Circuit { instrs: c }
}

/* -- SD6: the standard model --------------------------------------------------- */

fn sd6_circuit(patch: &Patch, rounds: usize, p: f64) -> Circuit {
    let na = patch.layers.ancillas.len();
    let nq = patch.num_qubits;
    let data: Vec<u32> = (0..patch.num_data as u32).collect();
    let anc = patch.layers.ancillas.clone();
    let idle = |busy: &[u32]| -> Vec<u32> {
        let mut used = vec![false; nq];
        for &q in busy {
            used[q as usize] = true;
        }
        (0..nq as u32).filter(|&q| !used[q as usize]).collect()
    };
    let depol1 = |c: &mut Vec<Instr>, qubits: Vec<u32>| {
        if p > 0.0 && !qubits.is_empty() {
            c.push(Instr::Depolarize1 { p, qubits });
        }
    };
    let hadamard_layer = |c: &mut Vec<Instr>| {
        c.push(Instr::Tick);
        if !patch.layers.hadamard.is_empty() {
            c.push(Instr::H(patch.layers.hadamard.clone()));
            depol1(c, patch.layers.hadamard.clone());
        }
        depol1(c, idle(&patch.layers.hadamard));
    };

    let mut c = Vec::new();
    coords_header(patch, &mut c);
    let mut m = 0usize;
    let mut prev: Option<Vec<usize>> = None;
    for r in 1..=rounds {
        // Reset: ancillas every round, data in the first; idle data otherwise.
        c.push(Instr::Tick);
        c.push(Instr::Reset { basis: Basis::Z, qubits: anc.clone() });
        if p > 0.0 {
            c.push(Instr::PauliError { pauli: 1, p, qubits: anc.clone() });
        }
        if r == 1 {
            reset_data(patch, &mut c, p);
        } else {
            depol1(&mut c, data.clone());
        }

        hadamard_layer(&mut c);

        for step in &patch.layers.steps {
            c.push(Instr::Tick);
            let cx: Vec<(u32, u32)> = step.iter().filter_map(|g| match *g { Gate2::Cx(a, b) => Some((a, b)), _ => None }).collect();
            let cz: Vec<(u32, u32)> = step.iter().filter_map(|g| match *g { Gate2::Cz(a, b) => Some((a, b)), _ => None }).collect();
            if !cx.is_empty() {
                c.push(Instr::Cx(cx));
            }
            if !cz.is_empty() {
                c.push(Instr::Cz(cz));
            }
            let pairs: Vec<(u32, u32)> = step.iter().map(|g| g.qubits()).collect();
            if p > 0.0 && !pairs.is_empty() {
                c.push(Instr::Depolarize2 { p, pairs: pairs.clone() });
            }
            let busy: Vec<u32> = pairs.iter().flat_map(|&(a, b)| [a, b]).collect();
            depol1(&mut c, idle(&busy));
        }

        hadamard_layer(&mut c);

        // Measure the ancillas; the data idle meanwhile.
        c.push(Instr::Tick);
        if p > 0.0 {
            c.push(Instr::PauliError { pauli: 1, p, qubits: anc.clone() });
        }
        c.push(Instr::Measure { basis: Basis::Z, reset: false, flip: 0.0, qubits: anc.clone() });
        let this: Vec<usize> = (0..na).map(|a| m + a).collect();
        m += na;
        depol1(&mut c, data.clone());

        let coords = |a: usize| {
            let (x, y) = patch.qubit_coords[patch.layers.ancillas[a] as usize];
            vec![x as f64, y as f64, r as f64]
        };
        match &prev {
            None => {
                for &a in &patch.deterministic {
                    c.push(Instr::Detector { coords: coords(a), recs: vec![lookback(m, this[a])] });
                }
            }
            Some(prev) => {
                for a in 0..na {
                    c.push(Instr::Detector { coords: coords(a), recs: vec![lookback(m, this[a]), lookback(m, prev[a])] });
                }
            }
        }
        prev = Some(this);
    }
    let last = prev.expect("at least one round");
    final_readout(patch, &mut c, &mut m, &last, (rounds + 1) as f64, p);
    Circuit { instrs: c }
}
```

`src/lib.rs`: `pub mod memory;`.

- [ ] **Step 5: Run the tests**

Run: `cargo test --release --no-default-features memory::`
Expected: 5 passed, 1 ignored.

If `xzzx_is_the_rotated_code_under_a_checkerboard_of_hadamards` fails but `xzzx_hadamard_pattern` returns `Ok`, the pattern is consistent and only the closed form in the test is wrong: correct the closed form from the printed qubit and keep the test. If the pattern returns `Err`, stop: the equivalence the XZZX memory circuit rests on does not hold.

If a single-fault test fails, **do not weaken it**. Print the failing mechanism's origin (build a one-mechanism circuit, or add a debug print of `m.detectors`/`m.observables` and the prediction) and find the cause. For XZZX under SD6 a failure means a two-qubit fault the transposed schedules do not handle; that is a finding to record in the README, and the fix (a schedule search over the SD6 model, following `xzzx_search_schedules`) belongs in this task.

- [ ] **Step 6: Commit**

```bash
git add src/memory.rs src/surface_code.rs src/lib.rs
git commit -m "feat(memory): surface-code memory experiments as circuits, under the engine's noise and SD6"
```

---

### Task 6: Old path and new path describe the same circuit (check 5)

**Files:**
- Create: `src/equivalence.rs`
- Modify: `src/lib.rs` (`#[cfg(test)] mod equivalence;`)

**Interfaces:**
- Consumes: `CircuitLayout::{propagate, fault_locations}` and `circuit_model::{Fault, StabKind, rounds_executed}` (all `pub(crate)` or `pub`), `RotatedSurfaceCode::{circuit_layout, x_stabilizers, z_stabilizers, simulate_circuit_noise_with_model}`, `XZZXSurfaceCode::{circuit_layout, stabilizers}`, `memory::{patch_for, memory_circuit, NoiseModel, CodeKind}`, `dem::{Dem, pauli_channel_1_independent, xor_prob}`, `dem_decoder::DemDecoder`, `frame_sampler::FrameSampler`.
- Produces: tests only.

- [ ] **Step 1: Write the test module**

```rust
//! Check 5 of the spec: the old per-code path and the new general path describe
//! the same circuit.
//!
//! Every elementary fault the old `CircuitLayout::propagate` enumerates is pushed
//! through the old frame, its detectors are named by (plaquette, round), and its
//! effect on the logical readout is computed from the residual. Merged by symptom
//! with the channel's independent probabilities, that is the old path's detector
//! error model. The new one is built from `memory_circuit(.., Current)`. They must
//! be identical: the same symptoms, and the same probability for each.

use std::collections::HashMap;

use crate::circuit::Basis;
use crate::circuit_model::{rounds_executed, CircuitLayout, Fault, StabKind};
use crate::dem::{pauli_channel_1_independent, xor_prob, Dem};
use crate::memory::{memory_circuit, patch_for, CodeKind, NoiseModel, Patch};
use crate::surface_code::{RotatedSurfaceCode, XZZXSurfaceCode};

type Model = HashMap<(Vec<u32>, u64), f64>;

fn old_path_model(
    layout: &CircuitLayout,
    patch: &Patch,
    stab_coords: &dyn Fn(StabKind, usize) -> (usize, usize),
    rounds: usize,
    p: f64,
    eta: f64,
    index: &HashMap<(i64, i64, i64), u32>,
) -> Model {
    let pz = p * eta / (eta + 1.0);
    let px = p / (2.0 * (eta + 1.0));
    let (qx, qy, qz) = pauli_channel_1_independent(px, px, pz).unwrap();
    let rounds_total = rounds_executed(rounds);
    let mut model = Model::new();
    for r in 1..rounds_total - 1 {
        for (fault, _) in layout.fault_locations() {
            let effect = layout.propagate(fault, r, rounds_total);
            let mut dets = Vec::new();
            for (kind, flips, n) in [
                (StabKind::X, &effect.flips_x_stab, layout.num_x_stabs),
                (StabKind::Z, &effect.flips_z_stab, layout.num_z_stabs),
            ] {
                for t in 1..rounds_total {
                    for s in 0..n {
                        if flips[t * n + s] != flips[(t - 1) * n + s] {
                            let (x, y) = stab_coords(kind, s);
                            let key = (x as i64, y as i64, t as i64);
                            dets.push(*index.get(&key).unwrap_or_else(|| panic!("no detector at {key:?}")));
                        }
                    }
                }
            }
            dets.sort_unstable();
            let mut obs = 0u64;
            for &q in &patch.observable {
                let q = q as usize;
                let hit = match patch.data_basis[q] {
                    Basis::Z => (effect.residual_x >> q) & 1 == 1,
                    Basis::X => (effect.residual_z >> q) & 1 == 1,
                };
                obs ^= hit as u64;
            }
            if dets.is_empty() && obs == 0 {
                continue;
            }
            let prob = match fault {
                Fault::Gate(_, 1) => qx,
                Fault::Gate(_, 2) => qz,
                Fault::Gate(_, _) => qy,
                Fault::Readout(_) => p,
            };
            let e = model.entry((dets, obs)).or_insert(0.0);
            *e = xor_prob(*e, prob);
        }
    }
    model
}

fn new_path_model(dem: &Dem) -> Model {
    let mut model = Model::new();
    for m in &dem.mechanisms {
        let e = model.entry((m.detectors.clone(), m.observables)).or_insert(0.0);
        *e = xor_prob(*e, m.p);
    }
    model
}

fn check(kind: CodeKind, d: usize, basis: Basis) {
    let (p, eta) = (0.003, 0.5);
    let patch = patch_for(kind, d, basis).unwrap();
    let dem = Dem::from_circuit(&memory_circuit(&patch, d, NoiseModel::Current { p, eta })).unwrap();
    let index: HashMap<(i64, i64, i64), u32> = dem
        .detector_coords
        .iter()
        .enumerate()
        .map(|(i, c)| ((c[0] as i64, c[1] as i64, c[2] as i64), i as u32))
        .collect();
    let old = match kind {
        CodeKind::Rotated => {
            let code = RotatedSurfaceCode::new(d);
            let coords = |k: StabKind, s: usize| match k {
                StabKind::X => code.x_stabilizers[s],
                StabKind::Z => code.z_stabilizers[s],
            };
            old_path_model(&code.circuit_layout(), &patch, &coords, d, p, eta, &index)
        }
        CodeKind::Xzzx => {
            let code = XZZXSurfaceCode::new(d);
            let coords = |_: StabKind, s: usize| code.stabilizers[s];
            old_path_model(&code.circuit_layout(), &patch, &coords, d, p, eta, &index)
        }
    };
    let new = new_path_model(&dem);
    let missing: Vec<_> = old.keys().filter(|k| !new.contains_key(*k)).take(3).collect();
    let extra: Vec<_> = new.keys().filter(|k| !old.contains_key(*k)).take(3).collect();
    assert!(missing.is_empty() && extra.is_empty(), "{kind:?} {basis:?} d = {d}: old-only {missing:?}, new-only {extra:?}");
    for (k, &po) in &old {
        let pn = new[k];
        assert!((po - pn).abs() <= 1e-9 * po.max(pn), "{kind:?} {basis:?} d = {d}: {k:?} old {po} new {pn}");
    }
    println!("{kind:?} {basis:?} d = {d}: {} mechanisms identical", old.len());
}

#[test]
fn old_and_new_paths_agree_d3_d5() {
    for d in [3, 5] {
        for kind in [CodeKind::Rotated, CodeKind::Xzzx] {
            for basis in [Basis::Z, Basis::X] {
                check(kind, d, basis);
            }
        }
    }
}

#[test]
#[ignore] // ~1 min; run in the verification step.
fn old_and_new_paths_agree_d7() {
    for kind in [CodeKind::Rotated, CodeKind::Xzzx] {
        for basis in [Basis::Z, Basis::X] {
            check(kind, 7, basis);
        }
    }
}

/// Not an assertion: the two decoders differ (the old one matches with unit
/// weights per Pauli type; the new one weights by probability), so their rates
/// are reported side by side for the README. Rotated only: there the old path's
/// class bit 0 is exactly "the memory-Z observable flipped".
#[test]
#[ignore]
fn current_rates_old_decoder_vs_new_decoder() {
    use crate::dem_decoder::DemDecoder;
    use crate::frame_sampler::FrameSampler;
    use crate::surface_code::Xorshift;
    let shots = 20_000;
    for d in [3, 5, 7] {
        for &p in &[0.002, 0.003, 0.004] {
            let code = RotatedSurfaceCode::new(d);
            let model = crate::circuit_model::build(&code.circuit_layout(), d);
            let mut old_fail = 0;
            for _ in 0..shots {
                let class = code.simulate_circuit_noise_with_model(&model, d, p, 0.5, "zero", 2, 0.0, 0);
                old_fail += (class & 1) as usize;
            }
            let circuit = crate::memory::generate(CodeKind::Rotated, d, d, NoiseModel::Current { p, eta: 0.5 }, Basis::Z).unwrap();
            let dec = DemDecoder::new(&Dem::from_circuit(&circuit).unwrap()).unwrap();
            let sampler = FrameSampler::new(&circuit).unwrap();
            let mut rng = Xorshift::new(0x5eed ^ (d as u64) ^ p.to_bits());
            let (mut new_fail, mut errors) = (0, 0);
            for _ in 0..shots {
                let shot = sampler.sample(&mut rng);
                match dec.decode_bools(&shot.detectors) {
                    Ok(pred) => new_fail += ((pred.observables ^ shot.observables) & 1) as usize,
                    Err(_) => { errors += 1; new_fail += 1; }
                }
            }
            println!(
                "d = {d}, p = {:.1}%: old (unit-weight MWPM) {:.3}%, new (weighted MWPM) {:.3}%, {errors} decode errors, {shots} shots each",
                p * 100.0, old_fail as f64 * 100.0 / shots as f64, new_fail as f64 * 100.0 / shots as f64
            );
        }
    }
}
```

`src/lib.rs`: `#[cfg(test)] mod equivalence;`.

- [ ] **Step 2: Run it**

Run: `cargo test --release --no-default-features equivalence:: -- --nocapture`
Expected: `old_and_new_paths_agree_d3_d5` passes and prints eight "identical" lines. A failure here is a real discrepancy between `round_program`-as-simulated and `memory_circuit(.., Current)`: fix `current_circuit` (not the test) until they agree, because the Current rows of the cross-check stand on this.

- [ ] **Step 3: Run the ignored ones**

Run: `cargo test --release --no-default-features equivalence:: -- --ignored --nocapture 2>&1 | tee /tmp/claude-equivalence.txt` (use the session scratchpad path, not `/tmp`)
Expected: d = 7 identical for all four; nine rate lines printed. Save the rate lines for the README (Task 11).

- [ ] **Step 4: Commit**

```bash
git add src/equivalence.rs src/lib.rs
git commit -m "test: the old per-code circuit and the new general one have the same error model"
```

---

### Task 7: Python bindings

**Files:**
- Create: `src/py_api.rs`
- Modify: `src/lib.rs` (`#[cfg(feature = "python")] mod py_api;` and register in the `#[pymodule]`)

**Interfaces:**
- Consumes: `memory::{generate, CodeKind, NoiseModel}`, `circuit::{Circuit, Basis}`, `dem::Dem`, `dem_decoder::DemDecoder`, `frame_sampler::FrameSampler`, `shots::{pack_row, read_b8, write_01}`, `surface_code::Xorshift`.
- Produces (Python, module `stabilizer_qec`):
  - `generate_circuit(code: str, d, rounds, noise: str, p, eta=0.5, basis="z") -> str` — `code` ∈ {"rotated","xzzx"}, `noise` ∈ {"current","sd6"}, `basis` ∈ {"z","x"}.
  - `dem_from_circuit(circuit_text, decompose=False) -> str`
  - `decode_b8(dem_text, packed: bytes, num_shots) -> (preds: bytes u64le, weights: bytes f64le, decode_errors: int, seconds: float)`
  - `decode_b8_own(circuit_text, packed, num_shots) -> same`
  - `sample_b8(circuit_text, num_shots, seed) -> (dets: bytes b8, obs: bytes b8)`
  - `b8_to_01(packed: bytes, num_bits) -> str`

- [ ] **Step 1: Implement `src/py_api.rs`**

```rust
//! Python bindings for the cross-check harness (`tools/xcheck.py`).
//!
//! Shots cross the boundary as bytes in Stim's b8 layout (numpy's
//! `packbits(..., bitorder="little")`), and predictions come back as
//! little-endian u64 and f64 arrays, so no numpy crate is needed on this side.

use std::time::Instant;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

use crate::circuit::{Basis, Circuit};
use crate::dem::Dem;
use crate::dem_decoder::DemDecoder;
use crate::frame_sampler::FrameSampler;
use crate::memory::{generate, CodeKind, NoiseModel};
use crate::shots::{pack_row, read_b8, write_01};
use crate::surface_code::Xorshift;

fn err(e: String) -> PyErr {
    PyValueError::new_err(e)
}

#[pyfunction]
#[pyo3(signature = (code, d, rounds, noise, p, eta=0.5, basis="z"))]
fn generate_circuit(code: &str, d: usize, rounds: usize, noise: &str, p: f64, eta: f64, basis: &str) -> PyResult<String> {
    let kind = match code {
        "rotated" => CodeKind::Rotated,
        "xzzx" => CodeKind::Xzzx,
        _ => return Err(err(format!("unknown code '{code}'"))),
    };
    let noise = match noise {
        "current" => NoiseModel::Current { p, eta },
        "sd6" => NoiseModel::Sd6 { p },
        _ => return Err(err(format!("unknown noise model '{noise}'"))),
    };
    let basis = match basis {
        "z" => Basis::Z,
        "x" => Basis::X,
        _ => return Err(err(format!("unknown basis '{basis}'"))),
    };
    Ok(generate(kind, d, rounds, noise, basis).map_err(err)?.to_stim())
}

#[pyfunction]
#[pyo3(signature = (circuit_text, decompose=false))]
fn dem_from_circuit(circuit_text: &str, decompose: bool) -> PyResult<String> {
    let c = Circuit::parse(circuit_text).map_err(err)?;
    Ok(Dem::from_circuit(&c).map_err(err)?.to_stim(decompose))
}

type Decoded<'py> = (Bound<'py, PyBytes>, Bound<'py, PyBytes>, usize, f64);

fn decode_packed<'py>(py: Python<'py>, dem: &Dem, packed: &[u8], num_shots: usize) -> PyResult<Decoded<'py>> {
    let decoder = DemDecoder::new(dem).map_err(err)?;
    let nd = dem.num_detectors;
    let stride = nd.div_ceil(8);
    if packed.len() != stride * num_shots {
        return Err(err(format!("{} bytes is not {num_shots} shots of {nd} detectors", packed.len())));
    }
    let mut preds = Vec::with_capacity(8 * num_shots);
    let mut weights = Vec::with_capacity(8 * num_shots);
    let mut errors = 0usize;
    let mut defects = Vec::new();
    let start = Instant::now();
    for s in 0..num_shots {
        defects.clear();
        let row = &packed[s * stride..(s + 1) * stride];
        for i in 0..nd {
            if (row[i / 8] >> (i % 8)) & 1 == 1 {
                defects.push(i as u32);
            }
        }
        match decoder.decode(&defects) {
            Ok(pred) => {
                preds.extend_from_slice(&pred.observables.to_le_bytes());
                weights.extend_from_slice(&pred.weight.to_le_bytes());
            }
            Err(_) => {
                errors += 1;
                preds.extend_from_slice(&u64::MAX.to_le_bytes());
                weights.extend_from_slice(&f64::NAN.to_le_bytes());
            }
        }
    }
    let seconds = start.elapsed().as_secs_f64();
    Ok((PyBytes::new_bound(py, &preds), PyBytes::new_bound(py, &weights), errors, seconds))
}

/// Decode with a model someone else wrote, pieces and all (Stim's decomposed DEM).
#[pyfunction]
fn decode_b8<'py>(py: Python<'py>, dem_text: &str, packed: &[u8], num_shots: usize) -> PyResult<Decoded<'py>> {
    let dem = Dem::parse(dem_text).map_err(err)?;
    decode_packed(py, &dem, packed, num_shots)
}

/// Decode with this engine's own model of the circuit, and its own decomposition.
#[pyfunction]
fn decode_b8_own<'py>(py: Python<'py>, circuit_text: &str, packed: &[u8], num_shots: usize) -> PyResult<Decoded<'py>> {
    let c = Circuit::parse(circuit_text).map_err(err)?;
    let dem = Dem::from_circuit(&c).map_err(err)?;
    decode_packed(py, &dem, packed, num_shots)
}

#[pyfunction]
fn sample_b8<'py>(py: Python<'py>, circuit_text: &str, num_shots: usize, seed: u64) -> PyResult<(Bound<'py, PyBytes>, Bound<'py, PyBytes>)> {
    let c = Circuit::parse(circuit_text).map_err(err)?;
    let sampler = FrameSampler::new(&c).map_err(err)?;
    let mut rng = Xorshift::new(seed);
    let no = sampler.num_observables();
    let mut dets = Vec::new();
    let mut obs = Vec::new();
    for _ in 0..num_shots {
        let shot = sampler.sample(&mut rng);
        pack_row(&shot.detectors, &mut dets);
        let bits: Vec<bool> = (0..no).map(|k| (shot.observables >> k) & 1 == 1).collect();
        pack_row(&bits, &mut obs);
    }
    Ok((PyBytes::new_bound(py, &dets), PyBytes::new_bound(py, &obs)))
}

#[pyfunction]
fn b8_to_01(packed: &[u8], num_bits: usize) -> PyResult<String> {
    Ok(write_01(&read_b8(packed, num_bits).map_err(err)?))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(generate_circuit, m)?)?;
    m.add_function(wrap_pyfunction!(dem_from_circuit, m)?)?;
    m.add_function(wrap_pyfunction!(decode_b8, m)?)?;
    m.add_function(wrap_pyfunction!(decode_b8_own, m)?)?;
    m.add_function(wrap_pyfunction!(sample_b8, m)?)?;
    m.add_function(wrap_pyfunction!(b8_to_01, m)?)?;
    Ok(())
}
```

`src/lib.rs`: add `#[cfg(feature = "python")] mod py_api;` and inside `fn stabilizer_qec(...)` add `py_api::register(m)?;` before `Ok(())`.

- [ ] **Step 2: Build into the virtualenv and smoke-test**

Run:
```bash
cargo build --release
VIRTUAL_ENV=$PWD/.venv .venv/bin/maturin develop --release
.venv/bin/python -c "
import stabilizer_qec as sq, stim
t = sq.generate_circuit('rotated', 3, 3, 'sd6', 0.003)
c = stim.Circuit(t)
print(c.num_detectors, c.num_observables, len(sq.dem_from_circuit(t)))
"
```
Expected: `24 1 <some length>`, and Stim parses our circuit without complaint. If maturin needs a `pyproject.toml`, add the minimal one (`[build-system] requires = ["maturin>=1,<2"] build-backend = "maturin"`, `[project] name = "stabilizer_qec"`, `requires-python = ">=3.9"`) and commit it with this task.

- [ ] **Step 3: Commit**

```bash
git add src/py_api.rs src/lib.rs
git commit -m "feat(python): bindings for the cross-check: circuits, models, sampling and decoding"
```

---

### Task 8: The cross-check harness and the reference data (checks 1–4)

**Files:**
- Create: `tools/xcheck.py`, `data/xcheck/` (generated)

**Interfaces:**
- Consumes: the Task 7 bindings; `stim`, `pymatching`, `numpy` from `.venv`.
- Produces: `data/xcheck/reference.json`:

```json
{
  "generated": "YYYY-MM-DD", "stim": "1.16.0", "pymatching": "2.4.0", "command": "...",
  "circuits": [
    {"name": "stim-rotated-memory-z-d3", "source": "stim", "d": 3, "rounds": 3, "p": 0.003,
     "detectors": 24, "mechanisms": 219, "bytes": 8889, "dem_sha256": "...",
     "ours": 219, "missing": 0, "extra": 0, "differing": 0, "max_rel": 2.1e-16},
    {"name": "rotated-sd6-d3", "source": "ours", "code": "rotated", "noise": "sd6", "d": 3, "rounds": 3,
     "p": 0.003, "eta": 0.5, "basis": "z", "circuit_sha256": "...", "...": "same fields"}
  ],
  "decoding": [
    {"code": "rotated", "noise": "sd6", "d": 3, "p": 0.003, "shots": 100000,
     "pymatching_failures": 0, "ours_failures": 0, "ours_own_dem_failures": 0,
     "disagreements": 0, "non_ties": 0, "weight_noise": 0.0,
     "pymatching_us": 0.0, "ours_us": 0.0,
     "sampler": {"chi2": 0.0, "dof": 0, "z": 0.0, "our_failures": 0, "our_decode_errors": 0},
     "decode_errors": 0}
  ]
}
```

plus `data/xcheck/<name>.dem.txt` (Stim's undecomposed, flattened DEM for every circuit) and `data/xcheck/stim-rotated-memory-z-d{3,5,7}.stim.txt`.

- [ ] **Step 1: Write `tools/xcheck.py`**

```python
"""Cross-check this engine against Stim and PyMatching.

Run from the repository root, with the engine built into the virtualenv:

    VIRTUAL_ENV=$PWD/.venv .venv/bin/maturin develop --release
    .venv/bin/python tools/xcheck.py              # full run; writes data/xcheck/
    .venv/bin/python tools/xcheck.py --quick      # a smoke test; writes nothing

Checks, strictest first (spec: docs/superpowers/specs/2026-09-24-stim-foundation-design.md):
  1. Detector error models are identical: every mechanism, to 1e-9 relative.
  2. Decoders agree shot for shot on Stim's samples; every disagreement is a tie.
  3. Samplers agree: per-detector firing rates, and logical error rates.
  4. Speed, reported as measured.
Exits non-zero if check 1, 2 or 3 fails.
"""

import argparse
import datetime
import hashlib
import json
import math
import pathlib
import sys
import time

import numpy as np
import pymatching
import stim

import stabilizer_qec as sq

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "xcheck"
P_MODEL = 0.003
TOL = 1e-9


def sha256(text):
    return hashlib.sha256(text.encode()).hexdigest()


def stim_generated(d):
    return stim.Circuit.generated(
        "surface_code:rotated_memory_z", distance=d, rounds=d,
        after_clifford_depolarization=P_MODEL, before_round_data_depolarization=P_MODEL,
        before_measure_flip_probability=P_MODEL, after_reset_flip_probability=P_MODEL)


def mechanisms(dem):
    """{(sorted detectors, observable mask): p}, identical symptoms merged."""
    out = {}
    for inst in dem.flattened():
        if inst.type != "error":
            continue
        dets, obs = set(), 0
        for t in inst.targets_copy():
            if t.is_relative_detector_id():
                dets ^= {t.val}
            elif t.is_logical_observable_id():
                obs ^= 1 << t.val
        key = (tuple(sorted(dets)), obs)
        p = inst.args_copy()[0]
        q = out.get(key, 0.0)
        out[key] = q * (1 - p) + p * (1 - q)
    return out


def check_model(entry, circuit_text):
    """Check 1 for one circuit. Returns the entry with its comparison filled in."""
    theirs = stim.Circuit(circuit_text).detector_error_model(decompose_errors=False).flattened()
    ours = stim.DetectorErrorModel(sq.dem_from_circuit(circuit_text, False))
    a, b = mechanisms(ours), mechanisms(theirs)
    rels = [abs(a[k] - b[k]) / max(a[k], b[k]) for k in a if k in b]
    text = str(theirs)
    entry.update(
        detectors=theirs.num_detectors, mechanisms=len(b), ours=len(a),
        missing=sum(1 for k in b if k not in a), extra=sum(1 for k in a if k not in b),
        differing=sum(1 for r in rels if r > TOL), max_rel=max(rels) if rels else 0.0,
        bytes=len(text.encode()), dem_sha256=sha256(text))
    return entry, text


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (mid - half, mid + half)


def unpack(packed, shots, bits):
    rows = np.frombuffer(packed, dtype=np.uint8).reshape(shots, -1)
    return np.unpackbits(rows, axis=1, bitorder="little")[:, :bits].astype(bool)


def check_decoding(code, d, p, shots, seed):
    """Checks 2, 3 and 4 at one point."""
    text = sq.generate_circuit(code, d, d, "sd6", p, 0.5, "z")
    circuit = stim.Circuit(text)
    dem = circuit.detector_error_model(decompose_errors=True)
    matching = pymatching.Matching.from_detector_error_model(dem)
    dets, obs = circuit.compile_detector_sampler(seed=seed).sample(shots, separate_observables=True)
    actual = obs[:, 0].astype(np.uint64)
    packed = np.packbits(dets, axis=1, bitorder="little").tobytes()

    t = time.perf_counter()
    pm = matching.decode_batch(dets)[:, 0].astype(np.uint64)
    pm_seconds = time.perf_counter() - t

    pred_b, w_b, errors, our_seconds = sq.decode_b8(str(dem), packed, shots)
    ours_raw = np.frombuffer(pred_b, dtype="<u8")
    ours = ours_raw & np.uint64(1)
    our_w = np.frombuffer(w_b, dtype="<f8")
    failed = ours_raw == np.uint64(2**64 - 1)

    # Check 2: every disagreement must be a tie. The weight noise between two
    # exact matchers that discretise weights differently is measured on shots
    # where they agree, and a disagreement is a tie if it sits inside that noise.
    agree = np.nonzero((pm == ours) & ~failed)[0][:500]
    noise = 0.0
    for i in agree:
        _, w = matching.decode(dets[i], return_weight=True)
        noise = max(noise, abs(w - our_w[i]))
    tie_tol = max(10 * noise, 1e-6)
    disagree = np.nonzero((pm != ours) & ~failed)[0]
    non_ties = 0
    for i in disagree:
        _, w = matching.decode(dets[i], return_weight=True)
        if abs(w - our_w[i]) > tie_tol:
            non_ties += 1

    own_b, _, own_errors, _ = sq.decode_b8_own(text, packed, shots)
    own = np.frombuffer(own_b, dtype="<u8")
    own_fail = int(np.count_nonzero((own & np.uint64(1)) != actual) )

    # Check 3: our sampler on the same circuit text, decoded by our decoder.
    od_b, oo_b = sq.sample_b8(text, shots, seed + 1)
    od = unpack(od_b, shots, circuit.num_detectors)
    oo = unpack(oo_b, shots, 1)[:, 0].astype(np.uint64)
    a, b = dets.sum(axis=0).astype(float), od.sum(axis=0).astype(float)
    mask = (a + b) > 0
    chi2 = float(((a - b) ** 2 / np.where(mask, a + b, 1))[mask].sum())
    dof = int(mask.sum())
    z = (chi2 - dof) / math.sqrt(2 * dof) if dof else 0.0
    s_pred_b, _, s_errors, _ = sq.decode_b8_own(text, np.packbits(od, axis=1, bitorder="little").tobytes(), shots)
    s_pred = np.frombuffer(s_pred_b, dtype="<u8") & np.uint64(1)

    return dict(
        code=code, noise="sd6", d=d, p=p, shots=shots,
        pymatching_failures=int(np.count_nonzero(pm != actual)),
        ours_failures=int(np.count_nonzero((ours != actual) | failed)),
        ours_own_dem_failures=own_fail + 0 * own_errors,
        disagreements=int(len(disagree)), non_ties=non_ties, weight_noise=noise,
        decode_errors=int(errors), own_decode_errors=int(own_errors),
        pymatching_us=pm_seconds / shots * 1e6, ours_us=our_seconds / shots * 1e6,
        sampler=dict(chi2=chi2, dof=dof, z=z,
                     our_failures=int(np.count_nonzero(s_pred != oo)), our_decode_errors=int(s_errors)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="few shots, d <= 5, write nothing")
    ap.add_argument("--shots", type=int, default=100_000)
    args = ap.parse_args()
    shots = 2_000 if args.quick else args.shots
    ds = (3, 5) if args.quick else (3, 5, 7)
    ok = True

    circuits, files = [], {}
    print("Check 1: detector error models")
    for d in ds:
        name = f"stim-rotated-memory-z-d{d}"
        text = str(stim_generated(d))
        entry, dem_text = check_model(dict(name=name, source="stim", d=d, rounds=d, p=P_MODEL), text)
        circuits.append(entry)
        files[f"{name}.stim.txt"] = text
        files[f"{name}.dem.txt"] = dem_text
    for code in ("rotated", "xzzx"):
        for noise in ("current", "sd6"):
            for d in ds:
                name = f"{code}-{noise}-d{d}"
                text = sq.generate_circuit(code, d, d, noise, P_MODEL, 0.5, "z")
                entry, dem_text = check_model(
                    dict(name=name, source="ours", code=code, noise=noise, d=d, rounds=d, p=P_MODEL,
                         eta=0.5, basis="z", circuit_sha256=sha256(text)), text)
                circuits.append(entry)
                files[f"{name}.dem.txt"] = dem_text
    for c in circuits:
        good = c["missing"] == c["extra"] == c["differing"] == 0
        ok &= good
        print(f"  {'ok ' if good else 'BAD'} {c['name']:<28} {c['detectors']:>5} detectors  "
              f"{c['ours']:>6} / {c['mechanisms']:<6} mechanisms  max rel {c['max_rel']:.1e}")

    print(f"Checks 2-4: decoding, {shots:,} shots per point")
    decoding = []
    for d in ds:
        for p in (0.003, 0.006):
            r = check_decoding("rotated", d, p, shots, seed=1000 * d + int(p * 1e4))
            decoding.append(r)
            lo_pm = wilson(r["pymatching_failures"], shots)
            lo_us = wilson(r["sampler"]["our_failures"], shots)
            overlap = lo_pm[0] <= lo_us[1] and lo_us[0] <= lo_pm[1]
            good = r["non_ties"] == 0 and r["decode_errors"] == 0 and abs(r["sampler"]["z"]) < 4 and overlap
            ok &= good
            print(f"  {'ok ' if good else 'BAD'} d={d} p={p:.3f}  PyMatching {r['pymatching_failures'] / shots:.4%}  "
                  f"ours {r['ours_failures'] / shots:.4%}  own-DEM {r['ours_own_dem_failures'] / shots:.4%}  "
                  f"our sampler {r['sampler']['our_failures'] / shots:.4%}  "
                  f"disagree {r['disagreements']} (non-ties {r['non_ties']}, weight noise {r['weight_noise']:.2e})  "
                  f"chi2 z {r['sampler']['z']:+.2f}  "
                  f"{r['pymatching_us']:.1f} us vs {r['ours_us']:.1f} us")

    if not args.quick:
        OUT.mkdir(parents=True, exist_ok=True)
        for name, text in files.items():
            (OUT / name).write_text(text)
        reference = dict(
            generated=datetime.date.today().isoformat(), stim=stim.__version__,
            pymatching=pymatching.__version__, command="python tools/xcheck.py",
            circuits=circuits, decoding=decoding)
        (OUT / "reference.json").write_text(json.dumps(reference, indent=1) + "\n")
        print(f"Wrote {OUT.relative_to(ROOT)}/ ({len(files)} files + reference.json)")
    print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Quick run**

Run: `.venv/bin/python tools/xcheck.py --quick`
Expected: every line `ok`, "ALL CHECKS PASSED". Any `BAD` in check 1 is a physics disagreement with Stim: diff the two DEMs for that circuit (the smallest failing distance first; print a few `missing`/`extra` keys) and fix the engine, never the tolerance. Record whatever was found for the README's defect write-ups.

- [ ] **Step 3: Full run**

Run in the background: `.venv/bin/python tools/xcheck.py > <scratchpad>/xcheck-full.txt 2>&1`
Expected: "ALL CHECKS PASSED" and the files in `data/xcheck/`. Keep the printed report for the README.

- [ ] **Step 4: Commit**

```bash
git add tools/xcheck.py data/xcheck
git commit -m "feat(xcheck): the engine against Stim and PyMatching, and the recorded reference"
```

---

### Task 9: WASM exports and the engine wrapper

**Files:**
- Create: `src/wasm_xc.rs`
- Modify: `src/lib.rs` (`#[cfg(not(feature = "python"))] mod wasm_xc;`; route `noise_mode == 3` in `wasm_run_benchmark`), `js/engine.js`, `js/worker.js`, `stabilizer_qec.wasm` (rebuilt)

**Interfaces:**
- Consumes: `memory::{generate, CodeKind, NoiseModel}`, `circuit::{Basis, Circuit}`, `dem::{Dem, compare}`, `dem_decoder::DemDecoder`, `frame_sampler::FrameSampler`, `surface_code::next_shot_rng`.
- Produces (WASM): `wasm_text_buf(len) -> *mut u8`, `wasm_text_ptr() -> *const u8`, `wasm_xc_generate(code, d, rounds, noise, p, eta, basis) -> usize` (text length; circuit left in the slot; text starts `ERROR:` on failure), `wasm_xc_load_circuit() -> usize` (JSON length), `wasm_xc_compare() -> usize` (JSON length), `wasm_xc_run(code, d, rounds, noise, p, eta, runs, decode) -> f64` (failure rate), `wasm_xc_decode_errors() -> usize`.
- Produces (JS, `engine.js`): `NOISE.SD6 = 3`, `NOISE_NAME[3]`, `XC_NOISE = { CURRENT: 0, SD6: 1 }`, `BASIS = { Z: 0, X: 1 }`, `xcGenerate(instance, cfg) -> string`, `xcLoadCircuit(instance, text) -> summary`, `xcCompare(instance, demText) -> summary`, `xcTiming(instance, cfg, runs) -> { decodeMicros, sampleMicros, runs }`; `runBenchmark` result gains `decodeErrors`.
- Produces (worker): ops `xcheck({ rows }) -> { rows }` with progress per row, and `xctiming({ cfg, runs })`.

- [ ] **Step 1: Implement `src/wasm_xc.rs`**

```rust
//! WebAssembly surface for the general path: the live Stim comparison in
//! section 10 and the SD6 option in sections 07 and 09.
//!
//! Text crosses the boundary through one byte buffer. JS asks for room with
//! `wasm_text_buf`, writes UTF-8 into it and calls an export; the export writes
//! its reply (circuit text or a JSON summary) back into the same buffer and
//! returns its length. One buffer, single-threaded, the same pattern as the
//! matching cost matrix.

use std::fmt::Write as _;

use crate::circuit::{Basis, Circuit};
use crate::dem::{compare, Dem};
use crate::dem_decoder::DemDecoder;
use crate::frame_sampler::FrameSampler;
use crate::memory::{generate, CodeKind, NoiseModel};

static mut TEXT: Vec<u8> = Vec::new();
static mut SLOT: Option<Circuit> = None;
static mut DECODE_ERRORS: usize = 0;
/// (code, d, rounds, noise, p bits, eta bits) → the compiled sampler and decoder.
#[allow(clippy::type_complexity)]
static mut CACHE: Option<((usize, usize, usize, usize, u64, u64), FrameSampler, DemDecoder)> = None;

fn text() -> &'static mut Vec<u8> {
    unsafe { &mut *std::ptr::addr_of_mut!(TEXT) }
}

fn reply(s: &str) -> usize {
    let t = text();
    t.clear();
    t.extend_from_slice(s.as_bytes());
    t.len()
}

fn json_error(e: &str) -> usize {
    let escaped = e.replace('\\', "\\\\").replace('"', "\\\"").replace('\n', " ");
    reply(&format!("{{\"ok\":false,\"error\":\"{escaped}\"}}"))
}

fn noise_model(noise: usize, p: f64, eta: f64) -> NoiseModel {
    if noise == 1 { NoiseModel::Sd6 { p } } else { NoiseModel::Current { p, eta } }
}

fn code_kind(code: usize) -> CodeKind {
    if code == 1 { CodeKind::Xzzx } else { CodeKind::Rotated }
}

#[no_mangle]
pub extern "C" fn wasm_text_buf(len: usize) -> *mut u8 {
    let t = text();
    t.clear();
    t.resize(len, 0);
    t.as_mut_ptr()
}

#[no_mangle]
pub extern "C" fn wasm_text_ptr() -> *const u8 {
    text().as_ptr()
}

/// Generate a memory circuit into the slot and reply with its Stim text.
#[no_mangle]
pub extern "C" fn wasm_xc_generate(code: usize, d: usize, rounds: usize, noise: usize, p: f64, eta: f64, basis: usize) -> usize {
    let basis = if basis == 1 { Basis::X } else { Basis::Z };
    match generate(code_kind(code), d, rounds, noise_model(noise, p, eta), basis) {
        Ok(c) => {
            let s = c.to_stim();
            unsafe { *std::ptr::addr_of_mut!(SLOT) = Some(c) };
            reply(&s)
        }
        Err(e) => reply(&format!("ERROR: {e}")),
    }
}

/// Parse the buffer as a circuit into the slot.
#[no_mangle]
pub extern "C" fn wasm_xc_load_circuit() -> usize {
    let src = String::from_utf8_lossy(text()).into_owned();
    match Circuit::parse(&src) {
        Ok(c) => {
            unsafe { *std::ptr::addr_of_mut!(SLOT) = Some(c) };
            reply("{\"ok\":true}")
        }
        Err(e) => json_error(&e),
    }
}

/// Parse the buffer as Stim's DEM, build ours from the slot, and compare.
#[no_mangle]
pub extern "C" fn wasm_xc_compare() -> usize {
    let src = String::from_utf8_lossy(text()).into_owned();
    let theirs = match Dem::parse(&src) {
        Ok(d) => d,
        Err(e) => return json_error(&format!("reading Stim's model: {e}")),
    };
    let circuit = match unsafe { (*std::ptr::addr_of!(SLOT)).as_ref() } {
        Some(c) => c,
        None => return json_error("no circuit loaded"),
    };
    let ours = match Dem::from_circuit(circuit) {
        Ok(d) => d,
        Err(e) => return json_error(&e),
    };
    let c = compare(&ours, &theirs, 1e-9);
    let mut s = String::new();
    let _ = write!(
        s,
        "{{\"ok\":true,\"detectors\":{},\"ours\":{},\"theirs\":{},\"missing\":{},\"extra\":{},\"differing\":{},\"maxRel\":{:e}}}",
        ours.num_detectors, c.ours, c.theirs, c.missing, c.extra, c.differing, c.max_rel
    );
    reply(&s)
}

/// Sample `runs` shots of a memory-Z experiment on the general path and decode
/// them (or only sample, when `decode` is 0, so JS can time the two apart).
/// Returns the failure rate; a shot that fails to decode counts as a failure
/// and is also counted in `wasm_xc_decode_errors`.
#[no_mangle]
pub extern "C" fn wasm_xc_run(code: usize, d: usize, rounds: usize, noise: usize, p: f64, eta: f64, runs: usize, decode: u32) -> f64 {
    let key = (code, d, rounds, noise, p.to_bits(), eta.to_bits());
    let cache = unsafe { &mut *std::ptr::addr_of_mut!(CACHE) };
    if cache.as_ref().map(|c| c.0) != Some(key) {
        let built = generate(code_kind(code), d, rounds, noise_model(noise, p, eta), Basis::Z).and_then(|c| {
            let dem = Dem::from_circuit(&c)?;
            Ok((key, FrameSampler::new(&c)?, DemDecoder::new(&dem)?))
        });
        match built {
            Ok(b) => *cache = Some(b),
            Err(_) => {
                unsafe { *std::ptr::addr_of_mut!(DECODE_ERRORS) = runs };
                return f64::NAN;
            }
        }
    }
    let (_, sampler, decoder) = cache.as_ref().unwrap();
    let mut rng = crate::surface_code::next_shot_rng();
    let (mut failures, mut errors) = (0usize, 0usize);
    for _ in 0..runs {
        let shot = sampler.sample(&mut rng);
        if decode == 0 {
            continue;
        }
        match decoder.decode_bools(&shot.detectors) {
            Ok(pred) => failures += ((pred.observables ^ shot.observables) & 1) as usize,
            Err(_) => {
                errors += 1;
                failures += 1;
            }
        }
    }
    unsafe { *std::ptr::addr_of_mut!(DECODE_ERRORS) = errors };
    if runs == 0 { 0.0 } else { failures as f64 / runs as f64 }
}

#[no_mangle]
pub extern "C" fn wasm_xc_decode_errors() -> usize {
    unsafe { *std::ptr::addr_of!(DECODE_ERRORS) }
}
```

In `src/lib.rs`: add `#[cfg(not(feature = "python"))] mod wasm_xc;`, and at the top of `wasm_run_benchmark`'s body:

```rust
    // Noise mode 3 is SD6, which only the general path models.
    #[cfg(not(feature = "python"))]
    if noise_mode == 3 {
        return wasm_xc::wasm_xc_run(code_type, d, num_rounds, 1, p, bias, num_runs, 1);
    }
```

- [ ] **Step 2: Rebuild the WASM and check the exports**

Run:
```bash
cargo build --release --target wasm32-unknown-unknown --no-default-features
cp target/wasm32-unknown-unknown/release/stabilizer_qec.wasm .
node -e "
const fs=require('fs');WebAssembly.instantiate(fs.readFileSync('stabilizer_qec.wasm'),{}).then(({instance:i})=>{
const e=i.exports; console.log(['wasm_text_buf','wasm_xc_generate','wasm_xc_compare','wasm_xc_run','wasm_xc_decode_errors'].map(n=>n+':'+typeof e[n]).join(' '));
const len=e.wasm_xc_generate(0,3,3,1,0.003,0.5,0); console.log('circuit bytes',len);
const t0=Date.now(); const r=e.wasm_run_benchmark(5,0,2,0.006,0.5,5,2000,3,0,0); console.log('SD6 d5 p=0.6%',r,'errors',e.wasm_xc_decode_errors(),(Date.now()-t0)/2000,'ms/shot');});"
```
Expected: every export is `function`; circuit bytes > 0; a rate between 0.1% and 10%; errors 0. Note ms/shot for Task 10's shot counts.

- [ ] **Step 3: `js/engine.js`**

Replace the `NOISE` and `NOISE_NAME` constants with:

```js
export const NOISE = { DATA: 0, PHENOM: 1, CIRCUIT: 2, SD6: 3 };

export const NOISE_NAME = {
  0: 'Data noise only',
  1: 'Phenomenological',
  2: 'Circuit-level',
  3: 'Circuit-level (SD6)',
};

/** wasm_xc_* noise argument: the engine's own circuit-level model, or SD6. */
export const XC_NOISE = { CURRENT: 0, SD6: 1 };

/** Memory-experiment basis. */
export const BASIS = { Z: 0, X: 1 };
```

In `runBenchmark`, after `const seconds = ...`, add

```js
  // SD6 runs on the general path, whose decoder refuses rather than falls back;
  // the count of refusals travels with the rate so the page can show it.
  const decodeErrors = c.noiseMode === NOISE.SD6 ? instance.exports.wasm_xc_decode_errors() : 0;
```

and include `decodeErrors` in the returned object.

Append:

```js
/* -- The general path ---------------------------------------------------- */

function writeText(instance, text) {
  const bytes = new TextEncoder().encode(text);
  const ptr = instance.exports.wasm_text_buf(bytes.length);
  new Uint8Array(instance.exports.memory.buffer, ptr, bytes.length).set(bytes);
}

function readText(instance, len) {
  const ptr = instance.exports.wasm_text_ptr();
  // Copy out before anything else can grow memory and detach the view.
  return new TextDecoder().decode(new Uint8Array(instance.exports.memory.buffer, ptr, len).slice());
}

/**
 * Generate a memory circuit, leave it loaded for xcCompare, and return its
 * Stim text.
 * @param {{codeType:number, d:number, rounds:number, noise:number, p:number, eta?:number, basis?:number}} cfg
 */
export function xcGenerate(instance, { codeType, d, rounds, noise, p, eta = 0.5, basis = BASIS.Z }) {
  const len = instance.exports.wasm_xc_generate(codeType, d, rounds, noise, p, eta, basis);
  const text = readText(instance, len);
  if (text.startsWith('ERROR:')) throw new Error(text.slice(7));
  return text;
}

/** Load a circuit from Stim text for xcCompare. */
export function xcLoadCircuit(instance, text) {
  writeText(instance, text);
  const out = JSON.parse(readText(instance, instance.exports.wasm_xc_load_circuit()));
  if (!out.ok) throw new Error(out.error);
  return out;
}

/** Build our model of the loaded circuit and compare it with Stim's DEM text. */
export function xcCompare(instance, demText) {
  writeText(instance, demText);
  return JSON.parse(readText(instance, instance.exports.wasm_xc_compare()));
}

/**
 * Time the decoder alone: sample-only and sample-and-decode over the same
 * number of shots, the difference being the decoding. The engine has no clock
 * of its own, so the timing has to happen out here.
 */
export function xcTiming(instance, cfg, runs) {
  const { codeType, d, rounds, noise, p, eta = 0.5 } = cfg;
  const run = (decode) => {
    const t0 = performance.now();
    instance.exports.wasm_xc_run(codeType, d, rounds, noise, p, eta, runs, decode);
    return performance.now() - t0;
  };
  run(0); // build and cache the model outside the timed calls
  const sample = run(0);
  const total = run(1);
  return { runs, sampleMicros: (sample * 1000) / runs, decodeMicros: (Math.max(0, total - sample) * 1000) / runs };
}
```

- [ ] **Step 4: `js/worker.js`**

Import `xcGenerate, xcLoadCircuit, xcCompare, xcTiming` from `./engine.js`. Add to `OPS`:

```js
  /**
   * Section 10's comparison with Stim: for each row, fetch Stim's model, build
   * ours for the same circuit, compare. A row whose reference cannot be
   * fetched says so; nothing is ever filled in from anywhere else.
   */
  async xcheck(instance, { rows }, report) {
    const out = [];
    for (const row of rows) {
      let result;
      try {
        let demText;
        try {
          demText = await fetchText(row.demUrl);
        } catch (error) {
          result = { ok: false, unavailable: true, error: error.message };
        }
        if (!result) {
          let stale = false;
          if (row.kind === 'stim') {
            xcLoadCircuit(instance, await fetchText(row.circuitUrl));
          } else {
            const text = xcGenerate(instance, row.gen);
            if (row.circuitSha256) stale = (await sha256Hex(text)) !== row.circuitSha256;
          }
          const t0 = performance.now();
          result = { ...xcCompare(instance, demText), ms: performance.now() - t0, stale };
        }
      } catch (error) {
        result = { ok: false, error: error.message };
      }
      out.push({ key: row.key, ...result });
      report({ done: out.length, total: rows.length, row: { key: row.key, ...result } });
      await new Promise((resolve) => setTimeout(resolve, 0));
    }
    return { rows: out };
  },

  async xctiming(instance, { cfg, runs }) {
    return xcTiming(instance, cfg, runs);
  },
```

and at module level:

```js
async function fetchText(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
  return response.text();
}

async function sha256Hex(text) {
  const digest = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(text));
  return [...new Uint8Array(digest)].map((b) => b.toString(16).padStart(2, '0')).join('');
}
```

- [ ] **Step 5: Verify end to end in Node**

Run a Node check that loads the WASM the way the worker would and compares against the committed Stim reference:

```bash
node --input-type=module -e "
import { readFile } from 'node:fs/promises';
import { xcGenerate, xcCompare, xcLoadCircuit, runBenchmark, NOISE } from './js/engine.js';
const { instance } = await WebAssembly.instantiate(await readFile('stabilizer_qec.wasm'), {});
xcGenerate(instance, { codeType: 0, d: 5, rounds: 5, noise: 1, p: 0.003 });
console.log(xcCompare(instance, await readFile('data/xcheck/rotated-sd6-d5.dem.txt', 'utf8')));
xcLoadCircuit(instance, await readFile('data/xcheck/stim-rotated-memory-z-d5.stim.txt', 'utf8'));
console.log(xcCompare(instance, await readFile('data/xcheck/stim-rotated-memory-z-d5.dem.txt', 'utf8')));
console.log(runBenchmark(instance, { noiseMode: NOISE.SD6, d: 3, rounds: 3, p: 0.006, runs: 5000 }));
"
```
Expected: both comparisons `ok: true` with `missing`, `extra` and `differing` all 0; the benchmark returns a rate and `decodeErrors: 0`.

Run: `cargo test --release --no-default-features` — everything passes.

- [ ] **Step 6: Commit**

```bash
git add src/wasm_xc.rs src/lib.rs js/engine.js js/worker.js stabilizer_qec.wasm
git commit -m "feat(wasm): the general path in the browser: live model comparison and SD6 runs"
```

---

### Task 10: Figure 8 — checked against Stim

**Files:**
- Create: `js/xcheck-format.js`, `js/sections/xcheck.js`
- Modify: `index.html` (section 10), `css/styles.css`, `js/main.js`, `tools/site-tests.mjs`

**Interfaces:**
- Consumes: `data/xcheck/reference.json` (Task 8), worker ops `xcheck`, `xctiming`, `benchmark` (Task 9), `NOISE.SD6`, `CODE`, `XC_NOISE`.
- Produces: `initXcheck(root, compute)`; pure `verdict(result)`, `formatRel(x)`, `superscript(n)`, `circuitLabel(c)`, `microseconds(us)`, `disagreementText(rec)`, `megabytes(bytes)`.

- [ ] **Step 1: Failing Node tests** (append to `tools/site-tests.mjs`, import at the top)

```js
import { verdict, formatRel, circuitLabel, microseconds, disagreementText, megabytes } from '../js/xcheck-format.js';

test('verdict: identical, differing, unavailable, stale, engine error, pending', () => {
  const base = { ok: true, missing: 0, extra: 0, differing: 0 };
  assert.deepEqual(verdict(base), { text: 'identical', tone: 'ok' });
  assert.deepEqual(verdict({ ...base, missing: 2, differing: 1 }), { text: '3 differ', tone: 'fail' });
  assert.equal(verdict({ ok: false, unavailable: true, error: '404' }).text, 'reference unavailable');
  assert.equal(verdict({ ...base, stale: true }).tone, 'fail');
  assert.equal(verdict({ ok: false, error: 'boom' }).text, 'engine error: boom');
  assert.equal(verdict(null).text, 'queued');
});

test('formatRel writes powers of ten with superscripts', () => {
  assert.equal(formatRel(0), '0');
  assert.equal(formatRel(2.2e-16), '2.2 × 10⁻¹⁶');
  assert.equal(formatRel(3e-10), '3.0 × 10⁻¹⁰');
  assert.equal(formatRel(NaN), '—');
});

test('circuit labels, durations, disagreements, sizes', () => {
  assert.equal(circuitLabel({ source: 'stim', d: 5 }), 'Stim’s own · d = 5');
  assert.equal(circuitLabel({ source: 'ours', code: 'xzzx', noise: 'sd6', d: 3 }), 'XZZX · SD6 · d = 3');
  assert.equal(circuitLabel({ source: 'ours', code: 'rotated', noise: 'current', d: 7 }), 'rotated · engine’s model · d = 7');
  assert.equal(microseconds(3.14), '3.1 µs');
  assert.equal(microseconds(42.4), '42 µs');
  assert.equal(microseconds(1520), '1.5 ms');
  assert.equal(disagreementText({ disagreements: 0, non_ties: 0 }), 'none');
  assert.equal(disagreementText({ disagreements: 12, non_ties: 0 }), '12 · all ties');
  assert.equal(disagreementText({ disagreements: 12, non_ties: 2 }), '12 · 2 not ties');
  assert.equal(megabytes(1_400_000), '1.4 MB');
});
```

Run: `node tools/site-tests.mjs` → the new tests fail (module missing).

- [ ] **Step 2: `js/xcheck-format.js`**

```js
/**
 * Pure formatting for Figure 8. No DOM, no engine: everything here is tested
 * in Node by tools/site-tests.mjs.
 */

const SUP = { '-': '⁻', 0: '⁰', 1: '¹', 2: '²', 3: '³', 4: '⁴', 5: '⁵', 6: '⁶', 7: '⁷', 8: '⁸', 9: '⁹' };

export function superscript(n) {
  return String(n).split('').map((c) => SUP[c]).join('');
}

/** A relative difference, as a power of ten. */
export function formatRel(x) {
  if (!Number.isFinite(x)) return '—';
  if (x === 0) return '0';
  const e = Math.floor(Math.log10(x));
  const m = x / 10 ** e;
  return `${m.toFixed(1)} × 10${superscript(e)}`;
}

/** What a comparison row says, and in which tone. */
export function verdict(r) {
  if (!r) return { text: 'queued', tone: 'idle' };
  if (r.pending) return { text: 'deriving…', tone: 'idle' };
  if (!r.ok) {
    return r.unavailable
      ? { text: 'reference unavailable', tone: 'idle' }
      : { text: `engine error: ${r.error}`, tone: 'fail' };
  }
  if (r.stale) return { text: 'circuit differs from the recorded one', tone: 'fail' };
  const bad = r.missing + r.extra + r.differing;
  return bad === 0 ? { text: 'identical', tone: 'ok' } : { text: `${bad} differ`, tone: 'fail' };
}

export function circuitLabel(c) {
  if (c.source === 'stim') return `Stim’s own · d = ${c.d}`;
  const code = c.code === 'xzzx' ? 'XZZX' : 'rotated';
  const noise = c.noise === 'sd6' ? 'SD6' : 'engine’s model';
  return `${code} · ${noise} · d = ${c.d}`;
}

export function microseconds(us) {
  if (!Number.isFinite(us)) return '—';
  if (us >= 1000) return `${(us / 1000).toFixed(1)} ms`;
  if (us >= 10) return `${Math.round(us)} µs`;
  return `${us.toFixed(1)} µs`;
}

export function disagreementText(rec) {
  if (rec.disagreements === 0) return 'none';
  const n = rec.disagreements.toLocaleString('en-US');
  return rec.non_ties === 0 ? `${n} · all ties` : `${n} · ${rec.non_ties} not ties`;
}

export function megabytes(bytes) {
  return `${(bytes / 1e6).toFixed(1)} MB`;
}
```

Run: `node tools/site-tests.mjs` → all pass.

- [ ] **Step 3: Markup** — in `index.html`, section 10 (`id="internals"`): replace the first paragraph of `.prose` with

```html
    <p>
      Nothing of ours is precomputed. Every number was produced by a Rust stabilizer simulator
      compiled to WebAssembly and executed on your machine while you read. The one exception is
      marked: Figure 8 checks this engine against Stim and PyMatching, the tools the field uses,
      and PyMatching's side of that check was recorded, because it does not run in a browser.
    </p>
```

and after the closing `</div>` of that `.prose` block (before `<dl class="spec-list">`) insert:

```html
  <div class="prose">
    <p>
      Every circuit on this page can also be written out in Stim's circuit language, and every
      circuit Stim writes can be read in. That makes the engine checkable against the reference
      implementation rather than only against itself. The strictest check is the error model:
      for each circuit, list every elementary fault, the detectors it trips and the logical
      observable it flips, and its probability. Stim derives that list one way; this engine
      derives it by walking the circuit backwards. The two lists must agree entry for entry.
    </p>
  </div>

  <figure class="figure" data-xcheck>
    <figcaption class="figure__cap">
      <span class="figure__num">Figure 8</span>
      <span class="figure__title">checked against Stim</span>
      <span class="figure__meta">† recorded, not computed here</span>
    </figcaption>
    <div class="figure__body">
      <div class="table-wrap">
        <table class="xcheck">
          <caption class="visually-hidden">This engine's detector error model against Stim's, circuit by circuit</caption>
          <thead>
            <tr>
              <th scope="col">Circuit, p = 0.3%, T = d</th>
              <th scope="col" class="num">Detectors</th>
              <th scope="col" class="num">Mechanisms, ours / Stim's</th>
              <th scope="col" class="num">Largest Δp / p</th>
              <th scope="col">Verdict</th>
            </tr>
          </thead>
          <tbody data-xcheck-models></tbody>
        </table>
      </div>
      <button class="btn xcheck__more" data-xcheck-more hidden></button>
      <div class="table-wrap">
        <table class="xcheck">
          <caption class="visually-hidden">Logical error rate and decoding, this engine against PyMatching, rotated code under SD6</caption>
          <thead>
            <tr>
              <th scope="col">Rotated, SD6</th>
              <th scope="col" class="num">p_L, ours</th>
              <th scope="col" class="num">p_L, PyMatching †</th>
              <th scope="col" class="num">Same shots, disagreements †</th>
              <th scope="col" class="num">Decode, ours</th>
              <th scope="col" class="num">Decode, PyMatching †</th>
            </tr>
          </thead>
          <tbody data-xcheck-decoding></tbody>
        </table>
      </div>
      <p class="status" role="status" data-xcheck-status>Runs when you scroll here.</p>
      <p class="note" data-xcheck-foot></p>
    </div>
  </figure>
```

- [ ] **Step 4: CSS** — in `css/styles.css`, in the Tables block after `.tag--flat`:

```css
/* Figure 8: the comparison with Stim. A value recorded offline rather than
   computed here is set in the secondary ink and carries a dagger in its text. */
.xcheck td.recorded { color: var(--ink-3); }
.xcheck .verdict-text--ok   { color: var(--ok-ink); font-weight: 600; }
.xcheck .verdict-text--fail { color: var(--fail-ink); font-weight: 600; }
.xcheck .verdict-text--idle { color: var(--ink-3); }
.xcheck + .xcheck__more,
.xcheck__more { margin: var(--s3) 0; }
.figure__body .table-wrap + .table-wrap { margin-top: var(--s5); }
```

Run: `node tools/contrast.mjs` → all ok (only existing tokens are used).

- [ ] **Step 5: `js/sections/xcheck.js`**

```js
/**
 * Section 10, Figure 8 — checked against Stim.
 *
 * The first table derives this engine's detector error model for each circuit,
 * live in the worker, and compares it mechanism by mechanism with the model
 * Stim wrote for the same circuit. The second runs our sampler and decoder live
 * beside PyMatching's recorded numbers. Recorded values wear a dagger and the
 * secondary ink. Nothing is shown that was not computed here or read from the
 * reference file, and a reference that cannot be read says so.
 */

import { CODE, NOISE, XC_NOISE } from '../engine.js';
import { wilson, percent } from '../compute.js';
import { $, fill, el } from '../dom.js';
import { verdict, formatRel, circuitLabel, microseconds, disagreementText, megabytes } from '../xcheck-format.js';

const DATA = new URL('../../data/xcheck/', import.meta.url);

/** Live shots per distance for the decoding table, set by measured cost. */
const LIVE_RUNS = { 3: 20000, 5: 8000, 7: 3000 };
const TIMING_RUNS = { 3: 2000, 5: 800, 7: 300 };

function job(c) {
  return {
    key: c.name,
    kind: c.source,
    demUrl: new URL(`${c.name}.dem.txt`, DATA).href,
    circuitUrl: c.source === 'stim' ? new URL(`${c.name}.stim.txt`, DATA).href : null,
    gen: c.source === 'ours'
      ? { codeType: c.code === 'xzzx' ? CODE.XZZX : CODE.ROTATED, d: c.d, rounds: c.rounds,
        noise: c.noise === 'sd6' ? XC_NOISE.SD6 : XC_NOISE.CURRENT, p: c.p, eta: c.eta, basis: 0 }
      : null,
    circuitSha256: c.circuit_sha256 ?? null,
  };
}

export function initXcheck(root, compute) {
  const figure = $('[data-xcheck]', root);
  if (!figure) return;
  const models = $('[data-xcheck-models]', figure);
  const decoding = $('[data-xcheck-decoding]', figure);
  const status = $('[data-xcheck-status]', figure);
  const foot = $('[data-xcheck-foot]', figure);
  const more = $('[data-xcheck-more]', figure);

  const modelRows = new Map();

  function modelRow(c) {
    const cells = {
      detectors: el('td', { class: 'num', text: c.detectors?.toLocaleString('en-US') ?? '—' }),
      mechanisms: el('td', { class: 'num', text: '—' }),
      rel: el('td', { class: 'num', text: '—' }),
      verdict: el('td', {}, [el('span', { class: 'verdict-text--idle', text: c.d >= 7 ? 'on request' : 'queued' })]),
    };
    const tr = el('tr', {}, [el('th', { scope: 'row', text: circuitLabel(c) }), cells.detectors, cells.mechanisms, cells.rel, cells.verdict]);
    modelRows.set(c.name, { c, cells });
    return tr;
  }

  function fillModel(result) {
    const row = modelRows.get(result.key);
    if (!row) return;
    const v = verdict(result);
    if (result.ok) {
      row.cells.detectors.textContent = result.detectors.toLocaleString('en-US');
      row.cells.mechanisms.textContent = `${result.ours.toLocaleString('en-US')} / ${result.theirs.toLocaleString('en-US')}`;
      row.cells.rel.textContent = formatRel(result.maxRel);
    }
    fill(row.cells.verdict, el('span', { class: `verdict-text--${v.tone}`, text: v.text }));
  }

  async function runModels(circuits) {
    for (const c of circuits) fillModel({ key: c.name, pending: true });
    await compute.call('xcheck', { rows: circuits.map(job) }, (p) => {
      fillModel(p.row);
      status.textContent = `Error models: ${p.done} of ${p.total} compared`;
    });
  }

  async function runDecoding(records) {
    for (const [i, rec] of records.entries()) {
      const { ours, oursTime } = rec.cells;
      status.textContent = `Decoding: d = ${rec.d}, p = ${percent(rec.p, 1)} (${i + 1} of ${records.length})`;
      const runs = LIVE_RUNS[rec.d] ?? 2000;
      const r = await compute.call('benchmark', { noiseMode: NOISE.SD6, codeType: CODE.ROTATED, d: rec.d, rounds: rec.d, p: rec.p, runs });
      const ci = wilson(r.rate, r.runs);
      fill(ours, [
        document.createTextNode(percent(r.rate)),
        el('span', { class: 'ci', text: `${percent(ci.lo)} – ${percent(ci.hi)} · ${r.runs.toLocaleString('en-US')} shots`
          + (r.decodeErrors ? ` · ${r.decodeErrors} refused` : '') }),
      ]);
      const t = await compute.call('xctiming', {
        cfg: { codeType: CODE.ROTATED, d: rec.d, rounds: rec.d, noise: XC_NOISE.SD6, p: rec.p },
        runs: TIMING_RUNS[rec.d] ?? 500,
      });
      oursTime.textContent = microseconds(t.decodeMicros);
    }
  }

  async function start() {
    status.textContent = 'Loading the reference…';
    let reference;
    try {
      const response = await fetch(new URL('reference.json', DATA));
      if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
      reference = await response.json();
    } catch (error) {
      status.textContent = `Reference unavailable (${error.message}); nothing to compare against.`;
      return;
    }

    fill(models, reference.circuits.map(modelRow));
    const now = reference.circuits.filter((c) => c.d < 7);
    const later = reference.circuits.filter((c) => c.d >= 7);

    const records = reference.decoding.map((rec) => {
      const cells = {
        ours: el('td', { class: 'num', text: '—' }),
        oursTime: el('td', { class: 'num', text: '—' }),
      };
      const ci = wilson(rec.pymatching_failures / rec.shots, rec.shots);
      decoding.append(el('tr', {}, [
        el('th', { scope: 'row', text: `d = ${rec.d}, p = ${percent(rec.p, 1)}` }),
        cells.ours,
        el('td', { class: 'num recorded' }, [
          document.createTextNode(`${percent(rec.pymatching_failures / rec.shots)} †`),
          el('span', { class: 'ci', text: `${percent(ci.lo)} – ${percent(ci.hi)} · ${rec.shots.toLocaleString('en-US')} shots` }),
        ]),
        el('td', { class: 'num recorded', text: `${disagreementText(rec)} †` }),
        cells.oursTime,
        el('td', { class: 'num recorded', text: `${microseconds(rec.pymatching_us)} †` }),
      ]));
      return { ...rec, cells };
    });

    foot.textContent = `† Recorded on ${reference.generated} with Stim ${reference.stim} and PyMatching `
      + `${reference.pymatching}, by tools/xcheck.py, which reproduces every one. "Same shots" means both `
      + 'decoders were given the identical detection events Stim sampled; two exact matchers may only '
      + 'disagree where two corrections tie in weight, and the recorded run checks every disagreement for '
      + 'that. Decode times: ours is measured now, in this tab, in WebAssembly; PyMatching\'s is native '
      + 'code on the machine that recorded it. Our native decoder in that same run: '
      + reference.decoding.map((r) => `d = ${r.d}, p = ${percent(r.p, 1)} ${microseconds(r.ours_us)}`).join('; ') + '.';

    if (later.length) {
      const bytes = later.reduce((sum, c) => sum + (c.bytes ?? 0), 0);
      more.textContent = `Check d = 7 too (${megabytes(bytes)} of reference)`;
      more.hidden = false;
      more.addEventListener('click', async () => {
        more.disabled = true;
        try {
          await runModels(later);
          status.textContent = 'd = 7 compared.';
        } catch (error) {
          status.textContent = `Failed: ${error.message}`;
        } finally {
          more.hidden = true;
        }
      }, { once: true });
    }

    try {
      await runModels(now);
      await runDecoding(records);
      status.textContent = 'Done.';
    } catch (error) {
      status.textContent = `Failed: ${error.message}`;
    }
  }

  let started = false;
  const observer = new IntersectionObserver((entries) => {
    if (started || !entries.some((e) => e.isIntersecting)) return;
    started = true;
    observer.disconnect();
    start();
  }, { rootMargin: '400px 0px' });
  observer.observe(figure);
}
```

In `js/main.js`: `import { initXcheck } from './sections/xcheck.js';` and inside `if (compute) { ... }` add `initXcheck($('#internals'), compute);`.

- [ ] **Step 6: Check it in the browser**

Serve with `python3 -m http.server 4174` (via `preview_start`, `.claude/launch.json` entry `{"name": "site", "runtimeExecutable": "python3", "runtimeArgs": ["-m", "http.server", "4174"], "port": 4174}`), scroll to Figure 8, and read the rows with `get_page_text`/`read_page`:
- every d ≤ 5 row reads "identical";
- the decoding rows fill with a live rate and a decode time;
- clicking "Check d = 7 too" fills the d = 7 rows with "identical";
- `read_console_messages` with `onlyErrors` is empty.
Screenshot the figure at desktop width and at 390 px (no horizontal page scroll; tables scroll inside `.table-wrap`). Adjust `LIVE_RUNS` so the whole decoding table finishes in under about a minute on this machine.

- [ ] **Step 7: Commit**

```bash
git add index.html css/styles.css js/xcheck-format.js js/sections/xcheck.js js/main.js tools/site-tests.mjs .claude/launch.json
git commit -m "feat(site): Figure 8, the engine checked against Stim, live"
```

---

### Task 11: SD6 in sections 07 and 09, the sweep tool, and the README

**Files:**
- Create: `js/sweep-config.js`, `tools/sweep.mjs`
- Modify: `js/sections/threshold.js`, `js/sections/bench.js`, `index.html`, `tools/site-tests.mjs`, `README.md`

**Interfaces:**
- Consumes: `NOISE.SD6`, `runBenchmark`, `fitThreshold`, the Task 6 and Task 8 reports.
- Produces: `js/sweep-config.js` exporting `DISTANCES`, `SWEEP_PS`, `SWEEP_RUNS` (moved verbatim from `threshold.js`, plus SD6 entries).

- [ ] **Step 1: Move the sweep configuration** — create `js/sweep-config.js` containing the `DISTANCES`, `SWEEP_PS` and `SWEEP_RUNS` constants *with their comment blocks*, moved verbatim from `threshold.js`, exported, importing `NOISE` from `./engine.js`. `threshold.js` imports them from there. Run `node tools/site-tests.mjs` and load section 07 in the browser: behaviour unchanged.

- [ ] **Step 2: Find the SD6 crossing and cost** — with the rebuilt WASM, in Node:

```bash
node --input-type=module -e "
import { readFile } from 'node:fs/promises';
import { runBenchmark, NOISE } from './js/engine.js';
const { instance } = await WebAssembly.instantiate(await readFile('stabilizer_qec.wasm'), {});
instance.exports.wasm_seed(12345, 678);
for (const p of [0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.01]) {
  const row = [3, 5, 7, 9].map((d) => { const r = runBenchmark(instance, { noiseMode: NOISE.SD6, d, rounds: d, p, runs: 1500 }); return 'd'+d+' '+(r.rate*100).toFixed(2)+'% '+(r.seconds*1000/1500).toFixed(2)+'ms'; });
  console.log((p*100).toFixed(1)+'%', row.join('  '));
}"
```

Choose `SWEEP_PS[NOISE.SD6]`: nine points centred on where the distances swap order, following the circuit-level comment's logic (no points where every distance reads 0.00%). Choose `SWEEP_RUNS[NOISE.SD6]` so the sweep costs no more wall-clock than the circuit-level one (~80 s). `DISTANCES[NOISE.SD6] = [3, 5, 7, 9]`. Write the comment for each from the measured numbers.

- [ ] **Step 3: Test the config** (append to `tools/site-tests.mjs`)

```js
import { DISTANCES, SWEEP_PS, SWEEP_RUNS } from '../js/sweep-config.js';

test('sweep config: every noise model has distances, an increasing window and a shot count', () => {
  for (const mode of [0, 1, 2, 3]) {
    assert.ok(DISTANCES[mode].length >= 4, `mode ${mode}`);
    const ps = SWEEP_PS[mode];
    assert.ok(ps.length >= 7 && ps.every((p, i) => i === 0 || p > ps[i - 1]), `mode ${mode}`);
    assert.ok(SWEEP_RUNS[mode] >= 1000, `mode ${mode}`);
  }
});
```

- [ ] **Step 4: Section 07 UI** — in `index.html`, the sweep's noise select gains `<option value="3">Circuit-level, SD6 (T = d, per basis)</option>`. In `threshold.js`'s `initThresholdSweep`, add a `syncDecoder()` called at init and on noise change:

```js
  // SD6 runs through the general path, which has one decoder: exact matching
  // weighted by each fault's probability. The selector does not apply to it.
  const syncDecoder = () => {
    const general = Number(noiseSelect.value) === NOISE.SD6;
    decoderSelect.disabled = general;
    decoderSelect.closest('.field').classList.toggle('field--inert', general);
  };
```

and remove the duplicated `describeSweep` definition nested inside the noise `change` handler (it shadows the outer one and is dead weight; the handler keeps calling `describeSweep()`).

Add one paragraph to section 07's prose, after the paragraph that ends "…reports the uncorrected value.":

```html
    <p>
      Circuit-level noise is offered two ways. The engine's own model puts an independent error on
      each qubit of every CNOT, so a gate fails about 2p of the time, and it leaves idle qubits
      alone. SD6, the standard model published thresholds assume, gives each CNOT one two-qubit
      error of total probability p and lets idle qubits err too. They are different physics, and
      they cross in different places: [the two fitted values, quoted from the README table]. SD6
      is scored the way experiments are, one logical basis at a time; that barely moves where the
      curves cross, but it roughly halves the rate below it.
    </p>
```

replacing the bracket with the fitted values from Step 7.

- [ ] **Step 5: Section 09** — the bench's noise select gains `<option value="3">Circuit-level (SD6, per basis)</option>`. In `bench.js`, extend `syncRounds` into `syncControls`: for SD6, mark the decoder, bias, erasure and correlated-noise fields inert and disable the channel button, with the channel status reading "The logical channel needs both bases at once, which a per-basis experiment does not measure." Status line after a run: when `result.decodeErrors` (sum it across chunks in the `stream` op) is non-zero, append `· N shots refused by the decoder`. In `worker.js`'s `stream`, accumulate `decodeErrors` from each `runBenchmark` result and include it in progress and the result.

Check in the browser: select SD6 in the bench, run 2,000 shots at d = 5, p = 0.5%; the rate is plausible, no console errors; the channel button is disabled with its explanation. Select SD6 in section 07 and run the sweep to completion; a threshold is reported.

- [ ] **Step 6: The sweep tool** — `tools/sweep.mjs`:

```js
// Threshold sweeps in Node, for the figures the README quotes.
// Run: node tools/sweep.mjs <noiseMode> <codeType> [repeats=4] [decoder=2]
// Uses the same distances, windows and shot counts as section 07.
import { readFile } from 'node:fs/promises';
import { runBenchmark } from '../js/engine.js';
import { fitThreshold } from '../js/compute.js';
import { DISTANCES, SWEEP_PS, SWEEP_RUNS } from '../js/sweep-config.js';

const [noiseMode, codeType, repeats = 4, decoder = 2] = process.argv.slice(2).map(Number);
const bytes = await readFile(new URL('../stabilizer_qec.wasm', import.meta.url));
const fits = [];
for (let rep = 0; rep < repeats; rep++) {
  const { instance } = await WebAssembly.instantiate(bytes, {});
  const seed = new Uint32Array(2);
  crypto.getRandomValues(seed);
  instance.exports.wasm_seed(seed[0], seed[1]);
  const points = [];
  for (const d of DISTANCES[noiseMode]) {
    for (const p of SWEEP_PS[noiseMode]) {
      const r = runBenchmark(instance, {
        noiseMode, codeType, decoder, d, p, bias: 0.5, runs: SWEEP_RUNS[noiseMode], rounds: noiseMode === 0 ? 1 : d,
      });
      if (r.decodeErrors) console.error(`d=${d} p=${p}: ${r.decodeErrors} decode errors`);
      points.push({ d, p, pL: r.rate, runs: r.runs });
    }
  }
  const fit = fitThreshold(points, { bootstrap: 120 });
  fits.push(fit);
  console.log(`sweep ${rep + 1}: p_th ${(fit.pTh * 100).toFixed(3)}% [${(fit.pThLo * 100).toFixed(3)}, ${(fit.pThHi * 100).toFixed(3)}]`
    + ` nu ${fit.nuDetermined ? fit.nu.toFixed(2) : 'n/d'} omega ${fit.corrected ? fit.omega.toFixed(2) : 'n/f'}`
    + ` chi2 ${fit.reducedChi2.toFixed(2)} window ${fit.pointsUsed}/${fit.pointsTotal} uncorrected ${(fit.leading.pTh * 100).toFixed(3)}%`
    + ` crossings ${fit.crossings.map((c) => `${c.small}/${c.large}@${(c.p * 100).toFixed(3)}%`).join(' ')}`);
}
const ok = fits.filter((f) => f.ok);
const mean = ok.reduce((s, f) => s + f.pTh, 0) / ok.length;
const sd = Math.sqrt(ok.reduce((s, f) => s + (f.pTh - mean) ** 2, 0) / Math.max(1, ok.length - 1));
console.log(`mean of ${ok.length}: ${(mean * 100).toFixed(2)}% ± ${(sd * 100).toFixed(2)}`);
```

Run in the background: `node tools/sweep.mjs 3 0 > <scratchpad>/sweep-sd6-rotated.txt` and `node tools/sweep.mjs 3 1 > <scratchpad>/sweep-sd6-xzzx.txt`.

- [ ] **Step 7: README** — add, after "### 12." (the last numbered defect section) and before "### Located loss under circuit-level noise", a section:

```markdown
## Checked against Stim and PyMatching
```

containing, from the recorded runs (numbers quoted only from the Task 6, Task 8 and Step 6 outputs):
- what the general path is (circuit IR, backward DEM, weighted MWPM, frame sampler) in one paragraph;
- check 1 as a table: circuit, detectors, mechanisms, largest Δp/p, verdict — every row;
- check 2: disagreements on identical shots and that every one is a tie, with the weight-noise level;
- check 3: the χ² z-scores and the sampler rates;
- check 4: µs per shot, ours native against PyMatching, stated plainly (and the factor);
- check 5: the old and new paths' error models are identical at d = 3, 5, 7 for both codes and both bases; the old-decoder vs new-decoder rates table from Task 6 with what it shows;
- check 6: single-fault counts at d = 3, 5, 7, both codes, both noise models;
- anything the checks caught along the way, written up in the style of the defect sections (what was wrong, how it showed, how it was found, the fix);
- how to reproduce: the maturin and `tools/xcheck.py` commands.

In the threshold table, add rows `rotated | SD6 (per basis)` and `XZZX | SD6 (per basis)` from Step 6's four sweeps, and one sentence under the table on why SD6 and the engine's circuit-level model differ. Add `src/circuit.rs`, `src/dem.rs`, `src/dem_decoder.rs`, `src/frame_sampler.rs`, `src/shots.rs`, `src/memory.rs`, `tools/xcheck.py`, `tools/sweep.mjs`, `data/xcheck/` and the new JS files to "Repository Structure", and the `.venv` setup to "Building & Running".

Replace the bracket in section 07's new paragraph (Step 4) with the fitted SD6 and engine-model values.

- [ ] **Step 8: Commit**

```bash
git add js/sweep-config.js js/sections/threshold.js js/sections/bench.js js/worker.js index.html tools/site-tests.mjs tools/sweep.mjs README.md
git commit -m "feat(site): SD6 in the threshold sweep and the bench; README: checked against Stim"
```

---

### Task 12: Verification and review

**Files:** none new.

- [ ] **Step 1: Everything, from clean**

```bash
cargo test --release --no-default-features
cargo test --release --no-default-features -- --ignored every_single_fault_is_corrected_d7 old_and_new_paths_agree_d7 --nocapture
cargo build --release
cargo build --release --target wasm32-unknown-unknown --no-default-features && cmp target/wasm32-unknown-unknown/release/stabilizer_qec.wasm stabilizer_qec.wasm
node tools/site-tests.mjs
node tools/contrast.mjs
.venv/bin/python tools/xcheck.py --quick
```
Expected: all pass; `cmp` silent (the committed WASM is the built one).

- [ ] **Step 2: The page** — serve; load with console listeners; scroll through every section; the existing sections behave as before (the results table completes; the threshold sweep runs for the old three models; the bench runs); Figure 8 completes with every row "identical"; zero console errors. Screenshots at 1440 and 390 px of Figure 8 and section 07's SD6 sweep.

- [ ] **Step 3: Review** — run the `code-review` skill at `high` over the branch diff against `master`, then fix every confirmed finding (tests first where the finding is a bug) and re-run Steps 1–2.

- [ ] **Step 4: Commit any fixes, and write the morning summary** — what landed, what each check found, every number with its source, anything surprising, and what B needs (the dataset download, with filenames and sizes, for Jasper to approve).

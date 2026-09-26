# Google's Hardware Data Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Decode Google's Willow and Sycamore surface-code memory data with this engine's plain and correlated matchers, measure Λ with one fitting implementation applied to ours and Google's predictions alike, and show it on the site: a recorded run and a live panel decoding real Willow readouts.

**Architecture:** Plan 2 of B2 (spec: `docs/superpowers/specs/2026-09-25-google-data-design.md`, Part 2). The parser learns sweep-controlled X and Pauli gates. A new `m2d` module turns raw measurements into detection events through noiseless tableau runs. `tools/google.py` streams every experiment from the zips, checks the conversion bit for bit against Google's files, decodes with every prior in both modes, and writes small committed records. `js/lambda-fit.js` fits ε and Λ from those records, in Node for the README and in the browser for the page. Section 11 shows the recorded fit, and decodes a 2,000-shot extract per distance live from raw measurements.

**Tech Stack:** Rust 2021 (no new crates), PyO3 (`maturin develop --profile python`), Python with Stim 1.16 / PyMatching 2.4 / NumPy, plain ES modules, canvas plots.

## Global Constraints

- No new crates, and no new JS dependencies. The page has no build step.
- Data stays in the zips under `data/google/`, which is gitignored. Nothing is unpacked. Only the results (`data/google-results/`) and the site extract (`data/willow-extract/`, about 2 MB, CC BY 4.0, credited) are committed.
- Every quoted number comes from a committed record or a command's printed output.
- The one fit implementation (`js/lambda-fit.js`) serves the README and the page, and it treats our predictions and Google's alike.
- The Python module builds with `VIRTUAL_ENV=$PWD/.venv CARGO_TARGET_DIR=target-py .venv/bin/maturin develop --profile python`. Run Python from `tools/` or `/tmp`: a stale `stabilizer_qec.so` in the repository root shadows the module otherwise.
- Tests: `cargo test --release --no-default-features`; `cargo build --release` and the wasm32 build must compile; `node tools/site-tests.mjs`; `node tools/contrast.mjs`.
- Commit after each task on `feature/willow-data` with the trailer `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. Never push, and never merge into `master`.

## Facts this plan relies on (measured 2026-09-25)

- **Willow:** 420 experiments of 50,000 shots.
  - 14 patches (d3 ×9, d5 ×4, d7 ×1), bases X and Z, rounds {1, 10, 13, 30, 50, …, 250}.
  - Path prefix: `google_105Q_surface_code_d3_d5_d7/{patch}/{X|Z}/r{NN}/`.
  - Files: `circuit_ideal.stim`, `circuit_noisy_si1000.stim`, `measurements.b8`, `sweep_bits.b8`, `detection_events.b8`, `obs_flips_actual.b8`, and `decoding_results/{pathway}/{error_model.dem, obs_flips_predicted.b8}` for five pathways.
- **Sycamore:** 130 surface-code experiments of 50,000 shots.
  - Directories `surface_code_b{X|Z}_d{3|5}_r{NN}_center_{r}_{c}/` hold `circuit_ideal.stim`, `circuit_noisy.stim`, `measurements.b8`, `sweep.b8`, `detection_events.b8`, `obs_flips_actual.01`, `circuit_detector_error_model.dem`, `pij_from_even_for_odd.dem`, `pij_from_odd_for_even.dem`, and `obs_flips_predicted_by_{pymatching,correlated_matching,belief_matching,tensor_network_contraction}.01`.
  - Google's PyMatching and correlated matching used `circuit_detector_error_model.dem` for every shot. Belief matching and tensor-network contraction used `pij_from_even_for_odd` on odd shots and `pij_from_odd_for_even` on even ones.
- **Instructions:** the circuits use exactly `R M H CX CZ X Y DETECTOR OBSERVABLE_INCLUDE QUBIT_COORDS TICK` in the ideal circuits, plus `DEPOLARIZE1 DEPOLARIZE2 X_ERROR I` in the noisy ones. `CX` appears only as `CX sweep[k] q …`.
- **Google's model files** are not Stim's model of their noisy circuit. At d5 r10, Stim gives 4,349 mechanisms against Google's 3,717, with probabilities differing by a median 3.8%. So our own model is a third prior, not a duplicate.
- **Speed:** d7 r250 has 12,000 detectors and 936 defects a shot on average. It decodes at 0.13 ms a shot plain and 0.28 ms correlated on 10 cores.
- **Published numbers:**
  - Willow (arXiv:2408.13687), neural-network decoder: Λ = 2.14 ± 0.02, and 0.143% ± 0.003% per cycle at d = 7.
  - Sycamore (arXiv:2207.06431): d5 2.914% ± 0.016%, d3 3.028% ± 0.023% per cycle.

## File map

| File | Change |
|---|---|
| `src/circuit.rs` | `Instr::Pauli`, `Instr::SweepX`; parse, emit, `qubits`, `Resolved::num_sweep_bits` |
| `src/dem.rs` | the new instructions are no-ops in the backward pass |
| `src/m2d.rs` | new: measurements and sweep bits to detection events |
| `src/lib.rs` | `pub mod m2d; pub mod wasm_hw;` |
| `src/py_api.rs` | `m2d_b8`, `circuit_to_stim` |
| `src/wasm_hw.rs` | new: WASM exports for the live panel |
| `tools/google.py` | new: experiments, `check`, `run`, `summary`, `extract` |
| `data/google-results/` | new: `checks.json`, `willow.json`, `sycamore.json` |
| `data/willow-extract/` | new: the live panel's data |
| `js/lambda-fit.js` | new: ε and Λ fits |
| `tools/lambda.mjs` | new: the fits for the README |
| `tools/site-tests.mjs` | fit tests |
| `js/plot.js` | a log y axis |
| `js/engine.js`, `js/worker.js` | the live panel's calls |
| `js/sections/hardware.js` | new: section 11 |
| `index.html`, `css/styles.css`, `js/main.js` | section 11 |
| `README.md` | "Google's hardware data" |

---

### Task 1: Sweep bits and Pauli gates in the parser

**Files:** Modify `src/circuit.rs`, `src/dem.rs:398-404`. Test: `src/circuit.rs` tests, `src/dem.rs` tests.

**Interfaces:**
- Produces:
  - `Instr::Pauli { pauli: Pauli, qubits: Vec<u32> }`, where `pauli` is 0 = I, 1 = X, 2 = Z, 3 = Y;
  - `Instr::SweepX(Vec<(u32, u32)>)` as (sweep bit, qubit);
  - `Resolved::num_sweep_bits: usize`.

- [ ] **Step 1: Failing tests.** Append to `mod tests` in `src/circuit.rs`:

```rust
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
```

Append to the tests in `src/dem.rs`:

```rust
    #[test]
    fn pauli_gates_and_sweep_bits_leave_the_model_alone() {
        // They flip signs, never which detectors a fault sets off.
        let base = Circuit::parse(crate::fixtures::REP3).unwrap();
        let text = format!("X 0 1\nY 2\nCX sweep[0] 1\nI 0\n{}", crate::fixtures::REP3);
        let with = Circuit::parse(&text).unwrap();
        assert_eq!(Dem::from_circuit(&with).unwrap().to_stim(true), Dem::from_circuit(&base).unwrap().to_stim(true));
    }
```

(If `crate::fixtures::REP3` begins with instructions that the prefix would reorder in a way that matters, such as a reset, the X lines still act on |0⟩ and only flip signs. The model is unchanged either way.)

Run: `cargo test --release --no-default-features sweep pauli` — expected: compile errors, since the new variants don't exist.

- [ ] **Step 2: Implement.** In `src/circuit.rs`:
  - Add the variants to `Instr`, after `Cz`:
    ```rust
    /// `I`, `X`, `Y`, `Z`: Pauli gates, which only flip signs. 0 = I.
    Pauli { pauli: Pauli, qubits: Vec<u32> },
    /// `CX sweep[k] q`: an X on `q` when sweep bit `k` of the shot is set.
    SweepX(Vec<(u32, u32)>),
    ```
  - In `qubits()`, add `| Instr::Pauli { qubits, .. }` to the first arm, and an arm `Instr::SweepX(pairs) => pairs.iter().map(|&(_, q)| q).collect(),`.
  - `Resolved` gains `pub num_sweep_bits: usize`. In `resolve`, track `let mut sweeps = 0usize;` and in the loop add `Instr::SweepX(pairs) => { for &(k, _) in pairs { sweeps = sweeps.max(k as usize + 1); } }`. Put `num_sweep_bits: sweeps` in the returned struct.
  - Add a helper:
    ```rust
    fn sweep_pairs(tokens: &[&str], name: &str) -> Result<Vec<(u32, u32)>, String> {
        if tokens.len() % 2 != 0 {
            return Err(format!("{name}: targets must come in pairs, got {}", tokens.len()));
        }
        tokens
            .chunks(2)
            .map(|c| {
                let bit = c[0]
                    .strip_prefix("sweep[")
                    .and_then(|s| s.strip_suffix(']'))
                    .and_then(|s| s.parse::<u32>().ok())
                    .ok_or_else(|| format!("{name}: a sweep-controlled pair needs 'sweep[k] q', got '{} {}'", c[0], c[1]))?;
                let q = c[1].parse::<u32>().map_err(|_| format!("{name}: bad qubit target '{}'", c[1]))?;
                Ok((bit, q))
            })
            .collect()
    }
    ```
  - In `parse_line`, replace the `"CX" | "CNOT" | "ZCX"` arm with:
    ```rust
        "CX" | "CNOT" | "ZCX" => {
            none()?;
            if t.iter().any(|x| x.starts_with("sweep[")) {
                Instr::SweepX(sweep_pairs(&t, &name)?)
            } else {
                Instr::Cx(pair_targets(&t, &name)?)
            }
        }
    ```
    `CZ sweep[k] q` already fails `qubit_targets`. Add the Pauli gates:
    ```rust
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
    ```
  - In `emit`:
    ```rust
            Instr::Pauli { pauli, qubits } => format!("{} {}", ["I", "X", "Z", "Y"][*pauli as usize], join_q(qubits)),
            Instr::SweepX(pairs) => format!(
                "CX {}",
                pairs.iter().map(|(k, q)| format!("sweep[{k}] {q}")).collect::<Vec<_>>().join(" ")
            ),
    ```
  - In `src/dem.rs`, add `| Instr::Pauli { .. } | Instr::SweepX(_)` to the no-op arm of `from_circuit`'s match. The frame sampler's `_ => {}` already ignores both.

- [ ] **Step 3: Run everything.** `cargo test --release --no-default-features`: all pass, including `sweep_controls_and_pauli_gates_parse_and_round_trip`, `sweep_bits_may_only_control_a_cx` and `pauli_gates_and_sweep_bits_leave_the_model_alone`.

- [ ] **Step 4: Commit.** `feat(circuit): sweep-controlled X and Pauli gates, as Google's circuits use them`.

---

### Task 2: From measurements to detection events

**Files:** Create `src/m2d.rs`. Modify `src/lib.rs`, `src/py_api.rs`.

**Interfaces:**
- Produces:
  - `m2d::M2d::new(&Circuit) -> Result<M2d, String>`;
  - `M2d::convert(&self, meas: &[bool], sweeps: &[bool]) -> (Vec<bool>, u64)`;
  - `M2d::convert_b8(&self, meas: &[u8], sweeps: &[u8], num_shots: usize) -> Result<(Vec<u8>, Vec<u8>), String>`, returning detectors and observables in b8;
  - fields `num_measurements`, `num_sweep_bits`, `num_detectors`, `num_observables` (all `pub usize`);
  - Python `sq.m2d_b8(circuit_text, meas: bytes, sweeps: bytes, num_shots) -> (bytes, bytes)` and `sq.circuit_to_stim(text) -> str`.

- [ ] **Step 1: Failing tests.** Create `src/m2d.rs` with only the test module at first:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::circuit::Circuit;

    #[test]
    fn a_sweep_bit_moves_the_reference() {
        let c = Circuit::parse(
            "R 0 1\nCX sweep[0] 0\nX 1\nM 0 1\nDETECTOR rec[-2]\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]\n",
        )
        .unwrap();
        let m = M2d::new(&c).unwrap();
        assert_eq!((m.num_measurements, m.num_sweep_bits, m.num_detectors), (2, 1, 2));
        // A noiseless shot with the bit set reads 1 1: nothing fired.
        assert_eq!(m.convert(&[true, true], &[true]), (vec![false, false], 0));
        // The same readout with the bit clear: qubit 0 flipped.
        assert_eq!(m.convert(&[true, true], &[false]), (vec![true, false], 0));
        // Qubit 1 read 0 against the reference's 1: D1 and the observable fire.
        assert_eq!(m.convert(&[false, false], &[false]), (vec![false, true], 1));
    }

    #[test]
    fn noiseless_runs_convert_to_silence() {
        use crate::memory::{generate, CodeKind, NoiseModel};
        use crate::circuit::Basis;
        // A memory circuit with a sweep bit on every data qubit, after the
        // first reset, as Google's circuits have them.
        let c = generate(CodeKind::Rotated, 3, 3, NoiseModel::Sd6 { p: 0.0 }, Basis::Z).unwrap();
        let text = c.to_stim();
        let first_tick = text.find("TICK").unwrap();
        let sweep: String = (0..9).map(|k| format!("sweep[{k}] {k} ")).collect();
        let text = format!("{}CX {sweep}\n{}", &text[..first_tick], &text[first_tick..]);
        let c = Circuit::parse(&text).unwrap();
        let m = M2d::new(&c).unwrap();
        let res = c.resolve().unwrap();
        let mut rng = crate::surface_code::Xorshift::new(5);
        for seed in 0..40u64 {
            let sweeps: Vec<bool> = (0..m.num_sweep_bits).map(|_| rng.next_u64() & 1 == 1).collect();
            let meas = run(&res, &sweeps, 100 + seed);
            let (dets, obs) = m.convert(&meas, &sweeps);
            assert!(dets.iter().all(|&b| !b), "seed {seed}: {dets:?}");
            assert_eq!(obs, 0);
        }
    }

    #[test]
    fn nondeterministic_detectors_are_refused() {
        let c = Circuit::parse("RX 0\nM 0\nDETECTOR rec[-1]\n").unwrap();
        assert!(M2d::new(&c).unwrap_err().contains("not deterministic"));
    }

    #[test]
    fn b8_conversion_matches_the_bool_path() {
        let c = Circuit::parse("R 0 1\nCX sweep[0] 0\nM 0 1\nDETECTOR rec[-2]\nDETECTOR rec[-1] rec[-2]\n").unwrap();
        let m = M2d::new(&c).unwrap();
        // Shots: meas (1,0) sweep 1; meas (1,1) sweep 0.
        let (d, o) = m.convert_b8(&[0b01, 0b11], &[1, 0], 2).unwrap();
        assert_eq!(d, vec![0b00, 0b01]);
        assert_eq!(o, Vec::<u8>::new());
        assert!(m.convert_b8(&[0], &[0], 2).is_err());
    }
}
```

The generated memory circuit writes its data qubits as 0…d²−1 only if `memory.rs` numbers them that way. Before running, check `generate`'s `QUBIT_COORDS` and use the data qubits' indices in `sweep`. Also, the insertion point must come after the first reset and before the first gate: use the first `TICK` only if the first layer is the reset layer, and otherwise find the reset line.

Add `pub mod m2d;` to `src/lib.rs`. Run `cargo test --release --no-default-features m2d`; it should fail to compile.

- [ ] **Step 2: Implement** above the tests in `src/m2d.rs`:

```rust
//! From raw measurements to detection events, as `stim m2d` computes them.
//!
//! A detector compares a parity of measurements with what a noiseless run of
//! the circuit gives. One noiseless run, with every sweep bit clear, is the
//! reference. A sweep bit (`CX sweep[k] q`) is a classically controlled X, set
//! per shot to prepare the data qubits in a pattern of 0s and 1s. The circuit
//! is Clifford and its detectors deterministic, so each bit's effect on every
//! detector is fixed: one more noiseless run with that bit set finds it. A
//! shot's detector then fires when its records' parity differs from the
//! reference's, after the fixed flips of the shot's set sweep bits.
//!
//! The runs use the stabilizer tableau, not the frame machinery the error model
//! is built on, so this is a second, independent reading of the circuit.
//! Google's own detection events are its oracle.

use crate::circuit::{Basis, Circuit, Instr, Resolved};
use crate::simulator::StabilizerSimulator;

pub struct M2d {
    pub num_measurements: usize,
    pub num_sweep_bits: usize,
    pub num_detectors: usize,
    pub num_observables: usize,
    detectors: Vec<Vec<usize>>,
    observables: Vec<Vec<usize>>,
    ref_det: Vec<bool>,
    ref_obs: u64,
    /// Per sweep bit, the detectors it flips, and the observables.
    sweep_det: Vec<Vec<u32>>,
    sweep_obs: Vec<u64>,
}

fn reset(sim: &mut StabilizerSimulator, basis: Basis, q: usize) {
    match basis {
        Basis::Z => {
            if sim.measure_z(q) == 1 {
                sim.apply_x(q);
            }
        }
        Basis::X => {
            if sim.measure_x(q) == 1 {
                sim.apply_z(q);
            }
        }
    }
}

/// One noiseless run of the circuit with the given sweep bits: its measurement record.
fn run(res: &Resolved, sweeps: &[bool], seed: u64) -> Vec<bool> {
    let mut sim = StabilizerSimulator::with_seed(res.num_qubits.max(1), seed);
    let mut rec = Vec::with_capacity(res.num_measurements);
    for ins in &res.instrs {
        match ins {
            Instr::Reset { basis, qubits } => {
                for &q in qubits {
                    reset(&mut sim, *basis, q as usize);
                }
            }
            Instr::H(qubits) => qubits.iter().for_each(|&q| sim.apply_h(q as usize)),
            Instr::Cx(pairs) => pairs.iter().for_each(|&(c, t)| sim.apply_cnot(c as usize, t as usize)),
            Instr::Cz(pairs) => {
                for &(a, b) in pairs {
                    sim.apply_h(b as usize);
                    sim.apply_cnot(a as usize, b as usize);
                    sim.apply_h(b as usize);
                }
            }
            Instr::Pauli { pauli, qubits } => {
                for &q in qubits {
                    match pauli {
                        1 => sim.apply_x(q as usize),
                        2 => sim.apply_z(q as usize),
                        3 => sim.apply_y(q as usize),
                        _ => {}
                    }
                }
            }
            Instr::SweepX(pairs) => {
                for &(k, q) in pairs {
                    if sweeps[k as usize] {
                        sim.apply_x(q as usize);
                    }
                }
            }
            Instr::Measure { basis, reset: then_reset, qubits, .. } => {
                for &q in qubits {
                    let q = q as usize;
                    let bit = match basis {
                        Basis::Z => sim.measure_z(q),
                        Basis::X => sim.measure_x(q),
                    };
                    rec.push(bit == 1);
                    if *then_reset {
                        reset(&mut sim, *basis, q);
                    }
                }
            }
            // Noise channels, annotations and ticks: the run is noiseless.
            _ => {}
        }
    }
    rec
}

fn parity(recs: &[usize], rec: &[bool]) -> bool {
    recs.iter().fold(false, |acc, &m| acc ^ rec[m])
}

fn evaluate(detectors: &[Vec<usize>], observables: &[Vec<usize>], rec: &[bool]) -> (Vec<bool>, u64) {
    let dets = detectors.iter().map(|r| parity(r, rec)).collect();
    let mut obs = 0u64;
    for (k, r) in observables.iter().enumerate() {
        if parity(r, rec) {
            obs |= 1 << k;
        }
    }
    (dets, obs)
}

impl M2d {
    pub fn new(circuit: &Circuit) -> Result<M2d, String> {
        let res = circuit.resolve()?;
        let (detectors, observables) = (res.detectors.clone(), res.observables.clone());
        let clear = vec![false; res.num_sweep_bits];
        let (ref_det, ref_obs) = evaluate(&detectors, &observables, &run(&res, &clear, 1));
        // A second reference, with other random outcomes, must agree.
        let (det2, obs2) = evaluate(&detectors, &observables, &run(&res, &clear, 2));
        if let Some(d) = (0..ref_det.len()).find(|&d| ref_det[d] != det2[d]) {
            return Err(format!("detector D{d} is not deterministic"));
        }
        if ref_obs != obs2 {
            return Err("an observable is not deterministic".into());
        }
        let mut sweep_det = Vec::with_capacity(res.num_sweep_bits);
        let mut sweep_obs = Vec::with_capacity(res.num_sweep_bits);
        for k in 0..res.num_sweep_bits {
            let mut bits = clear.clone();
            bits[k] = true;
            let (det, obs) = evaluate(&detectors, &observables, &run(&res, &bits, 3 + k as u64));
            sweep_det.push((0..det.len()).filter(|&d| det[d] != ref_det[d]).map(|d| d as u32).collect());
            sweep_obs.push(obs ^ ref_obs);
        }
        Ok(M2d {
            num_measurements: res.num_measurements,
            num_sweep_bits: res.num_sweep_bits,
            num_detectors: detectors.len(),
            num_observables: observables.len(),
            detectors,
            observables,
            ref_det,
            ref_obs,
            sweep_det,
            sweep_obs,
        })
    }

    /// One shot's detection events and observable flips.
    pub fn convert(&self, meas: &[bool], sweeps: &[bool]) -> (Vec<bool>, u64) {
        let (mut det, mut obs) = evaluate(&self.detectors, &self.observables, meas);
        for (d, &r) in det.iter_mut().zip(&self.ref_det) {
            *d ^= r;
        }
        obs ^= self.ref_obs;
        for (k, &set) in sweeps.iter().enumerate() {
            if set {
                for &d in &self.sweep_det[k] {
                    det[d as usize] ^= true;
                }
                obs ^= self.sweep_obs[k];
            }
        }
        (det, obs)
    }

    /// Many shots in Stim's b8 layout, rows padded to whole bytes.
    pub fn convert_b8(&self, meas: &[u8], sweeps: &[u8], num_shots: usize) -> Result<(Vec<u8>, Vec<u8>), String> {
        let (ms, ss) = (self.num_measurements.div_ceil(8), self.num_sweep_bits.div_ceil(8));
        if meas.len() != ms * num_shots || sweeps.len() != ss * num_shots {
            return Err(format!(
                "{} + {} bytes is not {num_shots} shots of {} measurements and {} sweep bits",
                meas.len(),
                sweeps.len(),
                self.num_measurements,
                self.num_sweep_bits
            ));
        }
        let unpack = |row: &[u8], n: usize| -> Vec<bool> { (0..n).map(|i| (row[i / 8] >> (i % 8)) & 1 == 1).collect() };
        let (mut dets, mut obs) = (Vec::new(), Vec::new());
        for s in 0..num_shots {
            let m = unpack(&meas[s * ms..(s + 1) * ms], self.num_measurements);
            let w = unpack(&sweeps[s * ss..(s + 1) * ss], self.num_sweep_bits);
            let (d, o) = self.convert(&m, &w);
            crate::shots::pack_row(&d, &mut dets);
            let ob: Vec<bool> = (0..self.num_observables).map(|k| (o >> k) & 1 == 1).collect();
            crate::shots::pack_row(&ob, &mut obs);
        }
        Ok((dets, obs))
    }
}
```

Check `shots::pack_row`'s signature (`pack_row(bits: &[bool], out: &mut Vec<u8>)`) and whether a zero-length row appends nothing. The test expects empty observable bytes when there are no observables; adjust that test line if `pack_row` pads to one byte.

In `src/py_api.rs`:

```rust
/// Raw measurements and sweep bits (b8) to detection events and observable
/// flips (b8), as `stim m2d` gives them.
#[pyfunction]
fn m2d_b8<'py>(
    py: Python<'py>,
    circuit_text: &str,
    meas: &[u8],
    sweeps: &[u8],
    num_shots: usize,
) -> PyResult<(Bound<'py, PyBytes>, Bound<'py, PyBytes>)> {
    let c = Circuit::parse(circuit_text).map_err(err)?;
    let m = crate::m2d::M2d::new(&c).map_err(err)?;
    let (d, o) = py.allow_threads(|| m.convert_b8(meas, sweeps, num_shots)).map_err(err)?;
    Ok((PyBytes::new_bound(py, &d), PyBytes::new_bound(py, &o)))
}

/// A circuit through this engine's parser and printer.
#[pyfunction]
fn circuit_to_stim(text: &str) -> PyResult<String> {
    Ok(Circuit::parse(text).map_err(err)?.to_stim())
}
```

Register both in `register`.

- [ ] **Step 3: Run.** `cargo test --release --no-default-features` (all pass), `cargo build --release`, then rebuild the Python module.

- [ ] **Step 4: Oracle on synthetic data.** Run from `/tmp`:

```bash
.venv/bin/python - <<'EOF'
import stim, numpy as np, stabilizer_qec as sq
c = stim.Circuit.generated("surface_code:rotated_memory_x", distance=3, rounds=4, after_clifford_depolarization=0.01)
# Put a sweep bit on each data qubit after the first reset, as Google's circuits do.
text = str(c.flattened())
lines = text.splitlines()
first_tick = lines.index("TICK")
data = [q for q in range(c.num_qubits) if q % 2 == 1][:9]
lines.insert(first_tick, "CX " + " ".join(f"sweep[{k}] {q}" for k, q in enumerate(data)))
circ = stim.Circuit("\n".join(lines))
shots = 5000
sweeps = np.random.default_rng(1).integers(0, 2, size=(shots, 9)).astype(bool)
meas = circ.compile_sampler(seed=3).sample(shots)  # sweeps all zero when sampled
conv = circ.compile_m2d_converter()
dets, obs = conv.convert(measurements=meas, sweep_bits=sweeps, separate_observables=True)
mp = np.packbits(meas, axis=1, bitorder="little").tobytes()
sp = np.packbits(sweeps, axis=1, bitorder="little").tobytes()
d_b, o_b = sq.m2d_b8(str(circ), mp, sp, shots)
ours = np.unpackbits(np.frombuffer(d_b, np.uint8).reshape(shots, -1), axis=1, bitorder="little")[:, :circ.num_detectors]
ours_o = np.unpackbits(np.frombuffer(o_b, np.uint8).reshape(shots, -1), axis=1, bitorder="little")[:, :1]
print("detectors equal:", (ours == dets).all(), " observables equal:", (ours_o == obs).all())
EOF
```

Expected: both `True`. Stim's generated circuits use `MR`, `RX`, `MX`, `H` and `CX`, so this exercises every reset and measurement kind the parser has. The sweep flips are converted even though the sampler never applied them, which is exactly the relationship m2d must reproduce.

- [ ] **Step 5: Commit.** `feat(m2d): raw measurements to detection events, by noiseless tableau runs`.

---

### Task 3: Google's data, checked

**Files:** Create `tools/google.py`; produce `data/google-results/checks.json`.

**Interfaces:**
- Produces:
  - `experiments(dataset) -> list[Exp]`, where `Exp` has `dataset, name, patch, basis, d, rounds, prefix`;
  - `read(exp, key) -> bytes`, with keys `ideal noisy meas sweep dets obs dem:<name> pred:<name>`;
  - `obs_bits(exp, key) -> np.ndarray[uint8]`, one 0/1 per shot, from b8 or 01;
  - the `check`, `run`, `summary` and `extract` commands.

- [ ] **Step 1: Write the tool's reading layer and `check`:**

```python
"""Google's surface-code memory data, decoded by this engine.

Run from the repository root, after building the engine into .venv:

    VIRTUAL_ENV=$PWD/.venv CARGO_TARGET_DIR=target-py .venv/bin/maturin develop --profile python
    .venv/bin/python tools/google.py check      # parse, m2d and model checks; writes data/google-results/checks.json
    .venv/bin/python tools/google.py run        # every prior, both matchers; writes data/google-results/{willow,sycamore}.json
    .venv/bin/python tools/google.py summary    # every failed check, unmatchable shot and model mismatch
    .venv/bin/python tools/google.py extract    # the site's live extract; writes data/willow-extract/

The datasets are read straight from the zips in data/google/ (see its fetch.sh),
licensed CC BY 4.0 by Google Quantum AI: Zenodo records 13273331 (Willow,
"Quantum error correction below the surface code threshold") and 6804040
(Sycamore, "Suppressing quantum errors by scaling a surface code logical qubit").
Spec: docs/superpowers/specs/2026-09-25-google-data-design.md, Part 2.
"""
```

Implementation notes, which the code must follow:
- `Exp` is a dataclass. Willow names are `willow/{patch}/{basis}/r{NN}`; Sycamore names are `sycamore/{dir}`.
- Willow's round count comes from `metadata.json`. Sycamore's comes from `properties.yml`, parsed as `key: value` lines, with no YAML dependency.
- `patch` is the Willow patch directory (`d5_at_q4_7`), or the Sycamore `center_{r}_{c}`.
- File keys map as in the "Facts" section. Willow `pred:{pathway}` is `decoding_results/{pathway}/obs_flips_predicted.b8`, and `dem:{pathway}` likewise. Sycamore `pred:{name}` is `obs_flips_predicted_by_{name}.01`, and `dem:{name}` is `{name}.dem`.
- Keep one `zipfile.ZipFile` per dataset open for the process (module-level cache).
- `obs_bits`: for b8 with one observable, `np.frombuffer(raw, np.uint8) & 1`; for 01, `np.frombuffer(raw, np.uint8)[::2] - ord('0')`. Assert the length equals `shots`.
- Counts come from Stim: `stim.Circuit(ideal)`, with `.num_measurements`, `.num_detectors` and `.num_sweep_bits`.
- `check` runs, for each experiment:
  1. **Parse**: `stim.Circuit(sq.circuit_to_stim(text)) == stim.Circuit(text)` for the ideal and the noisy circuit.
  2. **m2d**: `sq.m2d_b8(ideal, meas, sweep, shots)` must equal Google's `detection_events.b8` bytes exactly, and its observable bits must equal `obs_bits(exp, 'obs')`. Record mismatching shot counts; zero is the only pass.
  3. **Model against Stim**, on the noisy circuit: reuse `tools/xcheck.py`'s `check_model` (import it with `sys.path.insert(0, str(ROOT / 'tools'))`; its module-level code only checks the build). Record `missing, extra, differing, max_rel, edges_one_sided, splits_differ, edges_max_rel`.
  4. **Model against Google's prior** (Willow: the SI1000 pathway's DEM; Sycamore: `circuit_detector_error_model.dem`). Use the `mechanisms()` and `graph()` helpers from `xcheck`, and record how many mechanisms and edges are only ours, only theirs, and common, plus the median and maximum relative Δp on common edges.
  - Write `data/google-results/checks.json`: `{"generated": date, "stim": version, "experiments": {name: {...}}}`. Print one line per experiment, and a final tally of how many passed parse and m2d.
- `--dataset willow|sycamore|all` and `--limit N` restrict what runs (for quick trials).

- [ ] **Step 2: A quick trial.** Run: `.venv/bin/python tools/google.py check --dataset sycamore --limit 3` and then `--dataset willow --limit 3`. Expected: parse and m2d pass, with zero mismatching shots.
  - If m2d fails, diagnose on one shot: compare our reference record with `stim.Circuit(ideal).reference_sample()`, then compare the sweep effects. Don't guess.

- [ ] **Step 3: The full check.** Run `.venv/bin/python tools/google.py check` in the background, with output to the scratchpad. Expected: 550 experiments, all parsing and converting bit for bit (27.5 million shots). The model comparisons are recorded as they come out: against Stim they should be identical, as in A's checks; against Google's priors, the differences are reported.

- [ ] **Step 4: Commit** `tools/google.py` and `data/google-results/checks.json`: `feat(google): every Google experiment parsed, and its detection events reproduced bit for bit`.

---

### Task 4: The full decoding run

**Files:** Modify `tools/google.py` (`run`, `summary`); produce `data/google-results/willow.json` and `sycamore.json`.

- [ ] **Step 1: Write `run`.** For each experiment, with the shots as Google's `detection_events.b8` bytes (identical to ours after Task 3) and the truth from `obs_bits(exp, 'obs')`:
  - **Willow** priors, each decoded plain and correlated by `sq.decode_b8(dem_text, dets, shots, 0, correlated)`:
    - `si1000`: the `correlated_matching_decoder_with_si1000_prior` DEM;
    - `rl`: the `correlated_matching_decoder_with_rl_optimized_prior` DEM;
    - `ours`: `sq.decode_b8_own(noisy_text, dets, shots, 0, correlated)`.
  - **Sycamore** priors:
    - `circuit`: `circuit_detector_error_model.dem`;
    - `ours`: from `circuit_noisy.stim`;
    - `pij`: cross-fitted. Even rows go through `pij_from_odd_for_even`, odd rows through `pij_from_even_for_odd`. Take the rows with numpy reshaping and reassemble the predictions into shot order.
  - For every configuration, record `failures` (a decode error counts as a failure), `errors`, `seconds` and `threads` (`os.cpu_count()`).
  - Score Google's predictions: Willow's five pathways, and Sycamore's four files, against the same truth.
  - Also record `mean_defects` and `shots`.
  - **Sycamore check:**
    - For shots where our plain `circuit` prediction differs from Google's recorded PyMatching, get PyMatching 2.4's optimal weight on the same DEM (`pymatching.Matching.from_detector_error_model(dem).decode(dets[i], return_weight=True)`), and compare it with our weight from `decode_b8`'s weight output.
    - The tolerance comes from xcheck's method: noise measured on 300 agreeing shots, times 10, floored at 1e-6.
    - Record `pm_disagree`, `pm_ours_optimal` (our weight within tolerance of PyMatching 2.4's) and `pm_ours_not_optimal`. The last must be 0.
  - **Output and resume:** after each experiment, rewrite `data/google-results/{dataset}.json` as `{"generated", "engine_commit" (git rev-parse --short HEAD), "machine", "experiments": {name: record}}`, and skip experiments already present unless `--redo`. Print one line per experiment.
- [ ] **Step 2: Write `summary`.** It prints, from both result files and `checks.json`:
  - every failed check;
  - every configuration with `errors > 0`;
  - every Sycamore `pm_ours_not_optimal > 0`;
  - total shots decoded per dataset, and total decoding seconds.
- [ ] **Step 3: A quick trial.** `run --dataset sycamore --limit 2`, then `--dataset willow --limit 2`. Check the records by eye: failure rates rise with rounds, and correlated is at or below plain.
- [ ] **Step 4: The full run.** Run it in the background, logging to the scratchpad. Expected cost: tens of minutes to a couple of hours on 10 cores, dominated by d = 7 at long rounds. Then run `summary`. Every problem it lists is explained in the README, or fixed and the affected experiments re-run with `--redo`.
- [ ] **Step 5: Commit** `tools/google.py` and `data/google-results/*.json`: `feat(google): 27.5 million real shots decoded with every prior, plain and correlated`.

---

### Task 5: The fits

**Files:** Create `js/lambda-fit.js` and `tools/lambda.mjs`. Modify `tools/site-tests.mjs`.

**Interfaces:**
- Produces, from `js/lambda-fit.js`:
  - `fidelity(failures, shots) -> {F, sigma}`;
  - `fitEpsilon(points, {minRounds}) -> {eps, A, n, ok}`, where `points` are `{rounds, failures, shots}`;
  - `epsilonByDistance(records, decoderKey, {minRounds}) -> Map<d, {eps, fits}>`;
  - `lambdaFit(epsByD) -> {lambda, pairwise: [{from, to, lambda}]}`;
  - `bootstrap(records, decoderKey, opts, B, rng) -> {eps: Map<d, [lo, hi]>, lambda: [lo, hi]}`;
  - `decoderKeys(records)`.
- `records` is the array of experiment records from `data/google-results/*.json`, each `{d, patch, basis, rounds, shots, results: {key: failures}}`, where `key` is `ours/si1000/correlated` or `google/<pathway>`.

- [ ] **Step 1: Failing tests** in `tools/site-tests.mjs`:
  - `fitEpsilon` recovers ε = 0.3% from exact fidelities `F = 0.98 (1 − 2ε)^r` at r = 10…250 to 1e-9.
  - It excludes r < minRounds and points with F < 3σ.
  - `lambdaFit` on ε = {3: 0.8%, 5: 0.4%, 7: 0.2%} gives Λ = 2 exactly, and pairwise ratios of 2.
  - `bootstrap` with a seeded RNG brackets the point estimate.
  - `epsilonByDistance` averages patches and bases.
- [ ] **Step 2: Implement `js/lambda-fit.js`:**
  - **ε:** a weighted least-squares line through (r, ln F) with weights `(F/σ_F)²`. ε = (1 − e^slope)/2. Points with `F ≤ 3σ_F` or `rounds < minRounds` are dropped, and fewer than 2 points is `ok: false`.
  - **Λ:** an unweighted least-squares line through (d, ln ε_d). Λ = exp(−2·slope).
  - **Bootstrap:** redraw each point's failures from a normal approximation to the binomial (mean n·P, variance n·P(1−P), rounded and clamped to [0, n]), refit everything, and take the 2.5% and 97.5% percentiles. B defaults to 400. The RNG is injectable; the default is `Math.random`.
- [ ] **Step 3: `tools/lambda.mjs`.** It loads both result files and prints:
  - For Willow: per decoder key, ε₃, ε₅, ε₇, Λ₃/₅, Λ₅/₇ and Λ, with bootstrap intervals, at `minRounds = 10`, plus a robustness line at `minRounds = 30`.
  - For Sycamore: per decoder, ε₃, ε₅ and Λ at `minRounds = 3`, with alternatives at 1 and 5.
  - The validation line: our fit of Google's own tensor-network predictions against the published 3.028% ± 0.023% and 2.914% ± 0.016%. If it misses, try the start rounds 1, 3 and 5 and weighting on or off, and use the variant that reproduces the published numbers. Record which in the README. If none reproduces them, say so and quote both.
- [ ] **Step 4: Run** `node tools/site-tests.mjs` and `node tools/lambda.mjs | tee data/google-results/lambda.txt`.
- [ ] **Step 5: Commit** `feat(fit): logical error per cycle and Λ, one fit for ours and Google's`.

---

### Task 6: The live panel's engine calls

**Files:** Create `src/wasm_hw.rs`. Modify `src/lib.rs`, `js/engine.js`, `js/worker.js`.

**Interfaces:**
- WASM exports:
  - `wasm_bytes_buf(len) -> *mut u8` and `wasm_bytes_ptr() -> *const u8`, a second buffer for binary data;
  - `wasm_hw_m2d(num_shots) -> usize`, which parses the circuit in the text buffer, reads the measurement rows followed by the sweep rows from the bytes buffer, writes detection-event rows followed by observable rows back, and returns the byte length, or 0 with an error in the text buffer;
  - `wasm_hw_model_circuit(slot) -> usize` and `wasm_hw_model_dem(slot) -> usize`, which build decoder `slot` from the noisy circuit or the DEM in the text buffer and return a JSON reply;
  - `wasm_hw_decode(slot, num_shots, correlated) -> usize`, which decodes detection rows in the bytes buffer and writes one byte per shot back (the observable-0 prediction, or 255 on an error), returning the error count.
- JS:
  - `engine.js`: `hwM2d(instance, circuitText, meas, sweeps, shots)`, `hwModel(instance, slot, {circuit} | {dem})`, `hwDecode(instance, slot, dets, shots, correlated)`;
  - `worker.js`: op `hardware` with payload `{experiments: [{d, files}]}`, reporting progress per step.

- [ ] **Step 1: Implement `src/wasm_hw.rs`** in the text-buffer style of `wasm_xc.rs`. Decoders live in a `static mut [Option<DemDecoder>; 2]`. A Rust unit test (not wasm-specific) round-trips `wasm_hw_m2d` on the Task 2 hand example by calling the exported functions directly.
- [ ] **Step 2: The JS wrappers and worker op.** For each experiment the op:
  1. fetches the extract files (Task 7);
  2. runs `hwM2d`, then hashes the detection rows with SHA-256 and compares the hash with the manifest's;
  3. builds slot 0 from Google's SI1000 DEM and slot 1 from the noisy circuit;
  4. decodes plain and correlated in each, timing each;
  5. scores against `obs_flips_actual` and each Google pathway's predictions, and counts shot-by-shot agreement with Google's `correlated_matching_decoder_with_si1000_prior`.
- [ ] **Step 3: Build and test.**
  - `cargo test --release --no-default-features wasm_hw`;
  - the wasm32 build, copied to `stabilizer_qec.wasm`;
  - `node tools/site-tests.mjs`;
  - a Node smoke test: instantiate the WASM in Node, run `hwM2d` on the hand example, and assert the bytes.
- [ ] **Step 4: Commit** `feat(wasm): raw readouts to decoded predictions in the browser`.

---

### Task 7: The extract

**Files:** `tools/google.py extract`; produces `data/willow-extract/`.

- [ ] **Step 1: Choose the patches.**
  - d7 is `d7_at_q6_7`.
  - For d3 and d5, take the patches whose centres are nearest `(6, 7)`. List the patch names and choose by the numbers in the names: `d3_at_q6_7` if it exists, and the d5 patch nearest in centre. Record the choice in the manifest.
  - Z basis, 30 rounds, first 2,000 shots.
- [ ] **Step 2: Write, per experiment `d{d}/`:**
  - `circuit_ideal.stim` and `circuit_noisy_si1000.stim`;
  - `measurements.b8` and `sweep_bits.b8` (first 2,000 rows);
  - `obs_flips_actual.b8` (2,000 bytes);
  - `error_model_si1000.dem` (the SI1000 correlated-matching pathway's);
  - `pred_{pathway}.b8` for the five pathways.

  At the top, `manifest.json` records per experiment the patch, d, rounds, shots, measurements, detectors and sweep bits, and `detection_events_sha256` over Google's first 2,000 detection rows. It also records the credit and licence text, and the source zip, record and path.
- [ ] **Step 3: Check the size.** `du -sh data/willow-extract`; it should come in under 3 MB. If it doesn't, drop d7 to 1,000 shots and say so in the manifest.
- [ ] **Step 4: Commit** `feat(extract): 6,000 real Willow shots, raw, for the live panel`.

---

### Task 8: Section 11

**Files:** Modify `js/plot.js`, `index.html`, `css/styles.css` and `js/main.js`. Create `js/sections/hardware.js`.

- [ ] **Step 1: A log y axis in `Plot`.** This is the option `yLog: true`.
  - The y range becomes decades: from the floor of log10 of the smallest value to the ceiling of log10 of the largest.
  - Gridlines go at each decade, and at 2× and 5× with lighter rules. Labels read as percentages via `formatY`.
  - Error bars and points transform with it. Existing linear charts are unchanged.
  - A site test renders nothing in Node, since canvas isn't available. So pull the tick computation out as an exported pure `logTicks(min, max)` and test that.
- [ ] **Step 2: The markup.** Section 11, "Real hardware", goes after section 10:
  - one prose paragraph: what Google measured, what Λ is, and where the data comes from, with CC BY credit;
  - **Figure 9** (`data-hw-fit`, meta "† recorded run, fitted here"):
    - a canvas of ε_d against d, with the legend;
    - a Willow table (decoder; ε₃, ε₅, ε₇; Λ, with intervals), with Google's published Λ = 2.14 ± 0.02 as its own row labelled "neural-network decoder, published";
    - a Sycamore table;
    - the check lines;
  - **Figure 10** (`data-hw-live`, "decoded here, from raw readouts"):
    - a table per distance: shots, SHA-256 match, and failures for ours (SI1000 prior plain and correlated, our prior plain and correlated) and Google's five;
    - agreement with Google's correlated matcher, and µs per shot;
    - a status line and a note that 2,000 shots can't pin down Λ.
- [ ] **Step 3: `js/sections/hardware.js`.**
  - **Figure 9:** `initHardware(root, compute)` loads `data/google-results/{willow,sycamore}.json` and fits them with `js/lambda-fit.js` in the main thread (cheap, 400 bootstrap draws). Then it plots and fills the tables.
  - **Figure 10:** when scrolled into view, it runs the worker's `hardware` op and fills the table as each distance completes.
  - Formatting helpers go in a pure `js/hardware-format.js`, tested in site tests.
  - Wire `initHardware($('#hardware'), compute)` in `main.js`, and add the section to the nav if the nav lists sections.
- [ ] **Step 4: CSS.** Reuse `.figure`, `.xcheck` and `.table-wrap`, adding only what's missing. It must work at phone width. Keep the site's "figure from a paper" look: no new colours beyond the tokens, and series in `--d3`, `--d5`, `--d7`, `--ink-2`, `--x`, `--z`, `--y`.
- [ ] **Step 5: Verify in headless Chrome** with `scratchpad/cdp-shot.mjs`:
  - at 1440 px and 390 px, both figures render;
  - the live panel reaches "Done." with matching hashes and no console errors;
  - screenshots are taken.

  Then run `node tools/site-tests.mjs` and `node tools/contrast.mjs`.
- [ ] **Step 6: Commit** `feat(site): section 11, Google's hardware decoded; the recorded fit and a live panel from raw readouts`.

---

### Task 9: The README, verification and review

- [ ] **Step 1: README section "Google's hardware data"**, after "Correlated matching". It covers:
  - the datasets and their credit;
  - the parser additions, and the m2d check with its count;
  - the model comparisons against Stim and against Google's priors;
  - the run: shots, priors, time and machine;
  - the Willow and Sycamore tables from `data/google-results/lambda.txt`;
  - the fit method and its validation against Sycamore's published numbers;
  - the Sycamore optimality check;
  - what the published Λ = 2.14 is and isn't;
  - the site extract.

  Also update Key Features and Repository Structure (`src/m2d.rs`, `src/wasm_hw.rs`, `tools/google.py`, `data/google-results/`, `data/willow-extract/`, `js/lambda-fit.js`, `js/sections/hardware.js`).
- [ ] **Step 2: Verify everything:**

```bash
cargo test --release --no-default-features
cargo build --release
cargo build --release --target wasm32-unknown-unknown --no-default-features && cmp target/wasm32-unknown-unknown/release/stabilizer_qec.wasm stabilizer_qec.wasm
node tools/site-tests.mjs && node tools/contrast.mjs
.venv/bin/python tools/xcheck.py --quick
.venv/bin/python tools/google.py summary
node tools/lambda.mjs
```

Then check the page in headless Chrome: Figures 8, 9 and 10 finish, with no console errors.
- [ ] **Step 3: Commit** the README: `docs(google): Google's hardware data, Λ, and how it was checked`.
- [ ] **Step 4: Review.** Run the `code-review` skill at `high` over this plan's commits. Fix the confirmed findings, re-run Step 2, and commit.

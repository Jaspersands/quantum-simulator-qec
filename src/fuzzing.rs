//! What the fuzzers drive: an input text taken through every stage that reads it. Each stage
//! may refuse the input with an error; none may panic, hang or exhaust memory. Shared by the
//! randomized test below (stable Rust, in CI) and the `cargo fuzz` targets in `fuzz/`.

use crate::batch_sampler::BatchSampler;
use crate::belief::BeliefMatching;
use crate::bp::Method;
use crate::circuit::Circuit;
use crate::dem::Dem;
use crate::dem_decoder::DemDecoder;
use crate::frame_sampler::FrameSampler;
use crate::m2d::M2d;
use crate::osd::{BpOsd, OsdMethod};
use crate::surface_code::Xorshift;
use crate::window::{Mode, Model, WindowDecoder};

/// Sampling and m2d run only on inputs this small, so a fuzzer's huge-but-valid circuit (a
/// billion-round loop, qubit 16,777,215) costs it nothing; the limits that refuse what is too
/// large to model are exercised on every input regardless.
const SMALL_QUBITS: usize = 512;
const SMALL_MEASUREMENTS: usize = 4096;

/// A circuit's text through the parser, the printer (whose text must parse to the same
/// circuit), the resolver, both error models, both samplers, m2d and the decoders built on
/// its model.
pub fn exercise_circuit(text: &str) {
    let Ok(c) = Circuit::parse(text) else { return };
    let printed = c.to_stim();
    let again = Circuit::parse(&printed).unwrap_or_else(|e| panic!("the printer wrote text the parser refuses ({e}):\n{printed}"));
    assert!(again == c, "printing and parsing again changed the circuit:\n{printed}");
    let resolved = c.resolve();
    // Error models only for inputs a fuzzer can afford many of: the limits themselves were
    // exercised by resolve() above.
    let affordable = matches!(&resolved, Ok(r) if r.num_qubits <= 1 << 16 && r.instrs.len() <= 1 << 18);
    for decompose in [false, true] {
        if !affordable {
            break;
        }
        let dem = if decompose { Dem::from_circuit(&c) } else { Dem::from_circuit_undecomposed(&c) };
        if let Ok(dem) = dem {
            exercise_model(&dem);
            let printed = dem.to_stim(decompose);
            Dem::parse(&printed).unwrap_or_else(|e| panic!("the model printer wrote text the parser refuses ({e}):\n{printed}"));
        }
    }
    let Ok(batch) = BatchSampler::new(&c) else { return };
    let Ok(res) = resolved else { return };
    if res.num_qubits > SMALL_QUBITS || res.num_measurements > SMALL_MEASUREMENTS {
        return;
    }
    let mut rng = Xorshift::new(1);
    let b = batch.sample(&mut rng);
    assert_eq!(b.detectors.len(), batch.num_detectors);
    let (d, o) = batch.sample_seeded(5, 0, 70, 2);
    assert_eq!(d.len(), 70 * batch.num_detectors.div_ceil(8));
    assert_eq!(o.len(), 70 * batch.num_observables.div_ceil(8));
    if let Ok(frame) = FrameSampler::new(&c) {
        let shot = frame.sample(&mut rng);
        assert_eq!(shot.detectors.len(), frame.num_detectors());
    }
    if let Ok(m2d) = M2d::new(&c) {
        let (ms, ss) = (m2d.num_measurements.div_ceil(8), m2d.num_sweep_bits.div_ceil(8));
        let meas: Vec<u8> = (0..3 * ms).map(|i| (i * 37 % 256) as u8).collect();
        let _ = m2d.convert_b8(&meas, &vec![0xA5; 3 * ss], 3);
    }
}

/// A model through the matcher (plain and correlated), belief-matching, BP+OSD and the window
/// decoders, each decoding a few made-up syndromes.
pub fn exercise_model(dem: &Dem) {
    let nd = dem.num_detectors;
    if nd > 100_000 || dem.mechanisms.len() > 200_000 {
        return;
    }
    let syndromes: Vec<Vec<u32>> = (0..4u32).map(|k| (0..nd as u32).filter(|d| (d * 7 + k * 3) % 5 == 0).collect()).collect();
    if let Ok(dec) = DemDecoder::new(dem) {
        for s in &syndromes {
            let _ = dec.decode(s);
            let _ = dec.decode_correlated(s);
        }
    }
    if let Ok(bm) = BeliefMatching::from_dem(dem, Method::ProductSum, 5) {
        let mut w = bm.work();
        for s in &syndromes {
            let _ = bm.decode(s, &mut w);
        }
    }
    let columns: Vec<Vec<u32>> = dem.mechanisms.iter().map(|m| m.detectors.clone()).collect();
    let priors: Vec<f64> = dem.mechanisms.iter().map(|m| m.p).collect();
    if dem.mechanisms.len() <= 2000 {
        if let Ok(osd) = BpOsd::new(nd, columns, &priors, Method::MinSum { scale: 0.0 }, 5, OsdMethod::CombinationSweep(2)) {
            let mut w = osd.work();
            for s in &syndromes {
                let mut dense = vec![0u8; nd];
                for &d in s {
                    dense[d as usize] = 1;
                }
                let _ = osd.decode(&dense, &mut w);
            }
        }
    }
    for (commit, buffer, mode) in [(1, 1, Mode::Sliding), (2, 1, Mode::Parallel), (0, 0, Mode::Parallel)] {
        let Ok(model) = Model::new(dem) else { break };
        let Ok(wd) = WindowDecoder::new(model, commit, buffer, mode) else { continue };
        let mut scratches = wd.scratches();
        let mut live = vec![false; nd];
        for &d in &syndromes[0] {
            live[d as usize] = true;
        }
        let mut obs = 0u64;
        'windows: for phase in &wd.phases {
            for &wi in phase {
                if wd.decode_window(wi, &mut live, &mut obs, true, &mut scratches[wi]).is_err() {
                    break 'windows;
                }
            }
        }
    }
}

/// A model's text through the parser and everything built on a model.
pub fn exercise_dem(text: &str) {
    if let Ok(dem) = Dem::parse(text) {
        exercise_model(&dem);
        let printed = dem.to_stim(true);
        Dem::parse(&printed).unwrap_or_else(|e| panic!("the model printer wrote text the parser refuses ({e}):\n{printed}"));
    }
}

/* -- Inputs ------------------------------------------------------------------- */

/// Random circuit texts from the grammar, with edge values, then mutated.
pub struct Inputs {
    rng: Xorshift,
}

impl Inputs {
    pub fn new(seed: u64) -> Inputs {
        Inputs { rng: Xorshift::new(seed) }
    }

    fn below(&mut self, n: u64) -> u64 {
        self.rng.next_u64() % n.max(1)
    }

    fn pick<'a>(&mut self, xs: &[&'a str]) -> &'a str {
        xs[self.below(xs.len() as u64) as usize]
    }

    fn number(&mut self) -> String {
        match self.below(12) {
            0 => self.pick(&["0", "-1", "1e300", "nan", "inf", "-0", "0.5", "1", "2", "1e-320", "", "+3"]).to_string(),
            1 => self.pick(&["16777215", "16777216", "4294967295", "4294967296", "18446744073709551615", "99999999999999999999"]).to_string(),
            _ => self.below(8).to_string(),
        }
    }

    fn probability(&mut self) -> String {
        match self.below(8) {
            0 => self.number(),
            1 => "0".into(),
            _ => format!("{}", (self.below(1000) as f64) / 4000.0),
        }
    }

    fn targets(&mut self, pairs: bool) -> String {
        let n = if pairs { 2 * (1 + self.below(3)) } else { 1 + self.below(4) };
        (0..n)
            .map(|_| match self.below(20) {
                0 => self.number(),
                1 => format!("rec[-{}]", self.below(4)),
                2 => format!("sweep[{}]", self.below(3)),
                3 => format!("!{}", self.below(5)),
                _ => self.below(6).to_string(),
            })
            .collect::<Vec<_>>()
            .join(" ")
    }

    fn line(&mut self, depth: u32) -> String {
        const ONE: &[&str] = &[
            "H", "X", "Y", "Z", "I", "R", "RX", "RZ", "RY", "M", "MX", "MZ", "MY", "MR", "MRX", "MRZ", "MRY", "S", "S_DAG",
            "SQRT_X", "SQRT_Y_DAG", "H_XY", "C_XYZ", "C_ZNYX", "I_ERROR", "MPAD", "HERALDED_ERASE",
        ];
        const TWO: &[&str] = &[
            "CX", "CNOT", "ZCX", "CZ", "ZCZ", "CY", "XCY", "YCZ", "SWAP", "ISWAP", "ISWAP_DAG", "CXSWAP", "SQRT_YY", "II", "MXX",
            "MYY", "MZZ", "PAULI_CHANNEL_2",
        ];
        const NOISE1: &[&str] = &["X_ERROR", "Y_ERROR", "Z_ERROR", "DEPOLARIZE1"];
        match self.below(16) {
            0..=3 => format!("{} {}", self.pick(ONE), self.targets(false)),
            4 | 5 => format!("{} {}", self.pick(TWO), self.targets(true)),
            6 => format!("{}({}) {}", self.pick(NOISE1), self.probability(), self.targets(false)),
            7 => format!("DEPOLARIZE2({}) {}", self.probability(), self.targets(true)),
            8 => format!("PAULI_CHANNEL_1({}, {}, {}) {}", self.probability(), self.probability(), self.probability(), self.targets(false)),
            9 => format!("M({}) {}", self.probability(), self.targets(false)),
            10 | 11 => format!(
                "DETECTOR({}, {}) {}",
                self.number(),
                self.number(),
                (0..1 + self.below(3)).map(|_| format!("rec[-{}]", 1 + self.below(5))).collect::<Vec<_>>().join(" ")
            ),
            12 => format!("OBSERVABLE_INCLUDE({}) rec[-{}]", self.pick(&["0", "1", "63", "64", "-1", "2.5"]), 1 + self.below(3)),
            13 => format!("{}({}, {}) {}", self.pick(&["QUBIT_COORDS", "SHIFT_COORDS"]), self.number(), self.number(), self.targets(false)),
            14 if self.below(2) == 0 => {
                let product = |s: &mut Self| {
                    (0..1 + s.below(3))
                        .map(|_| format!("{}{}{}", if s.below(4) == 0 { "!" } else { "" }, s.pick(&["X", "Y", "Z", "W"]), s.below(5)))
                        .collect::<Vec<_>>()
                        .join("*")
                };
                match self.below(4) {
                    0 => format!("MPP({}) {} {}", self.probability(), product(self), product(self)),
                    1 => format!("{} {}", self.pick(&["SPP", "SPP_DAG"]), product(self)),
                    2 => format!("{}({}) {}", self.pick(&["E", "ELSE_CORRELATED_ERROR"]), self.probability(), product(self).replace('*', " ")),
                    _ => format!("PAULI_CHANNEL_2({}) 0 1", (0..15).map(|_| self.probability()).collect::<Vec<_>>().join(", ")),
                }
            }
            14 if self.below(2) == 0 => {
                let q = self.below(5);
                let control = self.pick(&["rec[-1]", "rec[-2]", "rec[-0]", "rec[-9]", "sweep[0]", "sweep[2]"]);
                match self.below(4) {
                    0 => format!("{} {control} {q}", self.pick(&["CX", "CY", "CZ"])),
                    1 => format!("{} {q} {control}", self.pick(&["CZ", "XCZ", "YCZ", "CX"])),
                    2 => format!("HERALDED_ERASE({}) {}", self.probability(), self.targets(false)),
                    _ => format!(
                        "HERALDED_PAULI_CHANNEL_1({}, {}, {}, {}) {}",
                        self.probability(),
                        self.probability(),
                        self.probability(),
                        self.probability(),
                        self.targets(false)
                    ),
                }
            }
            14 => self.pick(&["TICK", "", "# a comment", "  ", "}", "{"]).to_string(),
            _ if depth < 3 => {
                let count = match self.below(6) {
                    0 => self.number(),
                    1 => "1000000000".into(),
                    _ => (1 + self.below(4)).to_string(),
                };
                let body: Vec<String> = (0..1 + self.below(4)).map(|_| self.line(depth + 1)).collect();
                format!("REPEAT {count} {{\n{}\n}}", body.join("\n"))
            }
            _ => "TICK".into(),
        }
    }

    /// A circuit: a few resets and measurements so records exist, then random lines.
    pub fn circuit(&mut self) -> String {
        let mut lines = vec!["R 0 1 2 3".to_string(), "M 0 1".to_string()];
        lines.extend((0..1 + self.below(12)).map(|_| self.line(0)));
        let text = lines.join("\n");
        if self.below(4) == 0 {
            self.mutate(&text)
        } else {
            text
        }
    }

    /// A detector error model from the grammar, sometimes mutated.
    pub fn dem(&mut self) -> String {
        let mut lines = Vec::new();
        for _ in 0..1 + self.below(10) {
            lines.push(match self.below(8) {
                0 => format!("detector({}, {}) D{}", self.number(), self.number(), self.below(6)),
                1 => format!("logical_observable L{}", self.pick(&["0", "1", "63", "64"])),
                2 => format!("shift_detectors({}) {}", self.number(), self.below(4)),
                3 if self.below(3) == 0 => format!("repeat {} {{\nerror(0.1) D0 D1\nshift_detectors 2\n}}", self.number()),
                _ => {
                    let pieces: Vec<String> = (0..1 + self.below(3))
                        .map(|_| (0..1 + self.below(4)).map(|_| if self.below(5) == 0 { format!("L{}", self.below(3)) } else { format!("D{}", self.below(8)) }).collect::<Vec<_>>().join(" "))
                        .collect();
                    format!("error({}) {}", self.probability(), pieces.join(" ^ "))
                }
            });
        }
        let text = lines.join("\n");
        if self.below(4) == 0 {
            self.mutate(&text)
        } else {
            text
        }
    }

    fn mutate(&mut self, text: &str) -> String {
        let mut bytes = text.as_bytes().to_vec();
        for _ in 0..1 + self.below(4) {
            if bytes.is_empty() {
                break;
            }
            let i = self.below(bytes.len() as u64) as usize;
            match self.below(4) {
                0 => {
                    bytes.remove(i);
                }
                1 => {
                    const ALPHABET: &[u8] = b"0123456789-+.e[](){}!^ \n\tDLrecswp";
                    bytes.insert(i, ALPHABET[self.below(ALPHABET.len() as u64) as usize])
                }
                2 => bytes[i] = self.below(128) as u8,
                _ => {
                    let j = self.below(bytes.len() as u64) as usize;
                    let piece = bytes[i.min(j)..i.max(j)].to_vec();
                    bytes.splice(i..i, piece);
                }
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(cases: u64, seed: u64) {
        let mut inputs = Inputs::new(seed);
        for case in 0..cases {
            let (c, d) = (inputs.circuit(), inputs.dem());
            let r = std::panic::catch_unwind(|| exercise_circuit(&c));
            assert!(r.is_ok(), "case {case}: the circuit panicked:\n{c}");
            let r = std::panic::catch_unwind(|| exercise_dem(&d));
            assert!(r.is_ok(), "case {case}: the model panicked:\n{d}");
        }
    }

    #[test]
    fn random_inputs_never_panic() {
        run(20_000, 1);
    }

    #[test]
    #[ignore = "a longer fuzzing run: cargo test --release -- --ignored fuzz"]
    fn fuzz_random_inputs_at_length() {
        run(200_000, 2);
    }
}

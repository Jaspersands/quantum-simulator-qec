//! Detector error models, flat: which detectors and logical observables each elementary
//! fault flips, and how likely it is. The form the decoders take.
//!
//! A model is built by walking its circuit backwards, as Stim's error analyzer does, folding
//! the loops that repeat (`dem_build.rs`); it is held, read and printed as Stim's program,
//! `repeat` blocks and all (`dem_program.rs`); and it is unrolled into a `Dem` for a decoder.
//!
//! Going backwards through a gate conjugates what an X and a Z error would flip: H swaps them;
//! CX sends `sx[c] ^= sx[t]` and `sz[t] ^= sz[c]`; CZ sends `sx[a] ^= sz[b]` and `sx[b] ^=
//! sz[a]`. A Z-basis measurement adds what reads its record to `sx`; a reset clears both, since
//! an error before a reset is erased by it. The same walk checks determinism for free: reaching
//! a Z-basis reset with `sz` non-empty means some detector anticommutes with the state the
//! reset prepares, so its value is a coin flip. That is an error, not a warning.

use std::collections::HashMap;
use std::fmt::Write as _;

use crate::circuit::{fmt_args, Circuit};

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
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
    /// mechanism. Empty for a hyperedge nobody decomposed. Faults that share a
    /// symptom but split differently are separate mechanisms, each with its own
    /// probability, as in Stim's decomposed models.
    pub pieces: Vec<Piece>,
    /// The Stim tag of the instruction the fault came from (empty for none). Faults with
    /// different tags stay separate mechanisms, as in Stim.
    pub tag: String,
}

#[derive(Clone, Debug, Default)]
pub struct Dem {
    pub num_detectors: usize,
    pub num_observables: usize,
    pub detector_coords: Vec<Vec<f64>>,
    pub mechanisms: Vec<Mechanism>,
    /// Detector tags, by detector; shorter than the detectors when the last have none.
    pub detector_tags: Vec<String>,
    /// Observable tags, by observable; shorter likewise.
    pub observable_tags: Vec<String>,
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

/// `a·b + c`, rounded as Stim's build for this machine rounds it: clang fuses the product into
/// the sum on ARM64 (one rounding); an x86-64 build cannot (two).
pub fn fused(a: f64, b: f64, c: f64) -> f64 {
    if cfg!(target_arch = "aarch64") {
        a.mul_add(b, c)
    } else {
        a * b + c
    }
}

/// Independent (qx, qy, qz) equivalent to `PAULI_CHANNEL_1(px, py, pz)`, found as Stim finds
/// them (`try_disjoint_to_independent_xyz_errors_approx`): the cases relabelled so that
/// identity is the likeliest, the exact solution where one exists, and otherwise 50 Newton
/// steps, accepted if they come within 10⁻¹⁴. Where they do not, the channel has no
/// independent equivalent, and an error model holds it only approximately.
pub fn pauli_channel_1_independent(px: f64, py: f64, pz: f64) -> Result<(f64, f64, f64), String> {
    match disjoint_to_independent(px, py, pz, 50) {
        (true, [a, b, c]) => Ok((a, b, c)),
        _ => Err(format!(
            "PAULI_CHANNEL_1({px}, {py}, {pz}) has no equivalent set of independent errors"
        )),
    }
}

fn disjoint_to_independent(x: f64, y: f64, z: f64, max_steps: usize) -> (bool, [f64; 3]) {
    let i = (1.0 - x - y - z).max(0.0);
    if i < x {
        let (ok, [a, b, c]) = disjoint_to_independent(i, z, y, max_steps);
        return (ok, [1.0 - a, b, c]);
    }
    if i < y {
        let (ok, [a, b, c]) = disjoint_to_independent(z, i, x, max_steps);
        return (ok, [a, 1.0 - b, c]);
    }
    if i < z {
        let (ok, [a, b, c]) = disjoint_to_independent(y, x, i, max_steps);
        return (ok, [a, b, 1.0 - c]);
    }
    if x + z < 0.5 && x + y < 0.5 && y + z < 0.5 {
        let s_xz = (1.0 - 2.0 * x - 2.0 * z).sqrt();
        let s_xy = (1.0 - 2.0 * x - 2.0 * y).sqrt();
        let s_yz = (1.0 - 2.0 * y - 2.0 * z).sqrt();
        let a = 0.5 - 0.5 * s_xz * s_xy / s_yz;
        let b = 0.5 - 0.5 * s_xy * s_yz / s_xz;
        let c = 0.5 - 0.5 * s_xz * s_yz / s_xy;
        if a >= 0.0 && b >= 0.0 && c >= 0.0 {
            return (true, [a, b, c]);
        }
    }
    let (mut a, mut b, mut c) = (x, y, z);
    for _ in 0..max_steps {
        let (ab, ac, bc) = (a * b, a * c, b * c);
        let (a_i, b_i, c_i) = (1.0 - a, 1.0 - b, 1.0 - c);
        let (ab_i, ac_i, bc_i) = (a_i * b_i, a_i * c_i, b_i * c_i);
        let x2 = fused(a, bc_i, a_i * bc);
        let y2 = fused(b, ac_i, b_i * ac);
        let z2 = fused(c, ab_i, c_i * ab);
        let (dx, dy, dz) = (x2 - x, y2 - y, z2 - z);
        if dx.abs() + dy.abs() + dz.abs() < 1e-14 {
            return (true, [a, b, c]);
        }
        // Stim's step, its third derivative as written there (`ab_i - ac`).
        let (da, db, dc) = (bc_i - bc, ac_i - ac, ab_i - ac);
        a = (a - dx / da).max(0.0);
        b = (b - dy / db).max(0.0);
        c = (c - dz / dc).max(0.0);
    }
    (false, [a, b, c])
}

/// Probability that exactly one of two independent events happens.
pub fn xor_prob(a: f64, b: f64) -> f64 {
    a * (1.0 - b) + b * (1.0 - a)
}

impl Dem {
    pub fn from_circuit(circuit: &Circuit) -> Result<Dem, String> {
        Dem::build(circuit, true, None)
    }

    /// Either model, with Stim's `approximate_disjoint_errors`: `None` refuses channels whose
    /// cases are disjoint rather than independent (PAULI_CHANNEL_2, ELSE_CORRELATED_ERROR,
    /// the heralded errors, and PAULI_CHANNEL_1 where no independent equivalent exists);
    /// `Some(t)` approximates each case as an independent fault, refusing a channel with an
    /// argument above `t` (`Some(1.0)` is Stim's `True`).
    pub fn from_circuit_with(
        circuit: &Circuit,
        decompose: bool,
        approximate: Option<f64>,
    ) -> Result<Dem, String> {
        Dem::build(circuit, decompose, approximate)
    }

    /// The model without splitting faults into graph-like pieces: one
    /// mechanism per symptom, faults with the same symptom merged. What BP and
    /// BP+OSD decode, and the only model a code whose faults fire three or
    /// more checks has (a fault of one or two detectors keeps itself as its
    /// one piece).
    pub fn from_circuit_undecomposed(circuit: &Circuit) -> Result<Dem, String> {
        Dem::build(circuit, false, None)
    }

    fn build(circuit: &Circuit, decompose: bool, approximate: Option<f64>) -> Result<Dem, String> {
        crate::dem_build::build(circuit, decompose, approximate, true)?.to_dem_merged()
    }
}

/* -- Stim's .dem text ------------------------------------------------------ */

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
        let tagged = |t: &str| {
            if t.is_empty() {
                String::new()
            } else {
                format!("[{t}]")
            }
        };
        for (i, c) in self.detector_coords.iter().enumerate() {
            let tag = tagged(self.detector_tags.get(i).map_or("", |t| t.as_str()));
            if c.is_empty() {
                let _ = writeln!(s, "detector{tag} D{i}");
            } else {
                let _ = writeln!(s, "detector{tag}({}) D{i}", fmt_args(c));
            }
        }
        // Observables no fault flips are declared, so the model keeps its count; tagged ones,
        // so it keeps their tags.
        let flipped = self
            .mechanisms
            .iter()
            .fold(0u64, |acc, m| acc | m.observables);
        for k in 0..self.num_observables.min(64) {
            let tag = self.observable_tags.get(k).map_or("", |t| t.as_str());
            if !tag.is_empty() || (flipped >> k) & 1 == 0 {
                let _ = writeln!(s, "logical_observable{} L{k}", tagged(tag));
            }
        }
        for m in &self.mechanisms {
            let _ = write!(s, "error{}({})", tagged(&m.tag), m.p);
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

    /// Stim's text read (repeat blocks unrolled) into a flat model, faults in the text's order.
    pub fn parse(text: &str) -> Result<Dem, String> {
        crate::dem_program::DemProgram::parse(text)?.to_dem()
    }
}

/* -- Comparison ------------------------------------------------------------ */

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
        let e = out
            .entry((m.detectors.clone(), m.observables))
            .or_insert(0.0);
        *e = xor_prob(*e, m.p);
    }
    out
}

/// Mechanism-by-mechanism comparison, after merging identical symptoms on each side.
pub fn compare(ours: &Dem, theirs: &Dem, tol: f64) -> Comparison {
    let a = merged(ours);
    let b = merged(theirs);
    let mut c = Comparison {
        ours: a.len(),
        theirs: b.len(),
        missing: 0,
        extra: 0,
        differing: 0,
        max_rel: 0.0,
    };
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixtures::{REP3, SAMPLE};

    fn dem(text: &str) -> Dem {
        Dem::from_circuit(&Circuit::parse(text).unwrap()).unwrap()
    }

    fn p_of(d: &Dem, dets: &[u32], obs: u64) -> f64 {
        d.mechanisms
            .iter()
            .find(|m| m.detectors == dets && m.observables == obs)
            .map(|m| m.p)
            .unwrap()
    }

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() <= 1e-12 * a.abs().max(b.abs())
    }

    /// Numbers from Stim 1.16, `circuit.detector_error_model()`.
    #[test]
    fn depolarizing_channels_match_stim() {
        let d = dem("R 0\nDEPOLARIZE1(0.01) 0\nM 0\nDETECTOR rec[-1]");
        assert_eq!(d.mechanisms.len(), 1);
        assert!(
            close(d.mechanisms[0].p, 0.006666666666666613),
            "{}",
            d.mechanisms[0].p
        );

        let d = dem("R 0 1\nDEPOLARIZE2(0.01) 0 1\nM 0 1\nDETECTOR rec[-1]\nDETECTOR rec[-2]");
        assert_eq!(d.mechanisms.len(), 3);
        for m in &d.mechanisms {
            assert!(close(m.p, 0.002673815958446298), "{}", m.p);
        }
    }

    #[test]
    fn pauli_channel_and_flips_match_stim() {
        let d = dem(
            "R 0\nPAULI_CHANNEL_1(0.01, 0.02, 0.03) 0\nM 0\nDETECTOR rec[-1]\n\
                     RX 1\nPAULI_CHANNEL_1(0.01, 0.02, 0.03) 1\nMX 1\nDETECTOR rec[-1]",
        );
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
        let d =
            dem("R 0\nRX 1\nX_ERROR(0.1) 0\nCZ 0 1\nM 0\nMX 1\nDETECTOR rec[-2]\nDETECTOR rec[-1]");
        assert!(close(p_of(&d, &[0, 1], 0), 0.1));
        // H swaps: X before H flips an X-basis readout, Z before H does not.
        let d = dem("R 0\nX_ERROR(0.1) 0\nZ_ERROR(0.2) 0\nH 0\nMX 0\nDETECTOR rec[-1]");
        assert_eq!(d.mechanisms.len(), 1);
        assert!(close(p_of(&d, &[0], 0), 0.1));
    }

    #[test]
    fn nondeterministic_detectors_are_errors() {
        let e =
            Dem::from_circuit(&Circuit::parse("RX 0\nM 0\nDETECTOR rec[-1]").unwrap()).unwrap_err();
        assert!(e.contains("not deterministic"), "{e}");
        let e =
            Dem::from_circuit(&Circuit::parse("M 0\nMX 0\nDETECTOR rec[-1]").unwrap()).unwrap_err();
        assert!(e.contains("not deterministic"), "{e}");
    }

    #[test]
    fn undetectable_logical_errors_are_errors() {
        let e = Dem::from_circuit(
            &Circuit::parse("R 0\nX_ERROR(0.1) 0\nM 0\nOBSERVABLE_INCLUDE(0) rec[-1]").unwrap(),
        )
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
        let d = Dem::parse(
            "error(0.1) D0 D1 ^ D2\nrepeat 2 {\n  error(0.2) D0\n  shift_detectors 1\n}\ndetector(1, 2) D0\n",
        )
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
        assert_eq!(
            (c.ours, c.theirs, c.missing, c.extra, c.differing),
            (2, 3, 1, 0, 1)
        );
        assert!((c.max_rel - 0.2).abs() < 1e-12);
    }

    #[test]
    fn pauli_gates_and_sweep_bits_leave_the_model_alone() {
        // They flip signs, never which detectors a fault sets off.
        let base = Circuit::parse(crate::fixtures::REP3).unwrap();
        let text =
            crate::fixtures::REP3.replacen("TICK\n", "TICK\nX 0 1\nY 2\nCX sweep[0] 1\nI 0\n", 1);
        let with = Circuit::parse(&text).unwrap();
        assert_ne!(with, base);
        assert_eq!(
            Dem::from_circuit(&with).unwrap().to_stim(true),
            Dem::from_circuit(&base).unwrap().to_stim(true)
        );
    }

    /// A model as a set of lines, "p pieces", each piece's targets sorted and
    /// the pieces sorted: Stim's piece order within a line is not part of the
    /// model.
    fn normalised(dem: &Dem) -> Vec<String> {
        let mut out: Vec<String> = dem
            .mechanisms
            .iter()
            .map(|m| {
                let mut pieces: Vec<String> = m
                    .pieces
                    .iter()
                    .map(|pc| {
                        let mut t: Vec<String> =
                            pc.detectors.iter().map(|d| format!("D{d}")).collect();
                        t.extend(
                            (0..64)
                                .filter(|i| (pc.observables >> i) & 1 == 1)
                                .map(|i| format!("L{i}")),
                        );
                        t.sort();
                        t.join(" ")
                    })
                    .collect();
                pieces.sort();
                let p = format!("{:.9}", m.p);
                format!(
                    "{} {}",
                    p.trim_end_matches('0').trim_end_matches('.'),
                    pieces.join(" ^ ")
                )
            })
            .collect();
        out.sort();
        out
    }

    /// Where two faults fire the same detectors with different observables (a
    /// weight-two logical), a wider fault is split as Stim splits it: by the
    /// piece of the error class Stim sorts last, with a piece of observables
    /// alone where the pieces' observables fall short of the fault's. The
    /// first case differs from the second only in the order of two faults,
    /// which would change the piece if the first one found were kept. The
    /// expected models are Stim 1.16's.
    #[test]
    fn conflicting_pieces_split_as_stim_splits_them() {
        let cases: &[(&str, &str, &[&str])] = &[
        ("obs-on-q1-reordered", "R 0 1 2\nX_ERROR(0.05) 0\nCX 0 2\nX_ERROR(0.2) 1\nX_ERROR(0.1) 0\nX_ERROR(0.1) 2\nM 0 1 2\nDETECTOR rec[-3] rec[-2]\nDETECTOR rec[-3] rec[-2]\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-2]\n", &["0.05 D0 D1 L0 ^ D2 ^ L0", "0.1 D0 D1", "0.1 D2", "0.2 D0 D1 L0"]),
        ("obs-on-q1", "R 0 1 2\nX_ERROR(0.05) 0\nCX 0 2\nX_ERROR(0.1) 0\nX_ERROR(0.2) 1\nX_ERROR(0.1) 2\nM 0 1 2\nDETECTOR rec[-3] rec[-2]\nDETECTOR rec[-3] rec[-2]\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-2]\n", &["0.05 D0 D1 L0 ^ D2 ^ L0", "0.1 D0 D1", "0.1 D2", "0.2 D0 D1 L0"]),
        ("obs-on-q0", "R 0 1 2\nX_ERROR(0.05) 0\nCX 0 2\nX_ERROR(0.1) 0\nX_ERROR(0.2) 1\nX_ERROR(0.1) 2\nM 0 1 2\nDETECTOR rec[-3] rec[-2]\nDETECTOR rec[-3] rec[-2]\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-3]\n", &["0.05 D0 D1 L0 ^ D2", "0.1 D0 D1 L0", "0.1 D2", "0.2 D0 D1"]),
        ("split-through-q1", "R 0 1 2\nX_ERROR(0.05) 1\nCX 1 2\nX_ERROR(0.1) 0\nX_ERROR(0.2) 1\nX_ERROR(0.1) 2\nM 0 1 2\nDETECTOR rec[-3] rec[-2]\nDETECTOR rec[-3] rec[-2]\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-2]\n", &["0.05 D0 D1 L0 ^ D2", "0.1 D0 D1", "0.1 D2", "0.2 D0 D1 L0"]),
        ("three-way", "R 0 1 2 3\nX_ERROR(0.05) 0\nCX 0 3\nX_ERROR(0.04) 2\nCX 2 3\nX_ERROR(0.1) 0\nX_ERROR(0.2) 1\nX_ERROR(0.15) 2\nX_ERROR(0.1) 3\nM 0 1 2 3\nDETECTOR rec[-4] rec[-3] rec[-2]\nDETECTOR rec[-4] rec[-3] rec[-2]\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-3]\nOBSERVABLE_INCLUDE(1) rec[-2]\n", &["0.04 D0 D1 L1 ^ D2", "0.05 D0 D1 L1 ^ D2 ^ L1", "0.1 D0 D1", "0.1 D2", "0.15 D0 D1 L1", "0.2 D0 D1 L0"]),
        ];
        for &(name, text, stim) in cases {
            let dem = Dem::from_circuit(&Circuit::parse(text).unwrap()).unwrap();
            assert_eq!(
                normalised(&dem),
                stim.iter().map(|s| s.to_string()).collect::<Vec<_>>(),
                "{name}"
            );
        }
    }
}

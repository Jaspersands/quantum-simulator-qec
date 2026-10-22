//! Colour-code decoding by matching: Chromobius's Möbius construction (Gidney and Jones,
//! arXiv:2312.08813).
//!
//! WHY THIS EXISTS
//! ---------------
//! A colour-code fault sets off up to three detectors, one of each colour, so matching cannot
//! decode it directly: matching pairs detection events. Chromobius turns the problem into one
//! matching can solve. Every detector gets two copies, one in each of the two sub-graphs that
//! leave out a colour other than its own (a red detector is in "not green" and "not blue"), and
//! every basic fault becomes edges between copies: a three-colour fault becomes a triangle of
//! edges across the three sub-graphs, a two-detector fault two parallel edges, a single-
//! detector fault an edge joining its two copies. The doubled graph is a Möbius strip over the
//! code. Each detection event lights both its copies; a minimum-weight matching of the copies
//! is found by PyMatching's sparse blossom (`pm_exact`); the matched edges, with each event's
//! two copies joined, fall into cycles; and each cycle is lifted back to the code by carrying
//! colour charge around it, which says which observables the correction flips.
//!
//! This module is a port of Chromobius (quantumlib/chromobius, `src/chromobius`) in its order
//! of operations: the same basic faults (`collect_atomic_errors`), the same decomposition of
//! every other fault into them (`collect_composite_errors`), the same Möbius model, and the
//! same charge, drag and representative tables (`graphs`) and lifting (`decode`). The lifting
//! depends on the order of the matched edges, so the matching is PyMatching's to the edge and
//! to the order: Chromobius 1.1.1 links PyMatching 2.2.2, whose graph lists each node's
//! neighbours in the order its model's edges reach it (`pm_exact::NeighbourOrder::V2_2`), with
//! 2²⁴ weight levels, and `decode_detection_events_to_edges` lists the edges (`pm_exact`'s
//! `Mwpm::try_decode_to_edges`). Predictions and weights equal Chromobius's on every one of
//! 65,000 colour-code shots checked on macOS. Separately, Chromobius walks the charge graph's
//! neighbours in `std::unordered_map` order, which differs between C++ libraries, when it
//! looks for the shortest way to drag charge between two places; where two such ways flip
//! different observables the table entry is marked ambiguous, and a shot whose lifting uses one
//! is flagged as tied (no colour-code shot checked has needed one; small hand-made models with
//! faults above 1/2 have).

mod decode;
mod graphs;

use std::collections::BTreeMap;

use crate::dem::Dem;
use crate::dem_program::{DemInstr, DemProgram};

pub use decode::{ColorDecoder, ColorError, ColorWork};

pub(crate) const BOUNDARY: u32 = u32::MAX;

/// Colour charges: none, red, green, blue. Charges combine by XOR (R ^ G = B).
pub(crate) type Charge = u8;
pub(crate) const NEUTRAL: Charge = 0;
pub(crate) const G: Charge = 2;

/// The charge after `c` among the three colours, cyclically.
pub(crate) fn next_charge(c: Charge) -> Charge {
    c % 3 + 1
}

/// A detector's colour and basis (1 = X, 2 = Z), or `ignored` (4th coordinate −1).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct ColorBasis {
    pub color: Charge,
    pub basis: u8,
    pub ignored: bool,
}

/// A basic fault: up to three detectors, sorted, `BOUNDARY` filling the rest
/// (`AtomicErrorKey`). Ordered as Chromobius orders them, so maps iterate alike.
pub(crate) type Key = [u32; 3];

pub(crate) fn key(a: u32, b: u32, c: u32) -> Key {
    let mut k = [a, b, c];
    k.sort_unstable();
    k
}

fn key_of(dets: &[u32]) -> Key {
    key(dets.first().copied().unwrap_or(BOUNDARY), dets.get(1).copied().unwrap_or(BOUNDARY), dets.get(2).copied().unwrap_or(BOUNDARY))
}

const EMPTY: Key = [BOUNDARY; 3];

pub(crate) fn weight(k: &Key) -> usize {
    k.iter().filter(|&&d| d != BOUNDARY).count()
}

pub(crate) fn net_charge(k: &Key, colors: &[ColorBasis]) -> Charge {
    k.iter().filter(|&&d| d != BOUNDARY).fold(NEUTRAL, |c, &d| c ^ colors[d as usize].color)
}

/// The Möbius-graph edges of a basic fault (`AtomicErrorKey::iter_mobius_edges`).
pub(crate) fn mobius_edges(k: &Key, colors: &[ColorBasis]) -> Vec<(u32, u32)> {
    let [n1, n2, n3] = *k;
    if n1 == BOUNDARY {
        return Vec::new();
    }
    if n2 == BOUNDARY {
        return vec![(2 * n1, 2 * n1 + 1)];
    }
    if n3 == BOUNDARY {
        let flip = u32::from(colors[n1 as usize].color ^ colors[n2 as usize].color == G);
        return vec![(2 * n1, (2 * n2) ^ flip), (2 * n1 + 1, (2 * n2 + 1) ^ flip)];
    }
    let mut rgb = [BOUNDARY; 3];
    for n in [n1, n2, n3] {
        rgb[colors[n as usize].color as usize - 1] = n;
    }
    let [r, g, b] = rgb;
    // The copies: red in not-green (0) and not-blue (1), green in not-red (0) and not-blue (1),
    // blue in not-red (0) and not-green (1).
    let sorted = |a: u32, b: u32| (a.min(b), a.max(b));
    vec![sorted(2 * r + 1, 2 * g + 1), sorted(2 * g, 2 * b), sorted(2 * r, 2 * b + 1)]
}

/// Chromobius's options (`DecoderConfigOptions`).
#[derive(Clone, Copy, Debug)]
pub(crate) struct Options {
    pub drop_mobius_errors_involving_remnant_errors: bool,
    pub ignore_decomposition_failures: bool,
}

impl Default for Options {
    fn default() -> Self {
        Options { drop_mobius_errors_involving_remnant_errors: true, ignore_decomposition_failures: false }
    }
}

/// A model lifted to its Möbius matching problem, with the tables that lift a matching back.
pub(crate) struct Lifted {
    pub colors: Vec<ColorBasis>,
    /// The Möbius model as Stim's text: edges only, no observables.
    pub mobius_text: String,
    /// The Möbius model's detector declarations with coordinates (`collect_nodes`).
    pub mobius_decls: Vec<(Vec<f64>, u64)>,
    pub rgb_reps: Vec<graphs::RgbEdge>,
    pub charge: graphs::ChargeGraph,
    pub drag: graphs::DragGraph,
}

/// Each detector's colour and basis from its 4th coordinate, walking the model's declarations as
/// Chromobius does (`collect_nodes_from_dem`, `detector_instruction_to_color_basis`): a
/// declaration without a whole 4th coordinate in -1..=5 is refused with Chromobius's message,
/// and a detector declared twice takes its last declaration. Also the Möbius model's
/// declarations with coordinates (`include_coords_in_mobius_dem`, which `describe_decoder`
/// prints): each copy of a detector not ignored, at its detector's coordinates (shifted) with
/// its sub-graph appended (NotRed 1, NotGreen 2, NotBlue 3).
fn collect_nodes(program: &DemProgram, num_detectors: usize) -> Result<(Vec<ColorBasis>, Vec<(Vec<f64>, u64)>), String> {
    let mut colors = vec![ColorBasis::default(); num_detectors];
    let mut decls = Vec::new();
    let (mut offset, mut shift) = (0u64, Vec::new());
    collect_nodes_in(&program.instrs, &mut offset, &mut shift, &mut colors, &mut decls)?;
    Ok((colors, decls))
}

fn collect_nodes_in(instrs: &[DemInstr], offset: &mut u64, shift: &mut Vec<f64>, colors: &mut [ColorBasis], decls: &mut Vec<(Vec<f64>, u64)>) -> Result<(), String> {
    const MAPPING: [(Charge, u8); 6] = [(1, 1), (2, 1), (3, 1), (1, 2), (2, 2), (3, 2)];
    // The copies' sub-graphs: red in not-green and not-blue, green in not-red and not-blue,
    // blue in not-red and not-green.
    const SUBGRAPHS: [(f64, f64); 3] = [(2.0, 3.0), (1.0, 3.0), (1.0, 2.0)];
    for ins in instrs {
        match ins {
            DemInstr::Detector { coords, targets, .. } => {
                let c = if coords.len() > 3 { coords[3] + if shift.len() > 3 { shift[3] } else { 0.0 } } else { -2.0 };
                if coords.len() <= 3 || !(-1.0..=5.0).contains(&c) || (c as i32) as f64 != c {
                    return Err(format!("{UNCOLOURED}{}", instruction_text(ins)));
                }
                let cb = if c == -1.0 {
                    ColorBasis { color: NEUTRAL, basis: 0, ignored: true }
                } else {
                    let (color, basis) = MAPPING[c as usize];
                    ColorBasis { color, basis, ignored: false }
                };
                for &t in targets {
                    let n = t + *offset;
                    colors[n as usize] = cb;
                    if !cb.ignored {
                        // An offset is added only where the shift has one (`k < coord_offsets.size()`),
                        // so a coordinate past it keeps its value as written: -0 stays -0.
                        let mut at: Vec<f64> = coords.iter().enumerate().map(|(k, &v)| shift.get(k).map_or(v, |&o| v + o)).collect();
                        let (g0, g1) = SUBGRAPHS[cb.color as usize - 1];
                        at.push(g0);
                        decls.push((at.clone(), 2 * n));
                        *at.last_mut().expect("a sub-graph coordinate") = g1;
                        decls.push((at, 2 * n + 1));
                    }
                }
            }
            DemInstr::Shift { coords, by, .. } => {
                *offset += by;
                if shift.len() < coords.len() {
                    shift.resize(coords.len(), 0.0);
                }
                for (k, &v) in coords.iter().enumerate() {
                    shift[k] += v;
                }
            }
            DemInstr::Repeat { count, body, .. } => {
                for _ in 0..*count {
                    collect_nodes_in(body, offset, shift, colors, decls)?;
                }
            }
            DemInstr::Error { .. } | DemInstr::Observable { .. } => {}
        }
    }
    Ok(())
}

/// Chromobius's refusal of a detector without its colour (`detector_instruction_to_color_basis`,
/// `extract_obs_and_dets_from_error_instruction`), less the instruction it names.
const UNCOLOURED: &str = "Expected all detectors to have at least 4 coordinates, with the 4th identifying the basis and color (RedX=0, GreenX=1, BlueX=2, RedZ=3, GreenZ=4, BlueZ=5), but got ";

/// An instruction as Stim's `DemInstruction::str()` writes it (a fresh stream: precision 6).
fn instruction_text(ins: &DemInstr) -> String {
    DemProgram { instrs: vec![ins.clone()] }.to_text_with(crate::dem_program::fmt_g6).trim_end().to_string()
}

/// The refusals of `extract_obs_and_dets_from_error_instruction`, as `collect_atomic_errors`
/// meets them: error by error in the model's order (`iter_flatten_error_instructions`: loops
/// unrolled, detectors shifted), target by target as written, a detector neither coloured nor
/// ignored (one no declaration names) and an observable past L63 (Chromobius's observable masks
/// are 64 bits) are refused with Chromobius's messages, naming the shifted instruction.
fn check_error_targets(instrs: &[DemInstr], offset: &mut u64, colors: &[ColorBasis]) -> Result<(), String> {
    use crate::dem_program::OBS;
    for ins in instrs {
        match ins {
            DemInstr::Error { p, pieces, tag } => {
                for &t in pieces.iter().flatten() {
                    let refused = if t & OBS == 0 {
                        let cb = colors[(t + *offset) as usize];
                        !cb.ignored && cb.color == NEUTRAL
                    } else {
                        t & !OBS >= 64
                    };
                    if refused {
                        let shifted: Vec<Vec<u64>> = pieces.iter().map(|piece| piece.iter().map(|&u| if u & OBS == 0 { u + *offset } else { u }).collect()).collect();
                        let text = instruction_text(&DemInstr::Error { p: *p, pieces: shifted, tag: tag.clone() });
                        return Err(if t & OBS == 0 { format!("{UNCOLOURED}{text}") } else { format!("Max logical observable is L63 but a larger one appeared in '{text}'") });
                    }
                }
            }
            DemInstr::Shift { by, .. } => *offset += by,
            DemInstr::Repeat { count, body, .. } => {
                for _ in 0..*count {
                    check_error_targets(body, offset, colors)?;
                }
            }
            DemInstr::Detector { .. } | DemInstr::Observable { .. } => {}
        }
    }
    Ok(())
}

/// Chromobius's refusal of a model this crate cannot unroll into its flat form because it names
/// an observable past L63 (this crate's flat models, like Chromobius's masks, hold 64): the first
/// refusal `Decoder::from_dem` meets (a detector declared without its colour, then each error's
/// targets in order: a detector no declaration colours, an observable past L63). None where it
/// would not refuse there (no error names an observable past L63: Chromobius then decodes, its
/// predictions past L63 undefined), or the model is past this crate's other limits.
pub fn refusal_past_l63(program: &DemProgram) -> Option<String> {
    let stats = program.stats();
    if stats.num_observables <= 64 || stats.num_detectors > crate::dem_program::MAX_DETECTOR + 1 {
        return None;
    }
    // Within the unrolling limits the flat model has (so the walks below are bounded).
    program.check_unrolls().ok()?;
    let refusal = collect_nodes(program, stats.num_detectors as usize).and_then(|(colors, _)| check_error_targets(&program.instrs, &mut 0, &colors));
    refusal.err()
}

/// A fault's detectors less the ignored ones (`extract_obs_and_dets_from_error_instruction`).
fn fault_detectors(dets: &[u32], colors: &[ColorBasis]) -> Result<Vec<u32>, String> {
    let mut out = Vec::with_capacity(dets.len());
    for &d in dets {
        let cb = colors[d as usize];
        if cb.ignored {
            continue;
        }
        if cb.color == NEUTRAL {
            return Err(format!(
                "detector D{d} has no colour annotation: every detector a fault sets off needs at least 4 coordinates, the 4th identifying the basis and colour (RedX=0, GreenX=1, BlueX=2, RedZ=3, GreenZ=4, BlueZ=5)"
            ));
        }
        out.push(d);
    }
    Ok(out)
}

/// `extract_atomic_errors_from_dem_error_instruction_dets`: the fault as a basic fault, recorded
/// in `out` with its observables, if it is one; else the empty key.
fn extract_atomic(dets: &[u32], obs: u64, colors: &[ColorBasis], out: &mut BTreeMap<Key, u64>) -> Key {
    let cb = |i: usize| colors[dets[i] as usize];
    let k = match dets.len() {
        1 => key(dets[0], BOUNDARY, BOUNDARY),
        2 if cb(0).basis == cb(1).basis => key(dets[0], dets[1], BOUNDARY),
        3 if cb(0).color ^ cb(1).color ^ cb(2).color == NEUTRAL && cb(0).basis == cb(1).basis && cb(1).basis == cb(2).basis => key(dets[0], dets[1], dets[2]),
        _ => return EMPTY,
    };
    out.insert(k, obs);
    k
}

/// `try_grow_decomposition`: keep the better of splitting off `e1` (score 1) or `e2` (score 2)
/// where each is known, unless the rest would be a charged triplet.
fn try_grow(e1: Key, e2: Key, colors: &[ColorBasis], atomic: &BTreeMap<Key, u64>, out: &mut Key, best: &mut u8) {
    let (c1, c2) = (atomic.contains_key(&e1), atomic.contains_key(&e2));
    let score = u8::from(c1) + 2 * u8::from(c2);
    if score <= *best {
        return;
    }
    if score == 1 && weight(&e2) == 3 && net_charge(&e2, colors) != NEUTRAL {
        return;
    }
    if score == 2 && weight(&e1) == 3 && net_charge(&e1, colors) != NEUTRAL {
        return;
    }
    *out = if c2 { e2 } else { e1 };
    *best = score;
}

/// `decompose_single_basis_dets_into_atoms`: a basic fault to split off detectors of one basis
/// (in the order the buffer holds them), or the empty key.
fn decompose_single_basis(dets: &[u32], colors: &[ColorBasis], atomic: &BTreeMap<Key, u64>) -> Key {
    if dets.len() <= 3 {
        let solo = key_of(dets);
        if atomic.contains_key(&solo) {
            return solo;
        }
    }
    let b = |x: bool| usize::from(x);
    let mut best = 0u8;
    let mut out = EMPTY;
    match dets.len() {
        // (Chromobius tries the first detector alone, twice.)
        2 => {
            let a1 = key(dets[0], BOUNDARY, BOUNDARY);
            if atomic.contains_key(&a1) {
                return a1;
            }
        }
        3 => {
            for k1 in 0..3 {
                try_grow(key(dets[k1], BOUNDARY, BOUNDARY), key(dets[b(k1 == 0)], dets[1 + b(k1 <= 1)], BOUNDARY), colors, atomic, &mut out, &mut best);
            }
        }
        4 => {
            let mut k1 = 0;
            while k1 < 4 && best < 2 {
                for k2 in k1 + 1..4 {
                    let rest = key(dets[b(k1 == 0) + b(k2 <= 1)], dets[1 + b(k1 <= 1) + b(k2 <= 2)], BOUNDARY);
                    try_grow(key(dets[k1], dets[k2], BOUNDARY), rest, colors, atomic, &mut out, &mut best);
                }
                k1 += 1;
            }
            for k1 in 0..4 {
                let rest = key(dets[b(k1 == 0)], dets[1 + b(k1 <= 1)], dets[2 + b(k1 <= 2)]);
                try_grow(key(dets[k1], BOUNDARY, BOUNDARY), rest, colors, atomic, &mut out, &mut best);
            }
        }
        5 => {
            let mut k1 = 0;
            while k1 < 5 && best < 2 {
                for k2 in k1 + 1..5 {
                    let rest = key(dets[b(k1 == 0) + b(k2 <= 1)], dets[1 + b(k1 <= 1) + b(k2 <= 2)], dets[2 + b(k1 <= 2) + b(k2 <= 3)]);
                    try_grow(key(dets[k1], dets[k2], BOUNDARY), rest, colors, atomic, &mut out, &mut best);
                }
                k1 += 1;
            }
        }
        6 => {
            let mut k1 = 0;
            while k1 < 6 && best < 2 {
                for k2 in k1 + 1..6 {
                    for k3 in k2 + 1..6 {
                        let rest = key(
                            dets[b(k1 == 0) + b(k2 <= 1) + b(k3 <= 2)],
                            dets[1 + b(k1 <= 1) + b(k2 <= 2) + b(k3 <= 3)],
                            dets[2 + b(k1 <= 2) + b(k2 <= 3) + b(k3 <= 4)],
                        );
                        try_grow(key(dets[k1], dets[k2], dets[k3]), rest, colors, atomic, &mut out, &mut best);
                    }
                }
                k1 += 1;
            }
        }
        _ => {}
    }
    out
}

/// `decompose_dets_into_atoms`: the fault's basic pieces, splitting X and Z detectors apart,
/// two passes of known pieces and a third that takes what remains as a new piece (a remnant).
/// `Err` with the undecomposed detectors where some remain.
fn decompose(
    dets: &[u32],
    mut obs: u64,
    colors: &[ColorBasis],
    atomic: &BTreeMap<Key, u64>,
    atoms: &mut Vec<Key>,
    remnants: &mut BTreeMap<Key, u64>,
) -> Result<(), (Vec<u32>, Vec<u32>, u64)> {
    let mut x: Vec<u32> = dets.iter().copied().filter(|&d| colors[d as usize].basis == 1).collect();
    let mut z: Vec<u32> = dets.iter().copied().filter(|&d| colors[d as usize].basis != 1).collect();
    atoms.clear();
    for rep in 0..3 {
        for basis in [&mut x, &mut z] {
            let removed = if rep == 2 { extract_atomic(basis, obs, colors, remnants) } else { decompose_single_basis(basis, colors, atomic) };
            let w = weight(&removed);
            if w == 0 {
                continue;
            }
            for &d in &removed[..w] {
                if let Some(i) = basis.iter().position(|&b| b == d) {
                    basis.swap_remove(i);
                }
            }
            obs ^= atomic.get(&removed).or_else(|| remnants.get(&removed)).copied().unwrap_or(0);
            atoms.push(removed);
        }
    }
    if x.is_empty() && z.is_empty() {
        Ok(())
    } else {
        Err((x, z, obs))
    }
}

/// The coordinates of the detectors in `wanted`, as Stim's
/// `DetectorErrorModel::get_detector_coordinates` gives them: each detector's first declaration,
/// with the shift added only where the shift has an entry (so a negative zero past it keeps its
/// sign); a detector never declared is absent.
fn stim_detector_coords(program: &DemProgram, wanted: &std::collections::BTreeSet<u64>) -> BTreeMap<u64, Vec<f64>> {
    fn walk(instrs: &[DemInstr], offset: &mut u64, shift: &mut Vec<f64>, wanted: &std::collections::BTreeSet<u64>, out: &mut BTreeMap<u64, Vec<f64>>) {
        for ins in instrs {
            if out.len() == wanted.len() {
                return;
            }
            match ins {
                DemInstr::Detector { coords, targets, .. } => {
                    for &t in targets {
                        let n = t + *offset;
                        if wanted.contains(&n) && !out.contains_key(&n) {
                            out.insert(n, coords.iter().enumerate().map(|(k, &v)| shift.get(k).map_or(v, |&o| v + o)).collect());
                        }
                    }
                }
                DemInstr::Shift { coords, by, .. } => {
                    if shift.len() < coords.len() {
                        shift.resize(coords.len(), 0.0);
                    }
                    for (k, &v) in coords.iter().enumerate() {
                        shift[k] += v;
                    }
                    *offset += by;
                }
                DemInstr::Repeat { count, body, .. } => {
                    for _ in 0..*count {
                        if out.len() == wanted.len() {
                            return;
                        }
                        walk(body, offset, shift, wanted, out);
                    }
                }
                DemInstr::Error { .. } | DemInstr::Observable { .. } => {}
            }
        }
    }
    let mut out = BTreeMap::new();
    walk(&program.instrs, &mut 0, &mut Vec::new(), wanted, &mut out);
    out
}

/// Chromobius's message for a fault it cannot split into basic faults
/// (`decompose_dets_into_atoms`): the error instruction (unrolled; numbers at a fresh stream's
/// precision), the detectors left in each basis, the pieces split off, the observables left,
/// and each detector the instruction names with its coordinates (as Stim's
/// `get_detector_coordinates` gives them) and colour.
#[allow(clippy::too_many_arguments)]
fn decomposition_failure(
    program: &DemProgram,
    fault: usize,
    (x, z, obs): (&[u32], &[u32], u64),
    atoms: &[Key],
    atomic: &BTreeMap<Key, u64>,
    remnants: &BTreeMap<Key, u64>,
    colors: &[ColorBasis],
) -> String {
    use crate::dem_program::fmt_g6;
    use std::fmt::Write as _;
    let comma = |v: &[u32]| v.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(", ");
    let obs_list = |mask: u64| (0..64).filter(|k| mask >> k & 1 == 1).map(|k| format!(" L{k}")).collect::<String>();
    // The fault's instruction: the `fault`-th error of the unrolled model (`to_dem` makes one
    // fault per error, in order).
    let ins = program.flattened().ok().and_then(|flat| flat.instrs.into_iter().filter(|i| matches!(i, DemInstr::Error { .. })).nth(fault));
    let text = ins.as_ref().map_or(String::new(), |i| DemProgram { instrs: vec![i.clone()] }.to_text_with(fmt_g6));
    let mut s = String::from("Failed to decompose a complex error instruction into basic errors.\n");
    let _ = writeln!(s, "    The instruction (after shifting): {}", text.trim_end());
    let _ = writeln!(s, "    The undecomposed X detectors: [{}]", comma(x));
    let _ = writeln!(s, "    The undecomposed Z detectors: [{}]", comma(z));
    for e in atoms {
        let dets: String = e.iter().filter(|&&d| d != BOUNDARY).map(|d| format!(" D{d}")).collect();
        let l = atomic.get(e).or_else(|| remnants.get(e)).copied().unwrap_or(0);
        let _ = writeln!(s, "    Decomposed piece:{dets}{}", obs_list(l));
    }
    if obs != 0 {
        let _ = writeln!(s, "    The undecomposed observable mask:{}", obs_list(obs));
    }
    s += "    Detector data:\n";
    if let Some(DemInstr::Error { pieces, .. }) = &ins {
        let wanted = pieces.iter().flatten().copied().filter(|&t| t & crate::dem_program::OBS == 0).collect();
        let all = stim_detector_coords(program, &wanted);
        for &t in pieces.iter().flatten().filter(|&&t| t & crate::dem_program::OBS == 0) {
            let coords: Vec<String> = all.get(&t).map_or(&[][..], |c| &c[..]).iter().map(|&c| fmt_g6(c)).collect();
            let cb = colors[t as usize];
            let _ = writeln!(
                s,
                "        D{t}: coords=[{}] ColorBasis{{.color={}, .basis={}{}}}",
                coords.join(", "),
                ["NEUTRAL", "R", "G", "B"][cb.color as usize],
                ["UNKNOWN_BASIS", "X", "Z"][cb.basis as usize],
                if cb.ignored { ", .ignored=true" } else { "" }
            );
        }
    }
    s += "This problem can unfortunately be quite difficult to debug. Likely causes are:\n";
    s += "    (1) The source circuit has detectors with invalid color/basis annotations.\n";
    s += "    (2) The source circuit is producing errors too complex to decompose (e.g. more than 6 symptoms in one basis).\n";
    s += "    (3) chromobius is missing logic for a corner case present in the source circuit; a corner case that didn't appear in the test circuits used during development.\n";
    s
}

impl Lifted {
    /// `Decoder::from_dem`.
    pub fn from_dem(dem: &Dem, program: &DemProgram, options: Options) -> Result<Lifted, String> {
        let (colors, mobius_decls) = collect_nodes(program, dem.num_detectors)?;
        check_error_targets(&program.instrs, &mut 0, &colors)?;
        let mut faults = Vec::with_capacity(dem.mechanisms.len());
        for m in &dem.mechanisms {
            faults.push(fault_detectors(&m.detectors, &colors)?);
        }
        let mut atomic = BTreeMap::new();
        for (m, dets) in dem.mechanisms.iter().zip(&faults) {
            extract_atomic(dets, m.observables, &colors, &mut atomic);
        }
        let mut remnant_edges: BTreeMap<Key, u64> = BTreeMap::new();
        let mut text = String::new();
        let mut atoms = Vec::new();
        for (i, (m, dets)) in dem.mechanisms.iter().zip(&faults).enumerate() {
            if let Err((x, z, obs)) = decompose(dets, m.observables, &colors, &atomic, &mut atoms, &mut remnant_edges) {
                if !options.ignore_decomposition_failures {
                    return Err(decomposition_failure(program, i, (&x, &z, obs), &atoms, &atomic, &remnant_edges, &colors));
                }
            }
            if options.drop_mobius_errors_involving_remnant_errors && !remnant_edges.is_empty() {
                atoms.clear();
                remnant_edges.clear();
            }
            let mut targets = Vec::new();
            let mut corner = false;
            for atom in &atoms {
                corner |= atom[1] == BOUNDARY;
                for (a, b) in mobius_edges(atom, &colors) {
                    targets.push(format!("D{a} D{b}"));
                }
            }
            if !targets.is_empty() {
                let p = if corner { m.p * m.p } else { m.p };
                text += &format!("error({p}) {}\n", targets.join(" ^ "));
            }
        }
        for (k, obs) in remnant_edges {
            atomic.entry(k).or_insert(obs);
        }
        if !colors.is_empty() {
            text += &format!("detector D{}\n", 2 * colors.len() - 1);
        }
        let rgb_reps = graphs::choose_rgb_reps(&atomic, &colors);
        let charge = graphs::ChargeGraph::from_atomic(&atomic, colors.len());
        let drag = graphs::DragGraph::new(&charge, &atomic, &rgb_reps, &colors);
        Ok(Lifted { colors, mobius_text: text, mobius_decls, rgb_reps, charge, drag })
    }

    /// The decoder as Chromobius prints it (`operator<<(std::ostream &, const Decoder &)` with
    /// `include_coords_in_mobius_dem`, as `chromobius describe_decoder` writes it): the charge
    /// graph, each detector's representative, the drag graph and the Möbius model, its numbers
    /// as Stim prints a model's on this machine.
    pub fn describe(&self) -> Result<String, String> {
        use std::fmt::Write as _;
        let node = |n: u32| if n == BOUNDARY { "BOUNDARY_NODE".to_string() } else { n.to_string() };
        let charge = |c: Charge| ["NEUTRAL", "R", "G", "B"][c as usize];
        let mut s = String::from("chromobius::Decoder{\n\n.charge_graph=ChargeGraph{.nodes={\n");
        for (k, near) in self.charge.nodes.iter().enumerate() {
            let list: Vec<String> = near.iter().map(|(&n, &obs)| format!("{{{},{obs}}}", node(n))).collect();
            let _ = writeln!(s, "    ChargeGraphNode{{.neighbors={{{}}}}}, // node {k}", list.join(", "));
        }
        s += "}}\n\n.rgb_reps={";
        for (k, r) in self.rgb_reps.iter().enumerate() {
            let [red, green, blue] = r.nodes;
            let _ = write!(
                s,
                "\n    RgbEdge{{.red_node={}, .green_node={}, .blue_node={}, .obs_flip={}, .charge_flip={}}} // rep {k}",
                node(red),
                node(green),
                node(blue),
                r.obs,
                charge(r.charge)
            );
        }
        s += "\n}\n\n.drag_graph=DragGraph{.mmm={\n";
        for (&(n1, n2, c1, c2), &(flip, _)) in &self.drag.map {
            let _ = writeln!(s, "    {}@{n1}:{}@{n2} = {flip}", charge(c1), charge(c2));
        }
        // The Möbius model with coordinates: the declarations, the edges, and a last detector
        // declared when the model names fewer than two per detector.
        let mut text = String::new();
        // (Exactly, a negative zero's sign kept, so the model prints it as Stim does: "-0".)
        let exact = |c: f64| if c == 0.0 && c.is_sign_negative() { "-0".to_string() } else { crate::dem_program::fmt_exact(c) };
        for (coords, n) in &self.mobius_decls {
            let list: Vec<String> = coords.iter().map(|&c| exact(c)).collect();
            let _ = writeln!(text, "detector({}) D{n}", list.join(", "));
        }
        for line in self.mobius_text.lines().filter(|l| l.starts_with("error")) {
            text += line;
            text.push('\n');
        }
        let mut program = DemProgram::parse(&text)?;
        let n = 2 * self.colors.len() as u64;
        if (program.stats().num_detectors as u64) < n {
            program.instrs.push(DemInstr::Detector { coords: Vec::new(), targets: vec![n - 1], tag: String::new() });
        }
        let dem = program.to_stim();
        let _ = write!(s, "}}}}\n\n.mobius_dem=stim::DetectorErrorModel{{\n{}\n}}\n\n}}", dem.strip_suffix('\n').unwrap_or(&dem));
        Ok(s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rgb() -> Vec<ColorBasis> {
        (1..=3).map(|c| ColorBasis { color: c, basis: 1, ignored: false }).collect()
    }

    /// A triplet's three edges each join copies in one sub-graph: red-green in not-blue, green-
    /// blue in not-red, red-blue in not-green, as Chromobius's offsets place them.
    #[test]
    fn a_triplet_is_a_triangle_across_the_subgraphs() {
        let colors = rgb();
        let edges = mobius_edges(&key(0, 1, 2), &colors);
        assert_eq!(edges, vec![(1, 3), (2, 4), (0, 5)]);
        assert_eq!(mobius_edges(&key(1, BOUNDARY, BOUNDARY), &colors), vec![(2, 3)]);
        // Red and blue (charge G apart) swap copies; red and green do not.
        assert_eq!(mobius_edges(&key(0, 2, BOUNDARY), &colors), vec![(0, 5), (1, 4)]);
        assert_eq!(mobius_edges(&key(0, 1, BOUNDARY), &colors), vec![(0, 2), (1, 3)]);
    }

    /// A six-detector fault splits into two known triplets.
    #[test]
    fn a_composite_fault_splits_into_known_pieces() {
        let colors: Vec<ColorBasis> = (0..6).map(|d| ColorBasis { color: (d % 3 + 1) as u8, basis: 1, ignored: false }).collect();
        let atomic = BTreeMap::from([(key(0, 1, 2), 1u64), (key(3, 4, 5), 0)]);
        let (mut atoms, mut remnants) = (Vec::new(), BTreeMap::new());
        decompose(&[0, 1, 2, 3, 4, 5], 1, &colors, &atomic, &mut atoms, &mut remnants).unwrap();
        atoms.sort();
        assert_eq!(atoms, vec![key(0, 1, 2), key(3, 4, 5)]);
        assert!(remnants.is_empty());
        // Seven detectors of one basis cannot be decomposed.
        let colors7: Vec<ColorBasis> = (0..7).map(|d| ColorBasis { color: (d % 3 + 1) as u8, basis: 1, ignored: false }).collect();
        assert!(decompose(&[0, 1, 2, 3, 4, 5, 6], 0, &colors7, &atomic, &mut atoms, &mut remnants).is_err());
    }
}

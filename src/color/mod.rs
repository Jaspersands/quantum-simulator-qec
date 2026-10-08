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
//! is found by sparse blossom (`sparse`); the matched edges, with each event's two copies
//! joined, fall into cycles; and each cycle is lifted back to the code by carrying colour
//! charge around it, which says which observables the correction flips.
//!
//! This module is a port of Chromobius (quantumlib/chromobius, `src/chromobius`) in its order
//! of operations, so its predictions are Chromobius's but for the matcher's ties: the same
//! basic faults (`collect_atomic_errors`), the same decomposition of every other fault into
//! them (`collect_composite_errors`), the same Möbius model, and the same charge, drag and
//! representative tables (`graphs`) and lifting (`decode`). One step is not reproducible:
//! Chromobius walks the charge graph's neighbours in `std::unordered_map` order, which differs
//! between C++ libraries, when it looks for the shortest way to drag charge between two
//! places. Where two such ways flip different observables, the table entry is marked
//! ambiguous, and a shot whose lifting uses one is counted as a tie.

mod decode;
mod graphs;

use std::collections::BTreeMap;

use crate::dem::Dem;

pub use decode::{ColorDecoder, ColorWork};

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
    pub rgb_reps: Vec<graphs::RgbEdge>,
    pub drag: graphs::DragGraph,
}

/// Each detector's colour and basis from its 4th coordinate (`collect_nodes_from_dem`,
/// `detector_instruction_to_color_basis`).
fn collect_nodes(dem: &Dem) -> Result<Vec<ColorBasis>, String> {
    const MAPPING: [(Charge, u8); 6] = [(1, 1), (2, 1), (3, 1), (1, 2), (2, 2), (3, 2)];
    let mut colors = vec![ColorBasis::default(); dem.num_detectors];
    for (d, coords) in dem.detector_coords.iter().enumerate().take(dem.num_detectors) {
        if coords.is_empty() {
            continue;
        }
        let c = coords.get(3).copied().unwrap_or(-2.0);
        if !(-1.0..=5.0).contains(&c) || c.fract() != 0.0 || coords.len() < 4 {
            return Err(format!(
                "detector D{d}: expected at least 4 coordinates, the 4th identifying the basis and colour (RedX=0, GreenX=1, BlueX=2, RedZ=3, GreenZ=4, BlueZ=5, or -1 to ignore it), but got {coords:?}"
            ));
        }
        colors[d] = if c == -1.0 {
            ColorBasis { color: NEUTRAL, basis: 0, ignored: true }
        } else {
            let (color, basis) = MAPPING[c as usize];
            ColorBasis { color, basis, ignored: false }
        };
    }
    Ok(colors)
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

impl Lifted {
    /// `Decoder::from_dem`.
    pub fn from_dem(dem: &Dem, options: Options) -> Result<Lifted, String> {
        let colors = collect_nodes(dem)?;
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
                    let list = |v: &[u32]| v.iter().map(|d| format!("D{d}")).collect::<Vec<_>>().join(" ");
                    let mut msg = format!(
                        "fault {i} (detectors {}) cannot be decomposed into the colour code's basic faults: X detectors [{}] and Z detectors [{}] remain",
                        list(&m.detectors),
                        list(&x),
                        list(&z)
                    );
                    if obs != 0 {
                        msg += &format!(" with observables mask {obs:#x}");
                    }
                    msg += ". Likely causes: detectors with wrong colour or basis annotations, or faults too complex to decompose (more than 6 detectors of one basis), as Chromobius reports.";
                    return Err(msg);
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
        Ok(Lifted { colors, mobius_text: text, rgb_reps, drag })
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

//! Correlated matching: the rule table, the traced edges, the second pass, and
//! whether it helps. Spec: docs/superpowers/specs/2026-09-25-google-data-design.md, Part 1.

use crate::circuit::Basis;
use crate::dem::Dem;
use crate::dem_decoder::{edge_weight, DemDecoder};
use crate::frame_sampler::FrameSampler;
use crate::memory::{generate, CodeKind, NoiseModel};
use crate::surface_code::Xorshift;

fn decoder(text: &str) -> DemDecoder {
    DemDecoder::new(&Dem::parse(text).unwrap()).unwrap()
}

fn id(dec: &DemDecoder, u: u32, v: u32) -> u32 {
    dec.graph().edge_id(u, v).unwrap()
}

/// Edge c's rules, against (affected edge, probability), sorted by edge.
fn assert_rules(dec: &DemDecoder, c: u32, want: &[(u32, f64)]) {
    let got = dec.correlations().rules_of(c);
    assert_eq!(got.len(), want.len(), "edge {c}: {got:?}");
    for (&(a, p, w), &(wa, wp)) in got.iter().zip(want) {
        assert_eq!(a, wa, "edge {c}: {got:?}");
        assert!((p - wp).abs() < 1e-12, "edge {c} -> {a}: {p} against {wp}");
        assert_eq!(w, edge_weight(p).1, "edge {c} -> {a}: weight");
    }
}

fn x(a: f64, b: f64) -> f64 {
    a * (1.0 - b) + b * (1.0 - a)
}

#[test]
fn rules_follow_the_joint_probabilities() {
    let dec = decoder("error(0.1) D0 D1 ^ D2 D3\nerror(0.2) D0 D1\nerror(0.05) D2 D3 ^ D4\n");
    // Five detectors, so 5 is the boundary. Ids follow the sorted edges.
    let (e01, e23, e4) = (id(&dec, 0, 1), id(&dec, 2, 3), id(&dec, 4, 5));
    let m01 = x(0.1, 0.2);
    let m23 = x(0.1, 0.05);
    assert_rules(&dec, e01, &[(e23, 0.1 / m01)]);
    // 0.1 / 0.14 exceeds one half and is capped there.
    assert_rules(&dec, e23, &[(e01, 0.5), (e4, 0.05 / m23)]);
    assert_rules(&dec, e4, &[(e23, 0.5)]);
}

#[test]
fn every_pair_of_pieces_is_correlated() {
    let dec = decoder("error(0.1) D0 ^ D1 ^ D2\nerror(0.3) D0\n");
    let (e0, e1, e2) = (id(&dec, 0, 3), id(&dec, 1, 3), id(&dec, 2, 3));
    let m0 = x(0.1, 0.3);
    assert_rules(&dec, e0, &[(e1, 0.1 / m0), (e2, 0.1 / m0)]);
    assert_rules(&dec, e1, &[(e0, 0.5), (e2, 0.5)]);
    assert_rules(&dec, e2, &[(e0, 0.5), (e1, 0.5)]);
}

#[test]
fn errors_that_cannot_happen_add_no_rules() {
    let dec = decoder("error(0.1) D0 D1\nerror(0.1) D2 D3\nerror(0) D0 D1 ^ D2 D3\n");
    assert_eq!(dec.correlations().num_rules(), 0);
}

#[test]
fn a_piece_repeated_within_one_error_takes_p_twice_more() {
    // PyMatching's pair loop names the marginal twice over when both pieces
    // are one edge, so the marginal takes p four times from this error.
    let dec = decoder("error(0.1) D0 D1 ^ D0 D1 ^ D2 D3\nerror(0.4) D0 D1\n");
    let (e01, e23) = (id(&dec, 0, 1), id(&dec, 2, 3));
    let m01 = x(x(x(x(0.1, 0.1), 0.1), 0.1), 0.4);
    assert_rules(&dec, e01, &[(e23, x(0.1, 0.1) / m01)]);
}

#[test]
fn models_the_rules_cannot_use_are_refused() {
    // An undecomposed error on three detectors, and a piece with no detectors.
    assert!(DemDecoder::new(&Dem::parse("error(0.1) D0 D1 D2\n").unwrap()).is_err());
    assert!(DemDecoder::new(&Dem::parse("error(0.1) D0 ^ L0\n").unwrap()).is_err());
}

#[test]
fn circuit_models_have_rules() {
    let c = generate(CodeKind::Rotated, 3, 3, NoiseModel::Sd6 { p: 0.003 }, Basis::Z).unwrap();
    let dec = DemDecoder::new(&Dem::from_circuit(&c).unwrap()).unwrap();
    assert!(dec.correlations().num_rules() > 0);
}

fn defects_of(dets: &[bool]) -> Vec<u32> {
    dets.iter().enumerate().filter(|x| *x.1).map(|x| x.0 as u32).collect()
}

#[test]
fn traces_paths_through_empty_nodes_and_to_the_boundary() {
    let dec = decoder("error(0.1) D0 D1\nerror(0.1) D1 D2\nerror(0.1) D2 D3\nerror(0.01) D3\n");
    let mut e = dec.decode_to_edges(&[0, 2]).unwrap();
    e.sort();
    assert_eq!(e, vec![(0, 1), (1, 2)]);
    // One defect, and the only boundary edge is at D3 (4 is the boundary).
    let mut e = dec.decode_to_edges(&[1]).unwrap();
    e.sort();
    assert_eq!(e, vec![(1, 2), (2, 3), (3, 4)]);
    assert_eq!(dec.decode_to_edges(&[]).unwrap(), vec![]);
}

/// Layer 2. The traced edges have the shot's defects as their syndrome, and
/// weigh exactly the optimum: they are a minimum-weight correction. Their
/// observables are not compared with pass one's, since two equally short paths
/// may differ by a logical operator.
#[test]
fn traced_edges_are_a_minimum_weight_correction() {
    let mut rng = Xorshift::new(7);
    let mut shots = 0;
    for kind in [CodeKind::Rotated, CodeKind::Xzzx] {
        for d in [3usize, 5, 7] {
            for &p in &[0.003, 0.006] {
                let c = generate(kind, d, d, NoiseModel::Sd6 { p }, Basis::Z).unwrap();
                let dec = DemDecoder::new(&Dem::from_circuit(&c).unwrap()).unwrap();
                let sampler = FrameSampler::new(&c).unwrap();
                let g = dec.graph();
                let nd = g.num_nodes as u32;
                for _ in 0..300 {
                    let defects = defects_of(&sampler.sample(&mut rng).detectors);
                    let plain = dec.decode(&defects).unwrap();
                    let edges = dec.decode_to_edges(&defects).unwrap();
                    let mut ids: Vec<u32> = edges.iter().map(|&(u, v)| g.edge_id(u, v).unwrap()).collect();
                    ids.sort_unstable();
                    ids.dedup();
                    assert_eq!(ids.len(), edges.len(), "an edge listed twice");
                    let mut syndrome = vec![false; nd as usize];
                    for &(u, v) in &edges {
                        syndrome[u as usize] ^= true;
                        if v != nd {
                            syndrome[v as usize] ^= true;
                        }
                    }
                    let weight: i64 = ids.iter().map(|&e| g.weight_of(e)).sum();
                    assert_eq!(defects_of(&syndrome), defects, "{kind:?} d = {d}, p = {p}");
                    assert_eq!(weight, plain.iweight, "{kind:?} d = {d}, p = {p}: {defects:?}");
                    shots += 1;
                }
            }
        }
    }
    assert_eq!(shots, 3600);
}

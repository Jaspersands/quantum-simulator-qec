//! Correlated matching: the rule table, the traced edges, the second pass, and
//! whether it helps. Spec: docs/superpowers/specs/2026-09-25-google-data-design.md, Part 1.

use crate::circuit::Basis;
use crate::dem::Dem;
use super::*;
use crate::dem_decoder::{edge_weight, DecodeError, DemDecoder};
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
    // An undecomposed error on three detectors.
    assert!(DemDecoder::new(&Dem::parse("error(0.1) D0 D1 D2\n").unwrap()).is_err());
}

#[test]
fn a_piece_of_observables_alone_names_no_edge() {
    // Stim writes such a piece when a fault is split through pieces whose
    // observables fall short of its own (a weight-two logical). It fires
    // nothing, so plain matching drops it, as PyMatching's plain mode does, and
    // so do the rules; PyMatching's correlated mode refuses the model instead.
    let dec = decoder("error(0.1) D0 ^ D1 ^ L0\nerror(0.2) D0\nerror(0.2) D1\n");
    assert_eq!(dec.graph().num_edges(), 2);
    let (e0, e1) = (dec.graph().edge_id(0, 2).unwrap(), dec.graph().edge_id(1, 2).unwrap());
    assert_eq!(dec.correlations().rules_of(e0).iter().map(|r| r.0).collect::<Vec<_>>(), vec![e1]);
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

/// A model whose merged edges carry exactly the weights pass two matches on:
/// each edge's own probability, raised to the largest a rule of a used edge
/// implies. The weight falls as the probability rises, so the larger
/// probability gives the smaller weight, as `reweight` keeps.
fn reweighted_model(dem: &Dem, dec: &DemDecoder, used: &[(u32, u32)]) -> String {
    let (edges, _) = crate::dem_decoder::merged_edges(dem).unwrap();
    let g = dec.graph();
    let nd = g.num_nodes as u32;
    let mut prob: Vec<f64> = edges.iter().map(|e| e.2).collect();
    for &(u, v) in used {
        for (a, pa, _) in dec.correlations().rules_of(g.edge_id(u, v).unwrap()) {
            prob[a as usize] = prob[a as usize].max(pa);
        }
    }
    let mut text = format!("detector D{}\n", nd - 1);
    for (&(u, v, _, o), &q) in edges.iter().zip(&prob) {
        text.push_str(&format!("error({q:?}) D{u}"));
        if v != nd {
            text.push_str(&format!(" D{v}"));
        }
        for k in 0..64 {
            if (o >> k) & 1 == 1 {
                text.push_str(&format!(" L{k}"));
            }
        }
        text.push('\n');
    }
    text
}

/// Layer 3. Pass two is exact: on the reweighted graph its integer weight is
/// the dense matcher's.
#[test]
fn pass_two_is_exact_on_the_reweighted_graph() {
    let mut rng = Xorshift::new(41);
    let mut compared = 0;
    for kind in [CodeKind::Rotated, CodeKind::Xzzx] {
        for d in [3usize, 5, 7] {
            for &p in &[0.003, 0.006] {
                let c = generate(kind, d, d, NoiseModel::Sd6 { p }, Basis::Z).unwrap();
                let dem = Dem::from_circuit(&c).unwrap();
                let dec = DemDecoder::new(&dem).unwrap();
                let sampler = FrameSampler::new(&c).unwrap();
                for _ in 0..120 {
                    let defects = defects_of(&sampler.sample(&mut rng).detectors);
                    let used = dec.decode_to_edges(&defects).unwrap();
                    let oracle = DemDecoder::new(&Dem::parse(&reweighted_model(&dem, &dec, &used)).unwrap()).unwrap();
                    let dense = match oracle.decode_dense(&defects) {
                        Ok(x) => x,
                        Err(DecodeError::TooManyDefects(_)) => continue,
                        Err(e) => panic!("{e:?}"),
                    };
                    let two = dec.decode_correlated(&defects).unwrap();
                    assert_eq!(two.iweight, dense.iweight, "{kind:?} d = {d}, p = {p}: {defects:?}");
                    compared += 1;
                }
            }
        }
    }
    println!("pass two equal to the dense oracle on {compared} shots");
    assert!(compared > 1400, "{compared}");
}

#[test]
fn pass_two_from_given_edges_is_the_full_decode() {
    let c = generate(CodeKind::Rotated, 5, 5, NoiseModel::Sd6 { p: 0.006 }, Basis::Z).unwrap();
    let dec = DemDecoder::new(&Dem::from_circuit(&c).unwrap()).unwrap();
    let sampler = FrameSampler::new(&c).unwrap();
    let mut rng = Xorshift::new(3);
    for _ in 0..300 {
        let defects = defects_of(&sampler.sample(&mut rng).detectors);
        let used = dec.decode_to_edges(&defects).unwrap();
        assert_eq!(dec.decode_pass2(&defects, &used).unwrap(), dec.decode_correlated(&defects).unwrap());
    }
    assert!(dec.decode_pass2(&[0, 1], &[(0, 999_999)]).is_err());
}

#[test]
fn weights_are_restored_after_every_decode() {
    let c = generate(CodeKind::Xzzx, 5, 5, NoiseModel::Sd6 { p: 0.006 }, Basis::Z).unwrap();
    let dec = DemDecoder::new(&Dem::from_circuit(&c).unwrap()).unwrap();
    let sampler = FrameSampler::new(&c).unwrap();
    let (g, corr) = (dec.graph(), dec.correlations());
    let mut rng = Xorshift::new(11);
    let mut scratch = Scratch::new(g);
    for _ in 0..300 {
        let defects = defects_of(&sampler.sample(&mut rng).detectors);
        let two = g.decode_correlated(corr, &mut scratch, &defects).unwrap();
        assert_eq!(scratch.scan, g.scan);
        assert!(scratch.undo.is_empty());
        assert_eq!(g.decode(&mut scratch, &defects).unwrap(), dec.decode(&defects).unwrap());
        assert_eq!(two, g.decode_correlated(corr, &mut Scratch::new(g), &defects).unwrap());
    }
}

/// Layer 5, in miniature: under SD6 a Y error lights both halves of the
/// graph, and correlated matching should fail less often than plain matching.
/// The cross-check measures the gain properly (check 5).
#[test]
fn correlated_matching_beats_plain_matching_under_sd6() {
    let c = generate(CodeKind::Rotated, 5, 5, NoiseModel::Sd6 { p: 0.006 }, Basis::Z).unwrap();
    let dec = DemDecoder::new(&Dem::from_circuit(&c).unwrap()).unwrap();
    let sampler = FrameSampler::new(&c).unwrap();
    let mut rng = Xorshift::new(2026);
    let (mut plain, mut corr) = (0, 0);
    for _ in 0..5000 {
        let shot = sampler.sample(&mut rng);
        let defects = defects_of(&shot.detectors);
        plain += ((dec.decode(&defects).unwrap().observables ^ shot.observables) & 1) as usize;
        corr += ((dec.decode_correlated(&defects).unwrap().observables ^ shot.observables) & 1) as usize;
    }
    println!("d = 5, p = 0.6%, 5000 shots: plain {plain} failures, correlated {corr}");
    assert!(corr < plain, "correlated {corr}, plain {plain}");
}

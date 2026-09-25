use super::*;
use crate::dem::Dem;
use crate::dem_decoder::{DecodeError, DemDecoder};
use crate::surface_code::Xorshift;
use std::collections::HashSet;

type TestEdge = (u32, Option<u32>, f64, u64);

fn decoder(text: &str) -> DemDecoder {
    DemDecoder::new(&Dem::parse(text).unwrap()).unwrap()
}

/// Exhaustive minimum over edge subsets with the given boundary, in the same
/// integer weights both matchers use.
fn brute_force(edges: &[TestEdge], defects: &[u32]) -> Option<(i64, Vec<u64>)> {
    let target: u64 = defects.iter().fold(0, |acc, &d| acc ^ (1 << d));
    let mut best: Option<i64> = None;
    let mut masks = Vec::new();
    for mask in 0u32..(1 << edges.len()) {
        let (mut syn, mut w, mut obs) = (0u64, 0i64, 0u64);
        for (i, e) in edges.iter().enumerate() {
            if (mask >> i) & 1 == 1 {
                syn ^= 1 << e.0;
                if let Some(b) = e.1 {
                    syn ^= 1 << b;
                }
                w += crate::dem_decoder::edge_weight(e.2).1;
                obs ^= e.3;
            }
        }
        if syn != target {
            continue;
        }
        match best {
            Some(b) if w > b => {}
            Some(b) if w == b => masks.push(obs),
            _ => {
                best = Some(w);
                masks = vec![obs];
            }
        }
    }
    best.map(|b| (b, masks))
}

fn random_small(rng: &mut Xorshift) -> (usize, Vec<TestEdge>, String) {
    let nd = 3 + (rng.next_u64() % 6) as usize;
    let mut edges: Vec<TestEdge> = Vec::new();
    let mut seen = HashSet::new();
    let target = (2 * nd).min(14);
    let mut tries = 0;
    while edges.len() < target && tries < 1000 {
        tries += 1;
        let a = (rng.next_u64() % nd as u64) as u32;
        let b = if rng.next_u64() % 3 == 0 { None } else { Some((rng.next_u64() % nd as u64) as u32) };
        if b == Some(a) {
            continue;
        }
        let key = match b {
            Some(b) => (a.min(b), Some(a.max(b))),
            None => (a, None),
        };
        if !seen.insert(key) {
            continue;
        }
        edges.push((key.0, key.1, 0.01 + 0.3 * rng.next_f64(), rng.next_u64() % 2));
    }
    let mut text = String::new();
    for d in 0..nd {
        text.push_str(&format!("detector D{d}\n"));
    }
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
    (nd, edges, text)
}

#[test]
fn matches_brute_force_on_small_graphs() {
    let mut rng = Xorshift::new(17);
    let (mut compared, mut unmatchable) = (0, 0);
    for trial in 0..3000 {
        let (nd, edges, text) = random_small(&mut rng);
        let dec = decoder(&text);
        let defects: Vec<u32> = (0..nd as u32).filter(|_| rng.next_u64() % 2 == 0).collect();
        let mut scratch = Scratch::new(dec.graph());
        match (brute_force(&edges, &defects), dec.graph().decode_checked(&mut scratch, &defects)) {
            (None, Err(DecodeError::Unmatchable)) => unmatchable += 1,
            (Some((w, masks)), Ok(pred)) => {
                assert_eq!(pred.iweight, w, "trial {trial}");
                assert!(masks.contains(&pred.observables), "trial {trial}: {} not in {masks:?}", pred.observables);
                compared += usize::from(!defects.is_empty());
            }
            (b, s) => panic!("trial {trial}: brute force {b:?}, sparse {s:?}\n{text}defects {defects:?}"),
        }
    }
    assert!(compared > 1500 && unmatchable > 25, "{compared} compared, {unmatchable} unmatchable");
}

fn random_graph(rng: &mut Xorshift, nodes: usize, boundary: f64) -> String {
    let mut seen = HashSet::new();
    let mut text = String::new();
    for d in 0..nodes {
        text.push_str(&format!("detector D{d}\n"));
    }
    let obs = |rng: &mut Xorshift| if rng.next_u64() % 4 == 0 { " L0" } else { "" };
    for u in 0..nodes {
        for _ in 0..2 {
            let v = (rng.next_u64() % nodes as u64) as usize;
            if v == u || !seen.insert((u.min(v), u.max(v))) {
                continue;
            }
            let p = 0.001 + 0.3 * rng.next_f64();
            let o = obs(rng);
            text.push_str(&format!("error({p}) D{u} D{v}{o}\n"));
        }
        if rng.next_f64() < boundary {
            let p = 0.001 + 0.3 * rng.next_f64();
            let o = obs(rng);
            text.push_str(&format!("error({p}) D{u}{o}\n"));
        }
    }
    text
}

#[test]
fn agrees_with_the_dense_matcher_on_random_graphs() {
    let mut rng = Xorshift::new(2024);
    let (mut compared, mut unmatchable) = (0, 0);
    for trial in 0..20_000 {
        let nodes = 4 + (rng.next_u64() % 40) as usize;
        let boundary = [0.0, 0.1, 0.5][(rng.next_u64() % 3) as usize];
        let text = random_graph(&mut rng, nodes, boundary);
        let dec = decoder(&text);
        let defects: Vec<u32> = (0..nodes as u32).filter(|_| rng.next_u64() % 3 == 0).collect();
        let mut scratch = Scratch::new(dec.graph());
        match (dec.graph().decode_checked(&mut scratch, &defects), dec.decode_dense(&defects)) {
            (Ok(s), Ok(d)) => {
                assert_eq!(s.iweight, d.iweight, "trial {trial}\n{text}defects {defects:?}");
                compared += 1;
            }
            (Err(DecodeError::Unmatchable), Err(DecodeError::Unmatchable)) => unmatchable += 1,
            (s, d) => panic!("trial {trial}: sparse {s:?}, dense {d:?}\n{text}defects {defects:?}"),
        }
    }
    assert!(compared > 10_000 && unmatchable > 100, "{compared} compared, {unmatchable} unmatchable");
}

#[test]
fn one_scratch_decodes_many_shots() {
    let mut rng = Xorshift::new(5);
    let text = random_graph(&mut rng, 30, 0.5);
    let dec = decoder(&text);
    let mut scratch = Scratch::new(dec.graph());
    for _ in 0..500 {
        let defects: Vec<u32> = (0..30).filter(|_| rng.next_u64() % 3 == 0).collect();
        let a = dec.graph().decode(&mut scratch, &defects).map(|p| p.iweight);
        let b = dec.graph().decode(&mut Scratch::new(dec.graph()), &defects).map(|p| p.iweight);
        assert_eq!(a, b);
    }
}

#[test]
fn agrees_with_the_dense_matcher_on_surface_code_shots() {
    use crate::circuit::Basis;
    use crate::frame_sampler::FrameSampler;
    use crate::memory::{generate, CodeKind, NoiseModel};
    let mut rng = Xorshift::new(99);
    let (mut compared, mut obs_differ) = (0, 0);
    for kind in [CodeKind::Rotated, CodeKind::Xzzx] {
        for basis in [Basis::Z, Basis::X] {
            for d in [3usize, 5, 7] {
                for &p in &[0.002, 0.006, 0.012] {
                    for noise in [NoiseModel::Sd6 { p }, NoiseModel::Current { p, eta: 0.5 }] {
                        let c = generate(kind, d, d, noise, basis).unwrap();
                        let dec = DemDecoder::new(&Dem::from_circuit(&c).unwrap()).unwrap();
                        let sampler = FrameSampler::new(&c).unwrap();
                        let mut scratch = Scratch::new(dec.graph());
                        for shot_i in 0..60 {
                            let shot = sampler.sample(&mut rng);
                            let defects: Vec<u32> =
                                shot.detectors.iter().enumerate().filter(|x| *x.1).map(|x| x.0 as u32).collect();
                            let dense = match dec.decode_dense(&defects) {
                                Ok(x) => x,
                                Err(DecodeError::TooManyDefects(_)) => continue,
                                Err(e) => panic!("{e:?}"),
                            };
                            let sparse = if shot_i < 5 && d <= 5 {
                                dec.graph().decode_checked(&mut scratch, &defects)
                            } else {
                                dec.graph().decode(&mut scratch, &defects)
                            }
                            .unwrap();
                            assert_eq!(sparse.iweight, dense.iweight, "{kind:?} {basis:?} {noise:?} d = {d}: {defects:?}");
                            compared += 1;
                            obs_differ += usize::from(sparse.observables != dense.observables);
                        }
                    }
                }
            }
        }
    }
    println!("{compared} shots, equal weight on all; {obs_differ} tie-broken differently");
    assert!(compared > 3000, "{compared}");
}

#[test]
fn decodes_shots_far_beyond_the_dense_limit() {
    use crate::circuit::Basis;
    use crate::frame_sampler::FrameSampler;
    use crate::memory::{generate, CodeKind, NoiseModel};
    let c = generate(CodeKind::Rotated, 7, 60, NoiseModel::Sd6 { p: 0.006 }, Basis::Z).unwrap();
    let dec = DemDecoder::new(&Dem::from_circuit(&c).unwrap()).unwrap();
    let sampler = FrameSampler::new(&c).unwrap();
    let mut rng = Xorshift::new(7);
    let (mut most, mut failures) = (0, 0);
    for _ in 0..100 {
        let shot = sampler.sample(&mut rng);
        most = most.max(shot.detectors.iter().filter(|&&b| b).count());
        let pred = dec.decode_bools(&shot.detectors).expect("the sparse matcher has no defect ceiling");
        failures += ((pred.observables ^ shot.observables) & 1) as usize;
    }
    assert!(most > 256, "the test should exceed the dense limit; most was {most}");
    assert!(failures < 100, "every shot failed");
}

#[test]
#[ignore] // timing, for the README and the site
fn timing() {
    use crate::circuit::Basis;
    use crate::frame_sampler::FrameSampler;
    use crate::memory::{generate, CodeKind, NoiseModel};
    for d in [3usize, 5, 7, 9] {
        for &p in &[0.003, 0.006] {
            let c = generate(CodeKind::Rotated, d, d, NoiseModel::Sd6 { p }, Basis::Z).unwrap();
            let dec = DemDecoder::new(&Dem::from_circuit(&c).unwrap()).unwrap();
            let sampler = FrameSampler::new(&c).unwrap();
            let mut rng = Xorshift::new(1);
            let shots: Vec<Vec<u32>> = (0..2000)
                .map(|_| sampler.sample(&mut rng).detectors.iter().enumerate().filter(|x| *x.1).map(|x| x.0 as u32).collect())
                .collect();
            let mut scratch = Scratch::new(dec.graph());
            let t = std::time::Instant::now();
            for s in &shots {
                dec.graph().decode(&mut scratch, s).unwrap();
            }
            let sparse_us = t.elapsed().as_secs_f64() * 1e6 / shots.len() as f64;
            let corr = dec.correlations();
            let t = std::time::Instant::now();
            for s in &shots {
                dec.graph().decode_correlated(corr, &mut scratch, s).unwrap();
            }
            let correlated_us = t.elapsed().as_secs_f64() * 1e6 / shots.len() as f64;
            let t = std::time::Instant::now();
            let mut dense_n = 0;
            for s in shots.iter().take(300) {
                if dec.decode_dense(s).is_ok() {
                    dense_n += 1;
                }
            }
            let dense_us = t.elapsed().as_secs_f64() * 1e6 / dense_n.max(1) as f64;
            println!("d = {d}, p = {p}: sparse {sparse_us:.1} us/shot, correlated {correlated_us:.1} us/shot, dense {dense_us:.1} us/shot");
        }
    }
}

#[test]
fn edge_ids_name_both_halves() {
    use super::state::NONE;
    let dec = decoder("error(0.1) D0 D1 L0\nerror(0.2) D1 D2\nerror(0.3) D2\n");
    let g = dec.graph();
    assert_eq!(g.num_edges(), 3);
    let e01 = g.edge_id(0, 1).unwrap();
    assert_eq!(g.edge_id(1, 0), Some(e01));
    assert_eq!(g.edge_ends(e01), (0, 1));
    // 3 == num_nodes stands for the boundary, in either place.
    let eb = g.edge_id(2, 3).unwrap();
    assert_eq!(g.edge_id(3, 2), Some(eb));
    assert_eq!(g.edge_ends(eb), (2, 3));
    assert_eq!(g.edge_id(0, 2), None);
    assert_eq!(g.edge_id(7, 0), None);
    for id in 0..3u32 {
        let [a, b] = g.halves[id as usize];
        assert_eq!(g.edge_of[a as usize], id);
        if b != NONE {
            assert_eq!(g.edge_of[b as usize], id);
            assert_eq!(g.w[a as usize], g.w[b as usize]);
        }
    }
    assert_eq!(g.halves[eb as usize][1], NONE);
    assert_eq!(g.weight_of(e01), crate::dem_decoder::edge_weight(0.1).1);
    let scratch = Scratch::new(g);
    assert_eq!(scratch.w, g.w);
}

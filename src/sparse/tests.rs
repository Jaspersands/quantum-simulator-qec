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
                w += crate::dem_decoder::int_weight(((1.0 - e.2) / e.2).ln());
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

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
                            dets.push(
                                *index
                                    .get(&key)
                                    .unwrap_or_else(|| panic!("no detector at {key:?}")),
                            );
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
        let e = model
            .entry((m.detectors.clone(), m.observables))
            .or_insert(0.0);
        *e = xor_prob(*e, m.p);
    }
    model
}

fn check(kind: CodeKind, d: usize, basis: Basis) {
    let (p, eta) = (0.003, 0.5);
    let patch = patch_for(kind, d, basis).unwrap();
    let dem =
        Dem::from_circuit(&memory_circuit(&patch, d, NoiseModel::Current { p, eta })).unwrap();
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
    let missing: Vec<_> = old
        .keys()
        .filter(|k| !new.contains_key(*k))
        .take(3)
        .collect();
    let extra: Vec<_> = new
        .keys()
        .filter(|k| !old.contains_key(*k))
        .take(3)
        .collect();
    assert!(
        missing.is_empty() && extra.is_empty(),
        "{kind:?} {basis:?} d = {d}: old-only {missing:?}, new-only {extra:?}"
    );
    for (k, &po) in &old {
        let pn = new[k];
        assert!(
            (po - pn).abs() <= 1e-9 * po.max(pn),
            "{kind:?} {basis:?} d = {d}: {k:?} old {po} new {pn}"
        );
    }
    println!(
        "{kind:?} {basis:?} d = {d}: {} mechanisms identical",
        old.len()
    );
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
                let class =
                    code.simulate_circuit_noise_with_model(&model, d, p, 0.5, "zero", 2, 0.0, 0);
                old_fail += (class & 1) as usize;
            }
            let circuit = crate::memory::generate(
                CodeKind::Rotated,
                d,
                d,
                NoiseModel::Current { p, eta: 0.5 },
                Basis::Z,
            )
            .unwrap();
            let dec = DemDecoder::new(&Dem::from_circuit(&circuit).unwrap()).unwrap();
            let sampler = FrameSampler::new(&circuit).unwrap();
            let mut rng = Xorshift::new(0x5eed ^ (d as u64) ^ p.to_bits());
            let (mut new_fail, mut errors) = (0, 0);
            for _ in 0..shots {
                let shot = sampler.sample(&mut rng);
                match dec.decode_bools(&shot.detectors) {
                    Ok(pred) => new_fail += ((pred.observables ^ shot.observables) & 1) as usize,
                    Err(_) => {
                        errors += 1;
                        new_fail += 1;
                    }
                }
            }
            println!(
                "d = {d}, p = {:.1}%: old (unit-weight MWPM) {:.3}%, new (weighted MWPM) {:.3}%, {errors} decode errors, {shots} shots each",
                p * 100.0,
                old_fail as f64 * 100.0 / shots as f64,
                new_fail as f64 * 100.0 / shots as f64
            );
        }
    }
}

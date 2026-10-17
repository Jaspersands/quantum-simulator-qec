//! A perturbative expansion of a circuit's outcome distribution, as QuantumClifford.jl's
//! `petrajectories`: every noise channel is a site that applies none of its faults or one of
//! them, each with its probability, and the distribution of detection events and observable
//! flips is summed over the sets of at most `order` faults. A fault's effect is found once, by
//! carrying a Pauli frame for every fault of the circuit at once (a bit per fault on each
//! qubit) through the Clifford steps; the sum is a dynamic program over outcomes.
//!
//! Each outcome's weight is kept per number of faults, either as a number (each fault set's
//! exact probability, Π p_fault Π (1 − P_site)) or as a polynomial in a noise scale x, every
//! probability p read as p·x: its coefficients up to x^order are the exact Taylor coefficients
//! of the outcome's probability, as a fault set of k faults is O(x^k).
//!
//! A fault counts toward the order whether or not it changes an outcome (as QuantumClifford's
//! count does); faults of one site with the same effect are merged.

use std::collections::HashMap;

use crate::circuit::{Basis, Circuit};
use crate::statevec::{Program, Step};

/// A site: each of its faults' probability and effect (sorted detectors, then `num_detectors`
/// plus each observable).
#[derive(Clone, Debug)]
pub struct Site {
    pub faults: Vec<(f64, Vec<u32>)>,
}

/// The circuit's sites, after one frame pass; with its detector and observable counts.
pub struct Sites {
    pub sites: Vec<Site>,
    pub num_detectors: usize,
    pub num_observables: usize,
}

/// A bitset over fault columns.
type Bits = Vec<u64>;

fn xor_into(a: &mut Bits, b: &Bits) {
    for (x, y) in a.iter_mut().zip(b) {
        *x ^= *y;
    }
}

fn flip(a: &mut Bits, c: usize) {
    a[c / 64] ^= 1 << (c % 64);
}

/// What a step injects: (probability, Pauli product, whether the record it writes flips).
fn faults_of(step: &Step) -> Option<Vec<(f64, Vec<(usize, u8)>, bool)>> {
    match step {
        Step::Channel(branches) => Some(branches.iter().map(|(p, ps)| (*p, ps.clone(), false)).collect()),
        Step::Measure { flip, .. } | Step::Pad { flip, .. } if *flip > 0.0 => Some(vec![(*flip, Vec::new(), true)]),
        Step::Herald { q, probs } => Some((0..4).filter(|&k| probs[k] > 0.0).map(|k| (probs[k], if k == 0 { Vec::new() } else { vec![(*q, [0u8, 1, 3, 2][k])] }, true)).collect()),
        _ => None,
    }
}

/// The circuit's fault sites and their effects. The circuit must be Clifford with Pauli noise,
/// its detectors and observables deterministic.
pub fn sites(circuit: &Circuit) -> Result<Sites, String> {
    // Observables as records, any number of them.
    let program = Program::with_layout_for(circuit, false, usize::MAX, crate::statevec::Outcomes::Records)?;
    // Each chain of correlated errors one site, as the density matrix groups them.
    let steps = crate::density::grouped(&program.steps, "the perturbative expansion")?;
    for step in &steps {
        if matches!(step, Step::Rotation { .. } | Step::Unitary(..) | Step::Damp { .. } | Step::Dephase { .. } | Step::Leak { .. } | Step::Seep { .. } | Step::Transport { .. }) {
            return Err("the perturbative expansion takes Clifford circuits with Pauli noise (no rotations, T, U3, damping or leakage)".into());
        }
    }
    // Detectors and observables must be deterministic: the converter's backward walk checks,
    // noise aside (so any channel Stim samples is taken).
    crate::dem_build::sweep_effects(circuit, true)?;
    let width: usize = steps.iter().filter_map(faults_of).map(|f| f.len()).sum();
    let words = width.div_ceil(64).max(1);
    let n = program.num_qubits();
    let (mut x, mut z) = (vec![vec![0u64; words]; n], vec![vec![0u64; words]; n]);
    let mut rec: Vec<Bits> = Vec::with_capacity(program.num_measurements);
    // Each site's faults: (column, probability).
    let mut columns: Vec<Vec<(usize, f64)>> = Vec::new();
    let mut next = 0usize;
    let apply = |x: &mut Vec<Bits>, z: &mut Vec<Bits>, paulis: &[(usize, u8)], c: usize| {
        for &(q, p) in paulis {
            if p & 1 != 0 {
                flip(&mut x[q], c);
            }
            if p & 2 != 0 {
                flip(&mut z[q], c);
            }
        }
    };
    for step in &steps {
        // A step's faults happen with it: a channel's after nothing, a record's flips on the
        // record it writes.
        let faults = faults_of(step);
        let first = next;
        if let Some(f) = &faults {
            columns.push(f.iter().enumerate().map(|(k, (p, ..))| (first + k, *p)).collect());
            next += f.len();
        }
        match step {
            Step::H(q) => std::mem::swap(&mut x[*q], &mut z[*q]),
            Step::S(q) => {
                let xq = x[*q].clone();
                xor_into(&mut z[*q], &xq);
            }
            Step::Cx(c, t) => {
                let xc = x[*c].clone();
                xor_into(&mut x[*t], &xc);
                let zt = z[*t].clone();
                xor_into(&mut z[*c], &zt);
            }
            Step::Cz(a, b) => {
                let (xa, xb) = (x[*a].clone(), x[*b].clone());
                xor_into(&mut z[*a], &xb);
                xor_into(&mut z[*b], &xa);
            }
            Step::Paulis(_) => {}
            Step::Channel(branches) => {
                for (k, (_, paulis)) in branches.iter().enumerate() {
                    apply(&mut x, &mut z, paulis, first + k);
                }
            }
            Step::Measure { q, basis, reset, .. } => {
                let (seen, other) = match basis {
                    Basis::Z => (&mut x, &mut z),
                    Basis::X => (&mut z, &mut x),
                };
                let mut r = seen[*q].clone();
                if faults.is_some() {
                    flip(&mut r, first);
                }
                rec.push(r);
                // The measured basis's other frame acts trivially on the eigenstate left.
                other[*q].iter_mut().for_each(|w| *w = 0);
                if *reset {
                    seen[*q].iter_mut().for_each(|w| *w = 0);
                }
            }
            Step::Reset { q, .. } => {
                x[*q].iter_mut().for_each(|w| *w = 0);
                z[*q].iter_mut().for_each(|w| *w = 0);
            }
            Step::Pad { .. } => {
                let mut r = vec![0u64; words];
                if faults.is_some() {
                    flip(&mut r, first);
                }
                rec.push(r);
            }
            Step::Herald { .. } => {
                let mut r = vec![0u64; words];
                for (k, (_, paulis, _)) in faults.as_ref().unwrap().iter().enumerate() {
                    flip(&mut r, first + k);
                    apply(&mut x, &mut z, paulis, first + k);
                }
                rec.push(r);
            }
            Step::Feedback { pauli, rec: r, q } => {
                let bits = rec[*r].clone();
                if pauli & 1 != 0 {
                    xor_into(&mut x[*q], &bits);
                }
                if pauli & 2 != 0 {
                    xor_into(&mut z[*q], &bits);
                }
            }
            Step::Correlated { .. } => unreachable!("grouped"),
            Step::Rotation { .. } | Step::Unitary(..) | Step::Damp { .. } | Step::Dephase { .. } | Step::Leak { .. } | Step::Seep { .. } | Step::Transport { .. } => unreachable!("refused above"),
        }
    }
    // Each column's effect: the detectors and observables whose parity it flips.
    let (nd, no) = (program.num_detectors(), program.num_observables());
    let mut effect: Vec<Vec<u32>> = vec![Vec::new(); width];
    for (t, recs) in program.detector_records().iter().chain(program.observable_records()).enumerate() {
        let mut bits = vec![0u64; words];
        for &r in recs {
            xor_into(&mut bits, &rec[r]);
        }
        for (w, &word) in bits.iter().enumerate() {
            let mut word = word;
            while word != 0 {
                let b = word.trailing_zeros() as usize;
                word &= word - 1;
                effect[w * 64 + b].push(t as u32);
            }
        }
    }
    let sites = columns
        .into_iter()
        .map(|cols| {
            let mut faults: Vec<(f64, Vec<u32>)> = Vec::new();
            for (c, p) in cols {
                if p == 0.0 {
                    continue;
                }
                match faults.iter_mut().find(|f| f.1 == effect[c]) {
                    Some(f) => f.0 += p,
                    None => faults.push((p, effect[c].clone())),
                }
            }
            Site { faults }
        })
        .filter(|s| !s.faults.is_empty())
        .collect();
    Ok(Sites { sites, num_detectors: nd, num_observables: no })
}

/// The sorted symmetric difference.
fn sym_diff(a: &[u32], b: &[u32]) -> Vec<u32> {
    let mut out = Vec::with_capacity(a.len() + b.len());
    let (mut i, mut j) = (0, 0);
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Less => {
                out.push(a[i]);
                i += 1;
            }
            std::cmp::Ordering::Greater => {
                out.push(b[j]);
                j += 1;
            }
            std::cmp::Ordering::Equal => {
                i += 1;
                j += 1;
            }
        }
    }
    out.extend_from_slice(&a[i..]);
    out.extend_from_slice(&b[j..]);
    out
}

/// The expansion: each outcome (its effect, as `Site`'s) with its weight, a number (`degree`
/// `None`) or a polynomial's coefficients up to x^degree. At most `max_terms` outcome and
/// fault-count pairs at once.
pub fn expand(sites: &[Site], order: usize, degree: Option<usize>, max_terms: usize) -> Result<Vec<(Vec<u32>, Vec<f64>)>, String> {
    // No weight has a higher degree than the number of sites.
    let degree = degree.map(|d| d.min(sites.len()));
    let len = degree.map_or(1, |d| d + 1);
    // (effect, number of faults) → weight.
    let mut state: HashMap<(Vec<u32>, usize), Vec<f64>> = HashMap::new();
    let mut one = vec![0.0; len];
    one[0] = 1.0;
    state.insert((Vec::new(), 0), one);
    for site in sites {
        let total: f64 = site.faults.iter().map(|f| f.0).sum();
        let mut next: HashMap<(Vec<u32>, usize), Vec<f64>> = HashMap::with_capacity(state.len() * 2);
        for ((key, j), w) in state {
            if j < order {
                for (p, effect) in &site.faults {
                    let entry = next.entry((sym_diff(&key, effect), j + 1)).or_insert_with(|| vec![0.0; len]);
                    if degree.is_none() {
                        entry[0] += w[0] * p;
                    } else {
                        // × p x.
                        for k in (0..len - 1).rev() {
                            entry[k + 1] += w[k] * p;
                        }
                    }
                }
            }
            let entry = next.entry((key, j)).or_insert_with(|| vec![0.0; len]);
            if degree.is_none() {
                entry[0] += w[0] * (1.0 - total);
            } else {
                // × (1 − P x).
                for k in 0..len {
                    entry[k] += w[k] - if k > 0 { w[k - 1] * total } else { 0.0 };
                }
            }
            if next.len() > max_terms {
                return Err(format!("more than {max_terms} outcome terms; lower the order"));
            }
        }
        state = next;
    }
    let mut out: HashMap<Vec<u32>, Vec<f64>> = HashMap::new();
    for ((key, _), w) in state {
        let e = out.entry(key).or_insert_with(|| vec![0.0; len]);
        for (a, b) in e.iter_mut().zip(&w) {
            *a += b;
        }
    }
    let mut out: Vec<(Vec<u32>, Vec<f64>)> = out.into_iter().collect();
    out.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dist(text: &str, order: usize) -> HashMap<Vec<u32>, f64> {
        let c = Circuit::parse(text).unwrap();
        let s = sites(&c).unwrap();
        expand(&s.sites, order, None, 1 << 20).unwrap().into_iter().map(|(k, w)| (k, w[0])).collect()
    }

    fn check_exact(text: &str, nd: u32) {
        let pe = dist(text, 40);
        let exact = crate::statevec::Program::new(&Circuit::parse(text).unwrap()).unwrap().distribution(1 << 20).unwrap();
        assert_eq!(pe.values().filter(|&&p| p > 1e-15).count(), exact.len());
        for ((dets, obs), p) in exact {
            let mut key: Vec<u32> = dets.iter().enumerate().filter(|(_, &b)| b).map(|(i, _)| i as u32).collect();
            key.extend((0..64).filter(|k| obs >> k & 1 == 1).map(|k| nd + k));
            assert!((pe[&key] - p).abs() < 1e-12, "{key:?}: {} vs {p}", pe[&key]);
        }
    }

    #[test]
    fn full_order_is_exact() {
        // A repetition code round: each data error flips its two checks.
        check_exact("R 0 1 2 3 4\nX_ERROR(0.1) 0 1 2\nCX 0 3 1 3 1 4 2 4\nM(0.05) 3 4\nDEPOLARIZE1(0.2) 0\nM 0 1 2\nDETECTOR rec[-5]\nDETECTOR rec[-4]\nOBSERVABLE_INCLUDE(0) rec[-1]", 2);
        // Every kind of site: two-qubit channels, heralds, a correlated chain, feedback, MPP.
        check_exact(
            "R 0 1 2 3 4\nX_ERROR(0.03) 0 1 2\nPAULI_CHANNEL_1(0.03, 0.03, 0.03) 1\nCX 0 3 1 3 1 4 2 4\nDEPOLARIZE2(0.03) 1 4\nM(0.03) 3 4\nHERALDED_PAULI_CHANNEL_1(0.03, 0.03, 0, 0.03) 2\nE(0.03) X0 X1\nELSE_CORRELATED_ERROR(0.03) X2\nCX rec[-1] 2\nDEPOLARIZE1(0.03) 0\nM 0 1 2\nDETECTOR rec[-6]\nDETECTOR rec[-5]\nDETECTOR rec[-3] rec[-2]\nOBSERVABLE_INCLUDE(0) rec[-1]",
            3,
        );
        check_exact("H 0 2\nCX 0 1 2 3\nDEPOLARIZE1(0.1) 0 1 2 3\nCX 0 2 1 3\nMX(0.02) 2 3\nDETECTOR rec[-1] rec[-2]\nMPP(0.01) X0*X1 Z0*Z1\nOBSERVABLE_INCLUDE(0) rec[-2]\nOBSERVABLE_INCLUDE(1) rec[-1]", 1);
    }

    #[test]
    fn polynomials_have_the_exact_taylor_coefficients() {
        let text = "R 0 1\nDEPOLARIZE1(0.3) 0 1\nCX 0 1\nM 1\nDETECTOR rec[-1]";
        let c = Circuit::parse(text).unwrap();
        let s = sites(&c).unwrap();
        // Two sites, each flipping the detector with probability 0.2 (X and Y on either):
        // P(fire) = 2 q (1 − q) with q = 0.2 x, = 0.4 x − 0.08 x².
        let poly = expand(&s.sites, 2, Some(2), 1 << 10).unwrap();
        let fire = &poly.iter().find(|(k, _)| k == &vec![0u32]).unwrap().1;
        assert!((fire[0]).abs() < 1e-15 && (fire[1] - 0.4).abs() < 1e-12 && (fire[2] + 0.08).abs() < 1e-12);
    }
}

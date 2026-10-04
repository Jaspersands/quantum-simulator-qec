//! Belief-matching (Higgott, Bohdanowicz, Kubica, Flammia and Campbell, PRX 13,
//! 031007, 2023): belief propagation on the whole error model, then matching
//! on weights from its posteriors.
//!
//! WHY THIS EXISTS
//! ---------------
//! Matching weighs every graph edge by its prior alone. A Y error sets off both
//! halves of the graph, and correlated matching patches that with one round of
//! reweighting. Belief-matching instead lets the whole hypergraph speak: BP
//! (`bp`) computes every fault's posterior given this shot, and the matcher
//! then runs on edges weighted by those posteriors. On Sycamore's data it was
//! one of the two decoders that made d = 5 beat d = 3.
//!
//! This follows `beliefmatching` (the authors' package, v0.2.0) step for step,
//! so the two can be compared shot by shot:
//!
//! - **Hyperedges.** Faults are keyed by their detectors (the XOR of their
//!   pieces'). Faults with the same detectors merge, their probabilities
//!   combined as independent events; the observables of the last one win, and
//!   the pieces of the first one are its decomposition into graph edges.
//! - **BP** runs on that hypergraph with the priors, 20 iterations, product-sum.
//!   If it converges, its own correction is the answer.
//! - **Otherwise,** each graph edge's probability is the *sum* of the
//!   posteriors of the hyperedges whose decomposition contains it, clipped to
//!   [1e-14, 1 − 1e-14], and its weight is −ln p. The sparse matcher decodes
//!   the shot on those weights.

use std::collections::HashMap;

use crate::bp::{Bp, BpWork, Method};
use crate::dem::Dem;
use crate::dem_decoder::{int_weight, merged_edges, DecodeError};
use crate::sparse::{Scratch, SparseGraph};

const EPS: f64 = 1e-14;

pub struct BeliefMatching {
    pub bp: Bp,
    pub graph: SparseGraph,
    /// Per hyperedge, the observables it flips.
    hyper_obs: Vec<u64>,
    /// Per hyperedge, the graph edges of its decomposition.
    hyper_start: Vec<u32>,
    hyper_edges: Vec<u32>,
    pub method: Method,
    pub max_iter: usize,
}

/// One thread's workspace.
pub struct BeliefWork {
    pub bp: BpWork,
    scratch: Scratch,
    syndrome: Vec<u8>,
    edge_p: Vec<f64>,
    edge_w: Vec<i64>,
}

/// How one shot was decoded.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BeliefOutcome {
    pub observables: u64,
    /// The matching's weight on this shot's posterior weights (−ln p per
    /// edge), or NaN where BP converged and no matching was run.
    pub weight: f64,
    /// BP's own correction explained the syndrome; matching was not run.
    pub converged: bool,
    pub iterations: usize,
}

impl BeliefMatching {
    /// The model's order matters, as in `beliefmatching`: where several
    /// faults share a symptom, the first one's pieces are the decomposition
    /// and the last one's observables stand. A model parsed from text keeps
    /// the text's order; `Dem::from_circuit` sorts its mechanisms, so belief-
    /// matching on this engine's own model of a circuit can split a shared
    /// symptom differently from `beliefmatching` on Stim's model of it.
    pub fn from_dem(dem: &Dem, method: Method, max_iter: usize) -> Result<BeliefMatching, String> {
        let (edges, _) = merged_edges(dem)?;
        let graph = SparseGraph::from_edges(dem.num_detectors, &edges);
        let boundary = dem.num_detectors as u32;
        let mut ids: HashMap<Vec<u32>, usize> = HashMap::new();
        let (mut columns, mut priors, mut obs) = (Vec::new(), Vec::new(), Vec::new());
        let mut decomposition: Vec<Vec<u32>> = Vec::new();
        for m in &dem.mechanisms {
            let id = *ids.entry(m.detectors.clone()).or_insert_with(|| {
                columns.push(m.detectors.clone());
                priors.push(0.0);
                obs.push(0);
                let mut edges: Vec<u32> = m
                    .pieces
                    .iter()
                    .filter_map(|piece| match piece.detectors.as_slice() {
                        [a] => graph.edge_id(*a, boundary),
                        [a, b] => graph.edge_id(*a, *b),
                        _ => None,
                    })
                    .collect();
                edges.sort_unstable();
                edges.dedup();
                decomposition.push(edges);
                columns.len() - 1
            });
            let q: f64 = priors[id];
            priors[id] = q * (1.0 - m.p) + m.p * (1.0 - q);
            obs[id] = m.observables;
        }
        let bp = Bp::new(dem.num_detectors, &columns, &priors)?;
        let mut hyper_start = vec![0u32];
        let mut hyper_edges = Vec::new();
        for d in decomposition {
            hyper_edges.extend(d);
            hyper_start.push(hyper_edges.len() as u32);
        }
        Ok(BeliefMatching { bp, graph, hyper_obs: obs, hyper_start, hyper_edges, method, max_iter })
    }

    pub fn work(&self) -> BeliefWork {
        BeliefWork {
            bp: self.bp.work(),
            scratch: Scratch::new(&self.graph),
            syndrome: vec![0; self.bp.num_checks],
            edge_p: vec![0.0; self.graph.num_edges()],
            edge_w: vec![0; self.graph.num_edges()],
        }
    }

    /// Decode one shot, `defects` sorted and distinct.
    pub fn decode(&self, defects: &[u32], w: &mut BeliefWork) -> Result<BeliefOutcome, DecodeError> {
        w.syndrome.fill(0);
        for &d in defects {
            w.syndrome[d as usize] = 1;
        }
        let out = self.bp.decode(&w.syndrome, self.method, self.max_iter, &mut w.bp);
        if out.converged {
            let observables = (0..self.bp.num_vars).filter(|&v| w.bp.hard[v] == 1).fold(0u64, |o, v| o ^ self.hyper_obs[v]);
            return Ok(BeliefOutcome { observables, weight: f64::NAN, converged: true, iterations: out.iterations });
        }
        w.edge_p.fill(0.0);
        for h in 0..self.bp.num_vars {
            let p = 1.0 / (1.0 + w.bp.llr[h].exp());
            // Infinite messages meeting (a certain fault said both to have and
            // not to have happened) leave no posterior to weigh an edge by: the
            // shot is refused, not matched on a made-up weight.
            if p.is_nan() {
                return Err(DecodeError::MatcherDeclined);
            }
            for &e in &self.hyper_edges[self.hyper_start[h] as usize..self.hyper_start[h + 1] as usize] {
                w.edge_p[e as usize] += p;
            }
        }
        for (wt, &p) in w.edge_w.iter_mut().zip(&w.edge_p) {
            *wt = int_weight(-p.clamp(EPS, 1.0 - EPS).ln());
        }
        let prediction = self.graph.decode_with_weights(&mut w.scratch, defects, &w.edge_w)?;
        Ok(BeliefOutcome { observables: prediction.observables, weight: prediction.weight, converged: false, iterations: out.iterations })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::circuit::Basis;
    use crate::dem_decoder::DemDecoder;
    use crate::frame_sampler::FrameSampler;
    use crate::memory::{generate, CodeKind, NoiseModel};
    use crate::surface_code::Xorshift;

    fn defects_of(dets: &[bool]) -> Vec<u32> {
        dets.iter().enumerate().filter(|x| *x.1).map(|x| x.0 as u32).collect()
    }

    fn model(kind: CodeKind, d: usize, p: f64, basis: Basis) -> (crate::circuit::Circuit, Dem) {
        let c = generate(kind, d, d, NoiseModel::Sd6 { p }, basis).unwrap();
        let dem = Dem::from_circuit(&c).unwrap();
        (c, dem)
    }

    /// Every fault of the model, alone, is corrected: its own logical flip is
    /// predicted, whether BP converges on it or the matcher decides.
    #[test]
    fn every_single_fault_is_corrected() {
        for kind in [CodeKind::Rotated, CodeKind::Xzzx] {
            for basis in [Basis::Z, Basis::X] {
                let (_, dem) = model(kind, 3, 0.004, basis);
                let bm = BeliefMatching::from_dem(&dem, Method::ProductSum, 20).unwrap();
                let mut w = bm.work();
                let mut failed = 0;
                for m in &dem.mechanisms {
                    let got = bm.decode(&m.detectors, &mut w).unwrap();
                    failed += usize::from(got.observables != m.observables);
                }
                assert_eq!(failed, 0, "{kind:?} {basis:?}: {failed} of {} faults", dem.mechanisms.len());
            }
        }
    }

    /// With the matcher's weights set to the model's own, decoding with
    /// weights is plain decoding: the same prediction and weight.
    #[test]
    fn decoding_with_the_models_weights_is_plain_decoding() {
        let (c, dem) = model(CodeKind::Rotated, 5, 0.006, Basis::Z);
        let dec = DemDecoder::new(&dem).unwrap();
        let g = dec.graph();
        let own: Vec<i64> = (0..g.num_edges() as u32).map(|e| g.weight_of(e)).collect();
        let sampler = FrameSampler::new(&c).unwrap();
        let mut rng = Xorshift::new(3);
        let mut s = Scratch::new(g);
        for _ in 0..300 {
            let defects = defects_of(&sampler.sample(&mut rng).detectors);
            let a = g.decode(&mut s, &defects).unwrap();
            let b = g.decode_with_weights(&mut s, &defects, &own).unwrap();
            assert_eq!((a.observables, a.iweight), (b.observables, b.iweight));
        }
    }

    /// On sampled shots, belief-matching fails no more often than plain
    /// matching (within noise), and usually less.
    #[test]
    fn belief_matching_does_not_lose_to_matching() {
        let (c, dem) = model(CodeKind::Rotated, 5, 0.006, Basis::Z);
        let dec = DemDecoder::new(&dem).unwrap();
        let bm = BeliefMatching::from_dem(&dem, Method::ProductSum, 20).unwrap();
        let mut w = bm.work();
        let sampler = FrameSampler::new(&c).unwrap();
        let mut rng = Xorshift::new(4);
        let (mut plain, mut belief, mut converged) = (0i64, 0i64, 0);
        let shots = 3000;
        for _ in 0..shots {
            let shot = sampler.sample(&mut rng);
            let defects = defects_of(&shot.detectors);
            plain += ((dec.decode(&defects).unwrap().observables ^ shot.observables) & 1) as i64;
            let b = bm.decode(&defects, &mut w).unwrap();
            converged += usize::from(b.converged);
            belief += ((b.observables ^ shot.observables) & 1) as i64;
        }
        println!("plain {plain}, belief-matching {belief}, BP converged on {converged} of {shots}");
        assert!(belief <= plain + 4 * ((plain + belief) as f64).sqrt() as i64, "{belief} vs {plain}");
    }
}

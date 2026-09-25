//! Minimum-weight perfect matching over any detector error model.
//!
//! The per-code decoders match on a graph whose edges carry data-qubit
//! corrections and unit weights. This one takes the standard formulation: each
//! edge is a graph-like piece of an error mechanism, weighted ln((1 − p)/p), and
//! carrying the logical observables it flips. What comes out is a prediction of
//! the observables, not a correction, which is all a memory experiment needs and
//! all an external dataset can be scored against.
//!
//! Shortest paths are Dijkstra from each defect; the matching is the existing
//! Edmonds blossom, with the boundary modelled as one interchangeable copy per
//! defect. Nothing here falls back to anything: a syndrome too large for the
//! dense matcher, or one with no consistent explanation, is an error the caller
//! counts. Bug 12 in the README is why.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};

use crate::blossom::{min_weight_perfect_matching, MAX_VERTICES};
use crate::dem::{xor_prob, Dem};
use crate::sparse::{Correlations, Scratch, SparseGraph};

/// Integer weight resolution: 2^20 per unit of ln((1 − p)/p). Fine enough that
/// rounding cannot reorder paths that differ by more than a few parts in 10^6.
pub const SCALE: f64 = 1_048_576.0;
pub const HALF_SCALE: f64 = SCALE / 2.0;

/// An edge's integer weight: ln((1 − p)/p) at `SCALE`, rounded to an even
/// integer. Even, because the sparse matcher's regions grow toward each other
/// from both ends of an edge and must meet at an integer time. Shared, so the
/// dense and sparse matchers solve exactly the same integer problem, and their
/// optimal weights can be required to be equal rather than merely close.
pub fn int_weight(wf: f64) -> i64 {
    2 * (wf * HALF_SCALE).round() as i64
}

/// An edge of probability `p`: its weight ln((1 − p)/p), and that weight as
/// `int_weight`. The one definition both matchers build their graphs from.
pub fn edge_weight(p: f64) -> (f64, i64) {
    let wf = ((1.0 - p) / p).ln();
    (wf, int_weight(wf))
}

/// Cost of a pairing with no path. Dominates any real path, and stays far enough
/// below the blossom's own infinity that sums of 512 of them cannot overflow.
const UNREACHABLE: i64 = 1 << 44;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Prediction {
    pub observables: u64,
    /// Total weight of the chosen matching. The dense matcher sums the float
    /// weights; the sparse matcher reports `iweight / SCALE`, which differs
    /// from that sum by the rounding, at most a few parts in 10^6 per edge.
    pub weight: f64,
    /// The same, in the integer weights the matcher actually minimises.
    pub iweight: i64,
}

#[derive(Clone, Debug, PartialEq)]
pub enum DecodeError {
    /// More defects than the dense matcher accepts (2k vertices > MAX_VERTICES).
    TooManyDefects(usize),
    /// A defect with no path to any partner or to the boundary.
    Unmatchable,
    /// The blossom hit one of its internal ceilings.
    MatcherDeclined,
}

struct Arc {
    to: u32,
    w: i64,
    wf: f64,
    obs: u64,
}

pub struct DemDecoder {
    num_detectors: usize,
    adj: Vec<Vec<Arc>>,
    /// Parallel edges whose observable masks disagreed. Zero for any code of
    /// distance three or more: a conflict is a weight-two logical operator.
    pub conflicts: usize,
    sparse: SparseGraph,
    /// Correlated matching's rules (see `sparse::correlated`).
    corr: Correlations,
    /// Workspace for `decode`. A decoder is used by one thread at a time;
    /// callers decoding in parallel share `graph()` and hold a `Scratch` each.
    scratch: std::cell::RefCell<Scratch>,
}

/// The model's graph-like pieces as merged edges `(u, v, p, observables)`,
/// sorted, with `v == num_detectors` standing for the boundary, and the number
/// of conflicting parallel edges. Parallel edges with the same observables
/// combine as independent events, which is what PyMatching 2.4 does. With
/// different observables the more probable one is kept and the conflict is
/// counted: for any code of distance three or more that count is zero, since a
/// conflict is a weight-two logical operator.
pub(crate) fn merged_edges(dem: &Dem) -> Result<(Vec<(u32, u32, f64, u64)>, usize), String> {
    let boundary = dem.num_detectors as u32;
    let mut edges: HashMap<(u32, u32), (f64, u64)> = HashMap::new();
    let mut conflicts = 0usize;
    for m in &dem.mechanisms {
        if m.pieces.is_empty() {
            return Err(format!(
                "mechanism on detectors {:?} fires more than two detectors and has no decomposition",
                m.detectors
            ));
        }
        for piece in &m.pieces {
            let key = match piece.detectors.as_slice() {
                [a] => (*a, boundary),
                [a, b] => (*a.min(b), *a.max(b)),
                other => return Err(format!("piece with {} detectors cannot be an edge", other.len())),
            };
            match edges.get_mut(&key) {
                None => {
                    edges.insert(key, (m.p, piece.observables));
                }
                Some(e) if e.1 == piece.observables => e.0 = xor_prob(e.0, m.p),
                Some(e) => {
                    conflicts += 1;
                    if m.p > e.0 {
                        *e = (m.p, piece.observables);
                    }
                }
            }
        }
    }
    let mut out: Vec<(u32, u32, f64, u64)> = edges.into_iter().map(|((u, v), (p, o))| (u, v, p, o)).collect();
    out.sort_unstable_by(|a, b| (a.0, a.1).cmp(&(b.0, b.1)));
    for &(u, v, p, _) in &out {
        if !(p > 0.0 && p <= 0.5) {
            return Err(format!("edge ({u}, {v}) has probability {p}; weights need 0 < p <= 0.5"));
        }
    }
    Ok((out, conflicts))
}

/// Sorted, with detectors listed an even number of times removed.
fn cancel_repeats(defects: &[u32]) -> Vec<u32> {
    let mut set: Vec<u32> = defects.to_vec();
    set.sort_unstable();
    let mut out: Vec<u32> = Vec::with_capacity(set.len());
    for d in set {
        if out.last() == Some(&d) {
            out.pop();
        } else {
            out.push(d);
        }
    }
    out
}

impl DemDecoder {
    pub fn new(dem: &Dem) -> Result<DemDecoder, String> {
        let nd = dem.num_detectors;
        let (edges, conflicts) = merged_edges(dem)?;
        let mut adj: Vec<Vec<Arc>> = (0..=nd).map(|_| Vec::new()).collect();
        for &(u, v, p, obs) in &edges {
            let (wf, w) = edge_weight(p);
            adj[u as usize].push(Arc { to: v, w, wf, obs });
            adj[v as usize].push(Arc { to: u, w, wf, obs });
        }
        let sparse = SparseGraph::from_edges(nd, &edges);
        let corr = Correlations::from_dem(dem, &sparse)?;
        let scratch = std::cell::RefCell::new(Scratch::new(&sparse));
        Ok(DemDecoder { num_detectors: nd, adj, conflicts, sparse, corr, scratch })
    }

    /// The detector graph the sparse matcher grows on.
    pub fn graph(&self) -> &SparseGraph {
        &self.sparse
    }

    /// The rules correlated matching reweights by.
    pub fn correlations(&self) -> &Correlations {
        &self.corr
    }

    pub fn decode_bools(&self, dets: &[bool]) -> Result<Prediction, DecodeError> {
        let defects: Vec<u32> = dets.iter().enumerate().filter(|(_, &b)| b).map(|(i, _)| i as u32).collect();
        self.decode(&defects)
    }

    /// Decode a set of fired detectors. A detector listed twice has fired an
    /// even number of times and cancels, as detection events XOR.
    ///
    /// Decodes with the sparse matcher, which has no ceiling on the number of
    /// defects.
    pub fn decode(&self, defects: &[u32]) -> Result<Prediction, DecodeError> {
        if !defects.windows(2).all(|w| w[0] < w[1]) {
            return self.decode(&cancel_repeats(defects));
        }
        self.sparse.decode(&mut self.scratch.borrow_mut(), defects)
    }

    /// The edges the matching uses, as PyMatching's `decode_to_edges_array`
    /// gives them: one shortest path per matched pair, XORed together, each
    /// edge as its endpoints with `num_detectors` for the boundary.
    pub fn decode_to_edges(&self, defects: &[u32]) -> Result<Vec<(u32, u32)>, DecodeError> {
        if !defects.windows(2).all(|w| w[0] < w[1]) {
            return self.decode_to_edges(&cancel_repeats(defects));
        }
        self.sparse.decode_to_edges(&mut self.scratch.borrow_mut(), defects)
    }

    /// The dense matcher: Dijkstra from every defect, then Edmonds' blossom on
    /// the complete graph of defects. Exact, capped at 256 defects, and kept as
    /// the reference the sparse matcher is checked against.
    pub fn decode_dense(&self, defects: &[u32]) -> Result<Prediction, DecodeError> {
        if !defects.windows(2).all(|w| w[0] < w[1]) {
            return self.decode_dense(&cancel_repeats(defects));
        }
        let k = defects.len();
        if k == 0 {
            return Ok(Prediction { observables: 0, weight: 0.0, iweight: 0 });
        }
        if 2 * k > MAX_VERTICES {
            return Err(DecodeError::TooManyDefects(k));
        }
        let nd = self.num_detectors;
        let boundary = nd;
        let mut slot = vec![u32::MAX; nd + 1];
        for (i, &d) in defects.iter().enumerate() {
            slot[d as usize] = i as u32;
        }

        // Row i: distance, observable parity and float weight from defect i to
        // every defect j > i, with column k the boundary.
        let mut dist = vec![vec![UNREACHABLE; k + 1]; k];
        let mut obs = vec![vec![0u64; k + 1]; k];
        let mut wsum = vec![vec![0f64; k + 1]; k];

        let mut d_node = vec![i64::MAX; nd + 1];
        let mut o_node = vec![0u64; nd + 1];
        let mut f_node = vec![0f64; nd + 1];
        let mut touched: Vec<usize> = Vec::new();
        let mut heap = BinaryHeap::new();

        for (i, &src) in defects.iter().enumerate() {
            for &t in &touched {
                d_node[t] = i64::MAX;
            }
            touched.clear();
            heap.clear();
            let src = src as usize;
            d_node[src] = 0;
            o_node[src] = 0;
            f_node[src] = 0.0;
            touched.push(src);
            heap.push(Reverse((0i64, src)));
            // Targets still to settle: defects after i, and the boundary.
            let mut remaining = (k - 1 - i) + 1;
            while let Some(Reverse((du, u))) = heap.pop() {
                if du > d_node[u] {
                    continue;
                }
                if u == boundary {
                    dist[i][k] = du;
                    obs[i][k] = o_node[u];
                    wsum[i][k] = f_node[u];
                    remaining -= 1;
                    if remaining == 0 {
                        break;
                    }
                    // The boundary is not a node paths may pass through.
                    continue;
                }
                let j = slot[u];
                if j != u32::MAX && (j as usize) > i {
                    dist[i][j as usize] = du;
                    obs[i][j as usize] = o_node[u];
                    wsum[i][j as usize] = f_node[u];
                    remaining -= 1;
                    if remaining == 0 {
                        break;
                    }
                }
                for arc in &self.adj[u] {
                    let v = arc.to as usize;
                    let nd2 = du + arc.w;
                    if nd2 < d_node[v] {
                        if d_node[v] == i64::MAX {
                            touched.push(v);
                        }
                        d_node[v] = nd2;
                        o_node[v] = o_node[u] ^ arc.obs;
                        f_node[v] = f_node[u] + arc.wf;
                        heap.push(Reverse((nd2, v)));
                    }
                }
            }
        }

        // Defects 0..k, then one boundary copy per defect; copies pair for free.
        let n = 2 * k;
        let mut cost = vec![vec![0i64; n]; n];
        for i in 0..k {
            for j in (i + 1)..k {
                cost[i][j] = dist[i][j];
                cost[j][i] = dist[i][j];
            }
            for c in 0..k {
                cost[i][k + c] = dist[i][k];
                cost[k + c][i] = dist[i][k];
            }
        }
        let mate = min_weight_perfect_matching(n, &cost).ok_or(DecodeError::MatcherDeclined)?;

        let mut prediction = Prediction { observables: 0, weight: 0.0, iweight: 0 };
        for i in 0..k {
            let j = mate[i];
            let (d, o, w) = if j >= k {
                (dist[i][k], obs[i][k], wsum[i][k])
            } else if i < j {
                (dist[i][j], obs[i][j], wsum[i][j])
            } else {
                continue;
            };
            if d >= UNREACHABLE {
                return Err(DecodeError::Unmatchable);
            }
            prediction.observables ^= o;
            prediction.weight += w;
            prediction.iweight += d;
        }
        Ok(prediction)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::circuit::Circuit;
    use crate::fixtures::REP3;
    use crate::surface_code::Xorshift;
    use std::collections::HashSet;

    type TestEdge = (u32, Option<u32>, f64, u64);

    /// Minimum-weight edge set with the given boundary, by exhaustion. Returns
    /// the weight and every observable mask that attains it (within 1e-4).
    fn brute_force(edges: &[TestEdge], defects: &[u32]) -> Option<(f64, Vec<u64>)> {
        let target: u64 = defects.iter().fold(0, |acc, &d| acc ^ (1 << d));
        let mut found: Vec<(f64, u64)> = Vec::new();
        for mask in 0u32..(1 << edges.len()) {
            let (mut syn, mut w, mut obs) = (0u64, 0.0, 0u64);
            for (i, e) in edges.iter().enumerate() {
                if (mask >> i) & 1 == 1 {
                    syn ^= 1 << e.0;
                    if let Some(b) = e.1 {
                        syn ^= 1 << b;
                    }
                    w += ((1.0 - e.2) / e.2).ln();
                    obs ^= e.3;
                }
            }
            if syn == target {
                found.push((w, obs));
            }
        }
        let best = found.iter().map(|f| f.0).fold(f64::INFINITY, f64::min);
        if !best.is_finite() {
            return None;
        }
        let mut masks: Vec<u64> = found.iter().filter(|f| f.0 <= best + 1e-4).map(|f| f.1).collect();
        masks.sort_unstable();
        masks.dedup();
        Some((best, masks))
    }

    #[test]
    fn matches_brute_force_on_small_graphs() {
        let mut rng = Xorshift::new(7);
        let mut compared = 0;
        for trial in 0..300 {
            let nd = 3 + (rng.next_u64() % 5) as usize;
            let mut edges: Vec<TestEdge> = Vec::new();
            let mut seen = HashSet::new();
            while edges.len() < 2 * nd {
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
            for d in 0..nd {
                text.push_str(&format!("detector D{d}\n"));
            }
            let dec = DemDecoder::new(&Dem::parse(&text).unwrap()).unwrap();
            let defects: Vec<u32> = (0..nd as u32).filter(|_| rng.next_u64() % 2 == 0).collect();
            let brute = brute_force(&edges, &defects);
            for (name, got) in [("sparse", dec.decode(&defects)), ("dense", dec.decode_dense(&defects))] {
                match (&brute, got) {
                    (None, Err(DecodeError::Unmatchable)) => {}
                    (Some((w, masks)), Ok(pred)) => {
                        assert!((pred.weight - w).abs() < 1e-4, "{name} trial {trial}: weight {} vs brute {w}", pred.weight);
                        assert!(masks.contains(&pred.observables), "{name} trial {trial}: obs {} not in {masks:?}", pred.observables);
                        compared += !defects.is_empty() as usize;
                    }
                    (b, d) => panic!("{name} trial {trial}: brute {b:?}, decoder {d:?}"),
                }
            }
        }
        // Not vacuous: most trials must have been a real matching problem.
        assert!(compared > 200, "only {compared} non-trivial comparisons");
    }

    #[test]
    fn corrects_every_single_fault_of_the_repetition_code() {
        let dem = Dem::from_circuit(&Circuit::parse(REP3).unwrap()).unwrap();
        let dec = DemDecoder::new(&dem).unwrap();
        for m in &dem.mechanisms {
            let pred = dec.decode(&m.detectors).unwrap();
            assert_eq!(pred.observables, m.observables, "mechanism {:?}", m.detectors);
        }
        assert_eq!(dec.conflicts, 0);
    }

    #[test]
    fn empty_syndrome_predicts_nothing() {
        let dem = Dem::from_circuit(&Circuit::parse(REP3).unwrap()).unwrap();
        let pred = DemDecoder::new(&dem).unwrap().decode(&[]).unwrap();
        assert_eq!((pred.observables, pred.weight), (0, 0.0));
    }

    #[test]
    fn too_many_defects_is_an_error_not_a_fallback() {
        let text: String = (0..300).map(|d| format!("error(0.1) D{d}\n")).collect();
        let dec = DemDecoder::new(&Dem::parse(&text).unwrap()).unwrap();
        let defects: Vec<u32> = (0..257).collect();
        assert_eq!(dec.decode_dense(&defects), Err(DecodeError::TooManyDefects(257)));
        // The sparse matcher has no such ceiling.
        assert!(dec.decode(&defects).is_ok());
    }

    #[test]
    fn unmatchable_syndromes_are_errors() {
        let dec = DemDecoder::new(&Dem::parse("error(0.1) D0 D1\ndetector D2").unwrap()).unwrap();
        assert_eq!(dec.decode(&[2]), Err(DecodeError::Unmatchable));
        assert_eq!(dec.decode(&[0]), Err(DecodeError::Unmatchable));
        assert!(dec.decode(&[0, 1]).is_ok());
    }

    #[test]
    fn repeated_detectors_cancel() {
        let dec = DemDecoder::new(&Dem::parse("error(0.1) D0 L0\nerror(0.1) D0 D1\nerror(0.1) D1").unwrap()).unwrap();
        assert_eq!(dec.decode(&[1, 0, 0]), dec.decode(&[1]));
        assert_eq!(dec.decode(&[1, 1]).unwrap().observables, 0);
    }

    #[test]
    fn integer_weights_are_even_and_reported() {
        assert_eq!(int_weight(1.0) % 2, 0);
        assert_eq!(int_weight(0.0), 0);
        let dec = DemDecoder::new(&Dem::parse("error(0.1) D0 D1 L0\nerror(0.2) D1").unwrap()).unwrap();
        let pred = dec.decode_dense(&[0, 1]).unwrap();
        assert_eq!(pred.iweight, int_weight((0.9f64 / 0.1).ln()));
        assert_eq!(pred.iweight % 2, 0);
    }

    #[test]
    fn rejects_probabilities_above_one_half() {
        assert!(DemDecoder::new(&Dem::parse("error(0.6) D0").unwrap()).is_err());
    }

    #[test]
    fn parallel_edges_merge_like_pymatching() {
        // PyMatching 2.4 gives this pair one edge of p = 0.26, weight 1.0459685551826876.
        let dec = DemDecoder::new(&Dem::parse("error(0.1) D0 D1 L0\nerror(0.2) D0 D1 L0").unwrap()).unwrap();
        let pred = dec.decode_dense(&[0, 1]).unwrap();
        assert_eq!(pred.observables, 1);
        assert!((pred.weight - 1.0459685551826876).abs() < 1e-12);
        // The sparse matcher reports its weight from the integer one.
        let pred = dec.decode(&[0, 1]).unwrap();
        assert_eq!(pred.observables, 1);
        assert!((pred.weight - 1.0459685551826876).abs() < 1e-5);
        assert_eq!(dec.conflicts, 0);
    }

    #[test]
    fn conflicting_parallel_edges_are_counted() {
        let dec = DemDecoder::new(&Dem::parse("error(0.1) D0 D1 L0\nerror(0.2) D0 D1").unwrap()).unwrap();
        assert_eq!(dec.conflicts, 1);
        assert_eq!(dec.decode(&[0, 1]).unwrap().observables, 0);
    }

    #[test]
    fn undecomposed_hyperedges_are_refused() {
        assert!(DemDecoder::new(&Dem::parse("error(0.1) D0 D1 D2").unwrap()).is_err());
    }
}

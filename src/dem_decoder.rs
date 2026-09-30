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

use std::cell::{OnceCell, RefCell};
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};

use crate::blossom::{min_weight_perfect_matching, MAX_VERTICES};
use crate::dem::{xor_prob, Dem};
use crate::sparse::{Correlations, FaultEdges, Scratch, SparseGraph};

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

pub struct DemDecoder {
    num_detectors: usize,
    /// Parallel edges whose observable masks disagreed. Zero for any code of
    /// distance three or more: a conflict is a weight-two logical operator.
    pub conflicts: usize,
    sparse: SparseGraph,
    /// Correlated matching's rules (see `sparse::correlated`), built on first
    /// use from `faults`, which is then dropped: plain matching never reads
    /// them, and they cost more to build than the graph.
    corr: OnceCell<Correlations>,
    faults: RefCell<Option<FaultEdges>>,
    /// Workspace for `decode`. A decoder is used by one thread at a time;
    /// callers decoding in parallel share `graph()` and hold a `Scratch` each.
    scratch: RefCell<Scratch>,
    /// The dense reference matcher's workspace, made on first use.
    dense: RefCell<Option<DenseScratch>>,
}

/// What `decode_dense` keeps between shots: Dijkstra's state over the
/// detectors and the boundary, the defect-to-defect tables, and the
/// blossom's cost matrix.
struct DenseScratch {
    slot: Vec<u32>,
    d_node: Vec<i64>,
    o_node: Vec<u64>,
    f_node: Vec<f64>,
    touched: Vec<usize>,
    heap: BinaryHeap<Reverse<(i64, usize)>>,
    /// Row i, column j of k + 1: from defect i to defect j, column k the boundary.
    dist: Vec<i64>,
    obs: Vec<u64>,
    wsum: Vec<f64>,
    cost: Vec<Vec<i64>>,
}

impl DenseScratch {
    fn new(nd: usize) -> DenseScratch {
        DenseScratch {
            slot: vec![u32::MAX; nd + 1],
            d_node: vec![i64::MAX; nd + 1],
            o_node: vec![0; nd + 1],
            f_node: vec![0.0; nd + 1],
            touched: Vec::new(),
            heap: BinaryHeap::new(),
            dist: Vec::new(),
            obs: Vec::new(),
            wsum: Vec::new(),
            cost: Vec::new(),
        }
    }
}

/// Merged graph edges (u, v, probability, observables), and how many conflicts were counted.
pub(crate) type MergedEdges = (Vec<(u32, u32, f64, u64)>, usize);

/// The model's graph-like pieces as merged edges `(u, v, p, observables)`,
/// sorted, with `v == num_detectors` standing for the boundary, and the number
/// of conflicting parallel edges. Parallel edges with the same observables
/// combine as independent events, which is what PyMatching 2.4 does. With
/// different observables the more probable one is kept and the conflict is
/// counted: for any code of distance three or more that count is zero, since a
/// conflict is a weight-two logical operator.
pub(crate) fn merged_edges(dem: &Dem) -> Result<MergedEdges, String> {
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
                // A piece of observables alone fires nothing to match; PyMatching
                // drops it too.
                [] => continue,
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
        let sparse = SparseGraph::from_edges(nd, &edges);
        let faults = RefCell::new(Some(FaultEdges::from_dem(dem, &sparse)?));
        let scratch = RefCell::new(Scratch::new(&sparse));
        Ok(DemDecoder {
            num_detectors: nd,
            conflicts,
            sparse,
            corr: OnceCell::new(),
            faults,
            scratch,
            dense: RefCell::new(None),
        })
    }

    pub fn num_detectors(&self) -> usize {
        self.num_detectors
    }

    /// The detector graph the sparse matcher grows on.
    pub fn graph(&self) -> &SparseGraph {
        &self.sparse
    }

    /// The rules correlated matching reweights by, built on first use.
    pub fn correlations(&self) -> &Correlations {
        self.corr.get_or_init(|| {
            let faults = self.faults.borrow_mut().take().expect("the faults are kept until the rules are built");
            Correlations::from_faults(&faults)
        })
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

    /// Correlated matching, as PyMatching's `enable_correlations=True`.
    pub fn decode_correlated(&self, defects: &[u32]) -> Result<Prediction, DecodeError> {
        if !defects.windows(2).all(|w| w[0] < w[1]) {
            return self.decode_correlated(&cancel_repeats(defects));
        }
        self.sparse.decode_correlated(self.correlations(), &mut self.scratch.borrow_mut(), defects)
    }

    /// Correlated matching's second pass alone, from edges given as endpoints
    /// (`num_detectors` for the boundary), as PyMatching's
    /// `decode_to_edges_array` returns them.
    pub fn decode_pass2(&self, defects: &[u32], edges: &[(u32, u32)]) -> Result<Prediction, String> {
        let defects = if defects.windows(2).all(|w| w[0] < w[1]) { defects.to_vec() } else { cancel_repeats(defects) };
        let ids = edges
            .iter()
            .map(|&(u, v)| self.sparse.edge_id(u, v).ok_or_else(|| format!("({u}, {v}) is not an edge of the graph")))
            .collect::<Result<Vec<u32>, String>>()?;
        self.sparse
            .decode_pass2(self.correlations(), &mut self.scratch.borrow_mut(), &defects, &ids)
            .map_err(|e| format!("{e:?}"))
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
        let mut guard = self.dense.borrow_mut();
        let ws = guard.get_or_insert_with(|| DenseScratch::new(nd));
        for (i, &d) in defects.iter().enumerate() {
            ws.slot[d as usize] = i as u32;
        }
        let result = Self::dense_match(&self.sparse, ws, defects, boundary);
        for &d in defects {
            ws.slot[d as usize] = u32::MAX;
        }
        result
    }

    /// The dense matcher's work, on a workspace whose `slot` names the defects.
    fn dense_match(g: &SparseGraph, ws: &mut DenseScratch, defects: &[u32], boundary: usize) -> Result<Prediction, DecodeError> {
        let k = defects.len();
        let row = k + 1;
        // Row i: distance, observable parity and float weight from defect i to
        // every defect j > i, with column k the boundary.
        ws.dist.clear();
        ws.dist.resize(k * row, UNREACHABLE);
        ws.obs.clear();
        ws.obs.resize(k * row, 0);
        ws.wsum.clear();
        ws.wsum.resize(k * row, 0.0);

        for (i, &src) in defects.iter().enumerate() {
            for &t in &ws.touched {
                ws.d_node[t] = i64::MAX;
            }
            ws.touched.clear();
            ws.heap.clear();
            let src = src as usize;
            ws.d_node[src] = 0;
            ws.o_node[src] = 0;
            ws.f_node[src] = 0.0;
            ws.touched.push(src);
            ws.heap.push(Reverse((0i64, src)));
            // Targets still to settle: defects after i, and the boundary.
            let mut remaining = (k - 1 - i) + 1;
            while let Some(Reverse((du, u))) = ws.heap.pop() {
                if du > ws.d_node[u] {
                    continue;
                }
                if u == boundary {
                    ws.dist[i * row + k] = du;
                    ws.obs[i * row + k] = ws.o_node[u];
                    ws.wsum[i * row + k] = ws.f_node[u];
                    remaining -= 1;
                    if remaining == 0 {
                        break;
                    }
                    // The boundary is not a node paths may pass through.
                    continue;
                }
                let j = ws.slot[u];
                if j != u32::MAX && (j as usize) > i {
                    ws.dist[i * row + j as usize] = du;
                    ws.obs[i * row + j as usize] = ws.o_node[u];
                    ws.wsum[i * row + j as usize] = ws.f_node[u];
                    remaining -= 1;
                    if remaining == 0 {
                        break;
                    }
                }
                for (to, w, obs, wf) in g.arcs(u as u32) {
                    let v = to.map_or(boundary, |t| t as usize);
                    let nd2 = du + w;
                    if nd2 < ws.d_node[v] {
                        if ws.d_node[v] == i64::MAX {
                            ws.touched.push(v);
                        }
                        ws.d_node[v] = nd2;
                        ws.o_node[v] = ws.o_node[u] ^ obs;
                        ws.f_node[v] = ws.f_node[u] + wf;
                        ws.heap.push(Reverse((nd2, v)));
                    }
                }
            }
        }

        // Defects 0..k, then one boundary copy per defect; copies pair for free.
        let n = 2 * k;
        ws.cost.resize_with(n, Vec::new);
        ws.cost.truncate(n);
        for r in ws.cost.iter_mut() {
            r.clear();
            r.resize(n, 0);
        }
        for i in 0..k {
            for j in (i + 1)..k {
                ws.cost[i][j] = ws.dist[i * row + j];
                ws.cost[j][i] = ws.dist[i * row + j];
            }
            for c in 0..k {
                ws.cost[i][k + c] = ws.dist[i * row + k];
                ws.cost[k + c][i] = ws.dist[i * row + k];
            }
        }
        let mate = min_weight_perfect_matching(n, &ws.cost).ok_or(DecodeError::MatcherDeclined)?;

        let mut prediction = Prediction { observables: 0, weight: 0.0, iweight: 0 };
        for i in 0..k {
            let j = mate[i];
            let at = if j >= k {
                i * row + k
            } else if i < j {
                i * row + j
            } else {
                continue;
            };
            if ws.dist[at] >= UNREACHABLE {
                return Err(DecodeError::Unmatchable);
            }
            prediction.observables ^= ws.obs[at];
            prediction.weight += ws.wsum[at];
            prediction.iweight += ws.dist[at];
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
                let b = if rng.next_u64().is_multiple_of(3) { None } else { Some((rng.next_u64() % nd as u64) as u32) };
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
            let defects: Vec<u32> = (0..nd as u32).filter(|_| rng.next_u64().is_multiple_of(2)).collect();
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
    fn a_piece_of_observables_alone_is_no_edge() {
        // Stim's split of a fault through a conflicting piece; PyMatching builds
        // the same two edges from it.
        let dem = Dem::parse("error(0.05) D0 D1 L0 ^ D2 ^ L0\nerror(0.1) D0 D1\nerror(0.1) D2\nerror(0.2) D0 D1 L0").unwrap();
        let dec = DemDecoder::new(&dem).unwrap();
        assert_eq!(dec.graph().num_edges(), 2);
        assert!(dec.decode(&[0, 1, 2]).is_ok());
        assert!(dec.decode_correlated(&[0, 1, 2]).is_ok());
    }

    #[test]
    fn undecomposed_hyperedges_are_refused() {
        assert!(DemDecoder::new(&Dem::parse("error(0.1) D0 D1 D2").unwrap()).is_err());
    }
}

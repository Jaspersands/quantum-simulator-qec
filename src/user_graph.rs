// Copyright 2022 PyMatching Contributors
// Licensed under the Apache License, Version 2.0 (see src/pm_exact/LICENSE). Modified: transliterated
// to Rust for this crate; owned decoder scratch caching and fallible output allocation added.

//! PyMatching's `UserGraph` (`user_graph.cc`, PyMatching 2.4): a matching graph built by hand.
//! Nodes, edges carrying sets of fault ids, weights and error probabilities, boundary nodes, and
//! the five ways a parallel edge merges. It decodes on `pm_exact`, PyMatching's sparse blossom
//! transliterated, on the matching graph PyMatching builds from it (`UserGraph::to_mwpm`): integer
//! weights as they are, otherwise scaled so the heaviest edge is 2^24 − 1, rounded, and doubled;
//! a negative weight's edge pre-applied (its endpoints toggled, its fault ids flipped, its weight
//! counted) and matched at its absolute weight; edges to boundary nodes made boundary edges, the
//! lightest kept where several meet one node. So its answers are PyMatching's, ties included.
//!
//! A model loaded with correlations (`UserGraph::from_dem_with_correlations`) is PyMatching's
//! correlated load: each edge's endpoints in order, and each edge carrying the weights it implies
//! for the edges that fault with it, which the weight scale counts, so its plain decodes are
//! PyMatching's too.
//!
//! The 2.1 decoding methods (`UserDecoder::scratch`, `decode`, `decode_to_edges`,
//! `decode_to_matched_pairs`, the four-argument `decode_packed`, and `Decoded`) remain, deprecated,
//! as thin wrappers over `pm_exact`.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use crate::dem_program::{DemInstr, DemProgram, OBS};
use crate::pm_exact::{Graph, Mwpm, NeighbourOrder, NUM_DISTINCT_WEIGHTS};
use crate::sparse::{Scratch, SparseGraph};

/// PyMatching's `NUM_DISTINCT_WEIGHTS - 1`: the heaviest edge after scaling, and the largest
/// weight a user edge may have.
pub const MAX_USER_EDGE_WEIGHT: f64 = ((1u32 << 24) - 1) as f64;

/// How an edge joins one already between the same nodes (PyMatching's `merge_strategy`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Merge {
    Disallow,
    Independent,
    SmallestWeight,
    KeepOriginal,
    Replace,
}

impl Merge {
    pub fn parse(name: &str) -> Result<Merge, String> {
        Ok(match name {
            "disallow" => Merge::Disallow,
            "independent" => Merge::Independent,
            "smallest-weight" => Merge::SmallestWeight,
            "keep-original" => Merge::KeepOriginal,
            "replace" => Merge::Replace,
            _ => return Err(format!("Merge strategy \"{name}\" not recognised.")),
        })
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct UserEdge {
    pub node1: usize,
    /// None for the boundary.
    pub node2: Option<usize>,
    pub fault_ids: Vec<usize>,
    pub weight: f64,
    /// Outside [0, 1] when unknown.
    pub error_probability: f64,
}

/// PyMatching's merge of two independent mechanisms' weights (log-likelihood ratios), computed
/// without overflow.
pub fn merge_weights(a: f64, b: f64) -> f64 {
    let sgn = 1f64.copysign(a) * 1f64.copysign(b);
    let signed_min = sgn * a.abs().min(b.abs());
    signed_min + (1.0 + (-(a + b).abs()).exp()).ln() - (1.0 + (-(a - b).abs()).exp()).ln()
}

/// A weight a correlated load attaches to an edge for another edge (PyMatching's
/// `ImpliedWeightUnconverted`): when correlated matching's first pass uses the edge, edge
/// (`node1`, `node2`; None: the boundary) may be lowered to `weight`. Plain decoding uses only its
/// size: the weight scale counts it.
#[derive(Clone, Debug, PartialEq)]
pub struct ImpliedWeight {
    pub node1: usize,
    pub node2: Option<usize>,
    pub weight: f64,
}

#[derive(Clone, Debug, Default)]
pub struct UserGraph {
    /// Per node, the ids of the edges touching it, in the order they were added.
    adjacency: Vec<Vec<usize>>,
    pub edges: Vec<UserEdge>,
    boundary: BTreeSet<usize>,
    num_fault_ids: usize,
    all_have_probabilities: bool,
    /// Per edge id, the weights it implies for others (PyMatching's
    /// `implied_weights_for_other_edges`); empty unless a correlated load filled it.
    implied: Vec<Vec<ImpliedWeight>>,
}

/// The decoder of a graph as it was: PyMatching's matching graph, shared between threads; each
/// thread decodes on its own `Mwpm` over it.
#[derive(Clone, Debug)]
pub struct UserDecoder {
    graph: Arc<Graph>,
}

/// A published build outlives every scratch lease, including failed ones. In particular,
/// ensuring a larger public width does not change this decoder's historical tie path.
#[cfg(any(feature = "python", test))]
struct DecoderBuild {
    decoder: UserDecoder,
    search: bool,
}

#[cfg(any(feature = "python", test))]
pub(crate) struct Engine {
    pub decoder: UserDecoder,
    pub mwpm: Mwpm,
}

#[cfg(any(feature = "python", test))]
#[derive(Default)]
struct CacheState {
    build: Option<Arc<DecoderBuild>>,
    // Bound retained working sets without limiting independent active calls.
    idle: [Option<Mwpm>; 4],
}

#[cfg(any(feature = "python", test))]
#[derive(Default)]
pub(crate) struct DecoderCache(std::sync::Mutex<CacheState>);

#[cfg(any(feature = "python", test))]
impl DecoderCache {
    fn lock(&self) -> std::sync::MutexGuard<'_, CacheState> {
        self.0.lock().unwrap_or_else(|poison| {
            let mut state = poison.into_inner();
            // Only bookkeeping is locked; no possibly-dirty scratch is trusted after poison.
            state.idle = Default::default();
            self.0.clear_poison();
            state
        })
    }

    pub fn run<R>(&self, g: &UserGraph, search: bool, f: impl FnOnce(&mut Engine) -> Result<R, String>) -> Result<R, String> {
        let (build, scratch) = loop {
            let mut state = self.lock();
            let rebuild = state.build.as_ref().is_none_or(|b| search && !b.search && b.decoder.num_nodes() != 0);
            if !rebuild {
                let build = state.build.as_ref().unwrap().clone();
                let scratch = state.idle.iter_mut().find_map(Option::take);
                break (build, scratch);
            }
            let previous = state.build.clone();
            drop(state);
            // Construction may fail or unwind. Publish only a fully constructed candidate,
            // outside the bookkeeping lock, and never overwrite another caller's upgrade.
            let decoder = g.decoder()?;
            let scratch = decoder.mwpm();
            let candidate = Arc::new(DecoderBuild { search: search || decoder.num_fault_ids() > 64, decoder });
            let mut state = self.lock();
            let unchanged = match (&previous, &state.build) {
                (None, None) => true,
                (Some(a), Some(b)) => Arc::ptr_eq(a, b),
                _ => false,
            };
            if unchanged {
                state.build = Some(candidate.clone());
                state.idle = Default::default();
                break (candidate, Some(scratch));
            }
        };
        let mut engine = Engine { decoder: build.decoder.clone(), mwpm: scratch.unwrap_or_else(|| build.decoder.mwpm()) };
        // No cache lock is held during decoding. Errors and unwinds drop this engine;
        // only successful operations may return scratch to the still-current build.
        let result = f(&mut engine)?;
        let mut state = self.lock();
        if state.build.as_ref().is_some_and(|b| Arc::ptr_eq(b, &build)) {
            if let Some(slot) = state.idle.iter_mut().find(|slot| slot.is_none()) {
                *slot = Some(engine.mwpm);
            }
        }
        Ok(result)
    }
}

/// One shot's decoding: the fault ids flipped, and the weight (`UserDecoder::decode`).
#[deprecated(since = "2.2.0", note = "use `UserDecoder::mwpm` and `Mwpm::try_decode`, which give the fault ids as bytes")]
#[derive(Clone, Debug, PartialEq)]
pub struct Decoded {
    pub faults: Vec<bool>,
    pub weight: f64,
}

/// PyMatching's `SIZE_MAX` as an edge's second node: the boundary (in a correlated load's keys).
const BOUNDARY: usize = usize::MAX;

/// Per edge (node1 ≤ node2, `BOUNDARY` last), the probability it fires together with each other
/// edge, and alone (its marginal) on the diagonal: PyMatching's `joint_probabilites`.
type JointProbabilities = BTreeMap<(usize, usize), BTreeMap<(usize, usize), f64>>;

const TOO_LIKELY: &str = "Errors with probability greater than 0.5 are not supported with correlations enabled";
const UNDECOMPOSED: &str = "Encountered an undecomposed error instruction with 3 or mode detectors. This is not supported when using `enable_correlations=True`. Did you forget to set `decompose_errors=True` when converting the stim circuit to a detector error model?";
const UNDETECTABLE: &str = "Encountered a decomposed error instruction with an undetectable component (0 detectors). This is not supported.";

/// PyMatching's `bernoulli_xor`: the probability exactly one of two independent events happens.
fn bernoulli_xor(p1: f64, p2: f64) -> f64 {
    p1 * (1.0 - p2) + p2 * (1.0 - p1)
}

impl UserGraph {
    pub fn new(num_nodes: usize, num_fault_ids: usize) -> UserGraph {
        UserGraph { adjacency: vec![Vec::new(); num_nodes], edges: Vec::new(), boundary: BTreeSet::new(), num_fault_ids, all_have_probabilities: true, implied: Vec::new() }
    }

    pub fn num_nodes(&self) -> usize {
        self.adjacency.len()
    }

    pub fn num_edges(&self) -> usize {
        self.edges.len()
    }

    pub fn num_fault_ids(&self) -> usize {
        self.num_fault_ids
    }

    pub fn num_detectors(&self) -> usize {
        self.num_nodes() - self.boundary.len()
    }

    pub fn boundary(&self) -> &BTreeSet<usize> {
        &self.boundary
    }

    /// The ids of the edges touching `node`, in the order they were added (a self-loop once).
    pub fn adjacency(&self, node: usize) -> &[usize] {
        &self.adjacency[node]
    }

    pub fn all_edges_have_error_probabilities(&self) -> bool {
        self.all_have_probabilities
    }

    /// The weights edge `edge` implies for other edges (empty unless loaded with correlations).
    pub fn implied_weights(&self, edge: usize) -> &[ImpliedWeight] {
        self.implied.get(edge).map_or(&[], Vec::as_slice)
    }

    /// Sets the weights edge `edge` implies for other edges, as a correlated load does (for
    /// rebuilding such a graph).
    pub fn set_implied_weights(&mut self, edge: usize, implied: Vec<ImpliedWeight>) -> Result<(), String> {
        if edge >= self.edges.len() {
            return Err(format!("edge {edge} not in the graph, which has {} edges", self.edges.len()));
        }
        if self.implied.len() <= edge {
            self.implied.resize(edge + 1, Vec::new());
        }
        self.implied[edge] = implied;
        Ok(())
    }

    pub fn ensure_num_fault_ids(&mut self, n: usize) {
        self.num_fault_ids = self.num_fault_ids.max(n);
    }

    fn ensure_nodes(&mut self, n: usize) {
        if self.adjacency.len() < n {
            self.adjacency.resize(n, Vec::new());
        }
    }

    pub fn set_boundary(&mut self, boundary: BTreeSet<usize>) {
        if let Some(&m) = boundary.iter().next_back() {
            self.ensure_nodes(m + 1);
        }
        self.boundary = boundary;
    }

    pub fn is_boundary(&self, node: Option<usize>) -> bool {
        node.is_none_or(|n| self.boundary.contains(&n))
    }

    /// The edge between `node` and `other` (None: the boundary edge), if any.
    pub fn find(&self, node: usize, other: Option<usize>) -> Option<usize> {
        let adj = self.adjacency.get(node)?;
        adj.iter().copied().find(|&e| {
            let ed = &self.edges[e];
            let across = if ed.node1 == node { ed.node2 } else { Some(ed.node1) };
            across == other
        })
    }

    fn note_new(&mut self, fault_ids: &[usize], p: f64) {
        if let Some(&m) = fault_ids.iter().max() {
            self.num_fault_ids = self.num_fault_ids.max(m + 1);
        }
        if p < 0.0 || p > 1.0 {
            self.all_have_probabilities = false;
        }
    }

    fn merge(&mut self, e: usize, fault_ids: &[usize], weight: f64, p: f64, how: Merge) -> Result<bool, String> {
        let old = &self.edges[e];
        match how {
            Merge::Disallow => {
                let n2 = old.node2.map_or(usize::MAX, |v| v);
                Err(format!(
                    "Edge ({}, {n2}) already exists in the graph. Parallel edges not permitted with the provided `disallow` `merge_strategy`. Please provide a different `merge_strategy`.",
                    old.node1
                ))
            }
            Merge::KeepOriginal => Ok(false),
            Merge::SmallestWeight if weight >= old.weight => Ok(false),
            Merge::Replace | Merge::SmallestWeight => {
                let ed = &mut self.edges[e];
                ed.weight = weight;
                ed.error_probability = p;
                ed.fault_ids = fault_ids.to_vec();
                if p < 0.0 || p > 1.0 {
                    self.all_have_probabilities = false;
                }
                Ok(true)
            }
            Merge::Independent => {
                let ed = &mut self.edges[e];
                let q = ed.error_probability;
                let valid = |x: f64| (0.0..=1.0).contains(&x);
                ed.weight = merge_weights(weight, ed.weight);
                ed.error_probability = if valid(q) && valid(p) { p * (1.0 - q) + q * (1.0 - p) } else { -1.0 };
                if !valid(ed.error_probability) {
                    self.all_have_probabilities = false;
                }
                Ok(true)
            }
        }
    }

    /// PyMatching's `add_or_merge_edge`.
    pub fn add_edge(&mut self, node1: usize, node2: usize, fault_ids: &[usize], weight: f64, p: f64, how: Merge) -> Result<(), String> {
        self.add_edge_changed(node1, node2, fault_ids, weight, p, how).map(|_| ())
    }

    pub(crate) fn add_edge_changed(&mut self, node1: usize, node2: usize, fault_ids: &[usize], weight: f64, p: f64, how: Merge) -> Result<bool, String> {
        self.ensure_nodes(node1.max(node2) + 1);
        match self.find(node1, Some(node2)) {
            Some(e) => self.merge(e, fault_ids, weight, p, how),
            None => {
                let id = self.edges.len();
                self.edges.push(UserEdge { node1, node2: Some(node2), fault_ids: fault_ids.to_vec(), weight, error_probability: p });
                self.adjacency[node1].push(id);
                if node1 != node2 {
                    self.adjacency[node2].push(id);
                }
                self.note_new(fault_ids, p);
                Ok(true)
            }
        }
    }

    /// PyMatching's `add_or_merge_boundary_edge`.
    pub fn add_boundary_edge(&mut self, node: usize, fault_ids: &[usize], weight: f64, p: f64, how: Merge) -> Result<(), String> {
        self.add_boundary_edge_changed(node, fault_ids, weight, p, how).map(|_| ())
    }

    pub(crate) fn add_boundary_edge_changed(&mut self, node: usize, fault_ids: &[usize], weight: f64, p: f64, how: Merge) -> Result<bool, String> {
        self.ensure_nodes(node + 1);
        match self.find(node, None) {
            Some(e) => self.merge(e, fault_ids, weight, p, how),
            None => {
                let id = self.edges.len();
                self.edges.push(UserEdge { node1: node, node2: None, fault_ids: fault_ids.to_vec(), weight, error_probability: p });
                self.adjacency[node].push(id);
                self.note_new(fault_ids, p);
                Ok(true)
            }
        }
    }

    /// PyMatching's graph of a detector error model (`detector_error_model_to_user_graph`): each
    /// piece of each error with probability above 0, its targets as written (repeats kept), is an
    /// edge if it has two detectors (one: to the boundary; otherwise nothing) of weight
    /// ln((1 − p)/p), parallel pieces merged as independent mechanisms.
    pub fn from_dem(dem: &DemProgram) -> Result<UserGraph, String> {
        UserGraph::load_dem(dem, false)
    }

    /// PyMatching's graph of a detector error model loaded with `enable_correlations=True`
    /// (`iter_dem_instructions_include_correlations`, then `populate_implied_edge_weights`). As
    /// `from_dem`, except:
    /// - each piece's two detectors are put in order, so an edge's `node1` is the smaller;
    /// - an error is refused, with PyMatching's message, if its probability is above 0.5, if a
    ///   piece has three or more detectors, or if it has several pieces and one has none;
    /// - each edge carries the weights it implies for the edges that fault with it,
    ///   ln((1 − q)/q) with q = min(0.5, joint / marginal), and the weight scale counts them.
    pub fn from_dem_with_correlations(dem: &DemProgram) -> Result<UserGraph, String> {
        UserGraph::load_dem(dem, true)
    }

    fn load_dem(dem: &DemProgram, correlations: bool) -> Result<UserGraph, String> {
        let stats = dem.stats();
        let max = crate::dem_program::MAX_DETECTOR;
        if stats.num_detectors > max + 1 {
            return Err(format!("the model names detector {}, past the largest index, {max}", stats.num_detectors - 1));
        }
        let mut g = UserGraph::new(stats.num_detectors as usize, stats.num_observables);
        let mut joint = JointProbabilities::new();
        for i in &dem.flattened()?.instrs {
            let DemInstr::Error { p, pieces, .. } = i else { continue };
            let p = *p;
            if correlations {
                g.add_correlated_error(p, pieces, &mut joint)?;
                continue;
            }
            if p <= 0.0 {
                continue; // as PyMatching
            }
            for piece in pieces {
                let dets: Vec<usize> = piece.iter().filter(|&&t| t & OBS == 0).map(|&t| t as usize).collect();
                let obs: Vec<usize> = piece.iter().filter(|&&t| t & OBS != 0).map(|&t| (t & !OBS) as usize).collect();
                let w = ((1.0 - p) / p).ln();
                match dets.as_slice() {
                    [a, b] => g.add_edge(*a, *b, &obs, w, p, Merge::Independent)?,
                    [a] => g.add_boundary_edge(*a, &obs, w, p, Merge::Independent)?,
                    _ => {}
                }
            }
        }
        if correlations {
            g.populate_implied_edge_weights(&joint);
        }
        Ok(g)
    }

    /// One error of a correlated load (`iter_dem_instructions_include_correlations`'s body, then
    /// `handle_dem_instruction_include_correlations` and
    /// `add_decomposed_error_to_joint_probabilities`).
    fn add_correlated_error(&mut self, p: f64, pieces: &[Vec<u64>], joint: &mut JointProbabilities) -> Result<(), String> {
        if p > 0.5 {
            return Err(TOO_LIKELY.into());
        }
        if p == 0.0 {
            return Ok(());
        }
        // Each piece as (node1, node2 or BOUNDARY, observables), node1 ≤ node2, its targets read in
        // order: a third detector is refused on sight, an empty piece where pieces are separated.
        let mut components: Vec<(usize, usize, Vec<usize>)> = Vec::with_capacity(pieces.len());
        for piece in pieces {
            let (mut node1, mut node2, mut n) = (BOUNDARY, BOUNDARY, 0);
            let mut obs = Vec::new();
            for &t in piece {
                if t & OBS != 0 {
                    obs.push((t & !OBS) as usize);
                    continue;
                }
                let d = t as usize;
                n += 1;
                match n {
                    1 => node1 = d,
                    2 if node1 <= d => node2 = d,
                    2 => (node1, node2) = (d, node1),
                    _ => return Err(UNDECOMPOSED.into()),
                }
            }
            if n == 0 {
                if pieces.len() > 1 {
                    return Err(UNDETECTABLE.into());
                }
                return Ok(()); // undetectable, and not part of a decomposition: ignored
            }
            components.push((node1, node2, obs));
        }
        if components.is_empty() {
            return Ok(());
        }
        let w = ((1.0 - p) / p).ln();
        for (a, b, obs) in &components {
            if *b == BOUNDARY {
                self.add_boundary_edge(*a, obs, w, p, Merge::Independent)?;
            } else {
                self.add_edge(*a, *b, obs, w, p, Merge::Independent)?;
            }
        }
        if components.len() > 1 {
            for k0 in 0..components.len() {
                for k1 in k0 + 1..components.len() {
                    let e0 = (components[k0].0, components[k0].1);
                    let e1 = (components[k1].0, components[k1].1);
                    // PyMatching's two references; one entry, taking p twice, when e0 == e1.
                    for (a, b) in [(e0, e1), (e1, e0)] {
                        let q = joint.entry(a).or_default().entry(b).or_insert(0.0);
                        *q = bernoulli_xor(*q, p);
                    }
                }
            }
        }
        for (a, b, _) in &components {
            let q = joint.entry((*a, *b)).or_default().entry((*a, *b)).or_insert(0.0);
            *q = bernoulli_xor(*q, p);
        }
        Ok(())
    }

    /// PyMatching's `populate_implied_edge_weights`: for each edge, in order, and each other edge
    /// that faults with it (in key order), the weight ln((1 − q)/q), q = min(0.5, joint /
    /// marginal).
    fn populate_implied_edge_weights(&mut self, joint: &JointProbabilities) {
        self.implied = vec![Vec::new(); self.edges.len()];
        for (id, e) in self.edges.iter().enumerate() {
            let n2 = e.node2.unwrap_or(BOUNDARY);
            let causal = (e.node1.min(n2), e.node1.max(n2));
            let Some(row) = joint.get(&causal) else { continue };
            // Every key is a piece's edge, whose marginal (the diagonal) is set with it.
            let marginal = row.get(&causal).copied().unwrap_or(0.0);
            if marginal == 0.0 {
                continue;
            }
            for (&affected, &pj) in row {
                if affected != causal {
                    let ratio = pj / marginal;
                    let q = if ratio < 0.5 { ratio } else { 0.5 }; // std::min(0.5, ratio)
                    let node2 = (affected.1 != BOUNDARY).then_some(affected.1);
                    self.implied[id].push(ImpliedWeight { node1: affected.0, node2, weight: ((1.0 - q) / q).ln() });
                }
            }
        }
    }

    /// PyMatching's `get_edge_weight_normalising_constant` (before doubling).
    #[cfg(test)]
    pub(crate) fn normalising_constant(&self) -> Result<f64, String> {
        crate::pm_exact::normalising_constant_v2_4(self, NUM_DISTINCT_WEIGHTS)
    }

    /// The decoder for the graph as it is now (PyMatching's `to_mwpm`).
    pub fn decoder(&self) -> Result<UserDecoder, String> {
        Ok(UserDecoder { graph: Arc::new(Graph::try_from_user_graph(self, NeighbourOrder::V2_4, NUM_DISTINCT_WEIGHTS)?) })
    }
}

impl UserDecoder {
    pub fn graph(&self) -> &Arc<Graph> {
        &self.graph
    }

    /// A fresh `Mwpm` over the graph (one per thread).
    pub fn mwpm(&self) -> Mwpm {
        Mwpm::new(self.graph.clone())
    }

    pub fn num_fault_ids(&self) -> usize {
        self.graph.num_observables
    }

    pub fn num_nodes(&self) -> usize {
        self.graph.num_nodes
    }

    /// One shot on `mwpm` (this decoder's, from `mwpm()`), from its detection events (distinct
    /// node indices), into `out`: the fault ids flipped, bit-packed (`ceil(num_fault_ids / 8)`
    /// bytes, zeroed first). Returns the weight.
    pub fn decode_packed_with(&self, mwpm: &mut Mwpm, events: &[u64], out: &mut [u8]) -> Result<f64, String> {
        out.iter_mut().for_each(|b| *b = 0);
        let g = &self.graph;
        if g.num_observables <= 64 {
            let (mask, w) = mwpm.decode_to_mask(events)?;
            for (k, byte) in out.iter_mut().enumerate().take(8) {
                *byte = (mask >> (8 * k)) as u8;
            }
            Ok(w as f64 / g.normalising_constant)
        } else {
            let mut bytes = Vec::new();
            bytes.try_reserve_exact(g.num_observables).map_err(|e| format!("cannot allocate observable output: {e}"))?;
            bytes.resize(g.num_observables, 0u8);
            let w = mwpm.decode_into(events, &mut bytes)?;
            for (k, &b) in bytes.iter().enumerate() {
                out[k / 8] |= (b & 1) << (k % 8);
            }
            Ok(w as f64 / g.normalising_constant)
        }
    }

    /// The `Mwpm` a 2.1 scratch carries for this decoder, made (or remade, for a scratch from
    /// another decoder) on first use.
    fn mwpm_in<'s>(&self, scratch: &'s mut Scratch) -> &'s mut Mwpm {
        let ours = scratch.user_graph_mwpm.as_ref().is_some_and(|m| std::ptr::eq(m.graph(), self.graph.as_ref()));
        if !ours {
            scratch.user_graph_mwpm = Some(Box::new(self.mwpm()));
        }
        scratch.user_graph_mwpm.as_mut().expect("just set")
    }

    /// A workspace for the deprecated 2.1 methods: it carries this decoder's `Mwpm`, one per
    /// thread.
    #[deprecated(since = "2.2.0", note = "use `mwpm()`, one per thread, with `Mwpm`'s decoding methods or `decode_packed_with`")]
    pub fn scratch(&self) -> Scratch {
        let mut s = Scratch::new(&SparseGraph::from_weighted(0, &[]));
        s.user_graph_mwpm = Some(Box::new(self.mwpm()));
        s
    }

    /// One shot from its detection events as sorted distinct node indices, into `out` (the
    /// fault ids flipped, bit-packed, zeroed first; bits past the fault count clear): the
    /// weight. `buf` is no longer used.
    #[deprecated(since = "2.2.0", note = "use `decode_packed_with` on an `Mwpm` from `mwpm()`")]
    pub fn decode_packed(&self, scratch: &mut Scratch, sorted_events: &[u32], buf: &mut Vec<u32>, out: &mut [u8]) -> Result<f64, String> {
        let _ = buf;
        let events: Vec<u64> = sorted_events.iter().map(|&e| u64::from(e)).collect();
        let w = self.decode_packed_with(self.mwpm_in(scratch), &events, out)?;
        let nf = self.num_fault_ids();
        if !nf.is_multiple_of(8) && nf / 8 < out.len() {
            out[nf / 8] &= (1u8 << (nf % 8)) - 1;
        }
        Ok(w)
    }

    /// One shot: the fault ids flipped and the matching's weight (PyMatching's `decode`).
    #[deprecated(since = "2.2.0", note = "use `Mwpm::try_decode` on an `Mwpm` from `mwpm()`")]
    #[allow(deprecated)]
    pub fn decode(&self, scratch: &mut Scratch, events: &[u64]) -> Result<Decoded, String> {
        let (faults, weight) = self.mwpm_in(scratch).try_decode(events)?;
        Ok(Decoded { faults: faults.into_iter().map(|f| f != 0).collect(), weight })
    }

    /// The edges the correction flips, as user endpoints (-1: the boundary), the negative-weight
    /// edges included, in PyMatching's order (its `decode_to_edges_array`).
    #[deprecated(since = "2.2.0", note = "use `Mwpm::try_decode_to_edges` on an `Mwpm` from `mwpm()`")]
    pub fn decode_to_edges(&self, scratch: &mut Scratch, events: &[u64]) -> Result<Vec<(i64, i64)>, String> {
        self.mwpm_in(scratch).try_decode_to_edges(events)
    }

    /// The matched pairs of detection events (-1: the boundary), in PyMatching's order (its
    /// `decode_to_matched_dets_array`). Refused for graphs with negative weights, as PyMatching.
    #[deprecated(since = "2.2.0", note = "use `Mwpm::try_decode_to_match_edges` on an `Mwpm` from `mwpm()`")]
    pub fn decode_to_matched_pairs(&self, scratch: &mut Scratch, events: &[u64]) -> Result<Vec<(i64, i64)>, String> {
        self.mwpm_in(scratch).try_decode_to_match_edges(events)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tied_square() -> UserGraph {
        let mut g = UserGraph::new(4, 2);
        for (u, v, f) in [(0, 1, vec![0]), (1, 2, vec![]), (2, 3, vec![1]), (3, 0, vec![])] {
            g.add_edge(u, v, &f, 1.0, 0.1, Merge::Disallow).unwrap();
        }
        g
    }

    fn prediction(e: &mut Engine, width: usize) -> Result<Vec<u8>, String> {
        let mut out = vec![0; width];
        let weight = e.mwpm.decode_into(&[0, 2], &mut out)?;
        assert_eq!(weight as f64 / e.decoder.graph().normalising_constant, 2.0);
        Ok(out)
    }

    #[test]
    fn cache_calls_own_overlapping_engines_with_one_published_history() {
        use std::sync::mpsc::channel;
        use std::time::Duration;
        // First use races, and the stale-width case when all idle scratch is leased.
        for stale in [false, true] {
            let mut g = tied_square();
            let cache = DecoderCache::default();
            if stale {
                assert_eq!(cache.run(&g, false, |e| prediction(e, 2)).unwrap(), [1, 0]);
                g.ensure_num_fault_ids(70);
            }
            let width = g.num_fault_ids();
            std::thread::scope(|scope| {
                let (entered, entry) = channel();
                let mut release = Vec::new();
                let mut workers = Vec::new();
                for _ in 0..3 {
                    let (go, ready) = channel();
                    release.push(go);
                    let entered = entered.clone();
                    let (g, cache) = (&g, &cache);
                    workers.push(scope.spawn(move || cache.run(g, false, |e| {
                        // Synchronize AFTER production checkout, while owning mutable scratch.
                        // A mutex held across this closure cannot pass this bounded test.
                        entered.send(()).unwrap();
                        ready.recv_timeout(Duration::from_secs(5)).map_err(|e| e.to_string())?;
                        prediction(e, width)
                    })));
                }
                for _ in 0..3 { entry.recv_timeout(Duration::from_secs(5)).unwrap(); }
                for go in release { go.send(()).unwrap(); }
                for worker in workers {
                    let out = worker.join().unwrap().unwrap();
                    assert_eq!(&out[..2], &[1, 0]);
                    assert!(out[2..].iter().all(|&b| b == 0));
                }
            });
        }
    }

    #[test]
    fn old_inflight_call_cannot_undo_search_upgrade() {
        use std::sync::mpsc::channel;
        use std::time::Duration;
        let mut g = tied_square();
        let cache = DecoderCache::default();
        assert_eq!(cache.run(&g, false, |e| prediction(e, 2)).unwrap(), [1, 0]);
        g.ensure_num_fault_ids(70);
        std::thread::scope(|scope| {
            let (entered, entry) = channel();
            let (release, ready) = channel();
            let (g, cache) = (&g, &cache);
            let old = scope.spawn(move || cache.run(g, false, |e| {
                let out = prediction(e, 70)?;
                entered.send(()).unwrap();
                ready.recv_timeout(Duration::from_secs(5)).map_err(|e| e.to_string())?;
                Ok(out)
            }));
            entry.recv_timeout(Duration::from_secs(5)).unwrap();
            assert_eq!(&cache.run(g, true, |e| prediction(e, 70)).unwrap()[..2], &[0, 1]);
            release.send(()).unwrap();
            assert_eq!(&old.join().unwrap().unwrap()[..2], &[1, 0]);
        });
        for _ in 0..8 {
            assert_eq!(&cache.run(&g, false, |e| prediction(e, 70)).unwrap()[..2], &[0, 1]);
        }
    }

    #[test]
    fn failed_or_panicked_scratch_keeps_immutable_stale_build() {
        let mut g = tied_square();
        let cache = DecoderCache::default();
        cache.run(&g, false, |e| prediction(e, 2)).unwrap();
        g.ensure_num_fault_ids(70);
        assert!(cache.run(&g, false, |e| e.mwpm.decode_to_mask(&[0])).is_err());
        assert_eq!(&cache.run(&g, false, |e| prediction(e, 70)).unwrap()[..2], &[1, 0]);
        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _: Result<(), String> = cache.run(&g, false, |e| {
                prediction(e, 70)?;
                panic!("injected after decoder work")
            });
        }));
        assert!(panic.is_err());
        assert_eq!(&cache.run(&g, false, |e| prediction(e, 70)).unwrap()[..2], &[1, 0]);
        // A failed rebuild must not replace the older published graph.
        g.set_implied_weights(0, vec![ImpliedWeight { node1: 0, node2: Some(2), weight: 1.0 }]).unwrap();
        assert!(cache.run(&g, true, |_| Ok(())).is_err());
        assert_eq!(&cache.run(&g, false, |e| prediction(e, 70)).unwrap()[..2], &[1, 0]);
    }

    #[test]
    fn no_op_merges_report_no_change_without_growing_metadata() {
        let mut g = tied_square();
        for how in [Merge::KeepOriginal, Merge::SmallestWeight] {
            assert!(!g.add_edge_changed(0, 1, &[99], 10.0, -1.0, how).unwrap());
            assert_eq!(g.num_fault_ids(), 2);
            assert!(g.all_edges_have_error_probabilities());
        }
        assert!(g.add_edge_changed(0, 1, &[99], 1.0, 0.2, Merge::Replace).unwrap());
        assert_eq!(g.num_fault_ids(), 2); // Reference merges do not extend the width.
        assert_eq!(g.edges[0].fault_ids, [99]);
    }

    fn line(n: usize) -> UserGraph {
        let mut g = UserGraph::new(n, 0);
        for k in 0..n - 1 {
            g.add_edge(k, k + 1, &[k], 1.0, -1.0, Merge::Disallow).unwrap();
        }
        g.add_boundary_edge(0, &[], 1.0, -1.0, Merge::Disallow).unwrap();
        g.add_boundary_edge(n - 1, &[n - 1], 1.0, -1.0, Merge::Disallow).unwrap();
        g
    }

    #[test]
    fn a_repetition_line_decodes() {
        let g = line(5);
        let d = g.decoder().unwrap();
        let mut m = d.mwpm();
        let (faults, w) = m.decode(&[1, 3]);
        assert_eq!(faults, vec![0, 1, 1, 0, 0]);
        assert_eq!(w, 2.0);
        assert!(m.try_decode(&[9]).is_err());
        let mut out = [0u8];
        assert_eq!(d.decode_packed_with(&mut m, &[1, 3], &mut out).unwrap(), 2.0);
        assert_eq!(out, [0b110]);
    }

    #[test]
    fn merges() {
        let mut g = line(3);
        assert!(g.add_edge(0, 1, &[], 5.0, -1.0, Merge::Disallow).is_err());
        g.add_edge(0, 1, &[7], 5.0, -1.0, Merge::SmallestWeight).unwrap();
        assert_eq!(g.edges[0].fault_ids, vec![0]);
        g.add_edge(1, 0, &[7], 0.5, -1.0, Merge::SmallestWeight).unwrap();
        assert_eq!((g.edges[0].weight, g.edges[0].fault_ids.clone()), (0.5, vec![7]));
        g.add_edge(0, 1, &[1], 2.0, 0.1, Merge::Replace).unwrap();
        g.add_edge(0, 1, &[2], 2.0, 0.2, Merge::Independent).unwrap();
        assert!((g.edges[0].error_probability - (0.1 * 0.8 + 0.2 * 0.9)).abs() < 1e-12);
        assert_eq!(g.edges[0].fault_ids, vec![1]);
        // As PyMatching: only a new edge counts its fault ids, not a merge.
        assert_eq!(g.num_fault_ids(), 3);
    }

    #[test]
    #[allow(deprecated)]
    fn the_2_1_methods_still_decode() {
        let g = line(5);
        let d = g.decoder().unwrap();
        let mut s = d.scratch();
        let r = d.decode(&mut s, &[1, 3]).unwrap();
        assert_eq!(r, Decoded { faults: vec![false, true, true, false, false], weight: 2.0 });
        assert!(d.decode(&mut s, &[9]).is_err());
        assert_eq!(d.decode_to_edges(&mut s, &[1, 3]).unwrap(), d.mwpm().decode_to_edges(&[1, 3]));
        assert_eq!(d.decode_to_matched_pairs(&mut s, &[1, 3]).unwrap(), vec![(1, 3)]);
        let (mut buf, mut out) = (Vec::new(), [0xffu8]);
        assert_eq!(d.decode_packed(&mut s, &[1, 3], &mut buf, &mut out).unwrap(), 2.0);
        assert_eq!(out, [0b110]);
        // A scratch from another decoder is remade for this one.
        let mut other = line(3).decoder().unwrap().scratch();
        assert_eq!(d.decode(&mut other, &[1, 3]).unwrap().weight, 2.0);
        // Negative weights, as before: pre-applied, and pairs refused.
        let mut g = UserGraph::new(2, 0);
        g.add_edge(0, 1, &[0], -2.0, -1.0, Merge::Disallow).unwrap();
        g.add_boundary_edge(0, &[1], 3.0, -1.0, Merge::Disallow).unwrap();
        g.add_boundary_edge(1, &[], 3.0, -1.0, Merge::Disallow).unwrap();
        let d = g.decoder().unwrap();
        let mut s = d.scratch();
        assert_eq!(d.decode(&mut s, &[0, 1]).unwrap(), Decoded { faults: vec![true, false], weight: -2.0 });
        assert_eq!(d.decode_to_edges(&mut s, &[0, 1]).unwrap(), vec![(0, 1)]);
        assert!(d.decode_to_matched_pairs(&mut s, &[0, 1]).is_err());
    }

    /// A rare correlated error whose implied weight is the heaviest in the graph.
    const RARE: &str = "error(0.3) D1 D0\nerror(0.000001) D1 D0 ^ D2\nerror(0.3) D2";

    #[test]
    fn a_correlated_load_orders_endpoints_and_implies_weights() {
        let dem = DemProgram::parse(RARE).unwrap();
        let plain = UserGraph::from_dem(&dem).unwrap();
        let g = UserGraph::from_dem_with_correlations(&dem).unwrap();
        assert_eq!((plain.edges[0].node1, plain.edges[0].node2), (1, Some(0)));
        assert_eq!((g.edges[0].node1, g.edges[0].node2), (0, Some(1)));
        assert_eq!((g.edges[1].node1, g.edges[1].node2), (2, None));
        assert!(plain.implied_weights(0).is_empty());
        // PyMatching's joint probabilities: (0, 1) and the boundary edge at 2 fault together at
        // 1e-6; their marginals are each 0.3 combined with it.
        let m = bernoulli_xor(0.3, 0.000001);
        let w = |q: f64| ((1.0 - q) / q).ln();
        assert_eq!(g.implied_weights(0), &[ImpliedWeight { node1: 2, node2: None, weight: w(0.000001 / m) }]);
        let m2 = bernoulli_xor(0.000001, 0.3);
        assert_eq!(g.implied_weights(1), &[ImpliedWeight { node1: 0, node2: Some(1), weight: w(0.000001 / m2) }]);
        // The scale counts the implied weight, the heaviest here.
        assert_eq!(g.normalising_constant().unwrap(), MAX_USER_EDGE_WEIGHT / w(0.000001 / m));
        assert!(plain.normalising_constant().unwrap() > g.normalising_constant().unwrap());
        // An implied probability above 0.5 is held at 0.5: weight 0.
        let g = UserGraph::from_dem_with_correlations(&DemProgram::parse("error(0.1) D0 D1 ^ D2").unwrap()).unwrap();
        assert_eq!(g.implied_weights(1), &[ImpliedWeight { node1: 0, node2: Some(1), weight: 0.0 }]);
    }

    #[test]
    fn a_correlated_load_refuses_as_pymatching() {
        let load = |t: &str| UserGraph::from_dem_with_correlations(&DemProgram::parse(t).unwrap()).map(|g| g.num_edges());
        assert_eq!(load("error(0.6) D0 D1").unwrap_err(), TOO_LIKELY);
        assert_eq!(load("error(0.1) D0 D1 D2").unwrap_err(), UNDECOMPOSED);
        assert_eq!(load("error(0.1) D0 D1 ^ D2 D3 D4").unwrap_err(), UNDECOMPOSED);
        assert_eq!(load("error(0.1) L0 ^ D1").unwrap_err(), UNDETECTABLE);
        assert_eq!(load("error(0.1) D0 ^ L0").unwrap_err(), UNDETECTABLE);
        // The first refused error, in order.
        assert_eq!(load("error(0.6) D0\nerror(0.1) D0 D1 D2").unwrap_err(), TOO_LIKELY);
        // Undetectable alone, or impossible: ignored.
        assert_eq!(load("error(0.1) L0\nerror(0) D0 D1 D2\nerror(0.5) D0 D1").unwrap(), 1);
        // A weight implied for a self-loop, which the matching graph does not hold: loaded, but
        // no decoder is built (PyMatching's `convert_implied_weights`). A self-loop's own
        // implied weights are never converted.
        let g = UserGraph::from_dem_with_correlations(&DemProgram::parse("error(0.1) D0 D0 ^ D1\nerror(0.2) D0").unwrap()).unwrap();
        assert_eq!(g.decoder().unwrap_err(), "Failed to find neighbor.");
        let g = UserGraph::from_dem_with_correlations(&DemProgram::parse("error(0.1) D0 D0 ^ D1 D1\nerror(0.2) D0\nerror(0.2) D1").unwrap()).unwrap();
        assert!(g.decoder().is_ok());
        // Nor for an edge to what has since become a boundary node.
        let mut g = UserGraph::from_dem_with_correlations(&DemProgram::parse("error(0.1) D0 D1 ^ D2\nerror(0.2) D1").unwrap()).unwrap();
        assert!(g.decoder().is_ok());
        g.set_boundary([1].into());
        assert_eq!(g.decoder().unwrap_err(), "Failed to find neighbor.");
    }

    #[test]
    fn implied_weights_must_name_an_edge_and_keep_its_sign() {
        let mut g = line(3);
        g.set_implied_weights(0, vec![ImpliedWeight { node1: 1, node2: None, weight: 0.5 }]).unwrap();
        assert_eq!(g.decoder().unwrap_err(), "Edge rewrite rule refers to non-existent edge (1, 18446744073709551615)");
        g.set_implied_weights(0, vec![ImpliedWeight { node1: 2, node2: None, weight: -0.5 }]).unwrap();
        assert_eq!(g.decoder().unwrap_err(), "Edge weight rewrite rules that change the sign of an edge weight are not currently supported.");
        g.set_implied_weights(0, vec![ImpliedWeight { node1: 2, node2: None, weight: 4.5 }]).unwrap();
        assert_eq!(g.normalising_constant().unwrap(), MAX_USER_EDGE_WEIGHT / 4.5);
        assert!(g.set_implied_weights(9, Vec::new()).is_err());
    }

    #[test]
    fn a_model_keeps_its_targets_in_order() {
        let dem = DemProgram::parse("error(0.1) D2 D0 L1\nerror(0.2) D1 D1\nerror(0.3) D0 ^ D1 D2").unwrap();
        let g = UserGraph::from_dem(&dem).unwrap();
        assert_eq!((g.edges[0].node1, g.edges[0].node2), (2, Some(0)));
        assert_eq!((g.edges[1].node1, g.edges[1].node2), (1, Some(1)));
        assert_eq!((g.edges[2].node1, g.edges[2].node2), (0, None));
        assert_eq!((g.edges[3].node1, g.edges[3].node2), (1, Some(2)));
    }
}

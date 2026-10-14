//! A search decoder for any detector error model: Tesseract's A* over sets of faults (Beni,
//! Higgott and Shutty, Google, arXiv:2503.10988).
//!
//! WHY THIS EXISTS
//! ---------------
//! Matching needs faults that set off at most two detectors; BP-based decoders approximate.
//! Tesseract searches for the most likely set of faults directly. A search state is a set of
//! faults chosen so far and the detectors still lit; its cost is the faults' weight
//! `Σ −ln(p/(1 − p))` plus an admissible estimate of what explaining the lit detectors will
//! cost (each lit detector's cheapest share of a fault through it). States are expanded
//! cheapest first; a state with nothing lit is the answer. To keep the search finite:
//!
//! - a state only grows by faults through its first lit detector in a fixed detector order,
//!   and a fault passed over there stays blocked below it, so each set of faults is reached
//!   one way;
//! - states lighting more than `beam` detectors above the fewest yet seen are dropped, and a
//!   pattern of lit detectors already expanded is not expanded again;
//! - the queue is bounded, and the search is repeated over several detector orders (or
//!   beams), keeping the cheapest answer.
//!
//! With no beam, no queue bound and revisiting allowed it is exact: the most likely fault set
//! (tested against brute force). Skipping revisited patterns, Tesseract's default, ignores
//! that two ways to a pattern may have blocked different faults, so it can miss the optimum;
//! it is much faster. This is
//! Tesseract (quantumlib/tesseract-decoder, src/tesseract.cc and src/common.cc) ported with
//! its arithmetic in its order, so its answers are Tesseract's: equal-cost states leave the
//! queue in the order `std::priority_queue` gives them, which differs between libc++ and
//! libstdc++, so both are implemented (`Flavour`), defaulting to the platform's own; so is
//! the coin that orients its default detector orders. Detector orders are generated as
//! Tesseract generates them, through the platform library's distributions (`cpp_random`), and
//! its sparsified mode (`Sparsify`) is ported too.

use crate::cpp_random::{contracted, Library, Mt19937_64, Normal};
use std::collections::{HashMap, HashSet};

/// Which C++ standard library's tie-breaking to reproduce.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Flavour {
    LibCxx,
    LibStdCxx,
    Msvc,
}

impl Flavour {
    /// The platform's own: libc++ on Apple platforms, Microsoft's on Windows, libstdc++
    /// elsewhere.
    pub fn native() -> Flavour {
        if cfg!(target_vendor = "apple") {
            Flavour::LibCxx
        } else if cfg!(windows) {
            Flavour::Msvc
        } else {
            Flavour::LibStdCxx
        }
    }

    /// The library, for its distributions.
    pub fn library(self) -> Library {
        match self {
            Flavour::LibCxx => Library::LibCxx,
            Flavour::LibStdCxx => Library::LibStdCxx,
            Flavour::Msvc => Library::Msvc,
        }
    }
}

/// How detector orders are generated (Tesseract's `DetectorOrder::Method`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OrderMethod {
    /// Detector index order or its reverse, by a coin.
    Index,
    /// Breadth-first from random roots over the detectors faults join.
    Bfs,
    /// By projection of the coordinates on a random direction.
    Coordinate,
}

#[derive(Clone, Debug)]
pub struct SearchConfig {
    /// `None`: no beam (Tesseract's `INF_DET_BEAM`).
    pub beam: Option<usize>,
    pub beam_climbing: bool,
    pub no_revisit: bool,
    /// `None`: no bound.
    pub queue_limit: Option<usize>,
    pub orders: Vec<Vec<u32>>,
    pub detector_penalty: f64,
    pub merge_errors: bool,
    pub flavour: Flavour,
    /// Tesseract's sparsified mode: `None` off.
    pub sparsify: Option<Sparsify>,
    /// Record the search for Tesseract's visualizer (`Search::take_trace`).
    pub visualize: bool,
    /// Keep Tesseract's verbose log (`Search::take_log`).
    pub verbose: bool,
}

/// Tesseract's sparsified mode: faults with at most `base_degree` detectors are always
/// searched; of those with more (up to `max_degree`, -1 for any), only the
/// `reactivate_limit` (-1: Tesseract's suggestion) sharing most detectors with the shot are.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Sparsify {
    pub base_degree: i64,
    pub max_degree: i64,
    pub reactivate_limit: i64,
}

/// A cost as Tesseract's verbose log prints one (`std::cout` at precision 13).
fn g13(x: f64) -> String {
    crate::dem_program::fmt_g(x, 13)
}

/// `suggest_sparsify_reactivate_limit_capped`.
pub fn suggest_reactivate_limit(num_detectors: usize, base_degree: i64, max_limit: i64) -> Result<i64, String> {
    if base_degree < 0 {
        return Err("sparsify_base_degree must be >= 0.".into());
    }
    if num_detectors == 0 || max_limit <= 0 {
        return Ok(0);
    }
    let exponent = base_degree as f64 - 2.0;
    let max_result = max_limit as f64;
    let log_result = exponent * 4.5f64.ln() - 3.0f64.ln() + (num_detectors as f64).ln();
    if !log_result.is_finite() || log_result >= max_result.ln() {
        return Ok(max_limit);
    }
    let result = log_result.exp();
    if !result.is_finite() {
        return Ok(max_limit);
    }
    let rounded = result.round();
    if rounded >= max_result {
        return Ok(max_limit);
    }
    Ok(rounded as i64)
}

const INF_DET_BEAM: usize = u16::MAX as usize;

/// One retained fault.
#[derive(Clone, Debug)]
struct Fault {
    cost: f64,
    detectors: Vec<u32>,
    observables: u64,
}

/// The decoder: the faults after merging, the detector-to-fault index and the orders.
pub struct Search {
    num_detectors: usize,
    faults: Vec<Fault>,
    /// The model fault each retained fault stands for (the first of a merged group).
    pub fault_to_model: Vec<usize>,
    /// The retained fault each model fault went into (`usize::MAX` for one of probability 0).
    model_to_fault: Vec<usize>,
    /// Per detector, its faults by cheapest share of cost, then index.
    d2e: Vec<Vec<u32>>,
    /// Per fault, the detectors of faults sharing a detector with it, less its own.
    neighbors: Vec<Vec<u32>>,
    config: SearchConfig,
    /// Sparsified mode: the faults always searched, the optional ones, and how many of those
    /// to reactivate.
    mandatory: Vec<u32>,
    optional: Vec<u32>,
    reactivate: usize,
    /// The visualizer's lines, when asked for.
    trace: std::sync::Mutex<Vec<String>>,
    log: std::sync::Mutex<Vec<String>>,
}

#[derive(Clone, Copy, Default)]
struct Tuple {
    blocked: bool,
    count: u32,
}

#[derive(Clone, Copy)]
struct Node {
    cost: f64,
    num_dets: usize,
    depth: usize,
    chain: i64,
}

impl Node {
    /// `Node::operator>`: costlier, or as costly with fewer lit detectors.
    fn gt(&self, other: &Node) -> bool {
        self.cost > other.cost || (self.cost == other.cost && self.num_dets < other.num_dets)
    }
}

/// `std::priority_queue<Node, vector<Node>, greater<Node>>`, with its library's heap moves.
struct Queue {
    heap: Vec<Node>,
    flavour: Flavour,
}

impl Queue {
    /// `comp(a, b)` of a min-queue: a sorts below b.
    fn comp(a: &Node, b: &Node) -> bool {
        a.gt(b)
    }

    fn len(&self) -> usize {
        self.heap.len()
    }

    fn push(&mut self, n: Node) {
        self.heap.push(n);
        let last = self.heap.len() - 1;
        match self.flavour {
            Flavour::LibStdCxx => self.push_up_stdcxx(last, 0, n),
            Flavour::LibCxx => self.sift_up_cxx(self.heap.len()),
            Flavour::Msvc => crate::cpp_random::ms_push_heap_by_index(&mut self.heap, 0, last as isize, 0, n, &mut |a: &Node, b: &Node| Queue::comp(a, b)),
        }
    }

    /// libstdc++'s `__push_heap`.
    fn push_up_stdcxx(&mut self, mut hole: usize, top: usize, value: Node) {
        let h = &mut self.heap;
        let mut parent = hole.wrapping_sub(1) / 2;
        while hole > top && Queue::comp(&h[parent], &value) {
            h[hole] = h[parent];
            hole = parent;
            parent = hole.wrapping_sub(1) / 2;
        }
        h[hole] = value;
    }

    /// libc++'s `__sift_up` over the first `len` entries.
    fn sift_up_cxx(&mut self, len: usize) {
        if len <= 1 {
            return;
        }
        let h = &mut self.heap;
        let mut last = len - 1;
        let mut l = (len - 2) / 2;
        if Queue::comp(&h[l], &h[last]) {
            let t = h[last];
            loop {
                h[last] = h[l];
                last = l;
                if l == 0 {
                    break;
                }
                l = (l - 1) / 2;
                if !Queue::comp(&h[l], &t) {
                    break;
                }
            }
            h[last] = t;
        }
    }

    fn pop(&mut self) -> Option<Node> {
        let len = self.heap.len();
        if len == 0 {
            return None;
        }
        let top = self.heap[0];
        match self.flavour {
            Flavour::Msvc => {
                // pop_heap: the last value into the hole left at the root.
                if len >= 2 {
                    let value = self.heap[len - 1];
                    self.heap[len - 1] = top;
                    crate::cpp_random::ms_pop_heap_hole_by_index(&mut self.heap, 0, 0, (len - 1) as isize, value, &mut |a: &Node, b: &Node| Queue::comp(a, b));
                }
                self.heap.pop();
            }
            Flavour::LibStdCxx => {
                // __pop_heap: the last value goes through __adjust_heap from the root.
                let value = self.heap[len - 1];
                self.heap[len - 1] = top;
                let n = len - 1;
                let h = &mut self.heap;
                let mut hole = 0usize;
                let mut child = 0usize;
                while n > 0 && child < (n - 1) / 2 {
                    child = 2 * (child + 1);
                    if Queue::comp(&h[child], &h[child - 1]) {
                        child -= 1;
                    }
                    h[hole] = h[child];
                    hole = child;
                }
                if n > 0 && n.is_multiple_of(2) && child == (n - 2) / 2 {
                    child = 2 * (child + 1);
                    h[hole] = h[child - 1];
                    hole = child - 1;
                }
                if n > 0 {
                    self.push_up_stdcxx(hole, 0, value);
                }
                self.heap.pop();
            }
            Flavour::LibCxx => {
                if len > 1 {
                    // __floyd_sift_down, then the last value into the hole and sifted up.
                    let h = &mut self.heap;
                    let mut hole = 0usize;
                    let mut child = 0usize;
                    loop {
                        let mut ci = 2 * child + 1;
                        if ci + 1 < len && Queue::comp(&h[ci], &h[ci + 1]) {
                            ci += 1;
                        }
                        child = ci;
                        h[hole] = h[child];
                        hole = child;
                        if child > (len - 2) / 2 {
                            break;
                        }
                    }
                    let last = len - 1;
                    if hole == last {
                        h[hole] = top;
                    } else {
                        h[hole] = h[last];
                        h[last] = top;
                        self.sift_up_cxx(hole + 1);
                    }
                }
                self.heap.pop();
            }
        }
        Some(top)
    }
}

/// `merge_weights`: the cost of either of two independent faults with these costs.
fn merge_weights(a: f64, b: f64) -> f64 {
    let sgn = 1f64.copysign(a) * 1f64.copysign(b);
    let signed_min = sgn * a.abs().min(b.abs());
    signed_min + (1.0 + (-(a + b).abs()).exp()).ln() - (1.0 + (-(a - b).abs()).exp()).ln()
}

fn cost_of(p: f64) -> f64 {
    -(p / (1.0 - p)).ln()
}

fn probability_of(cost: f64) -> f64 {
    1.0 / (1.0 + cost.exp())
}

/// Detector orders as Tesseract generates them (`build_det_orders`): order `i` from
/// `std::mt19937_64(seed + i)`, through the flavour's library's distributions.
pub fn generated_orders(dem: &crate::dem::Dem, count: usize, method: OrderMethod, seed: u64, flavour: Flavour) -> Vec<Vec<u32>> {
    let n = dem.num_detectors;
    let lib = flavour.library();
    let graph = if method == OrderMethod::Bfs { detector_graph(dem) } else { Vec::new() };
    (0..count as u64)
        .map(|i| {
            let mut rng = Mt19937_64::new(seed.wrapping_add(i));
            match method {
                OrderMethod::Index => {
                    // uniform_int_distribution<int>(0, 1)
                    if lib.uniform(&mut rng, 0, 1) == 1 { (0..n as u32).rev().collect() } else { (0..n as u32).collect() }
                }
                OrderMethod::Bfs => bfs_order(&graph, &mut rng, lib),
                OrderMethod::Coordinate => coordinate_order(dem, &mut rng, lib),
            }
        })
        .collect()
}

/// `build_detector_graph`: each detector's neighbours through faults of nonzero probability,
/// sorted.
pub fn detector_graph(dem: &crate::dem::Dem) -> Vec<Vec<u32>> {
    let mut graph: Vec<Vec<u32>> = vec![Vec::new(); dem.num_detectors];
    for m in dem.mechanisms.iter().filter(|m| m.p > 0.0) {
        for (i, &a) in m.detectors.iter().enumerate() {
            for &b in &m.detectors[i + 1..] {
                graph[a as usize].push(b);
                graph[b as usize].push(a);
            }
        }
    }
    for g in &mut graph {
        g.sort_unstable();
        g.dedup();
    }
    graph
}

fn bfs_order(graph: &[Vec<u32>], rng: &mut Mt19937_64, lib: Library) -> Vec<u32> {
    let n = graph.len();
    let mut order = Vec::with_capacity(n);
    let mut visited = vec![false; n];
    let mut unvisited: Vec<u32> = (0..n as u32).collect();
    let mut position: Vec<usize> = (0..n).collect();
    let mut mark = |d: u32, visited: &mut Vec<bool>, unvisited: &mut Vec<u32>| {
        visited[d as usize] = true;
        let p = position[d as usize];
        let last = *unvisited.last().unwrap();
        unvisited[p] = last;
        position[last as usize] = p;
        unvisited.pop();
    };
    let mut queue = std::collections::VecDeque::new();
    while !unvisited.is_empty() {
        let start = unvisited[lib.uniform(rng, 0, unvisited.len() as u64 - 1) as usize];
        mark(start, &mut visited, &mut unvisited);
        queue.push_back(start);
        order.push(start);
        while let Some(cur) = queue.pop_front() {
            let mut neigh = graph[cur as usize].clone();
            lib.shuffle(&mut neigh, rng);
            for nb in neigh {
                if !visited[nb as usize] {
                    mark(nb, &mut visited, &mut unvisited);
                    queue.push_back(nb);
                    order.push(nb);
                }
            }
        }
    }
    order
}

fn coordinate_order(dem: &crate::dem::Dem, rng: &mut Mt19937_64, lib: Library) -> Vec<u32> {
    let n = dem.num_detectors;
    let coords = |d: usize| dem.detector_coords.get(d).map(Vec::as_slice).unwrap_or(&[]);
    let dims = (0..n).map(|d| coords(d).len()).max().unwrap_or(0);
    if dims == 0 {
        return (0..n as u32).collect();
    }
    let mut normal = Normal::default();
    let direction: Vec<f64> = (0..dims).map(|_| normal.sample(lib, rng)).collect();
    let dot: Vec<f64> = (0..n).map(|d| coords(d).iter().zip(&direction).fold(0.0, |acc, (a, b)| contracted(*a, *b, acc))).collect();
    let mut with: Vec<u32> = (0..n as u32).filter(|&d| !coords(d as usize).is_empty()).collect();
    // std::stable_sort, largest projection first.
    with.sort_by(|&a, &b| {
        let (x, y) = (dot[a as usize], dot[b as usize]);
        if x > y {
            std::cmp::Ordering::Less
        } else if y > x {
            std::cmp::Ordering::Greater
        } else {
            std::cmp::Ordering::Equal
        }
    });
    with.extend((0..n as u32).filter(|&d| coords(d as usize).is_empty()));
    with
}

/// One search's result: the model faults found (by model index) and whether the search gave
/// up (Tesseract's low-confidence flag).
#[derive(Clone, Debug, Default)]
pub struct Found {
    pub faults: Vec<usize>,
    pub low_confidence: bool,
}

impl Search {
    /// The decoder for a flat model with the given configuration (`orders` already
    /// generated; at least one).
    pub fn new(dem: &crate::dem::Dem, config: SearchConfig) -> Result<Search, String> {
        if config.orders.is_empty() {
            return Err("at least one detector order is needed".into());
        }
        let n = dem.num_detectors;
        for (k, order) in config.orders.iter().enumerate() {
            let mut seen = vec![false; n];
            if order.len() != n {
                return Err(format!("detector order {k} has {} detectors, but the model has {n}", order.len()));
            }
            for &d in order {
                if d as usize >= n || std::mem::replace(&mut seen[d as usize], true) {
                    return Err(format!("detector order {k} repeats or exceeds detector {d}"));
                }
            }
        }
        // Merge faults with the same detectors and observables (in first-seen order), then
        // drop those of probability 0, each step through a probability as Tesseract writes
        // its intermediate models.
        let mut model_to_merged = Vec::with_capacity(dem.mechanisms.len());
        // (cost, or the probability itself where nothing is merged; detectors; observables).
        let mut merged: Vec<(f64, Vec<u32>, u64)> = Vec::new();
        let mut by_symptom: HashMap<(Vec<u32>, u64), usize> = HashMap::new();
        for m in &dem.mechanisms {
            if !(0.0..=1.0).contains(&m.p) {
                return Err(format!("a fault has probability {}, outside [0, 1]", m.p));
            }
            let symptom = (m.detectors.clone(), m.observables);
            if config.merge_errors {
                let c = cost_of(m.p);
                if let Some(&k) = by_symptom.get(&symptom) {
                    merged[k].0 = merge_weights(c, merged[k].0);
                    model_to_merged.push(k);
                    continue;
                }
                by_symptom.insert(symptom.clone(), merged.len());
                model_to_merged.push(merged.len());
                merged.push((c, symptom.0, symptom.1));
            } else {
                model_to_merged.push(merged.len());
                merged.push((m.p, symptom.0, symptom.1));
            }
        }
        let mut faults = Vec::new();
        let mut merged_to_fault = vec![usize::MAX; merged.len()];
        for (k, (c, dets, obs)) in merged.into_iter().enumerate() {
            // The merged model stores probabilities: written back and read again.
            let p = if config.merge_errors { probability_of(c) } else { c };
            if p > 0.0 {
                merged_to_fault[k] = faults.len();
                faults.push(Fault { cost: cost_of(p), detectors: dets, observables: obs });
            }
        }
        let mut fault_to_model = vec![usize::MAX; faults.len()];
        let model_to_fault: Vec<usize> = model_to_merged.iter().map(|&k| merged_to_fault[k]).collect();
        for (model, &f) in model_to_fault.iter().enumerate().rev() {
            if f != usize::MAX {
                fault_to_model[f] = model;
            }
        }
        Self::assemble(n, faults, fault_to_model, model_to_fault, config)
    }

    /// A search over faults given directly (cost, detectors, observables), none merged or
    /// dropped; the faults are their own model faults.
    pub fn from_faults(num_detectors: usize, faults: Vec<(f64, Vec<u32>, u64)>, config: SearchConfig) -> Result<Search, String> {
        let faults: Vec<Fault> = faults
            .into_iter()
            .map(|(cost, mut detectors, observables)| {
                detectors.sort_unstable();
                Fault { cost, detectors, observables }
            })
            .collect();
        if let Some(f) = faults.iter().find(|f| f.detectors.iter().any(|&d| d as usize >= num_detectors)) {
            return Err(format!("a fault sets off detector {:?}, beyond {num_detectors}", f.detectors.last()));
        }
        let ids: Vec<usize> = (0..faults.len()).collect();
        Self::assemble(num_detectors, faults, ids.clone(), ids, config)
    }

    fn assemble(n: usize, faults: Vec<Fault>, fault_to_model: Vec<usize>, model_to_fault: Vec<usize>, config: SearchConfig) -> Result<Search, String> {
        let mut d2e: Vec<Vec<u32>> = vec![Vec::new(); n];
        for (ei, f) in faults.iter().enumerate() {
            for &d in &f.detectors {
                d2e[d as usize].push(ei as u32);
            }
        }
        let min_cost: Vec<f64> = faults.iter().map(|f| if f.detectors.is_empty() { f.cost } else { f.cost / f.detectors.len() as f64 }).collect();
        for list in &mut d2e {
            list.sort_by(|&a, &b| min_cost[a as usize].partial_cmp(&min_cost[b as usize]).unwrap_or(std::cmp::Ordering::Equal).then(a.cmp(&b)));
        }
        let mut neighbors = Vec::with_capacity(faults.len());
        let mut mark = vec![false; n];
        for f in &faults {
            let mut set = Vec::new();
            for &d in &f.detectors {
                for &oe in &d2e[d as usize] {
                    for &od in &faults[oe as usize].detectors {
                        if !mark[od as usize] {
                            mark[od as usize] = true;
                            set.push(od);
                        }
                    }
                }
            }
            for &d in &set {
                mark[d as usize] = false;
            }
            set.retain(|d| !f.detectors.contains(d));
            set.sort_unstable();
            neighbors.push(set);
        }
        let (mut mandatory, mut optional, mut reactivate) = (Vec::new(), Vec::new(), 0usize);
        if let Some(sp) = config.sparsify {
            if sp.base_degree <= 0 {
                return Err("sparsify_base_degree must be > 0 when sparsify_errors is enabled.".into());
            }
            if sp.max_degree < -1 {
                return Err("sparsify_max_degree must be >= -1.".into());
            }
            if sp.reactivate_limit < -1 {
                return Err("sparsify_reactivate_limit must be >= -1.".into());
            }
            if sp.max_degree >= 0 && sp.max_degree < sp.base_degree {
                return Err("sparsify_max_degree must be >= sparsify_base_degree.".into());
            }
            let limit = if sp.reactivate_limit == -1 { suggest_reactivate_limit(n, sp.base_degree, faults.len().min(i32::MAX as usize) as i64)? } else { sp.reactivate_limit };
            reactivate = limit as usize;
            for (ei, f) in faults.iter().enumerate() {
                let degree = f.detectors.len() as i64;
                if degree <= sp.base_degree {
                    mandatory.push(ei as u32);
                } else if sp.max_degree == -1 || degree <= sp.max_degree {
                    optional.push(ei as u32);
                }
            }
        }
        Ok(Search { num_detectors: n, faults, fault_to_model, model_to_fault, d2e, neighbors, config, mandatory, optional, reactivate, trace: std::sync::Mutex::new(Vec::new()), log: std::sync::Mutex::new(Vec::new()) })
    }

    /// `get_detcost`: detector `d`'s cheapest share of an unblocked fault through it.
    fn detcost(&self, d: u32, tuples: &[Tuple], d2e: &[Vec<u32>]) -> f64 {
        let mut min_cost = f64::INFINITY;
        let mut min_count = u32::MAX as f64;
        for &ei in &d2e[d as usize] {
            let f = &self.faults[ei as usize];
            if f.cost * min_count >= min_cost * f.detectors.len() as f64 {
                break;
            }
            let t = tuples[ei as usize];
            if !t.blocked && f.cost * min_count < min_cost * t.count as f64 {
                min_cost = f.cost;
                min_count = t.count as f64;
            }
        }
        min_cost / min_count + self.config.detector_penalty
    }

    /// `decode_to_errors`: the cheapest answer over the orders (or beams), as model faults.
    pub fn decode(&self, detections: &[u32]) -> Found {
        if detections.is_empty() {
            return Found::default();
        }
        let mut best: Option<Vec<usize>> = None;
        let mut best_cost = f64::MAX;
        let orders = self.config.orders.len();
        let beam = self.config.beam.unwrap_or(INF_DET_BEAM);
        let sparse = self.sparse_d2e(detections);
        let d2e = sparse.as_deref().unwrap_or(&self.d2e);
        let mut run = |order: usize, beam: usize| {
            let found = self.search(detections, order, beam, d2e);
            let cost: f64 = found.faults.iter().map(|&m| self.cost_of_model_fault(m)).sum();
            let (low, obs) = (found.low_confidence, self.observables(&found.faults));
            if !found.low_confidence && cost < best_cost {
                best = Some(found.faults);
                best_cost = cost;
            }
            if self.config.verbose {
                let mask: Vec<String> = (0..64).filter(|k| obs >> k & 1 == 1).map(|k| k.to_string()).collect();
                self.say(format!(
                    "for detector_order {order} beam {} got low confidence {} and cost {} and obs_mask [{}]. Best cost so far: {}",
                    beam as i64, low as u8, g13(cost), mask.join(", "), g13(best_cost)
                ));
            }
        };
        if self.config.beam_climbing {
            let (mut b, mut o) = (0, 0);
            for _ in 0..(beam + 1).max(orders) {
                run(o, b);
                b = (b + 1) % (beam + 1);
                o = (o + 1) % orders;
            }
        } else {
            for o in 0..orders {
                run(o, beam);
            }
        }
        Found { low_confidence: best_cost == f64::MAX, faults: best.unwrap_or_default() }
    }

    /// `decode_result` with error probabilities changed for this decode only (Tesseract's
    /// `ErrorProbabilityUpdate`s, by model index; a later update of one fault wins): the
    /// answer and its cost under the changed probabilities. The search's own probabilities
    /// (and its detectors' fault orders) are restored before it returns.
    pub fn decode_with_probabilities(&mut self, detections: &[u32], updates: &[(usize, f64)]) -> Result<(Found, f64), String> {
        let mut retained = Vec::with_capacity(updates.len());
        for &(model, p) in updates {
            if !p.is_finite() || p <= 0.0 || p >= 1.0 {
                return Err("Error probability must be finite and between 0 and 1.".into());
            }
            let n = self.model_to_fault.len();
            let f = *self.model_to_fault.get(model).ok_or_else(|| format!("DEM error index {model} is out of range for {n} DEM errors."))?;
            if f == usize::MAX {
                return Err(format!("DEM error index {model} was removed from the decoder."));
            }
            retained.push(f);
        }
        let mut previous: Vec<(usize, f64)> = Vec::new();
        for &f in &retained {
            if !previous.iter().any(|&(g, _)| g == f) {
                previous.push((f, self.faults[f].cost));
            }
        }
        let mut affected = vec![false; self.num_detectors];
        let mut detectors = Vec::new();
        for &(f, _) in &previous {
            for &d in &self.faults[f].detectors {
                if !std::mem::replace(&mut affected[d as usize], true) {
                    detectors.push(d);
                }
            }
        }
        for (&f, &(_, p)) in retained.iter().zip(updates) {
            self.faults[f].cost = cost_of(p);
        }
        self.resort(&detectors);
        let found = self.decode(detections);
        let cost: f64 = found.faults.iter().map(|&m| self.cost_of_model_fault(m)).sum();
        for &(f, c) in &previous {
            self.faults[f].cost = c;
        }
        self.resort(&detectors);
        Ok((found, cost))
    }

    /// Re-sorts these detectors' faults by cost per detector (`update_internal_costs`).
    fn resort(&mut self, detectors: &[u32]) {
        let faults = &self.faults;
        let min_cost = |e: u32| {
            let f = &faults[e as usize];
            if f.detectors.is_empty() { f.cost } else { f.cost / f.detectors.len() as f64 }
        };
        for &d in detectors {
            self.d2e[d as usize].sort_by(|&a, &b| min_cost(a).partial_cmp(&min_cost(b)).unwrap_or(std::cmp::Ordering::Equal).then(a.cmp(&b)));
        }
    }

    /// One search, with detector order `order` and beam `beam` (Tesseract's
    /// `decode_to_errors(detections, det_order, det_beam)`).
    pub fn decode_once(&self, detections: &[u32], order: usize, beam: usize) -> Found {
        let sparse = self.sparse_d2e(detections);
        self.search(detections, order, beam, sparse.as_deref().unwrap_or(&self.d2e))
    }

    /// `build_sparse_d2e`: in sparsified mode, each detector's faults that are searched for
    /// this shot.
    fn sparse_d2e(&self, detections: &[u32]) -> Option<Vec<Vec<u32>>> {
        self.config.sparsify?;
        let mut lit = vec![false; self.num_detectors];
        for &d in detections {
            if (d as usize) < self.num_detectors {
                lit[d as usize] = true;
            }
        }
        let mut active = vec![false; self.faults.len()];
        for &ei in &self.mandatory {
            active[ei as usize] = true;
        }
        // (fault, overlap, degree, cost), most overlap first, then fewest detectors, cheapest,
        // lowest index: a total order, so any sort gives Tesseract's.
        let mut candidates: Vec<(u32, usize, usize, f64)> = Vec::new();
        for &ei in &self.optional {
            let f = &self.faults[ei as usize];
            let overlap = f.detectors.iter().filter(|&&d| lit[d as usize]).count();
            if overlap > 0 {
                candidates.push((ei, overlap, f.detectors.len(), f.cost));
            }
        }
        candidates.sort_by(|a, b| b.1.cmp(&a.1).then(a.2.cmp(&b.2)).then(a.3.partial_cmp(&b.3).unwrap_or(std::cmp::Ordering::Equal)).then(a.0.cmp(&b.0)));
        for c in candidates.iter().take(self.reactivate) {
            active[c.0 as usize] = true;
        }
        Some(self.d2e.iter().map(|list| list.iter().copied().filter(|&ei| active[ei as usize]).collect()).collect())
    }

    /// The visualizer's lines recorded since the last call.
    pub fn take_trace(&self) -> Vec<String> {
        std::mem::take(&mut *self.trace.lock().unwrap_or_else(|e| e.into_inner()))
    }

    /// The verbose log's lines since the last call, as Tesseract prints them (its costs at the
    /// precision of 13 digits its search sets).
    pub fn take_log(&self) -> Vec<String> {
        std::mem::take(&mut *self.log.lock().unwrap_or_else(|e| e.into_inner()))
    }

    fn say(&self, line: String) {
        self.log.lock().unwrap_or_else(|e| e.into_inner()).push(line);
    }

    fn chain_text(&self, arena: &[(u32, u32, i64)], chain: i64, dets: &[u64]) -> (String, String) {
        let mut errors = String::from("activated_errors = ");
        let mut walker = chain;
        while walker != -1 {
            let (ei, _, parent) = arena[walker as usize];
            errors.push_str(&format!("{ei}, "));
            walker = parent;
        }
        let mut lit = String::from("activated_detectors = ");
        for d in 0..self.num_detectors {
            if dets[d / 64] >> (d % 64) & 1 == 1 {
                lit.push_str(&format!("{d}, "));
            }
        }
        (errors, lit)
    }

    /// `add_activated_errors` and `add_activated_detectors` for a node of the search.
    fn record(&self, arena: &[(u32, u32, i64)], chain: i64, dets: &[u64]) {
        let (errors, lit) = self.chain_text(arena, chain, dets);
        let mut t = self.trace.lock().unwrap_or_else(|e| e.into_inner());
        t.push(errors);
        t.push(lit);
    }

    /// The faults the search works with, after merging and dropping those of probability 0:
    /// (cost, detectors, observables).
    pub fn retained(&self) -> impl Iterator<Item = (f64, &[u32], u64)> + '_ {
        self.faults.iter().map(|f| (f.cost, f.detectors.as_slice(), f.observables))
    }

    /// The retained fault each model fault went into (`usize::MAX` for one dropped).
    pub fn model_to_fault(&self) -> &[usize] {
        &self.model_to_fault
    }

    /// The cost of the retained fault a model fault went into.
    fn cost_of_model_fault(&self, model: usize) -> f64 {
        self.faults.get(self.model_to_fault[model]).map_or(0.0, |f| f.cost)
    }

    /// The observables a set of model faults flips.
    pub fn observables(&self, found: &[usize]) -> u64 {
        found.iter().filter_map(|&m| self.faults.get(self.model_to_fault[m])).fold(0, |o, f| o ^ f.observables)
    }

    /// The total cost of a set of model faults.
    pub fn cost(&self, found: &[usize]) -> f64 {
        found.iter().map(|&m| self.cost_of_model_fault(m)).sum()
    }

    /// `decode_to_errors_with_graph`: one A* search.
    fn search(&self, detections: &[u32], order_index: usize, beam: usize, d2e: &[Vec<u32>]) -> Found {
        let n = self.num_detectors;
        let ne = self.faults.len();
        let order = &self.config.orders[order_index];
        let words = n.div_ceil(64);
        let flip = |bits: &mut [u64], d: u32| bits[d as usize / 64] ^= 1 << (d % 64);
        let get = |bits: &[u64], d: u32| bits[d as usize / 64] >> (d % 64) & 1 == 1;

        // (fault, the lowest-ordered lit detector it was chosen at, parent).
        let mut arena: Vec<(u32, u32, i64)> = Vec::new();
        let mut queue = Queue { heap: Vec::new(), flavour: self.config.flavour };
        let mut visited: HashMap<usize, HashSet<Vec<u64>>> = HashMap::new();
        let mut initial = vec![0u64; words];
        let mut initial_tuples = vec![Tuple::default(); ne];
        for &d in detections {
            flip(&mut initial, d);
            for &ei in &d2e[d as usize] {
                initial_tuples[ei as usize].count += 1;
            }
        }
        let mut initial_cost = 0.0;
        for &d in detections {
            initial_cost += self.detcost(d, &initial_tuples, d2e);
        }
        if initial_cost == f64::INFINITY {
            return Found { faults: Vec::new(), low_confidence: true };
        }
        let mut min_num_dets = detections.len();
        // Tesseract's size_t arithmetic: a beam given as a negative int wraps.
        let mut max_num_dets = min_num_dets.wrapping_add(beam);
        queue.push(Node { cost: initial_cost, num_dets: min_num_dets, depth: 0, chain: -1 });
        let mut pushed = 1usize;
        let limit = self.config.queue_limit.unwrap_or(usize::MAX);

        while let Some(node) = queue.pop() {
            if node.num_dets > max_num_dets {
                continue;
            }
            let mut dets = initial.clone();
            let mut tuples = vec![Tuple::default(); ne];
            let mut walker = node.chain;
            while walker != -1 {
                let (ei, min_det, parent) = arena[walker as usize];
                for &oe in &d2e[min_det as usize] {
                    tuples[oe as usize].blocked = true;
                    if oe == ei {
                        break;
                    }
                }
                for &d in &self.faults[ei as usize].detectors {
                    flip(&mut dets, d);
                }
                walker = parent;
            }
            if node.num_dets == 0 {
                if self.config.visualize {
                    self.record(&arena, node.chain, &dets);
                }
                if self.config.verbose {
                    let (errors, lit) = self.chain_text(&arena, node.chain, &dets);
                    self.say(errors);
                    self.say(lit);
                    self.say(format!("Decoding complete. Cost: {} num_pq_pushed = {pushed}", g13(node.cost)));
                }
                let mut faults = vec![0usize; node.depth];
                let mut walker = node.chain;
                for i in 0..node.depth {
                    let (ei, _, parent) = arena[walker as usize];
                    faults[node.depth - 1 - i] = self.fault_to_model[ei as usize];
                    walker = parent;
                }
                return Found { faults, low_confidence: false };
            }
            if self.config.no_revisit && !visited.entry(node.num_dets).or_default().insert(dets.clone()) {
                continue;
            }
            if self.config.visualize {
                self.record(&arena, node.chain, &dets);
            }
            if self.config.verbose {
                self.say(format!("len(pq) = {} num_pq_pushed = {pushed}", queue.len()));
                self.say(format!("num_dets = {} max_num_dets = {max_num_dets} cost = {}", node.num_dets, g13(node.cost)));
                let (errors, lit) = self.chain_text(&arena, node.chain, &dets);
                self.say(errors);
                self.say(lit);
            }
            if node.num_dets < min_num_dets {
                min_num_dets = node.num_dets;
                if self.config.no_revisit {
                    let from = min_num_dets.wrapping_add(beam).wrapping_add(1);
                    for (_, s) in visited.iter_mut().filter(|(&i, _)| i >= from && i <= max_num_dets) {
                        s.clear();
                    }
                }
                max_num_dets = max_num_dets.min(min_num_dets.wrapping_add(beam));
            }
            for d in 0..n as u32 {
                if get(&dets, d) {
                    for &ei in &d2e[d as usize] {
                        tuples[ei as usize].count += 1;
                    }
                }
            }
            let mut next_tuples = tuples.clone();
            let min_detector = *order.iter().find(|&&d| get(&dets, d)).expect("a lit detector");
            let mut prev: Option<u32> = None;
            let mut cache = vec![-1.0f64; n];
            for &ei in &d2e[min_detector as usize] {
                if tuples[ei as usize].blocked {
                    continue;
                }
                if let Some(pe) = prev {
                    for &d in &self.faults[pe as usize].detectors {
                        let lit = get(&dets, d);
                        for &oe in &d2e[d as usize] {
                            let c = &mut next_tuples[oe as usize].count;
                            *c = if lit { c.wrapping_add(1) } else { c.wrapping_sub(1) };
                        }
                    }
                }
                prev = Some(ei);
                let mut next_dets = dets.clone();
                next_tuples[ei as usize].blocked = true;
                let mut next_cost = node.cost + self.faults[ei as usize].cost;
                let mut next_num = node.num_dets;
                for &d in &self.faults[ei as usize].detectors {
                    flip(&mut next_dets, d);
                    let lit = get(&next_dets, d);
                    if lit {
                        next_num += 1;
                    } else {
                        next_num -= 1;
                    }
                    for &oe in &d2e[d as usize] {
                        let c = &mut next_tuples[oe as usize].count;
                        *c = if lit { c.wrapping_add(1) } else { c.wrapping_sub(1) };
                    }
                }
                if next_num > max_num_dets {
                    continue;
                }
                if self.config.no_revisit && visited.get(&next_num).is_some_and(|s| s.contains(&next_dets)) {
                    continue;
                }
                for &d in &self.faults[ei as usize].detectors {
                    if get(&dets, d) {
                        if cache[d as usize] == -1.0 {
                            cache[d as usize] = self.detcost(d, &tuples, d2e);
                        }
                        next_cost -= cache[d as usize];
                    } else {
                        next_cost += self.detcost(d, &next_tuples, d2e);
                    }
                }
                for &od in &self.neighbors[ei as usize] {
                    if !get(&dets, od) || !get(&next_dets, od) {
                        continue;
                    }
                    if cache[od as usize] == -1.0 {
                        cache[od as usize] = self.detcost(od, &tuples, d2e);
                    }
                    next_cost -= cache[od as usize];
                    next_cost += self.detcost(od, &next_tuples, d2e);
                }
                if next_cost == f64::INFINITY {
                    continue;
                }
                arena.push((ei, min_detector, node.chain));
                queue.push(Node { cost: next_cost, num_dets: next_num, depth: node.depth + 1, chain: arena.len() as i64 - 1 });
                pushed += 1;
                if pushed > limit {
                    if self.config.verbose {
                        self.say("setting low confidence flag".into());
                    }
                    return Found { faults: Vec::new(), low_confidence: true };
                }
            }
        }
        if self.config.verbose {
            self.say("Decoding failed to converge within beam limit.".into());
        }
        Found { faults: Vec::new(), low_confidence: true }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::surface_code::Xorshift;

    /// `std::mt19937_64`'s 10000th output for the default seed, as the standard requires.
    #[test]
    fn mt19937_64_is_the_standards() {
        let mut r = Mt19937_64::new(5489);
        for _ in 0..9999 {
            r.next_u64();
        }
        assert_eq!(r.next_u64(), 9981545732273789042);
    }

    /// Both heap flavours pop in priority order, whatever the pushes.
    #[test]
    fn the_queues_are_priority_queues() {
        let mut rng = Xorshift::new(4);
        for flavour in [Flavour::LibCxx, Flavour::LibStdCxx, Flavour::Msvc] {
            let mut q = Queue { heap: Vec::new(), flavour };
            for step in 0..5000 {
                if step % 3 == 2 {
                    let top = q.pop().unwrap();
                    assert!(q.heap.iter().all(|n| !top.gt(n)), "{flavour:?}");
                } else {
                    let cost = (rng.next_u64() % 20) as f64;
                    q.push(Node { cost, num_dets: (rng.next_u64() % 4) as usize, depth: 0, chain: step });
                }
            }
            let mut rest = Vec::new();
            while let Some(n) = q.pop() {
                rest.push(n);
            }
            assert!(rest.windows(2).all(|w| !w[0].gt(&w[1])), "{flavour:?}");
        }
    }

    fn model(columns: &[(f64, Vec<u32>, u64)], n: usize) -> crate::dem::Dem {
        crate::dem::Dem {
            num_detectors: n,
            num_observables: 1,
            mechanisms: columns.iter().map(|(p, d, o)| crate::dem::Mechanism { p: *p, detectors: d.clone(), observables: *o, pieces: Vec::new(), tag: String::new() }).collect(),
            ..Default::default()
        }
    }

    /// With no beam, no queue bound and revisits allowed, the search finds the most likely
    /// fault set: brute force over every subset of faults on small random models. (With
    /// revisits skipped, Tesseract's default, it does not always: trial 8 here finds 5.52
    /// where 5.28 exists.)
    #[test]
    fn unlimited_search_is_the_most_likely_error() {
        let mut rng = Xorshift::new(31);
        for trial in 0..60 {
            let (n, m) = (6, 11);
            let cols: Vec<(f64, Vec<u32>, u64)> = (0..m)
                .map(|_| {
                    let mut d: Vec<u32> = (0..n as u32).filter(|_| rng.next_f64() < 0.3).collect();
                    if d.is_empty() {
                        d.push((rng.next_u64() % n as u64) as u32);
                    }
                    (0.01 + 0.2 * rng.next_f64(), d, rng.next_u64() & 1)
                })
                .collect();
            let dem = model(&cols, n);
            for flavour in [Flavour::LibCxx, Flavour::LibStdCxx] {
                // Unmerged, so each fault is the brute force's own (merging two faults of one
                // symptom correctly makes either one more likely than both).
                let config = SearchConfig { beam: None, beam_climbing: false, no_revisit: false, queue_limit: None, orders: vec![(0..n as u32).collect()], detector_penalty: 0.0, merge_errors: false, flavour, sparsify: None, visualize: false, verbose: false };
                let s = Search::new(&dem, config).unwrap();
                for pattern in 1u32..1 << n {
                    let dets: Vec<u32> = (0..n as u32).filter(|&d| pattern >> d & 1 == 1).collect();
                    let mut best = f64::INFINITY;
                    for subset in 0u32..1 << m {
                        let mut syn = 0u32;
                        let mut cost = 0.0;
                        for (k, (p, d, _)) in cols.iter().enumerate() {
                            if subset >> k & 1 == 1 {
                                for &x in d {
                                    syn ^= 1 << x;
                                }
                                cost += -(p / (1.0 - p)).ln();
                            }
                        }
                        if syn == pattern {
                            best = best.min(cost);
                        }
                    }
                    let found = s.decode(&dets);
                    if best.is_infinite() {
                        assert!(found.low_confidence, "trial {trial}");
                        continue;
                    }
                    assert!(!found.low_confidence, "trial {trial} {dets:?}");
                    let mut syn = 0u32;
                    for &k in &found.faults {
                        for &x in &cols[k].1 {
                            syn ^= 1 << x;
                        }
                    }
                    assert_eq!(syn, pattern);
                    assert!((s.cost(&found.faults) - best).abs() < 1e-9, "trial {trial}: {} vs {best}", s.cost(&found.faults));
                }
            }
        }
    }
}

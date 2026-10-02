use super::{probability, BitTable, DetectorErrorModel, Error, Result};
use crate::batch::{belief_shots, bposd_shots, match_shots, window_info, window_shots};
use crate::dem_decoder::{DecodeError, DemDecoder};
use crate::sparse::{Correlations, Scratch, SparseGraph};
use crate::window::{Mode, Model, WindowDecoder};

/// A decoder's answer for one shot.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub struct Prediction {
    /// The observables predicted flipped, observable `k` in bit `k`.
    pub observables: u64,
    /// The correction's weight (Σ ln((1 − p)/p) over its edges), where the decoder matched.
    pub weight: Option<f64>,
    /// Whether belief propagation converged on its own, where the decoder ran it.
    pub bp_converged: Option<bool>,
}

impl Prediction {
    /// Whether observable `k` is predicted flipped.
    pub fn flips(&self, k: usize) -> bool {
        k < 64 && (self.observables >> k) & 1 == 1
    }
}

fn decode_error(e: DecodeError) -> Error {
    Error::new(match e {
        DecodeError::Unmatchable => "no matching explains the detection events (a defect with no path to a partner or the boundary)".to_string(),
        other => format!("the matcher declined the shot ({other:?})"),
    })
}

/// Check a batch's width and say which shot failed, if any.
fn check_width(shots: &BitTable, num_detectors: usize) -> Result<()> {
    if shots.num_bits() != num_detectors {
        return Err(Error::new(format!("{} bits per shot, not the model's {num_detectors} detectors", shots.num_bits())));
    }
    Ok(())
}

fn check_defects(defects: &[u32], num_detectors: usize) -> Result<()> {
    match defects.iter().find(|&&d| d as usize >= num_detectors) {
        Some(d) => Err(Error::new(format!("detector {d} is beyond the model's {num_detectors}"))),
        None => Ok(()),
    }
}

/// Exact minimum-weight perfect matching on a decomposed detector error model, by sparse
/// blossom, as PyMatching decodes: edges weighted ln((1 − p)/p), the observables of the lightest
/// correction returned. Correlated matching adds PyMatching 2.4's second pass.
///
/// `Send + Sync`: share one across threads, or let [`Matching::decode_batch`] cut a batch across
/// them.
///
/// ```
/// use stabilizer_qec::{DetectorErrorModel, Matching};
///
/// let dem: DetectorErrorModel = "error(0.1) D0\nerror(0.1) D0 D1 L0\nerror(0.1) D1".parse()?;
/// let m = Matching::new(&dem)?;
/// assert!(m.decode(&[0, 1])?.flips(0));
/// # Ok::<(), stabilizer_qec::Error>(())
/// ```
pub struct Matching {
    graph: SparseGraph,
    corr: Option<Correlations>,
    num_detectors: usize,
    num_observables: usize,
}

impl Matching {
    /// Plain matching. The model's faults must flip at most two detectors each, or be
    /// decomposed into such pieces.
    pub fn new(dem: &DetectorErrorModel) -> Result<Matching> {
        Matching::build(dem, false)
    }

    /// Correlated matching (PyMatching's `enable_correlations=True`).
    pub fn with_correlations(dem: &DetectorErrorModel) -> Result<Matching> {
        Matching::build(dem, true)
    }

    fn build(dem: &DetectorErrorModel, correlated: bool) -> Result<Matching> {
        let (graph, corr) = DemDecoder::new(&dem.inner)?.into_parts(correlated);
        Ok(Matching { graph, corr, num_detectors: dem.num_detectors(), num_observables: dem.num_observables() })
    }

    /// Detectors in the model.
    pub fn num_detectors(&self) -> usize {
        self.num_detectors
    }

    /// Observables in the model.
    pub fn num_observables(&self) -> usize {
        self.num_observables
    }

    /// One shot, given the detectors that fired.
    pub fn decode(&self, defects: &[u32]) -> Result<Prediction> {
        check_defects(defects, self.num_detectors)?;
        let mut sorted = defects.to_vec();
        sorted.sort_unstable();
        sorted.dedup();
        let mut scratch = Scratch::new(&self.graph);
        let p = match &self.corr {
            Some(corr) => self.graph.decode_correlated(corr, &mut scratch, &sorted),
            None => self.graph.decode(&mut scratch, &sorted),
        }
        .map_err(decode_error)?;
        Ok(Prediction { observables: p.observables, weight: Some(p.weight), bp_converged: None })
    }

    /// A batch of shots (one row per shot, one bit per detector), across `threads` threads
    /// (`0` is every core). A shot no matching explains is an error naming it.
    pub fn decode_batch(&self, shots: &BitTable, threads: usize) -> Result<Vec<Prediction>> {
        check_width(shots, self.num_detectors)?;
        let out = match_shots(&self.graph, self.corr.as_ref(), shots.as_bytes(), self.num_detectors, shots.num_rows(), threads);
        if let Some(s) = out.iter().position(|(_, w)| w.is_nan()) {
            return Err(Error::new(format!("shot {s}: no matching explains its detection events")));
        }
        Ok(out.into_iter().map(|(o, w)| Prediction { observables: o, weight: Some(w), bp_converged: None }).collect())
    }
}

/// How belief propagation updates its messages.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub enum BpMethod {
    /// Product-sum (sum-product) BP.
    ProductSum,
    /// Min-sum BP, its check messages scaled by `scaling_factor` (`0.0`: `ldpc`'s adaptive
    /// scaling, 1 − 2^−iteration).
    MinimumSum {
        /// The scaling factor.
        scaling_factor: f64,
    },
}

impl BpMethod {
    fn engine(self) -> crate::bp::Method {
        match self {
            BpMethod::ProductSum => crate::bp::Method::ProductSum,
            BpMethod::MinimumSum { scaling_factor } => crate::bp::Method::MinSum { scale: scaling_factor },
        }
    }
}

/// Belief-propagation settings.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BpOptions {
    max_iter: usize,
    method: BpMethod,
}

impl BpOptions {
    /// `max_iter` iterations of `method`.
    pub fn new(max_iter: usize, method: BpMethod) -> BpOptions {
        BpOptions { max_iter, method }
    }
}

/// How BP+OSD's ordered-statistics step searches when BP does not converge.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum OsdMethod {
    /// OSD-0: the most likely information set alone.
    Osd0,
    /// Exhaustive search over the given order's least reliable columns.
    Exhaustive(usize),
    /// Combination sweep of the given order (`ldpc`'s `osd_cs`).
    CombinationSweep(usize),
}

impl OsdMethod {
    fn engine(self) -> crate::osd::OsdMethod {
        match self {
            OsdMethod::Osd0 => crate::osd::OsdMethod::Osd0,
            OsdMethod::Exhaustive(k) => crate::osd::OsdMethod::Exhaustive(k),
            OsdMethod::CombinationSweep(k) => crate::osd::OsdMethod::CombinationSweep(k),
        }
    }
}

/// Belief-matching (Higgott et al. 2023), as the `beliefmatching` package decodes: BP on the
/// model's hypergraph; where BP's own correction explains the shot it is used, otherwise
/// matching on weights −ln p from BP's posteriors. The model must be decomposed.
pub struct BeliefMatching {
    inner: crate::belief::BeliefMatching,
    num_detectors: usize,
}

impl BeliefMatching {
    /// Belief-matching with the given BP settings (the package's defaults are 20 iterations of
    /// product-sum).
    pub fn new(dem: &DetectorErrorModel, bp: BpOptions) -> Result<BeliefMatching> {
        let inner = crate::belief::BeliefMatching::from_dem(&dem.inner, bp.method.engine(), bp.max_iter)?;
        Ok(BeliefMatching { inner, num_detectors: dem.num_detectors() })
    }

    /// One shot, given the detectors that fired.
    pub fn decode(&self, defects: &[u32]) -> Result<Prediction> {
        check_defects(defects, self.num_detectors)?;
        let mut sorted = defects.to_vec();
        sorted.sort_unstable();
        sorted.dedup();
        let o = self.inner.decode(&sorted, &mut self.inner.work()).map_err(decode_error)?;
        Ok(Prediction { observables: o.observables, weight: (!o.converged).then_some(o.weight), bp_converged: Some(o.converged) })
    }

    /// A batch of shots across `threads` threads (`0` is every core).
    pub fn decode_batch(&self, shots: &BitTable, threads: usize) -> Result<Vec<Prediction>> {
        check_width(shots, self.num_detectors)?;
        let out = belief_shots(&self.inner, shots.as_bytes(), self.num_detectors, shots.num_rows(), threads);
        if let Some(s) = out.iter().position(|x| x.2 == 2) {
            return Err(Error::new(format!("shot {s}: no correction explains its detection events")));
        }
        Ok(out
            .into_iter()
            .map(|(o, w, c)| Prediction { observables: o, weight: (c == 0).then_some(w), bp_converged: Some(c == 1) })
            .collect())
    }
}

/// BP+OSD on a detector error model (Roffe et al.): each fault a column with its prior,
/// corrections equal to `ldpc`'s `BpOsdDecoder` but for ties among equally likely columns. For
/// codes whose faults flip three or more detectors, such as the bivariate bicycle codes: give it
/// the undecomposed model.
pub struct BpOsd {
    inner: crate::osd::BpOsd,
    observables: Vec<u64>,
    num_detectors: usize,
}

impl BpOsd {
    /// BP+OSD with the given BP settings and OSD search.
    pub fn new(dem: &DetectorErrorModel, bp: BpOptions, osd: OsdMethod) -> Result<BpOsd> {
        let d = &dem.inner;
        let columns: Vec<Vec<u32>> = d.mechanisms.iter().map(|m| m.detectors.clone()).collect();
        let priors: Vec<f64> = d.mechanisms.iter().map(|m| m.p).collect();
        let inner = crate::osd::BpOsd::new(d.num_detectors, columns, &priors, bp.method.engine(), bp.max_iter, osd.engine())?;
        Ok(BpOsd { inner, observables: d.mechanisms.iter().map(|m| m.observables).collect(), num_detectors: d.num_detectors })
    }

    /// One shot, given the detectors that fired.
    pub fn decode(&self, defects: &[u32]) -> Result<Prediction> {
        check_defects(defects, self.num_detectors)?;
        let mut row = BitTable::zeros(1, self.num_detectors);
        for &d in defects {
            row.set(0, d as usize, true);
        }
        Ok(self.decode_batch(&row, 1)?.remove(0))
    }

    /// A batch of shots across `threads` threads (`0` is every core).
    pub fn decode_batch(&self, shots: &BitTable, threads: usize) -> Result<Vec<Prediction>> {
        check_width(shots, self.num_detectors)?;
        let out = bposd_shots(&self.inner, &self.observables, shots.as_bytes(), self.num_detectors, shots.num_rows(), threads);
        Ok(out.into_iter().map(|(o, c)| Prediction { observables: o, weight: None, bp_converged: Some(c == 1) }).collect())
    }
}

/// The order a window decoder runs its windows in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum WindowMode {
    /// One window after another, in time.
    Sliding,
    /// Alternate windows at once, then the gaps between them.
    Parallel,
}

/// A window decoder's schedule.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WindowOptions {
    commit: usize,
    buffer: usize,
    mode: WindowMode,
    correlations: bool,
}

impl WindowOptions {
    /// Windows that commit `commit` rounds with `buffer` rounds on either side, run in `mode`,
    /// plain matching.
    pub fn new(commit: usize, buffer: usize, mode: WindowMode) -> WindowOptions {
        WindowOptions { commit, buffer, mode, correlations: false }
    }

    /// Correlated matching in each window.
    pub fn with_correlations(mut self, yes: bool) -> WindowOptions {
        self.correlations = yes;
        self
    }
}

/// A window of a window decoder: it decodes rounds `first_layer..end_layer` and commits the
/// corrections in `commit_start..commit_end`.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct Window {
    /// The first round it decodes.
    pub first_layer: u32,
    /// One past the last round it decodes.
    pub end_layer: u32,
    /// The first round it commits.
    pub commit_start: u32,
    /// One past the last round it commits.
    pub commit_end: u32,
    /// The phase it runs in (phase 0 first).
    pub phase: usize,
    /// The windows whose commits it waits for.
    pub waits_for: Vec<usize>,
}

/// Window decoding (Skoric et al.; Tan et al.): the model cut into overlapping windows of rounds,
/// each matched alone and committing only its middle. The model's detectors need a time
/// coordinate (their last), as Stim's generated circuits give them.
pub struct WindowMatching {
    inner: WindowDecoder,
    correlations: bool,
    num_detectors: usize,
}

impl WindowMatching {
    /// Window decoding with the given schedule.
    pub fn new(dem: &DetectorErrorModel, options: WindowOptions) -> Result<WindowMatching> {
        let mode = match options.mode {
            WindowMode::Sliding => Mode::Sliding,
            WindowMode::Parallel => Mode::Parallel,
        };
        let inner = WindowDecoder::new(Model::new(&dem.inner)?, options.commit, options.buffer, mode)?;
        Ok(WindowMatching { inner, correlations: options.correlations, num_detectors: dem.num_detectors() })
    }

    /// The windows, in the order they are planned.
    pub fn windows(&self) -> Vec<Window> {
        window_info(&self.inner)
            .into_iter()
            .map(|(a, b, c0, c1, phase, waits_for)| Window { first_layer: a, end_layer: b, commit_start: c0, commit_end: c1, phase, waits_for })
            .collect()
    }

    /// A batch of shots across `threads` threads (`0` is every core).
    pub fn decode_batch(&self, shots: &BitTable, threads: usize) -> Result<Vec<Prediction>> {
        check_width(shots, self.num_detectors)?;
        let out = window_shots(&self.inner, shots.as_bytes(), self.num_detectors, shots.num_rows(), self.correlations, threads, false);
        if let Some(s) = out.iter().position(|x| x.3) {
            return Err(Error::new(format!("shot {s}: a window found no matching")));
        }
        if out.iter().any(|x| x.1 > 0) {
            return Err(Error::new("internal error: window commits left defects unexplained; please report it"));
        }
        Ok(out.into_iter().map(|x| Prediction { observables: x.0, weight: None, bp_converged: None }).collect())
    }
}

fn check_matrix(num_checks: usize, columns: &[Vec<u32>], priors: &[f64]) -> Result<()> {
    if columns.len() != priors.len() {
        return Err(Error::new(format!("{} columns but {} priors", columns.len(), priors.len())));
    }
    for (j, &p) in priors.iter().enumerate() {
        probability(p, &format!("prior {j}"))?;
    }
    if let Some((j, r)) = columns.iter().enumerate().find_map(|(j, c)| c.iter().find(|&&r| r as usize >= num_checks).map(|r| (j, r))) {
        return Err(Error::new(format!("column {j} touches check {r}, beyond {num_checks}")));
    }
    Ok(())
}

fn syndrome_bytes(syndrome: &[bool], num_checks: usize) -> Result<Vec<u8>> {
    if syndrome.len() != num_checks {
        return Err(Error::new(format!("{} syndrome bits for {num_checks} checks", syndrome.len())));
    }
    Ok(syndrome.iter().map(|&b| u8::from(b)).collect())
}

/// A BP run's result.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct BpOutcome {
    /// The hard decision, one per column.
    pub correction: Vec<bool>,
    /// Each column's posterior log-likelihood ratio, ln(P(0)/P(1)).
    pub log_prob_ratios: Vec<f64>,
    /// Whether the hard decision explains the syndrome.
    pub converged: bool,
    /// Iterations run.
    pub iterations: usize,
}

/// Belief propagation on a check matrix given by its columns (the checks each column touches),
/// flooding schedule, its arithmetic reproducing `ldpc`'s `BpDecoder` bit for bit.
pub struct BpDecoder {
    inner: crate::bp::Bp,
    options: BpOptions,
    num_checks: usize,
}

impl BpDecoder {
    /// A decoder for `num_checks` checks and the given columns, each with its prior.
    pub fn new(num_checks: usize, columns: &[Vec<u32>], priors: &[f64], options: BpOptions) -> Result<BpDecoder> {
        check_matrix(num_checks, columns, priors)?;
        Ok(BpDecoder { inner: crate::bp::Bp::new(num_checks, columns, priors)?, options, num_checks })
    }

    /// Decode one syndrome (one bit per check).
    pub fn decode(&self, syndrome: &[bool]) -> Result<BpOutcome> {
        let s = syndrome_bytes(syndrome, self.num_checks)?;
        let mut w = self.inner.work();
        let out = self.inner.decode(&s, self.options.method.engine(), self.options.max_iter, &mut w);
        Ok(BpOutcome {
            correction: w.hard.iter().map(|&b| b != 0).collect(),
            log_prob_ratios: w.llr.clone(),
            converged: out.converged,
            iterations: out.iterations,
        })
    }
}

/// A BP+OSD run's result.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct BpOsdOutcome {
    /// The correction, one per column.
    pub correction: Vec<bool>,
    /// Whether BP converged before OSD.
    pub converged: bool,
    /// BP's iterations.
    pub iterations: usize,
}

/// BP+OSD on a check matrix given by its columns, as `ldpc`'s `BpOsdDecoder`.
pub struct BpOsdDecoder {
    inner: crate::osd::BpOsd,
    num_checks: usize,
}

impl BpOsdDecoder {
    /// A decoder for `num_checks` checks and the given columns, each with its prior.
    pub fn new(num_checks: usize, columns: &[Vec<u32>], priors: &[f64], bp: BpOptions, osd: OsdMethod) -> Result<BpOsdDecoder> {
        check_matrix(num_checks, columns, priors)?;
        let inner = crate::osd::BpOsd::new(num_checks, columns.to_vec(), priors, bp.method.engine(), bp.max_iter, osd.engine())?;
        Ok(BpOsdDecoder { inner, num_checks })
    }

    /// Decode one syndrome (one bit per check).
    pub fn decode(&self, syndrome: &[bool]) -> Result<BpOsdOutcome> {
        let s = syndrome_bytes(syndrome, self.num_checks)?;
        let mut w = self.inner.work();
        let out = self.inner.decode(&s, &mut w);
        Ok(BpOsdOutcome { correction: w.correction.iter().map(|&b| b != 0).collect(), converged: out.converged, iterations: out.iterations })
    }
}

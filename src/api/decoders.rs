use super::{probability, BitTable, DetectorErrorModel, Error, Result};
use crate::batch::{StreamedWindows, belief_shots, bposd_shots, match_shots, streamed_window_shots, syndrome_shots, union_find_shots, window_info, window_shots};
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
        let (graph, corr) = DemDecoder::new(dem.flat()?)?.into_parts(correlated);
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

/// Weighted union-find decoding (Delfosse and Nickerson, with Huang, Newman and Brown's
/// weighted growth) on the same graph matching uses: clusters around the detection events grow
/// edge weight by edge weight until their events can pair inside them, and a spanning tree of
/// each is peeled for the correction. Linear time in practice, its logical error rate a little
/// above matching's; the standard baseline (this engine's matcher, near-linear too, is about
/// twice as fast). The model's faults must flip at most two detectors each, or be decomposed.
///
/// ```
/// use stabilizer_qec::{memory_circuit, Basis, DemOptions, Noise, SurfaceCode, UnionFind};
///
/// let c = memory_circuit(SurfaceCode::Rotated, 5, 5, Noise::Sd6 { p: 0.003 }, Basis::Z)?;
/// let dem = c.detector_error_model(&DemOptions::new().decompose_errors(true))?;
/// let samples = c.detector_sampler(1)?.sample(2000, 0);
/// let predictions = UnionFind::new(&dem)?.decode_batch(&samples.detectors, 0)?;
/// let wrong = (0..2000).filter(|&s| predictions[s].flips(0) != samples.observables.get(s, 0)).count();
/// assert!(wrong < 100);
/// # Ok::<(), stabilizer_qec::Error>(())
/// ```
pub struct UnionFind {
    graph: SparseGraph,
    num_detectors: usize,
    num_observables: usize,
}

impl UnionFind {
    /// The decoder for a decomposed model.
    pub fn new(dem: &DetectorErrorModel) -> Result<UnionFind> {
        let (graph, _) = DemDecoder::new(dem.flat()?)?.into_parts(false);
        Ok(UnionFind { graph, num_detectors: dem.num_detectors(), num_observables: dem.num_observables() })
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
        let mut scratch = self.graph.union_find_scratch();
        let o = self
            .graph
            .decode_union_find(&mut scratch, &sorted)
            .ok_or_else(|| Error::new("no correction explains the detection events (an odd component with no boundary)"))?;
        Ok(Prediction { observables: o, weight: None, bp_converged: None })
    }

    /// A batch of shots (one row per shot, one bit per detector), across `threads` threads
    /// (`0` is every core). A shot no correction explains is an error naming it.
    pub fn decode_batch(&self, shots: &BitTable, threads: usize) -> Result<Vec<Prediction>> {
        check_width(shots, self.num_detectors)?;
        let out = union_find_shots(&self.graph, shots.as_bytes(), self.num_detectors, shots.num_rows(), threads);
        out.into_iter()
            .enumerate()
            .map(|(s, o)| match o {
                Some(observables) => Ok(Prediction { observables, weight: None, bp_converged: None }),
                None => Err(Error::new(format!("shot {s}: no correction explains its detection events"))),
            })
            .collect()
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
        let inner = crate::belief::BeliefMatching::from_dem(dem.flat()?, bp.method.engine(), bp.max_iter)?;
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
        let d = dem.flat()?;
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

/// How BP+LSD's localized statistics decoding grows and solves its clusters.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LsdOptions {
    method: OsdMethod,
    bits_per_step: usize,
    always_run: bool,
}

impl LsdOptions {
    /// LSD with `method` searched inside each cluster: `OsdMethod::Osd0` solves each cluster
    /// once (LSD-0); the higher orders grow each cluster until it has that many free columns
    /// and search them as OSD does (LSD-E, LSD-CS). One fault joins a cluster per step.
    pub fn new(method: OsdMethod) -> LsdOptions {
        LsdOptions { method, bits_per_step: 1, always_run: false }
    }

    /// Faults joining each cluster per growth step (`0`: every candidate at once).
    pub fn with_bits_per_step(mut self, bits: usize) -> LsdOptions {
        self.bits_per_step = bits;
        self
    }

    /// Run LSD even where BP converges.
    pub fn with_always_run(mut self, yes: bool) -> LsdOptions {
        self.always_run = yes;
        self
    }

    fn engine(&self, num_checks: usize, columns: Vec<Vec<u32>>, priors: &[f64], bp: BpOptions) -> Result<crate::lsd::BpLsd> {
        Ok(crate::lsd::BpLsd::new(num_checks, columns, priors, bp.method.engine(), bp.max_iter, self.method.engine(), self.bits_per_step, self.always_run)?)
    }
}

/// BP+LSD on a detector error model (Hillmann et al., 2024): belief propagation, and where it
/// does not converge, localized statistics decoding, which grows a cluster around each
/// detection event by BP's posteriors, merges clusters that touch, and solves each alone.
/// Corrections equal to `ldpc`'s `BpLsdDecoder` but for ties among equally likely columns (and
/// the order `ldpc` merges three or more clusters in, which it takes from pointer hashes).
/// Faster than BP+OSD on large codes: its eliminations are cluster-sized. Give it the
/// undecomposed model.
pub struct BpLsd {
    inner: crate::lsd::BpLsd,
    observables: Vec<u64>,
    num_detectors: usize,
}

impl BpLsd {
    /// BP+LSD with the given BP and LSD settings.
    pub fn new(dem: &DetectorErrorModel, bp: BpOptions, lsd: LsdOptions) -> Result<BpLsd> {
        let d = dem.flat()?;
        let columns: Vec<Vec<u32>> = d.mechanisms.iter().map(|m| m.detectors.clone()).collect();
        let priors: Vec<f64> = d.mechanisms.iter().map(|m| m.p).collect();
        let inner = lsd.engine(d.num_detectors, columns, &priors, bp)?;
        Ok(BpLsd { inner, observables: d.mechanisms.iter().map(|m| m.observables).collect(), num_detectors: d.num_detectors })
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
        let out = syndrome_shots(&self.inner, &self.observables, shots.as_bytes(), self.num_detectors, shots.num_rows(), threads);
        if let Some(s) = out.iter().position(|x| x.1 == 2) {
            return Err(Error::new(format!("shot {s}: no correction explains the detection events")));
        }
        Ok(out.into_iter().map(|(o, c)| Prediction { observables: o, weight: None, bp_converged: Some(c == 1) }).collect())
    }
}

/// How Relay-BP runs: its legs, their memory strengths and when it stops.
#[derive(Clone, Debug, PartialEq)]
pub struct RelayOptions {
    config: crate::relay::RelayConfig,
}

impl Default for RelayOptions {
    fn default() -> Self {
        RelayOptions::new()
    }
}

impl RelayOptions {
    /// `relay_bp`'s sinter defaults: a first leg of 60 iterations with memory strength 0.1, then
    /// up to 60 legs of 60 iterations with strengths drawn from [−0.24, 0.66), stopping once 5
    /// have converged; min-sum unscaled; seed 0.
    pub fn new() -> RelayOptions {
        RelayOptions {
            config: crate::relay::RelayConfig {
                pre_iter: 60,
                legs: 60,
                leg_iter: 60,
                solutions: Some(5),
                gamma0: Some(0.1),
                gamma_range: (-0.24, 0.66),
                gammas: None,
                alpha: None,
                alpha_scaling: 1.0,
                seed: 0,
            },
        }
    }

    /// Legs after the first, at most.
    pub fn legs(mut self, legs: usize) -> Self {
        self.config.legs = legs;
        self
    }

    /// Stop once this many legs have converged (`None`: run every leg).
    pub fn solutions(mut self, solutions: Option<usize>) -> Self {
        self.config.solutions = solutions;
        self
    }

    /// Iterations of the first leg.
    pub fn pre_iterations(mut self, iterations: usize) -> Self {
        self.config.pre_iter = iterations;
        self
    }

    /// Iterations of each later leg.
    pub fn iterations(mut self, iterations: usize) -> Self {
        self.config.leg_iter = iterations;
        self
    }

    /// The first leg's memory strength (`None`: no memory, plain min-sum in every leg).
    pub fn gamma0(mut self, gamma0: Option<f64>) -> Self {
        self.config.gamma0 = gamma0;
        self
    }

    /// The interval later legs draw memory strengths from, uniformly.
    pub fn gamma_range(mut self, low: f64, high: f64) -> Self {
        self.config.gamma_range = (low, high);
        self
    }

    /// Explicit memory strengths instead of random ones: one row per leg (reused cyclically),
    /// one strength per fault.
    pub fn gammas(mut self, rows: Vec<Vec<f64>>) -> Self {
        self.config.gammas = Some(rows);
        self
    }

    /// The min-sum scaling (`None`: 1; `Some(0.0)`: 1 − 2^−t at iteration t).
    pub fn alpha(mut self, alpha: Option<f64>) -> Self {
        self.config.alpha = alpha;
        self
    }

    /// The seed of the random memory strengths.
    pub fn seed(mut self, seed: u64) -> Self {
        self.config.seed = seed;
        self
    }
}

/// Relay-BP on a detector error model (Müller et al., IBM, 2025): min-sum BP with memory, run
/// in legs that each draw fresh memory strengths and start from the last leg's posteriors; the
/// lightest of the first few converged corrections wins. No elimination: fast and simple
/// enough for hardware, and on the gross code about as accurate as BP+OSD. IBM's `relay_bp`
/// gives the same corrections on the same check matrix (see `RelayBpDecoder`). Shot `k` of a
/// batch draws its memory strengths from the seed and `k`, so results do not depend on the
/// thread count. Give it the undecomposed model.
pub struct RelayBp {
    inner: crate::relay::Relay,
    observables: Vec<u64>,
    num_detectors: usize,
}

impl RelayBp {
    /// Relay-BP with the given options.
    pub fn new(dem: &DetectorErrorModel, options: RelayOptions) -> Result<RelayBp> {
        let d = dem.flat()?;
        let columns: Vec<Vec<u32>> = d.mechanisms.iter().map(|m| m.detectors.clone()).collect();
        let priors: Vec<f64> = d.mechanisms.iter().map(|m| m.p).collect();
        let inner = crate::relay::Relay::new(d.num_detectors, &columns, &priors, options.config)?;
        Ok(RelayBp { inner, observables: d.mechanisms.iter().map(|m| m.observables).collect(), num_detectors: d.num_detectors })
    }

    /// One shot, given the detectors that fired (decoded as shot 0 of a batch).
    pub fn decode(&self, defects: &[u32]) -> Result<Prediction> {
        check_defects(defects, self.num_detectors)?;
        let mut row = BitTable::zeros(1, self.num_detectors);
        for &d in defects {
            row.set(0, d as usize, true);
        }
        Ok(self.decode_batch(&row, 1)?.remove(0))
    }

    /// A batch of shots across `threads` threads (`0` is every core). `bp_converged` says
    /// whether a leg converged; where none did, the prediction is the first leg's last guess.
    pub fn decode_batch(&self, shots: &BitTable, threads: usize) -> Result<Vec<Prediction>> {
        check_width(shots, self.num_detectors)?;
        let out = crate::batch::relay_shots(&self.inner, &self.observables, shots.as_bytes(), self.num_detectors, shots.num_rows(), threads);
        Ok(out.into_iter().map(|(o, c)| Prediction { observables: o, weight: None, bp_converged: Some(c == 1) }).collect())
    }
}

/// Colour-code decoding by matching, Chromobius's construction (Gidney and Jones, 2023): each
/// detector doubled into the two sub-graphs that leave out a colour other than its own, every
/// fault split into basic faults drawn as edges of that doubled (Möbius) graph, a minimum-
/// weight matching of it found by this crate's matcher, and the matching lifted back to the
/// code by carrying colour charge around its cycles. Predictions equal to the `chromobius`
/// package's but for ties between equally light matchings.
///
/// The model's detectors need Chromobius's annotation: a 4th coordinate giving the basis and
/// colour (0, 1, 2 red, green, blue X; 3, 4, 5 the same in Z; −1 to ignore the detector), as
/// [`CssCode::memory_circuit_with_colors`](crate::CssCode::memory_circuit_with_colors) writes
/// it. Give it the undecomposed model.
///
/// ```
/// use stabilizer_qec::{Basis, ColorMatching, CssCode, DemOptions};
///
/// let c = CssCode::color_code(5)?.memory_circuit_with_colors(5, 0.001, Basis::Z)?;
/// let decoder = ColorMatching::new(&c.detector_error_model(&DemOptions::new())?)?;
/// let samples = c.detector_sampler(1)?.sample(2000, 0);
/// let predictions = decoder.decode_batch(&samples.detectors, 0)?;
/// let wrong = (0..2000).filter(|&s| predictions[s].flips(0) != samples.observables.get(s, 0)).count();
/// assert!(wrong < 40);
/// # Ok::<(), stabilizer_qec::Error>(())
/// ```
pub struct ColorMatching {
    inner: crate::color::ColorDecoder,
    num_detectors: usize,
}

impl ColorMatching {
    /// The decoder for an annotated model; an error where a fault cannot be split into the
    /// code's basic faults (as in Chromobius).
    pub fn new(dem: &DetectorErrorModel) -> Result<ColorMatching> {
        ColorMatching::build(dem, false)
    }

    /// The same, leaving out of the Möbius graph what cannot be decomposed.
    pub fn ignoring_decomposition_failures(dem: &DetectorErrorModel) -> Result<ColorMatching> {
        ColorMatching::build(dem, true)
    }

    fn build(dem: &DetectorErrorModel, ignore: bool) -> Result<ColorMatching> {
        let d = dem.flat()?;
        Ok(ColorMatching { inner: crate::color::ColorDecoder::from_dem(d, ignore)?, num_detectors: d.num_detectors })
    }

    /// The Möbius model the matching is found on: two detectors per detector, edges only.
    pub fn mobius_model(&self) -> Result<DetectorErrorModel> {
        DetectorErrorModel::parse(self.inner.mobius_text())
    }

    /// One shot, given the detectors that fired. `weight` is the Möbius matching's.
    pub fn decode(&self, defects: &[u32]) -> Result<Prediction> {
        check_defects(defects, self.num_detectors)?;
        let mut row = BitTable::zeros(1, self.num_detectors);
        for &d in defects {
            row.set(0, d as usize, true);
        }
        Ok(self.decode_batch(&row, 1)?.remove(0))
    }

    /// A batch of shots across `threads` threads (`0` is every core). A shot no lifting of
    /// the matching explains (wrong annotations, or a model Chromobius cannot decode either)
    /// is an error naming it.
    pub fn decode_batch(&self, shots: &BitTable, threads: usize) -> Result<Vec<Prediction>> {
        check_width(shots, self.num_detectors)?;
        let out = crate::batch::color_shots(&self.inner, shots.as_bytes(), self.num_detectors, shots.num_rows(), threads);
        if let Some(s) = out.iter().position(|x| x.2 == 2) {
            return Err(Error::new(format!("shot {s}: no lifting of the matching explains its detection events (are the colour annotations right?)")));
        }
        Ok(out.into_iter().map(|(o, w, _)| Prediction { observables: o, weight: Some(w), bp_converged: None }).collect())
    }
}

/// How the search decoder generates detector orders.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum DetectorOrder {
    /// Detector index order or its reverse, by a coin (Tesseract's, exactly).
    Index,
    /// Breadth-first over the detectors faults join, from random roots.
    Bfs,
    /// By the coordinates' projection on a random direction.
    Coordinate,
}

/// The search decoder's settings; the defaults are Tesseract's.
#[derive(Clone, Debug, PartialEq)]
pub struct SearchOptions {
    beam: Option<usize>,
    beam_climbing: bool,
    no_revisit: bool,
    queue_limit: Option<usize>,
    orders: Option<Vec<Vec<u32>>>,
    method: DetectorOrder,
    num_orders: usize,
    seed: u64,
    penalty: f64,
    merge: bool,
}

impl Default for SearchOptions {
    fn default() -> Self {
        SearchOptions::new()
    }
}

impl SearchOptions {
    /// Tesseract's defaults: beam 5, no revisits, a queue of at most 200,000, 20 index orders
    /// from seed 2384753, no detector penalty, indistinguishable faults merged.
    pub fn new() -> SearchOptions {
        SearchOptions { beam: Some(5), beam_climbing: false, no_revisit: true, queue_limit: Some(200_000), orders: None, method: DetectorOrder::Index, num_orders: 20, seed: 2384753, penalty: 0.0, merge: true }
    }

    /// Drop states lighting more than `beam` detectors above the fewest seen (`None`: no beam).
    pub fn beam(mut self, beam: Option<usize>) -> Self {
        self.beam = beam;
        self
    }

    /// Search with every beam from 0 up, cycling through the orders.
    pub fn beam_climbing(mut self, yes: bool) -> Self {
        self.beam_climbing = yes;
        self
    }

    /// Skip patterns of lit detectors already expanded: much faster, not exact.
    pub fn no_revisit(mut self, yes: bool) -> Self {
        self.no_revisit = yes;
        self
    }

    /// Give up after this many states have been queued (`None`: never).
    pub fn queue_limit(mut self, limit: Option<usize>) -> Self {
        self.queue_limit = limit;
        self
    }

    /// Search these detector orders (each a permutation of the detectors).
    pub fn detector_orders(mut self, orders: Vec<Vec<u32>>) -> Self {
        self.orders = Some(orders);
        self
    }

    /// Generate `count` orders by `method` from `seed`.
    pub fn generated_orders(mut self, method: DetectorOrder, count: usize, seed: u64) -> Self {
        self.orders = None;
        self.method = method;
        self.num_orders = count;
        self.seed = seed;
        self
    }

    /// Add this to each lit detector's estimated cost.
    pub fn detector_penalty(mut self, penalty: f64) -> Self {
        self.penalty = penalty;
        self
    }

    /// Merge faults with the same detectors and observables into one.
    pub fn merge_errors(mut self, yes: bool) -> Self {
        self.merge = yes;
        self
    }
}

/// A search for the most likely error, Tesseract's A* (Beni, Higgott and Shutty, Google, 2025),
/// for any detector error model: sets of faults expanded cheapest first by their weight plus an
/// admissible estimate of explaining the detectors still lit, each grown only through its first
/// lit detector in a detector order, under a beam and a queue bound, once per order. Its
/// answers, fault for fault, are the `tesseract_decoder` package's on the same platform (ties
/// between equally promising states are broken as the platform's C++ library breaks them).
/// With no beam, no queue bound and revisits allowed it is exact.
///
/// ```
/// use stabilizer_qec::{CssCode, DemOptions, SearchDecoder, SearchOptions, Basis};
///
/// let c = CssCode::color_code(3)?.memory_circuit(3, 0.002, Basis::Z)?;
/// let decoder = SearchDecoder::new(&c.detector_error_model(&DemOptions::new())?, SearchOptions::new())?;
/// let samples = c.detector_sampler(1)?.sample(500, 0);
/// let predictions = decoder.decode_batch(&samples.detectors, 0)?;
/// let wrong = (0..500).filter(|&s| predictions[s].flips(0) != samples.observables.get(s, 0)).count();
/// assert!(wrong < 25);
/// # Ok::<(), stabilizer_qec::Error>(())
/// ```
pub struct SearchDecoder {
    inner: crate::search::Search,
    num_detectors: usize,
}

impl SearchDecoder {
    /// The decoder for a model.
    pub fn new(dem: &DetectorErrorModel, options: SearchOptions) -> Result<SearchDecoder> {
        let d = dem.flat()?;
        let flavour = crate::search::Flavour::native();
        let orders = match options.orders {
            Some(o) => o,
            None => {
                if options.num_orders == 0 {
                    return Err(Error::new("at least one detector order is needed"));
                }
                let method = match options.method {
                    DetectorOrder::Index => crate::search::OrderMethod::Index,
                    DetectorOrder::Bfs => crate::search::OrderMethod::Bfs,
                    DetectorOrder::Coordinate => crate::search::OrderMethod::Coordinate,
                };
                crate::search::generated_orders(d, options.num_orders, method, options.seed, flavour)
            }
        };
        let config = crate::search::SearchConfig {
            beam: options.beam,
            beam_climbing: options.beam_climbing,
            no_revisit: options.no_revisit,
            queue_limit: options.queue_limit,
            orders,
            detector_penalty: options.penalty,
            merge_errors: options.merge,
            flavour,
        };
        Ok(SearchDecoder { inner: crate::search::Search::new(d, config)?, num_detectors: d.num_detectors })
    }

    /// One shot, given the detectors that fired. `weight` is the found faults' cost, `None`
    /// where the search gave up (the prediction is then no flips).
    pub fn decode(&self, defects: &[u32]) -> Result<Prediction> {
        check_defects(defects, self.num_detectors)?;
        let mut row = BitTable::zeros(1, self.num_detectors);
        for &d in defects {
            row.set(0, d as usize, true);
        }
        Ok(self.decode_batch(&row, 1)?.remove(0))
    }

    /// The faults one shot's search found, as indices into the model's (flattened) faults, or
    /// `None` where it gave up.
    pub fn decode_to_faults(&self, defects: &[u32]) -> Result<Option<Vec<usize>>> {
        check_defects(defects, self.num_detectors)?;
        let mut sorted = defects.to_vec();
        sorted.sort_unstable();
        sorted.dedup();
        let found = self.inner.decode(&sorted);
        Ok((!found.low_confidence).then_some(found.faults))
    }

    /// A batch of shots across `threads` threads (`0` is every core).
    pub fn decode_batch(&self, shots: &BitTable, threads: usize) -> Result<Vec<Prediction>> {
        check_width(shots, self.num_detectors)?;
        let out = crate::batch::search_shots(&self.inner, shots.as_bytes(), self.num_detectors, shots.num_rows(), threads);
        Ok(out.into_iter().map(|(o, c, gave_up)| Prediction { observables: if gave_up { 0 } else { o }, weight: (!gave_up).then_some(c), bp_converged: None }).collect())
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
    template: Option<bool>,
}

impl Window {
    pub(crate) fn from_info((a, b, c0, c1, phase, waits_for): crate::batch::WindowInfo) -> Window {
        Window { first_layer: a, end_layer: b, commit_start: c0, commit_end: c1, phase, waits_for }
    }
}

impl WindowOptions {
    pub(crate) fn parts(self) -> (usize, usize, WindowMode, bool) {
        (self.commit, self.buffer, self.mode, self.correlations)
    }

    /// Windows that commit `commit` rounds with `buffer` rounds on either side, run in `mode`,
    /// plain matching.
    pub fn new(commit: usize, buffer: usize, mode: WindowMode) -> WindowOptions {
        WindowOptions { commit, buffer, mode, correlations: false, template: None }
    }

    /// Where the windows' graphs come from. `Some(true)`: a short template of the model's
    /// longest loop, whose middle windows serve every window of the long model away from its
    /// ends, shifted (the decoder never unrolls the model: a memory of millions of rounds
    /// decodes); `Some(false)`: the whole model, unrolled; `None` (the default): the template
    /// for a model of more than a million faults that has one, else the whole model. Both give
    /// the same predictions.
    pub fn with_template(mut self, template: Option<bool>) -> WindowOptions {
        self.template = template;
        self
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
    inner: Windows,
    correlations: bool,
    num_detectors: usize,
}

enum Windows {
    Whole(Box<WindowDecoder>),
    Streamed(Box<StreamedWindows>),
}

impl WindowMatching {
    /// Window decoding with the given schedule.
    pub fn new(dem: &DetectorErrorModel, options: WindowOptions) -> Result<WindowMatching> {
        let mode = match options.mode {
            WindowMode::Sliding => Mode::Sliding,
            WindowMode::Parallel => Mode::Parallel,
        };
        let streamed = || StreamedWindows::new(dem.program(), options.commit, options.buffer, mode).map(Box::new);
        let inner = match options.template {
            Some(true) => Windows::Streamed(streamed()?),
            None if dem.num_errors() > 1_000_000 => match streamed() {
                Ok(s) => Windows::Streamed(s),
                Err(_) => Windows::Whole(Box::new(WindowDecoder::new(Model::new(dem.flat()?)?, options.commit, options.buffer, mode)?)),
            },
            _ => Windows::Whole(Box::new(WindowDecoder::new(Model::new(dem.flat()?)?, options.commit, options.buffer, mode)?)),
        };
        Ok(WindowMatching { inner, correlations: options.correlations, num_detectors: dem.num_detectors() })
    }

    /// The windows, in the order they are planned.
    pub fn windows(&self) -> Vec<Window> {
        let info = match &self.inner {
            Windows::Whole(wd) => window_info(wd),
            Windows::Streamed(sw) => sw.windows(),
        };
        info.into_iter().map(Window::from_info).collect()
    }

    /// A batch of shots across `threads` threads (`0` is every core).
    pub fn decode_batch(&self, shots: &BitTable, threads: usize) -> Result<Vec<Prediction>> {
        check_width(shots, self.num_detectors)?;
        let out = match &self.inner {
            Windows::Whole(wd) => window_shots(wd, shots.as_bytes(), self.num_detectors, shots.num_rows(), self.correlations, threads, false),
            Windows::Streamed(sw) => streamed_window_shots(sw, shots.as_bytes(), self.num_detectors, shots.num_rows(), self.correlations, threads, false),
        };
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

/// BP+LSD on a check matrix given by its columns, as `ldpc`'s `BpLsdDecoder` (flooding
/// schedule).
pub struct BpLsdDecoder {
    inner: crate::lsd::BpLsd,
    num_checks: usize,
}

impl BpLsdDecoder {
    /// A decoder for `num_checks` checks and the given columns, each with its prior.
    pub fn new(num_checks: usize, columns: &[Vec<u32>], priors: &[f64], bp: BpOptions, lsd: LsdOptions) -> Result<BpLsdDecoder> {
        check_matrix(num_checks, columns, priors)?;
        Ok(BpLsdDecoder { inner: lsd.engine(num_checks, columns.to_vec(), priors, bp)?, num_checks })
    }

    /// Decode one syndrome (one bit per check). An error if no correction explains it.
    pub fn decode(&self, syndrome: &[bool]) -> Result<BpOsdOutcome> {
        let s = syndrome_bytes(syndrome, self.num_checks)?;
        let mut w = self.inner.work();
        let out = self.inner.decode(&s, &mut w);
        if !out.solved {
            return Err(Error::new("no correction explains the syndrome"));
        }
        Ok(BpOsdOutcome { correction: w.correction.iter().map(|&b| b != 0).collect(), converged: out.converged, iterations: out.iterations })
    }
}

/// A Relay-BP run's result.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct RelayOutcome {
    /// The correction, one per column: the lightest converged leg's, or where none converged,
    /// the first leg's last hard decision.
    pub correction: Vec<bool>,
    /// Whether a leg converged.
    pub converged: bool,
    /// Iterations over every leg run.
    pub iterations: usize,
    /// Legs run, the first included.
    pub legs: usize,
    /// The correction's weight, Σ ln((1 − p)/p) over its columns (infinite where none converged).
    pub weight: f64,
}

/// Relay-BP on a check matrix given by its columns, as IBM's `relay_bp` `RelayDecoderF64`:
/// the same corrections, iteration counts and convergence for the same options. As there, the
/// random memory strengths continue from one decode to the next.
pub struct RelayBpDecoder {
    inner: crate::relay::Relay,
    work: std::sync::Mutex<crate::relay::RelayWork>,
}

impl RelayBpDecoder {
    /// A decoder for `num_checks` checks and the given columns, each with its prior.
    pub fn new(num_checks: usize, columns: &[Vec<u32>], priors: &[f64], options: RelayOptions) -> Result<RelayBpDecoder> {
        check_matrix(num_checks, columns, priors)?;
        let inner = crate::relay::Relay::new(num_checks, columns, priors, options.config)?;
        let work = std::sync::Mutex::new(inner.work());
        Ok(RelayBpDecoder { inner, work })
    }

    /// Decode one syndrome (one bit per check).
    pub fn decode(&self, syndrome: &[bool]) -> Result<RelayOutcome> {
        let s = syndrome_bytes(syndrome, self.inner.num_checks)?;
        let mut w = self.work.lock().unwrap_or_else(|e| e.into_inner());
        let out = self.inner.decode(&s, &mut w);
        Ok(RelayOutcome {
            correction: w.correction.iter().map(|&b| b != 0).collect(),
            converged: out.converged,
            iterations: out.iterations,
            legs: out.legs,
            weight: if out.converged { out.weight } else { f64::INFINITY },
        })
    }
}

use std::fmt;
use std::ops::{Add, AddAssign, Mul, MulAssign};
use std::str::FromStr;

use super::{BitTable, Error, Result};
use crate::batch_sampler::{BatchSampler, Counts};
use crate::circuit;
use crate::dem::Dem;
use crate::dem_program::{DemProgram, Stats};
use std::sync::OnceLock;
use crate::m2d::M2d;

/// A stabilizer circuit in Stim's circuit language.
///
/// Parse one from Stim's text (`str::parse` or [`Circuit::parse`]); [`Display`](fmt::Display)
/// prints it back as written. Every Clifford gate of Stim's is read (H, S and CX natively, the
/// rest as their exact decompositions), with resets and measurements in all three bases,
/// inverted targets, Pauli-product measurements and rotations, every noise channel,
/// measurement feedback and sweep bits, `MPAD`, detectors, observables (of records and of Pauli
/// targets), coordinates, `TICK`, `REPEAT` and instruction tags. Counts are taken through
/// `REPEAT` blocks without unrolling them.
///
/// ```
/// use stabilizer_qec::Circuit;
///
/// let c: Circuit = "R 0 1\nH 0\nCX 0 1\nM 0 1\nDETECTOR rec[-1] rec[-2]".parse()?;
/// assert_eq!((c.num_qubits(), c.num_measurements(), c.num_detectors()), (2, 2, 1));
/// assert_eq!(c.to_string().parse::<Circuit>()?, c);
/// # Ok::<(), stabilizer_qec::Error>(())
/// ```
///
/// Circuits are also built in code, as in Stim: [`append`](Circuit::append) an instruction at
/// a time, `+` one circuit after another, and `*` a circuit into a `REPEAT` block. A piece may
/// read measurements made before it (a round's detectors comparing with the last round's);
/// sampling or analysing a circuit that reads before its first measurement is the error.
///
/// ```
/// use stabilizer_qec::{Circuit, Target};
///
/// let mut round = Circuit::new();
/// round.append("CX", &[Target::Qubit(0), Target::Qubit(2), Target::Qubit(1), Target::Qubit(2)], &[])?;
/// round.append("MR", &[Target::Qubit(2)], &[0.01])?;
/// round.append("DETECTOR", &[Target::Rec(1), Target::Rec(2)], &[])?;
/// let first: Circuit = "R 0 1 2\nCX 0 2 1 2\nMR 2".parse()?;
/// let memory = &first + &(&round * 10);
/// assert_eq!((memory.num_measurements(), memory.num_detectors()), (11, 10));
/// assert!(memory.to_string().contains("REPEAT 10 {"));
/// # Ok::<(), stabilizer_qec::Error>(())
/// ```
#[derive(Clone, Debug, PartialEq, Default)]
pub struct Circuit {
    pub(crate) inner: circuit::Circuit,
    counts: Counts,
}

/// A target of an instruction, for [`Circuit::append`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Target {
    /// A qubit (`3`).
    Qubit(u32),
    /// A qubit whose measurement result is inverted (`!3`).
    Inverted(u32),
    /// The measurement record this many measurements back (`rec[-1]` is `Rec(1)`, the latest).
    Rec(u32),
    /// A sweep bit (`sweep[0]`).
    Sweep(u32),
    /// A Pauli on a qubit (`X3`, or `!X3` when `inverted`), for `MPP`, `SPP`, `E` and
    /// `OBSERVABLE_INCLUDE`.
    Pauli {
        /// The Pauli.
        pauli: Pauli,
        /// Its qubit.
        qubit: u32,
        /// Whether it is negated.
        inverted: bool,
    },
    /// Joins the Pauli targets either side into one product (`X0*Z1`).
    Combiner,
}

/// A single-qubit Pauli.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Pauli {
    /// X.
    X,
    /// Y.
    Y,
    /// Z.
    Z,
}

impl fmt::Display for Target {
    /// The target as Stim writes it.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Target::Qubit(q) => write!(f, "{q}"),
            Target::Inverted(q) => write!(f, "!{q}"),
            Target::Rec(k) => write!(f, "rec[-{k}]"),
            Target::Sweep(k) => write!(f, "sweep[{k}]"),
            Target::Pauli { pauli, qubit, inverted } => {
                let p = match pauli {
                    Pauli::X => 'X',
                    Pauli::Y => 'Y',
                    Pauli::Z => 'Z',
                };
                write!(f, "{}{p}{qubit}", if inverted { "!" } else { "" })
            }
            Target::Combiner => f.write_str("*"),
        }
    }
}

impl Circuit {
    /// The empty circuit.
    pub fn new() -> Circuit {
        Circuit::default()
    }

    /// Parse Stim's circuit text.
    pub fn parse(text: &str) -> Result<Circuit> {
        Circuit::from_engine(circuit::Circuit::parse(text)?)
    }

    pub(crate) fn from_engine(inner: circuit::Circuit) -> Result<Circuit> {
        let counts = Counts::of(&inner.instrs)?;
        fits(&counts)?;
        Ok(Circuit { inner, counts })
    }

    /// Append one instruction: `name` (any of the circuit language's), its targets, and its
    /// arguments (probabilities, coordinates, an observable's index). As Stim's
    /// `Circuit.append`; an instruction the language does not allow is the error, and the
    /// circuit is left as it was.
    pub fn append(&mut self, name: &str, targets: &[Target], args: &[f64]) -> Result<()> {
        self.append_tagged(name, targets, args, "")
    }

    /// [`append`](Circuit::append), with Stim's instruction tag (`H[tag] 0`).
    pub fn append_tagged(&mut self, name: &str, targets: &[Target], args: &[f64], tag: &str) -> Result<()> {
        let targets: Vec<String> = targets.iter().map(Target::to_string).collect();
        let piece = Circuit::from_engine(circuit::Circuit::instruction(name, tag, args, &targets)?)?;
        self.append_circuit(&piece)
    }

    /// Append Stim's circuit text, which may hold several instructions and `REPEAT` blocks.
    pub fn append_text(&mut self, text: &str) -> Result<()> {
        let piece = Circuit::parse(text)?;
        self.append_circuit(&piece)
    }

    /// Append another circuit's instructions.
    pub fn append_circuit(&mut self, other: &Circuit) -> Result<()> {
        let counts = self.counts.then(other.counts)?;
        fits(&counts)?;
        self.inner.instrs.extend(other.inner.instrs.iter().cloned());
        self.counts = counts;
        Ok(())
    }

    /// `REPEAT count { self }`, as Stim's `circuit * count`: the empty circuit for 0, the
    /// circuit itself for 1. The error is a count whose measurements or detectors overflow.
    pub fn repeated(&self, count: u64) -> Result<Circuit> {
        let counts = if count == 0 { Counts::default() } else { self.counts.times(count)? };
        fits(&counts)?;
        Ok(Circuit { inner: self.inner.repeated(count), counts })
    }

    /// One more than the largest qubit index any instruction names.
    pub fn num_qubits(&self) -> usize {
        self.counts.qubits
    }

    /// Measurement records per shot.
    pub fn num_measurements(&self) -> usize {
        self.counts.measurements as usize
    }

    /// Detectors per shot.
    pub fn num_detectors(&self) -> usize {
        self.counts.detectors as usize
    }

    /// Logical observables.
    pub fn num_observables(&self) -> usize {
        self.counts.observables
    }

    /// Sweep bits the circuit reads (one more than the largest `sweep[k]`).
    pub fn num_sweep_bits(&self) -> usize {
        self.counts.sweep_bits
    }

    /// The circuit's detector error model, built by walking it backwards as Stim's error analyzer
    /// does (see [`DemOptions`]). Its loops are folded into `repeat` blocks where they repeat,
    /// as Stim folds them, so a long memory's model costs one period of the loop.
    ///
    /// ```
    /// use stabilizer_qec::{Circuit, DemOptions};
    ///
    /// let c: Circuit = "R 0\nX_ERROR(0.125) 0\nM 0\nDETECTOR rec[-1]".parse()?;
    /// assert_eq!(c.detector_error_model(&DemOptions::new())?.to_string().trim(), "error(0.125) D0");
    /// let long: Circuit = "R 0\nREPEAT 1000000 {\n X_ERROR(0.01) 0\n MR 0\n DETECTOR rec[-1]\n}".parse()?;
    /// let dem = long.detector_error_model(&DemOptions::new())?;
    /// assert_eq!((dem.num_detectors(), dem.num_errors()), (1_000_000, 1_000_000));
    /// assert!(dem.to_string().starts_with("repeat 999999 {"));
    /// # Ok::<(), stabilizer_qec::Error>(())
    /// ```
    pub fn detector_error_model(&self, options: &DemOptions) -> Result<DetectorErrorModel> {
        let program = crate::dem_build::build(&self.inner, options.decompose_errors, options.approximate_disjoint_errors, !options.flatten_loops)?;
        Ok(DetectorErrorModel::from_program(program))
    }

    /// A sampler of detection events and observable flips, seeded: the same seed gives the same
    /// shots on any machine and any number of threads.
    pub fn detector_sampler(&self, seed: u64) -> Result<DetectorSampler> {
        Ok(DetectorSampler { sampler: BatchSampler::new(&self.inner)?, seed, next: 0 })
    }

    /// A picture of the circuit, after Stim's `diagram` (see [`DiagramKind`]): its timeline as
    /// text or SVG, what its detectors compare at a moment, or its matching graph.
    ///
    /// ```
    /// use stabilizer_qec::{Circuit, DiagramKind};
    ///
    /// let c: Circuit = "R 0 1\nH 0\nTICK\nCX 0 1\nTICK\nM 0 1\nDETECTOR rec[-1] rec[-2]".parse()?;
    /// let text = c.diagram(DiagramKind::TimelineText)?;
    /// assert!(text.contains("q0: -R-H-@-M:rec[0]-DETECTOR:D0=rec[1]*rec[0]-"));
    /// assert!(text.contains("q1: -R---X-M:rec[1]-"));
    /// // Just before the measurements, the detector compares Z0 Z1.
    /// assert_eq!(c.diagram(DiagramKind::DetectorSliceText { tick: 2 })?, "D0: Z0 Z1\n");
    /// # Ok::<(), stabilizer_qec::Error>(())
    /// ```
    pub fn diagram(&self, kind: DiagramKind) -> Result<String> {
        use crate::diagram as d;
        Ok(match kind {
            DiagramKind::TimelineText => d::timeline_text(&self.inner)?,
            DiagramKind::TimelineSvg => d::timeline_svg(&self.inner)?,
            DiagramKind::DetectorSliceText { tick } => d::detslice_text(&self.inner, tick)?,
            DiagramKind::DetectorSliceSvg { tick } => d::detslice_svg(&self.inner, tick)?,
            DiagramKind::MatchGraphSvg => {
                let dem = self.detector_error_model(&DemOptions::new().decompose_errors(true).approximate_disjoint_errors(Some(1.0)))?;
                dem.matchgraph_svg()?
            }
        })
    }

    /// A converter from raw measurement records (and sweep bits) to detection events and
    /// observable flips, as `stim m2d`. Its reference run keeps a dense tableau: at most 16,384
    /// qubits.
    pub fn measurement_converter(&self) -> Result<MeasurementConverter> {
        Ok(MeasurementConverter { m2d: M2d::new(&self.inner)? })
    }
}

/// Counts this machine can index.
fn fits(c: &Counts) -> Result<()> {
    if usize::try_from(c.measurements).is_err() || usize::try_from(c.detectors).is_err() {
        return Err(Error::new("too many measurements or detectors for this machine"));
    }
    Ok(())
}

/// One circuit after another. Panics where [`Circuit::append_circuit`] would fail: when the
/// measurements or detectors overflow a count.
impl Add<&Circuit> for &Circuit {
    type Output = Circuit;

    fn add(self, rhs: &Circuit) -> Circuit {
        let mut out = self.clone();
        out += rhs;
        out
    }
}

/// One circuit after another (see `&Circuit + &Circuit`).
impl Add<&Circuit> for Circuit {
    type Output = Circuit;

    fn add(mut self, rhs: &Circuit) -> Circuit {
        self += rhs;
        self
    }
}

/// Append a circuit (see `&Circuit + &Circuit`).
impl AddAssign<&Circuit> for Circuit {
    fn add_assign(&mut self, rhs: &Circuit) {
        if let Err(e) = self.append_circuit(rhs) {
            panic!("{e}");
        }
    }
}

/// [`Circuit::repeated`]. Panics where it would fail: when the measurements or detectors
/// overflow a count.
impl Mul<u64> for &Circuit {
    type Output = Circuit;

    fn mul(self, count: u64) -> Circuit {
        self.repeated(count).unwrap_or_else(|e| panic!("{e}"))
    }
}

/// [`Circuit::repeated`] (see `&Circuit * u64`).
impl Mul<u64> for Circuit {
    type Output = Circuit;

    fn mul(self, count: u64) -> Circuit {
        &self * count
    }
}

/// [`Circuit::repeated`], in place (see `&Circuit * u64`).
impl MulAssign<u64> for Circuit {
    fn mul_assign(&mut self, count: u64) {
        *self = &*self * count;
    }
}

impl FromStr for Circuit {
    type Err = Error;

    fn from_str(text: &str) -> Result<Circuit> {
        Circuit::parse(text)
    }
}

impl fmt::Display for Circuit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.inner.to_stim())
    }
}

/// A kind of [`Circuit::diagram`], after Stim's.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum DiagramKind {
    /// The timeline as text (Stim's `timeline-text`): a row per qubit, each operation in the
    /// first column its wires are free, `TICK` groups bracketed, loops drawn once.
    TimelineText,
    /// The timeline as an SVG picture (`timeline-svg`).
    TimelineSvg,
    /// What each detector compares after `tick` `TICK`s, as text (`detslice-text`): a line per
    /// detector, its Paulis.
    DetectorSliceText {
        /// The moment: after this many `TICK`s (0 is the start).
        tick: u64,
    },
    /// The same as an SVG picture over the qubits' coordinates (`detslice-svg`).
    DetectorSliceSvg {
        /// The moment.
        tick: u64,
    },
    /// The decomposed model's matching graph as an SVG picture (`matchgraph-svg`).
    MatchGraphSvg,
}

/// How [`Circuit::detector_error_model`] builds a model.
///
/// ```
/// use stabilizer_qec::DemOptions;
///
/// let options = DemOptions::new().decompose_errors(true).approximate_disjoint_errors(Some(0.1));
/// # let _ = options;
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct DemOptions {
    decompose_errors: bool,
    approximate_disjoint_errors: Option<f64>,
    flatten_loops: bool,
}

impl DemOptions {
    /// Undecomposed, no approximation.
    pub fn new() -> DemOptions {
        DemOptions::default()
    }

    /// Split each fault into graph-like pieces (at most two detectors each) as Stim splits them,
    /// which matching needs.
    pub fn decompose_errors(mut self, yes: bool) -> DemOptions {
        self.decompose_errors = yes;
        self
    }

    /// Stim's `approximate_disjoint_errors`: `None` refuses channels whose cases are disjoint
    /// rather than independent (`PAULI_CHANNEL_2`, `ELSE_CORRELATED_ERROR`, heralded errors, a
    /// `PAULI_CHANNEL_1` without an independent equivalent); `Some(t)` approximates them case by
    /// case, refusing any with a probability above `t` (`Some(1.0)` is Stim's `True`).
    pub fn approximate_disjoint_errors(mut self, threshold: Option<f64>) -> DemOptions {
        self.approximate_disjoint_errors = threshold;
        self
    }

    /// Stim's `flatten_loops`: walk every pass of every loop and write the model without
    /// `repeat` blocks, rather than folding the loops that repeat (the default).
    pub fn flatten_loops(mut self, yes: bool) -> DemOptions {
        self.flatten_loops = yes;
        self
    }
}

/// A detector error model in Stim's format: independent faults, each with a probability, the
/// detectors it flips and the observables it flips. Parse one from Stim's text, or build one
/// with [`Circuit::detector_error_model`].
///
/// It is held as written, `repeat` blocks and all, and counted through its loops without
/// unrolling them; a decoder made from it unrolls it once (up to 2²⁴ faults and declarations
/// and detector 2²⁴ − 1, past which making the decoder is the error), taking its faults in the
/// order they are written, so a model and its text decode alike.
#[derive(Clone, Debug)]
pub struct DetectorErrorModel {
    program: DemProgram,
    stats: Stats,
    flat: OnceLock<std::result::Result<Dem, String>>,
}

impl DetectorErrorModel {
    pub(crate) fn from_program(program: DemProgram) -> DetectorErrorModel {
        DetectorErrorModel { stats: program.stats(), program, flat: OnceLock::new() }
    }

    /// The model as written.
    pub(crate) fn program(&self) -> &DemProgram {
        &self.program
    }

    /// The model unrolled, for a decoder.
    pub(crate) fn flat(&self) -> Result<&Dem> {
        self.flat.get_or_init(|| self.program.to_dem()).as_ref().map_err(|e| Error::new(e.clone()))
    }

    /// Parse Stim's detector-error-model text.
    pub fn parse(text: &str) -> Result<DetectorErrorModel> {
        Ok(DetectorErrorModel::from_program(DemProgram::parse(text)?))
    }

    /// Detectors: one more than the largest index named, counted through loops.
    pub fn num_detectors(&self) -> usize {
        usize::try_from(self.stats.num_detectors).unwrap_or(usize::MAX)
    }

    /// Logical observables.
    pub fn num_observables(&self) -> usize {
        self.stats.num_observables
    }

    /// Faults: `error` lines, counted through loops.
    pub fn num_errors(&self) -> usize {
        usize::try_from(self.stats.num_errors).unwrap_or(usize::MAX)
    }


    /// The matching graph as an SVG picture: each detector at its coordinates, each graph-like
    /// fault an edge, those flipping an observable heavier. The model must be decomposed, and
    /// unroll.
    pub fn matchgraph_svg(&self) -> Result<String> {
        Ok(crate::diagram::matchgraph_svg(self.flat()?)?)
    }

    /// The model without `repeat` blocks or `shift_detectors`, as Stim's `flattened`: every
    /// detector absolute, every coordinate shifted. The error is a model that unrolls past 2²⁴
    /// faults and declarations.
    ///
    /// ```
    /// use stabilizer_qec::DetectorErrorModel;
    ///
    /// let folded: DetectorErrorModel = "repeat 2 {\n error(0.125) D0\n shift_detectors 1\n}".parse()?;
    /// assert_eq!(folded.flattened()?.to_string(), "error(0.125) D0\nerror(0.125) D1\n");
    /// # Ok::<(), stabilizer_qec::Error>(())
    /// ```
    pub fn flattened(&self) -> Result<DetectorErrorModel> {
        Ok(DetectorErrorModel::from_program(self.program.flattened()?))
    }
}

impl FromStr for DetectorErrorModel {
    type Err = Error;

    fn from_str(text: &str) -> Result<DetectorErrorModel> {
        DetectorErrorModel::parse(text)
    }
}

impl fmt::Display for DetectorErrorModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.program.to_stim())
    }
}

/// Shots of detection events and of observable flips, one row per shot.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct Samples {
    /// One bit per detector.
    pub detectors: BitTable,
    /// One bit per observable.
    pub observables: BitTable,
}

/// Detection events and observable flips, 64 shots to a machine word. Made by
/// [`Circuit::detector_sampler`].
///
/// ```
/// use stabilizer_qec::Circuit;
///
/// let c: Circuit = "R 0\nX_ERROR(0.5) 0\nM 0\nDETECTOR rec[-1]".parse()?;
/// let a = c.detector_sampler(7)?.sample(1000, 1);
/// let b = c.detector_sampler(7)?.sample(1000, 4);
/// assert_eq!(a, b); // the same seed, the same shots, on any number of threads
/// # Ok::<(), stabilizer_qec::Error>(())
/// ```
pub struct DetectorSampler {
    sampler: BatchSampler,
    seed: u64,
    next: u64,
}

impl DetectorSampler {
    /// Detectors per shot.
    pub fn num_detectors(&self) -> usize {
        self.sampler.num_detectors
    }

    /// Observables per shot.
    pub fn num_observables(&self) -> usize {
        self.sampler.num_observables
    }

    /// `shots` more shots. `threads = 0` uses every core; the shots do not depend on it. Shots
    /// are drawn 64 at a time from streams seeded by the sampler's seed and the batch, and a
    /// call that ends partway through 64 discards the rest.
    pub fn sample(&mut self, shots: usize, threads: usize) -> Samples {
        let (d, o) = self.sampler.sample_seeded(self.seed, self.next, shots, threads);
        self.next += shots.div_ceil(64) as u64;
        Samples {
            detectors: BitTable::from_packed(shots, self.num_detectors(), d).expect("the sampler writes whole rows"),
            observables: BitTable::from_packed(shots, self.num_observables(), o).expect("the sampler writes whole rows"),
        }
    }
}

/// Raw measurement records to detection events and observable flips, as `stim m2d`: each
/// detector and observable compared with a noiseless reference run. Made by
/// [`Circuit::measurement_converter`].
pub struct MeasurementConverter {
    m2d: M2d,
}

impl MeasurementConverter {
    /// Measurement records per shot.
    pub fn num_measurements(&self) -> usize {
        self.m2d.num_measurements
    }

    /// Sweep bits per shot.
    pub fn num_sweep_bits(&self) -> usize {
        self.m2d.num_sweep_bits
    }

    /// Convert shots of measurement records (and, if the circuit reads any, sweep bits).
    pub fn convert(&self, measurements: &BitTable, sweep_bits: Option<&BitTable>) -> Result<Samples> {
        let shots = measurements.num_rows();
        if measurements.num_bits() != self.num_measurements() {
            return Err(Error::new(format!("{} measurement bits per shot, not {}", measurements.num_bits(), self.num_measurements())));
        }
        let zeros;
        let sweeps = match sweep_bits {
            Some(s) => {
                if s.num_rows() != shots || s.num_bits() != self.num_sweep_bits() {
                    return Err(Error::new(format!(
                        "sweep bits are {} shots of {}, not {shots} of {}",
                        s.num_rows(),
                        s.num_bits(),
                        self.num_sweep_bits()
                    )));
                }
                s
            }
            None => {
                zeros = BitTable::zeros(shots, self.num_sweep_bits());
                &zeros
            }
        };
        let (d, o) = self.m2d.convert_b8(measurements.as_bytes(), sweeps.as_bytes(), shots)?;
        Ok(Samples {
            detectors: BitTable::from_packed(shots, self.m2d.num_detectors, d)?,
            observables: BitTable::from_packed(shots, self.m2d.num_observables, o)?,
        })
    }
}

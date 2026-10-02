use std::fmt;
use std::str::FromStr;

use super::{BitTable, Error, Result};
use crate::batch_sampler::BatchSampler;
use crate::circuit::{self, Control, Instr};
use crate::dem::Dem;
use crate::m2d::M2d;

/// A stabilizer circuit in Stim's circuit language.
///
/// Parse one from Stim's text (`str::parse` or [`Circuit::parse`]); [`Display`](fmt::Display)
/// prints it back as written. Every Clifford gate of Stim's is read (H, S and CX natively, the
/// rest as their exact decompositions), with resets and measurements in all three bases,
/// inverted targets, Pauli-product measurements and rotations, every noise channel,
/// measurement feedback and sweep bits, `MPAD`, detectors, observables, coordinates, `TICK`
/// and `REPEAT`. Counts are taken through `REPEAT` blocks without unrolling them.
///
/// ```
/// use stabilizer_qec::Circuit;
///
/// let c: Circuit = "R 0 1\nH 0\nCX 0 1\nM 0 1\nDETECTOR rec[-1] rec[-2]".parse()?;
/// assert_eq!((c.num_qubits(), c.num_measurements(), c.num_detectors()), (2, 2, 1));
/// assert_eq!(c.to_string().parse::<Circuit>()?, c);
/// # Ok::<(), stabilizer_qec::Error>(())
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct Circuit {
    pub(crate) inner: circuit::Circuit,
    num_qubits: usize,
    num_measurements: usize,
    num_detectors: usize,
    num_observables: usize,
    num_sweep_bits: usize,
}

fn max_sweep_bit(instrs: &[Instr]) -> usize {
    instrs
        .iter()
        .map(|i| match i {
            Instr::SweepX(pairs) => pairs.iter().map(|&(k, _)| k as usize + 1).max().unwrap_or(0),
            Instr::Feedback { control: Control::Sweep(k), .. } => *k as usize + 1,
            Instr::Repeat { body, .. } | Instr::Gate { body, .. } => max_sweep_bit(body),
            _ => 0,
        })
        .max()
        .unwrap_or(0)
}

impl Circuit {
    /// Parse Stim's circuit text.
    pub fn parse(text: &str) -> Result<Circuit> {
        Circuit::from_engine(circuit::Circuit::parse(text)?)
    }

    pub(crate) fn from_engine(inner: circuit::Circuit) -> Result<Circuit> {
        let shape = BatchSampler::new(&inner)?;
        Ok(Circuit {
            num_qubits: inner.instrs.iter().flat_map(Instr::qubits).map(|q| q as usize + 1).max().unwrap_or(0),
            num_measurements: shape.num_measurements,
            num_detectors: shape.num_detectors,
            num_observables: shape.num_observables,
            num_sweep_bits: max_sweep_bit(&inner.instrs),
            inner,
        })
    }

    /// One more than the largest qubit index any instruction names.
    pub fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    /// Measurement records per shot.
    pub fn num_measurements(&self) -> usize {
        self.num_measurements
    }

    /// Detectors per shot.
    pub fn num_detectors(&self) -> usize {
        self.num_detectors
    }

    /// Logical observables.
    pub fn num_observables(&self) -> usize {
        self.num_observables
    }

    /// Sweep bits the circuit reads (one more than the largest `sweep[k]`).
    pub fn num_sweep_bits(&self) -> usize {
        self.num_sweep_bits
    }

    /// The circuit's detector error model, built by walking it backwards as Stim's error analyzer
    /// does (see [`DemOptions`]).
    ///
    /// ```
    /// use stabilizer_qec::{Circuit, DemOptions};
    ///
    /// let c: Circuit = "R 0\nX_ERROR(0.125) 0\nM 0\nDETECTOR rec[-1]".parse()?;
    /// assert_eq!(c.detector_error_model(&DemOptions::new())?.to_string().trim(), "detector D0\nerror(0.125) D0");
    /// # Ok::<(), stabilizer_qec::Error>(())
    /// ```
    pub fn detector_error_model(&self, options: &DemOptions) -> Result<DetectorErrorModel> {
        let dem = Dem::from_circuit_with(&self.inner, options.decompose_errors, options.approximate_disjoint_errors)?;
        Ok(DetectorErrorModel { inner: dem, pieces: options.decompose_errors })
    }

    /// A sampler of detection events and observable flips, seeded: the same seed gives the same
    /// shots on any machine and any number of threads.
    pub fn detector_sampler(&self, seed: u64) -> Result<DetectorSampler> {
        Ok(DetectorSampler { sampler: BatchSampler::new(&self.inner)?, seed, next: 0 })
    }

    /// A converter from raw measurement records (and sweep bits) to detection events and
    /// observable flips, as `stim m2d`. Its reference run keeps a dense tableau: at most 16,384
    /// qubits.
    pub fn measurement_converter(&self) -> Result<MeasurementConverter> {
        Ok(MeasurementConverter { m2d: M2d::new(&self.inner)? })
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
}

/// A detector error model in Stim's format: independent faults, each with a probability, the
/// detectors it flips and the observables it flips. Parse one from Stim's text, or build one
/// with [`Circuit::detector_error_model`].
#[derive(Clone, Debug)]
pub struct DetectorErrorModel {
    pub(crate) inner: Dem,
    pieces: bool,
}

impl DetectorErrorModel {
    /// Parse Stim's detector-error-model text.
    pub fn parse(text: &str) -> Result<DetectorErrorModel> {
        Ok(DetectorErrorModel { inner: Dem::parse(text)?, pieces: true })
    }

    /// Detectors.
    pub fn num_detectors(&self) -> usize {
        self.inner.num_detectors
    }

    /// Logical observables.
    pub fn num_observables(&self) -> usize {
        self.inner.num_observables
    }

    /// Faults (`error` lines).
    pub fn num_errors(&self) -> usize {
        self.inner.mechanisms.len()
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
        f.write_str(&self.inner.to_stim(self.pieces))
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

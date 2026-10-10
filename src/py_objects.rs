//! The compiled objects behind the public Python API (`python/stabilizer_qec`). Each parses or
//! builds once and is reused; shots go in and results come out as Stim's b8 rows and
//! little-endian arrays, which the Python layer turns into numpy arrays; every loop over shots
//! releases the GIL. The shot loops here also serve the 0.4 functions in `py_api`.

// PyO3's #[pymethods] expansion converts each PyResult's error into PyErr, which clippy flags
// as a useless conversion; it is the macro's, not this code's.
#![allow(clippy::useless_conversion)]

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

use crate::batch_sampler::BatchSampler;
use crate::belief::BeliefMatching;
use crate::circuit::Circuit;
use crate::dem::Dem;
use crate::dem_decoder::DemDecoder;
use crate::m2d::M2d;
use crate::osd::{BpOsd, OsdMethod};
use crate::batch::{belief_shots, bposd_shots, match_shots, syndrome_shots, window_info, window_shots, WindowInfo};
use crate::lsd::BpLsd;
use crate::relay::{Relay, RelayConfig};
use crate::window::{Mode, Model, WindowDecoder};

fn err(e: String) -> PyErr {
    PyValueError::new_err(e)
}

/// A buffer must hold exactly `num_shots` rows of `stride` bytes.
pub(crate) fn check_rows(len: usize, stride: usize, num_shots: usize, bits: usize, what: &str) -> PyResult<()> {
    if stride.checked_mul(num_shots) != Some(len) {
        return Err(err(format!("{len} bytes is not {num_shots} shots of {bits} {what}")));
    }
    Ok(())
}

/// Bits as a list of ints for Python (a `Vec<u8>` would arrive as `bytes`).
pub(crate) fn bits(xs: &[u8]) -> Vec<u32> {
    xs.iter().map(|&b| u32::from(b)).collect()
}

fn le_u64(xs: impl IntoIterator<Item = u64>) -> Vec<u8> {
    xs.into_iter().flat_map(u64::to_le_bytes).collect()
}

fn le_f64(xs: impl IntoIterator<Item = f64>) -> Vec<u8> {
    xs.into_iter().flat_map(f64::to_le_bytes).collect()
}

/* -- Circuits and models --------------------------------------------------------- */

/// A circuit, its counts taken through loops without unrolling them: the crate's `Circuit`,
/// so the two APIs build, count and validate circuits alike.
#[pyclass(name = "Circuit", module = "stabilizer_qec._core")]
pub struct PyCircuit {
    circuit: crate::api::Circuit,
}

/// A target as (target, coordinates, Stim's text).
type PyTarget = (String, Vec<f64>, String);
/// An error location: (ticks before it, Pauli product, flipped measurement, (gate, tag, args),
/// target range, targets in range, stack frames, Stim's text, the instruction's text).
type PyLocation = (u64, Vec<PyTarget>, Option<(u64, Vec<PyTarget>)>, (String, String, Vec<f64>), (u32, u32), Vec<PyTarget>, Vec<(u64, u64, u64)>, String, String);
/// An explained error: (Stim's text, terms, locations).
type PyExplained = (String, Vec<PyTarget>, Vec<PyLocation>);

fn explained_tuple(e: crate::api::ExplainedError) -> PyExplained {
    let tw = |v: Vec<crate::api::TargetWithCoords>| v.into_iter().map(|t| (t.to_string(), t)).map(|(s, t)| (t.target, t.coords, s)).collect::<Vec<_>>();
    let locations = e
        .circuit_error_locations()
        .into_iter()
        .map(|l| {
            let (text, instruction) = (l.to_string(), l.instruction_text().to_string());
            (l.tick_offset, tw(l.flipped_pauli_product), l.flipped_measurement.map(|(i, o)| (i, tw(o))), (l.gate, l.tag, l.args), l.target_range, tw(l.targets_in_range), l.stack_frames, text, instruction)
        })
        .collect();
    (e.to_string(), tw(e.terms_with_coords()), locations)
}

fn api_err(e: crate::api::Error) -> PyErr {
    PyValueError::new_err(e.to_string())
}

impl PyCircuit {
    /// The engine's circuit.
    pub(crate) fn engine(&self) -> &Circuit {
        &self.circuit.inner
    }
}

#[pymethods]
impl PyCircuit {
    #[new]
    fn new(text: &str) -> PyResult<Self> {
        Ok(PyCircuit { circuit: crate::api::Circuit::parse(text).map_err(api_err)? })
    }

    #[getter]
    fn num_qubits(&self) -> usize {
        self.circuit.num_qubits()
    }

    #[getter]
    fn num_sweep_bits(&self) -> usize {
        self.circuit.num_sweep_bits()
    }

    #[getter]
    fn num_measurements(&self) -> usize {
        self.circuit.num_measurements()
    }

    #[getter]
    fn num_detectors(&self) -> usize {
        self.circuit.num_detectors()
    }

    #[getter]
    fn num_observables(&self) -> usize {
        self.circuit.num_observables()
    }

    fn __str__(&self) -> String {
        self.circuit.to_string()
    }

    fn __eq__(&self, other: &PyCircuit) -> bool {
        self.circuit.inner == other.circuit.inner
    }

    /// One instruction from its parts (targets as their text); the circuit is unchanged on an
    /// error.
    fn append_instruction(&mut self, name: &str, tag: &str, args: Vec<f64>, targets: Vec<String>) -> PyResult<()> {
        let piece = Circuit::instruction(name, tag, &args, &targets).map_err(err)?;
        let piece = crate::api::Circuit::from_engine(piece).map_err(api_err)?;
        self.circuit.append_circuit(&piece).map_err(api_err)
    }

    fn append_text(&mut self, text: &str) -> PyResult<()> {
        self.circuit.append_text(text).map_err(api_err)
    }

    fn append_circuit(&mut self, other: &PyCircuit) -> PyResult<()> {
        self.circuit.append_circuit(&other.circuit).map_err(api_err)
    }

    /// Coherent rotations replaced by their Pauli twirls (with `merge`, equivalent rotations
    /// added up first).
    fn twirled(&self, merge: bool) -> PyResult<PyCircuit> {
        let inner = crate::coherent::twirled(&self.circuit.inner, merge).map_err(err)?;
        Ok(PyCircuit { circuit: crate::api::Circuit::from_engine(inner).map_err(api_err)? })
    }

    fn repeated(&self, count: u64) -> PyResult<PyCircuit> {
        Ok(PyCircuit { circuit: self.circuit.repeated(count).map_err(api_err)? })
    }

    /// The circuit's faults explained (see `Circuit::explain_errors`): per explained error,
    /// (Stim's text, its terms, its locations), targets as (text, coordinates).
    #[pyo3(signature = (filter=None, reduce=false))]
    fn explain(&self, py: Python<'_>, filter: Option<PyRef<'_, PyDem>>, reduce: bool) -> PyResult<Vec<PyExplained>> {
        let filter = filter.map(|f| f.dem.clone());
        let circuit = &self.circuit;
        let out = py.detach(move || circuit.explain_errors(filter.as_ref(), reduce)).map_err(api_err)?;
        Ok(out.into_iter().map(explained_tuple).collect())
    }

    /// The graph-like distance's faults, explained (see `Circuit::shortest_graphlike_error`).
    fn shortest_graphlike(&self, py: Python<'_>, ignore: bool, canonicalize: bool) -> PyResult<Vec<PyExplained>> {
        let circuit = &self.circuit;
        let out = py.detach(move || circuit.shortest_graphlike_error(ignore, canonicalize)).map_err(api_err)?;
        Ok(out.into_iter().map(explained_tuple).collect())
    }

    /// Stim's search for undetectable logical errors, explained (see
    /// `Circuit::search_for_undetectable_logical_errors`).
    fn search_undetectable(&self, py: Python<'_>, max_symptoms: usize, max_degree: usize, no_increase: bool, canonicalize: bool) -> PyResult<Vec<PyExplained>> {
        let circuit = &self.circuit;
        let out = py.detach(move || circuit.search_for_undetectable_logical_errors(max_symptoms, max_degree, no_increase, canonicalize)).map_err(api_err)?;
        Ok(out.into_iter().map(explained_tuple).collect())
    }

    /// A diagram: `kind` one of Stim's names (timeline-text, timeline-svg, detslice-text,
    /// detslice-svg, timeslice-svg, detslice-with-ops-svg, matchgraph-svg); `tick` the moment of
    /// a slice, or with `tick_end` the range `[tick, tick_end)`; `rows` the panels' rows.
    /// `filters` are Stim's `filter_coords`: (target as (is_observable, index), or None with a
    /// coordinate prefix, NaN matching anything).
    #[pyo3(signature = (kind, tick=None, tick_end=None, rows=None, filters=None))]
    #[allow(clippy::type_complexity)]
    fn diagram(&self, kind: &str, tick: Option<u64>, tick_end: Option<u64>, rows: Option<u32>, filters: Option<Vec<(Option<(bool, u64)>, Vec<f64>)>>) -> PyResult<String> {
        use crate::api::DiagramKind as K;
        let need_tick = || tick.ok_or_else(|| err(format!("a {kind} diagram needs tick=")));
        let range = || -> PyResult<(u64, u64)> {
            let t = need_tick()?;
            Ok((t, tick_end.unwrap_or(t + 1)))
        };
        let kind = match kind {
            "timeline-text" => K::TimelineText,
            "timeline-svg" => K::TimelineSvg,
            "detslice-text" => K::DetectorSliceText { tick: need_tick()? },
            "detslice-svg" if tick_end.is_none() && rows.is_none() => K::DetectorSliceSvg { tick: need_tick()? },
            "detslice-svg" => K::DetectorSlicesSvg { ticks: range()?, rows },
            "timeslice-svg" => K::TimeSliceSvg { ticks: range()?, rows },
            "detslice-with-ops-svg" => K::DetectorSliceWithOpsSvg { ticks: range()?, rows },
            "matchgraph-svg" => K::MatchGraphSvg,
            other => return Err(err(format!("unknown diagram '{other}': timeline-text, timeline-svg, detslice-text, detslice-svg, timeslice-svg, detslice-with-ops-svg or matchgraph-svg"))),
        };
        match filters {
            None => self.circuit.diagram(kind).map_err(api_err),
            Some(f) => self.circuit.diagram_with_filter(kind, &crate::py_stim::coord_filters(f)).map_err(api_err),
        }
    }

    fn copy(&self) -> PyCircuit {
        PyCircuit { circuit: self.circuit.clone() }
    }

    /// The detector error model; `decompose` splits faults into graph-like pieces as Stim does,
    /// `approximate` is Stim's `approximate_disjoint_errors` as a threshold (None: off), and
    /// `flatten` writes it without folding its loops; `ignore_failures` keeps faults that cannot
    /// be decomposed whole.
    #[pyo3(signature = (decompose, approximate=None, flatten=false, ignore_failures=false))]
    fn detector_error_model(&self, decompose: bool, approximate: Option<f64>, flatten: bool, ignore_failures: bool) -> PyResult<PyDem> {
        let options = crate::api::DemOptions::new().decompose_errors(decompose).approximate_disjoint_errors(approximate).flatten_loops(flatten).ignore_decomposition_failures(ignore_failures);
        Ok(PyDem { dem: self.circuit.detector_error_model(&options).map_err(api_err)? })
    }

    /// The noiseless reference run's measurement record (the one the converter compares with),
    /// as one b8 row.
    fn reference_sample<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let inner = &self.circuit.inner;
        let qubits = crate::batch_sampler::Counts::of(&inner.instrs).map_err(err)?.qubits;
        let bits = py.detach(|| crate::m2d::run(inner, qubits, &[], 1));
        let mut row = vec![0u8; bits.len().div_ceil(8)];
        for (k, &b) in bits.iter().enumerate() {
            if b {
                row[k / 8] |= 1 << (k % 8);
            }
        }
        Ok(PyBytes::new(py, &row))
    }

    /// A sampler of raw measurement records (`skip_reference`: flips from all zeros).
    fn measurement_sampler(&self, py: Python<'_>, seed: u64, skip_reference: bool) -> PyResult<PyMeasurementSampler> {
        let inner = &self.circuit.inner;
        let sampler = BatchSampler::new(inner).map_err(err)?;
        let reference = if skip_reference {
            Vec::new()
        } else {
            let qubits = crate::batch_sampler::Counts::of(&inner.instrs).map_err(err)?.qubits;
            py.detach(|| crate::m2d::run(inner, qubits, &[], 1))
        };
        Ok(PyMeasurementSampler { sampler, reference, seed, next: 0.into() })
    }

    fn sampler(&self, seed: u64) -> PyResult<PySampler> {
        Ok(PySampler { sampler: BatchSampler::new(&self.circuit.inner).map_err(err)?, seed, next: 0.into() })
    }

    fn m2d(&self) -> PyResult<PyM2d> {
        Ok(PyM2d { m2d: M2d::new(&self.circuit.inner).map_err(err)? })
    }

    /// The per-shot frame sampler, which runs leakage.
    #[pyo3(signature = (seed, leaked_reads_one=true))]
    fn leakage_sampler(&self, seed: u64, leaked_reads_one: bool) -> PyResult<PyLeakageSampler> {
        let mut sampler = crate::frame_sampler::FrameSampler::new(&self.circuit.inner).map_err(err)?;
        sampler.leaked_reads_one = leaked_reads_one;
        Ok(PyLeakageSampler { sampler, seed, next: 0.into() })
    }

    /// The exact state-vector sampler.
    fn exact_sampler(&self, seed: u64) -> PyResult<PyExactSampler> {
        Ok(PyExactSampler { program: crate::statevec::Program::new(&self.circuit.inner).map_err(err)?, seed, next: 0.into() })
    }

    /// The coherent sampler: the twirl, reweighted by the interference it leaves out.
    #[pyo3(signature = (seed, order=3, max_cluster=12, quads=true, hops=1))]
    fn coherent_sampler(&self, py: Python<'_>, seed: u64, order: usize, max_cluster: usize, quads: bool, hops: usize) -> PyResult<PyCoherentSampler> {
        let options = crate::coherent::kernel::CoherentOptions { order, max_cluster, quads, hops };
        let inner = &self.circuit.inner;
        let sampler = py.detach(|| crate::coherent::sampler::CoherentSampler::new(inner, options)).map_err(err)?;
        Ok(PyCoherentSampler { sampler, seed, next: 0.into() })
    }

    /// `exact_distribution` with the leaked-measurement rule chosen.
    fn exact_distribution_leaky(&self, py: Python<'_>, max_branches: usize, reads_one: bool) -> PyResult<Vec<(Vec<bool>, u64, f64)>> {
        let mut program = crate::statevec::Program::new(&self.circuit.inner).map_err(err)?;
        program.leaked_reads_one = reads_one;
        let d = py.detach(|| program.distribution(max_branches)).map_err(err)?;
        Ok(d.into_iter().map(|((dets, obs), p)| (dets, obs, p)).collect())
    }

    /// Every outcome's exact probability: (detection events, observable flips, probability).
    fn exact_distribution(&self, py: Python<'_>, max_branches: usize) -> PyResult<Vec<(Vec<bool>, u64, f64)>> {
        let program = crate::statevec::Program::new(&self.circuit.inner).map_err(err)?;
        let d = py.detach(|| program.distribution(max_branches)).map_err(err)?;
        Ok(d.into_iter().map(|((dets, obs), p)| (dets, obs, p)).collect())
    }
}

/// The per-shot frame sampler with leakage, one stream per shot.
#[pyclass(name = "LeakageSampler", module = "stabilizer_qec._core")]
pub struct PyLeakageSampler {
    sampler: crate::frame_sampler::FrameSampler,
    seed: u64,
    next: std::sync::atomic::AtomicU64,
}

#[pymethods]
impl PyLeakageSampler {
    #[getter]
    fn num_detectors(&self) -> usize {
        self.sampler.num_detectors()
    }

    #[getter]
    fn num_observables(&self) -> usize {
        self.sampler.num_observables()
    }

    /// (detection events b8, observable flips b8, heralds as one byte per measurement).
    fn sample<'py>(&self, py: Python<'py>, shots: usize, threads: usize) -> (Bound<'py, PyBytes>, Bound<'py, PyBytes>, Bound<'py, PyBytes>) {
        let first = self.next.fetch_add(shots as u64, std::sync::atomic::Ordering::Relaxed);
        let (sampler, seed) = (&self.sampler, self.seed);
        let rows = py.detach(|| {
            crate::parallel::parallel(shots, threads, |range| {
                range
                    .map(|s| {
                        let mut rng = crate::surface_code::Xorshift::new(crate::batch_sampler::batch_seed(seed, first + s as u64));
                        sampler.sample(&mut rng)
                    })
                    .collect()
            })
        });
        let (nd, no) = (sampler.num_detectors(), sampler.num_observables());
        let (dw, ow) = (nd.div_ceil(8), no.div_ceil(8));
        let (mut d, mut o, mut h) = (vec![0u8; shots * dw], vec![0u8; shots * ow], Vec::new());
        for (s, shot) in rows.iter().enumerate() {
            for (k, &b) in shot.detectors.iter().enumerate() {
                if b {
                    d[s * dw + k / 8] |= 1 << (k % 8);
                }
            }
            for k in 0..no {
                if shot.observables >> k & 1 == 1 {
                    o[s * ow + k / 8] |= 1 << (k % 8);
                }
            }
            h.extend(shot.heralds.iter().map(|&b| b as u8));
        }
        (PyBytes::new(py, &d), PyBytes::new(py, &o), PyBytes::new(py, &h))
    }
}

/// The coherent sampler, one stream per shot.
#[pyclass(name = "CoherentSampler", module = "stabilizer_qec._core")]
pub struct PyCoherentSampler {
    sampler: crate::coherent::sampler::CoherentSampler,
    seed: u64,
    next: std::sync::atomic::AtomicU64,
}

#[pymethods]
impl PyCoherentSampler {
    #[getter]
    fn num_detectors(&self) -> usize {
        self.sampler.program.num_detectors()
    }

    #[getter]
    fn num_observables(&self) -> usize {
        self.sampler.program.num_observables()
    }

    #[getter]
    fn num_locations(&self) -> usize {
        self.sampler.program.locations.len()
    }

    #[getter]
    fn num_generators(&self) -> usize {
        self.sampler.kernel.generators.len()
    }

    fn sample<'py>(&self, py: Python<'py>, shots: usize, threads: usize) -> (Bound<'py, PyBytes>, Bound<'py, PyBytes>, Vec<f64>) {
        let first = self.next.fetch_add(shots as u64, std::sync::atomic::Ordering::Relaxed);
        let (sampler, seed) = (&self.sampler, self.seed);
        let b = py.detach(|| sampler.sample_seeded(seed, first, shots, threads));
        (PyBytes::new(py, &b.detectors), PyBytes::new(py, &b.observables), b.weights)
    }
}

/// The exact state-vector sampler, one stream per shot.
#[pyclass(name = "ExactSampler", module = "stabilizer_qec._core")]
pub struct PyExactSampler {
    program: crate::statevec::Program,
    seed: u64,
    next: std::sync::atomic::AtomicU64,
}

#[pymethods]
impl PyExactSampler {
    #[getter]
    fn num_detectors(&self) -> usize {
        self.program.num_detectors()
    }

    #[getter]
    fn num_observables(&self) -> usize {
        self.program.num_observables()
    }

    #[getter]
    fn num_qubits(&self) -> usize {
        self.program.num_qubits()
    }

    fn sample<'py>(&self, py: Python<'py>, shots: usize, threads: usize) -> (Bound<'py, PyBytes>, Bound<'py, PyBytes>) {
        let first = self.next.fetch_add(shots as u64, std::sync::atomic::Ordering::Relaxed);
        let (program, seed) = (&self.program, self.seed);
        let (d, o) = py.detach(|| crate::statevec::sample_seeded(program, seed, first, shots, threads));
        (PyBytes::new(py, &d), PyBytes::new(py, &o))
    }
}

/// A detector error model, parsed from Stim's text or built from a circuit: the crate's, held
/// folded and unrolled once for a decoder.
#[pyclass(name = "Dem", module = "stabilizer_qec._core")]
pub struct PyDem {
    dem: crate::api::DetectorErrorModel,
}

impl PyDem {
    /// The model unrolled, for a decoder.
    fn flat(&self) -> PyResult<&Dem> {
        self.dem.flat().map_err(api_err)
    }
}

#[pymethods]
impl PyDem {
    #[new]
    fn new(text: &str) -> PyResult<Self> {
        Ok(PyDem { dem: crate::api::DetectorErrorModel::parse(text).map_err(api_err)? })
    }

    #[getter]
    fn num_detectors(&self) -> usize {
        self.dem.num_detectors()
    }

    #[getter]
    fn num_observables(&self) -> usize {
        self.dem.num_observables()
    }

    #[getter]
    fn num_errors(&self) -> usize {
        self.dem.num_errors()
    }

    fn __str__(&self) -> String {
        self.dem.to_string()
    }

    /// The graph-like distance's faults as a model (see
    /// `DetectorErrorModel::shortest_graphlike_error`).
    fn shortest_graphlike(&self, py: Python<'_>, ignore: bool) -> PyResult<PyDem> {
        let dem = &self.dem;
        Ok(PyDem { dem: py.detach(move || dem.shortest_graphlike_error(ignore)).map_err(api_err)? })
    }

    /// Stim's search for undetectable logical errors on the model.
    fn search_undetectable(&self, py: Python<'_>, max_symptoms: usize, max_degree: usize, no_increase: bool) -> PyResult<PyDem> {
        let dem = &self.dem;
        Ok(PyDem { dem: py.detach(move || dem.search_for_undetectable_logical_errors(max_symptoms, max_degree, no_increase)).map_err(api_err)? })
    }

    /// A sampler of the model's faults (see `DemSampler`).
    fn sampler(&self, seed: u64) -> PyResult<PyDemSampler> {
        Ok(PyDemSampler { sampler: crate::dem_sampler::DemSampler::new(self.dem.flat().map_err(api_err)?), seed, next: 0.into() })
    }

    fn matchgraph_svg(&self) -> PyResult<String> {
        self.dem.matchgraph_svg().map_err(api_err)
    }

    fn flattened(&self) -> PyResult<PyDem> {
        Ok(PyDem { dem: self.dem.flattened().map_err(api_err)? })
    }

    /// Stim's `matchgraph-3d` diagram (glTF).
    fn matchgraph_3d(&self) -> PyResult<String> {
        crate::clifford::gltf::matchgraph_3d(self.dem.program()).map_err(PyValueError::new_err)
    }

    /// Every error of the unrolled model, in order: its exact probability and its targets as
    /// (is_observable, index), detectors absolute, pieces concatenated.
    #[allow(clippy::type_complexity)]
    fn flat_errors(&self) -> PyResult<Vec<(f64, Vec<(bool, u64)>)>> {
        use crate::dem_program::{DemInstr, OBS};
        let flat = self.dem.program().flattened().map_err(PyValueError::new_err)?;
        Ok(flat
            .instrs
            .iter()
            .filter_map(|i| match i {
                DemInstr::Error { p, pieces, .. } => Some((*p, pieces.iter().flatten().map(|&t| (t & OBS != 0, t & !OBS)).collect())),
                _ => None,
            })
            .collect())
    }
}

/* -- Sampling and conversion ------------------------------------------------------ */

/// The bit-parallel sampler, drawing batch after batch from per-batch streams.
#[pyclass(name = "Sampler", module = "stabilizer_qec._core")]
pub struct PySampler {
    sampler: BatchSampler,
    seed: u64,
    /// The next batch's index: the shots drawn so far, in batches of 64. Each call reserves
    /// its batches before it samples, so threads sharing the sampler draw disjoint batches.
    next: std::sync::atomic::AtomicU64,
}

#[pymethods]
impl PySampler {
    #[getter]
    fn num_detectors(&self) -> usize {
        self.sampler.num_detectors
    }

    #[getter]
    fn num_observables(&self) -> usize {
        self.sampler.num_observables
    }

    /// `shots` shots as b8 rows of detectors and of observables. A call that ends mid-batch
    /// discards the batch's other lanes; the next call starts a new batch.
    fn sample<'py>(&self, py: Python<'py>, shots: usize, threads: usize) -> (Bound<'py, PyBytes>, Bound<'py, PyBytes>) {
        let first = self.next.fetch_add(shots.div_ceil(64) as u64, std::sync::atomic::Ordering::Relaxed);
        let (sampler, seed) = (&self.sampler, self.seed);
        let (d, o) = py.detach(|| sampler.sample_seeded(seed, first, shots, threads));
        (PyBytes::new(py, &d), PyBytes::new(py, &o))
    }
}

/// Raw measurement records, batch after batch: a noiseless reference run with each shot's flips.
#[pyclass(name = "MeasurementSampler", module = "stabilizer_qec._core")]
pub struct PyMeasurementSampler {
    sampler: BatchSampler,
    reference: Vec<bool>,
    seed: u64,
    next: std::sync::atomic::AtomicU64,
}

#[pymethods]
impl PyMeasurementSampler {
    #[getter]
    fn num_measurements(&self) -> usize {
        self.sampler.num_measurements
    }

    /// `shots` measurement records as b8 rows.
    fn sample<'py>(&self, py: Python<'py>, shots: usize, threads: usize) -> Bound<'py, PyBytes> {
        let first = self.next.fetch_add(shots.div_ceil(64) as u64, std::sync::atomic::Ordering::Relaxed);
        let (sampler, seed, reference) = (&self.sampler, self.seed, &self.reference);
        let rows = py.detach(|| sampler.sample_measurements_seeded(seed, first, shots, threads, reference));
        PyBytes::new(py, &rows)
    }
}

/// A detector error model's own sampler, batch after batch from per-batch streams.
#[pyclass(name = "DemSampler", module = "stabilizer_qec._core")]
pub struct PyDemSampler {
    sampler: crate::dem_sampler::DemSampler,
    seed: u64,
    next: std::sync::atomic::AtomicU64,
}

#[pymethods]
impl PyDemSampler {
    #[getter]
    fn num_detectors(&self) -> usize {
        self.sampler.num_detectors
    }

    #[getter]
    fn num_observables(&self) -> usize {
        self.sampler.num_observables
    }

    #[getter]
    fn num_errors(&self) -> usize {
        self.sampler.num_errors()
    }

    /// `shots` shots as b8 rows of detectors, of observables, and (with `errors`) of the faults
    /// that fired.
    fn sample<'py>(&self, py: Python<'py>, shots: usize, threads: usize, errors: bool) -> (Bound<'py, PyBytes>, Bound<'py, PyBytes>, Bound<'py, PyBytes>) {
        let first = self.next.fetch_add(shots.div_ceil(64) as u64, std::sync::atomic::Ordering::Relaxed);
        let (sampler, seed) = (&self.sampler, self.seed);
        let (d, o, e) = py.detach(|| sampler.sample_seeded(seed, first, shots, threads, errors));
        (PyBytes::new(py, &d), PyBytes::new(py, &o), PyBytes::new(py, &e))
    }

    /// The detection events and observable flips of recorded faults (b8 rows).
    fn replay<'py>(&self, py: Python<'py>, errors: &[u8], shots: usize) -> PyResult<(Bound<'py, PyBytes>, Bound<'py, PyBytes>)> {
        let (d, o) = self.sampler.replay(errors, shots).map_err(err)?;
        Ok((PyBytes::new(py, &d), PyBytes::new(py, &o)))
    }
}

/// Raw measurements and sweep bits to detection events and observable flips.
#[pyclass(name = "M2d", module = "stabilizer_qec._core")]
pub struct PyM2d {
    m2d: M2d,
}

#[pymethods]
impl PyM2d {
    #[getter]
    fn num_measurements(&self) -> usize {
        self.m2d.num_measurements
    }

    #[getter]
    fn num_sweep_bits(&self) -> usize {
        self.m2d.num_sweep_bits
    }

    #[getter]
    fn num_detectors(&self) -> usize {
        self.m2d.num_detectors
    }

    #[getter]
    fn num_observables(&self) -> usize {
        self.m2d.num_observables
    }

    fn convert<'py>(&self, py: Python<'py>, meas: &[u8], sweeps: &[u8], shots: usize) -> PyResult<(Bound<'py, PyBytes>, Bound<'py, PyBytes>)> {
        let m2d = &self.m2d;
        let (d, o) = py.detach(|| m2d.convert_b8(meas, sweeps, shots)).map_err(err)?;
        Ok((PyBytes::new(py, &d), PyBytes::new(py, &o)))
    }
}

/* -- Decoders on a detector error model ----------------------------------------------- */

/// Exact matching, plain or correlated, on a model's graph.
#[pyclass(name = "Matcher", module = "stabilizer_qec._core")]
pub struct PyMatcher {
    // The graph and correlations alone, as the crate's `Matching` holds them: they are `Sync`,
    // so the matcher can be shared between threads (free-threaded Python included).
    graph: crate::sparse::SparseGraph,
    corr: Option<crate::sparse::Correlations>,
    #[pyo3(get)]
    num_detectors: usize,
    #[pyo3(get)]
    num_observables: usize,
}

#[pymethods]
impl PyMatcher {
    #[new]
    fn new(dem: &PyDem, correlated: bool) -> PyResult<Self> {
        let d = dem.flat()?;
        let (graph, corr) = DemDecoder::new(d).map_err(err)?.into_parts(correlated);
        Ok(PyMatcher { graph, corr, num_detectors: d.num_detectors, num_observables: d.num_observables })
    }

    /// (observables as u64 per shot, weights as f64 per shot, the shots with no matching).
    fn decode_batch<'py>(&self, py: Python<'py>, packed: &[u8], shots: usize, threads: usize) -> PyResult<(Bound<'py, PyBytes>, Bound<'py, PyBytes>, Vec<usize>)> {
        let nd = self.num_detectors;
        check_rows(packed.len(), nd.div_ceil(8), shots, nd, "detectors")?;
        let out = py.detach(|| match_shots(&self.graph, self.corr.as_ref(), packed, nd, shots, threads));
        let failed = out.iter().enumerate().filter(|(_, (_, w))| w.is_nan()).map(|(s, _)| s).collect();
        let preds = le_u64(out.iter().map(|&(o, _)| o));
        let weights = le_f64(out.iter().map(|&(_, w)| w));
        Ok((PyBytes::new(py, &preds), PyBytes::new(py, &weights), failed))
    }
}

/// Weighted union-find on a decomposed model's matching graph.
#[pyclass(name = "UnionFinder", module = "stabilizer_qec._core")]
pub struct PyUnionFinder {
    graph: crate::sparse::SparseGraph,
    #[pyo3(get)]
    num_detectors: usize,
    #[pyo3(get)]
    num_observables: usize,
}

#[pymethods]
impl PyUnionFinder {
    #[new]
    fn new(dem: &PyDem) -> PyResult<Self> {
        let d = dem.flat()?;
        let (graph, _) = DemDecoder::new(d).map_err(err)?.into_parts(false);
        Ok(PyUnionFinder { graph, num_detectors: d.num_detectors, num_observables: d.num_observables })
    }

    /// (observables as u64 per shot, the shots with no correction).
    fn decode_batch<'py>(&self, py: Python<'py>, packed: &[u8], shots: usize, threads: usize) -> PyResult<(Bound<'py, PyBytes>, Vec<usize>)> {
        let nd = self.num_detectors;
        check_rows(packed.len(), nd.div_ceil(8), shots, nd, "detectors")?;
        let out = py.detach(|| crate::batch::union_find_shots(&self.graph, packed, nd, shots, threads));
        let failed = out.iter().enumerate().filter(|(_, o)| o.is_none()).map(|(s, _)| s).collect();
        Ok((PyBytes::new(py, &le_u64(out.iter().map(|o| o.unwrap_or(0)))), failed))
    }
}

fn bp_method(method: &str, scale: f64) -> PyResult<crate::bp::Method> {
    match method {
        "product_sum" => Ok(crate::bp::Method::ProductSum),
        "minimum_sum" => Ok(crate::bp::Method::MinSum { scale }),
        other => Err(err(format!("BP method '{other}' is neither product_sum nor minimum_sum"))),
    }
}

fn osd_method(name: &str, order: usize) -> PyResult<OsdMethod> {
    match name {
        "osd_0" | "osd0" => Ok(OsdMethod::Osd0),
        "osd_e" => Ok(OsdMethod::Exhaustive(order)),
        "osd_cs" => Ok(OsdMethod::CombinationSweep(order)),
        other => Err(err(format!("OSD method '{other}' is not osd_0, osd_e or osd_cs"))),
    }
}

/// Belief-matching on a decomposed model.
#[pyclass(name = "BeliefMatcher", module = "stabilizer_qec._core")]
pub struct PyBeliefMatcher {
    bm: BeliefMatching,
    #[pyo3(get)]
    num_detectors: usize,
    #[pyo3(get)]
    num_observables: usize,
}

#[pymethods]
impl PyBeliefMatcher {
    #[new]
    fn new(dem: &PyDem, max_iter: usize, method: &str, scale: f64) -> PyResult<Self> {
        let d = dem.flat()?;
        let bm = BeliefMatching::from_dem(d, bp_method(method, scale)?, max_iter).map_err(err)?;
        Ok(PyBeliefMatcher { bm, num_detectors: d.num_detectors, num_observables: d.num_observables })
    }

    /// (observables as u64, weights as f64 (NaN where BP converged), one byte per shot 1
    /// where BP converged, the shots that failed).
    #[allow(clippy::type_complexity)]
    fn decode_batch<'py>(
        &self,
        py: Python<'py>,
        packed: &[u8],
        shots: usize,
        threads: usize,
    ) -> PyResult<(Bound<'py, PyBytes>, Bound<'py, PyBytes>, Bound<'py, PyBytes>, Vec<usize>)> {
        let nd = self.num_detectors;
        check_rows(packed.len(), nd.div_ceil(8), shots, nd, "detectors")?;
        let bm = &self.bm;
        let out = py.detach(|| belief_shots(bm, packed, nd, shots, threads));
        let failed = out.iter().enumerate().filter(|(_, x)| x.2 == 2).map(|(s, _)| s).collect();
        let conv: Vec<u8> = out.iter().map(|x| u8::from(x.2 == 1)).collect();
        Ok((
            PyBytes::new(py, &le_u64(out.iter().map(|x| x.0))),
            PyBytes::new(py, &le_f64(out.iter().map(|x| x.1))),
            PyBytes::new(py, &conv),
            failed,
        ))
    }
}

/// BP+OSD on an undecomposed model: each fault a column, its prior the model's.
#[pyclass(name = "DemBpOsd", module = "stabilizer_qec._core")]
pub struct PyDemBpOsd {
    dec: BpOsd,
    obs: Vec<u64>,
    #[pyo3(get)]
    num_detectors: usize,
    #[pyo3(get)]
    num_observables: usize,
}

#[pymethods]
impl PyDemBpOsd {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(dem: &PyDem, max_iter: usize, method: &str, scale: f64, osd: &str, order: usize) -> PyResult<Self> {
        let d = dem.flat()?;
        let columns: Vec<Vec<u32>> = d.mechanisms.iter().map(|m| m.detectors.clone()).collect();
        let priors: Vec<f64> = d.mechanisms.iter().map(|m| m.p).collect();
        let dec = BpOsd::new(d.num_detectors, columns, &priors, bp_method(method, scale)?, max_iter, osd_method(osd, order)?)
            .map_err(err)?;
        let obs = d.mechanisms.iter().map(|m| m.observables).collect();
        Ok(PyDemBpOsd { dec, obs, num_detectors: d.num_detectors, num_observables: d.num_observables })
    }

    /// (observables as u64 per shot, one byte per shot 1 where BP converged).
    fn decode_batch<'py>(&self, py: Python<'py>, packed: &[u8], shots: usize, threads: usize) -> PyResult<(Bound<'py, PyBytes>, Bound<'py, PyBytes>)> {
        let nd = self.num_detectors;
        check_rows(packed.len(), nd.div_ceil(8), shots, nd, "detectors")?;
        let (dec, obs) = (&self.dec, &self.obs);
        let out = py.detach(|| bposd_shots(dec, obs, packed, nd, shots, threads));
        let conv: Vec<u8> = out.iter().map(|x| x.1).collect();
        Ok((PyBytes::new(py, &le_u64(out.iter().map(|x| x.0))), PyBytes::new(py, &conv)))
    }
}

/// BP+LSD on an undecomposed model: each fault a column, its prior the model's.
#[pyclass(name = "DemBpLsd", module = "stabilizer_qec._core")]
pub struct PyDemBpLsd {
    dec: BpLsd,
    obs: Vec<u64>,
    #[pyo3(get)]
    num_detectors: usize,
    #[pyo3(get)]
    num_observables: usize,
}

#[pymethods]
impl PyDemBpLsd {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(dem: &PyDem, max_iter: usize, method: &str, scale: f64, lsd: &str, order: usize, bits_per_step: usize) -> PyResult<Self> {
        let d = dem.flat()?;
        let columns: Vec<Vec<u32>> = d.mechanisms.iter().map(|m| m.detectors.clone()).collect();
        let priors: Vec<f64> = d.mechanisms.iter().map(|m| m.p).collect();
        // As ldpc: no iteration limit given means one per fault.
        let max_iter = if max_iter == 0 { columns.len() } else { max_iter };
        let dec = BpLsd::new(d.num_detectors, columns, &priors, bp_method(method, scale)?, max_iter, osd_method(lsd, order)?, bits_per_step, false).map_err(err)?;
        let obs = d.mechanisms.iter().map(|m| m.observables).collect();
        Ok(PyDemBpLsd { dec, obs, num_detectors: d.num_detectors, num_observables: d.num_observables })
    }

    /// (observables as u64 per shot, one byte per shot: 1 where BP converged, 0 where LSD
    /// ran, 2 where no correction explains the shot).
    fn decode_batch<'py>(&self, py: Python<'py>, packed: &[u8], shots: usize, threads: usize) -> PyResult<(Bound<'py, PyBytes>, Bound<'py, PyBytes>)> {
        let nd = self.num_detectors;
        check_rows(packed.len(), nd.div_ceil(8), shots, nd, "detectors")?;
        let (dec, obs) = (&self.dec, &self.obs);
        let out = py.detach(|| syndrome_shots(dec, obs, packed, nd, shots, threads));
        let flags: Vec<u8> = out.iter().map(|x| x.1).collect();
        Ok((PyBytes::new(py, &le_u64(out.iter().map(|x| x.0))), PyBytes::new(py, &flags)))
    }
}

/// Window decoding of a model cut by time.
#[pyclass(name = "WindowMatcher", module = "stabilizer_qec._core")]
pub struct PyWindowMatcher {
    wd: PyWindows,
    correlated: bool,
    #[pyo3(get)]
    num_detectors: usize,
    #[pyo3(get)]
    num_observables: usize,
}

enum PyWindows {
    Whole(Box<WindowDecoder>),
    Streamed(Box<crate::batch::StreamedWindows>),
}

#[pymethods]
impl PyWindowMatcher {
    /// `template`: windows from a template of the model's loop (True), from the whole model
    /// (False), or the template for a model of more than a million faults (None).
    #[new]
    #[pyo3(signature = (dem, commit, buffer, mode, correlated, template=None))]
    fn new(dem: &PyDem, commit: usize, buffer: usize, mode: &str, correlated: bool, template: Option<bool>) -> PyResult<Self> {
        let mode = match mode {
            "sliding" => Mode::Sliding,
            "parallel" => Mode::Parallel,
            other => return Err(err(format!("mode '{other}' is neither sliding nor parallel"))),
        };
        let (nd, no) = (dem.dem.num_detectors(), dem.dem.num_observables());
        let streamed = || crate::batch::StreamedWindows::new(dem.dem.program(), commit, buffer, mode).map(Box::new);
        let whole = || -> PyResult<PyWindows> {
            Ok(PyWindows::Whole(Box::new(WindowDecoder::new(Model::new(dem.flat()?).map_err(err)?, commit, buffer, mode).map_err(err)?)))
        };
        let wd = match template {
            Some(true) => PyWindows::Streamed(streamed().map_err(err)?),
            None if dem.dem.num_errors() > 1_000_000 => match streamed() {
                Ok(s) => PyWindows::Streamed(s),
                Err(_) => whole()?,
            },
            _ => whole()?,
        };
        Ok(PyWindowMatcher { wd, correlated, num_detectors: nd, num_observables: no })
    }

    fn windows(&self) -> Vec<WindowInfo> {
        match &self.wd {
            PyWindows::Whole(wd) => window_info(wd),
            PyWindows::Streamed(sw) => sw.windows(),
        }
    }

    /// Whether the windows come from a template of the model's loop.
    #[getter]
    fn streamed(&self) -> bool {
        matches!(self.wd, PyWindows::Streamed(_))
    }

    /// (observables as u64 per shot, the defects left unexplained, each window's decode
    /// seconds per shot as f64 when `timings`, the shots a window refused).
    #[allow(clippy::type_complexity)]
    fn decode_batch<'py>(
        &self,
        py: Python<'py>,
        packed: &[u8],
        shots: usize,
        threads: usize,
        timings: bool,
    ) -> PyResult<(Bound<'py, PyBytes>, usize, Bound<'py, PyBytes>, Vec<usize>)> {
        let nd = self.num_detectors;
        check_rows(packed.len(), nd.div_ceil(8), shots, nd, "detectors")?;
        let (wd, correlated) = (&self.wd, self.correlated);
        let out = py.detach(|| match wd {
            PyWindows::Whole(wd) => window_shots(wd, packed, nd, shots, correlated, threads, timings),
            PyWindows::Streamed(sw) => crate::batch::streamed_window_shots(sw, packed, nd, shots, correlated, threads, timings),
        });
        let unexplained = out.iter().map(|x| x.1).sum();
        let times = le_f64(out.iter().flat_map(|x| x.2.iter().copied()));
        let failed = out.iter().enumerate().filter(|(_, x)| x.3).map(|(s, _)| s).collect();
        Ok((PyBytes::new(py, &le_u64(out.iter().map(|x| x.0))), unexplained, PyBytes::new(py, &times), failed))
    }
}

/* -- Decoders on a check matrix ------------------------------------------------------------ */

/// Belief propagation on a check matrix given by its columns, flooding schedule, as `ldpc`.
#[pyclass(name = "Bp", module = "stabilizer_qec._core")]
pub struct PyBp {
    bp: crate::bp::Bp,
    method: crate::bp::Method,
    max_iter: usize,
    num_checks: usize,
}

#[pymethods]
impl PyBp {
    #[new]
    fn new(num_checks: usize, columns: Vec<Vec<u32>>, priors: Vec<f64>, max_iter: usize, method: &str, scale: f64) -> PyResult<Self> {
        let bp = crate::bp::Bp::new(num_checks, &columns, &priors).map_err(err)?;
        Ok(PyBp { bp, method: bp_method(method, scale)?, max_iter, num_checks })
    }

    /// (hard decision, posterior log-likelihood ratios, converged, iterations).
    fn decode(&self, syndrome: Vec<u8>) -> PyResult<(Vec<u32>, Vec<f64>, bool, usize)> {
        if syndrome.len() != self.num_checks {
            return Err(err(format!("{} syndrome bits for {} checks", syndrome.len(), self.num_checks)));
        }
        let mut w = self.bp.work();
        let out = self.bp.decode(&syndrome, self.method, self.max_iter, &mut w);
        Ok((bits(&w.hard), w.llr, out.converged, out.iterations))
    }
}

/// BP+OSD on a check matrix given by its columns.
#[pyclass(name = "BpOsd", module = "stabilizer_qec._core")]
pub struct PyBpOsd {
    dec: BpOsd,
    num_checks: usize,
}

#[pymethods]
impl PyBpOsd {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(num_checks: usize, columns: Vec<Vec<u32>>, priors: Vec<f64>, max_iter: usize, method: &str, scale: f64, osd: &str, order: usize) -> PyResult<Self> {
        let dec = BpOsd::new(num_checks, columns, &priors, bp_method(method, scale)?, max_iter, osd_method(osd, order)?).map_err(err)?;
        Ok(PyBpOsd { dec, num_checks })
    }

    /// (correction, BP converged, iterations).
    fn decode(&self, syndrome: Vec<u8>) -> PyResult<(Vec<u32>, bool, usize)> {
        if syndrome.len() != self.num_checks {
            return Err(err(format!("{} syndrome bits for {} checks", syndrome.len(), self.num_checks)));
        }
        let mut w = self.dec.work();
        let out = self.dec.decode(&syndrome, &mut w);
        Ok((bits(&w.correction), out.converged, out.iterations))
    }
}

/// BP+LSD on a check matrix given by its columns.
#[pyclass(name = "BpLsd", module = "stabilizer_qec._core")]
pub struct PyBpLsd {
    dec: BpLsd,
    num_checks: usize,
}

#[pymethods]
impl PyBpLsd {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(num_checks: usize, columns: Vec<Vec<u32>>, priors: Vec<f64>, max_iter: usize, method: &str, scale: f64, lsd: &str, order: usize, bits_per_step: usize, always_run: bool) -> PyResult<Self> {
        let dec = BpLsd::new(num_checks, columns, &priors, bp_method(method, scale)?, max_iter, osd_method(lsd, order)?, bits_per_step, always_run).map_err(err)?;
        Ok(PyBpLsd { dec, num_checks })
    }

    /// (correction, BP converged, iterations, solved, ties, each final cluster's bits in
    /// set order by cluster id when `record`).
    #[pyo3(signature = (syndrome, record=false))]
    #[allow(clippy::type_complexity)]
    fn decode(&self, syndrome: Vec<u8>, record: bool) -> PyResult<(Vec<u32>, bool, usize, bool, u32, Vec<(u32, Vec<u32>)>)> {
        if syndrome.len() != self.num_checks {
            return Err(err(format!("{} syndrome bits for {} checks", syndrome.len(), self.num_checks)));
        }
        let mut w = self.dec.work();
        w.record = record;
        let out = self.dec.decode(&syndrome, &mut w);
        Ok((bits(&w.correction), out.converged, out.iterations, out.solved, w.ties, std::mem::take(&mut w.cluster_bits)))
    }
}

#[allow(clippy::too_many_arguments)]
fn relay_config(
    pre_iter: usize,
    legs: usize,
    leg_iter: usize,
    solutions: Option<usize>,
    gamma0: Option<f64>,
    gamma_range: (f64, f64),
    gammas: Option<Vec<Vec<f64>>>,
    alpha: Option<f64>,
    alpha_scaling: f64,
    seed: u64,
) -> RelayConfig {
    RelayConfig { pre_iter, legs, leg_iter, solutions, gamma0, gamma_range, gammas, alpha, alpha_scaling, seed }
}

/// Relay-BP on a check matrix given by its columns. Its random memory strengths continue from
/// decode to decode, as IBM's decoder's do.
#[pyclass(name = "Relay", module = "stabilizer_qec._core")]
pub struct PyRelay {
    dec: Relay,
    work: std::sync::Mutex<crate::relay::RelayWork>,
}

#[pymethods]
impl PyRelay {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(
        num_checks: usize,
        columns: Vec<Vec<u32>>,
        priors: Vec<f64>,
        pre_iter: usize,
        legs: usize,
        leg_iter: usize,
        solutions: Option<usize>,
        gamma0: Option<f64>,
        gamma_range: (f64, f64),
        gammas: Option<Vec<Vec<f64>>>,
        alpha: Option<f64>,
        alpha_scaling: f64,
        seed: u64,
    ) -> PyResult<Self> {
        let config = relay_config(pre_iter, legs, leg_iter, solutions, gamma0, gamma_range, gammas, alpha, alpha_scaling, seed);
        let dec = Relay::new(num_checks, &columns, &priors, config).map_err(err)?;
        let work = std::sync::Mutex::new(dec.work());
        Ok(PyRelay { dec, work })
    }

    /// (correction, converged, iterations, legs, weight).
    fn decode(&self, syndrome: Vec<u8>) -> PyResult<(Vec<u32>, bool, usize, usize, f64)> {
        if syndrome.len() != self.dec.num_checks {
            return Err(err(format!("{} syndrome bits for {} checks", syndrome.len(), self.dec.num_checks)));
        }
        let mut w = self.work.lock().unwrap();
        let out = self.dec.decode(&syndrome, &mut w);
        Ok((bits(&w.correction), out.converged, out.iterations, out.legs, out.weight))
    }
}

/// Relay-BP on an undecomposed model: each fault a column, its prior the model's. Shot k of a
/// batch draws its memory strengths from a generator seeded by the seed and k.
#[pyclass(name = "DemRelay", module = "stabilizer_qec._core")]
pub struct PyDemRelay {
    dec: Relay,
    obs: Vec<u64>,
    #[pyo3(get)]
    num_detectors: usize,
    #[pyo3(get)]
    num_observables: usize,
}

#[pymethods]
impl PyDemRelay {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(
        dem: &PyDem,
        pre_iter: usize,
        legs: usize,
        leg_iter: usize,
        solutions: Option<usize>,
        gamma0: Option<f64>,
        gamma_range: (f64, f64),
        alpha: Option<f64>,
        seed: u64,
    ) -> PyResult<Self> {
        let d = dem.flat()?;
        let columns: Vec<Vec<u32>> = d.mechanisms.iter().map(|m| m.detectors.clone()).collect();
        let priors: Vec<f64> = d.mechanisms.iter().map(|m| m.p).collect();
        let config = relay_config(pre_iter, legs, leg_iter, solutions, gamma0, gamma_range, None, alpha, 1.0, seed);
        let dec = Relay::new(d.num_detectors, &columns, &priors, config).map_err(err)?;
        let obs = d.mechanisms.iter().map(|m| m.observables).collect();
        Ok(PyDemRelay { dec, obs, num_detectors: d.num_detectors, num_observables: d.num_observables })
    }

    /// (observables as u64 per shot, one byte per shot: 1 where a leg converged).
    fn decode_batch<'py>(&self, py: Python<'py>, packed: &[u8], shots: usize, threads: usize) -> PyResult<(Bound<'py, PyBytes>, Bound<'py, PyBytes>)> {
        let nd = self.num_detectors;
        check_rows(packed.len(), nd.div_ceil(8), shots, nd, "detectors")?;
        let (dec, obs) = (&self.dec, &self.obs);
        let out = py.detach(|| crate::batch::relay_shots(dec, obs, packed, nd, shots, threads));
        let flags: Vec<u8> = out.iter().map(|x| x.1).collect();
        Ok((PyBytes::new(py, &le_u64(out.iter().map(|x| x.0))), PyBytes::new(py, &flags)))
    }
}

/// Colour-code matching (Chromobius's construction) on an annotated model.
#[pyclass(name = "DemColorMatcher", module = "stabilizer_qec._core")]
pub struct PyDemColorMatcher {
    dec: crate::color::ColorDecoder,
    #[pyo3(get)]
    num_detectors: usize,
    #[pyo3(get)]
    num_observables: usize,
}

#[pymethods]
impl PyDemColorMatcher {
    #[new]
    fn new(dem: &PyDem, ignore_decomposition_failures: bool) -> PyResult<Self> {
        let d = dem.flat()?;
        let dec = crate::color::ColorDecoder::from_dem(d, ignore_decomposition_failures).map_err(err)?;
        Ok(PyDemColorMatcher { dec, num_detectors: d.num_detectors, num_observables: d.num_observables })
    }

    /// The Möbius model, as Stim's text.
    fn mobius_model(&self) -> String {
        self.dec.mobius_text().to_string()
    }

    /// (observables as u64 per shot, weights as f64, one byte per shot 1 where the lifting
    /// used an ambiguous table entry, the shots that failed).
    #[allow(clippy::type_complexity)]
    fn decode_batch<'py>(&self, py: Python<'py>, packed: &[u8], shots: usize, threads: usize) -> PyResult<(Bound<'py, PyBytes>, Bound<'py, PyBytes>, Bound<'py, PyBytes>, Vec<usize>)> {
        let nd = self.num_detectors;
        check_rows(packed.len(), nd.div_ceil(8), shots, nd, "detectors")?;
        let dec = &self.dec;
        let out = py.detach(|| crate::batch::color_shots(dec, packed, nd, shots, threads));
        let failed = out.iter().enumerate().filter(|(_, x)| x.2 == 2).map(|(s, _)| s).collect();
        let tied: Vec<u8> = out.iter().map(|x| u8::from(x.2 == 1)).collect();
        Ok((PyBytes::new(py, &le_u64(out.iter().map(|x| x.0))), PyBytes::new(py, &le_f64(out.iter().map(|x| x.1))), PyBytes::new(py, &tied), failed))
    }
}

/// The search decoder (Tesseract's A*) on a model.
#[pyclass(name = "DemSearch", module = "stabilizer_qec._core")]
pub struct PyDemSearch {
    dec: crate::search::Search,
    #[pyo3(get)]
    num_detectors: usize,
    #[pyo3(get)]
    num_observables: usize,
    #[pyo3(get)]
    orders: Vec<Vec<u32>>,
}

#[pymethods]
impl PyDemSearch {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(
        dem: &PyDem,
        beam: Option<usize>,
        beam_climbing: bool,
        no_revisit: bool,
        queue_limit: Option<usize>,
        orders: Option<Vec<Vec<u32>>>,
        num_orders: usize,
        method: &str,
        seed: u64,
        penalty: f64,
        merge: bool,
        flavour: &str,
    ) -> PyResult<Self> {
        use crate::search::{Flavour, OrderMethod};
        let d = dem.flat()?;
        let flavour = match flavour {
            "native" => Flavour::native(),
            "libc++" => Flavour::LibCxx,
            "libstdc++" => Flavour::LibStdCxx,
            other => return Err(err(format!("unknown flavour '{other}'"))),
        };
        let orders = match orders {
            Some(o) => o,
            None => {
                let method = match method {
                    "index" => OrderMethod::Index,
                    "bfs" => OrderMethod::Bfs,
                    "coordinate" => OrderMethod::Coordinate,
                    other => return Err(err(format!("det_order_method '{other}' is not index, bfs or coordinate"))),
                };
                if num_orders == 0 {
                    return Err(err("num_det_orders must be at least 1".into()));
                }
                crate::search::generated_orders(d, num_orders, method, seed, flavour)
            }
        };
        let config = crate::search::SearchConfig { beam, beam_climbing, no_revisit, queue_limit, orders: orders.clone(), detector_penalty: penalty, merge_errors: merge, flavour };
        let dec = crate::search::Search::new(d, config).map_err(err)?;
        Ok(PyDemSearch { dec, num_detectors: d.num_detectors, num_observables: d.num_observables, orders })
    }

    /// (observables as u64 per shot, costs as f64, one byte per shot 1 where the search gave
    /// up).
    #[allow(clippy::type_complexity)]
    fn decode_batch<'py>(&self, py: Python<'py>, packed: &[u8], shots: usize, threads: usize) -> PyResult<(Bound<'py, PyBytes>, Bound<'py, PyBytes>, Bound<'py, PyBytes>)> {
        let nd = self.num_detectors;
        check_rows(packed.len(), nd.div_ceil(8), shots, nd, "detectors")?;
        let dec = &self.dec;
        let out = py.detach(|| crate::batch::search_shots(dec, packed, nd, shots, threads));
        let gave_up: Vec<u8> = out.iter().map(|x| u8::from(x.2)).collect();
        Ok((PyBytes::new(py, &le_u64(out.iter().map(|x| x.0))), PyBytes::new(py, &le_f64(out.iter().map(|x| x.1))), PyBytes::new(py, &gave_up)))
    }

    /// The model faults one shot's search found (by index in the flattened model).
    fn decode_to_faults(&self, defects: Vec<u32>) -> (Vec<usize>, bool) {
        let found = self.dec.decode(&defects);
        (found.faults, found.low_confidence)
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyCircuit>()?;
    m.add_class::<PyDem>()?;
    m.add_class::<PySampler>()?;
    m.add_class::<PyDemSampler>()?;
    m.add_class::<PyMeasurementSampler>()?;
    m.add_class::<PyM2d>()?;
    m.add_class::<PyMatcher>()?;
    m.add_class::<PyUnionFinder>()?;
    m.add_class::<PyBeliefMatcher>()?;
    m.add_class::<PyDemBpOsd>()?;
    m.add_class::<PyWindowMatcher>()?;
    m.add_class::<PyBp>()?;
    m.add_class::<PyBpOsd>()?;
    m.add_class::<PyDemBpLsd>()?;
    m.add_class::<PyBpLsd>()?;
    m.add_class::<PyRelay>()?;
    m.add_class::<PyDemRelay>()?;
    m.add_class::<PyDemColorMatcher>()?;
    m.add_class::<PyDemSearch>()?;
    Ok(())
}
